from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import argparse
import os
import sys
import signal
import re
import json
from caffe2.proto import caffe2_pb2
try:
    import urllib.error as urlliberror
    import urllib.request as urllib
    HTTPError = urlliberror.HTTPError
    URLError = urlliberror.URLError
except ImportError:
    import urllib2 as urllib
    HTTPError = urllib.HTTPError
    URLError = urllib.URLError
DOWNLOAD_BASE_URL = 'https://s3.amazonaws.com/download.caffe2.ai/models/'
DOWNLOAD_COLUMNS = 70

def signalHandler(signal, frame):
    print('Killing download...')
    exit(0)
signal.signal(signal.SIGINT, signalHandler)

def deleteDirectory(top_dir):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.models.download.deleteDirectory', 'deleteDirectory(top_dir)', {'os': os, 'top_dir': top_dir}, 0)

def progressBar(percentage):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.models.download.progressBar', 'progressBar(percentage)', {'DOWNLOAD_COLUMNS': DOWNLOAD_COLUMNS, 'sys': sys, 'percentage': percentage}, 0)

def downloadFromURLToFile(url, filename, show_progress=True):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.models.download.downloadFromURLToFile', 'downloadFromURLToFile(url, filename, show_progress=True)', {'urllib': urllib, 'progressBar': progressBar, 'HTTPError': HTTPError, 'URLError': URLError, 'url': url, 'filename': filename, 'show_progress': show_progress}, 0)

def getURLFromName(name, filename):
    return '{base_url}{name}/{filename}'.format(base_url=DOWNLOAD_BASE_URL, name=name, filename=filename)

def downloadModel(model, args):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.models.download.downloadModel', 'downloadModel(model, args)', {'os': os, '__file__': __file__, 'raw_input': raw_input, 'deleteDirectory': deleteDirectory, 'downloadFromURLToFile': downloadFromURLToFile, 'getURLFromName': getURLFromName, 'model': model, 'args': args}, 0)

def validModelName(name):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.models.download.validModelName', 'validModelName(name)', {'re': re, 'name': name}, 1)


class ModelDownloader:
    
    def __init__(self, model_env_name='CAFFE2_MODELS'):
        self.model_env_name = model_env_name
    
    def _model_dir(self, model):
        caffe2_home = os.path.expanduser(os.getenv('CAFFE2_HOME', '~/.caffe2'))
        models_dir = os.getenv(self.model_env_name, os.path.join(caffe2_home, 'models'))
        return os.path.join(models_dir, model)
    
    def _download(self, model):
        model_dir = self._model_dir(model)
        assert not os.path.exists(model_dir)
        os.makedirs(model_dir)
        for f in ['predict_net.pb', 'init_net.pb', 'value_info.json']:
            url = getURLFromName(model, f)
            dest = os.path.join(model_dir, f)
            try:
                downloadFromURLToFile(url, dest, show_progress=False)
            except TypeError:
                downloadFromURLToFile(url, dest)
            except Exception:
                deleteDirectory(model_dir)
                raise
    
    def get_c2_model_dbg(self, model_name):
        debug_str = 'get_c2_model debug:\n'
        model_dir = self._model_dir(model_name)
        if not os.path.exists(model_dir):
            self._download(model_name)
        c2_predict_pb = os.path.join(model_dir, 'predict_net.pb')
        debug_str += 'c2_predict_pb path: ' + c2_predict_pb + '\n'
        c2_predict_net = caffe2_pb2.NetDef()
        with open(c2_predict_pb, 'rb') as f:
            len_read = c2_predict_net.ParseFromString(f.read())
            debug_str += 'c2_predict_pb ParseFromString = ' + str(len_read) + '\n'
        c2_predict_net.name = model_name
        c2_init_pb = os.path.join(model_dir, 'init_net.pb')
        debug_str += 'c2_init_pb path: ' + c2_init_pb + '\n'
        c2_init_net = caffe2_pb2.NetDef()
        with open(c2_init_pb, 'rb') as f:
            len_read = c2_init_net.ParseFromString(f.read())
            debug_str += 'c2_init_pb ParseFromString = ' + str(len_read) + '\n'
        c2_init_net.name = model_name + '_init'
        with open(os.path.join(model_dir, 'value_info.json')) as f:
            value_info = json.load(f)
        return (c2_init_net, c2_predict_net, value_info, debug_str)
    
    def get_c2_model(self, model_name):
        (init_net, predict_net, value_info, _) = self.get_c2_model_dbg(model_name)
        return (init_net, predict_net, value_info)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download or install pretrained models.')
    parser.add_argument('model', nargs='+', help='Model to download/install.')
    parser.add_argument('-i', '--install', action='store_true', help='Install the model.')
    parser.add_argument('-f', '--force', action='store_true', help='Force a download/installation.')
    args = parser.parse_args()
    for model in args.model:
        if validModelName(model):
            downloadModel(model, args)
        else:
            print("'{}' is not a valid model name.".format(model))

