import argparse
import flask
import glob
import numpy as np
import nvd3
import os
import sys
import tornado.httpserver
import tornado.wsgi
__folder__ = os.path.abspath(os.path.dirname(__file__))
app = flask.Flask(__name__, template_folder=os.path.join(__folder__, 'templates'), static_folder=os.path.join(__folder__, 'static'))
args = None

def jsonify_nvd3(chart):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.mint.app.jsonify_nvd3', 'jsonify_nvd3(chart)', {'flask': flask, 'chart': chart}, 1)

def visualize_summary(filename):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.mint.app.visualize_summary', 'visualize_summary(filename)', {'np': np, 'os': os, 'nvd3': nvd3, 'args': args, 'jsonify_nvd3': jsonify_nvd3, 'filename': filename}, 1)

def visualize_print_log(filename):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.mint.app.visualize_print_log', 'visualize_print_log(filename)', {'np': np, 'os': os, 'nvd3': nvd3, 'args': args, 'jsonify_nvd3': jsonify_nvd3, 'filename': filename}, 1)

def visualize_file(filename):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.mint.app.visualize_file', 'visualize_file(filename)', {'os': os, 'args': args, 'visualize_summary': visualize_summary, 'visualize_print_log': visualize_print_log, 'flask': flask, 'filename': filename}, 1)

@app.route('/')
def index():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.mint.app.index', 'index()', {'glob': glob, 'os': os, 'args': args, 'flask': flask, 'app': app}, 1)

@app.route('/visualization/<string:name>')
def visualization(name):
    ret = visualize_file(name)
    return ret

def main(argv):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.mint.app.main', 'main(argv)', {'argparse': argparse, 'tornado': tornado, 'app': app, 'argv': argv}, 0)
if __name__ == '__main__':
    main(sys.argv[1:])

