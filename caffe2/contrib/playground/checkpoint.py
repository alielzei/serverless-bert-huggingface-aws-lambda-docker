from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
import cPickle as pickle
from collections import OrderedDict
from caffe2.proto import caffe2_pb2
from caffe2.python import workspace, core, scope
import logging
logging.basicConfig()
log = logging.getLogger('AnyExpOnTerm')
log.setLevel(logging.DEBUG)

def initialize_params_from_file(model, weights_file, num_xpus, opts, broadcast_computed_param=False, reset_epoch=False):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.contrib.playground.checkpoint.initialize_params_from_file', 'initialize_params_from_file(model, weights_file, num_xpus, opts, broadcast_computed_param=False, reset_epoch=False)', {'initialize_master_xpu_model_params': initialize_master_xpu_model_params, 'broadcast_parameters': broadcast_parameters, 'model': model, 'weights_file': weights_file, 'num_xpus': num_xpus, 'opts': opts, 'broadcast_computed_param': broadcast_computed_param, 'reset_epoch': reset_epoch}, 3)

def initialize_master_xpu_model_params(model, weights_file, opts, reset_epoch):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.contrib.playground.checkpoint.initialize_master_xpu_model_params', 'initialize_master_xpu_model_params(model, weights_file, opts, reset_epoch)', {'log': log, 'pickle': pickle, 'workspace': workspace, 'OrderedDict': OrderedDict, 'unscope_name': unscope_name, 'caffe2_pb2': caffe2_pb2, 'core': core, 'scoped_name': scoped_name, 'np': np, 'model': model, 'weights_file': weights_file, 'opts': opts, 'reset_epoch': reset_epoch}, 3)

def broadcast_parameters(opts, model, num_xpus, broadcast_computed_param=False):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.contrib.playground.checkpoint.broadcast_parameters', 'broadcast_parameters(opts, model, num_xpus, broadcast_computed_param=False)', {'log': log, 'caffe2_pb2': caffe2_pb2, 'workspace': workspace, 'core': core, 'opts': opts, 'model': model, 'num_xpus': num_xpus, 'broadcast_computed_param': broadcast_computed_param}, 1)

def save_model_params(is_checkpoint, model, checkpoint_path, epoch, opts, best_metric):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.contrib.playground.checkpoint.save_model_params', 'save_model_params(is_checkpoint, model, checkpoint_path, epoch, opts, best_metric)', {'save_model_params_blob': save_model_params_blob, 'log': log, 'is_checkpoint': is_checkpoint, 'model': model, 'checkpoint_path': checkpoint_path, 'epoch': epoch, 'opts': opts, 'best_metric': best_metric}, 1)

def save_model_params_blob(model, params_file, epoch, opts, best_metric):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.contrib.playground.checkpoint.save_model_params_blob', 'save_model_params_blob(model, params_file, epoch, opts, best_metric)', {'log': log, 'workspace': workspace, 'unscope_name': unscope_name, 'pickle': pickle, 'IOError': IOError, 'model': model, 'params_file': params_file, 'epoch': epoch, 'opts': opts, 'best_metric': best_metric}, 0)

def unscope_name(blob_name):
    return blob_name[blob_name.rfind(scope._NAMESCOPE_SEPARATOR) + 1:]

def scoped_name(blob_name):
    return scope.CurrentNameScope() + blob_name

