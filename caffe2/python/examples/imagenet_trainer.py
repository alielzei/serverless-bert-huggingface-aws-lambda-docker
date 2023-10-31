from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import argparse
import logging
import numpy as np
import time
import os
from caffe2.python import core, workspace, experiment_util, data_parallel_model
from caffe2.python import dyndep, optimizer
from caffe2.python import timeout_guard, model_helper, brew
from caffe2.proto import caffe2_pb2
import caffe2.python.models.resnet as resnet
import caffe2.python.models.shufflenet as shufflenet
from caffe2.python.modeling.initializers import Initializer, PseudoFP16Initializer
import caffe2.python.predictor.predictor_exporter as pred_exp
import caffe2.python.predictor.predictor_py_utils as pred_utils
from caffe2.python.predictor_constants import predictor_constants as predictor_constants
'\nParallelized multi-GPU distributed trainer for Resne(X)t & Shufflenet.\nCan be used to train on imagenet data, for example.\nThe default parameters can train a standard Resnet-50 (1x64d), and parameters\ncan be provided to train ResNe(X)t models (e.g., ResNeXt-101 32x4d).\n\nTo run the trainer in single-machine multi-gpu mode by setting num_shards = 1.\n\nTo run the trainer in multi-machine multi-gpu mode with M machines,\nrun the same program on all machines, specifying num_shards = M, and\nshard_id = a unique integer in the set [0, M-1].\n\nFor rendezvous (the trainer processes have to know about each other),\nyou can either use a directory path that is visible to all processes\n(e.g. NFS directory), or use a Redis instance. Use the former by\npassing the `file_store_path` argument. Use the latter by passing the\n`redis_host` and `redis_port` arguments.\n'
logging.basicConfig()
log = logging.getLogger('Imagenet_trainer')
log.setLevel(logging.DEBUG)
dyndep.InitOpsLibrary('@/caffe2/caffe2/distributed:file_store_handler_ops')
dyndep.InitOpsLibrary('@/caffe2/caffe2/distributed:redis_store_handler_ops')

def AddImageInput(model, reader, batch_size, img_size, dtype, is_test, mean_per_channel=None, std_per_channel=None):
    """
    The image input operator loads image and label data from the reader and
    applies transformations to the images (random cropping, mirroring, ...).
    """
    (data, label) = brew.image_input(model, reader, ['data', 'label'], batch_size=batch_size, output_type=dtype, use_gpu_transform=(True if core.IsGPUDeviceType(model._device_type) else False), use_caffe_datum=True, mean_per_channel=mean_per_channel, std_per_channel=std_per_channel, mean=128.0, std=128.0, scale=256, crop=img_size, mirror=1, is_test=is_test)
    data = model.StopGradient(data, data)

def AddNullInput(model, reader, batch_size, img_size, dtype):
    """
    The null input function uses a gaussian fill operator to emulate real image
    input. A label blob is hardcoded to a single value. This is useful if you
    want to test compute throughput or don't have a dataset available.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.examples.imagenet_trainer.AddNullInput', 'AddNullInput(model, reader, batch_size, img_size, dtype)', {'core': core, 'model': model, 'reader': reader, 'batch_size': batch_size, 'img_size': img_size, 'dtype': dtype}, 0)

def SaveModel(args, train_model, epoch, use_ideep):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.examples.imagenet_trainer.SaveModel', 'SaveModel(args, train_model, epoch, use_ideep)', {'pred_exp': pred_exp, 'data_parallel_model': data_parallel_model, 'args': args, 'train_model': train_model, 'epoch': epoch, 'use_ideep': use_ideep}, 0)

def LoadModel(path, model, use_ideep):
    """
    Load pretrained model from file
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.examples.imagenet_trainer.LoadModel', 'LoadModel(path, model, use_ideep)', {'log': log, 'pred_exp': pred_exp, 'core': core, 'pred_utils': pred_utils, 'predictor_constants': predictor_constants, 'workspace': workspace, 'caffe2_pb2': caffe2_pb2, 'path': path, 'model': model, 'use_ideep': use_ideep}, 0)

def RunEpoch(args, epoch, train_model, test_model, total_batch_size, num_shards, expname, explog):
    """
    Run one epoch of the trainer.
    TODO: add checkpointing here.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.examples.imagenet_trainer.RunEpoch', 'RunEpoch(args, epoch, train_model, test_model, total_batch_size, num_shards, expname, explog)', {'log': log, 'timeout_guard': timeout_guard, 'time': time, 'workspace': workspace, 'data_parallel_model': data_parallel_model, 'np': np, 'args': args, 'epoch': epoch, 'train_model': train_model, 'test_model': test_model, 'total_batch_size': total_batch_size, 'num_shards': num_shards, 'expname': expname, 'explog': explog}, 1)

def Train(args):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.examples.imagenet_trainer.Train', 'Train(args)', {'log': log, 'model_helper': model_helper, 'os': os, 'workspace': workspace, 'core': core, 'PseudoFP16Initializer': PseudoFP16Initializer, 'Initializer': Initializer, 'brew': brew, 'resnet': resnet, 'shufflenet': shufflenet, 'optimizer': optimizer, 'AddNullInput': AddNullInput, 'AddImageInput': AddImageInput, 'data_parallel_model': data_parallel_model, 'LoadModel': LoadModel, 'experiment_util': experiment_util, 'RunEpoch': RunEpoch, 'SaveModel': SaveModel, 'args': args}, 1)

def main():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.examples.imagenet_trainer.main', 'main()', {'argparse': argparse, 'Train': Train}, 0)
if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=2'])
    main()

