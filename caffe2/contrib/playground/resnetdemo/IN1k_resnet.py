from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
from caffe2.python import workspace, cnn, core
from caffe2.python import timeout_guard
from caffe2.proto import caffe2_pb2

def init_model(self):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.contrib.playground.resnetdemo.IN1k_resnet.init_model', 'init_model(self)', {'cnn': cnn, 'self': self}, 0)

def fun_per_epoch_b4RunNet(self, epoch):
    pass

def fun_per_iter_b4RunNet(self, epoch, epoch_iter):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.contrib.playground.resnetdemo.IN1k_resnet.fun_per_iter_b4RunNet', 'fun_per_iter_b4RunNet(self, epoch, epoch_iter)', {'caffe2_pb2': caffe2_pb2, 'core': core, 'workspace': workspace, 'np': np, 'self': self, 'epoch': epoch, 'epoch_iter': epoch_iter}, 0)

def run_training_net(self):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.contrib.playground.resnetdemo.IN1k_resnet.run_training_net', 'run_training_net(self)', {'timeout_guard': timeout_guard, 'workspace': workspace, 'self': self}, 0)

