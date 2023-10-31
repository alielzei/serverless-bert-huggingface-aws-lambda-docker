from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import workspace, core
from caffe2.proto import caffe2_pb2

def gen_param_update_builder_fun(self, model, dataset, is_train):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.contrib.playground.resnetdemo.explicit_resnet_param_update.gen_param_update_builder_fun', 'gen_param_update_builder_fun(self, model, dataset, is_train)', {'core': core, 'caffe2_pb2': caffe2_pb2, 'workspace': workspace, 'self': self, 'model': model, 'dataset': dataset, 'is_train': is_train}, 1)

