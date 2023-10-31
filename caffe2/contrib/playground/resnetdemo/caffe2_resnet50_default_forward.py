from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import caffe2.python.models.resnet as resnet

def gen_forward_pass_builder_fun(self, model, dataset, is_train):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.contrib.playground.resnetdemo.caffe2_resnet50_default_forward.gen_forward_pass_builder_fun', 'gen_forward_pass_builder_fun(self, model, dataset, is_train)', {'resnet': resnet, 'self': self, 'model': model, 'dataset': dataset, 'is_train': is_train}, 1)

