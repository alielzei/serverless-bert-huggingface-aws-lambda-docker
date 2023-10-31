from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

def gen_input_builder_fun(self, model, dataset, is_train):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.contrib.playground.resnetdemo.gfs_IN1k.gen_input_builder_fun', 'gen_input_builder_fun(self, model, dataset, is_train)', {'self': self, 'model': model, 'dataset': dataset, 'is_train': is_train}, 1)

def get_input_dataset(opts):
    return []

def get_model_input_fun(self):
    pass

