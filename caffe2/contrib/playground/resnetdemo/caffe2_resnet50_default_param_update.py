from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

def gen_param_update_builder_fun(self, model, dataset, is_train):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.contrib.playground.resnetdemo.caffe2_resnet50_default_param_update.gen_param_update_builder_fun', 'gen_param_update_builder_fun(self, model, dataset, is_train)', {'self': self, 'model': model, 'dataset': dataset, 'is_train': is_train}, 1)

