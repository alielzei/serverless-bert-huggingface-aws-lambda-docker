from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import logging
logging.basicConfig()
log = logging.getLogger('AnyExp')
log.setLevel(logging.DEBUG)
BLOCK_CONFIG = {18: (2, 2, 2, 2), 34: (3, 4, 6, 3), 50: (3, 4, 6, 3), 101: (3, 4, 23, 3), 152: (3, 8, 36, 3), 200: (3, 32, 36, 3), 264: (3, 64, 36, 3), 284: (3, 32, 64, 3)}

def gen_forward_pass_builder_fun(self, model, dataset, is_train):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.contrib.playground.resnetdemo.explicit_resnet_forward.gen_forward_pass_builder_fun', 'gen_forward_pass_builder_fun(self, model, dataset, is_train)', {'resnet_imagenet_create_model': resnet_imagenet_create_model, 'self': self, 'model': model, 'dataset': dataset, 'is_train': is_train}, 1)

def resnet_imagenet_create_model(model, data, labels, split, opts, dataset):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.contrib.playground.resnetdemo.explicit_resnet_forward.resnet_imagenet_create_model', 'resnet_imagenet_create_model(model, data, labels, split, opts, dataset)', {'ResNetModelHelper': ResNetModelHelper, 'log': log, 'BLOCK_CONFIG': BLOCK_CONFIG, 'model': model, 'data': data, 'labels': labels, 'split': split, 'opts': opts, 'dataset': dataset}, 3)


class ResNetModelHelper:
    
    def __init__(self, model, split, opts):
        self.model = model
        self.split = split
        self.opts = opts
        self.engine = opts['model_param']['engine']
    
    def add_shortcut(self, blob_in, dim_in, dim_out, stride, prefix):
        if dim_in == dim_out:
            return blob_in
        conv_blob = self.model.Conv(blob_in, prefix, dim_in, dim_out, kernel=1, stride=stride, weight_init=('MSRAFill', {}), bias_init=('ConstantFill', {'value': 0.0}), no_bias=1, engine=self.engine)
        test_mode = False
        if self.split in ['test', 'val']:
            test_mode = True
        bn_blob = self.model.SpatialBN(conv_blob, prefix + '_bn', dim_out, epsilon=self.opts['model_param']['bn_epsilon'], momentum=self.opts['model_param']['bn_momentum'], is_test=test_mode)
        return bn_blob
    
    def conv_bn(self, blob_in, dim_in, dim_out, kernel, stride, prefix, group=1, pad=1):
        conv_blob = self.model.Conv(blob_in, prefix, dim_in, dim_out, kernel, stride=stride, pad=pad, group=group, weight_init=('MSRAFill', {}), bias_init=('ConstantFill', {'value': 0.0}), no_bias=1, engine=self.engine)
        test_mode = False
        if self.split in ['test', 'val']:
            test_mode = True
        bn_blob = self.model.SpatialBN(conv_blob, prefix + '_bn', dim_out, epsilon=self.opts['model_param']['bn_epsilon'], momentum=self.opts['model_param']['bn_momentum'], is_test=test_mode)
        return bn_blob
    
    def conv_bn_relu(self, blob_in, dim_in, dim_out, kernel, stride, prefix, pad=1, group=1):
        bn_blob = self.conv_bn(blob_in, dim_in, dim_out, kernel, stride, prefix, group=group, pad=pad)
        return self.model.Relu(bn_blob, bn_blob)
    
    def multiway_bottleneck_block(self, blob_in, dim_in, dim_out, stride, prefix, dim_inner, group):
        blob_out = self.conv_bn_relu(blob_in, dim_in, dim_inner, 1, 1, prefix + '_branch2a', pad=0)
        conv_blob = self.model.GroupConv_Deprecated(blob_out, prefix + '_branch2b', dim_inner, dim_inner, kernel=3, stride=stride, pad=1, group=group, weight_init=('MSRAFill', {}), bias_init=('ConstantFill', {'value': 0.0}), no_bias=1, engine=self.engine)
        test_mode = False
        if self.split in ['test', 'val']:
            test_mode = True
        bn_blob = self.model.SpatialBN(conv_blob, prefix + '_branch2b_bn', dim_out, epsilon=self.opts['model_param']['bn_epsilon'], momentum=self.opts['model_param']['bn_momentum'], is_test=test_mode)
        relu_blob = self.model.Relu(bn_blob, bn_blob)
        bn_blob = self.conv_bn(relu_blob, dim_inner, dim_out, 1, 1, prefix + '_branch2c', pad=0)
        if self.opts['model_param']['custom_bn_init']:
            self.model.param_init_net.ConstantFill([bn_blob + '_s'], bn_blob + '_s', value=self.opts['model_param']['bn_init_gamma'])
        sc_blob = self.add_shortcut(blob_in, dim_in, dim_out, stride, prefix=prefix + '_branch1')
        sum_blob = self.model.net.Sum([bn_blob, sc_blob], prefix + '_sum')
        return self.model.Relu(sum_blob, sum_blob)
    
    def group_bottleneck_block(self, blob_in, dim_in, dim_out, stride, prefix, dim_inner, group):
        blob_out = self.conv_bn_relu(blob_in, dim_in, dim_inner, 1, 1, prefix + '_branch2a', pad=0)
        blob_out = self.conv_bn_relu(blob_out, dim_inner, dim_inner, 3, stride, prefix + '_branch2b', group=group)
        bn_blob = self.conv_bn(blob_out, dim_inner, dim_out, 1, 1, prefix + '_branch2c', pad=0)
        if self.opts['model_param']['custom_bn_init']:
            self.model.param_init_net.ConstantFill([bn_blob + '_s'], bn_blob + '_s', value=self.opts['model_param']['bn_init_gamma'])
        sc_blob = self.add_shortcut(blob_in, dim_in, dim_out, stride, prefix=prefix + '_branch1')
        sum_blob = self.model.net.Sum([bn_blob, sc_blob], prefix + '_sum')
        return self.model.Relu(sum_blob, sum_blob)
    
    def bottleneck_block(self, blob_in, dim_in, dim_out, stride, prefix, dim_inner, group=None):
        blob_out = self.conv_bn_relu(blob_in, dim_in, dim_inner, 1, 1, prefix + '_branch2a', pad=0)
        blob_out = self.conv_bn_relu(blob_out, dim_inner, dim_inner, 3, stride, prefix + '_branch2b')
        bn_blob = self.conv_bn(blob_out, dim_inner, dim_out, 1, 1, prefix + '_branch2c', pad=0)
        if self.opts['model_param']['custom_bn_init']:
            self.model.param_init_net.ConstantFill([bn_blob + '_s'], bn_blob + '_s', value=self.opts['model_param']['bn_init_gamma'])
        sc_blob = self.add_shortcut(blob_in, dim_in, dim_out, stride, prefix=prefix + '_branch1')
        sum_blob = self.model.net.Sum([bn_blob, sc_blob], prefix + '_sum')
        return self.model.Relu(sum_blob, sum_blob)
    
    def basic_block(self, blob_in, dim_in, dim_out, stride, prefix, dim_inner=None, group=None):
        blob_out = self.conv_bn_relu(blob_in, dim_in, dim_out, 3, stride, prefix + '_branch2a')
        bn_blob = self.conv_bn(blob_out, dim_out, dim_out, 3, 1, prefix + '_branch2b', pad=1)
        sc_blob = self.add_shortcut(blob_in, dim_in, dim_out, stride, prefix=prefix + '_branch1')
        sum_blob = self.model.net.Sum([bn_blob, sc_blob], prefix + '_sum')
        return self.model.Relu(sum_blob, sum_blob)
    
    def residual_layer(self, block_fn, blob_in, dim_in, dim_out, stride, num_blocks, prefix, dim_inner=None, group=None):
        for idx in range(num_blocks):
            block_prefix = '{}_{}'.format(prefix, idx)
            block_stride = (2 if (idx == 0 and stride == 2) else 1)
            blob_in = block_fn(blob_in, dim_in, dim_out, block_stride, block_prefix, dim_inner, group)
            dim_in = dim_out
        return (blob_in, dim_in)


