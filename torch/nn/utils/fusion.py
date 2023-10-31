from __future__ import absolute_import, division, print_function, unicode_literals
import copy
import torch

def fuse_conv_bn_eval(conv, bn):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.utils.fusion.fuse_conv_bn_eval', 'fuse_conv_bn_eval(conv, bn)', {'copy': copy, 'fuse_conv_bn_weights': fuse_conv_bn_weights, 'conv': conv, 'bn': bn}, 1)

def fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.utils.fusion.fuse_conv_bn_weights', 'fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b)', {'torch': torch, 'conv_w': conv_w, 'conv_b': conv_b, 'bn_rm': bn_rm, 'bn_rv': bn_rv, 'bn_eps': bn_eps, 'bn_w': bn_w, 'bn_b': bn_b}, 2)

