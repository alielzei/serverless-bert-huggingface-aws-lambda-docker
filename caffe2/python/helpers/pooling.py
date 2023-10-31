from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

def max_pool(model, blob_in, blob_out, use_cudnn=False, order='NCHW', **kwargs):
    """Max pooling"""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.helpers.pooling.max_pool', "max_pool(model, blob_in, blob_out, use_cudnn=False, order='NCHW', **kwargs)", {'model': model, 'blob_in': blob_in, 'blob_out': blob_out, 'use_cudnn': use_cudnn, 'order': order, 'kwargs': kwargs}, 1)

def average_pool(model, blob_in, blob_out, use_cudnn=False, order='NCHW', **kwargs):
    """Average pooling"""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.helpers.pooling.average_pool', "average_pool(model, blob_in, blob_out, use_cudnn=False, order='NCHW', **kwargs)", {'model': model, 'blob_in': blob_in, 'blob_out': blob_out, 'use_cudnn': use_cudnn, 'order': order, 'kwargs': kwargs}, 1)

def max_pool_with_index(model, blob_in, blob_out, order='NCHW', **kwargs):
    """Max pooling with an explicit index of max position"""
    return model.net.MaxPoolWithIndex(blob_in, [blob_out, blob_out + '_index'], order=order, **kwargs)[0]

