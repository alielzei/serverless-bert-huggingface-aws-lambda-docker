from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

def transpose(model, blob_in, blob_out, use_cudnn=False, **kwargs):
    """Transpose."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.helpers.algebra.transpose', 'transpose(model, blob_in, blob_out, use_cudnn=False, **kwargs)', {'model': model, 'blob_in': blob_in, 'blob_out': blob_out, 'use_cudnn': use_cudnn, 'kwargs': kwargs}, 1)

def sum(model, blob_in, blob_out, **kwargs):
    """Sum"""
    return model.net.Sum(blob_in, blob_out, **kwargs)

def batch_mat_mul(model, blob_in, blob_out, enable_tensor_core=False, **kwargs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.helpers.algebra.batch_mat_mul', 'batch_mat_mul(model, blob_in, blob_out, enable_tensor_core=False, **kwargs)', {'model': model, 'blob_in': blob_in, 'blob_out': blob_out, 'enable_tensor_core': enable_tensor_core, 'kwargs': kwargs}, 1)

