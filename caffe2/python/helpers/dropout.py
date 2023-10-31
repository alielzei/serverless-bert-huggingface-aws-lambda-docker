from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

def dropout(model, blob_in, blob_out, use_cudnn=False, **kwargs):
    """dropout"""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.helpers.dropout.dropout', 'dropout(model, blob_in, blob_out, use_cudnn=False, **kwargs)', {'model': model, 'blob_in': blob_in, 'blob_out': blob_out, 'use_cudnn': use_cudnn, 'kwargs': kwargs}, 1)

