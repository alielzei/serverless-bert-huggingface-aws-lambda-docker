from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

def image_input(model, blob_in, blob_out, order='NCHW', use_gpu_transform=False, **kwargs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.helpers.tools.image_input', "image_input(model, blob_in, blob_out, order='NCHW', use_gpu_transform=False, **kwargs)", {'model': model, 'blob_in': blob_in, 'blob_out': blob_out, 'order': order, 'use_gpu_transform': use_gpu_transform, 'kwargs': kwargs}, 1)

def video_input(model, blob_in, blob_out, **kwargs):
    outputs = model.net.VideoInput(blob_in, blob_out, **kwargs)
    return outputs

