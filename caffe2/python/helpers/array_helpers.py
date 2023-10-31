from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

def concat(model, blobs_in, blob_out, **kwargs):
    """Depth Concat."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.helpers.array_helpers.concat', 'concat(model, blobs_in, blob_out, **kwargs)', {'model': model, 'blobs_in': blobs_in, 'blob_out': blob_out, 'kwargs': kwargs}, 1)

def depth_concat(model, blobs_in, blob_out, **kwargs):
    """The old depth concat function - we should move to use concat."""
    print('DepthConcat is deprecated. use Concat instead.')
    return concat(blobs_in, blob_out, **kwargs)

