"""muji.py does multi-gpu training for caffe2 with no need to change the c++
side code. Everything is defined on the computation graph level.

We support the following use cases:
  - 2 gpus, where peer access is enabled between them.
  - 4 gpus, where peer access are enabled between all of them.
  - 4 gpus, where peer access are enabled in two groups,
    between {1, 2} and {3, 4}
  - 8 gpus, where peer access are enabled in two groups,
    between {1, 2, 3, 4} and {5, 6, 7, 8}.
If above cases are not satisfied, a fallback function which does not rely on
peer access will be called.
"""

import numpy as np
from caffe2.proto import caffe2_pb2
from caffe2.python import workspace

def OnGPU(gpu_id):
    """A utility function that returns a device option protobuf of the
  specified gpu id.
  """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.muji.OnGPU', 'OnGPU(gpu_id)', {'caffe2_pb2': caffe2_pb2, 'workspace': workspace, 'gpu_id': gpu_id}, 1)

def OnCPU():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.muji.OnCPU', 'OnCPU()', {'caffe2_pb2': caffe2_pb2}, 1)

def Allreduce(net, blobs, reduced_affix='_reduced', gpu_indices=None):
    """The general Allreduce interface that reroutes the function calls.
    CPUs and AMD GPUs are not supported because
    GetGpuPeerAccessPattern is called to get gpu peer access pattern.
  """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.muji.Allreduce', "Allreduce(net, blobs, reduced_affix='_reduced', gpu_indices=None)", {'workspace': workspace, 'np': np, 'Allreduce2': Allreduce2, 'Allreduce4': Allreduce4, 'Allreduce4Group2': Allreduce4Group2, 'Allreduce8': Allreduce8, 'AllreduceFallback': AllreduceFallback, 'net': net, 'blobs': blobs, 'reduced_affix': reduced_affix, 'gpu_indices': gpu_indices}, 1)

def Allreduce2(net, blobs, reduced_affix, gpu_indices):
    """Allreduce for 2 gpus.

  Algorithm: 0r <- 0 + 1, 1r <- 0r, where r means "reduced"
  """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.muji.Allreduce2', 'Allreduce2(net, blobs, reduced_affix, gpu_indices)', {'OnGPU': OnGPU, 'net': net, 'blobs': blobs, 'reduced_affix': reduced_affix, 'gpu_indices': gpu_indices}, 2)

def Allreduce4(net, blobs, reduced_affix, gpu_indices):
    """Allreduce for 4 gpus.

  Algorithm: 2 level reduction.
      0r <- 0 + 1, 2r <- 2 + 3
      0r <- 0r + 2r
      2r <- 0r,
      1r <- 0r, 3r <- 2r
  """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.muji.Allreduce4', 'Allreduce4(net, blobs, reduced_affix, gpu_indices)', {'OnGPU': OnGPU, 'net': net, 'blobs': blobs, 'reduced_affix': reduced_affix, 'gpu_indices': gpu_indices}, 4)

def Allreduce4Group2(net, blobs, reduced_affix, gpu_indices):
    """Allreduce for 4 gpus where peer access are enabled in {0,1} and {2,3}

  Algorithm: 2 level reduction.
      0r <- 0 + 1, 2r <- 2 + 3
      0r <- 0r + 2r
      2r <- 0r,
      1r <- 0r, 3r <- 2r
  """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.muji.Allreduce4Group2', 'Allreduce4Group2(net, blobs, reduced_affix, gpu_indices)', {'OnGPU': OnGPU, 'net': net, 'blobs': blobs, 'reduced_affix': reduced_affix, 'gpu_indices': gpu_indices}, 4)

def Allreduce8(net, blobs, reduced_affix, gpu_indices):
    """Allreduce for 8 gpus.

  Algorithm: 3 level reduction.
      0r <- 0 + 1, 2r <- 2 + 3, 4r <- 4 + 5, 6r <- 6 + 7
      0r <- 0r + 2r, 4r <- 4r + 6r
      0r <- 0r + 4r
      4r <- 0r
      2r <- 0r, 6r <- 4r
      1r <- 0r, 3r <- 2r, 5r <- 4r, 7r <- 6r
  """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.muji.Allreduce8', 'Allreduce8(net, blobs, reduced_affix, gpu_indices)', {'OnGPU': OnGPU, 'net': net, 'blobs': blobs, 'reduced_affix': reduced_affix, 'gpu_indices': gpu_indices}, 1)

def AllreduceFallback(net, blobs, reduced_affix, gpu_indices):
    """A fallback option for Allreduce with no assumption on p2p.

  Algorithm: a flat operation on gpu 0
      0r <- 0
      0r <- 0r + i for i in gpu_indices[1:]
      ir <- 0r for i in gpu_indices[1:]
  """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.muji.AllreduceFallback', 'AllreduceFallback(net, blobs, reduced_affix, gpu_indices)', {'OnGPU': OnGPU, 'net': net, 'blobs': blobs, 'reduced_affix': reduced_affix, 'gpu_indices': gpu_indices}, 1)

