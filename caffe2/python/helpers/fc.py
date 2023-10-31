from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import core
from caffe2.python.modeling import initializers
from caffe2.python.modeling.parameter_info import ParameterTags

def _FC_or_packed_FC(model, op_call, blob_in, blob_out, dim_in, dim_out, weight_init=None, bias_init=None, WeightInitializer=None, BiasInitializer=None, enable_tensor_core=False, float16_compute=False, **kwargs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.helpers.fc._FC_or_packed_FC', '_FC_or_packed_FC(model, op_call, blob_in, blob_out, dim_in, dim_out, weight_init=None, bias_init=None, WeightInitializer=None, BiasInitializer=None, enable_tensor_core=False, float16_compute=False, **kwargs)', {'initializers': initializers, 'ParameterTags': ParameterTags, 'model': model, 'op_call': op_call, 'blob_in': blob_in, 'blob_out': blob_out, 'dim_in': dim_in, 'dim_out': dim_out, 'weight_init': weight_init, 'bias_init': bias_init, 'WeightInitializer': WeightInitializer, 'BiasInitializer': BiasInitializer, 'enable_tensor_core': enable_tensor_core, 'float16_compute': float16_compute, 'kwargs': kwargs}, 1)

def fc(model, *args, **kwargs):
    return _FC_or_packed_FC(model, model.net.FC, *args, **kwargs)

def packed_fc(model, *args, **kwargs):
    return _FC_or_packed_FC(model, model.net.PackedFC, *args, **kwargs)

def fc_decomp(model, blob_in, blob_out, dim_in, dim_out, rank_approx=5, weight_init=None, bias_init=None, WeightInitializer=None, BiasInitializer=None, **kwargs):
    """FC_Decomp version
    Here we assume that the rank of original input is bigger than 5.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.helpers.fc.fc_decomp', 'fc_decomp(model, blob_in, blob_out, dim_in, dim_out, rank_approx=5, weight_init=None, bias_init=None, WeightInitializer=None, BiasInitializer=None, **kwargs)', {'initializers': initializers, 'model': model, 'blob_in': blob_in, 'blob_out': blob_out, 'dim_in': dim_in, 'dim_out': dim_out, 'rank_approx': rank_approx, 'weight_init': weight_init, 'bias_init': bias_init, 'WeightInitializer': WeightInitializer, 'BiasInitializer': BiasInitializer, 'kwargs': kwargs}, 1)

def fc_prune(model, blob_in, blob_out, dim_in, dim_out, weight_init=None, bias_init=None, mask_init=None, threshold=1e-05, need_compress_rate=False, comp_lb=0.05, **kwargs):
    """FC_Prune version
    Runnable so far. Great!:)
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.helpers.fc.fc_prune', 'fc_prune(model, blob_in, blob_out, dim_in, dim_out, weight_init=None, bias_init=None, mask_init=None, threshold=1e-05, need_compress_rate=False, comp_lb=0.05, **kwargs)', {'core': core, 'model': model, 'blob_in': blob_in, 'blob_out': blob_out, 'dim_in': dim_in, 'dim_out': dim_out, 'weight_init': weight_init, 'bias_init': bias_init, 'mask_init': mask_init, 'threshold': threshold, 'need_compress_rate': need_compress_rate, 'comp_lb': comp_lb, 'kwargs': kwargs}, 1)

def fc_sparse(model, blob_in, blob_out, w_csr, iw, jw, bias, **kwargs):
    """FC_Sparse: Only takes in allocated weights"""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.helpers.fc.fc_sparse', 'fc_sparse(model, blob_in, blob_out, w_csr, iw, jw, bias, **kwargs)', {'model': model, 'blob_in': blob_in, 'blob_out': blob_out, 'w_csr': w_csr, 'iw': iw, 'jw': jw, 'bias': bias, 'kwargs': kwargs}, 1)

