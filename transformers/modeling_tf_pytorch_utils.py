""" PyTorch - TF 2.0 general utilities."""

import os
import re
import numpy
from .utils import logging
logger = logging.get_logger(__name__)

def convert_tf_weight_name_to_pt_weight_name(tf_name, start_prefix_to_remove=''):
    """Convert a TF 2.0 model variable name in a pytorch model weight name.

    Conventions for TF2.0 scopes -> PyTorch attribute names conversions:
        - '$1___$2' is replaced by $2 (can be used to duplicate or remove layers in TF2.0 vs PyTorch)
        - '_._' is replaced by a new level separation (can be used to convert TF2.0 lists in PyTorch nn.ModulesList)

    return tuple with:
        - pytorch model weight name
        - transpose: boolean indicating weither TF2.0 and PyTorch weights matrices are transposed with regards to each other
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.modeling_tf_pytorch_utils.convert_tf_weight_name_to_pt_weight_name', "convert_tf_weight_name_to_pt_weight_name(tf_name, start_prefix_to_remove='')", {'re': re, 'tf_name': tf_name, 'start_prefix_to_remove': start_prefix_to_remove}, 2)

def load_pytorch_checkpoint_in_tf2_model(tf_model, pytorch_checkpoint_path, tf_inputs=None, allow_missing_keys=False):
    """Load pytorch checkpoints in a TF 2.0 model"""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.modeling_tf_pytorch_utils.load_pytorch_checkpoint_in_tf2_model', 'load_pytorch_checkpoint_in_tf2_model(tf_model, pytorch_checkpoint_path, tf_inputs=None, allow_missing_keys=False)', {'logger': logger, 'os': os, 'load_pytorch_weights_in_tf2_model': load_pytorch_weights_in_tf2_model, 'tf_model': tf_model, 'pytorch_checkpoint_path': pytorch_checkpoint_path, 'tf_inputs': tf_inputs, 'allow_missing_keys': allow_missing_keys}, 1)

def load_pytorch_model_in_tf2_model(tf_model, pt_model, tf_inputs=None, allow_missing_keys=False):
    """Load pytorch checkpoints in a TF 2.0 model"""
    pt_state_dict = pt_model.state_dict()
    return load_pytorch_weights_in_tf2_model(tf_model, pt_state_dict, tf_inputs=tf_inputs, allow_missing_keys=allow_missing_keys)

def load_pytorch_weights_in_tf2_model(tf_model, pt_state_dict, tf_inputs=None, allow_missing_keys=False):
    """Load pytorch state_dict in a TF 2.0 model."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.modeling_tf_pytorch_utils.load_pytorch_weights_in_tf2_model', 'load_pytorch_weights_in_tf2_model(tf_model, pt_state_dict, tf_inputs=None, allow_missing_keys=False)', {'logger': logger, 'convert_tf_weight_name_to_pt_weight_name': convert_tf_weight_name_to_pt_weight_name, 're': re, 'numpy': numpy, 'tf_model': tf_model, 'pt_state_dict': pt_state_dict, 'tf_inputs': tf_inputs, 'allow_missing_keys': allow_missing_keys}, 1)

def load_tf2_checkpoint_in_pytorch_model(pt_model, tf_checkpoint_path, tf_inputs=None, allow_missing_keys=False):
    """Load TF 2.0 HDF5 checkpoint in a PyTorch model
    We use HDF5 to easily do transfer learning
    (see https://github.com/tensorflow/tensorflow/blob/ee16fcac960ae660e0e4496658a366e2f745e1f0/tensorflow/python/keras/engine/network.py#L1352-L1357).
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.modeling_tf_pytorch_utils.load_tf2_checkpoint_in_pytorch_model', 'load_tf2_checkpoint_in_pytorch_model(pt_model, tf_checkpoint_path, tf_inputs=None, allow_missing_keys=False)', {'logger': logger, 'load_tf2_model_in_pytorch_model': load_tf2_model_in_pytorch_model, 'pt_model': pt_model, 'tf_checkpoint_path': tf_checkpoint_path, 'tf_inputs': tf_inputs, 'allow_missing_keys': allow_missing_keys}, 1)

def load_tf2_model_in_pytorch_model(pt_model, tf_model, allow_missing_keys=False):
    """Load TF 2.0 model in a pytorch model"""
    weights = tf_model.weights
    return load_tf2_weights_in_pytorch_model(pt_model, weights, allow_missing_keys=allow_missing_keys)

def load_tf2_weights_in_pytorch_model(pt_model, tf_weights, allow_missing_keys=False):
    """Load TF2.0 symbolic weights in a PyTorch model"""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.modeling_tf_pytorch_utils.load_tf2_weights_in_pytorch_model', 'load_tf2_weights_in_pytorch_model(pt_model, tf_weights, allow_missing_keys=False)', {'logger': logger, 'convert_tf_weight_name_to_pt_weight_name': convert_tf_weight_name_to_pt_weight_name, 'numpy': numpy, 'pt_model': pt_model, 'tf_weights': tf_weights, 'allow_missing_keys': allow_missing_keys}, 1)

