from __future__ import absolute_import, division, print_function, unicode_literals
import copy
import itertools
import warnings
import torch
import torch.nn as nn
import torch.nn.intrinsic as nni
import torch.nn.quantized as nnq
from .default_mappings import DEFAULT_DYNAMIC_MODULE_MAPPING, DEFAULT_MODULE_MAPPING, DEFAULT_QAT_MODULE_MAPPING, DEFAULT_QCONFIG_PROPAGATE_WHITE_LIST
from .stubs import DeQuantStub, QuantWrapper
from .qconfig import default_dynamic_qconfig, float16_dynamic_qconfig

def _propagate_qconfig_helper(module, qconfig_dict, white_list=None, qconfig_parent=None, prefix=''):
    """This is a helper function for `propagate_qconfig_`

    Args:
        module: input module
        qconfig_dict: dictionary that maps from name of submodule to quantization
                     configuration
        white_list: list of quantizable modules
        qconfig_parent: quantization config of parent module, we will fallback to
                       this config when there is no specified config for current
                       module
        prefix: corresponding prefix of the current module, used as key in
                qconfig_dict

    Return:
        None, module is modified inplace with qconfig attached
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.quantization.quantize._propagate_qconfig_helper', "_propagate_qconfig_helper(module, qconfig_dict, white_list=None, qconfig_parent=None, prefix='')", {'DEFAULT_QCONFIG_PROPAGATE_WHITE_LIST': DEFAULT_QCONFIG_PROPAGATE_WHITE_LIST, '_propagate_qconfig_helper': _propagate_qconfig_helper, 'module': module, 'qconfig_dict': qconfig_dict, 'white_list': white_list, 'qconfig_parent': qconfig_parent, 'prefix': prefix}, 0)

def propagate_qconfig_(module, qconfig_dict=None):
    """Propagate qconfig through the module hierarchy and assign `qconfig`
    attribute on each leaf module

    Args:
        module: input module
        qconfig_dict: dictionary that maps from name or type of submodule to
            quantization configuration, qconfig applies to all submodules of a
            given module unless qconfig for the submodules are specified (when
            the submodule already has qconfig attribute)

    Return:
        None, module is modified inplace with qconfig attached
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.quantization.quantize.propagate_qconfig_', 'propagate_qconfig_(module, qconfig_dict=None)', {'_propagate_qconfig_helper': _propagate_qconfig_helper, 'module': module, 'qconfig_dict': qconfig_dict}, 0)

def _observer_forward_hook(self, input, output):
    """Forward hook that calls observer on the output
    """
    return self.activation_post_process(output)

def add_observer_(module):
    """Add observer for the leaf child of the module.

    This function insert observer module to all leaf child module that
    has a valid qconfig attribute.

    Args:
        module: input module with qconfig attributes for all the leaf modules that we want to quantize

    Return:
        None, module is modified inplace with added observer modules and forward_hooks
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.quantization.quantize.add_observer_', 'add_observer_(module)', {'nnq': nnq, 'add_observer_': add_observer_, 'torch': torch, '_observer_forward_hook': _observer_forward_hook, 'module': module}, 0)

def add_quant_dequant(module):
    """Wrap the leaf child module in QuantWrapper if it has a valid qconfig
    Note that this function will modify the children of module inplace and it
    can return a new module which wraps the input module as well.

    Args:
        module: input module with qconfig attributes for all the leaf modules
        that we want to quantize

    Return:
        Either the inplace modified module with submodules wrapped in
        `QuantWrapper` based on qconfig or a new `QuantWrapper` module which
        wraps the input module, the latter case only happens when the input
        module is a leaf module and we want to quantize it.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.quantization.quantize.add_quant_dequant', 'add_quant_dequant(module)', {'QuantWrapper': QuantWrapper, 'add_quant_dequant': add_quant_dequant, 'module': module}, 1)

def prepare(model, inplace=False):
    """Prepares a copy of the model for quantization calibration or quantization-aware training.

    Quantization configuration should be assigned preemptively
    to individual submodules in `.qconfig` attribute.

    The model will be attached with observer or fake quant modules, and qconfig
    will be propagated.

    Args:
        model: input model to be modified in-place
        inplace: carry out model transformations in-place, the original module is mutated
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.quantization.quantize.prepare', 'prepare(model, inplace=False)', {'copy': copy, 'propagate_qconfig_': propagate_qconfig_, 'warnings': warnings, 'add_observer_': add_observer_, 'model': model, 'inplace': inplace}, 1)

def quantize(model, run_fn, run_args, mapping=None, inplace=False):
    """Converts a float model to quantized model.

    First it will prepare the model for calibration or training, then it calls
    `run_fn` which will run the calibration step or training step,
    after that we will call `convert` which will convert the model to a
    quantized model.

    Args:
        model: input model
        run_fn: a function for evaluating the prepared model, can be a
            function that simply runs the prepared model or a training loop
        run_args: positional arguments for `run_fn`
        inplace: carry out model transformations in-place, the original module is mutated
        mapping: correspondence between original module types and quantized counterparts

    Return:
        Quantized model.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.quantization.quantize.quantize', 'quantize(model, run_fn, run_args, mapping=None, inplace=False)', {'DEFAULT_MODULE_MAPPING': DEFAULT_MODULE_MAPPING, 'copy': copy, 'prepare': prepare, 'convert': convert, 'model': model, 'run_fn': run_fn, 'run_args': run_args, 'mapping': mapping, 'inplace': inplace}, 1)

def quantize_dynamic(model, qconfig_spec=None, dtype=torch.qint8, mapping=None, inplace=False):
    """Converts a float model to dynamic (i.e. weights-only) quantized model.

    Replaces specified modules with dynamic weight-only quantized versions and output the quantized model.

    For simplest usage provide `dtype` argument that can be float16 or qint8. Weight-only quantization
    by default is performed for layers with large weights size - i.e. Linear and RNN variants.

    Fine grained control is possible with `qconfig` and `mapping` that act similarly to `quantize()`.
    If `qconfig` is provided, the `dtype` argument is ignored.

    Args:
        module: input model
        qconfig_spec: Either:

            - A dictionary that maps from name or type of submodule to quantization
              configuration, qconfig applies to all submodules of a given
              module unless qconfig for the submodules are specified (when the
              submodule already has qconfig attribute). Entries in the dictionary
              need to be QConfigDynamic instances.

            - A set of types and/or submodule names to apply dynamic quantization to,
              in which case the `dtype` argument is used to specifiy the bit-width

        inplace: carry out model transformations in-place, the original module is mutated
        mapping: maps type of a submodule to a type of corresponding dynamically quantized version
            with which the submodule needs to be replaced

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.quantization.quantize.quantize_dynamic', 'quantize_dynamic(model, qconfig_spec=None, dtype=torch.qint8, mapping=None, inplace=False)', {'torch': torch, 'nn': nn, 'default_dynamic_qconfig': default_dynamic_qconfig, 'float16_dynamic_qconfig': float16_dynamic_qconfig, 'itertools': itertools, 'DEFAULT_DYNAMIC_MODULE_MAPPING': DEFAULT_DYNAMIC_MODULE_MAPPING, 'copy': copy, 'propagate_qconfig_': propagate_qconfig_, 'convert': convert, 'model': model, 'qconfig_spec': qconfig_spec, 'dtype': dtype, 'mapping': mapping, 'inplace': inplace}, 1)

def prepare_qat(model, mapping=None, inplace=False):
    """
    Prepares a copy of the model for quantization calibration or
    quantization-aware training and convers it to quantized version.

    Quantization configuration should be assigned preemptively
    to individual submodules in `.qconfig` attribute.

    Args:
        model: input model to be modified in-place
        mapping: dictionary that maps float modules to quantized modules to be
                 replaced.
        inplace: carry out model transformations in-place, the original module
                 is mutated
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.quantization.quantize.prepare_qat', 'prepare_qat(model, mapping=None, inplace=False)', {'DEFAULT_QAT_MODULE_MAPPING': DEFAULT_QAT_MODULE_MAPPING, 'prepare': prepare, 'convert': convert, 'model': model, 'mapping': mapping, 'inplace': inplace}, 1)

def quantize_qat(model, run_fn, run_args, inplace=False):
    """Do quantization aware training and output a quantized model

    Args:
        model: input model
        run_fn: a function for evaluating the prepared model, can be a
                function that simply runs the prepared model or a training
                loop
        run_args: positional arguments for `run_fn`

    Return:
        Quantized model.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.quantization.quantize.quantize_qat', 'quantize_qat(model, run_fn, run_args, inplace=False)', {'copy': copy, 'prepare_qat': prepare_qat, 'convert': convert, 'model': model, 'run_fn': run_fn, 'run_args': run_args, 'inplace': inplace}, 1)

def convert(module, mapping=None, inplace=False):
    """Converts the float module with observers (where we can get quantization
    parameters) to a quantized module.

    Args:
        module: calibrated module with observers
        mapping: a dictionary that maps from float module type to quantized
                 module type, can be overwrritten to allow swapping user defined
                 Modules
        inplace: carry out model transformations in-place, the original module
                 is mutated

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.quantization.quantize.convert', 'convert(module, mapping=None, inplace=False)', {'DEFAULT_MODULE_MAPPING': DEFAULT_MODULE_MAPPING, 'copy': copy, 'nni': nni, 'convert': convert, 'swap_module': swap_module, 'module': module, 'mapping': mapping, 'inplace': inplace}, 1)

def swap_module(mod, mapping):
    """Swaps the module if it has a quantized counterpart and it has an
    `observer` attached.

    Args:
        mod: input module
        mapping: a dictionary that maps from nn module to nnq module

    Return:
        The corresponding quantized module of `mod`
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.quantization.quantize.swap_module', 'swap_module(mod, mapping)', {'DeQuantStub': DeQuantStub, 'mod': mod, 'mapping': mapping}, 1)

def get_observer_dict(mod, target_dict, prefix=''):
    """Traverse the modules and save all observers into dict.
    This is mainly used for quantization accuracy debug
    Args:
        mod: the top module we want to save all observers
        prefix: the prefix for the current module
        target_dict: the dictionary used to save all the observers
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.quantization.quantize.get_observer_dict', "get_observer_dict(mod, target_dict, prefix='')", {'get_observer_dict': get_observer_dict, 'mod': mod, 'target_dict': target_dict, 'prefix': prefix}, 1)

