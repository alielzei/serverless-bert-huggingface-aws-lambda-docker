from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import copy
import torch.nn.intrinsic.modules.fused as torch_fused

def fuse_conv_bn(conv, bn):
    """Given the conv and bn modules, fuses them and returns the fused module

    Args:
        conv: Module instance of type conv2d
        bn: Spatial BN instance that needs to be fused with the conv

    Examples::

        >>> m1 = nn.Conv2d(10, 20, 3)
        >>> b1 = nn.BatchNorm2d(20)
        >>> m2 = fuse_conv_bn(m1, b1)
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.quantization.fuse_modules.fuse_conv_bn', 'fuse_conv_bn(conv, bn)', {'torch': torch, 'conv': conv, 'bn': bn}, 1)

def fuse_conv_bn_relu(conv, bn, relu):
    """Given the conv and bn modules, fuses them and returns the fused module

    Args:
        conv: Module instance of type conv2d
        bn: Spatial BN instance that needs to be fused with the conv

    Examples::

        >>> m1 = nn.Conv2d(10, 20, 3)
        >>> b1 = nn.BatchNorm2d(20)
        >>> m2 = fuse_conv_bn(m1, b1)
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.quantization.fuse_modules.fuse_conv_bn_relu', 'fuse_conv_bn_relu(conv, bn, relu)', {'torch_fused': torch_fused, 'torch': torch, 'conv': conv, 'bn': bn, 'relu': relu}, 1)

def _get_module(model, submodule_key):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.quantization.fuse_modules._get_module', '_get_module(model, submodule_key)', {'model': model, 'submodule_key': submodule_key}, 1)

def _set_module(model, submodule_key, module):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.quantization.fuse_modules._set_module', '_set_module(model, submodule_key, module)', {'model': model, 'submodule_key': submodule_key, 'module': module}, 0)

def fuse_known_modules(mod_list):
    """Returns a list of modules that fuses the operations specified
     in the input module list.

    Fuses only the following sequence of modules:
    conv, bn
    conv, bn, relu
    conv, relu
    linear, relu
    For these sequences, the first element in the output module list performs
    the fused operation. The rest of the elements are set to nn.Identity()
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.quantization.fuse_modules.fuse_known_modules', 'fuse_known_modules(mod_list)', {'torch': torch, 'fuse_conv_bn': fuse_conv_bn, 'fuse_conv_bn_relu': fuse_conv_bn_relu, 'mod_list': mod_list}, 1)

def _fuse_modules(model, modules_to_fuse, fuser_func=fuse_known_modules):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.quantization.fuse_modules._fuse_modules', '_fuse_modules(model, modules_to_fuse, fuser_func=fuse_known_modules)', {'_get_module': _get_module, '_set_module': _set_module, 'model': model, 'modules_to_fuse': modules_to_fuse, 'fuser_func': fuser_func, 'fuse_known_modules': fuse_known_modules}, 0)

def fuse_modules(model, modules_to_fuse, inplace=False, fuser_func=fuse_known_modules):
    """Fuses a list of modules into a single module

    Fuses only the following sequence of modules:

    * conv, bn

    * conv, bn, relu

    * conv, relu

    * linear, relu

    All other sequences are left unchanged.
    For these sequences, replaces the first item in the list
    with the fused module, replacing the rest of the modules
    with identity.

    Arguments:
        model: Model containing the modules to be fused
        modules_to_fuse: list of list of module names to fuse. Can also be a list
                         of strings if there is only a single list of modules to fuse.
        inplace: bool specifying if fusion happens in place on the model, by default
                 a new model is returned
        fuser_func: Function that takes in a list of modules and outputs a list of fused modules
                    of the same length. For example,
                    fuser_func([convModule, BNModule]) returns the list [ConvBNModule, nn.Identity()]
                    Defaults to torch.quantization.fuse_known_modules
    Returns:
        model with fused modules. A new copy is created if inplace=True.

    Examples::

            >>> m = myModel()
            >>> # m is a module containing  the sub-modules below
            >>> modules_to_fuse = [ ['conv1', 'bn1', 'relu1'], ['submodule.conv', 'submodule.relu']]
            >>> fused_m = torch.quantization.fuse_modules(m, modules_to_fuse)
            >>> output = fused_m(input)

            >>> m = myModel()
            >>> # Alternately provide a single list of modules to fuse
            >>> modules_to_fuse = ['conv1', 'bn1', 'relu1']
            >>> fused_m = torch.quantization.fuse_modules(m, modules_to_fuse)
            >>> output = fused_m(input)

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.quantization.fuse_modules.fuse_modules', 'fuse_modules(model, modules_to_fuse, inplace=False, fuser_func=fuse_known_modules)', {'copy': copy, '_fuse_modules': _fuse_modules, 'model': model, 'modules_to_fuse': modules_to_fuse, 'inplace': inplace, 'fuser_func': fuser_func, 'fuse_known_modules': fuse_known_modules}, 1)

