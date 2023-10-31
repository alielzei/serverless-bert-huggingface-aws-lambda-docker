from __future__ import absolute_import, division, print_function, unicode_literals
from .quantize import *
from .observer import *
from .qconfig import *
from .fake_quantize import *
from .fuse_modules import fuse_modules
from .stubs import *

def default_eval_fn(model, calib_data):
    """
    Default evaluation function takes a torch.utils.data.Dataset or a list of
    input Tensors and run the model on the dataset
    """
    for (data, target) in calib_data:
        model(data)
_all__ = ['QuantWrapper', 'QuantStub', 'DeQuantStub', 'quantize', 'prepare', 'convert', 'propagate_qconfig_', 'add_quant_dequant', 'add_observer_', 'swap_module', 'default_eval_fn', 'get_observer_dict', 'ObserverBase', 'WeightObserver', 'observer', 'default_observer', 'default_weight_observer', 'QConfig', 'default_qconfig', 'default_dynamic_qconfig', 'float16_dynamic_qconfig', 'default_qat_qconfig', 'prepare_qat', 'quantize_qat', 'fuse_modules', 'quantize_dynamic']

