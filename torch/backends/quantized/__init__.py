from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import torch
import types

def _get_qengine_id(qengine):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.backends.quantized.__init__._get_qengine_id', '_get_qengine_id(qengine)', {'qengine': qengine}, 1)

def _get_qengine_str(qengine):
    all_engines = {0: 'none', 1: 'fbgemm', 2: 'qnnpack'}
    return all_engines.get(qengine)


class _QEngineProp(object):
    
    def __get__(self, obj, objtype):
        return _get_qengine_str(torch._C._get_qengine())
    
    def __set__(self, obj, val):
        torch._C._set_qengine(_get_qengine_id(val))



class _SupportedQEnginesProp(object):
    
    def __get__(self, obj, objtype):
        qengines = torch._C._supported_qengines()
        return [_get_qengine_str(qe) for qe in qengines]
    
    def __set__(self, obj, val):
        raise RuntimeError('Assignment not supported')



class QuantizedEngine(types.ModuleType):
    
    def __init__(self, m, name):
        super(QuantizedEngine, self).__init__(name)
        self.m = m
    
    def __getattr__(self, attr):
        return self.m.__getattribute__(attr)
    engine = _QEngineProp()
    supported_engines = _SupportedQEnginesProp()

sys.modules[__name__] = QuantizedEngine(sys.modules[__name__], __name__)

