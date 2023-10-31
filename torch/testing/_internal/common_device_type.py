import inspect
import threading
from functools import wraps
import unittest
import os
import torch
from torch.testing._internal.common_utils import TestCase, TEST_WITH_ROCM, TEST_MKL, skipCUDANonDefaultStreamIf
device_type_test_bases = []


class DeviceTypeTestBase(TestCase):
    device_type = 'generic_device_type'
    _tls = threading.local()
    _tls.precision = TestCase.precision
    
    @property
    def precision(self):
        return self._tls.precision
    
    @precision.setter
    def precision(self, prec):
        self._tls.precision = prec
    
    @classmethod
    def get_primary_device(cls):
        return cls.device_type
    
    @classmethod
    def get_all_devices(cls):
        return [cls.get_primary_device()]
    
    @classmethod
    def _get_dtypes(cls, test):
        if not hasattr(test, 'dtypes'):
            return None
        return test.dtypes.get(cls.device_type, test.dtypes.get('all', None))
    
    def _get_precision_override(self, test, dtype):
        if not hasattr(test, 'precision_overrides'):
            return self.precision
        return test.precision_overrides.get(dtype, self.precision)
    
    @classmethod
    def instantiate_test(cls, name, test):
        test_name = name + '_' + cls.device_type
        dtypes = cls._get_dtypes(test)
        if dtypes is None:
            assert not hasattr(cls, test_name), 'Redefinition of test {0}'.format(test_name)
            
            @wraps(test)
            def instantiated_test(self, test=test):
                device_arg = (cls.get_primary_device() if not hasattr(test, 'num_required_devices') else cls.get_all_devices())
                return test(self, device_arg)
            setattr(cls, test_name, instantiated_test)
        else:
            for dtype in dtypes:
                dtype_str = str(dtype).split('.')[1]
                dtype_test_name = test_name + '_' + dtype_str
                assert not hasattr(cls, dtype_test_name), 'Redefinition of test {0}'.format(dtype_test_name)
                
                @wraps(test)
                def instantiated_test(self, test=test, dtype=dtype):
                    device_arg = (cls.get_primary_device() if not hasattr(test, 'num_required_devices') else cls.get_all_devices())
                    guard_precision = self.precision
                    try:
                        self.precision = self._get_precision_override(test, dtype)
                        result = test(self, device_arg, dtype)
                    finally:
                        self.precision = guard_precision
                    return result
                setattr(cls, dtype_test_name, instantiated_test)



class CPUTestBase(DeviceTypeTestBase):
    device_type = 'cpu'



class CUDATestBase(DeviceTypeTestBase):
    device_type = 'cuda'
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True
    
    def has_cudnn(self):
        return not self.no_cudnn
    
    @classmethod
    def get_primary_device(cls):
        return cls.primary_device
    
    @classmethod
    def get_all_devices(cls):
        primary_device_idx = int(cls.get_primary_device().split(':')[1])
        num_devices = torch.cuda.device_count()
        prim_device = cls.get_primary_device()
        cuda_str = 'cuda:{0}'
        non_primary_devices = [cuda_str.format(idx) for idx in range(num_devices) if idx != primary_device_idx]
        return [prim_device] + non_primary_devices
    
    @classmethod
    def setUpClass(cls):
        t = torch.ones(1).cuda()
        cls.no_magma = not torch.cuda.has_magma
        cls.no_cudnn = not ((TEST_WITH_ROCM or torch.backends.cudnn.is_acceptable(t)))
        cls.cudnn_version = (None if cls.no_cudnn else torch.backends.cudnn.version())
        cls.primary_device = 'cuda:{0}'.format(torch.cuda.current_device())

device_type_test_bases.append(CPUTestBase)
if torch.cuda.is_available():
    device_type_test_bases.append(CUDATestBase)
PYTORCH_CUDA_MEMCHECK = os.getenv('PYTORCH_CUDA_MEMCHECK', '0') == '1'

def instantiate_device_type_tests(generic_test_class, scope, except_for=None):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.testing._internal.common_device_type.instantiate_device_type_tests', 'instantiate_device_type_tests(generic_test_class, scope, except_for=None)', {'device_type_test_bases': device_type_test_bases, 'inspect': inspect, 'generic_test_class': generic_test_class, 'scope': scope, 'except_for': except_for}, 0)


class skipIf(object):
    
    def __init__(self, dep, reason, device_type=None):
        self.dep = dep
        self.reason = reason
        self.device_type = device_type
    
    def __call__(self, fn):
        
        @wraps(fn)
        def dep_fn(slf, device, *args, **kwargs):
            if (self.device_type is None or self.device_type == slf.device_type):
                if ((isinstance(self.dep, str) and getattr(slf, self.dep, True)) or (isinstance(self.dep, bool) and self.dep)):
                    raise unittest.SkipTest(self.reason)
            return fn(slf, device, *args, **kwargs)
        return dep_fn



class skipCPUIf(skipIf):
    
    def __init__(self, dep, reason):
        super(skipCPUIf, self).__init__(dep, reason, device_type='cpu')



class skipCUDAIf(skipIf):
    
    def __init__(self, dep, reason):
        super(skipCUDAIf, self).__init__(dep, reason, device_type='cuda')


def largeCUDATensorTest(size):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_device_type.largeCUDATensorTest', 'largeCUDATensorTest(size)', {'torch': torch, 'unittest': unittest, 'size': size}, 1)


class expectedFailure(object):
    
    def __init__(self, device_type):
        self.device_type = device_type
    
    def __call__(self, fn):
        
        @wraps(fn)
        def efail_fn(slf, device, *args, **kwargs):
            if (self.device_type is None or self.device_type == slf.device_type):
                try:
                    fn(slf, device, *args, **kwargs)
                except Exception:
                    return
                else:
                    slf.fail('expected test to fail, but it passed')
            return fn(slf, device, *args, **kwargs)
        return efail_fn



class onlyOn(object):
    
    def __init__(self, device_type):
        self.device_type = device_type
    
    def __call__(self, fn):
        
        @wraps(fn)
        def only_fn(slf, device, *args, **kwargs):
            if self.device_type != slf.device_type:
                reason = 'Only runs on {0}'.format(self.device_type)
                raise unittest.SkipTest(reason)
            return fn(slf, device, *args, **kwargs)
        return only_fn



class deviceCountAtLeast(object):
    
    def __init__(self, num_required_devices):
        self.num_required_devices = num_required_devices
    
    def __call__(self, fn):
        assert not hasattr(fn, 'num_required_devices'), 'deviceCountAtLeast redefinition for {0}'.format(fn.__name__)
        fn.num_required_devices = self.num_required_devices
        
        @wraps(fn)
        def multi_fn(slf, devices, *args, **kwargs):
            if len(devices) < self.num_required_devices:
                reason = 'fewer than {0} devices detected'.format(self.num_required_devices)
                raise unittest.SkipTest(reason)
            return fn(slf, devices, *args, **kwargs)
        return multi_fn


def onlyOnCPUAndCUDA(fn):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_device_type.onlyOnCPUAndCUDA', 'onlyOnCPUAndCUDA(fn)', {'wraps': wraps, 'unittest': unittest, 'fn': fn}, 1)


class precisionOverride(object):
    
    def __init__(self, d):
        assert isinstance(d, dict), 'precisionOverride not given a dtype : precision dict!'
        for (dtype, prec) in d.items():
            assert isinstance(dtype, torch.dtype), 'precisionOverride given unknown dtype {0}'.format(dtype)
        self.d = d
    
    def __call__(self, fn):
        fn.precision_overrides = self.d
        return fn



class dtypes(object):
    
    def __init__(self, *args, **kwargs):
        assert (args is not None and len(args) != 0), 'No dtypes given'
        assert all((isinstance(arg, torch.dtype) for arg in args)), 'Unknown dtype in {0}'.format(str(args))
        self.args = args
        self.device_type = kwargs.get('device_type', 'all')
    
    def __call__(self, fn):
        d = getattr(fn, 'dtypes', {})
        assert self.device_type not in d, 'dtypes redefinition for {0}'.format(self.device_type)
        d[self.device_type] = self.args
        fn.dtypes = d
        return fn



class dtypesIfCPU(dtypes):
    
    def __init__(self, *args):
        super(dtypesIfCPU, self).__init__(*args, device_type='cpu')



class dtypesIfCUDA(dtypes):
    
    def __init__(self, *args):
        super(dtypesIfCUDA, self).__init__(*args, device_type='cuda')


def onlyCPU(fn):
    return onlyOn('cpu')(fn)

def onlyCUDA(fn):
    return onlyOn('cuda')(fn)

def expectedFailureCUDA(fn):
    return expectedFailure('cuda')(fn)

def skipCPUIfNoLapack(fn):
    return skipCPUIf(not torch._C.has_lapack, 'PyTorch compiled without Lapack')(fn)

def skipCPUIfNoMkl(fn):
    return skipCPUIf(not TEST_MKL, 'PyTorch is built without MKL support')(fn)

def skipCUDAIfNoMagma(fn):
    return skipCUDAIf('no_magma', 'no MAGMA library detected')(skipCUDANonDefaultStreamIf(True)(fn))

def skipCUDAIfRocm(fn):
    return skipCUDAIf(TEST_WITH_ROCM, "test doesn't currently work on the ROCm stack")(fn)

def skipCUDAIfNotRocm(fn):
    return skipCUDAIf(not TEST_WITH_ROCM, "test doesn't currently work on the CUDA stack")(fn)

def skipCUDAIfCudnnVersionLessThan(version=0):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_device_type.skipCUDAIfCudnnVersionLessThan', 'skipCUDAIfCudnnVersionLessThan(version=0)', {'wraps': wraps, 'unittest': unittest, 'version': version}, 1)

def skipCUDAIfNoCudnn(fn):
    return skipCUDAIfCudnnVersionLessThan(0)(fn)

