"""Importing this file must **not** initialize CUDA context. test_distributed
relies on this assumption to properly run. This means that when this is imported
no CUDA calls shall be made, including torch.cuda.device_count(), etc.

torch.testing._internal.common_cuda.py can freely initialize CUDA context when imported.
"""

import sys
import os
import platform
import re
import gc
import types
from functools import partial
import inspect
import io
import argparse
import unittest
import warnings
import random
import contextlib
import socket
import subprocess
import time
from collections import OrderedDict
from contextlib import contextmanager
from functools import wraps
from itertools import product
from copy import deepcopy
from numbers import Number
import tempfile
import json
if sys.version_info[0] == 2:
    from urllib2 import urlopen
else:
    from urllib.request import urlopen
import __main__
import errno
from torch.testing._internal import expecttest
import torch
import torch.cuda
from torch._utils_internal import get_writable_path
from torch._six import string_classes, inf
import torch.backends.cudnn
import torch.backends.mkl
from enum import Enum
from torch.autograd import gradcheck
from torch.autograd.gradcheck import gradgradcheck
torch.backends.disable_global_flags()
IS_SANDCASTLE = (os.getenv('SANDCASTLE') == '1' or os.getenv('TW_JOB_USER') == 'sandcastle')


class ProfilingMode(Enum):
    LEGACY = 1
    SIMPLE = 2
    PROFILING = 3


@contextmanager
def enable_profiling_mode():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.testing._internal.common_utils.enable_profiling_mode', 'enable_profiling_mode()', {'GRAPH_EXECUTOR': GRAPH_EXECUTOR, 'ProfilingMode': ProfilingMode, 'torch': torch, 'contextmanager': contextmanager}, 0)
func_call = torch._C.ScriptFunction.__call__
meth_call = torch._C.ScriptMethod.__call__

def prof_callable(callable, *args, **kwargs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_utils.prof_callable', 'prof_callable(callable, *args, **kwargs)', {'GRAPH_EXECUTOR': GRAPH_EXECUTOR, 'ProfilingMode': ProfilingMode, 'enable_profiling_mode': enable_profiling_mode, 'callable': callable, 'args': args, 'kwargs': kwargs}, 1)

def prof_func_call(*args, **kwargs):
    return prof_callable(func_call, *args, **kwargs)

def prof_meth_call(*args, **kwargs):
    return prof_callable(meth_call, *args, **kwargs)
torch._C.ScriptFunction.__call__ = prof_func_call
torch._C.ScriptMethod.__call__ = prof_meth_call
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--subprocess', action='store_true', help='whether to run each test in a subprocess')
parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--accept', action='store_true')
parser.add_argument('--ge_config', type=str)
parser.add_argument('--test_bailouts', action='store_true')
GRAPH_EXECUTOR = (ProfilingMode.SIMPLE if IS_SANDCASTLE else ProfilingMode.PROFILING)
(args, remaining) = parser.parse_known_args()
if args.ge_config == 'legacy':
    GRAPH_EXECUTOR = ProfilingMode.LEGACY
elif args.ge_config == 'simple':
    GRAPH_EXECUTOR = ProfilingMode.SIMPLE
TEST_BAILOUTS = args.test_bailouts
TEST_IN_SUBPROCESS = args.subprocess
SEED = args.seed
if not expecttest.ACCEPT:
    expecttest.ACCEPT = args.accept
UNITTEST_ARGS = [sys.argv[0]] + remaining
torch.manual_seed(SEED)

def shell(command, cwd=None, env=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_utils.shell', 'shell(command, cwd=None, env=None)', {'sys': sys, 'torch': torch, 'subprocess': subprocess, 'command': command, 'cwd': cwd, 'env': env}, 1)

def repeat_test_for_types(dtypes):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_utils.repeat_test_for_types', 'repeat_test_for_types(dtypes)', {'wraps': wraps, 'PY34': PY34, 'TestCase': TestCase, 'dtypes': dtypes}, 1)
IS_PYTORCH_CI = bool(os.environ.get('IS_PYTORCH_CI'))
IN_CIRCLECI = bool(os.environ.get('IN_CIRCLECI'))
TEST_REPORT_SOURCE_OVERRIDE = os.environ.get('TEST_REPORT_SOURCE_OVERRIDE')
PY3 = sys.version_info > (3, 0)
PY34 = sys.version_info >= (3, 4)

def run_tests(argv=UNITTEST_ARGS):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.testing._internal.common_utils.run_tests', 'run_tests(argv=UNITTEST_ARGS)', {'TEST_IN_SUBPROCESS': TEST_IN_SUBPROCESS, 'unittest': unittest, '__main__': __main__, 'shell': shell, 'sys': sys, 'IN_CIRCLECI': IN_CIRCLECI, 'TEST_REPORT_SOURCE_OVERRIDE': TEST_REPORT_SOURCE_OVERRIDE, 'os': os, 'PY3': PY3, 'argv': argv, 'UNITTEST_ARGS': UNITTEST_ARGS}, 0)
IS_WINDOWS = sys.platform == 'win32'
IS_MACOS = sys.platform == 'darwin'
IS_PPC = platform.machine() == 'ppc64le'
if IS_WINDOWS:
    
    @contextmanager
    def TemporaryFileName():
        import custom_funtemplate
        custom_funtemplate.rewrite_template('torch.testing._internal.common_utils.TemporaryFileName', 'TemporaryFileName()', {'tempfile': tempfile, 'os': os, 'contextmanager': contextmanager}, 0)
else:
    
    @contextmanager
    def TemporaryFileName():
        with tempfile.NamedTemporaryFile() as f:
            yield f.name

def _check_module_exists(name):
    """Returns if a top-level module with :attr:`name` exists *without**
    importing it. This is generally safer than try-catch block around a
    `import X`. It avoids third party libraries breaking assumptions of some of
    our tests, e.g., setting multiprocessing start method when imported
    (see librosa/#747, torchvision/#544).
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_utils._check_module_exists', '_check_module_exists(name)', {'PY3': PY3, 'PY34': PY34, 'name': name}, 1)
TEST_NUMPY = _check_module_exists('numpy')
TEST_SCIPY = _check_module_exists('scipy')
TEST_MKL = torch.backends.mkl.is_available()
TEST_NUMBA = _check_module_exists('numba')
TEST_DILL = (_check_module_exists('dill') and PY3)
TEST_LIBROSA = (_check_module_exists('librosa') and PY3)
NO_MULTIPROCESSING_SPAWN = (os.environ.get('NO_MULTIPROCESSING_SPAWN', '0') == '1' or sys.version_info[0] == 2)
TEST_WITH_ASAN = os.getenv('PYTORCH_TEST_WITH_ASAN', '0') == '1'
TEST_WITH_TSAN = os.getenv('PYTORCH_TEST_WITH_TSAN', '0') == '1'
TEST_WITH_UBSAN = os.getenv('PYTORCH_TEST_WITH_UBSAN', '0') == '1'
TEST_WITH_ROCM = os.getenv('PYTORCH_TEST_WITH_ROCM', '0') == '1'
TEST_WITH_SLOW = os.getenv('PYTORCH_TEST_WITH_SLOW', '0') == '1'
TEST_SKIP_FAST = os.getenv('PYTORCH_TEST_SKIP_FAST', '0') == '1'
if TEST_NUMPY:
    import numpy
ALL_TENSORTYPES = [torch.float, torch.double, torch.half]
if TEST_WITH_ROCM:
    ALL_TENSORTYPES2 = [torch.float, torch.double, torch.half, torch.bfloat16]
else:
    ALL_TENSORTYPES2 = ALL_TENSORTYPES

def skipIfRocm(fn):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_utils.skipIfRocm', 'skipIfRocm(fn)', {'wraps': wraps, 'TEST_WITH_ROCM': TEST_WITH_ROCM, 'unittest': unittest, 'fn': fn}, 1)

def skipIfCompiledWithoutNumpy(fn):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_utils.skipIfCompiledWithoutNumpy', 'skipIfCompiledWithoutNumpy(fn)', {'TEST_NUMPY': TEST_NUMPY, 'torch': torch, 'numpy': numpy, 'wraps': wraps, 'unittest': unittest, 'fn': fn}, 1)

def _test_function(fn, device):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_utils._test_function', '_test_function(fn, device)', {'fn': fn, 'device': device}, 1)

def skipIfNoLapack(fn):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_utils.skipIfNoLapack', 'skipIfNoLapack(fn)', {'wraps': wraps, 'torch': torch, 'unittest': unittest, 'fn': fn}, 1)

def skipIfNotRegistered(op_name, message):
    """Wraps the decorator to hide the import of the `core`.

    Args:
        op_name: Check if this op is registered in `core._REGISTERED_OPERATORS`.
        message: message to fail with.

    Usage:
        @skipIfNotRegistered('MyOp', 'MyOp is not linked!')
            This will check if 'MyOp' is in the caffe2.python.core
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_utils.skipIfNotRegistered', 'skipIfNotRegistered(op_name, message)', {'unittest': unittest, 'op_name': op_name, 'message': message}, 1)

def slowTest(fn):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_utils.slowTest', 'slowTest(fn)', {'wraps': wraps, 'TEST_WITH_SLOW': TEST_WITH_SLOW, 'unittest': unittest, 'fn': fn}, 1)

def skipCUDAMemoryLeakCheckIf(condition):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_utils.skipCUDAMemoryLeakCheckIf', 'skipCUDAMemoryLeakCheckIf(condition)', {'condition': condition}, 1)

def skipCUDANonDefaultStreamIf(condition):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_utils.skipCUDANonDefaultStreamIf', 'skipCUDANonDefaultStreamIf(condition)', {'condition': condition}, 1)

def suppress_warnings(fn):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_utils.suppress_warnings', 'suppress_warnings(fn)', {'wraps': wraps, 'warnings': warnings, 'fn': fn}, 1)

def get_cpu_type(type_name):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_utils.get_cpu_type', 'get_cpu_type(type_name)', {'torch': torch, 'type_name': type_name}, 1)

def get_gpu_type(type_name):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_utils.get_gpu_type', 'get_gpu_type(type_name)', {'torch': torch, 'type_name': type_name}, 1)

def to_gpu(obj, type_map=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_utils.to_gpu', 'to_gpu(obj, type_map=None)', {'torch': torch, 'get_gpu_type': get_gpu_type, 'to_gpu': to_gpu, 'deepcopy': deepcopy, 'obj': obj, 'type_map': type_map}, 1)

def get_function_arglist(func):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_utils.get_function_arglist', 'get_function_arglist(func)', {'sys': sys, 'inspect': inspect, 'func': func}, 1)

def set_rng_seed(seed):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.testing._internal.common_utils.set_rng_seed', 'set_rng_seed(seed)', {'torch': torch, 'random': random, 'TEST_NUMPY': TEST_NUMPY, 'numpy': numpy, 'seed': seed}, 0)

@contextlib.contextmanager
def freeze_rng_state():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.testing._internal.common_utils.freeze_rng_state', 'freeze_rng_state()', {'torch': torch, 'contextlib': contextlib}, 0)

def iter_indices(tensor):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_utils.iter_indices', 'iter_indices(tensor)', {'product': product, 'tensor': tensor}, 1)

def is_iterable(obj):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_utils.is_iterable', 'is_iterable(obj)', {'obj': obj}, 1)


class CudaNonDefaultStream:
    
    def __enter__(self):
        beforeDevice = torch.cuda.current_device()
        self.beforeStreams = []
        for d in range(torch.cuda.device_count()):
            self.beforeStreams.append(torch.cuda.current_stream(d))
            deviceStream = torch.cuda.Stream(device=d)
            torch._C._cuda_setStream(deviceStream._cdata)
        torch._C._cuda_setDevice(beforeDevice)
    
    def __exit__(self, exec_type, exec_value, traceback):
        beforeDevice = torch.cuda.current_device()
        for d in range(torch.cuda.device_count()):
            torch._C._cuda_setStream(self.beforeStreams[d]._cdata)
        torch._C._cuda_setDevice(beforeDevice)



class CudaMemoryLeakCheck:
    
    def __init__(self, testcase, name=None):
        self.name = (testcase.id() if name is None else name)
        self.testcase = testcase
        from torch.testing._internal.common_cuda import initialize_cuda_context_rng
        initialize_cuda_context_rng()
    
    @staticmethod
    def get_cuda_memory_usage():
        num_devices = torch.cuda.device_count()
        gc.collect()
        return tuple((torch.cuda.memory_allocated(i) for i in range(num_devices)))
    
    def __enter__(self):
        self.befores = self.get_cuda_memory_usage()
    
    def __exit__(self, exec_type, exec_value, traceback):
        if exec_type is not None:
            return
        afters = self.get_cuda_memory_usage()
        for (i, (before, after)) in enumerate(zip(self.befores, afters)):
            if not TEST_WITH_ROCM:
                self.testcase.assertEqual(before, after, '{} leaked {} bytes CUDA memory on device {}'.format(self.name, after - before, i))
            elif before != after:
                warnings.warn('{} leaked {} bytes ROCm memory on device {}'.format(self.name, after - before, i), RuntimeWarning)

try:
    import hypothesis
    if hypothesis.version.__version_info__ >= (3, 56, 0):
        hypothesis.settings.register_profile('pytorch_ci', hypothesis.settings(derandomize=True, suppress_health_check=[hypothesis.HealthCheck.too_slow], database=None, max_examples=100, verbosity=hypothesis.Verbosity.normal))
        hypothesis.settings.register_profile('dev', hypothesis.settings(suppress_health_check=[hypothesis.HealthCheck.too_slow], database=None, max_examples=10, verbosity=hypothesis.Verbosity.normal))
        hypothesis.settings.register_profile('debug', hypothesis.settings(suppress_health_check=[hypothesis.HealthCheck.too_slow], database=None, max_examples=1000, verbosity=hypothesis.Verbosity.verbose))
    else:
        hypothesis.settings.register_profile('pytorch_ci', hypothesis.settings(derandomize=True, suppress_health_check=[hypothesis.HealthCheck.too_slow], database=None, max_examples=100, min_satisfying_examples=1, verbosity=hypothesis.Verbosity.normal))
        hypothesis.settings.register_profile('dev', hypothesis.settings(suppress_health_check=[hypothesis.HealthCheck.too_slow], database=None, max_examples=10, min_satisfying_examples=1, verbosity=hypothesis.Verbosity.normal))
        hypothesis.settings.register_profile('debug', hypothesis.settings(suppress_health_check=[hypothesis.HealthCheck.too_slow], database=None, max_examples=1000, min_satisfying_examples=1, verbosity=hypothesis.Verbosity.verbose))
    hypothesis.settings.load_profile(('pytorch_ci' if IS_PYTORCH_CI else os.getenv('PYTORCH_HYPOTHESIS_PROFILE', 'dev')))
except ImportError:
    print('Fail to import hypothesis in common_utils, tests are not derandomized')
disabled_test_from_issues = None

def check_disabled(test_name):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.testing._internal.common_utils.check_disabled', 'check_disabled(test_name)', {'urlopen': urlopen, 'json': json, 'IS_SANDCASTLE': IS_SANDCASTLE, 'os': os, 'unittest': unittest, 'test_name': test_name}, 0)


class TestCase(expecttest.TestCase):
    precision = 1e-05
    maxDiff = None
    _do_cuda_memory_leak_check = False
    _do_cuda_non_default_stream = False
    exact_dtype = False
    
    def __init__(self, method_name='runTest'):
        super(TestCase, self).__init__(method_name)
        test_method = getattr(self, method_name)
        self._do_cuda_memory_leak_check &= getattr(test_method, '_do_cuda_memory_leak_check', True)
        if (self._do_cuda_memory_leak_check and not IS_WINDOWS):
            self.wrap_with_cuda_policy(method_name, self.assertLeaksNoCudaTensors)
        self._do_cuda_non_default_stream &= getattr(test_method, '_do_cuda_non_default_stream', True)
        if (self._do_cuda_non_default_stream and not IS_WINDOWS and not TEST_WITH_ROCM):
            self.wrap_with_cuda_policy(method_name, self.enforceNonDefaultStream)
    
    def assertLeaksNoCudaTensors(self, name=None):
        name = (self.id() if name is None else name)
        return CudaMemoryLeakCheck(self, name)
    
    def enforceNonDefaultStream(self):
        return CudaNonDefaultStream()
    
    def wrap_with_cuda_policy(self, method_name, policy):
        test_method = getattr(self, method_name)
        from torch.testing._internal.common_cuda import TEST_CUDA
        fullname = self.id().lower()
        if (TEST_CUDA and (('gpu' in fullname or 'cuda' in fullname))):
            setattr(self, method_name, self.wrap_method_with_cuda_policy(test_method, policy))
    
    def wrap_method_with_cuda_policy(self, method, policy):
        
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            with policy():
                method(*args, **kwargs)
        return types.MethodType(wrapper, self)
    
    def wrap_with_cuda_memory_check(self, method):
        return self.wrap_method_with_cuda_policy(method, self.assertLeaksNoCudaTensors)
    
    def setUp(self):
        if TEST_SKIP_FAST:
            if not getattr(self, self._testMethodName).__dict__.get('slow_test', False):
                raise unittest.SkipTest('test is fast; we disabled it with PYTORCH_TEST_SKIP_FAST')
        check_disabled(str(self))
        set_rng_seed(SEED)
    
    def assertTensorsSlowEqual(self, x, y, prec=None, message=''):
        max_err = 0
        self.assertEqual(x.size(), y.size())
        for index in iter_indices(x):
            max_err = max(max_err, abs(x[index] - y[index]))
        self.assertLessEqual(max_err, prec, message)
    
    def genSparseTensor(self, size, sparse_dim, nnz, is_uncoalesced, device='cpu'):
        assert (all((size[d] > 0 for d in range(sparse_dim))) or nnz == 0), 'invalid arguments'
        v_size = [nnz] + list(size[sparse_dim:])
        v = torch.randn(*v_size, device=device)
        i = torch.rand(sparse_dim, nnz, device=device)
        i.mul_(torch.tensor(size[:sparse_dim]).unsqueeze(1).to(i))
        i = i.to(torch.long)
        if is_uncoalesced:
            v = torch.cat([v, torch.randn_like(v)], 0)
            i = torch.cat([i, i], 1)
        x = torch.sparse_coo_tensor(i, v, torch.Size(size))
        if not is_uncoalesced:
            x = x.coalesce()
        else:
            x = x.detach().clone()
        return (x, x._indices().clone(), x._values().clone())
    
    def safeToDense(self, t):
        r = self.safeCoalesce(t)
        return r.to_dense()
    
    def safeCoalesce(self, t):
        tc = t.coalesce()
        self.assertEqual(tc.to_dense(), t.to_dense())
        self.assertTrue(tc.is_coalesced())
        if t._nnz() == 0:
            self.assertEqual(t._indices(), tc._indices())
            self.assertEqual(t._values(), tc._values())
            return tc
        value_map = {}
        for (idx, val) in zip(t._indices().t(), t._values()):
            idx_tup = tuple(idx.tolist())
            if idx_tup in value_map:
                value_map[idx_tup] += val
            else:
                value_map[idx_tup] = (val.clone() if isinstance(val, torch.Tensor) else val)
        new_indices = sorted(list(value_map.keys()))
        new_values = [value_map[idx] for idx in new_indices]
        if t._values().ndimension() < 2:
            new_values = t._values().new(new_values)
        else:
            new_values = torch.stack(new_values)
        new_indices = t._indices().new(new_indices).t()
        tg = t.new(new_indices, new_values, t.size())
        self.assertEqual(tc._indices(), tg._indices())
        self.assertEqual(tc._values(), tg._values())
        if t.is_coalesced():
            self.assertEqual(tc._indices(), t._indices())
            self.assertEqual(tc._values(), t._values())
        return tg
    
    def assertEqual(self, x, y, prec=None, message='', allow_inf=False, exact_dtype=None):
        if exact_dtype is None:
            exact_dtype = self.exact_dtype
        if (isinstance(prec, str) and message == ''):
            message = prec
            prec = None
        if prec is None:
            prec = self.precision
        if (isinstance(x, torch.Tensor) and isinstance(y, Number)):
            self.assertEqual(x.item(), y, prec=prec, message=message, allow_inf=allow_inf, exact_dtype=exact_dtype)
        elif (isinstance(y, torch.Tensor) and isinstance(x, Number)):
            self.assertEqual(x, y.item(), prec=prec, message=message, allow_inf=allow_inf, exact_dtype=exact_dtype)
        elif (isinstance(x, torch.Tensor) and isinstance(y, numpy.bool_)):
            self.assertEqual(x.item(), y, prec=prec, message=message, allow_inf=allow_inf, exact_dtype=exact_dtype)
        elif (isinstance(y, torch.Tensor) and isinstance(x, numpy.bool_)):
            self.assertEqual(x, y.item(), prec=prec, message=message, allow_inf=allow_inf, exact_dtype=exact_dtype)
        elif (isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)):
            
            def assertTensorsEqual(a, b):
                super(TestCase, self).assertEqual(a.size(), b.size(), message)
                if exact_dtype:
                    self.assertEqual(a.dtype, b.dtype)
                if a.numel() > 0:
                    if (a.device.type == 'cpu' and ((a.dtype == torch.float16 or a.dtype == torch.bfloat16))):
                        a = a.to(torch.float32)
                    if (a.device.type == 'cuda' and a.dtype == torch.bfloat16):
                        a = a.to(torch.float32)
                    b = b.to(a)
                    if a.dtype == torch.bool != (b.dtype == torch.bool):
                        raise TypeError('Was expecting both tensors to be bool type.')
                    else:
                        if (a.dtype == torch.bool and b.dtype == torch.bool):
                            a = a.to(torch.int)
                            b = b.to(torch.int)
                        diff = a - b
                        if (a.dtype.is_complex or a.dtype.is_floating_point):
                            nan_mask = torch.isnan(a)
                            self.assertTrue(torch.equal(nan_mask, torch.isnan(b)), message)
                            diff[nan_mask] = 0
                            if allow_inf:
                                inf_mask = torch.isinf(a)
                                inf_sign = inf_mask.sign()
                                self.assertTrue(torch.equal(inf_sign, torch.isinf(b).sign()), message)
                                diff[inf_mask] = 0
                        if (diff.is_signed() and diff.dtype != torch.int8):
                            diff = diff.abs()
                            if diff.dtype == torch.complex64:
                                diff = diff.to(torch.float)
                            elif diff.dtype == torch.complex128:
                                diff = diff.to(torch.double)
                        max_err = diff.max()
                        self.assertLessEqual(max_err, prec, message)
            super(TestCase, self).assertEqual(x.is_sparse, y.is_sparse, message)
            super(TestCase, self).assertEqual(x.is_quantized, y.is_quantized, message)
            if x.is_sparse:
                x = self.safeCoalesce(x)
                y = self.safeCoalesce(y)
                assertTensorsEqual(x._indices(), y._indices())
                assertTensorsEqual(x._values(), y._values())
            elif (x.is_quantized and y.is_quantized):
                self.assertEqual(x.qscheme(), y.qscheme(), prec=prec, message=message, allow_inf=allow_inf, exact_dtype=exact_dtype)
                if x.qscheme() == torch.per_tensor_affine:
                    self.assertEqual(x.q_scale(), y.q_scale(), prec=prec, message=message, allow_inf=allow_inf, exact_dtype=exact_dtype)
                    self.assertEqual(x.q_zero_point(), y.q_zero_point(), prec=prec, message=message, allow_inf=allow_inf, exact_dtype=exact_dtype)
                elif x.qscheme() == torch.per_channel_affine:
                    self.assertEqual(x.q_per_channel_scales(), y.q_per_channel_scales(), prec=prec, message=message, allow_inf=allow_inf, exact_dtype=exact_dtype)
                    self.assertEqual(x.q_per_channel_zero_points(), y.q_per_channel_zero_points(), prec=prec, message=message, allow_inf=allow_inf, exact_dtype=exact_dtype)
                    self.assertEqual(x.q_per_channel_axis(), y.q_per_channel_axis(), prec=prec, message=message)
                self.assertEqual(x.dtype, y.dtype)
                self.assertEqual(x.int_repr().to(torch.int32), y.int_repr().to(torch.int32), prec=prec, message=message, allow_inf=allow_inf, exact_dtype=exact_dtype)
            else:
                assertTensorsEqual(x, y)
        elif (isinstance(x, string_classes) and isinstance(y, string_classes)):
            super(TestCase, self).assertEqual(x, y, message)
        elif (type(x) == set and type(y) == set):
            super(TestCase, self).assertEqual(x, y, message)
        elif (isinstance(x, dict) and isinstance(y, dict)):
            if (isinstance(x, OrderedDict) and isinstance(y, OrderedDict)):
                self.assertEqual(x.items(), y.items(), prec=prec, message=message, allow_inf=allow_inf, exact_dtype=exact_dtype)
            else:
                self.assertEqual(set(x.keys()), set(y.keys()), prec=prec, message=message, allow_inf=allow_inf, exact_dtype=exact_dtype)
                key_list = list(x.keys())
                self.assertEqual([x[k] for k in key_list], [y[k] for k in key_list], prec=prec, message=message, allow_inf=allow_inf, exact_dtype=exact_dtype)
        elif (is_iterable(x) and is_iterable(y)):
            super(TestCase, self).assertEqual(len(x), len(y), message)
            for (x_, y_) in zip(x, y):
                self.assertEqual(x_, y_, prec=prec, message=message, allow_inf=allow_inf, exact_dtype=exact_dtype)
        elif (isinstance(x, bool) and isinstance(y, bool)):
            super(TestCase, self).assertEqual(x, y, message)
        elif (isinstance(x, Number) and isinstance(y, Number)):
            if (abs(x) == inf or abs(y) == inf):
                if allow_inf:
                    super(TestCase, self).assertEqual(x, y, message)
                else:
                    self.fail('Expected finite numeric values - x={}, y={}'.format(x, y))
                return
            super(TestCase, self).assertLessEqual(abs(x - y), prec, message)
        else:
            super(TestCase, self).assertEqual(x, y, message)
    
    def assertAlmostEqual(self, x, y, places=None, msg=None, delta=None, allow_inf=None):
        prec = delta
        if places:
            prec = 10**(-places)
        self.assertEqual(x, y, prec, msg, allow_inf)
    
    def assertNotEqual(self, x, y, prec=None, message=''):
        if (isinstance(prec, str) and message == ''):
            message = prec
            prec = None
        if prec is None:
            prec = self.precision
        if (isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)):
            if x.size() != y.size():
                super(TestCase, self).assertNotEqual(x.size(), y.size())
            self.assertGreater(x.numel(), 0)
            y = y.type_as(x)
            y = (y.cuda(device=x.get_device()) if x.is_cuda else y.cpu())
            nan_mask = x != x
            if torch.equal(nan_mask, y != y):
                diff = x - y
                if diff.is_signed():
                    diff = diff.abs()
                diff[nan_mask] = 0
                max_err = diff.max().item()
                self.assertGreaterEqual(max_err, prec, message)
        elif (type(x) == str and type(y) == str):
            super(TestCase, self).assertNotEqual(x, y)
        elif (is_iterable(x) and is_iterable(y)):
            super(TestCase, self).assertNotEqual(x, y)
        else:
            try:
                self.assertGreaterEqual(abs(x - y), prec, message)
                return
            except (TypeError, AssertionError):
                pass
            super(TestCase, self).assertNotEqual(x, y, message)
    
    def assertObjectIn(self, obj, iterable):
        for elem in iterable:
            if id(obj) == id(elem):
                return
        raise AssertionError('object not found in iterable')
    
    def assertExpectedRaises(self, exc_type, callable, *args, **kwargs):
        subname = None
        if 'subname' in kwargs:
            subname = kwargs['subname']
            del kwargs['subname']
        try:
            callable(*args, **kwargs)
        except exc_type as e:
            self.assertExpected(str(e), subname)
            return
        self.fail(msg='Did not raise when expected to')
    
    def assertNotWarn(self, callable, msg=''):
        """
        Test if :attr:`callable` does not raise a warning.
        """
        with self._reset_warning_registry(), warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            callable()
            self.assertTrue(len(ws) == 0, msg)
    
    def assertWarns(self, callable, msg=''):
        """
        Test if :attr:`callable` raises a warning.
        """
        with self._reset_warning_registry(), warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            callable()
            self.assertTrue(len(ws) > 0, msg)
    
    def assertWarnsRegex(self, callable, regex, msg=''):
        """
        Test if :attr:`callable` raises any warning with message that contains
        the regex pattern :attr:`regex`.
        """
        with self._reset_warning_registry(), warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            callable()
            self.assertTrue(len(ws) > 0, msg)
            found = any((re.search(regex, str(w.message)) is not None for w in ws))
            self.assertTrue(found, msg)
    
    @contextmanager
    def maybeWarnsRegex(self, category, regex=''):
        """Context manager for code that *may* warn, e.g. ``TORCH_WARN_ONCE``.

        This filters expected warnings from the test log and fails the test if
        any unexpected warnings are caught.
        """
        with self._reset_warning_registry(), warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            warnings.filterwarnings('ignore', message=regex, category=category)
            try:
                yield
            finally:
                if len(ws) != 0:
                    msg = 'Caught unexpected warnings:\n'
                    for w in ws:
                        msg += warnings.formatwarning(w.message, w.category, w.filename, w.lineno, w.line)
                        msg += '\n'
                    self.fail(msg)
    
    @contextmanager
    def _reset_warning_registry(self):
        """
        warnings.catch_warnings() in Python 2 misses already registered
        warnings. We need to manually clear the existing warning registries to
        ensure catching warnings in a scope.
        """
        if sys.version_info >= (3, ):
            yield
            return
        backup = {}
        for (name, mod) in list(sys.modules.items()):
            try:
                reg = mod.__warningregistry__
            except AttributeError:
                continue
            else:
                backup[name] = reg.copy()
                reg.clear()
        yield
        for (name, reg_orig) in backup.items():
            try:
                mod = sys.modules[name]
            except KeyError:
                continue
            try:
                reg = mod.__warningregistry__
            except AttributeError:
                mod.__warningregistry__ = reg_orig
            else:
                reg.clear()
                reg.update(reg_orig)
    
    def assertExpected(self, s, subname=None):
        """
        Test that a string matches the recorded contents of a file
        derived from the name of this test and subname.  This file
        is placed in the 'expect' directory in the same directory
        as the test script. You can automatically update the recorded test
        output using --accept.

        If you call this multiple times in a single function, you must
        give a unique subname each time.
        """
        if not ((isinstance(s, str) or (sys.version_info[0] == 2 and isinstance(s, unicode)))):
            raise TypeError('assertExpected is strings only')
        
        def remove_prefix(text, prefix):
            if text.startswith(prefix):
                return text[len(prefix):]
            return text
        module_id = self.__class__.__module__
        munged_id = remove_prefix(self.id(), module_id + '.')
        test_file = os.path.realpath(sys.modules[module_id].__file__)
        expected_file = os.path.join(os.path.dirname(test_file), 'expect', munged_id)
        subname_output = ''
        if subname:
            expected_file += '-' + subname
            subname_output = ' ({})'.format(subname)
        expected_file += '.expect'
        expected = None
        
        def accept_output(update_type):
            print('Accepting {} for {}{}:\n\n{}'.format(update_type, munged_id, subname_output, s))
            with open(expected_file, 'w') as f:
                f.write(s)
        try:
            with open(expected_file) as f:
                expected = f.read()
        except IOError as e:
            if e.errno != errno.ENOENT:
                raise
            elif expecttest.ACCEPT:
                return accept_output('output')
            else:
                raise RuntimeError('I got this output for {}{}:\n\n{}\n\nNo expect file exists; to accept the current output, run:\npython {} {} --accept'.format(munged_id, subname_output, s, __main__.__file__, munged_id))
        if IS_WINDOWS:
            expected = re.sub('CppOp\\[(.+?)\\]', 'CppOp[]', expected)
            s = re.sub('CppOp\\[(.+?)\\]', 'CppOp[]', s)
        if expecttest.ACCEPT:
            if expected != s:
                return accept_output('updated output')
        elif hasattr(self, 'assertMultiLineEqual'):
            self.assertMultiLineEqual(expected, s)
        else:
            self.assertEqual(s, expected)
    
    def assertExpectedStripMangled(self, s, subname=None):
        s = re.sub('__torch__[^ ]+', '', s)
        self.assertExpected(s, subname)
    
    @staticmethod
    def runWithPytorchAPIUsageStderr(code):
        import subprocess
        env = os.environ.copy()
        env['PYTORCH_API_USAGE_STDERR'] = '1'
        pipes = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        return pipes.communicate()[1].decode('ascii')
    if sys.version_info < (3, 2):
        assertRegex = unittest.TestCase.assertRegexpMatches
        assertRaisesRegex = unittest.TestCase.assertRaisesRegexp
    if sys.version_info < (3, 5):
        assertNotRegex = unittest.TestCase.assertNotRegexpMatches


def download_file(url, binary=True):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_utils.download_file', 'download_file(url, binary=True)', {'sys': sys, 'os': os, 'get_writable_path': get_writable_path, '__file__': __file__, 'warnings': warnings, 'unittest': unittest, 'url': url, 'binary': binary}, 1)

def find_free_port():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_utils.find_free_port', 'find_free_port()', {'socket': socket}, 1)
ADDRESS_IN_USE = 'Address already in use'
CONNECT_TIMEOUT = 'connect() timed out.'

def retry_on_connect_failures(func=None, connect_errors=ADDRESS_IN_USE):
    """Reruns a test if the test returns a RuntimeError and the exception
    matches exactly with one of the strings in connect_errors."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_utils.retry_on_connect_failures', 'retry_on_connect_failures(func=None, connect_errors=ADDRESS_IN_USE)', {'partial': partial, 'retry_on_connect_failures': retry_on_connect_failures, 'wraps': wraps, 'time': time, 'random': random, 'func': func, 'connect_errors': connect_errors, 'ADDRESS_IN_USE': ADDRESS_IN_USE}, 1)

def retry(ExceptionToCheck, tries=3, delay=3):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_utils.retry', 'retry(ExceptionToCheck, tries=3, delay=3)', {'wraps': wraps, 'time': time, 'ExceptionToCheck': ExceptionToCheck, 'tries': tries, 'delay': delay}, 1)

def prod_single_zero(dim_size):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_utils.prod_single_zero', 'prod_single_zero(dim_size)', {'torch': torch, 'dim_size': dim_size}, 1)

def random_square_matrix_of_rank(l, rank, dtype=torch.double, device='cpu'):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_utils.random_square_matrix_of_rank', "random_square_matrix_of_rank(l, rank, dtype=torch.double, device='cpu')", {'torch': torch, 'l': l, 'rank': rank, 'dtype': dtype, 'device': device}, 1)

def random_symmetric_matrix(l, *batches, **kwargs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_utils.random_symmetric_matrix', 'random_symmetric_matrix(l, *batches, **kwargs)', {'torch': torch, 'l': l, 'batches': batches, 'kwargs': kwargs}, 1)

def random_symmetric_psd_matrix(l, *batches, **kwargs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_utils.random_symmetric_psd_matrix', 'random_symmetric_psd_matrix(l, *batches, **kwargs)', {'torch': torch, 'l': l, 'batches': batches, 'kwargs': kwargs}, 1)

def random_symmetric_pd_matrix(matrix_size, *batch_dims, **kwargs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_utils.random_symmetric_pd_matrix', 'random_symmetric_pd_matrix(matrix_size, *batch_dims, **kwargs)', {'torch': torch, 'matrix_size': matrix_size, 'batch_dims': batch_dims, 'kwargs': kwargs}, 1)

def make_nonzero_det(A, sign=None, min_singular_value=0.1):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_utils.make_nonzero_det', 'make_nonzero_det(A, sign=None, min_singular_value=0.1)', {'torch': torch, 'A': A, 'sign': sign, 'min_singular_value': min_singular_value}, 1)

def random_fullrank_matrix_distinct_singular_value(matrix_size, *batch_dims, **kwargs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_utils.random_fullrank_matrix_distinct_singular_value', 'random_fullrank_matrix_distinct_singular_value(matrix_size, *batch_dims, **kwargs)', {'torch': torch, 'matrix_size': matrix_size, 'batch_dims': batch_dims, 'kwargs': kwargs}, 1)

def random_matrix(rows, columns, *batch_dims, **kwargs):
    """Return rectangular matrix or batches of rectangular matrices.

    Parameters:
      dtype - the data type
      device - the device kind
      singular - when True, the output will be singular
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_utils.random_matrix', 'random_matrix(rows, columns, *batch_dims, **kwargs)', {'torch': torch, 'rows': rows, 'columns': columns, 'batch_dims': batch_dims, 'kwargs': kwargs}, 1)

def random_lowrank_matrix(rank, rows, columns, *batch_dims, **kwargs):
    """Return rectangular matrix or batches of rectangular matrices with
    given rank.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_utils.random_lowrank_matrix', 'random_lowrank_matrix(rank, rows, columns, *batch_dims, **kwargs)', {'random_matrix': random_matrix, 'rank': rank, 'rows': rows, 'columns': columns, 'batch_dims': batch_dims, 'kwargs': kwargs}, 1)

def random_sparse_matrix(rows, columns, density=0.01, **kwargs):
    """Return rectangular random sparse matrix within given density.

    The density of the result approaches to given density as the size
    of the matrix is increased and a relatively small value of density
    is specified but higher than min(rows, columns)/(rows * columns)
    for non-singular matrices.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_utils.random_sparse_matrix', 'random_sparse_matrix(rows, columns, density=0.01, **kwargs)', {'torch': torch, 'random': random, 'rows': rows, 'columns': columns, 'density': density, 'kwargs': kwargs}, 1)

def random_sparse_pd_matrix(matrix_size, density=0.01, **kwargs):
    """Return random sparse positive-definite matrix with given density.

    The eigenvalues of the matrix are defined as::
      arange(1, matrix_size+1)/matrix_size

    Algorithm:
      A = diag(arange(1, matrix_size+1)/matrix_size)
      while <A density is smaller than required>:
          <choose random i, j in range(matrix_size), theta in [0, 2*pi]>
          R = <rotation matrix (i,j,theta)>
          A = R^T A R
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_utils.random_sparse_pd_matrix', 'random_sparse_pd_matrix(matrix_size, density=0.01, **kwargs)', {'random': random, 'matrix_size': matrix_size, 'density': density, 'kwargs': kwargs}, 1)

def do_test_dtypes(self, dtypes, layout, device):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.testing._internal.common_utils.do_test_dtypes', 'do_test_dtypes(self, dtypes, layout, device)', {'torch': torch, 'self': self, 'dtypes': dtypes, 'layout': layout, 'device': device}, 0)

def do_test_empty_full(self, dtypes, layout, device):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_utils.do_test_empty_full', 'do_test_empty_full(self, dtypes, layout, device)', {'torch': torch, 'operator': operator, 'self': self, 'dtypes': dtypes, 'layout': layout, 'device': device}, 1)
THESE_TAKE_WAY_TOO_LONG = {'test_Conv3d_groups', 'test_conv_double_backward', 'test_conv_double_backward_groups', 'test_Conv3d_dilated', 'test_Conv3d_stride_padding', 'test_Conv3d_dilated_strided', 'test_Conv3d', 'test_Conv2d_dilated', 'test_ConvTranspose3d_dilated', 'test_ConvTranspose2d_dilated', 'test_snli', 'test_Conv2d', 'test_Conv2d_padding', 'test_ConvTranspose2d_no_bias', 'test_ConvTranspose2d', 'test_ConvTranspose3d', 'test_Conv2d_no_bias', 'test_matmul_4d_4d', 'test_multinomial_invalid_probs'}
running_script_path = None

def set_running_script_path():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.testing._internal.common_utils.set_running_script_path', 'set_running_script_path()', {'os': os, 'sys': sys}, 0)

def check_test_defined_in_running_script(test_case):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_utils.check_test_defined_in_running_script', 'check_test_defined_in_running_script(test_case)', {'running_script_path': running_script_path, 'os': os, 'inspect': inspect, 'test_case': test_case}, 1)

def load_tests(loader, tests, pattern):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_utils.load_tests', 'load_tests(loader, tests, pattern)', {'set_running_script_path': set_running_script_path, 'unittest': unittest, 'check_test_defined_in_running_script': check_test_defined_in_running_script, 'loader': loader, 'tests': tests, 'pattern': pattern}, 1)


class BytesIOContext(io.BytesIO):
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        pass


def _assertGradAndGradgradChecks(test_case, apply_fn, inputs):
    test_case.assertTrue(gradcheck(apply_fn, inputs))
    test_case.assertTrue(gradgradcheck(apply_fn, inputs))
dtype2prec_DONTUSE = {torch.float: 1e-05, torch.double: 1e-05, torch.half: 0.01, torch.bfloat16: 0.1}

