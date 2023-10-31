from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import tempfile
import time
import unittest
import logging
import six
import traceback
from collections import namedtuple
from functools import wraps
import torch
import torch.distributed as c10d
from functools import partial, reduce
from torch.testing._internal.common_utils import TestCase, TEST_WITH_ROCM
TestSkip = namedtuple('TestSkip', 'exit_code, message')
TEST_SKIPS = {'multi-gpu': TestSkip(75, 'Need at least 2 CUDA devices'), 'nccl': TestSkip(76, 'c10d not compiled with NCCL support'), 'known_issues': TestSkip(77, 'Test skipped due to known issues'), 'skipIfRocm': TestSkip(78, 'Test skipped for ROCm')}

def skip_if_not_multigpu(func):
    """Multi-GPU tests requires at least 2 GPUS. Skip if this is not met."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_distributed.skip_if_not_multigpu', 'skip_if_not_multigpu(func)', {'wraps': wraps, 'torch': torch, 'sys': sys, 'TEST_SKIPS': TEST_SKIPS, 'func': func}, 1)

def skip_if_lt_x_gpu(x):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_distributed.skip_if_lt_x_gpu', 'skip_if_lt_x_gpu(x)', {'wraps': wraps, 'torch': torch, 'sys': sys, 'TEST_SKIPS': TEST_SKIPS, 'x': x}, 1)

def skip_for_known_issues(func):
    """Skips a test due to known issues (for c10d)."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_distributed.skip_for_known_issues', 'skip_for_known_issues(func)', {'wraps': wraps, 'sys': sys, 'TEST_SKIPS': TEST_SKIPS, 'func': func}, 1)

def requires_gloo():
    return unittest.skipUnless(c10d.is_gloo_available(), 'c10d was not compiled with the Gloo backend')

def requires_nccl_version(version, msg):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_distributed.requires_nccl_version', 'requires_nccl_version(version, msg)', {'c10d': c10d, 'unittest': unittest, 'torch': torch, 'version': version, 'msg': msg}, 1)

def requires_nccl():
    return unittest.skipUnless(c10d.is_nccl_available(), 'c10d was not compiled with the NCCL backend')

def requires_mpi():
    return unittest.skipUnless(c10d.is_mpi_available(), 'c10d was not compiled with the MPI backend')

def skip_if_rocm(func):
    """Skips a test for ROCm"""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_distributed.skip_if_rocm', 'skip_if_rocm(func)', {'wraps': wraps, 'TEST_WITH_ROCM': TEST_WITH_ROCM, 'sys': sys, 'TEST_SKIPS': TEST_SKIPS, 'func': func}, 1)
TIMEOUT_DEFAULT = 100
TIMEOUT_OVERRIDE = {}

def get_timeout(test_id):
    return TIMEOUT_OVERRIDE.get(test_id.split('.')[-1], TIMEOUT_DEFAULT)

def simple_sparse_reduce_tests(rank, world_size, num_inputs=1):
    """
    Generate a number of basic test cases for sparse reduction.
    These cover tensors with a varying number of sparse dimensions and a varying
    number of dense dimensions. The only reduction operation we support is sum.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_distributed.simple_sparse_reduce_tests', 'simple_sparse_reduce_tests(rank, world_size, num_inputs=1)', {'torch': torch, 'reduce': reduce, 'partial': partial, 'rank': rank, 'world_size': world_size, 'num_inputs': num_inputs}, 1)


class MultiProcessTestCase(TestCase):
    MAIN_PROCESS_RANK = -1
    TEST_ERROR_EXIT_CODE = 10
    
    @property
    def world_size(self):
        return 4
    
    @staticmethod
    def join_or_run(fn):
        
        @wraps(fn)
        def wrapper(self):
            if self.rank == self.MAIN_PROCESS_RANK:
                self._join_processes(fn)
            else:
                try:
                    fn(self)
                except Exception as e:
                    logging.error('Caught exception: \n{}exiting process with exit code: {}'.format(traceback.format_exc(), MultiProcessTestCase.TEST_ERROR_EXIT_CODE))
                    sys.exit(MultiProcessTestCase.TEST_ERROR_EXIT_CODE)
        return wrapper
    
    @classmethod
    def setUpClass(cls):
        for attr in dir(cls):
            if attr.startswith('test'):
                fn = getattr(cls, attr)
                setattr(cls, attr, cls.join_or_run(fn))
    
    def setUp(self):
        super(MultiProcessTestCase, self).setUp()
        self.skip_return_code_checks = []
        self.rank = self.MAIN_PROCESS_RANK
        self.file_name = tempfile.NamedTemporaryFile(delete=False).name
    
    def tearDown(self):
        super(MultiProcessTestCase, self).tearDown()
        for p in self.processes:
            p.terminate()
    
    def _current_test_name(self):
        return self.id().split('.')[-1]
    
    def _start_processes(self, proc):
        self.processes = []
        for rank in range(int(self.world_size)):
            process = proc(target=self.__class__._run, name='process ' + str(rank), args=(rank, self._current_test_name(), self.file_name))
            process.start()
            self.processes.append(process)
    
    def _fork_processes(self):
        if six.PY3:
            proc = torch.multiprocessing.get_context('fork').Process
        else:
            proc = torch.multiprocessing.Process
        self._start_processes(proc)
    
    def _spawn_processes(self):
        if six.PY3:
            proc = torch.multiprocessing.get_context('spawn').Process
        else:
            raise RuntimeError('Cannot use spawn start method with Python 2')
        self._start_processes(proc)
    
    @classmethod
    def _run(cls, rank, test_name, file_name):
        self = cls(test_name)
        self.rank = rank
        self.file_name = file_name
        getattr(self, test_name)()
        sys.exit(0)
    
    def _join_processes(self, fn):
        timeout = get_timeout(self.id())
        start_time = time.time()
        subprocess_error = False
        while True:
            for p in self.processes:
                if p.exitcode == MultiProcessTestCase.TEST_ERROR_EXIT_CODE:
                    print('Some process exited badly, terminating rest.')
                    active_children = torch.multiprocessing.active_children()
                    for ac in active_children:
                        ac.terminate()
                    subprocess_error = True
                    break
            if subprocess_error:
                break
            if all([p.exitcode is not None for p in self.processes]):
                break
            elapsed = time.time() - start_time
            if elapsed > timeout:
                print('Timing out after {} seconds and killing subprocesses.'.format(timeout))
                for p in self.processes:
                    p.terminate()
                break
            time.sleep(0.1)
        elapsed_time = time.time() - start_time
        if fn in self.skip_return_code_checks:
            self._check_no_test_errors(elapsed_time)
        else:
            self._check_return_codes(elapsed_time)
    
    def _check_no_test_errors(self, elapsed_time):
        """
        Checks that we didn't have any errors thrown in the child processes.
        """
        for (i, p) in enumerate(self.processes):
            if p.exitcode is None:
                raise RuntimeError('Process {} timed out after {} seconds'.format(i, elapsed_time))
            self.assertNotEqual(self.TEST_ERROR_EXIT_CODE, p.exitcode)
    
    def _check_return_codes(self, elapsed_time):
        """
        Checks that the return codes of all spawned processes match, and skips
        tests if they returned a return code indicating a skipping condition.
        """
        first_process = self.processes[0]
        errored_processes = [(i, p) for (i, p) in enumerate(self.processes) if p.exitcode == MultiProcessTestCase.TEST_ERROR_EXIT_CODE]
        if errored_processes:
            error = 'Processes {} exited with error code {}'.format(' '.join([str(i) for (i, _) in errored_processes]), MultiProcessTestCase.TEST_ERROR_EXIT_CODE)
            raise RuntimeError(error)
        for (i, p) in enumerate(self.processes):
            if p.exitcode is None:
                raise RuntimeError('Process {} terminated or timed out after {} seconds'.format(i, elapsed_time))
            self.assertEqual(p.exitcode, first_process.exitcode)
        for skip in TEST_SKIPS.values():
            if first_process.exitcode == skip.exit_code:
                raise unittest.SkipTest(skip.message)
        self.assertEqual(first_process.exitcode, 0)
    
    @property
    def is_master(self):
        return self.rank == 0


