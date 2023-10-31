from __future__ import absolute_import, division, print_function, unicode_literals
import time
from functools import partial, wraps
import re
import torch.distributed as dist
import torch.distributed.rpc as rpc
from torch.distributed.rpc import _rref_context_get_debug_info
if not dist.is_available():
    print('c10d not available, skipping tests')
    sys.exit(0)


class TestConfig:
    __slots__ = ['rpc_backend_name', 'build_rpc_backend_options']
    
    def __init__(self, *args, **kwargs):
        assert len(args) == 0, 'TestConfig only takes kwargs.'
        for (k, v) in kwargs.items():
            setattr(self, k, v)

TEST_CONFIG = TestConfig()
INIT_METHOD_TEMPLATE = 'file://{file_name}'

def dist_init(old_test_method=None, setup_rpc=True, clean_shutdown=True):
    """
    We use this decorator for setting up and tearing down state since
    MultiProcessTestCase runs each `test*` method in a separate process and
    each process just runs the `test*` method without actually calling
    'setUp' and 'tearDown' methods of unittest.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.dist_utils.dist_init', 'dist_init(old_test_method=None, setup_rpc=True, clean_shutdown=True)', {'partial': partial, 'dist_init': dist_init, 'wraps': wraps, 'rpc': rpc, 'old_test_method': old_test_method, 'setup_rpc': setup_rpc, 'clean_shutdown': clean_shutdown}, 1)
TEST_CONFIG.rpc_backend_name = 'PROCESS_GROUP'
TEST_CONFIG.build_rpc_backend_options = lambda test_object: rpc.backend_registry.construct_rpc_backend_options(test_object.rpc_backend, init_method=test_object.init_method, num_send_recv_threads=8)

def noop():
    pass

def wait_until_node_failure(rank, expected_error_regex='.*'):
    """
    Loops until an RPC to the given rank fails. This is used to
    indicate that the node has failed in unit tests.
    Args:
    rank (int): Rank of the node expected to fail
    expected_error_regex (optional, str): Regex of exception message expected. Useful to ensure a specific failure
    occurs, not just any.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.dist_utils.wait_until_node_failure', "wait_until_node_failure(rank, expected_error_regex='.*')", {'rpc': rpc, 'noop': noop, 'time': time, 're': re, 'rank': rank, 'expected_error_regex': expected_error_regex}, 1)

def get_shutdown_error_regex(rpc_backend):
    """
    Return various error message we may see from RPC agents while running tests that check for failures. This function
    is used to match against possible errors to ensure failures were raised properly.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.dist_utils.get_shutdown_error_regex', 'get_shutdown_error_regex(rpc_backend)', {'rpc_backend': rpc_backend}, 1)

def wait_until_pending_users_flushed():
    """
    The RRef protocol holds forkIds of rrefs in a map until those forks are
    confirmed by the owner. The message confirming the fork may arrive after
    our tests check whether this map is empty, which leads to failures and
    flaky tests. to_here also does not guarantee that we have finished
    processind the owner's confirmation message for the RRef. This function
    loops until the map is empty, which means the messages have been received
    as processed. Call this function before asserting the map returned by
    _get_debug_info is empty.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.dist_utils.wait_until_pending_users_flushed', 'wait_until_pending_users_flushed()', {'_rref_context_get_debug_info': _rref_context_get_debug_info, 'time': time}, 1)

def initialize_pg(init_method, rank, world_size):
    if not dist.is_initialized():
        dist.init_process_group(backend='gloo', init_method=init_method, rank=rank, world_size=world_size)

def worker_name(rank):
    return 'worker{}'.format(rank)

