from __future__ import print_function, division, absolute_import
import os
import warnings
import pytest
from random import random
from uuid import uuid4
from time import sleep
from .. import Parallel, delayed, parallel_config
from ..parallel import ThreadingBackend, AutoBatchingMixin
from .._dask import DaskDistributedBackend
distributed = pytest.importorskip('distributed')
dask = pytest.importorskip('dask')
from distributed import Client, LocalCluster, get_client
from distributed.metrics import time
from distributed.utils_test import cluster, inc

def noop(*args, **kwargs):
    pass

def slow_raise_value_error(condition, duration=0.05):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_dask.slow_raise_value_error', 'slow_raise_value_error(condition, duration=0.05)', {'sleep': sleep, 'condition': condition, 'duration': duration}, 0)

def count_events(event_name, client):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.test.test_dask.count_events', 'count_events(event_name, client)', {'event_name': event_name, 'client': client}, 1)

def test_simple(loop):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_dask.test_simple', 'test_simple(loop)', {'cluster': cluster, 'Client': Client, 'parallel_config': parallel_config, 'Parallel': Parallel, 'delayed': delayed, 'inc': inc, 'pytest': pytest, 'slow_raise_value_error': slow_raise_value_error, 'loop': loop}, 0)

def test_dask_backend_uses_autobatching(loop):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_dask.test_dask_backend_uses_autobatching', 'test_dask_backend_uses_autobatching(loop)', {'DaskDistributedBackend': DaskDistributedBackend, 'AutoBatchingMixin': AutoBatchingMixin, 'cluster': cluster, 'Client': Client, 'parallel_config': parallel_config, 'Parallel': Parallel, 'delayed': delayed, 'loop': loop}, 0)

def random2():
    return random()

def test_dont_assume_function_purity(loop):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_dask.test_dont_assume_function_purity', 'test_dont_assume_function_purity(loop)', {'cluster': cluster, 'Client': Client, 'parallel_config': parallel_config, 'Parallel': Parallel, 'delayed': delayed, 'random2': random2, 'loop': loop}, 0)

@pytest.mark.parametrize('mixed', [True, False])
def test_dask_funcname(loop, mixed):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.test.test_dask.test_dask_funcname', 'test_dask_funcname(loop, mixed)', {'delayed': delayed, 'inc': inc, 'cluster': cluster, 'Client': Client, 'parallel_config': parallel_config, 'Parallel': Parallel, 'pytest': pytest, 'loop': loop, 'mixed': mixed}, 1)

def test_no_undesired_distributed_cache_hit(loop):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.test.test_dask.test_no_undesired_distributed_cache_hit', 'test_no_undesired_distributed_cache_hit(loop)', {'pytest': pytest, 'uuid4': uuid4, 'LocalCluster': LocalCluster, 'Client': Client, 'parallel_config': parallel_config, 'Parallel': Parallel, 'delayed': delayed, 'count_events': count_events, 'loop': loop}, 1)


class CountSerialized(object):
    
    def __init__(self, x):
        self.x = x
        self.count = 0
    
    def __add__(self, other):
        return self.x + getattr(other, 'x', other)
    __radd__ = __add__
    
    def __reduce__(self):
        self.count += 1
        return (CountSerialized, (self.x, ))


def add5(a, b, c, d=0, e=0):
    return a + b + c + d + e

def test_manual_scatter(loop):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_dask.test_manual_scatter', 'test_manual_scatter(loop)', {'CountSerialized': CountSerialized, 'cluster': cluster, 'Client': Client, 'parallel_config': parallel_config, 'delayed': delayed, 'add5': add5, 'Parallel': Parallel, 'pytest': pytest, 'loop': loop}, 0)

def test_auto_scatter(loop_in_thread):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_dask.test_auto_scatter', 'test_auto_scatter(loop_in_thread)', {'pytest': pytest, 'cluster': cluster, 'Client': Client, 'parallel_config': parallel_config, 'Parallel': Parallel, 'delayed': delayed, 'noop': noop, 'count_events': count_events, 'loop_in_thread': loop_in_thread}, 0)

@pytest.mark.parametrize('retry_no', list(range(2)))
def test_nested_scatter(loop, retry_no):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.test.test_dask.test_nested_scatter', 'test_nested_scatter(loop, retry_no)', {'get_client': get_client, 'parallel_config': parallel_config, 'Parallel': Parallel, 'delayed': delayed, 'cluster': cluster, 'Client': Client, 'pytest': pytest, 'list': list, 'range': range, 'loop': loop, 'retry_no': retry_no}, 1)

def test_nested_backend_context_manager(loop_in_thread):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.test.test_dask.test_nested_backend_context_manager', 'test_nested_backend_context_manager(loop_in_thread)', {'Parallel': Parallel, 'delayed': delayed, 'os': os, 'cluster': cluster, 'Client': Client, 'parallel_config': parallel_config, 'loop_in_thread': loop_in_thread}, 1)

def test_nested_backend_context_manager_implicit_n_jobs(loop):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.test.test_dask.test_nested_backend_context_manager_implicit_n_jobs', 'test_nested_backend_context_manager_implicit_n_jobs(loop)', {'Parallel': Parallel, 'cluster': cluster, 'Client': Client, 'parallel_config': parallel_config, 'delayed': delayed, 'loop': loop}, 1)

def test_errors(loop):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_dask.test_errors', 'test_errors(loop)', {'pytest': pytest, 'parallel_config': parallel_config, 'loop': loop}, 0)

def test_correct_nested_backend(loop):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_dask.test_correct_nested_backend', 'test_correct_nested_backend(loop)', {'cluster': cluster, 'Client': Client, 'parallel_config': parallel_config, 'Parallel': Parallel, 'delayed': delayed, 'outer': outer, 'DaskDistributedBackend': DaskDistributedBackend, 'ThreadingBackend': ThreadingBackend, 'loop': loop}, 0)

def outer(nested_require):
    return Parallel(n_jobs=2, prefer='threads')((delayed(middle)(nested_require) for _ in range(1)))

def middle(require):
    return Parallel(n_jobs=2, require=require)((delayed(inner)() for _ in range(1)))

def inner():
    return Parallel()._backend

def test_secede_with_no_processes(loop):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_dask.test_secede_with_no_processes', 'test_secede_with_no_processes(loop)', {'Client': Client, 'parallel_config': parallel_config, 'Parallel': Parallel, 'delayed': delayed, 'loop': loop}, 0)

def _worker_address(_):
    from distributed import get_worker
    return get_worker().address

def test_dask_backend_keywords(loop):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_dask.test_dask_backend_keywords', 'test_dask_backend_keywords(loop)', {'cluster': cluster, 'Client': Client, 'parallel_config': parallel_config, 'Parallel': Parallel, 'delayed': delayed, '_worker_address': _worker_address, 'loop': loop}, 0)

def test_cleanup(loop):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_dask.test_cleanup', 'test_cleanup(loop)', {'Client': Client, 'parallel_config': parallel_config, 'Parallel': Parallel, 'delayed': delayed, 'inc': inc, 'time': time, 'sleep': sleep, 'loop': loop}, 0)

@pytest.mark.parametrize('cluster_strategy', ['adaptive', 'late_scaling'])
@pytest.mark.skipif((distributed.__version__ <= '2.1.1' and distributed.__version__ >= '1.28.0'), reason='distributed bug - https://github.com/dask/distributed/pull/2841')
def test_wait_for_workers(cluster_strategy):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_dask.test_wait_for_workers', 'test_wait_for_workers(cluster_strategy)', {'LocalCluster': LocalCluster, 'Client': Client, 'parallel_config': parallel_config, 'Parallel': Parallel, 'delayed': delayed, 'inc': inc, 'pytest': pytest, 'distributed': distributed, 'cluster_strategy': cluster_strategy}, 0)

def test_wait_for_workers_timeout():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_dask.test_wait_for_workers_timeout', 'test_wait_for_workers_timeout()', {'LocalCluster': LocalCluster, 'Client': Client, 'parallel_config': parallel_config, 'pytest': pytest, 'Parallel': Parallel, 'delayed': delayed, 'inc': inc}, 0)

@pytest.mark.parametrize('backend', ['loky', 'multiprocessing'])
def test_joblib_warning_inside_dask_daemonic_worker(backend):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.test.test_dask.test_joblib_warning_inside_dask_daemonic_worker', 'test_joblib_warning_inside_dask_daemonic_worker(backend)', {'LocalCluster': LocalCluster, 'Client': Client, 'warnings': warnings, 'Parallel': Parallel, 'delayed': delayed, 'inc': inc, 'pytest': pytest, 'backend': backend}, 1)

