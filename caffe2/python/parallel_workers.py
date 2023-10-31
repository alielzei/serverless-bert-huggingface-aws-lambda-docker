from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
'\nThis module provides a python-land multithreaded mechanism for executing work.\n\nBasic usage is as follows:\n   coordinator = parallel_workers.init_workers(\n      my_worker_fun,\n      worker_name="train"\n   )\n   ...\n   coordinator.start()\n\nFirst argument is the function to run in a loop on potentially multiple threads.\nIt has the call signature\n    worker_fun(worker_id)\n\nArgument \'worker_name\' is used to distinguish different workers,\nsuch as workers processing train data or workers processing test data.\n\nOptionally, one can define an "init function" that is called once before\nthreads start, and has call signature:\n   my_init_fun(worker_coordinator, global_coordinator)\n\nNote that for data_parallel_models, init_workers will be called\nfor each GPU. Note that the \'coordinator\' returned by the function is same\neach time.\n'
import logging
import threading
import atexit
import time
import collections
import six
import traceback
from abc import ABCMeta, abstractmethod
log = logging.getLogger('parallel_workers')
log.setLevel(logging.INFO)
LOG_INT_SECS = 60

def init_workers(worker_fun, num_worker_threads=2, worker_name='train', init_fun=None, external_loggers=None, shutdown_fun=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.parallel_workers.init_workers', "init_workers(worker_fun, num_worker_threads=2, worker_name='train', init_fun=None, external_loggers=None, shutdown_fun=None)", {'Metrics': Metrics, 'global_coordinator': global_coordinator, 'WorkerCoordinator': WorkerCoordinator, 'threading': threading, 'run_worker': run_worker, 'Worker': Worker, 'worker_fun': worker_fun, 'num_worker_threads': num_worker_threads, 'worker_name': worker_name, 'init_fun': init_fun, 'external_loggers': external_loggers, 'shutdown_fun': shutdown_fun}, 1)


class Metrics(object):
    
    def __init__(self, external_loggers):
        self._metrics = collections.defaultdict(lambda: 0)
        self._external_loggers = external_loggers
    
    def reset_metrics(self):
        self._metrics = collections.defaultdict(lambda: 0)
    
    def log_metrics(self):
        if not self._external_loggers:
            return
        for logger in self._external_loggers:
            try:
                logger.log(self._metrics)
            except Exception as e:
                print('Failed to call ExternalLogger: {}'.format(e))
    
    def put_metric(self, key, value, count=True):
        self._metrics[key] += value
        if count:
            count_key = '{}_count'.format(key)
            self._metrics[count_key] += 1



class State:
    six.add_metaclass(ABCMeta)
    
    @abstractmethod
    def start(self):
        pass
    
    @abstractmethod
    def stop(self):
        pass
    
    @abstractmethod
    def cleanup(self):
        pass



class WorkerCoordinator(object):
    
    def __init__(self, worker_name, worker_ids, init_fun, state=None, shutdown_fun=None):
        self._active = True
        self._started = False
        self._workers = []
        self._worker_name = worker_name
        self._worker_ids = worker_ids
        self._init_fun = init_fun
        self._state = state
        self._shutdown_fun = shutdown_fun
    
    def is_active(self):
        return self._active
    
    def init(self, global_coordinator):
        if (self._init_fun and not self._started):
            data_coordinator = self
            self._init_fun(data_coordinator, global_coordinator)
    
    def _start(self):
        if self._started:
            return
        self._active = True
        self._started = True
        if self._state:
            self._state.start()
        for w in self._workers:
            w.daemon = True
            w.start()
    
    def _stop(self, reason=None):
        self._active = False
        if reason is not None:
            log.error('Data input failed due to an error: {}'.format(reason))
        if (self._shutdown_fun and self._started):
            self._shutdown_fun()
        if self._state:
            self._state.stop()
        self._started = False
    
    def _wait_finish(self, cleanup=None):
        print('Wait for workers to die: {}'.format(self._worker_name))
        for w in self._workers:
            if w != threading.current_thread():
                w.join(5.0)
        success = True
        for w in self._workers:
            if w.is_alive():
                print('Worker {} failed to close while waiting'.format(w))
                success = False
        if (success and self._state):
            self._state.cleanup()
        print('All workers terminated: {}'.format(success))
        return success
    
    def get_worker_ids(self):
        return self._worker_ids



class GlobalWorkerCoordinator(object):
    
    def __init__(self):
        self._coordinators = []
        self._fetcher_id_seq = 0
        self._worker_ids = []
        self.register_shutdown_handler()
    
    def add(self, coordinator):
        self._coordinators.append(coordinator)
    
    def get_new_worker_id(self):
        worker_id = self._fetcher_id_seq
        self._worker_ids.append(worker_id)
        self._fetcher_id_seq += 1
        return worker_id
    
    def get_worker_ids(self):
        return self._worker_ids
    
    def start(self):
        for c in self._coordinators:
            c.init(self)
        for c in self._coordinators:
            c._start()
    
    def stop(self):
        all_success = True
        for c in self._coordinators:
            c._stop()
        for c in self._coordinators:
            success = c._wait_finish()
            all_success = (all_success and success)
        self._coordinators = []
        return all_success
    
    def stop_coordinator(self, worker_name):
        """
        Stop a specific coordinator
        """
        for c in self._coordinators:
            if c._worker_name == worker_name:
                c._stop()
                c._wait_finish()
        self._coordinators = [c for c in self._coordinators if c._worker_name != worker_name]
    
    def register_shutdown_handler(self):
        
        def cleanup():
            self.stop()
        atexit.register(cleanup)



class Worker(object):
    
    def __init__(self, coordinator, worker_id, worker_fun=None, metrics=None):
        self._coordinator = coordinator
        self._worker_id = worker_id
        self._worker_fun = worker_fun
        self._metrics = metrics
    
    def start(self):
        self._start_time = time.time()
    
    def run(self):
        self._worker_fun(self._worker_id)
    
    def handle_exception(self, e):
        traceback.print_exc()
        logging.exception('Exception in worker', e)
        self._coordinator._stop('Exception in worker {}: {}'.format(self._worker_id, e))
    
    def finish(self):
        self._metrics.put_metric('worker_time', time.time() - self._start_time)
        self._metrics.log_metrics()

global_coordinator = GlobalWorkerCoordinator()

def run_worker(coordinator, worker):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.parallel_workers.run_worker', 'run_worker(coordinator, worker)', {'coordinator': coordinator, 'worker': worker}, 0)

