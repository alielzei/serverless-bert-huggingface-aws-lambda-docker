"""Implements ProcessPoolExecutor.

The follow diagram and text describe the data-flow through the system:

|======================= In-process =====================|== Out-of-process ==|

+----------+     +----------+       +--------+     +-----------+    +---------+
|          |  => | Work Ids |       |        |     | Call Q    |    | Process |
|          |     +----------+       |        |     +-----------+    |  Pool   |
|          |     | ...      |       |        |     | ...       |    +---------+
|          |     | 6        |    => |        |  => | 5, call() | => |         |
|          |     | 7        |       |        |     | ...       |    |         |
| Process  |     | ...      |       | Local  |     +-----------+    | Process |
|  Pool    |     +----------+       | Worker |                      |  #1..n  |
| Executor |                        | Thread |                      |         |
|          |     +----------- +     |        |     +-----------+    |         |
|          | <=> | Work Items | <=> |        | <=  | Result Q  | <= |         |
|          |     +------------+     |        |     +-----------+    |         |
|          |     | 6: call()  |     |        |     | ...       |    |         |
|          |     |    future  |     +--------+     | 4, result |    |         |
|          |     | ...        |                    | 3, except |    |         |
+----------+     +------------+                    +-----------+    +---------+

Executor.submit() called:
- creates a uniquely numbered _WorkItem and adds it to the "Work Items" dict
- adds the id of the _WorkItem to the "Work Ids" queue

Local worker thread:
- reads work ids from the "Work Ids" queue and looks up the corresponding
  WorkItem from the "Work Items" dict: if the work item has been cancelled then
  it is simply removed from the dict, otherwise it is repackaged as a
  _CallItem and put in the "Call Q". New _CallItems are put in the "Call Q"
  until "Call Q" is full. NOTE: the size of the "Call Q" is kept small because
  calls placed in the "Call Q" can no longer be cancelled with Future.cancel().
- reads _ResultItems from "Result Q", updates the future stored in the
  "Work Items" dict and deletes the dict entry

Process #1..n:
- reads _CallItems from "Call Q", executes the calls, and puts the resulting
  _ResultItems in "Result Q"
"""

__author__ = 'Thomas Moreau (thomas.moreau.2010@gmail.com)'
import os
import gc
import sys
import queue
import struct
import weakref
import warnings
import itertools
import traceback
import threading
from time import time, sleep
import multiprocessing as mp
from functools import partial
from pickle import PicklingError
from concurrent.futures import Executor
from concurrent.futures._base import LOGGER
from concurrent.futures.process import BrokenProcessPool as _BPPException
from multiprocessing.connection import wait
from ._base import Future
from .backend import get_context
from .backend.context import cpu_count, _MAX_WINDOWS_WORKERS
from .backend.queues import Queue, SimpleQueue
from .backend.reduction import set_loky_pickler, get_loky_pickler_name
from .backend.utils import kill_process_tree, get_exitcodes_terminated_worker
from .initializers import _prepare_initializer
MAX_DEPTH = int(os.environ.get('LOKY_MAX_DEPTH', 10))
_CURRENT_DEPTH = 0
_MEMORY_LEAK_CHECK_DELAY = 1.0
_MAX_MEMORY_LEAK_SIZE = int(300000000.0)
try:
    from psutil import Process
    _USE_PSUTIL = True
    
    def _get_memory_usage(pid, force_gc=False):
        import custom_funtemplate
        return custom_funtemplate.rewrite_template('joblib.externals.loky.process_executor._get_memory_usage', '_get_memory_usage(pid, force_gc=False)', {'gc': gc, 'Process': Process, 'mp': mp, 'pid': pid, 'force_gc': force_gc}, 1)
except ImportError:
    _USE_PSUTIL = False


class _ThreadWakeup:
    
    def __init__(self):
        self._closed = False
        (self._reader, self._writer) = mp.Pipe(duplex=False)
    
    def close(self):
        if not self._closed:
            self._closed = True
            self._writer.close()
            self._reader.close()
    
    def wakeup(self):
        if not self._closed:
            self._writer.send_bytes(b'')
    
    def clear(self):
        if not self._closed:
            while self._reader.poll():
                self._reader.recv_bytes()



class _ExecutorFlags:
    """necessary references to maintain executor states without preventing gc

    It permits to keep the information needed by executor_manager_thread
    and crash_detection_thread to maintain the pool without preventing the
    garbage collection of unreferenced executors.
    """
    
    def __init__(self, shutdown_lock):
        self.shutdown = False
        self.broken = None
        self.kill_workers = False
        self.shutdown_lock = shutdown_lock
    
    def flag_as_shutting_down(self, kill_workers=None):
        with self.shutdown_lock:
            self.shutdown = True
            if kill_workers is not None:
                self.kill_workers = kill_workers
    
    def flag_as_broken(self, broken):
        with self.shutdown_lock:
            self.shutdown = True
            self.broken = broken

_global_shutdown = False
_global_shutdown_lock = threading.Lock()
_threads_wakeups = weakref.WeakKeyDictionary()

def _python_exit():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.externals.loky.process_executor._python_exit', '_python_exit()', {'_threads_wakeups': _threads_wakeups, 'mp': mp, '_global_shutdown_lock': _global_shutdown_lock}, 0)
mp.util.register_after_fork(_threads_wakeups, lambda obj: obj.clear())
process_pool_executor_at_exit = None
EXTRA_QUEUED_CALLS = 1


class _RemoteTraceback(Exception):
    """Embed stringification of remote traceback in local traceback"""
    
    def __init__(self, tb=None):
        self.tb = f'\n"""\n{tb}"""'
    
    def __str__(self):
        return self.tb



class _ExceptionWithTraceback:
    
    def __init__(self, exc):
        tb = getattr(exc, '__traceback__', None)
        if tb is None:
            (_, _, tb) = sys.exc_info()
        tb = traceback.format_exception(type(exc), exc, tb)
        tb = ''.join(tb)
        self.exc = exc
        self.tb = tb
    
    def __reduce__(self):
        return (_rebuild_exc, (self.exc, self.tb))


def _rebuild_exc(exc, tb):
    exc.__cause__ = _RemoteTraceback(tb)
    return exc


class _WorkItem:
    __slots__ = ['future', 'fn', 'args', 'kwargs']
    
    def __init__(self, future, fn, args, kwargs):
        self.future = future
        self.fn = fn
        self.args = args
        self.kwargs = kwargs



class _ResultItem:
    
    def __init__(self, work_id, exception=None, result=None):
        self.work_id = work_id
        self.exception = exception
        self.result = result



class _CallItem:
    
    def __init__(self, work_id, fn, args, kwargs):
        self.work_id = work_id
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.loky_pickler = get_loky_pickler_name()
    
    def __call__(self):
        set_loky_pickler(self.loky_pickler)
        return self.fn(*self.args, **self.kwargs)
    
    def __repr__(self):
        return f'CallItem({self.work_id}, {self.fn}, {self.args}, {self.kwargs})'



class _SafeQueue(Queue):
    """Safe Queue set exception to the future object linked to a job"""
    
    def __init__(self, max_size=0, ctx=None, pending_work_items=None, running_work_items=None, thread_wakeup=None, reducers=None):
        self.thread_wakeup = thread_wakeup
        self.pending_work_items = pending_work_items
        self.running_work_items = running_work_items
        super().__init__(max_size, reducers=reducers, ctx=ctx)
    
    def _on_queue_feeder_error(self, e, obj):
        if isinstance(obj, _CallItem):
            if isinstance(e, struct.error):
                raised_error = RuntimeError('The task could not be sent to the workers as it is too large for `send_bytes`.')
            else:
                raised_error = PicklingError('Could not pickle the task to send it to the workers.')
            tb = traceback.format_exception(type(e), e, getattr(e, '__traceback__', None))
            raised_error.__cause__ = _RemoteTraceback(''.join(tb))
            work_item = self.pending_work_items.pop(obj.work_id, None)
            self.running_work_items.remove(obj.work_id)
            if work_item is not None:
                work_item.future.set_exception(raised_error)
                del work_item
            self.thread_wakeup.wakeup()
        else:
            super()._on_queue_feeder_error(e, obj)


def _get_chunks(chunksize, *iterables):
    """Iterates over zip()ed iterables in chunks."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.externals.loky.process_executor._get_chunks', '_get_chunks(chunksize, *iterables)', {'itertools': itertools, 'chunksize': chunksize, 'iterables': iterables}, 1)

def _process_chunk(fn, chunk):
    """Processes a chunk of an iterable passed to map.

    Runs the function passed to map() on a chunk of the
    iterable passed to map.

    This function is run in a separate process.

    """
    return [fn(*args) for args in chunk]

def _sendback_result(result_queue, work_id, result=None, exception=None):
    """Safely send back the given result or exception"""
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.externals.loky.process_executor._sendback_result', '_sendback_result(result_queue, work_id, result=None, exception=None)', {'_ResultItem': _ResultItem, '_ExceptionWithTraceback': _ExceptionWithTraceback, 'result_queue': result_queue, 'work_id': work_id, 'result': result, 'exception': exception}, 0)

def _process_worker(call_queue, result_queue, initializer, initargs, processes_management_lock, timeout, worker_exit_lock, current_depth):
    """Evaluates calls from call_queue and places the results in result_queue.

    This worker is run in a separate process.

    Args:
        call_queue: A ctx.Queue of _CallItems that will be read and
            evaluated by the worker.
        result_queue: A ctx.Queue of _ResultItems that will written
            to by the worker.
        initializer: A callable initializer, or None
        initargs: A tuple of args for the initializer
        processes_management_lock: A ctx.Lock avoiding worker timeout while
            some workers are being spawned.
        timeout: maximum time to wait for a new item in the call_queue. If that
            time is expired, the worker will shutdown.
        worker_exit_lock: Lock to avoid flagging the executor as broken on
            workers timeout.
        current_depth: Nested parallelism level, to avoid infinite spawning.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.externals.loky.process_executor._process_worker', '_process_worker(call_queue, result_queue, initializer, initargs, processes_management_lock, timeout, worker_exit_lock, current_depth)', {'LOGGER': LOGGER, 'os': os, 'mp': mp, 'queue': queue, 'traceback': traceback, '_RemoteTraceback': _RemoteTraceback, 'sys': sys, '_python_exit': _python_exit, '_ExceptionWithTraceback': _ExceptionWithTraceback, '_ResultItem': _ResultItem, '_sendback_result': _sendback_result, '_USE_PSUTIL': _USE_PSUTIL, '_get_memory_usage': _get_memory_usage, 'time': time, '_MEMORY_LEAK_CHECK_DELAY': _MEMORY_LEAK_CHECK_DELAY, '_MAX_MEMORY_LEAK_SIZE': _MAX_MEMORY_LEAK_SIZE, 'gc': gc, 'call_queue': call_queue, 'result_queue': result_queue, 'initializer': initializer, 'initargs': initargs, 'processes_management_lock': processes_management_lock, 'timeout': timeout, 'worker_exit_lock': worker_exit_lock, 'current_depth': current_depth}, 1)


class _ExecutorManagerThread(threading.Thread):
    """Manages the communication between this process and the worker processes.

    The manager is run in a local thread.

    Args:
        executor: A reference to the ProcessPoolExecutor that owns
            this thread. A weakref will be own by the manager as well as
            references to internal objects used to introspect the state of
            the executor.
    """
    
    def __init__(self, executor):
        self.thread_wakeup = executor._executor_manager_thread_wakeup
        self.shutdown_lock = executor._shutdown_lock
        
        def weakref_cb(_, thread_wakeup=self.thread_wakeup, shutdown_lock=self.shutdown_lock):
            if mp is not None:
                mp.util.debug('Executor collected: triggering callback for QueueManager wakeup')
            with shutdown_lock:
                thread_wakeup.wakeup()
        self.executor_reference = weakref.ref(executor, weakref_cb)
        self.executor_flags = executor._flags
        self.processes = executor._processes
        self.call_queue = executor._call_queue
        self.result_queue = executor._result_queue
        self.work_ids_queue = executor._work_ids
        self.pending_work_items = executor._pending_work_items
        self.running_work_items = executor._running_work_items
        self.processes_management_lock = executor._processes_management_lock
        super().__init__(name='ExecutorManagerThread')
        if sys.version_info < (3, 9):
            self.daemon = True
    
    def run(self):
        while True:
            self.add_call_item_to_queue()
            (result_item, is_broken, bpe) = self.wait_result_broken_or_wakeup()
            if is_broken:
                self.terminate_broken(bpe)
                return
            if result_item is not None:
                self.process_result_item(result_item)
                del result_item
            if self.is_shutting_down():
                self.flag_executor_shutting_down()
                if not self.pending_work_items:
                    self.join_executor_internals()
                    return
    
    def add_call_item_to_queue(self):
        while True:
            if self.call_queue.full():
                return
            try:
                work_id = self.work_ids_queue.get(block=False)
            except queue.Empty:
                return
            else:
                work_item = self.pending_work_items[work_id]
                if work_item.future.set_running_or_notify_cancel():
                    self.running_work_items += [work_id]
                    self.call_queue.put(_CallItem(work_id, work_item.fn, work_item.args, work_item.kwargs), block=True)
                else:
                    del self.pending_work_items[work_id]
                    continue
    
    def wait_result_broken_or_wakeup(self):
        result_reader = self.result_queue._reader
        wakeup_reader = self.thread_wakeup._reader
        readers = [result_reader, wakeup_reader]
        worker_sentinels = [p.sentinel for p in list(self.processes.values())]
        ready = wait(readers + worker_sentinels)
        bpe = None
        is_broken = True
        result_item = None
        if result_reader in ready:
            try:
                result_item = result_reader.recv()
                if isinstance(result_item, _RemoteTraceback):
                    bpe = BrokenProcessPool('A task has failed to un-serialize. Please ensure that the arguments of the function are all picklable.')
                    bpe.__cause__ = result_item
                else:
                    is_broken = False
            except BaseException as e:
                bpe = BrokenProcessPool('A result has failed to un-serialize. Please ensure that the objects returned by the function are always picklable.')
                tb = traceback.format_exception(type(e), e, getattr(e, '__traceback__', None))
                bpe.__cause__ = _RemoteTraceback(''.join(tb))
        elif wakeup_reader in ready:
            is_broken = False
        else:
            exit_codes = ''
            if sys.platform != 'win32':
                exit_codes = f'\nThe exit codes of the workers are {get_exitcodes_terminated_worker(self.processes)}'
            mp.util.debug('A worker unexpectedly terminated. Workers that might have caused the breakage: ' + str({p.name: p.exitcode for p in list(self.processes.values()) if (p is not None and p.sentinel in ready)}))
            bpe = TerminatedWorkerError(f'A worker process managed by the executor was unexpectedly terminated. This could be caused by a segmentation fault while calling the function or by an excessive memory usage causing the Operating System to kill the worker.\n{exit_codes}')
        self.thread_wakeup.clear()
        return (result_item, is_broken, bpe)
    
    def process_result_item(self, result_item):
        if isinstance(result_item, int):
            with self.processes_management_lock:
                p = self.processes.pop(result_item, None)
            if p is not None:
                p._worker_exit_lock.release()
                mp.util.debug(f'joining {p.name} when processing {p.pid} as result_item')
                p.join()
                del p
            n_pending = len(self.pending_work_items)
            n_running = len(self.running_work_items)
            if (n_pending - n_running > 0 or n_running > len(self.processes)):
                executor = self.executor_reference()
                if (executor is not None and len(self.processes) < executor._max_workers):
                    warnings.warn('A worker stopped while some jobs were given to the executor. This can be caused by a too short worker timeout or by a memory leak.', UserWarning)
                    with executor._processes_management_lock:
                        executor._adjust_process_count()
                    executor = None
        else:
            work_item = self.pending_work_items.pop(result_item.work_id, None)
            if work_item is not None:
                if result_item.exception:
                    work_item.future.set_exception(result_item.exception)
                else:
                    work_item.future.set_result(result_item.result)
                self.running_work_items.remove(result_item.work_id)
    
    def is_shutting_down(self):
        executor = self.executor_reference()
        return (_global_shutdown or (((executor is None or self.executor_flags.shutdown)) and not self.executor_flags.broken))
    
    def terminate_broken(self, bpe):
        self.executor_flags.flag_as_broken(bpe)
        for work_item in self.pending_work_items.values():
            work_item.future.set_exception(bpe)
            del work_item
        self.pending_work_items.clear()
        self.kill_workers(reason='broken executor')
        self.join_executor_internals()
    
    def flag_executor_shutting_down(self):
        self.executor_flags.flag_as_shutting_down()
        if self.executor_flags.kill_workers:
            while self.pending_work_items:
                (_, work_item) = self.pending_work_items.popitem()
                work_item.future.set_exception(ShutdownExecutorError('The Executor was shutdown with `kill_workers=True` before this job could complete.'))
                del work_item
            self.kill_workers(reason='executor shutting down')
    
    def kill_workers(self, reason=''):
        while self.processes:
            (_, p) = self.processes.popitem()
            mp.util.debug(f'terminate process {p.name}, reason: {reason}')
            try:
                kill_process_tree(p)
            except ProcessLookupError:
                pass
    
    def shutdown_workers(self):
        with self.processes_management_lock:
            n_children_to_stop = 0
            for p in list(self.processes.values()):
                mp.util.debug(f'releasing worker exit lock on {p.name}')
                p._worker_exit_lock.release()
                n_children_to_stop += 1
        mp.util.debug(f'found {n_children_to_stop} processes to stop')
        n_sentinels_sent = 0
        cooldown_time = 0.001
        while (n_sentinels_sent < n_children_to_stop and self.get_n_children_alive() > 0):
            for _ in range(n_children_to_stop - n_sentinels_sent):
                try:
                    self.call_queue.put_nowait(None)
                    n_sentinels_sent += 1
                except queue.Full as e:
                    if cooldown_time > 5.0:
                        mp.util.info(f'failed to send all sentinels and exit with error.\ncall_queue size={self.call_queue._maxsize};  full is {self.call_queue.full()}; ')
                        raise e
                    mp.util.info('full call_queue prevented to send all sentinels at once, waiting...')
                    sleep(cooldown_time)
                    cooldown_time *= 1.2
                    break
        mp.util.debug(f'sent {n_sentinels_sent} sentinels to the call queue')
    
    def join_executor_internals(self):
        self.shutdown_workers()
        mp.util.debug('closing call_queue')
        self.call_queue.close()
        self.call_queue.join_thread()
        mp.util.debug('closing result_queue')
        self.result_queue.close()
        mp.util.debug('closing thread_wakeup')
        with self.shutdown_lock:
            self.thread_wakeup.close()
        with self.processes_management_lock:
            mp.util.debug(f'joining {len(self.processes)} processes')
            n_joined_processes = 0
            while True:
                try:
                    (pid, p) = self.processes.popitem()
                    mp.util.debug(f'joining process {p.name} with pid {pid}')
                    p.join()
                    n_joined_processes += 1
                except KeyError:
                    break
            mp.util.debug(f'executor management thread clean shutdown of {n_joined_processes} workers')
    
    def get_n_children_alive(self):
        with self.processes_management_lock:
            return sum((p.is_alive() for p in list(self.processes.values())))

_system_limits_checked = False
_system_limited = None

def _check_system_limits():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.externals.loky.process_executor._check_system_limits', '_check_system_limits()', {'os': os}, 1)

def _chain_from_iterable_of_lists(iterable):
    """
    Specialized implementation of itertools.chain.from_iterable.
    Each item in *iterable* should be a list.  This function is
    careful not to keep references to yielded objects.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.externals.loky.process_executor._chain_from_iterable_of_lists', '_chain_from_iterable_of_lists(iterable)', {'iterable': iterable}, 0)

def _check_max_depth(context):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.externals.loky.process_executor._check_max_depth', '_check_max_depth(context)', {'_CURRENT_DEPTH': _CURRENT_DEPTH, 'LokyRecursionError': LokyRecursionError, 'MAX_DEPTH': MAX_DEPTH, 'context': context}, 0)


class LokyRecursionError(RuntimeError):
    """A process tries to spawn too many levels of nested processes."""
    



class BrokenProcessPool(_BPPException):
    """
    Raised when the executor is broken while a future was in the running state.
    The cause can an error raised when unpickling the task in the worker
    process or when unpickling the result value in the parent process. It can
    also be caused by a worker process being terminated unexpectedly.
    """
    



class TerminatedWorkerError(BrokenProcessPool):
    """
    Raised when a process in a ProcessPoolExecutor terminated abruptly
    while a future was in the running state.
    """
    

BrokenExecutor = BrokenProcessPool


class ShutdownExecutorError(RuntimeError):
    """
    Raised when a ProcessPoolExecutor is shutdown while a future was in the
    running or pending state.
    """
    



class ProcessPoolExecutor(Executor):
    _at_exit = None
    
    def __init__(self, max_workers=None, job_reducers=None, result_reducers=None, timeout=None, context=None, initializer=None, initargs=(), env=None):
        """Initializes a new ProcessPoolExecutor instance.

        Args:
            max_workers: int, optional (default: cpu_count())
                The maximum number of processes that can be used to execute the
                given calls. If None or not given then as many worker processes
                will be created as the number of CPUs the current process
                can use.
            job_reducers, result_reducers: dict(type: reducer_func)
                Custom reducer for pickling the jobs and the results from the
                Executor. If only `job_reducers` is provided, `result_reducer`
                will use the same reducers
            timeout: int, optional (default: None)
                Idle workers exit after timeout seconds. If a new job is
                submitted after the timeout, the executor will start enough
                new Python processes to make sure the pool of workers is full.
            context: A multiprocessing context to launch the workers. This
                object should provide SimpleQueue, Queue and Process.
            initializer: An callable used to initialize worker processes.
            initargs: A tuple of arguments to pass to the initializer.
            env: A dict of environment variable to overwrite in the child
                process. The environment variables are set before any module is
                loaded. Note that this only works with the loky context.
        """
        _check_system_limits()
        if max_workers is None:
            self._max_workers = cpu_count()
        else:
            if max_workers <= 0:
                raise ValueError('max_workers must be greater than 0')
            self._max_workers = max_workers
        if (sys.platform == 'win32' and self._max_workers > _MAX_WINDOWS_WORKERS):
            warnings.warn(f'On Windows, max_workers cannot exceed {_MAX_WINDOWS_WORKERS} due to limitations of the operating system.')
            self._max_workers = _MAX_WINDOWS_WORKERS
        if context is None:
            context = get_context()
        self._context = context
        self._env = env
        (self._initializer, self._initargs) = _prepare_initializer(initializer, initargs)
        _check_max_depth(self._context)
        if result_reducers is None:
            result_reducers = job_reducers
        self._timeout = timeout
        self._executor_manager_thread = None
        self._processes = {}
        self._processes = {}
        self._queue_count = 0
        self._pending_work_items = {}
        self._running_work_items = []
        self._work_ids = queue.Queue()
        self._processes_management_lock = self._context.Lock()
        self._executor_manager_thread = None
        self._shutdown_lock = threading.Lock()
        self._executor_manager_thread_wakeup = _ThreadWakeup()
        self._flags = _ExecutorFlags(self._shutdown_lock)
        self._setup_queues(job_reducers, result_reducers)
        mp.util.debug('ProcessPoolExecutor is setup')
    
    def _setup_queues(self, job_reducers, result_reducers, queue_size=None):
        if queue_size is None:
            queue_size = 2 * self._max_workers + EXTRA_QUEUED_CALLS
        self._call_queue = _SafeQueue(max_size=queue_size, pending_work_items=self._pending_work_items, running_work_items=self._running_work_items, thread_wakeup=self._executor_manager_thread_wakeup, reducers=job_reducers, ctx=self._context)
        self._call_queue._ignore_epipe = True
        self._result_queue = SimpleQueue(reducers=result_reducers, ctx=self._context)
    
    def _start_executor_manager_thread(self):
        if self._executor_manager_thread is None:
            mp.util.debug('_start_executor_manager_thread called')
            self._executor_manager_thread = _ExecutorManagerThread(self)
            self._executor_manager_thread.start()
            _threads_wakeups[self._executor_manager_thread] = (self._shutdown_lock, self._executor_manager_thread_wakeup)
            global process_pool_executor_at_exit
            if process_pool_executor_at_exit is None:
                if sys.version_info < (3, 9):
                    process_pool_executor_at_exit = mp.util.Finalize(None, _python_exit, exitpriority=20)
                else:
                    process_pool_executor_at_exit = threading._register_atexit(_python_exit)
    
    def _adjust_process_count(self):
        while len(self._processes) < self._max_workers:
            worker_exit_lock = self._context.BoundedSemaphore(1)
            args = (self._call_queue, self._result_queue, self._initializer, self._initargs, self._processes_management_lock, self._timeout, worker_exit_lock, _CURRENT_DEPTH + 1)
            worker_exit_lock.acquire()
            try:
                p = self._context.Process(target=_process_worker, args=args, env=self._env)
            except TypeError:
                p = self._context.Process(target=_process_worker, args=args)
            p._worker_exit_lock = worker_exit_lock
            p.start()
            self._processes[p.pid] = p
        mp.util.debug(f'Adjusted process count to {self._max_workers}: {[(p.name, pid) for (pid, p) in self._processes.items()]}')
    
    def _ensure_executor_running(self):
        """ensures all workers and management thread are running"""
        with self._processes_management_lock:
            if len(self._processes) != self._max_workers:
                self._adjust_process_count()
            self._start_executor_manager_thread()
    
    def submit(self, fn, *args, **kwargs):
        with self._flags.shutdown_lock:
            if self._flags.broken is not None:
                raise self._flags.broken
            if self._flags.shutdown:
                raise ShutdownExecutorError('cannot schedule new futures after shutdown')
            if _global_shutdown:
                raise RuntimeError('cannot schedule new futures after interpreter shutdown')
            f = Future()
            w = _WorkItem(f, fn, args, kwargs)
            self._pending_work_items[self._queue_count] = w
            self._work_ids.put(self._queue_count)
            self._queue_count += 1
            self._executor_manager_thread_wakeup.wakeup()
            self._ensure_executor_running()
            return f
    submit.__doc__ = Executor.submit.__doc__
    
    def map(self, fn, *iterables, **kwargs):
        """Returns an iterator equivalent to map(fn, iter).

        Args:
            fn: A callable that will take as many arguments as there are
                passed iterables.
            timeout: The maximum number of seconds to wait. If None, then there
                is no limit on the wait time.
            chunksize: If greater than one, the iterables will be chopped into
                chunks of size chunksize and submitted to the process pool.
                If set to one, the items in the list will be sent one at a
                time.

        Returns:
            An iterator equivalent to: map(func, *iterables) but the calls may
            be evaluated out-of-order.

        Raises:
            TimeoutError: If the entire result iterator could not be generated
                before the given timeout.
            Exception: If fn(*args) raises for any values.
        """
        timeout = kwargs.get('timeout', None)
        chunksize = kwargs.get('chunksize', 1)
        if chunksize < 1:
            raise ValueError('chunksize must be >= 1.')
        results = super().map(partial(_process_chunk, fn), _get_chunks(chunksize, *iterables), timeout=timeout)
        return _chain_from_iterable_of_lists(results)
    
    def shutdown(self, wait=True, kill_workers=False):
        mp.util.debug(f'shutting down executor {self}')
        self._flags.flag_as_shutting_down(kill_workers)
        executor_manager_thread = self._executor_manager_thread
        executor_manager_thread_wakeup = self._executor_manager_thread_wakeup
        if executor_manager_thread_wakeup is not None:
            with self._shutdown_lock:
                self._executor_manager_thread_wakeup.wakeup()
        if (executor_manager_thread is not None and wait):
            with _global_shutdown_lock:
                executor_manager_thread.join()
                _threads_wakeups.pop(executor_manager_thread, None)
        self._executor_manager_thread = None
        self._executor_manager_thread_wakeup = None
        self._call_queue = None
        self._result_queue = None
        self._processes_management_lock = None
    shutdown.__doc__ = Executor.shutdown.__doc__


