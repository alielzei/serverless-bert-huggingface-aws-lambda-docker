import os
import sys
import errno
import weakref
import threading
from multiprocessing import util
from multiprocessing.queues import Full, Queue as mp_Queue, SimpleQueue as mp_SimpleQueue, _sentinel
from multiprocessing.context import assert_spawning
from .reduction import dumps
__all__ = ['Queue', 'SimpleQueue', 'Full']


class Queue(mp_Queue):
    
    def __init__(self, maxsize=0, reducers=None, ctx=None):
        super().__init__(maxsize=maxsize, ctx=ctx)
        self._reducers = reducers
    
    def __getstate__(self):
        assert_spawning(self)
        return (self._ignore_epipe, self._maxsize, self._reader, self._writer, self._reducers, self._rlock, self._wlock, self._sem, self._opid)
    
    def __setstate__(self, state):
        (self._ignore_epipe, self._maxsize, self._reader, self._writer, self._reducers, self._rlock, self._wlock, self._sem, self._opid) = state
        if sys.version_info >= (3, 9):
            self._reset()
        else:
            self._after_fork()
    
    def _start_thread(self):
        util.debug('Queue._start_thread()')
        self._buffer.clear()
        self._thread = threading.Thread(target=Queue._feed, args=(self._buffer, self._notempty, self._send_bytes, self._wlock, self._writer.close, self._reducers, self._ignore_epipe, self._on_queue_feeder_error, self._sem), name='QueueFeederThread')
        self._thread.daemon = True
        util.debug('doing self._thread.start()')
        self._thread.start()
        util.debug('... done self._thread.start()')
        created_by_this_process = self._opid == os.getpid()
        if (not self._joincancelled and not created_by_this_process):
            self._jointhread = util.Finalize(self._thread, Queue._finalize_join, [weakref.ref(self._thread)], exitpriority=-5)
        self._close = util.Finalize(self, Queue._finalize_close, [self._buffer, self._notempty], exitpriority=10)
    
    @staticmethod
    def _feed(buffer, notempty, send_bytes, writelock, close, reducers, ignore_epipe, onerror, queue_sem):
        util.debug('starting thread to feed data to pipe')
        nacquire = notempty.acquire
        nrelease = notempty.release
        nwait = notempty.wait
        bpopleft = buffer.popleft
        sentinel = _sentinel
        if sys.platform != 'win32':
            wacquire = writelock.acquire
            wrelease = writelock.release
        else:
            wacquire = None
        while True:
            try:
                nacquire()
                try:
                    if not buffer:
                        nwait()
                finally:
                    nrelease()
                try:
                    while True:
                        obj = bpopleft()
                        if obj is sentinel:
                            util.debug('feeder thread got sentinel -- exiting')
                            close()
                            return
                        obj_ = dumps(obj, reducers=reducers)
                        if wacquire is None:
                            send_bytes(obj_)
                        else:
                            wacquire()
                            try:
                                send_bytes(obj_)
                            finally:
                                wrelease()
                        del obj, obj_
                except IndexError:
                    pass
            except BaseException as e:
                if (ignore_epipe and getattr(e, 'errno', 0) == errno.EPIPE):
                    return
                if util.is_exiting():
                    util.info(f'error in queue thread: {e}')
                    return
                else:
                    queue_sem.release()
                    onerror(e, obj)
    
    def _on_queue_feeder_error(self, e, obj):
        """
        Private API hook called when feeding data in the background thread
        raises an exception.  For overriding by concurrent.futures.
        """
        import traceback
        traceback.print_exc()



class SimpleQueue(mp_SimpleQueue):
    
    def __init__(self, reducers=None, ctx=None):
        super().__init__(ctx=ctx)
        self._reducers = reducers
    
    def close(self):
        self._reader.close()
        self._writer.close()
    
    def __getstate__(self):
        assert_spawning(self)
        return (self._reader, self._writer, self._reducers, self._rlock, self._wlock)
    
    def __setstate__(self, state):
        (self._reader, self._writer, self._reducers, self._rlock, self._wlock) = state
    
    def put(self, obj):
        obj = dumps(obj, reducers=self._reducers)
        if self._wlock is None:
            self._writer.send_bytes(obj)
        else:
            with self._wlock:
                self._writer.send_bytes(obj)


