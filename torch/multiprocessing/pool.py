import multiprocessing
import multiprocessing.pool
import multiprocessing.util as util
from .queue import SimpleQueue

def clean_worker(*args, **kwargs):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.multiprocessing.pool.clean_worker', 'clean_worker(*args, **kwargs)', {'multiprocessing': multiprocessing, 'args': args, 'kwargs': kwargs}, 0)


class Pool(multiprocessing.pool.Pool):
    """Pool implementation which uses our version of SimpleQueue.
    This lets us pass tensors in shared memory across processes instead of
    serializing the underlying data."""
    
    def _setup_queues(self):
        self._inqueue = SimpleQueue()
        self._outqueue = SimpleQueue()
        self._quick_put = self._inqueue._writer.send
        self._quick_get = self._outqueue._reader.recv
    
    def _repopulate_pool(self):
        """Bring the number of pool processes up to the specified number,
        for use after reaping workers which have exited.
        """
        for i in range(self._processes - len(self._pool)):
            args = (self._inqueue, self._outqueue, self._initializer, self._initargs, self._maxtasksperchild)
            if hasattr(self, '_wrap_exception'):
                args += (self._wrap_exception, )
            w = self.Process(target=clean_worker, args=args)
            self._pool.append(w)
            w.name = w.name.replace('Process', 'PoolWorker')
            w.daemon = True
            w.start()
            util.debug('added worker')


