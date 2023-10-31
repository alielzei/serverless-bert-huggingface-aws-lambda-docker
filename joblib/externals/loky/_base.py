from concurrent.futures import Future as _BaseFuture
from concurrent.futures._base import LOGGER


class Future(_BaseFuture):
    
    def _invoke_callbacks(self):
        for callback in self._done_callbacks:
            try:
                callback(self)
            except BaseException:
                LOGGER.exception(f'exception calling callback for {self!r}')


