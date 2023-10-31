"""
Helper functionality for interoperability with stdlib `logging`.
"""

import logging
import sys
from contextlib import contextmanager
try:
    from typing import Iterator, List, Optional, Type
except ImportError:
    pass
from ..std import tqdm as std_tqdm


class _TqdmLoggingHandler(logging.StreamHandler):
    
    def __init__(self, tqdm_class=std_tqdm):
        super(_TqdmLoggingHandler, self).__init__()
        self.tqdm_class = tqdm_class
    
    def emit(self, record):
        try:
            msg = self.format(record)
            self.tqdm_class.write(msg, file=self.stream)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)


def _is_console_logging_handler(handler):
    return (isinstance(handler, logging.StreamHandler) and handler.stream in {sys.stdout, sys.stderr})

def _get_first_found_console_logging_handler(handlers):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('tqdm.contrib.logging._get_first_found_console_logging_handler', '_get_first_found_console_logging_handler(handlers)', {'_is_console_logging_handler': _is_console_logging_handler, 'handlers': handlers}, 1)

@contextmanager
def logging_redirect_tqdm(loggers=None, tqdm_class=std_tqdm):
    """
    Context manager redirecting console logging to `tqdm.write()`, leaving
    other logging handlers (e.g. log files) unaffected.

    Parameters
    ----------
    loggers  : list, optional
      Which handlers to redirect (default: [logging.root]).
    tqdm_class  : optional

    Example
    -------
    ```python
    import logging
    from tqdm import trange
    from tqdm.contrib.logging import logging_redirect_tqdm

    LOG = logging.getLogger(__name__)

    if __name__ == '__main__':
        logging.basicConfig(level=logging.INFO)
        with logging_redirect_tqdm():
            for i in trange(9):
                if i == 4:
                    LOG.info("console logging redirected to `tqdm.write()`")
        # logging restored
    ```
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('tqdm.contrib.logging.logging_redirect_tqdm', 'logging_redirect_tqdm(loggers=None, tqdm_class=std_tqdm)', {'logging': logging, '_TqdmLoggingHandler': _TqdmLoggingHandler, '_get_first_found_console_logging_handler': _get_first_found_console_logging_handler, '_is_console_logging_handler': _is_console_logging_handler, 'contextmanager': contextmanager, 'loggers': loggers, 'tqdm_class': tqdm_class, 'std_tqdm': std_tqdm}, 0)

@contextmanager
def tqdm_logging_redirect(*args, **kwargs):
    """
    Convenience shortcut for:
    ```python
    with tqdm_class(*args, **tqdm_kwargs) as pbar:
        with logging_redirect_tqdm(loggers=loggers, tqdm_class=tqdm_class):
            yield pbar
    ```

    Parameters
    ----------
    tqdm_class  : optional, (default: tqdm.std.tqdm).
    loggers  : optional, list.
    **tqdm_kwargs  : passed to `tqdm_class`.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('tqdm.contrib.logging.tqdm_logging_redirect', 'tqdm_logging_redirect(*args, **kwargs)', {'std_tqdm': std_tqdm, 'logging_redirect_tqdm': logging_redirect_tqdm, 'contextmanager': contextmanager, 'args': args, 'kwargs': kwargs}, 0)

