"""
Python HTTP library with thread-safe connection pooling, file post support, user friendly, and more
"""

from __future__ import absolute_import
import logging
import warnings
from logging import NullHandler
from . import exceptions
from ._version import __version__
from .connectionpool import HTTPConnectionPool, HTTPSConnectionPool, connection_from_url
from .filepost import encode_multipart_formdata
from .poolmanager import PoolManager, ProxyManager, proxy_from_url
from .response import HTTPResponse
from .util.request import make_headers
from .util.retry import Retry
from .util.timeout import Timeout
from .util.url import get_host
__author__ = 'Andrey Petrov (andrey.petrov@shazow.net)'
__license__ = 'MIT'
__version__ = __version__
__all__ = ('HTTPConnectionPool', 'HTTPSConnectionPool', 'PoolManager', 'ProxyManager', 'HTTPResponse', 'Retry', 'Timeout', 'add_stderr_logger', 'connection_from_url', 'disable_warnings', 'encode_multipart_formdata', 'get_host', 'make_headers', 'proxy_from_url')
logging.getLogger(__name__).addHandler(NullHandler())

def add_stderr_logger(level=logging.DEBUG):
    """
    Helper for quickly adding a StreamHandler to the logger. Useful for
    debugging.

    Returns the handler after adding it.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('urllib3.__init__.add_stderr_logger', 'add_stderr_logger(level=logging.DEBUG)', {'logging': logging, '__name__': __name__, 'level': level}, 1)
del NullHandler
warnings.simplefilter('always', exceptions.SecurityWarning, append=True)
warnings.simplefilter('default', exceptions.SubjectAltNameWarning, append=True)
warnings.simplefilter('default', exceptions.InsecurePlatformWarning, append=True)
warnings.simplefilter('default', exceptions.SNIMissingWarning, append=True)

def disable_warnings(category=exceptions.HTTPWarning):
    """
    Helper for quickly disabling all urllib3 warnings.
    """
    warnings.simplefilter('ignore', category)

