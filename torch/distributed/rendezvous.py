try:
    from urllib.parse import urlparse, urlunparse
except ImportError:
    from urlparse import urlparse, urlunparse
import torch._six as six
import numbers
import os
from . import FileStore, TCPStore
from .constants import default_pg_timeout
_rendezvous_handlers = {}

def register_rendezvous_handler(scheme, handler):
    """Registers a new rendezvous handler.

    Before we can run collective algorithms, participating processes
    need to find each other and exchange information to be able to
    communicate. We call this process rendezvous.

    The outcome of the rendezvous process is a triplet containing a
    shared key/value store, the rank of the process, and the total
    number of participating processes.

    If none of the bundled rendezvous methods apply to your execution
    environment you can opt to register your own rendezvous handler.
    Pick a unique name and use the URL scheme to identify it when
    calling the `rendezvous()` function.

    Arguments:
        scheme (str): URL scheme to identify your rendezvous handler.
        handler (function): Handler that is invoked when the
            `rendezvous()` function is called with a URL that uses
            the corresponding scheme. It must be a generator function
            that yields the triplet.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.distributed.rendezvous.register_rendezvous_handler', 'register_rendezvous_handler(scheme, handler)', {'_rendezvous_handlers': _rendezvous_handlers, 'scheme': scheme, 'handler': handler}, 0)

def rendezvous(url, rank=-1, world_size=-1, **kwargs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributed.rendezvous.rendezvous', 'rendezvous(url, rank=-1, world_size=-1, **kwargs)', {'six': six, 'numbers': numbers, 'urlparse': urlparse, 'urlunparse': urlunparse, '_rendezvous_handlers': _rendezvous_handlers, 'url': url, 'rank': rank, 'world_size': world_size, 'kwargs': kwargs}, 1)

def _rendezvous_error(msg):
    return ValueError('Error initializing torch.distributed using ' + msg)

def _file_rendezvous_handler(url, **kwargs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributed.rendezvous._file_rendezvous_handler', '_file_rendezvous_handler(url, **kwargs)', {'_rendezvous_error': _rendezvous_error, 'urlparse': urlparse, 'FileStore': FileStore, 'url': url, 'kwargs': kwargs}, 1)

def _tcp_rendezvous_handler(url, timeout=default_pg_timeout, **kwargs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributed.rendezvous._tcp_rendezvous_handler', '_tcp_rendezvous_handler(url, timeout=default_pg_timeout, **kwargs)', {'_rendezvous_error': _rendezvous_error, 'urlparse': urlparse, 'TCPStore': TCPStore, 'url': url, 'timeout': timeout, 'default_pg_timeout': default_pg_timeout, 'kwargs': kwargs}, 1)

def _env_rendezvous_handler(url, timeout=default_pg_timeout, **kwargs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributed.rendezvous._env_rendezvous_handler', '_env_rendezvous_handler(url, timeout=default_pg_timeout, **kwargs)', {'_rendezvous_error': _rendezvous_error, 'urlparse': urlparse, 'os': os, 'TCPStore': TCPStore, 'url': url, 'timeout': timeout, 'default_pg_timeout': default_pg_timeout, 'kwargs': kwargs}, 1)
register_rendezvous_handler('file', _file_rendezvous_handler)
register_rendezvous_handler('tcp', _tcp_rendezvous_handler)
register_rendezvous_handler('env', _env_rendezvous_handler)

