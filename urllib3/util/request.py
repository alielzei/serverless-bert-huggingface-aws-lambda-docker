from __future__ import absolute_import
from base64 import b64encode
from ..exceptions import UnrewindableBodyError
from ..packages.six import b, integer_types
SKIP_HEADER = '@@@SKIP_HEADER@@@'
SKIPPABLE_HEADERS = frozenset(['accept-encoding', 'host', 'user-agent'])
ACCEPT_ENCODING = 'gzip,deflate'
try:
    import brotli as _unused_module_brotli
except ImportError:
    pass
else:
    ACCEPT_ENCODING += ',br'
_FAILEDTELL = object()

def make_headers(keep_alive=None, accept_encoding=None, user_agent=None, basic_auth=None, proxy_basic_auth=None, disable_cache=None):
    """
    Shortcuts for generating request headers.

    :param keep_alive:
        If ``True``, adds 'connection: keep-alive' header.

    :param accept_encoding:
        Can be a boolean, list, or string.
        ``True`` translates to 'gzip,deflate'.
        List will get joined by comma.
        String will be used as provided.

    :param user_agent:
        String representing the user-agent you want, such as
        "python-urllib3/0.6"

    :param basic_auth:
        Colon-separated username:password string for 'authorization: basic ...'
        auth header.

    :param proxy_basic_auth:
        Colon-separated username:password string for 'proxy-authorization: basic ...'
        auth header.

    :param disable_cache:
        If ``True``, adds 'cache-control: no-cache' header.

    Example::

        >>> make_headers(keep_alive=True, user_agent="Batman/1.0")
        {'connection': 'keep-alive', 'user-agent': 'Batman/1.0'}
        >>> make_headers(accept_encoding=True)
        {'accept-encoding': 'gzip,deflate'}
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('urllib3.util.request.make_headers', 'make_headers(keep_alive=None, accept_encoding=None, user_agent=None, basic_auth=None, proxy_basic_auth=None, disable_cache=None)', {'ACCEPT_ENCODING': ACCEPT_ENCODING, 'b64encode': b64encode, 'b': b, 'keep_alive': keep_alive, 'accept_encoding': accept_encoding, 'user_agent': user_agent, 'basic_auth': basic_auth, 'proxy_basic_auth': proxy_basic_auth, 'disable_cache': disable_cache}, 1)

def set_file_position(body, pos):
    """
    If a position is provided, move file to that point.
    Otherwise, we'll attempt to record a position for future use.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('urllib3.util.request.set_file_position', 'set_file_position(body, pos)', {'rewind_body': rewind_body, 'IOError': IOError, '_FAILEDTELL': _FAILEDTELL, 'body': body, 'pos': pos}, 1)

def rewind_body(body, body_pos):
    """
    Attempt to rewind body to a certain position.
    Primarily used for request redirects and retries.

    :param body:
        File-like object that supports seek.

    :param int pos:
        Position to seek to in file.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('urllib3.util.request.rewind_body', 'rewind_body(body, body_pos)', {'integer_types': integer_types, 'IOError': IOError, 'UnrewindableBodyError': UnrewindableBodyError, '_FAILEDTELL': _FAILEDTELL, 'body': body, 'body_pos': body_pos}, 0)

