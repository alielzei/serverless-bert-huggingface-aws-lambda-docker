from __future__ import absolute_import
import socket
from urllib3.exceptions import LocationParseError
from ..contrib import _appengine_environ
from ..packages import six
from .wait import NoWayToWaitForSocketError, wait_for_read

def is_connection_dropped(conn):
    """
    Returns True if the connection is dropped and should be closed.

    :param conn:
        :class:`http.client.HTTPConnection` object.

    Note: For platforms like AppEngine, this will always return ``False`` to
    let the platform handle connection recycling transparently for us.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('urllib3.util.connection.is_connection_dropped', 'is_connection_dropped(conn)', {'wait_for_read': wait_for_read, 'NoWayToWaitForSocketError': NoWayToWaitForSocketError, 'conn': conn}, 1)

def create_connection(address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, source_address=None, socket_options=None):
    """Connect to *address* and return the socket object.

    Convenience function.  Connect to *address* (a 2-tuple ``(host,
    port)``) and return the socket object.  Passing the optional
    *timeout* parameter will set the timeout on the socket instance
    before attempting to connect.  If no *timeout* is supplied, the
    global default timeout setting returned by :func:`socket.getdefaulttimeout`
    is used.  If *source_address* is set it must be a tuple of (host, port)
    for the socket to bind as a source address before making the connection.
    An host of '' or port 0 tells the OS to use the default.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('urllib3.util.connection.create_connection', 'create_connection(address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, source_address=None, socket_options=None)', {'allowed_gai_family': allowed_gai_family, 'six': six, 'LocationParseError': LocationParseError, 'socket': socket, '_set_socket_options': _set_socket_options, 'address': address, 'timeout': timeout, 'source_address': source_address, 'socket_options': socket_options}, 1)

def _set_socket_options(sock, options):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('urllib3.util.connection._set_socket_options', '_set_socket_options(sock, options)', {'sock': sock, 'options': options}, 1)

def allowed_gai_family():
    """This function is designed to work in the context of
    getaddrinfo, where family=socket.AF_UNSPEC is the default and
    will perform a DNS search for both IPv6 and IPv4 records."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('urllib3.util.connection.allowed_gai_family', 'allowed_gai_family()', {'socket': socket, 'HAS_IPV6': HAS_IPV6}, 1)

def _has_ipv6(host):
    """Returns True if the system can bind an IPv6 address."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('urllib3.util.connection._has_ipv6', '_has_ipv6(host)', {'_appengine_environ': _appengine_environ, 'socket': socket, 'host': host}, 1)
HAS_IPV6 = _has_ipv6('::1')

