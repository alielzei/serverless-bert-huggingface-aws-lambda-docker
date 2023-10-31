from __future__ import absolute_import
import hmac
import os
import sys
import warnings
from binascii import hexlify, unhexlify
from hashlib import md5, sha1, sha256
from ..exceptions import InsecurePlatformWarning, ProxySchemeUnsupported, SNIMissingWarning, SSLError
from ..packages import six
from .url import BRACELESS_IPV6_ADDRZ_RE, IPV4_RE
SSLContext = None
SSLTransport = None
HAS_SNI = False
IS_PYOPENSSL = False
IS_SECURETRANSPORT = False
ALPN_PROTOCOLS = ['http/1.1']
HASHFUNC_MAP = {32: md5, 40: sha1, 64: sha256}

def _const_compare_digest_backport(a, b):
    """
    Compare two digests of equal length in constant time.

    The digests must be of type str/bytes.
    Returns True if the digests match, and False otherwise.
    """
    result = abs(len(a) - len(b))
    for (left, right) in zip(bytearray(a), bytearray(b)):
        result |= left ^ right
    return result == 0
_const_compare_digest = getattr(hmac, 'compare_digest', _const_compare_digest_backport)
try:
    import ssl
    from ssl import CERT_REQUIRED, wrap_socket
except ImportError:
    pass
try:
    from ssl import HAS_SNI
except ImportError:
    pass
try:
    from .ssltransport import SSLTransport
except ImportError:
    pass
try:
    from ssl import PROTOCOL_TLS
    PROTOCOL_SSLv23 = PROTOCOL_TLS
except ImportError:
    try:
        from ssl import PROTOCOL_SSLv23 as PROTOCOL_TLS
        PROTOCOL_SSLv23 = PROTOCOL_TLS
    except ImportError:
        PROTOCOL_SSLv23 = PROTOCOL_TLS = 2
try:
    from ssl import PROTOCOL_TLS_CLIENT
except ImportError:
    PROTOCOL_TLS_CLIENT = PROTOCOL_TLS
try:
    from ssl import OP_NO_COMPRESSION, OP_NO_SSLv2, OP_NO_SSLv3
except ImportError:
    (OP_NO_SSLv2, OP_NO_SSLv3) = (16777216, 33554432)
    OP_NO_COMPRESSION = 131072
try:
    from ssl import OP_NO_TICKET
except ImportError:
    OP_NO_TICKET = 16384
DEFAULT_CIPHERS = ':'.join(['ECDHE+AESGCM', 'ECDHE+CHACHA20', 'DHE+AESGCM', 'DHE+CHACHA20', 'ECDH+AESGCM', 'DH+AESGCM', 'ECDH+AES', 'DH+AES', 'RSA+AESGCM', 'RSA+AES', '!aNULL', '!eNULL', '!MD5', '!DSS'])
try:
    from ssl import SSLContext
except ImportError:
    
    
    class SSLContext(object):
        
        def __init__(self, protocol_version):
            self.protocol = protocol_version
            self.check_hostname = False
            self.verify_mode = ssl.CERT_NONE
            self.ca_certs = None
            self.options = 0
            self.certfile = None
            self.keyfile = None
            self.ciphers = None
        
        def load_cert_chain(self, certfile, keyfile):
            self.certfile = certfile
            self.keyfile = keyfile
        
        def load_verify_locations(self, cafile=None, capath=None, cadata=None):
            self.ca_certs = cafile
            if capath is not None:
                raise SSLError('CA directories not supported in older Pythons')
            if cadata is not None:
                raise SSLError('CA data not supported in older Pythons')
        
        def set_ciphers(self, cipher_suite):
            self.ciphers = cipher_suite
        
        def wrap_socket(self, socket, server_hostname=None, server_side=False):
            warnings.warn('A true SSLContext object is not available. This prevents urllib3 from configuring SSL appropriately and may cause certain SSL connections to fail. You can upgrade to a newer version of Python to solve this. For more information, see https://urllib3.readthedocs.io/en/1.26.x/advanced-usage.html#ssl-warnings', InsecurePlatformWarning)
            kwargs = {'keyfile': self.keyfile, 'certfile': self.certfile, 'ca_certs': self.ca_certs, 'cert_reqs': self.verify_mode, 'ssl_version': self.protocol, 'server_side': server_side}
            return wrap_socket(socket, ciphers=self.ciphers, **kwargs)
    

def assert_fingerprint(cert, fingerprint):
    """
    Checks if given fingerprint matches the supplied certificate.

    :param cert:
        Certificate as bytes object.
    :param fingerprint:
        Fingerprint as string of hexdigits, can be interspersed by colons.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('urllib3.util.ssl_.assert_fingerprint', 'assert_fingerprint(cert, fingerprint)', {'HASHFUNC_MAP': HASHFUNC_MAP, 'SSLError': SSLError, 'unhexlify': unhexlify, '_const_compare_digest': _const_compare_digest, 'hexlify': hexlify, 'cert': cert, 'fingerprint': fingerprint}, 0)

def resolve_cert_reqs(candidate):
    """
    Resolves the argument to a numeric constant, which can be passed to
    the wrap_socket function/method from the ssl module.
    Defaults to :data:`ssl.CERT_REQUIRED`.
    If given a string it is assumed to be the name of the constant in the
    :mod:`ssl` module or its abbreviation.
    (So you can specify `REQUIRED` instead of `CERT_REQUIRED`.
    If it's neither `None` nor a string we assume it is already the numeric
    constant which can directly be passed to wrap_socket.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('urllib3.util.ssl_.resolve_cert_reqs', 'resolve_cert_reqs(candidate)', {'CERT_REQUIRED': CERT_REQUIRED, 'ssl': ssl, 'candidate': candidate}, 1)

def resolve_ssl_version(candidate):
    """
    like resolve_cert_reqs
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('urllib3.util.ssl_.resolve_ssl_version', 'resolve_ssl_version(candidate)', {'PROTOCOL_TLS': PROTOCOL_TLS, 'ssl': ssl, 'candidate': candidate}, 1)

def create_urllib3_context(ssl_version=None, cert_reqs=None, options=None, ciphers=None):
    """All arguments have the same meaning as ``ssl_wrap_socket``.

    By default, this function does a lot of the same work that
    ``ssl.create_default_context`` does on Python 3.4+. It:

    - Disables SSLv2, SSLv3, and compression
    - Sets a restricted set of server ciphers

    If you wish to enable SSLv3, you can do::

        from urllib3.util import ssl_
        context = ssl_.create_urllib3_context()
        context.options &= ~ssl_.OP_NO_SSLv3

    You can do the same to enable compression (substituting ``COMPRESSION``
    for ``SSLv3`` in the last line above).

    :param ssl_version:
        The desired protocol version to use. This will default to
        PROTOCOL_SSLv23 which will negotiate the highest protocol that both
        the server and your installation of OpenSSL support.
    :param cert_reqs:
        Whether to require the certificate verification. This defaults to
        ``ssl.CERT_REQUIRED``.
    :param options:
        Specific OpenSSL options. These default to ``ssl.OP_NO_SSLv2``,
        ``ssl.OP_NO_SSLv3``, ``ssl.OP_NO_COMPRESSION``, and ``ssl.OP_NO_TICKET``.
    :param ciphers:
        Which cipher suites to allow the server to select.
    :returns:
        Constructed SSLContext object with specified options
    :rtype: SSLContext
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('urllib3.util.ssl_.create_urllib3_context', 'create_urllib3_context(ssl_version=None, cert_reqs=None, options=None, ciphers=None)', {'PROTOCOL_TLS': PROTOCOL_TLS, 'PROTOCOL_TLS_CLIENT': PROTOCOL_TLS_CLIENT, 'SSLContext': SSLContext, 'DEFAULT_CIPHERS': DEFAULT_CIPHERS, 'ssl': ssl, 'OP_NO_SSLv2': OP_NO_SSLv2, 'OP_NO_SSLv3': OP_NO_SSLv3, 'OP_NO_COMPRESSION': OP_NO_COMPRESSION, 'OP_NO_TICKET': OP_NO_TICKET, 'sys': sys, 'os': os, 'ssl_version': ssl_version, 'cert_reqs': cert_reqs, 'options': options, 'ciphers': ciphers}, 1)

def ssl_wrap_socket(sock, keyfile=None, certfile=None, cert_reqs=None, ca_certs=None, server_hostname=None, ssl_version=None, ciphers=None, ssl_context=None, ca_cert_dir=None, key_password=None, ca_cert_data=None, tls_in_tls=False):
    """
    All arguments except for server_hostname, ssl_context, and ca_cert_dir have
    the same meaning as they do when using :func:`ssl.wrap_socket`.

    :param server_hostname:
        When SNI is supported, the expected hostname of the certificate
    :param ssl_context:
        A pre-made :class:`SSLContext` object. If none is provided, one will
        be created using :func:`create_urllib3_context`.
    :param ciphers:
        A string of ciphers we wish the client to support.
    :param ca_cert_dir:
        A directory containing CA certificates in multiple separate files, as
        supported by OpenSSL's -CApath flag or the capath argument to
        SSLContext.load_verify_locations().
    :param key_password:
        Optional password if the keyfile is encrypted.
    :param ca_cert_data:
        Optional string containing CA certificates in PEM format suitable for
        passing as the cadata parameter to SSLContext.load_verify_locations()
    :param tls_in_tls:
        Use SSLTransport to wrap the existing socket.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('urllib3.util.ssl_.ssl_wrap_socket', 'ssl_wrap_socket(sock, keyfile=None, certfile=None, cert_reqs=None, ca_certs=None, server_hostname=None, ssl_version=None, ciphers=None, ssl_context=None, ca_cert_dir=None, key_password=None, ca_cert_data=None, tls_in_tls=False)', {'create_urllib3_context': create_urllib3_context, 'IOError': IOError, 'SSLError': SSLError, '_is_key_file_encrypted': _is_key_file_encrypted, 'ALPN_PROTOCOLS': ALPN_PROTOCOLS, 'is_ipaddress': is_ipaddress, 'HAS_SNI': HAS_SNI, 'IS_SECURETRANSPORT': IS_SECURETRANSPORT, 'warnings': warnings, 'SNIMissingWarning': SNIMissingWarning, '_ssl_wrap_socket_impl': _ssl_wrap_socket_impl, 'sock': sock, 'keyfile': keyfile, 'certfile': certfile, 'cert_reqs': cert_reqs, 'ca_certs': ca_certs, 'server_hostname': server_hostname, 'ssl_version': ssl_version, 'ciphers': ciphers, 'ssl_context': ssl_context, 'ca_cert_dir': ca_cert_dir, 'key_password': key_password, 'ca_cert_data': ca_cert_data, 'tls_in_tls': tls_in_tls}, 1)

def is_ipaddress(hostname):
    """Detects whether the hostname given is an IPv4 or IPv6 address.
    Also detects IPv6 addresses with Zone IDs.

    :param str hostname: Hostname to examine.
    :return: True if the hostname is an IP address, False otherwise.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('urllib3.util.ssl_.is_ipaddress', 'is_ipaddress(hostname)', {'six': six, 'IPV4_RE': IPV4_RE, 'BRACELESS_IPV6_ADDRZ_RE': BRACELESS_IPV6_ADDRZ_RE, 'hostname': hostname}, 1)

def _is_key_file_encrypted(key_file):
    """Detects if a key file is encrypted or not."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('urllib3.util.ssl_._is_key_file_encrypted', '_is_key_file_encrypted(key_file)', {'key_file': key_file}, 1)

def _ssl_wrap_socket_impl(sock, ssl_context, tls_in_tls, server_hostname=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('urllib3.util.ssl_._ssl_wrap_socket_impl', '_ssl_wrap_socket_impl(sock, ssl_context, tls_in_tls, server_hostname=None)', {'SSLTransport': SSLTransport, 'ProxySchemeUnsupported': ProxySchemeUnsupported, 'sock': sock, 'ssl_context': ssl_context, 'tls_in_tls': tls_in_tls, 'server_hostname': server_hostname}, 1)

