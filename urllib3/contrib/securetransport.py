"""
SecureTranport support for urllib3 via ctypes.

This makes platform-native TLS available to urllib3 users on macOS without the
use of a compiler. This is an important feature because the Python Package
Index is moving to become a TLSv1.2-or-higher server, and the default OpenSSL
that ships with macOS is not capable of doing TLSv1.2. The only way to resolve
this is to give macOS users an alternative solution to the problem, and that
solution is to use SecureTransport.

We use ctypes here because this solution must not require a compiler. That's
because pip is not allowed to require a compiler either.

This is not intended to be a seriously long-term solution to this problem.
The hope is that PEP 543 will eventually solve this issue for us, at which
point we can retire this contrib module. But in the short term, we need to
solve the impending tire fire that is Python on Mac without this kind of
contrib module. So...here we are.

To use this module, simply import and inject it::

    import urllib3.contrib.securetransport
    urllib3.contrib.securetransport.inject_into_urllib3()

Happy TLSing!

This code is a bastardised version of the code found in Will Bond's oscrypto
library. An enormous debt is owed to him for blazing this trail for us. For
that reason, this code should be considered to be covered both by urllib3's
license and by oscrypto's:

.. code-block::

    Copyright (c) 2015-2016 Will Bond <will@wbond.net>

    Permission is hereby granted, free of charge, to any person obtaining a
    copy of this software and associated documentation files (the "Software"),
    to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense,
    and/or sell copies of the Software, and to permit persons to whom the
    Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in
    all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.
"""

from __future__ import absolute_import
import contextlib
import ctypes
import errno
import os.path
import shutil
import socket
import ssl
import struct
import threading
import weakref
import six
from .. import util
from ..util.ssl_ import PROTOCOL_TLS_CLIENT
from ._securetransport.bindings import CoreFoundation, Security, SecurityConst
from ._securetransport.low_level import _assert_no_error, _build_tls_unknown_ca_alert, _cert_array_from_pem, _create_cfstring_array, _load_client_cert_chain, _temporary_keychain
try:
    from socket import _fileobject
except ImportError:
    _fileobject = None
    from ..packages.backports.makefile import backport_makefile
__all__ = ['inject_into_urllib3', 'extract_from_urllib3']
HAS_SNI = True
orig_util_HAS_SNI = util.HAS_SNI
orig_util_SSLContext = util.ssl_.SSLContext
_connection_refs = weakref.WeakValueDictionary()
_connection_ref_lock = threading.Lock()
SSL_WRITE_BLOCKSIZE = 16384
CIPHER_SUITES = [SecurityConst.TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384, SecurityConst.TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256, SecurityConst.TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384, SecurityConst.TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256, SecurityConst.TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256, SecurityConst.TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256, SecurityConst.TLS_DHE_RSA_WITH_AES_256_GCM_SHA384, SecurityConst.TLS_DHE_RSA_WITH_AES_128_GCM_SHA256, SecurityConst.TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA384, SecurityConst.TLS_ECDHE_ECDSA_WITH_AES_256_CBC_SHA, SecurityConst.TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA256, SecurityConst.TLS_ECDHE_ECDSA_WITH_AES_128_CBC_SHA, SecurityConst.TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA384, SecurityConst.TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA, SecurityConst.TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA256, SecurityConst.TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA, SecurityConst.TLS_DHE_RSA_WITH_AES_256_CBC_SHA256, SecurityConst.TLS_DHE_RSA_WITH_AES_256_CBC_SHA, SecurityConst.TLS_DHE_RSA_WITH_AES_128_CBC_SHA256, SecurityConst.TLS_DHE_RSA_WITH_AES_128_CBC_SHA, SecurityConst.TLS_AES_256_GCM_SHA384, SecurityConst.TLS_AES_128_GCM_SHA256, SecurityConst.TLS_RSA_WITH_AES_256_GCM_SHA384, SecurityConst.TLS_RSA_WITH_AES_128_GCM_SHA256, SecurityConst.TLS_AES_128_CCM_8_SHA256, SecurityConst.TLS_AES_128_CCM_SHA256, SecurityConst.TLS_RSA_WITH_AES_256_CBC_SHA256, SecurityConst.TLS_RSA_WITH_AES_128_CBC_SHA256, SecurityConst.TLS_RSA_WITH_AES_256_CBC_SHA, SecurityConst.TLS_RSA_WITH_AES_128_CBC_SHA]
_protocol_to_min_max = {util.PROTOCOL_TLS: (SecurityConst.kTLSProtocol1, SecurityConst.kTLSProtocol12), PROTOCOL_TLS_CLIENT: (SecurityConst.kTLSProtocol1, SecurityConst.kTLSProtocol12)}
if hasattr(ssl, 'PROTOCOL_SSLv2'):
    _protocol_to_min_max[ssl.PROTOCOL_SSLv2] = (SecurityConst.kSSLProtocol2, SecurityConst.kSSLProtocol2)
if hasattr(ssl, 'PROTOCOL_SSLv3'):
    _protocol_to_min_max[ssl.PROTOCOL_SSLv3] = (SecurityConst.kSSLProtocol3, SecurityConst.kSSLProtocol3)
if hasattr(ssl, 'PROTOCOL_TLSv1'):
    _protocol_to_min_max[ssl.PROTOCOL_TLSv1] = (SecurityConst.kTLSProtocol1, SecurityConst.kTLSProtocol1)
if hasattr(ssl, 'PROTOCOL_TLSv1_1'):
    _protocol_to_min_max[ssl.PROTOCOL_TLSv1_1] = (SecurityConst.kTLSProtocol11, SecurityConst.kTLSProtocol11)
if hasattr(ssl, 'PROTOCOL_TLSv1_2'):
    _protocol_to_min_max[ssl.PROTOCOL_TLSv1_2] = (SecurityConst.kTLSProtocol12, SecurityConst.kTLSProtocol12)

def inject_into_urllib3():
    """
    Monkey-patch urllib3 with SecureTransport-backed SSL-support.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('urllib3.contrib.securetransport.inject_into_urllib3', 'inject_into_urllib3()', {'util': util, 'SecureTransportContext': SecureTransportContext, 'HAS_SNI': HAS_SNI}, 0)

def extract_from_urllib3():
    """
    Undo monkey-patching by :func:`inject_into_urllib3`.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('urllib3.contrib.securetransport.extract_from_urllib3', 'extract_from_urllib3()', {'util': util, 'orig_util_SSLContext': orig_util_SSLContext, 'orig_util_HAS_SNI': orig_util_HAS_SNI}, 0)

def _read_callback(connection_id, data_buffer, data_length_pointer):
    """
    SecureTransport read callback. This is called by ST to request that data
    be returned from the socket.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('urllib3.contrib.securetransport._read_callback', '_read_callback(connection_id, data_buffer, data_length_pointer)', {'_connection_refs': _connection_refs, 'SecurityConst': SecurityConst, 'util': util, 'socket': socket, 'errno': errno, 'ctypes': ctypes, 'connection_id': connection_id, 'data_buffer': data_buffer, 'data_length_pointer': data_length_pointer}, 1)

def _write_callback(connection_id, data_buffer, data_length_pointer):
    """
    SecureTransport write callback. This is called by ST to request that data
    actually be sent on the network.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('urllib3.contrib.securetransport._write_callback', '_write_callback(connection_id, data_buffer, data_length_pointer)', {'_connection_refs': _connection_refs, 'SecurityConst': SecurityConst, 'ctypes': ctypes, 'util': util, 'socket': socket, 'errno': errno, 'connection_id': connection_id, 'data_buffer': data_buffer, 'data_length_pointer': data_length_pointer}, 1)
_read_callback_pointer = Security.SSLReadFunc(_read_callback)
_write_callback_pointer = Security.SSLWriteFunc(_write_callback)


class WrappedSocket(object):
    """
    API-compatibility wrapper for Python's OpenSSL wrapped socket object.

    Note: _makefile_refs, _drop(), and _reuse() are needed for the garbage
    collector of PyPy.
    """
    
    def __init__(self, socket):
        self.socket = socket
        self.context = None
        self._makefile_refs = 0
        self._closed = False
        self._exception = None
        self._keychain = None
        self._keychain_dir = None
        self._client_cert_chain = None
        self._timeout = self.socket.gettimeout()
        self.socket.settimeout(0)
    
    @contextlib.contextmanager
    def _raise_on_error(self):
        """
        A context manager that can be used to wrap calls that do I/O from
        SecureTransport. If any of the I/O callbacks hit an exception, this
        context manager will correctly propagate the exception after the fact.
        This avoids silently swallowing those exceptions.

        It also correctly forces the socket closed.
        """
        self._exception = None
        yield
        if self._exception is not None:
            (exception, self._exception) = (self._exception, None)
            self.close()
            raise exception
    
    def _set_ciphers(self):
        """
        Sets up the allowed ciphers. By default this matches the set in
        util.ssl_.DEFAULT_CIPHERS, at least as supported by macOS. This is done
        custom and doesn't allow changing at this time, mostly because parsing
        OpenSSL cipher strings is going to be a freaking nightmare.
        """
        ciphers = (Security.SSLCipherSuite * len(CIPHER_SUITES))(*CIPHER_SUITES)
        result = Security.SSLSetEnabledCiphers(self.context, ciphers, len(CIPHER_SUITES))
        _assert_no_error(result)
    
    def _set_alpn_protocols(self, protocols):
        """
        Sets up the ALPN protocols on the context.
        """
        if not protocols:
            return
        protocols_arr = _create_cfstring_array(protocols)
        try:
            result = Security.SSLSetALPNProtocols(self.context, protocols_arr)
            _assert_no_error(result)
        finally:
            CoreFoundation.CFRelease(protocols_arr)
    
    def _custom_validate(self, verify, trust_bundle):
        """
        Called when we have set custom validation. We do this in two cases:
        first, when cert validation is entirely disabled; and second, when
        using a custom trust DB.
        Raises an SSLError if the connection is not trusted.
        """
        if not verify:
            return
        successes = (SecurityConst.kSecTrustResultUnspecified, SecurityConst.kSecTrustResultProceed)
        try:
            trust_result = self._evaluate_trust(trust_bundle)
            if trust_result in successes:
                return
            reason = 'error code: %d' % (trust_result, )
        except Exception as e:
            reason = 'exception: %r' % (e, )
        rec = _build_tls_unknown_ca_alert(self.version())
        self.socket.sendall(rec)
        opts = struct.pack('ii', 1, 0)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, opts)
        self.close()
        raise ssl.SSLError('certificate verify failed, %s' % reason)
    
    def _evaluate_trust(self, trust_bundle):
        if os.path.isfile(trust_bundle):
            with open(trust_bundle, 'rb') as f:
                trust_bundle = f.read()
        cert_array = None
        trust = Security.SecTrustRef()
        try:
            cert_array = _cert_array_from_pem(trust_bundle)
            result = Security.SSLCopyPeerTrust(self.context, ctypes.byref(trust))
            _assert_no_error(result)
            if not trust:
                raise ssl.SSLError('Failed to copy trust reference')
            result = Security.SecTrustSetAnchorCertificates(trust, cert_array)
            _assert_no_error(result)
            result = Security.SecTrustSetAnchorCertificatesOnly(trust, True)
            _assert_no_error(result)
            trust_result = Security.SecTrustResultType()
            result = Security.SecTrustEvaluate(trust, ctypes.byref(trust_result))
            _assert_no_error(result)
        finally:
            if trust:
                CoreFoundation.CFRelease(trust)
            if cert_array is not None:
                CoreFoundation.CFRelease(cert_array)
        return trust_result.value
    
    def handshake(self, server_hostname, verify, trust_bundle, min_version, max_version, client_cert, client_key, client_key_passphrase, alpn_protocols):
        """
        Actually performs the TLS handshake. This is run automatically by
        wrapped socket, and shouldn't be needed in user code.
        """
        self.context = Security.SSLCreateContext(None, SecurityConst.kSSLClientSide, SecurityConst.kSSLStreamType)
        result = Security.SSLSetIOFuncs(self.context, _read_callback_pointer, _write_callback_pointer)
        _assert_no_error(result)
        with _connection_ref_lock:
            handle = id(self) % 2147483647
            while handle in _connection_refs:
                handle = (handle + 1) % 2147483647
            _connection_refs[handle] = self
        result = Security.SSLSetConnection(self.context, handle)
        _assert_no_error(result)
        if server_hostname:
            if not isinstance(server_hostname, bytes):
                server_hostname = server_hostname.encode('utf-8')
            result = Security.SSLSetPeerDomainName(self.context, server_hostname, len(server_hostname))
            _assert_no_error(result)
        self._set_ciphers()
        self._set_alpn_protocols(alpn_protocols)
        result = Security.SSLSetProtocolVersionMin(self.context, min_version)
        _assert_no_error(result)
        result = Security.SSLSetProtocolVersionMax(self.context, max_version)
        _assert_no_error(result)
        if (not verify or trust_bundle is not None):
            result = Security.SSLSetSessionOption(self.context, SecurityConst.kSSLSessionOptionBreakOnServerAuth, True)
            _assert_no_error(result)
        if client_cert:
            (self._keychain, self._keychain_dir) = _temporary_keychain()
            self._client_cert_chain = _load_client_cert_chain(self._keychain, client_cert, client_key)
            result = Security.SSLSetCertificate(self.context, self._client_cert_chain)
            _assert_no_error(result)
        while True:
            with self._raise_on_error():
                result = Security.SSLHandshake(self.context)
                if result == SecurityConst.errSSLWouldBlock:
                    raise socket.timeout('handshake timed out')
                elif result == SecurityConst.errSSLServerAuthCompleted:
                    self._custom_validate(verify, trust_bundle)
                    continue
                else:
                    _assert_no_error(result)
                    break
    
    def fileno(self):
        return self.socket.fileno()
    
    def _decref_socketios(self):
        if self._makefile_refs > 0:
            self._makefile_refs -= 1
        if self._closed:
            self.close()
    
    def recv(self, bufsiz):
        buffer = ctypes.create_string_buffer(bufsiz)
        bytes_read = self.recv_into(buffer, bufsiz)
        data = buffer[:bytes_read]
        return data
    
    def recv_into(self, buffer, nbytes=None):
        if self._closed:
            return 0
        if nbytes is None:
            nbytes = len(buffer)
        buffer = (ctypes.c_char * nbytes).from_buffer(buffer)
        processed_bytes = ctypes.c_size_t(0)
        with self._raise_on_error():
            result = Security.SSLRead(self.context, buffer, nbytes, ctypes.byref(processed_bytes))
        if result == SecurityConst.errSSLWouldBlock:
            if processed_bytes.value == 0:
                raise socket.timeout('recv timed out')
        elif result in (SecurityConst.errSSLClosedGraceful, SecurityConst.errSSLClosedNoNotify):
            self.close()
        else:
            _assert_no_error(result)
        return processed_bytes.value
    
    def settimeout(self, timeout):
        self._timeout = timeout
    
    def gettimeout(self):
        return self._timeout
    
    def send(self, data):
        processed_bytes = ctypes.c_size_t(0)
        with self._raise_on_error():
            result = Security.SSLWrite(self.context, data, len(data), ctypes.byref(processed_bytes))
        if (result == SecurityConst.errSSLWouldBlock and processed_bytes.value == 0):
            raise socket.timeout('send timed out')
        else:
            _assert_no_error(result)
        return processed_bytes.value
    
    def sendall(self, data):
        total_sent = 0
        while total_sent < len(data):
            sent = self.send(data[total_sent:total_sent + SSL_WRITE_BLOCKSIZE])
            total_sent += sent
    
    def shutdown(self):
        with self._raise_on_error():
            Security.SSLClose(self.context)
    
    def close(self):
        if self._makefile_refs < 1:
            self._closed = True
            if self.context:
                CoreFoundation.CFRelease(self.context)
                self.context = None
            if self._client_cert_chain:
                CoreFoundation.CFRelease(self._client_cert_chain)
                self._client_cert_chain = None
            if self._keychain:
                Security.SecKeychainDelete(self._keychain)
                CoreFoundation.CFRelease(self._keychain)
                shutil.rmtree(self._keychain_dir)
                self._keychain = self._keychain_dir = None
            return self.socket.close()
        else:
            self._makefile_refs -= 1
    
    def getpeercert(self, binary_form=False):
        if not binary_form:
            raise ValueError('SecureTransport only supports dumping binary certs')
        trust = Security.SecTrustRef()
        certdata = None
        der_bytes = None
        try:
            result = Security.SSLCopyPeerTrust(self.context, ctypes.byref(trust))
            _assert_no_error(result)
            if not trust:
                return None
            cert_count = Security.SecTrustGetCertificateCount(trust)
            if not cert_count:
                return None
            leaf = Security.SecTrustGetCertificateAtIndex(trust, 0)
            assert leaf
            certdata = Security.SecCertificateCopyData(leaf)
            assert certdata
            data_length = CoreFoundation.CFDataGetLength(certdata)
            data_buffer = CoreFoundation.CFDataGetBytePtr(certdata)
            der_bytes = ctypes.string_at(data_buffer, data_length)
        finally:
            if certdata:
                CoreFoundation.CFRelease(certdata)
            if trust:
                CoreFoundation.CFRelease(trust)
        return der_bytes
    
    def version(self):
        protocol = Security.SSLProtocol()
        result = Security.SSLGetNegotiatedProtocolVersion(self.context, ctypes.byref(protocol))
        _assert_no_error(result)
        if protocol.value == SecurityConst.kTLSProtocol13:
            raise ssl.SSLError('SecureTransport does not support TLS 1.3')
        elif protocol.value == SecurityConst.kTLSProtocol12:
            return 'TLSv1.2'
        elif protocol.value == SecurityConst.kTLSProtocol11:
            return 'TLSv1.1'
        elif protocol.value == SecurityConst.kTLSProtocol1:
            return 'TLSv1'
        elif protocol.value == SecurityConst.kSSLProtocol3:
            return 'SSLv3'
        elif protocol.value == SecurityConst.kSSLProtocol2:
            return 'SSLv2'
        else:
            raise ssl.SSLError('Unknown TLS version: %r' % protocol)
    
    def _reuse(self):
        self._makefile_refs += 1
    
    def _drop(self):
        if self._makefile_refs < 1:
            self.close()
        else:
            self._makefile_refs -= 1

if _fileobject:
    
    def makefile(self, mode, bufsize=-1):
        self._makefile_refs += 1
        return _fileobject(self, mode, bufsize, close=True)
else:
    
    def makefile(self, mode='r', buffering=None, *args, **kwargs):
        buffering = 0
        return backport_makefile(self, mode, buffering, *args, **kwargs)
WrappedSocket.makefile = makefile


class SecureTransportContext(object):
    """
    I am a wrapper class for the SecureTransport library, to translate the
    interface of the standard library ``SSLContext`` object to calls into
    SecureTransport.
    """
    
    def __init__(self, protocol):
        (self._min_version, self._max_version) = _protocol_to_min_max[protocol]
        self._options = 0
        self._verify = False
        self._trust_bundle = None
        self._client_cert = None
        self._client_key = None
        self._client_key_passphrase = None
        self._alpn_protocols = None
    
    @property
    def check_hostname(self):
        """
        SecureTransport cannot have its hostname checking disabled. For more,
        see the comment on getpeercert() in this file.
        """
        return True
    
    @check_hostname.setter
    def check_hostname(self, value):
        """
        SecureTransport cannot have its hostname checking disabled. For more,
        see the comment on getpeercert() in this file.
        """
        pass
    
    @property
    def options(self):
        return self._options
    
    @options.setter
    def options(self, value):
        self._options = value
    
    @property
    def verify_mode(self):
        return (ssl.CERT_REQUIRED if self._verify else ssl.CERT_NONE)
    
    @verify_mode.setter
    def verify_mode(self, value):
        self._verify = (True if value == ssl.CERT_REQUIRED else False)
    
    def set_default_verify_paths(self):
        pass
    
    def load_default_certs(self):
        return self.set_default_verify_paths()
    
    def set_ciphers(self, ciphers):
        if ciphers != util.ssl_.DEFAULT_CIPHERS:
            raise ValueError("SecureTransport doesn't support custom cipher strings")
    
    def load_verify_locations(self, cafile=None, capath=None, cadata=None):
        if capath is not None:
            raise ValueError('SecureTransport does not support cert directories')
        if cafile is not None:
            with open(cafile):
                pass
        self._trust_bundle = (cafile or cadata)
    
    def load_cert_chain(self, certfile, keyfile=None, password=None):
        self._client_cert = certfile
        self._client_key = keyfile
        self._client_cert_passphrase = password
    
    def set_alpn_protocols(self, protocols):
        """
        Sets the ALPN protocols that will later be set on the context.

        Raises a NotImplementedError if ALPN is not supported.
        """
        if not hasattr(Security, 'SSLSetALPNProtocols'):
            raise NotImplementedError('SecureTransport supports ALPN only in macOS 10.12+')
        self._alpn_protocols = [six.ensure_binary(p) for p in protocols]
    
    def wrap_socket(self, sock, server_side=False, do_handshake_on_connect=True, suppress_ragged_eofs=True, server_hostname=None):
        assert not server_side
        assert do_handshake_on_connect
        assert suppress_ragged_eofs
        wrapped_socket = WrappedSocket(sock)
        wrapped_socket.handshake(server_hostname, self._verify, self._trust_bundle, self._min_version, self._max_version, self._client_cert, self._client_key, self._client_key_passphrase, self._alpn_protocols)
        return wrapped_socket


