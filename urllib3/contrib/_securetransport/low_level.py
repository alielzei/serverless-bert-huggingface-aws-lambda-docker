"""
Low-level helpers for the SecureTransport bindings.

These are Python functions that are not directly related to the high-level APIs
but are necessary to get them to work. They include a whole bunch of low-level
CoreFoundation messing about and memory management. The concerns in this module
are almost entirely about trying to avoid memory leaks and providing
appropriate and useful assistance to the higher-level code.
"""

import base64
import ctypes
import itertools
import os
import re
import ssl
import struct
import tempfile
from .bindings import CFConst, CoreFoundation, Security
_PEM_CERTS_RE = re.compile(b'-----BEGIN CERTIFICATE-----\n(.*?)\n-----END CERTIFICATE-----', re.DOTALL)

def _cf_data_from_bytes(bytestring):
    """
    Given a bytestring, create a CFData object from it. This CFData object must
    be CFReleased by the caller.
    """
    return CoreFoundation.CFDataCreate(CoreFoundation.kCFAllocatorDefault, bytestring, len(bytestring))

def _cf_dictionary_from_tuples(tuples):
    """
    Given a list of Python tuples, create an associated CFDictionary.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('urllib3.contrib._securetransport.low_level._cf_dictionary_from_tuples', '_cf_dictionary_from_tuples(tuples)', {'CoreFoundation': CoreFoundation, 'tuples': tuples}, 1)

def _cfstr(py_bstr):
    """
    Given a Python binary data, create a CFString.
    The string must be CFReleased by the caller.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('urllib3.contrib._securetransport.low_level._cfstr', '_cfstr(py_bstr)', {'ctypes': ctypes, 'CoreFoundation': CoreFoundation, 'CFConst': CFConst, 'py_bstr': py_bstr}, 1)

def _create_cfstring_array(lst):
    """
    Given a list of Python binary data, create an associated CFMutableArray.
    The array must be CFReleased by the caller.

    Raises an ssl.SSLError on failure.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('urllib3.contrib._securetransport.low_level._create_cfstring_array', '_create_cfstring_array(lst)', {'CoreFoundation': CoreFoundation, 'ctypes': ctypes, '_cfstr': _cfstr, 'ssl': ssl, 'lst': lst}, 1)

def _cf_string_to_unicode(value):
    """
    Creates a Unicode string from a CFString object. Used entirely for error
    reporting.

    Yes, it annoys me quite a lot that this function is this complex.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('urllib3.contrib._securetransport.low_level._cf_string_to_unicode', '_cf_string_to_unicode(value)', {'ctypes': ctypes, 'CoreFoundation': CoreFoundation, 'CFConst': CFConst, 'value': value}, 1)

def _assert_no_error(error, exception_class=None):
    """
    Checks the return code and throws an exception if there is an error to
    report
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('urllib3.contrib._securetransport.low_level._assert_no_error', '_assert_no_error(error, exception_class=None)', {'Security': Security, '_cf_string_to_unicode': _cf_string_to_unicode, 'CoreFoundation': CoreFoundation, 'ssl': ssl, 'error': error, 'exception_class': exception_class}, 1)

def _cert_array_from_pem(pem_bundle):
    """
    Given a bundle of certs in PEM format, turns them into a CFArray of certs
    that can be used to validate a cert chain.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('urllib3.contrib._securetransport.low_level._cert_array_from_pem', '_cert_array_from_pem(pem_bundle)', {'base64': base64, '_PEM_CERTS_RE': _PEM_CERTS_RE, 'ssl': ssl, 'CoreFoundation': CoreFoundation, 'ctypes': ctypes, '_cf_data_from_bytes': _cf_data_from_bytes, 'Security': Security, 'pem_bundle': pem_bundle}, 1)

def _is_cert(item):
    """
    Returns True if a given CFTypeRef is a certificate.
    """
    expected = Security.SecCertificateGetTypeID()
    return CoreFoundation.CFGetTypeID(item) == expected

def _is_identity(item):
    """
    Returns True if a given CFTypeRef is an identity.
    """
    expected = Security.SecIdentityGetTypeID()
    return CoreFoundation.CFGetTypeID(item) == expected

def _temporary_keychain():
    """
    This function creates a temporary Mac keychain that we can use to work with
    credentials. This keychain uses a one-time password and a temporary file to
    store the data. We expect to have one keychain per socket. The returned
    SecKeychainRef must be freed by the caller, including calling
    SecKeychainDelete.

    Returns a tuple of the SecKeychainRef and the path to the temporary
    directory that contains it.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('urllib3.contrib._securetransport.low_level._temporary_keychain', '_temporary_keychain()', {'os': os, 'base64': base64, 'tempfile': tempfile, 'Security': Security, 'ctypes': ctypes, '_assert_no_error': _assert_no_error}, 2)

def _load_items_from_file(keychain, path):
    """
    Given a single file, loads all the trust objects from it into arrays and
    the keychain.
    Returns a tuple of lists: the first list is a list of identities, the
    second a list of certs.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('urllib3.contrib._securetransport.low_level._load_items_from_file', '_load_items_from_file(keychain, path)', {'CoreFoundation': CoreFoundation, 'Security': Security, 'ctypes': ctypes, '_assert_no_error': _assert_no_error, '_is_cert': _is_cert, '_is_identity': _is_identity, 'keychain': keychain, 'path': path}, 2)

def _load_client_cert_chain(keychain, *paths):
    """
    Load certificates and maybe keys from a number of files. Has the end goal
    of returning a CFArray containing one SecIdentityRef, and then zero or more
    SecCertificateRef objects, suitable for use as a client certificate trust
    chain.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('urllib3.contrib._securetransport.low_level._load_client_cert_chain', '_load_client_cert_chain(keychain, *paths)', {'_load_items_from_file': _load_items_from_file, 'Security': Security, 'ctypes': ctypes, '_assert_no_error': _assert_no_error, 'CoreFoundation': CoreFoundation, 'itertools': itertools, 'keychain': keychain, 'paths': paths}, 1)
TLS_PROTOCOL_VERSIONS = {'SSLv2': (0, 2), 'SSLv3': (3, 0), 'TLSv1': (3, 1), 'TLSv1.1': (3, 2), 'TLSv1.2': (3, 3)}

def _build_tls_unknown_ca_alert(version):
    """
    Builds a TLS alert record for an unknown CA.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('urllib3.contrib._securetransport.low_level._build_tls_unknown_ca_alert', '_build_tls_unknown_ca_alert(version)', {'TLS_PROTOCOL_VERSIONS': TLS_PROTOCOL_VERSIONS, 'struct': struct, 'version': version}, 1)

