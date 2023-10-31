"""
requests.utils
~~~~~~~~~~~~~~

This module provides utility functions that are used within Requests
that are also useful for external consumption.
"""

import codecs
import contextlib
import io
import os
import re
import socket
import struct
import sys
import tempfile
import warnings
import zipfile
from collections import OrderedDict
from urllib3.util import make_headers, parse_url
from . import certs
from .__version__ import __version__
from ._internal_utils import _HEADER_VALIDATORS_BYTE, _HEADER_VALIDATORS_STR, HEADER_VALIDATORS, to_native_string
from .compat import Mapping, basestring, bytes, getproxies, getproxies_environment, integer_types
from .compat import parse_http_list as _parse_list_header
from .compat import proxy_bypass, proxy_bypass_environment, quote, str, unquote, urlparse, urlunparse
from .cookies import cookiejar_from_dict
from .exceptions import FileModeWarning, InvalidHeader, InvalidURL, UnrewindableBodyError
from .structures import CaseInsensitiveDict
NETRC_FILES = ('.netrc', '_netrc')
DEFAULT_CA_BUNDLE_PATH = certs.where()
DEFAULT_PORTS = {'http': 80, 'https': 443}
DEFAULT_ACCEPT_ENCODING = ', '.join(re.split(',\\s*', make_headers(accept_encoding=True)['accept-encoding']))
if sys.platform == 'win32':
    
    def proxy_bypass_registry(host):
        import custom_funtemplate
        return custom_funtemplate.rewrite_template('requests.utils.proxy_bypass_registry', 'proxy_bypass_registry(host)', {'re': re, 'host': host}, 1)
    
    def proxy_bypass(host):
        """Return True, if the host should be bypassed.

        Checks proxy settings gathered from the environment, if specified,
        or the registry.
        """
        import custom_funtemplate
        return custom_funtemplate.rewrite_template('requests.utils.proxy_bypass', 'proxy_bypass(host)', {'getproxies_environment': getproxies_environment, 'proxy_bypass_environment': proxy_bypass_environment, 'proxy_bypass_registry': proxy_bypass_registry, 'host': host}, 1)

def dict_to_sequence(d):
    """Returns an internal sequence dictionary update."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('requests.utils.dict_to_sequence', 'dict_to_sequence(d)', {'d': d}, 1)

def super_len(o):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('requests.utils.super_len', 'super_len(o)', {'io': io, 'os': os, 'warnings': warnings, 'FileModeWarning': FileModeWarning, 'o': o}, 1)

def get_netrc_auth(url, raise_errors=False):
    """Returns the Requests tuple auth for a given url from netrc."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('requests.utils.get_netrc_auth', 'get_netrc_auth(url, raise_errors=False)', {'os': os, 'NETRC_FILES': NETRC_FILES, 'urlparse': urlparse, 'url': url, 'raise_errors': raise_errors}, 1)

def guess_filename(obj):
    """Tries to guess the filename of the given object."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('requests.utils.guess_filename', 'guess_filename(obj)', {'basestring': basestring, 'os': os, 'obj': obj}, 1)

def extract_zipped_paths(path):
    """Replace nonexistent paths that look like they refer to a member of a zip
    archive with the location of an extracted copy of the target, or else
    just return the provided path unchanged.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('requests.utils.extract_zipped_paths', 'extract_zipped_paths(path)', {'os': os, 'zipfile': zipfile, 'tempfile': tempfile, 'atomic_open': atomic_open, 'path': path}, 1)

@contextlib.contextmanager
def atomic_open(filename):
    """Write a file to the disk in an atomic fashion"""
    import custom_funtemplate
    custom_funtemplate.rewrite_template('requests.utils.atomic_open', 'atomic_open(filename)', {'tempfile': tempfile, 'os': os, 'contextlib': contextlib, 'filename': filename}, 0)

def from_key_val_list(value):
    """Take an object and test to see if it can be represented as a
    dictionary. Unless it can not be represented as such, return an
    OrderedDict, e.g.,

    ::

        >>> from_key_val_list([('key', 'val')])
        OrderedDict([('key', 'val')])
        >>> from_key_val_list('string')
        Traceback (most recent call last):
        ...
        ValueError: cannot encode objects that are not 2-tuples
        >>> from_key_val_list({'key': 'val'})
        OrderedDict([('key', 'val')])

    :rtype: OrderedDict
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('requests.utils.from_key_val_list', 'from_key_val_list(value)', {'OrderedDict': OrderedDict, 'value': value}, 1)

def to_key_val_list(value):
    """Take an object and test to see if it can be represented as a
    dictionary. If it can be, return a list of tuples, e.g.,

    ::

        >>> to_key_val_list([('key', 'val')])
        [('key', 'val')]
        >>> to_key_val_list({'key': 'val'})
        [('key', 'val')]
        >>> to_key_val_list('string')
        Traceback (most recent call last):
        ...
        ValueError: cannot encode objects that are not 2-tuples

    :rtype: list
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('requests.utils.to_key_val_list', 'to_key_val_list(value)', {'Mapping': Mapping, 'value': value}, 1)

def parse_list_header(value):
    """Parse lists as described by RFC 2068 Section 2.

    In particular, parse comma-separated lists where the elements of
    the list may include quoted-strings.  A quoted-string could
    contain a comma.  A non-quoted string could have quotes in the
    middle.  Quotes are removed automatically after parsing.

    It basically works like :func:`parse_set_header` just that items
    may appear multiple times and case sensitivity is preserved.

    The return value is a standard :class:`list`:

    >>> parse_list_header('token, "quoted value"')
    ['token', 'quoted value']

    To create a header from the :class:`list` again, use the
    :func:`dump_header` function.

    :param value: a string with a list header.
    :return: :class:`list`
    :rtype: list
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('requests.utils.parse_list_header', 'parse_list_header(value)', {'_parse_list_header': _parse_list_header, 'unquote_header_value': unquote_header_value, 'value': value}, 1)

def parse_dict_header(value):
    """Parse lists of key, value pairs as described by RFC 2068 Section 2 and
    convert them into a python dict:

    >>> d = parse_dict_header('foo="is a fish", bar="as well"')
    >>> type(d) is dict
    True
    >>> sorted(d.items())
    [('bar', 'as well'), ('foo', 'is a fish')]

    If there is no value for a key it will be `None`:

    >>> parse_dict_header('key_without_value')
    {'key_without_value': None}

    To create a header from the :class:`dict` again, use the
    :func:`dump_header` function.

    :param value: a string with a dict header.
    :return: :class:`dict`
    :rtype: dict
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('requests.utils.parse_dict_header', 'parse_dict_header(value)', {'_parse_list_header': _parse_list_header, 'unquote_header_value': unquote_header_value, 'value': value}, 1)

def unquote_header_value(value, is_filename=False):
    """Unquotes a header value.  (Reversal of :func:`quote_header_value`).
    This does not use the real unquoting but what browsers are actually
    using for quoting.

    :param value: the header value to unquote.
    :rtype: str
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('requests.utils.unquote_header_value', 'unquote_header_value(value, is_filename=False)', {'value': value, 'is_filename': is_filename}, 1)

def dict_from_cookiejar(cj):
    """Returns a key/value dictionary from a CookieJar.

    :param cj: CookieJar object to extract cookies from.
    :rtype: dict
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('requests.utils.dict_from_cookiejar', 'dict_from_cookiejar(cj)', {'cj': cj}, 1)

def add_dict_to_cookiejar(cj, cookie_dict):
    """Returns a CookieJar from a key/value dictionary.

    :param cj: CookieJar to insert cookies into.
    :param cookie_dict: Dict of key/values to insert into CookieJar.
    :rtype: CookieJar
    """
    return cookiejar_from_dict(cookie_dict, cj)

def get_encodings_from_content(content):
    """Returns encodings from given content string.

    :param content: bytestring to extract encodings from.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('requests.utils.get_encodings_from_content', 'get_encodings_from_content(content)', {'warnings': warnings, 're': re, 'content': content}, 1)

def _parse_content_type_header(header):
    """Returns content type and parameters from given header

    :param header: string
    :return: tuple containing content type and dictionary of
         parameters
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('requests.utils._parse_content_type_header', '_parse_content_type_header(header)', {'header': header}, 2)

def get_encoding_from_headers(headers):
    """Returns encodings from given HTTP Header Dict.

    :param headers: dictionary to extract encoding from.
    :rtype: str
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('requests.utils.get_encoding_from_headers', 'get_encoding_from_headers(headers)', {'_parse_content_type_header': _parse_content_type_header, 'headers': headers}, 1)

def stream_decode_response_unicode(iterator, r):
    """Stream decodes an iterator."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('requests.utils.stream_decode_response_unicode', 'stream_decode_response_unicode(iterator, r)', {'codecs': codecs, 'iterator': iterator, 'r': r}, 1)

def iter_slices(string, slice_length):
    """Iterate over slices of a string."""
    import custom_funtemplate
    custom_funtemplate.rewrite_template('requests.utils.iter_slices', 'iter_slices(string, slice_length)', {'string': string, 'slice_length': slice_length}, 0)

def get_unicode_from_response(r):
    """Returns the requested content back in unicode.

    :param r: Response object to get unicode content from.

    Tried:

    1. charset from content-type
    2. fall back and replace all unicode characters

    :rtype: str
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('requests.utils.get_unicode_from_response', 'get_unicode_from_response(r)', {'warnings': warnings, 'get_encoding_from_headers': get_encoding_from_headers, 'r': r}, 1)
UNRESERVED_SET = frozenset('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz' + '0123456789-._~')

def unquote_unreserved(uri):
    """Un-escape any percent-escape sequences in a URI that are unreserved
    characters. This leaves all reserved, illegal and non-ASCII bytes encoded.

    :rtype: str
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('requests.utils.unquote_unreserved', 'unquote_unreserved(uri)', {'InvalidURL': InvalidURL, 'UNRESERVED_SET': UNRESERVED_SET, 'uri': uri}, 1)

def requote_uri(uri):
    """Re-quote the given URI.

    This function passes the given URI through an unquote/quote cycle to
    ensure that it is fully and consistently quoted.

    :rtype: str
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('requests.utils.requote_uri', 'requote_uri(uri)', {'quote': quote, 'unquote_unreserved': unquote_unreserved, 'InvalidURL': InvalidURL, 'uri': uri}, 1)

def address_in_network(ip, net):
    """This function allows you to check if an IP belongs to a network subnet

    Example: returns True if ip = 192.168.1.1 and net = 192.168.1.0/24
             returns False if ip = 192.168.1.1 and net = 192.168.100.0/24

    :rtype: bool
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('requests.utils.address_in_network', 'address_in_network(ip, net)', {'struct': struct, 'socket': socket, 'dotted_netmask': dotted_netmask, 'ip': ip, 'net': net}, 1)

def dotted_netmask(mask):
    """Converts mask from /xx format to xxx.xxx.xxx.xxx

    Example: if mask is 24 function returns 255.255.255.0

    :rtype: str
    """
    bits = 4294967295 ^ (1 << 32 - mask) - 1
    return socket.inet_ntoa(struct.pack('>I', bits))

def is_ipv4_address(string_ip):
    """
    :rtype: bool
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('requests.utils.is_ipv4_address', 'is_ipv4_address(string_ip)', {'socket': socket, 'string_ip': string_ip}, 1)

def is_valid_cidr(string_network):
    """
    Very simple check of the cidr format in no_proxy variable.

    :rtype: bool
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('requests.utils.is_valid_cidr', 'is_valid_cidr(string_network)', {'socket': socket, 'string_network': string_network}, 1)

@contextlib.contextmanager
def set_environ(env_name, value):
    """Set the environment variable 'env_name' to 'value'

    Save previous value, yield, and then restore the previous value stored in
    the environment variable 'env_name'.

    If 'value' is None, do nothing"""
    import custom_funtemplate
    custom_funtemplate.rewrite_template('requests.utils.set_environ', 'set_environ(env_name, value)', {'os': os, 'contextlib': contextlib, 'env_name': env_name, 'value': value}, 0)

def should_bypass_proxies(url, no_proxy):
    """
    Returns whether we should bypass proxies or not.

    :rtype: bool
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('requests.utils.should_bypass_proxies', 'should_bypass_proxies(url, no_proxy)', {'os': os, 'urlparse': urlparse, 'is_ipv4_address': is_ipv4_address, 'is_valid_cidr': is_valid_cidr, 'address_in_network': address_in_network, 'set_environ': set_environ, 'proxy_bypass': proxy_bypass, 'socket': socket, 'url': url, 'no_proxy': no_proxy}, 1)

def get_environ_proxies(url, no_proxy=None):
    """
    Return a dict of environment proxies.

    :rtype: dict
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('requests.utils.get_environ_proxies', 'get_environ_proxies(url, no_proxy=None)', {'should_bypass_proxies': should_bypass_proxies, 'getproxies': getproxies, 'url': url, 'no_proxy': no_proxy}, 1)

def select_proxy(url, proxies):
    """Select a proxy for the url, if applicable.

    :param url: The url being for the request
    :param proxies: A dictionary of schemes or schemes and hosts to proxy URLs
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('requests.utils.select_proxy', 'select_proxy(url, proxies)', {'urlparse': urlparse, 'url': url, 'proxies': proxies}, 1)

def resolve_proxies(request, proxies, trust_env=True):
    """This method takes proxy information from a request and configuration
    input to resolve a mapping of target proxies. This will consider settings
    such a NO_PROXY to strip proxy configurations.

    :param request: Request or PreparedRequest
    :param proxies: A dictionary of schemes or schemes and hosts to proxy URLs
    :param trust_env: Boolean declaring whether to trust environment configs

    :rtype: dict
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('requests.utils.resolve_proxies', 'resolve_proxies(request, proxies, trust_env=True)', {'urlparse': urlparse, 'should_bypass_proxies': should_bypass_proxies, 'get_environ_proxies': get_environ_proxies, 'request': request, 'proxies': proxies, 'trust_env': trust_env}, 1)

def default_user_agent(name='python-requests'):
    """
    Return a string representing the default user agent.

    :rtype: str
    """
    return f'{name}/{__version__}'

def default_headers():
    """
    :rtype: requests.structures.CaseInsensitiveDict
    """
    return CaseInsensitiveDict({'User-Agent': default_user_agent(), 'Accept-Encoding': DEFAULT_ACCEPT_ENCODING, 'Accept': '*/*', 'Connection': 'keep-alive'})

def parse_header_links(value):
    """Return a list of parsed link headers proxies.

    i.e. Link: <http:/.../front.jpeg>; rel=front; type="image/jpeg",<http://.../back.jpeg>; rel=back;type="image/jpeg"

    :rtype: list
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('requests.utils.parse_header_links', 'parse_header_links(value)', {'re': re, 'value': value}, 1)
_null = '\x00'.encode('ascii')
_null2 = _null * 2
_null3 = _null * 3

def guess_json_utf(data):
    """
    :rtype: str
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('requests.utils.guess_json_utf', 'guess_json_utf(data)', {'codecs': codecs, '_null': _null, '_null2': _null2, '_null3': _null3, 'data': data}, 1)

def prepend_scheme_if_needed(url, new_scheme):
    """Given a URL that may or may not have a scheme, prepend the given scheme.
    Does not replace a present scheme with the one provided as an argument.

    :rtype: str
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('requests.utils.prepend_scheme_if_needed', 'prepend_scheme_if_needed(url, new_scheme)', {'parse_url': parse_url, 'urlunparse': urlunparse, 'url': url, 'new_scheme': new_scheme}, 1)

def get_auth_from_url(url):
    """Given a url with authentication components, extract them into a tuple of
    username,password.

    :rtype: (str,str)
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('requests.utils.get_auth_from_url', 'get_auth_from_url(url)', {'urlparse': urlparse, 'unquote': unquote, 'url': url}, 1)

def check_header_validity(header):
    """Verifies that header parts don't contain leading whitespace
    reserved characters, or return characters.

    :param header: tuple, in the format (name, value).
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('requests.utils.check_header_validity', 'check_header_validity(header)', {'_validate_header_part': _validate_header_part, 'header': header}, 0)

def _validate_header_part(header, header_part, header_validator_index):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('requests.utils._validate_header_part', '_validate_header_part(header, header_part, header_validator_index)', {'_HEADER_VALIDATORS_STR': _HEADER_VALIDATORS_STR, '_HEADER_VALIDATORS_BYTE': _HEADER_VALIDATORS_BYTE, 'InvalidHeader': InvalidHeader, 'header': header, 'header_part': header_part, 'header_validator_index': header_validator_index}, 0)

def urldefragauth(url):
    """
    Given a url remove the fragment and the authentication part.

    :rtype: str
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('requests.utils.urldefragauth', 'urldefragauth(url)', {'urlparse': urlparse, 'urlunparse': urlunparse, 'url': url}, 1)

def rewind_body(prepared_request):
    """Move file pointer back to its recorded starting position
    so it can be read again on redirect.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('requests.utils.rewind_body', 'rewind_body(prepared_request)', {'integer_types': integer_types, 'UnrewindableBodyError': UnrewindableBodyError, 'prepared_request': prepared_request}, 0)

