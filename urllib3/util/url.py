from __future__ import absolute_import
import re
from collections import namedtuple
from ..exceptions import LocationParseError
from ..packages import six
url_attrs = ['scheme', 'auth', 'host', 'port', 'path', 'query', 'fragment']
NORMALIZABLE_SCHEMES = ('http', 'https', None)
PERCENT_RE = re.compile('%[a-fA-F0-9]{2}')
SCHEME_RE = re.compile('^(?:[a-zA-Z][a-zA-Z0-9+-]*:|/)')
URI_RE = re.compile('^(?:([a-zA-Z][a-zA-Z0-9+.-]*):)?(?://([^\\\\/?#]*))?([^?#]*)(?:\\?([^#]*))?(?:#(.*))?$', re.UNICODE | re.DOTALL)
IPV4_PAT = '(?:[0-9]{1,3}\\.){3}[0-9]{1,3}'
HEX_PAT = '[0-9A-Fa-f]{1,4}'
LS32_PAT = '(?:{hex}:{hex}|{ipv4})'.format(hex=HEX_PAT, ipv4=IPV4_PAT)
_subs = {'hex': HEX_PAT, 'ls32': LS32_PAT}
_variations = ['(?:%(hex)s:){6}%(ls32)s', '::(?:%(hex)s:){5}%(ls32)s', '(?:%(hex)s)?::(?:%(hex)s:){4}%(ls32)s', '(?:(?:%(hex)s:)?%(hex)s)?::(?:%(hex)s:){3}%(ls32)s', '(?:(?:%(hex)s:){0,2}%(hex)s)?::(?:%(hex)s:){2}%(ls32)s', '(?:(?:%(hex)s:){0,3}%(hex)s)?::%(hex)s:%(ls32)s', '(?:(?:%(hex)s:){0,4}%(hex)s)?::%(ls32)s', '(?:(?:%(hex)s:){0,5}%(hex)s)?::%(hex)s', '(?:(?:%(hex)s:){0,6}%(hex)s)?::']
UNRESERVED_PAT = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._!\\-~'
IPV6_PAT = '(?:' + '|'.join([x % _subs for x in _variations]) + ')'
ZONE_ID_PAT = '(?:%25|%)(?:[' + UNRESERVED_PAT + ']|%[a-fA-F0-9]{2})+'
IPV6_ADDRZ_PAT = '\\[' + IPV6_PAT + '(?:' + ZONE_ID_PAT + ')?\\]'
REG_NAME_PAT = '(?:[^\\[\\]%:/?#]|%[a-fA-F0-9]{2})*'
TARGET_RE = re.compile('^(/[^?#]*)(?:\\?([^#]*))?(?:#.*)?$')
IPV4_RE = re.compile('^' + IPV4_PAT + '$')
IPV6_RE = re.compile('^' + IPV6_PAT + '$')
IPV6_ADDRZ_RE = re.compile('^' + IPV6_ADDRZ_PAT + '$')
BRACELESS_IPV6_ADDRZ_RE = re.compile('^' + IPV6_ADDRZ_PAT[2:-2] + '$')
ZONE_ID_RE = re.compile('(' + ZONE_ID_PAT + ')\\]$')
_HOST_PORT_PAT = '^(%s|%s|%s)(?::([0-9]{0,5}))?$' % (REG_NAME_PAT, IPV4_PAT, IPV6_ADDRZ_PAT)
_HOST_PORT_RE = re.compile(_HOST_PORT_PAT, re.UNICODE | re.DOTALL)
UNRESERVED_CHARS = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._-~')
SUB_DELIM_CHARS = set("!$&'()*+,;=")
USERINFO_CHARS = UNRESERVED_CHARS | SUB_DELIM_CHARS | {':'}
PATH_CHARS = USERINFO_CHARS | {'@', '/'}
QUERY_CHARS = FRAGMENT_CHARS = PATH_CHARS | {'?'}


class Url(namedtuple('Url', url_attrs)):
    """
    Data structure for representing an HTTP URL. Used as a return value for
    :func:`parse_url`. Both the scheme and host are normalized as they are
    both case-insensitive according to RFC 3986.
    """
    __slots__ = ()
    
    def __new__(cls, scheme=None, auth=None, host=None, port=None, path=None, query=None, fragment=None):
        if (path and not path.startswith('/')):
            path = '/' + path
        if scheme is not None:
            scheme = scheme.lower()
        return super(Url, cls).__new__(cls, scheme, auth, host, port, path, query, fragment)
    
    @property
    def hostname(self):
        """For backwards-compatibility with urlparse. We're nice like that."""
        return self.host
    
    @property
    def request_uri(self):
        """Absolute path including the query string."""
        uri = (self.path or '/')
        if self.query is not None:
            uri += '?' + self.query
        return uri
    
    @property
    def netloc(self):
        """Network location including host and port"""
        if self.port:
            return '%s:%d' % (self.host, self.port)
        return self.host
    
    @property
    def url(self):
        """
        Convert self into a url

        This function should more or less round-trip with :func:`.parse_url`. The
        returned url may not be exactly the same as the url inputted to
        :func:`.parse_url`, but it should be equivalent by the RFC (e.g., urls
        with a blank port will have : removed).

        Example: ::

            >>> U = parse_url('http://google.com/mail/')
            >>> U.url
            'http://google.com/mail/'
            >>> Url('http', 'username:password', 'host.com', 80,
            ... '/path', 'query', 'fragment').url
            'http://username:password@host.com:80/path?query#fragment'
        """
        (scheme, auth, host, port, path, query, fragment) = self
        url = ''
        if scheme is not None:
            url += scheme + '://'
        if auth is not None:
            url += auth + '@'
        if host is not None:
            url += host
        if port is not None:
            url += ':' + str(port)
        if path is not None:
            url += path
        if query is not None:
            url += '?' + query
        if fragment is not None:
            url += '#' + fragment
        return url
    
    def __str__(self):
        return self.url


def split_first(s, delims):
    """
    .. deprecated:: 1.25

    Given a string and an iterable of delimiters, split on the first found
    delimiter. Return two split parts and the matched delimiter.

    If not found, then the first part is the full input string.

    Example::

        >>> split_first('foo/bar?baz', '?/=')
        ('foo', 'bar?baz', '/')
        >>> split_first('foo/bar?baz', '123')
        ('foo/bar?baz', '', None)

    Scales linearly with number of delims. Not ideal for large number of delims.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('urllib3.util.url.split_first', 'split_first(s, delims)', {'s': s, 'delims': delims}, 3)

def _encode_invalid_chars(component, allowed_chars, encoding='utf-8'):
    """Percent-encodes a URI component without reapplying
    onto an already percent-encoded component.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('urllib3.util.url._encode_invalid_chars', "_encode_invalid_chars(component, allowed_chars, encoding='utf-8')", {'six': six, 'PERCENT_RE': PERCENT_RE, 'component': component, 'allowed_chars': allowed_chars, 'encoding': encoding}, 1)

def _remove_path_dot_segments(path):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('urllib3.util.url._remove_path_dot_segments', '_remove_path_dot_segments(path)', {'path': path}, 1)

def _normalize_host(host, scheme):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('urllib3.util.url._normalize_host', '_normalize_host(host, scheme)', {'six': six, 'NORMALIZABLE_SCHEMES': NORMALIZABLE_SCHEMES, 'IPV6_ADDRZ_RE': IPV6_ADDRZ_RE, 'ZONE_ID_RE': ZONE_ID_RE, '_encode_invalid_chars': _encode_invalid_chars, 'UNRESERVED_CHARS': UNRESERVED_CHARS, 'IPV4_RE': IPV4_RE, '_idna_encode': _idna_encode, 'host': host, 'scheme': scheme}, 1)

def _idna_encode(name):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('urllib3.util.url._idna_encode', '_idna_encode(name)', {'six': six, 'LocationParseError': LocationParseError, 'name': name}, 1)

def _encode_target(target):
    """Percent-encodes a request target so that there are no invalid characters"""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('urllib3.util.url._encode_target', '_encode_target(target)', {'TARGET_RE': TARGET_RE, '_encode_invalid_chars': _encode_invalid_chars, 'PATH_CHARS': PATH_CHARS, 'QUERY_CHARS': QUERY_CHARS, 'target': target}, 1)

def parse_url(url):
    """
    Given a url, return a parsed :class:`.Url` namedtuple. Best-effort is
    performed to parse incomplete urls. Fields not provided will be None.
    This parser is RFC 3986 compliant.

    The parser logic and helper functions are based heavily on
    work done in the ``rfc3986`` module.

    :param str url: URL to parse into a :class:`.Url` namedtuple.

    Partly backwards-compatible with :mod:`urlparse`.

    Example::

        >>> parse_url('http://google.com/mail/')
        Url(scheme='http', host='google.com', port=None, path='/mail/', ...)
        >>> parse_url('google.com:80')
        Url(scheme=None, host='google.com', port=80, path=None, ...)
        >>> parse_url('/foo?bar')
        Url(scheme=None, host=None, port=None, path='/foo', query='bar', ...)
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('urllib3.util.url.parse_url', 'parse_url(url)', {'Url': Url, 'SCHEME_RE': SCHEME_RE, 'URI_RE': URI_RE, 'NORMALIZABLE_SCHEMES': NORMALIZABLE_SCHEMES, '_HOST_PORT_RE': _HOST_PORT_RE, '_encode_invalid_chars': _encode_invalid_chars, 'USERINFO_CHARS': USERINFO_CHARS, 'LocationParseError': LocationParseError, '_normalize_host': _normalize_host, '_remove_path_dot_segments': _remove_path_dot_segments, 'PATH_CHARS': PATH_CHARS, 'QUERY_CHARS': QUERY_CHARS, 'FRAGMENT_CHARS': FRAGMENT_CHARS, 'six': six, 'url': url}, 1)

def get_host(url):
    """
    Deprecated. Use :func:`parse_url` instead.
    """
    p = parse_url(url)
    return ((p.scheme or 'http'), p.hostname, p.port)

