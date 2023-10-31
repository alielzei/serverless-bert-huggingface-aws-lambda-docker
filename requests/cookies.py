"""
requests.cookies
~~~~~~~~~~~~~~~~

Compatibility code to be able to use `cookielib.CookieJar` with requests.

requests.utils imports from here, so be careful with imports.
"""

import calendar
import copy
import time
from ._internal_utils import to_native_string
from .compat import Morsel, MutableMapping, cookielib, urlparse, urlunparse
try:
    import threading
except ImportError:
    import dummy_threading as threading


class MockRequest:
    """Wraps a `requests.Request` to mimic a `urllib2.Request`.

    The code in `cookielib.CookieJar` expects this interface in order to correctly
    manage cookie policies, i.e., determine whether a cookie can be set, given the
    domains of the request and the cookie.

    The original request object is read-only. The client is responsible for collecting
    the new headers via `get_new_headers()` and interpreting them appropriately. You
    probably want `get_cookie_header`, defined below.
    """
    
    def __init__(self, request):
        self._r = request
        self._new_headers = {}
        self.type = urlparse(self._r.url).scheme
    
    def get_type(self):
        return self.type
    
    def get_host(self):
        return urlparse(self._r.url).netloc
    
    def get_origin_req_host(self):
        return self.get_host()
    
    def get_full_url(self):
        if not self._r.headers.get('Host'):
            return self._r.url
        host = to_native_string(self._r.headers['Host'], encoding='utf-8')
        parsed = urlparse(self._r.url)
        return urlunparse([parsed.scheme, host, parsed.path, parsed.params, parsed.query, parsed.fragment])
    
    def is_unverifiable(self):
        return True
    
    def has_header(self, name):
        return (name in self._r.headers or name in self._new_headers)
    
    def get_header(self, name, default=None):
        return self._r.headers.get(name, self._new_headers.get(name, default))
    
    def add_header(self, key, val):
        """cookielib has no legitimate use for this method; add it back if you find one."""
        raise NotImplementedError('Cookie headers should be added with add_unredirected_header()')
    
    def add_unredirected_header(self, name, value):
        self._new_headers[name] = value
    
    def get_new_headers(self):
        return self._new_headers
    
    @property
    def unverifiable(self):
        return self.is_unverifiable()
    
    @property
    def origin_req_host(self):
        return self.get_origin_req_host()
    
    @property
    def host(self):
        return self.get_host()



class MockResponse:
    """Wraps a `httplib.HTTPMessage` to mimic a `urllib.addinfourl`.

    ...what? Basically, expose the parsed HTTP headers from the server response
    the way `cookielib` expects to see them.
    """
    
    def __init__(self, headers):
        """Make a MockResponse for `cookielib` to read.

        :param headers: a httplib.HTTPMessage or analogous carrying the headers
        """
        self._headers = headers
    
    def info(self):
        return self._headers
    
    def getheaders(self, name):
        self._headers.getheaders(name)


def extract_cookies_to_jar(jar, request, response):
    """Extract the cookies from the response into a CookieJar.

    :param jar: cookielib.CookieJar (not necessarily a RequestsCookieJar)
    :param request: our own requests.Request object
    :param response: urllib3.HTTPResponse object
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('requests.cookies.extract_cookies_to_jar', 'extract_cookies_to_jar(jar, request, response)', {'MockRequest': MockRequest, 'MockResponse': MockResponse, 'jar': jar, 'request': request, 'response': response}, 1)

def get_cookie_header(jar, request):
    """
    Produce an appropriate Cookie header string to be sent with `request`, or None.

    :rtype: str
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('requests.cookies.get_cookie_header', 'get_cookie_header(jar, request)', {'MockRequest': MockRequest, 'jar': jar, 'request': request}, 1)

def remove_cookie_by_name(cookiejar, name, domain=None, path=None):
    """Unsets a cookie by name, by default over all domains and paths.

    Wraps CookieJar.clear(), is O(n).
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('requests.cookies.remove_cookie_by_name', 'remove_cookie_by_name(cookiejar, name, domain=None, path=None)', {'cookiejar': cookiejar, 'name': name, 'domain': domain, 'path': path}, 0)


class CookieConflictError(RuntimeError):
    """There are two cookies that meet the criteria specified in the cookie jar.
    Use .get and .set and include domain and path args in order to be more specific.
    """
    



class RequestsCookieJar(cookielib.CookieJar, MutableMapping):
    """Compatibility class; is a cookielib.CookieJar, but exposes a dict
    interface.

    This is the CookieJar we create by default for requests and sessions that
    don't specify one, since some clients may expect response.cookies and
    session.cookies to support dict operations.

    Requests does not use the dict interface internally; it's just for
    compatibility with external client code. All requests code should work
    out of the box with externally provided instances of ``CookieJar``, e.g.
    ``LWPCookieJar`` and ``FileCookieJar``.

    Unlike a regular CookieJar, this class is pickleable.

    .. warning:: dictionary operations that are normally O(1) may be O(n).
    """
    
    def get(self, name, default=None, domain=None, path=None):
        """Dict-like get() that also supports optional domain and path args in
        order to resolve naming collisions from using one cookie jar over
        multiple domains.

        .. warning:: operation is O(n), not O(1).
        """
        try:
            return self._find_no_duplicates(name, domain, path)
        except KeyError:
            return default
    
    def set(self, name, value, **kwargs):
        """Dict-like set() that also supports optional domain and path args in
        order to resolve naming collisions from using one cookie jar over
        multiple domains.
        """
        if value is None:
            remove_cookie_by_name(self, name, domain=kwargs.get('domain'), path=kwargs.get('path'))
            return
        if isinstance(value, Morsel):
            c = morsel_to_cookie(value)
        else:
            c = create_cookie(name, value, **kwargs)
        self.set_cookie(c)
        return c
    
    def iterkeys(self):
        """Dict-like iterkeys() that returns an iterator of names of cookies
        from the jar.

        .. seealso:: itervalues() and iteritems().
        """
        for cookie in iter(self):
            yield cookie.name
    
    def keys(self):
        """Dict-like keys() that returns a list of names of cookies from the
        jar.

        .. seealso:: values() and items().
        """
        return list(self.iterkeys())
    
    def itervalues(self):
        """Dict-like itervalues() that returns an iterator of values of cookies
        from the jar.

        .. seealso:: iterkeys() and iteritems().
        """
        for cookie in iter(self):
            yield cookie.value
    
    def values(self):
        """Dict-like values() that returns a list of values of cookies from the
        jar.

        .. seealso:: keys() and items().
        """
        return list(self.itervalues())
    
    def iteritems(self):
        """Dict-like iteritems() that returns an iterator of name-value tuples
        from the jar.

        .. seealso:: iterkeys() and itervalues().
        """
        for cookie in iter(self):
            yield (cookie.name, cookie.value)
    
    def items(self):
        """Dict-like items() that returns a list of name-value tuples from the
        jar. Allows client-code to call ``dict(RequestsCookieJar)`` and get a
        vanilla python dict of key value pairs.

        .. seealso:: keys() and values().
        """
        return list(self.iteritems())
    
    def list_domains(self):
        """Utility method to list all the domains in the jar."""
        domains = []
        for cookie in iter(self):
            if cookie.domain not in domains:
                domains.append(cookie.domain)
        return domains
    
    def list_paths(self):
        """Utility method to list all the paths in the jar."""
        paths = []
        for cookie in iter(self):
            if cookie.path not in paths:
                paths.append(cookie.path)
        return paths
    
    def multiple_domains(self):
        """Returns True if there are multiple domains in the jar.
        Returns False otherwise.

        :rtype: bool
        """
        domains = []
        for cookie in iter(self):
            if (cookie.domain is not None and cookie.domain in domains):
                return True
            domains.append(cookie.domain)
        return False
    
    def get_dict(self, domain=None, path=None):
        """Takes as an argument an optional domain and path and returns a plain
        old Python dict of name-value pairs of cookies that meet the
        requirements.

        :rtype: dict
        """
        dictionary = {}
        for cookie in iter(self):
            if (((domain is None or cookie.domain == domain)) and ((path is None or cookie.path == path))):
                dictionary[cookie.name] = cookie.value
        return dictionary
    
    def __contains__(self, name):
        try:
            return super().__contains__(name)
        except CookieConflictError:
            return True
    
    def __getitem__(self, name):
        """Dict-like __getitem__() for compatibility with client code. Throws
        exception if there are more than one cookie with name. In that case,
        use the more explicit get() method instead.

        .. warning:: operation is O(n), not O(1).
        """
        return self._find_no_duplicates(name)
    
    def __setitem__(self, name, value):
        """Dict-like __setitem__ for compatibility with client code. Throws
        exception if there is already a cookie of that name in the jar. In that
        case, use the more explicit set() method instead.
        """
        self.set(name, value)
    
    def __delitem__(self, name):
        """Deletes a cookie given a name. Wraps ``cookielib.CookieJar``'s
        ``remove_cookie_by_name()``.
        """
        remove_cookie_by_name(self, name)
    
    def set_cookie(self, cookie, *args, **kwargs):
        if (hasattr(cookie.value, 'startswith') and cookie.value.startswith('"') and cookie.value.endswith('"')):
            cookie.value = cookie.value.replace('\\"', '')
        return super().set_cookie(cookie, *args, **kwargs)
    
    def update(self, other):
        """Updates this jar with cookies from another CookieJar or dict-like"""
        if isinstance(other, cookielib.CookieJar):
            for cookie in other:
                self.set_cookie(copy.copy(cookie))
        else:
            super().update(other)
    
    def _find(self, name, domain=None, path=None):
        """Requests uses this method internally to get cookie values.

        If there are conflicting cookies, _find arbitrarily chooses one.
        See _find_no_duplicates if you want an exception thrown if there are
        conflicting cookies.

        :param name: a string containing name of cookie
        :param domain: (optional) string containing domain of cookie
        :param path: (optional) string containing path of cookie
        :return: cookie.value
        """
        for cookie in iter(self):
            if cookie.name == name:
                if (domain is None or cookie.domain == domain):
                    if (path is None or cookie.path == path):
                        return cookie.value
        raise KeyError(f'name={name!r}, domain={domain!r}, path={path!r}')
    
    def _find_no_duplicates(self, name, domain=None, path=None):
        """Both ``__get_item__`` and ``get`` call this function: it's never
        used elsewhere in Requests.

        :param name: a string containing name of cookie
        :param domain: (optional) string containing domain of cookie
        :param path: (optional) string containing path of cookie
        :raises KeyError: if cookie is not found
        :raises CookieConflictError: if there are multiple cookies
            that match name and optionally domain and path
        :return: cookie.value
        """
        toReturn = None
        for cookie in iter(self):
            if cookie.name == name:
                if (domain is None or cookie.domain == domain):
                    if (path is None or cookie.path == path):
                        if toReturn is not None:
                            raise CookieConflictError(f'There are multiple cookies with name, {name!r}')
                        toReturn = cookie.value
        if toReturn:
            return toReturn
        raise KeyError(f'name={name!r}, domain={domain!r}, path={path!r}')
    
    def __getstate__(self):
        """Unlike a normal CookieJar, this class is pickleable."""
        state = self.__dict__.copy()
        state.pop('_cookies_lock')
        return state
    
    def __setstate__(self, state):
        """Unlike a normal CookieJar, this class is pickleable."""
        self.__dict__.update(state)
        if '_cookies_lock' not in self.__dict__:
            self._cookies_lock = threading.RLock()
    
    def copy(self):
        """Return a copy of this RequestsCookieJar."""
        new_cj = RequestsCookieJar()
        new_cj.set_policy(self.get_policy())
        new_cj.update(self)
        return new_cj
    
    def get_policy(self):
        """Return the CookiePolicy instance used."""
        return self._policy


def _copy_cookie_jar(jar):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('requests.cookies._copy_cookie_jar', '_copy_cookie_jar(jar)', {'copy': copy, 'jar': jar}, 1)

def create_cookie(name, value, **kwargs):
    """Make a cookie from underspecified parameters.

    By default, the pair of `name` and `value` will be set for the domain ''
    and sent on every request (this is sometimes called a "supercookie").
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('requests.cookies.create_cookie', 'create_cookie(name, value, **kwargs)', {'cookielib': cookielib, 'name': name, 'value': value, 'kwargs': kwargs}, 1)

def morsel_to_cookie(morsel):
    """Convert a Morsel object into a Cookie containing the one k/v pair."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('requests.cookies.morsel_to_cookie', 'morsel_to_cookie(morsel)', {'time': time, 'calendar': calendar, 'create_cookie': create_cookie, 'morsel': morsel}, 1)

def cookiejar_from_dict(cookie_dict, cookiejar=None, overwrite=True):
    """Returns a CookieJar from a key/value dictionary.

    :param cookie_dict: Dict of key/values to insert into CookieJar.
    :param cookiejar: (optional) A cookiejar to add the cookies to.
    :param overwrite: (optional) If False, will not replace cookies
        already in the jar with new ones.
    :rtype: CookieJar
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('requests.cookies.cookiejar_from_dict', 'cookiejar_from_dict(cookie_dict, cookiejar=None, overwrite=True)', {'RequestsCookieJar': RequestsCookieJar, 'create_cookie': create_cookie, 'cookie_dict': cookie_dict, 'cookiejar': cookiejar, 'overwrite': overwrite}, 1)

def merge_cookies(cookiejar, cookies):
    """Add cookies to cookiejar and returns a merged CookieJar.

    :param cookiejar: CookieJar object to add the cookies to.
    :param cookies: Dictionary or CookieJar object to be added.
    :rtype: CookieJar
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('requests.cookies.merge_cookies', 'merge_cookies(cookiejar, cookies)', {'cookielib': cookielib, 'cookiejar_from_dict': cookiejar_from_dict, 'cookiejar': cookiejar, 'cookies': cookies}, 1)

