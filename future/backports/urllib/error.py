"""Exception classes raised by urllib.

The base exception class is URLError, which inherits from IOError.  It
doesn't define any behavior of its own, but is the base class for all
exceptions defined in this package.

HTTPError is an exception class that is also a valid HTTP response
instance.  It behaves this way because HTTP protocol errors are valid
responses, with a status code, headers, and a body.  In some contexts,
an application may want to handle an exception like a regular
response.
"""

from __future__ import absolute_import, division, unicode_literals
from future import standard_library
from future.backports.urllib import response as urllib_response
__all__ = ['URLError', 'HTTPError', 'ContentTooShortError']


class URLError(IOError):
    
    def __init__(self, reason, filename=None):
        self.args = (reason, )
        self.reason = reason
        if filename is not None:
            self.filename = filename
    
    def __str__(self):
        return '<urlopen error %s>' % self.reason



class HTTPError(URLError, urllib_response.addinfourl):
    """Raised when HTTP error occurs, but also acts like non-error return"""
    __super_init = urllib_response.addinfourl.__init__
    
    def __init__(self, url, code, msg, hdrs, fp):
        self.code = code
        self.msg = msg
        self.hdrs = hdrs
        self.fp = fp
        self.filename = url
        if fp is not None:
            self.__super_init(fp, hdrs, url, code)
    
    def __str__(self):
        return 'HTTP Error %s: %s' % (self.code, self.msg)
    
    @property
    def reason(self):
        return self.msg
    
    def info(self):
        return self.hdrs



class ContentTooShortError(URLError):
    
    def __init__(self, message, content):
        URLError.__init__(self, message)
        self.content = content


