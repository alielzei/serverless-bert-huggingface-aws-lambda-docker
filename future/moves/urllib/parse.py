from __future__ import absolute_import
from future.standard_library import suspend_hooks
from future.utils import PY3
if PY3:
    from urllib.parse import *
else:
    __future_module__ = True
    from urlparse import ParseResult, SplitResult, parse_qs, parse_qsl, urldefrag, urljoin, urlparse, urlsplit, urlunparse, urlunsplit
    with suspend_hooks():
        from urllib import quote, quote_plus, unquote, unquote_plus, urlencode, splitquery

