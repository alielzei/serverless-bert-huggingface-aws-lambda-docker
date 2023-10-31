from __future__ import absolute_import
from future.standard_library import suspend_hooks
from future.utils import PY3
if PY3:
    from urllib.error import *
else:
    __future_module__ = True
    with suspend_hooks():
        from urllib import ContentTooShortError
        from urllib2 import URLError, HTTPError

