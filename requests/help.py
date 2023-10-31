"""Module containing bug report helper(s)."""

import json
import platform
import ssl
import sys
import idna
import urllib3
from . import __version__ as requests_version
try:
    import charset_normalizer
except ImportError:
    charset_normalizer = None
try:
    import chardet
except ImportError:
    chardet = None
try:
    from urllib3.contrib import pyopenssl
except ImportError:
    pyopenssl = None
    OpenSSL = None
    cryptography = None
else:
    import cryptography
    import OpenSSL

def _implementation():
    """Return a dict with the Python implementation and version.

    Provide both the name and the version of the Python implementation
    currently running. For example, on CPython 3.10.3 it will return
    {'name': 'CPython', 'version': '3.10.3'}.

    This function works best on CPython and PyPy: in particular, it probably
    doesn't work for Jython or IronPython. Future investigation should be done
    to work out the correct shape of the code for those platforms.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('requests.help._implementation', '_implementation()', {'platform': platform, 'sys': sys}, 1)

def info():
    """Generate information for a bug report."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('requests.help.info', 'info()', {'platform': platform, '_implementation': _implementation, 'urllib3': urllib3, 'charset_normalizer': charset_normalizer, 'chardet': chardet, 'OpenSSL': OpenSSL, 'cryptography': cryptography, 'idna': idna, 'ssl': ssl, 'pyopenssl': pyopenssl, 'requests_version': requests_version}, 1)

def main():
    """Pretty-print the bug information as JSON."""
    print(json.dumps(info(), sort_keys=True, indent=2))
if __name__ == '__main__':
    main()

