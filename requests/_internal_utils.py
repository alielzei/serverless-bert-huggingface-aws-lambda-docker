"""
requests._internal_utils
~~~~~~~~~~~~~~

Provides utility functions that are consumed internally by Requests
which depend on extremely few external helpers (such as compat)
"""

import re
from .compat import builtin_str
_VALID_HEADER_NAME_RE_BYTE = re.compile(b'^[^:\\s][^:\\r\\n]*$')
_VALID_HEADER_NAME_RE_STR = re.compile('^[^:\\s][^:\\r\\n]*$')
_VALID_HEADER_VALUE_RE_BYTE = re.compile(b'^\\S[^\\r\\n]*$|^$')
_VALID_HEADER_VALUE_RE_STR = re.compile('^\\S[^\\r\\n]*$|^$')
_HEADER_VALIDATORS_STR = (_VALID_HEADER_NAME_RE_STR, _VALID_HEADER_VALUE_RE_STR)
_HEADER_VALIDATORS_BYTE = (_VALID_HEADER_NAME_RE_BYTE, _VALID_HEADER_VALUE_RE_BYTE)
HEADER_VALIDATORS = {bytes: _HEADER_VALIDATORS_BYTE, str: _HEADER_VALIDATORS_STR}

def to_native_string(string, encoding='ascii'):
    """Given a string object, regardless of type, returns a representation of
    that string in the native string type, encoding and decoding where
    necessary. This assumes ASCII unless told otherwise.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('requests._internal_utils.to_native_string', "to_native_string(string, encoding='ascii')", {'builtin_str': builtin_str, 'string': string, 'encoding': encoding}, 1)

def unicode_is_ascii(u_string):
    """Determine if unicode string only contains ASCII characters.

    :param str u_string: unicode string to check. Must be unicode
        and not Python 2 `str`.
    :rtype: bool
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('requests._internal_utils.unicode_is_ascii', 'unicode_is_ascii(u_string)', {'u_string': u_string}, 1)

