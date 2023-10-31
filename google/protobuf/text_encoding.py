"""Encoding related utilities."""

import re
_cescape_chr_to_symbol_map = {}
_cescape_chr_to_symbol_map[9] = '\\t'
_cescape_chr_to_symbol_map[10] = '\\n'
_cescape_chr_to_symbol_map[13] = '\\r'
_cescape_chr_to_symbol_map[34] = '\\"'
_cescape_chr_to_symbol_map[39] = "\\'"
_cescape_chr_to_symbol_map[92] = '\\\\'
_cescape_unicode_to_str = [chr(i) for i in range(0, 256)]
for (byte, string) in _cescape_chr_to_symbol_map.items():
    _cescape_unicode_to_str[byte] = string
_cescape_byte_to_str = ['\\%03o' % i for i in range(0, 32)] + [chr(i) for i in range(32, 127)] + ['\\%03o' % i for i in range(127, 256)]
for (byte, string) in _cescape_chr_to_symbol_map.items():
    _cescape_byte_to_str[byte] = string
del byte, string

def CEscape(text, as_utf8):
    """Escape a bytes string for use in an text protocol buffer.

  Args:
    text: A byte string to be escaped.
    as_utf8: Specifies if result may contain non-ASCII characters.
        In Python 3 this allows unescaped non-ASCII Unicode characters.
        In Python 2 the return value will be valid UTF-8 rather than only ASCII.
  Returns:
    Escaped string (str).
  """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.text_encoding.CEscape', 'CEscape(text, as_utf8)', {'_cescape_chr_to_symbol_map': _cescape_chr_to_symbol_map, '_cescape_unicode_to_str': _cescape_unicode_to_str, '_cescape_byte_to_str': _cescape_byte_to_str, 'text': text, 'as_utf8': as_utf8}, 1)
_CUNESCAPE_HEX = re.compile('(\\\\+)x([0-9a-fA-F])(?![0-9a-fA-F])')

def CUnescape(text):
    """Unescape a text string with C-style escape sequences to UTF-8 bytes.

  Args:
    text: The data to parse in a str.
  Returns:
    A byte string.
  """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.text_encoding.CUnescape', 'CUnescape(text)', {'_CUNESCAPE_HEX': _CUNESCAPE_HEX, 'text': text}, 1)

