"""
This is Victor Stinner's pure-Python implementation of PEP 383: the "surrogateescape" error
handler of Python 3.

Source: misc/python/surrogateescape.py in https://bitbucket.org/haypo/misc
"""

import codecs
import sys
from future import utils
FS_ERRORS = 'surrogateescape'

def u(text):
    if utils.PY3:
        return text
    else:
        return text.decode('unicode_escape')

def b(data):
    if utils.PY3:
        return data.encode('latin1')
    else:
        return data
if utils.PY3:
    _unichr = chr
    bytes_chr = lambda code: bytes((code, ))
else:
    _unichr = unichr
    bytes_chr = chr

def surrogateescape_handler(exc):
    """
    Pure Python implementation of the PEP 383: the "surrogateescape" error
    handler of Python 3. Undecodable bytes will be replaced by a Unicode
    character U+DCxx on decoding, and these are translated into the
    original bytes on encoding.
    """
    mystring = exc.object[exc.start:exc.end]
    try:
        if isinstance(exc, UnicodeDecodeError):
            decoded = replace_surrogate_decode(mystring)
        elif isinstance(exc, UnicodeEncodeError):
            decoded = replace_surrogate_encode(mystring)
        else:
            raise exc
    except NotASurrogateError:
        raise exc
    return (decoded, exc.end)


class NotASurrogateError(Exception):
    pass


def replace_surrogate_encode(mystring):
    """
    Returns a (unicode) string, not the more logical bytes, because the codecs
    register_error functionality expects this.
    """
    decoded = []
    for ch in mystring:
        code = ord(ch)
        if not 55296 <= code <= 56575:
            raise NotASurrogateError
        if 56320 <= code <= 56447:
            decoded.append(_unichr(code - 56320))
        elif code <= 56575:
            decoded.append(_unichr(code - 56320))
        else:
            raise NotASurrogateError
    return str().join(decoded)

def replace_surrogate_decode(mybytes):
    """
    Returns a (unicode) string
    """
    decoded = []
    for ch in mybytes:
        if isinstance(ch, int):
            code = ch
        else:
            code = ord(ch)
        if 128 <= code <= 255:
            decoded.append(_unichr(56320 + code))
        elif code <= 127:
            decoded.append(_unichr(code))
        else:
            raise NotASurrogateError
    return str().join(decoded)

def encodefilename(fn):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.utils.surrogateescape.encodefilename', 'encodefilename(fn)', {'FS_ENCODING': FS_ENCODING, 'bytes_chr': bytes_chr, 'FS_ERRORS': FS_ERRORS, 'fn': fn}, 1)

def decodefilename(fn):
    return fn.decode(FS_ENCODING, FS_ERRORS)
FS_ENCODING = 'ascii'
fn = b('[abcÿ]')
encoded = u('[abc\udcff]')
FS_ENCODING = codecs.lookup(FS_ENCODING).name

def register_surrogateescape():
    """
    Registers the surrogateescape error handler on Python 2 (only)
    """
    if utils.PY3:
        return
    try:
        codecs.lookup_error(FS_ERRORS)
    except LookupError:
        codecs.register_error(FS_ERRORS, surrogateescape_handler)
if __name__ == '__main__':
    pass

