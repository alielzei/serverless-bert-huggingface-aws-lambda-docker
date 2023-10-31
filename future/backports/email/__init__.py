"""
Backport of the Python 3.3 email package for Python-Future.

A package for parsing, handling, and generating email messages.
"""

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from future.utils import surrogateescape
surrogateescape.register_surrogateescape()
__version__ = '5.1.0'
__all__ = ['base64mime', 'charset', 'encoders', 'errors', 'feedparser', 'generator', 'header', 'iterators', 'message', 'message_from_file', 'message_from_binary_file', 'message_from_string', 'message_from_bytes', 'mime', 'parser', 'quoprimime', 'utils']

def message_from_string(s, *args, **kws):
    """Parse a string into a Message object model.

    Optional _class and strict are passed to the Parser constructor.
    """
    from future.backports.email.parser import Parser
    return Parser(*args, **kws).parsestr(s)

def message_from_bytes(s, *args, **kws):
    """Parse a bytes string into a Message object model.

    Optional _class and strict are passed to the Parser constructor.
    """
    from future.backports.email.parser import BytesParser
    return BytesParser(*args, **kws).parsebytes(s)

def message_from_file(fp, *args, **kws):
    """Read a file and parse its contents into a Message object model.

    Optional _class and strict are passed to the Parser constructor.
    """
    from future.backports.email.parser import Parser
    return Parser(*args, **kws).parse(fp)

def message_from_binary_file(fp, *args, **kws):
    """Read a binary file and parse its contents into a Message object model.

    Optional _class and strict are passed to the Parser constructor.
    """
    from future.backports.email.parser import BytesParser
    return BytesParser(*args, **kws).parse(fp)

