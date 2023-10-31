"""Encodings and related functions."""

from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
from future.builtins import str
__all__ = ['encode_7or8bit', 'encode_base64', 'encode_noop', 'encode_quopri']
try:
    from base64 import encodebytes as _bencode
except ImportError:
    from base64 import encodestring as _bencode
from quopri import encodestring as _encodestring

def _qencode(s):
    enc = _encodestring(s, quotetabs=True)
    return enc.replace(' ', '=20')

def encode_base64(msg):
    """Encode the message's payload in Base64.

    Also, add an appropriate Content-Transfer-Encoding header.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('future.backports.email.encoders.encode_base64', 'encode_base64(msg)', {'_bencode': _bencode, 'msg': msg}, 0)

def encode_quopri(msg):
    """Encode the message's payload in quoted-printable.

    Also, add an appropriate Content-Transfer-Encoding header.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('future.backports.email.encoders.encode_quopri', 'encode_quopri(msg)', {'_qencode': _qencode, 'msg': msg}, 0)

def encode_7or8bit(msg):
    """Set the Content-Transfer-Encoding header to 7bit or 8bit."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.backports.email.encoders.encode_7or8bit', 'encode_7or8bit(msg)', {'msg': msg}, 1)

def encode_noop(msg):
    """Do nothing."""
    import custom_funtemplate
    custom_funtemplate.rewrite_template('future.backports.email.encoders.encode_noop', 'encode_noop(msg)', {'msg': msg}, 0)

