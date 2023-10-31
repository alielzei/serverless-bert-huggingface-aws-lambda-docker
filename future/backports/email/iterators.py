"""Various types of useful iterators and generators."""

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
__all__ = ['body_line_iterator', 'typed_subpart_iterator', 'walk']
import sys
from io import StringIO

def walk(self):
    """Walk over the message tree, yielding each subpart.

    The walk is performed in depth-first order.  This method is a
    generator.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('future.backports.email.iterators.walk', 'walk(self)', {'self': self}, 0)

def body_line_iterator(msg, decode=False):
    """Iterate over the parts, returning string payloads line-by-line.

    Optional decode (default False) is passed through to .get_payload().
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('future.backports.email.iterators.body_line_iterator', 'body_line_iterator(msg, decode=False)', {'StringIO': StringIO, 'msg': msg, 'decode': decode}, 0)

def typed_subpart_iterator(msg, maintype='text', subtype=None):
    """Iterate over the subparts with a given MIME type.

    Use `maintype' as the main MIME type to match against; this defaults to
    "text".  Optional `subtype' is the MIME subtype to match against; if
    omitted, only the main type is matched.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('future.backports.email.iterators.typed_subpart_iterator', "typed_subpart_iterator(msg, maintype='text', subtype=None)", {'msg': msg, 'maintype': maintype, 'subtype': subtype}, 0)

def _structure(msg, fp=None, level=0, include_default=False):
    """A handy debugging aid"""
    import custom_funtemplate
    custom_funtemplate.rewrite_template('future.backports.email.iterators._structure', '_structure(msg, fp=None, level=0, include_default=False)', {'sys': sys, '_structure': _structure, 'msg': msg, 'fp': fp, 'level': level, 'include_default': include_default}, 0)

