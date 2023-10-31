"""
General functions for HTML manipulation, backported from Py3.

Note that this uses Python 2.7 code with the corresponding Python 3
module names and locations.
"""

from __future__ import unicode_literals
_escape_map = {ord('&'): '&amp;', ord('<'): '&lt;', ord('>'): '&gt;'}
_escape_map_full = {ord('&'): '&amp;', ord('<'): '&lt;', ord('>'): '&gt;', ord('"'): '&quot;', ord("'"): '&#x27;'}

def escape(s, quote=True):
    """
    Replace special characters "&", "<" and ">" to HTML-safe sequences.
    If the optional flag quote is true (the default), the quotation mark
    characters, both double quote (") and single quote (') characters are also
    translated.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('future.backports.html.__init__.escape', 'escape(s, quote=True)', {'_escape_map_full': _escape_map_full, '_escape_map': _escape_map, 's': s, 'quote': quote}, 1)

