from __future__ import absolute_import
from future.utils import PY3
__future_module__ = True
if PY3:
    from html import *
else:
    '\n    General functions for HTML manipulation.\n    '
    
    def escape(s, quote=True):
        """
        Replace special characters "&", "<" and ">" to HTML-safe sequences.
        If the optional flag quote is true (the default), the quotation mark
        characters, both double quote (") and single quote (') characters are also
        translated.
        """
        import custom_funtemplate
        return custom_funtemplate.rewrite_template('future.moves.html.__init__.escape', 'escape(s, quote=True)', {'s': s, 'quote': quote}, 1)
    __all__ = ['escape']

