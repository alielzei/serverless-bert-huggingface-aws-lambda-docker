"""
For Python < 2.7.2. total_ordering in versions prior to 2.7.2 is buggy.
See http://bugs.python.org/issue10042 for details. For these versions use
code borrowed from Python 2.7.3.

From django.utils.
"""

import sys
if sys.version_info >= (2, 7, 2):
    from functools import total_ordering
else:
    
    def total_ordering(cls):
        """Class decorator that fills in missing ordering methods"""
        import custom_funtemplate
        return custom_funtemplate.rewrite_template('future.backports.total_ordering.total_ordering', 'total_ordering(cls)', {'cls': cls}, 1)

