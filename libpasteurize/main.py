"""
pasteurize: automatic conversion of Python 3 code to clean 2/3 code
===================================================================

``pasteurize`` attempts to convert existing Python 3 code into source-compatible
Python 2 and 3 code.

Use it like this on Python 3 code:

  $ pasteurize --verbose mypython3script.py

This removes any Py3-only syntax (e.g. new metaclasses) and adds these
import lines:

    from __future__ import absolute_import
    from __future__ import division
    from __future__ import print_function
    from __future__ import unicode_literals
    from future import standard_library
    standard_library.install_hooks()
    from builtins import *

To write changes to the files, use the -w flag.

It also adds any other wrappers needed for Py2/3 compatibility.

Note that separate stages are not available (or needed) when converting from
Python 3 with ``pasteurize`` as they are when converting from Python 2 with
``futurize``.

The --all-imports option forces adding all ``__future__`` imports,
``builtins`` imports, and standard library aliases, even if they don't
seem necessary for the current state of each module. (This can simplify
testing, and can reduce the need to think about Py2 compatibility when editing
the code further.)

"""

from __future__ import absolute_import, print_function, unicode_literals
import sys
import logging
import optparse
from lib2to3.main import main, warn, StdoutRefactoringTool
from lib2to3 import refactor
from future import __version__
from libpasteurize.fixes import fix_names

def main(args=None):
    """Main program.

    Returns a suggested exit status (0, 1, 2).
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('libpasteurize.main.main', 'main(args=None)', {'optparse': optparse, 'fix_names': fix_names, 'warn': warn, '__version__': __version__, 'sys': sys, 'logging': logging, 'StdoutRefactoringTool': StdoutRefactoringTool, 'refactor': refactor, 'args': args}, 1)

