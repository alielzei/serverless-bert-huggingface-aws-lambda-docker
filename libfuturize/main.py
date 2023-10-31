"""
futurize: automatic conversion to clean 2/3 code using ``python-future``
======================================================================

Like Armin Ronacher's modernize.py, ``futurize`` attempts to produce clean
standard Python 3 code that runs on both Py2 and Py3.

One pass
--------

Use it like this on Python 2 code:

  $ futurize --verbose mypython2script.py

This will attempt to port the code to standard Py3 code that also
provides Py2 compatibility with the help of the right imports from
``future``.

To write changes to the files, use the -w flag.

Two stages
----------

The ``futurize`` script can also be called in two separate stages. First:

  $ futurize --stage1 mypython2script.py

This produces more modern Python 2 code that is not yet compatible with Python
3. The tests should still run and the diff should be uncontroversial to apply to
most Python projects that are willing to drop support for Python 2.5 and lower.

After this, the recommended approach is to explicitly mark all strings that must
be byte-strings with a b'' prefix and all text (unicode) strings with a u''
prefix, and then invoke the second stage of Python 2 to 2/3 conversion with::

  $ futurize --stage2 mypython2script.py

Stage 2 adds a dependency on ``future``. It converts most remaining Python
2-specific code to Python 3 code and adds appropriate imports from ``future``
to restore Py2 support.

The command above leaves all unadorned string literals as native strings
(byte-strings on Py2, unicode strings on Py3). If instead you would like all
unadorned string literals to be promoted to unicode, you can also pass this
flag:

  $ futurize --stage2 --unicode-literals mypython2script.py

This adds the declaration ``from __future__ import unicode_literals`` to the
top of each file, which implicitly declares all unadorned string literals to be
unicode strings (``unicode`` on Py2).

All imports
-----------

The --all-imports option forces adding all ``__future__`` imports,
``builtins`` imports, and standard library aliases, even if they don't
seem necessary for the current state of each module. (This can simplify
testing, and can reduce the need to think about Py2 compatibility when editing
the code further.)

"""

from __future__ import absolute_import, print_function, unicode_literals
import future.utils
from future import __version__
import sys
import logging
import optparse
import os
from lib2to3.main import warn, StdoutRefactoringTool
from lib2to3 import refactor
from libfuturize.fixes import lib2to3_fix_names_stage1, lib2to3_fix_names_stage2, libfuturize_fix_names_stage1, libfuturize_fix_names_stage2
fixer_pkg = 'libfuturize.fixes'

def main(args=None):
    """Main program.

    Args:
        fixer_pkg: the name of a package where the fixers are located.
        args: optional; a list of command line arguments. If omitted,
              sys.argv[1:] is used.

    Returns a suggested exit status (0, 1, 2).
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('libfuturize.main.main', 'main(args=None)', {'optparse': optparse, 'warn': warn, 'sys': sys, 'logging': logging, 'lib2to3_fix_names_stage1': lib2to3_fix_names_stage1, 'libfuturize_fix_names_stage1': libfuturize_fix_names_stage1, 'lib2to3_fix_names_stage2': lib2to3_fix_names_stage2, 'libfuturize_fix_names_stage2': libfuturize_fix_names_stage2, '__version__': __version__, 'os': os, 'future': future, 'StdoutRefactoringTool': StdoutRefactoringTool, 'refactor': refactor, 'args': args}, 1)

