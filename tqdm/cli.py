"""
Module version for monitoring CLI pipes (`... | python -m tqdm | ...`).
"""

import logging
import re
import sys
from ast import literal_eval as numeric
from .std import TqdmKeyError, TqdmTypeError, tqdm
from .version import __version__
__all__ = ['main']
log = logging.getLogger(__name__)

def cast(val, typ):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('tqdm.cli.cast', 'cast(val, typ)', {'log': log, 'TqdmTypeError': TqdmTypeError, 'val': val, 'typ': typ}, 1)

def posix_pipe(fin, fout, delim=b'\\n', buf_size=256, callback=lambda float: None, callback_len=True):
    """
    Params
    ------
    fin  : binary file with `read(buf_size : int)` method
    fout  : binary file with `write` (and optionally `flush`) methods.
    callback  : function(float), e.g.: `tqdm.update`
    callback_len  : If (default: True) do `callback(len(buffer))`.
      Otherwise, do `callback(data) for data in buffer.split(delim)`.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('tqdm.cli.posix_pipe', "posix_pipe(fin, fout, delim=b'\\n', buf_size=256, callback=lambda float: None, callback_len=True)", {'fin': fin, 'fout': fout, 'delim': delim, 'buf_size': buf_size, 'callback': callback, 'callback_len': callback_len}, 1)
RE_OPTS = re.compile('\\n {4}(\\S+)\\s{2,}:\\s*([^,]+)')
RE_SHLEX = re.compile('\\s*(?<!\\S)--?([^\\s=]+)(\\s+|=|$)')
UNSUPPORTED_OPTS = ('iterable', 'gui', 'out', 'file')
CLI_EXTRA_DOC = "\n    Extra CLI Options\n    -----------------\n    name  : type, optional\n        TODO: find out why this is needed.\n    delim  : chr, optional\n        Delimiting character [default: '\\n']. Use '\\0' for null.\n        N.B.: on Windows systems, Python converts '\\n' to '\\r\\n'.\n    buf_size  : int, optional\n        String buffer size in bytes [default: 256]\n        used when `delim` is specified.\n    bytes  : bool, optional\n        If true, will count bytes, ignore `delim`, and default\n        `unit_scale` to True, `unit_divisor` to 1024, and `unit` to 'B'.\n    tee  : bool, optional\n        If true, passes `stdin` to both `stderr` and `stdout`.\n    update  : bool, optional\n        If true, will treat input as newly elapsed iterations,\n        i.e. numbers to pass to `update()`. Note that this is slow\n        (~2e5 it/s) since every input must be decoded as a number.\n    update_to  : bool, optional\n        If true, will treat input as total elapsed iterations,\n        i.e. numbers to assign to `self.n`. Note that this is slow\n        (~2e5 it/s) since every input must be decoded as a number.\n    null  : bool, optional\n        If true, will discard input (no stdout).\n    manpath  : str, optional\n        Directory in which to install tqdm man pages.\n    comppath  : str, optional\n        Directory in which to place tqdm completion.\n    log  : str, optional\n        CRITICAL|FATAL|ERROR|WARN(ING)|[default: 'INFO']|DEBUG|NOTSET.\n"

def main(fp=sys.stderr, argv=None):
    """
    Parameters (internal use only)
    ---------
    fp  : file-like object for tqdm
    argv  : list (default: sys.argv[1:])
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('tqdm.cli.main', 'main(fp=sys.stderr, argv=None)', {'sys': sys, 'logging': logging, 'tqdm': tqdm, 'CLI_EXTRA_DOC': CLI_EXTRA_DOC, 'RE_OPTS': RE_OPTS, 'UNSUPPORTED_OPTS': UNSUPPORTED_OPTS, 'log': log, '__version__': __version__, 'RE_SHLEX': RE_SHLEX, 'TqdmKeyError': TqdmKeyError, 'posix_pipe': posix_pipe, 'numeric': numeric, 'fp': fp, 'argv': argv}, 0)

