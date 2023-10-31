import sys
from distutils.log import *
from distutils.log import Log as old_Log
from distutils.log import _global_log
from numpy.distutils.misc_util import red_text, default_text, cyan_text, green_text, is_sequence, is_string

def _fix_args(args, flag=1):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.log._fix_args', '_fix_args(args, flag=1)', {'is_string': is_string, 'is_sequence': is_sequence, '_fix_args': _fix_args, 'args': args, 'flag': flag}, 1)


class Log(old_Log):
    
    def _log(self, level, msg, args):
        if level >= self.threshold:
            if args:
                msg = msg % _fix_args(args)
            if 0:
                if (msg.startswith('copying ') and msg.find(' -> ') != -1):
                    return
                if msg.startswith('byte-compiling '):
                    return
            print(_global_color_map[level](msg))
            sys.stdout.flush()
    
    def good(self, msg, *args):
        """
        If we log WARN messages, log this message as a 'nice' anti-warn
        message.

        """
        if WARN >= self.threshold:
            if args:
                print(green_text(msg % _fix_args(args)))
            else:
                print(green_text(msg))
            sys.stdout.flush()

_global_log.__class__ = Log
good = _global_log.good

def set_threshold(level, force=False):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.log.set_threshold', 'set_threshold(level, force=False)', {'_global_log': _global_log, 'DEBUG': DEBUG, 'info': info, 'level': level, 'force': force}, 1)

def get_threshold():
    return _global_log.threshold

def set_verbosity(v, force=False):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.log.set_verbosity', 'set_verbosity(v, force=False)', {'_global_log': _global_log, 'set_threshold': set_threshold, 'ERROR': ERROR, 'WARN': WARN, 'INFO': INFO, 'DEBUG': DEBUG, 'FATAL': FATAL, 'v': v, 'force': force}, 1)
_global_color_map = {DEBUG: cyan_text, INFO: default_text, WARN: red_text, ERROR: red_text, FATAL: red_text}
set_verbosity(0, force=True)
_error = error
_warn = warn
_info = info
_debug = debug

def error(msg, *a, **kw):
    _error(f'ERROR: {msg}', *a, **kw)

def warn(msg, *a, **kw):
    _warn(f'WARN: {msg}', *a, **kw)

def info(msg, *a, **kw):
    _info(f'INFO: {msg}', *a, **kw)

def debug(msg, *a, **kw):
    _debug(f'DEBUG: {msg}', *a, **kw)

