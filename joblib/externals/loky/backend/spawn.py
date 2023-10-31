import os
import sys
import runpy
import textwrap
import types
from multiprocessing import process, util
if sys.platform != 'win32':
    WINEXE = False
    WINSERVICE = False
else:
    import msvcrt
    from multiprocessing.reduction import duplicate
    WINEXE = (sys.platform == 'win32' and getattr(sys, 'frozen', False))
    WINSERVICE = sys.executable.lower().endswith('pythonservice.exe')
if WINSERVICE:
    _python_exe = os.path.join(sys.exec_prefix, 'python.exe')
else:
    _python_exe = sys.executable

def get_executable():
    return _python_exe

def _check_not_importing_main():
    if getattr(process.current_process(), '_inheriting', False):
        raise RuntimeError(textwrap.dedent('            An attempt has been made to start a new process before the\n            current process has finished its bootstrapping phase.\n\n            This probably means that you are not using fork to start your\n            child processes and you have forgotten to use the proper idiom\n            in the main module:\n\n                if __name__ == \'__main__\':\n                    freeze_support()\n                    ...\n\n            The "freeze_support()" line can be omitted if the program\n            is not going to be frozen to produce an executable.'))

def get_preparation_data(name, init_main_module=True):
    """Return info about parent needed by child to unpickle process object."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.externals.loky.backend.spawn.get_preparation_data', 'get_preparation_data(name, init_main_module=True)', {'_check_not_importing_main': _check_not_importing_main, 'util': util, 'process': process, 'sys': sys, 'os': os, 'msvcrt': msvcrt, 'WINEXE': WINEXE, 'WINSERVICE': WINSERVICE, 'name': name, 'init_main_module': init_main_module}, 1)
old_main_modules = []

def prepare(data, parent_sentinel=None):
    """Try to get current process ready to unpickle process object."""
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.externals.loky.backend.spawn.prepare', 'prepare(data, parent_sentinel=None)', {'process': process, 'util': util, 'sys': sys, 'os': os, 'duplicate': duplicate, 'msvcrt': msvcrt, '_fixup_main_from_name': _fixup_main_from_name, '_fixup_main_from_path': _fixup_main_from_path, 'data': data, 'parent_sentinel': parent_sentinel}, 0)

def _fixup_main_from_name(mod_name):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.externals.loky.backend.spawn._fixup_main_from_name', '_fixup_main_from_name(mod_name)', {'sys': sys, 'old_main_modules': old_main_modules, 'types': types, 'runpy': runpy, 'mod_name': mod_name}, 1)

def _fixup_main_from_path(main_path):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.externals.loky.backend.spawn._fixup_main_from_path', '_fixup_main_from_path(main_path)', {'sys': sys, 'os': os, 'old_main_modules': old_main_modules, 'types': types, 'runpy': runpy, 'main_path': main_path}, 1)

