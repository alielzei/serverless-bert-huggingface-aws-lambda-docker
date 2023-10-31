"""
exec_command

Implements exec_command function that is (almost) equivalent to
commands.getstatusoutput function but on NT, DOS systems the
returned status is actually correct (though, the returned status
values may be different by a factor). In addition, exec_command
takes keyword arguments for (re-)defining environment variables.

Provides functions:

  exec_command  --- execute command in a specified directory and
                    in the modified environment.
  find_executable --- locate a command using info from environment
                    variable PATH. Equivalent to posix `which`
                    command.

Author: Pearu Peterson <pearu@cens.ioc.ee>
Created: 11 January 2003

Requires: Python 2.x

Successfully tested on:

========  ============  =================================================
os.name   sys.platform  comments
========  ============  =================================================
posix     linux2        Debian (sid) Linux, Python 2.1.3+, 2.2.3+, 2.3.3
                        PyCrust 0.9.3, Idle 1.0.2
posix     linux2        Red Hat 9 Linux, Python 2.1.3, 2.2.2, 2.3.2
posix     sunos5        SunOS 5.9, Python 2.2, 2.3.2
posix     darwin        Darwin 7.2.0, Python 2.3
nt        win32         Windows Me
                        Python 2.3(EE), Idle 1.0, PyCrust 0.7.2
                        Python 2.1.1 Idle 0.8
nt        win32         Windows 98, Python 2.1.1. Idle 0.8
nt        win32         Cygwin 98-4.10, Python 2.1.1(MSC) - echo tests
                        fail i.e. redefining environment variables may
                        not work. FIXED: don't use cygwin echo!
                        Comment: also `cmd /c echo` will not work
                        but redefining environment variables do work.
posix     cygwin        Cygwin 98-4.10, Python 2.3.3(cygming special)
nt        win32         Windows XP, Python 2.3.3
========  ============  =================================================

Known bugs:

* Tests, that send messages to stderr, fail when executed from MSYS prompt
  because the messages are lost at some point.

"""

__all__ = ['exec_command', 'find_executable']
import os
import sys
import subprocess
import locale
import warnings
from numpy.distutils.misc_util import is_sequence, make_temp_file
from numpy.distutils import log

def filepath_from_subprocess_output(output):
    """
    Convert `bytes` in the encoding used by a subprocess into a filesystem-appropriate `str`.

    Inherited from `exec_command`, and possibly incorrect.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.exec_command.filepath_from_subprocess_output', 'filepath_from_subprocess_output(output)', {'locale': locale, 'output': output}, 1)

def forward_bytes_to_stdout(val):
    """
    Forward bytes from a subprocess call to the console, without attempting to
    decode them.

    The assumption is that the subprocess call already returned bytes in
    a suitable encoding.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('numpy.distutils.exec_command.forward_bytes_to_stdout', 'forward_bytes_to_stdout(val)', {'sys': sys, 'val': val}, 0)

def temp_file_name():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.exec_command.temp_file_name', 'temp_file_name()', {'warnings': warnings, 'make_temp_file': make_temp_file}, 1)

def get_pythonexe():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.exec_command.get_pythonexe', 'get_pythonexe()', {'sys': sys, 'os': os}, 1)

def find_executable(exe, path=None, _cache={}):
    """Return full path of a executable or None.

    Symbolic links are not followed.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.exec_command.find_executable', 'find_executable(exe, path=None, _cache={})', {'log': log, 'os': os, 'exe': exe, 'path': path, '_cache': _cache}, 1)

def _preserve_environment(names):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.exec_command._preserve_environment', '_preserve_environment(names)', {'log': log, 'os': os, 'names': names}, 1)

def _update_environment(**env):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('numpy.distutils.exec_command._update_environment', '_update_environment(**env)', {'log': log, 'os': os, 'env': env}, 0)

def exec_command(command, execute_in='', use_shell=None, use_tee=None, _with_python=1, **env):
    """
    Return (status,output) of executed command.

    .. deprecated:: 1.17
        Use subprocess.Popen instead

    Parameters
    ----------
    command : str
        A concatenated string of executable and arguments.
    execute_in : str
        Before running command ``cd execute_in`` and after ``cd -``.
    use_shell : {bool, None}, optional
        If True, execute ``sh -c command``. Default None (True)
    use_tee : {bool, None}, optional
        If True use tee. Default None (True)


    Returns
    -------
    res : str
        Both stdout and stderr messages.

    Notes
    -----
    On NT, DOS systems the returned status is correct for external commands.
    Wild cards will not work for non-posix systems or when use_shell=0.

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.exec_command.exec_command', "exec_command(command, execute_in='', use_shell=None, use_tee=None, _with_python=1, **env)", {'warnings': warnings, 'log': log, 'os': os, '__name__': __name__, '__file__': __file__, 'sys': sys, '_preserve_environment': _preserve_environment, '_update_environment': _update_environment, '_exec_command': _exec_command, 'command': command, 'execute_in': execute_in, 'use_shell': use_shell, 'use_tee': use_tee, '_with_python': _with_python, 'env': env}, 1)

def _exec_command(command, use_shell=None, use_tee=None, **env):
    """
    Internal workhorse for exec_command().
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.exec_command._exec_command', '_exec_command(command, use_shell=None, use_tee=None, **env)', {'os': os, 'is_sequence': is_sequence, '_quote_arg': _quote_arg, 'subprocess': subprocess, 'locale': locale, 'command': command, 'use_shell': use_shell, 'use_tee': use_tee, 'env': env}, 2)

def _quote_arg(arg):
    """
    Quote the argument for safe use in a shell command line.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('numpy.distutils.exec_command._quote_arg', '_quote_arg(arg)', {'arg': arg}, 1)

