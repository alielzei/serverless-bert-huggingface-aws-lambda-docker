import os
import sys
import msvcrt
import _winapi
from pickle import load
from multiprocessing import process, util
from multiprocessing.context import set_spawning_popen
from multiprocessing.popen_spawn_win32 import Popen as _Popen
from . import reduction, spawn
__all__ = ['Popen']

def _path_eq(p1, p2):
    return (p1 == p2 or os.path.normcase(p1) == os.path.normcase(p2))
WINENV = (hasattr(sys, '_base_executable') and not _path_eq(sys.executable, sys._base_executable))

def _close_handles(*handles):
    for handle in handles:
        _winapi.CloseHandle(handle)


class Popen(_Popen):
    """
    Start a subprocess to run the code of a process object.

    We differ from cpython implementation with the way we handle environment
    variables, in order to be able to modify then in the child processes before
    importing any library, in order to control the number of threads in C-level
    threadpools.

    We also use the loky preparation data, in particular to handle main_module
    inits and the loky resource tracker.
    """
    method = 'loky'
    
    def __init__(self, process_obj):
        prep_data = spawn.get_preparation_data(process_obj._name, getattr(process_obj, 'init_main_module', True))
        (rhandle, whandle) = _winapi.CreatePipe(None, 0)
        wfd = msvcrt.open_osfhandle(whandle, 0)
        cmd = get_command_line(parent_pid=os.getpid(), pipe_handle=rhandle)
        python_exe = spawn.get_executable()
        child_env = {**os.environ, **process_obj.env}
        if (WINENV and _path_eq(python_exe, sys.executable)):
            cmd[0] = python_exe = sys._base_executable
            child_env['__PYVENV_LAUNCHER__'] = sys.executable
        cmd = ' '.join((f'"{x}"' for x in cmd))
        with open(wfd, 'wb') as to_child:
            try:
                (hp, ht, pid, _) = _winapi.CreateProcess(python_exe, cmd, None, None, False, 0, child_env, None, None)
                _winapi.CloseHandle(ht)
            except BaseException:
                _winapi.CloseHandle(rhandle)
                raise
            self.pid = pid
            self.returncode = None
            self._handle = hp
            self.sentinel = int(hp)
            self.finalizer = util.Finalize(self, _close_handles, (self.sentinel, int(rhandle)))
            set_spawning_popen(self)
            try:
                reduction.dump(prep_data, to_child)
                reduction.dump(process_obj, to_child)
            finally:
                set_spawning_popen(None)


def get_command_line(pipe_handle, parent_pid, **kwds):
    """Returns prefix of command line used for spawning a child process."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.externals.loky.backend.popen_loky_win32.get_command_line', 'get_command_line(pipe_handle, parent_pid, **kwds)', {'sys': sys, 'util': util, 'spawn': spawn, 'pipe_handle': pipe_handle, 'parent_pid': parent_pid, 'kwds': kwds}, 1)

def is_forking(argv):
    """Return whether commandline indicates we are forking."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.externals.loky.backend.popen_loky_win32.is_forking', 'is_forking(argv)', {'argv': argv}, 1)

def main(pipe_handle, parent_pid=None):
    """Run code specified by data received over pipe."""
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.externals.loky.backend.popen_loky_win32.main', 'main(pipe_handle, parent_pid=None)', {'is_forking': is_forking, 'sys': sys, '_winapi': _winapi, 'reduction': reduction, 'msvcrt': msvcrt, 'os': os, 'process': process, 'load': load, 'spawn': spawn, 'pipe_handle': pipe_handle, 'parent_pid': parent_pid}, 0)

