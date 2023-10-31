import os
import sys
import time
import errno
import signal
import warnings
import subprocess
import traceback
try:
    import psutil
except ImportError:
    psutil = None

def kill_process_tree(process, use_psutil=True):
    """Terminate process and its descendants with SIGKILL"""
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.externals.loky.backend.utils.kill_process_tree', 'kill_process_tree(process, use_psutil=True)', {'psutil': psutil, '_kill_process_tree_with_psutil': _kill_process_tree_with_psutil, '_kill_process_tree_without_psutil': _kill_process_tree_without_psutil, 'process': process, 'use_psutil': use_psutil}, 0)

def recursive_terminate(process, use_psutil=True):
    warnings.warn('recursive_terminate is deprecated in loky 3.2, use kill_process_treeinstead', DeprecationWarning)
    kill_process_tree(process, use_psutil=use_psutil)

def _kill_process_tree_with_psutil(process):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.externals.loky.backend.utils._kill_process_tree_with_psutil', '_kill_process_tree_with_psutil(process)', {'psutil': psutil, 'process': process}, 1)

def _kill_process_tree_without_psutil(process):
    """Terminate a process and its descendants."""
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.externals.loky.backend.utils._kill_process_tree_without_psutil', '_kill_process_tree_without_psutil(process)', {'sys': sys, '_windows_taskkill_process_tree': _windows_taskkill_process_tree, '_posix_recursive_kill': _posix_recursive_kill, 'traceback': traceback, 'warnings': warnings, 'process': process}, 0)

def _windows_taskkill_process_tree(pid):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.externals.loky.backend.utils._windows_taskkill_process_tree', '_windows_taskkill_process_tree(pid)', {'subprocess': subprocess, 'pid': pid}, 0)

def _kill(pid):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.externals.loky.backend.utils._kill', '_kill(pid)', {'signal': signal, 'os': os, 'errno': errno, 'pid': pid}, 0)

def _posix_recursive_kill(pid):
    """Recursively kill the descendants of a process before killing it."""
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.externals.loky.backend.utils._posix_recursive_kill', '_posix_recursive_kill(pid)', {'subprocess': subprocess, '_posix_recursive_kill': _posix_recursive_kill, '_kill': _kill, 'pid': pid}, 0)

def get_exitcodes_terminated_worker(processes):
    """Return a formatted string with the exitcodes of terminated workers.

    If necessary, wait (up to .25s) for the system to correctly set the
    exitcode of one terminated worker.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.externals.loky.backend.utils.get_exitcodes_terminated_worker', 'get_exitcodes_terminated_worker(processes)', {'time': time, '_format_exitcodes': _format_exitcodes, 'processes': processes}, 1)

def _format_exitcodes(exitcodes):
    """Format a list of exit code with names of the signals if possible"""
    str_exitcodes = [f'{_get_exitcode_name(e)}({e})' for e in exitcodes if e is not None]
    return '{' + ', '.join(str_exitcodes) + '}'

def _get_exitcode_name(exitcode):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.externals.loky.backend.utils._get_exitcode_name', '_get_exitcode_name(exitcode)', {'sys': sys, 'exitcode': exitcode}, 1)

