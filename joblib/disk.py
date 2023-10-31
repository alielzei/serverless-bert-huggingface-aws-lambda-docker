"""
Disk management utilities.
"""

import os
import sys
import time
import errno
import shutil
from multiprocessing import util
try:
    WindowsError
except NameError:
    WindowsError = OSError

def disk_used(path):
    """ Return the disk usage in a directory."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.disk.disk_used', 'disk_used(path)', {'os': os, 'path': path}, 1)

def memstr_to_bytes(text):
    """ Convert a memory text to its value in bytes.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.disk.memstr_to_bytes', 'memstr_to_bytes(text)', {'text': text}, 1)

def mkdirp(d):
    """Ensure directory d exists (like mkdir -p on Unix)
    No guarantee that the directory is writable.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.disk.mkdirp', 'mkdirp(d)', {'os': os, 'errno': errno, 'd': d}, 0)
RM_SUBDIRS_RETRY_TIME = 0.1
RM_SUBDIRS_N_RETRY = 10

def rm_subdirs(path, onerror=None):
    """Remove all subdirectories in this path.

    The directory indicated by `path` is left in place, and its subdirectories
    are erased.

    If onerror is set, it is called to handle the error with arguments (func,
    path, exc_info) where func is os.listdir, os.remove, or os.rmdir;
    path is the argument to that function that caused it to fail; and
    exc_info is a tuple returned by sys.exc_info().  If onerror is None,
    an exception is raised.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.disk.rm_subdirs', 'rm_subdirs(path, onerror=None)', {'os': os, 'sys': sys, 'delete_folder': delete_folder, 'path': path, 'onerror': onerror}, 0)

def delete_folder(folder_path, onerror=None, allow_non_empty=True):
    """Utility function to cleanup a temporary folder if it still exists."""
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.disk.delete_folder', 'delete_folder(folder_path, onerror=None, allow_non_empty=True)', {'os': os, 'shutil': shutil, 'util': util, 'RM_SUBDIRS_N_RETRY': RM_SUBDIRS_N_RETRY, 'time': time, 'RM_SUBDIRS_RETRY_TIME': RM_SUBDIRS_RETRY_TIME, 'folder_path': folder_path, 'onerror': onerror, 'allow_non_empty': allow_non_empty}, 0)

