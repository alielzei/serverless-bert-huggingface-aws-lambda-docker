from __future__ import annotations
import os
import stat
import sys
from errno import EACCES, EISDIR
from pathlib import Path

def raise_on_not_writable_file(filename: str) -> None:
    """
    Raise an exception if attempting to open the file for writing would fail.
    This is done so files that will never be writable can be separated from
    files that are writable but currently locked
    :param filename: file to check
    :raises OSError: as if the file was opened for writing.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('filelock._util.raise_on_not_writable_file', 'raise_on_not_writable_file(filename)', {'os': os, 'stat': stat, 'EACCES': EACCES, 'sys': sys, 'EISDIR': EISDIR, 'filename': filename}, 1)

def ensure_directory_exists(filename: Path | str) -> None:
    """
    Ensure the directory containing the file exists (create it if necessary)
    :param filename: file.
    """
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
__all__ = ['raise_on_not_writable_file', 'ensure_directory_exists']

