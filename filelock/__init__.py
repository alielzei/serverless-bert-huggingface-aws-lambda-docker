"""
A platform independent file lock that supports the with-statement.

.. autodata:: filelock.__version__
   :no-value:

"""

from __future__ import annotations
import sys
import warnings
from typing import TYPE_CHECKING
from ._api import AcquireReturnProxy, BaseFileLock
from ._error import Timeout
from ._soft import SoftFileLock
from ._unix import UnixFileLock, has_fcntl
from ._windows import WindowsFileLock
from .version import version
__version__: str = version
if sys.platform == 'win32':
    _FileLock: type[BaseFileLock] = WindowsFileLock
elif has_fcntl:
    _FileLock: type[BaseFileLock] = UnixFileLock
else:
    _FileLock = SoftFileLock
    if warnings is not None:
        warnings.warn('only soft file lock is available', stacklevel=2)
if TYPE_CHECKING:
    FileLock = SoftFileLock
else:
    FileLock = _FileLock
__all__ = ['__version__', 'FileLock', 'SoftFileLock', 'Timeout', 'UnixFileLock', 'WindowsFileLock', 'BaseFileLock', 'AcquireReturnProxy']

