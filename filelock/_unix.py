from __future__ import annotations
import os
import sys
from contextlib import suppress
from errno import ENOSYS
from typing import cast
from ._api import BaseFileLock
from ._util import ensure_directory_exists
has_fcntl = False
if sys.platform == 'win32':
    
    
    class UnixFileLock(BaseFileLock):
        """Uses the :func:`fcntl.flock` to hard lock the lock file on unix systems."""
        
        def _acquire(self) -> None:
            raise NotImplementedError
        
        def _release(self) -> None:
            raise NotImplementedError
    
else:
    try:
        import fcntl
    except ImportError:
        pass
    else:
        has_fcntl = True
    
    
    class UnixFileLock(BaseFileLock):
        """Uses the :func:`fcntl.flock` to hard lock the lock file on unix systems."""
        
        def _acquire(self) -> None:
            ensure_directory_exists(self.lock_file)
            open_flags = os.O_RDWR | os.O_CREAT | os.O_TRUNC
            fd = os.open(self.lock_file, open_flags, self._context.mode)
            with suppress(PermissionError):
                os.fchmod(fd, self._context.mode)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exception:
                os.close(fd)
                if exception.errno == ENOSYS:
                    msg = 'FileSystem does not appear to support flock; user SoftFileLock instead'
                    raise NotImplementedError(msg) from exception
            else:
                self._context.lock_file_fd = fd
        
        def _release(self) -> None:
            fd = cast(int, self._context.lock_file_fd)
            self._context.lock_file_fd = None
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
    
__all__ = ['has_fcntl', 'UnixFileLock']

