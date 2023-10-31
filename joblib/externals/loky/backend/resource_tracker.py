import os
import shutil
import sys
import signal
import warnings
import threading
from _multiprocessing import sem_unlink
from multiprocessing import util
from . import spawn
if sys.platform == 'win32':
    import _winapi
    import msvcrt
    from multiprocessing.reduction import duplicate
__all__ = ['ensure_running', 'register', 'unregister']
_HAVE_SIGMASK = hasattr(signal, 'pthread_sigmask')
_IGNORED_SIGNALS = (signal.SIGINT, signal.SIGTERM)
_CLEANUP_FUNCS = {'folder': shutil.rmtree, 'file': os.unlink}
if os.name == 'posix':
    _CLEANUP_FUNCS['semlock'] = sem_unlink
VERBOSE = False


class ResourceTracker:
    
    def __init__(self):
        self._lock = threading.Lock()
        self._fd = None
        self._pid = None
    
    def getfd(self):
        self.ensure_running()
        return self._fd
    
    def ensure_running(self):
        """Make sure that resource tracker process is running.

        This can be run from any process.  Usually a child process will use
        the resource created by its parent."""
        with self._lock:
            if self._fd is not None:
                if self._check_alive():
                    return
                os.close(self._fd)
                if os.name == 'posix':
                    try:
                        os.waitpid(self._pid, 0)
                    except OSError:
                        pass
                self._fd = None
                self._pid = None
                warnings.warn('resource_tracker: process died unexpectedly, relaunching.  Some folders/sempahores might leak.')
            fds_to_pass = []
            try:
                fds_to_pass.append(sys.stderr.fileno())
            except Exception:
                pass
            (r, w) = os.pipe()
            if sys.platform == 'win32':
                _r = duplicate(msvcrt.get_osfhandle(r), inheritable=True)
                os.close(r)
                r = _r
            cmd = f'from {main.__module__} import main; main({r}, {VERBOSE})'
            try:
                try:
                    fds_to_pass.append(r)
                    exe = spawn.get_executable()
                    args = [exe, *util._args_from_interpreter_flags(), '-c', cmd]
                    util.debug(f'launching resource tracker: {args}')
                    try:
                        if _HAVE_SIGMASK:
                            signal.pthread_sigmask(signal.SIG_BLOCK, _IGNORED_SIGNALS)
                        pid = spawnv_passfds(exe, args, fds_to_pass)
                    finally:
                        if _HAVE_SIGMASK:
                            signal.pthread_sigmask(signal.SIG_UNBLOCK, _IGNORED_SIGNALS)
                except BaseException:
                    os.close(w)
                    raise
                else:
                    self._fd = w
                    self._pid = pid
            finally:
                if sys.platform == 'win32':
                    _winapi.CloseHandle(r)
                else:
                    os.close(r)
    
    def _check_alive(self):
        """Check for the existence of the resource tracker process."""
        try:
            self._send('PROBE', '', '')
        except BrokenPipeError:
            return False
        else:
            return True
    
    def register(self, name, rtype):
        """Register a named resource, and increment its refcount."""
        self.ensure_running()
        self._send('REGISTER', name, rtype)
    
    def unregister(self, name, rtype):
        """Unregister a named resource with resource tracker."""
        self.ensure_running()
        self._send('UNREGISTER', name, rtype)
    
    def maybe_unlink(self, name, rtype):
        """Decrement the refcount of a resource, and delete it if it hits 0"""
        self.ensure_running()
        self._send('MAYBE_UNLINK', name, rtype)
    
    def _send(self, cmd, name, rtype):
        if len(name) > 512:
            raise ValueError('name too long')
        msg = f'{cmd}:{name}:{rtype}\n'.encode('ascii')
        nbytes = os.write(self._fd, msg)
        assert nbytes == len(msg)

_resource_tracker = ResourceTracker()
ensure_running = _resource_tracker.ensure_running
register = _resource_tracker.register
maybe_unlink = _resource_tracker.maybe_unlink
unregister = _resource_tracker.unregister
getfd = _resource_tracker.getfd

def main(fd, verbose=0):
    """Run resource tracker."""
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.externals.loky.backend.resource_tracker.main', 'main(fd, verbose=0)', {'util': util, 'signal': signal, '_HAVE_SIGMASK': _HAVE_SIGMASK, '_IGNORED_SIGNALS': _IGNORED_SIGNALS, 'sys': sys, '_CLEANUP_FUNCS': _CLEANUP_FUNCS, 'msvcrt': msvcrt, 'os': os, 'warnings': warnings, 'fd': fd, 'verbose': verbose}, 0)

def spawnv_passfds(path, args, passfds):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.externals.loky.backend.resource_tracker.spawnv_passfds', 'spawnv_passfds(path, args, passfds)', {'sys': sys, 'os': os, '_winapi': _winapi, 'path': path, 'args': args, 'passfds': passfds}, 1)

