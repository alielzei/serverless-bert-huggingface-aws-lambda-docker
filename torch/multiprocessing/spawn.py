from __future__ import absolute_import, division, print_function, unicode_literals
import multiprocessing
import multiprocessing.connection
import signal
import sys
import warnings
from . import _prctl_pr_set_pdeathsig

def _wrap(fn, i, args, error_queue):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.multiprocessing.spawn._wrap', '_wrap(fn, i, args, error_queue)', {'_prctl_pr_set_pdeathsig': _prctl_pr_set_pdeathsig, 'signal': signal, 'sys': sys, 'fn': fn, 'i': i, 'args': args, 'error_queue': error_queue}, 0)
_supports_context = sys.version_info >= (3, 4)

def _python_version_check():
    if not _supports_context:
        raise RuntimeError('Requires python 3.4 or higher to use torch.multiprocessing.spawn and torch.multiprocessing.ProcessContext helper to launch multiple processes. If you are using this for distributed training and have a lower version of python, please use torch.distributed.launch instead.')


class ProcessContext:
    
    def __init__(self, processes, error_queues):
        _python_version_check()
        self.error_queues = error_queues
        self.processes = processes
        self.sentinels = {process.sentinel: index for (index, process) in enumerate(processes)}
    
    def pids(self):
        return [int(process.pid) for process in self.processes]
    
    def join(self, timeout=None):
        """
        Tries to join one or more processes in this spawn context.
        If one of them exited with a non-zero exit status, this function
        kills the remaining processes and raises an exception with the cause
        of the first process exiting.

        Returns ``True`` if all processes have been joined successfully,
        ``False`` if there are more processes that need to be joined.

        Arguments:
            timeout (float): Wait this long before giving up on waiting.
        """
        if len(self.sentinels) == 0:
            return True
        ready = multiprocessing.connection.wait(self.sentinels.keys(), timeout=timeout)
        error_index = None
        for sentinel in ready:
            index = self.sentinels.pop(sentinel)
            process = self.processes[index]
            process.join()
            if process.exitcode != 0:
                error_index = index
                break
        if error_index is None:
            return len(self.sentinels) == 0
        for process in self.processes:
            if process.is_alive():
                process.terminate()
            process.join()
        if self.error_queues[error_index].empty():
            exitcode = self.processes[error_index].exitcode
            if exitcode < 0:
                name = signal.Signals(-exitcode).name
                raise Exception('process %d terminated with signal %s' % (error_index, name))
            else:
                raise Exception('process %d terminated with exit code %d' % (error_index, exitcode))
        original_trace = self.error_queues[error_index].get()
        msg = '\n\n-- Process %d terminated with the following error:\n' % error_index
        msg += original_trace
        raise Exception(msg)



class SpawnContext(ProcessContext):
    
    def __init__(self, processes, error_queues):
        warnings.warn('SpawnContext is renamed to ProcessContext since 1.4 release.')
        super(SpawnContext, self).__init__(self, processes, error_queues)
    pass


def start_processes(fn, args=(), nprocs=1, join=True, daemon=False, start_method='spawn'):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.multiprocessing.spawn.start_processes', "start_processes(fn, args=(), nprocs=1, join=True, daemon=False, start_method='spawn')", {'_python_version_check': _python_version_check, 'multiprocessing': multiprocessing, '_wrap': _wrap, 'ProcessContext': ProcessContext, 'fn': fn, 'args': args, 'nprocs': nprocs, 'join': join, 'daemon': daemon, 'start_method': start_method}, 1)

def spawn(fn, args=(), nprocs=1, join=True, daemon=False, start_method='spawn'):
    """Spawns ``nprocs`` processes that run ``fn`` with ``args``.

    If one of the processes exits with a non-zero exit status, the
    remaining processes are killed and an exception is raised with the
    cause of termination. In the case an exception was caught in the
    child process, it is forwarded and its traceback is included in
    the exception raised in the parent process.

    Arguments:
        fn (function): Function is called as the entrypoint of the
            spawned process. This function must be defined at the top
            level of a module so it can be pickled and spawned. This
            is a requirement imposed by multiprocessing.

            The function is called as ``fn(i, *args)``, where ``i`` is
            the process index and ``args`` is the passed through tuple
            of arguments.

        args (tuple): Arguments passed to ``fn``.
        nprocs (int): Number of processes to spawn.
        join (bool): Perform a blocking join on all processes.
        daemon (bool): The spawned processes' daemon flag. If set to True,
                       daemonic processes will be created.
        start_method (string): (deprecated) this method will always use ``spawn``
                               as the start method. To use a different start method
                               use ``start_processes()``.

    Returns:
        None if ``join`` is ``True``,
        :class:`~ProcessContext` if ``join`` is ``False``

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.multiprocessing.spawn.spawn', "spawn(fn, args=(), nprocs=1, join=True, daemon=False, start_method='spawn')", {'warnings': warnings, 'start_processes': start_processes, 'fn': fn, 'args': args, 'nprocs': nprocs, 'join': join, 'daemon': daemon, 'start_method': start_method}, 1)

