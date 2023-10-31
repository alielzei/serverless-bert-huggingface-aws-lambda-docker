import os
import sys

def close_fds(keep_fds):
    """Close all the file descriptors except those in keep_fds."""
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.externals.loky.backend.fork_exec.close_fds', 'close_fds(keep_fds)', {'os': os, 'keep_fds': keep_fds}, 0)

def fork_exec(cmd, keep_fds, env=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.externals.loky.backend.fork_exec.fork_exec', 'fork_exec(cmd, keep_fds, env=None)', {'os': os, 'close_fds': close_fds, 'sys': sys, 'cmd': cmd, 'keep_fds': keep_fds, 'env': env}, 1)

