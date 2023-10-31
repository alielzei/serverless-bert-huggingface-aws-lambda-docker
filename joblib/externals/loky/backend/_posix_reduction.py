import os
import socket
import _socket
from multiprocessing.connection import Connection
from multiprocessing.context import get_spawning_popen
from .reduction import register
HAVE_SEND_HANDLE = (hasattr(socket, 'CMSG_LEN') and hasattr(socket, 'SCM_RIGHTS') and hasattr(socket.socket, 'sendmsg'))

def _mk_inheritable(fd):
    os.set_inheritable(fd, True)
    return fd

def DupFd(fd):
    """Return a wrapper for an fd."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.externals.loky.backend._posix_reduction.DupFd', 'DupFd(fd)', {'get_spawning_popen': get_spawning_popen, 'HAVE_SEND_HANDLE': HAVE_SEND_HANDLE, 'fd': fd}, 1)

def _reduce_socket(s):
    df = DupFd(s.fileno())
    return (_rebuild_socket, (df, s.family, s.type, s.proto))

def _rebuild_socket(df, family, type, proto):
    fd = df.detach()
    return socket.fromfd(fd, family, type, proto)

def rebuild_connection(df, readable, writable):
    fd = df.detach()
    return Connection(fd, readable, writable)

def reduce_connection(conn):
    df = DupFd(conn.fileno())
    return (rebuild_connection, (df, conn.readable, conn.writable))
register(socket.socket, _reduce_socket)
register(_socket.socket, _reduce_socket)
register(Connection, reduce_connection)

