import io
import sys
import time
import typing as t
from ctypes import byref
from ctypes import c_char
from ctypes import c_char_p
from ctypes import c_int
from ctypes import c_ssize_t
from ctypes import c_ulong
from ctypes import c_void_p
from ctypes import POINTER
from ctypes import py_object
from ctypes import Structure
from ctypes.wintypes import DWORD
from ctypes.wintypes import HANDLE
from ctypes.wintypes import LPCWSTR
from ctypes.wintypes import LPWSTR
from ._compat import _NonClosingTextIOWrapper
assert sys.platform == 'win32'
import msvcrt
from ctypes import windll
from ctypes import WINFUNCTYPE
c_ssize_p = POINTER(c_ssize_t)
kernel32 = windll.kernel32
GetStdHandle = kernel32.GetStdHandle
ReadConsoleW = kernel32.ReadConsoleW
WriteConsoleW = kernel32.WriteConsoleW
GetConsoleMode = kernel32.GetConsoleMode
GetLastError = kernel32.GetLastError
GetCommandLineW = WINFUNCTYPE(LPWSTR)(('GetCommandLineW', windll.kernel32))
CommandLineToArgvW = WINFUNCTYPE(POINTER(LPWSTR), LPCWSTR, POINTER(c_int))(('CommandLineToArgvW', windll.shell32))
LocalFree = WINFUNCTYPE(c_void_p, c_void_p)(('LocalFree', windll.kernel32))
STDIN_HANDLE = GetStdHandle(-10)
STDOUT_HANDLE = GetStdHandle(-11)
STDERR_HANDLE = GetStdHandle(-12)
PyBUF_SIMPLE = 0
PyBUF_WRITABLE = 1
ERROR_SUCCESS = 0
ERROR_NOT_ENOUGH_MEMORY = 8
ERROR_OPERATION_ABORTED = 995
STDIN_FILENO = 0
STDOUT_FILENO = 1
STDERR_FILENO = 2
EOF = b'\x1a'
MAX_BYTES_WRITTEN = 32767
try:
    from ctypes import pythonapi
except ImportError:
    get_buffer = None
else:
    
    
    class Py_buffer(Structure):
        _fields_ = [('buf', c_void_p), ('obj', py_object), ('len', c_ssize_t), ('itemsize', c_ssize_t), ('readonly', c_int), ('ndim', c_int), ('format', c_char_p), ('shape', c_ssize_p), ('strides', c_ssize_p), ('suboffsets', c_ssize_p), ('internal', c_void_p)]
    
    PyObject_GetBuffer = pythonapi.PyObject_GetBuffer
    PyBuffer_Release = pythonapi.PyBuffer_Release
    
    def get_buffer(obj, writable=False):
        import custom_funtemplate
        return custom_funtemplate.rewrite_template('click._winconsole.get_buffer', 'get_buffer(obj, writable=False)', {'Py_buffer': Py_buffer, 'PyBUF_WRITABLE': PyBUF_WRITABLE, 'PyBUF_SIMPLE': PyBUF_SIMPLE, 'PyObject_GetBuffer': PyObject_GetBuffer, 'py_object': py_object, 'byref': byref, 'c_char': c_char, 'PyBuffer_Release': PyBuffer_Release, 'obj': obj, 'writable': writable}, 1)


class _WindowsConsoleRawIOBase(io.RawIOBase):
    
    def __init__(self, handle):
        self.handle = handle
    
    def isatty(self):
        super().isatty()
        return True



class _WindowsConsoleReader(_WindowsConsoleRawIOBase):
    
    def readable(self):
        return True
    
    def readinto(self, b):
        bytes_to_be_read = len(b)
        if not bytes_to_be_read:
            return 0
        elif bytes_to_be_read % 2:
            raise ValueError('cannot read odd number of bytes from UTF-16-LE encoded console')
        buffer = get_buffer(b, writable=True)
        code_units_to_be_read = bytes_to_be_read // 2
        code_units_read = c_ulong()
        rv = ReadConsoleW(HANDLE(self.handle), buffer, code_units_to_be_read, byref(code_units_read), None)
        if GetLastError() == ERROR_OPERATION_ABORTED:
            time.sleep(0.1)
        if not rv:
            raise OSError(f'Windows error: {GetLastError()}')
        if buffer[0] == EOF:
            return 0
        return 2 * code_units_read.value



class _WindowsConsoleWriter(_WindowsConsoleRawIOBase):
    
    def writable(self):
        return True
    
    @staticmethod
    def _get_error_message(errno):
        if errno == ERROR_SUCCESS:
            return 'ERROR_SUCCESS'
        elif errno == ERROR_NOT_ENOUGH_MEMORY:
            return 'ERROR_NOT_ENOUGH_MEMORY'
        return f'Windows error {errno}'
    
    def write(self, b):
        bytes_to_be_written = len(b)
        buf = get_buffer(b)
        code_units_to_be_written = min(bytes_to_be_written, MAX_BYTES_WRITTEN) // 2
        code_units_written = c_ulong()
        WriteConsoleW(HANDLE(self.handle), buf, code_units_to_be_written, byref(code_units_written), None)
        bytes_written = 2 * code_units_written.value
        if (bytes_written == 0 and bytes_to_be_written > 0):
            raise OSError(self._get_error_message(GetLastError()))
        return bytes_written



class ConsoleStream:
    
    def __init__(self, text_stream: t.TextIO, byte_stream: t.BinaryIO) -> None:
        self._text_stream = text_stream
        self.buffer = byte_stream
    
    @property
    def name(self) -> str:
        return self.buffer.name
    
    def write(self, x: t.AnyStr) -> int:
        if isinstance(x, str):
            return self._text_stream.write(x)
        try:
            self.flush()
        except Exception:
            pass
        return self.buffer.write(x)
    
    def writelines(self, lines: t.Iterable[t.AnyStr]) -> None:
        for line in lines:
            self.write(line)
    
    def __getattr__(self, name: str) -> t.Any:
        return getattr(self._text_stream, name)
    
    def isatty(self) -> bool:
        return self.buffer.isatty()
    
    def __repr__(self):
        return f'<ConsoleStream name={self.name!r} encoding={self.encoding!r}>'


def _get_text_stdin(buffer_stream: t.BinaryIO) -> t.TextIO:
    text_stream = _NonClosingTextIOWrapper(io.BufferedReader(_WindowsConsoleReader(STDIN_HANDLE)), 'utf-16-le', 'strict', line_buffering=True)
    return t.cast(t.TextIO, ConsoleStream(text_stream, buffer_stream))

def _get_text_stdout(buffer_stream: t.BinaryIO) -> t.TextIO:
    text_stream = _NonClosingTextIOWrapper(io.BufferedWriter(_WindowsConsoleWriter(STDOUT_HANDLE)), 'utf-16-le', 'strict', line_buffering=True)
    return t.cast(t.TextIO, ConsoleStream(text_stream, buffer_stream))

def _get_text_stderr(buffer_stream: t.BinaryIO) -> t.TextIO:
    text_stream = _NonClosingTextIOWrapper(io.BufferedWriter(_WindowsConsoleWriter(STDERR_HANDLE)), 'utf-16-le', 'strict', line_buffering=True)
    return t.cast(t.TextIO, ConsoleStream(text_stream, buffer_stream))
_stream_factories: t.Mapping[(int, t.Callable[([t.BinaryIO], t.TextIO)])] = {0: _get_text_stdin, 1: _get_text_stdout, 2: _get_text_stderr}

def _is_console(f: t.TextIO) -> bool:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('click._winconsole._is_console', '_is_console(f)', {'io': io, 'msvcrt': msvcrt, 'GetConsoleMode': GetConsoleMode, 'byref': byref, 'DWORD': DWORD, 'f': f, 't': t}, 1)

def _get_windows_console_stream(f: t.TextIO, encoding: t.Optional[str], errors: t.Optional[str]) -> t.Optional[t.TextIO]:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('click._winconsole._get_windows_console_stream', '_get_windows_console_stream(f, encoding, errors)', {'get_buffer': get_buffer, '_is_console': _is_console, '_stream_factories': _stream_factories, 'f': f, 'encoding': encoding, 'errors': errors, 't': t, 't': t, 'str': str, 't': t, 'str': str, 't': t}, 1)

