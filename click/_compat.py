import codecs
import io
import os
import re
import sys
import typing as t
from weakref import WeakKeyDictionary
CYGWIN = sys.platform.startswith('cygwin')
WIN = sys.platform.startswith('win')
auto_wrap_for_ansi: t.Optional[t.Callable[([t.TextIO], t.TextIO)]] = None
_ansi_re = re.compile('\\033\\[[;?0-9]*[a-zA-Z]')

def _make_text_stream(stream: t.BinaryIO, encoding: t.Optional[str], errors: t.Optional[str], force_readable: bool = False, force_writable: bool = False) -> t.TextIO:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('click._compat._make_text_stream', '_make_text_stream(stream, encoding, errors, force_readable=False, force_writable=False)', {'get_best_encoding': get_best_encoding, '_NonClosingTextIOWrapper': _NonClosingTextIOWrapper, 'stream': stream, 'encoding': encoding, 'errors': errors, 'force_readable': force_readable, 'force_writable': force_writable, 't': t, 't': t, 'str': str, 't': t, 'str': str, 't': t}, 1)

def is_ascii_encoding(encoding: str) -> bool:
    """Checks if a given encoding is ascii."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('click._compat.is_ascii_encoding', 'is_ascii_encoding(encoding)', {'codecs': codecs, 'encoding': encoding}, 1)

def get_best_encoding(stream: t.IO[t.Any]) -> str:
    """Returns the default stream encoding if not found."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('click._compat.get_best_encoding', 'get_best_encoding(stream)', {'sys': sys, 'is_ascii_encoding': is_ascii_encoding, 'stream': stream, 't': t}, 1)


class _NonClosingTextIOWrapper(io.TextIOWrapper):
    
    def __init__(self, stream: t.BinaryIO, encoding: t.Optional[str], errors: t.Optional[str], force_readable: bool = False, force_writable: bool = False, **extra) -> None:
        self._stream = stream = t.cast(t.BinaryIO, _FixupStream(stream, force_readable, force_writable))
        super().__init__(stream, encoding, errors, **extra)
    
    def __del__(self) -> None:
        try:
            self.detach()
        except Exception:
            pass
    
    def isatty(self) -> bool:
        return self._stream.isatty()



class _FixupStream:
    """The new io interface needs more from streams than streams
    traditionally implement.  As such, this fix-up code is necessary in
    some circumstances.

    The forcing of readable and writable flags are there because some tools
    put badly patched objects on sys (one such offender are certain version
    of jupyter notebook).
    """
    
    def __init__(self, stream: t.BinaryIO, force_readable: bool = False, force_writable: bool = False):
        self._stream = stream
        self._force_readable = force_readable
        self._force_writable = force_writable
    
    def __getattr__(self, name: str) -> t.Any:
        return getattr(self._stream, name)
    
    def read1(self, size: int) -> bytes:
        f = getattr(self._stream, 'read1', None)
        if f is not None:
            return t.cast(bytes, f(size))
        return self._stream.read(size)
    
    def readable(self) -> bool:
        if self._force_readable:
            return True
        x = getattr(self._stream, 'readable', None)
        if x is not None:
            return t.cast(bool, x())
        try:
            self._stream.read(0)
        except Exception:
            return False
        return True
    
    def writable(self) -> bool:
        if self._force_writable:
            return True
        x = getattr(self._stream, 'writable', None)
        if x is not None:
            return t.cast(bool, x())
        try:
            self._stream.write('')
        except Exception:
            try:
                self._stream.write(b'')
            except Exception:
                return False
        return True
    
    def seekable(self) -> bool:
        x = getattr(self._stream, 'seekable', None)
        if x is not None:
            return t.cast(bool, x())
        try:
            self._stream.seek(self._stream.tell())
        except Exception:
            return False
        return True


def _is_binary_reader(stream: t.IO[t.Any], default: bool = False) -> bool:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('click._compat._is_binary_reader', '_is_binary_reader(stream, default=False)', {'stream': stream, 'default': default, 't': t}, 1)

def _is_binary_writer(stream: t.IO[t.Any], default: bool = False) -> bool:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('click._compat._is_binary_writer', '_is_binary_writer(stream, default=False)', {'stream': stream, 'default': default, 't': t}, 1)

def _find_binary_reader(stream: t.IO[t.Any]) -> t.Optional[t.BinaryIO]:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('click._compat._find_binary_reader', '_find_binary_reader(stream)', {'_is_binary_reader': _is_binary_reader, 't': t, 'stream': stream, 't': t, 't': t}, 1)

def _find_binary_writer(stream: t.IO[t.Any]) -> t.Optional[t.BinaryIO]:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('click._compat._find_binary_writer', '_find_binary_writer(stream)', {'_is_binary_writer': _is_binary_writer, 't': t, 'stream': stream, 't': t, 't': t}, 1)

def _stream_is_misconfigured(stream: t.TextIO) -> bool:
    """A stream is misconfigured if its encoding is ASCII."""
    return is_ascii_encoding((getattr(stream, 'encoding', None) or 'ascii'))

def _is_compat_stream_attr(stream: t.TextIO, attr: str, value: t.Optional[str]) -> bool:
    """A stream attribute is compatible if it is equal to the
    desired value or the desired value is unset and the attribute
    has a value.
    """
    stream_value = getattr(stream, attr, None)
    return (stream_value == value or (value is None and stream_value is not None))

def _is_compatible_text_stream(stream: t.TextIO, encoding: t.Optional[str], errors: t.Optional[str]) -> bool:
    """Check if a stream's encoding and errors attributes are
    compatible with the desired values.
    """
    return (_is_compat_stream_attr(stream, 'encoding', encoding) and _is_compat_stream_attr(stream, 'errors', errors))

def _force_correct_text_stream(text_stream: t.IO[t.Any], encoding: t.Optional[str], errors: t.Optional[str], is_binary: t.Callable[([t.IO[t.Any], bool], bool)], find_binary: t.Callable[([t.IO[t.Any]], t.Optional[t.BinaryIO])], force_readable: bool = False, force_writable: bool = False) -> t.TextIO:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('click._compat._force_correct_text_stream', '_force_correct_text_stream(text_stream, encoding, errors, is_binary, find_binary, force_readable=False, force_writable=False)', {'t': t, '_is_compatible_text_stream': _is_compatible_text_stream, '_stream_is_misconfigured': _stream_is_misconfigured, '_make_text_stream': _make_text_stream, 'text_stream': text_stream, 'encoding': encoding, 'errors': errors, 'is_binary': is_binary, 'find_binary': find_binary, 'force_readable': force_readable, 'force_writable': force_writable, 't': t, 't': t, 'str': str, 't': t, 'str': str, 't': t, 't': t, 't': t}, 1)

def _force_correct_text_reader(text_reader: t.IO[t.Any], encoding: t.Optional[str], errors: t.Optional[str], force_readable: bool = False) -> t.TextIO:
    return _force_correct_text_stream(text_reader, encoding, errors, _is_binary_reader, _find_binary_reader, force_readable=force_readable)

def _force_correct_text_writer(text_writer: t.IO[t.Any], encoding: t.Optional[str], errors: t.Optional[str], force_writable: bool = False) -> t.TextIO:
    return _force_correct_text_stream(text_writer, encoding, errors, _is_binary_writer, _find_binary_writer, force_writable=force_writable)

def get_binary_stdin() -> t.BinaryIO:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('click._compat.get_binary_stdin', 'get_binary_stdin()', {'_find_binary_reader': _find_binary_reader, 'sys': sys, 't': t}, 1)

def get_binary_stdout() -> t.BinaryIO:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('click._compat.get_binary_stdout', 'get_binary_stdout()', {'_find_binary_writer': _find_binary_writer, 'sys': sys, 't': t}, 1)

def get_binary_stderr() -> t.BinaryIO:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('click._compat.get_binary_stderr', 'get_binary_stderr()', {'_find_binary_writer': _find_binary_writer, 'sys': sys, 't': t}, 1)

def get_text_stdin(encoding: t.Optional[str] = None, errors: t.Optional[str] = None) -> t.TextIO:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('click._compat.get_text_stdin', 'get_text_stdin(encoding=None, errors=None)', {'_get_windows_console_stream': _get_windows_console_stream, 'sys': sys, '_force_correct_text_reader': _force_correct_text_reader, 'encoding': encoding, 'errors': errors, 't': t, 'str': str, 't': t, 'str': str, 't': t}, 1)

def get_text_stdout(encoding: t.Optional[str] = None, errors: t.Optional[str] = None) -> t.TextIO:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('click._compat.get_text_stdout', 'get_text_stdout(encoding=None, errors=None)', {'_get_windows_console_stream': _get_windows_console_stream, 'sys': sys, '_force_correct_text_writer': _force_correct_text_writer, 'encoding': encoding, 'errors': errors, 't': t, 'str': str, 't': t, 'str': str, 't': t}, 1)

def get_text_stderr(encoding: t.Optional[str] = None, errors: t.Optional[str] = None) -> t.TextIO:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('click._compat.get_text_stderr', 'get_text_stderr(encoding=None, errors=None)', {'_get_windows_console_stream': _get_windows_console_stream, 'sys': sys, '_force_correct_text_writer': _force_correct_text_writer, 'encoding': encoding, 'errors': errors, 't': t, 'str': str, 't': t, 'str': str, 't': t}, 1)

def _wrap_io_open(file: t.Union[(str, 'os.PathLike[str]', int)], mode: str, encoding: t.Optional[str], errors: t.Optional[str]) -> t.IO[t.Any]:
    """Handles not passing ``encoding`` and ``errors`` in binary mode."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('click._compat._wrap_io_open', '_wrap_io_open(file, mode, encoding, errors)', {'file': file, 'mode': mode, 'encoding': encoding, 'errors': errors, 't': t, 't': t, 'str': str, 't': t, 'str': str, 't': t}, 1)

def open_stream(filename: 't.Union[str, os.PathLike[str]]', mode: str = 'r', encoding: t.Optional[str] = None, errors: t.Optional[str] = 'strict', atomic: bool = False) -> t.Tuple[(t.IO[t.Any], bool)]:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('click._compat.open_stream', "open_stream(filename, mode='r', encoding=None, errors='strict', atomic=False)", {'os': os, 'get_binary_stdout': get_binary_stdout, 'get_text_stdout': get_text_stdout, 'get_binary_stdin': get_binary_stdin, 'get_text_stdin': get_text_stdin, '_wrap_io_open': _wrap_io_open, 't': t, '_AtomicFile': _AtomicFile, 'filename': filename, 'mode': mode, 'encoding': encoding, 'errors': errors, 'atomic': atomic, 't': t, 'str': str, 't': t, 'str': str, 't': t}, 2)


class _AtomicFile:
    
    def __init__(self, f: t.IO[t.Any], tmp_filename: str, real_filename: str) -> None:
        self._f = f
        self._tmp_filename = tmp_filename
        self._real_filename = real_filename
        self.closed = False
    
    @property
    def name(self) -> str:
        return self._real_filename
    
    def close(self, delete: bool = False) -> None:
        if self.closed:
            return
        self._f.close()
        os.replace(self._tmp_filename, self._real_filename)
        self.closed = True
    
    def __getattr__(self, name: str) -> t.Any:
        return getattr(self._f, name)
    
    def __enter__(self) -> '_AtomicFile':
        return self
    
    def __exit__(self, exc_type: t.Optional[t.Type[BaseException]], *_) -> None:
        self.close(delete=exc_type is not None)
    
    def __repr__(self) -> str:
        return repr(self._f)


def strip_ansi(value: str) -> str:
    return _ansi_re.sub('', value)

def _is_jupyter_kernel_output(stream: t.IO[t.Any]) -> bool:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('click._compat._is_jupyter_kernel_output', '_is_jupyter_kernel_output(stream)', {'_FixupStream': _FixupStream, '_NonClosingTextIOWrapper': _NonClosingTextIOWrapper, 'stream': stream, 't': t}, 1)

def should_strip_ansi(stream: t.Optional[t.IO[t.Any]] = None, color: t.Optional[bool] = None) -> bool:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('click._compat.should_strip_ansi', 'should_strip_ansi(stream=None, color=None)', {'sys': sys, 'isatty': isatty, '_is_jupyter_kernel_output': _is_jupyter_kernel_output, 'stream': stream, 'color': color, 't': t, 't': t, 'bool': bool}, 1)
if (sys.platform.startswith('win') and WIN):
    from ._winconsole import _get_windows_console_stream
    
    def _get_argv_encoding() -> str:
        import locale
        return locale.getpreferredencoding()
    _ansi_stream_wrappers: t.MutableMapping[(t.TextIO, t.TextIO)] = WeakKeyDictionary()
    
    def auto_wrap_for_ansi(stream: t.TextIO, color: t.Optional[bool] = None) -> t.TextIO:
        """Support ANSI color and style codes on Windows by wrapping a
        stream with colorama.
        """
        import custom_funtemplate
        return custom_funtemplate.rewrite_template('click._compat.auto_wrap_for_ansi', 'auto_wrap_for_ansi(stream, color=None)', {'_ansi_stream_wrappers': _ansi_stream_wrappers, 'should_strip_ansi': should_strip_ansi, 't': t, 'stream': stream, 'color': color, 't': t, 't': t, 'bool': bool, 't': t}, 1)
else:
    
    def _get_argv_encoding() -> str:
        return (getattr(sys.stdin, 'encoding', None) or sys.getfilesystemencoding())
    
    def _get_windows_console_stream(f: t.TextIO, encoding: t.Optional[str], errors: t.Optional[str]) -> t.Optional[t.TextIO]:
        return None

def term_len(x: str) -> int:
    return len(strip_ansi(x))

def isatty(stream: t.IO[t.Any]) -> bool:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('click._compat.isatty', 'isatty(stream)', {'stream': stream, 't': t}, 1)

def _make_cached_stream_func(src_func: t.Callable[([], t.Optional[t.TextIO])], wrapper_func: t.Callable[([], t.TextIO)]) -> t.Callable[([], t.Optional[t.TextIO])]:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('click._compat._make_cached_stream_func', '_make_cached_stream_func(src_func, wrapper_func)', {'t': t, 'WeakKeyDictionary': WeakKeyDictionary, 'src_func': src_func, 'wrapper_func': wrapper_func, 't': t, 't': t, 't': t}, 1)
_default_text_stdin = _make_cached_stream_func(lambda: sys.stdin, get_text_stdin)
_default_text_stdout = _make_cached_stream_func(lambda: sys.stdout, get_text_stdout)
_default_text_stderr = _make_cached_stream_func(lambda: sys.stderr, get_text_stderr)
binary_streams: t.Mapping[(str, t.Callable[([], t.BinaryIO)])] = {'stdin': get_binary_stdin, 'stdout': get_binary_stdout, 'stderr': get_binary_stderr}
text_streams: t.Mapping[(str, t.Callable[([t.Optional[str], t.Optional[str]], t.TextIO)])] = {'stdin': get_text_stdin, 'stdout': get_text_stdout, 'stderr': get_text_stderr}

