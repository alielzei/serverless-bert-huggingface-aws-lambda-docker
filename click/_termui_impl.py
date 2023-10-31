"""
This module contains implementations for the termui module. To keep the
import time of Click down, some infrequently used functionality is
placed in this module and only imported as needed.
"""

import contextlib
import math
import os
import sys
import time
import typing as t
from gettext import gettext as _
from io import StringIO
from types import TracebackType
from ._compat import _default_text_stdout
from ._compat import CYGWIN
from ._compat import get_best_encoding
from ._compat import isatty
from ._compat import open_stream
from ._compat import strip_ansi
from ._compat import term_len
from ._compat import WIN
from .exceptions import ClickException
from .utils import echo
V = t.TypeVar('V')
if os.name == 'nt':
    BEFORE_BAR = '\r'
    AFTER_BAR = '\n'
else:
    BEFORE_BAR = '\r\x1b[?25l'
    AFTER_BAR = '\x1b[?25h\n'


class ProgressBar(t.Generic[V]):
    
    def __init__(self, iterable: t.Optional[t.Iterable[V]], length: t.Optional[int] = None, fill_char: str = '#', empty_char: str = ' ', bar_template: str = '%(bar)s', info_sep: str = '  ', show_eta: bool = True, show_percent: t.Optional[bool] = None, show_pos: bool = False, item_show_func: t.Optional[t.Callable[([t.Optional[V]], t.Optional[str])]] = None, label: t.Optional[str] = None, file: t.Optional[t.TextIO] = None, color: t.Optional[bool] = None, update_min_steps: int = 1, width: int = 30) -> None:
        self.fill_char = fill_char
        self.empty_char = empty_char
        self.bar_template = bar_template
        self.info_sep = info_sep
        self.show_eta = show_eta
        self.show_percent = show_percent
        self.show_pos = show_pos
        self.item_show_func = item_show_func
        self.label: str = (label or '')
        if file is None:
            file = _default_text_stdout()
            if file is None:
                file = StringIO()
        self.file = file
        self.color = color
        self.update_min_steps = update_min_steps
        self._completed_intervals = 0
        self.width: int = width
        self.autowidth: bool = width == 0
        if length is None:
            from operator import length_hint
            length = length_hint(iterable, -1)
            if length == -1:
                length = None
        if iterable is None:
            if length is None:
                raise TypeError('iterable or length is required')
            iterable = t.cast(t.Iterable[V], range(length))
        self.iter: t.Iterable[V] = iter(iterable)
        self.length = length
        self.pos = 0
        self.avg: t.List[float] = []
        self.last_eta: float
        self.start: float
        self.start = self.last_eta = time.time()
        self.eta_known: bool = False
        self.finished: bool = False
        self.max_width: t.Optional[int] = None
        self.entered: bool = False
        self.current_item: t.Optional[V] = None
        self.is_hidden: bool = not isatty(self.file)
        self._last_line: t.Optional[str] = None
    
    def __enter__(self) -> 'ProgressBar[V]':
        self.entered = True
        self.render_progress()
        return self
    
    def __exit__(self, exc_type: t.Optional[t.Type[BaseException]], exc_value: t.Optional[BaseException], tb: t.Optional[TracebackType]) -> None:
        self.render_finish()
    
    def __iter__(self) -> t.Iterator[V]:
        if not self.entered:
            raise RuntimeError('You need to use progress bars in a with block.')
        self.render_progress()
        return self.generator()
    
    def __next__(self) -> V:
        return next(iter(self))
    
    def render_finish(self) -> None:
        if self.is_hidden:
            return
        self.file.write(AFTER_BAR)
        self.file.flush()
    
    @property
    def pct(self) -> float:
        if self.finished:
            return 1.0
        return min(self.pos / ((float((self.length or 1)) or 1)), 1.0)
    
    @property
    def time_per_iteration(self) -> float:
        if not self.avg:
            return 0.0
        return sum(self.avg) / float(len(self.avg))
    
    @property
    def eta(self) -> float:
        if (self.length is not None and not self.finished):
            return self.time_per_iteration * (self.length - self.pos)
        return 0.0
    
    def format_eta(self) -> str:
        if self.eta_known:
            t = int(self.eta)
            seconds = t % 60
            t //= 60
            minutes = t % 60
            t //= 60
            hours = t % 24
            t //= 24
            if t > 0:
                return f'{t}d {hours:02}:{minutes:02}:{seconds:02}'
            else:
                return f'{hours:02}:{minutes:02}:{seconds:02}'
        return ''
    
    def format_pos(self) -> str:
        pos = str(self.pos)
        if self.length is not None:
            pos += f'/{self.length}'
        return pos
    
    def format_pct(self) -> str:
        return f'{int(self.pct * 100): 4}%'[1:]
    
    def format_bar(self) -> str:
        if self.length is not None:
            bar_length = int(self.pct * self.width)
            bar = self.fill_char * bar_length
            bar += self.empty_char * (self.width - bar_length)
        elif self.finished:
            bar = self.fill_char * self.width
        else:
            chars = list(self.empty_char * ((self.width or 1)))
            if self.time_per_iteration != 0:
                chars[int((math.cos(self.pos * self.time_per_iteration) / 2.0 + 0.5) * self.width)] = self.fill_char
            bar = ''.join(chars)
        return bar
    
    def format_progress_line(self) -> str:
        show_percent = self.show_percent
        info_bits = []
        if (self.length is not None and show_percent is None):
            show_percent = not self.show_pos
        if self.show_pos:
            info_bits.append(self.format_pos())
        if show_percent:
            info_bits.append(self.format_pct())
        if (self.show_eta and self.eta_known and not self.finished):
            info_bits.append(self.format_eta())
        if self.item_show_func is not None:
            item_info = self.item_show_func(self.current_item)
            if item_info is not None:
                info_bits.append(item_info)
        return (self.bar_template % {'label': self.label, 'bar': self.format_bar(), 'info': self.info_sep.join(info_bits)}).rstrip()
    
    def render_progress(self) -> None:
        import shutil
        if self.is_hidden:
            if self._last_line != self.label:
                self._last_line = self.label
                echo(self.label, file=self.file, color=self.color)
            return
        buf = []
        if self.autowidth:
            old_width = self.width
            self.width = 0
            clutter_length = term_len(self.format_progress_line())
            new_width = max(0, shutil.get_terminal_size().columns - clutter_length)
            if new_width < old_width:
                buf.append(BEFORE_BAR)
                buf.append(' ' * self.max_width)
                self.max_width = new_width
            self.width = new_width
        clear_width = self.width
        if self.max_width is not None:
            clear_width = self.max_width
        buf.append(BEFORE_BAR)
        line = self.format_progress_line()
        line_len = term_len(line)
        if (self.max_width is None or self.max_width < line_len):
            self.max_width = line_len
        buf.append(line)
        buf.append(' ' * (clear_width - line_len))
        line = ''.join(buf)
        if line != self._last_line:
            self._last_line = line
            echo(line, file=self.file, color=self.color, nl=False)
            self.file.flush()
    
    def make_step(self, n_steps: int) -> None:
        self.pos += n_steps
        if (self.length is not None and self.pos >= self.length):
            self.finished = True
        if time.time() - self.last_eta < 1.0:
            return
        self.last_eta = time.time()
        if self.pos:
            step = (time.time() - self.start) / self.pos
        else:
            step = time.time() - self.start
        self.avg = self.avg[-6:] + [step]
        self.eta_known = self.length is not None
    
    def update(self, n_steps: int, current_item: t.Optional[V] = None) -> None:
        """Update the progress bar by advancing a specified number of
        steps, and optionally set the ``current_item`` for this new
        position.

        :param n_steps: Number of steps to advance.
        :param current_item: Optional item to set as ``current_item``
            for the updated position.

        .. versionchanged:: 8.0
            Added the ``current_item`` optional parameter.

        .. versionchanged:: 8.0
            Only render when the number of steps meets the
            ``update_min_steps`` threshold.
        """
        if current_item is not None:
            self.current_item = current_item
        self._completed_intervals += n_steps
        if self._completed_intervals >= self.update_min_steps:
            self.make_step(self._completed_intervals)
            self.render_progress()
            self._completed_intervals = 0
    
    def finish(self) -> None:
        self.eta_known = False
        self.current_item = None
        self.finished = True
    
    def generator(self) -> t.Iterator[V]:
        """Return a generator which yields the items added to the bar
        during construction, and updates the progress bar *after* the
        yielded block returns.
        """
        if not self.entered:
            raise RuntimeError('You need to use progress bars in a with block.')
        if self.is_hidden:
            yield from self.iter
        else:
            for rv in self.iter:
                self.current_item = rv
                if self._completed_intervals == 0:
                    self.render_progress()
                yield rv
                self.update(1)
            self.finish()
            self.render_progress()


def pager(generator: t.Iterable[str], color: t.Optional[bool] = None) -> None:
    """Decide what method to use for paging through text."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('click._termui_impl.pager', 'pager(generator, color=None)', {'_default_text_stdout': _default_text_stdout, 'StringIO': StringIO, 'isatty': isatty, 'sys': sys, '_nullpager': _nullpager, 'os': os, 'WIN': WIN, '_tempfilepager': _tempfilepager, '_pipepager': _pipepager, 'generator': generator, 'color': color, 't': t, 'str': str, 't': t, 'bool': bool}, 1)

def _pipepager(generator: t.Iterable[str], cmd: str, color: t.Optional[bool]) -> None:
    """Page through text by feeding it to another program.  Invoking a
    pager through this might support colors.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('click._termui_impl._pipepager', '_pipepager(generator, cmd, color)', {'os': os, 't': t, 'get_best_encoding': get_best_encoding, 'strip_ansi': strip_ansi, 'generator': generator, 'cmd': cmd, 'color': color, 't': t, 'str': str, 't': t, 'bool': bool}, 0)

def _tempfilepager(generator: t.Iterable[str], cmd: str, color: t.Optional[bool]) -> None:
    """Page through text by invoking a program on a temporary file."""
    import custom_funtemplate
    custom_funtemplate.rewrite_template('click._termui_impl._tempfilepager', '_tempfilepager(generator, cmd, color)', {'strip_ansi': strip_ansi, 'get_best_encoding': get_best_encoding, 'sys': sys, 'open_stream': open_stream, 'os': os, 'generator': generator, 'cmd': cmd, 'color': color, 't': t, 'str': str, 't': t, 'bool': bool}, 0)

def _nullpager(stream: t.TextIO, generator: t.Iterable[str], color: t.Optional[bool]) -> None:
    """Simply print unformatted text.  This is the ultimate fallback."""
    import custom_funtemplate
    custom_funtemplate.rewrite_template('click._termui_impl._nullpager', '_nullpager(stream, generator, color)', {'strip_ansi': strip_ansi, 'stream': stream, 'generator': generator, 'color': color, 't': t, 't': t, 'str': str, 't': t, 'bool': bool}, 0)


class Editor:
    
    def __init__(self, editor: t.Optional[str] = None, env: t.Optional[t.Mapping[(str, str)]] = None, require_save: bool = True, extension: str = '.txt') -> None:
        self.editor = editor
        self.env = env
        self.require_save = require_save
        self.extension = extension
    
    def get_editor(self) -> str:
        if self.editor is not None:
            return self.editor
        for key in ('VISUAL', 'EDITOR'):
            rv = os.environ.get(key)
            if rv:
                return rv
        if WIN:
            return 'notepad'
        for editor in ('sensible-editor', 'vim', 'nano'):
            if os.system(f'which {editor} >/dev/null 2>&1') == 0:
                return editor
        return 'vi'
    
    def edit_file(self, filename: str) -> None:
        import subprocess
        editor = self.get_editor()
        environ: t.Optional[t.Dict[(str, str)]] = None
        if self.env:
            environ = os.environ.copy()
            environ.update(self.env)
        try:
            c = subprocess.Popen(f'{editor} "{filename}"', env=environ, shell=True)
            exit_code = c.wait()
            if exit_code != 0:
                raise ClickException(_('{editor}: Editing failed').format(editor=editor))
        except OSError as e:
            raise ClickException(_('{editor}: Editing failed: {e}').format(editor=editor, e=e)) from e
    
    def edit(self, text: t.Optional[t.AnyStr]) -> t.Optional[t.AnyStr]:
        import tempfile
        if not text:
            data = b''
        elif isinstance(text, (bytes, bytearray)):
            data = text
        else:
            if (text and not text.endswith('\n')):
                text += '\n'
            if WIN:
                data = text.replace('\n', '\r\n').encode('utf-8-sig')
            else:
                data = text.encode('utf-8')
        (fd, name) = tempfile.mkstemp(prefix='editor-', suffix=self.extension)
        f: t.BinaryIO
        try:
            with os.fdopen(fd, 'wb') as f:
                f.write(data)
            os.utime(name, (os.path.getatime(name), os.path.getmtime(name) - 2))
            timestamp = os.path.getmtime(name)
            self.edit_file(name)
            if (self.require_save and os.path.getmtime(name) == timestamp):
                return None
            with open(name, 'rb') as f:
                rv = f.read()
            if isinstance(text, (bytes, bytearray)):
                return rv
            return rv.decode('utf-8-sig').replace('\r\n', '\n')
        finally:
            os.unlink(name)


def open_url(url: str, wait: bool = False, locate: bool = False) -> int:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('click._termui_impl.open_url', 'open_url(url, wait=False, locate=False)', {'sys': sys, 'WIN': WIN, 'os': os, 'CYGWIN': CYGWIN, 'url': url, 'wait': wait, 'locate': locate}, 1)

def _translate_ch_to_exc(ch: str) -> t.Optional[BaseException]:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('click._termui_impl._translate_ch_to_exc', '_translate_ch_to_exc(ch)', {'WIN': WIN, 'ch': ch, 't': t, 'BaseException': BaseException}, 1)
if WIN:
    import msvcrt
    
    @contextlib.contextmanager
    def raw_terminal() -> t.Iterator[int]:
        yield -1
    
    def getchar(echo: bool) -> str:
        import custom_funtemplate
        return custom_funtemplate.rewrite_template('click._termui_impl.getchar', 'getchar(echo)', {'t': t, 'msvcrt': msvcrt, '_translate_ch_to_exc': _translate_ch_to_exc, 'echo': echo}, 1)
else:
    import tty
    import termios
    
    @contextlib.contextmanager
    def raw_terminal() -> t.Iterator[int]:
        import custom_funtemplate
        custom_funtemplate.rewrite_template('click._termui_impl.raw_terminal', 'raw_terminal()', {'t': t, 'isatty': isatty, 'sys': sys, 'termios': termios, 'tty': tty, 'contextlib': contextlib, 't': t, 'int': int}, 0)
    
    def getchar(echo: bool) -> str:
        import custom_funtemplate
        return custom_funtemplate.rewrite_template('click._termui_impl.getchar', 'getchar(echo)', {'raw_terminal': raw_terminal, 'os': os, 'get_best_encoding': get_best_encoding, 'sys': sys, 'isatty': isatty, '_translate_ch_to_exc': _translate_ch_to_exc, 'echo': echo}, 1)

