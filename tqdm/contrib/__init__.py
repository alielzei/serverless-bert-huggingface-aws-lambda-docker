"""
Thin wrappers around common functions.

Subpackages contain potentially unstable extensions.
"""

from warnings import warn
from ..auto import tqdm as tqdm_auto
from ..std import TqdmDeprecationWarning, tqdm
from ..utils import ObjectWrapper
__author__ = {'github.com/': ['casperdcl']}
__all__ = ['tenumerate', 'tzip', 'tmap']


class DummyTqdmFile(ObjectWrapper):
    """Dummy file-like that will write to tqdm"""
    
    def __init__(self, wrapped):
        super(DummyTqdmFile, self).__init__(wrapped)
        self._buf = []
    
    def write(self, x, nolock=False):
        nl = (b'\n' if isinstance(x, bytes) else '\n')
        (pre, sep, post) = x.rpartition(nl)
        if sep:
            blank = type(nl)()
            tqdm.write(blank.join(self._buf + [pre, sep]), end=blank, file=self._wrapped, nolock=nolock)
            self._buf = [post]
        else:
            self._buf.append(x)
    
    def __del__(self):
        if self._buf:
            blank = type(self._buf[0])()
            try:
                tqdm.write(blank.join(self._buf), end=blank, file=self._wrapped)
            except (OSError, ValueError):
                pass


def builtin_iterable(func):
    """Returns `func`"""
    warn('This function has no effect, and will be removed in tqdm==5.0.0', TqdmDeprecationWarning, stacklevel=2)
    return func

def tenumerate(iterable, start=0, total=None, tqdm_class=tqdm_auto, **tqdm_kwargs):
    """
    Equivalent of `numpy.ndenumerate` or builtin `enumerate`.

    Parameters
    ----------
    tqdm_class  : [default: tqdm.auto.tqdm].
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('tqdm.contrib.__init__.tenumerate', 'tenumerate(iterable, start=0, total=None, tqdm_class=tqdm_auto, **tqdm_kwargs)', {'iterable': iterable, 'start': start, 'total': total, 'tqdm_class': tqdm_class, 'tqdm_auto': tqdm_auto, 'tqdm_kwargs': tqdm_kwargs}, 1)

def tzip(iter1, *iter2plus, **tqdm_kwargs):
    """
    Equivalent of builtin `zip`.

    Parameters
    ----------
    tqdm_class  : [default: tqdm.auto.tqdm].
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('tqdm.contrib.__init__.tzip', 'tzip(iter1, *iter2plus, **tqdm_kwargs)', {'tqdm_auto': tqdm_auto, 'iter1': iter1, 'iter2plus': iter2plus, 'tqdm_kwargs': tqdm_kwargs}, 0)

def tmap(function, *sequences, **tqdm_kwargs):
    """
    Equivalent of builtin `map`.

    Parameters
    ----------
    tqdm_class  : [default: tqdm.auto.tqdm].
    """
    for i in tzip(*sequences, **tqdm_kwargs):
        yield function(*i)

