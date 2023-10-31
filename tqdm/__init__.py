from ._monitor import TMonitor, TqdmSynchronisationWarning
from ._tqdm_pandas import tqdm_pandas
from .cli import main
from .gui import tqdm as tqdm_gui
from .gui import trange as tgrange
from .std import TqdmDeprecationWarning, TqdmExperimentalWarning, TqdmKeyError, TqdmMonitorWarning, TqdmTypeError, TqdmWarning, tqdm, trange
from .version import __version__
__all__ = ['tqdm', 'tqdm_gui', 'trange', 'tgrange', 'tqdm_pandas', 'tqdm_notebook', 'tnrange', 'main', 'TMonitor', 'TqdmTypeError', 'TqdmKeyError', 'TqdmWarning', 'TqdmDeprecationWarning', 'TqdmExperimentalWarning', 'TqdmMonitorWarning', 'TqdmSynchronisationWarning', '__version__']

def tqdm_notebook(*args, **kwargs):
    """See tqdm.notebook.tqdm for full documentation"""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('tqdm.__init__.tqdm_notebook', 'tqdm_notebook(*args, **kwargs)', {'TqdmDeprecationWarning': TqdmDeprecationWarning, 'args': args, 'kwargs': kwargs}, 1)

def tnrange(*args, **kwargs):
    """Shortcut for `tqdm.notebook.tqdm(range(*args), **kwargs)`."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('tqdm.__init__.tnrange', 'tnrange(*args, **kwargs)', {'TqdmDeprecationWarning': TqdmDeprecationWarning, 'args': args, 'kwargs': kwargs}, 1)

