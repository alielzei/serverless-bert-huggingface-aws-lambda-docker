from warnings import warn
from .gui import *
from .gui import __all__
from .std import TqdmDeprecationWarning
warn('This function will be removed in tqdm==5.0.0\nPlease use `tqdm.gui.*` instead of `tqdm._tqdm_gui.*`', TqdmDeprecationWarning, stacklevel=2)

