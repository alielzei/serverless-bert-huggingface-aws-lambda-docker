from warnings import warn
from .notebook import *
from .notebook import __all__
from .std import TqdmDeprecationWarning
warn('This function will be removed in tqdm==5.0.0\nPlease use `tqdm.notebook.*` instead of `tqdm._tqdm_notebook.*`', TqdmDeprecationWarning, stacklevel=2)

