from warnings import warn
from .std import *
from .std import __all__
from .std import TqdmDeprecationWarning
warn('This function will be removed in tqdm==5.0.0\nPlease use `tqdm.std.*` instead of `tqdm._tqdm.*`', TqdmDeprecationWarning, stacklevel=2)

