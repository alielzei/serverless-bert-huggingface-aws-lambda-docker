from warnings import warn
from .cli import *
from .cli import __all__
from .std import TqdmDeprecationWarning
warn('This function will be removed in tqdm==5.0.0\nPlease use `tqdm.cli.*` instead of `tqdm._main.*`', TqdmDeprecationWarning, stacklevel=2)

