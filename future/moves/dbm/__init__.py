from __future__ import absolute_import
from future.utils import PY3
if PY3:
    from dbm import *
else:
    __future_module__ = True
    from whichdb import *
    from anydbm import *
if PY3:
    from dbm import ndbm
else:
    try:
        from future.moves.dbm import ndbm
    except ImportError:
        ndbm = None

