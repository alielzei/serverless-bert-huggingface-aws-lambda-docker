import sys
if sys.version_info < (3, 8):
    try:
        import pickle5 as pickle
        from pickle5 import Pickler
    except ImportError:
        import pickle
        from pickle import _Pickler as Pickler
else:
    import pickle
    from pickle import Pickler

