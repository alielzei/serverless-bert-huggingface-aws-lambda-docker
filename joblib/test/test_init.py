try:
    from joblib import *
    _top_import_error = None
except Exception as ex:
    _top_import_error = ex

def test_import_joblib():
    assert _top_import_error is None

