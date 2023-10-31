"""
Pyodide and other single-threaded Python builds will be missing the
_multiprocessing module. Test that joblib still works in this environment.
"""

import os
import subprocess
import sys

def test_missing_multiprocessing(tmp_path):
    """
    Test that import joblib works even if _multiprocessing is missing.

    pytest has already imported everything from joblib. The most reasonable way
    to test importing joblib with modified environment is to invoke a separate
    Python process. This also ensures that we don't break other tests by
    importing a bad `_multiprocessing` module.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_missing_multiprocessing.test_missing_multiprocessing', 'test_missing_multiprocessing(tmp_path)', {'os': os, 'sys': sys, 'subprocess': subprocess, 'tmp_path': tmp_path}, 0)

