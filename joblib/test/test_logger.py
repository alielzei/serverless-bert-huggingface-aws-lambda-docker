"""
Test the logger module.
"""

import re
from joblib.logger import PrintTime

def test_print_time(tmpdir, capsys):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_logger.test_print_time', 'test_print_time(tmpdir, capsys)', {'PrintTime': PrintTime, 're': re, 'tmpdir': tmpdir, 'capsys': capsys}, 0)

