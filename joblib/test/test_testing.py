import sys
import re
from joblib.testing import raises, check_subprocess_call

def test_check_subprocess_call():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_testing.test_check_subprocess_call', 'test_check_subprocess_call()', {'check_subprocess_call': check_subprocess_call, 'sys': sys}, 0)

def test_check_subprocess_call_non_matching_regex():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_testing.test_check_subprocess_call_non_matching_regex', 'test_check_subprocess_call_non_matching_regex()', {'raises': raises, 'check_subprocess_call': check_subprocess_call, 'sys': sys}, 0)

def test_check_subprocess_call_wrong_command():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_testing.test_check_subprocess_call_wrong_command', 'test_check_subprocess_call_wrong_command()', {'raises': raises, 'check_subprocess_call': check_subprocess_call}, 0)

def test_check_subprocess_call_non_zero_return_code():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_testing.test_check_subprocess_call_non_zero_return_code', 'test_check_subprocess_call_non_zero_return_code()', {'re': re, 'raises': raises, 'check_subprocess_call': check_subprocess_call, 'sys': sys}, 0)

def test_check_subprocess_call_timeout():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_testing.test_check_subprocess_call_timeout', 'test_check_subprocess_call_timeout()', {'re': re, 'raises': raises, 'check_subprocess_call': check_subprocess_call, 'sys': sys}, 0)

