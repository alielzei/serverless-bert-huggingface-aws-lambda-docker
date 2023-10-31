"""
Unit tests for the disk utilities.
"""

from __future__ import with_statement
import array
import os
from joblib.disk import disk_used, memstr_to_bytes, mkdirp, rm_subdirs
from joblib.testing import parametrize, raises

def test_disk_used(tmpdir):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_disk.test_disk_used', 'test_disk_used(tmpdir)', {'array': array, 'os': os, 'disk_used': disk_used, 'tmpdir': tmpdir}, 0)

@parametrize('text,value', [('80G', 80 * 1024**3), ('1.4M', int(1.4 * 1024**2)), ('120M', 120 * 1024**2), ('53K', 53 * 1024)])
def test_memstr_to_bytes(text, value):
    assert memstr_to_bytes(text) == value

@parametrize('text,exception,regex', [('fooG', ValueError, 'Invalid literal for size.*fooG.*'), ('1.4N', ValueError, 'Invalid literal for size.*1.4N.*')])
def test_memstr_to_bytes_exception(text, exception, regex):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_disk.test_memstr_to_bytes_exception', 'test_memstr_to_bytes_exception(text, exception, regex)', {'raises': raises, 'memstr_to_bytes': memstr_to_bytes, 'parametrize': parametrize, 'ValueError': ValueError, 'text': text, 'exception': exception, 'regex': regex}, 0)

def test_mkdirp(tmpdir):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_disk.test_mkdirp', 'test_mkdirp(tmpdir)', {'mkdirp': mkdirp, 'os': os, 'raises': raises, 'tmpdir': tmpdir}, 0)

def test_rm_subdirs(tmpdir):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.test_disk.test_rm_subdirs', 'test_rm_subdirs(tmpdir)', {'os': os, 'mkdirp': mkdirp, 'rm_subdirs': rm_subdirs, 'tmpdir': tmpdir}, 0)

