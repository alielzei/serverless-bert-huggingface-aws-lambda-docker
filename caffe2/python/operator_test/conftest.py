from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import caffe2.python.serialized_test.serialized_test_util as serial

def pytest_addoption(parser):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.operator_test.conftest.pytest_addoption', 'pytest_addoption(parser)', {'serial': serial, 'parser': parser}, 0)

def pytest_configure(config):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.operator_test.conftest.pytest_configure', 'pytest_configure(config)', {'serial': serial, 'config': config}, 0)

