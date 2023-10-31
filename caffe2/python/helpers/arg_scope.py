from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import contextlib
import copy
import threading
_threadlocal_scope = threading.local()

@contextlib.contextmanager
def arg_scope(single_helper_or_list, **kwargs):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.helpers.arg_scope.arg_scope', 'arg_scope(single_helper_or_list, **kwargs)', {'copy': copy, 'get_current_scope': get_current_scope, '_threadlocal_scope': _threadlocal_scope, 'contextlib': contextlib, 'single_helper_or_list': single_helper_or_list, 'kwargs': kwargs}, 0)

def get_current_scope():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.helpers.arg_scope.get_current_scope', 'get_current_scope()', {'_threadlocal_scope': _threadlocal_scope}, 1)

