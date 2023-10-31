from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import contextlib
import threading
from past.builtins import basestring
from caffe2.proto import caffe2_pb2
_NAMESCOPE_SEPARATOR = '/'
_threadlocal_scope = threading.local()

def CurrentNameScope():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.scope.CurrentNameScope', 'CurrentNameScope()', {'_threadlocal_scope': _threadlocal_scope}, 1)

def CurrentDeviceScope():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.scope.CurrentDeviceScope', 'CurrentDeviceScope()', {'_threadlocal_scope': _threadlocal_scope}, 1)

@contextlib.contextmanager
def NameScope(prefix, reset=False):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.scope.NameScope', 'NameScope(prefix, reset=False)', {'basestring': basestring, 'CurrentNameScope': CurrentNameScope, '_NAMESCOPE_SEPARATOR': _NAMESCOPE_SEPARATOR, '_threadlocal_scope': _threadlocal_scope, 'contextlib': contextlib, 'prefix': prefix, 'reset': reset}, 0)

@contextlib.contextmanager
def DeviceScope(scope, node_name=None):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.scope.DeviceScope', 'DeviceScope(scope, node_name=None)', {'caffe2_pb2': caffe2_pb2, 'CurrentDeviceScope': CurrentDeviceScope, '_threadlocal_scope': _threadlocal_scope, 'contextlib': contextlib, 'scope': scope, 'node_name': node_name}, 0)

@contextlib.contextmanager
def EmptyNameScope():
    """
    Allow users to 'disable' the name scope behaviour.

    This sets the CurrentNameScope() to None, so that the field is
    not set in CreateOperator(...), etc.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.scope.EmptyNameScope', 'EmptyNameScope()', {'CurrentNameScope': CurrentNameScope, '_threadlocal_scope': _threadlocal_scope, 'contextlib': contextlib}, 1)

@contextlib.contextmanager
def EmptyDeviceScope():
    """
    Allow users to 'disable' the device scope behaviour (so it can be
    controlled at a NetDef::DeviceOption level, not overridden at
    OperatorDef::DeviceOption level).

    This sets the CurrentDeviceScope() to None, so that the field is
    not set in CreateOperator(...), etc.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.scope.EmptyDeviceScope', 'EmptyDeviceScope()', {'CurrentDeviceScope': CurrentDeviceScope, '_threadlocal_scope': _threadlocal_scope, 'contextlib': contextlib}, 1)

