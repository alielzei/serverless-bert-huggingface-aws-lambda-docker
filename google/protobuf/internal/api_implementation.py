"""Determine which implementation of the protobuf API is used in this process.
"""

import os
import sys
import warnings
try:
    from google.protobuf.internal import _api_implementation
    _api_version = _api_implementation.api_version
except ImportError:
    _api_version = -1
if _api_version == 1:
    raise ValueError('api_version=1 is no longer supported.')
_default_implementation_type = ('cpp' if _api_version > 0 else 'python')
_implementation_type = os.getenv('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION', _default_implementation_type)
if _implementation_type != 'python':
    _implementation_type = 'cpp'
if ('PyPy' in sys.version and _implementation_type == 'cpp'):
    warnings.warn('PyPy does not work yet with cpp protocol buffers. Falling back to the python implementation.')
    _implementation_type = 'python'
try:
    from google.protobuf import enable_deterministic_proto_serialization
    _python_deterministic_proto_serialization = True
except ImportError:
    _python_deterministic_proto_serialization = False

def Type():
    return _implementation_type

def _SetType(implementation_type):
    """Never use! Only for protobuf benchmark."""
    global _implementation_type
    _implementation_type = implementation_type

def Version():
    return 2

def IsPythonDefaultSerializationDeterministic():
    return _python_deterministic_proto_serialization

