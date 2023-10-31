"""Builds descriptors, message classes and services for generated _pb2.py.

This file is only called in python generated _pb2.py files. It builds
descriptors, message classes and services that users can directly use
in generated code.
"""

__author__ = 'jieluo@google.com (Jie Luo)'
from google.protobuf.internal import enum_type_wrapper
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
_sym_db = _symbol_database.Default()

def BuildMessageAndEnumDescriptors(file_des, module):
    """Builds message and enum descriptors.

  Args:
    file_des: FileDescriptor of the .proto file
    module: Generated _pb2 module
  """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('google.protobuf.internal.builder.BuildMessageAndEnumDescriptors', 'BuildMessageAndEnumDescriptors(file_des, module)', {'file_des': file_des, 'module': module}, 0)

def BuildTopDescriptorsAndMessages(file_des, module_name, module):
    """Builds top level descriptors and message classes.

  Args:
    file_des: FileDescriptor of the .proto file
    module_name: str, the name of generated _pb2 module
    module: Generated _pb2 module
  """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.builder.BuildTopDescriptorsAndMessages', 'BuildTopDescriptorsAndMessages(file_des, module_name, module)', {'_reflection': _reflection, '_message': _message, '_sym_db': _sym_db, 'enum_type_wrapper': enum_type_wrapper, 'file_des': file_des, 'module_name': module_name, 'module': module}, 1)

def BuildServices(file_des, module_name, module):
    """Builds services classes and services stub class.

  Args:
    file_des: FileDescriptor of the .proto file
    module_name: str, the name of generated _pb2 module
    module: Generated _pb2 module
  """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('google.protobuf.internal.builder.BuildServices', 'BuildServices(file_des, module_name, module)', {'file_des': file_des, 'module_name': module_name, 'module': module}, 0)

