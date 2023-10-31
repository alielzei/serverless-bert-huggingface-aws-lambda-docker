"""Dynamic Protobuf class creator."""

from collections import OrderedDict
import hashlib
import os
from google.protobuf import descriptor_pb2
from google.protobuf import descriptor
from google.protobuf import message_factory

def _GetMessageFromFactory(factory, full_name):
    """Get a proto class from the MessageFactory by name.

  Args:
    factory: a MessageFactory instance.
    full_name: str, the fully qualified name of the proto type.
  Returns:
    A class, for the type identified by full_name.
  Raises:
    KeyError, if the proto is not found in the factory's descriptor pool.
  """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.proto_builder._GetMessageFromFactory', '_GetMessageFromFactory(factory, full_name)', {'factory': factory, 'full_name': full_name}, 1)

def MakeSimpleProtoClass(fields, full_name=None, pool=None):
    """Create a Protobuf class whose fields are basic types.

  Note: this doesn't validate field names!

  Args:
    fields: dict of {name: field_type} mappings for each field in the proto. If
        this is an OrderedDict the order will be maintained, otherwise the
        fields will be sorted by name.
    full_name: optional str, the fully-qualified name of the proto type.
    pool: optional DescriptorPool instance.
  Returns:
    a class, the new protobuf class with a FileDescriptor.
  """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.proto_builder.MakeSimpleProtoClass', 'MakeSimpleProtoClass(fields, full_name=None, pool=None)', {'message_factory': message_factory, '_GetMessageFromFactory': _GetMessageFromFactory, 'OrderedDict': OrderedDict, 'hashlib': hashlib, '_MakeFileDescriptorProto': _MakeFileDescriptorProto, 'fields': fields, 'full_name': full_name, 'pool': pool}, 1)

def _MakeFileDescriptorProto(proto_file_name, full_name, field_items):
    """Populate FileDescriptorProto for MessageFactory's DescriptorPool."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.proto_builder._MakeFileDescriptorProto', '_MakeFileDescriptorProto(proto_file_name, full_name, field_items)', {'descriptor_pb2': descriptor_pb2, 'os': os, 'descriptor': descriptor, 'proto_file_name': proto_file_name, 'full_name': full_name, 'field_items': field_items}, 1)

