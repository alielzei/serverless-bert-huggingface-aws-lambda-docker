"""Generated protocol buffer code."""

from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
_sym_db = _symbol_database.Default()
DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1cgoogle/protobuf/struct.proto\x12\x0fgoogle.protobuf"\x84\x01\n\x06Struct\x123\n\x06fields\x18\x01 \x03(\x0b2#.google.protobuf.Struct.FieldsEntry\x1aE\n\x0bFieldsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12%\n\x05value\x18\x02 \x01(\x0b2\x16.google.protobuf.Value:\x028\x01"\xea\x01\n\x05Value\x120\n\nnull_value\x18\x01 \x01(\x0e2\x1a.google.protobuf.NullValueH\x00\x12\x16\n\x0cnumber_value\x18\x02 \x01(\x01H\x00\x12\x16\n\x0cstring_value\x18\x03 \x01(\tH\x00\x12\x14\n\nbool_value\x18\x04 \x01(\x08H\x00\x12/\n\x0cstruct_value\x18\x05 \x01(\x0b2\x17.google.protobuf.StructH\x00\x120\n\nlist_value\x18\x06 \x01(\x0b2\x1a.google.protobuf.ListValueH\x00B\x06\n\x04kind"3\n\tListValue\x12&\n\x06values\x18\x01 \x03(\x0b2\x16.google.protobuf.Value*\x1b\n\tNullValue\x12\x0e\n\nNULL_VALUE\x10\x00B\x7f\n\x13com.google.protobufB\x0bStructProtoP\x01Z/google.golang.org/protobuf/types/known/structpb\xf8\x01\x01\xa2\x02\x03GPB\xaa\x02\x1eGoogle.Protobuf.WellKnownTypesb\x06proto3')
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'google.protobuf.struct_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:
    DESCRIPTOR._options = None
    DESCRIPTOR._serialized_options = b'\n\x13com.google.protobufB\x0bStructProtoP\x01Z/google.golang.org/protobuf/types/known/structpb\xf8\x01\x01\xa2\x02\x03GPB\xaa\x02\x1eGoogle.Protobuf.WellKnownTypes'
    _STRUCT_FIELDSENTRY._options = None
    _STRUCT_FIELDSENTRY._serialized_options = b'8\x01'
    _NULLVALUE._serialized_start = 474
    _NULLVALUE._serialized_end = 501
    _STRUCT._serialized_start = 50
    _STRUCT._serialized_end = 182
    _STRUCT_FIELDSENTRY._serialized_start = 113
    _STRUCT_FIELDSENTRY._serialized_end = 182
    _VALUE._serialized_start = 185
    _VALUE._serialized_end = 419
    _LISTVALUE._serialized_start = 421
    _LISTVALUE._serialized_end = 472

