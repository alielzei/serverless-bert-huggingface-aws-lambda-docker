"""Generated protocol buffer code."""

from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
_sym_db = _symbol_database.Default()
from google.protobuf import any_pb2 as google_dot_protobuf_dot_any__pb2
from google.protobuf import source_context_pb2 as google_dot_protobuf_dot_source__context__pb2
DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1agoogle/protobuf/type.proto\x12\x0fgoogle.protobuf\x1a\x19google/protobuf/any.proto\x1a$google/protobuf/source_context.proto"\xd7\x01\n\x04Type\x12\x0c\n\x04name\x18\x01 \x01(\t\x12&\n\x06fields\x18\x02 \x03(\x0b2\x16.google.protobuf.Field\x12\x0e\n\x06oneofs\x18\x03 \x03(\t\x12(\n\x07options\x18\x04 \x03(\x0b2\x17.google.protobuf.Option\x126\n\x0esource_context\x18\x05 \x01(\x0b2\x1e.google.protobuf.SourceContext\x12\'\n\x06syntax\x18\x06 \x01(\x0e2\x17.google.protobuf.Syntax"\xd5\x05\n\x05Field\x12)\n\x04kind\x18\x01 \x01(\x0e2\x1b.google.protobuf.Field.Kind\x127\n\x0bcardinality\x18\x02 \x01(\x0e2".google.protobuf.Field.Cardinality\x12\x0e\n\x06number\x18\x03 \x01(\x05\x12\x0c\n\x04name\x18\x04 \x01(\t\x12\x10\n\x08type_url\x18\x06 \x01(\t\x12\x13\n\x0boneof_index\x18\x07 \x01(\x05\x12\x0e\n\x06packed\x18\x08 \x01(\x08\x12(\n\x07options\x18\t \x03(\x0b2\x17.google.protobuf.Option\x12\x11\n\tjson_name\x18\n \x01(\t\x12\x15\n\rdefault_value\x18\x0b \x01(\t"\xc8\x02\n\x04Kind\x12\x10\n\x0cTYPE_UNKNOWN\x10\x00\x12\x0f\n\x0bTYPE_DOUBLE\x10\x01\x12\x0e\n\nTYPE_FLOAT\x10\x02\x12\x0e\n\nTYPE_INT64\x10\x03\x12\x0f\n\x0bTYPE_UINT64\x10\x04\x12\x0e\n\nTYPE_INT32\x10\x05\x12\x10\n\x0cTYPE_FIXED64\x10\x06\x12\x10\n\x0cTYPE_FIXED32\x10\x07\x12\r\n\tTYPE_BOOL\x10\x08\x12\x0f\n\x0bTYPE_STRING\x10\t\x12\x0e\n\nTYPE_GROUP\x10\n\x12\x10\n\x0cTYPE_MESSAGE\x10\x0b\x12\x0e\n\nTYPE_BYTES\x10\x0c\x12\x0f\n\x0bTYPE_UINT32\x10\r\x12\r\n\tTYPE_ENUM\x10\x0e\x12\x11\n\rTYPE_SFIXED32\x10\x0f\x12\x11\n\rTYPE_SFIXED64\x10\x10\x12\x0f\n\x0bTYPE_SINT32\x10\x11\x12\x0f\n\x0bTYPE_SINT64\x10\x12"t\n\x0bCardinality\x12\x17\n\x13CARDINALITY_UNKNOWN\x10\x00\x12\x18\n\x14CARDINALITY_OPTIONAL\x10\x01\x12\x18\n\x14CARDINALITY_REQUIRED\x10\x02\x12\x18\n\x14CARDINALITY_REPEATED\x10\x03"\xce\x01\n\x04Enum\x12\x0c\n\x04name\x18\x01 \x01(\t\x12-\n\tenumvalue\x18\x02 \x03(\x0b2\x1a.google.protobuf.EnumValue\x12(\n\x07options\x18\x03 \x03(\x0b2\x17.google.protobuf.Option\x126\n\x0esource_context\x18\x04 \x01(\x0b2\x1e.google.protobuf.SourceContext\x12\'\n\x06syntax\x18\x05 \x01(\x0e2\x17.google.protobuf.Syntax"S\n\tEnumValue\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0e\n\x06number\x18\x02 \x01(\x05\x12(\n\x07options\x18\x03 \x03(\x0b2\x17.google.protobuf.Option";\n\x06Option\x12\x0c\n\x04name\x18\x01 \x01(\t\x12#\n\x05value\x18\x02 \x01(\x0b2\x14.google.protobuf.Any*.\n\x06Syntax\x12\x11\n\rSYNTAX_PROTO2\x10\x00\x12\x11\n\rSYNTAX_PROTO3\x10\x01B{\n\x13com.google.protobufB\tTypeProtoP\x01Z-google.golang.org/protobuf/types/known/typepb\xf8\x01\x01\xa2\x02\x03GPB\xaa\x02\x1eGoogle.Protobuf.WellKnownTypesb\x06proto3')
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'google.protobuf.type_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:
    DESCRIPTOR._options = None
    DESCRIPTOR._serialized_options = b'\n\x13com.google.protobufB\tTypeProtoP\x01Z-google.golang.org/protobuf/types/known/typepb\xf8\x01\x01\xa2\x02\x03GPB\xaa\x02\x1eGoogle.Protobuf.WellKnownTypes'
    _SYNTAX._serialized_start = 1413
    _SYNTAX._serialized_end = 1459
    _TYPE._serialized_start = 113
    _TYPE._serialized_end = 328
    _FIELD._serialized_start = 331
    _FIELD._serialized_end = 1056
    _FIELD_KIND._serialized_start = 610
    _FIELD_KIND._serialized_end = 938
    _FIELD_CARDINALITY._serialized_start = 940
    _FIELD_CARDINALITY._serialized_end = 1056
    _ENUM._serialized_start = 1059
    _ENUM._serialized_end = 1265
    _ENUMVALUE._serialized_start = 1267
    _ENUMVALUE._serialized_end = 1350
    _OPTION._serialized_start = 1352
    _OPTION._serialized_end = 1411

