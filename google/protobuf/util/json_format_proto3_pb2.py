"""Generated protocol buffer code."""

from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
_sym_db = _symbol_database.Default()
from google.protobuf import any_pb2 as google_dot_protobuf_dot_any__pb2
from google.protobuf import duration_pb2 as google_dot_protobuf_dot_duration__pb2
from google.protobuf import field_mask_pb2 as google_dot_protobuf_dot_field__mask__pb2
from google.protobuf import struct_pb2 as google_dot_protobuf_dot_struct__pb2
from google.protobuf import timestamp_pb2 as google_dot_protobuf_dot_timestamp__pb2
from google.protobuf import wrappers_pb2 as google_dot_protobuf_dot_wrappers__pb2
from google.protobuf import unittest_pb2 as google_dot_protobuf_dot_unittest__pb2
DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n-google/protobuf/util/json_format_proto3.proto\x12\x06proto3\x1a\x19google/protobuf/any.proto\x1a\x1egoogle/protobuf/duration.proto\x1a google/protobuf/field_mask.proto\x1a\x1cgoogle/protobuf/struct.proto\x1a\x1fgoogle/protobuf/timestamp.proto\x1a\x1egoogle/protobuf/wrappers.proto\x1a\x1egoogle/protobuf/unittest.proto"\x1c\n\x0bMessageType\x12\r\n\x05value\x18\x01 \x01(\x05"\x94\x05\n\x0bTestMessage\x12\x12\n\nbool_value\x18\x01 \x01(\x08\x12\x13\n\x0bint32_value\x18\x02 \x01(\x05\x12\x13\n\x0bint64_value\x18\x03 \x01(\x03\x12\x14\n\x0cuint32_value\x18\x04 \x01(\r\x12\x14\n\x0cuint64_value\x18\x05 \x01(\x04\x12\x13\n\x0bfloat_value\x18\x06 \x01(\x02\x12\x14\n\x0cdouble_value\x18\x07 \x01(\x01\x12\x14\n\x0cstring_value\x18\x08 \x01(\t\x12\x13\n\x0bbytes_value\x18\t \x01(\x0c\x12$\n\nenum_value\x18\n \x01(\x0e2\x10.proto3.EnumType\x12*\n\rmessage_value\x18\x0b \x01(\x0b2\x13.proto3.MessageType\x12\x1b\n\x13repeated_bool_value\x18\x15 \x03(\x08\x12\x1c\n\x14repeated_int32_value\x18\x16 \x03(\x05\x12\x1c\n\x14repeated_int64_value\x18\x17 \x03(\x03\x12\x1d\n\x15repeated_uint32_value\x18\x18 \x03(\r\x12\x1d\n\x15repeated_uint64_value\x18\x19 \x03(\x04\x12\x1c\n\x14repeated_float_value\x18\x1a \x03(\x02\x12\x1d\n\x15repeated_double_value\x18\x1b \x03(\x01\x12\x1d\n\x15repeated_string_value\x18\x1c \x03(\t\x12\x1c\n\x14repeated_bytes_value\x18\x1d \x03(\x0c\x12-\n\x13repeated_enum_value\x18\x1e \x03(\x0e2\x10.proto3.EnumType\x123\n\x16repeated_message_value\x18\x1f \x03(\x0b2\x13.proto3.MessageType"\x8c\x02\n\tTestOneof\x12\x1b\n\x11oneof_int32_value\x18\x01 \x01(\x05H\x00\x12\x1c\n\x12oneof_string_value\x18\x02 \x01(\tH\x00\x12\x1b\n\x11oneof_bytes_value\x18\x03 \x01(\x0cH\x00\x12,\n\x10oneof_enum_value\x18\x04 \x01(\x0e2\x10.proto3.EnumTypeH\x00\x122\n\x13oneof_message_value\x18\x05 \x01(\x0b2\x13.proto3.MessageTypeH\x00\x126\n\x10oneof_null_value\x18\x06 \x01(\x0e2\x1a.google.protobuf.NullValueH\x00B\r\n\x0boneof_value"\xe1\x04\n\x07TestMap\x12.\n\x08bool_map\x18\x01 \x03(\x0b2\x1c.proto3.TestMap.BoolMapEntry\x120\n\tint32_map\x18\x02 \x03(\x0b2\x1d.proto3.TestMap.Int32MapEntry\x120\n\tint64_map\x18\x03 \x03(\x0b2\x1d.proto3.TestMap.Int64MapEntry\x122\n\nuint32_map\x18\x04 \x03(\x0b2\x1e.proto3.TestMap.Uint32MapEntry\x122\n\nuint64_map\x18\x05 \x03(\x0b2\x1e.proto3.TestMap.Uint64MapEntry\x122\n\nstring_map\x18\x06 \x03(\x0b2\x1e.proto3.TestMap.StringMapEntry\x1a.\n\x0cBoolMapEntry\x12\x0b\n\x03key\x18\x01 \x01(\x08\x12\r\n\x05value\x18\x02 \x01(\x05:\x028\x01\x1a/\n\rInt32MapEntry\x12\x0b\n\x03key\x18\x01 \x01(\x05\x12\r\n\x05value\x18\x02 \x01(\x05:\x028\x01\x1a/\n\rInt64MapEntry\x12\x0b\n\x03key\x18\x01 \x01(\x03\x12\r\n\x05value\x18\x02 \x01(\x05:\x028\x01\x1a0\n\x0eUint32MapEntry\x12\x0b\n\x03key\x18\x01 \x01(\r\x12\r\n\x05value\x18\x02 \x01(\x05:\x028\x01\x1a0\n\x0eUint64MapEntry\x12\x0b\n\x03key\x18\x01 \x01(\x04\x12\r\n\x05value\x18\x02 \x01(\x05:\x028\x01\x1a0\n\x0eStringMapEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x05:\x028\x01"\x85\x06\n\rTestNestedMap\x124\n\x08bool_map\x18\x01 \x03(\x0b2".proto3.TestNestedMap.BoolMapEntry\x126\n\tint32_map\x18\x02 \x03(\x0b2#.proto3.TestNestedMap.Int32MapEntry\x126\n\tint64_map\x18\x03 \x03(\x0b2#.proto3.TestNestedMap.Int64MapEntry\x128\n\nuint32_map\x18\x04 \x03(\x0b2$.proto3.TestNestedMap.Uint32MapEntry\x128\n\nuint64_map\x18\x05 \x03(\x0b2$.proto3.TestNestedMap.Uint64MapEntry\x128\n\nstring_map\x18\x06 \x03(\x0b2$.proto3.TestNestedMap.StringMapEntry\x122\n\x07map_map\x18\x07 \x03(\x0b2!.proto3.TestNestedMap.MapMapEntry\x1a.\n\x0cBoolMapEntry\x12\x0b\n\x03key\x18\x01 \x01(\x08\x12\r\n\x05value\x18\x02 \x01(\x05:\x028\x01\x1a/\n\rInt32MapEntry\x12\x0b\n\x03key\x18\x01 \x01(\x05\x12\r\n\x05value\x18\x02 \x01(\x05:\x028\x01\x1a/\n\rInt64MapEntry\x12\x0b\n\x03key\x18\x01 \x01(\x03\x12\r\n\x05value\x18\x02 \x01(\x05:\x028\x01\x1a0\n\x0eUint32MapEntry\x12\x0b\n\x03key\x18\x01 \x01(\r\x12\r\n\x05value\x18\x02 \x01(\x05:\x028\x01\x1a0\n\x0eUint64MapEntry\x12\x0b\n\x03key\x18\x01 \x01(\x04\x12\r\n\x05value\x18\x02 \x01(\x05:\x028\x01\x1a0\n\x0eStringMapEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\x05:\x028\x01\x1aD\n\x0bMapMapEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12$\n\x05value\x18\x02 \x01(\x0b2\x15.proto3.TestNestedMap:\x028\x01"{\n\rTestStringMap\x128\n\nstring_map\x18\x01 \x03(\x0b2$.proto3.TestStringMap.StringMapEntry\x1a0\n\x0eStringMapEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x028\x01"\xee\x07\n\x0bTestWrapper\x12.\n\nbool_value\x18\x01 \x01(\x0b2\x1a.google.protobuf.BoolValue\x120\n\x0bint32_value\x18\x02 \x01(\x0b2\x1b.google.protobuf.Int32Value\x120\n\x0bint64_value\x18\x03 \x01(\x0b2\x1b.google.protobuf.Int64Value\x122\n\x0cuint32_value\x18\x04 \x01(\x0b2\x1c.google.protobuf.UInt32Value\x122\n\x0cuint64_value\x18\x05 \x01(\x0b2\x1c.google.protobuf.UInt64Value\x120\n\x0bfloat_value\x18\x06 \x01(\x0b2\x1b.google.protobuf.FloatValue\x122\n\x0cdouble_value\x18\x07 \x01(\x0b2\x1c.google.protobuf.DoubleValue\x122\n\x0cstring_value\x18\x08 \x01(\x0b2\x1c.google.protobuf.StringValue\x120\n\x0bbytes_value\x18\t \x01(\x0b2\x1b.google.protobuf.BytesValue\x127\n\x13repeated_bool_value\x18\x0b \x03(\x0b2\x1a.google.protobuf.BoolValue\x129\n\x14repeated_int32_value\x18\x0c \x03(\x0b2\x1b.google.protobuf.Int32Value\x129\n\x14repeated_int64_value\x18\r \x03(\x0b2\x1b.google.protobuf.Int64Value\x12;\n\x15repeated_uint32_value\x18\x0e \x03(\x0b2\x1c.google.protobuf.UInt32Value\x12;\n\x15repeated_uint64_value\x18\x0f \x03(\x0b2\x1c.google.protobuf.UInt64Value\x129\n\x14repeated_float_value\x18\x10 \x03(\x0b2\x1b.google.protobuf.FloatValue\x12;\n\x15repeated_double_value\x18\x11 \x03(\x0b2\x1c.google.protobuf.DoubleValue\x12;\n\x15repeated_string_value\x18\x12 \x03(\x0b2\x1c.google.protobuf.StringValue\x129\n\x14repeated_bytes_value\x18\x13 \x03(\x0b2\x1b.google.protobuf.BytesValue"n\n\rTestTimestamp\x12)\n\x05value\x18\x01 \x01(\x0b2\x1a.google.protobuf.Timestamp\x122\n\x0erepeated_value\x18\x02 \x03(\x0b2\x1a.google.protobuf.Timestamp"k\n\x0cTestDuration\x12(\n\x05value\x18\x01 \x01(\x0b2\x19.google.protobuf.Duration\x121\n\x0erepeated_value\x18\x02 \x03(\x0b2\x19.google.protobuf.Duration":\n\rTestFieldMask\x12)\n\x05value\x18\x01 \x01(\x0b2\x1a.google.protobuf.FieldMask"e\n\nTestStruct\x12&\n\x05value\x18\x01 \x01(\x0b2\x17.google.protobuf.Struct\x12/\n\x0erepeated_value\x18\x02 \x03(\x0b2\x17.google.protobuf.Struct"\\\n\x07TestAny\x12#\n\x05value\x18\x01 \x01(\x0b2\x14.google.protobuf.Any\x12,\n\x0erepeated_value\x18\x02 \x03(\x0b2\x14.google.protobuf.Any"b\n\tTestValue\x12%\n\x05value\x18\x01 \x01(\x0b2\x16.google.protobuf.Value\x12.\n\x0erepeated_value\x18\x02 \x03(\x0b2\x16.google.protobuf.Value"n\n\rTestListValue\x12)\n\x05value\x18\x01 \x01(\x0b2\x1a.google.protobuf.ListValue\x122\n\x0erepeated_value\x18\x02 \x03(\x0b2\x1a.google.protobuf.ListValue"\x89\x01\n\rTestBoolValue\x12\x12\n\nbool_value\x18\x01 \x01(\x08\x124\n\x08bool_map\x18\x02 \x03(\x0b2".proto3.TestBoolValue.BoolMapEntry\x1a.\n\x0cBoolMapEntry\x12\x0b\n\x03key\x18\x01 \x01(\x08\x12\r\n\x05value\x18\x02 \x01(\x05:\x028\x01"+\n\x12TestCustomJsonName\x12\x15\n\x05value\x18\x01 \x01(\x05R\x06@value"J\n\x0eTestExtensions\x128\n\nextensions\x18\x01 \x01(\x0b2$.protobuf_unittest.TestAllExtensions"\x84\x01\n\rTestEnumValue\x12%\n\x0benum_value1\x18\x01 \x01(\x0e2\x10.proto3.EnumType\x12%\n\x0benum_value2\x18\x02 \x01(\x0e2\x10.proto3.EnumType\x12%\n\x0benum_value3\x18\x03 \x01(\x0e2\x10.proto3.EnumType*\x1c\n\x08EnumType\x12\x07\n\x03FOO\x10\x00\x12\x07\n\x03BAR\x10\x01B,\n\x18com.google.protobuf.utilB\x10JsonFormatProto3b\x06proto3')
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'google.protobuf.util.json_format_proto3_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:
    DESCRIPTOR._options = None
    DESCRIPTOR._serialized_options = b'\n\x18com.google.protobuf.utilB\x10JsonFormatProto3'
    _TESTMAP_BOOLMAPENTRY._options = None
    _TESTMAP_BOOLMAPENTRY._serialized_options = b'8\x01'
    _TESTMAP_INT32MAPENTRY._options = None
    _TESTMAP_INT32MAPENTRY._serialized_options = b'8\x01'
    _TESTMAP_INT64MAPENTRY._options = None
    _TESTMAP_INT64MAPENTRY._serialized_options = b'8\x01'
    _TESTMAP_UINT32MAPENTRY._options = None
    _TESTMAP_UINT32MAPENTRY._serialized_options = b'8\x01'
    _TESTMAP_UINT64MAPENTRY._options = None
    _TESTMAP_UINT64MAPENTRY._serialized_options = b'8\x01'
    _TESTMAP_STRINGMAPENTRY._options = None
    _TESTMAP_STRINGMAPENTRY._serialized_options = b'8\x01'
    _TESTNESTEDMAP_BOOLMAPENTRY._options = None
    _TESTNESTEDMAP_BOOLMAPENTRY._serialized_options = b'8\x01'
    _TESTNESTEDMAP_INT32MAPENTRY._options = None
    _TESTNESTEDMAP_INT32MAPENTRY._serialized_options = b'8\x01'
    _TESTNESTEDMAP_INT64MAPENTRY._options = None
    _TESTNESTEDMAP_INT64MAPENTRY._serialized_options = b'8\x01'
    _TESTNESTEDMAP_UINT32MAPENTRY._options = None
    _TESTNESTEDMAP_UINT32MAPENTRY._serialized_options = b'8\x01'
    _TESTNESTEDMAP_UINT64MAPENTRY._options = None
    _TESTNESTEDMAP_UINT64MAPENTRY._serialized_options = b'8\x01'
    _TESTNESTEDMAP_STRINGMAPENTRY._options = None
    _TESTNESTEDMAP_STRINGMAPENTRY._serialized_options = b'8\x01'
    _TESTNESTEDMAP_MAPMAPENTRY._options = None
    _TESTNESTEDMAP_MAPMAPENTRY._serialized_options = b'8\x01'
    _TESTSTRINGMAP_STRINGMAPENTRY._options = None
    _TESTSTRINGMAP_STRINGMAPENTRY._serialized_options = b'8\x01'
    _TESTBOOLVALUE_BOOLMAPENTRY._options = None
    _TESTBOOLVALUE_BOOLMAPENTRY._serialized_options = b'8\x01'
    _ENUMTYPE._serialized_start = 4849
    _ENUMTYPE._serialized_end = 4877
    _MESSAGETYPE._serialized_start = 277
    _MESSAGETYPE._serialized_end = 305
    _TESTMESSAGE._serialized_start = 308
    _TESTMESSAGE._serialized_end = 968
    _TESTONEOF._serialized_start = 971
    _TESTONEOF._serialized_end = 1239
    _TESTMAP._serialized_start = 1242
    _TESTMAP._serialized_end = 1851
    _TESTMAP_BOOLMAPENTRY._serialized_start = 1557
    _TESTMAP_BOOLMAPENTRY._serialized_end = 1603
    _TESTMAP_INT32MAPENTRY._serialized_start = 1605
    _TESTMAP_INT32MAPENTRY._serialized_end = 1652
    _TESTMAP_INT64MAPENTRY._serialized_start = 1654
    _TESTMAP_INT64MAPENTRY._serialized_end = 1701
    _TESTMAP_UINT32MAPENTRY._serialized_start = 1703
    _TESTMAP_UINT32MAPENTRY._serialized_end = 1751
    _TESTMAP_UINT64MAPENTRY._serialized_start = 1753
    _TESTMAP_UINT64MAPENTRY._serialized_end = 1801
    _TESTMAP_STRINGMAPENTRY._serialized_start = 1803
    _TESTMAP_STRINGMAPENTRY._serialized_end = 1851
    _TESTNESTEDMAP._serialized_start = 1854
    _TESTNESTEDMAP._serialized_end = 2627
    _TESTNESTEDMAP_BOOLMAPENTRY._serialized_start = 1557
    _TESTNESTEDMAP_BOOLMAPENTRY._serialized_end = 1603
    _TESTNESTEDMAP_INT32MAPENTRY._serialized_start = 1605
    _TESTNESTEDMAP_INT32MAPENTRY._serialized_end = 1652
    _TESTNESTEDMAP_INT64MAPENTRY._serialized_start = 1654
    _TESTNESTEDMAP_INT64MAPENTRY._serialized_end = 1701
    _TESTNESTEDMAP_UINT32MAPENTRY._serialized_start = 1703
    _TESTNESTEDMAP_UINT32MAPENTRY._serialized_end = 1751
    _TESTNESTEDMAP_UINT64MAPENTRY._serialized_start = 1753
    _TESTNESTEDMAP_UINT64MAPENTRY._serialized_end = 1801
    _TESTNESTEDMAP_STRINGMAPENTRY._serialized_start = 1803
    _TESTNESTEDMAP_STRINGMAPENTRY._serialized_end = 1851
    _TESTNESTEDMAP_MAPMAPENTRY._serialized_start = 2559
    _TESTNESTEDMAP_MAPMAPENTRY._serialized_end = 2627
    _TESTSTRINGMAP._serialized_start = 2629
    _TESTSTRINGMAP._serialized_end = 2752
    _TESTSTRINGMAP_STRINGMAPENTRY._serialized_start = 2704
    _TESTSTRINGMAP_STRINGMAPENTRY._serialized_end = 2752
    _TESTWRAPPER._serialized_start = 2755
    _TESTWRAPPER._serialized_end = 3761
    _TESTTIMESTAMP._serialized_start = 3763
    _TESTTIMESTAMP._serialized_end = 3873
    _TESTDURATION._serialized_start = 3875
    _TESTDURATION._serialized_end = 3982
    _TESTFIELDMASK._serialized_start = 3984
    _TESTFIELDMASK._serialized_end = 4042
    _TESTSTRUCT._serialized_start = 4044
    _TESTSTRUCT._serialized_end = 4145
    _TESTANY._serialized_start = 4147
    _TESTANY._serialized_end = 4239
    _TESTVALUE._serialized_start = 4241
    _TESTVALUE._serialized_end = 4339
    _TESTLISTVALUE._serialized_start = 4341
    _TESTLISTVALUE._serialized_end = 4451
    _TESTBOOLVALUE._serialized_start = 4454
    _TESTBOOLVALUE._serialized_end = 4591
    _TESTBOOLVALUE_BOOLMAPENTRY._serialized_start = 1557
    _TESTBOOLVALUE_BOOLMAPENTRY._serialized_end = 1603
    _TESTCUSTOMJSONNAME._serialized_start = 4593
    _TESTCUSTOMJSONNAME._serialized_end = 4636
    _TESTEXTENSIONS._serialized_start = 4638
    _TESTEXTENSIONS._serialized_end = 4712
    _TESTENUMVALUE._serialized_start = 4715
    _TESTENUMVALUE._serialized_end = 4847

