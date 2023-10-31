"""Generated protocol buffer code."""

from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
_sym_db = _symbol_database.Default()
DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n&google/protobuf/util/json_format.proto\x12\x11protobuf_unittest"\x89\x01\n\x13TestFlagsAndStrings\x12\t\n\x01A\x18\x01 \x02(\x05\x12K\n\rrepeatedgroup\x18\x02 \x03(\n24.protobuf_unittest.TestFlagsAndStrings.RepeatedGroup\x1a\x1a\n\rRepeatedGroup\x12\t\n\x01f\x18\x03 \x02(\t"!\n\x14TestBase64ByteArrays\x12\t\n\x01a\x18\x01 \x02(\x0c"G\n\x12TestJavaScriptJSON\x12\t\n\x01a\x18\x01 \x01(\x05\x12\r\n\x05final\x18\x02 \x01(\x02\x12\n\n\x02in\x18\x03 \x01(\t\x12\x0b\n\x03Var\x18\x04 \x01(\t"Q\n\x18TestJavaScriptOrderJSON1\x12\t\n\x01d\x18\x01 \x01(\x05\x12\t\n\x01c\x18\x02 \x01(\x05\x12\t\n\x01x\x18\x03 \x01(\x08\x12\t\n\x01b\x18\x04 \x01(\x05\x12\t\n\x01a\x18\x05 \x01(\x05"\x89\x01\n\x18TestJavaScriptOrderJSON2\x12\t\n\x01d\x18\x01 \x01(\x05\x12\t\n\x01c\x18\x02 \x01(\x05\x12\t\n\x01x\x18\x03 \x01(\x08\x12\t\n\x01b\x18\x04 \x01(\x05\x12\t\n\x01a\x18\x05 \x01(\x05\x126\n\x01z\x18\x06 \x03(\x0b2+.protobuf_unittest.TestJavaScriptOrderJSON1"$\n\x0cTestLargeInt\x12\t\n\x01a\x18\x01 \x02(\x03\x12\t\n\x01b\x18\x02 \x02(\x04"\xa0\x01\n\x0bTestNumbers\x120\n\x01a\x18\x01 \x01(\x0e2%.protobuf_unittest.TestNumbers.MyType\x12\t\n\x01b\x18\x02 \x01(\x05\x12\t\n\x01c\x18\x03 \x01(\x02\x12\t\n\x01d\x18\x04 \x01(\x08\x12\t\n\x01e\x18\x05 \x01(\x01\x12\t\n\x01f\x18\x06 \x01(\r"(\n\x06MyType\x12\x06\n\x02OK\x10\x00\x12\x0b\n\x07WARNING\x10\x01\x12\t\n\x05ERROR\x10\x02"T\n\rTestCamelCase\x12\x14\n\x0cnormal_field\x18\x01 \x01(\t\x12\x15\n\rCAPITAL_FIELD\x18\x02 \x01(\x05\x12\x16\n\x0eCamelCaseField\x18\x03 \x01(\x05"|\n\x0bTestBoolMap\x12=\n\x08bool_map\x18\x01 \x03(\x0b2+.protobuf_unittest.TestBoolMap.BoolMapEntry\x1a.\n\x0cBoolMapEntry\x12\x0b\n\x03key\x18\x01 \x01(\x08\x12\r\n\x05value\x18\x02 \x01(\x05:\x028\x01"O\n\rTestRecursion\x12\r\n\x05value\x18\x01 \x01(\x05\x12/\n\x05child\x18\x02 \x01(\x0b2 .protobuf_unittest.TestRecursion"\x86\x01\n\rTestStringMap\x12C\n\nstring_map\x18\x01 \x03(\x0b2/.protobuf_unittest.TestStringMap.StringMapEntry\x1a0\n\x0eStringMapEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x028\x01"\xc4\x01\n\x14TestStringSerializer\x12\x15\n\rscalar_string\x18\x01 \x01(\t\x12\x17\n\x0frepeated_string\x18\x02 \x03(\t\x12J\n\nstring_map\x18\x03 \x03(\x0b26.protobuf_unittest.TestStringSerializer.StringMapEntry\x1a0\n\x0eStringMapEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x028\x01"$\n\x18TestMessageWithExtension*\x08\x08d\x10\x80\x80\x80\x80\x02"z\n\rTestExtension\x12\r\n\x05value\x18\x01 \x01(\t2Z\n\x03ext\x12+.protobuf_unittest.TestMessageWithExtension\x18d \x01(\x0b2 .protobuf_unittest.TestExtension"Q\n\x14TestDefaultEnumValue\x129\n\nenum_value\x18\x01 \x01(\x0e2\x1c.protobuf_unittest.EnumValue:\x07DEFAULT*2\n\tEnumValue\x12\x0c\n\x08PROTOCOL\x10\x00\x12\n\n\x06BUFFER\x10\x01\x12\x0b\n\x07DEFAULT\x10\x02')
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'google.protobuf.util.json_format_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:
    TestMessageWithExtension.RegisterExtension(_TESTEXTENSION.extensions_by_name['ext'])
    DESCRIPTOR._options = None
    _TESTBOOLMAP_BOOLMAPENTRY._options = None
    _TESTBOOLMAP_BOOLMAPENTRY._serialized_options = b'8\x01'
    _TESTSTRINGMAP_STRINGMAPENTRY._options = None
    _TESTSTRINGMAP_STRINGMAPENTRY._serialized_options = b'8\x01'
    _TESTSTRINGSERIALIZER_STRINGMAPENTRY._options = None
    _TESTSTRINGSERIALIZER_STRINGMAPENTRY._serialized_options = b'8\x01'
    _ENUMVALUE._serialized_start = 1607
    _ENUMVALUE._serialized_end = 1657
    _TESTFLAGSANDSTRINGS._serialized_start = 62
    _TESTFLAGSANDSTRINGS._serialized_end = 199
    _TESTFLAGSANDSTRINGS_REPEATEDGROUP._serialized_start = 173
    _TESTFLAGSANDSTRINGS_REPEATEDGROUP._serialized_end = 199
    _TESTBASE64BYTEARRAYS._serialized_start = 201
    _TESTBASE64BYTEARRAYS._serialized_end = 234
    _TESTJAVASCRIPTJSON._serialized_start = 236
    _TESTJAVASCRIPTJSON._serialized_end = 307
    _TESTJAVASCRIPTORDERJSON1._serialized_start = 309
    _TESTJAVASCRIPTORDERJSON1._serialized_end = 390
    _TESTJAVASCRIPTORDERJSON2._serialized_start = 393
    _TESTJAVASCRIPTORDERJSON2._serialized_end = 530
    _TESTLARGEINT._serialized_start = 532
    _TESTLARGEINT._serialized_end = 568
    _TESTNUMBERS._serialized_start = 571
    _TESTNUMBERS._serialized_end = 731
    _TESTNUMBERS_MYTYPE._serialized_start = 691
    _TESTNUMBERS_MYTYPE._serialized_end = 731
    _TESTCAMELCASE._serialized_start = 733
    _TESTCAMELCASE._serialized_end = 817
    _TESTBOOLMAP._serialized_start = 819
    _TESTBOOLMAP._serialized_end = 943
    _TESTBOOLMAP_BOOLMAPENTRY._serialized_start = 897
    _TESTBOOLMAP_BOOLMAPENTRY._serialized_end = 943
    _TESTRECURSION._serialized_start = 945
    _TESTRECURSION._serialized_end = 1024
    _TESTSTRINGMAP._serialized_start = 1027
    _TESTSTRINGMAP._serialized_end = 1161
    _TESTSTRINGMAP_STRINGMAPENTRY._serialized_start = 1113
    _TESTSTRINGMAP_STRINGMAPENTRY._serialized_end = 1161
    _TESTSTRINGSERIALIZER._serialized_start = 1164
    _TESTSTRINGSERIALIZER._serialized_end = 1360
    _TESTSTRINGSERIALIZER_STRINGMAPENTRY._serialized_start = 1113
    _TESTSTRINGSERIALIZER_STRINGMAPENTRY._serialized_end = 1161
    _TESTMESSAGEWITHEXTENSION._serialized_start = 1362
    _TESTMESSAGEWITHEXTENSION._serialized_end = 1398
    _TESTEXTENSION._serialized_start = 1400
    _TESTEXTENSION._serialized_end = 1522
    _TESTDEFAULTENUMVALUE._serialized_start = 1524
    _TESTDEFAULTENUMVALUE._serialized_end = 1605

