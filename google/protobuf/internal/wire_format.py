"""Constants and static functions to support protocol buffer wire format."""

__author__ = 'robinson@google.com (Will Robinson)'
import struct
from google.protobuf import descriptor
from google.protobuf import message
TAG_TYPE_BITS = 3
TAG_TYPE_MASK = (1 << TAG_TYPE_BITS) - 1
WIRETYPE_VARINT = 0
WIRETYPE_FIXED64 = 1
WIRETYPE_LENGTH_DELIMITED = 2
WIRETYPE_START_GROUP = 3
WIRETYPE_END_GROUP = 4
WIRETYPE_FIXED32 = 5
_WIRETYPE_MAX = 5
INT32_MAX = int((1 << 31) - 1)
INT32_MIN = int(-(1 << 31))
UINT32_MAX = (1 << 32) - 1
INT64_MAX = (1 << 63) - 1
INT64_MIN = -(1 << 63)
UINT64_MAX = (1 << 64) - 1
FORMAT_UINT32_LITTLE_ENDIAN = '<I'
FORMAT_UINT64_LITTLE_ENDIAN = '<Q'
FORMAT_FLOAT_LITTLE_ENDIAN = '<f'
FORMAT_DOUBLE_LITTLE_ENDIAN = '<d'
if struct.calcsize(FORMAT_UINT32_LITTLE_ENDIAN) != 4:
    raise AssertionError('Format "I" is not a 32-bit number.')
if struct.calcsize(FORMAT_UINT64_LITTLE_ENDIAN) != 8:
    raise AssertionError('Format "Q" is not a 64-bit number.')

def PackTag(field_number, wire_type):
    """Returns an unsigned 32-bit integer that encodes the field number and
  wire type information in standard protocol message wire format.

  Args:
    field_number: Expected to be an integer in the range [1, 1 << 29)
    wire_type: One of the WIRETYPE_* constants.
  """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.wire_format.PackTag', 'PackTag(field_number, wire_type)', {'_WIRETYPE_MAX': _WIRETYPE_MAX, 'message': message, 'TAG_TYPE_BITS': TAG_TYPE_BITS, 'field_number': field_number, 'wire_type': wire_type}, 1)

def UnpackTag(tag):
    """The inverse of PackTag().  Given an unsigned 32-bit number,
  returns a (field_number, wire_type) tuple.
  """
    return (tag >> TAG_TYPE_BITS, tag & TAG_TYPE_MASK)

def ZigZagEncode(value):
    """ZigZag Transform:  Encodes signed integers so that they can be
  effectively used with varint encoding.  See wire_format.h for
  more details.
  """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.wire_format.ZigZagEncode', 'ZigZagEncode(value)', {'value': value}, 1)

def ZigZagDecode(value):
    """Inverse of ZigZagEncode()."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.wire_format.ZigZagDecode', 'ZigZagDecode(value)', {'value': value}, 1)

def Int32ByteSize(field_number, int32):
    return Int64ByteSize(field_number, int32)

def Int32ByteSizeNoTag(int32):
    return _VarUInt64ByteSizeNoTag(18446744073709551615 & int32)

def Int64ByteSize(field_number, int64):
    return UInt64ByteSize(field_number, 18446744073709551615 & int64)

def UInt32ByteSize(field_number, uint32):
    return UInt64ByteSize(field_number, uint32)

def UInt64ByteSize(field_number, uint64):
    return TagByteSize(field_number) + _VarUInt64ByteSizeNoTag(uint64)

def SInt32ByteSize(field_number, int32):
    return UInt32ByteSize(field_number, ZigZagEncode(int32))

def SInt64ByteSize(field_number, int64):
    return UInt64ByteSize(field_number, ZigZagEncode(int64))

def Fixed32ByteSize(field_number, fixed32):
    return TagByteSize(field_number) + 4

def Fixed64ByteSize(field_number, fixed64):
    return TagByteSize(field_number) + 8

def SFixed32ByteSize(field_number, sfixed32):
    return TagByteSize(field_number) + 4

def SFixed64ByteSize(field_number, sfixed64):
    return TagByteSize(field_number) + 8

def FloatByteSize(field_number, flt):
    return TagByteSize(field_number) + 4

def DoubleByteSize(field_number, double):
    return TagByteSize(field_number) + 8

def BoolByteSize(field_number, b):
    return TagByteSize(field_number) + 1

def EnumByteSize(field_number, enum):
    return UInt32ByteSize(field_number, enum)

def StringByteSize(field_number, string):
    return BytesByteSize(field_number, string.encode('utf-8'))

def BytesByteSize(field_number, b):
    return TagByteSize(field_number) + _VarUInt64ByteSizeNoTag(len(b)) + len(b)

def GroupByteSize(field_number, message):
    return 2 * TagByteSize(field_number) + message.ByteSize()

def MessageByteSize(field_number, message):
    return TagByteSize(field_number) + _VarUInt64ByteSizeNoTag(message.ByteSize()) + message.ByteSize()

def MessageSetItemByteSize(field_number, msg):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.wire_format.MessageSetItemByteSize', 'MessageSetItemByteSize(field_number, msg)', {'TagByteSize': TagByteSize, '_VarUInt64ByteSizeNoTag': _VarUInt64ByteSizeNoTag, 'field_number': field_number, 'msg': msg}, 1)

def TagByteSize(field_number):
    """Returns the bytes required to serialize a tag with this field number."""
    return _VarUInt64ByteSizeNoTag(PackTag(field_number, 0))

def _VarUInt64ByteSizeNoTag(uint64):
    """Returns the number of bytes required to serialize a single varint
  using boundary value comparisons. (unrolled loop optimization -WPierce)
  uint64 must be unsigned.
  """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.wire_format._VarUInt64ByteSizeNoTag', '_VarUInt64ByteSizeNoTag(uint64)', {'UINT64_MAX': UINT64_MAX, 'message': message, 'uint64': uint64}, 1)
NON_PACKABLE_TYPES = (descriptor.FieldDescriptor.TYPE_STRING, descriptor.FieldDescriptor.TYPE_GROUP, descriptor.FieldDescriptor.TYPE_MESSAGE, descriptor.FieldDescriptor.TYPE_BYTES)

def IsTypePackable(field_type):
    """Return true iff packable = true is valid for fields of this type.

  Args:
    field_type: a FieldDescriptor::Type value.

  Returns:
    True iff fields of this type are packable.
  """
    return field_type not in NON_PACKABLE_TYPES

