"""Code for encoding protocol message primitives.

Contains the logic for encoding every logical protocol field type
into one of the 5 physical wire types.

This code is designed to push the Python interpreter's performance to the
limits.

The basic idea is that at startup time, for every field (i.e. every
FieldDescriptor) we construct two functions:  a "sizer" and an "encoder".  The
sizer takes a value of this field's type and computes its byte size.  The
encoder takes a writer function and a value.  It encodes the value into byte
strings and invokes the writer function to write those strings.  Typically the
writer function is the write() method of a BytesIO.

We try to do as much work as possible when constructing the writer and the
sizer rather than when calling them.  In particular:
* We copy any needed global functions to local variables, so that we do not need
  to do costly global table lookups at runtime.
* Similarly, we try to do any attribute lookups at startup time if possible.
* Every field's tag is encoded to bytes at startup, since it can't change at
  runtime.
* Whatever component of the field size we can compute at startup, we do.
* We *avoid* sharing code if doing so would make the code slower and not sharing
  does not burden us too much.  For example, encoders for repeated fields do
  not just call the encoders for singular fields in a loop because this would
  add an extra function call overhead for every loop iteration; instead, we
  manually inline the single-value encoder into the loop.
* If a Python function lacks a return statement, Python actually generates
  instructions to pop the result of the last statement off the stack, push
  None onto the stack, and then return that.  If we really don't care what
  value is returned, then we can save two instructions by returning the
  result of the last statement.  It looks funny but it helps.
* We assume that type and bounds checking has happened at a higher level.
"""

__author__ = 'kenton@google.com (Kenton Varda)'
import struct
from google.protobuf.internal import wire_format
_POS_INF = 1e10000
_NEG_INF = -_POS_INF

def _VarintSize(value):
    """Compute the size of a varint value."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.encoder._VarintSize', '_VarintSize(value)', {'value': value}, 1)

def _SignedVarintSize(value):
    """Compute the size of a signed varint value."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.encoder._SignedVarintSize', '_SignedVarintSize(value)', {'value': value}, 1)

def _TagSize(field_number):
    """Returns the number of bytes required to serialize a tag with this field
  number."""
    return _VarintSize(wire_format.PackTag(field_number, 0))

def _SimpleSizer(compute_value_size):
    """A sizer which uses the function compute_value_size to compute the size of
  each value.  Typically compute_value_size is _VarintSize."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.encoder._SimpleSizer', '_SimpleSizer(compute_value_size)', {'_TagSize': _TagSize, '_VarintSize': _VarintSize, 'compute_value_size': compute_value_size}, 1)

def _ModifiedSizer(compute_value_size, modify_value):
    """Like SimpleSizer, but modify_value is invoked on each value before it is
  passed to compute_value_size.  modify_value is typically ZigZagEncode."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.encoder._ModifiedSizer', '_ModifiedSizer(compute_value_size, modify_value)', {'_TagSize': _TagSize, '_VarintSize': _VarintSize, 'compute_value_size': compute_value_size, 'modify_value': modify_value}, 1)

def _FixedSizer(value_size):
    """Like _SimpleSizer except for a fixed-size field.  The input is the size
  of one value."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.encoder._FixedSizer', '_FixedSizer(value_size)', {'_TagSize': _TagSize, '_VarintSize': _VarintSize, 'value_size': value_size}, 1)
Int32Sizer = Int64Sizer = EnumSizer = _SimpleSizer(_SignedVarintSize)
UInt32Sizer = UInt64Sizer = _SimpleSizer(_VarintSize)
SInt32Sizer = SInt64Sizer = _ModifiedSizer(_SignedVarintSize, wire_format.ZigZagEncode)
Fixed32Sizer = SFixed32Sizer = FloatSizer = _FixedSizer(4)
Fixed64Sizer = SFixed64Sizer = DoubleSizer = _FixedSizer(8)
BoolSizer = _FixedSizer(1)

def StringSizer(field_number, is_repeated, is_packed):
    """Returns a sizer for a string field."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.encoder.StringSizer', 'StringSizer(field_number, is_repeated, is_packed)', {'_TagSize': _TagSize, '_VarintSize': _VarintSize, 'field_number': field_number, 'is_repeated': is_repeated, 'is_packed': is_packed}, 1)

def BytesSizer(field_number, is_repeated, is_packed):
    """Returns a sizer for a bytes field."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.encoder.BytesSizer', 'BytesSizer(field_number, is_repeated, is_packed)', {'_TagSize': _TagSize, '_VarintSize': _VarintSize, 'field_number': field_number, 'is_repeated': is_repeated, 'is_packed': is_packed}, 1)

def GroupSizer(field_number, is_repeated, is_packed):
    """Returns a sizer for a group field."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.encoder.GroupSizer', 'GroupSizer(field_number, is_repeated, is_packed)', {'_TagSize': _TagSize, 'field_number': field_number, 'is_repeated': is_repeated, 'is_packed': is_packed}, 1)

def MessageSizer(field_number, is_repeated, is_packed):
    """Returns a sizer for a message field."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.encoder.MessageSizer', 'MessageSizer(field_number, is_repeated, is_packed)', {'_TagSize': _TagSize, '_VarintSize': _VarintSize, 'field_number': field_number, 'is_repeated': is_repeated, 'is_packed': is_packed}, 1)

def MessageSetItemSizer(field_number):
    """Returns a sizer for extensions of MessageSet.

  The message set message looks like this:
    message MessageSet {
      repeated group Item = 1 {
        required int32 type_id = 2;
        required string message = 3;
      }
    }
  """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.encoder.MessageSetItemSizer', 'MessageSetItemSizer(field_number)', {'_TagSize': _TagSize, '_VarintSize': _VarintSize, 'field_number': field_number}, 1)

def MapSizer(field_descriptor, is_message_map):
    """Returns a sizer for a map field."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.encoder.MapSizer', 'MapSizer(field_descriptor, is_message_map)', {'MessageSizer': MessageSizer, 'field_descriptor': field_descriptor, 'is_message_map': is_message_map}, 1)

def _VarintEncoder():
    """Return an encoder for a basic varint value (does not include tag)."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.encoder._VarintEncoder', '_VarintEncoder()', {'struct': struct}, 1)

def _SignedVarintEncoder():
    """Return an encoder for a basic signed varint value (does not include
  tag)."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.encoder._SignedVarintEncoder', '_SignedVarintEncoder()', {'struct': struct}, 1)
_EncodeVarint = _VarintEncoder()
_EncodeSignedVarint = _SignedVarintEncoder()

def _VarintBytes(value):
    """Encode the given integer as a varint and return the bytes.  This is only
  called at startup time so it doesn't need to be fast."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.encoder._VarintBytes', '_VarintBytes(value)', {'_EncodeVarint': _EncodeVarint, 'value': value}, 1)

def TagBytes(field_number, wire_type):
    """Encode the given tag and return the bytes.  Only called at startup."""
    return bytes(_VarintBytes(wire_format.PackTag(field_number, wire_type)))

def _SimpleEncoder(wire_type, encode_value, compute_value_size):
    """Return a constructor for an encoder for fields of a particular type.

  Args:
      wire_type:  The field's wire type, for encoding tags.
      encode_value:  A function which encodes an individual value, e.g.
        _EncodeVarint().
      compute_value_size:  A function which computes the size of an individual
        value, e.g. _VarintSize().
  """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.encoder._SimpleEncoder', '_SimpleEncoder(wire_type, encode_value, compute_value_size)', {'TagBytes': TagBytes, 'wire_format': wire_format, '_EncodeVarint': _EncodeVarint, 'wire_type': wire_type, 'encode_value': encode_value, 'compute_value_size': compute_value_size}, 1)

def _ModifiedEncoder(wire_type, encode_value, compute_value_size, modify_value):
    """Like SimpleEncoder but additionally invokes modify_value on every value
  before passing it to encode_value.  Usually modify_value is ZigZagEncode."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.encoder._ModifiedEncoder', '_ModifiedEncoder(wire_type, encode_value, compute_value_size, modify_value)', {'TagBytes': TagBytes, 'wire_format': wire_format, '_EncodeVarint': _EncodeVarint, 'wire_type': wire_type, 'encode_value': encode_value, 'compute_value_size': compute_value_size, 'modify_value': modify_value}, 1)

def _StructPackEncoder(wire_type, format):
    """Return a constructor for an encoder for a fixed-width field.

  Args:
      wire_type:  The field's wire type, for encoding tags.
      format:  The format string to pass to struct.pack().
  """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.encoder._StructPackEncoder', '_StructPackEncoder(wire_type, format)', {'struct': struct, 'TagBytes': TagBytes, 'wire_format': wire_format, '_EncodeVarint': _EncodeVarint, 'wire_type': wire_type, 'format': format}, 1)

def _FloatingPointEncoder(wire_type, format):
    """Return a constructor for an encoder for float fields.

  This is like StructPackEncoder, but catches errors that may be due to
  passing non-finite floating-point values to struct.pack, and makes a
  second attempt to encode those values.

  Args:
      wire_type:  The field's wire type, for encoding tags.
      format:  The format string to pass to struct.pack().
  """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.encoder._FloatingPointEncoder', '_FloatingPointEncoder(wire_type, format)', {'struct': struct, '_POS_INF': _POS_INF, '_NEG_INF': _NEG_INF, 'TagBytes': TagBytes, 'wire_format': wire_format, '_EncodeVarint': _EncodeVarint, 'wire_type': wire_type, 'format': format}, 1)
Int32Encoder = Int64Encoder = EnumEncoder = _SimpleEncoder(wire_format.WIRETYPE_VARINT, _EncodeSignedVarint, _SignedVarintSize)
UInt32Encoder = UInt64Encoder = _SimpleEncoder(wire_format.WIRETYPE_VARINT, _EncodeVarint, _VarintSize)
SInt32Encoder = SInt64Encoder = _ModifiedEncoder(wire_format.WIRETYPE_VARINT, _EncodeVarint, _VarintSize, wire_format.ZigZagEncode)
Fixed32Encoder = _StructPackEncoder(wire_format.WIRETYPE_FIXED32, '<I')
Fixed64Encoder = _StructPackEncoder(wire_format.WIRETYPE_FIXED64, '<Q')
SFixed32Encoder = _StructPackEncoder(wire_format.WIRETYPE_FIXED32, '<i')
SFixed64Encoder = _StructPackEncoder(wire_format.WIRETYPE_FIXED64, '<q')
FloatEncoder = _FloatingPointEncoder(wire_format.WIRETYPE_FIXED32, '<f')
DoubleEncoder = _FloatingPointEncoder(wire_format.WIRETYPE_FIXED64, '<d')

def BoolEncoder(field_number, is_repeated, is_packed):
    """Returns an encoder for a boolean field."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.encoder.BoolEncoder', 'BoolEncoder(field_number, is_repeated, is_packed)', {'TagBytes': TagBytes, 'wire_format': wire_format, '_EncodeVarint': _EncodeVarint, 'field_number': field_number, 'is_repeated': is_repeated, 'is_packed': is_packed}, 1)

def StringEncoder(field_number, is_repeated, is_packed):
    """Returns an encoder for a string field."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.encoder.StringEncoder', 'StringEncoder(field_number, is_repeated, is_packed)', {'TagBytes': TagBytes, 'wire_format': wire_format, '_EncodeVarint': _EncodeVarint, 'field_number': field_number, 'is_repeated': is_repeated, 'is_packed': is_packed}, 1)

def BytesEncoder(field_number, is_repeated, is_packed):
    """Returns an encoder for a bytes field."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.encoder.BytesEncoder', 'BytesEncoder(field_number, is_repeated, is_packed)', {'TagBytes': TagBytes, 'wire_format': wire_format, '_EncodeVarint': _EncodeVarint, 'field_number': field_number, 'is_repeated': is_repeated, 'is_packed': is_packed}, 1)

def GroupEncoder(field_number, is_repeated, is_packed):
    """Returns an encoder for a group field."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.encoder.GroupEncoder', 'GroupEncoder(field_number, is_repeated, is_packed)', {'TagBytes': TagBytes, 'wire_format': wire_format, 'field_number': field_number, 'is_repeated': is_repeated, 'is_packed': is_packed}, 1)

def MessageEncoder(field_number, is_repeated, is_packed):
    """Returns an encoder for a message field."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.encoder.MessageEncoder', 'MessageEncoder(field_number, is_repeated, is_packed)', {'TagBytes': TagBytes, 'wire_format': wire_format, '_EncodeVarint': _EncodeVarint, 'field_number': field_number, 'is_repeated': is_repeated, 'is_packed': is_packed}, 1)

def MessageSetItemEncoder(field_number):
    """Encoder for extensions of MessageSet.

  The message set message looks like this:
    message MessageSet {
      repeated group Item = 1 {
        required int32 type_id = 2;
        required string message = 3;
      }
    }
  """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.encoder.MessageSetItemEncoder', 'MessageSetItemEncoder(field_number)', {'TagBytes': TagBytes, 'wire_format': wire_format, '_VarintBytes': _VarintBytes, '_EncodeVarint': _EncodeVarint, 'field_number': field_number}, 1)

def MapEncoder(field_descriptor):
    """Encoder for extensions of MessageSet.

  Maps always have a wire format like this:
    message MapEntry {
      key_type key = 1;
      value_type value = 2;
    }
    repeated MapEntry map = N;
  """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.encoder.MapEncoder', 'MapEncoder(field_descriptor)', {'MessageEncoder': MessageEncoder, 'field_descriptor': field_descriptor}, 1)

