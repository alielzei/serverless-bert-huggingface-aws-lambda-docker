"""Code for decoding protocol buffer primitives.

This code is very similar to encoder.py -- read the docs for that module first.

A "decoder" is a function with the signature:
  Decode(buffer, pos, end, message, field_dict)
The arguments are:
  buffer:     The string containing the encoded message.
  pos:        The current position in the string.
  end:        The position in the string where the current message ends.  May be
              less than len(buffer) if we're reading a sub-message.
  message:    The message object into which we're parsing.
  field_dict: message._fields (avoids a hashtable lookup).
The decoder reads the field and stores it into field_dict, returning the new
buffer position.  A decoder for a repeated field may proactively decode all of
the elements of that field, if they appear consecutively.

Note that decoders may throw any of the following:
  IndexError:  Indicates a truncated message.
  struct.error:  Unpacking of a fixed-width field failed.
  message.DecodeError:  Other errors.

Decoders are expected to raise an exception if they are called with pos > end.
This allows callers to be lax about bounds checking:  it's fineto read past
"end" as long as you are sure that someone else will notice and throw an
exception later on.

Something up the call stack is expected to catch IndexError and struct.error
and convert them to message.DecodeError.

Decoders are constructed using decoder constructors with the signature:
  MakeDecoder(field_number, is_repeated, is_packed, key, new_default)
The arguments are:
  field_number:  The field number of the field we want to decode.
  is_repeated:   Is the field a repeated field? (bool)
  is_packed:     Is the field a packed field? (bool)
  key:           The key to use when looking up the field within field_dict.
                 (This is actually the FieldDescriptor but nothing in this
                 file should depend on that.)
  new_default:   A function which takes a message object as a parameter and
                 returns a new instance of the default value for this field.
                 (This is called for repeated fields and sub-messages, when an
                 instance does not already exist.)

As with encoders, we define a decoder constructor for every type of field.
Then, for every field of every message class we construct an actual decoder.
That decoder goes into a dict indexed by tag, so when we decode a message
we repeatedly read a tag, look up the corresponding decoder, and invoke it.
"""

__author__ = 'kenton@google.com (Kenton Varda)'
import math
import struct
from google.protobuf.internal import containers
from google.protobuf.internal import encoder
from google.protobuf.internal import wire_format
from google.protobuf import message
_DecodeError = message.DecodeError

def _VarintDecoder(mask, result_type):
    """Return an encoder for a basic varint value (does not include tag).

  Decoded values will be bitwise-anded with the given mask before being
  returned, e.g. to limit them to 32 bits.  The returned decoder does not
  take the usual "end" parameter -- the caller is expected to do bounds checking
  after the fact (often the caller can defer such checking until later).  The
  decoder returns a (value, new_pos) pair.
  """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.decoder._VarintDecoder', '_VarintDecoder(mask, result_type)', {'_DecodeError': _DecodeError, 'mask': mask, 'result_type': result_type}, 2)

def _SignedVarintDecoder(bits, result_type):
    """Like _VarintDecoder() but decodes signed values."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.decoder._SignedVarintDecoder', '_SignedVarintDecoder(bits, result_type)', {'_DecodeError': _DecodeError, 'bits': bits, 'result_type': result_type}, 2)
_DecodeVarint = _VarintDecoder((1 << 64) - 1, int)
_DecodeSignedVarint = _SignedVarintDecoder(64, int)
_DecodeVarint32 = _VarintDecoder((1 << 32) - 1, int)
_DecodeSignedVarint32 = _SignedVarintDecoder(32, int)

def ReadTag(buffer, pos):
    """Read a tag from the memoryview, and return a (tag_bytes, new_pos) tuple.

  We return the raw bytes of the tag rather than decoding them.  The raw
  bytes can then be used to look up the proper decoder.  This effectively allows
  us to trade some work that would be done in pure-python (decoding a varint)
  for work that is done in C (searching for a byte string in a hash table).
  In a low-level language it would be much cheaper to decode the varint and
  use that, but not in Python.

  Args:
    buffer: memoryview object of the encoded bytes
    pos: int of the current position to start from

  Returns:
    Tuple[bytes, int] of the tag data and new position.
  """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.decoder.ReadTag', 'ReadTag(buffer, pos)', {'buffer': buffer, 'pos': pos}, 2)

def _SimpleDecoder(wire_type, decode_value):
    """Return a constructor for a decoder for fields of a particular type.

  Args:
      wire_type:  The field's wire type.
      decode_value:  A function which decodes an individual value, e.g.
        _DecodeVarint()
  """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.decoder._SimpleDecoder', '_SimpleDecoder(wire_type, decode_value)', {'_DecodeVarint': _DecodeVarint, '_DecodeError': _DecodeError, 'encoder': encoder, 'wire_type': wire_type, 'decode_value': decode_value}, 1)

def _ModifiedDecoder(wire_type, decode_value, modify_value):
    """Like SimpleDecoder but additionally invokes modify_value on every value
  before storing it.  Usually modify_value is ZigZagDecode.
  """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.decoder._ModifiedDecoder', '_ModifiedDecoder(wire_type, decode_value, modify_value)', {'_SimpleDecoder': _SimpleDecoder, 'wire_type': wire_type, 'decode_value': decode_value, 'modify_value': modify_value}, 2)

def _StructPackDecoder(wire_type, format):
    """Return a constructor for a decoder for a fixed-width field.

  Args:
      wire_type:  The field's wire type.
      format:  The format string to pass to struct.unpack().
  """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.decoder._StructPackDecoder', '_StructPackDecoder(wire_type, format)', {'struct': struct, '_SimpleDecoder': _SimpleDecoder, 'wire_type': wire_type, 'format': format}, 2)

def _FloatDecoder():
    """Returns a decoder for a float field.

  This code works around a bug in struct.unpack for non-finite 32-bit
  floating-point values.
  """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.decoder._FloatDecoder', '_FloatDecoder()', {'struct': struct, 'math': math, '_SimpleDecoder': _SimpleDecoder, 'wire_format': wire_format}, 2)

def _DoubleDecoder():
    """Returns a decoder for a double field.

  This code works around a bug in struct.unpack for not-a-number.
  """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.decoder._DoubleDecoder', '_DoubleDecoder()', {'struct': struct, 'math': math, '_SimpleDecoder': _SimpleDecoder, 'wire_format': wire_format}, 2)

def EnumDecoder(field_number, is_repeated, is_packed, key, new_default, clear_if_default=False):
    """Returns a decoder for enum field."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.decoder.EnumDecoder', 'EnumDecoder(field_number, is_repeated, is_packed, key, new_default, clear_if_default=False)', {'_DecodeVarint': _DecodeVarint, '_DecodeError': _DecodeError, '_DecodeSignedVarint32': _DecodeSignedVarint32, 'encoder': encoder, 'wire_format': wire_format, 'containers': containers, 'field_number': field_number, 'is_repeated': is_repeated, 'is_packed': is_packed, 'key': key, 'new_default': new_default, 'clear_if_default': clear_if_default}, 1)
Int32Decoder = _SimpleDecoder(wire_format.WIRETYPE_VARINT, _DecodeSignedVarint32)
Int64Decoder = _SimpleDecoder(wire_format.WIRETYPE_VARINT, _DecodeSignedVarint)
UInt32Decoder = _SimpleDecoder(wire_format.WIRETYPE_VARINT, _DecodeVarint32)
UInt64Decoder = _SimpleDecoder(wire_format.WIRETYPE_VARINT, _DecodeVarint)
SInt32Decoder = _ModifiedDecoder(wire_format.WIRETYPE_VARINT, _DecodeVarint32, wire_format.ZigZagDecode)
SInt64Decoder = _ModifiedDecoder(wire_format.WIRETYPE_VARINT, _DecodeVarint, wire_format.ZigZagDecode)
Fixed32Decoder = _StructPackDecoder(wire_format.WIRETYPE_FIXED32, '<I')
Fixed64Decoder = _StructPackDecoder(wire_format.WIRETYPE_FIXED64, '<Q')
SFixed32Decoder = _StructPackDecoder(wire_format.WIRETYPE_FIXED32, '<i')
SFixed64Decoder = _StructPackDecoder(wire_format.WIRETYPE_FIXED64, '<q')
FloatDecoder = _FloatDecoder()
DoubleDecoder = _DoubleDecoder()
BoolDecoder = _ModifiedDecoder(wire_format.WIRETYPE_VARINT, _DecodeVarint, bool)

def StringDecoder(field_number, is_repeated, is_packed, key, new_default, clear_if_default=False):
    """Returns a decoder for a string field."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.decoder.StringDecoder', 'StringDecoder(field_number, is_repeated, is_packed, key, new_default, clear_if_default=False)', {'_DecodeVarint': _DecodeVarint, 'encoder': encoder, 'wire_format': wire_format, '_DecodeError': _DecodeError, 'field_number': field_number, 'is_repeated': is_repeated, 'is_packed': is_packed, 'key': key, 'new_default': new_default, 'clear_if_default': clear_if_default}, 1)

def BytesDecoder(field_number, is_repeated, is_packed, key, new_default, clear_if_default=False):
    """Returns a decoder for a bytes field."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.decoder.BytesDecoder', 'BytesDecoder(field_number, is_repeated, is_packed, key, new_default, clear_if_default=False)', {'_DecodeVarint': _DecodeVarint, 'encoder': encoder, 'wire_format': wire_format, '_DecodeError': _DecodeError, 'field_number': field_number, 'is_repeated': is_repeated, 'is_packed': is_packed, 'key': key, 'new_default': new_default, 'clear_if_default': clear_if_default}, 1)

def GroupDecoder(field_number, is_repeated, is_packed, key, new_default):
    """Returns a decoder for a group field."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.decoder.GroupDecoder', 'GroupDecoder(field_number, is_repeated, is_packed, key, new_default)', {'encoder': encoder, 'wire_format': wire_format, '_DecodeError': _DecodeError, 'field_number': field_number, 'is_repeated': is_repeated, 'is_packed': is_packed, 'key': key, 'new_default': new_default}, 1)

def MessageDecoder(field_number, is_repeated, is_packed, key, new_default):
    """Returns a decoder for a message field."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.decoder.MessageDecoder', 'MessageDecoder(field_number, is_repeated, is_packed, key, new_default)', {'_DecodeVarint': _DecodeVarint, 'encoder': encoder, 'wire_format': wire_format, '_DecodeError': _DecodeError, 'field_number': field_number, 'is_repeated': is_repeated, 'is_packed': is_packed, 'key': key, 'new_default': new_default}, 1)
MESSAGE_SET_ITEM_TAG = encoder.TagBytes(1, wire_format.WIRETYPE_START_GROUP)

def MessageSetItemDecoder(descriptor):
    """Returns a decoder for a MessageSet item.

  The parameter is the message Descriptor.

  The message set message looks like this:
    message MessageSet {
      repeated group Item = 1 {
        required int32 type_id = 2;
        required string message = 3;
      }
    }
  """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.decoder.MessageSetItemDecoder', 'MessageSetItemDecoder(descriptor)', {'encoder': encoder, 'wire_format': wire_format, 'ReadTag': ReadTag, '_DecodeVarint': _DecodeVarint, 'SkipField': SkipField, '_DecodeError': _DecodeError, 'MESSAGE_SET_ITEM_TAG': MESSAGE_SET_ITEM_TAG, 'containers': containers, 'descriptor': descriptor}, 1)

def MapDecoder(field_descriptor, new_default, is_message_map):
    """Returns a decoder for a map field."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.decoder.MapDecoder', 'MapDecoder(field_descriptor, new_default, is_message_map)', {'encoder': encoder, 'wire_format': wire_format, '_DecodeVarint': _DecodeVarint, '_DecodeError': _DecodeError, 'field_descriptor': field_descriptor, 'new_default': new_default, 'is_message_map': is_message_map}, 1)

def _SkipVarint(buffer, pos, end):
    """Skip a varint value.  Returns the new position."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.decoder._SkipVarint', '_SkipVarint(buffer, pos, end)', {'_DecodeError': _DecodeError, 'buffer': buffer, 'pos': pos, 'end': end}, 1)

def _SkipFixed64(buffer, pos, end):
    """Skip a fixed64 value.  Returns the new position."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.decoder._SkipFixed64', '_SkipFixed64(buffer, pos, end)', {'_DecodeError': _DecodeError, 'buffer': buffer, 'pos': pos, 'end': end}, 1)

def _DecodeFixed64(buffer, pos):
    """Decode a fixed64."""
    new_pos = pos + 8
    return (struct.unpack('<Q', buffer[pos:new_pos])[0], new_pos)

def _SkipLengthDelimited(buffer, pos, end):
    """Skip a length-delimited value.  Returns the new position."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.decoder._SkipLengthDelimited', '_SkipLengthDelimited(buffer, pos, end)', {'_DecodeVarint': _DecodeVarint, '_DecodeError': _DecodeError, 'buffer': buffer, 'pos': pos, 'end': end}, 1)

def _SkipGroup(buffer, pos, end):
    """Skip sub-group.  Returns the new position."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.decoder._SkipGroup', '_SkipGroup(buffer, pos, end)', {'ReadTag': ReadTag, 'SkipField': SkipField, 'buffer': buffer, 'pos': pos, 'end': end}, 1)

def _DecodeUnknownFieldSet(buffer, pos, end_pos=None):
    """Decode UnknownFieldSet.  Returns the UnknownFieldSet and new position."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.decoder._DecodeUnknownFieldSet', '_DecodeUnknownFieldSet(buffer, pos, end_pos=None)', {'containers': containers, 'ReadTag': ReadTag, '_DecodeVarint': _DecodeVarint, 'wire_format': wire_format, '_DecodeUnknownField': _DecodeUnknownField, 'buffer': buffer, 'pos': pos, 'end_pos': end_pos}, 2)

def _DecodeUnknownField(buffer, pos, wire_type):
    """Decode a unknown field.  Returns the UnknownField and new position."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.decoder._DecodeUnknownField', '_DecodeUnknownField(buffer, pos, wire_type)', {'wire_format': wire_format, '_DecodeVarint': _DecodeVarint, '_DecodeFixed64': _DecodeFixed64, '_DecodeFixed32': _DecodeFixed32, '_DecodeUnknownFieldSet': _DecodeUnknownFieldSet, '_DecodeError': _DecodeError, 'buffer': buffer, 'pos': pos, 'wire_type': wire_type}, 2)

def _EndGroup(buffer, pos, end):
    """Skipping an END_GROUP tag returns -1 to tell the parent loop to break."""
    return -1

def _SkipFixed32(buffer, pos, end):
    """Skip a fixed32 value.  Returns the new position."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.decoder._SkipFixed32', '_SkipFixed32(buffer, pos, end)', {'_DecodeError': _DecodeError, 'buffer': buffer, 'pos': pos, 'end': end}, 1)

def _DecodeFixed32(buffer, pos):
    """Decode a fixed32."""
    new_pos = pos + 4
    return (struct.unpack('<I', buffer[pos:new_pos])[0], new_pos)

def _RaiseInvalidWireType(buffer, pos, end):
    """Skip function for unknown wire types.  Raises an exception."""
    raise _DecodeError('Tag had invalid wire type.')

def _FieldSkipper():
    """Constructs the SkipField function."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.decoder._FieldSkipper', '_FieldSkipper()', {'_SkipVarint': _SkipVarint, '_SkipFixed64': _SkipFixed64, '_SkipLengthDelimited': _SkipLengthDelimited, '_SkipGroup': _SkipGroup, '_EndGroup': _EndGroup, '_SkipFixed32': _SkipFixed32, '_RaiseInvalidWireType': _RaiseInvalidWireType, 'wire_format': wire_format}, 1)
SkipField = _FieldSkipper()

