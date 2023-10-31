"""Contains a metaclass and helper functions used to create
protocol message classes from Descriptor objects at runtime.

Recall that a metaclass is the "type" of a class.
(A class is to a metaclass what an instance is to a class.)

In this case, we use the GeneratedProtocolMessageType metaclass
to inject all the useful functionality into the classes
output by the protocol compiler at compile-time.

The upshot of all this is that the real implementation
details for ALL pure-Python protocol buffers are *here in
this file*.
"""

__author__ = 'robinson@google.com (Will Robinson)'
from io import BytesIO
import struct
import sys
import weakref
from google.protobuf.internal import api_implementation
from google.protobuf.internal import containers
from google.protobuf.internal import decoder
from google.protobuf.internal import encoder
from google.protobuf.internal import enum_type_wrapper
from google.protobuf.internal import extension_dict
from google.protobuf.internal import message_listener as message_listener_mod
from google.protobuf.internal import type_checkers
from google.protobuf.internal import well_known_types
from google.protobuf.internal import wire_format
from google.protobuf import descriptor as descriptor_mod
from google.protobuf import message as message_mod
from google.protobuf import text_format
_FieldDescriptor = descriptor_mod.FieldDescriptor
_AnyFullTypeName = 'google.protobuf.Any'
_ExtensionDict = extension_dict._ExtensionDict


class GeneratedProtocolMessageType(type):
    """Metaclass for protocol message classes created at runtime from Descriptors.

  We add implementations for all methods described in the Message class.  We
  also create properties to allow getting/setting all fields in the protocol
  message.  Finally, we create slots to prevent users from accidentally
  "setting" nonexistent fields in the protocol message, which then wouldn't get
  serialized / deserialized properly.

  The protocol compiler currently uses this metaclass to create protocol
  message classes at runtime.  Clients can also manually create their own
  classes at runtime, as in this example:

  mydescriptor = Descriptor(.....)
  factory = symbol_database.Default()
  factory.pool.AddDescriptor(mydescriptor)
  MyProtoClass = factory.GetPrototype(mydescriptor)
  myproto_instance = MyProtoClass()
  myproto.foo_field = 23
  ...
  """
    _DESCRIPTOR_KEY = 'DESCRIPTOR'
    
    def __new__(cls, name, bases, dictionary):
        """Custom allocation for runtime-generated class types.

    We override __new__ because this is apparently the only place
    where we can meaningfully set __slots__ on the class we're creating(?).
    (The interplay between metaclasses and slots is not very well-documented).

    Args:
      name: Name of the class (ignored, but required by the
        metaclass protocol).
      bases: Base classes of the class we're constructing.
        (Should be message.Message).  We ignore this field, but
        it's required by the metaclass protocol
      dictionary: The class dictionary of the class we're
        constructing.  dictionary[_DESCRIPTOR_KEY] must contain
        a Descriptor object describing this protocol message
        type.

    Returns:
      Newly-allocated class.

    Raises:
      RuntimeError: Generated code only work with python cpp extension.
    """
        descriptor = dictionary[GeneratedProtocolMessageType._DESCRIPTOR_KEY]
        if isinstance(descriptor, str):
            raise RuntimeError('The generated code only work with python cpp extension, but it is using pure python runtime.')
        new_class = getattr(descriptor, '_concrete_class', None)
        if new_class:
            return new_class
        if descriptor.full_name in well_known_types.WKTBASES:
            bases += (well_known_types.WKTBASES[descriptor.full_name], )
        _AddClassAttributesForNestedExtensions(descriptor, dictionary)
        _AddSlots(descriptor, dictionary)
        superclass = super(GeneratedProtocolMessageType, cls)
        new_class = superclass.__new__(cls, name, bases, dictionary)
        return new_class
    
    def __init__(cls, name, bases, dictionary):
        """Here we perform the majority of our work on the class.
    We add enum getters, an __init__ method, implementations
    of all Message methods, and properties for all fields
    in the protocol type.

    Args:
      name: Name of the class (ignored, but required by the
        metaclass protocol).
      bases: Base classes of the class we're constructing.
        (Should be message.Message).  We ignore this field, but
        it's required by the metaclass protocol
      dictionary: The class dictionary of the class we're
        constructing.  dictionary[_DESCRIPTOR_KEY] must contain
        a Descriptor object describing this protocol message
        type.
    """
        descriptor = dictionary[GeneratedProtocolMessageType._DESCRIPTOR_KEY]
        existing_class = getattr(descriptor, '_concrete_class', None)
        if existing_class:
            assert existing_class is cls, 'Duplicate `GeneratedProtocolMessageType` created for descriptor %r' % descriptor.full_name
            return
        cls._decoders_by_tag = {}
        if (descriptor.has_options and descriptor.GetOptions().message_set_wire_format):
            cls._decoders_by_tag[decoder.MESSAGE_SET_ITEM_TAG] = (decoder.MessageSetItemDecoder(descriptor), None)
        for field in descriptor.fields:
            _AttachFieldHelpers(cls, field)
        descriptor._concrete_class = cls
        _AddEnumValues(descriptor, cls)
        _AddInitMethod(descriptor, cls)
        _AddPropertiesForFields(descriptor, cls)
        _AddPropertiesForExtensions(descriptor, cls)
        _AddStaticMethods(cls)
        _AddMessageMethods(descriptor, cls)
        _AddPrivateHelperMethods(descriptor, cls)
        superclass = super(GeneratedProtocolMessageType, cls)
        superclass.__init__(name, bases, dictionary)


def _PropertyName(proto_field_name):
    """Returns the name of the public property attribute which
  clients can use to get and (in some cases) set the value
  of a protocol message field.

  Args:
    proto_field_name: The protocol message field name, exactly
      as it appears (or would appear) in a .proto file.
  """
    return proto_field_name

def _AddSlots(message_descriptor, dictionary):
    """Adds a __slots__ entry to dictionary, containing the names of all valid
  attributes for this message type.

  Args:
    message_descriptor: A Descriptor instance describing this message type.
    dictionary: Class dictionary to which we'll add a '__slots__' entry.
  """
    dictionary['__slots__'] = ['_cached_byte_size', '_cached_byte_size_dirty', '_fields', '_unknown_fields', '_unknown_field_set', '_is_present_in_parent', '_listener', '_listener_for_children', '__weakref__', '_oneofs']

def _IsMessageSetExtension(field):
    return (field.is_extension and field.containing_type.has_options and field.containing_type.GetOptions().message_set_wire_format and field.type == _FieldDescriptor.TYPE_MESSAGE and field.label == _FieldDescriptor.LABEL_OPTIONAL)

def _IsMapField(field):
    return (field.type == _FieldDescriptor.TYPE_MESSAGE and field.message_type.has_options and field.message_type.GetOptions().map_entry)

def _IsMessageMapField(field):
    value_type = field.message_type.fields_by_name['value']
    return value_type.cpp_type == _FieldDescriptor.CPPTYPE_MESSAGE

def _AttachFieldHelpers(cls, field_descriptor):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('google.protobuf.internal.python_message._AttachFieldHelpers', '_AttachFieldHelpers(cls, field_descriptor)', {'_FieldDescriptor': _FieldDescriptor, 'wire_format': wire_format, '_IsMapField': _IsMapField, 'encoder': encoder, '_IsMessageMapField': _IsMessageMapField, '_IsMessageSetExtension': _IsMessageSetExtension, 'type_checkers': type_checkers, '_DefaultValueConstructorForField': _DefaultValueConstructorForField, 'decoder': decoder, '_GetInitializeDefaultForMap': _GetInitializeDefaultForMap, 'cls': cls, 'field_descriptor': field_descriptor}, 0)

def _AddClassAttributesForNestedExtensions(descriptor, dictionary):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('google.protobuf.internal.python_message._AddClassAttributesForNestedExtensions', '_AddClassAttributesForNestedExtensions(descriptor, dictionary)', {'descriptor': descriptor, 'dictionary': dictionary}, 0)

def _AddEnumValues(descriptor, cls):
    """Sets class-level attributes for all enum fields defined in this message.

  Also exporting a class-level object that can name enum values.

  Args:
    descriptor: Descriptor object for this message type.
    cls: Class we're constructing for this message type.
  """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('google.protobuf.internal.python_message._AddEnumValues', '_AddEnumValues(descriptor, cls)', {'enum_type_wrapper': enum_type_wrapper, 'descriptor': descriptor, 'cls': cls}, 0)

def _GetInitializeDefaultForMap(field):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.python_message._GetInitializeDefaultForMap', '_GetInitializeDefaultForMap(field)', {'_FieldDescriptor': _FieldDescriptor, 'type_checkers': type_checkers, '_IsMessageMapField': _IsMessageMapField, 'containers': containers, 'field': field}, 1)

def _DefaultValueConstructorForField(field):
    """Returns a function which returns a default value for a field.

  Args:
    field: FieldDescriptor object for this field.

  The returned function has one argument:
    message: Message instance containing this field, or a weakref proxy
      of same.

  That function in turn returns a default value for this field.  The default
    value may refer back to |message| via a weak reference.
  """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.python_message._DefaultValueConstructorForField', '_DefaultValueConstructorForField(field)', {'_IsMapField': _IsMapField, '_GetInitializeDefaultForMap': _GetInitializeDefaultForMap, '_FieldDescriptor': _FieldDescriptor, 'containers': containers, 'type_checkers': type_checkers, '_OneofListener': _OneofListener, 'field': field}, 1)

def _ReraiseTypeErrorWithFieldName(message_name, field_name):
    """Re-raise the currently-handled TypeError with the field name added."""
    import custom_funtemplate
    custom_funtemplate.rewrite_template('google.protobuf.internal.python_message._ReraiseTypeErrorWithFieldName', '_ReraiseTypeErrorWithFieldName(message_name, field_name)', {'sys': sys, 'message_name': message_name, 'field_name': field_name}, 0)

def _AddInitMethod(message_descriptor, cls):
    """Adds an __init__ method to cls."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.python_message._AddInitMethod', '_AddInitMethod(message_descriptor, cls)', {'message_listener_mod': message_listener_mod, '_Listener': _Listener, '_GetFieldByName': _GetFieldByName, '_FieldDescriptor': _FieldDescriptor, '_IsMapField': _IsMapField, '_IsMessageMapField': _IsMessageMapField, '_ReraiseTypeErrorWithFieldName': _ReraiseTypeErrorWithFieldName, 'message_descriptor': message_descriptor, 'cls': cls}, 1)

def _GetFieldByName(message_descriptor, field_name):
    """Returns a field descriptor by field name.

  Args:
    message_descriptor: A Descriptor describing all fields in message.
    field_name: The name of the field to retrieve.
  Returns:
    The field descriptor associated with the field name.
  """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.python_message._GetFieldByName', '_GetFieldByName(message_descriptor, field_name)', {'message_descriptor': message_descriptor, 'field_name': field_name}, 1)

def _AddPropertiesForFields(descriptor, cls):
    """Adds properties for all fields in this protocol message type."""
    import custom_funtemplate
    custom_funtemplate.rewrite_template('google.protobuf.internal.python_message._AddPropertiesForFields', '_AddPropertiesForFields(descriptor, cls)', {'_AddPropertiesForField': _AddPropertiesForField, '_ExtensionDict': _ExtensionDict, 'descriptor': descriptor, 'cls': cls}, 0)

def _AddPropertiesForField(field, cls):
    """Adds a public property for a protocol message field.
  Clients can use this property to get and (in the case
  of non-repeated scalar fields) directly set the value
  of a protocol message field.

  Args:
    field: A FieldDescriptor for this field.
    cls: The class we're constructing.
  """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('google.protobuf.internal.python_message._AddPropertiesForField', '_AddPropertiesForField(field, cls)', {'_FieldDescriptor': _FieldDescriptor, '_AddPropertiesForRepeatedField': _AddPropertiesForRepeatedField, '_AddPropertiesForNonRepeatedCompositeField': _AddPropertiesForNonRepeatedCompositeField, '_AddPropertiesForNonRepeatedScalarField': _AddPropertiesForNonRepeatedScalarField, 'field': field, 'cls': cls}, 0)


class _FieldProperty(property):
    __slots__ = ('DESCRIPTOR', )
    
    def __init__(self, descriptor, getter, setter, doc):
        property.__init__(self, getter, setter, doc=doc)
        self.DESCRIPTOR = descriptor


def _AddPropertiesForRepeatedField(field, cls):
    """Adds a public property for a "repeated" protocol message field.  Clients
  can use this property to get the value of the field, which will be either a
  RepeatedScalarFieldContainer or RepeatedCompositeFieldContainer (see
  below).

  Note that when clients add values to these containers, we perform
  type-checking in the case of repeated scalar fields, and we also set any
  necessary "has" bits as a side-effect.

  Args:
    field: A FieldDescriptor for this field.
    cls: The class we're constructing.
  """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.python_message._AddPropertiesForRepeatedField', '_AddPropertiesForRepeatedField(field, cls)', {'_PropertyName': _PropertyName, '_FieldProperty': _FieldProperty, 'field': field, 'cls': cls}, 1)

def _AddPropertiesForNonRepeatedScalarField(field, cls):
    """Adds a public property for a nonrepeated, scalar protocol message field.
  Clients can use this property to get and directly set the value of the field.
  Note that when the client sets the value of a field by using this property,
  all necessary "has" bits are set as a side-effect, and we also perform
  type-checking.

  Args:
    field: A FieldDescriptor for this field.
    cls: The class we're constructing.
  """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.python_message._AddPropertiesForNonRepeatedScalarField', '_AddPropertiesForNonRepeatedScalarField(field, cls)', {'_PropertyName': _PropertyName, 'type_checkers': type_checkers, '_FieldProperty': _FieldProperty, 'field': field, 'cls': cls}, 1)

def _AddPropertiesForNonRepeatedCompositeField(field, cls):
    """Adds a public property for a nonrepeated, composite protocol message field.
  A composite field is a "group" or "message" field.

  Clients can use this property to get the value of the field, but cannot
  assign to the property directly.

  Args:
    field: A FieldDescriptor for this field.
    cls: The class we're constructing.
  """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.python_message._AddPropertiesForNonRepeatedCompositeField', '_AddPropertiesForNonRepeatedCompositeField(field, cls)', {'_PropertyName': _PropertyName, '_FieldProperty': _FieldProperty, 'field': field, 'cls': cls}, 1)

def _AddPropertiesForExtensions(descriptor, cls):
    """Adds properties for all fields in this protocol message type."""
    import custom_funtemplate
    custom_funtemplate.rewrite_template('google.protobuf.internal.python_message._AddPropertiesForExtensions', '_AddPropertiesForExtensions(descriptor, cls)', {'descriptor': descriptor, 'cls': cls}, 0)

def _AddStaticMethods(cls):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.python_message._AddStaticMethods', '_AddStaticMethods(cls)', {'_AttachFieldHelpers': _AttachFieldHelpers, 'cls': cls}, 1)

def _IsPresent(item):
    """Given a (FieldDescriptor, value) tuple from _fields, return true if the
  value should be included in the list returned by ListFields()."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.python_message._IsPresent', '_IsPresent(item)', {'_FieldDescriptor': _FieldDescriptor, 'item': item}, 1)

def _AddListFieldsMethod(message_descriptor, cls):
    """Helper for _AddMessageMethods()."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.python_message._AddListFieldsMethod', '_AddListFieldsMethod(message_descriptor, cls)', {'_IsPresent': _IsPresent, 'message_descriptor': message_descriptor, 'cls': cls}, 1)
_PROTO3_ERROR_TEMPLATE = 'Protocol message %s has no non-repeated submessage field "%s" nor marked as optional'
_PROTO2_ERROR_TEMPLATE = 'Protocol message %s has no non-repeated field "%s"'

def _AddHasFieldMethod(message_descriptor, cls):
    """Helper for _AddMessageMethods()."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.python_message._AddHasFieldMethod', '_AddHasFieldMethod(message_descriptor, cls)', {'_PROTO3_ERROR_TEMPLATE': _PROTO3_ERROR_TEMPLATE, '_PROTO2_ERROR_TEMPLATE': _PROTO2_ERROR_TEMPLATE, '_FieldDescriptor': _FieldDescriptor, 'descriptor_mod': descriptor_mod, 'message_descriptor': message_descriptor, 'cls': cls}, 1)

def _AddClearFieldMethod(message_descriptor, cls):
    """Helper for _AddMessageMethods()."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.python_message._AddClearFieldMethod', '_AddClearFieldMethod(message_descriptor, cls)', {'message_descriptor': message_descriptor, 'cls': cls}, 1)

def _AddClearExtensionMethod(cls):
    """Helper for _AddMessageMethods()."""
    import custom_funtemplate
    custom_funtemplate.rewrite_template('google.protobuf.internal.python_message._AddClearExtensionMethod', '_AddClearExtensionMethod(cls)', {'extension_dict': extension_dict, 'cls': cls}, 0)

def _AddHasExtensionMethod(cls):
    """Helper for _AddMessageMethods()."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.python_message._AddHasExtensionMethod', '_AddHasExtensionMethod(cls)', {'extension_dict': extension_dict, '_FieldDescriptor': _FieldDescriptor, 'cls': cls}, 1)

def _InternalUnpackAny(msg):
    """Unpacks Any message and returns the unpacked message.

  This internal method is different from public Any Unpack method which takes
  the target message as argument. _InternalUnpackAny method does not have
  target message type and need to find the message type in descriptor pool.

  Args:
    msg: An Any message to be unpacked.

  Returns:
    The unpacked message.
  """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.python_message._InternalUnpackAny', '_InternalUnpackAny(msg)', {'msg': msg}, 1)

def _AddEqualsMethod(message_descriptor, cls):
    """Helper for _AddMessageMethods()."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.python_message._AddEqualsMethod', '_AddEqualsMethod(message_descriptor, cls)', {'message_mod': message_mod, '_AnyFullTypeName': _AnyFullTypeName, '_InternalUnpackAny': _InternalUnpackAny, 'message_descriptor': message_descriptor, 'cls': cls}, 1)

def _AddStrMethod(message_descriptor, cls):
    """Helper for _AddMessageMethods()."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.python_message._AddStrMethod', '_AddStrMethod(message_descriptor, cls)', {'text_format': text_format, 'message_descriptor': message_descriptor, 'cls': cls}, 1)

def _AddReprMethod(message_descriptor, cls):
    """Helper for _AddMessageMethods()."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.python_message._AddReprMethod', '_AddReprMethod(message_descriptor, cls)', {'text_format': text_format, 'message_descriptor': message_descriptor, 'cls': cls}, 1)

def _AddUnicodeMethod(unused_message_descriptor, cls):
    """Helper for _AddMessageMethods()."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.python_message._AddUnicodeMethod', '_AddUnicodeMethod(unused_message_descriptor, cls)', {'text_format': text_format, 'unused_message_descriptor': unused_message_descriptor, 'cls': cls}, 1)

def _BytesForNonRepeatedElement(value, field_number, field_type):
    """Returns the number of bytes needed to serialize a non-repeated element.
  The returned byte count includes space for tag information and any
  other additional space associated with serializing value.

  Args:
    value: Value we're serializing.
    field_number: Field number of this value.  (Since the field number
      is stored as part of a varint-encoded tag, this has an impact
      on the total bytes required to serialize the value).
    field_type: The type of the field.  One of the TYPE_* constants
      within FieldDescriptor.
  """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.python_message._BytesForNonRepeatedElement', '_BytesForNonRepeatedElement(value, field_number, field_type)', {'type_checkers': type_checkers, 'message_mod': message_mod, 'value': value, 'field_number': field_number, 'field_type': field_type}, 1)

def _AddByteSizeMethod(message_descriptor, cls):
    """Helper for _AddMessageMethods()."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.python_message._AddByteSizeMethod', '_AddByteSizeMethod(message_descriptor, cls)', {'message_descriptor': message_descriptor, 'cls': cls}, 1)

def _AddSerializeToStringMethod(message_descriptor, cls):
    """Helper for _AddMessageMethods()."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.python_message._AddSerializeToStringMethod', '_AddSerializeToStringMethod(message_descriptor, cls)', {'message_mod': message_mod, 'message_descriptor': message_descriptor, 'cls': cls}, 1)

def _AddSerializePartialToStringMethod(message_descriptor, cls):
    """Helper for _AddMessageMethods()."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.python_message._AddSerializePartialToStringMethod', '_AddSerializePartialToStringMethod(message_descriptor, cls)', {'BytesIO': BytesIO, 'api_implementation': api_implementation, 'message_descriptor': message_descriptor, 'cls': cls}, 1)

def _AddMergeFromStringMethod(message_descriptor, cls):
    """Helper for _AddMessageMethods()."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.python_message._AddMergeFromStringMethod', '_AddMergeFromStringMethod(message_descriptor, cls)', {'message_mod': message_mod, 'struct': struct, 'decoder': decoder, 'containers': containers, 'wire_format': wire_format, 'message_descriptor': message_descriptor, 'cls': cls}, 1)

def _AddIsInitializedMethod(message_descriptor, cls):
    """Adds the IsInitialized and FindInitializationError methods to the
  protocol message class."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.python_message._AddIsInitializedMethod', '_AddIsInitializedMethod(message_descriptor, cls)', {'_FieldDescriptor': _FieldDescriptor, '_IsMapField': _IsMapField, '_IsMessageMapField': _IsMessageMapField, 'message_descriptor': message_descriptor, 'cls': cls}, 1)

def _FullyQualifiedClassName(klass):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.python_message._FullyQualifiedClassName', '_FullyQualifiedClassName(klass)', {'klass': klass}, 1)

def _AddMergeFromMethod(cls):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('google.protobuf.internal.python_message._AddMergeFromMethod', '_AddMergeFromMethod(cls)', {'_FieldDescriptor': _FieldDescriptor, '_FullyQualifiedClassName': _FullyQualifiedClassName, 'containers': containers, 'cls': cls}, 0)

def _AddWhichOneofMethod(message_descriptor, cls):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.python_message._AddWhichOneofMethod', '_AddWhichOneofMethod(message_descriptor, cls)', {'message_descriptor': message_descriptor, 'cls': cls}, 1)

def _Clear(self):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('google.protobuf.internal.python_message._Clear', '_Clear(self)', {'self': self}, 0)

def _UnknownFields(self):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.internal.python_message._UnknownFields', '_UnknownFields(self)', {'containers': containers, 'self': self}, 1)

def _DiscardUnknownFields(self):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('google.protobuf.internal.python_message._DiscardUnknownFields', '_DiscardUnknownFields(self)', {'_FieldDescriptor': _FieldDescriptor, '_IsMapField': _IsMapField, '_IsMessageMapField': _IsMessageMapField, 'self': self}, 0)

def _SetListener(self, listener):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('google.protobuf.internal.python_message._SetListener', '_SetListener(self, listener)', {'message_listener_mod': message_listener_mod, 'self': self, 'listener': listener}, 0)

def _AddMessageMethods(message_descriptor, cls):
    """Adds implementations of all Message methods to cls."""
    import custom_funtemplate
    custom_funtemplate.rewrite_template('google.protobuf.internal.python_message._AddMessageMethods', '_AddMessageMethods(message_descriptor, cls)', {'_AddListFieldsMethod': _AddListFieldsMethod, '_AddHasFieldMethod': _AddHasFieldMethod, '_AddClearFieldMethod': _AddClearFieldMethod, '_AddClearExtensionMethod': _AddClearExtensionMethod, '_AddHasExtensionMethod': _AddHasExtensionMethod, '_AddEqualsMethod': _AddEqualsMethod, '_AddStrMethod': _AddStrMethod, '_AddReprMethod': _AddReprMethod, '_AddUnicodeMethod': _AddUnicodeMethod, '_AddByteSizeMethod': _AddByteSizeMethod, '_AddSerializeToStringMethod': _AddSerializeToStringMethod, '_AddSerializePartialToStringMethod': _AddSerializePartialToStringMethod, '_AddMergeFromStringMethod': _AddMergeFromStringMethod, '_AddIsInitializedMethod': _AddIsInitializedMethod, '_AddMergeFromMethod': _AddMergeFromMethod, '_AddWhichOneofMethod': _AddWhichOneofMethod, '_Clear': _Clear, '_UnknownFields': _UnknownFields, '_DiscardUnknownFields': _DiscardUnknownFields, '_SetListener': _SetListener, 'message_descriptor': message_descriptor, 'cls': cls}, 0)

def _AddPrivateHelperMethods(message_descriptor, cls):
    """Adds implementation of private helper methods to cls."""
    import custom_funtemplate
    custom_funtemplate.rewrite_template('google.protobuf.internal.python_message._AddPrivateHelperMethods', '_AddPrivateHelperMethods(message_descriptor, cls)', {'message_descriptor': message_descriptor, 'cls': cls}, 0)


class _Listener(object):
    """MessageListener implementation that a parent message registers with its
  child message.

  In order to support semantics like:

    foo.bar.baz.qux = 23
    assert foo.HasField('bar')

  ...child objects must have back references to their parents.
  This helper class is at the heart of this support.
  """
    
    def __init__(self, parent_message):
        """Args:
      parent_message: The message whose _Modified() method we should call when
        we receive Modified() messages.
    """
        if isinstance(parent_message, weakref.ProxyType):
            self._parent_message_weakref = parent_message
        else:
            self._parent_message_weakref = weakref.proxy(parent_message)
        self.dirty = False
    
    def Modified(self):
        if self.dirty:
            return
        try:
            self._parent_message_weakref._Modified()
        except ReferenceError:
            pass



class _OneofListener(_Listener):
    """Special listener implementation for setting composite oneof fields."""
    
    def __init__(self, parent_message, field):
        """Args:
      parent_message: The message whose _Modified() method we should call when
        we receive Modified() messages.
      field: The descriptor of the field being set in the parent message.
    """
        super(_OneofListener, self).__init__(parent_message)
        self._field = field
    
    def Modified(self):
        """Also updates the state of the containing oneof in the parent message."""
        try:
            self._parent_message_weakref._UpdateOneofState(self._field)
            super(_OneofListener, self).Modified()
        except ReferenceError:
            pass


