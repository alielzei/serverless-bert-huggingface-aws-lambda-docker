"""Provides a factory class for generating dynamic messages.

The easiest way to use this class is if you have access to the FileDescriptor
protos containing the messages you want to create you can just do the following:

message_classes = message_factory.GetMessages(iterable_of_file_descriptors)
my_proto_instance = message_classes['some.proto.package.MessageName']()
"""

__author__ = 'matthewtoia@google.com (Matt Toia)'
from google.protobuf.internal import api_implementation
from google.protobuf import descriptor_pool
from google.protobuf import message
if api_implementation.Type() == 'cpp':
    from google.protobuf.pyext import cpp_message as message_impl
else:
    from google.protobuf.internal import python_message as message_impl
_GENERATED_PROTOCOL_MESSAGE_TYPE = message_impl.GeneratedProtocolMessageType


class MessageFactory(object):
    """Factory for creating Proto2 messages from descriptors in a pool."""
    
    def __init__(self, pool=None):
        """Initializes a new factory."""
        self.pool = (pool or descriptor_pool.DescriptorPool())
        self._classes = {}
    
    def GetPrototype(self, descriptor):
        """Obtains a proto2 message class based on the passed in descriptor.

    Passing a descriptor with a fully qualified name matching a previous
    invocation will cause the same class to be returned.

    Args:
      descriptor: The descriptor to build from.

    Returns:
      A class describing the passed in descriptor.
    """
        if descriptor not in self._classes:
            result_class = self.CreatePrototype(descriptor)
            self._classes[descriptor] = result_class
            return result_class
        return self._classes[descriptor]
    
    def CreatePrototype(self, descriptor):
        """Builds a proto2 message class based on the passed in descriptor.

    Don't call this function directly, it always creates a new class. Call
    GetPrototype() instead. This method is meant to be overridden in subblasses
    to perform additional operations on the newly constructed class.

    Args:
      descriptor: The descriptor to build from.

    Returns:
      A class describing the passed in descriptor.
    """
        descriptor_name = descriptor.name
        result_class = _GENERATED_PROTOCOL_MESSAGE_TYPE(descriptor_name, (message.Message, ), {'DESCRIPTOR': descriptor, '__module__': None})
        result_class._FACTORY = self
        self._classes[descriptor] = result_class
        for field in descriptor.fields:
            if field.message_type:
                self.GetPrototype(field.message_type)
        for extension in result_class.DESCRIPTOR.extensions:
            if extension.containing_type not in self._classes:
                self.GetPrototype(extension.containing_type)
            extended_class = self._classes[extension.containing_type]
            extended_class.RegisterExtension(extension)
        return result_class
    
    def GetMessages(self, files):
        """Gets all the messages from a specified file.

    This will find and resolve dependencies, failing if the descriptor
    pool cannot satisfy them.

    Args:
      files: The file names to extract messages from.

    Returns:
      A dictionary mapping proto names to the message classes. This will include
      any dependent messages as well as any messages defined in the same file as
      a specified message.
    """
        result = {}
        for file_name in files:
            file_desc = self.pool.FindFileByName(file_name)
            for desc in file_desc.message_types_by_name.values():
                result[desc.full_name] = self.GetPrototype(desc)
            for extension in file_desc.extensions_by_name.values():
                if extension.containing_type not in self._classes:
                    self.GetPrototype(extension.containing_type)
                extended_class = self._classes[extension.containing_type]
                extended_class.RegisterExtension(extension)
        return result

_FACTORY = MessageFactory()

def GetMessages(file_protos):
    """Builds a dictionary of all the messages available in a set of files.

  Args:
    file_protos: Iterable of FileDescriptorProto to build messages out of.

  Returns:
    A dictionary mapping proto names to the message classes. This will include
    any dependent messages as well as any messages defined in the same file as
    a specified message.
  """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('google.protobuf.message_factory.GetMessages', 'GetMessages(file_protos)', {'_FACTORY': _FACTORY, 'file_protos': file_protos}, 1)

