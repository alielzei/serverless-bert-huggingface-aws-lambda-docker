"""Protocol message implementation hooks for C++ implementation.

Contains helper functions used to create protocol message classes from
Descriptor objects at runtime backed by the protocol buffer C++ API.
"""

__author__ = 'tibell@google.com (Johan Tibell)'
from google.protobuf.pyext import _message


class GeneratedProtocolMessageType(_message.MessageMeta):
    """Metaclass for protocol message classes created at runtime from Descriptors.

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

  The above example will not work for nested types. If you wish to include them,
  use reflection.MakeClass() instead of manually instantiating the class in
  order to create the appropriate class structure.
  """
    _DESCRIPTOR_KEY = 'DESCRIPTOR'


