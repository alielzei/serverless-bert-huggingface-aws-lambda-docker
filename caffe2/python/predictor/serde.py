from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

def serialize_protobuf_struct(protobuf_struct):
    return protobuf_struct.SerializeToString()

def deserialize_protobuf_struct(serialized_protobuf, struct_type):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.predictor.serde.deserialize_protobuf_struct', 'deserialize_protobuf_struct(serialized_protobuf, struct_type)', {'serialized_protobuf': serialized_protobuf, 'struct_type': struct_type}, 1)

