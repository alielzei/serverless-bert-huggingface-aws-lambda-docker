from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.proto import caffe2_pb2
from caffe2.python.compatibility import container_abcs
from future.utils import viewitems
from google.protobuf.message import DecodeError, Message
from google.protobuf import text_format
import sys
import copy
import functools
import numpy as np
from six import integer_types, binary_type, text_type, string_types
OPTIMIZER_ITERATION_NAME = 'optimizer_iteration'
ITERATION_MUTEX_NAME = 'iteration_mutex'

def OpAlmostEqual(op_a, op_b, ignore_fields=None):
    """
    Two ops are identical except for each field in the `ignore_fields`.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.utils.OpAlmostEqual', 'OpAlmostEqual(op_a, op_b, ignore_fields=None)', {'text_type': text_type, 'copy': copy, 'op_a': op_a, 'op_b': op_b, 'ignore_fields': ignore_fields}, 1)

def CaffeBlobToNumpyArray(blob):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.utils.CaffeBlobToNumpyArray', 'CaffeBlobToNumpyArray(blob)', {'np': np, 'blob': blob}, 1)

def Caffe2TensorToNumpyArray(tensor):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.utils.Caffe2TensorToNumpyArray', 'Caffe2TensorToNumpyArray(tensor)', {'caffe2_pb2': caffe2_pb2, 'np': np, 'tensor': tensor}, 1)

def NumpyArrayToCaffe2Tensor(arr, name=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.utils.NumpyArrayToCaffe2Tensor', 'NumpyArrayToCaffe2Tensor(arr, name=None)', {'caffe2_pb2': caffe2_pb2, 'np': np, 'arr': arr, 'name': name}, 1)

def MakeArgument(key, value):
    """Makes an argument based on the value type."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.utils.MakeArgument', 'MakeArgument(key, value)', {'caffe2_pb2': caffe2_pb2, 'container_abcs': container_abcs, 'np': np, 'integer_types': integer_types, 'binary_type': binary_type, 'text_type': text_type, 'Message': Message, 'key': key, 'value': value}, 1)

def TryReadProtoWithClass(cls, s):
    """Reads a protobuffer with the given proto class.

    Inputs:
      cls: a protobuffer class.
      s: a string of either binary or text protobuffer content.

    Outputs:
      proto: the protobuffer of cls

    Throws:
      google.protobuf.message.DecodeError: if we cannot decode the message.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.utils.TryReadProtoWithClass', 'TryReadProtoWithClass(cls, s)', {'text_format': text_format, 'cls': cls, 's': s}, 1)

def GetContentFromProto(obj, function_map):
    """Gets a specific field from a protocol buffer that matches the given class
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.utils.GetContentFromProto', 'GetContentFromProto(obj, function_map)', {'viewitems': viewitems, 'obj': obj, 'function_map': function_map}, 1)

def GetContentFromProtoString(s, function_map):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.utils.GetContentFromProtoString', 'GetContentFromProtoString(s, function_map)', {'viewitems': viewitems, 'TryReadProtoWithClass': TryReadProtoWithClass, 'DecodeError': DecodeError, 's': s, 'function_map': function_map}, 1)

def ConvertProtoToBinary(proto_class, filename, out_filename):
    """Convert a text file of the given protobuf class to binary."""
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.utils.ConvertProtoToBinary', 'ConvertProtoToBinary(proto_class, filename, out_filename)', {'TryReadProtoWithClass': TryReadProtoWithClass, 'proto_class': proto_class, 'filename': filename, 'out_filename': out_filename}, 0)

def GetGPUMemoryUsageStats():
    """Get GPU memory usage stats from CUDAContext/HIPContext. This requires flag
       --caffe2_gpu_memory_tracking to be enabled"""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.utils.GetGPUMemoryUsageStats', 'GetGPUMemoryUsageStats()', {'np': np}, 1)

def ResetBlobs(blobs):
    from caffe2.python import workspace, core
    workspace.RunOperatorOnce(core.CreateOperator('Free', list(blobs), list(blobs), device_option=core.DeviceOption(caffe2_pb2.CPU)))


class DebugMode(object):
    """
    This class allows to drop you into an interactive debugger
    if there is an unhandled exception in your python script

    Example of usage:

    def main():
        # your code here
        pass

    if __name__ == '__main__':
        from caffe2.python.utils import DebugMode
        DebugMode.run(main)
    """
    
    @classmethod
    def run(cls, func):
        try:
            return func()
        except KeyboardInterrupt:
            raise
        except Exception:
            import pdb
            print('Entering interactive debugger. Type "bt" to print the full stacktrace. Type "help" to see command listing.')
            print(sys.exc_info()[1])
            print
            pdb.post_mortem()
            sys.exit(1)
            raise


def raiseIfNotEqual(a, b, msg):
    if a != b:
        raise Exception('{}. {} != {}'.format(msg, a, b))

def debug(f):
    """
    Use this method to decorate your function with DebugMode's functionality

    Example:

    @debug
    def test_foo(self):
        raise Exception("Bar")

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.utils.debug', 'debug(f)', {'functools': functools, 'DebugMode': DebugMode, 'f': f}, 1)

def BuildUniqueMutexIter(init_net, net, iter=None, iter_mutex=None, iter_val=0):
    """
    Often, a mutex guarded iteration counter is needed. This function creates a
    mutex iter in the net uniquely (if the iter already existing, it does
    nothing)

    This function returns the iter blob
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.utils.BuildUniqueMutexIter', 'BuildUniqueMutexIter(init_net, net, iter=None, iter_mutex=None, iter_val=0)', {'OPTIMIZER_ITERATION_NAME': OPTIMIZER_ITERATION_NAME, 'ITERATION_MUTEX_NAME': ITERATION_MUTEX_NAME, 'caffe2_pb2': caffe2_pb2, 'init_net': init_net, 'net': net, 'iter': iter, 'iter_mutex': iter_mutex, 'iter_val': iter_val}, 1)

def EnumClassKeyVals(cls):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.utils.EnumClassKeyVals', 'EnumClassKeyVals(cls)', {'string_types': string_types, 'cls': cls}, 1)

def ArgsToDict(args):
    """
    Convert a list of arguments to a name, value dictionary. Assumes that
    each argument has a name. Otherwise, the argument is skipped.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.utils.ArgsToDict', 'ArgsToDict(args)', {'args': args}, 1)

def NHWC2NCHW(tensor):
    assert tensor.ndim >= 1
    return tensor.transpose((0, tensor.ndim - 1) + tuple(range(1, tensor.ndim - 1)))

def NCHW2NHWC(tensor):
    assert tensor.ndim >= 2
    return tensor.transpose((0, ) + tuple(range(2, tensor.ndim)) + (1, ))

