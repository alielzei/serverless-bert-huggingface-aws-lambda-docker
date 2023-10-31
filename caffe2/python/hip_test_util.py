"""
The HIP test utils is a small addition on top of the hypothesis test utils
under caffe2/python, which allows one to more easily test HIP/ROCm related
operators.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.proto import caffe2_pb2

def run_in_hip(gc, dc):
    return (gc.device_type == caffe2_pb2.HIP or caffe2_pb2.HIP in {d.device_type for d in dc})

