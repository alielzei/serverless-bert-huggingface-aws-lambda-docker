from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from caffe2.python import core, workspace
from caffe2.proto import caffe2_pb2
import time
SHAPE_LEN = 4096
NUM_ITER = 1000
GB = 1024 * 1024 * 1024
NUM_REPLICAS = 48

def build_net(net_name, cross_socket):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.numa_benchmark.build_net', 'build_net(net_name, cross_socket)', {'core': core, 'caffe2_pb2': caffe2_pb2, 'NUM_REPLICAS': NUM_REPLICAS, 'SHAPE_LEN': SHAPE_LEN, 'net_name': net_name, 'cross_socket': cross_socket}, 2)

def main():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.numa_benchmark.main', 'main()', {'workspace': workspace, 'build_net': build_net, 'time': time, 'NUM_ITER': NUM_ITER, 'SHAPE_LEN': SHAPE_LEN, 'NUM_REPLICAS': NUM_REPLICAS, 'GB': GB}, 0)
if __name__ == '__main__':
    core.GlobalInit(['caffe2', '--caffe2_cpu_numa_enabled=1'])
    main()

