from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from caffe2.python import core, workspace
from caffe2.proto import caffe2_pb2
from caffe2.python.test_util import TestCase
import unittest
core.GlobalInit(['caffe2', '--caffe2_cpu_numa_enabled=1'])

def build_test_net(net_name):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.numa_test.build_test_net', 'build_test_net(net_name)', {'core': core, 'caffe2_pb2': caffe2_pb2, 'net_name': net_name}, 1)


@unittest.skipIf(not workspace.IsNUMAEnabled(), 'NUMA is not enabled')
@unittest.skipIf(workspace.GetNumNUMANodes() < 2, 'Not enough NUMA nodes')
@unittest.skipIf(not workspace.has_gpu_support, 'No GPU support')
class NUMATest(TestCase):
    
    def test_numa(self):
        net = build_test_net('test_numa')
        workspace.RunNetOnce(net)
        self.assertEqual(workspace.GetBlobNUMANode('output_blob_0'), 0)
        self.assertEqual(workspace.GetBlobNUMANode('output_blob_1'), 1)

if __name__ == '__main__':
    unittest.main()

