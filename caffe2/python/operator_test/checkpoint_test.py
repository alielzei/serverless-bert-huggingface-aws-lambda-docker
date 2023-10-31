from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import core, workspace, test_util
import os
import shutil
import tempfile
import unittest


class CheckpointTest(test_util.TestCase):
    """A simple test case to make sure that the checkpoint behavior is correct.
    """
    
    def testCheckpoint(self):
        temp_root = tempfile.mkdtemp()
        net = core.Net('test_checkpoint')
        net.Iter([], 'iter')
        net.ConstantFill([], 'value', shape=[1, 2, 3])
        net.Checkpoint(['iter', 'value'], [], db=os.path.join(temp_root, 'test_checkpoint_at_%05d'), db_type='leveldb', every=10, absolute_path=True)
        self.assertTrue(workspace.CreateNet(net))
        for i in range(100):
            self.assertTrue(workspace.RunNet('test_checkpoint'))
        for i in range(1, 10):
            self.assertTrue(os.path.exists(os.path.join(temp_root, 'test_checkpoint_at_%05d' % (i * 10))))
        shutil.rmtree(temp_root)

if __name__ == '__main__':
    unittest.main()

