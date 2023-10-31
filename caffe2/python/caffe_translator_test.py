from google.protobuf import text_format
import numpy as np
import os
import sys
CAFFE_FOUND = False
try:
    from caffe.proto import caffe_pb2
    from caffe2.python import caffe_translator
    CAFFE_FOUND = True
except Exception as e:
    if "'caffe'" in str(e):
        print('PyTorch/Caffe2 now requires a separate installation of caffe. Right now, this is not found, so we will skip the caffe translator test.')
from caffe2.python import utils, workspace, test_util
import unittest

def setUpModule():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.caffe_translator_test.setUpModule', 'setUpModule()', {'CAFFE_FOUND': CAFFE_FOUND, 'os': os, 'caffe_pb2': caffe_pb2, 'text_format': text_format, 'caffe_translator': caffe_translator, 'workspace': workspace, 'utils': utils, 'np': np}, 1)


@unittest.skipIf(not CAFFE_FOUND, 'No Caffe installation found.')
@unittest.skipIf(not os.path.exists('data/testdata/caffe_translator'), 'No testdata existing for the caffe translator test. Exiting.')
class TestNumericalEquivalence(test_util.TestCase):
    
    def testBlobs(self):
        names = ['conv1', 'pool1', 'norm1', 'conv2', 'pool2', 'norm2', 'conv3', 'conv4', 'conv5', 'pool5', 'fc6', 'fc7', 'fc8', 'prob']
        for name in names:
            print('Verifying {}'.format(name))
            caffe2_result = workspace.FetchBlob(name)
            reference = np.load('data/testdata/caffe_translator/' + name + '_dump.npy')
            self.assertEqual(caffe2_result.shape, reference.shape)
            scale = np.max(caffe2_result)
            np.testing.assert_almost_equal(caffe2_result / scale, reference / scale, decimal=5)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('If you do not explicitly ask to run this test, I will not run it. Pass in any argument to have the test run for you.')
        sys.exit(0)
    unittest.main()

