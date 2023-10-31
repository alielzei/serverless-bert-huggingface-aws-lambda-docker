from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
from scipy.sparse import coo_matrix
from caffe2.python import core, workspace
from caffe2.python.test_util import TestCase

def test_reshape(old_shape, new_shape, stride_only=False):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.experiments.python.sparse_reshape_op_test.test_reshape', 'test_reshape(old_shape, new_shape, stride_only=False)', {'core': core, 'np': np, 'coo_matrix': coo_matrix, 'workspace': workspace, 'old_shape': old_shape, 'new_shape': new_shape, 'stride_only': stride_only}, 0)


class TestSparseMatrixReshapeOp(TestCase):
    
    def test_basic_reshape(self):
        test_reshape(old_shape=(3, 4), new_shape=(4, 3))
    
    def test_missing_dim(self):
        test_reshape(old_shape=(2, 8), new_shape=(-1, 4))
    
    def test_stride_only(self):
        test_reshape(old_shape=(2, 8), new_shape=(-1, 4), stride_only=True)
    
    def test_sparse_reshape_mm(self):
        (M, N, K) = (300, 400, 500)
        A = np.random.rand(M, K).astype(np.float32)
        A_sparse = A * (np.random.rand(*A.shape) > 0.5)
        A_sparse = A_sparse.reshape((K, M))
        A_coo = coo_matrix(A_sparse)
        (idx0, idx1, a) = (A_coo.row, A_coo.col, A_coo.data)
        B = np.random.rand(K, N).astype(np.float32)
        workspace.FeedBlob('col', idx1.astype(np.int64))
        workspace.FeedBlob('row', idx0.astype(np.int32))
        workspace.FeedBlob('B', B)
        workspace.FeedBlob('a', a)
        reshape_op = core.CreateOperator('SparseMatrixReshape', ['col', 'row'], ['new_col', 'new_row'], old_shape=(K, M), new_shape=(M, K))
        mm_op = core.CreateOperator('SparseUnsortedSegmentWeightedSum', ['B', 'a', 'new_col', 'new_row'], ['Y'])
        workspace.RunOperatorOnce(reshape_op)
        workspace.RunOperatorOnce(mm_op)
        Y = workspace.FetchBlob('Y')
        np.testing.assert_allclose(A_sparse.reshape(M, K).dot(B), Y, rtol=0.0001)


