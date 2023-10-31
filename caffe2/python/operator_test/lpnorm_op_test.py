from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
from caffe2.python import core, workspace
import caffe2.python.hypothesis_test_util as hu
from hypothesis import given
import hypothesis.strategies as st


class LpnormTest(hu.HypothesisTestCase):
    
    @given(inputs=hu.tensors(n=1, min_dim=1, max_dim=3, dtype=np.float32), **hu.gcs_cpu_only)
    def test_Lp_Norm(self, inputs, gc, dc):
        X = inputs[0]
        X += 0.02 * np.sign(X)
        X[X == 0.0] += 0.02
        self.ws.create_blob('X').feed(X)
        op = core.CreateOperator('LpNorm', ['X'], ['l1_norm'], p=1)
        self.ws.run(op)
        np.testing.assert_allclose(self.ws.blobs['l1_norm'].fetch(), np.linalg.norm(X.flatten(), ord=1), rtol=0.0001, atol=0.0001)
        self.assertDeviceChecks(dc, op, [X], [0])
        self.assertGradientChecks(gc, op, [X], 0, [0], stepsize=0.01, threshold=0.01)
        op = core.CreateOperator('LpNorm', ['X'], ['l2_norm'], p=2)
        self.ws.run(op)
        np.testing.assert_allclose(self.ws.blobs['l2_norm'].fetch(), np.linalg.norm(X.flatten(), ord=2)**2, rtol=0.0001, atol=0.0001)
        self.assertDeviceChecks(dc, op, [X], [0])
        self.assertGradientChecks(gc, op, [X], 0, [0], stepsize=0.01, threshold=0.01)
        op = core.CreateOperator('LpNorm', ['X'], ['l2_averaged_norm'], p=2, average=True)
        self.ws.run(op)
        np.testing.assert_allclose(self.ws.blobs['l2_averaged_norm'].fetch(), np.linalg.norm(X.flatten(), ord=2)**2 / X.size, rtol=0.0001, atol=0.0001)
    
    @given(x=hu.tensor(min_dim=1, max_dim=10, dtype=np.float32, elements=st.integers(min_value=-100, max_value=100)), p=st.integers(1, 2), average=st.integers(0, 1))
    def test_lpnorm_shape_inference(self, x, p, average):
        workspace.FeedBlob('x', x)
        net = core.Net('lpnorm_test')
        result = net.LpNorm(['x'], p=p, average=bool(average))
        (shapes, types) = workspace.InferShapesAndTypes([net])
        workspace.RunNetOnce(net)
        self.assertEqual(shapes[result], list(workspace.blobs[result].shape))
        self.assertEqual(types[result], core.DataType.FLOAT)


