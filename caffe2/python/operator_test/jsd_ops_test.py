from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import core
from hypothesis import given
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
import hypothesis.strategies as st
import numpy as np

def entropy(p):
    q = 1.0 - p
    return -p * np.log(p) - q * np.log(q)

def jsd(p, q):
    return [entropy(p / 2.0 + q / 2.0) - entropy(p) / 2.0 - entropy(q) / 2.0]

def jsd_grad(go, o, pq_list):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.jsd_ops_test.jsd_grad', 'jsd_grad(go, o, pq_list)', {'np': np, 'go': go, 'o': o, 'pq_list': pq_list}, 1)


class TestJSDOps(serial.SerializedTestCase):
    
    @serial.given(n=st.integers(10, 100), **hu.gcs_cpu_only)
    def test_bernoulli_jsd(self, n, gc, dc):
        p = np.random.rand(n).astype(np.float32)
        q = np.random.rand(n).astype(np.float32)
        op = core.CreateOperator('BernoulliJSD', ['p', 'q'], ['l'])
        self.assertReferenceChecks(device_option=gc, op=op, inputs=[p, q], reference=jsd, output_to_grad='l', grad_reference=jsd_grad)


