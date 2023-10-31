from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import workspace, core, scope, gru_cell
from caffe2.python.model_helper import ModelHelper
from caffe2.python.rnn.rnn_cell_test_util import sigmoid, tanh, _prepare_rnn
import caffe2.python.hypothesis_test_util as hu
import caffe2.python.serialized_test.serialized_test_util as serial
from caffe2.proto import caffe2_pb2
from functools import partial
from hypothesis import given
from hypothesis import settings as ht_settings
import hypothesis.strategies as st
import numpy as np
import unittest
import os

def gru_unit(*args, **kwargs):
    """
    Implements one GRU unit, for one time step

    Shapes:
    hidden_t_prev.shape     = (1, N, D)
    gates_out_t.shape       = (1, N, G)
    seq_lenths.shape        = (N,)
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.gru_test.gru_unit', 'gru_unit(*args, **kwargs)', {'np': np, 'sigmoid': sigmoid, 'tanh': tanh, 'args': args, 'kwargs': kwargs}, 1)

def gru_reference(input, hidden_input, reset_gate_w, reset_gate_b, update_gate_w, update_gate_b, output_gate_w, output_gate_b, seq_lengths, drop_states=False, linear_before_reset=False):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.gru_test.gru_reference', 'gru_reference(input, hidden_input, reset_gate_w, reset_gate_b, update_gate_w, update_gate_b, output_gate_w, output_gate_b, seq_lengths, drop_states=False, linear_before_reset=False)', {'np': np, 'sigmoid': sigmoid, 'gru_unit': gru_unit, 'input': input, 'hidden_input': hidden_input, 'reset_gate_w': reset_gate_w, 'reset_gate_b': reset_gate_b, 'update_gate_w': update_gate_w, 'update_gate_b': update_gate_b, 'output_gate_w': output_gate_w, 'output_gate_b': output_gate_b, 'seq_lengths': seq_lengths, 'drop_states': drop_states, 'linear_before_reset': linear_before_reset}, 2)

def gru_unit_op_input():
    """
    Create input tensor where each dimension is from 1 to 4, ndim=3 and
    last dimension size is a factor of 3

    hidden_t_prev.shape     = (1, N, D)
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.gru_test.gru_unit_op_input', 'gru_unit_op_input()', {'st': st, 'hu': hu}, 1)

def gru_input():
    """
    Create input tensor where each dimension is from 1 to 4, ndim=3 and
    last dimension size is a factor of 3
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.gru_test.gru_input', 'gru_input()', {'st': st, 'hu': hu}, 1)

def _prepare_gru_unit_op(gc, n, d, outputs_with_grads, forward_only=False, drop_states=False, sequence_lengths=False, two_d_initial_states=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.gru_test._prepare_gru_unit_op', '_prepare_gru_unit_op(gc, n, d, outputs_with_grads, forward_only=False, drop_states=False, sequence_lengths=False, two_d_initial_states=None)', {'np': np, 'ModelHelper': ModelHelper, 'scope': scope, 'workspace': workspace, 'core': core, 'caffe2_pb2': caffe2_pb2, 'gc': gc, 'n': n, 'd': d, 'outputs_with_grads': outputs_with_grads, 'forward_only': forward_only, 'drop_states': drop_states, 'sequence_lengths': sequence_lengths, 'two_d_initial_states': two_d_initial_states}, 1)


class GRUCellTest(serial.SerializedTestCase):
    
    @serial.given(seed=st.integers(0, 2**32 - 1), input_tensor=gru_unit_op_input(), fwd_only=st.booleans(), drop_states=st.booleans(), sequence_lengths=st.booleans(), **hu.gcs)
    @ht_settings(max_examples=15)
    def test_gru_unit_op(self, seed, input_tensor, fwd_only, drop_states, sequence_lengths, gc, dc):
        np.random.seed(seed)
        outputs_with_grads = [0]
        ref = gru_unit
        ref = partial(ref)
        (t, n, d) = input_tensor.shape
        assert d % 3 == 0
        d = d // 3
        ref = partial(ref, drop_states=drop_states, sequence_lengths=sequence_lengths)
        with core.DeviceScope(gc):
            net = _prepare_gru_unit_op(gc, n, d, outputs_with_grads=outputs_with_grads, forward_only=fwd_only, drop_states=drop_states, sequence_lengths=sequence_lengths)[1]
        workspace.FeedBlob('test_name_scope/external/recurrent/i2h', input_tensor, device_option=gc)
        print(str(net.Proto()))
        op = net._net.op[-1]
        inputs = [workspace.FetchBlob(name) for name in op.input]
        self.assertReferenceChecks(gc, op, inputs, ref, input_device_options={'test_name_scope/timestep': hu.cpu_do}, outputs_to_check=[0])
        if not fwd_only:
            for param in range(2):
                print('Check param {}'.format(param))
                self.assertGradientChecks(device_option=gc, op=op, inputs=inputs, outputs_to_check=param, outputs_with_grads=outputs_with_grads, threshold=0.0001, stepsize=0.005, input_device_options={'test_name_scope/timestep': hu.cpu_do})
    
    @given(seed=st.integers(0, 2**32 - 1), input_tensor=gru_input(), fwd_only=st.booleans(), drop_states=st.booleans(), linear_before_reset=st.booleans(), **hu.gcs)
    @ht_settings(max_examples=20)
    def test_gru_main(self, seed, **kwargs):
        np.random.seed(seed)
        for outputs_with_grads in [[0], [1], [0, 1]]:
            self.gru_base(gru_cell.GRU, gru_reference, outputs_with_grads=outputs_with_grads, **kwargs)
    
    def gru_base(self, create_rnn, ref, outputs_with_grads, input_tensor, fwd_only, drop_states, linear_before_reset, gc, dc):
        print('GRU test parameters: ', locals())
        (t, n, d) = input_tensor.shape
        assert d % 3 == 0
        d = d // 3
        ref = partial(ref, drop_states=drop_states, linear_before_reset=linear_before_reset)
        with core.DeviceScope(gc):
            net = _prepare_rnn(t, n, d, create_rnn, outputs_with_grads=outputs_with_grads, memory_optim=False, forget_bias=0.0, forward_only=fwd_only, drop_states=drop_states, linear_before_reset=linear_before_reset, num_states=1)[1]
        workspace.FeedBlob('test_name_scope/external/recurrent/i2h', input_tensor, device_option=gc)
        op = net._net.op[-1]
        inputs = [workspace.FetchBlob(name) for name in op.input]
        self.assertReferenceChecks(gc, op, inputs, ref, input_device_options={'test_name_scope/timestep': hu.cpu_do}, outputs_to_check=list(range(2)))
        if not fwd_only:
            for param in range(2):
                print('Check param {}'.format(param))
                self.assertGradientChecks(device_option=gc, op=op, inputs=inputs, outputs_to_check=param, outputs_with_grads=outputs_with_grads, threshold=0.001, stepsize=0.005, input_device_options={'test_name_scope/timestep': hu.cpu_do})

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    unittest.main()

