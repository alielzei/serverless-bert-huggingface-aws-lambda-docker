from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import workspace, core, rnn_cell
from caffe2.python.model_helper import ModelHelper
from caffe2.python.rnn.rnn_cell_test_util import tanh
import caffe2.python.hypothesis_test_util as hu
from hypothesis import given
from hypothesis import settings as ht_settings
import hypothesis.strategies as st
import numpy as np
import unittest

def basic_rnn_reference(input, hidden_initial, i2h_w, i2h_b, gate_w, gate_b, seq_lengths, drop_states, use_sequence_lengths):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.basic_rnn_test.basic_rnn_reference', 'basic_rnn_reference(input, hidden_initial, i2h_w, i2h_b, gate_w, gate_b, seq_lengths, drop_states, use_sequence_lengths)', {'np': np, 'tanh': tanh, 'input': input, 'hidden_initial': hidden_initial, 'i2h_w': i2h_w, 'i2h_b': i2h_b, 'gate_w': gate_w, 'gate_b': gate_b, 'seq_lengths': seq_lengths, 'drop_states': drop_states, 'use_sequence_lengths': use_sequence_lengths}, 1)


class BasicRNNCellTest(hu.HypothesisTestCase):
    
    @given(seed=st.integers(0, 2**32 - 1), seq_length=st.integers(min_value=1, max_value=5), batch_size=st.integers(min_value=1, max_value=5), input_size=st.integers(min_value=1, max_value=5), hidden_size=st.integers(min_value=1, max_value=5), drop_states=st.booleans(), sequence_lengths=st.booleans(), **hu.gcs)
    @ht_settings(max_examples=15)
    def test_basic_rnn(self, seed, seq_length, batch_size, input_size, hidden_size, drop_states, sequence_lengths, gc, dc):
        np.random.seed(seed)
        seq_lengths_data = np.random.randint(1, seq_length + 1, size=(batch_size, )).astype(np.int32)
        input_blob_data = np.random.randn(seq_length, batch_size, input_size).astype(np.float32)
        initial_h_data = np.random.randn(batch_size, hidden_size).astype(np.float32)
        gates_t_w_data = np.random.randn(hidden_size, hidden_size).astype(np.float32)
        gates_t_b_data = np.random.randn(hidden_size).astype(np.float32)
        i2h_w_data = np.random.randn(hidden_size, input_size).astype(np.float32)
        i2h_b_data = np.random.randn(hidden_size).astype(np.float32)
        with core.DeviceScope(gc):
            with hu.temp_workspace():
                workspace.FeedBlob('input_blob', input_blob_data, device_option=gc)
                workspace.FeedBlob('seq_lengths', seq_lengths_data, device_option=gc)
                workspace.FeedBlob('initial_h', initial_h_data, device_option=gc)
                workspace.FeedBlob('basic_rnn/gates_t_w', gates_t_w_data, device_option=gc)
                workspace.FeedBlob('basic_rnn/gates_t_b', gates_t_b_data, device_option=gc)
                workspace.FeedBlob('basic_rnn/i2h_w', i2h_w_data, device_option=gc)
                workspace.FeedBlob('basic_rnn/i2h_b', i2h_b_data, device_option=gc)
                model = ModelHelper(name='model')
                (hidden_t_all, _) = rnn_cell.BasicRNN(model, 'input_blob', ('seq_lengths' if sequence_lengths else None), ['initial_h'], input_size, hidden_size, 'basic_rnn', activation='tanh', forward_only=True, drop_states=drop_states)
                workspace.RunNetOnce(model.net)
                result = workspace.FetchBlob(hidden_t_all)
        reference = basic_rnn_reference(input_blob_data, initial_h_data, i2h_w_data, i2h_b_data, gates_t_w_data, gates_t_b_data, (seq_lengths_data if sequence_lengths else None), drop_states=drop_states, use_sequence_lengths=sequence_lengths)
        np.testing.assert_allclose(result, reference, atol=0.0001, rtol=0.0001)

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    unittest.main()

