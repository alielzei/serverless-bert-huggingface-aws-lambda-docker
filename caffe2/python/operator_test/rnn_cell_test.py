from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import core, gradient_checker, rnn_cell, workspace, scope, utils
from caffe2.python.attention import AttentionType
from caffe2.python.model_helper import ModelHelper, ExtractPredictorNet
from caffe2.python.rnn.rnn_cell_test_util import sigmoid, tanh, _prepare_rnn
from caffe2.proto import caffe2_pb2
import caffe2.python.hypothesis_test_util as hu
from functools import partial
from hypothesis import assume, given
from hypothesis import settings as ht_settings
import hypothesis.strategies as st
import numpy as np
import unittest

def lstm_unit(*args, **kwargs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.rnn_cell_test.lstm_unit', 'lstm_unit(*args, **kwargs)', {'np': np, 'sigmoid': sigmoid, 'tanh': tanh, 'args': args, 'kwargs': kwargs}, 2)

def layer_norm_with_scale_and_bias_ref(X, scale, bias, axis=-1, epsilon=0.0001):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.rnn_cell_test.layer_norm_with_scale_and_bias_ref', 'layer_norm_with_scale_and_bias_ref(X, scale, bias, axis=-1, epsilon=0.0001)', {'np': np, 'X': X, 'scale': scale, 'bias': bias, 'axis': axis, 'epsilon': epsilon}, 1)

def layer_norm_lstm_reference(input, hidden_input, cell_input, gates_w, gates_b, gates_t_norm_scale, gates_t_norm_bias, seq_lengths, forget_bias, drop_states=False):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.rnn_cell_test.layer_norm_lstm_reference', 'layer_norm_lstm_reference(input, hidden_input, cell_input, gates_w, gates_b, gates_t_norm_scale, gates_t_norm_bias, seq_lengths, forget_bias, drop_states=False)', {'np': np, 'layer_norm_with_scale_and_bias_ref': layer_norm_with_scale_and_bias_ref, 'lstm_unit': lstm_unit, 'input': input, 'hidden_input': hidden_input, 'cell_input': cell_input, 'gates_w': gates_w, 'gates_b': gates_b, 'gates_t_norm_scale': gates_t_norm_scale, 'gates_t_norm_bias': gates_t_norm_bias, 'seq_lengths': seq_lengths, 'forget_bias': forget_bias, 'drop_states': drop_states}, 4)

def lstm_reference(input, hidden_input, cell_input, gates_w, gates_b, seq_lengths, forget_bias, drop_states=False):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.rnn_cell_test.lstm_reference', 'lstm_reference(input, hidden_input, cell_input, gates_w, gates_b, seq_lengths, forget_bias, drop_states=False)', {'np': np, 'lstm_unit': lstm_unit, 'input': input, 'hidden_input': hidden_input, 'cell_input': cell_input, 'gates_w': gates_w, 'gates_b': gates_b, 'seq_lengths': seq_lengths, 'forget_bias': forget_bias, 'drop_states': drop_states}, 4)

def multi_lstm_reference(input, hidden_input_list, cell_input_list, i2h_w_list, i2h_b_list, gates_w_list, gates_b_list, seq_lengths, forget_bias, drop_states=False):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.rnn_cell_test.multi_lstm_reference', 'multi_lstm_reference(input, hidden_input_list, cell_input_list, i2h_w_list, i2h_b_list, gates_w_list, gates_b_list, seq_lengths, forget_bias, drop_states=False)', {'np': np, 'lstm_reference': lstm_reference, 'input': input, 'hidden_input_list': hidden_input_list, 'cell_input_list': cell_input_list, 'i2h_w_list': i2h_w_list, 'i2h_b_list': i2h_b_list, 'gates_w_list': gates_w_list, 'gates_b_list': gates_b_list, 'seq_lengths': seq_lengths, 'forget_bias': forget_bias, 'drop_states': drop_states}, 4)

def compute_regular_attention_logits(hidden_t, weighted_decoder_hidden_state_t_w, weighted_decoder_hidden_state_t_b, attention_weighted_encoder_context_t_prev, weighted_prev_attention_context_w, weighted_prev_attention_context_b, attention_v, weighted_encoder_outputs, encoder_outputs_for_dot_product, coverage_prev, coverage_weights):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.rnn_cell_test.compute_regular_attention_logits', 'compute_regular_attention_logits(hidden_t, weighted_decoder_hidden_state_t_w, weighted_decoder_hidden_state_t_b, attention_weighted_encoder_context_t_prev, weighted_prev_attention_context_w, weighted_prev_attention_context_b, attention_v, weighted_encoder_outputs, encoder_outputs_for_dot_product, coverage_prev, coverage_weights)', {'np': np, 'hidden_t': hidden_t, 'weighted_decoder_hidden_state_t_w': weighted_decoder_hidden_state_t_w, 'weighted_decoder_hidden_state_t_b': weighted_decoder_hidden_state_t_b, 'attention_weighted_encoder_context_t_prev': attention_weighted_encoder_context_t_prev, 'weighted_prev_attention_context_w': weighted_prev_attention_context_w, 'weighted_prev_attention_context_b': weighted_prev_attention_context_b, 'attention_v': attention_v, 'weighted_encoder_outputs': weighted_encoder_outputs, 'encoder_outputs_for_dot_product': encoder_outputs_for_dot_product, 'coverage_prev': coverage_prev, 'coverage_weights': coverage_weights}, 1)

def compute_recurrent_attention_logits(hidden_t, weighted_decoder_hidden_state_t_w, weighted_decoder_hidden_state_t_b, attention_weighted_encoder_context_t_prev, weighted_prev_attention_context_w, weighted_prev_attention_context_b, attention_v, weighted_encoder_outputs, encoder_outputs_for_dot_product, coverage_prev, coverage_weights):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.rnn_cell_test.compute_recurrent_attention_logits', 'compute_recurrent_attention_logits(hidden_t, weighted_decoder_hidden_state_t_w, weighted_decoder_hidden_state_t_b, attention_weighted_encoder_context_t_prev, weighted_prev_attention_context_w, weighted_prev_attention_context_b, attention_v, weighted_encoder_outputs, encoder_outputs_for_dot_product, coverage_prev, coverage_weights)', {'np': np, 'hidden_t': hidden_t, 'weighted_decoder_hidden_state_t_w': weighted_decoder_hidden_state_t_w, 'weighted_decoder_hidden_state_t_b': weighted_decoder_hidden_state_t_b, 'attention_weighted_encoder_context_t_prev': attention_weighted_encoder_context_t_prev, 'weighted_prev_attention_context_w': weighted_prev_attention_context_w, 'weighted_prev_attention_context_b': weighted_prev_attention_context_b, 'attention_v': attention_v, 'weighted_encoder_outputs': weighted_encoder_outputs, 'encoder_outputs_for_dot_product': encoder_outputs_for_dot_product, 'coverage_prev': coverage_prev, 'coverage_weights': coverage_weights}, 1)

def compute_dot_attention_logits(hidden_t, weighted_decoder_hidden_state_t_w, weighted_decoder_hidden_state_t_b, attention_weighted_encoder_context_t_prev, weighted_prev_attention_context_w, weighted_prev_attention_context_b, attention_v, weighted_encoder_outputs, encoder_outputs_for_dot_product, coverage_prev, coverage_weights):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.rnn_cell_test.compute_dot_attention_logits', 'compute_dot_attention_logits(hidden_t, weighted_decoder_hidden_state_t_w, weighted_decoder_hidden_state_t_b, attention_weighted_encoder_context_t_prev, weighted_prev_attention_context_w, weighted_prev_attention_context_b, attention_v, weighted_encoder_outputs, encoder_outputs_for_dot_product, coverage_prev, coverage_weights)', {'np': np, 'hidden_t': hidden_t, 'weighted_decoder_hidden_state_t_w': weighted_decoder_hidden_state_t_w, 'weighted_decoder_hidden_state_t_b': weighted_decoder_hidden_state_t_b, 'attention_weighted_encoder_context_t_prev': attention_weighted_encoder_context_t_prev, 'weighted_prev_attention_context_w': weighted_prev_attention_context_w, 'weighted_prev_attention_context_b': weighted_prev_attention_context_b, 'attention_v': attention_v, 'weighted_encoder_outputs': weighted_encoder_outputs, 'encoder_outputs_for_dot_product': encoder_outputs_for_dot_product, 'coverage_prev': coverage_prev, 'coverage_weights': coverage_weights}, 1)

def compute_coverage_attention_logits(hidden_t, weighted_decoder_hidden_state_t_w, weighted_decoder_hidden_state_t_b, attention_weighted_encoder_context_t_prev, weighted_prev_attention_context_w, weighted_prev_attention_context_b, attention_v, weighted_encoder_outputs, encoder_outputs_for_dot_product, coverage_prev, coverage_weights):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.rnn_cell_test.compute_coverage_attention_logits', 'compute_coverage_attention_logits(hidden_t, weighted_decoder_hidden_state_t_w, weighted_decoder_hidden_state_t_b, attention_weighted_encoder_context_t_prev, weighted_prev_attention_context_w, weighted_prev_attention_context_b, attention_v, weighted_encoder_outputs, encoder_outputs_for_dot_product, coverage_prev, coverage_weights)', {'np': np, 'hidden_t': hidden_t, 'weighted_decoder_hidden_state_t_w': weighted_decoder_hidden_state_t_w, 'weighted_decoder_hidden_state_t_b': weighted_decoder_hidden_state_t_b, 'attention_weighted_encoder_context_t_prev': attention_weighted_encoder_context_t_prev, 'weighted_prev_attention_context_w': weighted_prev_attention_context_w, 'weighted_prev_attention_context_b': weighted_prev_attention_context_b, 'attention_v': attention_v, 'weighted_encoder_outputs': weighted_encoder_outputs, 'encoder_outputs_for_dot_product': encoder_outputs_for_dot_product, 'coverage_prev': coverage_prev, 'coverage_weights': coverage_weights}, 1)

def lstm_with_attention_reference(input, initial_hidden_state, initial_cell_state, initial_attention_weighted_encoder_context, gates_w, gates_b, decoder_input_lengths, encoder_outputs_transposed, weighted_prev_attention_context_w, weighted_prev_attention_context_b, weighted_decoder_hidden_state_t_w, weighted_decoder_hidden_state_t_b, weighted_encoder_outputs, coverage_weights, attention_v, attention_zeros, compute_attention_logits):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.rnn_cell_test.lstm_with_attention_reference', 'lstm_with_attention_reference(input, initial_hidden_state, initial_cell_state, initial_attention_weighted_encoder_context, gates_w, gates_b, decoder_input_lengths, encoder_outputs_transposed, weighted_prev_attention_context_w, weighted_prev_attention_context_b, weighted_decoder_hidden_state_t_w, weighted_decoder_hidden_state_t_b, weighted_encoder_outputs, coverage_weights, attention_v, attention_zeros, compute_attention_logits)', {'np': np, 'lstm_unit': lstm_unit, 'input': input, 'initial_hidden_state': initial_hidden_state, 'initial_cell_state': initial_cell_state, 'initial_attention_weighted_encoder_context': initial_attention_weighted_encoder_context, 'gates_w': gates_w, 'gates_b': gates_b, 'decoder_input_lengths': decoder_input_lengths, 'encoder_outputs_transposed': encoder_outputs_transposed, 'weighted_prev_attention_context_w': weighted_prev_attention_context_w, 'weighted_prev_attention_context_b': weighted_prev_attention_context_b, 'weighted_decoder_hidden_state_t_w': weighted_decoder_hidden_state_t_w, 'weighted_decoder_hidden_state_t_b': weighted_decoder_hidden_state_t_b, 'weighted_encoder_outputs': weighted_encoder_outputs, 'coverage_weights': coverage_weights, 'attention_v': attention_v, 'attention_zeros': attention_zeros, 'compute_attention_logits': compute_attention_logits}, 6)

def lstm_with_regular_attention_reference(input, initial_hidden_state, initial_cell_state, initial_attention_weighted_encoder_context, gates_w, gates_b, decoder_input_lengths, weighted_decoder_hidden_state_t_w, weighted_decoder_hidden_state_t_b, weighted_encoder_outputs, attention_v, attention_zeros, encoder_outputs_transposed):
    return lstm_with_attention_reference(input=input, initial_hidden_state=initial_hidden_state, initial_cell_state=initial_cell_state, initial_attention_weighted_encoder_context=initial_attention_weighted_encoder_context, gates_w=gates_w, gates_b=gates_b, decoder_input_lengths=decoder_input_lengths, encoder_outputs_transposed=encoder_outputs_transposed, weighted_prev_attention_context_w=None, weighted_prev_attention_context_b=None, weighted_decoder_hidden_state_t_w=weighted_decoder_hidden_state_t_w, weighted_decoder_hidden_state_t_b=weighted_decoder_hidden_state_t_b, weighted_encoder_outputs=weighted_encoder_outputs, coverage_weights=None, attention_v=attention_v, attention_zeros=attention_zeros, compute_attention_logits=compute_regular_attention_logits)

def lstm_with_recurrent_attention_reference(input, initial_hidden_state, initial_cell_state, initial_attention_weighted_encoder_context, gates_w, gates_b, decoder_input_lengths, weighted_prev_attention_context_w, weighted_prev_attention_context_b, weighted_decoder_hidden_state_t_w, weighted_decoder_hidden_state_t_b, weighted_encoder_outputs, attention_v, attention_zeros, encoder_outputs_transposed):
    return lstm_with_attention_reference(input=input, initial_hidden_state=initial_hidden_state, initial_cell_state=initial_cell_state, initial_attention_weighted_encoder_context=initial_attention_weighted_encoder_context, gates_w=gates_w, gates_b=gates_b, decoder_input_lengths=decoder_input_lengths, encoder_outputs_transposed=encoder_outputs_transposed, weighted_prev_attention_context_w=weighted_prev_attention_context_w, weighted_prev_attention_context_b=weighted_prev_attention_context_b, weighted_decoder_hidden_state_t_w=weighted_decoder_hidden_state_t_w, weighted_decoder_hidden_state_t_b=weighted_decoder_hidden_state_t_b, weighted_encoder_outputs=weighted_encoder_outputs, coverage_weights=None, attention_v=attention_v, attention_zeros=attention_zeros, compute_attention_logits=compute_recurrent_attention_logits)

def lstm_with_dot_attention_reference(input, initial_hidden_state, initial_cell_state, initial_attention_weighted_encoder_context, gates_w, gates_b, decoder_input_lengths, encoder_outputs_transposed, weighted_decoder_hidden_state_t_w, weighted_decoder_hidden_state_t_b):
    return lstm_with_attention_reference(input=input, initial_hidden_state=initial_hidden_state, initial_cell_state=initial_cell_state, initial_attention_weighted_encoder_context=initial_attention_weighted_encoder_context, gates_w=gates_w, gates_b=gates_b, decoder_input_lengths=decoder_input_lengths, encoder_outputs_transposed=encoder_outputs_transposed, weighted_prev_attention_context_w=None, weighted_prev_attention_context_b=None, weighted_decoder_hidden_state_t_w=weighted_decoder_hidden_state_t_w, weighted_decoder_hidden_state_t_b=weighted_decoder_hidden_state_t_b, weighted_encoder_outputs=None, coverage_weights=None, attention_v=None, attention_zeros=None, compute_attention_logits=compute_dot_attention_logits)

def lstm_with_dot_attention_reference_same_dim(input, initial_hidden_state, initial_cell_state, initial_attention_weighted_encoder_context, gates_w, gates_b, decoder_input_lengths, encoder_outputs_transposed):
    return lstm_with_dot_attention_reference(input=input, initial_hidden_state=initial_hidden_state, initial_cell_state=initial_cell_state, initial_attention_weighted_encoder_context=initial_attention_weighted_encoder_context, gates_w=gates_w, gates_b=gates_b, decoder_input_lengths=decoder_input_lengths, encoder_outputs_transposed=encoder_outputs_transposed, weighted_decoder_hidden_state_t_w=None, weighted_decoder_hidden_state_t_b=None)

def lstm_with_dot_attention_reference_different_dim(input, initial_hidden_state, initial_cell_state, initial_attention_weighted_encoder_context, gates_w, gates_b, decoder_input_lengths, weighted_decoder_hidden_state_t_w, weighted_decoder_hidden_state_t_b, encoder_outputs_transposed):
    return lstm_with_dot_attention_reference(input=input, initial_hidden_state=initial_hidden_state, initial_cell_state=initial_cell_state, initial_attention_weighted_encoder_context=initial_attention_weighted_encoder_context, gates_w=gates_w, gates_b=gates_b, decoder_input_lengths=decoder_input_lengths, encoder_outputs_transposed=encoder_outputs_transposed, weighted_decoder_hidden_state_t_w=weighted_decoder_hidden_state_t_w, weighted_decoder_hidden_state_t_b=weighted_decoder_hidden_state_t_b)

def lstm_with_coverage_attention_reference(input, initial_hidden_state, initial_cell_state, initial_attention_weighted_encoder_context, initial_coverage, gates_w, gates_b, decoder_input_lengths, weighted_decoder_hidden_state_t_w, weighted_decoder_hidden_state_t_b, weighted_encoder_outputs, coverage_weights, attention_v, attention_zeros, encoder_outputs_transposed):
    return lstm_with_attention_reference(input=input, initial_hidden_state=initial_hidden_state, initial_cell_state=initial_cell_state, initial_attention_weighted_encoder_context=initial_attention_weighted_encoder_context, gates_w=gates_w, gates_b=gates_b, decoder_input_lengths=decoder_input_lengths, encoder_outputs_transposed=encoder_outputs_transposed, weighted_prev_attention_context_w=None, weighted_prev_attention_context_b=None, weighted_decoder_hidden_state_t_w=weighted_decoder_hidden_state_t_w, weighted_decoder_hidden_state_t_b=weighted_decoder_hidden_state_t_b, weighted_encoder_outputs=weighted_encoder_outputs, coverage_weights=coverage_weights, attention_v=attention_v, attention_zeros=attention_zeros, compute_attention_logits=compute_coverage_attention_logits)

def milstm_reference(input, hidden_input, cell_input, gates_w, gates_b, alpha, beta1, beta2, b, seq_lengths, forget_bias, drop_states=False):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.rnn_cell_test.milstm_reference', 'milstm_reference(input, hidden_input, cell_input, gates_w, gates_b, alpha, beta1, beta2, b, seq_lengths, forget_bias, drop_states=False)', {'np': np, 'lstm_unit': lstm_unit, 'input': input, 'hidden_input': hidden_input, 'cell_input': cell_input, 'gates_w': gates_w, 'gates_b': gates_b, 'alpha': alpha, 'beta1': beta1, 'beta2': beta2, 'b': b, 'seq_lengths': seq_lengths, 'forget_bias': forget_bias, 'drop_states': drop_states}, 4)

def layer_norm_milstm_reference(input, hidden_input, cell_input, gates_w, gates_b, alpha, beta1, beta2, b, gates_t_norm_scale, gates_t_norm_bias, seq_lengths, forget_bias, drop_states=False):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.rnn_cell_test.layer_norm_milstm_reference', 'layer_norm_milstm_reference(input, hidden_input, cell_input, gates_w, gates_b, alpha, beta1, beta2, b, gates_t_norm_scale, gates_t_norm_bias, seq_lengths, forget_bias, drop_states=False)', {'np': np, 'layer_norm_with_scale_and_bias_ref': layer_norm_with_scale_and_bias_ref, 'lstm_unit': lstm_unit, 'input': input, 'hidden_input': hidden_input, 'cell_input': cell_input, 'gates_w': gates_w, 'gates_b': gates_b, 'alpha': alpha, 'beta1': beta1, 'beta2': beta2, 'b': b, 'gates_t_norm_scale': gates_t_norm_scale, 'gates_t_norm_bias': gates_t_norm_bias, 'seq_lengths': seq_lengths, 'forget_bias': forget_bias, 'drop_states': drop_states}, 4)

def lstm_input():
    """
    Create input tensor where each dimension is from 1 to 4, ndim=3 and
    last dimension size is a factor of 4
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.rnn_cell_test.lstm_input', 'lstm_input()', {'st': st, 'hu': hu}, 1)

def _prepare_attention(t, n, dim_in, encoder_dim, forward_only=False, T=None, dim_out=None, residual=False, final_dropout=False):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.rnn_cell_test._prepare_attention', '_prepare_attention(t, n, dim_in, encoder_dim, forward_only=False, T=None, dim_out=None, residual=False, final_dropout=False)', {'ModelHelper': ModelHelper, 'np': np, 'workspace': workspace, 'scope': scope, 'rnn_cell': rnn_cell, 'AttentionType': AttentionType, 't': t, 'n': n, 'dim_in': dim_in, 'encoder_dim': encoder_dim, 'forward_only': forward_only, 'T': T, 'dim_out': dim_out, 'residual': residual, 'final_dropout': final_dropout}, 1)


class MulCell(rnn_cell.RNNCell):
    
    def _apply(self, model, input_t, seq_lengths, states, timestep, extra_inputs):
        assert len(states) == 1
        result = model.net.Mul([input_t, states[0]])
        model.net.AddExternalOutput(result)
        return [result]
    
    def get_state_names(self):
        return [self.scope('state')]


def prepare_mul_rnn(model, input_blob, shape, T, outputs_with_grad, num_layers):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.rnn_cell_test.prepare_mul_rnn', 'prepare_mul_rnn(model, input_blob, shape, T, outputs_with_grad, num_layers)', {'MulCell': MulCell, 'rnn_cell': rnn_cell, 'model': model, 'input_blob': input_blob, 'shape': shape, 'T': T, 'outputs_with_grad': outputs_with_grad, 'num_layers': num_layers}, 1)


class RNNCellTest(hu.HypothesisTestCase):
    
    @given(input_tensor=hu.tensor(min_dim=3, max_dim=3, max_value=3), num_layers=st.integers(1, 4), outputs_with_grad=st.sampled_from([[0], [1], [0, 1]]))
    @ht_settings(max_examples=10)
    def test_unroll_mul(self, input_tensor, num_layers, outputs_with_grad):
        outputs = []
        nets = []
        input_blob = None
        for T in [input_tensor.shape[0], None]:
            model = ModelHelper('rnn_mul_{}'.format(('unroll' if T else 'dynamic')))
            input_blob = model.net.AddExternalInputs('input_blob')
            outputs.append(prepare_mul_rnn(model, input_blob, input_tensor.shape, T, outputs_with_grad, num_layers))
            workspace.RunNetOnce(model.param_init_net)
            nets.append(model.net)
            workspace.blobs[input_blob] = input_tensor
        gradient_checker.NetGradientChecker.CompareNets(nets, outputs, outputs_with_grad_ids=outputs_with_grad, inputs_with_grads=[input_blob])
    
    @given(input_tensor=hu.tensor(min_dim=3, max_dim=3, max_value=3), forget_bias=st.floats(-10.0, 10.0), drop_states=st.booleans(), dim_out=st.lists(elements=st.integers(min_value=1, max_value=3), min_size=1, max_size=3), outputs_with_grads=st.sampled_from([[0], [1], [0, 1], [0, 2], [0, 1, 2, 3]]))
    @ht_settings(max_examples=10)
    @utils.debug
    def test_unroll_lstm(self, input_tensor, dim_out, outputs_with_grads, **kwargs):
        lstms = [_prepare_rnn(*input_tensor.shape, create_rnn=rnn_cell.LSTM, outputs_with_grads=outputs_with_grads, T=T, two_d_initial_states=False, dim_out=dim_out, **kwargs) for T in [input_tensor.shape[0], None]]
        (outputs, nets, inputs) = zip(*lstms)
        workspace.FeedBlob(inputs[0][-1], input_tensor)
        assert inputs[0] == inputs[1]
        gradient_checker.NetGradientChecker.CompareNets(nets, outputs, outputs_with_grads, inputs_with_grads=inputs[0])
    
    @given(input_tensor=hu.tensor(min_dim=3, max_dim=3, max_value=3), encoder_length=st.integers(min_value=1, max_value=3), encoder_dim=st.integers(min_value=1, max_value=3), hidden_units=st.integers(min_value=1, max_value=3), num_layers=st.integers(min_value=1, max_value=3), residual=st.booleans(), final_dropout=st.booleans())
    @ht_settings(max_examples=10)
    @utils.debug
    def test_unroll_attention(self, input_tensor, encoder_length, encoder_dim, hidden_units, num_layers, residual, final_dropout):
        dim_out = [hidden_units] * num_layers
        encoder_tensor = np.random.random((encoder_length, input_tensor.shape[1], encoder_dim)).astype('float32')
        print('Decoder input shape: {}'.format(input_tensor.shape))
        print('Encoder output shape: {}'.format(encoder_tensor.shape))
        workspace.ResetWorkspace()
        (net, unrolled) = [_prepare_attention(t=input_tensor.shape[0], n=input_tensor.shape[1], dim_in=input_tensor.shape[2], encoder_dim=encoder_dim, T=T, dim_out=dim_out, residual=residual, final_dropout=final_dropout) for T in [input_tensor.shape[0], None]]
        workspace.FeedBlob(net['input_blob'], input_tensor)
        workspace.FeedBlob(net['encoder_outputs'], encoder_tensor)
        workspace.FeedBlob(net['weighted_encoder_outputs'], np.random.random(encoder_tensor.shape).astype('float32'))
        for input_name in ['input_blob', 'encoder_outputs', 'weighted_encoder_outputs']:
            assert net[input_name] == unrolled[input_name]
        for (state_name, unrolled_state_name) in zip(net['initial_states'], unrolled['initial_states']):
            assert state_name == unrolled_state_name
        inputs_with_grads = net['initial_states'] + [net['input_blob'], net['encoder_outputs'], net['weighted_encoder_outputs']]
        gradient_checker.NetGradientChecker.CompareNets([net['net'], unrolled['net']], [[net['final_output']], [unrolled['final_output']]], [0], inputs_with_grads=inputs_with_grads, threshold=1e-06)
    
    @given(input_tensor=hu.tensor(min_dim=3, max_dim=3), forget_bias=st.floats(-10.0, 10.0), forward_only=st.booleans(), drop_states=st.booleans())
    @ht_settings(max_examples=10)
    def test_layered_lstm(self, input_tensor, **kwargs):
        for outputs_with_grads in [[0], [1], [0, 1, 2, 3]]:
            for memory_optim in [False, True]:
                (_, net, inputs) = _prepare_rnn(*input_tensor.shape, create_rnn=rnn_cell.LSTM, outputs_with_grads=outputs_with_grads, memory_optim=memory_optim, **kwargs)
                workspace.FeedBlob(inputs[-1], input_tensor)
                workspace.RunNetOnce(net)
                workspace.ResetWorkspace()
    
    def test_lstm(self):
        self.lstm_base(lstm_type=(rnn_cell.LSTM, lstm_reference))
    
    def test_milstm(self):
        self.lstm_base(lstm_type=(rnn_cell.MILSTM, milstm_reference))
    
    @unittest.skip('This is currently numerically unstable')
    def test_norm_lstm(self):
        self.lstm_base(lstm_type=(rnn_cell.LayerNormLSTM, layer_norm_lstm_reference))
    
    @unittest.skip('This is currently numerically unstable')
    def test_norm_milstm(self):
        self.lstm_base(lstm_type=(rnn_cell.LayerNormMILSTM, layer_norm_milstm_reference))
    
    @given(seed=st.integers(0, 2**32 - 1), input_tensor=lstm_input(), forget_bias=st.floats(-10.0, 10.0), fwd_only=st.booleans(), drop_states=st.booleans(), memory_optim=st.booleans(), outputs_with_grads=st.sampled_from([[0], [1], [0, 1, 2, 3]]))
    def lstm_base(self, seed, lstm_type, outputs_with_grads, memory_optim, input_tensor, forget_bias, fwd_only, drop_states):
        np.random.seed(seed)
        (create_lstm, ref) = lstm_type
        ref = partial(ref, forget_bias=forget_bias)
        (t, n, d) = input_tensor.shape
        assert d % 4 == 0
        d = d // 4
        ref = partial(ref, forget_bias=forget_bias, drop_states=drop_states)
        net = _prepare_rnn(t, n, d, create_lstm, outputs_with_grads=outputs_with_grads, memory_optim=memory_optim, forget_bias=forget_bias, forward_only=fwd_only, drop_states=drop_states)[1]
        workspace.FeedBlob('test_name_scope/external/recurrent/i2h', input_tensor)
        op = net._net.op[-1]
        inputs = [workspace.FetchBlob(name) for name in op.input]
        if fwd_only:
            for arg in op.arg:
                self.assertFalse(arg.name == 'backward_step_net')
        self.assertReferenceChecks(hu.cpu_do, op, inputs, ref, outputs_to_check=list(range(4)))
        if not fwd_only:
            for param in range(5):
                self.assertGradientChecks(device_option=hu.cpu_do, op=op, inputs=inputs, outputs_to_check=param, outputs_with_grads=outputs_with_grads, threshold=0.01, stepsize=0.005)
    
    def test_lstm_extract_predictor_net(self):
        model = ModelHelper(name='lstm_extract_test')
        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CPU, 0)):
            (output, _, _, _) = rnn_cell.LSTM(model=model, input_blob='input', seq_lengths='seqlengths', initial_states=('hidden_init', 'cell_init'), dim_in=20, dim_out=40, scope='test', drop_states=True, return_last_layer_only=True)
        shapes = {}
        workspace.RunNetOnce(model.param_init_net)
        for b in workspace.Blobs():
            shapes[b] = workspace.FetchBlob(b).shape
        (predict_net, export_blobs) = ExtractPredictorNet(net_proto=model.net.Proto(), input_blobs=['input'], output_blobs=[output], device=core.DeviceOption(caffe2_pb2.CPU, 1))
        workspace.ResetWorkspace()
        shapes['input'] = [10, 4, 20]
        shapes['cell_init'] = [1, 4, 40]
        shapes['hidden_init'] = [1, 4, 40]
        print(predict_net.Proto().external_input)
        self.assertTrue('seqlengths' in predict_net.Proto().external_input)
        for einp in predict_net.Proto().external_input:
            if einp == 'seqlengths':
                workspace.FeedBlob('seqlengths', np.array([10] * 4, dtype=np.int32))
            else:
                workspace.FeedBlob(einp, np.zeros(shapes[einp]).astype(np.float32))
                if einp != 'input':
                    self.assertTrue(einp in export_blobs)
        print(str(predict_net.Proto()))
        self.assertTrue(workspace.CreateNet(predict_net.Proto()))
        self.assertTrue(workspace.RunNet(predict_net.Proto().name))
        for op in predict_net.Proto().op:
            if op.type == 'RecurrentNetwork':
                for arg in op.arg:
                    if arg.name == 'step_net':
                        for step_op in arg.n.op:
                            self.assertEqual(0, step_op.device_option.device_type)
                            self.assertEqual(1, step_op.device_option.device_id)
                    elif arg.name == 'backward_step_net':
                        self.assertEqual(caffe2_pb2.NetDef(), arg.n)
    
    def test_lstm_params(self):
        model = ModelHelper(name='lstm_params_test')
        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CPU, 0)):
            (output, _, _, _) = rnn_cell.LSTM(model=model, input_blob='input', seq_lengths='seqlengths', initial_states=None, dim_in=20, dim_out=40, scope='test', drop_states=True, return_last_layer_only=True)
        for param in model.GetParams():
            self.assertNotEqual(model.get_param_info(param), None)
    
    def test_milstm_params(self):
        model = ModelHelper(name='milstm_params_test')
        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CPU, 0)):
            (output, _, _, _) = rnn_cell.MILSTM(model=model, input_blob='input', seq_lengths='seqlengths', initial_states=None, dim_in=20, dim_out=[40, 20], scope='test', drop_states=True, return_last_layer_only=True)
        for param in model.GetParams():
            self.assertNotEqual(model.get_param_info(param), None)
    
    def test_layer_norm_lstm_params(self):
        model = ModelHelper(name='layer_norm_lstm_params_test')
        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CPU, 0)):
            (output, _, _, _) = rnn_cell.LayerNormLSTM(model=model, input_blob='input', seq_lengths='seqlengths', initial_states=None, dim_in=20, dim_out=40, scope='test', drop_states=True, return_last_layer_only=True)
        for param in model.GetParams():
            self.assertNotEqual(model.get_param_info(param), None)
    
    @given(encoder_output_length=st.integers(1, 3), encoder_output_dim=st.integers(1, 3), decoder_input_length=st.integers(1, 3), decoder_state_dim=st.integers(1, 3), batch_size=st.integers(1, 3), **hu.gcs)
    def test_lstm_with_regular_attention(self, encoder_output_length, encoder_output_dim, decoder_input_length, decoder_state_dim, batch_size, gc, dc):
        self.lstm_with_attention(partial(rnn_cell.LSTMWithAttention, attention_type=AttentionType.Regular), encoder_output_length, encoder_output_dim, decoder_input_length, decoder_state_dim, batch_size, lstm_with_regular_attention_reference, gc)
    
    @given(encoder_output_length=st.integers(1, 3), encoder_output_dim=st.integers(1, 3), decoder_input_length=st.integers(1, 3), decoder_state_dim=st.integers(1, 3), batch_size=st.integers(1, 3), **hu.gcs)
    def test_lstm_with_recurrent_attention(self, encoder_output_length, encoder_output_dim, decoder_input_length, decoder_state_dim, batch_size, gc, dc):
        self.lstm_with_attention(partial(rnn_cell.LSTMWithAttention, attention_type=AttentionType.Recurrent), encoder_output_length, encoder_output_dim, decoder_input_length, decoder_state_dim, batch_size, lstm_with_recurrent_attention_reference, gc)
    
    @given(encoder_output_length=st.integers(2, 2), encoder_output_dim=st.integers(4, 4), decoder_input_length=st.integers(3, 3), decoder_state_dim=st.integers(4, 4), batch_size=st.integers(5, 5), **hu.gcs)
    def test_lstm_with_dot_attention_same_dim(self, encoder_output_length, encoder_output_dim, decoder_input_length, decoder_state_dim, batch_size, gc, dc):
        self.lstm_with_attention(partial(rnn_cell.LSTMWithAttention, attention_type=AttentionType.Dot), encoder_output_length, encoder_output_dim, decoder_input_length, decoder_state_dim, batch_size, lstm_with_dot_attention_reference_same_dim, gc)
    
    @given(encoder_output_length=st.integers(1, 3), encoder_output_dim=st.integers(4, 4), decoder_input_length=st.integers(1, 3), decoder_state_dim=st.integers(5, 5), batch_size=st.integers(1, 3), **hu.gcs)
    def test_lstm_with_dot_attention_different_dim(self, encoder_output_length, encoder_output_dim, decoder_input_length, decoder_state_dim, batch_size, gc, dc):
        self.lstm_with_attention(partial(rnn_cell.LSTMWithAttention, attention_type=AttentionType.Dot), encoder_output_length, encoder_output_dim, decoder_input_length, decoder_state_dim, batch_size, lstm_with_dot_attention_reference_different_dim, gc)
    
    @given(encoder_output_length=st.integers(2, 3), encoder_output_dim=st.integers(1, 3), decoder_input_length=st.integers(1, 3), decoder_state_dim=st.integers(1, 3), batch_size=st.integers(1, 3), **hu.gcs)
    def test_lstm_with_coverage_attention(self, encoder_output_length, encoder_output_dim, decoder_input_length, decoder_state_dim, batch_size, gc, dc):
        self.lstm_with_attention(partial(rnn_cell.LSTMWithAttention, attention_type=AttentionType.SoftCoverage), encoder_output_length, encoder_output_dim, decoder_input_length, decoder_state_dim, batch_size, lstm_with_coverage_attention_reference, gc)
    
    def lstm_with_attention(self, create_lstm_with_attention, encoder_output_length, encoder_output_dim, decoder_input_length, decoder_state_dim, batch_size, ref, gc):
        model = ModelHelper(name='external')
        with core.DeviceScope(gc):
            (encoder_outputs, decoder_inputs, decoder_input_lengths, initial_decoder_hidden_state, initial_decoder_cell_state, initial_attention_weighted_encoder_context) = model.net.AddExternalInputs('encoder_outputs', 'decoder_inputs', 'decoder_input_lengths', 'initial_decoder_hidden_state', 'initial_decoder_cell_state', 'initial_attention_weighted_encoder_context')
            create_lstm_with_attention(model=model, decoder_inputs=decoder_inputs, decoder_input_lengths=decoder_input_lengths, initial_decoder_hidden_state=initial_decoder_hidden_state, initial_decoder_cell_state=initial_decoder_cell_state, initial_attention_weighted_encoder_context=initial_attention_weighted_encoder_context, encoder_output_dim=encoder_output_dim, encoder_outputs=encoder_outputs, encoder_lengths=None, decoder_input_dim=decoder_state_dim, decoder_state_dim=decoder_state_dim, scope='external/LSTMWithAttention')
            op = model.net._net.op[-2]
        workspace.RunNetOnce(model.param_init_net)
        decoder_input_blob = op.input[0]
        workspace.FeedBlob(decoder_input_blob, np.random.randn(decoder_input_length, batch_size, decoder_state_dim * 4).astype(np.float32))
        workspace.FeedBlob('external/LSTMWithAttention/encoder_outputs_transposed', np.random.randn(batch_size, encoder_output_dim, encoder_output_length).astype(np.float32))
        workspace.FeedBlob('external/LSTMWithAttention/weighted_encoder_outputs', np.random.randn(encoder_output_length, batch_size, encoder_output_dim).astype(np.float32))
        workspace.FeedBlob('external/LSTMWithAttention/coverage_weights', np.random.randn(encoder_output_length, batch_size, encoder_output_dim).astype(np.float32))
        workspace.FeedBlob(decoder_input_lengths, np.random.randint(0, decoder_input_length + 1, size=(batch_size, )).astype(np.int32))
        workspace.FeedBlob(initial_decoder_hidden_state, np.random.randn(1, batch_size, decoder_state_dim).astype(np.float32))
        workspace.FeedBlob(initial_decoder_cell_state, np.random.randn(1, batch_size, decoder_state_dim).astype(np.float32))
        workspace.FeedBlob(initial_attention_weighted_encoder_context, np.random.randn(1, batch_size, encoder_output_dim).astype(np.float32))
        workspace.FeedBlob('external/LSTMWithAttention/initial_coverage', np.zeros((1, batch_size, encoder_output_length)).astype(np.float32))
        inputs = [workspace.FetchBlob(name) for name in op.input]
        self.assertReferenceChecks(device_option=gc, op=op, inputs=inputs, reference=ref, grad_reference=None, output_to_grad=None, outputs_to_check=list(range(6)))
        gradients_to_check = [index for (index, input_name) in enumerate(op.input) if input_name != 'decoder_input_lengths']
        for param in gradients_to_check:
            self.assertGradientChecks(device_option=gc, op=op, inputs=inputs, outputs_to_check=param, outputs_with_grads=[0, 4], threshold=0.01, stepsize=0.001)
    
    @given(seed=st.integers(0, 2**32 - 1), n=st.integers(1, 10), d=st.integers(1, 10), t=st.integers(1, 10), dtype=st.sampled_from([np.float32, np.float16]), use_sequence_lengths=st.booleans(), **hu.gcs)
    def test_lstm_unit_recurrent_network(self, seed, n, d, t, dtype, dc, use_sequence_lengths, gc):
        np.random.seed(seed)
        if dtype == np.float16:
            assume(gc.device_type == workspace.GpuDeviceType)
            dc = [do for do in dc if do.device_type == workspace.GpuDeviceType]
        if use_sequence_lengths:
            op_inputs = ['hidden_t_prev', 'cell_t_prev', 'gates_t', 'seq_lengths', 'timestep']
        else:
            op_inputs = ['hidden_t_prev', 'cell_t_prev', 'gates_t', 'timestep']
        op = core.CreateOperator('LSTMUnit', op_inputs, ['hidden_t', 'cell_t'], sequence_lengths=use_sequence_lengths)
        cell_t_prev = np.random.randn(1, n, d).astype(dtype)
        hidden_t_prev = np.random.randn(1, n, d).astype(dtype)
        gates = np.random.randn(1, n, 4 * d).astype(dtype)
        seq_lengths = np.random.randint(1, t + 1, size=(n, )).astype(np.int32)
        timestep = np.random.randint(0, t, size=(1, )).astype(np.int32)
        if use_sequence_lengths:
            inputs = [hidden_t_prev, cell_t_prev, gates, seq_lengths, timestep]
        else:
            inputs = [hidden_t_prev, cell_t_prev, gates, timestep]
        input_device_options = {'timestep': hu.cpu_do}
        self.assertDeviceChecks(dc, op, inputs, [0], input_device_options=input_device_options)
        kwargs = {}
        if dtype == np.float16:
            kwargs['threshold'] = 0.1
        
        def lstm_unit_reference(*args, **kwargs):
            return lstm_unit(*args, sequence_lengths=use_sequence_lengths, **kwargs)
        self.assertReferenceChecks(gc, op, inputs, lstm_unit_reference, input_device_options=input_device_options, **kwargs)
        kwargs = {}
        if dtype == np.float16:
            kwargs['threshold'] = 0.5
        for i in range(2):
            self.assertGradientChecks(gc, op, inputs, i, [0, 1], input_device_options=input_device_options, **kwargs)
    
    @given(input_length=st.integers(2, 5), dim_in=st.integers(1, 3), max_num_units=st.integers(1, 3), num_layers=st.integers(2, 3), batch_size=st.integers(1, 3))
    def test_multi_lstm(self, input_length, dim_in, max_num_units, num_layers, batch_size):
        model = ModelHelper(name='external')
        (input_sequence, seq_lengths) = model.net.AddExternalInputs('input_sequence', 'seq_lengths')
        dim_out = [np.random.randint(1, max_num_units + 1) for _ in range(num_layers)]
        (h_all, h_last, c_all, c_last) = rnn_cell.LSTM(model=model, input_blob=input_sequence, seq_lengths=seq_lengths, initial_states=None, dim_in=dim_in, dim_out=dim_out, outputs_with_grads=(0, ), return_params=False, memory_optimization=False, forget_bias=0.0, forward_only=False, return_last_layer_only=True)
        workspace.RunNetOnce(model.param_init_net)
        seq_lengths_val = np.random.randint(1, input_length + 1, size=batch_size).astype(np.int32)
        input_sequence_val = np.random.randn(input_length, batch_size, dim_in).astype(np.float32)
        workspace.FeedBlob(seq_lengths, seq_lengths_val)
        workspace.FeedBlob(input_sequence, input_sequence_val)
        hidden_input_list = []
        cell_input_list = []
        i2h_w_list = []
        i2h_b_list = []
        gates_w_list = []
        gates_b_list = []
        for i in range(num_layers):
            hidden_input_list.append(workspace.FetchBlob('layer_{}/initial_hidden_state'.format(i)))
            cell_input_list.append(workspace.FetchBlob('layer_{}/initial_cell_state'.format(i)))
            prefix = ('layer_{}/'.format(i) if i > 0 else '')
            i2h_w_list.append(workspace.FetchBlob('{}i2h_w'.format(prefix)))
            i2h_b_list.append(workspace.FetchBlob('{}i2h_b'.format(prefix)))
            gates_w_list.append(workspace.FetchBlob('layer_{}/gates_t_w'.format(i)))
            gates_b_list.append(workspace.FetchBlob('layer_{}/gates_t_b'.format(i)))
        workspace.RunNetOnce(model.net)
        h_all_calc = workspace.FetchBlob(h_all)
        h_last_calc = workspace.FetchBlob(h_last)
        c_all_calc = workspace.FetchBlob(c_all)
        c_last_calc = workspace.FetchBlob(c_last)
        (h_all_ref, h_last_ref, c_all_ref, c_last_ref) = multi_lstm_reference(input_sequence_val, hidden_input_list, cell_input_list, i2h_w_list, i2h_b_list, gates_w_list, gates_b_list, seq_lengths_val, forget_bias=0.0)
        h_all_delta = np.abs(h_all_ref - h_all_calc).sum()
        h_last_delta = np.abs(h_last_ref - h_last_calc).sum()
        c_all_delta = np.abs(c_all_ref - c_all_calc).sum()
        c_last_delta = np.abs(c_last_ref - c_last_calc).sum()
        self.assertAlmostEqual(h_all_delta, 0.0, places=5)
        self.assertAlmostEqual(h_last_delta, 0.0, places=5)
        self.assertAlmostEqual(c_all_delta, 0.0, places=5)
        self.assertAlmostEqual(c_last_delta, 0.0, places=5)
        input_values = {'input_sequence': input_sequence_val, 'seq_lengths': seq_lengths_val}
        for param in model.GetParams():
            value = workspace.FetchBlob(param)
            input_values[str(param)] = value
        output_sum = model.net.SumElements([h_all], 'output_sum', average=True)
        fake_loss = model.net.Tanh(output_sum)
        for param in model.GetParams():
            gradient_checker.NetGradientChecker.Check(model.net, outputs_with_grad=[fake_loss], input_values=input_values, input_to_check=str(param), print_net=False, step_size=0.0001, threshold=0.05)

if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    unittest.main()

