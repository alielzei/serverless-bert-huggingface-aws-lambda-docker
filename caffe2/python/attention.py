from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import brew


class AttentionType:
    (Regular, Recurrent, Dot, SoftCoverage) = tuple(range(4))


def s(scope, name):
    return '{}/{}'.format(str(scope), str(name))

def _calc_weighted_context(model, encoder_outputs_transposed, encoder_output_dim, attention_weights_3d, scope):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.attention._calc_weighted_context', '_calc_weighted_context(model, encoder_outputs_transposed, encoder_output_dim, attention_weights_3d, scope)', {'brew': brew, 's': s, 'model': model, 'encoder_outputs_transposed': encoder_outputs_transposed, 'encoder_output_dim': encoder_output_dim, 'attention_weights_3d': attention_weights_3d, 'scope': scope}, 1)

def _calc_attention_weights(model, attention_logits_transposed, scope, encoder_lengths=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.attention._calc_attention_weights', '_calc_attention_weights(model, attention_logits_transposed, scope, encoder_lengths=None)', {'brew': brew, 's': s, 'model': model, 'attention_logits_transposed': attention_logits_transposed, 'scope': scope, 'encoder_lengths': encoder_lengths}, 1)

def _calc_attention_logits_from_sum_match(model, decoder_hidden_encoder_outputs_sum, encoder_output_dim, scope):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.attention._calc_attention_logits_from_sum_match', '_calc_attention_logits_from_sum_match(model, decoder_hidden_encoder_outputs_sum, encoder_output_dim, scope)', {'brew': brew, 's': s, 'model': model, 'decoder_hidden_encoder_outputs_sum': decoder_hidden_encoder_outputs_sum, 'encoder_output_dim': encoder_output_dim, 'scope': scope}, 1)

def _apply_fc_weight_for_sum_match(model, input, dim_in, dim_out, scope, name):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.attention._apply_fc_weight_for_sum_match', '_apply_fc_weight_for_sum_match(model, input, dim_in, dim_out, scope, name)', {'brew': brew, 's': s, 'model': model, 'input': input, 'dim_in': dim_in, 'dim_out': dim_out, 'scope': scope, 'name': name}, 1)

def apply_recurrent_attention(model, encoder_output_dim, encoder_outputs_transposed, weighted_encoder_outputs, decoder_hidden_state_t, decoder_hidden_state_dim, attention_weighted_encoder_context_t_prev, scope, encoder_lengths=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.attention.apply_recurrent_attention', 'apply_recurrent_attention(model, encoder_output_dim, encoder_outputs_transposed, weighted_encoder_outputs, decoder_hidden_state_t, decoder_hidden_state_dim, attention_weighted_encoder_context_t_prev, scope, encoder_lengths=None)', {'_apply_fc_weight_for_sum_match': _apply_fc_weight_for_sum_match, 's': s, '_calc_attention_logits_from_sum_match': _calc_attention_logits_from_sum_match, '_calc_attention_weights': _calc_attention_weights, '_calc_weighted_context': _calc_weighted_context, 'model': model, 'encoder_output_dim': encoder_output_dim, 'encoder_outputs_transposed': encoder_outputs_transposed, 'weighted_encoder_outputs': weighted_encoder_outputs, 'decoder_hidden_state_t': decoder_hidden_state_t, 'decoder_hidden_state_dim': decoder_hidden_state_dim, 'attention_weighted_encoder_context_t_prev': attention_weighted_encoder_context_t_prev, 'scope': scope, 'encoder_lengths': encoder_lengths}, 3)

def apply_regular_attention(model, encoder_output_dim, encoder_outputs_transposed, weighted_encoder_outputs, decoder_hidden_state_t, decoder_hidden_state_dim, scope, encoder_lengths=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.attention.apply_regular_attention', 'apply_regular_attention(model, encoder_output_dim, encoder_outputs_transposed, weighted_encoder_outputs, decoder_hidden_state_t, decoder_hidden_state_dim, scope, encoder_lengths=None)', {'_apply_fc_weight_for_sum_match': _apply_fc_weight_for_sum_match, 's': s, '_calc_attention_logits_from_sum_match': _calc_attention_logits_from_sum_match, '_calc_attention_weights': _calc_attention_weights, '_calc_weighted_context': _calc_weighted_context, 'model': model, 'encoder_output_dim': encoder_output_dim, 'encoder_outputs_transposed': encoder_outputs_transposed, 'weighted_encoder_outputs': weighted_encoder_outputs, 'decoder_hidden_state_t': decoder_hidden_state_t, 'decoder_hidden_state_dim': decoder_hidden_state_dim, 'scope': scope, 'encoder_lengths': encoder_lengths}, 3)

def apply_dot_attention(model, encoder_output_dim, encoder_outputs_transposed, decoder_hidden_state_t, decoder_hidden_state_dim, scope, encoder_lengths=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.attention.apply_dot_attention', 'apply_dot_attention(model, encoder_output_dim, encoder_outputs_transposed, decoder_hidden_state_t, decoder_hidden_state_dim, scope, encoder_lengths=None)', {'brew': brew, 's': s, '_calc_attention_weights': _calc_attention_weights, '_calc_weighted_context': _calc_weighted_context, 'model': model, 'encoder_output_dim': encoder_output_dim, 'encoder_outputs_transposed': encoder_outputs_transposed, 'decoder_hidden_state_t': decoder_hidden_state_t, 'decoder_hidden_state_dim': decoder_hidden_state_dim, 'scope': scope, 'encoder_lengths': encoder_lengths}, 3)

def apply_soft_coverage_attention(model, encoder_output_dim, encoder_outputs_transposed, weighted_encoder_outputs, decoder_hidden_state_t, decoder_hidden_state_dim, scope, encoder_lengths, coverage_t_prev, coverage_weights):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.attention.apply_soft_coverage_attention', 'apply_soft_coverage_attention(model, encoder_output_dim, encoder_outputs_transposed, weighted_encoder_outputs, decoder_hidden_state_t, decoder_hidden_state_dim, scope, encoder_lengths, coverage_t_prev, coverage_weights)', {'_apply_fc_weight_for_sum_match': _apply_fc_weight_for_sum_match, 's': s, 'brew': brew, '_calc_attention_logits_from_sum_match': _calc_attention_logits_from_sum_match, '_calc_attention_weights': _calc_attention_weights, '_calc_weighted_context': _calc_weighted_context, 'model': model, 'encoder_output_dim': encoder_output_dim, 'encoder_outputs_transposed': encoder_outputs_transposed, 'weighted_encoder_outputs': weighted_encoder_outputs, 'decoder_hidden_state_t': decoder_hidden_state_t, 'decoder_hidden_state_dim': decoder_hidden_state_dim, 'scope': scope, 'encoder_lengths': encoder_lengths, 'coverage_t_prev': coverage_t_prev, 'coverage_weights': coverage_weights}, 4)

