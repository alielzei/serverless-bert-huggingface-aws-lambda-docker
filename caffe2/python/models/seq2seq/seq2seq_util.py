""" A bunch of util functions to build Seq2Seq models with Caffe2."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import collections
from future.utils import viewitems
import caffe2.proto.caffe2_pb2 as caffe2_pb2
from caffe2.python import attention, core, rnn_cell, brew
PAD_ID = 0
PAD = '<PAD>'
GO_ID = 1
GO = '<GO>'
EOS_ID = 2
EOS = '<EOS>'
UNK_ID = 3
UNK = '<UNK>'

def gen_vocab(corpus, unk_threshold):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.models.seq2seq.seq2seq_util.gen_vocab', 'gen_vocab(corpus, unk_threshold)', {'collections': collections, 'PAD': PAD, 'GO': GO, 'EOS': EOS, 'UNK': UNK, 'viewitems': viewitems, 'corpus': corpus, 'unk_threshold': unk_threshold}, 1)

def get_numberized_sentence(sentence, vocab):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.models.seq2seq.seq2seq_util.get_numberized_sentence', 'get_numberized_sentence(sentence, vocab)', {'UNK': UNK, 'sentence': sentence, 'vocab': vocab}, 1)

def rnn_unidirectional_layer(model, inputs, input_lengths, input_size, num_units, dropout_keep_prob, forward_only, return_sequence_output, return_final_state, scope=None):
    """ Unidirectional LSTM encoder."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.models.seq2seq.seq2seq_util.rnn_unidirectional_layer', 'rnn_unidirectional_layer(model, inputs, input_lengths, input_size, num_units, dropout_keep_prob, forward_only, return_sequence_output, return_final_state, scope=None)', {'core': core, 'rnn_cell': rnn_cell, 'model': model, 'inputs': inputs, 'input_lengths': input_lengths, 'input_size': input_size, 'num_units': num_units, 'dropout_keep_prob': dropout_keep_prob, 'forward_only': forward_only, 'return_sequence_output': return_sequence_output, 'return_final_state': return_final_state, 'scope': scope}, 3)

def rnn_bidirectional_layer(model, inputs, input_lengths, input_size, num_units, dropout_keep_prob, forward_only, return_sequence_output, return_final_state, scope=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.models.seq2seq.seq2seq_util.rnn_bidirectional_layer', 'rnn_bidirectional_layer(model, inputs, input_lengths, input_size, num_units, dropout_keep_prob, forward_only, return_sequence_output, return_final_state, scope=None)', {'rnn_unidirectional_layer': rnn_unidirectional_layer, 'core': core, 'model': model, 'inputs': inputs, 'input_lengths': input_lengths, 'input_size': input_size, 'num_units': num_units, 'dropout_keep_prob': dropout_keep_prob, 'forward_only': forward_only, 'return_sequence_output': return_sequence_output, 'return_final_state': return_final_state, 'scope': scope}, 3)

def build_embeddings(model, vocab_size, embedding_size, name, freeze_embeddings):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.models.seq2seq.seq2seq_util.build_embeddings', 'build_embeddings(model, vocab_size, embedding_size, name, freeze_embeddings)', {'model': model, 'vocab_size': vocab_size, 'embedding_size': embedding_size, 'name': name, 'freeze_embeddings': freeze_embeddings}, 1)

def get_layer_scope(scope, layer_type, i):
    prefix = ((scope + '/' if scope else '')) + layer_type
    return '{}/layer{}'.format(prefix, i)

def build_embedding_encoder(model, encoder_params, num_decoder_layers, inputs, input_lengths, vocab_size, embeddings, embedding_size, use_attention, num_gpus=0, forward_only=False, scope=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.models.seq2seq.seq2seq_util.build_embedding_encoder', 'build_embedding_encoder(model, encoder_params, num_decoder_layers, inputs, input_lengths, vocab_size, embeddings, embedding_size, use_attention, num_gpus=0, forward_only=False, scope=None)', {'core': core, 'caffe2_pb2': caffe2_pb2, 'rnn_bidirectional_layer': rnn_bidirectional_layer, 'rnn_unidirectional_layer': rnn_unidirectional_layer, 'get_layer_scope': get_layer_scope, 'model': model, 'encoder_params': encoder_params, 'num_decoder_layers': num_decoder_layers, 'inputs': inputs, 'input_lengths': input_lengths, 'vocab_size': vocab_size, 'embeddings': embeddings, 'embedding_size': embedding_size, 'use_attention': use_attention, 'num_gpus': num_gpus, 'forward_only': forward_only, 'scope': scope}, 5)


class LSTMWithAttentionDecoder(object):
    
    def scope(self, name):
        return (self.name + '/' + name if self.name is not None else name)
    
    def _get_attention_type(self, attention_type_as_string):
        if attention_type_as_string == 'regular':
            return attention.AttentionType.Regular
        elif attention_type_as_string == 'recurrent':
            return attention.AttentionType.Recurrent
        else:
            assert False, 'Unknown type ' + attention_type_as_string
    
    def __init__(self, encoder_outputs, encoder_output_dim, encoder_lengths, vocab_size, attention_type, embedding_size, decoder_num_units, decoder_cells, residual_output_layers=None, name=None, weighted_encoder_outputs=None):
        self.name = name
        self.num_layers = len(decoder_cells)
        if attention_type == 'none':
            self.cell = rnn_cell.MultiRNNCell(decoder_cells, name=self.scope('decoder'), residual_output_layers=residual_output_layers)
            self.use_attention = False
            self.decoder_output_dim = decoder_num_units
            self.output_indices = self.cell.output_indices
        else:
            decoder_cell = rnn_cell.MultiRNNCell(decoder_cells, name=self.scope('decoder'), residual_output_layers=residual_output_layers)
            self.cell = rnn_cell.AttentionCell(encoder_output_dim=encoder_output_dim, encoder_outputs=encoder_outputs, encoder_lengths=encoder_lengths, decoder_cell=decoder_cell, decoder_state_dim=decoder_num_units, name=self.scope('attention_decoder'), attention_type=self._get_attention_type(attention_type), weighted_encoder_outputs=weighted_encoder_outputs, attention_memory_optimization=True)
            self.use_attention = True
            self.decoder_output_dim = decoder_num_units + encoder_output_dim
            self.output_indices = decoder_cell.output_indices
            self.output_indices.append(2 * self.num_layers)
    
    def get_state_names(self):
        return self.cell.get_state_names()
    
    def get_outputs_with_grads(self):
        return [2 * i for i in self.output_indices]
    
    def get_output_dim(self):
        return self.decoder_output_dim
    
    def get_attention_weights(self):
        assert self.use_attention
        return self.cell.get_attention_weights()
    
    def apply(self, model, input_t, seq_lengths, states, timestep):
        return self.cell.apply(model=model, input_t=input_t, seq_lengths=seq_lengths, states=states, timestep=timestep)
    
    def apply_over_sequence(self, model, inputs, seq_lengths, initial_states):
        return self.cell.apply_over_sequence(model=model, inputs=inputs, seq_lengths=seq_lengths, initial_states=initial_states, outputs_with_grads=self.get_outputs_with_grads())


def build_initial_rnn_decoder_states(model, encoder_units_per_layer, decoder_units_per_layer, final_encoder_hidden_states, final_encoder_cell_states, use_attention):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.models.seq2seq.seq2seq_util.build_initial_rnn_decoder_states', 'build_initial_rnn_decoder_states(model, encoder_units_per_layer, decoder_units_per_layer, final_encoder_hidden_states, final_encoder_cell_states, use_attention)', {'brew': brew, 'model': model, 'encoder_units_per_layer': encoder_units_per_layer, 'decoder_units_per_layer': decoder_units_per_layer, 'final_encoder_hidden_states': final_encoder_hidden_states, 'final_encoder_cell_states': final_encoder_cell_states, 'use_attention': use_attention}, 1)

def build_embedding_decoder(model, decoder_layer_configs, inputs, input_lengths, encoder_lengths, encoder_outputs, weighted_encoder_outputs, final_encoder_hidden_states, final_encoder_cell_states, encoder_units_per_layer, vocab_size, embeddings, embedding_size, attention_type, forward_only, num_gpus=0, scope=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.models.seq2seq.seq2seq_util.build_embedding_decoder', 'build_embedding_decoder(model, decoder_layer_configs, inputs, input_lengths, encoder_lengths, encoder_outputs, weighted_encoder_outputs, final_encoder_hidden_states, final_encoder_cell_states, encoder_units_per_layer, vocab_size, embeddings, embedding_size, attention_type, forward_only, num_gpus=0, scope=None)', {'core': core, 'caffe2_pb2': caffe2_pb2, 'rnn_cell': rnn_cell, 'get_layer_scope': get_layer_scope, 'build_initial_rnn_decoder_states': build_initial_rnn_decoder_states, 'LSTMWithAttentionDecoder': LSTMWithAttentionDecoder, 'model': model, 'decoder_layer_configs': decoder_layer_configs, 'inputs': inputs, 'input_lengths': input_lengths, 'encoder_lengths': encoder_lengths, 'encoder_outputs': encoder_outputs, 'weighted_encoder_outputs': weighted_encoder_outputs, 'final_encoder_hidden_states': final_encoder_hidden_states, 'final_encoder_cell_states': final_encoder_cell_states, 'encoder_units_per_layer': encoder_units_per_layer, 'vocab_size': vocab_size, 'embeddings': embeddings, 'embedding_size': embedding_size, 'attention_type': attention_type, 'forward_only': forward_only, 'num_gpus': num_gpus, 'scope': scope}, 2)

def output_projection(model, decoder_outputs, decoder_output_size, target_vocab_size, decoder_softmax_size):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.models.seq2seq.seq2seq_util.output_projection', 'output_projection(model, decoder_outputs, decoder_output_size, target_vocab_size, decoder_softmax_size)', {'brew': brew, 'model': model, 'decoder_outputs': decoder_outputs, 'decoder_output_size': decoder_output_size, 'target_vocab_size': target_vocab_size, 'decoder_softmax_size': decoder_softmax_size}, 1)

