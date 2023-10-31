"""PyTorch Longformer model. """

import math
import warnings
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn import functional as F
from .activations import ACT2FN, gelu
from .configuration_longformer import LongformerConfig
from .file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_callable, replace_return_docstrings
from .modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, MaskedLMOutput, MultipleChoiceModelOutput, QuestionAnsweringModelOutput, SequenceClassifierOutput, TokenClassifierOutput
from .modeling_utils import PreTrainedModel, apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from .utils import logging
logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = 'LongformerConfig'
_TOKENIZER_FOR_DOC = 'LongformerTokenizer'
LONGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = ['allenai/longformer-base-4096', 'allenai/longformer-large-4096', 'allenai/longformer-large-4096-finetuned-triviaqa', 'allenai/longformer-base-4096-extra.pos.embd.only', 'allenai/longformer-large-4096-extra.pos.embd.only']

def _get_question_end_index(input_ids, sep_token_id):
    """
    Computes the index of the first occurance of `sep_token_id`.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.modeling_longformer._get_question_end_index', '_get_question_end_index(input_ids, sep_token_id)', {'input_ids': input_ids, 'sep_token_id': sep_token_id}, 1)

def _compute_global_attention_mask(input_ids, sep_token_id, before_sep_token=True):
    """
    Computes global attention mask by putting attention on all tokens
    before `sep_token_id` if `before_sep_token is True` else after
    `sep_token_id`.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.modeling_longformer._compute_global_attention_mask', '_compute_global_attention_mask(input_ids, sep_token_id, before_sep_token=True)', {'_get_question_end_index': _get_question_end_index, 'torch': torch, 'input_ids': input_ids, 'sep_token_id': sep_token_id, 'before_sep_token': before_sep_token}, 1)

def create_position_ids_from_input_ids(input_ids, padding_idx):
    """Replace non-padding symbols with their position numbers. Position numbers begin at
    padding_idx+1. Padding symbols are ignored. This is modified from fairseq's
    `utils.make_positions`.

    :param torch.Tensor x:
    :return torch.Tensor:
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.modeling_longformer.create_position_ids_from_input_ids', 'create_position_ids_from_input_ids(input_ids, padding_idx)', {'torch': torch, 'input_ids': input_ids, 'padding_idx': padding_idx}, 1)


class LongformerEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """
    
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer('position_ids', torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx)
    
    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if position_ids is None:
            if input_ids is not None:
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx).to(input_ids.device)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    
    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """We are provided embeddings directly. We cannot infer which are padded so just generate
        sequential position ids.

        :param torch.Tensor inputs_embeds:
        :return torch.Tensor:
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]
        position_ids = torch.arange(self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device)
        return position_ids.unsqueeze(0).expand(input_shape)



class LongformerSelfAttention(nn.Module):
    
    def __init__(self, config, layer_id):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError('The hidden size (%d) is not a multiple of the number of attention heads (%d)' % (config.hidden_size, config.num_attention_heads))
        self.num_heads = config.num_attention_heads
        self.head_dim = int(config.hidden_size / config.num_attention_heads)
        self.embed_dim = config.hidden_size
        self.query = nn.Linear(config.hidden_size, self.embed_dim)
        self.key = nn.Linear(config.hidden_size, self.embed_dim)
        self.value = nn.Linear(config.hidden_size, self.embed_dim)
        self.query_global = nn.Linear(config.hidden_size, self.embed_dim)
        self.key_global = nn.Linear(config.hidden_size, self.embed_dim)
        self.value_global = nn.Linear(config.hidden_size, self.embed_dim)
        self.dropout = config.attention_probs_dropout_prob
        self.layer_id = layer_id
        attention_window = config.attention_window[self.layer_id]
        assert attention_window % 2 == 0, f'`attention_window` for layer {self.layer_id} has to be an even value. Given {attention_window}'
        assert attention_window > 0, f'`attention_window` for layer {self.layer_id} has to be positive. Given {attention_window}'
        self.one_sided_attn_window_size = attention_window // 2
    
    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        """
        LongformerSelfAttention expects `len(hidden_states)` to be multiple of `attention_window`.
        Padding to `attention_window` happens in LongformerModel.forward to avoid redoing the padding on each layer.

        The `attention_mask` is changed in `BertModel.forward` from 0, 1, 2 to
            -ve: no attention
              0: local attention
            +ve: global attention

        """
        attention_mask = attention_mask.squeeze(dim=2).squeeze(dim=1)
        is_index_masked = attention_mask < 0
        is_index_global_attn = attention_mask > 0
        is_global_attn = is_index_global_attn.flatten().any().item()
        hidden_states = hidden_states.transpose(0, 1)
        query_vectors = self.query(hidden_states)
        key_vectors = self.key(hidden_states)
        value_vectors = self.value(hidden_states)
        (seq_len, batch_size, embed_dim) = hidden_states.size()
        assert embed_dim == self.embed_dim, f'hidden_states should have embed_dim = {self.embed_dim}, but has {embed_dim}'
        query_vectors /= math.sqrt(self.head_dim)
        query_vectors = query_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        attn_scores = self._sliding_chunks_query_key_matmul(query_vectors, key_vectors, self.one_sided_attn_window_size)
        remove_from_windowed_attention_mask = (attention_mask != 0)[:, :, None, None]
        float_mask = remove_from_windowed_attention_mask.type_as(query_vectors).masked_fill(remove_from_windowed_attention_mask, -10000.0)
        diagonal_mask = self._sliding_chunks_query_key_matmul(float_mask.new_ones(size=float_mask.size()), float_mask, self.one_sided_attn_window_size)
        attn_scores += diagonal_mask
        assert list(attn_scores.size()) == [batch_size, seq_len, self.num_heads, self.one_sided_attn_window_size * 2 + 1], f'attn_probs should be of size ({batch_size}, {seq_len}, {self.num_heads}, {self.one_sided_attn_window_size * 2 + 1}), but is of size {attn_scores.size()}'
        if is_global_attn:
            (max_num_global_attn_indices, is_index_global_attn_nonzero, is_local_index_global_attn_nonzero, is_local_index_no_global_attn_nonzero) = self._get_global_attn_indices(is_index_global_attn)
            global_key_attn_scores = self._concat_with_global_key_attn_probs(query_vectors=query_vectors, key_vectors=key_vectors, max_num_global_attn_indices=max_num_global_attn_indices, is_index_global_attn_nonzero=is_index_global_attn_nonzero, is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero, is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero)
            attn_scores = torch.cat((global_key_attn_scores, attn_scores), dim=-1)
            del global_key_attn_scores
        attn_probs_fp32 = F.softmax(attn_scores, dim=-1, dtype=torch.float32)
        attn_probs = attn_probs_fp32.type_as(attn_scores)
        del attn_probs_fp32
        attn_probs = torch.masked_fill(attn_probs, is_index_masked[:, :, None, None], 0.0)
        attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)
        value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1)
        if is_global_attn:
            attn_output = self._compute_attn_output_with_global_indices(value_vectors=value_vectors, attn_probs=attn_probs, max_num_global_attn_indices=max_num_global_attn_indices, is_index_global_attn_nonzero=is_index_global_attn_nonzero, is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero)
        else:
            attn_output = self._sliding_chunks_matmul_attn_probs_value(attn_probs, value_vectors, self.one_sided_attn_window_size)
        assert attn_output.size() == (batch_size, seq_len, self.num_heads, self.head_dim), 'Unexpected size'
        attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        if is_global_attn:
            global_attn_output = self._compute_global_attn_output_from_hidden(hidden_states=hidden_states, max_num_global_attn_indices=max_num_global_attn_indices, is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero, is_index_global_attn_nonzero=is_index_global_attn_nonzero, is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero, is_index_masked=is_index_masked)
            nonzero_global_attn_output = global_attn_output[is_local_index_global_attn_nonzero[0], :, is_local_index_global_attn_nonzero[1]]
            attn_output[is_index_global_attn_nonzero[::-1]] = nonzero_global_attn_output.view(len(is_local_index_global_attn_nonzero[0]), -1)
        attn_output = attn_output.transpose(0, 1)
        if output_attentions:
            if is_global_attn:
                attn_probs = attn_probs.view(batch_size, self.num_heads, max_num_global_attn_indices, seq_len)
            else:
                attn_probs = attn_probs.permute(0, 2, 1, 3)
        outputs = ((attn_output, attn_probs) if output_attentions else (attn_output, ))
        return outputs
    
    @staticmethod
    def _pad_and_transpose_last_two_dims(hidden_states_padded, padding):
        """pads rows and then flips rows and columns"""
        hidden_states_padded = F.pad(hidden_states_padded, padding)
        hidden_states_padded = hidden_states_padded.view(*hidden_states_padded.size()[:-2], hidden_states_padded.size(-1), hidden_states_padded.size(-2))
        return hidden_states_padded
    
    @staticmethod
    def _pad_and_diagonalize(chunked_hidden_states):
        """shift every row 1 step right, converting columns into diagonals.
        Example:
              chunked_hidden_states: [ 0.4983,  2.6918, -0.0071,  1.0492,
                                       -1.8348,  0.7672,  0.2986,  0.0285,
                                       -0.7584,  0.4206, -0.0405,  0.1599,
                                       2.0514, -1.1600,  0.5372,  0.2629 ]
              window_overlap = num_rows = 4
             (pad & diagonilize) =>
             [ 0.4983,  2.6918, -0.0071,  1.0492, 0.0000,  0.0000,  0.0000
               0.0000,  -1.8348,  0.7672,  0.2986,  0.0285, 0.0000,  0.0000
               0.0000,  0.0000, -0.7584,  0.4206, -0.0405,  0.1599, 0.0000
               0.0000,  0.0000,  0.0000, 2.0514, -1.1600,  0.5372,  0.2629 ]
        """
        (total_num_heads, num_chunks, window_overlap, hidden_dim) = chunked_hidden_states.size()
        chunked_hidden_states = F.pad(chunked_hidden_states, (0, window_overlap + 1))
        chunked_hidden_states = chunked_hidden_states.view(total_num_heads, num_chunks, -1)
        chunked_hidden_states = chunked_hidden_states[:, :, :-window_overlap]
        chunked_hidden_states = chunked_hidden_states.view(total_num_heads, num_chunks, window_overlap, window_overlap + hidden_dim)
        chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        return chunked_hidden_states
    
    @staticmethod
    def _chunk(hidden_states, window_overlap):
        """convert into overlapping chunkings. Chunk size = 2w, overlap size = w"""
        hidden_states = hidden_states.view(hidden_states.size(0), hidden_states.size(1) // (window_overlap * 2), window_overlap * 2, hidden_states.size(2))
        chunk_size = list(hidden_states.size())
        chunk_size[1] = chunk_size[1] * 2 - 1
        chunk_stride = list(hidden_states.stride())
        chunk_stride[1] = chunk_stride[1] // 2
        return hidden_states.as_strided(size=chunk_size, stride=chunk_stride)
    
    @staticmethod
    def _mask_invalid_locations(input_tensor, affected_seq_len) -> torch.Tensor:
        beginning_mask_2d = input_tensor.new_ones(affected_seq_len, affected_seq_len + 1).tril().flip(dims=[0])
        beginning_mask = beginning_mask_2d[None, :, None, :]
        ending_mask = beginning_mask.flip(dims=(1, 3))
        beginning_input = input_tensor[:, :affected_seq_len, :, :affected_seq_len + 1]
        beginning_mask = beginning_mask.expand(beginning_input.size())
        beginning_input.masked_fill_(beginning_mask == 1, -float('inf'))
        ending_input = input_tensor[:, -affected_seq_len:, :, -(affected_seq_len + 1):]
        ending_mask = ending_mask.expand(ending_input.size())
        ending_input.masked_fill_(ending_mask == 1, -float('inf'))
    
    def _sliding_chunks_query_key_matmul(self, query: torch.Tensor, key: torch.Tensor, window_overlap: int):
        """Matrix multiplication of query and key tensors using with a sliding window attention pattern.
        This implementation splits the input into overlapping chunks of size 2w (e.g. 512 for pretrained Longformer)
        with an overlap of size window_overlap"""
        (batch_size, seq_len, num_heads, head_dim) = query.size()
        assert seq_len % (window_overlap * 2) == 0, f'Sequence length should be multiple of {window_overlap * 2}. Given {seq_len}'
        assert query.size() == key.size()
        chunks_count = seq_len // window_overlap - 1
        query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        chunked_query = self._chunk(query, window_overlap)
        chunked_key = self._chunk(key, window_overlap)
        chunked_attention_scores = torch.einsum('bcxd,bcyd->bcxy', (chunked_query, chunked_key))
        diagonal_chunked_attention_scores = self._pad_and_transpose_last_two_dims(chunked_attention_scores, padding=(0, 0, 0, 1))
        diagonal_attention_scores = diagonal_chunked_attention_scores.new_empty((batch_size * num_heads, chunks_count + 1, window_overlap, window_overlap * 2 + 1))
        diagonal_attention_scores[:, :-1, :, window_overlap:] = diagonal_chunked_attention_scores[:, :, :window_overlap, :window_overlap + 1]
        diagonal_attention_scores[:, -1, :, window_overlap:] = diagonal_chunked_attention_scores[:, -1, window_overlap:, :window_overlap + 1]
        diagonal_attention_scores[:, 1:, :, :window_overlap] = diagonal_chunked_attention_scores[:, :, -(window_overlap + 1):-1, window_overlap + 1:]
        diagonal_attention_scores[:, 0, 1:window_overlap, 1:window_overlap] = diagonal_chunked_attention_scores[:, 0, :window_overlap - 1, 1 - window_overlap:]
        diagonal_attention_scores = diagonal_attention_scores.view(batch_size, num_heads, seq_len, 2 * window_overlap + 1).transpose(2, 1)
        self._mask_invalid_locations(diagonal_attention_scores, window_overlap)
        return diagonal_attention_scores
    
    def _sliding_chunks_matmul_attn_probs_value(self, attn_probs: torch.Tensor, value: torch.Tensor, window_overlap: int):
        """Same as _sliding_chunks_query_key_matmul but for attn_probs and value tensors.
        Returned tensor will be of the same shape as `attn_probs`"""
        (batch_size, seq_len, num_heads, head_dim) = value.size()
        assert seq_len % (window_overlap * 2) == 0
        assert attn_probs.size()[:3] == value.size()[:3]
        assert attn_probs.size(3) == 2 * window_overlap + 1
        chunks_count = seq_len // window_overlap - 1
        chunked_attn_probs = attn_probs.transpose(1, 2).reshape(batch_size * num_heads, seq_len // window_overlap, window_overlap, 2 * window_overlap + 1)
        value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len, head_dim)
        padded_value = F.pad(value, (0, 0, window_overlap, window_overlap), value=-1)
        chunked_value_size = (batch_size * num_heads, chunks_count + 1, 3 * window_overlap, head_dim)
        chunked_value_stride = padded_value.stride()
        chunked_value_stride = (chunked_value_stride[0], window_overlap * chunked_value_stride[1], chunked_value_stride[1], chunked_value_stride[2])
        chunked_value = padded_value.as_strided(size=chunked_value_size, stride=chunked_value_stride)
        chunked_attn_probs = self._pad_and_diagonalize(chunked_attn_probs)
        context = torch.einsum('bcwd,bcdh->bcwh', (chunked_attn_probs, chunked_value))
        return context.view(batch_size, num_heads, seq_len, head_dim).transpose(1, 2)
    
    @staticmethod
    def _get_global_attn_indices(is_index_global_attn):
        """ compute global attn indices required throughout forward pass """
        num_global_attn_indices = is_index_global_attn.long().sum(dim=1)
        max_num_global_attn_indices = num_global_attn_indices.max()
        is_index_global_attn_nonzero = is_index_global_attn.nonzero(as_tuple=True)
        is_local_index_global_attn = torch.arange(max_num_global_attn_indices, device=is_index_global_attn.device) < num_global_attn_indices.unsqueeze(dim=-1)
        is_local_index_global_attn_nonzero = is_local_index_global_attn.nonzero(as_tuple=True)
        is_local_index_no_global_attn_nonzero = (is_local_index_global_attn == 0).nonzero(as_tuple=True)
        return (max_num_global_attn_indices, is_index_global_attn_nonzero, is_local_index_global_attn_nonzero, is_local_index_no_global_attn_nonzero)
    
    def _concat_with_global_key_attn_probs(self, key_vectors, query_vectors, max_num_global_attn_indices, is_index_global_attn_nonzero, is_local_index_global_attn_nonzero, is_local_index_no_global_attn_nonzero):
        batch_size = key_vectors.shape[0]
        key_vectors_only_global = key_vectors.new_zeros(batch_size, max_num_global_attn_indices, self.num_heads, self.head_dim)
        key_vectors_only_global[is_local_index_global_attn_nonzero] = key_vectors[is_index_global_attn_nonzero]
        attn_probs_from_global_key = torch.einsum('blhd,bshd->blhs', (query_vectors, key_vectors_only_global))
        attn_probs_from_global_key[is_local_index_no_global_attn_nonzero[0], :, :, is_local_index_no_global_attn_nonzero[1]] = -10000.0
        return attn_probs_from_global_key
    
    def _compute_attn_output_with_global_indices(self, value_vectors, attn_probs, max_num_global_attn_indices, is_index_global_attn_nonzero, is_local_index_global_attn_nonzero):
        batch_size = attn_probs.shape[0]
        attn_probs_only_global = attn_probs.narrow(-1, 0, max_num_global_attn_indices)
        value_vectors_only_global = value_vectors.new_zeros(batch_size, max_num_global_attn_indices, self.num_heads, self.head_dim)
        value_vectors_only_global[is_local_index_global_attn_nonzero] = value_vectors[is_index_global_attn_nonzero]
        attn_output_only_global = torch.matmul(attn_probs_only_global.transpose(1, 2), value_vectors_only_global.transpose(1, 2)).transpose(1, 2)
        attn_probs_without_global = attn_probs.narrow(-1, max_num_global_attn_indices, attn_probs.size(-1) - max_num_global_attn_indices).contiguous()
        attn_output_without_global = self._sliding_chunks_matmul_attn_probs_value(attn_probs_without_global, value_vectors, self.one_sided_attn_window_size)
        return attn_output_only_global + attn_output_without_global
    
    def _compute_global_attn_output_from_hidden(self, hidden_states, max_num_global_attn_indices, is_local_index_global_attn_nonzero, is_index_global_attn_nonzero, is_local_index_no_global_attn_nonzero, is_index_masked):
        (seq_len, batch_size) = hidden_states.shape[:2]
        global_attn_hidden_states = hidden_states.new_zeros(max_num_global_attn_indices, batch_size, self.embed_dim)
        global_attn_hidden_states[is_local_index_global_attn_nonzero[::-1]] = hidden_states[is_index_global_attn_nonzero[::-1]]
        global_query_vectors_only_global = self.query_global(global_attn_hidden_states)
        global_key_vectors = self.key_global(hidden_states)
        global_value_vectors = self.value_global(hidden_states)
        global_query_vectors_only_global /= math.sqrt(self.head_dim)
        global_query_vectors_only_global = global_query_vectors_only_global.contiguous().view(max_num_global_attn_indices, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        global_key_vectors = global_key_vectors.contiguous().view(-1, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        global_value_vectors = global_value_vectors.contiguous().view(-1, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        global_attn_scores = torch.bmm(global_query_vectors_only_global, global_key_vectors.transpose(1, 2))
        assert list(global_attn_scores.size()) == [batch_size * self.num_heads, max_num_global_attn_indices, seq_len], f'global_attn_scores have the wrong size. Size should be {(batch_size * self.num_heads, max_num_global_attn_indices, seq_len)}, but is {global_attn_scores.size()}.'
        global_attn_scores = global_attn_scores.view(batch_size, self.num_heads, max_num_global_attn_indices, seq_len)
        global_attn_scores[is_local_index_no_global_attn_nonzero[0], :, is_local_index_no_global_attn_nonzero[1], :] = -10000.0
        global_attn_scores = global_attn_scores.masked_fill(is_index_masked[:, None, None, :], -10000.0)
        global_attn_scores = global_attn_scores.view(batch_size * self.num_heads, max_num_global_attn_indices, seq_len)
        global_attn_probs_float = F.softmax(global_attn_scores, dim=-1, dtype=torch.float32)
        global_attn_probs = F.dropout(global_attn_probs_float.type_as(global_attn_scores), p=self.dropout, training=self.training)
        global_attn_output = torch.bmm(global_attn_probs, global_value_vectors)
        assert list(global_attn_output.size()) == [batch_size * self.num_heads, max_num_global_attn_indices, self.head_dim], f'global_attn_output tensor has the wrong size. Size should be {(batch_size * self.num_heads, max_num_global_attn_indices, self.head_dim)}, but is {global_attn_output.size()}.'
        global_attn_output = global_attn_output.view(batch_size, self.num_heads, max_num_global_attn_indices, self.head_dim)
        return global_attn_output



class LongformerSelfOutput(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states



class LongformerAttention(nn.Module):
    
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.self = LongformerSelfAttention(config, layer_id)
        self.output = LongformerSelfOutput(config)
        self.pruned_heads = set()
    
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        (heads, index) = find_pruneable_heads_and_indices(heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads)
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)
    
    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        self_outputs = self.self(hidden_states, attention_mask, output_attentions)
        attn_output = self.output(self_outputs[0], hidden_states)
        outputs = (attn_output, ) + self_outputs[1:]
        return outputs



class LongformerIntermediate(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
    
    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states



class LongformerOutput(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states



class LongformerLayer(nn.Module):
    
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.attention = LongformerAttention(config, layer_id)
        self.intermediate = LongformerIntermediate(config)
        self.output = LongformerOutput(config)
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
    
    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        self_attn_outputs = self.attention(hidden_states, attention_mask, output_attentions=output_attentions)
        attn_output = self_attn_outputs[0]
        outputs = self_attn_outputs[1:]
        layer_output = apply_chunking_to_forward(self.ff_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attn_output)
        outputs = (layer_output, ) + outputs
        return outputs
    
    def ff_chunk(self, attn_output):
        intermediate_output = self.intermediate(attn_output)
        layer_output = self.output(intermediate_output, attn_output)
        return layer_output



class LongformerEncoder(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([LongformerLayer(config, layer_id=i) for i in range(config.num_hidden_layers)])
    
    def forward(self, hidden_states, attention_mask=None, output_attentions=False, output_hidden_states=False, return_dict=False):
        all_hidden_states = (() if output_hidden_states else None)
        all_attentions = (() if output_attentions else None)
        for (i, layer_module) in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states, )
            if getattr(self.config, 'gradient_checkpointing', False):
                
                def create_custom_forward(module):
                    
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)
                    return custom_forward
                layer_outputs = torch.utils.checkpoint.checkpoint(create_custom_forward(layer_module), hidden_states, attention_mask)
            else:
                layer_outputs = layer_module(hidden_states, attention_mask, output_attentions)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1], )
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states, )
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None))
        return BaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions)



class LongformerPooler(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
    
    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output



class LongformerLMHead(nn.Module):
    """Longformer Head for masked language modeling."""
    
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias
    
    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)
        x = self.decoder(x)
        return x



class LongformerPreTrainedModel(PreTrainedModel):
    """An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained
    models.
    """
    config_class = LongformerConfig
    base_model_prefix = 'longformer'
    authorized_missing_keys = ['position_ids']
    
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if (isinstance(module, nn.Linear) and module.bias is not None):
            module.bias.data.zero_()

LONGFORMER_START_DOCSTRING = '\n\n    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic\n    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,\n    pruning heads etc.)\n\n    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__ subclass.\n    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general\n    usage and behavior.\n\n    Parameters:\n        config (:class:`~transformers.LongformerConfig`): Model configuration class with all the parameters of the\n            model. Initializing with a config file does not load the weights associated with the model, only the\n            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model\n            weights.\n'
LONGFORMER_INPUTS_DOCSTRING = "\n    Args:\n        input_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`):\n            Indices of input sequence tokens in the vocabulary.\n\n            Indices can be obtained using :class:`~transformers.LongformerTokenizer`.\n            See :meth:`transformers.PreTrainedTokenizer.encode` and\n            :meth:`transformers.PreTrainedTokenizer.__call__` for details.\n\n            `What are input IDs? <../glossary.html#input-ids>`__\n        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`({0})`, `optional`):\n            Mask to avoid performing attention on padding token indices.\n            Mask values selected in ``[0, 1]``:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            `What are attention masks? <../glossary.html#attention-mask>`__\n        global_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`({0})`, `optional`):\n            Mask to decide the attention given on each token, local attention or global attenion.\n            Tokens with global attention attends to all other tokens, and all other tokens attend to them. This is important for\n            task-specific finetuning because it makes the model more flexible at representing the task. For example,\n            for classification, the <s> token should be given global attention. For QA, all question tokens should also have\n            global attention. Please refer to the `Longformer paper <https://arxiv.org/abs/2004.05150>`__ for more details.\n            Mask values selected in ``[0, 1]``:\n\n            - 0 for local attention (a sliding window attention),\n            - 1 for global attention (tokens that attend to all other tokens, and all other tokens attend to them).\n\n        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):\n            Segment token indices to indicate first and second portions of the inputs.\n            Indices are selected in ``[0, 1]``:\n\n            - 0 corresponds to a `sentence A` token,\n            - 1 corresponds to a `sentence B` token.\n\n            `What are token type IDs? <../glossary.html#token-type-ids>`_\n        position_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):\n            Indices of positions of each input sequence tokens in the position embeddings.\n            Selected in the range ``[0, config.max_position_embeddings - 1]``.\n\n            `What are position IDs? <../glossary.html#position-ids>`_\n        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`({0}, hidden_size)`, `optional`):\n            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.\n            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated\n            vectors than the model's internal embedding lookup matrix.\n        output_attentions (:obj:`bool`, `optional`):\n            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned\n            tensors for more detail.\n        output_hidden_states (:obj:`bool`, `optional`):\n            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for\n            more detail.\n        return_dict (:obj:`bool`, `optional`):\n            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.\n"


@add_start_docstrings('The bare Longformer Model outputting raw hidden-states without any specific head on top.', LONGFORMER_START_DOCSTRING)
class LongformerModel(LongformerPreTrainedModel):
    """
    This class copied code from :class:`~transformers.RobertaModel` and overwrote standard self-attention with
    longformer self-attention to provide the ability to process
    long sequences following the self-attention approach described in `Longformer: the Long-Document Transformer
    <https://arxiv.org/abs/2004.05150>`__ by Iz Beltagy, Matthew E. Peters, and Arman Cohan. Longformer self-attention
    combines a local (sliding window) and global attention to extend to long documents without the O(n^2) increase in
    memory and compute.

    The self-attention module :obj:`LongformerSelfAttention` implemented here supports the combination of local and
    global attention but it lacks support for autoregressive attention and dilated attention. Autoregressive
    and dilated attention are more relevant for autoregressive language modeling than finetuning on downstream
    tasks. Future release will add support for autoregressive attention, but the support for dilated attention
    requires a custom CUDA kernel to be memory and compute efficient.

    """
    
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        if isinstance(config.attention_window, int):
            assert config.attention_window % 2 == 0, '`config.attention_window` has to be an even value'
            assert config.attention_window > 0, '`config.attention_window` has to be positive'
            config.attention_window = [config.attention_window] * config.num_hidden_layers
        else:
            assert len(config.attention_window) == config.num_hidden_layers, f'`len(config.attention_window)` should equal `config.num_hidden_layers`. Expected {config.num_hidden_layers}, given {len(config.attention_window)}'
        self.embeddings = LongformerEmbeddings(config)
        self.encoder = LongformerEncoder(config)
        self.pooler = (LongformerPooler(config) if add_pooling_layer else None)
        self.init_weights()
    
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings
    
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value
    
    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for (layer, heads) in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)
    
    def _pad_to_window_size(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, token_type_ids: torch.Tensor, position_ids: torch.Tensor, inputs_embeds: torch.Tensor, pad_token_id: int):
        """A helper function to pad tokens and mask to work with implementation of Longformer self-attention."""
        attention_window = (self.config.attention_window if isinstance(self.config.attention_window, int) else max(self.config.attention_window))
        assert attention_window % 2 == 0, f'`attention_window` should be an even value. Given {attention_window}'
        input_shape = (input_ids.shape if input_ids is not None else inputs_embeds.shape)
        (batch_size, seq_len) = input_shape[:2]
        padding_len = (attention_window - seq_len % attention_window) % attention_window
        if padding_len > 0:
            logger.info('Input ids are automatically padded from {} to {} to be a multiple of `config.attention_window`: {}'.format(seq_len, seq_len + padding_len, attention_window))
            if input_ids is not None:
                input_ids = F.pad(input_ids, (0, padding_len), value=pad_token_id)
            if position_ids is not None:
                position_ids = F.pad(position_ids, (0, padding_len), value=pad_token_id)
            if inputs_embeds is not None:
                input_ids_padding = inputs_embeds.new_full((batch_size, padding_len), self.config.pad_token_id, dtype=torch.long)
                inputs_embeds_padding = self.embeddings(input_ids_padding)
                inputs_embeds = torch.cat([inputs_embeds, inputs_embeds_padding], dim=-2)
            attention_mask = F.pad(attention_mask, (0, padding_len), value=False)
            token_type_ids = F.pad(token_type_ids, (0, padding_len), value=0)
        return (padding_len, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds)
    
    def _merge_to_attention_mask(self, attention_mask: torch.Tensor, global_attention_mask: torch.Tensor):
        if attention_mask is not None:
            attention_mask = attention_mask * (global_attention_mask + 1)
        else:
            attention_mask = global_attention_mask + 1
        return attention_mask
    
    @add_start_docstrings_to_callable(LONGFORMER_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=None, attention_mask=None, global_attention_mask=None, token_type_ids=None, position_ids=None, inputs_embeds=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        """

        Returns:

        Examples::

            >>> import torch
            >>> from transformers import LongformerModel, LongformerTokenizer

            >>> model = LongformerModel.from_pretrained('allenai/longformer-base-4096', return_dict=True)
            >>> tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')

            >>> SAMPLE_TEXT = ' '.join(['Hello world! '] * 1000)  # long input document
            >>> input_ids = torch.tensor(tokenizer.encode(SAMPLE_TEXT)).unsqueeze(0)  # batch of size 1

            >>> # Attention mask values -- 0: no attention, 1: local attention, 2: global attention
            >>> attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device) # initialize to local attention
            >>> global_attention_mask = torch.zeros(input_ids.shape, dtype=torch.long, device=input_ids.device) # initialize to global attention to be deactivated for all tokens
            >>> global_attention_mask[:, [1, 4, 21,]] = 1  # Set global attention to random tokens for the sake of this example
            ...                                     # Usually, set global attention based on the task. For example,
            ...                                     # classification: the <s> token
            ...                                     # QA: question tokens
            ...                                     # LM: potentially on the beginning of sentences and paragraphs
            >>> outputs = model(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask)
            >>> sequence_output = outputs.last_hidden_state
            >>> pooled_output = outputs.pooler_output
        """
        output_attentions = (output_attentions if output_attentions is not None else self.config.output_attentions)
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict = (return_dict if return_dict is not None else self.config.use_return_dict)
        if (input_ids is not None and inputs_embeds is not None):
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        device = (input_ids.device if input_ids is not None else inputs_embeds.device)
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        if global_attention_mask is not None:
            attention_mask = self._merge_to_attention_mask(attention_mask, global_attention_mask)
        (padding_len, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds) = self._pad_to_window_size(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, inputs_embeds=inputs_embeds, pad_token_id=self.config.pad_token_id)
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)
        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)
        encoder_outputs = self.encoder(embedding_output, attention_mask=extended_attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = encoder_outputs[0]
        pooled_output = (self.pooler(sequence_output) if self.pooler is not None else None)
        if padding_len > 0:
            sequence_output = sequence_output[:, :-padding_len]
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]
        return BaseModelOutputWithPooling(last_hidden_state=sequence_output, pooler_output=pooled_output, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)



@add_start_docstrings('Longformer Model with a `language modeling` head on top. ', LONGFORMER_START_DOCSTRING)
class LongformerForMaskedLM(LongformerPreTrainedModel):
    authorized_unexpected_keys = ['pooler']
    
    def __init__(self, config):
        super().__init__(config)
        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.lm_head = LongformerLMHead(config)
        self.init_weights()
    
    def get_output_embeddings(self):
        return self.lm_head.decoder
    
    @add_start_docstrings_to_callable(LONGFORMER_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=MaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=None, attention_mask=None, global_attention_mask=None, token_type_ids=None, position_ids=None, inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None, **kwargs):
        """
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.

        Returns:

        Examples::

            >>> import torch
            >>> from transformers import LongformerForMaskedLM, LongformerTokenizer

            >>> model = LongformerForMaskedLM.from_pretrained('allenai/longformer-base-4096', return_dict=True)
            >>> tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')

            >>> SAMPLE_TEXT = ' '.join(['Hello world! '] * 1000)  # long input document
            >>> input_ids = torch.tensor(tokenizer.encode(SAMPLE_TEXT)).unsqueeze(0)  # batch of size 1

            >>> attention_mask = None  # default is local attention everywhere, which is a good choice for MaskedLM
            ...                        # check ``LongformerModel.forward`` for more details how to set `attention_mask`
            >>> outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            >>> loss = outputs.loss
            >>> prediction_logits = output.logits
        """
        if 'masked_lm_labels' in kwargs:
            warnings.warn('The `masked_lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.', FutureWarning)
            labels = kwargs.pop('masked_lm_labels')
        assert kwargs == {}, f'Unexpected keyword arguments: {list(kwargs.keys())}.'
        return_dict = (return_dict if return_dict is not None else self.config.use_return_dict)
        outputs = self.longformer(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        if not return_dict:
            output = (prediction_scores, ) + outputs[2:]
            return ((masked_lm_loss, ) + output if masked_lm_loss is not None else output)
        return MaskedLMOutput(loss=masked_lm_loss, logits=prediction_scores, hidden_states=outputs.hidden_states, attentions=outputs.attentions)



@add_start_docstrings('Longformer Model transformer with a sequence classification/regression head on top (a linear layer\n    on top of the pooled output) e.g. for GLUE tasks. ', LONGFORMER_START_DOCSTRING)
class LongformerForSequenceClassification(LongformerPreTrainedModel):
    authorized_unexpected_keys = ['pooler']
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.classifier = LongformerClassificationHead(config)
        self.init_weights()
    
    @add_start_docstrings_to_callable(LONGFORMER_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint='allenai/longformer-base-4096', output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=None, attention_mask=None, global_attention_mask=None, token_type_ids=None, position_ids=None, inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        """
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (return_dict if return_dict is not None else self.config.use_return_dict)
        if global_attention_mask is None:
            logger.info('Initializing global attention on CLS token...')
            global_attention_mask = torch.zeros_like(input_ids)
            global_attention_mask[:, 0] = 1
        outputs = self.longformer(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if not return_dict:
            output = (logits, ) + outputs[2:]
            return ((loss, ) + output if loss is not None else output)
        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)



class LongformerClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
    
    def forward(self, hidden_states, **kwargs):
        hidden_states = hidden_states[:, 0, :]
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        output = self.out_proj(hidden_states)
        return output



@add_start_docstrings('Longformer Model with a span classification head on top for extractive question-answering tasks like SQuAD /\n    TriviaQA (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`). ', LONGFORMER_START_DOCSTRING)
class LongformerForQuestionAnswering(LongformerPreTrainedModel):
    authorized_unexpected_keys = ['pooler']
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()
    
    @add_start_docstrings_to_callable(LONGFORMER_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=QuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=None, attention_mask=None, global_attention_mask=None, token_type_ids=None, position_ids=None, inputs_embeds=None, start_positions=None, end_positions=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        """
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.

        Returns:

        Examples::

            >>> from transformers import LongformerTokenizer, LongformerForQuestionAnswering
            >>> import torch

            >>> tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-large-4096-finetuned-triviaqa")
            >>> model = LongformerForQuestionAnswering.from_pretrained("allenai/longformer-large-4096-finetuned-triviaqa", return_dict=True)

            >>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
            >>> encoding = tokenizer(question, text, return_tensors="pt")
            >>> input_ids = encoding["input_ids"]

            >>> # default is local attention everywhere
            >>> # the forward method will automatically set global attention on question tokens
            >>> attention_mask = encoding["attention_mask"]

            >>> outputs = model(input_ids, attention_mask=attention_mask)
            >>> start_logits = outputs.start_logits
            >>> end_logits = outputs.end_logits
            >>> all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

            >>> answer_tokens = all_tokens[torch.argmax(start_logits) :torch.argmax(end_logits)+1]
            >>> answer = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens)) # remove space prepending space token

        """
        return_dict = (return_dict if return_dict is not None else self.config.use_return_dict)
        if global_attention_mask is None:
            if input_ids is None:
                logger.warning('It is not possible to automatically generate the `global_attention_mask` because input_ids is None. Please make sure that it is correctly set.')
            else:
                global_attention_mask = _compute_global_attention_mask(input_ids, self.config.sep_token_id)
        outputs = self.longformer(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)
        (start_logits, end_logits) = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        total_loss = None
        if (start_positions is not None and end_positions is not None):
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss, ) + output if total_loss is not None else output)
        return QuestionAnsweringModelOutput(loss=total_loss, start_logits=start_logits, end_logits=end_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)



@add_start_docstrings('Longformer Model with a token classification head on top (a linear layer on top of\n    the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. ', LONGFORMER_START_DOCSTRING)
class LongformerForTokenClassification(LongformerPreTrainedModel):
    authorized_unexpected_keys = ['pooler']
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.longformer = LongformerModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()
    
    @add_start_docstrings_to_callable(LONGFORMER_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint='allenai/longformer-base-4096', output_type=TokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=None, attention_mask=None, global_attention_mask=None, token_type_ids=None, position_ids=None, inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        """
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
        """
        return_dict = (return_dict if return_dict is not None else self.config.use_return_dict)
        outputs = self.longformer(input_ids, attention_mask=attention_mask, global_attention_mask=global_attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels))
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if not return_dict:
            output = (logits, ) + outputs[2:]
            return ((loss, ) + output if loss is not None else output)
        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)



@add_start_docstrings('Longformer Model with a multiple choice classification head on top (a linear layer on top of\n    the pooled output and a softmax) e.g. for RocStories/SWAG tasks. ', LONGFORMER_START_DOCSTRING)
class LongformerForMultipleChoice(LongformerPreTrainedModel):
    
    def __init__(self, config):
        super().__init__(config)
        self.longformer = LongformerModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.init_weights()
    
    @add_start_docstrings_to_callable(LONGFORMER_INPUTS_DOCSTRING.format('batch_size, num_choices, sequence_length'))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint='allenai/longformer-base-4096', output_type=MultipleChoiceModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, global_attention_mask=None, labels=None, position_ids=None, inputs_embeds=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        """
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices-1]`` where :obj:`num_choices` is the size of the second dimension
            of the input tensors. (See :obj:`input_ids` above)
        """
        num_choices = (input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1])
        return_dict = (return_dict if return_dict is not None else self.config.use_return_dict)
        if (global_attention_mask is None and input_ids is not None):
            logger.info('Initializing global attention on multiple choice...')
            global_attention_mask = torch.stack([_compute_global_attention_mask(input_ids[:, i], self.config.sep_token_id, before_sep_token=False) for i in range(num_choices)], dim=1)
        flat_input_ids = (input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None)
        flat_position_ids = (position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None)
        flat_token_type_ids = (token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None)
        flat_attention_mask = (attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None)
        flat_global_attention_mask = (global_attention_mask.view(-1, global_attention_mask.size(-1)) if global_attention_mask is not None else None)
        flat_inputs_embeds = (inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1)) if inputs_embeds is not None else None)
        outputs = self.longformer(flat_input_ids, position_ids=flat_position_ids, token_type_ids=flat_token_type_ids, attention_mask=flat_attention_mask, global_attention_mask=flat_global_attention_mask, inputs_embeds=flat_inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
        if not return_dict:
            output = (reshaped_logits, ) + outputs[2:]
            return ((loss, ) + output if loss is not None else output)
        return MultipleChoiceModelOutput(loss=loss, logits=reshaped_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)


