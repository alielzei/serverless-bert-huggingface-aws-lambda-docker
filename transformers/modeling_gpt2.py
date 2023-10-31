"""PyTorch OpenAI GPT-2 model."""

import os
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from .activations import ACT2FN
from .configuration_gpt2 import GPT2Config
from .file_utils import ModelOutput, add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_callable, replace_return_docstrings
from .modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast
from .modeling_utils import Conv1D, PreTrainedModel, SequenceSummary, find_pruneable_heads_and_indices, prune_conv1d_layer
from .utils import logging
logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = 'GPT2Config'
_TOKENIZER_FOR_DOC = 'GPT2Tokenizer'
GPT2_PRETRAINED_MODEL_ARCHIVE_LIST = ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'distilgpt2']

def load_tf_weights_in_gpt2(model, config, gpt2_checkpoint_path):
    """Load tf checkpoints in a pytorch model"""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.modeling_gpt2.load_tf_weights_in_gpt2', 'load_tf_weights_in_gpt2(model, config, gpt2_checkpoint_path)', {'logger': logger, 'os': os, 'torch': torch, 'model': model, 'config': config, 'gpt2_checkpoint_path': gpt2_checkpoint_path}, 1)


class Attention(nn.Module):
    
    def __init__(self, nx, n_ctx, config, scale=False, is_cross_attention=False):
        super().__init__()
        n_state = nx
        assert n_state % config.n_head == 0
        self.register_buffer('bias', torch.tril(torch.ones((n_ctx, n_ctx), dtype=torch.uint8)).view(1, 1, n_ctx, n_ctx))
        self.register_buffer('masked_bias', torch.tensor(-10000.0))
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.is_cross_attention = is_cross_attention
        if self.is_cross_attention:
            self.c_attn = Conv1D(2 * n_state, nx)
            self.q_attn = Conv1D(n_state, nx)
        else:
            self.c_attn = Conv1D(3 * n_state, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.pruned_heads = set()
    
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        (heads, index) = find_pruneable_heads_and_indices(heads, self.n_head, self.split_size // self.n_head, self.pruned_heads)
        index_attn = torch.cat([index, index + self.split_size, index + 2 * self.split_size])
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)
        self.split_size = self.split_size // self.n_head * (self.n_head - len(heads))
        self.n_head = self.n_head - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)
    
    def _attn(self, q, k, v, attention_mask=None, head_mask=None, output_attentions=False):
        w = torch.matmul(q, k)
        if self.scale:
            w = w / float(v.size(-1))**0.5
        (nd, ns) = (w.size(-2), w.size(-1))
        if not self.is_cross_attention:
            mask = self.bias[:, :, ns - nd:ns, :ns]
            w = torch.where(mask.bool(), w, self.masked_bias.to(w.dtype))
        if attention_mask is not None:
            w = w + attention_mask
        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)
        if head_mask is not None:
            w = w * head_mask
        outputs = [torch.matmul(w, v)]
        if output_attentions:
            outputs.append(w)
        return outputs
    
    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1), )
        return x.view(*new_x_shape)
    
    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, use_cache=False, output_attentions=False):
        if encoder_hidden_states is not None:
            assert hasattr(self, 'q_attn'), 'If class is used as cross attention, the weights `q_attn` have to be defined. Please make sure to instantiate class with `Attention(..., is_cross_attention=True)`.'
            query = self.q_attn(hidden_states)
            (key, value) = self.c_attn(encoder_hidden_states).split(self.split_size, dim=2)
            attention_mask = encoder_attention_mask
        else:
            (query, key, value) = self.c_attn(hidden_states).split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        if layer_past is not None:
            (past_key, past_value) = (layer_past[0].transpose(-2, -1), layer_past[1])
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)
        if use_cache is True:
            present = torch.stack((key.transpose(-2, -1), value))
        else:
            present = (None, )
        attn_outputs = self._attn(query, key, value, attention_mask, head_mask, output_attentions)
        a = attn_outputs[0]
        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)
        outputs = [a, present] + attn_outputs[1:]
        return outputs



class MLP(nn.Module):
    
    def __init__(self, n_state, config):
        super().__init__()
        nx = config.n_embd
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = ACT2FN[config.activation_function]
        self.dropout = nn.Dropout(config.resid_pdrop)
    
    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)



class Block(nn.Module):
    
    def __init__(self, n_ctx, config, scale=False):
        super().__init__()
        hidden_size = config.n_embd
        inner_dim = (config.n_inner if config.n_inner is not None else 4 * hidden_size)
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = Attention(hidden_size, n_ctx, config, scale)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        if config.add_cross_attention:
            self.crossattention = Attention(hidden_size, n_ctx, config, scale, is_cross_attention=True)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = MLP(inner_dim, config)
    
    def forward(self, hidden_states, layer_past=None, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None, use_cache=False, output_attentions=False):
        attn_outputs = self.attn(self.ln_1(hidden_states), layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask, use_cache=use_cache, output_attentions=output_attentions)
        attn_output = attn_outputs[0]
        outputs = attn_outputs[1:]
        hidden_states = attn_output + hidden_states
        if encoder_hidden_states is not None:
            assert hasattr(self, 'crossattention'), f'If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`'
            cross_attn_outputs = self.crossattention(self.ln_cross_attn(hidden_states), attention_mask=attention_mask, head_mask=head_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, output_attentions=output_attentions)
            attn_output = cross_attn_outputs[0]
            hidden_states = hidden_states + attn_output
            outputs = outputs + cross_attn_outputs[1:]
        feed_forward_hidden_states = self.mlp(self.ln_2(hidden_states))
        hidden_states = hidden_states + feed_forward_hidden_states
        outputs = [hidden_states] + outputs
        return outputs



class GPT2PreTrainedModel(PreTrainedModel):
    """An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """
    config_class = GPT2Config
    load_tf_weights = load_tf_weights_in_gpt2
    base_model_prefix = 'transformer'
    
    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)
    
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if (isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None):
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)



@dataclass
class GPT2DoubleHeadsModelOutput(ModelOutput):
    """
    Base class for outputs of models predicting if two sentences are consecutive or not.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided):
            Language modeling loss.
        mc_loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`mc_labels` is provided):
            Multiple choice classification loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_choices, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        mc_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, num_choices)`):
            Prediction scores of the multiple choice classification head (scores for each choice before SoftMax).
        past_key_values (:obj:`List[torch.FloatTensor]`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            List of :obj:`torch.FloatTensor` of length :obj:`config.n_layers`,  with each tensor of shape
            :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`).

            Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
            :obj:`past_key_values` input) to speed up sequential decoding.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    loss: Optional[torch.FloatTensor] = None
    mc_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    mc_logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

GPT2_START_DOCSTRING = '\n\n    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic\n    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,\n    pruning heads etc.)\n\n    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__ subclass.\n    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general\n    usage and behavior.\n\n    Parameters:\n        config (:class:`~transformers.GPT2Config`): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the configuration.\n            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.\n'
GPT2_INPUTS_DOCSTRING = "\n    Args:\n        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, input_ids_length)`):\n            :obj:`input_ids_length` = ``sequence_length`` if :obj:`past_key_values` is ``None`` else\n            ``past_key_values[0].shape[-2]`` (``sequence_length`` of input past key value states).\n            Indices of input sequence tokens in the vocabulary.\n\n            If :obj:`past_key_values` is used, only ``input_ids`` that do not have their past calculated should be passed\n            as ``input_ids``.\n\n            Indices can be obtained using :class:`~transformers.GPT2Tokenizer`.\n            See :meth:`transformers.PreTrainedTokenizer.encode` and\n            :meth:`transformers.PreTrainedTokenizer.__call__` for details.\n\n            `What are input IDs? <../glossary.html#input-ids>`__\n        past_key_values (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers`):\n            Contains precomputed hidden-states (key and values in the attention blocks) as computed by the model\n            (see :obj:`past_key_values` output below). Can be used to speed up sequential decoding.\n            The ``input_ids`` which have their past given to this model should not be passed as ``input_ids`` as they\n            have already been computed.\n        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):\n            Mask to avoid performing attention on padding token indices.\n            Mask values selected in ``[0, 1]``:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            `What are attention masks? <../glossary.html#attention-mask>`__\n        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, input_ids_length)`, `optional`):\n            Segment token indices to indicate first and second portions of the inputs.\n            Indices are selected in ``[0, 1]``:\n\n            - 0 corresponds to a `sentence A` token,\n            - 1 corresponds to a `sentence B` token.\n\n            `What are token type IDs? <../glossary.html#token-type-ids>`_\n        position_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):\n            Indices of positions of each input sequence tokens in the position embeddings.\n            Selected in the range ``[0, config.max_position_embeddings - 1]``.\n\n            `What are position IDs? <../glossary.html#position-ids>`_\n        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):\n            Mask to nullify selected heads of the self-attention modules.\n            Mask values selected in ``[0, 1]``:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):\n            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.\n            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated\n            vectors than the model's internal embedding lookup matrix.\n\n            If :obj:`past_key_values` is used, optionally only the last :obj:`inputs_embeds` have to be input (see\n            :obj:`past_key_values`).\n        use_cache (:obj:`bool`, `optional`):\n            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up\n            decoding (see :obj:`past_key_values`).\n        output_attentions (:obj:`bool`, `optional`):\n            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned\n            tensors for more detail.\n        output_hidden_states (:obj:`bool`, `optional`):\n            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for\n            more detail.\n        return_dict (:obj:`bool`, `optional`):\n            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.\n"


@add_start_docstrings('The bare GPT2 Model transformer outputting raw hidden-states without any specific head on top.', GPT2_START_DOCSTRING)
class GPT2Model(GPT2PreTrainedModel):
    
    def __init__(self, config):
        super().__init__(config)
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([Block(config.n_ctx, config, scale=True) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.init_weights()
    
    def get_input_embeddings(self):
        return self.wte
    
    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings
    
    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for (layer, heads) in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)
    
    @add_start_docstrings_to_callable(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint='gpt2', output_type=BaseModelOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=None, past_key_values=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None, **kwargs):
        if 'past' in kwargs:
            warnings.warn('The `past` argument is deprecated and will be removed in a future version, use `past_key_values` instead.', FutureWarning)
            past_key_values = kwargs.pop('past')
        assert kwargs == {}, f'Unexpected keyword arguments: {list(kwargs.keys())}.'
        output_attentions = (output_attentions if output_attentions is not None else self.config.output_attentions)
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        use_cache = (use_cache if use_cache is not None else self.config.use_cache)
        return_dict = (return_dict if return_dict is not None else self.config.use_return_dict)
        if (input_ids is not None and inputs_embeds is not None):
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])
        if past_key_values is None:
            past_length = 0
            past_key_values = [None] * len(self.h)
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            device = (input_ids.device if input_ids is not None else inputs_embeds.device)
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
        if attention_mask is not None:
            assert batch_size > 0, 'batch_size has to be defined and > 0'
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=self.dtype)
            attention_mask = (1.0 - attention_mask) * -10000.0
        if (self.config.add_cross_attention and encoder_hidden_states is not None):
            (encoder_batch_size, encoder_sequence_length, _) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1), )
        presents = (() if use_cache else None)
        all_attentions = (() if output_attentions else None)
        all_hidden_states = (() if output_hidden_states else None)
        for (i, (block, layer_past)) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape), )
            if getattr(self.config, 'gradient_checkpointing', False):
                
                def create_custom_forward(module):
                    
                    def custom_forward(*inputs):
                        return tuple((output for output in module(*inputs, use_cache, output_attentions)))
                    return custom_forward
                outputs = torch.utils.checkpoint.checkpoint(create_custom_forward(block), hidden_states, layer_past, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask)
            else:
                outputs = block(hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask[i], encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, use_cache=use_cache, output_attentions=output_attentions)
            (hidden_states, present) = outputs[:2]
            if use_cache is True:
                presents = presents + (present, )
            if output_attentions:
                all_attentions = all_attentions + (outputs[2], )
        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(*output_shape)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states, )
        if not return_dict:
            return tuple((v for v in [hidden_states, presents, all_hidden_states, all_attentions] if v is not None))
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=presents, hidden_states=all_hidden_states, attentions=all_attentions)



@add_start_docstrings('The GPT2 Model transformer with a language modeling head on top\n    (linear layer with weights tied to the input embeddings). ', GPT2_START_DOCSTRING)
class GPT2LMHeadModel(GPT2PreTrainedModel):
    authorized_missing_keys = ['h\\.\\d+\\.attn\\.masked_bias', 'lm_head\\.weight']
    
    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.init_weights()
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
        attention_mask = kwargs.get('attention_mask', None)
        position_ids = kwargs.get('position_ids', None)
        if (attention_mask is not None and position_ids is None):
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {'input_ids': input_ids, 'past_key_values': past, 'use_cache': kwargs.get('use_cache'), 'position_ids': position_ids, 'attention_mask': attention_mask}
    
    @add_start_docstrings_to_callable(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint='gpt2', output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=None, past_key_values=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None, labels=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None, **kwargs):
        """
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling.
            Note that the labels **are shifted** inside the model, i.e. you can set ``labels = input_ids``
            Indices are selected in ``[-100, 0, ..., config.vocab_size]``
            All labels set to ``-100`` are ignored (masked), the loss is only
            computed for labels in ``[0, ..., config.vocab_size]``
        """
        if 'past' in kwargs:
            warnings.warn('The `past` argument is deprecated and will be removed in a future version, use `past_key_values` instead.', FutureWarning)
            past_key_values = kwargs.pop('past')
        assert kwargs == {}, f'Unexpected keyword arguments: {list(kwargs.keys())}.'
        return_dict = (return_dict if return_dict is not None else self.config.use_return_dict)
        transformer_outputs = self.transformer(input_ids, past_key_values=past_key_values, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        if not return_dict:
            output = (lm_logits, ) + transformer_outputs[1:]
            return ((loss, ) + output if loss is not None else output)
        return CausalLMOutputWithPast(loss=loss, logits=lm_logits, past_key_values=transformer_outputs.past_key_values, hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions)



@add_start_docstrings('The GPT2 Model transformer with a language modeling and a multiple-choice classification\n    head on top e.g. for RocStories/SWAG tasks. The two heads are two linear layers.\n    The language modeling head has its weights tied to the input embeddings,\n    the classification head takes as input the input of a specified classification token index in the input sequence).\n', GPT2_START_DOCSTRING)
class GPT2DoubleHeadsModel(GPT2PreTrainedModel):
    
    def __init__(self, config):
        super().__init__(config)
        config.num_labels = 1
        self.transformer = GPT2Model(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.multiple_choice_head = SequenceSummary(config)
        self.init_weights()
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
        return {'input_ids': input_ids, 'past_key_values': past, 'use_cache': kwargs.get('use_cache')}
    
    @add_start_docstrings_to_callable(GPT2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=GPT2DoubleHeadsModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=None, past_key_values=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, mc_token_ids=None, labels=None, mc_labels=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None, **kwargs):
        """
        mc_token_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, num_choices)`, `optional`, default to index of the last token of the input)
            Index of the classification token in each input sequence.
            Selected in the range ``[0, input_ids.size(-1) - 1[``.
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`)
            Labels for language modeling.
            Note that the labels **are shifted** inside the model, i.e. you can set ``labels = input_ids``
            Indices are selected in ``[-1, 0, ..., config.vocab_size]``
            All labels set to ``-100`` are ignored (masked), the loss is only
            computed for labels in ``[0, ..., config.vocab_size]``
        mc_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size)`, `optional`)
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices]`` where `num_choices` is the size of the second dimension
            of the input tensors. (see `input_ids` above)
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.

        Return:

        Example::

            >>> import torch
            >>> from transformers import GPT2Tokenizer, GPT2DoubleHeadsModel

            >>> tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            >>> model = GPT2DoubleHeadsModel.from_pretrained('gpt2, return_dict=True)

            >>> # Add a [CLS] to the vocabulary (we should train it also!)
            >>> num_added_tokens = tokenizer.add_special_tokens({'cls_token': '[CLS]'})

            >>> embedding_layer = model.resize_token_embeddings(len(tokenizer))  # Update the model embeddings with the new vocabulary size

            >>> choices = ["Hello, my dog is cute [CLS]", "Hello, my cat is cute [CLS]"]
            >>> encoded_choices = [tokenizer.encode(s) for s in choices]
            >>> cls_token_location = [tokens.index(tokenizer.cls_token_id) for tokens in encoded_choices]

            >>> input_ids = torch.tensor(encoded_choices).unsqueeze(0)  # Batch size: 1, number of choices: 2
            >>> mc_token_ids = torch.tensor([cls_token_location])  # Batch size: 1

            >>> outputs = model(input_ids, mc_token_ids=mc_token_ids)
            >>> lm_logits = outputs.lm_logits
            >>> mc_logits = outputs.mc_logits

        """
        if 'lm_labels' in kwargs:
            warnings.warn('The `lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.', FutureWarning)
            labels = kwargs.pop('lm_labels')
        if 'past' in kwargs:
            warnings.warn('The `past` argument is deprecated and will be removed in a future version, use `past_key_values` instead.', FutureWarning)
            past_key_values = kwargs.pop('past')
        assert kwargs == {}, f'Unexpected keyword arguments: {list(kwargs.keys())}.'
        return_dict = (return_dict if return_dict is not None else self.config.use_return_dict)
        transformer_outputs = self.transformer(input_ids, past_key_values=past_key_values, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)
        mc_logits = self.multiple_choice_head(hidden_states, mc_token_ids).squeeze(-1)
        mc_loss = None
        if mc_labels is not None:
            loss_fct = CrossEntropyLoss()
            mc_loss = loss_fct(mc_logits.view(-1, mc_logits.size(-1)), mc_labels.view(-1))
        lm_loss = None
        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        if not return_dict:
            output = (lm_logits, mc_logits) + transformer_outputs[1:]
            if mc_loss is not None:
                output = (mc_loss, ) + output
            return ((lm_loss, ) + output if lm_loss is not None else output)
        return GPT2DoubleHeadsModelOutput(loss=lm_loss, mc_loss=mc_loss, logits=lm_logits, mc_logits=mc_logits, past_key_values=transformer_outputs.past_key_values, hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions)



@add_start_docstrings('The GPT2 Model transformer with a sequence classification head on top\n    (linear layer).\n\n    :class:`~transformers.GPT2ForSequenceClassification` uses the last token in order to do the classification, as\n    other causal models (e.g. GPT-1) do.\n\n    Since it does classification on the last token, it requires to know the position of the last token.\n    If a :obj:`pad_token_id` is defined in the configuration, it finds the last token that is not a padding token\n    in each row. If no :obj:`pad_token_id` is defined, it simply takes the last value in each row of the batch.\n    Since it cannot guess the padding tokens when :obj:`inputs_embeds` are passed instead of :obj:`input_ids`, it\n    does the same (take the last value in each row of the batch).\n    ', GPT2_START_DOCSTRING)
class GPT2ForSequenceClassification(GPT2PreTrainedModel):
    authorized_missing_keys = ['h\\.\\d+\\.attn\\.masked_bias', 'lm_head\\.weight']
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.transformer = GPT2Model(config)
        self.score = nn.Linear(config.n_embd, self.num_labels, bias=False)
        self.init_weights()
    
    @add_start_docstrings_to_callable(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint='microsoft/dialogrpt', output_type=SequenceClassifierOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=None, past_key_values=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, labels=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        """
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (return_dict if return_dict is not None else self.config.use_return_dict)
        transformer_outputs = self.transformer(input_ids, past_key_values=past_key_values, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)
        if input_ids is not None:
            (batch_size, sequence_length) = input_ids.shape[:2]
        else:
            (batch_size, sequence_length) = inputs_embeds.shape[:2]
        assert (self.config.pad_token_id is not None or batch_size == 1), 'Cannot handle batch sizes > 1 if no padding token is defined.'
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        elif input_ids is not None:
            sequence_lengths = torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1
        else:
            sequence_lengths = -1
            logger.warning(f'{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be unexpected if using padding tokens in conjuction with `inputs_embeds.`')
        pooled_logits = logits[(range(batch_size), sequence_lengths)]
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(pooled_logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
        if not return_dict:
            output = (pooled_logits, ) + transformer_outputs[1:]
            return ((loss, ) + output if loss is not None else output)
        return SequenceClassifierOutputWithPast(loss=loss, logits=pooled_logits, past_key_values=transformer_outputs.past_key_values, hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions)


