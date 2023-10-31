from typing import Callable, Dict
import numpy as np
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen import compact
from .configuration_roberta import RobertaConfig
from .file_utils import add_start_docstrings
from .modeling_flax_utils import FlaxPreTrainedModel, gelu
from .utils import logging
logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = 'RobertaConfig'
_TOKENIZER_FOR_DOC = 'RobertaTokenizer'
ROBERTA_START_DOCSTRING = '\n\n    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic\n    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,\n    pruning heads etc.)\n\n    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__ subclass.\n    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general\n    usage and behavior.\n\n    Parameters:\n        config (:class:`~transformers.RobertaConfig`): Model configuration class with all the parameters of the\n            model. Initializing with a config file does not load the weights associated with the model, only the configuration.\n            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.\n'
ROBERTA_INPUTS_DOCSTRING = "\n    Args:\n        input_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`):\n            Indices of input sequence tokens in the vocabulary.\n\n            Indices can be obtained using :class:`~transformers.RobertaTokenizer`.\n            See :meth:`transformers.PreTrainedTokenizer.encode` and\n            :meth:`transformers.PreTrainedTokenizer.__call__` for details.\n\n            `What are input IDs? <../glossary.html#input-ids>`__\n        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`({0})`, `optional`):\n            Mask to avoid performing attention on padding token indices.\n            Mask values selected in ``[0, 1]``:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **maked**.\n\n            `What are attention masks? <../glossary.html#attention-mask>`__\n        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):\n            Segment token indices to indicate first and second portions of the inputs.\n            Indices are selected in ``[0, 1]``:\n\n            - 0 corresponds to a `sentence A` token,\n            - 1 corresponds to a `sentence B` token.\n\n            `What are token type IDs? <../glossary.html#token-type-ids>`_\n        position_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):\n            Indices of positions of each input sequence tokens in the position embeddings.\n            Selected in the range ``[0, config.max_position_embeddings - 1]``.\n\n            `What are position IDs? <../glossary.html#position-ids>`_\n        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):\n            Mask to nullify selected heads of the self-attention modules.\n            Mask values selected in ``[0, 1]``:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`({0}, hidden_size)`, `optional`):\n            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.\n            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated\n            vectors than the model's internal embedding lookup matrix.\n        output_attentions (:obj:`bool`, `optional`):\n            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned\n            tensors for more detail.\n        output_hidden_states (:obj:`bool`, `optional`):\n            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for\n            more detail.\n        return_dict (:obj:`bool`, `optional`):\n            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.\n"


class FlaxRobertaLayerNorm(nn.Module):
    """Layer normalization (https://arxiv.org/abs/1607.06450).
    Operates on the last axis of the input data.
    """
    epsilon: float = 1e-06
    dtype: jnp.dtype = jnp.float32
    bias: bool = True
    scale: bool = True
    bias_init: jnp.ndarray = nn.initializers.zeros
    scale_init: jnp.ndarray = nn.initializers.ones
    
    @compact
    def __call__(self, x):
        """Applies layer normalization on the input.
        It normalizes the activations of the layer for each given example in a
        batch independently, rather than across a batch like Batch Normalization.
        i.e. applies a transformation that maintains the mean activation within
        each example close to 0 and the activation standard deviation close to 1.
        Args:
          x: the inputs
          epsilon: A small float added to variance to avoid dividing by zero.
          dtype: the dtype of the computation (default: float32).
          bias:  If True, bias (beta) is added.
          scale: If True, multiply by scale (gamma). When the next layer is linear
            (also e.g. nn.relu), this can be disabled since the scaling will be done
            by the next layer.
          bias_init: Initializer for bias, by default, zero.
          scale_init: Initializer for scale, by default, one.
        Returns:
          Normalized inputs (the same shape as inputs).
        """
        features = x.shape[-1]
        mean = jnp.mean(x, axis=-1, keepdims=True)
        mean2 = jnp.mean(jax.lax.square(x), axis=-1, keepdims=True)
        var = mean2 - jax.lax.square(mean)
        mul = jax.lax.rsqrt(var + self.epsilon)
        if self.scale:
            mul = mul * jnp.asarray(self.param('gamma', self.scale_init, (features, )), self.dtype)
        y = (x - mean) * mul
        if self.bias:
            y = y + jnp.asarray(self.param('beta', self.bias_init, (features, )), self.dtype)
        return y



class FlaxRobertaEmbedding(nn.Module):
    """
    Specify a new class for doing the embedding stuff
    as Flax's one use 'embedding' for the parameter name
    and PyTorch use 'weight'
    """
    vocab_size: int
    hidden_size: int
    emb_init: Callable[(..., np.ndarray)] = nn.initializers.normal(stddev=0.1)
    
    @compact
    def __call__(self, inputs):
        embedding = self.param('weight', self.emb_init, (self.vocab_size, self.hidden_size))
        return jnp.take(embedding, inputs, axis=0)



class FlaxRobertaEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    vocab_size: int
    hidden_size: int
    type_vocab_size: int
    max_length: int
    
    @compact
    def __call__(self, input_ids, token_type_ids, position_ids, attention_mask):
        w_emb = FlaxRobertaEmbedding(self.vocab_size, self.hidden_size, name='word_embeddings')(jnp.atleast_2d(input_ids.astype('i4')))
        p_emb = FlaxRobertaEmbedding(self.max_length, self.hidden_size, name='position_embeddings')(jnp.atleast_2d(position_ids.astype('i4')))
        t_emb = FlaxRobertaEmbedding(self.type_vocab_size, self.hidden_size, name='token_type_embeddings')(jnp.atleast_2d(token_type_ids.astype('i4')))
        summed_emb = w_emb + jnp.broadcast_to(p_emb, w_emb.shape) + t_emb
        layer_norm = FlaxRobertaLayerNorm(name='layer_norm')(summed_emb)
        return layer_norm



class FlaxRobertaAttention(nn.Module):
    num_heads: int
    head_size: int
    
    @compact
    def __call__(self, hidden_state, attention_mask):
        self_att = nn.attention.SelfAttention(num_heads=self.num_heads, qkv_features=self.head_size, name='self')(hidden_state, attention_mask)
        layer_norm = FlaxRobertaLayerNorm(name='layer_norm')(self_att + hidden_state)
        return layer_norm



class FlaxRobertaIntermediate(nn.Module):
    output_size: int
    
    @compact
    def __call__(self, hidden_state):
        dense = nn.Dense(features=self.output_size, name='dense')(hidden_state)
        return gelu(dense)



class FlaxRobertaOutput(nn.Module):
    
    @compact
    def __call__(self, intermediate_output, attention_output):
        hidden_state = nn.Dense(attention_output.shape[-1], name='dense')(intermediate_output)
        hidden_state = FlaxRobertaLayerNorm(name='layer_norm')(hidden_state + attention_output)
        return hidden_state



class FlaxRobertaLayer(nn.Module):
    num_heads: int
    head_size: int
    intermediate_size: int
    
    @compact
    def __call__(self, hidden_state, attention_mask):
        attention = FlaxRobertaAttention(self.num_heads, self.head_size, name='attention')(hidden_state, attention_mask)
        intermediate = FlaxRobertaIntermediate(self.intermediate_size, name='intermediate')(attention)
        output = FlaxRobertaOutput(name='output')(intermediate, attention)
        return output



class FlaxRobertaLayerCollection(nn.Module):
    """
    Stores N RobertaLayer(s)
    """
    num_layers: int
    num_heads: int
    head_size: int
    intermediate_size: int
    
    @compact
    def __call__(self, inputs, attention_mask):
        assert self.num_layers > 0, f'num_layers should be >= 1, got ({self.num_layers})'
        input_i = inputs
        for i in range(self.num_layers):
            layer = FlaxRobertaLayer(self.num_heads, self.head_size, self.intermediate_size, name=f'{i}')
            input_i = layer(input_i, attention_mask)
        return input_i



class FlaxRobertaEncoder(nn.Module):
    num_layers: int
    num_heads: int
    head_size: int
    intermediate_size: int
    
    @compact
    def __call__(self, hidden_state, attention_mask):
        layer = FlaxRobertaLayerCollection(self.num_layers, self.num_heads, self.head_size, self.intermediate_size, name='layer')(hidden_state, attention_mask)
        return layer



class FlaxRobertaPooler(nn.Module):
    
    @compact
    def __call__(self, hidden_state):
        cls_token = hidden_state[:, 0]
        out = nn.Dense(hidden_state.shape[-1], name='dense')(cls_token)
        return jax.lax.tanh(out)



class FlaxRobertaModule(nn.Module):
    vocab_size: int
    hidden_size: int
    type_vocab_size: int
    max_length: int
    num_encoder_layers: int
    num_heads: int
    head_size: int
    intermediate_size: int
    
    @compact
    def __call__(self, input_ids, token_type_ids, position_ids, attention_mask):
        embeddings = FlaxRobertaEmbeddings(self.vocab_size, self.hidden_size, self.type_vocab_size, self.max_length, name='embeddings')(input_ids, token_type_ids, position_ids, attention_mask)
        encoder = FlaxRobertaEncoder(self.num_encoder_layers, self.num_heads, self.head_size, self.intermediate_size, name='encoder')(embeddings, attention_mask)
        pooled = FlaxRobertaPooler(name='pooler')(encoder)
        return (encoder, pooled)



@add_start_docstrings('The bare RoBERTa Model transformer outputting raw hidden-states without any specific head on top.', ROBERTA_START_DOCSTRING)
class FlaxRobertaModel(FlaxPreTrainedModel):
    """
    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need`_ by Ashish Vaswani,
    Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.
    """
    model_class = FlaxRobertaModule
    config_class = RobertaConfig
    base_model_prefix = 'roberta'
    
    @staticmethod
    def convert_from_pytorch(pt_state: Dict, config: RobertaConfig) -> Dict:
        jax_state = dict(pt_state)
        for (key, tensor) in pt_state.items():
            key_parts = set(key.split('.'))
            if 'dense.weight' in key:
                del jax_state[key]
                key = key.replace('weight', 'kernel')
                jax_state[key] = tensor
            if {'query', 'key', 'value'} & key_parts:
                if 'bias' in key:
                    jax_state[key] = tensor.reshape((config.num_attention_heads, -1))
                elif 'weight':
                    del jax_state[key]
                    key = key.replace('weight', 'kernel')
                    tensor = tensor.reshape((config.num_attention_heads, -1, config.hidden_size)).transpose((2, 0, 1))
                    jax_state[key] = tensor
            if 'attention.output.dense' in key:
                del jax_state[key]
                key = key.replace('attention.output.dense', 'attention.self.out')
                jax_state[key] = tensor
            if 'attention.output.LayerNorm' in key:
                del jax_state[key]
                key = key.replace('attention.output.LayerNorm', 'attention.LayerNorm')
                jax_state[key] = tensor
            if ('intermediate.dense.kernel' in key or 'output.dense.kernel' in key):
                jax_state[key] = tensor.T
            if 'out.kernel' in key:
                jax_state[key] = tensor.reshape((config.hidden_size, config.num_attention_heads, -1)).transpose(1, 2, 0)
            if 'pooler.dense.kernel' in key:
                jax_state[key] = tensor.T
            if 'LayerNorm' in key:
                del jax_state[key]
                new_key = key.replace('LayerNorm', 'layer_norm')
                if 'weight' in key:
                    new_key = new_key.replace('weight', 'gamma')
                elif 'bias' in key:
                    new_key = new_key.replace('bias', 'beta')
                jax_state[new_key] = tensor
        return jax_state
    
    def __init__(self, config: RobertaConfig, state: dict, seed: int = 0, **kwargs):
        model = FlaxRobertaModule(vocab_size=config.vocab_size, hidden_size=config.hidden_size, type_vocab_size=config.type_vocab_size, max_length=config.max_position_embeddings, num_encoder_layers=config.num_hidden_layers, num_heads=config.num_attention_heads, head_size=config.hidden_size, intermediate_size=config.intermediate_size)
        super().__init__(config, model, state, seed)
    
    @property
    def module(self) -> nn.Module:
        return self._module
    
    def __call__(self, input_ids, token_type_ids=None, position_ids=None, attention_mask=None):
        if token_type_ids is None:
            token_type_ids = jnp.ones_like(input_ids)
        if position_ids is None:
            position_ids = np.arange(self.config.pad_token_id + 1, np.atleast_2d(input_ids).shape[-1] + self.config.pad_token_id + 1)
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        return self.model.apply({'params': self.params}, jnp.array(input_ids, dtype='i4'), jnp.array(token_type_ids, dtype='i4'), jnp.array(position_ids, dtype='i4'), jnp.array(attention_mask, dtype='i4'))


