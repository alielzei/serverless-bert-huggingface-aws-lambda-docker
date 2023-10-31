""" TF 2.0 OpenAI GPT-2 model. """

from dataclasses import dataclass
from typing import List, Optional, Tuple
import tensorflow as tf
from .activations_tf import get_tf_activation
from .configuration_gpt2 import GPT2Config
from .file_utils import ModelOutput, add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_callable, replace_return_docstrings
from .modeling_tf_outputs import TFBaseModelOutputWithPast, TFCausalLMOutputWithPast
from .modeling_tf_utils import TFCausalLanguageModelingLoss, TFConv1D, TFPreTrainedModel, TFSequenceSummary, TFSharedEmbeddings, get_initializer, keras_serializable, shape_list
from .tokenization_utils import BatchEncoding
from .utils import logging
logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = 'GPT2Config'
_TOKENIZER_FOR_DOC = 'GPT2Tokenizer'
TF_GPT2_PRETRAINED_MODEL_ARCHIVE_LIST = ['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl', 'distilgpt2']


class TFAttention(tf.keras.layers.Layer):
    
    def __init__(self, nx, n_ctx, config, scale=False, **kwargs):
        super().__init__(**kwargs)
        n_state = nx
        assert n_state % config.n_head == 0
        self.n_ctx = n_ctx
        self.n_head = config.n_head
        self.split_size = n_state
        self.scale = scale
        self.output_attentions = config.output_attentions
        self.c_attn = TFConv1D(n_state * 3, nx, initializer_range=config.initializer_range, name='c_attn')
        self.c_proj = TFConv1D(n_state, nx, initializer_range=config.initializer_range, name='c_proj')
        self.attn_dropout = tf.keras.layers.Dropout(config.attn_pdrop)
        self.resid_dropout = tf.keras.layers.Dropout(config.resid_pdrop)
        self.pruned_heads = set()
    
    def prune_heads(self, heads):
        pass
    
    @staticmethod
    def causal_attention_mask(nd, ns, dtype):
        """1's in the lower triangle, counting from the lower right corner.
        Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
        """
        i = tf.range(nd)[:, None]
        j = tf.range(ns)
        m = i >= j - ns + nd
        return tf.cast(m, dtype)
    
    def _attn(self, q, k, v, attention_mask, head_mask, output_attentions, training=False):
        w = tf.matmul(q, k, transpose_b=True)
        if self.scale:
            dk = tf.cast(shape_list(k)[-1], tf.float32)
            w = w / tf.math.sqrt(dk)
        (_, _, nd, ns) = shape_list(w)
        b = self.causal_attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w * b - 10000.0 * (1 - b)
        if attention_mask is not None:
            w = w + attention_mask
        w = tf.nn.softmax(w, axis=-1)
        w = self.attn_dropout(w, training=training)
        if head_mask is not None:
            w = w * head_mask
        outputs = [tf.matmul(w, v)]
        if output_attentions:
            outputs.append(w)
        return outputs
    
    def merge_heads(self, x):
        x = tf.transpose(x, [0, 2, 1, 3])
        x_shape = shape_list(x)
        new_x_shape = x_shape[:-2] + [x_shape[-2] * x_shape[-1]]
        return tf.reshape(x, new_x_shape)
    
    def split_heads(self, x):
        x_shape = shape_list(x)
        new_x_shape = x_shape[:-1] + [self.n_head, x_shape[-1] // self.n_head]
        x = tf.reshape(x, new_x_shape)
        return tf.transpose(x, (0, 2, 1, 3))
    
    def call(self, x, layer_past, attention_mask, head_mask, use_cache, output_attentions, training=False):
        x = self.c_attn(x)
        (query, key, value) = tf.split(x, 3, axis=2)
        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)
        if layer_past is not None:
            (past_key, past_value) = tf.unstack(layer_past, axis=0)
            key = tf.concat([past_key, key], axis=-2)
            value = tf.concat([past_value, value], axis=-2)
        if use_cache:
            present = tf.stack([key, value], axis=0)
        else:
            present = (None, )
        attn_outputs = self._attn(query, key, value, attention_mask, head_mask, output_attentions, training=training)
        a = attn_outputs[0]
        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a, training=training)
        outputs = [a, present] + attn_outputs[1:]
        return outputs



class TFMLP(tf.keras.layers.Layer):
    
    def __init__(self, n_state, config, **kwargs):
        super().__init__(**kwargs)
        nx = config.n_embd
        self.c_fc = TFConv1D(n_state, nx, initializer_range=config.initializer_range, name='c_fc')
        self.c_proj = TFConv1D(nx, n_state, initializer_range=config.initializer_range, name='c_proj')
        self.act = get_tf_activation('gelu')
        self.dropout = tf.keras.layers.Dropout(config.resid_pdrop)
    
    def call(self, x, training=False):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        h2 = self.dropout(h2, training=training)
        return h2



class TFBlock(tf.keras.layers.Layer):
    
    def __init__(self, n_ctx, config, scale=False, **kwargs):
        super().__init__(**kwargs)
        nx = config.n_embd
        inner_dim = (config.n_inner if config.n_inner is not None else 4 * nx)
        self.ln_1 = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon, name='ln_1')
        self.attn = TFAttention(nx, n_ctx, config, scale, name='attn')
        self.ln_2 = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon, name='ln_2')
        self.mlp = TFMLP(inner_dim, config, name='mlp')
    
    def call(self, x, layer_past, attention_mask, head_mask, use_cache, output_attentions, training=False):
        a = self.ln_1(x)
        output_attn = self.attn(a, layer_past, attention_mask, head_mask, use_cache, output_attentions, training=training)
        a = output_attn[0]
        x = x + a
        m = self.ln_2(x)
        m = self.mlp(m, training=training)
        x = x + m
        outputs = [x] + output_attn[1:]
        return outputs



@keras_serializable
class TFGPT2MainLayer(tf.keras.layers.Layer):
    config_class = GPT2Config
    
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.use_cache = config.use_cache
        self.return_dict = config.use_return_dict
        self.num_hidden_layers = config.n_layer
        self.vocab_size = config.vocab_size
        self.n_embd = config.n_embd
        self.wte = TFSharedEmbeddings(config.vocab_size, config.hidden_size, initializer_range=config.initializer_range, name='wte')
        self.wpe = tf.keras.layers.Embedding(config.n_positions, config.n_embd, embeddings_initializer=get_initializer(config.initializer_range), name='wpe')
        self.drop = tf.keras.layers.Dropout(config.embd_pdrop)
        self.h = [TFBlock(config.n_ctx, config, scale=True, name='h_._{}'.format(i)) for i in range(config.n_layer)]
        self.ln_f = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_epsilon, name='ln_f')
    
    def get_input_embeddings(self):
        return self.wte
    
    def set_input_embeddings(self, value):
        self.wte.weight = value
        self.wte.vocab_size = self.wte.weight.shape[0]
    
    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        raise NotImplementedError
    
    def call(self, inputs, past=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None, training=False):
        if isinstance(inputs, (tuple, list)):
            input_ids = inputs[0]
            past = (inputs[1] if len(inputs) > 1 else past)
            attention_mask = (inputs[2] if len(inputs) > 2 else attention_mask)
            token_type_ids = (inputs[3] if len(inputs) > 3 else token_type_ids)
            position_ids = (inputs[4] if len(inputs) > 4 else position_ids)
            head_mask = (inputs[5] if len(inputs) > 5 else head_mask)
            inputs_embeds = (inputs[6] if len(inputs) > 6 else inputs_embeds)
            use_cache = (inputs[7] if len(inputs) > 7 else use_cache)
            output_attentions = (inputs[8] if len(inputs) > 8 else output_attentions)
            output_hidden_states = (inputs[9] if len(inputs) > 9 else output_hidden_states)
            return_dict = (inputs[10] if len(inputs) > 10 else return_dict)
            assert len(inputs) <= 11, 'Too many inputs.'
        elif isinstance(inputs, (dict, BatchEncoding)):
            input_ids = inputs.get('input_ids')
            past = inputs.get('past', past)
            attention_mask = inputs.get('attention_mask', attention_mask)
            token_type_ids = inputs.get('token_type_ids', token_type_ids)
            position_ids = inputs.get('position_ids', position_ids)
            head_mask = inputs.get('head_mask', head_mask)
            inputs_embeds = inputs.get('inputs_embeds', inputs_embeds)
            use_cache = inputs.get('use_cache', use_cache)
            output_attentions = inputs.get('output_attentions', output_attentions)
            output_hidden_states = inputs.get('output_hidden_states', output_hidden_states)
            return_dict = inputs.get('return_dict', return_dict)
            assert len(inputs) <= 11, 'Too many inputs.'
        else:
            input_ids = inputs
        output_attentions = (output_attentions if output_attentions is not None else self.output_attentions)
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.output_hidden_states)
        use_cache = (use_cache if use_cache is not None else self.use_cache)
        return_dict = (return_dict if return_dict is not None else self.return_dict)
        if (input_ids is not None and inputs_embeds is not None):
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
            input_ids = tf.reshape(input_ids, [-1, input_shape[-1]])
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = shape_list(past[0][0])[-2]
        if position_ids is None:
            position_ids = tf.range(past_length, input_shape[-1] + past_length, dtype=tf.int32)[tf.newaxis, :]
        if attention_mask is not None:
            attention_mask = attention_mask[:, tf.newaxis, tf.newaxis, :]
            attention_mask = tf.cast(attention_mask, tf.float32)
            attention_mask = (1.0 - attention_mask) * -10000.0
        else:
            attention_mask = None
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.num_hidden_layers
        position_ids = tf.reshape(position_ids, [-1, shape_list(position_ids)[-1]])
        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids, mode='embedding')
        position_embeds = self.wpe(position_ids)
        if token_type_ids is not None:
            token_type_ids = tf.reshape(token_type_ids, [-1, shape_list(token_type_ids)[-1]])
            token_type_embeds = self.wte(token_type_ids, mode='embedding')
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states, training=training)
        output_shape = input_shape + [shape_list(hidden_states)[-1]]
        presents = (() if use_cache else None)
        all_attentions = (() if output_attentions else None)
        all_hidden_states = (() if output_hidden_states else None)
        for (i, (block, layer_past)) in enumerate(zip(self.h, past)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (tf.reshape(hidden_states, output_shape), )
            outputs = block(hidden_states, layer_past, attention_mask, head_mask[i], use_cache, output_attentions, training=training)
            (hidden_states, present) = outputs[:2]
            if use_cache:
                presents = presents + (present, )
            if output_attentions:
                all_attentions = all_attentions + (outputs[2], )
        hidden_states = self.ln_f(hidden_states)
        hidden_states = tf.reshape(hidden_states, output_shape)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states, )
        if output_attentions:
            attention_output_shape = input_shape[:-1] + [-1] + shape_list(all_attentions[0])[-2:]
            all_attentions = tuple((tf.reshape(t, attention_output_shape) for t in all_attentions))
        if not return_dict:
            return tuple((v for v in [hidden_states, presents, all_hidden_states, all_attentions] if v is not None))
        return TFBaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=presents, hidden_states=all_hidden_states, attentions=all_attentions)



class TFGPT2PreTrainedModel(TFPreTrainedModel):
    """An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """
    config_class = GPT2Config
    base_model_prefix = 'transformer'



@dataclass
class TFGPT2DoubleHeadsModelOutput(ModelOutput):
    """
    Base class for outputs of models predicting if two sentences are consecutive or not.

    Args:
        logits (:obj:`tf.Tensor` of shape :obj:`(batch_size, num_choices, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        mc_logits (:obj:`tf.Tensor` of shape :obj:`(batch_size, num_choices)`):
            Prediction scores of the multiple choice classification head (scores for each choice before SoftMax).
        past_key_values (:obj:`List[tf.Tensor]`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            List of :obj:`tf.Tensor` of length :obj:`config.n_layers`,  with each tensor of shape
            :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`).

            Contains pre-computed hidden-states (key and values in the attention blocks) that can be used (see
            :obj:`past_key_values` input) to speed up sequential decoding.
        hidden_states (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`tf.Tensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(tf.Tensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`tf.Tensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """
    logits: tf.Tensor = None
    mc_logits: tf.Tensor = None
    past_key_values: Optional[List[tf.Tensor]] = None
    hidden_states: Optional[Tuple[tf.Tensor]] = None
    attentions: Optional[Tuple[tf.Tensor]] = None

GPT2_START_DOCSTRING = '\n\n    This model inherits from :class:`~transformers.TFPreTrainedModel`. Check the superclass documentation for the\n    generic methods the library implements for all its model (such as downloading or saving, resizing the input\n    embeddings, pruning heads etc.)\n\n    This model is also a `tf.keras.Model <https://www.tensorflow.org/api_docs/python/tf/keras/Model>`__ subclass.\n    Use it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general\n    usage and behavior.\n\n    .. note::\n\n        TF 2.0 models accepts two formats as inputs:\n\n        - having all inputs as keyword arguments (like PyTorch models), or\n        - having all inputs as a list, tuple or dict in the first positional arguments.\n\n        This second option is useful when using :meth:`tf.keras.Model.fit` method which currently requires having\n        all the tensors in the first argument of the model call function: :obj:`model(inputs)`.\n\n        If you choose this second option, there are three possibilities you can use to gather all the input Tensors\n        in the first positional argument :\n\n        - a single Tensor with :obj:`input_ids` only and nothing else: :obj:`model(inputs_ids)`\n        - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:\n          :obj:`model([input_ids, attention_mask])` or :obj:`model([input_ids, attention_mask, token_type_ids])`\n        - a dictionary with one or several input Tensors associated to the input names given in the docstring:\n          :obj:`model({"input_ids": input_ids, "token_type_ids": token_type_ids})`\n\n    Parameters:\n        config (:class:`~transformers.GPT2Config`): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the configuration.\n            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.\n'
GPT2_INPUTS_DOCSTRING = "\n    Args:\n        input_ids (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`(batch_size, input_ids_length)`):\n            :obj:`input_ids_length` = ``sequence_length`` if ``past`` is ``None`` else ``past[0].shape[-2]``\n            (``sequence_length`` of input past key value states).\n            Indices of input sequence tokens in the vocabulary.\n\n            If :obj:`past` is used, only input IDs that do not have their past calculated should be passed as\n            ``input_ids``.\n\n            Indices can be obtained using :class:`~transformers.GPT2Tokenizer`.\n            See :func:`transformers.PreTrainedTokenizer.__call__` and\n            :func:`transformers.PreTrainedTokenizer.encode` for details.\n\n            `What are input IDs? <../glossary.html#input-ids>`__\n        past (:obj:`List[tf.Tensor]` of length :obj:`config.n_layers`):\n            Contains pre-computed hidden-states (key and values in the attention blocks) as computed by the model\n            (see :obj:`past` output below). Can be used to speed up sequential decoding.\n            The token ids which have their past given to this model\n            should not be passed as input ids as they have already been computed.\n        attention_mask (:obj:`tf.Tensor` or :obj:`Numpy array` of shape :obj:`(batch_size, sequence_length)`, `optional`):\n            Mask to avoid performing attention on padding token indices.\n            Mask values selected in ``[0, 1]``:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            `What are attention masks? <../glossary.html#attention-mask>`__\n        token_type_ids (:obj:`tf.Tensor` or :obj:`Numpy array` of shape :obj:`(batch_size, sequence_length)`, `optional`):\n            Segment token indices to indicate first and second portions of the inputs.\n            Indices are selected in ``[0, 1]``:\n\n            - 0 corresponds to a `sentence A` token,\n            - 1 corresponds to a `sentence B` token.\n\n            `What are token type IDs? <../glossary.html#token-type-ids>`__\n        position_ids (:obj:`tf.Tensor` or :obj:`Numpy array` of shape :obj:`(batch_size, sequence_length)`, `optional`):\n            Indices of positions of each input sequence tokens in the position embeddings.\n            Selected in the range ``[0, config.max_position_embeddings - 1]``.\n\n            `What are position IDs? <../glossary.html#position-ids>`__\n        head_mask (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):\n            Mask to nullify selected heads of the self-attention modules.\n            Mask values selected in ``[0, 1]``:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        inputs_embeds (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):\n            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.\n            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated\n            vectors than the model's internal embedding lookup matrix.\n        output_attentions (:obj:`bool`, `optional`):\n            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned\n            tensors for more detail.\n        output_hidden_states (:obj:`bool`, `optional`):\n            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for\n            more detail.\n        return_dict (:obj:`bool`, `optional`):\n            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.\n        training (:obj:`bool`, `optional`, defaults to :obj:`False`):\n            Whether or not to use the model in training mode (some modules like dropout modules have different\n            behaviors between training and evaluation).\n"


@add_start_docstrings('The bare GPT2 Model transformer outputing raw hidden-states without any specific head on top.', GPT2_START_DOCSTRING)
class TFGPT2Model(TFGPT2PreTrainedModel):
    
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.transformer = TFGPT2MainLayer(config, name='transformer')
    
    @add_start_docstrings_to_callable(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint='gpt2', output_type=TFBaseModelOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def call(self, inputs, **kwargs):
        outputs = self.transformer(inputs, **kwargs)
        return outputs



@add_start_docstrings('The GPT2 Model transformer with a language modeling head on top\n    (linear layer with weights tied to the input embeddings). ', GPT2_START_DOCSTRING)
class TFGPT2LMHeadModel(TFGPT2PreTrainedModel, TFCausalLanguageModelingLoss):
    
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.transformer = TFGPT2MainLayer(config, name='transformer')
    
    def get_output_embeddings(self):
        return self.transformer.wte
    
    def prepare_inputs_for_generation(self, inputs, past, **kwargs):
        if past:
            inputs = tf.expand_dims(inputs[:, -1], -1)
        return {'inputs': inputs, 'past': past, 'use_cache': kwargs['use_cache']}
    
    @add_start_docstrings_to_callable(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint='gpt2', output_type=TFCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def call(self, inputs, past=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None, labels=None, training=False):
        """
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the cross entropy classification loss.
            Indices should be in ``[0, ..., config.vocab_size - 1]``.
        """
        return_dict = (return_dict if return_dict is not None else self.transformer.return_dict)
        if isinstance(inputs, (tuple, list)):
            labels = (inputs[11] if len(inputs) > 11 else labels)
            if len(inputs) > 11:
                inputs = inputs[:11]
        elif isinstance(inputs, (dict, BatchEncoding)):
            labels = inputs.pop('labels', labels)
        transformer_outputs = self.transformer(inputs, past=past, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        hidden_states = transformer_outputs[0]
        logits = self.transformer.wte(hidden_states, mode='linear')
        loss = None
        if labels is not None:
            logits = logits[:, :-1]
            labels = labels[:, 1:]
            loss = self.compute_loss(labels, logits)
        if not return_dict:
            output = (logits, ) + transformer_outputs[1:]
            return ((loss, ) + output if loss is not None else output)
        return TFCausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=transformer_outputs.past_key_values, hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions)



@add_start_docstrings('The GPT2 Model transformer with a language modeling and a multiple-choice classification\n    head on top e.g. for RocStories/SWAG tasks. The two heads are two linear layers.\n    The language modeling head has its weights tied to the input embeddings,\n    the classification head takes as input the input of a specified classification token index in the input sequence).\n', GPT2_START_DOCSTRING)
class TFGPT2DoubleHeadsModel(TFGPT2PreTrainedModel):
    
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        config.num_labels = 1
        self.transformer = TFGPT2MainLayer(config, name='transformer')
        self.multiple_choice_head = TFSequenceSummary(config, initializer_range=config.initializer_range, name='multiple_choice_head')
    
    def get_output_embeddings(self):
        return self.transformer.wte
    
    @add_start_docstrings_to_callable(GPT2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TFGPT2DoubleHeadsModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, inputs, past=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, mc_token_ids=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None, training=False):
        """
        mc_token_ids (:obj:`tf.Tensor` or :obj:`Numpy array` of shape :obj:`(batch_size, num_choices)`, `optional`, default to index of the last token of the input)
            Index of the classification token in each input sequence.
            Selected in the range ``[0, input_ids.size(-1) - 1[``.

        Return:

        Examples::

            >>> import tensorflow as tf
            >>> from transformers import GPT2Tokenizer, TFGPT2DoubleHeadsModel

            >>> tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            >>> model = TFGPT2DoubleHeadsModel.from_pretrained('gpt2')

            >>> # Add a [CLS] to the vocabulary (we should train it also!)
            >>> num_added_tokens = tokenizer.add_special_tokens({'cls_token': '[CLS]'})

            >>> embedding_layer = model.resize_token_embeddings(len(tokenizer))  # Update the model embeddings with the new vocabulary size

            >>> choices = ["Hello, my dog is cute [CLS]", "Hello, my cat is cute [CLS]"]
            >>> encoded_choices = [tokenizer.encode(s) for s in choices]
            >>> cls_token_location = [tokens.index(tokenizer.cls_token_id) for tokens in encoded_choices]

            >>> input_ids = tf.constant(encoded_choices)[None, :]  # Batch size: 1, number of choices: 2
            >>> mc_token_ids = tf.constant([cls_token_location])  # Batch size: 1

            >>> outputs = model(input_ids, mc_token_ids=mc_token_ids)
            >>> lm_prediction_scores, mc_prediction_scores = outputs[:2]

        """
        if isinstance(inputs, (tuple, list)):
            input_ids = inputs[0]
            past = (inputs[1] if len(inputs) > 1 else past)
            attention_mask = (inputs[2] if len(inputs) > 2 else attention_mask)
            token_type_ids = (inputs[3] if len(inputs) > 3 else token_type_ids)
            position_ids = (inputs[4] if len(inputs) > 4 else position_ids)
            head_mask = (inputs[5] if len(inputs) > 5 else head_mask)
            inputs_embeds = (inputs[6] if len(inputs) > 6 else inputs_embeds)
            mc_token_ids = (inputs[7] if len(inputs) > 7 else mc_token_ids)
            use_cache = (inputs[8] if len(inputs) > 8 else use_cache)
            output_attentions = (inputs[9] if len(inputs) > 9 else output_attentions)
            output_hidden_states = (inputs[10] if len(inputs) > 10 else output_hidden_states)
            return_dict = (inputs[11] if len(inputs) > 11 else return_dict)
            assert len(inputs) <= 12, 'Too many inputs.'
        elif isinstance(inputs, dict):
            input_ids = inputs.get('input_ids')
            past = inputs.get('past', past)
            attention_mask = inputs.get('attention_mask', attention_mask)
            token_type_ids = inputs.get('token_type_ids', token_type_ids)
            position_ids = inputs.get('position_ids', position_ids)
            head_mask = inputs.get('head_mask', head_mask)
            inputs_embeds = inputs.get('inputs_embeds', inputs_embeds)
            mc_token_ids = inputs.get('mc_token_ids', mc_token_ids)
            use_cache = inputs.get('use_cache', use_cache)
            output_attentions = inputs.get('output_attentions', output_attentions)
            output_hidden_states = inputs.get('output_hidden_states', output_hidden_states)
            return_dict = inputs.get('return_dict', return_dict)
            assert len(inputs) <= 12, 'Too many inputs.'
        else:
            input_ids = inputs
        return_dict = (return_dict if return_dict is not None else self.transformer.return_dict)
        if input_ids is not None:
            input_shapes = shape_list(input_ids)
        else:
            input_shapes = shape_list(inputs_embeds)[:-1]
        seq_length = input_shapes[-1]
        flat_input_ids = (tf.reshape(input_ids, (-1, seq_length)) if input_ids is not None else None)
        flat_attention_mask = (tf.reshape(attention_mask, (-1, seq_length)) if attention_mask is not None else None)
        flat_token_type_ids = (tf.reshape(token_type_ids, (-1, seq_length)) if token_type_ids is not None else None)
        flat_position_ids = (tf.reshape(position_ids, (-1, seq_length)) if position_ids is not None else None)
        transformer_outputs = self.transformer(flat_input_ids, past, flat_attention_mask, flat_token_type_ids, flat_position_ids, head_mask, inputs_embeds, use_cache, output_attentions, output_hidden_states, return_dict=return_dict, training=training)
        hidden_states = transformer_outputs[0]
        hidden_states = tf.reshape(hidden_states, input_shapes + shape_list(hidden_states)[-1:])
        lm_logits = self.transformer.wte(hidden_states, mode='linear')
        mc_logits = self.multiple_choice_head(hidden_states, mc_token_ids, training=training)
        mc_logits = tf.squeeze(mc_logits, axis=-1)
        if not return_dict:
            return (lm_logits, mc_logits) + transformer_outputs[1:]
        return TFGPT2DoubleHeadsModelOutput(logits=lm_logits, mc_logits=mc_logits, past_key_values=transformer_outputs.past_key_values, hidden_states=transformer_outputs.hidden_states, attentions=transformer_outputs.attentions)


