""" TF 2.0 BERT model. """

from dataclasses import dataclass
from typing import Optional, Tuple
import tensorflow as tf
from .activations_tf import get_tf_activation
from .configuration_bert import BertConfig
from .file_utils import MULTIPLE_CHOICE_DUMMY_INPUTS, ModelOutput, add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_callable, replace_return_docstrings
from .modeling_tf_outputs import TFBaseModelOutput, TFBaseModelOutputWithPooling, TFCausalLMOutput, TFMaskedLMOutput, TFMultipleChoiceModelOutput, TFNextSentencePredictorOutput, TFQuestionAnsweringModelOutput, TFSequenceClassifierOutput, TFTokenClassifierOutput
from .modeling_tf_utils import TFCausalLanguageModelingLoss, TFMaskedLanguageModelingLoss, TFMultipleChoiceLoss, TFPreTrainedModel, TFQuestionAnsweringLoss, TFSequenceClassificationLoss, TFTokenClassificationLoss, get_initializer, keras_serializable, shape_list
from .tokenization_utils import BatchEncoding
from .utils import logging
logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = 'BertConfig'
_TOKENIZER_FOR_DOC = 'BertTokenizer'
TF_BERT_PRETRAINED_MODEL_ARCHIVE_LIST = ['bert-base-uncased', 'bert-large-uncased', 'bert-base-cased', 'bert-large-cased', 'bert-base-multilingual-uncased', 'bert-base-multilingual-cased', 'bert-base-chinese', 'bert-base-german-cased', 'bert-large-uncased-whole-word-masking', 'bert-large-cased-whole-word-masking', 'bert-large-uncased-whole-word-masking-finetuned-squad', 'bert-large-cased-whole-word-masking-finetuned-squad', 'bert-base-cased-finetuned-mrpc', 'cl-tohoku/bert-base-japanese', 'cl-tohoku/bert-base-japanese-whole-word-masking', 'cl-tohoku/bert-base-japanese-char', 'cl-tohoku/bert-base-japanese-char-whole-word-masking', 'TurkuNLP/bert-base-finnish-cased-v1', 'TurkuNLP/bert-base-finnish-uncased-v1', 'wietsedv/bert-base-dutch-cased']


class TFBertEmbeddings(tf.keras.layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""
    
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.initializer_range = config.initializer_range
        self.position_embeddings = tf.keras.layers.Embedding(config.max_position_embeddings, config.hidden_size, embeddings_initializer=get_initializer(self.initializer_range), name='position_embeddings')
        self.token_type_embeddings = tf.keras.layers.Embedding(config.type_vocab_size, config.hidden_size, embeddings_initializer=get_initializer(self.initializer_range), name='token_type_embeddings')
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='LayerNorm')
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
    
    def build(self, input_shape):
        """Build shared word embedding layer """
        with tf.name_scope('word_embeddings'):
            self.word_embeddings = self.add_weight('weight', shape=[self.vocab_size, self.hidden_size], initializer=get_initializer(self.initializer_range))
        super().build(input_shape)
    
    def call(self, input_ids=None, position_ids=None, token_type_ids=None, inputs_embeds=None, mode='embedding', training=False):
        """Get token embeddings of inputs.
        Args:
            inputs: list of three int64 tensors with shape [batch_size, length]: (input_ids, position_ids, token_type_ids)
            mode: string, a valid value is one of "embedding" and "linear".
        Returns:
            outputs: (1) If mode == "embedding", output embedding tensor, float32 with
                shape [batch_size, length, embedding_size]; (2) mode == "linear", output
                linear tensor, float32 with shape [batch_size, length, vocab_size].
        Raises:
            ValueError: if mode is not valid.

        Shared weights logic adapted from
            https://github.com/tensorflow/models/blob/a009f4fb9d2fc4949e32192a944688925ef78659/official/transformer/v2/embedding_layer.py#L24
        """
        if mode == 'embedding':
            return self._embedding(input_ids, position_ids, token_type_ids, inputs_embeds, training=training)
        elif mode == 'linear':
            return self._linear(input_ids)
        else:
            raise ValueError('mode {} is not valid.'.format(mode))
    
    def _embedding(self, input_ids, position_ids, token_type_ids, inputs_embeds, training=False):
        """Applies embedding based on inputs tensor."""
        assert not ((input_ids is None and inputs_embeds is None))
        if input_ids is not None:
            input_shape = shape_list(input_ids)
        else:
            input_shape = shape_list(inputs_embeds)[:-1]
        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = tf.range(seq_length, dtype=tf.int32)[tf.newaxis, :]
        if token_type_ids is None:
            token_type_ids = tf.fill(input_shape, 0)
        if inputs_embeds is None:
            inputs_embeds = tf.gather(self.word_embeddings, input_ids)
        position_embeddings = tf.cast(self.position_embeddings(position_ids), inputs_embeds.dtype)
        token_type_embeddings = tf.cast(self.token_type_embeddings(token_type_ids), inputs_embeds.dtype)
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings, training=training)
        return embeddings
    
    def _linear(self, inputs):
        """Computes logits by running inputs through a linear layer.
        Args:
            inputs: A float32 tensor with shape [batch_size, length, hidden_size]
        Returns:
            float32 tensor with shape [batch_size, length, vocab_size].
        """
        batch_size = shape_list(inputs)[0]
        length = shape_list(inputs)[1]
        x = tf.reshape(inputs, [-1, self.hidden_size])
        logits = tf.matmul(x, self.word_embeddings, transpose_b=True)
        return tf.reshape(logits, [batch_size, length, self.vocab_size])



class TFBertSelfAttention(tf.keras.layers.Layer):
    
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError('The hidden size (%d) is not a multiple of the number of attention heads (%d)' % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        assert config.hidden_size % config.num_attention_heads == 0
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = tf.keras.layers.Dense(self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name='query')
        self.key = tf.keras.layers.Dense(self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name='key')
        self.value = tf.keras.layers.Dense(self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name='value')
        self.dropout = tf.keras.layers.Dropout(config.attention_probs_dropout_prob)
    
    def transpose_for_scores(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_attention_heads, self.attention_head_size))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, hidden_states, attention_mask, head_mask, output_attentions, training=False):
        batch_size = shape_list(hidden_states)[0]
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)
        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size)
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size)
        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        dk = tf.cast(shape_list(key_layer)[-1], attention_scores.dtype)
        attention_scores = attention_scores / tf.math.sqrt(dk)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = tf.nn.softmax(attention_scores, axis=-1)
        attention_probs = self.dropout(attention_probs, training=training)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        context_layer = tf.matmul(attention_probs, value_layer)
        context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])
        context_layer = tf.reshape(context_layer, (batch_size, -1, self.all_head_size))
        outputs = ((context_layer, attention_probs) if output_attentions else (context_layer, ))
        return outputs



class TFBertSelfOutput(tf.keras.layers.Layer):
    
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name='dense')
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='LayerNorm')
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
    
    def call(self, hidden_states, input_tensor, training=False):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states



class TFBertAttention(tf.keras.layers.Layer):
    
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.self_attention = TFBertSelfAttention(config, name='self')
        self.dense_output = TFBertSelfOutput(config, name='output')
    
    def prune_heads(self, heads):
        raise NotImplementedError
    
    def call(self, input_tensor, attention_mask, head_mask, output_attentions, training=False):
        self_outputs = self.self_attention(input_tensor, attention_mask, head_mask, output_attentions, training=training)
        attention_output = self.dense_output(self_outputs[0], input_tensor, training=training)
        outputs = (attention_output, ) + self_outputs[1:]
        return outputs



class TFBertIntermediate(tf.keras.layers.Layer):
    
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name='dense')
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.intermediate_act_fn = config.hidden_act
    
    def call(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states



class TFBertOutput(tf.keras.layers.Layer):
    
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name='dense')
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='LayerNorm')
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
    
    def call(self, hidden_states, input_tensor, training=False):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states



class TFBertLayer(tf.keras.layers.Layer):
    
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.attention = TFBertAttention(config, name='attention')
        self.intermediate = TFBertIntermediate(config, name='intermediate')
        self.bert_output = TFBertOutput(config, name='output')
    
    def call(self, hidden_states, attention_mask, head_mask, output_attentions, training=False):
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask, output_attentions, training=training)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.bert_output(intermediate_output, attention_output, training=training)
        outputs = (layer_output, ) + attention_outputs[1:]
        return outputs



class TFBertEncoder(tf.keras.layers.Layer):
    
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.layer = [TFBertLayer(config, name='layer_._{}'.format(i)) for i in range(config.num_hidden_layers)]
    
    def call(self, hidden_states, attention_mask, head_mask, output_attentions, output_hidden_states, return_dict, training=False):
        all_hidden_states = (() if output_hidden_states else None)
        all_attentions = (() if output_attentions else None)
        for (i, layer_module) in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states, )
            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i], output_attentions, training=training)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1], )
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states, )
        if not return_dict:
            return tuple((v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None))
        return TFBaseModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions)



class TFBertPooler(tf.keras.layers.Layer):
    
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), activation='tanh', name='dense')
    
    def call(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        return pooled_output



class TFBertPredictionHeadTransform(tf.keras.layers.Layer):
    
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.dense = tf.keras.layers.Dense(config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name='dense')
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name='LayerNorm')
    
    def call(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states



class TFBertLMPredictionHead(tf.keras.layers.Layer):
    
    def __init__(self, config, input_embeddings, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = config.vocab_size
        self.transform = TFBertPredictionHeadTransform(config, name='transform')
        self.input_embeddings = input_embeddings
    
    def build(self, input_shape):
        self.bias = self.add_weight(shape=(self.vocab_size, ), initializer='zeros', trainable=True, name='bias')
        super().build(input_shape)
    
    def call(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.input_embeddings(hidden_states, mode='linear')
        hidden_states = hidden_states + self.bias
        return hidden_states



class TFBertMLMHead(tf.keras.layers.Layer):
    
    def __init__(self, config, input_embeddings, **kwargs):
        super().__init__(**kwargs)
        self.predictions = TFBertLMPredictionHead(config, input_embeddings, name='predictions')
    
    def call(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores



class TFBertNSPHead(tf.keras.layers.Layer):
    
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.seq_relationship = tf.keras.layers.Dense(2, kernel_initializer=get_initializer(config.initializer_range), name='seq_relationship')
    
    def call(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score



@keras_serializable
class TFBertMainLayer(tf.keras.layers.Layer):
    config_class = BertConfig
    
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.num_hidden_layers = config.num_hidden_layers
        self.initializer_range = config.initializer_range
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.return_dict = config.use_return_dict
        self.embeddings = TFBertEmbeddings(config, name='embeddings')
        self.encoder = TFBertEncoder(config, name='encoder')
        self.pooler = TFBertPooler(config, name='pooler')
    
    def get_input_embeddings(self):
        return self.embeddings
    
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value
        self.embeddings.vocab_size = value.shape[0]
    
    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        raise NotImplementedError
    
    def call(self, inputs, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, output_attentions=None, output_hidden_states=None, return_dict=None, training=False):
        if isinstance(inputs, (tuple, list)):
            input_ids = inputs[0]
            attention_mask = (inputs[1] if len(inputs) > 1 else attention_mask)
            token_type_ids = (inputs[2] if len(inputs) > 2 else token_type_ids)
            position_ids = (inputs[3] if len(inputs) > 3 else position_ids)
            head_mask = (inputs[4] if len(inputs) > 4 else head_mask)
            inputs_embeds = (inputs[5] if len(inputs) > 5 else inputs_embeds)
            output_attentions = (inputs[6] if len(inputs) > 6 else output_attentions)
            output_hidden_states = (inputs[7] if len(inputs) > 7 else output_hidden_states)
            return_dict = (inputs[8] if len(inputs) > 8 else return_dict)
            assert len(inputs) <= 9, 'Too many inputs.'
        elif isinstance(inputs, (dict, BatchEncoding)):
            input_ids = inputs.get('input_ids')
            attention_mask = inputs.get('attention_mask', attention_mask)
            token_type_ids = inputs.get('token_type_ids', token_type_ids)
            position_ids = inputs.get('position_ids', position_ids)
            head_mask = inputs.get('head_mask', head_mask)
            inputs_embeds = inputs.get('inputs_embeds', inputs_embeds)
            output_attentions = inputs.get('output_attentions', output_attentions)
            output_hidden_states = inputs.get('output_hidden_states', output_hidden_states)
            return_dict = inputs.get('return_dict', return_dict)
            assert len(inputs) <= 9, 'Too many inputs.'
        else:
            input_ids = inputs
        output_attentions = (output_attentions if output_attentions is not None else self.output_attentions)
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.output_hidden_states)
        return_dict = (return_dict if return_dict is not None else self.return_dict)
        if (input_ids is not None and inputs_embeds is not None):
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        if attention_mask is None:
            attention_mask = tf.fill(input_shape, 1)
        if token_type_ids is None:
            token_type_ids = tf.fill(input_shape, 0)
        embedding_output = self.embeddings(input_ids, position_ids, token_type_ids, inputs_embeds, training=training)
        extended_attention_mask = attention_mask[:, tf.newaxis, tf.newaxis, :]
        extended_attention_mask = tf.cast(extended_attention_mask, embedding_output.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.num_hidden_layers
        encoder_outputs = self.encoder(embedding_output, extended_attention_mask, head_mask, output_attentions, output_hidden_states, return_dict, training=training)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]
        return TFBaseModelOutputWithPooling(last_hidden_state=sequence_output, pooler_output=pooled_output, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)



class TFBertPreTrainedModel(TFPreTrainedModel):
    """An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """
    config_class = BertConfig
    base_model_prefix = 'bert'



@dataclass
class TFBertForPreTrainingOutput(ModelOutput):
    """
    Output type of :class:`~transformers.TFBertForPreTrainingModel`.

    Args:
        prediction_logits (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_logits (:obj:`tf.Tensor` of shape :obj:`(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False
            continuation before SoftMax).
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
    prediction_logits: tf.Tensor = None
    seq_relationship_logits: tf.Tensor = None
    hidden_states: Optional[Tuple[tf.Tensor]] = None
    attentions: Optional[Tuple[tf.Tensor]] = None

BERT_START_DOCSTRING = '\n\n    This model inherits from :class:`~transformers.TFPreTrainedModel`. Check the superclass documentation for the\n    generic methods the library implements for all its model (such as downloading or saving, resizing the input\n    embeddings, pruning heads etc.)\n\n    This model is also a `tf.keras.Model <https://www.tensorflow.org/api_docs/python/tf/keras/Model>`__ subclass.\n    Use it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general\n    usage and behavior.\n\n    .. note::\n\n        TF 2.0 models accepts two formats as inputs:\n\n        - having all inputs as keyword arguments (like PyTorch models), or\n        - having all inputs as a list, tuple or dict in the first positional arguments.\n\n        This second option is useful when using :meth:`tf.keras.Model.fit` method which currently requires having\n        all the tensors in the first argument of the model call function: :obj:`model(inputs)`.\n\n        If you choose this second option, there are three possibilities you can use to gather all the input Tensors\n        in the first positional argument :\n\n        - a single Tensor with :obj:`input_ids` only and nothing else: :obj:`model(inputs_ids)`\n        - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:\n          :obj:`model([input_ids, attention_mask])` or :obj:`model([input_ids, attention_mask, token_type_ids])`\n        - a dictionary with one or several input Tensors associated to the input names given in the docstring:\n          :obj:`model({"input_ids": input_ids, "token_type_ids": token_type_ids})`\n\n    Args:\n        config (:class:`~transformers.BertConfig`): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the configuration.\n            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.\n'
BERT_INPUTS_DOCSTRING = "\n    Args:\n        input_ids (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`({0})`):\n            Indices of input sequence tokens in the vocabulary.\n\n            Indices can be obtained using :class:`~transformers.BertTokenizer`.\n            See :func:`transformers.PreTrainedTokenizer.__call__` and\n            :func:`transformers.PreTrainedTokenizer.encode` for details.\n\n            `What are input IDs? <../glossary.html#input-ids>`__\n        attention_mask (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`({0})`, `optional`):\n            Mask to avoid performing attention on padding token indices.\n            Mask values selected in ``[0, 1]``:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            `What are attention masks? <../glossary.html#attention-mask>`__\n        token_type_ids (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`({0})`, `optional`):\n            Segment token indices to indicate first and second portions of the inputs.\n            Indices are selected in ``[0, 1]``:\n\n            - 0 corresponds to a `sentence A` token,\n            - 1 corresponds to a `sentence B` token.\n\n            `What are token type IDs? <../glossary.html#token-type-ids>`__\n        position_ids (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`({0})`, `optional`):\n            Indices of positions of each input sequence tokens in the position embeddings.\n            Selected in the range ``[0, config.max_position_embeddings - 1]``.\n\n            `What are position IDs? <../glossary.html#position-ids>`__\n        head_mask (:obj:`Numpy array` or :obj:`tf.Tensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):\n            Mask to nullify selected heads of the self-attention modules.\n            Mask values selected in ``[0, 1]``:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        inputs_embeds (:obj:`tf.Tensor` of shape :obj:`({0}, hidden_size)`, `optional`):\n            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.\n            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated\n            vectors than the model's internal embedding lookup matrix.\n        output_attentions (:obj:`bool`, `optional`):\n            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned\n            tensors for more detail.\n        output_hidden_states (:obj:`bool`, `optional`):\n            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for\n            more detail.\n        return_dict (:obj:`bool`, `optional`):\n            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.\n        training (:obj:`bool`, `optional`, defaults to :obj:`False`):\n            Whether or not to use the model in training mode (some modules like dropout modules have different\n            behaviors between training and evaluation).\n"


@add_start_docstrings('The bare Bert Model transformer outputing raw hidden-states without any specific head on top.', BERT_START_DOCSTRING)
class TFBertModel(TFBertPreTrainedModel):
    
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.bert = TFBertMainLayer(config, name='bert')
    
    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint='bert-base-cased', output_type=TFBaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def call(self, inputs, **kwargs):
        outputs = self.bert(inputs, **kwargs)
        return outputs



@add_start_docstrings('Bert Model with two heads on top as done during the pre-training:\n    a `masked language modeling` head and a `next sentence prediction (classification)` head. ', BERT_START_DOCSTRING)
class TFBertForPreTraining(TFBertPreTrainedModel):
    
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.bert = TFBertMainLayer(config, name='bert')
        self.nsp = TFBertNSPHead(config, name='nsp___cls')
        self.mlm = TFBertMLMHead(config, self.bert.embeddings, name='mlm___cls')
    
    def get_output_embeddings(self):
        return self.bert.embeddings
    
    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=TFBertForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, inputs, **kwargs):
        """
        Return:

        Examples::

            >>> import tensorflow as tf
            >>> from transformers import BertTokenizer, TFBertForPreTraining

            >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            >>> model = TFBertForPreTraining.from_pretrained('bert-base-uncased')
            >>> input_ids = tf.constant(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True))[None, :]  # Batch size 1
            >>> outputs = model(input_ids)
            >>> prediction_scores, seq_relationship_scores = outputs[:2]

        """
        return_dict = kwargs.get('return_dict')
        return_dict = (return_dict if return_dict is not None else self.bert.return_dict)
        outputs = self.bert(inputs, **kwargs)
        (sequence_output, pooled_output) = outputs[:2]
        prediction_scores = self.mlm(sequence_output, training=kwargs.get('training', False))
        seq_relationship_score = self.nsp(pooled_output)
        if not return_dict:
            return (prediction_scores, seq_relationship_score) + outputs[2:]
        return TFBertForPreTrainingOutput(prediction_logits=prediction_scores, seq_relationship_logits=seq_relationship_score, hidden_states=outputs.hidden_states, attentions=outputs.attentions)



@add_start_docstrings('Bert Model with a `language modeling` head on top. ', BERT_START_DOCSTRING)
class TFBertForMaskedLM(TFBertPreTrainedModel, TFMaskedLanguageModelingLoss):
    authorized_unexpected_keys = ['pooler']
    authorized_missing_keys = ['pooler']
    
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        if config.is_decoder:
            logger.warning('If you want to use `TFBertForMaskedLM` make sure `config.is_decoder=False` for bi-directional self-attention.')
        self.bert = TFBertMainLayer(config, name='bert')
        self.mlm = TFBertMLMHead(config, self.bert.embeddings, name='mlm___cls')
    
    def get_output_embeddings(self):
        return self.bert.embeddings
    
    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint='bert-base-cased', output_type=TFMaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, inputs=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, output_attentions=None, output_hidden_states=None, return_dict=None, labels=None, training=False):
        """
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        """
        return_dict = (return_dict if return_dict is not None else self.bert.return_dict)
        if isinstance(inputs, (tuple, list)):
            labels = (inputs[9] if len(inputs) > 9 else labels)
            if len(inputs) > 9:
                inputs = inputs[:9]
        elif isinstance(inputs, (dict, BatchEncoding)):
            labels = inputs.pop('labels', labels)
        outputs = self.bert(inputs, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output = outputs[0]
        prediction_scores = self.mlm(sequence_output, training=training)
        loss = (None if labels is None else self.compute_loss(labels, prediction_scores))
        if not return_dict:
            output = (prediction_scores, ) + outputs[2:]
            return ((loss, ) + output if loss is not None else output)
        return TFMaskedLMOutput(loss=loss, logits=prediction_scores, hidden_states=outputs.hidden_states, attentions=outputs.attentions)



class TFBertLMHeadModel(TFBertPreTrainedModel, TFCausalLanguageModelingLoss):
    authorized_unexpected_keys = ['pooler']
    authorized_missing_keys = ['pooler']
    
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        if not config.is_decoder:
            logger.warning('If you want to use `TFBertLMHeadModel` as a standalone, add `is_decoder=True.`')
        self.bert = TFBertMainLayer(config, name='bert')
        self.mlm = TFBertMLMHead(config, self.bert.embeddings, name='mlm___cls')
    
    def get_output_embeddings(self):
        return self.bert.embeddings
    
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint='bert-base-cased', output_type=TFCausalLMOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, inputs=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, output_attentions=None, output_hidden_states=None, return_dict=None, labels=None, training=False):
        """
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the cross entropy classification loss.
            Indices should be in ``[0, ..., config.vocab_size - 1]``.
        """
        return_dict = (return_dict if return_dict is not None else self.bert.return_dict)
        if isinstance(inputs, (tuple, list)):
            labels = (inputs[9] if len(inputs) > 9 else labels)
            if len(inputs) > 9:
                inputs = inputs[:9]
        elif isinstance(inputs, (dict, BatchEncoding)):
            labels = inputs.pop('labels', labels)
        outputs = self.bert(inputs, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output = outputs[0]
        logits = self.mlm(sequence_output, training=training)
        loss = None
        if labels is not None:
            logits = logits[:, :-1]
            labels = labels[:, 1:]
            loss = self.compute_loss(labels, logits)
        if not return_dict:
            output = (logits, ) + outputs[2:]
            return ((loss, ) + output if loss is not None else output)
        return TFCausalLMOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)



@add_start_docstrings('Bert Model with a `next sentence prediction (classification)` head on top. ', BERT_START_DOCSTRING)
class TFBertForNextSentencePrediction(TFBertPreTrainedModel):
    
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.bert = TFBertMainLayer(config, name='bert')
        self.nsp = TFBertNSPHead(config, name='nsp___cls')
    
    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=TFNextSentencePredictorOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, inputs, **kwargs):
        """
        Return:

        Examples::

            >>> import tensorflow as tf
            >>> from transformers import BertTokenizer, TFBertForNextSentencePrediction

            >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            >>> model = TFBertForNextSentencePrediction.from_pretrained('bert-base-uncased')

            >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
            >>> next_sentence = "The sky is blue due to the shorter wavelength of blue light."
            >>> encoding = tokenizer(prompt, next_sentence, return_tensors='tf')

            >>> logits = model(encoding['input_ids'], token_type_ids=encoding['token_type_ids'])[0]
            >>> assert logits[0][0] < logits[0][1] # the next sentence was random
        """
        return_dict = kwargs.get('return_dict')
        return_dict = (return_dict if return_dict is not None else self.bert.return_dict)
        outputs = self.bert(inputs, **kwargs)
        pooled_output = outputs[1]
        seq_relationship_score = self.nsp(pooled_output)
        if not return_dict:
            return (seq_relationship_score, ) + outputs[2:]
        return TFNextSentencePredictorOutput(logits=seq_relationship_score, hidden_states=outputs.hidden_states, attentions=outputs.attentions)



@add_start_docstrings('Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of\n    the pooled output) e.g. for GLUE tasks. ', BERT_START_DOCSTRING)
class TFBertForSequenceClassification(TFBertPreTrainedModel, TFSequenceClassificationLoss):
    
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.bert = TFBertMainLayer(config, name='bert')
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name='classifier')
    
    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint='bert-base-cased', output_type=TFSequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, inputs=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, output_attentions=None, output_hidden_states=None, return_dict=None, labels=None, training=False):
        """
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = (return_dict if return_dict is not None else self.bert.return_dict)
        if isinstance(inputs, (tuple, list)):
            labels = (inputs[9] if len(inputs) > 9 else labels)
            if len(inputs) > 9:
                inputs = inputs[:9]
        elif isinstance(inputs, (dict, BatchEncoding)):
            labels = inputs.pop('labels', labels)
        outputs = self.bert(inputs, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output, training=training)
        logits = self.classifier(pooled_output)
        loss = (None if labels is None else self.compute_loss(labels, logits))
        if not return_dict:
            output = (logits, ) + outputs[2:]
            return ((loss, ) + output if loss is not None else output)
        return TFSequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)



@add_start_docstrings('Bert Model with a multiple choice classification head on top (a linear layer on top of\n    the pooled output and a softmax) e.g. for RocStories/SWAG tasks. ', BERT_START_DOCSTRING)
class TFBertForMultipleChoice(TFBertPreTrainedModel, TFMultipleChoiceLoss):
    
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.bert = TFBertMainLayer(config, name='bert')
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(1, kernel_initializer=get_initializer(config.initializer_range), name='classifier')
    
    @property
    def dummy_inputs(self):
        """Dummy inputs to build the network.

        Returns:
            tf.Tensor with dummy inputs
        """
        return {'input_ids': tf.constant(MULTIPLE_CHOICE_DUMMY_INPUTS)}
    
    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format('batch_size, num_choices, sequence_length'))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint='bert-base-cased', output_type=TFMultipleChoiceModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, inputs, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, output_attentions=None, output_hidden_states=None, return_dict=None, labels=None, training=False):
        """
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss.
            Indices should be in ``[0, ..., num_choices]`` where :obj:`num_choices` is the size of the second dimension
            of the input tensors. (See :obj:`input_ids` above)
        """
        if isinstance(inputs, (tuple, list)):
            input_ids = inputs[0]
            attention_mask = (inputs[1] if len(inputs) > 1 else attention_mask)
            token_type_ids = (inputs[2] if len(inputs) > 2 else token_type_ids)
            position_ids = (inputs[3] if len(inputs) > 3 else position_ids)
            head_mask = (inputs[4] if len(inputs) > 4 else head_mask)
            inputs_embeds = (inputs[5] if len(inputs) > 5 else inputs_embeds)
            output_attentions = (inputs[6] if len(inputs) > 6 else output_attentions)
            output_hidden_states = (inputs[7] if len(inputs) > 7 else output_hidden_states)
            return_dict = (inputs[8] if len(inputs) > 8 else return_dict)
            labels = (inputs[9] if len(inputs) > 9 else labels)
            assert len(inputs) <= 10, 'Too many inputs.'
        elif isinstance(inputs, (dict, BatchEncoding)):
            input_ids = inputs.get('input_ids')
            attention_mask = inputs.get('attention_mask', attention_mask)
            token_type_ids = inputs.get('token_type_ids', token_type_ids)
            position_ids = inputs.get('position_ids', position_ids)
            head_mask = inputs.get('head_mask', head_mask)
            inputs_embeds = inputs.get('inputs_embeds', inputs_embeds)
            output_attentions = inputs.get('output_attentions', output_attentions)
            output_hidden_states = inputs.get('output_hidden_states', output_hidden_states)
            return_dict = inputs.get('return_dict', return_dict)
            labels = inputs.get('labels', labels)
            assert len(inputs) <= 10, 'Too many inputs.'
        else:
            input_ids = inputs
        return_dict = (return_dict if return_dict is not None else self.bert.return_dict)
        if input_ids is not None:
            num_choices = shape_list(input_ids)[1]
            seq_length = shape_list(input_ids)[2]
        else:
            num_choices = shape_list(inputs_embeds)[1]
            seq_length = shape_list(inputs_embeds)[2]
        flat_input_ids = (tf.reshape(input_ids, (-1, seq_length)) if input_ids is not None else None)
        flat_attention_mask = (tf.reshape(attention_mask, (-1, seq_length)) if attention_mask is not None else None)
        flat_token_type_ids = (tf.reshape(token_type_ids, (-1, seq_length)) if token_type_ids is not None else None)
        flat_position_ids = (tf.reshape(position_ids, (-1, seq_length)) if position_ids is not None else None)
        flat_inputs_embeds = (tf.reshape(inputs_embeds, (-1, seq_length, shape_list(inputs_embeds)[3])) if inputs_embeds is not None else None)
        outputs = self.bert(flat_input_ids, flat_attention_mask, flat_token_type_ids, flat_position_ids, head_mask, flat_inputs_embeds, output_attentions, output_hidden_states, return_dict=return_dict, training=training)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output, training=training)
        logits = self.classifier(pooled_output)
        reshaped_logits = tf.reshape(logits, (-1, num_choices))
        loss = (None if labels is None else self.compute_loss(labels, reshaped_logits))
        if not return_dict:
            output = (reshaped_logits, ) + outputs[2:]
            return ((loss, ) + output if loss is not None else output)
        return TFMultipleChoiceModelOutput(loss=loss, logits=reshaped_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)



@add_start_docstrings('Bert Model with a token classification head on top (a linear layer on top of\n    the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. ', BERT_START_DOCSTRING)
class TFBertForTokenClassification(TFBertPreTrainedModel, TFTokenClassificationLoss):
    authorized_unexpected_keys = ['pooler']
    authorized_missing_keys = ['pooler']
    
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.bert = TFBertMainLayer(config, name='bert')
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)
        self.classifier = tf.keras.layers.Dense(config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name='classifier')
    
    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint='bert-base-cased', output_type=TFTokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, inputs=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, output_attentions=None, output_hidden_states=None, return_dict=None, labels=None, training=False):
        """
        labels (:obj:`tf.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
        """
        return_dict = (return_dict if return_dict is not None else self.bert.return_dict)
        if isinstance(inputs, (tuple, list)):
            labels = (inputs[9] if len(inputs) > 9 else labels)
            if len(inputs) > 9:
                inputs = inputs[:9]
        elif isinstance(inputs, (dict, BatchEncoding)):
            labels = inputs.pop('labels', labels)
        outputs = self.bert(inputs, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output, training=training)
        logits = self.classifier(sequence_output)
        loss = (None if labels is None else self.compute_loss(labels, logits))
        if not return_dict:
            output = (logits, ) + outputs[2:]
            return ((loss, ) + output if loss is not None else output)
        return TFTokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)



@add_start_docstrings('Bert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear\n    layer on top of the hidden-states output to compute `span start logits` and `span end logits`). ', BERT_START_DOCSTRING)
class TFBertForQuestionAnswering(TFBertPreTrainedModel, TFQuestionAnsweringLoss):
    authorized_unexpected_keys = ['pooler']
    authorized_missing_keys = ['pooler']
    
    def __init__(self, config, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.num_labels = config.num_labels
        self.bert = TFBertMainLayer(config, name='bert')
        self.qa_outputs = tf.keras.layers.Dense(config.num_labels, kernel_initializer=get_initializer(config.initializer_range), name='qa_outputs')
    
    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint='bert-base-cased', output_type=TFQuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def call(self, inputs=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, output_attentions=None, output_hidden_states=None, return_dict=None, start_positions=None, end_positions=None, training=False):
        """
        start_positions (:obj:`tf.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        end_positions (:obj:`tf.Tensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`).
            Position outside of the sequence are not taken into account for computing the loss.
        """
        return_dict = (return_dict if return_dict is not None else self.bert.return_dict)
        if isinstance(inputs, (tuple, list)):
            start_positions = (inputs[9] if len(inputs) > 9 else start_positions)
            end_positions = (inputs[10] if len(inputs) > 10 else end_positions)
            if len(inputs) > 9:
                inputs = inputs[:9]
        elif isinstance(inputs, (dict, BatchEncoding)):
            start_positions = inputs.pop('start_positions', start_positions)
            end_positions = inputs.pop('end_positions', start_positions)
        outputs = self.bert(inputs, attention_mask=attention_mask, token_type_ids=token_type_ids, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, training=training)
        sequence_output = outputs[0]
        logits = self.qa_outputs(sequence_output)
        (start_logits, end_logits) = tf.split(logits, 2, axis=-1)
        start_logits = tf.squeeze(start_logits, axis=-1)
        end_logits = tf.squeeze(end_logits, axis=-1)
        loss = None
        if (start_positions is not None and end_positions is not None):
            labels = {'start_position': start_positions}
            labels['end_position'] = end_positions
            loss = self.compute_loss(labels, (start_logits, end_logits))
        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((loss, ) + output if loss is not None else output)
        return TFQuestionAnsweringModelOutput(loss=loss, start_logits=start_logits, end_logits=end_logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)


