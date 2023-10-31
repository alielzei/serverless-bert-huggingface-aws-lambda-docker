"""PyTorch BERT model specific for generation. """

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from .configuration_bert_generation import BertGenerationConfig
from .file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_callable, replace_return_docstrings
from .modeling_bert import BertEncoder
from .modeling_outputs import BaseModelOutput, CausalLMOutput
from .modeling_utils import PreTrainedModel
from .utils import logging
logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = 'BertGenerationConfig'
_TOKENIZER_FOR_DOC = 'BertGenerationTokenizer'

def load_tf_weights_in_bert_generation(model, tf_hub_path, model_class, is_encoder_named_decoder=False, is_encoder=False):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.modeling_bert_generation.load_tf_weights_in_bert_generation', 'load_tf_weights_in_bert_generation(model, tf_hub_path, model_class, is_encoder_named_decoder=False, is_encoder=False)', {'logger': logger, 'torch': torch, 'model': model, 'tf_hub_path': tf_hub_path, 'model_class': model_class, 'is_encoder_named_decoder': is_encoder_named_decoder, 'is_encoder': is_encoder}, 1)


class BertGenerationEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.LayerNorm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.register_buffer('position_ids', torch.arange(config.max_position_embeddings).expand((1, -1)))
    
    def forward(self, input_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        seq_length = input_shape[1]
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = inputs_embeds + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings



class BertGenerationPreTrainedModel(PreTrainedModel):
    """An abstract class to handle weights initialization and
    a simple interface for downloading and loading pretrained models.
    """
    config_class = BertGenerationConfig
    base_model_prefix = 'bert'
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

BERT_GENERATION_START_DOCSTRING = '\n\n    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic\n    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,\n    pruning heads etc.)\n\n    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__ subclass.\n    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general\n    usage and behavior.\n\n    Parameters:\n        config (:class:`~transformers.BertGenerationConfig`): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the configuration.\n            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.\n'
BERT_GENERATION_INPUTS_DOCSTRING = "\n    Args:\n        input_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`):\n            Indices of input sequence tokens in the vocabulary.\n\n            Indices can be obtained using :class:`~transformers.BertGenerationTokenizer`.\n            See :meth:`transformers.PreTrainedTokenizer.__call__` and\n            :meth:`transformers.PreTrainedTokenizer.encode` for details.\n\n            `What are input IDs? <../glossary.html#input-ids>`__\n        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`({0})`, `optional`):\n            Mask to avoid performing attention on padding token indices.\n            Mask values selected in ``[0, 1]``:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            `What are attention masks? <../glossary.html#attention-mask>`__\n        position_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):\n            Indices of positions of each input sequence tokens in the position embeddings.\n            Selected in the range ``[0, config.max_position_embeddings - 1]``.\n\n            `What are position IDs? <../glossary.html#position-ids>`_\n        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):\n            Mask to nullify selected heads of the self-attention modules.\n            Mask values selected in ``[0, 1]``:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`({0}, hidden_size)`, `optional`):\n            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.\n            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated\n            vectors than the model's internal embedding lookup matrix.\n        output_attentions (:obj:`bool`, `optional`):\n            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned\n            tensors for more detail.\n        output_hidden_states (:obj:`bool`, `optional`):\n            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for\n            more detail.\n        return_dict (:obj:`bool`, `optional`):\n            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.\n"


@add_start_docstrings('The bare BertGeneration model transformer outputting raw hidden-states without any specific head on top.', BERT_GENERATION_START_DOCSTRING)
class BertGenerationEncoder(BertGenerationPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need
    <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
    Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    This model should be used when leveraging Bert or Roberta checkpoints for the
    :class:`~transformers.EncoderDecoderModel` class as described in `Leveraging Pre-trained Checkpoints for Sequence
    Generation Tasks <https://arxiv.org/abs/1907.12461>`__ by Sascha Rothe, Shashi Narayan, and Aliaksei Severyn.

    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`.
    To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an
    :obj:`encoder_hidden_states` is then expected as an input to the forward pass.
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embeddings = BertGenerationEmbeddings(config)
        self.encoder = BertEncoder(config)
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
    
    @add_start_docstrings_to_callable(BERT_GENERATION_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint='google/bert_for_seq_generation_L-24_bbc_encoder', output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=None, attention_mask=None, position_ids=None, head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        """
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
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
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)
        if (self.config.is_decoder and encoder_hidden_states is not None):
            (encoder_batch_size, encoder_sequence_length, _) = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids, inputs_embeds=inputs_embeds)
        encoder_outputs = self.encoder(embedding_output, attention_mask=extended_attention_mask, head_mask=head_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_extended_attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = encoder_outputs[0]
        if not return_dict:
            return (sequence_output, ) + encoder_outputs[1:]
        return BaseModelOutput(last_hidden_state=sequence_output, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)



class BertGenerationOnlyLMHead(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias
    
    def forward(self, hidden_states):
        logits = self.decoder(hidden_states)
        return logits



@add_start_docstrings('BertGeneration Model with a `language modeling` head on top for CLM fine-tuning. ', BERT_GENERATION_START_DOCSTRING)
class BertGenerationDecoder(BertGenerationPreTrainedModel):
    
    def __init__(self, config):
        super().__init__(config)
        if not config.is_decoder:
            logger.warn('If you want to use `BertGenerationDecoder` as a standalone, add `is_decoder=True.`')
        self.bert = BertGenerationEncoder(config)
        self.lm_head = BertGenerationOnlyLMHead(config)
        self.init_weights()
    
    def get_output_embeddings(self):
        return self.lm_head.decoder
    
    @add_start_docstrings_to_callable(BERT_GENERATION_INPUTS_DOCSTRING.format('batch_size, sequence_length'))
    @replace_return_docstrings(output_type=CausalLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=None, attention_mask=None, position_ids=None, head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        """
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the left-to-right language modeling loss (next word prediction).
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with
            labels in ``[0, ..., config.vocab_size]``

        Returns:

        Example::

            >>> from transformers import BertGenerationTokenizer, BertGenerationDecoder, BertGenerationConfig
            >>> import torch

            >>> tokenizer = BertGenerationTokenizer.from_pretrained('google/bert_for_seq_generation_L-24_bbc_encoder')
            >>> config = BertGenerationConfig.from_pretrained("google/bert_for_seq_generation_L-24_bbc_encoder")
            >>> config.is_decoder = True
            >>> model = BertGenerationDecoder.from_pretrained('google/bert_for_seq_generation_L-24_bbc_encoder', config=config, return_dict=True)

            >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
            >>> outputs = model(**inputs)

            >>> prediction_logits = outputs.logits
        """
        return_dict = (return_dict if return_dict is not None else self.config.use_return_dict)
        outputs = self.bert(input_ids, attention_mask=attention_mask, position_ids=position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)
        lm_loss = None
        if labels is not None:
            shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        if not return_dict:
            output = (prediction_scores, ) + outputs[1:]
            return ((lm_loss, ) + output if lm_loss is not None else output)
        return CausalLMOutput(loss=lm_loss, logits=prediction_scores, hidden_states=outputs.hidden_states, attentions=outputs.attentions)
    
    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)
        return {'input_ids': input_ids, 'attention_mask': attention_mask}


