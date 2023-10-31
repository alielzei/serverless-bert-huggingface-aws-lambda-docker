""" PyTorch Flaubert model, based on XLM. """

import random
import torch
from torch.nn import functional as F
from .configuration_flaubert import FlaubertConfig
from .file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_callable
from .modeling_outputs import BaseModelOutput
from .modeling_xlm import XLMForMultipleChoice, XLMForQuestionAnswering, XLMForQuestionAnsweringSimple, XLMForSequenceClassification, XLMForTokenClassification, XLMModel, XLMWithLMHeadModel, get_masks
from .utils import logging
logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = 'FlaubertConfig'
_TOKENIZER_FOR_DOC = 'FlaubertTokenizer'
FLAUBERT_PRETRAINED_MODEL_ARCHIVE_LIST = ['flaubert/flaubert_small_cased', 'flaubert/flaubert_base_uncased', 'flaubert/flaubert_base_cased', 'flaubert/flaubert_large_cased']
FLAUBERT_START_DOCSTRING = '\n\n    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic\n    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,\n    pruning heads etc.)\n\n    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__ subclass.\n    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general\n    usage and behavior.\n\n    Parameters:\n        config (:class:`~transformers.FlaubertConfig`): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the configuration.\n            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.\n'
FLAUBERT_INPUTS_DOCSTRING = "\n    Args:\n        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):\n            Indices of input sequence tokens in the vocabulary.\n\n            Indices can be obtained using :class:`~transformers.FlaubertTokenizer`.\n            See :meth:`transformers.PreTrainedTokenizer.encode` and\n            :meth:`transformers.PreTrainedTokenizer.__call__` for details.\n\n            `What are input IDs? <../glossary.html#input-ids>`__\n        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):\n            Mask to avoid performing attention on padding token indices.\n            Mask values selected in ``[0, 1]``:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            `What are attention masks? <../glossary.html#attention-mask>`__\n        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):\n            Segment token indices to indicate first and second portions of the inputs.\n            Indices are selected in ``[0, 1]``:\n\n            - 0 corresponds to a `sentence A` token,\n            - 1 corresponds to a `sentence B` token.\n\n            `What are token type IDs? <../glossary.html#token-type-ids>`_\n        position_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):\n            Indices of positions of each input sequence tokens in the position embeddings.\n            Selected in the range ``[0, config.max_position_embeddings - 1]``.\n\n            `What are position IDs? <../glossary.html#position-ids>`_\n        lengths (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):\n            Length of each sentence that can be used to avoid performing attention on padding token indices.\n            You can also use :obj:`attention_mask` for the same result (see above), kept here for compatbility.\n            Indices selected in ``[0, ..., input_ids.size(-1)]``:\n        cache (:obj:`Dict[str, torch.FloatTensor]`, `optional`):\n            Dictionary strings to ``torch.FloatTensor`` that contains precomputed\n            hidden-states (key and values in the attention blocks) as computed by the model\n            (see :obj:`cache` output below). Can be used to speed up sequential decoding.\n            The dictionary object will be modified in-place during the forward pass to add newly computed hidden-states.\n        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):\n            Mask to nullify selected heads of the self-attention modules.\n            Mask values selected in ``[0, 1]``:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):\n            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.\n            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated\n            vectors than the model's internal embedding lookup matrix.\n        output_attentions (:obj:`bool`, `optional`):\n            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned\n            tensors for more detail.\n        output_hidden_states (:obj:`bool`, `optional`):\n            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for\n            more detail.\n        return_dict (:obj:`bool`, `optional`):\n            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.\n"


@add_start_docstrings('The bare Flaubert Model transformer outputting raw hidden-states without any specific head on top.', FLAUBERT_START_DOCSTRING)
class FlaubertModel(XLMModel):
    config_class = FlaubertConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.layerdrop = getattr(config, 'layerdrop', 0.0)
        self.pre_norm = getattr(config, 'pre_norm', False)
    
    @add_start_docstrings_to_callable(FLAUBERT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(tokenizer_class=_TOKENIZER_FOR_DOC, checkpoint='flaubert/flaubert_base_cased', output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_ids=None, attention_mask=None, langs=None, token_type_ids=None, position_ids=None, lengths=None, cache=None, head_mask=None, inputs_embeds=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        output_attentions = (output_attentions if output_attentions is not None else self.config.output_attentions)
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict = (return_dict if return_dict is not None else self.config.use_return_dict)
        if input_ids is not None:
            (bs, slen) = input_ids.size()
        else:
            (bs, slen) = inputs_embeds.size()[:-1]
        device = (input_ids.device if input_ids is not None else inputs_embeds.device)
        if lengths is None:
            if input_ids is not None:
                lengths = (input_ids != self.pad_index).sum(dim=1).long()
            else:
                lengths = torch.tensor([slen] * bs, device=device)
        assert lengths.size(0) == bs
        assert lengths.max().item() <= slen
        (mask, attn_mask) = get_masks(slen, lengths, self.causal, padding_mask=attention_mask)
        if position_ids is None:
            position_ids = torch.arange(slen, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand((bs, slen))
        else:
            assert position_ids.size() == (bs, slen)
        if langs is not None:
            assert langs.size() == (bs, slen)
        head_mask = self.get_head_mask(head_mask, self.config.n_layers)
        if (cache is not None and input_ids is not None):
            _slen = slen - cache['slen']
            input_ids = input_ids[:, -_slen:]
            position_ids = position_ids[:, -_slen:]
            if langs is not None:
                langs = langs[:, -_slen:]
            mask = mask[:, -_slen:]
            attn_mask = attn_mask[:, -_slen:]
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
        tensor = inputs_embeds + self.position_embeddings(position_ids).expand_as(inputs_embeds)
        if (langs is not None and self.use_lang_emb and self.config.n_langs > 1):
            tensor = tensor + self.lang_embeddings(langs)
        if token_type_ids is not None:
            tensor = tensor + self.embeddings(token_type_ids)
        tensor = self.layer_norm_emb(tensor)
        tensor = F.dropout(tensor, p=self.dropout, training=self.training)
        tensor *= mask.unsqueeze(-1).to(tensor.dtype)
        hidden_states = (() if output_hidden_states else None)
        attentions = (() if output_attentions else None)
        for i in range(self.n_layers):
            dropout_probability = random.uniform(0, 1)
            if (self.training and dropout_probability < self.layerdrop):
                continue
            if output_hidden_states:
                hidden_states = hidden_states + (tensor, )
            if not self.pre_norm:
                attn_outputs = self.attentions[i](tensor, attn_mask, cache=cache, head_mask=head_mask[i], output_attentions=output_attentions)
                attn = attn_outputs[0]
                if output_attentions:
                    attentions = attentions + (attn_outputs[1], )
                attn = F.dropout(attn, p=self.dropout, training=self.training)
                tensor = tensor + attn
                tensor = self.layer_norm1[i](tensor)
            else:
                tensor_normalized = self.layer_norm1[i](tensor)
                attn_outputs = self.attentions[i](tensor_normalized, attn_mask, cache=cache, head_mask=head_mask[i])
                attn = attn_outputs[0]
                if output_attentions:
                    attentions = attentions + (attn_outputs[1], )
                attn = F.dropout(attn, p=self.dropout, training=self.training)
                tensor = tensor + attn
            if not self.pre_norm:
                tensor = tensor + self.ffns[i](tensor)
                tensor = self.layer_norm2[i](tensor)
            else:
                tensor_normalized = self.layer_norm2[i](tensor)
                tensor = tensor + self.ffns[i](tensor_normalized)
            tensor *= mask.unsqueeze(-1).to(tensor.dtype)
        if output_hidden_states:
            hidden_states = hidden_states + (tensor, )
        if cache is not None:
            cache['slen'] += tensor.size(1)
        if not return_dict:
            return tuple((v for v in [tensor, hidden_states, attentions] if v is not None))
        return BaseModelOutput(last_hidden_state=tensor, hidden_states=hidden_states, attentions=attentions)



@add_start_docstrings('The Flaubert Model transformer with a language modeling head on top\n    (linear layer with weights tied to the input embeddings). ', FLAUBERT_START_DOCSTRING)
class FlaubertWithLMHeadModel(XLMWithLMHeadModel):
    """
    This class overrides :class:`~transformers.XLMWithLMHeadModel`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """
    config_class = FlaubertConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.transformer = FlaubertModel(config)
        self.init_weights()



@add_start_docstrings('Flaubert Model with a sequence classification/regression head on top (a linear layer on top of\n    the pooled output) e.g. for GLUE tasks. ', FLAUBERT_START_DOCSTRING)
class FlaubertForSequenceClassification(XLMForSequenceClassification):
    """
    This class overrides :class:`~transformers.XLMForSequenceClassification`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """
    config_class = FlaubertConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.transformer = FlaubertModel(config)
        self.init_weights()



@add_start_docstrings('Flaubert Model with a token classification head on top (a linear layer on top of\n    the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. ', FLAUBERT_START_DOCSTRING)
class FlaubertForTokenClassification(XLMForTokenClassification):
    """
    This class overrides :class:`~transformers.XLMForTokenClassification`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """
    config_class = FlaubertConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.transformer = FlaubertModel(config)
        self.init_weights()



@add_start_docstrings('Flaubert Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layers on top of\n    the hidden-states output to compute `span start logits` and `span end logits`). ', FLAUBERT_START_DOCSTRING)
class FlaubertForQuestionAnsweringSimple(XLMForQuestionAnsweringSimple):
    """
    This class overrides :class:`~transformers.XLMForQuestionAnsweringSimple`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """
    config_class = FlaubertConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.transformer = FlaubertModel(config)
        self.init_weights()



@add_start_docstrings('Flaubert Model with a beam-search span classification head on top for extractive question-answering tasks like\n    SQuAD (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`). ', FLAUBERT_START_DOCSTRING)
class FlaubertForQuestionAnswering(XLMForQuestionAnswering):
    """
    This class overrides :class:`~transformers.XLMForQuestionAnswering`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """
    config_class = FlaubertConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.transformer = FlaubertModel(config)
        self.init_weights()



@add_start_docstrings('Flaubert Model with a multiple choice classification head on top (a linear layer on top of\n    the pooled output and a softmax) e.g. for RocStories/SWAG tasks. ', FLAUBERT_START_DOCSTRING)
class FlaubertForMultipleChoice(XLMForMultipleChoice):
    """
    This class overrides :class:`~transformers.XLMForMultipleChoice`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """
    config_class = FlaubertConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.transformer = FlaubertModel(config)
        self.init_weights()


