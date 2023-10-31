"""PyTorch MMBT model. """

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from .file_utils import add_start_docstrings, add_start_docstrings_to_callable, replace_return_docstrings
from .modeling_outputs import BaseModelOutputWithPooling, SequenceClassifierOutput
from .modeling_utils import ModuleUtilsMixin
from .utils import logging
logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = 'MMBTConfig'


class ModalEmbeddings(nn.Module):
    """Generic Modal Embeddings which takes in an encoder, and a transformer embedding."""
    
    def __init__(self, config, encoder, embeddings):
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.proj_embeddings = nn.Linear(config.modal_hidden_size, config.hidden_size)
        self.position_embeddings = embeddings.position_embeddings
        self.token_type_embeddings = embeddings.token_type_embeddings
        self.word_embeddings = embeddings.word_embeddings
        self.LayerNorm = embeddings.LayerNorm
        self.dropout = nn.Dropout(p=config.hidden_dropout_prob)
    
    def forward(self, input_modal, start_token=None, end_token=None, position_ids=None, token_type_ids=None):
        token_embeddings = self.proj_embeddings(self.encoder(input_modal))
        seq_length = token_embeddings.size(1)
        if start_token is not None:
            start_token_embeds = self.word_embeddings(start_token)
            seq_length += 1
            token_embeddings = torch.cat([start_token_embeds.unsqueeze(1), token_embeddings], dim=1)
        if end_token is not None:
            end_token_embeds = self.word_embeddings(end_token)
            seq_length += 1
            token_embeddings = torch.cat([token_embeddings, end_token_embeds.unsqueeze(1)], dim=1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_modal.device)
            position_ids = position_ids.unsqueeze(0).expand(input_modal.size(0), seq_length)
        if token_type_ids is None:
            token_type_ids = torch.zeros((input_modal.size(0), seq_length), dtype=torch.long, device=input_modal.device)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = token_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

MMBT_START_DOCSTRING = "\n    MMBT model was proposed in\n    `Supervised Multimodal Bitransformers for Classifying Images and Text <https://github.com/facebookresearch/mmbt>`__\n    by Douwe Kiela, Suvrat Bhooshan, Hamed Firooz, Davide Testuggine.\n    It's a supervised multimodal bitransformer model that fuses information from text and other image encoders,\n    and obtain state-of-the-art performance on various multimodal classification benchmark tasks.\n\n    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic\n    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,\n    pruning heads etc.)\n\n    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__ subclass.\n    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general\n    usage and behavior.\n\n    Parameters:\n        config (:class:`~transformers.MMBTConfig`): Model configuration class with all the parameters of the model.\n            Initializing with a config file does not load the weights associated with the model, only the configuration.\n        transformer (:class: `~nn.Module`): A text transformer that is used by MMBT.\n            It should have embeddings, encoder, and pooler attributes.\n        encoder (:class: `~nn.Module`): Encoder for the second modality.\n            It should take in a batch of modal inputs and return k, n dimension embeddings.\n"
MMBT_INPUTS_DOCSTRING = "\n    Args:\n        input_modal (``torch.FloatTensor`` of shape ``(batch_size, ***)``):\n            The other modality data. It will be the shape that the encoder for that type expects.\n            e.g. With an Image Encoder, the shape would be (batch_size, channels, height, width)\n        input_ids (``torch.LongTensor`` of shape ``(batch_size, sequence_length)``):\n            Indices of input sequence tokens in the vocabulary.\n            It does not expect [CLS] token to be added as it's appended to the end of other modality embeddings.\n            Indices can be obtained using :class:`~transformers.BertTokenizer`.\n            See :meth:`transformers.PreTrainedTokenizer.encode` and\n            :meth:`transformers.PreTrainedTokenizer.__call__` for details.\n\n            `What are input IDs? <../glossary.html#input-ids>`__\n        modal_start_tokens (``torch.LongTensor`` of shape ``(batch_size,)``, `optional`):\n            Optional start token to be added to Other Modality Embedding. [CLS] Most commonly used for classification\n            tasks.\n        modal_end_tokens (``torch.LongTensor`` of shape ``(batch_size,)``, `optional`):\n            Optional end token to be added to Other Modality Embedding. [SEP] Most commonly used.\n        attention_mask (`optional`) ``torch.FloatTensor`` of shape ``(batch_size, sequence_length)``:\n            Mask to avoid performing attention on padding token indices.\n            Mask values selected in ``[0, 1]``:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n            `What are attention masks? <../glossary.html#attention-mask>`__\n        token_type_ids (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:\n            Segment token indices to indicate first and second portions of the inputs.\n            Indices are selected in ``[0, 1]``:\n\n            - 0 corresponds to a `sentence A` token,\n            - 1 corresponds to a `sentence B` token.\n\n            `What are token type IDs? <../glossary.html#token-type-ids>`_\n        modal_token_type_ids (`optional`) ``torch.LongTensor`` of shape ``(batch_size, modal_sequence_length)``:\n            Segment token indices to indicate different portions of the non-text modality.\n            The embeddings from these tokens will be summed with the respective token embeddings for the non-text modality.\n        position_ids (``torch.LongTensor`` of shape ``(batch_size, sequence_length)``, `optional`):\n            Indices of positions of each input sequence tokens in the position embeddings.\n            Selected in the range ``[0, config.max_position_embeddings - 1]``.\n\n            `What are position IDs? <../glossary.html#position-ids>`__\n        modal_position_ids (``torch.LongTensor`` of shape ``(batch_size, modal_sequence_length)``, `optional`):\n            Indices of positions of each input sequence tokens in the position embeddings for the non-text modality.\n            Selected in the range ``[0, config.max_position_embeddings - 1]``.\n\n            `What are position IDs? <../glossary.html#position-ids>`__\n        head_mask (``torch.FloatTensor`` of shape ``(num_heads,)`` or ``(num_layers, num_heads)``, `optional`):\n            Mask to nullify selected heads of the self-attention modules.\n            Mask values selected in ``[0, 1]``:\n\n            - 1 indicates the head is **not masked**,\n            - 0 indicates the head is **masked**.\n\n        inputs_embeds (``torch.FloatTensor`` of shape ``(batch_size, sequence_length, embedding_dim)``, `optional`):\n            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.\n            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated\n            vectors than the model's internal embedding lookup matrix.\n        encoder_hidden_states (``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``, `optional`):\n            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if\n            the model is configured as a decoder.\n        encoder_attention_mask (``torch.FloatTensor`` of shape ``(batch_size, sequence_length)``, `optional`):\n            Mask to avoid performing attention on the padding token indices of the encoder input. This mask\n            is used in the cross-attention if the model is configured as a decoder.\n            Mask values selected in ``[0, 1]``:\n\n            - 1 for tokens that are **not masked**,\n            - 0 for tokens that are **masked**.\n\n        output_attentions (:obj:`bool`, `optional`):\n            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned\n            tensors for more detail.\n        output_hidden_states (:obj:`bool`, `optional`):\n            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for\n            more detail.\n        return_dict (:obj:`bool`, `optional`):\n            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.\n"


@add_start_docstrings('The bare MMBT Model outputting raw hidden-states without any specific head on top.', MMBT_START_DOCSTRING)
class MMBTModel(nn.Module, ModuleUtilsMixin):
    
    def __init__(self, config, transformer, encoder):
        super().__init__()
        self.config = config
        self.transformer = transformer
        self.modal_encoder = ModalEmbeddings(config, encoder, transformer.embeddings)
    
    @add_start_docstrings_to_callable(MMBT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def forward(self, input_modal, input_ids=None, modal_start_tokens=None, modal_end_tokens=None, attention_mask=None, token_type_ids=None, modal_token_type_ids=None, position_ids=None, modal_position_ids=None, head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        """
        Returns:

        Examples::

            # For example purposes. Not runnable.
            transformer = BertModel.from_pretrained('bert-base-uncased')
            encoder = ImageEncoder(args)
            mmbt = MMBTModel(config, transformer, encoder)
        """
        output_attentions = (output_attentions if output_attentions is not None else self.config.output_attentions)
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict = (return_dict if return_dict is not None else self.config.use_return_dict)
        if (input_ids is not None and inputs_embeds is not None):
            raise ValueError('You cannot specify both input_ids and inputs_embeds at the same time')
        elif input_ids is not None:
            input_txt_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_txt_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError('You have to specify either input_ids or inputs_embeds')
        device = (input_ids.device if input_ids is not None else inputs_embeds.device)
        modal_embeddings = self.modal_encoder(input_modal, start_token=modal_start_tokens, end_token=modal_end_tokens, position_ids=modal_position_ids, token_type_ids=modal_token_type_ids)
        input_modal_shape = modal_embeddings.size()[:-1]
        if token_type_ids is None:
            token_type_ids = torch.ones(input_txt_shape, dtype=torch.long, device=device)
        txt_embeddings = self.transformer.embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)
        embedding_output = torch.cat([modal_embeddings, txt_embeddings], 1)
        input_shape = embedding_output.size()[:-1]
        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        else:
            attention_mask = torch.cat([torch.ones(input_modal_shape, device=device, dtype=torch.long), attention_mask], dim=1)
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(input_shape, device=device)
        else:
            encoder_attention_mask = torch.cat([torch.ones(input_modal_shape, device=device), encoder_attention_mask], dim=1)
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, self.device)
        encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
        encoder_outputs = self.transformer.encoder(embedding_output, attention_mask=extended_attention_mask, head_mask=head_mask, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_extended_attention_mask, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        sequence_output = encoder_outputs[0]
        pooled_output = self.transformer.pooler(sequence_output)
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]
        return BaseModelOutputWithPooling(last_hidden_state=sequence_output, pooler_output=pooled_output, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions)
    
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings
    
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value



@add_start_docstrings('MMBT Model with a sequence classification/regression head on top (a linear layer on top of\n                      the pooled output)', MMBT_START_DOCSTRING, MMBT_INPUTS_DOCSTRING)
class MMBTForClassification(nn.Module):
    """
        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size,)``:
            Labels for computing the sequence classification/regression loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.
            If ``config.num_labels == 1`` a regression loss is computed (Mean-Square loss),
            If ``config.num_labels > 1`` a classification loss is computed (Cross-Entropy).

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Classification (or regression if config.num_labels==1) loss.
        **logits**: ``torch.FloatTensor`` of shape ``(batch_size, config.num_labels)``
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        **hidden_states**: (`optional`, returned when ``output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        # For example purposes. Not runnable.
        transformer = BertModel.from_pretrained('bert-base-uncased')
        encoder = ImageEncoder(args)
        model = MMBTForClassification(config, transformer, encoder)
        outputs = model(input_modal, input_ids, labels=labels)
        loss, logits = outputs[:2]
    """
    
    def __init__(self, config, transformer, encoder):
        super().__init__()
        self.num_labels = config.num_labels
        self.mmbt = MMBTModel(config, transformer, encoder)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
    
    def forward(self, input_modal, input_ids=None, modal_start_tokens=None, modal_end_tokens=None, attention_mask=None, token_type_ids=None, modal_token_type_ids=None, position_ids=None, modal_position_ids=None, head_mask=None, inputs_embeds=None, labels=None, return_dict=None):
        return_dict = (return_dict if return_dict is not None else self.config.use_return_dict)
        outputs = self.mmbt(input_modal=input_modal, input_ids=input_ids, modal_start_tokens=modal_start_tokens, modal_end_tokens=modal_end_tokens, attention_mask=attention_mask, token_type_ids=token_type_ids, modal_token_type_ids=modal_token_type_ids, position_ids=position_ids, modal_position_ids=modal_position_ids, head_mask=head_mask, inputs_embeds=inputs_embeds, return_dict=return_dict)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
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


