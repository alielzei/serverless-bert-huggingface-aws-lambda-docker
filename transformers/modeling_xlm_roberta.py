"""PyTorch XLM-RoBERTa model. """

from .configuration_xlm_roberta import XLMRobertaConfig
from .file_utils import add_start_docstrings
from .modeling_roberta import RobertaForCausalLM, RobertaForMaskedLM, RobertaForMultipleChoice, RobertaForQuestionAnswering, RobertaForSequenceClassification, RobertaForTokenClassification, RobertaModel
from .utils import logging
logger = logging.get_logger(__name__)
XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = ['xlm-roberta-base', 'xlm-roberta-large', 'xlm-roberta-large-finetuned-conll02-dutch', 'xlm-roberta-large-finetuned-conll02-spanish', 'xlm-roberta-large-finetuned-conll03-english', 'xlm-roberta-large-finetuned-conll03-german']
XLM_ROBERTA_START_DOCSTRING = '\n\n    This model inherits from :class:`~transformers.PreTrainedModel`. Check the superclass documentation for the generic\n    methods the library implements for all its model (such as downloading or saving, resizing the input embeddings,\n    pruning heads etc.)\n\n    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__ subclass.\n    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general\n    usage and behavior.\n\n    Parameters:\n        config (:class:`~transformers.XLMRobertaConfig`): Model configuration class with all the parameters of the\n            model. Initializing with a config file does not load the weights associated with the model, only the configuration.\n            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.\n'


@add_start_docstrings('The bare XLM-RoBERTa Model transformer outputting raw hidden-states without any specific head on top.', XLM_ROBERTA_START_DOCSTRING)
class XLMRobertaModel(RobertaModel):
    """
    This class overrides :class:`~transformers.RobertaModel`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """
    config_class = XLMRobertaConfig



@add_start_docstrings('XLM-RoBERTa Model with a `language modeling` head on top for CLM fine-tuning.', XLM_ROBERTA_START_DOCSTRING)
class XLMRobertaForCausalLM(RobertaForCausalLM):
    """
    This class overrides :class:`~transformers.RobertaForCausalLM`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """
    config_class = XLMRobertaConfig



@add_start_docstrings('XLM-RoBERTa Model with a `language modeling` head on top. ', XLM_ROBERTA_START_DOCSTRING)
class XLMRobertaForMaskedLM(RobertaForMaskedLM):
    """
    This class overrides :class:`~transformers.RobertaForMaskedLM`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """
    config_class = XLMRobertaConfig



@add_start_docstrings('XLM-RoBERTa Model transformer with a sequence classification/regression head on top (a linear layer\n    on top of the pooled output) e.g. for GLUE tasks. ', XLM_ROBERTA_START_DOCSTRING)
class XLMRobertaForSequenceClassification(RobertaForSequenceClassification):
    """
    This class overrides :class:`~transformers.RobertaForSequenceClassification`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """
    config_class = XLMRobertaConfig



@add_start_docstrings('XLM-RoBERTa Model with a multiple choice classification head on top (a linear layer on top of\n    the pooled output and a softmax) e.g. for RocStories/SWAG tasks. ', XLM_ROBERTA_START_DOCSTRING)
class XLMRobertaForMultipleChoice(RobertaForMultipleChoice):
    """
    This class overrides :class:`~transformers.RobertaForMultipleChoice`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """
    config_class = XLMRobertaConfig



@add_start_docstrings('XLM-RoBERTa Model with a token classification head on top (a linear layer on top of\n    the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. ', XLM_ROBERTA_START_DOCSTRING)
class XLMRobertaForTokenClassification(RobertaForTokenClassification):
    """
    This class overrides :class:`~transformers.RobertaForTokenClassification`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """
    config_class = XLMRobertaConfig



@add_start_docstrings('XLM-RoBERTa Model with a span classification head on top for extractive question-answering tasks like SQuAD (a\n    linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).', XLM_ROBERTA_START_DOCSTRING)
class XLMRobertaForQuestionAnswering(RobertaForQuestionAnswering):
    """
    This class overrides :class:`~transformers.RobertaForQuestionAnswering`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """
    config_class = XLMRobertaConfig


