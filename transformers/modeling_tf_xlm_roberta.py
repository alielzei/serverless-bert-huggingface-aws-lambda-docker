""" TF 2.0  XLM-RoBERTa model. """

from .configuration_xlm_roberta import XLMRobertaConfig
from .file_utils import add_start_docstrings
from .modeling_tf_roberta import TFRobertaForMaskedLM, TFRobertaForMultipleChoice, TFRobertaForQuestionAnswering, TFRobertaForSequenceClassification, TFRobertaForTokenClassification, TFRobertaModel
from .utils import logging
logger = logging.get_logger(__name__)
TF_XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = []
XLM_ROBERTA_START_DOCSTRING = '\n\n    This model inherits from :class:`~transformers.TFPreTrainedModel`. Check the superclass documentation for the\n    generic methods the library implements for all its model (such as downloading or saving, resizing the input\n    embeddings, pruning heads etc.)\n\n    This model is also a `tf.keras.Model <https://www.tensorflow.org/api_docs/python/tf/keras/Model>`__ subclass.\n    Use it as a regular TF 2.0 Keras Model and refer to the TF 2.0 documentation for all matter related to general\n    usage and behavior.\n\n    .. note::\n\n        TF 2.0 models accepts two formats as inputs:\n\n        - having all inputs as keyword arguments (like PyTorch models), or\n        - having all inputs as a list, tuple or dict in the first positional arguments.\n\n        This second option is useful when using :meth:`tf.keras.Model.fit` method which currently requires having\n        all the tensors in the first argument of the model call function: :obj:`model(inputs)`.\n\n        If you choose this second option, there are three possibilities you can use to gather all the input Tensors\n        in the first positional argument :\n\n        - a single Tensor with :obj:`input_ids` only and nothing else: :obj:`model(inputs_ids)`\n        - a list of varying length with one or several input Tensors IN THE ORDER given in the docstring:\n          :obj:`model([input_ids, attention_mask])` or :obj:`model([input_ids, attention_mask, token_type_ids])`\n        - a dictionary with one or several input Tensors associated to the input names given in the docstring:\n          :obj:`model({"input_ids": input_ids, "token_type_ids": token_type_ids})`\n\n    Parameters:\n        config (:class:`~transformers.XLMRobertaConfig`): Model configuration class with all the parameters of the\n            model. Initializing with a config file does not load the weights associated with the model, only the configuration.\n            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.\n'


@add_start_docstrings('The bare XLM-RoBERTa Model transformer outputting raw hidden-states without any specific head on top.', XLM_ROBERTA_START_DOCSTRING)
class TFXLMRobertaModel(TFRobertaModel):
    """
    This class overrides :class:`~transformers.TFRobertaModel`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """
    config_class = XLMRobertaConfig



@add_start_docstrings('XLM-RoBERTa Model with a `language modeling` head on top. ', XLM_ROBERTA_START_DOCSTRING)
class TFXLMRobertaForMaskedLM(TFRobertaForMaskedLM):
    """
    This class overrides :class:`~transformers.TFRobertaForMaskedLM`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """
    config_class = XLMRobertaConfig



@add_start_docstrings('XLM-RoBERTa Model transformer with a sequence classification/regression head on top (a linear layer\n    on top of the pooled output) e.g. for GLUE tasks. ', XLM_ROBERTA_START_DOCSTRING)
class TFXLMRobertaForSequenceClassification(TFRobertaForSequenceClassification):
    """
    This class overrides :class:`~transformers.TFRobertaForSequenceClassification`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """
    config_class = XLMRobertaConfig



@add_start_docstrings('XLM-RoBERTa Model with a token classification head on top (a linear layer on top of\n    the hidden-states output) e.g. for Named-Entity-Recognition (NER) tasks. ', XLM_ROBERTA_START_DOCSTRING)
class TFXLMRobertaForTokenClassification(TFRobertaForTokenClassification):
    """
    This class overrides :class:`~transformers.TFRobertaForTokenClassification`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """
    config_class = XLMRobertaConfig



@add_start_docstrings('XLM-RoBERTa Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`). ', XLM_ROBERTA_START_DOCSTRING)
class TFXLMRobertaForQuestionAnswering(TFRobertaForQuestionAnswering):
    """
    This class overrides :class:`~transformers.TFRobertaForQuestionAnsweringSimple`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """
    config_class = XLMRobertaConfig



@add_start_docstrings('Roberta Model with a multiple choice classification head on top (a linear layer on top of\n    the pooled output and a softmax) e.g. for RocStories/SWAG tasks. ', XLM_ROBERTA_START_DOCSTRING)
class TFXLMRobertaForMultipleChoice(TFRobertaForMultipleChoice):
    """
    This class overrides :class:`~transformers.TFRobertaForMultipleChoice`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """
    config_class = XLMRobertaConfig


