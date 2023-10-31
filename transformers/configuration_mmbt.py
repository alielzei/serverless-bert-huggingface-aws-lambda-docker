""" MMBT configuration """

from .utils import logging
logger = logging.get_logger(__name__)


class MMBTConfig(object):
    """
    This is the configuration class to store the configuration of a :class:`~transformers.MMBTModel`. It is used to
    instantiate a MMBT model according to the specified arguments, defining the model architecture.

    Args:
        config (:class:`~transformers.PreTrainedConfig`):
            Config of the underlying Transformer models. Its values are copied over to use a single config.
        num_labels (:obj:`int`, `optional`):
            Size of final Linear layer for classification.
        modal_hidden_size (:obj:`int`, `optional`, defaults to 2048):
            Embedding dimension of the non-text modality encoder.
    """
    
    def __init__(self, config, num_labels=None, modal_hidden_size=2048):
        self.__dict__ = config.__dict__
        self.modal_hidden_size = modal_hidden_size
        if num_labels:
            self.num_labels = num_labels


