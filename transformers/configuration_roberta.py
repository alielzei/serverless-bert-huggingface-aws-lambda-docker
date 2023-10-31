""" RoBERTa configuration """

from .configuration_bert import BertConfig
from .utils import logging
logger = logging.get_logger(__name__)
ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP = {'roberta-base': 'https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-config.json', 'roberta-large': 'https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-config.json', 'roberta-large-mnli': 'https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-config.json', 'distilroberta-base': 'https://s3.amazonaws.com/models.huggingface.co/bert/distilroberta-base-config.json', 'roberta-base-openai-detector': 'https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-openai-detector-config.json', 'roberta-large-openai-detector': 'https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-openai-detector-config.json'}


class RobertaConfig(BertConfig):
    """
    This is the configuration class to store the configuration of a :class:`~transformers.RobertaModel` or a
    :class:`~transformers.TFRobertaModel`. It is used to instantiate a RoBERTa model according to the specified
    arguments, defining the model architecture.


    Configuration objects inherit from  :class:`~transformers.PretrainedConfig` and can be used
    to control the model outputs. Read the documentation from  :class:`~transformers.PretrainedConfig`
    for more information.

    The :class:`~transformers.RobertaConfig` class directly inherits :class:`~transformers.BertConfig`.
    It reuses the same defaults. Please check the parent class for more information.

    Examples::

        >>> from transformers import RobertaConfig, RobertaModel

        >>> # Initializing a RoBERTa configuration
        >>> configuration = RobertaConfig()

        >>> # Initializing a model from the configuration
        >>> model = RobertaModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = 'roberta'
    
    def __init__(self, pad_token_id=1, bos_token_id=0, eos_token_id=2, **kwargs):
        """Constructs RobertaConfig."""
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)


