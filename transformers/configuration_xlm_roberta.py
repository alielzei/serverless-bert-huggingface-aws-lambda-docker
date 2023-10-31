""" XLM-RoBERTa configuration """

from .configuration_roberta import RobertaConfig
from .utils import logging
logger = logging.get_logger(__name__)
XLM_ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP = {'xlm-roberta-base': 'https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-base-config.json', 'xlm-roberta-large': 'https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-large-config.json', 'xlm-roberta-large-finetuned-conll02-dutch': 'https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-large-finetuned-conll02-dutch-config.json', 'xlm-roberta-large-finetuned-conll02-spanish': 'https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-large-finetuned-conll02-spanish-config.json', 'xlm-roberta-large-finetuned-conll03-english': 'https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-large-finetuned-conll03-english-config.json', 'xlm-roberta-large-finetuned-conll03-german': 'https://s3.amazonaws.com/models.huggingface.co/bert/xlm-roberta-large-finetuned-conll03-german-config.json'}


class XLMRobertaConfig(RobertaConfig):
    """
    This class overrides :class:`~transformers.RobertaConfig`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """
    model_type = 'xlm-roberta'


