""" CamemBERT configuration """

from .configuration_roberta import RobertaConfig
from .utils import logging
logger = logging.get_logger(__name__)
CAMEMBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {'camembert-base': 'https://s3.amazonaws.com/models.huggingface.co/bert/camembert-base-config.json', 'umberto-commoncrawl-cased-v1': 'https://s3.amazonaws.com/models.huggingface.co/bert/Musixmatch/umberto-commoncrawl-cased-v1/config.json', 'umberto-wikipedia-uncased-v1': 'https://s3.amazonaws.com/models.huggingface.co/bert/Musixmatch/umberto-wikipedia-uncased-v1/config.json'}


class CamembertConfig(RobertaConfig):
    """
    This class overrides :class:`~transformers.RobertaConfig`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """
    model_type = 'camembert'


