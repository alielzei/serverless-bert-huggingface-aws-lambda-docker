"""Tokenization classes for RetriBERT."""

from .tokenization_bert import BertTokenizer
from .utils import logging
logger = logging.get_logger(__name__)
VOCAB_FILES_NAMES = {'vocab_file': 'vocab.txt'}
PRETRAINED_VOCAB_FILES_MAP = {'vocab_file': {'yjernite/retribert-base-uncased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt'}}
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {'yjernite/retribert-base-uncased': 512}
PRETRAINED_INIT_CONFIGURATION = {'yjernite/retribert-base-uncased': {'do_lower_case': True}}


class RetriBertTokenizer(BertTokenizer):
    """
    Constructs a RetriBERT tokenizer.

    :class:`~transformers.RetroBertTokenizer` is identical to :class:`~transformers.BertTokenizer` and runs end-to-end
    tokenization: punctuation splitting and wordpiece.

    Refer to superclass :class:`~transformers.BertTokenizer` for usage examples and documentation concerning
    parameters.
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    model_input_names = ['attention_mask']


