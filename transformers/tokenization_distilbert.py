"""Tokenization classes for DistilBERT."""

from .tokenization_bert import BertTokenizer
from .utils import logging
logger = logging.get_logger(__name__)
VOCAB_FILES_NAMES = {'vocab_file': 'vocab.txt'}
PRETRAINED_VOCAB_FILES_MAP = {'vocab_file': {'distilbert-base-uncased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt', 'distilbert-base-uncased-distilled-squad': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt', 'distilbert-base-cased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-vocab.txt', 'distilbert-base-cased-distilled-squad': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-vocab.txt', 'distilbert-base-german-cased': 'https://s3.amazonaws.com/models.huggingface.co/bert/distilbert-base-german-cased-vocab.txt', 'distilbert-base-multilingual-cased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-vocab.txt'}}
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {'distilbert-base-uncased': 512, 'distilbert-base-uncased-distilled-squad': 512, 'distilbert-base-cased': 512, 'distilbert-base-cased-distilled-squad': 512, 'distilbert-base-german-cased': 512, 'distilbert-base-multilingual-cased': 512}
PRETRAINED_INIT_CONFIGURATION = {'distilbert-base-uncased': {'do_lower_case': True}, 'distilbert-base-uncased-distilled-squad': {'do_lower_case': True}, 'distilbert-base-cased': {'do_lower_case': False}, 'distilbert-base-cased-distilled-squad': {'do_lower_case': False}, 'distilbert-base-german-cased': {'do_lower_case': False}, 'distilbert-base-multilingual-cased': {'do_lower_case': False}}


class DistilBertTokenizer(BertTokenizer):
    """
    Construct a DistilBERT tokenizer.

    :class:`~transformers.DistilBertTokenizer` is identical to :class:`~transformers.BertTokenizer` and runs end-to-end
    tokenization: punctuation splitting and wordpiece.

    Refer to superclass :class:`~transformers.BertTokenizer` for usage examples and documentation concerning
    parameters.
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    model_input_names = ['attention_mask']


