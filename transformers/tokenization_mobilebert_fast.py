"""Tokenization classes for MobileBERT."""

from .tokenization_bert_fast import BertTokenizerFast
from .tokenization_mobilebert import MobileBertTokenizer
from .utils import logging
logger = logging.get_logger(__name__)
VOCAB_FILES_NAMES = {'vocab_file': 'vocab.txt', 'tokenizer_file': 'tokenizer.json'}
PRETRAINED_VOCAB_FILES_MAP = {'vocab_file': {'mobilebert-uncased': 'https://s3.amazonaws.com/models.huggingface.co/bert/google/mobilebert-uncased/vocab.txt'}, 'tokenizer_file': {'mobilebert-uncased': 'https://s3.amazonaws.com/models.huggingface.co/bert/google/mobilebert-uncased/tokenizer.json'}}
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {'mobilebert-uncased': 512}
PRETRAINED_INIT_CONFIGURATION = {}


class MobileBertTokenizerFast(BertTokenizerFast):
    """
    Construct a "fast" MobileBERT tokenizer (backed by HuggingFace's `tokenizers` library).

    :class:`~transformers.MobileBertTokenizerFast` is identical to :class:`~transformers.BertTokenizerFast` and runs
    end-to-end tokenization: punctuation splitting and wordpiece.

    Refer to superclass :class:`~transformers.BertTokenizerFast` for usage examples and documentation concerning
    parameters.
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    slow_tokenizer_class = MobileBertTokenizer


