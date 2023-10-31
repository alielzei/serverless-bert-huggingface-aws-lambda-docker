"""Tokenization classes for SqueezeBERT."""

from .tokenization_bert_fast import BertTokenizerFast
from .tokenization_squeezebert import SqueezeBertTokenizer
from .utils import logging
logger = logging.get_logger(__name__)
VOCAB_FILES_NAMES = {'vocab_file': 'vocab.txt', 'tokenizer_file': 'tokenizer.json'}
PRETRAINED_VOCAB_FILES_MAP = {'vocab_file': {'squeezebert/squeezebert-uncased': 'https://s3.amazonaws.com/models.huggingface.co/bert/squeezebert/squeezebert-uncased/vocab.txt', 'squeezebert/squeezebert-mnli': 'https://s3.amazonaws.com/models.huggingface.co/bert/squeezebert/squeezebert-mnli/vocab.txt', 'squeezebert/squeezebert-mnli-headless': 'https://s3.amazonaws.com/models.huggingface.co/bert/squeezebert/squeezebert-mnli-headless/vocab.txt'}, 'tokenizer_file': {'squeezebert/squeezebert-uncased': 'https://s3.amazonaws.com/models.huggingface.co/bert/squeezebert/squeezebert-uncased/tokenizer.json', 'squeezebert/squeezebert-mnli': 'https://s3.amazonaws.com/models.huggingface.co/bert/squeezebert/squeezebert-mnli/tokenizer.json', 'squeezebert/squeezebert-mnli-headless': 'https://s3.amazonaws.com/models.huggingface.co/bert/squeezebert/squeezebert-mnli-headless/tokenizer.json'}}
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {'squeezebert/squeezebert-uncased': 512, 'squeezebert/squeezebert-mnli': 512, 'squeezebert/squeezebert-mnli-headless': 512}
PRETRAINED_INIT_CONFIGURATION = {'squeezebert/squeezebert-uncased': {'do_lower_case': True}, 'squeezebert/squeezebert-mnli': {'do_lower_case': True}, 'squeezebert/squeezebert-mnli-headless': {'do_lower_case': True}}


class SqueezeBertTokenizerFast(BertTokenizerFast):
    """
    Constructs a  "Fast" SqueezeBert tokenizer (backed by HuggingFace's `tokenizers` library).

    :class:`~transformers.SqueezeBertTokenizerFast` is identical to :class:`~transformers.BertTokenizerFast` and
    runs end-to-end tokenization: punctuation splitting + wordpiece.

    Refer to superclass :class:`~transformers.BertTokenizerFast` for usage examples and documentation concerning
    parameters.
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    slow_tokenizer_class = SqueezeBertTokenizer


