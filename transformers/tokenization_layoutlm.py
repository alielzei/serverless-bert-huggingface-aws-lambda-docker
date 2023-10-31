""" Tokenization class for model LayoutLM."""

from .tokenization_bert import BertTokenizer
from .utils import logging
logger = logging.get_logger(__name__)
VOCAB_FILES_NAMES = {'vocab_file': 'vocab.txt'}
PRETRAINED_VOCAB_FILES_MAP = {'vocab_file': {'microsoft/layoutlm-base-uncased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt', 'microsoft/layoutlm-large-uncased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-vocab.txt'}}
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {'microsoft/layoutlm-base-uncased': 512, 'microsoft/layoutlm-large-uncased': 512}
PRETRAINED_INIT_CONFIGURATION = {'microsoft/layoutlm-base-uncased': {'do_lower_case': True}, 'microsoft/layoutlm-large-uncased': {'do_lower_case': True}}


class LayoutLMTokenizer(BertTokenizer):
    """
    Constructs a LayoutLM tokenizer.

    :class:`~transformers.LayoutLMTokenizer is identical to :class:`~transformers.BertTokenizer` and runs end-to-end
    tokenization: punctuation splitting + wordpiece.

    Refer to superclass :class:`~transformers.BertTokenizer` for usage examples and documentation concerning
    parameters.
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES


