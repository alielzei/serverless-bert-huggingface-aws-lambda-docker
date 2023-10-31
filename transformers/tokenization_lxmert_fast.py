from .tokenization_bert_fast import BertTokenizerFast
from .tokenization_lxmert import LxmertTokenizer
VOCAB_FILES_NAMES = {'vocab_file': 'vocab.txt', 'tokenizer_file': 'tokenizer.json'}
PRETRAINED_VOCAB_FILES_MAP = {'vocab_file': {'unc-nlp/lxmert-base-uncased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt'}, 'tokenizer_file': {'unc-nlp/lxmert-base-uncased': 'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-tokenizer.json'}}
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {'unc-nlp/lxmert-base-uncased': 512}
PRETRAINED_INIT_CONFIGURATION = {'unc-nlp/lxmert-base-uncased': {'do_lower_case': True}}


class LxmertTokenizerFast(BertTokenizerFast):
    """
    Construct a "fast" LXMERT tokenizer (backed by HuggingFace's `tokenizers` library).

    :class:`~transformers.LxmertTokenizerFast` is identical to :class:`~transformers.BertTokenizerFast` and runs
    end-to-end tokenization: punctuation splitting and wordpiece.

    Refer to superclass :class:`~transformers.BertTokenizerFast` for usage examples and documentation concerning
    parameters.
    """
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_init_configuration = PRETRAINED_INIT_CONFIGURATION
    slow_tokenizer_class = LxmertTokenizer


