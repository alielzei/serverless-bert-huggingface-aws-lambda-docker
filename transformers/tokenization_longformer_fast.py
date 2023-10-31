from .tokenization_longformer import LongformerTokenizer
from .tokenization_roberta_fast import RobertaTokenizerFast
from .utils import logging
logger = logging.get_logger(__name__)
vocab_url = 'https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-vocab.json'
merges_url = 'https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-merges.txt'
tokenizer_url = 'https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-tokenizer.json'
_all_longformer_models = ['allenai/longformer-base-4096', 'allenai/longformer-large-4096', 'allenai/longformer-large-4096-finetuned-triviaqa', 'allenai/longformer-base-4096-extra.pos.embd.only', 'allenai/longformer-large-4096-extra.pos.embd.only']
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {'allenai/longformer-base-4096': 4096, 'allenai/longformer-large-4096': 4096, 'allenai/longformer-large-4096-finetuned-triviaqa': 4096, 'allenai/longformer-base-4096-extra.pos.embd.only': 4096, 'allenai/longformer-large-4096-extra.pos.embd.only': 4096}


class LongformerTokenizerFast(RobertaTokenizerFast):
    """
    Construct a "fast" Longformer tokenizer (backed by HuggingFace's `tokenizers` library).

    :class:`~transformers.LongformerTokenizerFast` is identical to :class:`~transformers.RobertaTokenizerFast`. Refer
    to the superclass for usage examples and documentation concerning parameters.
    """
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    pretrained_vocab_files_map = {'vocab_file': {m: vocab_url for m in _all_longformer_models}, 'merges_file': {m: merges_url for m in _all_longformer_models}, 'tokenizer_file': {m: tokenizer_url for m in _all_longformer_models}}
    slow_tokenizer_class = LongformerTokenizer


