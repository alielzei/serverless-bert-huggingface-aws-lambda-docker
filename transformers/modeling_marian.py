"""PyTorch MarianMTModel model, ported from the Marian C++ repo."""

from .configuration_marian import MarianConfig
from .modeling_bart import BartForConditionalGeneration


class MarianMTModel(BartForConditionalGeneration):
    """
    Pytorch version of marian-nmt's transformer.h (c++). Designed for the OPUS-NMT translation checkpoints.
    Available models are listed `here <https://huggingface.co/models?search=Helsinki-NLP>`__.

    This class overrides :class:`~transformers.BartForConditionalGeneration`. Please check the
    superclass for the appropriate documentation alongside usage examples.

    Examples::

        >>> from transformers import MarianTokenizer, MarianMTModel
        >>> from typing import List
        >>> src = 'fr'  # source language
        >>> trg = 'en'  # target language
        >>> sample_text = "où est l'arrêt de bus ?"
        >>> mname = f'Helsinki-NLP/opus-mt-{src}-{trg}'

        >>> model = MarianMTModel.from_pretrained(mname)
        >>> tok = MarianTokenizer.from_pretrained(mname)
        >>> batch = tok.prepare_seq2seq_batch(src_texts=[sample_text])  # don't need tgt_text for inference
        >>> gen = model.generate(**batch)  # for forward pass: model(**batch)
        >>> words: List[str] = tok.batch_decode(gen, skip_special_tokens=True)  # returns "Where is the bus stop ?"

    """
    config_class = MarianConfig
    
    def adjust_logits_during_generation(self, logits, cur_len, max_length):
        logits[:, self.config.pad_token_id] = float('-inf')
        if (cur_len == max_length - 1 and self.config.eos_token_id is not None):
            self._force_token_ids_generation(logits, self.config.eos_token_id)
        return logits


