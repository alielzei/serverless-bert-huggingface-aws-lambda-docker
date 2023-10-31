import numpy as np
import tensorflow as tf
from .utils import logging
logger = logging.get_logger(__name__)


class TFGenerationMixin:
    """
    A class contraining all of the functions supporting generation, to be used as a mixin in
    :class:`~transfomers.TFPreTrainedModel`.
    """
    
    def prepare_inputs_for_generation(self, inputs, **kwargs):
        """
        Implement in subclasses of :class:`~transfomers.TFPreTrainedModel` for custom behavior to prepare inputs in the
        generate method.
        """
        return {'inputs': inputs}
    
    def _use_cache(self, outputs, use_cache):
        """During generation, decide whether to pass the `past` variable to the next forward pass."""
        if (len(outputs) <= 1 or use_cache is False):
            return False
        if (hasattr(self.config, 'mem_len') and self.config.mem_len == 0):
            return False
        return True
    
    def generate(self, input_ids=None, max_length=None, min_length=None, do_sample=None, early_stopping=None, num_beams=None, temperature=None, top_k=None, top_p=None, repetition_penalty=None, bad_words_ids=None, bos_token_id=None, pad_token_id=None, eos_token_id=None, length_penalty=None, no_repeat_ngram_size=None, num_return_sequences=None, attention_mask=None, decoder_start_token_id=None, use_cache=None):
        """
        Generates sequences for models with a language modeling head. The method currently supports greedy decoding,
        beam-search decoding, sampling with temperature, sampling with top-k or nucleus sampling.

        Adapted in part from `Facebook's XLM beam search code
        <https://github.com/facebookresearch/XLM/blob/9e6f6814d17be4fe5b15f2e6c43eb2b2d76daeb4/src/model/transformer.py#L529>`__.

        Apart from :obj:`input_ids` and :obj:`attention_mask`, all the arguments below will default to the value of the
        attribute of the same name inside the :class:`~transformers.PretrainedConfig` of the model. The default values
        indicated are the default values of those config.

        Most of these parameters are explained in more detail in `this blog post
        <https://huggingface.co/blog/how-to-generate>`__.

        Parameters:

            input_ids (:obj:`tf.Tensor` of :obj:`dtype=tf.int32` and shape :obj:`(batch_size, sequence_length)`, `optional`):
                The sequence used as a prompt for the generation. If :obj:`None` the method initializes
                it as an empty :obj:`tf.Tensor` of shape :obj:`(1,)`.
            max_length (:obj:`int`, `optional`, defaults to 20):
                The maximum length of the sequence to be generated.
            min_length (:obj:`int`, `optional`, defaults to 10):
                The minimum length of the sequence to be generated.
            do_sample (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether or not to use sampling ; use greedy decoding otherwise.
            early_stopping (:obj:`bool`, `optional`, defaults to :obj:`False`):
                Whether to stop the beam search when at least ``num_beams`` sentences are finished per batch or not.
            num_beams (:obj:`int`, `optional`, defaults to 1):
                Number of beams for beam search. 1 means no beam search.
            temperature (:obj:`float`, `optional`, defaults to 1.0):
                The value used to module the next token probabilities.
            top_k (:obj:`int`, `optional`, defaults to 50):
                The number of highest probability vocabulary tokens to keep for top-k-filtering.
            top_p (:obj:`float`, `optional`, defaults to 1.0):
                If set to float < 1, only the most probable tokens with probabilities that add up to ``top_p`` or
                higher are kept for generation.
            repetition_penalty (:obj:`float`, `optional`, defaults to 1.0):
                The parameter for repetition penalty. 1.0 means no penalty. See `this paper
                <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details.
            pad_token_id (:obj:`int`, `optional`):
                The id of the `padding` token.
            bos_token_id (:obj:`int`, `optional`):
                The id of the `beginning-of-sequence` token.
            eos_token_id (:obj:`int`, `optional`):
                The id of the `end-of-sequence` token.
            length_penalty (:obj:`float`, `optional`, defaults to 1.0):
                Exponential penalty to the length. 1.0 means no penalty.

                Set to values < 1.0 in order to encourage the model to generate shorter sequences, to a value > 1.0 in
                order to encourage the model to produce longer sequences.
            no_repeat_ngram_size (:obj:`int`, `optional`, defaults to 0):
                If set to int > 0, all ngrams of that size can only occur once.
            bad_words_ids(:obj:`List[int]`, `optional`):
                List of token ids that are not allowed to be generated. In order to get the tokens of the words that
                should not appear in the generated text, use :obj:`tokenizer.encode(bad_word, add_prefix_space=True)`.
            num_return_sequences(:obj:`int`, `optional`, defaults to 1):
                The number of independently computed returned sequences for each element in the batch.
            attention_mask (:obj:`tf.Tensor` of :obj:`dtype=tf.int32` and shape :obj:`(batch_size, sequence_length)`, `optional`):
                Mask to avoid performing attention on padding token indices. Mask values are in ``[0, 1]``, 1 for
                tokens that are not masked, and 0 for masked tokens.

                If not provided, will default to a tensor the same shape as :obj:`input_ids` that masks the pad token.

                `What are attention masks? <../glossary.html#attention-mask>`__
            decoder_start_token_id (:obj:`int`, `optional`):
                If an encoder-decoder model starts decoding with a different token than `bos`, the id of that token.
            use_cache: (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether or not the model should use the past last key/values attentions (if applicable to the model) to
                speed up decoding.
            model_specific_kwargs:
                Additional model specific kwargs will be forwarded to the :obj:`forward` function of the model.

        Return:

            :obj:`tf.Tensor` of :obj:`dtype=tf.int32` and shape :obj:`(batch_size * num_return_sequences, sequence_length)`:
            The generated sequences. The second dimension (sequence_length) is either equal to :obj:`max_length` or
            shorter if all batches finished early due to the :obj:`eos_token_id`.

        Examples::

            tokenizer = AutoTokenizer.from_pretrained('distilgpt2')   # Initialize tokenizer
            model = TFAutoModelWithLMHead.from_pretrained('distilgpt2')    # Download model and configuration from S3 and cache.
            outputs = model.generate(max_length=40)  # do greedy decoding
            print('Generated: {}'.format(tokenizer.decode(outputs[0], skip_special_tokens=True)))

            tokenizer = AutoTokenizer.from_pretrained('openai-gpt')   # Initialize tokenizer
            model = TFAutoModelWithLMHead.from_pretrained('openai-gpt')    # Download model and configuration from S3 and cache.
            input_context = 'The dog'
            input_ids = tokenizer.encode(input_context, return_tensors='tf')  # encode input context
            outputs = model.generate(input_ids=input_ids, num_beams=5, num_return_sequences=3, temperature=1.5)  # generate 3 independent sequences using beam search decoding (5 beams) with sampling from initial context 'The dog'
            for i in range(3): #  3 output sequences were generated
                print('Generated {}: {}'.format(i, tokenizer.decode(outputs[i], skip_special_tokens=True)))

            tokenizer = AutoTokenizer.from_pretrained('distilgpt2')   # Initialize tokenizer
            model = TFAutoModelWithLMHead.from_pretrained('distilgpt2')    # Download model and configuration from S3 and cache.
            input_context = 'The dog'
            input_ids = tokenizer.encode(input_context, return_tensors='tf')  # encode input context
            outputs = model.generate(input_ids=input_ids, max_length=40, temperature=0.7, num_return_sequences=3, do_sample=True)  # generate 3 candidates using sampling
            for i in range(3): #  3 output sequences were generated
                print('Generated {}: {}'.format(i, tokenizer.decode(outputs[i], skip_special_tokens=True)))

            tokenizer = AutoTokenizer.from_pretrained('ctrl')   # Initialize tokenizer
            model = TFAutoModelWithLMHead.from_pretrained('ctrl')    # Download model and configuration from S3 and cache.
            input_context = 'Legal My neighbor is'  # "Legal" is one of the control codes for ctrl
            input_ids = tokenizer.encode(input_context, return_tensors='tf')  # encode input context
            outputs = model.generate(input_ids=input_ids, max_length=50, temperature=0.7, repetition_penalty=1.2)  # generate sequences
            print('Generated: {}'.format(tokenizer.decode(outputs[0], skip_special_tokens=True)))

            tokenizer = AutoTokenizer.from_pretrained('gpt2')   # Initialize tokenizer
            model = TFAutoModelWithLMHead.from_pretrained('gpt2')    # Download model and configuration from S3 and cache.
            input_context = 'My cute dog'
            bad_words_ids = [tokenizer.encode(bad_word, add_prefix_space=True) for bad_word in ['idiot', 'stupid', 'shut up']]
            input_ids = tokenizer.encode(input_context, return_tensors='tf')  # encode input context
            outputs = model.generate(input_ids=input_ids, max_length=100, do_sample=True, bad_words_ids=bad_words_ids)  # generate sequences without allowing bad_words to be generated
        """
        if self.get_output_embeddings() is None:
            raise AttributeError('You tried to generate sequences with a model that does not have a LM Head.Please use another model class (e.g. `TFOpenAIGPTLMHeadModel`, `TFXLNetLMHeadModel`, `TFGPT2LMHeadModel`, `TFCTRLLMHeadModel`, `TFT5ForConditionalGeneration`, `TFTransfoXLLMHeadModel`)')
        max_length = (max_length if max_length is not None else self.config.max_length)
        min_length = (min_length if min_length is not None else self.config.min_length)
        do_sample = (do_sample if do_sample is not None else self.config.do_sample)
        early_stopping = (early_stopping if early_stopping is not None else self.config.early_stopping)
        use_cache = (use_cache if use_cache is not None else self.config.use_cache)
        num_beams = (num_beams if num_beams is not None else self.config.num_beams)
        temperature = (temperature if temperature is not None else self.config.temperature)
        top_k = (top_k if top_k is not None else self.config.top_k)
        top_p = (top_p if top_p is not None else self.config.top_p)
        repetition_penalty = (repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty)
        bos_token_id = (bos_token_id if bos_token_id is not None else self.config.bos_token_id)
        pad_token_id = (pad_token_id if pad_token_id is not None else self.config.pad_token_id)
        eos_token_id = (eos_token_id if eos_token_id is not None else self.config.eos_token_id)
        length_penalty = (length_penalty if length_penalty is not None else self.config.length_penalty)
        no_repeat_ngram_size = (no_repeat_ngram_size if no_repeat_ngram_size is not None else self.config.no_repeat_ngram_size)
        bad_words_ids = (bad_words_ids if bad_words_ids is not None else self.config.bad_words_ids)
        num_return_sequences = (num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences)
        decoder_start_token_id = (decoder_start_token_id if decoder_start_token_id is not None else self.config.decoder_start_token_id)
        if input_ids is not None:
            batch_size = shape_list(input_ids)[0]
        else:
            batch_size = 1
        assert (isinstance(max_length, int) and max_length > 0), '`max_length` should be a strictely positive integer.'
        assert (isinstance(min_length, int) and min_length >= 0), '`min_length` should be a positive integer.'
        assert isinstance(do_sample, bool), '`do_sample` should be a boolean.'
        assert isinstance(early_stopping, bool), '`early_stopping` should be a boolean.'
        assert isinstance(use_cache, bool), '`use_cache` should be a boolean.'
        assert (isinstance(num_beams, int) and num_beams > 0), '`num_beams` should be a strictly positive integer.'
        assert temperature > 0, '`temperature` should be strictely positive.'
        assert (isinstance(top_k, int) and top_k >= 0), '`top_k` should be a positive integer.'
        assert 0 <= top_p <= 1, '`top_p` should be between 0 and 1.'
        assert repetition_penalty >= 1.0, '`repetition_penalty` should be >= 1.'
        assert (input_ids is not None or (isinstance(bos_token_id, int) and bos_token_id >= 0)), 'If input_ids is not defined, `bos_token_id` should be a positive integer.'
        assert (pad_token_id is None or (isinstance(pad_token_id, int) and pad_token_id >= 0)), '`pad_token_id` should be a positive integer.'
        assert (eos_token_id is None or (isinstance(eos_token_id, int) and eos_token_id >= 0)), '`eos_token_id` should be a positive integer.'
        assert length_penalty > 0, '`length_penalty` should be strictely positive.'
        assert (isinstance(num_return_sequences, int) and num_return_sequences > 0), '`num_return_sequences` should be a strictely positive integer.'
        assert (bad_words_ids is None or (isinstance(bad_words_ids, list) and isinstance(bad_words_ids[0], list))), '`bad_words_ids` is either `None` or a list of lists of tokens that should not be generated'
        if input_ids is None:
            assert (isinstance(bos_token_id, int) and bos_token_id >= 0), 'you should either supply a context to complete as `input_ids` input or a `bos_token_id` (integer >= 0) as a first token to start the generation.'
            input_ids = tf.fill((batch_size, 1), bos_token_id)
        else:
            assert len(shape_list(input_ids)) == 2, 'Input prompt should be of shape (batch_size, sequence length).'
        if do_sample is False:
            if num_beams == 1:
                assert num_return_sequences == 1, 'Greedy decoding will always produce the same output for num_beams == 1 and num_return_sequences > 1. Please set num_return_sequences = 1'
            else:
                assert num_beams >= num_return_sequences, 'Greedy beam search decoding cannot return more sequences than it has beams. Please set num_beams >= num_return_sequences'
        if (attention_mask is None and pad_token_id is not None and pad_token_id in input_ids.numpy()):
            attention_mask = tf.cast(tf.math.not_equal(input_ids, pad_token_id), dtype=tf.int32)
        elif attention_mask is None:
            attention_mask = tf.ones_like(input_ids)
        if (pad_token_id is None and eos_token_id is not None):
            logger.warning('Setting `pad_token_id` to {} (first `eos_token_id`) to generate sequence'.format(eos_token_id))
            pad_token_id = eos_token_id
        cur_len = shape_list(input_ids)[1]
        vocab_size = self.config.vocab_size
        if do_sample:
            effective_batch_size = batch_size * num_return_sequences
            effective_batch_mult = num_return_sequences
        else:
            effective_batch_size = batch_size
            effective_batch_mult = 1
        if self.config.is_encoder_decoder:
            if decoder_start_token_id is None:
                decoder_start_token_id = bos_token_id
            assert decoder_start_token_id is not None, 'decoder_start_token_id or bos_token_id has to be defined for encoder-decoder generation'
            assert hasattr(self, 'get_encoder'), "{} should have a 'get_encoder' function defined".format(self)
            assert callable(self.get_encoder), '{} should be a method'.format(self.get_encoder)
            encoder = self.get_encoder()
            encoder_outputs = encoder(input_ids, attention_mask=attention_mask)
        if (num_return_sequences > 1 or num_beams > 1):
            input_ids_len = shape_list(input_ids)[-1]
            input_ids = tf.broadcast_to(tf.expand_dims(input_ids, 1), (batch_size, effective_batch_mult * num_beams, input_ids_len))
            attention_mask = tf.broadcast_to(tf.expand_dims(attention_mask, 1), (batch_size, effective_batch_mult * num_beams, input_ids_len))
            input_ids = tf.reshape(input_ids, (effective_batch_size * num_beams, input_ids_len))
            attention_mask = tf.reshape(attention_mask, (effective_batch_size * num_beams, input_ids_len))
        if self.config.is_encoder_decoder:
            input_ids = tf.ones((effective_batch_size * num_beams, 1), dtype=tf.int32) * decoder_start_token_id
            cur_len = 1
            assert batch_size == encoder_outputs[0].shape[0], f'expected encoder_outputs[0] to have 1st dimension bs={batch_size}, got {encoder_outputs[0].shape[0]} '
            expanded_batch_idxs = tf.reshape(tf.repeat(tf.expand_dims(tf.range(batch_size), -1), repeats=num_beams * effective_batch_mult, axis=1), shape=(-1, ))
            encoder_outputs = (tf.gather(encoder_outputs[0], expanded_batch_idxs, axis=0), *encoder_outputs[1:])
        else:
            encoder_outputs = None
            cur_len = shape_list(input_ids)[-1]
        assert cur_len < max_length, f'The context has {cur_len} number of tokens, but `max_length` is only {max_length}. Please make sure that `max_length` is bigger than the number of tokens, by setting either `generate(max_length=...,...)` or `config.max_length = ...`'
        if num_beams > 1:
            output = self._generate_beam_search(input_ids, cur_len=cur_len, max_length=max_length, min_length=min_length, do_sample=do_sample, early_stopping=early_stopping, temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, no_repeat_ngram_size=no_repeat_ngram_size, bad_words_ids=bad_words_ids, pad_token_id=pad_token_id, eos_token_id=eos_token_id, batch_size=effective_batch_size, num_return_sequences=num_return_sequences, length_penalty=length_penalty, num_beams=num_beams, vocab_size=vocab_size, encoder_outputs=encoder_outputs, attention_mask=attention_mask, use_cache=use_cache)
        else:
            output = self._generate_no_beam_search(input_ids, cur_len=cur_len, max_length=max_length, min_length=min_length, do_sample=do_sample, temperature=temperature, top_k=top_k, top_p=top_p, repetition_penalty=repetition_penalty, no_repeat_ngram_size=no_repeat_ngram_size, bad_words_ids=bad_words_ids, pad_token_id=pad_token_id, eos_token_id=eos_token_id, batch_size=effective_batch_size, vocab_size=vocab_size, encoder_outputs=encoder_outputs, attention_mask=attention_mask, use_cache=use_cache)
        return output
    
    def _generate_no_beam_search(self, input_ids, cur_len, max_length, min_length, do_sample, temperature, top_k, top_p, repetition_penalty, no_repeat_ngram_size, bad_words_ids, pad_token_id, eos_token_id, batch_size, vocab_size, encoder_outputs, attention_mask, use_cache):
        """Generate sequences for each example without beam search (num_beams == 1).
        All returned sequence are generated independantly.
        """
        unfinished_sents = tf.ones_like(input_ids[:, 0])
        sent_lengths = tf.ones_like(input_ids[:, 0]) * max_length
        past = encoder_outputs
        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache)
            outputs = self(**model_inputs)
            next_token_logits = outputs[0][:, -1, :]
            if self._use_cache(outputs, use_cache):
                past = outputs[1]
            if repetition_penalty != 1.0:
                next_token_logits_penalties = _create_next_token_logits_penalties(input_ids, next_token_logits, repetition_penalty)
                next_token_logits = tf.math.multiply(next_token_logits, next_token_logits_penalties)
            if no_repeat_ngram_size > 0:
                banned_tokens = calc_banned_ngram_tokens(input_ids, batch_size, no_repeat_ngram_size, cur_len)
                banned_tokens_indices_mask = []
                for banned_tokens_slice in banned_tokens:
                    banned_tokens_indices_mask.append([(True if token in banned_tokens_slice else False) for token in range(vocab_size)])
                next_token_logits = set_tensor_by_indices_to_value(next_token_logits, tf.convert_to_tensor(banned_tokens_indices_mask, dtype=tf.bool), -float('inf'))
            if bad_words_ids is not None:
                banned_tokens = calc_banned_bad_words_ids(input_ids, bad_words_ids)
                banned_tokens_indices_mask = []
                for banned_tokens_slice in banned_tokens:
                    banned_tokens_indices_mask.append([(True if token in banned_tokens_slice else False) for token in range(vocab_size)])
                next_token_logits = set_tensor_by_indices_to_value(next_token_logits, tf.convert_to_tensor(banned_tokens_indices_mask, dtype=tf.bool), -float('inf'))
            if (eos_token_id is not None and cur_len < min_length):
                is_token_logit_eos_token = tf.convert_to_tensor([(True if token is eos_token_id else False) for token in range(vocab_size)], dtype=tf.bool)
                eos_token_indices_mask = tf.broadcast_to(is_token_logit_eos_token, [batch_size, vocab_size])
                next_token_logits = set_tensor_by_indices_to_value(next_token_logits, eos_token_indices_mask, -float('inf'))
            if do_sample:
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                next_token_logits = tf_top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                next_token = tf.squeeze(tf.random.categorical(next_token_logits, dtype=tf.int32, num_samples=1), axis=1)
            else:
                next_token = tf.math.argmax(next_token_logits, axis=-1, output_type=tf.int32)
            if eos_token_id is not None:
                tokens_to_add = next_token * unfinished_sents + pad_token_id * (1 - unfinished_sents)
            else:
                tokens_to_add = next_token
            input_ids = tf.concat([input_ids, tf.expand_dims(tokens_to_add, -1)], 1)
            cur_len = cur_len + 1
            if eos_token_id is not None:
                eos_in_sents = tokens_to_add == eos_token_id
                is_sents_unfinished_and_token_to_add_is_eos = tf.math.multiply(unfinished_sents, tf.cast(eos_in_sents, tf.int32))
                sent_lengths = sent_lengths * (1 - is_sents_unfinished_and_token_to_add_is_eos) + cur_len * is_sents_unfinished_and_token_to_add_is_eos
                unfinished_sents -= is_sents_unfinished_and_token_to_add_is_eos
            if tf.math.reduce_max(unfinished_sents) == 0:
                break
            if self.config.is_encoder_decoder is False:
                attention_mask = tf.concat([attention_mask, tf.ones((shape_list(attention_mask)[0], 1), dtype=tf.int32)], axis=-1)
        min_sent_length = tf.math.reduce_min(sent_lengths)
        max_sent_length = tf.math.reduce_max(sent_lengths)
        if min_sent_length != max_sent_length:
            assert pad_token_id is not None, '`Pad_token_id` has to be defined if batches have different lengths'
            padding = tf.ones([batch_size, max_sent_length.numpy()], dtype=tf.int32) * pad_token_id
            broad_casted_sent_lengths = tf.broadcast_to(tf.expand_dims(sent_lengths, -1), [batch_size, max_sent_length])
            broad_casted_range = tf.transpose(tf.broadcast_to(tf.expand_dims(tf.range(max_sent_length), -1), [max_sent_length, batch_size]))
            decoded = tf.where(broad_casted_range < broad_casted_sent_lengths, input_ids, padding)
        else:
            decoded = input_ids
        return decoded
    
    def _generate_beam_search(self, input_ids, cur_len, max_length, min_length, do_sample, early_stopping, temperature, top_k, top_p, repetition_penalty, no_repeat_ngram_size, bad_words_ids, pad_token_id, eos_token_id, batch_size, num_return_sequences, length_penalty, num_beams, vocab_size, encoder_outputs, attention_mask, use_cache):
        """Generate sequences for each example with beam search."""
        generated_hyps = [BeamHypotheses(num_beams, max_length, length_penalty, early_stopping=early_stopping) for _ in range(batch_size)]
        if do_sample is False:
            beam_scores_begin = tf.zeros((batch_size, 1), dtype=tf.float32)
            beam_scores_end = tf.ones((batch_size, num_beams - 1), dtype=tf.float32) * -1000000000.0
            beam_scores = tf.concat([beam_scores_begin, beam_scores_end], -1)
        else:
            beam_scores = tf.zeros((batch_size, num_beams), dtype=tf.float32)
        beam_scores = tf.reshape(beam_scores, (batch_size * num_beams, ))
        past = encoder_outputs
        done = [False for _ in range(batch_size)]
        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache)
            outputs = self(**model_inputs)
            next_token_logits = outputs[0][:, -1, :]
            if self._use_cache(outputs, use_cache):
                past = outputs[1]
            if repetition_penalty != 1.0:
                next_token_logits_penalties = _create_next_token_logits_penalties(input_ids, next_token_logits, repetition_penalty)
                next_token_logits = tf.math.multiply(next_token_logits, next_token_logits_penalties)
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            scores = tf.nn.log_softmax(next_token_logits, axis=-1)
            if (eos_token_id is not None and cur_len < min_length):
                num_batch_hypotheses = batch_size * num_beams
                is_token_logit_eos_token = tf.convert_to_tensor([(True if token is eos_token_id else False) for token in range(vocab_size)], dtype=tf.bool)
                eos_token_indices_mask = tf.broadcast_to(is_token_logit_eos_token, [num_batch_hypotheses, vocab_size])
                scores = set_tensor_by_indices_to_value(scores, eos_token_indices_mask, -float('inf'))
            if no_repeat_ngram_size > 0:
                num_batch_hypotheses = batch_size * num_beams
                banned_tokens = calc_banned_ngram_tokens(input_ids, num_batch_hypotheses, no_repeat_ngram_size, cur_len)
                banned_tokens_indices_mask = []
                for banned_tokens_slice in banned_tokens:
                    banned_tokens_indices_mask.append([(True if token in banned_tokens_slice else False) for token in range(vocab_size)])
                scores = set_tensor_by_indices_to_value(scores, tf.convert_to_tensor(banned_tokens_indices_mask, dtype=tf.bool), -float('inf'))
            if bad_words_ids is not None:
                banned_tokens = calc_banned_bad_words_ids(input_ids, bad_words_ids)
                banned_tokens_indices_mask = []
                for banned_tokens_slice in banned_tokens:
                    banned_tokens_indices_mask.append([(True if token in banned_tokens_slice else False) for token in range(vocab_size)])
                scores = set_tensor_by_indices_to_value(scores, tf.convert_to_tensor(banned_tokens_indices_mask, dtype=tf.bool), -float('inf'))
            assert shape_list(scores) == [batch_size * num_beams, vocab_size]
            if do_sample:
                _scores = scores + tf.broadcast_to(beam_scores[:, None], (batch_size * num_beams, vocab_size))
                _scores = tf_top_k_top_p_filtering(_scores, top_k=top_k, top_p=top_p, min_tokens_to_keep=2)
                _scores = tf.reshape(_scores, (batch_size, num_beams * vocab_size))
                next_tokens = sample_without_replacement(_scores, num_samples=2 * num_beams)
                next_scores = tf.gather(_scores, next_tokens, batch_dims=1)
                next_scores_indices = tf.argsort(next_scores, direction='DESCENDING', axis=1)
                next_scores = tf.gather(next_scores, next_scores_indices, batch_dims=1)
                next_tokens = tf.gather(next_tokens, next_scores_indices, batch_dims=1)
            else:
                next_scores = scores + tf.broadcast_to(beam_scores[:, None], (batch_size * num_beams, vocab_size))
                next_scores = tf.reshape(next_scores, (batch_size, num_beams * vocab_size))
                (next_scores, next_tokens) = tf.math.top_k(next_scores, k=2 * num_beams, sorted=True)
            assert shape_list(next_scores) == shape_list(next_tokens) == [batch_size, 2 * num_beams]
            next_batch_beam = []
            for batch_idx in range(batch_size):
                if done[batch_idx]:
                    assert len(generated_hyps[batch_idx]) >= num_beams, 'Batch can only be done if at least {} beams have been generated'.format(num_beams)
                    assert (eos_token_id is not None and pad_token_id is not None), 'generated beams >= num_beams -> eos_token_id and pad_token have to be defined'
                    next_batch_beam.extend([(0, pad_token_id, 0)] * num_beams)
                    continue
                next_sent_beam = []
                for (beam_token_rank, (beam_token_id, beam_token_score)) in enumerate(zip(next_tokens[batch_idx], next_scores[batch_idx])):
                    beam_id = beam_token_id // vocab_size
                    token_id = beam_token_id % vocab_size
                    effective_beam_id = batch_idx * num_beams + beam_id
                    if (eos_token_id is not None and token_id.numpy() == eos_token_id):
                        is_beam_token_worse_than_top_num_beams = beam_token_rank >= num_beams
                        if is_beam_token_worse_than_top_num_beams:
                            continue
                        generated_hyps[batch_idx].add(tf.identity(input_ids[effective_beam_id]), beam_token_score.numpy())
                    else:
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))
                    if len(next_sent_beam) == num_beams:
                        break
                done[batch_idx] = (done[batch_idx] or generated_hyps[batch_idx].is_done(tf.reduce_max(next_scores[batch_idx]).numpy(), cur_len))
                assert len(next_sent_beam) == num_beams, 'Beam should always be full'
                next_batch_beam.extend(next_sent_beam)
                assert len(next_batch_beam) == num_beams * (batch_idx + 1)
            if all(done):
                break
            assert len(next_batch_beam) == batch_size * num_beams
            beam_scores = tf.convert_to_tensor([x[0] for x in next_batch_beam], dtype=tf.float32)
            beam_tokens = tf.convert_to_tensor([x[1] for x in next_batch_beam], dtype=tf.int32)
            beam_idx = tf.convert_to_tensor([x[2] for x in next_batch_beam], dtype=tf.int32)
            input_ids = tf.stack([tf.identity(input_ids[x, :]) for x in beam_idx])
            input_ids = tf.concat([input_ids, tf.expand_dims(beam_tokens, 1)], axis=-1)
            cur_len = cur_len + 1
            if past is not None:
                past = self._reorder_cache(past, beam_idx)
            if self.config.is_encoder_decoder is False:
                attention_mask = tf.concat([attention_mask, tf.ones((shape_list(attention_mask)[0], 1), dtype=tf.int32)], axis=-1)
        for batch_idx in range(batch_size):
            if done[batch_idx]:
                continue
            if (eos_token_id is not None and all(((token_id % vocab_size).numpy().item() != eos_token_id for token_id in next_tokens[batch_idx]))):
                assert tf.reduce_all(next_scores[batch_idx, :num_beams] == tf.reshape(beam_scores, (batch_size, num_beams))[batch_idx]), 'If batch_idx is not done, final next scores: {} have to equal to accumulated beam_scores: {}'.format(next_scores[:, :num_beams][batch_idx], tf.reshape(beam_scores, (batch_size, num_beams))[batch_idx])
            for beam_id in range(num_beams):
                effective_beam_id = batch_idx * num_beams + beam_id
                final_score = beam_scores[effective_beam_id].numpy().item()
                final_tokens = input_ids[effective_beam_id]
                generated_hyps[batch_idx].add(final_tokens, final_score)
        output_batch_size = (batch_size if do_sample else batch_size * num_return_sequences)
        output_num_return_sequences_per_batch = (1 if do_sample else num_return_sequences)
        sent_lengths_list = []
        best = []
        for (i, hypotheses) in enumerate(generated_hyps):
            sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
            for j in range(output_num_return_sequences_per_batch):
                best_hyp = sorted_hyps.pop()[1]
                sent_lengths_list.append(len(best_hyp))
                best.append(best_hyp)
        assert output_batch_size == len(best), 'Output batch size {} must match output beam hypotheses {}'.format(output_batch_size, len(best))
        sent_lengths = tf.convert_to_tensor(sent_lengths_list, dtype=tf.int32)
        if tf.reduce_min(sent_lengths).numpy() != tf.reduce_max(sent_lengths).numpy():
            assert pad_token_id is not None, '`Pad_token_id` has to be defined'
            sent_max_len = min(tf.reduce_max(sent_lengths).numpy() + 1, max_length)
            decoded_list = []
            for (i, hypo) in enumerate(best):
                assert sent_lengths[i] == shape_list(hypo)[0]
                if sent_lengths[i] == sent_max_len:
                    decoded_slice = hypo
                else:
                    num_pad_tokens = sent_max_len - sent_lengths[i]
                    padding = pad_token_id * tf.ones((num_pad_tokens, ), dtype=tf.int32)
                    decoded_slice = tf.concat([hypo, padding], axis=-1)
                    if sent_lengths[i] < max_length:
                        decoded_slice = tf.where(tf.range(sent_max_len, dtype=tf.int32) == sent_lengths[i], eos_token_id * tf.ones((sent_max_len, ), dtype=tf.int32), decoded_slice)
                decoded_list.append(decoded_slice)
            decoded = tf.stack(decoded_list)
        else:
            assert (len(hypo) == max_length for hypo in best)
            decoded = tf.stack(best)
        return decoded
    
    @staticmethod
    def _reorder_cache(past, beam_idx):
        return tuple((tf.gather(layer_past, beam_idx, axis=1) for layer_past in past))


def _create_next_token_logits_penalties(input_ids, logits, repetition_penalty):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.generation_tf_utils._create_next_token_logits_penalties', '_create_next_token_logits_penalties(input_ids, logits, repetition_penalty)', {'np': np, 'shape_list': shape_list, 'tf': tf, 'input_ids': input_ids, 'logits': logits, 'repetition_penalty': repetition_penalty}, 1)

def calc_banned_ngram_tokens(prev_input_ids, num_hypos, no_repeat_ngram_size, cur_len):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.generation_tf_utils.calc_banned_ngram_tokens', 'calc_banned_ngram_tokens(prev_input_ids, num_hypos, no_repeat_ngram_size, cur_len)', {'prev_input_ids': prev_input_ids, 'num_hypos': num_hypos, 'no_repeat_ngram_size': no_repeat_ngram_size, 'cur_len': cur_len}, 1)

def calc_banned_bad_words_ids(prev_input_ids, bad_words_ids):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.generation_tf_utils.calc_banned_bad_words_ids', 'calc_banned_bad_words_ids(prev_input_ids, bad_words_ids)', {'prev_input_ids': prev_input_ids, 'bad_words_ids': bad_words_ids}, 1)

def tf_top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float('Inf'), min_tokens_to_keep=1):
    """Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
        if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
            Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        Make sure we keep at least min_tokens_to_keep per batch example in the output
    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.generation_tf_utils.tf_top_k_top_p_filtering', "tf_top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float('Inf'), min_tokens_to_keep=1)", {'shape_list': shape_list, 'tf': tf, 'set_tensor_by_indices_to_value': set_tensor_by_indices_to_value, 'scatter_values_on_batch_indices': scatter_values_on_batch_indices, 'logits': logits, 'top_k': top_k, 'top_p': top_p, 'filter_value': filter_value, 'min_tokens_to_keep': min_tokens_to_keep}, 1)

def scatter_values_on_batch_indices(values, batch_indices):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.generation_tf_utils.scatter_values_on_batch_indices', 'scatter_values_on_batch_indices(values, batch_indices)', {'shape_list': shape_list, 'tf': tf, 'values': values, 'batch_indices': batch_indices}, 1)

def set_tensor_by_indices_to_value(tensor, indices, value):
    value_tensor = tf.zeros_like(tensor) + value
    return tf.where(indices, value_tensor, tensor)

def sample_without_replacement(logits, num_samples):
    """
    categorical sampling witouth replacement is currently not implemented
    the gumbel-max trick will do for now
    see https://github.com/tensorflow/tensorflow/issues/9260 for more info
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.generation_tf_utils.sample_without_replacement', 'sample_without_replacement(logits, num_samples)', {'tf': tf, 'shape_list': shape_list, 'logits': logits, 'num_samples': num_samples}, 1)

def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.generation_tf_utils.shape_list', 'shape_list(x)', {'tf': tf, 'x': x}, 1)


class BeamHypotheses(object):
    
    def __init__(self, num_beams, max_length, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1000000000.0
    
    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)
    
    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp)**self.length_penalty
        if (len(self) < self.num_beams or score > self.worst_score):
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_scores = sorted([(s, idx) for (idx, (s, _)) in enumerate(self.beams)])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)
    
    def is_done(self, best_sum_logprobs, cur_len):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """
        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            cur_score = best_sum_logprobs / cur_len**self.length_penalty
            ret = self.worst_score >= cur_score
            return ret


