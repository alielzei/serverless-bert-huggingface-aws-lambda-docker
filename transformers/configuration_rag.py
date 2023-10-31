""" RAG model configuration """

import copy
from .configuration_utils import PretrainedConfig
from .file_utils import add_start_docstrings
RAG_CONFIG_DOC = '\n    :class:`~transformers.RagConfig` stores the configuration of a `RagModel`.\n    Configuration objects inherit from  :class:`~transformers.PretrainedConfig` and can be used\n    to control the model outputs. Read the documentation from  :class:`~transformers.PretrainedConfig`\n    for more information.\n\n    Args:\n        title_sep (:obj:`str`, `optional`, defaults to  ``" / "``):\n            Separator inserted between the title and the text of the retrieved document when calling :class:`~transformers.RagRetriever`.\n        doc_sep (:obj:`str`, `optional`, defaults to  ``" // "``):\n            Separator inserted between the the text of the retrieved document and the original input when calliang :class:`~transformers.RagRetriever`.\n        n_docs (:obj:`int`, `optional`, defaults to 5):\n            Number of documents to retrieve.\n        max_combined_length (:obj:`int`, `optional`, defaults to 300):\n            Max length of contextualized input returned by :meth:`~transformers.RagRetriever.__call__`.\n        retrieval_vector_size (:obj:`int`, `optional`, defaults to 768):\n            Dimensionality of the document embeddings indexed by :class:`~transformers.RagRetriever`.\n        retrieval_batch_size (:obj:`int`, `optional`, defaults to 8):\n            Retrieval batch size, defined as the number of queries issues concurrently to the faiss index excapsulated\n            :class:`~transformers.RagRetriever`.\n        dataset (:obj:`str`, `optional`, defaults to :obj:`"wiki_dpr"`):\n            A dataset identifier of the indexed dataset in HuggingFace Datasets (list all available datasets and\n            ids using :obj:`datasets.list_datasets()`).\n        dataset_split (:obj:`str`, `optional`, defaults to :obj:`"train"`)\n            Which split of the :obj:`dataset` to load.\n        index_name (:obj:`str`, `optional`, defaults to :obj:`"compressed"`)\n            The index name of the index associated with the :obj:`dataset`. One can choose between :obj:`"legacy"`,\n            :obj:`"exact"` and :obj:`"compressed"`.\n        index_path (:obj:`str`, `optional`)\n            The path to the serialized faiss index on disk.\n        passages_path: (:obj:`str`, `optional`):\n            A path to text passages compatible with the faiss index. Required if using\n            :class:`~transformers.retrieval_rag.LegacyIndex`\n        use_dummy_dataset (:obj:`bool`, `optional`, defaults to ``False``)\n            Whether to load a "dummy" variant of the dataset specified by :obj:`dataset`.\n        label_smoothing (:obj:`float`, `optional`, defaults to 0.0):\n            Only relevant if ``return_loss`` is set to :obj:`True`. Controls the ``epsilon`` parameter value for label\n            smoothing in the loss calculation. If set to 0, no label smoothing is performed.\n        do_marginalize (:obj:`bool`, `optional`, defaults to :obj:`False`):\n            If :obj:`True`, the logits are marginalized over all documents\n            by making use of ``torch.nn.functional.log_softmax``.\n        reduce_loss (:obj:`bool`, `optional`, defaults to :obj:`False`):\n            Whether or not to reduce the NLL loss using the ``torch.Tensor.sum`` operation.\n        do_deduplication (:obj:`bool`, `optional`, defaults to :obj:`True`):\n            Whether or not to deduplicate the generations from different context documents for a given input.\n            Has to be set to :obj:`False` if used while training with distributed backend.\n        exclude_bos_score (:obj:`bool`, `optional`, defaults to :obj:`False`):\n            Whether or not to disregard the BOS token when computing the loss.\n        output_retrieved(:obj:`bool`, `optional`, defaults to :obj:`False`):\n            If set to ``True``, :obj:`retrieved_doc_embeds`, :obj:`retrieved_doc_ids`, :obj:`context_input_ids` and\n            :obj:`context_attention_mask` are returned. See returned tensors for more detail.\n'


@add_start_docstrings(RAG_CONFIG_DOC)
class RagConfig(PretrainedConfig):
    model_type = 'rag'
    
    def __init__(self, vocab_size=None, is_encoder_decoder=True, prefix=None, bos_token_id=None, pad_token_id=None, eos_token_id=None, decoder_start_token_id=None, title_sep=' / ', doc_sep=' // ', n_docs=5, max_combined_length=300, retrieval_vector_size=768, retrieval_batch_size=8, dataset='wiki_dpr', dataset_split='train', index_name='compressed', index_path=None, passages_path=None, use_dummy_dataset=False, reduce_loss=False, label_smoothing=0.0, do_deduplication=True, exclude_bos_score=False, do_marginalize=False, output_retrieved=False, **kwargs):
        super().__init__(bos_token_id=bos_token_id, pad_token_id=pad_token_id, eos_token_id=eos_token_id, decoder_start_token_id=decoder_start_token_id, is_encoder_decoder=is_encoder_decoder, prefix=prefix, vocab_size=vocab_size, **kwargs)
        assert ('question_encoder' in kwargs and 'generator' in kwargs), 'Config has to be initialized with question_encoder and generator config'
        question_encoder_config = kwargs.pop('question_encoder')
        question_encoder_model_type = question_encoder_config.pop('model_type')
        decoder_config = kwargs.pop('generator')
        decoder_model_type = decoder_config.pop('model_type')
        from .configuration_auto import AutoConfig
        self.question_encoder = AutoConfig.for_model(question_encoder_model_type, **question_encoder_config)
        self.generator = AutoConfig.for_model(decoder_model_type, **decoder_config)
        self.reduce_loss = reduce_loss
        self.label_smoothing = label_smoothing
        self.exclude_bos_score = exclude_bos_score
        self.do_marginalize = do_marginalize
        self.title_sep = title_sep
        self.doc_sep = doc_sep
        self.n_docs = n_docs
        self.max_combined_length = max_combined_length
        self.dataset = dataset
        self.dataset_split = dataset_split
        self.index_name = index_name
        self.retrieval_vector_size = retrieval_vector_size
        self.retrieval_batch_size = retrieval_batch_size
        self.passages_path = passages_path
        self.index_path = index_path
        self.use_dummy_dataset = use_dummy_dataset
        self.output_retrieved = output_retrieved
        self.do_deduplication = do_deduplication
    
    @classmethod
    def from_question_encoder_generator_configs(cls, question_encoder_config: PretrainedConfig, generator_config: PretrainedConfig, **kwargs) -> PretrainedConfig:
        """
        Instantiate a :class:`~transformers.EncoderDecoderConfig` (or a derived class) from a pre-trained encoder model configuration and decoder model configuration.

        Returns:
            :class:`EncoderDecoderConfig`: An instance of a configuration object
        """
        return cls(question_encoder=question_encoder_config.to_dict(), generator=generator_config.to_dict(), **kwargs)
    
    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default :meth:`~transformers.PretrainedConfig.to_dict`.

        Returns:
            :obj:`Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output['question_encoder'] = self.question_encoder.to_dict()
        output['generator'] = self.generator.to_dict()
        output['model_type'] = self.__class__.model_type
        return output


