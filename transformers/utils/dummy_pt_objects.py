from ..file_utils import requires_pytorch


class PyTorchBenchmark:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)



class PyTorchBenchmarkArguments:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)



class DataCollator:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)



class DataCollatorForLanguageModeling:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class DataCollatorForNextSentencePrediction:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)



class DataCollatorForPermutationLanguageModeling:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class DataCollatorForSOP:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)



class DataCollatorWithPadding:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)


def default_data_collator(*args, **kwargs):
    requires_pytorch(default_data_collator)


class GlueDataset:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)



class GlueDataTrainingArguments:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)



class LineByLineTextDataset:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)



class LineByLineWithSOPTextDataset:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)



class SquadDataset:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)



class SquadDataTrainingArguments:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)



class TextDataset:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)



class TextDatasetForNextSentencePrediction:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)


def top_k_top_p_filtering(*args, **kwargs):
    requires_pytorch(top_k_top_p_filtering)
ALBERT_PRETRAINED_MODEL_ARCHIVE_LIST = None


class AlbertForMaskedLM:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class AlbertForMultipleChoice:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class AlbertForPreTraining:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)



class AlbertForQuestionAnswering:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class AlbertForSequenceClassification:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class AlbertForTokenClassification:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class AlbertModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class AlbertPreTrainedModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)


def load_tf_weights_in_albert(*args, **kwargs):
    requires_pytorch(load_tf_weights_in_albert)
MODEL_FOR_CAUSAL_LM_MAPPING = None
MODEL_FOR_MASKED_LM_MAPPING = None
MODEL_FOR_MULTIPLE_CHOICE_MAPPING = None
MODEL_FOR_PRETRAINING_MAPPING = None
MODEL_FOR_QUESTION_ANSWERING_MAPPING = None
MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING = None
MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING = None
MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING = None
MODEL_MAPPING = None
MODEL_WITH_LM_HEAD_MAPPING = None


class AutoModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class AutoModelForCausalLM:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class AutoModelForMaskedLM:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class AutoModelForMultipleChoice:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class AutoModelForPreTraining:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class AutoModelForQuestionAnswering:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class AutoModelForSeq2SeqLM:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class AutoModelForSequenceClassification:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class AutoModelForTokenClassification:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class AutoModelWithLMHead:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)

BART_PRETRAINED_MODEL_ARCHIVE_LIST = None


class BartForConditionalGeneration:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class BartForQuestionAnswering:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class BartForSequenceClassification:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class BartModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class PretrainedBartModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)

BERT_PRETRAINED_MODEL_ARCHIVE_LIST = None


class BertForMaskedLM:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class BertForMultipleChoice:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class BertForNextSentencePrediction:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)



class BertForPreTraining:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)



class BertForQuestionAnswering:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class BertForSequenceClassification:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class BertForTokenClassification:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class BertLayer:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)



class BertLMHeadModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class BertModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class BertPreTrainedModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)


def load_tf_weights_in_bert(*args, **kwargs):
    requires_pytorch(load_tf_weights_in_bert)


class BertGenerationDecoder:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)



class BertGenerationEncoder:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)


def load_tf_weights_in_bert_generation(*args, **kwargs):
    requires_pytorch(load_tf_weights_in_bert_generation)
BLENDERBOT_PRETRAINED_MODEL_ARCHIVE_LIST = None


class BlenderbotForConditionalGeneration:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)

CAMEMBERT_PRETRAINED_MODEL_ARCHIVE_LIST = None


class CamembertForCausalLM:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)



class CamembertForMaskedLM:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class CamembertForMultipleChoice:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class CamembertForQuestionAnswering:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class CamembertForSequenceClassification:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class CamembertForTokenClassification:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class CamembertModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)

CTRL_PRETRAINED_MODEL_ARCHIVE_LIST = None


class CTRLLMHeadModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class CTRLModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class CTRLPreTrainedModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)

DEBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = None


class DebertaForSequenceClassification:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class DebertaModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class DebertaPreTrainedModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)

DISTILBERT_PRETRAINED_MODEL_ARCHIVE_LIST = None


class DistilBertForMaskedLM:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class DistilBertForMultipleChoice:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class DistilBertForQuestionAnswering:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class DistilBertForSequenceClassification:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class DistilBertForTokenClassification:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class DistilBertModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class DistilBertPreTrainedModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class DPRContextEncoder:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)



class DPRPretrainedContextEncoder:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)



class DPRPretrainedQuestionEncoder:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)



class DPRPretrainedReader:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)



class DPRQuestionEncoder:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)



class DPRReader:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)

ELECTRA_PRETRAINED_MODEL_ARCHIVE_LIST = None


class ElectraForMaskedLM:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class ElectraForMultipleChoice:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class ElectraForPreTraining:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)



class ElectraForQuestionAnswering:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class ElectraForSequenceClassification:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class ElectraForTokenClassification:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class ElectraModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class ElectraPreTrainedModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)


def load_tf_weights_in_electra(*args, **kwargs):
    requires_pytorch(load_tf_weights_in_electra)


class EncoderDecoderModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)

FLAUBERT_PRETRAINED_MODEL_ARCHIVE_LIST = None


class FlaubertForMultipleChoice:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class FlaubertForQuestionAnswering:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class FlaubertForQuestionAnsweringSimple:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class FlaubertForSequenceClassification:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class FlaubertForTokenClassification:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class FlaubertModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class FlaubertWithLMHeadModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class FSMTForConditionalGeneration:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class FSMTModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class PretrainedFSMTModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)

FUNNEL_PRETRAINED_MODEL_ARCHIVE_LIST = None


class FunnelBaseModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class FunnelForMaskedLM:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class FunnelForMultipleChoice:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class FunnelForPreTraining:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)



class FunnelForQuestionAnswering:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class FunnelForSequenceClassification:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class FunnelForTokenClassification:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class FunnelModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)


def load_tf_weights_in_funnel(*args, **kwargs):
    requires_pytorch(load_tf_weights_in_funnel)
GPT2_PRETRAINED_MODEL_ARCHIVE_LIST = None


class GPT2DoubleHeadsModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class GPT2ForSequenceClassification:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class GPT2LMHeadModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class GPT2Model:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class GPT2PreTrainedModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)


def load_tf_weights_in_gpt2(*args, **kwargs):
    requires_pytorch(load_tf_weights_in_gpt2)
LAYOUTLM_PRETRAINED_MODEL_ARCHIVE_LIST = None


class LayoutLMForMaskedLM:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class LayoutLMForTokenClassification:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class LayoutLMModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)

LONGFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = None


class LongformerForMaskedLM:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class LongformerForMultipleChoice:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class LongformerForQuestionAnswering:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class LongformerForSequenceClassification:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class LongformerForTokenClassification:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class LongformerModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class LongformerSelfAttention:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)



class LxmertEncoder:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)



class LxmertForPreTraining:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)



class LxmertForQuestionAnswering:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class LxmertModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class LxmertPreTrainedModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class LxmertVisualFeatureEncoder:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)



class LxmertXLayer:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)



class MarianMTModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class MBartForConditionalGeneration:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class MMBTForClassification:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)



class MMBTModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class ModalEmbeddings:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)

MOBILEBERT_PRETRAINED_MODEL_ARCHIVE_LIST = None


class MobileBertForMaskedLM:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class MobileBertForMultipleChoice:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class MobileBertForNextSentencePrediction:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)



class MobileBertForPreTraining:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)



class MobileBertForQuestionAnswering:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class MobileBertForSequenceClassification:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class MobileBertForTokenClassification:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class MobileBertLayer:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)



class MobileBertModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class MobileBertPreTrainedModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)


def load_tf_weights_in_mobilebert(*args, **kwargs):
    requires_pytorch(load_tf_weights_in_mobilebert)
OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_LIST = None


class OpenAIGPTDoubleHeadsModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class OpenAIGPTForSequenceClassification:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class OpenAIGPTLMHeadModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class OpenAIGPTModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class OpenAIGPTPreTrainedModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)


def load_tf_weights_in_openai_gpt(*args, **kwargs):
    requires_pytorch(load_tf_weights_in_openai_gpt)


class PegasusForConditionalGeneration:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)

PROPHETNET_PRETRAINED_MODEL_ARCHIVE_LIST = None


class ProphetNetDecoder:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)



class ProphetNetEncoder:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)



class ProphetNetForCausalLM:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)



class ProphetNetForConditionalGeneration:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class ProphetNetModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class ProphetNetPreTrainedModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class RagModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class RagSequenceForGeneration:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)



class RagTokenForGeneration:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)

REFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = None


class ReformerAttention:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)



class ReformerForMaskedLM:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class ReformerForQuestionAnswering:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class ReformerForSequenceClassification:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class ReformerLayer:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)



class ReformerModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class ReformerModelWithLMHead:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)

RETRIBERT_PRETRAINED_MODEL_ARCHIVE_LIST = None


class RetriBertModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class RetriBertPreTrainedModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)

ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = None


class RobertaForCausalLM:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)



class RobertaForMaskedLM:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class RobertaForMultipleChoice:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class RobertaForQuestionAnswering:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class RobertaForSequenceClassification:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class RobertaForTokenClassification:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class RobertaModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)

SQUEEZEBERT_PRETRAINED_MODEL_ARCHIVE_LIST = None


class SqueezeBertForMaskedLM:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class SqueezeBertForMultipleChoice:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class SqueezeBertForQuestionAnswering:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class SqueezeBertForSequenceClassification:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class SqueezeBertForTokenClassification:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class SqueezeBertModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class SqueezeBertModule:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)



class SqueezeBertPreTrainedModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)

T5_PRETRAINED_MODEL_ARCHIVE_LIST = None


class T5ForConditionalGeneration:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class T5Model:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class T5PreTrainedModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)


def load_tf_weights_in_t5(*args, **kwargs):
    requires_pytorch(load_tf_weights_in_t5)
TRANSFO_XL_PRETRAINED_MODEL_ARCHIVE_LIST = None


class AdaptiveEmbedding:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)



class TransfoXLLMHeadModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class TransfoXLModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class TransfoXLPreTrainedModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)


def load_tf_weights_in_transfo_xl(*args, **kwargs):
    requires_pytorch(load_tf_weights_in_transfo_xl)


class Conv1D:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)



class PreTrainedModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)


def apply_chunking_to_forward(*args, **kwargs):
    requires_pytorch(apply_chunking_to_forward)

def prune_layer(*args, **kwargs):
    requires_pytorch(prune_layer)
XLM_PRETRAINED_MODEL_ARCHIVE_LIST = None


class XLMForMultipleChoice:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class XLMForQuestionAnswering:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class XLMForQuestionAnsweringSimple:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class XLMForSequenceClassification:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class XLMForTokenClassification:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class XLMModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class XLMPreTrainedModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class XLMWithLMHeadModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)

XLM_PROPHETNET_PRETRAINED_MODEL_ARCHIVE_LIST = None


class XLMProphetNetDecoder:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)



class XLMProphetNetEncoder:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)



class XLMProphetNetForCausalLM:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)



class XLMProphetNetForConditionalGeneration:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class XLMProphetNetModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)

XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST = None


class XLMRobertaForCausalLM:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)



class XLMRobertaForMaskedLM:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class XLMRobertaForMultipleChoice:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class XLMRobertaForQuestionAnswering:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class XLMRobertaForSequenceClassification:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class XLMRobertaForTokenClassification:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class XLMRobertaModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)

XLNET_PRETRAINED_MODEL_ARCHIVE_LIST = None


class XLNetForMultipleChoice:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class XLNetForQuestionAnswering:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class XLNetForQuestionAnsweringSimple:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class XLNetForSequenceClassification:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class XLNetForTokenClassification:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class XLNetLMHeadModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class XLNetModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)



class XLNetPreTrainedModel:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)
    
    @classmethod
    def from_pretrained(self, *args, **kwargs):
        requires_pytorch(self)


def load_tf_weights_in_xlnet(*args, **kwargs):
    requires_pytorch(load_tf_weights_in_xlnet)


class Adafactor:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)



class AdamW:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)


def get_constant_schedule(*args, **kwargs):
    requires_pytorch(get_constant_schedule)

def get_constant_schedule_with_warmup(*args, **kwargs):
    requires_pytorch(get_constant_schedule_with_warmup)

def get_cosine_schedule_with_warmup(*args, **kwargs):
    requires_pytorch(get_cosine_schedule_with_warmup)

def get_cosine_with_hard_restarts_schedule_with_warmup(*args, **kwargs):
    requires_pytorch(get_cosine_with_hard_restarts_schedule_with_warmup)

def get_linear_schedule_with_warmup(*args, **kwargs):
    requires_pytorch(get_linear_schedule_with_warmup)

def get_polynomial_decay_schedule_with_warmup(*args, **kwargs):
    requires_pytorch(get_polynomial_decay_schedule_with_warmup)


class Trainer:
    
    def __init__(self, *args, **kwargs):
        requires_pytorch(self)


def torch_distributed_zero_first(*args, **kwargs):
    requires_pytorch(torch_distributed_zero_first)

