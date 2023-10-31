"""Convert RoBERTa checkpoint."""

import argparse
import pathlib
import fairseq
import torch
from fairseq.models.roberta import RobertaModel as FairseqRobertaModel
from fairseq.modules import TransformerSentenceEncoderLayer
from packaging import version
from transformers.modeling_bert import BertIntermediate, BertLayer, BertOutput, BertSelfAttention, BertSelfOutput
from transformers.modeling_roberta import RobertaConfig, RobertaForMaskedLM, RobertaForSequenceClassification
from transformers.utils import logging
if version.parse(fairseq.__version__) < version.parse('0.9.0'):
    raise Exception('requires fairseq >= 0.9.0')
logging.set_verbosity_info()
logger = logging.get_logger(__name__)
SAMPLE_TEXT = 'Hello world! cécé herlolip'

def convert_roberta_checkpoint_to_pytorch(roberta_checkpoint_path: str, pytorch_dump_folder_path: str, classification_head: bool):
    """
    Copy/paste/tweak roberta's weights to our BERT structure.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('transformers.convert_roberta_original_pytorch_checkpoint_to_pytorch.convert_roberta_checkpoint_to_pytorch', 'convert_roberta_checkpoint_to_pytorch(roberta_checkpoint_path, pytorch_dump_folder_path, classification_head)', {'FairseqRobertaModel': FairseqRobertaModel, 'RobertaConfig': RobertaConfig, 'RobertaForSequenceClassification': RobertaForSequenceClassification, 'RobertaForMaskedLM': RobertaForMaskedLM, 'torch': torch, 'BertLayer': BertLayer, 'TransformerSentenceEncoderLayer': TransformerSentenceEncoderLayer, 'BertSelfAttention': BertSelfAttention, 'BertSelfOutput': BertSelfOutput, 'BertIntermediate': BertIntermediate, 'BertOutput': BertOutput, 'SAMPLE_TEXT': SAMPLE_TEXT, 'pathlib': pathlib, 'roberta_checkpoint_path': roberta_checkpoint_path, 'pytorch_dump_folder_path': pytorch_dump_folder_path, 'classification_head': classification_head}, 0)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--roberta_checkpoint_path', default=None, type=str, required=True, help='Path the official PyTorch dump.')
    parser.add_argument('--pytorch_dump_folder_path', default=None, type=str, required=True, help='Path to the output PyTorch model.')
    parser.add_argument('--classification_head', action='store_true', help='Whether to convert a final classification head.')
    args = parser.parse_args()
    convert_roberta_checkpoint_to_pytorch(args.roberta_checkpoint_path, args.pytorch_dump_folder_path, args.classification_head)

