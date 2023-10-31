"""Convert BART checkpoint."""

import argparse
import os
from pathlib import Path
import fairseq
import torch
from packaging import version
from transformers import BartConfig, BartForConditionalGeneration, BartForSequenceClassification, BartModel, BartTokenizer
from transformers.modeling_bart import _make_linear_from_emb
from transformers.utils import logging
FAIRSEQ_MODELS = ['bart.large', 'bart.large.mnli', 'bart.large.cnn', 'bart_xsum/model.pt']
extra_arch = {'bart.large': BartModel, 'bart.large.mnli': BartForSequenceClassification}
if version.parse(fairseq.__version__) < version.parse('0.9.0'):
    raise Exception('requires fairseq >= 0.9.0')
logging.set_verbosity_info()
logger = logging.get_logger(__name__)
SAMPLE_TEXT = ' Hello world! cécé herlolip'
mnli_rename_keys = [('model.classification_heads.mnli.dense.weight', 'classification_head.dense.weight'), ('model.classification_heads.mnli.dense.bias', 'classification_head.dense.bias'), ('model.classification_heads.mnli.out_proj.weight', 'classification_head.out_proj.weight'), ('model.classification_heads.mnli.out_proj.bias', 'classification_head.out_proj.bias')]

def remove_ignore_keys_(state_dict):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('transformers.convert_bart_original_pytorch_checkpoint_to_pytorch.remove_ignore_keys_', 'remove_ignore_keys_(state_dict)', {'state_dict': state_dict}, 0)

def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val

def load_xsum_checkpoint(checkpoint_path):
    """Checkpoint path should end in model.pt"""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.convert_bart_original_pytorch_checkpoint_to_pytorch.load_xsum_checkpoint', 'load_xsum_checkpoint(checkpoint_path)', {'torch': torch, 'checkpoint_path': checkpoint_path}, 1)

@torch.no_grad()
def convert_bart_checkpoint(checkpoint_path, pytorch_dump_folder_path, hf_checkpoint_name=None):
    """
    Copy/paste/tweak model's weights to our BERT structure.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('transformers.convert_bart_original_pytorch_checkpoint_to_pytorch.convert_bart_checkpoint', 'convert_bart_checkpoint(checkpoint_path, pytorch_dump_folder_path, hf_checkpoint_name=None)', {'os': os, 'load_xsum_checkpoint': load_xsum_checkpoint, 'BartConfig': BartConfig, 'SAMPLE_TEXT': SAMPLE_TEXT, 'BartTokenizer': BartTokenizer, 'remove_ignore_keys_': remove_ignore_keys_, 'mnli_rename_keys': mnli_rename_keys, 'rename_key': rename_key, 'BartForSequenceClassification': BartForSequenceClassification, 'BartModel': BartModel, 'BartForConditionalGeneration': BartForConditionalGeneration, '_make_linear_from_emb': _make_linear_from_emb, 'Path': Path, 'torch': torch, 'checkpoint_path': checkpoint_path, 'pytorch_dump_folder_path': pytorch_dump_folder_path, 'hf_checkpoint_name': hf_checkpoint_name}, 0)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fairseq_path', type=str, help='bart.large, bart.large.cnn or a path to a model.pt on local filesystem.')
    parser.add_argument('pytorch_dump_folder_path', default=None, type=str, help='Path to the output PyTorch model.')
    parser.add_argument('--hf_config', default=None, type=str, help='Which huggingface architecture to use: bart-large-xsum')
    args = parser.parse_args()
    convert_bart_checkpoint(args.fairseq_path, args.pytorch_dump_folder_path, hf_checkpoint_name=args.hf_config)

