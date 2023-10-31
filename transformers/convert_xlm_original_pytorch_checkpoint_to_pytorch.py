"""Convert OpenAI GPT checkpoint."""

import argparse
import json
import numpy
import torch
from transformers import CONFIG_NAME, WEIGHTS_NAME
from transformers.tokenization_xlm import VOCAB_FILES_NAMES
from transformers.utils import logging
logging.set_verbosity_info()

def convert_xlm_checkpoint_to_pytorch(xlm_checkpoint_path, pytorch_dump_folder_path):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('transformers.convert_xlm_original_pytorch_checkpoint_to_pytorch.convert_xlm_checkpoint_to_pytorch', 'convert_xlm_checkpoint_to_pytorch(xlm_checkpoint_path, pytorch_dump_folder_path)', {'torch': torch, 'numpy': numpy, 'WEIGHTS_NAME': WEIGHTS_NAME, 'CONFIG_NAME': CONFIG_NAME, 'VOCAB_FILES_NAMES': VOCAB_FILES_NAMES, 'json': json, 'xlm_checkpoint_path': xlm_checkpoint_path, 'pytorch_dump_folder_path': pytorch_dump_folder_path}, 0)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--xlm_checkpoint_path', default=None, type=str, required=True, help='Path the official PyTorch dump.')
    parser.add_argument('--pytorch_dump_folder_path', default=None, type=str, required=True, help='Path to the output PyTorch model.')
    args = parser.parse_args()
    convert_xlm_checkpoint_to_pytorch(args.xlm_checkpoint_path, args.pytorch_dump_folder_path)

