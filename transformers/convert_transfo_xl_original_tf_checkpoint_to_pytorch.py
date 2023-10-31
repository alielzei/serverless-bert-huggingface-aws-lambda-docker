"""Convert Transformer XL checkpoint and datasets."""

import argparse
import os
import pickle
import sys
import torch
import transformers.tokenization_transfo_xl as data_utils
from transformers import CONFIG_NAME, WEIGHTS_NAME, TransfoXLConfig, TransfoXLLMHeadModel, load_tf_weights_in_transfo_xl
from transformers.tokenization_transfo_xl import CORPUS_NAME, VOCAB_FILES_NAMES
from transformers.utils import logging
logging.set_verbosity_info()
data_utils.Vocab = data_utils.TransfoXLTokenizer
data_utils.Corpus = data_utils.TransfoXLCorpus
sys.modules['data_utils'] = data_utils
sys.modules['vocabulary'] = data_utils

def convert_transfo_xl_checkpoint_to_pytorch(tf_checkpoint_path, transfo_xl_config_file, pytorch_dump_folder_path, transfo_xl_dataset_file):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('transformers.convert_transfo_xl_original_tf_checkpoint_to_pytorch.convert_transfo_xl_checkpoint_to_pytorch', 'convert_transfo_xl_checkpoint_to_pytorch(tf_checkpoint_path, transfo_xl_config_file, pytorch_dump_folder_path, transfo_xl_dataset_file)', {'pickle': pickle, 'VOCAB_FILES_NAMES': VOCAB_FILES_NAMES, 'torch': torch, 'CORPUS_NAME': CORPUS_NAME, 'os': os, 'TransfoXLConfig': TransfoXLConfig, 'TransfoXLLMHeadModel': TransfoXLLMHeadModel, 'load_tf_weights_in_transfo_xl': load_tf_weights_in_transfo_xl, 'WEIGHTS_NAME': WEIGHTS_NAME, 'CONFIG_NAME': CONFIG_NAME, 'tf_checkpoint_path': tf_checkpoint_path, 'transfo_xl_config_file': transfo_xl_config_file, 'pytorch_dump_folder_path': pytorch_dump_folder_path, 'transfo_xl_dataset_file': transfo_xl_dataset_file}, 0)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pytorch_dump_folder_path', default=None, type=str, required=True, help='Path to the folder to store the PyTorch model or dataset/vocab.')
    parser.add_argument('--tf_checkpoint_path', default='', type=str, help='An optional path to a TensorFlow checkpoint path to be converted.')
    parser.add_argument('--transfo_xl_config_file', default='', type=str, help='An optional config json file corresponding to the pre-trained BERT model. \nThis specifies the model architecture.')
    parser.add_argument('--transfo_xl_dataset_file', default='', type=str, help='An optional dataset file to be converted in a vocabulary.')
    args = parser.parse_args()
    convert_transfo_xl_checkpoint_to_pytorch(args.tf_checkpoint_path, args.transfo_xl_config_file, args.pytorch_dump_folder_path, args.transfo_xl_dataset_file)

