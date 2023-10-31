"""Convert ALBERT checkpoint."""

import argparse
import torch
from transformers import AlbertConfig, AlbertForPreTraining, load_tf_weights_in_albert
from transformers.utils import logging
logging.set_verbosity_info()

def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, albert_config_file, pytorch_dump_path):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('transformers.convert_albert_original_tf_checkpoint_to_pytorch.convert_tf_checkpoint_to_pytorch', 'convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, albert_config_file, pytorch_dump_path)', {'AlbertConfig': AlbertConfig, 'AlbertForPreTraining': AlbertForPreTraining, 'load_tf_weights_in_albert': load_tf_weights_in_albert, 'torch': torch, 'tf_checkpoint_path': tf_checkpoint_path, 'albert_config_file': albert_config_file, 'pytorch_dump_path': pytorch_dump_path}, 0)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tf_checkpoint_path', default=None, type=str, required=True, help='Path to the TensorFlow checkpoint path.')
    parser.add_argument('--albert_config_file', default=None, type=str, required=True, help='The config json file corresponding to the pre-trained ALBERT model. \nThis specifies the model architecture.')
    parser.add_argument('--pytorch_dump_path', default=None, type=str, required=True, help='Path to the output PyTorch model.')
    args = parser.parse_args()
    convert_tf_checkpoint_to_pytorch(args.tf_checkpoint_path, args.albert_config_file, args.pytorch_dump_path)

