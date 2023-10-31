"""Convert Funnel checkpoint."""

import argparse
import logging
import torch
from transformers import FunnelConfig, FunnelForPreTraining, load_tf_weights_in_funnel
logging.basicConfig(level=logging.INFO)

def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, config_file, pytorch_dump_path):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('transformers.convert_funnel_original_tf_checkpoint_to_pytorch.convert_tf_checkpoint_to_pytorch', 'convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, config_file, pytorch_dump_path)', {'FunnelConfig': FunnelConfig, 'FunnelForPreTraining': FunnelForPreTraining, 'load_tf_weights_in_funnel': load_tf_weights_in_funnel, 'torch': torch, 'tf_checkpoint_path': tf_checkpoint_path, 'config_file': config_file, 'pytorch_dump_path': pytorch_dump_path}, 0)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tf_checkpoint_path', default=None, type=str, required=True, help='Path to the TensorFlow checkpoint path.')
    parser.add_argument('--config_file', default=None, type=str, required=True, help='The config json file corresponding to the pre-trained model. \nThis specifies the model architecture.')
    parser.add_argument('--pytorch_dump_path', default=None, type=str, required=True, help='Path to the output PyTorch model.')
    args = parser.parse_args()
    convert_tf_checkpoint_to_pytorch(args.tf_checkpoint_path, args.config_file, args.pytorch_dump_path)

