"""Convert OpenAI GPT checkpoint."""

import argparse
import torch
from transformers import CONFIG_NAME, WEIGHTS_NAME, GPT2Config, GPT2Model, load_tf_weights_in_gpt2
from transformers.utils import logging
logging.set_verbosity_info()

def convert_gpt2_checkpoint_to_pytorch(gpt2_checkpoint_path, gpt2_config_file, pytorch_dump_folder_path):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('transformers.convert_gpt2_original_tf_checkpoint_to_pytorch.convert_gpt2_checkpoint_to_pytorch', 'convert_gpt2_checkpoint_to_pytorch(gpt2_checkpoint_path, gpt2_config_file, pytorch_dump_folder_path)', {'GPT2Config': GPT2Config, 'GPT2Model': GPT2Model, 'load_tf_weights_in_gpt2': load_tf_weights_in_gpt2, 'WEIGHTS_NAME': WEIGHTS_NAME, 'CONFIG_NAME': CONFIG_NAME, 'torch': torch, 'gpt2_checkpoint_path': gpt2_checkpoint_path, 'gpt2_config_file': gpt2_config_file, 'pytorch_dump_folder_path': pytorch_dump_folder_path}, 0)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpt2_checkpoint_path', default=None, type=str, required=True, help='Path to the TensorFlow checkpoint path.')
    parser.add_argument('--pytorch_dump_folder_path', default=None, type=str, required=True, help='Path to the output PyTorch model.')
    parser.add_argument('--gpt2_config_file', default='', type=str, help='An optional config json file corresponding to the pre-trained OpenAI model. \nThis specifies the model architecture.')
    args = parser.parse_args()
    convert_gpt2_checkpoint_to_pytorch(args.gpt2_checkpoint_path, args.gpt2_config_file, args.pytorch_dump_folder_path)

