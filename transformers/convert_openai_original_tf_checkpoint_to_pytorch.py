"""Convert OpenAI GPT checkpoint."""

import argparse
import torch
from transformers import CONFIG_NAME, WEIGHTS_NAME, OpenAIGPTConfig, OpenAIGPTModel, load_tf_weights_in_openai_gpt
from transformers.utils import logging
logging.set_verbosity_info()

def convert_openai_checkpoint_to_pytorch(openai_checkpoint_folder_path, openai_config_file, pytorch_dump_folder_path):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('transformers.convert_openai_original_tf_checkpoint_to_pytorch.convert_openai_checkpoint_to_pytorch', 'convert_openai_checkpoint_to_pytorch(openai_checkpoint_folder_path, openai_config_file, pytorch_dump_folder_path)', {'OpenAIGPTConfig': OpenAIGPTConfig, 'OpenAIGPTModel': OpenAIGPTModel, 'load_tf_weights_in_openai_gpt': load_tf_weights_in_openai_gpt, 'WEIGHTS_NAME': WEIGHTS_NAME, 'CONFIG_NAME': CONFIG_NAME, 'torch': torch, 'openai_checkpoint_folder_path': openai_checkpoint_folder_path, 'openai_config_file': openai_config_file, 'pytorch_dump_folder_path': pytorch_dump_folder_path}, 0)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--openai_checkpoint_folder_path', default=None, type=str, required=True, help='Path to the TensorFlow checkpoint path.')
    parser.add_argument('--pytorch_dump_folder_path', default=None, type=str, required=True, help='Path to the output PyTorch model.')
    parser.add_argument('--openai_config_file', default='', type=str, help='An optional config json file corresponding to the pre-trained OpenAI model. \nThis specifies the model architecture.')
    args = parser.parse_args()
    convert_openai_checkpoint_to_pytorch(args.openai_checkpoint_folder_path, args.openai_config_file, args.pytorch_dump_folder_path)

