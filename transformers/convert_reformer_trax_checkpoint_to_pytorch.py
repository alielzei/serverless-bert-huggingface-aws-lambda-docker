"""Convert Reformer checkpoint."""

import argparse
import pickle
import numpy as np
import torch
from transformers import ReformerConfig, ReformerModelWithLMHead
from transformers.utils import logging
logging.set_verbosity_info()

def set_param(torch_layer, weight, bias=None):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('transformers.convert_reformer_trax_checkpoint_to_pytorch.set_param', 'set_param(torch_layer, weight, bias=None)', {'torch': torch, 'torch_layer': torch_layer, 'weight': weight, 'bias': bias}, 0)

def set_layer_weights_in_torch_lsh(weights, torch_layer, hidden_size):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('transformers.convert_reformer_trax_checkpoint_to_pytorch.set_layer_weights_in_torch_lsh', 'set_layer_weights_in_torch_lsh(weights, torch_layer, hidden_size)', {'np': np, 'set_param': set_param, 'torch': torch, 'weights': weights, 'torch_layer': torch_layer, 'hidden_size': hidden_size}, 0)

def set_layer_weights_in_torch_local(weights, torch_layer, hidden_size):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('transformers.convert_reformer_trax_checkpoint_to_pytorch.set_layer_weights_in_torch_local', 'set_layer_weights_in_torch_local(weights, torch_layer, hidden_size)', {'np': np, 'set_param': set_param, 'torch': torch, 'weights': weights, 'torch_layer': torch_layer, 'hidden_size': hidden_size}, 0)

def set_block_weights_in_torch(weights, torch_block, hidden_size):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('transformers.convert_reformer_trax_checkpoint_to_pytorch.set_block_weights_in_torch', 'set_block_weights_in_torch(weights, torch_block, hidden_size)', {'np': np, 'set_param': set_param, 'torch': torch, 'set_layer_weights_in_torch_lsh': set_layer_weights_in_torch_lsh, 'set_layer_weights_in_torch_local': set_layer_weights_in_torch_local, 'weights': weights, 'torch_block': torch_block, 'hidden_size': hidden_size}, 0)

def set_model_weights_in_torch(weights, torch_model, hidden_size):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('transformers.convert_reformer_trax_checkpoint_to_pytorch.set_model_weights_in_torch', 'set_model_weights_in_torch(weights, torch_model, hidden_size)', {'np': np, 'set_param': set_param, 'torch': torch, 'set_block_weights_in_torch': set_block_weights_in_torch, 'weights': weights, 'torch_model': torch_model, 'hidden_size': hidden_size}, 0)

def convert_trax_checkpoint_to_pytorch(trax_model_pkl_path, config_file, pytorch_dump_path):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('transformers.convert_reformer_trax_checkpoint_to_pytorch.convert_trax_checkpoint_to_pytorch', 'convert_trax_checkpoint_to_pytorch(trax_model_pkl_path, config_file, pytorch_dump_path)', {'ReformerConfig': ReformerConfig, 'ReformerModelWithLMHead': ReformerModelWithLMHead, 'pickle': pickle, 'set_model_weights_in_torch': set_model_weights_in_torch, 'torch': torch, 'trax_model_pkl_path': trax_model_pkl_path, 'config_file': config_file, 'pytorch_dump_path': pytorch_dump_path}, 0)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trax_model_pkl_path', default=None, type=str, required=True, help='Path to the TensorFlow checkpoint path.')
    parser.add_argument('--config_file', default=None, type=str, required=True, help='The config json file corresponding to the pre-trained Reformer model. \nThis specifies the model architecture.')
    parser.add_argument('--pytorch_dump_path', default=None, type=str, required=True, help='Path to the output PyTorch model.')
    args = parser.parse_args()
    convert_trax_checkpoint_to_pytorch(args.trax_model_pkl_path, args.config_file, args.pytorch_dump_path)

