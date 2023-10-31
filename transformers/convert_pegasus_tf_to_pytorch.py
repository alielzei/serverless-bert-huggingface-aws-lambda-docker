import argparse
import os
from pathlib import Path
from typing import Dict
import tensorflow as tf
import torch
from tqdm import tqdm
from transformers import PegasusConfig, PegasusForConditionalGeneration, PegasusTokenizer
from transformers.configuration_pegasus import DEFAULTS, task_specific_params
PATTERNS = [['memory_attention', 'encoder_attn'], ['attention', 'attn'], ['/', '.'], ['.LayerNorm.gamma', '_layer_norm.weight'], ['.LayerNorm.beta', '_layer_norm.bias'], ['r.layer_', 'r.layers.'], ['output_proj', 'out_proj'], ['ffn.dense_1.', 'fc2.'], ['ffn.dense.', 'fc1.'], ['ffn_layer_norm', 'final_layer_norm'], ['kernel', 'weight'], ['encoder_layer_norm.', 'encoder.layer_norm.'], ['decoder_layer_norm.', 'decoder.layer_norm.'], ['embeddings.weights', 'shared.weight']]

def rename_state_dict_key(k):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.convert_pegasus_tf_to_pytorch.rename_state_dict_key', 'rename_state_dict_key(k)', {'PATTERNS': PATTERNS, 'k': k}, 1)

def convert_pegasus(tf_weights: dict, cfg_updates: dict) -> PegasusForConditionalGeneration:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.convert_pegasus_tf_to_pytorch.convert_pegasus', 'convert_pegasus(tf_weights, cfg_updates)', {'DEFAULTS': DEFAULTS, 'PegasusConfig': PegasusConfig, 'PegasusForConditionalGeneration': PegasusForConditionalGeneration, 'rename_state_dict_key': rename_state_dict_key, 'torch': torch, 'tf_weights': tf_weights, 'cfg_updates': cfg_updates}, 1)

def get_tf_weights_as_numpy(path='./ckpt/aeslc/model.ckpt-32000') -> Dict:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.convert_pegasus_tf_to_pytorch.get_tf_weights_as_numpy', "get_tf_weights_as_numpy(path='./ckpt/aeslc/model.ckpt-32000')", {'tf': tf, 'tqdm': tqdm, 'path': path}, 1)

def convert_pegasus_ckpt_to_pytorch(ckpt_path: str, save_dir: str):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('transformers.convert_pegasus_tf_to_pytorch.convert_pegasus_ckpt_to_pytorch', 'convert_pegasus_ckpt_to_pytorch(ckpt_path, save_dir)', {'Path': Path, 'task_specific_params': task_specific_params, 'PegasusTokenizer': PegasusTokenizer, 'get_tf_weights_as_numpy': get_tf_weights_as_numpy, 'convert_pegasus': convert_pegasus, 'torch': torch, 'ckpt_path': ckpt_path, 'save_dir': save_dir}, 0)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('tf_ckpt_path', type=str, help='passed to tf.train.list_variables')
    parser.add_argument('save_dir', default=None, type=str, help='Path to the output PyTorch model.')
    args = parser.parse_args()
    if args.save_dir is None:
        dataset = Path(args.tf_ckpt_path).parent.name
        args.save_dir = os.path.join('pegasus', dataset)
    convert_pegasus_ckpt_to_pytorch(args.tf_ckpt_path, args.save_dir)

