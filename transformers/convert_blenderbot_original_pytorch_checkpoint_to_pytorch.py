"""Convert Blenderbot checkpoint."""

import argparse
import torch
from transformers import BartConfig, BartForConditionalGeneration
from transformers.utils import logging
logging.set_verbosity_info()
logger = logging.get_logger(__name__)
PATTERNS = [['attention', 'attn'], ['encoder_attention', 'encoder_attn'], ['q_lin', 'q_proj'], ['k_lin', 'k_proj'], ['v_lin', 'v_proj'], ['out_lin', 'out_proj'], ['norm_embeddings', 'layernorm_embedding'], ['position_embeddings', 'embed_positions'], ['embeddings', 'embed_tokens'], ['ffn.lin', 'fc']]

def rename_state_dict_key(k):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.convert_blenderbot_original_pytorch_checkpoint_to_pytorch.rename_state_dict_key', 'rename_state_dict_key(k)', {'PATTERNS': PATTERNS, 'k': k}, 1)

def rename_layernorm_keys(sd):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('transformers.convert_blenderbot_original_pytorch_checkpoint_to_pytorch.rename_layernorm_keys', 'rename_layernorm_keys(sd)', {'sd': sd}, 0)
IGNORE_KEYS = ['START']

@torch.no_grad()
def convert_parlai_checkpoint(checkpoint_path, pytorch_dump_folder_path, config_json_path):
    """
    Copy/paste/tweak model's weights to our BERT structure.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('transformers.convert_blenderbot_original_pytorch_checkpoint_to_pytorch.convert_parlai_checkpoint', 'convert_parlai_checkpoint(checkpoint_path, pytorch_dump_folder_path, config_json_path)', {'BartConfig': BartConfig, 'BartForConditionalGeneration': BartForConditionalGeneration, 'IGNORE_KEYS': IGNORE_KEYS, 'rename_state_dict_key': rename_state_dict_key, 'rename_layernorm_keys': rename_layernorm_keys, 'torch': torch, 'checkpoint_path': checkpoint_path, 'pytorch_dump_folder_path': pytorch_dump_folder_path, 'config_json_path': config_json_path}, 0)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, help='like blenderbot-model.bin')
    parser.add_argument('--save_dir', default='hf_blenderbot', type=str, help='Where to save converted model.')
    parser.add_argument('--hf_config_json', default='blenderbot-3b-config.json', type=str, help='Path to config to use')
    args = parser.parse_args()
    convert_parlai_checkpoint(args.src_path, args.save_dir, args.hf_config_json)

