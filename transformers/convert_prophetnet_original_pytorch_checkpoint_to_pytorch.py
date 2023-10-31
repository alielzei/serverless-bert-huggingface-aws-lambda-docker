"""Convert ProphetNet checkpoint."""

import argparse
import torch
from transformers import logging
from transformers.modeling_prophetnet import ProphetNetForConditionalGeneration
from transformers.modeling_xlm_prophetnet import XLMProphetNetForConditionalGeneration
from transformers_old.modeling_prophetnet import ProphetNetForConditionalGeneration as ProphetNetForConditionalGenerationOld
from transformers_old.modeling_xlm_prophetnet import XLMProphetNetForConditionalGeneration as XLMProphetNetForConditionalGenerationOld
logger = logging.get_logger(__name__)
logging.set_verbosity_info()

def convert_prophetnet_checkpoint_to_pytorch(prophetnet_checkpoint_path: str, pytorch_dump_folder_path: str):
    """
    Copy/paste/tweak prohpetnet's weights to our prophetnet structure.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('transformers.convert_prophetnet_original_pytorch_checkpoint_to_pytorch.convert_prophetnet_checkpoint_to_pytorch', 'convert_prophetnet_checkpoint_to_pytorch(prophetnet_checkpoint_path, pytorch_dump_folder_path)', {'XLMProphetNetForConditionalGenerationOld': XLMProphetNetForConditionalGenerationOld, 'XLMProphetNetForConditionalGeneration': XLMProphetNetForConditionalGeneration, 'ProphetNetForConditionalGenerationOld': ProphetNetForConditionalGenerationOld, 'ProphetNetForConditionalGeneration': ProphetNetForConditionalGeneration, 'logger': logger, 'torch': torch, 'prophetnet_checkpoint_path': prophetnet_checkpoint_path, 'pytorch_dump_folder_path': pytorch_dump_folder_path}, 0)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prophetnet_checkpoint_path', default=None, type=str, required=True, help='Path the official PyTorch dump.')
    parser.add_argument('--pytorch_dump_folder_path', default=None, type=str, required=True, help='Path to the output PyTorch model.')
    args = parser.parse_args()
    convert_prophetnet_checkpoint_to_pytorch(args.prophetnet_checkpoint_path, args.pytorch_dump_folder_path)

