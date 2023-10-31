"""Convert ELECTRA checkpoint."""

import argparse
import torch
from transformers import ElectraConfig, ElectraForMaskedLM, ElectraForPreTraining, load_tf_weights_in_electra
from transformers.utils import logging
logging.set_verbosity_info()

def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, config_file, pytorch_dump_path, discriminator_or_generator):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('transformers.convert_electra_original_tf_checkpoint_to_pytorch.convert_tf_checkpoint_to_pytorch', 'convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, config_file, pytorch_dump_path, discriminator_or_generator)', {'ElectraConfig': ElectraConfig, 'ElectraForPreTraining': ElectraForPreTraining, 'ElectraForMaskedLM': ElectraForMaskedLM, 'load_tf_weights_in_electra': load_tf_weights_in_electra, 'torch': torch, 'tf_checkpoint_path': tf_checkpoint_path, 'config_file': config_file, 'pytorch_dump_path': pytorch_dump_path, 'discriminator_or_generator': discriminator_or_generator}, 0)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tf_checkpoint_path', default=None, type=str, required=True, help='Path to the TensorFlow checkpoint path.')
    parser.add_argument('--config_file', default=None, type=str, required=True, help='The config json file corresponding to the pre-trained model. \nThis specifies the model architecture.')
    parser.add_argument('--pytorch_dump_path', default=None, type=str, required=True, help='Path to the output PyTorch model.')
    parser.add_argument('--discriminator_or_generator', default=None, type=str, required=True, help="Whether to export the generator or the discriminator. Should be a string, either 'discriminator' or 'generator'.")
    args = parser.parse_args()
    convert_tf_checkpoint_to_pytorch(args.tf_checkpoint_path, args.config_file, args.pytorch_dump_path, args.discriminator_or_generator)

