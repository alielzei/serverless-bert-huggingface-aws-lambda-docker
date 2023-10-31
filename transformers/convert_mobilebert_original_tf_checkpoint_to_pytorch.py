import argparse
import torch
from transformers import MobileBertConfig, MobileBertForPreTraining, load_tf_weights_in_mobilebert
from transformers.utils import logging
logging.set_verbosity_info()

def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, mobilebert_config_file, pytorch_dump_path):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('transformers.convert_mobilebert_original_tf_checkpoint_to_pytorch.convert_tf_checkpoint_to_pytorch', 'convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, mobilebert_config_file, pytorch_dump_path)', {'MobileBertConfig': MobileBertConfig, 'MobileBertForPreTraining': MobileBertForPreTraining, 'load_tf_weights_in_mobilebert': load_tf_weights_in_mobilebert, 'torch': torch, 'tf_checkpoint_path': tf_checkpoint_path, 'mobilebert_config_file': mobilebert_config_file, 'pytorch_dump_path': pytorch_dump_path}, 0)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tf_checkpoint_path', default=None, type=str, required=True, help='Path to the TensorFlow checkpoint path.')
    parser.add_argument('--mobilebert_config_file', default=None, type=str, required=True, help='The config json file corresponding to the pre-trained MobileBERT model. \nThis specifies the model architecture.')
    parser.add_argument('--pytorch_dump_path', default=None, type=str, required=True, help='Path to the output PyTorch model.')
    args = parser.parse_args()
    convert_tf_checkpoint_to_pytorch(args.tf_checkpoint_path, args.mobilebert_config_file, args.pytorch_dump_path)

