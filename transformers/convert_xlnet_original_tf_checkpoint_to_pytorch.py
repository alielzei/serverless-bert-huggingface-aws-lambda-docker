"""Convert BERT checkpoint."""

import argparse
import os
import torch
from transformers import CONFIG_NAME, WEIGHTS_NAME, XLNetConfig, XLNetForQuestionAnswering, XLNetForSequenceClassification, XLNetLMHeadModel, load_tf_weights_in_xlnet
from transformers.utils import logging
GLUE_TASKS_NUM_LABELS = {'cola': 2, 'mnli': 3, 'mrpc': 2, 'sst-2': 2, 'sts-b': 1, 'qqp': 2, 'qnli': 2, 'rte': 2, 'wnli': 2}
logging.set_verbosity_info()

def convert_xlnet_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_folder_path, finetuning_task=None):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('transformers.convert_xlnet_original_tf_checkpoint_to_pytorch.convert_xlnet_checkpoint_to_pytorch', 'convert_xlnet_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file, pytorch_dump_folder_path, finetuning_task=None)', {'XLNetConfig': XLNetConfig, 'GLUE_TASKS_NUM_LABELS': GLUE_TASKS_NUM_LABELS, 'XLNetForSequenceClassification': XLNetForSequenceClassification, 'XLNetForQuestionAnswering': XLNetForQuestionAnswering, 'XLNetLMHeadModel': XLNetLMHeadModel, 'load_tf_weights_in_xlnet': load_tf_weights_in_xlnet, 'os': os, 'WEIGHTS_NAME': WEIGHTS_NAME, 'CONFIG_NAME': CONFIG_NAME, 'torch': torch, 'tf_checkpoint_path': tf_checkpoint_path, 'bert_config_file': bert_config_file, 'pytorch_dump_folder_path': pytorch_dump_folder_path, 'finetuning_task': finetuning_task}, 0)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tf_checkpoint_path', default=None, type=str, required=True, help='Path to the TensorFlow checkpoint path.')
    parser.add_argument('--xlnet_config_file', default=None, type=str, required=True, help='The config json file corresponding to the pre-trained XLNet model. \nThis specifies the model architecture.')
    parser.add_argument('--pytorch_dump_folder_path', default=None, type=str, required=True, help='Path to the folder to store the PyTorch model or dataset/vocab.')
    parser.add_argument('--finetuning_task', default=None, type=str, help='Name of a task on which the XLNet TensorFloaw model was fine-tuned')
    args = parser.parse_args()
    print(args)
    convert_xlnet_checkpoint_to_pytorch(args.tf_checkpoint_path, args.xlnet_config_file, args.pytorch_dump_folder_path, args.finetuning_task)

