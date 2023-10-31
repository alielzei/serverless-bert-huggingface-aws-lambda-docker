"""Convert Huggingface Pytorch checkpoint to Tensorflow checkpoint."""

import argparse
import os
import numpy as np
import tensorflow as tf
import torch
from transformers import BertModel

def convert_pytorch_checkpoint_to_tf(model: BertModel, ckpt_dir: str, model_name: str):
    """
    :param model:BertModel Pytorch model instance to be converted
    :param ckpt_dir: Tensorflow model directory
    :param model_name: model name
    :return:

    Currently supported HF models:
        Y BertModel
        N BertForMaskedLM
        N BertForPreTraining
        N BertForMultipleChoice
        N BertForNextSentencePrediction
        N BertForSequenceClassification
        N BertForQuestionAnswering
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.convert_bert_pytorch_checkpoint_to_original_tf.convert_pytorch_checkpoint_to_tf', 'convert_pytorch_checkpoint_to_tf(model, ckpt_dir, model_name)', {'os': os, 'np': np, 'tf': tf, 'model': model, 'ckpt_dir': ckpt_dir, 'model_name': model_name}, 1)

def main(raw_args=None):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('transformers.convert_bert_pytorch_checkpoint_to_original_tf.main', 'main(raw_args=None)', {'argparse': argparse, 'BertModel': BertModel, 'torch': torch, 'convert_pytorch_checkpoint_to_tf': convert_pytorch_checkpoint_to_tf, 'raw_args': raw_args}, 0)
if __name__ == '__main__':
    main()

