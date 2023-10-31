"""Convert RoBERTa checkpoint."""

import argparse
import pytorch_lightning as pl
import torch
from transformers.modeling_longformer import LongformerForQuestionAnswering, LongformerModel


class LightningModel(pl.LightningModule):
    
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.num_labels = 2
        self.qa_outputs = torch.nn.Linear(self.model.config.hidden_size, self.num_labels)
    
    def forward(self):
        pass


def convert_longformer_qa_checkpoint_to_pytorch(longformer_model: str, longformer_question_answering_ckpt_path: str, pytorch_dump_folder_path: str):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('transformers.convert_longformer_original_pytorch_lightning_to_pytorch.convert_longformer_qa_checkpoint_to_pytorch', 'convert_longformer_qa_checkpoint_to_pytorch(longformer_model, longformer_question_answering_ckpt_path, pytorch_dump_folder_path)', {'LongformerModel': LongformerModel, 'LightningModel': LightningModel, 'torch': torch, 'LongformerForQuestionAnswering': LongformerForQuestionAnswering, 'longformer_model': longformer_model, 'longformer_question_answering_ckpt_path': longformer_question_answering_ckpt_path, 'pytorch_dump_folder_path': pytorch_dump_folder_path}, 0)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--longformer_model', default=None, type=str, required=True, help='model identifier of longformer. Should be either `longformer-base-4096` or `longformer-large-4096`.')
    parser.add_argument('--longformer_question_answering_ckpt_path', default=None, type=str, required=True, help='Path the official PyTorch Lighning Checkpoint.')
    parser.add_argument('--pytorch_dump_folder_path', default=None, type=str, required=True, help='Path to the output PyTorch model.')
    args = parser.parse_args()
    convert_longformer_qa_checkpoint_to_pytorch(args.longformer_model, args.longformer_question_answering_ckpt_path, args.pytorch_dump_folder_path)

