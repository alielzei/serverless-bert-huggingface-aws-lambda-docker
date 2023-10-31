import argparse
import os
import torch
from transformers.file_utils import WEIGHTS_NAME
DIALOGPT_MODELS = ['small', 'medium', 'large']
OLD_KEY = 'lm_head.decoder.weight'
NEW_KEY = 'lm_head.weight'

def convert_dialogpt_checkpoint(checkpoint_path: str, pytorch_dump_folder_path: str):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('transformers.convert_dialogpt_original_pytorch_checkpoint_to_pytorch.convert_dialogpt_checkpoint', 'convert_dialogpt_checkpoint(checkpoint_path, pytorch_dump_folder_path)', {'torch': torch, 'NEW_KEY': NEW_KEY, 'OLD_KEY': OLD_KEY, 'os': os, 'WEIGHTS_NAME': WEIGHTS_NAME, 'checkpoint_path': checkpoint_path, 'pytorch_dump_folder_path': pytorch_dump_folder_path}, 0)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dialogpt_path', default='.', type=str)
    args = parser.parse_args()
    for MODEL in DIALOGPT_MODELS:
        checkpoint_path = os.path.join(args.dialogpt_path, f'{MODEL}_ft.pkl')
        pytorch_dump_folder_path = f'./DialoGPT-{MODEL}'
        convert_dialogpt_checkpoint(checkpoint_path, pytorch_dump_folder_path)

