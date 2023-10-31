""" Convert slow tokenizers checkpoints in fast (serialization format of the `tokenizers` library) """

import argparse
import os
import transformers
from transformers.convert_slow_tokenizer import SLOW_TO_FAST_CONVERTERS
from transformers.utils import logging
logging.set_verbosity_info()
logger = logging.get_logger(__name__)
TOKENIZER_CLASSES = {name: getattr(transformers, name + 'Fast') for name in SLOW_TO_FAST_CONVERTERS}

def convert_slow_checkpoint_to_fast(tokenizer_name, checkpoint_name, dump_path, force_download):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('transformers.convert_slow_tokenizers_checkpoints_to_fast.convert_slow_checkpoint_to_fast', 'convert_slow_checkpoint_to_fast(tokenizer_name, checkpoint_name, dump_path, force_download)', {'TOKENIZER_CLASSES': TOKENIZER_CLASSES, 'transformers': transformers, 'logger': logger, 'os': os, 'tokenizer_name': tokenizer_name, 'checkpoint_name': checkpoint_name, 'dump_path': dump_path, 'force_download': force_download}, 0)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dump_path', default=None, type=str, required=True, help='Path to output generated fast tokenizer files.')
    parser.add_argument('--tokenizer_name', default=None, type=str, help='Optional tokenizer type selected in the list of {}. If not given, will download and convert all the checkpoints from AWS.'.format(list(TOKENIZER_CLASSES.keys())))
    parser.add_argument('--checkpoint_name', default=None, type=str, help='Optional checkpoint name. If not given, will download and convert the canonical checkpoints from AWS.')
    parser.add_argument('--force_download', action='store_true', help='Re-dowload checkpoints.')
    args = parser.parse_args()
    convert_slow_checkpoint_to_fast(args.tokenizer_name, args.checkpoint_name, args.dump_path, args.force_download)

