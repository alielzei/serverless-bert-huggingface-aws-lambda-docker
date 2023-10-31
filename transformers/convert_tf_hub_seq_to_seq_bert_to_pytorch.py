"""Convert Seq2Seq TF Hub checkpoint."""

import argparse
from transformers import BertConfig, BertGenerationConfig, BertGenerationDecoder, BertGenerationEncoder, load_tf_weights_in_bert_generation, logging
logging.set_verbosity_info()

def convert_tf_checkpoint_to_pytorch(tf_hub_path, pytorch_dump_path, is_encoder_named_decoder, vocab_size, is_encoder):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('transformers.convert_tf_hub_seq_to_seq_bert_to_pytorch.convert_tf_checkpoint_to_pytorch', 'convert_tf_checkpoint_to_pytorch(tf_hub_path, pytorch_dump_path, is_encoder_named_decoder, vocab_size, is_encoder)', {'BertConfig': BertConfig, 'BertGenerationConfig': BertGenerationConfig, 'BertGenerationEncoder': BertGenerationEncoder, 'BertGenerationDecoder': BertGenerationDecoder, 'load_tf_weights_in_bert_generation': load_tf_weights_in_bert_generation, 'tf_hub_path': tf_hub_path, 'pytorch_dump_path': pytorch_dump_path, 'is_encoder_named_decoder': is_encoder_named_decoder, 'vocab_size': vocab_size, 'is_encoder': is_encoder}, 0)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tf_hub_path', default=None, type=str, required=True, help='Path to the TensorFlow checkpoint path.')
    parser.add_argument('--pytorch_dump_path', default=None, type=str, required=True, help='Path to the output PyTorch model.')
    parser.add_argument('--is_encoder_named_decoder', action='store_true', help='If decoder has to be renamed to encoder in PyTorch model.')
    parser.add_argument('--is_encoder', action='store_true', help='If model is an encoder.')
    parser.add_argument('--vocab_size', default=50358, type=int, help='Vocab size of model')
    args = parser.parse_args()
    convert_tf_checkpoint_to_pytorch(args.tf_hub_path, args.pytorch_dump_path, args.is_encoder_named_decoder, args.vocab_size, is_encoder=args.is_encoder)

