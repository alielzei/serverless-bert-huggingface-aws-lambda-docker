from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
import datetime
import numpy as np
from caffe2.python import core, workspace
DTYPES = {'uint8': np.uint8, 'uint8_fused': np.uint8, 'float': np.float32, 'float16': np.float16}

def benchmark_sparse_lengths_sum(dtype_str, categorical_limit, embedding_size, average_len, batch_size, iterations, flush_cache):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.operator_test.sparse_lengths_sum_benchmark.benchmark_sparse_lengths_sum', 'benchmark_sparse_lengths_sum(dtype_str, categorical_limit, embedding_size, average_len, batch_size, iterations, flush_cache)', {'datetime': datetime, 'np': np, 'workspace': workspace, 'DTYPES': DTYPES, 'core': core, 'dtype_str': dtype_str, 'categorical_limit': categorical_limit, 'embedding_size': embedding_size, 'average_len': average_len, 'batch_size': batch_size, 'iterations': iterations, 'flush_cache': flush_cache}, 0)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='minimal benchmark for sparse lengths sum.')
    parser.add_argument('-d', '--dtype', choices=list(DTYPES.keys()), default='float', help='The data type for the input lookup table.')
    parser.add_argument('-e', '--embedding-size', type=int, default=6000000, help='Lookup table size.')
    parser.add_argument('--embedding-dim', type=int, default=128, help='Embedding dimension.')
    parser.add_argument('--average-len', type=int, default=27, help='Sparse feature average lengths, default is 27')
    parser.add_argument('--batch-size', type=int, default=100, help='The batch size.')
    parser.add_argument('-i', '--iteration', type=int, default=100000, help='The number of iterations.')
    parser.add_argument('--flush-cache', action='store_true', help='If true, flush cache')
    (args, extra_args) = parser.parse_known_args()
    core.GlobalInit(['python'] + extra_args)
    benchmark_sparse_lengths_sum(args.dtype, args.embedding_size, args.embedding_dim, args.average_len, args.batch_size, args.iteration, args.flush_cache)

