"""
This script is used to generate test data for joblib/test/test_numpy_pickle.py
"""

import sys
import re
try:
    import numpy as np
except ImportError:
    np = None
import joblib

def get_joblib_version(joblib_version=joblib.__version__):
    """Normalize joblib version by removing suffix.

    >>> get_joblib_version('0.8.4')
    '0.8.4'
    >>> get_joblib_version('0.8.4b1')
    '0.8.4'
    >>> get_joblib_version('0.9.dev0')
    '0.9'
    """
    matches = [re.match('(\\d+).*', each) for each in joblib_version.split('.')]
    return '.'.join([m.group(1) for m in matches if m is not None])

def write_test_pickle(to_pickle, args):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.test.data.create_numpy_pickle.write_test_pickle', 'write_test_pickle(to_pickle, args)', {'get_joblib_version': get_joblib_version, 'sys': sys, 'np': np, 'joblib': joblib, 'to_pickle': to_pickle, 'args': args}, 0)
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Joblib pickle data generator.')
    parser.add_argument('--cache_size', action='store_true', help='Force creation of companion numpy files for pickled arrays.')
    parser.add_argument('--compress', action='store_true', help='Generate compress pickles.')
    parser.add_argument('--method', type=str, default='zlib', choices=['zlib', 'gzip', 'bz2', 'xz', 'lzma', 'lz4'], help='Set compression method.')
    to_pickle = [np.arange(5, dtype=np.dtype('<i8')), np.arange(5, dtype=np.dtype('<f8')), np.array([1, 'abc', {'a': 1, 'b': 2}], dtype='O'), np.arange(256, dtype=np.uint8).tobytes(), np.matrix([0, 1, 2], dtype=np.dtype('<i8')), "C'est l'été !"]
    write_test_pickle(to_pickle, parser.parse_args())

