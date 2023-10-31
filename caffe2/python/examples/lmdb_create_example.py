from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import argparse
import numpy as np
import lmdb
from caffe2.proto import caffe2_pb2
from caffe2.python import workspace, model_helper
'\nSimple example to create an lmdb database of random image data and labels.\nThis can be used a skeleton to write your own data import.\n\nIt also runs a dummy-model with Caffe2 that reads the data and\nvalidates the checksum is same.\n'

def create_db(output_file):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.examples.lmdb_create_example.create_db', 'create_db(output_file)', {'lmdb': lmdb, 'np': np, 'caffe2_pb2': caffe2_pb2, 'output_file': output_file}, 1)

def read_db_with_caffe2(db_file, expected_checksum):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.examples.lmdb_create_example.read_db_with_caffe2', 'read_db_with_caffe2(db_file, expected_checksum)', {'model_helper': model_helper, 'workspace': workspace, 'np': np, 'db_file': db_file, 'expected_checksum': expected_checksum}, 0)

def main():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.examples.lmdb_create_example.main', 'main()', {'argparse': argparse, 'create_db': create_db, 'read_db_with_caffe2': read_db_with_caffe2}, 0)
if __name__ == '__main__':
    main()

