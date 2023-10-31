from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace
import os
import tempfile
from zipfile import ZipFile
'\nGenerates a document in markdown format summrizing the coverage of serialized\ntesting. The document lives in\n`caffe2/python/serialized_test/SerializedTestCoverage.md`\n'
OpSchema = workspace.C.OpSchema

def gen_serialized_test_coverage(source_dir, output_dir):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.serialized_test.coverage.gen_serialized_test_coverage', 'gen_serialized_test_coverage(source_dir, output_dir)', {'gen_coverage_sets': gen_coverage_sets, 'os': os, 'source_dir': source_dir, 'output_dir': output_dir}, 0)

def gen_coverage_sets(source_dir):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.serialized_test.coverage.gen_coverage_sets', 'gen_coverage_sets(source_dir)', {'gen_covered_ops': gen_covered_ops, 'core': core, 'OpSchema': OpSchema, 'source_dir': source_dir}, 3)

def gen_covered_ops(source_dir):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.serialized_test.coverage.gen_covered_ops', 'gen_covered_ops(source_dir)', {'caffe2_pb2': caffe2_pb2, 'os': os, 'tempfile': tempfile, 'ZipFile': ZipFile, 'source_dir': source_dir}, 1)

