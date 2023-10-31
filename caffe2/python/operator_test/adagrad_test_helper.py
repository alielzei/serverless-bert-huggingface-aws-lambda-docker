from __future__ import absolute_import, division, print_function, unicode_literals
from functools import partial
import caffe2.python.hypothesis_test_util as hu
import numpy as np
from caffe2.python import core

def ref_adagrad(param_in, mom_in, grad, lr, epsilon, using_fp16=False, output_effective_lr=False, output_effective_lr_and_update=False, decay=1.0, row_wise=False):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.adagrad_test_helper.ref_adagrad', 'ref_adagrad(param_in, mom_in, grad, lr, epsilon, using_fp16=False, output_effective_lr=False, output_effective_lr_and_update=False, decay=1.0, row_wise=False)', {'np': np, 'param_in': param_in, 'mom_in': mom_in, 'grad': grad, 'lr': lr, 'epsilon': epsilon, 'using_fp16': using_fp16, 'output_effective_lr': output_effective_lr, 'output_effective_lr_and_update': output_effective_lr_and_update, 'decay': decay, 'row_wise': row_wise}, 4)

def adagrad_sparse_test_helper(parent_test, inputs, lr, epsilon, engine, ref_adagrad, gc, dc, row_wise=False):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.operator_test.adagrad_test_helper.adagrad_sparse_test_helper', 'adagrad_sparse_test_helper(parent_test, inputs, lr, epsilon, engine, ref_adagrad, gc, dc, row_wise=False)', {'np': np, 'core': core, 'partial': partial, 'hu': hu, 'parent_test': parent_test, 'inputs': inputs, 'lr': lr, 'epsilon': epsilon, 'engine': engine, 'ref_adagrad': ref_adagrad, 'gc': gc, 'dc': dc, 'row_wise': row_wise}, 2)

