from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import workspace, scope
from caffe2.python.model_helper import ModelHelper
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def tanh(x):
    return 2.0 * sigmoid(2.0 * x) - 1

def _prepare_rnn(t, n, dim_in, create_rnn, outputs_with_grads, forget_bias, memory_optim=False, forward_only=False, drop_states=False, T=None, two_d_initial_states=None, dim_out=None, num_states=2, **kwargs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.rnn.rnn_cell_test_util._prepare_rnn', '_prepare_rnn(t, n, dim_in, create_rnn, outputs_with_grads, forget_bias, memory_optim=False, forward_only=False, drop_states=False, T=None, two_d_initial_states=None, dim_out=None, num_states=2, **kwargs)', {'ModelHelper': ModelHelper, 'np': np, 'workspace': workspace, 'scope': scope, 't': t, 'n': n, 'dim_in': dim_in, 'create_rnn': create_rnn, 'outputs_with_grads': outputs_with_grads, 'forget_bias': forget_bias, 'memory_optim': memory_optim, 'forward_only': forward_only, 'drop_states': drop_states, 'T': T, 'two_d_initial_states': two_d_initial_states, 'dim_out': dim_out, 'num_states': num_states, 'kwargs': kwargs}, 1)

