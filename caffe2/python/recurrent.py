from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import core, workspace
from future.utils import viewitems, viewkeys

def recurrent_net(net, cell_net, inputs, initial_cell_inputs, links, timestep=None, scope=None, outputs_with_grads=(0, ), recompute_blobs_on_backward=None, forward_only=False):
    """
    net: the main net operator should be added to

    cell_net: cell_net which is executed in a recurrent fasion

    inputs: sequences to be fed into the recurrent net. Currently only one input
    is supported. It has to be in a format T x N x (D1...Dk) where T is lengths
    of the sequence. N is a batch size and (D1...Dk) are the rest of dimentions

    initial_cell_inputs: inputs of the cell_net for the 0 timestamp.
    Format for each input is:
        (cell_net_input_name, external_blob_with_data)

    links: a dictionary from cell_net input names in moment t+1 and
    output names of moment t. Currently we assume that each output becomes
    an input for the next timestep.

    timestep: name of the timestep blob to be used. If not provided "timestep"
    is used.

    scope: Internal blobs are going to be scoped in a format
    <scope_name>/<blob_name>
    If not provided we generate a scope name automatically

    outputs_with_grads : position indices of output blobs which will receive
    error gradient (from outside recurrent network) during backpropagation

    recompute_blobs_on_backward: specify a list of blobs that will be
                 recomputed for backward pass, and thus need not to be
                 stored for each forward timestep.

    forward_only: if True, only forward steps are executed
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.recurrent.recurrent_net', 'recurrent_net(net, cell_net, inputs, initial_cell_inputs, links, timestep=None, scope=None, outputs_with_grads=(0, ), recompute_blobs_on_backward=None, forward_only=False)', {'core': core, 'viewitems': viewitems, 'viewkeys': viewkeys, 'net': net, 'cell_net': cell_net, 'inputs': inputs, 'initial_cell_inputs': initial_cell_inputs, 'links': links, 'timestep': timestep, 'scope': scope, 'outputs_with_grads': outputs_with_grads, 'recompute_blobs_on_backward': recompute_blobs_on_backward, 'forward_only': forward_only}, 1)

def set_rnn_executor_config(rnn_op, num_threads=None, max_cuda_streams=None):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.recurrent.set_rnn_executor_config', 'set_rnn_executor_config(rnn_op, num_threads=None, max_cuda_streams=None)', {'rnn_op': rnn_op, 'num_threads': num_threads, 'max_cuda_streams': max_cuda_streams}, 0)

def retrieve_step_blobs(net, prefix='rnn'):
    """
    Retrieves blobs from step workspaces (which contain intermediate recurrent
    network computation for each timestep) and puts them in the global
    workspace. This allows access to the contents of this intermediate
    computation in python. Returns the list of extracted blob names.

    net: the net from which the step workspace blobs should be extracted

    prefix: prefix to append to extracted blob names when placing them in the
    global workspace
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.recurrent.retrieve_step_blobs', "retrieve_step_blobs(net, prefix='rnn')", {'workspace': workspace, 'core': core, 'net': net, 'prefix': prefix}, 1)

