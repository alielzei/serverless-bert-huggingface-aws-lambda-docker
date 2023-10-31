from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
'\nThe following methods are various utility methods for using the Tensor-Train\ndecomposition, or TT-decomposition introduced by I. V. Oseledets (2011) in his\npaper (http://epubs.siam.org/doi/abs/10.1137/090752286).\n\nBroadly speaking, these methods are used to replace fully connected layers in\nneural networks with Tensor-Train layers introduced by A. Novikov et. al. (2015)\nin their paper (http://arxiv.org/abs/1509.06569). More details about each of\nthe methods are provided in each respective docstring.\n'

def init_tt_cores(inp_sizes, out_sizes, tt_ranks, seed=1234):
    """
    Initialize randomized orthogonalized TT-cores.

    This method should be used when a TT-layer is trained from scratch. The
    sizes of each of the cores are specified by the inp_sizes and out_sizes, and
    the respective tt_ranks will dictate the ranks of each of the cores. Note
    that a larger set of tt_ranks will result in slower computation but will
    result in more accurate approximations. The size of the ith core is:

        tt_ranks[i] * inp_sizes[i] * out_sizes[i] * tt_ranks[i + 1].

    Note that the following relationships of lengths of each input is expected:

        len(inp_sizes) == len(out_sizes) == len(tt_ranks) - 1.

    Args:
        inp_sizes: list of the input dimensions of the respective cores
        out_sizes: list of the output dimensions of the respective cores
        tt_ranks: list of the ranks of the respective cores
        seed: integer to seed the random number generator

    Returns:
        cores: One-dimensional list of cores concatentated along an axis
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.tt_core.init_tt_cores', 'init_tt_cores(inp_sizes, out_sizes, tt_ranks, seed=1234)', {'np': np, 'inp_sizes': inp_sizes, 'out_sizes': out_sizes, 'tt_ranks': tt_ranks, 'seed': seed}, 1)

def matrix_to_tt(W, inp_sizes, out_sizes, tt_ranks):
    """
    Convert a matrix into the TT-format.

    This method will consume a 2D weight matrix such as those used in fully
    connected layers in a neural network and will compute the TT-decomposition
    of the weight matrix and return the TT-cores of the resulting computation.
    This method should be used when converting a trained, fully connected layer,
    into a TT-layer for increased speed and decreased parameter size. The size
    of the ith core is:

        tt_ranks[i] * inp_sizes[i] * out_sizes[i] * tt_ranks[i + 1].

    Note that the following relationships of lengths of each input is expected:

        len(inp_sizes) == len(out_sizes) == len(tt_ranks) - 1.

    We also require that np.prod(inp_sizes) == W.shape[0] and that
    np.prod(out_sizes) == W.shape[1].

    Args:
        W: two-dimensional weight matrix numpy array representing a fully
           connected layer to be converted to TT-format; note that the weight
           matrix is transposed before decomposed because we want to emulate the
           X * W^T operation that the FC layer performs.
        inp_sizes: list of the input dimensions of the respective cores
        out_sizes: list of the output dimensions of the respective cores
        tt_ranks: list of the ranks of the respective cores

    Returns:
        new_cores: One-dimensional list of cores concatentated along an axis
   """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.tt_core.matrix_to_tt', 'matrix_to_tt(W, inp_sizes, out_sizes, tt_ranks)', {'np': np, 'tt_svd': tt_svd, 'W': W, 'inp_sizes': inp_sizes, 'out_sizes': out_sizes, 'tt_ranks': tt_ranks}, 1)

def tt_svd(W, sizes, tt_ranks):
    """
    Helper method for the matrix_to_tt() method performing the TT-SVD
    decomposition.

    Uses the TT-decomposition algorithm to convert a matrix to TT-format using
    multiple reduced SVD operations.

    Args:
        W: two-dimensional weight matrix representing a fully connected layer to
           be converted to TT-format preprocessed by the matrix_to_tt() method.
        sizes: list of the dimensions of each of the cores
        tt_ranks: list of the ranks of the respective cores

    Returns:
        cores: One-dimensional list of cores concatentated along an axis
   """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.tt_core.tt_svd', 'tt_svd(W, sizes, tt_ranks)', {'np': np, 'W': W, 'sizes': sizes, 'tt_ranks': tt_ranks}, 1)

def fc_net_to_tt_net(net):
    pass

