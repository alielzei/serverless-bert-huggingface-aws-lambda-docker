from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import networkx as nx
import collections
import time
import copy
from caffe2.python import workspace, core
from caffe2.proto import caffe2_pb2
import enum
import logging
from future.utils import viewitems, viewvalues
import caffe2.python._import_c_extension as C
log = logging.getLogger('memonger')
log.setLevel(logging.INFO)
LiveRange = collections.namedtuple('LiveRange', ['defined', 'used', 'size'])

def share_grad_blobs(net, losses, param_grads, namescope, dont_share_blobs=None, share_activations=False, blob_shapes=None):
    """
    Implements similar optimization as Torch's shareGradInput():
    for the gradients that are passed between layers, share blobs between
    operators when possible. This yields significant memory savings with
    deep networks.

    Returns an optimized protobuf (assign to net._net)
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.memonger.share_grad_blobs', 'share_grad_blobs(net, losses, param_grads, namescope, dont_share_blobs=None, share_activations=False, blob_shapes=None)', {'log': log, 'copy': copy, 'time': time, 'C': C, 'caffe2_pb2': caffe2_pb2, 'verify_graph_equality': verify_graph_equality, 'verify_inplace_blobs': verify_inplace_blobs, 'net': net, 'losses': losses, 'param_grads': param_grads, 'namescope': namescope, 'dont_share_blobs': dont_share_blobs, 'share_activations': share_activations, 'blob_shapes': blob_shapes}, 1)

def optimize_inference_for_dag(net, input_blobs, namescope=''):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.memonger.optimize_inference_for_dag', "optimize_inference_for_dag(net, input_blobs, namescope='')", {'copy': copy, 'time': time, 'C': C, 'log': log, 'caffe2_pb2': caffe2_pb2, 'verify_graph_equality': verify_graph_equality, 'verify_inplace_blobs': verify_inplace_blobs, 'net': net, 'input_blobs': input_blobs, 'namescope': namescope}, 1)

def estimate_memory_usage(protos, shapes, types, devicescope):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.memonger.estimate_memory_usage', 'estimate_memory_usage(protos, shapes, types, devicescope)', {'caffe2_pb2': caffe2_pb2, 'log': log, 'collections': collections, 'protos': protos, 'shapes': shapes, 'types': types, 'devicescope': devicescope}, 1)

def release_blobs_when_used(netproto, dont_free_blobs, selector_fun=None):
    """
    Insert Free-ops after a blob has been used the last time, so that its
    memory can be reclaimed. Use this only with efficient caching memory
    managers (such as CUB, --caffe2_cuda_memory_pool=cub).

    Blobs used with Alias op won't be freed.

    @dont_free_blobs:  is a set of blobs that should not be freed
    @selector_fun:     optional lambda that return True if blob name
                       can be released. Use for easy special filtering, like
                       excluding blobs with "loss" in the name.

    Returns a new protobuffer. To use with a model, use:
        model.net._net = memonger.release_blobs_when_used(..)
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.memonger.release_blobs_when_used', 'release_blobs_when_used(netproto, dont_free_blobs, selector_fun=None)', {'copy': copy, 'core': core, 'netproto': netproto, 'dont_free_blobs': dont_free_blobs, 'selector_fun': selector_fun}, 1)

def _find_source_nodes(g):
    """ Return nodes without predecessors """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.memonger._find_source_nodes', '_find_source_nodes(g)', {'g': g}, 1)

def _find_target_nodes(g):
    """ Return nodes without successors """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.memonger._find_target_nodes', '_find_target_nodes(g)', {'g': g}, 1)

def _add_single_target_ifneeded(g):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.memonger._add_single_target_ifneeded', '_add_single_target_ifneeded(g)', {'_find_target_nodes': _find_target_nodes, 'copy': copy, 'g': g}, 1)

def _get_path(pred_list, dist_list):
    """ Get the path from nx.bellman_ford()'s output """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.memonger._get_path', '_get_path(pred_list, dist_list)', {'pred_list': pred_list, 'dist_list': dist_list}, 1)

def _get_longest_paths(g, source_nodes):
    """ Get the longest path for nodes in 'source_nodes'
        Find with bellman_ford() by setting weight = -1
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.memonger._get_longest_paths', '_get_longest_paths(g, source_nodes)', {'copy': copy, 'nx': nx, '_get_path': _get_path, 'g': g, 'source_nodes': source_nodes}, 1)

def _build_tree(paths):
    """ Build a tree for given paths based on common elements.
        Last elements of all paths are the same, which is the root of the tree.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.memonger._build_tree', '_build_tree(paths)', {'nx': nx, '_compute_tree_height': _compute_tree_height, 'paths': paths}, 2)

def _compute_tree_height(g, root):
    """ Compute the heights of the tree for all nodes
        Height of leaves are 0
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.memonger._compute_tree_height', '_compute_tree_height(g, root)', {'g': g, 'root': root}, 1)

def _sort_tree_leaves(g, root):
    """ For each node, sort its child nodes based on the height of the nodes.
        Return the leaf nodes of the tree after sorting.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.memonger._sort_tree_leaves', '_sort_tree_leaves(g, root)', {'g': g, 'root': root}, 1)

def topological_sort_traversal_longest_path(g):
    """ The graph 'g' may contain several source nodes (nodes without incoming
        edge), which could be in any order and still be a valid
        topological sorting result. We would like to arrange these source nodes
        so that the average live spans of the computed blobs are shorter.
        The idea is to sort the source nodes based on the length of their path to
        the target node so that the one with longer path is used first.
        This is done by:
        - Add a single target node if there are multiple target nodes in 'g'.
        - Find the longest path between each source and the target node.
        - Convert the longest paths to a tree with the target node being the root
          and source nodes being the leaves.
        - Sort the nodes of the tree based on the height of the tree.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.memonger.topological_sort_traversal_longest_path', 'topological_sort_traversal_longest_path(g)', {'_add_single_target_ifneeded': _add_single_target_ifneeded, '_find_source_nodes': _find_source_nodes, '_get_longest_paths': _get_longest_paths, '_build_tree': _build_tree, 'viewvalues': viewvalues, '_sort_tree_leaves': _sort_tree_leaves, 'nx': nx, 'g': g}, 1)

def topological_sort_traversal(g):
    return list(nx.topological_sort(g))

def compute_ranges(linearized_ops, blob_sizes=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.memonger.compute_ranges', 'compute_ranges(linearized_ops, blob_sizes=None)', {'log': log, 'collections': collections, 'LiveRange': LiveRange, 'linearized_ops': linearized_ops, 'blob_sizes': blob_sizes}, 1)

def is_compatible(candidate_range, assignment, static_blobs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.memonger.is_compatible', 'is_compatible(candidate_range, assignment, static_blobs)', {'candidate_range': candidate_range, 'assignment': assignment, 'static_blobs': static_blobs}, 1)

def compute_blob_assignments(assignments):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.memonger.compute_blob_assignments', 'compute_blob_assignments(assignments)', {'assignments': assignments}, 1)

def _get_max_size(assignment):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.memonger._get_max_size', '_get_max_size(assignment)', {'assignment': assignment}, 1)

def get_memory_usage(assignments):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.memonger.get_memory_usage', 'get_memory_usage(assignments)', {'_get_max_size': _get_max_size, 'assignments': assignments}, 1)

def compute_assignments_greedy(ranges_sorted, init_assignments=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.memonger.compute_assignments_greedy', 'compute_assignments_greedy(ranges_sorted, init_assignments=None)', {'is_compatible': is_compatible, '_get_max_size': _get_max_size, 'ranges_sorted': ranges_sorted, 'init_assignments': init_assignments}, 1)

def _get_count(assignments):
    """ Return number of blobs in assignments """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.memonger._get_count', '_get_count(assignments)', {'assignments': assignments}, 1)

def compute_assignments_dp(ranges_sorted, init_assignment, counter=None):
    """ Compute assignment for blobs in 'ranges_sorted' on top of 'init_assignment'
        using dynamic programming + recursion.

        ranges_sorted: blobs sorted by 'used'
        init_assignment: assignment to start with, blobs in 'ranges_sorted' should
                         not be used in 'init_assignment'

        Using f(b, k, init) to represent the best assignment for blobs b[0:k]
        given initial assignment 'init', we have
            f(b, k, init) = f(b, j, init) +
                            find_best(b[j:k], f(b, j, init))
        where j is the index of the last best assignment that is independent of
        blob b[k - 1] (b[k - 1] is compatible with all assignments in
        f(b, j, init)), and find_best(b1, init1) gives the best assignment
        for blobs in 'b1' based on the initial assignment 'init1', and blobs
        b1[0:-1] should be incompatible with b1[-1]. f(b, len(b), []) gives
        the best assignment for blobs 'b'.

        For find_best(b, init), since b[0:-1] are not compatible with b[-1], we
        could reduce it to a smaller problem to find best assignment for b[0:-1]
        as
            find_best(b, init) = min {
                f(b[0:-1], len(b) - 1, init - x) + [x, b[-1]] for x in init, or
                f(b[0:-1], len(b) - 1, init) + [b[-1]]
            }
        where min{} gives the assignment with minimum memory usage.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.memonger.compute_assignments_dp', 'compute_assignments_dp(ranges_sorted, init_assignment, counter=None)', {'is_compatible': is_compatible, 'copy': copy, 'compute_assignments_dp': compute_assignments_dp, 'get_memory_usage': get_memory_usage, 'log': log, '_get_count': _get_count, 'ranges_sorted': ranges_sorted, 'init_assignment': init_assignment, 'counter': counter}, 1)

def get_updated_ranges(ranges, max_live=None):
    """ Set LiveRange.defined = -1 if it is None
        Set LiveRange.used = max_live if it is None
        Set LiveRanee.size = 1 if it is None
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.memonger.get_updated_ranges', 'get_updated_ranges(ranges, max_live=None)', {'ranges': ranges, 'max_live': max_live}, 1)

def compute_assignments(ranges, static_blobs, algo):
    """
    algo: Method used to find assignments (AssignmentAlgorithm.GREEDY or
          AssignmentAlgorithm.DYNAMIC_PROGRAMMING).
          AssignmentAlgorithm.DYNAMIC_PROGRAMMING gives optimal solution at the
          cost of more computation.
          AssignmentAlgorithm.GREEDY may be better in the case 'blob_sizes' is
          not provided.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.memonger.compute_assignments', 'compute_assignments(ranges, static_blobs, algo)', {'viewitems': viewitems, 'get_updated_ranges': get_updated_ranges, 'log': log, 'AssignmentAlgorithm': AssignmentAlgorithm, 'compute_assignments_dp': compute_assignments_dp, 'compute_assignments_greedy': compute_assignments_greedy, 'ranges': ranges, 'static_blobs': static_blobs, 'algo': algo}, 1)

def verify_assignments(assignments):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.memonger.verify_assignments', 'verify_assignments(assignments)', {'assignments': assignments}, 0)

def compute_interference_graph(ops):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.memonger.compute_interference_graph', 'compute_interference_graph(ops)', {'nx': nx, 'ops': ops}, 1)
Optimization = collections.namedtuple('Optimization', ['net', 'assignments', 'blob_assignments'])

def apply_assignments(net, blob_assignments):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.memonger.apply_assignments', 'apply_assignments(net, blob_assignments)', {'apply_recurrent_blob_assignments': apply_recurrent_blob_assignments, 'net': net, 'blob_assignments': blob_assignments}, 1)

def apply_recurrent_blob_assignments(op, blob_assignments, canonical_name):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.memonger.apply_recurrent_blob_assignments', 'apply_recurrent_blob_assignments(op, blob_assignments, canonical_name)', {'log': log, 'apply_assignments': apply_assignments, 'viewitems': viewitems, 'caffe2_pb2': caffe2_pb2, 'op': op, 'blob_assignments': blob_assignments, 'canonical_name': canonical_name}, 0)


class AssignmentAlgorithm(enum.Enum):
    GREEDY = 0
    DYNAMIC_PROGRAMMING = 1


def optimize_inference_fast(net, static_blobs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.memonger.optimize_inference_fast', 'optimize_inference_fast(net, static_blobs)', {'caffe2_pb2': caffe2_pb2, 'C': C, 'net': net, 'static_blobs': static_blobs}, 1)

def optimize_interference(net, static_blobs, ordering_function=topological_sort_traversal, blob_sizes=None, algo=AssignmentAlgorithm.GREEDY):
    """
    ordering_function: topological_sort_traversal or
                       topological_sort_traversal_longest_path.
                       topological_sort_traversal_longest_path gives better
                       results but needs a bit more computation.
    algo: Method used to find assignments (AssignmentAlgorithm.GREEDY or
          AssignmentAlgorithm.DYNAMIC_PROGRAMMING).
          AssignmentAlgorithm.DYNAMIC_PROGRAMMING gives optimal solution at the
          cost of more computation.
          AssignmentAlgorithm.GREEDY may be better in the case 'blob_sizes' is
          not provided.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.memonger.optimize_interference', 'optimize_interference(net, static_blobs, ordering_function=topological_sort_traversal, blob_sizes=None, algo=AssignmentAlgorithm.GREEDY)', {'copy': copy, 'compute_interference_graph': compute_interference_graph, 'compute_ranges': compute_ranges, 'compute_assignments': compute_assignments, 'compute_blob_assignments': compute_blob_assignments, 'apply_assignments': apply_assignments, 'Optimization': Optimization, 'net': net, 'static_blobs': static_blobs, 'ordering_function': ordering_function, 'blob_sizes': blob_sizes, 'algo': algo, 'topological_sort_traversal': topological_sort_traversal}, 1)

def verify_inplace_blobs(net_a, net_b):
    """
    Verifies that net_a and net_b have the same in-place blob assignments.
    Particularly, that memonger did not add an in-place assignment when that
    did not exist before.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.memonger.verify_inplace_blobs', 'verify_inplace_blobs(net_a, net_b)', {'net_a': net_a, 'net_b': net_b}, 1)

def verify_graph_equality(net_a, net_b):
    """
    Determines if the execution of two graphs are identical.
    That is, all inputs blobs are mapped to the same output blobs
    for each operator in their respective positions.

    This is meant to check the output of memonger with the original graph.
    It assumes that the nets have same external input and output.

    O(E) runtime + O(1) amortized cost to hash for python dict
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.memonger.verify_graph_equality', 'verify_graph_equality(net_a, net_b)', {'net_a': net_a, 'net_b': net_b}, 1)
Statistics = collections.namedtuple('Statistics', ['baseline_nbytes', 'optimized_nbytes'])

def blob_nbytes(blob):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.memonger.blob_nbytes', 'blob_nbytes(blob)', {'workspace': workspace, 'log': log, 'blob': blob}, 1)

def compute_statistics(assignments):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.memonger.compute_statistics', 'compute_statistics(assignments)', {'blob_nbytes': blob_nbytes, 'viewvalues': viewvalues, 'Statistics': Statistics, 'assignments': assignments}, 1)

def collect_blob_sizes(net):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.memonger.collect_blob_sizes', 'collect_blob_sizes(net)', {'blob_nbytes': blob_nbytes, 'net': net}, 1)

