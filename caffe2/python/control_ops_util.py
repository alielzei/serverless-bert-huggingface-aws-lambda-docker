from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import core

def get_external_blob_names(net, lexical_scope):
    """
    Returns a set of blobs a given net depends on and a set of
    output blobs that are written by the net
    Inputs:
        net - net to return input/output blobs for;
        lexical_scope - all external blob names visible to the net
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.control_ops_util.get_external_blob_names', 'get_external_blob_names(net, lexical_scope)', {'core': core, 'net': net, 'lexical_scope': lexical_scope}, 2)

def add_if_op(if_net, cond_blob, lexical_scope, then_net, else_net=None):
    """
    A helper function to add an If op to the net.
    Automatically determines whether blobs in the then/else subnets are external
    (from the outer workspace) or local (visible only inside subnet's workspace)
    based on lexical scope - set of all outer blob names visible to the 'If'
    operator. All the blobs in then/else subnets with names matching a name in lexical
    scope and all the blobs that are first used as the operators' inputs are
    considered outer blobs - these blobs must exist in the outer workspace,
    then/else subnets can read their values and new values written into these blobs
    will be visible outside of the 'If' operator. All other blobs are local - exist
    only within inner workspaces for then/else.
    Inputs:
        if_net - net to add an If op to;
        cond_blob - scalar bool blob reference, used as If condition;
        lexical_scope - a set of outer blob names visible to then/else branches;
        then_net/else_net - nets (core.Net) for then/else branches
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.control_ops_util.add_if_op', 'add_if_op(if_net, cond_blob, lexical_scope, then_net, else_net=None)', {'get_external_blob_names': get_external_blob_names, 'core': core, 'if_net': if_net, 'cond_blob': cond_blob, 'lexical_scope': lexical_scope, 'then_net': then_net, 'else_net': else_net}, 0)

def add_while_op(while_net, cond_blob, lexical_scope, loop_body_net, condition_body_net=None):
    """
    A helper function to add a While op to the net. Same rules for determining
    outer and inner blobs as for the 'If' operator apply for the 'While' operator
    loop and condition subnets. If specified, condition net is executed in a separate
    workspace before the first and after each iteration, the last operator must have
    a single scalar boolean output that is written into the condition blob.
    Inputs:
        while_net - net to add a While op to;
        cond_blob - scalar bool blob reference, used as a stop condition;
        lexical_scope - a set of outer blob names visible to the loop's body;
        loop_body_net - net to execute on each iteration;
        condition_body_net - net to compute condition value
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.control_ops_util.add_while_op', 'add_while_op(while_net, cond_blob, lexical_scope, loop_body_net, condition_body_net=None)', {'get_external_blob_names': get_external_blob_names, 'core': core, 'while_net': while_net, 'cond_blob': cond_blob, 'lexical_scope': lexical_scope, 'loop_body_net': loop_body_net, 'condition_body_net': condition_body_net}, 0)

