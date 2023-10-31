"""
Implement functions for controlling execution of nets and steps, including
  Do
  DoParallel
  For-loop
  While-loop
  Do-While-loop
  Switch
  If
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import core
from future.utils import viewitems
_current_idx = 1
_used_step_names = set()

def _get_next_step_name(control_name, base_name):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.control._get_next_step_name', '_get_next_step_name(control_name, base_name)', {'_used_step_names': _used_step_names, 'control_name': control_name, 'base_name': base_name}, 1)

def _MakeList(input):
    """ input is a tuple.
    Example:
    (a, b, c)   --> [a, b, c]
    (a)         --> [a]
    ([a, b, c]) --> [a, b, c]
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.control._MakeList', '_MakeList(input)', {'input': input}, 1)

def _IsNets(nets_or_steps):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.control._IsNets', '_IsNets(nets_or_steps)', {'core': core, 'nets_or_steps': nets_or_steps}, 1)

def _PrependNets(nets_or_steps, *nets):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.control._PrependNets', '_PrependNets(nets_or_steps, *nets)', {'_MakeList': _MakeList, '_IsNets': _IsNets, 'Do': Do, 'nets_or_steps': nets_or_steps, 'nets': nets}, 1)

def _AppendNets(nets_or_steps, *nets):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.control._AppendNets', '_AppendNets(nets_or_steps, *nets)', {'_MakeList': _MakeList, '_IsNets': _IsNets, 'Do': Do, 'nets_or_steps': nets_or_steps, 'nets': nets}, 1)

def GetConditionBlobFromNet(condition_net):
    """
    The condition blob is the last external_output that must
    be a single bool
    """
    assert len(condition_net.Proto().external_output) > 0, 'Condition net %s must has at least one external output' % condition_net.Proto.name
    return core.BlobReference(condition_net.Proto().external_output[-1])

def BoolNet(*blobs_with_bool_value):
    """A net assigning constant bool values to blobs. It is mainly used for
    initializing condition blobs, for example, in multi-task learning, we
    need to access reader_done blobs before reader_net run. In that case,
    the reader_done blobs must be initialized.

    Args:
    blobs_with_bool_value: one or more (blob, bool_value) pairs. The net will
    assign each bool_value to the corresponding blob.

    returns
    bool_net: A net assigning constant bool values to blobs.

    Examples:
    - BoolNet((blob_1, bool_value_1), ..., (blob_n, bool_value_n))
    - BoolNet([(blob_1, net1), ..., (blob_n, bool_value_n)])
    - BoolNet((cond_1, bool_value_1))
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.control.BoolNet', 'BoolNet(*blobs_with_bool_value)', {'_MakeList': _MakeList, 'core': core, 'blobs_with_bool_value': blobs_with_bool_value}, 1)

def NotNet(condition_blob_or_net):
    """Not of a condition blob or net

    Args:
    condition_blob_or_net can be either blob or net. If condition_blob_or_net
    is Net, the condition is its last external_output
    that must be a single bool.

    returns
    not_net: the net NOT the input
    out_blob: the output blob of the not_net
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.control.NotNet', 'NotNet(condition_blob_or_net)', {'core': core, 'GetConditionBlobFromNet': GetConditionBlobFromNet, 'condition_blob_or_net': condition_blob_or_net}, 2)

def _CopyConditionBlobNet(condition_blob):
    """Make a condition net that copies the condition_blob

    Args:
    condition_blob is a single bool.

    returns
    not_net: the net NOT the input
    out_blob: the output blob of the not_net
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.control._CopyConditionBlobNet', '_CopyConditionBlobNet(condition_blob)', {'core': core, 'condition_blob': condition_blob}, 2)

def MergeConditionNets(name, condition_nets, relation):
    """
    Merge multi condition nets into a single condition nets.

    Args:
        name: name of the new condition net.
        condition_nets: a list of condition nets. The last external_output
                        of each condition net must be single bool value.
        relation: can be 'And' or 'Or'.

    Returns:
        - A new condition net. Its last external output is relation of all
          condition_nets.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.control.MergeConditionNets', 'MergeConditionNets(name, condition_nets, relation)', {'core': core, 'GetConditionBlobFromNet': GetConditionBlobFromNet, 'viewitems': viewitems, 'name': name, 'condition_nets': condition_nets, 'relation': relation}, 1)

def CombineConditions(name, condition_nets, relation):
    """
    Combine conditions of multi nets into a single condition nets. Unlike
    MergeConditionNets, the actual body of condition_nets is not copied into
    the combine condition net.

    One example is about multi readers. Each reader net has a reader_done
    condition. When we want to check whether all readers are done, we can
    use this function to build a new net.

    Args:
        name: name of the new condition net.
        condition_nets: a list of condition nets. The last external_output
                        of each condition net must be single bool value.
        relation: can be 'And' or 'Or'.

    Returns:
        - A new condition net. Its last external output is relation of all
          condition_nets.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.control.CombineConditions', 'CombineConditions(name, condition_nets, relation)', {'GetConditionBlobFromNet': GetConditionBlobFromNet, '_CopyConditionBlobNet': _CopyConditionBlobNet, 'core': core, 'name': name, 'condition_nets': condition_nets, 'relation': relation}, 1)

def Do(name, *nets_or_steps):
    """
    Execute the sequence of nets or steps once.

    Examples:
    - Do('myDo', net1, net2, ..., net_n)
    - Do('myDo', list_of_nets)
    - Do('myDo', step1, step2, ..., step_n)
    - Do('myDo', list_of_steps)
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.control.Do', 'Do(name, *nets_or_steps)', {'_MakeList': _MakeList, 'core': core, '_get_next_step_name': _get_next_step_name, 'name': name, 'nets_or_steps': nets_or_steps}, 1)

def DoParallel(name, *nets_or_steps):
    """
    Execute the nets or steps in parallel, waiting for all of them to finish

    Examples:
    - DoParallel('pDo', net1, net2, ..., net_n)
    - DoParallel('pDo', list_of_nets)
    - DoParallel('pDo', step1, step2, ..., step_n)
    - DoParallel('pDo', list_of_steps)
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.control.DoParallel', 'DoParallel(name, *nets_or_steps)', {'_MakeList': _MakeList, 'core': core, '_get_next_step_name': _get_next_step_name, 'name': name, 'nets_or_steps': nets_or_steps}, 1)

def _RunOnceIf(name, condition_blob_or_net, nets_or_steps):
    """
    Execute nets_or_steps once if condition_blob_or_net evaluates as true.

    If condition_blob_or_net is Net, the condition is its last external_output
    that must be a single bool. And this net will be executed before
    nets_or_steps so as to get the condition.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.control._RunOnceIf', '_RunOnceIf(name, condition_blob_or_net, nets_or_steps)', {'NotNet': NotNet, 'core': core, '_PrependNets': _PrependNets, '_get_next_step_name': _get_next_step_name, '_IsNets': _IsNets, 'BoolNet': BoolNet, 'Do': Do, 'name': name, 'condition_blob_or_net': condition_blob_or_net, 'nets_or_steps': nets_or_steps}, 1)

def _RunOnceIfNot(name, condition_blob_or_net, nets_or_steps):
    """
    Similar to _RunOnceIf() but Execute nets_or_steps once if
    condition_blob_or_net evaluates as false.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.control._RunOnceIfNot', '_RunOnceIfNot(name, condition_blob_or_net, nets_or_steps)', {'core': core, 'GetConditionBlobFromNet': GetConditionBlobFromNet, '_PrependNets': _PrependNets, '_CopyConditionBlobNet': _CopyConditionBlobNet, '_get_next_step_name': _get_next_step_name, 'name': name, 'condition_blob_or_net': condition_blob_or_net, 'nets_or_steps': nets_or_steps}, 1)

def For(name, nets_or_steps, iter_num):
    """
    Execute nets_or_steps iter_num times.

    Args:
    nets_or_steps: a ExecutionStep or a Net or a list of ExecutionSteps or
                   a list nets.
    iter_num:    the number times to execute the nets_or_steps.

    Returns:
    A ExecutionStep instance.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.control.For', 'For(name, nets_or_steps, iter_num)', {'core': core, '_get_next_step_name': _get_next_step_name, '_PrependNets': _PrependNets, 'Do': Do, 'name': name, 'nets_or_steps': nets_or_steps, 'iter_num': iter_num}, 1)

def While(name, condition_blob_or_net, nets_or_steps):
    """
    Execute nets_or_steps when condition_blob_or_net returns true.

    Args:
    condition_blob_or_net: If it is an instance of Net, its last
      external_output must be a single bool.
    nets_or_steps: a ExecutionStep or a Net or a list of ExecutionSteps or
                   a list nets.

    Returns:
    A ExecutionStep instance.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.control.While', 'While(name, condition_blob_or_net, nets_or_steps)', {'NotNet': NotNet, 'core': core, '_PrependNets': _PrependNets, '_get_next_step_name': _get_next_step_name, '_IsNets': _IsNets, 'BoolNet': BoolNet, 'Do': Do, 'name': name, 'condition_blob_or_net': condition_blob_or_net, 'nets_or_steps': nets_or_steps}, 1)

def Until(name, condition_blob_or_net, nets_or_steps):
    """
    Similar to While() but execute nets_or_steps when
    condition_blob_or_net returns false
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.control.Until', 'Until(name, condition_blob_or_net, nets_or_steps)', {'core': core, 'GetConditionBlobFromNet': GetConditionBlobFromNet, '_PrependNets': _PrependNets, '_get_next_step_name': _get_next_step_name, 'name': name, 'condition_blob_or_net': condition_blob_or_net, 'nets_or_steps': nets_or_steps}, 1)

def DoWhile(name, condition_blob_or_net, nets_or_steps):
    """
    Execute nets_or_steps when condition_blob_or_net returns true. It will
    execute nets_or_steps before evaluating condition_blob_or_net.

    Args:
    condition_blob_or_net: if it is an instance of Net, tts last external_output
      must be a single bool.
    nets_or_steps: a ExecutionStep or a Net or a list of ExecutionSteps or
                   a list nets.

    Returns:
    A ExecutionStep instance.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.control.DoWhile', 'DoWhile(name, condition_blob_or_net, nets_or_steps)', {'NotNet': NotNet, 'core': core, '_AppendNets': _AppendNets, 'BoolNet': BoolNet, 'Do': Do, '_get_next_step_name': _get_next_step_name, 'name': name, 'condition_blob_or_net': condition_blob_or_net, 'nets_or_steps': nets_or_steps}, 1)

def DoUntil(name, condition_blob_or_net, nets_or_steps):
    """
    Similar to DoWhile() but execute nets_or_steps when
    condition_blob_or_net returns false. It will execute
    nets_or_steps before evaluating condition_blob_or_net.

    Special case: if condition_blob_or_net is a blob and is pre-set to
    true, then only the first net/step of nets_or_steps will be executed and
    loop is exited. So you need to be careful about the initial value the
    condition blob when using DoUntil(), esp when DoUntil() is called twice.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.control.DoUntil', 'DoUntil(name, condition_blob_or_net, nets_or_steps)', {'core': core, '_get_next_step_name': _get_next_step_name, '_AppendNets': _AppendNets, 'GetConditionBlobFromNet': GetConditionBlobFromNet, 'BoolNet': BoolNet, 'Do': Do, 'name': name, 'condition_blob_or_net': condition_blob_or_net, 'nets_or_steps': nets_or_steps}, 1)

def Switch(name, *conditions):
    """
    Execute the steps for which the condition is true.
    Each condition is a tuple (condition_blob_or_net, nets_or_steps).
    Note:
      1. Multi steps can be executed if their conditions are true.
      2. The conditions_blob_or_net (if it is Net) of all steps will be
         executed once.

    Examples:
    - Switch('name', (cond_1, net_1), (cond_2, net_2), ..., (cond_n, net_n))
    - Switch('name', [(cond_1, net1), (cond_2, net_2), ..., (cond_n, net_n)])
    - Switch('name', (cond_1, net_1))
    """
    conditions = _MakeList(conditions)
    return core.scoped_execution_step(_get_next_step_name('Switch', name), [_RunOnceIf(name + '/Switch', cond, step) for (cond, step) in conditions])

def SwitchNot(name, *conditions):
    """
    Similar to Switch() but execute the steps for which the condition is False.
    """
    conditions = _MakeList(conditions)
    return core.scoped_execution_step(_get_next_step_name('SwitchNot', name), [_RunOnceIfNot(name + '/SwitchNot', cond, step) for (cond, step) in conditions])

def If(name, condition_blob_or_net, true_nets_or_steps, false_nets_or_steps=None):
    """
    condition_blob_or_net is first evaluated or executed. If the condition is
    true, true_nets_or_steps is then executed, otherwise, false_nets_or_steps
    is executed.

    If condition_blob_or_net is Net, the condition is its last external_output
    that must be a single bool. And this Net will be executred before both
    true/false_nets_or_steps so as to get the condition.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.control.If', 'If(name, condition_blob_or_net, true_nets_or_steps, false_nets_or_steps=None)', {'_RunOnceIf': _RunOnceIf, 'core': core, 'GetConditionBlobFromNet': GetConditionBlobFromNet, 'Do': Do, '_RunOnceIfNot': _RunOnceIfNot, 'name': name, 'condition_blob_or_net': condition_blob_or_net, 'true_nets_or_steps': true_nets_or_steps, 'false_nets_or_steps': false_nets_or_steps}, 1)

def IfNot(name, condition_blob_or_net, true_nets_or_steps, false_nets_or_steps=None):
    """
    If condition_blob_or_net returns false, executes true_nets_or_steps,
    otherwise executes false_nets_or_steps
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.control.IfNot', 'IfNot(name, condition_blob_or_net, true_nets_or_steps, false_nets_or_steps=None)', {'_RunOnceIfNot': _RunOnceIfNot, 'core': core, 'GetConditionBlobFromNet': GetConditionBlobFromNet, 'Do': Do, '_RunOnceIf': _RunOnceIf, 'name': name, 'condition_blob_or_net': condition_blob_or_net, 'true_nets_or_steps': true_nets_or_steps, 'false_nets_or_steps': false_nets_or_steps}, 1)

