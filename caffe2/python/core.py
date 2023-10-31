from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from collections import namedtuple, OrderedDict, defaultdict
from past.builtins import basestring
from future.utils import viewitems, viewkeys, viewvalues
from itertools import chain
from six import binary_type, string_types, text_type
from caffe2.proto import caffe2_pb2
from caffe2.python import scope, utils, workspace
from caffe2.python.control_ops_grad import gen_do_gradient, gen_if_gradient, gen_while_gradient, disambiguate_grad_if_op_output
import caffe2.python._import_c_extension as C
import copy
import pickle
import numpy as np
import sys
import traceback
import os
if (sys.platform == 'darwin' and 'leveldb' in C.registered_dbs()):
    print('If you are using homebrew leveldb on a Mac OS, you might see an error warning you that malloc_zone_unregister() failed. This is not a caffe2 issue but is due to the homebrew leveldb having an incompatible memory allocator. It does not affect usage.')
DeviceScope = scope.DeviceScope
NameScope = scope.NameScope


class DataType:
    pass


def _InitDataType():
    for (name, value) in caffe2_pb2.TensorProto.DataType.items():
        setattr(DataType, name, value)
_InitDataType()

def _GetRegisteredOperators():
    return set(workspace.RegisteredOperators())
_REGISTERED_OPERATORS = _GetRegisteredOperators()

def RefreshRegisteredOperators():
    global _REGISTERED_OPERATORS
    _REGISTERED_OPERATORS = _GetRegisteredOperators()
_GLOBAL_INIT_ARGS = []

def GlobalInit(args):
    _GLOBAL_INIT_ARGS.extend(args[1:])
    C.global_init(args)

def GetGlobalInitArgs():
    return _GLOBAL_INIT_ARGS[:]

def IsOperator(op_type):
    return IsOperatorWithEngine(op_type, engine='DEFAULT')

def IsOperatorWithEngine(op_type, engine):
    return C.op_registry_key(op_type, engine) in _REGISTERED_OPERATORS

def IsGPUDeviceType(device_type):
    return device_type in {caffe2_pb2.CUDA, caffe2_pb2.HIP}

def DeviceOption(device_type, device_id=0, random_seed=None, node_name=None, numa_node_id=None, extra_info=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.core.DeviceOption', 'DeviceOption(device_type, device_id=0, random_seed=None, node_name=None, numa_node_id=None, extra_info=None)', {'caffe2_pb2': caffe2_pb2, 'device_type': device_type, 'device_id': device_id, 'random_seed': random_seed, 'node_name': node_name, 'numa_node_id': numa_node_id, 'extra_info': extra_info}, 1)

def device_option_equal(opt1, opt2, ignore_node_name=True, ignore_random_seed=True):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.core.device_option_equal', 'device_option_equal(opt1, opt2, ignore_node_name=True, ignore_random_seed=True)', {'opt1': opt1, 'opt2': opt2, 'ignore_node_name': ignore_node_name, 'ignore_random_seed': ignore_random_seed}, 1)

def InferBlobDevices(net):
    """
    Compute mapping from parameters to devices by looking at the
    device option of the op that creates the blob has
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.core.InferBlobDevices', 'InferBlobDevices(net)', {'caffe2_pb2': caffe2_pb2, 'net': net}, 1)

def InferOpBlobDevicesAsDict(op):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.core.InferOpBlobDevicesAsDict', 'InferOpBlobDevicesAsDict(op)', {'InferOpBlobDevices': InferOpBlobDevices, 'op': op}, 2)

def InferOpBlobDevices(op):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.core.InferOpBlobDevices', 'InferOpBlobDevices(op)', {'C': C, 'caffe2_pb2': caffe2_pb2, 'op': op}, 2)

def InferOpDeviceAsBlobDevices(op):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.core.InferOpDeviceAsBlobDevices', 'InferOpDeviceAsBlobDevices(op)', {'caffe2_pb2': caffe2_pb2, 'op': op}, 2)
GradientSlice = namedtuple('GradientSlice', ['indices', 'values'])


class BlobReference(object):
    """A wrapper around a blob in a net.

    BlobReference gives us a way to refer to the network that the blob is
    generated from. Note that blobs are, essentially, just strings in the
    current workspace.
    """
    
    def __init__(self, name, net=None):
        """Initializes a blob reference.

        Note that this does not prepends the namescope. If needed, use
        ScopedBlobReference() to prepend the existing namespace.
        """
        if isinstance(name, string_types):
            self._name = name
        elif isinstance(name, binary_type):
            self._name = name.decode('utf-8')
        else:
            self._name = str(name)
        self._from_net = net
        self.meta = {}
    
    def __hash__(self):
        return hash(self._name)
    
    def __eq__(self, other):
        if isinstance(other, string_types):
            return self._name == other
        elif isinstance(other, binary_type):
            return self._name == other.decode('utf-8')
        elif isinstance(other, BlobReference):
            return self._name == other._name
        else:
            return False
    
    def __ne__(self, other):
        return not self == other
    
    def __str__(self):
        return self._name
    
    def __repr__(self):
        return 'BlobReference("{}")'.format(self._name)
    
    def __add__(self, other):
        if not isinstance(other, string_types):
            raise RuntimeError('Cannot add BlobReference to a non-string.')
        return BlobReference(self._name + other, self._from_net)
    
    def __radd__(self, other):
        if not isinstance(other, string_types):
            raise RuntimeError('Cannot add a non-string to BlobReference.')
        return BlobReference(other + self._name, self._from_net)
    
    def Net(self):
        return self._from_net
    
    def GetNameScope(self):
        return self._name[:self._name.rfind(scope._NAMESCOPE_SEPARATOR) + 1]
    
    def GetUnscopedName(self):
        return self._name[self._name.rfind(scope._NAMESCOPE_SEPARATOR) + 1:]
    
    def _CreateAndAddToNet(self, op_type, inputs=None, *args, **kwargs):
        """Internal function that routes the operator generation to the
        network's __getattr__ function.
        """
        inputs = ([] if inputs is None else inputs)
        if (isinstance(inputs, BlobReference) or isinstance(inputs, string_types)):
            inputs = [inputs]
        inputs.insert(0, self)
        return self._from_net.__getattr__(op_type)(inputs, *args, **kwargs)
    
    def __getattr__(self, op_type):
        """A wrapper allowing one to initiate operators from a blob reference.

        Example: for a blob reference b that comes from network n, doing
            b.Relu(...)
        is equivalent to doing
            net.Relu([b], ...)
        """
        if op_type.startswith('__'):
            raise AttributeError('Attribute {} not found.'.format(op_type))
        if self._from_net is None:
            raise AttributeError('You cannot use a blob reference that does not have a net source to create operators. Create the operator from an explicit net object.')
        if not IsOperator(op_type):
            raise AttributeError('Method ' + op_type + ' is not a registered operator.' + ' Did you mean: [' + ','.join(workspace.C.nearby_opnames(op_type)) + ']')
        return lambda *args, **kwargs: self._CreateAndAddToNet(op_type, *args, **kwargs)
    
    def __dir__(self):
        additional_methods = [op for op in _REGISTERED_OPERATORS if ('_ENGINE_' not in op or '_ENGINE_CUDNN' in op)]
        return sorted(set(chain(dir(type(self)), viewkeys(self.__dict__), additional_methods)))


def ScopedName(name):
    """prefix the name with the current scope."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.core.ScopedName', 'ScopedName(name)', {'binary_type': binary_type, 'scope': scope, 'name': name}, 1)

def ScopedBlobReference(name, *args, **kwargs):
    """Returns a blob reference with scope prefixed."""
    return BlobReference(ScopedName(name), *args, **kwargs)

def _RectifyInputOutput(blobs, net=None):
    """A helper function to rectify the input or output of the CreateOperator
    interface.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.core._RectifyInputOutput', '_RectifyInputOutput(blobs, net=None)', {'string_types': string_types, 'binary_type': binary_type, 'ScopedBlobReference': ScopedBlobReference, 'BlobReference': BlobReference, 'blobs': blobs, 'net': net}, 1)

def CreateOperator(operator_type, inputs, outputs, name='', control_input=None, device_option=None, arg=None, engine=None, debug_info=None, **kwargs):
    """A function wrapper that allows one to create operators based on the
    operator type. The type should be a string corresponding to an operator
    registered with Caffe2.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.core.CreateOperator', "CreateOperator(operator_type, inputs, outputs, name='', control_input=None, device_option=None, arg=None, engine=None, debug_info=None, **kwargs)", {'caffe2_pb2': caffe2_pb2, 'os': os, 'traceback': traceback, '_RectifyInputOutput': _RectifyInputOutput, 'text_type': text_type, 'scope': scope, 'viewitems': viewitems, 'utils': utils, 'workspace': workspace, 'operator_type': operator_type, 'inputs': inputs, 'outputs': outputs, 'name': name, 'control_input': control_input, 'device_option': device_option, 'arg': arg, 'engine': engine, 'debug_info': debug_info, 'kwargs': kwargs}, 1)

def _RegisterPythonImpl(f, grad_f=None, python_func_type=None, pass_workspace=False):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.core._RegisterPythonImpl', '_RegisterPythonImpl(f, grad_f=None, python_func_type=None, pass_workspace=False)', {'C': C, 'f': f, 'grad_f': grad_f, 'python_func_type': python_func_type, 'pass_workspace': pass_workspace}, 1)

def CreatePythonOperator(f, inputs, outputs, grad_f=None, pass_workspace=False, python_func_type=None, *args, **kwargs):
    """
    `f` should have a signature (inputs, outputs)

    If `pass_workspace` is True, the signature is changed to
    (inputs, outputs, workspace) where `workspace` is the workspace the op
    is going to run on. This is potentially dangerous (as the op can manipulate
    the workspace directly), use on your own risk.
    """
    kwargs['token'] = _RegisterPythonImpl(f, grad_f, python_func_type, pass_workspace=pass_workspace)
    return CreateOperator('Python', inputs, outputs, *args, **kwargs)

def GetIndexFromGradientList(g_list, name):
    """A helper function to get the index from a gradient list, None if not
    matching."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.core.GetIndexFromGradientList', 'GetIndexFromGradientList(g_list, name)', {'GradientSlice': GradientSlice, 'g_list': g_list, 'name': name}, 1)
OpSSA = namedtuple('OpSSA', ['op', 'in_versions', 'out_versions'])
GradGenMeta = namedtuple('GradGenMeta', ['grad_op', 'idx', 'gradient', 'device_option'])
SparseGradGenMeta = namedtuple('SparseGradGenMeta', ['grad_op_indices', 'idx_indices', 'grad_op_values', 'idx_values', 'gradient', 'device_option'])


class IR(object):
    """A simple IR class to keep track of all intermediate representations used
    in the gradient computation.
    """
    
    def __init__(self, operators):
        self.ssa = []
        self.input_usages = defaultdict(lambda: defaultdict(list))
        self.frontier = defaultdict(int)
        self.gradient_frontier = {}
        self.gradient_generators = defaultdict(lambda: defaultdict(list))
        self.out_version_history = defaultdict(list)
        self.in_version_history = defaultdict(list)
        for op in operators:
            self.Play(op)
        self.SanityCheck(operators)
    
    def SanityCheck(self, operators):
        for op in operators:
            if op.type == 'StopGradient':
                if op.output[0] not in self.input_usages:
                    raise ValueError("StopGradient's output '{}' is orphan.\nYou typically want to specify same input and output for\nStopGradient. Op:\n\n{}".format(op.output[0], str(op)))
    
    def Play(self, op):
        """"Adds an op to the current IR, and update the internal states to
        reflect the blobs and versions after the execution of the op.
        """
        in_versions = {}
        for s in op.input:
            in_versions[s] = self.frontier[s]
            self.input_usages[s][self.frontier[s]].append(len(self.ssa))
            self.in_version_history[s].append((op, self.frontier[s]))
        out_versions = {}
        for s in op.output:
            if s in self.frontier:
                self.frontier[s] += 1
            out_versions[s] = self.frontier[s]
            self.out_version_history[s].append((op, self.frontier[s]))
        self.ssa.append(OpSSA(op, in_versions, out_versions))
    
    def CheckGradientOperatorInput(self, grad_op_input, g_output, fwd_op_idx, locally_generated_blobs):
        """Checks if the gradient operators can be correctly carried out."""
        (forward_op, in_versions, out_versions) = self.ssa[fwd_op_idx]
        original_index = GetIndexFromGradientList(g_output, grad_op_input)
        
        def versionMismatchInfoOut(name):
            s = 'DEBUG HELP:\n'
            s += 'Maybe you use same output blob twice for different ops?\n'
            s += '== Version history of blob [{}]\n'.format(name)
            for (op, vers) in self.out_version_history[name]:
                s += 'Version (out) {} <-- {}'.format(vers, op)
                s += '\n'
            return s
        
        def versionMismatchInfoIn(name):
            s = 'DEBUG HELP:\n'
            s += 'Maybe the blob was overwritten by another op?\n'
            s += '== Version history of blob [{}]\n'.format(name)
            for (op, vers) in self.in_version_history[name]:
                s += 'version (in) {} <-- {}'.format(vers, op)
                s += '\n'
            return s
        if original_index is not None:
            original_name = forward_op.output[original_index]
            if out_versions[original_name] != self.gradient_frontier[original_name]:
                raise RuntimeError('Gradient name "%s" is expected to correspond to version %d of "%s", but currently we have version %d.\n\n' % (grad_op_input, out_versions[original_name], original_name, self.gradient_frontier[original_name]) + versionMismatchInfoOut(original_name))
        elif grad_op_input in out_versions:
            if self.frontier[grad_op_input] != out_versions[grad_op_input]:
                raise RuntimeError('Gradient operator needs output "%s" at version %d, but currently we have version %d.\n\n' % (grad_op_input, out_versions[grad_op_input], self.frontier[grad_op_input]) + versionMismatchInfoOut(grad_op_input))
        elif grad_op_input in in_versions:
            if self.frontier[grad_op_input] != in_versions[grad_op_input]:
                raise RuntimeError('Gradient operator needs input "%s" at version %d, but currently we have version %d.\n\n' % (grad_op_input, in_versions[grad_op_input], self.frontier[grad_op_input]) + versionMismatchInfoIn(grad_op_input))
        elif grad_op_input not in locally_generated_blobs:
            raise RuntimeError('Blob name "%s" not in the scope of operator: %s\nand is not generated by any of the local gradient operators.' % (grad_op_input, str(forward_op)))
    
    def AppendSparseGenerators(self, sparse_generators):
        for (name, input_generators) in viewitems(sparse_generators):
            for (version, generators) in viewitems(input_generators):
                if len(generators) == 1:
                    generator = generators[0]
                else:
                    assert len(generators) == 2
                    (op1_i, idx1_i, op1_v, idx1_v, g1, dev_1) = generators[0]
                    (op2_i, idx2_i, op2_v, idx2_v, g2, dev_2) = generators[1]
                    assert g1 == g2
                    assert dev_1 == dev_2, 'Unequal devices for sparse generators: {} and {}'.format(dev1, dev2)
                    assert (op1_i is None or op2_i is None)
                    assert (op1_v is None or op2_v is None)
                    assert (idx1_i == 0 or idx2_i == 0)
                    assert (idx1_v == 0 or idx2_v == 0)
                    generator = SparseGradGenMeta((op1_i or op2_i), idx1_i + idx2_i, (op1_v or op2_v), idx1_v + idx2_v, g1, dev_1)
                self.gradient_generators[name][version].append(generator)
    
    def BuildGradientGenerators(self, fwd_op_idx, gradient_ops, g_output, g_input):
        """Updates gradient_generators and gradient_frontier"""
        (forward_op, in_versions, out_versions) = self.ssa[fwd_op_idx]
        locally_generated_blobs = []
        sparse_generators = defaultdict(lambda: defaultdict(list))
        for grad_op in gradient_ops:
            for s in grad_op.input:
                self.CheckGradientOperatorInput(s, g_output, fwd_op_idx, locally_generated_blobs)
            locally_generated_blobs.extend([str(s) for s in grad_op.output])
            for (i, output) in enumerate(grad_op.output):
                input_index = GetIndexFromGradientList(g_input, output)
                if input_index is not None:
                    input_name = forward_op.input[input_index]
                    input_version = in_versions[input_name]
                    g = g_input[input_index]
                    if type(g) is GradientSlice:
                        if g.indices == output:
                            m = SparseGradGenMeta(grad_op, i, None, 0, g, grad_op.device_option)
                        else:
                            assert g.values == output
                            m = SparseGradGenMeta(None, 0, grad_op, i, g, grad_op.device_option)
                        sparse_generators[input_name][input_version].append(m)
                    else:
                        self.gradient_generators[input_name][input_version].append(GradGenMeta(grad_op, i, g, grad_op.device_option))
        self.AppendSparseGenerators(sparse_generators)
        for (input_index, g) in enumerate(g_input):
            input_name = forward_op.input[input_index]
            input_version = in_versions[input_name]
            if not g:
                continue
            if type(g) is GradientSlice:
                if (str(g.indices) not in locally_generated_blobs and str(g.values) not in locally_generated_blobs):
                    self.gradient_generators[input_name][input_version].append(SparseGradGenMeta(None, 0, None, 0, g, forward_op.device_option))
            elif str(g) not in locally_generated_blobs:
                self.gradient_generators[input_name][input_version].append(GradGenMeta(None, 0, g, forward_op.device_option))
        for (i, g) in enumerate(g_input):
            if g is not None:
                input_name = forward_op.input[i]
                input_version = in_versions[input_name]
                self.gradient_frontier[input_name] = input_version
    
    def _GetSumOpOutputName(self, generator, input_name):
        
        def remove_suffix(s, suffix):
            if s.endswith(suffix):
                return s[:-len(suffix)]
            return s
        for g in generator:
            if type(g) is GradGenMeta:
                (grad_op, idx, _, _) = g
                if grad_op:
                    return grad_op.output[idx]
            else:
                assert type(g) is SparseGradGenMeta
                (op_i, idx_i, op_v, idx_v, _, _) = g
                if op_i:
                    return remove_suffix(op_i.output[idx_i], '_indices')
                if op_v:
                    return remove_suffix(op_v.output[idx_v], '_values')
        return input_name + '_grad'
    IS_AUTO_GEN_SUM_OPS_TAG = 'is_auto_gen_sum_ops'
    
    def _SetSumOpsDeviceOption(self, sum_ops, generators):
        for generator in generators:
            for op in sum_ops:
                op.device_option.CopyFrom(generator.device_option)
                op.device_option.extra_info.extend(['{}:1'.format(IR.IS_AUTO_GEN_SUM_OPS_TAG)])
            break
    
    def _DisambiguateGradOpOutput(self, grad_op, idx, cnt):
        new_grad_output = '_' + grad_op.output[idx] + '_autosplit_{}'.format(cnt)
        if grad_op.type == 'If':
            disambiguate_grad_if_op_output(grad_op, idx, new_grad_output)
        else:
            grad_op.output[idx] = new_grad_output
        return (grad_op.output[idx], cnt + 1)
    
    def _CheckSumOpsConflict(self, out_base_name, g):
        if str(out_base_name) == str(g):
            raise RuntimeError('The gradient output of empty gradient op can not be the same as the normal name of the current input gradient.')
    
    def _MakeDenseSumOps(self, generators, out_base_name):
        sum_op_input = []
        cnt = 0
        assert len(generators) > 1
        first_grad_op = True
        for generator in generators:
            (grad_op, idx, g, _) = generator
            assert type(g) is not GradientSlice
            if grad_op:
                if first_grad_op:
                    first_grad_op = False
                    out = grad_op.output[idx]
                else:
                    (out, cnt) = self._DisambiguateGradOpOutput(grad_op, idx, cnt)
                sum_op_input.append(out)
            else:
                self._CheckSumOpsConflict(out_base_name, g)
                sum_op_input.append(str(g))
        if out_base_name in sum_op_input:
            idx = sum_op_input.index(out_base_name)
            (sum_op_input[0], sum_op_input[idx]) = (sum_op_input[idx], sum_op_input[0])
        sum_ops = [CreateOperator('Sum', [BlobReference(x) for x in sum_op_input], BlobReference(out_base_name))]
        return (sum_ops, out_base_name)
    
    def _MakeSparseSumOps(self, generators, out_base_name):
        indices_concat_input = []
        values_concat_input = []
        cnt_i = 0
        cnt_v = 0
        for generator in generators:
            assert type(generator) is SparseGradGenMeta
            (op_i, idx_i, op_v, idx_v, g, _) = generator
            if op_i:
                (out, cnt_i) = self._DisambiguateGradOpOutput(op_i, idx_i, cnt_i)
                indices_concat_input.append(out)
            else:
                self._CheckSumOpsConflict(out_base_name, g.indices)
                indices_concat_input.append(g.indices)
            if op_v:
                (out, cnt_v) = self._DisambiguateGradOpOutput(op_v, idx_v, cnt_v)
                values_concat_input.append(out)
            else:
                self._CheckSumOpsConflict(out_base_name, g.values)
                values_concat_input.append(g.values)
        indices_concat_output = out_base_name + '_indices_concat'
        indices_concat_split = out_base_name + '_indices_concat_split'
        values_concat_output = out_base_name + '_values_concat'
        values_concat_split = out_base_name + '_values_concat_split'
        sum_ops = [CreateOperator('Concat', [BlobReference(x) for x in indices_concat_input], [BlobReference(x) for x in [indices_concat_output, indices_concat_split]], axis=0), CreateOperator('Concat', [BlobReference(x) for x in values_concat_input], [BlobReference(x) for x in [values_concat_output, values_concat_split]], axis=0)]
        sum_op_output = GradientSlice(indices=indices_concat_output, values=values_concat_output)
        return (sum_ops, sum_op_output)
    
    def _MakeSumOps(self, input_name, input_version):
        generators = self.gradient_generators[input_name][input_version]
        out_base_name = self._GetSumOpOutputName(generators, input_name)
        types = list(set((type(x) for x in generators)))
        assert len(types) == 1
        if types[0] is GradGenMeta:
            (sum_ops, g) = self._MakeDenseSumOps(generators, out_base_name)
        else:
            assert types[0] is SparseGradGenMeta
            (sum_ops, g) = self._MakeSparseSumOps(generators, out_base_name)
        self._SetSumOpsDeviceOption(sum_ops, generators)
        return (sum_ops, g)
    
    def _VerifyGradientGenerators(self, generator):
        if len({type(g) for g in generator}) > 1:
            raise RuntimeError('Automatic aggregation of a mix of sparse and dense gradients is not supported yet')
        if len(generator) < 2:
            return False
        all_gradient_names = []
        all_device_options = []
        for g in generator:
            if g.device_option:
                all_device_options.append(g.device_option)
            if type(g) is GradGenMeta:
                if g.grad_op:
                    all_gradient_names.append(g.gradient)
            else:
                assert type(g) is SparseGradGenMeta
                if g.gradient.values:
                    all_gradient_names.append(g.gradient.values)
        if (len(all_device_options) >= 2 and not all((device_option_equal(d, all_device_options[0]) for d in all_device_options[1:]))):
            raise RuntimeError('Unexpected behavior: not all grad ops have the same device option.')
        return True
    
    def DoGradientAccumulation(self, fwd_op_idx):
        """For each input name in the forward op, check if we will need to
        add gradient accumulation. If so, do gradient accumulation and return
        the list of gradient operators.

        The criteria for doing gradient accumulation is:
        (1) the specific input version has been used by multiple operators.
        (2) the current fwd_op_idx is the first to use that input, i.e. in the
            backward pass, is the last to optionally generate the gradient for
            the op.
        (3) For the operators that used the input, their gradient operators
            have generated more than 1 gradient.

        When accumulating operators, our current solution is to rename all the
        created gradients with an internal intermediate name, and then add a
        Sum() operator that adds up all the gradients. This may use more memory
        due to intermediate storage, but is usually the fastest approach as one
        can do one single sum for multiple intermediate gradients.
        """
        (forward_op, in_versions, out_versions) = self.ssa[fwd_op_idx]
        additional_sum_ops = []
        grad_map = {}
        for (_i, input_name) in enumerate(set(forward_op.input)):
            input_version = in_versions[input_name]
            input_usage = self.input_usages[input_name][input_version]
            if (len(input_usage) <= 1 or fwd_op_idx != input_usage[0]):
                continue
            generator = self.gradient_generators[input_name][input_version]
            try:
                if not self._VerifyGradientGenerators(generator):
                    continue
            except RuntimeError as err:
                raise RuntimeError("Gradients for param ''{}'' failed to verify: {}".format(input_name, err))
            (sum_ops, g) = self._MakeSumOps(input_name, input_version)
            additional_sum_ops.extend(sum_ops)
            grad_map[input_name] = g
        return (additional_sum_ops, grad_map)
    
    def _AppendAutoGradGenerator(self, y, grad, autograd_op):
        generator = GradGenMeta(autograd_op, (0 if autograd_op else None), str(grad), autograd_op.device_option)
        self.gradient_generators[str(y)][self.frontier[str(y)]].append(generator)
    AUTOGEN_GRAD_SUFFIX = '_autogen_grad'
    
    def _GetInitGradients(self, ys):
        input_to_grad = {}
        gradient_ops = []
        for (y, g) in viewitems(ys):
            autograd_op = None
            if g is None:
                autograd_op = CreateOperator('ConstantFill', [y], [str(y) + IR.AUTOGEN_GRAD_SUFFIX], value=1.0)
                gradient_ops.append(autograd_op)
                g = autograd_op.output[0]
            input_to_grad[str(y)] = (GradientSlice(str(g[0]), str(g[1])) if isinstance(g, GradientSlice) else str(g))
            if autograd_op is not None:
                self._AppendAutoGradGenerator(y, g, autograd_op)
        return (input_to_grad, gradient_ops)
    
    def _GenerateGradientsForForwardOp(self, forward_op_idx, input_to_grad):
        new_input_to_grad = {}
        gradient_ops = []
        (forward_op, in_versions, out_versions) = self.ssa[forward_op_idx]
        g_output = list((input_to_grad.get(name, None) for name in forward_op.output))
        if (not all((g is None for g in g_output)) or forward_op.type == 'ZeroGradient'):
            (gradient_ops, g_input) = GradientRegistry.GetGradientForOp(forward_op, g_output)
            self.BuildGradientGenerators(forward_op_idx, gradient_ops, g_output, g_input)
            for (name, grad) in zip(forward_op.input, g_input):
                if (grad is not None or name not in input_to_grad or name in list(forward_op.output)):
                    new_input_to_grad[name] = grad
        return (new_input_to_grad, gradient_ops)
    
    def GetBackwardPass(self, ys):
        """Gets the backward pass that computes the derivatives of given blobs.

        Inputs:
          ys: a list or a dictionary specifying what blobs we want to compute
              derivatives of. If the input is a list, we will automatically
              generate their gradients with all-one values; if the input is a
              dictionary, for any dictionary entries that are not None, we will
              take the corresponding blobs as their gradients; for all those
              that are None, we will auto-fill them with 1.
        """
        if isinstance(ys, list):
            ys = dict(((y, None) for y in ys))
        elif not isinstance(ys, dict):
            raise TypeError('ys should either be a list or a dict.')
        for y in viewkeys(ys):
            self.gradient_frontier[y] = self.frontier[y]
            self.input_usages[str(y)][self.frontier[str(y)]].append(len(self.ssa))
        (all_input_to_grad, all_gradient_ops) = self._GetInitGradients(ys)
        for forward_op_idx in reversed(range(len(self.ssa))):
            (input_to_grad, gradient_ops) = self._GenerateGradientsForForwardOp(forward_op_idx, all_input_to_grad)
            all_input_to_grad.update(input_to_grad)
            all_gradient_ops += gradient_ops
            (additional_sum_ops, grad_map) = self.DoGradientAccumulation(forward_op_idx)
            all_input_to_grad.update(grad_map)
            all_gradient_ops += additional_sum_ops
        all_input_to_grad_out = {}
        for (key, val) in viewitems(all_input_to_grad):
            if val is not None:
                if (isinstance(val, string_types) or isinstance(val, binary_type)):
                    grad_out = BlobReference(val)
                else:
                    grad_out = GradientSlice(BlobReference(val[0]), BlobReference(val[1]))
                all_input_to_grad_out[BlobReference(key)] = grad_out
        return (all_gradient_ops, all_input_to_grad_out)



class GradientRegistry(object):
    """GradientRegistry holds the mapping from operators to their gradients."""
    gradient_registry_ = {}
    
    @classmethod
    def RegisterGradient(cls, op_type):
        """A decorator for registering gradient mappings."""
        
        def Wrapper(func):
            cls.gradient_registry_[op_type] = func
            return func
        return Wrapper
    
    @classmethod
    def _GetGradientForOpCC(cls, op_def, g_output):
        
        def from_untyped(grad):
            if grad is None:
                w = C.GradientWrapper()
                assert w.is_empty()
                return w
            try:
                (indices, values) = grad
                w = C.GradientWrapper()
                w.indices = indices
                w.values = values
                assert w.is_sparse()
                return w
            except ValueError:
                w = C.GradientWrapper()
                w.dense = grad
                assert w.is_dense()
                return w
        g_output = [from_untyped(grad) for grad in g_output]
        (grad_defs_str, g_input) = C.get_gradient_defs(op_def.SerializeToString(), g_output)
        
        def to_untyped(grad_wrapper):
            if grad_wrapper.is_empty():
                return None
            if grad_wrapper.is_sparse():
                return GradientSlice(grad_wrapper.indices, grad_wrapper.values)
            assert grad_wrapper.is_dense()
            return grad_wrapper.dense
        g_input = [to_untyped(grad_wrapper) for grad_wrapper in g_input]
        grad_defs = []
        for grad_def_str in grad_defs_str:
            grad_def = caffe2_pb2.OperatorDef()
            grad_def.ParseFromString(grad_def_str)
            grad_defs.append(grad_def)
        return (grad_defs, g_input)
    
    @classmethod
    def GetGradientForOp(cls, op, g_output):
        try:
            (gradient_ops, g_input) = cls._GetGradientForOpCC(op, g_output)
        except Exception as e:
            if op.type in cls.gradient_registry_:
                (gradient_ops, g_input) = cls.gradient_registry_[op.type](op, g_output)
            else:
                raise Exception('Exception when creating gradient for [{}]:{}.\nOp: \n{}'.format(op.type, e, str(op)))
        if gradient_ops is None:
            return ([], g_input)
        if type(gradient_ops) is not list:
            gradient_ops = [gradient_ops]
        return (gradient_ops, g_input)
    
    @classmethod
    def GetBackwardPass(cls, operators, ys, ys_generate_gradient=False):
        """Gets the backward pass for the list of operators.

        Args:
            operators: a list of operators constituting the forward pass.
            ys: a list or a dictionary specifying what blobs we want to compute
                derivatives of. If the input is a list, we will automatically
                generate their gradients with all-one values; if the input is a
                dictionary, for any dictionary entries that are not None, we'll
                take the corresponding blobs as their gradients; for all those
                that are None, we will auto-fill them with 1.
        Returns:
            gradient_ops: a list of gradient operators to run.
            all_input_to_grads: a map from input to their corresponding
                gradients.
        """
        ir = IR(operators)
        return ir.GetBackwardPass(ys)

GradientRegistry.RegisterGradient('Do')(gen_do_gradient)
GradientRegistry.RegisterGradient('If')(gen_if_gradient)
GradientRegistry.RegisterGradient('While')(gen_while_gradient)

def get_ssa(net, blob_versions=None):
    """
    Given a net, return a structure containing the version of each input and
    output blob used by each operator.

    Args:
        net:            either a Net or a NetDef
        blob_versions:  (optional) map with current version number for given
                        blob names. If not provided or blob not found, start
                        from version 0.
    Returns:
        Tuple (ssa, blob_versions)
        ssa:            list of tuples (versioned_inputs, versioned_outputs)
                        for each op in the net. A versioned input is a tuple
                        (blob_name, version).
        blob_versions:  updated map with latest version of each blob found in
                        the net.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.core.get_ssa', 'get_ssa(net, blob_versions=None)', {'Net': Net, 'caffe2_pb2': caffe2_pb2, 'get_ssa': get_ssa, 'net': net, 'blob_versions': blob_versions}, 2)

def get_undefined_blobs(ssa):
    """
    Given a ssa in the format produced by get_ssa(), return a set of blobs that
    are used before they are defined, which corresponds to inputs at version 0.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.core.get_undefined_blobs', 'get_undefined_blobs(ssa)', {'ssa': ssa}, 1)

def get_output_producers(ssa):
    """
    Given a ssa in the format produced by get_ssa(), returns a map from
    versioned blob into the operator index that produces that version of
    the blob. A versioned blob is a tuple (blob_name, version).
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.core.get_output_producers', 'get_output_producers(ssa)', {'ssa': ssa}, 1)

def get_op_ids_in_path(ssa, blob_versions, inputs, outputs):
    """
    Given a ssa and blob_versions as produced by get_ssa(), returns the list
    of op indices that are necessary in order to generate the blobs in
    `outputs`, given blobs in `inputs`.
    Consider that the `inputs` are given in their latest version.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.core.get_op_ids_in_path', 'get_op_ids_in_path(ssa, blob_versions, inputs, outputs)', {'get_output_producers': get_output_producers, 'ssa': ssa, 'blob_versions': blob_versions, 'inputs': inputs, 'outputs': outputs}, 1)

def recurrent_network_op_remap(op, prefix, blob_remap):
    """
    Parameters
    ----------
    op : Caffe2 operator (RecurrentNetworkOp or RecurrentNetworkGradientOp).
    prefix: this argument is not used in this function, just for legacy support.
    blob_remap : Dictionary that represents the map from old blob name to new.

    Updates blob names in arguments of RecurrentNetworkOp and
    RecurrentNetworkGradientOp to conform to cloned input and output of both
    operators and also makes sure names of locally generated blobs in arguments
    have the same prefix as the input and output of the operators.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.core.recurrent_network_op_remap', 'recurrent_network_op_remap(op, prefix, blob_remap)', {'binary_type': binary_type, 'remap_proto': remap_proto, 'op': op, 'prefix': prefix, 'blob_remap': blob_remap}, 1)

def control_op_remap(op, prefix, blob_remap):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.core.control_op_remap', 'control_op_remap(op, prefix, blob_remap)', {'Net': Net, 'op': op, 'prefix': prefix, 'blob_remap': blob_remap}, 0)
DEFAULT_REMAP_FUNCS = {'RecurrentNetwork': recurrent_network_op_remap, 'RecurrentNetworkGradient': recurrent_network_op_remap, 'If': control_op_remap, 'While': control_op_remap}

def remap_proto(argument, blob_remap):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.core.remap_proto', 'remap_proto(argument, blob_remap)', {'Net': Net, 'argument': argument, 'blob_remap': blob_remap}, 0)

def clone_and_bind_net(net, name, prefix, blob_remap=None, inputs=None, keep_schema=True):
    """
    Clone the given Net, binding its input schema to the given `inputs` record.
    Blob names defined by the net are prepended with the given `prefix`.

    Args:
        net:        the net to clone
        name:       the name of the new net
        prefix:     the prefix to append to local blobs
        blob_remap: (optional) dict with additional blob name remapping.
        inputs:     (optional) input record that will provide actual input
                    values for the cloned net. Must be compatible with the
                    net's input schema or be a strict superset of it
        keep_schema: by default (True), the original schema will be kept and
                     remapped accordingly. otherwise, the schema will be set as
                     inputs or left empty if inputs is not given.
    Returns:
        Tuple (cloned_net, blob_remap)
        clone_net:  the cloned Net
        blob_remap: a map from original blob names into remapped blob names
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.core.clone_and_bind_net', 'clone_and_bind_net(net, name, prefix, blob_remap=None, inputs=None, keep_schema=True)', {'Net': Net, 'get_ssa': get_ssa, 'get_undefined_blobs': get_undefined_blobs, 'viewkeys': viewkeys, 'net': net, 'name': name, 'prefix': prefix, 'blob_remap': blob_remap, 'inputs': inputs, 'keep_schema': keep_schema}, 2)

def _get_blob_ref(blob_name_or_ref):
    return (blob_name_or_ref if isinstance(input, BlobReference) else BlobReference(blob_name_or_ref))

def _recover_record_by_prefix(names, prefix=''):
    """
    Tries to recover record by taking a subset of blob names with
    a given prefix name and interpreting them as schema column names
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.core._recover_record_by_prefix', "_recover_record_by_prefix(names, prefix='')", {'_get_blob_ref': _get_blob_ref, 'names': names, 'prefix': prefix}, 1)


class Net(object):
    _net_names_used = set()
    operator_registry_ = {}
    
    @staticmethod
    def current_prefix():
        from caffe2.python.net_builder import NetBuilder
        builder = NetBuilder.current(required=False)
        return (builder.name if builder else '')
    
    @staticmethod
    def _get_next_net_name(basename):
        name = basename = '/'.join((x for x in [Net.current_prefix(), basename] if x))
        next_idx = 1
        while name in Net._net_names_used:
            name = basename + '_' + str(next_idx)
            next_idx += 1
        Net._net_names_used |= set([name])
        return name
    
    def __init__(self, name_or_proto):
        """
        Create a Net.
        Args:
            name_or_proto:  If a NetDef is provided, clone it. Otherwise,
                            create an empty net with the given name.
        """
        self._input_record = None
        self._output_record = None
        self._registered_blob_names = set()
        self._recreate_lookup_tables = False
        self._op_outputs = set()
        self._external_input_map = set()
        self._attr_dict = defaultdict(list)
        if type(name_or_proto) is caffe2_pb2.NetDef:
            proto = name_or_proto
            self._net = caffe2_pb2.NetDef()
            self._net.CopyFrom(proto)
            existing_outputs = [list(op.output) for op in self._net.op]
            self._external_input_map.update(list(self._net.external_input))
            existing_names = set(sum([list(op.input) for op in self._net.op], []) + sum(existing_outputs, []))
            for outs in existing_outputs:
                self._op_outputs.update(outs)
            prefix_len = len(self._net.name + '_blob_')
            autogen_indices = []
            for s in existing_names:
                if s.startswith(self._net.name + '_blob_'):
                    try:
                        autogen_indices.append(int(s[prefix_len]))
                    except ValueError:
                        pass
            if len(autogen_indices):
                self._next_name_index = max(autogen_indices) + 1
            else:
                self._next_name_index = 0
            name = self._net.name
        else:
            name = name_or_proto
            self._net = caffe2_pb2.NetDef()
            self._next_name_index = 0
        self._net.name = Net._get_next_net_name(name)
    
    def AppendNet(self, net, device_option=None):
        assert isinstance(net, Net)
        for i in net.Proto().external_input:
            if (i not in self.Proto().external_input and i not in self._op_outputs):
                self.Proto().external_input.append(i)
        self.Proto().external_output.extend([o for o in net.Proto().external_output if o not in self.Proto().external_output])
        ops = net.Proto().op
        if device_option is not None:
            ops = [copy.deepcopy(op) for op in ops]
            map(lambda x: x.device_option.CopyFrom(device_option), ops)
            for op in ops:
                if op.type == 'RecurrentNetwork':
                    for arg in op.arg:
                        if arg.name.endswith('step_net'):
                            for step_op in arg.n.op:
                                step_op.device_option.CopyFrom(device_option)
        self._ExtendOps(ops)
        return self
    
    def LogInfo(self, *msg_or_blobs):
        for msg_or_blob in msg_or_blobs:
            if not isinstance(msg_or_blob, BlobReference):
                blob = self.GivenTensorStringFill([], self.NextName('log'), shape=[], values=[msg_or_blob])
            else:
                blob = msg_or_blob
            self.Print(blob, [])
    
    def add_attribute(self, name, obj):
        """
        Add `obj` to the list of attributes in this net under the given `name`.
        Attributes are user-defined objects and have no pre-defined semantics.
        """
        self._attr_dict[name].append(obj)
    
    def get_attributes(self, name):
        """
        Returns the list of attributes in this net for a given `name`.
        Attributes are user-defined objects added with `add_attribute'.
        """
        return self._attr_dict.get(name, [])
    
    def set_rand_seed(self, seed=100, sequence_seed=True, seed_on_op_def=False):
        """
        Adds a random seed to each op in the net.
        If sequence_seed is set, the i-th op has rand_seed=`seed + i`
        If seed_on_op_def is set, the op rand_seed=hash(str(op))
        sequence_seed and seed_on_op_def cannot be both set to True.
        """
        assert not ((sequence_seed and seed_on_op_def)), 'sequence_seed and seed_on_op_def cannot be both set to True.'
        for (i, op) in enumerate(self.Proto().op):
            if sequence_seed:
                curr_seed = seed + i
            elif seed_on_op_def:
                curr_seed = hash(str(op) + str(seed)) % np.iinfo(np.uint32).max
            else:
                curr_seed = seed
            op.device_option.random_seed = curr_seed
    
    def Name(self):
        return self._net.name
    
    def __str__(self):
        return self.Name()
    
    def Const(self, array, blob_out=None, dtype=None):
        if isinstance(array, bool):
            return self.ConstantFill([], (blob_out or 1), dtype=DataType.BOOL, value=array)
        if dtype is None:
            array = np.array(array)
        else:
            array = np.array(array, dtype=dtype)
        
        def do_set(operator):
            return operator([], (blob_out or 1), shape=array.shape, values=array.flatten().tolist())
        if array.dtype == np.int32:
            return do_set(self.GivenTensorIntFill)
        elif array.dtype == np.int64:
            return do_set(self.GivenTensorInt64Fill)
        elif array.dtype == np.str:
            return do_set(self.GivenTensorStringFill)
        elif array.dtype == np.bool:
            return do_set(self.GivenTensorBoolFill)
        else:
            return do_set(self.GivenTensorFill)
    
    def BlobIsDefined(self, blob):
        """
        Returns true if the given BlobReference is produced as output of
        an operator in this net, or if it is provided as an external input.
        """
        if self._recreate_lookup_tables:
            self._RecreateLookupTables()
        name = str(blob)
        return (name in self._op_outputs or name in self._external_input_map)
    
    def UsesBlob(self, blob):
        """
        Returns true iff the given BlobReference is used by any operator
        or this net, or if it is one of the external inputs of the net.
        """
        blob_name = str(blob)
        for op in self._net.op:
            for input in op.input:
                if input == blob_name:
                    return True
        return blob_name in self._external_input_map
    
    def UsedBlobNames(self):
        """
        Returns a set of blob names used in the net
        """
        blob_names = set()
        for op in self._net.op:
            blob_names |= set(op.input)
            blob_names |= set(op.output)
        if self._net.external_input:
            blob_names |= set(self._net.external_input)
        if self._net.external_output:
            blob_names |= set(self._net.external_output)
        return blob_names
    
    def GetBlobRef(self, blob_name):
        """
        Given the name of a blob produced by this net, return a BlobReference
        to it. If the blob is not produced by any op in this net,
        raises KeyError.
        """
        blob_name = str(blob_name)
        if not self.BlobIsDefined(blob_name):
            raise KeyError('Net does not define blob %s' % blob_name)
        return BlobReference(blob_name, self)
    
    def Clone(self, name, blob_remap=None, op_id_mask=None, remap_funcs=None, keep_schema=True, update_external_list=False):
        """
        Clone this net.
        Args:
            name:        name of the cloned net
            blob_remap:  optional map with list of blob names to replace
            op_id_mask:  optional list of operator indices to include in
                         the cloned net. If not provided, all ops are included.
        """
        orig_remap_funcs = ({} if remap_funcs is None else remap_funcs)
        remap_funcs = DEFAULT_REMAP_FUNCS.copy()
        remap_funcs.update(orig_remap_funcs)
        proto = self._net
        new_proto = caffe2_pb2.NetDef()
        new_proto.CopyFrom(proto)
        new_proto.name = name
        if blob_remap is None:
            blob_remap = {}
        if op_id_mask is None:
            op_id_mask = list(range(0, len(proto.op)))
        
        def get_remapped_str(blob):
            blob_str = str(blob)
            return str(blob_remap.get(blob_str, blob_str))
        
        def remap_list(proto_list):
            new_list = [get_remapped_str(b) for b in proto_list]
            del proto_list[:]
            proto_list.extend(new_list)
        
        def remap_op(op):
            new_op = caffe2_pb2.OperatorDef()
            new_op.CopyFrom(op)
            remap_list(new_op.input)
            remap_list(new_op.output)
            if new_op.type in remap_funcs:
                remap_funcs[new_op.type](new_op, (name + '/' if name else ''), blob_remap)
            return new_op
        del new_proto.op[:]
        new_proto.op.extend([remap_op(proto.op[op_id]) for op_id in op_id_mask])
        remap_list(new_proto.external_input)
        remap_list(new_proto.external_output)
        new_net = Net(new_proto)
        if keep_schema:
            from caffe2.python import schema
            if self._input_record:
                new_net._input_record = schema.from_blob_list(self._input_record, [BlobReference(get_remapped_str(blob), net=new_net) for blob in self._input_record.field_blobs()])
            if self._output_record:
                new_net._output_record = schema.from_blob_list(self._output_record, [BlobReference(get_remapped_str(blob), net=new_net) for blob in self._output_record.field_blobs()])
        new_net._attr_dict.update(self._attr_dict)
        if update_external_list:
            existing_outputs = set()
            used_outputs = set()
            del new_net.Proto().external_input[:]
            del new_net.Proto().external_output[:]
            for op in new_net.Proto().op:
                for ib in op.input:
                    if ib not in existing_outputs:
                        new_net.Proto().external_input.extend([ib])
                    else:
                        used_outputs.add(ib)
                for ob in op.output:
                    existing_outputs.add(ob)
            for ob in existing_outputs:
                if ob not in used_outputs:
                    new_net.Proto().external_output.extend([ob])
        return new_net
    
    def ClonePartial(self, name, inputs, outputs, remap_funcs=None):
        """
        Clone this net, including only ops that are necessary in order to
        compute `outputs` given `inputs`. Return references to the cloned
        outputs. Internal blobs (blobs that are produced and consumed inside
        the net but not used as outputs) will be remapped to avoid name
        conflict.

        Args:
            name:    the name of the cloned net
            inputs:  map where the keys correspond to BlobReferences in the
                     original net, and the values correspond to external inputs
                     in the partially cloned net. If `inputs` is a list, don't
                     remap input names.
            outputs: outputs to be produced by the cloned net.

        Returns:
            Tuple (new_net, new_outputs)
                new_net:       a new Net object.
                new_outputs:   list of BlobReferences corresponding to the
                               outputs produced by new_net.
        """
        input_is_pair_list = (isinstance(inputs, list) and all(((isinstance(i, tuple) and len(i) == 2) for i in inputs)))
        inputs = (inputs if isinstance(inputs, (dict, OrderedDict)) else (OrderedDict(inputs) if input_is_pair_list else OrderedDict(zip(inputs, inputs))))
        for output in outputs:
            assert self.BlobIsDefined(output), '{} is not defined'.format(output)
        input_names = {str(k): str(v) for (k, v) in viewitems(inputs)}
        output_names = [str(o) for o in outputs]
        proto = self._net
        blob_versions = {str(i): 0 for i in inputs}
        (ssa, blob_versions) = get_ssa(proto, blob_versions)
        used_op_ids = get_op_ids_in_path(ssa, blob_versions, inputs, outputs)
        disallowed_op_ids = get_op_ids_in_path(ssa, blob_versions, [], inputs)
        assert len(set(used_op_ids) & set(disallowed_op_ids)) == 0, 'Cannot partially clone net: some of the ops required would ' + 'generate the given input.'
        sub_ssa = [op for (i, op) in enumerate(ssa) if i in used_op_ids]
        undef_blobs = get_undefined_blobs(sub_ssa) - set(viewkeys(input_names))
        prefix = (name + '/' if name else '')
        
        def remap(blob_name):
            if blob_name in input_names:
                return input_names[blob_name]
            elif blob_name in undef_blobs:
                return blob_name
            else:
                return prefix + blob_name
        blob_mapping = {b: remap(b) for b in viewkeys(blob_versions)}
        new_net = self.Clone(name, blob_mapping, used_op_ids, remap_funcs)
        new_in = [blob_mapping[i] for i in viewkeys(input_names)] + list(undef_blobs)
        new_out = [blob_mapping[o] for o in output_names]
        del new_net.Proto().external_input[:]
        new_net.Proto().external_input.extend(new_in)
        new_net._external_input_map = set(list(new_in))
        del new_net.Proto().external_output[:]
        new_net.Proto().external_output.extend(new_out)
        return (new_net, [new_net.GetBlobRef(o) for o in new_out])
    
    def Proto(self):
        self._InvalidateLookupTables()
        return self._net
    
    def insert_op_at_idx(self, op, op_idx):
        """ inserting operator at index. Will update external blob list.
        """
        assert op_idx >= 0
        temp_ops = self.Proto().op[op_idx:]
        del self.Proto().op[op_idx:]
        self.Proto().op.extend([op])
        self.Proto().op.extend(temp_ops)
        self.external_outputs.extend(op.output)
        self.external_inputs.extend(op.input)
    
    def reroute_tensor(self, tensor, new_producer, can_modify=None):
        """ reroute tensor to new_producer. And feed new tensor to consumers
        and interseciton with can_modify if provided.
        Inputs:
            tensor: str or blob_reference the tensor to reroute
            new_producer: an op takes in tensor gives new_tesnor
            can_modify: a list/set of operators that consumes tensor and can be
            modified

        Returns:
            reroute_cnt: how many consumer op has been changed

        Note: assume no inplace blob in net
        """
        
        def _find_tensor_input_op(tensor):
            if tensor in self.external_inputs:
                op_idx = -1
            else:
                assert tensor in new_producer.input, 'new producer {} is not taking in {}'.format(new_producer.type, tensor)
                op_idx = -2
                for (index, op) in enumerate(self.Proto().op):
                    if_found = False
                    for o in op.output:
                        if o == tensor:
                            if_found = True
                            op_idx = index
                            break
                    if if_found:
                        break
            return op_idx
        op_idx = max((_find_tensor_input_op(t) for t in new_producer.input))
        self.insert_op_at_idx(new_producer, op_idx + 1)
        new_tensor = new_producer.output[0]
        if tensor in self.external_outputs:
            new_list = [(new_tensor if b == tensor else b) for b in self.external_outputs]
            del self.Proto().external_output[:]
            self.Proto().external_output.extend(new_list)
        reroute_cnt = 0
        if can_modify:
            for op in self.Proto().op:
                if op in can_modify:
                    remap_input(op, {tensor: new_tensor})
                    reroute_cnt = reroute_cnt + 1
        return reroute_cnt
    
    def PopulateProtoWithFileName(self):
        net_tb = workspace.operator_tracebacks.get(self.Name(), None)
        if net_tb is not None:
            for (idx, op) in enumerate(self.Proto().op):
                if idx in net_tb:
                    op.name = ':'.join(map(str, net_tb[idx][0]))
    
    def NextScopedBlob(self, prefix='unnamed'):
        """Return the blob that has not been defined or registered in the
        current net. It returns `ScopedBlobReference(prefix)`, if it's valid,
        otherwise `ScopedBlobReference(prefix) + '_auto_' + ?`. Different calls
        is guaranteed to return blob with different names.
        """
        output_blob_base = ScopedName(prefix)
        return self.NextBlob(output_blob_base)
    
    def NextBlob(self, prefix='unnamed'):
        """Return the blob that has not been defined or registered in the
        current net. It returns `BlobReference(prefix)`, if it's valid,
        otherwise `BlobReference(prefix) + '_auto_' + ?`. Different calls
        is guaranteed to return blob with different names."""
        output_blob_base = BlobReference(prefix)
        output_blob = output_blob_base
        index = 0
        while (str(output_blob) in self._registered_blob_names or self.BlobIsDefined(output_blob)):
            output_blob = output_blob_base + '_auto_' + str(index)
            index += 1
        self._registered_blob_names.add(str(output_blob))
        return output_blob
    
    def NextName(self, prefix=None, output_id=None):
        """Returns the next name to be used, if you do not want to explicitly
        name your blob. [Deprecated, use NextBlob, NextScopedBlob instead]"""
        if prefix:
            output_name_base = self._net.name + '/' + prefix
            output_name = output_name_base
            if output_id is not None:
                output_name += ':' + str(output_id)
            index = 2
            while self.BlobIsDefined(str(ScopedBlobReference(output_name))):
                output_name = output_name_base + '_' + str(index)
                if output_id is not None:
                    output_name += ':' + str(output_id)
                index += 1
        else:
            output_name = self._net.name + '_blob_' + str(self._next_name_index)
            self._next_name_index += 1
        return str(output_name)
    
    def _ExtendOps(self, new_ops):
        self._net.op.extend(new_ops)
        for op in new_ops:
            self._op_outputs.update([text_type(o) for o in op.output])
    
    def _CheckLookupTables(self):
        """
        Called from unit tests to validate the internal lookup tables
        match the protobuf contents.
        """
        test_op_outputs = set()
        for op in self._net.op:
            for o in op.output:
                test_op_outputs.add(o)
        test_external_inp = set()
        for inp in self._net.external_input:
            test_external_inp.add(inp)
        assert test_op_outputs.difference(self._op_outputs) == set()
        assert test_external_inp.difference(self._external_input_map) == set()
    
    def _InvalidateLookupTables(self):
        self._recreate_lookup_tables = True
    
    def _RecreateLookupTables(self):
        self._op_outputs = set()
        for op in self._net.op:
            for o in op.output:
                self._op_outputs.add(o)
        self._external_input_map = set()
        for inp in self._net.external_input:
            self._external_input_map.add(inp)
        self._recreate_lookup_tables = False
    
    def AddGradientOperators(self, ys, skip=0):
        """Add the gradient for operators in the net.

        Inputs:
          ys: a list or a dictionary specifying what blobs we want to compute
              derivatives of. If the input is a list, we will automatically
              generate their gradients with all-one values; if the input is a
              dictionary, for any dictionary entries that are not None, we will
              take the corresponding blobs as their gradients; for all those
              that are None, we will auto-fill them with 1.
          skip: skips the first n operators. This is provided mainly because a
              lot of nets may use the first few operators for data generation
              like stuff which really do not need to have gradients.

        Outputs:
          returns a map from the blob name in the input network to a blob
          containing gradient or a GradientSlice in case of sparse gradient

        Currently, this is hard-coded for float operators if there are branches
        (i.e. a blob is used as input to multiple operators). This is because
        the gradient accumulation (Sum) is float only right now.
        """
        (grad_ops, input_to_grad) = GradientRegistry.GetBackwardPass(self._net.op[skip:], ys)
        if workspace.IsImmediate():
            for op in grad_ops:
                workspace.RunOperatorImmediate(op)
        self._ExtendOps(grad_ops)
        return input_to_grad
    
    def AddArgument(self, arg_name, arg_value):
        self._net.arg.extend([utils.MakeArgument(arg_name, arg_value)])
    
    def AddExternalInput(self, *inputs):
        assert len(inputs) > 0
        refs = []
        for input in inputs:
            input_name = str(input)
            assert str(input) not in self._external_input_map, 'Net already contains an input named %s' % input_name
        for input in inputs:
            input_name = str(input)
            self._net.external_input.extend([input_name])
            self._external_input_map.update([input_name])
            refs.append(_get_blob_ref(input_name))
        return (refs[0] if len(refs) == 1 else refs)
    
    def AddExternalOutput(self, *outputs):
        for output in outputs:
            assert isinstance(output, BlobReference)
            assert self.BlobIsDefined(output), '{} is not defined'.format(output)
        for output in outputs:
            self.Proto().external_output.extend([str(output)])
    
    def AddScopedExternalInputs(self, *inputs):
        res = self.AddExternalInput(*[ScopedBlobReference(b) for b in inputs])
        if not isinstance(res, list):
            res = [res]
        return res
    
    def AddScopedExternalOutputs(self, *outputs):
        return self.AddExternalOutput(*[ScopedBlobReference(b) for b in outputs])
    
    def AddObserver(self, observer_type):
        return C.add_observer_to_net(self._net.name, observer_type)
    
    def RemoveObserver(self, observer):
        C.remove_observer_from_net(self._net.name, observer)
    
    def NumObservers(self):
        return C.num_observers_on_net(self._net.name)
    
    @property
    def external_inputs(self):
        return [_get_blob_ref(x) for x in self._net.external_input]
    
    @property
    def external_outputs(self):
        return [_get_blob_ref(x) for x in self._net.external_output]
    
    def set_input_record(self, input_record):
        from caffe2.python import schema
        assert (self._input_record is None or (input_record.has_blobs() and set(input_record.field_blobs()) == set(self._input_record.field_blobs()))), 'Input schema cannot be reset'
        if not input_record.has_blobs():
            with NameScope(self.Name()):
                self._input_record = schema.NewRecord(self, input_record)
        else:
            self._input_record = input_record
        for blob in self._input_record.field_blobs():
            if blob not in self.external_inputs:
                self.AddExternalInput(blob)
        return self._input_record
    
    def recover_input_record_by_prefix(self, prefix):
        """
        Tries to recover input record by taking a subset of external_inputs with
        a given prefix name and interpreting them as schema column names
        """
        record = _recover_record_by_prefix(self._net.external_input, prefix)
        if record:
            self.set_input_record(record)
    
    def set_output_record(self, record):
        assert (self._output_record is None or (record.has_blobs() and set(record.field_blobs()) == set(self._output_record.field_blobs()))), 'Output schema cannot be reset'
        for blob in record.field_blobs():
            assert self.BlobIsDefined(blob), '{} is not defined'.format(blob)
        for blob in record.field_blobs():
            if blob not in self.external_outputs:
                self.AddExternalOutput(blob)
        self._output_record = record
    
    def recover_output_record_by_prefix(self, prefix):
        """
        Tries to recover out record by taking a subset of external_outputs with
        a given prefix name and interpreting them as schema column names
        """
        record = _recover_record_by_prefix(self._net.external_output, prefix)
        if record:
            self.set_output_record(record)
    
    def AppendOutputRecordField(self, field_name, record):
        from caffe2.python import schema
        assert self._output_record is not None, 'Tried to append to missing output record'
        for blob in record.field_blobs():
            assert self.BlobIsDefined(blob), '{} is not defined'.format(blob)
        for blob in record.field_blobs():
            self.AddExternalOutput(blob)
        self._output_record = self._output_record + schema.Struct((field_name, record))
    
    def input_record(self):
        return self._input_record
    
    def output_record(self):
        return self._output_record
    
    def AddExternalInputs(self, *inputs):
        return self.AddExternalInput(*inputs)
    
    def AddExternalOutputs(self, *outputs):
        self.AddExternalOutput(*outputs)
    
    def DeduplicateGradientSlices(self, g, aggregator='sum'):
        assert isinstance(g, GradientSlice)
        (unique, remapping) = self.Unique([g.indices], 2, engine='SparseHash')
        if aggregator.lower() == 'sum':
            new_g = self.UnsortedSegmentSum([g.values, remapping], 1)
        elif aggregator.lower() == 'mean':
            new_g = self.UnsortedSegmentMean([g.values, remapping], 1)
        else:
            raise ValueError('{} is not supported'.format(aggregator))
        return GradientSlice(indices=unique, values=new_g)
    
    @staticmethod
    def _RunAllOnGPU(net, gpu_id=0, use_cudnn=False):
        device_option = caffe2_pb2.DeviceOption()
        device_option.device_type = workspace.GpuDeviceType
        device_option.device_id = gpu_id
        net.device_option.CopyFrom(device_option)
        if use_cudnn:
            for op in net.op:
                op.engine = 'CUDNN'
        for op in net.op:
            if op.type != 'RecurrentNetwork':
                continue
            for arg in op.arg:
                if arg.name == 'step_net':
                    Net._RunAllOnGPU(arg.n, gpu_id, use_cudnn)
    
    def RunAllOnGPU(self, gpu_id=0, use_cudnn=False):
        """A convenient function to run everything on the GPU."""
        self._RunAllOnGPU(self._net, gpu_id, use_cudnn)
    
    def RunAllOnMKL(self):
        """A convenient function to run everything using MKLDNN."""
        device_option = caffe2_pb2.DeviceOption()
        device_option.device_type = caffe2_pb2.MKLDNN
        self._net.device_option.CopyFrom(device_option)
    
    def RunAllOnIDEEP(self):
        """A convenient function to run everything using IDEEP."""
        device_option = caffe2_pb2.DeviceOption()
        device_option.device_type = caffe2_pb2.IDEEP
        self._net.device_option.CopyFrom(device_option)
    
    def _CreateAndAddToSelf(self, op_type, inputs, outputs=None, **kwargs):
        """A helper function to create an operator and add it to self.
        """
        inputs = _RectifyInputOutput(inputs)
        for input in inputs:
            if not self.BlobIsDefined(input):
                assert input.Net() != self
                self.AddExternalInput(input)
        if outputs is None:
            outputs = self.NextName(prefix=op_type)
        elif type(outputs) is int:
            outputs = [self.NextName(prefix=op_type, output_id=i) for i in range(outputs)]
        outputs = _RectifyInputOutput(outputs, net=self)
        op = CreateOperator(op_type, inputs, outputs, **kwargs)
        self._ExtendOps([op])
        workspace.operator_tracebacks[self.Name()][len(self._net.op) - 1] = _extract_stacktrace()
        if len(op.output) == 0:
            return
        elif len(op.output) == 1:
            return BlobReference(op.output[0], self)
        else:
            return tuple((BlobReference(o, self) for o in op.output))
    
    def __getattr__(self, op_type):
        if op_type.startswith('__'):
            raise AttributeError('Attribute {} not found.'.format(op_type))
        if (not IsOperator(op_type) and not IsOperatorWithEngine(op_type, 'CUDNN')):
            raise AttributeError('Method ' + op_type + ' is not a registered operator.' + ' Did you mean: [' + ','.join(workspace.C.nearby_opnames(op_type)) + ']')
        return lambda *args, **kwargs: self._CreateAndAddToSelf(op_type, *args, **kwargs)
    
    def __dir__(self):
        additional_methods = [op for op in _REGISTERED_OPERATORS if '_ENGINE_' not in op]
        return sorted(set(chain(dir(type(self)), viewkeys(self.__dict__), additional_methods)))
    
    def Python(self, f, grad_f=None, python_func_type=None, pass_workspace=False, grad_output_indices=None, grad_input_indices=None):
        """
        Registers and returns a python operator.

        `f` and `grad_f` can be one of the following:
            - a function with signature (inputs, outputs), where inputs and
              outputs are a list of CPUTensor objects. This function will be
              called from C++ everytime the operator is executed.
            - a tuple (func, args, kwargs), here `func` is a callable, args is
              an argument list, and kwargs is a dict list. The call:
                  f = func(*args, kwargs)
              will be performed locally at node initialization time, on all of
              the nodes of the job, returning `f`, a callable that will be used
              as the python operator function to be called during Net execution.
              This is to be used when using python operator in a distributed
              context, and allows to create and keep local python state across
              calls to the operator.

        `python_func_type` is a type of an object that constructed as
        python_func_type(f) and provides an implementation to forward and
        backward functions. Its useful in such a case where users needs
        a statefull PythonOp (ex: use autograd for computing grad_f).

        If `pass_workspace` is True, the signature is changed to
        (inputs, outputs, workspace) where `workspace` is the workspace the op
        is going to run on. This is potentially dangerous (as the op can
        manipulate the workspace directly), use on your own risk.

        If a gradient function is specified (`grad_f`), by default its inputs
        will be: (1) all inputs to `f`, (2) followed by all outputs of `f`, (3)
        and then all gradient outputs of `f`. The outputs of `grad_f` will be
        (by default) all gradient inputs to `f`. If a subset of the gradient
        outputs or gradient inputs is desired instead, then the subsets can be
        specified by providing `grad_output_indices` and/or `grad_input_indices`
        which identify the indices of `f`'s inputs and outputs which have
        gradients.
        """
        assert IsOperator('Python')
        
        def make_builder(t):
            if not isinstance(t, tuple):
                return ''
            assert len(t) == 3, 'Expected builder tuple (func, args, kwargs)'
            (func, args, kwargs) = t
            normalized = (func, tuple(args), dict(kwargs))
            return pickle.dumps(normalized)
        f_builder = make_builder(f)
        grad_f_builder = make_builder(grad_f)
        assert (not grad_f or (not f_builder) == (not grad_f_builder)), 'A tuple has to be passed to both f and grad_f or neither.'
        core_kwargs = {}
        if f_builder:
            core_kwargs['pickled_builder'] = f_builder
            core_kwargs['pickled_grad_builder'] = grad_f_builder
            core_kwargs['pass_workspace'] = pass_workspace
        else:
            core_kwargs['token'] = _RegisterPythonImpl(f, grad_f, python_func_type, pass_workspace=pass_workspace)
        grad_output_indices = (grad_output_indices or [])
        grad_input_indices = (grad_input_indices or [])
        return lambda *args, **kwargs: self._CreateAndAddToSelf('Python', *args, grad_output_indices=grad_output_indices, grad_input_indices=grad_input_indices, **dict(chain(viewitems(kwargs), viewitems(core_kwargs))))
    
    def is_external_input(self, blob):
        name = str(blob)
        return name in self._external_input_map
    
    def extend_ops(self, new_ops):
        return self._ExtendOps(new_ops)


def remap_input(op, blob_name_remapping):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.core.remap_input', 'remap_input(op, blob_name_remapping)', {'op': op, 'blob_name_remapping': blob_name_remapping}, 0)

def copy_func_between_devices(src, dst):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.core.copy_func_between_devices', 'copy_func_between_devices(src, dst)', {'caffe2_pb2': caffe2_pb2, 'IsGPUDeviceType': IsGPUDeviceType, 'DeviceScope': DeviceScope, 'src': src, 'dst': dst}, 1)

def device_equal(src, dst):
    """
    We are using this fucntion instead of == operator because optional-value
    comparison between empty device_options and {device_type:0, device_id:0}
    returns not equal in some cases.
    """
    return (src.device_type == dst.device_type and src.device_id == dst.device_id)

def update_placeholder_op_output(op, blob_to_device):
    """
    Placeholder ops (for e.g. Recv) always runs on CPU. So ensure their
    output blobs reside on CPU.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.core.update_placeholder_op_output', 'update_placeholder_op_output(op, blob_to_device)', {'caffe2_pb2': caffe2_pb2, 'op': op, 'blob_to_device': blob_to_device}, 0)


class RemapEntry:
    
    def __init__(self, blob, device):
        self.blob = blob
        self.device = device
    
    def __eq__(self, other):
        return (self.blob == other.blob and self.device == other.device)
    
    def __hash__(self):
        return hash(self.blob + str(self.device))


def InjectCrossDeviceCopies(net, blob_to_device=None, blob_remap=None, placeHolderOps=None):
    """
    Injecting Copy functions between device within a net. Users can provide
    a net with part of operators using different device_options. This method
    will automatically create a new net with Copy ops inserted in it.

    Inputs:
      blob_to_device: If not None, it is a map of blobs and their device locations.
      blob_remap: If not None, it is a map from a pair (blob, device) to
                  the name of the blob in the given device. Blobs found in this
                  map are assumed to be cached and don't need to be copied.
    Outputs:
      new_net: A new net with CopyCPUToGPU inserted with correct device option

      required_external_to_device:
               A mapping between unresolved external inputs and their
               required device options.
    Assumptions:
      1. every external inputs of this net is already in blob_to_device!
      2. if not, this function will use net device option
      3. InferOpBlobDevices might fail to get the correct inference for ops like
         EnsureCPUOutput that could take in input from multiple places.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.core.InjectCrossDeviceCopies', 'InjectCrossDeviceCopies(net, blob_to_device=None, blob_remap=None, placeHolderOps=None)', {'caffe2_pb2': caffe2_pb2, 'defaultdict': defaultdict, 'InferOpDeviceAsBlobDevices': InferOpDeviceAsBlobDevices, 'InferOpBlobDevices': InferOpBlobDevices, 'device_equal': device_equal, 'RemapEntry': RemapEntry, 'copy_func_between_devices': copy_func_between_devices, 'IsGPUDeviceType': IsGPUDeviceType, 'update_placeholder_op_output': update_placeholder_op_output, 'net': net, 'blob_to_device': blob_to_device, 'blob_remap': blob_remap, 'placeHolderOps': placeHolderOps}, 1)

def InjectDeviceCopiesAmongNets(nets, blob_to_device_init=None):
    """
    Takes in a list of nets. They usually represent your whole execution graph.
    This function will insert cross device copy functions to all nets, and resolve
    inter-net external inputs dependencies. This method will insert Copy funcitons if
    external inputs of a net is produced on different device than it is required.
    Inputs:
      nets: a list of nets
    Outputs:
      new_nets: a list of new nets with device difference solved.

    Some notes from wyiming:
      1. You MUST pass nets in execution order. e.g. [train_init, train]
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.core.InjectDeviceCopiesAmongNets', 'InjectDeviceCopiesAmongNets(nets, blob_to_device_init=None)', {'Net': Net, 'InjectCrossDeviceCopies': InjectCrossDeviceCopies, 'nets': nets, 'blob_to_device_init': blob_to_device_init}, 2)

def InjectDeviceCopiesAmongNetsWithoutB2D(nets, blob_to_device_init=None):
    (new_nets, _) = InjectDeviceCopiesAmongNets(nets, blob_to_device_init)
    return new_nets

def get_net_name(netlike):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.core.get_net_name', 'get_net_name(netlike)', {'Net': Net, 'caffe2_pb2': caffe2_pb2, 'netlike': netlike}, 1)

def output_to_list(op_output):
    """
    Ensures that the output of an operator is a list.
    Use when an operator has a variable number of outputs, but a list of
    outputs is desired even when number of outputs is 1.

    Args:
        op_output: Either a BlobReferenece or an iterable of BlobReferences.

    Returns:
        A list of BlobReferences.
    """
    assert type(op_output) in (list, tuple, BlobReference)
    return ([op_output] if isinstance(op_output, BlobReference) else list(op_output))

def _add_net_to_dict(net_dict, net):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.core._add_net_to_dict', '_add_net_to_dict(net_dict, net)', {'get_net_name': get_net_name, 'Net': Net, 'net_dict': net_dict, 'net': net}, 1)


class ExecutionStep(object):
    _step_names_used = set()
    
    @staticmethod
    def _get_next_step_name(basename):
        name = basename
        next_idx = 1
        while name in ExecutionStep._step_names_used:
            name = basename + '_' + str(next_idx)
            next_idx += 1
        ExecutionStep._step_names_used |= set([name])
        return name
    
    def __init__(self, name, nets=None, num_iter=None):
        self._step = caffe2_pb2.ExecutionStep()
        self._step.name = (name or ExecutionStep._get_next_step_name('step'))
        self._net_dict = OrderedDict()
        self._is_used = False
        self._substeps = []
        if nets is not None:
            if type(nets) is Net:
                nets = [nets]
            for net in nets:
                if _add_net_to_dict(self._net_dict, net):
                    self._step.network.extend([get_net_name(net)])
        if num_iter is not None:
            self._step.num_iter = num_iter
    
    def get_net(self, name):
        return self._net_dict[name]
    
    def Name(self):
        return self._step.name
    
    def __str__(self):
        return self._step.name
    
    def _assert_can_mutate(self):
        assert not self._is_used, 'Cannot mutate a step that has already been added to a plan/step.'
    
    def _notify_is_used(self):
        self._is_used = True
    
    def Proto(self):
        return self._step
    
    def HasNets(self):
        return (self._step.network is not None and len(self._step.network) > 0)
    
    def HasSubsteps(self):
        return (self._step.substep is not None and len(self._step.substep) > 0)
    
    def Nets(self):
        return list(viewvalues(self._net_dict))
    
    def Substeps(self):
        return self._substeps
    
    def SetIter(self, num_iter):
        self._assert_can_mutate()
        self._step.num_iter = num_iter
    
    def SetCreateWorkspace(self, create_workspace):
        self._assert_can_mutate()
        self._step.create_workspace = create_workspace
    
    def SetNumConcurrentInstances(self, num_concurrent_instances):
        self._assert_can_mutate()
        self._step.num_concurrent_instances = num_concurrent_instances
    
    def SetOnlyOnce(self, only_once):
        self._assert_can_mutate()
        self._step.only_once = only_once
    
    def SetShouldStopBlob(self, should_stop_blob):
        assert isinstance(should_stop_blob, BlobReference), 'expects BlobReference here, got {}'.format(type(should_stop_blob))
        self._assert_can_mutate()
        self._step.should_stop_blob = str(should_stop_blob)
    
    def RunEveryMillis(self, interval):
        """
        Run this step every interval millisecods, as long as its
        siblings are still running. It is guaranteed that, after all
        siblings finish, this step will run at least one.

        This property is ignored for top-level ExecutionSteps.
        """
        self._step.run_every_ms = interval
    
    def SetReportNet(self, report_net, report_interval):
        """ DEPRECATED. Use RunEveryMillis instead. """
        self._assert_can_mutate()
        _add_net_to_dict(self._net_dict, report_net)
        self._step.report_net = get_net_name(report_net)
        self._step.report_interval = report_interval
    
    def AddSubstep(self, substep):
        self._assert_can_mutate()
        assert not self.HasNets(), 'Cannot have both network and substeps.'
        if isinstance(substep, ExecutionStep):
            substep._notify_is_used()
            if (not substep.HasNets() and not substep.HasSubsteps()):
                return self
            for net in substep.Nets():
                _add_net_to_dict(self._net_dict, net)
            self._substeps.append(substep)
            proto = substep.Proto()
        else:
            proto = substep
        self._step.substep.add().CopyFrom(proto)
        return self
    
    def SetConcurrentSubsteps(self, concurrent_substeps):
        self._assert_can_mutate()
        assert not self.HasNets(), 'Cannot have both network and substeps.'
        self._step.concurrent_substeps = concurrent_substeps
    
    def AddNet(self, net):
        self._assert_can_mutate()
        assert not self.HasSubsteps(), 'Cannot have both network and substeps.'
        assert isinstance(net, Net)
        _add_net_to_dict(self._net_dict, net)
        self._step.network.extend([get_net_name(net)])
        return self
    
    def get_all_attributes(self, name):
        """
        Return the list of all attributes under the given `name`, present in
        all of the nets used in this execution step and its children.
        """
        return [attr for net in viewvalues(self._net_dict) for attr in net.get_attributes(name)]
    
    @classmethod
    def create_from_proto(cls, step_proto, net_obj_dict, net_proto_dict):
        """
        Create ExecutionStep from ExecutionStep protobuf recursively
        """
        assert isinstance(step_proto, caffe2_pb2.ExecutionStep)
        assert ((len(step_proto.network) > 0 and len(step_proto.substep) == 0) or (len(step_proto.network) == 0 and len(step_proto.substep) > 0))
        steps_or_nets = []
        if len(step_proto.substep) > 0:
            for substep_proto in step_proto.substep:
                steps_or_nets.append(ExecutionStep.create_from_proto(substep_proto, net_obj_dict, net_proto_dict))
        else:
            for net_name in step_proto.network:
                if net_name not in net_obj_dict:
                    assert net_name in net_proto_dict
                    net = Net(net_proto_dict[net_name])
                    net_obj_dict[net_name] = net
                net = net_obj_dict[net_name]
                assert isinstance(net, Net)
                steps_or_nets.append(net)
        num_iter = (step_proto.num_iter if step_proto.HasField('num_iter') else None)
        concurrent_substeps = (step_proto.concurrent_substeps if step_proto.HasField('concurrent_substeps') else None)
        should_stop_blob = (BlobReference(step_proto.should_stop_blob) if step_proto.HasField('should_stop_blob') else None)
        only_once = (step_proto.only_once if step_proto.HasField('only_once') else None)
        num_concurrent_instances = (step_proto.num_concurrent_instances if step_proto.HasField('num_concurrent_instances') else None)
        create_workspace = (step_proto.create_workspace if step_proto.HasField('create_workspace') else None)
        run_every_ms = (step_proto.run_every_ms if step_proto.HasField('run_every_ms') else None)
        return execution_step(step_proto.name, steps_or_nets, num_iter=num_iter, report_net=None, report_interval=None, concurrent_substeps=concurrent_substeps, should_stop_blob=should_stop_blob, only_once=only_once, num_concurrent_instances=num_concurrent_instances, create_workspace=create_workspace, run_every_ms=run_every_ms)


def add_nets_in_order(step, net_list):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.core.add_nets_in_order', 'add_nets_in_order(step, net_list)', {'add_nets_in_order': add_nets_in_order, 'step': step, 'net_list': net_list}, 0)


class Plan(object):
    
    def __init__(self, name_or_step):
        self._plan = caffe2_pb2.PlanDef()
        self._net_dict = OrderedDict()
        self._steps = []
        if isinstance(name_or_step, ExecutionStep):
            self._plan.name = name_or_step.Name()
            self.AddStep(name_or_step)
        elif isinstance(name_or_step, basestring):
            self._plan.name = name_or_step
        else:
            raise ValueError('name_or_step must be a string or ExecutionStep')
    
    def __str__(self):
        return self._plan.name
    
    def Proto(self):
        return self._plan
    
    def AddNets(self, nets):
        for net in nets:
            if _add_net_to_dict(self._net_dict, net):
                assert isinstance(net, Net)
                self._plan.network.add().CopyFrom(net.Proto())
    
    def Nets(self):
        return list(viewvalues(self._net_dict))
    
    def AddStep(self, step):
        assert isinstance(step, ExecutionStep)
        step._notify_is_used()
        if (not step.HasNets() and not step.HasSubsteps()):
            return
        self._plan.execution_step.add().CopyFrom(step.Proto())
        self._steps.append(step)
        net_list = []
        add_nets_in_order(step, net_list)
        self.AddNets([step.get_net(n) for n in net_list])
    
    def Steps(self):
        return self._steps
    
    def get_all_attributes(self, name):
        """
        Return the list of all attributes under the given `name`, present in
        all of the nets used in this plan.
        """
        return [attr for net in viewvalues(self._net_dict) for attr in net.get_attributes(name)]
    
    @classmethod
    def create_from_proto(cls, plan_proto):
        assert isinstance(plan_proto, caffe2_pb2.PlanDef)
        plan = Plan(plan_proto.name)
        plan._plan.CopyFrom(plan_proto)
        del plan._plan.network[:]
        del plan._plan.execution_step[:]
        net_obj_dict = {}
        net_proto_dict = {}
        for net_proto in plan_proto.network:
            assert net_proto.name not in net_proto_dict
            net_proto_dict[net_proto.name] = net_proto
        for step_proto in plan_proto.execution_step:
            step = ExecutionStep.create_from_proto(step_proto, net_obj_dict, net_proto_dict)
            plan.AddStep(step)
        return plan


def to_execution_step(step_or_nets, default_name=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.core.to_execution_step', 'to_execution_step(step_or_nets, default_name=None)', {'ExecutionStep': ExecutionStep, 'execution_step': execution_step, 'step_or_nets': step_or_nets, 'default_name': default_name}, 1)

def execution_step(default_name, steps_or_nets, num_iter=None, report_net=None, report_interval=None, concurrent_substeps=None, should_stop_blob=None, only_once=None, num_concurrent_instances=None, create_workspace=False, run_every_ms=None):
    """
    Helper for creating an ExecutionStep.
    - steps_or_nets can be:
      - None
      - Net
      - ExecutionStep
      - list<Net>
      - list<ExecutionStep>
    - should_stop_blob is either None or a scalar boolean blob.
      - This blob is checked AFTER every substeps/subnets.
      - If specified and true, then this step will return immediately.
      - Be sure to handle race conditions if setting from concurrent threads.
    - if no should_stop_blob or num_iter is provided, defaults to num_iter=1
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.core.execution_step', 'execution_step(default_name, steps_or_nets, num_iter=None, report_net=None, report_interval=None, concurrent_substeps=None, should_stop_blob=None, only_once=None, num_concurrent_instances=None, create_workspace=False, run_every_ms=None)', {'ExecutionStep': ExecutionStep, 'Net': Net, 'to_execution_step': to_execution_step, 'default_name': default_name, 'steps_or_nets': steps_or_nets, 'num_iter': num_iter, 'report_net': report_net, 'report_interval': report_interval, 'concurrent_substeps': concurrent_substeps, 'should_stop_blob': should_stop_blob, 'only_once': only_once, 'num_concurrent_instances': num_concurrent_instances, 'create_workspace': create_workspace, 'run_every_ms': run_every_ms}, 1)

def scoped_execution_step(name, *args, **kwargs):
    """Same as execution_step() except that the step name is scoped."""
    default_name = (ScopedName(name) if name else name)
    return execution_step(default_name, *args, **kwargs)

def _extract_stacktrace():
    """
    This function extracts stacktrace without file system access
    by purely using sys._getframe() and removes part that belongs to
    this file (core.py). We are not using inspect module because
    its just a wrapper on top of sys._getframe() whose
    logic is based on accessing source files on disk - exactly what
    we are trying to avoid here. Same stands for traceback module

    The reason for file system access avoidance is that
    if code is located on an NFS, file access might be slow

    Function returns a list of tuples (file_name, line_number, function)
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.core._extract_stacktrace', '_extract_stacktrace()', {'sys': sys}, 1)
SetPerOpEnginePref = C.set_per_op_engine_pref
SetGlobalEnginePref = C.set_global_engine_pref
SetEnginePref = C.set_engine_pref
SetOpEnginePref = C.set_op_engine_pref

