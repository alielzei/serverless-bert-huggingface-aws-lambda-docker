import torch._C
import torch._jit_internal as _jit_internal
import torch.jit.annotations
import torch.testing
import torch.jit._recursive
from torch.jit._recursive import ScriptMethodStub
from torch.jit._builtins import _find_builtin, _get_builtin_table, _register_builtin
from torch._jit_internal import _qualified_name
from torch.autograd import Variable, function
from torch.jit.frontend import get_jit_class_def, get_jit_def, get_default_args
from torch.nn import Module
from torch.serialization import validate_cuda_device
from torch._six import PY2, PY37, with_metaclass, string_classes, get_function_from_type
from torch.utils import set_module
import collections
import contextlib
import copy
import functools
import inspect
import os
import pathlib
import pickle
import re
import sys
import textwrap
import warnings
import weakref
from torch._jit_internal import Final, _overload, _overload_method
from torch._jit_internal import ignore, export, unused

def _parse_env(name, default, true_message, false_message):
    value = os.environ.get(name)
    if value is None:
        return default
    if value.lower() in {'1', 'true', 'yes'}:
        return True
    elif value.lower() in {'0', 'false', 'no'}:
        return False
    if value == '1v':
        print(true_message)
        return True
    elif value == '0v':
        print(false_message)
        return False
    raise ValueError('Unknown setting of {}. Try using 0 or 1.'.format(name))
_enabled = _parse_env('PYTORCH_JIT', True, '> Using PyTorch JIT', '> PyTorch JIT DISABLED')
_flatten = torch._C._jit_flatten
_unflatten = torch._C._jit_unflatten
_jit_script_class_compile = torch._C._jit_script_class_compile
_python_cu = torch._C.CompilationUnit()
Future = torch._C.Future
set_module(Future, 'torch.jit')
_fork = torch._C.fork
_wait = torch._C.wait
if _enabled:
    Attribute = collections.namedtuple('Attribute', ['value', 'type'])
else:
    
    def Attribute(value, type):
        return value

@contextlib.contextmanager
def optimized_execution(should_optimize):
    """
    A context manager that controls whether the JIT's executor will run
    optimizations before executing a function.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.jit.__init__.optimized_execution', 'optimized_execution(should_optimize)', {'torch': torch, 'contextlib': contextlib, 'should_optimize': should_optimize}, 0)
DEFAULT_EXTRA_FILES_MAP = torch._C.ExtraFilesMap()

def save(m, f, _extra_files=DEFAULT_EXTRA_FILES_MAP):
    """
        Save an offline version of this module for use in a separate process. The saved
        module serializes all of the methods, submodules, parameters, and attributes of this
        module. It can be loaded into the C++ API using ``torch::jit::load(filename)`` or into the Python
        API with :func:`torch.jit.load <torch.jit.load>`.

        To be able to save a module, it must not make any calls to native Python functions.
        This means that all submodules must be subclasses of :class:`ScriptModule` as well.

        .. DANGER::
           All modules, no matter their device, are always loaded onto the CPU during loading.
           This is different from :func:`torch.load`'s semantics and may change in the future.

        Arguments:
            m: A :class:`ScriptModule` to save.
            f: A file-like object (has to implement write and flush) or a string
               containing a file name.
            _extra_files: Map from filename to contents which will be stored as part of 'f'.

        .. warning::
            If you are using Python 2, ``torch.jit.save`` does NOT support :any:`StringIO.StringIO`
            as a valid file-like object. This is because the write method should return
            the number of bytes written; ``StringIO.write()`` does not do this.

            Please use something like ``io.BytesIO`` instead.

        Example:

        .. testcode::

            import torch
            import io

            class MyModule(torch.nn.Module):
                def forward(self, x):
                    return x + 10

            m = torch.jit.script(MyModule())

            # Save to file
            torch.jit.save(m, 'scriptmodule.pt')
            # This line is equivalent to the previous
            m.save("scriptmodule.pt")

            # Save to io.BytesIO buffer
            buffer = io.BytesIO()
            torch.jit.save(m, buffer)

            # Save with extra files
            extra_files = torch._C.ExtraFilesMap()
            extra_files['foo.txt'] = 'bar'
            torch.jit.save(m, 'scriptmodule.pt', _extra_files=extra_files)
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.jit.__init__.save', 'save(m, f, _extra_files=DEFAULT_EXTRA_FILES_MAP)', {'pathlib': pathlib, 'm': m, 'f': f, '_extra_files': _extra_files, 'DEFAULT_EXTRA_FILES_MAP': DEFAULT_EXTRA_FILES_MAP}, 0)

def load(f, map_location=None, _extra_files=DEFAULT_EXTRA_FILES_MAP):
    """
        Load a :class:`ScriptModule` or :class:`ScriptFunction` previously
        saved with :func:`torch.jit.save <torch.jit.save>`

        All previously saved modules, no matter their device, are first loaded onto CPU,
        and then are moved to the devices they were saved from. If this fails (e.g. because
        the run time system doesn't have certain devices), an exception is raised.

        Arguments:
            f: a file-like object (has to implement read, readline, tell, and seek),
                or a string containing a file name
            map_location (string or torch.device): A simplified version of ``map_location`` in
                ``torch.save`` used to dynamically remap storages to an alternative set of devices.
            _extra_files (dictionary of filename to content): The extra
                filenames given in the map would be loaded and their content
                would be stored in the provided map.

        Returns:
            A :class:`ScriptModule` object.

        Example:

        .. testcode::

            import torch
            import io

            torch.jit.load('scriptmodule.pt')

            # Load ScriptModule from io.BytesIO object
            with open('scriptmodule.pt', 'rb') as f:
                buffer = io.BytesIO(f.read())

            # Load all tensors to the original device
            torch.jit.load(buffer)

            # Load all tensors onto CPU, using a device
            buffer.seek(0)
            torch.jit.load(buffer, map_location=torch.device('cpu'))

            # Load all tensors onto CPU, using a string
            buffer.seek(0)
            torch.jit.load(buffer, map_location='cpu')

            # Load with extra files.
            extra_files = torch._C.ExtraFilesMap()
            extra_files['foo.txt'] = 'bar'
            torch.jit.load('scriptmodule.pt', _extra_files=extra_files)
            print(extra_files['foo.txt'])

        .. testoutput::
            :hide:

            ...

        .. testcleanup::

            import os
            os.remove("scriptmodule.pt")
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.jit.__init__.load', 'load(f, map_location=None, _extra_files=DEFAULT_EXTRA_FILES_MAP)', {'string_classes': string_classes, 'os': os, 'torch': torch, 'validate_cuda_device': validate_cuda_device, 'pathlib': pathlib, 'f': f, 'map_location': map_location, '_extra_files': _extra_files, 'DEFAULT_EXTRA_FILES_MAP': DEFAULT_EXTRA_FILES_MAP}, 1)

def export_opnames(m):
    """
        Returns a list of operator names of a script module and its submodules
    """
    return torch._C._export_opnames(m._c)

def _get_trace_graph(f, args=(), kwargs=None, _force_outplace=False, return_inputs=False, _return_inputs_states=False):
    """
    .. warning::
        This function is internal-only and should only be used by the ONNX
        exporter. If you are trying to get a graph through tracing, please go
        through the public API instead::

            trace = torch.jit.trace(nn.LSTMCell(), (input, hidden))
            trace_graph = trace.graph

    Trace a function or model, returning a tuple consisting of the both the
    *trace* of an execution, as well as the original return value. If return_inputs,
    also returns the trace inputs as part of the tuple

    Tracing is guaranteed not to change the semantics of the function/module
    that is traced.

    Arguments:
        f (torch.nn.Module or function): the function or module
            to be traced.
        args (tuple or Tensor): the positional arguments to pass to the
            function/module to be traced.  A non-tuple is assumed to
            be a single positional argument to be passed to the model.
        kwargs (dict): the keyword arguments to pass to the function/module
            to be traced.

    Example (trace a cell):

    .. testcode::

        trace = torch.jit.trace(nn.LSTMCell(), (input, hidden))
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.jit.__init__._get_trace_graph', '_get_trace_graph(f, args=(), kwargs=None, _force_outplace=False, return_inputs=False, _return_inputs_states=False)', {'ONNXTracedModule': ONNXTracedModule, 'f': f, 'args': args, 'kwargs': kwargs, '_force_outplace': _force_outplace, 'return_inputs': return_inputs, '_return_inputs_states': _return_inputs_states}, 1)

def _unique_state_dict(module, keep_vars=False):
    state_dict = module.state_dict(keep_vars=True)
    filtered_dict = type(state_dict)()
    seen_ids = set()
    for (k, v) in state_dict.items():
        if id(v) in seen_ids:
            continue
        seen_ids.add(id(v))
        if keep_vars:
            filtered_dict[k] = v
        else:
            filtered_dict[k] = v.detach()
    return filtered_dict

def _create_interpreter_name_lookup_fn(frames_up=1):
    
    def _get_interpreter_name_for_var(var):
        frame = inspect.currentframe()
        i = 0
        while i < frames_up + 1:
            frame = frame.f_back
            i += 1
        f_locals = frame.f_locals
        f_globals = frame.f_globals
        for (k, v) in f_locals.items():
            if (isinstance(v, torch.Tensor) and var is v):
                return (k if k != 'self' else '')
        for (k, v) in f_globals.items():
            if (isinstance(v, torch.Tensor) and var is v):
                return (k if k != 'self' else '')
        return ''
    return _get_interpreter_name_for_var


class ONNXTracedModule(Module):
    
    def __init__(self, inner, force_outplace=False, return_inputs=False, return_inputs_states=False):
        super(ONNXTracedModule, self).__init__()
        self.inner = inner
        self._force_outplace = force_outplace
        self._return_inputs = return_inputs
        self._return_inputs_states = return_inputs_states
    
    def forward(self, *args):
        (in_vars, in_desc) = _flatten(args)
        module_state = list(_unique_state_dict(self, keep_vars=True).values())
        ret_inputs = []
        inputs_states = []
        outs = []
        
        def wrapper(*args):
            trace_inputs = _unflatten(args[:len(in_vars)], in_desc)
            ret_inputs.append(tuple((x.clone(memory_format=torch.preserve_format) for x in args)))
            if self._return_inputs_states:
                inputs_states.append(_unflatten(args[:len(in_vars)], in_desc))
            outs.append(self.inner(*trace_inputs))
            if self._return_inputs_states:
                inputs_states[0] = (inputs_states[0], trace_inputs)
            (out_vars, _) = _flatten(outs)
            if len(out_vars) == 1:
                return out_vars[0]
            else:
                return tuple(out_vars)
        (graph, out) = torch._C._create_graph_by_tracing(wrapper, in_vars + module_state, _create_interpreter_name_lookup_fn(), self._force_outplace)
        if self._return_inputs:
            return (graph, outs[0], ret_inputs[0])
        if self._return_inputs_states:
            return (graph, outs[0], inputs_states[0])
        else:
            return (graph, outs[0])


def _clone_inputs(args):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.jit.__init__._clone_inputs', '_clone_inputs(args)', {'torch': torch, 'args': args}, 1)
_JIT_TIME = os.environ.get('PYTORCH_JIT_TIME', False)
_JIT_DISABLE = os.environ.get('PYTORCH_JIT_DISABLE', False)
_JIT_STATS = os.environ.get('PYTORCH_JIT_STATS', False)

@contextlib.contextmanager
def _time(trace_name, name, time=True):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.jit.__init__._time', '_time(trace_name, name, time=True)', {'_JIT_TIME': _JIT_TIME, 'torch': torch, 'contextlib': contextlib, 'trace_name': trace_name, 'name': name, 'time': time}, 1)

def verify(model, args, loss_fn=torch.sum, devices=None):
    """
    Verify that a JIT compiled model has the same behavior as its uncompiled
    version along with its backwards pass.  If your model returns multiple
    outputs, you must also specify a `loss_fn` to produce a loss for which
    the backwards will be computed.

    This function has side-effects (e.g., it executes your model / saves and loads
    parameters), so don't expect the model to come out exactly the same as what
    you passed in.

    Arguments:
        model (compiled torch.nn.Module or function): the module/function to be
            verified.  The module/function definition MUST have been decorated with
            `@torch.jit.compile`.
        args (tuple or Tensor): the positional arguments to pass to the
            compiled function/module to be verified.  A non-tuple is assumed to
            be a single positional argument to be passed to the model.
        loss_fn (function, optional): the loss function to be applied to
            the output of the model, before backwards is invoked.  By default,
            we assume that a model returns a single result, and we :func:`torch.sum`
            before calling backwards; if this is inappropriate, you can pass your
            own loss function.  Note that if a model returns a tuple of results,
            these are passed as separate positional arguments to `loss_fn`.
        devices (iterable of device IDs, optional): the GPU devices which the
            compiled module will be run on.  This determines the RNG state we
            must save when running both compiled and uncompiled versions of the model.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.jit.__init__.verify', 'verify(model, args, loss_fn=torch.sum, devices=None)', {'torch': torch, 'Module': Module, '_clone_inputs': _clone_inputs, 'copy': copy, '_flatten': _flatten, '_verify_equal': _verify_equal, 'model': model, 'args': args, 'loss_fn': loss_fn, 'devices': devices}, 2)

def _verify_equal(xs, ys):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.jit.__init__._verify_equal', '_verify_equal(xs, ys)', {'xs': xs, 'ys': ys}, 0)

def indent(s):
    return '\n'.join(['\t' + line for line in s.splitlines()])


class TracingCheckError(Exception):
    
    def __init__(self, graph_diff_error, tensor_compare_error, extra_msg=None):
        self.message = 'Tracing failed sanity checks!\n'
        if extra_msg is not None:
            self.message += extra_msg + '\n'
        if graph_diff_error is not None:
            self.message += 'ERROR: Graphs differed across invocations!\n'
            self.message += indent(graph_diff_error) + '\n'
        if tensor_compare_error is not None:
            self.message += 'ERROR: Tensor-valued Constant nodes differed in value across invocations. This often indicates that the tracer has encountered untraceable code.\n'
            self.message += indent(tensor_compare_error) + '\n'
        super(TracingCheckError, self).__init__(self.message)


@torch.no_grad()
def _check_trace(check_inputs, func, traced_func, check_tolerance, force_outplace, is_trace_module, _module_class):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.jit.__init__._check_trace', '_check_trace(check_inputs, func, traced_func, check_tolerance, force_outplace, is_trace_module, _module_class)', {'_clone_inputs': _clone_inputs, 're': re, 'indent': indent, 'TracingCheckError': TracingCheckError, 'warnings': warnings, 'TracerWarning': TracerWarning, 'torch': torch, 'check_inputs': check_inputs, 'func': func, 'traced_func': traced_func, 'check_tolerance': check_tolerance, 'force_outplace': force_outplace, 'is_trace_module': is_trace_module, '_module_class': _module_class}, 2)


class TracerWarning(Warning):
    
    @staticmethod
    def ignore_lib_warnings():
        warnings.filterwarnings('ignore', category=TracerWarning, module='torch.(?!jit)')

TracerWarning.ignore_lib_warnings()
torch._C._tracer_warn_use_python()

def make_tuple(example_inputs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.jit.__init__.make_tuple', 'make_tuple(example_inputs)', {'torch': torch, 'example_inputs': example_inputs}, 1)

def make_module(mod, _module_class, _compilation_unit):
    if isinstance(mod, ScriptModule):
        return mod
    elif torch._jit_internal.module_has_exports(mod):
        
        def make_stubs_from_exported_methods(mod):
            exported = []
            for name in dir(mod):
                item = getattr(mod, name, None)
                if torch._jit_internal.get_torchscript_modifier(item) is _jit_internal.FunctionModifiers.EXPORT:
                    exported.append(name)
            stubs = []
            for method in exported:
                stubs.append(torch.jit._recursive.make_stub_from_method(mod, method))
            return stubs
        return torch.jit._recursive.create_script_module(mod, make_stubs_from_exported_methods, share_types=False)
    else:
        if _module_class is None:
            _module_class = TopLevelTracedModule
        return _module_class(mod, _compilation_unit=_compilation_unit)

def wrap_check_inputs(check_inputs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.jit.__init__.wrap_check_inputs', 'wrap_check_inputs(check_inputs)', {'check_inputs': check_inputs}, 1)

def trace(func, example_inputs, optimize=None, check_trace=True, check_inputs=None, check_tolerance=1e-05, _force_outplace=False, _module_class=None, _compilation_unit=_python_cu):
    """
    Trace a function and return an executable  or :class:`ScriptFunction`
    that will be optimized using just-in-time compilation. Tracing is ideal for
    code that operates only on ``Tensor``\s and lists, dictionaries, and tuples of ``Tensor``\s.

    Using ``torch.jit.trace`` and :func:`torch.jit.trace_module<torch.jit.trace_module>`, you can turn an existing module or Python
    function into a TorchScript :class:`ScriptFunction` or :class:`ScriptModule`. You must provide example inputs,
    and we run the function, recording the operations performed on all the tensors.

    * The resulting recording of a standalone function produces :class:`ScriptFunction`.
    * The resulting recording of ``forward`` function of ``nn.Module`` or ``nn.Module`` produces :class:`ScriptModule`.

    This module also contains any parameters that the original
    module had as well.

    .. warning::
        Tracing only correctly records functions and modules which are not data
        dependent (e.g., do not have conditionals on data in tensors) and do not have
        any untracked external dependencies (e.g., perform input/output or
        access global variables). Tracing only records operations done when the given
        function is run on the given
        tensors. Therefore, the returned :class:`ScriptModule` will always run the same traced
        graph on any input. This has some important implications when your module is
        expected to run different sets of operations, depending on the input and/or the
        module state. For example,

        * Tracing will not record any control-flow like if-statements or loops.
          When this control-flow is constant across your module, this is fine and it often
          inlines the control-flow decisions. But sometimes the control-flow is actually part
          of the model itself. For instance, a recurrent network is a loop over
          the (possibly dynamic) length of an input sequence.
        * In the returned :class:`ScriptModule`, operations that have different
          behaviors in ``training`` and ``eval`` modes will always behave as if it
          is in the mode it was in during tracing, no matter which mode the
          :class:`ScriptModule` is in.

        In cases like these, tracing would not be appropriate and :func:`scripting <torch.jit.script>` is a better
        choice. If you trace such models, you may silently get
        incorrect results on subsequent invocations of the model. The tracer
        will try to emit warnings when doing something that may cause an
        incorrect trace to be produced.

    Arguments:
        func (callable or torch.nn.Module):  A Python function or ``torch.nn.Module``
                                             that will be run with ``example_inputs``.
                                             arguments and returns to ``func`` must be tensors
                                             or (possibly nested) tuples that
                                             contain tensors. When a module is passed to
                                             :func:`torch.jit.trace <torch.jit.trace>`, only the
                                             ``forward`` method is run and traced
                                             (see :func:`torch.jit.trace <torch.jit.trace_module>` for details).
        example_inputs (tuple):  A tuple of example inputs that will be passed to the function
                                 while tracing. The resulting trace can be run with
                                 inputs of different types and shapes assuming the traced operations
                                 support those types and shapes. ``example_inputs`` may also be a single
                                 Tensor in which case it is automatically wrapped in a tuple.

    Keyword arguments:
        check_trace (bool, optional): Check if the same inputs run through
                                      traced code produce the same outputs. Default: ``True``. You might want
                                      to disable this if, for example, your network contains non-
                                      deterministic ops or if you are sure that the network is correct despite
                                      a checker failure.

        check_inputs (list of tuples, optional): A list of tuples of input arguments that should be used
                                                 to check the trace against what is expected. Each tuple
                                                 is equivalent to a set of input arguments that would
                                                 be specified in ``example_inputs``. For best results, pass in a
                                                 set of checking inputs representative of the space of
                                                 shapes and types of inputs you expect the network to see.
                                                 If not specified, the original ``example_inputs`` are used for checking
        check_tolerance (float, optional): Floating-point comparison tolerance to use in the checker procedure.
                                           This can be used to relax the checker strictness in the event that
                                           results diverge numerically for a known reason, such as operator fusion.

    Returns:
        If ``callable`` is ``nn.Module`` or ``forward`` of ``nn.Module``, ``trace`` returns
        a :class:`ScriptModule` object with a single ``forward`` method containing the traced code.
        The returned :class:`ScriptModule` will have the same set of sub-modules and parameters as the
        original ``nn.Module``.
        If ``callable`` is a standalone function, ``trace`` returns :class:`ScriptFunction`

    Example (tracing a function):

    .. testcode::

        import torch

        def foo(x, y):
            return 2 * x + y

        # Run `foo` with the provided inputs and record the tensor operations
        traced_foo = torch.jit.trace(foo, (torch.rand(3), torch.rand(3)))

        # `traced_foo` can now be run with the TorchScript interpreter or saved
        # and loaded in a Python-free environment

    Example (tracing an existing module)::

        import torch
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv = nn.Conv2d(1, 1, 3)

            def forward(self, x):
                return self.conv(x)

        n = Net()
        example_weight = torch.rand(1, 1, 3, 3)
        example_forward_input = torch.rand(1, 1, 3, 3)

        # Trace a specific method and construct `ScriptModule` with
        # a single `forward` method
        module = torch.jit.trace(n.forward, example_forward_input)

        # Trace a module (implicitly traces `forward`) and construct a
        # `ScriptModule` with a single `forward` method
        module = torch.jit.trace(n, example_forward_input)

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.jit.__init__.trace', 'trace(func, example_inputs, optimize=None, check_trace=True, check_inputs=None, check_tolerance=1e-05, _force_outplace=False, _module_class=None, _compilation_unit=_python_cu)', {'_enabled': _enabled, 'warnings': warnings, 'torch': torch, 'trace_module': trace_module, 'wrap_check_inputs': wrap_check_inputs, '_create_interpreter_name_lookup_fn': _create_interpreter_name_lookup_fn, '_qualified_name': _qualified_name, '_check_trace': _check_trace, 'func': func, 'example_inputs': example_inputs, 'optimize': optimize, 'check_trace': check_trace, 'check_inputs': check_inputs, 'check_tolerance': check_tolerance, '_force_outplace': _force_outplace, '_module_class': _module_class, '_compilation_unit': _compilation_unit, '_python_cu': _python_cu}, 1)
_trace_module_map = None

def trace_module(mod, inputs, optimize=None, check_trace=True, check_inputs=None, check_tolerance=1e-05, _force_outplace=False, _module_class=None, _compilation_unit=_python_cu):
    """
    Trace a module and return an executable :class:`ScriptModule` that will be optimized
    using just-in-time compilation. When a module is passed to :func:`torch.jit.trace <torch.jit.trace>`, only
    the ``forward`` method is run and traced. With ``trace_module``, you can specify a dictionary of
    method names to example inputs to trace (see the ``example_inputs``) argument below.

    See :func:`torch.jit.trace <torch.jit.trace>` for more information on tracing.

    Arguments:
        mod (torch.nn.Module):  A ``torch.nn.Module`` containing methods whose names are
                                specified in ``example_inputs``. The given methods will be compiled
                                as a part of a single `ScriptModule`.
        example_inputs (dict):  A dict containing sample inputs indexed by method names in ``mod``.
                                The inputs will be passed to methods whose names correspond to inputs'
                                keys while tracing.
                                ``{ 'forward' : example_forward_input, 'method2': example_method2_input}``
    Keyword arguments:
        check_trace (``bool``, optional): Check if the same inputs run through
                                      traced code produce the same outputs. Default: ``True``. You might want
                                      to disable this if, for example, your network contains non-
                                      deterministic ops or if you are sure that the network is correct despite
                                      a checker failure.

        check_inputs (list of dicts, optional): A list of dicts of input arguments that should be used
                                                 to check the trace against what is expected. Each tuple
                                                 is equivalent to a set of input arguments that would
                                                 be specified in ``example_inputs``. For best results, pass in a
                                                 set of checking inputs representative of the space of
                                                 shapes and types of inputs you expect the network to see.
                                                 If not specified, the original ``example_inputs`` are used for checking
        check_tolerance (float, optional): Floating-point comparison tolerance to use in the checker procedure.
                                           This can be used to relax the checker strictness in the event that
                                           results diverge numerically for a known reason, such as operator fusion.

    Returns:
        A :class:`ScriptModule` object with a single ``forward`` method containing the traced code.
        When ``func`` is a ``torch.nn.Module``, the returned :class:`ScriptModule` will have the same set of
        sub-modules and parameters as ``func``.

    Example (tracing a module with multiple methods)::

        import torch
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv = nn.Conv2d(1, 1, 3)

            def forward(self, x):
                return self.conv(x)

            def weighted_kernel_sum(self, weight):
                return weight * self.conv.weight


        n = Net()
        example_weight = torch.rand(1, 1, 3, 3)
        example_forward_input = torch.rand(1, 1, 3, 3)

        # Trace a specific method and construct `ScriptModule` with
        # a single `forward` method
        module = torch.jit.trace(n.forward, example_forward_input)

        # Trace a module (implicitly traces `forward`) and construct a
        # `ScriptModule` with a single `forward` method
        module = torch.jit.trace(n, example_forward_input)

        # Trace specific methods on a module (specified in `inputs`), constructs
        # a `ScriptModule` with `forward` and `weighted_kernel_sum` methods
        inputs = {'forward' : example_forward_input, 'weighted_kernel_sum' : example_weight}
        module = torch.jit.trace_module(n, inputs)

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.jit.__init__.trace_module', 'trace_module(mod, inputs, optimize=None, check_trace=True, check_inputs=None, check_tolerance=1e-05, _force_outplace=False, _module_class=None, _compilation_unit=_python_cu)', {'_enabled': _enabled, 'warnings': warnings, '_create_interpreter_name_lookup_fn': _create_interpreter_name_lookup_fn, 'torch': torch, 'make_module': make_module, 'make_tuple': make_tuple, '_check_trace': _check_trace, 'mod': mod, 'inputs': inputs, 'optimize': optimize, 'check_trace': check_trace, 'check_inputs': check_inputs, 'check_tolerance': check_tolerance, '_force_outplace': _force_outplace, '_module_class': _module_class, '_compilation_unit': _compilation_unit, '_python_cu': _python_cu}, 1)


class CompilationUnit(object):
    
    def __init__(self, lang=None, _frames_up=0):
        self._c = torch._C.CompilationUnit()
        if lang is not None:
            self.define(lang, _frames_up=_frames_up + 1)
    
    def define(self, lang, rcb=None, _frames_up=0):
        if not rcb:
            rcb = _jit_internal.createResolutionCallbackFromFrame(_frames_up + 1)
        self._c.define(lang, rcb)
    
    def __getattr__(self, attr):
        r = self._c.find_function(attr)
        if r is None:
            raise AttributeError("'CompilationUnit' has no attribute '{}'".format(attr))
        return r


def _try_get_dispatched_fn(fn):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.jit.__init__._try_get_dispatched_fn', '_try_get_dispatched_fn(fn)', {'_jit_internal': _jit_internal, 'fn': fn}, 1)

def _try_get_overloaded_fn(mod, field):
    return (mod._overloads.get(field, None) if isinstance(mod, ScriptModule) else None)


class ScriptWarning(Warning):
    pass


@contextlib.contextmanager
def _disable_emit_hooks():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.jit.__init__._disable_emit_hooks', '_disable_emit_hooks()', {'torch': torch, 'contextlib': contextlib}, 0)

def _is_new_style_class(cls):
    if hasattr(cls, '__class__'):
        return ('__dict__' in dir(cls) or hasattr(cls, '__slots__'))

def whichmodule(obj):
    """Find the module an object belong to."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.jit.__init__.whichmodule', 'whichmodule(obj)', {'sys': sys, '_getattribute': _getattribute, 'obj': obj}, 1)

def _compile_and_register_class(obj, rcb, qualified_name):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.jit.__init__._compile_and_register_class', '_compile_and_register_class(obj, rcb, qualified_name)', {'get_jit_class_def': get_jit_class_def, '_jit_script_class_compile': _jit_script_class_compile, '_add_script_class': _add_script_class, 'obj': obj, 'rcb': rcb, 'qualified_name': qualified_name}, 0)

def script(obj, optimize=None, _frames_up=0, _rcb=None):
    """
    Scripting a function or ``nn.Module`` will inspect the source code, compile
    it as TorchScript code using the TorchScript compiler, and return a :class:`ScriptModule` or
    :class:`ScriptFunction`. TorchScript itself is a subset of the Python language, so not all
    features in Python work, but we provide enough functionality to compute on
    tensors and do control-dependent operations. For a complete guide, see the
    :ref:`language-reference`.

    ``torch.jit.script`` can be used as a function for modules and functions, and as a decorator
    ``@torch.jit.script`` for :ref:`torchscript-classes` and functions.

    Arguments:
        obj (callable, class, or ``nn.Module``):  The ``nn.Module``, function, or class type to
                                                  compile.

    Returns:
        If ``obj`` is ``nn.Module``, ``script`` returns
        a :class:`ScriptModule` object. The returned :class:`ScriptModule` will
        have the same set of sub-modules and parameters as the
        original ``nn.Module``. If ``obj`` is a standalone function,
        a :class:`ScriptFunction` will be returned.

    **Scripting a function**
        The ``@torch.jit.script`` decorator will construct a :class:`ScriptFunction`
        by compiling the body of the function.

        Example (scripting a function):

        .. testcode::

            import torch

            @torch.jit.script
            def foo(x, y):
                if x.max() > y.max():
                    r = x
                else:
                    r = y
                return r

            print(type(foo))  # torch.jit.ScriptFuncion

            # See the compiled graph as Python code
            print(foo.code)

            # Call the function using the TorchScript interpreter
            foo(torch.ones(2, 2), torch.ones(2, 2))

        .. testoutput::
            :hide:

            ...

    **Scripting an nn.Module**
        Scripting an ``nn.Module`` by default will compile the ``forward`` method and recursively
        compile any methods, submodules, and functions called by ``forward``. If a ``nn.Module`` only uses
        features supported in TorchScript, no changes to the original module code should be necessary. ``script``
        will construct :class:`ScriptModule` that has copies of the attributes, parameters, and methods of
        the original module.

        Example (scripting a simple module with a Parameter):

        .. testcode::

            import torch

            class MyModule(torch.nn.Module):
                def __init__(self, N, M):
                    super(MyModule, self).__init__()
                    # This parameter will be copied to the new ScriptModule
                    self.weight = torch.nn.Parameter(torch.rand(N, M))

                    # When this submodule is used, it will be compiled
                    self.linear = torch.nn.Linear(N, M)

                def forward(self, input):
                    output = self.weight.mv(input)

                    # This calls the `forward` method of the `nn.Linear` module, which will
                    # cause the `self.linear` submodule to be compiled to a `ScriptModule` here
                    output = self.linear(output)
                    return output

            scripted_module = torch.jit.script(MyModule(2, 3))

        Example (scripting a module with traced submodules):

        .. testcode::

            import torch
            import torch.nn as nn
            import torch.nn.functional as F

            class MyModule(nn.Module):
                def __init__(self):
                    super(MyModule, self).__init__()
                    # torch.jit.trace produces a ScriptModule's conv1 and conv2
                    self.conv1 = torch.jit.trace(nn.Conv2d(1, 20, 5), torch.rand(1, 1, 16, 16))
                    self.conv2 = torch.jit.trace(nn.Conv2d(20, 20, 5), torch.rand(1, 20, 16, 16))

                def forward(self, input):
                  input = F.relu(self.conv1(input))
                  input = F.relu(self.conv2(input))
                  return input

            scripted_module = torch.jit.script(MyModule())

        To compile a method other than ``forward`` (and recursively compile anything it calls), add
        the :func:`@torch.jit.export <torch.jit.export>` decorator to the method. To opt out of compilation
        use :func:`@torch.jit.ignore <torch.jit.ignore>` or :func:`@torch.jit.unused <torch.jit.unused>`.

        Example (an exported and ignored method in a module)::

            import torch
            import torch.nn as nn

            class MyModule(nn.Module):
                def __init__(self):
                    super(MyModule, self).__init__()

                @torch.jit.export
                def some_entry_point(self, input):
                    return input + 10

                @torch.jit.ignore
                def python_only_fn(self, input):
                    # This function won't be compiled, so any
                    # Python APIs can be used
                    import pdb
                    pdb.set_trace()

                def forward(self, input):
                    if self.training:
                        self.python_only_fn(input)
                    return input * 99

            scripted_module = torch.jit.script(MyModule())
            print(scripted_module.some_entry_point(torch.randn(2, 2)))
            print(scripted_module(torch.randn(2, 2)))
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.jit.__init__.script', 'script(obj, optimize=None, _frames_up=0, _rcb=None)', {'_enabled': _enabled, 'warnings': warnings, 'ScriptModule': ScriptModule, 'torch': torch, '_qualified_name': _qualified_name, 'inspect': inspect, '_is_new_style_class': _is_new_style_class, '_jit_internal': _jit_internal, '_compile_and_register_class': _compile_and_register_class, '_check_directly_compile_overloaded': _check_directly_compile_overloaded, '_try_get_jit_cached_function': _try_get_jit_cached_function, 'get_jit_def': get_jit_def, 'get_default_args': get_default_args, '_set_jit_function_cache': _set_jit_function_cache, 'obj': obj, 'optimize': optimize, '_frames_up': _frames_up, '_rcb': _rcb}, 1)

def interface(obj):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.jit.__init__.interface', 'interface(obj)', {'inspect': inspect, '_is_new_style_class': _is_new_style_class, 'torch': torch, '_qualified_name': _qualified_name, '_jit_internal': _jit_internal, 'get_jit_class_def': get_jit_class_def, 'obj': obj}, 1)

def script_method(fn):
    if not _enabled:
        return fn
    _rcb = _jit_internal.createResolutionCallbackFromFrame(frames_up=2)
    ast = get_jit_def(fn, self_name='ScriptModule')
    return ScriptMethodStub(_rcb, ast, fn)


class OrderedDictWrapper(object):
    
    def __init__(self, _c):
        self._c = _c
    
    def keys(self):
        return [k for (k, v) in self.items()]
    
    def values(self):
        return [v for (k, v) in self.items()]
    
    def __len__(self):
        return len(self.values())
    
    def __delitem__(self, k):
        raise RuntimeError('cannot delete methods or parameters of a script module')
    
    def items(self):
        return self._c.items()
    
    def __setitem__(self, k, v):
        if k not in self:
            raise RuntimeError("Can't add a new parameter after ScriptModule construction. Tried to add '{}".format(k))
        self._c.setattr(k, v)
    
    def __contains__(self, k):
        return self._c.contains(k)
    
    def __getitem__(self, k):
        if k not in self:
            raise KeyError(k)
        return self._c.getattr(k)



class OrderedModuleDict(OrderedDictWrapper):
    
    def __init__(self, module, python_dict):
        super(OrderedModuleDict, self).__init__(torch._C.ModuleDict(module))
        self._python_modules = python_dict
    
    def items(self):
        r = self._python_modules.items()
        return r
    
    def __contains__(self, k):
        return k in self._python_modules
    
    def __setitem__(self, k, v):
        if isinstance(v, ScriptModule):
            self._c.setattr(k, v)
            self._python_modules[k] = v
        else:
            raise RuntimeError("Cannot re-assign modules in a ScriptModule with non-scripted module, tried to replace existing module '{}': {}".format(k, v))
    
    def __getitem__(self, k):
        return self._python_modules[k]



class ScriptMeta(type):
    
    def __init__(cls, name, bases, attrs):
        cls._methods = {}
        cls._constants_set = set(getattr(cls, '__constants__', ()))
        for base in reversed(bases):
            for (k, v) in getattr(base, '_methods', {}).items():
                cls._methods[k] = v
            base_constants = getattr(base, '_constants_set', set())
            cls._constants_set = cls._constants_set.union(base_constants)
        for (k, v) in sorted(attrs.items()):
            if isinstance(v, ScriptMethodStub):
                delattr(cls, k)
                cls._methods[v.original_method.__name__] = v
        if getattr(cls, '_disable_script_meta', False):
            return super(ScriptMeta, cls).__init__(name, bases, attrs)
        original_init = getattr(cls, '__init__', lambda self: None)
        
        @functools.wraps(original_init)
        def init_then_script(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            if type(self) == cls:
                
                def make_stubs(module):
                    cls = type(module)
                    return [v for (k, v) in sorted(cls._methods.items())]
                self.__dict__['_actual_script_module'] = torch.jit._recursive.create_script_module(self, make_stubs)
                concrete_type = self._actual_script_module._concrete_type
                for name in concrete_type.get_attributes():
                    delattr(self, name)
                for (name, _) in concrete_type.get_modules():
                    delattr(self, name)
                for name in ('_parameters', '_buffers', '_modules'):
                    delattr(self, name)
        cls.__init__ = init_then_script
        return super(ScriptMeta, cls).__init__(name, bases, attrs)

if _enabled:
    
    
    class _CachedForward(object):
        
        def __get__(self, obj, cls):
            return self.__getattr__('forward')
    
    
    
    class ScriptModule(with_metaclass(ScriptMeta, Module)):
        """
        ``ScriptModule``s wrap a C++ ``torch::jit::Module``. ``ScriptModule``s
        contain methods, attributes, parameters, and
        constants. These can be accessed the same as on a normal ``nn.Module``.
        """
        
        def __init__(self):
            super(ScriptModule, self).__init__()
        forward = _CachedForward()
        
        def __getattr__(self, attr):
            if '_actual_script_module' not in self.__dict__:
                return super(ScriptModule, self).__getattr__(attr)
            return getattr(self._actual_script_module, attr)
        
        def __setattr__(self, attr, value):
            if '_actual_script_module' not in self.__dict__:
                if isinstance(value, Attribute):
                    if not hasattr(self, '__annotations__'):
                        self.__annotations__ = {}
                    self.__annotations__[attr] = value.type
                    value = value.value
                return super(ScriptModule, self).__setattr__(attr, value)
            setattr(self._actual_script_module, attr, value)
        
        def define(self, src):
            if '_actual_script_module' in self.__dict__:
                return self._actual_script_module.define(src)
            rcb = _jit_internal.createResolutionCallbackFromFrame(frames_up=1)
            ast = torch._C._parse_source_def(src)
            self._methods[ast.name().name] = ScriptMethodStub(rcb, ast, None)
    
    
    
    class RecursiveScriptModule(ScriptModule):
        """
        The core data structure in TorchScript is the ``ScriptModule``. It is an
        analogue of torch's ``nn.Module`` and represents an entire model as a tree of
        submodules. Like normal modules, each individual module in a ``ScriptModule`` can
        have submodules, parameters, and methods. In ``nn.Module``\s methods are implemented
        as Python functions, but in ``ScriptModule``\s methods are implemented as
        TorchScript functions,  a statically-typed subset of Python that contains all
        of PyTorch's built-in Tensor operations. This difference allows your
        ``ScriptModule``\s code to run without the need for a Python interpreter.

        ``ScriptModule``\s should not be created manually, instead use
        either :func:`tracing <torch.jit.trace>` or :func:`scripting <torch.jit.script>`.
        Tracing and scripting can be applied incrementally and :ref:`composed as necessary <Types>`.

        * Tracing records the tensor operations as executed with a set of example inputs and uses these
          operations to construct a computation graph. You can use the full dynamic behavior of Python with tracing,
          but values other than Tensors and control flow aren't captured in the graph.

        * Scripting inspects the Python code of the model
          and compiles it to TorchScript. Scripting allows the use of many `types`_ of values and supports dynamic control flow.
          Many, but not all features of Python are supported by the compiler, so changes to the source code may be necessary.
        """
        _disable_script_meta = True
        
        def __init__(self, cpp_module):
            self.__dict__['_initializing'] = True
            self._c = cpp_module
            super(RecursiveScriptModule, self).__init__()
            delattr(self, 'training')
        
        @staticmethod
        def _construct(cpp_module, init_fn):
            """
            Construct a RecursiveScriptModule that's ready for use. PyTorch
            code should use this to construct a RecursiveScriptModule instead
            of instead of calling `__init__` directly, as it makes sure the
            object is properly finalized (and in the future we may take
            control of how the RecursiveScriptModule instance is created).

            Arguments:
                cpp_module:  The C++ Module that will hold the actual state of
                             this RecursiveScriptModule instance.
                init_fn:  Lambda that initializes the RecursiveScriptModule passed to it.
            """
            script_module = RecursiveScriptModule(cpp_module)
            init_fn(script_module)
            script_module._parameters = OrderedDictWrapper(torch._C.ParameterDict(script_module._c))
            script_module._buffers = OrderedDictWrapper(torch._C.BufferDict(script_module._c))
            script_module._modules = OrderedModuleDict(script_module._c, script_module._modules)
            script_module._initializing = False
            return script_module
        
        @property
        def graph(self):
            """
            Returns a string representation of the internal graph for the
            ``forward`` method. See `Interpreting Graphs`_ for details.
            """
            return self.forward.graph
        
        @property
        def inlined_graph(self):
            """
            Returns a string representation of the internal graph for the
            ``forward`` method. This graph will be preprocessed to inline all function and method calls.
            See `Interpreting Graphs`_ for details.
            """
            return self.forward.inlined_graph
        
        @property
        def code(self):
            """
            Returns a pretty-printed representation (as valid Python syntax) of
            the internal graph for the ``forward`` method. See `Inspecting Code`_
            for details.
            """
            return self.forward.code
        
        def save(self, *args, **kwargs):
            """
            save(f, _extra_files=ExtraFilesMap{})

            See :func:`torch.jit.save <torch.jit.save>` for details.
            """
            return self._c.save(*args, **kwargs)
        
        def _save_for_lite_interpreter(self, *args, **kwargs):
            """
            _save_for_lite_interpreter(f)

            Add (or update) the bytecode session to the script model. The updated model is used
            in lite interpreter for mobile applications.

            Arguments:
                f: a string containing a file name.
                _extra_files: Map from filename to contents which will be stored as part of 'f'.

            """
            return self._c._save_for_mobile(*args, **kwargs)
        
        def save_to_buffer(self, *args, **kwargs):
            return self._c.save_to_buffer(*args, **kwargs)
        
        def get_debug_state(self, *args, **kwargs):
            return self._c.get_debug_state()
        
        def extra_repr(self):
            return 'original_name={}'.format(self.original_name)
        
        def graph_for(self, *args, **kwargs):
            return self.forward.graph_for(*args, **kwargs)
        
        @property
        def original_name(self):
            if type(self) == str(self._c._type().name()):
                return ''
            return str(self._c._type().name())
        
        def define(self, src):
            rcb = _jit_internal.createResolutionCallbackFromFrame(frames_up=1)
            self._c._define(self._concrete_type, src, rcb)
        
        def __getattr__(self, attr):
            if '_initializing' not in self.__dict__:
                raise RuntimeError("ScriptModule has not been initialized, did you forget to call super's init?")
            if self._initializing:
                return super(RecursiveScriptModule, self).__getattr__(attr)
            if attr in self._modules:
                return self._modules[attr]
            elif self._c.hasattr(attr):
                return self._c.getattr(attr)
            elif self._c._has_method(attr):
                script_method = self._c._get_method(attr)
                self.__dict__[attr] = script_method
                return script_method
            return super(RecursiveScriptModule, self).__getattr__(attr)
        
        def __setattr__(self, attr, value):
            if self._initializing:
                return super(RecursiveScriptModule, self).__setattr__(attr, value)
            if attr in self._modules:
                self._modules[attr] = value
            elif self._c.hasattr(attr):
                self._c.setattr(attr, value)
            elif (hasattr(self, '_concrete_type') and attr in self._concrete_type.get_constants().keys()):
                raise AttributeError("Cannot mutate TorchScript constant value: '{}'. Value: '{}'".format(attr, value))
            else:
                return super(RecursiveScriptModule, self).__setattr__(attr, value)
        
        def copy(self):
            return torch.jit._recursive.wrap_cpp_module(self._c._clone())
        
        def copy_instance(self):
            return torch.jit._recursive.wrap_cpp_module(self._c._clone_instance())
        
        def __getstate__(self):
            raise pickle.PickleError('ScriptModules cannot be deepcopied using copy.deepcopy or saved using torch.save. ' + 'Mixed serialization of script and non-script modules is not supported. ' + 'For purely script modules use my_script_module.save(<filename>) instead.')
        
        def forward_magic_method(self, method_name, *args, **kwargs):
            self_method = getattr(self, method_name)
            if getattr(self_method, '__func__', None) == getattr(RecursiveScriptModule, method_name):
                raise NotImplementedError()
            return self_method(*args, **kwargs)
        
        def __iter__(self):
            return self.forward_magic_method('__iter__')
        
        def __getitem__(self, idx):
            return self.forward_magic_method('__getitem__', idx)
        
        def __len__(self):
            return self.forward_magic_method('__len__')
        
        def __contains__(self, key):
            return self.forward_magic_method('__contains__', key)
        
        def __dir__(self):
            self_method = self.__dir__
            if self_method.__func__ == get_function_from_type(RecursiveScriptModule, '__dir__'):
                return super(RecursiveScriptModule, self).__dir__()
            return self_method()
        
        def __bool__(self):
            self_method = self.__bool__
            if self_method.__func__ == get_function_from_type(RecursiveScriptModule, '__bool__'):
                return True
            return self_method()
    
    for (name, item) in RecursiveScriptModule.__dict__.items():
        if (not callable(item) and not isinstance(item, property)):
            continue
        if (name.startswith('__') or hasattr(ScriptModule, name)):
            continue
        setattr(ScriptModule, name, item)
    
    def _get_methods(cls):
        import inspect
        return inspect.getmembers(cls, predicate=lambda x: (inspect.isfunction(x) or inspect.ismethod(x)))
    _compiled_methods_whitelist = {'forward', 'register_buffer', 'register_parameter', 'add_module', '_apply', 'apply', 'cuda', 'cpu', 'to', 'type', 'float', 'double', 'half', 'state_dict', '_save_to_state_dict', 'load_state_dict', '_load_from_state_dict', '_named_members', 'parameters', 'named_parameters', 'buffers', 'named_buffers', 'children', 'named_children', 'modules', 'named_modules', 'zero_grad', 'share_memory', '_get_name', 'extra_repr', '_slow_forward', '_tracing_name', 'eval', 'train'}
    
    def _make_fail(name):
        
        def fail(self, *args, **kwargs):
            raise RuntimeError(name + ' is not supported on ScriptModules')
        return fail
    for (name, method) in _get_methods(torch.nn.Module):
        if name.startswith('__'):
            continue
        if (name not in RecursiveScriptModule.__dict__ and name not in _compiled_methods_whitelist):
            setattr(RecursiveScriptModule, method.__name__, _make_fail(name))
else:
    
    
    class ScriptModule(torch.nn.Module):
        
        def __init__(self):
            super(ScriptModule, self).__init__()
    


class TracedModule(ScriptModule):
    _disable_script_meta = True
    
    def __init__(self, orig, id_set=None, _compilation_unit=None):
        super(TracedModule, self).__init__()
        assert isinstance(orig, torch.nn.Module)
        id_set = set()
        
        
        class QualnameWrapper(torch.nn.Module):
            pass
        
        QualnameWrapper._jit_override_qualname = torch._jit_internal._qualified_name(type(orig))
        tmp_module = QualnameWrapper()
        
        def check_unique(param):
            if param in id_set:
                raise ValueError("TracedModules don't support parameter sharing between modules")
            id_set.add(param)
        tmp_module.training = orig.training
        for (name, param) in orig._parameters.items():
            if param is not None:
                tmp_module._parameters[name] = param
                check_unique(param)
        for (name, buf) in orig._buffers.items():
            if buf is not None:
                tmp_module._buffers[name] = buf
                check_unique(buf)
        for (name, val) in orig.__dict__.items():
            if (torch._C._jit_is_script_object(val) and name not in orig._parameters and name not in orig._buffers):
                setattr(tmp_module, name, val)
        if orig._backward_hooks:
            raise ValueError("Modules that have backward hooks assigned can't be compiled: " + str(orig))
        for (name, submodule) in orig._modules.items():
            tmp_module._modules[name] = make_module(submodule, TracedModule, _compilation_unit=None)
        script_module = torch.jit._recursive.create_script_module(tmp_module, lambda module: (), share_types=False)
        self.__dict__['_name'] = type(orig).__name__
        self.__dict__['_actual_script_module'] = script_module
        for name in ('_parameters', '_buffers', '_modules'):
            delattr(self, name)
    
    def forward(self, *args, **kwargs):
        raise RuntimeError('Trace submodules cannot be called.')
    
    def __getattr__(self, attr):
        if '_actual_script_module' not in self.__dict__:
            return super(TracedModule, self).__getattr__(attr)
        return getattr(self._actual_script_module, attr)
    
    def __setattr__(self, attr, value):
        if '_actual_script_module' not in self.__dict__:
            return super(TracedModule, self).__setattr__(attr, value)
        setattr(self._actual_script_module, attr, value)
    
    def _get_name(self):
        return self._name
    
    def extra_repr(self):
        return 'original_name={}'.format(self._name)

if _enabled:
    
    
    class TopLevelTracedModule(TracedModule):
        forward = _CachedForward()
    

def is_scripting():
    """
    Function that returns True when in compilation and False otherwise. This
    is useful especially with the @unused decorator to leave code in your
    model that is not yet TorchScript compatible.
    .. testcode::

        import torch

        @torch.jit.unused
        def unsupported_linear_op(x):
            return x

        def linear(x):
           if not torch.jit.is_scripting():
              return torch.linear(x)
           else:
              return unsupported_linear_op(x)
    """
    return False

def _unwrap_optional(x):
    assert x is not None, 'Unwrapping null optional'
    return x
_register_builtin(_unwrap_optional, 'aten::_unwrap_optional')
_register_builtin(_wait, 'aten::wait')
_register_builtin(is_scripting, 'aten::is_scripting')
_jit_caching_layer = weakref.WeakKeyDictionary()
_jit_function_overload_caching = weakref.WeakKeyDictionary()

def _try_get_jit_cached_overloads(key):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.jit.__init__._try_get_jit_cached_overloads', '_try_get_jit_cached_overloads(key)', {'_jit_function_overload_caching': _jit_function_overload_caching, '_python_cu': _python_cu, 'key': key}, 1)

def _set_jit_overload_cache(key, compiled_fns):
    _jit_function_overload_caching[key] = [fn.qualified_name for fn in compiled_fns]

def _try_get_jit_cached_function(key):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.jit.__init__._try_get_jit_cached_function', '_try_get_jit_cached_function(key)', {'_jit_caching_layer': _jit_caching_layer, '_python_cu': _python_cu, 'key': key}, 1)

def _set_jit_function_cache(key, value):
    assert isinstance(value, torch.jit.ScriptFunction)
    _jit_caching_layer[key] = value.qualified_name
_script_classes = {}

def _add_script_class(cls, name):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.jit.__init__._add_script_class', '_add_script_class(cls, name)', {'_script_classes': _script_classes, 'cls': cls, 'name': name}, 0)

def _get_script_class(name):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.jit.__init__._get_script_class', '_get_script_class(name)', {'_script_classes': _script_classes, 'name': name}, 1)

def _check_overload_defaults(impl_defaults, overload_defaults, loc):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.jit.__init__._check_overload_defaults', '_check_overload_defaults(impl_defaults, overload_defaults, loc)', {'torch': torch, 'impl_defaults': impl_defaults, 'overload_defaults': overload_defaults, 'loc': loc}, 0)

def _compile_function_with_overload(overload_fn, qual_name, impl_fn):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.jit.__init__._compile_function_with_overload', '_compile_function_with_overload(overload_fn, qual_name, impl_fn)', {'torch': torch, 'inspect': inspect, 'get_default_args': get_default_args, '_jit_internal': _jit_internal, '_check_overload_defaults': _check_overload_defaults, 'overload_fn': overload_fn, 'qual_name': qual_name, 'impl_fn': impl_fn}, 1)

def _get_overloads(obj):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.jit.__init__._get_overloads', '_get_overloads(obj)', {'_try_get_jit_cached_overloads': _try_get_jit_cached_overloads, '_qualified_name': _qualified_name, '_jit_internal': _jit_internal, '_compile_function_with_overload': _compile_function_with_overload, '_set_jit_overload_cache': _set_jit_overload_cache, 'obj': obj}, 1)

def _check_directly_compile_overloaded(obj):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.jit.__init__._check_directly_compile_overloaded', '_check_directly_compile_overloaded(obj)', {'_qualified_name': _qualified_name, '_jit_internal': _jit_internal, '_try_get_jit_cached_overloads': _try_get_jit_cached_overloads, 'obj': obj}, 0)
Error = torch._C.JITException
set_module(Error, 'torch.jit')
Error.__name__ = 'Error'
Error.__qualname__ = 'Error'

def _get_named_tuple_properties(obj):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.jit.__init__._get_named_tuple_properties', '_get_named_tuple_properties(obj)', {'torch': torch, '_jit_internal': _jit_internal, 'obj': obj}, 3)

def _create_named_tuple(t, unqual_name, field_names):
    TupleType = collections.namedtuple(unqual_name, field_names)
    return TupleType(*t)


class _disable_tracing(object):
    
    def __enter__(self):
        self.state = torch._C._get_tracing_state()
        torch._C._set_tracing_state(None)
    
    def __exit__(self, *args):
        torch._C._set_tracing_state(self.state)
        self.state = None


def annotate(the_type, the_value):
    return the_value
last_executed_optimized_graph = torch._C._last_executed_optimized_graph

def _graph_for(self, *args, **kwargs):
    self(*args, **kwargs)
    return last_executed_optimized_graph()
torch._C.ScriptMethod.graph_for = _graph_for
torch._C.ScriptFunction.graph_for = _graph_for
ScriptFunction = torch._C.ScriptFunction
ScriptFunction.__doc__ = '\nFunctionally equivalent to a :class:`ScriptModule`, but represents a single\nfunction and does not have any attributes or Parameters.\n'
set_module(ScriptFunction, 'torch.jit')
if not torch._C._jit_init():
    raise RuntimeError('JIT initialization failed')

