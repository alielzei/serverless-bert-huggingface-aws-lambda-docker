import inspect
import torch
import collections
import textwrap
import functools
import warnings
import torch._jit_internal as _jit_internal
from torch.jit.frontend import get_default_args
from torch.jit._builtins import _find_builtin
from torch.nn import Module
from torch._six import get_function_from_type, bind_method
ScriptMethodStub = collections.namedtuple('ScriptMethodStub', ('resolution_callback', 'def_', 'original_method'))
blacklist = ['_version', '_parameters', '_buffers', '_modules', '_initializing', '_backward_hooks', '_forward_hooks', '_forward_pre_hooks', '_state_dict_hooks', '_load_state_dict_pre_hooks', 'dump_patches']

def make_stub(func):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.jit._recursive.make_stub', 'make_stub(func)', {'_jit_internal': _jit_internal, 'torch': torch, 'ScriptMethodStub': ScriptMethodStub, 'func': func}, 1)

def make_stub_from_method(nn_module, method):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.jit._recursive.make_stub_from_method', 'make_stub_from_method(nn_module, method)', {'get_function_from_type': get_function_from_type, 'ScriptMethodStub': ScriptMethodStub, 'make_stub': make_stub, 'nn_module': nn_module, 'method': method}, 1)
_constant_types = (bool, float, int, str, type(None), torch.device, torch.layout, torch.dtype)

def _get_valid_constant(attr, v):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.jit._recursive._get_valid_constant', '_get_valid_constant(attr, v)', {'_constant_types': _constant_types, '_get_valid_constant': _get_valid_constant, 'textwrap': textwrap, 'attr': attr, 'v': v}, 1)


class SourceContext(torch._C._jit_tree_views.SourceRangeFactory):
    
    def __init__(self, source, filename, file_lineno, leading_whitespace_len):
        super(SourceContext, self).__init__(source, filename, file_lineno, leading_whitespace_len)


def infer_concrete_type_builder(nn_module):
    """
    Build a ConcreteModuleTypeBuilder from an nn.Module. This
    ConcreteModuleType doesn't have a JIT type associated with it yet, it
    must be filled in by the caller.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.jit._recursive.infer_concrete_type_builder', 'infer_concrete_type_builder(nn_module)', {'torch': torch, '_jit_internal': _jit_internal, 'concrete_type_store': concrete_type_store, 'warnings': warnings, '_get_valid_constant': _get_valid_constant, 'get_overload_name_mapping': get_overload_name_mapping, 'get_overload_annotations': get_overload_annotations, 'blacklist': blacklist, 'inspect': inspect, '_find_builtin': _find_builtin, 'nn_module': nn_module}, 1)


class ConcreteTypeStore(object):
    
    def __init__(self):
        self.type_store = {}
        self.methods_compiled = set()
    
    def get_or_create_concrete_type(self, nn_module):
        """
        Infer a ConcreteType from this `nn.Module` instance. Underlying JIT
        types are re-used if possible.
        """
        assert isinstance(nn_module, Module)
        if (isinstance(nn_module, torch.jit.ScriptModule) and hasattr(nn_module, '_concrete_type')):
            return nn_module._concrete_type
        concrete_type_builder = infer_concrete_type_builder(nn_module)
        nn_module_type = type(nn_module)
        if nn_module_type not in self.type_store:
            self.type_store[nn_module_type] = []
        known_types = self.type_store[nn_module_type]
        for known_type in known_types:
            if known_type.equals(concrete_type_builder):
                return known_type
        concrete_type = concrete_type_builder.build()
        self.type_store[nn_module_type].append(concrete_type)
        return concrete_type

concrete_type_store = ConcreteTypeStore()

def create_methods_from_stubs(concrete_type, stubs):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.jit._recursive.create_methods_from_stubs', 'create_methods_from_stubs(concrete_type, stubs)', {'get_default_args': get_default_args, 'concrete_type': concrete_type, 'stubs': stubs}, 0)

def create_script_module(nn_module, stubs_fn, share_types=True):
    """
    Creates a new ScriptModule from an nn.Module

    Arguments:
        nn_module:  The original Python nn.Module that we are creating a ScriptModule for.
        stubs_fn:  Lambda that takes an nn.Module and generates a list of ScriptMethodStubs to compile.
        share_types:  Whether to share underlying JIT types between modules (if possible).
            NOTE: Only set to False this when we cannot guarantee type sharing will work
                correctly. This only happens today for traced modules, where the same
                module can produce different traced methods depending on the inputs.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.jit._recursive.create_script_module', 'create_script_module(nn_module, stubs_fn, share_types=True)', {'torch': torch, 'check_module_initialized': check_module_initialized, 'concrete_type_store': concrete_type_store, 'infer_concrete_type_builder': infer_concrete_type_builder, 'create_script_module_impl': create_script_module_impl, 'nn_module': nn_module, 'stubs_fn': stubs_fn, 'share_types': share_types}, 1)

def create_script_module_impl(nn_module, concrete_type, stubs_fn):
    """
    Convert an nn.Module to a RecursiveScriptModule.

    Arguments:
        nn_module:  The original Python nn.Module that we are creating a ScriptModule for.
        concrete_type:  The fully initialized ConcreteType of the module.
        stubs_fn:  Lambda that takes an nn.Module and generates a list of ScriptMethodStubs to compile.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.jit._recursive.create_script_module_impl', 'create_script_module_impl(nn_module, concrete_type, stubs_fn)', {'torch': torch, 'Module': Module, 'interface_script': interface_script, 'create_script_module_impl': create_script_module_impl, 'infer_methods_to_compile': infer_methods_to_compile, 'inspect': inspect, '_jit_internal': _jit_internal, 'concrete_type_store': concrete_type_store, 'create_methods_from_stubs': create_methods_from_stubs, 'functools': functools, 'add_python_attr_to_scripted_model': add_python_attr_to_scripted_model, 'nn_module': nn_module, 'concrete_type': concrete_type, 'stubs_fn': stubs_fn}, 1)

def script_model_defines_attr(script_model, attr):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.jit._recursive.script_model_defines_attr', 'script_model_defines_attr(script_model, attr)', {'get_function_from_type': get_function_from_type, 'torch': torch, 'script_model': script_model, 'attr': attr}, 1)

def add_python_attr_to_scripted_model(script_model, orig, attr):
    if (hasattr(orig, attr) and script_model_defines_attr(script_model, attr)):
        setattr(script_model, attr, getattr(orig, attr))

def get_overload_annotations(mod):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.jit._recursive.get_overload_annotations', 'get_overload_annotations(mod)', {'_jit_internal': _jit_internal, 'mod': mod}, 1)

def get_overload_name_mapping(overload_info):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.jit._recursive.get_overload_name_mapping', 'get_overload_name_mapping(overload_info)', {'overload_info': overload_info}, 1)

def _check_no_signature(func):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.jit._recursive._check_no_signature', '_check_no_signature(func)', {'torch': torch, 'inspect': inspect, 'func': func}, 0)

def make_stubs_for_overloads(overload_info):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.jit._recursive.make_stubs_for_overloads', 'make_stubs_for_overloads(overload_info)', {'torch': torch, '_check_no_signature': _check_no_signature, '_jit_internal': _jit_internal, 'ScriptMethodStub': ScriptMethodStub, 'overload_info': overload_info}, 1)

def check_module_initialized(mod):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.jit._recursive.check_module_initialized', 'check_module_initialized(mod)', {'torch': torch, 'mod': mod}, 0)

def infer_methods_to_compile(nn_module):
    """
    Implements the default rules for which methods should act as starting
    points for compilation (TODO add a link when the rules are published).
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.jit._recursive.infer_methods_to_compile', 'infer_methods_to_compile(nn_module)', {'check_module_initialized': check_module_initialized, '_jit_internal': _jit_internal, 'get_function_from_type': get_function_from_type, 'torch': torch, 'get_overload_annotations': get_overload_annotations, 'get_overload_name_mapping': get_overload_name_mapping, 'make_stubs_for_overloads': make_stubs_for_overloads, 'make_stub_from_method': make_stub_from_method, 'nn_module': nn_module}, 1)

def interface_script(mod_interface, nn_module):
    """
    Makes a ScriptModule from an nn.Module, using the interface methods rule for
    determining which methods to compile.

    Arguments:
        mod_interface: the interface type that the module have
        nn_module:  The original Python nn.Module that we are creating a ScriptModule for.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.jit._recursive.interface_script', 'interface_script(mod_interface, nn_module)', {'torch': torch, 'check_module_initialized': check_module_initialized, 'make_stub_from_method': make_stub_from_method, 'create_script_module': create_script_module, 'mod_interface': mod_interface, 'nn_module': nn_module}, 1)

def try_compile_fn(fn, loc):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.jit._recursive.try_compile_fn', 'try_compile_fn(fn, loc)', {'_jit_internal': _jit_internal, 'torch': torch, 'inspect': inspect, 'fn': fn, 'loc': loc}, 1)

def wrap_cpp_module(cpp_module):
    """
    Wrap this torch._C.ScriptModule in a Python ScriptModule, recursively for all submodules
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.jit._recursive.wrap_cpp_module', 'wrap_cpp_module(cpp_module)', {'torch': torch, 'wrap_cpp_module': wrap_cpp_module, 'cpp_module': cpp_module}, 1)

def compile_unbound_method(concrete_type, fn):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.jit._recursive.compile_unbound_method', 'compile_unbound_method(concrete_type, fn)', {'_jit_internal': _jit_internal, 'make_stub': make_stub, 'torch': torch, 'create_methods_from_stubs': create_methods_from_stubs, 'concrete_type': concrete_type, 'fn': fn}, 1)

def lazy_bind(concrete_type, unbound_method):
    """
    Returns a function that lazily binds `unbound_method` to a provided
    Module IValue, then invokes the method. We do this so that any Python
    shenanigans that will poison type sharing are impossible at compile
    time.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.jit._recursive.lazy_bind', 'lazy_bind(concrete_type, unbound_method)', {'_jit_internal': _jit_internal, 'torch': torch, 'bind_method': bind_method, 'concrete_type': concrete_type, 'unbound_method': unbound_method}, 1)

