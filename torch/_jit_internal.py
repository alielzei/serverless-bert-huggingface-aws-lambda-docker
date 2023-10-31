"""
The weak_script annotation needs to be here instead of inside torch/jit/ so it
can be used in other places in torch/ (namely torch.nn) without running into
circular dependency problems
"""

import inspect
import weakref
import warnings
import torch
from torch._six import builtins
from torch._utils_internal import get_source_lines_and_file
boolean_dispatched = weakref.WeakKeyDictionary()

def createResolutionCallbackFromEnv(lookup_base):
    """
    Creates a resolution callback that will look up qualified names in an
    environment, starting with `lookup_base` for the base of any qualified
    names, then proceeding down the lookup chain with the resolved object.

    You should not use this directly, it should only be used from the other
    createResolutionCallbackFrom* functions.
    """
    
    def env(qualified_name, module):
        if '.' in qualified_name:
            parts = qualified_name.split('.')
            base = parts[0]
            remainding_pieces = '.'.join(parts[1:])
            module_value = getattr(module, base)
            return env(remainding_pieces, module_value)
        else:
            return getattr(module, qualified_name)
    return lambda key: env(key, lookup_base)

def createResolutionCallbackFromFrame(frames_up=0):
    """
    Creates a function which, given a string variable name,
    returns the value of the variable in the scope of the caller of
    the function which called createResolutionCallbackFromFrame (by default).

    This is used to enable access in-scope Python variables inside
    TorchScript fragments.

    frames_up is number of additional frames to go up on the stack.
    The default value is 0, which correspond to the frame of the caller
    of createResolutionCallbackFromFrame. Also for example, if frames_up is set
    to 1, then the frame of the caller's caller of createResolutionCallbackFromFrame
    will be taken.

    For example, the following program prints 2::

        def bar():
            cb = createResolutionCallbackFromFrame(1)
            print(cb("foo"))

        def baz():
            foo = 2
            bar()

        baz()
    """
    frame = inspect.currentframe()
    i = 0
    while i < frames_up + 1:
        frame = frame.f_back
        i += 1
    f_locals = frame.f_locals
    f_globals = frame.f_globals
    
    
    class env(object):
        
        def __getattr__(self, key):
            if key in f_locals:
                return f_locals[key]
            elif key in f_globals:
                return f_globals[key]
    
    return createResolutionCallbackFromEnv(env())

def get_closure(fn):
    """
    Get a dictionary of closed over variables from a function
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch._jit_internal.get_closure', 'get_closure(fn)', {'fn': fn}, 1)

def createResolutionCallbackFromClosure(fn):
    """
    Create a resolutionCallback by introspecting the function instead of
    looking up the stack for the enclosing scope
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch._jit_internal.createResolutionCallbackFromClosure', 'createResolutionCallbackFromClosure(fn)', {'get_closure': get_closure, 'builtins': builtins, 'createResolutionCallbackFromEnv': createResolutionCallbackFromEnv, 'fn': fn}, 1)

def can_compile_class(cls):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch._jit_internal.can_compile_class', 'can_compile_class(cls)', {'is_ignored_fn': is_ignored_fn, 'inspect': inspect, 'cls': cls}, 1)

def createResolutionCallbackForClassMethods(cls):
    """
    This looks at all the methods defined in a class and pulls their closed-over
    variables into a dictionary and uses that to resolve variables.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch._jit_internal.createResolutionCallbackForClassMethods', 'createResolutionCallbackForClassMethods(cls)', {'inspect': inspect, 'get_closure': get_closure, 'cls': cls}, 1)

def boolean_dispatch(arg_name, arg_index, default, if_true, if_false, module_name, func_name):
    """
    Dispatches to either of 2 script functions based on a boolean argument.
    In TorchScript, the boolean argument must be constant so that the correct
    function to use can be determined at compile time.
    """
    
    def fn(*args, **kwargs):
        dispatch_flag = False
        if arg_name in kwargs:
            dispatch_flag = kwargs[arg_name]
        elif arg_index < len(args):
            dispatch_flag = args[arg_index]
        if dispatch_flag:
            return if_true(*args, **kwargs)
        else:
            return if_false(*args, **kwargs)
    if (if_true.__doc__ is None and if_false.__doc__ is not None):
        doc = if_false.__doc__
        if_true.__doc__ = doc
    elif (if_false.__doc__ is None and if_true.__doc__ is not None):
        doc = if_true.__doc__
        if_false.__doc__ = doc
    elif (if_false.__doc__ is None and if_true.__doc__ is None):
        doc = None
    else:
        raise RuntimeError('only one function can have a docstring')
    fn.__doc__ = doc
    if module_name is not None:
        fn.__module__ = module_name
    if func_name is not None:
        fn.__name__ = func_name
    boolean_dispatched[fn] = {'if_true': if_true, 'if_false': if_false, 'index': arg_index, 'default': default, 'arg_name': arg_name}
    return fn


class FunctionModifiers(object):
    """
    Used to denote the behavior of a function in TorchScript. See export() and
    ignore() for details.
    """
    UNUSED = 'unused (ignored and replaced with raising of an exception)'
    IGNORE = "ignore (leave as a call to Python, cannot be torch.jit.save'd)"
    EXPORT = 'export (compile this function even if nothing calls it)'
    DEFAULT = 'default (compile if called from a exported function / forward)'
    COPY_TO_SCRIPT_WRAPPER = 'if this method is not scripted, copy the python method onto the scripted model'


def export(fn):
    """
    This decorator indicates that a method on an ``nn.Module`` is used as an entry point into a
    :class:`ScriptModule` and should be compiled.

    ``forward`` implicitly is assumed to be an entry point, so it does not need this decorator.
    Functions and methods called from ``forward`` are compiled as they are seen
    by the compiler, so they do not need this decorator either.

    Example (using ``@torch.jit.export`` on a method):

    .. testcode::

        import torch
        import torch.nn as nn

        class MyModule(nn.Module):
            def implicitly_compiled_method(self, x):
                return x + 99

            # `forward` is implicitly decorated with `@torch.jit.export`,
            # so adding it here would have no effect
            def forward(self, x):
                return x + 10

            @torch.jit.export
            def another_forward(self, x):
                # When the compiler sees this call, it will compile
                # `implicitly_compiled_method`
                return self.implicitly_compiled_method(x)

            def unused_method(self, x):
                return x - 20

        # `m` will contain compiled methods:
        #     `forward`
        #     `another_forward`
        #     `implicitly_compiled_method`
        # `unused_method` will not be compiled since it was not called from
        # any compiled methods and wasn't decorated with `@torch.jit.export`
        m = torch.jit.script(MyModule())
    """
    fn._torchscript_modifier = FunctionModifiers.EXPORT
    return fn

def unused(fn):
    """
    This decorator indicates to the compiler that a function or method should
    be ignored and replaced with the raising of an exception. This allows you
    to leave code in your model that is not yet TorchScript compatible and still
    export your model.

        Example (using ``@torch.jit.unused`` on a method)::

            import torch
            import torch.nn as nn

            class MyModule(nn.Module):
                def __init__(self, use_memory_efficent):
                    super(MyModule, self).__init__()
                    self.use_memory_efficent = use_memory_efficent

                @torch.jit.unused
                def memory_efficient(self, x):
                    import pdb
                    pdb.set_trace()
                    return x + 10

                def forward(self, x):
                    # Use not-yet-scriptable memory efficient mode
                    if self.use_memory_efficient:
                        return self.memory_efficient(x)
                    else:
                        return x + 10

            m = torch.jit.script(MyModule(use_memory_efficent=False))
            m.save("m.pt")

            m = torch.jit.script(MyModule(use_memory_efficient=True))
            # exception raised
            m(torch.rand(100))
    """
    fn._torchscript_modifier = FunctionModifiers.UNUSED
    return fn

def ignore(drop=False, **kwargs):
    """
    This decorator indicates to the compiler that a function or method should
    be ignored and left as a Python function. This allows you to leave code in
    your model that is not yet TorchScript compatible. If called from TorchScript,
    ignored functions will dispatch the call to the Python interpreter. Models with ignored
    functions cannot be exported; use :func:`@torch.jit.unused <torch.jit.unused>` instead.

    Example (using ``@torch.jit.ignore`` on a method)::

        import torch
        import torch.nn as nn

        class MyModule(nn.Module):
            @torch.jit.ignore
            def debugger(self, x):
                import pdb
                pdb.set_trace()

            def forward(self, x):
                x += 10
                # The compiler would normally try to compile `debugger`,
                # but since it is `@ignore`d, it will be left as a call
                # to Python
                self.debugger(x)
                return x

        m = torch.jit.script(MyModule())

        # Error! The call `debugger` cannot be saved since it calls into Python
        m.save("m.pt")

    Example (using ``@torch.jit.ignore(drop=True)`` on a method):

    .. testcode::

        import torch
        import torch.nn as nn

        class MyModule(nn.Module):
            @torch.jit.ignore(drop=True)
            def training_method(self, x):
                import pdb
                pdb.set_trace()

            def forward(self, x):
                if self.training:
                    self.training_method(x)
                return x

        m = torch.jit.script(MyModule())

        # This is OK since `training_method` is not saved, the call is replaced
        # with a `raise`.
        m.save("m.pt")

    .. testcleanup::

        import os
        os.remove('m.pt')
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch._jit_internal.ignore', 'ignore(drop=False, **kwargs)', {'FunctionModifiers': FunctionModifiers, 'warnings': warnings, 'drop': drop, 'kwargs': kwargs}, 1)

def _copy_to_script_wrapper(fn):
    fn._torchscript_modifier = FunctionModifiers.COPY_TO_SCRIPT_WRAPPER
    return fn

def module_has_exports(mod):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch._jit_internal.module_has_exports', 'module_has_exports(mod)', {'get_torchscript_modifier': get_torchscript_modifier, 'FunctionModifiers': FunctionModifiers, 'mod': mod}, 1)

def should_drop(fn):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch._jit_internal.should_drop', 'should_drop(fn)', {'get_torchscript_modifier': get_torchscript_modifier, 'FunctionModifiers': FunctionModifiers, 'fn': fn}, 1)

def is_ignored_fn(fn):
    mod = get_torchscript_modifier(fn)
    return (mod is FunctionModifiers.UNUSED or mod is FunctionModifiers.IGNORE)

def get_torchscript_modifier(fn):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch._jit_internal.get_torchscript_modifier', 'get_torchscript_modifier(fn)', {'FunctionModifiers': FunctionModifiers, 'fn': fn}, 1)

def copy_torchscript_modifier(orig, new):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch._jit_internal.copy_torchscript_modifier', 'copy_torchscript_modifier(orig, new)', {'get_torchscript_modifier': get_torchscript_modifier, 'orig': orig, 'new': new}, 1)
_overloaded_fns = {}

def _overload(func):
    qual_name = _qualified_name(func)
    global _overloaded_fns
    fn_overload_list = _overloaded_fns.get(qual_name)
    if fn_overload_list is None:
        fn_overload_list = []
        _overloaded_fns[qual_name] = fn_overload_list
    fn_overload_list.append(func)
    return func

def _get_fn_overloads(qual_name):
    return _overloaded_fns.get(qual_name)

def _clear_fn_overloads(qual_name):
    del _overloaded_fns[qual_name]

def get_class_name_lineno(method):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch._jit_internal.get_class_name_lineno', 'get_class_name_lineno(method)', {'inspect': inspect, 'method': method}, 2)
_overloaded_methods = {}
_overloaded_method_class_fileno = {}

def _overload_method(func):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch._jit_internal._overload_method', '_overload_method(func)', {'_qualified_name': _qualified_name, '_overloaded_methods': _overloaded_methods, 'get_class_name_lineno': get_class_name_lineno, '_overloaded_method_class_fileno': _overloaded_method_class_fileno, 'func': func}, 1)

def _get_overloaded_methods(method, mod_class):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch._jit_internal._get_overloaded_methods', '_get_overloaded_methods(method, mod_class)', {'_qualified_name': _qualified_name, '_overloaded_methods': _overloaded_methods, 'get_source_lines_and_file': get_source_lines_and_file, 'method': method, 'mod_class': mod_class}, 1)
try:
    import typing
    from typing import Tuple, List, Dict, Optional, Any
    
    def is_tuple(ann):
        import custom_funtemplate
        return custom_funtemplate.rewrite_template('torch._jit_internal.is_tuple', 'is_tuple(ann)', {'typing': typing, 'ann': ann}, 1)
    
    def is_list(ann):
        import custom_funtemplate
        return custom_funtemplate.rewrite_template('torch._jit_internal.is_list', 'is_list(ann)', {'typing': typing, 'ann': ann}, 1)
    
    def is_dict(ann):
        import custom_funtemplate
        return custom_funtemplate.rewrite_template('torch._jit_internal.is_dict', 'is_dict(ann)', {'typing': typing, 'ann': ann}, 1)
    
    def is_optional(ann):
        import custom_funtemplate
        return custom_funtemplate.rewrite_template('torch._jit_internal.is_optional', 'is_optional(ann)', {'inspect': inspect, 'typing': typing, 'ann': ann}, 1)
except ImportError:
    
    
    class TupleCls(object):
        
        def __getitem__(self, types):
            return TupleInstance(types)
    
    
    
    class TupleInstance(object):
        __slots__ = ['__args__']
        
        def __init__(self, types):
            self.__args__ = types
    
    
    
    class ListInstance(object):
        __slots__ = ['__args__']
        
        def __init__(self, types):
            self.__args__ = types
    
    
    
    class ListCls(object):
        
        def __getitem__(self, types):
            return TupleInstance(types)
    
    
    
    class DictInstance(object):
        __slots__ = ['__args__']
        
        def __init__(self, types):
            self.__args__ = types
    
    
    
    class DictCls(object):
        
        def __getitem__(self, types):
            return DictInstance(types)
    
    
    
    class OptionalInstance(object):
        __slots__ = ['__args__']
        
        def __init__(self, types):
            self.__args__ = types
    
    
    
    class OptionalCls(object):
        
        def __getitem__(self, types):
            return OptionalInstance(types)
    
    
    
    class AnyCls(object):
        pass
    
    Tuple = TupleCls()
    List = ListCls()
    Dict = DictCls()
    Optional = DictCls()
    Any = AnyCls()
    
    def is_tuple(ann):
        return isinstance(ann, TupleInstance)
    
    def is_list(ann):
        return isinstance(ann, ListInstance)
    
    def is_dict(ann):
        return isinstance(ann, DictInstance)
    
    def is_optional(ann):
        return isinstance(ann, OptionalInstance)
try:
    import typing_extensions
    from typing_extensions import Final
    
    def is_final(ann):
        return (ann.__module__ == 'typing_extensions' and getattr(ann, '__origin__', None) is typing_extensions.Final)
except ImportError:
    
    
    class FinalInstance(object):
        __slots__ = ['__args__']
        
        def __init__(self, types):
            self.__args__ = types
    
    
    
    class FinalCls(object):
        
        def __getitem__(self, types):
            return FinalInstance(types)
    
    Final = FinalCls()
    
    def is_final(ann):
        return isinstance(ann, FinalInstance)
try:
    from typing import TypeVar, Generic
    T = TypeVar('T')
    
    
    class RRef(Generic[T]):
        __slots__ = ['__args__']
        
        def __init__(self, types):
            self.__args__ = types
    
    
    def is_rref(ann):
        return getattr(ann, '__origin__', None) is RRef
except ImportError:
    
    
    class RRefInstance(object):
        __slots__ = ['__args__']
        
        def __init__(self, types):
            self.__args__ = types
    
    
    
    class RRefCls(object):
        
        def __getitem__(self, types):
            return RRefInstance(types)
    
    RRef = RRefCls()
    
    def is_rref(ann):
        return isinstance(ann, RRefInstance)


class BroadcastingListCls(object):
    
    def __getitem__(self, types):
        return

BroadcastingList1 = BroadcastingListCls()
for i in range(2, 7):
    globals()['BroadcastingList{}'.format(i)] = BroadcastingList1

def _qualified_name(obj):
    if hasattr(obj, '_jit_override_qualname'):
        return obj._jit_override_qualname
    if isinstance(obj, torch._C.ScriptFunction):
        return obj.qualified_name
    name = obj.__name__
    if name == '<lambda>':
        name = '_lambda'
    module_name = obj.__module__
    if module_name == 'torch._classes':
        return obj.qualified_name
    if module_name is None:
        raise RuntimeError("Could not get qualified name for class '{}': __module__ can't be None.".format(name))
    if module_name == '__main__':
        module_name = '__torch__'
    else:
        module_name = '__torch__.' + module_name
    if '.' in name:
        raise RuntimeError("Could not get qualified name for class '{}': '{}' is not a valid identifier".format(name, name))
    return module_name + '.' + name


class SourceContext(torch._C._jit_tree_views.SourceRangeFactory):
    
    def __init__(self, source, filename, file_lineno, leading_whitespace_len, uses_true_division=True):
        super(SourceContext, self).__init__(source, filename, file_lineno, leading_whitespace_len)
        self.uses_true_division = uses_true_division


def fake_range():
    return SourceContext('', None, 0, 0).make_raw_range(0, 1)

