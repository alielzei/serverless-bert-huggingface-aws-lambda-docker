"""
This class is defined to override standard pickle functionality

The goals of it follow:
-Serialize lambdas and nested functions to compiled byte code
-Deal with main module correctly
-Deal with other non-serializable objects

It does not include an unpickler, as standard python unpickling suffices.

This module was extracted from the `cloud` package, developed by `PiCloud, Inc.
<https://web.archive.org/web/20140626004012/http://www.picloud.com/>`_.

Copyright (c) 2012, Regents of the University of California.
Copyright (c) 2009 `PiCloud, Inc. <https://web.archive.org/web/20140626004012/http://www.picloud.com/>`_.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the University of California, Berkeley nor the
      names of its contributors may be used to endorse or promote
      products derived from this software without specific prior written
      permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import builtins
import dis
import opcode
import platform
import sys
import types
import weakref
import uuid
import threading
import typing
import warnings
from .compat import pickle
from collections import OrderedDict
from typing import ClassVar, Generic, Union, Tuple, Callable
from pickle import _getattribute
from importlib._bootstrap import _find_spec
try:
    import typing_extensions as _typing_extensions
    from typing_extensions import Literal, Final
except ImportError:
    _typing_extensions = Literal = Final = None
if sys.version_info >= (3, 8):
    from types import CellType
else:
    
    def f():
        import custom_funtemplate
        return custom_funtemplate.rewrite_template('joblib.externals.cloudpickle.cloudpickle.f', 'f()', {}, 1)
    CellType = type(f().__closure__[0])
DEFAULT_PROTOCOL = pickle.HIGHEST_PROTOCOL
_PICKLE_BY_VALUE_MODULES = set()
_DYNAMIC_CLASS_TRACKER_BY_CLASS = weakref.WeakKeyDictionary()
_DYNAMIC_CLASS_TRACKER_BY_ID = weakref.WeakValueDictionary()
_DYNAMIC_CLASS_TRACKER_LOCK = threading.Lock()
PYPY = platform.python_implementation() == 'PyPy'
builtin_code_type = None
if PYPY:
    builtin_code_type = type(float.__new__.__code__)
_extract_code_globals_cache = weakref.WeakKeyDictionary()

def _get_or_create_tracker_id(class_def):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.externals.cloudpickle.cloudpickle._get_or_create_tracker_id', '_get_or_create_tracker_id(class_def)', {'_DYNAMIC_CLASS_TRACKER_LOCK': _DYNAMIC_CLASS_TRACKER_LOCK, '_DYNAMIC_CLASS_TRACKER_BY_CLASS': _DYNAMIC_CLASS_TRACKER_BY_CLASS, 'uuid': uuid, '_DYNAMIC_CLASS_TRACKER_BY_ID': _DYNAMIC_CLASS_TRACKER_BY_ID, 'class_def': class_def}, 1)

def _lookup_class_or_track(class_tracker_id, class_def):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.externals.cloudpickle.cloudpickle._lookup_class_or_track', '_lookup_class_or_track(class_tracker_id, class_def)', {'_DYNAMIC_CLASS_TRACKER_LOCK': _DYNAMIC_CLASS_TRACKER_LOCK, '_DYNAMIC_CLASS_TRACKER_BY_ID': _DYNAMIC_CLASS_TRACKER_BY_ID, '_DYNAMIC_CLASS_TRACKER_BY_CLASS': _DYNAMIC_CLASS_TRACKER_BY_CLASS, 'class_tracker_id': class_tracker_id, 'class_def': class_def}, 1)

def register_pickle_by_value(module):
    """Register a module to make it functions and classes picklable by value.

    By default, functions and classes that are attributes of an importable
    module are to be pickled by reference, that is relying on re-importing
    the attribute from the module at load time.

    If `register_pickle_by_value(module)` is called, all its functions and
    classes are subsequently to be pickled by value, meaning that they can
    be loaded in Python processes where the module is not importable.

    This is especially useful when developing a module in a distributed
    execution environment: restarting the client Python process with the new
    source code is enough: there is no need to re-install the new version
    of the module on all the worker nodes nor to restart the workers.

    Note: this feature is considered experimental. See the cloudpickle
    README.md file for more details and limitations.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.externals.cloudpickle.cloudpickle.register_pickle_by_value', 'register_pickle_by_value(module)', {'types': types, 'sys': sys, '_PICKLE_BY_VALUE_MODULES': _PICKLE_BY_VALUE_MODULES, 'module': module}, 0)

def unregister_pickle_by_value(module):
    """Unregister that the input module should be pickled by value."""
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.externals.cloudpickle.cloudpickle.unregister_pickle_by_value', 'unregister_pickle_by_value(module)', {'types': types, '_PICKLE_BY_VALUE_MODULES': _PICKLE_BY_VALUE_MODULES, 'module': module}, 0)

def list_registry_pickle_by_value():
    return _PICKLE_BY_VALUE_MODULES.copy()

def _is_registered_pickle_by_value(module):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.externals.cloudpickle.cloudpickle._is_registered_pickle_by_value', '_is_registered_pickle_by_value(module)', {'_PICKLE_BY_VALUE_MODULES': _PICKLE_BY_VALUE_MODULES, 'module': module}, 1)

def _whichmodule(obj, name):
    """Find the module an object belongs to.

    This function differs from ``pickle.whichmodule`` in two ways:
    - it does not mangle the cases where obj's module is __main__ and obj was
      not found in any module.
    - Errors arising during module introspection are ignored, as those errors
      are considered unwanted side effects.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.externals.cloudpickle.cloudpickle._whichmodule', '_whichmodule(obj, name)', {'sys': sys, 'typing': typing, 'types': types, '_getattribute': _getattribute, 'obj': obj, 'name': name}, 1)

def _should_pickle_by_reference(obj, name=None):
    """Test whether an function or a class should be pickled by reference

     Pickling by reference means by that the object (typically a function or a
     class) is an attribute of a module that is assumed to be importable in the
     target Python environment. Loading will therefore rely on importing the
     module and then calling `getattr` on it to access the function or class.

     Pickling by reference is the only option to pickle functions and classes
     in the standard library. In cloudpickle the alternative option is to
     pickle by value (for instance for interactively or locally defined
     functions and classes or for attributes of modules that have been
     explicitly registered to be pickled by value.
     """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.externals.cloudpickle.cloudpickle._should_pickle_by_reference', '_should_pickle_by_reference(obj, name=None)', {'types': types, '_lookup_module_and_qualname': _lookup_module_and_qualname, '_is_registered_pickle_by_value': _is_registered_pickle_by_value, 'sys': sys, 'obj': obj, 'name': name}, 1)

def _lookup_module_and_qualname(obj, name=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.externals.cloudpickle.cloudpickle._lookup_module_and_qualname', '_lookup_module_and_qualname(obj, name=None)', {'_whichmodule': _whichmodule, 'sys': sys, '_getattribute': _getattribute, 'obj': obj, 'name': name}, 1)

def _extract_code_globals(co):
    """
    Find all globals names read or written to by codeblock co
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.externals.cloudpickle.cloudpickle._extract_code_globals', '_extract_code_globals(co)', {'_extract_code_globals_cache': _extract_code_globals_cache, '_walk_global_ops': _walk_global_ops, 'types': types, '_extract_code_globals': _extract_code_globals, 'co': co}, 1)

def _find_imported_submodules(code, top_level_dependencies):
    """
    Find currently imported submodules used by a function.

    Submodules used by a function need to be detected and referenced for the
    function to work correctly at depickling time. Because submodules can be
    referenced as attribute of their parent package (``package.submodule``), we
    need a special introspection technique that does not rely on GLOBAL-related
    opcodes to find references of them in a code object.

    Example:
    ```
    import concurrent.futures
    import cloudpickle
    def func():
        x = concurrent.futures.ThreadPoolExecutor
    if __name__ == '__main__':
        cloudpickle.dumps(func)
    ```
    The globals extracted by cloudpickle in the function's state include the
    concurrent package, but not its submodule (here, concurrent.futures), which
    is the module used by func. Find_imported_submodules will detect the usage
    of concurrent.futures. Saving this module alongside with func will ensure
    that calling func once depickled does not fail due to concurrent.futures
    not being imported
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.externals.cloudpickle.cloudpickle._find_imported_submodules', '_find_imported_submodules(code, top_level_dependencies)', {'types': types, 'sys': sys, 'code': code, 'top_level_dependencies': top_level_dependencies}, 1)

def cell_set(cell, value):
    """Set the value of a closure cell.

    The point of this function is to set the cell_contents attribute of a cell
    after its creation. This operation is necessary in case the cell contains a
    reference to the function the cell belongs to, as when calling the
    function's constructor
    ``f = types.FunctionType(code, globals, name, argdefs, closure)``,
    closure will not be able to contain the yet-to-be-created f.

    In Python3.7, cell_contents is writeable, so setting the contents of a cell
    can be done simply using
    >>> cell.cell_contents = value

    In earlier Python3 versions, the cell_contents attribute of a cell is read
    only, but this limitation can be worked around by leveraging the Python 3
    ``nonlocal`` keyword.

    In Python2 however, this attribute is read only, and there is no
    ``nonlocal`` keyword. For this reason, we need to come up with more
    complicated hacks to set this attribute.

    The chosen approach is to create a function with a STORE_DEREF opcode,
    which sets the content of a closure variable. Typically:

    >>> def inner(value):
    ...     lambda: cell  # the lambda makes cell a closure
    ...     cell = value  # cell is a closure, so this triggers a STORE_DEREF

    (Note that in Python2, A STORE_DEREF can never be triggered from an inner
    function. The function g for example here
    >>> def f(var):
    ...     def g():
    ...         var += 1
    ...     return g

    will not modify the closure variable ``var```inplace, but instead try to
    load a local variable var and increment it. As g does not assign the local
    variable ``var`` any initial value, calling f(1)() will fail at runtime.)

    Our objective is to set the value of a given cell ``cell``. So we need to
    somewhat reference our ``cell`` object into the ``inner`` function so that
    this object (and not the smoke cell of the lambda function) gets affected
    by the STORE_DEREF operation.

    In inner, ``cell`` is referenced as a cell variable (an enclosing variable
    that is referenced by the inner function). If we create a new function
    cell_set with the exact same code as ``inner``, but with ``cell`` marked as
    a free variable instead, the STORE_DEREF will be applied on its closure -
    ``cell``, which we can specify explicitly during construction! The new
    cell_set variable thus actually sets the contents of a specified cell!

    Note: we do not make use of the ``nonlocal`` keyword to set the contents of
    a cell in early python3 versions to limit possible syntax errors in case
    test and checker libraries decide to parse the whole file.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.externals.cloudpickle.cloudpickle.cell_set', 'cell_set(cell, value)', {'sys': sys, 'types': types, '_cell_set_template_code': _cell_set_template_code, 'cell': cell, 'value': value}, 0)

def _make_cell_set_template_code():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.externals.cloudpickle.cloudpickle._make_cell_set_template_code', '_make_cell_set_template_code()', {'types': types}, 1)
if sys.version_info[:2] < (3, 7):
    _cell_set_template_code = _make_cell_set_template_code()
STORE_GLOBAL = opcode.opmap['STORE_GLOBAL']
DELETE_GLOBAL = opcode.opmap['DELETE_GLOBAL']
LOAD_GLOBAL = opcode.opmap['LOAD_GLOBAL']
GLOBAL_OPS = (STORE_GLOBAL, DELETE_GLOBAL, LOAD_GLOBAL)
HAVE_ARGUMENT = dis.HAVE_ARGUMENT
EXTENDED_ARG = dis.EXTENDED_ARG
_BUILTIN_TYPE_NAMES = {}
for (k, v) in types.__dict__.items():
    if type(v) is type:
        _BUILTIN_TYPE_NAMES[v] = k

def _builtin_type(name):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.externals.cloudpickle.cloudpickle._builtin_type', '_builtin_type(name)', {'types': types, 'name': name}, 1)

def _walk_global_ops(code):
    """
    Yield referenced name for all global-referencing instructions in *code*.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('joblib.externals.cloudpickle.cloudpickle._walk_global_ops', '_walk_global_ops(code)', {'dis': dis, 'GLOBAL_OPS': GLOBAL_OPS, 'code': code}, 0)

def _extract_class_dict(cls):
    """Retrieve a copy of the dict of a class without the inherited methods"""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.externals.cloudpickle.cloudpickle._extract_class_dict', '_extract_class_dict(cls)', {'cls': cls}, 1)
if sys.version_info[:2] < (3, 7):
    
    def _is_parametrized_type_hint(obj):
        import custom_funtemplate
        return custom_funtemplate.rewrite_template('joblib.externals.cloudpickle.cloudpickle._is_parametrized_type_hint', '_is_parametrized_type_hint(obj)', {'obj': obj}, 1)
    
    def _create_parametrized_type_hint(origin, args):
        return origin[args]
else:
    _is_parametrized_type_hint = None
    _create_parametrized_type_hint = None

def parametrized_type_hint_getinitargs(obj):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.externals.cloudpickle.cloudpickle.parametrized_type_hint_getinitargs', 'parametrized_type_hint_getinitargs(obj)', {'Literal': Literal, 'Final': Final, 'ClassVar': ClassVar, 'Generic': Generic, 'Union': Union, 'Tuple': Tuple, 'Callable': Callable, 'Ellipsis': Ellipsis, 'pickle': pickle, 'obj': obj}, 1)

def is_tornado_coroutine(func):
    """
    Return whether *func* is a Tornado coroutine function.
    Running coroutines are not supported.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.externals.cloudpickle.cloudpickle.is_tornado_coroutine', 'is_tornado_coroutine(func)', {'sys': sys, 'func': func}, 1)

def _rebuild_tornado_coroutine(func):
    from tornado import gen
    return gen.coroutine(func)
load = pickle.load
loads = pickle.loads

def subimport(name):
    __import__(name)
    return sys.modules[name]

def dynamic_subimport(name, vars):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.externals.cloudpickle.cloudpickle.dynamic_subimport', 'dynamic_subimport(name, vars)', {'types': types, 'builtins': builtins, 'name': name, 'vars': vars}, 1)

def _gen_ellipsis():
    return Ellipsis

def _gen_not_implemented():
    return NotImplemented

def _get_cell_contents(cell):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.externals.cloudpickle.cloudpickle._get_cell_contents', '_get_cell_contents(cell)', {'_empty_cell_value': _empty_cell_value, 'cell': cell}, 1)

def instance(cls):
    """Create a new instance of a class.

    Parameters
    ----------
    cls : type
        The class to create an instance of.

    Returns
    -------
    instance : cls
        A new instance of ``cls``.
    """
    return cls()


@instance
class _empty_cell_value:
    """sentinel for empty closures
    """
    
    @classmethod
    def __reduce__(cls):
        return cls.__name__


def _fill_function(*args):
    """Fills in the rest of function data into the skeleton function object

    The skeleton itself is create by _make_skel_func().
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.externals.cloudpickle.cloudpickle._fill_function', '_fill_function(*args)', {'_empty_cell_value': _empty_cell_value, 'cell_set': cell_set, 'args': args}, 1)

def _make_function(code, globals, name, argdefs, closure):
    globals['__builtins__'] = __builtins__
    return types.FunctionType(code, globals, name, argdefs, closure)

def _make_empty_cell():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.externals.cloudpickle.cloudpickle._make_empty_cell', '_make_empty_cell()', {}, 1)

def _make_cell(value=_empty_cell_value):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.externals.cloudpickle.cloudpickle._make_cell', '_make_cell(value=_empty_cell_value)', {'_make_empty_cell': _make_empty_cell, 'cell_set': cell_set, 'value': value, '_empty_cell_value': _empty_cell_value}, 1)

def _make_skel_func(code, cell_count, base_globals=None):
    """ Creates a skeleton function object that contains just the provided
        code and the correct number of cells in func_closure.  All other
        func attributes (e.g. func_globals) are empty.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.externals.cloudpickle.cloudpickle._make_skel_func', '_make_skel_func(code, cell_count, base_globals=None)', {'warnings': warnings, '__builtins__': __builtins__, '_make_empty_cell': _make_empty_cell, 'types': types, 'code': code, 'cell_count': cell_count, 'base_globals': base_globals}, 1)

def _make_skeleton_class(type_constructor, name, bases, type_kwargs, class_tracker_id, extra):
    """Build dynamic class with an empty __dict__ to be filled once memoized

    If class_tracker_id is not None, try to lookup an existing class definition
    matching that id. If none is found, track a newly reconstructed class
    definition under that id so that other instances stemming from the same
    class id will also reuse this class definition.

    The "extra" variable is meant to be a dict (or None) that can be used for
    forward compatibility shall the need arise.
    """
    skeleton_class = types.new_class(name, bases, {'metaclass': type_constructor}, lambda ns: ns.update(type_kwargs))
    return _lookup_class_or_track(class_tracker_id, skeleton_class)

def _rehydrate_skeleton_class(skeleton_class, class_dict):
    """Put attributes from `class_dict` back on `skeleton_class`.

    See CloudPickler.save_dynamic_class for more info.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.externals.cloudpickle.cloudpickle._rehydrate_skeleton_class', '_rehydrate_skeleton_class(skeleton_class, class_dict)', {'skeleton_class': skeleton_class, 'class_dict': class_dict}, 1)

def _make_skeleton_enum(bases, name, qualname, members, module, class_tracker_id, extra):
    """Build dynamic enum with an empty __dict__ to be filled once memoized

    The creation of the enum class is inspired by the code of
    EnumMeta._create_.

    If class_tracker_id is not None, try to lookup an existing enum definition
    matching that id. If none is found, track a newly reconstructed enum
    definition under that id so that other instances stemming from the same
    class id will also reuse this enum definition.

    The "extra" variable is meant to be a dict (or None) that can be used for
    forward compatibility shall the need arise.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.externals.cloudpickle.cloudpickle._make_skeleton_enum', '_make_skeleton_enum(bases, name, qualname, members, module, class_tracker_id, extra)', {'_lookup_class_or_track': _lookup_class_or_track, 'bases': bases, 'name': name, 'qualname': qualname, 'members': members, 'module': module, 'class_tracker_id': class_tracker_id, 'extra': extra}, 1)

def _make_typevar(name, bound, constraints, covariant, contravariant, class_tracker_id):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.externals.cloudpickle.cloudpickle._make_typevar', '_make_typevar(name, bound, constraints, covariant, contravariant, class_tracker_id)', {'typing': typing, '_lookup_class_or_track': _lookup_class_or_track, 'name': name, 'bound': bound, 'constraints': constraints, 'covariant': covariant, 'contravariant': contravariant, 'class_tracker_id': class_tracker_id}, 1)

def _decompose_typevar(obj):
    return (obj.__name__, obj.__bound__, obj.__constraints__, obj.__covariant__, obj.__contravariant__, _get_or_create_tracker_id(obj))

def _typevar_reduce(obj):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.externals.cloudpickle.cloudpickle._typevar_reduce', '_typevar_reduce(obj)', {'_lookup_module_and_qualname': _lookup_module_and_qualname, '_make_typevar': _make_typevar, '_decompose_typevar': _decompose_typevar, '_is_registered_pickle_by_value': _is_registered_pickle_by_value, 'obj': obj}, 2)

def _get_bases(typ):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.externals.cloudpickle.cloudpickle._get_bases', '_get_bases(typ)', {'typ': typ}, 1)

def _make_dict_keys(obj, is_ordered=False):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.externals.cloudpickle.cloudpickle._make_dict_keys', '_make_dict_keys(obj, is_ordered=False)', {'OrderedDict': OrderedDict, 'obj': obj, 'is_ordered': is_ordered}, 1)

def _make_dict_values(obj, is_ordered=False):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.externals.cloudpickle.cloudpickle._make_dict_values', '_make_dict_values(obj, is_ordered=False)', {'OrderedDict': OrderedDict, 'obj': obj, 'is_ordered': is_ordered}, 1)

def _make_dict_items(obj, is_ordered=False):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.externals.cloudpickle.cloudpickle._make_dict_items', '_make_dict_items(obj, is_ordered=False)', {'OrderedDict': OrderedDict, 'obj': obj, 'is_ordered': is_ordered}, 1)

