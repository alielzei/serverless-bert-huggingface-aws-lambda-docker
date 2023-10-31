import torch
import torch._C as _C
import torch.utils.hooks as hooks
from torch._six import with_metaclass
import functools
import warnings
from collections import OrderedDict


class _ContextMethodMixin(object):
    
    def save_for_backward(self, *tensors):
        """Saves given tensors for a future call to :func:`~Function.backward`.

        **This should be called at most once, and only from inside the**
        :func:`forward` **method.**

        Later, saved tensors can be accessed through the :attr:`saved_tensors`
        attribute. Before returning them to the user, a check is made to ensure
        they weren't used in any in-place operation that modified their content.

        Arguments can also be ``None``.
        """
        self.to_save = tensors
    
    def mark_dirty(self, *args):
        """Marks given tensors as modified in an in-place operation.

        **This should be called at most once, only from inside the**
        :func:`forward` **method, and all arguments should be inputs.**

        Every tensor that's been modified in-place in a call to :func:`forward`
        should be given to this function, to ensure correctness of our checks.
        It doesn't matter whether the function is called before or after
        modification.
        """
        self.dirty_tensors = args
    
    def mark_shared_storage(self, *pairs):
        warnings.warn('mark_shared_storage is deprecated. Tensors with shared storages are automatically tracked. Note that calls to `set_()` are not tracked')
    
    def mark_non_differentiable(self, *args):
        """Marks outputs as non-differentiable.

        **This should be called at most once, only from inside the**
        :func:`forward` **method, and all arguments should be outputs.**

        This will mark outputs as not requiring gradients, increasing the
        efficiency of backward computation. You still need to accept a gradient
        for each output in :meth:`~Function.backward`, but it's always going to
        be a zero tensor with the same shape as the shape of a corresponding
        output.

        This is used e.g. for indices returned from a max :class:`Function`.
        """
        self.non_differentiable = args



class _HookMixin(object):
    
    @staticmethod
    def _register_hook(backward_hooks, hook):
        if backward_hooks is None:
            backward_hooks = OrderedDict()
        handle = hooks.RemovableHandle(backward_hooks)
        backward_hooks[handle.id] = hook
        return (backward_hooks, handle)



class BackwardCFunction(_C._FunctionBase, _ContextMethodMixin, _HookMixin):
    _is_legacy = False
    
    def apply(self, *args):
        return self._forward_cls.backward(self, *args)



class FunctionMeta(type):
    """Function metaclass.

    This metaclass sets up the following properties:
        _is_legacy: True if forward is not defined as a static method.
        _backward_cls: The Function class corresponding to the differentiated
            version of this function (which is generated on the fly by this
            metaclass).
    """
    
    def __init__(cls, name, bases, attrs):
        for super_cls in cls.mro():
            forward = super_cls.__dict__.get('forward')
            if forward is not None:
                has_static_forward = (isinstance(forward, staticmethod) or isinstance(forward, classmethod))
                break
        cls._is_legacy = not has_static_forward
        if not has_static_forward:
            return super(FunctionMeta, cls).__init__(name, bases, attrs)
        backward_fn = type(name + 'Backward', (BackwardCFunction, ), {'_forward_cls': cls})
        cls._backward_cls = backward_fn
        return super(FunctionMeta, cls).__init__(name, bases, attrs)



class Function(with_metaclass(FunctionMeta, _C._FunctionBase, _ContextMethodMixin, _HookMixin)):
    """Records operation history and defines formulas for differentiating ops.

    Every operation performed on :class:`Tensor` s creates a new function
    object, that performs the computation, and records that it happened.
    The history is retained in the form of a DAG of functions, with edges
    denoting data dependencies (``input <- output``). Then, when backward is
    called, the graph is processed in the topological ordering, by calling
    :func:`backward` methods of each :class:`Function` object, and passing
    returned gradients on to next :class:`Function` s.

    Normally, the only way users interact with functions is by creating
    subclasses and defining new operations. This is a recommended way of
    extending torch.autograd.

    Examples::

        >>> class Exp(Function):
        >>>
        >>>     @staticmethod
        >>>     def forward(ctx, i):
        >>>         result = i.exp()
        >>>         ctx.save_for_backward(result)
        >>>         return result
        >>>
        >>>     @staticmethod
        >>>     def backward(ctx, grad_output):
        >>>         result, = ctx.saved_tensors
        >>>         return grad_output * result
        >>>
        >>> #Use it by calling the apply method:
        >>> output = Exp.apply(input)
    """
    
    def __call__(self, *args, **kwargs):
        raise RuntimeError('Legacy autograd function with non-static forward method is deprecated. Please use new-style autograd function with static forward method. (Example: https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function)')
    is_traceable = False
    
    @staticmethod
    def forward(ctx, *args, **kwargs):
        """Performs the operation.

        This function is to be overridden by all subclasses.

        It must accept a context ctx as the first argument, followed by any
        number of arguments (tensors or other types).

        The context can be used to store tensors that can be then retrieved
        during the backward pass.
        """
        raise NotImplementedError('You must implement the forward function for custom autograd.Function.')
    
    @staticmethod
    def backward(ctx, *grad_outputs):
        """Defines a formula for differentiating the operation.

        This function is to be overridden by all subclasses.

        It must accept a context :attr:`ctx` as the first argument, followed by
        as many outputs did :func:`forward` return, and it should return as many
        tensors, as there were inputs to :func:`forward`. Each argument is the
        gradient w.r.t the given output, and each returned value should be the
        gradient w.r.t. the corresponding input.

        The context can be used to retrieve tensors saved during the forward
        pass. It also has an attribute :attr:`ctx.needs_input_grad` as a tuple
        of booleans representing whether each input needs gradient. E.g.,
        :func:`backward` will have ``ctx.needs_input_grad[0] = True`` if the
        first input to :func:`forward` needs gradient computated w.r.t. the
        output.
        """
        raise NotImplementedError('You must implement the backward function for custom autograd.Function.')


def once_differentiable(fn):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.autograd.function.once_differentiable', 'once_differentiable(fn)', {'functools': functools, 'torch': torch, 'fn': fn}, 1)

def traceable(fn_cls):
    """Marks Function as traceable for the JIT.

    Traceable functions have additional restrictions - they can't pass any
    data-dependent values to backward (e.g. Prod passes the output, which makes
    it non-traceable), and their backward should be implemented entirely in terms
    of operations on autograd Tensors in all cases.

    DON'T USE THIS DECORATOR. IT IS FOR INTERNAL USE ONLY AND SHOULD BE HANDLED WITH
    CARE (or can give incorrect results otherwise).
    """
    fn_cls.is_traceable = True
    return fn_cls


class InplaceFunction(Function):
    
    def __init__(self, inplace=False):
        super(InplaceFunction, self).__init__()
        self.inplace = inplace


def _nested_map(condition, fn, condition_msg=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.autograd.function._nested_map', '_nested_map(condition, fn, condition_msg=None)', {'torch': torch, 'condition': condition, 'fn': fn, 'condition_msg': condition_msg}, 1)

def _jit_unwrap_structured(obj):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.autograd.function._jit_unwrap_structured', '_jit_unwrap_structured(obj)', {'obj': obj}, 1)

def _iter_filter(condition, allow_unknown=False, condition_msg=None, conversion=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.autograd.function._iter_filter', '_iter_filter(condition, allow_unknown=False, condition_msg=None, conversion=None)', {'torch': torch, 'condition': condition, 'allow_unknown': allow_unknown, 'condition_msg': condition_msg, 'conversion': conversion}, 1)

def _unflatten(input, proto):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.autograd.function._unflatten', '_unflatten(input, proto)', {'input': input, 'proto': proto}, 1)
_iter_jit_values = _iter_filter(lambda o: (o is None or isinstance(o, torch._C.Value)), condition_msg="jit's Values or None")
_iter_tensors = _iter_filter(lambda x: isinstance(x, torch.Tensor), condition_msg='Tensors', conversion=_jit_unwrap_structured)
_iter_tensors_permissive = _iter_filter(lambda x: isinstance(x, torch.Tensor), allow_unknown=True, condition_msg='Tensors (permissive)')
_iter_None_tensors = _iter_filter(lambda o: (o is None or isinstance(o, torch.Tensor)), condition_msg='Tensors or None')
_map_tensor_data = _nested_map(lambda x: isinstance(x, torch.Tensor), lambda o: o.data, condition_msg='Tensors')


class NestedIOFunction(Function):
    
    def _do_forward(self, *input):
        self._nested_input = input
        flat_input = tuple(_iter_tensors(input))
        flat_output = super(NestedIOFunction, self)._do_forward(*flat_input)
        nested_output = self._nested_output
        nested_tensors = _unflatten(flat_output, self._nested_output)
        return nested_tensors
    
    def _do_backward(self, gradients, retain_variables):
        self.retain_variables = retain_variables
        result = super(NestedIOFunction, self)._do_backward(gradients, retain_variables)
        if not retain_variables:
            del self._nested_output
            del self._to_save_nested
        return result
    
    def backward(self, *gradients):
        nested_gradients = _unflatten(gradients, self._nested_output)
        result = self.backward_extended(*nested_gradients)
        return tuple(_iter_None_tensors(result))
    __call__ = _do_forward
    
    def forward(self, *args):
        nested_tensors = _map_tensor_data(self._nested_input)
        result = self.forward_extended(*nested_tensors)
        del self._nested_input
        self._nested_output = result
        return tuple(_iter_tensors(result))
    
    def save_for_backward(self, *args):
        self.to_save = tuple(_iter_tensors(args))
        self._to_save_nested = args
    
    @property
    def saved_tensors(self):
        flat_tensors = super(NestedIOFunction, self).saved_tensors
        return _unflatten(flat_tensors, self._to_save_nested)
    
    def mark_dirty(self, *args, **kwargs):
        self.dirty_tensors = tuple(_iter_tensors((args, kwargs)))
    
    def mark_non_differentiable(self, *args, **kwargs):
        self.non_differentiable = tuple(_iter_tensors((args, kwargs)))
    
    def forward_extended(self, *input):
        raise NotImplementedError
    
    def backward_extended(self, *grad_output):
        raise NotImplementedError


