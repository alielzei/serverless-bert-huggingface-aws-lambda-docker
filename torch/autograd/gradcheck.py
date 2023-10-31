import torch
from torch._six import container_abcs, istuple
import torch.testing
from itertools import product
import warnings

def zero_gradients(x):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.autograd.gradcheck.zero_gradients', 'zero_gradients(x)', {'torch': torch, 'container_abcs': container_abcs, 'zero_gradients': zero_gradients, 'x': x}, 0)

def make_jacobian(input, num_out):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.autograd.gradcheck.make_jacobian', 'make_jacobian(input, num_out)', {'torch': torch, 'container_abcs': container_abcs, 'make_jacobian': make_jacobian, 'input': input, 'num_out': num_out}, 1)

def iter_tensors(x, only_requiring_grad=False):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.autograd.gradcheck.iter_tensors', 'iter_tensors(x, only_requiring_grad=False)', {'torch': torch, 'container_abcs': container_abcs, 'iter_tensors': iter_tensors, 'x': x, 'only_requiring_grad': only_requiring_grad}, 0)

def get_numerical_jacobian(fn, input, target=None, eps=0.001):
    """
    input: input to `fn`
    target: the Tensors wrt whom Jacobians are calculated (default=`input`)

    Note that `target` may not even be part of `input` to `fn`, so please be
    **very careful** in this to not clone `target`.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.autograd.gradcheck.get_numerical_jacobian', 'get_numerical_jacobian(fn, input, target=None, eps=0.001)', {'make_jacobian': make_jacobian, 'iter_tensors': iter_tensors, 'product': product, 'torch': torch, 'fn': fn, 'input': input, 'target': target, 'eps': eps}, 1)

def get_analytical_jacobian(input, output, nondet_tol=0.0):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.autograd.gradcheck.get_analytical_jacobian', 'get_analytical_jacobian(input, output, nondet_tol=0.0)', {'torch': torch, 'iter_tensors': iter_tensors, 'make_jacobian': make_jacobian, 'input': input, 'output': output, 'nondet_tol': nondet_tol}, 3)

def _as_tuple(x):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.autograd.gradcheck._as_tuple', '_as_tuple(x)', {'istuple': istuple, 'x': x}, 1)

def _differentiable_outputs(x):
    return tuple((o for o in _as_tuple(x) if o.requires_grad))

def gradcheck(func, inputs, eps=1e-06, atol=1e-05, rtol=0.001, raise_exception=True, check_sparse_nnz=False, nondet_tol=0.0):
    """Check gradients computed via small finite differences against analytical
    gradients w.r.t. tensors in :attr:`inputs` that are of floating point type
    and with ``requires_grad=True``.

    The check between numerical and analytical gradients uses :func:`~torch.allclose`.

    .. note::
        The default values are designed for :attr:`input` of double precision.
        This check will likely fail if :attr:`input` is of less precision, e.g.,
        ``FloatTensor``.

    .. warning::
       If any checked tensor in :attr:`input` has overlapping memory, i.e.,
       different indices pointing to the same memory address (e.g., from
       :func:`torch.expand`), this check will likely fail because the numerical
       gradients computed by point perturbation at such indices will change
       values at all other indices that share the same memory address.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a Tensor or a tuple of Tensors
        inputs (tuple of Tensor or Tensor): inputs to the function
        eps (float, optional): perturbation for finite differences
        atol (float, optional): absolute tolerance
        rtol (float, optional): relative tolerance
        raise_exception (bool, optional): indicating whether to raise an exception if
            the check fails. The exception gives more information about the
            exact nature of the failure. This is helpful when debugging gradchecks.
        check_sparse_nnz (bool, optional): if True, gradcheck allows for SparseTensor input,
            and for any SparseTensor at input, gradcheck will perform check at nnz positions only.
        nondet_tol (float, optional): tolerance for non-determinism. When running
            identical inputs through the differentiation, the results must either match
            exactly (default, 0.0) or be within this tolerance.

    Returns:
        True if all differences satisfy allclose condition
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.autograd.gradcheck.gradcheck', 'gradcheck(func, inputs, eps=1e-06, atol=1e-05, rtol=0.001, raise_exception=True, check_sparse_nnz=False, nondet_tol=0.0)', {'_as_tuple': _as_tuple, 'torch': torch, 'warnings': warnings, '_differentiable_outputs': _differentiable_outputs, 'get_numerical_jacobian': get_numerical_jacobian, 'get_analytical_jacobian': get_analytical_jacobian, 'iter_tensors': iter_tensors, 'func': func, 'inputs': inputs, 'eps': eps, 'atol': atol, 'rtol': rtol, 'raise_exception': raise_exception, 'check_sparse_nnz': check_sparse_nnz, 'nondet_tol': nondet_tol}, 1)

def gradgradcheck(func, inputs, grad_outputs=None, eps=1e-06, atol=1e-05, rtol=0.001, gen_non_contig_grad_outputs=False, raise_exception=True, nondet_tol=0.0):
    """Check gradients of gradients computed via small finite differences
    against analytical gradients w.r.t. tensors in :attr:`inputs` and
    :attr:`grad_outputs` that are of floating point type and with
    ``requires_grad=True``.

    This function checks that backpropagating through the gradients computed
    to the given :attr:`grad_outputs` are correct.

    The check between numerical and analytical gradients uses :func:`~torch.allclose`.

    .. note::
        The default values are designed for :attr:`input` and
        :attr:`grad_outputs` of double precision. This check will likely fail if
        they are of less precision, e.g., ``FloatTensor``.

    .. warning::
       If any checked tensor in :attr:`input` and :attr:`grad_outputs` has
       overlapping memory, i.e., different indices pointing to the same memory
       address (e.g., from :func:`torch.expand`), this check will likely fail
       because the numerical gradients computed by point perturbation at such
       indices will change values at all other indices that share the same
       memory address.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a Tensor or a tuple of Tensors
        inputs (tuple of Tensor or Tensor): inputs to the function
        grad_outputs (tuple of Tensor or Tensor, optional): The gradients with
            respect to the function's outputs.
        eps (float, optional): perturbation for finite differences
        atol (float, optional): absolute tolerance
        rtol (float, optional): relative tolerance
        gen_non_contig_grad_outputs (bool, optional): if :attr:`grad_outputs` is
            ``None`` and :attr:`gen_non_contig_grad_outputs` is ``True``, the
            randomly generated gradient outputs are made to be noncontiguous
        raise_exception (bool, optional): indicating whether to raise an exception if
            the check fails. The exception gives more information about the
            exact nature of the failure. This is helpful when debugging gradchecks.
        nondet_tol (float, optional): tolerance for non-determinism. When running
            identical inputs through the differentiation, the results must either match
            exactly (default, 0.0) or be within this tolerance. Note that a small amount
            of nondeterminism in the gradient will lead to larger inaccuracies in
            the second derivative.

    Returns:
        True if all differences satisfy allclose condition
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.autograd.gradcheck.gradgradcheck', 'gradgradcheck(func, inputs, grad_outputs=None, eps=1e-06, atol=1e-05, rtol=0.001, gen_non_contig_grad_outputs=False, raise_exception=True, nondet_tol=0.0)', {'_as_tuple': _as_tuple, 'torch': torch, '_differentiable_outputs': _differentiable_outputs, 'gradcheck': gradcheck, 'func': func, 'inputs': inputs, 'grad_outputs': grad_outputs, 'eps': eps, 'atol': atol, 'rtol': rtol, 'gen_non_contig_grad_outputs': gen_non_contig_grad_outputs, 'raise_exception': raise_exception, 'nondet_tol': nondet_tol}, 1)

