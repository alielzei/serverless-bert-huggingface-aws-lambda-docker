import torch

def _as_tuple(inp, arg_name, fn_name):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.autograd.functional._as_tuple', '_as_tuple(inp, arg_name, fn_name)', {'torch': torch, 'inp': inp, 'arg_name': arg_name, 'fn_name': fn_name}, 2)

def _tuple_postprocess(res, to_unpack):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.autograd.functional._tuple_postprocess', '_tuple_postprocess(res, to_unpack)', {'res': res, 'to_unpack': to_unpack}, 1)

def _grad_preprocess(inputs, create_graph, need_graph):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.autograd.functional._grad_preprocess', '_grad_preprocess(inputs, create_graph, need_graph)', {'inputs': inputs, 'create_graph': create_graph, 'need_graph': need_graph}, 1)

def _grad_postprocess(inputs, create_graph):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.autograd.functional._grad_postprocess', '_grad_postprocess(inputs, create_graph)', {'torch': torch, '_grad_postprocess': _grad_postprocess, 'inputs': inputs, 'create_graph': create_graph}, 1)

def _validate_v(v, other, is_other_tuple):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.autograd.functional._validate_v', '_validate_v(v, other, is_other_tuple)', {'v': v, 'other': other, 'is_other_tuple': is_other_tuple}, 0)

def _check_requires_grad(inputs, input_type, strict):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.autograd.functional._check_requires_grad', '_check_requires_grad(inputs, input_type, strict)', {'inputs': inputs, 'input_type': input_type, 'strict': strict}, 1)

def _autograd_grad(outputs, inputs, grad_outputs=None, create_graph=False, retain_graph=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.autograd.functional._autograd_grad', '_autograd_grad(outputs, inputs, grad_outputs=None, create_graph=False, retain_graph=None)', {'torch': torch, 'outputs': outputs, 'inputs': inputs, 'grad_outputs': grad_outputs, 'create_graph': create_graph, 'retain_graph': retain_graph}, 1)

def _fill_in_zeros(grads, refs, strict, create_graph, stage):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.autograd.functional._fill_in_zeros', '_fill_in_zeros(grads, refs, strict, create_graph, stage)', {'torch': torch, 'grads': grads, 'refs': refs, 'strict': strict, 'create_graph': create_graph, 'stage': stage}, 1)

def vjp(func, inputs, v=None, create_graph=False, strict=False):
    """Function that computes the dot product between a vector ``v`` and the Jacobian of
    the given function at the point given by the inputs.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a tuple of Tensors or a Tensor.
        inputs (tuple of Tensors or Tensor): inputs to the function ``func``.
        v (tuple of Tensors or Tensor): The vector for which the vector Jacobian product is computed.
            Must be the same size as the output of ``func``. This argument is optional when
            ``func``'s output contains a single element and (if it is not provided) will be set as a Tensor
            containing a single ``1``.
        create_graph (bool, optional): If ``True``, both the output and result will be
            computed in a differentiable way. Note that when ``strict`` is ``False``, the result can not
            require gradients or be disconnected from the inputs.
            Defaults to ``False``.
        strict (bool, optional): If ``True``, an error will be raised when we detect that there exists an input
            such that all the outputs are independent of it. If ``False``, we return a Tensor of zeros as the
            vjp for said inputs, which is the expected mathematical value.
            Defaults to ``False``.

    Returns:
        func_output (tuple of Tensors or Tensor): output of ``func(inputs)``
        vjp (tuple of Tensors or Tensor): result of the dot product with the same shape
            as the inputs.

    Example::

        >>> def exp_reducer(x):
        ...   return x.exp().sum(dim=1)
        >>> inputs = torch.rand(4, 4)
        >>> v = torch.ones(4)
        >>> vjp(exp_reducer, inputs, v)
        (tensor([5.7817, 7.2458, 5.7830, 6.7782]),
         tensor([[1.4458, 1.3962, 1.3042, 1.6354],
                [2.1288, 1.0652, 1.5483, 2.5035],
                [2.2046, 1.1292, 1.1432, 1.3059],
                [1.3225, 1.6652, 1.7753, 2.0152]]))

        >>> vjp(exp_reducer, inputs, v, create_graph=True)
        (tensor([5.7817, 7.2458, 5.7830, 6.7782], grad_fn=<SumBackward1>),
         tensor([[1.4458, 1.3962, 1.3042, 1.6354],
                [2.1288, 1.0652, 1.5483, 2.5035],
                [2.2046, 1.1292, 1.1432, 1.3059],
                [1.3225, 1.6652, 1.7753, 2.0152]], grad_fn=<MulBackward0>))

        >>> def adder(x, y):
        ...   return 2 * x + 3 * y
        >>> inputs = (torch.rand(2), torch.rand(2))
        >>> v = torch.ones(2)
        >>> vjp(adder, inputs, v)
        (tensor([2.4225, 2.3340]),
         (tensor([2., 2.]), tensor([3., 3.])))
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.autograd.functional.vjp', 'vjp(func, inputs, v=None, create_graph=False, strict=False)', {'_as_tuple': _as_tuple, '_grad_preprocess': _grad_preprocess, '_check_requires_grad': _check_requires_grad, '_validate_v': _validate_v, '_autograd_grad': _autograd_grad, '_fill_in_zeros': _fill_in_zeros, '_grad_postprocess': _grad_postprocess, '_tuple_postprocess': _tuple_postprocess, 'func': func, 'inputs': inputs, 'v': v, 'create_graph': create_graph, 'strict': strict}, 2)

def jvp(func, inputs, v=None, create_graph=False, strict=False):
    """Function that computes the dot product between  the Jacobian of
    the given function at the point given by the inputs and a vector ``v``.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a tuple of Tensors or a Tensor.
        inputs (tuple of Tensors or Tensor): inputs to the function ``func``.
        v (tuple of Tensors or Tensor): The vector for which the Jacobian vector product is computed. Must be the
            same size as the input of ``func``. This argument is optional when
            ``func``'s input contains a single element and (if it is not provided) will be set as a Tensor
            containing a single ``1``.
        create_graph (bool, optional): If ``True``, both the output and result will be
            computed in a differentiable way. Note that when ``strict`` is ``False``, the result can not
            require gradients or be disconnected from the inputs.
            Defaults to ``False``.
        strict (bool, optional): If ``True``, an error will be raised when we detect that there exists an input
            such that all the outputs are independent of it. If ``False``, we return a Tensor of zeros as the
            jvp for said inputs, which is the expected mathematical value.
            Defaults to ``False``.

    Returns:
        func_output (tuple of Tensors or Tensor): output of ``func(inputs)``
        jvp (tuple of Tensors or Tensor): result of the dot product with the same shape
            as the output.

    Example::

        >>> def exp_reducer(x):
        ...   return x.exp().sum(dim=1)
        >>> inputs = torch.rand(4, 4)
        >>> v = torch.ones(4, 4)
        >>> jvp(exp_reducer, inputs, v)
        (tensor([6.3090, 4.6742, 7.9114, 8.2106]),
         tensor([6.3090, 4.6742, 7.9114, 8.2106]))

        >>> jvp(exp_reducer, inputs, v, create_graph=True)
        (tensor([6.3090, 4.6742, 7.9114, 8.2106], grad_fn=<SumBackward1>),
         tensor([6.3090, 4.6742, 7.9114, 8.2106], grad_fn=<SqueezeBackward1>))

        >>> def adder(x, y):
        ...   return 2 * x + 3 * y
        >>> inputs = (torch.rand(2), torch.rand(2))
        >>> v = (torch.ones(2), torch.ones(2))
        >>> jvp(adder, inputs, v)
        (tensor([2.2399, 2.5005]),
         tensor([5., 5.]))

    Note::

        The jvp is currently computed by using the backward of the backward (sometimes called the double
        backwards trick) as we don't have support for forward mode AD in PyTorch at the moment.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.autograd.functional.jvp', 'jvp(func, inputs, v=None, create_graph=False, strict=False)', {'_as_tuple': _as_tuple, '_grad_preprocess': _grad_preprocess, '_validate_v': _validate_v, '_check_requires_grad': _check_requires_grad, 'torch': torch, '_autograd_grad': _autograd_grad, '_fill_in_zeros': _fill_in_zeros, '_grad_postprocess': _grad_postprocess, '_tuple_postprocess': _tuple_postprocess, 'func': func, 'inputs': inputs, 'v': v, 'create_graph': create_graph, 'strict': strict}, 2)

def jacobian(func, inputs, create_graph=False, strict=False):
    """Function that computes the Jacobian of a given function.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a tuple of Tensors or a Tensor.
        inputs (tuple of Tensors or Tensor): inputs to the function ``func``.
        create_graph (bool, optional): If ``True``, the Jacobian will be computed in
            a differentiable manner. Note that when ``strict`` is ``False``, the result can not
            require gradients or be disconnected from the inputs.
            Defaults to ``False``.
        strict (bool, optional): If ``True``, an error will be raised when we detect that there exists an input
            such that all the outputs are independent of it. If ``False``, we return a Tensor of zeros as the
            jacobian for said inputs, which is the expected mathematical value.
            Defaults to ``False``.

    Returns:
        Jacobian (Tensor or nested tuple of Tensors) if there are a single input
            and output, this will be a single Tensor containing the Jacobian for the
            linearized inputs and output. If one of the two is a tuple, then the Jacobian
            will be a tuple of Tensors. If both of them are tuples, then the Jacobian will
            be a tuple of tuple of Tensors where ``Jacobian[i][j]`` will contain the Jacobian
            of the ``i``th output and ``j``th input and will have as size the concatenation of the
            sizes of the corresponding output and the corresponding input.

    Example::

        >>> def exp_reducer(x):
        ...   return x.exp().sum(dim=1)
        >>> inputs = torch.rand(2, 2)
        >>> jacobian(exp_reducer, inputs)
        tensor([[[1.4917, 2.4352],
                 [0.0000, 0.0000]],

                [[0.0000, 0.0000],
                 [2.4369, 2.3799]]])

        >>> jacobian(exp_reducer, inputs, create_graph=True)
        tensor([[[1.4917, 2.4352],
                 [0.0000, 0.0000]],

                [[0.0000, 0.0000],
                 [2.4369, 2.3799]]], grad_fn=<ViewBackward>)

        >>> def exp_adder(x, y):
        ...   return 2 * x.exp() + 3 * y
        >>> inputs = (torch.rand(2), torch.rand(2))
        >>> jacobian(exp_adder, inputs)
        (tensor([[2.8052, 0.0000],
                [0.0000, 3.3963]]),
         tensor([[3., 0.],
                 [0., 3.]]))
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.autograd.functional.jacobian', 'jacobian(func, inputs, create_graph=False, strict=False)', {'_as_tuple': _as_tuple, '_grad_preprocess': _grad_preprocess, '_check_requires_grad': _check_requires_grad, '_autograd_grad': _autograd_grad, 'torch': torch, '_grad_postprocess': _grad_postprocess, '_tuple_postprocess': _tuple_postprocess, 'func': func, 'inputs': inputs, 'create_graph': create_graph, 'strict': strict}, 1)

def hessian(func, inputs, create_graph=False, strict=False):
    """Function that computes the Hessian of a given scalar function.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a Tensor with a single element.
        inputs (tuple of Tensors or Tensor): inputs to the function ``func``.
        create_graph (bool, optional): If ``True``, the Hessian will be computed in
            a differentiable manner. Note that when ``strict`` is ``False``, the result can not
            require gradients or be disconnected from the inputs.
            Defaults to ``False``.
        strict (bool, optional): If ``True``, an error will be raised when we detect that there exists an input
            such that all the outputs are independent of it. If ``False``, we return a Tensor of zeros as the
            hessian for said inputs, which is the expected mathematical value.
            Defaults to ``False``.

    Returns:
        Hessian (Tensor or a tuple of tuple of Tensors) if there are a single input,
            this will be a single Tensor containing the Hessian for the input.
            If it is a tuple, then the Hessian will be a tuple of tuples where
            ``Hessian[i][j]`` will contain the Hessian of the ``i``th input
            and ``j``th input with size the sum of the size of the ``i``th input plus
            the size of the ``j``th input.

    Example::

        >>> def pow_reducer(x):
        ...   return x.pow(3).sum()
        >>> inputs = torch.rand(2, 2)
        >>> hessian(pow_reducer, inputs)
        tensor([[[[5.2265, 0.0000],
                  [0.0000, 0.0000]],

                 [[0.0000, 4.8221],
                  [0.0000, 0.0000]]],


                [[[0.0000, 0.0000],
                  [1.9456, 0.0000]],

                 [[0.0000, 0.0000],
                  [0.0000, 3.2550]]]])

        >>> hessian(pow_reducer, inputs, create_graph=True)
        tensor([[[[5.2265, 0.0000],
                  [0.0000, 0.0000]],

                 [[0.0000, 4.8221],
                  [0.0000, 0.0000]]],


                [[[0.0000, 0.0000],
                  [1.9456, 0.0000]],

                 [[0.0000, 0.0000],
                  [0.0000, 3.2550]]]], grad_fn=<ViewBackward>)


        >>> def pow_adder_reducer(x, y):
        ...   return (2 * x.pow(2) + 3 * y.pow(2)).sum()
        >>> inputs = (torch.rand(2), torch.rand(2))
        >>> hessian(pow_adder_reducer, inputs)
        ((tensor([[4., 0.],
                  [0., 4.]]),
          tensor([[0., 0.],
                  [0., 0.]])),
         (tensor([[0., 0.],
                  [0., 0.]]),
          tensor([[6., 0.],
                  [0., 6.]])))
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.autograd.functional.hessian', 'hessian(func, inputs, create_graph=False, strict=False)', {'_as_tuple': _as_tuple, '_check_requires_grad': _check_requires_grad, 'torch': torch, 'jacobian': jacobian, '_tuple_postprocess': _tuple_postprocess, 'func': func, 'inputs': inputs, 'create_graph': create_graph, 'strict': strict}, 1)

def vhp(func, inputs, v=None, create_graph=False, strict=False):
    """Function that computes the dot product between a vector ``v`` and the
    Hessian of a given scalar function at the point given by the inputs.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a Tensor with a single element.
        inputs (tuple of Tensors or Tensor): inputs to the function ``func``.
        v (tuple of Tensors or Tensor): The vector for which the vector Hessian product is computed. Must be the
            same size as the input of ``func``. This argument is optional when
            ``func``'s input contains a single element and (if it is not provided) will be set as a Tensor
            containing a single ``1``.
        create_graph (bool, optional): If ``True``, both the output and result will be
            computed in a differentiable way. Note that when ``strict`` is ``False``, the result can not
            require gradients or be disconnected from the inputs.
            Defaults to ``False``.
        strict (bool, optional): If ``True``, an error will be raised when we detect that there exists an input
            such that all the outputs are independent of it. If ``False``, we return a Tensor of zeros as the
            vhp for said inputs, which is the expected mathematical value.
            Defaults to ``False``.

    Returns:
        func_output (tuple of Tensors or Tensor): output of ``func(inputs)``
        vhp (tuple of Tensors or Tensor): result of the dot product with the same shape
            as the inputs.

    Example::

        >>> def pow_reducer(x):
        ...   return x.pow(3).sum()
        >>> inputs = torch.rand(2, 2)
        >>> v = torch.ones(2, 2)
        >>> vhp(pow_reducer, inputs, v)
       (tensor(0.5591),
        tensor([[1.0689, 1.2431],
                [3.0989, 4.4456]]))

        >>> vhp(pow_reducer, inputs, v, create_graph=True)
        (tensor(0.5591, grad_fn=<SumBackward0>),
         tensor([[1.0689, 1.2431],
                 [3.0989, 4.4456]], grad_fn=<MulBackward0>))


        >>> def pow_adder_reducer(x, y):
        ...   return (2 * x.pow(2) + 3 * y.pow(2)).sum()
        >>> inputs = (torch.rand(2), torch.rand(2))
        >>> v = (torch.zeros(2), torch.ones(2))
        >>> vhp(pow_adder_reducer, inputs, v)
        (tensor(4.8053),
         (tensor([0., 0.]),
          tensor([6., 6.])))

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.autograd.functional.vhp', 'vhp(func, inputs, v=None, create_graph=False, strict=False)', {'_as_tuple': _as_tuple, '_grad_preprocess': _grad_preprocess, '_validate_v': _validate_v, '_check_requires_grad': _check_requires_grad, 'torch': torch, '_autograd_grad': _autograd_grad, '_fill_in_zeros': _fill_in_zeros, '_grad_postprocess': _grad_postprocess, '_tuple_postprocess': _tuple_postprocess, 'func': func, 'inputs': inputs, 'v': v, 'create_graph': create_graph, 'strict': strict}, 2)

def hvp(func, inputs, v=None, create_graph=False, strict=False):
    """Function that computes the dot product between the Hessian of a given scalar
    function and a vector ``v`` at the point given by the inputs.

    Args:
        func (function): a Python function that takes Tensor inputs and returns
            a Tensor with a single element.
        inputs (tuple of Tensors or Tensor): inputs to the function ``func``.
        v (tuple of Tensors or Tensor): The vector for which the Hessian vector product is computed. Must be the
            same size as the input of ``func``. This argument is optional when
            ``func``'s input contains a single element and (if it is not provided) will be set as a Tensor
            containing a single ``1``.
        create_graph (bool, optional): If ``True``, both the output and result will be
            computed in a differentiable way. Note that when ``strict`` is ``False``, the result can not
            require gradients or be disconnected from the inputs.
            Defaults to ``False``.
        strict (bool, optional): If ``True``, an error will be raised when we detect that there exists an input
            such that all the outputs are independent of it. If ``False``, we return a Tensor of zeros as the
            hvp for said inputs, which is the expected mathematical value.
            Defaults to ``False``.

    Returns:
        func_output (tuple of Tensors or Tensor): output of ``func(inputs)``
        hvp (tuple of Tensors or Tensor): result of the dot product with the same shape
            as the inputs.

    Example::

        >>> def pow_reducer(x):
        ...   return x.pow(3).sum()
        >>> inputs = torch.rand(2, 2)
        >>> v = torch.ones(2, 2)
        >>> hvp(pow_reducer, inputs, v)
        (tensor(0.1448),
         tensor([[2.0239, 1.6456],
                 [2.4988, 1.4310]]))

        >>> hvp(pow_reducer, inputs, v, create_graph=True)
        (tensor(0.1448, grad_fn=<SumBackward0>),
         tensor([[2.0239, 1.6456],
                 [2.4988, 1.4310]], grad_fn=<MulBackward0>))


        >>> def pow_adder_reducer(x, y):
        ...   return (2 * x.pow(2) + 3 * y.pow(2)).sum()
        >>> inputs = (torch.rand(2), torch.rand(2))
        >>> v = (torch.zeros(2), torch.ones(2))
        >>> hvp(pow_adder_reducer, inputs, v)
        (tensor(2.3030),
         (tensor([0., 0.]),
          tensor([6., 6.])))

    Note::

        This function is significantly slower than `vhp` due to backward mode AD constraints.
        If your functions is twice continuously differentiable, then hvp = vhp.t(). So if you
        know that your function satisfies this condition, you should use vhp instead that is
        much faster with the current implementation.

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.autograd.functional.hvp', 'hvp(func, inputs, v=None, create_graph=False, strict=False)', {'_as_tuple': _as_tuple, '_grad_preprocess': _grad_preprocess, '_validate_v': _validate_v, '_check_requires_grad': _check_requires_grad, 'torch': torch, '_autograd_grad': _autograd_grad, '_fill_in_zeros': _fill_in_zeros, '_grad_postprocess': _grad_postprocess, '_tuple_postprocess': _tuple_postprocess, 'func': func, 'inputs': inputs, 'v': v, 'create_graph': create_graph, 'strict': strict}, 2)

