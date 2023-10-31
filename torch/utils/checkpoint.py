from __future__ import absolute_import, division, print_function, unicode_literals
import torch
import warnings

def detach_variable(inputs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.checkpoint.detach_variable', 'detach_variable(inputs)', {'torch': torch, 'inputs': inputs}, 1)

def check_backward_validity(inputs):
    if not any((inp.requires_grad for inp in inputs if isinstance(inp, torch.Tensor))):
        warnings.warn('None of the inputs have requires_grad=True. Gradients will be None')

def get_device_states(*args):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.checkpoint.get_device_states', 'get_device_states(*args)', {'torch': torch, 'args': args}, 2)

def set_device_states(devices, states):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.utils.checkpoint.set_device_states', 'set_device_states(devices, states)', {'torch': torch, 'devices': devices, 'states': states}, 0)


class CheckpointFunction(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, run_function, preserve_rng_state, *args):
        check_backward_validity(args)
        ctx.run_function = run_function
        ctx.preserve_rng_state = preserve_rng_state
        if preserve_rng_state:
            ctx.fwd_cpu_state = torch.get_rng_state()
            ctx.had_cuda_in_fwd = False
            if torch.cuda._initialized:
                ctx.had_cuda_in_fwd = True
                (ctx.fwd_gpu_devices, ctx.fwd_gpu_states) = get_device_states(*args)
        ctx.save_for_backward(*args)
        with torch.no_grad():
            outputs = run_function(*args)
        return outputs
    
    @staticmethod
    def backward(ctx, *args):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError('Checkpointing is not compatible with .grad(), please use .backward() if possible')
        inputs = ctx.saved_tensors
        rng_devices = []
        if (ctx.preserve_rng_state and ctx.had_cuda_in_fwd):
            rng_devices = ctx.fwd_gpu_devices
        with torch.random.fork_rng(devices=rng_devices, enabled=ctx.preserve_rng_state):
            if ctx.preserve_rng_state:
                torch.set_rng_state(ctx.fwd_cpu_state)
                if ctx.had_cuda_in_fwd:
                    set_device_states(ctx.fwd_gpu_devices, ctx.fwd_gpu_states)
            detached_inputs = detach_variable(inputs)
            with torch.enable_grad():
                outputs = ctx.run_function(*detached_inputs)
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs, )
        torch.autograd.backward(outputs, args)
        grads = tuple(((inp.grad if isinstance(inp, torch.Tensor) else inp) for inp in detached_inputs))
        return (None, None) + grads


def checkpoint(function, *args, **kwargs):
    """Checkpoint a model or part of the model

    Checkpointing works by trading compute for memory. Rather than storing all
    intermediate activations of the entire computation graph for computing
    backward, the checkpointed part does **not** save intermediate activations,
    and instead recomputes them in backward pass. It can be applied on any part
    of a model.

    Specifically, in the forward pass, :attr:`function` will run in
    :func:`torch.no_grad` manner, i.e., not storing the intermediate
    activations. Instead, the forward pass saves the inputs tuple and the
    :attr:`function` parameter. In the backwards pass, the saved inputs and
    :attr:`function` is retrieved, and the forward pass is computed on
    :attr:`function` again, now tracking the intermediate activations, and then
    the gradients are calculated using these activation values.

    .. warning::
        Checkpointing doesn't work with :func:`torch.autograd.grad`, but only
        with :func:`torch.autograd.backward`.

    .. warning::
        If :attr:`function` invocation during backward does anything different
        than the one during forward, e.g., due to some global variable, the
        checkpointed version won't be equivalent, and unfortunately it can't be
        detected.

    .. warning:
        At least one of the inputs needs to have :code:`requires_grad=True` if
        grads are needed for model inputs, otherwise the checkpointed part of the
        model won't have gradients.

    Args:
        function: describes what to run in the forward pass of the model or
            part of the model. It should also know how to handle the inputs
            passed as the tuple. For example, in LSTM, if user passes
            ``(activation, hidden)``, :attr:`function` should correctly use the
            first input as ``activation`` and the second input as ``hidden``
        preserve_rng_state(bool, optional, default=True):  Omit stashing and restoring
            the RNG state during each checkpoint.
        args: tuple containing inputs to the :attr:`function`

    Returns:
        Output of running :attr:`function` on :attr:`*args`
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.checkpoint.checkpoint', 'checkpoint(function, *args, **kwargs)', {'CheckpointFunction': CheckpointFunction, 'function': function, 'args': args, 'kwargs': kwargs}, 1)

def checkpoint_sequential(functions, segments, input, **kwargs):
    """A helper function for checkpointing sequential models.

    Sequential models execute a list of modules/functions in order
    (sequentially). Therefore, we can divide such a model in various segments
    and checkpoint each segment. All segments except the last will run in
    :func:`torch.no_grad` manner, i.e., not storing the intermediate
    activations. The inputs of each checkpointed segment will be saved for
    re-running the segment in the backward pass.

    See :func:`~torch.utils.checkpoint.checkpoint` on how checkpointing works.

    .. warning::
        Checkpointing doesn't work with :func:`torch.autograd.grad`, but only
        with :func:`torch.autograd.backward`.

    .. warning:
        At least one of the inputs needs to have :code:`requires_grad=True` if
        grads are needed for model inputs, otherwise the checkpointed part of the
        model won't have gradients.

    .. warning:
        Since PyTorch 1.4, it allows only one Tensor as the input and
        intermediate outputs, just like :class:`torch.nn.Sequential`.

    Args:
        functions: A :class:`torch.nn.Sequential` or the list of modules or
            functions (comprising the model) to run sequentially.
        segments: Number of chunks to create in the model
        input: A Tensor that is input to :attr:`functions`
        preserve_rng_state(bool, optional, default=True):  Omit stashing and restoring
            the RNG state during each checkpoint.

    Returns:
        Output of running :attr:`functions` sequentially on :attr:`*inputs`

    Example:
        >>> model = nn.Sequential(...)
        >>> input_var = checkpoint_sequential(model, chunks, input_var)
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.checkpoint.checkpoint_sequential', 'checkpoint_sequential(functions, segments, input, **kwargs)', {'torch': torch, 'checkpoint': checkpoint, 'functions': functions, 'segments': segments, 'input': input, 'kwargs': kwargs}, 1)

