from functools import reduce

def maybe_view(tensor, size, check_same_size=True):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.autograd._functions.utils.maybe_view', 'maybe_view(tensor, size, check_same_size=True)', {'tensor': tensor, 'size': size, 'check_same_size': check_same_size}, 1)

def maybe_unexpand(tensor, old_size, check_same_size=True):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.autograd._functions.utils.maybe_unexpand', 'maybe_unexpand(tensor, old_size, check_same_size=True)', {'tensor': tensor, 'old_size': old_size, 'check_same_size': check_same_size}, 1)

def check_onnx_broadcast(dims1, dims2):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.autograd._functions.utils.check_onnx_broadcast', 'check_onnx_broadcast(dims1, dims2)', {'reduce': reduce, 'dims1': dims1, 'dims2': dims2}, 1)

