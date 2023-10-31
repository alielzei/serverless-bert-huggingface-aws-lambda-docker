import torch

def parameters_to_vector(parameters):
    """Convert parameters to one vector

    Arguments:
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.

    Returns:
        The parameters represented by a single vector
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.utils.convert_parameters.parameters_to_vector', 'parameters_to_vector(parameters)', {'_check_param_device': _check_param_device, 'torch': torch, 'parameters': parameters}, 1)

def vector_to_parameters(vec, parameters):
    """Convert one vector to the parameters

    Arguments:
        vec (Tensor): a single vector represents the parameters of a model.
        parameters (Iterable[Tensor]): an iterator of Tensors that are the
            parameters of a model.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.nn.utils.convert_parameters.vector_to_parameters', 'vector_to_parameters(vec, parameters)', {'torch': torch, '_check_param_device': _check_param_device, 'vec': vec, 'parameters': parameters}, 0)

def _check_param_device(param, old_param_device):
    """This helper function is to check if the parameters are located
    in the same device. Currently, the conversion between model parameters
    and single vector form is not supported for multiple allocations,
    e.g. parameters in different GPUs, or mixture of CPU/GPU.

    Arguments:
        param ([Tensor]): a Tensor of a parameter of a model
        old_param_device (int): the device where the first parameter of a
                                model is allocated.

    Returns:
        old_param_device (int): report device for the first time
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.utils.convert_parameters._check_param_device', '_check_param_device(param, old_param_device)', {'param': param, 'old_param_device': old_param_device}, 1)

