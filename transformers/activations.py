import math
import torch
import torch.nn.functional as F
from .utils import logging
logger = logging.get_logger(__name__)

def swish(x):
    return x * torch.sigmoid(x)

def _gelu_python(x):
    """Original Implementation of the gelu activation function in Google Bert repo when initially created.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    This is now written in C in torch.nn.functional
    Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def gelu_new(x):
    """Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
    Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
if torch.__version__ < '1.4.0':
    gelu = _gelu_python
else:
    gelu = F.gelu

def gelu_fast(x):
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))

def mish(x):
    return x * torch.tanh(torch.nn.functional.softplus(x))

def linear_act(x):
    return x
ACT2FN = {'relu': F.relu, 'swish': swish, 'gelu': gelu, 'tanh': torch.tanh, 'gelu_new': gelu_new, 'gelu_fast': gelu_fast, 'mish': mish, 'linear': linear_act, 'sigmoid': torch.sigmoid}

def get_activation(activation_string):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.activations.get_activation', 'get_activation(activation_string)', {'ACT2FN': ACT2FN, 'activation_string': activation_string}, 1)

