import math
import tensorflow as tf

def gelu(x):
    """Gaussian Error Linear Unit.
    Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.activations_tf.gelu', 'gelu(x)', {'tf': tf, 'x': x}, 1)

def gelu_new(x):
    """Gaussian Error Linear Unit.
    This is a smoother version of the GELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
        x: float Tensor to perform activation.
    Returns:
        `x` with the GELU activation applied.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.activations_tf.gelu_new', 'gelu_new(x)', {'tf': tf, 'math': math, 'x': x}, 1)

def mish(x):
    x = tf.convert_to_tensor(x)
    return x * tf.tanh(tf.math.softplus(x))

def gelu_fast(x):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.activations_tf.gelu_fast', 'gelu_fast(x)', {'tf': tf, 'x': x}, 1)
ACT2FN = {'gelu': tf.keras.layers.Activation(gelu), 'relu': tf.keras.activations.relu, 'swish': tf.keras.activations.swish, 'gelu_new': tf.keras.layers.Activation(gelu_new), 'mish': tf.keras.layers.Activation(mish), 'tanh': tf.keras.activations.tanh, 'gelu_fast': tf.keras.layers.Activation(gelu_fast)}

def get_tf_activation(activation_string):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.activations_tf.get_tf_activation', 'get_tf_activation(activation_string)', {'ACT2FN': ACT2FN, 'activation_string': activation_string}, 1)

