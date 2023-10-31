import torch

def _quantize_weight(float_wt, observer):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.quantized.modules.utils._quantize_weight', '_quantize_weight(float_wt, observer)', {'torch': torch, 'float_wt': float_wt, 'observer': observer}, 1)

