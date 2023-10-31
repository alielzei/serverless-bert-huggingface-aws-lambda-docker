import torch.cuda
try:
    from torch._C import _cudnn
except ImportError:
    _cudnn = None

def get_cudnn_mode(mode):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.backends.cudnn.rnn.get_cudnn_mode', 'get_cudnn_mode(mode)', {'_cudnn': _cudnn, 'mode': mode}, 1)


class Unserializable(object):
    
    def __init__(self, inner):
        self.inner = inner
    
    def get(self):
        return self.inner
    
    def __getstate__(self):
        return '<unserializable>'
    
    def __setstate__(self, state):
        self.inner = None


def init_dropout_state(dropout, train, dropout_seed, dropout_state):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.backends.cudnn.rnn.init_dropout_state', 'init_dropout_state(dropout, train, dropout_seed, dropout_state)', {'torch': torch, 'Unserializable': Unserializable, 'dropout': dropout, 'train': train, 'dropout_seed': dropout_seed, 'dropout_state': dropout_state}, 1)

