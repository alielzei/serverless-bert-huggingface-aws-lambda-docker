from __future__ import absolute_import, division, print_function, unicode_literals
import torch


class MkldnnLinear(torch.jit.ScriptModule):
    
    def __init__(self, dense_module):
        super(MkldnnLinear, self).__init__()
        self.register_buffer('weight', dense_module.weight.to_mkldnn())
        if dense_module.bias is not None:
            self.register_buffer('bias', dense_module.bias.to_mkldnn())
        else:
            self.register_buffer('bias', torch.zeros([dense_module.weight.size(0)], dtype=torch.float).to_mkldnn())
    
    @torch.jit.script_method
    def __getstate__(self):
        return (self.weight.to_dense(), self.bias.to_dense(), self.training)
    
    @torch.jit.script_method
    def __setstate__(self, state):
        self.weight = state[0].to_mkldnn()
        self.bias = state[1].to_mkldnn()
        self.training = state[2]
    
    @torch.jit.script_method
    def forward(self, x):
        x_mkldnn = (x if x.is_mkldnn else x.to_mkldnn())
        y_mkldnn = torch._C._nn.mkldnn_linear(x_mkldnn, self.weight, self.bias)
        y = (y_mkldnn if x.is_mkldnn else y_mkldnn.to_dense())
        return y



class MkldnnConv2d(torch.jit.ScriptModule):
    __constants__ = ['stride', 'padding', 'dilation', 'groups']
    
    def __init__(self, dense_module):
        super(MkldnnConv2d, self).__init__()
        self.stride = dense_module.stride
        self.padding = dense_module.padding
        self.dilation = dense_module.dilation
        self.groups = dense_module.groups
        self.register_buffer('weight', torch._C._nn.mkldnn_reorder_conv2d_weight(dense_module.weight.to_mkldnn(), self.padding, self.stride, self.dilation, self.groups))
        if dense_module.bias is not None:
            self.register_buffer('bias', dense_module.bias.to_mkldnn())
        else:
            self.register_buffer('bias', torch.zeros([dense_module.weight.size(0)], dtype=torch.float).to_mkldnn())
    
    @torch.jit.script_method
    def __getstate__(self):
        return (self.weight.to_dense(), self.bias.to_dense(), self.training)
    
    @torch.jit.script_method
    def __setstate__(self, state):
        self.weight = torch._C._nn.mkldnn_reorder_conv2d_weight(state[0].to_mkldnn(), self.padding, self.stride, self.dilation, self.groups)
        self.bias = state[1].to_mkldnn()
        self.training = state[2]
    
    @torch.jit.script_method
    def forward(self, x):
        return torch.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)



class MkldnnBatchNorm2d(torch.jit.ScriptModule):
    __constants__ = ['exponential_average_factor', 'eps']
    
    def __init__(self, dense_module):
        super(MkldnnBatchNorm2d, self).__init__()
        assert not dense_module.training
        assert dense_module.track_running_stats
        assert dense_module.affine
        if dense_module.momentum is None:
            self.exponential_average_factor = 0.0
        else:
            self.exponential_average_factor = dense_module.momentum
        self.eps = dense_module.eps
        self.register_buffer('weight', dense_module.weight.to_mkldnn())
        self.register_buffer('bias', dense_module.bias.to_mkldnn())
        self.register_buffer('running_mean', dense_module.running_mean.to_mkldnn())
        self.register_buffer('running_var', dense_module.running_var.to_mkldnn())
    
    @torch.jit.script_method
    def __getstate__(self):
        weight = self.weight.to_dense()
        bias = self.bias.to_dense()
        running_mean = self.running_mean.to_dense()
        running_var = self.running_var.to_dense()
        return (weight, bias, running_mean, running_var, self.training)
    
    @torch.jit.script_method
    def __setstate__(self, state):
        self.weight = state[0].to_mkldnn()
        self.bias = state[1].to_mkldnn()
        self.running_mean = state[2].to_mkldnn()
        self.running_var = state[3].to_mkldnn()
        self.training = state[4]
    
    @torch.jit.script_method
    def forward(self, x):
        return torch.batch_norm(x, self.weight, self.bias, self.running_mean, self.running_var, False, self.exponential_average_factor, self.eps, False)


def to_mkldnn(module):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.mkldnn.to_mkldnn', 'to_mkldnn(module)', {'torch': torch, 'MkldnnLinear': MkldnnLinear, 'MkldnnConv2d': MkldnnConv2d, 'MkldnnBatchNorm2d': MkldnnBatchNorm2d, 'module': module}, 1)

