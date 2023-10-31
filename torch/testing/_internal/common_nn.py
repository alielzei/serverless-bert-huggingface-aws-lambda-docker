import math
import sys
import tempfile
import unittest
from copy import deepcopy
from functools import reduce
from itertools import product
from operator import mul
from math import pi
import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import _Reduction
from torch.testing._internal.common_utils import TestCase, to_gpu, freeze_rng_state, is_iterable, TEST_WITH_ROCM, _assertGradAndGradgradChecks
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.autograd.gradcheck import get_numerical_jacobian, iter_tensors
from torch.autograd import Variable
import torch.backends.cudnn
if sys.version_info[:2] == (3, 3):
    TemporaryFile = tempfile.NamedTemporaryFile
else:
    TemporaryFile = tempfile.TemporaryFile
PRECISION = 1e-05

def get_reduction(m):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_nn.get_reduction', 'get_reduction(m)', {'_Reduction': _Reduction, 'm': m}, 1)

def get_weight(m):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_nn.get_weight', 'get_weight(m)', {'m': m}, 1)
module_tests = [dict(module_name='Linear', constructor_args=(10, 8), cpp_constructor_args='torch::nn::LinearOptions(10, 8)', input_size=(4, 10), reference_fn=lambda i, p, _: torch.mm(i, p[0].t()) + p[1].view(1, -1).expand(4, 8)), dict(module_name='Linear', constructor_args=(10, 8, False), cpp_constructor_args='torch::nn::LinearOptions(10, 8).bias(false)', input_size=(4, 10), desc='no_bias', reference_fn=lambda i, p, _: torch.mm(i, p[0].t())), dict(module_name='Threshold', constructor_args=(2.0, 1.0), cpp_constructor_args='torch::nn::ThresholdOptions(2., 1.)', input_size=(2, 3, 4, 5), check_inplace=True, desc='threshold_value'), dict(module_name='Threshold', constructor_args=(2.0, 10.0), cpp_constructor_args='torch::nn::ThresholdOptions(2., 10.)', input_size=(2, 3, 4, 5), desc='large_value'), dict(module_name='ReLU', input_size=(2, 3, 4, 5), check_inplace=True), dict(module_name='ReLU6', input_size=(2, 3, 4, 5), check_inplace=True), dict(module_name='RReLU', input_size=(1, 2, 2), test_cuda=False), dict(module_name='RReLU', constructor_args=(0.1, 0.9), cpp_constructor_args='torch::nn::RReLUOptions().lower(0.1).upper(0.9)', input_size=(4, 4, 5), desc='with_up_down', test_cuda=False), dict(module_name='Hardtanh', input_size=(3, 2, 5), reference_fn=lambda i, *_: i.clamp(-1, 1)), dict(module_name='Sigmoid', input_size=(2, 3, 4, 5)), dict(module_name='Tanh', input_size=(2, 3, 4, 5)), dict(module_name='Flatten', input_size=(2, 3, 4, 5), reference_fn=lambda i, *_: torch.flatten(i, 1)), dict(module_name='Softmax', constructor_args=(1, ), cpp_constructor_args='torch::nn::SoftmaxOptions(1)', input_size=(10, 20), reference_fn=lambda i, *_: torch.exp(i).div(torch.exp(i).sum(1, True).expand(10, 20))), dict(module_name='Softmax2d', input_size=(1, 3, 10, 20), reference_fn=lambda i, *_: torch.exp(i).div(torch.exp(i).sum(1, False))), dict(module_name='LogSoftmax', constructor_args=(1, ), cpp_constructor_args='torch::nn::LogSoftmaxOptions(1)', input_size=(10, 20), reference_fn=lambda i, *_: torch.exp(i).div_(torch.exp(i).sum(1, True).expand(10, 20)).log_()), dict(module_name='LogSoftmax', constructor_args=(1, ), cpp_constructor_args='torch::nn::LogSoftmaxOptions(1)', input_size=(1, 3, 10, 20), reference_fn=lambda i, *_: torch.exp(i).div_(torch.exp(i).sum(1, False)).log_(), desc='multiparam'), dict(module_name='ELU', constructor_args=(2.0, ), cpp_constructor_args='torch::nn::ELUOptions().alpha(2.)', input_size=(3, 2, 5), reference_fn=lambda x, *_: torch.where(x >= 0, x, 2 * (x.exp() - 1))), dict(module_name='Hardshrink', constructor_args=(2.0, ), cpp_constructor_args='torch::nn::HardshrinkOptions(2.)', input_size=(4, 3, 2, 4)), dict(module_name='LeakyReLU', input_size=(3, 2, 5), check_inplace=True), dict(module_name='LeakyReLU', constructor_args=(0.5, ), cpp_constructor_args='torch::nn::LeakyReLUOptions().negative_slope(0.5)', input_size=(3, 2, 5), check_inplace=True, desc='with_negval'), dict(module_name='LogSigmoid', input_size=(2, 3, 4), reference_fn=lambda i, *_: i.sigmoid().log()), dict(module_name='Softplus', input_size=(10, 20), reference_fn=lambda i, *_: torch.log(1 + torch.exp(i))), dict(module_name='Softplus', constructor_args=(2, ), cpp_constructor_args='torch::nn::SoftplusOptions().beta(2)', input_size=(10, 20), reference_fn=lambda i, *_: 1.0 / 2.0 * torch.log(1 + torch.exp(2 * i)), desc='beta'), dict(module_name='Softplus', constructor_args=(2, -100), cpp_constructor_args='torch::nn::SoftplusOptions().beta(2).threshold(-100)', input_size=(10, 20), reference_fn=lambda i, *_: (i * 2 > -100).type_as(i) * i + (i * 2 <= -100).type_as(i) * 1.0 / 2.0 * torch.log(1 + torch.exp(2 * i)), desc='beta_threshold'), dict(module_name='Softshrink', input_size=(3, 2, 5)), dict(module_name='Softshrink', constructor_args=(1, ), cpp_constructor_args='torch::nn::SoftshrinkOptions(1)', input_size=(3, 2, 5), desc='lambda'), dict(module_name='CrossMapLRN2d', constructor_args=(5, 0.005, 0.001, 2), cpp_constructor_args='torch::nn::CrossMapLRN2dOptions(5).alpha(5e-3).beta(1e-3).k(2)', input_size=(2, 3, 6, 6), check_gradgrad=False), dict(module_name='PReLU', input_size=(2, 3, 4), reference_fn=lambda i, p, _: torch.clamp(i, min=0) + torch.clamp(i, max=0) * p[0][0], desc='1d'), dict(module_name='PReLU', constructor_args=(3, ), cpp_constructor_args='torch::nn::PReLUOptions().num_parameters(3)', input_size=(2, 3, 4), desc='1d_multiparam', reference_fn=lambda i, p, _: torch.clamp(i, min=0) + torch.clamp(i, max=0) * p[0][0]), dict(module_name='PReLU', input_size=(2, 3, 4, 5), desc='2d', reference_fn=lambda i, p, _: torch.clamp(i, min=0) + torch.clamp(i, max=0) * p[0][0]), dict(module_name='PReLU', constructor_args=(3, ), cpp_constructor_args='torch::nn::PReLUOptions().num_parameters(3)', input_size=(2, 3, 4, 5), desc='2d_multiparam', reference_fn=lambda i, p, _: torch.clamp(i, min=0) + torch.clamp(i, max=0) * p[0][0]), dict(module_name='PReLU', input_size=(2, 3, 4, 5, 6), reference_fn=lambda i, p, _: torch.clamp(i, min=0) + torch.clamp(i, max=0) * p[0][0], desc='3d'), dict(module_name='PReLU', constructor_args=(3, ), cpp_constructor_args='torch::nn::PReLUOptions().num_parameters(3)', input_size=(2, 3, 4, 5, 6), desc='3d_multiparam', reference_fn=lambda i, p, _: torch.clamp(i, min=0) + torch.clamp(i, max=0) * p[0][0]), dict(module_name='Softsign', input_size=(3, 2, 5), reference_fn=lambda i, *_: i.div(1 + torch.abs(i))), dict(module_name='Softmin', constructor_args=(1, ), cpp_constructor_args='torch::nn::SoftminOptions(1)', input_size=(10, 20)), dict(module_name='Softmin', constructor_args=(1, ), cpp_constructor_args='torch::nn::SoftminOptions(1)', input_size=(2, 3, 5, 10), desc='multidim'), dict(module_name='Tanhshrink', input_size=(2, 3, 4, 5))]

def _rand_tensor_non_equal(*size):
    total = reduce(mul, size, 1)
    return torch.randperm(total).view(*size).double()

def wrap_functional(fn, **kwargs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_nn.wrap_functional', 'wrap_functional(fn, **kwargs)', {'nn': nn, 'fn': fn, 'kwargs': kwargs}, 1)

def poissonnllloss_no_reduce_test():
    t = torch.randn(10, 10)
    return dict(fullname='PoissonNLLLoss_no_reduce', constructor=wrap_functional(lambda i: F.poisson_nll_loss(i, t.type_as(i), reduction='none')), cpp_function_call='F::poisson_nll_loss(i, t.to(i.options()), F::PoissonNLLLossFuncOptions().reduction(torch::kNone))', input_fn=lambda: torch.rand(10, 10), cpp_var_map={'i': '_get_input()', 't': t}, reference_fn=lambda i, *_: i.exp() - t.mul(i), pickle=False)

def bceloss_no_reduce_test():
    t = Variable(torch.randn(15, 10).gt(0).double())
    return dict(fullname='BCELoss_no_reduce', constructor=wrap_functional(lambda i: F.binary_cross_entropy(i, t.type_as(i), reduction='none')), cpp_function_call='F::binary_cross_entropy(i, t.to(i.options()), F::BinaryCrossEntropyFuncOptions().reduction(torch::kNone))', input_fn=lambda: torch.rand(15, 10).clamp_(0.028, 1 - 0.028), cpp_var_map={'i': '_get_input()', 't': t}, reference_fn=lambda i, *_: -(t * i.log() + (1 - t) * (1 - i).log()), pickle=False, precision=0.0007)

def bceloss_no_reduce_scalar_test():
    t = torch.randn(()).gt(0).double()
    return dict(fullname='BCELoss_no_reduce_scalar', constructor=wrap_functional(lambda i: F.binary_cross_entropy(i, t.type_as(i), reduction='none')), cpp_function_call='F::binary_cross_entropy(i, t.to(i.options()), F::BinaryCrossEntropyFuncOptions().reduction(torch::kNone))', input_fn=lambda: torch.rand(()).clamp_(0.028, 1 - 0.028), cpp_var_map={'i': '_get_input()', 't': t}, reference_fn=lambda i, *_: -(t * i.log() + (1 - t) * (1 - i).log()), pickle=False)

def bceloss_weights_no_reduce_test():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_nn.bceloss_weights_no_reduce_test', 'bceloss_weights_no_reduce_test()', {'Variable': Variable, 'torch': torch, 'wrap_functional': wrap_functional, 'F': F}, 1)

def bceloss_weights_no_reduce_scalar_test():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_nn.bceloss_weights_no_reduce_scalar_test', 'bceloss_weights_no_reduce_scalar_test()', {'torch': torch, 'wrap_functional': wrap_functional, 'F': F}, 1)

def bce_with_logistic_legacy_enum_test():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_nn.bce_with_logistic_legacy_enum_test', 'bce_with_logistic_legacy_enum_test()', {'Variable': Variable, 'torch': torch, 'nn': nn, 'wrap_functional': wrap_functional, 'F': F}, 1)

def bce_with_logistic_no_reduce_test():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_nn.bce_with_logistic_no_reduce_test', 'bce_with_logistic_no_reduce_test()', {'Variable': Variable, 'torch': torch, 'nn': nn, 'wrap_functional': wrap_functional, 'F': F}, 1)

def bce_with_logistic_no_reduce_scalar_test():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_nn.bce_with_logistic_no_reduce_scalar_test', 'bce_with_logistic_no_reduce_scalar_test()', {'torch': torch, 'nn': nn, 'wrap_functional': wrap_functional, 'F': F}, 1)

def kldivloss_with_target_no_reduce_test():
    i = torch.rand(10, 10).log()
    return dict(fullname='KLDivLoss_with_target_no_reduce', constructor=wrap_functional(lambda t: F.kl_div(i.type_as(t), t, reduction='none')), cpp_function_call='F::kl_div(i.to(t.options()), t, F::KLDivFuncOptions().reduction(torch::kNone))', input_fn=lambda: torch.rand(10, 10), cpp_var_map={'i': i, 't': '_get_input()'}, reference_fn=lambda t, *_: loss_reference_fns['KLDivLoss'](i.type_as(t), t, reduction='none'), pickle=False)

def kldivloss_no_reduce_test():
    t = torch.randn(10, 10)
    return dict(fullname='KLDivLoss_no_reduce', constructor=wrap_functional(lambda i: F.kl_div(i, t.type_as(i), reduction='none')), cpp_function_call='F::kl_div(i, t.to(i.options()), F::KLDivFuncOptions().reduction(torch::kNone))', input_fn=lambda: torch.rand(10, 10).log(), cpp_var_map={'i': '_get_input()', 't': t}, reference_fn=lambda i, *_: loss_reference_fns['KLDivLoss'](i, t.type_as(i), reduction='none'), pickle=False)

def kldivloss_no_reduce_scalar_test():
    t = torch.randn(())
    return dict(fullname='KLDivLoss_no_reduce_scalar', constructor=wrap_functional(lambda i: F.kl_div(i, t.type_as(i), reduction='none')), cpp_function_call='F::kl_div(i, t.to(i.options()), F::KLDivFuncOptions().reduction(torch::kNone))', input_fn=lambda: torch.rand(()).log(), cpp_var_map={'i': '_get_input()', 't': t}, reference_fn=lambda i, *_: loss_reference_fns['KLDivLoss'](i, t.type_as(i), reduction='none'), pickle=False)

def l1loss_no_reduce_test():
    t = torch.randn(2, 3, 4)
    return dict(fullname='L1Loss_no_reduce', constructor=wrap_functional(lambda i: F.l1_loss(i, t.type_as(i), reduction='none')), cpp_function_call='F::l1_loss(i, t.to(i.options()), F::L1LossFuncOptions().reduction(torch::kNone))', input_fn=lambda: torch.randn(2, 3, 4), cpp_var_map={'i': '_get_input()', 't': t}, reference_fn=lambda i, *_: (i - t.type_as(i)).abs(), pickle=False)

def l1loss_no_reduce_scalar_test():
    t = torch.randn(())
    return dict(fullname='L1Loss_no_reduce_scalar', constructor=wrap_functional(lambda i: F.l1_loss(i, t.type_as(i), reduction='none')), cpp_function_call='F::l1_loss(i, t.to(i.options()), F::L1LossFuncOptions().reduction(torch::kNone))', input_fn=lambda: torch.randn(()), cpp_var_map={'i': '_get_input()', 't': t}, reference_fn=lambda i, *_: (i - t.type_as(i)).abs(), pickle=False)

def mseloss_no_reduce_test():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_nn.mseloss_no_reduce_test', 'mseloss_no_reduce_test()', {'torch': torch, 'wrap_functional': wrap_functional, 'F': F}, 1)

def mseloss_no_reduce_scalar_test():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_nn.mseloss_no_reduce_scalar_test', 'mseloss_no_reduce_scalar_test()', {'torch': torch, 'wrap_functional': wrap_functional, 'F': F}, 1)

def nllloss_no_reduce_test():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_nn.nllloss_no_reduce_test', 'nllloss_no_reduce_test()', {'Variable': Variable, 'torch': torch, 'wrap_functional': wrap_functional, 'F': F, 'loss_reference_fns': loss_reference_fns}, 1)

def nllloss_no_reduce_ignore_index_test():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_nn.nllloss_no_reduce_ignore_index_test', 'nllloss_no_reduce_ignore_index_test()', {'Variable': Variable, 'torch': torch, 'wrap_functional': wrap_functional, 'F': F, 'loss_reference_fns': loss_reference_fns}, 1)

def nllloss_no_reduce_weights_test():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_nn.nllloss_no_reduce_weights_test', 'nllloss_no_reduce_weights_test()', {'Variable': Variable, 'torch': torch, 'wrap_functional': wrap_functional, 'F': F, 'loss_reference_fns': loss_reference_fns}, 1)

def nllloss_no_reduce_weights_ignore_index_test():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_nn.nllloss_no_reduce_weights_ignore_index_test', 'nllloss_no_reduce_weights_ignore_index_test()', {'Variable': Variable, 'torch': torch, 'wrap_functional': wrap_functional, 'F': F, 'loss_reference_fns': loss_reference_fns}, 1)

def nllloss_no_reduce_weights_ignore_index_neg_test():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_nn.nllloss_no_reduce_weights_ignore_index_neg_test', 'nllloss_no_reduce_weights_ignore_index_neg_test()', {'Variable': Variable, 'torch': torch, 'wrap_functional': wrap_functional, 'F': F, 'loss_reference_fns': loss_reference_fns}, 1)

def nllloss2d_no_reduce_test():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_nn.nllloss2d_no_reduce_test', 'nllloss2d_no_reduce_test()', {'Variable': Variable, 'torch': torch, 'wrap_functional': wrap_functional, 'F': F, 'loss_reference_fns': loss_reference_fns}, 1)

def nllloss2d_no_reduce_ignore_index_test():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_nn.nllloss2d_no_reduce_ignore_index_test', 'nllloss2d_no_reduce_ignore_index_test()', {'Variable': Variable, 'torch': torch, 'wrap_functional': wrap_functional, 'F': F, 'loss_reference_fns': loss_reference_fns}, 1)

def nllloss2d_no_reduce_weights_test():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_nn.nllloss2d_no_reduce_weights_test', 'nllloss2d_no_reduce_weights_test()', {'Variable': Variable, 'torch': torch, 'wrap_functional': wrap_functional, 'F': F, 'loss_reference_fns': loss_reference_fns}, 1)

def nlllossNd_no_reduce_test():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_nn.nlllossNd_no_reduce_test', 'nlllossNd_no_reduce_test()', {'Variable': Variable, 'torch': torch, 'wrap_functional': wrap_functional, 'F': F, 'loss_reference_fns': loss_reference_fns}, 1)

def nlllossNd_no_reduce_ignore_index_test():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_nn.nlllossNd_no_reduce_ignore_index_test', 'nlllossNd_no_reduce_ignore_index_test()', {'Variable': Variable, 'torch': torch, 'wrap_functional': wrap_functional, 'F': F, 'loss_reference_fns': loss_reference_fns}, 1)

def nlllossNd_no_reduce_weights_test():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_nn.nlllossNd_no_reduce_weights_test', 'nlllossNd_no_reduce_weights_test()', {'Variable': Variable, 'torch': torch, 'wrap_functional': wrap_functional, 'F': F, 'loss_reference_fns': loss_reference_fns}, 1)

def smoothl1loss_no_reduce_test():
    t = torch.randn(2, 3, 4)
    return dict(fullname='SmoothL1Loss_no_reduce', constructor=wrap_functional(lambda i: F.smooth_l1_loss(i, t.type_as(i), reduction='none')), cpp_function_call='F::smooth_l1_loss(\n            i, t.to(i.options()), F::SmoothL1LossFuncOptions().reduction(torch::kNone))', input_fn=lambda: torch.randn(2, 3, 4), cpp_var_map={'i': '_get_input()', 't': t}, reference_fn=lambda i, *_: loss_reference_fns['SmoothL1Loss'](i, t.type_as(i), reduction='none'), pickle=False)

def smoothl1loss_no_reduce_scalar_test():
    t = torch.randn(())
    return dict(fullname='SmoothL1Loss_no_reduce_scalar', constructor=wrap_functional(lambda i: F.smooth_l1_loss(i, t.type_as(i), reduction='none')), cpp_function_call='F::smooth_l1_loss(\n            i, t.to(i.options()), F::SmoothL1LossFuncOptions().reduction(torch::kNone))', input_fn=lambda: torch.randn(()), cpp_var_map={'i': '_get_input()', 't': t}, reference_fn=lambda i, *_: loss_reference_fns['SmoothL1Loss'](i, t.type_as(i), reduction='none'), pickle=False)

def multilabelmarginloss_0d_no_reduce_test():
    t = torch.zeros(()).long()
    return dict(fullname='MultiLabelMarginLoss_0d_no_reduce', constructor=wrap_functional(lambda i: F.multilabel_margin_loss(i, t.type_as(i).long(), reduction='none')), cpp_function_call='F::multilabel_margin_loss(\n            i, t.to(i.options()).to(torch::kLong), F::MultilabelMarginLossFuncOptions().reduction(torch::kNone))', input_fn=lambda: torch.randn(()), cpp_var_map={'i': '_get_input()', 't': t}, reference_fn=lambda i, *_: loss_reference_fns['MultiLabelMarginLoss'](i, t.data.type_as(i).long(), reduction='none'), check_sum_reduction=True, check_gradgrad=False, pickle=False)

def multilabelmarginloss_1d_no_reduce_test():
    t = Variable(torch.rand(10).mul(10).floor().long())
    return dict(fullname='MultiLabelMarginLoss_1d_no_reduce', constructor=wrap_functional(lambda i: F.multilabel_margin_loss(i, t.type_as(i).long(), reduction='none')), cpp_function_call='F::multilabel_margin_loss(\n            i, t.to(i.options()).to(torch::kLong), F::MultilabelMarginLossFuncOptions().reduction(torch::kNone))', input_fn=lambda: torch.randn(10), cpp_var_map={'i': '_get_input()', 't': t}, reference_fn=lambda i, *_: loss_reference_fns['MultiLabelMarginLoss'](i, t.data.type_as(i).long(), reduction='none'), check_sum_reduction=True, check_gradgrad=False, pickle=False)

def multilabelmarginloss_index_neg_test():
    t = Variable(torch.clamp(torch.rand(5, 10).add(-0.5).mul(20).floor().long(), min=-1))
    return dict(fullname='MultiLabelMarginLoss_index_neg', constructor=wrap_functional(lambda i: F.multilabel_margin_loss(i, t.type_as(i).long(), reduction='none')), cpp_function_call='F::multilabel_margin_loss(\n            i, t.to(i.options()).to(torch::kLong), F::MultilabelMarginLossFuncOptions().reduction(torch::kNone))', input_fn=lambda: torch.randn(5, 10), cpp_var_map={'i': '_get_input()', 't': t}, reference_fn=lambda i, *_: loss_reference_fns['MultiLabelMarginLoss'](i, t.data.type_as(i).long(), reduction='none'), check_sum_reduction=True, check_gradgrad=False, pickle=False)

def multilabelmarginloss_no_reduce_test():
    t = Variable(torch.rand(5, 10).mul(10).floor().long())
    return dict(fullname='MultiLabelMarginLoss_no_reduce', constructor=wrap_functional(lambda i: F.multilabel_margin_loss(i, t.type_as(i).long(), reduction='none')), cpp_function_call='F::multilabel_margin_loss(\n            i, t.to(i.options()).to(torch::kLong), F::MultilabelMarginLossFuncOptions().reduction(torch::kNone))', input_fn=lambda: torch.randn(5, 10), cpp_var_map={'i': '_get_input()', 't': t}, reference_fn=lambda i, *_: loss_reference_fns['MultiLabelMarginLoss'](i, t.data.type_as(i).long(), reduction='none'), check_sum_reduction=True, check_gradgrad=False, pickle=False)

def hingeembeddingloss_no_reduce_test():
    t = Variable(torch.randn(10).gt(0).double().mul_(2).sub(1))
    return dict(fullname='HingeEmbeddingLoss_no_reduce', constructor=wrap_functional(lambda i: F.hinge_embedding_loss(i, t.type_as(i), reduction='none')), cpp_function_call='F::hinge_embedding_loss(\n            i, t.to(i.options()), F::HingeEmbeddingLossFuncOptions().reduction(torch::kNone))', input_fn=lambda: torch.randn(10), cpp_var_map={'i': '_get_input()', 't': t}, reference_fn=lambda i, *_: loss_reference_fns['HingeEmbeddingLoss'](i, t.type_as(i), reduction='none'), check_sum_reduction=True, pickle=False)

def hingeembeddingloss_margin_no_reduce_test():
    t = Variable(torch.randn(10).gt(0).double().mul_(2).sub(1))
    return dict(fullname='HingeEmbeddingLoss_margin_no_reduce', constructor=wrap_functional(lambda i: F.hinge_embedding_loss(i, t.type_as(i), margin=0.5, reduction='none')), cpp_function_call='F::hinge_embedding_loss(\n            i, t.to(i.options()), F::HingeEmbeddingLossFuncOptions().margin(0.5).reduction(torch::kNone))', input_fn=lambda: torch.randn(10), cpp_var_map={'i': '_get_input()', 't': t}, reference_fn=lambda i, *_: loss_reference_fns['HingeEmbeddingLoss'](i, t.type_as(i), margin=0.5, reduction='none'), check_sum_reduction=True, pickle=False)

def softmarginloss_no_reduce_test():
    t = torch.randn(5, 5)
    return dict(fullname='SoftMarginLoss_no_reduce', constructor=wrap_functional(lambda i: F.soft_margin_loss(i, t.type_as(i), reduction='none')), cpp_function_call='F::soft_margin_loss(\n            i, t.to(i.options()), F::SoftMarginLossFuncOptions().reduction(torch::kNone))', input_fn=lambda: torch.randn(5, 5), cpp_var_map={'i': '_get_input()', 't': t}, reference_fn=lambda i, *_: loss_reference_fns['SoftMarginLoss'](i, t.type_as(i), reduction='none'), pickle=False)

def multilabelsoftmarginloss_no_reduce_test():
    t = torch.rand(5, 10).mul(2).floor()
    return dict(fullname='MultiLabelSoftMarginLoss_no_reduce', constructor=wrap_functional(lambda i: F.multilabel_soft_margin_loss(i, t.type_as(i), reduction='none')), cpp_function_call='F::multilabel_soft_margin_loss(\n            i, t.to(i.options()), F::MultilabelSoftMarginLossFuncOptions().reduction(torch::kNone))', input_fn=lambda: torch.randn(5, 10), cpp_var_map={'i': '_get_input()', 't': t}, reference_fn=lambda i, *_: (-(t * i.sigmoid().log() + (1 - t) * (-i).sigmoid().log())).sum(dim=1) / i.size(1), check_gradgrad=False, pickle=False)

def multilabelsoftmarginloss_weights_no_reduce_test():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_nn.multilabelsoftmarginloss_weights_no_reduce_test', 'multilabelsoftmarginloss_weights_no_reduce_test()', {'torch': torch, 'wrap_functional': wrap_functional, 'F': F}, 1)

def multimarginloss_no_reduce_test():
    t = torch.rand(5).mul(8).floor().long()
    return dict(fullname='MultiMarginLoss_no_reduce', constructor=wrap_functional(lambda i: F.multi_margin_loss(i, t.type_as(i).long(), reduction='none')), cpp_function_call='F::multi_margin_loss(\n            i, t.to(i.options()).to(torch::kLong), F::MultiMarginLossFuncOptions().reduction(torch::kNone))', input_fn=lambda: torch.randn(5, 10), cpp_var_map={'i': '_get_input()', 't': t}, reference_fn=lambda i, *_: loss_reference_fns['MultiMarginLoss'](i, t.data.type_as(i).long(), reduction='none'), check_sum_reduction=True, check_gradgrad=False, pickle=False)

def multimarginloss_1d_no_reduce_test():
    t = torch.rand(1).mul(8).floor().long()
    return dict(fullname='MultiMarginLoss_1d_no_reduce', constructor=wrap_functional(lambda i: F.multi_margin_loss(i, t.type_as(i).long(), reduction='none')), cpp_function_call='F::multi_margin_loss(\n            i, t.to(i.options()).to(torch::kLong), F::MultiMarginLossFuncOptions().reduction(torch::kNone))', input_fn=lambda: torch.randn(10), cpp_var_map={'i': '_get_input()', 't': t}, reference_fn=lambda i, *_: loss_reference_fns['MultiMarginLoss'](i, t.data.type_as(i).long(), reduction='none'), check_sum_reduction=True, check_gradgrad=False, pickle=False)

def multimarginloss_1d_input_0d_target_no_reduce_test():
    t = torch.rand(()).mul(8).floor().long()
    return dict(fullname='multimarginloss_1d_input_0d_target_no_reduce', constructor=wrap_functional(lambda i: F.multi_margin_loss(i, t.type_as(i).long(), reduction='none')), cpp_function_call='F::multi_margin_loss(\n            i, t.to(i.options()).to(torch::kLong), F::MultiMarginLossFuncOptions().reduction(torch::kNone))', input_fn=lambda: torch.randn(10), cpp_var_map={'i': '_get_input()', 't': t}, reference_fn=lambda i, *_: loss_reference_fns['MultiMarginLoss'](i, t.data.type_as(i).long(), reduction='none'), check_sum_reduction=True, check_gradgrad=False, pickle=False)

def multimarginloss_p_no_reduce_test():
    t = torch.rand(5).mul(8).floor().long()
    return dict(fullname='MultiMarginLoss_p_no_reduce', constructor=wrap_functional(lambda i: F.multi_margin_loss(i, t.type_as(i).long(), p=2, reduction='none')), cpp_function_call='F::multi_margin_loss(\n            i, t.to(i.options()).to(torch::kLong), F::MultiMarginLossFuncOptions().p(2).reduction(torch::kNone))', input_fn=lambda: torch.randn(5, 10).clamp_(0.01, 1 - 0.01), cpp_var_map={'i': '_get_input()', 't': t}, reference_fn=lambda i, *_: loss_reference_fns['MultiMarginLoss'](i, t.data.type_as(i).long(), p=2, reduction='none'), check_sum_reduction=True, check_gradgrad=False, pickle=False)

def multimarginloss_margin_no_reduce_test():
    t = torch.rand(5).mul(8).floor().long()
    return dict(fullname='MultiMarginLoss_margin_no_reduce', constructor=wrap_functional(lambda i: F.multi_margin_loss(i, t.type_as(i).long(), margin=0.5, reduction='none')), cpp_function_call='F::multi_margin_loss(\n            i, t.to(i.options()).to(torch::kLong),\n            F::MultiMarginLossFuncOptions().margin(0.5).reduction(torch::kNone))', input_fn=lambda: torch.randn(5, 10), cpp_var_map={'i': '_get_input()', 't': t}, reference_fn=lambda i, *_: loss_reference_fns['MultiMarginLoss'](i, t.data.type_as(i).long(), margin=0.5, reduction='none'), check_sum_reduction=True, check_gradgrad=False, pickle=False)

def multimarginloss_weights_no_reduce_test():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_nn.multimarginloss_weights_no_reduce_test', 'multimarginloss_weights_no_reduce_test()', {'torch': torch, 'wrap_functional': wrap_functional, 'F': F, 'loss_reference_fns': loss_reference_fns}, 1)

def fractional_max_pool2d_test(test_case):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_nn.fractional_max_pool2d_test', 'fractional_max_pool2d_test(test_case)', {'torch': torch, 'nn': nn, 'test_case': test_case}, 1)

def fractional_max_pool3d_test(test_case):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_nn.fractional_max_pool3d_test', 'fractional_max_pool3d_test(test_case)', {'torch': torch, 'nn': nn, 'test_case': test_case}, 1)
new_module_tests = [poissonnllloss_no_reduce_test(), bceloss_no_reduce_test(), bceloss_weights_no_reduce_test(), bce_with_logistic_legacy_enum_test(), bce_with_logistic_no_reduce_test(), bceloss_no_reduce_scalar_test(), bceloss_weights_no_reduce_scalar_test(), bce_with_logistic_no_reduce_scalar_test(), kldivloss_with_target_no_reduce_test(), kldivloss_no_reduce_test(), kldivloss_no_reduce_scalar_test(), l1loss_no_reduce_test(), l1loss_no_reduce_scalar_test(), mseloss_no_reduce_test(), mseloss_no_reduce_scalar_test(), nllloss_no_reduce_test(), nllloss_no_reduce_ignore_index_test(), nllloss_no_reduce_weights_test(), nllloss_no_reduce_weights_ignore_index_test(), nllloss_no_reduce_weights_ignore_index_neg_test(), nllloss2d_no_reduce_test(), nllloss2d_no_reduce_weights_test(), nllloss2d_no_reduce_ignore_index_test(), nlllossNd_no_reduce_test(), nlllossNd_no_reduce_weights_test(), nlllossNd_no_reduce_ignore_index_test(), smoothl1loss_no_reduce_test(), smoothl1loss_no_reduce_scalar_test(), multilabelmarginloss_0d_no_reduce_test(), multilabelmarginloss_1d_no_reduce_test(), multilabelmarginloss_index_neg_test(), multilabelmarginloss_no_reduce_test(), hingeembeddingloss_no_reduce_test(), hingeembeddingloss_margin_no_reduce_test(), softmarginloss_no_reduce_test(), multilabelsoftmarginloss_no_reduce_test(), multilabelsoftmarginloss_weights_no_reduce_test(), multimarginloss_no_reduce_test(), multimarginloss_1d_no_reduce_test(), multimarginloss_1d_input_0d_target_no_reduce_test(), multimarginloss_p_no_reduce_test(), multimarginloss_margin_no_reduce_test(), multimarginloss_weights_no_reduce_test(), fractional_max_pool2d_test('ratio'), fractional_max_pool2d_test('size'), fractional_max_pool3d_test('ratio'), fractional_max_pool3d_test('size'), fractional_max_pool3d_test('asymsize'), dict(module_name='BatchNorm1d', constructor_args=(10, ), cpp_constructor_args='torch::nn::BatchNorm1dOptions(10)', input_size=(4, 10), cudnn=True, check_eval=True, desc='affine', test_cuda=not TEST_WITH_ROCM), dict(module_name='BatchNorm1d', constructor_args=(5, ), cpp_constructor_args='torch::nn::BatchNorm1dOptions(5)', input_size=(4, 5, 3), cudnn=True, check_eval=True, desc='3d_input'), dict(module_name='BatchNorm1d', constructor_args=(10, 0.001, None), cpp_constructor_args='torch::nn::BatchNorm1dOptions(10).eps(1e-3).momentum(c10::nullopt)', input_size=(4, 10), cudnn=True, check_eval=True, desc='affine_simple_average', test_cuda=not TEST_WITH_ROCM), dict(module_name='BatchNorm1d', constructor_args=(10, 0.001, 0.3, False), cpp_constructor_args='torch::nn::BatchNorm1dOptions(10).eps(1e-3).momentum(0.3).affine(false)', input_size=(4, 10), cudnn=True, check_eval=True, desc='not_affine'), dict(module_name='BatchNorm1d', constructor_args=(10, 0.001, 0.3, True, False), cpp_constructor_args='torch::nn::BatchNorm1dOptions(10)\n                                .eps(1e-3).momentum(0.3).affine(true).track_running_stats(false)', input_size=(4, 10), cudnn=True, check_eval=True, desc='not_tracking_stats', test_cuda=not TEST_WITH_ROCM), dict(module_name='BatchNorm1d', constructor_args=(5, 0.001, 0.3, False), cpp_constructor_args='torch::nn::BatchNorm1dOptions(5).eps(1e-3).momentum(0.3).affine(false)', input_size=(4, 5, 3), cudnn=True, check_eval=True, desc='3d_input_not_affine'), dict(module_name='BatchNorm1d', constructor_args=(5, 0.001, 0.3, False), cpp_constructor_args='torch::nn::BatchNorm1dOptions(5).eps(1e-3).momentum(0.3).affine(false)', input_size=(0, 5, 9), cudnn=True, check_eval=True, desc='zero_batch'), dict(module_name='BatchNorm2d', constructor_args=(3, ), cpp_constructor_args='torch::nn::BatchNorm2dOptions(3)', input_size=(2, 3, 6, 6), cudnn=True, check_eval=True), dict(module_name='BatchNorm2d', constructor_args=(3, 0.001, None), cpp_constructor_args='torch::nn::BatchNorm2dOptions(3).eps(1e-3).momentum(c10::nullopt)', input_size=(2, 3, 6, 6), cudnn=True, check_eval=True, desc='2d_simple_average'), dict(module_name='BatchNorm2d', constructor_args=(3, 0.001, 0.8), cpp_constructor_args='torch::nn::BatchNorm2dOptions(3).eps(1e-3).momentum(0.8)', input_size=(2, 3, 6, 6), cudnn=True, check_eval=True, desc='momentum'), dict(module_name='BatchNorm2d', constructor_args=(3, 0.001, 0.8, False), cpp_constructor_args='torch::nn::BatchNorm2dOptions(3).eps(1e-3).momentum(0.8).affine(false)', input_size=(2, 3, 6, 6), cudnn=True, check_eval=True, desc='not_affine'), dict(module_name='BatchNorm2d', constructor_args=(3, 0.001, 0.8, True, False), cpp_constructor_args='torch::nn::BatchNorm2dOptions(3)\n                                .eps(1e-3).momentum(0.8).affine(true).track_running_stats(false)', input_size=(2, 3, 6, 6), cudnn=True, check_eval=True, desc='not_tracking_stats'), dict(module_name='BatchNorm2d', constructor_args=(5, 0.001, 0.3, False), cpp_constructor_args='torch::nn::BatchNorm2dOptions(5).eps(1e-3).momentum(0.3).affine(false)', input_size=(0, 5, 2, 2), cudnn=True, check_eval=True, desc='zero_batch'), dict(module_name='BatchNorm3d', constructor_args=(3, ), cpp_constructor_args='torch::nn::BatchNorm3dOptions(3)', input_size=(2, 3, 4, 4, 4), cudnn=True, check_eval=True), dict(module_name='BatchNorm3d', constructor_args=(3, 0.001, None), cpp_constructor_args='torch::nn::BatchNorm3dOptions(3).eps(1e-3).momentum(c10::nullopt)', input_size=(2, 3, 4, 4, 4), cudnn=True, check_eval=True, desc='3d_simple_average'), dict(module_name='BatchNorm3d', constructor_args=(3, 0.001, 0.7), cpp_constructor_args='torch::nn::BatchNorm3dOptions(3).eps(1e-3).momentum(0.7)', input_size=(2, 3, 4, 4, 4), cudnn=True, check_eval=True, desc='momentum'), dict(module_name='BatchNorm3d', constructor_args=(3, 0.001, 0.7, False), cpp_constructor_args='torch::nn::BatchNorm3dOptions(3).eps(1e-3).momentum(0.7).affine(false)', input_size=(2, 3, 4, 4, 4), cudnn=True, check_eval=True, desc='not_affine'), dict(module_name='BatchNorm3d', constructor_args=(3, 0.001, 0.7, True, False), cpp_constructor_args='torch::nn::BatchNorm3dOptions(3)\n                                .eps(1e-3).momentum(0.7).affine(true).track_running_stats(false)', input_size=(2, 3, 4, 4, 4), cudnn=True, check_eval=True, desc='not_tracking_stats'), dict(module_name='BatchNorm3d', constructor_args=(5, 0.001, 0.3, False), cpp_constructor_args='torch::nn::BatchNorm3dOptions(5).eps(1e-3).momentum(0.3).affine(false)', input_size=(0, 5, 2, 2, 2), cudnn=True, check_eval=True, desc='zero_batch'), dict(module_name='InstanceNorm1d', constructor_args=(3, 0.001, 0.3), cpp_constructor_args='torch::nn::InstanceNorm1dOptions(3).eps(1e-3).momentum(0.3)', input_size=(4, 3, 15), cudnn=True, check_eval=True), dict(module_name='InstanceNorm1d', constructor_args=(3, 0.001, 0.3, False, True), cpp_constructor_args='torch::nn::InstanceNorm1dOptions(3)\n                                .eps(1e-3).momentum(0.3).affine(false).track_running_stats(true)', input_size=(4, 3, 15), cudnn=True, check_eval=True, desc='tracking_stats'), dict(module_name='InstanceNorm2d', constructor_args=(3, 0.001, 0.3), cpp_constructor_args='torch::nn::InstanceNorm2dOptions(3).eps(1e-3).momentum(0.3)', input_size=(2, 3, 6, 6), cudnn=True, check_eval=True), dict(module_name='InstanceNorm2d', constructor_args=(3, 0.001, 0.3, False, True), cpp_constructor_args='torch::nn::InstanceNorm2dOptions(3)\n                                .eps(1e-3).momentum(0.3).affine(false).track_running_stats(true)', input_size=(2, 3, 6, 6), cudnn=True, check_eval=True, desc='tracking_stats'), dict(module_name='InstanceNorm3d', constructor_args=(3, 0.001, 0.3), cpp_constructor_args='torch::nn::InstanceNorm3dOptions(3).eps(1e-3).momentum(0.3)', input_size=(2, 3, 4, 4, 4), cudnn=True, check_eval=True), dict(module_name='InstanceNorm3d', constructor_args=(3, 0.001, 0.3, False, True), cpp_constructor_args='torch::nn::InstanceNorm3dOptions(3)\n                                .eps(1e-3).momentum(0.3).affine(false).track_running_stats(true)', input_size=(2, 3, 4, 4, 4), cudnn=True, check_eval=True, desc='tracking_stats'), dict(module_name='LayerNorm', constructor_args=([5], 0.001), cpp_constructor_args='torch::nn::LayerNormOptions({5}).eps(1e-3)', input_size=(4, 5, 5), cudnn=True, check_eval=True, desc='1d_elementwise_affine'), dict(module_name='LayerNorm', constructor_args=([5], 0.001, False), cpp_constructor_args='torch::nn::LayerNormOptions({5}).eps(1e-3).elementwise_affine(false)', input_size=(4, 5, 5), cudnn=True, check_eval=True, desc='1d_no_elementwise_affine'), dict(module_name='LayerNorm', constructor_args=([2, 2, 5], 0.001), cpp_constructor_args='torch::nn::LayerNormOptions({2, 2, 5}).eps(1e-3)', input_size=(4, 2, 2, 5), cudnn=True, check_eval=True, desc='3d_elementwise_affine'), dict(module_name='LayerNorm', constructor_args=([2, 2, 5], 0.001, False), cpp_constructor_args='torch::nn::LayerNormOptions({2, 2, 5}).eps(1e-3).elementwise_affine(false)', input_size=(4, 2, 2, 5), cudnn=True, check_eval=True, desc='3d_no_elementwise_affine'), dict(module_name='LayerNorm', constructor_args=([5], 0.001), cpp_constructor_args='torch::nn::LayerNormOptions({5}).eps(1e-3)', input_size=(0, 5), cudnn=True, check_eval=True, desc='1d_empty_elementwise_affine'), dict(module_name='GroupNorm', constructor_args=(3, 6, 0.001), cpp_constructor_args='torch::nn::GroupNormOptions(3, 6).eps(1e-3)', input_size=(4, 6, 5), cudnn=True, check_eval=True, desc='1d_affine'), dict(module_name='GroupNorm', constructor_args=(5, 5, 0.001, False), cpp_constructor_args='torch::nn::GroupNormOptions(5, 5).eps(1e-3).affine(false)', input_size=(4, 5, 5), cudnn=True, check_eval=True, desc='1d_no_affine_IN'), dict(module_name='GroupNorm', constructor_args=(1, 5, 0.001, False), cpp_constructor_args='torch::nn::GroupNormOptions(1, 5).eps(1e-3).affine(false)', input_size=(4, 5, 5), cudnn=True, check_eval=True, desc='1d_no_affine_LN'), dict(module_name='GroupNorm', constructor_args=(3, 6, 0.001), cpp_constructor_args='torch::nn::GroupNormOptions(3, 6).eps(1e-3)', input_size=(4, 6, 2, 3), cudnn=True, check_eval=True, desc='2d_affine'), dict(module_name='GroupNorm', constructor_args=(3, 3, 0.001, False), cpp_constructor_args='torch::nn::GroupNormOptions(3, 3).eps(1e-3).affine(false)', input_size=(4, 3, 2, 3), cudnn=True, check_eval=True, desc='2d_no_affine_IN'), dict(module_name='GroupNorm', constructor_args=(1, 3, 0.001, False), cpp_constructor_args='torch::nn::GroupNormOptions(1, 3).eps(1e-3).affine(false)', input_size=(4, 3, 2, 3), cudnn=True, check_eval=True, desc='2d_no_affine_LN'), dict(module_name='Conv1d', constructor_args=(4, 5, 3), cpp_constructor_args='torch::nn::Conv1dOptions(4, 5, 3)', input_size=(2, 4, 10), cudnn=True), dict(module_name='Conv1d', constructor_args=(4, 5, 3, 2), cpp_constructor_args='torch::nn::Conv1dOptions(4, 5, 3).stride(2)', input_size=(2, 4, 10), cudnn=True, desc='stride'), dict(module_name='Conv1d', constructor_args=(4, 5, 3, 1, 1), cpp_constructor_args='torch::nn::Conv1dOptions(4, 5, 3).stride(1).padding(1)', input_size=(2, 4, 10), cudnn=True, desc='pad1'), dict(module_name='Conv1d', constructor_args=(4, 5, 5, 1, 2), cpp_constructor_args='torch::nn::Conv1dOptions(4, 5, 5).stride(1).padding(2)', input_size=(2, 4, 10), cudnn=True, desc='pad2'), dict(module_name='Conv1d', constructor_args=(4, 4, 3, 1, 1), cpp_constructor_args='torch::nn::Conv1dOptions(4, 4, 3).stride(1).padding(1)', input_size=(1, 4, 1), cudnn=True, desc='pad1size1'), dict(module_name='Conv1d', constructor_args=(4, 4, 5, 1, 2), cpp_constructor_args='torch::nn::Conv1dOptions(4, 4, 5).stride(1).padding(2)', input_size=(1, 4, 1), cudnn=True, desc='pad2size1'), dict(module_name='Conv1d', constructor_args=(4, 5, 3), cpp_constructor_args='torch::nn::Conv1dOptions(4, 5, 3)', input_size=(0, 4, 10), cudnn=True, desc='zero_batch', test_cuda=not TEST_WITH_ROCM), dict(fullname='Conv1d_dilated', constructor=lambda: nn.Conv1d(4, 5, kernel_size=3, dilation=2), cpp_constructor_args='torch::nn::Conv1dOptions(4, 5, 3).dilation(2)', input_size=(2, 4, 10)), dict(fullname='Conv1d_groups', constructor=lambda: nn.Conv1d(4, 6, kernel_size=3, groups=2), cpp_constructor_args='torch::nn::Conv1dOptions(4, 6, 3).groups(2)', input_size=(2, 4, 6), cudnn=True), dict(fullname='ConvTranspose1d', constructor=lambda: nn.ConvTranspose1d(3, 4, kernel_size=3, stride=(3, ), padding=1, output_padding=(1, )), cpp_constructor_args='torch::nn::ConvTranspose1dOptions(3, 4, 3).stride(3).padding(1).output_padding(1)', cudnn=True, input_size=(1, 3, 7)), dict(module_name='ConvTranspose1d', constructor_args=(3, 4, 3, 2, 1, 1, 1, False), cpp_constructor_args='torch::nn::ConvTranspose1dOptions(3, 4, 3)\n                                .stride(2).padding(1).output_padding(1).groups(1).bias(false)', input_size=(1, 3, 6), cudnn=True, desc='no_bias'), dict(module_name='ConvTranspose1d', constructor_args=(3, 4, 3, 2, 1, 1, 1, True, 2), cpp_constructor_args='torch::nn::ConvTranspose1dOptions(3, 4, 3)\n                                .stride(2).padding(1).output_padding(1).groups(1).bias(true).dilation(2)', input_size=(1, 3, 6), cudnn=True, desc='dilated'), dict(fullname='ConvTranspose1d_groups', constructor=lambda: nn.ConvTranspose1d(4, 6, 3, stride=(3, ), padding=1, output_padding=(1, ), groups=2), cpp_constructor_args='torch::nn::ConvTranspose1dOptions(4, 6, 3)\n                                .stride(3).padding(1).output_padding(1).groups(2)', cudnn=True, input_size=(2, 4, 7)), dict(module_name='MaxPool1d', constructor_args=(4, ), cpp_constructor_args='torch::nn::MaxPool1dOptions(4)', input_size=(2, 10, 4)), dict(module_name='MaxPool1d', constructor_args=(4, 4), cpp_constructor_args='torch::nn::MaxPool1dOptions(4).stride(4)', input_size=(2, 10, 4), desc='stride'), dict(module_name='Conv2d', constructor_args=(3, 4, (3, 2)), cpp_constructor_args='torch::nn::Conv2dOptions(3, 4, {3, 2})', input_size=(2, 3, 7, 5), cudnn=True, check_with_long_tensor=True), dict(module_name='Conv2d', constructor_args=(3, 4, (3, 3), (2, 2)), cpp_constructor_args='torch::nn::Conv2dOptions(3, 4, {3, 3}).stride({2, 2})', input_size=(2, 3, 6, 6), cudnn=True, desc='strided', check_with_long_tensor=True), dict(module_name='Conv2d', constructor_args=(3, 4, (3, 3), (2, 2), (1, 1)), cpp_constructor_args='torch::nn::Conv2dOptions(3, 4, {3, 3}).stride({2, 2}).padding({1, 1})', input_size=(2, 3, 6, 6), cudnn=True, desc='padding', check_with_long_tensor=True), dict(module_name='Conv2d', constructor_args=(3, 2, (3, 3), (2, 2), (1, 1), (2, 2)), cpp_constructor_args='torch::nn::Conv2dOptions(3, 2, {3, 3}).stride({2, 2}).padding({1, 1}).dilation({2, 2})', input_size=(2, 3, 8, 8), cudnn=True, desc='dilated', check_with_long_tensor=True), dict(module_name='Conv2d', constructor_args=(3, 4, (3, 2), 1, 0, 1, 1, False), cpp_constructor_args='torch::nn::Conv2dOptions(3, 4, {3, 2})\n                                .stride(1).padding(0).dilation(1).groups(1).bias(false)', input_size=(2, 3, 6, 5), cudnn=True, desc='no_bias', check_with_long_tensor=True), dict(module_name='Conv2d', constructor_args=(3, 4, (3, 2)), cpp_constructor_args='torch::nn::Conv2dOptions(3, 4, {3, 2})', input_size=(0, 3, 7, 5), cudnn=True, desc='zero_batch', check_with_long_tensor=True, test_cuda=not TEST_WITH_ROCM), dict(fullname='Conv2d_groups', constructor=lambda: nn.Conv2d(4, 6, (3, 2), groups=2), cpp_constructor_args='torch::nn::Conv2dOptions(4, 6, {3, 2}).groups(2)', input_size=(2, 4, 6, 5), cudnn=True, check_with_long_tensor=True), dict(fullname='Conv2d_groups_thnn', constructor=lambda: nn.Conv2d(4, 6, (3, 2), groups=2), cpp_constructor_args='torch::nn::Conv2dOptions(4, 6, {3, 2}).groups(2)', input_size=(2, 4, 6, 5), check_with_long_tensor=True), dict(module_name='ConvTranspose2d', constructor_args=(3, 4, 3, (3, 2), 1, (1, 1)), cpp_constructor_args='torch::nn::ConvTranspose2dOptions(3, 4, 3)\n                                .stride({3, 2}).padding(1).output_padding({1, 1})', cudnn=True, input_size=(1, 3, 7, 6), check_with_long_tensor=True), dict(module_name='ConvTranspose2d', constructor_args=(3, 4, 3, (2, 3), 1, (1, 1), 1, False, (2, 2)), cpp_constructor_args='torch::nn::ConvTranspose2dOptions(3, 4, 3)\n                                .stride({2, 3})\n                                .padding(1)\n                                .output_padding({1, 1})\n                                .groups(1)\n                                .bias(false)\n                                .dilation({2, 2})', input_size=(1, 3, 6, 7), cudnn=True, desc='dilated', check_with_long_tensor=True), dict(module_name='ConvTranspose2d', constructor_args=(3, 4, 3, (2, 3), 1, (1, 1), 1, False), cpp_constructor_args='torch::nn::ConvTranspose2dOptions(3, 4, 3)\n                                .stride({2, 3}).padding(1).output_padding({1, 1}).groups(1).bias(false)', input_size=(1, 3, 6, 7), cudnn=True, desc='no_bias', check_with_long_tensor=True), dict(fullname='ConvTranspose2d_groups', constructor=lambda: nn.ConvTranspose2d(2, 4, (2, 3), groups=2), cpp_constructor_args='torch::nn::ConvTranspose2dOptions(2, 4, {2, 3}).groups(2)', input_size=(1, 2, 4, 5), cudnn=True, check_with_long_tensor=True), dict(fullname='Conv2d_depthwise', constructor=lambda: nn.Conv2d(4, 4, (3, 3), groups=4), cpp_constructor_args='torch::nn::Conv2dOptions(4, 4, {3, 3}).groups(4)', input_size=(2, 4, 6, 6)), dict(fullname='Conv2d_depthwise_with_multiplier', constructor=lambda: nn.Conv2d(4, 8, (3, 3), groups=4), cpp_constructor_args='torch::nn::Conv2dOptions(4, 8, {3, 3}).groups(4)', input_size=(2, 4, 6, 6)), dict(fullname='Conv2d_depthwise_strided', constructor=lambda: nn.Conv2d(4, 4, (3, 3), stride=(2, 2), groups=4), cpp_constructor_args='torch::nn::Conv2dOptions(4, 4, {3, 3}).stride({2, 2}).groups(4)', input_size=(2, 4, 6, 6)), dict(fullname='Conv2d_depthwise_padded', constructor=lambda: nn.Conv2d(4, 4, (3, 3), padding=(1, 1), groups=4), cpp_constructor_args='torch::nn::Conv2dOptions(4, 4, {3, 3}).padding({1, 1}).groups(4)', input_size=(2, 4, 6, 6)), dict(fullname='Conv2d_depthwise_dilated', constructor=lambda: nn.Conv2d(4, 4, (2, 2), dilation=(2, 2), groups=4), cpp_constructor_args='torch::nn::Conv2dOptions(4, 4, {2, 2}).dilation({2, 2}).groups(4)', input_size=(2, 4, 5, 5)), dict(module_name='MaxPool2d', constructor_args=((3, 3), (2, 2), (1, 1)), cpp_constructor_args='torch::nn::MaxPool2dOptions({3, 3}).stride({2, 2}).padding({1, 1})', input_size=(3, 7, 7), desc='3d_input'), dict(module_name='MaxPool2d', constructor_args=((3, 3), (2, 2), (1, 1)), cpp_constructor_args='torch::nn::MaxPool2dOptions({3, 3}).stride({2, 2}).padding({1, 1})', input_size=(1, 3, 7, 7), check_with_channels_last=True, desc='4d_input'), dict(module_name='AvgPool1d', constructor_args=(2, ), cpp_constructor_args='torch::nn::AvgPool1dOptions(2)', input_size=(2, 3, 6)), dict(module_name='AvgPool1d', constructor_args=((2, ), (2, )), cpp_constructor_args='torch::nn::AvgPool1dOptions(2).stride(2)', input_size=(2, 3, 6), desc='stride'), dict(module_name='AvgPool1d', constructor_args=(2, 2, 1), cpp_constructor_args='torch::nn::AvgPool1dOptions(2).stride(2).padding(1)', input_size=(2, 3, 6), desc='stride_pad'), dict(module_name='AvgPool2d', constructor_args=((2, 2), ), cpp_constructor_args='torch::nn::AvgPool2dOptions({2, 2})', input_size=(2, 3, 6, 6)), dict(module_name='AvgPool2d', constructor_args=((2, 2), (2, 2)), cpp_constructor_args='torch::nn::AvgPool2dOptions({2, 2}).stride({2, 2})', input_size=(2, 3, 6, 6), desc='stride'), dict(module_name='AvgPool2d', constructor_args=((2, 2), (2, 2), (1, 1)), cpp_constructor_args='torch::nn::AvgPool2dOptions({2, 2}).stride({2, 2}).padding({1, 1})', input_size=(2, 3, 6, 6), desc='stride_pad'), dict(fullname='AvgPool2d_divisor', constructor=lambda: nn.AvgPool2d((2, 2), divisor_override=1), cpp_constructor_args='torch::nn::AvgPool2dOptions({2, 2}).divisor_override(1)', input_size=(2, 3, 6, 6), check_with_long_tensor=True), dict(fullname='AvgPool2d_divisor_stride', constructor=lambda: nn.AvgPool2d((2, 2), (2, 2), divisor_override=1), cpp_constructor_args='torch::nn::AvgPool2dOptions({2, 2}).stride({2, 2}).divisor_override(1)', input_size=(2, 3, 6, 6), check_with_long_tensor=True), dict(fullname='AvgPool2d_divisor_stride_pad', constructor=lambda: nn.AvgPool2d((2, 2), (2, 2), (1, 1), divisor_override=1), cpp_constructor_args='torch::nn::AvgPool2dOptions({2, 2}).stride({2, 2}).padding({1, 1}).divisor_override(1)', input_size=(2, 3, 6, 6), check_with_long_tensor=True), dict(module_name='LPPool2d', constructor_args=(2, 2, 2), cpp_constructor_args='torch::nn::LPPool2dOptions(2, 2).stride(2)', input_size=(1, 3, 7, 7)), dict(module_name='LPPool2d', constructor_args=(1.5, 2), cpp_constructor_args='torch::nn::LPPool2dOptions(1.5, 2)', input_fn=lambda: torch.rand(1, 3, 7, 7), desc='norm'), dict(module_name='LPPool1d', constructor_args=(1.5, 2), cpp_constructor_args='torch::nn::LPPool1dOptions(1.5, 2)', input_fn=lambda: torch.rand(1, 3, 7), desc='norm'), dict(module_name='LPPool1d', constructor_args=(2, 2, 3), cpp_constructor_args='torch::nn::LPPool1dOptions(2, 2).stride(3)', input_size=(1, 3, 7)), dict(module_name='LocalResponseNorm', constructor_args=(3, ), cpp_constructor_args='torch::nn::LocalResponseNormOptions(3)', input_size=(1, 5, 7), desc='1d'), dict(module_name='LocalResponseNorm', constructor_args=(2, ), cpp_constructor_args='torch::nn::LocalResponseNormOptions(2)', input_size=(1, 5, 7, 7), desc='2d_uneven_pad'), dict(module_name='LocalResponseNorm', constructor_args=(1, 1.0, 0.5, 2.0), cpp_constructor_args='torch::nn::LocalResponseNormOptions(1).alpha(1.).beta(0.5).k(2.)', input_size=(1, 5, 7, 7, 7), desc='3d_custom_params'), dict(module_name='ReflectionPad1d', constructor_args=((1, 2), ), cpp_constructor_args='torch::nn::ReflectionPad1dOptions({1, 2})', input_size=(2, 3, 8)), dict(module_name='ReflectionPad2d', constructor_args=((1, 2, 3, 4), ), cpp_constructor_args='torch::nn::ReflectionPad2dOptions({1, 2, 3, 4})', input_size=(2, 3, 8, 8)), dict(module_name='ReplicationPad1d', constructor_args=((1, 2), ), cpp_constructor_args='torch::nn::ReplicationPad1dOptions({1, 2})', input_size=(2, 3, 4)), dict(module_name='ReplicationPad2d', constructor_args=((1, 2, 3, 4), ), cpp_constructor_args='torch::nn::ReplicationPad2dOptions({1, 2, 3, 4})', input_size=(2, 3, 4, 4)), dict(module_name='ZeroPad2d', constructor_args=((1, 2, 3, 4), ), cpp_constructor_args='torch::nn::ZeroPad2dOptions({1, 2, 3, 4})', input_size=(2, 3, 4, 4)), dict(module_name='ZeroPad2d', constructor_args=((-1, -1, -1, -2), ), cpp_constructor_args='torch::nn::ZeroPad2dOptions({-1, -1, -1, -2})', input_size=(2, 3, 4, 4), desc='negative_dims'), dict(module_name='ConstantPad1d', constructor_args=((1, 2), 2.0), cpp_constructor_args='torch::nn::ConstantPad1dOptions({1, 2}, 2.)', input_size=(2, 3, 4)), dict(module_name='ConstantPad2d', constructor_args=((1, 2, 3, 4), 2.0), cpp_constructor_args='torch::nn::ConstantPad2dOptions({1, 2, 3, 4}, 2.)', input_size=(2, 3, 4, 4)), dict(module_name='ConstantPad3d', constructor_args=((1, 2, 3, 4, 1, 0), 2.0), cpp_constructor_args='torch::nn::ConstantPad3dOptions({1, 2, 3, 4, 1, 0}, 2.)', input_size=(2, 3, 4, 4, 5)), dict(module_name='Conv3d', constructor_args=(3, 4, (2, 3, 4)), cpp_constructor_args='torch::nn::Conv3dOptions(3, 4, {2, 3, 4})', input_size=(2, 3, 3, 4, 5), cudnn=True, check_with_long_tensor=True), dict(module_name='Conv3d', constructor_args=(3, 4, (2, 3, 4), 1, 0, 1, 1, False), cpp_constructor_args='torch::nn::Conv3dOptions(3, 4, {2, 3, 4})\n                                .stride(1).padding(0).dilation(1).groups(1).bias(false)', input_size=(2, 3, 3, 4, 5), cudnn=True, desc='no_bias', check_with_long_tensor=True), dict(module_name='Conv3d', constructor_args=(3, 4, 2, 2), cpp_constructor_args='torch::nn::Conv3dOptions(3, 4, 2).stride(2)', input_size=(2, 3, 5, 5, 5), cudnn=True, desc='stride', check_with_long_tensor=True), dict(module_name='Conv3d', constructor_args=(3, 4, 2, 2, 1), cpp_constructor_args='torch::nn::Conv3dOptions(3, 4, 2).stride(2).padding(1)', input_size=(2, 3, 5, 5, 5), cudnn=True, desc='stride_padding', check_with_long_tensor=True), dict(module_name='Conv3d', constructor_args=(3, 4, (2, 3, 4)), cpp_constructor_args='torch::nn::Conv3dOptions(3, 4, {2, 3, 4})', input_size=(0, 3, 3, 4, 5), cudnn=True, check_with_long_tensor=True, desc='zero_batch', test_cuda=not TEST_WITH_ROCM), dict(fullname='Conv3d_groups', constructor=lambda: nn.Conv3d(4, 6, kernel_size=3, groups=2), cpp_constructor_args='torch::nn::Conv3dOptions(4, 6, 3).groups(2)', input_size=(2, 4, 4, 5, 4), cudnn=True, check_with_long_tensor=True), dict(fullname='Conv3d_dilated', constructor=lambda: nn.Conv3d(3, 4, kernel_size=2, dilation=2), cpp_constructor_args='torch::nn::Conv3dOptions(3, 4, 2).dilation(2)', input_size=(2, 3, 5, 5, 5)), dict(fullname='Conv3d_dilated_strided', constructor=lambda: nn.Conv3d(3, 4, kernel_size=2, dilation=2, stride=2), cpp_constructor_args='torch::nn::Conv3dOptions(3, 4, 2).dilation(2).stride(2)', input_size=(2, 3, 5, 5, 5)), dict(module_name='ConvTranspose3d', constructor_args=(2, 3, (2, 3, 2)), cpp_constructor_args='torch::nn::ConvTranspose3dOptions(2, 3, {2, 3, 2})', cudnn=True, input_size=(1, 2, 4, 5, 4)), dict(module_name='ConvTranspose3d', constructor_args=(2, 3, (2, 3, 2), 1, 0, 0, 1, True, (2, 2, 2)), cpp_constructor_args='torch::nn::ConvTranspose3dOptions(2, 3, {2, 3, 2})\n                                .stride(1).padding(0).output_padding(0).groups(1).bias(true).dilation({2, 2, 2})', cudnn=True, input_size=(1, 2, 4, 5, 4), desc='dilated'), dict(module_name='MaxPool3d', constructor_args=((2, 2, 2), ), cpp_constructor_args='torch::nn::MaxPool3dOptions({2, 2, 2})', input_size=(2, 3, 5, 5, 5)), dict(module_name='MaxPool3d', constructor_args=(2, (2, 2, 2)), cpp_constructor_args='torch::nn::MaxPool3dOptions(2).stride({2, 2, 2})', input_size=(2, 3, 5, 5, 5), desc='stride'), dict(module_name='MaxPool3d', constructor_args=(2, 2, (1, 1, 1)), cpp_constructor_args='torch::nn::MaxPool3dOptions(2).stride(2).padding({1, 1, 1})', input_size=(2, 3, 5, 5, 5), desc='stride_padding'), dict(module_name='AvgPool3d', constructor_args=((2, 2, 2), ), cpp_constructor_args='torch::nn::AvgPool3dOptions({2, 2, 2})', input_size=(2, 3, 4, 4, 4)), dict(module_name='AvgPool3d', constructor_args=(2, (2, 2, 2)), cpp_constructor_args='torch::nn::AvgPool3dOptions(2).stride({2, 2, 2})', input_size=(2, 3, 5, 5, 5), desc='stride'), dict(module_name='AvgPool3d', constructor_args=(2, 2, (1, 1, 1)), cpp_constructor_args='torch::nn::AvgPool3dOptions(2).stride(2).padding({1, 1, 1})', input_size=(2, 3, 5, 5, 5), desc='stride_pad'), dict(module_name='AvgPool3d', constructor_args=(4, 2, (1, 2, 1)), cpp_constructor_args='torch::nn::AvgPool3dOptions(4).stride(2).padding({1, 2, 1})', input_size=(2, 3, 5, 5, 5), desc='stride_pad_gpu_fixedkw_output'), dict(module_name='AvgPool3d', constructor_args=((2, 4, 8), 1, (1, 1, 2)), cpp_constructor_args='torch::nn::AvgPool3dOptions({2, 4, 8}).stride(1).padding({1, 1, 2})', input_size=(2, 3, 2, 4, 8), desc='stride_pad_gpu_general_output'), dict(module_name='AvgPool3d', constructor_args=(3, 1, 0), cpp_constructor_args='torch::nn::AvgPool3dOptions(3).stride(1).padding(0)', input_size=(2, 3, 4, 4, 4), desc='stride1_pad0_gpu_input'), dict(module_name='AvgPool3d', constructor_args=(2, 2, (1, 1, 1)), cpp_constructor_args='torch::nn::AvgPool3dOptions(2).stride(2).padding({1, 1, 1})', input_size=(2, 3, 4, 4, 4), desc='stride_pad_gpu_input_nooverlap'), dict(fullname='AvgPool3d_divisor', constructor=lambda: nn.AvgPool3d((2, 2, 2), divisor_override=1), cpp_constructor_args='torch::nn::AvgPool3dOptions({2, 2, 2}).divisor_override(1)', input_size=(2, 3, 4, 4, 4), check_with_long_tensor=True), dict(fullname='AvgPool3d_divisor_stride', constructor=lambda: nn.AvgPool3d(2, (2, 2, 2), divisor_override=1), cpp_constructor_args='torch::nn::AvgPool3dOptions(2).stride({2, 2, 2}).divisor_override(1)', input_size=(2, 3, 5, 5, 5), check_with_long_tensor=True), dict(fullname='AvgPool3d_divisor_stride_pad', constructor=lambda: nn.AvgPool3d(2, 2, (1, 1, 1), divisor_override=1), cpp_constructor_args='torch::nn::AvgPool3dOptions(2).stride(2).padding({1, 1, 1}).divisor_override(1)', input_size=(2, 3, 5, 5, 5), check_with_long_tensor=True), dict(fullname='AvgPool3d_divisor_stride_pad_gpu_fixedkw_output', constructor=lambda: nn.AvgPool3d(4, 2, (1, 2, 1), divisor_override=1), cpp_constructor_args='torch::nn::AvgPool3dOptions(4).stride(2).padding({1, 2, 1}).divisor_override(1)', input_size=(2, 3, 5, 5, 5), check_with_long_tensor=True), dict(fullname='AvgPool3d_divisor_stride_pad_gpu_general_output', constructor=lambda: nn.AvgPool3d((2, 4, 8), 1, (1, 1, 2), divisor_override=1), cpp_constructor_args='torch::nn::AvgPool3dOptions({2, 4, 8}).stride(1).padding({1, 1, 2}).divisor_override(1)', input_size=(2, 3, 2, 4, 8), check_with_long_tensor=True), dict(fullname='AvgPool3d_divisor_stride1_pad0_gpu_input', constructor=lambda: nn.AvgPool3d(3, 1, 0, divisor_override=1), cpp_constructor_args='torch::nn::AvgPool3dOptions(3).stride(1).padding(0).divisor_override(1)', input_size=(2, 3, 4, 4, 4), check_with_long_tensor=True), dict(fullname='AvgPool3d_divisor_stride_pad_gpu_input_nooverlap', constructor=lambda: nn.AvgPool3d(2, 2, (1, 1, 1), divisor_override=1), cpp_constructor_args='torch::nn::AvgPool3dOptions(2).stride(2).padding({1, 1, 1}).divisor_override(1)', input_size=(2, 3, 4, 4, 4), check_with_long_tensor=True), dict(module_name='ReplicationPad3d', constructor_args=((1, 2, 3, 4, 5, 6), ), cpp_constructor_args='torch::nn::ReplicationPad3dOptions({1, 2, 3, 4, 5, 6})', input_size=(2, 3, 5, 5, 5)), dict(module_name='Embedding', constructor_args=(4, 3), cpp_constructor_args='torch::nn::EmbeddingOptions(4, 3)', input_fn=lambda: torch.empty(2, 3, dtype=torch.long).random_(4), jacobian_input=False, check_gradgrad=False), dict(module_name='EmbeddingBag', constructor_args=(4, 3), cpp_constructor_args='torch::nn::EmbeddingBagOptions(4, 3)', input_fn=lambda: torch.empty(2, 3, dtype=torch.long).random_(4), jacobian_input=False, check_gradgrad=False, desc='mean'), dict(module_name='EmbeddingBag', constructor_args=(4, 3, None, 2.0, False, 'sum'), cpp_constructor_args='torch::nn::EmbeddingBagOptions(4, 3)\n                                .max_norm(c10::nullopt).norm_type(2.).scale_grad_by_freq(false).mode(torch::kSum)', input_fn=lambda: torch.empty(2, 3, dtype=torch.long).random_(4), jacobian_input=False, check_gradgrad=False, desc='sum'), dict(module_name='EmbeddingBag', constructor_args=(4, 3, None, 2.0, False, 'max'), cpp_constructor_args='torch::nn::EmbeddingBagOptions(4, 3)\n                                .max_norm(c10::nullopt).norm_type(2.).scale_grad_by_freq(false).mode(torch::kMax)', input_fn=lambda: torch.empty(2, 3, dtype=torch.long).random_(4), jacobian_input=False, check_gradgrad=False, desc='max'), dict(fullname='EmbeddingBag_sparse', constructor=lambda: nn.EmbeddingBag(4, 3, sparse=True), cpp_constructor_args='torch::nn::EmbeddingBagOptions(4, 3).sparse(true)', input_fn=lambda: torch.randperm(2).repeat(1, 2), jacobian_input=False, check_gradgrad=False), dict(constructor=lambda: nn.Embedding(4, 3, sparse=True), cpp_constructor_args='torch::nn::EmbeddingOptions(4, 3).sparse(true)', input_fn=lambda: torch.randperm(2).repeat(1, 2), jacobian_input=False, fullname='Embedding_sparse', check_gradgrad=False), dict(module_name='PixelShuffle', constructor_args=(3, ), cpp_constructor_args='torch::nn::PixelShuffleOptions(3)', input_size=(1, 9, 4, 4)), dict(constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='nearest'), cpp_options_args='F::InterpolateFuncOptions()\n                            .size(std::vector<int64_t>({12})).scale_factor(c10::nullopt).mode(torch::kNearest)', input_size=(1, 2, 4), fullname='interpolate_nearest_1d', pickle=False), dict(constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='nearest'), cpp_options_args='F::InterpolateFuncOptions()\n                            .size(std::vector<int64_t>({12})).scale_factor(c10::nullopt).mode(torch::kNearest)', input_size=(0, 2, 4), fullname='interpolate_nearest_1d_zero_dim', pickle=False), dict(constructor=wrap_functional(F.interpolate, size=(12, ), scale_factor=None, mode='nearest'), cpp_options_args='F::InterpolateFuncOptions()\n                            .size(std::vector<int64_t>({12})).scale_factor(c10::nullopt).mode(torch::kNearest)', input_size=(1, 2, 3), fullname='interpolate_nearest_tuple_1d', pickle=False), dict(constructor=wrap_functional(F.interpolate, size=None, scale_factor=4.0, mode='nearest'), cpp_options_args='F::InterpolateFuncOptions()\n                            .size(c10::nullopt).scale_factor(std::vector<double>({4.})).mode(torch::kNearest)', input_size=(1, 2, 4), fullname='interpolate_nearest_scale_1d', pickle=False), dict(constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='linear', align_corners=False), cpp_options_args='F::InterpolateFuncOptions()\n                            .size(std::vector<int64_t>({12}))\n                            .scale_factor(c10::nullopt)\n                            .mode(torch::kLinear)\n                            .align_corners(false)', input_size=(1, 2, 4), fullname='interpolate_linear_1d', pickle=False), dict(constructor=wrap_functional(F.interpolate, size=(4, ), scale_factor=None, mode='linear', align_corners=False), cpp_options_args='F::InterpolateFuncOptions()\n                            .size(std::vector<int64_t>({4}))\n                            .scale_factor(c10::nullopt)\n                            .mode(torch::kLinear)\n                            .align_corners(false)', input_size=(1, 2, 3), fullname='interpolate_linear_tuple_1d', pickle=False), dict(constructor=wrap_functional(F.interpolate, size=None, scale_factor=4.0, mode='linear', align_corners=False), cpp_options_args='F::InterpolateFuncOptions()\n                            .size(c10::nullopt)\n                            .scale_factor(std::vector<double>({4.}))\n                            .mode(torch::kLinear)\n                            .align_corners(false)', input_size=(1, 2, 4), fullname='interpolate_linear_scale_1d', pickle=False), dict(constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='linear', align_corners=False), cpp_options_args='F::InterpolateFuncOptions()\n                            .size(std::vector<int64_t>({12}))\n                            .scale_factor(c10::nullopt)\n                            .mode(torch::kLinear)\n                            .align_corners(false)', input_size=(0, 2, 4), fullname='interpolate_linear_1d_zero_dim', pickle=False), dict(constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='linear', align_corners=True), cpp_options_args='F::InterpolateFuncOptions()\n                            .size(std::vector<int64_t>({12}))\n                            .scale_factor(c10::nullopt)\n                            .mode(torch::kLinear)\n                            .align_corners(true)', input_size=(1, 2, 4), fullname='interpolate_linear_1d_align_corners', pickle=False), dict(constructor=wrap_functional(F.interpolate, size=None, scale_factor=4.0, mode='linear', align_corners=True), cpp_options_args='F::InterpolateFuncOptions()\n                            .size(c10::nullopt)\n                            .scale_factor(std::vector<double>({4.}))\n                            .mode(torch::kLinear)\n                            .align_corners(true)', input_size=(1, 2, 4), fullname='interpolate_linear_scale_1d_align_corners', pickle=False), dict(constructor=wrap_functional(F.interpolate, size=2, scale_factor=None, mode='nearest'), cpp_options_args='F::InterpolateFuncOptions()\n                            .size(std::vector<int64_t>({2, 2}))\n                            .scale_factor(c10::nullopt)\n                            .mode(torch::kNearest)', input_size=(1, 128, 1, 1), fullname='interpolate_nearest_2d_launch_configs', pickle=False), dict(constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='nearest'), cpp_options_args='F::InterpolateFuncOptions()\n                            .size(std::vector<int64_t>({12, 12}))\n                            .scale_factor(c10::nullopt)\n                            .mode(torch::kNearest)', input_size=(1, 2, 4, 4), fullname='interpolate_nearest_2d', pickle=False), dict(constructor=wrap_functional(F.interpolate, size=(12, 16), scale_factor=None, mode='nearest'), cpp_options_args='F::InterpolateFuncOptions()\n                            .size(std::vector<int64_t>({12, 16}))\n                            .scale_factor(c10::nullopt)\n                            .mode(torch::kNearest)', input_size=(1, 2, 3, 4), fullname='interpolate_nearest_tuple_2d', pickle=False), dict(constructor=wrap_functional(F.interpolate, size=None, scale_factor=4.0, mode='nearest'), cpp_options_args='F::InterpolateFuncOptions()\n                            .size(c10::nullopt)\n                            .scale_factor(std::vector<double>({4., 4.}))\n                            .mode(torch::kNearest)', input_size=(1, 2, 4, 4), fullname='interpolate_nearest_scale_2d', pickle=False), dict(constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='nearest'), cpp_options_args='F::InterpolateFuncOptions()\n                            .size(std::vector<int64_t>({12, 12}))\n                            .scale_factor(c10::nullopt)\n                            .mode(torch::kNearest)', input_size=(0, 2, 4, 4), fullname='interpolate_nearest_2d_zero_dim', pickle=False), dict(constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='bilinear', align_corners=False), cpp_options_args='F::InterpolateFuncOptions()\n                            .size(std::vector<int64_t>({12, 12}))\n                            .scale_factor(c10::nullopt)\n                            .mode(torch::kBilinear)\n                            .align_corners(false)', input_size=(1, 2, 4, 4), fullname='interpolate_bilinear_2d', pickle=False), dict(constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='bilinear', align_corners=False), cpp_options_args='F::InterpolateFuncOptions()\n                            .size(std::vector<int64_t>({12, 12}))\n                            .scale_factor(c10::nullopt)\n                            .mode(torch::kBilinear)\n                            .align_corners(false)', input_size=(0, 2, 4, 4), fullname='interpolate_bilinear_2d_zero_dim', pickle=False), dict(constructor=wrap_functional(F.interpolate, size=(4, 6), scale_factor=None, mode='bilinear', align_corners=False), cpp_options_args='F::InterpolateFuncOptions()\n                            .size(std::vector<int64_t>({4, 6}))\n                            .scale_factor(c10::nullopt)\n                            .mode(torch::kBilinear)\n                            .align_corners(false)', input_size=(1, 2, 2, 3), fullname='interpolate_bilinear_tuple_2d', pickle=False), dict(constructor=wrap_functional(F.interpolate, size=None, scale_factor=4.0, mode='bilinear', align_corners=False), cpp_options_args='F::InterpolateFuncOptions()\n                            .size(c10::nullopt)\n                            .scale_factor(std::vector<double>({4., 4.}))\n                            .mode(torch::kBilinear)\n                            .align_corners(false)', input_size=(1, 2, 4, 4), fullname='interpolate_bilinear_scale_2d', pickle=False), dict(constructor=wrap_functional(F.interpolate, size=None, scale_factor=(2.0, 2.0), mode='bilinear', align_corners=False), cpp_options_args='F::InterpolateFuncOptions()\n                            .size(c10::nullopt)\n                            .scale_factor(std::vector<double>({2., 2.}))\n                            .mode(torch::kBilinear)\n                            .align_corners(false)', input_size=(1, 2, 4, 4), fullname='interpolate_bilinear_scale_tuple_shared_2d', pickle=False), dict(constructor=wrap_functional(F.interpolate, size=None, scale_factor=(2.0, 1.0), mode='bilinear', align_corners=False), cpp_options_args='F::InterpolateFuncOptions()\n                            .size(c10::nullopt)\n                            .scale_factor(std::vector<double>({2., 1.}))\n                            .mode(torch::kBilinear)\n                            .align_corners(false)', input_size=(1, 2, 4, 4), fullname='interpolate_bilinear_scale_tuple_skewed_2d', pickle=False), dict(constructor=wrap_functional(F.interpolate, size=(4, 6), scale_factor=None, mode='bilinear', align_corners=True), cpp_options_args='F::InterpolateFuncOptions()\n                            .size(std::vector<int64_t>({4, 6}))\n                            .scale_factor(c10::nullopt)\n                            .mode(torch::kBilinear)\n                            .align_corners(true)', input_size=(1, 2, 4, 4), fullname='interpolate_bilinear_tuple_2d_align_corners', pickle=False), dict(constructor=wrap_functional(F.interpolate, size=None, scale_factor=(2.0, 1.0), mode='bilinear', align_corners=True), cpp_options_args='F::InterpolateFuncOptions()\n                            .size(c10::nullopt)\n                            .scale_factor(std::vector<double>({2., 1.}))\n                            .mode(torch::kBilinear)\n                            .align_corners(true)', input_size=(1, 2, 4, 4), fullname='interpolate_bilinear_scale_tuple_skewed_2d_align_corners', pickle=False), dict(constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='bicubic', align_corners=False), cpp_options_args='F::InterpolateFuncOptions()\n                            .size(std::vector<int64_t>({12, 12}))\n                            .scale_factor(c10::nullopt)\n                            .mode(torch::kBicubic)\n                            .align_corners(false)', input_size=(1, 2, 4, 4), fullname='interpolate_bicubic_2d', pickle=False), dict(constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='bicubic', align_corners=False), cpp_options_args='F::InterpolateFuncOptions()\n                            .size(std::vector<int64_t>({12, 12}))\n                            .scale_factor(c10::nullopt)\n                            .mode(torch::kBicubic)\n                            .align_corners(false)', input_size=(0, 2, 4, 4), fullname='interpolate_bicubic_2d_zero_dim', pickle=False), dict(constructor=wrap_functional(F.interpolate, size=(4, 6), scale_factor=None, mode='bicubic', align_corners=False), cpp_options_args='F::InterpolateFuncOptions()\n                            .size(std::vector<int64_t>({4, 6}))\n                            .scale_factor(c10::nullopt)\n                            .mode(torch::kBicubic)\n                            .align_corners(false)', input_size=(1, 2, 2, 3), fullname='interpolate_bicubic_tuple_2d', pickle=False), dict(constructor=wrap_functional(F.interpolate, size=None, scale_factor=4.0, mode='bicubic', align_corners=False), cpp_options_args='F::InterpolateFuncOptions()\n                            .size(c10::nullopt)\n                            .scale_factor(std::vector<double>({4., 4.}))\n                            .mode(torch::kBicubic)\n                            .align_corners(false)', input_size=(1, 2, 4, 4), fullname='interpolate_bicubic_scale_2d', pickle=False), dict(constructor=wrap_functional(F.interpolate, size=None, scale_factor=(2.0, 2.0), mode='bicubic', align_corners=False), cpp_options_args='F::InterpolateFuncOptions()\n                            .size(c10::nullopt)\n                            .scale_factor(std::vector<double>({2., 2.}))\n                            .mode(torch::kBicubic)\n                            .align_corners(false)', input_size=(1, 2, 4, 4), fullname='interpolate_bicubic_scale_tuple_shared_2d', pickle=False), dict(constructor=wrap_functional(F.interpolate, size=None, scale_factor=(2.0, 1.0), mode='bicubic', align_corners=False), cpp_options_args='F::InterpolateFuncOptions()\n                            .size(c10::nullopt)\n                            .scale_factor(std::vector<double>({2., 1.}))\n                            .mode(torch::kBicubic)\n                            .align_corners(false)', input_size=(1, 2, 4, 4), fullname='interpolate_bicubic_scale_tuple_skewed_2d', pickle=False), dict(constructor=wrap_functional(F.interpolate, size=(4, 6), scale_factor=None, mode='bicubic', align_corners=True), cpp_options_args='F::InterpolateFuncOptions()\n                            .size(std::vector<int64_t>({4, 6}))\n                            .scale_factor(c10::nullopt)\n                            .mode(torch::kBicubic)\n                            .align_corners(true)', input_size=(1, 2, 4, 4), fullname='interpolate_bicubic_tuple_2d_align_corners', pickle=False), dict(constructor=wrap_functional(F.interpolate, size=None, scale_factor=(2.0, 1.0), mode='bicubic', align_corners=True), cpp_options_args='F::InterpolateFuncOptions()\n                            .size(c10::nullopt)\n                            .scale_factor(std::vector<double>({2., 1.}))\n                            .mode(torch::kBicubic)\n                            .align_corners(true)', input_size=(1, 2, 4, 4), fullname='interpolate_bicubic_scale_tuple_skewed_2d_align_corners', pickle=False), dict(constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='nearest'), cpp_options_args='F::InterpolateFuncOptions()\n                            .size(std::vector<int64_t>({12, 12, 12}))\n                            .scale_factor(c10::nullopt)\n                            .mode(torch::kNearest)', input_size=(1, 2, 4, 4, 4), fullname='interpolate_nearest_3d', pickle=False), dict(constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='nearest'), cpp_options_args='F::InterpolateFuncOptions()\n                            .size(std::vector<int64_t>({12, 12, 12}))\n                            .scale_factor(c10::nullopt)\n                            .mode(torch::kNearest)', input_size=(0, 2, 4, 4, 4), fullname='interpolate_nearest_3d_zero_dim', pickle=False), dict(constructor=wrap_functional(F.interpolate, size=(12, 16, 16), scale_factor=None, mode='nearest'), cpp_options_args='F::InterpolateFuncOptions()\n                            .size(std::vector<int64_t>({12, 16, 16}))\n                            .scale_factor(c10::nullopt)\n                            .mode(torch::kNearest)', input_size=(1, 2, 3, 4, 4), fullname='interpolate_nearest_tuple_3d', pickle=False), dict(constructor=wrap_functional(F.interpolate, size=None, scale_factor=4.0, mode='nearest'), cpp_options_args='F::InterpolateFuncOptions()\n                            .size(c10::nullopt)\n                            .scale_factor(std::vector<double>({4., 4., 4.}))\n                            .mode(torch::kNearest)', input_size=(1, 2, 4, 4, 4), fullname='interpolate_nearest_scale_3d', pickle=False), dict(constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='trilinear', align_corners=False), cpp_options_args='F::InterpolateFuncOptions()\n                            .size(std::vector<int64_t>({12, 12, 12}))\n                            .scale_factor(c10::nullopt)\n                            .mode(torch::kTrilinear)\n                            .align_corners(false)', input_size=(1, 2, 4, 4, 4), fullname='interpolate_trilinear_3d', pickle=False), dict(constructor=wrap_functional(F.interpolate, size=12, scale_factor=None, mode='trilinear', align_corners=False), cpp_options_args='F::InterpolateFuncOptions()\n                            .size(std::vector<int64_t>({12, 12, 12}))\n                            .scale_factor(c10::nullopt)\n                            .mode(torch::kTrilinear)\n                            .align_corners(false)', input_size=(0, 2, 4, 4, 4), fullname='interpolate_trilinear_3d_zero_dim', pickle=False), dict(constructor=wrap_functional(F.interpolate, size=(4, 6, 6), scale_factor=None, mode='trilinear', align_corners=False), cpp_options_args='F::InterpolateFuncOptions()\n                            .size(std::vector<int64_t>({4, 6, 6}))\n                            .scale_factor(c10::nullopt)\n                            .mode(torch::kTrilinear)\n                            .align_corners(false)', input_size=(1, 2, 2, 3, 3), fullname='interpolate_trilinear_tuple_3d', pickle=False), dict(constructor=wrap_functional(F.interpolate, size=None, scale_factor=3.0, mode='trilinear', align_corners=False), cpp_options_args='F::InterpolateFuncOptions()\n                            .size(c10::nullopt)\n                            .scale_factor(std::vector<double>({3., 3., 3.}))\n                            .mode(torch::kTrilinear)\n                            .align_corners(false)', input_size=(1, 2, 3, 4, 4), fullname='interpolate_trilinear_scale_3d', precision=0.0003, pickle=False), dict(constructor=wrap_functional(F.interpolate, size=(4, 6, 6), scale_factor=None, mode='trilinear', align_corners=True), cpp_options_args='F::InterpolateFuncOptions()\n                            .size(std::vector<int64_t>({4, 6, 6}))\n                            .scale_factor(c10::nullopt)\n                            .mode(torch::kTrilinear)\n                            .align_corners(true)', input_size=(1, 2, 2, 3, 3), fullname='interpolate_trilinear_tuple_3d_align_corners', pickle=False), dict(constructor=wrap_functional(F.interpolate, size=None, scale_factor=3.0, mode='trilinear', align_corners=True), cpp_options_args='F::InterpolateFuncOptions()\n                            .size(c10::nullopt)\n                            .scale_factor(std::vector<double>({3., 3., 3.}))\n                            .mode(torch::kTrilinear)\n                            .align_corners(true)', input_size=(1, 2, 3, 4, 4), fullname='interpolate_trilinear_scale_3d_align_corners', precision=0.0003, pickle=False), dict(module_name='AdaptiveMaxPool1d', constructor_args=(3, ), cpp_constructor_args='torch::nn::AdaptiveMaxPool1dOptions(3)', input_fn=lambda: _rand_tensor_non_equal(1, 3, 5)), dict(module_name='AdaptiveMaxPool2d', constructor_args=(3, ), cpp_constructor_args='torch::nn::AdaptiveMaxPool2dOptions(3)', input_fn=lambda: _rand_tensor_non_equal(1, 3, 5, 6), desc='single'), dict(module_name='AdaptiveMaxPool2d', constructor_args=((3, 4), ), cpp_constructor_args='torch::nn::AdaptiveMaxPool2dOptions({3, 4})', input_fn=lambda: _rand_tensor_non_equal(1, 3, 5, 6), desc='tuple'), dict(module_name='AdaptiveMaxPool2d', constructor_args=((3, None), ), cpp_constructor_args='torch::nn::AdaptiveMaxPool2dOptions({3, c10::nullopt})', input_fn=lambda: _rand_tensor_non_equal(1, 3, 5, 6), desc='tuple_none'), dict(module_name='AdaptiveMaxPool3d', constructor_args=(3, ), cpp_constructor_args='torch::nn::AdaptiveMaxPool3dOptions(3)', input_fn=lambda: _rand_tensor_non_equal(2, 3, 5, 6, 7), desc='single'), dict(module_name='AdaptiveMaxPool3d', constructor_args=((3, 4, 5), ), cpp_constructor_args='torch::nn::AdaptiveMaxPool3dOptions({3, 4, 5})', input_fn=lambda: _rand_tensor_non_equal(2, 3, 5, 6, 7), desc='tuple'), dict(module_name='AdaptiveMaxPool3d', constructor_args=((3, None, 5), ), cpp_constructor_args='torch::nn::AdaptiveMaxPool3dOptions({3, c10::nullopt, 5})', input_fn=lambda: _rand_tensor_non_equal(2, 3, 5, 6, 7), desc='tuple_none'), dict(module_name='AdaptiveMaxPool3d', constructor_args=(3, ), cpp_constructor_args='torch::nn::AdaptiveMaxPool3dOptions(3)', input_fn=lambda: _rand_tensor_non_equal(2, 3, 12, 9, 3), desc='single_nonatomic'), dict(module_name='AdaptiveMaxPool3d', constructor_args=((3, 4, 5), ), cpp_constructor_args='torch::nn::AdaptiveMaxPool3dOptions({3, 4, 5})', input_fn=lambda: _rand_tensor_non_equal(2, 3, 6, 4, 10), desc='tuple_nonatomic'), dict(module_name='AdaptiveAvgPool1d', constructor_args=(3, ), cpp_constructor_args='torch::nn::AdaptiveAvgPool1dOptions(3)', input_fn=lambda: torch.rand(1, 3, 5)), dict(module_name='AdaptiveAvgPool1d', constructor_args=(1, ), cpp_constructor_args='torch::nn::AdaptiveAvgPool1dOptions(1)', input_fn=lambda: torch.rand(1, 3, 5), desc='one_output'), dict(module_name='AdaptiveAvgPool2d', constructor_args=(3, ), cpp_constructor_args='torch::nn::AdaptiveAvgPool2dOptions(3)', input_fn=lambda: torch.rand(1, 3, 5, 6), desc='single'), dict(module_name='AdaptiveAvgPool2d', constructor_args=(1, ), cpp_constructor_args='torch::nn::AdaptiveAvgPool2dOptions(1)', input_fn=lambda: torch.rand(1, 3, 5, 6), desc='single_1x1output'), dict(module_name='AdaptiveAvgPool2d', constructor_args=((3, 4), ), cpp_constructor_args='torch::nn::AdaptiveAvgPool2dOptions({3, 4})', input_fn=lambda: torch.rand(1, 3, 5, 6), desc='tuple'), dict(module_name='AdaptiveAvgPool2d', constructor_args=((3, None), ), cpp_constructor_args='torch::nn::AdaptiveAvgPool2dOptions({3, c10::nullopt})', input_fn=lambda: torch.rand(1, 3, 5, 6), desc='tuple_none'), dict(module_name='AdaptiveAvgPool3d', constructor_args=(3, ), cpp_constructor_args='torch::nn::AdaptiveAvgPool3dOptions(3)', input_fn=lambda: torch.rand(2, 3, 5, 2, 7), desc='single'), dict(module_name='AdaptiveAvgPool3d', constructor_args=((3, 4, 5), ), cpp_constructor_args='torch::nn::AdaptiveAvgPool3dOptions({3, 4, 5})', input_fn=lambda: torch.rand(2, 3, 5, 3, 7), desc='tuple'), dict(module_name='AdaptiveAvgPool3d', constructor_args=((None, 4, 5), ), cpp_constructor_args='torch::nn::AdaptiveAvgPool3dOptions({c10::nullopt, 4, 5})', input_fn=lambda: torch.rand(2, 3, 5, 3, 7), desc='tuple_none'), dict(module_name='SELU', input_size=(3, 2, 5), check_inplace=True), dict(module_name='SELU', input_size=(), check_inplace=True, desc='scalar'), dict(module_name='CELU', input_size=(3, 2, 5), constructor_args=(2.0, ), cpp_constructor_args='torch::nn::CELUOptions().alpha(2.)', check_inplace=True, reference_fn=lambda x, *_: torch.where(x >= 0, x, 2.0 * ((0.5 * x).exp() - 1))), dict(module_name='CELU', input_size=(), constructor_args=(2.0, ), cpp_constructor_args='torch::nn::CELUOptions().alpha(2.)', check_inplace=True, reference_fn=lambda x, *_: torch.where(x >= 0, x, 2.0 * ((0.5 * x).exp() - 1)), desc='scalar'), dict(module_name='GLU', input_size=(5, 6)), dict(module_name='GLU', constructor_args=(1, ), cpp_constructor_args='torch::nn::GLUOptions(1)', input_size=(5, 6, 7), desc='dim'), dict(module_name='GELU', input_size=(), desc='scalar', reference_fn=lambda x, *_: x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))), dict(module_name='GELU', input_size=(3, 2, 5), reference_fn=lambda x, *_: x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))), dict(constructor=wrap_functional(F.softmax, dim=-1), cpp_options_args='F::SoftmaxFuncOptions(-1)', input_size=(2, 128), fullname='softmax_lastdim', pickle=False), dict(constructor=wrap_functional(F.softmax, dim=1, dtype=torch.float64), cpp_options_args='F::SoftmaxFuncOptions(1).dtype(torch::kFloat64)', input_size=(2, 128), fullname='softmax_lastdim_dtype', pickle=False, test_cuda=False), dict(constructor=wrap_functional(F.softmax, dim=1), cpp_options_args='F::SoftmaxFuncOptions(1)', input_size=(2, 128, 2, 2), fullname='softmax_spatial_special', pickle=False, test_cuda=not TEST_WITH_ROCM), dict(constructor=wrap_functional(F.softmax, dim=1), cpp_options_args='F::SoftmaxFuncOptions(1)', input_size=(2, 2, 4, 4), fullname='softmax_spatial', pickle=False), dict(constructor=wrap_functional(F.softmax, dim=1, dtype=torch.float64), cpp_options_args='F::SoftmaxFuncOptions(1).dtype(torch::kFloat64)', input_size=(2, 2, 4, 4), fullname='softmax_spatial_dtype', pickle=False, test_cuda=False), dict(constructor=wrap_functional(F.softmax, dim=0), cpp_options_args='F::SoftmaxFuncOptions(0)', input_size=(2, 3, 4, 5), fullname='softmax_functional_dim0', test_cuda=False, pickle=False), dict(constructor=wrap_functional(F.softmax, dim=3), cpp_options_args='F::SoftmaxFuncOptions(3)', input_size=(2, 3, 4, 5), fullname='softmax_functional_dim3', test_cuda=False, pickle=False), dict(constructor=wrap_functional(F.softmax, dim=-1), cpp_options_args='F::SoftmaxFuncOptions(-1)', input_size=(), fullname='softmax_functional_scalar', test_cuda=False, pickle=False), dict(constructor=wrap_functional(F.log_softmax, dim=-1), cpp_options_args='F::LogSoftmaxFuncOptions(-1)', input_size=(2, 128), fullname='log_softmax_lastdim', pickle=False), dict(constructor=wrap_functional(F.log_softmax, dim=1), cpp_options_args='F::LogSoftmaxFuncOptions(1)', input_size=(2, 128, 2, 2), fullname='log_softmax_spatial_special', pickle=False, test_cuda=not TEST_WITH_ROCM), dict(constructor=wrap_functional(F.log_softmax, dim=1), cpp_options_args='F::LogSoftmaxFuncOptions(1)', input_size=(2, 2, 4, 4), fullname='log_softmax_spatial', pickle=False), dict(constructor=wrap_functional(F.log_softmax, dim=0), cpp_options_args='F::LogSoftmaxFuncOptions(0)', input_size=(2, 3, 4, 5), fullname='log_softmax_dim0', pickle=False), dict(constructor=wrap_functional(F.log_softmax, dim=3), cpp_options_args='F::LogSoftmaxFuncOptions(3)', input_size=(2, 3, 4, 5), fullname='log_softmax_dim3', pickle=False), dict(constructor=wrap_functional(F.log_softmax, dim=0), cpp_options_args='F::LogSoftmaxFuncOptions(0)', input_size=(), fullname='log_softmax_scalar', pickle=False), dict(fullname='Unfold', constructor=lambda: nn.Unfold((2, 2), (1, 1), (0, 0), (1, 1)), cpp_constructor_args='torch::nn::UnfoldOptions({2, 2}).dilation({1, 1}).padding({0, 0}).stride({1, 1})', input_size=(2, 4, 3, 3), check_gradgrad=False, test_cuda=True), dict(fullname='Fold', constructor=lambda: nn.Fold((3, 3), (2, 2), (1, 1), (0, 0), (1, 1)), cpp_constructor_args='torch::nn::FoldOptions({3, 3}, {2, 2}).dilation({1, 1}).padding({0, 0}).stride({1, 1})', input_size=(2, 16, 4), check_gradgrad=False, test_cuda=True), dict(fullname='Unfold_int_input', constructor=lambda: nn.Unfold(2, 1, 0, 1), cpp_constructor_args='torch::nn::UnfoldOptions(2).dilation(1).padding(0).stride(1)', input_size=(2, 4, 3, 3), check_gradgrad=False, test_cuda=True), dict(fullname='Fold_int_input', constructor=lambda: nn.Fold(3, 2, 1, 0, 1), cpp_constructor_args='torch::nn::FoldOptions(3, 2).dilation(1).padding(0).stride(1)', input_size=(2, 16, 4), check_gradgrad=False, test_cuda=True), dict(module_name='Threshold', constructor_args=(2.0, 1.0), cpp_constructor_args='torch::nn::ThresholdOptions(2., 1.)', input_size=(), check_inplace=True, desc='threshold_value_scalar'), dict(module_name='ReLU', input_size=(), check_inplace=True, desc='scalar'), dict(module_name='ReLU6', input_size=(), check_inplace=True, desc='scalar'), dict(module_name='RReLU', constructor_args=(0.1, 0.9), cpp_constructor_args='torch::nn::RReLUOptions().lower(0.1).upper(0.9)', input_size=(), desc='with_up_down_scalar', test_cuda=False), dict(module_name='Hardtanh', input_size=(), reference_fn=lambda i, *_: i.clamp(-1, 1), desc='scalar'), dict(module_name='Sigmoid', input_size=(), desc='scalar'), dict(module_name='Tanh', input_size=(), desc='scalar'), dict(module_name='Softmax', constructor_args=(0, ), cpp_constructor_args='torch::nn::SoftmaxOptions(0)', input_size=(), reference_fn=lambda i, *_: torch.exp(i).div(torch.exp(i).sum(0, True)), desc='scalar'), dict(module_name='LogSoftmax', constructor_args=(0, ), cpp_constructor_args='torch::nn::LogSoftmaxOptions(0)', input_size=(), reference_fn=lambda i, *_: torch.exp(i).div_(torch.exp(i).sum(0, False)).log_(), desc='multiparam_scalar'), dict(module_name='ELU', constructor_args=(2.0, ), cpp_constructor_args='torch::nn::ELUOptions().alpha(2.)', input_size=(), desc='scalar'), dict(module_name='Hardshrink', constructor_args=(2.0, ), cpp_constructor_args='torch::nn::HardshrinkOptions(2.)', input_size=(), desc='scalar'), dict(module_name='LeakyReLU', constructor_args=(0.5, ), cpp_constructor_args='torch::nn::LeakyReLUOptions().negative_slope(0.5)', input_size=(), check_inplace=True, desc='with_negval_scalar'), dict(module_name='LogSigmoid', input_size=(), reference_fn=lambda i, *_: i.sigmoid().log(), desc='scalar'), dict(module_name='Softplus', constructor_args=(2, -100), cpp_constructor_args='torch::nn::SoftplusOptions().beta(2).threshold(-100)', input_size=(), reference_fn=lambda i, *_: (i * 2 > -100).type_as(i) * i + (i * 2 <= -100).type_as(i) * 1.0 / 2.0 * torch.log(1 + torch.exp(2 * i)), desc='beta_threshold_scalar'), dict(module_name='Softshrink', constructor_args=(1, ), cpp_constructor_args='torch::nn::SoftshrinkOptions(1)', input_size=(), desc='lambda_scalar'), dict(module_name='PReLU', input_size=(), reference_fn=lambda i, p, _: torch.clamp(i, min=0) + torch.clamp(i, max=0) * p[0][0], desc='scalar'), dict(module_name='Softsign', input_size=(), reference_fn=lambda i, *_: i.div(1 + torch.abs(i)), desc='scalar'), dict(module_name='Softmin', constructor_args=(0, ), cpp_constructor_args='torch::nn::SoftminOptions(0)', input_size=(), desc='scalar'), dict(module_name='Tanhshrink', input_size=(), desc='scalar'), dict(fullname='Padding12_1dcircular', constructor=wrap_functional(F.pad, pad=(1, 2), mode='circular'), cpp_options_args='F::PadFuncOptions({1, 2}).mode(torch::kCircular)', input_fn=lambda: torch.arange(6, out=torch.DoubleTensor()).reshape([1, 2, 3]), reference_fn=lambda i, *_: padding1d_circular(i, (1, 2)), skip_double=TEST_WITH_ROCM, pickle=False), dict(fullname='Padding31_1dcircular', constructor=wrap_functional(F.pad, pad=(3, 1), mode='circular'), cpp_options_args='F::PadFuncOptions({3, 1}).mode(torch::kCircular)', input_fn=lambda: torch.arange(6, out=torch.DoubleTensor()).reshape([1, 2, 3]), reference_fn=lambda i, *_: padding1d_circular(i, (3, 1)), skip_double=TEST_WITH_ROCM, pickle=False), dict(fullname='Padding33_1dcircular', constructor=wrap_functional(F.pad, pad=(3, 3), mode='circular'), cpp_options_args='F::PadFuncOptions({3, 3}).mode(torch::kCircular)', input_fn=lambda: torch.arange(6, out=torch.DoubleTensor()).reshape([1, 2, 3]), reference_fn=lambda i, *_: padding1d_circular(i, (3, 3)), skip_double=TEST_WITH_ROCM, pickle=False), dict(fullname='Padding1221_2dcircular', constructor=wrap_functional(F.pad, pad=(1, 2, 2, 1), mode='circular'), cpp_options_args='F::PadFuncOptions({1, 2, 2, 1}).mode(torch::kCircular)', input_fn=lambda: torch.arange(6, out=torch.DoubleTensor()).reshape([1, 1, 2, 3]), reference_fn=lambda i, *_: padding2d_circular(i, (1, 2, 2, 1)), skip_double=TEST_WITH_ROCM, pickle=False), dict(fullname='Padding2322_2dcircular', constructor=wrap_functional(F.pad, pad=(2, 3, 2, 2), mode='circular'), cpp_options_args='F::PadFuncOptions({2, 3, 2, 2}).mode(torch::kCircular)', input_fn=lambda: torch.arange(6, out=torch.DoubleTensor()).reshape([1, 1, 2, 3]), reference_fn=lambda i, *_: padding2d_circular(i, (2, 3, 2, 2)), skip_double=TEST_WITH_ROCM, pickle=False), dict(fullname='Padding3331_2dcircular', constructor=wrap_functional(F.pad, pad=(3, 3, 3, 1), mode='circular'), cpp_options_args='F::PadFuncOptions({3, 3, 3, 1}).mode(torch::kCircular)', input_fn=lambda: torch.arange(9, out=torch.DoubleTensor()).reshape([1, 1, 3, 3]), reference_fn=lambda i, *_: padding2d_circular(i, (3, 3, 3, 1)), skip_double=TEST_WITH_ROCM, pickle=False), dict(fullname='Padding122112_3dcircular', constructor=wrap_functional(F.pad, pad=(1, 2, 2, 1, 1, 2), mode='circular'), cpp_options_args='F::PadFuncOptions({1, 2, 2, 1, 1, 2}).mode(torch::kCircular)', input_fn=lambda: torch.arange(12, out=torch.DoubleTensor()).reshape([1, 1, 2, 2, 3]), reference_fn=lambda i, *_: padding3d_circular(i, (1, 2, 2, 1, 1, 2)), skip_double=TEST_WITH_ROCM, pickle=False), dict(fullname='Padding322112_3dcircular', constructor=wrap_functional(F.pad, pad=(3, 2, 2, 1, 1, 2), mode='circular'), cpp_options_args='F::PadFuncOptions({3, 2, 2, 1, 1, 2}).mode(torch::kCircular)', input_fn=lambda: torch.arange(12, out=torch.DoubleTensor()).reshape([1, 1, 2, 2, 3]), reference_fn=lambda i, *_: padding3d_circular(i, (3, 2, 2, 1, 1, 2)), skip_double=TEST_WITH_ROCM, pickle=False), dict(fullname='Padding332122_3dcircular', constructor=wrap_functional(F.pad, pad=(3, 3, 2, 1, 2, 2), mode='circular'), cpp_options_args='F::PadFuncOptions({3, 3, 2, 1, 2, 2}).mode(torch::kCircular)', input_fn=lambda: torch.arange(12, out=torch.DoubleTensor()).reshape([1, 1, 2, 2, 3]), reference_fn=lambda i, *_: padding3d_circular(i, (3, 3, 2, 1, 2, 2)), skip_double=TEST_WITH_ROCM, pickle=False)]
for (padding_mode, cpp_padding_mode) in zip(['reflect', 'circular', 'replicate', 'zeros'], ['torch::kReflect', 'torch::kCircular', 'torch::kReplicate', 'torch::kZeros']):
    for d in (1, 2, 3):
        if (d == 3 and padding_mode == 'reflect'):
            continue
        new_module_tests.append(dict(module_name='Conv{}d'.format(d), constructor_args=(3, 4, 3, 2, 2, 1, 1, True, padding_mode), cpp_constructor_args='torch::nn::Conv{}dOptions(3, 4, 3)\n                                        .stride(2)\n                                        .padding(2)\n                                        .dilation(1)\n                                        .groups(1)\n                                        .bias(true)\n                                        .padding_mode({})'.format(d, cpp_padding_mode), input_size=(2, 3) + (3, ) * d, output_size=(2, 4) + (3, ) * d, cudnn=True, desc='{}_stride2_pad2'.format(padding_mode)))

def kldivloss_reference(input, target, reduction='mean'):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_nn.kldivloss_reference', "kldivloss_reference(input, target, reduction='mean')", {'results': results, 'input': input, 'target': target, 'reduction': reduction}, 1)

def nlllossNd_reference(input, target, weight=None, ignore_index=-100, reduction='mean'):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_nn.nlllossNd_reference', "nlllossNd_reference(input, target, weight=None, ignore_index=-100, reduction='mean')", {'torch': torch, 'product': product, 'input': input, 'target': target, 'weight': weight, 'ignore_index': ignore_index, 'reduction': reduction}, 1)

def nllloss_reference(input, target, weight=None, ignore_index=-100, reduction='mean'):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_nn.nllloss_reference', "nllloss_reference(input, target, weight=None, ignore_index=-100, reduction='mean')", {'input': input, 'target': target, 'weight': weight, 'ignore_index': ignore_index, 'reduction': reduction}, 2)

def smoothl1loss_reference(input, target, reduction='mean'):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_nn.smoothl1loss_reference', "smoothl1loss_reference(input, target, reduction='mean')", {'input': input, 'target': target, 'reduction': reduction}, 1)

def _multilabelmarginloss_reference(input, target):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_nn._multilabelmarginloss_reference', '_multilabelmarginloss_reference(input, target)', {'input': input, 'target': target}, 1)

def multilabelmarginloss_reference(input, target, reduction='mean'):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_nn.multilabelmarginloss_reference', "multilabelmarginloss_reference(input, target, reduction='mean')", {'_multilabelmarginloss_reference': _multilabelmarginloss_reference, 'input': input, 'target': target, 'reduction': reduction}, 1)

def hingeembeddingloss_reference(input, target, margin=1.0, reduction='mean'):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_nn.hingeembeddingloss_reference', "hingeembeddingloss_reference(input, target, margin=1.0, reduction='mean')", {'torch': torch, 'input': input, 'target': target, 'margin': margin, 'reduction': reduction}, 1)

def softmarginloss_reference(input, target, reduction='mean'):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_nn.softmarginloss_reference', "softmarginloss_reference(input, target, reduction='mean')", {'input': input, 'target': target, 'reduction': reduction}, 1)

def _multimarginloss_reference(input, target_idx, p, margin, weight):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_nn._multimarginloss_reference', '_multimarginloss_reference(input, target_idx, p, margin, weight)', {'input': input, 'target_idx': target_idx, 'p': p, 'margin': margin, 'weight': weight}, 1)

def multimarginloss_reference(input, target, p=1, margin=1, weight=None, reduction='mean'):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_nn.multimarginloss_reference', "multimarginloss_reference(input, target, p=1, margin=1, weight=None, reduction='mean')", {'_multimarginloss_reference': _multimarginloss_reference, 'input': input, 'target': target, 'p': p, 'margin': margin, 'weight': weight, 'reduction': reduction}, 1)

def cosineembeddingloss_reference(input1, input2, target, margin=0, reduction='mean'):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_nn.cosineembeddingloss_reference', "cosineembeddingloss_reference(input1, input2, target, margin=0, reduction='mean')", {'torch': torch, 'input1': input1, 'input2': input2, 'target': target, 'margin': margin, 'reduction': reduction}, 1)

def tripletmarginloss_reference(anchor, positive, negative, margin=1.0, p=2, eps=1e-06, swap=False, reduction='mean'):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_nn.tripletmarginloss_reference', "tripletmarginloss_reference(anchor, positive, negative, margin=1.0, p=2, eps=1e-06, swap=False, reduction='mean')", {'torch': torch, 'anchor': anchor, 'positive': positive, 'negative': negative, 'margin': margin, 'p': p, 'eps': eps, 'swap': swap, 'reduction': reduction}, 1)

def marginrankingloss_reference(input1, input2, target, margin=0, reduction='mean'):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_nn.marginrankingloss_reference', "marginrankingloss_reference(input1, input2, target, margin=0, reduction='mean')", {'input1': input1, 'input2': input2, 'target': target, 'margin': margin, 'reduction': reduction}, 1)

def ctcloss_reference(log_probs, targets, input_lengths, target_lengths, blank=0, reduction='mean'):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_nn.ctcloss_reference', "ctcloss_reference(log_probs, targets, input_lengths, target_lengths, blank=0, reduction='mean')", {'torch': torch, 'log_probs': log_probs, 'targets': targets, 'input_lengths': input_lengths, 'target_lengths': target_lengths, 'blank': blank, 'reduction': reduction}, 1)

def padding1d_circular(input, pad):
    """ input:
            [[[0., 1., 2.],
              [3., 4., 5.]]]
          pad: (1, 2)
          output:
            [[[2., 0., 1., 2., 0., 1.],
              [5., 3., 4., 5., 3., 4.]]]
    """
    return torch.cat([input[:, :, -pad[0]:], input, input[:, :, 0:pad[1]]], dim=2)

def padding2d_circular(input, pad):
    """input:
             [[[[0., 1., 2],
                [3., 4., 5.]]]]
            pad: (1, 2, 2, 1)
    output:
        [[[[2., 0., 1., 2., 0., 1.],
           [5., 3., 4., 5., 3., 4.],
           [2., 0., 1., 2., 0., 1.],
           [5., 3., 4., 5., 3., 4.],
           [2., 0., 1., 2., 0., 1.]]]]
    """
    input = torch.cat([input[:, :, -pad[2]:], input, input[:, :, 0:pad[3]]], dim=2)
    return torch.cat([input[:, :, :, -pad[0]:], input, input[:, :, :, 0:pad[1]]], dim=3)

def padding3d_circular(input, pad):
    """input:
            [[[[[ 0.,  1.,  2.],
                [ 3.,  4.,  5.]],
               [[ 6.,  7.,  8.],
                [ 9., 10., 11.]]]]]
        pad: (1, 2, 2, 1, 1, 2)
        output: [[[[[ 8.,  6.,  7.,  8.,  6.,  7.],
               [11.,  9., 10., 11.,  9., 10.],
               [ 8.,  6.,  7.,  8.,  6.,  7.],
               [11.,  9., 10., 11.,  9., 10.],
               [ 8.,  6.,  7.,  8.,  6.,  7.]],

              [[ 2.,  0.,  1.,  2.,  0.,  1.],
               [ 5.,  3.,  4.,  5.,  3.,  4.],
               [ 2.,  0.,  1.,  2.,  0.,  1.],
               [ 5.,  3.,  4.,  5.,  3.,  4.],
               [ 2.,  0.,  1.,  2.,  0.,  1.]],

              [[ 8.,  6.,  7.,  8.,  6.,  7.],
               [11.,  9., 10., 11.,  9., 10.],
               [ 8.,  6.,  7.,  8.,  6.,  7.],
               [11.,  9., 10., 11.,  9., 10.],
               [ 8.,  6.,  7.,  8.,  6.,  7.]],

              [[ 2.,  0.,  1.,  2.,  0.,  1.],
               [ 5.,  3.,  4.,  5.,  3.,  4.],
               [ 2.,  0.,  1.,  2.,  0.,  1.],
               [ 5.,  3.,  4.,  5.,  3.,  4.],
               [ 2.,  0.,  1.,  2.,  0.,  1.]],

              [[ 8.,  6.,  7.,  8.,  6.,  7.],
               [11.,  9., 10., 11.,  9., 10.],
               [ 8.,  6.,  7.,  8.,  6.,  7.],
               [11.,  9., 10., 11.,  9., 10.],
               [ 8.,  6.,  7.,  8.,  6.,  7.]]]]]
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.testing._internal.common_nn.padding3d_circular', 'padding3d_circular(input, pad)', {'torch': torch, 'input': input, 'pad': pad}, 1)
loss_reference_fns = {'KLDivLoss': kldivloss_reference, 'NLLLoss': nllloss_reference, 'NLLLossNd': nlllossNd_reference, 'SmoothL1Loss': smoothl1loss_reference, 'MultiLabelMarginLoss': multilabelmarginloss_reference, 'HingeEmbeddingLoss': hingeembeddingloss_reference, 'SoftMarginLoss': softmarginloss_reference, 'MultiMarginLoss': multimarginloss_reference, 'CosineEmbeddingLoss': cosineembeddingloss_reference, 'TripletMarginLoss': tripletmarginloss_reference, 'MarginRankingLoss': marginrankingloss_reference, 'CTCLoss': ctcloss_reference}
criterion_tests = [dict(module_name='L1Loss', input_size=(2, 3, 4), target_size=(2, 3, 4), reference_fn=lambda i, t, _: 1.0 / i.numel() * sum(((a - b).abs().sum() for (a, b) in zip(i, t)))), dict(module_name='NLLLoss', input_fn=lambda: torch.rand(15, 10).log(), target_fn=lambda: torch.Tensor(15).uniform_().mul(10).floor().long(), reference_fn=lambda i, t, m: nllloss_reference(i, t, reduction=get_reduction(m)), check_sum_reduction=True, check_bfloat16=TEST_WITH_ROCM), dict(module_name='NLLLoss', constructor_args=(None, None, 2), cpp_constructor_args='torch::nn::NLLLossOptions().weight({}).ignore_index(2)', input_fn=lambda: torch.rand(15, 10).log(), target_fn=lambda: torch.Tensor(15).uniform_().mul(10).floor().long(), reference_fn=lambda i, t, _: nllloss_reference(i, t, ignore_index=2), desc='ignore_index', check_bfloat16=TEST_WITH_ROCM), dict(module_name='NLLLoss', constructor_args_fn=lambda: (torch.rand(10), ), cpp_constructor_args='torch::nn::NLLLossOptions().weight(torch::rand(10))', input_fn=lambda: torch.rand(15, 10).add(0.01).log(), target_fn=lambda: torch.Tensor(15).uniform_().mul(10).floor().long(), reference_fn=lambda i, t, m: nllloss_reference(i, t, weight=get_weight(m)), desc='weights', check_bfloat16=TEST_WITH_ROCM), dict(module_name='NLLLoss', constructor_args_fn=lambda: (torch.rand(10), None, 2), cpp_constructor_args='torch::nn::NLLLossOptions().weight(torch::rand(10)).ignore_index(2)', input_fn=lambda: torch.rand(15, 10).add(0.01).log(), target_fn=lambda: torch.Tensor(15).uniform_().mul(10).floor().long(), reference_fn=lambda i, t, m: nllloss_reference(i, t, weight=get_weight(m), ignore_index=2), desc='weights_ignore_index', check_bfloat16=TEST_WITH_ROCM), dict(module_name='NLLLoss', constructor_args_fn=lambda: (torch.rand(10), None, -1), cpp_constructor_args='torch::nn::NLLLossOptions().weight(torch::rand(10)).ignore_index(-1)', input_fn=lambda: torch.rand(15, 10).add(0.01).log(), target_fn=lambda: torch.Tensor(15).uniform_().mul(10 + 1).floor().long() - 1, reference_fn=lambda i, t, m: nllloss_reference(i, t, weight=get_weight(m), ignore_index=-1), desc='weights_ignore_index_neg', check_bfloat16=TEST_WITH_ROCM), dict(module_name='KLDivLoss', input_fn=lambda: torch.rand(10, 10).log(), target_fn=lambda: torch.rand(10, 10), reference_fn=lambda i, t, m: kldivloss_reference(i, t, get_reduction(m)), check_sum_reduction=True), dict(module_name='MSELoss', input_size=(2, 3, 4, 5), target_size=(2, 3, 4, 5), reference_fn=lambda i, t, m: (i - t).abs().pow(2).sum() / ((i.numel() if get_reduction(m) == 'mean' else 1)), check_sum_reduction=True), dict(module_name='BCELoss', input_fn=lambda: torch.rand(15, 10).clamp_(0.01, 1 - 0.01), target_fn=lambda: torch.randn(15, 10).gt(0).double(), reference_fn=lambda i, t, m: -(t * i.log() + (1 - t) * (1 - i).log()).sum() / ((i.numel() if get_reduction(m) else 1)), check_gradgrad=False, check_bfloat16=TEST_WITH_ROCM), dict(module_name='BCELoss', constructor_args_fn=lambda: (torch.rand(10), ), cpp_constructor_args='torch::nn::BCELossOptions().weight(torch::rand(10))', input_fn=lambda: torch.rand(15, 10).clamp_(0.01, 1 - 0.01), target_fn=lambda: torch.randn(15, 10).gt(0).double(), reference_fn=lambda i, t, m: -((t * i.log() + (1 - t) * (1 - i).log()) * get_weight(m)).sum() / ((i.numel() if get_reduction(m) else 1)), desc='weights', check_gradgrad=False, check_bfloat16=TEST_WITH_ROCM), dict(module_name='CrossEntropyLoss', input_size=(15, 10), target_fn=lambda: torch.Tensor(15).uniform_().mul(10).floor().long()), dict(module_name='CrossEntropyLoss', constructor_args_fn=lambda: (torch.rand(10), ), cpp_constructor_args='torch::nn::CrossEntropyLossOptions().weight(torch::rand(10))', input_size=(15, 10), target_fn=lambda: torch.Tensor(15).uniform_().mul(10).floor().long(), desc='weights'), dict(module_name='HingeEmbeddingLoss', input_size=(10, ), target_fn=lambda: torch.randn(10).gt(0).double().mul_(2).sub(1), reference_fn=lambda i, t, m: hingeembeddingloss_reference(i, t, reduction=get_reduction(m)), check_sum_reduction=True), dict(module_name='HingeEmbeddingLoss', constructor_args=(0.5, ), cpp_constructor_args='torch::nn::HingeEmbeddingLossOptions().margin(0.5)', input_size=(10, ), target_fn=lambda: torch.randn(10).gt(0).double().mul_(2).sub(1), reference_fn=lambda i, t, m: hingeembeddingloss_reference(i, t, margin=0.5, reduction=get_reduction(m)), desc='margin', check_sum_reduction=True), dict(module_name='MultiLabelMarginLoss', input_size=(10, ), target_fn=lambda: torch.rand(10).mul(10).floor().long(), reference_fn=lambda i, t, m: multilabelmarginloss_reference(i, t, reduction=get_reduction(m)), desc='1d', check_sum_reduction=True, check_gradgrad=False, check_bfloat16=TEST_WITH_ROCM), dict(module_name='MultiLabelMarginLoss', input_size=(5, 10), target_fn=lambda: torch.rand(5, 10).mul(10).floor().long(), reference_fn=lambda i, t, m: multilabelmarginloss_reference(i, t, reduction=get_reduction(m)), check_sum_reduction=True, check_gradgrad=False, check_bfloat16=TEST_WITH_ROCM), dict(module_name='MultiLabelSoftMarginLoss', input_size=(5, 10), target_fn=lambda: torch.rand(5, 10).mul(2).floor(), reference_fn=lambda i, t, m: -(t * i.sigmoid().log() + (1 - t) * (-i).sigmoid().log()).sum() / i.numel(), check_gradgrad=False), dict(module_name='MultiMarginLoss', input_size=(5, 10), target_fn=lambda: torch.rand(5).mul(8).floor().long(), reference_fn=lambda i, t, m: multimarginloss_reference(i, t, reduction=get_reduction(m)), check_sum_reduction=True, check_gradgrad=False), dict(module_name='MultiMarginLoss', input_size=(10, ), target_fn=lambda: torch.rand(1).mul(8).floor().long(), reference_fn=lambda i, t, m: multimarginloss_reference(i, t, reduction=get_reduction(m)), desc='1d', check_sum_reduction=True, check_gradgrad=False), dict(module_name='MultiMarginLoss', constructor_args=(2, ), cpp_constructor_args='torch::nn::MultiMarginLossOptions().p(2)', input_fn=lambda: torch.rand(5, 10).clamp_(0.01, 1 - 0.01), target_fn=lambda: torch.rand(5).mul(8).floor().long(), reference_fn=lambda i, t, m: multimarginloss_reference(i, t, p=2, reduction=get_reduction(m)), desc='p', check_sum_reduction=True, check_gradgrad=False), dict(module_name='MultiMarginLoss', constructor_args=(1, 0.5), cpp_constructor_args='torch::nn::MultiMarginLossOptions().p(1).margin(0.5)', legacy_constructor_args=(1, None, 0.5), input_size=(5, 10), target_fn=lambda: torch.rand(5).mul(8).floor().long(), reference_fn=lambda i, t, m: multimarginloss_reference(i, t, margin=0.5, reduction=get_reduction(m)), desc='margin', check_sum_reduction=True, check_gradgrad=False), dict(module_name='MultiMarginLoss', constructor_args=(1, 1.0, torch.rand(10)), cpp_constructor_args='torch::nn::MultiMarginLossOptions().p(1).margin(1.).weight(torch::rand(10))', legacy_constructor_args=(1, torch.rand(10)), input_size=(5, 10), target_fn=lambda: torch.rand(5).mul(8).floor().long(), reference_fn=lambda i, t, m: multimarginloss_reference(i, t, weight=get_weight(m), reduction=get_reduction(m)), desc='weights', check_sum_reduction=True, check_gradgrad=False), dict(module_name='SmoothL1Loss', input_size=(5, 10), target_size=(5, 10), check_sum_reduction=True, reference_fn=lambda i, t, m: smoothl1loss_reference(i, t, reduction=get_reduction(m))), dict(module_name='SoftMarginLoss', input_size=(5, 5), target_fn=lambda: torch.randn(5, 5).sign(), reference_fn=lambda i, t, m: softmarginloss_reference(i, t, reduction=get_reduction(m)), check_sum_reduction=True), dict(module_name='CosineEmbeddingLoss', input_fn=lambda: (torch.rand(15, 10), torch.rand(15, 10)), target_fn=lambda: torch.randn(15).sign(), reference_fn=lambda i, t, m: cosineembeddingloss_reference(i[0], i[1], t, reduction=get_reduction(m)), check_sum_reduction=True), dict(module_name='CosineEmbeddingLoss', constructor_args=(0.7, ), cpp_constructor_args='torch::nn::CosineEmbeddingLossOptions().margin(0.7)', input_fn=lambda: (torch.rand(15, 10), torch.rand(15, 10)), target_fn=lambda: torch.randn(15).sign(), reference_fn=lambda i, t, m: cosineembeddingloss_reference(i[0], i[1], t, margin=0.7, reduction=get_reduction(m)), desc='margin', check_sum_reduction=True), dict(module_name='MarginRankingLoss', input_fn=lambda: (torch.randn(50).mul(10), torch.randn(50).mul(10)), target_fn=lambda: torch.randn(50).sign(), reference_fn=lambda i, t, m: marginrankingloss_reference(i[0], i[1], t, reduction=get_reduction(m)), check_sum_reduction=True), dict(module_name='MarginRankingLoss', constructor_args=(0.5, ), cpp_constructor_args='torch::nn::MarginRankingLossOptions().margin(0.5)', input_fn=lambda: (torch.randn(50).mul(10), torch.randn(50).mul(10)), target_fn=lambda: torch.randn(50).sign(), reference_fn=lambda i, t, m: marginrankingloss_reference(i[0], i[1], t, margin=0.5, reduction=get_reduction(m)), desc='margin', check_sum_reduction=True)]
new_criterion_tests = [dict(module_name='BCEWithLogitsLoss', input_fn=lambda: torch.rand(15, 10).clamp_(0.01, 1 - 0.01), target_fn=lambda: torch.randn(15, 10).gt(0).double()), dict(module_name='BCEWithLogitsLoss', constructor_args=(torch.rand(10), ), cpp_constructor_args='torch::nn::BCEWithLogitsLossOptions().weight(torch::rand(10))', input_fn=lambda: torch.rand(15, 10).clamp_(0.01, 1 - 0.01), target_fn=lambda: torch.randn(15, 10).gt(0).double(), desc='weights'), dict(module_name='BCEWithLogitsLoss', constructor_args=(torch.rand(()), ), cpp_constructor_args='torch::nn::BCEWithLogitsLossOptions().weight(torch::rand({}))', input_fn=lambda: torch.rand(()).clamp_(0.01, 1 - 0.01), target_fn=lambda: torch.randn(()).gt(0).double(), desc='scalar_weights'), dict(module_name='NLLLoss', input_size=(2, 3, 5, 5), target_fn=lambda: torch.rand(2, 5, 5).mul(3).floor().long(), reference_fn=lambda i, t, m: loss_reference_fns['NLLLossNd'](i, t, reduction=get_reduction(m)), check_sum_reduction=True, desc='2d', check_bfloat16=TEST_WITH_ROCM), dict(module_name='NLLLoss', constructor_args_fn=lambda: (torch.rand(3), ), cpp_constructor_args='torch::nn::NLLLossOptions().weight(torch::rand(3))', input_size=(2, 3, 5, 5), target=torch.rand(2, 5, 5).mul(3).floor().long(), reference_fn=lambda i, t, m: loss_reference_fns['NLLLossNd'](i, t, weight=get_weight(m)), desc='2d_weights', check_bfloat16=TEST_WITH_ROCM), dict(module_name='NLLLoss', constructor_args=(None, None, 1), cpp_constructor_args='torch::nn::NLLLossOptions().weight({}).ignore_index(1)', input_size=(2, 3, 5, 5), target_fn=lambda: torch.rand(2, 5, 5).mul(3).floor().long(), reference_fn=lambda i, t, m: loss_reference_fns['NLLLossNd'](i, t, ignore_index=1), desc='2d_ignore_index', check_bfloat16=TEST_WITH_ROCM), dict(module_name='NLLLoss', input_size=(2, 3, 5, 5, 2, 2), target_fn=lambda: torch.rand(2, 5, 5, 2, 2).mul(3).floor().long(), reference_fn=lambda i, t, m: loss_reference_fns['NLLLossNd'](i, t, reduction=get_reduction(m)), check_sum_reduction=True, desc='higher_dim', check_bfloat16=TEST_WITH_ROCM), dict(module_name='NLLLoss', input_size=(2, 3, 5), target_fn=lambda: torch.rand(2, 5).mul(3).floor().long(), reference_fn=lambda i, t, m: loss_reference_fns['NLLLossNd'](i, t, reduction=get_reduction(m)), check_sum_reduction=True, desc='dim_is_3', check_bfloat16=TEST_WITH_ROCM), dict(module_name='PoissonNLLLoss', input_size=(2, 3, 4, 5), target_fn=lambda: torch.randn(2, 3, 4, 5).floor_().abs_(), reference_fn=lambda i, t, _: (i.exp() - t.mul(i)).mean(), desc='no_full_loss'), dict(module_name='PoissonNLLLoss', constructor_args=(False, False), cpp_constructor_args='torch::nn::PoissonNLLLossOptions().log_input(false).full(false)', input_fn=lambda: torch.randn(2, 3, 4, 5).abs_().add_(0.001), target_fn=lambda: torch.randn(2, 3, 4, 5).floor_().abs_(), reference_fn=lambda i, t, _: (i - t.mul((i + 1e-08).log())).mean(), desc='no_full_loss_no_log_input'), dict(module_name='PoissonNLLLoss', constructor_args=(True, True), cpp_constructor_args='torch::nn::PoissonNLLLossOptions().log_input(true).full(true)', input_size=(2, 3, 4, 5), target_fn=lambda: torch.randn(2, 3, 4, 5).floor_().abs_(), reference_fn=lambda i, t, _: (i.exp() - t.mul(i) + (t.mul(t.log()) - t + 0.5 * (2.0 * pi * t).log()).masked_fill(t <= 1, 0)).mean(), desc='full_loss'), dict(module_name='PoissonNLLLoss', constructor_args=(False, True), cpp_constructor_args='torch::nn::PoissonNLLLossOptions().log_input(false).full(true)', input_fn=lambda: torch.randn(2, 3, 4, 5).abs_().add_(0.001), target_fn=lambda: torch.randn(2, 3, 4, 5).floor_().abs_(), reference_fn=lambda i, t, _: (i - t.mul((i + 1e-08).log()) + (t.mul(t.log()) - t + 0.5 * (2.0 * pi * t).log()).masked_fill(t <= 1, 0)).mean(), desc='full_loss_no_log_input'), dict(module_name='L1Loss', input_size=(), target_size=(), reference_fn=lambda i, t, _: 1.0 / i.numel() * (i - t).abs().sum(), desc='scalar'), dict(module_name='KLDivLoss', input_fn=lambda: torch.rand(()).log(), target_fn=lambda: torch.rand(()), reference_fn=lambda i, t, m: kldivloss_reference(i, t, get_reduction(m)), check_sum_reduction=True, desc='scalar'), dict(module_name='MSELoss', input_size=(), target_size=(), reference_fn=lambda i, t, m: (i - t).abs().pow(2).sum() / ((i.numel() if get_reduction(m) == 'mean' else 1)), check_sum_reduction=True, desc='scalar', check_bfloat16=TEST_WITH_ROCM), dict(module_name='MSELoss', input_fn=lambda: torch.ones(5, 68, 64, 64, dtype=torch.float) / 10, target_fn=lambda: torch.zeros(5, 68, 64, 64, dtype=torch.float), reference_fn=lambda i, t, m: (i - t).abs().pow(2).sum() / ((i.numel() if get_reduction(m) == 'mean' else 1)), check_forward_only=True, desc='prec', check_bfloat16=TEST_WITH_ROCM), dict(module_name='BCELoss', constructor_args_fn=lambda: (torch.rand(()), ), cpp_constructor_args='torch::nn::BCELossOptions().weight(torch::rand({}))', input_fn=lambda: torch.rand(()).clamp_(0.01, 1 - 0.01), target_fn=lambda: torch.rand(()).gt(0).double(), reference_fn=lambda i, t, m: -((t * i.log() + (1 - t) * (1 - i).log()) * get_weight(m)).sum() / ((i.numel() if get_reduction(m) == 'mean' else 1)), desc='scalar_weights', check_gradgrad=False, check_bfloat16=TEST_WITH_ROCM), dict(module_name='HingeEmbeddingLoss', constructor_args=(0.5, ), cpp_constructor_args='torch::nn::HingeEmbeddingLossOptions().margin(0.5)', input_size=(), target_fn=lambda: torch.randn(()).gt(0).double().mul_(2).sub(1), desc='scalar_margin', check_sum_reduction=True), dict(module_name='SmoothL1Loss', input_size=(), target_size=(), check_sum_reduction=True, reference_fn=lambda i, t, m: smoothl1loss_reference(i, t, reduction=get_reduction(m)), desc='scalar'), dict(module_name='MultiLabelSoftMarginLoss', constructor_args=(torch.rand(10), ), cpp_constructor_args='torch::nn::MultiLabelSoftMarginLossOptions().weight(torch::rand(10))', input_fn=lambda: torch.randn(5, 10), target_fn=lambda: torch.rand(5, 10).mul(2).floor(), reference_fn=lambda i, t, m: -((t * i.sigmoid().log() + (1 - t) * (-i).sigmoid().log()) * get_weight(m)).sum() / ((i.numel() if get_reduction(m) == 'mean' else (i.size(1) if get_reduction(m) == 'sum' else 1))), desc='weights', check_sum_reduction=True, check_gradgrad=False), dict(module_name='CTCLoss', constructor_args=(14, ), extra_args=([50, 50, 50], [30, 25, 20]), input_fn=lambda: torch.randn(50, 3, 15).log_softmax(2), target_fn=lambda: torch.randint(0, 14, (3, 30), dtype=torch.long), reference_fn=lambda i, t, il, tl, m: ctcloss_reference(i, t, il, tl, blank=14, reduction=get_reduction(m)), desc='lengths_intlists', check_sum_reduction=True, check_gradgrad=False, check_half=False, test_cpp_api_parity=False), dict(module_name='CTCLoss', constructor_args=(14, ), cpp_constructor_args='torch::nn::CTCLossOptions().blank(14)', extra_args=(torch.tensor([50, 50, 50]), torch.tensor([30, 25, 20])), input_fn=lambda: torch.randn(50, 3, 15).log_softmax(2), target_fn=lambda: torch.randint(0, 14, (3, 30), dtype=torch.long), reference_fn=lambda i, t, il, tl, m: ctcloss_reference(i, t, il, tl, blank=14, reduction=get_reduction(m)), desc='lengths_tensors', check_sum_reduction=True, check_gradgrad=False, check_half=False), dict(module_name='CTCLoss', desc='2d_int_target_lengths_intlists', constructor_args=(0, ), extra_args=([50, 50, 50], [30, 25, 20]), input_fn=lambda: torch.randn(50, 3, 15).log_softmax(2), target_fn=lambda: torch.randint(1, 15, (3, 30), dtype=torch.int), reference_fn=lambda i, t, il, tl, m: ctcloss_reference(i, t, il, tl, blank=0, reduction=get_reduction(m)), check_sum_reduction=True, check_gradgrad=False, check_half=False, convert_target=False, test_cpp_api_parity=False), dict(module_name='CTCLoss', desc='2d_int_target_lengths_tensors', constructor_args=(0, ), cpp_constructor_args='torch::nn::CTCLossOptions().blank(0)', extra_args=(torch.tensor([50, 50, 50]), torch.tensor([30, 25, 20])), input_fn=lambda: torch.randn(50, 3, 15).log_softmax(2), target_fn=lambda: torch.randint(1, 15, (3, 30), dtype=torch.int), reference_fn=lambda i, t, il, tl, m: ctcloss_reference(i, t, il, tl, blank=0, reduction=get_reduction(m)), check_sum_reduction=True, check_gradgrad=False, check_half=False, convert_target=False), dict(module_name='CTCLoss', desc='2d_lengths_tensors', constructor_args=(0, ), cpp_constructor_args='torch::nn::CTCLossOptions().blank(0)', extra_args=(torch.tensor([50, 50, 50]), torch.tensor([30, 25, 20])), input_fn=lambda: torch.randn(50, 3, 15).log_softmax(2), target_fn=lambda: torch.randint(1, 15, (3, 30), dtype=torch.int), reference_fn=lambda i, t, il, tl, m: ctcloss_reference(i, t, il, tl, blank=0, reduction=get_reduction(m)), check_sum_reduction=True, check_gradgrad=False, check_half=False, convert_target=False)]


class NNTestCase(TestCase):
    
    def _jacobian(self, input, num_out):
        if isinstance(input, tuple):
            return tuple((self._jacobian(elem, num_out) for elem in input))
        elif isinstance(input, list):
            return [self._jacobian(elem, num_out) for elem in input]
        else:
            return torch.zeros(input.nelement(), num_out)
    
    def _flatten_tensors(self, x):
        if isinstance(x, torch.Tensor):
            if x.is_sparse:
                return x.to_dense().view(-1)
            else:
                return x.view(-1)
        else:
            return tuple((self._flatten_tensors(a) for a in x))
    
    def _zero_grad_input(self, input):
        if isinstance(input, torch.Tensor):
            if (input.requires_grad and input.grad is not None):
                input.grad.zero_()
                input.grad.detach_()
        else:
            for i in input:
                self._zero_grad_input(i)
    
    def _analytical_jacobian(self, module, input, jacobian_input=True, jacobian_parameters=True):
        output = self._forward(module, input)
        output_size = output.nelement()
        if jacobian_input:
            jacobian_inp = self._jacobian(input, output_size)
            flat_jacobian_input = list(iter_tensors(jacobian_inp))
        if jacobian_parameters:
            num_param = sum((p.numel() for p in self._get_parameters(module)[0]))
            jacobian_param = torch.zeros(num_param, output_size)
        for i in range(output_size):
            (param, d_param) = self._get_parameters(module)
            d_param = [(torch.zeros_like(p) if d is None else d) for (p, d) in zip(param, d_param)]
            d_out = torch.zeros_like(output)
            flat_d_out = d_out.view(-1)
            flat_d_out[i] = 1
            if jacobian_parameters:
                self._zero_grad_parameters(module)
            if jacobian_input:
                self._zero_grad_input(input)
            d_input = self._backward(module, input, output, d_out)
            if jacobian_input:
                for (jacobian_x, d_x) in zip(flat_jacobian_input, iter_tensors(d_input)):
                    jacobian_x[:, i] = d_x.contiguous().view(-1)
            if jacobian_parameters:
                jacobian_param[:, i] = torch.cat(self._flatten_tensors(d_param), 0)
        res = tuple()
        if jacobian_input:
            res += (jacobian_inp, )
        if jacobian_parameters:
            res += (jacobian_param, )
        return res
    
    def _numerical_jacobian(self, module, input, jacobian_input=True, jacobian_parameters=True):
        
        def fw(input):
            return self._forward(module, input).detach()
        res = tuple()
        if jacobian_input:
            res += (get_numerical_jacobian(fw, input, eps=1e-06), )
        if jacobian_parameters:
            (param, _) = self._get_parameters(module)
            res += (torch.cat([get_numerical_jacobian(fw, input, p, eps=1e-06) for p in param], 0), )
        return res
    
    def check_jacobian(self, module, input, jacobian_input=True):
        jacobian_parameters = bool(self._get_parameters(module)[0])
        analytical = self._analytical_jacobian(module, input, jacobian_input, jacobian_parameters)
        numerical = self._numerical_jacobian(module, input, jacobian_input, jacobian_parameters)
        analytical_t = list(iter_tensors(analytical))
        numerical_t = list(iter_tensors(numerical))
        if input.numel() != 0:
            self.assertLessEqual(max((a.add(n, alpha=-1).abs().max() for (a, n) in zip(analytical_t, numerical_t))), PRECISION)
    
    def check_criterion_jacobian(self, criterion, input, target):
        eps = 1e-06
        self._forward_criterion(criterion, input, target)
        analytical_d_x = self._backward_criterion(criterion, input, target)
        numerical_d_x = deepcopy(analytical_d_x)
        input_t = iter_tensors(input)
        numerical_t = iter_tensors(numerical_d_x)
        for (x, d_x) in zip(input_t, numerical_t):
            x = x.view(-1).data
            d_x = d_x.view(-1).data
            for i in range(x.nelement()):
                original = x[i].item()
                x[i] = original + eps
                fx1 = self._forward_criterion(criterion, input, target)
                x[i] = original - eps
                fx2 = self._forward_criterion(criterion, input, target)
                deriv = (fx1 - fx2) / (2.0 * eps)
                d_x[i] = float(deriv)
                x[i] = original
        analytical_t = list(iter_tensors(analytical_d_x))
        numerical_t = list(iter_tensors(numerical_d_x))
        self.assertLessEqual(max((a.add(n, alpha=-1).abs().max() for (a, n) in zip(analytical_t, numerical_t))), PRECISION)



class TestBase(object):
    _required_arg_names = {'constructor_args', 'input', 'extra_args'}
    
    def __init__(self, constructor, desc='', reference_fn=None, fullname=None, **kwargs):
        self.desc = desc
        self.fullname = fullname
        self.constructor = constructor
        self.reference_fn = reference_fn
        for name in self._required_arg_names:
            if (name not in kwargs and name + '_fn' not in kwargs and name + '_size' not in kwargs):
                if name in {'constructor_args', 'extra_args'}:
                    kwargs[name] = tuple()
                else:
                    raise ValueError("{}: Specify {} by a value, a function to generate it, or it's size!".format(self.get_name(), name))
        self._extra_kwargs = kwargs
        self._arg_cache = {}
    
    def get_name(self):
        if self.fullname is not None:
            return 'test_' + self.fullname
        test_name = 'test_' + self.constructor.__name__
        if self.desc:
            test_name += '_' + self.desc
        return test_name
    
    def _unpack(self, value):
        if isinstance(value, torch.Tensor):
            return value
        elif is_iterable(value):
            return type(value)((self._unpack(v) for v in value))
        else:
            return value
    
    @property
    def constructor_args(self):
        return self._get_arg('constructor_args', True)
    
    @property
    def extra_args(self):
        return self._get_arg('extra_args', True)
    
    def _get_arg(self, name, unpack):
        assert name in self._required_arg_names
        if name not in self._arg_cache:
            fn_name = name + '_fn'
            size_name = name + '_size'
            if name in self._extra_kwargs:
                self._arg_cache[name] = self._extra_kwargs[name]
            elif fn_name in self._extra_kwargs:
                self._arg_cache[name] = self._extra_kwargs[fn_name]()
            else:
                assert size_name in self._extra_kwargs, 'Missing `{}`, `{}` or `{}` for {}'.format(name, size_name, fn_name, self.get_name())
                
                def map_tensor_sizes(sizes):
                    if isinstance(sizes, list):
                        return [map_tensor_sizes(s) for s in sizes]
                    elif isinstance(sizes, torch.Tensor):
                        return sizes.double()
                    else:
                        return torch.randn(sizes)
                self._arg_cache[name] = map_tensor_sizes(self._extra_kwargs[size_name])
        return (self._unpack(self._arg_cache[name]) if unpack else self._arg_cache[name])
    
    def _get_input(self, unpack=True):
        return self._get_arg('input', unpack)
    
    def __call__(self, test_case):
        raise NotImplementedError



class ModuleTest(TestBase):
    
    def __init__(self, *args, **kwargs):
        super(ModuleTest, self).__init__(*args, **kwargs)
        self.jacobian_input = kwargs.get('jacobian_input', True)
        self.should_test_cuda = kwargs.get('test_cuda', True)
        self.should_test_pickle = kwargs.get('pickle', True)
        self.check_gradgrad = kwargs.get('check_gradgrad', True)
        self.FIXME_no_cuda_gradgrad_comparison = kwargs.get('FIXME_no_cuda_gradgrad_comparison', False)
        self.precision = kwargs.get('precision', 0.0002)
        self.check_forward_only = kwargs.get('check_forward_only', False)
    
    def __call__(self, test_case):
        module = self.constructor(*self.constructor_args)
        input = self._get_input()
        if self.reference_fn is not None:
            out = test_case._forward(module, input)
            ref_input = deepcopy(input)
            ref_module = deepcopy(module)
            expected_out = self.reference_fn(ref_input, test_case._get_parameters(module)[0], ref_module)
            test_case.assertEqual(out, expected_out)
        if self.check_forward_only:
            return
        self.test_noncontig(test_case, module, input)
        if self.should_test_pickle:
            with TemporaryFile() as f:
                test_case._forward(module, input)
                torch.save(module, f)
                f.seek(0)
                module_copy = torch.load(f)
                test_case.assertEqual(test_case._forward(module, input), test_case._forward(module_copy, input))
        self._do_test(test_case, module, input)
    
    def noncontiguize(self, obj):
        if isinstance(obj, list):
            return [self.noncontiguize(o) for o in obj]
        tensor = obj
        ndim = tensor.dim()
        dim = ndim
        for d in range(ndim):
            if tensor.size(d) > 1:
                dim = d + 1
                break
        noncontig = torch.stack([torch.empty_like(tensor), tensor], dim).select(dim, 1).detach()
        assert (noncontig.numel() == 1 or noncontig.numel() == 0 or not noncontig.is_contiguous())
        noncontig.requires_grad = tensor.requires_grad
        return noncontig
    
    def test_noncontig(self, test_case, module, input):
        if (isinstance(input, torch.Tensor) and input.dim() == 0):
            return
        if any((i.dim() == 0 for i in input if isinstance(i, torch.Tensor))):
            return
        test_case._zero_grad_parameters(module)
        test_case._zero_grad_input(input)
        with freeze_rng_state():
            output = test_case._forward(module, input)
            grad_output = output.new(output.shape).normal_()
            output = output.clone()
            d_input = deepcopy(test_case._backward(module, input, output, grad_output))
            d_param = deepcopy(test_case._get_parameters(module)[1])
        nc_input = self.noncontiguize(input)
        nc_grad_output = self.noncontiguize(grad_output)
        for (contig_i, contig_g) in product((True, False), repeat=2):
            i = (input if contig_i else nc_input)
            go = deepcopy((grad_output if contig_g else nc_grad_output))
            test_case._zero_grad_parameters(module)
            test_case._zero_grad_input(i)
            with freeze_rng_state():
                out = test_case._forward(module, i)
                grad = test_case._backward(module, i, out, go)
                test_case.assertEqual(out, output)
                test_case.assertEqual(grad, d_input, 0.0001)
                test_case.assertEqual(test_case._get_parameters(module)[1], d_param)
    
    def test_cuda(self, test_case):
        if (not TEST_CUDA or not self.should_test_cuda):
            raise unittest.SkipTest('Excluded from CUDA tests')
        try:
            cpu_input = self._get_input()
            type_map = {'torch.DoubleTensor': torch.cuda.FloatTensor}
            gpu_input = to_gpu(cpu_input, type_map=type_map)
            cpu_module = self.constructor(*self.constructor_args)
            gpu_module = self.constructor(*self.constructor_args).float().cuda()
            cpu_param = test_case._get_parameters(cpu_module)
            gpu_param = test_case._get_parameters(gpu_module)
            for (cpu_p, gpu_p) in zip(cpu_param[0], gpu_param[0]):
                gpu_p.data.copy_(cpu_p)
            test_case._zero_grad_input(cpu_input)
            test_case._zero_grad_input(gpu_input)
            test_case._zero_grad_parameters(cpu_module)
            test_case._zero_grad_parameters(gpu_module)
            cpu_output = test_case._forward(cpu_module, cpu_input)
            gpu_output = test_case._forward(gpu_module, gpu_input)
            test_case.assertEqual(cpu_output, gpu_output, self.precision)
            for _ in range(5):
                cpu_gradOutput = cpu_output.clone().normal_()
                gpu_gradOutput = cpu_gradOutput.type('torch.cuda.FloatTensor')
                cpu_gradInput = test_case._backward(cpu_module, cpu_input, cpu_output, cpu_gradOutput)
                gpu_gradInput = test_case._backward(gpu_module, gpu_input, gpu_output, gpu_gradOutput)
                test_case.assertEqual(cpu_gradInput, gpu_gradInput, self.precision)
                for (cpu_d_p, gpu_d_p) in zip(cpu_param[1], gpu_param[1]):
                    test_case.assertEqual(cpu_d_p, gpu_d_p, self.precision)
            if (self.check_gradgrad and not self.FIXME_no_cuda_gradgrad_comparison):
                cpu_output = cpu_module(cpu_input)
                gpu_output = gpu_module(gpu_input)
                cpu_gradOutput = torch.randn_like(cpu_output, requires_grad=True)
                gpu_gradOutput = cpu_gradOutput.type_as(gpu_output).detach()
                gpu_gradOutput.requires_grad = True
                cpu_gradInputs = torch.autograd.grad(cpu_output, (cpu_input, ) + tuple(cpu_module.parameters()), cpu_gradOutput, create_graph=True)
                gpu_gradInputs = torch.autograd.grad(gpu_output, (gpu_input, ) + tuple(gpu_module.parameters()), gpu_gradOutput, create_graph=True)
                for (cpu_d_i, gpu_d_i) in zip(cpu_gradInputs, gpu_gradInputs):
                    test_case.assertEqual(cpu_d_i, gpu_d_i, self.precision)
                cpu_gg = torch.autograd.grad(cpu_output.sum() + sum(map(lambda x: x.sum(), cpu_gradInputs)), (cpu_input, cpu_gradOutput) + tuple(cpu_module.parameters()), retain_graph=True)
                gpu_gg = torch.autograd.grad(gpu_output.sum() + sum(map(lambda x: x.sum(), gpu_gradInputs)), (gpu_input, gpu_gradOutput) + tuple(gpu_module.parameters()), retain_graph=True)
                test_case.assertEqual(cpu_gradInput, gpu_gradInput, self.precision)
                for (cpu_d_p, gpu_d_p) in zip(cpu_gg, gpu_gg):
                    test_case.assertEqual(cpu_d_p, gpu_d_p, self.precision)
            self.test_noncontig(test_case, gpu_module, gpu_input)
        except NotImplementedError:
            pass
        except AttributeError as e:
            if (len(e.args) == 1 and "'FloatTensor' object has no attribute 'scatter_'" in e.args[0]):
                pass
            else:
                raise



class CriterionTest(TestBase):
    _required_arg_names = TestBase._required_arg_names.union({'target'})
    
    def __init__(self, *args, **kwargs):
        super(CriterionTest, self).__init__(*args, **kwargs)
        self.should_test_cuda = kwargs.get('test_cuda', True)
        self.check_forward_only = kwargs.get('check_forward_only', True)
    
    def _get_target(self):
        return self._get_arg('target', True)
    
    def __call__(self, test_case):
        module = self.constructor(*self.constructor_args)
        input = self._get_input()
        module.__repr__()
        str(module)
        target = self._get_target()
        if self.reference_fn is not None:
            out = test_case._forward_criterion(module, input, target, extra_args=self.extra_args)
            ref_args = (deepcopy(input), deepcopy(target)) + self.extra_args + (module, )
            expected_out = self.reference_fn(*ref_args)
            test_case.assertEqual(out, expected_out)
        if self.check_forward_only:
            return
        test_case.check_criterion_jacobian(module, input, target)
        self._do_extra_tests(test_case, module, input, target)
    
    def test_cuda(self, test_case):
        if (not TEST_CUDA or not self.should_test_cuda):
            raise unittest.SkipTest('Excluded from CUDA tests')
        try:
            cpu_input = self._get_input()
            type_map = {'torch.DoubleTensor': torch.cuda.FloatTensor}
            gpu_input = to_gpu(cpu_input, type_map=type_map)
            cpu_target = self._get_target()
            gpu_target = to_gpu(cpu_target, type_map=type_map)
            cpu_module = self.constructor(*self.constructor_args)
            gpu_module = self.constructor(*self.constructor_args).float().cuda()
            cpu_output = test_case._forward_criterion(cpu_module, cpu_input, cpu_target)
            gpu_output = test_case._forward_criterion(gpu_module, gpu_input, gpu_target)
            test_case.assertEqual(cpu_output, gpu_output, 0.0004)
            gradOutput = torch.randn(())
            cpu_gradInput = test_case._backward_criterion(cpu_module, cpu_input, cpu_target, gradOutput)
            gpu_gradInput = test_case._backward_criterion(gpu_module, gpu_input, gpu_target, gradOutput)
            test_case.assertEqual(cpu_gradInput, gpu_gradInput, 0.0004)
        except NotImplementedError:
            pass
    
    def _do_extra_tests(self, test_case, module, input, target):
        pass



class InputVariableMixin(object):
    
    def _get_input(self):
        input = TestBase._get_input(self, False)
        
        def map_variables(i):
            if isinstance(i, torch.Tensor):
                if i.is_floating_point():
                    i.requires_grad = True
                return i
            else:
                return type(i)((map_variables(elem) for elem in i))
        return map_variables(input)



class NewModuleTest(InputVariableMixin, ModuleTest):
    
    def __init__(self, *args, **kwargs):
        super(NewModuleTest, self).__init__(*args, **kwargs)
        self.cudnn = kwargs.get('cudnn', False)
        self.check_inplace = kwargs.get('check_inplace', False)
        self.check_gradgrad = kwargs.get('check_gradgrad', True)
        self.skip_double = kwargs.get('skip_double', False)
    
    def _do_test(self, test_case, module, input):
        test_case.check_jacobian(module, input, self.jacobian_input)
        if self.check_gradgrad:
            params = tuple((x for x in module.parameters()))
            _assertGradAndGradgradChecks(test_case, lambda x, *args, **kw: test_case._forward(module, x), (input, ) + params)
        module.__repr__()
        if self.check_inplace:
            module_ip = self.constructor(*self.constructor_args, inplace=True)
            input_version = input._version
            with freeze_rng_state():
                output = module(input)
            test_case.assertEqual(input._version, input_version)
            input_ip = deepcopy(input)
            input_ip_clone = input_ip.clone()
            with freeze_rng_state():
                output_ip = module_ip(input_ip_clone)
            test_case.assertNotEqual(input_ip_clone._version, input_version)
            test_case.assertEqual(output, output_ip)
            grad = output.data.clone().normal_()
            input.grad.data.zero_()
            output.backward(grad)
            output_ip.backward(grad)
            test_case.assertEqual(input.grad, input_ip.grad)
        if (isinstance(input, torch.LongTensor) and TEST_CUDA):
            input = input.cuda()
            module.float().cuda()
            module(input)
            for p in module.parameters():
                test_case.assertIsInstance(p, torch.cuda.FloatTensor)
                test_case.assertEqual(p.get_device(), 0)
            if torch.cuda.device_count() > 1:
                input = input.cuda(1)
                module.cuda(1)
                with torch.cuda.device(1):
                    module(input)
                for p in module.parameters():
                    test_case.assertIsInstance(p, torch.cuda.FloatTensor)
                    test_case.assertEqual(p.get_device(), 1)
        else:
            if not isinstance(input, torch.LongTensor):
                input = input.float()
            module.float()
            module(input)
            for p in module.parameters():
                test_case.assertIsInstance(p, torch.FloatTensor)
            if not isinstance(input, torch.LongTensor):
                input = input.double()
            module.double()
            module(input)
            for p in module.parameters():
                test_case.assertIsInstance(p, torch.DoubleTensor)
            if (TEST_CUDA and self.should_test_cuda):
                input = input.float().cuda()
                module.float().cuda()
                module(input)
                for p in module.parameters():
                    test_case.assertIsInstance(p, torch.cuda.FloatTensor)
                    test_case.assertEqual(p.get_device(), 0)
                input = input.cpu()
                module.cpu()
                module(input)
                for p in module.parameters():
                    test_case.assertIsInstance(p, torch.FloatTensor)
                input = input.cuda()
                module.cuda()
                module(input)
                for p in module.parameters():
                    test_case.assertIsInstance(p, torch.cuda.FloatTensor)
                    test_case.assertEqual(p.get_device(), 0)
                if self.cudnn:
                    with torch.backends.cudnn.flags(enabled=False):
                        module(input)
                        for p in module.parameters():
                            test_case.assertIsInstance(p, torch.cuda.FloatTensor)
                            test_case.assertEqual(p.get_device(), 0)
                if torch.cuda.device_count() >= 2:
                    input = input.cuda(1)
                    module.cuda(1)
                    with torch.cuda.device(1):
                        module(input)
                    for p in module.parameters():
                        test_case.assertIsInstance(p, torch.cuda.FloatTensor)
                        test_case.assertEqual(p.get_device(), 1)
                if not self.skip_double:
                    input = input.double().cuda()
                    module.double().cuda()
                    module(input)
                    for p in module.parameters():
                        test_case.assertIsInstance(p, torch.cuda.DoubleTensor)
                        test_case.assertEqual(p.get_device(), 0)
                input = input.half().cuda()
                module.half().cuda()
                module(input)
                for p in module.parameters():
                    test_case.assertIsInstance(p, torch.cuda.HalfTensor)
                    test_case.assertEqual(p.get_device(), 0)
    
    def _get_target(self):
        return self._get_arg('target', False)
    
    @property
    def constructor_args(self):
        return self._get_arg('constructor_args', False)



class NewCriterionTest(InputVariableMixin, CriterionTest):
    
    def __init__(self, *args, **kwargs):
        super(NewCriterionTest, self).__init__(*args, **kwargs)
        self.check_gradgrad = kwargs.get('check_gradgrad', True)
        self.check_half = kwargs.get('check_half', True)
        self.check_bfloat16 = kwargs.get('check_bfloat16', False)
        self.convert_target = kwargs.get('convert_target', True)
    
    def _do_extra_tests(self, test_case, module, input, target):
        if not self.check_gradgrad:
            return
        test_case.assertFalse(target.requires_grad)
        params = tuple((x for x in module.parameters()))
        if not isinstance(input, tuple):
            inputs = (input, ) + params
            
            def apply_fn(input, *params):
                return module(input, target)
        else:
            inputs = input + params
            
            def apply_fn(input1, input2, *params):
                return module(input1, input2, target)
        gradcheck(apply_fn, inputs)
        gradgradcheck(apply_fn, inputs)
    
    def test_cuda(self, test_case, dtype=None, extra_args=None):
        
        def convert_dtype(obj, dtype, requires_grad=False):
            if isinstance(obj, torch.Tensor):
                return obj.detach().to(dtype=dtype).requires_grad_(requires_grad)
            elif isinstance(obj, torch.Tensor):
                return obj.to(dtype)
            elif isinstance(obj, tuple):
                return tuple((convert_dtype(o, dtype, requires_grad) for o in obj))
            else:
                return obj
        if (not TEST_CUDA or not self.should_test_cuda):
            raise unittest.SkipTest('Excluded from CUDA tests')
        try:
            cpu_input = self._get_input()
            cpu_target = self._get_target()
            cpu_module = self.constructor(*self.constructor_args)
            gpu_module = self.constructor(*self.constructor_args)
            if dtype is not None:
                cpu_input = convert_dtype(cpu_input, dtype, True)
                if (not isinstance(cpu_target, torch.LongTensor) and self.convert_target):
                    cpu_target = convert_dtype(cpu_target, dtype)
                cpu_module.type(dtype)
                gpu_module.type(dtype)
            gpu_input = to_gpu(cpu_input)
            gpu_target = to_gpu(cpu_target)
            gpu_module.cuda()
            if dtype in {torch.half, torch.bfloat16}:
                cpu_input = self._get_input()
                cpu_target = self._get_target()
                cpu_module = self.constructor(*self.constructor_args)
            cpu_output = test_case._forward_criterion(cpu_module, cpu_input, cpu_target, extra_args=extra_args)
            gpu_output = test_case._forward_criterion(gpu_module, gpu_input, gpu_target, extra_args=extra_args)
            test_case.assertEqual(cpu_output, gpu_output, (0.1 if dtype in {torch.half, torch.bfloat16} else 0.0004))
            cpu_gradInput = test_case._backward_criterion(cpu_module, cpu_input, cpu_target, extra_args=extra_args)
            gpu_gradInput = test_case._backward_criterion(gpu_module, gpu_input, gpu_target, extra_args=extra_args)
            test_case.assertEqual(cpu_gradInput, gpu_gradInput, (0.1 if dtype in {torch.half, torch.bfloat16} else 0.0004))
        except NotImplementedError:
            pass
    
    def _get_target(self):
        return self._get_arg('target', False)
    
    @property
    def constructor_args(self):
        return self._get_arg('constructor_args', False)
    
    @property
    def extra_args(self):
        return self._get_arg('extra_args', False)


