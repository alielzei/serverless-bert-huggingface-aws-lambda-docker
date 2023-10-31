from torch import nn
import torch.nn.intrinsic as nni
import torch.nn.intrinsic.quantized as nniq
import torch.nn.intrinsic.qat as nniqat
import torch.nn.quantized as nnq
import torch.nn.quantized.dynamic as nnqd
import torch.nn.qat as nnqat
from .stubs import QuantStub, DeQuantStub
DEFAULT_MODULE_MAPPING = {nn.Linear: nnq.Linear, nn.ReLU: nnq.ReLU, nn.ReLU6: nnq.ReLU6, nn.Conv2d: nnq.Conv2d, nn.Conv3d: nnq.Conv3d, nn.BatchNorm2d: nnq.BatchNorm2d, nn.BatchNorm3d: nnq.BatchNorm3d, QuantStub: nnq.Quantize, DeQuantStub: nnq.DeQuantize, nnq.FloatFunctional: nnq.QFunctional, nni.ConvReLU2d: nniq.ConvReLU2d, nni.ConvReLU3d: nniq.ConvReLU3d, nni.LinearReLU: nniq.LinearReLU, nniqat.ConvReLU2d: nniq.ConvReLU2d, nniqat.LinearReLU: nniq.LinearReLU, nniqat.ConvBn2d: nnq.Conv2d, nniqat.ConvBnReLU2d: nniq.ConvReLU2d, nnqat.Linear: nnq.Linear, nnqat.Conv2d: nnq.Conv2d}
DEFAULT_QAT_MODULE_MAPPING = {nn.Linear: nnqat.Linear, nn.Conv2d: nnqat.Conv2d, nni.ConvBn2d: nniqat.ConvBn2d, nni.ConvBnReLU2d: nniqat.ConvBnReLU2d, nni.ConvReLU2d: nniqat.ConvReLU2d, nni.LinearReLU: nniqat.LinearReLU}
DEFAULT_DYNAMIC_MODULE_MAPPING = {nn.Linear: nnqd.Linear, nn.LSTM: nnqd.LSTM}
_EXCLUDE_QCONFIG_PROPAGATE_LIST = {DeQuantStub}
_INCLUDE_QCONFIG_PROPAGATE_LIST = {nn.Sequential}
DEFAULT_QCONFIG_PROPAGATE_WHITE_LIST = (set(DEFAULT_MODULE_MAPPING.keys()) | set(DEFAULT_QAT_MODULE_MAPPING.keys()) | set(DEFAULT_DYNAMIC_MODULE_MAPPING.keys()) | _INCLUDE_QCONFIG_PROPAGATE_LIST) - _EXCLUDE_QCONFIG_PROPAGATE_LIST

