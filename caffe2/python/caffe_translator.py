import argparse
import copy
import logging
import re
import numpy as np
from caffe2.proto import caffe2_pb2, caffe2_legacy_pb2
from caffe.proto import caffe_pb2
from caffe2.python import core, utils, workspace
from google.protobuf import text_format
logging.basicConfig()
log = logging.getLogger('caffe_translator')
log.setLevel(logging.INFO)

def _StateMeetsRule(state, rule):
    """A function that reproduces Caffe's StateMeetsRule functionality."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.caffe_translator._StateMeetsRule', '_StateMeetsRule(state, rule)', {'state': state, 'rule': rule}, 1)

def _ShouldInclude(net_state, layer):
    """A function that reproduces Caffe's inclusion and exclusion rule."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.caffe_translator._ShouldInclude', '_ShouldInclude(net_state, layer)', {'_StateMeetsRule': _StateMeetsRule, 'net_state': net_state, 'layer': layer}, 1)

def _GetLegacyDims(net, net_params, dummy_input, legacy_pad_ops):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.caffe_translator._GetLegacyDims', '_GetLegacyDims(net, net_params, dummy_input, legacy_pad_ops)', {'workspace': workspace, 'utils': utils, 'net': net, 'net_params': net_params, 'dummy_input': dummy_input, 'legacy_pad_ops': legacy_pad_ops}, 1)

def _GetLegacyPadArgs(op_def, arg_map):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.caffe_translator._GetLegacyPadArgs', '_GetLegacyPadArgs(op_def, arg_map)', {'op_def': op_def, 'arg_map': arg_map}, 1)

def _AdjustDims(op_def, arg_map, pads, dim1, dim2):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.caffe_translator._AdjustDims', '_AdjustDims(op_def, arg_map, pads, dim1, dim2)', {'caffe2_pb2': caffe2_pb2, 'op_def': op_def, 'arg_map': arg_map, 'pads': pads, 'dim1': dim1, 'dim2': dim2}, 0)

def _RemoveLegacyPad(net, net_params, input_dims):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.caffe_translator._RemoveLegacyPad', '_RemoveLegacyPad(net, net_params, input_dims)', {'re': re, 'np': np, '_GetLegacyDims': _GetLegacyDims, 'workspace': workspace, 'utils': utils, '_GetLegacyPadArgs': _GetLegacyPadArgs, '_AdjustDims': _AdjustDims, 'net': net, 'net_params': net_params, 'input_dims': input_dims}, 1)

def _GetBlobDimMap(net, net_params, dummy_input):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.caffe_translator._GetBlobDimMap', '_GetBlobDimMap(net, net_params, dummy_input)', {'workspace': workspace, 'utils': utils, 'net': net, 'net_params': net_params, 'dummy_input': dummy_input}, 1)

def _GetInputDims(caffe_net):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.caffe_translator._GetInputDims', '_GetInputDims(caffe_net)', {'caffe_net': caffe_net}, 1)


class TranslatorRegistry(object):
    registry_ = {}
    
    @classmethod
    def Register(cls, op_name):
        """A decorator for registering gradient mappings."""
        
        def Wrapper(func):
            cls.registry_[op_name] = func
            return func
        return Wrapper
    
    @classmethod
    def TranslateLayer(cls, layer, pretrained_blobs, is_test, **kwargs):
        try:
            (caffe_ops, params) = cls.registry_[layer.type](layer, pretrained_blobs, is_test, **kwargs)
        except KeyError:
            raise KeyError('No translator registered for layer: %s yet.' % str(layer))
        if caffe_ops is None:
            caffe_ops = []
        if type(caffe_ops) is not list:
            caffe_ops = [caffe_ops]
        return (caffe_ops, params)
    
    @classmethod
    def TranslateModel(cls, caffe_net, pretrained_net, is_test=False, net_state=None, remove_legacy_pad=False, input_dims=None):
        net_state = (caffe_pb2.NetState() if net_state is None else net_state)
        net = caffe2_pb2.NetDef()
        net.name = caffe_net.name
        net_params = caffe2_pb2.TensorProtos()
        if len(caffe_net.layers) > 0:
            raise ValueError('I think something is wrong. This translation script only accepts new style layers that are stored in the layer field.')
        if not input_dims:
            input_dims = _GetInputDims(caffe_net)
        for layer in caffe_net.layer:
            if not _ShouldInclude(net_state, layer):
                log.info('Current net state does not need layer {}'.format(layer.name))
                continue
            log.info('Translate layer {}'.format(layer.name))
            pretrained_layers = [l for l in pretrained_net.layer if l.name == layer.name] + [l for l in pretrained_net.layers if l.name == layer.name]
            if len(pretrained_layers) > 1:
                raise ValueError('huh? more than one pretrained layer of one name?')
            elif len(pretrained_layers) == 1:
                pretrained_blobs = [utils.CaffeBlobToNumpyArray(blob) for blob in pretrained_layers[0].blobs]
            else:
                pretrained_blobs = []
            (operators, params) = cls.TranslateLayer(layer, pretrained_blobs, is_test, net=net, net_params=net_params, input_dims=input_dims)
            net.op.extend(operators)
            net_params.protos.extend(params)
        if remove_legacy_pad:
            assert input_dims, 'Please specify input_dims to remove legacy_pad'
            net = _RemoveLegacyPad(net, net_params, input_dims)
        return (net, net_params)


def TranslateModel(*args, **kwargs):
    return TranslatorRegistry.TranslateModel(*args, **kwargs)

def ConvertTensorProtosToInitNet(net_params, input_name):
    """Takes the net_params returned from TranslateModel, and wrap it as an
    init net that contain GivenTensorFill.

    This is a very simple feature that only works with float tensors, and is
    only intended to be used in an environment where you want a single
    initialization file - for more complex cases, use a db to store the
    parameters.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.caffe_translator.ConvertTensorProtosToInitNet', 'ConvertTensorProtosToInitNet(net_params, input_name)', {'caffe2_pb2': caffe2_pb2, 'core': core, 'utils': utils, 'net_params': net_params, 'input_name': input_name}, 1)

def BaseTranslate(layer, caffe2_type):
    """A simple translate interface that maps the layer input and output."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.caffe_translator.BaseTranslate', 'BaseTranslate(layer, caffe2_type)', {'caffe2_pb2': caffe2_pb2, 'layer': layer, 'caffe2_type': caffe2_type}, 1)

def AddArgument(op, key, value):
    """Makes an argument based on the value type."""
    op.arg.extend([utils.MakeArgument(key, value)])

@TranslatorRegistry.Register('Input')
def TranslateInput(layer, pretrained_blobs, is_test, **kwargs):
    return ([], [])

@TranslatorRegistry.Register('VideoData')
def TranslateVideoData(layer, pretrained_blobs, is_test, **kwargs):
    return ([], [])

@TranslatorRegistry.Register('Data')
def TranslateData(layer, pretrained_blobs, is_test, **kwargs):
    return ([], [])

def _TranslateStridePadKernelHelper(param, caffe_op):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.caffe_translator._TranslateStridePadKernelHelper', '_TranslateStridePadKernelHelper(param, caffe_op)', {'AddArgument': AddArgument, 'param': param, 'caffe_op': caffe_op}, 0)

@TranslatorRegistry.Register('Convolution3D')
def TranslateConvNd(layer, pretrained_blobs, is_test, **kwargs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.caffe_translator.TranslateConvNd', 'TranslateConvNd(layer, pretrained_blobs, is_test, **kwargs)', {'BaseTranslate': BaseTranslate, 'AddArgument': AddArgument, 'utils': utils, 'TranslatorRegistry': TranslatorRegistry, 'layer': layer, 'pretrained_blobs': pretrained_blobs, 'is_test': is_test, 'kwargs': kwargs}, 2)

@TranslatorRegistry.Register('Convolution')
def TranslateConv(layer, pretrained_blobs, is_test, **kwargs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.caffe_translator.TranslateConv', 'TranslateConv(layer, pretrained_blobs, is_test, **kwargs)', {'BaseTranslate': BaseTranslate, '_TranslateStridePadKernelHelper': _TranslateStridePadKernelHelper, 'utils': utils, 'AddArgument': AddArgument, 'TranslatorRegistry': TranslatorRegistry, 'layer': layer, 'pretrained_blobs': pretrained_blobs, 'is_test': is_test, 'kwargs': kwargs}, 2)

@TranslatorRegistry.Register('Deconvolution')
def TranslateDeconv(layer, pretrained_blobs, is_test, **kwargs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.caffe_translator.TranslateDeconv', 'TranslateDeconv(layer, pretrained_blobs, is_test, **kwargs)', {'BaseTranslate': BaseTranslate, '_TranslateStridePadKernelHelper': _TranslateStridePadKernelHelper, 'AddArgument': AddArgument, 'utils': utils, 'TranslatorRegistry': TranslatorRegistry, 'layer': layer, 'pretrained_blobs': pretrained_blobs, 'is_test': is_test, 'kwargs': kwargs}, 2)

@TranslatorRegistry.Register('Crop')
def TranslateCrop(layer, pretrained_blobs, is_test, **kwargs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.caffe_translator.TranslateCrop', 'TranslateCrop(layer, pretrained_blobs, is_test, **kwargs)', {'np': np, '_GetBlobDimMap': _GetBlobDimMap, 'BaseTranslate': BaseTranslate, 'caffe2_pb2': caffe2_pb2, 'AddArgument': AddArgument, 'TranslatorRegistry': TranslatorRegistry, 'layer': layer, 'pretrained_blobs': pretrained_blobs, 'is_test': is_test, 'kwargs': kwargs}, 2)

@TranslatorRegistry.Register('ReLU')
def TranslateRelu(layer, pretrained_blobs, is_test, **kwargs):
    return (BaseTranslate(layer, 'Relu'), [])

@TranslatorRegistry.Register('Pooling')
def TranslatePool(layer, pretrained_blobs, is_test, **kwargs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.caffe_translator.TranslatePool', 'TranslatePool(layer, pretrained_blobs, is_test, **kwargs)', {'caffe_pb2': caffe_pb2, 'BaseTranslate': BaseTranslate, '_TranslateStridePadKernelHelper': _TranslateStridePadKernelHelper, 'AddArgument': AddArgument, 'caffe2_legacy_pb2': caffe2_legacy_pb2, 'TranslatorRegistry': TranslatorRegistry, 'layer': layer, 'pretrained_blobs': pretrained_blobs, 'is_test': is_test, 'kwargs': kwargs}, 2)

@TranslatorRegistry.Register('Pooling3D')
def TranslatePool3D(layer, pretrained_blobs, is_test, **kwargs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.caffe_translator.TranslatePool3D', 'TranslatePool3D(layer, pretrained_blobs, is_test, **kwargs)', {'caffe_pb2': caffe_pb2, 'BaseTranslate': BaseTranslate, 'AddArgument': AddArgument, 'TranslatorRegistry': TranslatorRegistry, 'layer': layer, 'pretrained_blobs': pretrained_blobs, 'is_test': is_test, 'kwargs': kwargs}, 2)

@TranslatorRegistry.Register('LRN')
def TranslateLRN(layer, pretrained_blobs, is_test, **kwargs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.caffe_translator.TranslateLRN', 'TranslateLRN(layer, pretrained_blobs, is_test, **kwargs)', {'BaseTranslate': BaseTranslate, 'caffe_pb2': caffe_pb2, 'AddArgument': AddArgument, 'TranslatorRegistry': TranslatorRegistry, 'layer': layer, 'pretrained_blobs': pretrained_blobs, 'is_test': is_test, 'kwargs': kwargs}, 2)

@TranslatorRegistry.Register('InnerProduct')
def TranslateInnerProduct(layer, pretrained_blobs, is_test, **kwargs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.caffe_translator.TranslateInnerProduct', 'TranslateInnerProduct(layer, pretrained_blobs, is_test, **kwargs)', {'BaseTranslate': BaseTranslate, 'utils': utils, 'TranslatorRegistry': TranslatorRegistry, 'layer': layer, 'pretrained_blobs': pretrained_blobs, 'is_test': is_test, 'kwargs': kwargs}, 2)

@TranslatorRegistry.Register('Dropout')
def TranslateDropout(layer, pretrained_blobs, is_test, **kwargs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.caffe_translator.TranslateDropout', 'TranslateDropout(layer, pretrained_blobs, is_test, **kwargs)', {'BaseTranslate': BaseTranslate, 'AddArgument': AddArgument, 'TranslatorRegistry': TranslatorRegistry, 'layer': layer, 'pretrained_blobs': pretrained_blobs, 'is_test': is_test, 'kwargs': kwargs}, 2)

@TranslatorRegistry.Register('Softmax')
def TranslateSoftmax(layer, pretrained_blobs, is_test, **kwargs):
    caffe_op = BaseTranslate(layer, 'Softmax')
    return (caffe_op, [])

@TranslatorRegistry.Register('SoftmaxWithLoss')
def TranslateSoftmaxWithLoss(layer, pretrained_blobs, is_test, **kwargs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.caffe_translator.TranslateSoftmaxWithLoss', 'TranslateSoftmaxWithLoss(layer, pretrained_blobs, is_test, **kwargs)', {'core': core, 'TranslatorRegistry': TranslatorRegistry, 'layer': layer, 'pretrained_blobs': pretrained_blobs, 'is_test': is_test, 'kwargs': kwargs}, 2)

@TranslatorRegistry.Register('Accuracy')
def TranslateAccuracy(layer, pretrained_blobs, is_test, **kwargs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.caffe_translator.TranslateAccuracy', 'TranslateAccuracy(layer, pretrained_blobs, is_test, **kwargs)', {'BaseTranslate': BaseTranslate, 'AddArgument': AddArgument, 'TranslatorRegistry': TranslatorRegistry, 'layer': layer, 'pretrained_blobs': pretrained_blobs, 'is_test': is_test, 'kwargs': kwargs}, 2)

@TranslatorRegistry.Register('Concat')
def TranslateConcat(layer, pretrained_blobs, is_test, **kwargs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.caffe_translator.TranslateConcat', 'TranslateConcat(layer, pretrained_blobs, is_test, **kwargs)', {'BaseTranslate': BaseTranslate, 'AddArgument': AddArgument, 'TranslatorRegistry': TranslatorRegistry, 'layer': layer, 'pretrained_blobs': pretrained_blobs, 'is_test': is_test, 'kwargs': kwargs}, 2)

@TranslatorRegistry.Register('TanH')
def TranslateTanH(layer, pretrained_blobs, is_test, **kwargs):
    caffe_op = BaseTranslate(layer, 'Tanh')
    return (caffe_op, [])

@TranslatorRegistry.Register('InstanceNorm')
def TranslateInstanceNorm(layer, pretrained_blobs, is_test, **kwargs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.caffe_translator.TranslateInstanceNorm', 'TranslateInstanceNorm(layer, pretrained_blobs, is_test, **kwargs)', {'BaseTranslate': BaseTranslate, 'utils': utils, 'AddArgument': AddArgument, 'TranslatorRegistry': TranslatorRegistry, 'layer': layer, 'pretrained_blobs': pretrained_blobs, 'is_test': is_test, 'kwargs': kwargs}, 2)

@TranslatorRegistry.Register('BatchNorm')
def TranslateBatchNorm(layer, pretrained_blobs, is_test, **kwargs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.caffe_translator.TranslateBatchNorm', 'TranslateBatchNorm(layer, pretrained_blobs, is_test, **kwargs)', {'BaseTranslate': BaseTranslate, 'AddArgument': AddArgument, 'utils': utils, 'np': np, 'TranslatorRegistry': TranslatorRegistry, 'layer': layer, 'pretrained_blobs': pretrained_blobs, 'is_test': is_test, 'kwargs': kwargs}, 2)

@TranslatorRegistry.Register('Eltwise')
def TranslateElementWise(layer, pretrained_blobs, is_test, **kwargs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.caffe_translator.TranslateElementWise', 'TranslateElementWise(layer, pretrained_blobs, is_test, **kwargs)', {'BaseTranslate': BaseTranslate, 'TranslatorRegistry': TranslatorRegistry, 'layer': layer, 'pretrained_blobs': pretrained_blobs, 'is_test': is_test, 'kwargs': kwargs}, 2)

@TranslatorRegistry.Register('Scale')
def TranslateScale(layer, pretrained_blobs, is_test, **kwargs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.caffe_translator.TranslateScale', 'TranslateScale(layer, pretrained_blobs, is_test, **kwargs)', {'BaseTranslate': BaseTranslate, 'AddArgument': AddArgument, 'utils': utils, 'copy': copy, 'TranslatorRegistry': TranslatorRegistry, 'layer': layer, 'pretrained_blobs': pretrained_blobs, 'is_test': is_test, 'kwargs': kwargs}, 2)

@TranslatorRegistry.Register('Reshape')
def TranslateReshape(layer, pretrained_blobs, is_test, **kwargs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.caffe_translator.TranslateReshape', 'TranslateReshape(layer, pretrained_blobs, is_test, **kwargs)', {'BaseTranslate': BaseTranslate, 'AddArgument': AddArgument, 'TranslatorRegistry': TranslatorRegistry, 'layer': layer, 'pretrained_blobs': pretrained_blobs, 'is_test': is_test, 'kwargs': kwargs}, 2)

@TranslatorRegistry.Register('Flatten')
def TranslateFlatten(layer, pretrained_blobs, is_test, **kwargs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.caffe_translator.TranslateFlatten', 'TranslateFlatten(layer, pretrained_blobs, is_test, **kwargs)', {'BaseTranslate': BaseTranslate, 'TranslatorRegistry': TranslatorRegistry, 'layer': layer, 'pretrained_blobs': pretrained_blobs, 'is_test': is_test, 'kwargs': kwargs}, 2)

@TranslatorRegistry.Register('Sigmoid')
def TranslateSigmoid(layer, pretrained_blobs, is_test, **kwargs):
    caffe_op = BaseTranslate(layer, 'Sigmoid')
    return (caffe_op, [])

@TranslatorRegistry.Register('ROIPooling')
def TranslateROIPooling(layer, pretrained_blobs, is_test, **kwargs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.caffe_translator.TranslateROIPooling', 'TranslateROIPooling(layer, pretrained_blobs, is_test, **kwargs)', {'BaseTranslate': BaseTranslate, 'AddArgument': AddArgument, 'TranslatorRegistry': TranslatorRegistry, 'layer': layer, 'pretrained_blobs': pretrained_blobs, 'is_test': is_test, 'kwargs': kwargs}, 2)

@TranslatorRegistry.Register('PReLU')
def TranslatePRelu(layer, pretrained_blobs, is_test, **kwargs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.caffe_translator.TranslatePRelu', 'TranslatePRelu(layer, pretrained_blobs, is_test, **kwargs)', {'BaseTranslate': BaseTranslate, 'utils': utils, 'TranslatorRegistry': TranslatorRegistry, 'layer': layer, 'pretrained_blobs': pretrained_blobs, 'is_test': is_test, 'kwargs': kwargs}, 2)

@TranslatorRegistry.Register('Reduction')
def TranslateReduction(layer, pretrained_blobs, is_test, **kwargs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.caffe_translator.TranslateReduction', 'TranslateReduction(layer, pretrained_blobs, is_test, **kwargs)', {'caffe_pb2': caffe_pb2, 'BaseTranslate': BaseTranslate, 'AddArgument': AddArgument, 'TranslatorRegistry': TranslatorRegistry, 'layer': layer, 'pretrained_blobs': pretrained_blobs, 'is_test': is_test, 'kwargs': kwargs}, 2)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Utilitity to convert pretrained caffe models to Caffe2 models.')
    parser.add_argument('prototext', help='Caffe prototext.')
    parser.add_argument('caffemodel', help='Caffe trained model.')
    parser.add_argument('--init_net', help='Caffe2 initialization net.', default='init_net.pb')
    parser.add_argument('--predict_net', help='Caffe2 prediction net.', default='predict_net.pb')
    parser.add_argument('--remove_legacy_pad', help='Remove legacy pad                         (Only works for nets with one input blob)', action='store_true', default=False)
    parser.add_argument('--input_dims', help='Dimension of input blob', nargs='+', type=int, default=[])
    args = parser.parse_args()
    caffenet = caffe_pb2.NetParameter()
    caffenet_pretrained = caffe_pb2.NetParameter()
    input_proto = args.prototext
    input_caffemodel = args.caffemodel
    output_init_net = args.init_net
    output_predict_net = args.predict_net
    with open(input_proto) as f:
        text_format.Merge(f.read(), caffenet)
    with open(input_caffemodel, 'rb') as f:
        caffenet_pretrained.ParseFromString(f.read())
    (net, pretrained_params) = TranslateModel(caffenet, caffenet_pretrained, is_test=True, remove_legacy_pad=args.remove_legacy_pad, input_dims=args.input_dims)
    external_input = net.op[0].input[0]
    external_output = net.op[-1].output[0]
    net.external_input.extend([external_input])
    net.external_input.extend([param.name for param in pretrained_params.protos])
    net.external_output.extend([external_output])
    init_net = ConvertTensorProtosToInitNet(pretrained_params, external_input)
    with open(output_predict_net, 'wb') as f:
        f.write(net.SerializeToString())
    with open(output_predict_net + 'txt', 'w') as f:
        f.write(str(net))
    with open(output_init_net, 'wb') as f:
        f.write(init_net.SerializeToString())

