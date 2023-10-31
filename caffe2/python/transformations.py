from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import caffe2.python._import_c_extension as C


class Transformer(object):
    
    def __init__(self):
        pass
    
    @classmethod
    def runTransform(cls, transform_name, net):
        pb = net.Proto().SerializeToString()
        if C.transform_exists(transform_name):
            output = C.run_transform(transform_name, pb)
        elif C.workspace_transform_exists(transform_name):
            output = C.run_workspace_transform(transform_name, pb)
        else:
            raise AttributeError('Transformation {} not found.'.format(transform_name))
        net.Proto().ParseFromString(output)
    
    def __getattr__(self, transform_name):
        return lambda net: self.runTransform(transform_name, net)


def fuseNNPACKConvRelu(net):
    net.Proto().ParseFromString(C.transform_fuseNNPACKConvRelu(net.Proto().SerializeToString()))

def optimizeForMKLDNN(net, training_mode=False):
    net.Proto().ParseFromString(C.transform_optimizeForMKLDNN(net.Proto().SerializeToString(), training_mode))

def fuseConvBN(net):
    net.Proto().ParseFromString(C.transform_fuseConvBN(net.Proto().SerializeToString()))

