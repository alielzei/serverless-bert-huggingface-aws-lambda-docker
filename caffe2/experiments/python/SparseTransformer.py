from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import workspace
import scipy.sparse


class NetDefNode:
    
    def __init__(self, name, optype, p=None, op=None):
        self.name = name
        self.optype = optype
        self.ops = {}
        self.prev = {}
        self.insertInput(p)
        self.visited = False
        self.op = op
    
    def insertInput(self, p):
        """
        Insert input of this op
        also maintain the output of previous op
        p: a node or a list of node
        """
        if isinstance(p, list):
            for i in p:
                self.prev[i.name] = i
                i.ops[self.name] = self
        elif isinstance(p, NetDefNode):
            self.prev[p.name] = p
            p.ops[self.name] = self
    
    def deleteInput(self, p):
        if isinstance(p, NetDefNode):
            del self.prev[p.name]
            del p.ops[self.name]


def maskNallocate(weight_name):
    """
    Combine mask and weights
    create wcsr, iw, jw, return their names
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.experiments.python.SparseTransformer.maskNallocate', 'maskNallocate(weight_name)', {'workspace': workspace, 'scipy': scipy, 'weight_name': weight_name}, 3)

def transFCRelu(cur, id2node, name2id, ops, model):
    """
    Add trans before and after this FC_Prune->(Relu)->FC_Prune chain.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.experiments.python.SparseTransformer.transFCRelu', 'transFCRelu(cur, id2node, name2id, ops, model)', {'NetDefNode': NetDefNode, 'maskNallocate': maskNallocate, 'cur': cur, 'id2node': id2node, 'name2id': name2id, 'ops': ops, 'model': model}, 0)

def Prune2Sparse(cur, id2node, name2id, ops, model):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.experiments.python.SparseTransformer.Prune2Sparse', 'Prune2Sparse(cur, id2node, name2id, ops, model)', {'transFCRelu': transFCRelu, 'Prune2Sparse': Prune2Sparse, 'cur': cur, 'id2node': id2node, 'name2id': name2id, 'ops': ops, 'model': model}, 0)

def net2list(net_root):
    """
    Use topological order(BFS) to print the op of a net in a list
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.experiments.python.SparseTransformer.net2list', 'net2list(net_root)', {'net_root': net_root}, 1)

def netbuilder(model):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.experiments.python.SparseTransformer.netbuilder', 'netbuilder(model)', {'NetDefNode': NetDefNode, 'model': model}, 3)

