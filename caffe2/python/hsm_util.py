from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.proto import hsm_pb2
'\n    Hierarchical softmax utility methods that can be used to:\n    1) create TreeProto structure given list of word_ids or NodeProtos\n    2) create HierarchyProto structure using the user-inputted TreeProto\n'

def create_node_with_words(words, name='node'):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.hsm_util.create_node_with_words', "create_node_with_words(words, name='node')", {'hsm_pb2': hsm_pb2, 'words': words, 'name': name}, 1)

def create_node_with_nodes(nodes, name='node'):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.hsm_util.create_node_with_nodes', "create_node_with_nodes(nodes, name='node')", {'hsm_pb2': hsm_pb2, 'nodes': nodes, 'name': name}, 1)

def create_hierarchy(tree_proto):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.hsm_util.create_hierarchy', 'create_hierarchy(tree_proto)', {'hsm_pb2': hsm_pb2, 'tree_proto': tree_proto}, 1)

