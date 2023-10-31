from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import core, workspace
from caffe2.python import dyndep
dyndep.InitOpsLibrary('@/caffe2/caffe2/distributed:file_store_handler_ops')

def gen_rendezvous_ctx(self, model, dataset, is_train):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.contrib.playground.resnetdemo.rendezvous_filestore.gen_rendezvous_ctx', 'gen_rendezvous_ctx(self, model, dataset, is_train)', {'workspace': workspace, 'core': core, 'self': self, 'model': model, 'dataset': dataset, 'is_train': is_train}, 1)

