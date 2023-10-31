from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from caffe2.python import brew, cnn, core, workspace, data_parallel_model, timeout_guard, model_helper, optimizer
from caffe2.python.test_util import TestCase
import caffe2.python.models.resnet as resnet
from caffe2.python.modeling.initializers import Initializer
from caffe2.python import convnet_benchmarks as cb
from caffe2.python import hypothesis_test_util as hu
import time
import numpy as np
from hypothesis import settings
CI_MAX_EXAMPLES = 2
CI_TIMEOUT = 600

def executor_test_settings(func):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.test.executor_test_util.executor_test_settings', 'executor_test_settings(func)', {'hu': hu, 'settings': settings, 'CI_MAX_EXAMPLES': CI_MAX_EXAMPLES, 'CI_TIMEOUT': CI_TIMEOUT, 'func': func}, 1)

def gen_test_resnet50(_order, _cudnn_ws):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.test.executor_test_util.gen_test_resnet50', 'gen_test_resnet50(_order, _cudnn_ws)', {'cnn': cnn, 'resnet': resnet, '_order': _order, '_cudnn_ws': _cudnn_ws}, 2)

def conv_model_generators():
    return {'AlexNet': cb.AlexNet, 'OverFeat': cb.OverFeat, 'VGGA': cb.VGGA, 'Inception': cb.Inception, 'MLP': cb.MLP, 'Resnet50': gen_test_resnet50}

def executor_test_model_names():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.test.executor_test_util.executor_test_model_names', 'executor_test_model_names()', {'hu': hu, 'conv_model_generators': conv_model_generators}, 1)

def build_conv_model(model_name, batch_size):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.test.executor_test_util.build_conv_model', 'build_conv_model(model_name, batch_size)', {'conv_model_generators': conv_model_generators, 'brew': brew, 'model_name': model_name, 'batch_size': batch_size}, 1)

def build_resnet50_dataparallel_model(num_gpus, batch_size, epoch_size, cudnn_workspace_limit_mb=64, num_channels=3, num_labels=1000, weight_decay=0.0001, base_learning_rate=0.1, image_size=227, use_cpu=False):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.test.executor_test_util.build_resnet50_dataparallel_model', 'build_resnet50_dataparallel_model(num_gpus, batch_size, epoch_size, cudnn_workspace_limit_mb=64, num_channels=3, num_labels=1000, weight_decay=0.0001, base_learning_rate=0.1, image_size=227, use_cpu=False)', {'model_helper': model_helper, 'brew': brew, 'Initializer': Initializer, 'resnet': resnet, 'optimizer': optimizer, 'core': core, 'data_parallel_model': data_parallel_model, 'num_gpus': num_gpus, 'batch_size': batch_size, 'epoch_size': epoch_size, 'cudnn_workspace_limit_mb': cudnn_workspace_limit_mb, 'num_channels': num_channels, 'num_labels': num_labels, 'weight_decay': weight_decay, 'base_learning_rate': base_learning_rate, 'image_size': image_size, 'use_cpu': use_cpu}, 1)

def run_resnet50_epoch(train_model, batch_size, epoch_size, skip_first_n_iter=0):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.test.executor_test_util.run_resnet50_epoch', 'run_resnet50_epoch(train_model, batch_size, epoch_size, skip_first_n_iter=0)', {'timeout_guard': timeout_guard, 'time': time, 'workspace': workspace, 'train_model': train_model, 'batch_size': batch_size, 'epoch_size': epoch_size, 'skip_first_n_iter': skip_first_n_iter}, 4)


class ExecutorTestBase(TestCase):
    
    def compare_executors(self, model, ref_executor, test_executor, model_run_func):
        model.Proto().type = ref_executor
        model.param_init_net.set_rand_seed(seed=13303778)
        model.net.set_rand_seed(seed=13303778)
        workspace.ResetWorkspace()
        workspace.RunNetOnce(model.param_init_net)
        workspace.CreateNet(model.net)
        model_run_func()
        ref_ws = {str(k): workspace.FetchBlob(k) for k in workspace.Blobs()}
        ref_ws = {k: v for (k, v) in ref_ws.items() if type(v) is np.ndarray}
        workspace.ResetWorkspace()
        workspace.RunNetOnce(model.param_init_net)
        model.Proto().type = test_executor
        workspace.CreateNet(model.net, overwrite=True)
        model_run_func()
        test_ws = {str(k): workspace.FetchBlob(k) for k in workspace.Blobs()}
        test_ws = {k: v for (k, v) in test_ws.items() if type(v) is np.ndarray}
        for (blob_name, ref_val) in ref_ws.items():
            self.assertTrue(blob_name in test_ws, 'Blob {} not found in {} run'.format(blob_name, test_executor))
            val = test_ws[blob_name]
            np.testing.assert_array_equal(val, ref_val, 'Blob {} differs in {} run'.format(blob_name, test_executor))


