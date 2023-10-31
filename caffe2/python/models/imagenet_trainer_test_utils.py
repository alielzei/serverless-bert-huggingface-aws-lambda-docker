from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
import time
from caffe2.python import workspace, cnn, memonger, core

def has_blob(proto, needle):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.models.imagenet_trainer_test_utils.has_blob', 'has_blob(proto, needle)', {'proto': proto, 'needle': needle}, 1)

def count_blobs(proto):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.models.imagenet_trainer_test_utils.count_blobs', 'count_blobs(proto)', {'proto': proto}, 1)

def count_shared_blobs(proto):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.models.imagenet_trainer_test_utils.count_shared_blobs', 'count_shared_blobs(proto)', {'proto': proto}, 1)

def test_shared_grads(with_shapes, create_model, conv_blob, last_out_blob, data_blob='gpu_0/data', label_blob='gpu_0/label', num_labels=1000):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.models.imagenet_trainer_test_utils.test_shared_grads', "test_shared_grads(with_shapes, create_model, conv_blob, last_out_blob, data_blob='gpu_0/data', label_blob='gpu_0/label', num_labels=1000)", {'cnn': cnn, 'core': core, 'workspace': workspace, 'count_blobs': count_blobs, 'memonger': memonger, 'np': np, 'with_shapes': with_shapes, 'create_model': create_model, 'conv_blob': conv_blob, 'last_out_blob': last_out_blob, 'data_blob': data_blob, 'label_blob': label_blob, 'num_labels': num_labels}, 1)

def test_forward_only(create_model, last_out_blob, data_blob='gpu_0/data', num_labels=1000):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.models.imagenet_trainer_test_utils.test_forward_only', "test_forward_only(create_model, last_out_blob, data_blob='gpu_0/data', num_labels=1000)", {'cnn': cnn, 'core': core, 'count_blobs': count_blobs, 'memonger': memonger, 'count_shared_blobs': count_shared_blobs, 'workspace': workspace, 'np': np, 'create_model': create_model, 'last_out_blob': last_out_blob, 'data_blob': data_blob, 'num_labels': num_labels}, 1)

def test_forward_only_fast_simplenet(create_model, last_out_blob, data_blob='gpu_0/data', num_labels=1000):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.models.imagenet_trainer_test_utils.test_forward_only_fast_simplenet', "test_forward_only_fast_simplenet(create_model, last_out_blob, data_blob='gpu_0/data', num_labels=1000)", {'cnn': cnn, 'core': core, 'count_blobs': count_blobs, 'time': time, 'memonger': memonger, 'count_shared_blobs': count_shared_blobs, 'workspace': workspace, 'np': np, 'create_model': create_model, 'last_out_blob': last_out_blob, 'data_blob': data_blob, 'num_labels': num_labels}, 1)

