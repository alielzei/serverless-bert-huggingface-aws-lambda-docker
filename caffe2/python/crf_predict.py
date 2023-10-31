from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from caffe2.python.crf import CRFWithLoss

def crf_update_predictions(model, crf_with_loss, classes):
    return apply_crf(model.param_init_net, model.net, crf_with_loss.transitions, classes, crf_with_loss.num_classes)

def apply_crf(init_net, net, transitions, predictions, num_classes):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.crf_predict.apply_crf', 'apply_crf(init_net, net, transitions, predictions, num_classes)', {'CRFWithLoss': CRFWithLoss, 'np': np, 'init_net': init_net, 'net': net, 'transitions': transitions, 'predictions': predictions, 'num_classes': num_classes}, 1)

