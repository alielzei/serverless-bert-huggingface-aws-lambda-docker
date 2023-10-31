from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import argparse
import logging
import time
from caffe2.python import workspace, data_parallel_model
from caffe2.python import cnn
import caffe2.python.models.resnet as resnet
'\nSimple benchmark that creates a data-parallel resnet-50 model\nand measures the time.\n'
logging.basicConfig()
log = logging.getLogger('net_construct_bench')
log.setLevel(logging.DEBUG)

def AddMomentumParameterUpdate(train_model, LR):
    """
    Add the momentum-SGD update.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.experiments.python.net_construct_bench.AddMomentumParameterUpdate', 'AddMomentumParameterUpdate(train_model, LR)', {'train_model': train_model, 'LR': LR}, 0)

def Create(args):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.experiments.python.net_construct_bench.Create', 'Create(args)', {'log': log, 'cnn': cnn, 'resnet': resnet, 'AddMomentumParameterUpdate': AddMomentumParameterUpdate, 'time': time, 'data_parallel_model': data_parallel_model, 'args': args}, 1)

def main():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.experiments.python.net_construct_bench.main', 'main()', {'argparse': argparse, 'Create': Create}, 0)
if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=2'])
    import cProfile
    cProfile.run('main()', sort='cumulative')

