from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import argparse
import json
import os
import caffe2.contrib.playground.AnyExp as AnyExp
import caffe2.contrib.playground.checkpoint as checkpoint
import logging
logging.basicConfig()
log = logging.getLogger('AnyExpOnTerm')
log.setLevel(logging.DEBUG)

def runShardedTrainLoop(opts, myTrainFun):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.contrib.playground.AnyExpOnTerm.runShardedTrainLoop', 'runShardedTrainLoop(opts, myTrainFun)', {'os': os, 'checkpoint': checkpoint, 'log': log, 'opts': opts, 'myTrainFun': myTrainFun}, 1)

def trainFun():
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.contrib.playground.AnyExpOnTerm.trainFun', 'trainFun()', {'AnyExp': AnyExp}, 1)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Any Experiment training.')
    parser.add_argument('--parameters-json', type=json.loads, help='model options in json format', dest='params')
    args = parser.parse_args()
    opts = args.params['opts']
    opts = AnyExp.initOpts(opts)
    log.info('opts is: {}'.format(str(opts)))
    AnyExp.initDefaultModuleMap()
    opts['input']['datasets'] = AnyExp.aquireDatasets(opts)
    ret = runShardedTrainLoop(opts, trainFun())
    log.info('ret is: {}'.format(str(ret)))

