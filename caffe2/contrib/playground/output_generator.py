from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import timeout_guard

def fun_conclude_operator(self):
    timeout_guard.EuthanizeIfNecessary(600.0)

def assembleAllOutputs(self):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.contrib.playground.output_generator.assembleAllOutputs', 'assembleAllOutputs(self)', {'self': self}, 1)

