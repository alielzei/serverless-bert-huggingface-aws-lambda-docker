from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import inspect
import logging
logging.basicConfig()
log = logging.getLogger('ModuleRegister')
log.setLevel(logging.DEBUG)
MODULE_MAPS = []

def registerModuleMap(module_map):
    MODULE_MAPS.append(module_map)
    log.info('ModuleRegister get modules from  ModuleMap content: {}'.format(inspect.getsource(module_map)))

def constructTrainerClass(myTrainerClass, opts):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.contrib.playground.ModuleRegister.constructTrainerClass', 'constructTrainerClass(myTrainerClass, opts)', {'log': log, 'getModule': getModule, 'myTrainerClass': myTrainerClass, 'opts': opts}, 1)

def overrideAdditionalMethods(myTrainerClass, opts):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.contrib.playground.ModuleRegister.overrideAdditionalMethods', 'overrideAdditionalMethods(myTrainerClass, opts)', {'log': log, 'inspect': inspect, 'getModule': getModule, 'myTrainerClass': myTrainerClass, 'opts': opts}, 1)

def getModule(moduleName):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.contrib.playground.ModuleRegister.getModule', 'getModule(moduleName)', {'log': log, 'MODULE_MAPS': MODULE_MAPS, 'inspect': inspect, 'moduleName': moduleName}, 1)

def getClassFromModule(moduleName, className):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.contrib.playground.ModuleRegister.getClassFromModule', 'getClassFromModule(moduleName, className)', {'MODULE_MAPS': MODULE_MAPS, 'inspect': inspect, 'log': log, 'moduleName': moduleName, 'className': className}, 1)

