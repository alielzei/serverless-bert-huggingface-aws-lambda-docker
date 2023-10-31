from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from importlib import import_module
import pkgutil
import sys
from . import layers

def import_recursive(package):
    """
    Takes a package and imports all modules underneath it
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.layers.__init__.import_recursive', 'import_recursive(package)', {'pkgutil': pkgutil, 'import_module': import_module, 'import_recursive': import_recursive, 'package': package}, 0)

def find_subclasses_recursively(base_cls, sub_cls):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.layers.__init__.find_subclasses_recursively', 'find_subclasses_recursively(base_cls, sub_cls)', {'find_subclasses_recursively': find_subclasses_recursively, 'base_cls': base_cls, 'sub_cls': sub_cls}, 0)
import_recursive(sys.modules[__name__])
model_layer_subcls = set()
find_subclasses_recursively(layers.ModelLayer, model_layer_subcls)
for cls in list(model_layer_subcls):
    layers.register_layer(cls.__name__, cls)

