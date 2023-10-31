from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import core, schema
from caffe2.python.layers.layers import InstantiationContext
from caffe2.python.layers.tags import Tags

def _filter_layers(layers, include_tags):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.layer_model_instantiator._filter_layers', '_filter_layers(layers, include_tags)', {'layers': layers, 'include_tags': include_tags}, 1)

def shrink_output_schema(net, out_schema):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.layer_model_instantiator.shrink_output_schema', 'shrink_output_schema(net, out_schema)', {'schema': schema, 'net': net, 'out_schema': out_schema}, 1)

def generate_predict_net(model, include_tags=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.layer_model_instantiator.generate_predict_net', 'generate_predict_net(model, include_tags=None)', {'core': core, '_filter_layers': _filter_layers, 'Tags': Tags, 'InstantiationContext': InstantiationContext, 'shrink_output_schema': shrink_output_schema, 'model': model, 'include_tags': include_tags}, 1)

def generate_eval_net(model, include_tags=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.layer_model_instantiator.generate_eval_net', 'generate_eval_net(model, include_tags=None)', {'core': core, '_filter_layers': _filter_layers, 'Tags': Tags, 'InstantiationContext': InstantiationContext, 'shrink_output_schema': shrink_output_schema, 'model': model, 'include_tags': include_tags}, 1)

def _generate_training_net_only(model, include_tags=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.layer_model_instantiator._generate_training_net_only', '_generate_training_net_only(model, include_tags=None)', {'core': core, '_filter_layers': _filter_layers, 'Tags': Tags, 'shrink_output_schema': shrink_output_schema, 'model': model, 'include_tags': include_tags}, 2)

def generate_training_nets_forward_only(model, include_tags=None):
    (train_init_net, train_net) = _generate_training_net_only(model, include_tags)
    return (train_init_net, train_net)

def generate_training_nets(model, include_tags=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.layer_model_instantiator.generate_training_nets', 'generate_training_nets(model, include_tags=None)', {'_generate_training_net_only': _generate_training_net_only, 'model': model, 'include_tags': include_tags}, 2)

