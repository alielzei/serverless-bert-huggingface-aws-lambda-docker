from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.proto import caffe2_pb2
from caffe2.proto import metanet_pb2
from caffe2.python import workspace, core, scope
from caffe2.python.predictor_constants import predictor_constants
import caffe2.python.predictor.serde as serde
import caffe2.python.predictor.predictor_py_utils as utils
from builtins import bytes
import collections

def get_predictor_exporter_helper(submodelNetName):
    """ constracting stub for the PredictorExportMeta
        Only used to construct names to subfields,
        such as calling to predict_net_name
        Args:
            submodelNetName - name of the model
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.predictor.predictor_exporter.get_predictor_exporter_helper', 'get_predictor_exporter_helper(submodelNetName)', {'core': core, 'PredictorExportMeta': PredictorExportMeta, 'submodelNetName': submodelNetName}, 1)


class PredictorExportMeta(collections.namedtuple('PredictorExportMeta', 'predict_net, parameters, inputs, outputs, shapes, name,         extra_init_net, global_init_net, net_type, num_workers, trainer_prefix')):
    """
    Metadata to be used for serializaing a net.

    parameters, inputs, outputs could be either BlobReference or blob's names

    predict_net can be either core.Net, NetDef, PlanDef or object

    Override the named tuple to provide optional name parameter.
    name will be used to identify multiple prediction nets.

    net_type is the type field in caffe2 NetDef - can be 'simple', 'dag', etc.

    num_workers specifies for net type 'dag' how many threads should run ops

    trainer_prefix specifies the type of trainer.

    extra_init_net gets appended to pred_init_net, useful for thread local init

    global_init_net gets appended to global_init_net, useful for global init
    on a shared across threads parameter workspace
    (in a case of multi-threaded inference)

    """
    
    def __new__(cls, predict_net, parameters, inputs, outputs, shapes=None, name='', extra_init_net=None, global_init_net=None, net_type=None, num_workers=None, trainer_prefix=None):
        inputs = [str(i) for i in inputs]
        outputs = [str(o) for o in outputs]
        assert len(set(inputs)) == len(inputs), 'All inputs to the predictor should be unique'
        parameters = [str(p) for p in parameters]
        assert set(parameters).isdisjoint(inputs), 'Parameters and inputs are required to be disjoint. Intersection: {}'.format(set(parameters).intersection(inputs))
        assert set(parameters).isdisjoint(outputs), 'Parameters and outputs are required to be disjoint. Intersection: {}'.format(set(parameters).intersection(outputs))
        shapes = (shapes or {})
        if isinstance(predict_net, (core.Net, core.Plan)):
            predict_net = predict_net.Proto()
        assert isinstance(predict_net, (caffe2_pb2.NetDef, caffe2_pb2.PlanDef))
        return super(PredictorExportMeta, cls).__new__(cls, predict_net, parameters, inputs, outputs, shapes, name, extra_init_net, global_init_net, net_type, num_workers, trainer_prefix)
    
    def inputs_name(self):
        return utils.get_comp_name(predictor_constants.INPUTS_BLOB_TYPE, self.name)
    
    def outputs_name(self):
        return utils.get_comp_name(predictor_constants.OUTPUTS_BLOB_TYPE, self.name)
    
    def parameters_name(self):
        return utils.get_comp_name(predictor_constants.PARAMETERS_BLOB_TYPE, self.name)
    
    def global_init_name(self):
        return utils.get_comp_name(predictor_constants.GLOBAL_INIT_NET_TYPE, self.name)
    
    def predict_init_name(self):
        return utils.get_comp_name(predictor_constants.PREDICT_INIT_NET_TYPE, self.name)
    
    def predict_net_name(self):
        return utils.get_comp_name(predictor_constants.PREDICT_NET_TYPE, self.name)
    
    def train_init_plan_name(self):
        plan_name = utils.get_comp_name(predictor_constants.TRAIN_INIT_PLAN_TYPE, self.name)
        return (self.trainer_prefix + '_' + plan_name if self.trainer_prefix else plan_name)
    
    def train_plan_name(self):
        plan_name = utils.get_comp_name(predictor_constants.TRAIN_PLAN_TYPE, self.name)
        return (self.trainer_prefix + '_' + plan_name if self.trainer_prefix else plan_name)


def prepare_prediction_net(filename, db_type, device_option=None):
    """
    Helper function which loads all required blobs from the db
    and returns prediction net ready to be used
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.predictor.predictor_exporter.prepare_prediction_net', 'prepare_prediction_net(filename, db_type, device_option=None)', {'load_from_db': load_from_db, 'utils': utils, 'predictor_constants': predictor_constants, 'workspace': workspace, 'core': core, 'filename': filename, 'db_type': db_type, 'device_option': device_option}, 1)

def _global_init_net(predictor_export_meta, db_type):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.predictor.predictor_exporter._global_init_net', '_global_init_net(predictor_export_meta, db_type)', {'core': core, 'predictor_constants': predictor_constants, 'utils': utils, 'predictor_export_meta': predictor_export_meta, 'db_type': db_type}, 1)

def get_meta_net_def(predictor_export_meta, ws=None, db_type=None):
    """
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.predictor.predictor_exporter.get_meta_net_def', 'get_meta_net_def(predictor_export_meta, ws=None, db_type=None)', {'workspace': workspace, 'metanet_pb2': metanet_pb2, 'utils': utils, '_global_init_net': _global_init_net, 'predictor_export_meta': predictor_export_meta, 'ws': ws, 'db_type': db_type}, 1)

def set_model_info(meta_net_def, project_str, model_class_str, version):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.predictor.predictor_exporter.set_model_info', 'set_model_info(meta_net_def, project_str, model_class_str, version)', {'metanet_pb2': metanet_pb2, 'meta_net_def': meta_net_def, 'project_str': project_str, 'model_class_str': model_class_str, 'version': version}, 0)

def save_to_db(db_type, db_destination, predictor_export_meta, use_ideep=False, *args, **kwargs):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.predictor.predictor_exporter.save_to_db', 'save_to_db(db_type, db_destination, predictor_export_meta, use_ideep=False, *args, **kwargs)', {'get_meta_net_def': get_meta_net_def, 'caffe2_pb2': caffe2_pb2, 'core': core, 'workspace': workspace, 'predictor_constants': predictor_constants, 'serde': serde, 'db_type': db_type, 'db_destination': db_destination, 'predictor_export_meta': predictor_export_meta, 'use_ideep': use_ideep, 'args': args, 'kwargs': kwargs}, 0)

def load_from_db(filename, db_type, device_option=None, *args, **kwargs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.predictor.predictor_exporter.load_from_db', 'load_from_db(filename, db_type, device_option=None, *args, **kwargs)', {'core': core, 'predictor_constants': predictor_constants, 'workspace': workspace, 'serde': serde, 'metanet_pb2': metanet_pb2, 'scope': scope, 'filename': filename, 'db_type': db_type, 'device_option': device_option, 'args': args, 'kwargs': kwargs}, 1)

