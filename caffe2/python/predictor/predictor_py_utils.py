from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from caffe2.python import core, scope

def create_predict_net(predictor_export_meta):
    """
    Return the input prediction net.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.predictor.predictor_py_utils.create_predict_net', 'create_predict_net(predictor_export_meta)', {'core': core, 'predictor_export_meta': predictor_export_meta}, 1)

def create_predict_init_net(ws, predictor_export_meta):
    """
    Return an initialization net that zero-fill all the input and
    output blobs, using the shapes from the provided workspace. This is
    necessary as there is no shape inference functionality in Caffe2.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.predictor.predictor_py_utils.create_predict_init_net', 'create_predict_init_net(ws, predictor_export_meta)', {'core': core, 'scope': scope, 'AddModelIdArg': AddModelIdArg, 'ws': ws, 'predictor_export_meta': predictor_export_meta}, 1)

def get_comp_name(string, name):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.predictor.predictor_py_utils.get_comp_name', 'get_comp_name(string, name)', {'string': string, 'name': name}, 1)

def _ProtoMapGet(field, key):
    """
    Given the key, get the value of the repeated field.
    Helper function used by protobuf since it doesn't have map construct
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.predictor.predictor_py_utils._ProtoMapGet', '_ProtoMapGet(field, key)', {'field': field, 'key': key}, 1)

def GetPlan(meta_net_def, key):
    return _ProtoMapGet(meta_net_def.plans, key)

def GetPlanOriginal(meta_net_def, key):
    return _ProtoMapGet(meta_net_def.plans, key)

def GetBlobs(meta_net_def, key):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.predictor.predictor_py_utils.GetBlobs', 'GetBlobs(meta_net_def, key)', {'_ProtoMapGet': _ProtoMapGet, 'meta_net_def': meta_net_def, 'key': key}, 1)

def GetBlobsByTypePrefix(meta_net_def, blob_type_prefix):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.predictor.predictor_py_utils.GetBlobsByTypePrefix', 'GetBlobsByTypePrefix(meta_net_def, blob_type_prefix)', {'meta_net_def': meta_net_def, 'blob_type_prefix': blob_type_prefix}, 1)

def GetNet(meta_net_def, key):
    return _ProtoMapGet(meta_net_def.nets, key)

def GetNetOriginal(meta_net_def, key):
    return _ProtoMapGet(meta_net_def.nets, key)

def GetApplicationSpecificInfo(meta_net_def, key):
    return _ProtoMapGet(meta_net_def.applicationSpecificInfo, key)

def AddBlobs(meta_net_def, blob_name, blob_def):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.predictor.predictor_py_utils.AddBlobs', 'AddBlobs(meta_net_def, blob_name, blob_def)', {'_ProtoMapGet': _ProtoMapGet, 'meta_net_def': meta_net_def, 'blob_name': blob_name, 'blob_def': blob_def}, 0)

def ReplaceBlobs(meta_net_def, blob_name, blob_def):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('caffe2.python.predictor.predictor_py_utils.ReplaceBlobs', 'ReplaceBlobs(meta_net_def, blob_name, blob_def)', {'_ProtoMapGet': _ProtoMapGet, 'meta_net_def': meta_net_def, 'blob_name': blob_name, 'blob_def': blob_def}, 0)

def AddPlan(meta_net_def, plan_name, plan_def):
    meta_net_def.plans.add(key=plan_name, value=plan_def)

def AddNet(meta_net_def, net_name, net_def):
    meta_net_def.nets.add(key=net_name, value=net_def)

def SetBlobsOrder(meta_net_def, blobs_order):
    for blob in blobs_order:
        meta_net_def.blobsOrder.append(blob)

def SetPreLoadBlobs(meta_net_def, pre_load_blobs):
    for blob in pre_load_blobs:
        meta_net_def.preLoadBlobs.append(blob)

def GetArgumentByName(net_def, arg_name):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.predictor.predictor_py_utils.GetArgumentByName', 'GetArgumentByName(net_def, arg_name)', {'net_def': net_def, 'arg_name': arg_name}, 1)

def AddModelIdArg(meta_net_def, net_def):
    """Takes the model_id from the predict_net of meta_net_def (if it is
    populated) and adds it to the net_def passed in. This is intended to be
    called on init_nets, as their model_id is not populated by default, but
    should be the same as that of the predict_net
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('caffe2.python.predictor.predictor_py_utils.AddModelIdArg', 'AddModelIdArg(meta_net_def, net_def)', {'GetArgumentByName': GetArgumentByName, 'meta_net_def': meta_net_def, 'net_def': net_def}, 1)

