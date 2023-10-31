import inspect
from functools import partial
from joblib.externals.cloudpickle import dumps, loads
WRAP_CACHE = {}


class CloudpickledObjectWrapper:
    
    def __init__(self, obj, keep_wrapper=False):
        self._obj = obj
        self._keep_wrapper = keep_wrapper
    
    def __reduce__(self):
        _pickled_object = dumps(self._obj)
        if not self._keep_wrapper:
            return (loads, (_pickled_object, ))
        return (_reconstruct_wrapper, (_pickled_object, self._keep_wrapper))
    
    def __getattr__(self, attr):
        if attr not in ['_obj', '_keep_wrapper']:
            return getattr(self._obj, attr)
        return getattr(self, attr)



class CallableObjectWrapper(CloudpickledObjectWrapper):
    
    def __call__(self, *args, **kwargs):
        return self._obj(*args, **kwargs)


def _wrap_non_picklable_objects(obj, keep_wrapper):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.externals.loky.cloudpickle_wrapper._wrap_non_picklable_objects', '_wrap_non_picklable_objects(obj, keep_wrapper)', {'CallableObjectWrapper': CallableObjectWrapper, 'CloudpickledObjectWrapper': CloudpickledObjectWrapper, 'obj': obj, 'keep_wrapper': keep_wrapper}, 1)

def _reconstruct_wrapper(_pickled_object, keep_wrapper):
    obj = loads(_pickled_object)
    return _wrap_non_picklable_objects(obj, keep_wrapper)

def _wrap_objects_when_needed(obj):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.externals.loky.cloudpickle_wrapper._wrap_objects_when_needed', '_wrap_objects_when_needed(obj)', {'partial': partial, '_wrap_objects_when_needed': _wrap_objects_when_needed, 'inspect': inspect, 'WRAP_CACHE': WRAP_CACHE, '_wrap_non_picklable_objects': _wrap_non_picklable_objects, 'obj': obj}, 1)

def wrap_non_picklable_objects(obj, keep_wrapper=True):
    """Wrapper for non-picklable object to use cloudpickle to serialize them.

    Note that this wrapper tends to slow down the serialization process as it
    is done with cloudpickle which is typically slower compared to pickle. The
    proper way to solve serialization issues is to avoid defining functions and
    objects in the main scripts and to implement __reduce__ functions for
    complex classes.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('joblib.externals.loky.cloudpickle_wrapper.wrap_non_picklable_objects', 'wrap_non_picklable_objects(obj, keep_wrapper=True)', {'inspect': inspect, 'CloudpickledObjectWrapper': CloudpickledObjectWrapper, '_wrap_non_picklable_objects': _wrap_non_picklable_objects, 'obj': obj, 'keep_wrapper': keep_wrapper}, 1)

