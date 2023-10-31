from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import six
from caffe2.python import context


@context.define_context(allow_default=True)
class TagContext(object):
    """
    Scope driven way to provide tags to the layers.
    """
    
    def __init__(self, tags=None):
        self.tags = (tags or [])
    
    def add_tags(self, tags):
        self.tags.extend(tags)
    
    def remove_tags(self, tags):
        assert self.tags[-len(tags):] == tags
        self.tags = self.tags[:-len(tags)]



class Tags(object):
    EXCLUDE_FROM_TRAIN = 'exclude_from_train'
    EXCLUDE_FROM_EVAL = 'exclude_from_eval'
    EXCLUDE_FROM_PREDICTION = 'exclude_from_prediction'
    EXCLUDE_FROM_ACCUMULATE_PRED = 'exclude_from_accumulate_pred'
    PREPROCESSING = 'preprocessing'
    HANDLE_AS_SPARSE_LAYER = 'handle_as_sparse_layer'
    PREFER_GPU = 'prefer_gpu'
    CPU_ONLY = 'cpu_only'
    LOCAL = 'local'
    '\n    Indicates a layer contains a sparse shardable parameter.  The parameter\n    should be sharded nd operators on those parameters should be done on\n    distributed parameter servers.\n    '
    SPARSE_SHARDED = 'sparse_sharded'
    '\n    Indicates a layer contains a sparse parameters among others, and that the\n    parameters should not be sharded (i.e. should be placed together on a node).\n    '
    SPARSE_DONT_SHARD = 'sparse_dont_shard'
    '\n    Used to manually indicate a component for an operator.  Parameters for\n    all operators with the same component should be colocated on the same\n    parameter server.\n    '
    COMPONENT = 'component:'
    '\n    Valid tag prefixes for distributed training framework.\n    '
    "\n    Used to pass on info to the 'extra_info' field in the net\n    Proto. Typically to provide info for distributed training.\n    "
    EXTRA_INFO = 'extra_info:'
    '\n    An empty tag, used to make conditional statement on with(Tags) block more concise\n    '
    EMPTY_TAG = 'empty_tag'
    DT_TAGS = (SPARSE_SHARDED, SPARSE_DONT_SHARD, COMPONENT)
    PREDICTION_SCHEMA = 'prediction_schema'
    FEATURE_TRANSFORM = 'feature_transform'
    FEATURE_TRANSFORM_SCHEMA = 'feature_transform_schema'
    
    def __init__(self, tags):
        if not isinstance(tags, list):
            tags = [tags]
        self.tags = tags
    
    def __enter__(self):
        TagContext.current().add_tags(self.tags)
        return self
    
    def __exit__(self, type, value, traceback):
        TagContext.current().remove_tags(self.tags)
    
    def __call__(self, func):
        
        @six.wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper

Tags.TRAIN_ONLY = [Tags.EXCLUDE_FROM_PREDICTION, Tags.EXCLUDE_FROM_EVAL, Tags.EXCLUDE_FROM_ACCUMULATE_PRED]
Tags.EVAL_ONLY = [Tags.EXCLUDE_FROM_PREDICTION, Tags.EXCLUDE_FROM_TRAIN, Tags.EXCLUDE_FROM_ACCUMULATE_PRED]
Tags.PREDICTION_ONLY = [Tags.EXCLUDE_FROM_TRAIN, Tags.EXCLUDE_FROM_EVAL, Tags.EXCLUDE_FROM_ACCUMULATE_PRED]

