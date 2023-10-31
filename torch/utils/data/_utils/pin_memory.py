""""Contains definitions of the methods used by the _BaseDataLoaderIter to put
fetched tensors into pinned memory.

These **needs** to be in global scope since Py2 doesn't support serializing
static methods.
"""

import torch
from torch._six import queue, container_abcs, string_classes
from . import MP_STATUS_CHECK_INTERVAL
from torch._utils import ExceptionWrapper

def _pin_memory_loop(in_queue, out_queue, device_id, done_event):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.utils.data._utils.pin_memory._pin_memory_loop', '_pin_memory_loop(in_queue, out_queue, device_id, done_event)', {'torch': torch, 'MP_STATUS_CHECK_INTERVAL': MP_STATUS_CHECK_INTERVAL, 'queue': queue, 'ExceptionWrapper': ExceptionWrapper, 'pin_memory': pin_memory, 'in_queue': in_queue, 'out_queue': out_queue, 'device_id': device_id, 'done_event': done_event}, 0)

def pin_memory(data):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.data._utils.pin_memory.pin_memory', 'pin_memory(data)', {'torch': torch, 'string_classes': string_classes, 'container_abcs': container_abcs, 'pin_memory': pin_memory, 'data': data}, 1)

