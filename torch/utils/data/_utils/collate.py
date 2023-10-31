""""Contains definitions of the methods used by the _BaseDataLoaderIter workers to
collate samples fetched from dataset into Tensor(s).

These **needs** to be in global scope since Py2 doesn't support serializing
static methods.
"""

import torch
import re
from torch._six import container_abcs, string_classes, int_classes
np_str_obj_array_pattern = re.compile('[SaUO]')

def default_convert(data):
    """Converts each NumPy array data field into a tensor"""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.data._utils.collate.default_convert', 'default_convert(data)', {'torch': torch, 'np_str_obj_array_pattern': np_str_obj_array_pattern, 'container_abcs': container_abcs, 'default_convert': default_convert, 'string_classes': string_classes, 'data': data}, 1)
default_collate_err_msg_format = 'default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found {}'

def default_collate(batch):
    """Puts each data field into a tensor with outer dimension batch size"""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.utils.data._utils.collate.default_collate', 'default_collate(batch)', {'torch': torch, 'np_str_obj_array_pattern': np_str_obj_array_pattern, 'default_collate_err_msg_format': default_collate_err_msg_format, 'default_collate': default_collate, 'int_classes': int_classes, 'string_classes': string_classes, 'container_abcs': container_abcs, 'batch': batch}, 1)

