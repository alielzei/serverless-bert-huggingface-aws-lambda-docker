"""Adds docstrings to Storage functions"""

import torch._C
from torch._C import _add_docstr as add_docstr
storage_classes = ['DoubleStorageBase', 'FloatStorageBase', 'LongStorageBase', 'IntStorageBase', 'ShortStorageBase', 'CharStorageBase', 'ByteStorageBase', 'BoolStorageBase', 'BFloat16StorageBase']

def add_docstr_all(method, docstr):
    for cls_name in storage_classes:
        cls = getattr(torch._C, cls_name)
        try:
            add_docstr(getattr(cls, method), docstr)
        except AttributeError:
            pass
add_docstr_all('from_file', '\nfrom_file(filename, shared=False, size=0) -> Storage\n\nIf `shared` is `True`, then memory is shared between all processes.\nAll changes are written to the file. If `shared` is `False`, then the changes on\nthe storage do not affect the file.\n\n`size` is the number of elements in the storage. If `shared` is `False`,\nthen the file must contain at least `size * sizeof(Type)` bytes\n(`Type` is the type of storage). If `shared` is `True` the file will be\ncreated if needed.\n\nArgs:\n    filename (str): file name to map\n    shared (bool): whether to share memory\n    size (int): number of elements in the storage\n')

