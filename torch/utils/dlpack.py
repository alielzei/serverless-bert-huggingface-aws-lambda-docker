from __future__ import absolute_import, division, print_function, unicode_literals
import torch
from torch._C import _from_dlpack as from_dlpack
from torch._C import _to_dlpack as to_dlpack
torch._C._add_docstr(from_dlpack, 'from_dlpack(dlpack) -> Tensor\n\nDecodes a DLPack to a tensor.\n\nArgs:\n    dlpack: a PyCapsule object with the dltensor\n\nThe tensor will share the memory with the object represented\nin the dlpack.\nNote that each dlpack can only be consumed once.\n')
torch._C._add_docstr(to_dlpack, 'to_dlpack(tensor) -> PyCapsule\n\nReturns a DLPack representing the tensor.\n\nArgs:\n    tensor: a tensor to be exported\n\nThe dlpack shares the tensors memory.\nNote that each dlpack can only be consumed once.\n')

