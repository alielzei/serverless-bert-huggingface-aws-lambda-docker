"""
Torch utilities for the Trainer class.
"""

import math
import warnings
from contextlib import contextmanager
from typing import List, Optional, Union
import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, Sampler
from .file_utils import is_torch_tpu_available
from .utils import logging
if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
PT_LR_SCHEDULER_WARNING = 'Please also save or load the state of the optimzer when saving or loading the scheduler.'
logger = logging.get_logger(__name__)

def nested_concat(tensors, new_tensors, dim=0):
    """Concat the `new_tensors` to `tensors` on `dim`. Works for tensors or nested list/tuples of tensors."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.trainer_pt_utils.nested_concat', 'nested_concat(tensors, new_tensors, dim=0)', {'nested_concat': nested_concat, 'torch': torch, 'np': np, 'tensors': tensors, 'new_tensors': new_tensors, 'dim': dim}, 1)

def nested_numpify(tensors):
    """Numpify `tensors` (even if it's a nested list/tuple of tensors)."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.trainer_pt_utils.nested_numpify', 'nested_numpify(tensors)', {'nested_numpify': nested_numpify, 'tensors': tensors}, 1)

def nested_detach(tensors):
    """Detach `tensors` (even if it's a nested list/tuple of tensors)."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.trainer_pt_utils.nested_detach', 'nested_detach(tensors)', {'nested_detach': nested_detach, 'tensors': tensors}, 1)

def nested_xla_mesh_reduce(tensors, name):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.trainer_pt_utils.nested_xla_mesh_reduce', 'nested_xla_mesh_reduce(tensors, name)', {'is_torch_tpu_available': is_torch_tpu_available, 'nested_xla_mesh_reduce': nested_xla_mesh_reduce, 'torch': torch, 'tensors': tensors, 'name': name}, 1)

def distributed_concat(tensor: 'torch.Tensor', num_total_examples: Optional[int] = None) -> torch.Tensor:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.trainer_pt_utils.distributed_concat', 'distributed_concat(tensor, num_total_examples=None)', {'distributed_concat': distributed_concat, 'torch': torch, 'tensor': tensor, 'num_total_examples': num_total_examples, 'Optional': Optional, 'int': int, 'torch': torch}, 1)

def distributed_broadcast_scalars(scalars: List[Union[(int, float)]], num_total_examples: Optional[int] = None) -> torch.Tensor:
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.trainer_pt_utils.distributed_broadcast_scalars', 'distributed_broadcast_scalars(scalars, num_total_examples=None)', {'torch': torch, 'scalars': scalars, 'num_total_examples': num_total_examples, 'List': List, 'Optional': Optional, 'int': int, 'torch': torch}, 1)

def reissue_pt_warnings(caught_warnings):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('transformers.trainer_pt_utils.reissue_pt_warnings', 'reissue_pt_warnings(caught_warnings)', {'PT_LR_SCHEDULER_WARNING': PT_LR_SCHEDULER_WARNING, 'warnings': warnings, 'caught_warnings': caught_warnings}, 0)

@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.

    Args:
        local_rank (:obj:`int`): The rank of the local process.
    """
    import custom_funtemplate
    custom_funtemplate.rewrite_template('transformers.trainer_pt_utils.torch_distributed_zero_first', 'torch_distributed_zero_first(local_rank)', {'torch': torch, 'contextmanager': contextmanager, 'local_rank': local_rank}, 0)


class SequentialDistributedSampler(Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.

    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """
    
    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError('Requires distributed package to be available')
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError('Requires distributed package to be available')
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
    
    def __iter__(self):
        indices = list(range(len(self.dataset)))
        indices += indices[:self.total_size - len(indices)]
        assert len(indices) == self.total_size, f'Indices length {len(indices)} and total size {self.total_size} mismatched'
        indices = indices[self.rank * self.num_samples:(self.rank + 1) * self.num_samples]
        assert len(indices) == self.num_samples, f'Indices length {len(indices)} and sample number {self.num_samples} mismatched'
        return iter(indices)
    
    def __len__(self):
        return self.num_samples


def get_tpu_sampler(dataset: torch.utils.data.dataset.Dataset):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.trainer_pt_utils.get_tpu_sampler', 'get_tpu_sampler(dataset)', {'xm': xm, 'RandomSampler': RandomSampler, 'DistributedSampler': DistributedSampler, 'dataset': dataset}, 1)

def nested_new_like(arrays, num_samples):
    """ Create the same nested structure as `arrays` with a first dimension always at `num_samples`."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.trainer_pt_utils.nested_new_like', 'nested_new_like(arrays, num_samples)', {'nested_new_like': nested_new_like, 'np': np, 'arrays': arrays, 'num_samples': num_samples}, 1)

def nested_truncate(tensors, limit):
    """Truncate `tensors` at `limit` (even if it's a nested list/tuple of tensors)."""
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('transformers.trainer_pt_utils.nested_truncate', 'nested_truncate(tensors, limit)', {'nested_truncate': nested_truncate, 'tensors': tensors, 'limit': limit}, 1)


class DistributedTensorGatherer:
    """
    A class responsible for properly gathering tensors (or nested list/tuple of tensors) on the CPU
    by chunks.

    If our dataset has 16 samples with a batch size of 2 on 3 processes and we gather then transfer on
    CPU at every step, our sampler will generate the following indices:

        :obj:`[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1]`

    to get something of size a multiple of 3 (so that each process gets the same dataset length). Then
    process 0, 1 and 2 will be responsible of making predictions for the following samples:

        - P0: :obj:`[0, 1, 2, 3, 4, 5]`
        - P1: :obj:`[6, 7, 8, 9, 10, 11]`
        - P2: :obj:`[12, 13, 14, 15, 0, 1]`

    The first batch treated on each process will be

        - P0: :obj:`[0, 1]`
        - P1: :obj:`[6, 7]`
        - P2: :obj:`[12, 13]`

    So if we gather at the end of the first batch, we will get a tensor (nested list/tuple of tensor)
    corresponding to the following indices:

        :obj:`[0, 1, 6, 7, 12, 13]`

    If we directly concatenate our results without taking any precautions, the user will then get
    the predictions for the indices in this order at the end of the prediction loop:

        :obj:`[0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5, 10, 11, 0, 1]`

    For some reason, that's not going to roll their boat. This class is there to solve that problem.

    Args:

        world_size (:obj:`int`):
            The number of processes used in the distributed training.
        num_samples (:obj:`int`):
            The number of samples in our dataset.
        make_multiple_of (:obj:`int`, `optional`):
            If passed, the class assumes the datasets passed to each process are made to be a multiple of this argument
            (by adding samples).
    """
    
    def __init__(self, world_size, num_samples, make_multiple_of=None):
        self.world_size = world_size
        self.num_samples = num_samples
        total_size = (world_size if make_multiple_of is None else world_size * make_multiple_of)
        self.total_samples = int(np.ceil(num_samples / total_size)) * total_size
        self.process_length = self.total_samples // world_size
        self._storage = None
        self._offsets = None
    
    def add_arrays(self, arrays):
        """
        Add :obj:`arrays` to the internal storage, Will initialize the storage to the full size at the first arrays
        passed so that if we're bound to get an OOM, it happens at the beginning.
        """
        if arrays is None:
            return
        if self._storage is None:
            self._storage = nested_new_like(arrays, self.total_samples)
            self._offsets = list(range(0, self.total_samples, self.process_length))
        slice_len = self._nested_set_tensors(self._storage, arrays)
        for i in range(self.world_size):
            self._offsets[i] += slice_len
    
    def _nested_set_tensors(self, storage, arrays):
        if isinstance(arrays, (list, tuple)):
            for (x, y) in zip(storage, arrays):
                slice_len = self._nested_set_tensors(x, y)
            return slice_len
        assert arrays.shape[0] % self.world_size == 0, f'Arrays passed should all have a first dimension multiple of {self.world_size}, found {arrays.shape[0]}.'
        slice_len = arrays.shape[0] // self.world_size
        for i in range(self.world_size):
            storage[self._offsets[i]:self._offsets[i] + slice_len] = arrays[i * slice_len:(i + 1) * slice_len]
        return slice_len
    
    def finalize(self):
        """
        Return the properly gathered arrays and truncate to the number of samples (since the sampler added some extras
        to get each process a dataset of the same length).
        """
        if self._storage is None:
            return
        if self._offsets[0] != self.process_length:
            logger.warn('Not all data has been set. Are you sure you passed all values?')
        return nested_truncate(self._storage, self.num_samples)


