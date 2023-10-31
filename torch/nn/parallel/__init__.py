from .parallel_apply import parallel_apply
from .replicate import replicate
from .data_parallel import DataParallel, data_parallel
from .scatter_gather import scatter, gather
from .distributed import DistributedDataParallel
__all__ = ['replicate', 'scatter', 'parallel_apply', 'gather', 'data_parallel', 'DataParallel', 'DistributedDataParallel']

def DistributedDataParallelCPU(*args, **kwargs):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.parallel.__init__.DistributedDataParallelCPU', 'DistributedDataParallelCPU(*args, **kwargs)', {'DistributedDataParallel': DistributedDataParallel, 'args': args, 'kwargs': kwargs}, 1)

