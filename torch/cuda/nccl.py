import warnings
import torch.cuda
__all__ = ['all_reduce', 'reduce', 'broadcast', 'all_gather', 'reduce_scatter']
SUM = 0

def is_available(tensors):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.cuda.nccl.is_available', 'is_available(tensors)', {'torch': torch, 'warnings': warnings, 'tensors': tensors}, 1)

def version():
    return torch._C._nccl_version()

def unique_id():
    return torch._C._nccl_unique_id()

def init_rank(num_ranks, uid, rank):
    return torch._C._nccl_init_rank(num_ranks, uid, rank)

def all_reduce(inputs, outputs=None, op=SUM, streams=None, comms=None):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.cuda.nccl.all_reduce', 'all_reduce(inputs, outputs=None, op=SUM, streams=None, comms=None)', {'torch': torch, 'inputs': inputs, 'outputs': outputs, 'op': op, 'streams': streams, 'comms': comms, 'SUM': SUM}, 0)

def reduce(inputs, outputs=None, root=0, op=SUM, streams=None, comms=None):
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.cuda.nccl.reduce', 'reduce(inputs, outputs=None, root=0, op=SUM, streams=None, comms=None)', {'torch': torch, 'inputs': inputs, 'outputs': outputs, 'root': root, 'op': op, 'streams': streams, 'comms': comms, 'SUM': SUM}, 0)

def broadcast(inputs, root=0, streams=None, comms=None):
    torch._C._nccl_broadcast(inputs, root, streams, comms)

def all_gather(inputs, outputs, streams=None, comms=None):
    torch._C._nccl_all_gather(inputs, outputs, streams, comms)

def reduce_scatter(inputs, outputs, op=SUM, streams=None, comms=None):
    torch._C._nccl_reduce_scatter(inputs, outputs, op, streams, comms)

