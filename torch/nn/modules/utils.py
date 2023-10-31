from torch._six import container_abcs
from itertools import repeat

def _ntuple(n):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.nn.modules.utils._ntuple', '_ntuple(n)', {'container_abcs': container_abcs, 'repeat': repeat, 'n': n}, 1)
_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)

def _repeat_tuple(t, n):
    """Repeat each element of `t` for `n` times.

    This can be used to translate padding arg used by Conv and Pooling modules
    to the ones used by `F.pad`.
    """
    return tuple((x for x in t for _ in range(n)))

def _list_with_default(out_size, defaults):
    if isinstance(out_size, int):
        return out_size
    if len(defaults) <= len(out_size):
        raise ValueError('Input dimension should be at least {}'.format(len(out_size) + 1))
    return [(v if v is not None else d) for (v, d) in zip(out_size, defaults[-len(out_size):])]

