"""This file is allowed to initialize CUDA context when imported."""

import torch
import torch.cuda
from torch.testing._internal.common_utils import TEST_WITH_ROCM, TEST_NUMBA
TEST_CUDA = torch.cuda.is_available()
TEST_MULTIGPU = (TEST_CUDA and torch.cuda.device_count() >= 2)
CUDA_DEVICE = (TEST_CUDA and torch.device('cuda:0'))
TEST_CUDNN = (TEST_CUDA and ((TEST_WITH_ROCM or torch.backends.cudnn.is_acceptable(torch.tensor(1.0, device=CUDA_DEVICE)))))
TEST_CUDNN_VERSION = (torch.backends.cudnn.version() if TEST_CUDNN else 0)
if TEST_NUMBA:
    import numba.cuda
    TEST_NUMBA_CUDA = numba.cuda.is_available()
else:
    TEST_NUMBA_CUDA = False
__cuda_ctx_rng_initialized = False

def initialize_cuda_context_rng():
    import custom_funtemplate
    custom_funtemplate.rewrite_template('torch.testing._internal.common_cuda.initialize_cuda_context_rng', 'initialize_cuda_context_rng()', {'TEST_CUDA': TEST_CUDA, 'torch': torch}, 0)

