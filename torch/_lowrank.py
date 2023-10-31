"""Implement various linear algebra algorithms for low rank matrices.
"""

__all__ = ['svd_lowrank', 'pca_lowrank']
import torch
from . import _linalg_utils as _utils
from ._overrides import has_torch_function, handle_torch_function

def get_approximate_basis(A, q, niter=2, M=None):
    """Return tensor :math:`Q` with :math:`q` orthonormal columns such
    that :math:`Q Q^H A` approximates :math:`A`. If :math:`M` is
    specified, then :math:`Q` is such that :math:`Q Q^H (A - M)`
    approximates :math:`A - M`.

    .. note:: The implementation is based on the Algorithm 4.4 from
              Halko et al, 2009.

    .. note:: For an adequate approximation of a k-rank matrix
              :math:`A`, where k is not known in advance but could be
              estimated, the number of :math:`Q` columns, q, can be
              choosen according to the following criteria: in general,
              :math:`k <= q <= min(2*k, m, n)`. For large low-rank
              matrices, take :math:`q = k + 5..10`.  If k is
              relatively small compared to :math:`min(m, n)`, choosing
              :math:`q = k + 0..2` may be sufficient.

    .. note:: To obtain repeatable results, reset the seed for the
              pseudorandom number generator

    Arguments::
        A (Tensor): the input tensor of size :math:`(*, m, n)`

        q (int): the dimension of subspace spanned by :math:`Q`
                 columns.

        niter (int, optional): the number of subspace iterations to
                               conduct; ``niter`` must be a
                               nonnegative integer. In most cases, the
                               default value 2 is more than enough.

        M (Tensor, optional): the input tensor's mean of size
                              :math:`(*, 1, n)`.

    References::
        - Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp, Finding
          structure with randomness: probabilistic algorithms for
          constructing approximate matrix decompositions,
          arXiv:0909.4061 [math.NA; math.PR], 2009 (available at
          `arXiv <http://arxiv.org/abs/0909.4061>`_).
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch._lowrank.get_approximate_basis', 'get_approximate_basis(A, q, niter=2, M=None)', {'_utils': _utils, 'torch': torch, 'A': A, 'q': q, 'niter': niter, 'M': M}, 1)

def svd_lowrank(A, q=6, niter=2, M=None):
    """Return the singular value decomposition ``(U, S, V)`` of a matrix,
    batches of matrices, or a sparse matrix :math:`A` such that
    :math:`A pprox U diag(S) V^T`. In case :math:`M` is given, then
    SVD is computed for the matrix :math:`A - M`.

    .. note:: The implementation is based on the Algorithm 5.1 from
              Halko et al, 2009.

    .. note:: To obtain repeatable results, reset the seed for the
              pseudorandom number generator

    .. note:: The input is assumed to be a low-rank matrix.

    .. note:: In general, use the full-rank SVD implementation
              ``torch.svd`` for dense matrices due to its 10-fold
              higher performance characteristics. The low-rank SVD
              will be useful for huge sparse matrices that
              ``torch.svd`` cannot handle.

    Arguments::
        A (Tensor): the input tensor of size :math:`(*, m, n)`

        q (int, optional): a slightly overestimated rank of A.

        niter (int, optional): the number of subspace iterations to
                               conduct; niter must be a nonnegative
                               integer, and defaults to 2

        M (Tensor, optional): the input tensor's mean of size
                              :math:`(*, 1, n)`.

    References::
        - Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp, Finding
          structure with randomness: probabilistic algorithms for
          constructing approximate matrix decompositions,
          arXiv:0909.4061 [math.NA; math.PR], 2009 (available at
          `arXiv <http://arxiv.org/abs/0909.4061>`_).

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch._lowrank.svd_lowrank', 'svd_lowrank(A, q=6, niter=2, M=None)', {'torch': torch, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'svd_lowrank': svd_lowrank, '_svd_lowrank': _svd_lowrank, 'A': A, 'q': q, 'niter': niter, 'M': M}, 1)

def _svd_lowrank(A, q=6, niter=2, M=None):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch._lowrank._svd_lowrank', '_svd_lowrank(A, q=6, niter=2, M=None)', {'_utils': _utils, 'get_approximate_basis': get_approximate_basis, 'torch': torch, 'A': A, 'q': q, 'niter': niter, 'M': M}, 3)

def pca_lowrank(A, q=None, center=True, niter=2):
    """Performs linear Principal Component Analysis (PCA) on a low-rank
    matrix, batches of such matrices, or sparse matrix.

    This function returns a namedtuple ``(U, S, V)`` which is the
    nearly optimal approximation of a singular value decomposition of
    a centered matrix :math:`A` such that :math:`A = U diag(S) V^T`.

    .. note:: The relation of ``(U, S, V)`` to PCA is as follows:

                - :math:`A` is a data matrix with ``m`` samples and
                  ``n`` features

                - the :math:`V` columns represent the principal directions

                - :math:`S ** 2 / (m - 1)` contains the eigenvalues of
                  :math:`A^T A / (m - 1)` which is the covariance of
                  ``A`` when ``center=True`` is provided.

                - ``matmul(A, V[:, :k])`` projects data to the first k
                  principal components

    .. note:: Different from the standard SVD, the size of returned
              matrices depend on the specified rank and q
              values as follows:

                - :math:`U` is m x q matrix

                - :math:`S` is q-vector

                - :math:`V` is n x q matrix

    .. note:: To obtain repeatable results, reset the seed for the
              pseudorandom number generator

    Arguments:

        A (Tensor): the input tensor of size :math:`(*, m, n)`

        q (int, optional): a slightly overestimated rank of
                           :math:`A`. By default, ``q = min(6, m,
                           n)``.

        center (bool, optional): if True, center the input tensor,
                                 otherwise, assume that the input is
                                 centered.

        niter (int, optional): the number of subspace iterations to
                               conduct; niter must be a nonnegative
                               integer, and defaults to 2.

    References::

        - Nathan Halko, Per-Gunnar Martinsson, and Joel Tropp, Finding
          structure with randomness: probabilistic algorithms for
          constructing approximate matrix decompositions,
          arXiv:0909.4061 [math.NA; math.PR], 2009 (available at
          `arXiv <http://arxiv.org/abs/0909.4061>`_).

    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch._lowrank.pca_lowrank', 'pca_lowrank(A, q=None, center=True, niter=2)', {'torch': torch, 'has_torch_function': has_torch_function, 'handle_torch_function': handle_torch_function, 'pca_lowrank': pca_lowrank, '_utils': _utils, '_svd_lowrank': _svd_lowrank, 'A': A, 'q': q, 'center': center, 'niter': niter}, 1)

