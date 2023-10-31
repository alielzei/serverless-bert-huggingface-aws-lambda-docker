import math
import warnings
from functools import total_ordering
import torch
from torch._six import inf
from .bernoulli import Bernoulli
from .beta import Beta
from .binomial import Binomial
from .categorical import Categorical
from .continuous_bernoulli import ContinuousBernoulli
from .dirichlet import Dirichlet
from .distribution import Distribution
from .exponential import Exponential
from .exp_family import ExponentialFamily
from .gamma import Gamma
from .geometric import Geometric
from .gumbel import Gumbel
from .half_normal import HalfNormal
from .independent import Independent
from .laplace import Laplace
from .lowrank_multivariate_normal import LowRankMultivariateNormal, _batch_lowrank_logdet, _batch_lowrank_mahalanobis
from .multivariate_normal import MultivariateNormal, _batch_mahalanobis
from .normal import Normal
from .one_hot_categorical import OneHotCategorical
from .pareto import Pareto
from .poisson import Poisson
from .transformed_distribution import TransformedDistribution
from .uniform import Uniform
from .utils import _sum_rightmost
_KL_REGISTRY = {}
_KL_MEMOIZE = {}

def register_kl(type_p, type_q):
    """
    Decorator to register a pairwise function with :meth:`kl_divergence`.
    Usage::

        @register_kl(Normal, Normal)
        def kl_normal_normal(p, q):
            # insert implementation here

    Lookup returns the most specific (type,type) match ordered by subclass. If
    the match is ambiguous, a `RuntimeWarning` is raised. For example to
    resolve the ambiguous situation::

        @register_kl(BaseP, DerivedQ)
        def kl_version1(p, q): ...
        @register_kl(DerivedP, BaseQ)
        def kl_version2(p, q): ...

    you should register a third most-specific implementation, e.g.::

        register_kl(DerivedP, DerivedQ)(kl_version1)  # Break the tie.

    Args:
        type_p (type): A subclass of :class:`~torch.distributions.Distribution`.
        type_q (type): A subclass of :class:`~torch.distributions.Distribution`.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributions.kl.register_kl', 'register_kl(type_p, type_q)', {'Distribution': Distribution, '_KL_REGISTRY': _KL_REGISTRY, '_KL_MEMOIZE': _KL_MEMOIZE, 'type_p': type_p, 'type_q': type_q}, 1)


@total_ordering
class _Match(object):
    __slots__ = ['types']
    
    def __init__(self, *types):
        self.types = types
    
    def __eq__(self, other):
        return self.types == other.types
    
    def __le__(self, other):
        for (x, y) in zip(self.types, other.types):
            if not issubclass(x, y):
                return False
            if x is not y:
                break
        return True


def _dispatch_kl(type_p, type_q):
    """
    Find the most specific approximate match, assuming single inheritance.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributions.kl._dispatch_kl', '_dispatch_kl(type_p, type_q)', {'_KL_REGISTRY': _KL_REGISTRY, 'NotImplemented': NotImplemented, '_Match': _Match, 'warnings': warnings, 'type_p': type_p, 'type_q': type_q}, 1)

def _infinite_like(tensor):
    """
    Helper function for obtaining infinite KL Divergence throughout
    """
    return torch.full_like(tensor, inf)

def _x_log_x(tensor):
    """
    Utility function for calculating x log x
    """
    return tensor * tensor.log()

def _batch_trace_XXT(bmat):
    """
    Utility function for calculating the trace of XX^{T} with X having arbitrary trailing batch dimensions
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributions.kl._batch_trace_XXT', '_batch_trace_XXT(bmat)', {'bmat': bmat}, 1)

def kl_divergence(p, q):
    """
    Compute Kullback-Leibler divergence :math:`KL(p \| q)` between two distributions.

    .. math::

        KL(p \| q) = \int p(x) \lograc {p(x)} {q(x)} \,dx

    Args:
        p (Distribution): A :class:`~torch.distributions.Distribution` object.
        q (Distribution): A :class:`~torch.distributions.Distribution` object.

    Returns:
        Tensor: A batch of KL divergences of shape `batch_shape`.

    Raises:
        NotImplementedError: If the distribution types have not been registered via
            :meth:`register_kl`.
    """
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributions.kl.kl_divergence', 'kl_divergence(p, q)', {'_KL_MEMOIZE': _KL_MEMOIZE, '_dispatch_kl': _dispatch_kl, 'NotImplemented': NotImplemented, 'p': p, 'q': q}, 1)
_euler_gamma = 0.5772156649015329

@register_kl(Bernoulli, Bernoulli)
def _kl_bernoulli_bernoulli(p, q):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributions.kl._kl_bernoulli_bernoulli', '_kl_bernoulli_bernoulli(p, q)', {'register_kl': register_kl, 'Bernoulli': Bernoulli, 'p': p, 'q': q}, 1)

@register_kl(Beta, Beta)
def _kl_beta_beta(p, q):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributions.kl._kl_beta_beta', '_kl_beta_beta(p, q)', {'torch': torch, 'register_kl': register_kl, 'Beta': Beta, 'p': p, 'q': q}, 1)

@register_kl(Binomial, Binomial)
def _kl_binomial_binomial(p, q):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributions.kl._kl_binomial_binomial', '_kl_binomial_binomial(p, q)', {'_infinite_like': _infinite_like, 'register_kl': register_kl, 'Binomial': Binomial, 'p': p, 'q': q}, 1)

@register_kl(Categorical, Categorical)
def _kl_categorical_categorical(p, q):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributions.kl._kl_categorical_categorical', '_kl_categorical_categorical(p, q)', {'register_kl': register_kl, 'Categorical': Categorical, 'p': p, 'q': q}, 1)

@register_kl(ContinuousBernoulli, ContinuousBernoulli)
def _kl_continuous_bernoulli_continuous_bernoulli(p, q):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributions.kl._kl_continuous_bernoulli_continuous_bernoulli', '_kl_continuous_bernoulli_continuous_bernoulli(p, q)', {'torch': torch, 'register_kl': register_kl, 'ContinuousBernoulli': ContinuousBernoulli, 'p': p, 'q': q}, 1)

@register_kl(Dirichlet, Dirichlet)
def _kl_dirichlet_dirichlet(p, q):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributions.kl._kl_dirichlet_dirichlet', '_kl_dirichlet_dirichlet(p, q)', {'register_kl': register_kl, 'Dirichlet': Dirichlet, 'p': p, 'q': q}, 1)

@register_kl(Exponential, Exponential)
def _kl_exponential_exponential(p, q):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributions.kl._kl_exponential_exponential', '_kl_exponential_exponential(p, q)', {'register_kl': register_kl, 'Exponential': Exponential, 'p': p, 'q': q}, 1)

@register_kl(ExponentialFamily, ExponentialFamily)
def _kl_expfamily_expfamily(p, q):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributions.kl._kl_expfamily_expfamily', '_kl_expfamily_expfamily(p, q)', {'torch': torch, '_sum_rightmost': _sum_rightmost, 'register_kl': register_kl, 'ExponentialFamily': ExponentialFamily, 'p': p, 'q': q}, 1)

@register_kl(Gamma, Gamma)
def _kl_gamma_gamma(p, q):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributions.kl._kl_gamma_gamma', '_kl_gamma_gamma(p, q)', {'torch': torch, 'register_kl': register_kl, 'Gamma': Gamma, 'p': p, 'q': q}, 1)

@register_kl(Gumbel, Gumbel)
def _kl_gumbel_gumbel(p, q):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributions.kl._kl_gumbel_gumbel', '_kl_gumbel_gumbel(p, q)', {'_euler_gamma': _euler_gamma, 'torch': torch, 'register_kl': register_kl, 'Gumbel': Gumbel, 'p': p, 'q': q}, 1)

@register_kl(Geometric, Geometric)
def _kl_geometric_geometric(p, q):
    return -p.entropy() - torch.log1p(-q.probs) / p.probs - q.logits

@register_kl(HalfNormal, HalfNormal)
def _kl_halfnormal_halfnormal(p, q):
    return _kl_normal_normal(p.base_dist, q.base_dist)

@register_kl(Laplace, Laplace)
def _kl_laplace_laplace(p, q):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributions.kl._kl_laplace_laplace', '_kl_laplace_laplace(p, q)', {'torch': torch, 'register_kl': register_kl, 'Laplace': Laplace, 'p': p, 'q': q}, 1)

@register_kl(LowRankMultivariateNormal, LowRankMultivariateNormal)
def _kl_lowrankmultivariatenormal_lowrankmultivariatenormal(p, q):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributions.kl._kl_lowrankmultivariatenormal_lowrankmultivariatenormal', '_kl_lowrankmultivariatenormal_lowrankmultivariatenormal(p, q)', {'_batch_lowrank_logdet': _batch_lowrank_logdet, '_batch_lowrank_mahalanobis': _batch_lowrank_mahalanobis, 'torch': torch, '_batch_trace_XXT': _batch_trace_XXT, 'register_kl': register_kl, 'LowRankMultivariateNormal': LowRankMultivariateNormal, 'p': p, 'q': q}, 1)

@register_kl(MultivariateNormal, LowRankMultivariateNormal)
def _kl_multivariatenormal_lowrankmultivariatenormal(p, q):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributions.kl._kl_multivariatenormal_lowrankmultivariatenormal', '_kl_multivariatenormal_lowrankmultivariatenormal(p, q)', {'_batch_lowrank_logdet': _batch_lowrank_logdet, '_batch_lowrank_mahalanobis': _batch_lowrank_mahalanobis, 'torch': torch, '_batch_trace_XXT': _batch_trace_XXT, 'register_kl': register_kl, 'MultivariateNormal': MultivariateNormal, 'LowRankMultivariateNormal': LowRankMultivariateNormal, 'p': p, 'q': q}, 1)

@register_kl(LowRankMultivariateNormal, MultivariateNormal)
def _kl_lowrankmultivariatenormal_multivariatenormal(p, q):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributions.kl._kl_lowrankmultivariatenormal_multivariatenormal', '_kl_lowrankmultivariatenormal_multivariatenormal(p, q)', {'_batch_lowrank_logdet': _batch_lowrank_logdet, '_batch_mahalanobis': _batch_mahalanobis, 'torch': torch, '_batch_trace_XXT': _batch_trace_XXT, 'register_kl': register_kl, 'LowRankMultivariateNormal': LowRankMultivariateNormal, 'MultivariateNormal': MultivariateNormal, 'p': p, 'q': q}, 1)

@register_kl(MultivariateNormal, MultivariateNormal)
def _kl_multivariatenormal_multivariatenormal(p, q):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributions.kl._kl_multivariatenormal_multivariatenormal', '_kl_multivariatenormal_multivariatenormal(p, q)', {'torch': torch, '_batch_trace_XXT': _batch_trace_XXT, '_batch_mahalanobis': _batch_mahalanobis, 'register_kl': register_kl, 'MultivariateNormal': MultivariateNormal, 'p': p, 'q': q}, 1)

@register_kl(Normal, Normal)
def _kl_normal_normal(p, q):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributions.kl._kl_normal_normal', '_kl_normal_normal(p, q)', {'register_kl': register_kl, 'Normal': Normal, 'p': p, 'q': q}, 1)

@register_kl(OneHotCategorical, OneHotCategorical)
def _kl_onehotcategorical_onehotcategorical(p, q):
    return _kl_categorical_categorical(p._categorical, q._categorical)

@register_kl(Pareto, Pareto)
def _kl_pareto_pareto(p, q):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributions.kl._kl_pareto_pareto', '_kl_pareto_pareto(p, q)', {'register_kl': register_kl, 'Pareto': Pareto, 'p': p, 'q': q}, 1)

@register_kl(Poisson, Poisson)
def _kl_poisson_poisson(p, q):
    return p.rate * (p.rate.log() - q.rate.log()) - (p.rate - q.rate)

@register_kl(TransformedDistribution, TransformedDistribution)
def _kl_transformed_transformed(p, q):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributions.kl._kl_transformed_transformed', '_kl_transformed_transformed(p, q)', {'kl_divergence': kl_divergence, '_sum_rightmost': _sum_rightmost, 'register_kl': register_kl, 'TransformedDistribution': TransformedDistribution, 'p': p, 'q': q}, 1)

@register_kl(Uniform, Uniform)
def _kl_uniform_uniform(p, q):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributions.kl._kl_uniform_uniform', '_kl_uniform_uniform(p, q)', {'register_kl': register_kl, 'Uniform': Uniform, 'p': p, 'q': q}, 1)

@register_kl(Bernoulli, Poisson)
def _kl_bernoulli_poisson(p, q):
    return -p.entropy() - (p.probs * q.rate.log() - q.rate)

@register_kl(Beta, ContinuousBernoulli)
def _kl_beta_continuous_bernoulli(p, q):
    return -p.entropy() - p.mean * q.logits - torch.log1p(-q.probs) - q._cont_bern_log_norm()

@register_kl(Beta, Pareto)
def _kl_beta_infinity(p, q):
    return _infinite_like(p.concentration1)

@register_kl(Beta, Exponential)
def _kl_beta_exponential(p, q):
    return -p.entropy() - q.rate.log() + q.rate * (p.concentration1 / (p.concentration1 + p.concentration0))

@register_kl(Beta, Gamma)
def _kl_beta_gamma(p, q):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributions.kl._kl_beta_gamma', '_kl_beta_gamma(p, q)', {'register_kl': register_kl, 'Beta': Beta, 'Gamma': Gamma, 'p': p, 'q': q}, 1)

@register_kl(Beta, Normal)
def _kl_beta_normal(p, q):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributions.kl._kl_beta_normal', '_kl_beta_normal(p, q)', {'math': math, 'register_kl': register_kl, 'Beta': Beta, 'Normal': Normal, 'p': p, 'q': q}, 1)

@register_kl(Beta, Uniform)
def _kl_beta_uniform(p, q):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributions.kl._kl_beta_uniform', '_kl_beta_uniform(p, q)', {'register_kl': register_kl, 'Beta': Beta, 'Uniform': Uniform, 'p': p, 'q': q}, 1)

@register_kl(ContinuousBernoulli, Pareto)
def _kl_continuous_bernoulli_infinity(p, q):
    return _infinite_like(p.probs)

@register_kl(ContinuousBernoulli, Exponential)
def _kl_continuous_bernoulli_exponential(p, q):
    return -p.entropy() - torch.log(q.rate) + q.rate * p.mean

@register_kl(ContinuousBernoulli, Normal)
def _kl_continuous_bernoulli_normal(p, q):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributions.kl._kl_continuous_bernoulli_normal', '_kl_continuous_bernoulli_normal(p, q)', {'math': math, 'torch': torch, 'register_kl': register_kl, 'ContinuousBernoulli': ContinuousBernoulli, 'Normal': Normal, 'p': p, 'q': q}, 1)

@register_kl(ContinuousBernoulli, Uniform)
def _kl_continuous_bernoulli_uniform(p, q):
    result = -p.entropy() + (q.high - q.low).log()
    return torch.where(torch.max(torch.ge(q.low, p.support.lower_bound), torch.le(q.high, p.support.upper_bound)), torch.ones_like(result) * inf, result)

@register_kl(Exponential, Beta)
@register_kl(Exponential, ContinuousBernoulli)
@register_kl(Exponential, Pareto)
@register_kl(Exponential, Uniform)
def _kl_exponential_infinity(p, q):
    return _infinite_like(p.rate)

@register_kl(Exponential, Gamma)
def _kl_exponential_gamma(p, q):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributions.kl._kl_exponential_gamma', '_kl_exponential_gamma(p, q)', {'torch': torch, '_euler_gamma': _euler_gamma, 'register_kl': register_kl, 'Exponential': Exponential, 'Gamma': Gamma, 'p': p, 'q': q}, 1)

@register_kl(Exponential, Gumbel)
def _kl_exponential_gumbel(p, q):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributions.kl._kl_exponential_gumbel', '_kl_exponential_gumbel(p, q)', {'torch': torch, 'register_kl': register_kl, 'Exponential': Exponential, 'Gumbel': Gumbel, 'p': p, 'q': q}, 1)

@register_kl(Exponential, Normal)
def _kl_exponential_normal(p, q):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributions.kl._kl_exponential_normal', '_kl_exponential_normal(p, q)', {'torch': torch, 'math': math, 'register_kl': register_kl, 'Exponential': Exponential, 'Normal': Normal, 'p': p, 'q': q}, 1)

@register_kl(Gamma, Beta)
@register_kl(Gamma, ContinuousBernoulli)
@register_kl(Gamma, Pareto)
@register_kl(Gamma, Uniform)
def _kl_gamma_infinity(p, q):
    return _infinite_like(p.concentration)

@register_kl(Gamma, Exponential)
def _kl_gamma_exponential(p, q):
    return -p.entropy() - q.rate.log() + q.rate * p.concentration / p.rate

@register_kl(Gamma, Gumbel)
def _kl_gamma_gumbel(p, q):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributions.kl._kl_gamma_gumbel', '_kl_gamma_gumbel(p, q)', {'torch': torch, 'register_kl': register_kl, 'Gamma': Gamma, 'Gumbel': Gumbel, 'p': p, 'q': q}, 1)

@register_kl(Gamma, Normal)
def _kl_gamma_normal(p, q):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributions.kl._kl_gamma_normal', '_kl_gamma_normal(p, q)', {'torch': torch, 'math': math, 'register_kl': register_kl, 'Gamma': Gamma, 'Normal': Normal, 'p': p, 'q': q}, 1)

@register_kl(Gumbel, Beta)
@register_kl(Gumbel, ContinuousBernoulli)
@register_kl(Gumbel, Exponential)
@register_kl(Gumbel, Gamma)
@register_kl(Gumbel, Pareto)
@register_kl(Gumbel, Uniform)
def _kl_gumbel_infinity(p, q):
    return _infinite_like(p.loc)

@register_kl(Gumbel, Normal)
def _kl_gumbel_normal(p, q):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributions.kl._kl_gumbel_normal', '_kl_gumbel_normal(p, q)', {'math': math, '_euler_gamma': _euler_gamma, 'register_kl': register_kl, 'Gumbel': Gumbel, 'Normal': Normal, 'p': p, 'q': q}, 1)

@register_kl(Laplace, Beta)
@register_kl(Laplace, ContinuousBernoulli)
@register_kl(Laplace, Exponential)
@register_kl(Laplace, Gamma)
@register_kl(Laplace, Pareto)
@register_kl(Laplace, Uniform)
def _kl_laplace_infinity(p, q):
    return _infinite_like(p.loc)

@register_kl(Laplace, Normal)
def _kl_laplace_normal(p, q):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributions.kl._kl_laplace_normal', '_kl_laplace_normal(p, q)', {'torch': torch, 'math': math, 'register_kl': register_kl, 'Laplace': Laplace, 'Normal': Normal, 'p': p, 'q': q}, 1)

@register_kl(Normal, Beta)
@register_kl(Normal, ContinuousBernoulli)
@register_kl(Normal, Exponential)
@register_kl(Normal, Gamma)
@register_kl(Normal, Pareto)
@register_kl(Normal, Uniform)
def _kl_normal_infinity(p, q):
    return _infinite_like(p.loc)

@register_kl(Normal, Gumbel)
def _kl_normal_gumbel(p, q):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributions.kl._kl_normal_gumbel', '_kl_normal_gumbel(p, q)', {'torch': torch, 'math': math, 'register_kl': register_kl, 'Normal': Normal, 'Gumbel': Gumbel, 'p': p, 'q': q}, 1)

@register_kl(Pareto, Beta)
@register_kl(Pareto, ContinuousBernoulli)
@register_kl(Pareto, Uniform)
def _kl_pareto_infinity(p, q):
    return _infinite_like(p.scale)

@register_kl(Pareto, Exponential)
def _kl_pareto_exponential(p, q):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributions.kl._kl_pareto_exponential', '_kl_pareto_exponential(p, q)', {'register_kl': register_kl, 'Pareto': Pareto, 'Exponential': Exponential, 'p': p, 'q': q}, 1)

@register_kl(Pareto, Gamma)
def _kl_pareto_gamma(p, q):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributions.kl._kl_pareto_gamma', '_kl_pareto_gamma(p, q)', {'register_kl': register_kl, 'Pareto': Pareto, 'Gamma': Gamma, 'p': p, 'q': q}, 1)

@register_kl(Pareto, Normal)
def _kl_pareto_normal(p, q):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributions.kl._kl_pareto_normal', '_kl_pareto_normal(p, q)', {'math': math, 'register_kl': register_kl, 'Pareto': Pareto, 'Normal': Normal, 'p': p, 'q': q}, 1)

@register_kl(Poisson, Bernoulli)
@register_kl(Poisson, Binomial)
def _kl_poisson_infinity(p, q):
    return _infinite_like(p.rate)

@register_kl(Uniform, Beta)
def _kl_uniform_beta(p, q):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributions.kl._kl_uniform_beta', '_kl_uniform_beta(p, q)', {'torch': torch, '_x_log_x': _x_log_x, 'register_kl': register_kl, 'Uniform': Uniform, 'Beta': Beta, 'p': p, 'q': q}, 1)

@register_kl(Uniform, ContinuousBernoulli)
def _kl_uniform_continuous_bernoulli(p, q):
    result = -p.entropy() - p.mean * q.logits - torch.log1p(-q.probs) - q._cont_bern_log_norm()
    return torch.where(torch.max(torch.ge(p.high, q.support.upper_bound), torch.le(p.low, q.support.lower_bound)), torch.ones_like(result) * inf, result)

@register_kl(Uniform, Exponential)
def _kl_uniform_exponetial(p, q):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributions.kl._kl_uniform_exponetial', '_kl_uniform_exponetial(p, q)', {'register_kl': register_kl, 'Uniform': Uniform, 'Exponential': Exponential, 'p': p, 'q': q}, 1)

@register_kl(Uniform, Gamma)
def _kl_uniform_gamma(p, q):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributions.kl._kl_uniform_gamma', '_kl_uniform_gamma(p, q)', {'_x_log_x': _x_log_x, 'register_kl': register_kl, 'Uniform': Uniform, 'Gamma': Gamma, 'p': p, 'q': q}, 1)

@register_kl(Uniform, Gumbel)
def _kl_uniform_gumbel(p, q):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributions.kl._kl_uniform_gumbel', '_kl_uniform_gumbel(p, q)', {'torch': torch, 'register_kl': register_kl, 'Uniform': Uniform, 'Gumbel': Gumbel, 'p': p, 'q': q}, 1)

@register_kl(Uniform, Normal)
def _kl_uniform_normal(p, q):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributions.kl._kl_uniform_normal', '_kl_uniform_normal(p, q)', {'math': math, 'register_kl': register_kl, 'Uniform': Uniform, 'Normal': Normal, 'p': p, 'q': q}, 1)

@register_kl(Uniform, Pareto)
def _kl_uniform_pareto(p, q):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributions.kl._kl_uniform_pareto', '_kl_uniform_pareto(p, q)', {'_x_log_x': _x_log_x, 'register_kl': register_kl, 'Uniform': Uniform, 'Pareto': Pareto, 'p': p, 'q': q}, 1)

@register_kl(Independent, Independent)
def _kl_independent_independent(p, q):
    import custom_funtemplate
    return custom_funtemplate.rewrite_template('torch.distributions.kl._kl_independent_independent', '_kl_independent_independent(p, q)', {'kl_divergence': kl_divergence, '_sum_rightmost': _sum_rightmost, 'register_kl': register_kl, 'Independent': Independent, 'p': p, 'q': q}, 1)

