"""Validated probability-distribution objects and support descriptors.

Each family is a frozen dataclass with validated parameters, a ``support``
descriptor, ``pdf``/``pmf`` and their logarithms, ``cdf``, a directly
computed ``sf``, ``quantile``, ``isf``, moment properties, and
``sample(size, *, seed)``. ``Affine``, ``Mixture``, and ``Truncated`` build
new laws from existing ones; ``Empirical`` wraps observed data.

Main entry points:
    ``Normal``: Gaussian law with location ``mu`` and standard deviation
        ``sigma``.
    ``LogNormal``: law of ``exp(Y)`` with ``Y ~ Normal(mu, sigma)``.
    ``Gamma``: shape-rate gamma law (``Gamma.from_scale`` for shape-scale).
    ``Weibull``: shape-scale Weibull law for lifetimes and strengths.
    ``Binomial``: number of successes in ``n`` independent trials.
    ``Poisson``: count with mean and variance ``lam``.
    ``Truncated``: restriction of any continuous law to an interval.
    ``Mixture``: finite weighted mixture of continuous component laws.

Assumptions:
    Parameters are finite and inside the documented domain (invalid values
    raise rather than being clipped). Each family's parameterization is the
    one in its class docstring; ``sample`` draws independent observations
    from one call-local generator per call.

Common pitfalls:
    - The second parameter of ``Normal`` and ``LogNormal`` is a standard
      deviation, not a variance, and ``LogNormal(mu, sigma)`` describes
      ``log X``: its mean is ``exp(mu + sigma^2 / 2)``, not ``mu``.
    - ``Exponential`` and ``Gamma`` take a rate (``Gamma(2.0, 3.0).mean``
      is ``2/3``); use ``Gamma.from_scale(shape, scale)`` for a scale.
      ``Geometric`` and ``NegativeBinomial`` count failures, with support
      starting at 0.
    - Keyword names differ between families: ``Triangular(lower, mode,
      upper)`` but ``PERT(low, mode, high)``, ``Normal(mu, sigma)`` but
      ``Laplace(loc, scale)``; check the class before passing keywords.
    - ``mean``, ``variance``, ``std``, ``skewness``, and ``kurtosis`` are
      properties; ``kurtosis`` is the excess kurtosis, and an undefined
      moment is ``nan`` (``Cauchy().mean``, ``StudentT(1.0).variance``)
      while a divergent one is ``inf`` (``StudentT(2.0).variance``).
    - ``sample(seed=None)`` draws fresh entropy. The same integer seed
      reused for two laws replays the same generator stream, so the draws
      are dependent (``Uniform().sample(n, seed=5)`` and
      ``Exponential().sample(n, seed=5)`` are monotone transforms of each
      other); give each input its own ``RandomStream`` child.

Guide: docs/foundations/continuous-distribution-catalog.md

References:
    [REF-JKB-1994] and [REF-JKB-1995] (continuous families),
    [REF-ROSS-2019] (discrete families), [REF-NIST-DISTRIBUTIONS]
    (parameterizations), [REF-DEVROYE-1986] (variate generation).

"""

from ._base import ContinuousDistribution, DiscreteDistribution
from .bernoulli import Bernoulli
from .beta import Beta
from .binomial import Binomial
from .categorical import Categorical
from .cauchy import Cauchy
from .chi_square import ChiSquare
from .discrete_uniform import DiscreteUniform
from .empirical import Empirical
from .exponential import Exponential
from .fisher_f import FisherF
from .gamma import Gamma
from .geometric import Geometric
from .gev import GeneralizedExtremeValue
from .gpd import GeneralizedPareto
from .gumbel import Gumbel
from .hypergeometric import Hypergeometric
from .laplace import Laplace
from .logistic import Logistic
from .lognormal import LogNormal
from .negative_binomial import NegativeBinomial
from .normal import Normal
from .pareto import Pareto
from .pert import PERT
from .poisson import Poisson
from .rayleigh import Rayleigh
from .student_t import StudentT
from .support import FiniteSupport, IntegerSupport, RealSupport
from .triangular import Triangular
from .truncated_normal import TruncatedNormal
from .uniform import Uniform
from .weibull import Weibull
from .wrappers import Affine, Mixture, Truncated

__all__ = [
    "Affine",
    "Bernoulli",
    "Beta",
    "Binomial",
    "Categorical",
    "Cauchy",
    "ChiSquare",
    "ContinuousDistribution",
    "DiscreteDistribution",
    "DiscreteUniform",
    "Empirical",
    "Exponential",
    "FiniteSupport",
    "FisherF",
    "Gamma",
    "GeneralizedExtremeValue",
    "GeneralizedPareto",
    "Geometric",
    "Gumbel",
    "Hypergeometric",
    "IntegerSupport",
    "Laplace",
    "Logistic",
    "LogNormal",
    "Mixture",
    "NegativeBinomial",
    "Normal",
    "Pareto",
    "PERT",
    "Poisson",
    "Rayleigh",
    "RealSupport",
    "StudentT",
    "Triangular",
    "Truncated",
    "TruncatedNormal",
    "Uniform",
    "Weibull",
]
