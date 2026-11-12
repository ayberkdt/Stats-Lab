"""Validated probability-distribution objects and support descriptors.

Automatic differentiation (ADR-0090): the parameters of every family with
real parameters may be :class:`~probstat.numerics.Dual` numbers or
reverse-mode :class:`~probstat.numerics.ReverseVariable` values. They are
validated on their primal (innermost float) value and kept, and ``log_pdf``
(continuous laws, whose evaluation point may also be a variable) and
``log_pmf`` (discrete laws, integer points) return a variable that carries
exact derivatives; the probabilistic programs of
:mod:`probstat.inference.ppl` use this for their reverse-mode gradients.
Float results are bit-identical to the float-only implementation. Every
other method (``pdf`` and ``pmf`` unless they are formed from the
log-density with operators, ``cdf``, ``sf``, ``log_cdf``, ``log_sf``,
``quantile``, ``isf``, ``hazard``, moments, ``entropy``, ``support`` and
``sample``) keeps requiring floats: with a variable it raises a
``TypeError`` (usually :class:`~probstat.numerics.AutodiffUnsupportedError`)
or returns a variable computed with operators, never a value whose
derivative was silently dropped. Integer parameters (counts, trials,
``DiscreteUniform``, ``Hypergeometric``), ``Empirical`` and the composite
``Affine``, ``Mixture`` and ``Truncated`` laws take floats only.
"""

from ._base import ContinuousDistribution, DiscreteDistribution
from .affine import Affine
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
from .mixture import Mixture
from .negative_binomial import NegativeBinomial
from .normal import Normal
from .pareto import Pareto
from .pert import PERT
from .poisson import Poisson
from .rayleigh import Rayleigh
from .student_t import StudentT
from .support import FiniteSupport, IntegerSupport, RealSupport
from .triangular import Triangular
from .truncated import Truncated
from .truncated_normal import TruncatedNormal
from .uniform import Uniform
from .weibull import Weibull

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
