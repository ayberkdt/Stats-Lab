"""Information theory: entropy, divergence, and mutual information.

* :mod:`.discrete` — Shannon, joint and conditional entropy, mutual
  information, Kullback-Leibler, cross-entropy, and Jensen-Shannon
  divergences for finite PMFs.
* :mod:`.differential` — closed-form differential entropies and Gaussian
  Kullback-Leibler divergences, plus :func:`differential_entropy` dispatching
  on distribution objects.
* :mod:`.estimators` — Kozachenko-Leonenko entropy and KSG mutual-information
  estimators from samples.

Every function takes ``unit="nats"`` (natural logarithm) or ``unit="bits"``.
The package sits at layer 2 and imports only layers 0-1.
"""

from ._units import UNITS, InformationUnit
from .differential import (
    beta_entropy,
    differential_entropy,
    exponential_entropy,
    gamma_entropy,
    gev_entropy,
    kl_multivariate_normal,
    kl_normal,
    lognormal_entropy,
    multivariate_normal_entropy,
    normal_entropy,
    uniform_entropy,
)
from .discrete import (
    conditional_entropy,
    cross_entropy,
    entropy,
    jensen_shannon_divergence,
    joint_entropy,
    kl_divergence,
    marginals,
    mutual_information,
)
from .estimators import kozachenko_leonenko_entropy, ksg_mutual_information

__all__ = [
    "UNITS",
    "InformationUnit",
    "beta_entropy",
    "conditional_entropy",
    "cross_entropy",
    "differential_entropy",
    "entropy",
    "exponential_entropy",
    "gamma_entropy",
    "gev_entropy",
    "jensen_shannon_divergence",
    "joint_entropy",
    "kl_divergence",
    "kl_multivariate_normal",
    "kl_normal",
    "kozachenko_leonenko_entropy",
    "ksg_mutual_information",
    "lognormal_entropy",
    "marginals",
    "multivariate_normal_entropy",
    "mutual_information",
    "normal_entropy",
    "uniform_entropy",
]
