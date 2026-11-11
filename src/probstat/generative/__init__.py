"""Generative models: normalizing flows and energy-based models (experimental).

``flows`` composes planar, radial, affine-coupling, actnorm, and permutation
layers over a standard-normal base with an exact change-of-variables density
and fits them by maximum likelihood (:func:`fit_flow`). ``energy`` defines
energy models ``p(x) ∝ exp(-E(x))`` with a quadratic-plus-MLP energy, fitted
by implicit or denoising score matching. ``langevin`` samples a density known
up to a constant by unadjusted or Metropolis-adjusted Langevin steps, and
``diagnostics`` estimates normalizing constants, mean log-likelihoods, and
Kullback-Leibler divergences so that fitted densities can be compared with
the truth (ADR-0074).

The networks, reverse-mode derivatives, and Adam optimizer are the shared
blocks :class:`probstat.numerics.MLP`,
:func:`probstat.numerics.reverse_value_and_gradient`, and
:func:`probstat.optimization.adam_minimize`.
"""

from .diagnostics import (
    NormalizingConstant,
    SampleMean,
    importance_normalizing_constant,
    kl_divergence_estimate,
    mean_log_likelihood,
    quadrature_normalizing_constant,
)
from .energy import (
    EnergyFit,
    EnergyModel,
    denoising_score_matching_objective,
    fit_denoising_score_matching,
    fit_score_matching,
    score_matching_objective,
)
from .flows import (
    ActNorm,
    AffineCoupling,
    FlowFit,
    FlowLayer,
    NormalizingFlow,
    Permutation,
    PlanarFlow,
    RadialFlow,
    fit_flow,
    real_nvp_flow,
)
from .langevin import LangevinResult, sample_langevin

__all__ = [
    "ActNorm",
    "AffineCoupling",
    "EnergyFit",
    "EnergyModel",
    "FlowFit",
    "FlowLayer",
    "LangevinResult",
    "NormalizingConstant",
    "NormalizingFlow",
    "Permutation",
    "PlanarFlow",
    "RadialFlow",
    "SampleMean",
    "denoising_score_matching_objective",
    "fit_denoising_score_matching",
    "fit_flow",
    "fit_score_matching",
    "importance_normalizing_constant",
    "kl_divergence_estimate",
    "mean_log_likelihood",
    "quadrature_normalizing_constant",
    "real_nvp_flow",
    "sample_langevin",
    "score_matching_objective",
]
