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

Main entry points:
    ``real_nvp_flow``: affine couplings with coordinate reversals and a
        final actnorm (dimension at least two).
    ``fit_flow``: maximum-likelihood fit of a normalizing flow by Adam.
    ``NormalizingFlow``: composed flow with an exact log-density.
    ``EnergyModel``: quadratic-plus-MLP energy ``E(x)``.
    ``fit_score_matching``: implicit (Hyvarinen) score-matching fit.
    ``fit_denoising_score_matching``: denoising score-matching fit.
    ``sample_langevin``: unadjusted or Metropolis-adjusted Langevin
        sampler.
    ``quadrature_normalizing_constant``: normalizing constant by adaptive
        quadrature on a box.

Assumptions:
    Training points are independent draws from a continuous law in one or
    a few dimensions (the methods are validated in one and two). Flow
    layers must stay invertible. Log-densities passed to ``sample_langevin``
    are written with operators and ``probstat.numerics.admath`` so that they
    can be differentiated.

Common pitfalls:
    - ``fit_flow``, ``fit_score_matching``, ``fit_denoising_score_matching``,
      and ``sample_langevin`` default to ``seed=0``: unseeded calls repeat
      the same run.
    - ``EnergyModel(d)`` includes an MLP with ``hidden=(16,)``; pass
      ``hidden=None`` for the plain Gaussian energy.
    - Denoising score matching fits the data smoothed by
      ``N(0, noise_scale^2 I)``, not the data themselves.
    - ``quadrature_normalizing_constant`` integrates only over the given
      box; its ``error`` does not include the mass outside it.
    - ``sample_langevin(..., adjusted=False)`` (ULA) is biased by ``O(h)``.

Guide: docs/foundations/generative-models.md

References:
    [REF-PAPAMAKARIOS-ETAL-2021], [REF-HYVARINEN-2005], [REF-VINCENT-2011],
    [REF-ROBERTS-TWEEDIE-1996].

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
