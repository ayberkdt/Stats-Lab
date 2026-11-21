"""Generative models: flows, energy models, flow matching, neural SBI (experimental).

``flows`` composes planar, radial, affine-coupling, actnorm, and permutation
layers over a standard-normal base with an exact change-of-variables density
and fits them by maximum likelihood (:func:`fit_flow`). ``energy`` defines
energy models ``p(x) ∝ exp(-E(x))`` with a quadratic-plus-MLP energy, fitted
by implicit or denoising score matching. ``langevin`` samples a density known
up to a constant by unadjusted or Metropolis-adjusted Langevin steps, and
``diagnostics`` estimates normalizing constants, mean log-likelihoods, and
Kullback-Leibler divergences so that fitted densities can be compared with
the truth (ADR-0074).

``flow_matching`` fits a time-dependent velocity field by regression on the
straight-line conditional path and samples by integrating an ordinary
differential equation; ``posterior_estimation`` conditions that field on
an observation to give an amortized, non-Gaussian posterior for a simulator;
``ratio_estimation`` trains a classifier whose logit estimates the
likelihood-to-evidence ratio (ADR-0129).

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
    ``fit_flow_matching``: velocity network by conditional flow matching.
    ``fit_flow_matching_posterior``: amortized posterior of a simulator by
        flow matching.
    ``fit_ratio_estimator``: neural likelihood-to-evidence ratio estimator.
    ``flow_log_density``: log-density of an ODE flow by the instantaneous
        change of variables.

Assumptions:
    Training points are independent draws from a continuous law in one or
    a few dimensions (the methods are validated in one and two). Flow
    layers must stay invertible. Log-densities passed to ``sample_langevin``
    are written with operators and ``probstat.numerics.admath`` so that they
    can be differentiated. The simulation-based estimators assume that the
    prior sampler and the simulator are the ones that produced the
    observation, and that the observation lies where simulated data are
    dense.

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
    - Amortized neural posteriors and ratio estimators can be overconfident;
      run ``probstat.inference.simulation_based_calibration`` on fresh
      simulations before trusting them.
    - The flow-matching loss does not tend to zero for a perfect fit, and
      Adam has no convergence test: ``loss_decreased`` is only a sanity
      check.
    - Flow samples and log-densities use a fixed-step ODE solver without
      error control; compare two step counts.

Guide: docs/foundations/generative-models.md

References:
    [REF-PAPAMAKARIOS-ETAL-2021], [REF-HYVARINEN-2005], [REF-VINCENT-2011],
    [REF-ROBERTS-TWEEDIE-1996], [REF-LIPMAN-ETAL-2023],
    [REF-WILDBERGER-ETAL-2023], [REF-HERMANS-BEGY-LOUPPE-2020],
    [REF-HERMANS-ETAL-2022].

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
from .flow_matching import (
    FlowMatchingFit,
    GaussianMarginalVelocity,
    VelocityNetwork,
    conditional_flow_path,
    fit_flow_matching,
    flow_log_density,
    flow_matching_loss_and_gradient,
    integrate_flow,
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
from .posterior_estimation import FlowMatchingPosterior, fit_flow_matching_posterior
from .ratio_estimation import (
    GridPosterior,
    RatioEstimator,
    fit_ratio_estimator,
    ratio_loss_and_gradient,
)

__all__ = [
    "ActNorm",
    "AffineCoupling",
    "EnergyFit",
    "EnergyModel",
    "FlowFit",
    "FlowLayer",
    "FlowMatchingFit",
    "FlowMatchingPosterior",
    "GaussianMarginalVelocity",
    "GridPosterior",
    "LangevinResult",
    "NormalizingConstant",
    "NormalizingFlow",
    "Permutation",
    "PlanarFlow",
    "RadialFlow",
    "RatioEstimator",
    "SampleMean",
    "VelocityNetwork",
    "conditional_flow_path",
    "denoising_score_matching_objective",
    "fit_denoising_score_matching",
    "fit_flow",
    "fit_flow_matching",
    "fit_flow_matching_posterior",
    "fit_ratio_estimator",
    "fit_score_matching",
    "flow_log_density",
    "flow_matching_loss_and_gradient",
    "importance_normalizing_constant",
    "integrate_flow",
    "kl_divergence_estimate",
    "mean_log_likelihood",
    "quadrature_normalizing_constant",
    "ratio_loss_and_gradient",
    "real_nvp_flow",
    "sample_langevin",
    "score_matching_objective",
]
