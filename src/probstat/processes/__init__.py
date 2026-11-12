"""Stochastic processes and Gaussian random fields.

``markov`` owns finite discrete-time Markov chains (classification, stationary
law, absorption, first passage, simulation); ``ctmc`` owns finite
continuous-time Markov chains solved analytically (uniformization with error
bounds, accumulated rewards, GTH steady state, absorption); ``continuous`` owns exactly
simulated continuous-time processes (Poisson, compound Poisson, Brownian
motion, geometric Brownian motion, Ornstein-Uhlenbeck); ``sde`` owns
Euler-Maruyama and Milstein schemes with strong and weak convergence
studies; ``semi_markov`` owns finite semi-Markov processes with arbitrary
sojourn laws (Markov renewal equation, exact absorption and limiting law,
simulation); ``random_fields`` owns covariance kernels and Karhunen-Loeve
expansions that turn a random field into a finite set of standard-normal
inputs (see :mod:`probstat.modeling.random_field`); ``score_diffusion`` owns
variance-preserving and variance-exploding forward SDEs, denoising score
matching, and reverse-time samplers (reverse SDE, predictor-corrector,
probability-flow ODE); ``neural_sde`` owns SDEs with network coefficients
fitted by Euler pseudo-likelihood.

Main entry points:
    ``MarkovChain``: finite discrete-time chain with classification,
        stationary law, and absorption analysis.
    ``ContinuousTimeMarkovChain``: finite continuous-time chain solved by
        uniformization.
    ``PoissonProcess``: homogeneous Poisson process with an exact sampler.
    ``OrnsteinUhlenbeck``: mean-reverting diffusion with exact Gaussian
        transitions.
    ``euler_maruyama``: Euler-Maruyama scheme for a ``ScalarSDE``.
    ``CovarianceKernel``: stationary isotropic covariance kernel.
    ``karhunen_loeve_interval``: truncated Karhunen-Loeve expansion of a
        Gaussian field on an interval.
    ``reverse_sde_sample``: reverse-time sampler of a score-based
        diffusion.

Assumptions:
    Markov chains are time-homogeneous on a finite state space. Kernels
    depend on the distance between points only (stationary and
    isotropic), and the Karhunen-Loeve expansion represents a Gaussian
    field; non-Gaussian fields come from a pointwise transform. The SDE
    schemes are scalar with a uniform step.

Common pitfalls:
    - ``CovarianceKernel(family, variance, length_scale)`` takes the
      variance, and ``OrnsteinUhlenbeck(reversion, level, volatility)`` has
      stationary variance ``volatility^2 / (2 reversion)``.
    - ``MarkovChain`` rejects a row that does not sum to one within
      ``1e-10``; nothing is renormalized.
    - ``stationary_distribution()`` raises when the chain has more than one
      closed class, and ``distribution_after`` of a periodic chain cycles
      instead of converging.
    - A truncated Karhunen-Loeve field has less pointwise variance than the
      kernel; report ``explained_variance_fraction`` with the results.
    - ``GeometricBrownianMotion(drift, volatility)`` has mean
      ``S0 exp(drift t)``; the drift of ``log S`` is
      ``drift - volatility^2 / 2``.

Guide: docs/foundations/stochastic-processes-and-random-fields.md

References:
    [REF-NORRIS-1997], [REF-KLOEDEN-PLATEN-1992], [REF-GHANEM-SPANOS-1991],
    [REF-RASMUSSEN-WILLIAMS-2006], [REF-SONG-ETAL-2021].

"""

from .continuous import (
    BrownianMotion,
    CompoundPoissonProcess,
    GeometricBrownianMotion,
    JumpLaw,
    OrnsteinUhlenbeck,
    PoissonProcess,
    SamplePath,
    brownian_increments,
    uniform_grid,
)
from .ctmc import (
    BoundedValue,
    ContinuousTimeMarkovChain,
    CTMCAbsorption,
    CTMCPath,
    OccupancySolution,
    TransientSolution,
)
from .markov import AbsorptionAnalysis, MarkovChain
from .neural_sde import (
    DIFFUSION_MODELS,
    NeuralSDE,
    NeuralSDEFit,
    VectorSamplePath,
    fit_neural_sde,
)
from .random_fields import (
    KERNEL_FAMILIES,
    QUADRATURE_RULES,
    CovarianceKernel,
    ExponentialKLMode,
    KarhunenLoeveExpansion,
    as_points,
    covariance_matrix,
    exponential_kernel_kl_modes,
    gauss_legendre,
    interval_rule,
    karhunen_loeve,
    karhunen_loeve_interval,
    rectangle_rule,
)
from .score_diffusion import (
    ODE_METHODS,
    DiffusionSample,
    ForwardSDE,
    GaussianScore,
    ScoreNetwork,
    TrainedScore,
    VarianceExplodingSDE,
    VariancePreservingSDE,
    perturb,
    perturbation_moments,
    predictor_corrector_sample,
    probability_flow_map,
    probability_flow_sample,
    reverse_sde_sample,
    train_score_network,
)
from .sde import (
    SCHEMES,
    ConvergenceStudy,
    ScalarSDE,
    empirical_order,
    euler_maruyama,
    milstein,
    solve_sde,
    strong_convergence_study,
    weak_convergence_study,
)
from .semi_markov import (
    RenewalCurve,
    SemiMarkovAbsorption,
    SemiMarkovEstimate,
    SemiMarkovMonteCarlo,
    SemiMarkovProcess,
    SemiMarkovTransient,
    SemiMarkovTransitionMatrix,
)

__all__ = [
    "DIFFUSION_MODELS",
    "KERNEL_FAMILIES",
    "ODE_METHODS",
    "QUADRATURE_RULES",
    "SCHEMES",
    "AbsorptionAnalysis",
    "BoundedValue",
    "BrownianMotion",
    "CTMCAbsorption",
    "CTMCPath",
    "CompoundPoissonProcess",
    "ContinuousTimeMarkovChain",
    "ConvergenceStudy",
    "CovarianceKernel",
    "DiffusionSample",
    "ExponentialKLMode",
    "ForwardSDE",
    "GaussianScore",
    "GeometricBrownianMotion",
    "JumpLaw",
    "KarhunenLoeveExpansion",
    "MarkovChain",
    "NeuralSDE",
    "NeuralSDEFit",
    "OccupancySolution",
    "OrnsteinUhlenbeck",
    "PoissonProcess",
    "RenewalCurve",
    "SamplePath",
    "ScalarSDE",
    "ScoreNetwork",
    "TrainedScore",

    "SemiMarkovAbsorption",
    "SemiMarkovEstimate",
    "SemiMarkovMonteCarlo",
    "SemiMarkovProcess",
    "SemiMarkovTransient",
    "SemiMarkovTransitionMatrix",
    "TransientSolution",
    "VarianceExplodingSDE",
    "VariancePreservingSDE",
    "VectorSamplePath",
    "as_points",
    "brownian_increments",
    "covariance_matrix",
    "empirical_order",
    "euler_maruyama",
    "exponential_kernel_kl_modes",
    "fit_neural_sde",
    "gauss_legendre",
    "interval_rule",
    "karhunen_loeve",
    "karhunen_loeve_interval",
    "milstein",
    "perturb",
    "perturbation_moments",
    "predictor_corrector_sample",
    "probability_flow_map",
    "probability_flow_sample",
    "rectangle_rule",
    "reverse_sde_sample",
    "solve_sde",
    "strong_convergence_study",
    "train_score_network",
    "uniform_grid",
    "weak_convergence_study",
]
