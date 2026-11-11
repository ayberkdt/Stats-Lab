"""Stochastic processes and Gaussian random fields.

``markov`` owns finite discrete-time Markov chains (classification, stationary
law, absorption, first passage, simulation); ``ctmc`` owns finite
continuous-time Markov chains solved analytically (uniformization with error
bounds, accumulated rewards, GTH steady state, absorption); ``continuous`` owns exactly
simulated continuous-time processes (Poisson, compound Poisson, Brownian
motion, geometric Brownian motion, Ornstein-Uhlenbeck); ``sde`` owns
Euler-Maruyama and Milstein schemes with strong and weak convergence
studies; ``random_fields`` owns covariance kernels and Karhunen-Loeve
expansions that turn a random field into a finite set of standard-normal
inputs (see :mod:`probstat.modeling.random_field`); ``score_diffusion`` owns
variance-preserving and variance-exploding forward SDEs, denoising score
matching, and reverse-time samplers (reverse SDE, predictor-corrector,
probability-flow ODE); ``neural_sde`` owns SDEs with network coefficients
fitted by Euler pseudo-likelihood.
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
    "SamplePath",
    "ScalarSDE",
    "ScoreNetwork",
    "TrainedScore",
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
