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
inputs (see :mod:`probstat.modeling.random_field`).
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
    "KERNEL_FAMILIES",
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
    "ExponentialKLMode",
    "GeometricBrownianMotion",
    "JumpLaw",
    "KarhunenLoeveExpansion",
    "MarkovChain",
    "OccupancySolution",
    "OrnsteinUhlenbeck",
    "PoissonProcess",
    "SamplePath",
    "ScalarSDE",
    "TransientSolution",
    "as_points",
    "brownian_increments",
    "covariance_matrix",
    "empirical_order",
    "euler_maruyama",
    "exponential_kernel_kl_modes",
    "gauss_legendre",
    "interval_rule",
    "karhunen_loeve",
    "karhunen_loeve_interval",
    "milstein",
    "rectangle_rule",
    "solve_sde",
    "strong_convergence_study",
    "uniform_grid",
    "weak_convergence_study",
]
