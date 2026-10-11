"""Forward-model surrogates (emulators) built on a probabilistic model.

``polynomial_chaos`` fits regression PCEs with closed-form leave-one-out error
and analytic moments and Sobol indices; ``gaussian_process`` fits GP
regression with ARD kernels by maximum marginal likelihood; ``adapter`` wraps
any fitted surrogate as a forward model so it can replace the original model
inside :class:`probstat.modeling.ProbabilisticModel`.
"""

from .adapter import Surrogate, SurrogateModel, surrogate_probabilistic_model
from .gaussian_process import (
    KERNELS,
    GaussianProcessHyperparameters,
    GaussianProcessLOO,
    GaussianProcessRegression,
    fit_gaussian_process,
    fit_gaussian_process_to_model,
    gaussian_process_log_marginal_likelihood,
    kernel_value,
)
from .polynomial_chaos import (
    BASES,
    PolynomialChaosExpansion,
    PolynomialChaosSobol,
    fit_polynomial_chaos,
    hermite_orthonormal,
    legendre_orthonormal,
    multi_indices,
    normalized_information_matrix,
    polynomial_chaos_from_samples,
)

__all__ = [
    "BASES",
    "KERNELS",
    "GaussianProcessHyperparameters",
    "GaussianProcessLOO",
    "GaussianProcessRegression",
    "PolynomialChaosExpansion",
    "PolynomialChaosSobol",
    "Surrogate",
    "SurrogateModel",
    "fit_gaussian_process",
    "fit_gaussian_process_to_model",
    "fit_polynomial_chaos",
    "gaussian_process_log_marginal_likelihood",
    "hermite_orthonormal",
    "kernel_value",
    "legendre_orthonormal",
    "multi_indices",
    "normalized_information_matrix",
    "polynomial_chaos_from_samples",
    "surrogate_probabilistic_model",
]
