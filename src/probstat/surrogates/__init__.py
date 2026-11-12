"""Forward-model surrogates (emulators) built on a probabilistic model.

``polynomial_chaos`` fits regression PCEs with closed-form leave-one-out error
and analytic moments and Sobol indices; ``gaussian_process`` fits GP
regression with ARD kernels by maximum marginal likelihood; ``adapter`` wraps
any fitted surrogate as a forward model so it can replace the original model
inside :class:`probstat.modeling.ProbabilisticModel`.

Main entry points:
    ``fit_polynomial_chaos``: regression PCE of one QoI on a design drawn from
        the model's input law.
    ``PolynomialChaosExpansion``: fitted expansion with leave-one-out error,
        moments, and Sobol indices.
    ``fit_gaussian_process_to_model``: GP surrogate of one QoI trained on an
        LHS design of the model.
    ``fit_gaussian_process``: GP regression on given points by maximum
        marginal likelihood.
    ``GaussianProcessRegularization``: opt-in bounds and prior that keep GP
        hyperparameters away from degenerate optima.
    ``surrogate_probabilistic_model``: a new probabilistic model whose forward
        map is the fitted surrogate.

Assumptions:
    The response is a smooth deterministic function of the inputs (a GP with
    ``estimate_noise=True`` also fits additive noise). A PCE in the Legendre
    basis needs independent uniform inputs; the Hermite basis accepts any
    marginals and any Gaussian copula. GP kernels are stationary with a
    constant mean.

Common pitfalls:
    - The GP ``nugget`` (default ``1e-10``) is a diagonal variance in
      standardized output units. With ``adaptive_nugget=True`` (the default
      of ``fit_gaussian_process``) it is multiplied by ten, up to ``1e-4``,
      when no start gives a positive-definite kernel matrix; that increase
      emits ``GaussianProcessJitterWarning`` and the value used is the
      result's ``nugget``. A condition number above ``1e12`` emits
      ``GaussianProcessConditioningWarning``.
    - Both GP fitters default to ``regularization=None``, the plain maximum
      likelihood fit, which on small designs can drift to huge length scales
      (a zero predictive variance) or to a white-noise optimum. Pass
      ``GaussianProcessRegularization()`` and read ``at_bound``.
    - A GP predictive variance measures distance from the training data, not
      the true approximation error.
    - A PCE leave-one-out error above one predicts held-out points worse than
      the output mean; its moments and indices are then meaningless.
      ``sobol_indices()`` raises for dependent inputs, and the Legendre basis
      raises for non-uniform or dependent inputs.
    - Every fitter defaults to ``seed=None`` (not reproducible); pass a seed
      or ``RandomStream``. Results computed on a surrogate inherit its error
      silently, so report its leave-one-out error with them.

Guide: docs/foundations/surrogate-models.md

References:
    [REF-XIU-KARNIADAKIS-2002], [REF-SUDRET-2008],
    [REF-BLATMAN-SUDRET-2011], [REF-RASMUSSEN-WILLIAMS-2006, Chapters 2, 4,
    and 5].

"""

from .adapter import Surrogate, SurrogateModel, surrogate_probabilistic_model
from .gaussian_process import (
    KERNELS,
    GaussianProcessConditioningWarning,
    GaussianProcessHyperparameters,
    GaussianProcessJitterWarning,
    GaussianProcessLOO,
    GaussianProcessRegression,
    GaussianProcessRegularization,
    fit_gaussian_process,
    fit_gaussian_process_to_model,
    gaussian_process_log_marginal_likelihood,
    gaussian_process_log_parameter_bounds,
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
    "GaussianProcessConditioningWarning",
    "GaussianProcessHyperparameters",
    "GaussianProcessJitterWarning",
    "GaussianProcessLOO",
    "GaussianProcessRegression",
    "GaussianProcessRegularization",
    "PolynomialChaosExpansion",
    "PolynomialChaosSobol",
    "Surrogate",
    "SurrogateModel",
    "fit_gaussian_process",
    "fit_gaussian_process_to_model",
    "fit_polynomial_chaos",
    "gaussian_process_log_marginal_likelihood",
    "gaussian_process_log_parameter_bounds",
    "hermite_orthonormal",
    "kernel_value",
    "legendre_orthonormal",
    "multi_indices",
    "normalized_information_matrix",
    "polynomial_chaos_from_samples",
    "surrogate_probabilistic_model",
]
