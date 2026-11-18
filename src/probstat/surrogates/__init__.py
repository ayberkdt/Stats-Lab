"""Forward-model surrogates (emulators) built on a probabilistic model.

``polynomial_chaos`` fits regression PCEs with closed-form leave-one-out error
and analytic moments and Sobol indices; ``sparse_chaos`` selects a sparse PCE
basis by hybrid least-angle regression or orthogonal matching pursuit with
degree and q-norm adaptivity; ``gaussian_process`` fits GP
regression with ARD kernels by maximum marginal likelihood; ``pc_kriging``
fits universal kriging with an arbitrary regression trend and PC-Kriging, a
universal kriging model whose trend is a LAR-selected sparse PCE; ``adapter`` wraps
any fitted surrogate as a forward model so it can replace the original model
inside :class:`probstat.modeling.ProbabilisticModel`.

Main entry points:
    ``fit_polynomial_chaos``: regression PCE of one QoI on a design drawn from
        the model's input law.
    ``PolynomialChaosExpansion``: fitted expansion with leave-one-out error,
        moments, and Sobol indices.
    ``fit_sparse_polynomial_chaos``: sparse PCE by hybrid LAR or OMP on a
        design of the model, scored by the corrected leave-one-out error;
        ``selection`` holds the LAR ranking of the basis terms.
    ``fit_gaussian_process_to_model``: GP surrogate of one QoI trained on an
        LHS design of the model.
    ``fit_gaussian_process``: GP regression on given points by maximum
        marginal likelihood.
    ``GaussianProcessRegularization``: opt-in bounds and prior that keep GP
        hyperparameters away from degenerate optima.
    ``fit_universal_kriging``: kriging with a regression trend (ordinary
        kriging for a constant), GLS trend and concentrated likelihood.
    ``fit_pc_kriging``: PC-Kriging of a model or of given samples, in the
        ``"sequential"`` or ``"optimal"`` variant.
    ``surrogate_probabilistic_model``: a new probabilistic model whose forward
        map is the fitted surrogate.

Assumptions:
    The response is a smooth deterministic function of the inputs (a GP with
    ``estimate_noise=True`` also fits additive noise). A PCE in the Legendre
    basis needs independent uniform inputs; the Hermite basis accepts any
    marginals and any Gaussian copula. GP and kriging kernels are
    stationary; ``fit_gaussian_process`` has a constant mean, universal
    kriging and PC-Kriging a regression trend.

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
    - A sparse PCE is scored on the same runs that chose its basis, so its
      corrected leave-one-out error is still somewhat optimistic; the
      ``selection`` path shows how flat the error is around the chosen set.
    - PC-Kriging correlates on the standardized PCE variables, so its
      ``length_scales`` are dimensionless, not in physical units. A length
      scale at the lower bound of its box means the residual process is
      close to white noise and the surrogate is essentially the sparse PCE.
    - Every fitter defaults to ``seed=None`` (not reproducible); pass a seed
      or ``RandomStream``. Results computed on a surrogate inherit its error
      silently, so report its leave-one-out error with them.

Guide: docs/foundations/surrogate-models.md

References:
    [REF-XIU-KARNIADAKIS-2002], [REF-SUDRET-2008],
    [REF-BLATMAN-SUDRET-2011], [REF-EFRON-HASTIE-JOHNSTONE-TIBSHIRANI-2004],
    [REF-RASMUSSEN-WILLIAMS-2006, Chapters 2, 4, and 5],
    [REF-SCHOBI-SUDRET-WIART-2015].

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
from .pc_kriging import (
    PC_KRIGING_VARIANTS,
    ChaosTrend,
    KrigingLOO,
    PCKriging,
    UniversalKriging,
    fit_pc_kriging,
    fit_universal_kriging,
    pc_kriging_from_samples,
)
from .polynomial_chaos import (
    BASES,
    PolynomialChaosExpansion,
    PolynomialChaosSobol,
    SparseChaosCandidate,
    SparseChaosSelection,
    fit_polynomial_chaos,
    hermite_orthonormal,
    legendre_orthonormal,
    multi_indices,
    normalized_information_matrix,
    polynomial_chaos_from_samples,
)
from .sparse_chaos import (
    SPARSE_CHAOS_METHODS,
    LeastAngleRegressionPath,
    fit_sparse_polynomial_chaos,
    least_angle_regression,
    sparse_polynomial_chaos_from_samples,
)

__all__ = [
    "BASES",
    "ChaosTrend",
    "KERNELS",
    "GaussianProcessConditioningWarning",
    "GaussianProcessHyperparameters",
    "GaussianProcessJitterWarning",
    "GaussianProcessLOO",
    "GaussianProcessRegression",
    "GaussianProcessRegularization",
    "KrigingLOO",
    "LeastAngleRegressionPath",
    "PCKriging",
    "PC_KRIGING_VARIANTS",
    "PolynomialChaosExpansion",
    "PolynomialChaosSobol",
    "SPARSE_CHAOS_METHODS",
    "SparseChaosCandidate",
    "SparseChaosSelection",
    "Surrogate",
    "SurrogateModel",
    "UniversalKriging",
    "fit_gaussian_process",
    "fit_gaussian_process_to_model",
    "fit_pc_kriging",
    "fit_polynomial_chaos",
    "fit_sparse_polynomial_chaos",
    "fit_universal_kriging",
    "gaussian_process_log_marginal_likelihood",
    "gaussian_process_log_parameter_bounds",
    "hermite_orthonormal",
    "kernel_value",
    "least_angle_regression",
    "legendre_orthonormal",
    "multi_indices",
    "normalized_information_matrix",
    "pc_kriging_from_samples",
    "polynomial_chaos_from_samples",
    "sparse_polynomial_chaos_from_samples",
    "surrogate_probabilistic_model",
]
