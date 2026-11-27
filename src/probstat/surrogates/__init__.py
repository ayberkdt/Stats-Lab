"""Forward-model surrogates (emulators) built on a probabilistic model.

``polynomial_chaos`` fits regression PCEs with closed-form leave-one-out error
and analytic moments and Sobol indices; ``sparse_chaos`` selects a sparse PCE
basis by hybrid least-angle regression or orthogonal matching pursuit with
degree and q-norm adaptivity; ``gaussian_process`` fits GP
regression with ARD kernels by maximum marginal likelihood; ``pc_kriging``
fits universal kriging with an arbitrary regression trend and PC-Kriging, a
universal kriging model whose trend is a LAR-selected sparse PCE; ``ensemble``
fits deep ensembles of Gaussian mean-variance networks whose predictive
variance splits into aleatoric and epistemic parts; ``multifidelity`` fits
recursive AR(1) co-kriging of several fidelity levels on nested designs;
``stochastic_kriging`` fits kriging metamodels of noisy simulators from
replicated runs; ``bayesian_quadrature`` integrates a function against a
Gaussian or uniform product measure with a squared-exponential GP prior;
``adapter`` wraps
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
    ``fit_deep_ensemble``: deep ensemble of small neural networks trained on
        the Gaussian negative log-likelihood; ``DeepEnsemble`` exposes
        ``predict``, ``predict_with_variance``, and ``epistemic_variance``.
    ``fit_deep_ensemble_to_model``: deep-ensemble surrogate of one QoI on a
        design of the model.
    ``fit_recursive_cokriging``: recursive AR(1) co-kriging of ``s`` fidelity
        levels on nested designs; ``RecursiveCoKriging`` predicts every level
        with a plug-in variance.
    ``gaussian_process_from_hyperparameters``: a GP conditioned on data with
        given kernel hyperparameters and constant mean (no fitting).
    ``fit_stochastic_kriging``: stochastic kriging from replicated noisy
        runs (``stochastic_kriging_from_summaries`` from sample means and
        their variances).
    ``bayesian_quadrature``: posterior mean and variance of an integral
        against ``GaussianMeasure`` / ``UniformMeasure`` product measures.
    ``surrogate_probabilistic_model``: a new probabilistic model whose forward
        map is the fitted surrogate.
    ``ProbabilisticSurrogate`` and ``SurrogateFactory``: the predictive
        mean-and-variance interface and the refit hook that active-learning
        reliability methods accept (``GaussianProcessFactory`` implements the
        hook for a GP).

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
    - A deep ensemble is trained in scalar Python: training costs seconds
      to minutes (see the guide's cost table), its epistemic variance is
      member disagreement rather than a calibrated error, and its aleatoric
      variance on a deterministic simulator only reflects the misfit.
    - Co-kriging needs nested designs (each higher-fidelity point is also a
      lower-fidelity point) and its variance treats the scale factors, trend
      constants, and kernel hyperparameters as known. With few expensive
      runs its likelihood is often bimodal; keep the default ten restarts.
    - Stochastic kriging plugs in the sample variances of the replications
      as if they were the true noise variances; with two or three
      replications per point they are very noisy.
    - The Bayesian quadrature variance is conditional on the kernel
      hyperparameters; when they are estimated from the same nodes it is
      usually too small.
    - Every fitter defaults to ``seed=None`` (not reproducible); pass a seed
      or ``RandomStream``. Results computed on a surrogate inherit its error
      silently, so report its leave-one-out error with them.

Guide: docs/foundations/surrogate-models.md

References:
    [REF-XIU-KARNIADAKIS-2002], [REF-SUDRET-2008],
    [REF-BLATMAN-SUDRET-2011], [REF-EFRON-HASTIE-JOHNSTONE-TIBSHIRANI-2004],
    [REF-RASMUSSEN-WILLIAMS-2006, Chapters 2, 4, and 5],
    [REF-SCHOBI-SUDRET-WIART-2015],
    [REF-LAKSHMINARAYANAN-PRITZEL-BLUNDELL-2017],
    [REF-KENNEDY-OHAGAN-2000], [REF-LE-GRATIET-GARNIER-2014],
    [REF-ANKENMAN-NELSON-STAUM-2010], [REF-OHAGAN-1991].

"""

from .adapter import (
    ProbabilisticSurrogate,
    Surrogate,
    SurrogateFactory,
    SurrogateModel,
    surrogate_probabilistic_model,
)
from .bayesian_quadrature import (
    BayesianQuadratureResult,
    GaussianMeasure,
    QuadratureMeasure,
    UniformMeasure,
    bayesian_quadrature,
    kernel_double_integral,
    kernel_mean,
)
from .ensemble import (
    DeepEnsemble,
    DeepEnsemblePrediction,
    fit_deep_ensemble,
    fit_deep_ensemble_to_model,
)
from .gaussian_process import (
    KERNELS,
    GaussianProcessConditioningWarning,
    GaussianProcessFactory,
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
from .gp_advanced import (
    DerivativeGaussianProcess,
    DerivativeKernelBlocks,
    GradientPrediction,
    HeteroscedasticGaussianProcess,
    Kernel,
    KernelGaussianProcess,
    LinearDifferentialOperator,
    LinearKernel,
    MultiOutputGaussianProcess,
    OperatorGaussianProcess,
    PeriodicKernel,
    ProductKernel,
    RationalQuadraticKernel,
    SparseGaussianProcess,
    SparseGPCost,
    StationaryKernel,
    SumKernel,
    WhiteKernel,
    derivative_kernel_blocks,
    fit_derivative_gp,
    fit_heteroscedastic_gp,
    fit_kernel_gp,
    fit_multi_output_gp,
    fit_operator_gp,
    fit_sparse_gp,
    gaussian_process_gradient,
    operator_covariances,
    select_inducing_points,
    sparse_gp_cost,
)
from .gp_anomaly import (
    ChannelSpec,
    DetectionStudy,
    FaultedSeries,
    NominalModel,
    ScoreSeries,
    SlidingWindowGP,
    SyntheticChannel,
    batch_window_predict,
    calibrate_cusum_threshold,
    contaminate_values,
    detection_study,
    fit_nominal_model,
    inject_fault,
    orbital_kernel,
    simulate_telemetry,
    sliding_window_scores,
)
from .gp_bayes import (
    BayesianGPPrediction,
    BayesianGPRegression,
    EmulatorValidation,
    GPHyperPrior,
    GPSobolResult,
    LaplaceGPClassifier,
    StudentTProcessRegression,
    WarpedGPRegression,
    box_cox,
    default_hyper_prior,
    fit_bayesian_gp,
    fit_laplace_gp_classifier,
    fit_student_t_process,
    fit_warped_gp,
    gp_predictive_moments,
    gp_sobol_posterior,
    inverse_box_cox,
    sobol_estimates,
    student_t_process_log_marginal_likelihood,
    validate_emulator_moments,
    validate_gp_emulator,
)
from .latent_force import (
    LatentForceChannel,
    LatentForceLikelihood,
    LatentForceModel,
    LatentForcePrediction,
    StateSpaceLatentForceFit,
    StateSpaceLatentForceResult,
    fit_latent_force_model,
    fit_latent_force_state_space,
    latent_force_log_likelihood,
    latent_force_model,
    lfm_force_covariance,
    lfm_mean,
    lfm_output_covariance,
    mass_spring_damper_channel,
    matern_state_space,
    smooth_latent_force_state_space,
    state_space_latent_force_log_likelihood,
)
from .multifidelity import (
    RecursiveCoKriging,
    fit_recursive_cokriging,
    gaussian_process_from_hyperparameters,
)
from .nonlinear_multifidelity import (
    NargpPrediction,
    NargpValidation,
    NonlinearMultiFidelityGP,
    fit_nargp,
    nargp_kernel,
    validate_nargp,
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
from .physics_informed import (
    DeepONet,
    DeepONetDataset,
    DeepONetFit,
    HybridComparison,
    HybridODEResult,
    ModelScore,
    PINNCondition,
    PINNLossWeights,
    PINNResult,
    ResidualHybridModel,
    SparseDistillation,
    antiderivative_dataset,
    antiderivative_exact,
    compare_physics_ml_hybrid,
    deeponet_relative_error,
    distill_sparse_regression,
    fit_deeponet,
    fit_hybrid_ode,
    fit_pinn_1d,
    fit_residual_hybrid,
    network_jet,
    split_conformal_halfwidth,
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
from .reduced_order import (
    DMDResult,
    EDMDResult,
    EnsembleSINDyResult,
    FeatureDictionary,
    OperatorInferenceResult,
    PODBasis,
    ReducedModel,
    RegularizationSelection,
    ROMTrajectory,
    SINDyResult,
    custom_dictionary,
    eigen_general,
    finite_difference_derivatives,
    fit_dmd,
    fit_edmd,
    fit_ensemble_sindy,
    fit_operator_inference,
    fit_pod,
    fit_sindy,
    galerkin_reduced_model,
    local_polynomial_derivatives,
    polynomial_dictionary,
    quadratic_features,
    rbf_dictionary,
    select_operator_inference_regularization,
)
from .sparse_chaos import (
    SPARSE_CHAOS_METHODS,
    LeastAngleRegressionPath,
    fit_sparse_polynomial_chaos,
    least_angle_regression,
    sparse_polynomial_chaos_from_samples,
)
from .stochastic_kriging import (
    StochasticKriging,
    fit_stochastic_kriging,
    stochastic_kriging_from_summaries,
)

__all__ = [
    "ChannelSpec",
    "SyntheticChannel",
    "simulate_telemetry",
    "contaminate_values",
    "orbital_kernel",
    "NominalModel",
    "fit_nominal_model",
    "SlidingWindowGP",
    "batch_window_predict",
    "ScoreSeries",
    "sliding_window_scores",
    "calibrate_cusum_threshold",
    "FaultedSeries",
    "inject_fault",
    "DetectionStudy",
    "detection_study",
    "NargpPrediction",
    "NargpValidation",
    "NonlinearMultiFidelityGP",
    "fit_nargp",
    "nargp_kernel",
    "validate_nargp",
    "lfm_mean",
    "lfm_force_covariance",
    "lfm_output_covariance",
    "latent_force_log_likelihood",
    "LatentForceLikelihood",
    "LatentForceModel",
    "LatentForcePrediction",
    "latent_force_model",
    "fit_latent_force_model",
    "LatentForceChannel",
    "mass_spring_damper_channel",
    "matern_state_space",
    "state_space_latent_force_log_likelihood",
    "smooth_latent_force_state_space",
    "StateSpaceLatentForceResult",
    "fit_latent_force_state_space",
    "StateSpaceLatentForceFit",
    "BayesianGPPrediction",
    "BayesianGPRegression",
    "EmulatorValidation",
    "GPHyperPrior",
    "GPSobolResult",
    "LaplaceGPClassifier",
    "StudentTProcessRegression",
    "WarpedGPRegression",
    "box_cox",
    "default_hyper_prior",
    "fit_bayesian_gp",
    "fit_laplace_gp_classifier",
    "fit_student_t_process",
    "fit_warped_gp",
    "gp_predictive_moments",
    "gp_sobol_posterior",
    "inverse_box_cox",
    "sobol_estimates",
    "student_t_process_log_marginal_likelihood",
    "validate_emulator_moments",
    "validate_gp_emulator",
    "DMDResult",
    "EDMDResult",
    "EnsembleSINDyResult",
    "FeatureDictionary",
    "OperatorInferenceResult",
    "PODBasis",
    "ROMTrajectory",
    "ReducedModel",
    "RegularizationSelection",
    "SINDyResult",
    "custom_dictionary",
    "eigen_general",
    "finite_difference_derivatives",
    "fit_dmd",
    "fit_edmd",
    "fit_ensemble_sindy",
    "fit_operator_inference",
    "fit_pod",
    "fit_sindy",
    "galerkin_reduced_model",
    "local_polynomial_derivatives",
    "polynomial_dictionary",
    "quadratic_features",
    "rbf_dictionary",
    "select_operator_inference_regularization",
    "DeepONet",
    "DeepONetDataset",
    "DeepONetFit",
    "HybridComparison",
    "HybridODEResult",
    "ModelScore",
    "PINNCondition",
    "PINNLossWeights",
    "PINNResult",
    "ResidualHybridModel",
    "SparseDistillation",
    "antiderivative_dataset",
    "antiderivative_exact",
    "compare_physics_ml_hybrid",
    "deeponet_relative_error",
    "distill_sparse_regression",
    "fit_deeponet",
    "fit_hybrid_ode",
    "fit_pinn_1d",
    "fit_residual_hybrid",
    "network_jet",
    "split_conformal_halfwidth",
    "Kernel",
    "StationaryKernel",
    "PeriodicKernel",
    "RationalQuadraticKernel",
    "LinearKernel",
    "WhiteKernel",
    "SumKernel",
    "ProductKernel",
    "KernelGaussianProcess",
    "fit_kernel_gp",
    "DerivativeKernelBlocks",
    "derivative_kernel_blocks",
    "GradientPrediction",
    "DerivativeGaussianProcess",
    "fit_derivative_gp",
    "gaussian_process_gradient",
    "select_inducing_points",
    "SparseGaussianProcess",
    "fit_sparse_gp",
    "SparseGPCost",
    "sparse_gp_cost",
    "HeteroscedasticGaussianProcess",
    "fit_heteroscedastic_gp",
    "MultiOutputGaussianProcess",
    "fit_multi_output_gp",
    "LinearDifferentialOperator",
    "operator_covariances",
    "OperatorGaussianProcess",
    "fit_operator_gp",
    "BASES",
    "ChaosTrend",
    "KERNELS",
    "DeepEnsemble",
    "DeepEnsemblePrediction",
    "GaussianProcessConditioningWarning",
    "GaussianProcessFactory",
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
    "ProbabilisticSurrogate",
    "Surrogate",
    "SurrogateFactory",
    "SurrogateModel",
    "UniversalKriging",
    "fit_deep_ensemble",
    "fit_deep_ensemble_to_model",
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
    "BayesianQuadratureResult",
    "GaussianMeasure",
    "QuadratureMeasure",
    "UniformMeasure",
    "bayesian_quadrature",
    "kernel_double_integral",
    "kernel_mean",
    "RecursiveCoKriging",
    "fit_recursive_cokriging",
    "gaussian_process_from_hyperparameters",
    "StochasticKriging",
    "fit_stochastic_kriging",
    "stochastic_kriging_from_summaries",
]
