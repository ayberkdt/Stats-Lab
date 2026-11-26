"""Observation models and auditable inverse-problem calibration workflows.

``calibration`` holds the linear-Gaussian intercept-slope calibration with its
identifiability diagnostic. ``nonlinear`` declares a calibration problem on a
:class:`~probstat.modeling.ProbabilisticModel` whose joint input law is the
prior, with least squares, Fisher identifiability, the Laplace approximation,
and MCMC. ``discrepancy`` adds a Gaussian-process model discrepancy
(Kennedy-O'Hagan) and heteroscedastic noise, and ``prediction`` pushes
posterior draws through the forward model. ``regularization`` holds the
classical deterministic methods for ill-posed problems (Tikhonov, truncated
SVD, and the L-curve, GCV, and discrepancy parameter choices),
``test_problems`` discretized ill-posed problems with exact solutions, and
``ensemble_inversion`` the derivative-free ensemble Kalman inversion and
ES-MDA. ``interacting_langevin``, ``unscented_inversion`` and
``consensus_sampling`` hold three further derivative-free methods on the
same problem: an interacting Langevin sampler, a deterministic sigma-point
Gaussian iteration, and consensus-based sampling.

Main entry points:
    ``CalibrationProblem``: nonlinear calibration of model parameters from
    experiments, one ``NoiseModel`` per observed quantity.
    ``Experiment``: observed values with the conditions they were taken at.
    ``NoiseModel``: known or uncertain Gaussian or Student-t measurement
    noise, with an optional discrepancy variance and, for Gaussian noise,
    an optional ``ErrorCorrelation`` of the errors across experiments.
    ``predict``: posterior predictions under new conditions, latent or with
    measurement noise.
    ``DiscrepancyCalibration``: calibration with a Gaussian-process
    discrepancy over declared inputs.
    ``fisher_identifiability``: local identifiability from the prior-scaled
    Fisher information.
    ``calibrate_linear_gaussian``: conjugate posterior of an intercept and a
    slope under known noise.
    ``tikhonov``: Tikhonov solution by QR of ``[A; lambda L]``, with filter
    factors for ``L = I``; ``nonlinear_tikhonov`` for a nonlinear residual.
    ``truncated_svd`` and ``picard_coefficients``: SVD filtering and the
    discrete Picard condition.
    ``l_curve``, ``generalized_cross_validation``,
    ``discrepancy_principle``: choice of the Tikhonov parameter.
    ``backward_heat_problem``, ``second_derivative_problem``: test problems
    with exact solutions.
    ``ensemble_kalman_inversion``: derivative-free misfit minimization by
    iterated ensemble Kalman updates, with discrepancy-principle stopping.
    ``ensemble_smoother_mda``: ES-MDA, exact for linear-Gaussian problems as
    the ensemble grows; ``EnsembleInversionResult`` holds either run.
    ``interacting_langevin_sampler``: gradient-free affine-invariant
    Langevin ensemble, exact for linear-Gaussian problems at any ensemble
    size above ``d + 1`` (time-averaged, many forward runs).
    ``unscented_kalman_inversion``: Gaussian posterior approximation from
    ``2 d + 1`` sigma points per iteration, exact for linear problems.
    ``consensus_based_sampling``: weighted-ensemble sampler (Gaussian fixed
    point) or optimizer using only values of the posterior potential.
    Matrix-free and research-grade layer (experimental, 0.28; ADR-0144):
    ``LinearInverseProblem`` with ``solve_regularized`` (weighted, affine
    Tikhonov by QR, SVD, LSQR, LSMR or CGLS), ``linear_estimator_analysis``
    (resolution, influence, sampling covariance), constrained baselines
    (``equality_constrained_least_squares``, ``nonnegative_least_squares``,
    ``bounded_least_squares``, ``total_least_squares``);
    ``spectral_diagnostics`` and ``generalized_svd``; ``landweber``,
    ``krylov_regularization`` and ``hybrid_lsqr``; ``regularization_path``
    with ``select_regularization`` (GCV, UPRE, discrepancy, L-curve,
    evidence) and ``grouped_cross_validation``;
    ``l1_regularized_least_squares`` and
    ``total_variation_regularized_least_squares``; ``gauss_newton_krylov``
    and ``iteratively_regularized_gauss_newton`` on a ``DifferentiableForward``,
    ``newton_trust_region_inversion`` (Gauss-Newton or full Hessian actions),
    with ``check_derivatives``; ``linear_gaussian_posterior`` and
    ``low_rank_posterior``; ``EllipticDiffusion1D`` with tangent and adjoint
    actions and ``HeatEquation1D`` / ``HeatSource1D`` with exact discrete
    transposes; ``INVERSE_BENCHMARK_CATALOG`` with the benchmark builders
    (L01-L06) that withhold the truth and flag inverse crimes;
    ``profile_likelihood`` with ``profile_interval`` and
    ``efficient_information`` for practical identifiability; function-space
    bases (``SineBasis``, ``HatBasis``, ``PiecewiseConstantBasis``) with the
    truncation-consistent ``SpectralGaussianPrior`` and ``credible_bands``;
    ``WhitenedGaussianPriorTarget`` with ``sample_pcn`` / ``sample_hmc``
    adapters to the existing samplers and
    ``simulation_based_calibration_linear_gaussian``;
    ``ensemble_kalman_update`` (stochastic or square-root, observation or
    ensemble space) and ``tikhonov_ensemble_kalman_inversion`` with
    covariance factors; observation laws (``PoissonLaw``, ``BinomialLaw``,
    ``StudentTLaw``, ``CensoredGaussianLaw``, ``GaussianLaw``,
    ``MissingMask``) with ``maximum_likelihood_linear``.

Assumptions:
    The forward model is deterministic given its inputs. Measurement errors
    are independent across observations unless a ``NoiseModel`` declares an
    ``ErrorCorrelation`` (Gaussian, per quantity, across experiments);
    ``DiscrepancyCalibration`` adds a correlated model discrepancy, not
    correlated measurement errors. The prior is
    the model's joint input law. The Fisher diagnostic is local and assumes
    Gaussian errors; the Laplace approximation is a Gaussian at the mode.
    Predictions outside the calibration domain are conditional mathematics;
    the data cannot confirm the model form there.

Common pitfalls:
    - ``predict`` defaults to ``include_noise=False``: its intervals cover the
      latent mean response, not a future measurement.
    - ``CalibrationProblem.fit_least_squares`` ignores the prior; it is the
      frequentist baseline, not a MAP estimate.
    - ``Experiment`` conditions may not name a calibration parameter, and an
      observed quantity without a ``NoiseModel`` is rejected; both raise
      ``ValueError`` when the problem is built.
    - ``calibrate_linear_gaussian`` raises for an unidentifiable design but
      accepts a weakly conditioned one; the message is only in
      ``identifiability.warning``, and no Python warning is emitted.
    - ``sample_posterior`` and ``predict`` default to ``seed=None`` (fresh
      entropy); pass an integer seed to reproduce draws.
    - ``tikhonov`` takes ``lambda`` and penalizes ``lambda^2 ||L x||^2``: a
      ridge weight ``alpha`` corresponds to ``lambda = sqrt(alpha)``.
    - ``discrepancy_principle`` needs the noise NORM ``delta ~ s sqrt(m)``,
      not the noise standard deviation ``s``.
    - ``nonlinear_tikhonov`` centers the penalty at ``x0`` unless
      ``prior_mean`` is given.
    - ``ensemble_kalman_inversion`` is an optimizer: its ensemble collapses,
      so its spread is not a posterior uncertainty. Both ensemble methods
      accept known-scale Gaussian noise only and raise ``ValueError``
      otherwise.
    - ``interacting_langevin_sampler`` reports time averages; its final
      ensemble holds only ``J`` draws. With ``correction=False`` the
      covariance is biased low for a small ensemble.
    - ``solve_regularized`` centres the penalty at ``reference`` and whitens
      with ``noise``; its SVD route returns the minimum ``||x - x_ref||``
      solution, not the minimum ``||x||`` one. ``krylov_regularization`` and
      ``landweber`` regularize by stopping: without ``noise_norm`` they
      over-fit noisy data.
    - The ``l1``/TV coefficient ``alpha`` multiplies an UNSQUARED penalty,
      unlike the ``lambda^2`` of Tikhonov.
    - ``unscented_kalman_inversion`` and ``consensus_based_sampling`` return
      Gaussian-type approximations for nonlinear models; check
      ``converged`` and compare with ``sample_posterior`` when the
      posterior may be skewed or multimodal.

Guide: docs/foundations/inverse-problems-and-calibration.md
(deterministic regularization:
docs/foundations/regularization-and-ill-posed-problems.md)

References:
    [REF-TARANTOLA-2005], [REF-GELMAN-ETAL-2013], [REF-KENNEDY-OHAGAN-2001],
    [REF-BRYNJARSDOTTIR-OHAGAN-2014], [REF-HAARIO-SAKSMAN-TAMMINEN-2001],
    [REF-TIKHONOV-1963], [REF-HANSEN-2008], [REF-HANSEN-OLEARY-1993],
    [REF-IGLESIAS-LAW-STUART-2013], [REF-EMERICK-REYNOLDS-2013],
    [REF-GARBUNO-INIGO-NUSKEN-REICH-2020], [REF-HUANG-SCHNEIDER-STUART-2022],
    [REF-CARRILLO-ETAL-2022], [REF-PAIGE-SAUNDERS-1982],
    [REF-GAZZOLA-HANSEN-NAGY-2019], [REF-STUART-2010],
    [REF-BECK-TEBOULLE-2009], [REF-CHAMBOLLE-POCK-2011].

"""

from .approximation_error import (
    ApproximationErrorModel,
    estimate_approximation_error,
)
from .benchmark_extended import (
    EXTENDED_BENCHMARK_CATALOG,
    GaussianBlur2DOperator,
    NonlinearBenchmark,
    RayTransformOperator,
    elliptic_benchmark,
    exponential_decay_benchmark,
    heat_initial_condition_benchmark,
    image_deblur_benchmark,
    masked_tomography_benchmark,
    parallel_beam_rays,
    ray_pixel_intersections,
    squared_map_benchmark,
)
from .benchmark_problems import (
    INVERSE_BENCHMARK_CATALOG,
    BenchmarkCase,
    InverseBenchmark,
    correlated_gaussian_benchmark,
    deconvolution_benchmark,
    diagonal_spectrum_benchmark,
    sparse_sensing_benchmark,
)
from .calibration import (
    IdentifiabilityDiagnostic,
    LinearCalibrationResult,
    PredictionTarget,
    calibrate_linear_gaussian,
    diagnose_linear_identifiability,
)
from .consensus_sampling import ConsensusSamplingResult, consensus_based_sampling
from .discrepancy import (
    DISCREPANCY_KERNELS,
    DiscrepancyCalibration,
    DiscrepancyFit,
    DiscrepancyIdentifiability,
    DiscrepancyPosterior,
    DiscrepancyPrediction,
    DiscrepancyProcess,
    KnownNoise,
    LogLinearNoise,
    discrepancy_kernel,
)
from .ensemble_inversion import (
    EnsembleInversionResult,
    ensemble_kalman_inversion,
    ensemble_smoother_mda,
)
from .error_correlation import (
    ERROR_CORRELATION_KERNELS,
    ErrorCorrelation,
    error_correlation_kernel,
)
from .function_space import (
    CredibleBands,
    HatBasis,
    PiecewiseConstantBasis,
    SineBasis,
    SpectralGaussianPrior,
    credible_bands,
)
from .gauss_newton import (
    CallbackForward,
    DerivativeCheck,
    DifferentiableForward,
    GaussNewtonIteration,
    GaussNewtonResult,
    IRGNMIteration,
    IRGNMResult,
    ModelCounts,
    check_derivatives,
    gauss_newton_krylov,
    iteratively_regularized_gauss_newton,
)
from .heat_models import (
    HeatEquation1D,
    HeatSolveCounts,
    HeatSource1D,
    heat_amplification,
    heat_semigroup_singular_values,
)
from .interacting_langevin import (
    InteractingLangevinResult,
    interacting_langevin_sampler,
)
from .iterative import (
    IterativeRegularizationResult,
    expected_noise_norm,
    hybrid_lsqr,
    krylov_regularization,
    landweber,
)
from .linear_gaussian import (
    GaussianLinearPosterior,
    LowRankGaussianPosterior,
    PredictiveDistribution,
    PriorWhitenedInformation,
    linear_gaussian_posterior,
    low_rank_posterior,
    prior_whitened_information,
)
from .linear_problem import (
    ConstrainedSolution,
    LinearEstimatorAnalysis,
    LinearInverseProblem,
    LinearInverseSolution,
    TotalLeastSquaresSolution,
    bounded_least_squares,
    equality_constrained_least_squares,
    linear_estimator_analysis,
    nonnegative_least_squares,
    solve_regularized,
    total_least_squares,
)
from .newton_trust_region import (
    NewtonTrustRegionCounts,
    NewtonTrustRegionResult,
    newton_trust_region_inversion,
)
from .nonlinear import (
    CALIBRATION_GRADIENTS,
    CalibrationPosterior,
    CalibrationProblem,
    Experiment,
    FisherIdentifiability,
    LeastSquaresCalibration,
    NoiseModel,
    fisher_identifiability,
)
from .observation import ForwardResponse, GaussianObservationModel, ParameterVector
from .observation_laws import (
    BinomialLaw,
    CensoredGaussianLaw,
    GaussianLaw,
    LikelihoodIteration,
    MaximumLikelihoodResult,
    MissingMask,
    ObservationLaw,
    PoissonLaw,
    StudentTLaw,
    maximum_likelihood_linear,
)
from .parameter_selection import (
    CrossValidationPath,
    RegularizationPath,
    RegularizationSelection,
    grouped_cross_validation,
    logarithmic_grid,
    regularization_path,
    select_regularization,
)
from .pde_models import (
    AdjointGradient,
    EllipticDiffusion1D,
    FluxBalance,
    PDESolveCounts,
    TridiagonalFactorization,
    adjoint_gradient,
    solve_tridiagonal,
    solve_tridiagonal_transpose,
)
from .posterior_targets import (
    SBC_CAVEAT,
    LinearGaussianSBCResult,
    TargetCounts,
    WhitenedGaussianPriorTarget,
    physical_posterior,
    sample_hmc,
    sample_pcn,
    simulation_based_calibration_linear_gaussian,
)
from .prediction import PosteriorPrediction, PredictedQuantity, predict
from .profile_likelihood import (
    EstimabilityResult,
    ProfileInterval,
    ProfileLikelihoodResult,
    ProfilePoint,
    delta_method_variance,
    efficient_information,
    functional_estimability,
    profile_interval,
    profile_likelihood,
)
from .regularization import (
    NonlinearTikhonovResult,
    PicardCoefficients,
    RegularizationParameterChoice,
    TikhonovSolution,
    TruncatedSVDSolution,
    discrepancy_principle,
    generalized_cross_validation,
    l_curve,
    nonlinear_tikhonov,
    picard_coefficients,
    tikhonov,
    truncated_svd,
)
from .regularized_ensemble import (
    EnsembleKalmanUpdate,
    TikhonovEnsembleResult,
    ensemble_kalman_update,
    tikhonov_ensemble_kalman_inversion,
)
from .sparse import (
    L1Result,
    TotalVariationResult,
    frobenius_norm_bound,
    l1_regularized_least_squares,
    lasso_zero_threshold,
    total_variation_regularized_least_squares,
    total_variation_value,
)
from .spectral import (
    GeneralizedSVD,
    SpectralDiagnostics,
    generalized_svd,
    gsvd_tikhonov,
    spectral_diagnostics,
)
from .test_problems import (
    DiscreteIllPosedProblem,
    backward_heat_problem,
    second_derivative_problem,
)
from .unscented_inversion import UnscentedInversionResult, unscented_kalman_inversion

__all__ = [
    "EXTENDED_BENCHMARK_CATALOG",
    "GaussianBlur2DOperator",
    "NonlinearBenchmark",
    "RayTransformOperator",
    "elliptic_benchmark",
    "exponential_decay_benchmark",
    "heat_initial_condition_benchmark",
    "image_deblur_benchmark",
    "masked_tomography_benchmark",
    "parallel_beam_rays",
    "ray_pixel_intersections",
    "squared_map_benchmark",
    "ApproximationErrorModel",
    "estimate_approximation_error",
    "BinomialLaw",
    "CensoredGaussianLaw",
    "GaussianLaw",
    "LikelihoodIteration",
    "MaximumLikelihoodResult",
    "MissingMask",
    "ObservationLaw",
    "PoissonLaw",
    "StudentTLaw",
    "maximum_likelihood_linear",
    "EnsembleKalmanUpdate",
    "TikhonovEnsembleResult",
    "ensemble_kalman_update",
    "tikhonov_ensemble_kalman_inversion",
    "SBC_CAVEAT",
    "LinearGaussianSBCResult",
    "TargetCounts",
    "WhitenedGaussianPriorTarget",
    "physical_posterior",
    "sample_hmc",
    "sample_pcn",
    "simulation_based_calibration_linear_gaussian",
    "NewtonTrustRegionCounts",
    "NewtonTrustRegionResult",
    "newton_trust_region_inversion",
    "HeatEquation1D",
    "HeatSolveCounts",
    "HeatSource1D",
    "heat_amplification",
    "heat_semigroup_singular_values",
    "CredibleBands",
    "HatBasis",
    "PiecewiseConstantBasis",
    "SineBasis",
    "SpectralGaussianPrior",
    "credible_bands",
    "EstimabilityResult",
    "ProfileInterval",
    "ProfileLikelihoodResult",
    "ProfilePoint",
    "delta_method_variance",
    "efficient_information",
    "functional_estimability",
    "profile_interval",
    "profile_likelihood",
    "INVERSE_BENCHMARK_CATALOG",
    "BenchmarkCase",
    "InverseBenchmark",
    "correlated_gaussian_benchmark",
    "deconvolution_benchmark",
    "diagonal_spectrum_benchmark",
    "sparse_sensing_benchmark",
    "CallbackForward",
    "DerivativeCheck",
    "DifferentiableForward",
    "GaussNewtonIteration",
    "GaussNewtonResult",
    "IRGNMIteration",
    "IRGNMResult",
    "ModelCounts",
    "check_derivatives",
    "gauss_newton_krylov",
    "iteratively_regularized_gauss_newton",
    "IterativeRegularizationResult",
    "expected_noise_norm",
    "hybrid_lsqr",
    "krylov_regularization",
    "landweber",
    "GaussianLinearPosterior",
    "LowRankGaussianPosterior",
    "PredictiveDistribution",
    "PriorWhitenedInformation",
    "linear_gaussian_posterior",
    "low_rank_posterior",
    "prior_whitened_information",
    "ConstrainedSolution",
    "LinearEstimatorAnalysis",
    "LinearInverseProblem",
    "LinearInverseSolution",
    "TotalLeastSquaresSolution",
    "bounded_least_squares",
    "equality_constrained_least_squares",
    "linear_estimator_analysis",
    "nonnegative_least_squares",
    "solve_regularized",
    "total_least_squares",
    "CrossValidationPath",
    "RegularizationPath",
    "RegularizationSelection",
    "grouped_cross_validation",
    "logarithmic_grid",
    "regularization_path",
    "select_regularization",
    "AdjointGradient",
    "EllipticDiffusion1D",
    "FluxBalance",
    "PDESolveCounts",
    "TridiagonalFactorization",
    "adjoint_gradient",
    "solve_tridiagonal",
    "solve_tridiagonal_transpose",
    "L1Result",
    "TotalVariationResult",
    "frobenius_norm_bound",
    "l1_regularized_least_squares",
    "lasso_zero_threshold",
    "total_variation_regularized_least_squares",
    "total_variation_value",
    "GeneralizedSVD",
    "SpectralDiagnostics",
    "generalized_svd",
    "gsvd_tikhonov",
    "spectral_diagnostics",
    "DISCREPANCY_KERNELS",
    "DiscrepancyCalibration",
    "DiscrepancyFit",
    "DiscrepancyIdentifiability",
    "DiscrepancyPosterior",
    "DiscrepancyPrediction",
    "DiscrepancyProcess",
    "KnownNoise",
    "LogLinearNoise",
    "discrepancy_kernel",
    "EnsembleInversionResult",
    "ensemble_kalman_inversion",
    "ensemble_smoother_mda",
    "ConsensusSamplingResult",
    "InteractingLangevinResult",
    "UnscentedInversionResult",
    "consensus_based_sampling",
    "interacting_langevin_sampler",
    "unscented_kalman_inversion",
    "CALIBRATION_GRADIENTS",
    "ERROR_CORRELATION_KERNELS",
    "CalibrationPosterior",
    "CalibrationProblem",
    "ErrorCorrelation",
    "Experiment",
    "FisherIdentifiability",
    "ForwardResponse",
    "GaussianObservationModel",
    "IdentifiabilityDiagnostic",
    "LeastSquaresCalibration",
    "LinearCalibrationResult",
    "NoiseModel",
    "ParameterVector",
    "PosteriorPrediction",
    "PredictedQuantity",
    "PredictionTarget",
    "calibrate_linear_gaussian",
    "diagnose_linear_identifiability",
    "error_correlation_kernel",
    "fisher_identifiability",
    "predict",
    "DiscreteIllPosedProblem",
    "NonlinearTikhonovResult",
    "PicardCoefficients",
    "RegularizationParameterChoice",
    "TikhonovSolution",
    "TruncatedSVDSolution",
    "backward_heat_problem",
    "discrepancy_principle",
    "generalized_cross_validation",
    "l_curve",
    "nonlinear_tikhonov",
    "picard_coefficients",
    "second_derivative_problem",
    "tikhonov",
    "truncated_svd",
]
