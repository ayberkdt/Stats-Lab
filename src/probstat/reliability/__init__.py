"""Reliability, failure-probability, FORM/SORM, rare-event, and system analysis.

``analysis`` holds the original vector-level estimators. ``model_based``,
``simulation``, ``adaptive``, and ``systems`` work on a
:class:`~probstat.modeling.ProbabilisticModel` through
:class:`ModelLimitState`, sharing one joint input law across FORM, SORM,
importance sampling, subset simulation, cross-entropy importance sampling,
line sampling, system reliability, and inverse design. ``design_points``
finds multiple design points and combines them (multi-point FORM, mixture
importance sampling); ``active_learning`` implements AK-MCS on a Gaussian
process surrogate from :mod:`probstat.surrogates` (or any probabilistic
surrogate through a refit hook), with the learning functions and stopping
rules of ``learning`` and the candidate-population posteriors of
``population_posterior``; ``ak_subset`` combines the same learning with
subset simulation on the surrogate (AK-SS). ``data`` estimates
component parameters from test and field data (Beta/Gamma posteriors,
chi-square bounds, demonstration tests, Crow-AMSAA growth) and
``allocation`` apportions a system reliability target to units, including
cost-optimal allocation. ``particles`` implements moving particles (the
last-particle multilevel splitting algorithm), ``directional`` model-based
directional simulation with several roots per ray, and ``splitting`` the
coefficient of variation and optimal level count of ideal splitting.
``certification`` decides whether a simulated
failure probability is below an acceptable risk with a bounded
false-certification probability (crude Monte Carlo and moving particles).
``improved_cross_entropy`` and ``sequential_importance`` replace the failure
indicator by ``Phi(-g / sigma)`` with a decreasing ``sigma`` (a fitted
Gaussian or Gaussian-mixture proposal, or resampling with Markov-chain
moves), and ``bayesian_updating`` turns Bayesian updating into a rare-event
problem (BUS) solved by subset simulation.

Main entry points:
    ``ModelLimitState``: one model output and a threshold as a limit state
    whose failure set is ``g <= 0``.
    ``form_hlrf``: FORM design point and signed reliability index by the
    improved HL-RF iteration.
    ``sorm``: Breitung and Hohenbichler-Rackwitz corrections at a converged
    FORM design point.
    ``subset_simulation``: small failure probabilities without a design point,
    with modified Metropolis, conditional-sampling, adaptive conditional-sampling,
    or Hamiltonian Monte Carlo chains (``SUBSET_SAMPLERS``).
    ``moving_particles``: last-particle splitting with an exact Poisson
    interval.
    ``directional_simulation``: radial chi integration along random
    directions, every boundary crossing of each ray included.
    ``optimal_splitting``: number of splitting levels that minimizes the
    c.o.v. for a fixed budget (``splitting_coefficient_of_variation`` for a
    given level count).
    ``compare_reliability_methods``: FORM, SORM, importance sampling, subset
    simulation, and crude Monte Carlo on one limit state.
    ``find_design_points``: several design points of one limit state.
    ``ak_mcs``: Gaussian-process active learning for expensive models
    (learning functions ``U``, ``EFF``, ``H``; learning or bound stopping).
    ``ak_ss``: the same active learning on the levels of a subset simulation
    run on the surrogate, for very small failure probabilities.
    ``estimate_failure_probability``: direct Monte Carlo on explicit vectors.
    ``improved_cross_entropy``: importance sampling with a Gaussian or
    Gaussian-mixture proposal adapted through smooth indicator levels.
    ``sequential_importance_sampling``: the same smooth levels sampled by
    resampling and conditional-sampling Markov chains.
    ``bus_subset_simulation``: posterior draws and model evidence from the
    acceptance event ``p <= c L(theta)`` (``bus_limit_state`` gives the
    limit state for any other reliability method).
    ``scenario_sample_size``, ``scenario_violation_bound``,
    ``scenario_posterior_violation_bound``: scenario-approach guarantees for
    convex programs with uncertain constraints (a priori and from the number
    of support constraints).
    ``certify_design``: exact Clopper-Pearson upper bound on the violation
    probability of any design from a fresh validation sample.
    ``solve_chance_constrained``: sample-average chance-constrained
    minimisation with an out-of-sample certificate;
    ``normal_linear_chance_margin`` and ``distribution_free_chance_margin``
    are the closed-form deterministic equivalents.
    ``rbdo_sora``: decoupled reliability-based design optimisation by
    sequential optimisation and reliability assessment (``rbdo_double_loop``
    is the nested reference, ``minimize_constrained`` the augmented-Lagrangian
    solver, ``verify_design_reliability`` the SORM or importance-sampling
    re-check).

Assumptions:
    Failure is the closed set ``g <= 0``: a sample exactly on the boundary
    fails. Model-based methods work in independent standard-normal space
    through the model's joint input law; FORM, SORM, and the design-point
    methods need continuous inputs. ``Phi(-beta)`` is exact only for a limit
    state that is linear in that space. SORM needs ``beta > 0`` and
    ``1 + beta kappa_i > 0`` for every principal curvature. Sampling
    estimators assume independent draws; the reported coefficient of
    variation is a Monte Carlo error, not a model-form error.

Common pitfalls:
    - ``ModelLimitState`` defaults to ``failure="above"`` (failure when
      ``y >= t``). A safety margin needs ``failure="below"``; the wrong side
      still converges and reports ``beta`` with the opposite sign and
      ``P_f`` close to one.
    - ``form_hlrf`` does not raise when ``max_iterations`` is reached: it
      returns ``converged=False``, and ``sorm`` then refuses the result. Check
      ``converged`` before reading ``beta``.
    - With correlated inputs ``importance_factors`` (``alpha_i^2``) depend on
      the input order; ``gamma`` refers to the physical inputs and does not.
    - ``seed=None`` draws fresh entropy, so repeated runs differ; pass an
      integer seed. ``compare_reliability_methods`` runs ``METHODS``, which
      leaves out ``"ce"`` and ``"line"`` (both are in ``AVAILABLE_METHODS``).
    - Zero observed failures give ``standard_error == 0``; the Wilson
      interval keeps a positive upper bound, which is the number to report.
    - ``improved_cross_entropy`` with one Gaussian follows a single failure
      region; several regions need ``family="mixture"``. The c.o.v. of
      ``sequential_importance_sampling`` is a heuristic that understates the
      scatter between runs.
    - BUS draws are exact only if ``c L <= 1`` everywhere; check
      ``constant_violated``, which covers the evaluated points only.

Guide: docs/foundations/advanced-reliability.md

References:
    [REF-MELCHERS-BECK-2018], [REF-HASOFER-LIND-1974],
    [REF-DERKIUREGHIAN-2005], [REF-BREITUNG-1984], [REF-AU-BECK-2001],
    [REF-WILSON-1927], [REF-GUYADER-HENGARTNER-MATZNER-LOBER-2011],
    [REF-BJERAGER-1988], [REF-LAGNOUX-2006],
    [REF-PAPAIOANNOU-GEYER-STRAUB-2019],
    [REF-PAPAIOANNOU-PAPADIMITRIOU-STRAUB-2016], [REF-STRAUB-PAPAIOANNOU-2015],
    [REF-CAMPI-GARATTI-2008], [REF-CAMPI-GARATTI-2018], [REF-DU-CHEN-2004].

"""

from .active_learning import (
    AKMCS_DESIGNS,
    AKMCS_GP_REGULARIZATION,
    AKMCSIteration,
    AKMCSResult,
    ak_mcs,
)
from .adaptive import (
    CE_COVARIANCE_FLOOR,
    CE_MAX_UNCOVERED_ELITE_FRACTION,
    CE_MIN_FAILURE_ESS_FRACTION,
    CE_MIN_MEAN_NORM,
    CrossEntropyResult,
    LineSamplingResult,
    cross_entropy_importance_sampling,
    line_sampling,
)
from .ak_subset import AKSS_GP_REGULARIZATION, AKSSResult, AKSSRound, ak_ss
from .allocation import (
    AllocationResult,
    MettasCost,
    agree_allocation,
    arinc_allocation,
    cost_optimal_allocation,
    cost_optimal_series_allocation,
    equal_apportionment,
    feasibility_of_objectives,
)
from .analysis import (
    DirectionalSamplingResult,
    DirectReliabilityResult,
    LimitState,
    LinearNormalFORMResult,
    WeightedFailureResult,
    estimate_directional_failure_probability,
    estimate_failure_probability,
    estimate_weighted_failure_probability,
    linear_normal_form,
)
from .bayesian_updating import (
    BUS_MARGIN,
    BUS_SAMPLERS,
    BUS_SCALED_LIKELIHOOD,
    BUS_VIOLATION_TOLERANCE,
    BUSResult,
    bus_limit_state,
    bus_subset_simulation,
)
from .certification import (
    CERTIFICATION_MAX_SIZE,
    CERTIFICATION_RULES,
    CERTIFICATION_SCAN_PLATEAUS,
    CertificationProbability,
    CertificationSampleSize,
    MovingParticlesCertification,
    certification_margin,
    certification_probability,
    certification_sample_size,
    moving_particles_certification_probability,
)
from .chance_constraints import (
    DISTRIBUTION_FREE_INEQUALITIES,
    ChanceConstrainedResult,
    ScenarioSampleSize,
    ViolationCertificate,
    certify_design,
    certify_violation_frequency,
    distribution_free_chance_margin,
    normal_linear_chance_margin,
    normal_linear_violation_probability,
    scenario_confidence,
    scenario_posterior_violation_bound,
    scenario_sample_size,
    scenario_violation_bound,
    solve_chance_constrained,
)
from .data import (
    SIDES,
    TERMINATIONS,
    CrowAMSAAFit,
    DemandFailureEstimate,
    FailureRateBounds,
    FailureRateEstimate,
    crow_amsaa_fit,
    demand_failure_probability,
    demonstration_test_time,
    failure_rate_bounds,
    failure_rate_estimate,
    success_run_sample_size,
)
from .design_optimization import (
    VERIFICATION_METHODS,
    ConstrainedResult,
    DesignVerification,
    RBDOIteration,
    RBDOResult,
    minimize_constrained,
    rbdo_double_loop,
    rbdo_sora,
    verify_design_reliability,
)
from .design_points import (
    MixtureImportanceResult,
    MultipleDesignPointsResult,
    MultiPointFORMResult,
    bulge_value,
    find_design_points,
    mixture_importance_sampling,
    multi_point_form,
)
from .directional import (
    RADIUS_TAIL_PROBABILITY,
    DirectionalSimulationResult,
    default_radius_max,
    directional_simulation,
)
from .improved_cross_entropy import (
    ICE_COVARIANCE_FLOOR,
    ICE_FAMILIES,
    ICE_TARGET_CV,
    ImprovedCrossEntropyResult,
    improved_cross_entropy,
    smoothing_scale,
)
from .learning import (
    BOUNDS_QUANTILE,
    LEARNING_FUNCTIONS,
    LEARNING_THRESHOLDS,
    STOPPING_RULES,
    bounds_satisfied,
    learning_value,
    probability_bounds,
    select_learning_point,
    stopping_satisfied,
)
from .model_based import (
    DesignParameterResult,
    FORMResult,
    InverseFORMResult,
    ModelLimitState,
    SORMResult,
    correlated_importance,
    design_for_target_beta,
    form_hlrf,
    generalized_reliability_index,
    inverse_form,
    principal_curvatures,
    second_order_probabilities,
    sorm,
)
from .particles import (
    MovingParticlesResult,
    moving_particles,
    moving_particles_interval,
)
from .population_posterior import (
    GaussianProcessPopulationPosterior,
    PopulationPosterior,
    SurrogatePopulationPosterior,
)
from .sequential_importance import (
    SequentialImportanceResult,
    sequential_importance_sampling,
)
from .simulation import (
    AVAILABLE_METHODS,
    METHODS,
    OPTIONAL_METHODS,
    SUBSET_SAMPLERS,
    DesignPointImportanceResult,
    MethodComparison,
    MethodComparisonRow,
    ModelMonteCarloResult,
    SubsetSimulationResult,
    compare_reliability_methods,
    design_point_importance_sampling,
    model_monte_carlo,
    subset_simulation,
)
from .splitting import (
    OPTIMAL_LEVEL_EXPONENT,
    OPTIMAL_LEVEL_PROBABILITY,
    OPTIMAL_LEVELS_PER_LOG,
    OptimalSplittingResult,
    moving_particles_coefficient_of_variation,
    optimal_splitting,
    splitting_coefficient_of_variation,
)
from .systems import (
    SeriesSystemBounds,
    SystemReliabilityResult,
    bivariate_normal_cdf,
    gauss_legendre,
    series_system_bounds,
    system_failure_probability,
)

__all__ = [
    "AKMCSIteration",
    "AKMCSResult",
    "AKMCS_GP_REGULARIZATION",
    "AKSSResult",
    "AKSSRound",
    "AKSS_GP_REGULARIZATION",
    "AVAILABLE_METHODS",
    "AllocationResult",
    "CERTIFICATION_MAX_SIZE",
    "CERTIFICATION_RULES",
    "CERTIFICATION_SCAN_PLATEAUS",
    "CE_COVARIANCE_FLOOR",
    "CE_MAX_UNCOVERED_ELITE_FRACTION",
    "CE_MIN_FAILURE_ESS_FRACTION",
    "CE_MIN_MEAN_NORM",
    "CertificationProbability",
    "CertificationSampleSize",
    "CrossEntropyResult",
    "CrowAMSAAFit",
    "DemandFailureEstimate",
    "DesignParameterResult",
    "DesignPointImportanceResult",
    "DirectReliabilityResult",
    "DirectionalSamplingResult",
    "DirectionalSimulationResult",
    "FORMResult",
    "FailureRateBounds",
    "FailureRateEstimate",
    "InverseFORMResult",
    "LimitState",
    "LineSamplingResult",
    "LinearNormalFORMResult",
    "METHODS",
    "MethodComparison",
    "MethodComparisonRow",
    "MettasCost",
    "MixtureImportanceResult",
    "ModelLimitState",
    "ModelMonteCarloResult",
    "MovingParticlesResult",
    "MovingParticlesCertification",
    "MultiPointFORMResult",
    "MultipleDesignPointsResult",
    "OPTIMAL_LEVELS_PER_LOG",
    "OPTIMAL_LEVEL_EXPONENT",
    "OPTIMAL_LEVEL_PROBABILITY",
    "OPTIONAL_METHODS",
    "OptimalSplittingResult",
    "RADIUS_TAIL_PROBABILITY",
    "SIDES",
    "SORMResult",
    "SeriesSystemBounds",
    "SUBSET_SAMPLERS",
    "SubsetSimulationResult",
    "SystemReliabilityResult",
    "TERMINATIONS",
    "WeightedFailureResult",
    "agree_allocation",
    "ak_mcs",
    "ak_ss",
    "arinc_allocation",
    "bivariate_normal_cdf",
    "bulge_value",
    "certification_margin",
    "certification_probability",
    "certification_sample_size",
    "compare_reliability_methods",
    "correlated_importance",
    "cost_optimal_allocation",
    "cost_optimal_series_allocation",
    "cross_entropy_importance_sampling",
    "crow_amsaa_fit",
    "default_radius_max",
    "demand_failure_probability",
    "demonstration_test_time",
    "design_for_target_beta",
    "design_point_importance_sampling",
    "directional_simulation",
    "equal_apportionment",
    "estimate_directional_failure_probability",
    "estimate_failure_probability",
    "estimate_weighted_failure_probability",
    "failure_rate_bounds",
    "failure_rate_estimate",
    "feasibility_of_objectives",
    "find_design_points",
    "form_hlrf",
    "gauss_legendre",
    "generalized_reliability_index",
    "inverse_form",
    "line_sampling",
    "linear_normal_form",
    "mixture_importance_sampling",
    "model_monte_carlo",
    "moving_particles",
    "moving_particles_coefficient_of_variation",
    "moving_particles_interval",
    "moving_particles_certification_probability",
    "multi_point_form",
    "optimal_splitting",
    "principal_curvatures",
    "second_order_probabilities",
    "series_system_bounds",
    "sorm",
    "splitting_coefficient_of_variation",
    "subset_simulation",
    "success_run_sample_size",
    "system_failure_probability",
    "AKMCS_DESIGNS",
    "BOUNDS_QUANTILE",
    "LEARNING_FUNCTIONS",
    "LEARNING_THRESHOLDS",
    "STOPPING_RULES",
    "bounds_satisfied",
    "learning_value",
    "probability_bounds",
    "select_learning_point",
    "stopping_satisfied",
    "GaussianProcessPopulationPosterior",
    "PopulationPosterior",
    "SurrogatePopulationPosterior",
    "BUSResult",
    "BUS_MARGIN",
    "BUS_SAMPLERS",
    "BUS_SCALED_LIKELIHOOD",
    "BUS_VIOLATION_TOLERANCE",
    "ICE_COVARIANCE_FLOOR",
    "ICE_FAMILIES",
    "ICE_TARGET_CV",
    "ImprovedCrossEntropyResult",
    "SequentialImportanceResult",
    "bus_limit_state",
    "bus_subset_simulation",
    "improved_cross_entropy",
    "sequential_importance_sampling",
    "smoothing_scale",
    "DISTRIBUTION_FREE_INEQUALITIES",
    "ChanceConstrainedResult",
    "ScenarioSampleSize",
    "ViolationCertificate",
    "certify_design",
    "certify_violation_frequency",
    "distribution_free_chance_margin",
    "normal_linear_chance_margin",
    "normal_linear_violation_probability",
    "scenario_confidence",
    "scenario_posterior_violation_bound",
    "scenario_sample_size",
    "scenario_violation_bound",
    "solve_chance_constrained",
    "VERIFICATION_METHODS",
    "ConstrainedResult",
    "DesignVerification",
    "RBDOIteration",
    "RBDOResult",
    "minimize_constrained",
    "rbdo_double_loop",
    "rbdo_sora",
    "verify_design_reliability",
]
