"""Monte Carlo estimators, cost accounting, MLMC, multifidelity, Stein methods.

Main entry points:
    ``estimate_mean``: sample mean with unbiased variance, standard error,
        and a normal-approximation interval.
    ``estimate_probability``: event probability from exact Boolean
        indicators with a Wilson interval.
    ``estimate_importance_sampling``: ordinary importance-sampling estimate
        with the effective-sample-size diagnostic.
    ``compare_efficiency``: cost-normalized comparison
        ``SE^2 * evaluations`` of two estimators.
    ``adaptive_multilevel_monte_carlo``: Giles' MLMC to a target root mean
        square error.
    ``select_mfmc_models`` and ``mfmc_estimate``: multifidelity Monte Carlo
        with the closed-form optimal allocation for a cost budget.
    ``mlblue_allocation`` and ``mlblue_estimate``: multilevel best linear
        unbiased estimator over model groups with a certified allocation.
    ``compare_multifidelity_estimators``: predicted variances of plain MC,
        MFMC, ACV-MF, and MLBLUE for one covariance, cost vector, and budget.
    ``estimate_zero_variance``: zero-variance (Stein) control variates from
        the score of the target density.
    ``kernel_stein_discrepancy``: sample-quality diagnostic against a target
        known up to a constant.

Assumptions:
    Contributions are independent and identically distributed with finite
    variance unless an estimator says otherwise (the MCMC standard error of
    the Stein estimators); intervals are large-sample normal approximations.
    Importance sampling needs normalized target and proposal densities and a
    proposal that is positive wherever the target integrand contributes.
    Costs are the caller's evaluation counts, not wall time. The
    multifidelity estimators take the model covariance as an input that is
    independent of the estimation samples; their reported variances are
    predictions from that covariance.

Common pitfalls:
    - ``estimate_mean`` reports a normal (z) interval even for a handful of
      values: for ``(1, 2, 3, 4)`` it is ``2.5 +/- 1.96 * 0.6455``, narrower
      than the Student-t interval, so it under-covers for small samples.
    - ``estimate_probability`` with no observed events has standard error
      0.0 but a Wilson upper bound of 0.037 for 100 trials; report the
      interval, not the plug-in standard error. Indicators must be exact
      Booleans (``1`` and ``0`` raise ``TypeError``).
    - The sampling estimators ``estimate_uniform_integral``,
      ``estimate_antithetic_uniform``, ``estimate_stratified_uniform``, and
      ``estimate_importance_sampling`` take ``seed: int | None`` and seed a
      call-local ``random.Random``; the default ``None`` is not reproducible.
    - ``estimate_antithetic_uniform`` charges two evaluations per pair:
      ``sample_size`` is the number of pairs and ``evaluations`` twice that.
      Compare methods with ``compare_efficiency``, which multiplies the
      squared standard error by ``evaluations``, never by ``sample_size``.
    - ``estimate_importance_sampling`` is the ordinary (not self-normalized)
      estimator, so unnormalized densities bias it; the effective sample size
      describes weight concentration and is not a sample count.
    - A pilot covariance used for multifidelity weights must come from
      samples that are not reused in the estimator, otherwise the estimate
      is biased; ``variance`` of a ``MultifidelityEstimate`` is a prediction
      from the supplied covariance, not a sample variance.

Guide: docs/foundations/advanced-monte-carlo.md

References:
    [REF-ROBERT-CASELLA-2004, Chapters 3-4], [REF-OWEN-2013, Chapters 2, 8,
    and 9], [REF-WILSON-1927], [REF-GILES-2008], [REF-GILES-2015],
    [REF-MIRA-SOLGI-IMPARATO-2013], [REF-GORHAM-MACKEY-2017],
    [REF-PEHERSTORFER-WILLCOX-GUNZBURGER-2016],
    [REF-GORODETSKY-GERACI-ELDRED-JAKEMAN-2020], [REF-SCHADEN-ULLMANN-2020].

"""

from .approximate_control_variates import (
    ACVAllocation,
    ACVDesign,
    acv_allocation,
    acv_design,
    acv_estimate,
)
from .estimation import (
    MonteCarloEstimate,
    ProbabilityEstimate,
    convergence_trace,
    estimate_mean,
    estimate_mean_streaming,
    estimate_probability,
    estimate_uniform_integral,
)
from .limit_simulation import (
    CLTSimulation,
    RunningMeanPaths,
    WeakLawPoint,
    simulate_clt_distance,
    simulate_running_means,
    simulate_weak_law,
)
from .multifidelity import (
    InputSampler,
    MFMCAllocation,
    ModelFunction,
    MultifidelityEstimate,
    PilotStatistics,
    mfmc_admissible,
    mfmc_allocation,
    mfmc_estimate,
    mfmc_variance,
    pilot_statistics,
    select_mfmc_models,
)
from .multilevel import (
    LevelSampler,
    LevelStatistics,
    MultilevelResult,
    adaptive_multilevel_monte_carlo,
    convergence_rates,
    multilevel_monte_carlo,
    optimal_level_allocation,
)
from .multilevel_blue import (
    MLBLUEAllocation,
    MultifidelityComparison,
    MultifidelityComparisonRow,
    compare_multifidelity_estimators,
    mlblue_allocation,
    mlblue_estimate,
    mlblue_groups,
    mlblue_variance,
)
from .stein import (
    ControlFunctionalResult,
    ControlFunctionalWeightsResult,
    KernelSteinDiscrepancy,
    ZeroVarianceResult,
    estimate_control_functional,
    estimate_control_functional_weights,
    estimate_zero_variance,
    kernel_stein_discrepancy,
    median_heuristic_length_scale,
    stein_kernel,
    stein_polynomial_controls,
)
from .variance_reduction import (
    ControlVariateResult,
    CostedEstimate,
    EfficiencyComparison,
    ImportanceSamplingResult,
    StratifiedEstimate,
    allocate_stratified_budget,
    compare_efficiency,
    estimate_antithetic_uniform,
    estimate_conditional_expectation,
    estimate_control_variate,
    estimate_importance_sampling,
    estimate_stratified_uniform,
)

__all__ = [
    "MonteCarloEstimate",
    "ProbabilityEstimate",
    "ControlVariateResult",
    "CostedEstimate",
    "EfficiencyComparison",
    "ImportanceSamplingResult",
    "LevelSampler",
    "LevelStatistics",
    "MultilevelResult",
    "adaptive_multilevel_monte_carlo",
    "convergence_rates",
    "multilevel_monte_carlo",
    "optimal_level_allocation",
    "ACVAllocation",
    "ACVDesign",
    "InputSampler",
    "MFMCAllocation",
    "MLBLUEAllocation",
    "ModelFunction",
    "MultifidelityComparison",
    "MultifidelityComparisonRow",
    "MultifidelityEstimate",
    "PilotStatistics",
    "acv_allocation",
    "acv_design",
    "acv_estimate",
    "compare_multifidelity_estimators",
    "mfmc_admissible",
    "mfmc_allocation",
    "mfmc_estimate",
    "mfmc_variance",
    "mlblue_allocation",
    "mlblue_estimate",
    "mlblue_groups",
    "mlblue_variance",
    "pilot_statistics",
    "select_mfmc_models",
    "StratifiedEstimate",
    "allocate_stratified_budget",
    "compare_efficiency",
    "convergence_trace",
    "estimate_antithetic_uniform",
    "estimate_conditional_expectation",
    "estimate_control_variate",
    "estimate_importance_sampling",
    "estimate_mean",
    "estimate_mean_streaming",
    "estimate_probability",
    "estimate_stratified_uniform",
    "estimate_uniform_integral",
    "CLTSimulation",
    "RunningMeanPaths",
    "WeakLawPoint",
    "simulate_clt_distance",
    "simulate_running_means",
    "simulate_weak_law",
    "ControlFunctionalResult",
    "ControlFunctionalWeightsResult",
    "KernelSteinDiscrepancy",
    "ZeroVarianceResult",
    "estimate_control_functional",
    "estimate_control_functional_weights",
    "estimate_zero_variance",
    "kernel_stein_discrepancy",
    "median_heuristic_length_scale",
    "stein_kernel",
    "stein_polynomial_controls",
]
