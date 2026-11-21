"""A laboratory for deriving new estimators and certifying their claims.

The other packages of the library *contain* methods. This one supplies the
tools for *making* one and for finding out whether it does what its author
claims:

``estimators``
    the contract (a seeded draw with a cost) and the record of how an
    estimator was derived (its lineage);
``combinators``
    construction principles from the literature as operations on
    estimators: randomized truncation, Richardson extrapolation, best linear
    unbiased combination, control variates, antithetic pairs, jackknife and
    bootstrap bias correction, empirical influence functions and the
    one-step correction;
``certify``
    checks of claims by seeded replication (unbiasedness, variance,
    coverage, convergence rate, efficiency gain, metamorphic relations)
    with scoped verdicts, collected in a method card;
``simulation_study``
    pre-specified simulation studies in the ADEMP structure with Monte
    Carlo standard errors and a protocol fingerprint;
``racing``
    selection among a finite set of candidates by efficiency ranking and
    sequential halving.

Main entry points:
    ``Estimator``, ``estimator_from_function``: the estimator contract.
    ``single_term_estimator``, ``coupled_sum_estimator``: unbiased
        estimators from a convergent sequence of approximations.
    ``extrapolated_estimator``: Richardson extrapolation of a family.
    ``combine_unbiased``, ``control_variate_estimator``: linear
        combinations with frozen pilot weights.
    ``jackknife_bias_corrected``, ``empirical_influence``,
        ``one_step_estimator``: corrections and standard errors for
        statistics of a sample.
    ``check_unbiasedness``, ``check_coverage``, ``check_efficiency_gain``,
        ``method_card``: certification of claims.
    ``run_simulation_study``: an ADEMP study with a fingerprinted protocol.
    ``rank_by_efficiency``, ``sequential_halving``: candidate selection.

Assumptions:
    Every construction is exact only under the conditions its docstring
    states (for example a finite second-moment series for randomized
    truncation, a known bias expansion for extrapolation, weights
    independent of the combined draws for linear combinations). The
    library cannot prove these conditions; the certification checks can
    refute them on a tested problem, and a passed check supports a claim
    only for the tested mechanisms, sizes, and seeds.

Common pitfalls:
    - "Not refuted" is not "supported": without a tolerance declared before
      the run no check can support a claim.
    - A randomized-truncation estimator with a level law that is too light
      has infinite variance; its sample mean looks unbiased and its
      standard error is meaningless. Run ``check_variance_stability``.
    - Weights or coefficients estimated from the same draws they multiply
      make a combination biased; the combinators freeze them from a pilot.
    - The candidate selected by a search looks better than it is; certify
      it on fresh seeds.
    - Comparing estimators by variance alone ignores cost; use the
      work-normalized variance.
    - A simulation study designed after looking at results can make any
      method win; publish the protocol fingerprint first.

Guide: docs/foundations/method-development.md

References:
    [REF-RHEE-GLYNN-2015], [REF-GLASSERMAN-2003, Section 1.1.3],
    [REF-MORRIS-WHITE-CROWTHER-2019], [REF-PAWEL-KOOK-REEVE-2024],
    [REF-KARNIN-KOREN-SOMEKH-2013], [REF-KENNEDY-2022].

"""

from .certify import (
    VERDICTS,
    CheckResult,
    InvarianceResult,
    MethodCard,
    VarianceStability,
    check_convergence_rate,
    check_coverage,
    check_efficiency_gain,
    check_invariance,
    check_unbiasedness,
    check_variance,
    check_variance_stability,
    interval_verdict,
    method_card,
)
from .combinators import (
    InfluenceAnalysis,
    LevelDistribution,
    LinearCombination,
    OneStepEstimate,
    TruncationDesign,
    antithetic_estimator,
    best_linear_unbiased_weights,
    bootstrap_bias_corrected,
    combine_unbiased,
    control_variate_estimator,
    coupled_sum_estimator,
    empirical_influence,
    extrapolated_estimator,
    extrapolation_weights,
    finite_levels,
    geometric_levels,
    jackknife_bias_corrected,
    one_step_estimator,
    optimal_level_probabilities,
    pilot_covariance,
    single_term_estimator,
    sum_of_estimators,
    thinned_estimator,
    truncation_design,
)
from .estimators import (
    Draw,
    Estimator,
    ReplicateSummary,
    estimator_from_function,
    sample_mean,
    summarize,
    summarize_draws,
)
from .racing import (
    EfficiencyRanking,
    HalvingResult,
    HalvingRound,
    rank_by_efficiency,
    sequential_halving,
)
from .simulation_study import (
    MEASURES,
    DataGeneratingMechanism,
    Method,
    MethodOutput,
    Performance,
    PrecisionComparison,
    SimulationStudyResult,
    StudyProtocol,
    performance_measures,
    repetitions_for_standard_error,
    run_simulation_study,
)

__all__ = [
    "MEASURES",
    "VERDICTS",
    "CheckResult",
    "DataGeneratingMechanism",
    "Draw",
    "EfficiencyRanking",
    "Estimator",
    "HalvingResult",
    "HalvingRound",
    "InfluenceAnalysis",
    "InvarianceResult",
    "LevelDistribution",
    "LinearCombination",
    "Method",
    "MethodCard",
    "MethodOutput",
    "OneStepEstimate",
    "Performance",
    "PrecisionComparison",
    "ReplicateSummary",
    "SimulationStudyResult",
    "StudyProtocol",
    "TruncationDesign",
    "VarianceStability",
    "antithetic_estimator",
    "best_linear_unbiased_weights",
    "bootstrap_bias_corrected",
    "check_convergence_rate",
    "check_coverage",
    "check_efficiency_gain",
    "check_invariance",
    "check_unbiasedness",
    "check_variance",
    "check_variance_stability",
    "combine_unbiased",
    "control_variate_estimator",
    "coupled_sum_estimator",
    "empirical_influence",
    "estimator_from_function",
    "extrapolated_estimator",
    "extrapolation_weights",
    "finite_levels",
    "geometric_levels",
    "interval_verdict",
    "jackknife_bias_corrected",
    "method_card",
    "one_step_estimator",
    "optimal_level_probabilities",
    "performance_measures",
    "pilot_covariance",
    "rank_by_efficiency",
    "repetitions_for_standard_error",
    "run_simulation_study",
    "sample_mean",
    "sequential_halving",
    "single_term_estimator",
    "sum_of_estimators",
    "summarize",
    "summarize_draws",
    "thinned_estimator",
    "truncation_design",
]
