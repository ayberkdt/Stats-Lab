"""Forward uncertainty propagation and output summaries.

:func:`nested_propagation` separates aleatory and epistemic uncertainty in a
double loop and returns a probability box and epistemic intervals of output
statistics.

Main entry points:
    ``propagate``: push a model's joint input law through every QoI with a
        named design (``"mc"``, ``"lhs"``, ``"halton"``, ``"sobol"``).
    ``PropagationStudy``: the propagated sample; indexing by QoI name gives an
        ``OutputDistribution``.
    ``OutputDistribution``: one QoI sample with moments, type-7 quantiles,
        exceedance probabilities, and interval methods.
    ``compare_designs``: spread of the mean estimate over independent
        replicates of several designs.
    ``nested_propagation``: aleatory-epistemic double loop to a probability
        box.
    ``propagate_samples``: explicit named realizations through a forward
        model.
    ``expectation_by_quadrature``: one-dimensional expectation on explicit
        finite bounds.
    ``wilks_rank``, ``wilks_sample_size``, ``wilks_bound``,
        ``wilks_tolerance_interval``: distribution-free one-sided quantile
        bounds and two-sided tolerance intervals from order statistics.
    ``normal_tolerance_factor``, ``normal_tolerance_bound``,
        ``normal_tolerance_interval``, ``normal_tolerance_sample_size``,
        ``normality_price``: exact normal-theory tolerance factors (one-sided
        noncentral t, two-sided exact and Howe), bounds, intervals, and the
        comparison with the Wilks bound.
    ``Requirement``, ``quantify_margin``, ``quantify_margin_mixed``,
        ``compliance_probability``: margin ``M``, uncertainty ``U`` and the
        ratio ``M/U`` for a limit or band, with an exact compliance interval.
    ``apply_contingency``, ``contingency_for_confidence``,
        ``confidence_of_contingency``: explicit growth-allowance and
        system-margin arithmetic and its reading as a confidence level.
    ``TechnicalPerformanceMeasure``: margin history, compliance probability
        and plan comparison of one measure over project milestones.

Assumptions:
    Without ``replicates`` the error statements treat the outputs as
    independent draws, which holds for plain Monte Carlo only. Model runs are
    deterministic, so repeated runs at one input agree. Failed runs that are
    dropped are assumed unrelated to the output.

Common pitfalls:
    - ``propagate`` defaults to ``design="mc"`` and ``seed=None`` (fresh
      operating-system entropy, not reproducible), while
      ``workflow.UQStudy.propagate`` defaults to ``design="sobol"`` on a
      child of the study seed. The same call shape therefore gives different
      designs in the two places.
    - Without ``replicates``, ``mean_interval`` and the Wilson interval of
      ``exceedance`` assume independent draws; for LHS and quasi-random
      designs they are conservative references, not error bars. Pass
      ``replicates=r`` (cost ``r * size``) for Student-t intervals over
      independent randomizations.
    - ``exceedance(t)`` estimates the strict event ``Y > t``; values equal to
      ``t`` are not counted, which matters for discrete or clipped outputs.
    - ``variance_interval`` defaults to ``method="normal"``, the chi-square
      interval that is exact only for normal outputs; use
      ``method="bootstrap"`` for skewed or heavy-tailed outputs.
    - ``expectation_by_quadrature`` integrates only over ``[lower, upper]``;
      for an unbounded law the result is a truncated expectation.

Guide: docs/foundations/uncertainty-propagation.md

References:
    [REF-SMITH-UQ-2024, Chapters 3-4], [REF-L-ECUYER-2018],
    [REF-OWEN-2013, Chapter 17], [REF-HYNDMAN-FAN-1996], [REF-WILSON-1927],
    [REF-FERSON-ET-AL-2003], [REF-WILKS-1941], [REF-WILKS-1942],
    [REF-HOWE-1969], [REF-KRISHNAMOORTHY-MATHEW-2009], [REF-HELTON-2011],
    [REF-NASA-SEH-2016].

"""

from .forward import (
    OutputSummary,
    PropagationResult,
    expectation_by_quadrature,
    propagate_samples,
)
from .margins import (
    ComplianceProbability,
    ContingencyBudget,
    ContingencyLevel,
    MarginAssessment,
    MarginSide,
    Milestone,
    MilestoneAssessment,
    MilestoneFlag,
    PerformanceAssessment,
    Requirement,
    TechnicalPerformanceMeasure,
    allocation_margin_fraction,
    apply_contingency,
    compliance_probability,
    confidence_of_contingency,
    contingency_for_confidence,
    convert_margin_fraction,
    quantify_margin,
    quantify_margin_mixed,
)
from .mixed import (
    EpistemicStatistic,
    MixedPropagationResult,
    ProbabilityBox,
    nested_propagation,
)
from .study import (
    DESIGNS,
    OutputDistribution,
    PropagationStudy,
    QuantileInterval,
    compare_designs,
    make_design,
    order_statistic_interval,
    propagate,
    replicate_interval,
    summarize,
)
from .tolerance import (
    NormalityPrice,
    NormalToleranceBound,
    NormalToleranceInterval,
    normal_tolerance_bound,
    normal_tolerance_factor,
    normal_tolerance_interval,
    normal_tolerance_sample_size,
    normality_price,
)
from .wilks import (
    WilksBound,
    WilksRank,
    WilksToleranceInterval,
    wilks_bound,
    wilks_rank,
    wilks_sample_size,
    wilks_tolerance_interval,
)

__all__ = [
    "ComplianceProbability",
    "ContingencyBudget",
    "ContingencyLevel",
    "DESIGNS",
    "EpistemicStatistic",
    "MarginAssessment",
    "MarginSide",
    "Milestone",
    "MilestoneAssessment",
    "MilestoneFlag",
    "MixedPropagationResult",
    "NormalToleranceBound",
    "NormalToleranceInterval",
    "NormalityPrice",
    "OutputDistribution",
    "OutputSummary",
    "PerformanceAssessment",
    "ProbabilityBox",
    "PropagationResult",
    "PropagationStudy",
    "QuantileInterval",
    "Requirement",
    "TechnicalPerformanceMeasure",
    "WilksBound",
    "WilksRank",
    "WilksToleranceInterval",
    "allocation_margin_fraction",
    "apply_contingency",
    "compare_designs",
    "compliance_probability",
    "confidence_of_contingency",
    "contingency_for_confidence",
    "convert_margin_fraction",
    "expectation_by_quadrature",
    "make_design",
    "nested_propagation",
    "normal_tolerance_bound",
    "normal_tolerance_factor",
    "normal_tolerance_interval",
    "normal_tolerance_sample_size",
    "normality_price",
    "order_statistic_interval",
    "propagate",
    "propagate_samples",
    "quantify_margin",
    "quantify_margin_mixed",
    "replicate_interval",
    "summarize",
    "wilks_bound",
    "wilks_rank",
    "wilks_sample_size",
    "wilks_tolerance_interval",
]
