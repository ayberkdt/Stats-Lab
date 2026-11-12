"""Finite sample spaces, probability and general finite measures, and counting.

Exact probability on finite outcome sets: events and probability measures,
random variables as mappings with induced laws, general finite measures
with explicit sigma-algebras, counting rules, and exact or proven results
for limit theorems (Berry-Esseen, Edgeworth, tail bounds, delta method).

Main entry points:
    ``FiniteSampleSpace``: a non-empty set of distinct hashable outcomes
        whose subsets are events.
    ``FiniteProbabilityMeasure``: outcome weights with probability,
        conditioning, independence, total probability, and Bayes' rule.
    ``FiniteRandomVariable``: a mapping of outcomes with its induced PMF,
        CDF, moments, and generating functions.
    ``FiniteMeasure``: a finite measure on an explicit sigma-algebra
        (integrals, densities, conditional expectation).
    ``combinations``: binomial coefficient ``C(n, r)``.
    ``compare_tail_bounds``: exact sample-mean tails next to Chebyshev,
        Hoeffding, and Bernstein bounds.
    ``kolmogorov_distance_to_normal``: exact distance of a standardized
        mean to the normal law.

Assumptions:
    Outcome sets are finite and every probability is an exact sum of
    outcome weights; nothing here samples. ``FiniteProbabilityMeasure``
    weights must already sum to one. The limit-theorem functions assume
    i.i.d. Bernoulli, exponential, or normal summands.

Common pitfalls:
    - Weights are probabilities, not counts: each must lie in ``[0, 1]``
      and their sum must be one within ``1e-12``, so
      ``{1: 2, 2: 1, 3: 1}`` raises ``ValueError`` instead of being
      normalized.
    - Sample spaces are identified by object, not by content: two
      ``FiniteSampleSpace((1, 2, 3))`` are different experiments, and a
      measure rejects events of the other one. Two random variables form a
      joint law only when they share the same measure object.
    - ``are_independent`` compares ``P(A and B)`` with ``P(A) P(B)`` to an
      absolute tolerance of ``1e-12``; for rare events (probabilities near
      ``1e-7``) even disjoint events pass, so lower ``tolerance``.
    - Conditioning on a zero-probability event raises ``ValueError``,
      and ``combinations(n, r)`` and ``permutations(n, r)`` raise for
      ``r > n`` instead of returning 0.
    - ``FiniteRandomVariable.mean``, ``.variance``, and
      ``FiniteJointDistribution.correlation`` are properties (no
      parentheses). ``berry_esseen_bound`` is not capped at one, whereas
      the Chebyshev, Hoeffding, and Bernstein bounds are.

Guide: docs/foundations/probability.md

References:
    [REF-KOLMOGOROV-1956] (axioms), [REF-ROSS-2019] (finite probability
    and counting), [REF-BILLINGSLEY-1995] and [REF-DURRETT-2019] (measure
    theory), [REF-HOEFFDING-1963] (tail bounds), [REF-HALL-1992]
    (Edgeworth expansions).

"""

from .counting import combinations, factorial, multiplication_rule, permutations
from .finite import FiniteEvent, FiniteProbabilityMeasure, FiniteSampleSpace
from .limit_theorems import (
    BERRY_ESSEEN_CONSTANT,
    CLTErrorRow,
    DeltaMethodComparison,
    SummandLaw,
    TailBoundComparison,
    bernoulli_summand,
    bernstein_bound,
    berry_esseen_bound,
    chebyshev_bound,
    clt_error_profile,
    compare_tail_bounds,
    delta_method_bernoulli,
    delta_method_exponential_reciprocal,
    delta_method_normal_exp,
    delta_method_normal_square,
    delta_method_variance,
    edgeworth_cdf,
    exponential_summand,
    hoeffding_bound,
    kolmogorov_distance_to_normal,
    normal_summand,
    sample_mean_two_sided_tail,
    sample_mean_upper_tail,
    standardized_mean_cdf,
)
from .measure import (
    CompletenessCheck,
    FiniteMeasurableSpace,
    FiniteMeasure,
    FubiniCheck,
    check_completeness,
    is_complete_statistic,
    is_sigma_algebra,
)
from .random_variables import FiniteJointDistribution, FiniteRandomVariable

__all__ = [
    "FiniteEvent",
    "FiniteJointDistribution",
    "FiniteProbabilityMeasure",
    "FiniteRandomVariable",
    "FiniteSampleSpace",
    "combinations",
    "factorial",
    "multiplication_rule",
    "permutations",
    "BERRY_ESSEEN_CONSTANT",
    "CLTErrorRow",
    "DeltaMethodComparison",
    "SummandLaw",
    "TailBoundComparison",
    "bernoulli_summand",
    "bernstein_bound",
    "berry_esseen_bound",
    "chebyshev_bound",
    "clt_error_profile",
    "compare_tail_bounds",
    "delta_method_bernoulli",
    "delta_method_exponential_reciprocal",
    "delta_method_normal_exp",
    "delta_method_normal_square",
    "delta_method_variance",
    "edgeworth_cdf",
    "exponential_summand",
    "hoeffding_bound",
    "kolmogorov_distance_to_normal",
    "normal_summand",
    "sample_mean_two_sided_tail",
    "sample_mean_upper_tail",
    "standardized_mean_cdf",
    "CompletenessCheck",
    "FiniteMeasurableSpace",
    "FiniteMeasure",
    "FubiniCheck",
    "check_completeness",
    "is_complete_statistic",
    "is_sigma_algebra",
]

