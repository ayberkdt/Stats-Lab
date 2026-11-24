"""Requirements, margins and uncertainties, contingency, and performance tracking.

A design team states a requirement as a limit on a performance quantity: it
must stay *below* an upper limit (mass, power draw, temperature), stay
*above* a lower limit (link margin, stiffness, thrust), or inside a *band*.
This module turns that statement into numbers a review can use.

**Margin and uncertainty (QMU).** For a best estimate ``y*`` and an upper
limit ``L`` the margin is ``M = L - y*`` (for a lower limit ``M = y* - L``).
The uncertainty ``U`` is the distance from ``y*`` to a stated bound of the
performance on the unsafe side, for example the 95 % quantile at 95 %
confidence (``U = y_bound - y*`` upward, ``U = y* - y_bound`` downward). The
confidence ratio is ``M / U`` ([EQ-PROP-MARGIN-001]): above one the margin
covers the stated uncertainty, below one it does not. The bound is obtained
in one of three ways, chosen by the caller and recorded in the result:

- ``"wilks"``: distribution-free, from samples
  (:func:`probstat.propagation.wilks_bound`);
- ``"normal-tolerance"``: under the **assumption of a normal law**, from
  samples (:func:`probstat.propagation.normal_tolerance_bound`);
- ``"moments"``: from a given mean and standard uncertainty with a
  coverage factor (default: the normal quantile ``z_p``), when no samples
  exist.

For a nested aleatory-epistemic study, :func:`quantify_margin_mixed` keeps
the two kinds of uncertainty apart: the aleatory part of ``U`` is what the
quantile would be at the most favourable epistemic point, the epistemic part
is the additional distance to the least favourable one. The split uses the
outer points of the study only; it carries no sampling error of the inner
loop.

**Contingency.** Resources that grow during design (mass, power) are
budgeted with percentages. This module makes the arithmetic explicit and
unambiguous ([EQ-PROP-MARGIN-002]): a *growth allowance* ``g`` is applied to
the basic estimate, ``predicted = basic (1 + g)``; a *system margin* ``m``
sits on top and is measured either *relative to the estimate*
(``allocation = predicted (1 + m)``) or *relative to the allocation*
(``allocation = predicted / (1 - m)``). The same physical margin has two
different percentages under the two conventions (:func:`convert_margin_fraction`
converts), which is the usual source of review disagreement. All percentages
are inputs; no agency value is built in. The bridge to probability
([EQ-PROP-MARGIN-003]) reads a fixed percentage as a confidence level
(:func:`confidence_of_contingency`) and a confidence level as a percentage
(:func:`contingency_for_confidence`).

**Tracking.** :class:`TechnicalPerformanceMeasure` follows one measure over
project milestones (label, best estimate, standard uncertainty or samples,
optional planned value and planned margin) and reports the margin history,
the probability of meeting the requirement, the comparison with the planned
margin-depletion profile, and the milestones that need attention.

Assumptions: samples are independent draws of the performance (not Latin
hypercube or quasi-random designs); the best estimate is a deterministic
value supplied by the engineer, not necessarily the sample mean; the unit of
every number of one measure is the same. A compliance probability computed
from a standard uncertainty assumes a normal law centred at the estimate.

References:
    [REF-HELTON-2011], [REF-PILCH-TRUCANO-HELTON-2006], [REF-NASA-SEH-2016],
    [REF-CLOPPER-PEARSON-1934], [REF-WILKS-1941].

"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from probstat.foundations.validation import require_finite_real
from probstat.inference import ConfidenceInterval, clopper_pearson_interval
from probstat.statistics import quantile

from .mixed import MixedPropagationResult, ProbabilityBox
from .tolerance import _choice, _level, _z, normal_tolerance_bound
from .wilks import wilks_bound

SENSES = ("upper", "lower", "band")
SIDES = ("upper", "lower")
UNCERTAINTY_METHODS = ("wilks", "normal-tolerance", "moments")
MARGIN_BASES = ("estimate", "allocation")
FLAG_REASONS = ("requirement-violated", "below-plan", "low-ratio")


def _finite(value: float, *, name: str) -> float:
    """Return a finite real number.

    Raises:
        TypeError: If the value is not real.
        ValueError: If the value is not finite.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No mathematical
        equation is evaluated.

    """
    return require_finite_real(value, name=name)


def _optional_finite(value: float | None, *, name: str) -> float | None:
    """Return ``None`` or a finite real number.

    Raises:
        TypeError: If the value is neither ``None`` nor real.
        ValueError: If the value is not finite.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No mathematical
        equation is evaluated.

    """
    return None if value is None else _finite(value, name=name)


def _finite_values(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    """Return a non-empty tuple of finite reals.

    Raises:
        TypeError: If a value is not real.
        ValueError: If the sample is empty or a value is not finite.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No mathematical
        equation is evaluated.

    """
    sample = tuple(_finite(value, name=name) for value in values)
    if not sample:
        raise ValueError(f"{name} must contain at least one value")
    return sample


def _confidence_ratio(margin: float, uncertainty: float) -> float:
    """Return ``M / U`` with the conventions for ``U <= 0``.

    A non-positive ``U`` means the stated bound is on the safe side of the
    best estimate, so uncertainty does not consume margin: the ratio is
    ``+inf`` for a positive margin, ``0`` for a zero margin and ``-inf`` for
    a negative one.

    References:
        [EQ-PROP-MARGIN-001].

    """
    if uncertainty > 0.0:
        return margin / uncertainty
    if margin > 0.0:
        return math.inf
    return 0.0 if margin == 0.0 else -math.inf


@dataclass(frozen=True, slots=True)
class Requirement:
    """A limit or band that a performance quantity must respect.

    Build one with :meth:`upper_limit`, :meth:`lower_limit`, or :meth:`band`.

    Args:
        sense: ``"upper"`` (value must stay at or below ``upper``),
            ``"lower"`` (at or above ``lower``), or ``"band"`` (both).
        lower: Lower limit, required for ``"lower"`` and ``"band"``.
        upper: Upper limit, required for ``"upper"`` and ``"band"``.

    Raises:
        TypeError: If a limit is not real.
        ValueError: If the sense is unknown, a needed limit is missing or
            not finite, an unneeded one is given, or ``lower >= upper`` for
            a band.

    References:
        [EQ-PROP-MARGIN-001], [REF-NASA-SEH-2016].

    """

    sense: str
    lower: float | None = None
    upper: float | None = None

    def __post_init__(self) -> None:
        """Validate the sense and the limits.

        References:
            [EQ-PROP-MARGIN-001]. No new equation.

        """
        _choice(self.sense, SENSES, name="sense")
        low = _optional_finite(self.lower, name="lower")
        high = _optional_finite(self.upper, name="upper")
        needs_low = self.sense in ("lower", "band")
        needs_high = self.sense in ("upper", "band")
        if needs_low != (low is not None) or needs_high != (high is not None):
            raise ValueError(f"sense {self.sense!r} needs exactly its own limits")
        if low is not None and high is not None and low >= high:
            raise ValueError("lower must be below upper for a band")
        object.__setattr__(self, "lower", low)
        object.__setattr__(self, "upper", high)

    @classmethod
    def upper_limit(cls, limit: float) -> Requirement:
        """Return the requirement "value stays at or below ``limit``".

        References:
            [EQ-PROP-MARGIN-001].

        """
        return cls("upper", None, limit)

    @classmethod
    def lower_limit(cls, limit: float) -> Requirement:
        """Return the requirement "value stays at or above ``limit``".

        References:
            [EQ-PROP-MARGIN-001].

        """
        return cls("lower", limit, None)

    @classmethod
    def band(cls, lower: float, upper: float) -> Requirement:
        """Return the requirement "value stays inside ``[lower, upper]``".

        References:
            [EQ-PROP-MARGIN-001].

        """
        return cls("band", lower, upper)

    def sides(self) -> tuple[tuple[str, float], ...]:
        """Return ``(side, limit)`` pairs; ``"upper"`` means value below limit.

        References:
            [EQ-PROP-MARGIN-001]. No new equation.

        """
        found: list[tuple[str, float]] = []
        if self.upper is not None:
            found.append(("upper", self.upper))
        if self.lower is not None:
            found.append(("lower", self.lower))
        return tuple(found)

    def is_met(self, value: float) -> bool:
        """Return whether ``value`` satisfies the requirement (limits included).

        Args:
            value: Finite performance value.

        Returns:
            ``True`` when every side holds.

        Raises:
            TypeError: If ``value`` is not real.
            ValueError: If ``value`` is not finite.

        References:
            [EQ-PROP-MARGIN-001].

        """
        number = _finite(value, name="value")
        return all(
            number <= limit if side == "upper" else number >= limit
            for side, limit in self.sides()
        )

    def margin(self, estimate: float) -> float:
        """Return the margin ``M``: the smallest signed distance to a limit.

        ``L - y*`` for an upper limit, ``y* - L`` for a lower limit, and the
        smaller of the two for a band. Negative when the requirement is
        violated at the estimate.

        Args:
            estimate: Finite best estimate ``y*``.

        Returns:
            The margin in the unit of the performance.

        Raises:
            TypeError: If ``estimate`` is not real.
            ValueError: If ``estimate`` is not finite.

        Equation:
            [EQ-PROP-MARGIN-001].

        References:
            [REF-HELTON-2011], [REF-NASA-SEH-2016].

        """
        value = _finite(estimate, name="estimate")
        return min(_side_margin(side, limit, value) for side, limit in self.sides())


def _side_margin(side: str, limit: float, estimate: float) -> float:
    """Return the signed distance from the estimate to one limit.

    References:
        [EQ-PROP-MARGIN-001].

    """
    return limit - estimate if side == "upper" else estimate - limit


@dataclass(frozen=True, slots=True)
class MarginSide:
    """Margin, uncertainty, and confidence ratio for one side of a requirement.

    Args:
        side: ``"upper"`` (value must stay below ``limit``) or ``"lower"``.
        limit: The limit on this side.
        margin: ``M``, signed distance from the estimate to the limit.
        bound: The stated bound of the performance on the unsafe side.
        uncertainty: ``U``, signed distance from the estimate to ``bound``
            towards the limit (negative if the bound is on the safe side).
        confidence_ratio: ``M / U`` (see :func:`quantify_margin`).
        aleatory_uncertainty: Part of ``U`` that remains at the most
            favourable epistemic point; ``None`` unless the margin came
            from :func:`quantify_margin_mixed`.
        epistemic_uncertainty: ``U`` minus the aleatory part; ``None``
            unless from :func:`quantify_margin_mixed`.

    References:
        [EQ-PROP-MARGIN-001], [REF-HELTON-2011].

    """

    side: str
    limit: float
    margin: float
    bound: float
    uncertainty: float
    confidence_ratio: float
    aleatory_uncertainty: float | None = None
    epistemic_uncertainty: float | None = None


@dataclass(frozen=True, slots=True)
class MarginAssessment:
    """Quantification of margin and uncertainty for one requirement.

    Args:
        requirement: The requirement assessed.
        estimate: Best estimate ``y*``.
        sides: One :class:`MarginSide` per side of the requirement.
        probability: Content ``p`` of the stated bound (the bounded quantile
            level on the unsafe side).
        confidence: Statistical confidence of the bound, ``None`` when the
            method carries none (``"moments"`` and ``"p-box"``).
        method: ``"wilks"``, ``"normal-tolerance"``, ``"moments"``, or
            ``"p-box"``.
        size: Number of samples used, ``None`` for ``"moments"``.

    References:
        [EQ-PROP-MARGIN-001], [REF-HELTON-2011].

    """

    requirement: Requirement
    estimate: float
    sides: tuple[MarginSide, ...]
    probability: float
    confidence: float | None
    method: str
    size: int | None
    _binding: MarginSide = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Select the binding side: smallest ratio, then smallest margin.

        References:
            [EQ-PROP-MARGIN-001]. No new equation.

        """
        object.__setattr__(
            self,
            "_binding",
            min(self.sides, key=lambda s: (s.confidence_ratio, s.margin)),
        )

    @property
    def binding(self) -> MarginSide:
        """Return the side with the smallest confidence ratio.

        References:
            [EQ-PROP-MARGIN-001]. No new equation.

        """
        return self._binding

    @property
    def margin(self) -> float:
        """Return ``M`` of the binding side.

        References:
            [EQ-PROP-MARGIN-001]. No new equation.

        """
        return self._binding.margin

    @property
    def uncertainty(self) -> float:
        """Return ``U`` of the binding side.

        References:
            [EQ-PROP-MARGIN-001]. No new equation.

        """
        return self._binding.uncertainty

    @property
    def confidence_ratio(self) -> float:
        """Return ``M / U`` of the binding side.

        References:
            [EQ-PROP-MARGIN-001]. No new equation.

        """
        return self._binding.confidence_ratio


def _assessment(
    requirement: Requirement,
    estimate: float,
    bounds: dict[str, float],
    *,
    probability: float,
    confidence: float | None,
    method: str,
    size: int | None,
    split: dict[str, tuple[float, float]] | None = None,
) -> MarginAssessment:
    """Assemble the per-side records from the bounds of each side.

    References:
        [EQ-PROP-MARGIN-001].

    """
    sides: list[MarginSide] = []
    for side, limit in requirement.sides():
        margin = _side_margin(side, limit, estimate)
        bound = bounds[side]
        uncertainty = bound - estimate if side == "upper" else estimate - bound
        aleatory, epistemic = (None, None) if split is None else split[side]
        sides.append(
            MarginSide(
                side,
                limit,
                margin,
                bound,
                uncertainty,
                _confidence_ratio(margin, uncertainty),
                aleatory,
                epistemic,
            )
        )
    return MarginAssessment(
        requirement, estimate, tuple(sides), probability, confidence, method, size
    )


def quantify_margin(
    requirement: Requirement,
    estimate: float,
    *,
    samples: Sequence[float] | None = None,
    standard_uncertainty: float | None = None,
    mean: float | None = None,
    probability: float = 0.95,
    confidence: float = 0.95,
    method: str = "wilks",
    coverage_factor: float | None = None,
) -> MarginAssessment:
    """Return margin ``M``, uncertainty ``U`` and the ratio ``M / U``.

    The unsafe-side bound of the performance is the ``probability``-quantile
    bound for an upper side and the ``(1 - probability)``-quantile bound
    for a lower side, so that at least a fraction ``probability`` of the
    population lies on the safe side of it.

    - ``method="wilks"`` (needs ``samples``): the order-statistic bound at
      the stated ``confidence``, distribution-free; at least 59 samples for
      95 %/95 %.
    - ``method="normal-tolerance"`` (needs ``samples``): ``xbar +- k s``
      with the exact one-sided factor; **assumes a normal law**.
    - ``method="moments"`` (needs ``standard_uncertainty``): the bound is
      ``mean +- c u`` with ``c = coverage_factor`` (default ``z_p``, normal
      quantile); ``mean`` defaults to ``estimate``. No statistical
      confidence is attached, so ``confidence`` is ignored.

    Args:
        requirement: The limit or band.
        estimate: Best estimate ``y*``.
        samples: Independent draws of the performance.
        standard_uncertainty: Standard uncertainty ``u`` (non-negative).
        mean: Centre of the performance law for ``"moments"``.
        probability: Content ``p`` strictly inside ``(0, 1)``.
        confidence: Confidence strictly inside ``(0, 1)``.
        method: ``"wilks"``, ``"normal-tolerance"``, or ``"moments"``.
        coverage_factor: Multiplier ``c`` for ``"moments"``.

    Returns:
        The assessment, one :class:`MarginSide` per side.

    Raises:
        TypeError: If a number is not real.
        ValueError: If the inputs do not match the method, a level is out
            of range, ``standard_uncertainty`` is negative, or the sample is
            too small for the method.

    Equation:
        [EQ-PROP-MARGIN-001], [EQ-PROP-TOL-001], [EQ-PROP-WILKS-001].

    References:
        [REF-HELTON-2011], [REF-PILCH-TRUCANO-HELTON-2006],
        [REF-WILKS-1941].

    """
    best = _finite(estimate, name="estimate")
    p = _level(probability, name="probability")
    gamma = _level(confidence, name="confidence")
    _choice(method, UNCERTAINTY_METHODS, name="method")
    bounds: dict[str, float] = {}
    if method == "moments":
        if samples is not None or standard_uncertainty is None:
            raise ValueError(
                "method 'moments' needs standard_uncertainty and no samples"
            )
        spread = _finite(standard_uncertainty, name="standard_uncertainty")
        if spread < 0.0:
            raise ValueError("standard_uncertainty must be non-negative")
        centre = best if mean is None else _finite(mean, name="mean")
        factor = (
            _z(p)
            if coverage_factor is None
            else _finite(coverage_factor, name="coverage_factor")
        )
        for side, _ in requirement.sides():
            sign = 1.0 if side == "upper" else -1.0
            bounds[side] = centre + sign * factor * spread
        return _assessment(
            requirement, best, bounds, probability=p, confidence=None,
            method=method, size=None,
        )  # fmt: skip
    if samples is None or standard_uncertainty is not None:
        raise ValueError(f"method {method!r} needs samples and no standard_uncertainty")
    data = _finite_values(samples, name="samples")
    for side, _ in requirement.sides():
        if method == "wilks":
            level = p if side == "upper" else 1.0 - p
            bounds[side] = wilks_bound(data, level, gamma, sided=side).bound
        else:
            bounds[side] = normal_tolerance_bound(data, p, gamma, sided=side).bound
    return _assessment(
        requirement, best, bounds, probability=p, confidence=gamma,
        method=method, size=len(data),
    )  # fmt: skip


def quantify_margin_mixed(
    requirement: Requirement,
    estimate: float,
    uncertainty: ProbabilityBox | MixedPropagationResult,
    *,
    probability: float = 0.95,
) -> MarginAssessment:
    """Return margin and uncertainty from a nested aleatory-epistemic study.

    Each outer (epistemic) point has its own inner (aleatory) output sample;
    its type-7 ``probability``-quantile (``1 - probability`` on a lower
    side) is the aleatory answer at that point. The bound is the least
    favourable of these over the outer points, so ``U`` covers aleatory
    variability and epistemic uncertainty together. It is split exactly as
    ``U = U_aleatory + U_epistemic``: ``U_aleatory`` is the distance to the
    quantile at the *most favourable* epistemic point (what would remain if
    the epistemic uncertainty were resolved in the best way) and
    ``U_epistemic`` is the further distance to the least favourable one.
    ``U_aleatory`` is negative when even the best quantile is on the safe
    side of the estimate.

    The statement is about the outer points that were run (corners, design
    or samples); it holds for the full epistemic region only to the extent
    that they reach its extremes, and the inner-loop sampling error of the
    quantiles is not included.

    Args:
        requirement: The limit or band.
        estimate: Best estimate ``y*``.
        uncertainty: A :class:`~probstat.propagation.ProbabilityBox` or the
            :class:`~probstat.propagation.MixedPropagationResult` holding
            one.
        probability: Content ``p`` strictly inside ``(0, 1)``.

    Returns:
        The assessment with the aleatory and epistemic parts per side.

    Raises:
        TypeError: If ``estimate`` is not real or ``uncertainty`` has the
            wrong type.
        ValueError: If a number is not finite or ``probability`` is out of
            range.

    Equation:
        [EQ-PROP-MARGIN-001].

    References:
        [REF-HELTON-2011], [REF-FERSON-ET-AL-2003].

    """
    best = _finite(estimate, name="estimate")
    p = _level(probability, name="probability")
    if isinstance(uncertainty, MixedPropagationResult):
        box = uncertainty.pbox
    elif isinstance(uncertainty, ProbabilityBox):
        box = uncertainty
    else:
        raise TypeError(
            "uncertainty must be a ProbabilityBox or MixedPropagationResult"
        )
    bounds: dict[str, float] = {}
    split: dict[str, tuple[float, float]] = {}
    for side, _ in requirement.sides():
        level = p if side == "upper" else 1.0 - p
        quantiles = [quantile(list(sample), level) for sample in box.sorted_outputs]
        worst, favourable = (
            (max(quantiles), min(quantiles))
            if side == "upper"
            else (min(quantiles), max(quantiles))
        )
        bounds[side] = worst
        sign = 1.0 if side == "upper" else -1.0
        total = sign * (worst - best)
        aleatory = sign * (favourable - best)
        split[side] = (aleatory, total - aleatory)
    return _assessment(
        requirement, best, bounds, probability=p, confidence=None,
        method="p-box", size=len(box.sorted_outputs[0]), split=split,
    )  # fmt: skip


@dataclass(frozen=True, slots=True)
class ComplianceProbability:
    """Estimated probability of meeting a requirement, with an exact interval.

    Args:
        successes: Number of samples that meet the requirement.
        trials: Number of samples.
        probability: ``successes / trials``.
        interval: Two-sided Clopper-Pearson interval at ``confidence``.
        lower_bound: One-sided exact lower confidence bound at
            ``confidence`` (the lower end of the two-sided interval at
            level ``2 confidence - 1``); ``None`` when ``confidence <= 0.5``.
        confidence: Confidence level of the interval and bound.
        requirement: The requirement tested.

    References:
        [REF-CLOPPER-PEARSON-1934], [EQ-INF-CI-CP-001].

    """

    successes: int
    trials: int
    probability: float
    interval: ConfidenceInterval
    lower_bound: float | None
    confidence: float
    requirement: Requirement


def compliance_probability(
    values: Sequence[float],
    limit: float | Requirement,
    *,
    sense: str = "upper",
    confidence: float = 0.95,
) -> ComplianceProbability:
    """Return the fraction of samples meeting a limit, with an exact interval.

    A value on the limit meets it. The interval is the Clopper-Pearson
    interval of :func:`probstat.inference.clopper_pearson_interval`; it is
    exact for independent draws and conservative because the binomial is
    discrete. With zero failures in ``n`` samples the lower bound is
    ``(1 - confidence)^(1/n)`` for the one-sided bound at level
    ``confidence``, the success-run result.

    Args:
        values: Independent finite samples of the performance.
        limit: A number (with ``sense``) or a :class:`Requirement`.
        sense: ``"upper"`` or ``"lower"`` when ``limit`` is a number.
        confidence: Confidence in ``(0, 1)``.

    Returns:
        Counts, the point estimate, the interval, and the one-sided bound.

    Raises:
        TypeError: If a value or ``limit`` is not real or a Requirement.
        ValueError: If no samples are given, a number is not finite, a
            level is out of range, or ``sense`` is ``"band"`` for a number.

    Equation:
        [EQ-INF-CI-CP-001], [EQ-PROP-MARGIN-003].

    References:
        [REF-CLOPPER-PEARSON-1934].

    """
    data = _finite_values(values, name="values")
    gamma = _level(confidence, name="confidence")
    if isinstance(limit, Requirement):
        requirement = limit
    else:
        _choice(sense, SIDES, name="sense")
        number = _finite(limit, name="limit")
        requirement = (
            Requirement.upper_limit(number)
            if sense == "upper"
            else Requirement.lower_limit(number)
        )
    successes = sum(1 for value in data if requirement.is_met(value))
    interval = clopper_pearson_interval(successes, len(data), confidence=gamma)
    one_sided = (
        clopper_pearson_interval(
            successes, len(data), confidence=2.0 * gamma - 1.0
        ).lower
        if gamma > 0.5
        else None
    )
    return ComplianceProbability(
        successes,
        len(data),
        successes / len(data),
        interval,
        one_sided,
        gamma,
        requirement,
    )


@dataclass(frozen=True, slots=True)
class ContingencyBudget:
    """Explicit growth-allowance and system-margin arithmetic.

    ``predicted = basic (1 + g)``; with ``margin_basis="estimate"``,
    ``allocation = predicted (1 + m)``; with ``margin_basis="allocation"``,
    ``allocation = predicted / (1 - m)``. Both conventions describe an upper
    allocation that a growing resource must not exceed.

    Args:
        basic_estimate: Basic (current best) estimate before growth.
        growth_fraction: Growth allowance ``g`` as a fraction of the basic
            estimate.
        growth_allowance: ``basic g`` in the unit of the estimate.
        predicted: ``basic (1 + g)``.
        margin_fraction: System margin ``m`` in the stated basis.
        margin_basis: ``"estimate"`` or ``"allocation"``.
        system_margin: Absolute margin ``allocation - predicted``.
        allocation: The allocation that carries the margin.

    References:
        [EQ-PROP-MARGIN-002], [REF-NASA-SEH-2016].

    """

    basic_estimate: float
    growth_fraction: float
    growth_allowance: float
    predicted: float
    margin_fraction: float
    margin_basis: str
    system_margin: float
    allocation: float


def apply_contingency(
    basic_estimate: float,
    growth_fraction: float,
    margin_fraction: float,
    *,
    margin_basis: str = "estimate",
) -> ContingencyBudget:
    """Apply a growth allowance and then a system margin to a basic estimate.

    ``predicted = basic (1 + g)``. The margin is added on top of the
    predicted value: relative to the estimate it is ``m * predicted``;
    relative to the allocation ``m`` is the share of the allocation that is
    margin, so ``allocation = predicted / (1 - m)``. The percentages are the
    caller's; none is built in.

    Args:
        basic_estimate: Positive basic estimate.
        growth_fraction: Non-negative growth allowance ``g`` (0.15 is 15 %).
        margin_fraction: System margin ``m``, non-negative; below one when
            relative to the allocation.
        margin_basis: ``"estimate"`` or ``"allocation"``.

    Returns:
        The budget with every intermediate quantity.

    Raises:
        TypeError: If a number is not real.
        ValueError: If a number is out of range or the basis is unknown.

    Equation:
        [EQ-PROP-MARGIN-002].

    References:
        [REF-NASA-SEH-2016].

    """
    basic = _finite(basic_estimate, name="basic_estimate")
    growth = _finite(growth_fraction, name="growth_fraction")
    fraction = _finite(margin_fraction, name="margin_fraction")
    _choice(margin_basis, MARGIN_BASES, name="margin_basis")
    if basic <= 0.0:
        raise ValueError("basic_estimate must be positive")
    if growth < 0.0 or fraction < 0.0:
        raise ValueError("growth_fraction and margin_fraction must be non-negative")
    if margin_basis == "allocation" and fraction >= 1.0:
        raise ValueError("a margin relative to the allocation must be below one")
    predicted = basic * (1.0 + growth)
    allocation = (
        predicted * (1.0 + fraction)
        if margin_basis == "estimate"
        else predicted / (1.0 - fraction)
    )
    return ContingencyBudget(
        basic, growth, basic * growth, predicted, fraction, margin_basis,
        allocation - predicted, allocation,
    )  # fmt: skip


def allocation_margin_fraction(
    predicted: float, allocation: float, *, basis: str = "estimate"
) -> float:
    """Return the margin fraction of an allocation over a predicted value.

    ``(allocation - predicted) / predicted`` relative to the estimate,
    ``(allocation - predicted) / allocation`` relative to the allocation.
    Negative when the prediction exceeds the allocation.

    Args:
        predicted: Positive predicted value.
        allocation: Positive allocation.
        basis: ``"estimate"`` or ``"allocation"``.

    Returns:
        The margin as a fraction (not a percentage).

    Raises:
        TypeError: If a number is not real.
        ValueError: If a number is not positive or the basis is unknown.

    Equation:
        [EQ-PROP-MARGIN-002].

    References:
        [REF-NASA-SEH-2016].

    """
    value = _finite(predicted, name="predicted")
    cap = _finite(allocation, name="allocation")
    _choice(basis, MARGIN_BASES, name="basis")
    if value <= 0.0 or cap <= 0.0:
        raise ValueError("predicted and allocation must be positive")
    return (cap - value) / (value if basis == "estimate" else cap)


def convert_margin_fraction(fraction: float, *, source: str, target: str) -> float:
    """Convert a margin fraction between the two conventions.

    ``m_allocation = m_estimate / (1 + m_estimate)`` and
    ``m_estimate = m_allocation / (1 - m_allocation)``: a 25 % margin
    relative to the estimate is a 20 % margin relative to the allocation.

    Args:
        fraction: Margin fraction in the ``source`` basis (above ``-1``
            relative to the estimate, below one relative to the
            allocation).
        source: ``"estimate"`` or ``"allocation"``.
        target: ``"estimate"`` or ``"allocation"``.

    Returns:
        The same physical margin in the ``target`` basis.

    Raises:
        TypeError: If ``fraction`` is not real.
        ValueError: If a basis is unknown or ``fraction`` is out of range.

    Equation:
        [EQ-PROP-MARGIN-002].

    References:
        [REF-NASA-SEH-2016].

    """
    value = _finite(fraction, name="fraction")
    _choice(source, MARGIN_BASES, name="source")
    _choice(target, MARGIN_BASES, name="target")
    if source == "estimate" and value <= -1.0:
        raise ValueError("a margin relative to the estimate must exceed -1")
    if source == "allocation" and value >= 1.0:
        raise ValueError("a margin relative to the allocation must be below one")
    if source == target:
        return value
    if source == "estimate":
        return value / (1.0 + value)
    return value / (1.0 - value)


@dataclass(frozen=True, slots=True)
class ContingencyLevel:
    """The contingency that makes an estimate safe at a stated confidence.

    Args:
        estimate: Positive best estimate the contingency is added to.
        level: Probability that the performance stays at or below ``value``.
        value: ``estimate (1 + contingency)``, the ``level``-quantile (or
            its Wilks upper bound).
        contingency: ``value - estimate`` in the unit of the estimate.
        contingency_fraction: ``(value - estimate) / estimate``; negative
            if the quantile lies below the estimate.
        statistical_confidence: Confidence of the Wilks bound, ``None`` for
            the plain empirical quantile.
        method: ``"empirical-quantile"`` or ``"wilks"``.
        size: Number of samples.

    References:
        [EQ-PROP-MARGIN-003], [REF-WILKS-1941].

    """

    estimate: float
    level: float
    value: float
    contingency: float
    contingency_fraction: float
    statistical_confidence: float | None
    method: str
    size: int


def contingency_for_confidence(
    samples: Sequence[float],
    estimate: float,
    confidence: float,
    *,
    statistical_confidence: float | None = None,
) -> ContingencyLevel:
    """Return the contingency fraction that reaches a probability level.

    The performance stays at or below ``estimate (1 + c)`` with probability
    ``confidence`` when ``c`` is chosen so that ``estimate (1 + c)`` is the
    ``confidence``-quantile of the performance. By default it is the type-7
    sample quantile (a point estimate with sampling error); with
    ``statistical_confidence`` it is the Wilks upper bound of that quantile,
    so the level holds with that additional statistical confidence.

    Args:
        samples: Independent draws of the performance.
        estimate: Positive best estimate.
        confidence: Probability level in ``(0, 1)`` (for example 0.7 or
            0.95).
        statistical_confidence: Optional confidence in ``(0, 1)`` for a
            distribution-free bound.

    Returns:
        The contingency in absolute and relative terms.

    Raises:
        TypeError: If a number is not real.
        ValueError: If the estimate is not positive, a level is out of
            range, or the sample is too small for the Wilks bound.

    Equation:
        [EQ-PROP-MARGIN-003].

    References:
        [REF-WILKS-1941], [REF-NASA-SEH-2016].

    """
    data = _finite_values(samples, name="samples")
    best = _finite(estimate, name="estimate")
    level = _level(confidence, name="confidence")
    if best <= 0.0:
        raise ValueError("estimate must be positive")
    if statistical_confidence is None:
        value = float(quantile(data, level))
        label = "empirical-quantile"
        gamma = None
    else:
        gamma = _level(statistical_confidence, name="statistical_confidence")
        value = wilks_bound(data, level, gamma, sided="upper").bound
        label = "wilks"
    return ContingencyLevel(
        best, level, value, value - best, (value - best) / best, gamma, label, len(data)
    )


def confidence_of_contingency(
    samples: Sequence[float],
    estimate: float,
    contingency: float,
    *,
    confidence: float = 0.95,
) -> ComplianceProbability:
    """Return the probability that ``estimate (1 + contingency)`` suffices.

    The reading of a fixed percentage as a confidence level: the fraction
    of samples at or below ``estimate (1 + contingency)`` with an exact
    Clopper-Pearson interval (see :func:`compliance_probability`).

    Args:
        samples: Independent draws of the performance.
        estimate: Positive best estimate.
        contingency: Contingency as a fraction of the estimate (0.3 is 30 %).
        confidence: Confidence of the interval.

    Returns:
        The compliance probability against the limit
        ``estimate (1 + contingency)``.

    Raises:
        TypeError: If a number is not real.
        ValueError: If the estimate is not positive or a number is invalid.

    Equation:
        [EQ-PROP-MARGIN-003].

    References:
        [REF-CLOPPER-PEARSON-1934], [REF-NASA-SEH-2016].

    """
    best = _finite(estimate, name="estimate")
    extra = _finite(contingency, name="contingency")
    if best <= 0.0:
        raise ValueError("estimate must be positive")
    return compliance_probability(
        samples, best * (1.0 + extra), sense="upper", confidence=confidence
    )


@dataclass(frozen=True, slots=True)
class Milestone:
    """One record of a technical performance measure at a project milestone.

    At most one of ``standard_uncertainty`` and ``samples`` may be given; a
    milestone with neither has a margin but no uncertainty.

    Args:
        label: Non-empty milestone name (for example a review).
        estimate: Best estimate at this milestone.
        standard_uncertainty: Standard uncertainty ``u`` (non-negative).
        samples: Independent draws of the performance.
        planned_value: Planned best estimate at this milestone.
        planned_margin: Margin the plan requires at this milestone, in the
            unit of the measure (distance from estimate to limit).

    Raises:
        TypeError: If a number is not real or the label is not a string.
        ValueError: If the label is empty, a number is not finite, ``u`` is
            negative, or both uncertainty forms are given.

    References:
        [EQ-PROP-MARGIN-001], [REF-NASA-SEH-2016].

    """

    label: str
    estimate: float
    standard_uncertainty: float | None = None
    samples: tuple[float, ...] | None = None
    planned_value: float | None = None
    planned_margin: float | None = None

    def __post_init__(self) -> None:
        """Validate and normalize the fields.

        References:
            [EQ-PROP-MARGIN-001]. No new equation.

        """
        if not isinstance(self.label, str):
            raise TypeError("label must be a string")
        if not self.label.strip():
            raise ValueError("label must not be empty")
        object.__setattr__(self, "estimate", _finite(self.estimate, name="estimate"))
        spread = _optional_finite(
            self.standard_uncertainty, name="standard_uncertainty"
        )
        if spread is not None and spread < 0.0:
            raise ValueError("standard_uncertainty must be non-negative")
        object.__setattr__(self, "standard_uncertainty", spread)
        if self.samples is not None:
            object.__setattr__(
                self, "samples", _finite_values(self.samples, name="samples")
            )
        if spread is not None and self.samples is not None:
            raise ValueError("give standard_uncertainty or samples, not both")
        object.__setattr__(
            self,
            "planned_value",
            _optional_finite(self.planned_value, name="planned_value"),
        )
        object.__setattr__(
            self,
            "planned_margin",
            _optional_finite(self.planned_margin, name="planned_margin"),
        )


@dataclass(frozen=True, slots=True)
class MilestoneAssessment:
    """Margin, uncertainty, and compliance at one milestone.

    Args:
        label: Milestone name.
        estimate: Best estimate.
        margin: ``M`` against the requirement.
        planned_margin: Margin required by the plan, or ``None``.
        margin_vs_plan: ``margin - planned_margin`` (negative: below plan),
            or ``None``.
        deviation_from_plan: ``estimate - planned_value``, or ``None``.
        uncertainty: ``U`` of the binding side, or ``None`` without data.
        confidence_ratio: ``M / U`` of the binding side, or ``None``.
        compliance_probability: Probability of meeting the requirement, or
            ``None`` without data.
        compliance_interval: Exact interval of the probability (samples
            only), otherwise ``None``.
        method: Uncertainty method used, or ``None`` without data.

    References:
        [EQ-PROP-MARGIN-001], [REF-NASA-SEH-2016].

    """

    label: str
    estimate: float
    margin: float
    planned_margin: float | None
    margin_vs_plan: float | None
    deviation_from_plan: float | None
    uncertainty: float | None
    confidence_ratio: float | None
    compliance_probability: float | None
    compliance_interval: tuple[float, float] | None
    method: str | None


@dataclass(frozen=True, slots=True)
class MilestoneFlag:
    """A milestone that needs attention.

    Args:
        label: Milestone name.
        reason: ``"requirement-violated"`` (margin below zero),
            ``"below-plan"`` (margin below the planned margin), or
            ``"low-ratio"`` (``M / U`` below the threshold).
        detail: Short plain-text description with the numbers.

    References:
        [EQ-PROP-MARGIN-001].

    """

    label: str
    reason: str
    detail: str


@dataclass(frozen=True, slots=True)
class PerformanceAssessment:
    """The assessed history of a technical performance measure.

    Args:
        name: Measure name.
        unit: Unit text.
        requirement: The requirement.
        milestones: One record per milestone, in order.
        flags: Milestones that need attention.
        probability: Content ``p`` of the uncertainty bound.
        confidence: Statistical confidence of the bound and intervals.
        ratio_threshold: ``M / U`` below this value raises a flag.

    References:
        [EQ-PROP-MARGIN-001], [REF-NASA-SEH-2016].

    """

    name: str
    unit: str
    requirement: Requirement
    milestones: tuple[MilestoneAssessment, ...]
    flags: tuple[MilestoneFlag, ...]
    probability: float
    confidence: float
    ratio_threshold: float

    def render(self) -> str:
        """Return a plain-text table of the milestones followed by the flags.

        Returns:
            Fixed-width text; missing entries are shown as ``-``.

        References:
            [REF-NASA-SEH-2016]. No mathematical equation is evaluated.

        """

        def cell(value: float | None) -> str:
            """Format an optional number to four significant digits.

            References:
                [REF-NASA-SEH-2016]. No mathematical equation is evaluated.

            """
            return "-" if value is None else f"{value:.4g}"

        header = ("milestone", "estimate", "margin", "planned", "vs plan",
                  "U", "M/U", "P(meet)")  # fmt: skip
        rows = [header]
        for item in self.milestones:
            rows.append(
                (
                    item.label,
                    cell(item.estimate),
                    cell(item.margin),
                    cell(item.planned_margin),
                    cell(item.margin_vs_plan),
                    cell(item.uncertainty),
                    cell(item.confidence_ratio),
                    cell(item.compliance_probability),
                )
            )
        widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
        lines = [
            f"{self.name} [{self.unit}]: p = {self.probability:g}, "
            f"confidence = {self.confidence:g}, M/U threshold = "
            f"{self.ratio_threshold:g}"
        ]
        for index, row in enumerate(rows):
            lines.append(
                row[0].ljust(widths[0])
                + "  "
                + "  ".join(
                    text.rjust(width)
                    for text, width in zip(row[1:], widths[1:], strict=True)
                )
            )
            if index == 0:
                lines.append("-" * len(lines[-1]))
        if self.flags:
            lines.append("flags:")
            lines.extend(
                f"  {flag.label}: {flag.reason} ({flag.detail})" for flag in self.flags
            )
        else:
            lines.append("flags: none")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class TechnicalPerformanceMeasure:
    """A named performance measure tracked across project milestones.

    Args:
        name: Non-empty measure name (for example ``"dry mass"``).
        unit: Unit text shown in tables.
        requirement: The requirement the measure must meet.
        milestones: Milestone records in chronological order (at least
            one, with distinct labels).

    Raises:
        TypeError: If a field has the wrong type.
        ValueError: If a name or label is empty or duplicated or there are
            no milestones.

    References:
        [REF-NASA-SEH-2016], [EQ-PROP-MARGIN-001].

    """

    name: str
    unit: str
    requirement: Requirement
    milestones: tuple[Milestone, ...]

    def __post_init__(self) -> None:
        """Validate the measure.

        References:
            [EQ-PROP-MARGIN-001]. No new equation.

        """
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        if not isinstance(self.unit, str):
            raise TypeError("unit must be a string")
        if not isinstance(self.requirement, Requirement):
            raise TypeError("requirement must be a Requirement")
        records = tuple(self.milestones)
        if not records or not all(isinstance(item, Milestone) for item in records):
            raise ValueError("milestones must be a non-empty sequence of Milestone")
        labels = [item.label for item in records]
        if len(set(labels)) != len(labels):
            raise ValueError("milestone labels must be distinct")
        object.__setattr__(self, "milestones", records)

    def margin_history(self) -> tuple[float, ...]:
        """Return the margin ``M`` at each milestone.

        Returns:
            Margins in milestone order, in the unit of the measure.

        Equation:
            [EQ-PROP-MARGIN-001].

        References:
            [REF-NASA-SEH-2016].

        """
        return tuple(self.requirement.margin(item.estimate) for item in self.milestones)

    def compliance_probabilities(
        self, *, confidence: float = 0.95
    ) -> tuple[float | None, ...]:
        """Return the probability of meeting the requirement at each milestone.

        From samples: the fraction meeting it (exact interval available from
        :meth:`assess`). From a standard uncertainty: the probability under
        a **normal law centred at the estimate**. ``None`` without either.

        Args:
            confidence: Confidence used for the sample intervals (the point
                values do not depend on it).

        Returns:
            One entry per milestone.

        Raises:
            ValueError: If ``confidence`` is outside ``(0, 1)``.

        Equation:
            [EQ-PROP-MARGIN-003].

        References:
            [REF-CLOPPER-PEARSON-1934].

        """
        return tuple(
            probability
            for probability, _ in (
                self._compliance(item, _level(confidence, name="confidence"))
                for item in self.milestones
            )
        )

    def plan_comparison(self) -> tuple[float | None, ...]:
        """Return ``margin - planned_margin`` at each milestone.

        Negative entries are milestones whose margin is below the planned
        margin-depletion profile; ``None`` where no planned margin is given.

        Returns:
            One entry per milestone.

        Equation:
            [EQ-PROP-MARGIN-001].

        References:
            [REF-NASA-SEH-2016].

        """
        history = self.margin_history()
        return tuple(
            None if item.planned_margin is None else margin - item.planned_margin
            for item, margin in zip(self.milestones, history, strict=True)
        )

    def _compliance(
        self, item: Milestone, confidence: float
    ) -> tuple[float | None, tuple[float, float] | None]:
        """Return the compliance probability and its interval at one milestone.

        References:
            [EQ-PROP-MARGIN-003], [REF-CLOPPER-PEARSON-1934].

        """
        if item.samples is not None:
            result = compliance_probability(
                item.samples, self.requirement, confidence=confidence
            )
            return result.probability, (result.interval.lower, result.interval.upper)
        if item.standard_uncertainty is None:
            return None, None
        spread = item.standard_uncertainty
        if spread == 0.0:
            return (1.0 if self.requirement.is_met(item.estimate) else 0.0), None
        upper = (
            1.0
            if self.requirement.upper is None
            else _phi((self.requirement.upper - item.estimate) / spread)
        )
        lower = (
            0.0
            if self.requirement.lower is None
            else _phi((self.requirement.lower - item.estimate) / spread)
        )
        return max(0.0, upper - lower), None

    def assess(
        self,
        *,
        probability: float = 0.95,
        confidence: float = 0.95,
        ratio_threshold: float = 1.0,
        method: str = "normal-tolerance",
    ) -> PerformanceAssessment:
        """Assess every milestone and flag those that need attention.

        A milestone with samples uses ``method`` (``"wilks"`` or
        ``"normal-tolerance"``) for ``U``; one with a standard uncertainty
        uses ``"moments"`` (normal quantile ``z_p``); one with neither has
        no ``U``. A flag is raised when the margin is negative
        (``requirement-violated``), below the planned margin
        (``below-plan``), or when ``M / U`` is below ``ratio_threshold``
        (``low-ratio``). With the default ``"normal-tolerance"`` a milestone
        needs at least two samples; ``"wilks"`` needs 59 for the default
        levels.

        Args:
            probability: Content ``p`` of the uncertainty bound.
            confidence: Statistical confidence of bounds and intervals.
            ratio_threshold: Smallest acceptable ``M / U`` (finite).
            method: ``"wilks"`` or ``"normal-tolerance"`` for sampled
                milestones.

        Returns:
            The assessed history with flags and a text renderer.

        Raises:
            TypeError: If a number is not real.
            ValueError: If a level is out of range, ``method`` is unknown,
                or a sample is too small for the method.

        Equation:
            [EQ-PROP-MARGIN-001].

        References:
            [REF-HELTON-2011], [REF-NASA-SEH-2016].

        """
        p = _level(probability, name="probability")
        gamma = _level(confidence, name="confidence")
        threshold = _finite(ratio_threshold, name="ratio_threshold")
        _choice(method, ("wilks", "normal-tolerance"), name="method")
        records: list[MilestoneAssessment] = []
        flags: list[MilestoneFlag] = []
        for item in self.milestones:
            margin = self.requirement.margin(item.estimate)
            uncertainty = ratio = used = None
            if item.samples is not None:
                found = quantify_margin(
                    self.requirement, item.estimate, samples=item.samples,
                    probability=p, confidence=gamma, method=method,
                )  # fmt: skip
            elif item.standard_uncertainty is not None:
                found = quantify_margin(
                    self.requirement, item.estimate,
                    standard_uncertainty=item.standard_uncertainty,
                    probability=p, method="moments",
                )  # fmt: skip
            else:
                found = None
            if found is not None:
                uncertainty, ratio, used = (
                    found.uncertainty, found.confidence_ratio, found.method
                )  # fmt: skip
            chance, interval = self._compliance(item, gamma)
            versus = (
                None if item.planned_margin is None else margin - item.planned_margin
            )
            drift = (
                None
                if item.planned_value is None
                else item.estimate - item.planned_value
            )
            records.append(
                MilestoneAssessment(
                    item.label,
                    item.estimate,
                    margin,
                    item.planned_margin,
                    versus,
                    drift,
                    uncertainty,
                    ratio,
                    chance,
                    interval,
                    used,
                )  # fmt: skip
            )
            if margin < 0.0:
                flags.append(
                    MilestoneFlag(
                        item.label, "requirement-violated", f"margin {margin:.4g}"
                    )
                )
            if versus is not None and versus < 0.0:
                flags.append(
                    MilestoneFlag(
                        item.label,
                        "below-plan",
                        f"margin {margin:.4g} vs planned {item.planned_margin:.4g}",
                    )
                )
            if ratio is not None and ratio < threshold:
                flags.append(
                    MilestoneFlag(
                        item.label,
                        "low-ratio",
                        f"M/U {ratio:.4g} below {threshold:.4g}",
                    )
                )
        return PerformanceAssessment(
            self.name, self.unit, self.requirement, tuple(records), tuple(flags),
            p, gamma, threshold,
        )  # fmt: skip


def _phi(x: float) -> float:
    """Return the standard normal distribution function.

    References:
        [REF-NIST-DLMF-ELEMENTARY]. No new equation.

    """
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


__all__ = [
    "ComplianceProbability",
    "ContingencyBudget",
    "ContingencyLevel",
    "MarginAssessment",
    "MarginSide",
    "Milestone",
    "MilestoneAssessment",
    "MilestoneFlag",
    "PerformanceAssessment",
    "Requirement",
    "TechnicalPerformanceMeasure",
    "allocation_margin_fraction",
    "apply_contingency",
    "compliance_probability",
    "confidence_of_contingency",
    "contingency_for_confidence",
    "convert_margin_fraction",
    "quantify_margin",
    "quantify_margin_mixed",
]
