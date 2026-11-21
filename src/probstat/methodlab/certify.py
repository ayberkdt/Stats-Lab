"""Certification checks that confront the claims of a method with evidence.

A derived estimator comes with claims: "it is unbiased for ``mu``", "its
variance is ``v``", "its interval covers at the nominal level", "its error
decays at rate ``r``", "it is more efficient than the baseline". Each
function here tests one such claim by seeded replication and returns a
:class:`CheckResult` with the estimate, its Monte Carlo standard error, a
confidence interval, and a verdict.

Verdicts are deliberately asymmetric, in the tradition of equivalence
testing:

- ``"refuted"``: the confidence interval excludes the claimed value (or lies
  entirely outside the tolerance band);
- ``"supported"``: a tolerance band was declared *before* the run and the
  whole confidence interval lies inside it;
- ``"not refuted"``: no tolerance was declared and the interval contains the
  claimed value. Absence of evidence against a claim is not evidence for it;
- ``"inconclusive"``: a tolerance was declared and the interval overlaps its
  boundary; more replicates are needed.

No check can prove a claim: simulation evidence holds for the tested
data-generating mechanism and replicate count only. A :class:`MethodCard`
collects the checks of one method into a report that says exactly that.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TypeVar

from probstat.distributions import Normal, StudentT
from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)
from probstat.inference import clopper_pearson_interval

from .estimators import (
    Estimator,
    _log_work_normalized_variance_error,
    summarize,
    summarize_draws,
)

T = TypeVar("T")

VERDICTS = ("supported", "not refuted", "inconclusive", "refuted")


@dataclass(frozen=True, slots=True)
class CheckResult:
    """The outcome of one certification check.

    Args:
        check: Name of the check, e.g. ``"unbiasedness"``.
        claim: The claim in words, with its numerical target.
        estimate: Point estimate of the tested quantity.
        standard_error: Monte Carlo standard error of ``estimate`` (``nan``
            when the interval is not of the Wald type).
        lower: Lower confidence limit.
        upper: Upper confidence limit.
        target: The value the claim asserts for the tested quantity.
        tolerance: Declared half-width of the equivalence band around
            ``target``, or ``nan`` when none was declared.
        confidence: Confidence level of the interval.
        replicates: Number of replicates behind the estimate.
        verdict: One of :data:`VERDICTS`.

    References:
        [REF-SCHUIRMANN-1987] (interval-inclusion logic of equivalence
        testing), [REF-MORRIS-WHITE-CROWTHER-2019], [EQ-MLAB-VERDICT-001].

    """

    check: str
    claim: str
    estimate: float
    standard_error: float
    lower: float
    upper: float
    target: float
    tolerance: float
    confidence: float
    replicates: int
    verdict: str

    def sentence(self) -> str:
        """Return a one-line statement of the evidence and the verdict.

        References:
            Project reporting policy [REF-PROBSTAT-REPORTING]. No equation is
            evaluated.

        """
        band = "" if math.isnan(self.tolerance) else f", tolerance {self.tolerance:.4g}"
        return (
            f"{self.check}: {self.claim} -> {self.verdict} "
            f"(estimate {self.estimate:.6g}, {100 * self.confidence:.4g}% interval "
            f"[{self.lower:.6g}, {self.upper:.6g}], target {self.target:.6g}{band}, "
            f"{self.replicates} replicates)"
        )


def interval_verdict(
    lower: float, upper: float, target: float, tolerance: float | None
) -> str:
    """Return the verdict implied by a confidence interval.

    With tolerance ``delta`` the band is ``[target - delta, target + delta]``:
    ``"supported"`` if the interval lies inside the band, ``"refuted"`` if it
    lies entirely outside, ``"inconclusive"`` otherwise. Without a tolerance:
    ``"refuted"`` if the interval excludes ``target``, else ``"not refuted"``
    ([EQ-MLAB-VERDICT-001]). When the interval has level ``1 - alpha``, a
    wrong ``"refuted"`` has probability at most ``alpha`` and, by the
    interval-inclusion argument of two one-sided tests, so has a wrong
    ``"supported"``. The guarantee is only as good as the interval: it
    holds for the Clopper-Pearson interval of :func:`check_coverage` and,
    for normal replicates, for the t interval of
    :func:`check_unbiasedness`; the delta-method intervals of
    :func:`check_variance`, :func:`check_convergence_rate`, and
    :func:`check_efficiency_gain` reach their level only for many
    replicates and light tails (their docstrings give measured rates).

    Args:
        lower: Lower confidence limit.
        upper: Upper confidence limit, not below ``lower``.
        target: Claimed value.
        tolerance: Non-negative half-width of the equivalence band, or
            ``None``.

    Returns:
        One of :data:`VERDICTS`.

    Raises:
        ValueError: If the limits are reversed or the tolerance is negative.

    References:
        [REF-SCHUIRMANN-1987], [REF-LEHMANN-ROMANO-2005]
        (duality of tests and confidence sets), [EQ-MLAB-VERDICT-001].

    """
    if upper < lower:
        raise ValueError("upper must not be below lower")
    if tolerance is None:
        return "refuted" if (target < lower or target > upper) else "not refuted"
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    low, high = target - tolerance, target + tolerance
    if low <= lower and upper <= high:
        return "supported"
    if upper < low or lower > high:
        return "refuted"
    return "inconclusive"


def _confidence(value: float) -> float:
    """Validate a confidence level strictly between zero and one.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    level = require_finite_real(value, name="confidence")
    if not 0.0 < level < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    return level


def _replicate_count(value: int, minimum: int = 2) -> int:
    """Validate a replicate count.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    count = require_non_negative_integer(value, name="replicates")
    if count < minimum:
        raise ValueError(f"replicates must be at least {minimum}")
    return count


def _wald(
    check: str,
    claim: str,
    estimate: float,
    standard_error: float,
    target: float,
    tolerance: float | None,
    confidence: float,
    replicates: int,
    *,
    degrees_of_freedom: int | None = None,
) -> CheckResult:
    """Build a check result from a Wald interval.

    The critical value is the normal quantile, or the Student-t quantile
    with ``degrees_of_freedom`` when given (the interval of a mean of
    ``R`` replicates with estimated variance uses ``R - 1``).

    References:
        [EQ-MLAB-MCSE-001], [EQ-MLAB-VERDICT-001],
        [REF-MORRIS-WHITE-CROWTHER-2019].

    """
    if degrees_of_freedom is None:
        z = Normal(0.0, 1.0).quantile(0.5 + 0.5 * confidence)
    else:
        z = StudentT(float(degrees_of_freedom)).quantile(0.5 + 0.5 * confidence)
    lower = estimate - z * standard_error
    upper = estimate + z * standard_error
    return CheckResult(
        check,
        claim,
        estimate,
        standard_error,
        lower,
        upper,
        target,
        math.nan if tolerance is None else tolerance,
        confidence,
        replicates,
        interval_verdict(lower, upper, target, tolerance),
    )


def check_unbiasedness(
    estimator: Estimator,
    truth: float,
    *,
    replicates: int,
    seed: int | RandomStream,
    tolerance: float | None = None,
    confidence: float = 0.99,
) -> CheckResult:
    """Test the claim that an estimator is unbiased for a known value.

    The bias estimate is the replicate mean minus ``truth`` with Monte Carlo
    standard error ``s / sqrt(R)`` ([EQ-MLAB-MCSE-001]); the interval is the
    Student-t interval of the mean with ``R - 1`` degrees of freedom. Its
    level is exact for normally distributed replicates and approximate
    otherwise: over 2000 seeds at ``confidence = 0.99`` a true claim was
    refuted in 0.8% to 1.3% of runs for normal replicates (``R = 5`` to
    ``100``), and for exponential replicates in 4.9% (``R = 10``), 4.1%
    (``R = 30``), 1.9% (``R = 100``) and 0.8% (``R = 1000``). The interval
    presumes a finite
    variance. A heavy-tailed estimator (infinite variance, as a
    randomized-truncation estimator with a badly chosen level law has) makes
    it unreliable: also run :func:`check_variance_stability`.

    Args:
        estimator: Estimator under test.
        truth: Exactly known target value (closed form or a reference
            computation whose error is negligible against the tolerance).
        replicates: Number of independent replicates.
        seed: Root seed or stream.
        tolerance: Bias magnitude regarded as negligible; declare it before
            running. ``None`` allows only "refuted" or "not refuted".
        confidence: Confidence level of the interval for the bias.

    Returns:
        Check result whose estimate is the bias.

    Raises:
        ValueError: If ``replicates`` is below two or ``confidence`` is not
            in ``(0, 1)``.

    References:
        [REF-MORRIS-WHITE-CROWTHER-2019, Table 6] (bias and its Monte Carlo
        standard error), [EQ-MLAB-MCSE-001], [EQ-MLAB-VERDICT-001].

    """
    target = require_finite_real(truth, name="truth")
    level = _confidence(confidence)
    summary = summarize(estimator, _replicate_count(replicates), seed)
    return _wald(
        "unbiasedness",
        f"E[{estimator.name}] = {target:.6g} (bias 0)",
        summary.mean - target,
        summary.mean_standard_error,
        0.0,
        tolerance,
        level,
        summary.count,
        degrees_of_freedom=summary.count - 1,
    )


def check_variance(
    estimator: Estimator,
    claimed_variance: float,
    *,
    replicates: int,
    seed: int | RandomStream,
    relative_tolerance: float | None = None,
    confidence: float = 0.99,
) -> CheckResult:
    """Test a claimed variance of an estimator.

    The tested quantity is the ratio ``s^2 / v`` of the replicate variance
    to the claimed variance ``v``, with the distribution-free standard error
    of [EQ-MLAB-MCSE-002] (it uses the sample fourth moment, so no normality
    of the estimator is assumed). The claim is ``ratio = 1``.

    The interval is a large-sample normal approximation and needs a finite
    fourth moment; with few replicates or heavy tails it is too narrow.
    Measured over 2000 seeds at ``confidence = 0.99``, a *true* variance was
    refuted in 5.9%, 2.5%, 1.0% of runs for normal replicates with
    ``R = 30, 100, 1000``; in 19%, 9%, 2.4% for exponential replicates; in
    39%, 26%, 12% for lognormal replicates (4% at ``R = 5000``); and in 14%
    at ``R = 5000`` for Student-t replicates with three degrees of freedom,
    whose fourth moment is infinite. Treat a ``"refuted"`` from a skewed or
    heavy-tailed estimator with fewer than several thousand replicates as
    unconfirmed.

    Args:
        estimator: Estimator under test.
        claimed_variance: Positive variance predicted for one draw.
        replicates: Number of independent replicates (at least four).
        seed: Root seed or stream.
        relative_tolerance: Half-width of the equivalence band for the
            ratio, or ``None``.
        confidence: Confidence level of the interval for the ratio.

    Returns:
        Check result whose estimate is the variance ratio.

    Raises:
        ValueError: If the claimed variance is not positive.

    References:
        [REF-CASELLA-BERGER-2002, Exercise 5.8], [EQ-MLAB-MCSE-002],
        [EQ-MLAB-VERDICT-001].

    """
    claimed = require_finite_real(claimed_variance, name="claimed_variance")
    if claimed <= 0.0:
        raise ValueError("claimed_variance must be positive")
    level = _confidence(confidence)
    summary = summarize(estimator, _replicate_count(replicates, 4), seed)
    return _wald(
        "variance",
        f"Var[{estimator.name}] = {claimed:.6g} (ratio 1)",
        summary.variance / claimed,
        summary.variance_standard_error / claimed,
        1.0,
        relative_tolerance,
        level,
        summary.count,
    )


@dataclass(frozen=True, slots=True)
class VarianceStability:
    """Evidence on whether the variance of an estimator is finite.

    Args:
        block_variances: Sample variance of each successive block of
            replicates.
        largest_share: Largest single squared deviation divided by the sum
            of squared deviations over all replicates.
        growth: Ratio of the last block variance to the median of the
            earlier ones.
        suspicious: Whether ``largest_share`` exceeds the threshold used.

    References:
        [REF-RHEE-GLYNN-2015] (finite-variance condition),
        [EQ-MLAB-STAB-001].

    """

    block_variances: tuple[float, ...]
    largest_share: float
    growth: float
    suspicious: bool


def check_variance_stability(
    estimator: Estimator,
    *,
    replicates: int,
    seed: int | RandomStream,
    blocks: int = 8,
    share_threshold: float = 0.5,
) -> VarianceStability:
    """Look for symptoms of an infinite variance.

    A finite-sample run can never establish that a variance is finite. This
    diagnostic reports two symptoms of an infinite one: block variances that
    do not settle, and one replicate that carries most of the sum of squared
    deviations ([EQ-MLAB-STAB-001]). ``suspicious`` is a heuristic flag, not
    a test with an error probability, and it misses most infinite-variance
    estimators: with the default threshold it was raised in 40% of 300
    seeded runs (1000 replicates) for a single-term estimator whose terms
    ``E[Delta_n^2] / p_n`` grow like ``2.5^n``, in 25% when they grow like
    ``1.25^n``, and in 37% for a Pareto variate with tail index 1.5; 20 000
    replicates did not raise these rates. It was never raised for normal
    replicates or for a finite-variance single-term estimator. An unraised
    flag is therefore no evidence of a finite variance; compare the decay
    rates of the differences and the level law analytically.

    Args:
        estimator: Estimator under test.
        replicates: Number of replicates, at least ``2 * blocks``.
        seed: Root seed or stream.
        blocks: Number of successive blocks (at least two).
        share_threshold: Share of the sum of squares above which a single
            replicate is flagged.

    Returns:
        Block variances, the largest share, the growth ratio, and the flag.

    Raises:
        ValueError: If there are too few replicates or blocks.

    References:
        [REF-RHEE-GLYNN-2015], [REF-OWEN-2013, Chapter 9]
        (diagnosing heavy-tailed weights), [EQ-MLAB-STAB-001].

    """
    block_count = require_non_negative_integer(blocks, name="blocks")
    if block_count < 2:
        raise ValueError("blocks must be at least two")
    count = _replicate_count(replicates, 2 * block_count)
    threshold = require_finite_real(share_threshold, name="share_threshold")
    values = [draw.value for draw in estimator.replicates(count, seed)]
    mean = math.fsum(values) / count
    squares = [(value - mean) ** 2 for value in values]
    total = math.fsum(squares)
    share = max(squares) / total if total > 0.0 else 0.0
    size = count // block_count
    variances: list[float] = []
    for index in range(block_count):
        block = values[index * size : (index + 1) * size]
        centre = math.fsum(block) / len(block)
        variances.append(math.fsum((v - centre) ** 2 for v in block) / (len(block) - 1))
    earlier = sorted(variances[:-1])
    median = earlier[len(earlier) // 2]
    growth = variances[-1] / median if median > 0.0 else math.inf
    return VarianceStability(tuple(variances), share, growth, share > threshold)


IntervalProcedure = Callable[[RandomStream], tuple[float, float]]


def check_coverage(
    name: str,
    procedure: IntervalProcedure,
    truth: float,
    *,
    nominal: float,
    replicates: int,
    seed: int | RandomStream,
    tolerance: float | None = None,
    confidence: float = 0.99,
    conservative_ok: bool = False,
) -> CheckResult:
    """Test the claim that an interval procedure covers at its nominal level.

    Each replicate calls ``procedure`` on its own child stream and records
    whether ``truth`` lies in the returned interval. The coverage estimate
    gets an exact Clopper-Pearson interval ([EQ-INF-CI-CP-001]), so the
    check itself has guaranteed error probability whatever the coverage.

    Args:
        name: Label of the procedure.
        procedure: Maps a stream to ``(lower, upper)``.
        truth: The value the interval is meant to cover.
        nominal: Claimed coverage probability in ``(0, 1)``.
        replicates: Number of independent replicates.
        seed: Root seed or stream; replicate ``i`` uses child
            ``("replicate", i)``.
        tolerance: Half-width of the equivalence band around ``nominal``.
        confidence: Confidence level of the Clopper-Pearson interval.
        conservative_ok: Treat coverage above nominal as satisfying the
            claim "coverage at least nominal": the verdict is then
            ``"refuted"`` only when the whole interval lies below
            ``nominal`` (minus the tolerance, if any), and ``"supported"``
            when it lies entirely at or above that bound.

    Returns:
        Check result whose estimate is the empirical coverage.

    Raises:
        ValueError: If ``nominal`` or ``confidence`` is not in ``(0, 1)``.

    References:
        [REF-CLOPPER-PEARSON-1934], [REF-MORRIS-WHITE-CROWTHER-2019,
        Table 6] (coverage), [EQ-INF-CI-CP-001], [EQ-MLAB-VERDICT-001].

    """
    target = require_finite_real(truth, name="truth")
    level = _confidence(confidence)
    claimed = _confidence(nominal)
    count = _replicate_count(replicates, 1)
    root = resolve_stream(seed)
    hits = 0
    for index in range(count):
        lower, upper = procedure(root.child("replicate", index))
        if lower <= target <= upper:
            hits += 1
    interval = clopper_pearson_interval(hits, count, confidence=level)
    if conservative_ok:
        bound = claimed - (tolerance or 0.0)
        if interval.upper < bound:
            verdict = "refuted"
        elif interval.lower >= bound:
            verdict = "supported"
        else:
            verdict = "inconclusive" if tolerance is not None else "not refuted"
    else:
        verdict = interval_verdict(interval.lower, interval.upper, claimed, tolerance)
    relation = ">=" if conservative_ok else "="
    return CheckResult(
        "coverage",
        f"P(truth in interval of {name}) {relation} {claimed:.4g}",
        hits / count,
        math.sqrt(max(hits * (count - hits), 0) / count**3),
        interval.lower,
        interval.upper,
        claimed,
        math.nan if tolerance is None else tolerance,
        level,
        count,
        verdict,
    )


def check_convergence_rate(
    name: str,
    family: Callable[[int], Estimator],
    sizes: Sequence[int],
    truth: float,
    *,
    expected_rate: float,
    replicates: int,
    seed: int | RandomStream,
    tolerance: float | None = None,
    confidence: float = 0.99,
) -> CheckResult:
    """Test a claimed decay rate of the mean squared error.

    For each size ``n`` the mean squared error against ``truth`` is
    estimated from independent replicates; the rate is the negative
    least-squares slope of ``log MSE`` on ``log n`` ([EQ-MLAB-RATE-001]).
    Its standard error follows from the delta method applied to each
    ``log MSE`` (independent across sizes). A claimed rate ``r`` means
    ``MSE = O(n^-r)``: ``r = 1`` for plain Monte Carlo, ``r = 2`` for a
    root-mean-square error of order ``1/n``. The fit describes the tested
    range of sizes only; pre-asymptotic behaviour biases it. The interval is
    a large-sample approximation (squared errors are skewed): for the mean
    of ``n = 4, 16, 64`` variates, whose rate is exactly one, the true rate
    was refuted at ``confidence = 0.99`` in 4.4%, 2.0%, 1.6% of 2000 seeded
    runs with 10, 30, 100 replicates per size for normal variates and in
    11%, 9%, 7% for lognormal variates.

    Args:
        name: Label of the family.
        family: ``family(n)`` returns the estimator at size ``n``.
        sizes: At least three distinct positive sizes.
        truth: Exactly known target value.
        expected_rate: Claimed rate ``r``.
        replicates: Replicates per size.
        seed: Root seed or stream; size ``n`` uses child ``("size", n)``.
        tolerance: Half-width of the equivalence band for the rate.
        confidence: Confidence level of the interval for the rate.

    Returns:
        Check result whose estimate is the fitted rate.

    Raises:
        ValueError: If fewer than three distinct sizes are given, a size
            repeats, or a mean squared error is zero.

    References:
        [REF-OBERKAMPF-ROY-2010] (observed order of accuracy),
        [REF-MORRIS-WHITE-CROWTHER-2019, Table 6] (MSE and its standard
        error), [EQ-MLAB-RATE-001].

    """
    target = require_finite_real(truth, name="truth")
    claimed = require_finite_real(expected_rate, name="expected_rate")
    level = _confidence(confidence)
    count = _replicate_count(replicates)
    grid = [require_non_negative_integer(n, name="size") for n in sizes]
    if len(set(grid)) < 3 or any(n == 0 for n in grid):
        raise ValueError("at least three distinct positive sizes are required")
    if len(set(grid)) != len(grid):
        # A repeated size would reuse the child stream ("size", n): the same
        # replicates would enter the fit twice as if they were independent.
        raise ValueError("sizes must not repeat")
    root = resolve_stream(seed)
    xs: list[float] = []
    ys: list[float] = []
    variances: list[float] = []
    for size in grid:
        draws = family(size).replicates(count, root.child("size", size))
        errors = [(draw.value - target) ** 2 for draw in draws]
        mse = math.fsum(errors) / count
        if mse <= 0.0:
            raise ValueError(f"mean squared error at size {size} is zero")
        spread = math.fsum((e - mse) ** 2 for e in errors) / (count * (count - 1))
        xs.append(math.log(size))
        ys.append(math.log(mse))
        variances.append(spread / mse**2)
    centre = math.fsum(xs) / len(xs)
    scale = math.fsum((x - centre) ** 2 for x in xs)
    coefficients = [(x - centre) / scale for x in xs]
    slope = math.fsum(c * y for c, y in zip(coefficients, ys, strict=True))
    slope_variance = math.fsum(
        c * c * v for c, v in zip(coefficients, variances, strict=True)
    )
    return _wald(
        "convergence rate",
        f"MSE of {name} = O(n^-{claimed:.4g})",
        -slope,
        math.sqrt(slope_variance),
        claimed,
        tolerance,
        level,
        count,
    )


def check_efficiency_gain(
    candidate: Estimator,
    baseline: Estimator,
    *,
    replicates: int,
    seed: int | RandomStream,
    claimed_gain: float = 1.0,
    confidence: float = 0.99,
) -> CheckResult:
    """Test the claim that a candidate is more efficient than a baseline.

    Efficiency is compared through the work-normalized variance
    ``Var x E[cost]`` ([EQ-MLAB-WNV-001]): the gain is
    ``WNV(baseline) / WNV(candidate)``, the factor by which the candidate
    reduces the computing budget needed for a given accuracy when both are
    (nearly) unbiased. The interval is built on the log scale by the delta
    method from independent replicate sets (the two estimators use the
    child streams ``"candidate"`` and ``"baseline"``). The standard error
    of each ``log WNV`` covers the sample variance ([EQ-MLAB-MCSE-002]),
    the mean cost, and their covariance; it needs finite fourth moments of
    the values and a finite variance of the costs, and it is a
    large-sample approximation (see ADR-0131 for measured error rates: a
    randomized-truncation estimator whose cost has infinite variance makes
    the verdict unreliable).
    The verdict is one-sided: ``"supported"`` when the whole interval
    exceeds ``claimed_gain``, ``"refuted"`` when it lies entirely below,
    ``"inconclusive"`` otherwise. A biased candidate can show a spurious
    gain; check unbiasedness first.

    Args:
        candidate: The new estimator.
        baseline: The reference estimator.
        replicates: Replicates of each estimator (at least four).
        seed: Root seed or stream.
        claimed_gain: Gain the candidate is claimed to exceed (default
            one: "at least as efficient").
        confidence: Confidence level of the interval for the gain.

    Returns:
        Check result whose estimate is the efficiency gain.

    Raises:
        ValueError: If ``claimed_gain`` is not positive or a variance is
            zero.

    References:
        [REF-GLASSERMAN-2003, Section 1.1.3], [EQ-MLAB-WNV-001], [EQ-MLAB-MCSE-002].

    """
    gain = require_finite_real(claimed_gain, name="claimed_gain")
    if gain <= 0.0:
        raise ValueError("claimed_gain must be positive")
    level = _confidence(confidence)
    count = _replicate_count(replicates, 4)
    root = resolve_stream(seed)
    new_draws = candidate.replicates(count, root.child("candidate"))
    old_draws = baseline.replicates(count, root.child("baseline"))
    new = summarize_draws(candidate.name, new_draws)
    old = summarize_draws(baseline.name, old_draws)
    if new.variance <= 0.0 or old.variance <= 0.0:
        raise ValueError("both estimators need a positive replicate variance")
    if new.mean_cost <= 0.0 or old.mean_cost <= 0.0:
        raise ValueError("both estimators need a positive mean cost")
    estimate = old.work_normalized_variance / new.work_normalized_variance
    log_se = math.hypot(
        _log_work_normalized_variance_error(old_draws),
        _log_work_normalized_variance_error(new_draws),
    )
    z = Normal(0.0, 1.0).quantile(0.5 + 0.5 * level)
    lower = estimate * math.exp(-z * log_se)
    upper = estimate * math.exp(z * log_se)
    if lower > gain:
        verdict = "supported"
    elif upper < gain:
        verdict = "refuted"
    else:
        verdict = "inconclusive"
    return CheckResult(
        "efficiency gain",
        f"WNV({baseline.name}) / WNV({candidate.name}) > {gain:.4g}",
        estimate,
        estimate * log_se,
        lower,
        upper,
        gain,
        math.nan,
        level,
        count,
        verdict,
    )


@dataclass(frozen=True, slots=True)
class InvarianceResult:
    """Outcome of a metamorphic relation over generated cases.

    Args:
        relation: The relation in words.
        cases: Number of generated cases.
        violations: Number of cases whose discrepancy exceeded the
            tolerance.
        largest_discrepancy: Largest absolute discrepancy observed.
        tolerance: Absolute tolerance used.

    References:
        [REF-CHEN-ETAL-2018] (metamorphic testing), [EQ-MLAB-META-001].

    """

    relation: str
    cases: int
    violations: int
    largest_discrepancy: float
    tolerance: float

    @property
    def holds(self) -> bool:
        """Return whether no generated case violated the relation.

        References:
            [REF-CHEN-ETAL-2018]. No equation is evaluated.

        """
        return self.violations == 0


def check_invariance(
    relation: str,
    procedure: Callable[[T], float],
    generate: Callable[[RandomStream], T],
    transform_input: Callable[[T], T],
    transform_output: Callable[[float], float],
    *,
    cases: int,
    seed: int | RandomStream,
    tolerance: float = 1e-9,
) -> InvarianceResult:
    """Check a metamorphic relation of a deterministic procedure.

    A metamorphic relation states how the output must change when the input
    is transformed, without knowing the correct output:
    ``procedure(transform_input(x)) = transform_output(procedure(x))``
    ([EQ-MLAB-META-001]). Examples: permutation invariance (identity output
    transform), translation equivariance of a location estimator, scale
    equivariance of a standard error. One violated case refutes the
    relation; any number of passing cases only fails to refute it.

    Args:
        relation: The relation in words, for the report.
        procedure: Deterministic function under test.
        generate: Builds one input case from a stream.
        transform_input: The input transformation.
        transform_output: The implied output transformation.
        cases: Positive number of generated cases.
        seed: Root seed or stream; case ``i`` uses child ``("case", i)``.
        tolerance: Absolute tolerance on the discrepancy, scaled by
            ``max(1, |expected|)``.

    Returns:
        Number of violations and the largest discrepancy.

    Raises:
        ValueError: If ``cases`` is not positive or ``tolerance`` negative.

    References:
        [REF-CHEN-ETAL-2018], [EQ-MLAB-META-001].

    """
    count = _replicate_count(cases, 1)
    bound = require_finite_real(tolerance, name="tolerance")
    if bound < 0.0:
        raise ValueError("tolerance must be non-negative")
    root = resolve_stream(seed)
    violations = 0
    largest = 0.0
    for index in range(count):
        case = generate(root.child("case", index))
        expected = float(transform_output(float(procedure(case))))
        observed = float(procedure(transform_input(case)))
        discrepancy = abs(observed - expected)
        largest = max(largest, discrepancy)
        if discrepancy > bound * max(1.0, abs(expected)):
            violations += 1
    return InvarianceResult(relation, count, violations, largest, bound)


@dataclass(frozen=True, slots=True)
class MethodCard:
    """The claims of one method and the evidence gathered for each.

    Args:
        name: Method label.
        lineage: Derivation steps of the method.
        scope: The data-generating mechanisms and settings the evidence
            covers, in words.
        checks: Results of the certification checks.
        invariances: Results of the metamorphic relations.

    References:
        [REF-MORRIS-WHITE-CROWTHER-2019] (reporting of simulation
        evidence), project reporting policy [REF-PROBSTAT-REPORTING].

    """

    name: str
    lineage: tuple[str, ...]
    scope: str
    checks: tuple[CheckResult, ...]
    invariances: tuple[InvarianceResult, ...] = ()

    @property
    def refuted(self) -> tuple[str, ...]:
        """Return the names of the refuted claims and violated relations.

        References:
            [EQ-MLAB-VERDICT-001]. No further equation is evaluated.

        """
        failed = [c.check for c in self.checks if c.verdict == "refuted"]
        failed.extend(r.relation for r in self.invariances if not r.holds)
        return tuple(failed)

    @property
    def verdict(self) -> str:
        """Return the scoped overall verdict of the card.

        ``"refuted"`` if any claim is refuted or relation violated;
        ``"supported within scope"`` if every check is ``"supported"``;
        otherwise ``"not refuted within scope"``. The card never says
        "validated": the evidence covers the stated scope only.

        References:
            [EQ-MLAB-VERDICT-001], [REF-OBERKAMPF-ROY-2010]
            (scoped validation statements).

        """
        if self.refuted:
            return "refuted"
        if self.checks and all(c.verdict == "supported" for c in self.checks):
            return "supported within scope"
        return "not refuted within scope"

    def render(self) -> str:
        """Return the card as Markdown.

        Returns:
            A heading, the lineage, the scope, one line per check and
            relation, and the scoped verdict.

        References:
            Project reporting policy [REF-PROBSTAT-REPORTING]. No equation is
            evaluated.

        """
        lines = [f"## Method card: {self.name}", ""]
        if self.lineage:
            lines.append("Derivation:")
            lines.extend(f"{i}. {step}" for i, step in enumerate(self.lineage, 1))
        else:
            lines.append("Derivation: base method (no derivation steps).")
        lines.extend(["", f"Scope of the evidence: {self.scope}", "", "Checks:"])
        lines.extend(f"- {check.sentence()}" for check in self.checks)
        for relation in self.invariances:
            state = "holds" if relation.holds else "VIOLATED"
            lines.append(
                f"- relation: {relation.relation} -> {state} on {relation.cases} "
                f"cases (largest discrepancy {relation.largest_discrepancy:.3g})"
            )
        lines.extend(["", f"Verdict: {self.verdict}."])
        return "\n".join(lines)


def method_card(
    estimator: Estimator,
    scope: str,
    checks: Sequence[CheckResult],
    invariances: Sequence[InvarianceResult] = (),
) -> MethodCard:
    """Assemble the method card of an estimator.

    Args:
        estimator: The certified estimator; its name and lineage are copied.
        scope: What the evidence covers (mechanisms, sizes, seeds).
        checks: Certification check results.
        invariances: Metamorphic relation results.

    Returns:
        The card.

    Raises:
        ValueError: If ``scope`` is empty.

    References:
        [REF-MORRIS-WHITE-CROWTHER-2019], project reporting policy
        [REF-PROBSTAT-REPORTING].

    """
    if not isinstance(scope, str) or not scope.strip():
        raise ValueError("scope must describe what the evidence covers")
    return MethodCard(
        estimator.name, estimator.lineage, scope, tuple(checks), tuple(invariances)
    )
