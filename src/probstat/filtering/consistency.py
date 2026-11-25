"""Consistency of a filter's covariance with its actual errors.

A filter reports an estimate and a covariance. The covariance is *consistent*
when the actual errors are distributed as the covariance says. Two standard
chi-square checks, both from the estimation literature on tracking, and one
distributional check for a set of (error, covariance) pairs are provided:

- **NEES** (normalised estimation error squared), ``e' P^-1 e`` for the state
  error ``e = x - x_hat`` and the reported covariance ``P``. It needs the
  truth, hence simulation or a reference solution. Over ``N`` independent
  runs the average NEES at one time step satisfies ``N * mean ~ chi2(N n)``
  when the filter is consistent, which gives two-sided acceptance bounds.
- **NIS** (normalised innovation squared), ``v' S^-1 v`` for the innovation
  ``v`` and its covariance ``S``. It needs no truth. For one run with
  white innovations, the time average over ``T`` steps satisfies
  ``T * mean ~ chi2(T m)``. A sample-autocorrelation whiteness test of the
  standardised innovations checks the assumption behind it.
- **Covariance realism** of ``N`` pairs: the squared Mahalanobis distances
  ``d^2 = e' P^-1 e`` should be ``chi2(n)``. A Kolmogorov-Smirnov, a
  Cramer-von Mises and an Anderson-Darling test (:mod:`probstat.inference`)
  judge the shape, and
  the covariance scale factor ``mean(d^2) / n`` with a chi-square
  confidence interval says by how much ``P`` should be multiplied. The
  procedure of the NASA conjunction assessment handbook (Appendix I) is the
  same in outline: squared Mahalanobis distances compared with a chi-square
  law (three degrees of freedom for a position), their mean as a first-moment
  check, and empirical-distribution-function tests, for which the handbook
  recommends the Cramer-von Mises statistic over Anderson-Darling because
  outliers are frequent in practice; ``covariance_realism`` computes all
  three p-values and keeps Anderson-Darling as the default of the reported
  one (``test="cramer-von-mises"`` selects the handbook's choice), and the
  scale factor of the handbook is read from the normalised mean without a
  stated interval.

Verdicts use three words. ``"optimistic"``: the covariance is too small (the
statistic is above its upper bound); ``"pessimistic"``: the covariance is too
large (below its lower bound); ``"consistent"``: neither.

Assumptions: errors are Gaussian with the stated covariance under the null
hypothesis; NEES runs are independent and share the model; for the NEES
verdict see :func:`nees_test` (the bounds of one step are applied to the time
mean, which is conservative because steps are positively dependent); NIS and
whiteness need white innovations (a correct filter). The NIS and NEES tests
detect an inconsistent covariance, they do not locate the cause: a halved
process-noise covariance and a halved measurement-noise covariance can both
make a filter optimistic.

References:
    [REF-BAR-SHALOM-LI-KIRUBARAJAN-2001], [REF-HORWOOD-ETAL-2014],
    [REF-NASA-CARA-HANDBOOK], [REF-MARDIA-KENT-BIBBY-1979],
    [REF-LJUNG-BOX-1978].

"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from probstat.distributions import ChiSquare, Normal
from probstat.foundations.validation import require_finite_real
from probstat.inference import (
    anderson_darling_test,
    cramer_von_mises_test,
    kolmogorov_smirnov_test,
)
from probstat.linalg import Matrix, Vector, as_vector, cholesky, solve_lower, symmetrize

from .kalman import KalmanFilterResult

VERDICTS = ("consistent", "optimistic", "pessimistic")
"""The three verdicts of a consistency test."""

REALISM_TESTS = ("anderson-darling", "kolmogorov-smirnov", "cramer-von-mises")
"""Goodness-of-fit tests available to :func:`covariance_realism`."""


@dataclass(frozen=True, slots=True)
class NeesResult:
    """Monte-Carlo average NEES with its acceptance bounds.

    Args:
        statistic: Average NEES over the runs at each time step.
        lower: Lower bound ``chi2_(alpha/2)(N n) / N``.
        upper: Upper bound ``chi2_(1-alpha/2)(N n) / N``.
        degrees_of_freedom: ``N n`` for every step.
        runs: Number of runs ``N``.
        dimension: State dimension ``n``.
        confidence: Two-sided acceptance probability ``1 - alpha``.
        mean_statistic: Mean of ``statistic`` over the time steps.
        fraction_below: Share of steps with ``statistic < lower``.
        fraction_above: Share of steps with ``statistic > upper``.
        verdict: ``"consistent"``, ``"optimistic"`` or ``"pessimistic"``,
            decided on ``mean_statistic`` against ``[lower, upper]``.

    References:
        [REF-BAR-SHALOM-LI-KIRUBARAJAN-2001].

    """

    statistic: tuple[float, ...]
    lower: float
    upper: float
    degrees_of_freedom: int
    runs: int
    dimension: int
    confidence: float
    mean_statistic: float
    fraction_below: float
    fraction_above: float
    verdict: str


@dataclass(frozen=True, slots=True)
class NisResult:
    """Time-average NIS of one run with its acceptance bounds.

    Args:
        statistic: Mean NIS over the steps with an observation.
        lower: Lower bound ``chi2_(alpha/2)(D) / T``.
        upper: Upper bound ``chi2_(1-alpha/2)(D) / T``.
        degrees_of_freedom: ``D``, the total innovation dimension over the
            ``T`` used steps.
        steps: Number of steps ``T`` with an observation.
        confidence: Two-sided acceptance probability.
        per_step: NIS of every step (``None`` for a missing observation).
        verdict: ``"consistent"``, ``"optimistic"`` or ``"pessimistic"``.

    References:
        [REF-BAR-SHALOM-LI-KIRUBARAJAN-2001].

    """

    statistic: float
    lower: float
    upper: float
    degrees_of_freedom: int
    steps: int
    confidence: float
    per_step: tuple[float | None, ...]
    verdict: str


@dataclass(frozen=True, slots=True)
class WhitenessResult:
    """Sample autocorrelation of standardised innovations.

    Args:
        lags: Lags ``1..L`` examined.
        autocorrelation: Sample autocorrelation at each lag.
        bound: Large-sample two-sided bound ``z / sqrt(T m)`` for one lag.
        outside: Number of lags whose autocorrelation exceeds the bound.
        statistic: Portmanteau statistic
            ``Q = T m (T + 2) sum_tau rho(tau)^2 / (T - tau)``.
        degrees_of_freedom: ``L``, the number of lags.
        p_value: ``P(chi2_L >= Q)``.
        steps: Number ``T`` of steps with an observation.
        dimension: Innovation dimension ``m``.
        confidence: Two-sided probability of the per-lag bound and one minus
            the level of the portmanteau test.
        white: Whether ``p_value >= 1 - confidence``.

    References:
        [REF-BAR-SHALOM-LI-KIRUBARAJAN-2001].

    """

    lags: tuple[int, ...]
    autocorrelation: tuple[float, ...]
    bound: float
    outside: int
    statistic: float
    degrees_of_freedom: int
    p_value: float
    steps: int
    dimension: int
    confidence: float
    white: bool


@dataclass(frozen=True, slots=True)
class CovarianceRealismResult:
    """Distributional check of (error, covariance) pairs.

    Args:
        count: Number ``N`` of pairs.
        dimension: Error dimension ``n``.
        squared_distances: ``d_i^2 = e_i' P_i^-1 e_i``.
        scale_factor: ``mean(d^2) / n``: the factor by which the covariances
            should be multiplied (above one: they are too small).
        scale_interval: Two-sided chi-square confidence interval for the
            scale factor, ``[N n s / chi2_(1-alpha/2), N n s / chi2_(alpha/2)]``
            with ``N n`` degrees of freedom (independent pairs).
        confidence: Confidence level.
        test: ``"anderson-darling"``, ``"kolmogorov-smirnov"`` or
            ``"cramer-von-mises"``; the test that produced ``p_value``.
        p_value: Tail probability of the chosen test against ``chi2(n)``.
        ks_p_value: Kolmogorov-Smirnov p-value.
        ad_p_value: Anderson-Darling p-value (``None`` if not computed).
        cvm_p_value: Cramer-von Mises p-value (``None`` if not computed).
        shape_consistent: ``p_value >= 1 - confidence``.
        verdict: ``"consistent"`` when the scale interval contains one,
            ``"optimistic"`` when it lies above one, ``"pessimistic"`` when
            below.

    References:
        [REF-HORWOOD-ETAL-2014], [REF-BAR-SHALOM-LI-KIRUBARAJAN-2001].

    """

    count: int
    dimension: int
    squared_distances: tuple[float, ...]
    scale_factor: float
    scale_interval: tuple[float, float]
    confidence: float
    test: str
    p_value: float
    ks_p_value: float
    ad_p_value: float | None
    shape_consistent: bool
    verdict: str
    cvm_p_value: float | None = None


def _confidence_level(confidence: float) -> float:
    """Return a confidence strictly inside ``(0, 1)``.

    Raises:
        TypeError: If the value is not real.
        ValueError: If it is outside ``(0, 1)``.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    level = require_finite_real(confidence, name="confidence")
    if not 0.0 < level < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    return level


def _covariance_factor(covariance: Iterable[Iterable[float]], size: int) -> Matrix:
    """Return the Cholesky factor of a validated covariance of order ``size``.

    Raises:
        ValueError: If the shape is wrong or the matrix is not symmetric
            positive definite.

    References:
        [REF-MARDIA-KENT-BIBBY-1979]. No equation is evaluated.

    """
    matrix = symmetrize(covariance, name="covariance")
    if len(matrix) != size:
        raise ValueError("covariance must be square with the error dimension")
    return cholesky(matrix)


def nees(error: Iterable[float], covariance: Iterable[Iterable[float]]) -> float:
    """Return the normalised squared error ``e' P^-1 e``.

    Args:
        error: Error vector ``e = x - x_hat`` (finite reals).
        covariance: Symmetric positive-definite covariance ``P``.

    Returns:
        The squared Mahalanobis length of ``e``; ``chi2(n)`` under
        consistency. The same function gives the NIS for an innovation and
        its covariance.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If the error is empty, shapes disagree, or ``P`` is not
            symmetric positive definite.

    Equation:
        [EQ-FILT-NEES-001].

    References:
        [REF-BAR-SHALOM-LI-KIRUBARAJAN-2001],
        [REF-MARDIA-KENT-BIBBY-1979].

    """
    vector = as_vector(error, name="error")
    if not vector:
        raise ValueError("error must contain at least one component")
    factor = _covariance_factor(covariance, len(vector))
    whitened = solve_lower(factor, vector)
    return math.fsum(v * v for v in whitened)


def nees_series(
    errors: Sequence[Iterable[float]],
    covariances: Sequence[Iterable[Iterable[float]]],
) -> tuple[float, ...]:
    """Return the NEES of every (error, covariance) pair of one run.

    Args:
        errors: One error vector per time step.
        covariances: One covariance per time step.

    Returns:
        The NEES series.

    Raises:
        ValueError: If the sequences are empty or of different length, or an
            entry is invalid (see :func:`nees`).

    Equation:
        [EQ-FILT-NEES-001].

    References:
        [REF-BAR-SHALOM-LI-KIRUBARAJAN-2001].

    """
    if len(errors) != len(covariances):
        raise ValueError("errors and covariances must have the same length")
    if not errors:
        raise ValueError("at least one step is required")
    return tuple(nees(e, c) for e, c in zip(errors, covariances, strict=True))


def nees_from_filter(
    result: KalmanFilterResult,
    truth: Sequence[Iterable[float]],
    *,
    states: str = "filtered",
) -> tuple[float, ...]:
    """Return the NEES series of a filter run against the true states.

    Args:
        result: Output of a Gaussian filter (for example
            :func:`~probstat.filtering.kalman_filter`).
        truth: True states aligned with ``result.filtered`` (``T + 1``
            entries, index 0 the initial state) or with ``result.predicted``
            (``T`` entries).
        states: ``"filtered"`` or ``"predicted"``.

    Returns:
        The NEES at every step of the chosen sequence.

    Raises:
        ValueError: If ``states`` is unknown or ``truth`` has the wrong
            length.

    Equation:
        [EQ-FILT-NEES-001].

    References:
        [REF-BAR-SHALOM-LI-KIRUBARAJAN-2001].

    """
    if states == "filtered":
        chosen = result.filtered
    elif states == "predicted":
        chosen = result.predicted
    else:
        raise ValueError("states must be 'filtered' or 'predicted'")
    if len(truth) != len(chosen):
        raise ValueError(
            f"truth has {len(truth)} entries; the {states} sequence has {len(chosen)}"
        )
    errors = tuple(
        tuple(t - m for t, m in zip(as_vector(x, name="truth"), s.mean, strict=True))
        for x, s in zip(truth, chosen, strict=True)
    )
    return nees_series(errors, tuple(s.covariance for s in chosen))


def _verdict(statistic: float, lower: float, upper: float) -> str:
    """Return the verdict of a statistic against two-sided bounds.

    References:
        [REF-BAR-SHALOM-LI-KIRUBARAJAN-2001]. No equation is
        evaluated.

    """
    if statistic > upper:
        return "optimistic"
    if statistic < lower:
        return "pessimistic"
    return "consistent"


def nees_test(
    runs: Sequence[Sequence[float]], *, dimension: int, confidence: float = 0.95
) -> NeesResult:
    """Test the Monte-Carlo average NEES against chi-square bounds.

    For ``N`` independent runs, ``N`` times the average NEES at a step is
    ``chi2(N n)`` under consistency, so the acceptance interval of the
    average is ``[chi2_(alpha/2)(N n)/N, chi2_(1-alpha/2)(N n)/N]``. These
    bounds are exact per step. The verdict compares the time mean of the
    averages with the same interval; because NEES values at neighbouring
    steps are positively dependent, the variance of that mean does not
    exceed the per-step variance, so the interval is not too narrow. The
    fractions of steps below and above the bounds are reported as
    diagnostics (about ``1 - confidence`` of them in total if consistent,
    more when the steps are dependent).

    Args:
        runs: ``N`` series of per-step NEES values, all of the same length.
        dimension: State dimension ``n`` (the degrees of freedom of one NEES).
        confidence: Two-sided acceptance probability.

    Returns:
        Averages, bounds, degrees of freedom, and the verdict.

    Raises:
        TypeError: If ``dimension`` is not an integer.
        ValueError: If there is no run, the runs differ in length, a value is
            negative, ``dimension < 1``, or ``confidence`` is outside
            ``(0, 1)``.

    Equation:
        [EQ-FILT-NEES-002].

    References:
        [REF-BAR-SHALOM-LI-KIRUBARAJAN-2001].

    """
    if isinstance(dimension, bool) or not isinstance(dimension, int):
        raise TypeError("dimension must be an integer")
    if dimension < 1:
        raise ValueError("dimension must be at least one")
    level = _confidence_level(confidence)
    table = tuple(as_vector(run, name="run") for run in runs)
    if not table:
        raise ValueError("at least one run is required")
    steps = len(table[0])
    if steps == 0 or any(len(run) != steps for run in table):
        raise ValueError("runs must be non-empty and of the same length")
    if any(v < 0.0 for run in table for v in run):
        raise ValueError("NEES values must be non-negative")
    count = len(table)
    dof = count * dimension
    law = ChiSquare(float(dof))
    tail = 0.5 * (1.0 - level)
    lower = float(law.quantile(tail)) / count
    upper = float(law.quantile(1.0 - tail)) / count
    average = tuple(math.fsum(run[k] for run in table) / count for k in range(steps))
    mean = math.fsum(average) / steps
    return NeesResult(
        average,
        lower,
        upper,
        dof,
        count,
        dimension,
        level,
        mean,
        sum(1 for a in average if a < lower) / steps,
        sum(1 for a in average if a > upper) / steps,
        _verdict(mean, lower, upper),
    )


def nees_test_from_filters(
    results: Sequence[KalmanFilterResult],
    truths: Sequence[Sequence[Iterable[float]]],
    *,
    confidence: float = 0.95,
    states: str = "filtered",
) -> NeesResult:
    """Run :func:`nees_test` on independent filter runs and their truths.

    Args:
        results: One filter result per Monte-Carlo run.
        truths: The true state sequence of each run (see
            :func:`nees_from_filter`).
        confidence: Two-sided acceptance probability.
        states: ``"filtered"`` or ``"predicted"``.

    Returns:
        The Monte-Carlo NEES test.

    Raises:
        ValueError: If the numbers of results and truths differ, no run is
            given, or an input is invalid (see :func:`nees_from_filter`).

    Equation:
        [EQ-FILT-NEES-002].

    References:
        [REF-BAR-SHALOM-LI-KIRUBARAJAN-2001].

    """
    if len(results) != len(truths):
        raise ValueError("results and truths must have the same length")
    if not results:
        raise ValueError("at least one run is required")
    series = tuple(
        nees_from_filter(r, t, states=states)
        for r, t in zip(results, truths, strict=True)
    )
    return nees_test(
        series, dimension=len(results[0].filtered[0].mean), confidence=confidence
    )


def nis_series(
    innovations: Sequence[Iterable[float] | None],
    covariances: Sequence[Iterable[Iterable[float]] | None],
) -> tuple[float | None, ...]:
    """Return the NIS of every step; ``None`` marks a missing observation.

    Args:
        innovations: Innovation vector per step, or ``None``.
        covariances: Innovation covariance per step, or ``None``.

    Returns:
        ``v' S^-1 v`` per step, ``None`` where either entry is ``None``.

    Raises:
        ValueError: If the lengths differ, a step has only one of the two
            entries, or an entry is invalid (see :func:`nees`).

    Equation:
        [EQ-FILT-NEES-001].

    References:
        [REF-BAR-SHALOM-LI-KIRUBARAJAN-2001].

    """
    if len(innovations) != len(covariances):
        raise ValueError("innovations and covariances must have the same length")
    values: list[float | None] = []
    for v, s in zip(innovations, covariances, strict=True):
        if v is None and s is None:
            values.append(None)
        elif v is None or s is None:
            raise ValueError("an innovation and its covariance must both be present")
        else:
            values.append(nees(v, s))
    return tuple(values)


def nis_test(
    innovations: Sequence[Iterable[float] | None],
    covariances: Sequence[Iterable[Iterable[float]] | None],
    *,
    confidence: float = 0.95,
) -> NisResult:
    """Test the time-average NIS of one run against chi-square bounds.

    With ``T`` observed steps of innovation dimension ``m`` the sum of NIS
    values is ``chi2(T m)`` when the filter is consistent (white innovations
    with the stated covariances), so the mean has the acceptance interval
    ``[chi2_(alpha/2)(D)/T, chi2_(1-alpha/2)(D)/T]``, ``D`` the total
    dimension. A mean above the interval means the innovation covariance is
    too small ("optimistic"). Check whiteness with
    :func:`innovation_whiteness_test`.

    Args:
        innovations: Innovation vector per step, or ``None``.
        covariances: Innovation covariance per step, or ``None``.
        confidence: Two-sided acceptance probability.

    Returns:
        Mean NIS, bounds, degrees of freedom, per-step values, and verdict.

    Raises:
        ValueError: If no step has an observation, or an input is invalid
            (see :func:`nis_series`).

    Equation:
        [EQ-FILT-NIS-001].

    References:
        [REF-BAR-SHALOM-LI-KIRUBARAJAN-2001].

    """
    level = _confidence_level(confidence)
    per_step = nis_series(innovations, covariances)
    used = [(i, v) for i, v in enumerate(per_step) if v is not None]
    if not used:
        raise ValueError("at least one step with an observation is required")
    dof = sum(len(as_vector(innovations[i] or (), name="innovation")) for i, _ in used)
    steps = len(used)
    law = ChiSquare(float(dof))
    tail = 0.5 * (1.0 - level)
    lower = float(law.quantile(tail)) / steps
    upper = float(law.quantile(1.0 - tail)) / steps
    mean = math.fsum(v for _, v in used if v is not None) / steps
    return NisResult(
        mean, lower, upper, dof, steps, level, per_step, _verdict(mean, lower, upper)
    )


def nis_test_from_filter(
    result: KalmanFilterResult, *, confidence: float = 0.95, burn_in: int = 0
) -> NisResult:
    """Run :func:`nis_test` on the innovations stored in a filter result.

    Args:
        result: Output of a Gaussian filter; its ``innovations`` and
            ``innovation_covariances`` are used.
        confidence: Two-sided acceptance probability.
        burn_in: Number of leading steps to drop.

    Returns:
        The time-average NIS test.

    Raises:
        ValueError: If ``burn_in`` is negative or leaves no observed step.

    Equation:
        [EQ-FILT-NIS-001].

    References:
        [REF-BAR-SHALOM-LI-KIRUBARAJAN-2001].

    """
    skip = _burn_in(burn_in)
    return nis_test(
        result.innovations[skip:],
        result.innovation_covariances[skip:],
        confidence=confidence,
    )


def _burn_in(burn_in: int) -> int:
    """Return a non-negative integer burn-in.

    Raises:
        TypeError: If the value is not an integer.
        ValueError: If it is negative.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    if isinstance(burn_in, bool) or not isinstance(burn_in, int):
        raise TypeError("burn_in must be an integer")
    if burn_in < 0:
        raise ValueError("burn_in must be non-negative")
    return burn_in


def innovation_whiteness_test(
    innovations: Sequence[Iterable[float] | None],
    covariances: Sequence[Iterable[Iterable[float]] | None],
    *,
    max_lag: int = 20,
    confidence: float = 0.95,
) -> WhitenessResult:
    """Test whether standardised innovations are serially uncorrelated.

    With ``L`` the Cholesky factor of each innovation covariance, the
    standardised innovation ``z_k = L^-1 v_k`` has identity covariance for a
    consistent filter, and ``z_k, z_(k+tau)`` are uncorrelated for white
    innovations. The sample autocorrelation is

    ``rho(tau) = sum_k z_k' z_(k+tau) / sum_k z_k' z_k``

    over the pairs of observed steps (a missing step breaks the pair). For
    ``T`` observed steps of dimension ``m`` it is approximately
    ``N(0, 1/(T m))``, giving the per-lag bound ``z_((1+c)/2) / sqrt(T m)``
    that is reported with the number of lags outside it. The verdict uses the
    portmanteau statistic ``Q = T m (T + 2) sum_tau rho(tau)^2 / (T - tau)``,
    approximately ``chi2(L)`` under whiteness (the Ljung-Box weights applied
    to the pooled standardised innovations; with missing steps ``T - tau`` is
    approximate), because counting lags outside a per-lag bound has little
    power against a few large correlations.

    Args:
        innovations: Innovation vector per step, or ``None``.
        covariances: Innovation covariance per step, or ``None``.
        max_lag: Largest lag ``L``.
        confidence: Two-sided probability of the per-lag bound.

    Returns:
        Autocorrelations, the bound, the count outside, the portmanteau
        statistic with its p-value, and the verdict.

    Raises:
        TypeError: If ``max_lag`` is not an integer.
        ValueError: If ``max_lag < 1``, fewer than ``max_lag + 2`` steps are
            observed, the innovation dimension varies, or an input is
            invalid.

    Equation:
        [EQ-FILT-WHITE-001].

    References:
        [REF-BAR-SHALOM-LI-KIRUBARAJAN-2001],
        [REF-LJUNG-BOX-1978].

    """
    if isinstance(max_lag, bool) or not isinstance(max_lag, int):
        raise TypeError("max_lag must be an integer")
    if max_lag < 1:
        raise ValueError("max_lag must be at least one")
    level = _confidence_level(confidence)
    if len(innovations) != len(covariances):
        raise ValueError("innovations and covariances must have the same length")
    standardised: list[Vector | None] = []
    width: int | None = None
    for v, s in zip(innovations, covariances, strict=True):
        if v is None and s is None:
            standardised.append(None)
            continue
        if v is None or s is None:
            raise ValueError("an innovation and its covariance must both be present")
        vector = as_vector(v, name="innovation")
        if width is None:
            width = len(vector)
        elif len(vector) != width:
            raise ValueError("the innovation dimension must not vary")
        standardised.append(solve_lower(_covariance_factor(s, len(vector)), vector))
    observed = [z for z in standardised if z is not None]
    if width is None or len(observed) < max_lag + 2:
        raise ValueError("too few observed steps for the requested lag")
    energy = math.fsum(v * v for z in observed for v in z)
    if energy == 0.0:
        raise ValueError("the standardised innovations are identically zero")
    values: list[float] = []
    for lag in range(1, max_lag + 1):
        cross = math.fsum(
            a * b
            for first, second in zip(
                standardised[:-lag], standardised[lag:], strict=True
            )
            if first is not None and second is not None
            for a, b in zip(first, second, strict=True)
        )
        values.append(cross / energy)
    steps = len(observed)
    bound = float(Normal().quantile(0.5 * (1.0 + level))) / math.sqrt(steps * width)
    outside = sum(1 for r in values if abs(r) > bound)
    portmanteau = (
        steps
        * width
        * (steps + 2)
        * math.fsum(r * r / (steps - lag) for lag, r in enumerate(values, start=1))
    )
    p_value = float(ChiSquare(float(max_lag)).sf(portmanteau))
    return WhitenessResult(
        tuple(range(1, max_lag + 1)),
        tuple(values),
        bound,
        outside,
        portmanteau,
        max_lag,
        p_value,
        steps,
        width,
        level,
        p_value >= 1.0 - level,
    )


def covariance_realism(
    errors: Sequence[Iterable[float]],
    covariances: Sequence[Iterable[Iterable[float]]],
    *,
    confidence: float = 0.95,
    test: str = "anderson-darling",
) -> CovarianceRealismResult:
    """Check that errors are distributed as their covariances say.

    For each pair the squared Mahalanobis distance ``d^2 = e' P^-1 e`` is
    ``chi2(n)`` when ``e ~ N(0, P)``. The function (1) tests the sample of
    ``d^2`` against ``chi2(n)`` with the Kolmogorov-Smirnov, Cramer-von Mises
    and Anderson-Darling tests of :mod:`probstat.inference` (fully specified
    null: ``chi2(n)`` has no estimated parameter; all three p-values are
    reported). The Anderson-Darling statistic weights the tails, which matter
    for a collision-probability use of the covariance, but is also the more
    outlier-sensitive of the two quadratic statistics; the NASA handbook
    prefers the Cramer-von Mises statistic for that reason, and it is
    available as ``test="cramer-von-mises"``. The default stays
    Anderson-Darling so that existing callers keep their verdict
    (ADR-0140). (2) The function estimates the scale
    factor ``s = mean(d^2) / n`` (a factor on the covariance, the square of
    a factor on standard deviations). The interval assumes independent pairs;
    the NASA handbook advises one comparison point per propagation state and
    successive ephemerides that share less than half of their generating
    data. If the true error covariance is ``c P``,
    then ``N n s / c ~ chi2(N n)`` for independent pairs, which gives the
    confidence interval of ``c``. The scale factor is a one-number
    correction (``P -> s P``); it cannot repair a wrong shape, which the
    goodness-of-fit p-value flags.

    Args:
        errors: ``N`` error vectors of equal dimension ``n``.
        covariances: The ``N`` covariances the errors should follow.
        confidence: Confidence of the scale interval and level of the shape
            judgement.
        test: ``"anderson-darling"``, ``"kolmogorov-smirnov"`` or
            ``"cramer-von-mises"``: which p-value is reported as ``p_value``
            and judged by ``shape_consistent``.

    Returns:
        Squared distances, scale factor with interval, p-values and verdict.

    Raises:
        ValueError: If the sequences are empty or of different length,
            ``test`` is unknown, or an entry is invalid (see :func:`nees`).

    Equation:
        [EQ-FILT-REALISM-001], [EQ-INF-GOF-CVM-004].

    References:
        [REF-HORWOOD-ETAL-2014], [REF-NASA-CARA-HANDBOOK, Appendix I],
        [REF-BAR-SHALOM-LI-KIRUBARAJAN-2001], [REF-MARDIA-KENT-BIBBY-1979].

    """
    level = _confidence_level(confidence)
    if test not in REALISM_TESTS:
        raise ValueError(f"test must be one of {REALISM_TESTS}")
    vectors = tuple(as_vector(e, name="error") for e in errors)
    distances = nees_series(vectors, covariances)
    dimension = len(vectors[0])
    count = len(distances)
    law = ChiSquare(float(dimension))
    ks = kolmogorov_smirnov_test(distances, law)
    ad = anderson_darling_test(distances, law)
    cvm = cramer_von_mises_test(distances, law)
    ks_p = ks.p_value
    ad_p = ad.p_value
    cvm_p = cvm.p_value
    chosen = {
        "anderson-darling": ad_p,
        "kolmogorov-smirnov": ks_p,
        "cramer-von-mises": cvm_p,
    }[test]
    p_value = 0.0 if chosen is None else chosen
    dof = count * dimension
    total = math.fsum(distances)
    interval_law = ChiSquare(float(dof))
    tail = 0.5 * (1.0 - level)
    scale_interval = (
        total / float(interval_law.quantile(1.0 - tail)),
        total / float(interval_law.quantile(tail)),
    )
    scale = total / dimension / count
    if scale_interval[0] > 1.0:
        verdict = "optimistic"
    elif scale_interval[1] < 1.0:
        verdict = "pessimistic"
    else:
        verdict = "consistent"
    return CovarianceRealismResult(
        count,
        dimension,
        distances,
        scale,
        scale_interval,
        level,
        test,
        p_value,
        0.0 if ks_p is None else ks_p,
        ad_p,
        p_value >= 1.0 - level,
        verdict,
        cvm_p,
    )


__all__ = [
    "REALISM_TESTS",
    "VERDICTS",
    "CovarianceRealismResult",
    "NeesResult",
    "NisResult",
    "WhitenessResult",
    "covariance_realism",
    "innovation_whiteness_test",
    "nees",
    "nees_from_filter",
    "nees_series",
    "nees_test",
    "nees_test_from_filters",
    "nis_series",
    "nis_test",
    "nis_test_from_filter",
]
