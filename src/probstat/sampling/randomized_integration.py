"""Randomized quasi-Monte Carlo integration on the unit cube.

Each estimator evaluates the integrand on ``R`` independently randomized
copies of an ``n``-point rule and combines the ``R`` equal-weight averages
``Q_1, ..., Q_R``; the cost is always ``R n`` evaluations.

* ``rqmc_mean`` reports their mean with the standard error from their sample
  variance and a Student-t interval. The mean is unbiased whenever every
  randomized point is uniform on the cube. The interval is approximate: ``R``
  is small and the law of a replicate is in general not normal.
* ``rqmc_median`` reports their median. The median discards the few
  replicates on which a randomization does badly, which is what lets a
  median of randomized rules reach the accuracy the integrand's smoothness
  allows without that smoothness being specified. It comes without a
  confidence interval.
* ``median_random_lattice`` is the lattice form in which nothing is
  constructed: each replicate draws its generating vector uniformly from
  ``{1, ..., n - 1}^s`` (prime ``n``). Most random vectors are good and a few
  are very bad, so the median is accurate where the mean of the same
  replicates is not.

The rules available to the first two estimators are a randomly shifted
rank-1 lattice (``"lattice"``), the Sobol net with Owen's nested scrambling
or a digital shift (``"sobol-owen"``, ``"sobol-shift"``), and the Halton
points with random digit permutations (``"halton"``).
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from probstat.distributions.student_t import StudentT
from probstat.foundations.random_streams import RandomStream, resolve_stream

from .designs import SamplingDesign, halton, sobol
from .lattice import (
    _generating_vector,
    _require_positive_integer,
    _require_prime_size,
    cbc_generating_vector,
    rank1_lattice,
)

Integrand = Callable[[tuple[float, ...]], float]

RQMC_METHODS = ("lattice", "sobol-owen", "sobol-shift", "halton")

_NO_INTERVAL = (
    "no interval: the median of the replicates estimates the median of the "
    "replicate distribution, which equals the integral only if that "
    "distribution is symmetric about it; no interval with guaranteed "
    "coverage of the integral is provided"
)


@dataclass(frozen=True, slots=True)
class RQMCResult:
    """An integral estimate from independently randomized quasi-Monte Carlo rules.

    Args:
        estimate: Mean or median of the replicate estimates.
        estimator: ``"mean"`` or ``"median"``.
        method: Rule used: ``"lattice"``, ``"sobol-owen"``, ``"sobol-shift"``,
            ``"halton"``, or ``"random-lattice"``.
        replicate_estimates: The ``R`` equal-weight averages, in replicate
            order.
        standard_error: Standard error of the mean estimator,
            ``s / sqrt(R)`` with ``s`` the sample standard deviation of the
            replicates; ``None`` for the median, for which no standard error
            is provided.
        interval: Student-t interval of the mean estimator; ``None`` for the
            median.
        confidence: Nominal level of ``interval``; ``None`` for the median.
        interval_method: What the interval is and how far to trust it, or why
            there is none.
        replicate_standard_deviation: Sample standard deviation of the
            replicate estimates. For the median this describes the spread of
            the replicates (outliers included) and is not an error estimate
            of the median.
        replicates: Number of independent randomizations ``R``.
        size: Points per replicate ``n``.
        dimension: Dimension ``s`` of the cube.
        evaluations: Integrand evaluations, ``R n``.
        generating_vectors: Lattice generating vector of every replicate
            (one shared vector repeated for ``"lattice"``; empty for nets and
            Halton points).
        seed: Seed of the root randomization stream.

    References:
        [EQ-SAMP-RQMC-002], [EQ-SAMP-RQMC-003], [EQ-SAMP-RQMC-004].

    """

    estimate: float
    estimator: str
    method: str
    replicate_estimates: tuple[float, ...]
    standard_error: float | None
    interval: tuple[float, float] | None
    confidence: float | None
    interval_method: str
    replicate_standard_deviation: float
    replicates: int
    size: int
    dimension: int
    evaluations: int
    generating_vectors: tuple[tuple[int, ...], ...]
    seed: int

    def summary(self) -> str:
        """Return a one-sentence description of the estimate and its error.

        Returns:
            Text with the estimate, cost, and the interval or the reason why
            none is given.

        References:
            [EQ-SAMP-RQMC-002], [EQ-SAMP-RQMC-003].

        """
        head = (
            f"{self.estimator} of {self.replicates} randomized {self.method} "
            f"rules with {self.size} points ({self.evaluations} evaluations): "
            f"{self.estimate:.10g}"
        )
        if self.interval is None or self.standard_error is None:
            return f"{head}; {self.interval_method}."
        return (
            f"{head}, standard error {self.standard_error:.3g}, "
            f"{self.interval_method} [{self.interval[0]:.10g}, "
            f"{self.interval[1]:.10g}]."
        )


def _average(function: Integrand, design: SamplingDesign) -> float:
    """Return the equal-weight average of an integrand over a design.

    Raises:
        ValueError: If the integrand returns a non-finite value.

    References:
        [EQ-SAMP-RQMC-002].

    """
    values = []
    for point in design.points:
        value = float(function(point))
        if not math.isfinite(value):
            raise ValueError(f"the integrand returned a non-finite value at {point}")
        values.append(value)
    return math.fsum(values) / len(values)


def _check_common(function: object, dimension: object, size: object) -> tuple[int, int]:
    """Validate the integrand, dimension, and points per replicate.

    Raises:
        TypeError: If ``function`` is not callable or a count is not an
            integer.
        ValueError: If a count is not positive.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    if not callable(function):
        raise TypeError("function must be callable")
    s = _require_positive_integer(dimension, name="dimension")
    n = _require_positive_integer(size, name="size")
    return s, n


def _replicate_estimates(
    function: Integrand,
    dimension: int,
    size: int,
    replicates: int,
    method: str,
    generating_vector: Sequence[int] | None,
    weights: Sequence[float] | None,
    stream: RandomStream,
) -> tuple[tuple[float, ...], tuple[tuple[int, ...], ...]]:
    """Evaluate ``replicates`` independent randomizations of one rule.

    Replicate ``r`` is randomized from the child stream ``("replicate", r)``.

    Returns:
        The replicate averages and the lattice generating vectors (empty for
        nets and Halton points).

    Raises:
        ValueError: If the method is unknown or its arguments are invalid.

    References:
        [EQ-SAMP-RQMC-001], [EQ-SAMP-LATTICE-001].

    """
    if method not in RQMC_METHODS:
        raise ValueError(f"method must be one of {RQMC_METHODS}, got {method!r}")
    vector: tuple[int, ...] | None = None
    if method == "lattice":
        if generating_vector is None:
            if size < 2:
                raise ValueError("a constructed lattice needs a prime size")
            vector = cbc_generating_vector(
                size, dimension, weights=weights
            ).generating_vector
        else:
            vector = _generating_vector(size, generating_vector)
            if len(vector) != dimension:
                raise ValueError(
                    f"generating_vector has {len(vector)} components but "
                    f"dimension is {dimension}"
                )
    elif generating_vector is not None or weights is not None:
        raise ValueError("generating_vector and weights apply only to method='lattice'")
    estimates = []
    for replicate in range(replicates):
        child = stream.child("replicate", replicate)
        if vector is not None:
            design = rank1_lattice(size, vector, shift=True, seed=child)
        elif method == "halton":
            design = halton(size, dimension, scramble=True, seed=child)
        else:
            design = sobol(
                size, dimension, scramble=method.removeprefix("sobol-"), seed=child
            )
        estimates.append(_average(function, design))
    vectors = (vector,) * replicates if vector is not None else ()
    return tuple(estimates), vectors


def _sample_standard_deviation(values: Sequence[float]) -> float:
    """Return the sample standard deviation (divisor ``R - 1``).

    References:
        [EQ-SAMP-RQMC-002].

    """
    return statistics.stdev(values) if len(values) > 1 else 0.0


def rqmc_mean(
    function: Integrand,
    dimension: int,
    size: int,
    *,
    replicates: int = 10,
    method: str = "lattice",
    generating_vector: Sequence[int] | None = None,
    weights: Sequence[float] | None = None,
    confidence: float = 0.95,
    seed: int | RandomStream | None = None,
) -> RQMCResult:
    """Estimate ``int_[0,1]^s f`` by the mean of randomized QMC replicates.

    ``mu_hat = (1/R) sum_r Q_r`` with ``Q_r`` the equal-weight average over
    the ``r``-th independent randomization, standard error
    ``s / sqrt(R)``, ``s^2 = sum_r (Q_r - mu_hat)^2 / (R - 1)``, and interval
    ``mu_hat +- t_{R-1, (1+c)/2} s / sqrt(R)``.

    Every randomization offered here makes each point uniform on the cube,
    so ``mu_hat`` is unbiased and ``s^2 / R`` is an unbiased estimate of its
    variance for any integrable (square-integrable) ``f``. The Student-t
    interval is approximate: its coverage is exact only if the ``Q_r`` are
    normal, which randomized QMC replicates need not be (their law can be
    skewed or heavy-tailed, and it need not approach normality as ``n``
    grows with ``R`` fixed). Measured coverage can fall below the nominal
    level; see ADR-0118 for numbers.

    Args:
        function: Integrand taking a point (tuple of ``s`` coordinates in
            ``(0, 1)``) and returning a finite float.
        dimension: Dimension ``s``.
        size: Points per replicate ``n``. A lattice without a supplied
            generating vector needs a prime ``n``; Sobol nets keep their
            balance only for powers of two.
        replicates: Number of independent randomizations ``R >= 2``.
        method: ``"lattice"`` (random shift), ``"sobol-owen"``,
            ``"sobol-shift"``, or ``"halton"`` (random digit permutations).
        generating_vector: Lattice generating vector; if omitted it is
            constructed by :func:`cbc_generating_vector`.
        weights: Product weights for that construction (lattice only).
        confidence: Nominal level of the interval, in ``(0, 1)``.
        seed: Seed or stream; ``None`` draws fresh entropy.

    Returns:
        Result with the mean, its standard error, and the t interval.

    Raises:
        TypeError: If ``function`` is not callable or a count is not an
            integer.
        ValueError: If ``replicates < 2``, ``confidence`` is not in
            ``(0, 1)``, the method or its arguments are invalid, or the
            integrand returns a non-finite value.

    Equation:
        [EQ-SAMP-RQMC-002], [EQ-SAMP-RQMC-001].

    References:
        [REF-CRANLEY-PATTERSON-1976], [REF-OWEN-1995],
        [REF-LECUYER-NAKAYAMA-OWEN-TUFFIN-2023] (reliability of intervals).

    """
    s, n = _check_common(function, dimension, size)
    count = _require_positive_integer(replicates, name="replicates")
    if count < 2:
        raise ValueError("replicates must be at least 2 to estimate a variance")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise TypeError("confidence must be a real number")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1")
    stream = resolve_stream(seed)
    estimates, vectors = _replicate_estimates(
        function, s, n, count, method, generating_vector, weights, stream
    )
    mean = math.fsum(estimates) / count
    deviation = _sample_standard_deviation(estimates)
    standard_error = deviation / math.sqrt(count)
    quantile = StudentT(float(count - 1)).quantile(0.5 * (1.0 + confidence))
    half_width = quantile * standard_error
    return RQMCResult(
        estimate=mean,
        estimator="mean",
        method=method,
        replicate_estimates=estimates,
        standard_error=standard_error,
        interval=(mean - half_width, mean + half_width),
        confidence=float(confidence),
        interval_method=(
            f"approximate {100.0 * confidence:g}% Student-t interval over "
            f"{count} randomized replicates (coverage not guaranteed)"
        ),
        replicate_standard_deviation=deviation,
        replicates=count,
        size=n,
        dimension=s,
        evaluations=count * n,
        generating_vectors=vectors,
        seed=stream.seed,
    )


def _odd_replicates(replicates: object) -> int:
    """Validate an odd number of replicates of at least three.

    Raises:
        TypeError: If ``replicates`` is not an integer.
        ValueError: If it is even or below three.

    References:
        [EQ-SAMP-RQMC-003].

    """
    count = _require_positive_integer(replicates, name="replicates")
    if count < 3 or count % 2 == 0:
        raise ValueError(
            f"replicates must be an odd number of at least 3 so that the "
            f"median is one of the replicate estimates, got {count}"
        )
    return count


def _median_result(
    estimates: tuple[float, ...],
    vectors: tuple[tuple[int, ...], ...],
    method: str,
    size: int,
    dimension: int,
    stream: RandomStream,
) -> RQMCResult:
    """Assemble the result of a median-of-replicates estimator.

    References:
        [EQ-SAMP-RQMC-003].

    """
    return RQMCResult(
        estimate=statistics.median(estimates),
        estimator="median",
        method=method,
        replicate_estimates=estimates,
        standard_error=None,
        interval=None,
        confidence=None,
        interval_method=_NO_INTERVAL,
        replicate_standard_deviation=_sample_standard_deviation(estimates),
        replicates=len(estimates),
        size=size,
        dimension=dimension,
        evaluations=len(estimates) * size,
        generating_vectors=vectors,
        seed=stream.seed,
    )


def rqmc_median(
    function: Integrand,
    dimension: int,
    size: int,
    *,
    replicates: int = 11,
    method: str = "lattice",
    generating_vector: Sequence[int] | None = None,
    weights: Sequence[float] | None = None,
    seed: int | RandomStream | None = None,
) -> RQMCResult:
    """Estimate ``int_[0,1]^s f`` by the median of randomized QMC replicates.

    ``mu_tilde = median(Q_1, ..., Q_R)`` for an odd ``R``, with the same
    replicates as :func:`rqmc_mean` for the same seed.

    Motivation: the error of one randomized rule is typically small with
    high probability and occasionally large; the mean inherits the rare
    large errors through its variance, while the median is wrong only if
    more than half of the replicates are. The median of independently
    randomized rules can therefore converge at the rate the integrand's
    smoothness permits without that smoothness entering the construction
    (the subject of Pan and Owen for scrambled nets in one dimension and of
    Goda and L'Ecuyer for randomly chosen rules). With a fixed,
    well-constructed rule and a smooth integrand the replicates have no
    outliers and the median has no such advantage over the mean.

    No confidence interval is attached, and none with guaranteed coverage is
    provided by this library: the median of the replicates is an estimate of
    the median of the replicate distribution, which coincides with the
    integral only when that distribution is symmetric about it. The median
    is in general a biased estimator of the integral (the bias is small when
    the replicates concentrate). An order-statistic interval over the
    replicates would cover the replicate median, not the integral.
    ``replicate_standard_deviation`` reports the spread of the replicates as
    a diagnostic only.

    Args:
        function: Integrand taking a point (tuple of ``s`` coordinates in
            ``(0, 1)``) and returning a finite float.
        dimension: Dimension ``s``.
        size: Points per replicate ``n`` (see :func:`rqmc_mean`).
        replicates: Odd number of independent randomizations ``R >= 3``.
        method: ``"lattice"``, ``"sobol-owen"``, ``"sobol-shift"``, or
            ``"halton"``.
        generating_vector: Lattice generating vector; constructed by
            :func:`cbc_generating_vector` if omitted.
        weights: Product weights for that construction (lattice only).
        seed: Seed or stream; ``None`` draws fresh entropy.

    Returns:
        Result with the median; ``standard_error`` and ``interval`` are
        ``None``.

    Raises:
        TypeError: If ``function`` is not callable or a count is not an
            integer.
        ValueError: If ``replicates`` is even or below 3, the method or its
            arguments are invalid, or the integrand returns a non-finite
            value.

    Equation:
        [EQ-SAMP-RQMC-003].

    References:
        [REF-PAN-OWEN-2023], [REF-GODA-LECUYER-2022].

    """
    s, n = _check_common(function, dimension, size)
    count = _odd_replicates(replicates)
    stream = resolve_stream(seed)
    estimates, vectors = _replicate_estimates(
        function, s, n, count, method, generating_vector, weights, stream
    )
    return _median_result(estimates, vectors, method, n, s, stream)


def median_random_lattice(
    function: Integrand,
    dimension: int,
    size: int,
    *,
    replicates: int = 11,
    shift: bool = True,
    seed: int | RandomStream | None = None,
) -> RQMCResult:
    """Estimate an integral by the median of randomly generated lattice rules.

    Replicate ``r`` draws ``z^(r)`` uniformly from ``{1, ..., n - 1}^s`` for a
    prime ``n`` and evaluates the equal-weight rule on
    ``frac(k z^(r) / n + Delta^(r))``; the estimate is the median of the
    ``R`` averages. Nothing is constructed and no weights or smoothness
    enter. A uniformly drawn vector is good with high probability but can be
    very bad (for example two equal components put all points on a
    diagonal), so the mean of these replicates has a large variance while
    the median is not affected unless most replicates are bad.

    With ``shift=True`` (the default, a choice of this library) every
    replicate also gets an independent Cranley-Patterson shift
    ``Delta^(r) ~ U[0, 1)^s``, which makes each replicate unbiased and suits
    integrands that are not periodic. With ``shift=False`` the unshifted
    lattices are used; they all contain the origin, so the integrand is
    evaluated there in every replicate.

    As for :func:`rqmc_median`, no confidence interval is provided: the
    median is a generally biased estimator whose error is not measured by
    the spread of the replicates.

    Args:
        function: Integrand taking a point (tuple of ``s`` coordinates in
            ``[0, 1)``) and returning a finite float.
        dimension: Dimension ``s``.
        size: Prime number of points per replicate ``n >= 3``.
        replicates: Odd number of replicates ``R >= 3``.
        shift: Add an independent random shift in every replicate.
        seed: Seed or stream; ``None`` draws fresh entropy.

    Returns:
        Result with the median, the replicate averages, and the generating
        vector of every replicate.

    Raises:
        TypeError: If ``function`` is not callable or a count is not an
            integer.
        ValueError: If ``size`` is not a prime of at least 3, ``replicates``
            is even or below 3, or the integrand returns a non-finite value.

    Equation:
        [EQ-SAMP-RQMC-004], [EQ-SAMP-RQMC-003].

    References:
        [REF-GODA-LECUYER-2022].

    """
    s, _ = _check_common(function, dimension, size)
    n = _require_prime_size(size)
    if n < 3:
        raise ValueError("size must be a prime of at least 3")
    count = _odd_replicates(replicates)
    stream = resolve_stream(seed)
    estimates = []
    vectors = []
    for replicate in range(count):
        child = stream.child("replicate", replicate)
        generator = child.child("vector").generator()
        vector = tuple(generator.randrange(1, n) for _ in range(s))
        design = rank1_lattice(n, vector, shift=shift, seed=child.child("shift"))
        estimates.append(_average(function, design))
        vectors.append(vector)
    return _median_result(
        tuple(estimates), tuple(vectors), "random-lattice", n, s, stream
    )
