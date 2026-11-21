"""Mean estimators with sub-Gaussian-type deviations under heavy tails.

The sample mean of ``n`` observations with variance ``sigma^2`` deviates by
about ``sigma sqrt(1/(n delta))`` at confidence ``1 - delta`` when only a
finite variance is assumed (Chebyshev), which is far worse than the
``sigma sqrt(log(1/delta)/n)`` of the Gaussian case. Two classical
estimators recover the logarithmic dependence on ``delta`` with nothing but
a finite variance:

* :func:`median_of_means` splits the sample into blocks and returns the
  median of the block means;
* :func:`catoni_mean` solves a soft-truncated estimating equation.

Both depend on the confidence level through a tuning parameter (the number
of blocks, the scale), so each is an estimator *for one* ``delta``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    coerce_finite_sample,
    require_finite_real,
)
from probstat.optimization import brent_root

_ASSIGNMENTS = ("random", "sequential")


def _open_unit(value: object, *, name: str) -> float:
    """Validate a number strictly between 0 and 1.

    Args:
        value: Candidate value.
        name: Argument name for error messages.

    Returns:
        The value as a float.

    Raises:
        ValueError: If the value is not in ``(0, 1)``.

    References:
        [REF-LUGOSI-MENDELSON-2019]. No equation is evaluated.

    """
    number = require_finite_real(value, name=name)
    if not 0.0 < number < 1.0:
        raise ValueError(f"{name} must lie strictly between 0 and 1")
    return number


def median_of_means_block_count(delta: float) -> int:
    """Return the block count ``k = ceil(8 log(1/delta))``.

    With this many blocks the failure probability bound ``exp(-k/8)`` of
    [EQ-STAT-ROB-002] is at most ``delta``.

    Args:
        delta: Failure probability in ``(0, 1)``.

    Returns:
        The number of blocks, at least 1.

    Raises:
        ValueError: If ``delta`` is not in ``(0, 1)``.

    Equation:
        [EQ-STAT-ROB-002].

    References:
        [REF-LUGOSI-MENDELSON-2019].

    """
    level = _open_unit(delta, name="delta")
    return max(1, math.ceil(8.0 * math.log(1.0 / level)))


@dataclass(frozen=True, slots=True)
class MedianOfMeansResult:
    """Median-of-means estimate with its block structure.

    Args:
        estimate: Median of the block means.
        block_means: Mean of every block, in block order.
        block_sizes: Number of observations in every block.
        blocks: Number of blocks ``k``.
        observations: Sample size ``n``.
        assignment: ``"random"`` or ``"sequential"``.
        failure_probability_bound: ``exp(-k/8)``, the probability bound that
            goes with :meth:`deviation_bound`.

    References:
        [EQ-STAT-ROB-001], [EQ-STAT-ROB-002], [REF-LUGOSI-MENDELSON-2019].

    """

    estimate: float
    block_means: tuple[float, ...]
    block_sizes: tuple[int, ...]
    blocks: int
    observations: int
    assignment: str
    failure_probability_bound: float

    def deviation_bound(self, sigma: float) -> float:
        """Return ``2 sigma / sqrt(min block size)``.

        For independent observations with mean ``mu`` and standard deviation
        at most ``sigma``, and blocks chosen without looking at the values,
        ``|estimate - mu|`` exceeds this bound with probability at most
        ``failure_probability_bound``. The bound follows from Chebyshev's
        inequality in every block and Hoeffding's inequality for the number
        of failing blocks; it is not tight.

        Args:
            sigma: Upper bound on the standard deviation of one observation.

        Returns:
            The deviation bound in the units of the data.

        Raises:
            ValueError: If ``sigma`` is negative or not finite.

        Equation:
            [EQ-STAT-ROB-002].

        References:
            [REF-LUGOSI-MENDELSON-2019].

        """
        scale = require_finite_real(sigma, name="sigma")
        if scale < 0.0:
            raise ValueError("sigma must be non-negative")
        return 2.0 * scale / math.sqrt(min(self.block_sizes))


def median_of_means(
    values: Iterable[float],
    *,
    blocks: int | None = None,
    delta: float | None = None,
    assignment: str = "random",
    seed: int | RandomStream | None = None,
) -> MedianOfMeansResult:
    """Return the median of the means of ``k`` disjoint blocks.

    The ``n`` observations are split into ``k`` blocks whose sizes differ by
    at most one (``floor(n/k)`` or ``ceil(n/k)``), each block is averaged,
    and the median of the ``k`` averages is returned (the mean of the two
    middle averages for even ``k``). For independent observations with
    finite variance ``sigma^2`` the error is of order
    ``sigma sqrt(log(1/delta)/n)`` with probability ``1 - delta`` when
    ``k`` is of order ``log(1/delta)``; :meth:`MedianOfMeansResult.deviation_bound`
    gives the explicit bound derived for this implementation. ``k = 1``
    gives the sample mean.

    Assumptions: the guarantee needs independent observations with a common
    mean and a block assignment that does not depend on the values.
    ``assignment="random"`` shuffles the indices with the seeded stream and
    is safe for any ordering of the data; ``"sequential"`` cuts the sample
    in its given order and is appropriate only when that order carries no
    information. The estimate depends on the assignment, and it is tuned to
    one ``delta``; it is not a general-purpose replacement for the mean.

    Args:
        values: Finite real observations.
        blocks: Number of blocks ``k`` with ``1 <= k <= n``. Give either
            ``blocks`` or ``delta``.
        delta: Failure probability; sets ``k = ceil(8 log(1/delta))``.
        assignment: ``"random"`` (default) or ``"sequential"``.
        seed: Integer seed, :class:`~probstat.foundations.RandomStream`, or
            ``None`` for fresh entropy (``"random"`` assignment only).

    Returns:
        Estimate, block means and sizes, and the failure probability bound.

    Raises:
        TypeError: If an observation is not real or ``blocks`` is not an
            integer.
        ValueError: If the sample is empty, both or neither of ``blocks`` and
            ``delta`` are given, ``k`` is outside ``[1, n]``, or the
            assignment is unknown.

    Equation:
        [EQ-STAT-ROB-001], [EQ-STAT-ROB-002].

    References:
        [REF-LUGOSI-MENDELSON-2019].

    """
    sample = coerce_finite_sample(values)
    n = len(sample)
    if (blocks is None) == (delta is None):
        raise ValueError("give exactly one of blocks and delta")
    if blocks is None:
        assert delta is not None
        k = median_of_means_block_count(delta)
        if k > n:
            raise ValueError(
                f"delta requires {k} blocks but only {n} observations are available"
            )
    else:
        if isinstance(blocks, bool) or not isinstance(blocks, int):
            raise TypeError("blocks must be an integer")
        k = blocks
        if not 1 <= k <= n:
            raise ValueError("blocks must satisfy 1 <= blocks <= number of values")
    if assignment not in _ASSIGNMENTS:
        raise ValueError("assignment must be 'random' or 'sequential'")
    order = list(range(n))
    if assignment == "random" and k > 1:
        resolve_stream(seed).child("median-of-means").generator().shuffle(order)
    base, extra = divmod(n, k)
    means: list[float] = []
    sizes: list[int] = []
    start = 0
    for block in range(k):
        size = base + (1 if block < extra else 0)
        members = order[start : start + size]
        means.append(math.fsum(sample[index] for index in members) / size)
        sizes.append(size)
        start += size
    ordered = sorted(means)
    middle = k // 2
    estimate = (
        ordered[middle] if k % 2 == 1 else 0.5 * (ordered[middle - 1] + ordered[middle])
    )
    return MedianOfMeansResult(
        estimate=estimate,
        block_means=tuple(means),
        block_sizes=tuple(sizes),
        blocks=k,
        observations=n,
        assignment=assignment,
        failure_probability_bound=math.exp(-k / 8.0),
    )


def catoni_influence(x: float) -> float:
    """Return Catoni's influence function ``sign(x) log(1 + |x| + x^2/2)``.

    The function is odd, increasing, equal to ``x - x^3/6 + O(x^4)`` near
    zero, and grows only logarithmically, which is what limits the weight of
    extreme observations.

    Args:
        x: Finite real argument.

    Returns:
        The value ``psi(x)``.

    Raises:
        ValueError: If ``x`` is not finite.

    Equation:
        [EQ-STAT-ROB-003].

    References:
        [REF-CATONI-2012].

    """
    value = require_finite_real(x, name="x")
    magnitude = abs(value)
    return math.copysign(math.log1p(magnitude + 0.5 * magnitude * magnitude), value)


def catoni_scale(observations: int, variance_bound: float, delta: float) -> float:
    """Return the default scale ``a = sqrt(2 log(1/delta) / (n v))``.

    This is a choice of the order analysed by Catoni; the source gives the
    exact constants and the admissible range of ``delta`` for its deviation
    bound, which this library does not reproduce.

    Args:
        observations: Sample size ``n >= 1``.
        variance_bound: Positive bound ``v`` on the variance.
        delta: Failure probability in ``(0, 1)``.

    Returns:
        The positive scale ``a`` in inverse data units.

    Raises:
        ValueError: If an argument is out of range.

    Equation:
        [EQ-STAT-ROB-004].

    References:
        [REF-CATONI-2012].

    """
    if isinstance(observations, bool) or not isinstance(observations, int):
        raise ValueError("observations must be a positive integer")
    if observations < 1:
        raise ValueError("observations must be a positive integer")
    v = require_finite_real(variance_bound, name="variance_bound")
    if not v > 0.0:
        raise ValueError("variance_bound must be positive")
    level = _open_unit(delta, name="delta")
    return math.sqrt(2.0 * math.log(1.0 / level) / (observations * v))


@dataclass(frozen=True, slots=True)
class CatoniMeanResult:
    """Catoni M-estimate of a mean with the root-search diagnostics.

    Args:
        estimate: Root ``theta`` of the estimating equation.
        scale: Scale ``a`` that was used (inverse data units).
        variance_bound: Variance bound behind the default scale, or ``None``
            when ``scale`` was given.
        variance_estimated: Whether that bound is the sample variance of the
            same data (a heuristic plug-in) rather than a supplied bound.
        residual: ``sum_i psi(a (x_i - theta))`` at the estimate.
        iterations: Root-finder iterations.
        evaluations: Evaluations of the estimating function.
        converged: Whether the root finder met its tolerance.

    References:
        [EQ-STAT-ROB-003], [EQ-STAT-ROB-004], [REF-CATONI-2012].

    """

    estimate: float
    scale: float
    variance_bound: float | None
    variance_estimated: bool
    residual: float
    iterations: int
    evaluations: int
    converged: bool


def catoni_mean(
    values: Iterable[float],
    *,
    scale: float | None = None,
    variance_bound: float | None = None,
    delta: float = 0.05,
    tolerance: float = 1e-12,
    max_iterations: int = 200,
) -> CatoniMeanResult:
    """Return Catoni's M-estimator of the mean.

    The estimate is the root in ``theta`` of
    ``sum_i psi(a (x_i - theta)) = 0`` with
    ``psi(x) = sign(x) log(1 + |x| + x^2/2)``. The left-hand side is
    continuous and strictly decreasing in ``theta``, non-negative at the
    sample minimum and non-positive at the sample maximum, so the root is
    unique and is bracketed by the sample range; Brent's method finds it. As
    ``a -> 0`` the estimate tends to the sample mean because
    ``psi(x) = x + O(x^3)``. The estimator is translation equivariant but
    not scale equivariant unless ``a`` is rescaled with the data.

    When ``scale`` is not given it is ``sqrt(2 log(1/delta) / (n v))``
    ([EQ-STAT-ROB-004]) with ``v = variance_bound``. If no bound is given
    either, the unbiased sample variance of the same data is plugged in;
    that is a heuristic (for heavy tails the sample variance is itself
    unreliable) and is flagged in the result. For independent observations
    with variance at most ``v`` the error is of order
    ``sqrt(v log(1/delta)/n)`` with probability ``1 - delta``; see the source
    for the constants.

    Args:
        values: Finite real observations.
        scale: Positive scale ``a`` in inverse data units.
        variance_bound: Positive bound on the variance, used for the default
            scale. Ignored when ``scale`` is given.
        delta: Failure probability in ``(0, 1)`` for the default scale.
        tolerance: Absolute tolerance of the root search in data units.
        max_iterations: Maximum root-finder iterations.

    Returns:
        Estimate, scale, residual, and convergence diagnostics.

    Raises:
        TypeError: If an observation is not real.
        ValueError: If the sample is empty or an argument is out of range.

    Equation:
        [EQ-STAT-ROB-003], [EQ-STAT-ROB-004], [EQ-OPT-BRENT-001].

    References:
        [REF-CATONI-2012], [REF-BRENT-1973].

    """
    sample = coerce_finite_sample(values)
    n = len(sample)
    stop = require_finite_real(tolerance, name="tolerance")
    if not stop > 0.0:
        raise ValueError("tolerance must be positive")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int):
        raise ValueError("max_iterations must be a positive integer")
    if max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    level = _open_unit(delta, name="delta")
    lower, upper = min(sample), max(sample)
    bound: float | None = None
    estimated = False
    if scale is not None:
        a = require_finite_real(scale, name="scale")
        if not a > 0.0:
            raise ValueError("scale must be positive")
    elif lower == upper:
        a = 1.0
    else:
        if variance_bound is None:
            if n < 2:
                raise ValueError("at least two observations are required")
            centre = math.fsum(sample) / n
            bound = math.fsum((x - centre) ** 2 for x in sample) / (n - 1)
            estimated = True
        else:
            bound = require_finite_real(variance_bound, name="variance_bound")
        a = catoni_scale(n, bound, level)
    if lower == upper:
        return CatoniMeanResult(
            estimate=lower,
            scale=a,
            variance_bound=bound,
            variance_estimated=estimated,
            residual=0.0,
            iterations=0,
            evaluations=0,
            converged=True,
        )

    def estimating_function(theta: float) -> float:
        """Return ``sum_i psi(a (x_i - theta))``.

        References:
            [EQ-STAT-ROB-003], [REF-CATONI-2012].

        """
        total = []
        for x in sample:
            z = a * (x - theta)
            magnitude = abs(z)
            total.append(math.copysign(math.log1p(magnitude + 0.5 * z * z), z))
        return math.fsum(total)

    root = brent_root(
        estimating_function,
        lower,
        upper,
        tolerance=stop,
        max_iterations=max_iterations,
    )
    return CatoniMeanResult(
        estimate=root.root,
        scale=a,
        variance_bound=bound,
        variance_estimated=estimated,
        residual=root.value,
        iterations=root.iterations,
        evaluations=root.evaluations,
        converged=root.converged,
    )
