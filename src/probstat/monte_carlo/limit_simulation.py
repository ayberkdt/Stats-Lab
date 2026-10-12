"""Simulation demonstrations of the laws of large numbers and the CLT.

These helpers *illustrate* convergence with pseudo-random draws from a
:class:`~probstat.foundations.RandomStream`; every number they return is a
Monte Carlo estimate with sampling error. Exact distributions and proven
bounds for the same questions are in :mod:`probstat.probability.limit_theorems`
and should be preferred whenever they apply.

* Convergence in probability: ``P(|Xbar_n - mu| > eps)`` against ``n``.
* Almost-sure convergence: running-mean paths and the fraction of paths with
  ``max_{m <= k <= N} |Xbar_k - mu| > eps`` on a finite horizon ``N``.
* Convergence in distribution: the Kolmogorov-Smirnov distance between the
  empirical law of ``sqrt(n) (Xbar_n - mu) / sigma`` and ``Phi``.

A sampler is a callable ``sampler(rng) -> float`` that draws one summand
from the supplied :class:`random.Random`.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from probstat.foundations import RandomStream, require_finite_real, resolve_stream
from probstat.foundations.validation import require_non_negative_integer

Sampler = Callable[[random.Random], float]


def _positive_integer(value: object, *, name: str) -> int:
    """Validate a positive integer.

    Args:
        value: Candidate value.
        name: Parameter name.

    Returns:
        The integer.

    Raises:
        ValueError: If the value is not a positive integer.

    References:
        Project validation policy. No equation is evaluated.

    """
    number = require_non_negative_integer(value, name=name)
    if number < 1:
        raise ValueError(f"{name} must be a positive integer")
    return number


def _sizes(values: Iterable[int]) -> tuple[int, ...]:
    """Validate and sort distinct positive checkpoint sizes.

    Args:
        values: Candidate sample sizes.

    Returns:
        Sorted distinct sizes.

    Raises:
        ValueError: If no size is supplied.

    References:
        Project validation policy. No equation is evaluated.

    """
    sizes = tuple(sorted({_positive_integer(v, name="n") for v in values}))
    if not sizes:
        raise ValueError("at least one sample size is required")
    return sizes


@dataclass(frozen=True, slots=True)
class WeakLawPoint:
    """Monte Carlo estimate of ``P(|Xbar_n - mu| > eps)`` at one ``n``.

    Args:
        n: Sample size.
        probability: Fraction of replications with ``|Xbar_n - mu| > eps``.
        standard_error: Binomial standard error ``sqrt(p (1 - p) / R)``.

    References:
        [EQ-LIM-009].

    """

    n: int
    probability: float
    standard_error: float


def simulate_weak_law(
    sampler: Sampler,
    mean: float,
    epsilon: float,
    sizes: Iterable[int],
    *,
    replications: int,
    seed: int | RandomStream | None = None,
) -> tuple[WeakLawPoint, ...]:
    """Estimate ``P(|Xbar_n - mu| > eps)`` for several ``n`` by simulation.

    Each replication draws one path of length ``max(sizes)`` from
    ``seed.child("replication", r)`` and records the running mean at every
    checkpoint, so estimates at different ``n`` are positively correlated.

    Args:
        sampler: One-summand sampler ``sampler(rng) -> float``.
        mean: True mean ``mu``.
        epsilon: Positive tolerance.
        sizes: Positive checkpoint sample sizes.
        replications: Positive number of independent paths ``R``.
        seed: Integer seed, stream, or ``None`` (not reproducible).

    Returns:
        One estimate per checkpoint, sorted by ``n``.

    Equation:
        [EQ-LIM-009].

    References:
        [REF-CASELLA-BERGER-2002, Section 5.5.1], [REF-WASSERMAN-2004,
        Chapter 5].

    """
    mu = require_finite_real(mean, name="mean")
    eps = require_finite_real(epsilon, name="epsilon")
    if eps <= 0.0:
        raise ValueError("epsilon must be positive")
    checkpoints = _sizes(sizes)
    count = _positive_integer(replications, name="replications")
    stream = resolve_stream(seed)
    exceed = [0] * len(checkpoints)
    for replication in range(count):
        rng = stream.child("replication", replication).generator()
        total = 0.0
        position = 0
        for k in range(1, checkpoints[-1] + 1):
            total += float(sampler(rng))
            if k == checkpoints[position]:
                if abs(total / k - mu) > eps:
                    exceed[position] += 1
                position += 1
    points: list[WeakLawPoint] = []
    for n, hits in zip(checkpoints, exceed, strict=True):
        p = hits / count
        points.append(WeakLawPoint(n, p, math.sqrt(p * (1.0 - p) / count)))
    return tuple(points)


@dataclass(frozen=True, slots=True)
class RunningMeanPaths:
    """Running-mean paths and finite-horizon uniform-deviation frequencies.

    Args:
        checkpoints: Sample sizes at which each path is recorded.
        paths: ``paths[j][i]`` is ``Xbar_{checkpoints[i]}`` on path ``j``.
        epsilon: Tolerance used for the deviation frequencies.
        sup_exceedance: For each checkpoint ``m``, the fraction of paths with
            ``max_{m <= k <= N} |Xbar_k - mu| > eps`` (every ``k``, not only
            checkpoints), ``N = checkpoints[-1]``.
        standard_errors: Binomial standard errors of ``sup_exceedance``.

    References:
        [EQ-LIM-009].

    """

    checkpoints: tuple[int, ...]
    paths: tuple[tuple[float, ...], ...]
    epsilon: float
    sup_exceedance: tuple[float, ...]
    standard_errors: tuple[float, ...]


def simulate_running_means(
    sampler: Sampler,
    mean: float,
    epsilon: float,
    checkpoints: Iterable[int],
    *,
    paths: int,
    seed: int | RandomStream | None = None,
) -> RunningMeanPaths:
    """Simulate running-mean paths for an almost-sure-convergence picture.

    The strong law says ``P(sup_{k >= m} |Xbar_k - mu| > eps) -> 0`` as
    ``m -> inf``. On the finite horizon ``N`` the frequency reported for
    checkpoint ``m`` estimates ``P(max_{m <= k <= N} |Xbar_k - mu| > eps)``,
    which is a lower bound for the infinite-horizon probability.

    Args:
        sampler: One-summand sampler ``sampler(rng) -> float``.
        mean: True mean ``mu``.
        epsilon: Positive tolerance.
        checkpoints: Positive recording sizes; the largest is the horizon.
        paths: Positive number of independent paths.
        seed: Integer seed, stream, or ``None`` (not reproducible).

    Returns:
        Paths and deviation frequencies.

    Equation:
        [EQ-LIM-009].

    References:
        [REF-CASELLA-BERGER-2002, Section 5.5.2], [REF-WASSERMAN-2004,
        Chapter 5].

    """
    mu = require_finite_real(mean, name="mean")
    eps = require_finite_real(epsilon, name="epsilon")
    if eps <= 0.0:
        raise ValueError("epsilon must be positive")
    sizes = _sizes(checkpoints)
    count = _positive_integer(paths, name="paths")
    stream = resolve_stream(seed)
    horizon = sizes[-1]
    recorded: list[tuple[float, ...]] = []
    hits = [0] * len(sizes)
    for path in range(count):
        rng = stream.child("path", path).generator()
        total = 0.0
        position = 0
        means: list[float] = []
        deviations = [0.0] * (horizon + 1)
        for k in range(1, horizon + 1):
            total += float(sampler(rng))
            deviations[k] = abs(total / k - mu)
            if k == sizes[position]:
                means.append(total / k)
                position += 1
        tail_max = [0.0] * (horizon + 2)
        for k in range(horizon, 0, -1):
            tail_max[k] = max(deviations[k], tail_max[k + 1])
        for index, m in enumerate(sizes):
            if tail_max[m] > eps:
                hits[index] += 1
        recorded.append(tuple(means))
    frequencies = tuple(h / count for h in hits)
    errors = tuple(math.sqrt(f * (1.0 - f) / count) for f in frequencies)
    return RunningMeanPaths(sizes, tuple(recorded), eps, frequencies, errors)


@dataclass(frozen=True, slots=True)
class CLTSimulation:
    """Kolmogorov-Smirnov distance of simulated standardized means to ``Phi``.

    Args:
        n: Sample size of each mean.
        replications: Number of simulated means ``R``.
        ks_distance: ``sup_z |F_R(z) - Phi(z)|`` for the empirical CDF.
        sampling_scale: ``1 / sqrt(R)``; with an exactly normal law the
            distance is of this order (DKW), so distances near this scale do
            not resolve the true CLT error.

    References:
        [EQ-LIM-009].

    """

    n: int
    replications: int
    ks_distance: float
    sampling_scale: float


def simulate_clt_distance(
    sampler: Sampler,
    mean: float,
    standard_deviation: float,
    n: int,
    *,
    replications: int,
    seed: int | RandomStream | None = None,
) -> CLTSimulation:
    """Estimate the CLT error by the KS distance of simulated means to ``Phi``.

    The returned distance mixes the true Kolmogorov distance with Monte Carlo
    noise of order ``1 / sqrt(R)``; compare with
    :func:`probstat.probability.limit_theorems.kolmogorov_distance_to_normal`
    where an exact answer exists.

    Args:
        sampler: One-summand sampler ``sampler(rng) -> float``.
        mean: True mean ``mu``.
        standard_deviation: True ``sigma > 0``.
        n: Positive sample size of each mean.
        replications: Number of simulated means ``R >= 2``.
        seed: Integer seed, stream, or ``None`` (not reproducible).

    Returns:
        Simulation summary.

    Equation:
        [EQ-LIM-009].

    References:
        [REF-CASELLA-BERGER-2002, Section 5.5.3],
        [REF-DVORETZKY-KIEFER-WOLFOWITZ-1956], [REF-MASSART-1990].

    """
    mu = require_finite_real(mean, name="mean")
    sigma = require_finite_real(standard_deviation, name="standard_deviation")
    if sigma <= 0.0:
        raise ValueError("standard_deviation must be positive")
    size = _positive_integer(n, name="n")
    count = _positive_integer(replications, name="replications")
    if count < 2:
        raise ValueError("replications must be at least two")
    stream = resolve_stream(seed)
    root = math.sqrt(size)
    standardized: list[float] = []
    for replication in range(count):
        rng = stream.child("replication", replication).generator()
        total = math.fsum(float(sampler(rng)) for _ in range(size))
        standardized.append(root * (total / size - mu) / sigma)
    standardized.sort()
    distance = 0.0
    for index, z in enumerate(standardized):
        phi = 0.5 * math.erfc(-z / math.sqrt(2.0))
        distance = max(distance, (index + 1) / count - phi, phi - index / count)
    return CLTSimulation(size, count, distance, 1.0 / math.sqrt(count))
