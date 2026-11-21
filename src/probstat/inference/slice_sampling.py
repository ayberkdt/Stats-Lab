"""Slice sampling: coordinate-wise stepping-out and elliptical slice sampling.

Both samplers are Markov chains without a rejection step and without a
proposal scale that must be tuned for correctness.

Univariate slice sampling (:func:`slice_sampler`, [EQ-INF-SLICE-001])
    To update one coordinate ``x`` of a density ``f``: draw a height
    ``y ~ U(0, f(x))``, which defines the slice ``{x' : f(x') > y}``; place
    an interval of width ``w`` uniformly around ``x`` and extend each end in
    steps of ``w`` until it leaves the slice ("stepping out", optionally at
    most ``m`` steps shared at random between the two ends); draw points
    uniformly from the interval, shrinking it towards ``x`` after each point
    outside the slice, until a point inside the slice is found (Neal 2003,
    Figures 3 and 5). The update leaves ``f`` invariant for every ``w > 0``;
    ``w`` only affects the number of density evaluations. Applied to each
    coordinate in turn it is a Gibbs-type sweep, so strongly correlated
    coordinates mix slowly, as with any single-site update.

Elliptical slice sampling (:func:`elliptical_slice_sampler`, [EQ-INF-SLICE-002])
    For a posterior ``pi(f) ∝ L(f) N(f; mu, Sigma)``: draw ``nu ~ N(0,
    Sigma)`` and a log-height ``log y = log L(f) + log U``; draw an angle
    ``theta ~ U[0, 2 pi)`` with bracket ``[theta - 2 pi, theta]``; propose
    ``f' = mu + (f - mu) cos(theta) + nu sin(theta)`` and accept when
    ``log L(f') > log y``, otherwise shrink the bracket towards ``theta = 0``
    and redraw (Murray, Adams and MacKay 2010, Figure 2). Every point of the
    ellipse has the prior law, so only the likelihood is evaluated, and the
    current state (``theta = 0``) is always in the slice, so the loop ends.

Reported quantities
    The results are ordinary :class:`~probstat.inference.MCMCResult` objects.
    ``acceptance_rate`` is ``1`` by construction (every update ends with an
    accepted point); ``evaluations`` counts every log-density or
    log-likelihood call, the cost that actually varies. The
    ``proposal_covariance`` field holds ``diag(w^2)`` for the coordinate
    sampler and the prior covariance for the elliptical sampler.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)
from probstat.linalg import Matrix, Vector, as_vector, cholesky, matvec, symmetrize

from .mcmc import (
    LogDensity,
    MarkovChain,
    MCMCResult,
    _checked_log_density,
    _initial_points,
    _names,
    _validate_run,
    _with_convergence_check,
)

_STEP_GUARD = 100_000
_SHRINK_GUARD = 1_000
_TWO_PI = 2.0 * math.pi


class _Counter:
    """Log-density wrapper that counts evaluations.

    References:
        [REF-NEAL-2003]. No equation is evaluated.

    """

    __slots__ = ("calls", "function")

    def __init__(self, function: LogDensity) -> None:
        """Store the function and zero the counter.

        References:
            Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

        """
        self.function = function
        self.calls = 0

    def __call__(self, point: Vector) -> float:
        """Evaluate the function once and count the call.

        References:
            [REF-NEAL-2003]. No equation is evaluated.

        """
        self.calls += 1
        return self.function(point)


def _slice_coordinate(
    target: _Counter,
    state: Vector,
    current: float,
    index: int,
    width: float,
    max_steps: int | None,
    generator: random.Random,
) -> tuple[Vector, float]:
    """Update one coordinate by stepping out and shrinkage.

    References:
        [EQ-INF-SLICE-001], [REF-NEAL-2003, Figures 3 and 5].

    """
    level = current + math.log(1.0 - generator.random())
    origin = state[index]

    def at(value: float) -> tuple[Vector, float]:
        """Return the state with coordinate ``index`` replaced and its density.

        References:
            [EQ-INF-SLICE-001], [REF-NEAL-2003].

        """
        point = state[:index] + (value,) + state[index + 1 :]
        return point, target(point)

    left = origin - width * generator.random()
    right = left + width
    if max_steps is None:
        left_budget = right_budget = _STEP_GUARD
    else:
        left_budget = int(math.floor(max_steps * generator.random()))
        right_budget = max_steps - 1 - left_budget
    while left_budget > 0 and at(left)[1] > level:
        left -= width
        left_budget -= 1
    while right_budget > 0 and at(right)[1] > level:
        right += width
        right_budget -= 1
    if max_steps is None and (left_budget == 0 or right_budget == 0):
        raise ValueError(
            "stepping out did not end after 100000 steps: the density does "
            "not decay along a coordinate (improper target?) or the width is "
            "far too small"
        )
    for _ in range(_SHRINK_GUARD):
        candidate = left + (right - left) * generator.random()
        point, value = at(candidate)
        if value > level:
            return point, value
        if candidate < origin:
            left = candidate
        else:
            right = candidate
        if not left < right:
            break
    # The bracket collapsed onto the current point at floating-point
    # resolution; staying there is the limit of the shrinkage procedure.
    return state, current


def _run_slice_chain(
    log_density: LogDensity,
    start: Vector,
    *,
    draws: int,
    warmup: int,
    thin: int,
    widths: Vector,
    max_steps: int | None,
    stream: RandomStream,
) -> MarkovChain:
    """Run one chain of coordinate-wise slice-sampling sweeps.

    References:
        [EQ-INF-SLICE-001], [REF-NEAL-2003].

    """
    generator = stream.generator()
    target = _Counter(log_density)
    state = start
    current = target(state)
    if not math.isfinite(current):
        raise ValueError("initial point must have a finite log-density")
    samples: list[Vector] = []
    trace: list[float] = []
    for iteration in range(warmup + draws * thin):
        for index, width in enumerate(widths):
            state, current = _slice_coordinate(
                target, state, current, index, width, max_steps, generator
            )
        if iteration >= warmup and (iteration - warmup + 1) % thin == 0:
            samples.append(state)
            trace.append(current)
    dimension = len(start)
    covariance = tuple(
        tuple(widths[i] ** 2 if i == j else 0.0 for j in range(dimension))
        for i in range(dimension)
    )
    return MarkovChain(
        tuple(samples),
        tuple(trace),
        1.0,
        1.0 if warmup else math.nan,
        covariance,
        target.calls,
        stream.seed,
    )


def slice_sampler(
    log_density: LogDensity,
    initial: Sequence[float] | Sequence[Sequence[float]],
    *,
    draws: int,
    warmup: int = 0,
    chains: int = 1,
    thin: int = 1,
    widths: float | Sequence[float] = 1.0,
    max_steps: int | None = None,
    seed: int | RandomStream | None = None,
    names: Sequence[str] | None = None,
    check_convergence: bool = True,
) -> MCMCResult:
    """Sample a log-density by coordinate-wise slice sampling.

    One iteration updates every coordinate once, in order, with Neal's
    stepping-out and shrinkage procedures: the height is
    ``log y = log p(x) - E`` with ``E ~ Exp(1)``; the initial interval of
    width ``w_i`` is placed uniformly around ``x_i`` and extended by ``w_i``
    while its ends are inside the slice; candidates are drawn uniformly from
    the interval, which shrinks to each rejected candidate.

    Args:
        log_density: Target ``log p(x)`` up to a constant; ``-inf`` marks
            points outside the support.
        initial: One start shared by all chains, or one start per chain.
        draws: Kept draws per chain.
        warmup: Discarded initial iterations per chain (nothing is adapted).
        chains: Number of independent chains, seeded by ``RandomStream.spawn``.
        thin: Keep every ``thin``-th post-warm-up state.
        widths: Initial interval width ``w`` (one positive value, or one per
            coordinate). A value near the scale of the conditional
            distributions costs the fewest evaluations; any positive value
            is valid.
        max_steps: Limit ``m`` on the number of interval extensions per
            update (split at random between the two ends, which keeps the
            update valid); ``None`` extends until both ends leave the slice.
        seed: Integer seed or stream; ``None`` is not reproducible.
        names: Optional coordinate names.
        check_convergence: Run :func:`convergence_warnings` on the draws and
            emit each finding as a :class:`ConvergenceWarning`.

    Returns:
        Multi-chain result; ``acceptance_rate`` is ``1`` by construction and
        ``evaluations`` is the number of log-density calls.

    Raises:
        TypeError: If ``log_density`` is not callable.
        ValueError: For invalid run lengths or widths, ``max_steps < 1``, a
            start with non-finite log-density, a target returning NaN or
            ``+inf``, or an unlimited stepping-out that does not terminate.

    Equation:
        [EQ-INF-SLICE-001].

    References:
        [REF-NEAL-2003].

    """
    checked = _checked_log_density(log_density)
    draws, warmup, chains, thin = _validate_run(draws, warmup, chains, thin)
    starts = _initial_points(initial, chains)
    dimension = len(starts[0])
    if isinstance(widths, (int, float)):
        width_vector: Vector = (require_finite_real(widths, name="widths"),) * dimension
    else:
        width_vector = as_vector(widths, name="widths")
    if len(width_vector) != dimension:
        raise ValueError("widths must have one entry per coordinate")
    if any(width <= 0.0 for width in width_vector):
        raise ValueError("widths must be positive")
    steps: int | None = None
    if max_steps is not None:
        steps = require_non_negative_integer(max_steps, name="max_steps")
        if steps == 0:
            raise ValueError("max_steps must be at least 1")
    streams = resolve_stream(seed).spawn(chains, label="mcmc-chain")
    results = tuple(
        _run_slice_chain(
            checked,
            start,
            draws=draws,
            warmup=warmup,
            thin=thin,
            widths=width_vector,
            max_steps=steps,
            stream=stream,
        )
        for start, stream in zip(starts, streams, strict=True)
    )
    result = MCMCResult(
        results, _names(names, dimension), "slice sampling", warmup, thin
    )
    return _with_convergence_check(result, check_convergence)


def _run_elliptical_chain(
    log_likelihood: LogDensity,
    start: Vector,
    *,
    draws: int,
    warmup: int,
    thin: int,
    mean: Vector,
    covariance: Matrix,
    factor: Matrix,
    stream: RandomStream,
) -> MarkovChain:
    """Run one elliptical slice sampling chain.

    References:
        [EQ-INF-SLICE-002], [REF-MURRAY-ADAMS-MACKAY-2010, Figure 2].

    """
    generator = stream.generator()
    target = _Counter(log_likelihood)
    dimension = len(start)
    state = start
    current = target(state)
    if not math.isfinite(current):
        raise ValueError("initial point must have a finite log-likelihood")
    samples: list[Vector] = []
    trace: list[float] = []
    for iteration in range(warmup + draws * thin):
        normal = tuple(generator.gauss(0.0, 1.0) for _ in range(dimension))
        auxiliary = matvec(factor, normal)
        level = current + math.log(1.0 - generator.random())
        angle = _TWO_PI * generator.random()
        lower = angle - _TWO_PI
        upper = angle
        centered = tuple(x - m for x, m in zip(state, mean, strict=True))
        for _ in range(_SHRINK_GUARD):
            cosine = math.cos(angle)
            sine = math.sin(angle)
            proposal = tuple(
                m + c * cosine + a * sine
                for m, c, a in zip(mean, centered, auxiliary, strict=True)
            )
            value = target(proposal)
            if value > level:
                state, current = proposal, value
                break
            if angle < 0.0:
                lower = angle
            else:
                upper = angle
            angle = lower + (upper - lower) * generator.random()
        if iteration >= warmup and (iteration - warmup + 1) % thin == 0:
            samples.append(state)
            trace.append(current)
    return MarkovChain(
        tuple(samples),
        tuple(trace),
        1.0,
        1.0 if warmup else math.nan,
        covariance,
        target.calls,
        stream.seed,
    )


def elliptical_slice_sampler(
    log_likelihood: LogDensity,
    initial: Sequence[float] | Sequence[Sequence[float]],
    *,
    prior_covariance: Sequence[Sequence[float]],
    draws: int,
    prior_mean: Sequence[float] | None = None,
    warmup: int = 0,
    chains: int = 1,
    thin: int = 1,
    seed: int | RandomStream | None = None,
    names: Sequence[str] | None = None,
    check_convergence: bool = True,
) -> MCMCResult:
    """Sample ``pi(f) ∝ L(f) N(f; mu, Sigma)`` by elliptical slice sampling.

    Each iteration draws ``nu ~ N(0, Sigma)``, a height
    ``log y = log L(f) + log U``, and an angle ``theta ~ U[0, 2 pi)`` with
    bracket ``[theta - 2 pi, theta]``; it proposes
    ``f' = mu + (f - mu) cos(theta) + nu sin(theta)``, accepts the first
    proposal with ``log L(f') > log y``, and after each rejection shrinks the
    bracket (``theta_min = theta`` if ``theta < 0``, else
    ``theta_max = theta``) and redraws ``theta`` uniformly from it. The
    Gaussian prior is part of the target and is never evaluated: pass the
    log-likelihood only.

    Args:
        log_likelihood: ``log L(f)`` up to a constant; ``-inf`` marks zero
            likelihood.
        initial: One start shared by all chains, or one start per chain,
            with finite log-likelihood.
        prior_covariance: Symmetric positive-definite ``Sigma``.
        draws: Kept draws per chain.
        prior_mean: Prior mean ``mu`` (zero when omitted).
        warmup: Discarded initial iterations per chain (nothing is adapted).
        chains: Number of independent chains, seeded by ``RandomStream.spawn``.
        thin: Keep every ``thin``-th post-warm-up state.
        seed: Integer seed or stream; ``None`` is not reproducible.
        names: Optional coordinate names.
        check_convergence: Run :func:`convergence_warnings` on the draws and
            emit each finding as a :class:`ConvergenceWarning`.

    Returns:
        Multi-chain result. The ``log_density`` trace holds ``log L`` (the
        prior term is not evaluated), ``proposal_covariance`` is ``Sigma``,
        and ``evaluations`` counts log-likelihood calls.

    Raises:
        TypeError: If ``log_likelihood`` is not callable.
        ValueError: For invalid run lengths, a covariance that is not
            symmetric positive definite or does not match the dimension, a
            start with non-finite log-likelihood, or a likelihood returning
            NaN or ``+inf``.

    Equation:
        [EQ-INF-SLICE-002].

    References:
        [REF-MURRAY-ADAMS-MACKAY-2010].

    """
    checked = _checked_log_density(log_likelihood)
    draws, warmup, chains, thin = _validate_run(draws, warmup, chains, thin)
    starts = _initial_points(initial, chains)
    dimension = len(starts[0])
    covariance = symmetrize(prior_covariance, name="prior_covariance")
    if len(covariance) != dimension:
        raise ValueError("prior_covariance must match the state dimension")
    try:
        factor = cholesky(covariance)
    except ValueError as error:
        raise ValueError("prior_covariance must be positive definite") from error
    if prior_mean is None:
        mean: Vector = (0.0,) * dimension
    else:
        mean = as_vector(prior_mean, name="prior_mean")
        if len(mean) != dimension:
            raise ValueError("prior_mean must match the state dimension")
    streams = resolve_stream(seed).spawn(chains, label="mcmc-chain")
    results = tuple(
        _run_elliptical_chain(
            checked,
            start,
            draws=draws,
            warmup=warmup,
            thin=thin,
            mean=mean,
            covariance=covariance,
            factor=factor,
            stream=stream,
        )
        for start, stream in zip(starts, streams, strict=True)
    )
    result = MCMCResult(
        results, _names(names, dimension), "elliptical slice sampling", warmup, thin
    )
    return _with_convergence_check(result, check_convergence)
