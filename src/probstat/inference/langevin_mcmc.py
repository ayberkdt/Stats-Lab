"""Metropolis-adjusted Langevin algorithm (MALA).

The Langevin diffusion ``dX = (1/2) grad log pi(X) dt + dW`` has ``pi`` as
its stationary law. Its Euler discretization with time step ``eps^2``,

    ``x' = x + (eps^2 / 2) grad log pi(x) + eps xi``, ``xi ~ N(0, I)``,

does not: the unadjusted chain has a bias of the order of the step and can
be transient for light-tailed targets (Roberts and Tweedie 1996). MALA uses
the Euler step as a Metropolis-Hastings proposal. The proposal density is
``q(x' | x) = N(x'; x + (eps^2 / 2) grad log pi(x), eps^2 I)``, which is not
symmetric, so the acceptance probability keeps both proposal terms
([EQ-INF-MALA-001]):

    ``alpha = min(1, pi(x') q(x | x') / (pi(x) q(x' | x)))``.

Compared with a random walk, the drift moves proposals towards high density
and allows larger steps in moderate and high dimension; compared with
Hamiltonian Monte Carlo it takes a single gradient step per iteration.

Limits stated by the source
    Roberts and Tweedie show that MALA is not geometrically ergodic when the
    gradient grows faster than linearly in the tails (targets lighter than
    Gaussian): from a far-out start the drift overshoots and nearly every
    proposal is rejected. Start in the bulk, and prefer a smaller step or
    another sampler when chains stick. The step is fixed here (no
    adaptation); tune it from the reported acceptance rate.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.linalg import Vector, as_vector

from .hamiltonian import GradientFunction, _Target
from .mcmc import (
    LogDensity,
    MarkovChain,
    MCMCResult,
    _initial_points,
    _names,
    _validate_run,
    _with_convergence_check,
)


def mala_log_acceptance_ratio(
    log_density_current: float,
    log_density_proposal: float,
    current: Sequence[float],
    proposal: Sequence[float],
    gradient_current: Sequence[float],
    gradient_proposal: Sequence[float],
    step_size: float,
) -> float:
    """Return the log Metropolis-Hastings ratio of a Langevin proposal.

    With ``m(x) = x + (eps^2 / 2) grad log pi(x)`` the value is

        ``log pi(x') - log pi(x)
        - |x - m(x')|^2 / (2 eps^2) + |x' - m(x)|^2 / (2 eps^2)``;

    the move is accepted with probability ``min(1, exp(value))``.

    Args:
        log_density_current: ``log pi(x)``.
        log_density_proposal: ``log pi(x')``.
        current: State ``x``.
        proposal: Proposed state ``x'``.
        gradient_current: ``grad log pi(x)``.
        gradient_proposal: ``grad log pi(x')``.
        step_size: ``eps > 0``.

    Returns:
        ``log(pi(x') q(x | x') / (pi(x) q(x' | x)))``.

    Raises:
        ValueError: For mismatched lengths, non-finite inputs, or a
            non-positive step.

    Equation:
        [EQ-INF-MALA-001].

    References:
        [REF-ROBERTS-TWEEDIE-1996].

    """
    eps = require_finite_real(step_size, name="step_size")
    if eps <= 0.0:
        raise ValueError("step_size must be positive")
    x = as_vector(current, name="current")
    y = as_vector(proposal, name="proposal")
    gx = as_vector(gradient_current, name="gradient_current")
    gy = as_vector(gradient_proposal, name="gradient_proposal")
    if not len(x) == len(y) == len(gx) == len(gy):
        raise ValueError("states and gradients must have the same length")
    return _log_ratio(
        require_finite_real(log_density_current, name="log_density_current"),
        require_finite_real(log_density_proposal, name="log_density_proposal"),
        x,
        y,
        gx,
        gy,
        eps,
    )


def _log_ratio(
    value_x: float,
    value_y: float,
    x: Vector,
    y: Vector,
    gx: Vector,
    gy: Vector,
    eps: float,
) -> float:
    """Evaluate the log Metropolis-Hastings ratio without validation.

    References:
        [EQ-INF-MALA-001], [REF-ROBERTS-TWEEDIE-1996].

    """
    drift = 0.5 * eps * eps
    forward = math.fsum(
        (b - a - drift * g) ** 2 for a, b, g in zip(x, y, gx, strict=True)
    )
    backward = math.fsum(
        (a - b - drift * g) ** 2 for a, b, g in zip(x, y, gy, strict=True)
    )
    return value_y - value_x + (forward - backward) / (2.0 * eps * eps)


def _run_mala_chain(
    target: _Target,
    start: Vector,
    *,
    draws: int,
    warmup: int,
    thin: int,
    step: float,
    stream: RandomStream,
) -> MarkovChain:
    """Run one MALA chain with a fixed step.

    References:
        [EQ-INF-MALA-001], [REF-ROBERTS-TWEEDIE-1996].

    """
    generator = stream.generator()
    before = target.density_calls
    state = start
    current, gradient = target.start(state)
    drift = 0.5 * step * step
    warm_accepted = 0
    kept_accepted = 0
    samples: list[Vector] = []
    trace: list[float] = []
    for iteration in range(warmup + draws * thin):
        proposal = tuple(
            x + drift * g + step * generator.gauss(0.0, 1.0)
            for x, g in zip(state, gradient, strict=True)
        )
        value, proposed_gradient = target.evaluate(proposal)
        uniform = math.log(1.0 - generator.random())
        accepted = False
        if proposed_gradient is not None:
            ratio = _log_ratio(
                current, value, state, proposal, gradient, proposed_gradient, step
            )
            accepted = uniform < ratio
        if accepted and proposed_gradient is not None:
            state, current, gradient = proposal, value, proposed_gradient
        if iteration < warmup:
            warm_accepted += accepted
        else:
            kept_accepted += accepted
            if (iteration - warmup + 1) % thin == 0:
                samples.append(state)
                trace.append(current)
    dimension = len(start)
    return MarkovChain(
        tuple(samples),
        tuple(trace),
        kept_accepted / (draws * thin),
        warm_accepted / warmup if warmup else math.nan,
        tuple(
            tuple(step * step if i == j else 0.0 for j in range(dimension))
            for i in range(dimension)
        ),
        target.density_calls - before,
        stream.seed,
    )


def metropolis_adjusted_langevin(
    log_density: LogDensity,
    initial: Sequence[float] | Sequence[Sequence[float]],
    *,
    step_size: float,
    draws: int,
    warmup: int = 0,
    chains: int = 1,
    thin: int = 1,
    gradient: GradientFunction | str = "auto",
    relative_step: float = 1e-6,
    seed: int | RandomStream | None = None,
    names: Sequence[str] | None = None,
    check_convergence: bool = True,
) -> MCMCResult:
    """Sample a log-density by the Metropolis-adjusted Langevin algorithm.

    Each iteration proposes
    ``x' = x + (eps^2 / 2) grad log pi(x) + eps xi`` with ``xi ~ N(0, I)``
    and accepts it with probability
    ``min(1, pi(x') q(x | x') / (pi(x) q(x' | x)))``, where ``q`` is the
    Gaussian density of the proposal (:func:`mala_log_acceptance_ratio`).
    A proposal where the log-density or its gradient is not finite is
    rejected.

    Args:
        log_density: Target ``log pi(x)`` up to a constant, differentiable
            on its support.
        initial: One start shared by all chains, or one start per chain.
        step_size: Fixed ``eps > 0`` (the proposal standard deviation; the
            drift is ``eps^2 / 2`` times the gradient).
        draws: Kept draws per chain.
        warmup: Discarded initial iterations per chain (nothing is adapted).
        chains: Number of independent chains, seeded by ``RandomStream.spawn``.
        thin: Keep every ``thin``-th post-warm-up state.
        gradient: ``grad log pi``: a callable, or one of the options of
            :class:`~probstat.numerics.GradientEvaluator` (``"auto"`` uses
            forward-mode autodiff and falls back to finite differences).
            The chain targets ``pi`` exactly only with an exact gradient;
            a finite-difference gradient is deterministic, so the
            Metropolis-Hastings correction stays valid for the proposal
            actually used.
        relative_step: Finite-difference step relative to ``max(1, |x_i|)``.
        seed: Integer seed or stream; ``None`` is not reproducible.
        names: Optional coordinate names.
        check_convergence: Run :func:`convergence_warnings` on the draws and
            emit each finding as a :class:`ConvergenceWarning`.

    Returns:
        Multi-chain result. ``proposal_covariance`` is ``eps^2 I`` and
        ``evaluations`` counts calls of ``log_density`` (one dual-number
        call returns value and gradient).

    Raises:
        TypeError: If ``log_density`` is not callable.
        ValueError: For invalid run lengths, a non-positive step, or a start
            where the log-density or its gradient is not finite.

    Equation:
        [EQ-INF-MALA-001].

    References:
        [REF-ROBERTS-TWEEDIE-1996].

    """
    if not callable(log_density):
        raise TypeError("log_density must be callable")
    draws, warmup, chains, thin = _validate_run(draws, warmup, chains, thin)
    eps = require_finite_real(step_size, name="step_size")
    if eps <= 0.0:
        raise ValueError("step_size must be positive")
    h_step = require_finite_real(relative_step, name="relative_step")
    if h_step <= 0.0:
        raise ValueError("relative_step must be positive")
    starts = _initial_points(initial, chains)
    dimension = len(starts[0])
    target = _Target(log_density, gradient, h_step)
    streams = resolve_stream(seed).spawn(chains, label="mcmc-chain")
    results = tuple(
        _run_mala_chain(
            target,
            start,
            draws=draws,
            warmup=warmup,
            thin=thin,
            step=eps,
            stream=stream,
        )
        for start, stream in zip(starts, streams, strict=True)
    )
    result = MCMCResult(
        results,
        _names(names, dimension),
        "Metropolis-adjusted Langevin",
        warmup,
        thin,
    )
    return _with_convergence_check(result, check_convergence)
