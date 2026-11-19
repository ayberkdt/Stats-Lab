"""Preconditioned Crank-Nicolson MCMC for targets with a Gaussian reference law.

The sampler targets densities of the form ``pi(u) ∝ L(u) N(u; 0, I)``: a
standard-normal prior (or reference) law reweighted by a likelihood ``L``.
Writing ``Phi(u) = -log L(u)``, the proposal of Cotter, Roberts, Stuart and
White (2013)

    ``v = sqrt(1 - beta^2) u + beta xi``, ``xi ~ N(0, I)``, ``0 < beta <= 1``,

is an AR(1) step that is reversible with respect to ``N(0, I)``. The Gaussian
factors therefore cancel in the Metropolis-Hastings ratio and the move is
accepted with probability ``min(1, exp(Phi(u) - Phi(v)))`` ([EQ-MCMC-PCN-001]).
Only the likelihood enters the acceptance test.

Why the step does not shrink with dimension
    A random-walk proposal ``v = u + s xi`` has to scale ``s ~ d^(-1/2)``
    because the prior part of the ratio, ``exp((|u|^2 - |v|^2) / 2)``,
    degenerates as the dimension ``d`` grows. The pCN proposal leaves the
    prior exactly invariant, so when the likelihood depends on a
    finite-dimensional (or rapidly decaying) part of ``u``, as for a
    discretized random field observed at a few points, the acceptance rate
    for a fixed ``beta`` has a limit as the discretization is refined (Cotter
    et al. 2013, Figure 1; Hairer, Stuart and Vollmer 2014 prove a
    dimension-independent spectral gap). ``beta = 1`` is the independence
    sampler that proposes fresh prior draws.

Warm-up adaptation
    During warm-up only, ``beta = 1 / (1 + exp(-eta))`` is adapted by a
    Robbins-Monro recursion on the log-odds,
    ``eta_k = eta_{k-1} + k^(-0.6) (alpha_k - alpha*)``, where ``alpha_k`` is
    the acceptance *probability* of iteration ``k`` (less noisy than the
    accept indicator) and ``alpha*`` the target ([EQ-MCMC-PCN-002]; Andrieu
    and Thoms 2008). The step is frozen at the end of warm-up, so
    every kept draw comes from one time-homogeneous kernel and the
    diagnostics of :mod:`probstat.inference.mcmc` apply unchanged.

Default target acceptance 0.25
    There is no optimal-scaling theorem for pCN comparable to the 0.234 rule
    of random-walk Metropolis. Cotter et al. (2013) tune ``beta`` to an
    average acceptance of about 0.234 (Sections 5.1.3 and 5.3.4) or 25 %
    (Section 5.2.4), and report one problem (Section 5.2.5) in which a value
    near 50 % converged faster. 0.25 follows their common practice; it is a
    tuning heuristic, not an optimum, and ``target_acceptance`` exposes it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.linalg import Matrix, Vector

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

_GAIN_EXPONENT = 0.6


def _logistic(value: float) -> float:
    """Return ``1 / (1 + exp(-value))`` without overflow.

    References:
        [EQ-MCMC-PCN-002], [REF-ANDRIEU-THOMS-2008]. No new equation is
        introduced.

    """
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _identity_scaled(dimension: int, variance: float) -> Matrix:
    """Return ``variance * I``.

    References:
        [EQ-MCMC-PCN-001]. No new equation is introduced.

    """
    return tuple(
        tuple(variance if i == j else 0.0 for j in range(dimension))
        for i in range(dimension)
    )


def _run_pcn_chain(
    log_likelihood: LogDensity,
    start: Vector,
    *,
    draws: int,
    warmup: int,
    thin: int,
    step: float,
    target: float,
    adapt: bool,
    stream: RandomStream,
) -> MarkovChain:
    """Run one pCN chain, adapting ``beta`` during warm-up only.

    References:
        [EQ-MCMC-PCN-001], [EQ-MCMC-PCN-002],
        [REF-COTTER-ROBERTS-STUART-WHITE-2013], [REF-ANDRIEU-THOMS-2008].

    """
    generator = stream.generator()
    dimension = len(start)
    evaluations = 1
    state = start
    current = log_likelihood(state)
    if not math.isfinite(current):
        raise ValueError("initial point must have a finite log-likelihood")
    beta = step
    # beta = 1 (independence sampler) starts adaptation at logistic(30).
    log_odds = math.log(beta) - math.log1p(-beta) if beta < 1.0 else 30.0
    warm_accepted = 0
    kept_accepted = 0
    samples: list[Vector] = []
    trace: list[float] = []
    total = warmup + draws * thin
    for iteration in range(total):
        contraction = math.sqrt(max(0.0, 1.0 - beta * beta))
        proposal = tuple(
            contraction * value + beta * generator.gauss(0.0, 1.0) for value in state
        )
        proposed = log_likelihood(proposal)
        evaluations += 1
        difference = proposed - current
        accepted = math.log(1.0 - generator.random()) < difference
        if accepted:
            state, current = proposal, proposed
        if iteration < warmup:
            warm_accepted += accepted
            if adapt:
                probability = 1.0 if difference >= 0.0 else math.exp(difference)
                gain = (iteration + 1) ** -_GAIN_EXPONENT
                log_odds += gain * (probability - target)
                beta = _logistic(log_odds)
        else:
            kept_accepted += accepted
            if (iteration - warmup + 1) % thin == 0:
                samples.append(state)
                trace.append(current - 0.5 * math.fsum(x * x for x in state))
    return MarkovChain(
        tuple(samples),
        tuple(trace),
        kept_accepted / (draws * thin),
        warm_accepted / warmup if warmup else math.nan,
        _identity_scaled(dimension, beta * beta),
        evaluations,
        stream.seed,
    )


def preconditioned_crank_nicolson(
    log_likelihood: LogDensity,
    initial: Sequence[float] | Sequence[Sequence[float]],
    *,
    draws: int,
    warmup: int,
    chains: int = 4,
    thin: int = 1,
    step: float = 0.25,
    target_acceptance: float = 0.25,
    adapt: bool = True,
    seed: int | RandomStream | None = None,
    names: Sequence[str] | None = None,
    check_convergence: bool = True,
) -> MCMCResult:
    """Sample ``pi(u) ∝ L(u) N(u; 0, I)`` by preconditioned Crank-Nicolson.

    Each iteration proposes ``v = sqrt(1 - beta^2) u + beta xi`` with
    ``xi ~ N(0, I)`` and accepts it when
    ``log U < log L(v) - log L(u)``. The standard-normal reference law is
    part of the target and is never evaluated: pass the log-likelihood only,
    in coordinates in which the prior is exactly ``N(0, I)`` (for example the
    Nataf standard-normal space of a model's inputs). With ``adapt`` the
    step ``beta`` is tuned during warm-up by a Robbins-Monro recursion on its
    log-odds towards ``target_acceptance`` and then frozen; each chain adapts
    from its own history only.

    Args:
        log_likelihood: ``log L(u)``; ``-inf`` marks zero likelihood (always
            rejected). Normalizing constants are irrelevant here.
        initial: One start shared by all chains, or one start per chain.
        draws: Kept draws per chain.
        warmup: Discarded (and, with ``adapt``, adaptation) iterations per
            chain; ``0`` runs the fixed step ``step``.
        chains: Number of independent chains, seeded by ``RandomStream.spawn``.
        thin: Keep every ``thin``-th post-warm-up state.
        step: Initial (or, without adaptation, fixed) ``beta`` in ``(0, 1]``.
        target_acceptance: Warm-up target ``alpha*`` in ``(0, 1)``; 0.25
            follows the practice of Cotter et al. (2013) (module docstring).
        adapt: Adapt ``beta`` during warm-up.
        seed: Integer seed or stream; ``None`` is not reproducible.
        names: Optional coordinate names.
        check_convergence: Run :func:`convergence_warnings` on the draws and
            emit each finding as a :class:`ConvergenceWarning`.

    Returns:
        Multi-chain result. Each chain's ``proposal_covariance`` is the
        frozen innovation covariance ``beta^2 I`` (so ``beta`` is the square
        root of a diagonal entry), and its ``log_density`` trace holds
        ``log L(u) - |u|^2 / 2``, the target log-density up to a constant.

    Raises:
        TypeError: If ``log_likelihood`` is not callable.
        ValueError: For invalid run lengths, a ``step`` outside ``(0, 1]``, a
            ``target_acceptance`` outside ``(0, 1)``, a start with non-finite
            log-likelihood, or a likelihood returning NaN or ``+inf``.

    Equation:
        [EQ-MCMC-PCN-001], [EQ-MCMC-PCN-002].

    References:
        [REF-COTTER-ROBERTS-STUART-WHITE-2013],
        [REF-HAIRER-STUART-VOLLMER-2014], [REF-ANDRIEU-THOMS-2008].

    """
    checked = _checked_log_density(log_likelihood)
    draws, warmup, chains, thin = _validate_run(draws, warmup, chains, thin)
    beta = require_finite_real(step, name="step")
    if not 0.0 < beta <= 1.0:
        raise ValueError("step must lie in (0, 1]")
    target = require_finite_real(target_acceptance, name="target_acceptance")
    if not 0.0 < target < 1.0:
        raise ValueError("target_acceptance must lie in (0, 1)")
    if not isinstance(adapt, bool):
        raise TypeError("adapt must be a bool")
    starts = _initial_points(initial, chains)
    dimension = len(starts[0])
    streams = resolve_stream(seed).spawn(chains, label="mcmc-chain")
    results = tuple(
        _run_pcn_chain(
            checked,
            start,
            draws=draws,
            warmup=warmup,
            thin=thin,
            step=beta,
            target=target,
            adapt=adapt,
            stream=stream,
        )
        for start, stream in zip(starts, streams, strict=True)
    )
    result = MCMCResult(
        results,
        _names(names, dimension),
        "preconditioned Crank-Nicolson",
        warmup,
        thin,
    )
    return _with_convergence_check(result, check_convergence)
