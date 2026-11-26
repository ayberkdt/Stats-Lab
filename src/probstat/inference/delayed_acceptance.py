"""Exact-target two-stage delayed-acceptance Metropolis sampling.

Experimental (added in 0.28).

Setting
    The target ``pi`` is expensive to evaluate; ``pi~`` is a cheap surrogate
    (for example a coarse forward model inside a likelihood). Plain
    Metropolis-Hastings run on ``pi~`` alone targets the *wrong* law. Delayed
    acceptance (Christen and Fox 2005) keeps the exact target ``pi``
    invariant while spending an exact evaluation only on proposals that the
    surrogate has already accepted.

Kernel
    From state ``x`` draw ``x' ~ q(. | x)``.

    * Stage 1 (surrogate only): with probability
      ``a1 = min(1, pi~(x') q(x | x') / [pi~(x) q(x' | x)])`` continue,
      otherwise stay at ``x`` *without evaluating* ``pi`` ([EQ-INF-DA-001]).
    * Stage 2 (exact): with probability
      ``a2 = min(1, pi(x') pi~(x) / [pi(x) pi~(x')])`` move to ``x'``,
      otherwise stay ([EQ-INF-DA-002]).

    Detailed balance for a *fixed* surrogate with ``pi~ > 0`` on the support
    of ``pi``: for ``x != x'`` the flux is
    ``F(x, x') = pi(x) q(x'|x) min(1, r) min(1, s)`` with
    ``r = pi~(x') q(x|x') / [pi~(x) q(x'|x)]`` and
    ``s = pi(x') pi~(x) / [pi(x) pi~(x')]``. Swapping ``x`` and ``x'`` sends
    ``r -> 1/r`` and ``s -> 1/s``; in each of the four cases
    (``r >= 1`` or ``< 1``, ``s >= 1`` or ``< 1``) direct substitution gives
    ``F(x, x') = F(x', x)`` ([EQ-INF-DA-004]). Hence ``pi`` is reversible,
    therefore invariant, for the composite kernel.

Support policy
    * ``log_surrogate = -inf`` at a proposal: stage 1 rejects; no exact call.
    * ``log_target = -inf`` at a stage-1-accepted proposal: stage 2 rejects.
    * The initial state must have finite exact and surrogate values.
    * ``+inf`` and ``nan`` values raise :class:`ValueError`.
    * If the surrogate is zero (``-inf``) on a region where the target is
      positive, that region is unreachable: irreducibility is destroyed and
      the sampler is **not exact** for ``pi``. Exactness requires the
      surrogate support to contain the target support. This is not checked;
      it cannot be verified from samples.

Proposals
    ``"random_walk"``: ``x' = x + step * z``, ``z ~ N(0, I)``, ``step`` a
    positive scalar or per-coordinate scales; symmetric, so ``q`` cancels.
    ``"pcn"`` (Cotter et al. 2013): ``x' = sqrt(1 - beta^2) x + beta z`` with
    ``beta`` in ``(0, 1]``; it is reversible with respect to ``N(0, I)``. The
    prior must be exactly standard normal in the supplied coordinates, and
    ``log_target`` and ``log_surrogate`` must then be **log-likelihoods**
    (the prior cancels and is never evaluated); the invariant law is
    ``pi(x) ∝ L(x) N(x; 0, I)`` ([EQ-INF-DA-003]). Pass
    ``log_prior="standard_normal"`` to declare this; the random-walk proposal
    requires ``log_prior=None`` and ``log_target`` is then the full
    log-density.

No adaptation
    The proposal is fixed during the whole run. Adapting the proposal or the
    surrogate from the chain history would void the fixed-surrogate argument
    above; adapt only in a separate pilot run and pass the frozen result.

Accounting
    Evaluation counts and acceptance rates cover the whole run (burn-in
    included). The exact and surrogate values of the current state are
    cached and replaced only on final acceptance, so ``exact_evaluations``
    equals the number of stage-1 acceptances plus one (the initial state).
    ``ess_per_exact_evaluation`` divides the smallest per-coordinate effective
    sample size of the kept draws by that count ([EQ-INF-DA-005]).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.linalg import Vector

from .mcmc import effective_sample_size

LogFunction = Callable[[Vector], float]

__all__ = ["DelayedAcceptanceResult", "delayed_acceptance_metropolis"]


@dataclass(frozen=True, slots=True)
class DelayedAcceptanceResult:
    """Output of :func:`delayed_acceptance_metropolis`.

    Attributes:
        samples: Draws after burn-in, one tuple per iteration.
        log_target: Exact ``log_target`` at each kept draw.
        stage1_acceptance_rate: Fraction of all iterations passing stage 1.
        stage2_acceptance_rate: Fraction of stage-1 acceptances accepted at
            stage 2 (``nan`` if stage 1 never accepted).
        acceptance_rate: Fraction of all iterations ending in a move.
        exact_evaluations: Exact-target calls in the whole run.
        surrogate_evaluations: Surrogate calls in the whole run.
        effective_sample_size: Per-coordinate ESS of the kept draws
            (``nan`` when fewer than 4 draws or a constant coordinate).
        ess_per_exact_evaluation: Smallest finite per-coordinate ESS divided
            by ``exact_evaluations`` (``nan`` if none is finite).
        seed: Seed of the random stream.
        proposal: ``"random_walk"`` or ``"pcn"``.
        step: Fixed proposal scale per coordinate; ``(beta,)`` for pCN.
        burn_in: Discarded iterations.
        exact_target: Always ``True``: the invariant law is the exact target
            for a fixed surrogate whose support contains the target's.

    Equation:
        [EQ-INF-DA-004], [EQ-INF-DA-005].

    References:
        [REF-CHRISTEN-FOX-2005].

    """

    samples: tuple[Vector, ...]
    log_target: tuple[float, ...]
    stage1_acceptance_rate: float
    stage2_acceptance_rate: float
    acceptance_rate: float
    exact_evaluations: int
    surrogate_evaluations: int
    effective_sample_size: tuple[float, ...]
    ess_per_exact_evaluation: float
    seed: int | None
    proposal: str
    step: tuple[float, ...]
    burn_in: int
    exact_target: bool = True


def _checked(function: object, name: str) -> LogFunction:
    """Wrap a log-function so NaN and ``+inf`` raise.

    References:
        [REF-CHRISTEN-FOX-2005]. No equation is evaluated.

    """
    if not callable(function):
        raise TypeError(f"{name} must be callable")

    def call(point: Vector) -> float:
        """Evaluate and validate one value.

        References:
            [REF-CHRISTEN-FOX-2005]. No equation is evaluated.

        """
        value = float(function(point))
        if math.isnan(value):
            raise ValueError(f"{name} returned NaN")
        if value == math.inf:
            raise ValueError(f"{name} returned +inf")
        return value

    return call


def _positive_steps(step: float | Sequence[float], dimension: int) -> tuple[float, ...]:
    """Return per-coordinate positive finite scales.

    References:
        [EQ-INF-DA-001].

    """
    if isinstance(step, Sequence):
        values = tuple(require_finite_real(s, name="step") for s in step)
        if len(values) != dimension:
            raise ValueError("step must be a scalar or have one entry per coordinate")
    else:
        values = (require_finite_real(step, name="step"),) * dimension
    if any(s <= 0.0 for s in values):
        raise ValueError("step entries must be positive")
    return values


def delayed_acceptance_metropolis(
    log_target: LogFunction,
    log_surrogate: LogFunction,
    initial: Sequence[float],
    *,
    proposal: Literal["random_walk", "pcn"] = "random_walk",
    step: float | Sequence[float] = 1.0,
    samples: int,
    burn_in: int = 0,
    seed: int | RandomStream | None = None,
    log_prior: Literal["standard_normal"] | None = None,
) -> DelayedAcceptanceResult:
    """Sample the exact target by two-stage delayed-acceptance Metropolis.

    Proposals are screened by the cheap ``log_surrogate`` (stage 1); only the
    survivors are evaluated with the expensive ``log_target`` (stage 2),
    which corrects the surrogate bias exactly. See the module docstring for
    the kernel, the support policy and the pCN convention.

    Args:
        log_target: Exact (expensive) log-density, or log-likelihood for pCN.
        log_surrogate: Cheap surrogate of the same quantity. Its support must
            contain the target's, otherwise the sampler is not exact.
        initial: Start point with finite exact and surrogate values.
        proposal: ``"random_walk"`` or ``"pcn"``.
        step: Positive scalar or per-coordinate scales (random walk); a
            scalar ``beta`` in ``(0, 1]`` (pCN).
        samples: Number of kept draws (positive).
        burn_in: Discarded initial iterations (non-negative).
        seed: Integer seed or stream; ``None`` is not reproducible.
        log_prior: ``"standard_normal"`` for pCN (required: declares that
            the prior is ``N(0, I)`` and that the two callables are
            log-likelihoods); must be ``None`` for the random walk.

    Returns:
        A :class:`DelayedAcceptanceResult` with ``exact_target=True``.

    Raises:
        TypeError: If a callable is not callable or ``samples``/``burn_in``
            are not integers.
        ValueError: For a non-finite start, NaN or ``+inf`` function values,
            invalid ``step``, ``samples`` or ``burn_in``, an unknown
            ``proposal``, or an inconsistent ``log_prior`` declaration.

    Equation:
        [EQ-INF-DA-001], [EQ-INF-DA-002], [EQ-INF-DA-003].

    References:
        [REF-CHRISTEN-FOX-2005], [REF-COTTER-ROBERTS-STUART-WHITE-2013].

    """
    exact = _checked(log_target, "log_target")
    cheap = _checked(log_surrogate, "log_surrogate")
    if proposal not in ("random_walk", "pcn"):
        raise ValueError("proposal must be 'random_walk' or 'pcn'")
    for label, count in (("samples", samples), ("burn_in", burn_in)):
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError(f"{label} must be an integer")
    if samples < 1 or burn_in < 0:
        raise ValueError("samples must be positive and burn_in non-negative")
    state = tuple(require_finite_real(v, name="initial") for v in initial)
    if not state:
        raise ValueError("initial must be non-empty")
    dimension = len(state)
    beta = 0.0
    scales: tuple[float, ...]
    if proposal == "pcn":
        if log_prior != "standard_normal":
            raise ValueError(
                "pcn requires log_prior='standard_normal' (standard-normal prior "
                "and log-likelihood callables)"
            )
        if isinstance(step, Sequence):
            raise ValueError("pcn step must be a scalar beta")
        beta = require_finite_real(step, name="step")
        if not 0.0 < beta <= 1.0:
            raise ValueError("pcn step (beta) must lie in (0, 1]")
        scales = (beta,)
    else:
        if log_prior is not None:
            raise ValueError("log_prior must be None for the random-walk proposal")
        scales = _positive_steps(step, dimension)
    stream = resolve_stream(seed)
    generator = stream.generator()
    current_exact = exact(state)
    current_cheap = cheap(state)
    if not (math.isfinite(current_exact) and math.isfinite(current_cheap)):
        raise ValueError("initial point must have finite exact and surrogate values")
    exact_calls, cheap_calls = 1, 1
    stage1 = stage2 = 0
    kept: list[Vector] = []
    trace: list[float] = []
    contraction = math.sqrt(max(0.0, 1.0 - beta * beta))
    for iteration in range(burn_in + samples):
        if proposal == "pcn":
            moved = tuple(
                contraction * x + beta * generator.gauss(0.0, 1.0) for x in state
            )
        else:
            moved = tuple(
                x + s * generator.gauss(0.0, 1.0)
                for x, s in zip(state, scales, strict=True)
            )
        proposed_cheap = cheap(moved)
        cheap_calls += 1
        if proposed_cheap > -math.inf and (
            math.log(1.0 - generator.random()) < proposed_cheap - current_cheap
        ):
            stage1 += 1
            proposed_exact = exact(moved)
            exact_calls += 1
            if proposed_exact > -math.inf and math.log(1.0 - generator.random()) < (
                (proposed_exact - proposed_cheap) - (current_exact - current_cheap)
            ):
                stage2 += 1
                state, current_exact, current_cheap = (
                    moved,
                    proposed_exact,
                    proposed_cheap,
                )
        if iteration >= burn_in:
            kept.append(state)
            trace.append(current_exact)
    total = burn_in + samples
    ess = tuple(
        effective_sample_size([[draw[j] for draw in kept]])
        if samples >= 4
        else math.nan
        for j in range(dimension)
    )
    finite = [e for e in ess if math.isfinite(e)]
    return DelayedAcceptanceResult(
        samples=tuple(kept),
        log_target=tuple(trace),
        stage1_acceptance_rate=stage1 / total,
        stage2_acceptance_rate=stage2 / stage1 if stage1 else math.nan,
        acceptance_rate=stage2 / total,
        exact_evaluations=exact_calls,
        surrogate_evaluations=cheap_calls,
        effective_sample_size=ess,
        ess_per_exact_evaluation=min(finite) / exact_calls if finite else math.nan,
        seed=stream.seed,
        proposal=proposal,
        step=scales,
        burn_in=burn_in,
    )
