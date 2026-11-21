"""Sequential importance sampling for small failure probabilities.

The method samples the sequence of smooth targets ``h_k(u)`` proportional to
``Phi(-g(u) / sigma_k) phi_n(u)`` of :mod:`improved_cross_entropy` without a
parametric density: the sample of level ``k`` is reweighted towards level
``k + 1``, resampled, and moved by a Markov chain that leaves ``h_{k+1}``
invariant. The failure probability is the product of the mean incremental
weights times a final correction from the smooth indicator to the exact one.

The Markov kernel is the conditional-sampling (preconditioned
Crank-Nicolson) proposal of subset simulation, ``v = rho u + sqrt(1 -
rho^2) z``. It is reversible with respect to ``phi_n``, so the
Metropolis-Hastings ratio for the target ``Phi(-g / sigma) phi_n`` reduces to
the ratio of the two ``Phi`` terms.
"""

from __future__ import annotations

import math
import random
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.linalg import Vector

from .adaptive import _require_count, _require_limit_state
from .improved_cross_entropy import (
    ICE_TARGET_CV,
    _log_smooth_indicator,
    _stopping_cv,
    smoothing_scale,
)
from .model_based import ModelLimitState


def _smoothed_chain(
    evaluate: Callable[[Vector], float],
    generator: random.Random,
    start: Vector,
    start_value: float,
    sigma: float,
    length: int,
    burn_in: int,
    correlation: float,
    points: list[Vector],
    values: list[float],
) -> int:
    """Append one chain with invariant law ``Phi(-g / sigma) phi_n`` and count moves.

    Each step proposes ``v = rho u + sqrt(1 - rho^2) z`` and accepts it with
    probability ``min(1, Phi(-g(v) / sigma) / Phi(-g(u) / sigma))``. The
    first ``burn_in`` steps are discarded; the state after them is stored
    first, followed by ``length - 1`` further states.

    References:
        [EQ-REL-SIS-002], [EQ-REL-SUBSET-004],
        [REF-PAPAIOANNOU-PAPADIMITRIOU-STRAUB-2016],
        [REF-COTTER-ROBERTS-STUART-WHITE-2013].

    """
    spread = math.sqrt(1.0 - correlation * correlation)
    current, current_value = start, start_value
    current_log = _log_smooth_indicator(current_value, sigma)
    accepted = 0
    for step in range(burn_in + length - 1):
        if step == burn_in:
            points.append(current)
            values.append(current_value)
        proposal = tuple(
            correlation * coordinate + spread * generator.gauss(0.0, 1.0)
            for coordinate in current
        )
        proposal_value = evaluate(proposal)
        if not math.isfinite(proposal_value):
            raise ValueError("the limit state returned a non-finite value")
        proposal_log = _log_smooth_indicator(proposal_value, sigma)
        difference = proposal_log - current_log
        if difference >= 0.0 or generator.random() < math.exp(difference):
            current, current_value, current_log = (
                proposal,
                proposal_value,
                proposal_log,
            )
            accepted += 1
        if step >= burn_in:
            points.append(current)
            values.append(current_value)
    if length == 1:
        points.append(current)
        values.append(current_value)
    return accepted


def _batch_relative_variance(terms: Sequence[float], batch: int) -> float:
    """Return the squared c.o.v. of the mean of ``terms`` from batch means.

    ``terms`` is split into consecutive batches of ``batch`` values (one
    Markov chain each, or single values for an independent sample); the
    variance of the mean is the sample variance of the batch means divided
    by their number.

    References:
        [EQ-REL-SIS-004], [REF-GEYER-1992].

    """
    count = len(terms) // batch
    means = [
        math.fsum(terms[index * batch : (index + 1) * batch]) / batch
        for index in range(count)
    ]
    centre = math.fsum(means) / count
    if count < 2 or centre == 0.0:
        return math.inf
    variance = math.fsum((item - centre) ** 2 for item in means) / (count - 1)
    return variance / count / (centre * centre)


@dataclass(frozen=True, slots=True)
class SequentialImportanceResult:
    """Sequential importance-sampling estimate of a failure probability.

    Args:
        limit_state: Label of the analyzed limit state.
        probability: ``(prod_k S_k) C`` with the final correction ``C``.
        coefficient_of_variation: Heuristic c.o.v.: the root of the summed
            squared c.o.v. of every factor, each from the variance between
            chain means (level 0 from independent draws). It ignores the
            dependence between levels and between chains that share a
            resampled seed and typically understates the scatter between
            independent runs: over 60 runs each on linear limit states
            (``beta`` 3 to 5.5, 2 to 50 dimensions, defaults) the standard
            deviation between runs was 1.3 to 1.4 times the mean reported
            value, and 10 to 17 % of the estimates were further than two
            reported c.o.v. from the exact value on the log scale.
        standard_error: ``coefficient_of_variation * probability``.
        scales: Smoothing scales ``sigma_1 > ... > sigma_L``.
        ratios: Mean incremental weights ``S_1, ..., S_L``.
        correction: ``C = mean(I(g <= 0) / Phi(-g / sigma_L))`` over the last
            sample (the failure fraction when no level was needed).
        weight_coefficients_of_variation: C.o.v. of the incremental weights
            at each level (the target unless a boundary case of
            :func:`smoothing_scale` occurred).
        stopping_coefficients_of_variation: C.o.v. of ``I(g <= 0) /
            Phi(-g / sigma_k)`` for every level sample, level 0 first.
        acceptance_rates: Accepted fraction of proposed moves per level.
        level_correlations: Proposal correlation ``rho`` at the end of each
            level.
        levels: Number of smooth levels ``L``.
        samples_per_level: Samples ``N`` per level.
        chains: Chains per level.
        failures: Failed points in the last sample.
        evaluations: Model evaluations.
        converged: Whether the stopping c.o.v. reached the target within
            ``max_levels``.

    References:
        [REF-PAPAIOANNOU-PAPADIMITRIOU-STRAUB-2016].

    """

    limit_state: str
    probability: float
    coefficient_of_variation: float
    standard_error: float
    scales: Vector
    ratios: Vector
    correction: float
    weight_coefficients_of_variation: Vector
    stopping_coefficients_of_variation: Vector
    acceptance_rates: Vector
    level_correlations: Vector
    levels: int
    samples_per_level: int
    chains: int
    failures: int
    evaluations: int
    converged: bool


def sequential_importance_sampling(
    limit_state: ModelLimitState,
    *,
    samples_per_level: int = 1000,
    target_cv: float = ICE_TARGET_CV,
    chain_fraction: float = 0.02,
    burn_in: int = 10,
    initial_scale: float = 0.6,
    target_acceptance: float = 0.44,
    adaptation_chains: int = 2,
    max_levels: int = 50,
    seed: int | RandomStream | None = None,
) -> SequentialImportanceResult:
    """Estimate ``P_f`` by sequential importance sampling.

    Unnormalized targets: ``eta_0 = phi_n`` and ``eta_k(u) =
    Phi(-g(u) / sigma_k) phi_n(u)`` for ``k >= 1``, with normalizing
    constants ``P_0 = 1`` and ``P_k = E_phi[Phi(-g / sigma_k)]``. Because
    ``I(g <= 0) phi_n = [I(g <= 0) / Phi(-g / sigma_L)] eta_L``::

        P_f = P_L E_{h_L}[I(g <= 0) / Phi(-g / sigma_L)],
        P_L = prod_{k=1..L} P_k / P_{k-1}.

    The level-0 target is ``phi_n`` itself, so the first incremental weight
    is ``w_1 = Phi(-g / sigma_1)``, not ``Phi(-g / sigma_1) / Phi(0)``; with
    the factor ``Phi(0) = 1/2`` the product would be twice too large.

    Algorithm. Level 0 draws ``N`` independent standard-normal points. At
    level ``k`` (sample approximately from ``h_k``):

    1. Stop if the c.o.v. of ``I(g <= 0) / Phi(-g / sigma_k)`` over the
       sample is at most ``target_cv`` (for ``k = 0`` the c.o.v. of the
       indicator).
    2. Choose ``sigma_{k+1} < sigma_k`` by :func:`smoothing_scale` so that
       the incremental weights ``w = Phi(-g / sigma_{k+1}) /
       Phi(-g / sigma_k)`` have c.o.v. ``target_cv``; ``S_{k+1} = mean(w)``
       estimates ``P_{k+1} / P_k``.
    3. Draw ``N_c = chain_fraction N`` seeds with probabilities
       proportional to ``w`` (with replacement) and run from each a chain of
       ``N / N_c`` states with the kernel of [EQ-REL-SIS-002] after
       ``burn_in`` discarded steps. The proposal spread
       ``sqrt(1 - rho^2) = min(lambda, 1)`` is adapted every
       ``adaptation_chains`` chains towards ``target_acceptance`` by the rule
       of adaptive conditional sampling [EQ-REL-SUBSET-005] and carried to
       the next level.

    The estimator is consistent as ``N`` grows. For finite ``N`` the level
    samples are only approximately distributed as ``h_k`` (resampled seeds
    plus chains that do not forget their start). A level sample that lags
    behind its target biases the next ratio in the same direction, so the
    errors of successive ratios add up instead of cancelling. The defaults
    (20 long chains per 1000 samples and 10 burn-in steps) are choices of
    this library made for that reason: with 100 chains of 10 states and no
    burn-in the scatter between runs on a linear limit state with
    ``beta = 4.5`` in 20 dimensions was about four times larger at a similar
    cost (see ADR-0124). ``burn_in`` costs ``N_c burn_in`` evaluations per
    level.

    Args:
        limit_state: Model limit state.
        samples_per_level: Samples ``N`` per level (at least 10).
        target_cv: Target c.o.v. of the incremental weights and of the
            stopping ratio (positive).
        chain_fraction: Seeds per level as a fraction of ``N`` in
            ``(0, 1]``; ``N chain_fraction`` must be an integer of at least
            2 that divides ``N``.
        burn_in: Discarded steps at the start of every chain.
        initial_scale: Starting spread factor ``lambda > 0``.
        target_acceptance: Target acceptance rate in ``(0, 1)``.
        adaptation_chains: Chains between two adaptations of ``lambda``.
        max_levels: Maximum number of smooth levels.
        seed: Integer seed, random stream, or ``None`` for fresh entropy.

    Returns:
        Estimate, heuristic c.o.v., level history, acceptance rates,
        evaluation count, and convergence flag. An unconverged run returns
        the estimate of its last level and is unreliable.

    Raises:
        TypeError: If ``limit_state`` has the wrong type.
        ValueError: If an option is invalid or a limit-state value is not
            finite.

    Warns:
        UserWarning: If the stopping criterion was not reached.

    Equation:
        [EQ-REL-ICE-001], [EQ-REL-ICE-002], [EQ-REL-ICE-004],
        [EQ-REL-SIS-001], [EQ-REL-SIS-002], [EQ-REL-SIS-003],
        [EQ-REL-SIS-004], [EQ-REL-SUBSET-005].

    References:
        [REF-PAPAIOANNOU-PAPADIMITRIOU-STRAUB-2016],
        [REF-PAPAIOANNOU-BETZ-ZWIRGLMAIER-STRAUB-2015],
        [REF-COTTER-ROBERTS-STUART-WHITE-2013].

    """
    state = _require_limit_state(limit_state)
    count = _require_count(samples_per_level, "samples_per_level", 10)
    target = require_finite_real(target_cv, name="target_cv")
    if target <= 0.0:
        raise ValueError("target_cv must be positive")
    fraction = require_finite_real(chain_fraction, name="chain_fraction")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("chain_fraction must lie in (0, 1]")
    chains = round(count * fraction)
    if chains < 2 or abs(chains - count * fraction) > 1e-9 or count % chains != 0:
        raise ValueError(
            "samples_per_level * chain_fraction must be an integer of at least "
            "2 that divides samples_per_level"
        )
    length = count // chains
    discard = _require_count(burn_in, "burn_in", 0)
    if length == 1 and discard == 0:
        raise ValueError("chains of one state need burn_in >= 1 to move the seeds")
    scale = require_finite_real(initial_scale, name="initial_scale")
    if scale <= 0.0:
        raise ValueError("initial_scale must be positive")
    rate_target = require_finite_real(target_acceptance, name="target_acceptance")
    if not 0.0 < rate_target < 1.0:
        raise ValueError("target_acceptance must lie in (0, 1)")
    window_size = _require_count(adaptation_chains, "adaptation_chains", 1)
    levels_limit = _require_count(max_levels, "max_levels", 1)
    generator = resolve_stream(seed).child("sequential-importance").generator()
    dimension = state.dimension
    before = state.model.evaluations
    points: list[Vector] = [
        tuple(generator.gauss(0.0, 1.0) for _ in range(dimension)) for _ in range(count)
    ]
    values = [state.standard(u) for u in points]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("the limit state returned a non-finite value")
    sigma = math.inf
    scales: list[float] = []
    ratios: list[float] = []
    weight_cvs: list[float] = []
    stopping: list[float] = []
    acceptance: list[float] = []
    correlations: list[float] = []
    relative_variance = 0.0
    batch = 1
    converged = False
    moves_per_chain = discard + length - 1
    while True:
        stop = _stopping_cv(values, sigma)
        stopping.append(stop)
        if stop <= target:
            converged = True
            break
        if len(scales) >= levels_limit:
            break
        # Level 0 is phi_n itself: its smoothing factor is 1, not Phi(0) = 1/2.
        base = (
            [0.0] * count
            if not scales
            else [-_log_smooth_indicator(value, sigma) for value in values]
        )
        sigma, attained = smoothing_scale(
            values, base, previous=sigma, target_cv=target
        )
        weights = [
            math.exp(_log_smooth_indicator(value, sigma) + offset)
            for value, offset in zip(values, base, strict=True)
        ]
        total = math.fsum(weights)
        if not total > 0.0:
            raise ValueError("every incremental weight underflowed to zero")
        scales.append(sigma)
        weight_cvs.append(attained)
        ratios.append(total / count)
        relative_variance += _batch_relative_variance(weights, batch)
        seeds = generator.choices(range(count), weights=weights, k=chains)
        new_points: list[Vector] = []
        new_values: list[float] = []
        accepted = 0
        window: list[float] = []
        updates = 0
        spread = min(scale, 1.0)
        correlation = math.sqrt(1.0 - spread * spread)
        for chain, index in enumerate(seeds):
            gained = _smoothed_chain(
                state.standard,
                generator,
                points[index],
                values[index],
                sigma,
                length,
                discard,
                correlation,
                new_points,
                new_values,
            )
            accepted += gained
            window.append(gained / moves_per_chain)
            if (chain + 1) % window_size == 0:
                rate = math.fsum(window) / len(window)
                window.clear()
                scale = math.exp(
                    math.log(scale) + (rate - rate_target) / math.sqrt(updates + 1)
                )
                updates += 1
                spread = min(scale, 1.0)
                correlation = math.sqrt(1.0 - spread * spread)
        acceptance.append(accepted / (chains * moves_per_chain))
        correlations.append(correlation)
        points, values = new_points, new_values
        batch = length
    final_terms = [
        math.exp(-_log_smooth_indicator(value, sigma)) if value <= 0.0 else 0.0
        for value in values
    ]
    if not scales:
        final_terms = [1.0 if value <= 0.0 else 0.0 for value in values]
    correction = math.fsum(final_terms) / count
    probability = math.prod(ratios) * correction
    relative_variance += _batch_relative_variance(final_terms, batch)
    cov = math.sqrt(relative_variance) if probability > 0.0 else math.inf
    if not converged:
        warnings.warn(
            "sequential importance sampling did not reach the stopping criterion "
            f"within {levels_limit} levels; the estimate is unreliable",
            stacklevel=2,
        )
    return SequentialImportanceResult(
        limit_state=state.label,
        probability=probability,
        coefficient_of_variation=cov,
        standard_error=cov * probability if probability > 0.0 else math.inf,
        scales=tuple(scales),
        ratios=tuple(ratios),
        correction=correction,
        weight_coefficients_of_variation=tuple(weight_cvs),
        stopping_coefficients_of_variation=tuple(stopping),
        acceptance_rates=tuple(acceptance),
        level_correlations=tuple(correlations),
        levels=len(scales),
        samples_per_level=count,
        chains=chains,
        failures=sum(1 for value in values if value <= 0.0),
        evaluations=state.model.evaluations - before,
        converged=converged,
    )
