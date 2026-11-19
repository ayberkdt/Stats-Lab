"""AK-SS: active-learning Kriging (GP) surrogate combined with subset simulation.

AK-MCS (:mod:`probstat.reliability.active_learning`) needs a Monte Carlo
population of about ``10 / P_f`` points or more, which a pure-Python GP
cannot classify when ``P_f`` is ``1e-5`` or smaller. Huang, Chen, and Zhu
replace the population by the samples of a subset simulation run on the
surrogate: the levels concentrate the candidates on the path from the bulk
of the input law to the failure region, so a few thousand candidates per
level represent a failure probability that crude Monte Carlo would need
millions of points for. The scheme implemented here:

1. fit a GP to ``G`` on a space-filling design in ``[-b, b]^d`` (standard
   space), optionally with extra seed points;
2. run subset simulation on the surrogate mean ``mu(u)`` (no model calls)
   with a chosen Markov-chain kernel; the distinct samples of all levels form
   the candidate pool;
3. stop if the surrogate's uncertainty about the last level's failure count
   is small (relative ``P_f`` bounds from ``mu -/+ 1.96 sigma``);
4. otherwise add the pool point with the best learning value (``min U`` by
   default) to the training set (one model call), update the posterior, and
   repeat until the learning function's stopping value holds on the pool;
   then rerun subset simulation on the refined surrogate (step 2).

Every rerun uses the same random stream (common random numbers), so the
pools of consecutive rounds coincide where the surrogate has not changed.

The candidate pool and the posterior updates use the
:class:`~probstat.reliability.PopulationPosterior` protocol (ADR-0107); the
level kernels are those of :func:`~probstat.reliability.subset_simulation`
(ADR-0106). This module depends on :mod:`probstat.surrogates` (same
architecture layer, ADR-0041).
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.linalg import Vector
from probstat.surrogates import (
    KERNELS,
    GaussianProcessRegression,
    GaussianProcessRegularization,
    ProbabilisticSurrogate,
    SurrogateFactory,
)
from probstat.surrogates import fit_gaussian_process as _fit_gp

from .active_learning import (
    AKMCS_DESIGNS,
    _design_points,
    _extra_points,
    _require_count,
    _require_positive,
)
from .learning import (
    BOUNDS_QUANTILE,
    LEARNING_FUNCTIONS,
    LEARNING_THRESHOLDS,
    select_learning_point,
    stopping_satisfied,
)
from .model_based import ModelLimitState, generalized_reliability_index
from .population_posterior import (
    GaussianProcessPopulationPosterior,
    PopulationPosterior,
    SurrogatePopulationPosterior,
)
from .simulation import (
    SUBSET_SAMPLERS,
    _acs_level,
    _chain_lengths,
    _cs_level,
    _hmc_level,
    _level_cov,
    _LevelSample,
    _mma_level,
    _require_unit_interval,
)

__all__ = ["AKSS_GP_REGULARIZATION", "AKSSResult", "AKSSRound", "ak_ss"]

AKSS_GP_REGULARIZATION = GaussianProcessRegularization(length_scale_prior_sd=1.0)
"""Default GP regularization of :func:`ak_ss`: the bounds of ADR-0054 plus a
log-normal length-scale prior ``log ell_i ~ N(log E_i, 1)`` (a maximum a
posteriori fit; ``E_i`` is the training extent of input ``i``)."""

_SQRT5 = math.sqrt(5.0)
_SQRT3 = math.sqrt(3.0)


@dataclass(frozen=True, slots=True)
class AKSSRound:
    """One AK-SS round: a subset run on the surrogate and its enrichment.

    Args:
        round: Round index, starting at ``0``.
        training_size: Training points of the surrogate the subset run used.
        probability: Subset-simulation estimate of ``P(mu <= 0)`` on that
            surrogate.
        coefficient_of_variation: Its subset c.o.v. ``sqrt(sum_j delta_j^2)``
            (surrogate misclassification excluded).
        levels: Number of subset levels of the run (including level 0).
        pool_size: Distinct level samples forming the candidate pool.
        added: Points added to the training set in this round.
        initial_learning: Best learning value on the fresh pool before any
            addition (``min U``, ``max EFF``, or ``max H``).
        final_learning: Best learning value when the round ended.
        reached_failure: Whether the subset run reached the threshold ``0``
            within ``max_levels``.
        lower_probability: ``P_f^-`` of the fresh run from the last level's
            samples with ``mu + 1.96 sigma <= 0`` [EQ-REL-AKSS-002].
        upper_probability: ``P_f^+`` with ``mu - 1.96 sigma <= 0``.

    References:
        [REF-HUANG-CHEN-ZHU-2016], [REF-ECHARD-GAYTON-LEMAIRE-2011].

    """

    round: int
    training_size: int
    probability: float
    coefficient_of_variation: float
    levels: int
    pool_size: int
    added: int
    initial_learning: float
    final_learning: float
    reached_failure: bool
    lower_probability: float = math.nan
    upper_probability: float = math.nan


@dataclass(frozen=True, slots=True)
class AKSSResult:
    """AK-SS failure-probability estimate and its learning record.

    Args:
        limit_state: Label of the analyzed limit state.
        probability: Subset-simulation estimate on the final surrogate,
            ``p0^(m-1) N_f / N`` with ``N_f`` counted by ``mu <= 0``.
        thresholds: Intermediate thresholds ``b_1 > ... > b_m = 0`` of the
            final subset run (on the surrogate mean).
        conditional_probabilities: Estimated ``P(F_j | F_{j-1})`` per level.
        level_coefficients_of_variation: Per-level c.o.v. ``delta_j`` with
            the Markov-chain correlation factor.
        coefficient_of_variation: ``sqrt(sum_j delta_j^2)`` of the final
            subset run. It measures only the subset-sampling error on the
            surrogate and excludes surrogate misclassification.
        coefficient_of_variation_upper: ``sum_j delta_j`` (fully correlated
            levels).
        samples_per_level: Samples ``N`` per subset level.
        levels: Levels of the final subset run (including level 0).
        sampler: Markov-chain kernel of the conditional levels.
        evaluations: True limit-state (model) calls of the method, read from
            the model's evaluation counter; excludes the optional checks.
        initial_size: Initial training set: space-filling design plus
            ``extra_points``.
        added: Points added by active learning.
        rounds: Subset runs on the surrogate.
        min_learning: Best learning value on the final pool.
        pool_size: Distinct samples of the final subset run.
        converged: Whether the final run, on a GP fitted to all training
            data, met the relative-bound stop or needed no addition, reached
            the failure threshold, and gave a positive estimate.
        message: Why the run stopped.
        history: One record per round.
        surrogate: Final surrogate fitted on all training data (a
            :class:`~probstat.surrogates.GaussianProcessRegression` unless a
            ``surrogate`` factory was given).
        refits: Surrogate fits (GP hyperparameter fits, or factory calls).
        learning: Learning function used.
        learning_threshold: Stopping value of the learning function.
        training_points: Training inputs in standard-normal space.
        training_values: Limit-state values at ``training_points``.
        check_samples: Size of the optional crude Monte Carlo check
            population (``0`` if none).
        check_probability: Crude Monte Carlo ``P_f`` of the true limit state
            on the check population (``nan`` if none).
        check_misclassified: Check points whose surrogate sign differs from
            the true sign (``-1`` if none).
        check_evaluations: Model calls of the check population.
        level_misclassified: Final-pool points whose surrogate sign differs
            from the true sign (``-1`` unless ``check_levels``).
        level_evaluations: Model calls of the final-pool check.
        lower_probability: ``P_f^-`` of the final run [EQ-REL-AKSS-002].
        upper_probability: ``P_f^+`` of the final run.
        bounds_tolerance: Relative bound width that stops the iteration.

    References:
        [REF-HUANG-CHEN-ZHU-2016], [REF-AU-BECK-2001],
        [REF-ECHARD-GAYTON-LEMAIRE-2011].

    """

    limit_state: str
    probability: float
    thresholds: Vector
    conditional_probabilities: Vector
    level_coefficients_of_variation: Vector
    coefficient_of_variation: float
    coefficient_of_variation_upper: float
    samples_per_level: int
    levels: int
    sampler: str
    evaluations: int
    initial_size: int
    added: int
    rounds: int
    min_learning: float
    pool_size: int
    converged: bool
    message: str
    history: tuple[AKSSRound, ...]
    surrogate: ProbabilisticSurrogate
    refits: int
    learning: str = "U"
    learning_threshold: float = 2.0
    training_points: tuple[Vector, ...] = field(default=())
    training_values: tuple[float, ...] = field(default=())
    check_samples: int = 0
    check_probability: float = math.nan
    check_misclassified: int = -1
    check_evaluations: int = 0
    level_misclassified: int = -1
    level_evaluations: int = 0
    lower_probability: float = math.nan
    upper_probability: float = math.nan
    bounds_tolerance: float = 0.05

    @property
    def beta(self) -> float:
        """Return the generalized reliability index of the estimate.

        Equation:
            [EQ-REL-BETA-001].

        References:
            [REF-DITLEVSEN-MADSEN-1996].

        """
        return generalized_reliability_index(self.probability)


@dataclass(frozen=True, slots=True)
class _SubsetRun:
    """One subset-simulation run on the surrogate mean.

    Args:
        probability: ``p0^(m-1) N_f / N``.
        thresholds: Level thresholds, the last one ``0``.
        conditionals: Conditional level probabilities.
        level_covs: Per-level c.o.v.
        reached: Whether ``N_f >= N p0`` was reached within ``max_levels``.
        pool: Distinct samples of all levels in order of appearance.
        final: The ``N`` samples of the last level (chain repetitions kept).
        product: ``p0^(m-1)``, the probability of the last level's domain.

    References:
        [REF-AU-BECK-2001], [REF-HUANG-CHEN-ZHU-2016]. No equation is
        evaluated.

    """

    probability: float
    thresholds: Vector
    conditionals: Vector
    level_covs: Vector
    reached: bool
    pool: tuple[Vector, ...]
    final: tuple[Vector, ...]
    product: float

    @property
    def cov(self) -> float:
        """Return ``sqrt(sum_j delta_j^2)``.

        References:
            [EQ-REL-SUBSET-003], [REF-AU-BECK-2001].

        """
        return math.sqrt(math.fsum(delta * delta for delta in self.level_covs))


@dataclass(frozen=True, slots=True)
class _KernelSettings:
    """Validated subset-simulation kernel options.

    References:
        [REF-AU-BECK-2001], [REF-PAPAIOANNOU-BETZ-ZWIRGLMAIER-STRAUB-2015],
        [REF-WANG-BROCCARDO-SONG-2019]. No equation is evaluated.

    """

    sampler: str
    total: int
    seeds: int
    p0: float
    chain_length: int
    lengths: tuple[int, ...]
    levels: int
    spread: float
    correlation: float
    scale: float
    target: float
    window: int
    time: float


def _posterior_mean(
    posterior: GaussianProcessPopulationPosterior,
) -> Callable[[Vector], float]:
    """Return ``u -> mu(u)`` of the posterior's GP at ``O(n)`` per call.

    With ``K + s_n^2 I = L L'`` and ``z = L^-1 (y - m)`` (standardized
    units) the mean is ``m + s k(u)' alpha`` with ``alpha = L'^-1 z``,
    obtained once by back substitution.

    References:
        [EQ-SUR-GP-003], [REF-RASMUSSEN-WILLIAMS-2006, Algorithm 2.1].

    """
    lower = posterior.lower
    count = len(lower)
    alpha = [0.0] * count
    for row in range(count - 1, -1, -1):
        tail = math.fsum(lower[j][row] * alpha[j] for j in range(row + 1, count))
        alpha[row] = (posterior.z[row] - tail) / lower[row][row]
    scales = posterior.scales
    # Training inputs divided by the length scales, so that one distance costs
    # d subtractions; the kernel is inlined because the subset run evaluates
    # the mean m N times per round (same formulas as ``kernel_value``).
    training = [
        tuple(c / ell for c, ell in zip(point, scales, strict=True))
        for point in posterior.training
    ]
    weights = [posterior.signal * a for a in alpha]
    kernel = posterior.kernel
    mean, scale = posterior.mean, posterior.scale
    sqrt, exp = math.sqrt, math.exp

    def evaluate(point: Vector) -> float:
        """Return the posterior mean at one standard-normal point.

        References:
            [EQ-SUR-GP-001], [EQ-SUR-GP-002], [EQ-SUR-GP-003],
            [EQ-SUR-GP-008].

        """
        x = [c / ell for c, ell in zip(point, scales, strict=True)]
        terms: list[float] = []
        for known, weight in zip(training, weights, strict=True):
            squared = math.fsum(
                (a - b) * (a - b) for a, b in zip(x, known, strict=True)
            )
            if kernel == "matern52":
                r = sqrt(squared)
                value = (1.0 + _SQRT5 * r + 5.0 * squared / 3.0) * exp(-_SQRT5 * r)
            elif kernel == "squared_exponential":
                value = exp(-0.5 * squared)
            else:
                r = sqrt(squared)
                value = (1.0 + _SQRT3 * r) * exp(-_SQRT3 * r)
            terms.append(weight * value)
        return mean + scale * math.fsum(terms)

    return evaluate


def _final_level_bounds(
    run: _SubsetRun, posterior: PopulationPosterior, total: int
) -> tuple[float, float]:
    """Return ``(P_f^-, P_f^+)`` from the last level's samples.

    ``P_f^- = p0^(m-1) #{mu + 1.96 sigma <= 0} / N`` and
    ``P_f^+ = p0^(m-1) #{mu - 1.96 sigma <= 0} / N`` over the ``N`` samples
    of the last level (chain repetitions counted), with ``mu`` and ``sigma``
    read from the posterior over the pool.

    References:
        [EQ-REL-AKSS-002], [EQ-REL-AL-BOUNDS-001],
        [REF-SCHOBI-SUDRET-MARELLI-2017].

    """
    index = {point: position for position, point in enumerate(posterior.population)}
    means, variances = posterior.means, posterior.variances
    lower = upper = 0
    for point in run.final:
        position = index[point]
        spread = BOUNDS_QUANTILE * math.sqrt(max(variances[position], 0.0))
        mean = means[position]
        if mean + spread <= 0.0:
            lower += 1
        if mean - spread <= 0.0:
            upper += 1
    return run.product * lower / total, run.product * upper / total


def _surrogate_mean(model: ProbabilisticSurrogate) -> Callable[[Vector], float]:
    """Return ``u -> mu(u)`` of a probabilistic surrogate.

    References:
        [EQ-SUR-GP-003]. No equation is evaluated beyond the surrogate's.

    """

    def evaluate(point: Vector) -> float:
        """Return the surrogate mean at one standard-normal point.

        References:
            [EQ-SUR-GP-003]. No equation is evaluated beyond the surrogate's.

        """
        return float(model.predict(point))

    return evaluate


def _surrogate_subset(
    evaluate: Callable[[Vector], float],
    dimension: int,
    generator: random.Random,
    settings: _KernelSettings,
) -> _SubsetRun:
    """Run subset simulation on a cheap response and keep every level sample.

    The level construction is that of
    :func:`~probstat.reliability.subset_simulation`: the threshold is the
    midpoint between the ``N p0``-th and ``(N p0 + 1)``-th smallest values,
    the ``N p0`` smallest samples seed the chains, and the run stops when at
    least ``N p0`` samples satisfy ``mu <= 0`` or ``max_levels`` is reached.
    Ties of the surrogate values at a threshold are not an error here (a
    flat surrogate can produce them); the tied value is then the threshold.

    References:
        [EQ-REL-SUBSET-001], [EQ-REL-SUBSET-003], [EQ-REL-SUBSET-006],
        [REF-AU-BECK-2001], [REF-HUANG-CHEN-ZHU-2016].

    """
    total, seeds_count = settings.total, settings.seeds
    lengths = list(settings.lengths)
    points: list[Vector] = [
        tuple(generator.gauss(0.0, 1.0) for _ in range(dimension)) for _ in range(total)
    ]
    values = [evaluate(u) for u in points]
    pool: dict[Vector, None] = dict.fromkeys(points)
    thresholds: list[float] = []
    conditionals: list[float] = []
    level_covs: list[float] = []
    carry = {
        "mma": settings.spread,
        "cs": settings.correlation,
        "acs": settings.scale,
        "hmc": settings.time,
    }[settings.sampler]
    product = 1.0
    level = 0
    while True:
        correlated = level > 0
        failures = sum(1 for value in values if value <= 0.0)
        if failures >= seeds_count or level + 1 >= settings.levels:
            fraction = failures / total
            thresholds.append(0.0)
            conditionals.append(fraction)
            level_covs.append(
                _level_cov(
                    [value <= 0.0 for value in values],
                    seeds_count,
                    fraction,
                    correlated,
                    lengths,
                )
            )
            return _SubsetRun(
                product * fraction,
                tuple(thresholds),
                tuple(conditionals),
                tuple(level_covs),
                failures >= seeds_count,
                tuple(pool),
                tuple(points),
                product,
            )
        order = sorted(range(total), key=values.__getitem__)
        threshold = 0.5 * (values[order[seeds_count - 1]] + values[order[seeds_count]])
        thresholds.append(threshold)
        conditionals.append(settings.p0)
        level_covs.append(
            _level_cov(
                [value <= threshold for value in values],
                seeds_count,
                settings.p0,
                correlated,
                lengths,
            )
        )
        product *= settings.p0
        seeds = [(points[index], values[index]) for index in order[:seeds_count]]
        sample: _LevelSample
        if settings.sampler == "mma":
            sample = _mma_level(
                evaluate,
                generator,
                seeds,
                threshold,
                settings.chain_length,
                settings.spread,
            )
        elif settings.sampler == "cs":
            sample = _cs_level(
                evaluate, generator, seeds, threshold, lengths, settings.correlation
            )
        elif settings.sampler == "acs":
            sample = _acs_level(
                evaluate,
                generator,
                seeds,
                threshold,
                lengths,
                carry,
                "unit",
                settings.window,
                settings.target,
            )
        else:
            sample = _hmc_level(
                evaluate, generator, seeds, threshold, lengths, carry, settings.window
            )
        carry = sample.carry
        points, values, lengths = sample.points, sample.values, sample.lengths
        pool.update(dict.fromkeys(points))
        level += 1


def _kernel_settings(
    sampler: object,
    samples_per_level: object,
    level_probability: object,
    max_levels: object,
    proposal_spread: object,
    correlation: object,
    initial_scale: object,
    target_acceptance: object,
    trajectory_time: object,
) -> _KernelSettings:
    """Validate the subset-simulation options of :func:`ak_ss`.

    The rules are those of :func:`~probstat.reliability.subset_simulation`.

    Raises:
        ValueError: If an option is invalid.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    if not isinstance(sampler, str) or sampler not in SUBSET_SAMPLERS:
        raise ValueError(f"sampler must be one of {SUBSET_SAMPLERS}")
    total = _require_count(samples_per_level, "samples_per_level", 10)
    p0 = require_finite_real(level_probability, name="level_probability")
    if not 0.0 < p0 <= 0.5:
        raise ValueError("level_probability must lie in (0, 0.5]")
    seeds = round(total * p0)
    if seeds < 1 or abs(seeds - total * p0) > 1e-9:
        raise ValueError("samples_per_level * level_probability must be an integer")
    if sampler == "mma" and total % seeds != 0:
        raise ValueError("samples_per_level * level_probability must divide N")
    levels = _require_count(max_levels, "max_levels", 1)
    spread = _require_positive(proposal_spread, "proposal_spread")
    rho = _require_unit_interval(correlation, "correlation", closed_low=True)
    scale = _require_positive(initial_scale, "initial_scale")
    target = _require_unit_interval(
        target_acceptance, "target_acceptance", closed_low=False
    )
    time = require_finite_real(trajectory_time, name="trajectory_time")
    if not 0.0 < time <= 0.5 * math.pi:
        raise ValueError("trajectory_time must lie in (0, pi/2]")
    chain_length = total // seeds
    lengths = (
        [chain_length] * seeds if sampler == "mma" else _chain_lengths(total, seeds)
    )
    return _KernelSettings(
        sampler=sampler,
        total=total,
        seeds=seeds,
        p0=p0,
        chain_length=chain_length,
        lengths=tuple(lengths),
        levels=levels,
        spread=spread,
        correlation=rho,
        scale=scale,
        target=target,
        window=max(1, math.ceil(100.0 * seeds / total - 1e-9)),
        time=time,
    )


def ak_ss(
    limit_state: ModelLimitState,
    *,
    samples_per_level: int = 2000,
    level_probability: float = 0.1,
    sampler: str = "acs",
    learning: str = "U",
    learning_threshold: float | None = None,
    bounds_tolerance: float = 0.05,
    initial_size: int | None = None,
    design: str = "lhs-maximin",
    design_bound: float | None = 5.0,
    extra_points: Sequence[Sequence[float]] | None = None,
    kernel: str = "matern52",
    surrogate: SurrogateFactory | None = None,
    max_added: int = 300,
    max_rounds: int = 20,
    max_levels: int = 20,
    refit_growth: float = 1.5,
    restarts: int = 3,
    nugget: float = 1e-10,
    regularization: GaussianProcessRegularization | None = AKSS_GP_REGULARIZATION,
    proposal_spread: float = 1.0,
    correlation: float = 0.8,
    initial_scale: float = 0.6,
    target_acceptance: float = 0.44,
    trajectory_time: float = math.pi / 4.0,
    check_population: int = 0,
    check_levels: bool = False,
    seed: int | RandomStream | None = None,
) -> AKSSResult:
    """Estimate a small ``P_f`` by AK-SS (GP active learning on subset levels).

    Scheme of Huang, Chen, and Zhu [REF-HUANG-CHEN-ZHU-2016]: subset
    simulation runs on the surrogate mean, all level samples are the
    candidates of active learning, the next model call goes to
    ``argmin U(u) = |mu(u)| / sigma(u)`` over the candidates
    [EQ-REL-AKMCS-001], [REF-ECHARD-GAYTON-LEMAIRE-2011], and learning on a
    pool stops at ``min U >= 2``. ``learning="EFF"`` and ``learning="H"``
    choose and stop by Bichon's expected feasibility or the entropy instead
    (:mod:`probstat.reliability.learning`).

    Algorithm [EQ-REL-AKSS-001]: with training set ``T`` (the initial
    design), repeat for rounds ``r = 0, 1, ...``:

    1. run subset simulation on ``mu`` with ``N = samples_per_level``,
       ``p0 = level_probability``, and the kernel ``sampler``; its estimate
       is ``P_r = p0^(m-1) #{mu <= 0} / N`` over the ``N`` samples of the
       last level, and its distinct level samples are the pool ``C_r``;
    2. with ``P^-`` and ``P^+`` counting the last level's samples with
       ``mu + 1.96 sigma <= 0`` and ``mu - 1.96 sigma <= 0``
       [EQ-REL-AKSS-002], stop with ``P_r`` when the run reached the failure
       threshold, ``P_r > 0``, and ``(P^+ - P^-) / P_r <=
       bounds_tolerance``;
    3. otherwise, while the best learning value on ``C_r`` misses its
       stopping value (``min U < 2``), evaluate ``G`` at the best candidate
       that is not a training point, add it to ``T``, and update the
       posterior over ``C_r``; if no point was needed, stop with ``P_r``.

    A stop in step 2 or 3 is accepted only on a GP fitted to all of ``T``;
    if points were added since the last fit, the GP is refitted and the round
    repeated, so the reported estimate and the returned surrogate agree.

    Implementation choices that are not part of the published scheme:
    every subset run uses the same random stream (common random numbers),
    so a surrogate that no longer changes reproduces its run and its pool;
    the pool is the set of distinct states of all levels (chain repetitions
    are dropped); the stop of step 2, a relative-bound rule in the spirit of
    Schobi et al. [REF-SCHOBI-SUDRET-MARELLI-2017] applied to the last
    level, is this library's definition of "the estimate is stable" (a fresh
    pool almost always holds a few candidates with ``U < 2`` near the limit
    state, so requiring ``min U >= 2`` on a new run rarely stops); within a
    round the GP posterior over the pool is updated by exact rank-one steps
    at fixed hyperparameters, re-estimated (warm-started) when the training
    set has grown by ``refit_growth`` and before a stop; the GP uses
    :data:`AKSS_GP_REGULARIZATION`, which adds a log-normal length-scale
    prior to the bounds of :func:`ak_mcs`; the initial design is a maximin
    LHS in ``[-b, b]^d`` (``b = design_bound``).

    If ``max_added`` is reached during a round, a last subset run on the
    surrogate refitted to all points provides the estimate and the result is
    not converged (unless that run meets the stop of step 2); the same holds
    after ``max_rounds`` rounds. A final run that does not reach the failure
    threshold within ``max_levels`` or predicts no failure is reported as
    not converged.

    Error: ``coefficient_of_variation`` is Au and Beck's subset c.o.v. of the
    final run on the surrogate [EQ-REL-SUBSET-003]; it excludes
    misclassification by the surrogate. ``lower_probability`` and
    ``upper_probability`` describe the surrogate's own uncertainty on the
    last level and are as overconfident as its ``sigma``.
    ``check_population`` evaluates ``G`` on that many fresh standard-normal
    points (crude Monte Carlo) and counts sign disagreements with the final
    surrogate; useful only when ``P_f`` times the size gives several
    failures. ``check_levels`` evaluates ``G`` on every sample of the
    final pool, which tests the classification where the estimate is made.
    Both checks cost model calls that ``evaluations`` does not include.

    Initial design and blind spots: the learning function only sees where
    the surrogate is uncertain. A failure region that no training point is
    near and whose surrogate mean stays clearly positive is never sampled by
    the subset run, and no candidate is proposed there. A small design mapped
    through ``Phi^-1`` (``design_bound=None``) stays within ``|u| < 2`` and
    can miss distant regions; the box design and ``extra_points`` (for
    example the FORM design points of
    :func:`~probstat.reliability.find_design_points`) are the remedies.

    Cost: one round costs about ``m N`` surrogate-mean evaluations at
    ``O(n)`` each (``m`` levels, ``n`` training points), ``O(n^2)`` per
    distinct pool point to set up the posterior, ``O(n)`` per pool point per
    added point, and a GP likelihood fit. In pure Python ``N = 2000``,
    ``m = 6``, and ``n = 50`` take seconds per round; the true model must be
    much more expensive than that for the method to pay off.

    Args:
        limit_state: Model limit state (``G <= 0`` is failure).
        samples_per_level: Samples ``N`` per subset level; ``N p0`` must be
            a positive integer (and divide ``N`` for ``sampler="mma"``).
        level_probability: Conditional level probability ``p0`` in
            ``(0, 0.5]``.
        sampler: Subset kernel, one of ``SUBSET_SAMPLERS``.
        learning: ``"U"``, ``"EFF"``, or ``"H"``.
        learning_threshold: Positive stopping value, or ``None`` for the
            learning function's default.
        bounds_tolerance: Positive relative width ``(P^+ - P^-) / P_r`` of
            the last level's bounds that stops the iteration.
        initial_size: Initial design size (default ``max(12, 2 d + 2)``).
        design: One of :data:`AKMCS_DESIGNS`.
        design_bound: Half-width ``b`` of the design box, or ``None`` to map
            the design through ``Phi^-1``.
        extra_points: Standard-normal points added to the initial design.
        kernel: GP kernel name (``"matern52"`` or ``"squared_exponential"``).
        surrogate: Optional refit hook ``(points, values, previous) ->
            surrogate`` replacing the incremental GP; it is called on the
            initial design and after every added point, and the pool is
            re-predicted each time (``kernel``, ``refit_growth``,
            ``restarts``, ``nugget``, and ``regularization`` then do not
            apply).
        max_added: Maximum number of active-learning additions.
        max_rounds: Rounds after which no further point is added (one last
            subset run then gives the estimate).
        max_levels: Maximum subset levels per run.
        refit_growth: Training-set growth factor (``> 1``) between GP fits
            within a round.
        restarts: Optimizer starts of the first GP fit.
        nugget: GP diagonal variance in standardized units.
        regularization: GP hyperparameter regularization, or ``None``.
        proposal_spread: Random-walk spread ``s`` (``"mma"``).
        correlation: Fixed correlation ``rho`` in ``[0, 1)`` (``"cs"``).
        initial_scale: Starting scale ``lambda > 0`` (``"acs"``).
        target_acceptance: Target acceptance in ``(0, 1)`` (``"acs"``).
        trajectory_time: Starting trajectory time in ``(0, pi/2]``
            (``"hmc"``).
        check_population: Size of an optional crude Monte Carlo check
            population (``0`` for none).
        check_levels: Evaluate ``G`` on the final pool and count sign
            errors.
        seed: Integer seed, random stream, or ``None`` for fresh entropy.

    Returns:
        Estimate, level thresholds and conditional probabilities, subset
        c.o.v., true-model call count, per-round history, final surrogate
        and training data, and the optional check diagnostics.

    Raises:
        TypeError: If ``limit_state`` has the wrong type or ``surrogate`` is
            not callable.
        ValueError: If options are invalid or the surrogate cannot be fitted.

    Equation:
        [EQ-REL-AKSS-001], [EQ-REL-AKSS-002], [EQ-REL-AKMCS-001],
        [EQ-REL-SUBSET-001], [EQ-REL-SUBSET-003], [EQ-REL-SUBSET-005],
        [EQ-REL-AL-EFF-001], [EQ-REL-AL-H-001], [EQ-REL-AL-BOUNDS-001],
        [EQ-SUR-GP-003], [EQ-SUR-GP-004], [EQ-SUR-GP-006].

    References:
        [REF-HUANG-CHEN-ZHU-2016], [REF-ECHARD-GAYTON-LEMAIRE-2011],
        [REF-AU-BECK-2001], [REF-PAPAIOANNOU-BETZ-ZWIRGLMAIER-STRAUB-2015],
        [REF-SCHOBI-SUDRET-MARELLI-2017], [REF-BECT-LI-VAZQUEZ-2017],
        [REF-RASMUSSEN-WILLIAMS-2006].

    """
    if not isinstance(limit_state, ModelLimitState):
        raise TypeError("limit_state must be a ModelLimitState")
    state = limit_state
    dimension = state.dimension
    settings = _kernel_settings(
        sampler,
        samples_per_level,
        level_probability,
        max_levels,
        proposal_spread,
        correlation,
        initial_scale,
        target_acceptance,
        trajectory_time,
    )
    if learning not in LEARNING_FUNCTIONS:
        raise ValueError(f"learning must be one of {LEARNING_FUNCTIONS}")
    threshold = (
        LEARNING_THRESHOLDS[learning]
        if learning_threshold is None
        else _require_positive(learning_threshold, "learning_threshold")
    )
    first_size = _require_count(
        max(12, 2 * dimension + 2) if initial_size is None else initial_size,
        "initial_size",
        2,
    )
    if design not in AKMCS_DESIGNS:
        raise ValueError(f"design must be one of {AKMCS_DESIGNS}")
    if design_bound is not None:
        _require_positive(design_bound, "design_bound")
    if kernel not in KERNELS:
        raise ValueError(f"kernel must be one of {KERNELS}")
    if surrogate is not None and not callable(surrogate):
        raise TypeError("surrogate must be a callable surrogate factory")
    bound_width = _require_positive(bounds_tolerance, "bounds_tolerance")
    additions_limit = _require_count(max_added, "max_added", 0)
    rounds_limit = _require_count(max_rounds, "max_rounds", 1)
    refit_factor = _require_positive(refit_growth, "refit_growth")
    if refit_factor <= 1.0:
        raise ValueError("refit_growth must exceed 1")
    starts = _require_count(restarts, "restarts", 1)
    _require_positive(nugget, "nugget")
    check_size = _require_count(check_population, "check_population", 0)
    if not isinstance(check_levels, bool):
        raise TypeError("check_levels must be a bool")
    seeds = _extra_points(extra_points, dimension)
    criterion = (
        "min U >= threshold" if learning == "U" else f"max {learning} <= threshold"
    )
    root = resolve_stream(seed)

    before = state.model.evaluations
    training_u = _design_points(
        design, first_size, dimension, design_bound, root.child("ak-ss", "design")
    )
    training_u.extend(seeds)
    training_g: list[float] = [state.standard(u) for u in training_u]
    refits = 0
    previous_theta: Vector | None = None

    def fit() -> GaussianProcessRegression:
        """Fit the GP to the current training set (warm-started).

        References:
            [EQ-SUR-GP-004], [REF-RASMUSSEN-WILLIAMS-2006, Section 5.4].

        """
        nonlocal refits, previous_theta
        refits += 1
        stream = root.child("ak-ss", "fit", refits)
        try:
            fitted = _fit_gp(
                training_u,
                training_g,
                kernel=kernel,
                nugget=nugget,
                restarts=starts if previous_theta is None else 1,
                seed=stream,
                initial_log_parameters=previous_theta,
                regularization=regularization,
            )
        except ValueError:
            if previous_theta is None:
                raise
            fitted = _fit_gp(
                training_u,
                training_g,
                kernel=kernel,
                nugget=nugget,
                restarts=starts,
                seed=stream,
                regularization=regularization,
            )
        previous_theta = fitted.log_parameters
        return fitted

    generic: SurrogatePopulationPosterior | None = None
    if surrogate is not None:
        generic = SurrogatePopulationPosterior(surrogate, training_u, training_g, [])
    gp: GaussianProcessRegression | None = None
    gp_posterior: GaussianProcessPopulationPosterior | None = None
    fitted_size = -1

    def refit() -> tuple[GaussianProcessRegression, GaussianProcessPopulationPosterior]:
        """Refit the GP on all training data; return it with an empty-pool posterior.

        References:
            [EQ-SUR-GP-003], [EQ-SUR-GP-004].

        """
        nonlocal fitted_size
        fitted = fit()
        fitted_size = len(training_u)
        return fitted, GaussianProcessPopulationPosterior(fitted, [])

    def satisfied(index: int, value: float) -> bool:
        """Return whether learning on the current pool may stop.

        References:
            [EQ-REL-AKMCS-001], [EQ-REL-AL-EFF-001], [EQ-REL-AL-H-001].

        """
        return index < 0 or stopping_satisfied(learning, value, threshold)

    history: list[AKSSRound] = []
    added = 0
    rounds = 0
    converged = False
    message = ""
    last_round = False
    refit_due = False
    pending = ""
    while True:
        posterior: PopulationPosterior
        final_model: ProbabilisticSurrogate
        mean: Callable[[Vector], float]
        if generic is not None:
            mean = _surrogate_mean(generic.surrogate)
            posterior = generic
            final_model = generic.surrogate
            fresh = True
        else:
            if gp is None or gp_posterior is None or refit_due:
                gp, gp_posterior = refit()
                refit_due = False
            mean = _posterior_mean(gp_posterior)
            posterior = gp_posterior
            final_model = gp
            # The fitted GP object equals the posterior only right after a fit.
            fresh = fitted_size == len(training_u)
        size_before = len(training_u)
        run = _surrogate_subset(
            mean,
            dimension,
            root.child("ak-ss", "subset").generator(),
            settings,
        )
        if gp_posterior is not None:
            gp_posterior.replace_population(run.pool, reuse=True)
        else:
            posterior.replace_population(run.pool)
        # Pool points that are already training points are never proposed
        # again (their sigma is ~0, so |mu| / sigma can still be small).
        known = set(training_u)
        selected = {index for index, u in enumerate(run.pool) if u in known}
        best_index, best = select_learning_point(
            learning, posterior.means, posterior.variances, selected
        )
        start = best
        lower, upper = _final_level_bounds(run, posterior, settings.total)
        stable = (
            run.reached
            and run.probability > 0.0
            and (upper - lower) / run.probability <= bound_width
        )
        added_round = 0
        while not stable and not last_round and not satisfied(best_index, best):
            if added >= additions_limit:
                break
            point = run.pool[best_index]
            value_g = state.standard(point)
            training_u.append(point)
            training_g.append(value_g)
            selected.add(best_index)
            added += 1
            added_round += 1
            if generic is None and len(training_u) >= refit_factor * fitted_size:
                gp, gp_posterior = refit()
                gp_posterior.add_population(run.pool)
                posterior = gp_posterior
            else:
                posterior.add_training(point, value_g)
            best_index, best = select_learning_point(
                learning, posterior.means, posterior.variances, selected
            )
        history.append(
            AKSSRound(
                rounds,
                size_before,
                run.probability,
                run.cov,
                len(run.conditionals),
                len(run.pool),
                added_round,
                start,
                best,
                run.reached,
                lower,
                upper,
            )
        )
        rounds += 1
        if added_round == 0:
            if not fresh:
                # Rerun on a GP refitted to every training point, so that the
                # reported estimate and the returned surrogate agree.
                refit_due = True
                continue
            if stable:
                converged = True
                message = "final-level P_f bounds within tolerance on a fresh run"
            elif satisfied(best_index, best):
                converged = run.reached and run.probability > 0.0
                if converged:
                    message = f"{criterion} on a fresh run without additions"
                elif not run.reached:
                    message = (
                        f"{criterion}, but the subset run did not reach the "
                        "failure threshold within max_levels"
                    )
                else:
                    message = f"{criterion}, but no failure is predicted"
            else:
                message = pending or f"max_added reached before {criterion}"
            break
        if not satisfied(best_index, best):
            pending = f"max_added reached before {criterion}"
            last_round = refit_due = True
        elif rounds >= rounds_limit:
            pending = "max_rounds reached before a converged round"
            last_round = refit_due = True
    evaluations = state.model.evaluations - before
    if generic is not None:
        refits = generic.fits
    check_probability = math.nan
    check_misclassified = -1
    check_calls = 0
    if check_size > 0:
        generator = root.child("ak-ss", "check").generator()
        sample = [
            tuple(generator.gauss(0.0, 1.0) for _ in range(dimension))
            for _ in range(check_size)
        ]
        check_before = state.model.evaluations
        truth = [state.standard(u) for u in sample]
        check_calls = state.model.evaluations - check_before
        check_probability = sum(1 for g in truth if g <= 0.0) / check_size
        check_misclassified = sum(
            1
            for g, u in zip(truth, sample, strict=True)
            if (g <= 0.0) != (mean(u) <= 0.0)
        )
    level_misclassified = -1
    level_calls = 0
    if check_levels:
        level_before = state.model.evaluations
        truth = [state.standard(u) for u in run.pool]
        level_calls = state.model.evaluations - level_before
        level_misclassified = sum(
            1
            for g, mu in zip(truth, posterior.means, strict=True)
            if (g <= 0.0) != (mu <= 0.0)
        )
    return AKSSResult(
        limit_state=state.label,
        probability=run.probability,
        thresholds=run.thresholds,
        conditional_probabilities=run.conditionals,
        level_coefficients_of_variation=run.level_covs,
        coefficient_of_variation=run.cov,
        coefficient_of_variation_upper=math.fsum(run.level_covs),
        samples_per_level=settings.total,
        levels=len(run.conditionals),
        sampler=settings.sampler,
        evaluations=evaluations,
        initial_size=first_size + len(seeds),
        added=added,
        rounds=rounds,
        min_learning=best,
        pool_size=len(run.pool),
        converged=converged,
        message=message,
        history=tuple(history),
        surrogate=final_model,
        refits=refits,
        learning=learning,
        learning_threshold=threshold,
        training_points=tuple(training_u),
        training_values=tuple(training_g),
        check_samples=check_size,
        check_probability=check_probability,
        check_misclassified=check_misclassified,
        check_evaluations=check_calls,
        level_misclassified=level_misclassified,
        level_evaluations=level_calls,
        lower_probability=lower,
        upper_probability=upper,
        bounds_tolerance=bound_width,
    )
