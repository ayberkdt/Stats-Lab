"""Multilevel Monte Carlo estimation of a finest-level expectation.

A user hierarchy of approximations ``P_0, P_1, ..., P_L`` (for example
Euler-Maruyama paths with ``M^l`` time steps) is combined through the
telescoping identity::

    E[P_L] = E[P_0] + sum_{l=1}^{L} E[P_l - P_{l-1}]

Each correction ``Y_l = P_l - P_{l-1}`` is estimated from *coupled* fine and
coarse samples that share the same underlying randomness, so ``Var(Y_l)``
shrinks with ``l`` and most samples are taken on cheap coarse levels. The
adaptive driver follows Giles' algorithm: allocate ``N_l`` proportional to
``sqrt(V_l / C_l)`` for a sampling variance ``eps^2 / 2`` and add levels until
the estimated remaining bias is below ``eps / sqrt 2``, so that the root mean
square error is at most ``eps``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real

LevelSampler = Callable[
    [int, int, RandomStream], tuple[Sequence[float], Sequence[float]]
]
PathCost = Callable[[int], float]


@dataclass(frozen=True, slots=True)
class LevelStatistics:
    """Sample statistics and cost of one multilevel correction.

    Args:
        level: Level index ``l`` (0 is the coarsest).
        samples: Number of coupled samples ``N_l``.
        mean: Sample mean of ``Y_l = P_l - P_{l-1}`` (``Y_0 = P_0``).
        variance: Unbiased sample variance ``V_l`` of ``Y_l``.
        fine_mean: Sample mean of the fine approximation ``P_l``.
        fine_variance: Unbiased sample variance of ``P_l``.
        cost_per_sample: Cost ``C_l`` of one coupled sample (fine plus coarse
            path).

    References:
        [REF-GILES-2008], [REF-GILES-2015].

    """

    level: int
    samples: int
    mean: float
    variance: float
    fine_mean: float
    fine_variance: float
    cost_per_sample: float

    @property
    def total_cost(self) -> float:
        """Return ``N_l C_l``, the cost spent on this level.

        Equation:
            [EQ-MC-MLMC-002].

        References:
            [REF-GILES-2015].

        """
        return self.samples * self.cost_per_sample


@dataclass(frozen=True, slots=True)
class MultilevelResult:
    """Multilevel Monte Carlo estimate with cost and convergence diagnostics.

    Args:
        estimate: Telescoping sum of level means.
        variance: Estimator sampling variance ``sum_l V_l / N_l``.
        bias_estimate: Extrapolated magnitude of ``E[P - P_L]`` from the last
            level means and ``alpha`` (``nan`` with fewer than two levels).
        target_rmse: Requested root mean square error, ``None`` for a fixed
            allocation.
        levels: Per-level statistics, coarsest first.
        alpha: Weak rate: ``|E[Y_l]|`` decays like ``M^(-alpha l)``.
        beta: Variance rate: ``V_l`` decays like ``M^(-beta l)``.
        gamma: Cost rate: ``C_l`` grows like ``M^(gamma l)``.
        refinement: Refinement factor ``M`` between levels.
        total_cost: ``sum_l N_l C_l``.
        single_level_cost: Cost of plain Monte Carlo on the finest level with
            the same sampling variance, ``Var(P_L) C_L^fine / variance``.
        converged: Whether the adaptive bias test passed (always ``True`` for
            a fixed allocation).

    References:
        [REF-GILES-2008], [REF-GILES-2015].

    """

    estimate: float
    variance: float
    bias_estimate: float
    target_rmse: float | None
    levels: tuple[LevelStatistics, ...]
    alpha: float
    beta: float
    gamma: float
    refinement: int
    total_cost: float
    single_level_cost: float
    converged: bool

    @property
    def standard_error(self) -> float:
        """Return the sampling standard error ``sqrt(sum_l V_l / N_l)``.

        Equation:
            [EQ-MC-MLMC-001].

        References:
            [REF-GILES-2015].

        """
        return math.sqrt(self.variance)

    @property
    def rmse_estimate(self) -> float:
        """Return ``sqrt(variance + bias^2)`` with a ``nan`` bias read as zero.

        Equation:
            [EQ-MC-MLMC-003].

        References:
            [REF-GILES-2015].

        """
        bias = 0.0 if math.isnan(self.bias_estimate) else self.bias_estimate
        return math.sqrt(self.variance + bias * bias)

    @property
    def finest_level(self) -> int:
        """Return the finest level ``L``.

        References:
            [REF-GILES-2008].

        """
        return self.levels[-1].level

    @property
    def samples(self) -> tuple[int, ...]:
        """Return ``(N_0, ..., N_L)``.

        References:
            [REF-GILES-2008].

        """
        return tuple(item.samples for item in self.levels)

    @property
    def evaluations(self) -> int:
        """Return the number of coupled samples drawn, ``sum_l N_l``.

        References:
            [REF-GILES-2008]. No equation is evaluated.

        """
        return sum(item.samples for item in self.levels)

    @property
    def cost_ratio(self) -> float:
        """Return ``single_level_cost / total_cost`` (above one favors MLMC).

        Equation:
            [EQ-MC-MLMC-002].

        References:
            [REF-GILES-2015].

        """
        return self.single_level_cost / self.total_cost


def _default_path_cost(refinement: int) -> PathCost:
    """Return ``l -> M^l``, the step count of a uniformly refined path.

    References:
        [REF-GILES-2008]. No named equation is evaluated.

    """

    def cost(level: int) -> float:
        """Return ``M^level``.

        References:
            [REF-GILES-2008].

        """
        return float(refinement**level)

    return cost


def _require_refinement(value: object) -> int:
    """Validate an integer refinement factor of at least two.

    References:
        Project validation policy. No equation is evaluated.

    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 2:
        raise ValueError("refinement must be an integer of at least two")
    return value


def _level_cost(path_cost: PathCost, level: int) -> float:
    """Return ``C_l = c(l) + c(l - 1)`` (``C_0 = c(0)``) after validation.

    References:
        [EQ-MC-MLMC-002], [REF-GILES-2015].

    """
    fine = require_finite_real(path_cost(level), name="path_cost")
    if fine <= 0.0:
        raise ValueError("path_cost must return positive values")
    if level == 0:
        return fine
    coarse = require_finite_real(path_cost(level - 1), name="path_cost")
    if coarse <= 0.0:
        raise ValueError("path_cost must return positive values")
    return fine + coarse


def optimal_level_allocation(
    variances: Sequence[float],
    costs: Sequence[float],
    target_variance: float,
) -> tuple[int, ...]:
    """Return Giles' cost-optimal sample sizes for a target sampling variance.

    Minimizing ``sum_l N_l C_l`` subject to ``sum_l V_l / N_l = v`` gives
    ``N_l = sqrt(V_l / C_l) sum_k sqrt(V_k C_k) / v``; the result is rounded
    up so the constraint holds.

    Args:
        variances: Non-negative level variances ``V_l``.
        costs: Positive level costs ``C_l``.
        target_variance: Positive target estimator variance ``v``.

    Returns:
        Integer sample sizes ``N_l`` (zero only where ``V_l = 0``).

    Raises:
        ValueError: If lengths differ, a value is out of domain, or the input
            is empty.

    Equation:
        [EQ-MC-MLMC-002].

    References:
        [REF-GILES-2008], [REF-GILES-2015].

    """
    if len(variances) != len(costs) or not variances:
        raise ValueError("variances and costs must be non-empty and equal length")
    target = require_finite_real(target_variance, name="target_variance")
    if target <= 0.0:
        raise ValueError("target_variance must be positive")
    checked_v = [require_finite_real(v, name="variance") for v in variances]
    checked_c = [require_finite_real(c, name="cost") for c in costs]
    if any(v < 0.0 for v in checked_v):
        raise ValueError("variances must be non-negative")
    if any(c <= 0.0 for c in checked_c):
        raise ValueError("costs must be positive")
    total = math.fsum(
        math.sqrt(v * c) for v, c in zip(checked_v, checked_c, strict=True)
    )
    return tuple(
        math.ceil(math.sqrt(v / c) * total / target) if v > 0.0 else 0
        for v, c in zip(checked_v, checked_c, strict=True)
    )


def _slope(levels: Sequence[int], values: Sequence[float], base: int) -> float:
    """Return the least-squares slope of ``log_base(values)`` against level.

    Non-positive values are skipped; ``nan`` is returned with fewer than two
    usable points.

    References:
        [EQ-MC-MLMC-003], [REF-GILES-2015].

    """
    points = [
        (float(level), math.log(value) / math.log(base))
        for level, value in zip(levels, values, strict=True)
        if value > 0.0 and math.isfinite(value)
    ]
    if len(points) < 2:
        return math.nan
    mean_x = math.fsum(x for x, _ in points) / len(points)
    mean_y = math.fsum(y for _, y in points) / len(points)
    sxx = math.fsum((x - mean_x) ** 2 for x, _ in points)
    sxy = math.fsum((x - mean_x) * (y - mean_y) for x, y in points)
    return sxy / sxx


def convergence_rates(
    means: Sequence[float],
    variances: Sequence[float],
    costs: Sequence[float],
    *,
    refinement: int = 2,
) -> tuple[float, float, float]:
    """Estimate MLMC rates ``(alpha, beta, gamma)`` by log-linear regression.

    For levels ``l >= 1`` fit ``log_M |E[Y_l]| ~ -alpha l``,
    ``log_M V_l ~ -beta l``, and ``log_M C_l ~ gamma l`` by ordinary least
    squares. Level 0 is excluded because ``Y_0 = P_0`` is not a difference.

    Args:
        means: Level means ``E[Y_l]`` for ``l = 0..L``.
        variances: Level variances ``V_l``.
        costs: Level costs ``C_l``.
        refinement: Refinement factor ``M``.

    Returns:
        ``(alpha, beta, gamma)``; an entry is ``nan`` when fewer than two
        usable levels exist.

    Raises:
        ValueError: If the sequences have different lengths.

    Equation:
        [EQ-MC-MLMC-003].

    References:
        [REF-GILES-2008], [REF-GILES-2015].

    """
    base = _require_refinement(refinement)
    if not len(means) == len(variances) == len(costs):
        raise ValueError("means, variances, and costs must have equal length")
    levels = list(range(1, len(means)))
    alpha = -_slope(levels, [abs(m) for m in means[1:]], base)
    beta = -_slope(levels, list(variances[1:]), base)
    gamma = _slope(levels, list(costs[1:]), base)
    return alpha, beta, gamma


class _LevelAccumulator:
    """Running sums of coupled fine/coarse samples on one level.

    References:
        [REF-GILES-2015].

    """

    __slots__ = ("level", "differences", "fines", "batches")

    def __init__(self, level: int) -> None:
        """Create an empty accumulator for ``level``.

        References:
            [REF-GILES-2015]. No equation is evaluated.

        """
        self.level = level
        self.differences: list[float] = []
        self.fines: list[float] = []
        self.batches = 0

    def draw(self, sampler: LevelSampler, count: int, root: RandomStream) -> None:
        """Draw ``count`` coupled samples from a fresh child stream.

        Args:
            sampler: User level sampler.
            count: Positive number of samples.
            root: Root stream; batch ``k`` of level ``l`` uses
                ``root.child("mlmc-level", l, k)``.

        Raises:
            ValueError: If the sampler returns the wrong number of values or
                non-finite values.

        References:
            [EQ-MC-MLMC-001], [REF-GILES-2008].

        """
        stream = root.child("mlmc-level", self.level, self.batches)
        self.batches += 1
        fine, coarse = sampler(self.level, count, stream)
        fine_values = [require_finite_real(v, name="fine sample") for v in fine]
        coarse_values = [require_finite_real(v, name="coarse sample") for v in coarse]
        if len(fine_values) != count or len(coarse_values) != count:
            raise ValueError(
                f"level_sampler returned the wrong number of samples at level "
                f"{self.level}: expected {count}"
            )
        if self.level == 0:
            self.differences.extend(fine_values)
        else:
            self.differences.extend(
                f - c for f, c in zip(fine_values, coarse_values, strict=True)
            )
        self.fines.extend(fine_values)

    @property
    def count(self) -> int:
        """Return the number of samples drawn so far.

        References:
            [REF-GILES-2015]. No equation is evaluated.

        """
        return len(self.differences)

    def statistics(self, cost: float) -> LevelStatistics:
        """Return the level's sample statistics.

        References:
            [EQ-MC-MEAN-001], [EQ-MC-VAR-001], [REF-GILES-2015].

        """
        mean, variance = _mean_variance(self.differences)
        fine_mean, fine_variance = _mean_variance(self.fines)
        return LevelStatistics(
            self.level, self.count, mean, variance, fine_mean, fine_variance, cost
        )


def _mean_variance(values: Sequence[float]) -> tuple[float, float]:
    """Return the sample mean and unbiased variance (zero for one value).

    References:
        [EQ-MC-MEAN-001], [EQ-MC-VAR-001], [REF-ROBERT-CASELLA-2004, Chapter 3].

    """
    count = len(values)
    if count == 0:
        return 0.0, 0.0
    mean = math.fsum(values) / count
    if count < 2:
        return mean, 0.0
    return mean, math.fsum((v - mean) ** 2 for v in values) / (count - 1)


def _assemble(
    accumulators: Sequence[_LevelAccumulator],
    costs: Sequence[float],
    path_cost: PathCost,
    refinement: int,
    target_rmse: float | None,
    alpha_fixed: float | None,
    converged: bool,
) -> MultilevelResult:
    """Combine per-level sums into a :class:`MultilevelResult`.

    References:
        [EQ-MC-MLMC-001], [EQ-MC-MLMC-003], [REF-GILES-2015].

    """
    levels = tuple(
        acc.statistics(cost) for acc, cost in zip(accumulators, costs, strict=True)
    )
    estimate = math.fsum(item.mean for item in levels)
    variance = math.fsum(item.variance / item.samples for item in levels)
    alpha, beta, gamma = convergence_rates(
        [item.mean for item in levels],
        [item.variance for item in levels],
        [item.cost_per_sample for item in levels],
        refinement=refinement,
    )
    bias = _bias_estimate(
        [abs(item.mean) for item in levels],
        alpha_fixed if alpha_fixed is not None else alpha,
        refinement,
    )
    finest = levels[-1]
    fine_cost = require_finite_real(path_cost(finest.level), name="path_cost")
    single = finest.fine_variance * fine_cost / variance if variance > 0.0 else math.nan
    return MultilevelResult(
        estimate=estimate,
        variance=variance,
        bias_estimate=bias,
        target_rmse=target_rmse,
        levels=levels,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        refinement=refinement,
        total_cost=math.fsum(item.total_cost for item in levels),
        single_level_cost=single,
        converged=converged,
    )


def _bias_estimate(abs_means: Sequence[float], alpha: float, refinement: int) -> float:
    """Return Giles' extrapolated bias ``max_j m_{L-j} M^(-j alpha) / (M^alpha - 1)``.

    Uses up to three of the finest corrections, ``j = 0, 1, 2``; ``nan`` when
    only level 0 exists or ``alpha`` is not positive and finite.

    References:
        [EQ-MC-MLMC-003], [REF-GILES-2015].

    """
    finest = len(abs_means) - 1
    if finest < 1 or not math.isfinite(alpha) or alpha <= 0.0:
        return math.nan
    factor: float = float(refinement) ** alpha
    candidates = [abs_means[finest - j] / factor**j for j in range(min(3, finest))]
    return max(candidates) / (factor - 1.0)


def multilevel_monte_carlo(
    level_sampler: LevelSampler,
    samples: Sequence[int],
    *,
    path_cost: PathCost | None = None,
    refinement: int = 2,
    seed: int | RandomStream | None = None,
) -> MultilevelResult:
    """Estimate ``E[P_L]`` with a fixed multilevel sample allocation.

    ``level_sampler(l, n, stream)`` must return two length-``n`` sequences
    ``(fine, coarse)`` of *coupled* samples ``P_l`` and ``P_{l-1}`` computed
    from the same randomness drawn from ``stream``; at level 0 ``coarse`` is
    ignored (``P_{-1} = 0``) and may be any length-``n`` sequence.

    Args:
        level_sampler: Coupled fine/coarse sampler.
        samples: Sample sizes ``(N_0, ..., N_L)``, each at least two.
        path_cost: Cost ``c(l)`` of one fine path at level ``l``; default
            ``M^l``. A coupled sample costs ``C_l = c(l) + c(l - 1)``.
        refinement: Refinement factor ``M`` used for the rate regression.
        seed: Root seed or stream.

    Returns:
        Estimate, sampling variance, per-level statistics, rates, and costs.

    Raises:
        ValueError: If a sample size is below two, the sampler output is
            malformed, or a cost is not positive.

    Equation:
        [EQ-MC-MLMC-001], [EQ-MC-MLMC-003].

    References:
        [REF-GILES-2008], [REF-GILES-2015].

    """
    factor = _require_refinement(refinement)
    sizes = tuple(samples)
    if not sizes:
        raise ValueError("samples must name at least one level")
    for size in sizes:
        if isinstance(size, bool) or not isinstance(size, int) or size < 2:
            raise ValueError("every level needs an integer sample size of at least 2")
    cost_model = path_cost if path_cost is not None else _default_path_cost(factor)
    root = resolve_stream(seed)
    accumulators = [_LevelAccumulator(level) for level in range(len(sizes))]
    costs = [_level_cost(cost_model, level) for level in range(len(sizes))]
    for accumulator, size in zip(accumulators, sizes, strict=True):
        accumulator.draw(level_sampler, size, root)
    return _assemble(accumulators, costs, cost_model, factor, None, None, True)


def adaptive_multilevel_monte_carlo(
    level_sampler: LevelSampler,
    target_rmse: float,
    *,
    path_cost: PathCost | None = None,
    refinement: int = 2,
    initial_levels: int = 2,
    max_level: int = 10,
    pilot_samples: int = 100,
    alpha: float | None = None,
    beta: float | None = None,
    seed: int | RandomStream | None = None,
) -> MultilevelResult:
    """Estimate ``E[P]`` to a root mean square error ``eps`` by adaptive MLMC.

    Giles' algorithm: start with levels ``0..L0`` and ``pilot_samples`` each;
    repeatedly (1) estimate ``m_l = |mean Y_l|`` and ``V_l``, floored for
    ``l >= 2`` at half the value extrapolated from level ``l - 1``,
    (2) estimate ``alpha`` and ``beta`` by regression unless given (with a
    lower bound of ``1/2``), (3) set ``N_l`` by
    :func:`optimal_level_allocation` with sampling variance ``eps^2 / 2`` and
    draw the missing samples. When no level needs more than 1% extra samples,
    the extrapolated bias is compared with ``eps / sqrt 2``; if larger, level
    ``L + 1`` is added with ``V_{L+1} = V_L / M^beta`` and the loop repeats.

    Args:
        level_sampler: Coupled fine/coarse sampler (see
            :func:`multilevel_monte_carlo`).
        target_rmse: Positive accuracy ``eps``.
        path_cost: Cost ``c(l)`` of one fine path at level ``l``; default
            ``M^l``.
        refinement: Refinement factor ``M``.
        initial_levels: Initial finest level ``L0`` (at least 1).
        max_level: Largest admissible finest level.
        pilot_samples: Initial samples per level (at least 2).
        alpha: Optional known weak rate (skips its regression).
        beta: Optional known variance rate (skips its regression).
        seed: Root seed or stream.

    Returns:
        Result whose ``converged`` flag is ``False`` when ``max_level`` was
        reached before the bias test passed.

    Raises:
        ValueError: If options are out of domain or the sampler output is
            malformed.

    Equation:
        [EQ-MC-MLMC-001], [EQ-MC-MLMC-002], [EQ-MC-MLMC-003].

    References:
        [REF-GILES-2008], [REF-GILES-2015].

    """
    factor = _require_refinement(refinement)
    eps = require_finite_real(target_rmse, name="target_rmse")
    if eps <= 0.0:
        raise ValueError("target_rmse must be positive")
    for value, name, minimum in (
        (initial_levels, "initial_levels", 1),
        (pilot_samples, "pilot_samples", 2),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer of at least {minimum}")
    if isinstance(max_level, bool) or not isinstance(max_level, int):
        raise ValueError("max_level must be an integer")
    if max_level < initial_levels:
        raise ValueError("max_level must not be below initial_levels")
    for rate, name in ((alpha, "alpha"), (beta, "beta")):
        if rate is not None and require_finite_real(rate, name=name) <= 0.0:
            raise ValueError(f"{name} must be positive when given")
    cost_model = path_cost if path_cost is not None else _default_path_cost(factor)
    root = resolve_stream(seed)
    finest = initial_levels
    accumulators = [_LevelAccumulator(level) for level in range(finest + 1)]
    costs = [_level_cost(cost_model, level) for level in range(finest + 1)]
    pending = [pilot_samples] * (finest + 1)
    current_alpha = alpha if alpha is not None else 0.0
    current_beta = beta if beta is not None else 0.0
    converged = False
    while True:
        for accumulator, extra in zip(accumulators, pending, strict=True):
            if extra > 0:
                accumulator.draw(level_sampler, extra, root)
        means = []
        variances = []
        for accumulator in accumulators:
            mean, variance = _mean_variance(accumulator.differences)
            means.append(abs(mean))
            variances.append(variance)
        for level in range(2, finest + 1):
            means[level] = max(
                means[level], 0.5 * means[level - 1] / float(factor) ** current_alpha
            )
            variances[level] = max(
                variances[level],
                0.5 * variances[level - 1] / float(factor) ** current_beta,
            )
        estimated_alpha, estimated_beta, _ = convergence_rates(
            means, variances, costs, refinement=factor
        )
        if alpha is None:
            current_alpha = max(
                0.5, estimated_alpha if math.isfinite(estimated_alpha) else 0.5
            )
        if beta is None:
            current_beta = max(
                0.5, estimated_beta if math.isfinite(estimated_beta) else 0.5
            )
        target_variance = 0.5 * eps * eps
        wanted = optimal_level_allocation(variances, costs, target_variance)
        counts = [accumulator.count for accumulator in accumulators]
        pending = [max(0, w - n) for w, n in zip(wanted, counts, strict=True)]
        if all(p <= 0.01 * n for p, n in zip(pending, counts, strict=True)):
            remaining = _bias_estimate(means, current_alpha, factor)
            converged = remaining <= eps / math.sqrt(2.0)
            if converged or finest >= max_level:
                if sum(pending) == 0:
                    break
            else:
                finest += 1
                accumulators.append(_LevelAccumulator(finest))
                costs.append(_level_cost(cost_model, finest))
                variances.append(variances[-1] / float(factor) ** current_beta)
                wanted = optimal_level_allocation(variances, costs, target_variance)
                counts.append(0)
                pending = [max(0, w - n) for w, n in zip(wanted, counts, strict=True)]
                pending[-1] = max(pending[-1], 2)
                converged = False
        if sum(pending) == 0:
            break
    return _assemble(
        accumulators, costs, cost_model, factor, eps, current_alpha, converged
    )
