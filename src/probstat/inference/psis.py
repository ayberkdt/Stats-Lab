"""Pareto-smoothed importance sampling and PSIS leave-one-out cross-validation.

Importance sampling estimates ``E_p[h]`` from draws ``x_s ~ g`` with the
self-normalized average ``sum_s w_s h(x_s) / sum_s w_s``, ``w_s`` being
proportional to the ratio ``r_s = p(x_s) / g(x_s)``. When the ratios have a
heavy right tail the few largest ones dominate the sum and the estimate has
a large or infinite variance. Pareto-smoothed importance sampling (PSIS;
Vehtari, Simpson, Gelman, Yao and Gabry 2024) replaces the ``M`` largest
ratios by the expected order statistics of a generalized Pareto distribution
fitted to them, and reports the fitted shape ``k-hat`` as a diagnostic of how
many moments the ratios have (``1 / k`` of them in the limit).

Steps of :func:`pareto_smooth` ([EQ-INF-PSIS-001])
    1. Tail size ``M = ceil(min(S / 5, 3 sqrt(S)))`` (:func:`psis_tail_size`).
    2. Fit a generalized Pareto distribution to the exceedances of the ``M``
       largest ratios over the largest ratio that is *not* in the tail
       (:func:`fit_generalized_pareto`, [EQ-INF-PSIS-002]).
    3. Replace the ``z``-th smallest tail ratio by the fitted quantile
       ``u + F^-1((z - 1/2) / M)`` and truncate at the largest raw ratio.
    4. Normalize; report ``k-hat``, the threshold
       ``min(1 - 1 / log10(S), 0.7)``, and the effective sample size
       ``1 / sum w~^2`` ([EQ-INF-PSIS-003]).

The generalized Pareto fit
    The estimator is the one PSIS uses: the empirical-Bayes profile estimator
    of Zhang and Stephens as described by Vehtari et al. (2024), a
    posterior mean of ``theta = -k / sigma`` over a data-dependent grid under
    the profile likelihood, followed by a weak shrinkage of ``k`` towards
    ``0.5`` worth ten observations. The library's peaks-over-threshold fit
    lives in ``probstat.risk``, a higher layer that ``inference`` may not
    import, so the estimator is implemented here.

PSIS-LOO (:func:`psis_loo`, [EQ-INF-PSIS-004])
    With posterior draws ``theta_s`` and the pointwise log-likelihood
    ``log p(y_i | theta_s)``, the leave-one-out posterior of observation
    ``i`` has ratios ``1 / p(y_i | theta_s)``; the smoothed weights give
    ``elpd_loo,i = log(sum_s w_s p(y_i | theta_s) / sum_s w_s)`` without
    refitting the model (Vehtari, Gelman and Gabry 2017).

What the diagnostic does and does not say
    ``k-hat`` below the threshold means the *observed* tail is light enough
    for the smoothed estimate to be reliable at this sample size. It is an
    estimate from ``M`` tail draws and cannot detect a region of large ratios
    that the proposal never visited. The Monte Carlo standard error reported
    here treats the draws as independent; for autocorrelated MCMC draws it is
    too small.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)

_MINIMUM_TAIL = 5
_PRIOR_GRID_CONSTANT = 3.0
_MINIMUM_GRID_POINTS = 30
_SHRINKAGE_COUNT = 10.0
_SHRINKAGE_MEAN = 0.5
_MAXIMUM_THRESHOLD = 0.7


def _log_sum_exp(values: Sequence[float]) -> float:
    """Return ``log sum_i exp(v_i)`` without overflow.

    References:
        [EQ-INF-PSIS-003]. No new equation is introduced.

    """
    top = max(values)
    if not math.isfinite(top):
        return top
    return top + math.log(math.fsum(math.exp(value - top) for value in values))


@dataclass(frozen=True, slots=True)
class GeneralizedParetoFit:
    """Shape and scale of a generalized Pareto law fitted to exceedances.

    The distribution function is ``F(x) = 1 - (1 + k x / sigma)^(-1/k)`` for
    ``x >= 0`` (``1 - exp(-x / sigma)`` when ``k = 0``); its moments of order
    below ``1 / k`` are finite.

    Args:
        shape: Shape ``k`` (after the weak prior when ``regularize`` was
            requested); ``inf`` when the sample was degenerate.
        scale: Scale ``sigma`` (``nan`` when the sample was degenerate).
        count: Number of exceedances used.
        regularized: Whether ``k`` was shrunk towards ``0.5``.

    References:
        [EQ-INF-PSIS-002], [REF-VEHTARI-ETAL-2024].

    """

    shape: float
    scale: float
    count: int
    regularized: bool

    def quantile(self, probability: float) -> float:
        """Return ``F^-1(p) = sigma ((1 - p)^(-k) - 1) / k``.

        Args:
            probability: Level ``p`` in ``[0, 1)``.

        Returns:
            The quantile of the fitted law (``-sigma log(1 - p)`` for
            ``k = 0``).

        Raises:
            ValueError: If ``p`` is outside ``[0, 1)`` or the fit is
                degenerate.

        References:
            [EQ-INF-PSIS-001], [REF-VEHTARI-ETAL-2024].

        """
        p = require_finite_real(probability, name="probability")
        if not 0.0 <= p < 1.0:
            raise ValueError("probability must lie in [0, 1)")
        if not (math.isfinite(self.shape) and math.isfinite(self.scale)):
            raise ValueError("the fit is degenerate and has no quantile function")
        log_survival = math.log1p(-p)
        if self.shape == 0.0:
            return -self.scale * log_survival
        return self.scale * math.expm1(-self.shape * log_survival) / self.shape


def fit_generalized_pareto(
    exceedances: Sequence[float], *, regularize: bool = True
) -> GeneralizedParetoFit:
    """Fit a generalized Pareto law to non-negative exceedances (PSIS estimator).

    With the sorted sample ``x_(1) <= ... <= x_(N)`` and
    ``theta = -k / sigma``, the profile log-likelihood is
    ``l(theta) = N (log(-theta / k(theta)) - k(theta) - 1)`` with
    ``k(theta) = (1/N) sum_i log(1 - theta x_i)``. The estimator averages
    ``theta`` over the grid
    ``theta_j = 1 / x_(N) + (1 - sqrt(G / (j - 1/2))) / (3 x*)``,
    ``j = 1, ..., G``, ``G = 30 + floor(sqrt(N))``, ``x*`` the order
    statistic of rank ``floor(N / 4 + 1/2)``, with weights proportional to
    ``exp(l(theta_j))``, then sets ``k = k(theta_hat)`` and
    ``sigma = -k / theta_hat``. With ``regularize`` the shape is replaced by
    ``(N k + 5) / (N + 10)``, a weak prior centred at ``0.5``.

    Args:
        exceedances: At least five non-negative finite values (excesses over
            a threshold).
        regularize: Apply the weak prior to ``k`` (the PSIS default). The
            scale is not changed by it.

    Returns:
        The fitted shape and scale. A sample whose lower quartile equals its
        minimum, or whose maximum is zero, has no usable profile likelihood;
        the fit is then returned with ``shape = inf`` and ``scale = nan``.

    Raises:
        ValueError: For fewer than five values, or a negative or non-finite
            value.

    Equation:
        [EQ-INF-PSIS-002].

    References:
        [REF-VEHTARI-ETAL-2024].

    """
    if not isinstance(regularize, bool):
        raise TypeError("regularize must be a bool")
    values = sorted(require_finite_real(v, name="exceedance") for v in exceedances)
    count = len(values)
    if count < _MINIMUM_TAIL:
        raise ValueError("at least 5 exceedances are required")
    if values[0] < 0.0:
        raise ValueError("exceedances must be non-negative")
    quartile = values[max(int(math.floor(count / 4.0 + 0.5)), 1) - 1]
    largest = values[-1]
    if not (largest > 0.0 and quartile > values[0]):
        return GeneralizedParetoFit(math.inf, math.nan, count, regularize)
    grid = _MINIMUM_GRID_POINTS + int(math.isqrt(count))
    thetas = [
        1.0 / largest
        + (1.0 - math.sqrt(grid / (j - 0.5))) / (_PRIOR_GRID_CONSTANT * quartile)
        for j in range(1, grid + 1)
    ]

    def shape_at(theta: float) -> float:
        """Return ``k(theta) = mean_i log(1 - theta x_i)``.

        References:
            [EQ-INF-PSIS-002], [REF-VEHTARI-ETAL-2024].

        """
        return math.fsum(math.log1p(-theta * value) for value in values) / count

    profile: list[float] = []
    for theta in thetas:
        k_theta = shape_at(theta)
        ratio = -theta / k_theta if k_theta != 0.0 else math.nan
        if not ratio > 0.0:
            profile.append(-math.inf)
            continue
        profile.append(count * (math.log(ratio) - k_theta - 1.0))
    total = _log_sum_exp(profile)
    if not math.isfinite(total):
        return GeneralizedParetoFit(math.inf, math.nan, count, regularize)
    theta_hat = math.fsum(
        theta * math.exp(value - total)
        for theta, value in zip(thetas, profile, strict=True)
    )
    shape = shape_at(theta_hat)
    if theta_hat == 0.0 or not math.isfinite(shape):
        return GeneralizedParetoFit(math.inf, math.nan, count, regularize)
    scale = -shape / theta_hat
    if regularize:
        shape = (count * shape + _SHRINKAGE_COUNT * _SHRINKAGE_MEAN) / (
            count + _SHRINKAGE_COUNT
        )
    return GeneralizedParetoFit(shape, scale, count, regularize)


def psis_tail_size(sample_size: int) -> int:
    """Return the PSIS tail size ``M = ceil(min(S / 5, 3 sqrt(S)))``.

    Args:
        sample_size: Number of draws ``S`` (positive).

    Returns:
        Number of largest ratios that are smoothed. The ceiling follows the
        reference implementation of the source; the paper states the rule
        without the rounding.

    Raises:
        ValueError: If ``sample_size`` is not a positive integer.

    Equation:
        [EQ-INF-PSIS-001].

    References:
        [REF-VEHTARI-ETAL-2024].

    """
    size = require_non_negative_integer(sample_size, name="sample_size")
    if size == 0:
        raise ValueError("sample_size must be positive")
    return int(math.ceil(min(size / 5.0, 3.0 * math.sqrt(size))))


def pareto_k_threshold(sample_size: int) -> float:
    """Return the reliability threshold ``min(1 - 1 / log10(S), 0.7)``.

    The first term is the largest ``k`` for which ``S`` exceeds the minimum
    sample size ``10^(1 / (1 - k))`` of the source; the cap ``0.7`` is the
    value above which the source finds the bias of the smoothed estimate to
    dominate for any practical ``S``. ``S <= 10`` gives a threshold of zero
    or less: no positive ``k-hat`` counts as reliable.

    Args:
        sample_size: Number of draws ``S`` (at least 2).

    Returns:
        The threshold; estimates with ``k-hat`` below it are flagged
        reliable.

    Raises:
        ValueError: If ``sample_size`` is less than 2.

    Equation:
        [EQ-INF-PSIS-003].

    References:
        [REF-VEHTARI-ETAL-2024].

    """
    size = require_non_negative_integer(sample_size, name="sample_size")
    if size < 2:
        raise ValueError("sample_size must be at least 2")
    return min(1.0 - 1.0 / math.log10(size), _MAXIMUM_THRESHOLD)


@dataclass(frozen=True, slots=True)
class ImportanceEstimate:
    """A self-normalized importance-sampling estimate with its diagnostics.

    Args:
        estimate: ``sum_s w~_s h_s`` with normalized smoothed weights.
        standard_error: ``sqrt(sum_s w~_s^2 (h_s - estimate)^2)``, the
            delta-method standard error for independent draws.
        effective_sample_size: ``1 / sum_s w~_s^2``.
        pareto_k: Tail-shape diagnostic of the weights.
        reliable: Whether ``pareto_k`` is below the threshold for this
            sample size.

    References:
        [EQ-INF-PSIS-003], [REF-VEHTARI-ETAL-2024].

    """

    estimate: float
    standard_error: float
    effective_sample_size: float
    pareto_k: float
    reliable: bool


@dataclass(frozen=True, slots=True)
class PSISResult:
    """Pareto-smoothed importance weights and their diagnostics.

    Args:
        log_weights: Normalized smoothed log-weights in the order of the
            input (``sum exp = 1``).
        pareto_k: Fitted tail shape ``k-hat``; ``inf`` when the tail could
            not be fitted (fewer than five tail draws or a degenerate tail),
            ``-inf`` when all ratios are equal (library convention: the
            weights are uniform and nothing needs smoothing).
        scale: Fitted generalized Pareto scale of the tail exceedances, in
            units of the largest raw ratio (``nan`` when not fitted).
        tail_size: Number ``M`` of ratios in the tail.
        threshold: ``min(1 - 1 / log10(S), 0.7)``.
        smoothed: Whether the tail was replaced by fitted quantiles.
        effective_sample_size: ``1 / sum w~^2`` of the smoothed weights.

    References:
        [EQ-INF-PSIS-001], [EQ-INF-PSIS-003], [REF-VEHTARI-ETAL-2024].

    """

    log_weights: tuple[float, ...]
    pareto_k: float
    scale: float
    tail_size: int
    threshold: float
    smoothed: bool
    effective_sample_size: float

    @property
    def sample_size(self) -> int:
        """Return the number of draws ``S``.

        References:
            [REF-VEHTARI-ETAL-2024]. No equation is evaluated.

        """
        return len(self.log_weights)

    @property
    def weights(self) -> tuple[float, ...]:
        """Return the normalized smoothed weights ``w~_s`` (they sum to one).

        References:
            [EQ-INF-PSIS-001], [REF-VEHTARI-ETAL-2024].

        """
        return tuple(math.exp(value) for value in self.log_weights)

    @property
    def reliable(self) -> bool:
        """Return whether ``k-hat`` is below the sample-size threshold.

        References:
            [EQ-INF-PSIS-003], [REF-VEHTARI-ETAL-2024].

        """
        return self.pareto_k < self.threshold

    def expectation(self, values: Sequence[float]) -> ImportanceEstimate:
        """Return the self-normalized estimate ``sum_s w~_s h_s``.

        Args:
            values: ``h(x_s)`` for every draw, in the order of the ratios.

        Returns:
            Estimate, delta-method standard error for independent draws,
            effective sample size, and the reliability flag.

        Raises:
            ValueError: If the length differs from the number of draws or a
                value is not finite.

        Equation:
            [EQ-INF-PSIS-003].

        References:
            [REF-VEHTARI-ETAL-2024].

        """
        checked = tuple(require_finite_real(v, name="value") for v in values)
        if len(checked) != len(self.log_weights):
            raise ValueError("values must have one entry per draw")
        weights = self.weights
        estimate = math.fsum(w * v for w, v in zip(weights, checked, strict=True))
        variance = math.fsum(
            (w * (v - estimate)) ** 2 for w, v in zip(weights, checked, strict=True)
        )
        return ImportanceEstimate(
            estimate,
            math.sqrt(variance),
            self.effective_sample_size,
            self.pareto_k,
            self.reliable,
        )


def pareto_smooth(log_ratios: Sequence[float]) -> PSISResult:
    """Pareto-smooth importance ratios given on the log scale.

    The ``M`` largest ratios (``M`` from :func:`psis_tail_size`) are replaced
    by ``u + F^-1((z - 1/2) / M)``, ``z = 1, ..., M`` in increasing order,
    where ``u`` is the largest ratio outside the tail and ``F`` the
    generalized Pareto law fitted to the tail exceedances over ``u``; values
    above the largest raw ratio are truncated to it. All arithmetic is done
    relative to the largest ratio, so only differences of ``log_ratios``
    matter. When the tail has fewer than five draws or cannot be fitted the
    raw ratios are returned with ``pareto_k = inf`` (flagged unreliable).

    Args:
        log_ratios: ``log r_s = log p(x_s) - log g(x_s)`` up to a common
            additive constant, one per draw; ``-inf`` (zero ratio) is
            allowed, NaN and ``+inf`` are not.

    Returns:
        Smoothed normalized log-weights with ``k-hat``, the threshold, and
        the effective sample size.

    Raises:
        ValueError: For fewer than two ratios, a NaN or ``+inf`` ratio, or
            when every ratio is zero.

    Equation:
        [EQ-INF-PSIS-001], [EQ-INF-PSIS-003].

    References:
        [REF-VEHTARI-ETAL-2024].

    """
    raw = [float(value) for value in log_ratios]
    size = len(raw)
    if size < 2:
        raise ValueError("at least 2 log ratios are required")
    if any(math.isnan(value) or value == math.inf for value in raw):
        raise ValueError("log ratios must not be NaN or +inf")
    top = max(raw)
    if top == -math.inf:
        raise ValueError("at least one ratio must be positive")
    shifted = [value - top for value in raw]
    tail_size = psis_tail_size(size)
    threshold = pareto_k_threshold(size)
    pareto_k = math.inf
    scale = math.nan
    smoothed = False
    if min(shifted) == 0.0:
        pareto_k = -math.inf
    elif _MINIMUM_TAIL <= tail_size < size:
        order = sorted(range(size), key=shifted.__getitem__)
        tail = order[size - tail_size :]
        cutoff = math.exp(shifted[order[size - tail_size - 1]])
        fit = fit_generalized_pareto(
            [math.exp(shifted[index]) - cutoff for index in tail]
        )
        pareto_k = fit.shape
        scale = fit.scale
        if math.isfinite(fit.shape) and math.isfinite(fit.scale):
            smoothed = True
            for rank, index in enumerate(tail):
                value = cutoff + fit.quantile((rank + 0.5) / tail_size)
                shifted[index] = min(math.log(value), 0.0) if value > 0.0 else -math.inf
    total = _log_sum_exp(shifted)
    normalized = tuple(value - total for value in shifted)
    ess = 1.0 / math.fsum(math.exp(2.0 * value) for value in normalized)
    return PSISResult(normalized, pareto_k, scale, tail_size, threshold, smoothed, ess)


def psis_expectation(
    log_ratios: Sequence[float], values: Sequence[float]
) -> ImportanceEstimate:
    """Estimate ``E_p[h]`` from proposal draws by PSIS in one call.

    Equivalent to ``pareto_smooth(log_ratios).expectation(values)``.

    Args:
        log_ratios: ``log p(x_s) - log g(x_s)`` up to a constant.
        values: ``h(x_s)``.

    Returns:
        Self-normalized estimate with standard error, effective sample size,
        ``k-hat``, and the reliability flag.

    Raises:
        ValueError: As for :func:`pareto_smooth` and
            :meth:`PSISResult.expectation`.

    Equation:
        [EQ-INF-PSIS-001], [EQ-INF-PSIS-003].

    References:
        [REF-VEHTARI-ETAL-2024].

    """
    return pareto_smooth(log_ratios).expectation(values)


@dataclass(frozen=True, slots=True)
class PSISLOOResult:
    """PSIS leave-one-out predictive accuracy of a fitted Bayesian model.

    Args:
        elpd_loo: ``sum_i elpd_loo,i``, the estimated expected log pointwise
            predictive density.
        standard_error: ``sqrt(n Var_i(elpd_loo,i))``; it describes the
            variation over observations, not the Monte Carlo error.
        p_loo: ``lpd - elpd_loo``, the effective number of parameters, with
            ``lpd = sum_i log((1/S) sum_s p(y_i | theta_s))``.
        p_loo_standard_error: ``sqrt(n Var_i(p_loo,i))``.
        pointwise_elpd: ``elpd_loo,i`` per observation.
        pointwise_p_loo: ``lpd_i - elpd_loo,i`` per observation.
        pareto_k: ``k-hat`` of the leave-one-out ratios per observation.
        effective_sample_sizes: ``1 / sum_s w~_s^2`` per observation.
        threshold: ``min(1 - 1 / log10(S), 0.7)``.
        flagged: Indices of observations with ``k-hat`` at or above the
            threshold (their ``elpd_loo,i`` should not be trusted; refit
            without them or use K-fold cross-validation).
        draws: Number of posterior draws ``S``.

    References:
        [EQ-INF-PSIS-004], [REF-VEHTARI-GELMAN-GABRY-2017],
        [REF-VEHTARI-ETAL-2024].

    """

    elpd_loo: float
    standard_error: float
    p_loo: float
    p_loo_standard_error: float
    pointwise_elpd: tuple[float, ...]
    pointwise_p_loo: tuple[float, ...]
    pareto_k: tuple[float, ...]
    effective_sample_sizes: tuple[float, ...]
    threshold: float
    flagged: tuple[int, ...]
    draws: int

    @property
    def reliable(self) -> bool:
        """Return whether no observation was flagged.

        References:
            [EQ-INF-PSIS-003], [REF-VEHTARI-ETAL-2024].

        """
        return not self.flagged

    @property
    def observations(self) -> int:
        """Return the number of observations ``n``.

        References:
            [REF-VEHTARI-GELMAN-GABRY-2017]. No equation is evaluated.

        """
        return len(self.pointwise_elpd)


def _total_standard_error(values: Sequence[float]) -> float:
    """Return ``sqrt(n Var_i(v_i))`` with the ``n - 1`` variance divisor.

    References:
        [EQ-INF-PSIS-004], [REF-VEHTARI-GELMAN-GABRY-2017].

    """
    count = len(values)
    if count < 2:
        return math.nan
    center = math.fsum(values) / count
    variance = math.fsum((value - center) ** 2 for value in values) / (count - 1)
    return math.sqrt(count * variance)


def psis_loo(log_likelihood: Sequence[Sequence[float]]) -> PSISLOOResult:
    """Estimate leave-one-out predictive accuracy from posterior draws by PSIS.

    For observation ``i`` the ratios of the leave-one-out posterior to the
    full posterior are ``r_s = 1 / p(y_i | theta_s)``. With their smoothed
    weights ``w_s`` (:func:`pareto_smooth`),

        ``elpd_loo,i = log( sum_s w_s p(y_i | theta_s) / sum_s w_s )``,

    ``elpd_loo = sum_i elpd_loo,i`` with standard error
    ``sqrt(n Var_i(elpd_loo,i))``, and ``p_loo = lpd - elpd_loo``.

    Args:
        log_likelihood: Matrix with one row per posterior draw and one
            column per observation, ``log p(y_i | theta_s)`` including all
            normalizing constants. The observations must be conditionally
            independent given the parameters for the factorization to hold.

    Returns:
        Totals, pointwise values, and the per-observation ``k-hat`` with the
        indices that exceed the threshold.

    Raises:
        ValueError: For fewer than two draws, no observations, a ragged
            matrix, or a non-finite entry.

    Equation:
        [EQ-INF-PSIS-004].

    References:
        [REF-VEHTARI-GELMAN-GABRY-2017], [REF-VEHTARI-ETAL-2024].

    """
    rows = tuple(
        tuple(require_finite_real(value, name="log-likelihood value") for value in row)
        for row in log_likelihood
    )
    draws = len(rows)
    if draws < 2:
        raise ValueError("at least 2 posterior draws are required")
    observations = len(rows[0])
    if observations == 0:
        raise ValueError("at least one observation is required")
    if any(len(row) != observations for row in rows):
        raise ValueError("every draw needs one log-likelihood per observation")
    threshold = pareto_k_threshold(draws)
    elpd: list[float] = []
    p_loo: list[float] = []
    shapes: list[float] = []
    sizes: list[float] = []
    for i in range(observations):
        column = [row[i] for row in rows]
        smoothed = pareto_smooth([-value for value in column])
        value = _log_sum_exp(
            [w + v for w, v in zip(smoothed.log_weights, column, strict=True)]
        )
        lpd = _log_sum_exp(column) - math.log(draws)
        elpd.append(value)
        p_loo.append(lpd - value)
        shapes.append(smoothed.pareto_k)
        sizes.append(smoothed.effective_sample_size)
    flagged = tuple(i for i, shape in enumerate(shapes) if not shape < threshold)
    return PSISLOOResult(
        math.fsum(elpd),
        _total_standard_error(elpd),
        math.fsum(p_loo),
        _total_standard_error(p_loo),
        tuple(elpd),
        tuple(p_loo),
        tuple(shapes),
        tuple(sizes),
        threshold,
        flagged,
        draws,
    )
