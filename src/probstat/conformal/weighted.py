"""Weighted conformal quantiles: covariate shift and fixed data-independent weights.

Split conformal prediction gives every calibration score and the unseen test
score the same mass ``1/(n + 1)``. Two extensions replace that uniform law by
a weighted one and differ in what the weights mean and in what is guaranteed.

**Covariate shift** (Tibshirani, Barber, Candes and Ramdas 2019). The
calibration inputs follow ``P`` and the test input follows ``Q`` while the
conditional law of ``Y`` given ``X`` is the same. With the likelihood ratio
``w = dQ/dP`` *known* (up to a constant factor),

``p_i(x) = w(X_i) / (sum_j w(X_j) + w(x))``,
``p_{n+1}(x) = w(x) / (sum_j w(X_j) + w(x))``,

and the interval at ``x`` uses the ``1 - alpha`` quantile of
``sum_i p_i(x) delta_{S_i} + p_{n+1}(x) delta_{+inf}``. The threshold depends
on the test point through ``w(x)``. Marginal coverage under ``Q`` is at
least ``1 - alpha``; with estimated weights the guarantee is only
approximate.

**Fixed weights** (Barber, Candes, Ramdas and Tibshirani 2023). Weights
``w_i`` in ``[0, 1]`` chosen without looking at the data (for example
``rho^(n + 1 - i)`` to favour recent points) are normalized as
``w~_i = w_i / (sum_j w_j + 1)`` with ``w~_{n+1} = 1 / (sum_j w_j + 1)``.
Coverage is at least ``1 - alpha - sum_i w~_i d_TV(R(Z), R(Z^i))``, where
``Z^i`` swaps data points ``i`` and ``n + 1``. The total-variation terms are
properties of the unknown data law: the gap is *not computable from data*.
:func:`nonexchangeable_coverage_gap` only evaluates the sum for
user-supplied bounds on those distances.

Both reduce to split conformal prediction when every weight is one. The
quantile is located in exact rational arithmetic (weights as the binary
rationals their floats are, ``alpha`` as its decimal text), so that
reduction is exact.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from fractions import Fraction

from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_finite_real,
    require_probability,
)
from probstat.linalg import Vector, as_vector

from .quantile import _alpha_fraction
from .regression import (
    PointPredictor,
    PredictionInterval,
    PredictorLike,
    _calibration,
    _evaluate,
    _positive_scale,
    as_predictor,
)

WeightFunction = Callable[[Sequence[float]], float]


def _prefix_sums(weights: Sequence[float]) -> tuple[Fraction, ...]:
    """Return exact cumulative sums of float weights.

    References:
        [EQ-CONF-006]. Exact arithmetic policy of [EQ-CONF-001].

    """
    total = Fraction(0)
    sums: list[Fraction] = []
    for weight in weights:
        total += Fraction(weight)
        sums.append(total)
    return tuple(sums)


def _quantile_from_prefix(
    scores: Sequence[float],
    prefix: Sequence[Fraction],
    test_weight: float,
    level: Fraction,
) -> float:
    """Return the weighted conformal quantile from sorted scores and prefix sums.

    The result is the smallest score ``s_(k)`` whose cumulative weight reaches
    ``(1 - alpha)(W + w_test)``, or ``+inf`` when the calibration weight ``W``
    alone does not reach it.

    Raises:
        ValueError: If the total weight is zero.

    References:
        [EQ-CONF-006], [REF-TIBSHIRANI-ETAL-2019].

    """
    total = prefix[-1] + Fraction(test_weight)
    if total == 0:
        raise ValueError("the calibration and test weights must not all be zero")
    index = bisect.bisect_left(prefix, (1 - level) * total)
    return math.inf if index >= len(prefix) else scores[index]


def _sorted_pairs(
    scores: Iterable[object], weights: Iterable[object], *, unit: bool
) -> tuple[Vector, Vector]:
    """Validate scores and weights and return them sorted by score.

    Args:
        scores: Finite calibration scores.
        weights: Non-negative finite weights (in ``[0, 1]`` when ``unit``).
        unit: Whether weights must not exceed one.

    Raises:
        ValueError: If the inputs are empty, mismatched, or out of range.

    References:
        [EQ-CONF-006], [EQ-CONF-007]. No equation is evaluated.

    """
    values = tuple(require_finite_real(score, name="score") for score in scores)
    masses = tuple(
        require_probability(weight, name="weight")
        if unit
        else require_non_negative_finite_real(weight, name="weight")
        for weight in weights
    )
    if not values:
        raise ValueError("at least one calibration score is required")
    if len(values) != len(masses):
        raise ValueError("scores and weights must have equal length")
    order = sorted(range(len(values)), key=values.__getitem__)
    return tuple(values[i] for i in order), tuple(masses[i] for i in order)


def weighted_conformal_quantile(
    scores: Iterable[float],
    weights: Iterable[float],
    test_weight: float,
    alpha: float,
) -> float:
    """Return the weighted conformal quantile with mass at ``+inf``.

    With ``p_i = w_i / (sum_j w_j + w_test)`` and
    ``p_test = w_test / (sum_j w_j + w_test)`` this is
    ``inf{q : sum_{i : s_i <= q} p_i >= 1 - alpha}``, the ``1 - alpha``
    quantile of ``sum_i p_i delta_{s_i} + p_test delta_{+inf}``. Only the
    ratios of the weights matter. Equal weights give
    :func:`~probstat.conformal.conformal_quantile` exactly.

    Args:
        scores: Finite calibration scores (at least one).
        weights: Non-negative finite weights, one per score.
        test_weight: Non-negative finite weight of the test point.
        alpha: Miscoverage level in ``(0, 1)``.

    Returns:
        The quantile, ``+inf`` when the calibration mass below ``+inf`` is
        less than ``1 - alpha``.

    Raises:
        TypeError: If a value is not real.
        ValueError: If inputs are empty, mismatched, negative, non-finite,
            all weights are zero, or ``alpha`` is outside ``(0, 1)``.

    Equation:
        [EQ-CONF-006].

    References:
        [REF-TIBSHIRANI-ETAL-2019], [REF-BARBER-ETAL-2023].

    """
    level = _alpha_fraction(alpha)
    ordered, masses = _sorted_pairs(scores, weights, unit=False)
    mass = require_non_negative_finite_real(test_weight, name="test_weight")
    return _quantile_from_prefix(ordered, _prefix_sums(masses), mass, level)


def nonexchangeable_conformal_quantile(
    scores: Iterable[float], weights: Iterable[float], alpha: float
) -> float:
    """Return the conformal quantile for fixed weights ``w_i`` in ``[0, 1]``.

    The normalized weights are ``w~_i = w_i / (sum_j w_j + 1)`` and the test
    point carries ``w~_{n+1} = 1 / (sum_j w_j + 1)`` at ``+inf``; the result
    is the ``1 - alpha`` quantile of that weighted law. The weights must be
    fixed before seeing the data. Coverage is at least
    ``1 - alpha - sum_i w~_i d_TV(R(Z), R(Z^i))``; the gap cannot be computed
    from data.

    Args:
        scores: Finite calibration scores (at least one).
        weights: Weights in ``[0, 1]``, one per score.
        alpha: Miscoverage level in ``(0, 1)``.

    Returns:
        The quantile, possibly ``+inf``.

    Raises:
        TypeError: If a value is not real.
        ValueError: If inputs are empty or mismatched, a weight is outside
            ``[0, 1]``, or ``alpha`` is outside ``(0, 1)``.

    Equation:
        [EQ-CONF-007].

    References:
        [REF-BARBER-ETAL-2023].

    """
    level = _alpha_fraction(alpha)
    ordered, masses = _sorted_pairs(scores, weights, unit=True)
    return _quantile_from_prefix(ordered, _prefix_sums(masses), 1.0, level)


def nonexchangeable_coverage_gap(
    weights: Iterable[float], total_variation_bounds: Iterable[float]
) -> float:
    """Return ``sum_i w~_i d_i`` for user-supplied total-variation bounds.

    ``w~_i = w_i / (sum_j w_j + 1)``. With ``d_i`` an upper bound on
    ``d_TV(R(Z), R(Z^i))``, nonexchangeable conformal prediction covers with
    probability at least ``1 - alpha`` minus the returned value. The
    distances depend on the unknown data law and cannot be estimated by this
    function: it only evaluates the sum for bounds the user is prepared to
    assume. Exchangeable data have ``d_i = 0`` and no gap.

    Args:
        weights: Weights in ``[0, 1]``.
        total_variation_bounds: Assumed bounds ``d_i`` in ``[0, 1]``, one per
            weight.

    Returns:
        The coverage-gap bound, in ``[0, 1)``.

    Raises:
        TypeError: If a value is not real.
        ValueError: If inputs are empty or mismatched or a value is outside
            ``[0, 1]``.

    Equation:
        [EQ-CONF-007].

    References:
        [REF-BARBER-ETAL-2023].

    """
    masses = tuple(require_probability(w, name="weight") for w in weights)
    bounds = tuple(
        require_probability(d, name="total_variation_bound")
        for d in total_variation_bounds
    )
    if not masses:
        raise ValueError("at least one weight is required")
    if len(masses) != len(bounds):
        raise ValueError("weights and total_variation_bounds must have equal length")
    return math.fsum(w * d for w, d in zip(masses, bounds, strict=True)) / (
        math.fsum(masses) + 1.0
    )


def _weight_at(function: WeightFunction, x: Sequence[float]) -> float:
    """Evaluate a weight function and require a finite non-negative value.

    References:
        [EQ-CONF-006], [REF-TIBSHIRANI-ETAL-2019]. No equation is evaluated.

    """
    return require_non_negative_finite_real(function(x), name="weight(x)")


@dataclass(frozen=True)
class WeightedConformalRegressor:
    """Weighted split conformal intervals ``mu(x) +/- q(x) s(x)``.

    Args:
        predictor: Point predictor ``mu``.
        scale: Positive spread ``s`` for normalized scores, or ``None``.
        alpha: Miscoverage level.
        scores: Sorted calibration scores.
        weights: Calibration weights aligned with ``scores``.
        weight_function: Likelihood ratio ``w(x)`` evaluated at test inputs,
            or ``None`` when test weights are passed explicitly or fixed.
        test_weight: Fixed test-point weight (``1`` for nonexchangeable
            conformal with fixed weights), or ``None``.
        cumulative_weights: Exact prefix sums of ``weights``.

    References:
        [EQ-CONF-006], [EQ-CONF-007], [REF-TIBSHIRANI-ETAL-2019],
        [REF-BARBER-ETAL-2023].

    """

    predictor: PointPredictor
    scale: PointPredictor | None
    alpha: float
    scores: Vector
    weights: Vector
    weight_function: WeightFunction | None
    test_weight: float | None
    cumulative_weights: tuple[Fraction, ...] = field(repr=False, compare=False)

    @property
    def calibration_size(self) -> int:
        """Return the number ``n`` of calibration points.

        References:
            [REF-TIBSHIRANI-ETAL-2019]. No equation is evaluated.

        """
        return len(self.scores)

    @property
    def effective_sample_size(self) -> float:
        """Return the Kish heuristic ``(sum w)^2 / sum w^2`` of the weights.

        A diagnostic, not part of the guarantee: a value far below ``n``
        means a few calibration points carry most of the mass and the
        intervals will be wide or infinite. It is ``0`` when all weights
        vanish.

        References:
            [EQ-CONF-006], [REF-TIBSHIRANI-ETAL-2019].

        """
        squares = math.fsum(w * w for w in self.weights)
        return 0.0 if squares == 0.0 else math.fsum(self.weights) ** 2 / squares

    def quantile(self, test_weight: float) -> float:
        """Return the weighted conformal quantile for one test-point weight.

        Args:
            test_weight: Non-negative finite weight of the test point.

        Returns:
            ``q``, possibly ``+inf``.

        Raises:
            ValueError: If the weight is invalid or all weights are zero.

        Equation:
            [EQ-CONF-006].

        References:
            [REF-TIBSHIRANI-ETAL-2019], [REF-BARBER-ETAL-2023].

        """
        mass = require_non_negative_finite_real(test_weight, name="test_weight")
        return _quantile_from_prefix(
            self.scores, self.cumulative_weights, mass, _alpha_fraction(self.alpha)
        )

    def interval(
        self, x: Sequence[float], *, weight: float | None = None
    ) -> PredictionInterval:
        """Return the weighted conformal interval at one input vector.

        Args:
            x: Input vector.
            weight: Test-point weight ``w(x)``; required when the regressor
                has neither a weight function nor a fixed test weight, and
                overrides them otherwise.

        Returns:
            ``[mu(x) - q(x) s(x), mu(x) + q(x) s(x)]``.

        Raises:
            ValueError: If no test weight is available or a value is invalid.

        Equation:
            [EQ-CONF-006], [EQ-CONF-007].

        References:
            [REF-TIBSHIRANI-ETAL-2019], [REF-BARBER-ETAL-2023].

        """
        point = as_vector(x, name="x")
        if weight is not None:
            mass = weight
        elif self.weight_function is not None:
            mass = _weight_at(self.weight_function, point)
        elif self.test_weight is not None:
            mass = self.test_weight
        else:
            raise ValueError(
                "calibration weights were given as values; pass weight=w(x)"
            )
        center = _evaluate(self.predictor, point, "prediction")
        spread = 1.0 if self.scale is None else _positive_scale(self.scale, point)
        radius = self.quantile(mass)
        if radius == math.inf:
            return PredictionInterval(-math.inf, math.inf, center)
        half = radius * spread
        return PredictionInterval(center - half, center + half, center)

    def intervals(
        self,
        xs: Sequence[Sequence[float]],
        *,
        weights: Sequence[float] | None = None,
    ) -> tuple[PredictionInterval, ...]:
        """Return weighted conformal intervals at many input vectors.

        Args:
            xs: Input vectors.
            weights: Optional test-point weights, one per input.

        Returns:
            One interval per input.

        Raises:
            ValueError: If ``weights`` does not match ``xs``.

        Equation:
            [EQ-CONF-006], [EQ-CONF-007].

        References:
            [REF-TIBSHIRANI-ETAL-2019], [REF-BARBER-ETAL-2023].

        """
        if weights is None:
            return tuple(self.interval(x) for x in xs)
        if len(weights) != len(xs):
            raise ValueError("weights must match the number of inputs")
        return tuple(
            self.interval(x, weight=w) for x, w in zip(xs, weights, strict=True)
        )


def _build(
    predictor: PredictorLike,
    x_calibration: Sequence[Sequence[float]],
    y_calibration: Sequence[float],
    weights: Sequence[object],
    alpha: float,
    scale: PredictorLike | None,
    *,
    weight_function: WeightFunction | None,
    test_weight: float | None,
    unit: bool,
) -> WeightedConformalRegressor:
    """Score the calibration set and assemble a weighted regressor.

    References:
        [EQ-CONF-002], [EQ-CONF-006], [EQ-CONF-007].

    """
    mu = as_predictor(predictor)
    spread = None if scale is None else as_predictor(scale)
    level = _alpha_fraction(alpha)
    design, responses = _calibration(x_calibration, y_calibration)
    if len(weights) != len(design):
        raise ValueError("weights must match the number of calibration points")
    scores: list[float] = []
    for x, y in zip(design, responses, strict=True):
        residual = abs(y - _evaluate(mu, x, "prediction"))
        scores.append(
            residual if spread is None else residual / _positive_scale(spread, x)
        )
    ordered, masses = _sorted_pairs(scores, weights, unit=unit)
    return WeightedConformalRegressor(
        mu,
        spread,
        float(level),
        ordered,
        masses,
        weight_function,
        test_weight,
        _prefix_sums(masses),
    )


def weighted_split_conformal_regression(
    predictor: PredictorLike,
    x_calibration: Sequence[Sequence[float]],
    y_calibration: Sequence[float],
    weight: WeightFunction | Sequence[float],
    *,
    alpha: float = 0.1,
    scale: PredictorLike | None = None,
) -> WeightedConformalRegressor:
    """Calibrate split conformal intervals for a test law with shifted inputs.

    Assumes covariate shift: calibration inputs follow ``P``, test inputs
    follow ``Q``, the conditional law of ``Y`` given ``X`` is shared, and
    ``weight`` is the likelihood ratio ``dQ/dP`` up to a constant factor.
    Then ``P_Q(Y in C(X)) >= 1 - alpha`` marginally over the calibration set
    and the test point. The guarantee needs the true ratio; with an
    estimated ratio it holds only approximately, and nothing is guaranteed
    under a shift of the conditional law. Regions where ``Q`` has mass and
    ``P`` has almost none give large weights and infinite intervals.

    Args:
        predictor: Point predictor fitted without the calibration data.
        x_calibration: Held-out input vectors drawn from ``P``.
        y_calibration: Held-out responses.
        weight: Either a callable ``w(x) >= 0`` (evaluated at calibration
            and test inputs) or the calibration weights ``w(X_i)``; in the
            second case pass ``weight=w(x)`` to ``interval``.
        alpha: Miscoverage level in ``(0, 1)``.
        scale: Optional positive spread ``s(x)`` for normalized scores.

    Returns:
        Regressor whose threshold depends on the test point through ``w(x)``.

    Raises:
        TypeError: If a predictor is not callable or a weight is not real.
        ValueError: If the calibration set is empty or mismatched, a weight
            is negative or non-finite, or ``alpha`` is invalid.

    Equation:
        [EQ-CONF-006].

    References:
        [REF-TIBSHIRANI-ETAL-2019].

    """
    if callable(weight):
        design, _ = _calibration(x_calibration, y_calibration)
        values: Sequence[object] = [_weight_at(weight, x) for x in design]
        function: WeightFunction | None = weight
    else:
        values = weight
        function = None
    return _build(
        predictor,
        x_calibration,
        y_calibration,
        values,
        alpha,
        scale,
        weight_function=function,
        test_weight=None,
        unit=False,
    )


def nonexchangeable_split_conformal_regression(
    predictor: PredictorLike,
    x_calibration: Sequence[Sequence[float]],
    y_calibration: Sequence[float],
    weights: Sequence[float],
    *,
    alpha: float = 0.1,
    scale: PredictorLike | None = None,
) -> WeightedConformalRegressor:
    """Calibrate split conformal intervals with fixed weights in ``[0, 1]``.

    The weights must be chosen without looking at the data, typically
    decreasing with the distance (in time or otherwise) from the test point.
    The interval uses one threshold, the ``1 - alpha`` quantile of
    ``sum_i w~_i delta_{S_i} + w~_{n+1} delta_{+inf}``. Marginal coverage is
    at least ``1 - alpha - sum_i w~_i d_TV(R(Z), R(Z^i))``: exactly
    ``1 - alpha`` or more for exchangeable data, and degraded by a gap that
    cannot be computed from data otherwise.

    Args:
        predictor: Point predictor fitted without the calibration data.
        x_calibration: Held-out input vectors.
        y_calibration: Held-out responses.
        weights: Fixed weights in ``[0, 1]``, one per calibration point.
        alpha: Miscoverage level in ``(0, 1)``.
        scale: Optional positive spread ``s(x)`` for normalized scores.

    Returns:
        Regressor with test-point weight one.

    Raises:
        TypeError: If a predictor is not callable or a weight is not real.
        ValueError: If the calibration set is empty or mismatched, a weight
            is outside ``[0, 1]``, or ``alpha`` is invalid.

    Equation:
        [EQ-CONF-007].

    References:
        [REF-BARBER-ETAL-2023].

    """
    return _build(
        predictor,
        x_calibration,
        y_calibration,
        weights,
        alpha,
        scale,
        weight_function=None,
        test_weight=1.0,
        unit=True,
    )
