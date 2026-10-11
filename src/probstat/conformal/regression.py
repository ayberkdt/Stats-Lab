"""Split conformal, conformalized quantile regression, and jackknife+ intervals.

All three wrap an arbitrary predictor and return prediction intervals whose
*marginal* coverage, averaged over the calibration data and the test point, is
at least ``1 - alpha`` (``1 - 2 alpha`` for jackknife+) when the data are
exchangeable. Nothing is assumed about the predictor: a poor model gives
valid but wide intervals. Coverage conditional on a particular ``x`` is not
guaranteed.

Predictors are callables ``x -> float`` on input vectors, or any object with a
``predict(x)`` method such as the fitted surrogates in
:mod:`probstat.surrogates`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from probstat.foundations.validation import require_finite_real
from probstat.linalg import Vector, as_vector
from probstat.surrogates import Surrogate

from .quantile import (
    conformal_quantile,
    conformal_rank,
    lower_conformal_rank,
    split_conformal_coverage,
)

PointPredictor = Callable[[Sequence[float]], float]
PredictorLike = PointPredictor | Surrogate
FitFunction = Callable[[tuple[Vector, ...], Vector], PredictorLike]
ScoreKind = Literal["absolute", "normalized"]


def as_predictor(predictor: PredictorLike) -> PointPredictor:
    """Return a plain callable for a callable or a ``predict``-bearing object.

    Args:
        predictor: Callable ``x -> float`` or object with ``predict(x)``.

    Returns:
        Callable evaluating the predictor.

    Raises:
        TypeError: If ``predictor`` is neither.

    References:
        [REF-LEI-ETAL-2018]. No equation is evaluated.

    """
    if isinstance(predictor, Surrogate):
        return predictor.predict
    if callable(predictor):
        return predictor
    raise TypeError("predictor must be callable or provide predict(x)")


def _evaluate(function: PointPredictor, x: Sequence[float], name: str) -> float:
    """Evaluate a user function and require a finite real result.

    References:
        Project validation policy. No equation is evaluated.

    """
    return require_finite_real(function(x), name=name)


def _calibration(
    points: Sequence[Sequence[float]], values: Sequence[float]
) -> tuple[tuple[Vector, ...], Vector]:
    """Validate a calibration set of input vectors and responses.

    References:
        Project validation policy. No equation is evaluated.

    """
    design = tuple(as_vector(point, name="x") for point in points)
    responses = as_vector(values, name="y")
    if not design:
        raise ValueError("at least one calibration point is required")
    if len(design) != len(responses):
        raise ValueError("x and y must have equal length")
    return design, responses


@dataclass(frozen=True, slots=True)
class PredictionInterval:
    """A prediction interval ``[lower, upper]`` and the point it was built on.

    Args:
        lower: Lower limit (may be ``-inf``).
        upper: Upper limit (may be ``+inf``); ``upper < lower`` encodes an
            empty set, which conformalized quantile regression can produce.
        prediction: Point prediction used to build the interval (the
            median leave-one-out prediction for jackknife+).

    References:
        [REF-LEI-ETAL-2018].

    """

    lower: float
    upper: float
    prediction: float

    @property
    def width(self) -> float:
        """Return ``max(upper - lower, 0)``.

        References:
            [REF-LEI-ETAL-2018]. No equation is evaluated.

        """
        return max(self.upper - self.lower, 0.0)

    def contains(self, value: float) -> bool:
        """Return whether ``lower <= value <= upper``.

        Args:
            value: Finite response.

        Returns:
            Coverage indicator.

        References:
            [REF-LEI-ETAL-2018]. No equation is evaluated.

        """
        y = require_finite_real(value, name="value")
        return self.lower <= y <= self.upper


@dataclass(frozen=True)
class SplitConformalRegressor:
    """Split conformal intervals ``mu(x) +/- q_hat s(x)`` around a predictor.

    Args:
        predictor: Point predictor ``mu``.
        scale: Positive spread function ``s`` for normalized scores, or
            ``None`` for absolute residuals (``s = 1``).
        alpha: Miscoverage level.
        score: ``"absolute"`` or ``"normalized"``.
        scores: Sorted calibration scores.
        rank: Order-statistic index ``k``.
        quantile: Conformal quantile ``q_hat`` (``inf`` when ``k > n``).

    References:
        [EQ-CONF-001], [EQ-CONF-002], [REF-LEI-ETAL-2018],
        [REF-VOVK-GAMMERMAN-SHAFER-2005].

    """

    predictor: PointPredictor
    scale: PointPredictor | None
    alpha: float
    score: ScoreKind
    scores: Vector
    rank: int
    quantile: float

    @property
    def calibration_size(self) -> int:
        """Return the number ``n`` of calibration points.

        References:
            [REF-LEI-ETAL-2018]. No equation is evaluated.

        """
        return len(self.scores)

    @property
    def guaranteed_coverage(self) -> float:
        """Return the marginal coverage ``k / (n + 1) >= 1 - alpha``.

        References:
            [EQ-CONF-005], [REF-LEI-ETAL-2018].

        """
        return split_conformal_coverage(self.calibration_size, self.alpha)

    def interval(self, x: Sequence[float]) -> PredictionInterval:
        """Return the conformal interval at one input vector.

        Args:
            x: Input vector.

        Returns:
            ``[mu(x) - q_hat s(x), mu(x) + q_hat s(x)]``.

        Raises:
            ValueError: If the predictor or scale returns an invalid value.

        Equation:
            [EQ-CONF-002].

        References:
            [REF-LEI-ETAL-2018].

        """
        point = as_vector(x, name="x")
        center = _evaluate(self.predictor, point, "prediction")
        spread = 1.0 if self.scale is None else _positive_scale(self.scale, point)
        half = self.quantile * spread
        return PredictionInterval(center - half, center + half, center)

    def intervals(
        self, xs: Sequence[Sequence[float]]
    ) -> tuple[PredictionInterval, ...]:
        """Return conformal intervals at many input vectors.

        Args:
            xs: Input vectors.

        Returns:
            One interval per input.

        Equation:
            [EQ-CONF-002].

        References:
            [REF-LEI-ETAL-2018].

        """
        return tuple(self.interval(x) for x in xs)


def _positive_scale(scale: PointPredictor, x: Sequence[float]) -> float:
    """Evaluate a spread function and require a positive finite value.

    References:
        [REF-LEI-ETAL-2018]. No equation is evaluated.

    """
    value = _evaluate(scale, x, "scale")
    if not value > 0.0:
        raise ValueError("scale(x) must be positive")
    return value


def split_conformal_regression(
    predictor: PredictorLike,
    x_calibration: Sequence[Sequence[float]],
    y_calibration: Sequence[float],
    *,
    alpha: float = 0.1,
    scale: PredictorLike | None = None,
) -> SplitConformalRegressor:
    """Calibrate split conformal regression intervals on held-out data.

    Scores are ``|y_i - mu(x_i)|`` (absolute) or ``|y_i - mu(x_i)| / s(x_i)``
    (normalized, locally weighted) when ``scale`` is given; ``q_hat`` is their
    ``ceil((1 - alpha)(n + 1))``-th smallest value. The predictor and scale
    must not have been fitted on the calibration points.

    Args:
        predictor: Point predictor (callable or object with ``predict``).
        x_calibration: Held-out input vectors.
        y_calibration: Held-out responses.
        alpha: Miscoverage level in ``(0, 1)``.
        scale: Optional positive spread estimate ``s(x)`` (for example a GP
            predictive standard deviation or a fitted absolute-residual
            model).

    Returns:
        Fitted regressor producing intervals.

    Raises:
        TypeError: If a predictor is not callable.
        ValueError: If the calibration set is empty or mismatched, ``alpha``
            is invalid, or a prediction or scale is invalid.

    Equation:
        [EQ-CONF-001], [EQ-CONF-002].

    References:
        [REF-LEI-ETAL-2018], [REF-VOVK-GAMMERMAN-SHAFER-2005],
        [REF-ANGELOPOULOS-BATES-2023].

    """
    mu = as_predictor(predictor)
    spread = None if scale is None else as_predictor(scale)
    design, responses = _calibration(x_calibration, y_calibration)
    scores: list[float] = []
    for x, y in zip(design, responses, strict=True):
        residual = abs(y - _evaluate(mu, x, "prediction"))
        scores.append(
            residual if spread is None else residual / _positive_scale(spread, x)
        )
    ordered = tuple(sorted(scores))
    return SplitConformalRegressor(
        mu,
        spread,
        float(alpha),
        "absolute" if spread is None else "normalized",
        ordered,
        conformal_rank(len(ordered), alpha),
        conformal_quantile(ordered, alpha),
    )


@dataclass(frozen=True)
class ConformalizedQuantileRegressor:
    """Conformalized quantile regression ``[q_lo(x) - q_hat, q_hi(x) + q_hat]``.

    Args:
        lower_predictor: Estimated lower conditional quantile ``q_lo``.
        upper_predictor: Estimated upper conditional quantile ``q_hi``.
        alpha: Miscoverage level.
        scores: Sorted scores ``max(q_lo(x_i) - y_i, y_i - q_hi(x_i))``.
        rank: Order-statistic index ``k``.
        quantile: Conformal quantile ``q_hat`` (negative values shrink the
            band; ``inf`` when ``k > n``).

    References:
        [EQ-CONF-003], [REF-ROMANO-PATTERSON-CANDES-2019].

    """

    lower_predictor: PointPredictor
    upper_predictor: PointPredictor
    alpha: float
    scores: Vector
    rank: int
    quantile: float

    @property
    def guaranteed_coverage(self) -> float:
        """Return the marginal coverage ``k / (n + 1) >= 1 - alpha``.

        References:
            [EQ-CONF-005], [REF-ROMANO-PATTERSON-CANDES-2019].

        """
        return split_conformal_coverage(len(self.scores), self.alpha)

    def interval(self, x: Sequence[float]) -> PredictionInterval:
        """Return the conformalized quantile interval at one input.

        Args:
            x: Input vector.

        Returns:
            ``[q_lo(x) - q_hat, q_hi(x) + q_hat]`` with the midpoint of the
            two quantile predictions as ``prediction``.

        Equation:
            [EQ-CONF-003].

        References:
            [REF-ROMANO-PATTERSON-CANDES-2019].

        """
        point = as_vector(x, name="x")
        low = _evaluate(self.lower_predictor, point, "lower prediction")
        high = _evaluate(self.upper_predictor, point, "upper prediction")
        return PredictionInterval(
            low - self.quantile, high + self.quantile, 0.5 * (low + high)
        )

    def intervals(
        self, xs: Sequence[Sequence[float]]
    ) -> tuple[PredictionInterval, ...]:
        """Return conformalized quantile intervals at many inputs.

        Args:
            xs: Input vectors.

        Returns:
            One interval per input.

        Equation:
            [EQ-CONF-003].

        References:
            [REF-ROMANO-PATTERSON-CANDES-2019].

        """
        return tuple(self.interval(x) for x in xs)


def conformalized_quantile_regression(
    lower_predictor: PredictorLike,
    upper_predictor: PredictorLike,
    x_calibration: Sequence[Sequence[float]],
    y_calibration: Sequence[float],
    *,
    alpha: float = 0.1,
) -> ConformalizedQuantileRegressor:
    """Calibrate given lower/upper quantile predictors on held-out data.

    The quantile predictors are typically estimates of the ``alpha/2`` and
    ``1 - alpha/2`` conditional quantiles fitted on a separate training set;
    any pair of functions is accepted and coverage holds regardless.

    Args:
        lower_predictor: Lower quantile predictor.
        upper_predictor: Upper quantile predictor.
        x_calibration: Held-out input vectors.
        y_calibration: Held-out responses.
        alpha: Miscoverage level in ``(0, 1)``.

    Returns:
        Fitted regressor producing adaptive-width intervals.

    Raises:
        TypeError: If a predictor is not callable.
        ValueError: If the calibration set is invalid or ``alpha`` is outside
            ``(0, 1)``.

    Equation:
        [EQ-CONF-001], [EQ-CONF-003].

    References:
        [REF-ROMANO-PATTERSON-CANDES-2019], [REF-ANGELOPOULOS-BATES-2023].

    """
    low = as_predictor(lower_predictor)
    high = as_predictor(upper_predictor)
    design, responses = _calibration(x_calibration, y_calibration)
    scores = tuple(
        sorted(
            max(
                _evaluate(low, x, "lower prediction") - y,
                y - _evaluate(high, x, "upper prediction"),
            )
            for x, y in zip(design, responses, strict=True)
        )
    )
    return ConformalizedQuantileRegressor(
        low,
        high,
        float(alpha),
        scores,
        conformal_rank(len(scores), alpha),
        conformal_quantile(scores, alpha),
    )


@dataclass(frozen=True)
class JackknifePlusRegressor:
    """Jackknife+ intervals from leave-one-out predictors and residuals.

    Args:
        loo_predictors: ``mu_{-i}``, the model refitted without point ``i``.
        residuals: ``R_i = |y_i - mu_{-i}(x_i)|``.
        alpha: Miscoverage level.

    References:
        [EQ-CONF-004], [REF-BARBER-ETAL-2021].

    """

    loo_predictors: tuple[PointPredictor, ...]
    residuals: Vector
    alpha: float

    @property
    def size(self) -> int:
        """Return the number ``n`` of training points.

        References:
            [REF-BARBER-ETAL-2021]. No equation is evaluated.

        """
        return len(self.residuals)

    def interval(self, x: Sequence[float]) -> PredictionInterval:
        """Return the jackknife+ interval at one input.

        ``lower`` is the ``floor(alpha (n + 1))``-th smallest of
        ``mu_{-i}(x) - R_i`` (``-inf`` when that index is 0) and ``upper`` the
        ``ceil((1 - alpha)(n + 1))``-th smallest of ``mu_{-i}(x) + R_i``
        (``+inf`` when it exceeds ``n``).

        Args:
            x: Input vector.

        Returns:
            Interval with the median leave-one-out prediction as
            ``prediction``.

        Equation:
            [EQ-CONF-004].

        References:
            [REF-BARBER-ETAL-2021].

        """
        point = as_vector(x, name="x")
        predictions = [
            _evaluate(model, point, "loo prediction") for model in self.loo_predictors
        ]
        lows = sorted(p - r for p, r in zip(predictions, self.residuals, strict=True))
        highs = sorted(p + r for p, r in zip(predictions, self.residuals, strict=True))
        n = self.size
        lower_index = lower_conformal_rank(n, self.alpha)
        upper_index = conformal_rank(n, self.alpha)
        lower = -math.inf if lower_index == 0 else lows[lower_index - 1]
        upper = math.inf if upper_index > n else highs[upper_index - 1]
        ordered = sorted(predictions)
        middle = n // 2
        center = (
            ordered[middle]
            if n % 2 == 1
            else 0.5 * (ordered[middle - 1] + ordered[middle])
        )
        return PredictionInterval(lower, upper, center)

    def intervals(
        self, xs: Sequence[Sequence[float]]
    ) -> tuple[PredictionInterval, ...]:
        """Return jackknife+ intervals at many inputs.

        Args:
            xs: Input vectors.

        Returns:
            One interval per input.

        Equation:
            [EQ-CONF-004].

        References:
            [REF-BARBER-ETAL-2021].

        """
        return tuple(self.interval(x) for x in xs)


def jackknife_plus_regression(
    fit: FitFunction,
    x_train: Sequence[Sequence[float]],
    y_train: Sequence[float],
    *,
    alpha: float = 0.1,
) -> JackknifePlusRegressor:
    """Build jackknife+ intervals by refitting the model ``n`` times.

    Every training point is left out once; the refit predicts it to give the
    residual ``R_i``. No data are held out for calibration, at the cost of
    ``n`` refits. The marginal coverage is at least ``1 - 2 alpha`` for any
    symmetric fitting algorithm and exchangeable data, and typically close to
    ``1 - alpha``.

    Args:
        fit: Function ``(points, values) -> predictor``; it must treat the
            training points symmetrically (order-independent).
        x_train: Training input vectors (at least two).
        y_train: Training responses.
        alpha: Miscoverage level in ``(0, 1)``.

    Returns:
        Regressor holding the ``n`` leave-one-out predictors.

    Raises:
        TypeError: If ``fit`` is not callable or returns a non-predictor.
        ValueError: If fewer than two points are given or inputs mismatch.

    Equation:
        [EQ-CONF-004].

    References:
        [REF-BARBER-ETAL-2021].

    """
    if not callable(fit):
        raise TypeError("fit must be callable")
    design, responses = _calibration(x_train, y_train)
    n = len(design)
    if n < 2:
        raise ValueError("jackknife+ needs at least two training points")
    lower_conformal_rank(n, alpha)
    models: list[PointPredictor] = []
    residuals: list[float] = []
    for i in range(n):
        points = design[:i] + design[i + 1 :]
        values = responses[:i] + responses[i + 1 :]
        model = as_predictor(fit(points, values))
        models.append(model)
        residuals.append(abs(responses[i] - _evaluate(model, design[i], "prediction")))
    return JackknifePlusRegressor(tuple(models), tuple(residuals), float(alpha))
