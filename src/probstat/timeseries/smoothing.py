"""Exponential smoothing: simple (level) and Holt's linear trend method.

Error-correction form with one-step errors ``e_t = y_t - y_hat_(t|t-1)``:

* simple: ``y_hat = l_(t-1)``, ``l_t = l_(t-1) + alpha e_t``;
* Holt: ``y_hat = l_(t-1) + b_(t-1)``, ``l_t = l_(t-1) + b_(t-1) + alpha e_t``,
  ``b_t = b_(t-1) + alpha beta e_t``.

The Holt recursion is algebraically the textbook form
``l_t = alpha y_t + (1 - alpha)(l_(t-1) + b_(t-1))`` and
``b_t = beta (l_t - l_(t-1)) + (1 - beta) b_(t-1)``. Smoothing parameters are
chosen by minimizing the one-step sum of squared errors over ``(0, 1)``
through a logistic transform. Because every error is an affine function of the
initial states, the initial level (and trend) are concentrated out exactly by
linear least squares for each candidate parameter value -- a project choice
that removes the arbitrariness of heuristic starting values.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from probstat.foundations.validation import (
    coerce_finite_sample,
    require_finite_real,
    require_non_negative_integer,
)
from probstat.linalg import solve_spd
from probstat.optimization import minimize_bfgs

Vector = tuple[float, ...]


def _logistic(value: float) -> float:
    """Return ``1 / (1 + exp(-value))`` without overflow.

    Args:
        value: Real number.

    Returns:
        Value in ``(0, 1)``.

    References:
        Project numerical policy. No named equation is needed.

    """
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _logit(probability: float) -> float:
    """Return ``log(p / (1 - p))``.

    Args:
        probability: Value in ``(0, 1)``.

    Returns:
        Unconstrained value.

    References:
        Project numerical policy. No named equation is needed.

    """
    return math.log(probability / (1.0 - probability))


def _errors(
    data: Sequence[float], alpha: float, beta: float | None, initial: Sequence[float]
) -> tuple[list[float], list[float], float, float]:
    """Run the smoothing recursion and return errors, fits, and final states.

    Args:
        data: Observations.
        alpha: Level smoothing parameter.
        beta: Trend smoothing parameter or ``None`` for simple smoothing.
        initial: ``(l_0,)`` or ``(l_0, b_0)``.

    Returns:
        One-step errors, one-step predictions, final level, final trend.

    Equation:
        [EQ-TS-ES-001], [EQ-TS-ES-002].

    References:
        [REF-BROCKWELL-DAVIS-2016, Chapter 10].

    """
    level = initial[0]
    trend = initial[1] if beta is not None else 0.0
    errors: list[float] = []
    fits: list[float] = []
    for value in data:
        prediction = level + trend
        error = value - prediction
        fits.append(prediction)
        errors.append(error)
        level = prediction + alpha * error
        if beta is not None:
            trend = trend + alpha * beta * error
    return errors, fits, level, trend


def _concentrate(
    data: Sequence[float], alpha: float, beta: float | None
) -> tuple[Vector, float]:
    """Return least-squares initial states and the resulting SSE.

    The errors satisfy ``e(s) = e(0) - A s`` for initial states ``s``; the
    columns of ``A`` come from runs with unit initial states.

    Args:
        data: Observations.
        alpha: Level parameter.
        beta: Trend parameter or ``None``.

    Returns:
        Optimal initial states and the minimal sum of squared errors.

    Raises:
        ValueError: If the normal equations are singular.

    Equation:
        [EQ-TS-ES-001], [EQ-TS-ES-002].

    References:
        [REF-BROCKWELL-DAVIS-2016, Chapter 10]; concentration is a project
        choice.

    """
    size = 1 if beta is None else 2
    base, _, _, _ = _errors(data, alpha, beta, (0.0, 0.0))
    columns: list[list[float]] = []
    for k in range(size):
        unit = [0.0, 0.0]
        unit[k] = 1.0
        shifted, _, _, _ = _errors(data, alpha, beta, unit)
        columns.append([b - s for b, s in zip(base, shifted, strict=True)])
    normal = tuple(
        tuple(math.fsum(a * b for a, b in zip(ci, cj, strict=True)) for cj in columns)
        for ci in columns
    )
    right = tuple(
        math.fsum(a * b for a, b in zip(c, base, strict=True)) for c in columns
    )
    states = solve_spd(normal, right)
    initial = list(states) + [0.0] * (2 - size)
    errors, _, _, _ = _errors(data, alpha, beta, initial)
    return tuple(states), math.fsum(e * e for e in errors)


@dataclass(frozen=True, slots=True)
class ExponentialSmoothingFit:
    """Fitted simple or Holt exponential smoothing model.

    Args:
        method: ``"simple"`` or ``"holt"``.
        alpha: Level smoothing parameter.
        beta: Trend smoothing parameter (``None`` for simple smoothing).
        initial_level: Least-squares ``l_0``.
        initial_trend: Least-squares ``b_0`` (``None`` for simple smoothing).
        level: Final level ``l_n``.
        trend: Final trend ``b_n`` (``None`` for simple smoothing).
        fitted: One-step predictions ``y_hat_(t|t-1)``.
        residuals: One-step errors.
        sse: Sum of squared one-step errors.
        variance: ``sse / n``.
        converged: Optimizer convergence flag (``True`` for fixed parameters).

    References:
        [REF-BROCKWELL-DAVIS-2016, Chapter 10], [REF-HOLT-2004].

    """

    method: str
    alpha: float
    beta: float | None
    initial_level: float
    initial_trend: float | None
    level: float
    trend: float | None
    fitted: Vector
    residuals: Vector
    sse: float
    variance: float
    converged: bool

    def forecast(self, steps: int) -> Vector:
        """Return point forecasts ``l_n + h b_n`` for ``h = 1..steps``.

        Args:
            steps: Horizon ``>= 1``.

        Returns:
            Point forecasts.

        Raises:
            ValueError: If ``steps`` is zero.

        Equation:
            [EQ-TS-ES-001], [EQ-TS-ES-002].

        References:
            [REF-BROCKWELL-DAVIS-2016, Chapter 10].

        """
        h = require_non_negative_integer(steps, name="steps")
        if h < 1:
            raise ValueError("steps must be positive")
        slope = self.trend if self.trend is not None else 0.0
        return tuple(self.level + k * slope for k in range(1, h + 1))


def _parameter(value: float | None, name: str) -> float | None:
    """Validate an optional smoothing parameter in ``(0, 1)``.

    Args:
        value: Candidate or ``None``.
        name: Name used in error messages.

    Returns:
        The validated value or ``None``.

    Raises:
        ValueError: If the value is outside ``(0, 1)``.

    References:
        Project validation policy. No equation is evaluated.

    """
    if value is None:
        return None
    result = require_finite_real(value, name=name)
    if not 0.0 < result < 1.0:
        raise ValueError(f"{name} must lie strictly between zero and one")
    return result


def _fit(
    data: Vector, alpha: float | None, beta: float | None, holt: bool
) -> ExponentialSmoothingFit:
    """Optimize free smoothing parameters and assemble the fit.

    Args:
        data: Observations.
        alpha: Fixed level parameter or ``None`` to optimize.
        beta: Fixed trend parameter or ``None`` to optimize (Holt only).
        holt: Whether the trend component is present.

    Returns:
        Fitted model.

    Equation:
        [EQ-TS-ES-001], [EQ-TS-ES-002], [EQ-OPT-BFGS-001].

    References:
        [REF-BROCKWELL-DAVIS-2016, Chapter 10].

    """
    free: list[str] = []
    start: list[float] = []
    if alpha is None:
        free.append("alpha")
        start.append(_logit(0.3))
    if holt and beta is None:
        free.append("beta")
        start.append(_logit(0.1))
    scale = max(1.0, math.fsum(v * v for v in data) / len(data))

    def resolve(u: Sequence[float]) -> tuple[float, float | None]:
        """Return ``(alpha, beta)`` from free unconstrained values.

        References:
            Project numerical policy. No equation is evaluated.

        """
        values = dict(zip(free, (_logistic(x) for x in u), strict=True))
        a = values.get("alpha", alpha if alpha is not None else 0.0)
        b = values.get("beta", beta) if holt else None
        return a, b

    def objective(u: Sequence[float]) -> float:
        """Return the concentrated SSE scaled by ``n`` and the data scale.

        References:
            [EQ-TS-ES-001], [EQ-TS-ES-002].

        """
        a, b = resolve(u)
        return _concentrate(data, a, b)[1] / (len(data) * scale)

    converged = True
    best: Sequence[float] = ()
    if start:
        result = minimize_bfgs(objective, start, gradient_tolerance=1e-9)
        best, converged = result.x, result.converged
    a, b = resolve(best)
    states, sse = _concentrate(data, a, b)
    initial = list(states) + [0.0] * (2 - len(states))
    errors, fits, level, trend = _errors(data, a, b, initial)
    return ExponentialSmoothingFit(
        method="holt" if holt else "simple",
        alpha=a,
        beta=b,
        initial_level=states[0],
        initial_trend=states[1] if holt else None,
        level=level,
        trend=trend if holt else None,
        fitted=tuple(fits),
        residuals=tuple(errors),
        sse=sse,
        variance=sse / len(data),
        converged=converged,
    )


def simple_exponential_smoothing(
    series: Iterable[float], *, alpha: float | None = None
) -> ExponentialSmoothingFit:
    """Fit simple exponential smoothing with an optimized ``alpha``.

    Args:
        series: At least three finite observations.
        alpha: Fixed level parameter in ``(0, 1)``; optimized when ``None``.

    Returns:
        Fitted model; forecasts are flat at the final level.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the series is too short or ``alpha`` is invalid.

    Equation:
        [EQ-TS-ES-001].

    References:
        [REF-BROCKWELL-DAVIS-2016, Chapter 10], [REF-HOLT-2004].

    """
    data = coerce_finite_sample(series, name="series")
    if len(data) < 3:
        raise ValueError("series must contain at least three observations")
    return _fit(data, _parameter(alpha, "alpha"), None, holt=False)


def holt_linear(
    series: Iterable[float],
    *,
    alpha: float | None = None,
    beta: float | None = None,
) -> ExponentialSmoothingFit:
    """Fit Holt's linear-trend exponential smoothing with optimized parameters.

    Args:
        series: At least four finite observations.
        alpha: Fixed level parameter in ``(0, 1)``; optimized when ``None``.
        beta: Fixed trend parameter in ``(0, 1)``; optimized when ``None``.

    Returns:
        Fitted model; forecasts follow ``l_n + h b_n``.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the series is too short or a parameter is invalid.

    Equation:
        [EQ-TS-ES-002].

    References:
        [REF-HOLT-2004], [REF-BROCKWELL-DAVIS-2016, Chapter 10].

    """
    data = coerce_finite_sample(series, name="series")
    if len(data) < 4:
        raise ValueError("series must contain at least four observations")
    return _fit(data, _parameter(alpha, "alpha"), _parameter(beta, "beta"), holt=True)
