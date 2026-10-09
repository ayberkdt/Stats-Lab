"""Least-squares and simple linear-regression estimators."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from probstat.foundations import coerce_finite_sample


@dataclass(frozen=True, slots=True)
class LinearRegressionResult:
    """Immutable diagnostics for an intercept-and-slope least-squares fit.

    References:
        [REF-MONTGOMERY-PECK-VINING-2021, Chapters 1-2].

    """

    intercept: float
    slope: float
    residuals: tuple[float, ...]
    residual_sum_squares: float
    total_sum_squares: float
    residual_variance: float
    r_squared: float

    def predict(self, predictors: Iterable[object]) -> tuple[float, ...]:
        """Evaluate the fitted conditional-mean line at finite predictors.

        Args:
            predictors: Finite real predictor values.

        Returns:
            Fitted values in input order.

        Equation:
            [EQ-REG-LINE-001].

        References:
            [REF-MONTGOMERY-PECK-VINING-2021, Chapter 1].

        """
        checked = coerce_finite_sample(predictors, name="predictors")
        return tuple(self.intercept + self.slope * value for value in checked)


def fit_simple_linear_regression(
    predictors: Iterable[object], observations: Iterable[object]
) -> LinearRegressionResult:
    """Fit ``y = intercept + slope*x + error`` by ordinary least squares.

    Args:
        predictors: At least three finite real predictor values with variation.
        observations: Finite real responses paired with the predictors.

    Returns:
        Coefficients, residuals, sums of squares, unbiased residual variance,
        and coefficient of determination.

    Raises:
        TypeError: If an item is not real.
        ValueError: If lengths differ, fewer than three pairs are supplied, or
            all predictors are equal.

    Equation:
        [EQ-REG-OLS-001], [EQ-REG-DIAG-001].

    References:
        [REF-MONTGOMERY-PECK-VINING-2021, Chapters 1-2].

    """
    x_values = coerce_finite_sample(predictors, name="predictors")
    y_values = coerce_finite_sample(observations, name="observations")
    if len(x_values) != len(y_values):
        raise ValueError("predictors and observations must have equal length")
    if len(x_values) < 3:
        raise ValueError("linear regression requires at least three pairs")
    x_mean = math.fsum(x_values) / len(x_values)
    y_mean = math.fsum(y_values) / len(y_values)
    centered_sum = math.fsum((value - x_mean) ** 2 for value in x_values)
    if centered_sum == 0.0:
        raise ValueError("predictors must contain at least two distinct values")
    cross_sum = math.fsum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x_values, y_values, strict=True)
    )
    slope = cross_sum / centered_sum
    intercept = y_mean - slope * x_mean
    residuals = tuple(
        y_value - (intercept + slope * x_value)
        for x_value, y_value in zip(x_values, y_values, strict=True)
    )
    residual_sum_squares = math.fsum(value * value for value in residuals)
    total_sum_squares = math.fsum((value - y_mean) ** 2 for value in y_values)
    r_squared = (
        1.0
        if total_sum_squares == 0.0
        else 1.0 - residual_sum_squares / total_sum_squares
    )
    return LinearRegressionResult(
        intercept=intercept,
        slope=slope,
        residuals=residuals,
        residual_sum_squares=residual_sum_squares,
        total_sum_squares=total_sum_squares,
        residual_variance=residual_sum_squares / (len(x_values) - 2),
        r_squared=r_squared,
    )
