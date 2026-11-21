"""Prediction-powered inference for a population mean.

A small sample with trusted labels ``Y`` is combined with a large sample on
which only the predictions ``f(X)`` of a machine-learning model are
available. The predictions are never trusted: the labelled sample measures
their bias (the rectifier), and the estimator stays unbiased for ``E[Y]``
whatever the quality of ``f``. Good predictions shorten the interval; bad
ones cannot bias it and, with the tuned weight, cannot lengthen it beyond
the classical interval in estimated variance.

The intervals are large-sample normal approximations. They assume that the
labelled pairs and the unlabelled covariates are independent draws from one
population and that the predictor was fitted on data independent of both
samples.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from statistics import NormalDist

from probstat.foundations import coerce_finite_sample
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_finite_real,
)

from ._hypothesis import check_level
from .confidence import ConfidenceInterval


def _positive_size(value: object, *, name: str) -> int:
    """Validate a positive sample size.

    Args:
        value: Candidate size.
        name: Parameter name used in error messages.

    Returns:
        The size as an integer of at least one.

    Raises:
        ValueError: If ``value`` is not a positive integer.

    References:
        [EQ-INF-PPI-002]. No equation is evaluated.

    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def prediction_powered_variance(
    *,
    label_variance: float,
    prediction_variance: float,
    covariance: float,
    labelled_size: int,
    unlabelled_size: int,
    weight: float,
) -> float:
    """Return the exact variance of the prediction-powered mean estimator.

    For independent samples of ``n`` labelled pairs and ``N`` unlabelled
    predictions, ``theta_hat = mean_n(Y) + w (mean_N(f) - mean_n(f))`` is the
    sum of the independent terms ``mean_n(Y - w f)`` and ``w mean_N(f)``, so

    ``Var(theta_hat) = (Var(Y) - 2 w Cov(Y, f) + w^2 Var(f)) / n
    + w^2 Var(f) / N``.

    The formula is exact for every fixed weight ``w`` and every law with
    finite second moments; no normality is involved.

    Args:
        label_variance: ``Var(Y) >= 0``.
        prediction_variance: ``Var(f(X)) >= 0``.
        covariance: ``Cov(Y, f(X))`` with
            ``covariance^2 <= label_variance * prediction_variance``.
        labelled_size: Number ``n >= 1`` of labelled pairs.
        unlabelled_size: Number ``N >= 1`` of unlabelled predictions.
        weight: Fixed weight ``w`` (``lambda`` in the literature).

    Returns:
        The variance of the estimator.

    Raises:
        TypeError: If an argument is not real.
        ValueError: If a variance is negative, the covariance violates the
            Cauchy-Schwarz bound, or a size is not a positive integer.

    Equation:
        [EQ-INF-PPI-002].

    References:
        [REF-ANGELOPOULOS-DUCHI-ZRNIC-2023], [REF-ANGELOPOULOS-ETAL-2023].

    """
    var_y = require_non_negative_finite_real(label_variance, name="label_variance")
    var_f = require_non_negative_finite_real(
        prediction_variance, name="prediction_variance"
    )
    cov = require_finite_real(covariance, name="covariance")
    if cov * cov > var_y * var_f * (1.0 + 1.0e-12):
        raise ValueError("covariance exceeds the Cauchy-Schwarz bound")
    n = _positive_size(labelled_size, name="labelled_size")
    big_n = _positive_size(unlabelled_size, name="unlabelled_size")
    w = require_finite_real(weight, name="weight")
    residual = max(0.0, var_y - 2.0 * w * cov + w * w * var_f)
    return residual / n + w * w * var_f / big_n


def prediction_powered_weight(
    *,
    prediction_variance: float,
    covariance: float,
    labelled_size: int,
    unlabelled_size: int,
) -> float:
    """Return the weight that minimizes the prediction-powered variance.

    The variance [EQ-INF-PPI-002] is a convex quadratic in ``w`` with
    derivative ``-2 Cov(Y, f)/n + 2 w Var(f) (1/n + 1/N)``, which vanishes at

    ``w* = Cov(Y, f) / ((1 + n/N) Var(f))``.

    ``w* = 0`` when the predictions are uncorrelated with the labels (the
    classical estimator), and ``w*`` tends to the regression slope of ``Y``
    on ``f`` as ``N / n`` grows. The minimum variance is
    ``Var(Y) (1 - rho^2 / (1 + n/N)) / n`` with ``rho`` the correlation.

    Args:
        prediction_variance: ``Var(f(X)) >= 0``.
        covariance: ``Cov(Y, f(X))``.
        labelled_size: Number ``n >= 1`` of labelled pairs.
        unlabelled_size: Number ``N >= 1`` of unlabelled predictions.

    Returns:
        The optimal weight; zero when the predictions are constant.

    Raises:
        TypeError: If an argument is not real.
        ValueError: If the variance is negative or a size is not a positive
            integer.

    Equation:
        [EQ-INF-PPI-003].

    References:
        [REF-ANGELOPOULOS-DUCHI-ZRNIC-2023].

    """
    var_f = require_non_negative_finite_real(
        prediction_variance, name="prediction_variance"
    )
    cov = require_finite_real(covariance, name="covariance")
    n = _positive_size(labelled_size, name="labelled_size")
    big_n = _positive_size(unlabelled_size, name="unlabelled_size")
    if var_f == 0.0:
        return 0.0
    return cov / ((1.0 + n / big_n) * var_f)


@dataclass(frozen=True, slots=True)
class PredictionPoweredMean:
    """Prediction-powered estimate of a mean with its classical counterpart.

    Args:
        estimate: ``mean_n(Y) + w (mean_N(f) - mean_n(f))``.
        standard_error: Square root of the estimated variance
            [EQ-INF-PPI-002] with sample variances inserted.
        interval: Normal-approximation confidence interval.
        weight: Weight ``w`` used (``lambda`` in the literature).
        tuned: Whether ``w`` was estimated from the data.
        classical_estimate: The labelled mean ``mean_n(Y)`` (``w = 0``).
        classical_standard_error: Its standard error ``s_Y / sqrt(n)``.
        rectifier: ``mean_n(Y - f)``, the estimated bias of the predictions
            with the sign that corrects them.
        variance_ratio: Estimated variance divided by the classical one;
            below one when the predictions help. NaN when the labels are
            constant.
        labelled_size: Number ``n`` of labelled pairs.
        unlabelled_size: Number ``N`` of unlabelled predictions.

    References:
        [EQ-INF-PPI-001], [EQ-INF-PPI-002], [REF-ANGELOPOULOS-ETAL-2023],
        [REF-ANGELOPOULOS-DUCHI-ZRNIC-2023].

    """

    estimate: float
    standard_error: float
    interval: ConfidenceInterval
    weight: float
    tuned: bool
    classical_estimate: float
    classical_standard_error: float
    rectifier: float
    variance_ratio: float
    labelled_size: int
    unlabelled_size: int


def _mean_and_centered(sample: tuple[float, ...]) -> tuple[float, tuple[float, ...]]:
    """Return the mean of a sample and its deviations from the mean.

    Args:
        sample: Non-empty finite observations.

    Returns:
        ``(mean, deviations)``.

    References:
        [EQ-INF-PPI-002]. Sample moments only.

    """
    mean = math.fsum(sample) / len(sample)
    return mean, tuple(value - mean for value in sample)


def prediction_powered_mean(
    labels: Iterable[object],
    labelled_predictions: Iterable[object],
    unlabelled_predictions: Iterable[object],
    *,
    weight: float | None = None,
    confidence: float = 0.95,
) -> PredictionPoweredMean:
    """Estimate a mean from few labels and many predictions.

    With labels ``Y_1..Y_n``, predictions ``f_i = f(X_i)`` on the same
    units, and predictions ``F_j = f(X~_j)`` on ``N`` further unlabelled
    units,

    ``theta_hat_w = mean_n(Y) + w (mean_N(F) - mean_n(f))``.

    ``E[theta_hat_w] = E[Y]`` for every fixed ``w`` because the two
    prediction means have the same expectation. ``w = 0`` is the classical
    labelled mean, ``w = 1`` the original prediction-powered estimator
    ``mean_N(F) + mean_n(Y - f)``. The variance is estimated by
    ``V(w) = s^2(Y - w f) / n + w^2 s^2(F) / N`` with sample variances
    (divisor ``n - 1`` and ``N - 1``), and the interval is
    ``theta_hat_w +- z_{(1 + confidence)/2} sqrt(V(w))``.

    With ``weight=None`` the weight is tuned as the exact minimizer of the
    estimated variance,
    ``w_hat = (s_Yf / n) / (s_f^2 / n + s_F^2 / N)``,
    which estimates ``Cov(Y, f) / ((1 + n/N) Var(f))``. By construction
    ``V(w_hat) <= V(w)`` for every ``w``, in particular for ``w = 0`` and
    ``w = 1``. ``w_hat`` is not clipped to ``[0, 1]``: predictions that are
    anti-correlated with the labels get a negative weight.

    Assumptions: the labelled pairs ``(X_i, Y_i)`` and the unlabelled
    ``X~_j`` are independent and identically distributed draws from one
    population (no covariate shift between the two samples), the two samples
    are independent of each other, ``f`` is fixed (trained on other data),
    and ``Y`` and ``f(X)`` have finite variances. The interval is asymptotic
    in ``n``: for small ``n`` it undercovers like any z interval with an
    estimated variance, and a data-tuned weight adds a bias of order ``1/n``
    that the normal approximation ignores.

    Args:
        labels: At least two labels ``Y_i``.
        labelled_predictions: Predictions ``f(X_i)`` for the same units, in
            the same order.
        unlabelled_predictions: At least two predictions on unlabelled
            units.
        weight: Fixed weight ``w``, or ``None`` to tune it from the data.
        confidence: Nominal two-sided confidence level in ``(0, 1)``.

    Returns:
        The estimate, its standard error and interval, the weight, and the
        classical estimate for comparison.

    Raises:
        TypeError: If a value is not real.
        ValueError: If a sample is too small, the labelled samples differ in
            length, or the confidence level is invalid.

    Equation:
        [EQ-INF-PPI-001], [EQ-INF-PPI-002], [EQ-INF-PPI-003].

    References:
        [REF-ANGELOPOULOS-ETAL-2023], [REF-ANGELOPOULOS-DUCHI-ZRNIC-2023].

    """
    y = coerce_finite_sample(labels, name="labels")
    f = coerce_finite_sample(labelled_predictions, name="labelled_predictions")
    big_f = coerce_finite_sample(unlabelled_predictions, name="unlabelled_predictions")
    if len(y) != len(f):
        raise ValueError("labels and labelled_predictions must have equal length")
    if len(y) < 2:
        raise ValueError("at least two labelled observations are required")
    if len(big_f) < 2:
        raise ValueError("at least two unlabelled predictions are required")
    level = check_level(confidence, name="confidence")
    n = len(y)
    big_n = len(big_f)
    mean_y, dev_y = _mean_and_centered(y)
    mean_f, dev_f = _mean_and_centered(f)
    mean_big, dev_big = _mean_and_centered(big_f)
    var_y = math.fsum(d * d for d in dev_y) / (n - 1)
    var_f = math.fsum(d * d for d in dev_f) / (n - 1)
    cov = math.fsum(a * b for a, b in zip(dev_y, dev_f, strict=True)) / (n - 1)
    var_big = math.fsum(d * d for d in dev_big) / (big_n - 1)
    if weight is None:
        denominator = var_f / n + var_big / big_n
        w = (cov / n) / denominator if denominator > 0.0 else 0.0
    else:
        w = require_finite_real(weight, name="weight")
    residual = max(0.0, var_y - 2.0 * w * cov + w * w * var_f)
    variance = residual / n + w * w * var_big / big_n
    classical_variance = var_y / n
    estimate = mean_y + w * (mean_big - mean_f)
    standard_error = math.sqrt(variance)
    half_width = NormalDist().inv_cdf(0.5 + 0.5 * level) * standard_error
    label = "tuned" if weight is None else "fixed"
    return PredictionPoweredMean(
        estimate=estimate,
        standard_error=standard_error,
        interval=ConfidenceInterval(
            estimate - half_width,
            estimate + half_width,
            level,
            f"prediction-powered normal approximation ({label} weight)",
        ),
        weight=w,
        tuned=weight is None,
        classical_estimate=mean_y,
        classical_standard_error=math.sqrt(classical_variance),
        rectifier=mean_y - mean_f,
        variance_ratio=(
            variance / classical_variance if classical_variance > 0.0 else math.nan
        ),
        labelled_size=n,
        unlabelled_size=big_n,
    )
