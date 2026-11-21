"""CV+ prediction intervals: the K-fold version of jackknife+.

Jackknife+ refits the model ``n`` times. CV+ (Barber, Candes, Ramdas and
Tibshirani 2021) splits the training points into ``K`` disjoint
folds, fits ``K`` models ``mu_{-S_k}`` each without one fold, and scores
point ``i`` with the model that did not see it:
``R_i = |y_i - mu_{-S_k(i)}(x_i)|``. The interval at ``x`` is

- lower: the ``floor(alpha (n + 1))``-th smallest of
  ``mu_{-S_k(i)}(x) - R_i``,
- upper: the ``ceil((1 - alpha)(n + 1))``-th smallest of
  ``mu_{-S_k(i)}(x) + R_i``,

with ``i`` running over all ``n`` training points. With ``K = n`` it is
jackknife+. For exchangeable data and a fitting rule symmetric in its
training points, the marginal coverage is at least ``1 - 2 alpha`` minus a
slack that depends on ``n`` and ``K`` and is given in the source (stated
there for folds of equal size); the slack is zero for ``K = n``. Typical
coverage is near ``1 - alpha``, but that is an empirical observation, not a
guarantee.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_non_negative_integer
from probstat.linalg import Vector, as_vector

from .quantile import conformal_rank, lower_conformal_rank
from .regression import (
    FitFunction,
    PointPredictor,
    PredictionInterval,
    _calibration,
    _evaluate,
    as_predictor,
)


@dataclass(frozen=True)
class CVPlusRegressor:
    """CV+ intervals from ``K`` fold predictors and ``n`` out-of-fold residuals.

    Args:
        fold_predictors: ``mu_{-S_k}``, the model fitted without fold ``k``.
        fold_of: Fold index ``k(i)`` of every training point.
        residuals: ``R_i = |y_i - mu_{-S_k(i)}(x_i)|``.
        alpha: Miscoverage level.

    References:
        [EQ-CONF-012], [REF-BARBER-ETAL-2021].

    """

    fold_predictors: tuple[PointPredictor, ...]
    fold_of: tuple[int, ...]
    residuals: Vector
    alpha: float

    @property
    def size(self) -> int:
        """Return the number ``n`` of training points.

        References:
            [REF-BARBER-ETAL-2021]. No equation is evaluated.

        """
        return len(self.residuals)

    @property
    def folds(self) -> int:
        """Return the number ``K`` of folds (model fits).

        References:
            [REF-BARBER-ETAL-2021]. No equation is evaluated.

        """
        return len(self.fold_predictors)

    @property
    def model_fits(self) -> int:
        """Return the number of model fits spent, equal to ``K``.

        References:
            [REF-BARBER-ETAL-2021]. No equation is evaluated.

        """
        return len(self.fold_predictors)

    def interval(self, x: Sequence[float]) -> PredictionInterval:
        """Return the CV+ interval at one input.

        Args:
            x: Input vector.

        Returns:
            Interval whose ``prediction`` is the median over training points
            of the out-of-fold model predictions at ``x``.

        Equation:
            [EQ-CONF-012].

        References:
            [REF-BARBER-ETAL-2021].

        """
        point = as_vector(x, name="x")
        by_fold = [
            _evaluate(model, point, "fold prediction") for model in self.fold_predictors
        ]
        predictions = [by_fold[k] for k in self.fold_of]
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
        """Return CV+ intervals at many inputs.

        Args:
            xs: Input vectors.

        Returns:
            One interval per input.

        Equation:
            [EQ-CONF-012].

        References:
            [REF-BARBER-ETAL-2021].

        """
        return tuple(self.interval(x) for x in xs)


def cv_plus_regression(
    fit: FitFunction,
    x_train: Sequence[Sequence[float]],
    y_train: Sequence[float],
    *,
    folds: int,
    alpha: float = 0.1,
    seed: int | RandomStream | None = None,
) -> CVPlusRegressor:
    """Build CV+ intervals from ``K`` model fits on K-fold splits.

    Fold sizes differ by at most one. Without ``seed`` the folds are
    contiguous blocks in the given order (``fold(i) = floor(i K / n)``),
    which is deterministic and appropriate when the rows are already in
    exchangeable order; with ``seed`` the balanced fold labels are shuffled
    with the child stream ``"cv-plus-folds"``. Either way the assignment
    does not depend on the data values.

    For exchangeable data and a symmetric fitting rule the marginal coverage
    is at least ``1 - 2 alpha`` minus a slack depending on ``n`` and ``K``
    given in the source, where it is stated for equal-size folds; the slack
    vanishes for ``folds = n``, which reproduces
    :func:`~probstat.conformal.jackknife_plus_regression`.

    Args:
        fit: Function ``(points, values) -> predictor``, symmetric in the
            training points.
        x_train: Training input vectors.
        y_train: Training responses.
        folds: Number ``K`` of folds, ``2 <= K <= n``.
        alpha: Miscoverage level in ``(0, 1)``.
        seed: Optional seed or stream used to shuffle the fold labels.

    Returns:
        Regressor holding the ``K`` fold predictors.

    Raises:
        TypeError: If ``fit`` is not callable or returns a non-predictor, or
            ``folds`` or ``seed`` has the wrong type.
        ValueError: If ``folds`` is outside ``2..n`` or inputs mismatch.

    Equation:
        [EQ-CONF-012].

    References:
        [REF-BARBER-ETAL-2021].

    """
    if not callable(fit):
        raise TypeError("fit must be callable")
    design, responses = _calibration(x_train, y_train)
    n = len(design)
    count = require_non_negative_integer(folds, name="folds")
    if not 2 <= count <= n:
        raise ValueError("folds must satisfy 2 <= folds <= number of points")
    lower_conformal_rank(n, alpha)
    labels = [i * count // n for i in range(n)]
    if seed is not None:
        resolve_stream(seed).child("cv-plus-folds").generator().shuffle(labels)
    models: list[PointPredictor] = []
    residuals = [0.0] * n
    for k in range(count):
        keep = [i for i in range(n) if labels[i] != k]
        model = as_predictor(
            fit(tuple(design[i] for i in keep), tuple(responses[i] for i in keep))
        )
        models.append(model)
        for i in range(n):
            if labels[i] == k:
                residuals[i] = abs(
                    responses[i] - _evaluate(model, design[i], "prediction")
                )
    return CVPlusRegressor(tuple(models), tuple(labels), tuple(residuals), float(alpha))
