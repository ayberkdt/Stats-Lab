"""Conformal error bars for surrogate (emulator) predictions of a model QoI.

A fitted PCE or GP replaces an expensive forward model, but its own
approximation error is usually judged only by a leave-one-out score. Here the
true model is run at ``n`` fresh inputs drawn independently from the model's
joint input law, and split conformal calibration turns the surrogate's
residuals into intervals ``[y_hat(x) - q, y_hat(x) + q]`` that contain the true
model output with probability at least ``1 - alpha`` for a new input drawn
from the same law.

Calibration inputs are plain Monte Carlo draws on purpose: Latin hypercube or
low-discrepancy designs are not exchangeable with a random test input, so the
finite-sample guarantee would not apply. The calibration points must also be
independent of the surrogate's training design.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.linalg import Vector
from probstat.modeling import ProbabilisticModel
from probstat.surrogates import Surrogate

from .regression import (
    PredictionInterval,
    SplitConformalRegressor,
    split_conformal_regression,
)


@runtime_checkable
class VarianceSurrogate(Protocol):
    """A surrogate that also reports a predictive variance (e.g. a GP).

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Chapter 2].

    """

    def predict(self, x: Sequence[float]) -> float:
        """Return the predictive mean at ``x``.

        References:
            [REF-RASMUSSEN-WILLIAMS-2006, Chapter 2]. No equation is evaluated.

        """
        ...

    def predict_with_variance(
        self, x: Sequence[float], *, include_noise: bool = False
    ) -> tuple[float, float]:
        """Return the predictive mean and variance at ``x``.

        References:
            [REF-RASMUSSEN-WILLIAMS-2006, Chapter 2]. No equation is evaluated.

        """
        ...


@dataclass(frozen=True)
class SurrogateConformalResult:
    """Conformal calibration of a surrogate against its true model.

    Args:
        quantity: QoI name.
        regressor: Split conformal regressor around the surrogate.
        calibration_points: Physical inputs where the true model was run.
        calibration_values: True model outputs at those inputs.
        model_evaluations: True-model calls spent on calibration.

    References:
        [EQ-CONF-002], [REF-LEI-ETAL-2018], [REF-SUDRET-2008].

    """

    quantity: str
    regressor: SplitConformalRegressor
    calibration_points: tuple[Vector, ...]
    calibration_values: Vector
    model_evaluations: int

    @property
    def half_width(self) -> float:
        """Return ``q_hat`` (absolute score) or the multiplier of ``s(x)``.

        References:
            [EQ-CONF-002], [REF-LEI-ETAL-2018].

        """
        return self.regressor.quantile

    @property
    def guaranteed_coverage(self) -> float:
        """Return ``k / (n + 1)``, the marginal coverage guarantee.

        References:
            [EQ-CONF-005], [REF-LEI-ETAL-2018].

        """
        return self.regressor.guaranteed_coverage

    def interval(self, x: Sequence[float]) -> PredictionInterval:
        """Return the conformal interval for the true QoI at ``x``.

        Args:
            x: Physical input vector in model order.

        Returns:
            Interval around the surrogate prediction.

        Equation:
            [EQ-CONF-002].

        References:
            [REF-LEI-ETAL-2018].

        """
        return self.regressor.interval(x)


def conformalize_surrogate(
    model: ProbabilisticModel,
    surrogate: Surrogate,
    *,
    quantity: str | None = None,
    alpha: float = 0.1,
    size: int = 200,
    normalized: bool = False,
    scale_offset: float = 0.0,
    seed: int | RandomStream | None = None,
) -> SurrogateConformalResult:
    """Calibrate conformal error bars for a surrogate of one model QoI.

    Draws ``size`` independent inputs from ``model.inputs`` (stream child
    ``"conformal-calibration"``), evaluates the true model there, and
    calibrates split conformal intervals around ``surrogate.predict``. With
    ``normalized=True`` the surrogate must expose ``predict_with_variance``
    (a GP), and scores are ``|y - mu(x)| / (sd(x) + scale_offset)``, so
    intervals widen where the GP is uncertain. A GP that interpolates very
    smooth data can report ``sd(x) = 0`` through cancellation; a positive
    ``scale_offset`` (in output units, chosen before seeing the calibration
    data) keeps the scores finite. Coverage holds for any fixed offset.

    Args:
        model: Probabilistic model whose QoI the surrogate approximates.
        surrogate: Fitted surrogate with ``predict(x)``.
        quantity: QoI name; defaults to the first declared quantity.
        alpha: Miscoverage level in ``(0, 1)``.
        size: Number of true-model calibration runs (at least one).
        normalized: Use GP-standard-deviation normalized scores.
        scale_offset: Non-negative constant added to the GP standard
            deviation in normalized scores.
        seed: Integer seed or stream.

    Returns:
        Calibrated result; ``model.evaluations`` grows by ``size``.

    Raises:
        TypeError: If ``surrogate`` lacks the required methods.
        KeyError: If ``quantity`` is unknown.
        ValueError: If ``size``, ``alpha``, or ``scale_offset`` is invalid,
            or a normalized score has a zero denominator (raise
            ``scale_offset``).

    Equation:
        [EQ-CONF-001], [EQ-CONF-002].

    References:
        [REF-LEI-ETAL-2018], [REF-ANGELOPOULOS-BATES-2023],
        [REF-SUDRET-2008], [REF-RASMUSSEN-WILLIAMS-2006, Chapter 2].

    """
    if not isinstance(surrogate, Surrogate):
        raise TypeError("surrogate must provide predict(x)")
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("size must be a positive integer")
    name = model.quantity_names[0] if quantity is None else quantity
    model.quantity(name)
    offset = require_finite_real(scale_offset, name="scale_offset")
    if offset < 0.0:
        raise ValueError("scale_offset must be non-negative")
    scale = None
    if normalized:
        if not isinstance(surrogate, VarianceSurrogate):
            raise TypeError("normalized scores need predict_with_variance(x)")
        variance_model = surrogate

        def scale(x: Sequence[float]) -> float:
            """Return the predictive standard deviation plus the offset.

            References:
                [REF-LEI-ETAL-2018], [REF-RASMUSSEN-WILLIAMS-2006, Chapter 2].

            """
            variance = variance_model.predict_with_variance(x)[1]
            return math.sqrt(max(variance, 0.0)) + offset

    stream = resolve_stream(seed).child("conformal-calibration")
    points = model.inputs.sample(size, seed=stream)
    before = model.evaluations
    values = model.evaluate_many(points)[name]
    try:
        regressor = split_conformal_regression(
            surrogate, points, values, alpha=alpha, scale=scale
        )
    except ValueError as error:
        if normalized and "scale(x)" in str(error):
            raise ValueError(
                "the GP predictive standard deviation vanished at a calibration "
                "point; pass a positive scale_offset"
            ) from error
        raise
    return SurrogateConformalResult(
        name, regressor, points, tuple(values), model.evaluations - before
    )
