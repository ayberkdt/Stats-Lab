"""Relaxation of the analysis ensemble towards the prior (RTPP and RTPS).

An analysis shrinks the ensemble spread where observations are dense, and
with a small ensemble it shrinks it too much. Relaxation restores part of the
prior spread after the analysis, without touching the analysis mean:

* relaxation to prior perturbations (RTPP; Zhang, Snyder & Sun 2004)
  blends the anomalies, ``x'^a <- (1 - alpha) x'^a + alpha x'^f``
  [EQ-DA-RELAX-001];
* relaxation to prior spread (RTPS; Whitaker & Hamill 2012) rescales the
  anomalies of every state component so that its spread becomes
  ``(1 - alpha) sigma^a + alpha sigma^f`` [EQ-DA-RELAX-002].

Both are applied after the analysis and only at steps with an observation;
``alpha = 0`` leaves the analysis unchanged. The multiplicative ``inflation``
of :func:`~probstat.data_assimilation.ensemble_kalman_filter` acts on the
forecast instead, and the two can be combined.

References:
    [REF-ZHANG-SNYDER-SUN-2004], [REF-WHITAKER-HAMILL-2012].

"""

from __future__ import annotations

import math
from dataclasses import dataclass

from probstat.foundations.validation import require_finite_real
from probstat.linalg import Matrix

from .enkf import Ensemble, _mean

RELAXATION_METHODS = ("rtpp", "rtps")
"""Relaxation schemes: to prior perturbations and to prior spread."""


@dataclass(frozen=True, slots=True)
class Relaxation:
    """A relaxation scheme and its factor.

    Args:
        method: ``"rtpp"`` (to prior perturbations) or ``"rtps"`` (to prior
            spread).
        factor: Relaxation factor ``alpha`` in ``[0, 1]``.

    Raises:
        ValueError: If the method or the factor is invalid.

    References:
        [EQ-DA-RELAX-001], [EQ-DA-RELAX-002], [REF-ZHANG-SNYDER-SUN-2004],
        [REF-WHITAKER-HAMILL-2012].

    """

    method: str
    factor: float

    def __post_init__(self) -> None:
        """Validate the method and the factor.

        Raises:
            ValueError: If either is invalid.

        References:
            [REF-WHITAKER-HAMILL-2012]. No equation is evaluated.

        """
        if self.method not in RELAXATION_METHODS:
            raise ValueError(
                f"method must be one of {RELAXATION_METHODS}, got {self.method!r}"
            )
        value = _factor(self.factor)
        object.__setattr__(self, "factor", value)

    def apply(self, forecast: Ensemble, analysis: Ensemble) -> Ensemble:
        """Relax an analysis ensemble towards its forecast.

        Args:
            forecast: Forecast ensemble that entered the analysis.
            analysis: Analysis ensemble with members in the same order.

        Returns:
            The relaxed analysis ensemble.

        Raises:
            ValueError: If the ensembles differ in shape.

        Equation:
            [EQ-DA-RELAX-001], [EQ-DA-RELAX-002].

        References:
            [REF-ZHANG-SNYDER-SUN-2004], [REF-WHITAKER-HAMILL-2012].

        """
        if self.method == "rtpp":
            return relax_to_prior_perturbations(forecast, analysis, self.factor)
        return relax_to_prior_spread(forecast, analysis, self.factor)

    def compose(
        self,
        forecast: Ensemble,
        analysis: Ensemble,
        transforms: tuple[Matrix, ...],
    ) -> tuple[Matrix, ...]:
        """Fold the relaxation into ensemble transforms ``E^a_j = E^f_j T_j``.

        RTPP gives ``T 11'/N + (1 - alpha) T Q + alpha Q`` with
        ``Q = I - 11'/N``; RTPS gives ``T (11'/N + f_j Q)`` per component,
        with ``f_j`` the spread ratio of [EQ-DA-RELAX-002].

        Args:
            forecast: Forecast ensemble.
            analysis: Unrelaxed analysis ensemble.
            transforms: One shared or one per-component transform.

        Returns:
            Transforms of the relaxed analysis.

        Equation:
            [EQ-DA-RELAX-001], [EQ-DA-RELAX-002], [EQ-DA-ENKS-001].

        References:
            [REF-ZHANG-SNYDER-SUN-2004], [REF-WHITAKER-HAMILL-2012],
            [REF-EVENSEN-2003].

        """
        count = forecast.size
        alpha = self.factor
        if self.method == "rtpp":
            return tuple(
                _blend(transform, 1.0 - alpha, alpha, count) for transform in transforms
            )
        factors = _spread_factors(forecast, analysis, alpha)
        return tuple(
            _blend(
                transforms[0] if len(transforms) == 1 else transforms[j],
                factor,
                0.0,
                count,
            )
            for j, factor in enumerate(factors)
        )


def _blend(transform: Matrix, scale: float, prior: float, count: int) -> Matrix:
    """Return ``T 11'/N + scale T Q + prior Q`` with ``Q = I - 11'/N``.

    References:
        [EQ-DA-RELAX-001], [EQ-DA-ENKS-001].

    """
    row_means = [math.fsum(row) / count for row in transform]
    return tuple(
        tuple(
            row_means[a]
            + scale * (transform[a][i] - row_means[a])
            + prior * ((1.0 if a == i else 0.0) - 1.0 / count)
            for i in range(count)
        )
        for a in range(count)
    )


def _factor(value: float) -> float:
    """Validate a relaxation factor in ``[0, 1]``.

    Raises:
        ValueError: If the factor is outside ``[0, 1]``.

    References:
        [REF-WHITAKER-HAMILL-2012]. No equation is evaluated.

    """
    alpha = require_finite_real(value, name="factor")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"relaxation factor must lie in [0, 1], got {value!r}")
    return alpha


def _check_pair(forecast: Ensemble, analysis: Ensemble) -> None:
    """Require two ensembles of the same size and dimension.

    Raises:
        ValueError: If they differ.

    References:
        [REF-WHITAKER-HAMILL-2012]. No equation is evaluated.

    """
    if forecast.size != analysis.size or len(forecast.members[0]) != len(
        analysis.members[0]
    ):
        raise ValueError("forecast and analysis ensembles must have the same shape")


def relax_to_prior_perturbations(
    forecast: Ensemble, analysis: Ensemble, factor: float
) -> Ensemble:
    """Blend analysis and forecast anomalies (RTPP).

    ``x_i^a <- xbar^a + (1 - alpha)(x_i^a - xbar^a) + alpha (x_i^f - xbar^f)``
    [EQ-DA-RELAX-001]; the analysis mean is unchanged. Members are paired by
    position, so the analysis must keep the member order of the forecast
    (every analysis of this package does).

    Args:
        forecast: Forecast ensemble that entered the analysis.
        analysis: Analysis ensemble.
        factor: ``alpha`` in ``[0, 1]``.

    Returns:
        The relaxed analysis ensemble.

    Raises:
        ValueError: If the factor is outside ``[0, 1]`` or the ensembles
            differ in shape.

    Equation:
        [EQ-DA-RELAX-001].

    References:
        [REF-ZHANG-SNYDER-SUN-2004].

    """
    alpha = _factor(factor)
    _check_pair(forecast, analysis)
    if alpha == 0.0:
        return analysis
    prior = _mean(forecast.members)
    center = _mean(analysis.members)
    return Ensemble._trusted(
        tuple(
            tuple(
                m + (1.0 - alpha) * (a - m) + alpha * (f - p)
                for a, m, f, p in zip(member, center, previous, prior, strict=True)
            )
            for member, previous in zip(analysis.members, forecast.members, strict=True)
        )
    )


def _spread_factors(
    forecast: Ensemble, analysis: Ensemble, alpha: float
) -> list[float]:
    """Return the RTPS factors ``alpha (sigma^f - sigma^a) / sigma^a + 1``.

    A component with zero analysis spread has zero anomalies, which no
    factor changes; its factor is reported as one.

    References:
        [EQ-DA-RELAX-002], [REF-WHITAKER-HAMILL-2012].

    """
    count = forecast.size
    prior = _mean(forecast.members)
    center = _mean(analysis.members)
    factors = []
    for j in range(len(prior)):
        sigma_f = math.sqrt(
            math.fsum((member[j] - prior[j]) ** 2 for member in forecast.members)
            / (count - 1)
        )
        sigma_a = math.sqrt(
            math.fsum((member[j] - center[j]) ** 2 for member in analysis.members)
            / (count - 1)
        )
        factors.append(
            1.0 if sigma_a == 0.0 else alpha * (sigma_f - sigma_a) / sigma_a + 1.0
        )
    return factors


def relax_to_prior_spread(
    forecast: Ensemble, analysis: Ensemble, factor: float
) -> Ensemble:
    """Rescale the analysis anomalies towards the prior spread (RTPS).

    For every state component ``j`` with forecast and analysis spreads
    ``sigma^f_j`` and ``sigma^a_j`` (divisor ``N - 1``), the analysis
    anomalies are multiplied by ``alpha (sigma^f_j - sigma^a_j) / sigma^a_j
    + 1``, so the spread becomes ``(1 - alpha) sigma^a_j + alpha sigma^f_j``
    [EQ-DA-RELAX-002]; the mean is unchanged. A component with zero
    analysis spread stays as it is.

    Args:
        forecast: Forecast ensemble that entered the analysis.
        analysis: Analysis ensemble.
        factor: ``alpha`` in ``[0, 1]``.

    Returns:
        The relaxed analysis ensemble.

    Raises:
        ValueError: If the factor is outside ``[0, 1]`` or the ensembles
            differ in shape.

    Equation:
        [EQ-DA-RELAX-002].

    References:
        [REF-WHITAKER-HAMILL-2012].

    """
    alpha = _factor(factor)
    _check_pair(forecast, analysis)
    if alpha == 0.0:
        return analysis
    factors = _spread_factors(forecast, analysis, alpha)
    center = _mean(analysis.members)
    return Ensemble._trusted(
        tuple(
            tuple(
                m + f * (a - m) for a, m, f in zip(member, center, factors, strict=True)
            )
            for member in analysis.members
        )
    )
