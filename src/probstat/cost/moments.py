"""Exact moments of a weighted sum of correlated element costs.

``C = c_0 + sum_i a_i X_i`` has mean ``c_0 + sum a_i mu_i`` and variance
``sum_i sum_j a_i a_j rho_ij sigma_i sigma_j`` for *any* element laws with
Pearson correlations ``rho_ij``. These two numbers are exact; only the shape
assumed for the total in :mod:`.cost_risk` is an approximation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from probstat.foundations import require_finite_real


def _correlation(
    values: Sequence[Sequence[float]] | None, size: int
) -> tuple[tuple[float, ...], ...]:
    """Validate a Pearson correlation matrix (identity when ``None``).

    Args:
        values: Candidate square matrix or ``None``.
        size: Required dimension.

    Returns:
        Symmetric matrix with unit diagonal and entries in ``[-1, 1]``.

    Raises:
        ValueError: If the shape, diagonal, symmetry, or range is invalid.

    References:
        [REF-ROSS-2019, Chapter 7]. No equation is evaluated.

    """
    if values is None:
        return tuple(
            tuple(1.0 if i == j else 0.0 for j in range(size)) for i in range(size)
        )
    rows = tuple(
        tuple(require_finite_real(v, name="correlation") for v in row) for row in values
    )
    if len(rows) != size or any(len(row) != size for row in rows):
        raise ValueError(f"correlation must be {size} x {size}")
    for i in range(size):
        if abs(rows[i][i] - 1.0) > 1e-12:
            raise ValueError("correlation diagonal must be one")
        for j in range(size):
            if abs(rows[i][j] - rows[j][i]) > 1e-12:
                raise ValueError("correlation must be symmetric")
            if not -1.0 <= rows[i][j] <= 1.0:
                raise ValueError("correlations must lie in [-1, 1]")
    return rows


@dataclass(frozen=True, slots=True)
class MomentTotal:
    """Exact first two moments of a weighted sum of element costs.

    Args:
        mean: ``c_0 + sum a_i mu_i``.
        variance: ``sum_ij a_i a_j rho_ij sigma_i sigma_j``.
        standard_deviation: Square root of ``variance``.
        independent_variance: ``sum a_i^2 sigma_i^2`` (all ``rho_ij = 0``).
        independent_standard_deviation: Its square root.

    References:
        [REF-GARVEY-2000], [REF-NASA-CEH-2015].

    """

    mean: float
    variance: float
    standard_deviation: float
    independent_variance: float
    independent_standard_deviation: float

    @property
    def spread_understatement(self) -> float:
        """Return ``1 - sd_independent / sd``: the fraction of the spread lost.

        Positive when positive correlations are ignored, zero without
        correlation, negative when correlations are negative.

        Equation:
            [EQ-COST-MOM-001].

        References:
            [REF-GARVEY-2000].

        """
        if self.standard_deviation == 0.0:
            return 0.0
        return 1.0 - self.independent_standard_deviation / self.standard_deviation


def method_of_moments_total(
    means: Sequence[float],
    standard_deviations: Sequence[float],
    correlation: Sequence[Sequence[float]] | None = None,
    *,
    weights: Sequence[float] | None = None,
    constant: float = 0.0,
) -> MomentTotal:
    """Return the exact mean and variance of ``c_0 + sum a_i X_i``.

    Args:
        means: Element means ``mu_i``.
        standard_deviations: Non-negative element standard deviations.
        correlation: Pearson correlation matrix; ``None`` is independence.
        weights: Coefficients ``a_i`` (default 1).
        constant: Deterministic part ``c_0``.

    Returns:
        Moments of the total and of its independent counterpart.

    Raises:
        ValueError: If lengths disagree, a deviation is negative, or the
            correlation matrix yields a negative variance (not positive
            semi-definite).

    Equation:
        [EQ-COST-MOM-001].

    References:
        [REF-GARVEY-2000], [REF-NASA-CEH-2015].

    """
    mu = tuple(require_finite_real(v, name="mean") for v in means)
    sd = tuple(
        require_finite_real(v, name="standard deviation") for v in standard_deviations
    )
    if len(mu) != len(sd) or not mu:
        raise ValueError("means and standard_deviations must be non-empty and aligned")
    if any(s < 0.0 for s in sd):
        raise ValueError("standard deviations must be non-negative")
    a = (
        tuple(1.0 for _ in mu)
        if weights is None
        else tuple(require_finite_real(v, name="weight") for v in weights)
    )
    if len(a) != len(mu):
        raise ValueError("weights must have one entry per element")
    rho = _correlation(correlation, len(mu))
    scaled = [a[i] * sd[i] for i in range(len(mu))]
    variance = math.fsum(
        scaled[i] * scaled[j] * rho[i][j]
        for i in range(len(mu))
        for j in range(len(mu))
    )
    independent = math.fsum(s * s for s in scaled)
    if variance < -1e-12 * max(1.0, independent):
        raise ValueError("correlation matrix is not positive semi-definite")
    variance = max(variance, 0.0)
    mean = require_finite_real(constant, name="constant") + math.fsum(
        a[i] * mu[i] for i in range(len(mu))
    )
    return MomentTotal(
        mean, variance, math.sqrt(variance), independent, math.sqrt(independent)
    )
