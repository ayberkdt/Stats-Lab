"""Ledoit-Wolf linear shrinkage of a sample covariance matrix.

When the dimension ``p`` is comparable to the sample size ``n`` the sample
covariance matrix is ill-conditioned (singular for ``p > n``): its large
eigenvalues are too large and its small ones too small. Ledoit and Wolf
shrink it towards a multiple of the identity with an intensity estimated
from the data, which yields a positive definite matrix with a smaller
expected Frobenius loss in their large-dimensional asymptotic framework.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from probstat.linalg import Matrix, Vector, as_matrix


@dataclass(frozen=True, slots=True)
class LedoitWolfResult:
    """Shrinkage covariance estimate with the quantities that define it.

    All squared norms are ``||A||^2 = tr(A A') / p``.

    Args:
        covariance: ``shrinkage * target_scale * I + (1 - shrinkage) * S``.
        shrinkage: Intensity ``b^2 / d^2`` in ``[0, 1]`` (0 when ``d^2 = 0``,
            where ``S`` already equals its target).
        target_scale: ``m = tr(S) / p``, the mean sample variance.
        dispersion: ``d^2 = ||S - m I||^2``.
        sample_error: ``b^2 = min(bbar^2, d^2)``, the estimated squared error
            of ``S``.
        raw_sample_error: ``bbar^2 = (1/n^2) sum_k ||x_k x_k' - S||^2``.
        sample_covariance: ``S`` with divisor ``n``.
        location: The column means that were subtracted (zeros when
            ``center=False``).
        observations: Sample size ``n``.
        dimension: Number of variables ``p``.

    References:
        [EQ-STAT-LW-001], [REF-LEDOIT-WOLF-2004].

    """

    covariance: Matrix
    shrinkage: float
    target_scale: float
    dispersion: float
    sample_error: float
    raw_sample_error: float
    sample_covariance: Matrix
    location: Vector
    observations: int
    dimension: int


def ledoit_wolf_covariance(
    data: Iterable[Iterable[object]], *, center: bool = True
) -> LedoitWolfResult:
    """Return the Ledoit-Wolf shrinkage estimate of a covariance matrix.

    With rows ``x_k`` (after centring), ``S = (1/n) sum_k x_k x_k'``, the
    norm ``||A||^2 = tr(A A')/p``, ``m = tr(S)/p``, ``d^2 = ||S - m I||^2``,
    ``bbar^2 = (1/n^2) sum_k ||x_k x_k' - S||^2`` and
    ``b^2 = min(bbar^2, d^2)``, the estimate is
    ``(b^2/d^2) m I + (1 - b^2/d^2) S``. Its eigenvalues are those of ``S``
    pulled towards their mean ``m`` by the factor ``1 - b^2/d^2``, so it is
    positive definite whenever the intensity and ``m`` are positive, also
    for ``p > n``.

    Conventions: ``S`` divides by ``n``, not ``n - 1``. The source treats
    observations with known mean zero (``center=False`` reproduces it
    exactly); with ``center=True`` (default) the column means are
    subtracted first and the same formulas are applied to the centred rows,
    which is the usual practice but leaves a relative effect of order
    ``1/n`` that the source does not analyse. The optimality statement of
    the source is asymptotic (``p/n`` bounded, finite fourth moments,
    independent observations) and concerns the expected Frobenius loss among
    linear combinations of ``I`` and ``S``; nothing is claimed about the
    inverse or the eigenvectors.

    The sum defining ``bbar^2`` is evaluated as
    ``(sum_k |x_k|^4 - n tr(S^2)) / (p n^2)``, which equals the definition
    because ``sum_k x_k x_k' = n S``.

    Args:
        data: ``n`` rows of ``p`` finite observations each.
        center: Subtract the column means (default) or treat the mean as
            known to be zero.

    Returns:
        The estimate, the intensity, and the intermediate quantities.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If the data are empty, ragged, or non-finite, or if
            ``center=True`` with fewer than two rows.

    Equation:
        [EQ-STAT-LW-001].

    References:
        [REF-LEDOIT-WOLF-2004].

    """
    rows = as_matrix(data, name="data")
    n, p = len(rows), len(rows[0])
    if center:
        if n < 2:
            raise ValueError("at least two observations are required to centre")
        location = tuple(math.fsum(row[j] for row in rows) / n for j in range(p))
        rows = tuple(tuple(row[j] - location[j] for j in range(p)) for row in rows)
    else:
        location = (0.0,) * p
    sample = [[0.0] * p for _ in range(p)]
    for i in range(p):
        for j in range(i, p):
            value = math.fsum(row[i] * row[j] for row in rows) / n
            sample[i][j] = value
            sample[j][i] = value
    m = math.fsum(sample[i][i] for i in range(p)) / p
    trace_square = math.fsum(sample[i][j] ** 2 for i in range(p) for j in range(p))
    d2 = (
        math.fsum(
            (sample[i][j] - (m if i == j else 0.0)) ** 2
            for i in range(p)
            for j in range(p)
        )
        / p
    )
    fourth = math.fsum(math.fsum(x * x for x in row) ** 2 for row in rows)
    bbar2 = max((fourth - n * trace_square) / (p * n * n), 0.0)
    b2 = min(bbar2, d2)
    intensity = b2 / d2 if d2 > 0.0 else 0.0
    covariance = tuple(
        tuple(
            (1.0 - intensity) * sample[i][j] + (intensity * m if i == j else 0.0)
            for j in range(p)
        )
        for i in range(p)
    )
    return LedoitWolfResult(
        covariance=covariance,
        shrinkage=intensity,
        target_scale=m,
        dispersion=d2,
        sample_error=b2,
        raw_sample_error=bbar2,
        sample_covariance=tuple(tuple(row) for row in sample),
        location=location,
        observations=n,
        dimension=p,
    )
