"""Deterministic square-root ensemble analyses: the ETKF and the serial EAKF.

A square-root filter updates the ensemble without perturbing the
observations. The analysis mean and the analysis sample covariance then equal
the Kalman update computed from the forecast ensemble mean and covariance
exactly (for a linear observation operator), instead of only in expectation
as for the stochastic EnKF.

* The ensemble transform Kalman filter (ETKF) of Bishop, Etherton & Majumdar
  (2001), in the symmetric-square-root form of Hunt, Kostelich & Szunyogh
  (2007) [EQ-DA-ETKF-001], computes the analysis in the ``N``-dimensional
  space spanned by the forecast anomalies:
  ``x_i^a = xbar^f + X^f (wbar^a + W^a_i)``.
* The ensemble adjustment Kalman filter (EAKF) of Anderson (2001), in the
  serial two-step form of Anderson (2003) [EQ-DA-EAKF-001], assimilates
  scalar observations with independent errors one at a time: it adjusts the
  observed ensemble deterministically in observation space and regresses
  the increments onto the state.

Both analyses also return the ensemble transforms ``E^a = E^f T`` (one
``N`` by ``N`` matrix), which the ensemble smoother applies to earlier
ensembles ([EQ-DA-ENKS-001]).

References:
    [REF-BISHOP-ETHERTON-MAJUMDAR-2001], [REF-HUNT-KOSTELICH-SZUNYOGH-2007],
    [REF-ANDERSON-2001], [REF-ANDERSON-JL-2003].

"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from probstat.linalg import (
    Matrix,
    Vector,
    as_vector,
    cholesky,
    symmetric_eigen,
    symmetrize,
)

from .enkf import Ensemble, _mean


@dataclass(frozen=True, slots=True)
class EnsembleTransformAnalysis:
    """Outcome of one deterministic (square-root) ensemble analysis.

    Args:
        ensemble: Analysis ensemble.
        innovation: ``y - mean(h(x^f))``.
        transforms: Ensemble transforms with ``E^a_j = E^f_j T_j`` for the
            row ``j`` of the member matrix ``E`` (state component ``j``
            over the members): one shared ``N`` by ``N`` matrix for a
            global analysis, one per state component for a local one.
        method: ``"etkf"``, ``"eakf"``, or ``"letkf"``.

    References:
        [EQ-DA-ETKF-001], [EQ-DA-EAKF-001], [EQ-DA-ENKS-001],
        [REF-HUNT-KOSTELICH-SZUNYOGH-2007].

    """

    ensemble: Ensemble
    innovation: Vector
    transforms: tuple[Matrix, ...]
    method: str

    def transform_for(self, component: int) -> Matrix:
        """Return the transform that acts on one state component.

        Args:
            component: State index.

        Returns:
            The ``N`` by ``N`` transform of that component.

        References:
            [EQ-DA-ENKS-001].

        """
        if len(self.transforms) == 1:
            return self.transforms[0]
        return self.transforms[component]


def _observation_images(
    ensemble: Ensemble,
    observation: Sequence[float],
    observation_function: Callable[[Vector], Sequence[float]],
    observation_covariance: Sequence[Sequence[float]],
) -> tuple[Vector, Matrix, tuple[Vector, ...], Vector]:
    """Validate the inputs of an analysis and map the members to observations.

    Args:
        ensemble: Forecast ensemble.
        observation: Observation ``y``.
        observation_function: Measurement map ``h``.
        observation_covariance: Positive-definite ``R``.

    Returns:
        ``y``, ``R``, the images ``h(x_i)``, and their mean.

    Raises:
        ValueError: If shapes disagree or ``R`` is not symmetric.

    References:
        [EQ-FILT-SSM-002], [REF-HUNT-KOSTELICH-SZUNYOGH-2007].

    """
    noise = symmetrize(observation_covariance, name="observation_covariance")
    y = as_vector(observation, name="observation")
    if len(noise) != len(y):
        raise ValueError("observation_covariance must match the observation length")
    images = tuple(
        as_vector(observation_function(member), name="h(x)")
        for member in ensemble.members
    )
    if any(len(image) != len(y) for image in images):
        raise ValueError("h(x) length must equal the observation length")
    return y, noise, images, _mean(images)


def _whiten(lower: Matrix, rows: Sequence[Sequence[float]]) -> list[list[float]]:
    """Solve ``L Z = A`` by forward substitution for every column of ``A``.

    Args:
        lower: Lower Cholesky factor ``L`` (``p`` by ``p``).
        rows: ``A`` as ``p`` rows.

    Returns:
        ``Z = L^-1 A`` as rows.

    References:
        [EQ-LINALG-001], [REF-GOLUB-VANLOAN-2013, Section 3.1].

    """
    solved: list[list[float]] = []
    for i, row in enumerate(rows):
        values = list(row)
        for k in range(i):
            factor = lower[i][k]
            if factor != 0.0:
                previous = solved[k]
                values = [v - factor * z for v, z in zip(values, previous, strict=True)]
        pivot = lower[i][i]
        solved.append([v / pivot for v in values])
    return solved


def _transform_weights(
    whitened: Sequence[Sequence[float]], innovation: Sequence[float], count: int
) -> tuple[list[float], list[list[float]]]:
    """Return the ETKF mean weights ``wbar^a`` and symmetric transform ``W^a``.

    With ``Yhat`` the ``p`` by ``N`` whitened (and possibly tapered)
    observation anomalies, ``dhat`` the whitened innovation, and
    ``a = N - 1``: ``Ptilde = (a I + Yhat' Yhat)^-1``,
    ``wbar = Ptilde Yhat' dhat``, ``W = (a Ptilde)^(1/2)`` (symmetric
    square root) [EQ-DA-ETKF-001]. For ``p >= N`` the ``N`` by ``N`` matrix
    is diagonalized by :func:`probstat.linalg.symmetric_eigen`; for
    ``p < N`` the ``p`` by ``p`` Gram matrix ``Yhat Yhat' = U S U'`` is
    diagonalized instead and ``W = I + sqrt(a) B g(S) B'``,
    ``wbar = B (a I + S)^-1 U' dhat`` with ``B = Yhat' U`` and
    ``g(s) = -1 / (sqrt(a) sqrt(a + s) (sqrt(a) + sqrt(a + s)))``, the same
    matrices written without cancellation for small ``s``.

    Args:
        whitened: ``Yhat`` as ``p`` rows of length ``N``.
        innovation: ``dhat`` of length ``p``.
        count: Ensemble size ``N``.

    Returns:
        ``wbar^a`` and ``W^a`` (as rows).

    Raises:
        ValueError: If an eigenvalue of ``a I + Yhat' Yhat`` is not positive.

    Equation:
        [EQ-DA-ETKF-001], [EQ-LINALG-005].

    References:
        [REF-HUNT-KOSTELICH-SZUNYOGH-2007], [REF-BISHOP-ETHERTON-MAJUMDAR-2001],
        [REF-GOLUB-VANLOAN-2013, Section 8.5].

    """
    a = float(count - 1)
    root_a = math.sqrt(a)
    p = len(whitened)
    if p == 0:
        return [0.0] * count, [
            [1.0 if i == j else 0.0 for j in range(count)] for i in range(count)
        ]
    if p >= count:
        columns = list(zip(*whitened, strict=True))
        matrix = tuple(
            tuple(
                (a if i == j else 0.0)
                + math.fsum(u * v for u, v in zip(columns[i], columns[j], strict=True))
                for j in range(count)
            )
            for i in range(count)
        )
        values, vectors = symmetric_eigen(matrix)
        if values[0] <= 0.0:
            raise ValueError("the ETKF analysis matrix is not positive definite")
        projected = [
            math.fsum(u * d for u, d in zip(column, innovation, strict=True))
            for column in columns
        ]
        coefficients = [
            math.fsum(vectors[i][k] * projected[i] for i in range(count)) / values[k]
            for k in range(count)
        ]
        mean = [
            math.fsum(vectors[i][k] * coefficients[k] for k in range(count))
            for i in range(count)
        ]
        scales = [root_a / math.sqrt(value) for value in values]
        transform = [
            [
                math.fsum(
                    vectors[i][k] * scales[k] * vectors[j][k] for k in range(count)
                )
                for j in range(count)
            ]
            for i in range(count)
        ]
        return mean, transform
    gram = tuple(
        tuple(
            math.fsum(u * v for u, v in zip(whitened[i], whitened[j], strict=True))
            for j in range(p)
        )
        for i in range(p)
    )
    values, vectors = symmetric_eigen(gram)
    shifted = [a + value for value in values]
    if any(value <= 0.0 for value in shifted):
        raise ValueError("the ETKF analysis matrix is not positive definite")
    basis = [
        [
            math.fsum(whitened[row][i] * vectors[row][k] for row in range(p))
            for k in range(p)
        ]
        for i in range(count)
    ]
    coefficients = [
        math.fsum(vectors[row][k] * innovation[row] for row in range(p)) / shifted[k]
        for k in range(p)
    ]
    mean = [
        math.fsum(basis[i][k] * coefficients[k] for k in range(p)) for i in range(count)
    ]
    corrections = [
        -root_a / (root_a * math.sqrt(value) * (root_a + math.sqrt(value)))
        for value in shifted
    ]
    transform = [
        [
            (1.0 if i == j else 0.0)
            + math.fsum(
                basis[i][k] * corrections[k] * basis[j][k] for k in range(p)
            )
            for j in range(count)
        ]
        for i in range(count)
    ]
    return mean, transform


def _apply_weights(
    values: Sequence[float], mean: Sequence[float], transform: Sequence[Sequence[float]]
) -> tuple[list[float], Matrix]:
    """Update one state component of every member and return its transform.

    ``x_i^a = xbar + sum_k X_k (wbar_k + W_ki)`` for the forecast values
    ``x`` of one component over the members; the transform is
    ``T = 11'/N + (I - 11'/N)(wbar 1' + W)`` [EQ-DA-ETKF-001].

    Args:
        values: Forecast values of the component, one per member.
        mean: ``wbar^a``.
        transform: ``W^a`` as rows.

    Returns:
        The analysis values and ``T``.

    References:
        [EQ-DA-ETKF-001], [EQ-DA-ENKS-001], [REF-HUNT-KOSTELICH-SZUNYOGH-2007].

    """
    count = len(values)
    center = math.fsum(values) / count
    anomalies = [value - center for value in values]
    combined = [
        [mean[k] + transform[k][i] for i in range(count)] for k in range(count)
    ]
    updated = [
        center + math.fsum(anomalies[k] * combined[k][i] for k in range(count))
        for i in range(count)
    ]
    column_means = [
        math.fsum(combined[k][i] for k in range(count)) / count for i in range(count)
    ]
    weights = tuple(
        tuple(
            1.0 / count + combined[k][i] - column_means[i] for i in range(count)
        )
        for k in range(count)
    )
    return updated, weights


def _local_weights(
    anomalies: Sequence[Sequence[float]],
    innovation: Sequence[float],
    noise: Matrix,
    indices: Sequence[int],
    tapers: Sequence[float],
    count: int,
) -> tuple[list[float], list[list[float]]]:
    """Return ETKF weights from a (tapered) subset of the observations.

    The rows ``indices`` of the observation anomalies and of the innovation
    are multiplied by ``sqrt(rho_l)`` and whitened by the Cholesky factor of
    the corresponding block of ``R``, so that ``Yhat' Yhat =
    Y' D^(1/2) R_loc^-1 D^(1/2) Y`` with ``D = diag(rho)``: for a diagonal
    ``R`` each inverse variance is multiplied by ``rho_l`` (R-localization,
    [EQ-DA-LETKF-001]). The global ETKF uses all rows with ``rho = 1``.

    Args:
        anomalies: Observation anomalies as ``m`` rows of length ``N``.
        innovation: ``y - ybar`` of length ``m``.
        noise: ``R``.
        indices: Observation indices used.
        tapers: ``rho_l`` for each used index, in ``(0, 1]``.
        count: Ensemble size ``N``.

    Returns:
        ``wbar^a`` and ``W^a``.

    Raises:
        ValueError: If the block of ``R`` is not positive definite.

    Equation:
        [EQ-DA-ETKF-001], [EQ-DA-LETKF-001].

    References:
        [REF-HUNT-KOSTELICH-SZUNYOGH-2007].

    """
    block = tuple(tuple(noise[i][j] for j in indices) for i in indices)
    try:
        lower = cholesky(block)
    except ValueError as error:
        raise ValueError("observation_covariance must be positive definite") from error
    roots = [math.sqrt(taper) for taper in tapers]
    tapered = [
        [root * value for value in anomalies[index]]
        for index, root in zip(indices, roots, strict=True)
    ]
    scaled_innovation = [
        [root * innovation[index]] for index, root in zip(indices, roots, strict=True)
    ]
    whitened_rows = _whiten(lower, tapered)
    whitened_innovation = [row[0] for row in _whiten(lower, scaled_innovation)]
    return _transform_weights(whitened_rows, whitened_innovation, count)


def etkf_analysis(
    ensemble: Ensemble,
    observation: Sequence[float],
    observation_function: Callable[[Vector], Sequence[float]],
    observation_covariance: Sequence[Sequence[float]],
) -> EnsembleTransformAnalysis:
    """Update a forecast ensemble with one observation by the ETKF.

    With forecast anomalies ``X``, observation anomalies ``Y`` (of
    ``h(x_i)``), and ``a = N - 1``: ``Ptilde = (a I + Y' R^-1 Y)^-1``,
    ``wbar = Ptilde Y' R^-1 (y - ybar)``, ``W = (a Ptilde)^(1/2)`` (the
    symmetric square root), and ``x_i^a = xbar + X (wbar + W_i)``
    [EQ-DA-ETKF-001]. For a linear ``h`` the analysis mean and sample
    covariance equal the Kalman update of the forecast ensemble mean and
    sample covariance; the symmetric square root keeps the analysis mean
    equal to the updated mean and changes the anomalies as little as
    possible. ``R^-1`` is applied through Cholesky solves.

    Args:
        ensemble: Forecast ensemble.
        observation: Observation ``y``.
        observation_function: Measurement map ``h``.
        observation_covariance: Positive-definite ``R``.

    Returns:
        Analysis ensemble, innovation, and the shared transform.

    Raises:
        ValueError: If shapes disagree or ``R`` is not positive definite.

    Equation:
        [EQ-DA-ETKF-001].

    References:
        [REF-BISHOP-ETHERTON-MAJUMDAR-2001], [REF-HUNT-KOSTELICH-SZUNYOGH-2007].

    """
    y, noise, images, predicted = _observation_images(
        ensemble, observation, observation_function, observation_covariance
    )
    count = ensemble.size
    anomalies = [
        [image[row] - predicted[row] for image in images] for row in range(len(y))
    ]
    innovation = tuple(a - b for a, b in zip(y, predicted, strict=True))
    indices = tuple(range(len(y)))
    mean, transform = _local_weights(
        anomalies, innovation, noise, indices, tuple(1.0 for _ in indices), count
    )
    columns = []
    weights: Matrix = ()
    for component in range(len(ensemble.members[0])):
        values, weights = _apply_weights(
            [member[component] for member in ensemble.members], mean, transform
        )
        columns.append(values)
    members = tuple(tuple(column[i] for column in columns) for i in range(count))
    return EnsembleTransformAnalysis(
        Ensemble._trusted(members), innovation, (weights,), "etkf"
    )


def eakf_analysis(
    ensemble: Ensemble,
    observation: Sequence[float],
    observation_function: Callable[[Vector], Sequence[float]],
    observation_covariance: Sequence[Sequence[float]],
) -> EnsembleTransformAnalysis:
    """Assimilate the observation components one at a time by the serial EAKF.

    For each scalar observation ``y_o`` with error variance ``r``, the prior
    observed ensemble ``y_i`` (mean ``ybar``, variance ``s_p``) is adjusted
    to ``y_i^a = ybar^a + sqrt(r / (r + s_p)) (y_i - ybar)`` with
    ``s_a = (1/s_p + 1/r)^-1`` and ``ybar^a = s_a (ybar/s_p + y_o/r)``; every
    state component and every not-yet-assimilated observed component ``z``
    then receives ``z_i += (cov(z, y) / s_p)(y_i^a - y_i)`` [EQ-DA-EAKF-001].
    The observed ensemble is computed once from the forecast and updated by
    the same regression, so a nonlinear ``h`` is handled through the joint
    state-observation ensemble. For a linear ``h`` the result equals the
    Kalman update of the forecast ensemble moments.

    Args:
        ensemble: Forecast ensemble.
        observation: Observation ``y``.
        observation_function: Measurement map ``h``.
        observation_covariance: Diagonal positive-definite ``R``.

    Returns:
        Analysis ensemble, innovation, and the shared transform.

    Raises:
        ValueError: If shapes disagree, ``R`` is not diagonal with positive
            entries, or an observed component has zero ensemble variance.

    Equation:
        [EQ-DA-EAKF-001].

    References:
        [REF-ANDERSON-2001], [REF-ANDERSON-JL-2003].

    """
    y, noise, images, predicted = _observation_images(
        ensemble, observation, observation_function, observation_covariance
    )
    size = len(y)
    for i in range(size):
        if not noise[i][i] > 0.0:
            raise ValueError("observation variances must be positive")
        if any(noise[i][j] != 0.0 for j in range(size) if j != i):
            raise ValueError("the serial EAKF needs a diagonal observation_covariance")
    count = ensemble.size
    dimension = len(ensemble.members[0])
    state_rows = [[member[j] for member in ensemble.members] for j in range(dimension)]
    observed_rows = [[image[j] for image in images] for j in range(size)]
    identity_rows = [
        [1.0 if i == j else 0.0 for i in range(count)] for j in range(count)
    ]
    rows = state_rows + observed_rows + identity_rows
    for index in range(size):
        prior = rows[dimension + index]
        center = math.fsum(prior) / count
        deviations = [value - center for value in prior]
        spread = math.fsum(d * d for d in deviations) / (count - 1)
        if not spread > 0.0:
            raise ValueError(f"observed component {index} has zero ensemble variance")
        variance = noise[index][index]
        posterior = 1.0 / (1.0 / spread + 1.0 / variance)
        target = posterior * (center / spread + y[index] / variance)
        shrink = math.sqrt(variance / (variance + spread))
        increments = [
            target + shrink * d - value
            for d, value in zip(deviations, prior, strict=True)
        ]
        for row in rows:
            row_center = math.fsum(row) / count
            covariance = math.fsum(
                (value - row_center) * d for value, d in zip(row, deviations, strict=True)
            ) / (count - 1)
            slope = covariance / spread
            if slope != 0.0:
                row[:] = [
                    value + slope * inc
                    for value, inc in zip(row, increments, strict=True)
                ]
    members = tuple(
        tuple(state_rows[j][i] for j in range(dimension)) for i in range(count)
    )
    transform = tuple(tuple(row) for row in identity_rows)
    innovation = tuple(a - b for a, b in zip(y, predicted, strict=True))
    return EnsembleTransformAnalysis(
        Ensemble._trusted(members), innovation, (transform,), "eakf"
    )
