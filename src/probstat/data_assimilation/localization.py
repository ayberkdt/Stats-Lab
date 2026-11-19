"""Distance-based localization of ensemble analyses and the LETKF.

With ``N`` members the sample covariance has rank at most ``N - 1`` and its
long-range entries are mostly sampling noise. Localization damps them with a
correlation function of distance that vanishes beyond a cut-off:

* the fifth-order piecewise rational function of Gaspari & Cohn (1999)
  [EQ-DA-LOC-001], a compactly supported correlation function with
  half-width ``c`` (it is zero beyond ``2c``);
* Schur-product covariance localization of the stochastic EnKF gain,
  ``K = (rho_xy o C_xy)(rho_yy o C_yy + R)^-1`` (Houtekamer & Mitchell
  2001) [EQ-DA-LOC-002];
* the local ensemble transform Kalman filter (LETKF) of Hunt, Kostelich &
  Szunyogh (2007): an ETKF analysis at every state component that uses only
  nearby observations, each with its inverse error variance tapered by the
  correlation function of its distance (R-localization) [EQ-DA-LETKF-001].

Distances are Euclidean between coordinate points; with a ``period`` every
axis is periodic with that length (the Lorenz-96 circle has period ``J``).
A half-width of ``math.inf`` switches localization off (every weight is
exactly one).

References:
    [REF-GASPARI-COHN-1999], [REF-HOUTEKAMER-MITCHELL-2001],
    [REF-HUNT-KOSTELICH-SZUNYOGH-2007].

"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream
from probstat.linalg import Matrix, Vector

from .enkf import Ensemble, EnsembleAnalysis, _stochastic_update
from .square_root import (
    EnsembleTransformAnalysis,
    _apply_weights,
    _local_weights,
    _observation_images,
)


def gaspari_cohn(distance: float, half_width: float) -> float:
    """Return the Gaspari-Cohn fifth-order compactly supported correlation.

    With ``z = |d| / c``: ``-z^5/4 + z^4/2 + 5z^3/8 - 5z^2/3 + 1`` for
    ``z <= 1``, ``z^5/12 - z^4/2 + 5z^3/8 + 5z^2/3 - 5z + 4 - 2/(3z)`` for
    ``1 < z <= 2``, and ``0`` beyond [EQ-DA-LOC-001]. The function is a
    valid (positive-definite) correlation function in up to three
    dimensions, equals one at zero and ``5/24`` at ``d = c``, and is twice
    continuously differentiable.

    Args:
        distance: Distance ``d`` (its absolute value is used).
        half_width: Half-width ``c > 0``; ``math.inf`` gives one everywhere.

    Returns:
        The correlation ``rho(d)`` in ``[0, 1]``.

    Raises:
        ValueError: If the distance is not finite or the half-width is not
            positive.

    Equation:
        [EQ-DA-LOC-001].

    References:
        [REF-GASPARI-COHN-1999, Equation 4.10].

    """
    d = abs(float(distance))
    c = float(half_width)
    if not math.isfinite(d):
        raise ValueError(f"distance must be finite, got {distance!r}")
    if not c > 0.0 or math.isnan(c):
        raise ValueError(f"half_width must be positive, got {half_width!r}")
    if c == math.inf:
        return 1.0
    z = d / c
    if z <= 1.0:
        return (((-0.25 * z + 0.5) * z + 0.625) * z - 5.0 / 3.0) * z * z + 1.0
    if z <= 2.0:
        value = (
            ((((z / 12.0 - 0.5) * z + 0.625) * z + 5.0 / 3.0) * z - 5.0) * z
            + 4.0
            - 2.0 / (3.0 * z)
        )
        return max(value, 0.0)
    return 0.0


def _points(values: Sequence[float | Sequence[float]], name: str) -> tuple[Vector, ...]:
    """Normalize coordinates to a tuple of equally long finite points.

    Args:
        values: Scalars (one-dimensional coordinates) or coordinate tuples.
        name: Name used in error messages.

    Returns:
        The points.

    Raises:
        ValueError: If no points are given, a point is not finite, or the
            points have different dimensions.

    References:
        [REF-GASPARI-COHN-1999]. No equation is evaluated.

    """
    points: list[Vector] = []
    for value in values:
        if isinstance(value, (int, float)):
            point: Vector = (float(value),)
        else:
            point = tuple(float(entry) for entry in value)
        if not point or not all(map(math.isfinite, point)):
            raise ValueError(f"{name} must be finite and non-empty")
        points.append(point)
    if not points:
        raise ValueError(f"{name} must not be empty")
    if any(len(point) != len(points[0]) for point in points):
        raise ValueError(f"{name} must have one common dimension")
    return tuple(points)


def _distance(left: Vector, right: Vector, period: float | None) -> float:
    """Return the Euclidean distance, periodic per axis when ``period`` is set.

    References:
        [EQ-DA-LOC-001]. No further equation is evaluated.

    """
    total = 0.0
    for a, b in zip(left, right, strict=True):
        gap = abs(a - b)
        if period is not None:
            gap = math.fmod(gap, period)
            gap = min(gap, period - gap)
        total += gap * gap
    return math.sqrt(total)


def localization_matrix(
    left_coordinates: Sequence[float | Sequence[float]],
    right_coordinates: Sequence[float | Sequence[float]],
    half_width: float,
    *,
    period: float | None = None,
) -> Matrix:
    """Return the Gaspari-Cohn weights between two sets of points.

    Entry ``(i, j)`` is ``rho(d(a_i, b_j))`` with ``rho`` of
    [EQ-DA-LOC-001]. Built from the same points on both sides, the matrix is
    positive semidefinite (for coordinates of dimension at most three
    without a period), so its Schur product with a covariance is again a
    covariance.

    Args:
        left_coordinates: Points ``a_i`` (scalars or tuples).
        right_coordinates: Points ``b_j`` of the same dimension.
        half_width: Half-width ``c``; ``math.inf`` gives all ones.
        period: Optional length of a periodic domain (every axis).

    Returns:
        The weight matrix.

    Raises:
        ValueError: If the coordinates, half-width, or period are invalid.

    Equation:
        [EQ-DA-LOC-001].

    References:
        [REF-GASPARI-COHN-1999].

    """
    left = _points(left_coordinates, "left_coordinates")
    right = _points(right_coordinates, "right_coordinates")
    if len(left[0]) != len(right[0]):
        raise ValueError("coordinates must have the same dimension on both sides")
    length = _period(period)
    return tuple(
        tuple(gaspari_cohn(_distance(a, b, length), half_width) for b in right)
        for a in left
    )


def _period(period: float | None) -> float | None:
    """Validate an optional periodic domain length.

    Raises:
        ValueError: If the period is not finite and positive.

    References:
        [REF-GASPARI-COHN-1999]. No equation is evaluated.

    """
    if period is None:
        return None
    length = float(period)
    if not (math.isfinite(length) and length > 0.0):
        raise ValueError(f"period must be finite and positive, got {period!r}")
    return length


@dataclass(frozen=True, slots=True)
class Localization:
    """Coordinates and Gaspari-Cohn half-width of a localized analysis.

    Args:
        state_coordinates: One point per state component (scalars or
            tuples).
        observation_coordinates: One point per observation component.
        half_width: Gaspari-Cohn half-width ``c`` (weights vanish beyond
            ``2c``); ``math.inf`` switches localization off.
        period: Optional length of a periodic domain (every axis).

    Raises:
        ValueError: If coordinates, half-width, or period are invalid.

    References:
        [EQ-DA-LOC-001], [REF-GASPARI-COHN-1999],
        [REF-HUNT-KOSTELICH-SZUNYOGH-2007].

    """

    state_coordinates: tuple[Vector, ...]
    observation_coordinates: tuple[Vector, ...]
    half_width: float
    period: float | None = None

    def __post_init__(self) -> None:
        """Normalize and validate the coordinates.

        Raises:
            ValueError: If an argument is invalid.

        References:
            [REF-GASPARI-COHN-1999]. No equation is evaluated.

        """
        state = _points(self.state_coordinates, "state_coordinates")
        observed = _points(self.observation_coordinates, "observation_coordinates")
        if len(state[0]) != len(observed[0]):
            raise ValueError("state and observation coordinates differ in dimension")
        width = float(self.half_width)
        if not width > 0.0 or math.isnan(width):
            raise ValueError("half_width must be positive")
        object.__setattr__(self, "state_coordinates", state)
        object.__setattr__(self, "observation_coordinates", observed)
        object.__setattr__(self, "half_width", width)
        object.__setattr__(self, "period", _period(self.period))

    def state_observation_weights(self) -> Matrix:
        """Return ``rho_xy``, the ``n`` by ``m`` state-observation weights.

        References:
            [EQ-DA-LOC-001], [EQ-DA-LOC-002].

        """
        return localization_matrix(
            self.state_coordinates,
            self.observation_coordinates,
            self.half_width,
            period=self.period,
        )

    def observation_weights(self) -> Matrix:
        """Return ``rho_yy``, the ``m`` by ``m`` observation-observation weights.

        References:
            [EQ-DA-LOC-001], [EQ-DA-LOC-002].

        """
        return localization_matrix(
            self.observation_coordinates,
            self.observation_coordinates,
            self.half_width,
            period=self.period,
        )

    def check(self, state_dimension: int, observation_dimension: int) -> None:
        """Require one coordinate per state and observation component.

        Args:
            state_dimension: ``n``.
            observation_dimension: ``m``.

        Raises:
            ValueError: If the counts differ.

        References:
            [REF-HUNT-KOSTELICH-SZUNYOGH-2007]. No equation is evaluated.

        """
        if len(self.state_coordinates) != state_dimension:
            raise ValueError("localization needs one coordinate per state component")
        if len(self.observation_coordinates) != observation_dimension:
            raise ValueError(
                "localization needs one coordinate per observation component"
            )


def localized_ensemble_analysis(
    ensemble: Ensemble,
    observation: Sequence[float],
    observation_function: Callable[[Vector], Sequence[float]],
    observation_covariance: Sequence[Sequence[float]],
    localization: Localization,
    *,
    seed: int | RandomStream | None = None,
) -> EnsembleAnalysis:
    """Stochastic EnKF analysis with Schur-product covariance localization.

    The ensemble covariances of [EQ-DA-ENKF-001] are tapered entrywise
    before the gain is formed: ``K = (rho_xy o C_xy)(rho_yy o C_yy + R)^-1``
    [EQ-DA-LOC-002]; every member then receives its perturbed-observation
    update with this gain. The observation perturbations use the same
    stream as :func:`~probstat.data_assimilation.ensemble_analysis`, so
    ``half_width=math.inf`` reproduces it exactly.

    Args:
        ensemble: Forecast ensemble.
        observation: Observation ``y``.
        observation_function: Measurement map ``h``.
        observation_covariance: Positive-definite ``R``.
        localization: Coordinates and half-width.
        seed: Integer seed or stream for the observation perturbations.

    Returns:
        Analysis ensemble, localized gain, and innovation statistics.

    Raises:
        ValueError: If shapes disagree or a covariance is not positive
            definite.

    Equation:
        [EQ-DA-ENKF-001], [EQ-DA-LOC-002].

    References:
        [REF-HOUTEKAMER-MITCHELL-2001], [REF-GASPARI-COHN-1999].

    """
    localization.check(len(ensemble.members[0]), len(observation))
    return _stochastic_update(
        ensemble,
        observation,
        observation_function,
        observation_covariance,
        seed,
        state_weights=localization.state_observation_weights(),
        observation_weights=localization.observation_weights(),
    )[0]


def _local_sets(
    weights: Matrix,
) -> tuple[tuple[tuple[int, ...], tuple[float, ...]], ...]:
    """Return, per state component, the observations with positive weight.

    Args:
        weights: ``rho_xy``.

    Returns:
        For each row, the observation indices with positive weight and the
        weights themselves.

    References:
        [EQ-DA-LETKF-001].

    """
    return tuple(
        (
            tuple(index for index, value in enumerate(row) if value > 0.0),
            tuple(value for value in row if value > 0.0),
        )
        for row in weights
    )


def _letkf(
    ensemble: Ensemble,
    observation: Sequence[float],
    observation_function: Callable[[Vector], Sequence[float]],
    observation_covariance: Sequence[Sequence[float]],
    local_sets: tuple[tuple[tuple[int, ...], tuple[float, ...]], ...],
) -> EnsembleTransformAnalysis:
    """Run the local analyses of the LETKF from precomputed local sets.

    References:
        [EQ-DA-ETKF-001], [EQ-DA-LETKF-001],
        [REF-HUNT-KOSTELICH-SZUNYOGH-2007].

    """
    y, noise, images, predicted = _observation_images(
        ensemble, observation, observation_function, observation_covariance
    )
    count = ensemble.size
    dimension = len(ensemble.members[0])
    if len(local_sets) != dimension:
        raise ValueError("localization needs one coordinate per state component")
    anomalies = [
        [image[row] - predicted[row] for image in images] for row in range(len(y))
    ]
    innovation = tuple(a - b for a, b in zip(y, predicted, strict=True))
    identity = tuple(
        tuple(1.0 if i == j else 0.0 for j in range(count)) for i in range(count)
    )
    columns = []
    transforms = []
    for component, (indices, tapers) in enumerate(local_sets):
        values = [member[component] for member in ensemble.members]
        if not indices:
            columns.append(values)
            transforms.append(identity)
            continue
        mean, transform = _local_weights(
            anomalies, innovation, noise, indices, tapers, count
        )
        updated, weights = _apply_weights(values, mean, transform)
        columns.append(updated)
        transforms.append(weights)
    members = tuple(tuple(column[i] for column in columns) for i in range(count))
    return EnsembleTransformAnalysis(
        Ensemble._trusted(members), innovation, tuple(transforms), "letkf"
    )


def letkf_analysis(
    ensemble: Ensemble,
    observation: Sequence[float],
    observation_function: Callable[[Vector], Sequence[float]],
    observation_covariance: Sequence[Sequence[float]],
    localization: Localization,
) -> EnsembleTransformAnalysis:
    """Update a forecast ensemble by the local ensemble transform Kalman filter.

    For every state component ``j`` the observations with positive weight
    ``rho_jl = rho(d(s_j, o_l))`` form the local set; their anomaly rows and
    innovations are multiplied by ``sqrt(rho_jl)`` and whitened by the
    Cholesky factor of the local block of ``R``, which multiplies each
    inverse error variance by ``rho_jl`` for a diagonal ``R``
    (R-localization) [EQ-DA-LETKF-001]. The ETKF weights of that local
    problem [EQ-DA-ETKF-001] update component ``j`` only. A component
    without nearby observations keeps its forecast. With
    ``half_width=math.inf`` every local problem is the global one and the
    result equals :func:`~probstat.data_assimilation.etkf_analysis`.

    Args:
        ensemble: Forecast ensemble.
        observation: Observation ``y``.
        observation_function: Measurement map ``h``.
        observation_covariance: Positive-definite ``R``.
        localization: Coordinates and half-width.

    Returns:
        Analysis ensemble, innovation, and one transform per state component.

    Raises:
        ValueError: If shapes disagree or a local block of ``R`` is not
            positive definite.

    Equation:
        [EQ-DA-ETKF-001], [EQ-DA-LETKF-001].

    References:
        [REF-HUNT-KOSTELICH-SZUNYOGH-2007], [REF-GASPARI-COHN-1999].

    """
    localization.check(len(ensemble.members[0]), len(observation))
    return _letkf(
        ensemble,
        observation,
        observation_function,
        observation_covariance,
        _local_sets(localization.state_observation_weights()),
    )
