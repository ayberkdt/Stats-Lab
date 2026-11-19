"""Stochastic ensemble Kalman filter with perturbed observations.

The EnKF replaces the Kalman covariance by the sample covariance of an
ensemble of model states. The forecast step runs every member through the
(possibly nonlinear) dynamics with sampled process noise; the analysis step
updates every member with a Kalman gain built from ensemble covariances and an
observation perturbed by its own draw of measurement noise. Without that
perturbation the analysis ensemble would be systematically too narrow.

Sampling error is the defining limitation: with ``N`` members the covariance
estimate has rank at most ``N - 1`` and relative error of order ``N^-1/2``,
which typically under-disperses the ensemble. Multiplicative inflation is
exposed as an explicit parameter; nothing is inflated or localized silently.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.linalg import (
    Matrix,
    Vector,
    as_matrix,
    as_vector,
    cholesky,
    cholesky_semidefinite,
    cholesky_solve,
    symmetrize,
)


class AdditiveGaussianModel(Protocol):
    """Structural interface of an additive-Gaussian state-space model.

    Both :class:`probstat.filtering.LinearGaussianStateSpace` and
    :class:`probstat.filtering.NonlinearStateSpace` satisfy it, so this
    package needs no import from the filtering package.

    References:
        [REF-EVENSEN-2003], [REF-SARKKA-2013, Chapter 4].

    """

    @property
    def state_dimension(self) -> int:
        """Return the state dimension.

        References:
            [REF-SARKKA-2013, Chapter 4].

        """
        ...

    @property
    def observation_dimension(self) -> int:
        """Return the observation dimension.

        References:
            [REF-SARKKA-2013, Chapter 4].

        """
        ...

    @property
    def process_covariance(self) -> Matrix:
        """Return the process-noise covariance ``Q``.

        References:
            [REF-SARKKA-2013, Chapter 4].

        """
        ...

    @property
    def observation_covariance(self) -> Matrix:
        """Return the measurement-noise covariance ``R``.

        References:
            [REF-SARKKA-2013, Chapter 4].

        """
        ...

    @property
    def initial_mean(self) -> Vector:
        """Return the prior mean ``m_0``.

        References:
            [REF-SARKKA-2013, Chapter 4].

        """
        ...

    @property
    def initial_covariance(self) -> Matrix:
        """Return the prior covariance ``P_0``.

        References:
            [REF-SARKKA-2013, Chapter 4].

        """
        ...

    def transition(
        self, state: Sequence[float], control: Sequence[float] | None = None
    ) -> Vector:
        """Return the noise-free transition of one state.

        References:
            [EQ-FILT-SSM-002], [REF-SARKKA-2013, Chapter 5].

        """
        ...

    def observe(self, state: Sequence[float]) -> Vector:
        """Return the noise-free observation of one state.

        References:
            [EQ-FILT-SSM-002], [REF-SARKKA-2013, Chapter 5].

        """
        ...


@dataclass(frozen=True, slots=True)
class Ensemble:
    """An ensemble of state vectors with sample moments.

    Args:
        members: At least two equally long finite state vectors.

    References:
        [REF-EVENSEN-2003].

    """

    members: tuple[Vector, ...]

    def __post_init__(self) -> None:
        """Validate member count and shape.

        Raises:
            ValueError: If fewer than two members or ragged members are given.

        References:
            [REF-EVENSEN-2003]. No equation is evaluated.

        """
        members = tuple(
            as_vector(member, name=f"members[{index}]")
            for index, member in enumerate(self.members)
        )
        if len(members) < 2:
            raise ValueError("an ensemble needs at least two members")
        if not members[0] or any(len(m) != len(members[0]) for m in members):
            raise ValueError("ensemble members must be non-empty and equally long")
        object.__setattr__(self, "members", members)

    @classmethod
    def _trusted(cls, members: tuple[Vector, ...]) -> Ensemble:
        """Wrap internally generated, already-validated members without copying.

        Args:
            members: Finite, equally long member tuples produced in this module.

        Returns:
            Ensemble sharing ``members``.

        References:
            [REF-EVENSEN-2003]. No equation is evaluated.

        """
        ensemble = object.__new__(cls)
        object.__setattr__(ensemble, "members", members)
        return ensemble

    @property
    def size(self) -> int:
        """Return the number of members ``N``.

        References:
            [REF-EVENSEN-2003].

        """
        return len(self.members)

    @property
    def mean(self) -> Vector:
        """Return the ensemble mean.

        Equation:
            [EQ-DA-ENKF-002].

        References:
            [REF-EVENSEN-2003].

        """
        return _mean(self.members)

    @property
    def covariance(self) -> Matrix:
        """Return the unbiased ensemble covariance (divisor ``N - 1``).

        Equation:
            [EQ-DA-ENKF-002].

        References:
            [REF-EVENSEN-2003].

        """
        return _cross_covariance(self.members, self.members)


@dataclass(frozen=True, slots=True)
class EnsembleAnalysis:
    """Outcome of one stochastic EnKF analysis step.

    Args:
        ensemble: Analysis ensemble.
        gain: Ensemble Kalman gain ``K``.
        innovation: ``y - mean(h(x^f))``.
        innovation_covariance: ``C_yy + R`` from the forecast ensemble.

    References:
        [REF-BURGERS-VANLEEUWEN-EVENSEN-1998], [REF-EVENSEN-2003].

    """

    ensemble: Ensemble
    gain: Matrix
    innovation: Vector
    innovation_covariance: Matrix


def _mean(members: Sequence[Vector]) -> Vector:
    """Return the arithmetic mean vector of equally long members.

    References:
        [EQ-DA-ENKF-002], [REF-EVENSEN-2003].

    """
    count = len(members)
    return tuple(
        math.fsum(member[j] for member in members) / count
        for j in range(len(members[0]))
    )


def _cross_covariance(left: Sequence[Vector], right: Sequence[Vector]) -> Matrix:
    """Return ``(N-1)^-1 sum_i (a_i - abar)(b_i - bbar)'``.

    Args:
        left: First member sequence.
        right: Second member sequence of equal count.

    Returns:
        Sample cross-covariance matrix.

    Equation:
        [EQ-DA-ENKF-002].

    References:
        [REF-EVENSEN-2003].

    """
    count = len(left)
    left_mean = _mean(left)
    right_mean = _mean(right)
    left_anomalies = [
        tuple(a - m for a, m in zip(member, left_mean, strict=True)) for member in left
    ]
    right_anomalies = [
        tuple(b - m for b, m in zip(member, right_mean, strict=True))
        for member in right
    ]
    return tuple(
        tuple(
            math.fsum(
                a[i] * b[j]
                for a, b in zip(left_anomalies, right_anomalies, strict=True)
            )
            / (count - 1)
            for j in range(len(right_mean))
        )
        for i in range(len(left_mean))
    )


def _gaussian_noise(gauss: Callable[[], float], lower: Matrix) -> Vector:
    """Return ``L z`` with independent standard-normal ``z``.

    Plain summation is used: the result is a random draw, so compensated
    summation would add cost without changing its distribution.

    References:
        [REF-DEVROYE-1986, Chapter XI]. No named equation is needed.

    """
    z = tuple(gauss() for _ in range(len(lower)))
    return tuple(sum(a * b for a, b in zip(row, z, strict=True)) for row in lower)


def initial_ensemble(
    model: AdditiveGaussianModel,
    size: int,
    *,
    seed: int | RandomStream | None = None,
) -> Ensemble:
    """Draw ``size`` members from the prior ``N(m_0, P_0)``.

    Args:
        model: Additive-Gaussian state-space model.
        size: Number of members, at least two.
        seed: Integer seed or stream.

    Returns:
        Initial ensemble.

    Raises:
        ValueError: If ``size < 2``.

    Equation:
        [EQ-DA-ENKF-002].

    References:
        [REF-EVENSEN-2003].

    """
    if isinstance(size, bool) or not isinstance(size, int) or size < 2:
        raise ValueError("ensemble size must be an integer of at least two")
    gauss = resolve_stream(seed).generator().gauss
    lower = cholesky_semidefinite(model.initial_covariance)
    members = []
    for _ in range(size):
        noise = _gaussian_noise(gauss, lower)
        members.append(
            tuple(a + b for a, b in zip(model.initial_mean, noise, strict=True))
        )
    return Ensemble._trusted(tuple(members))


def inflate(ensemble: Ensemble, factor: float) -> Ensemble:
    """Apply multiplicative covariance inflation about the ensemble mean.

    Anomalies are scaled by ``sqrt(factor)`` so the sample covariance is
    multiplied by ``factor``; the mean is unchanged.

    Args:
        ensemble: Ensemble to inflate.
        factor: Covariance inflation factor ``>= 1`` (``1`` is no inflation).

    Returns:
        Inflated ensemble.

    Raises:
        ValueError: If ``factor < 1``.

    Equation:
        [EQ-DA-ENKF-002].

    References:
        [REF-EVENSEN-2003].

    """
    value = require_finite_real(factor, name="inflation")
    if value < 1.0:
        raise ValueError("inflation must be at least one")
    if value == 1.0:
        return ensemble
    center = ensemble.mean
    scale = math.sqrt(value)
    return Ensemble._trusted(
        tuple(
            tuple(m + scale * (x - m) for x, m in zip(member, center, strict=True))
            for member in ensemble.members
        )
    )


def ensemble_forecast(
    model: AdditiveGaussianModel,
    ensemble: Ensemble,
    *,
    seed: int | RandomStream | None = None,
    control: Sequence[float] | None = None,
) -> Ensemble:
    """Propagate every member through the dynamics with sampled process noise.

    Args:
        model: Additive-Gaussian state-space model.
        ensemble: Analysis ensemble at time ``k-1``.
        seed: Integer seed or stream for the process noise.
        control: Optional control (linear models only).

    Returns:
        Forecast ensemble ``x_i^f = f(x_i^a) + w_i``.

    Equation:
        [EQ-FILT-SSM-002], [EQ-DA-ENKF-001].

    References:
        [REF-EVENSEN-1994].

    """
    gauss = resolve_stream(seed).generator().gauss
    lower = cholesky_semidefinite(model.process_covariance)
    members = []
    for member in ensemble.members:
        mean = model.transition(member, control)
        noise = _gaussian_noise(gauss, lower)
        members.append(tuple(a + b for a, b in zip(mean, noise, strict=True)))
    return Ensemble._trusted(tuple(members))


def ensemble_analysis(
    ensemble: Ensemble,
    observation: Sequence[float],
    observation_function: Callable[[Vector], Sequence[float]],
    observation_covariance: Sequence[Sequence[float]],
    *,
    seed: int | RandomStream | None = None,
) -> EnsembleAnalysis:
    """Update a forecast ensemble with one observation (stochastic EnKF).

    ``C_xy`` and ``C_yy`` are ensemble (cross-)covariances of ``x`` and
    ``h(x)``; ``K = C_xy (C_yy + R)^-1``; each member receives
    ``x_i^a = x_i^f + K (y + e_i - h(x_i^f))`` with ``e_i ~ N(0, R)``. For a
    linear ``h`` this is the Burgers-van Leeuwen-Evensen scheme; for a
    nonlinear ``h`` the ensemble covariances act as a statistical
    linearization.

    Args:
        ensemble: Forecast ensemble.
        observation: Observation ``y``.
        observation_function: Measurement map ``h``.
        observation_covariance: Positive-definite ``R``.
        seed: Integer seed or stream for the observation perturbations.

    Returns:
        Analysis ensemble, gain, and innovation statistics.

    Raises:
        ValueError: If shapes disagree or ``C_yy + R`` is not positive
            definite.

    Equation:
        [EQ-DA-ENKF-001], [EQ-DA-ENKF-002].

    References:
        [REF-BURGERS-VANLEEUWEN-EVENSEN-1998], [REF-EVENSEN-2003].

    """
    return _stochastic_update(
        ensemble, observation, observation_function, observation_covariance, seed
    )[0]


def _stochastic_update(
    ensemble: Ensemble,
    observation: Sequence[float],
    observation_function: Callable[[Vector], Sequence[float]],
    observation_covariance: Sequence[Sequence[float]],
    seed: int | RandomStream | None,
    *,
    state_weights: Matrix | None = None,
    observation_weights: Matrix | None = None,
    transforms: bool = False,
) -> tuple[EnsembleAnalysis, tuple[Matrix, ...] | None]:
    """Perturbed-observation analysis, optionally Schur-localized.

    Without weights this is exactly the arithmetic of
    :func:`ensemble_analysis`. With localization weights ``rho_xy`` (``n`` by
    ``m``) and ``rho_yy`` (``m`` by ``m``) the gain is
    ``K = (rho_xy o C_xy)(rho_yy o C_yy + R)^-1`` ([EQ-DA-LOC-002]). The
    ensemble transforms ``E^a_j = E^f_j T_j`` with
    ``T_j = I + Y' diag(rho_j) S^-1 D / (N - 1)`` (``Y`` the anomalies of
    ``h(x)``, ``D`` the perturbed innovations) are returned on request for
    the ensemble smoother ([EQ-DA-ENKS-001]).

    Args:
        ensemble: Forecast ensemble.
        observation: Observation ``y``.
        observation_function: Measurement map ``h``.
        observation_covariance: Positive-definite ``R``.
        seed: Integer seed or stream for the observation perturbations.
        state_weights: Optional ``rho_xy``.
        observation_weights: Optional ``rho_yy``; required with
            ``state_weights``.
        transforms: Also return the ensemble transforms (one shared matrix
            without localization, one per state component with it).

    Returns:
        The analysis and the transforms (or ``None``).

    Raises:
        ValueError: If shapes disagree or a covariance is not positive
            definite.

    Equation:
        [EQ-DA-ENKF-001], [EQ-DA-LOC-002], [EQ-DA-ENKS-001].

    References:
        [REF-BURGERS-VANLEEUWEN-EVENSEN-1998], [REF-EVENSEN-2003],
        [REF-HOUTEKAMER-MITCHELL-2001].

    """
    noise = symmetrize(observation_covariance, name="observation_covariance")
    y = as_vector(observation, name="observation")
    if len(noise) != len(y):
        raise ValueError("observation_covariance must match the observation length")
    try:
        noise_factor = cholesky(noise)
    except ValueError as error:
        raise ValueError("observation_covariance must be positive definite") from error
    images = tuple(
        as_vector(observation_function(member), name="h(x)")
        for member in ensemble.members
    )
    if any(len(image) != len(y) for image in images):
        raise ValueError("h(x) length must equal the observation length")
    cross = _cross_covariance(ensemble.members, images)
    image_covariance = _cross_covariance(images, images)
    if state_weights is not None:
        if observation_weights is None:
            raise ValueError("observation_weights are required with state_weights")
        cross = _schur(cross, state_weights)
        image_covariance = _schur(image_covariance, observation_weights)
    innovation_covariance = tuple(
        tuple(c + r for c, r in zip(row_c, row_r, strict=True))
        for row_c, row_r in zip(image_covariance, noise, strict=True)
    )
    try:
        lower = cholesky(innovation_covariance)
    except ValueError as error:
        raise ValueError("innovation covariance is not positive definite") from error
    gain = tuple(cholesky_solve(lower, row) for row in cross)
    gauss = resolve_stream(seed).generator().gauss
    members = []
    residuals = []
    for member, image in zip(ensemble.members, images, strict=True):
        perturbation = _gaussian_noise(gauss, noise_factor)
        residual = tuple(
            yi + ei - hi for yi, ei, hi in zip(y, perturbation, image, strict=True)
        )
        residuals.append(residual)
        members.append(
            tuple(
                x + sum(k * r for k, r in zip(row, residual, strict=True))
                for x, row in zip(member, gain, strict=True)
            )
        )
    predicted = _mean(images)
    innovation = tuple(a - b for a, b in zip(y, predicted, strict=True))
    analysis = EnsembleAnalysis(
        Ensemble._trusted(tuple(members)), gain, innovation, innovation_covariance
    )
    if not transforms:
        return analysis, None
    count = ensemble.size
    anomalies = tuple(
        tuple(value - mean for value, mean in zip(image, predicted, strict=True))
        for image in images
    )
    solved = tuple(cholesky_solve(lower, residual) for residual in residuals)
    rows: tuple[Vector | None, ...] = (
        (None,) if state_weights is None else tuple(state_weights)
    )
    result = []
    for weights in rows:
        scaled = (
            anomalies
            if weights is None
            else tuple(
                tuple(a * w for a, w in zip(anomaly, weights, strict=True))
                for anomaly in anomalies
            )
        )
        result.append(
            tuple(
                tuple(
                    (1.0 if a == i else 0.0)
                    + sum(u * v for u, v in zip(scaled[a], solved[i], strict=True))
                    / (count - 1)
                    for i in range(count)
                )
                for a in range(count)
            )
        )
    return analysis, tuple(result)


def _schur(matrix: Matrix, weights: Matrix) -> Matrix:
    """Return the entrywise (Schur) product of two equally shaped matrices.

    Args:
        matrix: Covariance block.
        weights: Localization weights of the same shape.

    Returns:
        ``weights o matrix``.

    Raises:
        ValueError: If the shapes differ.

    Equation:
        [EQ-DA-LOC-002].

    References:
        [REF-HOUTEKAMER-MITCHELL-2001].

    """
    if len(weights) != len(matrix) or any(
        len(row_w) != len(row_m) for row_w, row_m in zip(weights, matrix, strict=True)
    ):
        raise ValueError("localization weights do not match the covariance shape")
    return tuple(
        tuple(c * w for c, w in zip(row_c, row_w, strict=True))
        for row_c, row_w in zip(matrix, weights, strict=True)
    )
