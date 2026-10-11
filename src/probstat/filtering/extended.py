"""Extended and unscented Kalman filters for nonlinear state-space models.

Both filters approximate the non-Gaussian filtering distribution of a
nonlinear model by a Gaussian. The extended Kalman filter (EKF) linearizes
``f`` and ``h`` with first-order Taylor expansions at the current mean; the
unscented Kalman filter (UKF) propagates a deterministic set of sigma points
through the exact nonlinear maps and re-fits the first two moments. On a
linear model both reduce exactly to the Kalman filter, which the test-suite
checks. Neither is exact for a nonlinear model, and neither reports how large
its linearization or moment-matching error is.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from probstat.linalg import (
    Matrix,
    Vector,
    add,
    as_vector,
    cholesky_semidefinite,
    matmul,
    matvec,
    outer,
    sandwich,
    transpose,
)

from .kalman import (
    GaussianState,
    KalmanFilterResult,
    MeasurementUpdate,
    Observation,
    _observation_vector,
    _require_state,
    gain_from_cross_covariance,
    gaussian_log_density,
    linear_gaussian_update,
)
from .state_space import StateSpaceModel, _control_plan


def extended_kalman_filter(
    model: StateSpaceModel,
    observations: Sequence[Observation],
    *,
    controls: Sequence[Sequence[float] | None] | None = None,
) -> KalmanFilterResult:
    """Run the extended Kalman filter.

    Prediction: ``m^- = f(m)``, ``P^- = F_x P F_x' + Q`` with
    ``F_x = df/dx(m)``. Update: the Joseph-form Kalman update with
    ``y_hat = h(m^-)`` and ``H_x = dh/dx(m^-)``. Jacobians come from the model
    (analytic when supplied, otherwise central differences).

    Args:
        model: Linear or nonlinear additive-Gaussian model.
        observations: Observations ``y_1..y_T`` (``None`` if missing).
        controls: Optional controls (linear models only).

    Returns:
        Forward-pass record; ``log_likelihood`` is the linearized
        approximation of ``log p(y_1:T)``.

    Raises:
        ValueError: If observations are empty or a covariance loses
            positive definiteness.

    Equation:
        [EQ-FILT-EKF-001], [EQ-FILT-KF-002], [EQ-FILT-KF-003],
        [EQ-FILT-KF-004].

    References:
        [REF-SARKKA-2013, Chapter 5], [REF-SIMON-2006, Chapter 13].

    """
    data = tuple(observations)
    if not data:
        raise ValueError("observations must be non-empty")
    plan = _control_plan(controls, len(data))
    state = GaussianState(model.initial_mean, model.initial_covariance)
    filtered = [state]
    predicted_states: list[GaussianState] = []
    cross_covariances: list[Matrix] = []
    innovations: list[Vector | None] = []
    innovation_covariances: list[Matrix | None] = []
    gains: list[Matrix | None] = []
    terms: list[float] = []
    for observation, control in zip(data, plan, strict=True):
        jacobian = model.transition_jacobian(state.mean)
        predicted = GaussianState(
            model.transition(state.mean, control),
            add(sandwich(jacobian, state.covariance), model.process_covariance),
        )
        cross_covariances.append(matmul(state.covariance, transpose(jacobian)))
        predicted_states.append(predicted)
        if observation is None:
            state = predicted
            innovations.append(None)
            innovation_covariances.append(None)
            gains.append(None)
            terms.append(0.0)
        else:
            update = linear_gaussian_update(
                predicted,
                observation,
                model.observe(predicted.mean),
                model.observation_jacobian(predicted.mean),
                model.observation_covariance,
            )
            state = update.state
            innovations.append(update.innovation)
            innovation_covariances.append(update.innovation_covariance)
            gains.append(update.gain)
            terms.append(update.log_likelihood)
        filtered.append(state)
    return KalmanFilterResult(
        tuple(filtered),
        tuple(predicted_states),
        tuple(cross_covariances),
        tuple(innovations),
        tuple(innovation_covariances),
        tuple(gains),
        tuple(terms),
        "extended",
    )


@dataclass(frozen=True, slots=True)
class UnscentedWeights:
    """Scaled unscented-transform parameters and the derived weights.

    Args:
        alpha: Spread of sigma points around the mean (``0 < alpha``).
        beta: Prior-knowledge term; ``beta = 2`` is optimal for Gaussians.
        kappa: Secondary scaling; ``n + lambda`` must stay positive.

    References:
        [REF-JULIER-UHLMANN-2004], [REF-SARKKA-2013, Chapter 5].

    """

    alpha: float = 1.0
    beta: float = 2.0
    kappa: float = 0.0

    def __post_init__(self) -> None:
        """Validate the scaling parameters.

        Raises:
            ValueError: If a parameter is non-finite or ``alpha <= 0``.

        References:
            [REF-SARKKA-2013, Chapter 5]. No equation is evaluated.

        """
        for value, name in (
            (self.alpha, "alpha"),
            (self.beta, "beta"),
            (self.kappa, "kappa"),
        ):
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.alpha <= 0.0:
            raise ValueError("alpha must be positive")

    def scaling(self, dimension: int) -> float:
        """Return ``lambda = alpha^2 (n + kappa) - n``.

        Args:
            dimension: State dimension ``n``.

        Returns:
            Composite scaling ``lambda``.

        Raises:
            ValueError: If ``n + lambda <= 0``.

        Equation:
            [EQ-FILT-UKF-001].

        References:
            [REF-SARKKA-2013, Chapter 5].

        """
        value = self.alpha**2 * (dimension + self.kappa) - dimension
        if dimension + value <= 0.0:
            raise ValueError("n + lambda must be positive; increase alpha or kappa")
        return value

    def weights(self, dimension: int) -> tuple[Vector, Vector]:
        """Return mean weights ``W^m`` and covariance weights ``W^c``.

        ``W^m_0 = lambda/(n+lambda)``,
        ``W^c_0 = lambda/(n+lambda) + 1 - alpha^2 + beta``, and
        ``W^m_i = W^c_i = 1/(2(n+lambda))`` for ``i = 1..2n``.

        Args:
            dimension: State dimension ``n``.

        Returns:
            Two weight vectors of length ``2n+1``.

        Equation:
            [EQ-FILT-UKF-001].

        References:
            [REF-SARKKA-2013, Chapter 5], [REF-JULIER-UHLMANN-2004].

        """
        lam = self.scaling(dimension)
        total = dimension + lam
        side = 1.0 / (2.0 * total)
        mean_weights = (lam / total,) + tuple(side for _ in range(2 * dimension))
        covariance_weights = (lam / total + 1.0 - self.alpha**2 + self.beta,) + tuple(
            side for _ in range(2 * dimension)
        )
        return mean_weights, covariance_weights


@dataclass(frozen=True, slots=True)
class UnscentedTransformResult:
    """Moments of ``g(X)`` estimated by the unscented transform.

    Args:
        mean: Weighted sigma-point mean of ``g``.
        covariance: Weighted sigma-point covariance of ``g`` (no added noise).
        cross_covariance: ``Cov(X, g(X))`` estimate.
        sigma_points: The ``2n+1`` input sigma points.

    References:
        [REF-JULIER-UHLMANN-2004].

    """

    mean: Vector
    covariance: Matrix
    cross_covariance: Matrix
    sigma_points: tuple[Vector, ...]


def unscented_transform(
    function: Callable[[Vector], Sequence[float]],
    mean: Sequence[float],
    covariance: Sequence[Sequence[float]],
    *,
    weights: UnscentedWeights | None = None,
) -> UnscentedTransformResult:
    """Propagate a Gaussian through a nonlinear map with sigma points.

    Sigma points are ``m`` and ``m +/- sqrt(n + lambda) L_i`` where ``L_i`` is
    column ``i`` of a lower square root of ``P``. The transform reproduces the
    exact mean and covariance for linear maps and the exact mean for
    quadratic maps.

    Args:
        function: Map ``g``.
        mean: Input mean ``m``.
        covariance: Positive-semidefinite input covariance ``P``.
        weights: Scaling parameters; defaults to ``alpha=1, beta=2, kappa=0``.

    Returns:
        Mean, covariance, and cross-covariance estimates.

    Raises:
        ValueError: If ``P`` is not positive semidefinite or shapes disagree.

    Equation:
        [EQ-FILT-UKF-001].

    References:
        [REF-JULIER-UHLMANN-2004], [REF-SARKKA-2013, Chapter 5].

    """
    parameters = weights if weights is not None else UnscentedWeights()
    center = as_vector(mean, name="mean")
    size = len(center)
    matrix = tuple(as_vector(row, name="covariance row") for row in covariance)
    if len(matrix) != size or any(len(row) != size for row in matrix):
        raise ValueError("covariance must be square with the mean dimension")
    lower = cholesky_semidefinite(matrix)
    lam = parameters.scaling(size)
    spread = math.sqrt(size + lam)
    columns = transpose(lower)
    points: list[Vector] = [center]
    for sign in (1.0, -1.0):
        for column in columns:
            points.append(
                tuple(
                    c + sign * spread * v for c, v in zip(center, column, strict=True)
                )
            )
    mean_weights, covariance_weights = parameters.weights(size)
    images = [as_vector(function(point), name="g(sigma point)") for point in points]
    width = len(images[0])
    if any(len(image) != width for image in images):
        raise ValueError("function changed output length between sigma points")
    out_mean = tuple(
        math.fsum(w * image[j] for w, image in zip(mean_weights, images, strict=True))
        for j in range(width)
    )
    out_covariance = tuple(tuple(0.0 for _ in range(width)) for _ in range(width))
    cross = tuple(tuple(0.0 for _ in range(width)) for _ in range(size))
    for w, point, image in zip(covariance_weights, points, images, strict=True):
        dy = tuple(a - b for a, b in zip(image, out_mean, strict=True))
        dx = tuple(a - b for a, b in zip(point, center, strict=True))
        out_covariance = add(out_covariance, outer(dy, dy), scale=w)
        cross = add(cross, outer(dx, dy), scale=w)
    symmetric = tuple(
        tuple(0.5 * (out_covariance[i][j] + out_covariance[j][i]) for j in range(width))
        for i in range(width)
    )
    return UnscentedTransformResult(out_mean, symmetric, cross, tuple(points))


def unscented_kalman_filter(
    model: StateSpaceModel,
    observations: Sequence[Observation],
    *,
    weights: UnscentedWeights | None = None,
    controls: Sequence[Sequence[float] | None] | None = None,
) -> KalmanFilterResult:
    """Run the (non-augmented, additive-noise) unscented Kalman filter.

    Prediction: unscented transform of ``f`` plus ``Q``. Update: unscented
    transform of ``h`` at the predicted moments, ``S = P_yy + R``,
    ``K = P_xy S^-1``, ``m+ = m^- + K (y - mu)``, ``P+ = P^- - K S K'``.
    Negative ``W^c_0`` (small ``alpha``) can make covariances indefinite; the
    filter then raises instead of repairing them.

    Args:
        model: Linear or nonlinear additive-Gaussian model.
        observations: Observations ``y_1..y_T`` (``None`` if missing).
        weights: Unscented scaling parameters.
        controls: Optional controls (linear models only).

    Returns:
        Forward-pass record whose ``cross_covariances`` support the
        unscented RTS smoother via :func:`rts_smoother`.

    Raises:
        ValueError: If observations are empty or a covariance loses
            positive (semi)definiteness.

    Equation:
        [EQ-FILT-UKF-001], [EQ-FILT-UKF-002], [EQ-FILT-KF-004].

    References:
        [REF-JULIER-UHLMANN-2004], [REF-SARKKA-2013, Chapters 5 and 9].

    """
    parameters = weights if weights is not None else UnscentedWeights()
    data = tuple(observations)
    if not data:
        raise ValueError("observations must be non-empty")
    plan = _control_plan(controls, len(data))
    state = GaussianState(model.initial_mean, model.initial_covariance)
    filtered = [state]
    predicted_states: list[GaussianState] = []
    cross_covariances: list[Matrix] = []
    innovations: list[Vector | None] = []
    innovation_covariances: list[Matrix | None] = []
    gains: list[Matrix | None] = []
    terms: list[float] = []
    for observation, control in zip(data, plan, strict=True):

        def dynamics(
            point: Vector, control: Sequence[float] | None = control
        ) -> Vector:
            """Apply the noise-free transition with the step's control.

            References:
                [EQ-FILT-SSM-002], [REF-SARKKA-2013, Chapter 5].

            """
            return model.transition(point, control)

        forecast = unscented_transform(
            dynamics, state.mean, state.covariance, weights=parameters
        )
        predicted = GaussianState(
            forecast.mean, add(forecast.covariance, model.process_covariance)
        )
        cross_covariances.append(forecast.cross_covariance)
        predicted_states.append(predicted)
        if observation is None:
            state = predicted
            innovations.append(None)
            innovation_covariances.append(None)
            gains.append(None)
            terms.append(0.0)
        else:
            update = unscented_update(model, predicted, observation, parameters)
            state = update.state
            innovations.append(update.innovation)
            innovation_covariances.append(update.innovation_covariance)
            gains.append(update.gain)
            terms.append(update.log_likelihood)
        filtered.append(state)
    return KalmanFilterResult(
        tuple(filtered),
        tuple(predicted_states),
        tuple(cross_covariances),
        tuple(innovations),
        tuple(innovation_covariances),
        tuple(gains),
        tuple(terms),
        "unscented",
    )


def unscented_update(
    model: StateSpaceModel,
    predicted: GaussianState,
    observation: Sequence[float],
    weights: UnscentedWeights | None = None,
) -> MeasurementUpdate:
    """Condition a predicted Gaussian on one observation by sigma points.

    Args:
        model: Additive-Gaussian model supplying ``h`` and ``R``.
        predicted: Predicted state ``N(m^-, P^-)``.
        observation: Observation ``y``.
        weights: Unscented scaling parameters.

    Returns:
        Measurement update with ``P+ = P^- - K S K'``.

    Raises:
        ValueError: If shapes disagree or ``S`` is not positive definite.

    Equation:
        [EQ-FILT-UKF-001], [EQ-FILT-UKF-002].

    References:
        [REF-JULIER-UHLMANN-2004], [REF-SARKKA-2013, Chapter 5].

    """
    _require_state(predicted, model.state_dimension)
    y = _observation_vector(observation, model.observation_dimension)
    measured = unscented_transform(
        model.observe, predicted.mean, predicted.covariance, weights=weights
    )
    innovation_covariance = add(measured.covariance, model.observation_covariance)
    gain = gain_from_cross_covariance(measured.cross_covariance, innovation_covariance)
    innovation = tuple(a - b for a, b in zip(y, measured.mean, strict=True))
    correction = matvec(gain, innovation)
    mean = tuple(a + b for a, b in zip(predicted.mean, correction, strict=True))
    covariance = add(
        predicted.covariance, sandwich(gain, innovation_covariance), scale=-1.0
    )
    return MeasurementUpdate(
        GaussianState(mean, covariance),
        innovation,
        innovation_covariance,
        gain,
        gaussian_log_density(innovation, innovation_covariance),
    )
