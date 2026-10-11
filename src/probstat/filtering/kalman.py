"""Kalman filtering, Rauch-Tung-Striebel smoothing, and steady-state gains.

The Kalman filter is the exact sequential Bayesian update for the
linear-Gaussian state-space model: the *prediction* step pushes the current
Gaussian belief through the dynamics (covariance propagation plus process
noise), and the *measurement update* conditions that prediction on one new
observation. The filter answers ``p(x_k | y_1:k)``; the RTS smoother runs a
backward pass to answer ``p(x_k | y_1:T)``, which uses future data and is
therefore never less certain than the filter.

Covariance updates use the Joseph form, which stays symmetric positive
semidefinite under rounding and under any (even suboptimal) gain.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from probstat.linalg import (
    Matrix,
    Vector,
    add,
    as_vector,
    cholesky,
    cholesky_solve,
    identity,
    matmul,
    matvec,
    sandwich,
    solve_lower,
    transpose,
)

from .state_space import LinearGaussianStateSpace, _control_plan

_LOG_TWO_PI = math.log(2.0 * math.pi)

Observation = Sequence[float] | None


@dataclass(frozen=True, slots=True)
class GaussianState:
    """Gaussian belief ``N(mean, covariance)`` about one state.

    Args:
        mean: State mean vector.
        covariance: Symmetric positive-semidefinite covariance.

    References:
        [REF-SARKKA-2013, Chapter 4].

    """

    mean: Vector
    covariance: Matrix

    @property
    def variances(self) -> Vector:
        """Return the marginal variances (covariance diagonal).

        References:
            [REF-SARKKA-2013, Chapter 4].

        """
        return tuple(self.covariance[i][i] for i in range(len(self.mean)))

    @property
    def standard_deviations(self) -> Vector:
        """Return marginal standard deviations.

        References:
            [REF-SARKKA-2013, Chapter 4].

        """
        return tuple(math.sqrt(max(value, 0.0)) for value in self.variances)


@dataclass(frozen=True, slots=True)
class MeasurementUpdate:
    """Outcome of one Gaussian measurement update.

    Args:
        state: Posterior (filtered) Gaussian state.
        innovation: Residual ``y - y_hat``.
        innovation_covariance: ``S = H P H' + R`` (or its sigma-point analog).
        gain: Kalman gain ``K``.
        log_likelihood: ``log N(y; y_hat, S)``.

    References:
        [REF-KALMAN-1960], [REF-SARKKA-2013, Chapter 4].

    """

    state: GaussianState
    innovation: Vector
    innovation_covariance: Matrix
    gain: Matrix
    log_likelihood: float


@dataclass(frozen=True, slots=True)
class KalmanFilterResult:
    """Complete forward-pass record of a Gaussian filter.

    Args:
        filtered: Filtered states ``p(x_k | y_1:k)`` for ``k = 0..T``; index 0
            is the prior ``N(m_0, P_0)``.
        predicted: Predicted states ``p(x_k | y_1:k-1)`` for ``k = 1..T``.
        cross_covariances: ``Cov(x_(k-1), x_k | y_1:k-1)`` for ``k = 1..T``;
            ``P_(k-1) F'`` for linear dynamics. Used by the RTS smoother.
        innovations: Innovation per step, ``None`` for a missing observation.
        innovation_covariances: Innovation covariance per step or ``None``.
        gains: Gain per step or ``None``.
        log_likelihood_terms: ``log p(y_k | y_1:k-1)`` per step (0 if missing).
        method: Filter name such as ``"kalman"`` or ``"extended"``.

    References:
        [REF-SARKKA-2013, Chapters 4 and 12].

    """

    filtered: tuple[GaussianState, ...]
    predicted: tuple[GaussianState, ...]
    cross_covariances: tuple[Matrix, ...]
    innovations: tuple[Vector | None, ...]
    innovation_covariances: tuple[Matrix | None, ...]
    gains: tuple[Matrix | None, ...]
    log_likelihood_terms: tuple[float, ...]
    method: str

    @property
    def steps(self) -> int:
        """Return the number of filtered time steps ``T``.

        References:
            [REF-SARKKA-2013, Chapter 4].

        """
        return len(self.predicted)

    @property
    def log_likelihood(self) -> float:
        """Return ``log p(y_1:T)`` by the innovation (prediction-error) form.

        Equation:
            [EQ-FILT-KF-004].

        References:
            [REF-SARKKA-2013, Chapter 12].

        """
        return math.fsum(self.log_likelihood_terms)

    @property
    def means(self) -> tuple[Vector, ...]:
        """Return filtered means for ``k = 0..T``.

        References:
            [REF-SARKKA-2013, Chapter 4].

        """
        return tuple(state.mean for state in self.filtered)


@dataclass(frozen=True, slots=True)
class SmootherResult:
    """Backward-pass record of a Rauch-Tung-Striebel smoother.

    Args:
        smoothed: Smoothed states ``p(x_k | y_1:T)`` for ``k = 0..T``.
        gains: Smoother gains ``G_k`` for ``k = 0..T-1``.

    References:
        [REF-RAUCH-TUNG-STRIEBEL-1965], [REF-SARKKA-2013, Chapter 8].

    """

    smoothed: tuple[GaussianState, ...]
    gains: tuple[Matrix, ...]

    @property
    def means(self) -> tuple[Vector, ...]:
        """Return smoothed means for ``k = 0..T``.

        References:
            [REF-SARKKA-2013, Chapter 8].

        """
        return tuple(state.mean for state in self.smoothed)


@dataclass(frozen=True, slots=True)
class SteadyStateKalman:
    """Limit of the Riccati recursion for a time-invariant model.

    Args:
        predicted_covariance: Fixed point ``P^-`` of the predicted covariance.
        filtered_covariance: Corresponding filtered covariance ``P``.
        gain: Steady-state Kalman gain ``K``.
        iterations: Riccati iterations performed.
        converged: Whether the relative change met the tolerance.

    References:
        [REF-SIMON-2006, Chapter 7], [REF-BAR-SHALOM-LI-KIRUBARAJAN-2001,
        Chapter 5].

    """

    predicted_covariance: Matrix
    filtered_covariance: Matrix
    gain: Matrix
    iterations: int
    converged: bool


def _require_state(state: GaussianState, size: int) -> GaussianState:
    """Validate a Gaussian state's type and dimension.

    Args:
        state: Candidate state.
        size: Required dimension.

    Returns:
        The unchanged state.

    Raises:
        TypeError: If ``state`` is not a :class:`GaussianState`.
        ValueError: If the dimension differs.

    References:
        Project validation policy. No equation is evaluated.

    """
    if not isinstance(state, GaussianState):
        raise TypeError("state must be a GaussianState")
    if len(state.mean) != size or len(state.covariance) != size:
        raise ValueError(f"state must have dimension {size}")
    return state


def _observation_vector(observation: Sequence[float], size: int) -> Vector:
    """Validate one observation vector.

    Args:
        observation: Candidate observation.
        size: Required observation dimension.

    Returns:
        Finite observation vector.

    Raises:
        ValueError: If the length differs.

    References:
        Project validation policy. No equation is evaluated.

    """
    vector = as_vector(observation, name="observation")
    if len(vector) != size:
        raise ValueError(f"observation must have length {size}")
    return vector


def gaussian_log_density(residual: Vector, covariance: Matrix) -> float:
    """Return ``log N(residual; 0, covariance)`` through a Cholesky factor.

    Args:
        residual: Residual vector.
        covariance: Positive-definite covariance of matching dimension.

    Returns:
        Natural log-density.

    Raises:
        ValueError: If the covariance is not positive definite.

    Equation:
        [EQ-DIST-MVN-001], [EQ-FILT-KF-004].

    References:
        [REF-MARDIA-KENT-BIBBY-1979, Chapter 1], [REF-SARKKA-2013, Chapter 12].

    """
    lower = cholesky(covariance)
    whitened = solve_lower(lower, residual)
    log_det = 2.0 * math.fsum(math.log(lower[i][i]) for i in range(len(lower)))
    quadratic = math.fsum(value * value for value in whitened)
    return -0.5 * (len(residual) * _LOG_TWO_PI + log_det + quadratic)


def gain_from_cross_covariance(cross: Matrix, innovation_covariance: Matrix) -> Matrix:
    """Return ``K = C S^-1`` by Cholesky solves, never by explicit inversion.

    Args:
        cross: ``n`` by ``m`` state-observation cross-covariance ``C``.
        innovation_covariance: ``m`` by ``m`` positive-definite ``S``.

    Returns:
        ``n`` by ``m`` gain.

    Raises:
        ValueError: If ``S`` is not positive definite.

    Equation:
        [EQ-FILT-KF-002].

    References:
        [REF-SARKKA-2013, Chapter 4], [REF-GOLUB-VANLOAN-2013, Section 4.2].

    """
    lower = cholesky(innovation_covariance)
    return tuple(cholesky_solve(lower, row) for row in cross)


def linear_gaussian_update(
    predicted: GaussianState,
    observation: Sequence[float],
    predicted_observation: Vector,
    observation_matrix: Matrix,
    observation_covariance: Matrix,
) -> MeasurementUpdate:
    """Condition a Gaussian prediction on a linear(ized) observation.

    ``S = H P H' + R``, ``K = P H' S^-1``, ``m+ = m + K (y - y_hat)``, and the
    Joseph form ``P+ = (I - K H) P (I - K H)' + K R K'``.

    Args:
        predicted: Predicted state ``N(m, P)``.
        observation: Observation ``y``.
        predicted_observation: ``y_hat`` (``H m`` or ``h(m)``).
        observation_matrix: ``H`` (exact or a Jacobian).
        observation_covariance: Positive-definite ``R``.

    Returns:
        Filtered state, innovation, ``S``, gain, and log-likelihood term.

    Raises:
        ValueError: If shapes disagree or ``S`` is not positive definite.

    Equation:
        [EQ-FILT-KF-002], [EQ-FILT-KF-003], [EQ-FILT-KF-004].

    References:
        [REF-KALMAN-1960], [REF-SIMON-2006, Chapter 5],
        [REF-BAR-SHALOM-LI-KIRUBARAJAN-2001, Chapter 5].

    """
    size = len(predicted.mean)
    y = _observation_vector(observation, len(observation_matrix))
    if len(observation_matrix[0]) != size:
        raise ValueError("observation matrix columns must equal state dimension")
    innovation = tuple(a - b for a, b in zip(y, predicted_observation, strict=True))
    innovation_covariance = add(
        sandwich(observation_matrix, predicted.covariance), observation_covariance
    )
    cross = matmul(predicted.covariance, transpose(observation_matrix))
    gain = gain_from_cross_covariance(cross, innovation_covariance)
    correction = matvec(gain, innovation)
    mean = tuple(a + b for a, b in zip(predicted.mean, correction, strict=True))
    residual_map = add(identity(size), matmul(gain, observation_matrix), scale=-1.0)
    covariance = add(
        sandwich(residual_map, predicted.covariance),
        sandwich(gain, observation_covariance),
    )
    log_likelihood = gaussian_log_density(innovation, innovation_covariance)
    return MeasurementUpdate(
        GaussianState(mean, covariance),
        innovation,
        innovation_covariance,
        gain,
        log_likelihood,
    )


def kalman_predict(
    model: LinearGaussianStateSpace,
    state: GaussianState,
    control: Sequence[float] | None = None,
) -> GaussianState:
    """Propagate a Gaussian belief one step through linear dynamics.

    ``m^- = F m + B u`` and ``P^- = F P F' + Q``.

    Args:
        model: Linear-Gaussian model.
        state: Filtered state at time ``k-1``.
        control: Optional control ``u_k``.

    Returns:
        Predicted state at time ``k``.

    Raises:
        TypeError: If the model or state has the wrong type.
        ValueError: If dimensions disagree.

    Equation:
        [EQ-FILT-KF-001].

    References:
        [REF-KALMAN-1960], [REF-SARKKA-2013, Chapter 4].

    """
    if not isinstance(model, LinearGaussianStateSpace):
        raise TypeError("model must be a LinearGaussianStateSpace")
    _require_state(state, model.state_dimension)
    mean = model.transition(state.mean, control)
    covariance = add(
        sandwich(model.transition_matrix, state.covariance), model.process_covariance
    )
    return GaussianState(mean, covariance)


def kalman_update(
    model: LinearGaussianStateSpace,
    predicted: GaussianState,
    observation: Sequence[float],
) -> MeasurementUpdate:
    """Condition a predicted state on one linear-Gaussian observation.

    Args:
        model: Linear-Gaussian model supplying ``H`` and ``R``.
        predicted: Predicted state at time ``k``.
        observation: Observation ``y_k``.

    Returns:
        Measurement update with the Joseph-form filtered covariance.

    Raises:
        TypeError: If the model or state has the wrong type.
        ValueError: If dimensions disagree.

    Equation:
        [EQ-FILT-KF-002], [EQ-FILT-KF-003], [EQ-FILT-KF-004].

    References:
        [REF-KALMAN-1960], [REF-SIMON-2006, Chapter 5].

    """
    if not isinstance(model, LinearGaussianStateSpace):
        raise TypeError("model must be a LinearGaussianStateSpace")
    _require_state(predicted, model.state_dimension)
    return linear_gaussian_update(
        predicted,
        observation,
        model.observe(predicted.mean),
        model.observation_matrix,
        model.observation_covariance,
    )


def kalman_filter(
    model: LinearGaussianStateSpace,
    observations: Sequence[Observation],
    *,
    controls: Sequence[Sequence[float] | None] | None = None,
) -> KalmanFilterResult:
    """Run the Kalman filter over an observation sequence.

    A ``None`` observation is treated as missing: the update is skipped, the
    filtered state equals the prediction, and the likelihood term is zero.

    Args:
        model: Linear-Gaussian model.
        observations: Observations ``y_1, ..., y_T`` (``None`` if missing).
        controls: Optional controls ``u_1, ..., u_T``.

    Returns:
        Filtered and predicted states, innovations, gains, and the
        innovation-form log marginal likelihood.

    Raises:
        TypeError: If the model has the wrong type.
        ValueError: If observations are empty or misshapen.

    Equation:
        [EQ-FILT-BAYES-001], [EQ-FILT-KF-001], [EQ-FILT-KF-002],
        [EQ-FILT-KF-003], [EQ-FILT-KF-004].

    References:
        [REF-KALMAN-1960], [REF-SARKKA-2013, Chapters 4 and 12].

    """
    if not isinstance(model, LinearGaussianStateSpace):
        raise TypeError("model must be a LinearGaussianStateSpace")
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
        predicted = kalman_predict(model, state, control)
        cross_covariances.append(
            matmul(state.covariance, transpose(model.transition_matrix))
        )
        predicted_states.append(predicted)
        if observation is None:
            state = predicted
            innovations.append(None)
            innovation_covariances.append(None)
            gains.append(None)
            terms.append(0.0)
        else:
            update = kalman_update(model, predicted, observation)
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
        "kalman",
    )


def rts_smoother(result: KalmanFilterResult) -> SmootherResult:
    """Run the Rauch-Tung-Striebel backward pass on a filter result.

    ``G_k = C_(k+1) (P^-_(k+1))^-1``,
    ``m^s_k = m_k + G_k (m^s_(k+1) - m^-_(k+1))``, and
    ``P^s_k = P_k + G_k (P^s_(k+1) - P^-_(k+1)) G_k'``, where
    ``C_(k+1) = Cov(x_k, x_(k+1) | y_1:k)`` equals ``P_k F'`` for linear
    dynamics. The same recursion smooths extended (Jacobian) and unscented
    (sigma-point cross-covariance) filter outputs.

    Args:
        result: Forward-pass record from a Gaussian filter.

    Returns:
        Smoothed states for ``k = 0..T`` and smoother gains.

    Raises:
        ValueError: If a predicted covariance is not positive definite.

    Equation:
        [EQ-FILT-RTS-001].

    References:
        [REF-RAUCH-TUNG-STRIEBEL-1965], [REF-SARKKA-2013, Chapters 8-9].

    """
    steps = result.steps
    smoothed: list[GaussianState] = [result.filtered[-1]]
    gains: list[Matrix] = []
    for k in range(steps - 1, -1, -1):
        filtered = result.filtered[k]
        predicted = result.predicted[k]
        following = smoothed[0]
        try:
            gain = gain_from_cross_covariance(
                result.cross_covariances[k], predicted.covariance
            )
        except ValueError as error:
            raise ValueError(
                f"predicted covariance at step {k + 1} is not positive definite"
            ) from error
        mean_shift = matvec(
            gain,
            tuple(a - b for a, b in zip(following.mean, predicted.mean, strict=True)),
        )
        mean = tuple(a + b for a, b in zip(filtered.mean, mean_shift, strict=True))
        covariance = add(
            filtered.covariance,
            sandwich(gain, add(following.covariance, predicted.covariance, scale=-1.0)),
        )
        smoothed.insert(0, GaussianState(mean, covariance))
        gains.insert(0, gain)
    return SmootherResult(tuple(smoothed), tuple(gains))


def steady_state_kalman(
    model: LinearGaussianStateSpace,
    *,
    tolerance: float = 1e-12,
    max_iterations: int = 10_000,
) -> SteadyStateKalman:
    """Iterate the discrete Riccati recursion to its fixed point.

    ``P^-_(k+1) = F (P^-_k - P^-_k H' S_k^-1 H P^-_k) F' + Q`` starting from
    ``P^-_1 = F P_0 F' + Q``. Convergence requires the usual detectability and
    stabilizability conditions; ``converged=False`` is reported rather than
    raised when the iteration limit is hit.

    Args:
        model: Time-invariant linear-Gaussian model.
        tolerance: Relative max-entry change that declares convergence.
        max_iterations: Maximum number of Riccati iterations.

    Returns:
        Steady-state predicted/filtered covariances and gain.

    Raises:
        TypeError: If the model has the wrong type.
        ValueError: If ``tolerance`` or ``max_iterations`` is invalid.

    Equation:
        [EQ-FILT-KF-005].

    References:
        [REF-SIMON-2006, Chapter 7], [REF-BAR-SHALOM-LI-KIRUBARAJAN-2001,
        Chapter 5].

    """
    if not isinstance(model, LinearGaussianStateSpace):
        raise TypeError("model must be a LinearGaussianStateSpace")
    if not tolerance > 0.0:
        raise ValueError("tolerance must be positive")
    if isinstance(max_iterations, bool) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    predicted = add(
        sandwich(model.transition_matrix, model.initial_covariance),
        model.process_covariance,
    )
    observation = model.observation_matrix
    zero = tuple(0.0 for _ in range(model.observation_dimension))
    for iteration in range(1, max_iterations + 1):
        update = linear_gaussian_update(
            GaussianState(model.initial_mean, predicted),
            zero,
            zero,
            observation,
            model.observation_covariance,
        )
        following = add(
            sandwich(model.transition_matrix, update.state.covariance),
            model.process_covariance,
        )
        scale = max(1.0, max(abs(v) for row in predicted for v in row))
        change = max(
            abs(a - b)
            for row_a, row_b in zip(following, predicted, strict=True)
            for a, b in zip(row_a, row_b, strict=True)
        )
        predicted = following
        if change <= tolerance * scale:
            final = linear_gaussian_update(
                GaussianState(model.initial_mean, predicted),
                zero,
                zero,
                observation,
                model.observation_covariance,
            )
            return SteadyStateKalman(
                predicted, final.state.covariance, final.gain, iteration, True
            )
    final = linear_gaussian_update(
        GaussianState(model.initial_mean, predicted),
        zero,
        zero,
        observation,
        model.observation_covariance,
    )
    return SteadyStateKalman(
        predicted, final.state.covariance, final.gain, max_iterations, False
    )
