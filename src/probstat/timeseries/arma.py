"""ARMA / ARIMA models: state-space likelihood, estimation, forecasting, simulation.

Model (Box-Jenkins sign convention)::

    phi(B) (1 - B)^d (y_t) = mu_d + theta(B) e_t,   e_t ~ N(0, sigma^2) i.i.d.,

with ``phi(z) = 1 - phi_1 z - ... - phi_p z^p`` and
``theta(z) = 1 + theta_1 z + ... + theta_q z^q``. Writing ``x_t`` for the
``d``-times differenced series, ``x_t - mu`` is a zero-mean ARMA(p, q) process.

**State-space form.** With ``r = max(p, q + 1)`` and ``phi_j = 0`` for
``j > p``, ``theta_j = 0`` for ``j > q``, the Harvey representation is

    alpha_t = T alpha_(t-1) + R e_t,   x_t - mu = Z alpha_t,

where ``T`` has ``(phi_1, ..., phi_r)`` in its first column and ones on the
super-diagonal, ``R = (1, theta_1, ..., theta_(r-1))'`` and
``Z = (1, 0, ..., 0)``. The stationary initial covariance solves the Lyapunov
equation ``P = T P T' + sigma^2 R R'``. The measurement equation has **no**
noise term, so the exact likelihood follows from the prediction-error
decomposition of the Kalman filter with a zero observation variance.
:class:`probstat.filtering.LinearGaussianStateSpace` requires a positive-
definite observation covariance, so :meth:`ARMAStateSpace.kalman` runs the
filter recursion itself while calling the same Joseph-form measurement update
:func:`probstat.filtering.linear_gaussian_update` that
:func:`probstat.filtering.kalman_filter` uses, and returns the same
:class:`probstat.filtering.KalmanFilterResult`. With a positive observation
variance the two paths are identical. Estimation uses a scalar
specialization of the same recursion (``Z`` selects the first state, so every
step is ``O(r^2)``) with ``sigma^2`` concentrated out.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from probstat.filtering import (
    GaussianState,
    KalmanFilterResult,
    LinearGaussianStateSpace,
    linear_gaussian_update,
)
from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    coerce_finite_sample,
    require_finite_real,
    require_non_negative_integer,
)
from probstat.inference import information_criteria
from probstat.linalg import (
    Matrix,
    add,
    inverse_spd,
    lu_solve,
    matmul,
    matvec,
    sandwich,
    transpose,
)
from probstat.modeling import standard_normal_quantile
from probstat.optimization import central_hessian, minimize_bfgs

from .descriptive import (
    LjungBoxResult,
    acf,
    autocovariance,
    difference,
    durbin_levinson,
    ljung_box,
)
from .polynomials import (
    RootCheck,
    ar_root_check,
    coefficients_to_pacf,
    is_stationary,
    ma_root_check,
    pacf_to_coefficients,
)

Vector = tuple[float, ...]
_LOG_TWO_PI = math.log(2.0 * math.pi)
_STEADY_TOLERANCE = 1e-13


def _coefficients(values: Sequence[float], name: str) -> Vector:
    """Materialize a possibly empty finite coefficient vector.

    Args:
        values: Candidate coefficients.
        name: Name used in error messages.

    Returns:
        Tuple of finite floats.

    Raises:
        TypeError: If a coefficient is not real.
        ValueError: If a coefficient is non-finite.

    References:
        Project validation policy. No equation is evaluated.

    """
    items = tuple(values)
    return coerce_finite_sample(items, name=name) if items else ()


def _positive(value: object, name: str) -> float:
    """Validate a finite positive real number.

    Args:
        value: Candidate value.
        name: Name used in error messages.

    Returns:
        The value as a float.

    Raises:
        TypeError: If the value is not real.
        ValueError: If the value is not finite and positive.

    References:
        Project validation policy. No equation is evaluated.

    """
    result = require_finite_real(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _harvey_matrices(ar: Vector, ma: Vector) -> tuple[Matrix, Vector]:
    """Return the Harvey transition matrix ``T`` and selection vector ``R``.

    Args:
        ar: AR coefficients ``phi_1..phi_p``.
        ma: MA coefficients ``theta_1..theta_q``.

    Returns:
        ``r`` by ``r`` transition and length-``r`` selection, ``r = max(p, q+1)``.

    Equation:
        [EQ-TS-SSM-001].

    References:
        [REF-DURBIN-KOOPMAN-2012, Chapter 3], [REF-HAMILTON-1994, Chapter 13].

    """
    size = max(len(ar), len(ma) + 1)
    phi = list(ar) + [0.0] * (size - len(ar))
    theta = [1.0] + list(ma) + [0.0] * (size - 1 - len(ma))
    transition = tuple(
        tuple(phi[i] if j == 0 else (1.0 if j == i + 1 else 0.0) for j in range(size))
        for i in range(size)
    )
    return transition, tuple(theta)


def _stationary_covariance(transition: Matrix, selection: Vector) -> Matrix:
    """Solve ``P = T P T' + R R'`` for the stationary state covariance.

    The symmetric unknowns ``P_ij`` (``i <= j``) satisfy a square linear system
    solved by pivoted Gaussian elimination.

    Args:
        transition: Stable transition matrix ``T``.
        selection: Selection vector ``R``.

    Returns:
        Symmetric stationary covariance for unit innovation variance.

    Raises:
        ValueError: If the system is singular (a unit-circle eigenvalue).

    Equation:
        [EQ-TS-SSM-001].

    References:
        [REF-DURBIN-KOOPMAN-2012, Chapter 5], [REF-HAMILTON-1994, Chapter 13].

    """
    size = len(transition)
    pairs = [(i, j) for i in range(size) for j in range(i, size)]
    index = {pair: position for position, pair in enumerate(pairs)}
    system: list[list[float]] = []
    right: list[float] = []
    for i, j in pairs:
        row = [0.0] * len(pairs)
        row[index[(i, j)]] += 1.0
        for k in range(size):
            tik = transition[i][k]
            if tik == 0.0:
                continue
            for m in range(size):
                tjm = transition[j][m]
                if tjm == 0.0:
                    continue
                key = (k, m) if k <= m else (m, k)
                row[index[key]] -= tik * tjm
        system.append(row)
        right.append(selection[i] * selection[j])
    solution = lu_solve(tuple(tuple(row) for row in system), tuple(right))
    return tuple(
        tuple(solution[index[(i, j) if i <= j else (j, i)]] for j in range(size))
        for i in range(size)
    )


@dataclass(frozen=True)
class ARMAStateSpace:
    """Harvey state-space form of a stationary Gaussian ARMA(p, q) process.

    Args:
        ar: AR coefficients ``phi_1..phi_p`` (stationary).
        ma: MA coefficients ``theta_1..theta_q``.
        variance: Innovation variance ``sigma^2``.
        mean: Process mean ``mu``.
        transition_matrix: ``T``.
        selection: ``R``.
        initial_covariance: Stationary ``Var(alpha_t) = sigma^2 P``.

    Equation:
        [EQ-TS-SSM-001].

    References:
        [REF-DURBIN-KOOPMAN-2012, Chapter 3], [REF-HAMILTON-1994, Chapter 13].

    """

    ar: Vector
    ma: Vector
    variance: float
    mean: float
    transition_matrix: Matrix
    selection: Vector
    initial_covariance: Matrix

    @property
    def state_dimension(self) -> int:
        """Return ``r = max(p, q + 1)``.

        References:
            [REF-DURBIN-KOOPMAN-2012, Chapter 3].

        """
        return len(self.selection)

    @property
    def process_covariance(self) -> Matrix:
        """Return ``sigma^2 R R'``.

        Equation:
            [EQ-TS-SSM-001].

        References:
            [REF-DURBIN-KOOPMAN-2012, Chapter 3].

        """
        return tuple(
            tuple(self.variance * a * b for b in self.selection) for a in self.selection
        )

    @property
    def observation_matrix(self) -> Matrix:
        """Return ``Z = (1, 0, ..., 0)`` as a one-row matrix.

        References:
            [REF-DURBIN-KOOPMAN-2012, Chapter 3].

        """
        return (tuple(1.0 if j == 0 else 0.0 for j in range(self.state_dimension)),)

    def to_linear_gaussian(
        self, observation_variance: float
    ) -> LinearGaussianStateSpace:
        """Return the model plus measurement noise as a filtering model.

        The returned model describes ``y_t - mu = Z alpha_t + v_t`` with
        ``v_t ~ N(0, observation_variance)``; feed it centred data.

        Args:
            observation_variance: Positive measurement-noise variance.

        Returns:
            :class:`probstat.filtering.LinearGaussianStateSpace`.

        Raises:
            ValueError: If ``observation_variance`` is not positive.

        Equation:
            [EQ-TS-SSM-001], [EQ-FILT-SSM-001].

        References:
            [REF-DURBIN-KOOPMAN-2012, Chapter 3].

        """
        noise = _positive(observation_variance, "observation_variance")
        return LinearGaussianStateSpace(
            transition_matrix=self.transition_matrix,
            process_covariance=self.process_covariance,
            observation_matrix=self.observation_matrix,
            observation_covariance=((noise,),),
            initial_mean=tuple(0.0 for _ in self.selection),
            initial_covariance=self.initial_covariance,
        )

    def kalman(
        self, series: Iterable[float], *, observation_variance: float = 0.0
    ) -> KalmanFilterResult:
        """Run the exact Kalman filter on ``y_t - mu`` and return the full record.

        The prediction step is ``a^- = T a``, ``P^- = T P T' + sigma^2 R R'``;
        the update calls :func:`probstat.filtering.linear_gaussian_update` with
        observation variance ``observation_variance`` (zero for a pure ARMA).

        Args:
            series: Observations ``y_1..y_n``.
            observation_variance: Non-negative measurement-noise variance.

        Returns:
            Filter record; ``log_likelihood`` is the exact Gaussian
            log-likelihood.

        Raises:
            ValueError: If the series is empty or the variance is negative.

        Equation:
            [EQ-TS-LIK-001], [EQ-FILT-KF-001], [EQ-FILT-KF-002],
            [EQ-FILT-KF-004].

        References:
            [REF-DURBIN-KOOPMAN-2012, Chapter 7], [REF-HAMILTON-1994, Chapter 13].

        """
        data = coerce_finite_sample(series, name="series")
        noise = require_finite_real(observation_variance, name="observation_variance")
        if noise < 0.0:
            raise ValueError("observation_variance must be non-negative")
        size = self.state_dimension
        transition = self.transition_matrix
        transition_t = transpose(transition)
        process = self.process_covariance
        observation = self.observation_matrix
        noise_matrix = ((noise,),)
        state = GaussianState(tuple(0.0 for _ in range(size)), self.initial_covariance)
        filtered = [state]
        predicted_states: list[GaussianState] = []
        cross: list[Matrix] = []
        innovations: list[Vector | None] = []
        innovation_covariances: list[Matrix | None] = []
        gains: list[Matrix | None] = []
        terms: list[float] = []
        for value in data:
            predicted = GaussianState(
                matvec(transition, state.mean),
                add(sandwich(transition, state.covariance), process),
            )
            cross.append(matmul(state.covariance, transition_t))
            predicted_states.append(predicted)
            update = linear_gaussian_update(
                predicted,
                (value - self.mean,),
                (predicted.mean[0],),
                observation,
                noise_matrix,
            )
            state = update.state
            filtered.append(state)
            innovations.append(update.innovation)
            innovation_covariances.append(update.innovation_covariance)
            gains.append(update.gain)
            terms.append(update.log_likelihood)
        return KalmanFilterResult(
            tuple(filtered),
            tuple(predicted_states),
            tuple(cross),
            tuple(innovations),
            tuple(innovation_covariances),
            tuple(gains),
            tuple(terms),
            "kalman-arma",
        )


def arma_state_space(
    ar: Sequence[float] = (),
    ma: Sequence[float] = (),
    *,
    variance: float = 1.0,
    mean: float = 0.0,
) -> ARMAStateSpace:
    """Build the Harvey state-space form of a stationary ARMA(p, q) model.

    Args:
        ar: AR coefficients ``phi_1..phi_p``.
        ma: MA coefficients ``theta_1..theta_q``.
        variance: Positive innovation variance ``sigma^2``.
        mean: Process mean ``mu``.

    Returns:
        State-space model with its stationary initial covariance.

    Raises:
        TypeError: If an argument is not real.
        ValueError: If the AR polynomial is not stationary or ``variance`` is
            not positive.

    Equation:
        [EQ-TS-ARMA-001], [EQ-TS-SSM-001].

    References:
        [REF-DURBIN-KOOPMAN-2012, Chapter 3], [REF-HAMILTON-1994, Chapter 13].

    """
    phi = _coefficients(ar, "ar")
    theta = _coefficients(ma, "ma")
    sigma2 = _positive(variance, "variance")
    mu = require_finite_real(mean, name="mean")
    if not is_stationary(phi):
        raise ValueError("AR polynomial is not stationary (a root has |z| <= 1)")
    transition, selection = _harvey_matrices(phi, theta)
    unit = _stationary_covariance(transition, selection)
    covariance = tuple(tuple(sigma2 * value for value in row) for row in unit)
    return ARMAStateSpace(phi, theta, sigma2, mu, transition, selection, covariance)


def arma_autocovariance(
    ar: Sequence[float], ma: Sequence[float], max_lag: int, *, variance: float = 1.0
) -> Vector:
    """Return the theoretical autocovariances ``gamma(0..max_lag)`` of an ARMA.

    ``gamma(k) = Z T^k P Z'`` with the stationary state covariance ``P``.

    Args:
        ar: Stationary AR coefficients.
        ma: MA coefficients.
        max_lag: Largest lag ``K >= 0``.
        variance: Innovation variance ``sigma^2``.

    Returns:
        Autocovariances for lags ``0..K``.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the AR polynomial is not stationary.

    Equation:
        [EQ-TS-SSM-001], [EQ-TS-ARMA-002].

    References:
        [REF-BROCKWELL-DAVIS-2016, Chapter 3], [REF-HAMILTON-1994, Chapter 3].

    """
    lag = require_non_negative_integer(max_lag, name="max_lag")
    model = arma_state_space(ar, ma, variance=variance)
    column = tuple(row[0] for row in model.initial_covariance)
    values = [column[0]]
    for _ in range(lag):
        column = matvec(model.transition_matrix, column)
        values.append(column[0])
    return tuple(values)


@dataclass(frozen=True, slots=True)
class _FilterOutput:
    """Unit-variance output of the scalar ARMA Kalman recursion.

    Args:
        sum_log_f: ``sum_t log F_t``.
        sum_squares: ``sum_t v_t^2 / F_t``.
        innovations: ``v_t`` (empty unless collected).
        variances: ``F_t`` (empty unless collected).
        state_mean: Predicted ``a_(n+1)``.
        state_covariance: Predicted ``P_(n+1)`` for unit ``sigma^2``.

    References:
        [REF-DURBIN-KOOPMAN-2012, Chapter 7].

    """

    sum_log_f: float
    sum_squares: float
    innovations: Vector
    variances: Vector
    state_mean: Vector
    state_covariance: Matrix


def _scalar_filter(
    centred: Sequence[float], ar: Vector, ma: Vector, *, collect: bool = False
) -> _FilterOutput:
    """Run the unit-variance Kalman filter specialized to ``Z = e_1``.

    Uses ``F_t = P_t[0][0]``, ``K_t = P_t e_1 / F_t`` and the companion
    structure of ``T`` so each step is ``O(r^2)``. Once the predicted
    covariance changes by less than ``1e-13`` (relative), it is frozen at its
    steady state (the residual change is below double-precision rounding).

    Args:
        centred: ``x_t - mu``.
        ar: Stationary AR coefficients.
        ma: MA coefficients.
        collect: Whether to keep per-step innovations and variances.

    Returns:
        Sufficient sums for the likelihood and the final predicted state.

    Raises:
        ValueError: If the AR part is not stationary or an innovation variance
            is not positive.

    Equation:
        [EQ-TS-SSM-001], [EQ-TS-LIK-001].

    References:
        [REF-DURBIN-KOOPMAN-2012, Chapters 4 and 7], [REF-HAMILTON-1994,
        Chapter 13].

    """
    if not is_stationary(ar):
        raise ValueError("AR polynomial is not stationary (a root has |z| <= 1)")
    transition, selection = _harvey_matrices(ar, ma)
    size = len(selection)
    phi = [transition[i][0] for i in range(size)]
    rr = [[a * b for b in selection] for a in selection]
    p_matrix = [list(row) for row in _stationary_covariance(transition, selection)]
    a = [0.0] * size
    sum_log_f = 0.0
    sum_squares = 0.0
    innovations: list[float] = []
    variances: list[float] = []
    steady = False
    gain: list[float] = []
    f_value = 0.0
    log_f = 0.0
    for value in centred:
        if not steady:
            f_value = p_matrix[0][0]
            if not f_value > 0.0:
                raise ValueError("innovation variance is not positive")
            gain = [p_matrix[i][0] / f_value for i in range(size)]
            log_f = math.log(f_value)
        v = value - a[0]
        sum_log_f += log_f
        sum_squares += v * v / f_value
        if collect:
            innovations.append(v)
            variances.append(f_value)
        filtered = [a[i] + gain[i] * v for i in range(size)]
        a = [
            phi[i] * filtered[0] + (filtered[i + 1] if i + 1 < size else 0.0)
            for i in range(size)
        ]
        if steady:
            continue
        row0 = p_matrix[0]
        pf = [
            [p_matrix[i][j] - gain[i] * row0[j] for j in range(size)]
            for i in range(size)
        ]
        tp = [
            [
                phi[i] * pf[0][j] + (pf[i + 1][j] if i + 1 < size else 0.0)
                for j in range(size)
            ]
            for i in range(size)
        ]
        new = [
            [
                tp[i][0] * phi[j] + (tp[i][j + 1] if j + 1 < size else 0.0) + rr[i][j]
                for j in range(size)
            ]
            for i in range(size)
        ]
        change = max(
            abs(new[i][j] - p_matrix[i][j]) for i in range(size) for j in range(size)
        )
        scale = max(1.0, max(abs(new[i][i]) for i in range(size)))
        p_matrix = new
        if change <= _STEADY_TOLERANCE * scale:
            steady = True
            f_value = p_matrix[0][0]
            if not f_value > 0.0:
                raise ValueError("innovation variance is not positive")
            gain = [p_matrix[i][0] / f_value for i in range(size)]
            log_f = math.log(f_value)
    return _FilterOutput(
        sum_log_f,
        sum_squares,
        tuple(innovations),
        tuple(variances),
        tuple(a),
        tuple(tuple(row) for row in p_matrix),
    )


def _full_log_likelihood(output: _FilterOutput, n: int, variance: float) -> float:
    """Return the exact log-likelihood for a given innovation variance.

    Args:
        output: Unit-variance filter output.
        n: Number of observations.
        variance: Innovation variance ``sigma^2``.

    Returns:
        ``-n/2 log 2 pi - 1/2 sum log(sigma^2 F_t) - sum v_t^2 / (2 sigma^2 F_t)``.

    Equation:
        [EQ-TS-LIK-001].

    References:
        [REF-DURBIN-KOOPMAN-2012, Chapter 7].

    """
    return -0.5 * (
        n * _LOG_TWO_PI
        + n * math.log(variance)
        + output.sum_log_f
        + output.sum_squares / variance
    )


def _concentrated(output: _FilterOutput, n: int) -> tuple[float, float]:
    """Return ``(sigma_hat^2, concentrated log-likelihood)``.

    Args:
        output: Unit-variance filter output.
        n: Number of observations.

    Returns:
        ``sigma_hat^2 = (1/n) sum v_t^2 / F_t`` and
        ``-n/2 (log 2 pi + 1 + log sigma_hat^2) - 1/2 sum log F_t``.

    Equation:
        [EQ-TS-LIK-001].

    References:
        [REF-DURBIN-KOOPMAN-2012, Chapter 7], [REF-HAMILTON-1994, Chapter 5].

    """
    sigma2 = output.sum_squares / n
    if not sigma2 > 0.0:
        raise ValueError("series has zero prediction error variance")
    value = -0.5 * (n * (_LOG_TWO_PI + 1.0 + math.log(sigma2)) + output.sum_log_f)
    return sigma2, value


def arma_log_likelihood(
    series: Iterable[float],
    ar: Sequence[float] = (),
    ma: Sequence[float] = (),
    *,
    variance: float = 1.0,
    mean: float = 0.0,
) -> float:
    """Return the exact Gaussian log-likelihood of a stationary ARMA(p, q).

    Prediction-error decomposition
    ``log L = -1/2 sum_t [log 2 pi + log(sigma^2 F_t) + v_t^2 / (sigma^2 F_t)]``
    with innovations ``v_t`` and unit variances ``F_t`` from the Kalman filter
    started at the stationary distribution.

    Args:
        series: Observations ``y_1..y_n``.
        ar: Stationary AR coefficients.
        ma: MA coefficients (non-invertible values are allowed).
        variance: Positive innovation variance.
        mean: Process mean.

    Returns:
        Exact log-likelihood.

    Raises:
        TypeError: If an argument is not real.
        ValueError: If the series is empty, the AR part is not stationary, or
            ``variance`` is not positive.

    Equation:
        [EQ-TS-LIK-001], [EQ-TS-SSM-001].

    References:
        [REF-DURBIN-KOOPMAN-2012, Chapter 7], [REF-HAMILTON-1994, Chapters 5
        and 13].

    """
    data = coerce_finite_sample(series, name="series")
    phi = _coefficients(ar, "ar")
    theta = _coefficients(ma, "ma")
    sigma2 = _positive(variance, "variance")
    mu = require_finite_real(mean, name="mean")
    output = _scalar_filter([value - mu for value in data], phi, theta)
    return _full_log_likelihood(output, len(data), sigma2)


@dataclass(frozen=True, slots=True)
class YuleWalkerResult:
    """Yule-Walker (method-of-moments) AR(p) estimate.

    Args:
        coefficients: ``phi_hat_1..phi_hat_p`` (always stationary).
        variance: ``sigma_hat^2 = gamma_hat(0) prod_k (1 - phi_kk^2)``.
        standard_errors: Asymptotic ``sqrt(diag(sigma^2 Gamma_p^-1) / n)``.
        mean: Sample mean removed before estimation.
        sample_size: ``n``.

    References:
        [REF-BROCKWELL-DAVIS-2016, Chapter 5], [REF-BOX-JENKINS-2015, Chapter 7].

    """

    coefficients: Vector
    variance: float
    standard_errors: Vector
    mean: float
    sample_size: int


def yule_walker(series: Iterable[float], order: int) -> YuleWalkerResult:
    """Estimate AR(p) coefficients by solving the Yule-Walker equations.

    ``Gamma_p phi = gamma_p`` is solved by Durbin-Levinson on the biased
    sample autocovariances, which guarantees a stationary estimate.

    Args:
        series: Observations.
        order: AR order ``p >= 1`` with ``p < n``.

    Returns:
        Coefficients, innovation variance, and asymptotic standard errors.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``order`` is invalid or the series is constant.

    Equation:
        [EQ-TS-YW-001], [EQ-TS-PACF-001].

    References:
        [REF-BROCKWELL-DAVIS-2016, Chapter 5], [REF-BOX-JENKINS-2015, Chapter 7].

    """
    data = coerce_finite_sample(series, name="series")
    p = require_non_negative_integer(order, name="order")
    if p < 1 or p >= len(data):
        raise ValueError("order must satisfy 1 <= order < len(series)")
    gamma = autocovariance(data, p)
    result = durbin_levinson(tuple(value / gamma[0] for value in gamma))
    coefficients = result.coefficients[-1]
    sigma2 = gamma[0] * result.prediction_variances[-1]
    toeplitz = tuple(tuple(gamma[abs(i - j)] for j in range(p)) for i in range(p))
    inverse = inverse_spd(toeplitz)
    n = len(data)
    errors = tuple(math.sqrt(sigma2 * inverse[i][i] / n) for i in range(p))
    return YuleWalkerResult(coefficients, sigma2, errors, math.fsum(data) / n, n)


@dataclass(frozen=True, slots=True)
class ARIMAForecast:
    """h-step forecasts with exact Gaussian prediction intervals.

    Args:
        mean: Point forecasts ``y_hat_(n+1..n+h)``.
        standard_errors: Forecast-error standard deviations.
        lower: Lower interval bounds.
        upper: Upper interval bounds.
        level: Two-sided interval level.

    References:
        [REF-DURBIN-KOOPMAN-2012, Chapter 4], [REF-BOX-JENKINS-2015, Chapter 5].

    """

    mean: Vector
    standard_errors: Vector
    lower: Vector
    upper: Vector
    level: float


@dataclass(frozen=True, slots=True)
class ResidualDiagnostics:
    """Residual checks for a fitted ARIMA model.

    Args:
        standardized_residuals: ``v_t / sqrt(sigma^2 F_t)``.
        autocorrelations: Residual ACF for lags ``0..lags``.
        ljung_box: Ljung-Box test with ``p + q`` fitted parameters removed.
        mean: Mean of the standardized residuals.
        variance: Variance of the standardized residuals.

    References:
        [REF-BOX-JENKINS-2015, Chapter 8], [REF-LJUNG-BOX-1978].

    """

    standardized_residuals: Vector
    autocorrelations: Vector
    ljung_box: LjungBoxResult
    mean: float
    variance: float


@dataclass(frozen=True)
class ARIMAFit:
    """Maximum-likelihood ARIMA(p, d, q) fit.

    Args:
        order: ``(p, d, q)``.
        ar: AR estimates.
        ma: MA estimates.
        mean: Estimated mean of the differenced series, or ``None``.
        variance: Innovation variance ``sigma_hat^2``.
        log_likelihood: Maximized exact log-likelihood of the differenced data.
        aic: Akaike information criterion.
        aicc: Small-sample corrected AIC.
        bic: Bayesian information criterion.
        parameter_names: Names aligned with the standard errors.
        estimates: Estimates aligned with ``parameter_names``.
        standard_errors: Observed-information standard errors or ``None``.
        covariance: Inverse observed information or ``None``.
        converged: Optimizer convergence flag.
        message: Optimizer termination message.
        evaluations: Objective evaluations used by the optimizer.
        series: Original observations.
        differenced: ``(1 - B)^d y``.
        innovations: One-step prediction errors ``v_t``.
        innovation_variances: ``sigma^2 F_t``.
        state_mean: Predicted state ``a_(n+1)``.
        state_covariance: Predicted state covariance ``sigma^2 P_(n+1)``.

    References:
        [REF-BOX-JENKINS-2015, Chapter 7], [REF-DURBIN-KOOPMAN-2012, Chapter 7].

    """

    order: tuple[int, int, int]
    ar: Vector
    ma: Vector
    mean: float | None
    variance: float
    log_likelihood: float
    aic: float
    aicc: float
    bic: float
    parameter_names: tuple[str, ...]
    estimates: Vector
    standard_errors: Vector | None
    covariance: Matrix | None
    converged: bool
    message: str
    evaluations: int
    series: Vector
    differenced: Vector
    innovations: Vector
    innovation_variances: Vector
    state_mean: Vector
    state_covariance: Matrix

    @property
    def ar_check(self) -> RootCheck:
        """Return the stationarity root check of the fitted AR polynomial.

        Equation:
            [EQ-TS-ROOT-001], [EQ-TS-ROOT-002].

        References:
            [REF-BROCKWELL-DAVIS-2016, Chapter 3].

        """
        return ar_root_check(self.ar)

    @property
    def ma_check(self) -> RootCheck:
        """Return the invertibility root check of the fitted MA polynomial.

        Equation:
            [EQ-TS-ROOT-001], [EQ-TS-ROOT-002].

        References:
            [REF-BROCKWELL-DAVIS-2016, Chapter 3].

        """
        return ma_root_check(self.ma)

    @property
    def standardized_residuals(self) -> Vector:
        """Return ``v_t / sqrt(sigma^2 F_t)``.

        Equation:
            [EQ-TS-LIK-001].

        References:
            [REF-DURBIN-KOOPMAN-2012, Chapter 7].

        """
        return tuple(
            v / math.sqrt(f)
            for v, f in zip(self.innovations, self.innovation_variances, strict=True)
        )

    def diagnostics(self, lags: int = 10) -> ResidualDiagnostics:
        """Return residual ACF and Ljung-Box diagnostics.

        Args:
            lags: Number of residual autocorrelations tested (``> p + q``).

        Returns:
            Residual diagnostics.

        Raises:
            ValueError: If ``lags`` does not exceed ``p + q`` or is too large.

        Equation:
            [EQ-TS-LB-001], [EQ-TS-ACF-001].

        References:
            [REF-BOX-JENKINS-2015, Chapter 8], [REF-LJUNG-BOX-1978].

        """
        residuals = self.standardized_residuals
        p, _, q = self.order
        test = ljung_box(residuals, lags, fitted_parameters=p + q)
        n = len(residuals)
        centre = math.fsum(residuals) / n
        spread = math.fsum((r - centre) ** 2 for r in residuals) / n
        return ResidualDiagnostics(
            residuals, acf(residuals, test.lags), test, centre, spread
        )

    def forecast(self, steps: int, *, level: float = 0.95) -> ARIMAForecast:
        """Forecast ``h`` steps ahead with exact Gaussian prediction intervals.

        See :func:`forecast_arima`.

        Args:
            steps: Horizon ``h >= 1``.
            level: Two-sided interval level.

        Returns:
            Forecast means, standard errors, and intervals.

        Raises:
            ValueError: If ``steps`` or ``level`` is invalid.

        Equation:
            [EQ-TS-FC-001].

        References:
            [REF-DURBIN-KOOPMAN-2012, Chapter 4], [REF-BOX-JENKINS-2015,
            Chapter 5].

        """
        return forecast_arima(self, steps, level=level)


def _transform(u: Sequence[float], p: int, q: int) -> tuple[Vector, Vector]:
    """Map unconstrained values to stationary AR and invertible MA coefficients.

    ``r = tanh(u)`` gives reflection coefficients in ``(-1, 1)``; the AR part
    is the step-up image of ``r_ar`` and the MA part is minus the step-up image
    of ``r_ma`` (so ``theta(z) = 1 - sum phi_j(r_ma) z^j``).

    Args:
        u: Unconstrained values (at least ``p + q``).
        p: AR order.
        q: MA order.

    Returns:
        AR and MA coefficients.

    Equation:
        [EQ-TS-PARAM-001].

    References:
        [REF-JONES-1980], [REF-MONAHAN-1984].

    """
    ar = pacf_to_coefficients([math.tanh(value) for value in u[:p]])
    ma_image = pacf_to_coefficients([math.tanh(value) for value in u[p : p + q]])
    return ar, tuple(-value for value in ma_image)


def _clipped_atanh(value: float, limit: float = 0.95) -> float:
    """Return ``atanh`` of a value clipped into ``[-limit, limit]``.

    Args:
        value: Value in ``(-1, 1)``.
        limit: Clipping bound below one.

    Returns:
        Unconstrained starting value.

    References:
        Project numerical policy. No equation is evaluated.

    """
    return math.atanh(max(-limit, min(limit, value)))


def fit_arima(
    series: Iterable[float],
    order: tuple[int, int, int],
    *,
    include_mean: bool | None = None,
    max_iterations: int = 200,
) -> ARIMAFit:
    """Fit an ARIMA(p, d, q) model by exact Gaussian maximum likelihood.

    The series is differenced ``d`` times; the ARMA(p, q) likelihood of the
    differenced data is evaluated exactly by the Kalman filter with
    ``sigma^2`` concentrated out, and maximized by BFGS over unconstrained
    values ``u`` with reflection coefficients ``tanh(u)`` (stationary AR and
    invertible MA by construction). Starting AR values are Yule-Walker
    estimates. Standard errors come from the inverse central-difference
    Hessian of the negative log-likelihood in ``(phi, theta, mu, log sigma^2)``
    (delta method for ``sigma^2``); they are ``None`` when the Hessian is not
    positive definite or a finite-difference step leaves the stationary region.

    Args:
        series: Observations ``y_1..y_n``.
        order: ``(p, d, q)`` with non-negative integers.
        include_mean: Estimate a mean (drift when ``d > 0``); default is
            ``True`` for ``d = 0`` and ``False`` otherwise.
        max_iterations: BFGS iteration limit.

    Returns:
        Fitted model with information criteria, standard errors, innovations,
        and the final predicted state for forecasting.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the order is invalid or too few observations remain.

    Equation:
        [EQ-TS-LIK-001], [EQ-TS-PARAM-001], [EQ-TS-SE-001], [EQ-INF-IC-001],
        [EQ-INF-IC-002], [EQ-INF-IC-003], [EQ-OPT-BFGS-001], [EQ-OPT-FD-002].

    References:
        [REF-BOX-JENKINS-2015, Chapter 7], [REF-DURBIN-KOOPMAN-2012, Chapter 7],
        [REF-JONES-1980], [REF-HAMILTON-1994, Chapter 5].

    """
    data = coerce_finite_sample(series, name="series")
    if len(order) != 3:
        raise ValueError("order must be a (p, d, q) triple")
    p = require_non_negative_integer(order[0], name="p")
    d = require_non_negative_integer(order[1], name="d")
    q = require_non_negative_integer(order[2], name="q")
    mean_flag = (d == 0) if include_mean is None else bool(include_mean)
    differenced = difference(data, d) if d else data
    n = len(differenced)
    parameter_count = p + q + int(mean_flag) + 1
    if n < parameter_count + 2:
        raise ValueError("too few observations for the requested order")
    centre = math.fsum(differenced) / n
    spread = math.sqrt(math.fsum((x - centre) ** 2 for x in differenced) / n)
    if spread == 0.0:
        raise ValueError("differenced series must not be constant")
    start = [0.0] * (p + q)
    if p:
        try:
            initial = yule_walker(differenced, p).coefficients
            reflections = coefficients_to_pacf(initial)
            if reflections is not None:
                start[:p] = [_clipped_atanh(r) for r in reflections]
        except ValueError:
            pass
    if mean_flag:
        start.append(0.0)

    def unpack(u: Sequence[float]) -> tuple[Vector, Vector, float]:
        """Return AR, MA, and mean from an unconstrained vector.

        References:
            [EQ-TS-PARAM-001], [REF-JONES-1980].

        """
        ar, ma = _transform(u, p, q)
        mu = centre + spread * u[p + q] if mean_flag else 0.0
        return ar, ma, mu

    def objective(u: Sequence[float]) -> float:
        """Return the negative concentrated log-likelihood per observation.

        References:
            [EQ-TS-LIK-001], [REF-DURBIN-KOOPMAN-2012, Chapter 7].

        """
        ar, ma, mu = unpack(u)
        output = _scalar_filter([x - mu for x in differenced], ar, ma)
        return -_concentrated(output, n)[1] / n

    if start:
        result = minimize_bfgs(
            objective, start, gradient_tolerance=1e-7, max_iterations=max_iterations
        )
        best = result.x
        converged, message, evaluations = (
            result.converged,
            result.message,
            result.evaluations,
        )
    else:
        best, converged, message, evaluations = (), True, "no free parameters", 0
    ar, ma, mu = unpack(best)
    output = _scalar_filter([x - mu for x in differenced], ar, ma, collect=True)
    sigma2, log_likelihood = _concentrated(output, n)
    names = (
        tuple(f"ar{i + 1}" for i in range(p))
        + tuple(f"ma{j + 1}" for j in range(q))
        + (("mean",) if mean_flag else ())
        + ("variance",)
    )
    estimates = ar + ma + ((mu,) if mean_flag else ()) + (sigma2,)

    def negative_log_likelihood(theta: Sequence[float]) -> float:
        """Return ``-log L`` in ``(phi, theta, mu, log sigma^2)`` coordinates.

        References:
            [EQ-TS-LIK-001], [REF-DURBIN-KOOPMAN-2012, Chapter 7].

        """
        phi = tuple(theta[:p])
        psi = tuple(theta[p : p + q])
        level = theta[p + q] if mean_flag else 0.0
        filtered = _scalar_filter([x - level for x in differenced], phi, psi)
        return -_full_log_likelihood(filtered, n, math.exp(theta[-1]))

    point = estimates[:-1] + (math.log(sigma2),)
    errors: Vector | None = None
    covariance: Matrix | None = None
    try:
        hessian, _ = central_hessian(negative_log_likelihood, point)
        inverse = inverse_spd(hessian)
        jacobian = [1.0] * (len(point) - 1) + [sigma2]
        covariance = tuple(
            tuple(jacobian[i] * inverse[i][j] * jacobian[j] for j in range(len(point)))
            for i in range(len(point))
        )
        errors = tuple(math.sqrt(covariance[i][i]) for i in range(len(point)))
    except ValueError:
        errors, covariance = None, None
    aic, aicc, bic = information_criteria(log_likelihood, parameter_count, n)
    return ARIMAFit(
        order=(p, d, q),
        ar=ar,
        ma=ma,
        mean=mu if mean_flag else None,
        variance=sigma2,
        log_likelihood=log_likelihood,
        aic=aic,
        aicc=aicc,
        bic=bic,
        parameter_names=names,
        estimates=estimates,
        standard_errors=errors,
        covariance=covariance,
        converged=converged,
        message=message,
        evaluations=evaluations,
        series=data,
        differenced=differenced,
        innovations=output.innovations,
        innovation_variances=tuple(sigma2 * f for f in output.variances),
        state_mean=output.state_mean,
        state_covariance=tuple(
            tuple(sigma2 * value for value in row) for row in output.state_covariance
        ),
    )


def forecast_arima(fit: ARIMAFit, steps: int, *, level: float = 0.95) -> ARIMAForecast:
    """Forecast a fitted ARIMA model with exact finite-sample intervals.

    From the filter's predicted state ``(a_(n+1), P_(n+1))`` the differenced
    process has conditional means ``mu + Z T^(j-1) a_(n+1)`` and covariances
    ``Cov(x_(n+i), x_(n+j)) = Z T^(j-i) P_(n+i|n) Z'`` with
    ``P_(n+i+1|n) = T P_(n+i|n) T' + sigma^2 R R'``. Undoing ``d`` differences
    gives ``y_(n+h) = c_h + sum_{j<=h} w_(h,j) x_(n+j)`` with
    ``w_(h,j) = binom(h - j + d - 1, d - 1)``, so the forecast-error variance is
    ``w_h' C w_h``. Parameter-estimation uncertainty is not included.

    Args:
        fit: Fitted model from :func:`fit_arima`.
        steps: Horizon ``h >= 1``.
        level: Two-sided interval level.

    Returns:
        Point forecasts, standard errors, and Gaussian intervals.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``steps`` is zero or ``level`` is invalid.

    Equation:
        [EQ-TS-FC-001].

    References:
        [REF-DURBIN-KOOPMAN-2012, Chapter 4], [REF-BOX-JENKINS-2015, Chapter 5],
        [REF-BROCKWELL-DAVIS-2016, Chapter 6].

    """
    if not isinstance(fit, ARIMAFit):
        raise TypeError("fit must be an ARIMAFit")
    h = require_non_negative_integer(steps, name="steps")
    if h < 1:
        raise ValueError("steps must be positive")
    confidence = require_finite_real(level, name="level")
    if not 0.0 < confidence < 1.0:
        raise ValueError("level must lie strictly between zero and one")
    _, d, _ = fit.order
    transition, selection = _harvey_matrices(fit.ar, fit.ma)
    size = len(selection)
    process = tuple(tuple(fit.variance * a * b for b in selection) for a in selection)
    mu = fit.mean if fit.mean is not None else 0.0
    state = fit.state_mean
    covariance = fit.state_covariance
    x_hat: list[float] = []
    covariances = [[0.0] * h for _ in range(h)]
    for i in range(h):
        x_hat.append(mu + state[0])
        column = tuple(covariance[k][0] for k in range(size))
        covariances[i][i] = column[0]
        for j in range(i + 1, h):
            column = matvec(transition, column)
            covariances[i][j] = covariances[j][i] = column[0]
        state = matvec(transition, state)
        covariance = add(sandwich(transition, covariance), process)
    history = list(fit.series)
    binomials = [math.comb(d, k) * (-1) ** k for k in range(d + 1)]
    means: list[float] = []
    for i in range(h):
        value = x_hat[i] - math.fsum(
            binomials[k] * history[-k] for k in range(1, d + 1)
        )
        history.append(value)
        means.append(value)
    z = standard_normal_quantile(0.5 + 0.5 * confidence)
    errors: list[float] = []
    for horizon in range(1, h + 1):
        if d == 0:
            weights = [1.0 if j == horizon - 1 else 0.0 for j in range(horizon)]
        else:
            weights = [
                float(math.comb(horizon - 1 - j + d - 1, d - 1)) for j in range(horizon)
            ]
        variance = math.fsum(
            weights[i] * weights[j] * covariances[i][j]
            for i in range(horizon)
            for j in range(horizon)
        )
        errors.append(math.sqrt(max(variance, 0.0)))
    return ARIMAForecast(
        tuple(means),
        tuple(errors),
        tuple(m - z * e for m, e in zip(means, errors, strict=True)),
        tuple(m + z * e for m, e in zip(means, errors, strict=True)),
        confidence,
    )


def simulate_arma(
    ar: Sequence[float],
    ma: Sequence[float],
    length: int,
    *,
    variance: float = 1.0,
    mean: float = 0.0,
    burn_in: int | None = None,
    seed: int | RandomStream | None = None,
) -> Vector:
    """Simulate a stationary Gaussian ARMA(p, q) path with a burn-in period.

    ``w_t = sum phi_i w_(t-i) + e_t + sum theta_j e_(t-j)`` starts from zeros;
    the first ``burn_in`` values are discarded and ``mu`` is added.

    Args:
        ar: Stationary AR coefficients.
        ma: MA coefficients.
        length: Number of returned observations ``n >= 1``.
        variance: Positive innovation variance.
        mean: Process mean.
        burn_in: Discarded warm-up steps (default ``200 + 10 (p + q)``).
        seed: Integer seed or :class:`RandomStream` (child ``"arma"``).

    Returns:
        Simulated observations.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If the AR part is not stationary, ``length`` is zero, or
            ``variance`` is not positive.

    Equation:
        [EQ-TS-SIM-001].

    References:
        [REF-BROCKWELL-DAVIS-2016, Chapter 3], [REF-BOX-JENKINS-2015, Chapter 3].

    """
    phi = _coefficients(ar, "ar")
    theta = _coefficients(ma, "ma")
    n = require_non_negative_integer(length, name="length")
    if n < 1:
        raise ValueError("length must be positive")
    sigma = math.sqrt(_positive(variance, "variance"))
    mu = require_finite_real(mean, name="mean")
    if not is_stationary(phi):
        raise ValueError("AR polynomial is not stationary (a root has |z| <= 1)")
    warm = (
        200 + 10 * (len(phi) + len(theta))
        if burn_in is None
        else require_non_negative_integer(burn_in, name="burn_in")
    )
    generator = resolve_stream(seed).child("arma").generator()
    total = warm + n
    noise = [generator.gauss(0.0, sigma) for _ in range(total)]
    path: list[float] = []
    for t in range(total):
        value = noise[t]
        for i, coefficient in enumerate(phi, start=1):
            if t - i >= 0:
                value += coefficient * path[t - i]
        for j, coefficient in enumerate(theta, start=1):
            if t - j >= 0:
                value += coefficient * noise[t - j]
        path.append(value)
    return tuple(mu + value for value in path[warm:])


def simulate_arima(
    ar: Sequence[float],
    d: int,
    ma: Sequence[float],
    length: int,
    *,
    variance: float = 1.0,
    drift: float = 0.0,
    burn_in: int | None = None,
    seed: int | RandomStream | None = None,
) -> Vector:
    """Simulate an ARIMA(p, d, q) path by integrating a simulated ARMA path.

    The differenced series ``(1 - B)^d y`` is an ARMA(p, q) path with mean
    ``drift``; integration starts from ``d`` zeros, which are not returned.

    Args:
        ar: Stationary AR coefficients.
        d: Integration order.
        ma: MA coefficients.
        length: Number of returned observations.
        variance: Positive innovation variance.
        drift: Mean of the differenced series.
        burn_in: ARMA warm-up steps.
        seed: Integer seed or :class:`RandomStream`.

    Returns:
        Simulated observations ``y_1..y_n``.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an argument is invalid.

    Equation:
        [EQ-TS-SIM-001], [EQ-TS-DIFF-001].

    References:
        [REF-BOX-JENKINS-2015, Chapter 4].

    """
    order = require_non_negative_integer(d, name="d")
    increments = simulate_arma(
        ar, ma, length, variance=variance, mean=drift, burn_in=burn_in, seed=seed
    )
    values = list(increments)
    for _ in range(order):
        total = 0.0
        integrated: list[float] = []
        for value in values:
            total += value
            integrated.append(total)
        values = integrated
    return tuple(values)
