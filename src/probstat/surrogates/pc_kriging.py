"""Universal kriging and polynomial-chaos Kriging (PC-Kriging).

Universal kriging models a response as a regression trend plus a stationary
Gaussian process, ``Y(x) = f(x)' beta + sigma^2 Z(x)`` with ``Z`` of unit
variance and correlation ``R(x, x'; ell)``. Given the correlation
hyperparameters, the trend coefficients follow by generalized least squares
and the process variance has a closed-form maximum likelihood estimate, so
the likelihood is maximized over the length scales alone (the *concentrated*
likelihood). Ordinary kriging is the special case of a constant regressor.

PC-Kriging (Schöbi, Sudret and Wiart 2015) takes the trend from a sparse
polynomial chaos expansion: the orthonormal polynomials selected by
least-angle regression capture the global behaviour, and the Gaussian process
interpolates the local residual. Two variants are provided:

* ``"sequential"``: the hybrid-LAR sparse PCE chooses the polynomial set, and
  one universal kriging model is fitted with that set as its trend;
* ``"optimal"``: the polynomials are added one at a time in the order LAR
  ranked them, a universal kriging model is fitted for each trend size, and
  the model with the smallest leave-one-out error is kept. Its last
  candidate is the sequential model, fitted identically, so its leave-one-out
  error is never larger.

The polynomials and the correlation function both act on the standardized
PCE variables (``xi in [-1, 1]^d`` for the Legendre basis, the independent
standard-normal ``u`` of the Nataf transformation for the Hermite basis).
This keeps the length scales of different inputs on a common, dimensionless
scale and lets the trend and the residual process share one coordinate
system; it is a modelling choice, the published method also allows physical
coordinates for the correlation.

All algebra is dense pure Python (the reference routines of
:mod:`probstat.surrogates.gaussian_process`), whatever the active
:func:`probstat.linalg.get_backend`; a likelihood evaluation costs about
``N^3 / 2`` multiply-adds for ``N`` training runs.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.linalg import Vector, as_vector
from probstat.modeling import JointInput, ProbabilisticModel
from probstat.optimization import minimize_bfgs
from probstat.propagation import make_design

from .gaussian_process import (
    KERNELS,
    GaussianProcessConditioningWarning,
    _condition_number,
    _correlation,
    _fast_cholesky,
    _inverse_factor_columns,
    _sumprod,
)
from .polynomial_chaos import (
    MultiIndex,
    PolynomialChaosExpansion,
    _basis_row,
    _check_basis,
)
from .sparse_chaos import sparse_polynomial_chaos_from_samples

PC_KRIGING_VARIANTS = ("sequential", "optimal")

# Computational choices (not published constants). Length scales are searched
# in ``[0.01 E_i, 20 E_i]`` with ``E_i`` the extent of input ``i`` over the
# training points (the upper factor matches GaussianProcessRegularization);
# an estimated nugget lies in ``[nugget, 1]`` relative to the process
# variance. A GLS residual whose largest entry is below ``1e-10`` times the
# largest absolute response counts as an exact trend: the length scales then
# stay at the first start, because the likelihood carries no information
# about a residual process that is identically zero. A whitened trend column
# whose squared sine to the span of the earlier ones is below ``1e-12`` makes
# the trend rank deficient.
_LOWER_EXTENT_FACTOR = 0.01
_UPPER_EXTENT_FACTOR = 20.0
_EXACT_TREND_TOLERANCE = 1e-10
_DEPENDENCE_TOLERANCE = 1e-12
_VARIANCE_FLOOR = 1e-300
_CONDITION_WARNING = 1e12
_CACHE_SIZE = 64
_SQRT3 = math.sqrt(3.0)
_SQRT5 = math.sqrt(5.0)

Regressors = Callable[[Vector], Sequence[float]] | Sequence[Callable[[Vector], float]]


def _forward(lower: Sequence[Sequence[float]], target: Sequence[float]) -> list[float]:
    """Solve ``L w = b`` for the ragged lower factor rows ``L[i][:i+1]``.

    Equation:
        [EQ-LINALG-002].

    References:
        [REF-GOLUB-VANLOAN-2013, Section 3.1].

    """
    result: list[float] = []
    for i, row in enumerate(lower):
        result.append((target[i] - _sumprod(row[:i], result)) / row[i])
    return result


def _backward(lower: Sequence[Sequence[float]], target: Sequence[float]) -> list[float]:
    """Solve ``L' x = b`` for the ragged lower factor rows ``L[i][:i+1]``.

    Equation:
        [EQ-LINALG-002].

    References:
        [REF-GOLUB-VANLOAN-2013, Section 3.1].

    """
    count = len(lower)
    result = [0.0] * count
    for i in range(count - 1, -1, -1):
        total = target[i]
        for k in range(i + 1, count):
            total -= lower[k][i] * result[k]
        result[i] = total / lower[i][i]
    return result


def _resolve_regressors(regressors: Regressors | None) -> Callable[[Vector], Vector]:
    """Return ``x -> f(x)`` as a tuple-valued function.

    ``None`` is the constant regressor (ordinary kriging); a callable is used
    as the vector function; a sequence of callables is evaluated term by
    term.

    Raises:
        TypeError: If ``regressors`` is neither callable nor a sequence of
            callables.

    References:
        [REF-SCHOBI-SUDRET-WIART-2015]. No equation is evaluated.

    """
    if regressors is None:
        return _constant_regressor
    if callable(regressors):
        function = regressors

        def vector(x: Vector) -> Vector:
            """Evaluate the user's regressor vector.

            References:
                [REF-SCHOBI-SUDRET-WIART-2015]. No equation is evaluated.

            """
            return tuple(float(value) for value in function(x))

        return vector
    terms = tuple(regressors)
    if not terms or not all(callable(term) for term in terms):
        raise TypeError("regressors must be callable or a sequence of callables")

    def stacked(x: Vector) -> Vector:
        """Evaluate each regressor callable.

        References:
            [REF-SCHOBI-SUDRET-WIART-2015]. No equation is evaluated.

        """
        return tuple(float(term(x)) for term in terms)

    return stacked


def _constant_regressor(x: Vector) -> Vector:
    """Return ``(1,)``, the ordinary-kriging regressor.

    References:
        [REF-SCHOBI-SUDRET-WIART-2015]. No equation is evaluated.

    """
    return (1.0,)


@dataclass(frozen=True, slots=True)
class ChaosTrend:
    """Orthonormal polynomial regressors ``Psi_alpha(xi)`` for a kriging trend.

    Called on a standardized point ``xi`` it returns ``Psi_alpha(xi)`` for
    every multi-index, in the stored order.

    Args:
        basis: ``"hermite"`` or ``"legendre"``.
        indices: Multi-indices of the trend polynomials.

    References:
        [REF-SCHOBI-SUDRET-WIART-2015], [REF-XIU-KARNIADAKIS-2002].

    """

    basis: str
    indices: tuple[MultiIndex, ...]

    def __call__(self, xi: Vector) -> Vector:
        """Evaluate every trend polynomial at a standardized point.

        Args:
            xi: Standardized input vector.

        Returns:
            One value per multi-index.

        Equation:
            [EQ-SUR-PCE-001].

        References:
            [REF-XIU-KARNIADAKIS-2002].

        """
        degree = max((max(alpha, default=0) for alpha in self.indices), default=0)
        return _basis_row(xi, self.basis, self.indices, degree)


@dataclass(frozen=True, slots=True)
class KrigingLOO:
    """Closed-form leave-one-out diagnostics of a universal kriging model.

    Args:
        means: LOO predictive means ``mu_(-i)``.
        variances: LOO predictive variances ``sigma^2_(-i)``.
        residuals: ``y_i - mu_(-i)``.
        relative_error: ``mean(residual^2) / Var(y)`` with the sample
            variance (divisor ``N - 1``).

    References:
        [REF-SCHOBI-SUDRET-WIART-2015].

    """

    means: Vector
    variances: Vector
    residuals: Vector
    relative_error: float


@dataclass(frozen=True, slots=True)
class _KrigingState:
    """Factorizations of a fitted universal kriging model.

    References:
        [REF-SCHOBI-SUDRET-WIART-2015]. No new equation.

    """

    lower: tuple[Vector, ...]
    whitened_columns: tuple[Vector, ...]
    gram_lower: tuple[Vector, ...]
    beta: Vector
    alpha: Vector
    process_variance: float
    log_det: float
    exact_trend: bool


@dataclass(frozen=True, slots=True)
class _PairData:
    """Squared coordinate differences of every training pair ``a > b``.

    ``pairs[k]`` holds ``((x_a1 - x_b1)^2, ..., (x_ad - x_bd)^2)`` for the
    ``k``-th pair in row-major lower-triangle order; ``axes[i]`` holds the
    same numbers for input ``i`` across all pairs. Both depend only on the
    design, so they are formed once per fit.

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Section 4.2]. No new equation.

    """

    count: int
    pairs: tuple[Vector, ...]
    axes: tuple[Vector, ...]


def _pair_data(points: Sequence[Vector]) -> _PairData:
    """Return the squared coordinate differences of the training pairs.

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Section 4.2]. No new equation.

    """
    pairs: list[Vector] = []
    for a, left in enumerate(points):
        for b in range(a):
            pairs.append(
                tuple((p - q) ** 2 for p, q in zip(left, points[b], strict=True))
            )
    axes = tuple(tuple(pair[axis] for pair in pairs) for axis in range(len(points[0])))
    return _PairData(len(points), tuple(pairs), axes)


def _correlation_rows(
    data: _PairData, scales: Sequence[float], nugget: float, kernel: str
) -> tuple[list[list[float]], list[float]]:
    """Return the lower-triangle rows of ``R`` and the derivative factors.

    ``R_ab = rho(r_ab)`` with ``r_ab^2 = sum_i (x_ai - x_bi)^2 / ell_i^2``
    and ``R_aa = 1 + nugget``; the factor ``g_ab`` gives
    ``dR_ab / d log ell_i = g_ab (x_ai - x_bi)^2 / ell_i^2``.

    Equation:
        [EQ-SUR-GP-001], [EQ-SUR-GP-002], [EQ-SUR-GP-008], [EQ-SUR-UK-002].

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Sections 4.2 and 5.4].

    """
    inverse = tuple(1.0 / (ell * ell) for ell in scales)
    squared = [_sumprod(pair, inverse) for pair in data.pairs]
    values: list[float] = []
    factors: list[float] = []
    if kernel == "squared_exponential":
        values = [math.exp(-0.5 * s) for s in squared]
        factors = values
    elif kernel == "matern32":
        for s in squared:
            r = math.sqrt(s)
            decay = math.exp(-_SQRT3 * r)
            values.append((1.0 + _SQRT3 * r) * decay)
            factors.append(3.0 * decay)
    else:
        for s in squared:
            r = math.sqrt(s)
            decay = math.exp(-_SQRT5 * r)
            values.append((1.0 + _SQRT5 * r + 5.0 * s / 3.0) * decay)
            factors.append((5.0 / 3.0) * (1.0 + _SQRT5 * r) * decay)
    rows: list[list[float]] = []
    start = 0
    diagonal = 1.0 + nugget
    for a in range(data.count):
        row = values[start : start + a]
        row.append(diagonal)
        rows.append(row)
        start += a
    return rows, factors


def _evaluate_state(
    data: _PairData,
    values: Vector,
    regressor_rows: Sequence[Vector],
    scales: Sequence[float],
    nugget: float,
    kernel: str,
) -> tuple[_KrigingState, list[list[float]], list[float]]:
    """Factorize ``R``, solve the GLS trend, and profile the process variance.

    ``R = L L'``; ``F~ = L^-1 F``, ``y~ = L^-1 y``;
    ``beta = (F~'F~)^-1 F~'y~``; ``sigma^2 = ||y~ - F~ beta||^2 / N``;
    ``alpha = R^-1 (y - F beta)``.

    Returns:
        The state, the lower triangle of ``R``, and the length-scale
        derivative factors.

    Raises:
        ValueError: If ``R`` is not positive definite or the trend columns
            are numerically dependent under ``R``.

    Equation:
        [EQ-SUR-UK-001].

    References:
        [REF-SCHOBI-SUDRET-WIART-2015].

    """
    rows, factors = _correlation_rows(data, scales, nugget, kernel)
    try:
        lower = _fast_cholesky(rows)
    except ValueError as error:
        raise ValueError(
            "correlation matrix is not positive definite; increase the nugget"
        ) from error
    count = len(values)
    terms = len(regressor_rows[0])
    columns = [
        _forward(lower, [row[j] for row in regressor_rows]) for j in range(terms)
    ]
    whitened_y = _forward(lower, values)
    gram = [
        [_sumprod(columns[a], columns[b]) for b in range(a + 1)] for a in range(terms)
    ]
    try:
        gram_lower = _fast_cholesky(gram)
    except ValueError as error:
        raise ValueError(
            "the trend regressors are numerically dependent on the design"
        ) from error
    if any(
        gram_lower[a][a] ** 2 <= _DEPENDENCE_TOLERANCE * gram[a][a]
        for a in range(terms)
    ):
        raise ValueError("the trend regressors are numerically dependent on the design")
    projected = [_sumprod(column, whitened_y) for column in columns]
    beta = _backward(gram_lower, _forward(gram_lower, projected))
    residual = [
        whitened_y[i] - math.fsum(columns[j][i] * beta[j] for j in range(terms))
        for i in range(count)
    ]
    raw_residual = [values[i] - _sumprod(regressor_rows[i], beta) for i in range(count)]
    largest = max(abs(value) for value in values)
    exact = max(abs(value) for value in raw_residual) <= _EXACT_TREND_TOLERANCE * max(
        largest, _VARIANCE_FLOOR
    )
    variance = max(_sumprod(residual, residual) / count, _VARIANCE_FLOOR)
    alpha = _backward(lower, residual)
    log_det = 2.0 * math.fsum(math.log(lower[i][i]) for i in range(count))
    state = _KrigingState(
        tuple(tuple(row) for row in lower),
        tuple(tuple(column) for column in columns),
        tuple(tuple(row) for row in gram_lower),
        tuple(beta),
        tuple(alpha),
        variance,
        log_det,
        exact,
    )
    return state, rows, factors


def _objective(
    data: _PairData,
    values: Vector,
    regressor_rows: Sequence[Vector],
    theta: Sequence[float],
    dimension: int,
    nugget: float,
    estimate_nugget: bool,
    kernel: str,
) -> tuple[float, list[float]]:
    """Return ``(N log sigma^2 + log det R) / 2`` and its exact gradient.

    With ``alpha = R^-1 (y - F beta)`` and ``W = R^-1 - alpha alpha' /
    sigma^2`` the gradient is ``(1/2) tr(W dR/d theta_j)``: by the envelope
    theorem the GLS coefficients do not contribute, because they minimize
    the quadratic form. The entries of ``R^-1`` come from one triangular
    inversion of the Cholesky factor [EQ-SUR-GP-007].

    Raises:
        ValueError: If ``R`` is not positive definite.

    Equation:
        [EQ-SUR-UK-002], [EQ-SUR-GP-007].

    References:
        [REF-SCHOBI-SUDRET-WIART-2015], [REF-RASMUSSEN-WILLIAMS-2006, Section 5.4].

    """
    scales = tuple(math.exp(value) for value in theta[:dimension])
    extra = math.exp(theta[dimension]) if estimate_nugget else 0.0
    state, _, factors = _evaluate_state(
        data, values, regressor_rows, scales, nugget + extra, kernel
    )
    count = len(values)
    value = 0.5 * (count * math.log(state.process_variance) + state.log_det)
    columns = _inverse_factor_columns(state.lower)
    alpha = state.alpha
    inverse_variance = 1.0 / state.process_variance
    weighted: list[float] = []
    position = 0
    for a in range(count):
        column_a = columns[a]
        alpha_a = alpha[a] * inverse_variance
        for b in range(a):
            weight = _sumprod(columns[b][a - b :], column_a) - alpha[b] * alpha_a
            weighted.append(weight * factors[position])
            position += 1
    gradient = [
        _sumprod(weighted, axis) / (ell * ell)
        for axis, ell in zip(data.axes, scales, strict=True)
    ]
    if estimate_nugget:
        trace = math.fsum(
            _sumprod(columns[a], columns[a]) - alpha[a] * alpha[a] * inverse_variance
            for a in range(count)
        )
        gradient.append(0.5 * extra * trace)
    return value, gradient


@dataclass(frozen=True)
class UniversalKriging:
    """A fitted universal kriging model ``f(x)' beta + GP(0, sigma^2 R)``.

    Args:
        kernel: Correlation kernel name (one of ``KERNELS``).
        points: Training inputs in the kriging coordinates.
        values: Training responses.
        regressor: The trend function ``x -> f(x)``.
        length_scales: Fitted (or fixed) ARD length scales ``ell_i``.
        nugget: Diagonal of ``R`` minus one, relative to the process
            variance (fixed nugget plus any estimated part).
        trend_coefficients: GLS coefficients ``beta``.
        process_variance: Profiled process variance ``sigma^2``.
        log_likelihood: Concentrated log-likelihood
            ``-(N/2)(log(2 pi sigma^2) + 1) - (1/2) log det R``.
        converged: Whether the best optimizer start met its tolerance
            (``True`` when the hyperparameters were fixed).
        restarts: Number of optimizer starts.
        evaluations: Likelihood evaluations (factorizations of ``R``).
        log_parameter_bounds: Box searched for ``log ell_i`` (and the log
            estimated nugget).
        at_bound: Parameters within ``0.01`` (log units) of a bound, as
            ``"length_scale[0]:upper"`` and similar.
        exact_trend: Whether the trend reproduced the responses to
            ``1e-10`` relative (the length scales then stay at the first
            start).
        condition_number: 1-norm condition number of ``R`` (computed).
        loo_error: Relative closed-form leave-one-out error (computed).

    References:
        [REF-SCHOBI-SUDRET-WIART-2015],
        [REF-RASMUSSEN-WILLIAMS-2006, Chapter 2].

    """

    kernel: str
    points: tuple[Vector, ...]
    values: Vector
    regressor: Callable[[Vector], Vector]
    length_scales: Vector
    nugget: float
    trend_coefficients: Vector
    process_variance: float
    log_likelihood: float
    converged: bool
    restarts: int
    evaluations: int
    log_parameter_bounds: tuple[tuple[float, float], ...] = ()
    at_bound: tuple[str, ...] = ()
    exact_trend: bool = False
    _state: _KrigingState = field(init=False, repr=False, compare=False)
    condition_number: float = field(init=False, compare=False, default=math.nan)
    loo_error: float = field(init=False, compare=False, default=math.nan)

    def __post_init__(self) -> None:
        """Factorize the correlation matrix once for predictions and LOO.

        Raises:
            ValueError: If the correlation matrix is not positive definite or
                the trend regressors are dependent.

        Equation:
            [EQ-SUR-UK-001], [EQ-SUR-GP-007].

        References:
            [REF-SCHOBI-SUDRET-WIART-2015], [REF-GOLUB-VANLOAN-2013, Section 2.6].

        """
        rows_f = tuple(self.regressor(point) for point in self.points)
        state, rows, _ = _evaluate_state(
            _pair_data(self.points),
            self.values,
            rows_f,
            self.length_scales,
            self.nugget,
            self.kernel,
        )
        object.__setattr__(self, "_state", state)
        object.__setattr__(
            self, "condition_number", _condition_number(rows, state.lower)
        )
        object.__setattr__(self, "loo_error", self.leave_one_out().relative_error)

    @property
    def size(self) -> int:
        """Return the number of trend regressors ``P``.

        References:
            [REF-SCHOBI-SUDRET-WIART-2015].

        """
        return len(self.trend_coefficients)

    def trend(self, x: Sequence[float]) -> float:
        """Return the regression part ``f(x)' beta``.

        Args:
            x: Input vector in the kriging coordinates.

        Returns:
            Trend value.

        Equation:
            [EQ-SUR-UK-003].

        References:
            [REF-SCHOBI-SUDRET-WIART-2015].

        """
        point = as_vector(x, name="x")
        return _sumprod(self.regressor(point), self.trend_coefficients)

    def predict_with_variance(
        self, x: Sequence[float], *, include_noise: bool = False
    ) -> tuple[float, float]:
        """Return the kriging mean and variance at one input.

        ``mu = f' beta + r' R^-1 (y - F beta)`` and
        ``s^2 = sigma^2 (1 - r'R^-1 r + u'(F'R^-1F)^-1 u)``,
        ``u = F'R^-1 r - f``: the last term is the variance added by
        estimating ``beta`` (the block-matrix formula
        ``sigma^2 (1 - [f' r'] [[0, F'], [F, R]]^-1 [f; r])``). Tiny negative
        values from cancellation are reported as zero.

        Args:
            x: Input vector in the kriging coordinates.
            include_noise: Add the nugget variance ``sigma^2 nugget``
                (prediction of a new observation).

        Returns:
            Predictive mean and variance.

        Raises:
            ValueError: If ``x`` has the wrong length.

        Equation:
            [EQ-SUR-UK-003].

        References:
            [REF-SCHOBI-SUDRET-WIART-2015],
            [REF-RASMUSSEN-WILLIAMS-2006, Chapter 2].

        """
        point = as_vector(x, name="x")
        if len(point) != len(self.points[0]):
            raise ValueError("x must have the training input dimension")
        state = self._state
        inverse = tuple(1.0 / (ell * ell) for ell in self.length_scales)
        cross = [
            _correlation(
                self.kernel,
                _sumprod(
                    [(a - b) ** 2 for a, b in zip(point, train, strict=True)], inverse
                ),
            )
            for train in self.points
        ]
        regressors = self.regressor(point)
        mean = _sumprod(regressors, state.beta) + _sumprod(cross, state.alpha)
        whitened = _forward(state.lower, cross)
        excess = [
            _sumprod(column, whitened) - f
            for column, f in zip(state.whitened_columns, regressors, strict=True)
        ]
        trend_part = _forward(state.gram_lower, excess)
        scaled = 1.0 - _sumprod(whitened, whitened) + _sumprod(trend_part, trend_part)
        variance = state.process_variance * max(scaled, 0.0)
        if include_noise:
            variance += state.process_variance * self.nugget
        return mean, variance

    def predict(self, x: Sequence[float]) -> float:
        """Return the kriging mean at one input.

        Args:
            x: Input vector in the kriging coordinates.

        Returns:
            Predictive mean.

        Equation:
            [EQ-SUR-UK-003].

        References:
            [REF-SCHOBI-SUDRET-WIART-2015].

        """
        return self.predict_with_variance(x)[0]

    def leave_one_out(self) -> KrigingLOO:
        """Return closed-form LOO predictions at fixed hyperparameters.

        With ``B = [[sigma^2 R, F], [F', 0]]^-1`` the top-left block is
        ``P / sigma^2``, ``P = R^-1 - R^-1 F (F'R^-1F)^-1 F'R^-1`` and
        ``P y = alpha``, so ``mu_(-i) = y_i - alpha_i / P_ii`` and
        ``sigma^2_(-i) = sigma^2 / P_ii``. The trend coefficients are
        re-estimated in every fold; the length scales and ``sigma^2`` are not.

        Returns:
            LOO means, variances, residuals, and relative error.

        Equation:
            [EQ-SUR-UK-004].

        References:
            [REF-SCHOBI-SUDRET-WIART-2015].

        """
        state = self._state
        count = len(self.points)
        columns = _inverse_factor_columns(state.lower)
        transposed = [
            _backward(state.lower, column) for column in state.whitened_columns
        ]
        means: list[float] = []
        variances: list[float] = []
        residuals: list[float] = []
        for i in range(count):
            precision = _sumprod(columns[i], columns[i])
            row = [column[i] for column in transposed]
            reduced = _forward(state.gram_lower, row)
            diagonal = precision - _sumprod(reduced, reduced)
            mean = self.values[i] - state.alpha[i] / diagonal
            means.append(mean)
            variances.append(state.process_variance / diagonal)
            residuals.append(self.values[i] - mean)
        center = math.fsum(self.values) / count
        spread = math.fsum((v - center) ** 2 for v in self.values) / (count - 1)
        error = math.fsum(r * r for r in residuals) / count
        return KrigingLOO(
            tuple(means),
            tuple(variances),
            tuple(residuals),
            error / spread if spread > 0.0 else math.inf,
        )


def _extents(points: Sequence[Vector]) -> list[float]:
    """Return the extent of every input over the training points.

    Raises:
        ValueError: If an input is constant.

    References:
        [EQ-SUR-UK-002]. No new equation.

    """
    extents: list[float] = []
    for axis in range(len(points[0])):
        column = [point[axis] for point in points]
        width = max(column) - min(column)
        if width <= 0.0:
            raise ValueError(f"input {axis} is constant across the training points")
        extents.append(width)
    return extents


def _kriging_bounds(
    extents: Sequence[float], nugget: float, estimate_nugget: bool
) -> tuple[tuple[float, float], ...]:
    """Return the box for ``(log ell_1..d[, log nugget_extra])``.

    References:
        [EQ-SUR-UK-002]. No new equation.

    """
    bounds = [
        (math.log(_LOWER_EXTENT_FACTOR * width), math.log(_UPPER_EXTENT_FACTOR * width))
        for width in extents
    ]
    if estimate_nugget:
        bounds.append((math.log(nugget), 0.0))
    return tuple(bounds)


def _kriging_starts(
    extents: Sequence[float],
    bounds: Sequence[tuple[float, float]],
    restarts: int,
    estimate_nugget: bool,
    seed: int | RandomStream | None,
) -> tuple[Vector, ...]:
    """Return the optimizer starts, clipped to the inner 98 % of the box.

    The first start has ``ell_i = E_i / 2`` (and an estimated nugget of
    ``1e-6`` relative, or the lower bound if larger); further starts perturb
    the log length scales uniformly by up to ``+/- 1.5``.

    References:
        [EQ-SUR-UK-002]. No new equation.

    """
    first = [math.log(0.5 * width) for width in extents]
    if estimate_nugget:
        first.append(max(math.log(1e-6), bounds[-1][0]))
    generator = resolve_stream(seed).child("kriging-restarts").generator()
    raw = [tuple(first)]
    for _ in range(restarts - 1):
        perturbed = [
            value + generator.uniform(-1.5, 1.5) for value in first[: len(extents)]
        ]
        if estimate_nugget:
            perturbed.append(first[-1])
        raw.append(tuple(perturbed))
    starts: list[Vector] = []
    for start in raw:
        clipped = []
        for value, (low, high) in zip(start, bounds, strict=True):
            margin = 0.01 * (high - low)
            clipped.append(min(max(value, low + margin), high - margin))
        starts.append(tuple(clipped))
    return tuple(starts)


def _bound_hits(
    theta: Sequence[float], bounds: Sequence[tuple[float, float]], dimension: int
) -> tuple[str, ...]:
    """Name the log parameters within ``0.01`` of a bound.

    References:
        [EQ-SUR-UK-002]. No new equation.

    """
    names = (*(f"length_scale[{axis}]" for axis in range(dimension)), "nugget")
    hits: list[str] = []
    for name, value, (low, high) in zip(names, theta, bounds, strict=False):
        if value - low <= 0.01:
            hits.append(f"{name}:lower")
        elif high - value <= 0.01:
            hits.append(f"{name}:upper")
    return tuple(hits)


def _fit(
    points: tuple[Vector, ...],
    values: Vector,
    regressor: Callable[[Vector], Vector],
    *,
    kernel: str,
    nugget: float,
    estimate_nugget: bool,
    starts: Sequence[Vector],
    bounds: tuple[tuple[float, float], ...],
    optimize: bool,
) -> UniversalKriging:
    """Fit a universal kriging model from given starts (shared core).

    Raises:
        ValueError: If no start gives a positive-definite correlation matrix
            or the trend is rank deficient.

    Equation:
        [EQ-SUR-UK-001], [EQ-SUR-UK-002].

    References:
        [REF-SCHOBI-SUDRET-WIART-2015], [REF-NOCEDAL-WRIGHT-2006, Section 6.1].

    """
    dimension = len(points[0])
    count = len(points)
    regressor_rows = tuple(regressor(point) for point in points)
    terms = len(regressor_rows[0])
    if terms == 0 or any(len(row) != terms for row in regressor_rows):
        raise ValueError("the regressors must return equally long, non-empty vectors")
    if terms >= count:
        raise ValueError("universal kriging needs more training points than regressors")
    data = _pair_data(points)
    cache: dict[Vector, tuple[float, list[float]]] = {}
    evaluations = [0]

    def objective(theta: Vector) -> tuple[float, list[float]]:
        """Return the cached concentrated objective and gradient.

        References:
            [EQ-SUR-UK-002].

        """
        key = tuple(theta)
        hit = cache.get(key)
        if hit is not None:
            return hit
        evaluations[0] += 1
        result = _objective(
            data,
            values,
            regressor_rows,
            key,
            dimension,
            nugget,
            estimate_nugget,
            kernel,
        )
        if len(cache) >= _CACHE_SIZE:
            del cache[next(iter(cache))]
        cache[key] = result
        return result

    def value(theta: Vector) -> float:
        """Return the objective value.

        References:
            [EQ-SUR-UK-002].

        """
        return objective(theta)[0]

    def gradient(theta: Vector) -> Vector:
        """Return the objective gradient.

        References:
            [EQ-SUR-UK-002].

        """
        return tuple(objective(theta)[1])

    first = starts[0]
    first_scales = tuple(math.exp(v) for v in first[:dimension])
    first_extra = math.exp(first[dimension]) if estimate_nugget else 0.0
    probe, _, _ = _evaluate_state(
        data, values, regressor_rows, first_scales, nugget + first_extra, kernel
    )
    exact = probe.exact_trend
    run_optimizer = optimize and not exact
    candidates: list[tuple[float, Vector, bool]] = []
    for start in starts if run_optimizer else starts[:1]:
        try:
            if run_optimizer:
                result = minimize_bfgs(
                    value,
                    start,
                    gradient=gradient,
                    gradient_tolerance=1e-5,
                    max_iterations=200,
                    bounds=bounds,
                )
                candidates.append((result.value, result.x, result.converged))
            else:
                candidates.append((value(start), start, True))
        except ValueError:
            continue
    if not candidates:
        raise ValueError(
            "no start produced a positive-definite correlation matrix; "
            "increase the nugget"
        )
    objective_value, theta, converged = min(candidates, key=lambda item: item[0])
    scales = tuple(math.exp(v) for v in theta[:dimension])
    total_nugget = nugget + (math.exp(theta[dimension]) if estimate_nugget else 0.0)
    state, _, _ = _evaluate_state(
        data, values, regressor_rows, scales, total_nugget, kernel
    )
    log_likelihood = (
        -0.5 * count * (math.log(2.0 * math.pi * state.process_variance) + 1.0)
        - 0.5 * state.log_det
    )
    fitted = UniversalKriging(
        kernel,
        points,
        values,
        regressor,
        scales,
        total_nugget,
        state.beta,
        state.process_variance,
        log_likelihood,
        converged,
        len(starts) if run_optimizer else 1,
        evaluations[0],
        log_parameter_bounds=bounds,
        at_bound=_bound_hits(theta, bounds, dimension) if run_optimizer else (),
        exact_trend=exact,
    )
    del objective_value
    if fitted.condition_number > _CONDITION_WARNING:
        warnings.warn(
            f"the fitted correlation matrix has 1-norm condition number "
            f"{fitted.condition_number:.3g}; predictions carry rounding noise "
            "of about cond(R) * 1e-16 relative. A larger nugget bounds it.",
            GaussianProcessConditioningWarning,
            stacklevel=3,
        )
    return fitted


def _validate_common(kernel: str, nugget: float, restarts: int) -> float:
    """Validate kernel, nugget, and restarts; return the nugget.

    Raises:
        ValueError: If any value is out of range.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].
        No equation is evaluated.

    """
    if kernel not in KERNELS:
        raise ValueError(f"kernel must be one of {KERNELS}")
    jitter = require_finite_real(nugget, name="nugget")
    if jitter <= 0.0:
        raise ValueError("nugget must be positive")
    if isinstance(restarts, bool) or not isinstance(restarts, int) or restarts < 1:
        raise ValueError("restarts must be a positive integer")
    return jitter


def fit_universal_kriging(
    points: Sequence[Sequence[float]],
    values: Sequence[float],
    *,
    regressors: Regressors | None = None,
    kernel: str = "matern52",
    nugget: float = 1e-10,
    estimate_nugget: bool = False,
    length_scales: Sequence[float] | None = None,
    restarts: int = 3,
    seed: int | RandomStream | None = None,
) -> UniversalKriging:
    """Fit universal kriging by maximum concentrated likelihood.

    For given length scales the trend coefficients are the generalized least
    squares estimate ``beta = (F'R^-1F)^-1 F'R^-1 y`` and the process
    variance is ``sigma^2 = (y - F beta)'R^-1(y - F beta) / N``
    [EQ-SUR-UK-001]. The length scales (and, with ``estimate_nugget``, a
    nugget) then minimize ``sigma^2 det(R)^(1/N)``, through BFGS with the
    exact gradient [EQ-SUR-UK-002] on ``log ell_i`` inside the box
    ``[log(0.01 E_i), log(20 E_i)]`` (``E_i`` the extent of input ``i``,
    computational choices). The first start is ``ell_i = E_i / 2``; further
    starts perturb it uniformly by up to ``+/- 1.5`` in log units. The
    lowest objective over the starts wins.

    With ``regressors=None`` the trend is a constant: ordinary kriging. A
    trend that reproduces the responses to ``1e-10`` relative leaves no
    residual process to learn; the length scales then stay at the first
    start and ``exact_trend`` is set.

    Args:
        points: Training inputs (at least two, none constant in every input).
        values: Training responses.
        regressors: Trend basis: ``None`` (constant), a callable returning
            the regressor vector ``f(x)``, or a sequence of scalar callables.
        kernel: ``"squared_exponential"``, ``"matern52"``, or ``"matern32"``.
        nugget: Fixed positive diagonal added to ``R`` (relative to the
            process variance).
        estimate_nugget: Also estimate an extra nugget in ``[nugget, 1]``.
        length_scales: Fixed length scales; skips the optimization (only
            with ``estimate_nugget=False``).
        restarts: Optimizer starts (at least one).
        seed: Integer seed or stream for the restart perturbations.

    Returns:
        Fitted model with closed-form leave-one-out error ``loo_error``.

    Raises:
        TypeError: If ``regressors`` has the wrong type.
        ValueError: If inputs are invalid, there are not more points than
            regressors, the trend is rank deficient, or no start gives a
            positive-definite correlation matrix.

    Equation:
        [EQ-SUR-UK-001], [EQ-SUR-UK-002], [EQ-SUR-UK-003], [EQ-SUR-UK-004].

    References:
        [REF-SCHOBI-SUDRET-WIART-2015],
        [REF-RASMUSSEN-WILLIAMS-2006, Chapter 2],
        [REF-NOCEDAL-WRIGHT-2006, Section 6.1].

    """
    jitter = _validate_common(kernel, nugget, restarts)
    x = tuple(as_vector(point, name="point") for point in points)
    y = as_vector(values, name="values")
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("need at least two points with one value each")
    dimension = len(x[0])
    if dimension == 0 or any(len(point) != dimension for point in x):
        raise ValueError("points must be non-empty and equally long")
    regressor = _resolve_regressors(regressors)
    extents = _extents(x)
    bounds = _kriging_bounds(extents, jitter, estimate_nugget)
    if length_scales is not None:
        if estimate_nugget:
            raise ValueError("fixed length_scales require estimate_nugget=False")
        fixed = as_vector(length_scales, name="length_scales")
        if len(fixed) != dimension or any(not ell > 0.0 for ell in fixed):
            raise ValueError("length_scales must be positive, one per input")
        return _fit(
            x,
            y,
            regressor,
            kernel=kernel,
            nugget=jitter,
            estimate_nugget=False,
            starts=(tuple(math.log(ell) for ell in fixed),),
            bounds=bounds,
            optimize=False,
        )
    starts = _kriging_starts(extents, bounds, restarts, estimate_nugget, seed)
    return _fit(
        x,
        y,
        regressor,
        kernel=kernel,
        nugget=jitter,
        estimate_nugget=estimate_nugget,
        starts=starts,
        bounds=bounds,
        optimize=True,
    )


@dataclass(frozen=True)
class PCKriging:
    """A fitted polynomial-chaos Kriging surrogate on physical inputs.

    Predictions map a physical input ``x`` to the standardized PCE variables
    ``xi(x)`` and evaluate the universal kriging model there, whose trend is
    ``sum_alpha beta_alpha Psi_alpha(xi)``.

    Args:
        inputs: Joint input law (defines ``xi(x)``).
        basis: ``"hermite"`` or ``"legendre"``.
        variant: ``"sequential"`` or ``"optimal"``.
        quantity: Name of the approximated quantity.
        design: Physical training inputs.
        responses: Training responses.
        trend_indices: Multi-indices of the selected trend, constant first,
            then in least-angle-regression order.
        kriging: The universal kriging model on the standardized inputs.
        expansion: The sparse PCE whose LAR ranking supplied the
            candidate polynomials.
        candidate_loo_errors: Relative leave-one-out error of the kriging
            model with the constant plus the first ``k`` ranked polynomials,
            ``k = 0, 1, ...`` (``"optimal"``); one entry for the sequential
            fit.

    References:
        [REF-SCHOBI-SUDRET-WIART-2015], [REF-BLATMAN-SUDRET-2011].

    """

    inputs: JointInput
    basis: str
    variant: str
    quantity: str
    design: tuple[Vector, ...]
    responses: Vector
    trend_indices: tuple[MultiIndex, ...]
    kriging: UniversalKriging
    expansion: PolynomialChaosExpansion
    candidate_loo_errors: Vector

    @property
    def loo_error(self) -> float:
        """Return the relative closed-form leave-one-out error.

        Equation:
            [EQ-SUR-UK-004].

        References:
            [REF-SCHOBI-SUDRET-WIART-2015].

        """
        return self.kriging.loo_error

    @property
    def length_scales(self) -> Vector:
        """Return the fitted length scales in standardized coordinates.

        References:
            [REF-SCHOBI-SUDRET-WIART-2015].

        """
        return self.kriging.length_scales

    @property
    def process_variance(self) -> float:
        """Return the profiled process variance ``sigma^2``.

        Equation:
            [EQ-SUR-UK-001].

        References:
            [REF-SCHOBI-SUDRET-WIART-2015].

        """
        return self.kriging.process_variance

    def standardize(self, x: Sequence[float]) -> Vector:
        """Map a physical input to the standardized PCE variables.

        Args:
            x: Physical input vector.

        Returns:
            ``u`` (Hermite, Nataf map) or ``xi in [-1, 1]^d`` (Legendre).

        Equation:
            [EQ-MOD-NATAF-001], [EQ-SUR-PCE-001].

        References:
            [REF-LIU-DERKIUREGHIAN-1986], [REF-XIU-KARNIADAKIS-2002].

        """
        return self.expansion.standardize(x)

    def predict_with_variance(
        self, x: Sequence[float], *, include_noise: bool = False
    ) -> tuple[float, float]:
        """Return the PC-Kriging mean and variance at a physical input.

        Args:
            x: Physical input vector.
            include_noise: Add the nugget variance.

        Returns:
            Predictive mean and variance.

        Equation:
            [EQ-SUR-UK-003], [EQ-SUR-PCK-001].

        References:
            [REF-SCHOBI-SUDRET-WIART-2015].

        """
        return self.kriging.predict_with_variance(
            self.standardize(x), include_noise=include_noise
        )

    def predict(self, x: Sequence[float]) -> float:
        """Return the PC-Kriging mean at a physical input.

        Args:
            x: Physical input vector.

        Returns:
            Predictive mean.

        Equation:
            [EQ-SUR-UK-003], [EQ-SUR-PCK-001].

        References:
            [REF-SCHOBI-SUDRET-WIART-2015].

        """
        return self.kriging.predict(self.standardize(x))

    def leave_one_out(self) -> KrigingLOO:
        """Return the closed-form LOO diagnostics of the kriging model.

        Returns:
            LOO means, variances, residuals, and relative error.

        Equation:
            [EQ-SUR-UK-004].

        References:
            [REF-SCHOBI-SUDRET-WIART-2015].

        """
        return self.kriging.leave_one_out()


def _warm_start(
    fitted: UniversalKriging,
    nugget: float,
    estimate_nugget: bool,
    bounds: Sequence[tuple[float, float]],
) -> Vector:
    """Return a fitted model's log parameters, clipped inside the box.

    References:
        [EQ-SUR-PCK-001]. No new equation.

    """
    theta = [math.log(ell) for ell in fitted.length_scales]
    if estimate_nugget:
        theta.append(math.log(max(fitted.nugget - nugget, nugget)))
    clipped = []
    for value, (low, high) in zip(theta, bounds, strict=True):
        margin = 0.01 * (high - low)
        clipped.append(min(max(value, low + margin), high - margin))
    return tuple(clipped)


def pc_kriging_from_samples(
    inputs: JointInput,
    points: Sequence[Sequence[float]],
    values: Sequence[float],
    *,
    degrees: Iterable[int] = range(1, 6),
    q_norms: Sequence[float] = (1.0,),
    basis: str = "hermite",
    kernel: str = "matern52",
    variant: str = "sequential",
    nugget: float = 1e-10,
    estimate_nugget: bool = False,
    restarts: int = 3,
    seed: int | RandomStream | None = None,
    quantity: str = "y",
) -> PCKriging:
    """Fit PC-Kriging on given samples of a joint input law.

    1. A sparse PCE is fitted by hybrid least-angle regression with degree
       and q-norm adaptivity (:func:`sparse_polynomial_chaos_from_samples`);
       its LAR entry order ranks the selected polynomials.
    2. ``variant="sequential"``: one universal kriging model is fitted with
       the constant and all selected polynomials as its trend.
       ``variant="optimal"``: a model is fitted for the constant plus the
       first ``k`` ranked polynomials, ``k = 0, ..., P - 1``, and the one
       with the smallest closed-form leave-one-out error is kept
       [EQ-SUR-PCK-001]. Each smaller trend is optimized from one start,
       the optimum of the previous size (a computational shortcut); the
       full trend uses all ``restarts`` starts exactly as the sequential
       variant, so the last candidate is the sequential model and the
       optimal variant's error is never larger.

    The trend polynomials and the correlation function both act on the
    standardized variables ``xi(x)`` of the PCE basis.

    Args:
        inputs: Joint input law of the samples.
        points: Physical input vectors.
        values: Model responses, one per point.
        degrees: Candidate maximal total degrees of the sparse PCE.
        q_norms: Candidate hyperbolic norms in ``(0, 1]``.
        basis: ``"hermite"`` or ``"legendre"``.
        kernel: ``"squared_exponential"``, ``"matern52"``, or ``"matern32"``.
        variant: ``"sequential"`` or ``"optimal"``.
        nugget: Fixed positive diagonal of the correlation matrix.
        estimate_nugget: Also estimate an extra nugget.
        restarts: Optimizer starts per kriging fit.
        seed: Integer seed or stream for the restart perturbations.
        quantity: Label of the approximated quantity.

    Returns:
        Fitted PC-Kriging surrogate.

    Raises:
        ValueError: If the basis, kernel, or variant is unknown or
            inapplicable, the samples are too few, or a kriging fit fails.

    Equation:
        [EQ-SUR-PCK-001], [EQ-SUR-UK-001], [EQ-SUR-UK-002],
        [EQ-SUR-SPCE-004].

    References:
        [REF-SCHOBI-SUDRET-WIART-2015], [REF-BLATMAN-SUDRET-2011].

    """
    _check_basis(inputs, basis)
    if variant not in PC_KRIGING_VARIANTS:
        raise ValueError(f"variant must be one of {PC_KRIGING_VARIANTS}")
    jitter = _validate_common(kernel, nugget, restarts)
    expansion = sparse_polynomial_chaos_from_samples(
        inputs,
        points,
        values,
        degrees=degrees,
        q_norms=q_norms,
        basis=basis,
        method="lar",
        quantity=quantity,
    )
    selection = expansion.selection
    assert selection is not None
    chosen = selection.chosen
    constant = expansion.indices[0]
    ranked = (constant, *chosen.entry_order[: chosen.selected_terms])
    design = expansion.design
    responses = expansion.responses
    standardized = tuple(expansion.standardize(point) for point in design)
    extents = _extents(standardized)
    bounds = _kriging_bounds(extents, jitter, estimate_nugget)
    starts = _kriging_starts(extents, bounds, restarts, estimate_nugget, seed)
    sizes = range(1, len(ranked) + 1) if variant == "optimal" else (len(ranked),)
    best: UniversalKriging | None = None
    best_size = 0
    errors: list[float] = []
    warm = starts[0]
    for size in sizes:
        try:
            fitted = _fit(
                standardized,
                responses,
                ChaosTrend(basis, ranked[:size]),
                kernel=kernel,
                nugget=jitter,
                estimate_nugget=estimate_nugget,
                starts=starts if size == len(ranked) else (warm,),
                bounds=bounds,
                optimize=True,
            )
        except ValueError:
            if variant == "sequential":
                raise
            errors.append(math.inf)
            continue
        warm = _warm_start(fitted, jitter, estimate_nugget, bounds)
        errors.append(fitted.loo_error)
        if best is None or fitted.loo_error <= best.loo_error:
            best, best_size = fitted, size
    if best is None:
        raise ValueError("no trend size produced a valid kriging fit")
    return PCKriging(
        inputs,
        basis,
        variant,
        quantity,
        design,
        responses,
        ranked[:best_size],
        best,
        expansion,
        tuple(errors),
    )


def fit_pc_kriging(
    model: ProbabilisticModel
    | tuple[JointInput, Sequence[Sequence[float]], Sequence[float]],
    quantity: str | None = None,
    *,
    size: int | None = None,
    degrees: Iterable[int] = range(1, 6),
    q_norms: Sequence[float] = (1.0,),
    basis: str = "hermite",
    kernel: str = "matern52",
    variant: str = "sequential",
    nugget: float = 1e-10,
    estimate_nugget: bool = False,
    restarts: int = 3,
    design: str = "lhs",
    seed: int | RandomStream | None = None,
) -> PCKriging:
    """Build a PC-Kriging surrogate of a model, or of given samples.

    With a :class:`~probstat.modeling.ProbabilisticModel`, ``size`` design
    points are drawn on the unit hypercube (``make_design``, child stream
    ``"pc-kriging-design"``), mapped through the joint input law, and
    evaluated once. With a tuple ``(inputs, points, values)`` the samples are
    used as given (``size`` and ``design`` are ignored). Either way
    :func:`pc_kriging_from_samples` does the fit.

    Args:
        model: Probabilistic model, or ``(inputs, points, values)``.
        quantity: QoI name; defaults to the first declared quantity (``"y"``
            for samples).
        size: Number of model evaluations (required for a model).
        degrees: Candidate maximal total degrees of the sparse PCE.
        q_norms: Candidate hyperbolic norms in ``(0, 1]``.
        basis: ``"hermite"`` or ``"legendre"``.
        kernel: ``"squared_exponential"``, ``"matern52"``, or ``"matern32"``.
        variant: ``"sequential"`` or ``"optimal"``.
        nugget: Fixed positive diagonal of the correlation matrix.
        estimate_nugget: Also estimate an extra nugget.
        restarts: Optimizer starts per kriging fit.
        design: ``"mc"``, ``"lhs"``, ``"halton"``, or ``"sobol"``.
        seed: Integer seed or stream (design and restarts).

    Returns:
        Fitted PC-Kriging surrogate; ``model.evaluations`` grows by ``size``.

    Raises:
        ValueError: If ``size`` is missing or below three for a model, or the
            fit fails (see :func:`pc_kriging_from_samples`).

    Equation:
        [EQ-SUR-PCK-001], [EQ-SAMP-INV-001].

    References:
        [REF-SCHOBI-SUDRET-WIART-2015], [REF-BLATMAN-SUDRET-2011].

    """
    root = resolve_stream(seed)
    if isinstance(model, tuple):
        inputs, points, values = model
        name = "y" if quantity is None else quantity
    else:
        name = model.quantity_names[0] if quantity is None else quantity
        model.quantity(name)
        _check_basis(model.inputs, basis)
        if size is None or isinstance(size, bool) or not isinstance(size, int):
            raise ValueError("size must be an integer of at least 3")
        if size < 3:
            raise ValueError("size must be an integer of at least 3")
        unit = make_design(
            design, size, model.dimension, seed=root.child("pc-kriging-design")
        )
        inputs = model.inputs
        points = model.inputs.sample_design(unit)
        values = model.evaluate_many(points)[name]
    return pc_kriging_from_samples(
        inputs,
        points,
        values,
        degrees=degrees,
        q_norms=q_norms,
        basis=basis,
        kernel=kernel,
        variant=variant,
        nugget=nugget,
        estimate_nugget=estimate_nugget,
        restarts=restarts,
        seed=root,
        quantity=name,
    )
