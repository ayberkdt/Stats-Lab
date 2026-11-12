"""Vector autoregression VAR(p): least squares, order selection, structural analysis.

Model for a ``K``-dimensional series::

    y_t = nu + A_1 y_(t-1) + ... + A_p y_(t-p) + u_t,   u_t i.i.d. (0, Sigma_u).

Writing ``Z_t = (1, y_(t-1)', ..., y_(t-p)')'`` and ``B = [nu, A_1, ...,
A_p]``, the multivariate least-squares estimator over ``t = p+1..n``
(``T = n - p`` equations per component) is ``B_hat = Y Z' (Z Z')^-1``; it is
equation-by-equation OLS and coincides with conditional Gaussian maximum
likelihood. ``Sigma_u`` is estimated with the degrees-of-freedom divisor
``T - K p - 1`` and ``Cov(vec B_hat) = (Z Z')^-1 (x) Sigma_u_hat``
[EQ-TS-VAR-001].

Structural analysis follows the moving-average representation
``y_t = mu + sum_i Phi_i u_(t-i)`` with ``Phi_0 = I`` and
``Phi_i = sum_{j=1}^{min(i,p)} Phi_(i-j) A_j`` [EQ-TS-VAR-004]: the
orthogonalized impulse responses are ``Theta_i = Phi_i P`` with ``P`` the
lower Cholesky factor of ``Sigma_u`` (so the ordering of the variables
matters), and the forecast-error variance decomposition divides the
cumulated squared ``Theta`` entries by the forecast MSE [EQ-TS-VAR-005].
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    coerce_finite_sample,
    require_non_negative_integer,
    require_probability,
)
from probstat.inference import chi_square_sf, f_sf
from probstat.linalg import (
    Matrix,
    cholesky,
    inverse_spd,
    log_det_spd,
    matmul,
    transpose,
)
from probstat.modeling import standard_normal_quantile
from probstat.statistics import quantiles

from ._eigen import eigenvalues as _eigenvalues

Vector = tuple[float, ...]
_CRITERIA = ("aic", "bic", "hq", "fpe")


def _rows(data: Sequence[Sequence[float]]) -> tuple[Vector, ...]:
    """Validate a ``n x K`` data matrix given as rows.

    Args:
        data: Observations, one row per time point.

    Returns:
        Rows as tuples of floats.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If rows are ragged, empty, or non-finite.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    rows = tuple(coerce_finite_sample(tuple(row), name="row") for row in data)
    if not rows:
        raise ValueError("data must contain at least one row")
    width = len(rows[0])
    if width < 1 or any(len(row) != width for row in rows):
        raise ValueError("every row must have the same positive length")
    return rows


def _regression(
    rows: Sequence[Vector], order: int, start: int, include_constant: bool
) -> tuple[list[list[float]], list[Vector]]:
    """Return the regressor rows ``Z_t'`` and responses ``y_t'`` from ``start``.

    Args:
        rows: Observations.
        order: Lag order ``p``.
        start: First 0-based response index (at least ``order``).
        include_constant: Whether ``Z_t`` starts with a one.

    Returns:
        Regressor rows and response rows.

    Equation:
        [EQ-TS-VAR-001].

    References:
        [REF-LUTKEPOHL-2005, Chapter 3].

    """
    regressors: list[list[float]] = []
    responses: list[Vector] = []
    for t in range(start, len(rows)):
        z = [1.0] if include_constant else []
        for lag in range(1, order + 1):
            z.extend(rows[t - lag])
        regressors.append(z)
        responses.append(rows[t])
    return regressors, responses


def _least_squares(
    regressors: Sequence[Sequence[float]], responses: Sequence[Vector]
) -> tuple[list[list[float]], Matrix | None, list[Vector]]:
    """Solve the multivariate least-squares problem.

    Args:
        regressors: ``T x m`` regressor rows.
        responses: ``T x K`` response rows.

    Returns:
        Coefficients ``B'`` (``m x K``), ``(Z Z')^-1`` (``None`` when
        ``m = 0``), and residual rows.

    Raises:
        ValueError: If ``Z Z'`` is not positive definite.

    Equation:
        [EQ-TS-VAR-001].

    References:
        [REF-LUTKEPOHL-2005, Chapter 3].

    """
    m = len(regressors[0]) if regressors else 0
    k = len(responses[0])
    if m == 0:
        return [], None, list(responses)
    gram = tuple(
        tuple(math.fsum(z[i] * z[j] for z in regressors) for j in range(m))
        for i in range(m)
    )
    inverse = inverse_spd(gram)
    cross = [
        [
            math.fsum(z[i] * y[c] for z, y in zip(regressors, responses, strict=True))
            for c in range(k)
        ]
        for i in range(m)
    ]
    coefficients = [
        [math.fsum(inverse[i][j] * cross[j][c] for j in range(m)) for c in range(k)]
        for i in range(m)
    ]
    residuals = [
        tuple(
            y[c] - math.fsum(z[i] * coefficients[i][c] for i in range(m))
            for c in range(k)
        )
        for z, y in zip(regressors, responses, strict=True)
    ]
    return coefficients, inverse, residuals


def _cross_product(residuals: Sequence[Vector], divisor: float) -> Matrix:
    """Return ``U' U / divisor``.

    Args:
        residuals: Residual rows.
        divisor: Positive divisor.

    Returns:
        Symmetric ``K x K`` matrix.

    Equation:
        [EQ-TS-VAR-001].

    References:
        [REF-LUTKEPOHL-2005, Chapter 3].

    """
    k = len(residuals[0])
    return tuple(
        tuple(math.fsum(u[i] * u[j] for u in residuals) / divisor for j in range(k))
        for i in range(k)
    )


@dataclass(frozen=True, slots=True)
class GrangerCausalityResult:
    """Wald and F tests that some variables do not Granger-cause others.

    Args:
        caused: Indices of the response variables.
        causing: Indices of the variables whose lags are restricted.
        restrictions: Number of zero restrictions ``N``.
        wald_statistic: ``lambda_W``, asymptotically ``chi^2(N)``.
        wald_p_value: Upper chi-square tail probability.
        f_statistic: ``lambda_W / N``.
        f_degrees_of_freedom: ``(N, K (T - K p - 1))``.
        f_p_value: Upper F tail probability.

    References:
        [REF-GRANGER-1969], [REF-LUTKEPOHL-2005, Chapter 3].

    """

    caused: tuple[int, ...]
    causing: tuple[int, ...]
    restrictions: int
    wald_statistic: float
    wald_p_value: float
    f_statistic: float
    f_degrees_of_freedom: tuple[int, int]
    f_p_value: float


@dataclass(frozen=True, slots=True)
class VARForecast:
    """Point forecasts and Gaussian intervals of a VAR.

    Args:
        mean: ``y_hat_(n+h)`` for ``h = 1..H`` (rows).
        mse: Forecast MSE matrices ``Sigma_y(h) = sum_{i<h} Phi_i Sigma_u
            Phi_i'``.
        lower: Lower interval bounds (rows).
        upper: Upper interval bounds (rows).
        level: Two-sided interval level.

    References:
        [REF-LUTKEPOHL-2005, Chapter 2].

    """

    mean: tuple[Vector, ...]
    mse: tuple[Matrix, ...]
    lower: tuple[Vector, ...]
    upper: tuple[Vector, ...]
    level: float


@dataclass(frozen=True)
class ImpulseResponseBands:
    """Pointwise bootstrap bands for VAR impulse responses.

    Entry ``[h][i][j]`` of each field refers to the response of variable
    ``i``, ``h`` periods after a shock in variable ``j``.

    Args:
        point: Responses of the fitted VAR (``Theta_h`` or ``Phi_h``).
        lower: Lower pointwise bounds.
        upper: Upper pointwise bounds.
        standard_errors: Standard deviations of the bootstrap responses.
        level: Two-sided level.
        replications: Bootstrap replications.
        orthogonalized: Whether the shocks are Cholesky-orthogonalized.
        interval: ``"percentile"`` or ``"hall"``.

    References:
        [REF-RUNKLE-1987], [REF-HALL-1992].

    """

    point: tuple[Matrix, ...]
    lower: tuple[Matrix, ...]
    upper: tuple[Matrix, ...]
    standard_errors: tuple[Matrix, ...]
    level: float
    replications: int
    orthogonalized: bool
    interval: str


@dataclass(frozen=True)
class VARFit:
    """Least-squares VAR(p) fit.

    Args:
        order: Lag order ``p``.
        names: Variable names.
        intercept: Estimated ``nu``, or ``None`` without a constant.
        coefficients: ``A_1..A_p`` (``K x K`` each; row ``i`` is equation
            ``i``).
        intercept_standard_errors: Standard errors of ``nu`` or ``None``.
        coefficient_standard_errors: Standard errors aligned with
            ``coefficients``.
        residual_covariance: ``Sigma_u_hat = U'U / (T - K p - c)`` with
            ``c = 1`` for a constant and ``0`` otherwise.
        residual_covariance_mle: ``Sigma_u_tilde = U'U / T``.
        residuals: ``u_hat_t`` rows for ``t = p+1..n``.
        regressor_inverse: ``(Z Z')^-1``.
        log_likelihood: Conditional Gaussian log-likelihood at the estimate.
        aic: ``log det Sigma_u_tilde + 2 p K^2 / T``.
        bic: ``log det Sigma_u_tilde + log(T) p K^2 / T``.
        hq: ``log det Sigma_u_tilde + 2 log(log T) p K^2 / T``.
        fpe: ``((T + K p + c)/(T - K p - c))^K det Sigma_u_tilde``.
        series: Observations.
        include_constant: Whether ``nu`` was estimated.

    References:
        [REF-LUTKEPOHL-2005, Chapters 3 and 4].

    """

    order: int
    names: tuple[str, ...]
    intercept: Vector | None
    coefficients: tuple[Matrix, ...]
    intercept_standard_errors: Vector | None
    coefficient_standard_errors: tuple[Matrix, ...]
    residual_covariance: Matrix
    residual_covariance_mle: Matrix
    residuals: tuple[Vector, ...]
    regressor_inverse: Matrix | None
    log_likelihood: float
    aic: float
    bic: float
    hq: float
    fpe: float
    series: tuple[Vector, ...]
    include_constant: bool

    @property
    def dimension(self) -> int:
        """Return the number of variables ``K``.

        References:
            [REF-LUTKEPOHL-2005, Chapter 2].

        """
        return len(self.names)

    @property
    def nobs(self) -> int:
        """Return the number of equations per variable ``T = n - p``.

        References:
            [REF-LUTKEPOHL-2005, Chapter 3].

        """
        return len(self.residuals)

    def companion_matrix(self) -> Matrix:
        """Return the ``Kp x Kp`` companion matrix of ``A_1..A_p``.

        Returns:
            ``[[A_1, ..., A_p], [I, 0], ...]``; for ``p = 0`` an empty matrix.

        Equation:
            [EQ-TS-VAR-003].

        References:
            [REF-LUTKEPOHL-2005, Chapter 2].

        """
        k = self.dimension
        size = k * self.order
        rows: list[list[float]] = [[0.0] * size for _ in range(size)]
        for lag, matrix in enumerate(self.coefficients):
            for i in range(k):
                for j in range(k):
                    rows[i][lag * k + j] = matrix[i][j]
        for i in range(k, size):
            rows[i][i - k] = 1.0
        return tuple(tuple(row) for row in rows)

    def eigenvalues(self) -> tuple[complex, ...]:
        """Return the companion eigenvalues, largest modulus first.

        Returns:
            Eigenvalues; the VAR is stable when every modulus is below one.

        Equation:
            [EQ-TS-VAR-003].

        References:
            [REF-LUTKEPOHL-2005, Chapter 2], [REF-GOLUB-VANLOAN-2013,
            Section 7.5].

        """
        return _eigenvalues(self.companion_matrix())

    def is_stable(self) -> bool:
        """Return whether every companion eigenvalue has modulus below one.

        Equation:
            [EQ-TS-VAR-003].

        References:
            [REF-LUTKEPOHL-2005, Chapter 2].

        """
        return all(abs(value) < 1.0 for value in self.eigenvalues())

    def ma_coefficients(self, horizon: int) -> tuple[Matrix, ...]:
        """Return ``Phi_0..Phi_H`` of the moving-average representation.

        Args:
            horizon: Last index ``H >= 0``.

        Returns:
            ``Phi_0 = I`` and ``Phi_i = sum_{j=1}^{min(i,p)} Phi_(i-j) A_j``.

        Raises:
            ValueError: If ``horizon`` is negative.

        Equation:
            [EQ-TS-VAR-004].

        References:
            [REF-LUTKEPOHL-2005, Chapter 2].

        """
        steps = require_non_negative_integer(horizon, name="horizon")
        k = self.dimension
        identity: Matrix = tuple(
            tuple(1.0 if i == j else 0.0 for j in range(k)) for i in range(k)
        )
        phis: list[Matrix] = [identity]
        for i in range(1, steps + 1):
            total = [[0.0] * k for _ in range(k)]
            for j in range(1, min(i, self.order) + 1):
                product = matmul(phis[i - j], self.coefficients[j - 1])
                for r in range(k):
                    for c in range(k):
                        total[r][c] += product[r][c]
            phis.append(tuple(tuple(row) for row in total))
        return tuple(phis)

    def impulse_response(
        self, horizon: int, *, orthogonalized: bool = True
    ) -> tuple[Matrix, ...]:
        """Return impulse responses for horizons ``0..H``.

        Entry ``[i][j]`` of matrix ``h`` is the response of variable ``i``,
        ``h`` periods later, to a shock in variable ``j``: a unit innovation
        ``u_j`` (``Phi_h``) or, when orthogonalized, a one-standard-deviation
        orthogonal shock (``Theta_h = Phi_h P``, ``P`` the lower Cholesky
        factor of ``Sigma_u_hat``, which depends on the variable order).

        Args:
            horizon: Last horizon ``H >= 0``.
            orthogonalized: Use Cholesky-orthogonalized shocks.

        Returns:
            ``H + 1`` response matrices.

        Raises:
            ValueError: If ``horizon`` is negative.

        Equation:
            [EQ-TS-VAR-004].

        References:
            [REF-LUTKEPOHL-2005, Chapter 2].

        """
        phis = self.ma_coefficients(horizon)
        if not orthogonalized:
            return phis
        factor = cholesky(self.residual_covariance)
        return tuple(matmul(phi, factor) for phi in phis)

    def impulse_response_bands(
        self,
        horizon: int,
        *,
        replications: int = 499,
        level: float = 0.95,
        orthogonalized: bool = True,
        interval: str = "percentile",
        seed: int | RandomStream | None = 0,
    ) -> ImpulseResponseBands:
        """Return residual-bootstrap confidence bands for the impulse responses.

        Each replication resamples the centred residuals with replacement,
        rebuilds a series by the fitted recursion from the first ``p``
        observations, refits the VAR(p) with the same constant choice, and
        recomputes the responses (orthogonalized with the replication's own
        ``Sigma_u``). ``"percentile"`` reports the ``alpha/2`` and
        ``1 - alpha/2`` bootstrap quantiles; ``"hall"`` reflects them about
        the estimate, ``[2 theta_hat - q_hi, 2 theta_hat - q_lo]``.

        Least-squares VAR coefficients are biased toward zero in short
        samples, and percentile bands inherit that bias; for persistent
        systems and small ``T`` the coverage can fall below the nominal
        level [REF-KILIAN-1998]. The band is pointwise in ``h``, not a
        simultaneous band for the whole response path.

        Args:
            horizon: Last horizon ``H >= 0``.
            replications: Bootstrap replications ``R >= 2``.
            level: Two-sided level in ``(0, 1)``.
            orthogonalized: Use Cholesky-orthogonalized shocks.
            interval: ``"percentile"`` or ``"hall"``.
            seed: Seed or stream; replication ``r`` uses child ``("irf", r)``.
                ``None`` is not reproducible.

        Returns:
            Point responses, pointwise bounds, and bootstrap standard errors.

        Raises:
            ValueError: If an argument is invalid or a replication's
                regressors are collinear.

        Equation:
            [EQ-TS-VAR-004], [EQ-TS-VAR-008].

        References:
            [REF-RUNKLE-1987], [REF-KILIAN-1998],
            [REF-EFRON-TIBSHIRANI-1993, Chapter 13], [REF-HALL-1992],
            [REF-LUTKEPOHL-2005, Chapter 3].

        """
        steps = require_non_negative_integer(horizon, name="horizon")
        count = require_non_negative_integer(replications, name="replications")
        if count < 2:
            raise ValueError("replications must be at least 2")
        coverage = require_probability(level, name="level")
        if not 0.0 < coverage < 1.0:
            raise ValueError("level must lie strictly between 0 and 1")
        if interval not in {"percentile", "hall"}:
            raise ValueError("interval must be 'percentile' or 'hall'")
        k = self.dimension
        p = self.order
        point = self.impulse_response(steps, orthogonalized=orthogonalized)
        means = tuple(
            math.fsum(row[i] for row in self.residuals) / self.nobs for i in range(k)
        )
        centred = tuple(
            tuple(row[i] - means[i] for i in range(k)) for row in self.residuals
        )
        nu = self.intercept if self.intercept is not None else (0.0,) * k
        root = resolve_stream(seed)
        draws: list[tuple[Matrix, ...]] = []
        for replication in range(count):
            generator = root.child("irf", replication).generator()
            series: list[Vector] = list(self.series[:p])
            for _ in range(self.nobs):
                shock = centred[generator.randrange(len(centred))]
                value = [nu[i] + shock[i] for i in range(k)]
                for lag, matrix in enumerate(self.coefficients, start=1):
                    past = series[-lag]
                    for i in range(k):
                        value[i] += math.fsum(matrix[i][j] * past[j] for j in range(k))
                series.append(tuple(value))
            refit = fit_var(
                series, p, include_constant=self.include_constant, names=self.names
            )
            draws.append(refit.impulse_response(steps, orthogonalized=orthogonalized))
        tail = 0.5 * (1.0 - coverage)
        lower: list[Matrix] = []
        upper: list[Matrix] = []
        errors: list[Matrix] = []
        for h in range(steps + 1):
            low_rows, high_rows, error_rows = [], [], []
            for i in range(k):
                low_row, high_row, error_row = [], [], []
                for j in range(k):
                    values = [draw[h][i][j] for draw in draws]
                    q_low, q_high = quantiles(values, (tail, 1.0 - tail))
                    if interval == "hall":
                        estimate = point[h][i][j]
                        q_low, q_high = 2.0 * estimate - q_high, 2.0 * estimate - q_low
                    centre = math.fsum(values) / count
                    spread = math.fsum((v - centre) ** 2 for v in values) / (count - 1)
                    low_row.append(q_low)
                    high_row.append(q_high)
                    error_row.append(math.sqrt(spread))
                low_rows.append(tuple(low_row))
                high_rows.append(tuple(high_row))
                error_rows.append(tuple(error_row))
            lower.append(tuple(low_rows))
            upper.append(tuple(high_rows))
            errors.append(tuple(error_rows))
        return ImpulseResponseBands(
            point=point,
            lower=tuple(lower),
            upper=tuple(upper),
            standard_errors=tuple(errors),
            level=coverage,
            replications=count,
            orthogonalized=orthogonalized,
            interval=interval,
        )

    def fevd(self, horizon: int) -> tuple[Matrix, ...]:
        """Return the forecast-error variance decomposition for ``h = 1..H``.

        Entry ``[i][j]`` of matrix ``h - 1`` is the share of the ``h``-step
        forecast-error variance of variable ``i`` due to orthogonal shock
        ``j``: ``sum_{s<h} Theta_s[i][j]^2 / sum_{s<h} sum_l
        Theta_s[i][l]^2``. Each row sums to one.

        Args:
            horizon: Largest horizon ``H >= 1``.

        Returns:
            ``H`` share matrices.

        Raises:
            ValueError: If ``horizon < 1``.

        Equation:
            [EQ-TS-VAR-005].

        References:
            [REF-LUTKEPOHL-2005, Chapter 2].

        """
        steps = require_non_negative_integer(horizon, name="horizon")
        if steps < 1:
            raise ValueError("horizon must be at least 1")
        thetas = self.impulse_response(steps - 1)
        k = self.dimension
        cumulative = [[0.0] * k for _ in range(k)]
        shares: list[Matrix] = []
        for theta in thetas:
            for i in range(k):
                for j in range(k):
                    cumulative[i][j] += theta[i][j] ** 2
            shares.append(
                tuple(
                    tuple(value / math.fsum(row) for value in row) for row in cumulative
                )
            )
        return tuple(shares)

    def forecast(self, steps: int, *, level: float = 0.95) -> VARForecast:
        """Forecast ``H`` steps ahead with Gaussian intervals.

        Point forecasts iterate the fitted recursion from the last ``p``
        observations; the MSE is ``Sigma_y(h) = sum_{i=0}^{h-1} Phi_i
        Sigma_u_hat Phi_i'`` (parameter uncertainty ignored).

        Args:
            steps: Horizon ``H >= 1``.
            level: Two-sided interval level in ``(0, 1)``.

        Returns:
            Means, MSE matrices, and interval bounds.

        Raises:
            ValueError: If ``steps`` or ``level`` is invalid.

        Equation:
            [EQ-TS-VAR-007].

        References:
            [REF-LUTKEPOHL-2005, Chapter 2].

        """
        horizon = require_non_negative_integer(steps, name="steps")
        if horizon < 1:
            raise ValueError("steps must be at least 1")
        coverage = require_probability(level, name="level")
        if not 0.0 < coverage < 1.0:
            raise ValueError("level must lie strictly between 0 and 1")
        k = self.dimension
        history = list(self.series)
        means: list[Vector] = []
        for _ in range(horizon):
            value = list(self.intercept) if self.intercept is not None else [0.0] * k
            for lag, matrix in enumerate(self.coefficients, start=1):
                past = history[-lag]
                for i in range(k):
                    value[i] += math.fsum(matrix[i][j] * past[j] for j in range(k))
            means.append(tuple(value))
            history.append(tuple(value))
        phis = self.ma_coefficients(horizon - 1)
        sigma = self.residual_covariance
        cumulative = [[0.0] * k for _ in range(k)]
        mse: list[Matrix] = []
        for phi in phis:
            term = matmul(matmul(phi, sigma), transpose(phi))
            for i in range(k):
                for j in range(k):
                    cumulative[i][j] += term[i][j]
            mse.append(tuple(tuple(row) for row in cumulative))
        z = standard_normal_quantile(0.5 + 0.5 * coverage)
        lower = tuple(
            tuple(m[i] - z * math.sqrt(s[i][i]) for i in range(k))
            for m, s in zip(means, mse, strict=True)
        )
        upper = tuple(
            tuple(m[i] + z * math.sqrt(s[i][i]) for i in range(k))
            for m, s in zip(means, mse, strict=True)
        )
        return VARForecast(tuple(means), tuple(mse), lower, upper, coverage)

    def granger_causality(
        self, caused: int | Sequence[int], causing: int | Sequence[int]
    ) -> GrangerCausalityResult:
        """Test that the ``causing`` variables do not Granger-cause ``caused``.

        The null sets every lag coefficient ``A_l[i][j]`` with ``i`` in
        ``caused`` and ``j`` in ``causing`` (``l = 1..p``) to zero. With
        ``C`` selecting those ``N`` coefficients of ``vec(B)``,
        ``lambda_W = (C b)' [C ((Z Z')^-1 (x) Sigma_u_hat) C']^-1 (C b)`` is
        asymptotically ``chi^2(N)``; ``lambda_F = lambda_W / N`` is compared
        with ``F(N, K (T - K p - c))``.

        Args:
            caused: Index or indices of the response variables.
            causing: Index or indices of the restricted variables (disjoint
                from ``caused``).

        Returns:
            Wald and F statistics with p-values.

        Raises:
            ValueError: If the indices are invalid, overlap, or ``p = 0``.

        Equation:
            [EQ-TS-VAR-006], [EQ-INF-DIST-CHI2-001].

        References:
            [REF-GRANGER-1969], [REF-LUTKEPOHL-2005, Chapter 3].

        """
        k = self.dimension
        targets = _indices(caused, k, "caused")
        sources = _indices(causing, k, "causing")
        if set(targets) & set(sources):
            raise ValueError("caused and causing must be disjoint")
        if self.order == 0 or self.regressor_inverse is None:
            raise ValueError("a VAR(0) has no lag coefficients to test")
        offset = 1 if self.include_constant else 0
        selected: list[tuple[int, int, float]] = []
        for i in targets:
            for lag in range(self.order):
                for j in sources:
                    column = offset + lag * k + j
                    selected.append((i, column, self.coefficients[lag][i][j]))
        count = len(selected)
        inverse = self.regressor_inverse
        sigma = self.residual_covariance
        covariance = tuple(
            tuple(inverse[c1][c2] * sigma[i1][i2] for (i2, c2, _) in selected)
            for (i1, c1, _) in selected
        )
        values = [value for (_, _, value) in selected]
        precision = inverse_spd(covariance)
        wald = math.fsum(
            values[a] * precision[a][b] * values[b]
            for a in range(count)
            for b in range(count)
        )
        denominator = k * (self.nobs - k * self.order - offset)
        f_statistic = wald / count
        return GrangerCausalityResult(
            caused=targets,
            causing=sources,
            restrictions=count,
            wald_statistic=wald,
            wald_p_value=chi_square_sf(wald, count),
            f_statistic=f_statistic,
            f_degrees_of_freedom=(count, denominator),
            f_p_value=f_sf(f_statistic, count, denominator),
        )


def _indices(value: int | Sequence[int], k: int, name: str) -> tuple[int, ...]:
    """Validate variable indices.

    Args:
        value: Index or indices.
        k: Number of variables.
        name: Name used in error messages.

    Returns:
        Sorted distinct indices.

    Raises:
        ValueError: If an index is out of range or the set is empty.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    items = (value,) if isinstance(value, int) else tuple(value)
    indices = tuple(sorted({require_non_negative_integer(i, name=name) for i in items}))
    if not indices or indices[-1] >= k:
        raise ValueError(f"{name} must hold indices in [0, {k - 1}]")
    return indices


def _criteria(
    log_det: float, order: int, k: int, t: int, constant: int
) -> tuple[float, float, float, float]:
    """Return ``(AIC, BIC, HQ, FPE)`` for one order on a fixed sample.

    Args:
        log_det: ``log det Sigma_u_tilde``.
        order: Lag order ``m``.
        k: Dimension ``K``.
        t: Sample size ``T``.
        constant: ``1`` with an intercept, ``0`` without.

    Returns:
        The four criteria.

    Equation:
        [EQ-TS-VAR-002].

    References:
        [REF-LUTKEPOHL-2005, Chapter 4], [REF-AKAIKE-1974],
        [REF-SCHWARZ-1978], [REF-HANNAN-QUINN-1979].

    """
    penalty = order * k * k / t
    aic = log_det + 2.0 * penalty
    bic = log_det + math.log(t) * penalty
    hq = log_det + 2.0 * math.log(math.log(t)) * penalty
    regressors = k * order + constant
    fpe = ((t + regressors) / (t - regressors)) ** k * math.exp(log_det)
    return aic, bic, hq, fpe


def fit_var(
    data: Sequence[Sequence[float]],
    order: int,
    *,
    include_constant: bool = True,
    names: Sequence[str] | None = None,
) -> VARFit:
    """Fit a VAR(p) by multivariate least squares.

    Args:
        data: Observations as ``n`` rows of ``K`` values.
        order: Lag order ``p >= 0``.
        include_constant: Estimate the intercept ``nu``.
        names: Variable names (default ``y1..yK``).

    Returns:
        The fit with coefficient standard errors, both residual covariance
        estimates, the conditional log-likelihood, and the criteria of
        [EQ-TS-VAR-002] computed on this fit's own sample.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If the data are ragged, too short for the order, or the
            regressors are collinear.

    Equation:
        [EQ-TS-VAR-001], [EQ-TS-VAR-002].

    References:
        [REF-LUTKEPOHL-2005, Chapter 3].

    """
    rows = _rows(data)
    p = require_non_negative_integer(order, name="order")
    k = len(rows[0])
    constant = 1 if include_constant else 0
    labels = tuple(names) if names is not None else tuple(f"y{i + 1}" for i in range(k))
    if len(labels) != k:
        raise ValueError("names must have one entry per variable")
    t = len(rows) - p
    regressor_count = k * p + constant
    if t - regressor_count < k + 1:
        raise ValueError("too few observations for the requested order")
    regressors, responses = _regression(rows, p, p, include_constant)
    coefficients, inverse, residuals = _least_squares(regressors, responses)
    sigma = _cross_product(residuals, t - regressor_count)
    sigma_mle = _cross_product(residuals, t)
    log_det = log_det_spd(sigma_mle)
    intercept = tuple(coefficients[0]) if include_constant else None
    lags = tuple(
        tuple(
            tuple(coefficients[constant + lag * k + j][i] for j in range(k))
            for i in range(k)
        )
        for lag in range(p)
    )
    if inverse is not None:
        intercept_errors = (
            tuple(math.sqrt(sigma[i][i] * inverse[0][0]) for i in range(k))
            if include_constant
            else None
        )
        lag_errors = tuple(
            tuple(
                tuple(
                    math.sqrt(
                        sigma[i][i]
                        * inverse[constant + lag * k + j][constant + lag * k + j]
                    )
                    for j in range(k)
                )
                for i in range(k)
            )
            for lag in range(p)
        )
    else:
        intercept_errors, lag_errors = None, ()
    log_likelihood = -0.5 * t * (k * math.log(2.0 * math.pi) + log_det + k)
    aic, bic, hq, fpe = _criteria(log_det, p, k, t, constant)
    return VARFit(
        order=p,
        names=labels,
        intercept=intercept,
        coefficients=lags,
        intercept_standard_errors=intercept_errors,
        coefficient_standard_errors=lag_errors,
        residual_covariance=sigma,
        residual_covariance_mle=sigma_mle,
        residuals=tuple(residuals),
        regressor_inverse=inverse,
        log_likelihood=log_likelihood,
        aic=aic,
        bic=bic,
        hq=hq,
        fpe=fpe,
        series=rows,
        include_constant=include_constant,
    )


@dataclass(frozen=True, slots=True)
class VAROrderSelection:
    """Information criteria for VAR orders ``0..max_order`` on one sample.

    Args:
        orders: Candidate orders ``0..max_order``.
        aic: AIC for each order.
        bic: BIC (Schwarz) for each order.
        hq: Hannan-Quinn criterion for each order.
        fpe: Final prediction error for each order.
        selected: Order minimizing each criterion, keyed by name.
        nobs: Common sample size ``T = n - max_order``.

    References:
        [REF-LUTKEPOHL-2005, Chapter 4].

    """

    orders: tuple[int, ...]
    aic: Vector
    bic: Vector
    hq: Vector
    fpe: Vector
    selected: dict[str, int]
    nobs: int


def select_var_order(
    data: Sequence[Sequence[float]],
    max_order: int,
    *,
    include_constant: bool = True,
) -> VAROrderSelection:
    """Compare VAR orders ``0..max_order`` by AIC, BIC, HQ, and FPE.

    Every order is fitted on the same observations ``t = max_order+1..n``
    (``T = n - max_order``) so the criteria are comparable; the penalties
    count the ``m K^2`` lag coefficients (the intercepts are common to all
    orders and do not change which order is selected). BIC and HQ are
    consistent for the true order; AIC and FPE tend to choose larger orders.

    Args:
        data: Observations as rows.
        max_order: Largest order ``M >= 1``.
        include_constant: Include an intercept in every candidate.

    Returns:
        Criteria tables and the order each criterion selects.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If ``max_order < 1`` or the sample is too short.

    Equation:
        [EQ-TS-VAR-001], [EQ-TS-VAR-002].

    References:
        [REF-LUTKEPOHL-2005, Chapter 4], [REF-AKAIKE-1974],
        [REF-SCHWARZ-1978], [REF-HANNAN-QUINN-1979].

    """
    rows = _rows(data)
    largest = require_non_negative_integer(max_order, name="max_order")
    if largest < 1:
        raise ValueError("max_order must be at least 1")
    k = len(rows[0])
    constant = 1 if include_constant else 0
    t = len(rows) - largest
    if t - (k * largest + constant) < k + 1:
        raise ValueError("too few observations for max_order")
    table: list[tuple[float, float, float, float]] = []
    for m in range(largest + 1):
        regressors, responses = _regression(rows, m, largest, include_constant)
        if regressors and not regressors[0]:
            residuals: list[Vector] = list(responses)
        else:
            _, _, residuals = _least_squares(regressors, responses)
        log_det = log_det_spd(_cross_product(residuals, t))
        table.append(_criteria(log_det, m, k, t, constant))
    columns = tuple(tuple(row[i] for row in table) for i in range(4))
    selected = {
        name: min(range(largest + 1), key=column.__getitem__)
        for name, column in zip(_CRITERIA, columns, strict=True)
    }
    return VAROrderSelection(
        orders=tuple(range(largest + 1)),
        aic=columns[0],
        bic=columns[1],
        hq=columns[2],
        fpe=columns[3],
        selected=selected,
        nobs=t,
    )


def simulate_var(
    coefficients: Sequence[Sequence[Sequence[float]]],
    covariance: Sequence[Sequence[float]],
    n: int,
    *,
    intercept: Sequence[float] | None = None,
    seed: int | RandomStream | None = None,
    burn_in: int = 200,
) -> tuple[Vector, ...]:
    """Simulate a Gaussian VAR(p) path with a burn-in.

    ``u_t = L z_t`` with ``L`` the lower Cholesky factor of ``Sigma_u`` and
    ``z_t`` standard normal from the child stream ``"var"``; the recursion
    starts from zeros and the first ``burn_in`` rows are discarded.

    Args:
        coefficients: ``A_1..A_p`` (each ``K x K``).
        covariance: Positive-definite ``Sigma_u``.
        n: Number of returned rows.
        intercept: ``nu`` (default zero).
        seed: Seed or stream; ``None`` is not reproducible.
        burn_in: Discarded initial rows.

    Returns:
        ``n`` rows of ``K`` values.

    Raises:
        ValueError: If shapes disagree, ``Sigma_u`` is not positive definite,
            or the VAR is not stable.

    Equation:
        [EQ-TS-VAR-001], [EQ-TS-VAR-003].

    References:
        [REF-LUTKEPOHL-2005, Chapter 2].

    """
    sigma = tuple(tuple(float(v) for v in row) for row in covariance)
    k = len(sigma)
    lags = tuple(tuple(tuple(float(v) for v in row) for row in a) for a in coefficients)
    if any(len(a) != k or any(len(row) != k for row in a) for a in lags):
        raise ValueError("every coefficient matrix must be K x K")
    nu = tuple(float(v) for v in intercept) if intercept is not None else (0.0,) * k
    if len(nu) != k:
        raise ValueError("intercept must have K entries")
    count = require_non_negative_integer(n, name="n")
    burn = require_non_negative_integer(burn_in, name="burn_in")
    p = len(lags)
    if p:
        size = k * p
        companion = [[0.0] * size for _ in range(size)]
        for lag, a in enumerate(lags):
            for i in range(k):
                for j in range(k):
                    companion[i][lag * k + j] = a[i][j]
        for i in range(k, size):
            companion[i][i - k] = 1.0
        if any(abs(value) >= 1.0 for value in _eigenvalues(companion)):
            raise ValueError(
                "the VAR is not stable (a companion eigenvalue has |z| >= 1)"
            )
    factor = cholesky(sigma)
    generator = resolve_stream(seed).child("var").generator()
    history: list[Vector] = [(0.0,) * k] * max(p, 1)
    output: list[Vector] = []
    for _ in range(burn + count):
        z = [generator.gauss(0.0, 1.0) for _ in range(k)]
        value = [
            nu[i] + math.fsum(factor[i][j] * z[j] for j in range(i + 1))
            for i in range(k)
        ]
        for lag, a in enumerate(lags, start=1):
            past = history[-lag]
            for i in range(k):
                value[i] += math.fsum(a[i][j] * past[j] for j in range(k))
        history.append(tuple(value))
        output.append(tuple(value))
    return tuple(output[burn:])
