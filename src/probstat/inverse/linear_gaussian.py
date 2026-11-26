"""Exact and low-rank linear-Gaussian Bayesian inversion.

For ``x ~ N(mu, C0)`` and ``y = A x + eps``, ``eps ~ N(0, Gamma)`` independent,
the posterior is Gaussian in closed form. With ``S = A C0 A' + Gamma`` the
data covariance, the posterior mean is ``mu + C0 A' S^-1 (y - A mu)``, the
covariance ``C0 - C0 A' S^-1 A C0`` and the log evidence the normal log
density of ``y`` under ``N(A mu, S)`` [EQ-INV-LG-001], [EQ-INV-LG-002].

The prior is never inverted. It is supplied through a factor ``P`` with
``C0 = P P'`` (either a covariance factor from :mod:`probstat.linalg` or a
rectangular :class:`~probstat.linalg.LinearOperator` ``n x r``, which allows a
singular, finite-rank prior); the noise through a covariance factor with the
whitening ``W = Gamma^-1/2``. Two exact organizations are provided:

- ``method="data_space"``: the dense ``m x m`` matrix ``S`` is assembled by
  applying ``A C0 A'`` to the ``m`` unit vectors (``O(m)`` applications of
  ``A``, ``A'`` and ``P P'``) and factored by Cholesky. Only ``S`` must be
  positive definite, so a singular prior is allowed. The covariance
  ``C0 - C0 A' S^-1 A C0`` is a difference and loses digits in strongly
  informed directions.
- ``method="parameter_space"``: with ``K = W A P`` the posterior of the
  whitened coordinate ``z`` (``x = mu + P z``, prior ``z ~ N(0, I)``) has
  precision ``I + K'K`` and covariance ``P (I + K'K)^-1 P'``, a form that
  cannot go negative [EQ-INV-LG-003]. The evidence uses the determinant lemma
  ``log det S = log det Gamma + log det(I + K'K)`` [EQ-INV-LG-004]. The dense
  system is ``r x r``, built with ``r`` applications of ``K'K``.

Both give the same posterior up to rounding. Exact conditional simulation
[EQ-INV-LG-005], the posterior predictive [EQ-INV-LG-006] and a low-rank
posterior for large problems are included. The low-rank form works on the
prior-preconditioned data-misfit Hessian ``H = K'K ~ V diag(lambda) V'``:
the whitened posterior covariance ``I - V diag(lambda/(1+lambda)) V'`` is
exact when all directions with ``lambda > 0`` are retained and keeps the
prior variance in unresolved directions otherwise [EQ-INV-LG-007],
[EQ-INV-LG-008]; the information gain and effective dimension follow from
the spectrum [EQ-INV-LG-009]. An ``n x n`` matrix is never formed by the
low-rank path.

Improper priors are not supported. Experimental (added in 0.28).

References:
    [REF-TARANTOLA-2005], [REF-KAIPIO-SOMERSALO-2007],
    [REF-BUI-THANH-ETAL-2013], [REF-SPANTINI-ETAL-2015],
    [REF-HALKO-MARTINSSON-TROPP-2011], [REF-STUART-2010].

"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.linalg import (
    BlockDiagonalCovariance,
    CallbackOperator,
    DenseCovariance,
    DiagonalCovariance,
    LinearOperator,
    Matrix,
    OperatorCounts,
    Vector,
    as_vector,
    cholesky,
    cholesky_solve,
    lanczos_eigenpairs,
    materialize,
    randomized_svd,
    require_operator,
    singular_value_decomposition,
    solve_lower,
    solve_upper,
    transpose,
)

_TWO_PI_LOG = math.log(2.0 * math.pi)
_PSD_ROUNDOFF = 1e-10

CovarianceFactor = DiagonalCovariance | DenseCovariance | BlockDiagonalCovariance
LinearGaussianMethod = Literal["data_space", "parameter_space"]
LowRankMethod = Literal["randomized", "lanczos"]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the correctly rounded inner product.

    References:
        [EQ-INV-LG-001].

    """
    return math.fsum(a * b for a, b in zip(left, right, strict=True))


def _add(left: Sequence[float], right: Sequence[float]) -> Vector:
    """Return ``left + right``.

    References:
        [EQ-INV-LG-001].

    """
    return tuple(a + b for a, b in zip(left, right, strict=True))


def _subtract(left: Sequence[float], right: Sequence[float]) -> Vector:
    """Return ``left - right``.

    References:
        [EQ-INV-LG-001].

    """
    return tuple(a - b for a, b in zip(left, right, strict=True))


def _unit(size: int, index: int) -> Vector:
    """Return the ``index``-th unit vector of length ``size``.

    References:
        [EQ-INV-LG-001].

    """
    return tuple(1.0 if i == index else 0.0 for i in range(size))


def _symmetric_from_columns(columns: list[Vector]) -> Matrix:
    """Return the symmetrized matrix whose ``j``-th column is ``columns[j]``.

    References:
        [EQ-INV-LG-001].

    """
    size = len(columns)
    return tuple(
        tuple(0.5 * (columns[j][i] + columns[i][j]) for j in range(size))
        for i in range(size)
    )


def _require_count(value: object, name: str) -> int:
    """Return a positive integer or raise.

    Raises:
        TypeError: If ``value`` is not an integer.
        ValueError: If ``value`` is not positive.

    References:
        [EQ-INV-LG-005].

    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _require_noise(noise: object, size: int) -> CovarianceFactor:
    """Return ``noise`` after checking that it is a covariance factor of ``size``.

    Raises:
        TypeError: If ``noise`` is not a covariance factor.
        ValueError: If its dimension differs from ``size``.

    References:
        [EQ-INV-LG-001].

    """
    if not isinstance(
        noise, (DiagonalCovariance, DenseCovariance, BlockDiagonalCovariance)
    ):
        raise TypeError("noise must be a probstat.linalg covariance factor")
    if noise.dimension != size:
        raise ValueError(f"noise has dimension {noise.dimension}, expected {size}")
    return noise


def _resolve_prior_factor(
    prior_covariance: object | None,
    prior_factor: object | None,
    size: int,
) -> LinearOperator:
    """Return the prior factor ``P`` (``size x r``) with ``C0 = P P'``.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If both or neither prior description is given, or the
            dimension is wrong.

    References:
        [EQ-INV-LG-001], [EQ-INV-LG-003].

    """
    if (prior_covariance is None) == (prior_factor is None):
        raise ValueError("give exactly one of prior_covariance and prior_factor")
    candidate = prior_covariance if prior_covariance is not None else prior_factor
    if isinstance(
        candidate, (DiagonalCovariance, DenseCovariance, BlockDiagonalCovariance)
    ):
        if candidate.dimension != size:
            raise ValueError(
                f"prior covariance has dimension {candidate.dimension}, expected {size}"
            )
        return candidate.factor_operator()
    if prior_covariance is not None:
        raise TypeError("prior_covariance must be a probstat.linalg covariance")
    factor = require_operator(prior_factor, name="prior_factor")
    rows, columns = factor.shape
    if rows != size:
        raise ValueError(f"prior_factor has {rows} rows, expected {size}")
    if columns < 1:
        raise ValueError("prior_factor must have at least one column")
    return factor


def _prepare(
    operator: object,
    data: Sequence[float],
    noise: object,
    prior_mean: Sequence[float] | None,
    prior_covariance: object | None,
    prior_factor: object | None,
) -> tuple[LinearOperator, Vector, CovarianceFactor, Vector, LinearOperator]:
    """Validate and normalize the inputs of a linear-Gaussian problem.

    Returns:
        ``(A, y, noise, mu, P)``.

    References:
        [EQ-INV-LG-001].

    """
    forward = require_operator(operator, name="operator")
    m, n = forward.shape
    if m < 1 or n < 1:
        raise ValueError("operator must have positive dimensions")
    y = as_vector(data, name="data")
    if len(y) != m:
        raise ValueError(f"data has length {len(y)}, expected {m}")
    checked_noise = _require_noise(noise, m)
    if prior_mean is None:
        mu: Vector = (0.0,) * n
    else:
        mu = as_vector(prior_mean, name="prior_mean")
        if len(mu) != n:
            raise ValueError(f"prior_mean has length {len(mu)}, expected {n}")
    factor = _resolve_prior_factor(prior_covariance, prior_factor, n)
    return forward, y, checked_noise, mu, factor


def _whitened_operator(
    forward: LinearOperator, noise: CovarianceFactor, factor: LinearOperator
) -> CallbackOperator:
    """Return ``K = W A P`` with ``rmatvec = K'``.

    References:
        [EQ-INV-LG-003].

    """
    m = forward.shape[0]
    r = factor.shape[1]

    def forward_action(z: Vector) -> Vector:
        """Return ``W A P z``.

        References:
            [EQ-INV-LG-003].

        """
        return noise.whiten(forward.matvec(factor.matvec(z)))

    def adjoint_action(v: Vector) -> Vector:
        """Return ``P' A' W' v``.

        References:
            [EQ-INV-LG-003].

        """
        return factor.rmatvec(forward.rmatvec(noise.whiten_transpose(v)))

    return CallbackOperator(
        (m, r), forward_action, adjoint_action, name="whitened forward map"
    )


def _prior_covariance_action(factor: LinearOperator, vector: Vector) -> Vector:
    """Return ``C0 v = P P' v``.

    References:
        [EQ-INV-LG-001].

    """
    return factor.matvec(factor.rmatvec(vector))


def _standard_normal(generator: random.Random, size: int) -> Vector:
    """Return ``size`` independent standard normal draws from ``generator``.

    References:
        [EQ-INV-LG-005].

    """
    return tuple(generator.gauss(0.0, 1.0) for _ in range(size))


@dataclass(frozen=True, slots=True)
class PredictiveDistribution:
    """Gaussian posterior predictive of a new linear functional ``H x``.

    Attributes:
        mean: ``H E[x | y]``.
        covariance: ``H C_post H'`` plus the optional new-observation noise
            covariance, as a dense symmetric matrix.
        includes_noise: Whether a noise covariance was added.

    References:
        [EQ-INV-LG-006].

    """

    mean: Vector
    covariance: Matrix
    includes_noise: bool


@dataclass(frozen=True, slots=True)
class GaussianLinearPosterior:
    """Exact posterior of a linear-Gaussian inverse problem.

    Attributes:
        mean: Posterior mean ``E[x | y]`` (length ``n``).
        log_evidence: ``log p(y)``, the normal log density of ``y`` under
            ``N(A mu, S)`` (dimensionless for a density of ``y``; its unit
            is the log of the unit of ``y`` to the power ``-m``).
        method: ``"data_space"`` or ``"parameter_space"``.
        prior_mean: Prior mean ``mu``.
        data: The observed ``y``.
        residual: ``y - A mu``.
        data_misfit: ``(y - A mu)' S^-1 (y - A mu)``.
        log_det_data_covariance: ``log det S``.
        whitened_mean: ``E[z | y]`` with ``x = mu + P z``.
        cholesky_factor: Lower Cholesky factor of the dense system, ``S``
            (``m x m``) for data space or ``I + K'K`` (``r x r``) for
            parameter space.
        operator: The forward operator ``A``.
        noise: The noise covariance factor.
        prior_factor: The prior factor ``P`` (``n x r``).

    References:
        [EQ-INV-LG-001], [EQ-INV-LG-002], [EQ-INV-LG-003], [EQ-INV-LG-004].

    """

    mean: Vector
    log_evidence: float
    method: str
    prior_mean: Vector
    data: Vector
    residual: Vector
    data_misfit: float
    log_det_data_covariance: float
    whitened_mean: Vector
    cholesky_factor: Matrix
    operator: LinearOperator
    noise: CovarianceFactor
    prior_factor: LinearOperator

    @property
    def parameter_dimension(self) -> int:
        """Return ``n``, the length of ``x``.

        References:
            [EQ-INV-LG-001].

        """
        return self.operator.shape[1]

    @property
    def data_dimension(self) -> int:
        """Return ``m``, the length of ``y``.

        References:
            [EQ-INV-LG-001].

        """
        return self.operator.shape[0]

    def covariance_action(self, vector: Sequence[float]) -> Vector:
        """Return ``C_post v`` without forming the covariance.

        Data space: ``C0 v - C0 A' S^-1 A C0 v``. Parameter space:
        ``P (I + K'K)^-1 P' v``.

        Args:
            vector: Vector of length ``n``.

        Returns:
            ``C_post v``.

        Raises:
            ValueError: If the length is wrong.

        References:
            [EQ-INV-LG-001], [EQ-INV-LG-003].

        """
        v = as_vector(vector, name="vector")
        if len(v) != self.parameter_dimension:
            raise ValueError(
                f"vector has length {len(v)}, expected {self.parameter_dimension}"
            )
        if self.method == "parameter_space":
            weights = cholesky_solve(self.cholesky_factor, self.prior_factor.rmatvec(v))
            return self.prior_factor.matvec(weights)
        prior_image = _prior_covariance_action(self.prior_factor, v)
        solved = cholesky_solve(self.cholesky_factor, self.operator.matvec(prior_image))
        correction = _prior_covariance_action(
            self.prior_factor, self.operator.rmatvec(solved)
        )
        return _subtract(prior_image, correction)

    def marginal_variances(self) -> Vector:
        """Return the exact diagonal of ``C_post`` (``n`` unit-vector passes).

        Parameter space uses ``||L^-1 P' e_i||^2``, which cannot be negative.
        Data space uses ``||P' e_i||^2 - ||L_S^-1 A C0 e_i||^2``; a negative
        result from cancellation is reported, never clipped.

        Returns:
            Vector of posterior variances.

        Raises:
            ValueError: If a data-space variance comes out negative; use
                ``method="parameter_space"`` for such strongly informed
                problems.

        References:
            [EQ-INV-LG-001], [EQ-INV-LG-003].

        """
        n = self.parameter_dimension
        variances: list[float] = []
        for i in range(n):
            reduced = self.prior_factor.rmatvec(_unit(n, i))
            if self.method == "parameter_space":
                whitened = solve_lower(self.cholesky_factor, reduced)
                variances.append(_dot(whitened, whitened))
                continue
            prior_image = self.prior_factor.matvec(reduced)
            projected = solve_lower(
                self.cholesky_factor, self.operator.matvec(prior_image)
            )
            value = _dot(reduced, reduced) - _dot(projected, projected)
            if value < 0.0:
                raise ValueError(
                    "negative data-space marginal variance from cancellation; "
                    "use method='parameter_space'"
                )
            variances.append(value)
        return tuple(variances)

    def predictive(
        self,
        new_operator: object,
        *,
        noise: object | None = None,
    ) -> PredictiveDistribution:
        """Return the posterior predictive of ``H x`` for a new operator ``H``.

        Args:
            new_operator: ``m' x n`` operator ``H`` (or dense matrix).
            noise: Optional covariance factor of dimension ``m'`` added to
                predict noisy new observations.

        Returns:
            :class:`PredictiveDistribution` with mean ``H mean`` and
            covariance ``H C_post H' (+ Gamma_new)``; it costs ``m'``
            covariance actions.

        Raises:
            TypeError: If an argument has the wrong type.
            ValueError: If a dimension disagrees.

        References:
            [EQ-INV-LG-006].

        """
        h = require_operator(new_operator, name="new_operator")
        m_new, n = h.shape
        if n != self.parameter_dimension:
            raise ValueError(
                f"new_operator has {n} columns, expected {self.parameter_dimension}"
            )
        extra = None if noise is None else _require_noise(noise, m_new)
        columns: list[Vector] = []
        for j in range(m_new):
            column = h.matvec(self.covariance_action(h.rmatvec(_unit(m_new, j))))
            if extra is not None:
                column = _add(column, extra.apply(_unit(m_new, j)))
            columns.append(column)
        return PredictiveDistribution(
            mean=h.matvec(self.mean),
            covariance=_symmetric_from_columns(columns),
            includes_noise=extra is not None,
        )

    def sample(
        self, count: int, seed: int | RandomStream | None = 0
    ) -> tuple[Vector, ...]:
        """Draw exact posterior samples.

        Data space uses conditional simulation: ``x0 = mu + P xi``,
        ``eps0 = C_noise eta`` and
        ``x = x0 + C0 A' S^-1 (y - A x0 - eps0)``. Parameter space draws
        ``z = z_mean + L^-T xi`` with ``L L' = I + K'K`` and returns
        ``mu + P z``.

        Args:
            count: Number of samples.
            seed: Integer seed or stream (``None`` is not reproducible).

        Returns:
            ``count`` vectors of length ``n``.

        Raises:
            TypeError: If ``count`` is not an integer.
            ValueError: If ``count`` is not positive.

        References:
            [EQ-INV-LG-005], [EQ-INV-LG-003].

        """
        draws = _require_count(count, "count")
        generator = resolve_stream(seed).child("linear_gaussian").generator()
        r = self.prior_factor.shape[1]
        m = self.data_dimension
        upper = transpose(self.cholesky_factor)
        samples: list[Vector] = []
        for _ in range(draws):
            xi = _standard_normal(generator, r)
            if self.method == "parameter_space":
                z = _add(self.whitened_mean, solve_upper(upper, xi))
                samples.append(_add(self.prior_mean, self.prior_factor.matvec(z)))
                continue
            eta = _standard_normal(generator, m)
            x0 = _add(self.prior_mean, self.prior_factor.matvec(xi))
            eps0 = self.noise.unwhiten(eta)
            gap = _subtract(_subtract(self.data, self.operator.matvec(x0)), eps0)
            solved = cholesky_solve(self.cholesky_factor, gap)
            update = _prior_covariance_action(
                self.prior_factor, self.operator.rmatvec(solved)
            )
            samples.append(_add(x0, update))
        return tuple(samples)


def _check_dense_limit(size: int, limit: int) -> None:
    """Raise if the dense system exceeds the limit.

    Raises:
        TypeError: If ``limit`` is not an integer.
        ValueError: If ``size`` exceeds ``limit``.

    References:
        [EQ-INV-LG-001].

    """
    _require_count(limit, "max_dense_dimension")
    if size > limit:
        raise ValueError(
            f"dense system of dimension {size} exceeds max_dense_dimension={limit}"
        )


def linear_gaussian_posterior(
    operator: object,
    data: Sequence[float],
    *,
    noise: object,
    prior_mean: Sequence[float] | None = None,
    prior_covariance: object | None = None,
    prior_factor: object | None = None,
    method: LinearGaussianMethod = "data_space",
    max_dense_dimension: int = 2000,
) -> GaussianLinearPosterior:
    """Return the exact posterior of ``y = A x + eps``, ``x ~ N(mu, P P')``.

    Data space assembles ``S = A C0 A' + Gamma`` with ``m`` applications of
    ``A C0 A'`` to unit vectors and solves by Cholesky; only ``S`` has to be
    positive definite, so a singular prior is allowed. Parameter space builds
    ``I + K'K`` (``K = W A P``) with ``r`` applications of ``K'K``. Neither
    path inverts ``C0``.

    Args:
        operator: ``A`` (``m x n``), a ``LinearOperator`` or dense matrix.
        data: Observation ``y`` of length ``m``.
        noise: Covariance factor of ``Gamma`` (dimension ``m``), required.
        prior_mean: ``mu`` (default zeros).
        prior_covariance: Covariance factor of dimension ``n`` giving
            ``P`` through its factor operator. Exclusive with
            ``prior_factor``.
        prior_factor: ``P`` as an ``n x r`` operator (``C0 = P P'`` may be
            singular). Exclusive with ``prior_covariance``.
        method: ``"data_space"`` or ``"parameter_space"``.
        max_dense_dimension: Largest allowed dense system dimension.

    Returns:
        :class:`GaussianLinearPosterior`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If dimensions disagree, the method is unknown, the dense
            system is too large, or ``S`` / ``I + K'K`` is not positive
            definite.

    References:
        [EQ-INV-LG-001], [EQ-INV-LG-002], [EQ-INV-LG-003], [EQ-INV-LG-004],
        [REF-TARANTOLA-2005], [REF-KAIPIO-SOMERSALO-2007].

    """
    if method not in ("data_space", "parameter_space"):
        raise ValueError("method must be 'data_space' or 'parameter_space'")
    forward, y, checked_noise, mu, factor = _prepare(
        operator, data, noise, prior_mean, prior_covariance, prior_factor
    )
    m = forward.shape[0]
    r = factor.shape[1]
    residual = _subtract(y, forward.matvec(mu))
    if method == "data_space":
        _check_dense_limit(m, max_dense_dimension)
        columns = [
            _add(
                forward.matvec(
                    _prior_covariance_action(factor, forward.rmatvec(_unit(m, j)))
                ),
                checked_noise.apply(_unit(m, j)),
            )
            for j in range(m)
        ]
        try:
            lower = cholesky(_symmetric_from_columns(columns))
        except ValueError as error:
            raise ValueError("data covariance S is not positive definite") from error
        solved = cholesky_solve(lower, residual)
        misfit = _dot(residual, solved)
        log_det = 2.0 * math.fsum(math.log(lower[i][i]) for i in range(m))
        z_mean = factor.rmatvec(forward.rmatvec(solved))
    else:
        _check_dense_limit(r, max_dense_dimension)
        k = _whitened_operator(forward, checked_noise, factor)
        columns = [
            _add(_unit(r, j), k.rmatvec(k.matvec(_unit(r, j)))) for j in range(r)
        ]
        try:
            lower = cholesky(_symmetric_from_columns(columns))
        except ValueError as error:
            raise ValueError("I + K'K is not positive definite") from error
        whitened_residual = checked_noise.whiten(residual)
        u = k.rmatvec(whitened_residual)
        z_mean = cholesky_solve(lower, u)
        reduced = solve_lower(lower, u)
        misfit = _dot(whitened_residual, whitened_residual) - _dot(reduced, reduced)
        log_det = checked_noise.log_det() + 2.0 * math.fsum(
            math.log(lower[i][i]) for i in range(r)
        )
    evidence = -0.5 * (misfit + log_det + m * _TWO_PI_LOG)
    return GaussianLinearPosterior(
        mean=_add(mu, factor.matvec(z_mean)),
        log_evidence=evidence,
        method=method,
        prior_mean=mu,
        data=y,
        residual=residual,
        data_misfit=misfit,
        log_det_data_covariance=log_det,
        whitened_mean=z_mean,
        cholesky_factor=lower,
        operator=forward,
        noise=checked_noise,
        prior_factor=factor,
    )


@dataclass(frozen=True, slots=True)
class LowRankGaussianPosterior:
    """Low-rank posterior from the prior-preconditioned data-misfit Hessian.

    The whitened posterior is approximated by
    ``z | y ~ N(V diag(1/(1+lambda)) V' K' W r, I - V diag(lambda/(1+lambda)) V')``
    with ``K = W A P``, ``r = y - A mu`` and the retained eigenpairs
    ``(lambda_i, v_i)`` of ``K'K``. This is exact if every direction with
    ``lambda > 0`` is retained; otherwise the covariance equals the prior in
    the unresolved directions and the mean omits their data information.

    Attributes:
        mean: Approximate posterior mean ``mu + P z_mean``.
        whitened_mean: ``z_mean``.
        eigenvalues: Retained ``lambda_i`` in descending order.
        eigenvectors: ``eigenvectors[i]`` is ``v_i`` (length ``r``).
        rank: Number of retained pairs.
        method: ``"randomized"`` or ``"lanczos"``.
        residual_norms: A posteriori residuals of the eigenpair computation
            (``||K v_i - sigma_i u_i||`` for randomized; ``||K'K v - lambda v||``
            for Lanczos).
        information_gain: ``1/2 sum log(1 + lambda_i)`` over the retained
            spectrum (a lower bound of the full value).
        effective_dimension: ``sum lambda_i / (1 + lambda_i)`` over the
            retained spectrum.
        smallest_retained_eigenvalue: ``lambda_rank``; a small value
            indicates the truncation has reached the weakly informed part of
            the spectrum. It is an indicator, not a certified error bound.
        counts: Applications of ``K`` and ``K'`` (each costs one forward
            and one adjoint pass of ``A``) spent by the eigensolver.
        prior_mean: ``mu``.
        prior_factor: ``P``.

    References:
        [EQ-INV-LG-007], [EQ-INV-LG-008], [EQ-INV-LG-009].

    """

    mean: Vector
    whitened_mean: Vector
    eigenvalues: Vector
    eigenvectors: Matrix
    rank: int
    method: str
    residual_norms: Vector
    information_gain: float
    effective_dimension: float
    smallest_retained_eigenvalue: float
    counts: OperatorCounts
    prior_mean: Vector
    prior_factor: LinearOperator

    def covariance_action(self, vector: Sequence[float]) -> Vector:
        """Return ``P (I - V diag(lambda/(1+lambda)) V') P' v``.

        Args:
            vector: Vector of length ``n``.

        Returns:
            Approximate ``C_post v``.

        Raises:
            ValueError: If the length is wrong.

        References:
            [EQ-INV-LG-007].

        """
        v = as_vector(vector, name="vector")
        if len(v) != self.prior_factor.shape[0]:
            raise ValueError(
                f"vector has length {len(v)}, expected {self.prior_factor.shape[0]}"
            )
        u = list(self.prior_factor.rmatvec(v))
        for lam, direction in zip(self.eigenvalues, self.eigenvectors, strict=True):
            coefficient = lam / (1.0 + lam) * _dot(direction, u)
            u = [a - coefficient * b for a, b in zip(u, direction, strict=True)]
        return self.prior_factor.matvec(tuple(u))

    def sample(
        self, count: int, seed: int | RandomStream | None = 0
    ) -> tuple[Vector, ...]:
        """Draw samples of the approximate posterior.

        Uses ``z = z_mean + xi + V diag(1/sqrt(1+lambda) - 1) V' xi`` with
        ``xi ~ N(0, I)``, whose covariance is the low-rank whitened
        posterior covariance, and returns ``mu + P z``.

        Args:
            count: Number of samples.
            seed: Integer seed or stream.

        Returns:
            ``count`` vectors of length ``n``.

        Raises:
            TypeError: If ``count`` is not an integer.
            ValueError: If ``count`` is not positive.

        References:
            [EQ-INV-LG-008].

        """
        draws = _require_count(count, "count")
        generator = resolve_stream(seed).child("low_rank").generator()
        r = self.prior_factor.shape[1]
        samples: list[Vector] = []
        for _ in range(draws):
            xi = _standard_normal(generator, r)
            z = list(xi)
            for lam, direction in zip(self.eigenvalues, self.eigenvectors, strict=True):
                coefficient = (1.0 / math.sqrt(1.0 + lam) - 1.0) * _dot(direction, xi)
                z = [a + coefficient * b for a, b in zip(z, direction, strict=True)]
            z_total = _add(self.whitened_mean, tuple(z))
            samples.append(_add(self.prior_mean, self.prior_factor.matvec(z_total)))
        return tuple(samples)


def low_rank_posterior(
    operator: object,
    data: Sequence[float],
    *,
    noise: object,
    prior_factor: object,
    prior_mean: Sequence[float] | None = None,
    rank: int,
    oversampling: int = 10,
    power_iterations: int = 2,
    seed: int | RandomStream | None = 0,
    method: LowRankMethod = "randomized",
) -> LowRankGaussianPosterior:
    """Return the low-rank posterior of a linear-Gaussian problem.

    The leading ``rank`` eigenpairs of ``H = K'K`` with ``K = W A P`` come
    from the randomized SVD of ``K`` (``lambda = sigma^2``) or from Lanczos on
    ``K'K``; only actions of ``A``, ``A'``, ``P``, ``P'`` and the noise
    whitening are used, and no ``n x n`` or ``r x r`` array is formed.

    Args:
        operator: ``A`` (``m x n``).
        data: Observation ``y``.
        noise: Covariance factor of the noise (dimension ``m``).
        prior_factor: ``P`` (``n x r``), an operator or covariance factor.
        prior_mean: ``mu`` (default zeros).
        rank: Number of eigenpairs to retain.
        oversampling: Extra samples (randomized) or extra Lanczos steps.
        power_iterations: Power passes (randomized only).
        seed: Seed or stream of the eigensolver.
        method: ``"randomized"`` or ``"lanczos"``.

    Returns:
        :class:`LowRankGaussianPosterior`. The ``rank`` is lower than
        requested only if Lanczos found an invariant subspace earlier.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If dimensions disagree, ``rank`` is invalid, or the
            method is unknown.

    References:
        [EQ-INV-LG-007], [EQ-INV-LG-008], [EQ-INV-LG-009],
        [REF-BUI-THANH-ETAL-2013], [REF-SPANTINI-ETAL-2015],
        [REF-HALKO-MARTINSSON-TROPP-2011].

    """
    if method not in ("randomized", "lanczos"):
        raise ValueError("method must be 'randomized' or 'lanczos'")
    forward, y, checked_noise, mu, factor = _prepare(
        operator, data, noise, prior_mean, None, prior_factor
    )
    r = factor.shape[1]
    k_count = _require_count(rank, "rank")
    k = _whitened_operator(forward, checked_noise, factor)
    residual = _subtract(y, forward.matvec(mu))
    eigenvalues: list[float]
    vectors: list[Vector]
    residuals: list[float]
    if method == "randomized":
        svd = randomized_svd(
            k,
            k_count,
            oversampling=oversampling,
            power_iterations=power_iterations,
            seed=seed,
        )
        eigenvalues = [s * s for s in svd.singular_values]
        vectors = [
            tuple(svd.v[i][j] for i in range(r)) for j in range(len(eigenvalues))
        ]
        residuals = list(svd.right_residuals)
        counts = svd.counts
    else:
        if k_count > r:
            raise ValueError("rank must not exceed the prior factor columns")
        if isinstance(oversampling, bool) or not isinstance(oversampling, int):
            raise TypeError("oversampling must be an integer")
        if oversampling < 0:
            raise ValueError("oversampling must be non-negative")
        gram = CallbackOperator(
            (r, r),
            lambda z: k.rmatvec(k.matvec(z)),
            lambda z: k.rmatvec(k.matvec(z)),
            name="K'K",
        )
        lanczos = lanczos_eigenpairs(gram, min(k_count + oversampling, r), seed=seed)
        kept = min(k_count, len(lanczos.eigenvalues))
        eigenvalues = list(lanczos.eigenvalues[:kept])
        vectors = [tuple(v) for v in lanczos.eigenvectors[:kept]]
        residuals = list(lanczos.residual_norms[:kept])
        counts = lanczos.counts
    scale = max(eigenvalues[0], 1.0) if eigenvalues else 1.0
    if min(eigenvalues) < -_PSD_ROUNDOFF * scale:
        raise ValueError("negative eigenvalue of the PSD Hessian beyond roundoff")
    # K'K is positive semidefinite: roundoff-level negatives are projected to 0.
    eigenvalues = [max(value, 0.0) for value in eigenvalues]
    reduced = k.rmatvec(checked_noise.whiten(residual))
    z = [0.0] * r
    for lam, direction in zip(eigenvalues, vectors, strict=True):
        coefficient = _dot(direction, reduced) / (1.0 + lam)
        z = [a + coefficient * b for a, b in zip(z, direction, strict=True)]
    return LowRankGaussianPosterior(
        mean=_add(mu, factor.matvec(tuple(z))),
        whitened_mean=tuple(z),
        eigenvalues=tuple(eigenvalues),
        eigenvectors=tuple(vectors),
        rank=len(eigenvalues),
        method=method,
        residual_norms=tuple(residuals),
        information_gain=0.5 * math.fsum(math.log1p(v) for v in eigenvalues),
        effective_dimension=math.fsum(v / (1.0 + v) for v in eigenvalues),
        smallest_retained_eigenvalue=eigenvalues[-1],
        counts=counts,
        prior_mean=mu,
        prior_factor=factor,
    )


@dataclass(frozen=True, slots=True)
class PriorWhitenedInformation:
    """Spectrum of the prior-preconditioned data-misfit Hessian ``K'K``.

    Attributes:
        eigenvalues: The ``r`` eigenvalues ``lambda_i`` of ``K'K`` in
            descending order (``r - min(m, r)`` structural zeros last).
        shrinkage_factors: ``1 / (1 + lambda_i)``, the posterior-to-prior
            variance ratio along each whitened direction.
        effective_dimension: ``sum lambda_i / (1 + lambda_i)``.
        mutual_information: ``1/2 sum log(1 + lambda_i)``, the information
            gain (mutual information of ``x`` and ``y``) in nats.

    References:
        [EQ-INV-LG-009].

    """

    eigenvalues: Vector
    shrinkage_factors: Vector
    effective_dimension: float
    mutual_information: float


def prior_whitened_information(
    operator: object,
    noise: object,
    prior_factor: object,
    *,
    max_entries: int = 1_000_000,
) -> PriorWhitenedInformation:
    """Return the spectrum of ``K'K`` by a dense SVD of the materialized ``K``.

    For small problems only: ``K = W A P`` is materialized with ``r``
    applications and decomposed by the Jacobi SVD, so ``lambda_i = sigma_i^2``.

    Args:
        operator: ``A`` (``m x n``).
        noise: Covariance factor of the noise.
        prior_factor: ``P`` (``n x r``), an operator or covariance factor.
        max_entries: Limit on ``m r`` passed to ``materialize``.

    Returns:
        :class:`PriorWhitenedInformation`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If dimensions disagree or the problem is too large.

    References:
        [EQ-INV-LG-009], [REF-SPANTINI-ETAL-2015],
        [REF-GOLUB-VANLOAN-2013, Section 8.6].

    """
    forward = require_operator(operator, name="operator")
    m, n = forward.shape
    checked_noise = _require_noise(noise, m)
    factor = _resolve_prior_factor(None, prior_factor, n)
    r = factor.shape[1]
    k = _whitened_operator(forward, checked_noise, factor)
    svd = singular_value_decomposition(materialize(k, max_entries=max_entries))
    values = [s * s for s in svd.singular_values]
    values.extend([0.0] * (r - len(values)))
    return PriorWhitenedInformation(
        eigenvalues=tuple(values),
        shrinkage_factors=tuple(1.0 / (1.0 + v) for v in values),
        effective_dimension=math.fsum(v / (1.0 + v) for v in values),
        mutual_information=0.5 * math.fsum(math.log1p(v) for v in values),
    )
