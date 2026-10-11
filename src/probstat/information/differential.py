"""Closed-form differential entropies and Gaussian Kullback-Leibler divergences.

Differential entropy ``h(X) = -E[log f(X)]`` is not invariant to a change of
variables and may be negative (``h = log(b - a) < 0`` for a uniform law on an
interval shorter than one). The functions below take parameters directly;
:func:`differential_entropy` dispatches on the distribution objects of
:mod:`probstat.distributions` without adding methods to them.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from probstat.distributions import (
    Beta,
    Exponential,
    Gamma,
    GeneralizedExtremeValue,
    GeneralizedPareto,
    Gumbel,
    LogNormal,
    Normal,
    Uniform,
)
from probstat.distributions._special import log1pmx, stirling_error
from probstat.foundations import require_finite_real
from probstat.linalg import (
    Matrix,
    Vector,
    as_matrix,
    as_vector,
    cholesky,
    cholesky_solve,
    solve_lower,
    symmetric_eigen,
)
from probstat.numerics import digamma, log_minus_digamma

from ._units import InformationUnit, unit_factor

_EULER_GAMMA = 0.5772156649015329
_LOG_TWO_PI_E = math.log(2.0 * math.pi) + 1.0
_SYMMETRY_TOLERANCE = 1e-10
_LOG_SQRT_TWO_PI = 0.5 * math.log(2.0 * math.pi)
_ASYMPTOTIC_SHAPE = 15.0
_SERIES_RATIO = 0.5


def _positive(value: float, *, name: str) -> float:
    """Validate a positive finite parameter.

    Args:
        value: Candidate parameter.
        name: Parameter name.

    Returns:
        The parameter as a float.

    Raises:
        ValueError: If it is not positive and finite.

    References:
        Project validation policy. No equation is evaluated.

    """
    number = require_finite_real(value, name=name)
    if not number > 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def normal_entropy(sigma: float, *, unit: InformationUnit = "nats") -> float:
    """Return ``h = log(sigma sqrt(2 pi e))`` for ``Normal(mu, sigma^2)``.

    Args:
        sigma: Positive standard deviation.
        unit: ``"nats"`` or ``"bits"``.

    Returns:
        Differential entropy (the maximum among laws with variance
        ``sigma^2``).

    Raises:
        ValueError: If ``sigma`` or the unit is invalid.

    Equation:
        [EQ-INFO-DH-001].

    References:
        [REF-COVER-THOMAS-2006, Section 8.1].

    """
    factor = unit_factor(unit)
    scale = _positive(sigma, name="sigma")
    return (math.log(scale) + 0.5 * _LOG_TWO_PI_E) * factor


def _spd(matrix: Iterable[Iterable[object]], *, name: str) -> Matrix:
    """Validate a symmetric positive-definite covariance matrix.

    Symmetry is checked relative to the matrix scale,
    ``|A_ij - A_ji| <= 1e-10 max_kl |A_kl|``, so the test is invariant to a
    rescaling of the covariance (an absolute floor would accept any
    asymmetry in a matrix whose entries are all tiny).

    Args:
        matrix: Candidate square matrix.
        name: Argument name.

    Returns:
        Validated matrix.

    Raises:
        ValueError: If the matrix is not square, symmetric, or positive
            definite.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 4.2]. No equation is evaluated.

    """
    checked = as_matrix(matrix, name=name, square=True)
    size = len(checked)
    scale = max(abs(value) for row in checked for value in row)
    for i in range(size):
        for j in range(i):
            a, b = checked[i][j], checked[j][i]
            if abs(a - b) > _SYMMETRY_TOLERANCE * scale:
                raise ValueError(f"{name} must be symmetric")
    cholesky(checked)
    return checked


def _log_det_from_lower(lower: Matrix) -> float:
    """Return ``log det A = 2 sum log L_ii`` from a Cholesky factor.

    Args:
        lower: Lower Cholesky factor.

    Returns:
        Log-determinant.

    Equation:
        [EQ-LINALG-006].

    References:
        [REF-GOLUB-VANLOAN-2013, Section 4.2].

    """
    return 2.0 * math.fsum(math.log(lower[i][i]) for i in range(len(lower)))


def multivariate_normal_entropy(
    covariance: Iterable[Iterable[object]], *, unit: InformationUnit = "nats"
) -> float:
    """Return ``h = (d log(2 pi e) + log det Sigma)/2`` for a ``d``-variate normal.

    Args:
        covariance: Symmetric positive-definite covariance matrix.
        unit: ``"nats"`` or ``"bits"``.

    Returns:
        Differential entropy.

    Raises:
        ValueError: If the covariance or unit is invalid.

    Equation:
        [EQ-INFO-DH-001].

    References:
        [REF-COVER-THOMAS-2006, Section 8.4].

    """
    factor = unit_factor(unit)
    matrix = _spd(covariance, name="covariance")
    log_det = _log_det_from_lower(cholesky(matrix))
    return 0.5 * (len(matrix) * _LOG_TWO_PI_E + log_det) * factor


def uniform_entropy(
    lower: float, upper: float, *, unit: InformationUnit = "nats"
) -> float:
    """Return ``h = log(b - a)`` for ``Uniform(a, b)``.

    Args:
        lower: Finite lower bound ``a``.
        upper: Finite upper bound ``b > a``.
        unit: ``"nats"`` or ``"bits"``.

    Returns:
        Differential entropy (negative when ``b - a < 1``).

    Raises:
        ValueError: If the bounds or unit are invalid.

    Equation:
        [EQ-INFO-DH-002].

    References:
        [REF-COVER-THOMAS-2006, Section 8.1].

    """
    factor = unit_factor(unit)
    law = Uniform(lower, upper)
    return math.log(law.upper - law.lower) * factor


def exponential_entropy(rate: float, *, unit: InformationUnit = "nats") -> float:
    """Return ``h = 1 - log lambda`` for ``Exponential(lambda)``.

    Args:
        rate: Positive rate ``lambda``.
        unit: ``"nats"`` or ``"bits"``.

    Returns:
        Differential entropy.

    Raises:
        ValueError: If the rate or unit is invalid.

    Equation:
        [EQ-INFO-DH-002].

    References:
        [REF-COVER-THOMAS-2006, Section 8.1].

    """
    factor = unit_factor(unit)
    return (1.0 - math.log(_positive(rate, name="rate"))) * factor


def gamma_entropy(
    shape: float, rate: float = 1.0, *, unit: InformationUnit = "nats"
) -> float:
    """Return ``h = k - log lambda + log Gamma(k) + (1 - k) psi(k)``.

    For ``k >= 15`` the terms of size ``k log k`` cancel analytically: with
    the Stirling remainder ``e(k) = log Gamma(k) - (k - 1/2) log k + k -
    log sqrt(2 pi)`` and ``L(k) = log k - psi(k)``,
    ``h = log sqrt(2 pi) + (log k)/2 + e(k) + (k - 1) L(k) - log lambda``,
    which keeps full relative accuracy for any large ``k`` (the direct sum
    loses all digits near ``k = 1e15``).

    Args:
        shape: Positive shape ``k``.
        rate: Positive rate ``lambda``.
        unit: ``"nats"`` or ``"bits"``.

    Returns:
        Differential entropy of ``Gamma(k, lambda)``.

    Raises:
        ValueError: If a parameter or the unit is invalid.

    Equation:
        [EQ-INFO-DH-003], [EQ-NUM-STIRLING-001], [EQ-NUM-DIGAMMA-001].

    References:
        [REF-COVER-THOMAS-2006, Chapter 8], [REF-NIST-DLMF-GAMMA, Sections
        5.2 and 5.11].

    """
    factor = unit_factor(unit)
    k = _positive(shape, name="shape")
    lam = _positive(rate, name="rate")
    if k < _ASYMPTOTIC_SHAPE:
        value = k + math.lgamma(k) + (1.0 - k) * digamma(k)
    else:
        value = (
            _LOG_SQRT_TWO_PI
            + 0.5 * math.log(k)
            + stirling_error(k)
            + (k - 1.0) * log_minus_digamma(k)
        )
    return (value - math.log(lam)) * factor


def beta_entropy(alpha: float, beta: float, *, unit: InformationUnit = "nats") -> float:
    """Return the ``Beta(a, b)`` entropy.

    ``h = log B(a, b) - (a - 1) psi(a) - (b - 1) psi(b) + (a + b - 2) psi(a + b)``.

    When ``max(a, b) >= 15`` each large argument ``x`` is expanded as
    ``log Gamma(x) = (x - 1/2) log x - x + log sqrt(2 pi) + e(x)`` and
    ``psi(x) = log x - L(x)`` (Stirling remainder ``e``, ``L = log - psi``);
    the terms of size ``x log x`` and ``x`` then cancel analytically, e.g.
    for ``a, b >= 15``
    ``h = log sqrt(2 pi) + (log a + log b - 3 log s)/2 + e(a) + e(b) - e(s)
    + (a-1) L(a) + (b-1) L(b) - (s-2) L(s)`` with ``s = a + b``. The direct
    sum loses all digits for shapes near ``1e15``.

    Args:
        alpha: Positive first shape ``a``.
        beta: Positive second shape ``b``.
        unit: ``"nats"`` or ``"bits"``.

    Returns:
        Differential entropy (``<= 0``, zero for the uniform ``a = b = 1``).

    Raises:
        ValueError: If a parameter or the unit is invalid.

    Equation:
        [EQ-INFO-DH-003], [EQ-NUM-STIRLING-001], [EQ-NUM-DIGAMMA-001].

    References:
        [REF-COVER-THOMAS-2006, Chapter 8], [REF-NIST-DLMF-GAMMA, Sections
        5.2 and 5.11].

    """
    factor = unit_factor(unit)
    a = _positive(alpha, name="alpha")
    b = _positive(beta, name="beta")
    small, large = min(a, b), max(a, b)
    if large < _ASYMPTOTIC_SHAPE:
        log_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
        value = (
            log_beta
            - (a - 1.0) * digamma(a)
            - (b - 1.0) * digamma(b)
            + (a + b - 2.0) * digamma(a + b)
        )
        return value * factor
    total = small + large
    # Large-argument part: (log x)/2 + e(x) + (x - 1) L(x) for the larger
    # shape, minus 3 (log s)/2 + e(s) + (s - 2) L(s) for the sum.
    value = (
        0.5 * math.log(large)
        - 1.5 * math.log(total)
        + stirling_error(large)
        - stirling_error(total)
        + (large - 1.0) * log_minus_digamma(large)
        - (total - 2.0) * log_minus_digamma(total)
    )
    if small >= _ASYMPTOTIC_SHAPE:
        # The linear terms -a - b + s cancel exactly; one log sqrt(2 pi) stays.
        value += (
            _LOG_SQRT_TWO_PI
            + 0.5 * math.log(small)
            + stirling_error(small)
            + (small - 1.0) * log_minus_digamma(small)
        )
    else:
        # The linear terms -large + s equal the small shape exactly.
        value += math.lgamma(small) - (small - 1.0) * digamma(small) + small
    return value * factor


def lognormal_entropy(
    mu: float, sigma: float, *, unit: InformationUnit = "nats"
) -> float:
    """Return ``h = mu + log(sigma sqrt(2 pi e))`` when ``log X ~ N(mu, sigma^2)``.

    Args:
        mu: Finite log-location.
        sigma: Positive log-scale.
        unit: ``"nats"`` or ``"bits"``.

    Returns:
        Differential entropy (``h(e^Y) = h(Y) + E[Y]``).

    Raises:
        ValueError: If a parameter or the unit is invalid.

    Equation:
        [EQ-INFO-DH-004].

    References:
        [REF-COVER-THOMAS-2006, Chapter 8].

    """
    factor = unit_factor(unit)
    location = require_finite_real(mu, name="mu")
    return (location + normal_entropy(sigma)) * factor


def gev_entropy(sigma: float, xi: float, *, unit: InformationUnit = "nats") -> float:
    """Return ``h = log sigma + gamma_E (1 + xi) + 1`` for the GEV law.

    With ``T = exp(-y)`` standard exponential, ``E[y] = gamma_E`` and
    ``E[T] = 1``; ``xi = 0`` gives the Gumbel value ``log sigma + gamma_E + 1``.

    Args:
        sigma: Positive scale.
        xi: Finite shape.
        unit: ``"nats"`` or ``"bits"``.

    Returns:
        Differential entropy.

    Raises:
        ValueError: If a parameter or the unit is invalid.

    Equation:
        [EQ-INFO-DH-005].

    References:
        [REF-COLES-2001, Section 3.1], [REF-COVER-THOMAS-2006, Chapter 8].

    """
    factor = unit_factor(unit)
    scale = _positive(sigma, name="sigma")
    shape = require_finite_real(xi, name="xi")
    return (math.log(scale) + _EULER_GAMMA * (1.0 + shape) + 1.0) * factor


def differential_entropy(law: object, *, unit: InformationUnit = "nats") -> float:
    """Return the closed-form differential entropy of a supported law.

    Supported: :class:`~probstat.distributions.Normal`, ``Uniform``,
    ``Exponential``, ``Gamma``, ``Beta``, ``LogNormal``, ``Gumbel``,
    ``GeneralizedExtremeValue``, and ``GeneralizedPareto``.

    Args:
        law: Distribution object.
        unit: ``"nats"`` or ``"bits"``.

    Returns:
        Differential entropy.

    Raises:
        TypeError: If the law has no closed form here.
        ValueError: If the unit is invalid.

    Equation:
        [EQ-INFO-DH-001], [EQ-INFO-DH-002], [EQ-INFO-DH-003],
        [EQ-INFO-DH-004], [EQ-INFO-DH-005], [EQ-DIST-GPD-003].

    References:
        [REF-COVER-THOMAS-2006, Chapter 8].

    """
    factor = unit_factor(unit)
    if isinstance(law, Normal):
        return normal_entropy(law.sigma, unit=unit)
    if isinstance(law, Uniform):
        return uniform_entropy(law.lower, law.upper, unit=unit)
    if isinstance(law, Exponential):
        return exponential_entropy(law.rate, unit=unit)
    if isinstance(law, Gamma):
        return gamma_entropy(law.shape, law.rate, unit=unit)
    if isinstance(law, Beta):
        return beta_entropy(law.alpha, law.beta, unit=unit)
    if isinstance(law, LogNormal):
        return lognormal_entropy(law.mu, law.sigma, unit=unit)
    if isinstance(law, Gumbel):
        return gev_entropy(law.beta, 0.0, unit=unit)
    if isinstance(law, GeneralizedExtremeValue):
        return gev_entropy(law.sigma, law.xi, unit=unit)
    if isinstance(law, GeneralizedPareto):
        return law.entropy * factor
    raise TypeError(f"no closed-form entropy for {type(law).__name__}")


def kl_normal(
    mu0: float,
    sigma0: float,
    mu1: float,
    sigma1: float,
    *,
    unit: InformationUnit = "nats",
) -> float:
    """Return ``D(N0 || N1)`` for univariate normal laws.

    ``log(s1/s0) + (s0^2 + (m0 - m1)^2)/(2 s1^2) - 1/2``.

    With ``u = (s0/s1)^2 - 1 = d (d + 2)``, ``d = (s0 - s1)/s1``, this is
    ``(u - log(1 + u))/2 + ((m0 - m1)/s1)^2/2``; for ``|u| <= 1/2`` the
    first term is evaluated as ``-log1pmx(u)/2`` so that nearly equal laws
    keep full relative accuracy (the direct form returns rounding noise of
    order ``1e-16`` once the divergence falls below it).

    Args:
        mu0: Mean of the reference law.
        sigma0: Positive standard deviation of the reference law.
        mu1: Mean of the approximating law.
        sigma1: Positive standard deviation of the approximating law.
        unit: ``"nats"`` or ``"bits"``.

    Returns:
        Divergence ``>= 0``.

    Raises:
        ValueError: If a parameter or the unit is invalid.

    Equation:
        [EQ-INFO-KLG-001], [EQ-NUM-STIRLING-001].

    References:
        [REF-COVER-THOMAS-2006, Section 8.5], [REF-DIDONATO-MORRIS-1992].

    """
    factor = unit_factor(unit)
    m0 = require_finite_real(mu0, name="mu0")
    m1 = require_finite_real(mu1, name="mu1")
    s0 = _positive(sigma0, name="sigma0")
    s1 = _positive(sigma1, name="sigma1")
    shift = (m0 - m1) / s1
    relative = (s0 - s1) / s1
    excess = relative * (relative + 2.0)
    if abs(excess) <= _SERIES_RATIO:
        variance_term = -log1pmx(excess)
    else:
        ratio = s0 / s1
        variance_term = ratio * ratio - 1.0 - 2.0 * math.log(ratio)
    value = 0.5 * (variance_term + shift * shift)
    return max(value, 0.0) * factor


def kl_multivariate_normal(
    mean0: Iterable[object],
    covariance0: Iterable[Iterable[object]],
    mean1: Iterable[object],
    covariance1: Iterable[Iterable[object]],
    *,
    unit: InformationUnit = "nats",
) -> float:
    """Return ``D(N(m0, S0) || N(m1, S1))`` in closed form.

    ``(tr(S1^-1 S0) + (m1 - m0)' S1^-1 (m1 - m0) - d + log det S1
    - log det S0)/2``, evaluated with Cholesky factors.

    With ``S1 = L1 L1'`` the covariance part equals
    ``sum_i (mu_i - log(1 + mu_i))`` over the eigenvalues ``mu_i`` of
    ``W = L1^-1 (S0 - S1) L1^-T`` (``1 + mu_i`` are the eigenvalues of
    ``S1^-1 S0``). When every ``|mu_i| <= 1/2`` the sum is evaluated as
    ``-sum log1pmx(mu_i)`` from the whitened difference, which keeps full
    relative accuracy for nearly equal laws; otherwise the trace and
    log-determinant form is used (no harmful cancellation there).

    Args:
        mean0: Mean of the reference law.
        covariance0: Symmetric positive-definite covariance of the reference.
        mean1: Mean of the approximating law.
        covariance1: Symmetric positive-definite covariance of the
            approximating law.
        unit: ``"nats"`` or ``"bits"``.

    Returns:
        Divergence ``>= 0``.

    Raises:
        ValueError: If dimensions disagree or a covariance is invalid.

    Equation:
        [EQ-INFO-KLG-002], [EQ-LINALG-005], [EQ-NUM-STIRLING-001].

    References:
        [REF-COVER-THOMAS-2006, Section 8.5], [REF-GOLUB-VANLOAN-2013,
        Section 8.5], [REF-DIDONATO-MORRIS-1992].

    """
    factor = unit_factor(unit)
    m0: Vector = as_vector(mean0, name="mean0")
    m1: Vector = as_vector(mean1, name="mean1")
    s0 = _spd(covariance0, name="covariance0")
    s1 = _spd(covariance1, name="covariance1")
    size = len(m0)
    if not (len(m1) == size and len(s0) == size and len(s1) == size):
        raise ValueError("means and covariances must share one dimension")
    lower0, lower1 = cholesky(s0), cholesky(s1)
    whitened = solve_lower(lower1, tuple(b - a for a, b in zip(m0, m1, strict=True)))
    mahalanobis = math.fsum(value * value for value in whitened)
    eigenvalues = _whitened_difference_eigenvalues(s0, s1, lower1)
    if all(abs(value) <= _SERIES_RATIO for value in eigenvalues):
        covariance_term = -math.fsum(log1pmx(value) for value in eigenvalues)
        return max(0.5 * (covariance_term + mahalanobis), 0.0) * factor
    trace = math.fsum(
        cholesky_solve(lower1, tuple(s0[i][j] for i in range(size)))[j]
        for j in range(size)
    )
    value = 0.5 * (
        trace
        + mahalanobis
        - size
        + _log_det_from_lower(lower1)
        - _log_det_from_lower(lower0)
    )
    return max(value, 0.0) * factor


def _whitened_difference_eigenvalues(
    covariance0: Matrix, covariance1: Matrix, lower1: Matrix
) -> Vector:
    """Return the eigenvalues of ``W = L1^-1 (S0 - S1) L1^-T``.

    The difference ``S0 - S1`` is formed before whitening, so eigenvalues of
    a small ``W`` carry relative (not absolute) accuracy.

    Args:
        covariance0: Reference covariance ``S0``.
        covariance1: Approximating covariance ``S1``.
        lower1: Lower Cholesky factor of ``S1``.

    Returns:
        Eigenvalues of the symmetric matrix ``W`` in ascending order.

    Equation:
        [EQ-INFO-KLG-002], [EQ-LINALG-005].

    References:
        [REF-GOLUB-VANLOAN-2013, Sections 4.2 and 8.5].

    """
    size = len(covariance0)
    difference = [
        [covariance0[i][j] - covariance1[i][j] for j in range(size)]
        for i in range(size)
    ]
    # Columns of X = L1^-1 D; then W = L1^-1 X' because D is symmetric.
    left = [
        solve_lower(lower1, tuple(difference[i][j] for i in range(size)))
        for j in range(size)
    ]
    columns = [
        solve_lower(lower1, tuple(left[i][j] for i in range(size))) for j in range(size)
    ]
    matrix = tuple(
        tuple(0.5 * (columns[j][i] + columns[i][j]) for j in range(size))
        for i in range(size)
    )
    if not any(value for row in matrix for value in row):
        return tuple(0.0 for _ in range(size))
    eigenvalues, _ = symmetric_eigen(matrix)
    return eigenvalues
