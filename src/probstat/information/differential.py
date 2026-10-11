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
from probstat.foundations import require_finite_real
from probstat.linalg import (
    Matrix,
    Vector,
    as_matrix,
    as_vector,
    cholesky,
    cholesky_solve,
    solve_lower,
)
from probstat.numerics import digamma

from ._units import InformationUnit, unit_factor

_EULER_GAMMA = 0.5772156649015329
_LOG_TWO_PI_E = math.log(2.0 * math.pi) + 1.0
_SYMMETRY_TOLERANCE = 1e-10


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
    for i in range(size):
        for j in range(i):
            a, b = checked[i][j], checked[j][i]
            if abs(a - b) > _SYMMETRY_TOLERANCE * max(1.0, abs(a), abs(b)):
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

    Args:
        shape: Positive shape ``k``.
        rate: Positive rate ``lambda``.
        unit: ``"nats"`` or ``"bits"``.

    Returns:
        Differential entropy of ``Gamma(k, lambda)``.

    Raises:
        ValueError: If a parameter or the unit is invalid.

    Equation:
        [EQ-INFO-DH-003].

    References:
        [REF-COVER-THOMAS-2006, Chapter 8], [REF-NIST-DLMF-GAMMA, Section 5.2].

    """
    factor = unit_factor(unit)
    k = _positive(shape, name="shape")
    lam = _positive(rate, name="rate")
    return (k - math.log(lam) + math.lgamma(k) + (1.0 - k) * digamma(k)) * factor


def beta_entropy(alpha: float, beta: float, *, unit: InformationUnit = "nats") -> float:
    """Return the ``Beta(a, b)`` entropy.

    ``h = log B(a, b) - (a - 1) psi(a) - (b - 1) psi(b) + (a + b - 2) psi(a + b)``.

    Args:
        alpha: Positive first shape ``a``.
        beta: Positive second shape ``b``.
        unit: ``"nats"`` or ``"bits"``.

    Returns:
        Differential entropy (``<= 0``, zero for the uniform ``a = b = 1``).

    Raises:
        ValueError: If a parameter or the unit is invalid.

    Equation:
        [EQ-INFO-DH-003].

    References:
        [REF-COVER-THOMAS-2006, Chapter 8], [REF-NIST-DLMF-GAMMA, Section 5.2].

    """
    factor = unit_factor(unit)
    a = _positive(alpha, name="alpha")
    b = _positive(beta, name="beta")
    log_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    value = (
        log_beta
        - (a - 1.0) * digamma(a)
        - (b - 1.0) * digamma(b)
        + (a + b - 2.0) * digamma(a + b)
    )
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
        [EQ-INFO-KLG-001].

    References:
        [REF-COVER-THOMAS-2006, Section 8.5].

    """
    factor = unit_factor(unit)
    m0 = require_finite_real(mu0, name="mu0")
    m1 = require_finite_real(mu1, name="mu1")
    s0 = _positive(sigma0, name="sigma0")
    s1 = _positive(sigma1, name="sigma1")
    ratio = s0 / s1
    shift = (m0 - m1) / s1
    value = -math.log(ratio) + 0.5 * (ratio * ratio + shift * shift) - 0.5
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
        [EQ-INFO-KLG-002].

    References:
        [REF-COVER-THOMAS-2006, Section 8.5].

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
    trace = math.fsum(
        cholesky_solve(lower1, tuple(s0[i][j] for i in range(size)))[j]
        for j in range(size)
    )
    whitened = solve_lower(lower1, tuple(b - a for a, b in zip(m0, m1, strict=True)))
    mahalanobis = math.fsum(value * value for value in whitened)
    value = 0.5 * (
        trace
        + mahalanobis
        - size
        + _log_det_from_lower(lower1)
        - _log_det_from_lower(lower0)
    )
    return max(value, 0.0) * factor
