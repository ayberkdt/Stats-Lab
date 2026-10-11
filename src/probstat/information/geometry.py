"""Fisher information matrices and Fisher-Rao distances.

The Fisher information of a parametric family ``f(x; theta)`` is
``I(theta) = E[s s'] = -E[d^2 log f / d theta d theta']`` with the score
``s = d log f / d theta``. Closed forms are provided for common families in
their usual parameterizations, and :func:`numerical_fisher_information`
evaluates either expectation for any distribution-object family by
quantile quadrature, integer summation, or Monte Carlo.

The Fisher-Rao distance is the geodesic distance of the metric
``ds^2 = d theta' I(theta) d theta``. Closed forms used here:

* univariate normal ``(mu, sigma)``: a scaled Poincare half-plane;
* categorical, Bernoulli and binomial: the map ``p -> 2 sqrt(p)`` onto a
  sphere of radius two;
* Poisson ``2 |sqrt(l1) - sqrt(l2)|`` and exponential ``|log(l1/l2)|``;
* multivariate normal mean with a fixed covariance: Mahalanobis distance.

All matrices are returned as tuples of rows, including ``1 x 1`` ones.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence

from probstat.distributions import (
    Bernoulli,
    Beta,
    Binomial,
    Exponential,
    Gamma,
    IntegerSupport,
    Normal,
    Poisson,
)
from probstat.foundations import require_finite_real, require_probability
from probstat.linalg import (
    Matrix,
    Vector,
    as_vector,
    cholesky,
    inverse_spd,
    solve_lower,
)
from probstat.numerics import trigamma
from probstat.optimization import central_gradient, central_hessian

from ._quantile_quadrature import quantile_at, simpson_normal_nodes
from .differential import _spd

_SQRT_TWO = math.sqrt(2.0)


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


def _open_probability(value: float, *, name: str) -> float:
    """Validate a probability strictly inside ``(0, 1)``.

    Args:
        value: Candidate probability.
        name: Parameter name.

    Returns:
        The probability as a float.

    Raises:
        ValueError: If it is not in the open unit interval.

    References:
        Project validation policy. No equation is evaluated.

    """
    probability = require_probability(value, name=name)
    if not 0.0 < probability < 1.0:
        raise ValueError(f"{name} must lie strictly between 0 and 1")
    return probability


def normal_fisher_information(sigma: float) -> Matrix:
    """Return ``I(mu, sigma) = diag(1/sigma^2, 2/sigma^2)`` for ``N(mu, sigma^2)``.

    Args:
        sigma: Positive standard deviation.

    Returns:
        ``2 x 2`` Fisher information in ``(mu, sigma)``.

    Raises:
        ValueError: If ``sigma`` is not positive.

    Equation:
        [EQ-IG-FISHER-002].

    References:
        [REF-MIYAMOTO-ETAL-2024, Section 4.4.1], [REF-CASELLA-BERGER-2002,
        Section 7.3].

    """
    scale = _positive(sigma, name="sigma")
    precision = 1.0 / (scale * scale)
    return ((precision, 0.0), (0.0, 2.0 * precision))


def exponential_fisher_information(rate: float) -> Matrix:
    """Return ``I(lambda) = 1/lambda^2`` for ``Exponential(lambda)``.

    Args:
        rate: Positive rate.

    Returns:
        ``1 x 1`` Fisher information.

    Raises:
        ValueError: If the rate is not positive.

    Equation:
        [EQ-IG-FISHER-002].

    References:
        [REF-MIYAMOTO-ETAL-2024, Section 4.1].

    """
    lam = _positive(rate, name="rate")
    return ((1.0 / (lam * lam),),)


def gamma_fisher_information(shape: float, rate: float = 1.0) -> Matrix:
    """Return the ``Gamma(k, lambda)`` information in ``(k, lambda)``.

    ``log f = k log lambda - log Gamma(k) + (k - 1) log x - lambda x`` gives
    ``I = [[psi'(k), -1/lambda], [-1/lambda, k/lambda^2]]``.

    Args:
        shape: Positive shape ``k``.
        rate: Positive rate ``lambda``.

    Returns:
        ``2 x 2`` Fisher information.

    Raises:
        ValueError: If a parameter is not positive.

    Equation:
        [EQ-IG-FISHER-002].

    References:
        [REF-CASELLA-BERGER-2002, Section 7.3], [REF-NIST-DLMF-GAMMA,
        Section 5.15].

    """
    k = _positive(shape, name="shape")
    lam = _positive(rate, name="rate")
    return ((trigamma(k), -1.0 / lam), (-1.0 / lam, k / (lam * lam)))


def beta_fisher_information(alpha: float, beta: float) -> Matrix:
    """Return the ``Beta(a, b)`` information in ``(a, b)``.

    ``I = [[psi'(a) - psi'(a + b), -psi'(a + b)],
    [-psi'(a + b), psi'(b) - psi'(a + b)]]``.

    Args:
        alpha: Positive first shape ``a``.
        beta: Positive second shape ``b``.

    Returns:
        ``2 x 2`` Fisher information.

    Raises:
        ValueError: If a parameter is not positive.

    Equation:
        [EQ-IG-FISHER-002].

    References:
        [REF-CASELLA-BERGER-2002, Section 7.3], [REF-NIST-DLMF-GAMMA,
        Section 5.15].

    """
    a = _positive(alpha, name="alpha")
    b = _positive(beta, name="beta")
    both = trigamma(a + b)
    return ((trigamma(a) - both, -both), (-both, trigamma(b) - both))


def bernoulli_fisher_information(p: float) -> Matrix:
    """Return ``I(p) = 1/(p (1 - p))`` for ``Bernoulli(p)``.

    Args:
        p: Success probability in ``(0, 1)``.

    Returns:
        ``1 x 1`` Fisher information.

    Raises:
        ValueError: If ``p`` is not in ``(0, 1)``.

    Equation:
        [EQ-IG-FISHER-002].

    References:
        [REF-MIYAMOTO-ETAL-2024, Section 3.1].

    """
    probability = _open_probability(p, name="p")
    return ((1.0 / (probability * (1.0 - probability)),),)


def binomial_fisher_information(n: int, p: float) -> Matrix:
    """Return ``I(p) = n/(p (1 - p))`` for ``Binomial(n, p)`` with ``n`` known.

    Args:
        n: Positive number of trials.
        p: Success probability in ``(0, 1)``.

    Returns:
        ``1 x 1`` Fisher information.

    Raises:
        ValueError: If ``n`` is not positive or ``p`` is not in ``(0, 1)``.

    Equation:
        [EQ-IG-FISHER-002].

    References:
        [REF-MIYAMOTO-ETAL-2024, Section 3.1].

    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    probability = _open_probability(p, name="p")
    return ((n / (probability * (1.0 - probability)),),)


def poisson_fisher_information(lam: float) -> Matrix:
    """Return ``I(lambda) = 1/lambda`` for ``Poisson(lambda)``.

    Args:
        lam: Positive mean.

    Returns:
        ``1 x 1`` Fisher information.

    Raises:
        ValueError: If the mean is not positive.

    Equation:
        [EQ-IG-FISHER-002].

    References:
        [REF-MIYAMOTO-ETAL-2024, Section 3.2].

    """
    return ((1.0 / _positive(lam, name="lam"),),)


def _pmf(values: Iterable[object], *, name: str) -> Vector:
    """Validate a strictly positive PMF.

    Args:
        values: Candidate probabilities.
        name: Argument name.

    Returns:
        Probabilities as floats.

    Raises:
        ValueError: If an entry is not positive or the sum is not one within
            ``1e-9``.

    References:
        [REF-COVER-THOMAS-2006, Section 2.1]. No equation is evaluated.

    """
    probabilities = as_vector(values, name=name)
    if len(probabilities) < 2:
        raise ValueError(f"{name} must have at least two categories")
    if any(not value > 0.0 for value in probabilities):
        raise ValueError(f"{name} entries must be positive")
    if abs(math.fsum(probabilities) - 1.0) > 1e-9:
        raise ValueError(f"{name} must sum to one")
    return probabilities


def categorical_fisher_information(probabilities: Iterable[object]) -> Matrix:
    """Return the categorical information in the first ``k - 1`` probabilities.

    With ``p_k = 1 - sum_{i<k} p_i``, ``I_ij = delta_ij/p_i + 1/p_k``.

    Args:
        probabilities: Strictly positive PMF of length ``k >= 2``.

    Returns:
        ``(k - 1) x (k - 1)`` Fisher information.

    Raises:
        ValueError: If the PMF is invalid.

    Equation:
        [EQ-IG-FISHER-002].

    References:
        [REF-MIYAMOTO-ETAL-2024, Section 3.5].

    """
    p = _pmf(probabilities, name="probabilities")
    last = 1.0 / p[-1]
    size = len(p) - 1
    return tuple(
        tuple((1.0 / p[i] if i == j else 0.0) + last for j in range(size))
        for i in range(size)
    )


def multivariate_normal_mean_fisher_information(
    covariance: Iterable[Iterable[object]],
) -> Matrix:
    """Return ``I(m) = S^(-1)`` for the mean of ``N(m, S)`` with ``S`` known.

    Args:
        covariance: Symmetric positive-definite covariance ``S``.

    Returns:
        The precision matrix.

    Raises:
        ValueError: If the covariance is invalid.

    Equation:
        [EQ-IG-FISHER-002].

    References:
        [REF-CASELLA-BERGER-2002, Section 7.3].

    """
    return inverse_spd(_spd(covariance, name="covariance"))


def fisher_information(law: object) -> Matrix:
    """Return the closed-form Fisher information of a supported law.

    Parameterizations: ``Normal`` in ``(mu, sigma)``, ``Exponential`` in
    ``rate``, ``Gamma`` in ``(shape, rate)``, ``Beta`` in ``(alpha, beta)``,
    ``Bernoulli`` and ``Binomial`` (``n`` known) in ``p``, and ``Poisson`` in
    ``lam``.

    Args:
        law: Distribution object.

    Returns:
        Fisher information matrix.

    Raises:
        TypeError: If the law has no closed form here.

    Equation:
        [EQ-IG-FISHER-002].

    References:
        [REF-MIYAMOTO-ETAL-2024], [REF-CASELLA-BERGER-2002, Section 7.3].

    """
    if isinstance(law, Normal):
        return normal_fisher_information(law.sigma)
    if isinstance(law, Exponential):
        return exponential_fisher_information(law.rate)
    if isinstance(law, Gamma):
        return gamma_fisher_information(law.shape, law.rate)
    if isinstance(law, Beta):
        return beta_fisher_information(law.alpha, law.beta)
    if isinstance(law, Bernoulli):
        return bernoulli_fisher_information(law.p)
    if isinstance(law, Binomial):
        return binomial_fisher_information(law.n, law.p)
    if isinstance(law, Poisson):
        return poisson_fisher_information(law.lam)
    raise TypeError(f"no closed-form Fisher information for {type(law).__name__}")


def _log_density(law: object) -> Callable[[float], float]:
    """Return the log-density or log-mass method of a distribution object.

    Args:
        law: Distribution object with ``log_pdf`` or ``log_pmf``.

    Returns:
        The bound method.

    Raises:
        TypeError: If neither method exists.

    References:
        Project protocol policy. No equation is evaluated.

    """
    for attribute in ("log_pdf", "log_pmf"):
        method = getattr(law, attribute, None)
        if callable(method):
            return method  # type: ignore[no-any-return]
    raise TypeError(f"{type(law).__name__} has no log_pdf or log_pmf method")


def _integer_nodes(law: object, tail_mass: float) -> list[tuple[float, float]]:
    """Return ``(k, P(X = k))`` over the integer support up to a tail mass.

    Args:
        law: Discrete law with ``support`` and ``log_pmf``.
        tail_mass: Upper-tail probability that may be omitted.

    Returns:
        Support points and probabilities.

    Raises:
        TypeError: If the support is not an :class:`IntegerSupport`.

    Equation:
        [EQ-IG-FISHER-001].

    References:
        [REF-CASELLA-BERGER-2002, Section 7.3].

    """
    support = getattr(law, "support", None)
    if not isinstance(support, IntegerSupport):
        raise TypeError("discrete laws need an IntegerSupport")
    upper = support.upper
    if upper is None:
        upper = int(law.quantile(1.0 - tail_mass)) + 1  # type: ignore[attr-defined]
    log_pmf = _log_density(law)
    nodes = []
    for k in range(support.lower, upper + 1):
        value = log_pmf(k)
        if value > -math.inf:
            nodes.append((float(k), math.exp(value)))
    return nodes


def numerical_fisher_information(
    family: Callable[[Vector], object],
    parameters: Iterable[object],
    *,
    method: str = "score",
    integration: str = "quadrature",
    intervals: int = 400,
    z_max: float = 8.0,
    samples: int = 20_000,
    seed: int | None = None,
    relative_step: float | None = None,
    tail_mass: float = 1e-14,
) -> Matrix:
    """Return a numerical Fisher information for any distribution family.

    ``family(theta)`` must return an object with ``log_pdf`` (continuous)
    or ``log_pmf`` (discrete). Derivatives in ``theta`` are central finite
    differences (:func:`probstat.optimization.central_gradient` and
    :func:`~probstat.optimization.central_hessian`, steps relative to
    ``max(1, |theta_i|)``). The expectation over ``X ~ f(.; theta)`` is

    * ``integration="quadrature"``: continuous laws use Simpson quadrature in
      ``z`` with ``x = Q(Phi(z))``; discrete laws with an integer support sum
      the PMF up to an upper tail of ``tail_mass``;
    * ``integration="monte_carlo"``: the average over ``samples`` draws of
      ``law.sample(samples, seed=seed)``.

    Args:
        family: Map from a parameter tuple to a distribution object.
        parameters: Parameter vector ``theta``.
        method: ``"score"`` for ``E[s s']`` or ``"hessian"`` for
            ``-E[Hessian of log f]``.
        integration: ``"quadrature"`` or ``"monte_carlo"``.
        intervals: Positive even Simpson interval count (continuous case).
        z_max: Positive truncation point in the normal variable.
        samples: Positive Monte Carlo sample size.
        seed: Optional Monte Carlo seed.
        relative_step: Finite-difference step; defaults to ``1e-5`` for the
            score and ``1e-4`` for the Hessian.
        tail_mass: Omitted upper-tail mass for unbounded integer supports.

    Returns:
        Symmetric Fisher information matrix.

    Raises:
        ValueError: If an option is invalid or a log-density is not finite at
            a perturbed parameter.
        TypeError: If the family does not produce a supported object.

    Equation:
        [EQ-IG-FISHER-001].

    References:
        [REF-CASELLA-BERGER-2002, Section 7.3], [REF-MIYAMOTO-ETAL-2024,
        Section 1], [REF-AMARI-2016].

    """
    theta = as_vector(parameters, name="parameters")
    if not theta:
        raise ValueError("parameters must not be empty")
    if method not in ("score", "hessian"):
        raise ValueError("method must be 'score' or 'hessian'")
    if integration not in ("quadrature", "monte_carlo"):
        raise ValueError("integration must be 'quadrature' or 'monte_carlo'")
    step = (
        (1e-5 if method == "score" else 1e-4)
        if relative_step is None
        else (_positive(relative_step, name="relative_step"))
    )
    law = family(theta)
    continuous = callable(getattr(law, "log_pdf", None))
    _log_density(law)
    nodes: Sequence[tuple[float, float]]
    if integration == "monte_carlo":
        if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
            raise ValueError("samples must be a positive integer")
        draws = law.sample(samples, seed=seed)  # type: ignore[attr-defined]
        nodes = [(float(x), 1.0 / samples) for x in draws]
    elif continuous:
        nodes = [
            (quantile_at(law, z), weight)
            for z, weight in simpson_normal_nodes(intervals, z_max)
        ]
    else:
        nodes = _integer_nodes(law, _positive(tail_mass, name="tail_mass"))
    size = len(theta)
    total = [[0.0] * size for _ in range(size)]
    for x, weight in nodes:
        point = int(x) if not continuous else x

        def log_f(values: Sequence[float], point: float = point) -> float:
            """Return ``log f(x; theta)`` at the fixed observation.

            References:
                [EQ-IG-FISHER-001], [REF-CASELLA-BERGER-2002, Section 7.3].

            """
            return _log_density(family(tuple(values)))(point)

        if method == "score":
            score, _ = central_gradient(log_f, theta, relative_step=step)
            for i in range(size):
                for j in range(size):
                    total[i][j] += weight * score[i] * score[j]
        else:
            hessian, _ = central_hessian(log_f, theta, relative_step=step)
            for i in range(size):
                for j in range(size):
                    total[i][j] -= weight * hessian[i][j]
    return tuple(
        tuple(0.5 * (total[i][j] + total[j][i]) for j in range(size))
        for i in range(size)
    )


def fisher_rao_normal(mu0: float, sigma0: float, mu1: float, sigma1: float) -> float:
    """Return the Fisher-Rao distance between univariate normal laws.

    The metric ``(d mu^2 + 2 d sigma^2)/sigma^2`` is a Poincare half-plane
    scaled by ``sqrt 2`` in the coordinates ``(mu/sqrt 2, sigma)``, giving
    ``d = sqrt 2 arccosh(1 + ((mu0 - mu1)^2/2 + (sigma0 - sigma1)^2)/(2 sigma0
    sigma1))``, evaluated as ``2 sqrt 2 asinh(sqrt(delta/(4 sigma0 sigma1)))``
    to keep precision at small distances.

    Args:
        mu0: Mean of the first law.
        sigma0: Positive standard deviation of the first law.
        mu1: Mean of the second law.
        sigma1: Positive standard deviation of the second law.

    Returns:
        The Fisher-Rao distance.

    Raises:
        ValueError: If a standard deviation is not positive.

    Equation:
        [EQ-IG-FR-001].

    References:
        [REF-MIYAMOTO-ETAL-2024, Section 4.4.1], [REF-ATKINSON-MITCHELL-1981].

    """
    m0 = require_finite_real(mu0, name="mu0")
    m1 = require_finite_real(mu1, name="mu1")
    s0 = _positive(sigma0, name="sigma0")
    s1 = _positive(sigma1, name="sigma1")
    delta = 0.5 * (m0 - m1) ** 2 + (s0 - s1) ** 2
    return 2.0 * _SQRT_TWO * math.asinh(math.sqrt(delta / (4.0 * s0 * s1)))


def fisher_rao_categorical(
    probabilities0: Iterable[object], probabilities1: Iterable[object]
) -> float:
    """Return ``2 arccos(sum_i sqrt(p_i q_i))`` between categorical laws.

    ``p -> 2 sqrt(p)`` maps the simplex isometrically onto a sphere of
    radius two; the great-circle distance is evaluated as
    ``4 asin(|sqrt p - sqrt q| / 2)``, which equals the arccos form and is
    accurate for nearby laws. Zero probabilities (the boundary) are allowed.

    Args:
        probabilities0: PMF ``p``.
        probabilities1: PMF ``q`` on the same categories.

    Returns:
        The Fisher-Rao distance in ``[0, pi]``.

    Raises:
        ValueError: If a PMF is invalid or the lengths differ.

    Equation:
        [EQ-IG-FR-002].

    References:
        [REF-MIYAMOTO-ETAL-2024, Section 3.5].

    """
    p = as_vector(probabilities0, name="probabilities0")
    q = as_vector(probabilities1, name="probabilities1")
    if len(p) != len(q) or not p:
        raise ValueError("PMFs must be non-empty and of equal length")
    for values, name in ((p, "probabilities0"), (q, "probabilities1")):
        if any(value < 0.0 for value in values):
            raise ValueError(f"{name} entries must be non-negative")
        if abs(math.fsum(values) - 1.0) > 1e-9:
            raise ValueError(f"{name} must sum to one")
    # sqrt(a) - sqrt(b) = (a - b) / (sqrt(a) + sqrt(b)) avoids cancellation
    # for nearby laws.
    chord = math.sqrt(
        math.fsum(
            ((a - b) / (math.sqrt(a) + math.sqrt(b))) ** 2
            for a, b in zip(p, q, strict=True)
            if a + b > 0.0
        )
    )
    return 4.0 * math.asin(min(1.0, 0.5 * chord))


def fisher_rao_bernoulli(p0: float, p1: float) -> float:
    """Return ``2 |asin sqrt(p0) - asin sqrt(p1)|`` between Bernoulli laws.

    Args:
        p0: Success probability in ``[0, 1]``.
        p1: Success probability in ``[0, 1]``.

    Returns:
        The Fisher-Rao distance in ``[0, pi]``.

    Raises:
        ValueError: If a probability is outside ``[0, 1]``.

    Equation:
        [EQ-IG-FR-002].

    References:
        [REF-MIYAMOTO-ETAL-2024, Section 3.1].

    """
    return fisher_rao_binomial(1, p0, p1)


def fisher_rao_binomial(n: int, p0: float, p1: float) -> float:
    """Return ``2 sqrt(n) |asin sqrt(p0) - asin sqrt(p1)|`` (``n`` known).

    Args:
        n: Positive number of trials.
        p0: Success probability in ``[0, 1]``.
        p1: Success probability in ``[0, 1]``.

    Returns:
        The Fisher-Rao distance.

    Raises:
        ValueError: If ``n`` is not positive or a probability is invalid.

    Equation:
        [EQ-IG-FR-002].

    References:
        [REF-MIYAMOTO-ETAL-2024, Section 3.1].

    """
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    a = require_probability(p0, name="p0")
    b = require_probability(p1, name="p1")
    return 2.0 * math.sqrt(n) * abs(math.asin(math.sqrt(a)) - math.asin(math.sqrt(b)))


def fisher_rao_poisson(lam0: float, lam1: float) -> float:
    """Return ``2 |sqrt(lambda0) - sqrt(lambda1)|`` between Poisson laws.

    Args:
        lam0: Non-negative mean.
        lam1: Non-negative mean.

    Returns:
        The Fisher-Rao distance.

    Raises:
        ValueError: If a mean is negative.

    Equation:
        [EQ-IG-FR-003].

    References:
        [REF-MIYAMOTO-ETAL-2024, Section 3.2].

    """
    a = require_finite_real(lam0, name="lam0")
    b = require_finite_real(lam1, name="lam1")
    if a < 0.0 or b < 0.0:
        raise ValueError("Poisson means must be non-negative")
    return 2.0 * abs(math.sqrt(a) - math.sqrt(b))


def fisher_rao_exponential(rate0: float, rate1: float) -> float:
    """Return ``|log(lambda0/lambda1)|`` between exponential laws.

    Args:
        rate0: Positive rate.
        rate1: Positive rate.

    Returns:
        The Fisher-Rao distance.

    Raises:
        ValueError: If a rate is not positive.

    Equation:
        [EQ-IG-FR-003].

    References:
        [REF-MIYAMOTO-ETAL-2024, Section 4.1].

    """
    a = _positive(rate0, name="rate0")
    b = _positive(rate1, name="rate1")
    return abs(math.log(a) - math.log(b))


def fisher_rao_normal_mean(
    mean0: Iterable[object],
    mean1: Iterable[object],
    covariance: Iterable[Iterable[object]],
) -> float:
    """Return the Mahalanobis distance ``sqrt((m0 - m1)' S^-1 (m0 - m1))``.

    With ``S`` known, ``I(m) = S^-1`` is constant, so the metric is flat
    and geodesics are straight lines.

    Args:
        mean0: First mean.
        mean1: Second mean.
        covariance: Shared symmetric positive-definite covariance.

    Returns:
        The Fisher-Rao distance on the mean submanifold.

    Raises:
        ValueError: If dimensions disagree or the covariance is invalid.

    Equation:
        [EQ-IG-FR-003].

    References:
        [REF-MIYAMOTO-ETAL-2024, Section 1], [REF-CASELLA-BERGER-2002, Section 7.3].

    """
    m0 = as_vector(mean0, name="mean0")
    m1 = as_vector(mean1, name="mean1")
    matrix = _spd(covariance, name="covariance")
    if not (len(m0) == len(m1) == len(matrix)):
        raise ValueError("means and covariance must share one dimension")
    whitened = solve_lower(
        cholesky(matrix), tuple(a - b for a, b in zip(m0, m1, strict=True))
    )
    return math.sqrt(math.fsum(value * value for value in whitened))
