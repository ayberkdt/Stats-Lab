"""Extended observation laws: counts, intervals, outliers, t, state-dependent noise.

This module extends :mod:`probstat.inverse.observation_laws` with laws that
share its :class:`~probstat.inverse.observation_laws.ObservationLaw` interface
(``log_likelihood``, ``score``, ``fisher_information``, ``simulate``,
``irls_system``, ``restrict``), so that every law below can be passed to
:func:`~probstat.inverse.observation_laws.maximum_likelihood_linear` and
wrapped by :class:`~probstat.inverse.observation_laws.MissingMask` wherever the
mathematics allows it. Every log-likelihood is fully normalized unless its
docstring says otherwise.

- :class:`NegativeBinomialLaw`: mean-dispersion parametrisation
  ``Var = mu + mu^2 / theta`` with a log link; ``theta -> inf`` is the Poisson
  law [EQ-INV-OLX-001], [EQ-INV-OLX-002].
- :class:`IntervalCensoredGaussianLaw`: per-observation intervals ``[l, u]``
  (one-sided, two-sided or exact) with the stable log-difference of normal
  CDFs [EQ-INV-OLX-003], [EQ-INV-OLX-004].
- :class:`GaussianMixtureOutlierLaw`: the contaminated normal
  ``(1 - eps) N(0, s^2) + eps N(0, (kappa s)^2)`` with responsibilities
  [EQ-INV-OLX-005], [EQ-INV-OLX-006].
- :class:`HuberLaw`: the Huber loss turned into the density
  ``exp(-rho_k(u)) / Z_k``. The Huber *loss* is an M-estimation criterion; it
  is a generative model only because the exact constant ``Z_k`` is included
  here. :meth:`HuberLaw.pseudo_log_likelihood` returns the unnormalized
  criterion; :func:`huber_sandwich_covariance` gives the robust covariance
  [EQ-INV-OLX-007], [EQ-INV-OLX-008].
- :class:`MultivariateStudentTLaw`: joint ``t_nu(eta, Sigma)`` with a full
  scatter matrix (NOT a product of univariate t laws and NOT a covariance)
  [EQ-INV-OLX-009], [EQ-INV-OLX-010].
- :class:`HeteroscedasticGaussianLaw` and :class:`StateDependentCovarianceLaw`:
  Gaussian noise whose variance or covariance depends on the prediction; the
  gradient contains the log-determinant and covariance-derivative terms
  [EQ-INV-OLX-011], [EQ-INV-OLX-012].
- :class:`ProfiledScaleGaussianLaw` (profile likelihood in an unknown common
  scale) [EQ-INV-OLX-013] and :class:`MarginalScaleGaussianLaw` (the scale is
  integrated out under an inverse-gamma prior, giving a multivariate
  Student-t marginal) [EQ-INV-OLX-014].

Experimental (added in 0.28).

References:
    [REF-MCCULLAGH-NELDER-1989], [REF-HUBER-1964], [REF-HUBER-1981],
    [REF-KOTZ-NADARAJAH-2004], [REF-GELMAN-ETAL-2013],
    [REF-LANGE-LITTLE-TAYLOR-1989], [REF-WHITE-1982].

"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Final

from probstat.distributions import Normal, Poisson
from probstat.distributions._special import log_gamma_ratio
from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.linalg import (
    DenseCovariance,
    DenseOperator,
    LinearOperator,
    Matrix,
    ScaledOperator,
    Vector,
    cholesky,
    cholesky_solve,
    inverse_spd,
    materialize,
    require_operator,
)
from probstat.linalg.covariance_operators import CovarianceFactor, is_covariance_factor

from .observation_laws import (
    GaussianLaw,
    ObservationLaw,
    _check_indices,
    _diagonal_whitening,
    _positive_vector,
    _real_vector,
    _scoring_response,
    _softplus,
    _std_log_pdf,
)

__all__ = [
    "GaussianMixtureOutlierLaw",
    "HeteroscedasticGaussianLaw",
    "HuberLaw",
    "IntervalCensoredGaussianLaw",
    "MarginalScaleGaussianLaw",
    "MultivariateStudentTLaw",
    "NegativeBinomialLaw",
    "ProfiledScaleGaussianLaw",
    "StateDependentCovarianceLaw",
    "huber_sandwich_covariance",
]

_STD: Final = Normal(0.0, 1.0)
_LOG_2PI: Final = math.log(2.0 * math.pi)
_SQRT_2PI: Final = math.sqrt(2.0 * math.pi)
_LOG_RATE_BOUND: Final = 700.0

# Five-point Gauss-Legendre rule on [-1, 1] (Davis and Rabinowitz).
_GL_NODES: Final = (
    -0.9061798459386640,
    -0.5384693101056831,
    0.0,
    0.5384693101056831,
    0.9061798459386640,
)
_GL_WEIGHTS: Final = (
    0.2369268850561891,
    0.4786286704993665,
    0.5688888888888889,
    0.4786286704993665,
    0.2369268850561891,
)
_NARROW_INTERVAL: Final = 0.1


def _real(value: float, name: str) -> float:
    """Return a finite real number or raise.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].

    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _positive(value: float, name: str) -> float:
    """Return a strictly positive finite real or raise.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].

    """
    number = _real(value, name)
    if number <= 0.0:
        raise ValueError(f"{name} must be strictly positive")
    return number


def _logsumexp(values: Sequence[float]) -> float:
    """Return ``log(sum(exp(values)))`` without overflow.

    References:
        [EQ-INV-OLX-005].

    """
    peak = max(values)
    if peak == -math.inf:
        return -math.inf
    return peak + math.log(math.fsum(math.exp(v - peak) for v in values))


def _log_diff_exp(upper: float, lower: float) -> float:
    """Return ``log(exp(upper) - exp(lower))`` for ``upper >= lower``.

    References:
        [EQ-INV-OLX-003].

    """
    if lower == -math.inf:
        return upper
    if upper <= lower:
        return -math.inf
    return upper + math.log1p(-math.exp(lower - upper))


def _symmetric_matrix(
    matrix: Sequence[Sequence[float]], size: int, name: str
) -> Matrix:
    """Return a validated symmetric ``size x size`` matrix of floats.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].

    """
    rows = tuple(_real_vector(row, name, size=size) for row in matrix)
    if len(rows) != size:
        raise ValueError(f"{name} must be {size} x {size}")
    scale = max(abs(v) for row in rows for v in row)
    for i in range(size):
        for j in range(i):
            if abs(rows[i][j] - rows[j][i]) > 1e-10 * max(scale, 1e-300):
                raise ValueError(f"{name} must be symmetric")
    return tuple(
        tuple(0.5 * (rows[i][j] + rows[j][i]) for j in range(size)) for i in range(size)
    )


def _inverse_columns(factor: CovarianceFactor, size: int) -> Matrix:
    """Return the precision matrix ``Gamma^-1`` by ``size`` solves.

    References:
        [EQ-INV-OLX-012].

    """
    columns: list[Vector] = []
    for j in range(size):
        unit = [0.0] * size
        unit[j] = 1.0
        columns.append(tuple(float(v) for v in factor.solve(unit)))
    return tuple(
        tuple(0.5 * (columns[j][i] + columns[i][j]) for j in range(size))
        for i in range(size)
    )


def _matmul(a: Matrix, b: Matrix) -> Matrix:
    """Return the dense product ``a b`` (small matrices only).

    References:
        [REF-GOLUB-VANLOAN-2013].

    """
    cols = list(zip(*b, strict=True))
    return tuple(
        tuple(math.fsum(x * y for x, y in zip(row, col, strict=True)) for col in cols)
        for row in a
    )


def _trace_product(a: Matrix, b: Matrix) -> float:
    """Return ``tr(a b)`` without forming the product.

    References:
        [EQ-INV-OLX-012].

    """
    return math.fsum(a[i][k] * b[k][i] for i in range(len(a)) for k in range(len(a[i])))


# ---------------------------------------------------------------------------
# 1. Negative binomial
# ---------------------------------------------------------------------------


class NegativeBinomialLaw(ObservationLaw):
    """Negative binomial counts in the mean-dispersion parametrisation.

    ``y_i | mu_i ~ NB(mu_i, theta)`` with ``mu_i = exp(eta_i)`` (log link),
    ``E y = mu`` and ``Var y = mu + mu^2 / theta``. Equivalently
    ``y | lambda ~ Poisson(lambda)`` with ``lambda ~ Gamma(theta, mu / theta)``.
    In terms of :class:`probstat.distributions.NegativeBinomial` (integer
    ``r`` successes, success probability ``p``) this is ``r = theta`` and
    ``p = theta / (theta + mu)``; here ``theta`` may be any positive real. The
    limit ``theta -> inf`` is the Poisson law with rate ``mu``
    [EQ-INV-OLX-001], [EQ-INV-OLX-002].

    Args:
        dispersion: The size (inverse-overdispersion) ``theta > 0``, or
            ``math.inf`` for the Poisson limit. The overdispersion coefficient
            is ``1 / theta``.

    Raises:
        TypeError: If ``dispersion`` is not a real number.
        ValueError: If ``dispersion`` is not positive or is ``nan``.

    References:
        [EQ-INV-OLX-001], [EQ-INV-OLX-002], [REF-CAMERON-TRIVEDI-2013],
        [REF-MCCULLAGH-NELDER-1989].

    """

    def __init__(self, dispersion: float) -> None:
        """Validate the dispersion.

        References:
            [EQ-INV-OLX-001].

        """
        if isinstance(dispersion, bool) or not isinstance(dispersion, (int, float)):
            raise TypeError("dispersion must be a real number")
        theta = float(dispersion)
        if math.isnan(theta) or theta <= 0.0:
            raise ValueError("dispersion must be positive (or math.inf)")
        self.dispersion = theta

    @property
    def size(self) -> None:
        """Return ``None``: any observation length is valid.

        References:
            [EQ-INV-OLX-001].

        """
        return None

    def _counts(
        self, y: Sequence[float], eta: Sequence[float]
    ) -> tuple[Vector, Vector]:
        """Validate nonnegative integer counts and ``|eta| <= 700``.

        References:
            [EQ-INV-OLX-001].

        """
        y_vector, eta_vector = self._pair(y, eta)
        for value in y_vector:
            if value < 0.0 or value != math.floor(value):
                raise ValueError(
                    "negative binomial counts must be nonnegative integers"
                )
        if any(abs(e) > _LOG_RATE_BOUND for e in eta_vector):
            raise ValueError("log link requires |eta| <= 700")
        return y_vector, eta_vector

    def log_likelihood(self, y: Sequence[float], eta: Sequence[float]) -> float:
        """Return ``sum_i log NB(y_i; mu_i = exp(eta_i), theta)``.

        ``log p = log Gamma(y + theta) - log Gamma(theta) - log y!
        - theta log(1 + mu/theta) + y log(mu / (theta + mu))``, evaluated with
        a stable gamma ratio and ``softplus`` so large ``theta`` and extreme
        ``eta`` neither overflow nor cancel.

        Args:
            y: Nonnegative integer counts.
            eta: Log means.

        Returns:
            The normalized log-likelihood.

        Raises:
            ValueError: For negative or non-integer counts or ``|eta| > 700``.

        References:
            [EQ-INV-OLX-001], [EQ-INV-OLX-002].

        """
        y_vector, eta_vector = self._counts(y, eta)
        theta = self.dispersion
        terms: list[float] = []
        for k, e in zip(y_vector, eta_vector, strict=True):
            if math.isinf(theta):
                terms.append(k * e - math.exp(e) - math.lgamma(k + 1.0))
                continue
            log_theta = math.log(theta)
            soft = _softplus(e - log_theta)
            ratio = log_gamma_ratio(theta, k) if k > 0.0 else 0.0
            terms.append(
                ratio - math.lgamma(k + 1.0) - theta * soft + k * (e - log_theta - soft)
            )
        return math.fsum(terms)

    def _mean_over_theta(self, eta: Vector) -> list[float]:
        """Return ``mu / theta`` (zero in the Poisson limit).

        References:
            [EQ-INV-OLX-002].

        """
        if math.isinf(self.dispersion):
            return [0.0] * len(eta)
        return [math.exp(e) / self.dispersion for e in eta]

    def score(self, y: Sequence[float], eta: Sequence[float]) -> Vector:
        """Return ``(y - mu) / (1 + mu / theta)``.

        Args:
            y: Counts.
            eta: Log means.

        Returns:
            The score ``d log p / d eta``.

        References:
            [EQ-INV-OLX-002].

        """
        y_vector, eta_vector = self._counts(y, eta)
        ratios = self._mean_over_theta(eta_vector)
        return tuple(
            (k - math.exp(e)) / (1.0 + q)
            for k, e, q in zip(y_vector, eta_vector, ratios, strict=True)
        )

    def fisher_information(self, eta: Sequence[float]) -> Vector:
        """Return ``mu / (1 + mu / theta)`` (``mu`` for the Poisson limit).

        Args:
            eta: Log means.

        Returns:
            The expected information with respect to ``eta``.

        Raises:
            ValueError: If ``|eta| > 700``.

        References:
            [EQ-INV-OLX-002].

        """
        eta_vector = _real_vector(eta, "eta")
        if any(abs(e) > _LOG_RATE_BOUND for e in eta_vector):
            raise ValueError("log link requires |eta| <= 700")
        ratios = self._mean_over_theta(eta_vector)
        return tuple(
            math.exp(e) / (1.0 + q) for e, q in zip(eta_vector, ratios, strict=True)
        )

    def simulate(
        self, eta: Sequence[float], seed: int | RandomStream | None = None
    ) -> Vector:
        """Draw counts from the gamma-Poisson mixture.

        Args:
            eta: Log means.
            seed: Integer seed or stream.

        Returns:
            Independent counts as floats.

        References:
            [EQ-INV-OLX-001].

        """
        eta_vector = _real_vector(eta, "eta")
        if any(abs(e) > _LOG_RATE_BOUND for e in eta_vector):
            raise ValueError("log link requires |eta| <= 700")
        stream = resolve_stream(seed)
        draws: list[float] = []
        for i, e in enumerate(eta_vector):
            rate = math.exp(e)
            if math.isfinite(self.dispersion):
                generator = stream.child(2 * i).generator()
                rate *= generator.gammavariate(self.dispersion, 1.0 / self.dispersion)
            if rate <= 0.0:
                draws.append(0.0)
                continue
            draws.append(
                float(Poisson(rate).sample(1, seed=stream.child(2 * i + 1))[0])
            )
        return tuple(draws)

    def irls_system(
        self, y: Sequence[float], eta: Sequence[float]
    ) -> tuple[LinearOperator, Vector]:
        """Return Fisher-scoring weights and working response.

        Args:
            y: Counts.
            eta: Log means.

        Returns:
            ``(diag(sqrt(I)), eta + score / I)``.

        References:
            [EQ-INV-OLX-002], [REF-MCCULLAGH-NELDER-1989].

        """
        eta_vector = _real_vector(eta, "eta")
        weights, response = _scoring_response(
            eta_vector, self.score(y, eta), self.fisher_information(eta)
        )
        return _diagonal_whitening(weights), response

    def restrict(self, indices: Sequence[int]) -> NegativeBinomialLaw:
        """Return the same law (it is elementwise).

        Args:
            indices: Strictly increasing indices.

        Returns:
            ``self``.

        References:
            [EQ-INV-OLX-001].

        """
        _check_indices(indices, None)
        return self


# ---------------------------------------------------------------------------
# 2. Interval-censored Gaussian
# ---------------------------------------------------------------------------


def _log_interval_mass(a: float, b: float, width: float | None = None) -> float:
    """Return ``log(Phi(b) - Phi(a))`` for standardized ``a < b``.

    Infinite limits are allowed. Wide intervals use log-CDF / log-survival
    differences in the lower or upper tail (finite far beyond ``erfc``
    underflow); narrow ones (``(b - a)(1 + max|.|) < 0.1``) use a five-point
    Gauss-Legendre rule on the log scale, which avoids the cancellation of
    two nearly equal CDF values. ``width`` may carry the exactly computed
    ``(u - l) / s`` so that a very narrow interval keeps its full relative
    width after standardization.

    References:
        [EQ-INV-OLX-003], [REF-DAVIS-RABINOWITZ-1984].

    """
    if a == -math.inf and b == math.inf:
        return 0.0
    if a == -math.inf:
        return _STD.log_cdf(b)
    if b == math.inf:
        return _STD.log_sf(a)
    if width is None:
        width = b - a
    if width * (1.0 + max(abs(a), abs(b))) < _NARROW_INTERVAL:
        mid, half = a + 0.5 * width, 0.5 * width
        terms = [
            math.log(w) + _std_log_pdf(mid + half * x)
            for x, w in zip(_GL_NODES, _GL_WEIGHTS, strict=True)
        ]
        return math.log(half) + _logsumexp(terms)
    if a > 0.0:
        return _log_diff_exp(_STD.log_sf(a), _STD.log_sf(b))
    if b < 0.0:
        return _log_diff_exp(_STD.log_cdf(b), _STD.log_cdf(a))
    mass = _STD.cdf(b) - _STD.cdf(a)
    return math.log(mass) if mass > 0.0 else -math.inf


class IntervalCensoredGaussianLaw(ObservationLaw):
    """Gaussian errors observed only through intervals ``[l_i, u_i]``.

    Entry ``i`` contributes

    - ``log N(l_i; eta_i, s_i^2)`` if ``l_i == u_i`` (an exact observation);
    - ``log(Phi(b_i) - Phi(a_i))`` with ``a = (l - eta)/s``, ``b = (u - eta)/s``
      if ``l_i < u_i`` (interval censoring); an infinite limit gives left or
      right censoring and ``(-inf, inf)`` an uninformative entry (``0``).

    The intervals are data fixed at construction; the ``y`` argument of the
    interface is a *representative* value that must lie in ``[l_i, u_i]``
    (see :meth:`representative`) and is otherwise not used, so the
    likelihood is ``P(l < Y <= u)`` and the censored entries are never
    imputed [EQ-INV-OLX-003]. ``fisher_information`` returns the OBSERVED
    information ``-d^2 log p / d eta^2``, which is nonnegative because the
    interval probability is log-concave in ``eta``; it is the Newton weight of
    the scoring step (an expected information is not defined for fixed
    intervals) [EQ-INV-OLX-004].

    Args:
        sd: Strictly positive standard deviations ``s_i``.
        lower: Lower limits (``None`` or ``-inf`` for no limit).
        upper: Upper limits (``None`` or ``inf`` for no limit).

    Raises:
        TypeError: For non-real limits.
        ValueError: If lengths disagree, a limit is ``nan`` or
            ``lower > upper`` somewhere.

    References:
        [EQ-INV-OLX-003], [EQ-INV-OLX-004], [REF-LANGE-LITTLE-TAYLOR-1989].

    """

    def __init__(
        self,
        sd: Sequence[float],
        lower: Sequence[float | None],
        upper: Sequence[float | None],
    ) -> None:
        """Validate scales and intervals.

        References:
            [EQ-INV-OLX-003].

        """
        self.sd = _positive_vector(sd, "sd")
        m = len(self.sd)
        self.lower = self._limits(lower, m, -math.inf, "lower")
        self.upper = self._limits(upper, m, math.inf, "upper")
        if any(lo > up for lo, up in zip(self.lower, self.upper, strict=True)):
            raise ValueError("lower limits must not exceed upper limits")
        if any(
            lo == up and not math.isfinite(lo)
            for lo, up in zip(self.lower, self.upper, strict=True)
        ):
            raise ValueError("an exact observation must be finite")

    @staticmethod
    def _limits(
        values: Sequence[float | None], size: int, default: float, name: str
    ) -> Vector:
        """Return limits with ``None`` replaced by the infinite default.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].

        """
        if len(values) != size:
            raise ValueError(f"{name} must have length {size}")
        result: list[float] = []
        for value in values:
            if value is None:
                result.append(default)
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must contain real numbers or None")
            number = float(value)
            if math.isnan(number):
                raise ValueError(f"{name} must not contain nan")
            result.append(number)
        return tuple(result)

    @property
    def size(self) -> int:
        """Return the observation dimension.

        References:
            [EQ-INV-OLX-003].

        """
        return len(self.sd)

    def representative(self) -> Vector:
        """Return a finite value inside each interval (midpoint or finite limit).

        Returns:
            A data vector ``y`` accepted by the other methods.

        References:
            [EQ-INV-OLX-003].

        """
        result: list[float] = []
        for lo, up in zip(self.lower, self.upper, strict=True):
            if math.isfinite(lo) and math.isfinite(up):
                result.append(0.5 * (lo + up))
            elif math.isfinite(lo):
                result.append(lo)
            elif math.isfinite(up):
                result.append(up)
            else:
                result.append(0.0)
        return tuple(result)

    def _checked(self, y: Sequence[float], eta: Sequence[float]) -> Vector:
        """Validate shapes and that every ``y_i`` lies in its interval.

        References:
            [EQ-INV-OLX-003].

        """
        y_vector, eta_vector = self._pair(y, eta)
        for v, lo, up in zip(y_vector, self.lower, self.upper, strict=True):
            if v < lo or v > up:
                raise ValueError("observation lies outside its interval")
        return eta_vector

    def log_likelihood(self, y: Sequence[float], eta: Sequence[float]) -> float:
        """Return the interval-censored log-likelihood.

        Args:
            y: Representatives inside the intervals.
            eta: Predictions.

        Returns:
            ``sum_i`` of log-density (exact) and log-interval-mass terms.

        Raises:
            ValueError: If a representative lies outside its interval.

        References:
            [EQ-INV-OLX-003].

        """
        eta_vector = self._checked(y, eta)
        terms: list[float] = []
        for e, s, lo, up in zip(
            eta_vector, self.sd, self.lower, self.upper, strict=True
        ):
            if lo == up:
                terms.append(Normal(e, s).log_pdf(lo))
            else:
                terms.append(
                    _log_interval_mass((lo - e) / s, (up - e) / s, (up - lo) / s)
                )
        return math.fsum(terms)

    def _derivatives(self, eta: Vector) -> list[tuple[float, float]]:
        """Return ``(score, observed information)`` per entry.

        References:
            [EQ-INV-OLX-004].

        """
        out: list[tuple[float, float]] = []
        for e, s, lo, up in zip(eta, self.sd, self.lower, self.upper, strict=True):
            if lo == up:
                out.append(((lo - e) / (s * s), 1.0 / (s * s)))
                continue
            a = (lo - e) / s if math.isfinite(lo) else -math.inf
            b = (up - e) / s if math.isfinite(up) else math.inf
            if a == -math.inf and b == math.inf:
                out.append((0.0, 0.0))
                continue
            width = (
                (up - lo) / s if math.isfinite(lo) and math.isfinite(up) else math.inf
            )
            finite = math.isfinite(width)
            if finite and width * (1.0 + max(abs(a), abs(b))) < _NARROW_INTERVAL:
                mid, half = a + 0.5 * width, 0.5 * width
                zs = [mid + half * x for x in _GL_NODES]
                ws = [
                    w * math.exp(_std_log_pdf(z))
                    for w, z in zip(_GL_WEIGHTS, zs, strict=True)
                ]
                total = math.fsum(ws)
                mean = math.fsum(w * z for w, z in zip(ws, zs, strict=True)) / total
                var = (
                    math.fsum(w * (z - mean) ** 2 for w, z in zip(ws, zs, strict=True))
                    / total
                )
                out.append((mean / s, (1.0 - var) / (s * s)))
                continue
            log_mass = _log_interval_mass(a, b)
            ra = math.exp(_std_log_pdf(a) - log_mass) if math.isfinite(a) else 0.0
            rb = math.exp(_std_log_pdf(b) - log_mass) if math.isfinite(b) else 0.0
            ta = a * ra if math.isfinite(a) else 0.0
            tb = b * rb if math.isfinite(b) else 0.0
            score = (ra - rb) / s
            out.append((score, score * score - (ta - tb) / (s * s)))
        return out

    def score(self, y: Sequence[float], eta: Sequence[float]) -> Vector:
        """Return ``(phi(a) - phi(b)) / (s (Phi(b) - Phi(a)))``.

        Exact entries give ``(l - eta) / s^2``; narrow intervals use the
        quadrature weighted mean of ``(x - eta) / s^2``.

        Args:
            y: Representatives inside the intervals.
            eta: Predictions.

        Returns:
            The score vector.

        References:
            [EQ-INV-OLX-004].

        """
        eta_vector = self._checked(y, eta)
        return tuple(d[0] for d in self._derivatives(eta_vector))

    def fisher_information(self, eta: Sequence[float]) -> Vector:
        """Return the observed information ``(m'/m)^2 - m''/m`` (``1/s^2`` if exact).

        With ``ra = phi(a)/mass`` and ``rb = phi(b)/mass`` it equals
        ``score^2 - (a ra - b rb)/s^2``; for a narrow interval
        ``(1 - Var_q(z))/s^2`` under the quadrature-weighted law ``q``.

        Args:
            eta: Predictions.

        Returns:
            Nonnegative diagonal Newton weights.

        References:
            [EQ-INV-OLX-004].

        """
        eta_vector = _real_vector(eta, "eta", size=self.size)
        return tuple(max(d[1], 0.0) for d in self._derivatives(eta_vector))

    def simulate(
        self, eta: Sequence[float], seed: int | RandomStream | None = None
    ) -> Vector:
        """Draw the latent value ``eta + s Z`` conditional on lying in its interval.

        The truncated normal is drawn by the inverse CDF (reflected into the
        lower tail so the tail probability keeps full relative accuracy).
        Exact entries return their value.

        Args:
            eta: Predictions.
            seed: Integer seed or stream.

        Returns:
            One latent draw per entry.

        Raises:
            ValueError: If an interval carries no representable probability mass.

        References:
            [EQ-INV-OLX-003].

        """
        eta_vector = _real_vector(eta, "eta", size=self.size)
        generator = resolve_stream(seed).generator()
        draws: list[float] = []
        for e, s, lo, up in zip(
            eta_vector, self.sd, self.lower, self.upper, strict=True
        ):
            if lo == up:
                draws.append(lo)
                continue
            a = (lo - e) / s if math.isfinite(lo) else -math.inf
            b = (up - e) / s if math.isfinite(up) else math.inf
            flip = a > 0.0
            if flip:
                a, b = -b, -a
            p_lo = _STD.cdf(a) if math.isfinite(a) else 0.0
            p_hi = _STD.cdf(b) if math.isfinite(b) else 1.0
            if not p_hi > p_lo:
                raise ValueError("interval has no representable probability mass")
            p = p_lo + generator.random() * (p_hi - p_lo)
            p = min(max(p, p_lo), p_hi)
            z = _STD.quantile(p) if 0.0 < p < 1.0 else (a if p <= 0.0 else b)
            z = min(max(z, a), b)
            draws.append(e + s * (-z if flip else z))
        return tuple(draws)

    def irls_system(
        self, y: Sequence[float], eta: Sequence[float]
    ) -> tuple[LinearOperator, Vector]:
        """Return Newton weights and working response.

        Args:
            y: Representatives inside the intervals.
            eta: Predictions.

        Returns:
            ``(diag(sqrt(I)), eta + score / I)`` with the observed information.

        References:
            [EQ-INV-OLX-004].

        """
        eta_vector = _real_vector(eta, "eta", size=self.size)
        weights, response = _scoring_response(
            eta_vector, self.score(y, eta), self.fisher_information(eta)
        )
        return _diagonal_whitening(weights), response

    def restrict(self, indices: Sequence[int]) -> IntervalCensoredGaussianLaw:
        """Return the law of the selected entries.

        Args:
            indices: Strictly increasing indices.

        Returns:
            An :class:`IntervalCensoredGaussianLaw`.

        References:
            [EQ-INV-OLX-003].

        """
        chosen = _check_indices(indices, self.size)
        return IntervalCensoredGaussianLaw(
            [self.sd[i] for i in chosen],
            [self.lower[i] for i in chosen],
            [self.upper[i] for i in chosen],
        )


# ---------------------------------------------------------------------------
# 3. Gaussian-mixture (contaminated normal) outliers
# ---------------------------------------------------------------------------


class GaussianMixtureOutlierLaw(ObservationLaw):
    """Contaminated normal errors ``y_i = eta_i + e_i``.

    ``e_i ~ (1 - eps) N(0, s_i^2) + eps N(0, (kappa s_i)^2)`` with outlier
    probability ``0 < eps < 1`` and variance inflation ``kappa > 1``. The
    log-likelihood is a two-term log-sum-exp; the posterior probability that
    entry ``i`` is an outlier is its *responsibility*
    ``rho_i = eps N_out / ((1 - eps) N_in + eps N_out)``
    [EQ-INV-OLX-005]. The score is ``r (1 - rho + rho / kappa^2) / s^2`` and
    the EM (majorization) weight of the scoring step is the same factor
    [EQ-INV-OLX-006]; the likelihood is multimodal, so the IRLS limit is a
    local maximum and a robust start (e.g. a median fit) is advisable.

    Args:
        scale: Strictly positive inlier standard deviations ``s_i``.
        epsilon: Outlier probability in ``(0, 1)``.
        kappa: Outlier scale ratio ``> 1``.

    Raises:
        TypeError: For non-real parameters.
        ValueError: For ``epsilon`` outside ``(0, 1)`` or ``kappa <= 1``.

    References:
        [EQ-INV-OLX-005], [EQ-INV-OLX-006], [REF-LANGE-LITTLE-TAYLOR-1989],
        [REF-MCLACHLAN-PEEL-2000].

    """

    def __init__(self, scale: Sequence[float], epsilon: float, kappa: float) -> None:
        """Validate the mixture parameters.

        References:
            [EQ-INV-OLX-005].

        """
        self.scale = _positive_vector(scale, "scale")
        eps = _real(epsilon, "epsilon")
        if not 0.0 < eps < 1.0:
            raise ValueError("epsilon must lie strictly between 0 and 1")
        ratio = _real(kappa, "kappa")
        if ratio <= 1.0:
            raise ValueError("kappa must exceed 1")
        self.epsilon = eps
        self.kappa = ratio
        self._information_factor: float | None = None

    @property
    def size(self) -> int:
        """Return the observation dimension.

        References:
            [EQ-INV-OLX-005].

        """
        return len(self.scale)

    def _entry(self, r: float, s: float) -> tuple[float, float]:
        """Return ``(log p, rho)`` for one residual.

        References:
            [EQ-INV-OLX-005].

        """
        log_in = math.log1p(-self.epsilon) + _std_log_pdf(r / s) - math.log(s)
        log_out = (
            math.log(self.epsilon)
            + _std_log_pdf(r / (self.kappa * s))
            - math.log(self.kappa * s)
        )
        total = _logsumexp([log_in, log_out])
        return total, math.exp(log_out - total)

    def log_likelihood(self, y: Sequence[float], eta: Sequence[float]) -> float:
        """Return ``sum_i log`` of the two-component mixture density.

        Args:
            y: Observations.
            eta: Predictions.

        Returns:
            The normalized log-likelihood.

        References:
            [EQ-INV-OLX-005].

        """
        y_vector, eta_vector = self._pair(y, eta)
        return math.fsum(
            self._entry(v - e, s)[0]
            for v, e, s in zip(y_vector, eta_vector, self.scale, strict=True)
        )

    def responsibilities(self, y: Sequence[float], eta: Sequence[float]) -> Vector:
        """Return the posterior outlier probabilities ``rho_i``.

        Args:
            y: Observations.
            eta: Predictions.

        Returns:
            Values in ``[0, 1]``, near one for gross outliers.

        References:
            [EQ-INV-OLX-005].

        """
        y_vector, eta_vector = self._pair(y, eta)
        return tuple(
            self._entry(v - e, s)[1]
            for v, e, s in zip(y_vector, eta_vector, self.scale, strict=True)
        )

    def irls_weights(self, y: Sequence[float], eta: Sequence[float]) -> Vector:
        """Return the EM weights ``(1 - rho + rho / kappa^2) / s^2``.

        Args:
            y: Observations.
            eta: Predictions.

        Returns:
            Positive weights; they equal ``score / r``.

        References:
            [EQ-INV-OLX-006].

        """
        rho = self.responsibilities(y, eta)
        k2 = self.kappa * self.kappa
        return tuple(
            (1.0 - p + p / k2) / (s * s) for p, s in zip(rho, self.scale, strict=True)
        )

    def score(self, y: Sequence[float], eta: Sequence[float]) -> Vector:
        """Return ``r (1 - rho + rho / kappa^2) / s^2``.

        Args:
            y: Observations.
            eta: Predictions.

        Returns:
            The score vector.

        References:
            [EQ-INV-OLX-006].

        """
        y_vector, eta_vector = self._pair(y, eta)
        weights = self.irls_weights(y, eta)
        return tuple(
            (v - e) * w for v, e, w in zip(y_vector, eta_vector, weights, strict=True)
        )

    def information_factor(self) -> float:
        """Return ``J(eps, kappa) = int g'(u)^2 / g(u) du`` for the unit-scale density.

        ``g(u) = (1 - eps) phi(u) + eps phi(u / kappa) / kappa``. The integral
        has no closed form; it is computed once by composite Simpson rules
        (step ``0.02`` on ``[0, 10]`` and ``0.02 kappa`` on
        ``[10, 12 kappa]``, integrand symmetric) and cached. Because the
        integrand is smooth and decays like ``exp(-u^2 / (2 kappa^2))`` the
        truncation error is below ``1e-30``.

        Returns:
            The unit-scale Fisher information ``J`` (the information of entry
            ``i`` is ``J / s_i^2``).

        References:
            [EQ-INV-OLX-006].

        """
        if self._information_factor is not None:
            return self._information_factor
        kappa = self.kappa

        def integrand(u: float) -> float:
            """Return ``g(u) (u (1 - rho + rho / kappa^2))^2`` at ``u >= 0``.

            References:
                [EQ-INV-OLX-006].

            """
            log_g, rho = self._entry(u, 1.0)
            factor = u * (1.0 - rho + rho / (kappa * kappa))
            return math.exp(log_g) * factor * factor

        def simpson(lo: float, hi: float, panels: int) -> float:
            """Return the composite Simpson rule with an even panel count.

            References:
                [REF-DAVIS-RABINOWITZ-1984].

            """
            n = panels + panels % 2
            h = (hi - lo) / n
            total = integrand(lo) + integrand(hi)
            for i in range(1, n):
                total += (4.0 if i % 2 else 2.0) * integrand(lo + i * h)
            return total * h / 3.0

        near = simpson(0.0, 10.0, 500)
        far_panels = max(2, math.ceil((12.0 * kappa - 10.0) / (0.02 * kappa)))
        far = simpson(10.0, 12.0 * kappa, far_panels)
        self._information_factor = 2.0 * (near + far)
        return self._information_factor

    def fisher_information(self, eta: Sequence[float]) -> Vector:
        """Return ``J(eps, kappa) / s_i^2``.

        Args:
            eta: Predictions (only the length is used).

        Returns:
            The diagonal information.

        References:
            [EQ-INV-OLX-006].

        """
        _real_vector(eta, "eta", size=self.size)
        factor = self.information_factor()
        return tuple(factor / (s * s) for s in self.scale)

    def simulate(
        self, eta: Sequence[float], seed: int | RandomStream | None = None
    ) -> Vector:
        """Draw ``eta + s Z`` with ``Z`` scaled by ``kappa`` with probability ``eps``.

        Args:
            eta: Predictions.
            seed: Integer seed or stream.

        Returns:
            One simulated data vector.

        References:
            [EQ-INV-OLX-005].

        """
        eta_vector = _real_vector(eta, "eta", size=self.size)
        generator = resolve_stream(seed).generator()
        draws: list[float] = []
        for e, s in zip(eta_vector, self.scale, strict=True):
            width = s * (self.kappa if generator.random() < self.epsilon else 1.0)
            draws.append(e + width * generator.gauss(0.0, 1.0))
        return tuple(draws)

    def irls_system(
        self, y: Sequence[float], eta: Sequence[float]
    ) -> tuple[LinearOperator, Vector]:
        """Return the EM weights of [EQ-INV-OLX-006] and ``z = y``.

        The weighted least-squares step minimizes a quadratic majorizer of
        the negative log-likelihood (Gaussian scale mixture), so it never
        increases the objective.

        Args:
            y: Observations.
            eta: Current predictions.

        Returns:
            ``(diag(sqrt(w)), y)``.

        References:
            [EQ-INV-OLX-006], [REF-LANGE-LITTLE-TAYLOR-1989].

        """
        y_vector, _ = self._pair(y, eta)
        return _diagonal_whitening(self.irls_weights(y, eta)), y_vector

    def restrict(self, indices: Sequence[int]) -> GaussianMixtureOutlierLaw:
        """Return the law of the selected observations.

        Args:
            indices: Strictly increasing indices.

        Returns:
            A :class:`GaussianMixtureOutlierLaw`.

        References:
            [EQ-INV-OLX-005].

        """
        chosen = _check_indices(indices, self.size)
        return GaussianMixtureOutlierLaw(
            [self.scale[i] for i in chosen], self.epsilon, self.kappa
        )


# ---------------------------------------------------------------------------
# 4. Huber
# ---------------------------------------------------------------------------


def _huber_rho(u: float, k: float) -> float:
    """Return the Huber loss ``u^2/2`` (``|u| <= k``) or ``k|u| - k^2/2``.

    References:
        [EQ-INV-OLX-007], [REF-HUBER-1964].

    """
    return 0.5 * u * u if abs(u) <= k else k * abs(u) - 0.5 * k * k


class HuberLaw(ObservationLaw):
    """Huber's loss as an M-estimation criterion and as a normalized density.

    The Huber loss ``rho_k(u) = u^2/2`` for ``|u| <= k`` and ``k|u| - k^2/2``
    otherwise is a *loss*: minimizing ``sum rho_k(r_i / s_i)`` is an
    M-estimator whose score is the clipped residual ``psi_k(u) =
    max(-k, min(k, u))``. A loss defines a probabilistic model only after
    normalization. This class uses the density ``exp(-rho_k(u)) / Z_k`` with the
    exact constant ``Z_k = sqrt(2 pi) (2 Phi(k) - 1) + 2 exp(-k^2/2) / k``
    (Gaussian centre, exponential tails: Huber's least-favourable law), so
    that :meth:`log_likelihood` is a legitimate normalized log-density with
    ``y = eta + s u`` [EQ-INV-OLX-007]. Posterior and evidence statements are
    valid under this declared density only; if you use the loss as a
    frequentist criterion, use :meth:`pseudo_log_likelihood` (no constant) and
    :func:`huber_sandwich_covariance` instead of the model-based covariance
    [EQ-INV-OLX-008].

    Args:
        scale: Strictly positive scales ``s_i`` (not standard deviations: the
            variance of the density is not ``s^2``).
        threshold: The Huber constant ``k > 0``.

    Raises:
        TypeError: For non-real parameters.
        ValueError: For non-positive scale or threshold.

    References:
        [EQ-INV-OLX-007], [EQ-INV-OLX-008], [REF-HUBER-1964], [REF-HUBER-1981].

    """

    def __init__(self, scale: Sequence[float], threshold: float = 1.345) -> None:
        """Validate scale and threshold.

        References:
            [EQ-INV-OLX-007].

        """
        self.scale = _positive_vector(scale, "scale")
        self.threshold = _positive(threshold, "threshold")

    @property
    def size(self) -> int:
        """Return the observation dimension.

        References:
            [EQ-INV-OLX-007].

        """
        return len(self.scale)

    def normalizing_constant(self) -> float:
        """Return ``Z_k = int exp(-rho_k(u)) du`` in closed form.

        Returns:
            ``sqrt(2 pi) erf(k / sqrt 2) + 2 exp(-k^2 / 2) / k``.

        References:
            [EQ-INV-OLX-007].

        """
        k = self.threshold
        return (
            _SQRT_2PI * math.erf(k / math.sqrt(2.0)) + 2.0 * math.exp(-0.5 * k * k) / k
        )

    def central_probability(self) -> float:
        """Return ``P(|u| <= k)`` under the density, also the unit information.

        Returns:
            ``sqrt(2 pi) erf(k / sqrt 2) / Z_k``.

        References:
            [EQ-INV-OLX-007].

        """
        k = self.threshold
        return _SQRT_2PI * math.erf(k / math.sqrt(2.0)) / self.normalizing_constant()

    def log_likelihood(self, y: Sequence[float], eta: Sequence[float]) -> float:
        """Return ``-sum_i [rho_k(r_i / s_i) + log(s_i Z_k)]``.

        Args:
            y: Observations.
            eta: Predictions.

        Returns:
            The normalized log-likelihood of the Huber density.

        References:
            [EQ-INV-OLX-007].

        """
        y_vector, eta_vector = self._pair(y, eta)
        log_z = math.log(self.normalizing_constant())
        return math.fsum(
            -_huber_rho((v - e) / s, self.threshold) - math.log(s) - log_z
            for v, e, s in zip(y_vector, eta_vector, self.scale, strict=True)
        )

    def pseudo_log_likelihood(self, y: Sequence[float], eta: Sequence[float]) -> float:
        """Return ``-sum_i rho_k(r_i / s_i)``: the M-estimation criterion.

        This omits ``log(s Z_k)``. It is NOT a normalized log-density and must
        not be used for evidence, model comparison or scale estimation.

        Args:
            y: Observations.
            eta: Predictions.

        Returns:
            The negative Huber loss.

        References:
            [EQ-INV-OLX-008].

        """
        y_vector, eta_vector = self._pair(y, eta)
        return -math.fsum(
            _huber_rho((v - e) / s, self.threshold)
            for v, e, s in zip(y_vector, eta_vector, self.scale, strict=True)
        )

    def score(self, y: Sequence[float], eta: Sequence[float]) -> Vector:
        """Return ``psi_k(r / s) / s`` with the clipped residual ``psi_k``.

        Args:
            y: Observations.
            eta: Predictions.

        Returns:
            The score vector.

        References:
            [EQ-INV-OLX-008].

        """
        y_vector, eta_vector = self._pair(y, eta)
        k = self.threshold
        return tuple(
            max(-k, min(k, (v - e) / s)) / s
            for v, e, s in zip(y_vector, eta_vector, self.scale, strict=True)
        )

    def curvature(self, y: Sequence[float], eta: Sequence[float]) -> Vector:
        """Return ``-d^2 log p / d eta^2 = 1{|r| < k s} / s^2`` (observed).

        Args:
            y: Observations.
            eta: Predictions.

        Returns:
            The observed curvature, the ``bread`` of the sandwich covariance.

        References:
            [EQ-INV-OLX-008].

        """
        y_vector, eta_vector = self._pair(y, eta)
        return tuple(
            (1.0 if abs(v - e) < self.threshold * s else 0.0) / (s * s)
            for v, e, s in zip(y_vector, eta_vector, self.scale, strict=True)
        )

    def fisher_information(self, eta: Sequence[float]) -> Vector:
        """Return ``P(|u| <= k) / s^2`` under the Huber density.

        Args:
            eta: Predictions (only the length is used).

        Returns:
            The diagonal information (``E psi^2 = E psi'`` for this density).

        References:
            [EQ-INV-OLX-008].

        """
        _real_vector(eta, "eta", size=self.size)
        mass = self.central_probability()
        return tuple(mass / (s * s) for s in self.scale)

    def simulate(
        self, eta: Sequence[float], seed: int | RandomStream | None = None
    ) -> Vector:
        """Draw from the Huber density (truncated normal centre, exponential tails).

        Args:
            eta: Predictions.
            seed: Integer seed or stream.

        Returns:
            One simulated data vector.

        References:
            [EQ-INV-OLX-007].

        """
        eta_vector = _real_vector(eta, "eta", size=self.size)
        generator = resolve_stream(seed).generator()
        k = self.threshold
        mass = self.central_probability()
        mass_cdf = math.erf(k / math.sqrt(2.0))
        draws: list[float] = []
        for e, s in zip(eta_vector, self.scale, strict=True):
            if generator.random() < mass:
                p = 0.5 + (generator.random() - 0.5) * mass_cdf
                u = _STD.quantile(min(max(p, 1e-300), 1.0 - 1e-16))
                u = max(-k, min(k, u))
            else:
                u = (k + generator.expovariate(k)) * (
                    1.0 if generator.random() < 0.5 else -1.0
                )
            draws.append(e + s * u)
        return tuple(draws)

    def irls_system(
        self, y: Sequence[float], eta: Sequence[float]
    ) -> tuple[LinearOperator, Vector]:
        """Return the half-quadratic weights ``min(1, k s / |r|) / s^2`` and ``z = y``.

        Args:
            y: Observations.
            eta: Current predictions.

        Returns:
            ``(diag(sqrt(w)), y)``. The step minimizes a quadratic majorizer
            of the Huber loss, so the criterion never increases.

        References:
            [EQ-INV-OLX-008], [REF-HUBER-1981].

        """
        y_vector, eta_vector = self._pair(y, eta)
        k = self.threshold
        weights = tuple(
            min(1.0, k * s / abs(v - e) if v != e else 1.0) / (s * s)
            for v, e, s in zip(y_vector, eta_vector, self.scale, strict=True)
        )
        return _diagonal_whitening(weights), y_vector

    def restrict(self, indices: Sequence[int]) -> HuberLaw:
        """Return the law of the selected observations.

        Args:
            indices: Strictly increasing indices.

        Returns:
            A :class:`HuberLaw`.

        References:
            [EQ-INV-OLX-007].

        """
        chosen = _check_indices(indices, self.size)
        return HuberLaw([self.scale[i] for i in chosen], self.threshold)


def huber_sandwich_covariance(
    operator: LinearOperator | Sequence[Sequence[float]],
    data: Sequence[float],
    law: HuberLaw,
    estimate: Sequence[float],
) -> Matrix:
    """Return the robust sandwich covariance ``B^-1 M B^-1`` of a Huber fit.

    ``B = A' diag(c) A`` with the observed curvature ``c_i = 1{|r_i| < k s_i}
    / s_i^2`` and ``M = A' diag(psi_i^2) A`` with the scores ``psi_i``. The
    formula treats ``rho_k`` as an M-estimation criterion for independent
    observations and is justified asymptotically for a large number of
    observations with ``B`` of full rank; it does NOT use the model-based
    ``Fisher^-1`` (which would be right only if the Huber density were the
    true error law) [EQ-INV-OLX-008].

    Args:
        operator: Predictor matrix or operator ``A`` (``m x n``).
        data: Observations.
        law: The :class:`HuberLaw`.
        estimate: The fitted coefficient vector.

    Returns:
        The ``n x n`` covariance matrix.

    Raises:
        TypeError: For wrong argument types.
        ValueError: If dimensions disagree or ``B`` is not positive definite
            (too few unclipped observations).

    References:
        [EQ-INV-OLX-008], [REF-WHITE-1982], [REF-HUBER-1981].

    """
    if not isinstance(law, HuberLaw):
        raise TypeError("law must be a HuberLaw")
    a = require_operator(operator, name="operator")
    m, n = a.shape
    if law.size != m:
        raise ValueError("law dimension must equal the number of observations")
    x = _real_vector(estimate, "estimate", size=n)
    dense = materialize(a)
    eta = a.matvec(x)
    curvature = law.curvature(data, eta)
    scores = law.score(data, eta)
    bread = tuple(
        tuple(
            math.fsum(dense[r][i] * curvature[r] * dense[r][j] for r in range(m))
            for j in range(n)
        )
        for i in range(n)
    )
    meat = tuple(
        tuple(
            math.fsum(dense[r][i] * scores[r] ** 2 * dense[r][j] for r in range(m))
            for j in range(n)
        )
        for i in range(n)
    )
    inverse = inverse_spd(bread)
    return _matmul(_matmul(inverse, meat), inverse)


# ---------------------------------------------------------------------------
# 5. Multivariate Student-t
# ---------------------------------------------------------------------------


class MultivariateStudentTLaw(ObservationLaw):
    """Joint Student-t errors ``y ~ t_nu(eta, Sigma)`` with a full scatter matrix.

    ``log p = log Gamma((nu+m)/2) - log Gamma(nu/2) - (m/2) log(nu pi)
    - 1/2 log det Sigma - (nu+m)/2 log(1 + delta/nu)`` with
    ``delta = r' Sigma^-1 r`` [EQ-INV-OLX-009]. ``Sigma`` is the SCATTER
    matrix, not the covariance: the covariance is ``nu/(nu-2) Sigma`` for
    ``nu > 2`` and does not exist for ``nu <= 2`` (:meth:`covariance_matrix`
    refuses). The components share one mixing variable and are therefore
    dependent even when ``Sigma`` is diagonal, unlike
    :class:`~probstat.inverse.observation_laws.StudentTLaw`. Any marginal is
    again a multivariate t with the same ``nu`` and the sub-scatter matrix,
    so :class:`~probstat.inverse.observation_laws.MissingMask` applies
    [EQ-INV-OLX-010].

    Args:
        scatter: Symmetric positive-definite ``m x m`` scatter matrix.
        degrees_of_freedom: ``nu > 0`` (real values allowed).

    Raises:
        TypeError: For wrong types.
        ValueError: For a non-symmetric or non-positive-definite scatter
            matrix or ``nu <= 0``.

    References:
        [EQ-INV-OLX-009], [EQ-INV-OLX-010], [REF-KOTZ-NADARAJAH-2004],
        [REF-GELMAN-ETAL-2013].

    """

    def __init__(
        self, scatter: Sequence[Sequence[float]], degrees_of_freedom: float
    ) -> None:
        """Validate and factor the scatter matrix.

        References:
            [EQ-INV-OLX-009].

        """
        rows = tuple(tuple(row) for row in scatter)
        size = len(rows)
        if size == 0:
            raise ValueError("scatter must be non-empty")
        self.scatter = _symmetric_matrix(rows, size, "scatter")
        self.degrees_of_freedom = _positive(degrees_of_freedom, "degrees_of_freedom")
        self._factor = DenseCovariance(self.scatter)
        self._lower = cholesky(self.scatter)
        precision = _inverse_columns(self._factor, size)
        self._precision_diagonal = tuple(precision[i][i] for i in range(size))

    @property
    def size(self) -> int:
        """Return the observation dimension ``m``.

        References:
            [EQ-INV-OLX-009].

        """
        return len(self.scatter)

    def covariance_matrix(self) -> Matrix:
        """Return ``nu / (nu - 2) Sigma`` for ``nu > 2``.

        Returns:
            The covariance matrix of the law.

        Raises:
            ValueError: If ``nu <= 2`` (the covariance does not exist).

        References:
            [EQ-INV-OLX-009].

        """
        nu = self.degrees_of_freedom
        if nu <= 2.0:
            raise ValueError("the covariance exists only for degrees_of_freedom > 2")
        factor = nu / (nu - 2.0)
        return tuple(tuple(factor * v for v in row) for row in self.scatter)

    def _delta(self, y: Vector, eta: Vector) -> float:
        """Return the squared Mahalanobis distance ``r' Sigma^-1 r``.

        References:
            [EQ-INV-OLX-009].

        """
        return float(
            self._factor.quadratic_form([a - b for a, b in zip(y, eta, strict=True)])
        )

    def log_likelihood(self, y: Sequence[float], eta: Sequence[float]) -> float:
        """Return the normalized multivariate-t log-density of ``y - eta``.

        Args:
            y: Observations (length ``m``).
            eta: Predictions (length ``m``).

        Returns:
            The log-likelihood (the gamma ratio is evaluated stably for large
            ``nu``).

        References:
            [EQ-INV-OLX-009].

        """
        y_vector, eta_vector = self._pair(y, eta)
        nu, m = self.degrees_of_freedom, self.size
        delta = self._delta(y_vector, eta_vector)
        return math.fsum(
            (
                log_gamma_ratio(0.5 * nu, 0.5 * m),
                -0.5 * m * math.log(nu * math.pi),
                -0.5 * self._factor.log_det(),
                -0.5 * (nu + m) * math.log1p(delta / nu),
            )
        )

    def _weight(self, y: Vector, eta: Vector) -> float:
        """Return the EM weight ``(nu + m) / (nu + delta)``.

        References:
            [EQ-INV-OLX-010].

        """
        return (self.degrees_of_freedom + self.size) / (
            self.degrees_of_freedom + self._delta(y, eta)
        )

    def score(self, y: Sequence[float], eta: Sequence[float]) -> Vector:
        """Return ``(nu + m) / (nu + delta) Sigma^-1 r``.

        Args:
            y: Observations.
            eta: Predictions.

        Returns:
            The score vector.

        References:
            [EQ-INV-OLX-010].

        """
        y_vector, eta_vector = self._pair(y, eta)
        weight = self._weight(y_vector, eta_vector)
        residual = [a - b for a, b in zip(y_vector, eta_vector, strict=True)]
        return tuple(weight * float(v) for v in self._factor.solve(residual))

    def fisher_information(self, eta: Sequence[float]) -> Vector:
        """Return ``(nu + m)/(nu + m + 2) diag(Sigma^-1)``.

        The full information is ``(nu + m)/(nu + m + 2) Sigma^-1``.

        Args:
            eta: Predictions (only the length is used).

        Returns:
            The diagonal of the information matrix.

        References:
            [EQ-INV-OLX-010].

        """
        _real_vector(eta, "eta", size=self.size)
        nu, m = self.degrees_of_freedom, self.size
        factor = (nu + m) / (nu + m + 2.0)
        return tuple(factor * d for d in self._precision_diagonal)

    def simulate(
        self, eta: Sequence[float], seed: int | RandomStream | None = None
    ) -> Vector:
        """Draw ``eta + L z / sqrt(W / nu)`` with ``W ~ chi^2(nu)``.

        Args:
            eta: Predictions.
            seed: Integer seed or stream.

        Returns:
            One simulated data vector.

        References:
            [EQ-INV-OLX-009].

        """
        eta_vector = _real_vector(eta, "eta", size=self.size)
        generator = resolve_stream(seed).generator()
        z = [generator.gauss(0.0, 1.0) for _ in eta_vector]
        chi2 = generator.gammavariate(0.5 * self.degrees_of_freedom, 2.0)
        factor = math.sqrt(self.degrees_of_freedom / chi2)
        return tuple(
            e + factor * math.fsum(self._lower[i][j] * z[j] for j in range(i + 1))
            for i, e in enumerate(eta_vector)
        )

    def irls_system(
        self, y: Sequence[float], eta: Sequence[float]
    ) -> tuple[LinearOperator, Vector]:
        """Return ``W = sqrt((nu+m)/(nu+delta)) C^-1`` and ``z = y``.

        The weighted least-squares step minimizes a quadratic majorizer of the
        negative log-likelihood (scale-mixture EM), so it never increases the
        objective.

        Args:
            y: Observations.
            eta: Current predictions.

        Returns:
            The whitening operator and ``y``.

        References:
            [EQ-INV-OLX-010], [REF-LANGE-LITTLE-TAYLOR-1989].

        """
        y_vector, eta_vector = self._pair(y, eta)
        weight = self._weight(y_vector, eta_vector)
        return (
            ScaledOperator(self._factor.whitening_operator(), math.sqrt(weight)),
            y_vector,
        )

    def restrict(self, indices: Sequence[int]) -> MultivariateStudentTLaw:
        """Return the marginal ``t_nu`` law with the sub-scatter matrix.

        Args:
            indices: Strictly increasing indices.

        Returns:
            A :class:`MultivariateStudentTLaw` of dimension ``len(indices)``.

        References:
            [EQ-INV-OLX-010].

        """
        chosen = _check_indices(indices, self.size)
        sub = [[self.scatter[i][j] for j in chosen] for i in chosen]
        return MultivariateStudentTLaw(sub, self.degrees_of_freedom)


# ---------------------------------------------------------------------------
# 6. State-dependent noise
# ---------------------------------------------------------------------------


class HeteroscedasticGaussianLaw(ObservationLaw):
    """Independent Gaussian errors with variance ``v(eta_i)`` PREDICTED from the mean.

    ``y_i ~ N(eta_i, v(eta_i))`` where ``v`` is a positive function of the
    prediction (for example ``a + b eta^2``). This differs from known fixed
    noise: because ``v`` depends on ``eta``, the score contains the
    covariance-derivative terms,
    ``d log p / d eta = r/v + (v'/2)(r^2/v^2 - 1/v)``, and the information is
    ``1/v + v'^2/(2 v^2)`` [EQ-INV-OLX-011]. The log-determinant ``log v``
    is part of the likelihood, which is why dropping it (treating ``v`` as
    fixed) gives the wrong objective and a biased fit.

    Args:
        variance: Callable ``eta -> (v, dv/deta)`` with ``v > 0``.
        size: Optional observation dimension (``None`` accepts any length).

    Raises:
        TypeError: If ``variance`` is not callable.
        ValueError: If a returned variance is not positive and finite.

    References:
        [EQ-INV-OLX-011], [REF-TARANTOLA-2005].

    """

    def __init__(
        self,
        variance: Callable[[float], tuple[float, float]],
        size: int | None = None,
    ) -> None:
        """Store the variance function.

        References:
            [EQ-INV-OLX-011].

        """
        if not callable(variance):
            raise TypeError("variance must be callable")
        if size is not None and (
            isinstance(size, bool) or not isinstance(size, int) or size < 1
        ):
            raise ValueError("size must be a positive integer or None")
        self.variance = variance
        self._size = size

    @classmethod
    def quadratic(
        cls, base: float, slope: float, size: int | None = None
    ) -> HeteroscedasticGaussianLaw:
        """Return the law with ``v(eta) = base + slope eta^2``.

        Args:
            base: ``a > 0``.
            slope: ``b >= 0``.
            size: Optional dimension.

        Returns:
            The law with derivative ``2 b eta``.

        Raises:
            ValueError: If ``base <= 0`` or ``slope < 0``.

        References:
            [EQ-INV-OLX-011].

        """
        a = _positive(base, "base")
        b = _real(slope, "slope")
        if b < 0.0:
            raise ValueError("slope must be nonnegative")
        return cls(lambda e: (a + b * e * e, 2.0 * b * e), size)

    @property
    def size(self) -> int | None:
        """Return the observation dimension, or ``None`` if any length is valid.

        References:
            [EQ-INV-OLX-011].

        """
        return self._size

    def _moments(self, eta: Vector) -> list[tuple[float, float]]:
        """Return validated ``(v, v')`` per entry.

        References:
            [EQ-INV-OLX-011].

        """
        out: list[tuple[float, float]] = []
        for e in eta:
            v, dv = self.variance(e)
            if not (math.isfinite(v) and math.isfinite(dv)) or v <= 0.0:
                raise ValueError("variance function must return finite v > 0 and v'")
            out.append((v, dv))
        return out

    def log_likelihood(self, y: Sequence[float], eta: Sequence[float]) -> float:
        """Return ``-1/2 sum_i [r_i^2 / v_i + log v_i + log 2 pi]``.

        Args:
            y: Observations.
            eta: Predictions.

        Returns:
            The normalized log-likelihood including ``log v``.

        References:
            [EQ-INV-OLX-011].

        """
        y_vector, eta_vector = self._pair(y, eta)
        moments = self._moments(eta_vector)
        return math.fsum(
            -0.5 * ((v - e) ** 2 / var + math.log(var) + _LOG_2PI)
            for v, e, (var, _) in zip(y_vector, eta_vector, moments, strict=True)
        )

    def score(self, y: Sequence[float], eta: Sequence[float]) -> Vector:
        """Return ``r/v + (v'/2)(r^2/v^2 - 1/v)``.

        Args:
            y: Observations.
            eta: Predictions.

        Returns:
            The exact gradient with respect to ``eta`` (mean and variance
            paths).

        References:
            [EQ-INV-OLX-011].

        """
        y_vector, eta_vector = self._pair(y, eta)
        moments = self._moments(eta_vector)
        return tuple(
            (v - e) / var + 0.5 * dv * ((v - e) ** 2 / (var * var) - 1.0 / var)
            for v, e, (var, dv) in zip(y_vector, eta_vector, moments, strict=True)
        )

    def fisher_information(self, eta: Sequence[float]) -> Vector:
        """Return ``1/v + v'^2/(2 v^2)``.

        Args:
            eta: Predictions.

        Returns:
            The expected information with respect to ``eta``.

        References:
            [EQ-INV-OLX-011].

        """
        eta_vector = _real_vector(eta, "eta", size=self.size)
        return tuple(
            1.0 / var + 0.5 * (dv / var) ** 2 for var, dv in self._moments(eta_vector)
        )

    def simulate(
        self, eta: Sequence[float], seed: int | RandomStream | None = None
    ) -> Vector:
        """Draw ``eta + sqrt(v(eta)) Z``.

        Args:
            eta: Predictions.
            seed: Integer seed or stream.

        Returns:
            One simulated data vector.

        References:
            [EQ-INV-OLX-011].

        """
        eta_vector = _real_vector(eta, "eta", size=self.size)
        generator = resolve_stream(seed).generator()
        return tuple(
            e + math.sqrt(var) * generator.gauss(0.0, 1.0)
            for e, (var, _) in zip(eta_vector, self._moments(eta_vector), strict=True)
        )

    def irls_system(
        self, y: Sequence[float], eta: Sequence[float]
    ) -> tuple[LinearOperator, Vector]:
        """Return Fisher-scoring weights and working response.

        Args:
            y: Observations.
            eta: Predictions.

        Returns:
            ``(diag(sqrt(I)), eta + score / I)``.

        References:
            [EQ-INV-OLX-011].

        """
        eta_vector = _real_vector(eta, "eta", size=self.size)
        weights, response = _scoring_response(
            eta_vector, self.score(y, eta), self.fisher_information(eta)
        )
        return _diagonal_whitening(weights), response

    def restrict(self, indices: Sequence[int]) -> HeteroscedasticGaussianLaw:
        """Return the law of the selected entries (it is elementwise).

        Args:
            indices: Strictly increasing indices.

        Returns:
            A law with the same variance function.

        References:
            [EQ-INV-OLX-011].

        """
        chosen = _check_indices(indices, self.size)
        return HeteroscedasticGaussianLaw(
            self.variance, None if self.size is None else len(chosen)
        )


class StateDependentCovarianceLaw(ObservationLaw):
    """Gaussian errors ``y ~ N(eta, Gamma(eta))`` with a state-dependent covariance.

    The user supplies ``covariance_function(eta) -> (Gamma, [dGamma/d eta_j])``:
    the symmetric positive-definite matrix and the list of its ``m`` partial
    derivatives. With ``r = y - eta`` and ``a = Gamma^-1 r`` the exact gradient
    is ``d log p / d eta_j = a_j - 1/2 tr(Gamma^-1 Gamma_j)
    + 1/2 a' Gamma_j a``: the mean path plus the log-determinant and
    quadratic-form covariance paths [EQ-INV-OLX-011]. The expected information
    is ``F_jk = (Gamma^-1)_jk + 1/2 tr(Gamma^-1 Gamma_j Gamma^-1 Gamma_k)``
    [EQ-INV-OLX-012]; :meth:`irls_system` uses the full matrix ``F``
    (Cholesky), so the scoring step is exact for correlated errors. The cost
    is ``O(m^4)``; this class is for small ``m``. For a nonlinear predictor
    ``eta = G(x)`` the gradient in ``x`` is ``J' score`` with the Jacobian
    ``J`` of ``G``.

    Marginalizing (``restrict``) is NOT supported: the marginal covariance
    would depend on the dropped entries of ``eta``, so
    :class:`~probstat.inverse.observation_laws.MissingMask` cannot wrap this
    law.

    Args:
        dimension: Observation dimension ``m``.
        covariance_function: Callable returning ``(Gamma, derivatives)``.

    Raises:
        TypeError: If the function is not callable.
        ValueError: For a non-positive dimension; later, for a non-symmetric,
            non-positive-definite or wrongly shaped returned covariance.

    References:
        [EQ-INV-OLX-011], [EQ-INV-OLX-012], [REF-TARANTOLA-2005].

    """

    def __init__(
        self,
        dimension: int,
        covariance_function: Callable[
            [Vector],
            tuple[Sequence[Sequence[float]], Sequence[Sequence[Sequence[float]]]],
        ],
    ) -> None:
        """Store the dimension and covariance function.

        References:
            [EQ-INV-OLX-011].

        """
        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension < 1
        ):
            raise ValueError("dimension must be a positive integer")
        if not callable(covariance_function):
            raise TypeError("covariance_function must be callable")
        self._dimension = dimension
        self.covariance_function = covariance_function

    @property
    def size(self) -> int:
        """Return the observation dimension ``m``.

        References:
            [EQ-INV-OLX-011].

        """
        return self._dimension

    def _evaluate(
        self, eta: Vector
    ) -> tuple[Matrix, tuple[Matrix, ...], DenseCovariance]:
        """Return validated ``(Gamma, derivatives, factor)`` at ``eta``.

        References:
            [EQ-INV-OLX-011].

        """
        m = self._dimension
        matrix, derivatives = self.covariance_function(eta)
        gamma = _symmetric_matrix(matrix, m, "covariance")
        if len(derivatives) != m:
            raise ValueError("covariance_function must return one derivative per entry")
        checked = tuple(
            _symmetric_matrix(d, m, "covariance derivative") for d in derivatives
        )
        return gamma, checked, DenseCovariance(gamma)

    def log_likelihood(self, y: Sequence[float], eta: Sequence[float]) -> float:
        """Return ``-1/2 (r' Gamma^-1 r + log det Gamma + m log 2 pi)``.

        Args:
            y: Observations.
            eta: Predictions.

        Returns:
            The normalized log-likelihood.

        References:
            [EQ-INV-OLX-011].

        """
        y_vector, eta_vector = self._pair(y, eta)
        _, _, factor = self._evaluate(eta_vector)
        residual = [a - b for a, b in zip(y_vector, eta_vector, strict=True)]
        return -0.5 * (
            factor.quadratic_form(residual) + factor.log_det() + self.size * _LOG_2PI
        )

    def score(self, y: Sequence[float], eta: Sequence[float]) -> Vector:
        """Return ``a_j - tr(Gamma^-1 Gamma_j)/2 + a' Gamma_j a / 2``.

        Args:
            y: Observations.
            eta: Predictions.

        Returns:
            The exact gradient with respect to ``eta``.

        References:
            [EQ-INV-OLX-011].

        """
        y_vector, eta_vector = self._pair(y, eta)
        _, derivatives, factor = self._evaluate(eta_vector)
        residual = [a - b for a, b in zip(y_vector, eta_vector, strict=True)]
        a = tuple(float(v) for v in factor.solve(residual))
        precision = _inverse_columns(factor, self.size)
        result: list[float] = []
        for j, gj in enumerate(derivatives):
            quad = math.fsum(
                a[p] * gj[p][q] * a[q]
                for p in range(self.size)
                for q in range(self.size)
            )
            result.append(a[j] - 0.5 * _trace_product(precision, gj) + 0.5 * quad)
        return tuple(result)

    def fisher_matrix(self, eta: Sequence[float]) -> Matrix:
        """Return ``F_jk = (Gamma^-1)_jk + tr(Gamma^-1 Gamma_j Gamma^-1 Gamma_k)/2``.

        Args:
            eta: Predictions.

        Returns:
            The full symmetric positive-definite information matrix.

        References:
            [EQ-INV-OLX-012].

        """
        eta_vector = _real_vector(eta, "eta", size=self.size)
        _, derivatives, factor = self._evaluate(eta_vector)
        precision = _inverse_columns(factor, self.size)
        products = [_matmul(precision, gj) for gj in derivatives]
        m = self.size
        info = [
            [
                precision[j][k] + 0.5 * _trace_product(products[j], products[k])
                for k in range(m)
            ]
            for j in range(m)
        ]
        return tuple(
            tuple(0.5 * (info[j][k] + info[k][j]) for k in range(m)) for j in range(m)
        )

    def fisher_information(self, eta: Sequence[float]) -> Vector:
        """Return the diagonal of :meth:`fisher_matrix`.

        Args:
            eta: Predictions.

        Returns:
            The diagonal information.

        References:
            [EQ-INV-OLX-012].

        """
        matrix = self.fisher_matrix(eta)
        return tuple(matrix[i][i] for i in range(self.size))

    def simulate(
        self, eta: Sequence[float], seed: int | RandomStream | None = None
    ) -> Vector:
        """Draw ``eta + C z`` with ``Gamma(eta) = C C'``.

        Args:
            eta: Predictions.
            seed: Integer seed or stream.

        Returns:
            One simulated data vector.

        References:
            [EQ-INV-OLX-011].

        """
        eta_vector = _real_vector(eta, "eta", size=self.size)
        _, _, factor = self._evaluate(eta_vector)
        generator = resolve_stream(seed).generator()
        noise = factor.unwhiten([generator.gauss(0.0, 1.0) for _ in eta_vector])
        return tuple(e + float(n) for e, n in zip(eta_vector, noise, strict=True))

    def irls_system(
        self, y: Sequence[float], eta: Sequence[float]
    ) -> tuple[LinearOperator, Vector]:
        """Return ``W = L'`` (``F = L L'``) and ``z = eta + F^-1 score``.

        The step ``min ||W (z - A x)||^2`` is the exact Fisher-scoring step
        with the full information matrix.

        Args:
            y: Observations.
            eta: Predictions.

        Returns:
            The whitening operator and the working response.

        References:
            [EQ-INV-OLX-012].

        """
        eta_vector = _real_vector(eta, "eta", size=self.size)
        score = self.score(y, eta)
        lower = cholesky(self.fisher_matrix(eta))
        step = cholesky_solve(lower, score)
        response = tuple(e + float(s) for e, s in zip(eta_vector, step, strict=True))
        upper = tuple(
            tuple(lower[j][i] for j in range(self.size)) for i in range(self.size)
        )
        return DenseOperator(upper), response

    def restrict(self, indices: Sequence[int]) -> ObservationLaw:
        """Not supported: the marginal depends on the dropped predictions.

        Args:
            indices: Ignored.

        Raises:
            NotImplementedError: Always.

        References:
            [EQ-INV-OLX-011].

        """
        raise NotImplementedError(
            "a state-dependent covariance has no marginal law on a subset of "
            "the entries; masks are not supported"
        )


# ---------------------------------------------------------------------------
# 7. Nuisance scale: profile likelihood and inverse-gamma marginalisation
# ---------------------------------------------------------------------------


class ProfiledScaleGaussianLaw(ObservationLaw):
    """Gaussian errors ``N(eta, sigma^2 Gamma0)`` with ``sigma`` profiled out.

    ``Gamma0`` is a known covariance factor and the common scale ``sigma`` is
    unknown. For fixed ``eta`` the maximizer is
    ``sigma_hat^2 = q / m`` with ``q = r' Gamma0^-1 r``, and the PROFILE
    log-likelihood is ``-(m/2)(log 2 pi + 1 + log(q/m)) - 1/2 log det Gamma0``
    [EQ-INV-OLX-013]. The gradient is ``(m / q) Gamma0^-1 r``. A profile
    likelihood is not a probability model for ``y``: it is not normalized over
    ``y``, it is biased for ``sigma`` (the divisor is ``m``, not ``m - p``),
    and it ignores the uncertainty in ``sigma``; therefore
    :meth:`fisher_information` and :meth:`simulate` are not defined (use
    :meth:`simulate_with_scale`). The IRLS step is the weighted
    least-squares step with ``Gamma0^-1 / sigma_hat^2``, which for a linear
    predictor is the exact generalized least-squares estimate
    [EQ-INV-OLX-013].

    Args:
        covariance: Known covariance factor ``Gamma0``.

    Raises:
        TypeError: If ``covariance`` is not a covariance factor.

    References:
        [EQ-INV-OLX-013], [REF-TARANTOLA-2005].

    """

    def __init__(self, covariance: CovarianceFactor) -> None:
        """Store the covariance factor.

        References:
            [EQ-INV-OLX-013].

        """
        if not is_covariance_factor(covariance):
            raise TypeError("covariance must be a probstat.linalg covariance factor")
        self.covariance = covariance

    @property
    def size(self) -> int:
        """Return the observation dimension ``m``.

        References:
            [EQ-INV-OLX-013].

        """
        return int(self.covariance.dimension)

    def _quadratic(self, y: Sequence[float], eta: Sequence[float]) -> float:
        """Return ``q = r' Gamma0^-1 r`` (must be positive).

        References:
            [EQ-INV-OLX-013].

        """
        y_vector, eta_vector = self._pair(y, eta)
        q = float(
            self.covariance.quadratic_form(
                [a - b for a, b in zip(y_vector, eta_vector, strict=True)]
            )
        )
        if q <= 0.0:
            raise ValueError(
                "zero residual: the profile likelihood in sigma is unbounded"
            )
        return q

    def scale_estimate(self, y: Sequence[float], eta: Sequence[float]) -> float:
        """Return the profile maximizer ``sigma_hat = sqrt(q / m)``.

        Args:
            y: Observations.
            eta: Predictions.

        Returns:
            ``sigma_hat`` (the maximum-likelihood, not the unbiased, estimate).

        Raises:
            ValueError: If the residual is exactly zero.

        References:
            [EQ-INV-OLX-013].

        """
        return math.sqrt(self._quadratic(y, eta) / self.size)

    def log_likelihood(self, y: Sequence[float], eta: Sequence[float]) -> float:
        """Return the profile log-likelihood at ``sigma = sigma_hat(eta)``.

        Args:
            y: Observations.
            eta: Predictions.

        Returns:
            ``-(m/2)(log 2 pi + 1 + log(q/m)) - log det(Gamma0)/2``.

        Raises:
            ValueError: If the residual is exactly zero.

        References:
            [EQ-INV-OLX-013].

        """
        q = self._quadratic(y, eta)
        m = self.size
        return -0.5 * m * (_LOG_2PI + 1.0 + math.log(q / m)) - 0.5 * float(
            self.covariance.log_det()
        )

    def score(self, y: Sequence[float], eta: Sequence[float]) -> Vector:
        """Return ``(m / q) Gamma0^-1 r``.

        Args:
            y: Observations.
            eta: Predictions.

        Returns:
            The gradient of the profile log-likelihood.

        References:
            [EQ-INV-OLX-013].

        """
        q = self._quadratic(y, eta)
        y_vector, eta_vector = self._pair(y, eta)
        solved = self.covariance.solve(
            [a - b for a, b in zip(y_vector, eta_vector, strict=True)]
        )
        return tuple(self.size / q * float(v) for v in solved)

    def fisher_information(self, eta: Sequence[float]) -> Vector:
        """Not defined: the profile likelihood has no expected information.

        Args:
            eta: Ignored.

        Raises:
            NotImplementedError: Always.

        References:
            [EQ-INV-OLX-013].

        """
        raise NotImplementedError(
            "a profile likelihood is not a probability model for y; "
            "use MarginalScaleGaussianLaw for a proper model"
        )

    def simulate(
        self, eta: Sequence[float], seed: int | RandomStream | None = None
    ) -> Vector:
        """Not defined without a value of ``sigma``; use :meth:`simulate_with_scale`.

        Args:
            eta: Ignored.
            seed: Ignored.

        Raises:
            NotImplementedError: Always.

        References:
            [EQ-INV-OLX-013].

        """
        raise NotImplementedError("sigma is unknown: use simulate_with_scale")

    def simulate_with_scale(
        self,
        eta: Sequence[float],
        sigma: float,
        seed: int | RandomStream | None = None,
    ) -> Vector:
        """Draw ``eta + sigma C z`` for a stated scale.

        Args:
            eta: Predictions.
            sigma: Positive scale.
            seed: Integer seed or stream.

        Returns:
            One simulated data vector.

        References:
            [EQ-INV-OLX-013].

        """
        eta_vector = _real_vector(eta, "eta", size=self.size)
        s = _positive(sigma, "sigma")
        generator = resolve_stream(seed).generator()
        noise = self.covariance.unwhiten(
            [generator.gauss(0.0, 1.0) for _ in eta_vector]
        )
        return tuple(e + s * float(n) for e, n in zip(eta_vector, noise, strict=True))

    def irls_system(
        self, y: Sequence[float], eta: Sequence[float]
    ) -> tuple[LinearOperator, Vector]:
        """Return ``W = C^-1 / sigma_hat`` and ``z = y``.

        Args:
            y: Observations.
            eta: Current predictions.

        Returns:
            The scaled whitening and ``y``.

        References:
            [EQ-INV-OLX-013].

        """
        y_vector, _ = self._pair(y, eta)
        sigma = self.scale_estimate(y, eta)
        return ScaledOperator(
            self.covariance.whitening_operator(), 1.0 / sigma
        ), y_vector

    def restrict(self, indices: Sequence[int]) -> ProfiledScaleGaussianLaw:
        """Return the law with the marginal covariance submatrix at ``indices``.

        Args:
            indices: Strictly increasing indices.

        Returns:
            A :class:`ProfiledScaleGaussianLaw`.

        References:
            [EQ-INV-OLX-013].

        """
        return ProfiledScaleGaussianLaw(
            GaussianLaw(self.covariance).restrict(indices).covariance
        )


class MarginalScaleGaussianLaw(ObservationLaw):
    """Gaussian errors ``N(eta, sigma^2 Gamma0)`` with ``sigma^2`` integrated out.

    The prior is inverse-gamma, ``sigma^2 ~ IG(alpha, beta)`` with density
    ``beta^alpha / Gamma(alpha) x^(-alpha-1) exp(-beta / x)`` (shape ``alpha``,
    scale ``beta``). Integrating the Gaussian over ``sigma^2`` gives the
    marginal likelihood
    ``log p = log Gamma(alpha + m/2) - log Gamma(alpha) + alpha log beta
    - (m/2) log(2 pi) - log det(Gamma0)/2 - (alpha + m/2) log(beta + q/2)``
    with ``q = r' Gamma0^-1 r`` [EQ-INV-OLX-014]. This is the multivariate
    ``t_{2 alpha}(eta, (beta/alpha) Gamma0)``: heavy tails arise from
    uncertainty about the scale, and the posterior of ``sigma^2`` given
    ``eta`` is ``IG(alpha + m/2, beta + q/2)`` (:meth:`scale_posterior`).

    Args:
        covariance: Known covariance factor ``Gamma0``.
        shape: Inverse-gamma shape ``alpha > 0``.
        scale: Inverse-gamma scale ``beta > 0`` (units of variance).

    Raises:
        TypeError: For wrong types.
        ValueError: For non-positive hyperparameters.

    References:
        [EQ-INV-OLX-014], [REF-GELMAN-ETAL-2013], [REF-KOTZ-NADARAJAH-2004].

    """

    def __init__(
        self, covariance: CovarianceFactor, shape: float, scale: float
    ) -> None:
        """Validate the hyperparameters.

        References:
            [EQ-INV-OLX-014].

        """
        if not is_covariance_factor(covariance):
            raise TypeError("covariance must be a probstat.linalg covariance factor")
        self.covariance = covariance
        self.shape = _positive(shape, "shape")
        self.scale = _positive(scale, "scale")

    @property
    def size(self) -> int:
        """Return the observation dimension ``m``.

        References:
            [EQ-INV-OLX-014].

        """
        return int(self.covariance.dimension)

    @property
    def degrees_of_freedom(self) -> float:
        """Return the equivalent Student-t degrees of freedom ``2 alpha``.

        References:
            [EQ-INV-OLX-014].

        """
        return 2.0 * self.shape

    @property
    def scatter_factor(self) -> float:
        """Return ``beta / alpha``: the scatter matrix is this times ``Gamma0``.

        References:
            [EQ-INV-OLX-014].

        """
        return self.scale / self.shape

    def _quadratic(self, y: Sequence[float], eta: Sequence[float]) -> float:
        """Return ``q = r' Gamma0^-1 r``.

        References:
            [EQ-INV-OLX-014].

        """
        y_vector, eta_vector = self._pair(y, eta)
        return float(
            self.covariance.quadratic_form(
                [a - b for a, b in zip(y_vector, eta_vector, strict=True)]
            )
        )

    def scale_posterior(
        self, y: Sequence[float], eta: Sequence[float]
    ) -> tuple[float, float]:
        """Return ``(alpha + m/2, beta + q/2)`` of ``sigma^2 | y, eta``.

        Args:
            y: Observations.
            eta: Predictions.

        Returns:
            Shape and scale of the inverse-gamma posterior of ``sigma^2``.

        References:
            [EQ-INV-OLX-014].

        """
        q = self._quadratic(y, eta)
        return self.shape + 0.5 * self.size, self.scale + 0.5 * q

    def log_likelihood(self, y: Sequence[float], eta: Sequence[float]) -> float:
        """Return the marginal log-likelihood of [EQ-INV-OLX-014].

        Args:
            y: Observations.
            eta: Predictions.

        Returns:
            The normalized log-likelihood after integrating out ``sigma^2``.

        References:
            [EQ-INV-OLX-014].

        """
        q = self._quadratic(y, eta)
        m = self.size
        return math.fsum(
            (
                log_gamma_ratio(self.shape, 0.5 * m),
                self.shape * math.log(self.scale),
                -0.5 * m * _LOG_2PI,
                -0.5 * float(self.covariance.log_det()),
                -(self.shape + 0.5 * m) * math.log(self.scale + 0.5 * q),
            )
        )

    def _weight(self, y: Sequence[float], eta: Sequence[float]) -> float:
        """Return ``(alpha + m/2) / (beta + q/2)``.

        References:
            [EQ-INV-OLX-014].

        """
        return (self.shape + 0.5 * self.size) / (
            self.scale + 0.5 * self._quadratic(y, eta)
        )

    def score(self, y: Sequence[float], eta: Sequence[float]) -> Vector:
        """Return ``(alpha + m/2) / (beta + q/2) Gamma0^-1 r``.

        Args:
            y: Observations.
            eta: Predictions.

        Returns:
            The score vector.

        References:
            [EQ-INV-OLX-014].

        """
        y_vector, eta_vector = self._pair(y, eta)
        weight = self._weight(y, eta)
        solved = self.covariance.solve(
            [a - b for a, b in zip(y_vector, eta_vector, strict=True)]
        )
        return tuple(weight * float(v) for v in solved)

    def fisher_information(self, eta: Sequence[float]) -> Vector:
        """Return ``(2 alpha + m)/(2 alpha + m + 2) (alpha / beta) diag(Gamma0^-1)``.

        Args:
            eta: Predictions (only the length is used).

        Returns:
            The diagonal of the information of the marginal t law.

        References:
            [EQ-INV-OLX-014].

        """
        _real_vector(eta, "eta", size=self.size)
        m = self.size
        factor = (2.0 * self.shape + m) / (2.0 * self.shape + m + 2.0) * self.shape
        diagonal = _inverse_columns(self.covariance, m)
        return tuple(factor / self.scale * diagonal[i][i] for i in range(m))

    def simulate(
        self, eta: Sequence[float], seed: int | RandomStream | None = None
    ) -> Vector:
        """Draw ``sigma^2 ~ IG(alpha, beta)`` then ``eta + sigma C z``.

        Args:
            eta: Predictions.
            seed: Integer seed or stream.

        Returns:
            One simulated data vector from the marginal law.

        References:
            [EQ-INV-OLX-014].

        """
        eta_vector = _real_vector(eta, "eta", size=self.size)
        generator = resolve_stream(seed).generator()
        variance = self.scale / generator.gammavariate(self.shape, 1.0)
        noise = self.covariance.unwhiten(
            [generator.gauss(0.0, 1.0) for _ in eta_vector]
        )
        sigma = math.sqrt(variance)
        return tuple(
            e + sigma * float(n) for e, n in zip(eta_vector, noise, strict=True)
        )

    def irls_system(
        self, y: Sequence[float], eta: Sequence[float]
    ) -> tuple[LinearOperator, Vector]:
        """Return ``W = sqrt((alpha + m/2)/(beta + q/2)) C^-1`` and ``z = y``.

        The step minimizes a quadratic majorizer of the negative marginal
        log-likelihood (the logarithm is concave in ``q``), so it never
        increases the objective.

        Args:
            y: Observations.
            eta: Current predictions.

        Returns:
            The weighted whitening and ``y``.

        References:
            [EQ-INV-OLX-014], [REF-LANGE-LITTLE-TAYLOR-1989].

        """
        y_vector, _ = self._pair(y, eta)
        weight = self._weight(y, eta)
        return (
            ScaledOperator(self.covariance.whitening_operator(), math.sqrt(weight)),
            y_vector,
        )

    def restrict(self, indices: Sequence[int]) -> MarginalScaleGaussianLaw:
        """Return the marginal law on a subset (the scale prior is shared).

        Args:
            indices: Strictly increasing indices.

        Returns:
            A :class:`MarginalScaleGaussianLaw` with the sub-covariance.

        References:
            [EQ-INV-OLX-014].

        """
        sub = GaussianLaw(self.covariance).restrict(indices).covariance
        return MarginalScaleGaussianLaw(sub, self.shape, self.scale)
