"""Observation laws ``y ~ p(y | eta)`` for vector inverse problems.

A vector inverse problem predicts ``eta = G(x)`` and observes ``y``. An
*observation law* is the normalized density (or mass function) ``p(y | eta)``
of the data given the prediction. Every law in this module exposes

- ``log_likelihood(y, eta)``: the fully normalized ``log p(y | eta)``, so that
  evidences and model comparisons are meaningful;
- ``score(y, eta)``: the vector ``d log p / d eta``;
- ``fisher_information(eta)``: the diagonal of the expected information
  ``E[-d^2 log p / d eta^2]`` with respect to ``eta``;
- ``simulate(eta, seed)``: one reproducible draw from the same law that the
  likelihood evaluates;
- ``irls_system(y, eta)``: a whitening operator ``W`` and working response
  ``z`` such that a Fisher-scoring step is the weighted least-squares problem
  ``min_x ||W (z - A x)||^2`` for a linear predictor ``eta = A x``.

The laws are :class:`GaussianLaw` (any covariance factor of
:mod:`probstat.linalg`, correlated errors allowed) [EQ-INV-OBSLAW-001],
:class:`PoissonLaw` (identity or log link) [EQ-INV-OBSLAW-002],
:class:`BinomialLaw` (known trials, logit link) [EQ-INV-OBSLAW-003],
:class:`StudentTLaw` (independent heavy-tailed errors; NOT a multivariate t)
[EQ-INV-OBSLAW-004] and :class:`CensoredGaussianLaw` (detection limits, with
stable log-CDF and log-survival tails) [EQ-INV-OBSLAW-005],
[EQ-INV-OBSLAW-006]. :class:`MissingMask` ignores masked observations
explicitly; a correlated Gaussian uses the marginal covariance submatrix of
the observed entries, never a sub-selection of the precision
[EQ-INV-OBSLAW-007]. Missingness is assumed ignorable (missing at random);
informative missingness is not modelled.

:func:`maximum_likelihood_linear` computes the penalized maximum-likelihood or
MAP estimate for a linear predictor by Fisher scoring (IRLS) with a
backtracking line search on the true penalized negative log-likelihood
[EQ-INV-OBSLAW-008]. For :class:`StudentTLaw` the weights are
``(nu + 1) / (nu s^2 + r^2)`` [EQ-INV-OBSLAW-009] and the iteration finds a
local optimum of a non-convex objective.

Experimental (added in 0.28).

References:
    [REF-MCCULLAGH-NELDER-1989], [REF-LANGE-LITTLE-TAYLOR-1989],
    [REF-TARANTOLA-2005], [REF-NOCEDAL-WRIGHT-2006].

"""

from __future__ import annotations

import math
import sys
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from probstat.distributions import Binomial, Normal, Poisson, StudentT
from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.linalg import (
    DenseCovariance,
    DiagonalCovariance,
    DiagonalOperator,
    IdentityOperator,
    LinearOperator,
    ScaledOperator,
    SelectionOperator,
    Vector,
    VStackOperator,
    lsqr,
    require_operator,
)
from probstat.linalg.covariance_operators import CovarianceFactor, is_covariance_factor
from probstat.linalg.operators import ComposedOperator

_STANDARD_NORMAL = Normal(0.0, 1.0)
_HALF_LOG_TWO_PI = 0.5 * math.log(2.0 * math.pi)
_LOG_RATE_BOUND = 700.0


def _real_vector(
    values: Sequence[float],
    name: str,
    *,
    size: int | None = None,
    finite: bool = True,
) -> Vector:
    """Return ``values`` as a tuple of floats after validation.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].

    """
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of real numbers")
    result: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must contain real numbers")
        number = float(value)
        if finite and not math.isfinite(number):
            raise ValueError(f"{name} must be finite")
        result.append(number)
    if size is not None and len(result) != size:
        raise ValueError(f"{name} must have length {size}, got {len(result)}")
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return tuple(result)


def _positive_vector(values: Sequence[float], name: str) -> Vector:
    """Return strictly positive finite floats.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].

    """
    result = _real_vector(values, name)
    if any(value <= 0.0 for value in result):
        raise ValueError(f"{name} must be strictly positive")
    return result


def _softplus(value: float) -> float:
    """Return ``log(1 + exp(value))`` without overflow.

    References:
        [EQ-INV-OBSLAW-003].

    """
    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))


def _std_log_pdf(z: float) -> float:
    """Return ``log phi(z)`` of the standard normal density.

    References:
        [EQ-INV-OBSLAW-005].

    """
    return -0.5 * z * z - _HALF_LOG_TWO_PI


class ObservationLaw(ABC):
    """Abstract observation law ``p(y | eta)`` of a prediction vector ``eta``.

    Concrete laws validate their inputs strictly (``TypeError`` or
    ``ValueError``); nothing is clipped or imputed.

    References:
        [EQ-INV-OBSLAW-008], [REF-MCCULLAGH-NELDER-1989].

    """

    @property
    @abstractmethod
    def size(self) -> int | None:
        """Return the observation dimension, or ``None`` if any length is valid.

        References:
            [EQ-INV-OBSLAW-008].

        """

    @abstractmethod
    def log_likelihood(self, y: Sequence[float], eta: Sequence[float]) -> float:
        """Return the fully normalized ``log p(y | eta)``.

        Args:
            y: Observations.
            eta: Predictions.

        Returns:
            The log-likelihood (``-inf`` is never returned for valid inputs
            unless the law assigns zero probability to ``y``).

        References:
            [EQ-INV-OBSLAW-008].

        """

    @abstractmethod
    def score(self, y: Sequence[float], eta: Sequence[float]) -> Vector:
        """Return ``d log p(y | eta) / d eta`` as a vector.

        Args:
            y: Observations.
            eta: Predictions.

        Returns:
            The score vector.

        References:
            [EQ-INV-OBSLAW-008].

        """

    @abstractmethod
    def fisher_information(self, eta: Sequence[float]) -> Vector:
        """Return the diagonal of the expected information with respect to eta.

        Args:
            eta: Predictions.

        Returns:
            Nonnegative diagonal entries of ``E[-d^2 log p / d eta^2]``.

        References:
            [EQ-INV-OBSLAW-008].

        """

    @abstractmethod
    def simulate(
        self, eta: Sequence[float], seed: int | RandomStream | None = None
    ) -> Vector:
        """Draw one observation vector from ``p(. | eta)``.

        Args:
            eta: Predictions.
            seed: Integer seed or :class:`RandomStream`; ``None`` draws fresh
                entropy and is not reproducible.

        Returns:
            One simulated data vector.

        References:
            [EQ-INV-OBSLAW-008].

        """

    @abstractmethod
    def irls_system(
        self, y: Sequence[float], eta: Sequence[float]
    ) -> tuple[LinearOperator, Vector]:
        """Return ``(W, z)`` defining the weighted least-squares scoring step.

        With ``W'W`` the working weights and ``z`` the working response, the
        Fisher-scoring (or IRLS) step for ``eta = A x`` solves
        ``min_x ||W (z - A x)||^2`` (plus a penalty).

        Args:
            y: Observations.
            eta: Current predictions.

        Returns:
            The whitening operator ``W`` (``k x m``) and ``z`` (length ``m``).

        References:
            [EQ-INV-OBSLAW-008], [REF-MCCULLAGH-NELDER-1989].

        """

    @abstractmethod
    def restrict(self, indices: Sequence[int]) -> ObservationLaw:
        """Return the marginal law of the observations at ``indices``.

        Args:
            indices: Strictly increasing observation indices.

        Returns:
            The law of ``y[indices]`` given ``eta[indices]``.

        References:
            [EQ-INV-OBSLAW-007].

        """

    def _pair(self, y: Sequence[float], eta: Sequence[float]) -> tuple[Vector, Vector]:
        """Validate ``y`` and ``eta`` as equal-length finite vectors.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].

        """
        eta_vector = _real_vector(eta, "eta", size=self.size)
        y_vector = _real_vector(y, "y", size=len(eta_vector))
        return y_vector, eta_vector


def _check_indices(indices: Sequence[int], size: int | None) -> tuple[int, ...]:
    """Return strictly increasing in-range integer indices.

    References:
        [EQ-INV-OBSLAW-007].

    """
    result: list[int] = []
    for index in indices:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("indices must be integers")
        if index < 0 or (size is not None and index >= size):
            raise ValueError("index out of range")
        if result and index <= result[-1]:
            raise ValueError("indices must be strictly increasing")
        result.append(index)
    if not result:
        raise ValueError("indices must be non-empty")
    return tuple(result)


def _diagonal_whitening(weights: Sequence[float]) -> LinearOperator:
    """Return ``diag(sqrt(w))`` for nonnegative weights.

    References:
        [EQ-INV-OBSLAW-008].

    """
    return DiagonalOperator([math.sqrt(w) for w in weights])


def _scoring_response(
    eta: Vector, score: Vector, information: Vector
) -> tuple[Vector, Vector]:
    """Return working weights and response ``z = eta + score / information``.

    A zero information entry contributes weight zero and ``z = eta`` (the
    observation carries no first-order information at this ``eta``).

    References:
        [EQ-INV-OBSLAW-008].

    """
    response = tuple(
        e + s / w if w > 0.0 else e
        for e, s, w in zip(eta, score, information, strict=True)
    )
    return information, response


class GaussianLaw(ObservationLaw):
    """Gaussian law ``y ~ N(eta, Gamma)`` with a covariance factor.

    ``log p = -1/2 (r' Gamma^-1 r + log det Gamma + m log 2 pi)`` with
    ``r = y - eta`` [EQ-INV-OBSLAW-001]; correlated errors are allowed.

    Args:
        covariance: A covariance factor of :mod:`probstat.linalg`.

    Raises:
        TypeError: If ``covariance`` is not a covariance factor.

    References:
        [EQ-INV-OBSLAW-001], [REF-TARANTOLA-2005].

    """

    def __init__(self, covariance: CovarianceFactor) -> None:
        """Store the covariance factor.

        References:
            [EQ-INV-OBSLAW-001].

        """
        if not is_covariance_factor(covariance):
            raise TypeError("covariance must be a probstat.linalg covariance factor")
        self.covariance = covariance

    @property
    def size(self) -> int:
        """Return the observation dimension ``m``.

        References:
            [EQ-INV-OBSLAW-001].

        """
        return int(self.covariance.dimension)

    def log_likelihood(self, y: Sequence[float], eta: Sequence[float]) -> float:
        """Return ``-1/2 (r' Gamma^-1 r + log det Gamma + m log 2 pi)``.

        Args:
            y: Observations (length ``m``).
            eta: Predictions (length ``m``).

        Returns:
            The normalized Gaussian log-density.

        References:
            [EQ-INV-OBSLAW-001].

        """
        y_vector, eta_vector = self._pair(y, eta)
        residual = [a - b for a, b in zip(y_vector, eta_vector, strict=True)]
        quadratic = self.covariance.quadratic_form(residual)
        return -0.5 * (
            quadratic + self.covariance.log_det() + self.size * 2.0 * _HALF_LOG_TWO_PI
        )

    def score(self, y: Sequence[float], eta: Sequence[float]) -> Vector:
        """Return ``Gamma^-1 (y - eta)``.

        Args:
            y: Observations.
            eta: Predictions.

        Returns:
            The score with respect to ``eta``.

        References:
            [EQ-INV-OBSLAW-001].

        """
        y_vector, eta_vector = self._pair(y, eta)
        residual = [a - b for a, b in zip(y_vector, eta_vector, strict=True)]
        return tuple(self.covariance.solve(residual))

    def fisher_information(self, eta: Sequence[float]) -> Vector:
        """Return ``diag(Gamma^-1)`` (the full information is ``Gamma^-1``).

        Uses ``m`` solves for a non-diagonal covariance.

        Args:
            eta: Predictions (only the length is used).

        Returns:
            The diagonal of the precision matrix.

        References:
            [EQ-INV-OBSLAW-001].

        """
        eta_vector = _real_vector(eta, "eta", size=self.size)
        if isinstance(self.covariance, DiagonalCovariance):
            return tuple(1.0 / (s * s) for s in self.covariance.standard_deviations)
        m = len(eta_vector)
        diagonal: list[float] = []
        for i in range(m):
            unit = [0.0] * m
            unit[i] = 1.0
            diagonal.append(float(self.covariance.solve(unit)[i]))
        return tuple(diagonal)

    def simulate(
        self, eta: Sequence[float], seed: int | RandomStream | None = None
    ) -> Vector:
        """Draw ``eta + C z`` with ``z ~ N(0, I)`` and ``Gamma = C C'``.

        Args:
            eta: Predictions.
            seed: Integer seed or stream.

        Returns:
            One simulated data vector.

        References:
            [EQ-INV-OBSLAW-001].

        """
        eta_vector = _real_vector(eta, "eta", size=self.size)
        generator = resolve_stream(seed).generator()
        z = [generator.gauss(0.0, 1.0) for _ in eta_vector]
        noise = self.covariance.unwhiten(z)
        return tuple(e + n for e, n in zip(eta_vector, noise, strict=True))

    def irls_system(
        self, y: Sequence[float], eta: Sequence[float]
    ) -> tuple[LinearOperator, Vector]:
        """Return the whitening ``W = C^-1`` and ``z = y`` (the step is exact).

        Args:
            y: Observations.
            eta: Predictions (validated only).

        Returns:
            ``(W, y)``.

        References:
            [EQ-INV-OBSLAW-008].

        """
        y_vector, _ = self._pair(y, eta)
        return self.covariance.whitening_operator(), y_vector

    def restrict(self, indices: Sequence[int]) -> GaussianLaw:
        """Return the law with the marginal covariance submatrix at ``indices``.

        The marginal covariance of ``y[k]`` is ``Gamma[k, k]``; this is NOT the
        sub-selection of the precision matrix (which would be the conditional
        covariance).

        Args:
            indices: Strictly increasing indices.

        Returns:
            A :class:`GaussianLaw` of dimension ``len(indices)``.

        References:
            [EQ-INV-OBSLAW-007].

        """
        chosen = _check_indices(indices, self.size)
        if isinstance(self.covariance, DiagonalCovariance):
            sd = self.covariance.standard_deviations
            return GaussianLaw(DiagonalCovariance([sd[i] ** 2 for i in chosen]))
        m = self.size
        columns: list[Vector] = []
        for j in chosen:
            unit = [0.0] * m
            unit[j] = 1.0
            full = self.covariance.apply(unit)
            columns.append(tuple(float(full[i]) for i in chosen))
        sub = [
            [0.5 * (columns[j][i] + columns[i][j]) for j in range(len(chosen))]
            for i in range(len(chosen))
        ]
        return GaussianLaw(DenseCovariance(sub))


class PoissonLaw(ObservationLaw):
    """Poisson counts ``y_i ~ Poisson(mu_i)`` with identity or log link.

    ``link="identity"`` means ``eta`` is the rate and must be strictly
    positive; ``link="log"`` means ``mu = exp(eta)`` (``|eta| <= 700``).
    ``y = 0`` is handled exactly (the term ``y log mu`` is dropped), so the
    limit ``eta -> 0+`` is finite and free of ``nan`` [EQ-INV-OBSLAW-002].

    Args:
        link: ``"identity"`` or ``"log"``.

    Raises:
        ValueError: If ``link`` is not one of the two names.

    References:
        [EQ-INV-OBSLAW-002], [REF-MCCULLAGH-NELDER-1989].

    """

    def __init__(self, link: str = "identity") -> None:
        """Validate the link name.

        References:
            [EQ-INV-OBSLAW-002].

        """
        if link not in ("identity", "log"):
            raise ValueError("link must be 'identity' or 'log'")
        self.link = link

    @property
    def size(self) -> None:
        """Return ``None``: any observation length is valid.

        References:
            [EQ-INV-OBSLAW-002].

        """
        return None

    def _counts_and_rates(
        self, y: Sequence[float], eta: Sequence[float]
    ) -> tuple[Vector, Vector]:
        """Validate counts and return ``(y, mu)`` with the rate ``mu``.

        References:
            [EQ-INV-OBSLAW-002].

        """
        y_vector, eta_vector = self._pair(y, eta)
        for value in y_vector:
            if value < 0.0 or value != math.floor(value):
                raise ValueError("Poisson counts must be nonnegative integers")
        if self.link == "identity":
            if any(e <= 0.0 for e in eta_vector):
                raise ValueError("identity link requires eta > 0 for every entry")
            return y_vector, eta_vector
        if any(abs(e) > _LOG_RATE_BOUND for e in eta_vector):
            raise ValueError("log link requires |eta| <= 700")
        return y_vector, tuple(math.exp(e) for e in eta_vector)

    def log_likelihood(self, y: Sequence[float], eta: Sequence[float]) -> float:
        """Return ``sum_i log Poisson(y_i; mu_i)``.

        The identity link delegates to :class:`probstat.distributions.Poisson`;
        the log link uses ``y eta - exp(eta) - log y!`` (exact for ``eta``
        so negative that ``exp(eta)`` underflows).

        Args:
            y: Nonnegative integer counts.
            eta: Rates (identity) or log-rates (log).

        Returns:
            The normalized log-likelihood.

        Raises:
            ValueError: For negative or non-integer counts, ``eta <= 0`` under
                the identity link, or ``|eta| > 700`` under the log link.

        References:
            [EQ-INV-OBSLAW-002].

        """
        y_vector, mu = self._counts_and_rates(y, eta)
        if self.link == "identity":
            return math.fsum(
                Poisson(m).log_pmf(int(k)) for k, m in zip(y_vector, mu, strict=True)
            )
        eta_vector = _real_vector(eta, "eta")
        return math.fsum(
            k * e - m - math.lgamma(k + 1.0)
            for k, e, m in zip(y_vector, eta_vector, mu, strict=True)
        )

    def score(self, y: Sequence[float], eta: Sequence[float]) -> Vector:
        """Return ``y/eta - 1`` (identity) or ``y - exp(eta)`` (log).

        Args:
            y: Counts.
            eta: Predictions.

        Returns:
            The score vector.

        References:
            [EQ-INV-OBSLAW-002].

        """
        y_vector, mu = self._counts_and_rates(y, eta)
        if self.link == "identity":
            return tuple(k / m - 1.0 for k, m in zip(y_vector, mu, strict=True))
        return tuple(k - m for k, m in zip(y_vector, mu, strict=True))

    def fisher_information(self, eta: Sequence[float]) -> Vector:
        """Return ``1/eta`` (identity) or ``exp(eta)`` (log).

        Args:
            eta: Predictions.

        Returns:
            The diagonal information.

        References:
            [EQ-INV-OBSLAW-002].

        """
        _, mu = self._counts_and_rates((0.0,) * len(tuple(eta)), eta)
        if self.link == "identity":
            return tuple(1.0 / m for m in mu)
        return mu

    def simulate(
        self, eta: Sequence[float], seed: int | RandomStream | None = None
    ) -> Vector:
        """Draw independent Poisson counts.

        Args:
            eta: Predictions.
            seed: Integer seed or stream.

        Returns:
            Counts as floats.

        References:
            [EQ-INV-OBSLAW-002].

        """
        _, mu = self._counts_and_rates((0.0,) * len(tuple(eta)), eta)
        stream = resolve_stream(seed)
        return tuple(
            float(Poisson(m).sample(1, seed=stream.child(i))[0])
            for i, m in enumerate(mu)
        )

    def irls_system(
        self, y: Sequence[float], eta: Sequence[float]
    ) -> tuple[LinearOperator, Vector]:
        """Return Fisher-scoring weights ``I(eta)`` and ``z = eta + s/I``.

        Args:
            y: Counts.
            eta: Predictions.

        Returns:
            ``(diag(sqrt(I)), z)``.

        References:
            [EQ-INV-OBSLAW-008].

        """
        eta_vector = _real_vector(eta, "eta")
        weights, response = _scoring_response(
            eta_vector, self.score(y, eta), self.fisher_information(eta)
        )
        return _diagonal_whitening(weights), response

    def restrict(self, indices: Sequence[int]) -> PoissonLaw:
        """Return the same law (it is elementwise).

        Args:
            indices: Strictly increasing indices.

        Returns:
            ``self``.

        References:
            [EQ-INV-OBSLAW-007].

        """
        _check_indices(indices, None)
        return self


class BinomialLaw(ObservationLaw):
    """Binomial counts ``y_i ~ Bin(n_i, p_i)`` with ``p = expit(eta)``.

    The log-likelihood uses ``log p = -softplus(-eta)`` and
    ``log(1-p) = -softplus(eta)``, which stay finite for any finite ``eta``
    [EQ-INV-OBSLAW-003].

    Args:
        trials: Known nonnegative integer trial counts ``n_i``.
        link: Only ``"logit"`` is supported.

    Raises:
        ValueError: For an unsupported link or invalid trials.

    References:
        [EQ-INV-OBSLAW-003], [REF-MCCULLAGH-NELDER-1989].

    """

    def __init__(self, trials: Sequence[int], link: str = "logit") -> None:
        """Validate trials and link.

        References:
            [EQ-INV-OBSLAW-003].

        """
        if link != "logit":
            raise ValueError("only link='logit' is supported")
        checked: list[int] = []
        for n in trials:
            if isinstance(n, bool) or not isinstance(n, int):
                raise TypeError("trials must be integers")
            if n < 0:
                raise ValueError("trials must be nonnegative")
            checked.append(n)
        if not checked:
            raise ValueError("trials must be non-empty")
        self.trials = tuple(checked)
        self.link = link

    @property
    def size(self) -> int:
        """Return the number of binomial observations.

        References:
            [EQ-INV-OBSLAW-003].

        """
        return len(self.trials)

    def _counts(
        self, y: Sequence[float], eta: Sequence[float]
    ) -> tuple[Vector, Vector]:
        """Validate ``0 <= y_i <= n_i`` integer counts.

        References:
            [EQ-INV-OBSLAW-003].

        """
        y_vector, eta_vector = self._pair(y, eta)
        for k, n in zip(y_vector, self.trials, strict=True):
            if k < 0.0 or k > n or k != math.floor(k):
                raise ValueError("binomial counts must be integers in [0, trials]")
        return y_vector, eta_vector

    def log_likelihood(self, y: Sequence[float], eta: Sequence[float]) -> float:
        """Return ``sum_i [log C(n_i, y_i) + y_i log p_i + (n_i-y_i) log(1-p_i)]``.

        Args:
            y: Counts in ``[0, n_i]``.
            eta: Logits.

        Returns:
            The normalized log-likelihood.

        Raises:
            ValueError: For counts outside ``[0, n_i]`` or non-integers.

        References:
            [EQ-INV-OBSLAW-003].

        """
        y_vector, eta_vector = self._counts(y, eta)
        terms: list[float] = []
        for k, n, e in zip(y_vector, self.trials, eta_vector, strict=True):
            log_choose = (
                math.lgamma(n + 1.0) - math.lgamma(k + 1.0) - math.lgamma(n - k + 1.0)
            )
            terms.append(log_choose - k * _softplus(-e) - (n - k) * _softplus(e))
        return math.fsum(terms)

    def score(self, y: Sequence[float], eta: Sequence[float]) -> Vector:
        """Return ``y - n expit(eta)``.

        Args:
            y: Counts.
            eta: Logits.

        Returns:
            The score vector.

        References:
            [EQ-INV-OBSLAW-003].

        """
        y_vector, eta_vector = self._counts(y, eta)
        return tuple(
            k - n * math.exp(-_softplus(-e))
            for k, n, e in zip(y_vector, self.trials, eta_vector, strict=True)
        )

    def fisher_information(self, eta: Sequence[float]) -> Vector:
        """Return ``n p (1 - p)``.

        Args:
            eta: Logits.

        Returns:
            The diagonal information.

        References:
            [EQ-INV-OBSLAW-003].

        """
        eta_vector = _real_vector(eta, "eta", size=self.size)
        return tuple(
            n * math.exp(-_softplus(-e) - _softplus(e))
            for n, e in zip(self.trials, eta_vector, strict=True)
        )

    def simulate(
        self, eta: Sequence[float], seed: int | RandomStream | None = None
    ) -> Vector:
        """Draw independent binomial counts.

        Args:
            eta: Logits.
            seed: Integer seed or stream.

        Returns:
            Counts as floats.

        References:
            [EQ-INV-OBSLAW-003].

        """
        eta_vector = _real_vector(eta, "eta", size=self.size)
        stream = resolve_stream(seed)
        draws: list[float] = []
        for i, (n, e) in enumerate(zip(self.trials, eta_vector, strict=True)):
            p = min(1.0, max(0.0, math.exp(-_softplus(-e))))
            draws.append(float(Binomial(n, p).sample(1, seed=stream.child(i))[0]))
        return tuple(draws)

    def irls_system(
        self, y: Sequence[float], eta: Sequence[float]
    ) -> tuple[LinearOperator, Vector]:
        """Return Fisher-scoring weights and working response.

        Args:
            y: Counts.
            eta: Logits.

        Returns:
            ``(diag(sqrt(I)), z)``.

        References:
            [EQ-INV-OBSLAW-008].

        """
        eta_vector = _real_vector(eta, "eta", size=self.size)
        weights, response = _scoring_response(
            eta_vector, self.score(y, eta), self.fisher_information(eta)
        )
        return _diagonal_whitening(weights), response

    def restrict(self, indices: Sequence[int]) -> BinomialLaw:
        """Return the law of the selected trials.

        Args:
            indices: Strictly increasing indices.

        Returns:
            A :class:`BinomialLaw`.

        References:
            [EQ-INV-OBSLAW-007].

        """
        chosen = _check_indices(indices, self.size)
        return BinomialLaw([self.trials[i] for i in chosen], self.link)


class StudentTLaw(ObservationLaw):
    """Independent Student-t errors ``y_i = eta_i + s_i T_i``, ``T_i ~ t(nu)``.

    This is the PRODUCT of univariate t densities. It is NOT the multivariate
    t law (whose components are dependent through a shared mixing variable)
    and is not obtained from a multivariate t by whitening
    [EQ-INV-OBSLAW-004].

    Args:
        scale: Strictly positive scales ``s_i`` (a scale, not a standard
            deviation: the variance is ``s^2 nu/(nu-2)`` for ``nu > 2`` and
            does not exist for ``nu <= 2``).
        degrees_of_freedom: ``nu > 0``.

    Raises:
        ValueError: If ``nu <= 0`` or a scale is not positive.

    References:
        [EQ-INV-OBSLAW-004], [REF-LANGE-LITTLE-TAYLOR-1989].

    """

    def __init__(self, scale: Sequence[float], degrees_of_freedom: float) -> None:
        """Validate scales and degrees of freedom.

        References:
            [EQ-INV-OBSLAW-004].

        """
        self.scale = _positive_vector(scale, "scale")
        if isinstance(degrees_of_freedom, bool) or not isinstance(
            degrees_of_freedom, (int, float)
        ):
            raise TypeError("degrees_of_freedom must be a real number")
        nu = float(degrees_of_freedom)
        if not math.isfinite(nu) or nu <= 0.0:
            raise ValueError("degrees_of_freedom must be finite and positive")
        self.degrees_of_freedom = nu

    @property
    def size(self) -> int:
        """Return the observation dimension.

        References:
            [EQ-INV-OBSLAW-004].

        """
        return len(self.scale)

    def log_likelihood(self, y: Sequence[float], eta: Sequence[float]) -> float:
        """Return ``sum_i log t_nu((y_i - eta_i)/s_i) - log s_i``.

        Delegates to :class:`probstat.distributions.StudentT`.

        Args:
            y: Observations.
            eta: Predictions.

        Returns:
            The normalized log-likelihood.

        References:
            [EQ-INV-OBSLAW-004].

        """
        y_vector, eta_vector = self._pair(y, eta)
        return math.fsum(
            StudentT(self.degrees_of_freedom, e, s).log_pdf(v)
            for v, e, s in zip(y_vector, eta_vector, self.scale, strict=True)
        )

    def score(self, y: Sequence[float], eta: Sequence[float]) -> Vector:
        """Return ``(nu + 1) r / (nu s^2 + r^2)`` with ``r = y - eta``.

        Args:
            y: Observations.
            eta: Predictions.

        Returns:
            The score vector.

        References:
            [EQ-INV-OBSLAW-004].

        """
        y_vector, eta_vector = self._pair(y, eta)
        nu = self.degrees_of_freedom
        return tuple(
            (nu + 1.0) * (v - e) / (nu * s * s + (v - e) ** 2)
            for v, e, s in zip(y_vector, eta_vector, self.scale, strict=True)
        )

    def fisher_information(self, eta: Sequence[float]) -> Vector:
        """Return ``(nu + 1) / ((nu + 3) s^2)``.

        Args:
            eta: Predictions (only the length is used).

        Returns:
            The diagonal information.

        References:
            [EQ-INV-OBSLAW-004].

        """
        _real_vector(eta, "eta", size=self.size)
        nu = self.degrees_of_freedom
        return tuple((nu + 1.0) / ((nu + 3.0) * s * s) for s in self.scale)

    def simulate(
        self, eta: Sequence[float], seed: int | RandomStream | None = None
    ) -> Vector:
        """Draw independent ``eta_i + s_i T_i``.

        Args:
            eta: Predictions.
            seed: Integer seed or stream.

        Returns:
            One simulated data vector.

        References:
            [EQ-INV-OBSLAW-004].

        """
        eta_vector = _real_vector(eta, "eta", size=self.size)
        stream = resolve_stream(seed)
        return tuple(
            StudentT(self.degrees_of_freedom, e, s).sample(1, seed=stream.child(i))[0]
            for i, (e, s) in enumerate(zip(eta_vector, self.scale, strict=True))
        )

    def irls_weights(self, y: Sequence[float], eta: Sequence[float]) -> Vector:
        """Return ``(nu + 1) / (nu s^2 + r^2)`` with ``r = y - eta``.

        Args:
            y: Observations.
            eta: Predictions.

        Returns:
            The EM / IRLS weights.

        References:
            [EQ-INV-OBSLAW-009], [REF-LANGE-LITTLE-TAYLOR-1989].

        """
        y_vector, eta_vector = self._pair(y, eta)
        nu = self.degrees_of_freedom
        return tuple(
            (nu + 1.0) / (nu * s * s + (v - e) ** 2)
            for v, e, s in zip(y_vector, eta_vector, self.scale, strict=True)
        )

    def irls_system(
        self, y: Sequence[float], eta: Sequence[float]
    ) -> tuple[LinearOperator, Vector]:
        """Return the majorizing weights of [EQ-INV-OBSLAW-009] and ``z = y``.

        The weighted least-squares step minimizes a quadratic majorizer of the
        negative log-likelihood tangent at ``eta``, so it never increases the
        objective; the limit is a local optimum of a non-convex problem.

        Args:
            y: Observations.
            eta: Current predictions.

        Returns:
            ``(diag(sqrt(w)), y)``.

        References:
            [EQ-INV-OBSLAW-009], [REF-LANGE-LITTLE-TAYLOR-1989].

        """
        y_vector, _ = self._pair(y, eta)
        return _diagonal_whitening(self.irls_weights(y, eta)), y_vector

    def restrict(self, indices: Sequence[int]) -> StudentTLaw:
        """Return the law of the selected observations.

        Args:
            indices: Strictly increasing indices.

        Returns:
            A :class:`StudentTLaw`.

        References:
            [EQ-INV-OBSLAW-007].

        """
        chosen = _check_indices(indices, self.size)
        return StudentTLaw([self.scale[i] for i in chosen], self.degrees_of_freedom)


class CensoredGaussianLaw(ObservationLaw):
    """Independent Gaussian errors with left and right detection limits.

    ``y_i`` is the recorded value: an entry equal to (or beyond) its lower
    limit ``L_i`` means "the true value was at most ``L_i``" with likelihood
    ``Phi((L_i - eta_i)/s_i)``; an entry at or above ``U_i`` means "at least
    ``U_i``" with likelihood ``1 - Phi((U_i - eta_i)/s_i)``; any other entry is
    an exact reading with density ``N(y_i; eta_i, s_i^2)``. Censored entries
    are NOT imputed. Tails use the log-CDF and log-survival functions of
    :class:`probstat.distributions.Normal`, finite far beyond ``erfc``
    underflow [EQ-INV-OBSLAW-005].

    Args:
        sd: Strictly positive standard deviations.
        lower: Optional per-entry lower limits (``None`` or ``-inf`` entries
            mean no limit); ``None`` for no left censoring at all.
        upper: Optional per-entry upper limits (``None`` or ``inf`` entries
            mean no limit).

    Raises:
        ValueError: If ``lower >= upper`` somewhere or lengths disagree.

    References:
        [EQ-INV-OBSLAW-005], [EQ-INV-OBSLAW-006], [REF-LANGE-LITTLE-TAYLOR-1989].

    """

    def __init__(
        self,
        sd: Sequence[float],
        lower: Sequence[float | None] | None = None,
        upper: Sequence[float | None] | None = None,
    ) -> None:
        """Validate scales and limits.

        References:
            [EQ-INV-OBSLAW-005].

        """
        self.sd = _positive_vector(sd, "sd")
        m = len(self.sd)
        self.lower = self._limits(lower, m, -math.inf, "lower")
        self.upper = self._limits(upper, m, math.inf, "upper")
        if any(lo >= up for lo, up in zip(self.lower, self.upper, strict=True)):
            raise ValueError("lower limits must be strictly below upper limits")

    @staticmethod
    def _limits(
        values: Sequence[float | None] | None, size: int, default: float, name: str
    ) -> Vector:
        """Return limits with ``None`` replaced by the infinite default.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].

        """
        if values is None:
            return (default,) * size
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
            [EQ-INV-OBSLAW-005].

        """
        return len(self.sd)

    def _classify(self, y: Vector) -> list[int]:
        """Return -1 (left censored), +1 (right censored) or 0 (exact) per entry.

        References:
            [EQ-INV-OBSLAW-005].

        """
        kinds: list[int] = []
        for v, lo, up in zip(y, self.lower, self.upper, strict=True):
            if v < lo or v > up:
                raise ValueError("observation lies beyond its detection limit")
            if v == lo:
                kinds.append(-1)
            elif v == up:
                kinds.append(1)
            else:
                kinds.append(0)
        return kinds

    def log_likelihood(self, y: Sequence[float], eta: Sequence[float]) -> float:
        """Return the censored-Gaussian log-likelihood.

        Args:
            y: Recorded values; entries equal to a limit are censored.
            eta: Predictions.

        Returns:
            ``sum`` of log-density (exact), log-CDF (left) and log-survival
            (right) terms.

        Raises:
            ValueError: If an entry lies beyond its detection limit.

        References:
            [EQ-INV-OBSLAW-005].

        """
        y_vector, eta_vector = self._pair(y, eta)
        kinds = self._classify(y_vector)
        terms: list[float] = []
        for v, e, s, lo, up, kind in zip(
            y_vector, eta_vector, self.sd, self.lower, self.upper, kinds, strict=True
        ):
            law = Normal(e, s)
            if kind < 0:
                terms.append(law.log_cdf(lo))
            elif kind > 0:
                terms.append(law.log_sf(up))
            else:
                terms.append(law.log_pdf(v))
        return math.fsum(terms)

    def score(self, y: Sequence[float], eta: Sequence[float]) -> Vector:
        """Return the score; censored entries use the inverse Mills ratio.

        Args:
            y: Recorded values.
            eta: Predictions.

        Returns:
            ``(y - eta)/s^2`` (exact), ``-phi(a)/(s Phi(a))`` (left,
            ``a = (L - eta)/s``) or ``phi(b)/(s S(b))`` (right,
            ``b = (U - eta)/s``), the ratios computed in log space.

        References:
            [EQ-INV-OBSLAW-005].

        """
        y_vector, eta_vector = self._pair(y, eta)
        kinds = self._classify(y_vector)
        result: list[float] = []
        for v, e, s, lo, up, kind in zip(
            y_vector, eta_vector, self.sd, self.lower, self.upper, kinds, strict=True
        ):
            if kind < 0:
                a = (lo - e) / s
                result.append(
                    -math.exp(_std_log_pdf(a) - _STANDARD_NORMAL.log_cdf(a)) / s
                )
            elif kind > 0:
                b = (up - e) / s
                result.append(
                    math.exp(_std_log_pdf(b) - _STANDARD_NORMAL.log_sf(b)) / s
                )
            else:
                result.append((v - e) / (s * s))
        return tuple(result)

    def fisher_information(self, eta: Sequence[float]) -> Vector:
        """Return the expected information of the censored observation.

        With ``a = (L - eta)/s`` and ``b = (U - eta)/s`` the information times
        ``s^2`` is
        ``P(a<Z<b) - b phi(b) + a phi(a) + phi(a)^2/Phi(a) + phi(b)^2/S(b)``
        (terms with an infinite limit vanish); it equals ``1/s^2`` without
        limits.

        Args:
            eta: Predictions.

        Returns:
            The diagonal information.

        References:
            [EQ-INV-OBSLAW-006].

        """
        eta_vector = _real_vector(eta, "eta", size=self.size)
        result: list[float] = []
        for e, s, lo, up in zip(
            eta_vector, self.sd, self.lower, self.upper, strict=True
        ):
            a = (lo - e) / s if math.isfinite(lo) else -math.inf
            b = (up - e) / s if math.isfinite(up) else math.inf
            if a > 0.0:
                mass = _STANDARD_NORMAL.sf(a) - (
                    _STANDARD_NORMAL.sf(b) if math.isfinite(b) else 0.0
                )
            else:
                mass = (_STANDARD_NORMAL.cdf(b) if math.isfinite(b) else 1.0) - (
                    _STANDARD_NORMAL.cdf(a) if math.isfinite(a) else 0.0
                )
            total = mass
            if math.isfinite(a):
                total += a * math.exp(_std_log_pdf(a))
                total += math.exp(2.0 * _std_log_pdf(a) - _STANDARD_NORMAL.log_cdf(a))
            if math.isfinite(b):
                total -= b * math.exp(_std_log_pdf(b))
                total += math.exp(2.0 * _std_log_pdf(b) - _STANDARD_NORMAL.log_sf(b))
            result.append(max(total, 0.0) / (s * s))
        return tuple(result)

    def simulate(
        self, eta: Sequence[float], seed: int | RandomStream | None = None
    ) -> Vector:
        """Draw ``clip(eta + s Z, L, U)``: censored values are recorded at the limit.

        Args:
            eta: Predictions.
            seed: Integer seed or stream.

        Returns:
            One simulated recorded data vector.

        References:
            [EQ-INV-OBSLAW-005].

        """
        eta_vector = _real_vector(eta, "eta", size=self.size)
        generator = resolve_stream(seed).generator()
        return tuple(
            min(up, max(lo, e + s * generator.gauss(0.0, 1.0)))
            for e, s, lo, up in zip(
                eta_vector, self.sd, self.lower, self.upper, strict=True
            )
        )

    def irls_system(
        self, y: Sequence[float], eta: Sequence[float]
    ) -> tuple[LinearOperator, Vector]:
        """Return Fisher-scoring weights and working response.

        Args:
            y: Recorded values.
            eta: Predictions.

        Returns:
            ``(diag(sqrt(I)), eta + score/I)``.

        References:
            [EQ-INV-OBSLAW-008].

        """
        eta_vector = _real_vector(eta, "eta", size=self.size)
        weights, response = _scoring_response(
            eta_vector, self.score(y, eta), self.fisher_information(eta)
        )
        return _diagonal_whitening(weights), response

    def restrict(self, indices: Sequence[int]) -> CensoredGaussianLaw:
        """Return the law of the selected observations.

        Args:
            indices: Strictly increasing indices.

        Returns:
            A :class:`CensoredGaussianLaw`.

        References:
            [EQ-INV-OBSLAW-007].

        """
        chosen = _check_indices(indices, self.size)
        return CensoredGaussianLaw(
            [self.sd[i] for i in chosen],
            [self.lower[i] for i in chosen],
            [self.upper[i] for i in chosen],
        )


class MissingMask(ObservationLaw):
    """Wrap a law so that masked (missing) observations are ignored explicitly.

    The likelihood is that of the observed entries only, under the
    missing-at-random assumption (the mask does not depend on the unobserved
    values); informative missingness is not modelled. The wrapped law is
    restricted to its observed marginal, so a correlated
    :class:`GaussianLaw` uses the covariance SUBMATRIX of the observed entries
    [EQ-INV-OBSLAW-007]. Entries of ``y`` at masked positions are never read
    (they may be ``nan``); the score and information are zero there, and
    :meth:`simulate` returns ``nan`` there.

    Args:
        law: The full-length law.
        observed: Boolean flags, ``True`` for an observed entry.

    Raises:
        ValueError: If no entry is observed or the lengths disagree.

    References:
        [EQ-INV-OBSLAW-007].

    """

    def __init__(self, law: ObservationLaw, observed: Sequence[bool]) -> None:
        """Build the restricted law.

        References:
            [EQ-INV-OBSLAW-007].

        """
        if not isinstance(law, ObservationLaw):
            raise TypeError("law must be an ObservationLaw")
        flags = tuple(observed)
        if any(not isinstance(flag, bool) for flag in flags):
            raise TypeError("observed must contain booleans")
        if not flags:
            raise ValueError("observed must be non-empty")
        if law.size is not None and law.size != len(flags):
            raise ValueError("mask length must equal the law dimension")
        indices = tuple(i for i, flag in enumerate(flags) if flag)
        if not indices:
            raise ValueError(
                "every observation is masked: the likelihood is empty "
                "(prior-only inference is not handled here)"
            )
        self.law = law
        self.observed = flags
        self.indices = indices
        self.inner = law.restrict(indices)

    @property
    def size(self) -> int:
        """Return the full observation dimension (observed and missing).

        References:
            [EQ-INV-OBSLAW-007].

        """
        return len(self.observed)

    def _select(self, values: Sequence[float], name: str, *, finite: bool) -> Vector:
        """Return the entries at the observed positions.

        References:
            [EQ-INV-OBSLAW-007].

        """
        full = _real_vector(values, name, size=self.size, finite=finite)
        return tuple(full[i] for i in self.indices)

    def log_likelihood(self, y: Sequence[float], eta: Sequence[float]) -> float:
        """Return the log-likelihood of the observed entries.

        Args:
            y: Full-length data; masked entries are ignored.
            eta: Full-length predictions; only observed entries are used.

        Returns:
            The log-likelihood of the observed marginal law.

        References:
            [EQ-INV-OBSLAW-007].

        """
        return self.inner.log_likelihood(
            self._select(y, "y", finite=False), self._select(eta, "eta", finite=False)
        )

    def _scatter(self, values: Vector) -> Vector:
        """Place observed-length ``values`` into a full vector with zeros.

        References:
            [EQ-INV-OBSLAW-007].

        """
        full = [0.0] * self.size
        for position, value in zip(self.indices, values, strict=True):
            full[position] = value
        return tuple(full)

    def score(self, y: Sequence[float], eta: Sequence[float]) -> Vector:
        """Return the score, zero at masked entries.

        Args:
            y: Full-length data.
            eta: Full-length predictions.

        Returns:
            The full-length score.

        References:
            [EQ-INV-OBSLAW-007].

        """
        return self._scatter(
            self.inner.score(
                self._select(y, "y", finite=False),
                self._select(eta, "eta", finite=False),
            )
        )

    def fisher_information(self, eta: Sequence[float]) -> Vector:
        """Return the diagonal information of the observed marginal, zero elsewhere.

        Args:
            eta: Full-length predictions.

        Returns:
            The full-length diagonal information.

        References:
            [EQ-INV-OBSLAW-007].

        """
        return self._scatter(
            self.inner.fisher_information(self._select(eta, "eta", finite=False))
        )

    def simulate(
        self, eta: Sequence[float], seed: int | RandomStream | None = None
    ) -> Vector:
        """Simulate the observed entries; masked entries are ``nan``.

        Args:
            eta: Full-length predictions.
            seed: Integer seed or stream.

        Returns:
            A full-length vector with ``nan`` at masked positions.

        References:
            [EQ-INV-OBSLAW-007].

        """
        drawn = self.inner.simulate(self._select(eta, "eta", finite=False), seed)
        full = [math.nan] * self.size
        for position, value in zip(self.indices, drawn, strict=True):
            full[position] = value
        return tuple(full)

    def irls_system(
        self, y: Sequence[float], eta: Sequence[float]
    ) -> tuple[LinearOperator, Vector]:
        """Return ``(W S, z)`` with ``S`` the selection of observed entries.

        Args:
            y: Full-length data.
            eta: Full-length predictions.

        Returns:
            The whitening of the observed marginal composed with the
            selection, and the working response scattered into full length
            (zeros at masked entries, which ``W S`` never reads).

        References:
            [EQ-INV-OBSLAW-007], [EQ-INV-OBSLAW-008].

        """
        whitening, response = self.inner.irls_system(
            self._select(y, "y", finite=False), self._select(eta, "eta", finite=False)
        )
        selection = SelectionOperator(self.size, self.indices)
        return ComposedOperator([whitening, selection]), self._scatter(response)

    def restrict(self, indices: Sequence[int]) -> ObservationLaw:
        """Restrict to observed positions only.

        Args:
            indices: Strictly increasing indices, all observed.

        Returns:
            The restricted law.

        Raises:
            ValueError: If an index is masked.

        References:
            [EQ-INV-OBSLAW-007].

        """
        chosen = _check_indices(indices, self.size)
        if any(not self.observed[i] for i in chosen):
            raise ValueError("cannot restrict to a masked observation")
        return self.law.restrict(chosen)


@dataclass(frozen=True, slots=True)
class LikelihoodIteration:
    """One accepted Fisher-scoring step.

    Attributes:
        iteration: 1-based step number.
        objective: Penalized negative log-likelihood after the step.
        decrement: ``-g'd`` of the full step before the line search.
        step_length: Accepted backtracking factor (1 is the full step).
        gradient_norm: Euclidean norm of the penalized gradient after the step.
        linear_iterations: LSQR iterations of the weighted solve.

    References:
        [EQ-INV-OBSLAW-008].

    """

    iteration: int
    objective: float
    decrement: float
    step_length: float
    gradient_norm: float
    linear_iterations: int


@dataclass(frozen=True, slots=True)
class MaximumLikelihoodResult:
    """Result of :func:`maximum_likelihood_linear`.

    Attributes:
        estimate: The final iterate ``x``.
        objective: Penalized negative log-likelihood
            ``-log p(y | A x) + lambda^2/2 ||L (x - x_ref)||^2`` (fully
            normalized, so it includes the law's constants).
        log_likelihood: ``log p(y | A x)``.
        gradient_norm: Norm of the penalized gradient at ``x``.
        iterations: Number of accepted steps.
        converged: Whether the decrement tolerance was met.
        stop_reason: ``"decrement_tolerance"``, ``"max_iterations"``,
            ``"line_search_failed"`` or ``"linear_solve_failed"``.
        history: The accepted steps.
        linear_iterations: Total LSQR iterations.

    References:
        [EQ-INV-OBSLAW-008].

    """

    estimate: Vector
    objective: float
    log_likelihood: float
    gradient_norm: float
    iterations: int
    converged: bool
    stop_reason: str
    history: tuple[LikelihoodIteration, ...]
    linear_iterations: int


def _norm(values: Sequence[float]) -> float:
    """Return the Euclidean norm.

    References:
        [REF-GOLUB-VANLOAN-2013].

    """
    return math.sqrt(math.fsum(v * v for v in values))


def maximum_likelihood_linear(
    operator: LinearOperator | Sequence[Sequence[float]],
    data: Sequence[float],
    law: ObservationLaw,
    *,
    regularization: float = 0.0,
    penalty: LinearOperator | Sequence[Sequence[float]] | None = None,
    reference: Sequence[float] | None = None,
    initial: Sequence[float] | None = None,
    max_iterations: int = 50,
    tolerance: float = 1e-12,
) -> MaximumLikelihoodResult:
    """Penalized maximum likelihood / MAP for ``eta = A x`` by Fisher scoring.

    Minimizes ``Phi(x) = -log p(y | A x) + lambda^2/2 ||L (x - x_ref)||^2``.
    Each step solves the weighted least-squares problem
    ``min ||W (z - A x)||^2 + lambda^2 ||L (x - x_ref)||^2`` for the stacked
    whitened system by LSQR, with ``(W, z)`` from
    :meth:`ObservationLaw.irls_system` (Fisher information weights and working
    response ``z = eta + score / information``), then backtracks on the true
    ``Phi`` with an Armijo rule [EQ-INV-OBSLAW-008]. For a Gaussian law one
    step is exact and the result equals the weighted affine Tikhonov solution
    of :func:`probstat.inverse.solve_regularized`. For :class:`StudentTLaw`
    the weights are ``(nu + 1)/(nu s^2 + r^2)`` [EQ-INV-OBSLAW-009]; the
    limit is a LOCAL optimum of a non-convex objective, and different starts
    can give different answers.

    A trial point at which the law is undefined (e.g. a non-positive Poisson
    rate under the identity link) counts as an infinite objective and is
    backtracked away from; the starting point itself must be feasible.

    Args:
        operator: ``A`` (``m x n``).
        data: Observations (masked entries may be ``nan`` under
            :class:`MissingMask`).
        law: The observation law.
        regularization: ``lambda >= 0``.
        penalty: ``L``; identity by default.
        reference: ``x_ref``; zero by default.
        initial: Starting point (``x_ref`` by default); must be feasible.
        max_iterations: Maximum number of steps.
        tolerance: Stop when the decrement ``-g'd`` is at most
            ``tolerance * (1 + |Phi|)``.

    Returns:
        :class:`MaximumLikelihoodResult`.

    Raises:
        TypeError: For wrong argument types.
        ValueError: For inconsistent dimensions, invalid data for the law, an
            infeasible starting point, a negative ``lambda`` or a non-positive
            tolerance or iteration count.

    References:
        [EQ-INV-OBSLAW-008], [EQ-INV-OBSLAW-009], [REF-MCCULLAGH-NELDER-1989],
        [REF-LANGE-LITTLE-TAYLOR-1989], [REF-NOCEDAL-WRIGHT-2006].

    """
    if not isinstance(law, ObservationLaw):
        raise TypeError("law must be an ObservationLaw")
    a = require_operator(operator, name="operator")
    m, n = a.shape
    if isinstance(regularization, bool) or not isinstance(regularization, (int, float)):
        raise TypeError("regularization must be a real number")
    lam = float(regularization)
    if not math.isfinite(lam) or lam < 0.0:
        raise ValueError("regularization must be finite and non-negative")
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations < 1
    ):
        raise ValueError("max_iterations must be a positive integer")
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be positive")
    if law.size is not None and law.size != m:
        raise ValueError("law dimension must equal the number of observations")
    y = _real_vector(data, "data", size=m, finite=False)
    l_operator = IdentityOperator(n) if penalty is None else require_operator(penalty)
    if l_operator.shape[1] != n:
        raise ValueError("penalty must have as many columns as the operator")
    x_ref = (
        (0.0,) * n
        if reference is None
        else _real_vector(reference, "reference", size=n)
    )
    x = x_ref if initial is None else _real_vector(initial, "initial", size=n)

    def penalized(point: Vector) -> tuple[float, float]:
        """Return ``(Phi, log-likelihood)``; ``(inf, -inf)`` if infeasible.

        References:
            [EQ-INV-OBSLAW-008].

        """
        try:
            value = law.log_likelihood(y, a.matvec(point))
        except ValueError:
            return math.inf, -math.inf
        shifted = [p - r for p, r in zip(point, x_ref, strict=True)]
        pen = _norm(l_operator.matvec(shifted))
        return -value + 0.5 * lam * lam * pen * pen, value

    def gradient(point: Vector) -> Vector:
        """Return the gradient of the penalized objective.

        References:
            [EQ-INV-OBSLAW-008].

        """
        s = law.score(y, a.matvec(point))
        data_part = a.rmatvec(s)
        shifted = [p - r for p, r in zip(point, x_ref, strict=True)]
        pen_part = l_operator.rmatvec(l_operator.matvec(shifted))
        return tuple(
            -d + lam * lam * p for d, p in zip(data_part, pen_part, strict=True)
        )

    # The starting point must be feasible: errors for bad data surface here.
    eta0 = a.matvec(x)
    law.log_likelihood(y, eta0)
    objective, loglik = penalized(x)
    history: list[LikelihoodIteration] = []
    total_linear = 0
    stop_reason = "max_iterations"
    converged = False
    for iteration in range(1, max_iterations + 1):
        eta = a.matvec(x)
        whitening, z = law.irls_system(y, eta)
        weighted = ComposedOperator([whitening, a])
        shifted_response = [zi - ai for zi, ai in zip(z, a.matvec(x_ref), strict=True)]
        rhs = whitening.matvec(shifted_response)
        if lam > 0.0:
            stacked: LinearOperator = VStackOperator(
                [weighted, ScaledOperator(l_operator, lam)]
            )
            rhs = tuple(rhs) + (0.0,) * l_operator.shape[0]
        else:
            stacked = weighted
        start = [p - r for p, r in zip(x, x_ref, strict=True)]
        solve = lsqr(
            stacked,
            rhs,
            initial=start,
            atol=1e-13,
            btol=1e-13,
            condition_limit=1e15,
            max_iterations=10 * n + 100,
        )
        total_linear += solve.iterations
        if not solve.converged:
            stop_reason = "linear_solve_failed"
            break
        candidate = tuple(s + r for s, r in zip(solve.solution, x_ref, strict=True))
        direction = [c - p for c, p in zip(candidate, x, strict=True)]
        grad = gradient(x)
        decrement = -math.fsum(g * d for g, d in zip(grad, direction, strict=True))
        if decrement <= tolerance * (1.0 + abs(objective)):
            stop_reason = "decrement_tolerance"
            converged = True
            break
        if decrement < 0.0:
            stop_reason = "line_search_failed"
            break
        step = 1.0
        accepted = False
        for _ in range(60):
            trial = tuple(p + step * d for p, d in zip(x, direction, strict=True))
            trial_objective, trial_loglik = penalized(trial)
            slack = 4.0 * sys.float_info.epsilon * abs(objective)
            if trial_objective <= objective - 1e-4 * step * decrement + slack:
                accepted = True
                break
            step *= 0.5
        if not accepted:
            stop_reason = "line_search_failed"
            break
        x, objective, loglik = trial, trial_objective, trial_loglik
        history.append(
            LikelihoodIteration(
                iteration,
                objective,
                decrement,
                step,
                _norm(gradient(x)),
                solve.iterations,
            )
        )
    return MaximumLikelihoodResult(
        estimate=x,
        objective=objective,
        log_likelihood=loglik,
        gradient_norm=_norm(gradient(x)),
        iterations=len(history),
        converged=converged,
        stop_reason=stop_reason,
        history=tuple(history),
        linear_iterations=total_linear,
    )
