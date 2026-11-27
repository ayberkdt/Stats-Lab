"""Laplace, randomize-then-optimize and Laplace-preconditioned MCMC inversion.

Experimental (added in 0.28).

Setting [EQ-INV-LAPX-001]: ``y = G(x) + eps``, ``eps ~ N(0, Gamma)`` and a
Gaussian prior ``x = mu + P z``, ``z ~ N(0, I_r)`` (``P`` is ``n x r``; the
prior covariance is ``P P'``). With ``W' W = Gamma^-1`` the posterior of the
whitened coordinates ``z`` is ``pi(z) ~ exp(-1/2 ||H(z)||^2)`` with the
stacked residual ``H(z) = [W (G(mu + P z) - y); z]``. Everything below is
matrix-free in ``G`` (``jvp``/``vjp`` of a
:class:`~probstat.inverse.DifferentiableForward`) except randomize-then-
optimize, which forms dense ``r x r`` Jacobians and is meant for moderate
``r``.

Contents:

* :class:`NonlinearBayesProblem` - the validated problem with work counters.
* :func:`compute_map` - the MAP point by Gauss-Newton-Krylov or Newton trust
  region (existing solvers, in whitened coordinates).
* :func:`laplace_approximation` - Laplace posterior with the dominant
  eigenpairs of the prior-preconditioned Gauss-Newton Hessian from
  :func:`~probstat.inverse.matrix_free_lis`, low-rank covariance actions
  [EQ-INV-LAPX-003], samples, exact or probing marginal variances
  [EQ-INV-LAPX-005] and the Laplace log-evidence [EQ-INV-LAPX-004].
* :func:`randomize_then_optimize` - RTO of Bardsley et al. [EQ-INV-LAPX-006]
  with an independence Metropolis-Hastings or importance correction
  [EQ-INV-LAPX-007].
* :func:`randomized_maximum_likelihood` - RML, an *approximate* sampler that is
  exact only for linear Gaussian problems [EQ-INV-LAPX-008].
* :func:`pcn_chain`, :func:`gpcn_chain`, :func:`dili_chain` - plain pCN,
  generalized pCN with the Laplace proposal covariance [EQ-INV-LAPX-009] and a
  DILI-style split of Langevin in the likelihood-informed subspace and pCN in
  its complement [EQ-INV-LAPX-010]; efficiency is measured as effective sample
  size per model evaluation [EQ-INV-LAPX-011].

Common pitfalls:
    - The Laplace approximation uses the Gauss-Newton Hessian, not the full
      Hessian; they agree for linear ``G`` and differ by the residual-weighted
      second-derivative term otherwise. The Laplace posterior is a Gaussian
      centred at the MAP; it cannot represent skewness, banana shapes or
      multimodality. Compare with an MCMC reference before trusting moments.
    - The truncated Laplace log-evidence omits ``1/2 sum log(1 + lambda_i)``
      over the unresolved eigenvalues and so is biased upward.
    - RTO requires that ``z -> Q' H(z)`` is a bijection (Bardsley et al.); the
      Newton solves can fail otherwise and failures are counted, not hidden.
      Proposals whose solve failed are dropped (importance) or rejected
      (Metropolis), which biases the result if the failure rate is not tiny.
    - RML is approximate for nonlinear ``G``; its samples are not distributed
      as the posterior. Never use it as a substitute for MCMC without a check.
    - ``gpcn_chain`` and ``dili_chain`` are valid MCMC for every ``G`` for the
      stated target; the Laplace information only changes efficiency.
    - ESS values come from a single chain (split-chain estimate); they are
      noisy for short chains.

References:
    [REF-KAIPIO-SOMERSALO-2007], [REF-BARDSLEY-SOLONEN-HAARIO-LAINE-2014],
    [REF-CUI-LAW-MARZOUK-2016], [REF-BUI-THANH-ETAL-2013].

"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.inference import effective_sample_size
from probstat.linalg import (
    BlockDiagonalCovariance,
    CallbackOperator,
    DenseCovariance,
    DiagonalCovariance,
    LinearOperator,
    Matrix,
    ScaledIdentityCovariance,
    Vector,
    as_vector,
    householder_qr,
    log_abs_determinant,
    lu_solve,
    require_operator,
)

from .function_space_matrix_free import matrix_free_lis
from .gauss_newton import DifferentiableForward, gauss_newton_krylov
from .newton_trust_region import newton_trust_region_inversion
from .posterior_targets import WhitenedGaussianPriorTarget

__all__ = [
    "ChainResult",
    "EvaluationCounts",
    "LaplaceApproximation",
    "MapEstimate",
    "MarginalVariances",
    "NonlinearBayesProblem",
    "RMLResult",
    "RTOResult",
    "compute_map",
    "dili_chain",
    "gpcn_chain",
    "laplace_approximation",
    "pcn_chain",
    "randomize_then_optimize",
    "randomized_maximum_likelihood",
    "sample_moments",
]

_LOG_TWO_PI = math.log(2.0 * math.pi)
_COVARIANCE_TYPES = (
    DiagonalCovariance,
    ScaledIdentityCovariance,
    DenseCovariance,
    BlockDiagonalCovariance,
)


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the Euclidean inner product.

    Args:
        left: First vector.
        right: Second vector of the same length.

    Returns:
        ``left' right``.

    References:
        [EQ-INV-LAPX-001].

    """
    return math.fsum(a * b for a, b in zip(left, right, strict=True))


def _norm(vector: Sequence[float]) -> float:
    """Return the Euclidean norm.

    Args:
        vector: Vector.

    Returns:
        ``||v||``.

    References:
        [EQ-INV-LAPX-001].

    """
    return math.sqrt(math.fsum(a * a for a in vector))


def _is_finite(vector: Sequence[float]) -> bool:
    """Return whether every entry is finite.

    Args:
        vector: Vector.

    Returns:
        ``True`` if all entries are finite.

    References:
        [EQ-INV-LAPX-001].

    """
    return all(math.isfinite(v) for v in vector)


def _positive_int(value: object, name: str) -> int:
    """Return a strictly positive integer.

    Args:
        value: Candidate.
        name: Argument name.

    Returns:
        The integer.

    Raises:
        TypeError: If ``value`` is not an integer.
        ValueError: If ``value < 1``.

    References:
        [EQ-INV-LAPX-001].

    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be at least 1")
    return value


def _non_negative_int(value: object, name: str) -> int:
    """Return a non-negative integer.

    Args:
        value: Candidate.
        name: Argument name.

    Returns:
        The integer.

    Raises:
        TypeError: If ``value`` is not an integer.
        ValueError: If ``value < 0``.

    References:
        [EQ-INV-LAPX-001].

    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _positive_real(value: object, name: str) -> float:
    """Return a strictly positive finite float.

    Args:
        value: Candidate.
        name: Argument name.

    Returns:
        The float.

    Raises:
        TypeError: If ``value`` is not a real number.
        ValueError: If ``value`` is not positive and finite.

    References:
        [EQ-INV-LAPX-001].

    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not (math.isfinite(number) and number > 0.0):
        raise ValueError(f"{name} must be positive and finite")
    return number


def _standard_normal(generator: random.Random, size: int) -> Vector:
    """Return a standard normal vector.

    Args:
        generator: Random generator.
        size: Length.

    Returns:
        ``xi ~ N(0, I_size)``.

    References:
        [EQ-INV-LAPX-001].

    """
    return tuple(generator.gauss(0.0, 1.0) for _ in range(size))


def sample_moments(
    samples: Sequence[Sequence[float]], weights: Sequence[float] | None = None
) -> tuple[Vector, Vector]:
    """Return the componentwise (weighted) sample mean and variance.

    Unweighted draws use the divisor ``N - 1``; normalized weights ``w`` use
    the reliability-weighted variance ``sum w (x - mean)^2 / (1 - sum w^2)``.

    Args:
        samples: At least two vectors of equal length.
        weights: Optional non-negative weights (normalized internally).

    Returns:
        ``(mean, variance)``.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If fewer than two draws are given, lengths disagree, the
            weights are negative, non-finite or sum to zero, or all the weight
            sits on one draw.

    References:
        [EQ-INV-LAPX-007].

    """
    draws = [as_vector(s, name="sample") for s in samples]
    count = len(draws)
    if count < 2:
        raise ValueError("at least two samples are required")
    size = len(draws[0])
    if any(len(d) != size for d in draws):
        raise ValueError("samples must have equal length")
    if weights is None:
        w = [1.0 / count] * count
        denominator = 1.0 - 1.0 / count
    else:
        raw = as_vector(weights, name="weights")
        if len(raw) != count or any(v < 0.0 for v in raw):
            raise ValueError("weights must be non-negative, one per sample")
        total = math.fsum(raw)
        if total <= 0.0:
            raise ValueError("weights must have a positive sum")
        w = [v / total for v in raw]
        denominator = 1.0 - math.fsum(v * v for v in w)
        if denominator <= 0.0:
            raise ValueError("all weight sits on a single sample")
    mean = tuple(
        math.fsum(w[k] * draws[k][i] for k in range(count)) for i in range(size)
    )
    variance = tuple(
        math.fsum(w[k] * (draws[k][i] - mean[i]) ** 2 for k in range(count))
        / denominator
        for i in range(size)
    )
    return mean, variance


# --------------------------------------------------------------------------
# problem
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class EvaluationCounts:
    """Number of forward-model calls made through a problem.

    Attributes:
        value: Calls of ``G``.
        jvp: Calls of ``J(x) v``.
        vjp: Calls of ``J(x)' w``.

    References:
        [EQ-INV-LAPX-011].

    """

    value: int
    jvp: int
    vjp: int

    @property
    def work(self) -> int:
        """Return ``value + jvp + vjp``, the cost unit of the ESS measure.

        Returns:
            Total number of model calls (one PDE-solve-like unit each).

        References:
            [EQ-INV-LAPX-011].

        """
        return self.value + self.jvp + self.vjp

    def minus(self, other: EvaluationCounts) -> EvaluationCounts:
        """Return the componentwise difference ``self - other``.

        Args:
            other: Earlier snapshot.

        Returns:
            The calls made between the two snapshots.

        References:
            [EQ-INV-LAPX-011].

        """
        return EvaluationCounts(
            self.value - other.value, self.jvp - other.jvp, self.vjp - other.vjp
        )


class _CountedModel:
    """Forward model wrapper that counts its calls.

    References:
        [EQ-INV-LAPX-011].

    """

    __slots__ = ("calls", "model")

    def __init__(self, model: DifferentiableForward) -> None:
        """Wrap ``model``.

        Args:
            model: Differentiable forward map.

        References:
            [EQ-INV-LAPX-011].

        """
        self.model = model
        self.calls = [0, 0, 0]

    @property
    def input_dimension(self) -> int:
        """Return ``n``.

        References:
            [EQ-INV-LAPX-001].

        """
        return self.model.input_dimension

    @property
    def output_dimension(self) -> int:
        """Return ``m``.

        References:
            [EQ-INV-LAPX-001].

        """
        return self.model.output_dimension

    def value(self, x: Sequence[float]) -> Vector:
        """Return ``G(x)`` and count the call.

        Args:
            x: Parameters.

        Returns:
            The model output.

        References:
            [EQ-INV-LAPX-001].

        """
        self.calls[0] += 1
        return tuple(float(v) for v in self.model.value(x))

    def jvp(self, x: Sequence[float], v: Sequence[float]) -> Vector:
        """Return ``J(x) v`` and count the call.

        Args:
            x: Parameters.
            v: Direction.

        Returns:
            The Jacobian action.

        References:
            [EQ-INV-LAPX-001].

        """
        self.calls[1] += 1
        return tuple(float(t) for t in self.model.jvp(x, v))

    def vjp(self, x: Sequence[float], w: Sequence[float]) -> Vector:
        """Return ``J(x)' w`` and count the call.

        Args:
            x: Parameters.
            w: Output-space vector.

        Returns:
            The adjoint action.

        References:
            [EQ-INV-LAPX-001].

        """
        self.calls[2] += 1
        return tuple(float(t) for t in self.model.vjp(x, w))


class _WhitenedForward:
    """The forward map ``z -> G(mu + P z)`` as a differentiable model.

    References:
        [EQ-INV-LAPX-001].

    """

    __slots__ = ("_problem",)

    def __init__(self, problem: NonlinearBayesProblem) -> None:
        """Bind to a problem.

        Args:
            problem: The Bayesian problem.

        References:
            [EQ-INV-LAPX-001].

        """
        self._problem = problem

    @property
    def input_dimension(self) -> int:
        """Return ``r``.

        References:
            [EQ-INV-LAPX-001].

        """
        return self._problem.dimension

    @property
    def output_dimension(self) -> int:
        """Return ``m``.

        References:
            [EQ-INV-LAPX-001].

        """
        return self._problem.data_dimension

    def value(self, x: Sequence[float]) -> Vector:
        """Return ``G(mu + P z)``.

        Args:
            x: Whitened coordinates ``z``.

        Returns:
            The model output.

        References:
            [EQ-INV-LAPX-001].

        """
        return self._problem._model.value(self._problem.to_physical(x))

    def jvp(self, x: Sequence[float], v: Sequence[float]) -> Vector:
        """Return ``J(mu + P z) P v``.

        Args:
            x: Whitened coordinates ``z``.
            v: Whitened direction.

        Returns:
            The Jacobian action in whitened coordinates.

        References:
            [EQ-INV-LAPX-001].

        """
        problem = self._problem
        direction = problem._factor.matvec(tuple(v))
        return problem._model.jvp(problem.to_physical(x), direction)

    def vjp(self, x: Sequence[float], w: Sequence[float]) -> Vector:
        """Return ``P' J(mu + P z)' w``.

        Args:
            x: Whitened coordinates ``z``.
            w: Output-space vector.

        Returns:
            The adjoint action in whitened coordinates.

        References:
            [EQ-INV-LAPX-001].

        """
        problem = self._problem
        pulled = problem._model.vjp(problem.to_physical(x), tuple(w))
        return tuple(problem._factor.rmatvec(pulled))


class NonlinearBayesProblem:
    """Nonlinear Bayesian inverse problem with a Gaussian prior and noise.

    Posterior of ``y = G(x) + eps`` in prior-whitened coordinates
    ``x = mu + P z`` [EQ-INV-LAPX-001]. The object counts every call of ``G``,
    ``J v`` and ``J' w`` (:attr:`counts`) and caches the last forward value, so
    evaluating the likelihood and then its gradient at the same ``z`` costs one
    forward solve and one adjoint. It is stateful and not thread safe.

    References:
        [EQ-INV-LAPX-001], [REF-KAIPIO-SOMERSALO-2007],
        [REF-COTTER-ROBERTS-STUART-WHITE-2013].

    """

    __slots__ = (
        "_cache",
        "_data",
        "_factor",
        "_mean",
        "_model",
        "_noise",
        "log_likelihood_constant",
    )

    def __init__(
        self,
        model: DifferentiableForward,
        data: Sequence[float],
        *,
        noise: object,
        prior_factor: object,
        prior_mean: Sequence[float] | None = None,
    ) -> None:
        """Validate and store the problem.

        Args:
            model: Differentiable forward map ``G`` (``value``, ``jvp`` and
                ``vjp`` are used; check them with
                :func:`~probstat.inverse.check_derivatives`).
            data: Observation ``y`` of length ``m``.
            noise: ``probstat.linalg`` covariance factor of ``Gamma``
                (dimension ``m``).
            prior_factor: ``P`` (``n x r``) with prior covariance ``P P'``: a
                covariance factor, a linear operator or a dense matrix.
            prior_mean: ``mu`` of length ``n`` (default zeros).

        Raises:
            TypeError: If ``model`` or ``noise`` has the wrong type.
            ValueError: If dimensions disagree or an input is not finite.

        References:
            [EQ-INV-LAPX-001].

        """
        if not isinstance(model, DifferentiableForward):
            raise TypeError("model must implement DifferentiableForward")
        n = model.input_dimension
        m = model.output_dimension
        y = as_vector(data, name="data")
        if len(y) != m:
            raise ValueError(f"data has length {len(y)}, expected {m}")
        if not isinstance(noise, _COVARIANCE_TYPES):
            raise TypeError("noise must be a probstat.linalg covariance factor")
        if noise.dimension != m:
            raise ValueError(f"noise has dimension {noise.dimension}, expected {m}")
        if prior_mean is None:
            mu: Vector = (0.0,) * n
        else:
            mu = as_vector(prior_mean, name="prior_mean")
            if len(mu) != n:
                raise ValueError(f"prior_mean has length {len(mu)}, expected {n}")
        factor: LinearOperator
        if isinstance(prior_factor, _COVARIANCE_TYPES):
            factor = prior_factor.factor_operator()
        else:
            factor = require_operator(prior_factor, name="prior_factor")
        if factor.shape[0] != n:
            raise ValueError(f"prior_factor has {factor.shape[0]} rows, expected {n}")
        if factor.shape[1] < 1:
            raise ValueError("prior_factor must have at least one column")
        self._model = _CountedModel(model)
        self._data = y
        self._noise: Any = noise
        self._mean = mu
        self._factor = factor
        self._cache: tuple[Vector, Vector, Vector] | None = None
        self.log_likelihood_constant = -0.5 * (m * _LOG_TWO_PI + noise.log_det())

    @property
    def dimension(self) -> int:
        """Return ``r``, the number of whitened coordinates.

        References:
            [EQ-INV-LAPX-001].

        """
        return self._factor.shape[1]

    @property
    def physical_dimension(self) -> int:
        """Return ``n``, the length of ``x``.

        References:
            [EQ-INV-LAPX-001].

        """
        return self._factor.shape[0]

    @property
    def data_dimension(self) -> int:
        """Return ``m``, the length of ``y``.

        References:
            [EQ-INV-LAPX-001].

        """
        return len(self._data)

    @property
    def data(self) -> Vector:
        """Return the observation ``y``.

        References:
            [EQ-INV-LAPX-001].

        """
        return self._data

    @property
    def noise(self) -> Any:
        """Return the noise covariance factor.

        References:
            [EQ-INV-LAPX-001].

        """
        return self._noise

    @property
    def prior_factor(self) -> LinearOperator:
        """Return ``P``.

        References:
            [EQ-INV-LAPX-001].

        """
        return self._factor

    @property
    def prior_mean(self) -> Vector:
        """Return ``mu``.

        References:
            [EQ-INV-LAPX-001].

        """
        return self._mean

    @property
    def counts(self) -> EvaluationCounts:
        """Return a snapshot of the model-call counters.

        References:
            [EQ-INV-LAPX-011].

        """
        value, jvp, vjp = self._model.calls
        return EvaluationCounts(value, jvp, vjp)

    def _check_z(self, z: Sequence[float]) -> Vector:
        """Return ``z`` as a finite vector of length ``r``.

        Raises:
            TypeError: If ``z`` is not a real sequence.
            ValueError: If the length is wrong or an entry is not finite.

        References:
            [EQ-INV-LAPX-001].

        """
        point = as_vector(z, name="z")
        if len(point) != self.dimension:
            raise ValueError(f"z has length {len(point)}, expected {self.dimension}")
        return point

    def to_physical(self, z: Sequence[float]) -> Vector:
        """Return ``x = mu + P z``.

        Args:
            z: Whitened coordinates of length ``r``.

        Returns:
            Physical parameters of length ``n``.

        Raises:
            TypeError: If ``z`` is not a real sequence.
            ValueError: If the length is wrong or an entry is not finite.

        References:
            [EQ-INV-LAPX-001].

        """
        shift = self._factor.matvec(self._check_z(z))
        return tuple(a + b for a, b in zip(self._mean, shift, strict=True))

    def _forward(self, z: Vector) -> tuple[Vector, Vector]:
        """Return ``(x, G(x))`` using a one-entry cache.

        References:
            [EQ-INV-LAPX-001].

        """
        cache = self._cache
        if cache is not None and cache[0] == z:
            return cache[1], cache[2]
        x = self.to_physical(z)
        value = self._model.value(x)
        self._cache = (z, x, value)
        return x, value

    def whitened_residual(self, z: Sequence[float]) -> Vector | None:
        """Return ``W (G(mu + P z) - y)``, or ``None`` if ``G`` is not finite.

        Args:
            z: Whitened coordinates.

        Returns:
            The whitened data residual or ``None`` for a failed solve.

        Raises:
            TypeError: If ``z`` is not a real sequence.
            ValueError: If the length is wrong or an entry is not finite.

        References:
            [EQ-INV-LAPX-001].

        """
        point = self._check_z(z)
        _, value = self._forward(point)
        if not _is_finite(value):
            return None
        residual = tuple(g - d for g, d in zip(value, self._data, strict=True))
        whitened = tuple(self._noise.whiten(residual))
        return whitened if _is_finite(whitened) else None

    def log_likelihood(self, z: Sequence[float]) -> float:
        """Return ``-1/2 ||W (G(mu + P z) - y)||^2`` (no normalizing constant).

        Args:
            z: Whitened coordinates.

        Returns:
            The whitened half misfit (``<= 0``) or ``-inf`` for a failed or
            overflowing forward solve.

        Raises:
            TypeError: If ``z`` is not a real sequence.
            ValueError: If the length is wrong or an entry is not finite.

        References:
            [EQ-INV-LAPX-001].

        """
        residual = self.whitened_residual(z)
        if residual is None:
            return -math.inf
        try:
            return -0.5 * math.fsum(e * e for e in residual)
        except OverflowError:
            return -math.inf

    def log_posterior(self, z: Sequence[float]) -> float:
        """Return ``log_likelihood(z) - 1/2 ||z||^2`` (unnormalized).

        Args:
            z: Whitened coordinates.

        Returns:
            The log posterior density up to an additive constant.

        Raises:
            TypeError: If ``z`` is not a real sequence.
            ValueError: If the length is wrong or an entry is not finite.

        References:
            [EQ-INV-LAPX-001].

        """
        point = self._check_z(z)
        return self.log_likelihood(point) - 0.5 * math.fsum(v * v for v in point)

    def likelihood_and_gradient(
        self, z: Sequence[float]
    ) -> tuple[float, Vector | None]:
        """Return the log likelihood and its gradient in ``z``.

        The gradient is ``-P' J' W' W (G(x) - y)``; the posterior gradient is
        this minus ``z``. One forward solve (shared with the likelihood) and
        one adjoint.

        Args:
            z: Whitened coordinates.

        Returns:
            ``(log_likelihood, gradient)``; the gradient is ``None`` if the
            forward value or the adjoint is not finite.

        Raises:
            TypeError: If ``z`` is not a real sequence.
            ValueError: If the length is wrong or an entry is not finite.

        References:
            [EQ-INV-LAPX-001].

        """
        point = self._check_z(z)
        residual = self.whitened_residual(point)
        if residual is None:
            return -math.inf, None
        try:
            ll = -0.5 * math.fsum(e * e for e in residual)
        except OverflowError:
            return -math.inf, None
        x, _ = self._forward(point)
        weights = self._noise.whiten_transpose(residual)
        adjoint = self._model.vjp(x, tuple(weights))
        if not _is_finite(adjoint):
            return ll, None
        pulled = self._factor.rmatvec(adjoint)
        return ll, tuple(-p for p in pulled)

    def jacobian_operator(self, z: Sequence[float]) -> LinearOperator:
        """Return ``J(mu + P z)`` (``m x n``) as a matrix-free operator.

        Args:
            z: Whitened coordinates.

        Returns:
            A callback operator built from ``jvp`` and ``vjp``.

        Raises:
            TypeError: If ``z`` is not a real sequence.
            ValueError: If the length is wrong or an entry is not finite.

        References:
            [EQ-INV-LAPX-002].

        """
        x = self.to_physical(z)
        model = self._model

        def forward(v: Vector) -> Vector:
            """Apply ``J(x)``.

            References:
                [EQ-INV-LAPX-002].

            """
            return model.jvp(x, v)

        def adjoint(w: Vector) -> Vector:
            """Apply ``J(x)'``.

            References:
                [EQ-INV-LAPX-002].

            """
            return model.vjp(x, w)

        return CallbackOperator(
            (self.data_dimension, self.physical_dimension),
            forward,
            adjoint,
            name="forward Jacobian",
        )

    def whitened_jacobian_dense(self, z: Sequence[float]) -> Matrix:
        """Return the dense ``W J(x) P`` (``m x r``) by ``r`` JVPs.

        Args:
            z: Whitened coordinates.

        Returns:
            Rows of the whitened Jacobian.

        Raises:
            TypeError: If ``z`` is not a real sequence.
            ValueError: If the length is wrong or an entry is not finite.

        References:
            [EQ-INV-LAPX-006].

        """
        point = self._check_z(z)
        x = self.to_physical(point)
        r = self.dimension
        columns: list[Vector] = []
        for k in range(r):
            unit = tuple(1.0 if i == k else 0.0 for i in range(r))
            direction = self._factor.matvec(unit)
            columns.append(tuple(self._noise.whiten(self._model.jvp(x, direction))))
        return tuple(
            tuple(columns[k][i] for k in range(r)) for i in range(self.data_dimension)
        )

    def whitened_forward(self) -> DifferentiableForward:
        """Return ``z -> G(mu + P z)`` as a differentiable forward map.

        Returns:
            An object implementing
            :class:`~probstat.inverse.DifferentiableForward` whose calls are
            counted in :attr:`counts`.

        References:
            [EQ-INV-LAPX-002].

        """
        return _WhitenedForward(self)

    def as_target(self) -> WhitenedGaussianPriorTarget:
        """Return the same posterior as a ``WhitenedGaussianPriorTarget``.

        This lets the existing pCN/HMC/NUTS adapters of
        :mod:`~probstat.inverse.posterior_targets` sample the problem; the
        wrapped model still counts its calls here.

        Returns:
            The whitened target (the noise must be a diagonal, dense or block
            covariance factor).

        Raises:
            TypeError: If the noise type is not accepted by the target.

        References:
            [EQ-INV-LAPX-001].

        """
        return WhitenedGaussianPriorTarget(
            self._model,
            self._data,
            noise=self._noise,
            prior_mean=self._mean,
            prior_factor=self._factor,
        )


def _require_problem(problem: object) -> NonlinearBayesProblem:
    """Return ``problem`` after a type check.

    Raises:
        TypeError: If it is not a :class:`NonlinearBayesProblem`.

    References:
        [EQ-INV-LAPX-001].

    """
    if not isinstance(problem, NonlinearBayesProblem):
        raise TypeError("problem must be a NonlinearBayesProblem")
    return problem


# --------------------------------------------------------------------------
# MAP
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class MapEstimate:
    """Maximum a posteriori point.

    Attributes:
        whitened: ``z`` at the MAP.
        physical: ``x = mu + P z``.
        objective: ``1/2 ||W (G(x) - y)||^2 + 1/2 ||z||^2``.
        gradient_norm: Norm of its gradient in ``z`` (reported by the solver).
        iterations: Accepted iterations.
        converged: Whether the solver reached its gradient criterion.
        stop_reason: The solver's stop reason.
        method: ``"gauss_newton"`` or ``"trust_region"``.
        counts: Model calls spent.

    References:
        [EQ-INV-LAPX-002].

    """

    whitened: Vector
    physical: Vector
    objective: float
    gradient_norm: float
    iterations: int
    converged: bool
    stop_reason: str
    method: str
    counts: EvaluationCounts


def compute_map(
    problem: NonlinearBayesProblem,
    *,
    method: Literal["gauss_newton", "trust_region"] = "gauss_newton",
    initial: Sequence[float] | None = None,
    max_iterations: int = 100,
    gradient_tolerance: float = 1e-9,
    inner_tolerance: float = 1e-10,
) -> MapEstimate:
    """Return the MAP point ``argmin 1/2 ||H(z)||^2``.

    The objective is the whitened least-squares functional of
    [EQ-INV-LAPX-002]; it is minimized with the existing
    :func:`~probstat.inverse.gauss_newton_krylov` (``lambda = 1``, ``L = I``) or
    :func:`~probstat.inverse.newton_trust_region_inversion` in whitened
    coordinates. Non-convergence is reported in the result, never hidden. A
    nonlinear least-squares objective can have several local minima; the
    result is a local one.

    Args:
        problem: The problem.
        method: ``"gauss_newton"`` or ``"trust_region"``.
        initial: Start ``z_0`` (default zeros, the prior mean).
        max_iterations: Iteration cap (``>= 1``).
        gradient_tolerance: Relative gradient tolerance (``>= 0``).
        inner_tolerance: Inner Krylov tolerance of Gauss-Newton (``> 0``).

    Returns:
        :class:`MapEstimate`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an option is invalid or ``G`` is not finite at ``initial``.

    References:
        [EQ-INV-LAPX-002], [REF-NOCEDAL-WRIGHT-2006].

    """
    problem = _require_problem(problem)
    if method not in ("gauss_newton", "trust_region"):
        raise ValueError("method must be 'gauss_newton' or 'trust_region'")
    before = problem.counts
    forward = problem.whitened_forward()
    start = None if initial is None else problem._check_z(initial)
    if method == "gauss_newton":
        gn = gauss_newton_krylov(
            forward,
            problem.data,
            noise=problem.noise,
            regularization=1.0,
            initial=start,
            max_iterations=max_iterations,
            gradient_tolerance=gradient_tolerance,
            inner_tolerance=inner_tolerance,
        )
        z, objective, gradient_norm = gn.estimate, gn.objective, gn.gradient_norm
        iterations, converged, reason = gn.iterations, gn.converged, gn.stop_reason
    else:
        tr = newton_trust_region_inversion(
            forward,
            problem.data,
            noise=problem.noise,
            regularization=1.0,
            initial=start,
            max_iterations=max_iterations,
            gradient_tolerance=gradient_tolerance,
        )
        z, objective, gradient_norm = tr.estimate, tr.objective, tr.gradient_norm
        iterations, converged, reason = tr.iterations, tr.converged, tr.stop_reason
    return MapEstimate(
        whitened=tuple(z),
        physical=problem.to_physical(z),
        objective=objective,
        gradient_norm=gradient_norm,
        iterations=iterations,
        converged=converged,
        stop_reason=reason,
        method=method,
        counts=problem.counts.minus(before),
    )


# --------------------------------------------------------------------------
# Laplace approximation
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class MarginalVariances:
    """Marginal posterior variances of the physical parameters.

    Attributes:
        values: Variance estimate per parameter.
        standard_errors: Monte Carlo standard error of each entry for the
            probing method; zeros for the exact method.
        method: ``"exact"`` (``n`` applications of ``P'``) or ``"probing"``.
        probes: Number of Rademacher probes (``0`` for exact).

    References:
        [EQ-INV-LAPX-005].

    """

    values: Vector
    standard_errors: Vector
    method: str
    probes: int


@dataclass(frozen=True, slots=True)
class LaplaceApproximation:
    """Laplace posterior ``N(x_map, P Sigma_z P')`` with a low-rank ``Sigma_z``.

    ``Sigma_z = (I + V diag(lambda) V')^-1 = I - V diag(lambda/(1+lambda)) V'``
    for the retained eigenpairs ``(lambda_i, v_i)`` of the prior-
    preconditioned Gauss-Newton Hessian at the MAP [EQ-INV-LAPX-003]. It is
    exact for a linear ``G`` when every direction with ``lambda > 0`` is kept.

    Attributes:
        problem: The problem.
        map_estimate: The MAP point.
        eigenvalues: Retained eigenvalues ``lambda_i`` (descending).
        eigenvectors: Retained unit eigenvectors ``v_i`` (rows of length ``r``).
        residual_norms: Eigenpair residuals ``||H v_i - lambda_i v_i||``.
        rank: Number of retained pairs.
        threshold: Eigenvalue threshold used to choose the rank.
        lis_converged: Whether the eigensolver decided the rank.
        lis_stop_reason: Its stop reason.
        converged: ``map_estimate.converged and lis_converged``.
        log_evidence: Laplace estimate of ``log p(y)`` [EQ-INV-LAPX-004].
        log_evidence_truncation_bound: Upper bound
            ``1/2 (r - rank) log(1 + threshold)`` of the omitted
            ``1/2 sum log(1 + lambda_i)`` if the eigensolver bracketed the
            threshold (assuming the computed eigenvalues are the leading ones;
            not certified), ``0`` if the whole spectrum was resolved
            (``"full_dimension"`` or ``"invariant_subspace"``), else ``nan``.
            The evidence is overestimated by at most this amount.
        misfit: ``1/2 ||W (G(x_map) - y)||^2``.
        counts_map: Model calls spent on the MAP.
        counts_hessian: Model calls spent on the eigenproblem.

    References:
        [EQ-INV-LAPX-003], [EQ-INV-LAPX-004], [REF-BUI-THANH-ETAL-2013].

    """

    problem: NonlinearBayesProblem
    map_estimate: MapEstimate
    eigenvalues: Vector
    eigenvectors: Matrix
    residual_norms: Vector
    rank: int
    threshold: float
    lis_converged: bool
    lis_stop_reason: str
    converged: bool
    log_evidence: float
    log_evidence_truncation_bound: float
    misfit: float
    counts_map: EvaluationCounts
    counts_hessian: EvaluationCounts

    @property
    def mean(self) -> Vector:
        """Return the Laplace mean, the MAP point in physical coordinates.

        References:
            [EQ-INV-LAPX-003].

        """
        return self.map_estimate.physical

    def whitened_precision_action(self, vector: Sequence[float]) -> Vector:
        """Return ``(I + V diag(lambda) V') v`` (the whitened posterior precision).

        Args:
            vector: Vector of length ``r``.

        Returns:
            The precision action.

        Raises:
            TypeError: If ``vector`` is not real.
            ValueError: If the length is not ``r``.

        References:
            [EQ-INV-LAPX-003].

        """
        v = self.problem._check_z(vector)
        out = list(v)
        for lam, basis in zip(self.eigenvalues, self.eigenvectors, strict=True):
            coefficient = lam * _dot(basis, v)
            for i in range(len(out)):
                out[i] += coefficient * basis[i]
        return tuple(out)

    def whitened_covariance_action(self, vector: Sequence[float]) -> Vector:
        """Return ``Sigma_z v = v - V diag(lambda/(1+lambda)) V' v``.

        Args:
            vector: Vector of length ``r``.

        Returns:
            The covariance action.

        Raises:
            TypeError: If ``vector`` is not real.
            ValueError: If the length is not ``r``.

        References:
            [EQ-INV-LAPX-003].

        """
        v = self.problem._check_z(vector)
        out = list(v)
        for lam, basis in zip(self.eigenvalues, self.eigenvectors, strict=True):
            coefficient = lam / (1.0 + lam) * _dot(basis, v)
            for i in range(len(out)):
                out[i] -= coefficient * basis[i]
        return tuple(out)

    def covariance_action(self, vector: Sequence[float]) -> Vector:
        """Return ``P Sigma_z P' u``, the physical posterior covariance action.

        Args:
            vector: Vector of length ``n``.

        Returns:
            The covariance action.

        Raises:
            TypeError: If ``vector`` is not real.
            ValueError: If the length is not ``n``.

        References:
            [EQ-INV-LAPX-003].

        """
        u = as_vector(vector, name="vector")
        if len(u) != self.problem.physical_dimension:
            raise ValueError("vector must have length n")
        factor = self.problem.prior_factor
        return tuple(factor.matvec(self.whitened_covariance_action(factor.rmatvec(u))))

    def sample(
        self,
        count: int,
        *,
        seed: int | RandomStream | None = 0,
        whitened: bool = False,
    ) -> tuple[Vector, ...]:
        """Return Laplace samples ``z = z_map + Sigma_z^(1/2) xi``.

        With the eigendecomposition the symmetric square root is
        ``Sigma_z^(1/2) = I - V diag(1 - (1+lambda)^(-1/2)) V'``
        [EQ-INV-LAPX-003]; no ``r x r`` matrix is formed.

        Args:
            count: Number of draws (``>= 1``).
            seed: Seed or stream.
            whitened: Return ``z`` instead of ``x = mu + P z``.

        Returns:
            ``count`` samples.

        Raises:
            TypeError: If ``count`` is not an integer.
            ValueError: If ``count < 1``.

        References:
            [EQ-INV-LAPX-003].

        """
        draws = _positive_int(count, "count")
        generator = resolve_stream(seed).generator()
        z_map = self.map_estimate.whitened
        out: list[Vector] = []
        for _ in range(draws):
            xi = _standard_normal(generator, len(z_map))
            shifted = list(xi)
            for lam, basis in zip(self.eigenvalues, self.eigenvectors, strict=True):
                coefficient = (1.0 - (1.0 + lam) ** -0.5) * _dot(basis, xi)
                for i in range(len(shifted)):
                    shifted[i] -= coefficient * basis[i]
            z = tuple(a + b for a, b in zip(z_map, shifted, strict=True))
            out.append(z if whitened else self.problem.to_physical(z))
        return tuple(out)

    def marginal_variances(
        self, *, probes: int | None = None, seed: int | RandomStream | None = 0
    ) -> MarginalVariances:
        """Return the marginal variances ``diag(P Sigma_z P')`` [EQ-INV-LAPX-005].

        ``probes=None`` is exact: ``diag(P P')_i = ||P' e_i||^2`` minus
        ``sum_k lambda_k/(1+lambda_k) (P v_k)_i^2`` (``n`` adjoint
        applications of ``P``). ``probes=K`` is the Bekas-Kokiopoulou-Saad
        estimator ``mean_j s_j * (C s_j)`` with Rademacher ``s_j`` and the
        low-rank covariance action ``C``; it is unbiased with per-entry
        standard error reported from the probe sample.

        Args:
            probes: ``None`` for exact, otherwise the probe count (``>= 2``).
            seed: Seed or stream (probing only).

        Returns:
            :class:`MarginalVariances`.

        Raises:
            TypeError: If ``probes`` is not an integer.
            ValueError: If ``probes < 2``.

        References:
            [EQ-INV-LAPX-005], [REF-BEKAS-KOKIOPOULOU-SAAD-2007].

        """
        n = self.problem.physical_dimension
        factor = self.problem.prior_factor
        if probes is None:
            diagonal = []
            for i in range(n):
                unit = tuple(1.0 if j == i else 0.0 for j in range(n))
                diagonal.append(math.fsum(t * t for t in factor.rmatvec(unit)))
            for lam, basis in zip(self.eigenvalues, self.eigenvectors, strict=True):
                direction = factor.matvec(basis)
                weight = lam / (1.0 + lam)
                for i in range(n):
                    diagonal[i] -= weight * direction[i] ** 2
            return MarginalVariances(tuple(diagonal), (0.0,) * n, "exact", 0)
        count = _positive_int(probes, "probes")
        if count < 2:
            raise ValueError("probes must be at least 2")
        generator = resolve_stream(seed).generator()
        total = [0.0] * n
        square = [0.0] * n
        for _ in range(count):
            s = tuple(1.0 if generator.random() < 0.5 else -1.0 for _ in range(n))
            action = self.covariance_action(s)
            for i in range(n):
                term = s[i] * action[i]
                total[i] += term
                square[i] += term * term
        mean = [t / count for t in total]
        standard = tuple(
            math.sqrt(max(square[i] / count - mean[i] ** 2, 0.0) / (count - 1))
            for i in range(n)
        )
        return MarginalVariances(tuple(mean), standard, "probing", count)


def laplace_approximation(
    problem: NonlinearBayesProblem,
    *,
    threshold: float = 1e-2,
    max_rank: int | None = None,
    map_estimate: MapEstimate | None = None,
    map_method: Literal["gauss_newton", "trust_region"] = "gauss_newton",
    initial: Sequence[float] | None = None,
    lis_method: Literal["lanczos", "randomized"] = "randomized",
    residual_tolerance: float = 1e-8,
    seed: int | RandomStream | None = 0,
    max_iterations: int = 100,
    gradient_tolerance: float = 1e-9,
) -> LaplaceApproximation:
    """Return the Laplace approximation at the MAP.

    1. The MAP ``z_map`` is found by :func:`compute_map` (or passed in).
    2. The eigenpairs of ``H = P' J' Gamma^-1 J P`` at ``x_map`` above
       ``threshold`` come from :func:`~probstat.inverse.matrix_free_lis` with the
       matrix-free Jacobian (Lanczos or randomized; no ``r x r`` matrix)
       [EQ-INV-LAPX-002].
    3. The posterior is ``N(z_map, (I + H)^-1)`` truncated to the retained
       eigenpairs [EQ-INV-LAPX-003] and the evidence is
       ``log p(y) ~ c - 1/2 ||W(G(x_map)-y)||^2 - 1/2 ||z_map||^2
       - 1/2 sum log(1 + lambda_i)`` [EQ-INV-LAPX-004] with
       ``c = -1/2 (m log 2 pi + log det Gamma)``.

    Args:
        problem: The problem.
        threshold: Eigenvalue threshold of the retained subspace (``> 0``).
        max_rank: Optional cap on the retained rank.
        map_estimate: Precomputed MAP (skips the optimization).
        map_method: Solver for the MAP.
        initial: Start for the MAP solver.
        lis_method: ``"randomized"`` or ``"lanczos"`` eigensolver.
        residual_tolerance: Eigenpair residual tolerance relative to the
            largest eigenvalue.
        seed: Seed or stream of the eigensolver.
        max_iterations: MAP iteration cap.
        gradient_tolerance: MAP gradient tolerance.

    Returns:
        :class:`LaplaceApproximation`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an option is invalid or the forward map is not finite at
            the MAP.

    References:
        [EQ-INV-LAPX-002], [EQ-INV-LAPX-003], [EQ-INV-LAPX-004],
        [REF-BUI-THANH-ETAL-2013], [REF-VILLA-PETRA-GHATTAS-2021].

    """
    problem = _require_problem(problem)
    if map_estimate is None:
        map_estimate = compute_map(
            problem,
            method=map_method,
            initial=initial,
            max_iterations=max_iterations,
            gradient_tolerance=gradient_tolerance,
        )
    elif not isinstance(map_estimate, MapEstimate):
        raise TypeError("map_estimate must be a MapEstimate")
    z_map = problem._check_z(map_estimate.whitened)
    residual = problem.whitened_residual(z_map)
    if residual is None:
        raise ValueError("forward map is not finite at the MAP point")
    misfit = 0.5 * math.fsum(e * e for e in residual)
    before = problem.counts
    lis = matrix_free_lis(
        [problem.jacobian_operator(z_map)],
        noise=problem.noise,
        prior_factor=problem.prior_factor,
        threshold=threshold,
        method=lis_method,
        max_rank=max_rank,
        residual_tolerance=residual_tolerance,
        seed=seed,
    )
    counts_hessian = problem.counts.minus(before)
    rank = lis.rank
    eigenvalues = tuple(lis.eigenvalues[:rank])
    eigenvectors = tuple(tuple(v) for v in lis.basis)
    residual_norms = tuple(lis.residual_norms[:rank])
    log_det_term = 0.5 * math.fsum(math.log1p(lam) for lam in eigenvalues)
    log_evidence = (
        problem.log_likelihood_constant
        - misfit
        - 0.5 * math.fsum(v * v for v in z_map)
        - log_det_term
    )
    omitted = problem.dimension - rank
    if lis.stop_reason in ("full_dimension", "invariant_subspace"):
        bound = 0.0
    elif lis.bracketed:
        bound = 0.5 * omitted * math.log1p(threshold)
    else:
        bound = math.nan
    return LaplaceApproximation(
        problem=problem,
        map_estimate=map_estimate,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        residual_norms=residual_norms,
        rank=rank,
        threshold=float(threshold),
        lis_converged=lis.converged,
        lis_stop_reason=lis.stop_reason,
        converged=map_estimate.converged and lis.converged,
        log_evidence=log_evidence,
        log_evidence_truncation_bound=bound,
        misfit=misfit,
        counts_map=map_estimate.counts,
        counts_hessian=counts_hessian,
    )


# --------------------------------------------------------------------------
# randomize-then-optimize
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class RTOResult:
    """Output of :func:`randomize_then_optimize`.

    Attributes:
        samples: Physical samples ``x`` (one per retained draw).
        whitened_samples: The same in whitened coordinates ``z``.
        weights: Normalized weights of the draws (uniform for ``"metropolis"``
            and ``"none"``; importance weights for ``"importance"``).
        log_weights: Unnormalized log importance weight
            ``log pi(z)/q(z)`` of every successful proposal in proposal order.
        correction: ``"metropolis"``, ``"importance"`` or ``"none"``.
        acceptance_rate: Metropolis acceptance fraction among proposals (``nan``
            otherwise).
        weight_ess: Kish effective sample size ``1/sum w^2`` of the importance
            weights of all successful proposals (the efficiency of the
            proposal; equals the number of proposals when ``q = pi``).
        proposals: Number of attempted proposals.
        solver_failures: Proposals whose optimization failed.
        newton_iterations: Total Newton iterations over all solves.
        stop_reason: ``"completed"`` or ``"no_valid_proposal"``.
        reference: The linearization point ``z_ref``.
        counts: Model calls spent.

    References:
        [EQ-INV-LAPX-006], [EQ-INV-LAPX-007].

    """

    samples: tuple[Vector, ...]
    whitened_samples: tuple[Vector, ...]
    weights: Vector
    log_weights: Vector
    correction: str
    acceptance_rate: float
    weight_ess: float
    proposals: int
    solver_failures: int
    newton_iterations: int
    stop_reason: str
    reference: Vector
    counts: EvaluationCounts


def randomize_then_optimize(
    problem: NonlinearBayesProblem,
    *,
    draws: int,
    reference: Sequence[float] | None = None,
    correction: Literal["metropolis", "importance", "none"] = "metropolis",
    seed: int | RandomStream | None = 0,
    solver_tolerance: float = 1e-10,
    max_newton_iterations: int = 50,
    map_options: dict[str, object] | None = None,
) -> RTOResult:
    """Sample the posterior by randomize-then-optimize (Bardsley et al., 2014).

    With ``H(z) = [W (G(mu + P z) - y); z]`` the posterior is
    ``pi(z) ~ exp(-1/2 ||H(z)||^2)``. At a reference point (default the MAP)
    take the thin QR ``H'(z_ref) = Q R`` and, for ``v ~ N(0, I_(m+r))``, solve
    the *perturbed optimization* ``z* = argmin_z ||Q'(H(z) - v)||^2``, i.e.
    ``h(z*) = xi`` with ``h = Q' H`` and ``xi = Q'v ~ N(0, I_r)``
    [EQ-INV-LAPX-006]. The proposal density is
    ``q(z) ~ |det h'(z)| exp(-1/2 ||h(z)||^2)``, so the weight is
    ``pi/q ~ exp(-1/2 ||(I - Q Q') H(z)||^2) / |det Q' H'(z)|``
    [EQ-INV-LAPX-007]; this is used in an independence Metropolis-Hastings
    chain (``"metropolis"``, exact target), as importance weights
    (``"importance"``) or not at all (``"none"``, approximate). For a linear
    ``G`` the weight is constant and the proposal equals the posterior.

    Each solve is a damped Newton iteration with a dense ``r x r`` Jacobian
    (``r`` JVPs per iteration) and is intended for moderate ``r``. A failed
    solve (singular Jacobian, no descent, non-finite ``G``) is counted in
    ``solver_failures`` and the proposal is dropped (importance) or rejected
    (Metropolis, chain stays).

    Args:
        problem: The problem.
        draws: Number of proposals (``>= 1``); the chain length for
            ``"metropolis"``.
        reference: ``z_ref`` (default the MAP from :func:`compute_map`).
        correction: ``"metropolis"``, ``"importance"`` or ``"none"``.
        seed: Seed or stream.
        solver_tolerance: Newton tolerance ``||h - xi|| <= tol (1 + ||xi||)``.
        max_newton_iterations: Newton cap per proposal.
        map_options: Keyword options for :func:`compute_map` when
            ``reference`` is not given.

    Returns:
        :class:`RTOResult`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an option is invalid or ``H'(z_ref)`` is rank deficient.

    References:
        [EQ-INV-LAPX-006], [EQ-INV-LAPX-007],
        [REF-BARDSLEY-SOLONEN-HAARIO-LAINE-2014].

    """
    problem = _require_problem(problem)
    count = _positive_int(draws, "draws")
    if correction not in ("metropolis", "importance", "none"):
        raise ValueError("correction must be 'metropolis', 'importance' or 'none'")
    tol = _positive_real(solver_tolerance, "solver_tolerance")
    newton_cap = _positive_int(max_newton_iterations, "max_newton_iterations")
    before = problem.counts
    if reference is None:
        z_ref = compute_map(problem, **(map_options or {})).whitened  # type: ignore[arg-type]
    else:
        z_ref = problem._check_z(reference)
    r = problem.dimension
    m = problem.data_dimension
    base = problem.whitened_jacobian_dense(z_ref)
    stacked = tuple(base) + tuple(
        tuple(1.0 if i == j else 0.0 for j in range(r)) for i in range(r)
    )
    decomposition = householder_qr(stacked)
    if decomposition.rank_deficient:
        raise ValueError("stacked Jacobian at the reference is rank deficient")
    q1 = decomposition.q[:m]
    q2 = decomposition.q[m:]

    def project(z: Vector) -> tuple[Vector, float] | None:
        """Return ``(Q' H(z), ||H(z)||^2)`` or ``None`` if ``G`` failed.

        References:
            [EQ-INV-LAPX-006].

        """
        res = problem.whitened_residual(z)
        if res is None:
            return None
        h = tuple(
            math.fsum(q1[i][j] * res[i] for i in range(m))
            + math.fsum(q2[i][j] * z[i] for i in range(r))
            for j in range(r)
        )
        total = math.fsum(e * e for e in res) + math.fsum(v * v for v in z)
        return h, total

    def jacobian(z: Vector) -> Matrix:
        """Return the dense ``Q' H'(z)`` (``r x r``).

        References:
            [EQ-INV-LAPX-006].

        """
        jw = problem.whitened_jacobian_dense(z)
        return tuple(
            tuple(
                math.fsum(q1[i][j] * jw[i][k] for i in range(m)) + q2[k][j]
                for k in range(r)
            )
            for j in range(r)
        )

    iterations = [0]

    def solve(xi: Vector) -> tuple[Vector, float] | None:
        """Solve ``h(z) = xi`` and return ``(z, log weight)`` or ``None``.

        References:
            [EQ-INV-LAPX-006], [EQ-INV-LAPX-007].

        """
        z = z_ref
        state = project(z)
        if state is None:
            return None
        scale = 1.0 + _norm(xi)
        for _ in range(newton_cap + 1):
            h, total = state
            gap = tuple(a - b for a, b in zip(h, xi, strict=True))
            jac = jacobian(z)
            if _norm(gap) <= tol * scale:
                try:
                    log_det = log_abs_determinant(jac)
                except ValueError:
                    return None
                if not math.isfinite(log_det):
                    return None
                perpendicular = total - math.fsum(v * v for v in h)
                return z, -0.5 * perpendicular - log_det
            try:
                step = lu_solve(jac, tuple(-g for g in gap))
            except ValueError:
                return None
            iterations[0] += 1
            old = math.fsum(g * g for g in gap)
            t = 1.0
            for _back in range(40):
                trial = tuple(a + t * b for a, b in zip(z, step, strict=True))
                moved = project(trial)
                if moved is not None:
                    new_gap = math.fsum(
                        (a - b) ** 2 for a, b in zip(moved[0], xi, strict=True)
                    )
                    if new_gap <= (1.0 - 2e-4 * t) * old:
                        z, state = trial, moved
                        break
                t *= 0.5
            else:
                return None
        return None

    generator = resolve_stream(seed).generator()
    accepted_states: list[tuple[Vector, float]] = []
    chain: list[tuple[Vector, float]] = []
    current: tuple[Vector, float] | None = None
    failures = 0
    accepts = 0
    decisions = 0
    for _ in range(count):
        proposal = solve(_standard_normal(generator, r))
        if proposal is None:
            failures += 1
        else:
            accepted_states.append(proposal)
            if correction == "metropolis":
                if current is None:
                    current = proposal
                else:
                    decisions += 1
                    log_ratio = proposal[1] - current[1]
                    if log_ratio >= 0.0 or generator.random() < math.exp(log_ratio):
                        current = proposal
                        accepts += 1
        if correction == "metropolis" and current is not None:
            chain.append(current)
    log_weights = tuple(s[1] for s in accepted_states)
    if correction == "metropolis":
        kept = chain
        weights: Vector = (1.0 / len(kept),) * len(kept) if kept else ()
    elif correction == "importance":
        kept = accepted_states
        top = max(log_weights) if log_weights else 0.0
        raw = [math.exp(lw - top) for lw in log_weights]
        total_weight = math.fsum(raw)
        weights = tuple(w / total_weight for w in raw) if raw else ()
    else:
        kept = accepted_states
        weights = (1.0 / len(kept),) * len(kept) if kept else ()
    if log_weights:
        top = max(log_weights)
        raw_all = [math.exp(lw - top) for lw in log_weights]
        weight_ess = math.fsum(raw_all) ** 2 / math.fsum(w * w for w in raw_all)
    else:
        weight_ess = 0.0
    return RTOResult(
        samples=tuple(problem.to_physical(s[0]) for s in kept),
        whitened_samples=tuple(s[0] for s in kept),
        weights=weights,
        log_weights=log_weights,
        correction=correction,
        acceptance_rate=(
            accepts / decisions
            if correction == "metropolis" and decisions
            else math.nan
        ),
        weight_ess=weight_ess,
        proposals=count,
        solver_failures=failures,
        newton_iterations=iterations[0],
        stop_reason="completed" if kept else "no_valid_proposal",
        reference=z_ref,
        counts=problem.counts.minus(before),
    )


# --------------------------------------------------------------------------
# randomized maximum likelihood
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class RMLResult:
    """Output of :func:`randomized_maximum_likelihood` (approximate sampler).

    Attributes:
        samples: Physical samples of converged optimizations.
        whitened_samples: The same in whitened coordinates.
        requested: Number of optimizations attempted.
        failures: Optimizations that did not converge (excluded).
        approximate: Always ``True``: for nonlinear ``G`` the samples are not
            distributed as the posterior.
        caveat: One-sentence statement of that limitation.
        counts: Model calls spent.

    References:
        [EQ-INV-LAPX-008].

    """

    samples: tuple[Vector, ...]
    whitened_samples: tuple[Vector, ...]
    requested: int
    failures: int
    approximate: bool
    caveat: str
    counts: EvaluationCounts


def randomized_maximum_likelihood(
    problem: NonlinearBayesProblem,
    *,
    draws: int,
    reference: Sequence[float] | None = None,
    seed: int | RandomStream | None = 0,
    max_iterations: int = 100,
    gradient_tolerance: float = 1e-9,
) -> RMLResult:
    """Sample approximately by randomized maximum likelihood (RML).

    Each draw minimizes the objective with perturbed data and prior mean,
    ``z = argmin 1/2 ||W (G(mu + P z) - y - eps)||^2 + 1/2 ||z - zeta||^2``,
    ``eps ~ N(0, Gamma)``, ``zeta ~ N(0, I)`` [EQ-INV-LAPX-008]. It is exact
    for a linear ``G`` (the optimum is an affine map of ``(eps, zeta)``) and
    **approximate** for a nonlinear ``G``: no Jacobian correction is applied,
    so the law of the draws differs from the posterior. Every optimization
    starts at ``reference`` (default the MAP); non-converged ones are dropped
    and counted.

    Args:
        problem: The problem.
        draws: Number of optimizations (``>= 1``).
        reference: Start point (default the MAP).
        seed: Seed or stream.
        max_iterations: Gauss-Newton cap per draw.
        gradient_tolerance: Gauss-Newton gradient tolerance.

    Returns:
        :class:`RMLResult` with ``approximate = True``.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an option is invalid.

    References:
        [EQ-INV-LAPX-008], [REF-KITANIDIS-1995],
        [REF-BARDSLEY-SOLONEN-HAARIO-LAINE-2014].

    """
    problem = _require_problem(problem)
    count = _positive_int(draws, "draws")
    before = problem.counts
    start = (
        compute_map(problem).whitened
        if reference is None
        else problem._check_z(reference)
    )
    generator = resolve_stream(seed).generator()
    forward = problem.whitened_forward()
    m = problem.data_dimension
    r = problem.dimension
    kept: list[Vector] = []
    failures = 0
    for _ in range(count):
        noise_draw = problem.noise.unwhiten(_standard_normal(generator, m))
        data = tuple(a + b for a, b in zip(problem.data, noise_draw, strict=True))
        prior_draw = _standard_normal(generator, r)
        try:
            result = gauss_newton_krylov(
                forward,
                data,
                noise=problem.noise,
                regularization=1.0,
                reference=prior_draw,
                initial=start,
                max_iterations=max_iterations,
                gradient_tolerance=gradient_tolerance,
                inner_tolerance=1e-10,
            )
        except ValueError:
            failures += 1
            continue
        if result.converged:
            kept.append(tuple(result.estimate))
        else:
            failures += 1
    return RMLResult(
        samples=tuple(problem.to_physical(z) for z in kept),
        whitened_samples=tuple(kept),
        requested=count,
        failures=failures,
        approximate=True,
        caveat=(
            "RML is exact only for linear Gaussian problems; for nonlinear G "
            "the draws are not distributed as the posterior"
        ),
        counts=problem.counts.minus(before),
    )


# --------------------------------------------------------------------------
# Laplace-preconditioned MCMC
# --------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ChainResult:
    """Output of :func:`pcn_chain`, :func:`gpcn_chain` and :func:`dili_chain`.

    Attributes:
        method: ``"pcn"``, ``"gpcn"`` or ``"dili"``.
        draws: Number of post-burn-in iterations.
        burn_in: Number of adaptation/burn-in iterations (discarded).
        acceptance_rate: Post-burn-in acceptance fraction.
        beta: Final pCN parameter (complement parameter for DILI).
        step: Final Langevin step ``h`` of DILI (``nan`` otherwise).
        lis_rank: Dimension of the Langevin subspace (DILI) or of the Laplace
            covariance correction (gpCN); ``0`` for pCN.
        mean: Running mean of the physical parameters over the draws.
        variance: Running variance (divisor ``draws - 1``) of the physical
            parameters.
        samples: Thinned physical samples (empty if ``store`` was false).
        log_likelihoods: Thinned trace of ``log_likelihood``.
        ess: Split-chain ESS of the log-likelihood and of each quantity.
        min_ess: Smallest finite ESS (``nan`` if none).
        sampling_counts: Model calls during the post-burn-in phase.
        total_counts: Model calls including burn-in.
        ess_per_forward: ``min_ess / sampling_counts.value``.
        ess_per_gradient: ``min_ess / sampling_counts.vjp`` (``nan`` without
            gradients).
        ess_per_work: ``min_ess / (value + vjp)`` calls (the common cost unit
            of a forward solve plus an adjoint solve when both are needed).
        nonfinite_rejections: Proposals rejected for a failed forward or
            adjoint evaluation.

    References:
        [EQ-INV-LAPX-009], [EQ-INV-LAPX-010], [EQ-INV-LAPX-011].

    """

    method: str
    draws: int
    burn_in: int
    acceptance_rate: float
    beta: float
    step: float
    lis_rank: int
    mean: Vector
    variance: Vector
    samples: tuple[Vector, ...]
    log_likelihoods: Vector
    ess: Vector
    min_ess: float
    sampling_counts: EvaluationCounts
    total_counts: EvaluationCounts
    ess_per_forward: float
    ess_per_gradient: float
    ess_per_work: float
    nonfinite_rejections: int


def _coordinate(index: int) -> Callable[[Vector], float]:
    """Return the function ``x -> x[index]``.

    Args:
        index: Component index.

    Returns:
        The coordinate function.

    References:
        [EQ-INV-LAPX-011].

    """

    def pick(x: Vector) -> float:
        """Return the component.

        References:
            [EQ-INV-LAPX-011].

        """
        return x[index]

    return pick


def _sigmoid(value: float) -> float:
    """Return the logistic function.

    Args:
        value: Argument.

    Returns:
        ``1 / (1 + exp(-value))``.

    References:
        [EQ-INV-LAPX-009].

    """
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    e = math.exp(value)
    return e / (1.0 + e)


def _run_chain(
    problem: NonlinearBayesProblem,
    laplace: LaplaceApproximation | None,
    method: str,
    *,
    draws: int,
    burn_in: int,
    beta: float,
    step: float,
    adapt: bool,
    target_acceptance: float | None,
    initial: Sequence[float] | None,
    seed: int | RandomStream | None,
    thin: int,
    store: bool,
    quantities: Sequence[Callable[[Vector], float]] | None,
    complement_beta: float | None,
    rank: int | None,
    max_beta: float,
) -> ChainResult:
    """Run one of the three chains.

    References:
        [EQ-INV-LAPX-009], [EQ-INV-LAPX-010], [EQ-INV-LAPX-011].

    """
    problem = _require_problem(problem)
    n_draws = _positive_int(draws, "draws")
    n_burn = _non_negative_int(burn_in, "burn_in")
    stride = _positive_int(thin, "thin")
    r = problem.dimension
    if not 0.0 < max_beta <= 1.0:
        raise ValueError("max_beta must lie in (0, 1]")
    if not 0.0 < beta <= max_beta:
        raise ValueError("beta must lie in (0, max_beta]")
    _positive_real(step, "step")
    if complement_beta is not None and not 0.0 < complement_beta <= 1.0:
        raise ValueError("complement_beta must lie in (0, 1]")
    if method != "pcn":
        if not isinstance(laplace, LaplaceApproximation):
            raise TypeError("laplace must be a LaplaceApproximation")
        if laplace.problem is not problem:
            raise ValueError("laplace was computed for a different problem")
    lam: tuple[float, ...] = ()
    basis: Matrix = ()
    z_map: Vector = (0.0,) * r
    if laplace is not None and method != "pcn":
        k = (
            laplace.rank
            if rank is None
            else min(_positive_int(rank, "rank"), laplace.rank)
        )
        lam = laplace.eigenvalues[:k]
        basis = laplace.eigenvectors[:k]
        z_map = laplace.map_estimate.whitened
    generator = resolve_stream(seed).generator()
    z = problem._check_z(initial) if initial is not None else z_map
    ll, grad_ll = problem.likelihood_and_gradient(z)
    if not math.isfinite(ll):
        raise ValueError("the initial state has a non-finite likelihood")
    if method == "dili" and grad_ll is None:
        raise ValueError("the initial state has a non-finite gradient")
    goal = (
        target_acceptance
        if target_acceptance is not None
        else (0.574 if method == "dili" else 0.3)
    )
    if not 0.0 < goal < 1.0:
        raise ValueError("target_acceptance must lie in (0, 1)")
    if method == "dili":
        theta = math.log(step)
    else:
        fraction = beta / max_beta
        theta = math.log(fraction / (1.0 - fraction)) if fraction < 1.0 else 30.0
    n = problem.physical_dimension
    indices = sorted({0, n // 2, n - 1})
    funcs: list[Callable[[Vector], float]]
    if quantities is None:
        funcs = [_coordinate(i) for i in indices]
    else:
        funcs = list(quantities)
    start_counts = problem.counts
    mid_counts = start_counts
    accepted_post = 0
    nonfinite = 0
    mean = [0.0] * n
    m2 = [0.0] * n
    seen = 0
    samples: list[Vector] = []
    ll_trace: list[float] = []
    traces: list[list[float]] = [[] for _ in funcs]
    k_basis = len(basis)
    inv_precision = tuple(1.0 / (1.0 + v) for v in lam)

    def lis_coordinates(vector: Vector) -> list[float]:
        """Return ``V v`` (coordinates in the retained subspace).

        References:
            [EQ-INV-LAPX-010].

        """
        return [_dot(b, vector) for b in basis]

    def complement(vector: Vector, coords: list[float]) -> list[float]:
        """Return ``v - V' V v`` given the coordinates ``V v``.

        References:
            [EQ-INV-LAPX-010].

        """
        out = list(vector)
        for c, b in zip(coords, basis, strict=True):
            for i in range(r):
                out[i] -= c * b[i]
        return out

    def quadratic(vector: Vector) -> float:
        """Return ``||d||^2 + sum lambda_i (v_i'd)^2`` for ``d = z - z_map``.

        References:
            [EQ-INV-LAPX-009].

        """
        d = tuple(a - b for a, b in zip(vector, z_map, strict=True))
        return math.fsum(t * t for t in d) + math.fsum(
            la * _dot(b, d) ** 2 for la, b in zip(lam, basis, strict=True)
        )

    total_iterations = n_burn + n_draws
    for t in range(total_iterations):
        if t == n_burn:
            mid_counts = problem.counts
        burning = t < n_burn
        xi = _standard_normal(generator, r)
        proposal_gradient: Vector | None = None
        if method == "dili":
            h = math.exp(theta)
            rho_c = (
                math.sqrt(1.0 - complement_beta**2)
                if complement_beta is not None
                else (1.0 - h / 4.0) / (1.0 + h / 4.0)
            )
            beta_c = (
                complement_beta
                if complement_beta is not None
                else math.sqrt(max(1.0 - rho_c * rho_c, 0.0))
            )
            assert grad_ll is not None
            u = lis_coordinates(z)
            zperp = complement(z, u)
            grad_post = tuple(g - a for g, a in zip(grad_ll, z, strict=True))
            g_coords = lis_coordinates(grad_post)
            mean_u = [
                ui + 0.5 * h * inv_precision[i] * g_coords[i] for i, ui in enumerate(u)
            ]
            u_new = [
                mean_u[i] + math.sqrt(h * inv_precision[i]) * generator.gauss(0.0, 1.0)
                for i in range(k_basis)
            ]
            xi_perp = complement(xi, lis_coordinates(xi))
            z_prop_list = [rho_c * zperp[i] + beta_c * xi_perp[i] for i in range(r)]
            for c, b in zip(u_new, basis, strict=True):
                for i in range(r):
                    z_prop_list[i] += c * b[i]
            z_prop = tuple(z_prop_list)
            ll_prop, proposal_gradient = problem.likelihood_and_gradient(z_prop)
            if not math.isfinite(ll_prop) or proposal_gradient is None:
                log_alpha = -math.inf
                nonfinite += 1
            else:
                grad_post_new = tuple(
                    g - a for g, a in zip(proposal_gradient, z_prop, strict=True)
                )
                g_new = lis_coordinates(grad_post_new)
                forward_mean = [
                    u_new[i] + 0.5 * h * inv_precision[i] * g_new[i]
                    for i in range(k_basis)
                ]
                log_q_forward = -math.fsum(
                    (u_new[i] - mean_u[i]) ** 2 / inv_precision[i]
                    for i in range(k_basis)
                ) / (2.0 * h)
                log_q_reverse = -math.fsum(
                    (u[i] - forward_mean[i]) ** 2 / inv_precision[i]
                    for i in range(k_basis)
                ) / (2.0 * h)
                log_alpha = (
                    ll_prop
                    - ll
                    - 0.5 * math.fsum(c * c for c in u_new)
                    + 0.5 * math.fsum(c * c for c in u)
                    + log_q_reverse
                    - log_q_forward
                )
        else:
            current_beta = max_beta * _sigmoid(theta)
            rho = math.sqrt(1.0 - current_beta * current_beta)
            if method == "pcn":
                z_prop = tuple(
                    rho * a + current_beta * b for a, b in zip(z, xi, strict=True)
                )
            else:
                coords = lis_coordinates(xi)
                shaped = list(xi)
                for c, b, v in zip(coords, basis, lam, strict=True):
                    scale = (1.0 - (1.0 + v) ** -0.5) * c
                    for i in range(r):
                        shaped[i] -= scale * b[i]
                z_prop = tuple(
                    z_map[i] + rho * (z[i] - z_map[i]) + current_beta * shaped[i]
                    for i in range(r)
                )
            ll_prop = problem.log_likelihood(z_prop)
            if not math.isfinite(ll_prop):
                log_alpha = -math.inf
                nonfinite += 1
            elif method == "pcn":
                log_alpha = ll_prop - ll
            else:
                log_alpha = (
                    ll_prop
                    - ll
                    - 0.5 * math.fsum(a * a for a in z_prop)
                    + 0.5 * math.fsum(a * a for a in z)
                    + 0.5 * quadratic(z_prop)
                    - 0.5 * quadratic(z)
                )
        alpha = 1.0 if log_alpha >= 0.0 else math.exp(log_alpha)
        accept = generator.random() < alpha
        if accept:
            z = z_prop
            ll = ll_prop
            if method == "dili":
                grad_ll = proposal_gradient
        if burning:
            if adapt:
                gain = 1.0 / (t + 1.0) ** 0.6
                theta += gain * (alpha - goal)
        else:
            accepted_post += 1 if accept else 0
            x = problem.to_physical(z)
            seen += 1
            for i in range(n):
                delta = x[i] - mean[i]
                mean[i] += delta / seen
                m2[i] += delta * (x[i] - mean[i])
            if (seen - 1) % stride == 0:
                ll_trace.append(ll)
                for trace, func in zip(traces, funcs, strict=True):
                    trace.append(float(func(x)))
                if store:
                    samples.append(x)
    end_counts = problem.counts
    sampling = end_counts.minus(mid_counts)
    ess_values: list[float] = []
    for series in [ll_trace, *traces]:
        if len(series) >= 4:
            ess_values.append(effective_sample_size([series]))
        else:
            ess_values.append(math.nan)
    finite = [e for e in ess_values if math.isfinite(e)]
    min_ess = min(finite) if finite else math.nan
    final_beta = (
        max_beta * _sigmoid(theta)
        if method != "dili"
        else (complement_beta if complement_beta is not None else math.nan)
    )
    final_step = math.exp(theta) if method == "dili" else math.nan
    if method == "dili" and complement_beta is None:
        h_final = math.exp(theta)
        rho_final = (1.0 - h_final / 4.0) / (1.0 + h_final / 4.0)
        final_beta = math.sqrt(max(1.0 - rho_final * rho_final, 0.0))
    variance = tuple(v / (seen - 1) if seen > 1 else math.nan for v in m2)
    return ChainResult(
        method=method,
        draws=n_draws,
        burn_in=n_burn,
        acceptance_rate=accepted_post / n_draws,
        beta=final_beta,
        step=final_step,
        lis_rank=k_basis,
        mean=tuple(mean),
        variance=variance,
        samples=tuple(samples),
        log_likelihoods=tuple(ll_trace),
        ess=tuple(ess_values),
        min_ess=min_ess,
        sampling_counts=sampling,
        total_counts=end_counts.minus(start_counts),
        ess_per_forward=min_ess / sampling.value if sampling.value else math.nan,
        ess_per_gradient=min_ess / sampling.vjp if sampling.vjp else math.nan,
        ess_per_work=(
            min_ess / (sampling.value + sampling.vjp)
            if sampling.value + sampling.vjp
            else math.nan
        ),
        nonfinite_rejections=nonfinite,
    )


def pcn_chain(
    problem: NonlinearBayesProblem,
    *,
    draws: int,
    burn_in: int = 0,
    beta: float = 0.2,
    adapt: bool = True,
    target_acceptance: float | None = None,
    initial: Sequence[float] | None = None,
    seed: int | RandomStream | None = 0,
    thin: int = 1,
    store: bool = True,
    quantities: Sequence[Callable[[Vector], float]] | None = None,
    max_beta: float = 0.99,
) -> ChainResult:
    """Run plain preconditioned Crank-Nicolson on the whitened posterior.

    ``z' = sqrt(1 - beta^2) z + beta xi`` is reversible for the prior
    ``N(0, I)`` and is accepted with probability
    ``min(1, exp(ell(z') - ell(z)))`` (``ell`` the log likelihood)
    [EQ-INV-LAPX-009]; the acceptance rate does not degrade with the
    discretization dimension for a fixed function-space problem. With
    ``adapt`` the logit of ``beta`` follows a Robbins-Monro rule toward
    ``target_acceptance`` (default 0.3) during burn-in and is then frozen.
    This is the baseline for :func:`gpcn_chain` and :func:`dili_chain`.

    Args:
        problem: The problem.
        draws: Post-burn-in iterations (``>= 1``).
        burn_in: Burn-in iterations (``>= 0``), adapted and discarded.
        beta: Initial step ``beta`` in ``(0, 1]``.
        adapt: Adapt ``beta`` during burn-in.
        target_acceptance: Target acceptance in ``(0, 1)``.
        initial: Start ``z`` (default zeros).
        seed: Seed or stream.
        thin: Keep every ``thin``-th draw in the traces (``>= 1``).
        store: Whether to keep the thinned physical samples.
        quantities: Functions of the physical ``x`` whose ESS is reported
            (default: the entries at the first, middle and last index).
        max_beta: Upper bound of the adapted ``beta`` (``beta = max_beta *
            sigmoid(theta)``); below one so that a very accurate proposal does
            not degenerate into a pure independence sampler.

    Returns:
        :class:`ChainResult`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an option is invalid or the start has non-finite
            likelihood.

    References:
        [EQ-INV-LAPX-009], [EQ-INV-LAPX-011],
        [REF-COTTER-ROBERTS-STUART-WHITE-2013].

    """
    return _run_chain(
        problem,
        None,
        "pcn",
        draws=draws,
        burn_in=burn_in,
        beta=beta,
        step=1.0,
        adapt=adapt,
        target_acceptance=target_acceptance,
        initial=initial,
        seed=seed,
        thin=thin,
        store=store,
        quantities=quantities,
        complement_beta=None,
        rank=None,
        max_beta=max_beta,
    )


def gpcn_chain(
    problem: NonlinearBayesProblem,
    laplace: LaplaceApproximation,
    *,
    draws: int,
    burn_in: int = 0,
    beta: float = 0.3,
    adapt: bool = True,
    target_acceptance: float | None = None,
    initial: Sequence[float] | None = None,
    seed: int | RandomStream | None = 0,
    thin: int = 1,
    store: bool = True,
    quantities: Sequence[Callable[[Vector], float]] | None = None,
    rank: int | None = None,
    max_beta: float = 0.5,
) -> ChainResult:
    """Run generalized pCN with the Laplace covariance as proposal covariance.

    The proposal ``z' = m + sqrt(1 - beta^2)(z - m) + beta Sigma^(1/2) xi`` with
    ``m = z_map`` and ``Sigma = (I + V diag(lambda) V')^-1`` is reversible for
    ``N(m, Sigma)``, so the acceptance probability is
    ``min(1, exp(ell(z') - ell(z) - 1/2||z'||^2 + 1/2||z||^2 + 1/2 q(z') -
    1/2 q(z)))`` with ``q(z) = ||z - m||^2 + sum lambda_i (v_i'(z - m))^2``
    [EQ-INV-LAPX-009]. Because ``Sigma`` differs from ``I`` only on the finite
    rank subspace, the Gaussian measures are equivalent and the sampler is
    well defined in the function-space limit. It is exact for the posterior
    for every ``G``; for linear ``G`` and a full-rank Laplace covariance the
    proposal with ``beta = 1`` is the posterior (acceptance 1). One forward
    solve per iteration, no adjoint.

    Args:
        problem: The problem.
        laplace: Laplace approximation of the same problem.
        draws: Post-burn-in iterations (``>= 1``).
        burn_in: Burn-in iterations, adapted and discarded.
        beta: Initial step in ``(0, 1]``.
        adapt: Adapt the logit of ``beta`` during burn-in.
        target_acceptance: Target acceptance in ``(0, 1)`` (default 0.3).
        initial: Start ``z`` (default ``z_map``).
        seed: Seed or stream.
        thin: Trace thinning (``>= 1``).
        store: Whether to keep the thinned physical samples.
        quantities: Functions of ``x`` whose ESS is reported.
        rank: Use only the leading ``rank`` Laplace eigenpairs.
        max_beta: Upper bound of the adapted ``beta`` (see :func:`pcn_chain`).
            The default 0.5 is deliberate: when the Laplace covariance is
            narrower than the posterior (heavier tails), ``beta`` near one
            behaves like an independence sampler that mixes in the tails very
            slowly while its split-chain ESS still looks large. On the
            two-parameter test problem with a Laplace variance ratio of 0.66
            the 200000-draw variance ratio to quadrature was 1.03 at
            ``beta = 0.3`` but 0.82 at ``beta = 0.99``.

    Returns:
        :class:`ChainResult`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an option is invalid, ``laplace`` belongs to another
            problem or the start has non-finite likelihood.

    References:
        [EQ-INV-LAPX-009], [REF-RUDOLF-SPRUNGK-2018],
        [REF-COTTER-ROBERTS-STUART-WHITE-2013].

    """
    return _run_chain(
        problem,
        laplace,
        "gpcn",
        draws=draws,
        burn_in=burn_in,
        beta=beta,
        step=1.0,
        adapt=adapt,
        target_acceptance=target_acceptance,
        initial=initial,
        seed=seed,
        thin=thin,
        store=store,
        quantities=quantities,
        complement_beta=None,
        rank=rank,
        max_beta=max_beta,
    )


def dili_chain(
    problem: NonlinearBayesProblem,
    laplace: LaplaceApproximation,
    *,
    draws: int,
    burn_in: int = 0,
    step: float = 0.5,
    complement_beta: float | None = None,
    adapt: bool = True,
    target_acceptance: float | None = None,
    initial: Sequence[float] | None = None,
    seed: int | RandomStream | None = 0,
    thin: int = 1,
    store: bool = True,
    quantities: Sequence[Callable[[Vector], float]] | None = None,
    rank: int | None = None,
) -> ChainResult:
    """Run a DILI-style split: Langevin in the LIS, pCN in its complement.

    Write ``z = V u + z_perp`` with ``V`` the Laplace eigenvectors. The
    likelihood-informed coordinates ``u`` move by a Hessian-preconditioned
    Langevin step ``u' = u + h/2 M g(z) + sqrt(h M) zeta`` with
    ``M = diag(1/(1 + lambda_i))`` and ``g = V' grad log pi``; the complement
    moves by pCN, ``z_perp' = rho z_perp + beta (xi - V V' xi)`` with
    ``beta = sqrt(h)/(1 + h/4)`` (the Crank-Nicolson pairing) unless
    ``complement_beta`` is given [EQ-INV-LAPX-010]. The complement pCN is
    reversible for its Gaussian prior marginal and cancels in the ratio, which
    leaves the Metropolis-adjusted Langevin ratio in ``u`` only; the chain is
    therefore exact for every ``G``. Each iteration costs one forward solve
    and one adjoint (the gradient at the proposal). With ``adapt`` the
    log step follows Robbins-Monro toward ``target_acceptance`` (default
    0.574) during burn-in. This follows the structure of dimension-independent
    likelihood-informed MCMC; it is not a reproduction of the published
    algorithm's operator-weighted proposal.

    Args:
        problem: The problem.
        laplace: Laplace approximation of the same problem.
        draws: Post-burn-in iterations (``>= 1``).
        burn_in: Burn-in iterations, adapted and discarded.
        step: Initial Langevin step ``h > 0``.
        complement_beta: Fixed complement pCN parameter in ``(0, 1]`` instead
            of the pairing with ``h``.
        adapt: Adapt ``log h`` during burn-in.
        target_acceptance: Target acceptance in ``(0, 1)`` (default 0.574).
        initial: Start ``z`` (default ``z_map``).
        seed: Seed or stream.
        thin: Trace thinning (``>= 1``).
        store: Whether to keep the thinned physical samples.
        quantities: Functions of ``x`` whose ESS is reported.
        rank: Use only the leading ``rank`` Laplace eigenpairs as the LIS.

    Returns:
        :class:`ChainResult`.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If an option is invalid, ``laplace`` has rank zero or
            belongs to another problem, or the start has a non-finite
            likelihood or gradient.

    References:
        [EQ-INV-LAPX-010], [REF-CUI-LAW-MARZOUK-2016],
        [REF-COTTER-ROBERTS-STUART-WHITE-2013].

    """
    if isinstance(laplace, LaplaceApproximation) and laplace.rank == 0:
        raise ValueError("dili_chain needs a Laplace approximation of rank >= 1")
    return _run_chain(
        problem,
        laplace,
        "dili",
        draws=draws,
        burn_in=burn_in,
        beta=0.5,
        step=step,
        adapt=adapt,
        target_acceptance=target_acceptance,
        initial=initial,
        seed=seed,
        thin=thin,
        store=store,
        quantities=quantities,
        complement_beta=complement_beta,
        rank=rank,
        max_beta=1.0,
    )
