"""Count, discrepancy and functional benchmark cases (roadmap 19: S01, M01, F01).

Experimental (added in 0.28). Continues the catalogs of
:mod:`probstat.inverse.benchmark_problems` (L01-L06) and
:mod:`probstat.inverse.benchmark_extended` (L07, N01-N02, P01-P02) with three
cases that exercise methods the earlier cases cannot:

- S01 :func:`poisson_scalar_benchmark` and :func:`poisson_image_benchmark`:
  blurred non-negative intensities observed through Poisson counts. The exact
  scalar case (one pixel: ``MLE = count / exposure``, Fisher information
  ``exposure / x``) is a closed-form reference; the 1D and small 2D cases carry
  a reference maximum-likelihood solution from the library's Fisher-scoring
  routine :func:`~probstat.inverse.observation_laws.maximum_likelihood_linear`
  or from the multiplicative ML-EM iteration :func:`poisson_em`
  [EQ-INV-BENCH3-001], [EQ-INV-BENCH3-003].
- M01 :func:`discrepancy_benchmark`: a linear model plus a Gaussian-process
  model discrepancy and AR(1)-correlated noise, with the exact marginalised
  Gaussian conditional of the parameters in closed form
  (:func:`gaussian_conditional_reference`) and a measured coverage study
  (:func:`discrepancy_coverage_study`) showing that ignoring the discrepancy
  gives overconfident intervals [EQ-INV-BENCH3-005], [EQ-INV-BENCH3-006],
  [EQ-INV-BENCH3-007].
- F01 :func:`functional_benchmark`: a Fredholm first-kind problem on
  ``[0, 1]`` whose target is a LOCAL AVERAGE of the unknown, with data-informed
  modes and averaging kernels from :func:`functional_analysis`
  [EQ-INV-BENCH3-008], [EQ-INV-BENCH3-009].

Contract (shared with the earlier catalogs). Each instance separates what a
solver may see (operator, data, noise model, optional prior) from what only a
benchmark may see (truth, noise-free data, withheld noise and discrepancy
draws). Truth and noise come from DISTINCT children of one
:class:`~probstat.foundations.RandomStream`, so changing the noise seed never
changes the truth. ``same_discretization`` marks the inverse-crime flag: ``True``
means the data were produced by the very discrete model the solver inverts
(algebraic verification only); ``False`` means they come from an independent,
finer representation, so a genuine discretization error is present
[EQ-INV-BENCH3-010].

Scope and limitations. All three cases are synthetic and small: dense
oracles (``O(n^3)``) are used on purpose, so none of them is a performance
benchmark. S01 uses the identity-link Poisson law with a pixel-wise Gaussian
blur kernel; positivity of the intensity is a constraint, not a
parametrisation (see :class:`PoissonReference`). M01 uses a two-parameter
linear model and a known covariance structure; hyperparameters are NOT
estimated, so the reference is the exact conditional for the true
hyperparameters. F01 uses a piecewise-constant representation of the unknown
and the truncated SVD; it makes no statement about the continuum limit
beyond the closed-form singular values of the Brownian-motion kernel.

References:
    [REF-HANSEN-2008], [REF-KAIPIO-SOMERSALO-2007],
    [REF-MCCULLAGH-NELDER-1989], [REF-KENNEDY-OHAGAN-2001],
    [REF-RASMUSSEN-WILLIAMS-2006].

"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from probstat.distributions import ChiSquare, Normal, Poisson
from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.inverse.benchmark_extended import (
    GaussianBlur2DOperator,
    _block_average,
    _finite_real,
    _phantom_image,
    _real_positive,
)
from probstat.inverse.benchmark_problems import (
    BenchmarkCase,
    _gaussian_kernel,
    _noise,
    _piecewise_truth,
    _positive_int,
    _smooth_truth,
)
from probstat.inverse.discrepancy import DISCREPANCY_KERNELS, discrepancy_kernel
from probstat.inverse.function_space import PiecewiseConstantBasis, _composite_rule
from probstat.inverse.observation_laws import PoissonLaw, maximum_likelihood_linear
from probstat.inverse.spectral import spectral_diagnostics
from probstat.linalg import (
    Convolution1DOperator,
    DenseCovariance,
    DenseOperator,
    DiagonalOperator,
    LinearOperator,
    Matrix,
    ScaledIdentityCovariance,
    ScaledOperator,
    Vector,
    cholesky,
    cholesky_solve,
    inverse_spd,
    materialize,
    require_operator,
    solve_lower,
)

__all__ = [
    "CATALOG_BENCHMARK_CASES",
    "COVERAGE_SEEDS_FULL",
    "COVERAGE_SEEDS_QUICK",
    "CoverageStudy",
    "DiscrepancyBenchmark",
    "FunctionalAnalysis",
    "FunctionalBenchmark",
    "FunctionalTradeoff",
    "GaussianConditionalReference",
    "PoissonBenchmark",
    "PoissonEMResult",
    "PoissonReference",
    "PoissonScalarReference",
    "discrepancy_benchmark",
    "discrepancy_coverage_study",
    "fredholm_kernel",
    "functional_analysis",
    "functional_benchmark",
    "functional_truth_profile",
    "gaussian_conditional_reference",
    "poisson_em",
    "poisson_image_benchmark",
    "poisson_scalar_benchmark",
    "poisson_scalar_reference",
    "zero_count_upper_limit",
]

_DENSE_FISHER_LIMIT = 250_000
_NORMAL = Normal(0.0, 1.0)


# ---------------------------------------------------------------------------
# Shared validation
# ---------------------------------------------------------------------------


def _non_negative_real(value: object, name: str) -> float:
    """Validate a finite real number that is at least zero.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].

    """
    number = _finite_real(value, name)
    if number < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _level(value: object, name: str = "level") -> float:
    """Validate a probability level strictly inside ``(0, 1)``.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].

    """
    number = _finite_real(value, name)
    if not 0.0 < number < 1.0:
        raise ValueError(f"{name} must lie strictly between 0 and 1")
    return number


def _count_vector(counts: Sequence[float], name: str = "counts") -> Vector:
    """Validate Poisson counts: finite non-negative integers.

    References:
        [EQ-INV-BENCH3-001], [REF-PROBSTAT-VALIDATION].

    """
    values: list[float] = []
    for entry in counts:
        if isinstance(entry, bool) or not isinstance(entry, (int, float)):
            raise TypeError(f"{name} must contain real numbers")
        number = float(entry)
        if not math.isfinite(number) or number < 0.0 or number != math.floor(number):
            raise ValueError(f"{name} must be finite non-negative integers")
        values.append(number)
    if not values:
        raise ValueError(f"{name} must be non-empty")
    return tuple(values)


def _poisson_log_likelihood(counts: Sequence[float], rates: Sequence[float]) -> float:
    """Return ``sum_i log Poisson(y_i; mu_i)`` allowing ``mu_i = 0``.

    The term ``y log mu`` is dropped for ``y = 0`` (so ``mu = 0`` is exact) and
    the value is ``-inf`` for ``y > 0`` with ``mu = 0``.

    References:
        [EQ-INV-BENCH3-001], [REF-MCCULLAGH-NELDER-1989].

    """
    terms: list[float] = []
    for k, mu in zip(counts, rates, strict=True):
        if mu < 0.0:
            raise ValueError("Poisson rates must be non-negative")
        if mu == 0.0:
            if k > 0.0:
                return -math.inf
            continue
        terms.append(k * math.log(mu) - mu - math.lgamma(k + 1.0))
    return math.fsum(terms)


# ---------------------------------------------------------------------------
# S01: Poisson count / image inversion
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PoissonScalarReference:
    """Closed-form reference of one Poisson count ``y ~ Poisson(t x)``.

    Attributes:
        count: The observed count ``y``.
        exposure: The exposure ``t > 0`` (rate ``mu = t x``).
        mle: ``y / t``, the exact maximum-likelihood intensity (``0`` for a
            zero count, a boundary solution).
        log_likelihood: ``log Poisson(y; t * mle)``.
        fisher_information: ``t / x`` at the MLE; ``None`` at a zero count
            (the information diverges at the boundary ``x = 0``).
        standard_error: ``sqrt(x / t)`` at the MLE (Wald); ``None`` at zero.
        boundary: ``True`` when the MLE lies on the constraint ``x = 0``.
        zero_count_upper_limit: Exact one-sided 95 percent upper confidence
            limit ``-log(0.05) / t`` for a zero count, else ``None``.

    References:
        [EQ-INV-BENCH3-001], [EQ-INV-BENCH3-002], [REF-GARWOOD-1936].

    """

    count: int
    exposure: float
    mle: float
    log_likelihood: float
    fisher_information: float | None
    standard_error: float | None
    boundary: bool
    zero_count_upper_limit: float | None

    def garwood_interval(self, level: float = 0.95) -> tuple[float, float]:
        """Return the exact (Garwood) central interval for the intensity.

        With ``k = y`` and ``alpha = 1 - level``: the lower limit is
        ``chi2_{alpha/2}(2k) / (2 t)`` (``0`` for ``k = 0``) and the upper limit
        ``chi2_{1-alpha/2}(2k + 2) / (2 t)`` [EQ-INV-BENCH3-002].

        Args:
            level: Coverage level in ``(0, 1)``.

        Returns:
            ``(lower, upper)`` intensity limits.

        Raises:
            ValueError: If ``level`` is not strictly between 0 and 1.

        References:
            [EQ-INV-BENCH3-002], [REF-GARWOOD-1936].

        """
        confidence = _level(level)
        alpha = 1.0 - confidence
        lower = (
            0.0
            if self.count == 0
            else ChiSquare(2 * self.count).quantile(alpha / 2.0) / (2.0 * self.exposure)
        )
        upper = ChiSquare(2 * self.count + 2).quantile(1.0 - alpha / 2.0) / (
            2.0 * self.exposure
        )
        return lower, upper


def zero_count_upper_limit(exposure: float, confidence: float = 0.95) -> float:
    """Return the exact one-sided upper limit of the intensity for a zero count.

    ``P(Y = 0 | x) = exp(-t x) >= 1 - confidence`` gives
    ``x <= -log(1 - confidence) / t`` [EQ-INV-BENCH3-002].

    Args:
        exposure: Exposure ``t > 0``.
        confidence: One-sided confidence in ``(0, 1)``.

    Returns:
        ``-log(1 - confidence) / t``.

    Raises:
        TypeError: If an argument is not a real number.
        ValueError: If ``exposure`` is not positive and finite or the
            confidence is outside ``(0, 1)``.

    References:
        [EQ-INV-BENCH3-002], [REF-GARWOOD-1936].

    """
    t = _real_positive(exposure, "exposure")
    c = _level(confidence, "confidence")
    return -math.log(1.0 - c) / t


def poisson_scalar_reference(count: int, exposure: float) -> PoissonScalarReference:
    """Return the closed-form MLE and Fisher information of one Poisson count.

    ``log p(y | x) = y log(t x) - t x - log y!``; the score ``y / x - t``
    vanishes at ``x = y / t`` and the information is ``t^2 / (t x) = t / x``
    [EQ-INV-BENCH3-001]. A zero count has the boundary maximizer ``x = 0``
    [EQ-INV-BENCH3-002].

    Args:
        count: Non-negative integer count.
        exposure: Exposure ``t > 0``.

    Returns:
        :class:`PoissonScalarReference`.

    Raises:
        TypeError: If ``count`` is not an integer or ``exposure`` is not real.
        ValueError: If ``count`` is negative or ``exposure`` is not positive and
            finite.

    References:
        [EQ-INV-BENCH3-001], [EQ-INV-BENCH3-002], [REF-MCCULLAGH-NELDER-1989].

    """
    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("count must be an integer")
    if count < 0:
        raise ValueError("count must be non-negative")
    t = _real_positive(exposure, "exposure")
    mle = count / t
    loglik = _poisson_log_likelihood((float(count),), (t * mle,))
    if count == 0:
        return PoissonScalarReference(
            count, t, 0.0, loglik, None, None, True, zero_count_upper_limit(t)
        )
    return PoissonScalarReference(
        count, t, mle, loglik, t / mle, math.sqrt(mle / t), False, None
    )


@dataclass(frozen=True, slots=True)
class PoissonEMResult:
    """Result of the multiplicative ML-EM (Richardson-Lucy) iteration.

    Attributes:
        estimate: Final intensity (non-negative by construction).
        iterations: Number of multiplicative updates performed.
        converged: Whether the relative update fell below the tolerance.
        stop_reason: ``"update_tolerance"`` or ``"max_iterations"``.
        log_likelihood: ``log p(y | A x)`` at the estimate.
        kkt_residual: ``max_j |min(x_j, g_j)|`` with the negative
            log-likelihood gradient ``g = A'(1 - y / (A x))``; zero exactly at a
            constrained maximizer.

    References:
        [EQ-INV-BENCH3-003], [REF-SHEPP-VARDI-1982], [REF-LUCY-1974].

    """

    estimate: Vector
    iterations: int
    converged: bool
    stop_reason: str
    log_likelihood: float
    kkt_residual: float


def _nll_gradient(operator: LinearOperator, counts: Vector, point: Vector) -> Vector:
    """Return ``A'(1 - y / (A x))``, the gradient of the negative log-likelihood.

    References:
        [EQ-INV-BENCH3-003].

    """
    rates = operator.matvec(point)
    weights: list[float] = []
    for k, mu in zip(counts, rates, strict=True):
        if mu > 0.0:
            weights.append(1.0 - k / mu)
        elif k == 0.0:
            weights.append(1.0)
        else:
            raise ValueError("a positive count has zero predicted rate")
    return tuple(operator.rmatvec(weights))


def _kkt_residual(point: Vector, gradient: Vector) -> float:
    """Return ``max_j |min(x_j, g_j)|`` (complementarity of the KKT conditions).

    References:
        [EQ-INV-BENCH3-003], [REF-NOCEDAL-WRIGHT-2006].

    """
    return max(abs(min(x, g)) for x, g in zip(point, gradient, strict=True))


def _uniform_start(operator: LinearOperator, counts: Vector) -> Vector:
    """Return the constant intensity whose total predicted count equals ``sum y``.

    References:
        [EQ-INV-BENCH3-003].

    """
    n = operator.shape[1]
    total_response = math.fsum(operator.matvec((1.0,) * n))
    total_counts = math.fsum(counts)
    if total_response <= 0.0:
        raise ValueError("the operator has no positive response")
    return (total_counts / total_response,) * n


def poisson_em(
    operator: LinearOperator | Sequence[Sequence[float]],
    counts: Sequence[float],
    *,
    initial: Sequence[float] | None = None,
    max_iterations: int = 2000,
    tolerance: float = 1e-10,
) -> PoissonEMResult:
    """Maximize the Poisson likelihood of ``y ~ Poisson(A x)`` over ``x >= 0``.

    Multiplicative update ``x_j <- x_j / s_j * [A'(y / (A x))]_j`` with the
    column sums ``s = A' 1``. For ``A >= 0``, ``s > 0`` and a positive start,
    every iterate is positive (positivity by construction, not by projection),
    the likelihood is non-decreasing, and the limit satisfies the KKT
    conditions of the constrained maximizer [EQ-INV-BENCH3-003]. Convergence
    is linear and slow for ill-conditioned ``A``; the result states whether the
    tolerance was reached.

    Args:
        operator: Non-negative operator ``A`` (``m x n``). Non-negativity is
            checked only through the predicted rates at the iterates.
        counts: Non-negative integer counts of length ``m``.
        initial: Positive start (default: the constant matching ``sum y``).
        max_iterations: Positive iteration cap.
        tolerance: Stop when the largest update is at most
            ``tolerance * max(x)``.

    Returns:
        :class:`PoissonEMResult`.

    Raises:
        TypeError: For wrong argument types.
        ValueError: For inconsistent sizes, invalid counts, a non-positive start,
            a zero column sum, a negative predicted rate, or a positive count
            with zero predicted rate.

    References:
        [EQ-INV-BENCH3-003], [REF-SHEPP-VARDI-1982], [REF-LUCY-1974],
        [REF-RICHARDSON-1972].

    """
    a = require_operator(operator, name="operator")
    m, n = a.shape
    y = _count_vector(counts)
    if len(y) != m:
        raise ValueError("counts must have one entry per row of the operator")
    iterations_cap = _positive_int(max_iterations, "max_iterations")
    tol = _real_positive(tolerance, "tolerance")
    column_sums = a.rmatvec((1.0,) * m)
    if any(s <= 0.0 for s in column_sums):
        raise ValueError("every column of the operator must have a positive sum")
    if initial is None:
        x = _uniform_start(a, y)
        if x[0] <= 0.0:
            raise ValueError("all counts are zero; the maximizer x = 0 is a boundary")
    else:
        x = tuple(_finite_real(v, "initial") for v in initial)
        if len(x) != n or any(v <= 0.0 for v in x):
            raise ValueError("initial must be positive with one entry per unknown")
    converged = False
    stop_reason = "max_iterations"
    done = 0
    for _ in range(iterations_cap):
        rates = a.matvec(x)
        ratios: list[float] = []
        for k, mu in zip(y, rates, strict=True):
            if mu < 0.0:
                raise ValueError("the operator produced a negative predicted rate")
            if mu == 0.0:
                if k > 0.0:
                    raise ValueError("a positive count has zero predicted rate")
                ratios.append(0.0)
            else:
                ratios.append(k / mu)
        back = a.rmatvec(ratios)
        updated = tuple(
            xj * bj / sj for xj, bj, sj in zip(x, back, column_sums, strict=True)
        )
        change = max(abs(u - v) for u, v in zip(updated, x, strict=True))
        x = updated
        done += 1
        if change <= tol * max(max(x), 1e-300):
            converged = True
            stop_reason = "update_tolerance"
            break
    gradient = _nll_gradient(a, y, x)
    return PoissonEMResult(
        estimate=x,
        iterations=done,
        converged=converged,
        stop_reason=stop_reason,
        log_likelihood=_poisson_log_likelihood(y, a.matvec(x)),
        kkt_residual=_kkt_residual(x, gradient),
    )


@dataclass(frozen=True, slots=True)
class PoissonReference:
    """Reference maximum-likelihood solution of a Poisson count benchmark.

    POSITIVITY. The intensity must satisfy ``x >= 0``; it is a CONSTRAINT, not
    a log parametrisation. ``method = "closed_form"`` solves pixelwise
    (``x = y / t``, exactly non-negative, boundary at zero counts);
    ``"fisher_scoring"`` runs the library's penalized Fisher scoring with the
    identity link, which keeps the predicted rate ``A x`` positive through its
    line search but does NOT constrain ``x`` (so ``positive`` can be
    ``False``); ``"em"`` runs :func:`poisson_em`, positive by construction.

    Attributes:
        estimate: The reference intensity.
        method: ``"closed_form"``, ``"boundary_zero"``, ``"fisher_scoring"``
            or ``"em"``.
        converged: Whether the method reports convergence.
        stop_reason: The method's stop reason.
        iterations: Number of iterations (``0`` for closed forms).
        regularization: Ridge weight ``lambda`` of the penalized objective
            (``0`` for the plain MLE).
        log_likelihood: ``log p(y | A x)`` at the estimate.
        min_estimate: ``min_j x_j``.
        positive: ``True`` when every entry is at least zero.
        kkt_residual: ``max_j |min(x_j, g_j)|`` of the unpenalized objective
            (meaningful for ``regularization = 0``).
        fisher_information: Dense ``A' diag(1 / (A x)) A`` at the estimate, or
            ``None`` when a predicted rate is zero or the matrix is large.

    References:
        [EQ-INV-BENCH3-001], [EQ-INV-BENCH3-003], [REF-MCCULLAGH-NELDER-1989].

    """

    estimate: Vector
    method: str
    converged: bool
    stop_reason: str
    iterations: int
    regularization: float
    log_likelihood: float
    min_estimate: float
    positive: bool
    kkt_residual: float
    fisher_information: Matrix | None


@dataclass(frozen=True, slots=True)
class PoissonBenchmark:
    """One Poisson count benchmark instance: solver inputs plus withheld truth.

    Attributes:
        identifier: Catalog ID (``"S01"``).
        name: Short description.
        operator: Forward operator ``A`` including the exposure; the mean
            counts are ``A x``.
        data: Observed counts (as floats).
        exposure: The scalar exposure folded into ``A``.
        truth: Withheld intensity on the solver's grid.
        exact_data: Withheld noise-free mean counts ``b``.
        realized_noise_norm: ``||y - b||`` of this draw (reported).
        same_discretization: ``True`` when ``b = A x_true`` exactly (inverse
            crime, algebraic verification only).
        positivity: Statement of how ``x >= 0`` is treated.
        zero_count_indices: Indices of pixels with a zero count.
        reference: Reference maximum-likelihood solution.
        pathologies: Expected difficulties.
        seeds: The stream paths used for the noise (count sampling).

    References:
        [EQ-INV-BENCH3-001], [EQ-INV-BENCH3-004], [EQ-INV-BENCH3-010].

    """

    identifier: str
    name: str
    operator: LinearOperator
    data: Vector
    exposure: float
    truth: Vector
    exact_data: Vector
    realized_noise_norm: float
    same_discretization: bool
    positivity: str
    zero_count_indices: tuple[int, ...]
    reference: PoissonReference
    pathologies: tuple[str, ...]
    seeds: Mapping[str, str] = field(default_factory=dict)


def _sample_counts(mean_counts: Sequence[float], stream: RandomStream) -> Vector:
    """Draw independent Poisson counts, one child stream per entry.

    References:
        [EQ-INV-BENCH3-001].

    """
    return tuple(
        float(Poisson(mu).sample(1, seed=stream.child(i))[0])
        for i, mu in enumerate(mean_counts)
    )


def _dense_fisher(operator: LinearOperator, rates: Vector) -> Matrix | None:
    """Return ``A' diag(1 / mu) A`` when it is small and every rate is positive.

    References:
        [EQ-INV-BENCH3-001].

    """
    m, n = operator.shape
    if m * n > _DENSE_FISHER_LIMIT or any(mu <= 0.0 for mu in rates):
        return None
    dense = materialize(operator, max_entries=_DENSE_FISHER_LIMIT)
    return tuple(
        tuple(
            math.fsum(dense[i][j] * dense[i][k] / rates[i] for i in range(m))
            for k in range(n)
        )
        for j in range(n)
    )


def _poisson_reference(
    operator: LinearOperator,
    counts: Vector,
    exposure: float,
    *,
    pixelwise: bool,
    method: str,
    regularization: float,
) -> PoissonReference:
    """Compute the reference solution of a Poisson count problem.

    References:
        [EQ-INV-BENCH3-001], [EQ-INV-BENCH3-003].

    """
    if method not in ("fisher_scoring", "em"):
        raise ValueError("reference_method must be 'fisher_scoring' or 'em'")
    n = operator.shape[1]
    if pixelwise:
        estimate = tuple(k / exposure for k in counts)
        rates = operator.matvec(estimate)
        return PoissonReference(
            estimate=estimate,
            method="closed_form",
            converged=True,
            stop_reason="closed_form",
            iterations=0,
            regularization=0.0,
            log_likelihood=_poisson_log_likelihood(counts, rates),
            min_estimate=min(estimate),
            positive=True,
            kkt_residual=_kkt_residual(
                estimate, _nll_gradient(operator, counts, estimate)
            ),
            fisher_information=_dense_fisher(operator, rates),
        )
    if math.fsum(counts) == 0.0:
        zeros = (0.0,) * n
        return PoissonReference(
            estimate=zeros,
            method="boundary_zero",
            converged=True,
            stop_reason="all_counts_zero",
            iterations=0,
            regularization=0.0,
            log_likelihood=_poisson_log_likelihood(counts, operator.matvec(zeros)),
            min_estimate=0.0,
            positive=True,
            kkt_residual=0.0,
            fisher_information=None,
        )
    if method == "em":
        em = poisson_em(operator, counts)
        point = em.estimate
        converged, reason, iterations = em.converged, em.stop_reason, em.iterations
        lam = 0.0
    else:
        lam = _non_negative_real(regularization, "reference_regularization")
        fit = maximum_likelihood_linear(
            operator,
            counts,
            PoissonLaw("identity"),
            regularization=lam,
            initial=_uniform_start(operator, counts),
            max_iterations=200,
        )
        point = fit.estimate
        converged, reason, iterations = fit.converged, fit.stop_reason, fit.iterations
    rates = operator.matvec(point)
    return PoissonReference(
        estimate=point,
        method=method,
        converged=converged,
        stop_reason=reason,
        iterations=iterations,
        regularization=lam,
        log_likelihood=_poisson_log_likelihood(counts, rates),
        min_estimate=min(point),
        positive=min(point) >= 0.0,
        kkt_residual=_kkt_residual(point, _nll_gradient(operator, counts, point)),
        fisher_information=_dense_fisher(operator, rates),
    )


_POSITIVITY_PIXELWISE = (
    "constraint x >= 0; pixelwise MLE y / t is non-negative in closed form"
)
_POSITIVITY_BLURRED = (
    "constraint x >= 0 (no log parametrisation); the truth is strictly positive "
    "through a background; the reference reports min(x) and KKT residuals"
)


def poisson_scalar_benchmark(
    intensity: float = 4.0,
    exposure: float = 5.0,
    *,
    count: int | None = None,
    seed: int | RandomStream | None = 11,
) -> PoissonBenchmark:
    """Return S01 (scalar): one pixel with ``y ~ Poisson(t x)``.

    The reference is the exact closed form of
    :func:`poisson_scalar_reference`. With ``count`` given the count is not
    sampled (use ``count = 0`` for the zero-count boundary case); otherwise it
    is drawn from the ``noise`` stream.

    Args:
        intensity: True intensity ``x >= 0`` (``0`` always gives a zero count).
        exposure: Exposure ``t > 0``.
        count: Optional fixed count overriding the draw.
        seed: Root seed (only the count draw is random).

    Returns:
        :class:`PoissonBenchmark` with ``same_discretization=True`` (there is no
        discretization in the scalar case).

    Raises:
        TypeError: If an argument is not a real number or ``count`` is not an
            integer.
        ValueError: If ``intensity`` is negative or non-finite, ``exposure``
            is not positive and finite, or ``count`` is negative.

    References:
        [EQ-INV-BENCH3-001], [EQ-INV-BENCH3-002], [REF-MCCULLAGH-NELDER-1989].

    """
    x = _non_negative_real(intensity, "intensity")
    t = _real_positive(exposure, "exposure")
    mean = t * x
    stream = resolve_stream(seed).child("S01")
    if count is None:
        data = _sample_counts((mean,), stream.child("noise"))
    else:
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError("count must be an integer")
        if count < 0:
            raise ValueError("count must be non-negative")
        data = (float(count),)
    scalar = poisson_scalar_reference(int(data[0]), t)
    operator = DenseOperator([[t]])
    rates = operator.matvec((scalar.mle,))
    reference = PoissonReference(
        estimate=(scalar.mle,),
        method="closed_form",
        converged=True,
        stop_reason="closed_form",
        iterations=0,
        regularization=0.0,
        log_likelihood=scalar.log_likelihood,
        min_estimate=scalar.mle,
        positive=True,
        kkt_residual=_kkt_residual(
            (scalar.mle,), _nll_gradient(operator, data, (scalar.mle,))
        ),
        fisher_information=_dense_fisher(operator, rates),
    )
    return PoissonBenchmark(
        identifier="S01",
        name=f"Poisson scalar (x={x}, t={t}, y={int(data[0])})",
        operator=operator,
        data=data,
        exposure=t,
        truth=(x,),
        exact_data=(mean,),
        realized_noise_norm=abs(data[0] - mean),
        same_discretization=True,
        positivity=_POSITIVITY_PIXELWISE,
        zero_count_indices=(0,) if data[0] == 0.0 else (),
        reference=reference,
        pathologies=("poisson_noise", "zero_counts")
        if data[0] == 0.0
        else ("poisson_noise",),
        seeds=MappingProxyType({"noise": "S01/noise"}),
    )


def poisson_image_benchmark(
    size: int = 16,
    *,
    dimension: int = 1,
    width: float | None = 0.04,
    exposure: float = 20.0,
    background: float = 0.2,
    peak: float = 1.0,
    truth: str = "smooth",
    refinement: int = 1,
    reference_method: str = "fisher_scoring",
    reference_regularization: float = 0.0,
    seed: int | RandomStream | None = 11,
) -> PoissonBenchmark:
    """Return S01 (image): blurred non-negative intensities with Poisson counts.

    The solver's operator is ``A = t B`` where ``B`` is a non-negative Gaussian
    blur (zero boundary; ``None`` for the identity, the pixelwise case with an
    exact closed-form MLE) on ``n`` cells of ``[0, 1]`` (1D) or ``n x n`` pixels
    of the unit square (2D, row-major). The truth is ``background + peak * shape``
    with an analytic shape, so every mean count is strictly positive and
    ``Poisson`` sampling is well defined; zero COUNTS still occur at small
    exposure. With ``refinement = r > 1`` the intensity is sampled on an
    ``r``-times finer grid, blurred there with the same physical kernel and
    cell-averaged, so the mean counts carry a genuine discretization error
    [EQ-INV-BENCH3-004]. Counts come from the ``noise`` stream.

    Args:
        size: Coarse cells per side: ``>= 8`` in 1D, ``>= 4`` in 2D.
        dimension: ``1`` or ``2``.
        width: Physical blur standard deviation, or ``None`` (no blur).
        exposure: Exposure ``t > 0`` scaling counts per unit intensity.
        background: Strictly positive intensity floor.
        peak: Non-negative amplitude of the analytic shape.
        truth: ``"smooth"`` or ``"piecewise"`` in 1D (the 2D shape is the
            phantom of :func:`~probstat.inverse.benchmark_extended.
            image_deblur_benchmark`).
        refinement: Integer factor ``r >= 1`` of the data-generating grid.
        reference_method: ``"fisher_scoring"`` (library Poisson maximum
            likelihood) or ``"em"`` (ML-EM); ignored when ``width`` is
            ``None`` (closed form).
        reference_regularization: Ridge weight of the Fisher-scoring reference
            (``0`` for the plain MLE; ignored for ``"em"``).
        seed: Root seed.

    Returns:
        :class:`PoissonBenchmark`.

    Raises:
        TypeError: If a numeric argument is not real.
        ValueError: On out-of-range sizes, a non-positive exposure, width or
            background, an unknown ``truth`` or reference method, or a
            non-finite argument.

    References:
        [EQ-INV-BENCH3-001], [EQ-INV-BENCH3-003], [EQ-INV-BENCH3-004],
        [REF-GAZZOLA-HANSEN-NAGY-2019], [REF-MCCULLAGH-NELDER-1989].

    """
    if isinstance(dimension, bool) or dimension not in (1, 2):
        raise ValueError("dimension must be 1 or 2")
    n = _positive_int(size, "size", 8 if dimension == 1 else 4)
    r = _positive_int(refinement, "refinement", 1)
    t = _real_positive(exposure, "exposure")
    floor = _real_positive(background, "background")
    amplitude = _non_negative_real(peak, "peak")
    sigma = None if width is None else _real_positive(width, "width")
    if truth not in ("smooth", "piecewise"):
        raise ValueError("truth must be 'smooth' or 'piecewise'")
    if reference_method not in ("fisher_scoring", "em"):
        raise ValueError("reference_method must be 'fisher_scoring' or 'em'")
    if dimension == 1:
        profile = _smooth_truth if truth == "smooth" else _piecewise_truth
        h = 1.0 / n
        fine_n = r * n
        fine_h = 1.0 / fine_n
        fine_truth = [
            floor + amplitude * profile((j + 0.5) * fine_h) for j in range(fine_n)
        ]
        coarse = tuple(math.fsum(fine_truth[i * r : (i + 1) * r]) / r for i in range(n))
        if sigma is None:
            base: LinearOperator = DiagonalOperator([1.0] * n)
            exact_rate = coarse
        else:
            base = Convolution1DOperator(_gaussian_kernel(sigma, h), n, boundary="zero")
            if r == 1:
                exact_rate = tuple(base.matvec(coarse))
            else:
                fine_op = Convolution1DOperator(
                    _gaussian_kernel(sigma, fine_h), fine_n, boundary="zero"
                )
                blurred = fine_op.matvec(fine_truth)
                exact_rate = tuple(
                    math.fsum(blurred[i * r : (i + 1) * r]) / r for i in range(n)
                )
        shape_name = f"1D {truth}"
        cells = n
    else:
        fine_rows = r * n
        fine_image = [
            floor + amplitude * v for v in _phantom_image(fine_rows, fine_rows)
        ]
        coarse = _block_average(fine_image, n, n, r)
        if sigma is None:
            base = DiagonalOperator([1.0] * (n * n))
            exact_rate = coarse
        else:
            base = GaussianBlur2DOperator(n, n, width=sigma)
            if r == 1:
                exact_rate = tuple(base.matvec(coarse))
            else:
                fine_blur = GaussianBlur2DOperator(fine_rows, fine_rows, width=sigma)
                exact_rate = _block_average(fine_blur.matvec(fine_image), n, n, r)
        shape_name = "2D phantom"
        cells = n * n
    operator: LinearOperator = ScaledOperator(base, t)
    exact = tuple(t * v for v in exact_rate)
    stream = resolve_stream(seed).child("S01")
    data = _sample_counts(exact, stream.child("noise"))
    reference = _poisson_reference(
        operator,
        data,
        t,
        pixelwise=sigma is None,
        method=reference_method,
        regularization=reference_regularization,
    )
    zeros = tuple(i for i, k in enumerate(data) if k == 0.0)
    pathologies = ["poisson_noise", "positivity"]
    if sigma is not None:
        pathologies.append("ill_conditioned")
    if zeros:
        pathologies.append("zero_counts")
    return PoissonBenchmark(
        identifier="S01",
        name=f"Poisson {shape_name} ({cells} cells, t={t}, refinement {r})",
        operator=operator,
        data=data,
        exposure=t,
        truth=coarse,
        exact_data=exact,
        realized_noise_norm=math.hypot(
            *(k - b for k, b in zip(data, exact, strict=True))
        ),
        same_discretization=(r == 1) or sigma is None,
        positivity=_POSITIVITY_PIXELWISE if sigma is None else _POSITIVITY_BLURRED,
        zero_count_indices=zeros,
        reference=reference,
        pathologies=tuple(pathologies),
        seeds=MappingProxyType({"noise": "S01/noise"}),
    )


# ---------------------------------------------------------------------------
# M01: GP model discrepancy and correlated noise
# ---------------------------------------------------------------------------

COVERAGE_SEEDS_QUICK: tuple[int, ...] = tuple(range(1001, 1101))
"""Predeclared seeds of the quick coverage study (100 replicates)."""

COVERAGE_SEEDS_FULL: tuple[int, ...] = tuple(range(1001, 2001))
"""Predeclared seeds of the slow coverage study (1000 replicates)."""

_DISCREPANCY_MODELS = ("full", "no_discrepancy", "white_noise")


@dataclass(frozen=True, slots=True)
class GaussianConditionalReference:
    """Exact Gaussian conditional of linear-model parameters given the data.

    Attributes:
        mean: Posterior mean ``mu_post``.
        covariance: Posterior covariance ``Sigma_post``.
        log_evidence: ``log N(y; H m0, C + H P0 H')``, the marginal density of
            the data under the stated covariance model.
        covariance_model: ``"full"``, ``"no_discrepancy"`` or
            ``"white_noise"`` (the data covariance assumed).

    References:
        [EQ-INV-BENCH3-006].

    """

    mean: Vector
    covariance: Matrix
    log_evidence: float
    covariance_model: str

    def standard_deviations(self) -> Vector:
        """Return the posterior marginal standard deviations.

        Returns:
            ``sqrt(Sigma_ii)`` for every parameter.

        References:
            [EQ-INV-BENCH3-006].

        """
        return tuple(math.sqrt(self.covariance[i][i]) for i in range(len(self.mean)))

    def interval(self, level: float = 0.95) -> tuple[tuple[float, float], ...]:
        """Return central marginal intervals ``mu_i +- z sd_i``.

        Args:
            level: Coverage level in ``(0, 1)``.

        Returns:
            One ``(lower, upper)`` pair per parameter.

        Raises:
            ValueError: If ``level`` is not strictly between 0 and 1.

        References:
            [EQ-INV-BENCH3-006].

        """
        z = _NORMAL.quantile(0.5 + 0.5 * _level(level))
        return tuple(
            (m - z * s, m + z * s)
            for m, s in zip(self.mean, self.standard_deviations(), strict=True)
        )


def _matrix_rows(matrix: Sequence[Sequence[float]], name: str) -> Matrix:
    """Validate a finite rectangular matrix and return it as tuples.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].

    """
    rows = tuple(tuple(_finite_real(v, name) for v in row) for row in matrix)
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError(f"{name} must be a non-empty rectangular matrix")
    return rows


def gaussian_conditional_reference(
    design: Sequence[Sequence[float]],
    data: Sequence[float],
    data_covariance: Sequence[Sequence[float]],
    prior_mean: Sequence[float],
    prior_covariance: Sequence[Sequence[float]],
    *,
    covariance_model: str = "full",
) -> GaussianConditionalReference:
    """Return the exact conditional of ``theta`` for ``y = H theta + e``.

    With ``e ~ N(0, C)`` (``C`` includes the marginalised discrepancy and the
    correlated noise) and ``theta ~ N(m0, P0)``: with ``L L' = C`` and
    ``H~ = L^-1 H``, ``y~ = L^-1 y``,
    ``Sigma_post^-1 = H~'H~ + P0^-1`` and
    ``mu_post = Sigma_post (H~'y~ + P0^-1 m0)``; the evidence is
    ``N(y; H m0, C + H P0 H')`` [EQ-INV-BENCH3-006]. No jitter is added.

    Args:
        design: ``H`` (``n x p``).
        data: ``y`` (length ``n``).
        data_covariance: ``C`` (``n x n``, symmetric positive definite).
        prior_mean: ``m0`` (length ``p``).
        prior_covariance: ``P0`` (``p x p``, symmetric positive definite).
        covariance_model: Label stored in the result.

    Returns:
        :class:`GaussianConditionalReference`.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If shapes disagree, entries are non-finite, or a
            covariance is not positive definite.

    References:
        [EQ-INV-BENCH3-006], [REF-RASMUSSEN-WILLIAMS-2006, Section 2.2],
        [REF-KAIPIO-SOMERSALO-2007].

    """
    h = _matrix_rows(design, "design")
    c = _matrix_rows(data_covariance, "data_covariance")
    p0 = _matrix_rows(prior_covariance, "prior_covariance")
    y = tuple(_finite_real(v, "data") for v in data)
    m0 = tuple(_finite_real(v, "prior_mean") for v in prior_mean)
    n, p = len(h), len(h[0])
    if len(y) != n or len(c) != n or len(c[0]) != n:
        raise ValueError("data and data_covariance must match the rows of design")
    if len(m0) != p or len(p0) != p or len(p0[0]) != p:
        raise ValueError("prior_mean and prior_covariance must match the columns")
    try:
        lower = cholesky(c)
        prior_precision = inverse_spd(p0)
    except ValueError as error:
        raise ValueError("covariances must be positive definite") from error
    h_columns = [solve_lower(lower, tuple(h[i][j] for i in range(n))) for j in range(p)]
    y_white = solve_lower(lower, y)
    precision = tuple(
        tuple(
            math.fsum(a * b for a, b in zip(h_columns[j], h_columns[k], strict=True))
            + prior_precision[j][k]
            for k in range(p)
        )
        for j in range(p)
    )
    covariance = inverse_spd(precision)
    rhs = tuple(
        math.fsum(a * b for a, b in zip(h_columns[j], y_white, strict=True))
        + math.fsum(prior_precision[j][k] * m0[k] for k in range(p))
        for j in range(p)
    )
    mean = tuple(
        math.fsum(covariance[j][k] * rhs[k] for k in range(p)) for j in range(p)
    )
    prior_mean_data = [math.fsum(h[i][k] * m0[k] for k in range(p)) for i in range(n)]
    marginal = [
        [
            c[i][k]
            + math.fsum(
                h[i][a] * p0[a][b] * h[k][b] for a in range(p) for b in range(p)
            )
            for k in range(n)
        ]
        for i in range(n)
    ]
    try:
        marginal_lower = cholesky(tuple(tuple(r) for r in marginal))
    except ValueError as error:
        raise ValueError(
            "the marginal data covariance is not positive definite"
        ) from error
    residual = tuple(a - b for a, b in zip(y, prior_mean_data, strict=True))
    solved = cholesky_solve(marginal_lower, residual)
    misfit = math.fsum(a * b for a, b in zip(residual, solved, strict=True))
    log_det = 2.0 * math.fsum(math.log(marginal_lower[i][i]) for i in range(n))
    evidence = -0.5 * (misfit + log_det + n * math.log(2.0 * math.pi))
    return GaussianConditionalReference(mean, covariance, evidence, covariance_model)


@dataclass(frozen=True, slots=True)
class DiscrepancyBenchmark:
    """One M01 instance: linear model, GP discrepancy and correlated noise.

    The data are ``y = H theta + delta(x) + e`` with ``delta ~ GP(0, k)`` (plus
    an explicit nugget) and AR(1) noise ``e``. The solver is given ``H``, ``y``,
    the correlated-noise covariance and the prior; the discrepancy covariance
    is also provided because the benchmark's reference conditions on it.

    Attributes:
        identifier: Catalog ID (``"M01"``).
        name: Short description.
        operator: Design operator ``H`` with rows ``(1, x_i)``.
        data: Observations ``y``.
        noise: Correlated-noise covariance ``Sigma_e`` (AR(1)).
        noise_level: Marginal noise standard deviation ``sigma_e``.
        realized_noise_norm: ``||e||`` of the withheld noise draw.
        truth: Withheld parameters ``theta`` (intercept, slope).
        exact_data: Withheld noise-free data ``H theta + delta``.
        same_discretization: ``True``: the data are generated with exactly the
            covariance structure the reference conditions on, so the case
            verifies calibration of the exact conditional, not discretization.
        prior_mean: Prior mean of ``theta``.
        prior_covariance: Prior covariance of ``theta``.
        discrepancy_covariance: ``K + nugget I`` of the discrepancy at the
            design points.
        total_covariance: ``K + nugget I + Sigma_e``.
        discrepancy_truth: Withheld discrepancy draw ``delta(x_i)``.
        noise_truth: Withheld noise draw ``e``.
        design_points: The inputs ``x_i`` in ``[0, 1]``.
        reference: Exact conditional under the TRUE full covariance.
        pathologies: Expected difficulties.
        seeds: The stream paths used for the truth and the noise.

    References:
        [EQ-INV-BENCH3-005], [EQ-INV-BENCH3-006], [EQ-INV-BENCH3-010].

    """

    identifier: str
    name: str
    operator: DenseOperator
    data: Vector
    noise: DenseCovariance
    noise_level: float
    realized_noise_norm: float
    truth: Vector
    exact_data: Vector
    same_discretization: bool
    prior_mean: Vector
    prior_covariance: DenseCovariance
    discrepancy_covariance: DenseCovariance
    total_covariance: DenseCovariance
    discrepancy_truth: Vector
    noise_truth: Vector
    design_points: Vector
    reference: GaussianConditionalReference
    pathologies: tuple[str, ...]
    seeds: Mapping[str, str] = field(default_factory=dict)
    _matrices: tuple[Matrix, Matrix, Matrix, Matrix] = field(
        default=((), (), (), ()), repr=False
    )

    def conditional_reference(
        self, model: str = "full"
    ) -> GaussianConditionalReference:
        """Return the exact conditional under an assumed covariance model.

        ``"full"`` uses ``K + nugget I + Sigma_e`` (the truth), ``"no_discrepancy"``
        uses ``Sigma_e`` only (the discrepancy is ignored but the noise
        correlation is kept) and ``"white_noise"`` uses ``sigma_e^2 I``
        (both ignored) [EQ-INV-BENCH3-006].

        Args:
            model: ``"full"``, ``"no_discrepancy"`` or ``"white_noise"``.

        Returns:
            :class:`GaussianConditionalReference`.

        Raises:
            ValueError: If ``model`` is unknown.

        References:
            [EQ-INV-BENCH3-006], [EQ-INV-BENCH3-007].

        """
        if model not in _DISCREPANCY_MODELS:
            raise ValueError(f"model must be one of {_DISCREPANCY_MODELS}")
        design, prior, noise_matrix, total = self._matrices
        n = len(design)
        if model == "full":
            covariance = total
        elif model == "no_discrepancy":
            covariance = noise_matrix
        else:
            covariance = tuple(
                tuple(self.noise_level**2 if i == j else 0.0 for j in range(n))
                for i in range(n)
            )
        return gaussian_conditional_reference(
            design,
            self.data,
            covariance,
            self.prior_mean,
            prior,
            covariance_model=model,
        )


def _ar1_covariance(n: int, sd: float, correlation: float) -> Matrix:
    """Return ``sd^2 rho^|i - j|``.

    References:
        [EQ-INV-BENCH3-005].

    """
    return tuple(
        tuple(sd * sd * correlation ** abs(i - j) for j in range(n)) for i in range(n)
    )


def discrepancy_benchmark(
    size: int = 24,
    *,
    discrepancy_sd: float = 0.3,
    discrepancy_length: float = 0.25,
    kernel: str = "squared_exponential",
    discrepancy_nugget: float = 1e-8,
    noise_sd: float = 0.1,
    noise_correlation: float = 0.6,
    prior_sd: float = 10.0,
    theta: Sequence[float] = (1.0, -0.5),
    seed: int | RandomStream | None = 11,
) -> DiscrepancyBenchmark:
    """Return M01: ``y = theta_0 + theta_1 x + delta(x) + e`` with exact reference.

    Truth. The discrepancy is drawn as ``L_K z`` with ``L_K L_K' = K +
    nugget I`` (``truth`` stream, Cholesky route); the noise is drawn by the
    AR(1) recursion ``e_1 = sigma z_1``, ``e_i = rho e_{i-1} + sigma sqrt(1 -
    rho^2) z_i`` (``noise`` stream), an algorithm independent of the
    covariance matrix the reference uses [EQ-INV-BENCH3-005]. The nugget is part
    of the declared model (no silent jitter): the reference conditions on
    ``K + nugget I`` exactly.

    Args:
        size: Number of design points ``n >= 4`` (equally spaced on ``[0, 1]``).
        discrepancy_sd: Discrepancy standard deviation ``sigma_delta > 0``.
        discrepancy_length: Correlation length ``ell > 0``.
        kernel: ``"squared_exponential"`` or ``"matern52"``.
        discrepancy_nugget: Non-negative variance added to the discrepancy.
        noise_sd: Marginal noise standard deviation ``sigma_e > 0``.
        noise_correlation: AR(1) coefficient ``rho`` in ``[0, 1)``.
        prior_sd: Prior standard deviation of both parameters (``> 0``).
        theta: True ``(intercept, slope)``.
        seed: Root seed.

    Returns:
        :class:`DiscrepancyBenchmark`.

    Raises:
        TypeError: If a numeric argument is not real.
        ValueError: On out-of-range arguments, an unknown kernel, a theta
            of the wrong length, or a failed Cholesky factorization.

    References:
        [EQ-INV-BENCH3-005], [EQ-INV-BENCH3-006], [REF-KENNEDY-OHAGAN-2001],
        [REF-RASMUSSEN-WILLIAMS-2006, Section 4.2].

    """
    n = _positive_int(size, "size", 4)
    sd_delta = _real_positive(discrepancy_sd, "discrepancy_sd")
    ell = _real_positive(discrepancy_length, "discrepancy_length")
    nugget = _non_negative_real(discrepancy_nugget, "discrepancy_nugget")
    sd_noise = _real_positive(noise_sd, "noise_sd")
    rho = _finite_real(noise_correlation, "noise_correlation")
    if not 0.0 <= rho < 1.0:
        raise ValueError("noise_correlation must lie in [0, 1)")
    sd_prior = _real_positive(prior_sd, "prior_sd")
    if kernel not in DISCREPANCY_KERNELS:
        raise ValueError(f"kernel must be one of {DISCREPANCY_KERNELS}")
    true_theta = tuple(_finite_real(v, "theta") for v in theta)
    if len(true_theta) != 2:
        raise ValueError("theta must have two entries (intercept, slope)")
    points = tuple(i / (n - 1) for i in range(n))
    design = tuple((1.0, x) for x in points)
    k_matrix = tuple(
        tuple(
            discrepancy_kernel(kernel, (a,), (b,), (ell,), sd_delta**2)
            + (nugget if i == j else 0.0)
            for j, b in enumerate(points)
        )
        for i, a in enumerate(points)
    )
    noise_matrix = _ar1_covariance(n, sd_noise, rho)
    total = tuple(
        tuple(k_matrix[i][j] + noise_matrix[i][j] for j in range(n)) for i in range(n)
    )
    stream = resolve_stream(seed).child("M01")
    try:
        k_lower = cholesky(k_matrix)
    except ValueError as error:
        raise ValueError(
            "the discrepancy covariance is not positive definite; increase "
            "discrepancy_nugget or use the matern52 kernel"
        ) from error
    truth_generator = stream.child("truth").generator()
    z = [truth_generator.gauss(0.0, 1.0) for _ in range(n)]
    delta = tuple(
        math.fsum(k_lower[i][j] * z[j] for j in range(i + 1)) for i in range(n)
    )
    noise_generator = stream.child("noise").generator()
    innovations = [noise_generator.gauss(0.0, 1.0) for _ in range(n)]
    errors = [sd_noise * innovations[0]]
    scale = sd_noise * math.sqrt(1.0 - rho * rho)
    for i in range(1, n):
        errors.append(rho * errors[-1] + scale * innovations[i])
    simulator = tuple(a * true_theta[0] + b * true_theta[1] for a, b in design)
    exact = tuple(s + d for s, d in zip(simulator, delta, strict=True))
    data = tuple(b + e for b, e in zip(exact, errors, strict=True))
    prior_matrix = ((sd_prior**2, 0.0), (0.0, sd_prior**2))
    prior_mean = (0.0, 0.0)
    reference = gaussian_conditional_reference(
        design, data, total, prior_mean, prior_matrix, covariance_model="full"
    )
    return DiscrepancyBenchmark(
        identifier="M01",
        name=(
            f"GP discrepancy + AR(1) noise (n={n}, sigma_delta={sd_delta}, "
            f"ell={ell}, rho={rho})"
        ),
        operator=DenseOperator(design),
        data=data,
        noise=DenseCovariance(noise_matrix),
        noise_level=sd_noise,
        realized_noise_norm=math.hypot(*errors),
        truth=true_theta,
        exact_data=exact,
        same_discretization=True,
        prior_mean=prior_mean,
        prior_covariance=DenseCovariance(prior_matrix),
        discrepancy_covariance=DenseCovariance(k_matrix),
        total_covariance=DenseCovariance(total),
        discrepancy_truth=delta,
        noise_truth=tuple(errors),
        design_points=points,
        reference=reference,
        pathologies=(
            "model_discrepancy",
            "correlated_noise",
            "theta_delta_confounding",
        ),
        seeds=MappingProxyType({"truth": "M01/truth", "noise": "M01/noise"}),
        _matrices=(design, prior_matrix, noise_matrix, total),
    )


@dataclass(frozen=True, slots=True)
class CoverageStudy:
    """Measured interval coverage of the full and the misspecified references.

    Attributes:
        seeds: The predeclared seeds, one replicate each.
        level: Nominal central-interval level.
        parameter_names: ``("intercept", "slope")``.
        coverage_full: Fraction of replicates whose interval (exact conditional
            under the true covariance) contains the true parameter, per
            parameter.
        coverage_no_discrepancy: The same for the model that ignores the
            discrepancy but keeps the correlated noise.
        coverage_white_noise: The same for the model that ignores both.
        mcse: Binomial Monte Carlo standard error ``sqrt(level (1 - level) /
            N)`` of a coverage estimate at the nominal level.
        mean_width_full: Mean interval width per parameter (full model).
        mean_width_no_discrepancy: Mean width ignoring the discrepancy.
        mean_width_white_noise: Mean width ignoring both.
        replicates: Number of replicates ``N``.

    References:
        [EQ-INV-BENCH3-007].

    """

    seeds: tuple[int, ...]
    level: float
    parameter_names: tuple[str, ...]
    coverage_full: tuple[float, ...]
    coverage_no_discrepancy: tuple[float, ...]
    coverage_white_noise: tuple[float, ...]
    mcse: float
    mean_width_full: tuple[float, ...]
    mean_width_no_discrepancy: tuple[float, ...]
    mean_width_white_noise: tuple[float, ...]
    replicates: int


def discrepancy_coverage_study(
    seeds: Sequence[int] = COVERAGE_SEEDS_QUICK,
    *,
    level: float = 0.95,
    size: int = 24,
    discrepancy_sd: float = 0.3,
    discrepancy_length: float = 0.25,
    kernel: str = "squared_exponential",
    noise_sd: float = 0.1,
    noise_correlation: float = 0.6,
    prior_sd: float = 10.0,
) -> CoverageStudy:
    """Measure interval coverage over predeclared seeds, with and without ``delta``.

    One :func:`discrepancy_benchmark` per seed (fixed true ``theta``, fresh
    discrepancy and noise draws); the central marginal ``level`` interval of
    each assumed covariance model is checked against the true parameter
    [EQ-INV-BENCH3-007]. With a wide prior the exact conditional under the TRUE
    covariance is calibrated up to the (small) prior shrinkage; the models
    that ignore the discrepancy use a too-small covariance and their intervals
    are overconfident. Coverage values are reported, not asserted.

    Args:
        seeds: Distinct integer seeds (default :data:`COVERAGE_SEEDS_QUICK`).
        level: Nominal level in ``(0, 1)``.
        size: Design points per replicate.
        discrepancy_sd: ``sigma_delta``.
        discrepancy_length: ``ell``.
        kernel: Discrepancy kernel name.
        noise_sd: ``sigma_e``.
        noise_correlation: AR(1) coefficient.
        prior_sd: Prior standard deviation of the parameters.

    Returns:
        :class:`CoverageStudy`.

    Raises:
        TypeError: If a seed is not an integer.
        ValueError: If ``seeds`` is empty or has duplicates, or a benchmark
            argument is out of range.

    References:
        [EQ-INV-BENCH3-007], [REF-KENNEDY-OHAGAN-2001],
        [REF-BRYNJARSDOTTIR-OHAGAN-2014].

    """
    seed_list = tuple(seeds)
    if not seed_list:
        raise ValueError("seeds must be non-empty")
    for s in seed_list:
        if isinstance(s, bool) or not isinstance(s, int):
            raise TypeError("seeds must be integers")
    if len(set(seed_list)) != len(seed_list):
        raise ValueError("seeds must be distinct")
    nominal = _level(level)
    hits = {model: [0, 0] for model in _DISCREPANCY_MODELS}
    widths = {model: [0.0, 0.0] for model in _DISCREPANCY_MODELS}
    for s in seed_list:
        case = discrepancy_benchmark(
            size,
            discrepancy_sd=discrepancy_sd,
            discrepancy_length=discrepancy_length,
            kernel=kernel,
            noise_sd=noise_sd,
            noise_correlation=noise_correlation,
            prior_sd=prior_sd,
            seed=s,
        )
        for model in _DISCREPANCY_MODELS:
            intervals = case.conditional_reference(model).interval(nominal)
            for j, (lo, hi) in enumerate(intervals):
                hits[model][j] += 1 if lo <= case.truth[j] <= hi else 0
                widths[model][j] += hi - lo
    count = len(seed_list)

    def rate(model: str) -> tuple[float, ...]:
        """Return the coverage fraction per parameter for one model.

        References:
            [EQ-INV-BENCH3-007].

        """
        return tuple(h / count for h in hits[model])

    def width(model: str) -> tuple[float, ...]:
        """Return the mean interval width per parameter for one model.

        References:
            [EQ-INV-BENCH3-007].

        """
        return tuple(w / count for w in widths[model])

    return CoverageStudy(
        seeds=seed_list,
        level=nominal,
        parameter_names=("intercept", "slope"),
        coverage_full=rate("full"),
        coverage_no_discrepancy=rate("no_discrepancy"),
        coverage_white_noise=rate("white_noise"),
        mcse=math.sqrt(nominal * (1.0 - nominal) / count),
        mean_width_full=width("full"),
        mean_width_no_discrepancy=width("no_discrepancy"),
        mean_width_white_noise=width("white_noise"),
        replicates=count,
    )


# ---------------------------------------------------------------------------
# F01: Fredholm first-kind functional inversion
# ---------------------------------------------------------------------------

_KERNEL_NAMES = ("gaussian", "brownian")
_PROFILE_NAMES = ("smooth", "bump")


def fredholm_kernel(
    name: str, *, width: float = 0.1
) -> Callable[[float, float], float]:
    """Return the first-kind kernel ``k(s, t)`` on ``[0, 1]^2``.

    ``"gaussian"``: ``exp(-(s - t)^2 / (2 w^2))`` (severely ill-posed).
    ``"brownian"``: ``min(s, t)``, the Green function of ``-u''`` with
    ``u(0) = 0`` and ``u'(1) = 0``. Its integral operator is self-adjoint with
    the exact eigenpairs ``sigma_k = 1 / ((k - 1/2)^2 pi^2)``,
    ``sqrt(2) sin((k - 1/2) pi s)`` (mildly ill-posed) [EQ-INV-BENCH3-008].

    Args:
        name: ``"gaussian"`` or ``"brownian"``.
        width: Positive width ``w`` of the Gaussian kernel.

    Returns:
        A callable ``k(s, t)``.

    Raises:
        ValueError: If ``name`` is unknown or ``width`` is not positive.

    References:
        [EQ-INV-BENCH3-008], [REF-HANSEN-2008, Chapter 2].

    """
    if name not in _KERNEL_NAMES:
        raise ValueError(f"kernel must be one of {_KERNEL_NAMES}")
    w = _real_positive(width, "kernel_width")
    if name == "brownian":
        return lambda s, t: min(s, t)
    return lambda s, t: math.exp(-0.5 * ((s - t) / w) ** 2)


def functional_truth_profile(name: str) -> Callable[[float], float]:
    """Return the analytic truth ``f(t)`` of the F01 benchmark.

    ``"smooth"``: ``0.6 sin(2 pi t) + 0.8 exp(-((t - 0.35) / 0.12)^2)``.
    ``"bump"``: ``exp(-((t - 0.4) / 0.04)^2)``, narrow enough to be unresolved.

    Args:
        name: ``"smooth"`` or ``"bump"``.

    Returns:
        A callable ``f(t)``.

    Raises:
        ValueError: If ``name`` is unknown.

    References:
        [EQ-INV-BENCH3-008].

    """
    if name == "smooth":
        return lambda t: (
            0.6 * math.sin(2.0 * math.pi * t)
            + 0.8 * math.exp(-(((t - 0.35) / 0.12) ** 2))
        )
    if name == "bump":
        return lambda t: math.exp(-(((t - 0.4) / 0.04) ** 2))
    raise ValueError(f"truth must be one of {_PROFILE_NAMES}")


@dataclass(frozen=True, slots=True)
class FunctionalBenchmark:
    """One F01 instance: Fredholm data and a local-average target functional.

    Attributes:
        identifier: Catalog ID (``"F01"``).
        name: Short description.
        operator: Dense ``m x n`` operator mapping piecewise-constant cell
            values to ``g(s_i) = int k(s_i, t) f(t) dt``.
        data: Noisy data ``y``.
        noise: Noise covariance factor given to the solver.
        noise_level: Noise standard deviation (iid).
        realized_noise_norm: ``||y - b||`` of this draw.
        truth: Withheld cell averages of ``f`` on the solver's grid.
        exact_data: Withheld noise-free data ``b`` (fine quadrature of the
            analytic ``f`` unless ``same_discretization``).
        same_discretization: ``True`` when ``b = A x_true`` exactly.
        prior_covariance: ``None`` (no prior).
        singular_values: Dense singular values of ``A``.
        pathologies: Expected difficulties.
        functional: Row ``q`` with ``Q(x) = q' x``: the window average in the
            solver's representation (``sum q = 1``).
        window: ``(lower, upper)`` of the averaging window.
        functional_truth: Withheld exact window average of the analytic ``f``.
        functional_truth_coarse: Withheld ``q' x_true``.
        grid: Cell centres of the unknown.
        observation_points: The points ``s_i``.
        seeds: The stream paths used for the noise.

    References:
        [EQ-INV-BENCH3-008], [EQ-INV-BENCH3-009], [EQ-INV-BENCH3-010].

    """

    identifier: str
    name: str
    operator: DenseOperator
    data: Vector
    noise: ScaledIdentityCovariance
    noise_level: float
    realized_noise_norm: float
    truth: Vector
    exact_data: Vector
    same_discretization: bool
    prior_covariance: DenseCovariance | None
    singular_values: Vector | None
    pathologies: tuple[str, ...]
    functional: Vector
    window: tuple[float, float]
    functional_truth: float
    functional_truth_coarse: float
    grid: Vector
    observation_points: Vector
    seeds: Mapping[str, str] = field(default_factory=dict)


def _merged_breakpoints(edges: Sequence[float], extra: Sequence[float]) -> list[float]:
    """Return sorted breakpoints from ``edges`` plus ``extra``, deduplicated.

    References:
        [EQ-INV-BENCH3-008].

    """
    merged: list[float] = []
    for value in sorted([*edges, *extra]):
        if not merged or value - merged[-1] > 1e-12:
            merged.append(value)
    merged[0], merged[-1] = 0.0, 1.0
    return merged


def _quadrature(
    function: Callable[[float], float], breakpoints: Sequence[float]
) -> float:
    """Integrate over strictly increasing panels with the five-point Gauss rule.

    The panels may span any interval ``[b_0, b_P]``; they are mapped onto
    ``[0, 1]`` for :func:`~probstat.inverse.function_space._composite_rule`.

    References:
        [EQ-INV-BENCH3-008], [REF-GOLUB-WELSCH-1969].

    """
    lo, hi = breakpoints[0], breakpoints[-1]
    scaled = [(b - lo) / (hi - lo) for b in breakpoints]
    nodes, weights = _composite_rule(1.0, scaled)
    return (hi - lo) * math.fsum(
        w * function(lo + (hi - lo) * s) for s, w in zip(nodes, weights, strict=True)
    )


def _kernel_slice(
    kernel: Callable[[float, float], float], s: float
) -> Callable[[float], float]:
    """Return ``t -> k(s, t)`` with ``s`` bound.

    References:
        [EQ-INV-BENCH3-008].

    """
    return lambda t: kernel(s, t)


def _kernel_times(
    kernel: Callable[[float, float], float],
    s: float,
    profile: Callable[[float], float],
) -> Callable[[float], float]:
    """Return ``t -> k(s, t) f(t)`` with ``s`` bound.

    References:
        [EQ-INV-BENCH3-008].

    """
    return lambda t: kernel(s, t) * profile(t)


def functional_benchmark(
    size: int = 32,
    observations: int = 16,
    *,
    kernel: str = "gaussian",
    kernel_width: float = 0.1,
    truth: str = "smooth",
    window_centre: float = 0.4,
    window_halfwidth: float = 0.1,
    relative_noise: float = 1e-2,
    fine_data: bool = True,
    fine_panels: int = 400,
    seed: int | RandomStream | None = 11,
) -> FunctionalBenchmark:
    """Return F01: first-kind Fredholm data with a local-average target.

    ``y_i = int_0^1 k(s_i, t) f(t) dt + e_i`` at ``m`` points ``s_i = (i + 1/2)
    / m``. The solver represents ``f`` by ``n`` piecewise-constant cell values
    (:class:`~probstat.inverse.function_space.PiecewiseConstantBasis`); the
    operator rows are integrated with panels that include the cell edges and
    ``s_i``, so a kink of the kernel at ``s = t`` costs no accuracy. The target
    is the window average ``Q = (1 / |W|) int_W f`` over
    ``W = [c - h, c + h]``. With ``fine_data = True`` the data are the
    composite-Gauss quadrature of the ANALYTIC ``f`` on ``fine_panels``
    panels (independent of the solver's basis: ``same_discretization = False``,
    a genuine discretization error); with ``False`` they are ``A x_true``
    (inverse crime) [EQ-INV-BENCH3-008].

    Args:
        size: Number of cells ``n >= 8``.
        observations: Number of data points ``m >= 2``.
        kernel: ``"gaussian"`` or ``"brownian"``.
        kernel_width: Width of the Gaussian kernel.
        truth: ``"smooth"`` or ``"bump"``.
        window_centre: Window centre ``c``.
        window_halfwidth: Window half-width ``h > 0``; ``[c - h, c + h]`` must
            lie inside ``[0, 1]``.
        relative_noise: Noise level relative to ``||b|| / sqrt(m)``.
        fine_data: Generate data from the fine analytic quadrature.
        fine_panels: Panels ``>= 8`` of the fine data generator.
        seed: Root seed (only the noise is random; the truth is analytic).

    Returns:
        :class:`FunctionalBenchmark`.

    Raises:
        TypeError: If a numeric argument is not real.
        ValueError: On out-of-range sizes, an unknown kernel or truth, a window
            outside ``[0, 1]``, or a non-positive noise level.

    References:
        [EQ-INV-BENCH3-008], [EQ-INV-BENCH3-009], [REF-HANSEN-2008],
        [REF-GOLUB-WELSCH-1969].

    """
    n = _positive_int(size, "size", 8)
    m = _positive_int(observations, "observations", 2)
    panels = _positive_int(fine_panels, "fine_panels", 8)
    k = fredholm_kernel(kernel, width=kernel_width)
    profile = functional_truth_profile(truth)
    centre = _finite_real(window_centre, "window_centre")
    half = _real_positive(window_halfwidth, "window_halfwidth")
    lower, upper = centre - half, centre + half
    if lower < 0.0 or upper > 1.0:
        raise ValueError("the window must lie inside [0, 1]")
    basis = PiecewiseConstantBasis(n, 1.0)
    h = 1.0 / n
    edges = [i * h for i in range(n + 1)]
    points = tuple((i + 0.5) / m for i in range(m))
    breakpoints = _merged_breakpoints(edges, points)
    kernels = [_kernel_slice(k, s) for s in points]
    operator = basis.integral_operator(kernels, breakpoints)
    coarse = tuple(_quadrature(profile, [edges[i], edges[i + 1]]) / h for i in range(n))
    if fine_data:
        fine_edges = _merged_breakpoints(
            [i / panels for i in range(panels + 1)], points
        )
        exact = tuple(
            _quadrature(_kernel_times(k, s, profile), fine_edges) for s in points
        )
    else:
        exact = tuple(operator.matvec(coarse))
    q = tuple(
        max(0.0, min(edges[i + 1], upper) - max(edges[i], lower)) / (upper - lower)
        for i in range(n)
    )
    window_edges = [lower + (upper - lower) * i / 40 for i in range(41)]
    functional_truth = _quadrature(profile, window_edges) / (upper - lower)
    stream = resolve_stream(seed).child("F01")
    data, level, realized = _noise(exact, relative_noise, stream)
    sigma = spectral_diagnostics(operator).singular_values
    return FunctionalBenchmark(
        identifier="F01",
        name=(
            f"Fredholm {kernel} kernel, window average [{lower:.3g}, {upper:.3g}] "
            f"(n={n}, m={m})"
        ),
        operator=operator,
        data=data,
        noise=ScaledIdentityCovariance(m, level * level),
        noise_level=level,
        realized_noise_norm=realized,
        truth=coarse,
        exact_data=exact,
        same_discretization=not fine_data,
        prior_covariance=None,
        singular_values=sigma,
        pathologies=("ill_conditioned", "underdetermined", "unresolved_functional"),
        functional=q,
        window=(lower, upper),
        functional_truth=functional_truth,
        functional_truth_coarse=math.fsum(
            a * b for a, b in zip(q, coarse, strict=True)
        ),
        grid=tuple((i + 0.5) * h for i in range(n)),
        observation_points=points,
        seeds=MappingProxyType({"noise": "F01/noise"}),
    )


@dataclass(frozen=True, slots=True)
class FunctionalTradeoff:
    """One truncation level of the resolution versus noise trade-off.

    Attributes:
        rank: Truncation ``k``.
        resolved_fraction: ``||V_k V_k' q||^2 / ||q||^2``.
        standard_error: Noise standard deviation of the estimate,
            ``s sqrt(sum_{i<=k} (q'v_i / sigma_i)^2)``.
        bias_truth: ``-(q'x - sum_{i<=k} (q'v_i)(v_i'x))`` for a supplied truth
            ``x`` (benchmark-only), else ``None``.

    References:
        [EQ-INV-BENCH3-009].

    """

    rank: int
    resolved_fraction: float
    standard_error: float
    bias_truth: float | None


@dataclass(frozen=True, slots=True)
class FunctionalAnalysis:
    """Truncated-SVD estimate of a linear functional with its averaging kernel.

    Attributes:
        rank: Truncation ``k`` used.
        informed_modes: Number of leading modes with whitened singular value
            ``sigma_i / s >= threshold`` (data-informed modes at unit
            coefficient).
        singular_values: Singular values ``sigma_i`` of ``A`` (all modes).
        mode_coefficients: ``q' v_i`` for every mode.
        averaging_kernel: ``kappa = V_k V_k' q``: the noise-free estimate equals
            ``kappa' x`` for any truth ``x``.
        kernel_sum: ``sum_j kappa_j``.
        kernel_centre: ``sum_j s_j |kappa_j| / sum_j |kappa_j|`` on the cell
            centres.
        resolved_fraction: ``||kappa||^2 / ||q||^2`` (1 means the functional
            is in the data-resolved subspace).
        estimator_weights: ``a`` with estimate ``= a' y``.
        estimate: ``a' y``.
        standard_error: ``s ||a||``.
        bias_truth: ``kappa' x - q' x`` for a supplied truth (benchmark-only),
            else ``None``.
        tradeoff: One :class:`FunctionalTradeoff` per truncation ``1..r``.

    References:
        [EQ-INV-BENCH3-009].

    """

    rank: int
    informed_modes: int
    singular_values: Vector
    mode_coefficients: Vector
    averaging_kernel: Vector
    kernel_sum: float
    kernel_centre: float
    resolved_fraction: float
    estimator_weights: Vector
    estimate: float
    standard_error: float
    bias_truth: float | None
    tradeoff: tuple[FunctionalTradeoff, ...]


def functional_analysis(
    problem: FunctionalBenchmark,
    *,
    rank: int | None = None,
    threshold: float = 1.0,
    data: Sequence[float] | None = None,
    truth: Sequence[float] | None = None,
) -> FunctionalAnalysis:
    """Return the TSVD functional estimate, averaging kernel and trade-off.

    The SVD ``A = U S V'`` comes from
    :func:`~probstat.inverse.spectral.spectral_diagnostics`. The TSVD solution
    ``x_k = sum_{i<=k} (u_i'y / sigma_i) v_i`` gives ``Q_k = q' x_k = a' y`` with
    ``a = sum_{i<=k} (q'v_i / sigma_i) u_i`` and variance ``s^2 ||a||^2``; for
    noise-free data ``Q_k = kappa' x`` with the averaging kernel
    ``kappa = V_k V_k' q`` [EQ-INV-BENCH3-009]. By default only the operator,
    data and noise of ``problem`` are used; the truth enters only if passed
    explicitly (to report the bias) and is never an input of the estimate.

    Args:
        problem: A :class:`FunctionalBenchmark`.
        rank: Truncation ``k`` (default: the number of data-informed modes).
        threshold: Whitened-singular-value threshold for informed modes.
        data: Data to use instead of ``problem.data``.
        truth: Benchmark-only truth for the bias; ``None`` leaves it unset.

    Returns:
        :class:`FunctionalAnalysis`.

    Raises:
        ValueError: If no mode is data-informed, ``rank`` is out of range, or
            ``data`` / ``truth`` have the wrong length.

    References:
        [EQ-INV-BENCH3-009], [REF-HANSEN-2008, Section 2.4],
        [REF-GOLUB-VANLOAN-2013, Section 8.6].

    """
    thr = _real_positive(threshold, "threshold")
    y = problem.data if data is None else tuple(_finite_real(v, "data") for v in data)
    m, n = problem.operator.shape
    if len(y) != m:
        raise ValueError("data must have one entry per observation")
    x_truth: Vector | None = None
    if truth is not None:
        x_truth = tuple(_finite_real(v, "truth") for v in truth)
        if len(x_truth) != n:
            raise ValueError("truth must have one entry per unknown")
    s = problem.noise_level
    diagnostics = spectral_diagnostics(problem.operator, noise=problem.noise)
    sigma = tuple(w * s for w in diagnostics.singular_values)
    available = diagnostics.rank
    informed = 0
    for w in diagnostics.singular_values[:available]:
        if w < thr:
            break
        informed += 1
    if rank is None:
        if informed == 0:
            raise ValueError("no mode is data-informed at this threshold")
        k = informed
    else:
        k = _positive_int(rank, "rank", 1)
        if k > available:
            raise ValueError(f"rank must not exceed the numerical rank {available}")
    v = diagnostics.right_vectors
    u = diagnostics.left_vectors
    q = problem.functional
    qv = tuple(math.fsum(q[j] * v[j][i] for j in range(n)) for i in range(len(sigma)))
    q_norm_sq = math.fsum(a * a for a in q)
    vx: tuple[float, ...] | None = None
    if x_truth is not None:
        vx = tuple(
            math.fsum(v[j][i] * x_truth[j] for j in range(n)) for i in range(len(sigma))
        )
    q_truth = (
        None
        if x_truth is None
        else math.fsum(a * b for a, b in zip(q, x_truth, strict=True))
    )

    def bias_at(level: int) -> float | None:
        """Return ``kappa_k' x - q' x`` for the supplied truth, else ``None``.

        References:
            [EQ-INV-BENCH3-009].

        """
        if vx is None or q_truth is None:
            return None
        return math.fsum(qv[i] * vx[i] for i in range(level)) - q_truth

    tradeoff = tuple(
        FunctionalTradeoff(
            rank=level,
            resolved_fraction=math.fsum(c * c for c in qv[:level]) / q_norm_sq,
            standard_error=s
            * math.sqrt(math.fsum((qv[i] / sigma[i]) ** 2 for i in range(level))),
            bias_truth=bias_at(level),
        )
        for level in range(1, available + 1)
    )
    kappa = tuple(math.fsum(qv[i] * v[j][i] for i in range(k)) for j in range(n))
    weights = tuple(
        math.fsum((qv[i] / sigma[i]) * u[r][i] for i in range(k)) for r in range(m)
    )
    centres = problem.grid
    abs_mass = math.fsum(abs(c) for c in kappa)
    return FunctionalAnalysis(
        rank=k,
        informed_modes=informed,
        singular_values=sigma,
        mode_coefficients=qv,
        averaging_kernel=kappa,
        kernel_sum=math.fsum(kappa),
        kernel_centre=(
            math.fsum(c * abs(a) for c, a in zip(centres, kappa, strict=True))
            / abs_mass
            if abs_mass > 0.0
            else math.nan
        ),
        resolved_fraction=tradeoff[k - 1].resolved_fraction,
        estimator_weights=weights,
        estimate=math.fsum(a * b for a, b in zip(weights, y, strict=True)),
        standard_error=tradeoff[k - 1].standard_error,
        bias_truth=bias_at(k),
        tradeoff=tradeoff,
    )


# ---------------------------------------------------------------------------
# Catalog entries
# ---------------------------------------------------------------------------

CATALOG_BENCHMARK_CASES: tuple[BenchmarkCase, ...] = (
    BenchmarkCase(
        "S01",
        "Poisson image/count inversion",
        "inverse.benchmark_catalog.poisson_image_benchmark",
        "closed-form scalar MLE and Fisher information; library Poisson MLE / ML-EM",
        ("poisson_noise", "positivity", "zero_counts", "ill_conditioned"),
        ("maximum_likelihood_linear", "poisson_em"),
    ),
    BenchmarkCase(
        "M01",
        "GP discrepancy + correlated noise",
        "inverse.benchmark_catalog.discrepancy_benchmark",
        "exact marginalised Gaussian conditional; measured coverage",
        ("model_discrepancy", "correlated_noise", "theta_delta_confounding"),
        ("gaussian_conditional_reference", "linear_gaussian_posterior"),
    ),
    BenchmarkCase(
        "F01",
        "integral-kernel functional inversion",
        "inverse.benchmark_catalog.functional_benchmark",
        "closed-form Brownian-kernel spectrum; fine-quadrature data",
        ("ill_conditioned", "underdetermined", "unresolved_functional"),
        ("spectral_diagnostics", "functional_analysis"),
    ),
)
"""The S01, M01 and F01 catalog entries (roadmap 19)."""
