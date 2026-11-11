"""Stein control variates, control functionals, and kernel Stein discrepancy.

Every estimator here uses only the score ``u(x) = grad log pi(x)`` of the
target density ``pi``; the normalizing constant of ``pi`` is never needed. The
Langevin-Stein operator ``(A phi)(x) = Laplacian phi(x) + grad phi(x) . u(x)``
maps a smooth trial function ``phi`` to a function with zero expectation under
``pi`` whenever ``phi pi`` decays fast enough for integration by parts.

* Zero-variance control variates apply the operator to polynomials of degree
  one or two and fit the coefficients by least squares.
* Control functionals apply it to a reproducing-kernel Hilbert space, which
  gives the Stein reproducing kernel ``k0`` and a kernel-quadrature rule.
* The kernel Stein discrepancy measures how far a sample is from ``pi`` with
  the same kernel ``k0``.

Samples may come from any source (independent draws or a Markov chain). The
integrand and the score are each evaluated exactly once per sample point, and
the two counts are reported separately.

References:
    [REF-ASSARAF-CAFFAREL-1999], [REF-MIRA-SOLGI-IMPARATO-2013, Section 4],
    [REF-OATES-GIROLAMI-CHOPIN-2017, Sections 2.2-2.3],
    [REF-LIU-LEE-JORDAN-2016, Theorem 3.6], [REF-GORHAM-MACKEY-2017,
    Section 3].

"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from numbers import Real
from statistics import NormalDist
from typing import Literal

from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_finite_real,
    require_non_negative_integer,
    require_probability,
)
from probstat.inference import ConfidenceInterval, monte_carlo_standard_error
from probstat.linalg import Matrix, Vector, cholesky, cholesky_solve, householder_qr

BaseKernel = Literal["gaussian", "imq"]
StandardErrorMethod = Literal["iid", "mcmc"]
ZeroVarianceMethod = Literal["same-sample", "split-sample"]
LengthScaleRule = Literal["median", "cv"]

_MAXIMUM_REGULARIZATION = 1e-2
_DEFAULT_CV_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0, 4.0)


@dataclass(frozen=True, slots=True)
class ZeroVarianceResult:
    """A zero-variance control-variate estimate with fit diagnostics.

    Args:
        estimate: Control-variate estimate of ``E_pi[f]``.
        standard_error: Estimated standard deviation of ``estimate``.
        confidence_interval: Two-sided normal-approximation interval.
        raw_variance: Unbiased sample variance of the integrand values.
        adjusted_variance: Variance of the adjusted contributions
            ``f(x_i) - theta' h(x_i)`` (residual degrees of freedom
            ``n - p - 1`` for the same-sample fit, ``n - 1`` for the
            cross-fitted estimator).
        variance_reduction_factor: ``raw_variance / adjusted_variance``;
            infinity when the adjusted variance is exactly zero.
        coefficients: Fitted ``theta``; one tuple for the same-sample fit and
            one per fold for the split-sample estimator.
        degree: Polynomial degree of the trial functions (1 or 2).
        ridge: Ridge penalty ``lambda`` applied to ``||theta||^2``.
        method: ``"same-sample"`` or ``"split-sample"``.
        folds: Number of cross-fitting folds (1 for the same-sample fit).
        standard_error_method: ``"iid"`` or ``"mcmc"``.
        sample_size: Number of sample points ``n``.
        integrand_evaluations: Number of integrand calls.
        score_evaluations: Number of score (``grad log pi``) calls.

    References:
        [EQ-MC-STEIN-001], [EQ-MC-STEIN-002], [EQ-MC-STEIN-003],
        [REF-MIRA-SOLGI-IMPARATO-2013, Section 4].

    """

    estimate: float
    standard_error: float
    confidence_interval: ConfidenceInterval
    raw_variance: float
    adjusted_variance: float
    variance_reduction_factor: float
    coefficients: tuple[tuple[float, ...], ...]
    degree: int
    ridge: float
    method: ZeroVarianceMethod
    folds: int
    standard_error_method: StandardErrorMethod
    sample_size: int
    integrand_evaluations: int
    score_evaluations: int

    def __post_init__(self) -> None:
        """Validate estimate, uncertainty, and cost invariants.

        Raises:
            TypeError: If the interval has an invalid type.
            ValueError: If a numerical field lies outside its domain.

        References:
            [EQ-MC-STEIN-002], [REF-OWEN-2013, Section 8.9].

        """
        _validate_common(
            self.estimate,
            self.standard_error,
            self.confidence_interval,
            self.raw_variance,
            self.adjusted_variance,
            self.variance_reduction_factor,
            self.sample_size,
            self.integrand_evaluations,
            self.score_evaluations,
        )
        if self.degree not in (1, 2):
            raise ValueError("degree must be 1 or 2")
        require_non_negative_finite_real(self.ridge, name="ridge")

    @property
    def evaluations(self) -> int:
        """Return the integrand evaluations charged to the estimate.

        Score evaluations are reported separately in ``score_evaluations``.

        References:
            [EQ-MC-EFF-001], [REF-OWEN-2013, Chapter 2].

        """
        return self.integrand_evaluations


@dataclass(frozen=True, slots=True)
class ControlFunctionalResult:
    """A cross-fitted (split-sample) control-functional estimate.

    Args:
        estimate: Mean of the out-of-fold contributions ``f(x_i) - psi(x_i)``.
        standard_error: Estimated standard deviation of ``estimate``.
        confidence_interval: Two-sided normal-approximation interval.
        raw_variance: Unbiased sample variance of the integrand values.
        adjusted_variance: Unbiased sample variance of the contributions.
        variance_reduction_factor: ``raw_variance / adjusted_variance``;
            infinity when the adjusted variance is exactly zero.
        kernel: Base kernel name, ``"gaussian"`` or ``"imq"``.
        length_scales: Base-kernel scale used in each fold.
        regularizations: Relative diagonal shift used in each fold.
        length_scale_rule: ``"median"``, ``"cv"``, or ``"fixed"``.
        folds: Number of cross-fitting folds.
        standard_error_method: ``"iid"`` or ``"mcmc"``.
        sample_size: Number of sample points ``n``.
        integrand_evaluations: Number of integrand calls.
        score_evaluations: Number of score calls.

    References:
        [EQ-MC-STEIN-003], [EQ-MC-STEIN-005],
        [REF-OATES-GIROLAMI-CHOPIN-2017, Section 2.3.2].

    """

    estimate: float
    standard_error: float
    confidence_interval: ConfidenceInterval
    raw_variance: float
    adjusted_variance: float
    variance_reduction_factor: float
    kernel: BaseKernel
    length_scales: tuple[float, ...]
    regularizations: tuple[float, ...]
    length_scale_rule: str
    folds: int
    standard_error_method: StandardErrorMethod
    sample_size: int
    integrand_evaluations: int
    score_evaluations: int

    def __post_init__(self) -> None:
        """Validate estimate, uncertainty, and cost invariants.

        Raises:
            TypeError: If the interval has an invalid type.
            ValueError: If a numerical field lies outside its domain.

        References:
            [EQ-MC-STEIN-003], [REF-OATES-GIROLAMI-CHOPIN-2017, Section 2.3].

        """
        _validate_common(
            self.estimate,
            self.standard_error,
            self.confidence_interval,
            self.raw_variance,
            self.adjusted_variance,
            self.variance_reduction_factor,
            self.sample_size,
            self.integrand_evaluations,
            self.score_evaluations,
        )
        if len(self.length_scales) != self.folds or len(
            self.regularizations
        ) != self.folds:
            raise ValueError("one length scale and regularization per fold")

    @property
    def evaluations(self) -> int:
        """Return the integrand evaluations charged to the estimate.

        References:
            [EQ-MC-EFF-001], [REF-OWEN-2013, Chapter 2].

        """
        return self.integrand_evaluations


@dataclass(frozen=True, slots=True)
class ControlFunctionalWeightsResult:
    """A control-functional estimate written as a kernel-quadrature rule.

    Args:
        estimate: ``sum_i w_i f(x_i)``.
        weights: Quadrature weights ``w``; they sum to one and may be negative.
        stein_discrepancy: ``sqrt(w' K0 w)``, the kernel Stein discrepancy of
            the weighted sample and the worst-case integration error over the
            unit ball of the Stein reproducing-kernel Hilbert space.
        kernel: Base kernel name.
        length_scale: Base-kernel scale.
        regularization: Relative diagonal shift applied to ``K0``.
        length_scale_rule: ``"median"``, ``"cv"``, or ``"fixed"``.
        sample_size: Number of sample points ``n``.
        integrand_evaluations: Number of integrand calls.
        score_evaluations: Number of score calls.

    References:
        [EQ-MC-STEIN-005], [EQ-MC-STEIN-007],
        [REF-OATES-GIROLAMI-CHOPIN-2017, Section 2.3.4].

    """

    estimate: float
    weights: tuple[float, ...]
    stein_discrepancy: float
    kernel: BaseKernel
    length_scale: float
    regularization: float
    length_scale_rule: str
    sample_size: int
    integrand_evaluations: int
    score_evaluations: int

    def __post_init__(self) -> None:
        """Validate weights, discrepancy, and cost invariants.

        Raises:
            ValueError: If a field lies outside its domain.

        References:
            [EQ-MC-STEIN-005], [REF-OATES-GIROLAMI-CHOPIN-2017, Section 2.3.4].

        """
        require_finite_real(self.estimate, name="estimate")
        require_non_negative_finite_real(
            self.stein_discrepancy, name="stein_discrepancy"
        )
        if len(self.weights) != self.sample_size:
            raise ValueError("weights must have one entry per sample point")
        if not math.isclose(math.fsum(self.weights), 1.0, abs_tol=1e-8):
            raise ValueError("quadrature weights must sum to one")
        _validate_counts(
            self.sample_size, self.integrand_evaluations, self.score_evaluations
        )

    @property
    def evaluations(self) -> int:
        """Return the integrand evaluations charged to the estimate.

        References:
            [EQ-MC-EFF-001], [REF-OWEN-2013, Chapter 2].

        """
        return self.integrand_evaluations


@dataclass(frozen=True, slots=True)
class KernelSteinDiscrepancy:
    """Kernel Stein discrepancy of a sample against a target score.

    Args:
        discrepancy: ``sqrt`` of the V-statistic, the discrepancy of the
            empirical measure of the sample.
        v_statistic: ``n^-2 sum_(i,j) k0(x_i, x_j)`` (non-negative).
        u_statistic: ``(n(n-1))^-1 sum_(i != j) k0(x_i, x_j)``, an unbiased
            estimator of the squared discrepancy for independent draws; it
            may be negative.
        kernel: Base kernel name.
        length_scale: IMQ ``c`` or Gaussian length-scale.
        exponent: IMQ exponent ``beta`` (ignored for the Gaussian kernel).
        sample_size: Number of sample points ``n``.
        score_evaluations: Number of score calls.

    References:
        [EQ-MC-STEIN-007], [REF-GORHAM-MACKEY-2017, Proposition 2],
        [REF-LIU-LEE-JORDAN-2016, Section 4].

    """

    discrepancy: float
    v_statistic: float
    u_statistic: float
    kernel: BaseKernel
    length_scale: float
    exponent: float
    sample_size: int
    score_evaluations: int

    def __post_init__(self) -> None:
        """Validate the discrepancy statistics and counts.

        Raises:
            ValueError: If a field lies outside its domain.

        References:
            [EQ-MC-STEIN-007], [REF-GORHAM-MACKEY-2017, Proposition 2].

        """
        require_non_negative_finite_real(self.discrepancy, name="discrepancy")
        require_non_negative_finite_real(self.v_statistic, name="v_statistic")
        require_finite_real(self.u_statistic, name="u_statistic")
        _validate_counts(self.sample_size, self.sample_size, self.score_evaluations)


@dataclass(frozen=True, slots=True)
class _SteinSample:
    """Validated sample points with their integrand values and scores.

    Attributes:
        points: Sample points as ``d``-tuples.
        values: Integrand values, empty when no integrand was evaluated.
        scores: Score vectors ``grad log pi(x_i)``.
        dimension: Common dimension ``d``.

    References:
        [EQ-MC-STEIN-001], [REF-OATES-GIROLAMI-CHOPIN-2017, Section 2.1].

    """

    points: tuple[Vector, ...]
    values: Vector
    scores: tuple[Vector, ...]
    dimension: int


@dataclass(frozen=True, slots=True)
class _KernelFit:
    """Regularized fit of ``c + psi`` on a set of training points.

    Attributes:
        constant: Fitted constant ``c``.
        coefficients: Kernel coefficients ``a`` with ``psi = sum_j a_j k0``.
        ones_solve: ``A^-1 1`` for the regularized Gram matrix ``A``.
        regularization: Relative diagonal shift that made ``A`` factorable.

    References:
        [EQ-MC-STEIN-005], [REF-OATES-GIROLAMI-CHOPIN-2017, Section 2.3.4].

    """

    constant: float
    coefficients: Vector
    ones_solve: Vector
    regularization: float


def _validate_counts(sample_size: int, integrand: int, score: int) -> None:
    """Check that the sample size and evaluation counts are consistent.

    Args:
        sample_size: Number of sample points, at least two.
        integrand: Integrand evaluations, at least ``sample_size``.
        score: Score evaluations, at least ``sample_size``.

    Raises:
        ValueError: If a count is invalid.

    References:
        [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    size = require_non_negative_integer(sample_size, name="sample_size")
    if size < 2:
        raise ValueError("sample_size must be at least two")
    for name, value in (
        ("integrand_evaluations", integrand),
        ("score_evaluations", score),
    ):
        if require_non_negative_integer(value, name=name) < size:
            raise ValueError(f"{name} must not be below sample_size")


def _validate_common(
    estimate: float,
    standard_error: float,
    interval: ConfidenceInterval,
    raw_variance: float,
    adjusted_variance: float,
    factor: float,
    sample_size: int,
    integrand: int,
    score: int,
) -> None:
    """Check the fields shared by the control-variate result types.

    Args:
        estimate: Finite point estimate.
        standard_error: Non-negative standard error.
        interval: Confidence interval instance.
        raw_variance: Non-negative raw variance.
        adjusted_variance: Non-negative adjusted variance.
        factor: Positive (possibly infinite) variance-reduction factor.
        sample_size: Number of sample points.
        integrand: Integrand evaluations.
        score: Score evaluations.

    Raises:
        TypeError: If the interval has an invalid type.
        ValueError: If a field lies outside its domain.

    References:
        [REF-PROBSTAT-VALIDATION], [EQ-MC-STEIN-002].

    """
    require_finite_real(estimate, name="estimate")
    require_non_negative_finite_real(standard_error, name="standard_error")
    require_non_negative_finite_real(raw_variance, name="raw_variance")
    require_non_negative_finite_real(adjusted_variance, name="adjusted_variance")
    if not isinstance(interval, ConfidenceInterval):
        raise TypeError("confidence_interval must be a ConfidenceInterval")
    if math.isnan(factor) or factor < 0.0:
        raise ValueError("variance_reduction_factor must be non-negative")
    _validate_counts(sample_size, integrand, score)


def _coerce_point(value: object, index: int) -> Vector:
    """Convert one sample point to a finite tuple of floats.

    Args:
        value: A real number (one-dimensional point) or a sequence of reals.
        index: Position of the point, used in error messages.

    Returns:
        The point as a tuple.

    Raises:
        TypeError: If the point is not real or a sequence of reals.
        ValueError: If a coordinate is not finite or the point is empty.

    References:
        [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    if isinstance(value, Real) and not isinstance(value, bool):
        return (require_finite_real(value, name=f"samples[{index}]"),)
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(f"samples[{index}] must be a real number or a sequence")
    if not value:
        raise ValueError(f"samples[{index}] must not be empty")
    return tuple(
        require_finite_real(item, name=f"samples[{index}][{axis}]")
        for axis, item in enumerate(value)
    )


def _coerce_vector(value: object, dimension: int, *, name: str) -> Vector:
    """Convert a callable's return value to a finite ``d``-vector.

    Args:
        value: A real number (only when ``dimension == 1``) or a sequence.
        dimension: Required length ``d``.
        name: Label used in error messages.

    Returns:
        The vector as a tuple.

    Raises:
        TypeError: If the value is neither real nor a sequence of reals.
        ValueError: If the length is wrong or an entry is not finite.

    References:
        [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    if isinstance(value, Real) and not isinstance(value, bool):
        if dimension != 1:
            raise ValueError(f"{name} must have length {dimension}")
        return (require_finite_real(value, name=name),)
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a real number or a sequence")
    if len(value) != dimension:
        raise ValueError(f"{name} must have length {dimension}")
    return tuple(
        require_finite_real(item, name=f"{name}[{axis}]")
        for axis, item in enumerate(value)
    )


def _evaluate_sample(
    samples: Iterable[object],
    score: Callable[..., object],
    integrand: Callable[..., object] | None,
) -> _SteinSample:
    """Validate the points and evaluate the score and integrand once each.

    A one-dimensional sample given as plain numbers passes plain numbers to
    the callables; otherwise each point is passed as a tuple.

    Args:
        samples: At least two points of a common dimension.
        score: Callable returning ``grad log pi(x)``.
        integrand: Optional callable returning ``f(x)``.

    Returns:
        Points, values, scores, and dimension.

    Raises:
        TypeError: If a callable or a point has an invalid type.
        ValueError: If fewer than two points are given, dimensions differ, or
            a returned value is not finite.

    References:
        [REF-PROBSTAT-VALIDATION], [EQ-MC-STEIN-001].

    """
    if not callable(score):
        raise TypeError("score must be callable")
    if integrand is not None and not callable(integrand):
        raise TypeError("integrand must be callable")
    if isinstance(samples, str) or not isinstance(samples, Iterable):
        raise TypeError("samples must be an iterable of points")
    raw = tuple(samples)
    if len(raw) < 2:
        raise ValueError("at least two sample points are required")
    scalar = all(isinstance(item, Real) and not isinstance(item, bool) for item in raw)
    points = tuple(_coerce_point(item, index) for index, item in enumerate(raw))
    dimension = len(points[0])
    if any(len(point) != dimension for point in points):
        raise ValueError("all sample points must have the same dimension")
    scores: list[Vector] = []
    values: list[float] = []
    for index, point in enumerate(points):
        argument: object = point[0] if scalar else point
        scores.append(
            _coerce_vector(score(argument), dimension, name=f"score(samples[{index}])")
        )
        if integrand is not None:
            values.append(
                require_finite_real(
                    integrand(argument), name=f"integrand(samples[{index}])"
                )
            )
    return _SteinSample(points, tuple(values), tuple(scores), dimension)


def _require_confidence(confidence: float) -> float:
    """Validate a two-sided confidence level strictly inside ``(0, 1)``.

    Args:
        confidence: Nominal confidence level.

    Returns:
        The validated level.

    Raises:
        ValueError: If the level is not strictly between zero and one.

    References:
        [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    level = require_probability(confidence, name="confidence")
    if level in (0.0, 1.0):
        raise ValueError("confidence must be strictly between zero and one")
    return level


def _normal_interval(
    estimate: float, standard_error: float, confidence: float, label: str
) -> ConfidenceInterval:
    """Return the normal-approximation interval ``estimate +- z SE``.

    Args:
        estimate: Point estimate.
        standard_error: Non-negative standard error.
        confidence: Level strictly inside ``(0, 1)``.
        label: Method label stored in the interval.

    Returns:
        Two-sided interval (degenerate when ``standard_error`` is zero).

    References:
        [EQ-MC-SE-001], [REF-CASELLA-BERGER-2002, Chapter 9].

    """
    half_width = NormalDist().inv_cdf(0.5 + 0.5 * confidence) * standard_error
    return ConfidenceInterval(
        estimate - half_width, estimate + half_width, confidence, label
    )


def _standard_error(
    contributions: Vector, variance: float, method: StandardErrorMethod
) -> float:
    """Return the standard error of the mean of the adjusted contributions.

    ``"iid"`` uses ``sqrt(variance / n)``. ``"mcmc"`` uses the
    effective-sample-size standard error of the contribution series, scaled by
    ``sqrt(variance / s^2)`` so that a degrees-of-freedom correction in
    ``variance`` carries over.

    Args:
        contributions: Adjusted contributions in sampling order.
        variance: Variance estimate of one contribution.
        method: ``"iid"`` or ``"mcmc"``.

    Returns:
        Non-negative standard error.

    Raises:
        ValueError: If the method is unknown or the MCMC estimate is undefined.

    References:
        [EQ-MC-SE-001], [EQ-MCMC-MCSE-001], [REF-GEYER-1992].

    """
    if method not in ("iid", "mcmc"):
        raise ValueError("standard_error must be 'iid' or 'mcmc'")
    size = len(contributions)
    if variance == 0.0:
        return 0.0
    if method == "iid":
        return math.sqrt(variance / size)
    if size < 4:
        raise ValueError("the MCMC standard error needs at least four points")
    plain = statistics.variance(contributions)
    if plain == 0.0:
        return 0.0
    error = monte_carlo_standard_error([contributions])
    if not math.isfinite(error):
        raise ValueError("the MCMC standard error is undefined for this sample")
    return error * math.sqrt(variance / plain)


def _fold_slices(size: int, folds: int) -> tuple[range, ...]:
    """Partition ``0..size-1`` into contiguous, nearly equal folds.

    Contiguous blocks keep the folds of a Markov chain nearly independent.

    Args:
        size: Number of points.
        folds: Number of folds, at least two and at most ``size``.

    Returns:
        One index range per fold.

    References:
        [EQ-MC-STEIN-003], [REF-OATES-GIROLAMI-CHOPIN-2017, Section 2.2.1].

    """
    bounds = [fold * size // folds for fold in range(folds + 1)]
    return tuple(range(bounds[k], bounds[k + 1]) for k in range(folds))


def _require_folds(folds: object, size: int) -> int:
    """Validate a cross-fitting fold count.

    Args:
        folds: Requested number of folds.
        size: Number of sample points.

    Returns:
        The validated fold count.

    Raises:
        ValueError: If the count is below two or above the sample size.

    References:
        [REF-PROBSTAT-VALIDATION]. No equation is evaluated.

    """
    checked = require_non_negative_integer(folds, name="folds")
    if not 2 <= checked <= size:
        raise ValueError("folds must lie between two and the sample size")
    return checked


def _variance_ratio(raw: float, adjusted: float) -> float:
    """Return ``raw / adjusted``, infinite for a zero adjusted variance.

    Args:
        raw: Raw variance.
        adjusted: Adjusted variance.

    Returns:
        Variance-reduction factor.

    References:
        [EQ-MC-STEIN-002], [REF-OWEN-2013, Section 8.9].

    """
    if adjusted == 0.0:
        return math.inf if raw > 0.0 else 1.0
    return raw / adjusted


# ---------------------------------------------------------------------------
# Zero-variance (polynomial Stein) control variates
# ---------------------------------------------------------------------------


def stein_polynomial_controls(
    point: Sequence[float], score: Sequence[float], *, degree: int = 2
) -> Vector:
    """Return the Langevin-Stein operator applied to monomials of degree <= 2.

    With ``u = grad log pi(x)`` and ``(A phi) = Laplacian phi + grad phi . u``,
    the controls are ``A x_j = u_j`` for every coordinate and, for
    ``degree=2``, additionally ``A x_j^2 = 2 + 2 x_j u_j`` and
    ``A (x_j x_k) = x_k u_j + x_j u_k`` for ``j < k``, in the order
    ``(1), (1,1), (1,2), ..., (1,d), (2,2), ...``. Each control has
    expectation zero under ``pi`` when ``x_j x_k pi(x)`` vanishes at infinity.

    Args:
        point: Point ``x`` of dimension ``d``.
        score: Score ``u(x)`` of the same dimension.
        degree: Polynomial degree, 1 (``d`` controls) or 2
            (``d + d(d+1)/2`` controls).

    Returns:
        Control values ``h(x)``.

    Raises:
        ValueError: If the degree is not 1 or 2 or the lengths differ.

    Equation:
        [EQ-MC-STEIN-001].

    References:
        [REF-MIRA-SOLGI-IMPARATO-2013, Section 4, Eq. (8)],
        [REF-OATES-GIROLAMI-CHOPIN-2017, Section 2.2.2].

    """
    if isinstance(degree, bool) or degree not in (1, 2):
        raise ValueError("degree must be 1 or 2")
    x = tuple(float(value) for value in point)
    u = tuple(float(value) for value in score)
    if len(x) != len(u) or not x:
        raise ValueError("point and score must be non-empty and of equal length")
    controls = list(u)
    if degree == 2:
        for j in range(len(x)):
            for k in range(j, len(x)):
                if j == k:
                    controls.append(2.0 + 2.0 * x[j] * u[j])
                else:
                    controls.append(x[k] * u[j] + x[j] * u[k])
    return tuple(controls)


def _fit_control_coefficients(
    controls: Sequence[Vector], values: Sequence[float], ridge: float
) -> Vector:
    """Fit ``theta`` by (ridge) least squares of ``f`` on ``1`` and ``h``.

    Minimizes ``sum_i (f_i - b - theta' h_i)^2 + ridge ||theta||^2`` with an
    unpenalized intercept ``b`` through a Householder QR factorization of the
    design (augmented with ``sqrt(ridge) I`` rows when ``ridge > 0``).

    Args:
        controls: Control rows ``h_i`` of common length ``p``.
        values: Integrand values ``f_i``.
        ridge: Non-negative penalty.

    Returns:
        Fitted ``theta`` (the intercept is discarded).

    Raises:
        ValueError: If the unpenalized design is rank deficient or too small.

    Equation:
        [EQ-MC-STEIN-002], [EQ-LINALG-009].

    References:
        [REF-MIRA-SOLGI-IMPARATO-2013, Section 4],
        [REF-GOLUB-VANLOAN-2013, Section 5.3].

    """
    width = len(controls[0])
    if len(controls) < width + 2:
        raise ValueError(
            f"fitting {width} control coefficients needs at least {width + 2} points"
        )
    rows: list[Vector] = [(1.0, *row) for row in controls]
    target = list(values)
    if ridge > 0.0:
        root = math.sqrt(ridge)
        for column in range(width):
            rows.append(
                tuple(
                    root if index == column + 1 else 0.0
                    for index in range(width + 1)
                )
            )
            target.append(0.0)
    factorization = householder_qr(tuple(rows))
    if not factorization.full_rank:
        raise ValueError(
            "the control design is rank deficient; use more points, a lower "
            "degree, or ridge > 0"
        )
    return factorization.solve(tuple(target))[1:]


def estimate_zero_variance(
    samples: Iterable[object],
    integrand: Callable[..., object],
    score: Callable[..., object],
    *,
    degree: int = 2,
    ridge: float = 0.0,
    method: ZeroVarianceMethod = "same-sample",
    folds: int = 2,
    standard_error: StandardErrorMethod = "iid",
    confidence: float = 0.95,
) -> ZeroVarianceResult:
    """Estimate ``E_pi[f]`` with zero-variance (polynomial Stein) controls.

    The adjusted contribution is ``f(x_i) - theta' h(x_i)`` with ``h`` from
    :func:`stein_polynomial_controls`. ``"same-sample"`` fits ``theta`` on all
    points and reports the mean of the adjusted contributions with residual
    degrees of freedom ``n - p - 1``; its estimator has an ``O(1/n)`` bias
    from the fitted coefficients. ``"split-sample"`` cross-fits: the points
    are cut into ``folds`` contiguous blocks, ``theta`` is fitted without the
    block, and the block's contributions use that out-of-fold ``theta``. For
    independent draws every contribution then has expectation ``E_pi[f]``
    exactly. For a degree-2 basis and a Gaussian target, every polynomial of
    degree at most two is integrated exactly (zero adjusted variance).

    Args:
        samples: At least two points (numbers for ``d = 1`` or sequences),
            independent or from a Markov chain targeting ``pi``.
        integrand: ``f``; called once per point.
        score: ``grad log pi``; called once per point; returns a number when
            ``d = 1`` or a length-``d`` sequence.
        degree: Trial-polynomial degree, 1 or 2.
        ridge: Non-negative ridge penalty on ``||theta||^2``.
        method: ``"same-sample"`` or ``"split-sample"``.
        folds: Number of cross-fitting folds for ``"split-sample"``.
        standard_error: ``"iid"`` for ``sqrt(s^2/n)`` or ``"mcmc"`` for the
            effective-sample-size standard error of the contribution series.
        confidence: Nominal confidence level in ``(0, 1)``.

    Returns:
        Estimate, standard error, interval, variances, variance-reduction
        factor, coefficients, and separate integrand and score counts.

    Raises:
        TypeError: If a callable or a point has an invalid type.
        ValueError: If the degree, ridge, method, folds, or confidence is
            invalid, the sample is too small for the basis, or the design is
            rank deficient without a ridge penalty.

    Equation:
        [EQ-MC-STEIN-001], [EQ-MC-STEIN-002], [EQ-MC-STEIN-003].

    References:
        [REF-ASSARAF-CAFFAREL-1999], [REF-MIRA-SOLGI-IMPARATO-2013, Sections 4
        and 5], [REF-OATES-GIROLAMI-CHOPIN-2017, Section 2.2.1].

    """
    if isinstance(degree, bool) or degree not in (1, 2):
        raise ValueError("degree must be 1 or 2")
    penalty = require_non_negative_finite_real(ridge, name="ridge")
    if method not in ("same-sample", "split-sample"):
        raise ValueError("method must be 'same-sample' or 'split-sample'")
    if standard_error not in ("iid", "mcmc"):
        raise ValueError("standard_error must be 'iid' or 'mcmc'")
    level = _require_confidence(confidence)
    sample = _evaluate_sample(samples, score, integrand)
    size = len(sample.points)
    controls = tuple(
        stein_polynomial_controls(point, gradient, degree=degree)
        for point, gradient in zip(sample.points, sample.scores, strict=True)
    )
    width = len(controls[0])
    values = sample.values
    if method == "same-sample":
        theta = _fit_control_coefficients(controls, values, penalty)
        coefficients: tuple[Vector, ...] = (theta,)
        contributions = tuple(
            value - math.fsum(t * h for t, h in zip(theta, row, strict=True))
            for value, row in zip(values, controls, strict=True)
        )
        center = math.fsum(contributions) / size
        adjusted = math.fsum((c - center) ** 2 for c in contributions) / (
            size - width - 1
        )
        fold_count = 1
    else:
        fold_count = _require_folds(folds, size)
        fitted: list[Vector] = []
        adjusted_list = [0.0] * size
        for block in _fold_slices(size, fold_count):
            train = [index for index in range(size) if index not in block]
            theta = _fit_control_coefficients(
                [controls[index] for index in train],
                [values[index] for index in train],
                penalty,
            )
            fitted.append(theta)
            for index in block:
                adjusted_list[index] = values[index] - math.fsum(
                    t * h for t, h in zip(theta, controls[index], strict=True)
                )
        coefficients = tuple(fitted)
        contributions = tuple(adjusted_list)
        center = math.fsum(contributions) / size
        adjusted = statistics.variance(contributions)
    raw = statistics.variance(values)
    error = _standard_error(contributions, adjusted, standard_error)
    label = f"normal approximation, zero-variance control variates ({method})"
    return ZeroVarianceResult(
        estimate=center,
        standard_error=error,
        confidence_interval=_normal_interval(center, error, level, label),
        raw_variance=raw,
        adjusted_variance=adjusted,
        variance_reduction_factor=_variance_ratio(raw, adjusted),
        coefficients=coefficients,
        degree=degree,
        ridge=penalty,
        method=method,
        folds=fold_count,
        standard_error_method=standard_error,
        sample_size=size,
        integrand_evaluations=size,
        score_evaluations=size,
    )


# ---------------------------------------------------------------------------
# Stein reproducing kernel and control functionals
# ---------------------------------------------------------------------------


def _radial_derivatives(
    squared_distance: float, kernel: BaseKernel, length_scale: float, exponent: float
) -> tuple[float, float, float]:
    """Return ``g(z)``, ``g'(z)``, and ``g''(z)`` of a radial base kernel.

    The base kernel is ``k(x, y) = g(||x - y||^2)`` with
    ``g(z) = exp(-z / (2 l^2))`` (Gaussian) or ``g(z) = (l^2 + z)^beta``
    (inverse multiquadric with ``c = l``).

    Args:
        squared_distance: ``z = ||x - y||^2``.
        kernel: ``"gaussian"`` or ``"imq"``.
        length_scale: ``l > 0``.
        exponent: IMQ exponent ``beta``.

    Returns:
        The value and first two derivatives of ``g`` at ``z``.

    Equation:
        [EQ-MC-STEIN-004].

    References:
        [REF-GORHAM-MACKEY-2017, Section 3.2],
        [REF-OATES-GIROLAMI-CHOPIN-2017, Section 2.3.1].

    """
    if kernel == "gaussian":
        value = math.exp(-squared_distance / (2.0 * length_scale * length_scale))
        first = -value / (2.0 * length_scale * length_scale)
        return value, first, value / (4.0 * length_scale**4)
    base = length_scale * length_scale + squared_distance
    value = base**exponent
    first = exponent * value / base
    second = exponent * (exponent - 1.0) * value / (base * base)
    return value, first, second


def _stein_kernel_value(
    x: Vector,
    y: Vector,
    u_x: Vector,
    u_y: Vector,
    kernel: BaseKernel,
    length_scale: float,
    exponent: float,
) -> float:
    """Evaluate ``k0(x, y)`` for an already validated radial base kernel.

    With ``r = x - y``, ``z = ||r||^2``, and ``g`` from
    :func:`_radial_derivatives`,
    ``k0 = -4 g'' z - 2 d g' - 2 g' (u_x - u_y) . r + (u_x . u_y) g``.

    Args:
        x: First point.
        y: Second point.
        u_x: Score at ``x``.
        u_y: Score at ``y``.
        kernel: Base kernel name.
        length_scale: Base-kernel scale.
        exponent: IMQ exponent.

    Returns:
        The Stein-kernel value.

    Equation:
        [EQ-MC-STEIN-004].

    References:
        [REF-OATES-GIROLAMI-CHOPIN-2017, Theorem 1],
        [REF-GORHAM-MACKEY-2017, Proposition 2, Eq. (3)].

    """
    difference = tuple(a - b for a, b in zip(x, y, strict=True))
    squared = math.fsum(value * value for value in difference)
    value, first, second = _radial_derivatives(squared, kernel, length_scale, exponent)
    drift = math.fsum(
        (a - b) * r for a, b, r in zip(u_x, u_y, difference, strict=True)
    )
    alignment = math.fsum(a * b for a, b in zip(u_x, u_y, strict=True))
    return (
        -4.0 * second * squared
        - 2.0 * len(x) * first
        - 2.0 * first * drift
        + alignment * value
    )


def _require_kernel(
    kernel: object, length_scale: object, exponent: object
) -> tuple[BaseKernel, float, float]:
    """Validate a base-kernel name, scale, and IMQ exponent.

    Args:
        kernel: ``"gaussian"`` or ``"imq"``.
        length_scale: Positive finite scale.
        exponent: IMQ exponent in ``(-1, 0)``.

    Returns:
        The validated triple.

    Raises:
        ValueError: If a value lies outside its domain.

    References:
        [REF-GORHAM-MACKEY-2017, Theorem 8], [REF-PROBSTAT-VALIDATION].

    """
    if kernel not in ("gaussian", "imq"):
        raise ValueError("kernel must be 'gaussian' or 'imq'")
    scale = require_finite_real(length_scale, name="length_scale")
    if scale <= 0.0:
        raise ValueError("length_scale must be positive")
    beta = require_finite_real(exponent, name="exponent")
    if not -1.0 < beta < 0.0:
        raise ValueError("exponent must lie strictly between -1 and 0")
    name: BaseKernel = "gaussian" if kernel == "gaussian" else "imq"
    return name, scale, beta


def stein_kernel(
    x: Sequence[float] | float,
    y: Sequence[float] | float,
    score_x: Sequence[float] | float,
    score_y: Sequence[float] | float,
    *,
    kernel: BaseKernel = "gaussian",
    length_scale: float = 1.0,
    exponent: float = -0.5,
) -> float:
    """Return the Stein reproducing kernel ``k0(x, y)`` of a radial base kernel.

    ``k0(x, y) = div_x div_y k + u(x) . grad_y k + u(y) . grad_x k +
    u(x) . u(y) k`` with ``u = grad log pi``. For every fixed ``y``,
    ``E_pi[k0(X, y)] = 0`` under the integration-by-parts conditions of the
    source. The base kernel is Gaussian, ``exp(-||x-y||^2 / (2 l^2))``, or
    inverse multiquadric, ``(l^2 + ||x-y||^2)^beta``.

    Args:
        x: First point (number for ``d = 1`` or sequence).
        y: Second point of the same dimension.
        score_x: ``grad log pi(x)``.
        score_y: ``grad log pi(y)``.
        kernel: ``"gaussian"`` or ``"imq"``.
        length_scale: Gaussian length-scale or IMQ constant ``c``.
        exponent: IMQ exponent ``beta`` in ``(-1, 0)``.

    Returns:
        The kernel value.

    Raises:
        TypeError: If an argument has an invalid type.
        ValueError: If dimensions differ or a parameter is invalid.

    Equation:
        [EQ-MC-STEIN-004].

    References:
        [REF-OATES-GIROLAMI-CHOPIN-2017, Theorem 1],
        [REF-LIU-LEE-JORDAN-2016, Theorem 3.6],
        [REF-GORHAM-MACKEY-2017, Proposition 2].

    """
    name, scale, beta = _require_kernel(kernel, length_scale, exponent)
    first = _coerce_point(x, 0)
    dimension = len(first)
    second = _coerce_vector(y, dimension, name="y")
    u_x = _coerce_vector(score_x, dimension, name="score_x")
    u_y = _coerce_vector(score_y, dimension, name="score_y")
    return _stein_kernel_value(first, second, u_x, u_y, name, scale, beta)


def _stein_gram(
    sample: _SteinSample, kernel: BaseKernel, length_scale: float, exponent: float
) -> Matrix:
    """Return the symmetric Stein-kernel matrix ``K0`` of a sample.

    Args:
        sample: Points and scores.
        kernel: Base kernel name.
        length_scale: Base-kernel scale.
        exponent: IMQ exponent.

    Returns:
        ``n x n`` matrix with entries ``k0(x_i, x_j)``.

    Equation:
        [EQ-MC-STEIN-004].

    References:
        [REF-OATES-GIROLAMI-CHOPIN-2017, Section 2.3.1].

    """
    size = len(sample.points)
    rows = [[0.0] * size for _ in range(size)]
    for i in range(size):
        for j in range(i, size):
            value = _stein_kernel_value(
                sample.points[i],
                sample.points[j],
                sample.scores[i],
                sample.scores[j],
                kernel,
                length_scale,
                exponent,
            )
            rows[i][j] = value
            rows[j][i] = value
    return tuple(tuple(row) for row in rows)


def median_heuristic_length_scale(samples: Iterable[object]) -> float:
    """Return the median-heuristic length-scale ``sqrt(H_n / 2)``.

    ``H_n`` is the median of the squared pairwise distances
    ``||x_i - x_j||^2`` over ``i < j``.

    Args:
        samples: At least two points (numbers or sequences) of one dimension.

    Returns:
        Positive length-scale.

    Raises:
        TypeError: If a point has an invalid type.
        ValueError: If fewer than two points are given, dimensions differ, or
            all points coincide.

    Equation:
        [EQ-MC-STEIN-006].

    References:
        [REF-GARREAU-JITKRITTUM-KANAGAWA-2017, Section 2.2, Eq. (2)].

    """
    raw = tuple(samples)
    if len(raw) < 2:
        raise ValueError("at least two sample points are required")
    points = tuple(_coerce_point(item, index) for index, item in enumerate(raw))
    if any(len(point) != len(points[0]) for point in points):
        raise ValueError("all sample points must have the same dimension")
    return _median_length_scale(points)


def _median_length_scale(points: Sequence[Vector]) -> float:
    """Return ``sqrt(H_n / 2)`` for validated points.

    Args:
        points: Validated points of one dimension.

    Returns:
        Positive length-scale.

    Raises:
        ValueError: If all points coincide.

    Equation:
        [EQ-MC-STEIN-006].

    References:
        [REF-GARREAU-JITKRITTUM-KANAGAWA-2017, Section 2.2].

    """
    squared = [
        math.fsum((a - b) ** 2 for a, b in zip(points[i], points[j], strict=True))
        for i in range(len(points))
        for j in range(i + 1, len(points))
    ]
    median = statistics.median(squared)
    if median <= 0.0:
        raise ValueError("the median pairwise distance is zero")
    return math.sqrt(median / 2.0)


def _fit_kernel(
    gram: Matrix, rows: Sequence[int], values: Sequence[float], regularization: float
) -> _KernelFit:
    """Fit ``c + psi`` on the training rows by a regularized Cholesky solve.

    With ``A = K0 + lambda mean(diag K0) I``, ``c = 1' A^-1 f / 1' A^-1 1``
    and ``a = A^-1 (f - c 1)``. If ``A`` is not numerically positive definite,
    ``lambda`` is raised by powers of ten (starting from ``1e-12`` when it is
    zero) up to ``1e-2``.

    Args:
        gram: Full Stein-kernel matrix.
        rows: Training indices.
        values: Training integrand values aligned with ``rows``.
        regularization: Starting relative shift ``lambda >= 0``.

    Returns:
        Constant, coefficients, ``A^-1 1``, and the applied ``lambda``.

    Raises:
        ValueError: If no shift up to ``1e-2`` gives a Cholesky factor.

    Equation:
        [EQ-MC-STEIN-005], [EQ-LINALG-001].

    References:
        [REF-OATES-GIROLAMI-CHOPIN-2017, Sections 2.3.2 and 2.3.4],
        [REF-GOLUB-VANLOAN-2013, Section 4.2].

    """
    size = len(rows)
    sub = tuple(tuple(gram[i][j] for j in rows) for i in rows)
    mean_diagonal = math.fsum(sub[i][i] for i in range(size)) / size
    shift = regularization
    while True:
        shifted = tuple(
            tuple(
                value + (shift * mean_diagonal if i == j else 0.0)
                for j, value in enumerate(row)
            )
            for i, row in enumerate(sub)
        )
        try:
            lower = cholesky(shifted)
            break
        except ValueError:
            shift = 1e-12 if shift == 0.0 else 10.0 * shift
            if shift > _MAXIMUM_REGULARIZATION * (1.0 + 1e-9):
                raise ValueError(
                    "the Stein kernel matrix is not positive definite even with "
                    "regularization 1e-2; check the score or the length scale"
                ) from None
    ones_solve = cholesky_solve(lower, (1.0,) * size)
    value_solve = cholesky_solve(lower, tuple(values))
    constant = math.fsum(value_solve) / math.fsum(ones_solve)
    coefficients = tuple(
        a - constant * b for a, b in zip(value_solve, ones_solve, strict=True)
    )
    return _KernelFit(constant, coefficients, ones_solve, shift)


def _predict(gram: Matrix, index: int, rows: Sequence[int], fit: _KernelFit) -> float:
    """Return ``psi(x_index) = sum_j a_j k0(x_index, x_rows[j])``.

    Args:
        gram: Full Stein-kernel matrix.
        index: Evaluation index.
        rows: Training indices of the fit.
        fit: Kernel fit.

    Returns:
        The control-functional value (without the constant ``c``).

    Equation:
        [EQ-MC-STEIN-005].

    References:
        [REF-OATES-GIROLAMI-CHOPIN-2017, Section 2.3.2, Eq. (3)].

    """
    return math.fsum(
        a * gram[index][j] for a, j in zip(fit.coefficients, rows, strict=True)
    )


def _cross_validated_scale(
    sample: _SteinSample,
    rows: Sequence[int],
    kernel: BaseKernel,
    exponent: float,
    regularization: float,
    candidates: Sequence[float] | None,
    cv_folds: int,
) -> float:
    """Choose the length-scale minimizing the K-fold prediction error of ``f``.

    Every candidate is scored by ``sum_i (f_i - c - psi(x_i))^2`` over the
    held-out points of ``cv_folds`` contiguous folds of ``rows``; the default
    grid is the median heuristic of ``rows`` times ``0.25, 0.5, 1, 2, 4``.
    Ties keep the earlier (smaller) candidate.

    Args:
        sample: Points, values, and scores.
        rows: Indices available for selection (the training fold only).
        kernel: Base kernel name.
        exponent: IMQ exponent.
        regularization: Starting relative diagonal shift.
        candidates: Optional explicit positive candidates.
        cv_folds: Number of cross-validation folds.

    Returns:
        The selected length-scale.

    Raises:
        ValueError: If a candidate is invalid or too few rows are available.

    Equation:
        [EQ-MC-STEIN-006].

    References:
        [REF-OATES-GIROLAMI-CHOPIN-2017, Section 2.3.4],
        [REF-GARREAU-JITKRITTUM-KANAGAWA-2017, Section 2.2].

    """
    if candidates is None:
        anchor = _median_length_scale([sample.points[i] for i in rows])
        grid = tuple(anchor * factor for factor in _DEFAULT_CV_MULTIPLIERS)
    else:
        grid = tuple(
            require_finite_real(value, name="length_scale_candidates")
            for value in candidates
        )
        if not grid or any(value <= 0.0 for value in grid):
            raise ValueError("length_scale_candidates must be positive and non-empty")
    folds = _require_folds(cv_folds, len(rows))
    if len(rows) - len(rows) // folds - 1 < 2:
        raise ValueError("too few points for length-scale cross-validation")
    best_scale = grid[0]
    best_error = math.inf
    for scale in grid:
        gram = _stein_gram(sample, kernel, scale, exponent)
        error = 0.0
        for block in _fold_slices(len(rows), folds):
            held = [rows[i] for i in block]
            train = [rows[i] for i in range(len(rows)) if i not in block]
            fit = _fit_kernel(
                gram, train, [sample.values[i] for i in train], regularization
            )
            error += math.fsum(
                (sample.values[i] - fit.constant - _predict(gram, i, train, fit)) ** 2
                for i in held
            )
        if error < best_error:
            best_scale, best_error = scale, error
    return best_scale


def _resolve_scale(
    sample: _SteinSample,
    rows: Sequence[int],
    kernel: BaseKernel,
    exponent: float,
    regularization: float,
    length_scale: float | LengthScaleRule,
    candidates: Sequence[float] | None,
    cv_folds: int,
) -> tuple[float, str]:
    """Return the length-scale for a fit on ``rows`` and the rule's name.

    Args:
        sample: Points, values, and scores.
        rows: Training indices.
        kernel: Base kernel name.
        exponent: IMQ exponent.
        regularization: Starting relative diagonal shift.
        length_scale: Positive number, ``"median"``, or ``"cv"``.
        candidates: Optional cross-validation grid.
        cv_folds: Number of cross-validation folds.

    Returns:
        Length-scale and the rule label (``"fixed"``, ``"median"``, ``"cv"``).

    Raises:
        ValueError: If the rule or the number is invalid.

    Equation:
        [EQ-MC-STEIN-006].

    References:
        [REF-OATES-GIROLAMI-CHOPIN-2017, Section 2.3.4],
        [REF-GARREAU-JITKRITTUM-KANAGAWA-2017, Section 2.2].

    """
    if length_scale == "median":
        return _median_length_scale([sample.points[i] for i in rows]), "median"
    if length_scale == "cv":
        scale = _cross_validated_scale(
            sample, rows, kernel, exponent, regularization, candidates, cv_folds
        )
        return scale, "cv"
    if isinstance(length_scale, str):
        raise ValueError("length_scale must be positive, 'median', or 'cv'")
    value = require_finite_real(length_scale, name="length_scale")
    if value <= 0.0:
        raise ValueError("length_scale must be positive")
    return value, "fixed"


def _require_regularization(regularization: object) -> float:
    """Validate a starting relative regularization in ``[0, 1e-2]``.

    Args:
        regularization: Relative diagonal shift.

    Returns:
        The validated value.

    Raises:
        ValueError: If the value lies outside ``[0, 1e-2]``.

    References:
        [REF-OATES-GIROLAMI-CHOPIN-2017, Section 2.3.4],
        [REF-PROBSTAT-VALIDATION].

    """
    value = require_non_negative_finite_real(regularization, name="regularization")
    if value > _MAXIMUM_REGULARIZATION:
        raise ValueError("regularization must not exceed 1e-2")
    return value


def estimate_control_functional(
    samples: Iterable[object],
    integrand: Callable[..., object],
    score: Callable[..., object],
    *,
    kernel: BaseKernel = "gaussian",
    length_scale: float | LengthScaleRule = "cv",
    exponent: float = -0.5,
    regularization: float = 1e-8,
    folds: int = 2,
    length_scale_candidates: Sequence[float] | None = None,
    cv_folds: int = 5,
    standard_error: StandardErrorMethod = "iid",
    confidence: float = 0.95,
) -> ControlFunctionalResult:
    """Estimate ``E_pi[f]`` with the split-sample control-functional estimator.

    The points are cut into ``folds`` contiguous blocks. For each block, the
    other blocks fit ``f ~ c + psi`` with ``psi`` in the Stein
    reproducing-kernel Hilbert space (kernel ``k0``), and the block contributes
    ``f(x_i) - psi(x_i)``. Because ``E_pi[psi] = 0``, every contribution of an
    independent sample has expectation ``E_pi[f]`` regardless of the fit; with
    two folds this is the symmetrized form of the source's split estimator.
    The length-scale of each fold is chosen from that fold's training points
    only (median heuristic, cross-validation, or a fixed number), which keeps
    the estimator unbiased.

    Args:
        samples: At least four points (numbers for ``d = 1`` or sequences).
        integrand: ``f``; called once per point.
        score: ``grad log pi``; called once per point.
        kernel: Base kernel, ``"gaussian"`` or ``"imq"``.
        length_scale: Positive number, ``"median"``, or ``"cv"`` (default;
            cross-validation over the training points).
        exponent: IMQ exponent ``beta`` in ``(-1, 0)``.
        regularization: Starting relative diagonal shift in ``[0, 1e-2]``.
        folds: Number of cross-fitting folds.
        length_scale_candidates: Optional grid for ``"cv"``.
        cv_folds: Number of cross-validation folds for ``"cv"``.
        standard_error: ``"iid"`` or ``"mcmc"``.
        confidence: Nominal confidence level in ``(0, 1)``.

    Returns:
        Estimate, standard error, interval, variances, per-fold length-scales
        and regularizations, and separate integrand and score counts.

    Raises:
        TypeError: If a callable or a point has an invalid type.
        ValueError: If a parameter is invalid or the kernel matrix cannot be
            factorized.

    Equation:
        [EQ-MC-STEIN-003], [EQ-MC-STEIN-004], [EQ-MC-STEIN-005].

    References:
        [REF-OATES-GIROLAMI-CHOPIN-2017, Sections 2.2.1, 2.3.2 (Lemma 3,
        Eq. (3)) and 2.3.4].

    """
    name, _, beta = _require_kernel(kernel, 1.0, exponent)
    shift = _require_regularization(regularization)
    if standard_error not in ("iid", "mcmc"):
        raise ValueError("standard_error must be 'iid' or 'mcmc'")
    level = _require_confidence(confidence)
    sample = _evaluate_sample(samples, score, integrand)
    size = len(sample.points)
    fold_count = _require_folds(folds, size)
    contributions = [0.0] * size
    scales: list[float] = []
    shifts: list[float] = []
    rule = "fixed"
    for block in _fold_slices(size, fold_count):
        train = [index for index in range(size) if index not in block]
        if len(train) < 2:
            raise ValueError("each training fold needs at least two points")
        scale, rule = _resolve_scale(
            sample,
            train,
            name,
            beta,
            shift,
            length_scale,
            length_scale_candidates,
            cv_folds,
        )
        gram = _stein_gram(sample, name, scale, beta)
        fit = _fit_kernel(gram, train, [sample.values[i] for i in train], shift)
        scales.append(scale)
        shifts.append(fit.regularization)
        for index in block:
            contributions[index] = sample.values[index] - _predict(
                gram, index, train, fit
            )
    adjusted_values = tuple(contributions)
    center = math.fsum(adjusted_values) / size
    adjusted = statistics.variance(adjusted_values)
    raw = statistics.variance(sample.values)
    error = _standard_error(adjusted_values, adjusted, standard_error)
    label = "normal approximation, split-sample control functional"
    return ControlFunctionalResult(
        estimate=center,
        standard_error=error,
        confidence_interval=_normal_interval(center, error, level, label),
        raw_variance=raw,
        adjusted_variance=adjusted,
        variance_reduction_factor=_variance_ratio(raw, adjusted),
        kernel=name,
        length_scales=tuple(scales),
        regularizations=tuple(shifts),
        length_scale_rule=rule,
        folds=fold_count,
        standard_error_method=standard_error,
        sample_size=size,
        integrand_evaluations=size,
        score_evaluations=size,
    )


def estimate_control_functional_weights(
    samples: Iterable[object],
    integrand: Callable[..., object],
    score: Callable[..., object],
    *,
    kernel: BaseKernel = "gaussian",
    length_scale: float | LengthScaleRule = "cv",
    exponent: float = -0.5,
    regularization: float = 1e-8,
    length_scale_candidates: Sequence[float] | None = None,
    cv_folds: int = 5,
) -> ControlFunctionalWeightsResult:
    """Estimate ``E_pi[f]`` with the control-functional kernel-quadrature rule.

    Fitting ``f ~ c + psi`` on all ``n`` points and returning ``c`` gives
    ``c = sum_i w_i f(x_i)`` with ``w = A^-1 1 / (1' A^-1 1)`` and
    ``A = K0 + lambda mean(diag K0) I``. The weights sum to one, depend on the
    integrand only through a cross-validated length-scale, and may be
    negative. The rule carries no sampling standard error; the reported
    ``stein_discrepancy = sqrt(w' K0 w)`` bounds the integration error of
    every integrand ``f = c + psi`` by ``||psi|| * stein_discrepancy``.

    Args:
        samples: At least two points (numbers for ``d = 1`` or sequences).
        integrand: ``f``; called once per point.
        score: ``grad log pi``; called once per point.
        kernel: Base kernel, ``"gaussian"`` or ``"imq"``.
        length_scale: Positive number, ``"median"``, or ``"cv"`` (default;
            cross-validation over the training points).
        exponent: IMQ exponent ``beta`` in ``(-1, 0)``.
        regularization: Starting relative diagonal shift in ``[0, 1e-2]``.
        length_scale_candidates: Optional grid for ``"cv"``.
        cv_folds: Number of cross-validation folds for ``"cv"``.

    Returns:
        Estimate, weights, weighted Stein discrepancy, kernel settings, and
        separate integrand and score counts.

    Raises:
        TypeError: If a callable or a point has an invalid type.
        ValueError: If a parameter is invalid or the kernel matrix cannot be
            factorized.

    Equation:
        [EQ-MC-STEIN-004], [EQ-MC-STEIN-005], [EQ-MC-STEIN-007].

    References:
        [REF-OATES-GIROLAMI-CHOPIN-2017, Section 2.3.4],
        [REF-GORHAM-MACKEY-2017, Proposition 2].

    """
    name, _, beta = _require_kernel(kernel, 1.0, exponent)
    shift = _require_regularization(regularization)
    sample = _evaluate_sample(samples, score, integrand)
    size = len(sample.points)
    rows = list(range(size))
    scale, rule = _resolve_scale(
        sample,
        rows,
        name,
        beta,
        shift,
        length_scale,
        length_scale_candidates,
        cv_folds,
    )
    gram = _stein_gram(sample, name, scale, beta)
    fit = _fit_kernel(gram, rows, sample.values, shift)
    total = math.fsum(fit.ones_solve)
    weights = tuple(value / total for value in fit.ones_solve)
    quadratic = math.fsum(
        weights[i] * weights[j] * gram[i][j] for i in range(size) for j in range(size)
    )
    return ControlFunctionalWeightsResult(
        estimate=math.fsum(w * f for w, f in zip(weights, sample.values, strict=True)),
        weights=weights,
        stein_discrepancy=math.sqrt(max(0.0, quadratic)),
        kernel=name,
        length_scale=scale,
        regularization=fit.regularization,
        length_scale_rule=rule,
        sample_size=size,
        integrand_evaluations=size,
        score_evaluations=size,
    )


# ---------------------------------------------------------------------------
# Kernel Stein discrepancy
# ---------------------------------------------------------------------------


def kernel_stein_discrepancy(
    samples: Iterable[object],
    score: Callable[..., object],
    *,
    kernel: BaseKernel = "imq",
    length_scale: float = 1.0,
    exponent: float = -0.5,
) -> KernelSteinDiscrepancy:
    """Return the kernel Stein discrepancy of a sample against ``pi``.

    The V-statistic ``n^-2 sum_(i,j) k0(x_i, x_j)`` is the squared discrepancy
    of the empirical measure of the sample; the U-statistic drops the diagonal
    and is unbiased for independent draws from the sampling law. The default
    inverse multiquadric base kernel ``(1 + ||x - y||^2)^(-1/2)`` is the one
    shown to detect non-convergence for distantly dissipative targets
    (``beta`` in ``(-1, 0)``). Smaller values indicate a sample closer to
    ``pi``; the value is a diagnostic, not a test, and its scale depends on the
    kernel settings.

    Args:
        samples: At least two points (numbers for ``d = 1`` or sequences),
            from any source.
        score: ``grad log pi``; called once per point.
        kernel: Base kernel, ``"imq"`` (default) or ``"gaussian"``.
        length_scale: IMQ constant ``c`` or Gaussian length-scale.
        exponent: IMQ exponent ``beta`` in ``(-1, 0)``.

    Returns:
        Discrepancy, V- and U-statistics, kernel settings, and the score count.

    Raises:
        TypeError: If the score or a point has an invalid type.
        ValueError: If a parameter is invalid.

    Equation:
        [EQ-MC-STEIN-004], [EQ-MC-STEIN-007].

    References:
        [REF-GORHAM-MACKEY-2017, Section 3.1, Proposition 2 and Theorem 8],
        [REF-LIU-LEE-JORDAN-2016, Theorem 3.6 and Section 4, Eq. (14)].

    """
    name, scale, beta = _require_kernel(kernel, length_scale, exponent)
    sample = _evaluate_sample(samples, score, None)
    size = len(sample.points)
    gram = _stein_gram(sample, name, scale, beta)
    diagonal = math.fsum(gram[i][i] for i in range(size))
    off_diagonal = math.fsum(
        2.0 * gram[i][j] for i in range(size) for j in range(i + 1, size)
    )
    v_statistic = max(0.0, (diagonal + off_diagonal) / (size * size))
    return KernelSteinDiscrepancy(
        discrepancy=math.sqrt(v_statistic),
        v_statistic=v_statistic,
        u_statistic=off_diagonal / (size * (size - 1)),
        kernel=name,
        length_scale=scale,
        exponent=beta,
        sample_size=size,
        score_evaluations=size,
    )
