"""Nonlinear Bayesian optimal experimental design from callable models.

Experimental (added in 0.28).

The expected information gain (EIG) of a design ``d = (d_1, ..., d_m)`` (for
example sensor times or positions) for the model
``y_j = g(theta; d_j) + e_j``, ``e_j ~ N(0, s_j^2)`` independent, with a
Gaussian prior ``theta ~ N(mean, Sigma)`` is
``EIG(d) = E_{theta, y}[log p(y | theta, d) - log p(y | d)]``
([EQ-OED-EIG-001]). For a model that is linear in ``theta`` it equals
``1/2 log det(I + Sigma J' R^-1 J)`` ([EQ-OED-EIG-LIN-001]); for a nonlinear
model it must be estimated. This module wraps a plain callable ``g`` (no
:class:`~probstat.modeling.ProbabilisticModel` needed; compare
:mod:`probstat.design.bayesian`, which takes the model from a calibration
problem) and offers three estimators with stated bias:

* :func:`nested_monte_carlo_eig` ([EQ-DES-NOED-001]): outer draws
  ``(theta_n, y_n)`` and fresh inner draws for the evidence. Consistent, but
  the inner log-mean is biased low by Jensen's inequality, so the estimator
  is biased **upward**, ``O(1/M)`` in the inner size ``M``, with variance
  ``O(1/N)``.
* :func:`contrastive_lower_bound_eig` ([EQ-DES-NOED-002]): the ``N`` outer
  prior draws serve as the contrast set of every term (the prior contrastive
  bound of Foster et al.). Its expectation is a **lower bound** of the EIG
  and it never exceeds ``log N``; it costs ``N`` model evaluations per
  observation.
* :func:`laplace_eig` ([EQ-DES-NOED-003]): the Gauss-Newton Laplace
  approximation at the data-generating parameter,
  ``E_theta[1/2 log det(I + Sigma^(1/2)' J' R^-1 J Sigma^(1/2))]``. Exact for
  linear models; for nonlinear models it is an approximation with no
  guaranteed sign of the error, and it is cheap (``2 p`` evaluations per
  prior draw).

:func:`exhaustive_nonlinear_design` and :func:`greedy_nonlinear_design`
choose ``m`` points from a finite candidate set by maximizing one of the
estimators with common random numbers (the same seed for every design)
([EQ-DES-NOED-004]). Greedy choice is myopic and not guaranteed to match the
exhaustive optimum. The estimators carry Monte Carlo error; ranking
differences smaller than the reported standard errors are not meaningful.

Two synthetic illustration models are provided, :func:`exponential_decay_model`
([EQ-DES-NOED-005]) and :func:`source_location_model` ([EQ-DES-NOED-006]).
They are mathematical examples, not data from a real instrument.
"""

from __future__ import annotations

import itertools
import math
import random
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)
from probstat.linalg import Matrix, Vector, as_matrix, as_vector, cholesky, log_det_spd

ForwardFunction = Callable[[Vector, float], float]

_METHODS = ("laplace", "nested", "lower-bound")
_MAX_DESIGNS = 20_000


@dataclass(frozen=True, slots=True)
class NonlinearExperimentModel:
    """Scalar-observation model with a Gaussian prior and Gaussian noise.

    Observation ``j`` of design point ``d_j`` is
    ``y_j = forward(theta, d_j) + e_j``, ``e_j ~ N(0, noise_std(d_j)^2)``.

    Attributes:
        forward: ``(theta, design_point) -> float`` prediction in the
            observation's units.
        prior_mean: Prior mean of ``theta`` (length ``p``).
        prior_covariance: Symmetric positive definite prior covariance.
        noise_std: Constant noise standard deviation or a callable of the
            design point; positive.

    References:
        [EQ-OED-EIG-001], [REF-CHALONER-VERDINELLI-1995],
        [REF-HUAN-MARZOUK-2013].

    """

    forward: ForwardFunction
    prior_mean: Vector
    prior_covariance: Matrix
    noise_std: float | Callable[[float], float]

    def __post_init__(self) -> None:
        """Validate the prior and the noise specification.

        Raises:
            TypeError: If ``forward`` is not callable.
            ValueError: For a non-SPD covariance or a wrong shape.

        References:
            [EQ-OED-EIG-001]. No equation is evaluated.

        """
        if not callable(self.forward):
            raise TypeError("forward must be callable")
        mean = as_vector(self.prior_mean, name="prior_mean")
        cov = as_matrix(self.prior_covariance, name="prior_covariance")
        if len(cov) != len(mean) or any(len(row) != len(mean) for row in cov):
            raise ValueError("prior_covariance must be square of size len(prior_mean)")
        for i in range(len(mean)):
            for j in range(i):
                if abs(cov[i][j] - cov[j][i]) > 1e-12 * (
                    1.0 + abs(cov[i][j]) + abs(cov[j][i])
                ):
                    raise ValueError("prior_covariance must be symmetric")
        cholesky(cov)
        if not callable(self.noise_std):
            sd = require_finite_real(self.noise_std, name="noise_std")
            if sd <= 0.0:
                raise ValueError("noise_std must be positive")
        object.__setattr__(self, "prior_mean", mean)
        object.__setattr__(self, "prior_covariance", cov)

    @property
    def dimension(self) -> int:
        """Return the parameter dimension ``p``.

        Returns:
            ``len(prior_mean)``.

        References:
            [EQ-OED-EIG-001]. No equation is evaluated.

        """
        return len(self.prior_mean)

    def _factor(self) -> Matrix:
        """Return the lower Cholesky factor of the prior covariance.

        Returns:
            ``L`` with ``L L' = Sigma``.

        References:
            [EQ-DES-NOED-003].

        """
        return cholesky(self.prior_covariance)

    def _draw(self, factor: Matrix, generator: random.Random) -> Vector:
        """Draw one prior sample.

        Args:
            factor: Cholesky factor.
            generator: A ``random.Random``.

        Returns:
            ``mean + L z``.

        References:
            [EQ-DES-NOED-001].

        """
        z = [generator.gauss(0.0, 1.0) for _ in range(len(factor))]
        return tuple(
            self.prior_mean[i] + sum(factor[i][j] * z[j] for j in range(i + 1))
            for i in range(len(factor))
        )

    def _scales(self, design: Sequence[float]) -> Vector:
        """Return the noise standard deviation of each design point.

        Args:
            design: Design points.

        Returns:
            Positive standard deviations.

        Raises:
            ValueError: If a callable noise returns a non-positive value.

        References:
            [EQ-DES-NOED-001].

        """
        if callable(self.noise_std):
            out = tuple(
                require_finite_real(self.noise_std(d), name="noise_std(d)")
                for d in design
            )
            if min(out) <= 0.0:
                raise ValueError("noise_std must be positive at every design point")
            return out
        return (float(self.noise_std),) * len(design)

    def _predict(self, theta: Vector, design: Sequence[float]) -> Vector:
        """Return the noise-free predictions of a design.

        Args:
            theta: Parameter.
            design: Design points.

        Returns:
            ``(g(theta; d_j))_j``.

        Raises:
            ValueError: If the model returns a non-finite value.

        References:
            [EQ-DES-NOED-001].

        """
        out = []
        for d in design:
            value = require_finite_real(self.forward(theta, d), name="forward value")
            out.append(value)
        return tuple(out)


@dataclass(frozen=True, slots=True)
class EIGEstimate:
    """An expected-information-gain estimate with its error and cost.

    Attributes:
        value: Estimate in nats.
        standard_error: Monte Carlo standard error over the outer samples;
            it omits the bias, and for the contrastive bound it is optimistic
            because the terms share their contrast set.
        method: ``"nested"``, ``"lower-bound"`` or ``"laplace"``.
        bias: ``"upward"``, ``"downward"`` or ``"approximation"``.
        outer_samples: Outer (prior) samples.
        inner_samples: Inner samples per outer sample (0 if none).
        model_evaluations: Forward-model calls.

    References:
        [EQ-DES-NOED-001], [EQ-DES-NOED-002], [EQ-DES-NOED-003].

    """

    value: float
    standard_error: float
    method: str
    bias: str
    outer_samples: int
    inner_samples: int
    model_evaluations: int


def _check_design(model: NonlinearExperimentModel, design: Sequence[float]) -> Vector:
    """Validate a design.

    Args:
        model: Experiment model.
        design: Non-empty finite design points.

    Returns:
        The design as a tuple.

    Raises:
        TypeError: If ``model`` has the wrong type.
        ValueError: For an empty or non-finite design.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].
        No equation is evaluated.

    """
    if not isinstance(model, NonlinearExperimentModel):
        raise TypeError("model must be a NonlinearExperimentModel")
    points = as_vector(design, name="design")
    if not points:
        raise ValueError("design must contain at least one point")
    return points


def _log_mean_exp(values: Sequence[float]) -> float:
    """Return ``log(mean(exp(values)))`` stably.

    Args:
        values: Finite numbers.

    Returns:
        The log-mean-exp.

    References:
        [EQ-DES-NOED-001].

    """
    top = max(values)
    return top + math.log(math.fsum(math.exp(v - top) for v in values) / len(values))


def _log_kernel(y: Vector, g: Vector, inverse_scale: Vector) -> float:
    """Return the Gaussian log likelihood without its constant.

    Args:
        y: Data.
        g: Prediction.
        inverse_scale: ``1 / s_j``.

    Returns:
        ``-1/2 sum_j ((y_j - g_j) / s_j)^2``.

    References:
        [EQ-DES-NOED-001].

    """
    total = 0.0
    for a, b, w in zip(y, g, inverse_scale, strict=True):
        r = (a - b) * w
        total += r * r
    return -0.5 * total


def _mean_error(values: Sequence[float]) -> tuple[float, float]:
    """Return the mean and its standard error.

    Args:
        values: Samples.

    Returns:
        ``(mean, sd / sqrt(n))``.

    References:
        [EQ-DES-NOED-001].

    """
    mean = math.fsum(values) / len(values)
    return mean, statistics.stdev(values) / math.sqrt(len(values))


def nested_monte_carlo_eig(
    model: NonlinearExperimentModel,
    design: Sequence[float],
    *,
    outer: int = 500,
    inner: int = 100,
    seed: int | RandomStream | None = None,
) -> EIGEstimate:
    """Estimate the EIG by nested Monte Carlo with fresh inner samples.

    ``EIG_hat = (1/N) sum_n [log p(y_n | theta_n) - log((1/M) sum_m
    p(y_n | theta_nm))]`` with ``theta_n, theta_nm ~ p(theta)`` and
    ``y_n ~ p(y | theta_n)`` ([EQ-DES-NOED-001]). Since ``log`` is concave
    the inner log-mean is biased low and the estimator is biased **upward**,
    with bias ``O(1/M)`` and variance ``O(1/N)``; ``M`` of order
    ``sqrt(N)`` balances them. Cost ``N (1 + M)`` predictions of the design.

    Args:
        model: Experiment model.
        design: Design points.
        outer: Outer sample size ``N`` (at least 2).
        inner: Inner sample size ``M`` (at least 1).
        seed: Integer seed or stream; reuse it across designs for common
            random numbers.

    Returns:
        :class:`EIGEstimate` with ``bias="upward"``.

    Raises:
        TypeError: If ``model`` has the wrong type.
        ValueError: For invalid sizes or design, or a non-finite prediction.

    Equation:
        [EQ-DES-NOED-001], [EQ-OED-EIG-001], [EQ-OED-NMC-001].

    References:
        [REF-RYAN-2003], [REF-HUAN-MARZOUK-2013], [REF-RAINFORTH-ETAL-2018].

    """
    points = _check_design(model, design)
    n_outer = require_non_negative_integer(outer, name="outer")
    n_inner = require_non_negative_integer(inner, name="inner")
    if n_outer < 2 or n_inner < 1:
        raise ValueError("outer must be at least 2 and inner at least 1")
    stream = resolve_stream(seed)
    factor = model._factor()
    scales = model._scales(points)
    inverse = tuple(1.0 / s for s in scales)
    theta_rng = stream.child("theta").generator()
    noise_rng = stream.child("noise").generator()
    terms = []
    evaluations = 0
    for n in range(n_outer):
        theta = model._draw(factor, theta_rng)
        center = model._predict(theta, points)
        y = tuple(
            c + s * noise_rng.gauss(0.0, 1.0)
            for c, s in zip(center, scales, strict=True)
        )
        inner_rng = stream.child("inner", n).generator()
        contrast = [
            _log_kernel(
                y, model._predict(model._draw(factor, inner_rng), points), inverse
            )
            for _ in range(n_inner)
        ]
        terms.append(_log_kernel(y, center, inverse) - _log_mean_exp(contrast))
        evaluations += 1 + n_inner
    value, error = _mean_error(terms)
    return EIGEstimate(
        value, error, "nested", "upward", n_outer, n_inner, evaluations * len(points)
    )


def contrastive_lower_bound_eig(
    model: NonlinearExperimentModel,
    design: Sequence[float],
    *,
    samples: int = 500,
    seed: int | RandomStream | None = None,
) -> EIGEstimate:
    """Estimate a lower bound of the EIG by the prior contrastive bound.

    Draw ``theta_1, ..., theta_N`` from the prior and ``y_n ~ p(y |
    theta_n)``, then average ``log p(y_n | theta_n) - log((1/N) sum_m
    p(y_n | theta_m))`` ([EQ-DES-NOED-002]). Every term has the same
    expectation as the prior contrastive bound with ``N - 1`` contrastive
    samples, which is at most the EIG and at most ``log N``. The estimate is
    a **lower bound in expectation**; it can exceed the EIG in a single
    run by Monte Carlo error (reported by ``standard_error``, which is
    optimistic because the terms share the contrast set). Cost ``N``
    predictions of the design, ``N`` times cheaper than the nested estimator.

    Args:
        model: Experiment model.
        design: Design points.
        samples: Number of prior draws ``N`` (at least 2).
        seed: Integer seed or stream.

    Returns:
        :class:`EIGEstimate` with ``bias="downward"``.

    Raises:
        TypeError: If ``model`` has the wrong type.
        ValueError: For invalid sizes or design, or a non-finite prediction.

    Equation:
        [EQ-DES-NOED-002], [EQ-OED-NMC-001].

    References:
        [REF-FOSTER-ETAL-2020], [REF-HUAN-MARZOUK-2013].

    """
    points = _check_design(model, design)
    count = require_non_negative_integer(samples, name="samples")
    if count < 2:
        raise ValueError("samples must be at least 2")
    stream = resolve_stream(seed)
    factor = model._factor()
    scales = model._scales(points)
    inverse = tuple(1.0 / s for s in scales)
    theta_rng = stream.child("theta").generator()
    noise_rng = stream.child("noise").generator()
    centers = [
        model._predict(model._draw(factor, theta_rng), points) for _ in range(count)
    ]
    terms = []
    for center in centers:
        y = tuple(
            c + s * noise_rng.gauss(0.0, 1.0)
            for c, s in zip(center, scales, strict=True)
        )
        evidence = _log_mean_exp([_log_kernel(y, other, inverse) for other in centers])
        terms.append(_log_kernel(y, center, inverse) - evidence)
    value, error = _mean_error(terms)
    return EIGEstimate(
        value, error, "lower-bound", "downward", count, count, count * len(points)
    )


def laplace_eig(
    model: NonlinearExperimentModel,
    design: Sequence[float],
    *,
    samples: int = 500,
    step: float = 1e-6,
    seed: int | RandomStream | None = None,
) -> EIGEstimate:
    """Estimate the EIG by the Gauss-Newton Laplace approximation.

    For each prior draw ``theta_n`` the Jacobian ``J_n`` of the design
    predictions is taken by central differences with step
    ``step * max(1, |theta_i|)`` and the posterior is approximated by the
    Gaussian with precision ``Sigma^-1 + J' R^-1 J`` centered at ``theta_n``;
    the estimate averages ``1/2 log det(I + L' J' R^-1 J L)``, ``L L' =
    Sigma`` ([EQ-DES-NOED-003]). It is exact for models linear in ``theta``
    (``J`` constant) and an approximation otherwise: it ignores the data
    noise in the posterior mode and non-Gaussian posterior shape. The sign
    of its error is not known. Cost ``2 p`` predictions per draw.

    Args:
        model: Experiment model.
        design: Design points.
        samples: Number of prior draws (at least 2).
        step: Relative central-difference step.
        seed: Integer seed or stream.

    Returns:
        :class:`EIGEstimate` with ``bias="approximation"``.

    Raises:
        TypeError: If ``model`` has the wrong type.
        ValueError: For invalid sizes, step or design, or a non-finite
            prediction.

    Equation:
        [EQ-DES-NOED-003], [EQ-OED-LAP-001], [EQ-OED-EIG-LIN-001].

    References:
        [REF-TIERNEY-KADANE-1986], [REF-HUAN-MARZOUK-2013].

    """
    points = _check_design(model, design)
    count = require_non_negative_integer(samples, name="samples")
    if count < 2:
        raise ValueError("samples must be at least 2")
    h = require_finite_real(step, name="step")
    if h <= 0.0:
        raise ValueError("step must be positive")
    stream = resolve_stream(seed)
    factor = model._factor()
    scales = model._scales(points)
    p = model.dimension
    m = len(points)
    theta_rng = stream.child("theta").generator()
    terms = []
    for _ in range(count):
        theta = model._draw(factor, theta_rng)
        jac = [[0.0] * p for _ in range(m)]
        for i in range(p):
            delta = h * max(1.0, abs(theta[i]))
            up = (*theta[:i], theta[i] + delta, *theta[i + 1 :])
            dn = (*theta[:i], theta[i] - delta, *theta[i + 1 :])
            g_up = model._predict(up, points)
            g_dn = model._predict(dn, points)
            for j in range(m):
                jac[j][i] = (g_up[j] - g_dn[j]) / (2.0 * delta) / scales[j]
        # B = (R^-1/2 J) L, gram = B' B
        b = [
            [math.fsum(jac[j][k] * factor[k][i] for k in range(p)) for i in range(p)]
            for j in range(m)
        ]
        gram = tuple(
            tuple(
                math.fsum(b[j][a] * b[j][c] for j in range(m))
                + (1.0 if a == c else 0.0)
                for c in range(p)
            )
            for a in range(p)
        )
        terms.append(0.5 * log_det_spd(gram))
    value, error = _mean_error(terms)
    return EIGEstimate(
        value, error, "laplace", "approximation", count, 0, count * (2 * p) * m
    )


def estimate_eig(
    model: NonlinearExperimentModel,
    design: Sequence[float],
    *,
    method: str = "laplace",
    samples: int = 500,
    inner: int = 100,
    seed: int | RandomStream | None = None,
) -> EIGEstimate:
    """Estimate the EIG of a design by the chosen method.

    Args:
        model: Experiment model.
        design: Design points.
        method: ``"laplace"``, ``"nested"`` or ``"lower-bound"``.
        samples: Outer / prior sample size.
        inner: Inner size (``"nested"`` only).
        seed: Integer seed or stream.

    Returns:
        :class:`EIGEstimate`.

    Raises:
        ValueError: For an unknown method or invalid inputs.

    Equation:
        [EQ-DES-NOED-001], [EQ-DES-NOED-002], [EQ-DES-NOED-003].

    References:
        [REF-HUAN-MARZOUK-2013], [REF-FOSTER-ETAL-2020].

    """
    if method == "laplace":
        return laplace_eig(model, design, samples=samples, seed=seed)
    if method == "nested":
        return nested_monte_carlo_eig(
            model, design, outer=samples, inner=inner, seed=seed
        )
    if method == "lower-bound":
        return contrastive_lower_bound_eig(model, design, samples=samples, seed=seed)
    raise ValueError(f"method must be one of {_METHODS}")


@dataclass(frozen=True, slots=True)
class NonlinearDesignResult:
    """Outcome of a design search.

    Attributes:
        strategy: ``"exhaustive"`` or ``"greedy"``.
        method: EIG estimator used.
        design: Chosen design points (greedy: in selection order;
            exhaustive: in candidate order).
        value: Estimated EIG of ``design``.
        standard_error: Its Monte Carlo standard error.
        ranking: ``(design, estimate)`` pairs, best first (exhaustive: all
            designs; greedy: the winning design at every step).
        model_evaluations: Total forward-model calls.

    References:
        [EQ-DES-NOED-004].

    """

    strategy: str
    method: str
    design: Vector
    value: float
    standard_error: float
    ranking: tuple[tuple[Vector, EIGEstimate], ...]
    model_evaluations: int


def _check_candidates(candidates: Sequence[float]) -> Vector:
    """Validate the candidate points.

    Args:
        candidates: Finite points.

    Returns:
        Candidates as a tuple.

    Raises:
        ValueError: If empty or containing duplicates.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].
        No equation is evaluated.

    """
    points = as_vector(candidates, name="candidates")
    if not points:
        raise ValueError("candidates must not be empty")
    if len(set(points)) != len(points):
        raise ValueError("candidates must be distinct")
    return points


def exhaustive_nonlinear_design(
    model: NonlinearExperimentModel,
    candidates: Sequence[float],
    size: int,
    *,
    method: str = "laplace",
    samples: int = 300,
    inner: int = 50,
    allow_repeats: bool = False,
    seed: int | RandomStream | None = 0,
) -> NonlinearDesignResult:
    """Evaluate every design of ``size`` candidates and return the best.

    All designs are scored with the same seed (common random numbers), so
    differences between designs are less noisy than the individual
    estimates ([EQ-DES-NOED-004]). The number of designs is
    ``C(n, size)`` (``C(n + size - 1, size)`` with repeats) and must not
    exceed 20000.

    Args:
        model: Experiment model.
        candidates: Distinct candidate design points.
        size: Number of observations in the design.
        method: ``"laplace"``, ``"nested"`` or ``"lower-bound"``.
        samples: Outer / prior sample size.
        inner: Inner size (nested only).
        allow_repeats: Allow the same candidate several times.
        seed: Integer seed or stream shared by all designs.

    Returns:
        :class:`NonlinearDesignResult` with the full ranking.

    Raises:
        ValueError: For invalid inputs or too many designs.

    Equation:
        [EQ-DES-NOED-004].

    References:
        [REF-HUAN-MARZOUK-2013], [REF-CHALONER-VERDINELLI-1995].

    """
    points = _check_candidates(candidates)
    count = require_non_negative_integer(size, name="size")
    if count < 1:
        raise ValueError("size must be at least 1")
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}")
    stream = resolve_stream(seed)
    combos = (
        itertools.combinations_with_replacement(points, count)
        if allow_repeats
        else itertools.combinations(points, count)
    )
    total = (
        math.comb(len(points) + count - 1, count)
        if allow_repeats
        else math.comb(len(points), count)
    )
    if total == 0:
        raise ValueError("size exceeds the number of candidates")
    if total > _MAX_DESIGNS:
        raise ValueError("too many designs; use greedy_nonlinear_design")
    scored = [
        (
            combo,
            estimate_eig(
                model, combo, method=method, samples=samples, inner=inner, seed=stream
            ),
        )
        for combo in combos
    ]
    scored.sort(key=lambda item: -item[1].value)
    best_design, best = scored[0]
    return NonlinearDesignResult(
        "exhaustive",
        method,
        best_design,
        best.value,
        best.standard_error,
        tuple(scored),
        sum(item[1].model_evaluations for item in scored),
    )


def greedy_nonlinear_design(
    model: NonlinearExperimentModel,
    candidates: Sequence[float],
    size: int,
    *,
    method: str = "laplace",
    samples: int = 300,
    inner: int = 50,
    allow_repeats: bool = False,
    seed: int | RandomStream | None = 0,
) -> NonlinearDesignResult:
    """Choose design points one at a time by maximal estimated EIG.

    Step ``k`` adds the candidate that maximizes the estimated EIG of the
    enlarged design, with common random numbers across candidates
    ([EQ-DES-NOED-004]). The result is myopic: it need not equal the
    exhaustive optimum.

    Args:
        model: Experiment model.
        candidates: Distinct candidate design points.
        size: Number of observations to select.
        method: ``"laplace"``, ``"nested"`` or ``"lower-bound"``.
        samples: Outer / prior sample size.
        inner: Inner size (nested only).
        allow_repeats: Allow the same candidate several times.
        seed: Integer seed or stream shared by all evaluations.

    Returns:
        :class:`NonlinearDesignResult`; ``ranking`` holds the design after
        each step.

    Raises:
        ValueError: For invalid inputs or more points than candidates
            without repeats.

    Equation:
        [EQ-DES-NOED-004], [EQ-OED-GREEDY-001].

    References:
        [REF-HUAN-MARZOUK-2013], [REF-CHALONER-VERDINELLI-1995].

    """
    points = _check_candidates(candidates)
    count = require_non_negative_integer(size, name="size")
    if count < 1:
        raise ValueError("size must be at least 1")
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}")
    if not allow_repeats and count > len(points):
        raise ValueError("size exceeds the number of candidates")
    stream = resolve_stream(seed)
    chosen: list[float] = []
    steps: list[tuple[Vector, EIGEstimate]] = []
    evaluations = 0
    for _ in range(count):
        best: tuple[float, EIGEstimate] | None = None
        for candidate in points:
            if not allow_repeats and candidate in chosen:
                continue
            estimate = estimate_eig(
                model,
                (*chosen, candidate),
                method=method,
                samples=samples,
                inner=inner,
                seed=stream,
            )
            evaluations += estimate.model_evaluations
            if best is None or estimate.value > best[1].value:
                best = (candidate, estimate)
        assert best is not None
        chosen.append(best[0])
        steps.append((tuple(chosen), best[1]))
    final = steps[-1][1]
    return NonlinearDesignResult(
        "greedy",
        method,
        tuple(chosen),
        final.value,
        final.standard_error,
        tuple(steps),
        evaluations,
    )


def exponential_decay_model(
    *,
    amplitude: float,
    log_rate_mean: float,
    log_rate_std: float,
    noise_std: float,
) -> NonlinearExperimentModel:
    """Build the synthetic exponential-decay experiment (one parameter).

    The parameter is ``theta = log lambda`` with ``lambda`` the decay rate in
    1/s, prior ``N(log_rate_mean, log_rate_std^2)``; the design point is the
    sensor time ``t`` in s and the prediction is
    ``g = A exp(-exp(theta) t)`` in the units of ``A`` ([EQ-DES-NOED-005]).
    At ``t = 0`` the prediction does not depend on ``theta`` (zero
    information). A mathematical illustration, not data from an instrument.

    Args:
        amplitude: Initial amplitude ``A`` (nonzero).
        log_rate_mean: Prior mean of ``log lambda``.
        log_rate_std: Prior standard deviation (positive).
        noise_std: Observation noise standard deviation (positive).

    Returns:
        :class:`NonlinearExperimentModel`.

    Raises:
        ValueError: For a zero amplitude or non-positive scales.

    Equation:
        [EQ-DES-NOED-005].

    References:
        [REF-HUAN-MARZOUK-2013].

    """
    a = require_finite_real(amplitude, name="amplitude")
    if a == 0.0:
        raise ValueError("amplitude must be nonzero")
    sd = require_finite_real(log_rate_std, name="log_rate_std")
    if sd <= 0.0:
        raise ValueError("log_rate_std must be positive")

    def forward(theta: Vector, t: float) -> float:
        """Return ``A exp(-exp(theta_0) t)``.

        References:
            [EQ-DES-NOED-005].

        """
        return a * math.exp(-math.exp(theta[0]) * t)

    return NonlinearExperimentModel(
        forward,
        (require_finite_real(log_rate_mean, name="log_rate_mean"),),
        ((sd * sd,),),
        noise_std,
    )


def source_location_model(
    *,
    strength: float,
    height: float,
    prior_mean: float,
    prior_std: float,
    noise_std: float,
) -> NonlinearExperimentModel:
    """Build the synthetic one-dimensional source-location experiment.

    The parameter is the source position ``theta`` in m with prior
    ``N(prior_mean, prior_std^2)``; the design point is the sensor position
    ``x`` in m and the prediction is the inverse-square intensity
    ``g = q / ((x - theta)^2 + h^2)`` of a point source at height ``h`` in
    m above the sensor line, in the units of ``q`` ([EQ-DES-NOED-006]). The
    response is nonmonotone in ``theta``, so the posterior can be bimodal for
    a single sensor. A mathematical illustration, not data from an instrument.

    Args:
        strength: Source strength ``q`` (nonzero).
        height: Source height ``h`` in m (positive).
        prior_mean: Prior mean of the position in m.
        prior_std: Prior standard deviation in m (positive).
        noise_std: Observation noise standard deviation (positive).

    Returns:
        :class:`NonlinearExperimentModel`.

    Raises:
        ValueError: For a zero strength or non-positive scales.

    Equation:
        [EQ-DES-NOED-006].

    References:
        [REF-HUAN-MARZOUK-2013].

    """
    q = require_finite_real(strength, name="strength")
    h = require_finite_real(height, name="height")
    sd = require_finite_real(prior_std, name="prior_std")
    if q == 0.0 or h <= 0.0 or sd <= 0.0:
        raise ValueError("strength must be nonzero; height and prior_std positive")

    def forward(theta: Vector, x: float) -> float:
        """Return ``q / ((x - theta_0)^2 + h^2)``.

        References:
            [EQ-DES-NOED-006].

        """
        return q / ((x - theta[0]) ** 2 + h * h)

    return NonlinearExperimentModel(
        forward,
        (require_finite_real(prior_mean, name="prior_mean"),),
        ((sd * sd,),),
        noise_std,
    )
