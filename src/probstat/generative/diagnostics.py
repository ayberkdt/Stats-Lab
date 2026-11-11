"""Diagnostics that compare fitted densities with each other and with the truth.

An energy model is known only up to ``Z = int exp(-E(x)) dx``, and a flow's
exact density should integrate to one; both are checked by estimating a
normalizing constant ([EQ-GEN-DIAG-001]):

* :func:`quadrature_normalizing_constant` integrates ``exp(log f)`` by
  adaptive Gauss-Kronrod quadrature, nested in two dimensions, after
  shifting ``log f`` by its maximum on a coarse grid;
* :func:`importance_normalizing_constant` averages the importance weights
  ``f(y) / q(y)`` of draws from a normalized proposal (for instance a
  fitted flow), with its standard error and weight effective sample size.

Densities on a common scale are compared by the mean log-likelihood of data
([EQ-GEN-DIAG-002]) and by the Monte Carlo Kullback-Leibler estimate
``mean log p(x) - log q(x)``, ``x ~ p`` ([EQ-GEN-DIAG-003]).

References:
    [REF-OWEN-2013, Chapters 2 and 9], [REF-COVER-THOMAS-2006, Section 2.3],
    [REF-PIESSENS-QUADPACK-1983].

"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ..numerics import integrate
from ._common import Point, as_points, mean_and_standard_error

LogFunction = Callable[[Point], float]


@dataclass(frozen=True)
class NormalizingConstant:
    """Estimate of ``log Z`` for ``Z = int f(x) dx``.

    Args:
        log_value: Estimated ``log Z``.
        error: Estimated absolute error of ``log Z`` (relative error of
            ``Z``): the quadrature error estimate, or the importance-sampling
            standard error divided by the estimate.
        method: ``"quadrature"`` or ``"importance-sampling"``.
        evaluations: Evaluations of ``log f``.
        effective_sample_size: Weight effective sample size
            (importance sampling only).

    References:
        [EQ-GEN-DIAG-001].

    """

    log_value: float
    error: float
    method: str
    evaluations: int
    effective_sample_size: float | None = None

    @property
    def value(self) -> float:
        """Return ``Z = exp(log Z)``.

        References:
            [EQ-GEN-DIAG-001]. No new equation is introduced.

        """
        return math.exp(self.log_value)


@dataclass(frozen=True)
class SampleMean:
    """A sample mean with its standard error ``s / sqrt(n)``.

    Args:
        estimate: The sample mean.
        standard_error: Standard error of the mean (0 for one value).
        sample_size: Number of terms.

    References:
        [EQ-GEN-DIAG-002].

    """

    estimate: float
    standard_error: float
    sample_size: int


def _bounds(lower: object, upper: object) -> tuple[Point, Point]:
    """Validate integration limits of dimension one or two.

    Raises:
        TypeError: If a limit is not a real number or a sequence of reals.
        ValueError: For a dimension other than one or two, NaN limits, or an
            empty interval.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].
        No equation is evaluated.

    """
    lows = _limits(lower, "lower")
    highs = _limits(upper, "upper")
    if len(lows) != len(highs) or len(lows) not in (1, 2):
        raise ValueError("limits must have one common dimension, 1 or 2")
    for low, high in zip(lows, highs, strict=True):
        if not low < high:
            raise ValueError("each lower limit must be below its upper limit")
    return lows, highs


def _limits(value: object, name: str) -> Point:
    """Return a limit as a tuple of floats (``inf`` allowed).

    Raises:
        TypeError: If ``value`` is not a real number or a sequence of reals.
        ValueError: If a limit is NaN.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].
        No equation is evaluated.

    """
    items = value if isinstance(value, Sequence) else (value,)
    result: list[float] = []
    for item in items:
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise TypeError(f"{name} must hold real numbers")
        number = float(item)
        if math.isnan(number):
            raise ValueError(f"{name} must not be NaN")
        result.append(number)
    return tuple(result)


def _grid(low: float, high: float, count: int = 9) -> list[float]:
    """Return ``count`` equally spaced points of ``[low, high]`` clipped to ``±10``.

    References:
        [EQ-GEN-DIAG-001]. No equation is evaluated.

    """
    a = max(low, -10.0)
    b = min(high, 10.0)
    if not a < b:
        a, b = (low, low + 1.0) if math.isfinite(low) else (high - 1.0, high)
    return [a + (b - a) * k / (count - 1) for k in range(count)]


def quadrature_normalizing_constant(
    log_function: LogFunction,
    lower: object,
    upper: object,
    *,
    rel_tol: float = 1e-6,
    shift: float | None = None,
) -> NormalizingConstant:
    """Return ``log int exp(log f(x)) dx`` over a box by adaptive quadrature.

    ``exp(log f - c)`` is integrated with adaptive Gauss-Kronrod (G7-K15)
    quadrature, nested as ``int (int ... dx_2) dx_1`` in two dimensions, and
    ``c`` is added back ([EQ-GEN-DIAG-001]). The shift ``c`` defaults to the
    maximum of ``log f`` on a ``9 x 9`` grid of the box (clipped to
    ``[-10, 10]``), which avoids overflow and underflow.

    Args:
        log_function: ``x -> log f(x)`` for a point tuple ``x``.
        lower: Lower limit(s); a number or a sequence of one or two numbers
            (``-inf`` allowed).
        upper: Upper limit(s) (``inf`` allowed).
        rel_tol: Positive relative error target of each integration.
        shift: Optional shift ``c``.

    Returns:
        :class:`NormalizingConstant` with ``method="quadrature"``.

    Raises:
        ValueError: For invalid limits or tolerance, or a non-finite shift.

    References:
        [EQ-GEN-DIAG-001], [REF-PIESSENS-QUADPACK-1983].

    """
    lows, highs = _bounds(lower, upper)
    if not math.isfinite(rel_tol) or rel_tol <= 0.0:
        raise ValueError("rel_tol must be positive and finite")
    counter = [0]

    def evaluate(point: Point) -> float:
        """Count and evaluate ``log f``.

        References:
            [EQ-GEN-DIAG-001]. No new equation is introduced.

        """
        counter[0] += 1
        return float(log_function(point))

    if shift is None:
        grids = [_grid(low, high) for low, high in zip(lows, highs, strict=True)]
        if len(grids) == 1:
            candidates = [evaluate((a,)) for a in grids[0]]
        else:
            candidates = [evaluate((a, b)) for a in grids[0] for b in grids[1]]
        finite = [value for value in candidates if math.isfinite(value)]
        if not finite:
            raise ValueError("log_function is not finite on the search grid")
        shift = max(finite)
    if not math.isfinite(shift):
        raise ValueError("shift must be finite")
    offset = shift
    abs_tol = 1e-14

    if len(lows) == 1:
        result = integrate(
            lambda a: math.exp(evaluate((a,)) - offset),
            lows[0],
            highs[0],
            method="gauss-kronrod",
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        )
        value, error = result.value, result.error_estimate
    else:
        inner_errors: list[float] = []

        def inner(a: float) -> float:
            """Integrate over the second coordinate at ``x_1 = a``.

            References:
                [EQ-GEN-DIAG-001], [REF-PIESSENS-QUADPACK-1983].

            """
            section = integrate(
                lambda b: math.exp(evaluate((a, b)) - offset),
                lows[1],
                highs[1],
                method="gauss-kronrod",
                rel_tol=rel_tol,
                abs_tol=abs_tol,
            )
            inner_errors.append(section.error_estimate)
            return section.value

        result = integrate(
            inner,
            lows[0],
            highs[0],
            method="gauss-kronrod",
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        )
        width = highs[0] - lows[0]
        spread = max(inner_errors) * width if math.isfinite(width) else 0.0
        value, error = result.value, result.error_estimate + spread
    if value <= 0.0:
        raise ValueError("the integral is not positive")
    return NormalizingConstant(
        log_value=math.log(value) + offset,
        error=error / value,
        method="quadrature",
        evaluations=counter[0],
    )


def importance_normalizing_constant(
    log_function: LogFunction,
    samples: object,
    proposal_log_densities: Sequence[float],
) -> NormalizingConstant:
    """Return ``log Z`` from draws of a normalized proposal ``q``.

    ``Z_hat = mean_i f(y_i) / q(y_i)`` is unbiased for ``Z`` ([EQ-MC-IS-001]);
    the weights are combined in log space. The reported error is
    ``SE(Z_hat) / Z_hat`` and the weight effective sample size is
    ``(sum w)^2 / sum w^2`` ([EQ-MC-ESS-001]).

    Args:
        log_function: ``x -> log f(x)``.
        samples: Draws ``y_i ~ q`` (at least two).
        proposal_log_densities: ``log q(y_i)`` for each draw.

    Returns:
        :class:`NormalizingConstant` with ``method="importance-sampling"``.

    Raises:
        ValueError: For mismatched lengths or non-finite log-densities.

    References:
        [EQ-GEN-DIAG-001], [EQ-MC-IS-001], [EQ-MC-ESS-001],
        [REF-OWEN-2013, Chapter 9].

    """
    points = as_points(samples, name="samples", minimum=2)
    if len(proposal_log_densities) != len(points):
        raise ValueError("one proposal log-density is needed per sample")
    log_weights: list[float] = []
    for point, log_q in zip(points, proposal_log_densities, strict=True):
        value = float(log_function(point)) - float(log_q)
        if math.isnan(value) or value == math.inf:
            raise ValueError("log-weights must be finite or -inf")
        log_weights.append(value)
    top = max(log_weights)
    if not math.isfinite(top):
        raise ValueError("every importance weight is zero")
    weights = [math.exp(value - top) for value in log_weights]
    mean, error = mean_and_standard_error(weights)
    ess = math.fsum(weights) ** 2 / math.fsum(w * w for w in weights)
    return NormalizingConstant(
        log_value=math.log(mean) + top,
        error=error / mean,
        method="importance-sampling",
        evaluations=len(points),
        effective_sample_size=ess,
    )


def mean_log_likelihood(log_density: LogFunction, data: object) -> SampleMean:
    """Return the mean log-density of data and its standard error.

    Args:
        log_density: Normalized ``x -> log p(x)``.
        data: Points (at least one).

    Returns:
        :class:`SampleMean` of ``log p(x_t)``.

    Raises:
        ValueError: For invalid data or a non-finite log-density.

    References:
        [EQ-GEN-DIAG-002], [REF-PAPAMAKARIOS-ETAL-2021, Section 2.3.1].

    """
    points = as_points(data)
    values = [float(log_density(point)) for point in points]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("log_density must be finite at every data point")
    mean, error = mean_and_standard_error(values)
    return SampleMean(mean, error, len(values))


def kl_divergence_estimate(
    log_p: LogFunction, log_q: LogFunction, samples: object
) -> SampleMean:
    """Return the Monte Carlo estimate of ``KL(p || q)`` from draws of ``p``.

    ``KL(p || q) = E_p[log p(X) - log q(X)]`` is estimated by the sample mean
    of the log-ratio over ``x_i ~ p`` ([EQ-GEN-DIAG-003]); both densities
    must be normalized. The estimate is unbiased but can be negative.

    Args:
        log_p: Normalized ``log p``.
        log_q: Normalized ``log q``.
        samples: Draws from ``p`` (at least one).

    Returns:
        :class:`SampleMean` of the log-ratio.

    Raises:
        ValueError: For invalid samples or a non-finite log-ratio.

    References:
        [EQ-GEN-DIAG-003], [REF-COVER-THOMAS-2006, Section 2.3],
        [REF-PAPAMAKARIOS-ETAL-2021, Section 2.3.1].

    """
    points = as_points(samples, name="samples")
    values = [float(log_p(point)) - float(log_q(point)) for point in points]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("the log-ratio must be finite at every sample")
    mean, error = mean_and_standard_error(values)
    return SampleMean(mean, error, len(values))
