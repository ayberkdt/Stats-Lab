"""Bayesian quadrature with a squared-exponential kernel.

The integral ``I = int f(x) pi(dx)`` of a function with a Gaussian-process
prior ``f ~ GP(m, k)`` is itself Gaussian. Given evaluations ``y_i = f(x_i)``
its posterior mean and variance are [EQ-SUR-BQ-001]

    ``E[I | y] = m + z' K^-1 (y - m 1)``,  ``Var[I | y] = kk - z' K^-1 z``,

with the kernel mean ``z_i = int k(x, x_i) pi(dx)`` and the prior variance
``kk = int int k(x, x') pi(dx) pi(dx')`` (O'Hagan's Bayes-Hermite quadrature
when ``pi`` is Gaussian). For the squared-exponential kernel
``k(x, x') = sigma_f^2 prod_j exp(-(x_j - x'_j)^2 / (2 ell_j^2))`` and a
product measure both integrals factorize over the coordinates, and each
factor has a closed form for a Gaussian measure [EQ-SUR-BQ-002] and for a
uniform measure [EQ-SUR-BQ-003]. The closed forms were derived for this
module and are checked against the library's adaptive quadrature in the
tests.

Warning: the posterior standard deviation is not an error bound. It is the
spread of the integral under the stated squared-exponential prior and says
nothing about integrands that the prior makes improbable. Measured with
thirteen equally spaced nodes on ``[-3, 3]``, a standard normal measure and
the fixed length scale 1 (reported standard deviation ``9.0e-5`` in every
case, because the variance does not depend on the function values): the
actual error was 0.2 standard deviations for ``cos x``, 4 for
``exp(x / 2)``, 6 for ``1 / (1 + x^2)``, 17 for ``x^2``, 39 for
``sin(3 x) + x^4 / 10``, and 250 for ``|x|``. When the hyperparameters are
estimated from the same evaluations (``length_scales=None``) their
uncertainty is not included either, and for integrands that are nearly
polynomial the fitted length scale becomes so long that the kernel matrix is
numerically singular; a variance that cancels below rounding is then
reported as zero. Check the result against a second rule or a second set of
nodes before relying on the reported deviation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream
from probstat.foundations.validation import require_finite_real
from probstat.linalg import Vector, as_vector, cholesky, cholesky_solve

from .gaussian_process import fit_gaussian_process, kernel_value

_SQRT_HALF_PI = math.sqrt(0.5 * math.pi)
_SQRT_TWO = math.sqrt(2.0)


@dataclass(frozen=True, slots=True)
class GaussianMeasure:
    """The normal probability measure ``N(mean, sd^2)`` of one coordinate.

    Args:
        mean: Mean ``b``.
        sd: Standard deviation ``s > 0``.

    Raises:
        ValueError: If ``sd`` is not positive.

    References:
        [REF-OHAGAN-1991].

    """

    mean: float = 0.0
    sd: float = 1.0

    def __post_init__(self) -> None:
        """Validate the parameters.

        Raises:
            ValueError: If a parameter is not finite or ``sd <= 0``.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION]. No equation
            is evaluated.

        """
        require_finite_real(self.mean, name="mean")
        if require_finite_real(self.sd, name="sd") <= 0.0:
            raise ValueError("sd must be positive")


@dataclass(frozen=True, slots=True)
class UniformMeasure:
    """The uniform probability measure on ``[lower, upper]`` of one coordinate.

    Args:
        lower: Left end ``a``.
        upper: Right end ``c > a``.

    Raises:
        ValueError: If the interval is empty.

    References:
        [REF-OHAGAN-1991] (the general Bayesian quadrature rule; the uniform
        closed forms are derived in [EQ-SUR-BQ-003]).

    """

    lower: float = 0.0
    upper: float = 1.0

    def __post_init__(self) -> None:
        """Validate the interval.

        Raises:
            ValueError: If an end is not finite or ``upper <= lower``.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION]. No equation
            is evaluated.

        """
        low = require_finite_real(self.lower, name="lower")
        if require_finite_real(self.upper, name="upper") <= low:
            raise ValueError("upper must exceed lower")


QuadratureMeasure = GaussianMeasure | UniformMeasure


def _require_measure(measure: object) -> QuadratureMeasure:
    """Return ``measure`` if it is a supported one-dimensional measure.

    Raises:
        TypeError: Otherwise.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    if not isinstance(measure, GaussianMeasure | UniformMeasure):
        raise TypeError("measure must be a GaussianMeasure or a UniformMeasure")
    return measure


def kernel_mean(measure: QuadratureMeasure, x: float, length_scale: float) -> float:
    """Return ``int exp(-(t - x)^2 / (2 ell^2)) pi(dt)`` for one coordinate.

    Gaussian measure ``N(b, s^2)`` (a Gaussian convolution):
    ``sqrt(ell^2 / (ell^2 + s^2)) exp(-(x - b)^2 / (2 (ell^2 + s^2)))``.
    Uniform measure on ``[a, c]``:
    ``ell sqrt(pi/2) / (c - a) (erf((c - x) / (sqrt(2) ell))
    - erf((a - x) / (sqrt(2) ell)))``.

    Args:
        measure: Integration measure of the coordinate.
        x: Node coordinate.
        length_scale: Kernel length scale ``ell > 0``.

    Returns:
        The unit-variance kernel mean at ``x``, in ``(0, 1]``.

    Raises:
        TypeError: If the measure type is unsupported.
        ValueError: If ``length_scale`` is not positive.

    Equation:
        [EQ-SUR-BQ-002], [EQ-SUR-BQ-003].

    References:
        [REF-OHAGAN-1991], [REF-NIST-DLMF, Section 7.2] (error function).

    """
    pi = _require_measure(measure)
    ell = require_finite_real(length_scale, name="length_scale")
    if ell <= 0.0:
        raise ValueError("length_scale must be positive")
    node = require_finite_real(x, name="x")
    if isinstance(pi, GaussianMeasure):
        total = ell * ell + pi.sd * pi.sd
        return (ell / math.sqrt(total)) * math.exp(-0.5 * (node - pi.mean) ** 2 / total)
    width = pi.upper - pi.lower
    scale = _SQRT_TWO * ell
    difference = math.erf((pi.upper - node) / scale) - math.erf(
        (pi.lower - node) / scale
    )
    return ell * _SQRT_HALF_PI * difference / width


def kernel_double_integral(measure: QuadratureMeasure, length_scale: float) -> float:
    """Return ``int int exp(-(t - u)^2 / (2 ell^2)) pi(dt) pi(du)``.

    Gaussian measure ``N(b, s^2)``: ``t - u ~ N(0, 2 s^2)`` gives
    ``sqrt(ell^2 / (ell^2 + 2 s^2))``. Uniform measure on an interval of
    width ``w``: the difference ``|t - u|`` has density ``2 (w - v) / w^2``
    on ``[0, w]``, which gives
    ``(2 / w^2) (w ell sqrt(pi/2) erf(w / (sqrt(2) ell))
    - ell^2 (1 - exp(-w^2 / (2 ell^2))))``.

    Args:
        measure: Integration measure of the coordinate.
        length_scale: Kernel length scale ``ell > 0``.

    Returns:
        The unit-variance prior variance factor, in ``(0, 1]``.

    Raises:
        TypeError: If the measure type is unsupported.
        ValueError: If ``length_scale`` is not positive.

    Equation:
        [EQ-SUR-BQ-002], [EQ-SUR-BQ-003].

    References:
        [REF-OHAGAN-1991], [REF-NIST-DLMF, Section 7.2] (error function).

    """
    pi = _require_measure(measure)
    ell = require_finite_real(length_scale, name="length_scale")
    if ell <= 0.0:
        raise ValueError("length_scale must be positive")
    if isinstance(pi, GaussianMeasure):
        return ell / math.sqrt(ell * ell + 2.0 * pi.sd * pi.sd)
    width = pi.upper - pi.lower
    ratio = width / ell
    first = width * ell * _SQRT_HALF_PI * math.erf(ratio / _SQRT_TWO)
    second = -ell * ell * math.expm1(-0.5 * ratio * ratio)
    return 2.0 * (first - second) / (width * width)


@dataclass(frozen=True, slots=True)
class BayesianQuadratureResult:
    """Posterior of an integral under a squared-exponential GP prior.

    Args:
        estimate: Posterior mean of the integral.
        variance: Posterior variance (``>= 0``; rounding below zero is
            clipped), conditional on the hyperparameters.
        weights: Quadrature weights ``w = K^-1 z``, so that
            ``estimate = m + w' (y - m 1)``.
        kernel_means: The vector ``z``.
        prior_variance: ``kk``, the variance of the integral before any
            evaluation.
        prior_mean: Constant prior mean ``m``.
        signal_variance: Kernel variance ``sigma_f^2``.
        length_scales: Kernel length scales.
        noise_variance: Diagonal variance added to ``K``.
        evaluations: Number of integrand values used (the nodes).
        hyperparameters_estimated: Whether the kernel hyperparameters and
            the prior mean were fitted to the same evaluations.
        converged: Convergence flag of that fit (``True`` when the
            hyperparameters were given).

    Equation:
        [EQ-SUR-BQ-001].

    References:
        [REF-OHAGAN-1991].

    """

    estimate: float
    variance: float
    weights: Vector
    kernel_means: Vector
    prior_variance: float
    prior_mean: float
    signal_variance: float
    length_scales: Vector
    noise_variance: float
    evaluations: int
    hyperparameters_estimated: bool = False
    converged: bool = True

    @property
    def standard_deviation(self) -> float:
        """Return the posterior standard deviation of the integral.

        References:
            [EQ-SUR-BQ-001], [REF-OHAGAN-1991].

        """
        return math.sqrt(self.variance)


def bayesian_quadrature(
    points: Sequence[Sequence[float]],
    values: Sequence[float],
    measures: Sequence[QuadratureMeasure],
    *,
    length_scales: Sequence[float] | None = None,
    signal_variance: float = 1.0,
    noise_variance: float = 0.0,
    prior_mean: float = 0.0,
    restarts: int = 3,
    seed: int | RandomStream | None = None,
) -> BayesianQuadratureResult:
    """Integrate a function against a product measure by Bayesian quadrature.

    With given ``length_scales`` the GP prior is
    ``GP(prior_mean, signal_variance * SE)`` and the result is exact GP
    algebra. With ``length_scales=None`` a squared-exponential GP is first
    fitted to the evaluations by maximum marginal likelihood
    (:func:`fit_gaussian_process`); its signal variance, length scales, and
    output mean (as prior mean) replace the corresponding arguments, and
    the diagonal variance is the larger of the fitted nugget and
    ``noise_variance``.

    The returned variance is conditional on the kernel and its
    hyperparameters and is not an error bound: errors of 4 to 250 reported
    standard deviations were measured on smooth and non-smooth integrands
    (see the module docstring).

    Args:
        points: Nodes ``x_i`` in ``R^d`` (distinct unless
            ``noise_variance > 0``).
        values: Integrand values ``f(x_i)``.
        measures: One :class:`GaussianMeasure` or :class:`UniformMeasure`
            per coordinate; the integration measure is their product.
        length_scales: Kernel length scales, or ``None`` to estimate all
            hyperparameters.
        signal_variance: Kernel variance ``sigma_f^2 > 0``.
        noise_variance: Diagonal variance ``>= 0`` added to ``K`` (noisy
            evaluations or numerical regularization). With
            ``length_scales=None`` it is a lower bound on the fitted nugget.
        prior_mean: Constant prior mean ``m``.
        restarts: Optimizer starts of the hyperparameter fit.
        seed: Seed or stream of the hyperparameter fit.

    Returns:
        Posterior mean and variance of the integral with the weights.

    Raises:
        TypeError: If a measure has an unsupported type.
        ValueError: If inputs are inconsistent or ``K`` is not numerically
            positive definite (add ``noise_variance`` or shorten the length
            scales).

    Equation:
        [EQ-SUR-BQ-001], [EQ-SUR-BQ-002], [EQ-SUR-BQ-003].

    References:
        [REF-OHAGAN-1991], [REF-RASMUSSEN-WILLIAMS-2006, Chapter 2].

    """
    x = tuple(as_vector(point, name="point") for point in points)
    y = as_vector(values, name="values")
    if not x or len(x) != len(y):
        raise ValueError("need at least one node with one value each")
    laws = tuple(_require_measure(measure) for measure in measures)
    dimension = len(laws)
    if dimension == 0 or any(len(point) != dimension for point in x):
        raise ValueError("need one measure per coordinate of the nodes")
    estimated = length_scales is None
    converged = True
    if length_scales is None:
        model = fit_gaussian_process(
            x, y, kernel="squared_exponential", restarts=restarts, seed=seed
        )
        scales = model.hyperparameters.length_scales
        signal = model.hyperparameters.signal_variance
        floor = require_finite_real(noise_variance, name="noise_variance")
        if floor < 0.0:
            raise ValueError("noise_variance must be non-negative")
        noise = max(model.hyperparameters.noise_variance, floor)
        centre = model.output_mean
        converged = model.converged
    else:
        scales = as_vector(length_scales, name="length_scales")
        if len(scales) != dimension or any(scale <= 0.0 for scale in scales):
            raise ValueError("length_scales must be positive, one per coordinate")
        signal = require_finite_real(signal_variance, name="signal_variance")
        if signal <= 0.0:
            raise ValueError("signal_variance must be positive")
        noise = require_finite_real(noise_variance, name="noise_variance")
        if noise < 0.0:
            raise ValueError("noise_variance must be non-negative")
        centre = require_finite_real(prior_mean, name="prior_mean")
    rows = []
    for i, left in enumerate(x):
        row = [
            kernel_value("squared_exponential", left, right, scales, signal)
            for right in x
        ]
        row[i] += noise
        rows.append(tuple(row))
    try:
        lower = cholesky(tuple(rows))
    except ValueError as error:
        raise ValueError(
            "kernel matrix is not positive definite; add noise_variance or "
            "use shorter length scales"
        ) from error
    embedding = tuple(
        signal
        * math.prod(
            kernel_mean(law, coordinate, scale)
            for law, coordinate, scale in zip(laws, point, scales, strict=True)
        )
        for point in x
    )
    prior = signal * math.prod(
        kernel_double_integral(law, scale)
        for law, scale in zip(laws, scales, strict=True)
    )
    weights = cholesky_solve(lower, embedding)
    estimate = centre + math.fsum(
        w * (value - centre) for w, value in zip(weights, y, strict=True)
    )
    variance = prior - math.fsum(w * z for w, z in zip(weights, embedding, strict=True))
    return BayesianQuadratureResult(
        estimate,
        max(variance, 0.0),
        weights,
        embedding,
        prior,
        centre,
        signal,
        scales,
        noise,
        len(x),
        hyperparameters_estimated=estimated,
        converged=converged,
    )
