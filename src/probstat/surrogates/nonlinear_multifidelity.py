"""Nonlinear autoregressive multi-fidelity Gaussian processes (NARGP).

Experimental (added in 0.28).

Linear co-kriging (:mod:`probstat.surrogates.multifidelity`) assumes
``Z_t(x) = rho Z_(t-1)(x) + delta_t(x)``: the cross-correlation between
fidelities is a fixed scalar. Perdikaris et al. replace it by an unknown
*function* of the lower-fidelity output, recursively [EQ-SUR-NARGP-001]

    ``f_t(x) = g_t(x, f_(t-1)(x))``,

with ``g_t`` a Gaussian process on the augmented input ``(x, f_(t-1)(x))``
whose kernel is structured as [EQ-SUR-NARGP-002]

    ``k_t((x, f), (x', f')) = k_rho(x, x') k_f(f, f') + k_delta(x, x')``.

``k_rho(x, x')`` lets the strength of the dependence on the lower level vary
over the input space, ``k_f(f, f')`` models the (possibly nonlinear) map
``f_(t-1) -> f_t``, and ``k_delta`` the additive discrepancy. The linear
AR(1) model is the special case ``g_t(x, f) = rho f + delta(x)``.

Training. Level 1 is an ordinary GP of the cheapest code. At level ``t`` the
second input of the training set ``D_t`` is the *posterior mean* of level
``t - 1`` at the points of ``D_t`` [EQ-SUR-NARGP-003]. If the designs are
nested (``D_t`` a subset of ``D_(t-1)``) this is, up to the small diagonal
nugget, the observed lower-fidelity response; the posterior-mean rule needs no
nesting, so non-nested designs are accepted (the lower level is then
*predicted* at the new points and its uncertainty at them is ignored during
training). :attr:`NonlinearMultiFidelityGP.nested` records which case held.

Prediction. The exact predictive law of ``f_t(x*)`` is a Gaussian mixture
(``g_t`` is nonlinear in its second input), which is not Gaussian. Two
predictors are provided:

* :meth:`NonlinearMultiFidelityGP.predict_mc` draws
  ``z ~ N(m_(t-1), s_(t-1)^2)`` from the lower predictive, evaluates the GP
  of level ``t`` at ``(x*, z)``, and for a third level samples again, down
  the chain [EQ-SUR-NARGP-004]. By the law of total variance the mean and
  variance of the mixture are the Monte Carlo averages of ``m_t`` and of
  ``s_t^2 + (m_t - mean)^2``; their Monte Carlo standard errors are reported
  [EQ-SUR-NARGP-005].
* :meth:`NonlinearMultiFidelityGP.predict_plugin` feeds the lower *mean*
  through the chain [EQ-SUR-NARGP-006]. It is cheap and deterministic, but it
  discards the uncertainty of the lower levels and therefore understates the
  predictive variance wherever ``g_t`` is nonlinear in the lower output
  (measured in the tests: ratio of plug-in to propagated variance).

Both treat the kernel hyperparameters (type-II maximum likelihood on each
level separately) as known, so intervals are narrower than a fully Bayesian
predictive; the test suite reports measured coverage. :func:`validate_nargp`
runs the library's emulator diagnostics (:mod:`probstat.surrogates.gp_bayes`)
on the top level using the propagated moments with a diagonal covariance
(correlation between test points is ignored; screening only)
[EQ-SUR-NARGP-007].

All levels are fitted with :func:`probstat.surrogates.gp_advanced.fit_kernel_gp`
(composite kernels with exact gradients). Outputs are centered but not
scaled; no jitter is added beyond the explicit relative nugget.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.linalg import Vector, as_vector

from .gaussian_process import KERNELS
from .gp_advanced import (
    Kernel,
    KernelGaussianProcess,
    ProductKernel,
    StationaryKernel,
    SumKernel,
    _check_theta,
    fit_kernel_gp,
)
from .gp_bayes import EmulatorValidation, validate_emulator_moments

__all__ = [
    "NargpPrediction",
    "NargpValidation",
    "NonlinearMultiFidelityGP",
    "fit_nargp",
    "nargp_kernel",
    "validate_nargp",
]


@dataclass(frozen=True, slots=True)
class _ActiveKernel(Kernel):
    """A stationary kernel acting on selected coordinates of the input.

    Args:
        base: Stationary kernel over the selected coordinates.
        indices: Coordinates of the full input the kernel looks at.
        free_variance: If ``False`` the signal variance is fixed at one and
            is not a hyperparameter (the variance of a product of two
            kernels is only identified once).

    References:
        [EQ-SUR-NARGP-002], [REF-PERDIKARIS-ETAL-2017].

    """

    base: StationaryKernel
    indices: tuple[int, ...]
    free_variance: bool = True

    @property
    def _skip(self) -> int:
        """Return the number of leading log parameters that are not exposed.

        References:
            [EQ-SUR-NARGP-002], [REF-PERDIKARIS-ETAL-2017].

        """
        return 0 if self.free_variance else 1

    @property
    def log_parameters(self) -> Vector:
        """Return the exposed log hyperparameters.

        References:
            [EQ-SUR-NARGP-002], [REF-PERDIKARIS-ETAL-2017].

        """
        return self.base.log_parameters[self._skip :]

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Return the names of the exposed log hyperparameters.

        References:
            [EQ-SUR-NARGP-002], [REF-PERDIKARIS-ETAL-2017].

        """
        return self.base.parameter_names[self._skip :]

    def with_log_parameters(self, log_parameters: Sequence[float]) -> Kernel:
        """Return the kernel with new log parameters.

        Args:
            log_parameters: Exposed log hyperparameters.

        Returns:
            Updated kernel with the same structure.

        References:
            [EQ-SUR-NARGP-002], [REF-PERDIKARIS-ETAL-2017].

        """
        count = self.base.parameter_count - self._skip
        theta = _check_theta(log_parameters, count)
        full = theta if self.free_variance else (0.0, *theta)
        updated = self.base.with_log_parameters(full)
        assert isinstance(updated, StationaryKernel)
        return _ActiveKernel(updated, self.indices, self.free_variance)

    def _select(self, x: Sequence[float]) -> Vector:
        """Return the selected coordinates of ``x``.

        Raises:
            ValueError: If ``x`` is too short.

        References:
            [EQ-SUR-NARGP-002], [REF-PERDIKARIS-ETAL-2017].

        """
        if len(x) <= max(self.indices):
            raise ValueError("input has too few coordinates for this kernel")
        return tuple(x[i] for i in self.indices)

    def evaluate(self, x: Sequence[float], y: Sequence[float]) -> float:
        """Return the covariance of the selected coordinates.

        Args:
            x: First input.
            y: Second input.

        Returns:
            The covariance.

        References:
            [EQ-SUR-NARGP-002], [REF-PERDIKARIS-ETAL-2017].

        """
        return self.base.evaluate(self._select(x), self._select(y))

    def gradient(self, x: Sequence[float], y: Sequence[float]) -> Vector:
        """Return the derivatives with respect to the exposed log parameters.

        Args:
            x: First input.
            y: Second input.

        Returns:
            One derivative per exposed log parameter.

        References:
            [EQ-SUR-NARGP-002], [REF-PERDIKARIS-ETAL-2017].

        """
        return self.base.gradient(self._select(x), self._select(y))[self._skip :]


def nargp_kernel(
    dimension: int,
    *,
    kernel: str = "matern52",
    rho_length_scales: Sequence[float] | None = None,
    f_length_scale: float = 1.0,
    f_variance: float = 1.0,
    delta_length_scales: Sequence[float] | None = None,
    delta_variance: float = 0.1,
) -> Kernel:
    """Build the NARGP kernel on the augmented input ``(x, f)`` [EQ-SUR-NARGP-002].

    ``k((x, f), (x', f')) = k_rho(x, x') k_f(f, f') + k_delta(x, x')`` where
    ``k_rho`` is a unit-variance stationary ARD kernel in ``x``, ``k_f`` a
    stationary kernel in the scalar ``f`` with variance ``f_variance`` (the
    variance of a product is identified once, so it lives here), and
    ``k_delta`` a stationary ARD kernel in ``x`` with variance
    ``delta_variance``. The log hyperparameters, in order, are
    ``(log ell_rho[0..d-1], log s_f^2, log ell_f, log s_delta^2,
    log ell_delta[0..d-1])``, ``2 d + 3`` in all. The kernel is positive
    semi-definite as a sum of a product of kernels and a kernel.

    Args:
        dimension: Dimension ``d >= 1`` of ``x``; the augmented input has
            ``d + 1`` coordinates with ``f`` last.
        kernel: One of ``"squared_exponential"``, ``"matern52"``,
            ``"matern32"`` (shared by the three components).
        rho_length_scales: Positive ARD length scales of ``k_rho``
            (default ``1`` each).
        f_length_scale: Positive length scale of ``k_f``.
        f_variance: Positive variance of ``k_f``.
        delta_length_scales: Positive ARD length scales of ``k_delta``
            (default ``1`` each).
        delta_variance: Positive variance of ``k_delta``.

    Returns:
        A :class:`~probstat.surrogates.gp_advanced.Kernel` usable with
        :func:`~probstat.surrogates.gp_advanced.fit_kernel_gp`.

    Raises:
        ValueError: If a name, count, or value is invalid.

    Equation:
        [EQ-SUR-NARGP-002].

    References:
        [REF-PERDIKARIS-ETAL-2017].

    """
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
        raise ValueError("dimension must be a positive integer")
    if kernel not in KERNELS:
        raise ValueError(f"kernel must be one of {KERNELS}")
    rho = (1.0,) * dimension if rho_length_scales is None else rho_length_scales
    delta = (1.0,) * dimension if delta_length_scales is None else delta_length_scales
    rho_scales = as_vector(rho, name="rho_length_scales")
    delta_scales = as_vector(delta, name="delta_length_scales")
    if len(rho_scales) != dimension or len(delta_scales) != dimension:
        raise ValueError("length scales must have one entry per input dimension")
    x_index = tuple(range(dimension))
    k_rho = _ActiveKernel(StationaryKernel(kernel, 1.0, rho_scales), x_index, False)
    k_f = _ActiveKernel(
        StationaryKernel(kernel, f_variance, (f_length_scale,)), (dimension,), True
    )
    k_delta = _ActiveKernel(
        StationaryKernel(kernel, delta_variance, delta_scales), x_index, True
    )
    return SumKernel(ProductKernel(k_rho, k_f), k_delta)


@dataclass(frozen=True, slots=True)
class NargpPrediction:
    """Propagated predictive moments of one level at one input.

    Args:
        mean: Monte Carlo mean of the predictive mixture.
        variance: Mixture variance (law of total variance).
        mean_mcse: Monte Carlo standard error of ``mean``.
        variance_mcse: Monte Carlo standard error of ``variance``.
        samples: Number of lower-level draws.
        plugin_mean: Mean of the plug-in predictor.
        plugin_variance: Variance of the plug-in predictor (understates
            ``variance`` when the cross-level map is nonlinear).
        level: Fidelity level ``1..s`` of the prediction.

    Equation:
        [EQ-SUR-NARGP-004], [EQ-SUR-NARGP-005], [EQ-SUR-NARGP-006].

    References:
        [REF-PERDIKARIS-ETAL-2017].

    """

    mean: float
    variance: float
    mean_mcse: float
    variance_mcse: float
    samples: int
    plugin_mean: float
    plugin_variance: float
    level: int

    @property
    def standard_deviation(self) -> float:
        """Return the propagated predictive standard deviation.

        References:
            [EQ-SUR-NARGP-004], [REF-PERDIKARIS-ETAL-2017].

        """
        return math.sqrt(self.variance)


@dataclass(frozen=True, slots=True)
class NonlinearMultiFidelityGP:
    """A fitted NARGP surrogate with ``s`` fidelity levels.

    ``levels[0]`` is the GP of the cheapest code on ``x``; ``levels[t - 1]``
    for ``t >= 2`` is the GP of ``f_t`` on the augmented input ``(x, m_(t-1))``
    with the structured kernel of :func:`nargp_kernel`.

    Args:
        levels: One fitted GP per level, cheapest first.
        input_dimension: Dimension ``d`` of ``x``.
        nested: Whether every design is a subset of the one below.
        kernel_name: Kernel family used on all levels.

    Raises:
        ValueError: If no level is given or the dimensions are inconsistent.

    Equation:
        [EQ-SUR-NARGP-001], [EQ-SUR-NARGP-002], [EQ-SUR-NARGP-003].

    References:
        [REF-PERDIKARIS-ETAL-2017].

    """

    levels: tuple[KernelGaussianProcess, ...]
    input_dimension: int
    nested: bool
    kernel_name: str

    def __post_init__(self) -> None:
        """Validate the level structure.

        Raises:
            ValueError: If the structure is inconsistent.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION]. No equation
            is evaluated.

        """
        if not self.levels:
            raise ValueError("at least one level is required")
        d = self.input_dimension
        for index, gp in enumerate(self.levels):
            expected = d if index == 0 else d + 1
            if len(gp.points[0]) != expected:
                raise ValueError("level input dimensions are inconsistent")

    @property
    def level_count(self) -> int:
        """Return the number of fidelity levels ``s``.

        References:
            [REF-PERDIKARIS-ETAL-2017]. No equation is evaluated.

        """
        return len(self.levels)

    @property
    def design_sizes(self) -> tuple[int, ...]:
        """Return the number of training runs of every level.

        References:
            [REF-PERDIKARIS-ETAL-2017]. No equation is evaluated.

        """
        return tuple(len(gp.points) for gp in self.levels)

    def _top(self, level: int | None) -> int:
        """Return a validated 1-based level index.

        Raises:
            ValueError: If ``level`` is not an integer in ``1..s``.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION]. No equation
            is evaluated.

        """
        count = len(self.levels)
        if level is None:
            return count
        if isinstance(level, bool) or not isinstance(level, int):
            raise ValueError("level must be an integer")
        if not 1 <= level <= count:
            raise ValueError(f"level must lie in 1..{count}")
        return level

    def _point(self, x: Sequence[float]) -> Vector:
        """Return ``x`` as a vector of the input dimension.

        Raises:
            ValueError: If the dimension is wrong.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION]. No equation
            is evaluated.

        """
        point = as_vector(x, name="x")
        if len(point) != self.input_dimension:
            raise ValueError("x must have the input dimension of the model")
        return point

    def predict_plugin(
        self, x: Sequence[float], *, level: int | None = None
    ) -> tuple[float, float]:
        """Return the plug-in predictive mean and variance of one level.

        The lower-level mean replaces the lower-level random variable:
        ``z_1 = m_1(x)``, ``(m_t, s_t^2) = GP_t(x, z_(t-1))``,
        ``z_t = m_t`` [EQ-SUR-NARGP-006]. The variance is that of the last
        GP at the plugged-in point only, so the uncertainty of the lower
        levels is ignored and the variance is too small wherever ``g_t`` is
        nonlinear in its second argument.

        Args:
            x: Input vector.
            level: Fidelity level ``1..s``; ``None`` is the highest.

        Returns:
            Plug-in mean and variance in output units.

        Raises:
            ValueError: If ``x`` or ``level`` is invalid.

        Equation:
            [EQ-SUR-NARGP-006].

        References:
            [REF-PERDIKARIS-ETAL-2017].

        """
        top = self._top(level)
        point = self._point(x)
        mean, variance = self.levels[0].predict_with_variance(point)
        for gp in self.levels[1:top]:
            mean, variance = gp.predict_with_variance((*point, mean))
        return mean, variance

    def predict_mc(
        self,
        x: Sequence[float],
        *,
        samples: int = 500,
        seed: int | RandomStream | None = None,
        level: int | None = None,
    ) -> NargpPrediction:
        """Propagate the lower predictive laws by Monte Carlo [EQ-SUR-NARGP-004].

        For each of ``N`` independent draws: ``z_1 ~ N(m_1, s_1^2)`` from
        level 1; ``(m_2, s_2^2) = GP_2(x, z_1)``; for a higher level
        ``z_2 ~ N(m_2, s_2^2)`` and so on up to ``level``. The predictive
        mixture has mean ``mean = (1/N) sum m_i`` and variance
        ``(1/N) sum s_i^2 + (1/(N-1)) sum (m_i - mean)^2`` (law of total
        variance). Standard errors: ``sd(m_i)/sqrt(N)`` for the mean, and
        ``sd(u_i)/sqrt(N)`` with ``u_i = s_i^2 + (m_i - mean)^2`` for the
        variance (the estimated mean is used in ``u_i``, so the second error
        is a first-order estimate) [EQ-SUR-NARGP-005]. For level 1 the law is
        Gaussian and exact, so the errors are zero.

        Args:
            x: Input vector.
            samples: Number of Monte Carlo draws (``>= 2``).
            seed: Integer seed or stream.
            level: Fidelity level ``1..s``; ``None`` is the highest.

        Returns:
            :class:`NargpPrediction` with the propagated moments, their
            Monte Carlo errors, and the plug-in values for comparison.

        Raises:
            TypeError: If ``samples`` is not an integer.
            ValueError: If an argument is invalid.

        Equation:
            [EQ-SUR-NARGP-004], [EQ-SUR-NARGP-005].

        References:
            [REF-PERDIKARIS-ETAL-2017].

        """
        if isinstance(samples, bool) or not isinstance(samples, int):
            raise TypeError("samples must be an integer")
        if samples < 2:
            raise ValueError("samples must be at least 2")
        top = self._top(level)
        point = self._point(x)
        plug_mean, plug_variance = self.predict_plugin(point, level=top)
        if top == 1:
            return NargpPrediction(
                plug_mean, plug_variance, 0.0, 0.0, samples, plug_mean, plug_variance, 1
            )
        generator = resolve_stream(seed).child("nargp-mc").generator()
        first_mean, first_variance = self.levels[0].predict_with_variance(point)
        means: list[float] = []
        variances: list[float] = []
        for _ in range(samples):
            mean, variance = first_mean, first_variance
            for gp in self.levels[1:top]:
                lower = mean + math.sqrt(variance) * generator.gauss(0.0, 1.0)
                mean, variance = gp.predict_with_variance((*point, lower))
            means.append(mean)
            variances.append(variance)
        centre = math.fsum(means) / samples
        squares = [(m - centre) ** 2 for m in means]
        between = math.fsum(squares) / (samples - 1)
        mixture = math.fsum(variances) / samples + between
        mean_se = math.sqrt(between / samples)
        totals = [v + s for v, s in zip(variances, squares, strict=True)]
        mean_total = math.fsum(totals) / samples
        spread = math.fsum((u - mean_total) ** 2 for u in totals) / (samples - 1)
        return NargpPrediction(
            centre,
            mixture,
            mean_se,
            math.sqrt(spread / samples),
            samples,
            plug_mean,
            plug_variance,
            top,
        )

    def predict_mc_many(
        self,
        points: Sequence[Sequence[float]],
        *,
        samples: int = 500,
        seed: int | RandomStream | None = None,
        level: int | None = None,
    ) -> tuple[NargpPrediction, ...]:
        """Return :meth:`predict_mc` at several inputs with independent streams.

        Input ``i`` uses the child stream ``("point", i)`` of ``seed``, so the
        result does not depend on the other points.

        Args:
            points: Inputs.
            samples: Monte Carlo draws per input.
            seed: Integer seed or stream.
            level: Fidelity level ``1..s``; ``None`` is the highest.

        Returns:
            One prediction per input.

        Raises:
            ValueError: If an argument is invalid.

        Equation:
            [EQ-SUR-NARGP-004], [EQ-SUR-NARGP-005].

        References:
            [REF-PERDIKARIS-ETAL-2017].

        """
        root = resolve_stream(seed)
        return tuple(
            self.predict_mc(
                p, samples=samples, seed=root.child("point", i), level=level
            )
            for i, p in enumerate(points)
        )


def _require_design(
    design: Sequence[Sequence[float]], values: Sequence[float], index: int
) -> tuple[tuple[Vector, ...], Vector]:
    """Return a validated design and its responses for level ``index``.

    Raises:
        ValueError: If the design is empty, ragged, repeated, or mismatched.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    points = tuple(as_vector(p, name="point") for p in design)
    outputs = as_vector(values, name="responses")
    if len(points) < 3:
        raise ValueError(f"level {index} needs at least three runs")
    if len(points) != len(outputs):
        raise ValueError(f"level {index} needs one response per design point")
    if any(len(p) != len(points[0]) for p in points) or not points[0]:
        raise ValueError(f"level {index} points must be non-empty and equally long")
    if len(set(points)) != len(points):
        raise ValueError(f"level {index} design has repeated points")
    return points, outputs


def _spread(values: Sequence[float]) -> tuple[float, float]:
    """Return the sample variance and the range of ``values`` (never zero).

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    centre = math.fsum(values) / len(values)
    variance = math.fsum((v - centre) ** 2 for v in values) / len(values)
    width = max(values) - min(values)
    return (variance if variance > 0.0 else 1.0), (width if width > 0.0 else 1.0)


_SCAN = (0.05, 0.15, 0.5)


def _fit_best_start(
    inputs: Sequence[Vector],
    values: Vector,
    starts: Sequence[Kernel],
    *,
    nugget: float,
    restarts: int,
    seed: RandomStream,
    max_iterations: int,
) -> KernelGaussianProcess:
    """Fit from the start kernel with the highest marginal likelihood.

    The likelihood of a GP with a short-wavelength signal has a poor local
    maximum at tiny length scales (the data look like noise). Every start of
    a small deterministic scan is scored by one factorization (no
    optimization); the best one initializes the optimizer. A start whose
    matrix is not positive definite is skipped.

    Raises:
        ValueError: If no start is usable.

    Equation:
        [EQ-SUR-NARGP-002].

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Section 5.4].

    """
    best: tuple[float, Kernel] | None = None
    for start in starts:
        try:
            trial = fit_kernel_gp(
                inputs, values, start, nugget=nugget, optimize=False, restarts=1
            )
        except ValueError:
            continue
        if best is None or trial.log_marginal_likelihood > best[0]:
            best = (trial.log_marginal_likelihood, start)
    if best is None:
        raise ValueError("no start kernel produced a positive-definite matrix")
    return fit_kernel_gp(
        inputs,
        values,
        best[1],
        nugget=nugget,
        restarts=restarts,
        seed=seed,
        max_iterations=max_iterations,
    )


def fit_nargp(
    designs: Sequence[Sequence[Sequence[float]]],
    responses: Sequence[Sequence[float]],
    *,
    kernel: str = "squared_exponential",
    nugget: float = 1e-8,
    restarts: int = 4,
    seed: int | RandomStream | None = None,
    max_iterations: int = 100,
) -> NonlinearMultiFidelityGP:
    """Fit a NARGP surrogate recursively, cheapest level first.

    Level 1: a stationary ARD GP of ``z_1`` on ``D_1``. Level ``t >= 2``: the
    second input at the points of ``D_t`` is the posterior (plug-in) mean of
    level ``t - 1`` there [EQ-SUR-NARGP-003], and the GP with the kernel of
    :func:`nargp_kernel` is fitted to ``z_t(D_t)`` by type-II maximum
    likelihood [EQ-SUR-NARGP-001], [EQ-SUR-NARGP-002]. Initial length
    scales are ``0.3`` of the data range of each coordinate (``0.5`` of the
    range of the lower-level mean for ``k_f``), the initial variances the
    sample variance of the responses (``0.1`` of it for ``k_delta``).
    ``nugget`` is relative to the sample variance of each level's responses.

    Args:
        designs: One design per level, cheapest first (each at least three
            points; non-nested designs are allowed, see the module
            docstring).
        responses: Responses of each level at its design (not all equal).
        kernel: ``"squared_exponential"``, ``"matern52"``, or
            ``"matern32"``.
        nugget: Relative diagonal variance (positive; no further jitter).
        restarts: Optimizer starts per level.
        seed: Integer seed or stream; level ``t`` uses the child
            ``("nargp", t)``.
        max_iterations: BFGS iteration limit per start.

    Returns:
        The fitted :class:`NonlinearMultiFidelityGP`.

    Raises:
        TypeError: If an option has the wrong type.
        ValueError: If the designs are invalid or a level cannot be fitted.

    Equation:
        [EQ-SUR-NARGP-001], [EQ-SUR-NARGP-002], [EQ-SUR-NARGP-003].

    References:
        [REF-PERDIKARIS-ETAL-2017], [REF-RASMUSSEN-WILLIAMS-2006].

    """
    count = len(designs)
    if count < 1 or len(responses) != count:
        raise ValueError("need one response vector per design and one level at least")
    if kernel not in KERNELS:
        raise ValueError(f"kernel must be one of {KERNELS}")
    jitter = require_finite_real(nugget, name="nugget")
    if jitter <= 0.0:
        raise ValueError("nugget must be positive")
    root = resolve_stream(seed)
    data = [
        _require_design(d, r, i + 1)
        for i, (d, r) in enumerate(zip(designs, responses, strict=True))
    ]
    dimension = len(data[0][0][0])
    if any(len(points[0]) != dimension for points, _ in data):
        raise ValueError("all levels must share the input dimension")
    nested = all(set(data[i][0]) <= set(data[i - 1][0]) for i in range(1, count))
    levels: list[KernelGaussianProcess] = []
    for index, (points, values) in enumerate(data):
        variance, _ = _spread(values)
        stream = root.child("nargp", index + 1)
        ranges = [_spread([p[j] for p in points])[1] for j in range(dimension)]
        starts: list[Kernel] = []
        if index == 0:
            inputs = points
            for factor in _SCAN:
                starts.append(
                    StationaryKernel(
                        kernel, variance, tuple(factor * r for r in ranges)
                    )
                )
        else:
            partial = NonlinearMultiFidelityGP(tuple(levels), dimension, nested, kernel)
            lower = tuple(partial.predict_plugin(p, level=index)[0] for p in points)
            inputs = tuple((*p, z) for p, z in zip(points, lower, strict=True))
            width = _spread(lower)[1]
            for factor in _SCAN:
                for f_factor in _SCAN:
                    starts.append(
                        nargp_kernel(
                            dimension,
                            kernel=kernel,
                            rho_length_scales=tuple(factor * r for r in ranges),
                            f_length_scale=f_factor * width,
                            f_variance=variance,
                            delta_length_scales=tuple(factor * r for r in ranges),
                            delta_variance=0.1 * variance,
                        )
                    )
        try:
            levels.append(
                _fit_best_start(
                    inputs,
                    values,
                    starts,
                    nugget=jitter * variance,
                    restarts=restarts,
                    seed=stream,
                    max_iterations=max_iterations,
                )
            )
        except ValueError as error:
            raise ValueError(
                f"level {index + 1} could not be fitted: {error}"
            ) from error
    return NonlinearMultiFidelityGP(tuple(levels), dimension, nested, kernel)


@dataclass(frozen=True, slots=True)
class NargpValidation:
    """Held-out validation of the top level of a NARGP surrogate.

    Args:
        emulator: Library emulator diagnostics computed from the propagated
            means and a *diagonal* covariance of the propagated variances.
        rmse: Root-mean-square error of the propagated mean.
        plugin_rmse: Root-mean-square error of the plug-in mean.
        coverage: Fraction of outputs inside ``mean +/- 1.96 sd`` of the
            propagated law.
        plugin_coverage: The same with the plug-in mean and variance.
        max_relative_variance_mcse: Largest ratio of the Monte Carlo
            standard error of the propagated variance to the variance.
        max_mean_mcse: Largest Monte Carlo standard error of the mean.
        predictions: The propagated predictions at the validation inputs.

    Equation:
        [EQ-SUR-NARGP-005], [EQ-SUR-NARGP-007].

    References:
        [REF-BASTOS-OHAGAN-2009], [REF-PERDIKARIS-ETAL-2017].

    """

    emulator: EmulatorValidation
    rmse: float
    plugin_rmse: float
    coverage: float
    plugin_coverage: float
    max_relative_variance_mcse: float
    max_mean_mcse: float
    predictions: tuple[NargpPrediction, ...]


def validate_nargp(
    model: NonlinearMultiFidelityGP,
    points: Sequence[Sequence[float]],
    values: Sequence[float],
    *,
    samples: int = 300,
    seed: int | RandomStream | None = None,
    level: float = 0.95,
) -> NargpValidation:
    """Validate the top level on held-out runs [EQ-SUR-NARGP-007].

    The propagated means and variances at the validation inputs are passed
    to :func:`probstat.surrogates.gp_bayes.validate_emulator_moments` with a
    diagonal predictive covariance. The joint law of the mixture at several
    points is not available in closed form, so the correlation between test
    points is ignored: the standardized-error, coverage, and pivot checks are
    marginal screening diagnostics and the Mahalanobis statistic is
    approximate. The Monte Carlo errors of the propagated moments are
    reported so that the Monte Carlo noise can be compared with the
    emulator error.

    Args:
        model: Fitted surrogate.
        points: Validation inputs.
        values: Top-level responses at them.
        samples: Monte Carlo draws per point.
        seed: Integer seed or stream.
        level: Nominal credible level of the coverage check.

    Returns:
        :class:`NargpValidation`.

    Raises:
        TypeError: If ``model`` is not a :class:`NonlinearMultiFidelityGP`.
        ValueError: If sizes disagree or an input is invalid.

    Equation:
        [EQ-SUR-NARGP-007].

    References:
        [REF-BASTOS-OHAGAN-2009], [REF-PERDIKARIS-ETAL-2017].

    """
    if not isinstance(model, NonlinearMultiFidelityGP):
        raise TypeError("model must be a NonlinearMultiFidelityGP")
    observed = as_vector(values, name="values")
    if len(points) != len(observed) or len(observed) < 2:
        raise ValueError("need at least two validation points with one value each")
    predictions = model.predict_mc_many(points, samples=samples, seed=seed)
    means = tuple(p.mean for p in predictions)
    size = len(observed)
    covariance = tuple(
        tuple(predictions[i].variance if i == j else 0.0 for j in range(size))
        for i in range(size)
    )
    emulator = validate_emulator_moments(observed, means, covariance, level=level)
    count = len(observed)
    rmse = math.sqrt(
        math.fsum((o - p.mean) ** 2 for o, p in zip(observed, predictions, strict=True))
        / count
    )
    plugin_rmse = math.sqrt(
        math.fsum(
            (o - p.plugin_mean) ** 2 for o, p in zip(observed, predictions, strict=True)
        )
        / count
    )
    inside = sum(
        abs(o - p.mean) <= 1.96 * p.standard_deviation
        for o, p in zip(observed, predictions, strict=True)
    )
    plugin_inside = sum(
        abs(o - p.plugin_mean) <= 1.96 * math.sqrt(p.plugin_variance)
        for o, p in zip(observed, predictions, strict=True)
    )
    relative = max(
        (p.variance_mcse / p.variance if p.variance > 0.0 else 0.0) for p in predictions
    )
    return NargpValidation(
        emulator,
        rmse,
        plugin_rmse,
        inside / count,
        plugin_inside / count,
        relative,
        max(p.mean_mcse for p in predictions),
        predictions,
    )
