"""Recursive AR(1) co-kriging for simulators with several levels of fidelity.

A cheap, coarse simulator ``z_1`` and progressively more expensive ones
``z_2, ..., z_s`` are linked by the first-order autoregressive model of
Kennedy and O'Hagan

    ``Z_t(x) = rho_(t-1) Z_(t-1)(x) + delta_t(x)``,  ``delta_t`` independent
    of ``Z_(t-1)``,

with Gaussian-process priors on ``Z_1`` and on every discrepancy ``delta_t``.
On *nested* designs ``D_s subset ... subset D_1`` the joint posterior
factorizes: conditioning on all the data is the same as conditioning ``Z_1``
on ``z_1`` and every ``delta_t`` on ``z_t - rho_(t-1) z_(t-1)(D_t)``, which
are ``s`` independent ordinary GP regressions (Le Gratiet and Garnier). The
predictor of the top level is then assembled recursively
[EQ-SUR-MF-001] and equals the joint co-kriging predictor with the block
AR(1) covariance for the same hyperparameters; the test suite builds that
joint covariance explicitly and checks the equality.

Every level is a :class:`~probstat.surrogates.GaussianProcessRegression`, so
the kernels, the marginal-likelihood fit, and the prediction algebra are the
library's GP code. The scale factors ``rho`` are either given or estimated by
generalized least squares together with the discrepancy trend constant
inside the library's universal kriging fit [EQ-SUR-MF-002].

The predictive variance treats ``rho``, the trend constants, and the kernel
hyperparameters as known (plug-in). It therefore understates the uncertainty
on small designs; the extra terms of the universal co-kriging variance of Le
Gratiet and Garnier are not implemented. It is not a full predictive
uncertainty. Measured coverage of the nominal 95% interval
``m +/- 1.96 s`` of the top level over 40 random nested designs and 50 test
points each, for the two-fidelity pair of the tests
(``(6x - 2)^2 sin(12x - 4)`` and its cheap version): 0.82 with 11 cheap and
4 expensive runs (one standardized error in twenty exceeded 15), 0.94 with
15 and 5, 0.96 with 21 and 7, 0.97 with 30 and 10. For a second pair
(``sin(8x) + x`` with ``0.7 sin(8x + 0.3) + 0.2``) the coverage was 0.69
with 11 and 4 runs and 0.94 with 21 and 7.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.linalg import Vector, as_vector

from .gaussian_process import (
    KERNELS,
    GaussianProcessHyperparameters,
    GaussianProcessRegression,
    gaussian_process_log_marginal_likelihood,
)
from .pc_kriging import fit_universal_kriging

_LOG_BOX = 25.0


def gaussian_process_from_hyperparameters(
    points: Sequence[Sequence[float]],
    values: Sequence[float],
    *,
    kernel: str = "matern52",
    signal_variance: float,
    length_scales: Sequence[float],
    noise_variance: float = 0.0,
    mean: float = 0.0,
) -> GaussianProcessRegression:
    """Condition a GP with given hyperparameters and constant mean on data.

    No likelihood is maximized: the result is the posterior of
    ``f ~ GP(mean, sigma_f^2 r)`` given ``y_i = f(x_i) + e_i``,
    ``e_i ~ N(0, noise_variance)``, with predictive mean
    ``mean + k*' K^-1 (y - mean)`` and variance ``k** - k*' K^-1 k*``
    [EQ-SUR-GP-003]. Outputs are not standardized (``output_mean = mean``,
    ``output_scale = 1``), so all hyperparameters are in the units of the
    data.

    Args:
        points: Training inputs.
        values: Training outputs.
        kernel: ``"squared_exponential"``, ``"matern52"``, or ``"matern32"``.
        signal_variance: Process variance ``sigma_f^2 > 0``.
        length_scales: Positive length scales, one per input.
        noise_variance: Diagonal noise variance ``>= 0`` (``0`` interpolates;
            the kernel matrix must then be numerically positive definite).
        mean: Known constant prior mean.

    Returns:
        Fitted regression with ``converged=True``, ``restarts=0``, and
        ``evaluations=1`` (the single factorization of the likelihood value).

    Raises:
        ValueError: If an input is invalid, a log hyperparameter leaves
            ``|theta| <= 25``, or the kernel matrix is not positive definite.

    Equation:
        [EQ-SUR-GP-003], [EQ-SUR-GP-004].

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Algorithm 2.1].

    """
    if kernel not in KERNELS:
        raise ValueError(f"kernel must be one of {KERNELS}")
    x = tuple(as_vector(point, name="point") for point in points)
    y = as_vector(values, name="values")
    if len(x) != len(y) or not x:
        raise ValueError("need at least one point with one value each")
    dimension = len(x[0])
    if dimension == 0 or any(len(point) != dimension for point in x):
        raise ValueError("points must be non-empty and equally long")
    signal = require_finite_real(signal_variance, name="signal_variance")
    if signal <= 0.0:
        raise ValueError("signal_variance must be positive")
    scales = as_vector(length_scales, name="length_scales")
    if len(scales) != dimension or any(scale <= 0.0 for scale in scales):
        raise ValueError("length_scales must be positive, one per input")
    noise = require_finite_real(noise_variance, name="noise_variance")
    if noise < 0.0:
        raise ValueError("noise_variance must be non-negative")
    centre = require_finite_real(mean, name="mean")
    theta = (math.log(signal), *(math.log(scale) for scale in scales))
    if any(abs(value) > _LOG_BOX for value in theta):
        raise ValueError("log hyperparameters must satisfy |theta| <= 25")
    residuals = tuple(value - centre for value in y)
    likelihood, _ = gaussian_process_log_marginal_likelihood(
        x, residuals, theta, kernel=kernel, nugget=noise
    )
    return GaussianProcessRegression(
        kernel,
        x,
        y,
        GaussianProcessHyperparameters(signal, scales, noise),
        theta,
        noise,
        False,
        centre,
        1.0,
        likelihood,
        True,
        0,
        1,
    )


@dataclass(frozen=True, slots=True)
class RecursiveCoKriging:
    """A fitted recursive AR(1) co-kriging surrogate of ``s`` fidelity levels.

    ``levels[0]`` is the GP of the cheapest code ``Z_1``; ``levels[t]`` for
    ``t >= 1`` is the GP of the discrepancy ``delta_(t+1)`` trained on
    ``z_(t+1) - rho_t z_t`` at the design of level ``t + 1``. The object
    satisfies the :class:`~probstat.surrogates.ProbabilisticSurrogate`
    protocol for the highest level.

    Args:
        levels: One Gaussian process per level (``s >= 1``).
        scale_factors: ``(rho_1, ..., rho_(s-1))``.
        scale_estimated: Whether the scale factors were estimated by
            generalized least squares (``False`` when they were given).
        converged: Whether the hyperparameter optimizer of every level met
            its tolerance.
        likelihood_evaluations: Kernel factorizations spent on all
            hyperparameter fits.

    Raises:
        ValueError: If the numbers of levels and scale factors disagree or
            the levels have different input dimensions.

    Equation:
        [EQ-SUR-MF-001].

    References:
        [REF-KENNEDY-OHAGAN-2000], [REF-LE-GRATIET-GARNIER-2014].

    """

    levels: tuple[GaussianProcessRegression, ...]
    scale_factors: Vector
    scale_estimated: bool = False
    converged: bool = True
    likelihood_evaluations: int = 0

    def __post_init__(self) -> None:
        """Validate the level and scale-factor counts.

        Raises:
            ValueError: If the structure is inconsistent.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION]. No equation
            is evaluated.

        """
        levels = tuple(self.levels)
        if not levels:
            raise ValueError("at least one level is required")
        factors = as_vector(self.scale_factors, name="scale_factors")
        if len(factors) != len(levels) - 1:
            raise ValueError("need one scale factor per level above the first")
        dimension = len(levels[0].points[0])
        if any(len(level.points[0]) != dimension for level in levels):
            raise ValueError("all levels must share the input dimension")
        object.__setattr__(self, "levels", levels)
        object.__setattr__(self, "scale_factors", factors)

    @property
    def level_count(self) -> int:
        """Return the number of fidelity levels ``s``.

        References:
            [REF-LE-GRATIET-GARNIER-2014]. No equation is evaluated.

        """
        return len(self.levels)

    @property
    def design_sizes(self) -> tuple[int, ...]:
        """Return the number of training runs of every level.

        References:
            [REF-LE-GRATIET-GARNIER-2014]. No equation is evaluated.

        """
        return tuple(len(level.points) for level in self.levels)

    def level_predictions(
        self, x: Sequence[float], *, include_noise: bool = False
    ) -> tuple[tuple[float, float], ...]:
        """Return the predictive mean and variance of every level at ``x``.

        ``m_1, s_1^2`` come from the first GP; for ``t >= 2``
        ``m_t = rho_(t-1) m_(t-1) + m_delta_t`` and
        ``s_t^2 = rho_(t-1)^2 s_(t-1)^2 + s_delta_t^2``.

        Args:
            x: Input vector.
            include_noise: Add each GP's diagonal noise variance (prediction
                of a new noisy run instead of the latent response).

        Returns:
            ``((m_1, s_1^2), ..., (m_s, s_s^2))``.

        Raises:
            ValueError: If ``x`` has the wrong dimension.

        Equation:
            [EQ-SUR-MF-001].

        References:
            [REF-LE-GRATIET-GARNIER-2014], [REF-KENNEDY-OHAGAN-2000].

        """
        mean, variance = self.levels[0].predict_with_variance(
            x, include_noise=include_noise
        )
        result = [(mean, variance)]
        for rho, level in zip(self.scale_factors, self.levels[1:], strict=True):
            shift, extra = level.predict_with_variance(x, include_noise=include_noise)
            mean = rho * mean + shift
            variance = rho * rho * variance + extra
            result.append((mean, variance))
        return tuple(result)

    def predict_with_variance(
        self,
        x: Sequence[float],
        *,
        level: int | None = None,
        include_noise: bool = False,
    ) -> tuple[float, float]:
        """Return the predictive mean and variance of one level at ``x``.

        Args:
            x: Input vector.
            level: Fidelity level ``1..s``; ``None`` means the highest.
            include_noise: Add the GPs' diagonal noise variances.

        Returns:
            Predictive mean and plug-in variance (hyperparameters, trend
            constants, and scale factors treated as known). Intervals built
            from it are too narrow on small designs: measured coverage of
            the nominal 95% interval was 0.69 to 0.82 with four expensive
            runs (see the module docstring).

        Raises:
            ValueError: If ``level`` is out of range or ``x`` has the wrong
                dimension.

        Equation:
            [EQ-SUR-MF-001].

        References:
            [REF-LE-GRATIET-GARNIER-2014], [REF-KENNEDY-OHAGAN-2000].

        """
        count = len(self.levels)
        if level is None:
            index = count
        elif isinstance(level, bool) or not isinstance(level, int):
            raise ValueError("level must be an integer")
        else:
            index = level
        if not 1 <= index <= count:
            raise ValueError(f"level must lie in 1..{count}")
        return self.level_predictions(x, include_noise=include_noise)[index - 1]

    def predict(self, x: Sequence[float]) -> float:
        """Return the predictive mean of the highest level at ``x``.

        Args:
            x: Input vector.

        Returns:
            ``m_s(x)``.

        Equation:
            [EQ-SUR-MF-001].

        References:
            [REF-LE-GRATIET-GARNIER-2014].

        """
        return self.predict_with_variance(x)[0]


def _nested_values(
    design: Sequence[Vector],
    lookup: dict[Vector, float],
    level: int,
) -> Vector:
    """Return the level ``level - 1`` responses at the points of ``design``.

    Raises:
        ValueError: If a point is missing from the lower design.

    References:
        [REF-LE-GRATIET-GARNIER-2014] (nested designs). No equation is
        evaluated.

    """
    values = []
    for point in design:
        if point not in lookup:
            raise ValueError(
                f"designs must be nested: a point of level {level} is not in "
                f"the design of level {level - 1}"
            )
        values.append(lookup[point])
    return tuple(values)


@dataclass(frozen=True, slots=True)
class _LowerFidelityTrend:
    """Regressor ``x -> (1, z_(t-1)(x))`` defined on a lower design only.

    Args:
        lookup: Observed lower-fidelity response per design point.

    References:
        [EQ-SUR-MF-002], [REF-LE-GRATIET-GARNIER-2014].

    """

    lookup: dict[Vector, float]

    def __call__(self, x: Vector) -> Vector:
        """Return ``(1, z_(t-1)(x))`` for a point of the lower design.

        Raises:
            ValueError: If ``x`` is not a point of the lower design.

        References:
            [EQ-SUR-MF-002], [REF-LE-GRATIET-GARNIER-2014].

        """
        key = tuple(x)
        if key not in self.lookup:
            raise ValueError("the lower-fidelity trend is known on its design only")
        return (1.0, self.lookup[key])


def fit_recursive_cokriging(
    designs: Sequence[Sequence[Sequence[float]]],
    responses: Sequence[Sequence[float]],
    *,
    kernel: str = "matern52",
    nugget: float = 1e-10,
    restarts: int = 10,
    seed: int | RandomStream | None = None,
    scale_factors: Sequence[float] | None = None,
) -> RecursiveCoKriging:
    """Fit recursive AR(1) co-kriging on nested designs, cheapest level first.

    Level 1 is ordinary kriging of ``z_1``. For each higher level ``t`` with
    design ``D_t subset D_(t-1)`` the model
    ``z_t(D_t) = rho z_(t-1)(D_t) + beta + delta_t(D_t)`` is universal
    kriging with the two regressors ``(1, z_(t-1)(x))``: for given length
    scales ``(beta, rho)`` is the generalized least-squares estimate and the
    process variance is profiled, and the length scales maximize the
    concentrated likelihood (:func:`fit_universal_kriging`,
    [EQ-SUR-MF-002]). The lower code enters only through its observed runs,
    which is why the designs must be nested. With given ``scale_factors`` the
    discrepancy ``z_t - rho z_(t-1)(D_t)`` is fitted by ordinary kriging.

    Each fitted level is stored as a
    :class:`~probstat.surrogates.GaussianProcessRegression` with the
    estimated trend constant as known mean, so its predictive variance is
    the simple-kriging variance: the uncertainty of ``rho``, of the trend
    constants, and of the kernel hyperparameters is not propagated.

    If a level is reproduced exactly by ``rho z_(t-1) + beta`` the profiled
    process variance is zero; it is then floored at ``1e-10`` times the
    sample variance of that level's responses so that a GP can be formed
    (its mean is the constant ``beta`` and its variance is negligible).

    Args:
        designs: One design per level, cheapest first; each higher design
            must be a subset (exact coordinates) of the one below and, when
            its scale factor is estimated, hold at least three points.
        responses: Responses of each level at its design.
        kernel: Kernel name shared by all levels.
        nugget: Diagonal added to each correlation matrix, relative to the
            process variance.
        restarts: Optimizer starts of each level. The default of ten is
            deliberate: with a handful of expensive runs the concentrated
            likelihood is often bimodal (a short length scale with a poor
            ``rho`` against a long one with the right ``rho``), and three
            starts missed the higher mode in the two-fidelity test problem
            of this module's tests.
        seed: Integer seed or stream; level ``t`` uses the child
            ``("cokriging", t)``.
        scale_factors: Optional fixed ``(rho_1, ..., rho_(s-1))``.

    Returns:
        Fitted co-kriging surrogate; ``converged`` is the conjunction of the
        optimizer flags of all levels.

    Raises:
        ValueError: If the designs are not nested, contain repeated points,
            a regression is rank deficient (for example constant lower
            responses on a higher design), or an option is invalid.

    Equation:
        [EQ-SUR-MF-001], [EQ-SUR-MF-002].

    References:
        [REF-LE-GRATIET-GARNIER-2014], [REF-KENNEDY-OHAGAN-2000].

    """
    count = len(designs)
    if count < 1 or len(responses) != count:
        raise ValueError("need one response vector per design and one level at least")
    fixed: Vector | None = None
    if scale_factors is not None:
        fixed = as_vector(scale_factors, name="scale_factors")
        if len(fixed) != count - 1:
            raise ValueError("need one scale factor per level above the first")
    points = [tuple(as_vector(p, name="point") for p in design) for design in designs]
    outputs = [as_vector(values, name="responses") for values in responses]
    for index, (design, values) in enumerate(zip(points, outputs, strict=True)):
        if len(design) != len(values):
            raise ValueError(f"level {index + 1} needs one response per design point")
        if len(set(design)) != len(design):
            raise ValueError(f"level {index + 1} design has repeated points")
    root = resolve_stream(seed)
    levels: list[GaussianProcessRegression] = []
    factors: list[float] = []
    converged = True
    evaluations = 0
    for index in range(count):
        design = points[index]
        higher = outputs[index]
        stream = root.child("cokriging", index + 1)
        regressors: Callable[[Vector], Vector] | None = None
        rho = 0.0
        data = higher
        if index > 0:
            lookup = dict(zip(points[index - 1], outputs[index - 1], strict=True))
            lower = _nested_values(design, lookup, index + 1)
            if fixed is None:
                regressors = _LowerFidelityTrend(lookup)
            else:
                rho = fixed[index - 1]
                data = tuple(h - rho * z for h, z in zip(higher, lower, strict=True))
        try:
            kriging = fit_universal_kriging(
                design,
                data,
                regressors=regressors,
                kernel=kernel,
                nugget=nugget,
                restarts=restarts,
                seed=stream,
            )
        except ValueError as error:
            message = f"level {index + 1} could not be fitted: {error}"
            raise ValueError(message) from error
        beta = kriging.trend_coefficients[0]
        if regressors is not None:
            rho = kriging.trend_coefficients[1]
            data = tuple(h - rho * z for h, z in zip(higher, lower, strict=True))
        centre = math.fsum(higher) / len(higher)
        spread = math.fsum((value - centre) ** 2 for value in higher) / len(higher)
        floor = 1e-10 * spread if spread > 0.0 else 1e-10
        signal = max(kriging.process_variance, floor)
        levels.append(
            gaussian_process_from_hyperparameters(
                design,
                data,
                kernel=kernel,
                signal_variance=signal,
                length_scales=kriging.length_scales,
                noise_variance=kriging.nugget * signal,
                mean=beta,
            )
        )
        if index > 0:
            factors.append(rho)
        converged = converged and kriging.converged
        evaluations += kriging.evaluations
    return RecursiveCoKriging(
        tuple(levels),
        tuple(factors),
        scale_estimated=fixed is None and count > 1,
        converged=converged,
        likelihood_evaluations=evaluations,
    )
