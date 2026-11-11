"""Score-based diffusion: forward SDEs, score matching, and reverse-time samplers.

A forward SDE ``dx = f(x, t) dt + g(t) dw`` on ``t in [0, 1]`` turns the data
law ``p_0`` into a tractable prior ``p_1``. Both families provided here have
a Gaussian perturbation kernel ``p_0t(x_t | x_0) = N(m(t) x_0, v(t) I)``:

* variance preserving (VP), ``f = -beta(t) x / 2``, ``g = sqrt(beta(t))``
  with a linear ``beta`` [EQ-PROC-DIFF-001];
* variance exploding (VE), ``f = 0``, ``g = sqrt(d sigma^2 / dt)`` with a
  geometric ``sigma(t)`` [EQ-PROC-DIFF-002].

Generation runs time backwards. Anderson's reverse-time SDE
``dx = [f - g^2 grad log p_t] dt + g dw_bar`` [EQ-PROC-DIFF-005] and the
probability-flow ODE ``dx/dt = f - g^2 grad log p_t / 2`` [EQ-PROC-DIFF-007]
both carry ``p_1`` back to ``p_0``; they need only the score
``grad log p_t``. :class:`GaussianScore` gives that score in closed form for
Gaussian data [EQ-PROC-DIFF-003], so every sampler can be verified without
training. :func:`train_score_network` fits a time-conditioned
:class:`~probstat.numerics.MLP` by denoising score matching with the
weighting ``lambda(t)`` [EQ-PROC-DIFF-004], differentiated by reverse mode
and minimized by :func:`~probstat.optimization.adam_minimize`.

Samplers: reverse-time Euler-Maruyama, predictor-corrector with a Langevin
corrector [EQ-PROC-DIFF-006], and the probability-flow ODE with the classical
Runge-Kutta or Heun integrator. The arithmetic is scalar Python; the methods
are meant for one- and two-dimensional laws and small networks.

References:
    [REF-SONG-ETAL-2021, Sections 3-4 and Appendices C, G],
    [REF-HO-JAIN-ABBEEL-2020], [REF-ANDERSON-1982], [REF-VINCENT-2011].

"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.numerics import MLP, reverse_value_and_gradient
from probstat.optimization import AdamResult, adam_minimize

from .continuous import _positive, _positive_integer

Point = tuple[float, ...]
ScoreFunction = Callable[[Point, float], Sequence[float]]

ODE_METHODS: tuple[str, ...] = ("rk4", "heun")


class ForwardSDE(Protocol):
    """Forward diffusion with an affine Gaussian perturbation kernel.

    Implementations describe ``dx = f(x, t) dt + g(t) dw`` on ``[0, 1]`` with
    ``x_t | x_0 ~ N(m(t) x_0, v(t) I)`` and a prior ``N(0, prior_variance I)``
    close to ``p_1``.

    References:
        [REF-SONG-ETAL-2021, Equation 5 and Section 3.4].

    """

    @property
    def prior_variance(self) -> float:
        """Return the variance of the Gaussian prior that replaces ``p_1``.

        References:
            [REF-SONG-ETAL-2021, Section 3.4].

        """
        ...

    @property
    def default_end_time(self) -> float:
        """Return the smallest time a sampler integrates down to by default.

        References:
            [REF-SONG-ETAL-2021, Appendix C].

        """
        ...

    def drift(self, x: Sequence[Any], t: float) -> list[Any]:
        """Return ``f(x, t)``.

        References:
            [REF-SONG-ETAL-2021, Equation 5].

        """
        ...

    def diffusion(self, t: float) -> float:
        """Return ``g(t)``.

        References:
            [REF-SONG-ETAL-2021, Equation 5].

        """
        ...

    def mean_coefficient(self, t: float) -> float:
        """Return ``m(t)`` in the perturbation-kernel mean ``m(t) x_0``.

        References:
            [EQ-PROC-DIFF-001], [EQ-PROC-DIFF-002].

        """
        ...

    def kernel_variance(self, t: float) -> float:
        """Return ``v(t)``, the perturbation-kernel variance per coordinate.

        References:
            [EQ-PROC-DIFF-001], [EQ-PROC-DIFF-002].

        """
        ...

    def corrector_scale(self, t: float, step: float) -> float:
        """Return the factor ``alpha`` of the Langevin corrector step size.

        References:
            [EQ-PROC-DIFF-006], [REF-SONG-ETAL-2021, Appendix G].

        """
        ...


def _unit_time(t: object, *, name: str = "t") -> float:
    """Validate a diffusion time in ``[0, 1]``.

    Args:
        t: Candidate time.
        name: Parameter name for error messages.

    Returns:
        The time as a float.

    Raises:
        ValueError: If the time is not finite or lies outside ``[0, 1]``.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].
        No equation is evaluated.

    """
    value = require_finite_real(t, name=name)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return value


@dataclass(frozen=True, slots=True)
class VariancePreservingSDE:
    """Variance-preserving SDE ``dx = -beta(t) x / 2 dt + sqrt(beta(t)) dw``.

    ``beta(t) = beta_min + t (beta_max - beta_min)``; with
    ``B(t) = beta_min t + (beta_max - beta_min) t^2 / 2`` the kernel is
    ``N(exp(-B/2) x_0, (1 - exp(-B)) I)`` and the prior is ``N(0, I)``
    [EQ-PROC-DIFF-001]. It is the continuous limit of the DDPM noising chain.

    Args:
        beta_min: Positive noise rate at ``t = 0`` (default ``0.1``).
        beta_max: Noise rate at ``t = 1``, at least ``beta_min`` (default
            ``20``).

    References:
        [EQ-PROC-DIFF-001], [REF-SONG-ETAL-2021, Equations 11 and 33],
        [REF-HO-JAIN-ABBEEL-2020, Equation 4].

    """

    beta_min: float = 0.1
    beta_max: float = 20.0

    def __post_init__(self) -> None:
        """Validate the noise schedule.

        Raises:
            ValueError: Unless ``0 < beta_min <= beta_max`` are finite.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].
            No equation is evaluated.

        """
        low = _positive(self.beta_min, name="beta_min")
        high = _positive(self.beta_max, name="beta_max")
        if high < low:
            raise ValueError("beta_max must be at least beta_min")

    @property
    def prior_variance(self) -> float:
        """Return ``1``, the variance of the standard-normal prior.

        References:
            [EQ-PROC-DIFF-001].

        """
        return 1.0

    @property
    def default_end_time(self) -> float:
        """Return ``1e-3``, the sampling end time used for VP models.

        References:
            [REF-SONG-ETAL-2021, Appendix C].

        """
        return 1e-3

    def beta(self, t: float) -> float:
        """Return the noise rate ``beta(t)``.

        Args:
            t: Time in ``[0, 1]``.

        Returns:
            ``beta_min + t (beta_max - beta_min)``.

        References:
            [EQ-PROC-DIFF-001].

        """
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def integrated_beta(self, t: float) -> float:
        """Return ``B(t) = int_0^t beta(s) ds``.

        Args:
            t: Time in ``[0, 1]``.

        Returns:
            ``beta_min t + (beta_max - beta_min) t^2 / 2``.

        References:
            [EQ-PROC-DIFF-001].

        """
        return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t * t

    def drift(self, x: Sequence[Any], t: float) -> list[Any]:
        """Return ``f(x, t) = -beta(t) x / 2``.

        Args:
            x: State vector (floats or differentiable variables).
            t: Time in ``[0, 1]``.

        Returns:
            Drift vector.

        References:
            [EQ-PROC-DIFF-001].

        """
        half = -0.5 * self.beta(t)
        return [half * value for value in x]

    def diffusion(self, t: float) -> float:
        """Return ``g(t) = sqrt(beta(t))``.

        Args:
            t: Time in ``[0, 1]``.

        Returns:
            Diffusion coefficient.

        References:
            [EQ-PROC-DIFF-001].

        """
        return math.sqrt(self.beta(t))

    def mean_coefficient(self, t: float) -> float:
        """Return ``m(t) = exp(-B(t) / 2)``.

        Args:
            t: Time in ``[0, 1]``.

        Returns:
            Kernel mean coefficient.

        References:
            [EQ-PROC-DIFF-001].

        """
        return math.exp(-0.5 * self.integrated_beta(t))

    def kernel_variance(self, t: float) -> float:
        """Return ``v(t) = 1 - exp(-B(t))``.

        Args:
            t: Time in ``[0, 1]``.

        Returns:
            Kernel variance, computed with ``expm1`` for small ``t``.

        References:
            [EQ-PROC-DIFF-001].

        """
        return -math.expm1(-self.integrated_beta(t))

    def corrector_scale(self, t: float, step: float) -> float:
        """Return ``alpha = 1 - beta(t) h`` for the Langevin corrector.

        Args:
            t: Time of the corrector.
            step: Time step ``h`` of the discretization.

        Returns:
            The discrete ``alpha_i = 1 - beta_i`` of the VP chain.

        Raises:
            ValueError: If ``beta(t) h >= 1`` (too few steps).

        References:
            [EQ-PROC-DIFF-006], [REF-SONG-ETAL-2021, Appendix G].

        """
        alpha = 1.0 - self.beta(t) * step
        if alpha <= 0.0:
            raise ValueError("beta(t) * step must be below 1; use more steps")
        return alpha


@dataclass(frozen=True, slots=True)
class VarianceExplodingSDE:
    """Variance-exploding SDE ``dx = sqrt(d sigma^2(t) / dt) dw``.

    ``sigma(t) = sigma_min (sigma_max / sigma_min)^t``, so
    ``g(t) = sigma(t) sqrt(2 log(sigma_max / sigma_min))``. The exact kernel
    from time ``0`` is ``N(x_0, (sigma(t)^2 - sigma_min^2) I)`` and the prior
    is ``N(0, sigma_max^2 I)`` [EQ-PROC-DIFF-002].

    Args:
        sigma_min: Positive smallest noise scale (default ``0.01``).
        sigma_max: Largest noise scale, above ``sigma_min`` (default ``50``);
            it should exceed the spread of the data.

    References:
        [EQ-PROC-DIFF-002], [REF-SONG-ETAL-2021, Equations 9, 30, and 31].

    """

    sigma_min: float = 0.01
    sigma_max: float = 50.0

    def __post_init__(self) -> None:
        """Validate the noise scales.

        Raises:
            ValueError: Unless ``0 < sigma_min < sigma_max`` are finite.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].
            No equation is evaluated.

        """
        low = _positive(self.sigma_min, name="sigma_min")
        high = _positive(self.sigma_max, name="sigma_max")
        if high <= low:
            raise ValueError("sigma_max must exceed sigma_min")

    @property
    def prior_variance(self) -> float:
        """Return ``sigma_max^2``.

        References:
            [EQ-PROC-DIFF-002].

        """
        return self.sigma_max * self.sigma_max

    @property
    def default_end_time(self) -> float:
        """Return ``1e-5``, the sampling end time used for VE models.

        References:
            [REF-SONG-ETAL-2021, Appendix C].

        """
        return 1e-5

    def sigma(self, t: float) -> float:
        """Return the noise scale ``sigma(t)``.

        Args:
            t: Time in ``[0, 1]``.

        Returns:
            ``sigma_min (sigma_max / sigma_min)^t``.

        References:
            [EQ-PROC-DIFF-002].

        """
        return self.sigma_min * math.exp(t * math.log(self.sigma_max / self.sigma_min))

    def drift(self, x: Sequence[Any], t: float) -> list[Any]:
        """Return the zero drift.

        Args:
            x: State vector.
            t: Time in ``[0, 1]``.

        Returns:
            Zero vector of the same length.

        References:
            [EQ-PROC-DIFF-002].

        """
        return [0.0 for _ in x]

    def diffusion(self, t: float) -> float:
        """Return ``g(t) = sigma(t) sqrt(2 log(sigma_max / sigma_min))``.

        Args:
            t: Time in ``[0, 1]``.

        Returns:
            Diffusion coefficient.

        References:
            [EQ-PROC-DIFF-002].

        """
        ratio = math.log(self.sigma_max / self.sigma_min)
        return self.sigma(t) * math.sqrt(2.0 * ratio)

    def mean_coefficient(self, t: float) -> float:
        """Return ``m(t) = 1``.

        Args:
            t: Time in ``[0, 1]``.

        Returns:
            ``1.0``.

        References:
            [EQ-PROC-DIFF-002].

        """
        return 1.0

    def kernel_variance(self, t: float) -> float:
        """Return ``v(t) = sigma(t)^2 - sigma_min^2``.

        Args:
            t: Time in ``[0, 1]``.

        Returns:
            Exact kernel variance of the SDE started at time ``0``.

        References:
            [EQ-PROC-DIFF-002].

        """
        ratio = math.log(self.sigma_max / self.sigma_min)
        return self.sigma_min * self.sigma_min * math.expm1(2.0 * ratio * t)

    def corrector_scale(self, t: float, step: float) -> float:
        """Return ``alpha = 1`` for the VE Langevin corrector.

        Args:
            t: Time of the corrector.
            step: Time step (unused).

        Returns:
            ``1.0``.

        References:
            [EQ-PROC-DIFF-006], [REF-SONG-ETAL-2021, Appendix G].

        """
        return 1.0


def perturbation_moments(
    sde: ForwardSDE, x0: Sequence[float], t: float
) -> tuple[Point, float]:
    """Return the mean and per-coordinate variance of ``x_t | x_0``.

    Args:
        sde: Forward SDE.
        x0: Clean data point.
        t: Time in ``[0, 1]``.

    Returns:
        ``(m(t) x_0, v(t))``.

    Raises:
        ValueError: If ``t`` is outside ``[0, 1]`` or ``x0`` is not finite.

    References:
        [EQ-PROC-DIFF-001], [EQ-PROC-DIFF-002].

    """
    time = _unit_time(t)
    point = _as_point(x0, name="x0")
    coefficient = sde.mean_coefficient(time)
    return tuple(coefficient * value for value in point), sde.kernel_variance(time)


def perturb(
    sde: ForwardSDE,
    x0: Sequence[float],
    t: float,
    *,
    seed: int | RandomStream | None = None,
) -> Point:
    """Draw ``x_t ~ N(m(t) x_0, v(t) I)`` from the perturbation kernel.

    Args:
        sde: Forward SDE.
        x0: Clean data point.
        t: Time in ``[0, 1]``.
        seed: Integer seed or stream.

    Returns:
        One perturbed point.

    References:
        [EQ-PROC-DIFF-001], [EQ-PROC-DIFF-002].

    """
    mean, variance = perturbation_moments(sde, x0, t)
    generator = resolve_stream(seed).child("perturb").generator()
    scale = math.sqrt(variance)
    return tuple(value + scale * generator.gauss(0.0, 1.0) for value in mean)


def _as_point(value: float | Sequence[float], *, name: str) -> Point:
    """Normalize a number or a sequence of numbers into a finite point.

    Args:
        value: Scalar (a one-dimensional point) or sequence.
        name: Parameter name for error messages.

    Returns:
        Tuple of finite floats.

    Raises:
        ValueError: If the point is empty or not finite.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].
        No equation is evaluated.

    """
    if isinstance(value, int | float):
        return (require_finite_real(value, name=name),)
    point = tuple(
        require_finite_real(v, name=f"{name}[{i}]") for i, v in enumerate(value)
    )
    if not point:
        raise ValueError(f"{name} must not be empty")
    return point


def _as_points(
    data: Sequence[float | Sequence[float]], *, name: str
) -> tuple[Point, ...]:
    """Normalize data into points of one common dimension.

    Args:
        data: Non-empty sequence of numbers or of equal-length sequences.
        name: Parameter name for error messages.

    Returns:
        Tuple of points.

    Raises:
        ValueError: If the data are empty, ragged, or not finite.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].
        No equation is evaluated.

    """
    points = tuple(_as_point(item, name=f"{name}[{i}]") for i, item in enumerate(data))
    if not points:
        raise ValueError(f"{name} must not be empty")
    dimension = len(points[0])
    if any(len(point) != dimension for point in points):
        raise ValueError(f"every point in {name} must have the same dimension")
    return points


@dataclass(frozen=True)
class GaussianScore:
    """Exact score of Gaussian data diffused by a forward SDE.

    For data ``N(mu, diag(s^2))`` the marginal at time ``t`` is
    ``N(m(t) mu, diag(m(t)^2 s^2 + v(t)))`` and its score is
    ``-(x - m(t) mu) / (m(t)^2 s^2 + v(t))`` coordinatewise
    [EQ-PROC-DIFF-003]. Calling the object evaluates the score, so it can be
    passed to every sampler in place of a trained network.

    Args:
        sde: Forward SDE.
        mean: Data mean ``mu`` (a number for one dimension).
        variance: Positive data variances ``s^2``, one per coordinate (a
            number is broadcast).

    References:
        [EQ-PROC-DIFF-003], [REF-SONG-ETAL-2021, Section 3.3].

    """

    sde: ForwardSDE
    mean: float | Sequence[float]
    variance: float | Sequence[float]
    _mean: Point = field(init=False, repr=False)
    _variance: Point = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate and normalize the data moments.

        Raises:
            ValueError: If a variance is not positive or the lengths differ.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].
            No equation is evaluated.

        """
        mean = _as_point(self.mean, name="mean")
        if isinstance(self.variance, int | float):
            variance = tuple(
                _positive(self.variance, name="variance") for _ in mean
            )
        else:
            variance = tuple(
                _positive(v, name=f"variance[{i}]") for i, v in enumerate(self.variance)
            )
        if len(variance) != len(mean):
            raise ValueError("mean and variance must have the same length")
        object.__setattr__(self, "_mean", mean)
        object.__setattr__(self, "_variance", variance)

    @property
    def dimension(self) -> int:
        """Return the data dimension.

        References:
            Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

        """
        return len(self._mean)

    def marginal_mean(self, t: float) -> Point:
        """Return the marginal mean ``m(t) mu`` at time ``t``.

        Args:
            t: Time in ``[0, 1]``.

        Returns:
            Mean vector.

        References:
            [EQ-PROC-DIFF-003].

        """
        coefficient = self.sde.mean_coefficient(_unit_time(t))
        return tuple(coefficient * value for value in self._mean)

    def marginal_variance(self, t: float) -> Point:
        """Return the marginal variances ``m(t)^2 s^2 + v(t)`` at time ``t``.

        Args:
            t: Time in ``[0, 1]``.

        Returns:
            Variance vector.

        References:
            [EQ-PROC-DIFF-003].

        """
        time = _unit_time(t)
        coefficient = self.sde.mean_coefficient(time)
        noise = self.sde.kernel_variance(time)
        return tuple(coefficient * coefficient * s2 + noise for s2 in self._variance)

    def __call__(self, x: Sequence[float], t: float) -> Point:
        """Return ``grad log p_t(x)``.

        Args:
            x: Point of the data dimension.
            t: Time in ``[0, 1]``.

        Returns:
            Score vector.

        Raises:
            ValueError: If ``x`` has the wrong length.

        References:
            [EQ-PROC-DIFF-003].

        """
        if len(x) != self.dimension:
            raise ValueError(f"expected a point of dimension {self.dimension}")
        means = self.marginal_mean(t)
        variances = self.marginal_variance(t)
        return tuple(
            -(value - m) / v for value, m, v in zip(x, means, variances, strict=True)
        )


@dataclass(frozen=True)
class ScoreNetwork:
    """Time-conditioned score model ``s_theta(x, t)`` built on an MLP.

    The network sees ``(c(t) x, 2 t - 1)`` with ``c(t) = 1 / sqrt(m(t)^2 +
    v(t))`` (the inverse standard deviation of ``x_t`` for unit-variance
    data) and its output is divided by the kernel standard deviation:
    ``s_theta(x, t) = N_theta(c(t) x, 2 t - 1) / sqrt(v(t))``. With the
    weighting ``lambda(t) = v(t)`` the denoising objective then reduces to
    the noise-prediction loss of DDPM [EQ-PROC-DIFF-004].

    Args:
        sde: Forward SDE.
        dimension: Data dimension ``d``.
        hidden: Hidden-layer widths (default one layer of 16).
        activation: Hidden activation of the MLP (default ``"tanh"``).

    References:
        [EQ-PROC-DIFF-004], [EQ-NN-MLP-001],
        [REF-HO-JAIN-ABBEEL-2020, Equation 14].

    """

    sde: ForwardSDE
    dimension: int = 1
    hidden: tuple[int, ...] = (16,)
    activation: str = "tanh"
    network: MLP = field(init=False)

    def __post_init__(self) -> None:
        """Validate the dimension and build the network.

        Raises:
            ValueError: For a non-positive dimension or invalid widths.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].
            No equation is evaluated.

        """
        dimension = _positive_integer(self.dimension, name="dimension")
        widths = tuple(self.hidden)
        object.__setattr__(self, "hidden", widths)
        network = MLP((dimension + 1, *widths, dimension), activation=self.activation)
        object.__setattr__(self, "network", network)

    @property
    def parameter_count(self) -> int:
        """Return the number of network parameters.

        References:
            [EQ-NN-MLP-001].

        """
        return self.network.parameter_count

    def initial_parameters(self, seed: int | RandomStream) -> tuple[float, ...]:
        """Return Glorot-uniform initial parameters.

        Args:
            seed: Integer seed or stream.

        Returns:
            Flat parameter vector.

        References:
            [EQ-NN-GLOROT-001].

        """
        return self.network.initial_parameters(seed)

    def noise_prediction(
        self, parameters: Sequence[Any], x: Sequence[Any], t: float
    ) -> list[Any]:
        """Return the raw network output ``N_theta(c(t) x, 2 t - 1)``.

        Args:
            parameters: Flat parameter vector (floats or tape variables).
            x: Point of the data dimension.
            t: Time in ``(0, 1]``.

        Returns:
            Network output, an estimate of ``-z`` for ``x = m x_0 + sqrt(v) z``.

        References:
            [EQ-PROC-DIFF-004].

        """
        variance = self.sde.kernel_variance(t)
        total = self.sde.mean_coefficient(t) ** 2 + variance
        scale = 1.0 / math.sqrt(total)
        level = math.sqrt(variance / total)
        return self.network.forward(
            parameters, [scale * v for v in x] + [2.0 * level - 1.0]
        )

    def score(self, parameters: Sequence[float], x: Sequence[float], t: float) -> Point:
        """Return ``s_theta(x, t)`` as floats.

        Args:
            parameters: Flat parameter vector.
            x: Point of the data dimension.
            t: Time in ``(0, 1]``.

        Returns:
            Score estimate.

        Raises:
            ValueError: If ``t`` gives a zero kernel variance.

        References:
            [EQ-PROC-DIFF-004].

        """
        variance = self.sde.kernel_variance(t)
        if variance <= 0.0:
            raise ValueError("the score network is undefined where v(t) = 0")
        std = math.sqrt(variance)
        return tuple(float(v) / std for v in self.noise_prediction(parameters, x, t))


@dataclass(frozen=True)
class TrainedScore:
    """Score network with fitted parameters; callable as ``score(x, t)``.

    Args:
        network: Architecture and forward SDE.
        parameters: Fitted flat parameters.
        optimization: Adam trace of the denoising objective.
        weighting: Name of the weighting used (``"variance"`` or
            ``"custom"``).
        training_points: Number of data points.

    References:
        [EQ-PROC-DIFF-004], [REF-VINCENT-2011].

    """

    network: ScoreNetwork
    parameters: tuple[float, ...]
    optimization: AdamResult
    weighting: str
    training_points: int

    @property
    def dimension(self) -> int:
        """Return the data dimension.

        References:
            Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

        """
        return self.network.dimension

    def __call__(self, x: Sequence[float], t: float) -> Point:
        """Return ``s_theta(x, t)``.

        Args:
            x: Point of the data dimension.
            t: Time in ``(0, 1]``.

        Returns:
            Score estimate.

        References:
            [EQ-PROC-DIFF-004].

        """
        return self.network.score(self.parameters, x, t)


def train_score_network(
    sde: ForwardSDE,
    data: Sequence[float | Sequence[float]],
    *,
    hidden: Sequence[int] = (16,),
    activation: str = "tanh",
    weighting: Callable[[float], float] | None = None,
    iterations: int = 800,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    tail_fraction: float = 0.2,
    time_floor: float = 1e-5,
    seed: int | RandomStream = 0,
) -> TrainedScore:
    """Fit a time-conditioned score network by denoising score matching.

    Each Adam iteration draws a mini-batch of data points ``x_0`` (with
    replacement), times ``t ~ U(time_floor, 1)``, and noise ``z ~ N(0, I)``,
    forms ``x_t = m(t) x_0 + sqrt(v(t)) z``, and minimizes the mean of
    ``lambda(t) ||s_theta(x_t, t) + z / sqrt(v(t))||^2`` [EQ-PROC-DIFF-004],
    whose minimizer over all functions is the marginal score of the
    perturbed data law.

    Args:
        sde: Forward SDE.
        data: Training points (numbers for one dimension).
        hidden: Hidden-layer widths.
        activation: Hidden activation.
        weighting: ``lambda(t)``; ``None`` uses ``lambda(t) = v(t)``, the
            choice ``lambda proportional to 1 / E||grad log p_0t||^2``.
        iterations: Positive number of Adam updates.
        batch_size: Positive mini-batch size.
        learning_rate: Adam step size.
        tail_fraction: Fraction of final iterates averaged by Adam.
        time_floor: Smallest training time in ``(0, 1)``.
        seed: Integer seed or stream (network initialization and batches).

    Returns:
        :class:`TrainedScore` with the loss trace.

    Raises:
        ValueError: For invalid data or settings, or a weighting that is not
            positive and finite.

    Equation:
        [EQ-PROC-DIFF-004].

    References:
        [REF-SONG-ETAL-2021, Equation 7 and Section 3.3], [REF-VINCENT-2011],
        [REF-HO-JAIN-ABBEEL-2020, Equation 14], [REF-KINGMA-BA-2015].

    """
    points = _as_points(data, name="data")
    dimension = len(points[0])
    steps = _positive_integer(iterations, name="iterations")
    batch = _positive_integer(batch_size, name="batch_size")
    floor = require_finite_real(time_floor, name="time_floor")
    if not 0.0 < floor < 1.0:
        raise ValueError("time_floor must lie in (0, 1)")
    network = ScoreNetwork(sde, dimension, tuple(hidden), activation)
    stream = resolve_stream(seed)
    x0 = network.initial_parameters(stream.child("init"))

    def objective(
        theta: tuple[float, ...], batch_stream: RandomStream
    ) -> tuple[float, Sequence[float]]:
        """Return the mini-batch denoising loss and its reverse-mode gradient.

        Args:
            theta: Current parameters.
            batch_stream: Stream for this iteration's batch.

        Returns:
            ``(loss, gradient)``.

        References:
            [EQ-PROC-DIFF-004], [EQ-AD-REVERSE-001].

        """
        generator = batch_stream.generator()
        cases: list[tuple[list[float], float, list[float], float]] = []
        for _ in range(batch):
            clean = points[generator.randrange(len(points))]
            t = floor + (1.0 - floor) * generator.random()
            noise = [generator.gauss(0.0, 1.0) for _ in range(dimension)]
            m = sde.mean_coefficient(t)
            variance = sde.kernel_variance(t)
            std = math.sqrt(variance)
            noisy = [m * c + std * z for c, z in zip(clean, noise, strict=True)]
            factor = (variance if weighting is None else weighting(t)) / variance
            if not math.isfinite(factor) or factor <= 0.0:
                raise ValueError("weighting must return positive finite values")
            cases.append((noisy, t, noise, factor))

        def loss(parameters: Sequence[Any]) -> Any:
            """Return the weighted denoising loss on the fixed batch.

            References:
                [EQ-PROC-DIFF-004].

            """
            total: Any = 0.0
            for noisy, t, noise, factor in cases:
                output = network.noise_prediction(parameters, noisy, t)
                error: Any = 0.0
                for value, z in zip(output, noise, strict=True):
                    residual = value + z
                    error = error + residual * residual
                total = total + factor * error
            return total / batch

        return reverse_value_and_gradient(loss, theta)

    result = adam_minimize(
        objective,
        x0,
        iterations=steps,
        learning_rate=learning_rate,
        tail_fraction=tail_fraction,
        seed=stream.child("adam"),
    )
    return TrainedScore(
        network,
        result.x,
        result,
        "variance" if weighting is None else "custom",
        len(points),
    )


@dataclass(frozen=True, slots=True)
class DiffusionSample:
    """Points produced by a reverse-time sampler.

    Args:
        points: Generated points.
        method: Sampler name.
        steps: Number of time steps.
        end_time: Final (smallest) diffusion time reached.
        score_evaluations: Calls of the score function.

    References:
        [REF-SONG-ETAL-2021, Section 4].

    """

    points: tuple[Point, ...]
    method: str
    steps: int
    end_time: float
    score_evaluations: int

    @property
    def means(self) -> Point:
        """Return the per-coordinate sample means.

        References:
            [EQ-MC-MEAN-001], [REF-OWEN-2013, Chapter 2].

        """
        count = len(self.points)
        return tuple(
            math.fsum(point[j] for point in self.points) / count
            for j in range(len(self.points[0]))
        )

    @property
    def variances(self) -> Point:
        """Return the per-coordinate unbiased sample variances.

        References:
            [EQ-MC-MEAN-001], [REF-OWEN-2013, Chapter 2].

        """
        count = len(self.points)
        if count < 2:
            raise ValueError("at least two points are needed for a variance")
        return tuple(
            math.fsum((point[j] - mean) ** 2 for point in self.points) / (count - 1)
            for j, mean in enumerate(self.means)
        )


def _evaluate_score(score: ScoreFunction, x: Point, t: float, dimension: int) -> Point:
    """Call the score function and validate its output.

    Args:
        score: Score function ``(x, t) -> grad log p_t(x)``.
        x: Point.
        t: Time.
        dimension: Expected output length.

    Returns:
        Score vector as floats.

    Raises:
        ValueError: If the output has the wrong length or is not finite.

    References:
        [REF-SONG-ETAL-2021, Section 3.3]. No equation is evaluated.

    """
    values = tuple(float(v) for v in score(x, t))
    if len(values) != dimension:
        raise ValueError("score output length does not match the dimension")
    if not all(math.isfinite(v) for v in values):
        raise ValueError(f"score is not finite at t = {t}")
    return values


def _sampler_settings(
    sde: ForwardSDE,
    count: int,
    dimension: int,
    steps: int,
    end_time: float | None,
) -> tuple[int, int, int, float]:
    """Validate the common sampler arguments.

    Args:
        sde: Forward SDE.
        count: Number of points.
        dimension: Data dimension.
        steps: Number of time steps.
        end_time: Final time or ``None`` for the SDE default.

    Returns:
        ``(count, dimension, steps, end_time)``.

    Raises:
        ValueError: If a value is invalid.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].
        No equation is evaluated.

    """
    final = sde.default_end_time if end_time is None else end_time
    final = require_finite_real(final, name="end_time")
    if not 0.0 < final < 1.0:
        raise ValueError("end_time must lie in (0, 1)")
    return (
        _positive_integer(count, name="count"),
        _positive_integer(dimension, name="dimension"),
        _positive_integer(steps, name="steps"),
        final,
    )


def _prior_points(
    sde: ForwardSDE, count: int, dimension: int, stream: RandomStream
) -> list[list[float]]:
    """Draw ``count`` points from the Gaussian prior ``N(0, prior_variance I)``.

    Args:
        sde: Forward SDE.
        count: Number of points.
        dimension: Data dimension.
        stream: Random stream.

    Returns:
        Mutable prior points.

    References:
        [REF-SONG-ETAL-2021, Section 3.4].

    """
    generator = stream.child("prior").generator()
    scale = math.sqrt(sde.prior_variance)
    return [
        [scale * generator.gauss(0.0, 1.0) for _ in range(dimension)]
        for _ in range(count)
    ]


def _reverse_step(
    sde: ForwardSDE,
    score: ScoreFunction,
    x: list[float],
    t: float,
    h: float,
    noise: Sequence[float],
) -> list[float]:
    """Apply one Euler-Maruyama step of the reverse-time SDE from ``t`` to ``t - h``.

    Args:
        sde: Forward SDE.
        score: Score function.
        x: Current point.
        t: Current time.
        h: Positive step.
        noise: Standard normal draws, one per coordinate.

    Returns:
        ``x - [f(x, t) - g(t)^2 s(x, t)] h + g(t) sqrt(h) z``.

    References:
        [EQ-PROC-DIFF-005], [EQ-PROC-SDE-001].

    """
    g = sde.diffusion(t)
    s = _evaluate_score(score, tuple(x), t, len(x))
    f = sde.drift(x, t)
    root = g * math.sqrt(h)
    return [
        xi - (fi - g * g * si) * h + root * zi
        for xi, fi, si, zi in zip(x, f, s, noise, strict=True)
    ]


def reverse_sde_sample(
    sde: ForwardSDE,
    score: ScoreFunction,
    count: int,
    *,
    dimension: int = 1,
    steps: int = 500,
    end_time: float | None = None,
    seed: int | RandomStream | None = None,
) -> DiffusionSample:
    """Sample by integrating the reverse-time SDE with Euler-Maruyama.

    Prior points ``x_1 ~ N(0, prior_variance I)`` are moved from ``t = 1``
    to ``t = end_time`` on a uniform grid by
    ``x <- x - [f - g^2 s] h + g sqrt(h) z`` [EQ-PROC-DIFF-005].

    Args:
        sde: Forward SDE.
        score: Score function ``(x, t) -> grad log p_t(x)``.
        count: Positive number of points.
        dimension: Data dimension.
        steps: Positive number of time steps.
        end_time: Final time in ``(0, 1)``; ``None`` uses the SDE default.
        seed: Integer seed or stream.

    Returns:
        :class:`DiffusionSample`.

    Raises:
        ValueError: For invalid settings or a non-finite score.

    Equation:
        [EQ-PROC-DIFF-005].

    References:
        [REF-ANDERSON-1982], [REF-SONG-ETAL-2021, Equation 6],
        [REF-KLOEDEN-PLATEN-1992, Section 10.2].

    """
    n, d, k, final = _sampler_settings(sde, count, dimension, steps, end_time)
    stream = resolve_stream(seed)
    points = _prior_points(sde, n, d, stream)
    generator = stream.child("reverse_sde").generator()
    h = (1.0 - final) / k
    for i in range(k):
        t = 1.0 - i * h
        for index, x in enumerate(points):
            noise = [generator.gauss(0.0, 1.0) for _ in range(d)]
            points[index] = _reverse_step(sde, score, x, t, h, noise)
    return DiffusionSample(
        tuple(tuple(x) for x in points), "reverse_sde", k, final, n * k
    )


def predictor_corrector_sample(
    sde: ForwardSDE,
    score: ScoreFunction,
    count: int,
    *,
    dimension: int = 1,
    steps: int = 500,
    corrector_steps: int = 1,
    snr: float = 0.16,
    end_time: float | None = None,
    seed: int | RandomStream | None = None,
) -> DiffusionSample:
    """Sample with a reverse-SDE predictor and a Langevin corrector.

    Each time step applies the reverse-time Euler-Maruyama predictor
    [EQ-PROC-DIFF-005] and then ``corrector_steps`` Langevin moves at the new
    time ``x <- x + eps s + sqrt(2 eps) z`` with
    ``eps = 2 alpha (r ||z|| / ||s||)^2`` [EQ-PROC-DIFF-006]. The norms are
    averaged over the batch of points, so one step size is shared by all.

    Args:
        sde: Forward SDE.
        score: Score function.
        count: Positive number of points.
        dimension: Data dimension.
        steps: Positive number of time steps.
        corrector_steps: Non-negative Langevin moves per time step.
        snr: Positive signal-to-noise ratio ``r``.
        end_time: Final time in ``(0, 1)``; ``None`` uses the SDE default.
        seed: Integer seed or stream.

    Returns:
        :class:`DiffusionSample`.

    Raises:
        ValueError: For invalid settings or a non-finite score.

    Equation:
        [EQ-PROC-DIFF-005], [EQ-PROC-DIFF-006].

    References:
        [REF-SONG-ETAL-2021, Section 4.2 and Appendix G, Algorithms 2-5].

    """
    n, d, k, final = _sampler_settings(sde, count, dimension, steps, end_time)
    if isinstance(corrector_steps, bool) or not isinstance(corrector_steps, int):
        raise ValueError("corrector_steps must be a non-negative integer")
    if corrector_steps < 0:
        raise ValueError("corrector_steps must be a non-negative integer")
    ratio = _positive(snr, name="snr")
    stream = resolve_stream(seed)
    points = _prior_points(sde, n, d, stream)
    generator = stream.child("predictor_corrector").generator()
    h = (1.0 - final) / k
    evaluations = 0
    for i in range(k):
        t = 1.0 - i * h
        for index, x in enumerate(points):
            noise = [generator.gauss(0.0, 1.0) for _ in range(d)]
            points[index] = _reverse_step(sde, score, x, t, h, noise)
        evaluations += n
        t_next = t - h
        alpha = sde.corrector_scale(t_next, h)
        for _ in range(corrector_steps):
            scores = [_evaluate_score(score, tuple(x), t_next, d) for x in points]
            noises = [[generator.gauss(0.0, 1.0) for _ in range(d)] for _ in points]
            evaluations += n
            score_norm = math.fsum(math.hypot(*s) for s in scores) / n
            noise_norm = math.fsum(math.hypot(*z) for z in noises) / n
            if score_norm == 0.0:
                continue
            eps = 2.0 * alpha * (ratio * noise_norm / score_norm) ** 2
            root = math.sqrt(2.0 * eps)
            points = [
                [xi + eps * si + root * zi for xi, si, zi in zip(x, s, z, strict=True)]
                for x, s, z in zip(points, scores, noises, strict=True)
            ]
    return DiffusionSample(
        tuple(tuple(x) for x in points), "predictor_corrector", k, final, evaluations
    )


def _flow_velocity(
    sde: ForwardSDE, score: ScoreFunction, x: Sequence[float], t: float
) -> list[float]:
    """Return the probability-flow velocity ``f(x, t) - g(t)^2 s(x, t) / 2``.

    Args:
        sde: Forward SDE.
        score: Score function.
        x: Point.
        t: Time.

    Returns:
        Velocity vector.

    References:
        [EQ-PROC-DIFF-007].

    """
    g2 = sde.diffusion(t) ** 2
    s = _evaluate_score(score, tuple(x), t, len(x))
    f = sde.drift(x, t)
    return [fi - 0.5 * g2 * si for fi, si in zip(f, s, strict=True)]


def _ode_step(
    sde: ForwardSDE,
    score: ScoreFunction,
    x: list[float],
    t: float,
    h: float,
    method: str,
) -> list[float]:
    """Advance the probability-flow ODE by a signed step ``h``.

    Args:
        sde: Forward SDE.
        score: Score function.
        x: Current point.
        t: Current time.
        h: Signed time step (negative towards the data).
        method: ``"rk4"`` (classical Runge-Kutta) or ``"heun"``.

    Returns:
        Point at time ``t + h``.

    References:
        [EQ-PROC-DIFF-007], [REF-HAIRER-NORSETT-WANNER-1993, Section II.1].

    """

    def shifted(base: Sequence[float], slope: Sequence[float], c: float) -> list[float]:
        """Return ``base + c slope``.

        References:
            [EQ-PROC-DIFF-007].

        """
        return [b + c * s for b, s in zip(base, slope, strict=True)]

    k1 = _flow_velocity(sde, score, x, t)
    if method == "heun":
        k2 = _flow_velocity(sde, score, shifted(x, k1, h), t + h)
        return [xi + 0.5 * h * (a + b) for xi, a, b in zip(x, k1, k2, strict=True)]
    k2 = _flow_velocity(sde, score, shifted(x, k1, 0.5 * h), t + 0.5 * h)
    k3 = _flow_velocity(sde, score, shifted(x, k2, 0.5 * h), t + 0.5 * h)
    k4 = _flow_velocity(sde, score, shifted(x, k3, h), t + h)
    return [
        xi + h * (a + 2.0 * b + 2.0 * c + e) / 6.0
        for xi, a, b, c, e in zip(x, k1, k2, k3, k4, strict=True)
    ]


def probability_flow_map(
    sde: ForwardSDE,
    score: ScoreFunction,
    points: Sequence[float | Sequence[float]],
    *,
    start_time: float = 1.0,
    end_time: float | None = None,
    steps: int = 100,
    method: str = "rk4",
) -> DiffusionSample:
    """Transport points along the probability-flow ODE between two times.

    The ODE ``dx/dt = f(x, t) - g(t)^2 s(x, t) / 2`` [EQ-PROC-DIFF-007] has
    the same marginals ``p_t`` as the forward SDE, so the map is a
    deterministic bijection; integrating from ``start_time = 1`` decodes
    prior points into data, and from a small time to ``1`` encodes data.

    Args:
        sde: Forward SDE.
        score: Score function.
        points: Points at ``start_time`` (numbers for one dimension).
        start_time: Initial time in ``(0, 1]``.
        end_time: Final time in ``(0, 1]``; ``None`` uses the SDE default.
        steps: Positive number of uniform steps.
        method: ``"rk4"`` or ``"heun"``.

    Returns:
        :class:`DiffusionSample` with the transported points.

    Raises:
        ValueError: For invalid points, times, steps, or method.

    Equation:
        [EQ-PROC-DIFF-007].

    References:
        [REF-SONG-ETAL-2021, Equation 13 and Section 4.3],
        [REF-HAIRER-NORSETT-WANNER-1993, Section II.1].

    """
    if method not in ODE_METHODS:
        raise ValueError(f"method must be one of {ODE_METHODS}")
    start = _unit_time(start_time, name="start_time")
    final = _unit_time(
        sde.default_end_time if end_time is None else end_time, name="end_time"
    )
    if start == 0.0 or final == 0.0:
        raise ValueError("start_time and end_time must be positive")
    if start == final:
        raise ValueError("start_time and end_time must differ")
    k = _positive_integer(steps, name="steps")
    current = [list(p) for p in _as_points(points, name="points")]
    h = (final - start) / k
    for i in range(k):
        t = start + i * h
        current = [_ode_step(sde, score, x, t, h, method) for x in current]
    per_step = 4 if method == "rk4" else 2
    return DiffusionSample(
        tuple(tuple(x) for x in current),
        f"probability_flow_{method}",
        k,
        final,
        per_step * k * len(current),
    )


def probability_flow_sample(
    sde: ForwardSDE,
    score: ScoreFunction,
    count: int,
    *,
    dimension: int = 1,
    steps: int = 100,
    method: str = "rk4",
    end_time: float | None = None,
    seed: int | RandomStream | None = None,
) -> DiffusionSample:
    """Sample by decoding prior draws with the probability-flow ODE.

    Args:
        sde: Forward SDE.
        score: Score function.
        count: Positive number of points.
        dimension: Data dimension.
        steps: Positive number of ODE steps.
        method: ``"rk4"`` or ``"heun"``.
        end_time: Final time in ``(0, 1)``; ``None`` uses the SDE default.
        seed: Integer seed or stream (the prior draws only).

    Returns:
        :class:`DiffusionSample`.

    Raises:
        ValueError: For invalid settings.

    Equation:
        [EQ-PROC-DIFF-007].

    References:
        [REF-SONG-ETAL-2021, Section 4.3].

    """
    n, d, k, final = _sampler_settings(sde, count, dimension, steps, end_time)
    prior = _prior_points(sde, n, d, resolve_stream(seed))
    return probability_flow_map(
        sde, score, prior, start_time=1.0, end_time=final, steps=k, method=method
    )
