"""Energy-based models trained by score matching.

An energy model defines the unnormalized density ``p(x) ∝ exp(-E(x))``
([EQ-GEN-EBM-001]). Here the energy is a positive-definite quadratic around a
learned center plus an optional :class:`~probstat.numerics.MLP` correction,

``E(x) = |L'(x - mu)|^2 / 2 + N_theta(x)``,

with ``L`` lower triangular with a positive (log-parameterized) diagonal, so
the quadratic part dominates the bounded-slope network and the density is
normalizable. Without a network the model is a Gaussian with precision
``L L'``.

Training never needs the normalizing constant:

* :func:`fit_score_matching` minimizes the implicit score-matching
  objective of Hyvärinen (2005), ``mean(|grad E|^2 / 2 - Laplacian E)``
  ([EQ-GEN-EBM-002]). Its parameter gradient involves third derivatives of
  the energy, so the input gradient and Laplacian are propagated forward
  through the network on tape variables of the parameters
  ([EQ-GEN-EBM-005]) and one reverse sweep differentiates the objective.
  :func:`score_matching_objective` evaluates the objective with exact
  forward-over-reverse Hessians as an independent check.
* :func:`fit_denoising_score_matching` minimizes the denoising objective of
  Vincent (2011) with Gaussian corruption, which needs only first input
  derivatives ([EQ-GEN-EBM-004]).

References:
    [REF-HYVARINEN-2005, Theorem 1 and Section 3.1],
    [REF-VINCENT-2011, Section 4.2], [REF-GRIEWANK-WALTHER-2008, Chapter 13].

"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..foundations.random_streams import RandomStream
from ..numerics import (
    MLP,
    admath,
    reverse_gradient,
    reverse_value_and_gradient,
    reverse_value_gradient_and_hessian,
)
from ..optimization import adam_minimize
from ._common import (
    Point,
    as_point,
    as_points,
    positive_integer,
    positive_real,
    resolve,
)
from ._jet import LaplacianJet
from .langevin import LangevinResult, sample_langevin

_TWICE_DIFFERENTIABLE = ("tanh", "sigmoid", "softplus")


@dataclass(frozen=True)
class EnergyModel:
    """Energy ``E(x) = |L'(x - mu)|^2 / 2 + N_theta(x)`` on ``R^d``.

    Parameter layout: ``mu`` (``d``), the lower triangle of ``L`` row by row
    with each diagonal entry stored as its logarithm (``d (d + 1) / 2``), and
    the network parameters.

    Attributes:
        dimension: Number of coordinates.
        hidden: Hidden widths of the network ``N_theta``, or ``None`` for a
            purely quadratic (Gaussian) energy.
        activation: Hidden activation; ``tanh``, ``sigmoid``, or
            ``softplus`` (score matching needs two input derivatives).

    References:
        [EQ-GEN-EBM-001], [EQ-NN-MLP-001], [REF-HYVARINEN-2005, Section 2].

    """

    dimension: int
    hidden: tuple[int, ...] | None = (16,)
    activation: str = "tanh"
    network: MLP | None = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Validate the settings and build the network.

        Raises:
            ValueError: For a non-positive dimension, an empty hidden tuple,
                or an activation without a second derivative.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].
            No equation is evaluated.

        """
        positive_integer(self.dimension, name="dimension")
        if self.activation not in _TWICE_DIFFERENTIABLE:
            raise ValueError(f"activation must be one of {_TWICE_DIFFERENTIABLE}")
        network: MLP | None = None
        if self.hidden is not None:
            hidden = tuple(self.hidden)
            if not hidden:
                raise ValueError("hidden must be None or a non-empty tuple of widths")
            network = MLP((self.dimension, *hidden, 1), activation=self.activation)
            object.__setattr__(self, "hidden", hidden)
        object.__setattr__(self, "network", network)

    @property
    def quadratic_parameter_count(self) -> int:
        """Return ``d + d (d + 1) / 2``, the size of ``(mu, L)``.

        References:
            [EQ-GEN-EBM-001]. Counting only.

        """
        d = self.dimension
        return d + d * (d + 1) // 2

    @property
    def parameter_count(self) -> int:
        """Return the total number of parameters.

        References:
            [EQ-GEN-EBM-001]. Counting only.

        """
        extra = 0 if self.network is None else self.network.parameter_count
        return self.quadratic_parameter_count + extra

    def initial_parameters(
        self, seed: int | RandomStream = 0, data: object | None = None
    ) -> tuple[float, ...]:
        """Return starting parameters; the network output layer starts at zero.

        With ``data``, ``mu`` is the sample mean and ``L`` is diagonal with
        entries ``1 / sd``; otherwise ``mu = 0`` and ``L = I``.

        Args:
            seed: Integer seed or :class:`RandomStream`.
            data: Optional points used to center and scale the quadratic.

        Returns:
            Flat parameter vector.

        Raises:
            ValueError: For invalid data or a coordinate with zero spread.

        References:
            [EQ-GEN-EBM-001], [EQ-NN-GLOROT-001].

        """
        d = self.dimension
        center = [0.0] * d
        log_scales = [0.0] * d
        if data is not None:
            points = as_points(data, minimum=2)
            if len(points[0]) != d:
                raise ValueError(f"data must have dimension {d}")
            n = len(points)
            for i in range(d):
                column = [p[i] for p in points]
                center[i] = math.fsum(column) / n
                spread = math.sqrt(math.fsum((v - center[i]) ** 2 for v in column) / n)
                if spread <= 0.0:
                    raise ValueError("a coordinate has zero spread")
                log_scales[i] = -math.log(spread)
        triangle: list[float] = []
        for row in range(d):
            for entry in range(row + 1):
                triangle.append(log_scales[row] if row == entry else 0.0)
        network_parameters: tuple[float, ...] = ()
        if self.network is not None:
            network_parameters = self.network.initial_parameters(
                resolve(seed).child("energy-network"), output_scale=0.0
            )
        return (*center, *triangle, *network_parameters)

    def _check(self, parameters: Sequence[Any]) -> None:
        """Raise unless ``parameters`` has the model's length.

        Raises:
            ValueError: On a length mismatch.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].
            No equation is evaluated.

        """
        if len(parameters) != self.parameter_count:
            raise ValueError(
                f"EnergyModel expects {self.parameter_count} parameters, "
                f"got {len(parameters)}"
            )

    def _cholesky_factor(self, parameters: Sequence[Any]) -> list[list[Any]]:
        """Return ``L`` (lower triangular, positive diagonal) of any numeric kind.

        References:
            [EQ-GEN-EBM-001].

        """
        d = self.dimension
        factor: list[list[Any]] = [[0.0] * d for _ in range(d)]
        offset = d
        for row in range(d):
            for column in range(row + 1):
                value = parameters[offset]
                factor[row][column] = admath.exp(value) if row == column else value
                offset += 1
        return factor

    def energy(self, parameters: Sequence[Any], x: Sequence[Any]) -> Any:
        """Return ``E(x)`` for floats, tape variables, or Laplacian jets.

        Args:
            parameters: Flat parameter vector.
            x: Point of the model dimension.

        Returns:
            The energy, of the numeric kind of the inputs.

        Raises:
            ValueError: On a length mismatch.

        References:
            [EQ-GEN-EBM-001], [EQ-NN-MLP-001].

        """
        self._check(parameters)
        d = self.dimension
        if len(x) != d:
            raise ValueError(f"x must have dimension {d}")
        factor = self._cholesky_factor(parameters)
        difference = [x[i] - parameters[i] for i in range(d)]
        total: Any = 0.0
        for column in range(d):
            projection: Any = 0.0
            for row in range(column, d):
                projection = projection + factor[row][column] * difference[row]
            total = total + projection * projection
        total = 0.5 * total
        if self.network is not None:
            network_parameters = parameters[self.quadratic_parameter_count :]
            total = total + self.network.forward(network_parameters, list(x))[0]
        return total

    def precision(self, parameters: Sequence[float]) -> tuple[tuple[float, ...], ...]:
        """Return the precision ``L L'`` of the quadratic part.

        Raises:
            ValueError: On a length mismatch.

        References:
            [EQ-GEN-EBM-001], [EQ-GEN-EBM-003].

        """
        self._check(parameters)
        factor = [[float(v) for v in row] for row in self._cholesky_factor(parameters)]
        d = self.dimension
        return tuple(
            tuple(
                math.fsum(factor[i][k] * factor[j][k] for k in range(d))
                for j in range(d)
            )
            for i in range(d)
        )

    def center(self, parameters: Sequence[float]) -> Point:
        """Return the center ``mu`` of the quadratic part.

        Raises:
            ValueError: On a length mismatch.

        References:
            [EQ-GEN-EBM-001]. No equation is evaluated.

        """
        self._check(parameters)
        return tuple(float(v) for v in parameters[: self.dimension])

    def energy_gradient(self, parameters: Sequence[float], x: object) -> Point:
        """Return ``grad_x E(x)`` by reverse mode.

        Raises:
            ValueError: For an invalid point or parameter vector.

        References:
            [EQ-GEN-EBM-001], [EQ-AD-REVERSE-001].

        """
        point = as_point(x, self.dimension)
        values = tuple(float(v) for v in parameters)
        return reverse_gradient(lambda y: self.energy(values, y), point)

    def score(self, parameters: Sequence[float], x: object) -> Point:
        """Return the model score ``grad_x log p(x) = -grad_x E(x)``.

        Raises:
            ValueError: For an invalid point or parameter vector.

        References:
            [EQ-GEN-EBM-002], [REF-HYVARINEN-2005, Equation 1].

        """
        return tuple(-value for value in self.energy_gradient(parameters, x))

    def energy_laplacian(self, parameters: Sequence[float], x: object) -> float:
        """Return ``Laplacian_x E(x)`` from the exact forward-over-reverse Hessian.

        Raises:
            ValueError: For an invalid point or parameter vector.

        References:
            [EQ-AD-FOR-001], [EQ-GEN-EBM-002].

        """
        point = as_point(x, self.dimension)
        values = tuple(float(v) for v in parameters)
        _, _, hessian = reverse_value_gradient_and_hessian(
            lambda y: self.energy(values, y), point
        )
        return math.fsum(hessian[i][i] for i in range(self.dimension))

    def quadratic_terms(
        self, parameters: Sequence[Any]
    ) -> tuple[list[Any], list[list[Any]], Any]:
        """Return ``(mu, A, trace A)`` with ``A = L L'``, of any numeric kind.

        Training loops compute these once per mini-batch and pass them to
        :meth:`jet`.

        Raises:
            ValueError: On a length mismatch.

        References:
            [EQ-GEN-EBM-001].

        """
        self._check(parameters)
        d = self.dimension
        factor = self._cholesky_factor(parameters)
        precision: list[list[Any]] = [[0.0] * d for _ in range(d)]
        for i in range(d):
            for j in range(i + 1):
                total: Any = 0.0
                for k in range(j + 1):
                    total = total + factor[i][k] * factor[j][k]
                precision[i][j] = total
                precision[j][i] = total
        trace: Any = 0.0
        for i in range(d):
            trace = trace + precision[i][i]
        return list(parameters[:d]), precision, trace

    def jet(
        self,
        parameters: Sequence[Any],
        x: Sequence[float],
        *,
        laplacian: bool = True,
        quadratic: tuple[list[Any], list[list[Any]], Any] | None = None,
    ) -> LaplacianJet:
        """Return the forward jet ``(E, grad_x E, Laplacian_x E)`` at ``x``.

        The quadratic part contributes ``v'Av / 2``, ``Av`` and ``trace A``
        with ``v = x - mu`` in closed form; the network part is evaluated on
        :class:`LaplacianJet` seeds ([EQ-GEN-EBM-005]). The components have
        the numeric kind of ``parameters`` (floats or reverse-mode tape
        variables).

        Args:
            parameters: Flat parameter vector.
            x: Point of the model dimension.
            laplacian: Whether to propagate the Laplacian.
            quadratic: Optional precomputed :meth:`quadratic_terms`.

        Returns:
            The jet of the energy.

        Raises:
            ValueError: On a length mismatch.

        References:
            [EQ-GEN-EBM-005], [REF-GRIEWANK-WALTHER-2008, Chapter 13].

        """
        d = self.dimension
        if len(x) != d:
            raise ValueError(f"x must have dimension {d}")
        center, precision, trace = (
            self.quadratic_terms(parameters) if quadratic is None else quadratic
        )
        difference = [x[i] - center[i] for i in range(d)]
        gradient: list[Any] = []
        for i in range(d):
            total: Any = 0.0
            for j in range(d):
                total = total + precision[i][j] * difference[j]
            gradient.append(total)
        value: Any = 0.0
        for i in range(d):
            value = value + difference[i] * gradient[i]
        value = 0.5 * value
        second: Any = trace if laplacian else None
        if self.network is not None:
            seeds = LaplacianJet.seeds(x, laplacian=laplacian)
            network_parameters = parameters[self.quadratic_parameter_count :]
            output = self.network.forward(network_parameters, list(seeds))[0]
            if isinstance(output, LaplacianJet):
                value = value + output.value
                gradient = [
                    g + h for g, h in zip(gradient, output.gradient, strict=True)
                ]
                if laplacian:
                    second = second + output.laplacian
            else:
                value = value + output
        return LaplacianJet(value, tuple(gradient), second)


def _implicit_term(
    model: EnergyModel,
    parameters: Sequence[Any],
    x: Point,
    quadratic: tuple[list[Any], list[list[Any]], Any] | None = None,
) -> Any:
    """Return ``|grad E|^2 / 2 - Laplacian E`` at one point (jet evaluation).

    References:
        [EQ-GEN-EBM-002], [EQ-GEN-EBM-005].

    """
    jet = model.jet(parameters, x, quadratic=quadratic)
    total: Any = 0.0
    for entry in jet.gradient:
        total = total + entry * entry
    return 0.5 * total - jet.laplacian


def score_matching_objective(
    model: EnergyModel, parameters: Sequence[float], data: object
) -> float:
    """Return the implicit score-matching objective on a data set.

    ``J = mean_t [ |grad E(x_t)|^2 / 2 - Laplacian E(x_t) ]``, the sample
    version of Hyvärinen's Theorem 1 with ``psi = -grad E``
    ([EQ-GEN-EBM-002]); the Laplacian is the trace of the exact
    forward-over-reverse Hessian.

    Args:
        model: The energy model.
        parameters: Flat float parameters.
        data: Points of the model dimension.

    Returns:
        The objective value.

    Raises:
        ValueError: For invalid data or parameters.

    References:
        [EQ-GEN-EBM-002], [EQ-AD-FOR-001], [REF-HYVARINEN-2005, Equation 4].

    """
    points = as_points(data)
    if len(points[0]) != model.dimension:
        raise ValueError(f"data must have dimension {model.dimension}")
    values = tuple(float(v) for v in parameters)
    terms: list[float] = []
    for point in points:
        _, gradient, hessian = reverse_value_gradient_and_hessian(
            lambda y: model.energy(values, y), point
        )
        trace = math.fsum(hessian[i][i] for i in range(model.dimension))
        terms.append(0.5 * math.fsum(g * g for g in gradient) - trace)
    return math.fsum(terms) / len(terms)


def _denoising_term(
    model: EnergyModel,
    parameters: Sequence[Any],
    x: Point,
    noise: Sequence[float],
    noise_scale: float,
    quadratic: tuple[list[Any], list[list[Any]], Any] | None = None,
) -> Any:
    """Return ``|-grad E(x + s xi) + xi / s|^2 / 2`` for one corrupted point.

    With ``x_tilde = x + s xi`` the score of the Gaussian corruption kernel
    is ``(x - x_tilde) / s^2 = -xi / s``.

    References:
        [EQ-GEN-EBM-004], [REF-VINCENT-2011, Equations 9-10].

    """
    corrupted = [xi + noise_scale * e for xi, e in zip(x, noise, strict=True)]
    jet = model.jet(parameters, corrupted, laplacian=False, quadratic=quadratic)
    total: Any = 0.0
    for entry, e in zip(jet.gradient, noise, strict=True):
        residual = -entry + e / noise_scale
        total = total + residual * residual
    return 0.5 * total


def denoising_score_matching_objective(
    model: EnergyModel,
    parameters: Sequence[float],
    data: object,
    noise_scale: float,
    *,
    seed: int | RandomStream = 0,
) -> float:
    """Return a one-corruption-per-point estimate of the denoising objective.

    Args:
        model: The energy model.
        parameters: Flat float parameters.
        data: Points of the model dimension.
        noise_scale: Positive standard deviation ``s`` of the corruption.
        seed: Integer seed or :class:`RandomStream` of the corruption noise.

    Returns:
        ``mean_t |-grad E(x_t + s xi_t) + xi_t / s|^2 / 2``.

    Raises:
        ValueError: For invalid data, parameters, or noise scale.

    References:
        [EQ-GEN-EBM-004], [REF-VINCENT-2011, Section 4.2].

    """
    points = as_points(data)
    if len(points[0]) != model.dimension:
        raise ValueError(f"data must have dimension {model.dimension}")
    scale = positive_real(noise_scale, name="noise_scale")
    values = tuple(float(v) for v in parameters)
    generator = resolve(seed).child("dsm-noise").generator()
    terms = []
    for point in points:
        noise = [generator.gauss(0.0, 1.0) for _ in point]
        terms.append(float(_denoising_term(model, values, point, noise, scale)))
    return math.fsum(terms) / len(terms)


@dataclass(frozen=True)
class EnergyFit:
    """Fitted energy model.

    Args:
        model: The energy model.
        parameters: Fitted flat parameters.
        losses: Mini-batch objective at every iteration.
        method: ``"score-matching"`` or ``"denoising-score-matching"``.
        objective: Full-data objective at the fitted parameters (the
            implicit objective by exact Hessians, or the denoising estimate
            with one corruption per point).
        noise_scale: Corruption scale for denoising, else ``None``.
        iterations: Adam iterations.
        batch_size: Mini-batch size.

    References:
        [EQ-GEN-EBM-002], [EQ-GEN-EBM-004].

    """

    model: EnergyModel
    parameters: tuple[float, ...]
    losses: tuple[float, ...]
    method: str
    objective: float
    noise_scale: float | None
    iterations: int
    batch_size: int

    def energy(self, x: object) -> float:
        """Return the fitted energy at one point.

        Raises:
            ValueError: If ``x`` is not a finite point of the model dimension.

        References:
            [EQ-GEN-EBM-001].

        """
        point = as_point(x, self.model.dimension)
        return float(self.model.energy(self.parameters, point))

    def log_unnormalized_density(self, x: object) -> float:
        """Return ``-E(x)``, the log-density up to its normalizing constant.

        Raises:
            ValueError: If ``x`` is not a finite point of the model dimension.

        References:
            [EQ-GEN-EBM-001].

        """
        return -self.energy(x)

    def score(self, x: object) -> Point:
        """Return the fitted score ``-grad E(x)``.

        Raises:
            ValueError: If ``x`` is not a finite point of the model dimension.

        References:
            [EQ-GEN-EBM-002].

        """
        return self.model.score(self.parameters, x)

    def sample_langevin(
        self,
        initial: object,
        *,
        n_samples: int,
        step_size: float,
        adjusted: bool = True,
        burn_in: int = 0,
        thin: int = 1,
        seed: int | RandomStream = 0,
    ) -> LangevinResult:
        """Sample the fitted density ``∝ exp(-E)`` by (adjusted) Langevin steps.

        Args:
            initial: Starting point.
            n_samples: Positive number of retained states.
            step_size: Positive step ``h``.
            adjusted: Apply the Metropolis-Hastings correction (MALA).
            burn_in: Non-negative number of discarded iterations.
            thin: Positive thinning interval.
            seed: Integer seed or :class:`RandomStream`.

        Returns:
            :class:`~probstat.generative.LangevinResult`.

        Raises:
            ValueError: For invalid settings.

        References:
            [EQ-GEN-LANG-001], [EQ-GEN-LANG-002], [REF-ROBERTS-TWEEDIE-1996].

        """
        parameters = self.parameters
        model = self.model

        def log_density(x: Sequence[Any]) -> Any:
            """Return ``-E(x)`` at the fitted parameters.

            References:
                [EQ-GEN-EBM-001].

            """
            return -model.energy(parameters, x)

        return sample_langevin(
            log_density,
            initial,
            n_samples=n_samples,
            step_size=step_size,
            adjusted=adjusted,
            burn_in=burn_in,
            thin=thin,
            seed=seed,
        )


def _prepare(
    data: object,
    model: EnergyModel | None,
    iterations: int,
    batch_size: int,
    initial_parameters: Sequence[float] | None,
    stream: RandomStream,
) -> tuple[list[Point], EnergyModel, int, int, tuple[float, ...]]:
    """Validate the common fitting inputs and choose starting parameters.

    Raises:
        ValueError: For invalid data or settings.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION], [EQ-GEN-EBM-001].

    """
    points = as_points(data, minimum=2)
    dimension = len(points[0])
    if model is None:
        model = EnergyModel(dimension)
    if model.dimension != dimension:
        raise ValueError(f"data dimension {dimension} does not match the model")
    steps = positive_integer(iterations, name="iterations")
    size = min(positive_integer(batch_size, name="batch_size"), len(points))
    if initial_parameters is None:
        start = model.initial_parameters(stream.child("initial"), data=points)
    else:
        start = tuple(float(v) for v in initial_parameters)
        model._check(start)
    return points, model, steps, size, start


def fit_score_matching(
    data: object,
    model: EnergyModel | None = None,
    *,
    iterations: int = 400,
    batch_size: int = 64,
    learning_rate: float = 0.01,
    initial_parameters: Sequence[float] | None = None,
    tail_fraction: float = 0.0,
    seed: int | RandomStream = 0,
) -> EnergyFit:
    """Fit an energy model by implicit score matching with Adam mini-batches.

    The mini-batch loss is ``mean |grad E|^2 / 2 - Laplacian E``
    ([EQ-GEN-EBM-002]) with the input derivatives propagated forward on
    parameter tape variables ([EQ-GEN-EBM-005]); one reverse sweep gives its
    exact parameter gradient. The cost per point is about ``d + 2`` network
    evaluations, so the method targets low dimension.

    Args:
        data: Training points (at least two).
        model: Energy model; ``None`` selects ``EnergyModel(d)``.
        iterations: Positive number of Adam steps.
        batch_size: Positive mini-batch size (capped at the data size).
        learning_rate: Adam step size.
        initial_parameters: Optional starting parameters (default: the
            data-scaled :meth:`EnergyModel.initial_parameters`).
        tail_fraction: Fraction of final iterates to average, in ``[0, 1)``.
        seed: Integer seed or :class:`RandomStream`.

    Returns:
        :class:`EnergyFit` with the loss trace and the full-data objective.

    Raises:
        ValueError: For invalid data or settings, or a non-finite loss.

    References:
        [EQ-GEN-EBM-002], [EQ-OPT-ADAM-002], [REF-HYVARINEN-2005, Theorem 1].

    """
    stream = resolve(seed)
    points, energy_model, steps, size, start = _prepare(
        data, model, iterations, batch_size, initial_parameters, stream
    )

    def objective(
        parameters: tuple[float, ...], batch_stream: RandomStream
    ) -> tuple[float, tuple[float, ...]]:
        """Return the mini-batch implicit objective and its gradient.

        References:
            [EQ-GEN-EBM-002], [EQ-GEN-EBM-005].

        """
        generator = batch_stream.child("batch").generator()
        batch = [points[i] for i in generator.sample(range(len(points)), size)]

        def loss(theta: Sequence[Any]) -> Any:
            """Return the mean implicit score-matching term over the batch.

            References:
                [EQ-GEN-EBM-002].

            """
            quadratic = energy_model.quadratic_terms(theta)
            total: Any = 0.0
            for point in batch:
                total = total + _implicit_term(energy_model, theta, point, quadratic)
            return total / len(batch)

        return reverse_value_and_gradient(loss, parameters)

    result = adam_minimize(
        objective,
        start,
        iterations=steps,
        learning_rate=learning_rate,
        tail_fraction=tail_fraction,
        seed=stream.child("adam"),
    )
    return EnergyFit(
        model=energy_model,
        parameters=result.x,
        losses=result.losses,
        method="score-matching",
        objective=score_matching_objective(energy_model, result.x, points),
        noise_scale=None,
        iterations=steps,
        batch_size=size,
    )


def fit_denoising_score_matching(
    data: object,
    model: EnergyModel | None = None,
    *,
    noise_scale: float = 0.1,
    iterations: int = 400,
    batch_size: int = 64,
    learning_rate: float = 0.01,
    initial_parameters: Sequence[float] | None = None,
    tail_fraction: float = 0.0,
    seed: int | RandomStream = 0,
) -> EnergyFit:
    """Fit an energy model by denoising score matching with Gaussian noise.

    Each iteration corrupts a mini-batch, ``x_tilde = x + s xi`` with fresh
    ``xi ~ N(0, I)``, and takes an Adam step on
    ``mean |-grad E(x_tilde) - (x - x_tilde) / s^2|^2 / 2``
    ([EQ-GEN-EBM-004]). The minimizer matches the score of the data
    smoothed by ``N(0, s^2 I)``, not of the data themselves; for a Gaussian
    energy the precision tends to ``(S + s^2 I)^{-1}``.

    Args:
        data: Training points (at least two).
        model: Energy model; ``None`` selects ``EnergyModel(d)``.
        noise_scale: Positive corruption standard deviation ``s``.
        iterations: Positive number of Adam steps.
        batch_size: Positive mini-batch size (capped at the data size).
        learning_rate: Adam step size.
        initial_parameters: Optional starting parameters.
        tail_fraction: Fraction of final iterates to average, in ``[0, 1)``.
        seed: Integer seed or :class:`RandomStream`.

    Returns:
        :class:`EnergyFit` with the loss trace and a full-data objective
        estimate.

    Raises:
        ValueError: For invalid data or settings, or a non-finite loss.

    References:
        [EQ-GEN-EBM-004], [EQ-OPT-ADAM-002], [REF-VINCENT-2011, Section 4.2].

    """
    scale = positive_real(noise_scale, name="noise_scale")
    stream = resolve(seed)
    points, energy_model, steps, size, start = _prepare(
        data, model, iterations, batch_size, initial_parameters, stream
    )
    dimension = energy_model.dimension

    def objective(
        parameters: tuple[float, ...], batch_stream: RandomStream
    ) -> tuple[float, tuple[float, ...]]:
        """Return the mini-batch denoising objective and its gradient.

        References:
            [EQ-GEN-EBM-004].

        """
        generator = batch_stream.child("batch").generator()
        batch = [points[i] for i in generator.sample(range(len(points)), size)]
        noises = [[generator.gauss(0.0, 1.0) for _ in range(dimension)] for _ in batch]

        def loss(theta: Sequence[Any]) -> Any:
            """Return the mean denoising term over the corrupted batch.

            References:
                [EQ-GEN-EBM-004].

            """
            quadratic = energy_model.quadratic_terms(theta)
            total: Any = 0.0
            for point, noise in zip(batch, noises, strict=True):
                total = total + _denoising_term(
                    energy_model, theta, point, noise, scale, quadratic
                )
            return total / len(batch)

        return reverse_value_and_gradient(loss, parameters)

    result = adam_minimize(
        objective,
        start,
        iterations=steps,
        learning_rate=learning_rate,
        tail_fraction=tail_fraction,
        seed=stream.child("adam"),
    )
    return EnergyFit(
        model=energy_model,
        parameters=result.x,
        losses=result.losses,
        method="denoising-score-matching",
        objective=denoising_score_matching_objective(
            energy_model, result.x, points, scale, seed=stream.child("objective")
        ),
        noise_scale=scale,
        iterations=steps,
        batch_size=size,
    )
