"""Flow-matching posterior estimation for simulation-based inference.

For a simulator with an intractable likelihood, draw ``N`` pairs
``theta_n ~ p(theta)``, ``x_n ~ p(x | theta_n)`` and train a velocity
network ``v_phi(t, theta_t, x)`` on the conditional flow-matching loss with
the observation as an extra input:

``E |v_phi(t, theta_t, x) - (theta_1 - (1 - s) theta_0)|^2``,
``theta_t = (1 - (1 - s) t) theta_0 + t theta_1``,

``(theta_1, x)`` a simulated pair, ``theta_0 ~ N(0, I)``, ``t ~ U(0, 1)``
[EQ-GEN-FMPE-001]. For every fixed ``x`` the minimizer is the marginal field
that transports ``N(0, I)`` to ``p(theta | x)`` (convolved with
``N(0, s^2 I)``), because the pairs with that ``x`` have ``theta_1``
distributed as the posterior. Posterior draws for an observation are
obtained by integrating ``d theta / dt = v_phi(t, theta, x_obs)``; training
is paid once (amortized). Unlike
:func:`probstat.inference.fit_amortized_gaussian_posterior` the family is
not Gaussian, so skewed and multimodal posteriors can be represented.

An amortized neural posterior can be overconfident (Hermans et al. 2022),
and this one is at the small training sizes used in the library's tests. On
the linear-Gaussian problem ``theta ~ N(0, 1)``, ``x = 2 theta + N(0, 1)``
(1000 simulations, two hidden layers of 16, 1200 iterations) the central
intervals of the fitted posterior were too narrow for eight of ten training
seeds: their width was 0.87 to 1.02 times the exact one (mean 0.94) and the
nominal 90 % interval covered 86 % to 91 % (mean 88.7 %) under the exact
posterior. Simulation-based calibration with 2000 replicates rejects
uniform ranks at ``p < 0.01`` for two of six training seeds; with the 200
replicates of the test suite the same fits are not rejected. More training
reduces the error (4000 simulations and 5000 iterations: width ratio 0.99,
90 % coverage 89 % for two seeds) but nothing in the loss shows it. Check every
fit with :func:`probstat.inference.simulation_based_calibration` on fresh
simulations, with enough replicates to see a ten per cent error of the
width, before using it. Networks are small and trained in scalar Python;
large-scale training is out of scope.

References:
    [REF-WILDBERGER-ETAL-2023], [REF-LIPMAN-ETAL-2023],
    [REF-HERMANS-ETAL-2022], [REF-TALTS-ETAL-2018].

"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ..foundations.random_streams import RandomStream, resolve_stream
from ..foundations.validation import require_finite_real
from ._common import Point, positive_integer, positive_real
from .flow_matching import (
    VelocityNetwork,
    _sigma_min,
    flow_log_density,
    integrate_flow,
    loss_window_means,
    standardize_columns,
    train_velocity_network,
)

PriorSampler = Callable[[random.Random], Sequence[float]]
Simulator = Callable[[Sequence[float], random.Random], Sequence[float]]


def finite_vector(values: Sequence[object], name: str, width: int | None) -> Point:
    """Validate a finite, non-empty vector of an expected width.

    Args:
        values: Candidate vector.
        name: Name used in error messages.
        width: Required length, or ``None`` for any positive length.

    Returns:
        The vector as a float tuple.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If the vector is empty, has the wrong width, or holds a
            non-finite entry.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].
        No equation is evaluated.

    """
    vector = tuple(require_finite_real(v, name=name) for v in values)
    if not vector or (width is not None and len(vector) != width):
        raise ValueError(f"{name} must be a non-empty vector of a fixed width")
    return vector


def simulate_pairs(
    prior_sampler: PriorSampler,
    simulator: Simulator,
    simulations: object,
    stream: RandomStream,
) -> tuple[list[Point], list[Point]]:
    """Draw ``(theta_n, x_n)`` pairs from the prior and the simulator.

    Pair ``n`` uses the child stream ``("simulation", n)`` for both draws,
    so the training set is reproducible and independent of the order of
    evaluation.

    Args:
        prior_sampler: ``prior_sampler(rng) -> theta``.
        simulator: ``simulator(theta, rng) -> x``.
        simulations: Number ``N >= 2`` of pairs.
        stream: Random stream.

    Returns:
        ``(thetas, observations)``.

    Raises:
        TypeError: If a callable is missing or a value is not real.
        ValueError: If ``N < 2`` or the vectors change width or are not
            finite.

    References:
        [EQ-GEN-FMPE-001], [REF-WILDBERGER-ETAL-2023].

    """
    if not callable(prior_sampler) or not callable(simulator):
        raise TypeError("prior_sampler and simulator must be callable")
    count = positive_integer(simulations, name="simulations")
    if count < 2:
        raise ValueError("simulations must be at least 2")
    thetas: list[Point] = []
    observations: list[Point] = []
    for index in range(count):
        rng = stream.child("simulation", index).generator()
        theta = finite_vector(
            prior_sampler(rng), "prior draw", len(thetas[0]) if thetas else None
        )
        observation = finite_vector(
            simulator(theta, rng),
            "simulated observation",
            len(observations[0]) if observations else None,
        )
        thetas.append(theta)
        observations.append(observation)
    return thetas, observations


@dataclass(frozen=True, slots=True)
class FlowMatchingPosterior:
    """A trained flow-matching posterior ``q(theta | x)``.

    The flow acts on standardized parameters and is conditioned on
    standardized observations. Adam has no convergence test:
    ``loss_decreased`` is a necessary sanity check only, and calibration
    must be checked separately.

    Args:
        network: Velocity model with the observation as conditioning input.
        parameters: Trained flat network parameters.
        summary_location: Training means of the observations.
        summary_scale: Training standard deviations of the observations.
        parameter_location: Training means of ``theta``.
        parameter_scale: Training standard deviations of ``theta``.
        sigma_min: Path width ``s`` (standardized units).
        losses: Mini-batch loss per Adam iteration.
        initial_loss: Mean loss over the first tenth of the iterations.
        final_loss: Mean loss over the last tenth of the iterations.
        simulations: Number of simulator runs used for training.

    References:
        [EQ-GEN-FMPE-001], [REF-WILDBERGER-ETAL-2023].

    """

    network: VelocityNetwork
    parameters: tuple[float, ...]
    summary_location: Point
    summary_scale: Point
    parameter_location: Point
    parameter_scale: Point
    sigma_min: float
    losses: tuple[float, ...]
    initial_loss: float
    final_loss: float
    simulations: int

    @property
    def dimension(self) -> int:
        """Return the number ``p`` of parameters.

        References:
            [EQ-GEN-FMPE-001].

        """
        return len(self.parameter_location)

    @property
    def loss_decreased(self) -> bool:
        """Return whether the final loss window is below the initial one.

        References:
            Project reporting policy [REF-PROBSTAT-REPORTING].
            No equation is evaluated.

        """
        return self.final_loss < self.initial_loss

    def _condition(self, summary: Sequence[object]) -> Point:
        """Return the standardized observation.

        Raises:
            ValueError: If the width is wrong or an entry is not finite.

        References:
            [EQ-GEN-FMPE-001]. No new equation is introduced.

        """
        values = finite_vector(summary, "summary", len(self.summary_location))
        return tuple(
            (v - m) / s
            for v, m, s in zip(
                values, self.summary_location, self.summary_scale, strict=True
            )
        )

    def _field(
        self, condition: Point
    ) -> Callable[[float, Sequence[float]], list[float]]:
        """Return the standardized velocity field at one observation.

        References:
            [EQ-GEN-FMPE-001].

        """

        def velocity(t: float, z: Sequence[float]) -> list[float]:
            """Return ``v_phi(t, z, x)``.

            References:
                [EQ-GEN-FMPE-001].

            """
            return self.network.velocity(self.parameters, t, z, condition)

        return velocity

    def sample(
        self,
        summary: Sequence[object],
        size: int,
        *,
        steps: int = 20,
        method: str = "rk4",
        seed: int | RandomStream | None = None,
    ) -> tuple[Point, ...]:
        """Draw posterior samples at one observation.

        Each draw integrates ``d theta / dt = v_phi(t, theta, x)`` from a
        standard-normal point with ``steps`` fixed steps.

        Args:
            summary: Observed vector of the training width.
            size: Positive number of draws.
            steps: Positive number of ODE steps per draw.
            method: ``"euler"``, ``"heun"``, or ``"rk4"``.
            seed: Integer seed or stream; ``None`` is not reproducible.

        Returns:
            ``size`` parameter vectors.

        Raises:
            TypeError: If an entry of ``summary`` is not real.
            ValueError: If ``size``, a setting, or the summary is invalid.

        References:
            [EQ-GEN-FMPE-001], [EQ-NUM-ODE-001], [REF-WILDBERGER-ETAL-2023].

        """
        count = positive_integer(size, name="size")
        velocity = self._field(self._condition(summary))
        generator = resolve_stream(seed).child("fmpe-sample").generator()
        draws = []
        for _ in range(count):
            base = [generator.gauss(0.0, 1.0) for _ in range(self.dimension)]
            end = integrate_flow(velocity, base, steps=steps, method=method)
            draws.append(
                tuple(
                    m + s * z
                    for m, s, z in zip(
                        self.parameter_location, self.parameter_scale, end, strict=True
                    )
                )
            )
        return tuple(draws)

    def log_density(
        self,
        theta: Sequence[object],
        summary: Sequence[object],
        *,
        steps: int = 20,
        method: str = "rk4",
    ) -> float:
        """Return ``log q(theta | x)`` by the instantaneous change of variables.

        The divergence of the network is exact (forward-mode
        differentiation); the time integrals use ``steps`` fixed steps.

        Args:
            theta: Parameter vector of width ``p``.
            summary: Observed vector.
            steps: Positive number of ODE steps.
            method: ``"euler"``, ``"heun"``, or ``"rk4"``.

        Returns:
            The log-density on the original parameter scale.

        Raises:
            TypeError: If an entry is not real.
            ValueError: If a width is wrong, an entry is not finite, or a
                setting is invalid.

        References:
            [EQ-GEN-FM-004], [EQ-GEN-FMPE-001], [REF-WILDBERGER-ETAL-2023].

        """
        values = finite_vector(theta, "theta", self.dimension)
        condition = self._condition(summary)
        z = [
            (v - m) / s
            for v, m, s in zip(
                values, self.parameter_location, self.parameter_scale, strict=True
            )
        ]

        def trace(t: float, position: Sequence[float]) -> float:
            """Return the exact divergence of the conditional field.

            References:
                [EQ-GEN-FM-004].

            """
            return self.network.divergence(self.parameters, t, position, condition)

        value = flow_log_density(
            self._field(condition), z, steps=steps, method=method, divergence=trace
        )
        return value - math.fsum(math.log(s) for s in self.parameter_scale)


def fit_flow_matching_posterior(
    prior_sampler: PriorSampler,
    simulator: Simulator,
    simulations: int,
    *,
    hidden_layers: Sequence[int] = (32, 32),
    sigma_min: float = 0.0,
    iterations: int = 3000,
    batch_size: int = 64,
    learning_rate: float = 0.005,
    tail_fraction: float = 0.2,
    seed: int | RandomStream | None = None,
) -> FlowMatchingPosterior:
    """Train a flow-matching posterior on simulated parameter/data pairs.

    Parameters and observations are standardized with their training
    moments; Adam minimizes the mini-batch conditional flow-matching loss
    with gradients by backpropagation.

    Args:
        prior_sampler: ``prior_sampler(rng) -> theta`` draws from the prior.
        simulator: ``simulator(theta, rng) -> x`` returns the observation
            (or summary statistics).
        simulations: Number ``N >= 2`` of training pairs.
        hidden_layers: Hidden widths of the ``tanh`` velocity network.
        sigma_min: Path width ``s`` in ``[0, 1)`` (standardized units).
        iterations: Positive number of Adam iterations.
        batch_size: Positive mini-batch size (capped at ``N``).
        learning_rate: Positive Adam step size.
        tail_fraction: Fraction in ``[0, 1)`` of final iterates averaged.
        seed: Integer seed or stream; ``None`` is not reproducible.

    Returns:
        The trained posterior with its loss trace and simulation count.

    Raises:
        TypeError: If a callable is missing or a value is not real.
        ValueError: If a setting is invalid or the simulated vectors change
            width or are not finite.

    References:
        [EQ-GEN-FMPE-001], [EQ-GEN-FM-001], [EQ-OPT-ADAM-002],
        [REF-WILDBERGER-ETAL-2023], [REF-LIPMAN-ETAL-2023].

    """
    steps = positive_integer(iterations, name="iterations")
    batch = positive_integer(batch_size, name="batch_size")
    rate = positive_real(learning_rate, name="learning_rate")
    width = _sigma_min(sigma_min)
    stream = resolve_stream(seed)
    thetas, observations = simulate_pairs(prior_sampler, simulator, simulations, stream)
    theta_location, theta_scale, targets = standardize_columns(thetas)
    summary_location, summary_scale, conditions = standardize_columns(observations)
    network = VelocityNetwork(
        len(theta_location), len(summary_location), tuple(hidden_layers)
    )
    result = train_velocity_network(
        network,
        targets,
        conditions,
        sigma_min=width,
        iterations=steps,
        batch_size=batch,
        learning_rate=rate,
        tail_fraction=tail_fraction,
        stream=stream.child("fmpe"),
    )
    initial_loss, final_loss = loss_window_means(result.losses)
    return FlowMatchingPosterior(
        network=network,
        parameters=result.x,
        summary_location=summary_location,
        summary_scale=summary_scale,
        parameter_location=theta_location,
        parameter_scale=theta_scale,
        sigma_min=width,
        losses=result.losses,
        initial_loss=initial_loss,
        final_loss=final_loss,
        simulations=len(thetas),
    )
