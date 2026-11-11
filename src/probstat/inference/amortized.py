"""Amortized simulation-based inference with a Gaussian neural posterior.

For a simulator with an intractable likelihood, draw ``N`` pairs
``theta_n ~ p(theta)``, ``x_n ~ p(x | theta_n)`` (``x`` a vector of summary
statistics) and fit a conditional density ``q_phi(theta | x)`` by maximum
likelihood, ``max_phi sum_n log q_phi(theta_n | x_n)``. As ``N`` grows the
maximizer is the exact posterior ``p(theta | x)`` for every ``x``
(Papamakarios and Murray 2016, Proposition 1, with the prior as proposal),
provided the family contains it. Training is paid once; afterwards the
posterior for any new observation costs one network evaluation, which is
what *amortized* means.

Here ``q_phi`` is a diagonal Gaussian whose means and log standard
deviations are the outputs of a :class:`~probstat.numerics.MLP` applied to
the standardized summaries (a one-component mixture density network). It is
exact when the posterior is Gaussian with independent coordinates and
otherwise a moment-matching approximation; check calibration on fresh
simulations (for example with
:func:`probstat.inference.simulation_based_calibration`) before trusting it.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)
from probstat.numerics import MLP, admath, reverse_value_and_gradient
from probstat.optimization import adam_minimize

PriorSampler = Callable[[random.Random], Sequence[float]]
Simulator = Callable[[Sequence[float], random.Random], Sequence[float]]
_LOG_SQRT_TWO_PI = 0.5 * math.log(2.0 * math.pi)


@dataclass(frozen=True, slots=True)
class AmortizedGaussianPosterior:
    """A trained Gaussian neural posterior ``q(theta | x)``.

    Args:
        network: Network mapping standardized summaries to ``2 p`` outputs
            (standardized means, then log standard deviations).
        parameters: Trained flat network parameters.
        summary_location: Training means of the summaries.
        summary_scale: Training standard deviations of the summaries.
        parameter_location: Training means of ``theta``.
        parameter_scale: Training standard deviations of ``theta``.
        losses: Mini-batch negative log-likelihood per Adam iteration.
        simulations: Number of simulated training pairs.

    References:
        [EQ-INF-NPE-001], [REF-PAPAMAKARIOS-MURRAY-2016].

    """

    network: MLP
    parameters: tuple[float, ...]
    summary_location: tuple[float, ...]
    summary_scale: tuple[float, ...]
    parameter_location: tuple[float, ...]
    parameter_scale: tuple[float, ...]
    losses: tuple[float, ...]
    simulations: int

    @property
    def dimension(self) -> int:
        """Return the number ``p`` of parameters.

        References:
            [EQ-INF-NPE-001].

        """
        return len(self.parameter_location)

    def moments(
        self, summary: Sequence[object]
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Return posterior means and standard deviations at one observation.

        Args:
            summary: Observed summary vector of the training width.

        Returns:
            ``(means, standard_deviations)`` on the original parameter scale.

        Raises:
            TypeError: If an entry is not real.
            ValueError: If the width is wrong or an entry is not finite.

        Equation:
            [EQ-INF-NPE-001].

        References:
            [REF-PAPAMAKARIOS-MURRAY-2016, Section 2.3].

        """
        features = _standardize(
            [require_finite_real(v, name="summary") for v in summary],
            self.summary_location,
            self.summary_scale,
        )
        outputs = self.network.forward(self.parameters, features)
        p = self.dimension
        means = tuple(
            self.parameter_location[j] + self.parameter_scale[j] * float(outputs[j])
            for j in range(p)
        )
        sds = tuple(
            self.parameter_scale[j] * math.exp(float(outputs[p + j])) for j in range(p)
        )
        return means, sds

    def log_density(
        self, theta: Sequence[object], summary: Sequence[object]
    ) -> float:
        """Return ``log q(theta | x)``.

        Args:
            theta: Parameter vector of width ``p``.
            summary: Observed summary vector.

        Returns:
            The diagonal-Gaussian log density.

        Raises:
            ValueError: If a width is wrong or an entry is not finite.

        Equation:
            [EQ-INF-NPE-001].

        References:
            [REF-PAPAMAKARIOS-MURRAY-2016, Section 2.3].

        """
        values = [require_finite_real(v, name="theta") for v in theta]
        if len(values) != self.dimension:
            raise ValueError("theta has the wrong width")
        means, sds = self.moments(summary)
        return math.fsum(
            -_LOG_SQRT_TWO_PI - math.log(s) - 0.5 * ((t - m) / s) ** 2
            for t, m, s in zip(values, means, sds, strict=True)
        )

    def sample(
        self,
        summary: Sequence[object],
        size: int,
        *,
        seed: int | RandomStream | None = None,
    ) -> tuple[tuple[float, ...], ...]:
        """Draw posterior samples at one observation.

        Args:
            summary: Observed summary vector.
            size: Non-negative number of draws.
            seed: Integer seed or stream; ``None`` is not reproducible.

        Returns:
            ``size`` parameter vectors.

        Raises:
            ValueError: If ``size`` is negative or the summary is invalid.

        Equation:
            [EQ-INF-NPE-001].

        References:
            [REF-PAPAMAKARIOS-MURRAY-2016, Section 2.2].

        """
        count = require_non_negative_integer(size, name="size")
        means, sds = self.moments(summary)
        rng = resolve_stream(seed).child("npe-sample").generator()
        return tuple(
            tuple(rng.gauss(m, s) for m, s in zip(means, sds, strict=True))
            for _ in range(count)
        )

    def credible_intervals(
        self, summary: Sequence[object], level: float = 0.9
    ) -> tuple[tuple[float, float], ...]:
        """Return central marginal credible intervals at one observation.

        Args:
            summary: Observed summary vector.
            level: Credible level strictly inside ``(0, 1)``.

        Returns:
            One ``(lower, upper)`` pair per parameter.

        Raises:
            ValueError: If ``level`` is outside ``(0, 1)``.

        Equation:
            [EQ-INF-NPE-001].

        References:
            [REF-PAPAMAKARIOS-MURRAY-2016, Section 2.2].

        """
        mass = require_finite_real(level, name="level")
        if not 0.0 < mass < 1.0:
            raise ValueError("level must lie strictly inside (0, 1)")
        z = statistics.NormalDist().inv_cdf(0.5 + 0.5 * mass)
        means, sds = self.moments(summary)
        return tuple((m - z * s, m + z * s) for m, s in zip(means, sds, strict=True))


def _standardize(
    values: Sequence[float], location: Sequence[float], scale: Sequence[float]
) -> list[float]:
    """Return ``(v - location) / scale`` elementwise with a width check.

    References:
        [EQ-INF-NPE-001]. No new equation is introduced.

    """
    if len(values) != len(location):
        raise ValueError(f"expected width {len(location)}, got {len(values)}")
    return [(v - c) / s for v, c, s in zip(values, location, scale, strict=True)]


def _location_scale(
    rows: Sequence[Sequence[float]],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return column means and standard deviations (``1`` for constant columns).

    References:
        [EQ-INF-NPE-001]. No new equation is introduced.

    """
    columns = list(zip(*rows, strict=True))
    location = tuple(statistics.fmean(column) for column in columns)
    scale = tuple(statistics.pstdev(column) or 1.0 for column in columns)
    return location, scale


def _finite_vector(
    values: Sequence[object], name: str, width: int | None
) -> list[float]:
    """Validate a finite vector of an expected width.

    References:
        [EQ-INF-NPE-001]. No new equation is introduced.

    """
    vector = [require_finite_real(v, name=name) for v in values]
    if not vector or (width is not None and len(vector) != width):
        raise ValueError(f"{name} must be a non-empty vector of a fixed width")
    return vector


def fit_amortized_gaussian_posterior(
    prior_sampler: PriorSampler,
    simulator: Simulator,
    simulations: int,
    *,
    hidden_layers: Sequence[int] = (16,),
    iterations: int = 2000,
    batch_size: int = 64,
    learning_rate: float = 0.01,
    seed: int | RandomStream | None = None,
) -> AmortizedGaussianPosterior:
    """Train a Gaussian neural posterior on simulated parameter/summary pairs.

    The loss is the mean negative log density
    ``sum_j [log sigma_j(x) + (theta_j - mu_j(x))^2 / (2 sigma_j(x)^2)]``
    (plus a constant) over a random mini-batch, minimized by Adam with
    reverse-mode gradients; parameters and summaries are standardized with
    their training moments.

    Args:
        prior_sampler: ``prior_sampler(rng) -> theta`` draws from the prior.
        simulator: ``simulator(theta, rng) -> x`` returns summary statistics.
        simulations: Number ``N >= 2`` of training pairs.
        hidden_layers: Hidden widths of the network (``tanh`` units).
        iterations: Positive number of Adam iterations.
        batch_size: Positive mini-batch size (capped at ``N``).
        learning_rate: Positive Adam step size.
        seed: Integer seed or stream; ``None`` is not reproducible.

    Returns:
        The trained amortized posterior and its loss trace.

    Raises:
        TypeError: If a callable is missing or a value is not real.
        ValueError: If a setting is invalid or the simulated vectors change
            width or are not finite.

    Equation:
        [EQ-INF-NPE-001], [EQ-NN-MLP-001], [EQ-OPT-ADAM-002].

    References:
        [REF-PAPAMAKARIOS-MURRAY-2016, Proposition 1 and Algorithm 2],
        [REF-KINGMA-BA-2015].

    """
    if not callable(prior_sampler) or not callable(simulator):
        raise TypeError("prior_sampler and simulator must be callable")
    count = require_non_negative_integer(simulations, name="simulations")
    if count < 2:
        raise ValueError("simulations must be at least 2")
    batch = require_non_negative_integer(batch_size, name="batch_size")
    if batch == 0:
        raise ValueError("batch_size must be positive")
    widths = tuple(
        require_non_negative_integer(width, name="hidden_layers")
        for width in hidden_layers
    )
    if any(width == 0 for width in widths):
        raise ValueError("hidden layer widths must be positive")
    stream = resolve_stream(seed)
    thetas: list[list[float]] = []
    summaries: list[list[float]] = []
    for index in range(count):
        rng = stream.child("npe-simulation", index).generator()
        theta = _finite_vector(
            prior_sampler(rng), "prior draw", len(thetas[0]) if thetas else None
        )
        summary = _finite_vector(
            simulator(theta, rng),
            "simulated summary",
            len(summaries[0]) if summaries else None,
        )
        thetas.append(theta)
        summaries.append(summary)
    theta_location, theta_scale = _location_scale(thetas)
    summary_location, summary_scale = _location_scale(summaries)
    targets = [_standardize(t, theta_location, theta_scale) for t in thetas]
    features = [_standardize(s, summary_location, summary_scale) for s in summaries]
    p = len(theta_location)
    network = MLP((len(summary_location), *widths, 2 * p))
    initial = network.initial_parameters(
        stream.child("npe-network"), output_scale=0.1
    )
    take = min(batch, count)

    def objective(
        weights: tuple[float, ...], batch_stream: RandomStream
    ) -> tuple[float, Sequence[float]]:
        """Return the mini-batch negative log-likelihood and its gradient.

        References:
            [EQ-INF-NPE-001], [REF-PAPAMAKARIOS-MURRAY-2016, Section 2.2].

        """
        rows = batch_stream.generator().sample(range(count), take)

        def loss(phi: tuple[Any, ...]) -> Any:
            """Return the mean negative Gaussian log density over the batch.

            References:
                [EQ-INF-NPE-001].

            """
            total: Any = 0.0
            for row in rows:
                outputs = network.forward(phi, features[row])
                for j in range(p):
                    log_sd = outputs[p + j]
                    error = targets[row][j] - outputs[j]
                    precision = admath.exp(-2.0 * log_sd)
                    total = total + log_sd + 0.5 * error * error * precision
            return total / take

        return reverse_value_and_gradient(loss, weights)

    result = adam_minimize(
        objective,
        initial,
        iterations=iterations,
        learning_rate=learning_rate,
        tail_fraction=0.1,
        seed=stream.child("npe-adam"),
    )
    return AmortizedGaussianPosterior(
        network=network,
        parameters=result.x,
        summary_location=summary_location,
        summary_scale=summary_scale,
        parameter_location=theta_location,
        parameter_scale=theta_scale,
        losses=result.losses,
        simulations=count,
    )
