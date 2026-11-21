"""Neural ratio estimation for simulation-based inference.

Train a binary classifier ``d(theta, x)`` to tell pairs drawn from the joint
law ``p(theta, x) = p(theta) p(x | theta)`` (label 1) from pairs drawn from
the product of the marginals ``p(theta) p(x)`` (label 0, obtained by
shuffling ``theta`` among the simulated pairs). With equal class weights the
cross-entropy

``-E_joint log d(theta, x) - E_product log(1 - d(theta, x))``

is minimized pointwise by ``d* = p(theta, x) / (p(theta, x) + p(theta) p(x))``
(set the derivative of ``-a log d - b log(1 - d)`` to zero: ``d = a / (a +
b)``), so the logit of the optimal classifier is

``log d* / (1 - d*) = log p(theta, x) / (p(theta) p(x))
= log p(x | theta) / p(x) = log p(theta | x) / p(theta)``

[EQ-GEN-NRE-001]. The classifier here outputs the logit directly, so the
trained network is an estimate of the log likelihood-to-evidence ratio
``log r(theta, x)``, and ``log p(theta) + log r(theta, x_obs)`` is an
unnormalized log posterior that any sampler of the library can use
[EQ-GEN-NRE-002].

The estimate is only as good as the classifier: it is reliable where
simulated pairs are dense and unconstrained elsewhere, and it can be
overconfident (Hermans et al. 2022). The integral
``int p(theta) r(theta, x) d theta`` equals one for the exact ratio and is
reported by :meth:`RatioEstimator.posterior_on_grid` as a diagnostic.
Networks are small and trained in scalar Python; large-scale training is out
of scope.

References:
    [REF-HERMANS-BEGY-LOUPPE-2020], [REF-HERMANS-ETAL-2022].

"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ..foundations.random_streams import RandomStream, resolve_stream
from ..inference import MCMCResult, random_walk_metropolis
from ..numerics import MLP
from ..optimization import adam_minimize
from . import _backprop
from ._common import Point, positive_integer, positive_real
from .flow_matching import loss_window_means, standardize_columns
from .posterior_estimation import (
    PriorSampler,
    Simulator,
    finite_vector,
    simulate_pairs,
)

LogPrior = Callable[[Sequence[float]], float]


def _softplus(value: float) -> float:
    """Return ``log(1 + exp(value))`` without overflow.

    References:
        [EQ-GEN-NRE-001]. No new equation is introduced.

    """
    return max(value, 0.0) + math.log1p(math.exp(-abs(value)))


def ratio_loss_and_gradient(
    network: MLP,
    parameters: Sequence[float],
    joint: Sequence[Sequence[float]],
    product: Sequence[Sequence[float]],
) -> tuple[float, list[float]]:
    """Return the balanced cross-entropy of a fixed batch and its gradient.

    With logit ``f``, a joint row contributes ``softplus(-f)`` and a product
    row ``softplus(f)``; the loss is the mean over all rows.

    Args:
        network: Classifier network with one output (the logit).
        parameters: Flat parameter vector.
        joint: Network inputs ``(theta, x)`` drawn from the joint law.
        product: Network inputs with ``theta`` and ``x`` independent.

    Returns:
        ``(loss, gradient)``.

    Raises:
        ValueError: If a batch is empty or the parameters have the wrong
            length.

    References:
        [EQ-GEN-NRE-001], [EQ-GEN-FM-005], [REF-HERMANS-BEGY-LOUPPE-2020].

    """
    if not joint or not product:
        raise ValueError("joint and product batches must not be empty")
    first = len(joint)

    def cross_entropy(case: int, output: list[float]) -> tuple[float, list[float]]:
        """Return the cross-entropy of one row and its logit derivative.

        References:
            [EQ-GEN-NRE-001].

        """
        logit = output[0]
        probability = 0.5 * (1.0 + math.tanh(0.5 * logit))
        if case < first:
            return _softplus(-logit), [probability - 1.0]
        return _softplus(logit), [probability]

    return _backprop.batch_loss_and_gradient(
        network.layer_sizes, parameters, [*joint, *product], cross_entropy
    )


@dataclass(frozen=True, slots=True)
class GridPosterior:
    """One-dimensional posterior tabulated on a grid.

    Args:
        grid: Increasing parameter values.
        density: Posterior density at the grid points, normalized so that
            its trapezoidal integral over the grid is one.
        mean: Posterior mean by the trapezoidal rule.
        standard_deviation: Posterior standard deviation by the same rule.
        ratio_normalization: Trapezoidal ``int p(theta) r(theta, x) d theta``
            when the supplied prior is normalized; one for the exact ratio,
            so its distance from one measures the error of the classifier
            (and of a grid that misses prior mass).

    References:
        [EQ-GEN-NRE-002], [REF-HERMANS-BEGY-LOUPPE-2020].

    """

    grid: tuple[float, ...]
    density: tuple[float, ...]
    mean: float
    standard_deviation: float
    ratio_normalization: float


def _trapezoid(grid: Sequence[float], values: Sequence[float]) -> float:
    """Return the trapezoidal integral of tabulated values.

    References:
        [EQ-GEN-NRE-002]. Composite trapezoidal rule; no new equation.

    """
    return math.fsum(
        0.5 * (grid[i + 1] - grid[i]) * (values[i + 1] + values[i])
        for i in range(len(grid) - 1)
    )


@dataclass(frozen=True, slots=True)
class RatioEstimator:
    """A trained estimator of ``log r(theta, x) = log p(x | theta) / p(x)``.

    Adam has no convergence test: ``loss_decreased`` is a necessary sanity
    check only. A classifier that cannot separate the classes has loss
    ``log 2``, which is also the value at initialization.

    Args:
        network: Classifier network; its single output is the logit.
        parameters: Trained flat network parameters.
        summary_location: Training means of the observations.
        summary_scale: Training standard deviations of the observations.
        parameter_location: Training means of ``theta``.
        parameter_scale: Training standard deviations of ``theta``.
        losses: Mini-batch cross-entropy per Adam iteration.
        initial_loss: Mean loss over the first tenth of the iterations.
        final_loss: Mean loss over the last tenth of the iterations.
        simulations: Number of simulator runs used for training.

    References:
        [EQ-GEN-NRE-001], [REF-HERMANS-BEGY-LOUPPE-2020].

    """

    network: MLP
    parameters: tuple[float, ...]
    summary_location: Point
    summary_scale: Point
    parameter_location: Point
    parameter_scale: Point
    losses: tuple[float, ...]
    initial_loss: float
    final_loss: float
    simulations: int

    @property
    def dimension(self) -> int:
        """Return the number ``p`` of parameters.

        References:
            [EQ-GEN-NRE-001].

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

    def _standardized_summary(self, summary: Sequence[object]) -> list[float]:
        """Return the standardized observation.

        Raises:
            ValueError: If the width is wrong or an entry is not finite.

        References:
            [EQ-GEN-NRE-001]. No new equation is introduced.

        """
        values = finite_vector(summary, "summary", len(self.summary_location))
        return [
            (v - m) / s
            for v, m, s in zip(
                values, self.summary_location, self.summary_scale, strict=True
            )
        ]

    def _logit_function(
        self, summary: Sequence[object]
    ) -> Callable[[Sequence[float]], float]:
        """Return ``theta -> log r(theta, x)`` for one fixed observation.

        References:
            [EQ-GEN-NRE-001].

        """
        tail = self._standardized_summary(summary)
        layers = _backprop.split_layers(self.network.layer_sizes, self.parameters)
        location = self.parameter_location
        scale = self.parameter_scale

        def logit(theta: Sequence[float]) -> float:
            """Return the classifier logit at ``(theta, x)``.

            References:
                [EQ-GEN-NRE-001].

            """
            head = [(v - m) / s for v, m, s in zip(theta, location, scale, strict=True)]
            return _backprop.forward(layers, head + tail)[0]

        return logit

    def log_ratio(self, theta: Sequence[object], summary: Sequence[object]) -> float:
        """Return the estimated ``log p(x | theta) / p(x)``.

        Args:
            theta: Parameter vector of width ``p``.
            summary: Observed vector of the training width.

        Returns:
            The classifier logit.

        Raises:
            TypeError: If an entry is not real.
            ValueError: If a width is wrong or an entry is not finite.

        References:
            [EQ-GEN-NRE-001], [REF-HERMANS-BEGY-LOUPPE-2020].

        """
        values = finite_vector(theta, "theta", self.dimension)
        return self._logit_function(summary)(values)

    def posterior_on_grid(
        self,
        summary: Sequence[object],
        log_prior: LogPrior,
        grid: Sequence[float],
    ) -> GridPosterior:
        """Tabulate the posterior of a scalar parameter on a grid.

        The unnormalized density ``p(theta) r(theta, x)`` is evaluated at
        the grid points and integrated by the trapezoidal rule. The grid
        must cover the posterior mass; the result is not extrapolated.

        Args:
            summary: Observed vector.
            log_prior: ``log_prior(theta) -> float`` for a length-one
                ``theta``; ``-inf`` marks points outside the support.
            grid: At least three strictly increasing finite values.

        Returns:
            :class:`GridPosterior` with density, mean, standard deviation,
            and the normalization diagnostic.

        Raises:
            TypeError: If ``log_prior`` is not callable or an entry is not
                real.
            ValueError: If the estimator is not one-dimensional, the grid
                is invalid, the prior returns NaN or ``+inf``, or the
                density vanishes on the whole grid.

        References:
            [EQ-GEN-NRE-002], [REF-HERMANS-BEGY-LOUPPE-2020].

        """
        if self.dimension != 1:
            raise ValueError("posterior_on_grid needs a one-dimensional parameter")
        if not callable(log_prior):
            raise TypeError("log_prior must be callable")
        points = finite_vector(grid, "grid", None)
        if len(points) < 3 or any(
            b <= a for a, b in zip(points, points[1:], strict=False)
        ):
            raise ValueError("grid needs at least three strictly increasing values")
        logit = self._logit_function(summary)
        logs = []
        for point in points:
            prior = float(log_prior((point,)))
            if math.isnan(prior) or prior == math.inf:
                raise ValueError("log_prior must return a finite value or -inf")
            logs.append(prior + logit((point,)) if prior > -math.inf else -math.inf)
        peak = max(logs)
        if peak == -math.inf:
            raise ValueError("the prior density vanishes on the whole grid")
        shifted = [math.exp(value - peak) for value in logs]
        mass = _trapezoid(points, shifted)
        density = tuple(value / mass for value in shifted)
        mean = _trapezoid(points, [p * d for p, d in zip(points, density, strict=True)])
        variance = _trapezoid(
            points, [(p - mean) ** 2 * d for p, d in zip(points, density, strict=True)]
        )
        return GridPosterior(
            grid=points,
            density=density,
            mean=mean,
            standard_deviation=math.sqrt(variance),
            ratio_normalization=mass * math.exp(peak),
        )

    def sample_posterior(
        self,
        summary: Sequence[object],
        log_prior: LogPrior,
        initial: Sequence[float],
        *,
        proposal_scale: float | Sequence[float],
        draws: int,
        warmup: int = 0,
        chains: int = 1,
        thin: int = 1,
        seed: int | RandomStream | None = None,
        check_convergence: bool = True,
    ) -> MCMCResult:
        """Sample ``p(theta) r(theta, x)`` by random-walk Metropolis.

        The target log-density is ``log_prior(theta) + log r(theta, x)``;
        the chain is run by
        :func:`probstat.inference.random_walk_metropolis` with a diagonal
        Gaussian proposal, and its convergence diagnostics apply. The
        chain is exact for the *estimated* ratio, not for the true
        posterior.

        Args:
            summary: Observed vector.
            log_prior: ``log_prior(theta) -> float``; ``-inf`` outside the
                support.
            initial: Starting point of width ``p`` with finite prior.
            proposal_scale: Positive proposal standard deviation (one
                number, or one per parameter).
            draws: Kept draws per chain.
            warmup: Discarded initial iterations per chain.
            chains: Number of chains.
            thin: Thinning interval.
            seed: Integer seed or stream; ``None`` is not reproducible.
            check_convergence: Emit convergence warnings of the sampler.

        Returns:
            The multi-chain result of the sampler.

        Raises:
            TypeError: If ``log_prior`` is not callable.
            ValueError: For invalid settings, widths, or a start with
                non-finite log-density.

        References:
            [EQ-GEN-NRE-002], [EQ-MCMC-MH-001],
            [REF-HERMANS-BEGY-LOUPPE-2020].

        """
        if not callable(log_prior):
            raise TypeError("log_prior must be callable")
        start = finite_vector(initial, "initial", self.dimension)
        if isinstance(proposal_scale, Sequence):
            scales = [positive_real(s, name="proposal_scale") for s in proposal_scale]
        else:
            scales = [positive_real(proposal_scale, name="proposal_scale")] * len(start)
        if len(scales) != len(start):
            raise ValueError("proposal_scale must have one entry per parameter")
        logit = self._logit_function(summary)

        def log_density(theta: Sequence[float]) -> float:
            """Return ``log p(theta) + log r(theta, x)``.

            References:
                [EQ-GEN-NRE-002].

            """
            prior = float(log_prior(theta))
            if prior == -math.inf:
                return prior
            return prior + logit(theta)

        covariance = [
            [scales[i] ** 2 if i == j else 0.0 for j in range(len(start))]
            for i in range(len(start))
        ]
        return random_walk_metropolis(
            log_density,
            start,
            proposal_covariance=covariance,
            draws=draws,
            warmup=warmup,
            chains=chains,
            thin=thin,
            seed=seed,
            check_convergence=check_convergence,
        )


def fit_ratio_estimator(
    prior_sampler: PriorSampler,
    simulator: Simulator,
    simulations: int,
    *,
    hidden_layers: Sequence[int] = (32, 32),
    iterations: int = 3000,
    batch_size: int = 64,
    learning_rate: float = 0.005,
    tail_fraction: float = 0.2,
    seed: int | RandomStream | None = None,
) -> RatioEstimator:
    """Train a likelihood-to-evidence ratio estimator on simulated pairs.

    Every Adam iteration draws ``B`` simulated pairs without replacement
    (label 1) and builds ``B`` product-of-marginals pairs by giving
    observation ``i`` the parameter of pair ``i + k`` (cyclically) for a
    random shift ``k`` in ``1..B-1`` (label 0), so no contrast pair equals a
    joint pair. The loss is the mean cross-entropy over the ``2 B`` rows.
    Parameters and observations are standardized with their training
    moments.

    Args:
        prior_sampler: ``prior_sampler(rng) -> theta`` draws from the prior.
        simulator: ``simulator(theta, rng) -> x`` returns the observation
            (or summary statistics).
        simulations: Number ``N >= 2`` of training pairs.
        hidden_layers: Hidden widths of the ``tanh`` classifier.
        iterations: Positive number of Adam iterations.
        batch_size: Mini-batch size ``B >= 2`` (capped at ``N``).
        learning_rate: Positive Adam step size.
        tail_fraction: Fraction in ``[0, 1)`` of final iterates averaged.
        seed: Integer seed or stream; ``None`` is not reproducible.

    Returns:
        The trained estimator with its loss trace and simulation count.

    Raises:
        TypeError: If a callable is missing or a value is not real.
        ValueError: If a setting is invalid or the simulated vectors change
            width or are not finite.

    References:
        [EQ-GEN-NRE-001], [EQ-OPT-ADAM-002], [REF-HERMANS-BEGY-LOUPPE-2020],
        [REF-KINGMA-BA-2015].

    """
    steps = positive_integer(iterations, name="iterations")
    batch = positive_integer(batch_size, name="batch_size")
    if batch < 2:
        raise ValueError("batch_size must be at least 2")
    rate = positive_real(learning_rate, name="learning_rate")
    widths = tuple(positive_integer(w, name="hidden_layers") for w in hidden_layers)
    if not widths:
        raise ValueError("hidden_layers needs at least one layer")
    stream = resolve_stream(seed)
    thetas, observations = simulate_pairs(prior_sampler, simulator, simulations, stream)
    theta_location, theta_scale, heads = standardize_columns(thetas)
    summary_location, summary_scale, tails = standardize_columns(observations)
    count = len(thetas)
    take = min(batch, count)
    network = MLP((len(theta_location) + len(summary_location), *widths, 1))

    def objective(
        parameters: tuple[float, ...], batch_stream: RandomStream
    ) -> tuple[float, Sequence[float]]:
        """Return the mini-batch cross-entropy and its gradient.

        References:
            [EQ-GEN-NRE-001], [EQ-GEN-FM-005].

        """
        generator = batch_stream.generator()
        rows = generator.sample(range(count), take)
        shift = generator.randrange(1, take)
        joint = [[*heads[row], *tails[row]] for row in rows]
        product = [
            [*heads[rows[(i + shift) % take]], *tails[row]]
            for i, row in enumerate(rows)
        ]
        return ratio_loss_and_gradient(network, parameters, joint, product)

    result = adam_minimize(
        objective,
        network.initial_parameters(stream.child("nre-network")),
        iterations=steps,
        learning_rate=rate,
        tail_fraction=tail_fraction,
        seed=stream.child("nre-adam"),
    )
    initial_loss, final_loss = loss_window_means(result.losses)
    return RatioEstimator(
        network=network,
        parameters=result.x,
        summary_location=summary_location,
        summary_scale=summary_scale,
        parameter_location=theta_location,
        parameter_scale=theta_scale,
        losses=result.losses,
        initial_loss=initial_loss,
        final_loss=final_loss,
        simulations=count,
    )
