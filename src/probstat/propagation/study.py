"""Forward propagation of a :class:`~probstat.modeling.ProbabilisticModel`.

``propagate`` draws inputs from the model's joint law with a chosen design
(Monte Carlo, Latin hypercube, Halton, or Sobol), evaluates every quantity of
interest once per input, and returns raw outputs together with distributional
summaries: moments with sampling uncertainty, quantiles, exceedance curves,
and input-output rank correlations.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.linalg import Vector
from probstat.modeling import ProbabilisticModel
from probstat.monte_carlo import (
    MonteCarloEstimate,
    ProbabilityEstimate,
    estimate_mean,
    estimate_probability,
)
from probstat.sampling import (
    SamplingDesign,
    halton,
    latin_hypercube,
    monte_carlo_design,
    sobol,
)
from probstat.statistics import (
    excess_kurtosis,
    quantile,
    skewness,
    spearman_correlation,
)

DESIGNS = ("mc", "lhs", "halton", "sobol")


@dataclass(frozen=True, slots=True)
class OutputDistribution:
    """Empirical distribution of one propagated quantity of interest.

    Args:
        name: QoI name.
        values: Propagated outputs in input-design order.
        mean_estimate: Mean with variance, standard error, and interval. The
            interval assumes independent draws; for LHS/QMC designs it is a
            conservative Monte Carlo reference, not a randomized-QMC interval.

    References:
        [REF-SMITH-UQ-2024, Chapters 3-4], [REF-HYNDMAN-FAN-1996].

    """

    name: str
    values: tuple[float, ...]
    mean_estimate: MonteCarloEstimate

    @property
    def mean(self) -> float:
        """Return the sample mean.

        Equation:
            [EQ-MC-MEAN-001].

        References:
            [REF-SMITH-UQ-2024, Chapter 3].

        """
        return self.mean_estimate.estimate

    @property
    def standard_deviation(self) -> float:
        """Return the unbiased sample standard deviation.

        Equation:
            [EQ-MC-VAR-001].

        References:
            [REF-SMITH-UQ-2024, Chapter 3].

        """
        return math.sqrt(self.mean_estimate.sample_variance)

    @property
    def skewness(self) -> float:
        """Return the moment skewness of the outputs.

        Equation:
            [EQ-STAT-SHAPE-001].

        References:
            [REF-JOANES-GILL-1998].

        """
        return skewness(self.values)

    @property
    def excess_kurtosis(self) -> float:
        """Return the moment excess kurtosis of the outputs.

        Equation:
            [EQ-STAT-SHAPE-001].

        References:
            [REF-JOANES-GILL-1998].

        """
        return excess_kurtosis(self.values)

    def quantile(self, probability: float) -> float:
        """Return the type-7 sample quantile of the outputs.

        Args:
            probability: Level in ``[0, 1]``.

        Returns:
            Output quantile.

        Equation:
            [EQ-STAT-QUANT-001].

        References:
            [REF-HYNDMAN-FAN-1996].

        """
        return quantile(self.values, probability)

    def exceedance(
        self, threshold: float, *, confidence: float = 0.95
    ) -> ProbabilityEstimate:
        """Estimate ``P(Y > threshold)`` with a Wilson interval.

        Args:
            threshold: Finite threshold.
            confidence: Nominal interval confidence.

        Returns:
            Probability estimate.

        Equation:
            [EQ-PROP-THRESH-001], [EQ-MC-PROB-001].

        References:
            [REF-WILSON-1927], [REF-SMITH-UQ-2024, Chapter 4].

        """
        level = require_finite_real(threshold, name="threshold")
        return estimate_probability(
            (value > level for value in self.values), confidence=confidence
        )

    def exceedance_curve(self, points: int = 50) -> tuple[tuple[float, float], ...]:
        """Return the empirical exceedance curve ``(y, P(Y > y))``.

        Args:
            points: Number of evenly spaced levels between the extremes.

        Returns:
            Pairs of output level and empirical exceedance fraction.

        Raises:
            ValueError: If ``points < 2``.

        Equation:
            [EQ-STAT-005], [EQ-PROP-THRESH-001].

        References:
            [REF-SMITH-UQ-2024, Chapter 4].

        """
        if points < 2:
            raise ValueError("points must be at least two")
        ordered = sorted(self.values)
        low, high = ordered[0], ordered[-1]
        n = len(ordered)
        curve: list[tuple[float, float]] = []
        index = 0
        for k in range(points):
            level = low + (high - low) * k / (points - 1)
            while index < n and ordered[index] <= level:
                index += 1
            curve.append((level, (n - index) / n))
        return tuple(curve)


@dataclass(frozen=True, slots=True)
class PropagationStudy:
    """Inputs, outputs, and summaries of one forward propagation.

    Args:
        model_name: Label of the propagated model.
        design: Unit-hypercube design that generated the inputs.
        inputs: Physical input vectors in design order.
        input_names: Input names in model order.
        outputs: Read-only QoI-name mapping to output distributions.
        model_evaluations: Forward-model calls charged to this study.

    References:
        [REF-SMITH-UQ-2024, Chapters 3-4], [REF-OWEN-2013, Chapter 10].

    """

    model_name: str
    design: SamplingDesign
    inputs: tuple[Vector, ...]
    input_names: tuple[str, ...]
    outputs: Mapping[str, OutputDistribution]
    model_evaluations: int

    def __getitem__(self, name: str) -> OutputDistribution:
        """Return the output distribution of one QoI.

        Args:
            name: QoI name.

        Returns:
            Output distribution.

        References:
            [REF-SMITH-UQ-2024, Chapter 3]. No equation is evaluated.

        """
        return self.outputs[name]

    def input_column(self, name: str) -> tuple[float, ...]:
        """Return the sampled values of one input.

        Args:
            name: Input name.

        Returns:
            Values in design order.

        Raises:
            ValueError: If the input name is unknown.

        References:
            [REF-SMITH-UQ-2024, Chapter 3]. No equation is evaluated.

        """
        index = self.input_names.index(name)
        return tuple(point[index] for point in self.inputs)

    def rank_correlations(self, quantity: str) -> Mapping[str, float]:
        """Return Spearman correlations between each input and one QoI.

        This is a screening measure of monotone association, not a
        variance-based sensitivity index.

        Args:
            quantity: QoI name.

        Returns:
            Read-only input-name mapping to rank correlation.

        Equation:
            [EQ-STAT-RANK-001].

        References:
            [REF-HOLLANDER-WOLFE-CHICKEN-2014, Chapter 8],
            [REF-SALTELLI-2008, Chapter 1].

        """
        values = self.outputs[quantity].values
        result: dict[str, float] = {}
        for name in self.input_names:
            try:
                result[name] = spearman_correlation(self.input_column(name), values)
            except ValueError:
                result[name] = math.nan
        return MappingProxyType(result)


def make_design(
    method: str,
    size: int,
    dimension: int,
    *,
    seed: int | RandomStream | None = None,
) -> SamplingDesign:
    """Build a unit-hypercube design by name.

    Args:
        method: One of ``"mc"``, ``"lhs"``, ``"halton"``, or ``"sobol"``.
        size: Positive number of points.
        dimension: Positive dimension.
        seed: Seed for randomized designs; quasi-random designs are
            randomized (scrambled/shifted) whenever a seed is supplied.

    Returns:
        Sampling design.

    Raises:
        ValueError: If ``method`` is unknown.

    Equation:
        [EQ-SAMP-LHS-001], [EQ-SAMP-HALTON-001], [EQ-SAMP-SOBOL-001].

    References:
        [REF-OWEN-2013, Chapters 10 and 15].

    """
    if method == "mc":
        return monte_carlo_design(size, dimension, seed=seed)
    if method == "lhs":
        return latin_hypercube(size, dimension, seed=seed)
    if method == "halton":
        randomized = seed is not None
        return halton(size, dimension, seed=seed, scramble=randomized, shift=randomized)
    if method == "sobol":
        return sobol(size, dimension, seed=seed, scramble=seed is not None)
    raise ValueError(f"design must be one of {DESIGNS}")


def propagate(
    model: ProbabilisticModel,
    size: int,
    *,
    design: str = "mc",
    seed: int | RandomStream | None = None,
    confidence: float = 0.95,
) -> PropagationStudy:
    """Propagate the joint input law of a model through every QoI.

    Args:
        model: Probabilistic model to propagate.
        size: Number of input realizations, at least two.
        design: Sampling design name (see :data:`DESIGNS`).
        seed: Integer seed or stream; the design uses the child stream
            ``"propagation"`` so other workflows can share one root seed.
        confidence: Nominal confidence for mean intervals.

    Returns:
        Propagation study with raw outputs and summaries.

    Raises:
        ValueError: If ``size < 2`` or the design is unknown.

    Equation:
        [EQ-UQ-001], [EQ-MC-MEAN-001], [EQ-SAMP-INV-001].

    References:
        [REF-SMITH-UQ-2024, Chapters 3-4], [REF-OWEN-2013, Chapter 10].

    """
    if isinstance(size, bool) or not isinstance(size, int) or size < 2:
        raise ValueError("size must be an integer of at least two")
    stream = resolve_stream(seed).child("propagation", design)
    unit = make_design(design, size, model.dimension, seed=stream)
    inputs = model.inputs.sample_design(unit)
    before = model.evaluations
    raw = model.evaluate_many(inputs)
    outputs = {
        name: OutputDistribution(
            name, values, estimate_mean(values, confidence=confidence)
        )
        for name, values in raw.items()
    }
    return PropagationStudy(
        model.name,
        unit,
        inputs,
        model.inputs.names,
        MappingProxyType(outputs),
        model.evaluations - before,
    )


def compare_designs(
    model: ProbabilisticModel,
    quantity: str,
    size: int,
    *,
    designs: Sequence[str] = DESIGNS,
    replicates: int = 10,
    seed: int | RandomStream | None = None,
) -> Mapping[str, tuple[float, float]]:
    """Compare designs by the spread of the mean estimate over replicates.

    Each design is repeated with independent randomization; the empirical
    standard deviation of the replicate means measures its estimation error.

    Args:
        model: Probabilistic model.
        quantity: QoI whose mean is estimated.
        size: Points per replicate.
        designs: Design names to compare.
        replicates: Independent replicates per design, at least two.
        seed: Root seed or stream.

    Returns:
        Read-only mapping ``design -> (mean of means, std of means)``.

    Raises:
        ValueError: If ``replicates < 2``.

    Equation:
        [EQ-MC-VAR-001], [EQ-SAMP-RQMC-001].

    References:
        [REF-OWEN-2013, Section 17.2], [REF-L-ECUYER-2018].

    """
    if replicates < 2:
        raise ValueError("replicates must be at least two")
    root = resolve_stream(seed)
    summary: dict[str, tuple[float, float]] = {}
    for method in designs:
        means: list[float] = []
        for replicate in range(replicates):
            study = propagate(
                model,
                size,
                design=method,
                seed=root.child("compare", method, replicate),
            )
            means.append(study[quantity].mean)
        center = math.fsum(means) / replicates
        spread = math.sqrt(
            math.fsum((value - center) ** 2 for value in means) / (replicates - 1)
        )
        summary[method] = (center, spread)
    return MappingProxyType(summary)


def summarize(
    values: Iterable[float], probabilities: Sequence[float] = (0.05, 0.5, 0.95)
) -> Mapping[str, float]:
    """Return a compact numeric summary of an output sample.

    Args:
        values: Finite output sample with at least two values.
        probabilities: Quantile levels to report.

    Returns:
        Mapping with ``mean``, ``std`` and ``q<level>`` entries.

    Equation:
        [EQ-MC-MEAN-001], [EQ-MC-VAR-001], [EQ-STAT-QUANT-001].

    References:
        [REF-HYNDMAN-FAN-1996], [REF-SMITH-UQ-2024, Chapter 3].

    """
    sample = tuple(values)
    estimate = estimate_mean(sample)
    result = {"mean": estimate.estimate, "std": math.sqrt(estimate.sample_variance)}
    for p in probabilities:
        result[f"q{p:g}"] = quantile(sample, p)
    return MappingProxyType(result)
