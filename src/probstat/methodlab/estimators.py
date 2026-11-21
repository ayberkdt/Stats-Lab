"""The estimator contract of the method laboratory.

An :class:`Estimator` is a named, seeded procedure that returns one random
replicate of an estimate together with the work it cost. Everything else in
:mod:`probstat.methodlab` is written against this one contract: combinators
take estimators and return new estimators, and the certification checks take
an estimator and a claim about it. An estimator also carries its *lineage*,
the ordered list of derivation steps that produced it, so that a derived
method documents how it was obtained.

The contract is deliberately small:

- ``sampler(stream)`` returns a :class:`Draw` and consumes randomness only
  from ``stream`` (a :class:`~probstat.foundations.RandomStream`) or its
  children, so two calls with the same stream return the same draw;
- ``Draw.value`` is one realization of the estimator and ``Draw.cost`` is the
  work charged for it in user-chosen units (model evaluations, seconds, ...).

Whether the estimator is unbiased for some target is a *claim*, not a
property of the type; :mod:`probstat.methodlab.certify` tests such claims.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_finite_real,
    require_non_negative_integer,
)


@dataclass(frozen=True, slots=True)
class Draw:
    """One replicate of an estimator and the work it cost.

    Args:
        value: Finite realized estimate.
        cost: Finite non-negative work charged for this replicate.

    References:
        [REF-GLASSERMAN-2003, Section 1.1.3] (cost-aware comparison of estimators). No
        equation is evaluated.

    """

    value: float
    cost: float

    def __post_init__(self) -> None:
        """Validate the realized value and its cost.

        Raises:
            TypeError: If a field is not a real number.
            ValueError: If ``value`` is not finite or ``cost`` is negative.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION]. No equation
            is evaluated.

        """
        object.__setattr__(self, "value", require_finite_real(self.value, name="value"))
        object.__setattr__(
            self, "cost", require_non_negative_finite_real(self.cost, name="cost")
        )


Sampler = Callable[[RandomStream], Draw]


@dataclass(frozen=True, slots=True)
class Estimator:
    """A named, seeded estimator with the record of how it was derived.

    Args:
        name: Non-empty label used in reports.
        sampler: Function mapping a random stream to one :class:`Draw`. It
            must be deterministic given the stream.
        lineage: Derivation steps, oldest first. A base estimator has an
            empty lineage; every combinator appends one sentence.

    References:
        [REF-GLASSERMAN-2003, Section 1.1.3]. No equation is evaluated.

    """

    name: str
    sampler: Sampler
    lineage: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate the label, the sampler, and the lineage.

        Raises:
            TypeError: If ``sampler`` is not callable or a lineage entry is
                not a string.
            ValueError: If ``name`` is empty.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION]. No equation
            is evaluated.

        """
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        if not callable(self.sampler):
            raise TypeError("sampler must be callable")
        lineage = tuple(self.lineage)
        if not all(isinstance(step, str) for step in lineage):
            raise TypeError("lineage entries must be strings")
        object.__setattr__(self, "lineage", lineage)

    def draw(self, stream: int | RandomStream) -> Draw:
        """Return one replicate of the estimator.

        Args:
            stream: Seed or stream that supplies all randomness of the draw.

        Returns:
            The realized value and its cost.

        Raises:
            TypeError: If the sampler does not return a :class:`Draw`.

        References:
            [REF-GLASSERMAN-2003, Section 1.1.3]. No equation is evaluated.

        """
        result = self.sampler(resolve_stream(stream))
        if not isinstance(result, Draw):
            raise TypeError(f"sampler of {self.name!r} must return a Draw")
        return result

    def replicates(self, count: int, seed: int | RandomStream) -> tuple[Draw, ...]:
        """Return independent replicates on numbered child streams.

        Replicate ``i`` uses the child stream ``("replicate", i)`` of ``seed``,
        so two estimators replicated with the same seed see common random
        numbers replicate by replicate.

        Args:
            count: Positive number of replicates.
            seed: Root seed or stream.

        Returns:
            ``count`` draws in replicate order.

        Raises:
            ValueError: If ``count`` is not positive.

        References:
            [REF-STEELE-LEA-FLOOD-2014], [EQ-RNG-SPLITMIX-001].

        """
        size = require_non_negative_integer(count, name="count")
        if size == 0:
            raise ValueError("count must be positive")
        root = resolve_stream(seed)
        return tuple(self.draw(root.child("replicate", i)) for i in range(size))

    def derived(self, name: str, sampler: Sampler, step: str) -> Estimator:
        """Return a new estimator whose lineage extends this one.

        Args:
            name: Label of the derived estimator.
            sampler: Sampler of the derived estimator.
            step: One sentence describing the derivation step.

        Returns:
            Estimator with lineage ``self.lineage + (step,)``.

        References:
            Project reporting policy [REF-PROBSTAT-REPORTING]. No equation is
            evaluated.

        """
        return Estimator(name, sampler, (*self.lineage, step))


@dataclass(frozen=True, slots=True)
class ReplicateSummary:
    """Sample moments of a set of replicates of one estimator.

    Args:
        name: Estimator label.
        count: Number of replicates ``R >= 2``.
        mean: Sample mean of the replicate values.
        variance: Unbiased sample variance of the replicate values.
        mean_cost: Sample mean of the replicate costs.
        fourth_central_moment: Sample fourth central moment (divisor ``R``),
            used for the standard error of the variance.

    References:
        [REF-MORRIS-WHITE-CROWTHER-2019, Table 6], [EQ-MLAB-MCSE-001],
        [EQ-MLAB-WNV-001].

    """

    name: str
    count: int
    mean: float
    variance: float
    mean_cost: float
    fourth_central_moment: float

    @property
    def standard_deviation(self) -> float:
        """Return the empirical standard deviation of the estimator.

        References:
            [REF-MORRIS-WHITE-CROWTHER-2019, Table 6] (empirical standard
            error). No further equation is evaluated.

        """
        return math.sqrt(self.variance)

    @property
    def mean_standard_error(self) -> float:
        """Return the Monte Carlo standard error of the replicate mean.

        Returns:
            ``sqrt(variance / R)`` ([EQ-MLAB-MCSE-001]).

        References:
            [REF-MORRIS-WHITE-CROWTHER-2019, Table 6], [EQ-MLAB-MCSE-001].

        """
        return math.sqrt(self.variance / self.count)

    @property
    def variance_standard_error(self) -> float:
        """Return the Monte Carlo standard error of the sample variance.

        Returns:
            ``sqrt((m4 - (R - 3) / (R - 1) * s^4) / R)`` with the sample
            fourth central moment ``m4``; no normality is assumed
            ([EQ-MLAB-MCSE-002]). Zero when the bracket is negative through
            sampling noise.

        References:
            [REF-CASELLA-BERGER-2002, Exercise 5.8], [EQ-MLAB-MCSE-002].

        """
        size = self.count
        bracket = (
            self.fourth_central_moment - (size - 3) / (size - 1) * self.variance**2
        )
        return math.sqrt(max(bracket, 0.0) / size)

    @property
    def work_normalized_variance(self) -> float:
        """Return variance times mean cost.

        The product is the asymptotic efficiency measure of Glynn and Whitt:
        under a fixed computing budget ``c`` the mean of independent
        replicates has variance close to ``work_normalized_variance / c``, so
        the estimator with the smaller product is the more efficient one
        ([EQ-MLAB-WNV-001]).

        References:
            [REF-GLASSERMAN-2003, Section 1.1.3], [EQ-MLAB-WNV-001].

        """
        return self.variance * self.mean_cost


def summarize_draws(name: str, draws: Sequence[Draw]) -> ReplicateSummary:
    """Summarize replicate draws by their sample moments.

    Args:
        name: Estimator label.
        draws: At least two replicate draws.

    Returns:
        Mean, unbiased variance, mean cost, and fourth central moment.

    Raises:
        ValueError: If fewer than two draws are given.

    References:
        [REF-MORRIS-WHITE-CROWTHER-2019, Table 6], [EQ-MLAB-MCSE-001].

    """
    size = len(draws)
    if size < 2:
        raise ValueError("at least two replicate draws are required")
    values = [draw.value for draw in draws]
    mean = math.fsum(values) / size
    deviations = [value - mean for value in values]
    variance = math.fsum(d * d for d in deviations) / (size - 1)
    fourth = math.fsum(d**4 for d in deviations) / size
    mean_cost = math.fsum(draw.cost for draw in draws) / size
    return ReplicateSummary(name, size, mean, variance, mean_cost, fourth)


def _log_work_normalized_variance_error(draws: Sequence[Draw]) -> float:
    """Return the standard error of ``log(s^2 * mean cost)`` of replicates.

    Delta method for ``log WNV = log s^2 + log c_bar``: with
    ``a_r = (Z_r - Z_bar)^2 / m_2`` (``m_2`` the mean squared deviation) and
    ``b_r = c_r / c_bar``,

    ``Var[log WNV] ~ Var[s^2] / s^4 + (Var[b] + 2 Cov[a, b]) / R``,

    where the first term is [EQ-MLAB-MCSE-002] and the sample variance and
    covariance use the divisor ``R - 1``. For a constant cost the second
    term vanishes. The approximation needs finite fourth moments of the
    values and a finite variance of the costs.

    Raises:
        ValueError: If fewer than two draws are given or the variance or
            the mean cost is zero.

    References:
        [EQ-MLAB-WNV-001], [EQ-MLAB-MCSE-002],
        [REF-GLASSERMAN-2003, Section 1.1.3].

    """
    summary = summarize_draws("draws", draws)
    if summary.variance <= 0.0 or summary.mean_cost <= 0.0:
        raise ValueError("a positive variance and mean cost are required")
    size = summary.count
    second = summary.variance * (size - 1) / size
    scaled_squares = [(d.value - summary.mean) ** 2 / second - 1.0 for d in draws]
    scaled_costs = [d.cost / summary.mean_cost - 1.0 for d in draws]
    cost_variance = math.fsum(b * b for b in scaled_costs) / (size - 1)
    cross = math.fsum(
        a * b for a, b in zip(scaled_squares, scaled_costs, strict=True)
    ) / (size - 1)
    variance_term = (summary.variance_standard_error / summary.variance) ** 2
    total = variance_term + (cost_variance + 2.0 * cross) / size
    return math.sqrt(max(total, 0.0))


def summarize(
    estimator: Estimator, replicates: int, seed: int | RandomStream
) -> ReplicateSummary:
    """Replicate an estimator and summarize the draws.

    Args:
        estimator: Estimator to replicate.
        replicates: Number of independent replicates, at least two.
        seed: Root seed or stream; replicate ``i`` uses child
            ``("replicate", i)``.

    Returns:
        Sample moments of the replicates.

    Raises:
        ValueError: If ``replicates`` is below two.

    References:
        [REF-MORRIS-WHITE-CROWTHER-2019, Table 6], [EQ-MLAB-MCSE-001].

    """
    return summarize_draws(estimator.name, estimator.replicates(replicates, seed))


def estimator_from_function(
    name: str,
    function: Callable[[random.Random], float],
    *,
    cost: float = 1.0,
) -> Estimator:
    """Wrap a function of a random generator as a base estimator.

    Args:
        name: Estimator label.
        function: Maps a seeded :class:`random.Random` to one realized
            value. It must take all its randomness from that generator.
        cost: Work charged per call (default one unit).

    Returns:
        Base estimator with an empty lineage and constant cost.

    Raises:
        TypeError: If ``function`` is not callable.
        ValueError: If ``cost`` is negative or not finite.

    References:
        [REF-GLASSERMAN-2003, Section 1.1.3], [REF-MATSUMOTO-NISHIMURA-1998].

    """
    if not callable(function):
        raise TypeError("function must be callable")
    unit_cost = require_non_negative_finite_real(cost, name="cost")

    def sampler(stream: RandomStream) -> Draw:
        """Evaluate the wrapped function on the stream's generator.

        References:
            [REF-MATSUMOTO-NISHIMURA-1998]. No equation is evaluated.

        """
        return Draw(float(function(stream.generator())), unit_cost)

    return Estimator(name, sampler)


def sample_mean(estimator: Estimator, size: int) -> Estimator:
    """Derive the mean of ``size`` independent draws of an estimator.

    The derived estimator has the same expectation, variance divided by
    ``size``, and cost multiplied by ``size``; its work-normalized variance
    is unchanged ([EQ-MLAB-WNV-001]). Draw ``i`` uses the child stream
    ``("mean", i)``.

    Args:
        estimator: Estimator to average.
        size: Positive number of independent draws.

    Returns:
        Derived estimator ``(1/size) sum_i Z_i``.

    Raises:
        ValueError: If ``size`` is not positive.

    References:
        [REF-GLASSERMAN-2003, Section 1.1.3], [EQ-MC-MEAN-001].

    """
    count = require_non_negative_integer(size, name="size")
    if count == 0:
        raise ValueError("size must be positive")

    def sampler(stream: RandomStream) -> Draw:
        """Average independent draws on numbered child streams.

        References:
            [EQ-MC-MEAN-001], [REF-GLASSERMAN-2003, Section 1.1.3].

        """
        draws = [estimator.draw(stream.child("mean", i)) for i in range(count)]
        return Draw(
            math.fsum(d.value for d in draws) / count,
            math.fsum(d.cost for d in draws),
        )

    return estimator.derived(
        f"mean[{count}]({estimator.name})",
        sampler,
        f"sample mean of {count} independent draws",
    )
