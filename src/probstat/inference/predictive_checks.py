"""Simulation-based posterior predictive checks for any posterior sampler.

Given posterior draws ``theta^(s)`` (from NUTS, Metropolis, ADVI, or exact
sampling), a simulator of replicated data ``y_rep ~ p(y | theta)``, and test
quantities, :func:`posterior_predictive_pvalues` returns the replicated test
quantities and the posterior predictive p-values

``p_B = Pr(T(y_rep, theta) >= T(y, theta) | y)``
``    ~= (1/S) sum_s 1[T(y_rep^(s), theta^(s)) >= T(y, theta^(s))]``

([EQ-PPC-001]; Rubin 1984; Gelman, Meng, and Stern 1996). Test quantities may
depend on the data only (``statistics``) or also on the parameters
(``discrepancies``, the realized discrepancies of Gelman, Meng, and Stern).
The Monte Carlo standard error of ``p_B`` uses the effective sample size of
the indicator chains when the draws come from an MCMC result, and the
binomial formula ``sqrt(p (1 - p) / S)`` for independent draws.

Interpretation
    A posterior predictive p-value near 0 or 1 flags an aspect of the data the
    model does not reproduce. It is **not** uniformly distributed under a
    correct model in general: because the data are used twice, it is
    typically concentrated near 0.5 for statistics that the posterior fits
    directly (e.g. the sample mean when the mean is a free parameter). It is
    close to uniform for statistics whose sampling law does not depend on the
    unknown parameters.

PIT values
    :func:`pit_values` computes the probability integral transform
    ``PIT_i = Pr(y_rep,i <= y_i | y)`` of each observation from replicated
    datasets ([EQ-PPC-002]). This simple version conditions on all the data
    (it is not leave-one-out), so its values are pulled toward 0.5 in the
    same way; :func:`pit_uniformity` reports a Kolmogorov-Smirnov distance
    from uniformity as a descriptive diagnostic only.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

from probstat.distributions import Uniform
from probstat.foundations import coerce_finite_sample
from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_non_negative_integer
from probstat.linalg import Vector, as_vector

from .goodness_of_fit import KolmogorovSmirnovResult, kolmogorov_smirnov_test
from .hamiltonian import HamiltonianResult
from .mcmc import MCMCResult, effective_sample_size

Simulator = Callable[[Vector, random.Random], Sequence[float]]
Statistic = Callable[[Vector], float]
Discrepancy = Callable[[Vector, Vector], float]
DrawSource = MCMCResult | HamiltonianResult | Sequence[Sequence[float]]


@dataclass(frozen=True, slots=True)
class PredictiveStatisticCheck:
    """Replicated values and tail probabilities of one test quantity.

    Args:
        name: Test-quantity label.
        observed: ``T(y, theta^(s))`` per draw (constant for data-only
            statistics).
        replicated: ``T(y_rep^(s), theta^(s))`` per draw.
        p_value: Upper-tail ``Pr(T_rep >= T_obs)`` (ties count as extreme).
        lower_p_value: Lower-tail ``Pr(T_rep <= T_obs)``.
        mcse: Monte Carlo standard error of ``p_value``.
        parameter_dependent: Whether the quantity is a realized discrepancy.

    References:
        [REF-GELMAN-MENG-STERN-1996], [REF-RUBIN-1984].

    """

    name: str
    observed: tuple[float, ...]
    replicated: tuple[float, ...]
    p_value: float
    lower_p_value: float
    mcse: float
    parameter_dependent: bool

    @property
    def two_sided_p_value(self) -> float:
        """Return ``min(1, 2 min(upper, lower))``.

        References:
            [REF-GELMAN-ETAL-2013, Chapter 6]. Equation [EQ-PPC-001].

        """
        return min(1.0, 2.0 * min(self.p_value, self.lower_p_value))


@dataclass(frozen=True, slots=True)
class PosteriorPredictivePValues:
    """Posterior predictive check of several test quantities.

    Args:
        checks: One record per test quantity, statistics first.
        draws: Number of posterior draws used.
        chains: Number of chains the draws came from (``1`` for plain draws).
        replicates: Replicated datasets (empty unless requested).

    References:
        [REF-GELMAN-MENG-STERN-1996], [REF-GELMAN-ETAL-2013, Chapter 6].

    """

    checks: tuple[PredictiveStatisticCheck, ...]
    draws: int
    chains: int
    replicates: tuple[Vector, ...]

    def __getitem__(self, name: str) -> PredictiveStatisticCheck:
        """Return the check named ``name``.

        Raises:
            KeyError: If no check has that name.

        References:
            Project API policy. No equation is evaluated.

        """
        for check in self.checks:
            if check.name == name:
                return check
        raise KeyError(name)

    def p_values(self) -> dict[str, float]:
        """Return ``{name: upper-tail p-value}``.

        References:
            [REF-GELMAN-MENG-STERN-1996]. Equation [EQ-PPC-001].

        """
        return {check.name: check.p_value for check in self.checks}


def _draw_chains(draws: DrawSource) -> tuple[tuple[tuple[Vector, ...], ...], bool]:
    """Return draws grouped by chain and whether they form Markov chains.

    References:
        Project API policy. No equation is evaluated.

    """
    if isinstance(draws, HamiltonianResult):
        draws = draws.mcmc
    if isinstance(draws, MCMCResult):
        return tuple(chain.samples for chain in draws.chains), True
    vectors = tuple(as_vector(draw, name="posterior draw") for draw in draws)
    if not vectors:
        raise ValueError("draws must be non-empty")
    return (vectors,), False


def _thinned(
    chains: tuple[tuple[Vector, ...], ...], max_draws: int | None
) -> tuple[tuple[Vector, ...], ...]:
    """Thin every chain by a common stride so at most ``max_draws`` remain.

    References:
        [REF-GELMAN-ETAL-2013, Chapter 11]. No equation is evaluated.

    """
    total = sum(len(chain) for chain in chains)
    if max_draws is None or total <= max_draws:
        return chains
    # Each chain keeps ceil(L_i / stride) draws, so the per-chain rounding can
    # overshoot total / stride; increase the stride until the cap holds.
    stride = math.ceil(total / max_draws)
    while sum(math.ceil(len(chain) / stride) for chain in chains) > max_draws:
        stride += 1
    return tuple(chain[::stride] for chain in chains)


def _tail_mcse(indicators: Sequence[Sequence[float]], markov: bool) -> float:
    """Return the Monte Carlo standard error of an indicator mean.

    References:
        [REF-VEHTARI-ETAL-2021], [REF-GEYER-1992]. Equation [EQ-MCMC-MCSE-001].

    """
    pooled = [value for chain in indicators for value in chain]
    count = len(pooled)
    p = math.fsum(pooled) / count
    variance = p * (1.0 - p)
    if variance == 0.0:
        return 0.0
    if markov and min(len(chain) for chain in indicators) >= 4:
        lengths = {len(chain) for chain in indicators}
        if len(lengths) == 1:
            ess = effective_sample_size(indicators)
            if math.isfinite(ess) and ess > 0.0:
                return math.sqrt(variance / ess)
    return math.sqrt(variance / count)


def posterior_predictive_pvalues(
    draws: DrawSource,
    simulate: Simulator,
    observed: Iterable[object],
    *,
    statistics: Mapping[str, Statistic] | None = None,
    discrepancies: Mapping[str, Discrepancy] | None = None,
    max_draws: int | None = None,
    keep_replicates: bool = False,
    seed: int | RandomStream | None = None,
) -> PosteriorPredictivePValues:
    """Run a simulation-based posterior predictive check.

    For every retained posterior draw ``theta^(s)`` one replicated dataset
    ``y_rep^(s) = simulate(theta^(s), generator)`` is drawn with a single
    generator from ``seed``, and every test quantity is evaluated on the
    observed and replicated data.

    Args:
        draws: :class:`MCMCResult`, :class:`HamiltonianResult`, or a sequence
            of parameter vectors (treated as independent draws).
        simulate: ``(theta, generator) -> y_rep`` of the observed length.
        observed: Observed data ``y``.
        statistics: ``{name: T(y)}`` data-only test statistics.
        discrepancies: ``{name: T(y, theta)}`` realized discrepancies.
        max_draws: Optional cap; chains are thinned by a common stride.
        keep_replicates: Whether to return the replicated datasets.
        seed: Integer seed or stream.

    Returns:
        Replicated test quantities, p-values, and their Monte Carlo errors.

    Raises:
        ValueError: If no test quantity is given, names repeat, or a
            replicate has the wrong length or non-finite values.

    Equation:
        [EQ-PPC-001], [EQ-MCMC-MCSE-001].

    References:
        [REF-RUBIN-1984], [REF-GELMAN-MENG-STERN-1996],
        [REF-GELMAN-ETAL-2013, Chapter 6].

    """
    if not callable(simulate):
        raise TypeError("simulate must be callable")
    data = coerce_finite_sample(observed, name="observed")
    stats = dict(statistics or {})
    discs = dict(discrepancies or {})
    if not stats and not discs:
        raise ValueError("provide at least one statistic or discrepancy")
    if set(stats) & set(discs):
        raise ValueError("statistic and discrepancy names must be distinct")
    cap = None
    if max_draws is not None:
        cap = require_non_negative_integer(max_draws, name="max_draws")
        if cap == 0:
            raise ValueError("max_draws must be positive")
    chains, markov = _draw_chains(draws)
    chains = _thinned(chains, cap)
    generator = resolve_stream(seed).generator()
    observed_data: Vector = tuple(data)
    observed_stats = {
        name: float(function(observed_data)) for name, function in stats.items()
    }
    names = list(stats) + list(discs)
    observed_values: dict[str, list[list[float]]] = {n: [] for n in names}
    replicated_values: dict[str, list[list[float]]] = {n: [] for n in names}
    replicates: list[Vector] = []
    for chain in chains:
        for name in names:
            observed_values[name].append([])
            replicated_values[name].append([])
        for theta in chain:
            replicate = tuple(float(v) for v in simulate(theta, generator))
            if len(replicate) != len(observed_data):
                raise ValueError("replicated data must match the observed length")
            if not all(math.isfinite(v) for v in replicate):
                raise ValueError("replicated data must be finite")
            if keep_replicates:
                replicates.append(replicate)
            for name, statistic in stats.items():
                observed_values[name][-1].append(observed_stats[name])
                replicated_values[name][-1].append(float(statistic(replicate)))
            for name, discrepancy in discs.items():
                observed_values[name][-1].append(
                    float(discrepancy(observed_data, theta))
                )
                replicated_values[name][-1].append(float(discrepancy(replicate, theta)))
    total = sum(len(chain) for chain in chains)
    checks: list[PredictiveStatisticCheck] = []
    for name in names:
        upper = [
            [1.0 if rep >= obs else 0.0 for rep, obs in zip(r, o, strict=True)]
            for r, o in zip(replicated_values[name], observed_values[name], strict=True)
        ]
        lower = [
            [1.0 if rep <= obs else 0.0 for rep, obs in zip(r, o, strict=True)]
            for r, o in zip(replicated_values[name], observed_values[name], strict=True)
        ]
        flat_obs = tuple(v for chain in observed_values[name] for v in chain)
        flat_rep = tuple(v for chain in replicated_values[name] for v in chain)
        if not all(math.isfinite(v) for v in flat_obs + flat_rep):
            raise ValueError(f"test quantity {name!r} returned a non-finite value")
        checks.append(
            PredictiveStatisticCheck(
                name,
                flat_obs,
                flat_rep,
                math.fsum(v for chain in upper for v in chain) / total,
                math.fsum(v for chain in lower for v in chain) / total,
                _tail_mcse(upper, markov),
                name in discs,
            )
        )
    return PosteriorPredictivePValues(
        tuple(checks), total, len(chains), tuple(replicates)
    )


def pit_values(
    observed: Iterable[object],
    replicates: Sequence[Sequence[float]],
    *,
    randomize_ties: bool = False,
    seed: int | RandomStream | None = None,
) -> tuple[float, ...]:
    """Return PIT values ``Pr(y_rep,i <= y_i | y)`` from replicated datasets.

    ``PIT_i = (#{y_rep,i < y_i} + c_i #{y_rep,i = y_i}) / S`` with
    ``c_i = 1/2`` (mid-ties) or ``c_i ~ Uniform(0, 1)`` when
    ``randomize_ties`` is true (the randomized PIT for discrete data).

    Args:
        observed: Observed data of length ``n``.
        replicates: ``S >= 1`` replicated datasets of length ``n``.
        randomize_ties: Use randomized instead of mid-tie counting.
        seed: Seed for the tie randomization.

    Returns:
        One PIT value per observation.

    Raises:
        ValueError: For empty or misaligned replicates.

    Equation:
        [EQ-PPC-002].

    References:
        [REF-GNEITING-BALABDAOUI-RAFTERY-2007], [REF-DAWID-1984].

    """
    data = coerce_finite_sample(observed, name="observed")
    reps = tuple(tuple(float(v) for v in replicate) for replicate in replicates)
    if not reps:
        raise ValueError("at least one replicated dataset is required")
    if any(len(replicate) != len(data) for replicate in reps):
        raise ValueError("replicates must match the observed length")
    generator = resolve_stream(seed).generator() if randomize_ties else None
    values: list[float] = []
    for index, value in enumerate(data):
        below = sum(1 for replicate in reps if replicate[index] < value)
        ties = sum(1 for replicate in reps if replicate[index] == value)
        share = generator.random() if generator is not None else 0.5
        values.append((below + share * ties) / len(reps))
    return tuple(values)


def pit_uniformity(values: Iterable[object]) -> KolmogorovSmirnovResult:
    """Return the KS distance of PIT values from ``Uniform(0, 1)``.

    The p-value assumes independent, exactly uniform PIT values; with
    non-LOO posterior PIT values it is descriptive only.

    Args:
        values: PIT values in ``[0, 1]``.

    Returns:
        Kolmogorov-Smirnov statistic and nominal p-value.

    Equation:
        [EQ-PPC-002], [EQ-INF-GOF-KS-001].

    References:
        [REF-GNEITING-BALABDAOUI-RAFTERY-2007], [REF-DAWID-1984].

    """
    sample = coerce_finite_sample(values, name="values")
    if any(v < 0.0 or v > 1.0 for v in sample):
        raise ValueError("PIT values must lie in [0, 1]")
    return kolmogorov_smirnov_test(sample, Uniform(0.0, 1.0))
