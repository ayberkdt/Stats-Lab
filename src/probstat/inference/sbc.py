"""Simulation-based calibration (SBC) of Bayesian inference algorithms.

Talts, Betancourt, Simpson, Vehtari and Gelman (2018): if
``theta~ ~ p(theta)``, ``y~ ~ p(y | theta~)`` and ``theta_1..L`` are draws
from the exact posterior ``p(theta | y~)``, then the rank statistic
``r = #{l : theta_l < theta~}`` of every scalar parameter is uniform on
``{0, ..., L}`` ([EQ-SBC-RANK-001]). A sampler (or an approximation) that is
biased, over- or under-dispersed produces a non-uniform rank histogram:
too-narrow posteriors give a U shape, too-wide ones a hump, a shift a slope.

:func:`simulation_based_calibration` runs the loop for user-supplied prior,
simulator and posterior sampler, and tests each parameter's ranks for
uniformity with a chi-square test on ``B`` bins (expected counts follow the
exact number of ranks in each bin, so ``B`` need not divide ``L + 1``).

Caveats stated by Talts et al.: the posterior draws should be (nearly)
independent, so thin Markov chains to about their effective sample size
first; autocorrelated draws make the histogram look over-dispersed even for a
correct sampler. Passing the test is necessary, not sufficient, for a correct
sampler: SBC averages over the prior and can miss errors confined to
low-prior regions.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from probstat.distributions import ChiSquare
from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)


@dataclass(frozen=True, slots=True)
class SBCResult:
    """Rank statistics and uniformity tests of a simulation-based calibration.

    Args:
        names: Parameter names.
        ranks: Per parameter, one rank in ``{0, ..., draws}`` per replicate.
        draws: Posterior draws ``L`` per replicate.
        bins: Histogram bins ``B`` of the chi-square test.
        histograms: Per parameter, the counts in each bin.
        expected: Expected count of each bin under uniform ranks.
        chi_square: Pearson statistic per parameter.
        p_values: Upper-tail chi-square probability with ``B - 1`` degrees of
            freedom per parameter.

    References:
        [REF-TALTS-ETAL-2018], [REF-AGRESTI-2013, Chapter 1].

    """

    names: tuple[str, ...]
    ranks: tuple[tuple[int, ...], ...]
    draws: int
    bins: int
    histograms: tuple[tuple[int, ...], ...]
    expected: tuple[float, ...]
    chi_square: tuple[float, ...]
    p_values: tuple[float, ...]

    @property
    def replicates(self) -> int:
        """Return the number of simulated data sets.

        References:
            [REF-TALTS-ETAL-2018]. No equation is evaluated.

        """
        return len(self.ranks[0]) if self.ranks else 0

    def min_p_value(self) -> float:
        """Return the smallest per-parameter p-value.

        References:
            [REF-TALTS-ETAL-2018]. No equation is evaluated.

        """
        return min(self.p_values)

    def calibrated(self, level: float = 0.01) -> bool:
        """Return whether no parameter rejects uniformity at a Bonferroni level.

        Args:
            level: Family-wise significance level; each of the ``k``
                parameters is tested at ``level / k``.

        Returns:
            ``True`` when every p-value exceeds ``level / k``.

        Raises:
            ValueError: If ``level`` is not in ``(0, 1)``.

        References:
            [REF-TALTS-ETAL-2018]. No equation is evaluated.

        """
        checked = require_finite_real(level, name="level")
        if not 0.0 < checked < 1.0:
            raise ValueError("level must lie in (0, 1)")
        return all(p > checked / len(self.p_values) for p in self.p_values)


def rank_statistic(
    truth: float, draws: Sequence[float], *, tie_break: float = 0.5
) -> int:
    """Return ``#{l : draw_l < truth}`` plus a share of exact ties.

    Ties (possible for discrete parameters or rounded output) are broken
    randomly: with ``t`` draws equal to ``truth`` the rank is increased by
    ``floor(tie_break (t + 1))`` for ``tie_break ~ Uniform(0, 1)``, which
    keeps the rank uniform when ties are exchangeable.

    Args:
        truth: Prior draw ``theta~``.
        draws: Posterior draws of the same scalar parameter.
        tie_break: Number in ``[0, 1)`` used only when ties occur.

    Returns:
        Rank in ``{0, ..., len(draws)}``.

    Raises:
        ValueError: If a value is not finite or ``tie_break`` is outside
            ``[0, 1)``.

    Equation:
        [EQ-SBC-RANK-001].

    References:
        [REF-TALTS-ETAL-2018].

    """
    value = require_finite_real(truth, name="truth")
    share = require_finite_real(tie_break, name="tie_break")
    if not 0.0 <= share < 1.0:
        raise ValueError("tie_break must lie in [0, 1)")
    below = 0
    ties = 0
    for index, draw in enumerate(draws):
        checked = require_finite_real(draw, name=f"draws[{index}]")
        if checked < value:
            below += 1
        elif checked == value:
            ties += 1
    return below + math.floor(share * (ties + 1))


def _default_bins(draws: int, replicates: int) -> int:
    """Return ``min(20, L + 1, max(2, N // 5))`` histogram bins.

    At least five expected counts per bin keep the chi-square approximation
    reasonable.

    References:
        [REF-TALTS-ETAL-2018]. No equation is evaluated.

    """
    return max(2, min(20, draws + 1, replicates // 5))


def simulation_based_calibration(
    prior_sampler: Callable[..., Sequence[float]],
    simulator: Callable[..., object],
    posterior_sampler: Callable[[object, RandomStream], Sequence[Sequence[float]]],
    *,
    replicates: int,
    bins: int | None = None,
    seed: int | RandomStream | None = None,
    names: Sequence[str] | None = None,
) -> SBCResult:
    """Run simulation-based calibration and test the ranks for uniformity.

    For replicate ``n``: ``theta~ = prior_sampler(generator)``,
    ``y~ = simulator(theta~, generator)``, and
    ``draws = posterior_sampler(y~, stream)`` (``L`` vectors). For each
    coordinate the rank of ``theta~`` among the ``L`` draws is recorded
    (:func:`rank_statistic`). Ranks in ``{0, ..., L}`` are grouped into ``B``
    bins by ``b = floor(r B / (L + 1))``; Pearson's statistic against the
    exact expected counts has approximately ``B - 1`` degrees of freedom.

    Every replicate uses its own child streams (``"prior"``, ``"data"``,
    ``"posterior"``, ``"ties"``), so results are reproducible by ``seed`` and
    independent of the cost of the posterior sampler.

    Args:
        prior_sampler: ``generator -> theta~`` (a ``random.Random`` is passed).
        simulator: ``(theta~, generator) -> data``.
        posterior_sampler: ``(data, stream) -> draws``; the draws should be
            nearly independent (thin MCMC output to its ESS).
        replicates: Number of simulated data sets (at least two).
        bins: Histogram bins (default ``min(20, L + 1, max(2, N // 5))``).
        seed: Integer seed or stream.
        names: Parameter names (default ``theta0, theta1, ...``).

    Returns:
        Ranks, histograms, chi-square statistics and p-values.

    Raises:
        TypeError: If an argument that must be callable is not.
        ValueError: If the replicate count or bins are invalid, the number
            or dimension of posterior draws changes between replicates, or a
            dimension does not match the prior draw.

    Equation:
        [EQ-SBC-RANK-001], [EQ-INF-CHI2-GOF-001].

    References:
        [REF-TALTS-ETAL-2018], [REF-AGRESTI-2013, Chapter 1].

    """
    for label, function in (
        ("prior_sampler", prior_sampler),
        ("simulator", simulator),
        ("posterior_sampler", posterior_sampler),
    ):
        if not callable(function):
            raise TypeError(f"{label} must be callable")
    count = require_non_negative_integer(replicates, name="replicates")
    if count < 2:
        raise ValueError("replicates must be at least two")
    root = resolve_stream(seed)
    ranks: list[list[int]] = []
    draw_count: int | None = None
    dimension: int | None = None
    for index, stream in enumerate(root.spawn(count, label="sbc-replicate")):
        truth = tuple(
            require_finite_real(v, name="prior draw")
            for v in prior_sampler(stream.child("prior").generator())
        )
        if dimension is None:
            dimension = len(truth)
            if dimension == 0:
                raise ValueError("prior draws must be non-empty")
            ranks = [[] for _ in range(dimension)]
        elif len(truth) != dimension:
            raise ValueError("prior draws must keep one dimension")
        data = simulator(truth, stream.child("data").generator())
        draws = [tuple(d) for d in posterior_sampler(data, stream.child("posterior"))]
        if draw_count is None:
            draw_count = len(draws)
            if draw_count == 0:
                raise ValueError("posterior_sampler returned no draws")
        elif len(draws) != draw_count:
            raise ValueError(
                f"replicate {index}: posterior_sampler returned {len(draws)} draws, "
                f"expected {draw_count}"
            )
        if any(len(d) != dimension for d in draws):
            raise ValueError("posterior draws must match the prior dimension")
        ties = stream.child("ties").generator()
        for k in range(dimension):
            ranks[k].append(
                rank_statistic(truth[k], [d[k] for d in draws], tie_break=ties.random())
            )
    assert draw_count is not None and dimension is not None
    bin_count = (
        _default_bins(draw_count, count)
        if bins is None
        else require_non_negative_integer(bins, name="bins")
    )
    if not 2 <= bin_count <= draw_count + 1:
        raise ValueError("bins must lie in [2, L + 1]")
    width = [0] * bin_count
    for rank in range(draw_count + 1):
        width[rank * bin_count // (draw_count + 1)] += 1
    expected = tuple(count * w / (draw_count + 1) for w in width)
    law = ChiSquare(bin_count - 1)
    histograms: list[tuple[int, ...]] = []
    statistics: list[float] = []
    p_values: list[float] = []
    for column in ranks:
        histogram = [0] * bin_count
        for rank in column:
            histogram[rank * bin_count // (draw_count + 1)] += 1
        statistic = math.fsum(
            (observed - e) ** 2 / e
            for observed, e in zip(histogram, expected, strict=True)
        )
        histograms.append(tuple(histogram))
        statistics.append(statistic)
        p_values.append(law.sf(statistic))
    labels = (
        tuple(f"theta{i}" for i in range(dimension)) if names is None else tuple(names)
    )
    if len(labels) != dimension:
        raise ValueError("names must match the parameter dimension")
    return SBCResult(
        labels,
        tuple(tuple(column) for column in ranks),
        draw_count,
        bin_count,
        tuple(histograms),
        expected,
        tuple(statistics),
        tuple(p_values),
    )
