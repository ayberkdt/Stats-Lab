"""Plotting helpers that keep mathematical definitions visible."""

from __future__ import annotations

from collections.abc import Iterable
from numbers import Real
from typing import Any

from probstat.distributions import Bernoulli, Normal
from probstat.foundations.validation import coerce_finite_sample
from probstat.probability import FiniteEvent, FiniteProbabilityMeasure
from probstat.statistics import empirical_cdf


def _pyplot() -> Any:
    """Import and return Matplotlib's plotting interface on demand.

    Returns:
        The ``matplotlib.pyplot`` module.

    Raises:
        ImportError: If the optional visualization dependency is not installed.

    References:
        [REF-MATPLOTLIB]. This dependency adapter evaluates no equation.

    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ImportError(
            "Visualization requires `pip install -e .[visualization]`"
        ) from error
    return plt


def plot_bernoulli_pmf(distribution: Bernoulli, *, ax: Any | None = None) -> Any:
    """Plot and annotate the two masses of a Bernoulli distribution.

    Args:
        distribution: Validated Bernoulli model.
        ax: Optional Matplotlib axes; a new axes is created when omitted.

    Returns:
        The populated Matplotlib axes.

    Equation:
        [EQ-DIST-BER-001].

    References:
        [REF-ROSS-2019, Section 4.2], [REF-MATPLOTLIB].

    """
    plt = _pyplot()
    axes = ax if ax is not None else plt.subplots()[1]
    outcomes = (0, 1)
    masses = tuple(distribution.pmf(value) for value in outcomes)
    axes.bar(outcomes, masses, width=0.45, color=("#4C78A8", "#F58518"))
    axes.set(xticks=outcomes, ylim=(0.0, 1.05), xlabel="Outcome k", ylabel="P(X = k)")
    axes.set_title(f"Bernoulli PMF (p = {distribution.p:.3g})")
    for outcome, mass in zip(outcomes, masses, strict=True):
        axes.annotate(f"{mass:.3f}", (outcome, mass), ha="center", va="bottom")
    return axes


def plot_finite_measure(
    measure: FiniteProbabilityMeasure[Any],
    *,
    highlight: FiniteEvent[Any] | None = None,
    ax: Any | None = None,
) -> Any:
    """Plot outcome masses and optionally highlight one finite event.

    Args:
        measure: Validated finite probability measure.
        highlight: Optional event from the measure's sample space. Highlighted
            outcomes use a contrasting color and the title reports ``P(A)``.
        ax: Optional Matplotlib axes; a new axes is created when omitted.

    Returns:
        The populated Matplotlib axes.

    Raises:
        TypeError: If ``highlight`` is not a finite event.
        ValueError: If ``highlight`` belongs to another sample space.

    Equation:
        [EQ-PROB-002].

    References:
        [REF-ROSS-2019, Chapter 2], [REF-MATPLOTLIB].

    """
    event_probability: float | None = None
    if highlight is not None:
        event_probability = measure.probability(highlight)
    plt = _pyplot()
    axes = ax if ax is not None else plt.subplots()[1]
    outcomes = tuple(measure.space)
    masses = tuple(measure.weights[outcome] for outcome in outcomes)
    colors = tuple(
        "#F58518" if highlight is not None and outcome in highlight else "#4C78A8"
        for outcome in outcomes
    )
    positions = tuple(range(len(outcomes)))
    axes.bar(positions, masses, color=colors, width=0.65)
    axes.set(
        xticks=positions,
        xticklabels=tuple(str(outcome) for outcome in outcomes),
        xlabel="Outcome",
        ylabel="Probability mass",
        ylim=(0.0, max(masses) * 1.18),
    )
    title = "Finite probability measure"
    if event_probability is not None:
        title += f" — highlighted event P(A) = {event_probability:.3f}"
    axes.set_title(title)
    for position, mass in zip(positions, masses, strict=True):
        axes.annotate(f"{mass:.3f}", (position, mass), ha="center", va="bottom")
    return axes


def plot_normal_pdf(
    distribution: Normal,
    *,
    standard_deviations: float = 4.0,
    points: int = 401,
    ax: Any | None = None,
) -> Any:
    """Plot a normal density over a symmetric interval around its mean.

    Args:
        distribution: Validated normal model.
        standard_deviations: Positive half-width in units of ``sigma``.
        points: Number of equally spaced evaluation points, at least two.
        ax: Optional Matplotlib axes; a new axes is created when omitted.

    Returns:
        The populated Matplotlib axes.

    Raises:
        ValueError: If ``standard_deviations <= 0`` or ``points < 2``.

    Equation:
        [EQ-DIST-NORM-001].

    References:
        [REF-CASELLA-BERGER-2002, Section 3.3], [REF-MATPLOTLIB].

    """
    if standard_deviations <= 0.0:
        raise ValueError("standard_deviations must be positive")
    if points < 2:
        raise ValueError("points must be at least two")
    plt = _pyplot()
    axes = ax if ax is not None else plt.subplots()[1]
    lower = distribution.mu - standard_deviations * distribution.sigma
    step = 2.0 * standard_deviations * distribution.sigma / (points - 1)
    x_values = tuple(lower + index * step for index in range(points))
    axes.plot(x_values, tuple(distribution.pdf(x) for x in x_values), color="#4C78A8")
    axes.axvline(distribution.mu, color="#E45756", linestyle="--", label="mean")
    axes.set(xlabel="x", ylabel="density f(x)")
    axes.set_title(
        f"Normal PDF (mu = {distribution.mu:g}, sigma = {distribution.sigma:g})"
    )
    axes.legend()
    return axes


def plot_empirical_cdf(values: Iterable[Real], *, ax: Any | None = None) -> Any:
    """Plot the right-continuous empirical CDF of a finite sample.

    Args:
        values: Non-empty finite real observations.
        ax: Optional Matplotlib axes; a new axes is created when omitted.

    Returns:
        The populated Matplotlib axes.

    Raises:
        TypeError: If an observation is not real.
        ValueError: If the sample is empty or contains a non-finite value.

    Equation:
        [EQ-STAT-005].

    References:
        [REF-WASSERMAN-2004, Section 7.3], [REF-MATPLOTLIB].

    """
    sample = tuple(sorted(coerce_finite_sample(values)))
    plt = _pyplot()
    axes = ax if ax is not None else plt.subplots()[1]
    y_values = tuple(empirical_cdf(sample, x) for x in sample)
    axes.step(sample, y_values, where="post", color="#54A24B")
    axes.scatter(sample, y_values, color="#54A24B", s=18)
    axes.set(xlabel="x", ylabel="F_n(x)", ylim=(0.0, 1.05))
    axes.set_title(f"Empirical CDF (n = {len(sample)})")
    return axes

