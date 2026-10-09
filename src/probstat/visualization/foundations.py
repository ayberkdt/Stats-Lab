"""Plotting helpers that keep mathematical definitions visible."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

from probstat.distributions import (
    Bernoulli,
    Categorical,
    Geometric,
    NegativeBinomial,
    Normal,
)
from probstat.foundations import (
    ContinuousProbabilityLaw,
    CumulativeDistribution,
    ProbabilityMassFunction,
    SurvivalFunction,
)
from probstat.foundations.validation import (
    coerce_finite_sample,
    require_finite_real,
    require_non_negative_integer,
)
from probstat.monte_carlo import EfficiencyComparison, MonteCarloEstimate
from probstat.probability import (
    FiniteEvent,
    FiniteJointDistribution,
    FiniteProbabilityMeasure,
    FiniteRandomVariable,
)
from probstat.statistics import empirical_cdf
from probstat.uq import observed_order_of_accuracy, richardson_extrapolation


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


def plot_integer_pmf(
    distribution: ProbabilityMassFunction[int],
    *,
    lower: int,
    upper: int,
    ax: Any | None = None,
) -> Any:
    """Plot an integer-valued PMF over an explicit finite viewing window.

    The window is explicit because count distributions such as Poisson may have
    infinite support; a plot must not imply that omitted tail mass is zero.

    Args:
        distribution: Integer-valued object exposing ``pmf``.
        lower: Inclusive lower display bound.
        upper: Inclusive upper display bound, not below ``lower``.
        ax: Optional Matplotlib axes; a new axes is created when omitted.

    Returns:
        The populated Matplotlib axes.

    Raises:
        TypeError: If either bound is not an integer.
        ValueError: If ``upper < lower``.

    Equation:
        [EQ-DIST-BIN-001], [EQ-DIST-POIS-001].

    References:
        [REF-ROSS-2019, Chapter 4], [REF-MATPLOTLIB].

    """
    checked_lower = require_non_negative_integer(lower, name="lower")
    checked_upper = require_non_negative_integer(upper, name="upper")
    if checked_upper < checked_lower:
        raise ValueError("upper must not be below lower")
    support = tuple(range(checked_lower, checked_upper + 1))
    masses = tuple(distribution.pmf(value) for value in support)
    plt = _pyplot()
    axes = ax if ax is not None else plt.subplots()[1]
    axes.bar(support, masses, width=0.72, color="#4C78A8")
    axes.set(
        xlabel="Count k",
        ylabel="P(X = k)",
        ylim=(0.0, max(masses, default=0.0) * 1.15 or 1.0),
    )
    axes.set_title(
        f"{type(distribution).__name__} PMF on k = {checked_lower},...,{checked_upper}"
    )
    return axes


def plot_categorical_pmf(
    distribution: Categorical[Any], *, ax: Any | None = None
) -> Any:
    """Plot an unordered categorical PMF without imposing numeric meaning.

    Args:
        distribution: Validated categorical distribution.
        ax: Optional Matplotlib axes; a new axes is created when omitted.

    Returns:
        The populated Matplotlib axes.

    Equation:
        [EQ-DIST-CAT-001].

    References:
        [REF-ROSS-2019, Section 4.6], [REF-MATPLOTLIB].

    """
    labels = distribution.support.values
    masses = tuple(distribution.pmf(label) for label in labels)
    positions = tuple(range(len(labels)))
    plt = _pyplot()
    axes = ax if ax is not None else plt.subplots()[1]
    axes.bar(positions, masses, width=0.65, color="#72B7B2")
    axes.set(
        xticks=positions,
        xticklabels=tuple(str(label) for label in labels),
        xlabel="Category label",
        ylabel="Probability mass",
        ylim=(0.0, max(masses) * 1.18),
    )
    axes.set_title("Categorical PMF (labels are unordered)")
    for position, mass in zip(positions, masses, strict=True):
        axes.annotate(f"{mass:.3f}", (position, mass), ha="center", va="bottom")
    return axes


def plot_waiting_count_pmfs(
    geometric: Geometric,
    negative_binomial: NegativeBinomial,
    *,
    upper: int = 20,
    ax: Any | None = None,
) -> Any:
    """Compare two failures-before-success PMFs on one integer window.

    Args:
        geometric: First-success waiting-count law.
        negative_binomial: Target-success waiting-count law.
        upper: Inclusive non-negative count-window endpoint.
        ax: Optional Matplotlib axes; a new axes is created when omitted.

    Returns:
        The populated Matplotlib axes.

    Raises:
        TypeError: If ``upper`` is not an integer.
        ValueError: If ``upper`` is negative.

    Equation:
        [EQ-DIST-GEOM-001], [EQ-DIST-NBIN-001].

    References:
        [REF-ROSS-2019, Section 4.5], [REF-MATPLOTLIB].

    """
    checked_upper = require_non_negative_integer(upper, name="upper")
    counts = tuple(range(checked_upper + 1))
    geometric_masses = tuple(geometric.pmf(k) for k in counts)
    negative_masses = tuple(negative_binomial.pmf(k) for k in counts)
    plt = _pyplot()
    axes = ax if ax is not None else plt.subplots()[1]
    axes.plot(
        counts,
        geometric_masses,
        marker="o",
        label=f"Geometric(p={geometric.p:g})",
        color="#4C78A8",
    )
    axes.plot(
        counts,
        negative_masses,
        marker="s",
        label=(
            "NegativeBinomial"
            f"(r={negative_binomial.r}, p={negative_binomial.p:g})"
        ),
        color="#F58518",
    )
    axes.set(
        xlabel="Failures k before target success",
        ylabel="P(X = k)",
        xlim=(-0.25, checked_upper + 0.25),
        ylim=(0.0, 1.1 * max((*geometric_masses, *negative_masses))),
    )
    axes.set_title("Waiting-count PMFs (zero-based convention)")
    axes.legend()
    return axes


def plot_continuous_functions(
    distribution: ContinuousProbabilityLaw,
    *,
    lower: float,
    upper: float,
    points: int = 401,
    ax: Any | None = None,
) -> Any:
    """Plot PDF, CDF, and survival function on coordinated axes.

    Density uses the left axis because it can exceed one; CDF and survival use
    the right probability axis. This prevents density from being mislabeled as
    probability while keeping their relationship visible.

    Args:
        distribution: Continuous law that also exposes a direct survival method.
        lower: Finite lower viewing bound.
        upper: Finite upper viewing bound, strictly greater than ``lower``.
        points: Number of equally spaced evaluation points, at least two.
        ax: Optional primary Matplotlib axes; a new axes is created when omitted.

    Returns:
        The primary density axes; its figure also contains a probability axis.

    Raises:
        TypeError: If bounds or points have invalid types, or survival is absent.
        ValueError: If bounds are invalid or fewer than two points are requested.

    Equation:
        [EQ-DIST-UNIF-001], [EQ-DIST-UNIF-002], [EQ-DIST-EXP-001],
        [EQ-DIST-EXP-002], [EQ-DIST-NORM-001], [EQ-DIST-NORM-005].

    References:
        [REF-CASELLA-BERGER-2002, Chapter 3], [REF-MATPLOTLIB].

    """
    if not isinstance(distribution, SurvivalFunction):
        raise TypeError("distribution must expose a survival function")
    checked_lower = require_finite_real(lower, name="lower")
    checked_upper = require_finite_real(upper, name="upper")
    checked_points = require_non_negative_integer(points, name="points")
    if checked_upper <= checked_lower:
        raise ValueError("upper must be greater than lower")
    if checked_points < 2:
        raise ValueError("points must be at least two")
    step = (checked_upper - checked_lower) / (checked_points - 1)
    x_values = tuple(
        checked_lower + index * step for index in range(checked_points)
    )
    plt = _pyplot()
    axes = ax if ax is not None else plt.subplots()[1]
    probability_axes = axes.twinx()
    density_line = axes.plot(
        x_values,
        tuple(distribution.pdf(x) for x in x_values),
        color="#4C78A8",
        label="PDF",
    )[0]
    cdf_line = probability_axes.plot(
        x_values,
        tuple(distribution.cdf(x) for x in x_values),
        color="#54A24B",
        label="CDF",
    )[0]
    survival_line = probability_axes.plot(
        x_values,
        tuple(distribution.sf(x) for x in x_values),
        color="#E45756",
        linestyle="--",
        label="survival",
    )[0]
    axes.set(xlabel="x", ylabel="Density f(x)")
    probability_axes.set(ylabel="Probability", ylim=(0.0, 1.05))
    axes.set_title(f"{type(distribution).__name__}: density and tail functions")
    axes.legend(
        handles=(density_line, cdf_line, survival_line),
        labels=("PDF", "CDF", "survival"),
        loc="center right",
    )
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


def plot_random_variable_pmf(
    variable: FiniteRandomVariable[Any, Any], *, ax: Any | None = None
) -> Any:
    """Plot the induced PMF of a finite random variable.

    Args:
        variable: Finite random variable with hashable support values.
        ax: Optional Matplotlib axes; a new axes is created when omitted.

    Returns:
        The populated Matplotlib axes.

    Equation:
        [EQ-RV-001].

    References:
        [REF-ROSS-2019, Chapters 4-5], [REF-MATPLOTLIB].

    """
    plt = _pyplot()
    axes = ax if ax is not None else plt.subplots()[1]
    support = variable.support
    masses = tuple(variable.pmf(value) for value in support)
    positions = tuple(range(len(support)))
    axes.bar(positions, masses, width=0.65, color="#72B7B2")
    axes.set(
        xticks=positions,
        xticklabels=tuple(str(value) for value in support),
        xlabel=f"{variable.name} value",
        ylabel="Probability mass",
        ylim=(0.0, max(masses) * 1.18),
    )
    axes.set_title(f"Induced PMF of {variable.name}")
    for position, mass in zip(positions, masses, strict=True):
        axes.annotate(f"{mass:.3f}", (position, mass), ha="center", va="bottom")
    return axes


def plot_joint_pmf(
    joint: FiniteJointDistribution[Any, Any, Any], *, ax: Any | None = None
) -> Any:
    """Plot a finite joint PMF as an annotated probability-mass matrix.

    Args:
        joint: Joint law of two finite random variables.
        ax: Optional Matplotlib axes; a new axes is created when omitted.

    Returns:
        The populated Matplotlib axes.

    Equation:
        [EQ-RV-010].

    References:
        [REF-ROSS-2019, Chapter 6], [REF-MATPLOTLIB].

    """
    plt = _pyplot()
    axes = ax if ax is not None else plt.subplots()[1]
    first_support = joint.first.support
    second_support = joint.second.support
    matrix = tuple(
        tuple(joint.joint_pmf(first, second) for first in first_support)
        for second in second_support
    )
    maximum_mass = max(max(row) for row in matrix)
    image = axes.imshow(matrix, cmap="Blues", vmin=0.0)
    axes.set(
        xticks=tuple(range(len(first_support))),
        xticklabels=tuple(str(value) for value in first_support),
        yticks=tuple(range(len(second_support))),
        yticklabels=tuple(str(value) for value in second_support),
        xlabel=joint.first.name,
        ylabel=joint.second.name,
    )
    axes.set_title(f"Joint PMF of {joint.first.name} and {joint.second.name}")
    for row, second in enumerate(second_support):
        for column, first in enumerate(first_support):
            mass = joint.joint_pmf(first, second)
            text_color = "white" if mass > 0.5 * maximum_mass else "#1F1F1F"
            axes.text(
                column,
                row,
                f"{mass:.3f}",
                ha="center",
                va="center",
                color=text_color,
            )
    axes.figure.colorbar(image, ax=axes, label="Probability mass")
    return axes


def plot_characteristic_function(
    variable: FiniteRandomVariable[Any, Any],
    *,
    t_min: float = -3.141592653589793,
    t_max: float = 3.141592653589793,
    points: int = 401,
    ax: Any | None = None,
) -> Any:
    """Plot real and imaginary parts of a finite characteristic function.

    Args:
        variable: Finite real-valued random variable.
        t_min: Finite lower frequency bound.
        t_max: Finite upper frequency bound, strictly greater than ``t_min``.
        points: Number of equally spaced frequencies, at least two.
        ax: Optional Matplotlib axes; a new axes is created when omitted.

    Returns:
        The populated Matplotlib axes.

    Raises:
        TypeError: If bounds, points, or mapped values have invalid types.
        ValueError: If bounds, points, or mapped values have invalid domains.
        OverflowError: If a characteristic-function phase cannot be represented.

    Equation:
        [EQ-GF-003].

    References:
        [REF-ROSS-2019, Section 6.5], [REF-MATPLOTLIB].

    """
    lower = require_finite_real(t_min, name="t_min")
    upper = require_finite_real(t_max, name="t_max")
    checked_points = require_non_negative_integer(points, name="points")
    if upper <= lower:
        raise ValueError("t_max must be greater than t_min")
    if checked_points < 2:
        raise ValueError("points must be at least two")
    step = (upper - lower) / (checked_points - 1)
    frequencies = tuple(lower + index * step for index in range(checked_points))
    values = tuple(variable.characteristic_function(t) for t in frequencies)

    plt = _pyplot()
    axes = ax if ax is not None else plt.subplots()[1]
    axes.plot(
        frequencies,
        tuple(value.real for value in values),
        label="real part",
        color="#4C78A8",
    )
    axes.plot(
        frequencies,
        tuple(value.imag for value in values),
        label="imaginary part",
        color="#E45756",
    )
    axes.axhline(0.0, color="#777777", linewidth=0.8)
    axes.set(
        xlabel="Frequency t",
        ylabel="Characteristic-function component",
        ylim=(-1.05, 1.05),
    )
    axes.set_title(f"Characteristic function of {variable.name}")
    axes.legend()
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


def plot_empirical_cdf(values: Iterable[float], *, ax: Any | None = None) -> Any:
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


def plot_fitted_cdf(
    values: Iterable[float],
    distribution: CumulativeDistribution,
    *,
    points: int = 401,
    ax: Any | None = None,
) -> Any:
    """Compare an empirical CDF with a fitted or specified model CDF.

    The plot is diagnostic rather than proof of fit. Parameter estimation from
    the same observations changes the calibration of formal goodness-of-fit
    statistics and must be reported separately.

    Args:
        values: Non-empty finite real observations.
        distribution: Object exposing a cumulative distribution function.
        points: Number of model-curve evaluation points, at least two.
        ax: Optional Matplotlib axes; a new axes is created when omitted.

    Returns:
        Axes containing the empirical step function and model CDF.

    Raises:
        TypeError: If observations, distribution, or points have invalid types.
        ValueError: If observations are invalid or fewer than two points are used.

    Equation:
        [EQ-STAT-005], [EQ-INF-GOF-KS-001].

    References:
        [REF-NIST-GOF], [REF-MATPLOTLIB].

    """
    sample = tuple(sorted(coerce_finite_sample(values)))
    if not isinstance(distribution, CumulativeDistribution):
        raise TypeError("distribution must expose cdf(x)")
    checked_points = require_non_negative_integer(points, name="points")
    if checked_points < 2:
        raise ValueError("points must be at least two")
    span = sample[-1] - sample[0]
    padding = 0.05 * span if span > 0.0 else max(1.0, abs(sample[0]) * 0.05)
    lower = sample[0] - padding
    upper = sample[-1] + padding
    step = (upper - lower) / (checked_points - 1)
    x_values = tuple(lower + index * step for index in range(checked_points))
    unique = tuple(dict.fromkeys(sample))
    empirical = tuple(empirical_cdf(sample, value) for value in unique)
    model = tuple(distribution.cdf(value) for value in x_values)
    plt = _pyplot()
    axes = ax if ax is not None else plt.subplots()[1]
    axes.step(
        unique,
        empirical,
        where="post",
        color="#4C78A8",
        label=f"empirical CDF (n={len(sample)})",
    )
    axes.plot(x_values, model, color="#E45756", label="model CDF")
    axes.set(xlabel="x", ylabel="Cumulative probability", ylim=(0.0, 1.05))
    axes.set_title(f"Empirical versus {type(distribution).__name__} CDF")
    axes.legend()
    return axes


def plot_refinement_convergence(
    coarse: float,
    medium: float,
    fine: float,
    *,
    refinement_ratio: float = 2.0,
    ax: Any | None = None,
) -> Any:
    """Visualize three-level convergence against Richardson extrapolation.

    Args:
        coarse: Quantity of interest on the coarsest resolution.
        medium: Quantity after one uniform refinement.
        fine: Quantity after a second uniform refinement.
        refinement_ratio: Common resolution ratio greater than one.
        ax: Optional Matplotlib axes; a new axes is created when omitted.

    Returns:
        Log-log error axes annotated with the observed convergence order.

    Raises:
        TypeError: If any refinement argument is not real.
        ValueError: If refinement assumptions needed by the estimators fail.
        OverflowError: If extrapolation is not representable.

    Equation:
        [EQ-VV-001], [EQ-VV-002].

    References:
        [REF-ROACHE-1998, Chapters 4-5], [REF-MATPLOTLIB].

    """
    order = observed_order_of_accuracy(
        coarse,
        medium,
        fine,
        refinement_ratio=refinement_ratio,
    )
    extrapolated = richardson_extrapolation(
        medium,
        fine,
        refinement_ratio=refinement_ratio,
        order=order,
    )
    ratio = require_finite_real(refinement_ratio, name="refinement_ratio")
    step_sizes = (1.0, 1.0 / ratio, 1.0 / (ratio * ratio))
    estimates = (coarse, medium, fine)
    errors = tuple(abs(value - extrapolated) for value in estimates)
    plt = _pyplot()
    axes = ax if ax is not None else plt.subplots()[1]
    axes.loglog(
        step_sizes,
        errors,
        marker="o",
        color="#4C78A8",
        label="absolute error to extrapolated value",
    )
    axes.invert_xaxis()
    axes.set(
        xlabel="Relative step size h",
        ylabel="|QoI(h) - extrapolated QoI|",
    )
    axes.set_title(f"Uniform refinement convergence (observed p = {order:.3g})")
    axes.legend()
    return axes


def plot_monte_carlo_convergence(
    estimates: Iterable[MonteCarloEstimate],
    *,
    reference: float | None = None,
    ax: Any | None = None,
) -> Any:
    """Plot nested Monte Carlo estimates and their reported confidence bands.

    Args:
        estimates: Non-empty sequence with strictly increasing sample sizes.
        reference: Optional finite analytic/reference value.
        ax: Optional Matplotlib axes; a new axes is created when omitted.

    Returns:
        Axes with log-sample-size estimate path and confidence band.

    Raises:
        TypeError: If an item is not a Monte Carlo estimate.
        ValueError: If input is empty, sizes do not increase, or reference is
            non-finite.

    Equation:
        [EQ-MC-MEAN-001], [EQ-MC-SE-001], [EQ-MC-ROOTN-001].

    References:
        [REF-ROBERT-CASELLA-2004, Chapters 3-4], [REF-MATPLOTLIB].

    """
    trace = tuple(estimates)
    if not trace:
        raise ValueError("estimates must not be empty")
    for index, estimate in enumerate(trace):
        if not isinstance(estimate, MonteCarloEstimate):
            raise TypeError(f"estimates[{index}] must be a MonteCarloEstimate")
        if index and estimate.sample_size <= trace[index - 1].sample_size:
            raise ValueError("estimate sample sizes must be strictly increasing")
    checked_reference = (
        None
        if reference is None
        else require_finite_real(reference, name="reference")
    )
    sample_sizes = tuple(estimate.sample_size for estimate in trace)
    centers = tuple(estimate.estimate for estimate in trace)
    lowers = tuple(estimate.confidence_interval.lower for estimate in trace)
    uppers = tuple(estimate.confidence_interval.upper for estimate in trace)
    plt = _pyplot()
    axes = ax if ax is not None else plt.subplots()[1]
    axes.plot(
        sample_sizes,
        centers,
        marker="o",
        color="#4C78A8",
        label="Monte Carlo estimate",
    )
    axes.fill_between(
        sample_sizes,
        lowers,
        uppers,
        color="#4C78A8",
        alpha=0.2,
        label="reported confidence interval",
    )
    if checked_reference is not None:
        axes.axhline(
            checked_reference,
            color="#E45756",
            linestyle="--",
            label="reference",
        )
    axes.set_xscale("log")
    axes.set(xlabel="Sample size N", ylabel="Estimate")
    axes.set_title("Monte Carlo convergence and sampling uncertainty")
    axes.legend()
    return axes


def plot_efficiency_comparison(
    comparisons: Iterable[EfficiencyComparison],
    *,
    ax: Any | None = None,
) -> Any:
    """Plot cost-normalized efficiency gains against a common baseline.

    Args:
        comparisons: Non-empty comparisons sharing the same baseline label.
        ax: Optional Matplotlib axes; a new axes is created when omitted.

    Returns:
        Axes whose bars show variance-cost gain; the horizontal value one marks
        parity with the baseline. Infinite gains are rejected because no finite
        logarithmic bar can represent a zero-variance candidate.

    Raises:
        TypeError: If an item is not an efficiency comparison.
        ValueError: If input is empty, baselines differ, or a gain is non-finite.

    Equation:
        [EQ-MC-EFF-001].

    References:
        [REF-OWEN-2013, Chapter 2], [REF-MATPLOTLIB].

    """
    records = tuple(comparisons)
    if not records:
        raise ValueError("comparisons must not be empty")
    for index, comparison in enumerate(records):
        if not isinstance(comparison, EfficiencyComparison):
            raise TypeError(
                f"comparisons[{index}] must be an EfficiencyComparison"
            )
        if comparison.baseline_label != records[0].baseline_label:
            raise ValueError("all comparisons must share one baseline label")
        if not math.isfinite(comparison.efficiency_gain):
            raise ValueError("efficiency gains must be finite for plotting")
    labels = tuple(record.candidate_label for record in records)
    gains = tuple(record.efficiency_gain for record in records)
    plt = _pyplot()
    axes = ax if ax is not None else plt.subplots()[1]
    colors = tuple("#54A24B" if gain >= 1.0 else "#E45756" for gain in gains)
    axes.bar(labels, gains, color=colors)
    axes.axhline(1.0, color="#4C78A8", linestyle="--", label="baseline parity")
    axes.set_yscale("log")
    axes.set(
        xlabel="Candidate strategy",
        ylabel="Variance-cost efficiency gain",
    )
    axes.set_title(f"Efficiency relative to {records[0].baseline_label}")
    axes.tick_params(axis="x", rotation=20)
    axes.legend()
    return axes

