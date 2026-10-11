"""Optional explanatory plots built on stable domain APIs.

All plots share one theme (:mod:`probstat.visualization.theme`): semantic
colour roles, assertion titles, direct labels, and quiet axes. Use
``chart_theme(GRAPHITE)`` for dark output and ``save_figure`` to export.
"""

from .foundations import (
    plot_bernoulli_pmf,
    plot_categorical_pmf,
    plot_characteristic_function,
    plot_continuous_functions,
    plot_efficiency_comparison,
    plot_empirical_cdf,
    plot_finite_measure,
    plot_fitted_cdf,
    plot_integer_pmf,
    plot_joint_pmf,
    plot_monte_carlo_convergence,
    plot_normal_pdf,
    plot_random_variable_pmf,
    plot_refinement_convergence,
    plot_waiting_count_pmfs,
)
from .theme import (
    GRAPHITE,
    PORCELAIN,
    ChartPalette,
    active_palette,
    chart_theme,
    save_figure,
)

__all__ = [
    "GRAPHITE",
    "PORCELAIN",
    "ChartPalette",
    "active_palette",
    "chart_theme",
    "save_figure",
    "plot_bernoulli_pmf",
    "plot_categorical_pmf",
    "plot_characteristic_function",
    "plot_continuous_functions",
    "plot_empirical_cdf",
    "plot_efficiency_comparison",
    "plot_finite_measure",
    "plot_fitted_cdf",
    "plot_joint_pmf",
    "plot_monte_carlo_convergence",
    "plot_integer_pmf",
    "plot_normal_pdf",
    "plot_random_variable_pmf",
    "plot_refinement_convergence",
    "plot_waiting_count_pmfs",
]
