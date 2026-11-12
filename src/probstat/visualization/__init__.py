"""Optional explanatory plots built on stable domain APIs.

All plots share one theme (:mod:`probstat.visualization.theme`): semantic
colour roles, assertion titles, direct labels, and quiet axes. Use
``chart_theme(GRAPHITE)`` for dark output and ``save_figure`` to export.

Main entry points:
    ``chart_theme``: context manager that applies the theme rcParams and a
        palette to the enclosed plotting code.
    ``active_palette``: palette of the innermost active theme.
    ``PORCELAIN``: light palette (default).
    ``GRAPHITE``: dark palette.
    ``save_figure``: export with the theme canvas, tight bounds, and an
        embedded text alternative.
    ``plot_monte_carlo_convergence``: Monte Carlo estimates and their
        reported intervals against sample size.

Assumptions:
    Matplotlib is an optional dependency (the ``visualization`` extra); the
    rest of the library runs without it. Plot functions take finished result
    objects and never refit a model or re-run an analysis.

Common pitfalls:
    - Without Matplotlib, ``chart_theme`` and the ``plot_*`` functions
      exported here raise ``ImportError`` with install advice;
      ``theme.matplotlib_available()`` checks first. HTML study reports
      still work and omit the figures.
    - ``chart_theme()`` with no argument is always ``PORCELAIN``. The
      ``PROBSTAT_CHART_THEME`` environment variable (``graphite`` or
      ``dark``) only sets the palette that library figures use outside an
      explicit theme, and it is read once, when
      ``probstat.visualization.theme`` is first imported.
    - ``theme.band`` caps the fill opacity at ``MAX_BAND_ALPHA = 0.18``; a
      larger ``alpha`` is silently lowered.
    - ``save_figure`` builds the stored text alternative from the title,
      subtitle, and axes (``theme.title_block`` upper-cases the first letter
      of the title); pass ``description=`` when the generated text is not
      enough. Parent directories of the path are created.

Guide: docs/foundations/visual-language.md

References:
    [REF-MATPLOTLIB], [REF-W3C-WCAG21-2018], [REF-PNG-SPEC-2003],
    [REF-FRITSCH-CARLSON-1980].

"""

from .foundation_plots import (
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
