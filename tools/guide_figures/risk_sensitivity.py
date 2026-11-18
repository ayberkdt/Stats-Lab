"""Figures of the rare-event, guarantee, and sampling-based sensitivity sections.

The figures are drawn by the two laboratories that teach these methods,
``examples/risk_analysis_lab.py`` (four-branch limit state, Wilks bounds,
certification) and ``examples/flood_sensitivity_lab.py`` (flood model), so a
guide shows exactly the computation a reader can rerun. Each laboratory
caches its seeded computations, so the light and dark renderings share one
run and report the same facts.

References:
    [REF-GUYADER-HENGARTNER-MATZNER-LOBER-2011], [REF-BJERAGER-1988],
    [REF-CLOPPER-PEARSON-1934], [REF-WILKS-1941], [REF-HELTON-ETAL-2006],
    [REF-DA-VEIGA-2015], [REF-MATPLOTLIB].

"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from guide_figures import GuideFigure

ROOT = Path(__file__).resolve().parents[2]


def _lab(name: str) -> Any:
    """Import a laboratory script of ``examples/`` once and return it.

    References:
        [REF-PROBSTAT-REPORTING]. No equation is evaluated.

    """
    key = f"_guide_example_{name}"
    if key in sys.modules:
        return sys.modules[key]
    spec = importlib.util.spec_from_file_location(key, ROOT / "examples" / f"{name}.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load examples/{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[key] = module
    spec.loader.exec_module(module)
    return module


def draw_moving_particles() -> tuple[Any, dict[str, str]]:
    """Draw moving particles on the four-branch limit state (``N = 100``).

    References:
        [EQ-REL-MP-001], [EQ-REL-MP-002],
        [REF-GUYADER-HENGARTNER-MATZNER-LOBER-2011].

    """
    figure, facts = _lab("risk_analysis_lab").figure_moving_particles()
    return figure, facts


def draw_directional_rays() -> tuple[Any, dict[str, str]]:
    """Draw the rays and roots of directional simulation on the four branches.

    References:
        [EQ-REL-DIR-002], [REF-BJERAGER-1988].

    """
    figure, facts = _lab("risk_analysis_lab").figure_directional_rays()
    return figure, facts


def draw_certification() -> tuple[Any, dict[str, str]]:
    """Draw the certification probability against ``n alpha``.

    References:
        [EQ-REL-CERT-002], [REF-CLOPPER-PEARSON-1934].

    """
    figure, facts = _lab("risk_analysis_lab").figure_certification()
    return figure, facts


def draw_wilks_quantiles() -> tuple[Any, dict[str, str]]:
    """Draw the replicate distributions of the empirical quantile and Wilks bound.

    References:
        [EQ-PROP-WILKS-001], [REF-WILKS-1941].

    """
    figure, facts = _lab("risk_analysis_lab").figure_wilks_quantiles()
    return figure, facts


def draw_flood_top_decile() -> tuple[Any, dict[str, str]]:
    """Draw the flood-model overflow against each input, top decile highlighted.

    The laboratory also draws the cobweb plot of the same runs; its thousand
    antialiased polylines exceed the size bound of a guide image, so the
    guide shows the same selection as one scatterplot per input.

    References:
        [EQ-MOD-FLOOD-001], [REF-HELTON-ETAL-2006].

    """
    figure, facts = _lab("flood_sensitivity_lab").figure_top_decile_scatter()
    return figure, facts


def draw_flood_cost_measures() -> tuple[Any, dict[str, str]]:
    """Draw Taylor, SRC^2, Sobol, and HSIC measures of the flood-model cost.

    References:
        [EQ-SENS-TAYLOR-001], [EQ-SENS-SRC-001], [EQ-SENS-FIRST-001],
        [EQ-SENS-HSIC-001], [REF-DA-VEIGA-2015].

    """
    figure, facts = _lab("flood_sensitivity_lab").figure_measure_comparison("cost")
    return figure, facts


FIGURES: list[GuideFigure] = [
    GuideFigure(
        "advanced-reliability-moving-particles",
        "advanced-reliability",
        "Standard-normal map of the four-branch limit state with 100 initial "
        "particles, the final particles, and two intermediate level sets",
        draw_moving_particles,
    ),
    GuideFigure(
        "advanced-reliability-directional-rays",
        "advanced-reliability",
        "Standard-normal map of the four-branch limit state with 200 rays from "
        "the origin and the roots where they enter the failure domain",
        draw_directional_rays,
    ),
    GuideFigure(
        "reliability-certification-probability",
        "reliability-analysis",
        "Probability of certifying a failure probability against the number of "
        "runs times the acceptable risk, for three true probabilities and two rules",
        draw_certification,
    ),
    GuideFigure(
        "propagation-wilks-quantiles",
        "uncertainty-propagation",
        "Empirical CDFs over replicates of the empirical 0.95-quantile and of "
        "the Wilks upper bound, with the true quantile",
        draw_wilks_quantiles,
    ),
    GuideFigure(
        "sensitivity-flood-top-decile",
        "sensitivity-analysis",
        "Eight scatterplots of the flood-model overflow against each input for "
        "1,000 runs, with the top decile of the overflow highlighted",
        draw_flood_top_decile,
    ),
    GuideFigure(
        "sensitivity-flood-cost-measures",
        "sensitivity-analysis",
        "Four panels of Taylor importance factors, squared regression "
        "coefficients, first-order Sobol indices, and HSIC indices of the "
        "flood-model cost for eight inputs",
        draw_flood_cost_measures,
    ),
]
