"""Figures of the nonlinear Bayesian inversion guide.

One seeded computation on the synthetic heat-flux case of the guide's
section 9: the flux into a slab recovered from one thermocouple 6 mm deep,
with the data generated on a finer mesh than the inversion uses. The figure
draws the true triangular pulse, the Gaussian-process posterior mean with its
95 % pointwise band and the zeroth-order Tikhonov estimate chosen by the
discrepancy principle. The GCV choice is quoted, not drawn: its estimate is
about fifty times the size of the pulse. The facts are the numbers the
caption in the guide quotes. The slab is synthetic, not a real heat shield.

References:
    [REF-PROBSTAT-REPORTING], [REF-MATPLOTLIB], [REF-BECK-BLACKWELL-STCLAIR-1985],
    [REF-RASMUSSEN-WILLIAMS-2006].

"""

from __future__ import annotations

import functools
import math
from collections.abc import Sequence
from typing import Any

from probstat.inverse import (
    SlabFluxOperator,
    bayesian_flux_posterior,
    synthetic_ablator_slab,
    synthetic_ihcp_data,
    tikhonov_flux_estimate,
    triangular_pulse,
)
from probstat.inverse.parameter_selection import logarithmic_grid
from probstat.visualization import theme

from . import GuideFigure

WIDE = (7.6, 4.4)
DEPTH = 0.006  # m, sensor depth
INTERVALS = 30
INTERVAL_LENGTH = 2.0  # s


def _flux(t: float) -> float:
    """Return the true triangular heat pulse (W/m^2) at time ``t`` (s).

    References:
        [EQ-INV-IHCP-001]. No equation is evaluated.

    """
    return triangular_pulse(t, 1.0e5, 6.0, 24.0, 42.0)


def _relative_error(estimate: Sequence[float], truth: Sequence[float]) -> float:
    """Return ``||estimate - truth|| / ||truth||``.

    References:
        [EQ-INV-IHCP-009]. No equation is evaluated.

    """
    top = math.fsum((a - b) ** 2 for a, b in zip(estimate, truth, strict=True))
    return math.sqrt(top / math.fsum(b * b for b in truth))


@functools.cache
def _reconstruction() -> dict[str, Any]:
    """Invert the 6 mm case with GCV, the discrepancy rule and the GP posterior.

    The same synthetic case, seed and settings as the guide's section 9.

    References:
        [EQ-INV-IHCP-009], [EQ-INV-IHCP-010].

    """
    slab = synthetic_ablator_slab()
    op = SlabFluxOperator(
        slab, (DEPTH,), flux_intervals=INTERVALS, interval_length=INTERVAL_LENGTH
    )
    case = synthetic_ihcp_data(
        slab,
        [DEPTH],
        _flux,
        flux_intervals=INTERVALS,
        interval_length=INTERVAL_LENGTH,
        seed=3,
    )
    matrix = op.sensitivity_matrix()
    scale = math.sqrt(sum(v * v for row in matrix for v in row)) / case.noise_std
    grid = logarithmic_grid(1e-6 * scale, scale, 24)
    fits = {
        rule: tikhonov_flux_estimate(
            op,
            case.noisy,
            case.noise_std,
            order=0,
            criterion=rule,
            regularizations=grid,
        )
        for rule in ("gcv", "discrepancy")
    }
    posterior = bayesian_flux_posterior(
        op,
        case.noisy,
        case.noise_std,
        op.flux_times,
        amplitude=6.0e4,
        length_scale=12.0,
    )
    return {
        "times": list(op.flux_times),
        "truth": list(case.truth),
        "discrepancy": list(fits["discrepancy"].estimate),
        "gcv": list(fits["gcv"].estimate),
        "posterior": posterior,
    }


def _flux_figure() -> tuple[Any, dict[str, str]]:
    """Draw the 6 mm reconstruction against the true pulse.

    References:
        [EQ-INV-IHCP-009], [EQ-INV-IHCP-010], [REF-BECK-BLACKWELL-STCLAIR-1985].

    """
    data = _reconstruction()
    times = data["times"]
    truth = data["truth"]
    posterior = data["posterior"]
    kilo = 1.0e-3  # W/m^2 to kW/m^2
    bounds = zip(posterior.lower, posterior.upper, truth, strict=True)
    covered = sum(1 for low, high, value in bounds if low <= value <= high)
    facts = {
        "depth_mm": f"{DEPTH * 1e3:.0f}",
        "gp_error": f"{_relative_error(posterior.mean, truth):.3f}",
        "discrepancy_error": f"{_relative_error(data['discrepancy'], truth):.3f}",
        "gcv_error": f"{_relative_error(data['gcv'], truth):.1f}",
        "covered": f"{covered}",
        "points": f"{len(truth)}",
    }
    axes = theme.new_axes(figsize=WIDE)
    model = theme.role_colour("model")
    observed = theme.role_colour("observed")
    theme.band(
        axes,
        times,
        [v * kilo for v in posterior.lower],
        [v * kilo for v in posterior.upper],
        color=model,
    )
    theme.line(axes, times, [v * kilo for v in truth], entity="truth")
    theme.line(axes, times, [v * kilo for v in posterior.mean], color=model)
    theme.line(
        axes,
        times,
        [v * kilo for v in data["discrepancy"]],
        color=observed,
        width=1.4,
    )
    axes.set_xlim(times[0], times[-1])
    axes.set_ylim(-120.0, 150.0)
    theme.direct_labels(
        axes,
        [
            (times[-1], truth[-1] * kilo, "true pulse", theme.entity_colour("truth")),
            (times[-1], posterior.mean[-1] * kilo, "GP posterior mean", model),
            (
                times[-1],
                data["discrepancy"][-1] * kilo,
                "Tikhonov, discrepancy",
                observed,
            ),
        ],
        gap_points=13.0,
    )
    theme.axis_labels(axes, "Time (s)", "Surface heat flux (kW/m²)")
    theme.title_block(
        axes,
        f"From {facts['depth_mm']} mm deep the GP band covers the pulse at "
        f"{facts['covered']} of {facts['points']} times;\n"
        f"GCV's choice (relative error {facts['gcv_error']}, not drawn) does not",
        "Synthetic slab, data from a finer mesh with 1 % noise; "
        "shaded: 95 % pointwise posterior band",
    )
    return axes.figure, facts


FIGURES = [
    GuideFigure(
        name="inversion-heat-flux-6mm",
        guide="nonlinear-bayesian-inversion",
        alt=(
            "Heat flux against time: a dashed true triangular pulse, the "
            "Gaussian-process posterior mean with a shaded 95 percent band, and a "
            "Tikhonov estimate, from a sensor 6 mm deep"
        ),
        draw=_flux_figure,
    ),
]
