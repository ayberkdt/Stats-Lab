"""Render every recorded stage of a :class:`~probstat.workflow.UQStudy`."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from probstat.workflow import UQStudy

from .theme import pyplot, save_figure
from .uq import (
    plot_loss_exceedance,
    plot_method_comparison,
    plot_output_distribution,
    plot_reliability_comparison,
    plot_sobol_indices,
)


def _figures_for(record: Any) -> list[tuple[str, Any]]:
    """Return ``(suffix, axes)`` pairs for one stage record.

    Args:
        record: Stage record of a study.

    Returns:
        Figures to save; empty for stage kinds without a standard figure.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    result = record.result
    if record.stage == "propagation":
        return [
            (name, plot_output_distribution(output.values, name=name))
            for name, output in result.outputs.items()
        ]
    if record.stage == "sensitivity":
        return [
            (
                result.quantity,
                plot_sobol_indices(
                    result.input_names,
                    tuple(index.estimate for index in result.first_order),
                    tuple(index.estimate for index in result.total_effect),
                ),
            )
        ]
    if record.stage == "screening":
        entries = tuple(
            (name, value, None, None)
            for name, value in zip(result.input_names, result.mu_star, strict=True)
        )
        return [
            (
                "morris",
                plot_method_comparison(
                    entries,
                    quantity_label="mu* (mean absolute elementary effect)",
                    title=f"{result.ranking()[0]} has the largest elementary effects",
                    subtitle="Morris screening on the unit scale of each input",
                ),
            )
        ]
    if record.stage == "reliability":
        return [("methods", plot_reliability_comparison(result.rows))]
    if record.stage == "risk":
        return [
            (
                result.quantity,
                plot_loss_exceedance(
                    result.losses,
                    result.value_at_risk,
                    result.conditional_value_at_risk,
                ),
            )
        ]
    return []


def save_study_figures(study: UQStudy, directory: str | Path) -> tuple[Path, ...]:
    """Save one themed figure per supported stage of a study.

    Args:
        study: Study whose records are rendered.
        directory: Output directory (created when missing).

    Returns:
        Paths of the written PNG files, in stage order.

    References:
        [REF-MATPLOTLIB].

    """
    plt = pyplot()
    destination = Path(directory)
    written: list[Path] = []
    for index, record in enumerate(study.records, start=1):
        for suffix, axes in _figures_for(record):
            path = destination / f"{study.name}-{index:02d}-{record.stage}-{suffix}.png"
            written.append(save_figure(axes.figure, path))
            plt.close(axes.figure)
    return tuple(written)
