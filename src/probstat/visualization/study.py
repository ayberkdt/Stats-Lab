"""Render every recorded stage of a :class:`~probstat.workflow.UQStudy`."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from probstat.foundations.messages import (
    check_locale,
    format_number,
    html_message,
    message_or,
)
from probstat.inference import ConfidenceInterval
from probstat.validation import CoverageAtLevel
from probstat.workflow import UQStudy

from .calibration_plots import (
    plot_discrepancy_estimate,
    plot_discrepancy_predictions,
)
from .design_plots import plot_greedy_design
from .scoring_plots import plot_validation_predictions
from .stage_plots import (
    plot_active_learning,
    plot_design_path,
    plot_design_points,
    plot_information_sensitivity,
    plot_posterior_marginals,
    plot_tail_analysis,
)
from .theme import pyplot, save_figure
from .uq import (
    plot_loss_exceedance,
    plot_method_comparison,
    plot_output_distribution,
    plot_reliability_comparison,
    plot_sobol_indices,
)


def _sobol_for(study: UQStudy | None, quantity: str) -> Any | None:
    """Return the latest Sobol result of ``quantity`` in a study, if any.

    Args:
        study: Study whose records are searched (``None`` for no study).
        quantity: QoI name.

    Returns:
        The Sobol result, or ``None``.

    References:
        [REF-SALTELLI-2010]. No equation is evaluated.

    """
    if study is None:
        return None
    for record in reversed(study.records):
        if (
            record.stage == "sensitivity"
            and getattr(record.result, "quantity", None) == quantity
        ):
            return record.result
    return None


def _scalar_candidates(candidates: Any) -> tuple[str, list[float]] | None:
    """Return the factor name and values of one-factor increasing candidates.

    Args:
        candidates: Candidate conditions of a design recommendation.

    Returns:
        ``(name, values)`` when every candidate sets the same single
        condition to strictly increasing values, else ``None``.

    References:
        [REF-HUAN-MARZOUK-2013]. No equation is evaluated.

    """
    if len(candidates) < 2:
        return None
    keys = {tuple(item) for item in candidates}
    if len(keys) != 1 or len(next(iter(keys))) != 1:
        return None
    (name,) = next(iter(keys))
    values = [float(item[name]) for item in candidates]
    if any(b <= a for a, b in zip(values, values[1:], strict=False)):
        return None
    return name, values


_COVERAGE_KEY = re.compile(r"^coverage\[(\d+(?:\.\d+)?)%\]$")
_COVERAGE_TEXT = re.compile(
    r"^(\d+)/(\d+), \[(\d+(?:\.\d+)?), (\d+(?:\.\d+)?)\]$"
)


def _coverage_from_summary(summary: Mapping[str, object]) -> list[CoverageAtLevel]:
    """Rebuild the coverage records that ``UQStudy.validate`` summarized.

    The stage stores each nominal level as ``coverage[90%]`` (the rate) and
    ``coverage[90%].interval`` (``"covered/size, [lower, upper]"`` with the
    95% Clopper-Pearson limits); entries that do not parse are skipped.

    Args:
        summary: Stage summary of a validation record.

    Returns:
        Coverage records in summary order.

    References:
        [REF-CLOPPER-PEARSON-1934]. No equation is evaluated.

    """
    records: list[CoverageAtLevel] = []
    for key, value in summary.items():
        match = _COVERAGE_KEY.match(str(key))
        if match is None:
            continue
        text = _COVERAGE_TEXT.match(str(summary.get(f"{key}.interval", "")))
        if text is None or not isinstance(value, float | int):
            continue
        covered, size = int(text.group(1)), int(text.group(2))
        lower, upper = float(text.group(3)), float(text.group(4))
        if size < 1 or not 0.0 <= lower <= upper <= 1.0:
            continue
        records.append(
            CoverageAtLevel(
                float(match.group(1)) / 100.0,
                covered,
                size,
                float(value),
                ConfidenceInterval(lower, upper, 0.95, "Clopper-Pearson interval"),
            )
        )
    return records


def _figures_for(record: Any, study: UQStudy | None = None) -> list[tuple[str, Any]]:
    """Return ``(suffix, axes)`` pairs for one stage record.

    Args:
        record: Stage record of a study.
        study: Owning study; lets the information figure show the Sobol
            indices of the same QoI next to the mutual information.

    Returns:
        Figures to save; empty for stage kinds without a standard figure.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    result = record.result
    if result is None or not getattr(record, "complete", True):
        return []
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
    if record.stage == "shapley":
        entries = tuple(
            (name, value, value - 1.96 * error, value + 1.96 * error)
            for name, value, error in zip(
                result.input_names, result.effects, result.standard_errors, strict=True
            )
        )
        return [
            (
                result.quantity,
                plot_method_comparison(
                    entries,
                    quantity_label="Shapley effect (share of output variance)",
                    title=f"{result.ranking()[0]} carries the largest Shapley share",
                    subtitle="Dot: effect. Whisker: about 95% Monte Carlo interval",
                ),
            )
        ]
    if record.stage == "reliability":
        return [("methods", plot_reliability_comparison(result.rows))]
    if record.stage == "extremes":
        empirical: dict[float, float] = {}
        for item in result.return_levels:
            value = record.summary.get(f"empirical x[T={item.return_period:g}]")
            if isinstance(value, float | int):
                empirical[item.return_period] = float(value)
        return [
            (
                result.quantity,
                plot_tail_analysis(
                    result,
                    empirical_levels=empirical,
                    confidence=result.return_levels[0].confidence
                    if result.return_levels
                    else 0.95,
                ),
            )
        ]
    if record.stage == "information":
        return [
            (
                result.quantity,
                plot_information_sensitivity(
                    result, sobol=_sobol_for(study, result.quantity)
                ),
            )
        ]
    if record.stage == "design":
        figures = [("eig", plot_design_path(result))]
        factor = _scalar_candidates(result.candidates)
        if factor is not None:
            figures.append(
                (
                    "candidates",
                    plot_greedy_design(result.selection, factor[1], factor=factor[0]),
                )
            )
        return figures
    if record.stage == "design_points":
        if len(result.points.forms[0].design_point_u) < 2:
            return []
        names = study.model.inputs.names if study is not None else None
        return [("u-space", plot_design_points(result, input_names=names))]
    if record.stage == "active_reliability":
        if len(result.history) < 2:
            return []
        return [("learning", plot_active_learning(result))]
    if record.stage == "calibration":
        prior = study.model.inputs if study is not None else None
        return [("posterior", plot_posterior_marginals(result, prior=prior))]
    if record.stage == "discrepancy_calibration":
        if result.grid_input is None or len(result.grid) < 2:
            return []
        return [
            (
                "predictions",
                plot_discrepancy_predictions(
                    result.calibration,
                    result.grid,
                    result.grid_input,
                    level=result.level,
                ),
            ),
            (
                "discrepancy",
                plot_discrepancy_estimate(
                    result.grid, result.grid_input, level=result.level
                ),
            ),
        ]
    if record.stage == "validation":
        pit_mean = record.summary.get("pit.mean")
        return [
            (
                result.quantity,
                plot_validation_predictions(
                    result,
                    coverage=_coverage_from_summary(record.summary),
                    pit_mean=float(pit_mean) if pit_mean is not None else None,
                ),
            )
        ]
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


def _safe_stem(text: str) -> str:
    """Return a file-name stem without path separators or leading dots.

    Study and QoI names are user text; this keeps every figure inside the
    requested directory.

    Args:
        text: Raw stem.

    Returns:
        Stem made of letters, digits, ``-``, ``_`` and ``.``.

    References:
        [REF-PYTHON-DATAMODEL]. No equation is evaluated.

    """
    import re

    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", text).lstrip(".-")
    return cleaned or "figure"


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
        for suffix, axes in _figures_for(record, study):
            stem = _safe_stem(f"{study.name}-{index:02d}-{record.stage}-{suffix}")
            path = destination / f"{stem}.png"
            written.append(save_figure(axes.figure, path))
            plt.close(axes.figure)
    return tuple(written)


_HTML_STYLE = """
:root {
  --canvas: #F6F4EF; --surface: #FFFFFF; --ink: #23262B; --muted: #656A73;
  --rule: #DCD8CF; --accent: #44599C;
}
@media (prefers-color-scheme: dark) {
  :root {
    --canvas: #141519; --surface: #1E2026; --ink: #F1F0EB; --muted: #A6A9B1;
    --rule: #303339; --accent: #9BA9F0;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--canvas); color: var(--ink);
  font: 16px/1.55 "Source Sans 3", "Segoe UI", system-ui, sans-serif;
}
header, nav, main, footer { max-width: 920px; margin: 0 auto; padding: 0 16px; }
header { padding-top: 32px; }
footer { padding-bottom: 64px; }
main:focus { outline: none; }
.skip-link { position: absolute; left: 16px; top: -64px; padding: 12px 16px;
  background: var(--surface); color: var(--accent); font-size: 14px;
  border: 1px solid var(--accent); border-radius: 6px; z-index: 1; }
.skip-link:focus { top: 8px; }
h1 { font-size: 24px; margin: 0 0 8px; }
h2 { font-size: 16px; margin: 0 0 4px; }
.meta, .stage-meta { color: var(--muted); font-size: 14px; margin: 0; }
.meta { margin-bottom: 24px; }
nav ol { list-style: none; padding: 0; margin: 0 0 32px; display: flex;
  flex-wrap: wrap; gap: 8px 24px; font-size: 14px; }
nav a { color: var(--accent); text-decoration: none; display: inline-block;
  padding: 6px 0; }
section { border-top: 1px solid var(--rule); padding: 24px 0; }
figure { margin: 16px 0; }
figure img { width: 100%; height: auto; border-radius: 6px; display: block; }
table { border-collapse: collapse; width: 100%; font-size: 14px;
  font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 4px 8px; border-bottom: 1px solid var(--rule); }
th { color: var(--muted); font-weight: 600; }
td.value { text-align: right; }
.warning { color: var(--ink); background: var(--surface); border-left: 3px solid
  var(--accent); padding: 4px 8px; font-size: 14px; }
section.child { border-top: 2px solid var(--accent); }
footer { color: var(--muted); font-size: 12px; margin-top: 32px; }
"""


def _encode_figure(axes: Any) -> tuple[str, str]:
    """Return a PNG data URI and text alternative for a figure, then close it.

    The description comes from the figure's titles, subtitle, and axes
    (:func:`~probstat.visualization.accessibility.describe_figure`); it is
    also written into the PNG ``Title`` and ``Description`` text chunks.

    Args:
        axes: Matplotlib axes or figure.

    Returns:
        ``("data:image/png;base64,...", description)``.

    References:
        [REF-MATPLOTLIB], [REF-PNG-SPEC-2003], [REF-W3C-WCAG21-2018].
        No equation is evaluated.

    """
    import base64
    import io

    from .accessibility import describe_figure, figure_title

    figure = getattr(axes, "figure", axes)
    description = describe_figure(figure)
    buffer = io.BytesIO()
    figure.savefig(
        buffer,
        format="png",
        dpi=150,
        facecolor=figure.get_facecolor(),
        bbox_inches="tight",
        pad_inches=0.2,
        metadata={
            "Title": figure_title(figure) or description,
            "Description": description,
        },
    )
    pyplot().close(figure)
    uri = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    return uri, description


def _stage_sections(
    study: UQStudy,
    prefix: str,
    figures_enabled: bool,
    level: int,
    locale: str = "en",
) -> tuple[list[str], list[str]]:
    """Return the HTML sections and navigation entries of a study's stages.

    Args:
        study: Study whose records are rendered.
        prefix: Anchor and number prefix (``""`` for the top-level study).
        figures_enabled: Whether Matplotlib figures are drawn.
        level: Heading level of the stage titles.
        locale: Catalogue locale of the prose.

    Returns:
        ``(sections, navigation)``.

    References:
        [REF-MATPLOTLIB], [REF-UNICODE-CLDR]. No equation is evaluated.

    """
    import html

    def say(key: str, **fields: object) -> str:
        """Return one escaped catalogue sentence in the page locale.

        References:
            [REF-WHATWG-HTML], [REF-UNICODE-CLDR]. No equation is evaluated.

        """
        return html_message(key, locale, **fields)

    kind = "surrogate" if study.evaluation_kind == "surrogate" else "forward"
    children: dict[int, list[UQStudy]] = {}
    for index, child in study.children:
        children.setdefault(index, []).append(child)
    sections: list[str] = []
    navigation: list[str] = []
    for index, record in enumerate(study.records, start=1):
        number = f"{prefix}{index}"
        anchor = f"stage-{number.replace('.', '-')}"
        navigation.append(
            f'<li><a href="#{anchor}">'
            + say("study_html.nav_item", number=number, stage=record.stage)
            + "</a></li>"
        )
        figures = _stage_figures(record, study, locale) if figures_enabled else []
        rows = "".join(
            f"<tr><td>{html.escape(str(key))}</td>"
            f'<td class="value">{html.escape(_format_value(value, locale))}</td></tr>'
            for key, value in record.summary.items()
        )
        meta = [say(f"study_html.stage.evaluations.{kind}", count=record.evaluations)]
        if record.failed_evaluations:
            meta.append(say("study_html.stage.failed", count=record.failed_evaluations))
        if record.cache_hits:
            meta.append(say("study_html.stage.cache_hits", count=record.cache_hits))
        meta.append(say("study_html.stage.seed", seed=record.seed))
        meta.append(
            say(
                "study_html.stage.input_law",
                label=record.input_law,
                hash=record.input_law_hash,
            )
        )
        if not record.complete:
            status = message_or(
                f"study.status.{record.status}", record.status, locale
            )
            meta.append(
                "<strong>" + say("study_html.stage.status", status=status) + "</strong>"
            )
        extra = ""
        if record.error:
            error = say("study.error", error=record.error)
            extra += f'<p class="stage-meta">{error}</p>'
        if record.settings:
            text = "; ".join(
                f"{key}={_format_value(value)}"
                for key, value in record.settings.items()
            )
            extra += (
                '<p class="stage-meta">' + say("study.settings", settings=text) + "</p>"
            )
        for warning in record.warnings:
            extra += (
                '<p class="warning">'
                + say("study_html.warning", message=warning)
                + "</p>"
            )
        heading = say(
            "study.stage_heading", index=number, stage=record.stage, label=record.label
        )
        sections.append(
            f'<section id="{anchor}">'
            f"<h{level}>{heading}</h{level}>"
            f'<p class="stage-meta">{" · ".join(meta)}</p>'
            + extra
            + "".join(figures)
            + f"<table><thead><tr><th>{say('study.table.quantity')}</th>"
            f"<th>{say('study.table.value')}</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></section>"
        )
        for child in children.get(index, []):
            child_sections, _ = _stage_sections(
                child, f"{number}.", figures_enabled, min(level + 1, 6), locale
            )
            title = say("study_html.child_title", number=number, name=child.name)
            evaluations = say("study_html.child_meta", count=child.total_evaluations)
            sections.append(
                '<section class="child">'
                f"<h{level}>{title}</h{level}>"
                f'<p class="stage-meta">{evaluations}</p></section>'
            )
            sections.extend(child_sections)
    return sections, navigation


def _stage_figures(record: Any, study: UQStudy, locale: str = "en") -> list[str]:
    """Return light/dark ``<figure>`` elements for one stage record.

    Each image's ``alt`` text is the generated description of its light
    variant; both variants show the same marks, so one text serves both.
    Figure titles and descriptions are English; in a page of another
    language the ``figure`` element declares ``lang="en"`` so assistive
    technology reads them with English pronunciation (WCAG 2.1 SC 3.1.2).

    References:
        [REF-MATPLOTLIB], [REF-W3C-WCAG21-2018], [REF-WHATWG-HTML].
        No equation is evaluated.

    """
    import html

    from .theme import GRAPHITE, PORCELAIN, chart_theme

    with chart_theme(PORCELAIN):
        light = [_encode_figure(axes) for _, axes in _figures_for(record, study)]
    with chart_theme(GRAPHITE):
        dark = [_encode_figure(axes)[0] for _, axes in _figures_for(record, study)]
    opening = "<figure>" if locale == "en" else '<figure lang="en">'
    return [
        f"{opening}<picture>"
        f'<source media="(prefers-color-scheme: dark)" srcset="{dark_uri}">'
        f'<img alt="{html.escape(description)}" src="{light_uri}">'
        "</picture></figure>"
        for (light_uri, description), dark_uri in zip(light, dark, strict=True)
    ]


def study_html(study: UQStudy, *, locale: str = "en") -> str:
    """Return a self-contained HTML report of every stage of a study.

    Each stage becomes one section with its headline numbers, cost, failed
    runs, cache hits, stream seed, input law, settings, warnings, and (where
    supported) its figure in light and dark variants that follow the
    reader's colour scheme. Figures are embedded as data URIs, so the file
    has no external dependencies. Surrogate studies created by
    ``UQStudy.emulate`` follow their emulation stage. The header states the
    library, Python, and platform versions, the UTC creation time, and the
    root seed; the footer repeats the root seed.

    Without Matplotlib the report is still written, without figures, and
    says so (figures need the optional ``visualization`` extra).

    The page is navigable without sight (ADR-0078): it declares its
    language, opens with a skip link to the stage results, uses ``header``,
    ``nav``, ``main``, and ``footer`` landmarks, and gives every figure a
    text alternative generated from its titles and axes.

    Every sentence comes from the message catalogue (ADR-0085); the English
    page is unchanged byte for byte. ``locale="tr"`` writes the Turkish page
    (``lang="tr"``, decimal commas); recorded stage data and the figures stay
    as recorded.

    Args:
        study: Study to render.
        locale: Catalogue locale (``"en"`` or ``"tr"``); never read from the
            environment.

    Returns:
        HTML document text.

    Raises:
        ValueError: If the locale is unknown.

    References:
        [REF-MATPLOTLIB], [REF-W3C-WCAG21-2018], [REF-WHATWG-HTML],
        [REF-UNICODE-CLDR]. No equation is evaluated.

    """
    from .theme import matplotlib_available

    code = check_locale(locale)

    def say(key: str, **fields: object) -> str:
        """Return one escaped catalogue sentence in the page locale.

        References:
            [REF-WHATWG-HTML], [REF-UNICODE-CLDR]. No equation is evaluated.

        """
        return html_message(key, code, **fields)

    model = study.model
    figures_enabled = matplotlib_available()
    sections, navigation = _stage_sections(study, "", figures_enabled, 2, code)
    shown = navigation[:7]
    more = len(navigation) - len(shown)
    extra = (
        f'<p class="meta">{say("study_html.more_stages", count=more)}</p>'
        if more > 0
        else ""
    )
    if not figures_enabled:
        extra += f'<p class="warning">{say("study_html.figures_omitted")}</p>'
    dependence = (
        say("study.dependence.independent")
        if model.inputs.is_independent
        else say(
            "study.dependence.copula", family=model.inputs.copula_family.capitalize()
        )
    )
    info = study.provenance()
    label, law_hash = study.input_law
    totals = [
        say("study_html.inputs", names=", ".join(model.inputs.names)),
        say("study_html.quantities", names=", ".join(model.quantity_names)),
        dependence,
        say("study_html.total", count=study.total_evaluations),
    ]
    failed = study.total_failed_evaluations
    if failed:
        totals.append("<strong>" + say("study_html.failed", count=failed) + "</strong>")
    budget = study.evaluation_budget
    if budget is not None:
        remaining = study.remaining_budget or 0
        totals.append(say("study_html.budget", budget=budget, remaining=remaining))
    law = [
        say("study_html.input_law", label=label, hash=law_hash),
        say("study_html.root_seed", seed=study.root_seed),
    ]
    software = [
        say("study_html.library", version=str(info["library_version"])),
        say("study_html.python", version=str(info["python_version"])),
        say("study_html.platform", platform=str(info["platform"])),
        say("study_html.created", created=str(info["created_utc"])),
    ]
    title = say("study.title.forward", name=study.name)
    footer = " ".join(
        (
            say("study_html.footer.generated"),
            say("study_html.footer.reproducible", seed=study.root_seed),
        )
    )
    return (
        f'<!doctype html><html lang="{code}"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{title}</title>"
        f"<style>{_HTML_STYLE}</style></head><body>"
        f'<a class="skip-link" href="#content">{say("study_html.skip_link")}</a>'
        f"<header><h1>{title}</h1>"
        f'<p class="meta">{" · ".join(totals)}<br>{" · ".join(law)}'
        f"<br>{' · '.join(software)}</p></header>"
        f'<nav aria-label="{say("study_html.nav_label")}">'
        f'<ol>{"".join(shown)}</ol></nav>'
        f'<main id="content" tabindex="-1">{extra}'
        + "".join(sections)
        + f"</main><footer>{footer}</footer></body></html>"
    )


def _format_value(value: object, locale: str = "en") -> str:
    """Format a summary value for display.

    Args:
        value: Recorded summary or settings value.
        locale: Catalogue locale; floats get its decimal separator.

    References:
        [REF-PYTHON-DATAMODEL], [REF-UNICODE-CLDR]. No equation is evaluated.

    """
    if isinstance(value, float):
        return format_number(value, ".6g", locale)
    return str(value)


def write_study_html(study: UQStudy, path: str | Path, *, locale: str = "en") -> Path:
    """Write :func:`study_html` to a file.

    Args:
        study: Study to render.
        path: Destination ``.html`` path (parents are created).
        locale: Catalogue locale (``"en"`` or ``"tr"``).

    Returns:
        The written path.

    References:
        [REF-MATPLOTLIB], [REF-UNICODE-CLDR]. No equation is evaluated.

    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(study_html(study, locale=locale), encoding="utf-8")
    return destination
