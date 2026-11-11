"""Inline-SVG chart primitives and the page shell of the interactive explainers.

The explainers of :mod:`probstat.visualization.interactive` are single,
offline HTML files. This module holds what they share: linear and
logarithmic scales, SVG elements with escaped attributes, axes with
localized tick labels, direct-label placement without overlap, the
stylesheet built from the chart palette tokens for light and dark colour
schemes, the state-selection script, and the page skeleton with its
landmarks.

The script never computes a statistical quantity. Every state of a page is
a precomputed set of element attributes and texts; the script selects the
state named by the slider and the mode buttons and copies it into the
document.

References:
    [REF-WHATWG-HTML], [REF-W3C-SVG2-2018], [REF-W3C-WAI-ARIA-12-2023],
    [REF-W3C-WCAG21-2018].

"""

from __future__ import annotations

import html
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from probstat.foundations.messages import format_number

from .accessibility import legible_colour
from .theme import BAND_ALPHA, GRAPHITE, PORCELAIN, ChartPalette

Update = dict[str, dict[str, Any]]

MINUS = "−"


@dataclass(frozen=True, slots=True)
class Frame:
    """Size of one chart and of the margins around its plot area.

    Args:
        width: ``viewBox`` width in user units.
        height: ``viewBox`` height.
        left: Margin for the y tick labels.
        right: Margin for direct labels at line ends.
        top: Margin for the horizontal y-axis title.
        bottom: Margin for the x tick labels and the x-axis title.

    References:
        [REF-W3C-SVG2-2018]. No equation is evaluated.

    """

    width: float = 600.0
    height: float = 330.0
    left: float = 60.0
    right: float = 24.0
    top: float = 40.0
    bottom: float = 58.0

    @property
    def x0(self) -> float:
        """Return the left edge of the plot area.

        References:
            [REF-W3C-SVG2-2018]. No equation is evaluated.

        """
        return self.left

    @property
    def x1(self) -> float:
        """Return the right edge of the plot area.

        References:
            [REF-W3C-SVG2-2018]. No equation is evaluated.

        """
        return self.width - self.right

    @property
    def y0(self) -> float:
        """Return the top edge of the plot area.

        References:
            [REF-W3C-SVG2-2018]. No equation is evaluated.

        """
        return self.top

    @property
    def y1(self) -> float:
        """Return the bottom edge of the plot area (the x baseline).

        References:
            [REF-W3C-SVG2-2018]. No equation is evaluated.

        """
        return self.height - self.bottom


@dataclass(frozen=True, slots=True)
class Scale:
    """Map data values to user units, linearly or logarithmically.

    Args:
        low: Data value at ``start``.
        high: Data value at ``end``.
        start: User-unit coordinate of ``low``.
        end: User-unit coordinate of ``high``.
        log: Whether the mapping is linear in ``log10`` of the value.

    References:
        [REF-W3C-SVG2-2018]. No equation is evaluated.

    """

    low: float
    high: float
    start: float
    end: float
    log: bool = False

    def __call__(self, value: float) -> float:
        """Return the user-unit coordinate of a data value.

        References:
            [REF-W3C-SVG2-2018]. No equation is evaluated.

        """
        if self.log:
            share = (math.log10(value) - math.log10(self.low)) / (
                math.log10(self.high) - math.log10(self.low)
            )
        else:
            share = (value - self.low) / (self.high - self.low)
        return self.start + share * (self.end - self.start)


def number(value: float) -> str:
    """Return a coordinate with one decimal and no trailing ``.0``.

    References:
        [REF-W3C-SVG2-2018]. No equation is evaluated.

    """
    text = f"{value:.1f}"
    return text[:-2] if text.endswith(".0") else text


def tag(name: str, attributes: Mapping[str, object], content: str | None = None) -> str:
    """Return one element with escaped attribute values.

    Args:
        name: Element name.
        attributes: Attribute names and values (``None`` values are skipped).
        content: Already escaped content, or ``None`` for an empty element.

    Returns:
        Markup text.

    References:
        [REF-WHATWG-HTML], [REF-W3C-SVG2-2018]. No equation is evaluated.

    """
    text = "".join(
        f' {key}="{html.escape(str(value))}"'
        for key, value in attributes.items()
        if value is not None
    )
    if content is None:
        return f"<{name}{text}/>"
    return f"<{name}{text}>{content}</{name}>"


def path(xs: Sequence[float], ys: Sequence[float], sx: Scale, sy: Scale) -> str:
    """Return the ``d`` attribute of a polyline through data points.

    References:
        [REF-W3C-SVG2-2018]. No equation is evaluated.

    """
    points = [f"{number(sx(x))},{number(sy(y))}" for x, y in zip(xs, ys, strict=True)]
    return "M" + " L".join(points)


def band_path(
    xs: Sequence[float],
    lower: Sequence[float],
    upper: Sequence[float],
    sx: Scale,
    sy: Scale,
) -> str:
    """Return the ``d`` attribute of a closed band between two curves.

    References:
        [REF-W3C-SVG2-2018]. No equation is evaluated.

    """
    top = [f"{number(sx(x))},{number(sy(y))}" for x, y in zip(xs, upper, strict=True)]
    bottom = [
        f"{number(sx(x))},{number(sy(y))}"
        for x, y in zip(reversed(xs), reversed(lower), strict=True)
    ]
    return "M" + " L".join(top + bottom) + " Z"


def tick_label(value: float, spec: str, locale: str) -> str:
    """Return a localized tick label with a typographic minus sign.

    References:
        [REF-UNICODE-CLDR]. No equation is evaluated.

    """
    text = format_number(value, spec, locale)
    return MINUS + text[1:] if text.startswith("-") else text


def axes(
    frame: Frame,
    sx: Scale,
    sy: Scale,
    xticks: Iterable[float],
    yticks: Iterable[float],
    *,
    xtitle: str,
    ytitle: str,
    locale: str,
    xspec: str = "g",
    yspec: str = "g",
) -> str:
    """Return gridlines, baseline, tick labels, and axis titles of a chart.

    Horizontal hairlines mark the y ticks; the x baseline is the bottom of
    the plot area; the y title is written horizontally above the plot.

    Args:
        frame: Chart frame.
        sx: x scale.
        sy: y scale.
        xticks: x tick values.
        yticks: y tick values.
        xtitle: Escaped x-axis title.
        ytitle: Escaped y-axis title.
        locale: Number locale of the tick labels.
        xspec: Format specification of x tick labels.
        yspec: Format specification of y tick labels.

    Returns:
        SVG markup.

    References:
        [REF-W3C-SVG2-2018], [REF-UNICODE-CLDR]. No equation is evaluated.

    """
    parts: list[str] = []
    for value in yticks:
        y = number(sy(value))
        parts.append(
            tag(
                "line",
                {
                    "class": "grid",
                    "x1": number(frame.x0),
                    "x2": number(frame.x1),
                    "y1": y,
                    "y2": y,
                },
            )
        )
        parts.append(
            tag(
                "text",
                {
                    "class": "tick",
                    "x": number(frame.x0 - 8),
                    "y": y,
                    "dy": "0.32em",
                    "text-anchor": "end",
                },
                html.escape(tick_label(value, yspec, locale)),
            )
        )
    baseline = number(frame.y1)
    parts.append(
        tag(
            "line",
            {
                "class": "axis",
                "x1": number(frame.x0),
                "x2": number(frame.x1),
                "y1": baseline,
                "y2": baseline,
            },
        )
    )
    for value in xticks:
        parts.append(
            tag(
                "text",
                {
                    "class": "tick",
                    "x": number(sx(value)),
                    "y": number(frame.y1 + 22),
                    "text-anchor": "middle",
                },
                html.escape(tick_label(value, xspec, locale)),
            )
        )
    parts.append(
        tag(
            "text",
            {
                "class": "axis-title",
                "x": number((frame.x0 + frame.x1) / 2),
                "y": number(frame.height - 8),
                "text-anchor": "middle",
            },
            xtitle,
        )
    )
    parts.append(
        tag(
            "text",
            {"class": "axis-title", "x": "0", "y": number(frame.top - 18)},
            ytitle,
        )
    )
    return "".join(parts)


def spread(
    targets: Sequence[float], gap: float, low: float, high: float
) -> list[float]:
    """Return label positions at least ``gap`` apart and close to their targets.

    Labels are pushed apart in target order and then shifted back inside
    ``[low, high]``, so two direct labels never overlap.

    Args:
        targets: Desired positions (user units, increasing downward).
        gap: Minimum distance between neighbouring labels.
        low: Smallest allowed position.
        high: Largest allowed position.

    Returns:
        Positions in the order of ``targets``.

    References:
        [REF-W3C-SVG2-2018]. No equation is evaluated.

    """
    order = sorted(range(len(targets)), key=lambda index: targets[index])
    placed = [min(max(targets[index], low), high) for index in order]
    for k in range(1, len(placed)):
        placed[k] = max(placed[k], placed[k - 1] + gap)
    if placed and placed[-1] > high:
        placed[-1] = high
        for k in range(len(placed) - 2, -1, -1):
            placed[k] = min(placed[k], placed[k + 1] - gap)
    result = [0.0] * len(targets)
    for position, index in zip(placed, order, strict=True):
        result[index] = position
    return result


def svg_chart(
    chart_id: str,
    frame: Frame,
    title: str,
    description: str,
    body: str,
) -> str:
    """Return an inline SVG chart with its title and description as text alternative.

    Args:
        chart_id: Identifier prefix of the chart's elements.
        frame: Chart frame.
        title: Escaped short title.
        description: Escaped description of the current state.
        body: Chart markup.

    Returns:
        ``<svg role="img">`` markup labelled by its ``title`` and ``desc``.

    References:
        [REF-W3C-SVG2-2018], [REF-W3C-WCAG21-2018], [REF-W3C-WAI-ARIA-12-2023].
        No equation is evaluated.

    """
    return tag(
        "svg",
        {
            "id": chart_id,
            "viewBox": f"0 0 {number(frame.width)} {number(frame.height)}",
            "role": "img",
            "aria-labelledby": f"{chart_id}-title {chart_id}-desc",
            "focusable": "false",
        },
        tag("title", {"id": f"{chart_id}-title"}, title)
        + tag("desc", {"id": f"{chart_id}-desc"}, description)
        + body,
    )


def _palette_tokens(palette: ChartPalette, surface: str) -> str:
    """Return the CSS custom properties of one palette.

    Series hues draw marks (3:1 against the canvas); label tokens are the
    same hues mixed toward the ink until text reaches 4.5:1.

    References:
        [EQ-VIZ-WCAG-002], [REF-W3C-WCAG21-2018]. No equation is evaluated.

    """
    tokens = [
        f"--canvas: {palette.canvas}",
        f"--surface: {surface}",
        f"--ink: {palette.ink}",
        f"--muted: {palette.muted}",
        f"--rule: {palette.rule}",
        f"--accent: {palette.accent}",
    ]
    for index, colour in enumerate(palette.series[:2]):
        tokens.append(f"--s{index}: {colour}")
        tokens.append(
            f"--l{index}: {legible_colour(colour, palette.canvas, palette.ink)}"
        )
    return "; ".join(tokens) + ";"


def stylesheet() -> str:
    """Return the page stylesheet with light and dark palette tokens.

    The light tokens are the porcelain chart palette, the dark ones the
    graphite palette, chosen by ``prefers-color-scheme``. Every interactive
    control is at least 44 px high and shows a 3 px focus ring.

    References:
        [REF-W3C-MEDIAQUERIES-5], [REF-W3C-WCAG21-2018]. No equation is evaluated.

    """
    light = _palette_tokens(PORCELAIN, "#FFFFFF")
    dark = _palette_tokens(GRAPHITE, "#1E2026")
    return f"""
:root {{ {light} color-scheme: light dark; }}
@media (prefers-color-scheme: dark) {{ :root {{ {dark} }} }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--canvas); color: var(--ink);
  font: 17px/1.55 "Source Sans 3", "Segoe UI", system-ui, sans-serif; }}
header, main, footer {{ max-width: 760px; margin: 0 auto; padding: 0 16px; }}
header {{ padding-top: 32px; }}
footer {{ padding-bottom: 48px; color: var(--muted); font-size: 15px; }}
h1 {{ font-size: 28px; line-height: 1.25; margin: 0 0 8px; }}
h2 {{ font-size: 20px; margin: 0 0 8px; }}
h3 {{ font-size: 17px; margin: 24px 0 4px; }}
p {{ margin: 0 0 12px; }}
.eyebrow, .note {{ color: var(--muted); font-size: 15px; }}
.skip-link {{ position: absolute; left: 16px; top: -64px; padding: 12px 16px;
  background: var(--surface); color: var(--accent); border: 1px solid var(--accent);
  border-radius: 6px; z-index: 1; }}
.skip-link:focus {{ top: 8px; }}
main:focus {{ outline: none; }}
section {{ border-top: 1px solid var(--rule); padding: 24px 0; }}
.control {{ display: grid; gap: 4px; margin: 0 0 16px; }}
.control label, .control .group-label {{ font-weight: 600; }}
.control output {{ font-variant-numeric: tabular-nums; }}
input[type="range"] {{ width: 100%; min-height: 44px; margin: 0;
  accent-color: var(--accent); }}
.modes {{ display: flex; flex-wrap: wrap; gap: 8px; }}
.modes button {{ min-height: 44px; padding: 0 16px; font: inherit; color: var(--ink);
  background: var(--surface); border: 2px solid var(--rule); border-radius: 6px;
  cursor: pointer; }}
.modes button[aria-pressed="true"] {{ border-color: var(--accent);
  font-weight: 600; }}
:focus-visible {{ outline: 3px solid var(--accent); outline-offset: 2px; }}
.readout {{ background: var(--surface); border-left: 3px solid var(--accent);
  padding: 12px 16px; margin: 0 0 8px; font-variant-numeric: tabular-nums; }}
figure {{ margin: 16px 0 0; }}
.chart {{ overflow-x: auto; }}
svg {{ display: block; width: 100%; min-width: 480px; max-width: 640px;
  height: auto; }}
svg text {{ font-size: 15px; fill: var(--ink); }}
svg .tick {{ fill: var(--muted); font-variant-numeric: tabular-nums; }}
svg .axis-title {{ font-weight: 600; }}
.grid {{ stroke: var(--rule); stroke-width: 1; }}
.axis {{ stroke: var(--muted); stroke-width: 1; }}
.line {{ fill: none; stroke-width: 2.5; stroke-linejoin: round; }}
.line.s0 {{ stroke: var(--s0); }}
.line.s1 {{ stroke: var(--s1); }}
.line.context {{ stroke: var(--muted); stroke-width: 1.5; stroke-opacity: 0.55; }}
.line.prior {{ stroke: var(--muted); stroke-dasharray: 6 5; }}
.ref {{ stroke: var(--accent); stroke-width: 1.5; stroke-dasharray: 7 6; }}
.guide {{ stroke: var(--muted); stroke-width: 1; stroke-dasharray: 2 4; }}
.band {{ stroke: none; fill-opacity: {BAND_ALPHA}; }}
.band.s0 {{ fill: var(--s0); }}
.band.s1 {{ fill: var(--s1); }}
.dot {{ stroke: var(--canvas); stroke-width: 2; }}
.dot.s0 {{ fill: var(--s0); }}
.dot.ref-dot {{ fill: var(--accent); }}
svg .lab-s0 {{ fill: var(--l0); font-weight: 600; }}
svg .lab-s1 {{ fill: var(--l1); font-weight: 600; }}
svg .lab-ref {{ fill: var(--accent); font-weight: 600; }}
.key {{ list-style: none; display: flex; flex-wrap: wrap; gap: 4px 20px;
  padding: 0; margin: 8px 0 0; font-size: 15px; }}
.key li {{ display: flex; align-items: center; gap: 8px; }}
.swatch {{ display: inline-block; width: 24px; height: 0; border-top: 3px solid
  var(--muted); }}
.swatch.s0 {{ border-top-color: var(--s0); }}
.swatch.s1 {{ border-top-color: var(--s1); }}
.swatch.prior {{ border-top-style: dashed; }}
.swatch.ref {{ border-top: 2px dashed var(--accent); }}
.swatch.context {{ border-top-width: 2px; opacity: 0.55; }}
.swatch.band {{ height: 14px; border-top: 0; opacity: 0.35; background: var(--muted); }}
.swatch.band.s0 {{ background: var(--s0); }}
.swatch.band.s1 {{ background: var(--s1); }}
.sources {{ padding-left: 20px; margin: 0 0 12px; }}
code {{ font-family: "Cascadia Mono", Consolas, monospace; font-size: 15px; }}
"""


_SCRIPT = """
(function () {
  "use strict";
  var view = JSON.parse(document.getElementById("explainer-view").textContent);
  var slider = document.getElementById("state");
  var buttons = Array.prototype.slice.call(document.querySelectorAll("[data-mode]"));
  var mode = view.initialMode;
  function apply(update) {
    if (!update) { return; }
    Object.keys(update.attrs || {}).forEach(function (id) {
      var node = document.getElementById(id);
      var attrs = update.attrs[id];
      Object.keys(attrs).forEach(function (name) {
        node.setAttribute(name, attrs[name]);
      });
    });
    Object.keys(update.text || {}).forEach(function (id) {
      document.getElementById(id).textContent = update.text[id];
    });
  }
  function render() {
    var index = Number(slider.value);
    apply(view.slider[index]);
    apply(view.mode[mode]);
    apply(view.both[index][mode]);
    buttons.forEach(function (button) {
      button.setAttribute("aria-pressed", String(Number(button.dataset.mode) === mode));
    });
  }
  slider.addEventListener("input", render);
  buttons.forEach(function (button) {
    button.addEventListener("click", function () {
      mode = Number(button.dataset.mode);
      render();
    });
  });
  render();
})();
"""


def script() -> str:
    """Return the state-selection script of every explainer page.

    The script reads the precomputed states from the ``explainer-view``
    JSON block and, on slider input or a mode button press, copies the
    attributes and texts of the selected state into the document. It
    computes nothing beyond the integer state index.

    References:
        [REF-WHATWG-HTML], [REF-W3C-WAI-ARIA-12-2023]. No equation is evaluated.

    """
    return _SCRIPT


def json_block(element_id: str, payload: object) -> str:
    """Return a ``<script type="application/json">`` block that cannot end early.

    Every ``</`` is written with an escaped solidus (a valid JSON escape),
    so no string in the payload can close the script element.

    References:
        [REF-WHATWG-HTML]. No equation is evaluated.

    """
    text = json.dumps(
        payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    )
    return (
        f'<script type="application/json" id="{element_id}">'
        + text.replace("</", "<\\/")
        + "</script>"
    )


@dataclass(slots=True)
class States:
    """Precomputed states of one page: by slider index, by mode, and by both.

    Args:
        slider: Updates that depend only on the slider index.
        mode: Updates that depend only on the selected mode.
        both: ``both[i][j]`` updates for slider index ``i`` and mode ``j``.
        initial_index: Slider index shown on load and without script.
        initial_mode: Mode shown on load and without script.

    References:
        [REF-WHATWG-HTML]. No equation is evaluated.

    """

    slider: list[Update]
    mode: list[Update]
    both: list[list[Update]]
    initial_index: int
    initial_mode: int = 0

    def initial(self) -> Update:
        """Return the merged update of the initial state.

        References:
            [REF-WHATWG-HTML]. No equation is evaluated.

        """
        merged: Update = {"attrs": {}, "text": {}}
        for update in (
            self.slider[self.initial_index],
            self.mode[self.initial_mode],
            self.both[self.initial_index][self.initial_mode],
        ):
            for key, attrs in update.get("attrs", {}).items():
                merged["attrs"].setdefault(key, {}).update(attrs)
            merged["text"].update(update.get("text", {}))
        return merged

    def payload(self) -> dict[str, object]:
        """Return the JSON payload read by the page script.

        References:
            [REF-WHATWG-HTML]. No equation is evaluated.

        """
        return {
            "initialMode": self.initial_mode,
            "slider": self.slider,
            "mode": self.mode,
            "both": self.both,
        }


def dynamic(
    name: str,
    element_id: str,
    initial: Update,
    attributes: Mapping[str, object] | None = None,
    content: str | None = None,
) -> str:
    """Return an element whose state-dependent attributes and text start initial.

    Args:
        name: Element name.
        element_id: Element ``id``; the key of its entries in each state.
        initial: Merged initial update (:meth:`States.initial`).
        attributes: Static attributes.
        content: Static escaped content, used when the state has no text.

    Returns:
        Markup of the element in its initial state.

    References:
        [REF-WHATWG-HTML], [REF-W3C-SVG2-2018]. No equation is evaluated.

    """
    merged: dict[str, object] = {"id": element_id, **(attributes or {})}
    merged.update(initial["attrs"].get(element_id, {}))
    text = initial["text"].get(element_id)
    body = html.escape(text) if text is not None else content
    return tag(name, merged, body)
