"""Self-contained interactive explanations of library results (ADR-0085).

Each explainer is one offline HTML file: inline CSS, an inline SVG chart
per view, and a short vanilla script, with no external script, font,
stylesheet, or network request. Every number on the page is computed here,
in Python, by the library function the page explains, and embedded twice:

* ``explainer-data``: the raw library outputs as JSON, so a reader (or a
  test) can check every number;
* ``explainer-view``: the precomputed page states (element attributes and
  sentences per slider position and mode).

The script only selects among the precomputed states; it never evaluates a
distribution, a bound, or an estimator. Without script the page shows its
initial state and says so.

The pages follow ADR-0078: they declare their language, open with a skip
link, use ``header``, ``main``, ``footer``, and labelled ``section``
landmarks, give every chart a title and a state description, announce the
current numbers in an ``aria-live`` region, and use native, labelled
controls (``input type="range"`` and buttons with ``aria-pressed``) with a
visible focus ring. Colours are the chart palette tokens of ADR-0024 for the
light and the dark colour scheme. All prose comes from the message
catalogue (:mod:`probstat.foundations.messages`), so ``locale="tr"`` writes
the Turkish page with decimal commas.

Three explainers are provided:

* :func:`clt_explainer`: exact Kolmogorov distance of the standardized
  exponential mean to the normal law against the Berry-Esseen bound;
* :func:`bayesian_updating_explainer`: Beta-Binomial prior, likelihood, and
  posterior, with the credible interval or the Wilson interval;
* :func:`monte_carlo_error_explainer`: Monte Carlo estimates of ``pi`` and
  their confidence bands against the sample size for several seeds.

References:
    [REF-WHATWG-HTML], [REF-W3C-WAI-ARIA-12-2023], [REF-W3C-WCAG21-2018],
    [REF-SHEVTSOVA-2011], [REF-GELMAN-ETAL-2013], [REF-ROBERT-CASELLA-2004].

"""

from __future__ import annotations

import html
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import probstat
from probstat.distributions import Beta
from probstat.foundations.messages import check_locale, html_message, message
from probstat.foundations.random_streams import RandomStream
from probstat.inference import wilson_score_interval
from probstat.monte_carlo import MonteCarloEstimate, convergence_trace
from probstat.probability import (
    clt_error_profile,
    exponential_summand,
    standardized_mean_cdf,
)
from probstat.probability.limit_theorems import standard_normal_cdf
from probstat.reliability import demand_failure_probability

from ._explainer_page import (
    Frame,
    Scale,
    States,
    Update,
    axes,
    band_path,
    dynamic,
    json_block,
    number,
    path,
    script,
    spread,
    stylesheet,
    svg_chart,
    tag,
)

CLT_SIZES: tuple[int, ...] = (
    1,
    2,
    3,
    4,
    5,
    6,
    8,
    10,
    12,
    15,
    20,
    25,
    30,
    40,
    50,
    60,
    80,
    100,
    150,
    200,
    300,
    500,
    1000,
)
MC_CHECKPOINTS: tuple[int, ...] = (
    10,
    20,
    50,
    100,
    200,
    500,
    1_000,
    2_000,
    5_000,
    10_000,
    20_000,
    50_000,
)
DEFAULT_SEED = 2026

_Z_GRID: tuple[float, ...] = tuple(round(-4.0 + 0.05 * k, 10) for k in range(161))
_P_GRID: tuple[float, ...] = tuple((k + 0.5) / 100.0 for k in range(100))
_LABEL_GAP = 20.0

# Short citations shown on the pages; the full entries are in
# docs/references.md under the same keys.
_SOURCES: Mapping[str, str] = {
    "REF-SHEVTSOVA-2011": (
        "I. Shevtsova (2011). On the absolute constants in the Berry-Esseen type "
        "inequalities for identically distributed summands."
    ),
    "REF-CASELLA-BERGER-2002": (
        "G. Casella and R. L. Berger (2002). Statistical Inference, 2nd edition."
    ),
    "REF-BRENT-1973": (
        "R. P. Brent (1973). Algorithms for Minimization without Derivatives."
    ),
    "REF-GELMAN-ETAL-2013": (
        "A. Gelman, J. B. Carlin, H. S. Stern, D. B. Dunson, A. Vehtari, and "
        "D. B. Rubin (2013). Bayesian Data Analysis, 3rd edition."
    ),
    "REF-ATWOOD-NUREG-6823-2003": (
        "C. L. Atwood et al. (2003). Handbook of Parameter Estimation for "
        "Probabilistic Risk Assessment, NUREG/CR-6823."
    ),
    "REF-WILSON-1927": (
        "E. B. Wilson (1927). Probable inference, the law of succession, and "
        "statistical inference."
    ),
    "REF-ROBERT-CASELLA-2004": (
        "C. P. Robert and G. Casella (2004). Monte Carlo Statistical Methods, "
        "2nd edition."
    ),
    "REF-STEELE-LEA-FLOOD-2014": (
        "G. L. Steele, D. Lea, and C. H. Flood (2014). Fast splittable "
        "pseudorandom number generators."
    ),
}


@dataclass(frozen=True, slots=True)
class Explainer:
    """One interactive explanation: its embedded data and its HTML page.

    Args:
        name: File-name stem such as ``"clt-explainer"``.
        locale: Catalogue locale of the prose.
        title: Page title (plain text).
        data: JSON-safe library outputs embedded as ``explainer-data``.
        html: Complete, self-contained HTML document.

    References:
        [REF-WHATWG-HTML]. No equation is evaluated.

    """

    name: str
    locale: str
    title: str
    data: Mapping[str, Any]
    html: str

    def write(self, path: str | Path) -> Path:
        """Write the page as UTF-8 (parents are created).

        Args:
            path: Destination ``.html`` path.

        Returns:
            The written path.

        References:
            [REF-WHATWG-HTML]. No equation is evaluated.

        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.html, encoding="utf-8")
        return destination


def _sayer(locale: str) -> Callable[..., str]:
    """Return a function giving escaped catalogue sentences in one locale.

    References:
        [REF-UNICODE-CLDR], [REF-WHATWG-HTML]. No equation is evaluated.

    """

    def say(key: str, **fields: object) -> str:
        """Return one escaped catalogue sentence.

        References:
            [REF-UNICODE-CLDR]. No equation is evaluated.

        """
        return html_message(key, locale, **fields)

    return say


def _texter(locale: str) -> Callable[..., str]:
    """Return a function giving plain catalogue sentences in one locale.

    Plain text is used for state texts, which the script assigns with
    ``textContent`` (no markup is ever interpreted).

    References:
        [REF-UNICODE-CLDR]. No equation is evaluated.

    """

    def text(key: str, **fields: object) -> str:
        """Return one plain catalogue sentence.

        References:
            [REF-UNICODE-CLDR]. No equation is evaluated.

        """
        return message(key, locale, **fields)

    return text


def _nice(value: float, *, up: bool) -> float:
    """Return the nearest 1-2-5 number above (or below) a positive value.

    References:
        [REF-W3C-SVG2-2018]. No equation is evaluated.

    """
    exponent = math.floor(math.log10(value))
    steps = (1.0, 2.0, 5.0, 10.0)
    candidates = [step * 10.0**exponent for step in steps]
    if up:
        return min(c for c in candidates if c >= value * (1 - 1e-12))
    return max(c for c in candidates if c <= value * (1 + 1e-12))


def _decades(low: float, high: float) -> list[float]:
    """Return the powers of ten inside ``[low, high]``.

    References:
        [REF-W3C-SVG2-2018]. No equation is evaluated.

    """
    first = math.ceil(math.log10(low) - 1e-12)
    last = math.floor(math.log10(high) + 1e-12)
    return [10.0**k for k in range(first, last + 1)]


def _text_width(text: str, size: float = 15.0) -> float:
    """Return a generous estimate of the rendered width of one line of text.

    References:
        [REF-W3C-SVG2-2018]. No equation is evaluated.

    """
    return 0.62 * size * len(text)


def _slider(label: str, count: int, initial: Update, initial_index: int) -> str:
    """Return the labelled range control and its output.

    References:
        [REF-WHATWG-HTML], [REF-W3C-WAI-ARIA-12-2023]. No equation is evaluated.

    """
    return (
        '<div class="control">'
        f'<label for="state">{label}</label>'
        + dynamic(
            "input",
            "state",
            initial,
            {
                "type": "range",
                "min": 0,
                "max": count - 1,
                "step": 1,
                "value": initial_index,
                "aria-describedby": "state-value",
            },
        )
        + dynamic("output", "state-value", initial, {"for": "state"})
        + "</div>"
    )


def _modes(label: str, names: Sequence[str], initial_mode: int) -> str:
    """Return a labelled group of toggle buttons, one per mode.

    References:
        [REF-WHATWG-HTML], [REF-W3C-WAI-ARIA-12-2023]. No equation is evaluated.

    """
    buttons = "".join(
        tag(
            "button",
            {
                "type": "button",
                "data-mode": index,
                "aria-pressed": "true" if index == initial_mode else "false",
            },
            name,
        )
        for index, name in enumerate(names)
    )
    return (
        '<div class="control" role="group" aria-labelledby="mode-label">'
        f'<span class="group-label" id="mode-label">{label}</span>'
        f'<div class="modes">{buttons}</div></div>'
    )


def _figure(heading_id: str, heading: str, svg: str, key: str = "") -> str:
    """Return a chart heading, the scrollable chart, and its optional key.

    References:
        [REF-WHATWG-HTML], [REF-W3C-SVG2-2018]. No equation is evaluated.

    """
    return (
        f'<h3 id="{heading_id}">{heading}</h3>'
        f'<figure aria-labelledby="{heading_id}"><div class="chart">{svg}</div>'
        f"{key}</figure>"
    )


def _key(items: Sequence[tuple[str, str]], extra: str = "") -> str:
    """Return a key of swatches; each item is ``(swatch classes, label markup)``.

    Args:
        items: Static key entries.
        extra: Markup of further ``li`` entries (for state-dependent ones).

    References:
        [REF-WHATWG-HTML]. No equation is evaluated.

    """
    return (
        '<ul class="key">'
        + "".join(
            f'<li><span class="swatch {classes}" aria-hidden="true"></span>{label}</li>'
            for classes, label in items
        )
        + extra
        + "</ul>"
    )


def _page(
    *,
    locale: str,
    title: str,
    lede: str,
    controls: str,
    figures: str,
    method: Sequence[str],
    equations: Sequence[str],
    references: Sequence[str],
    data: Mapping[str, Any],
    states: States,
) -> str:
    """Return the complete explainer document.

    Args:
        locale: Catalogue locale (also the ``lang`` attribute).
        title: Escaped page title.
        lede: Escaped introduction.
        controls: Control markup.
        figures: Figure markup.
        method: Escaped method paragraphs.
        equations: ``EQ-*`` keys cited on the page.
        references: ``REF-*`` keys cited on the page.
        data: Raw library outputs.
        states: Precomputed page states.

    Returns:
        HTML document text.

    References:
        [REF-WHATWG-HTML], [REF-W3C-WAI-ARIA-12-2023], [REF-W3C-WCAG21-2018].
        No equation is evaluated.

    """
    say = _sayer(locale)
    initial = states.initial()
    readout = dynamic("p", "live", initial, {"class": "readout", "aria-live": "polite"})
    equation_items = "".join(
        f"<li><code>{key}</code> {say(f'equation.{key}')}</li>" for key in equations
    )
    source_items = "".join(
        f"<li><code>{key}</code> {html.escape(_SOURCES[key])}</li>"
        for key in references
    )
    paragraphs = "".join(f"<p>{text}</p>" for text in method)
    return (
        f'<!doctype html><html lang="{locale}"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{title} · probstat</title>"
        f"<style>{stylesheet()}</style></head><body>"
        f'<a class="skip-link" href="#explainer">{say("explainer.skip_link")}</a>'
        f'<header><p class="eyebrow">{say("explainer.eyebrow")}</p>'
        f"<h1>{title}</h1><p>{lede}</p></header>"
        '<main id="explainer" tabindex="-1">'
        '<section aria-labelledby="explore-heading">'
        f'<h2 id="explore-heading">{say("explainer.heading.explore")}</h2>'
        f"{controls}{readout}"
        f'<noscript><p class="note">{say("explainer.noscript")}</p></noscript>'
        f"{figures}</section>"
        '<section aria-labelledby="method-heading">'
        f'<h2 id="method-heading">{say("explainer.heading.method")}</h2>'
        f"{paragraphs}</section></main>"
        f"<footer><h2>{say('explainer.heading.sources')}</h2>"
        f'<ul class="sources">{equation_items}{source_items}</ul>'
        f"<p>{say('explainer.offline', version=probstat.__version__)}</p></footer>"
        + json_block("explainer-data", data)
        + json_block("explainer-view", states.payload())
        + f"<script>{script()}</script></body></html>"
    )


def _index_of(values: Sequence[int], wanted: int, name: str) -> int:
    """Return the position of ``wanted`` in ``values`` or raise.

    Raises:
        ValueError: If ``wanted`` is not one of ``values``.

    References:
        [REF-PYTHON-DATAMODEL]. No equation is evaluated.

    """
    if wanted not in values:
        raise ValueError(f"{name} must be one of {tuple(values)}")
    return list(values).index(wanted)


def _sizes(values: Sequence[int], name: str, minimum: int) -> tuple[int, ...]:
    """Return strictly increasing integer sizes of at least ``minimum``.

    Raises:
        ValueError: If fewer than two sizes are given, a size is too small,
            or the sizes do not increase.

    References:
        [REF-PYTHON-DATAMODEL]. No equation is evaluated.

    """
    sizes = tuple(int(value) for value in values)
    if len(sizes) < 2 or sizes[0] < minimum:
        raise ValueError(f"{name} needs at least two sizes, each at least {minimum}")
    if any(b <= a for a, b in zip(sizes, sizes[1:], strict=False)):
        raise ValueError(f"{name} must be strictly increasing")
    return sizes


# ---------------------------------------------------------------------------
# Central limit theorem
# ---------------------------------------------------------------------------


def clt_explainer(
    *,
    rate: float = 1.0,
    sizes: Sequence[int] = CLT_SIZES,
    initial_size: int = 30,
    locale: str = "en",
) -> Explainer:
    """Return the central-limit-theorem explainer for exponential summands.

    For each sample size ``n`` the page shows ``F_n(z) - Phi(z)`` on a grid
    of ``z`` (``F_n`` the exact distribution function of the standardized
    mean of ``n`` Exponential(``rate``) summands) and, on a log-log chart,
    the exact Kolmogorov distance ``K_n`` against the Berry-Esseen bound for
    every ``n``. A slider selects ``n``.

    Args:
        rate: Rate of the exponential summands (the standardized law does
            not depend on it).
        sizes: Strictly increasing sample sizes, at least 1.
        initial_size: Size shown on load; one of ``sizes``.
        locale: Catalogue locale (``"en"`` or ``"tr"``).

    Returns:
        The explainer; ``data`` holds ``sizes``, ``kolmogorov_normal``,
        ``berry_esseen``, the ``z`` grid, and ``cdf_difference`` per size.

    Raises:
        ValueError: If the sizes, the initial size, or the locale are invalid.

    Equation:
        [EQ-LIM-001], [EQ-LIM-002], [EQ-LIM-003], [EQ-LIM-010].

    References:
        [REF-SHEVTSOVA-2011], [REF-CASELLA-BERGER-2002], [REF-BRENT-1973].

    """
    code = check_locale(locale)
    say, text = _sayer(code), _texter(code)
    summand = exponential_summand(rate)
    checked = _sizes(sizes, "sizes", 1)
    start = _index_of(checked, initial_size, "initial_size")
    rows = clt_error_profile(summand, checked)
    distance = [row.kolmogorov_normal for row in rows]
    bound = [row.berry_esseen for row in rows]
    difference = [
        [standardized_mean_cdf(summand, n, z) - standard_normal_cdf(z) for z in _Z_GRID]
        for n in checked
    ]
    data: dict[str, Any] = {
        "explainer": "central-limit-theorem",
        "summand": {
            "law": "exponential",
            "rate": float(rate),
            "mean": summand.mean,
            "standard_deviation": summand.standard_deviation,
            "third_absolute_central_moment": summand.third_absolute_central_moment,
        },
        "sizes": list(checked),
        "kolmogorov_normal": distance,
        "berry_esseen": bound,
        "z": list(_Z_GRID),
        "cdf_difference": difference,
        "equations": ["EQ-LIM-001", "EQ-LIM-002", "EQ-LIM-003", "EQ-LIM-010"],
        "references": [
            "REF-SHEVTSOVA-2011",
            "REF-CASELLA-BERGER-2002",
            "REF-BRENT-1973",
        ],
    }

    # Chart A: F_n - Phi over z, with the exact distance as +-K_n lines.
    frame_a = Frame(right=96.0)
    largest = max(abs(value) for row in difference for value in row)
    limit = max(0.2, math.ceil(largest * 10.0 - 1e-9) / 10.0)
    ax = Scale(-4.0, 4.0, frame_a.x0, frame_a.x1)
    ay = Scale(-limit, limit, frame_a.y1, frame_a.y0)
    # Chart B: log-log K_n and the Berry-Esseen bound against n.
    frame_b = Frame(right=168.0)
    bx = Scale(
        _nice(checked[0], up=False),
        _nice(checked[-1], up=True),
        frame_b.x0,
        frame_b.x1,
        log=True,
    )
    low = _nice(min(distance), up=False)
    high = _nice(max(bound), up=True)
    by = Scale(low, high, frame_b.y1, frame_b.y0, log=True)

    slider_updates: list[Update] = []
    for index, n in enumerate(checked):
        k_n, b_n = distance[index], bound[index]
        curve = difference[index]
        gaps = [abs(value) for value in curve]
        farthest = gaps.index(max(gaps))
        label_y = spread(
            [ay(curve[-1]), ay(k_n)], _LABEL_GAP, frame_a.y0 + 8, frame_a.y1 - 8
        )
        live = [text("clt.live.distance", n=n, distance=k_n)]
        if b_n >= 1.0:
            live.append(text("clt.live.bound_vacuous", bound=b_n))
        else:
            live.append(text("clt.live.bound", bound=b_n, ratio=b_n / k_n))
        upper, lower = number(ay(k_n)), number(ay(-k_n))
        slider_updates.append(
            {
                "attrs": {
                    "state": {"aria-valuetext": text("clt.slider_value", n=n)},
                    "clt-a-curve": {"d": path(_Z_GRID, curve, ax, ay)},
                    "clt-a-upper": {"y1": upper, "y2": upper},
                    "clt-a-lower": {"y1": lower, "y2": lower},
                    "clt-a-curve-label": {"y": number(label_y[0])},
                    "clt-a-ref-label": {"y": number(label_y[1])},
                    "clt-b-guide": {"x1": number(bx(n)), "x2": number(bx(n))},
                    "clt-b-dot-k": {"cx": number(bx(n)), "cy": number(by(k_n))},
                    "clt-b-dot-b": {"cx": number(bx(n)), "cy": number(by(b_n))},
                },
                "text": {
                    "state-value": text("clt.slider_value", n=n),
                    "live": " ".join(live),
                    "clt-a-desc": text(
                        "clt.chart_a.desc", n=n, distance=k_n, z=_Z_GRID[farthest]
                    ),
                    "clt-b-desc": text(
                        "clt.chart_b.desc",
                        first=checked[0],
                        last=checked[-1],
                        k_first=distance[0],
                        k_last=distance[-1],
                        b_first=bound[0],
                        b_last=bound[-1],
                        n=n,
                    ),
                },
            }
        )
    states = States(slider_updates, [{}], [[{}] for _ in checked], start)
    initial = states.initial()

    right_a = number(frame_a.x1 + 8)
    body_a = (
        axes(
            frame_a,
            ax,
            ay,
            (-4, -2, 0, 2, 4),
            [limit * k / 2 for k in (-2, -1, 0, 1, 2)],
            xtitle=say("clt.chart_a.x"),
            ytitle=say("clt.chart_a.y"),
            locale=code,
        )
        + dynamic(
            "line",
            "clt-a-upper",
            initial,
            {"class": "ref", "x1": number(frame_a.x0), "x2": number(frame_a.x1)},
        )
        + dynamic(
            "line",
            "clt-a-lower",
            initial,
            {"class": "ref", "x1": number(frame_a.x0), "x2": number(frame_a.x1)},
        )
        + dynamic("path", "clt-a-curve", initial, {"class": "line s0"})
        + dynamic(
            "text",
            "clt-a-curve-label",
            initial,
            {"class": "lab-s0", "x": right_a, "dy": "0.32em"},
            say("clt.chart_a.label.curve"),
        )
        + dynamic(
            "text",
            "clt-a-ref-label",
            initial,
            {"class": "lab-ref", "x": right_a, "dy": "0.32em"},
            say("clt.chart_a.label.distance"),
        )
    )
    end_labels = spread(
        [by(bound[-1]), by(distance[-1])], _LABEL_GAP, frame_b.y0 + 8, frame_b.y1 - 8
    )
    right_b = number(frame_b.x1 + 8)
    body_b = (
        axes(
            frame_b,
            bx,
            by,
            _decades(bx.low, bx.high),
            _decades(by.low, by.high),
            xtitle=say("clt.chart_b.x"),
            ytitle=say("clt.chart_b.y"),
            locale=code,
            xspec=",g",
        )
        + dynamic(
            "line",
            "clt-b-guide",
            initial,
            {"class": "guide", "y1": number(frame_b.y0), "y2": number(frame_b.y1)},
        )
        + tag(
            "path", {"class": "ref", "fill": "none", "d": path(checked, bound, bx, by)}
        )
        + tag("path", {"class": "line s0", "d": path(checked, distance, bx, by)})
        + dynamic("circle", "clt-b-dot-b", initial, {"class": "dot ref-dot", "r": 5})
        + dynamic("circle", "clt-b-dot-k", initial, {"class": "dot s0", "r": 5})
        + tag(
            "text",
            {
                "class": "lab-ref",
                "x": right_b,
                "y": number(end_labels[0]),
                "dy": "0.32em",
            },
            say("clt.chart_b.label.bound"),
        )
        + tag(
            "text",
            {
                "class": "lab-s0",
                "x": right_b,
                "y": number(end_labels[1]),
                "dy": "0.32em",
            },
            say("clt.chart_b.label.exact"),
        )
    )
    figures = _figure(
        "clt-a-heading",
        say("clt.chart_a.heading"),
        svg_chart(
            "clt-a",
            frame_a,
            say("clt.chart_a.title"),
            html.escape(initial["text"]["clt-a-desc"]),
            body_a,
        ),
    ) + _figure(
        "clt-b-heading",
        say("clt.chart_b.heading"),
        svg_chart(
            "clt-b",
            frame_b,
            say("clt.chart_b.title"),
            html.escape(initial["text"]["clt-b-desc"]),
            body_b,
        ),
    )
    controls = _slider(say("clt.slider_label"), len(checked), initial, start)
    method = [
        say("clt.method.law", rate=float(rate)),
        say("clt.method.distance"),
        say("clt.method.bound", rho=summand.third_absolute_central_moment),
    ]
    title = message("clt.title", code)
    page = _page(
        locale=code,
        title=html.escape(title),
        lede=say("clt.lede"),
        controls=controls,
        figures=figures,
        method=method,
        equations=data["equations"],
        references=data["references"],
        data=data,
        states=states,
    )
    return Explainer("clt-explainer", code, title, data, page)


# ---------------------------------------------------------------------------
# Bayesian updating
# ---------------------------------------------------------------------------


def bernoulli_outcomes(probability: float, trials: int, seed: int) -> tuple[int, ...]:
    """Return seeded Bernoulli outcomes from the library's random streams.

    The draws come from the child stream ``"bernoulli"`` of
    ``RandomStream(seed)``: outcome ``k`` is ``1`` when the ``k``-th uniform
    is below ``probability``.

    Args:
        probability: Success probability in ``(0, 1)``.
        trials: Number of trials (at least 1).
        seed: Root seed.

    Returns:
        Outcomes ``0`` or ``1``.

    Raises:
        ValueError: If ``probability`` or ``trials`` is out of range.

    Equation:
        [EQ-RNG-SPLITMIX-001].

    References:
        [REF-STEELE-LEA-FLOOD-2014].

    """
    if not 0.0 < probability < 1.0:
        raise ValueError("probability must lie in (0, 1)")
    if trials < 1:
        raise ValueError("trials must be at least 1")
    generator = RandomStream(seed).child("bernoulli").generator()
    return tuple(1 if generator.random() < probability else 0 for _ in range(trials))


def bayesian_updating_explainer(
    *,
    probability: float = 0.3,
    trials: int = 40,
    prior: tuple[float, float] = (2.0, 2.0),
    confidence: float = 0.95,
    seed: int = DEFAULT_SEED,
    initial_trials: int = 10,
    locale: str = "en",
) -> Explainer:
    """Return the Beta-Binomial updating explainer.

    Simulated Bernoulli trials (:func:`bernoulli_outcomes`) update a
    ``Beta(a, b)`` prior one trial at a time. For each number of observed
    trials ``n`` the page draws the prior, the likelihood scaled to unit
    area, and the posterior ``Beta(a + x, b + n - x)``
    (:func:`~probstat.reliability.demand_failure_probability`), and shades
    either the equal-tailed credible interval or the Wilson score interval
    (:func:`~probstat.inference.wilson_score_interval`), chosen by a button.

    Args:
        probability: True success probability of the simulated trials.
        trials: Number of trials the slider can reveal.
        prior: Prior parameters ``(a, b)``.
        confidence: Level of both intervals.
        seed: Root seed of the trials.
        initial_trials: Number of trials shown on load.
        locale: Catalogue locale (``"en"`` or ``"tr"``).

    Returns:
        The explainer; ``data["states"][n]`` holds the counts, posterior
        parameters, both intervals, and the likelihood and posterior
        densities on ``data["grid"]``.

    Raises:
        ValueError: If an argument is out of range or the locale is unknown.

    Equation:
        [EQ-REL-DATA-001], [EQ-DIST-BETA-001], [EQ-DIST-INV-001],
        [EQ-INF-CI-WILSON-001], [EQ-VIZ-LIK-001].

    References:
        [REF-GELMAN-ETAL-2013, Chapter 2], [REF-ATWOOD-NUREG-6823-2003],
        [REF-WILSON-1927].

    """
    code = check_locale(locale)
    say, text = _sayer(code), _texter(code)
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    if not 0 <= initial_trials <= trials:
        raise ValueError("initial_trials must lie between 0 and trials")
    outcomes = bernoulli_outcomes(probability, trials, seed)
    prior_law = Beta(float(prior[0]), float(prior[1]))
    tail = 0.5 * (1.0 - confidence)
    percent = round(100.0 * confidence, 6)
    prior_density = [prior_law.pdf(p) for p in _P_GRID]
    records: list[dict[str, Any]] = []
    for n in range(trials + 1):
        x = sum(outcomes[:n])
        if n == 0:
            posterior = prior_law
            credible = (prior_law.quantile(tail), prior_law.quantile(1.0 - tail))
            wilson: list[float] | None = None
        else:
            estimate = demand_failure_probability(
                x, n, prior=prior_law, confidence=confidence
            )
            posterior = estimate.posterior
            credible = estimate.credible_interval
            interval = wilson_score_interval(x, n, confidence=confidence)
            wilson = [interval.lower, interval.upper]
        likelihood = Beta(x + 1.0, n - x + 1.0)
        records.append(
            {
                "trials": n,
                "successes": x,
                "posterior": [posterior.alpha, posterior.beta],
                "posterior_mean": posterior.mean,
                "credible_interval": [float(credible[0]), float(credible[1])],
                "wilson_interval": wilson,
                "likelihood_density": [likelihood.pdf(p) for p in _P_GRID],
                "posterior_density": [posterior.pdf(p) for p in _P_GRID],
            }
        )
    data: dict[str, Any] = {
        "explainer": "bayesian-updating",
        "probability": probability,
        "prior": [prior_law.alpha, prior_law.beta],
        "confidence": confidence,
        "seed": seed,
        "outcomes": list(outcomes),
        "grid": list(_P_GRID),
        "prior_density": prior_density,
        "states": records,
        "equations": [
            "EQ-REL-DATA-001",
            "EQ-DIST-BETA-001",
            "EQ-DIST-INV-001",
            "EQ-VIZ-LIK-001",
            "EQ-INF-CI-WILSON-001",
            "EQ-RNG-SPLITMIX-001",
        ],
        "references": [
            "REF-GELMAN-ETAL-2013",
            "REF-ATWOOD-NUREG-6823-2003",
            "REF-WILSON-1927",
            "REF-STEELE-LEA-FLOOD-2014",
        ],
    }

    frame = Frame()
    tallest = max(
        max(prior_density),
        *(max(r["likelihood_density"] + r["posterior_density"]) for r in records),
    )
    top = float(math.ceil(tallest))
    step = 1.0 if top <= 6 else 2.0
    top = step * math.ceil(top / step)
    sx = Scale(0.0, 1.0, frame.x0, frame.x1)
    sy = Scale(0.0, top, frame.y1, frame.y0)
    a, b = prior_law.alpha, prior_law.beta

    slider_updates: list[Update] = []
    both: list[list[Update]] = []
    for record in records:
        n, x = record["trials"], record["successes"]
        slider_updates.append(
            {
                "attrs": {
                    "state": {"aria-valuetext": text("bayes.slider_value", n=n, x=x)},
                    "bayes-likelihood": {
                        "d": path(_P_GRID, record["likelihood_density"], sx, sy)
                    },
                    "bayes-posterior": {
                        "d": path(_P_GRID, record["posterior_density"], sx, sy)
                    },
                },
                "text": {"state-value": text("bayes.slider_value", n=n, x=x)},
            }
        )
        alpha, beta = record["posterior"]
        density: list[float] = record["posterior_density"]
        peak = _P_GRID[density.index(max(density))]
        row: list[Update] = []
        for interval_name in ("credible", "wilson"):
            bounds = record[f"{interval_name}_interval"]
            if bounds is None:
                row.append(
                    {
                        "attrs": {"bayes-band": {"visibility": "hidden"}},
                        "text": {
                            "live": text("bayes.live.wilson_none"),
                            "bayes-desc": text(
                                "bayes.chart.desc_none", mode=peak, p=probability
                            ),
                        },
                    }
                )
                continue
            lower, upper = bounds
            if interval_name == "credible":
                key = "bayes.live.credible_prior" if n == 0 else "bayes.live.credible"
                live = text(
                    key,
                    count=n,
                    successes=x,
                    alpha=alpha,
                    beta=beta,
                    percent=percent,
                    lower=lower,
                    upper=upper,
                    mean=record["posterior_mean"],
                )
            else:
                live = text(
                    "bayes.live.wilson",
                    count=n,
                    successes=x,
                    percent=percent,
                    lower=lower,
                    upper=upper,
                    share=x / n,
                )
            row.append(
                {
                    "attrs": {
                        "bayes-band": {
                            "visibility": "visible",
                            "x": number(sx(lower)),
                            "width": number(sx(upper) - sx(lower)),
                        }
                    },
                    "text": {
                        "live": live,
                        "bayes-desc": text(
                            f"bayes.chart.desc_{interval_name}",
                            count=n,
                            mode=peak,
                            percent=percent,
                            lower=lower,
                            upper=upper,
                            p=probability,
                        ),
                    },
                }
            )
        both.append(row)
    mode_updates: list[Update] = [
        {
            "attrs": {
                "bayes-band": {"class": "band s1"},
                "bayes-key-band": {"class": "swatch band s1"},
            },
            "text": {
                "bayes-key-band-text": text("bayes.mode.credible", percent=percent)
            },
        },
        {
            "attrs": {
                "bayes-band": {"class": "band s0"},
                "bayes-key-band": {"class": "swatch band s0"},
            },
            "text": {"bayes-key-band-text": text("bayes.mode.wilson", percent=percent)},
        },
    ]
    states = States(slider_updates, mode_updates, both, initial_trials, 0)
    initial = states.initial()

    truth_x = sx(probability)
    truth_label = text("bayes.key.truth", p=probability)
    title_end = _text_width(text("bayes.chart.y")) + 16.0
    if (
        truth_x + 6.0 + _text_width(truth_label) <= frame.width
        and truth_x + 6 > title_end
    ):
        label_attrs: dict[str, object] = {"x": number(truth_x + 6.0)}
    else:
        label_attrs = {"x": number(truth_x - 6.0), "text-anchor": "end"}
    body = (
        axes(
            frame,
            sx,
            sy,
            (0, 0.2, 0.4, 0.6, 0.8, 1.0),
            [step * k for k in range(int(top / step) + 1)],
            xtitle=say("bayes.chart.x"),
            ytitle=say("bayes.chart.y"),
            locale=code,
        )
        + dynamic(
            "rect",
            "bayes-band",
            initial,
            {"y": number(frame.y0), "height": number(frame.y1 - frame.y0)},
        )
        + tag(
            "path", {"class": "line prior", "d": path(_P_GRID, prior_density, sx, sy)}
        )
        + dynamic("path", "bayes-likelihood", initial, {"class": "line s0"})
        + dynamic("path", "bayes-posterior", initial, {"class": "line s1"})
        + tag(
            "line",
            {
                "class": "ref",
                "x1": number(truth_x),
                "x2": number(truth_x),
                "y1": number(frame.y0),
                "y2": number(frame.y1),
            },
        )
        + tag(
            "text",
            {"class": "lab-ref", "y": number(frame.y0 - 18), **label_attrs},
            html.escape(truth_label),
        )
    )
    key = _key(
        [
            ("prior", say("bayes.key.prior", a=a, b=b)),
            ("s0", say("bayes.key.likelihood")),
            ("s1", say("bayes.key.posterior")),
            ("ref", say("bayes.key.truth", p=probability)),
        ],
        "<li>"
        + dynamic("span", "bayes-key-band", initial, {"aria-hidden": "true"}, "")
        + dynamic("span", "bayes-key-band-text", initial)
        + "</li>",
    )
    figures = _figure(
        "bayes-heading",
        say("bayes.chart.heading"),
        svg_chart(
            "bayes",
            frame,
            say("bayes.chart.title"),
            html.escape(initial["text"]["bayes-desc"]),
            body,
        ),
        key,
    )
    controls = _slider(
        say("bayes.slider_label"), trials + 1, initial, initial_trials
    ) + _modes(
        say("bayes.mode_label"),
        [
            say("bayes.mode.credible", percent=percent),
            say("bayes.mode.wilson", percent=percent),
        ],
        0,
    )
    method = [
        say("bayes.method.trials", p=probability, seed=seed),
        say("bayes.method.posterior"),
        say("bayes.method.likelihood"),
        say("bayes.method.wilson"),
    ]
    title = message("bayes.title", code)
    page = _page(
        locale=code,
        title=html.escape(title),
        lede=say("bayes.lede", a=a, b=b, trials=trials, p=probability),
        controls=controls,
        figures=figures,
        method=method,
        equations=data["equations"],
        references=data["references"],
        data=data,
        states=states,
    )
    return Explainer("bayesian-updating-explainer", code, title, data, page)


# ---------------------------------------------------------------------------
# Monte Carlo error
# ---------------------------------------------------------------------------


def pi_estimate_traces(
    *,
    seeds: int = 4,
    checkpoints: Sequence[int] = MC_CHECKPOINTS,
    seed: int = DEFAULT_SEED,
    confidence: float = 0.95,
) -> tuple[tuple[MonteCarloEstimate, ...], ...]:
    """Return nested-prefix Monte Carlo estimates of ``pi`` for several seeds.

    Run ``k`` (``k = 1, ..., seeds``) draws uniform points from the child
    stream ``("seed", k)`` of ``RandomStream(seed)``; its contributions are
    ``4 I(U1^2 + U2^2 <= 1)`` and its trace is
    :func:`~probstat.monte_carlo.convergence_trace` at ``checkpoints``.

    Args:
        seeds: Number of independent runs (1 to 5).
        checkpoints: Strictly increasing sample sizes (at least 2).
        seed: Root seed.
        confidence: Level of the normal-approximation intervals.

    Returns:
        One trace per run.

    Raises:
        ValueError: If ``seeds`` or ``checkpoints`` are out of range.

    Equation:
        [EQ-MC-MEAN-001], [EQ-MC-SE-001], [EQ-MC-ROOTN-001],
        [EQ-RNG-SPLITMIX-001].

    References:
        [REF-ROBERT-CASELLA-2004, Chapters 3-4], [REF-STEELE-LEA-FLOOD-2014].

    """
    if not 1 <= seeds <= 5:
        raise ValueError("seeds must lie between 1 and 5")
    checked = _sizes(checkpoints, "checkpoints", 2)
    root = RandomStream(seed)
    traces = []
    for run in range(1, seeds + 1):
        generator = root.child("seed", run).generator()
        values = tuple(
            4.0 if generator.random() ** 2 + generator.random() ** 2 <= 1.0 else 0.0
            for _ in range(checked[-1])
        )
        traces.append(convergence_trace(values, checked, confidence=confidence))
    return tuple(traces)


def monte_carlo_error_explainer(
    *,
    seeds: int = 4,
    checkpoints: Sequence[int] = MC_CHECKPOINTS,
    seed: int = DEFAULT_SEED,
    confidence: float = 0.95,
    initial_size: int = 1_000,
    locale: str = "en",
) -> Explainer:
    """Return the Monte Carlo error explainer (estimates of ``pi``).

    The page draws, on a logarithmic sample-size axis, the running estimate
    of every run of :func:`pi_estimate_traces` in muted ink and the selected
    run with its confidence band. A slider moves along the checkpoints and
    buttons select the run.

    Args:
        seeds: Number of runs (one button each, 1 to 5).
        checkpoints: Sample sizes of the trace.
        seed: Root seed.
        confidence: Level of the confidence bands.
        initial_size: Sample size shown on load; one of ``checkpoints``.
        locale: Catalogue locale (``"en"`` or ``"tr"``).

    Returns:
        The explainer; ``data["runs"][k]`` holds the stream seed, estimates,
        standard errors, and interval limits of run ``k + 1``.

    Raises:
        ValueError: If an argument is out of range or the locale is unknown.

    Equation:
        [EQ-MC-MEAN-001], [EQ-MC-SE-001], [EQ-MC-ROOTN-001],
        [EQ-RNG-SPLITMIX-001].

    References:
        [REF-ROBERT-CASELLA-2004, Chapters 3-4], [REF-STEELE-LEA-FLOOD-2014].

    """
    code = check_locale(locale)
    say, text = _sayer(code), _texter(code)
    checked = _sizes(checkpoints, "checkpoints", 2)
    start = _index_of(checked, initial_size, "initial_size")
    traces = pi_estimate_traces(
        seeds=seeds, checkpoints=checked, seed=seed, confidence=confidence
    )
    root = RandomStream(seed)
    runs: list[dict[str, Any]] = [
        {
            "run": run,
            "stream_seed": root.child("seed", run).seed,
            "estimate": [item.estimate for item in trace],
            "standard_error": [item.standard_error for item in trace],
            "lower": [item.confidence_interval.lower for item in trace],
            "upper": [item.confidence_interval.upper for item in trace],
        }
        for run, trace in enumerate(traces, start=1)
    ]
    percent = round(100.0 * confidence, 6)
    data: dict[str, Any] = {
        "explainer": "monte-carlo-error",
        "seed": seed,
        "confidence": confidence,
        "checkpoints": list(checked),
        "true_value": math.pi,
        "runs": runs,
        "equations": [
            "EQ-MC-MEAN-001",
            "EQ-MC-SE-001",
            "EQ-MC-ROOTN-001",
            "EQ-RNG-SPLITMIX-001",
        ],
        "references": ["REF-ROBERT-CASELLA-2004", "REF-STEELE-LEA-FLOOD-2014"],
    }

    frame = Frame(right=112.0)
    sx = Scale(
        _nice(checked[0], up=False),
        _nice(checked[-1], up=True),
        frame.x0,
        frame.x1,
        log=True,
    )
    settled = [i for i, n in enumerate(checked) if n >= 50] or [len(checked) - 1]
    lowest = min(r["lower"][i] for r in runs for i in settled)
    highest = max(r["upper"][i] for r in runs for i in settled)
    y_low = min(math.floor(lowest * 2.0) / 2.0, math.pi - 0.5)
    y_high = max(math.ceil(highest * 2.0) / 2.0, math.pi + 0.5)
    sy = Scale(y_low, y_high, frame.y1, frame.y0)
    yticks = [y_low + 0.5 * k for k in range(int(round((y_high - y_low) / 0.5)) + 1)]

    slider_updates: list[Update] = []
    both: list[list[Update]] = []
    for index, n in enumerate(checked):
        guide = number(sx(n))
        slider_updates.append(
            {
                "attrs": {
                    "state": {"aria-valuetext": text("mc.slider_value", n=n)},
                    "mc-guide": {"x1": guide, "x2": guide},
                    "mc-dot": {"cx": guide},
                },
                "text": {"state-value": text("mc.slider_value", n=n)},
            }
        )
        estimates = [r["estimate"][index] for r in runs]
        row: list[Update] = []
        for run in runs:
            lower, upper = run["lower"][index], run["upper"][index]
            covers = lower <= math.pi <= upper
            sentence = text(
                "mc.live.covers" if covers else "mc.live.misses",
                n=n,
                run=run["run"],
                estimate=run["estimate"][index],
                half=(upper - lower) / 2.0,
                percent=percent,
                lower=lower,
                upper=upper,
            )
            spread_text = text(
                "mc.live.spread",
                count=len(runs),
                low=min(estimates),
                high=max(estimates),
            )
            row.append(
                {
                    "attrs": {
                        "mc-dot": {
                            "cy": number(
                                min(max(sy(run["estimate"][index]), frame.y0), frame.y1)
                            )
                        }
                    },
                    "text": {
                        "live": f"{sentence} {spread_text}",
                        "mc-desc": text(
                            "mc.chart.desc",
                            first=checked[0],
                            last=checked[-1],
                            run=run["run"],
                            percent=percent,
                            half_first=(run["upper"][0] - run["lower"][0]) / 2.0,
                            half_last=(run["upper"][-1] - run["lower"][-1]) / 2.0,
                            n=n,
                        ),
                    },
                }
            )
        both.append(row)
    right = frame.x1 + 8.0
    mode_updates: list[Update] = []
    for run in runs:
        positions = spread(
            [sy(math.pi), sy(run["estimate"][-1])],
            _LABEL_GAP,
            frame.y0 + 8,
            frame.y1 - 8,
        )
        attrs: dict[str, dict[str, Any]] = {
            "mc-band": {"d": band_path(checked, run["lower"], run["upper"], sx, sy)},
            "mc-selected": {"d": path(checked, run["estimate"], sx, sy)},
            "mc-pi-label": {"y": number(positions[0])},
            "mc-run-label": {"y": number(positions[1])},
        }
        mode_updates.append(
            {
                "attrs": attrs,
                "text": {"mc-run-label": text("mc.label.run", run=run["run"])},
            }
        )
    states = States(slider_updates, mode_updates, both, start, 0)
    initial = states.initial()

    clip = (
        '<clipPath id="mc-clip">'
        + tag(
            "rect",
            {
                "x": number(frame.x0),
                "y": number(frame.y0),
                "width": number(frame.x1 - frame.x0),
                "height": number(frame.y1 - frame.y0),
            },
        )
        + "</clipPath>"
    )
    context = "".join(
        tag(
            "path", {"class": "line context", "d": path(checked, r["estimate"], sx, sy)}
        )
        for r in runs
    )
    pi_y = number(sy(math.pi))
    body = (
        clip
        + axes(
            frame,
            sx,
            sy,
            _decades(sx.low, sx.high),
            yticks,
            xtitle=say("mc.chart.x"),
            ytitle=say("mc.chart.y"),
            locale=code,
            xspec=",g",
            yspec=".1f",
        )
        + '<g clip-path="url(#mc-clip)">'
        + dynamic("path", "mc-band", initial, {"class": "band s0"})
        + context
        + tag(
            "line",
            {
                "class": "ref",
                "x1": number(frame.x0),
                "x2": number(frame.x1),
                "y1": pi_y,
                "y2": pi_y,
            },
        )
        + dynamic("path", "mc-selected", initial, {"class": "line s0"})
        + "</g>"
        + dynamic(
            "line",
            "mc-guide",
            initial,
            {"class": "guide", "y1": number(frame.y0), "y2": number(frame.y1)},
        )
        + dynamic("circle", "mc-dot", initial, {"class": "dot s0", "r": 5})
        + dynamic(
            "text",
            "mc-pi-label",
            initial,
            {"class": "lab-ref", "x": number(right), "dy": "0.32em"},
            say("mc.label.pi", pi=math.pi),
        )
        + dynamic(
            "text",
            "mc-run-label",
            initial,
            {"class": "lab-s0", "x": number(right), "dy": "0.32em"},
        )
    )
    key = _key(
        [
            ("s0", say("mc.key.selected", percent=percent)),
            ("context", say("mc.key.others")),
            ("ref", say("mc.key.pi")),
        ]
    )
    figures = _figure(
        "mc-heading",
        say("mc.chart.heading"),
        svg_chart(
            "mc",
            frame,
            say("mc.chart.title"),
            html.escape(initial["text"]["mc-desc"]),
            body,
        ),
        key,
    )
    controls = _slider(say("mc.slider_label"), len(checked), initial, start) + _modes(
        say("mc.mode_label"), [say("mc.mode.run", run=r["run"]) for r in runs], 0
    )
    method = [
        say("mc.method.streams", seed=seed),
        say("mc.method.estimate"),
        say("mc.method.rate"),
    ]
    title = message("mc.title", code)
    page = _page(
        locale=code,
        title=html.escape(title),
        lede=say("mc.lede", percent=percent),
        controls=controls,
        figures=figures,
        method=method,
        equations=data["equations"],
        references=data["references"],
        data=data,
        states=states,
    )
    return Explainer("monte-carlo-error-explainer", code, title, data, page)


def write_explainers(
    directory: str | Path, *, locale: str = "en", seed: int = DEFAULT_SEED
) -> tuple[Path, ...]:
    """Write the three explainers with their default settings.

    File names are ``<name>.html`` for English and ``<name>.<locale>.html``
    otherwise.

    Args:
        directory: Output directory (created when missing).
        locale: Catalogue locale.
        seed: Root seed of the Bayesian and Monte Carlo explainers.

    Returns:
        Paths of the written pages.

    References:
        [REF-WHATWG-HTML], [EQ-LIM-003], [EQ-REL-DATA-001], [EQ-MC-ROOTN-001].

    """
    code = check_locale(locale)
    destination = Path(directory)
    suffix = ".html" if code == "en" else f".{code}.html"
    pages = (
        clt_explainer(locale=code),
        bayesian_updating_explainer(seed=seed, locale=code),
        monte_carlo_error_explainer(seed=seed, locale=code),
    )
    return tuple(page.write(destination / f"{page.name}{suffix}") for page in pages)
