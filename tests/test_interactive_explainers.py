"""Tests for the offline interactive explainers (ADR-0085).

Each page must be self-contained (no external script, stylesheet, font, or
URL), embed library outputs that equal fresh library calls, label every
control, announce its numbers in an ``aria-live`` region, give every chart
a title and description, define light and dark palette tokens, and keep its
script to selecting precomputed states.
"""

from __future__ import annotations

import functools
import json
import math
import re
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path

from probstat.distributions import Beta
from probstat.inference import wilson_score_interval
from probstat.monte_carlo import convergence_trace
from probstat.probability import (
    clt_error_profile,
    exponential_summand,
    standardized_mean_cdf,
)
from probstat.probability.limit_theorems import standard_normal_cdf
from probstat.reliability import demand_failure_probability
from probstat.visualization import GRAPHITE, PORCELAIN
from probstat.visualization.accessibility import contrast_ratio
from probstat.visualization.interactive import (
    Explainer,
    bayesian_updating_explainer,
    bernoulli_outcomes,
    clt_explainer,
    monte_carlo_error_explainer,
    pi_estimate_traces,
    write_explainers,
)


class _Document(HTMLParser):
    """Collect elements, ids, and script blocks of one page."""

    def __init__(self) -> None:
        """Start with empty collections."""
        super().__init__(convert_charrefs=True)
        self.elements: list[tuple[str, dict[str, str]]] = []
        self.ids: dict[str, tuple[str, dict[str, str]]] = {}
        self.scripts: list[tuple[dict[str, str], str]] = []
        self.visible: list[str] = []
        self._script: dict[str, str] | None = None
        self._style = False

    def handle_starttag(self, tag, attrs):
        """Record an element and its id."""
        attributes = {key: value or "" for key, value in attrs}
        self.elements.append((tag, attributes))
        if "id" in attributes:
            self.ids[attributes["id"]] = (tag, attributes)
        if tag == "script":
            self._script = attributes
        if tag == "style":
            self._style = True

    def handle_startendtag(self, tag, attrs):
        """Record an empty element."""
        self.handle_starttag(tag, attrs)
        if tag == "script":
            self._script = None

    def handle_endtag(self, tag):
        """Close script and style blocks."""
        if tag == "script":
            self._script = None
        if tag == "style":
            self._style = False

    def handle_data(self, data):
        """Keep script text apart from visible text."""
        if self._script is not None:
            self.scripts.append((self._script, data))
        elif not self._style:
            self.visible.append(data)

    def json(self, element_id: str) -> object:
        """Return the parsed JSON block with the given id."""
        for attributes, text in self.scripts:
            if attributes.get("id") == element_id:
                return json.loads(text)
        raise KeyError(element_id)


def _parse(page: str) -> _Document:
    """Return the parsed document."""
    document = _Document()
    document.feed(page)
    return document


@functools.cache
def _pages(locale: str) -> dict[str, Explainer]:
    """Return the three default explainers of a locale, computed once per run."""
    return {
        "clt": clt_explainer(locale=locale),
        "bayes": bayesian_updating_explainer(locale=locale),
        "mc": monte_carlo_error_explainer(locale=locale),
    }


class ExplainerFixture(unittest.TestCase):
    """Share the English and Turkish explainers between the test classes."""

    @classmethod
    def setUpClass(cls) -> None:
        """Bind the cached pages and their parsed documents."""
        cls.pages = _pages("en")
        cls.turkish = _pages("tr")
        cls.documents = {name: _parse(page.html) for name, page in cls.pages.items()}


class SelfContainedTests(ExplainerFixture):
    """Pages need nothing outside the file."""

    def test_no_external_resources_or_urls(self) -> None:
        """No ``http(s)`` URL, ``<script src``, stylesheet link, or font import."""
        for name, page in {
            **self.pages,
            **{f"{k}.tr": v for k, v in self.turkish.items()},
        }.items():
            with self.subTest(page=name):
                text = page.html
                self.assertNotRegex(text, r"https?://")
                self.assertNotRegex(text, r"<script[^>]*\bsrc=")
                self.assertNotIn("<link", text)
                self.assertNotIn("@import", text)
                self.assertNotIn("@font-face", text)
                self.assertEqual(re.findall(r"url\((?!#)", text), [])
                self.assertNotIn("fetch(", text)
                self.assertNotIn("XMLHttpRequest", text)

    def test_script_only_selects_precomputed_states(self) -> None:
        """The inline script evaluates no mathematics."""
        for name, document in self.documents.items():
            with self.subTest(page=name):
                code = [
                    text
                    for attributes, text in document.scripts
                    if attributes.get("type") != "application/json"
                ]
                self.assertEqual(len(code), 1)
                self.assertNotIn("Math.", code[0])
                self.assertNotRegex(code[0], r"\b(exp|log|sqrt|pow)\(")

    def test_written_files_match_the_pages(self) -> None:
        """``write_explainers`` writes the three default pages."""
        with tempfile.TemporaryDirectory() as directory:
            english = write_explainers(directory)
            self.assertEqual(
                [path.name for path in english],
                [
                    "clt-explainer.html",
                    "bayesian-updating-explainer.html",
                    "monte-carlo-error-explainer.html",
                ],
            )
            self.assertEqual(
                Path(english[0]).read_text(encoding="utf-8"), self.pages["clt"].html
            )


class EmbeddedDataTests(ExplainerFixture):
    """The embedded JSON equals the library outputs."""

    def test_embedded_json_equals_explainer_data(self) -> None:
        """The ``explainer-data`` block round-trips to ``Explainer.data``."""
        for name, document in self.documents.items():
            with self.subTest(page=name):
                self.assertEqual(
                    document.json("explainer-data"),
                    json.loads(json.dumps(self.pages[name].data)),
                )

    def test_clt_numbers_come_from_the_library(self) -> None:
        """Distances, bounds, and distribution-function gaps equal fresh calls."""
        data = self.documents["clt"].json("explainer-data")
        summand = exponential_summand(1.0)
        rows = clt_error_profile(summand, data["sizes"])
        self.assertEqual(data["kolmogorov_normal"], [r.kolmogorov_normal for r in rows])
        self.assertEqual(data["berry_esseen"], [r.berry_esseen for r in rows])
        for index in (0, 12, len(data["sizes"]) - 1):
            n = data["sizes"][index]
            expected = [
                standardized_mean_cdf(summand, n, z) - standard_normal_cdf(z)
                for z in data["z"]
            ]
            self.assertEqual(data["cdf_difference"][index], expected)
            gap = max(abs(value) for value in expected)
            self.assertLessEqual(gap, data["kolmogorov_normal"][index] + 1e-12)
        self.assertTrue(
            all(
                k < b
                for k, b in zip(
                    data["kolmogorov_normal"], data["berry_esseen"], strict=True
                )
            )
        )

    def test_bayes_numbers_come_from_the_library(self) -> None:
        """Posteriors, both intervals, and densities equal fresh library calls."""
        data = self.documents["bayes"].json("explainer-data")
        outcomes = bernoulli_outcomes(data["probability"], 40, data["seed"])
        self.assertEqual(data["outcomes"], list(outcomes))
        prior = Beta(*data["prior"])
        for record in data["states"][1:]:
            n, x = record["trials"], record["successes"]
            self.assertEqual(x, sum(outcomes[:n]))
            estimate = demand_failure_probability(x, n, prior=prior, confidence=0.95)
            self.assertEqual(
                record["posterior"], [estimate.posterior.alpha, estimate.posterior.beta]
            )
            self.assertEqual(
                record["credible_interval"], list(estimate.credible_interval)
            )
            wilson = wilson_score_interval(x, n, confidence=0.95)
            self.assertEqual(record["wilson_interval"], [wilson.lower, wilson.upper])
            likelihood = Beta(x + 1.0, n - x + 1.0)
            self.assertEqual(
                record["likelihood_density"][::25],
                [likelihood.pdf(p) for p in data["grid"][::25]],
            )
        first = data["states"][0]
        self.assertIsNone(first["wilson_interval"])
        tail = 0.5 * (1.0 - 0.95)
        self.assertEqual(
            first["credible_interval"],
            [prior.quantile(tail), prior.quantile(1.0 - tail)],
        )
        self.assertEqual(data["prior_density"], [prior.pdf(p) for p in data["grid"]])

    def test_monte_carlo_numbers_come_from_the_library(self) -> None:
        """Every run equals ``convergence_trace`` of the same seeded draws."""
        data = self.documents["mc"].json("explainer-data")
        traces = pi_estimate_traces(seed=data["seed"], checkpoints=data["checkpoints"])
        self.assertEqual(data["true_value"], math.pi)
        for run, trace in zip(data["runs"], traces, strict=True):
            self.assertEqual(run["estimate"], [item.estimate for item in trace])
            self.assertEqual(
                run["standard_error"], [item.standard_error for item in trace]
            )
            self.assertEqual(
                run["lower"], [item.confidence_interval.lower for item in trace]
            )
        values = (4.0, 0.0, 4.0, 4.0)
        self.assertEqual(convergence_trace(values, (2, 4))[1].estimate, 3.0)


class AccessibilityTests(ExplainerFixture):
    """Landmarks, labels, live region, text alternatives, and colour tokens."""

    def test_language_and_landmarks(self) -> None:
        """``lang``, skip link, header, main, footer, and labelled sections."""
        for name, page in self.pages.items():
            document = self.documents[name]
            tags = [tag for tag, _ in document.elements]
            with self.subTest(page=name):
                self.assertTrue(page.html.startswith('<!doctype html><html lang="en">'))
                self.assertTrue(
                    self.turkish[name].html.startswith(
                        '<!doctype html><html lang="tr">'
                    )
                )
                for landmark in ("header", "main", "footer"):
                    self.assertEqual(tags.count(landmark), 1)
                skip = [a for tag, a in document.elements if tag == "a"][0]
                self.assertEqual(skip["href"], "#explainer")
                self.assertIn("explainer", document.ids)
                for tag, attributes in document.elements:
                    if tag == "section":
                        self.assertIn(attributes["aria-labelledby"], document.ids)

    def test_controls_are_native_and_labelled(self) -> None:
        """The range input is labelled; buttons state ``aria-pressed``."""
        for name, document in self.documents.items():
            with self.subTest(page=name):
                tag, slider = document.ids["state"]
                self.assertEqual((tag, slider["type"]), ("input", "range"))
                labels = [a for t, a in document.elements if t == "label"]
                self.assertIn("state", [a.get("for") for a in labels])
                self.assertTrue(slider["aria-valuetext"])
                buttons = [a for t, a in document.elements if t == "button"]
                self.assertLessEqual(len(buttons), 5)
                for button in buttons:
                    self.assertEqual(button["type"], "button")
                    self.assertIn(button["aria-pressed"], ("true", "false"))
                if buttons:
                    groups = [
                        a for t, a in document.elements if a.get("role") == "group"
                    ]
                    self.assertIn(groups[0]["aria-labelledby"], document.ids)
                    pressed = [b for b in buttons if b["aria-pressed"] == "true"]
                    self.assertEqual(len(pressed), 1)
        self.assertEqual(
            len([1 for t, _ in self.documents["mc"].elements if t == "button"]), 4
        )

    def test_live_region_states_the_numbers(self) -> None:
        """An ``aria-live`` region holds the initial sentence with its numbers."""
        clt = self.documents["clt"]
        tag, live = clt.ids["live"]
        self.assertEqual(live["aria-live"], "polite")
        text = "".join(clt.visible)
        self.assertIn("At n = 30 the exact Kolmogorov distance", text)
        self.assertIn("0.0243", text)
        turkish = _parse(self.turkish["clt"].html)
        self.assertIn("0,0243", "".join(turkish.visible))
        self.assertIn("n = 30 için", "".join(turkish.visible))

    def test_every_chart_has_a_text_alternative(self) -> None:
        """Each SVG is an image labelled by a non-empty title and description."""
        for name, document in self.documents.items():
            charts = [a for tag, a in document.elements if tag == "svg"]
            with self.subTest(page=name):
                self.assertGreaterEqual(len(charts), 1)
                for chart in charts:
                    self.assertEqual(chart["role"], "img")
                    title, description = chart["aria-labelledby"].split()
                    self.assertEqual(document.ids[title][0], "title")
                    self.assertEqual(document.ids[description][0], "desc")
                    view = document.json("explainer-view")
                    texts = [
                        update.get("text", {}).get(description)
                        for row in view["both"]
                        for update in row
                    ] + [
                        update.get("text", {}).get(description)
                        for update in view["slider"]
                    ]
                    self.assertTrue(any(texts), description)

    def test_both_colour_schemes_are_defined_from_the_palettes(self) -> None:
        """Light tokens are the porcelain palette, dark tokens the graphite one."""
        page = self.pages["clt"].html
        style = page[page.index("<style>") : page.index("</style>")]
        light, _, dark = style.partition("@media (prefers-color-scheme: dark)")
        for palette, block in ((PORCELAIN, light), (GRAPHITE, dark)):
            with self.subTest(palette=palette.name):
                for token, colour in (
                    ("canvas", palette.canvas),
                    ("ink", palette.ink),
                    ("muted", palette.muted),
                    ("accent", palette.accent),
                    ("s0", palette.series[0]),
                    ("s1", palette.series[1]),
                ):
                    self.assertIn(f"--{token}: {colour}", block)
                for label in ("l0", "l1"):
                    colour = re.search(rf"--{label}: (#[0-9A-Fa-f]{{6}})", block)
                    assert colour is not None
                    self.assertGreaterEqual(
                        contrast_ratio(colour.group(1), palette.canvas), 4.5
                    )
        self.assertIn(":focus-visible", style)
        self.assertIn("min-height: 44px", style)

    def test_citations_are_visible_on_the_page(self) -> None:
        """Every ``EQ-*`` and ``REF-*`` key of the data appears in the page text."""
        for name, document in self.documents.items():
            text = "".join(document.visible)
            data = self.pages[name].data
            with self.subTest(page=name):
                for key in [*data["equations"], *data["references"]]:
                    self.assertIn(key, text)


class StateTests(ExplainerFixture):
    """The precomputed states address existing elements and cover every input."""

    def test_states_cover_the_slider_and_modes(self) -> None:
        """One slider state per position, one mode state per button."""
        for name, document in self.documents.items():
            view = document.json("explainer-view")
            _, slider = document.ids["state"]
            buttons = [a for t, a in document.elements if t == "button"]
            modes = max(1, len(buttons))
            with self.subTest(page=name):
                self.assertEqual(len(view["slider"]), int(slider["max"]) + 1)
                self.assertEqual(len(view["mode"]), modes)
                self.assertEqual(len(view["both"]), len(view["slider"]))
                self.assertTrue(all(len(row) == modes for row in view["both"]))

    def test_states_address_existing_elements(self) -> None:
        """Every id a state updates exists in the document."""
        for name, document in self.documents.items():
            view = document.json("explainer-view")
            updates = [
                *view["slider"],
                *view["mode"],
                *(u for row in view["both"] for u in row),
            ]
            with self.subTest(page=name):
                for update in updates:
                    for key in [*update.get("attrs", {}), *update.get("text", {})]:
                        self.assertIn(key, document.ids)

    def test_initial_markup_equals_the_initial_state(self) -> None:
        """Without script the page shows the default state's sentence."""
        page = self.pages["mc"]
        document = self.documents["mc"]
        view = document.json("explainer-view")
        _, slider = document.ids["state"]
        index = int(slider["value"])
        live = view["both"][index][view["initialMode"]]["text"]["live"]
        self.assertIn(live, "".join(document.visible))
        self.assertEqual(page.data["checkpoints"][index], 1_000)

    def test_invalid_arguments_are_rejected(self) -> None:
        """Unknown locales and impossible settings raise."""
        with self.assertRaises(ValueError):
            clt_explainer(locale="de")
        with self.assertRaises(ValueError):
            clt_explainer(initial_size=7)
        with self.assertRaises(ValueError):
            monte_carlo_error_explainer(seeds=6)
        with self.assertRaises(ValueError):
            bayesian_updating_explainer(initial_trials=41)
        with self.assertRaises(ValueError):
            bernoulli_outcomes(1.5, 10, 1)


if __name__ == "__main__":
    unittest.main()
