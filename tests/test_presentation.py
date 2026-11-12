"""Contract tests of the overview presentation (ADR-0097).

The decks ``docs/presentation/index.html`` and ``index.tr.html`` show only
numbers that ``tools/build_presentation_data.py`` computes. These tests rerun
its data function and check that the stored ``deck-data.json`` still equals
it, that every number on a slide is a bound value with the right text, that
the decks load nothing from the network, and that the Turkish deck carries
the same slides and the same data as the English one.

When a test here fails after a change elsewhere in the repository (a new
test crossing a hundred, a changed benchmark file), rebuild the deck data:
``PYTHONPATH=src python tools/build_presentation_data.py``.
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
import sys
import unittest
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DECK = ROOT / "docs" / "presentation"
TOOL_PATH = ROOT / "tools" / "build_presentation_data.py"
DECKS = {"en": DECK / "index.html", "tr": DECK / "index.tr.html"}
RELATIVE = 1e-6
STANDARD_ERRORS = 4.0
REBUILD = (
    "deck data is stale: run PYTHONPATH=src python tools/build_presentation_data.py"
)


def load_tool() -> Any:
    """Import the deck builder from ``tools/``.

    Returns:
        The ``build_presentation_data`` module.

    References:
        [REF-PYTHON-DATAMODEL]. No mathematical equation is evaluated.

    """
    spec = importlib.util.spec_from_file_location("build_presentation_data", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TOOL = load_tool()


def mismatches(
    stored: Any, fresh: Any, stochastic: dict[str, float], path: str = ""
) -> list[str]:
    """List the differences between stored and freshly computed deck data.

    Strings, booleans, and integers must be equal. Reals agree to a relative
    tolerance of ``1e-6`` (platform libm), except Monte Carlo quantities,
    which agree within four of their standard errors.

    Args:
        stored: Value read from ``deck-data.json``.
        fresh: Value from the data function.
        stochastic: Dotted key to Monte Carlo standard error.
        path: Dotted key of the compared values.

    Returns:
        One line per difference.

    References:
        [REF-PYTHON-MATH], [REF-PROBSTAT-REPORTING].

    """
    if isinstance(stored, dict) and isinstance(fresh, dict):
        lines = []
        if set(stored) != set(fresh):
            lines.append(f"{path or '<root>'}: keys {sorted(set(stored) ^ set(fresh))}")
        for key in sorted(set(stored) & set(fresh)):
            lines += mismatches(
                stored[key], fresh[key], stochastic, f"{path}.{key}" if path else key
            )
        return lines
    if path in stochastic:
        if abs(float(stored) - float(fresh)) > STANDARD_ERRORS * stochastic[path]:
            return [
                f"{path}: {stored} vs {fresh} (standard error {stochastic[path]:.3g})"
            ]
        return []
    if (
        isinstance(stored, bool)
        or isinstance(fresh, bool)
        or isinstance(stored, str)
        or stored is None
    ):
        return [] if stored == fresh else [f"{path}: {stored!r} vs {fresh!r}"]
    if isinstance(stored, int) and isinstance(fresh, int):
        return [] if stored == fresh else [f"{path}: {stored} vs {fresh}"]
    if isinstance(stored, int | float) and isinstance(fresh, int | float):
        if math.isclose(stored, fresh, rel_tol=RELATIVE, abs_tol=1e-12):
            return []
    return [f"{path}: {stored!r} vs {fresh!r}"]


class _VisibleText(HTMLParser):
    """Collect, per slide, the text outside bound values and citation lists.

    References:
        [REF-WHATWG-HTML]. No mathematical equation is evaluated.

    """

    def __init__(self) -> None:
        """Start outside any slide.

        References:
            [REF-WHATWG-HTML]. No equation is evaluated.

        """
        super().__init__(convert_charrefs=True)
        self.free: list[list[str]] = []
        self.references: list[str] = []
        self._stack: list[tuple[str, bool]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Track whether the text that follows is bound or exempt.

        References:
            [REF-WHATWG-HTML]. No equation is evaluated.

        """
        values = dict(attrs)
        for name in ("src", "href"):
            if values.get(name):
                self.references.append(str(values[name]))
        if tag in ("img", "source", "br", "meta", "link", "input"):
            return
        if tag == "section" and "slayt" in (values.get("class") or "").split():
            self.free.append([])
        exempt = (
            "data-k" in values or "data-literal" in values or tag in ("script", "style")
        )
        self._stack.append((tag, exempt))

    def handle_endtag(self, tag: str) -> None:
        """Close the innermost element.

        References:
            [REF-WHATWG-HTML]. No equation is evaluated.

        """
        if self._stack:
            self._stack.pop()

    def handle_data(self, data: str) -> None:
        """Keep text that is neither bound nor exempt.

        References:
            [REF-WHATWG-HTML]. No equation is evaluated.

        """
        inside = any(tag == "section" for tag, _ in self._stack)
        if inside and self.free and not any(exempt for _, exempt in self._stack):
            self.free[-1].append(data)


def visible(text: str) -> _VisibleText:
    """Parse a deck and return its free text and references.

    References:
        [REF-WHATWG-HTML]. No equation is evaluated.

    """
    parser = _VisibleText()
    parser.feed(text)
    parser.close()
    return parser


# Module imports at the start of a line (comments that show usage do not count).
IMPORT = re.compile(r"^\s*import\s[^'\n]*from '(\./[^']+)'", re.MULTILINE)

# Identifiers and literal tokens that contain digits but are not deck numbers.
NOT_A_NUMBER = re.compile(
    r"(?:EQ|REF|ADR)-[A-Z0-9-]+|\(t,m,s\)|[A-Za-z]\d|\d[A-Za-z]|\[\d+\]"
)


class PresentationDataTests(unittest.TestCase):
    """``deck-data.json`` equals a fresh run of the data function."""

    fresh: dict[str, Any]
    stored: dict[str, Any]

    @classmethod
    def setUpClass(cls) -> None:
        """Run the data function once for every test in the class.

        References:
            [REF-PROBSTAT-REPORTING]. No equation is evaluated.

        """
        cls.stored = json.loads(TOOL.DATA_PATH.read_text(encoding="utf-8"))
        cls.fresh = TOOL.build_data()

    def test_stored_data_equals_a_fresh_run(self) -> None:
        """Seeded and exact values match; Monte Carlo values within their error."""
        stochastic = {**self.fresh["stochastic"], **self.stored["stochastic"]}
        lines = mismatches(
            {k: v for k, v in self.stored.items() if k != "stochastic"},
            {k: v for k, v in self.fresh.items() if k != "stochastic"},
            stochastic,
        )
        self.assertEqual(lines, [], REBUILD)
        self.assertEqual(
            set(self.stored["stochastic"]), set(self.fresh["stochastic"]), REBUILD
        )

    def test_repository_counts_are_reread(self) -> None:
        """The floors shown on the deck follow the repository as it is now."""
        counts = TOOL.repository_counts()
        self.assertEqual(self.stored["repository"], counts, REBUILD)
        self.assertEqual(counts["tests_floor"] % TOOL.FLOORS["tests"], 0)
        self.assertLessEqual(counts["tests_floor"], TOOL.count_tests())
        self.assertGreater(TOOL.count_tests(), counts["tests_floor"] - 1)

    def test_exact_reference_is_independent_of_the_estimators(self) -> None:
        """The quadrature reference agrees with SORM and lies in the MC interval."""
        reliability = self.fresh["study"]["reliability"]
        reference = reliability["reference"]
        monte_carlo = reliability["methods"]["mc"]
        self.assertLess(monte_carlo["ci95_low"], reference)
        self.assertLess(reference, monte_carlo["ci95_high"])
        self.assertLess(reliability["sorm_relative_error"], 0.01)


class PresentationHtmlTests(unittest.TestCase):
    """The decks bind every number and reference only local files."""

    data: dict[str, Any]
    texts: dict[str, str]

    @classmethod
    def setUpClass(cls) -> None:
        """Read both decks and the stored data.

        References:
            [REF-WHATWG-HTML]. No equation is evaluated.

        """
        cls.data = json.loads(TOOL.DATA_PATH.read_text(encoding="utf-8"))
        cls.texts = {
            lang: path.read_text(encoding="utf-8") for lang, path in DECKS.items()
        }

    def test_bound_values_resolve_and_show_the_formatted_data(self) -> None:
        """Every ``data-k`` key exists and its text is the formatted value."""
        for lang, text in self.texts.items():
            self.assertEqual(TOOL.deck_language(text), lang)
            bound = list(TOOL.bound_values(text))
            self.assertGreater(len(bound), 50, lang)
            for key, fmt, shown in bound:
                with self.subTest(lang=lang, key=key):
                    expected = TOOL.format_value(TOOL.lookup(self.data, key), fmt, lang)
                    self.assertEqual(shown, expected, REBUILD)

    def test_no_number_is_typed_outside_a_bound_value(self) -> None:
        """Slide text and notes carry digits only inside ``data-k`` elements."""
        for lang, text in self.texts.items():
            for index, parts in enumerate(visible(text).free, start=1):
                free = NOT_A_NUMBER.sub(" ", " ".join(parts))
                with self.subTest(lang=lang, slide=index):
                    self.assertEqual(re.findall(r"\d+", free), [])

    def test_decks_reference_only_local_files(self) -> None:
        """No ``src``/``href`` leaves the folder; every referenced file exists."""
        for lang, text in self.texts.items():
            for reference in visible(text).references:
                with self.subTest(lang=lang, reference=reference):
                    self.assertNotRegex(reference, r"^(?:[a-z]+:)?//")
                    if reference.startswith("data:"):
                        continue
                    self.assertTrue(
                        (DECK / reference.split("#")[0]).is_file(), reference
                    )
            for module in re.findall(IMPORT, text):
                self.assertTrue((DECK / module).is_file(), module)
        for path in (DECK / "runtime").iterdir():
            source = path.read_text(encoding="utf-8")
            self.assertNotRegex(
                source,
                r"(?:src|href)=[\"']https?:|@import|url\(\s*[\"']?https?:",
                path.name,
            )
            for module in re.findall(IMPORT, source):
                self.assertTrue(
                    (path.parent / module).is_file(), f"{path.name}: {module}"
                )

    def test_turkish_deck_has_the_same_slides_and_data(self) -> None:
        """Same slide count, page types, slide ids, and bound keys per slide."""
        english = TOOL.extract_slides(self.texts["en"])
        turkish = TOOL.extract_slides(self.texts["tr"])
        self.assertEqual(len(english), len(turkish))
        self.assertTrue(12 <= len(english) <= 16, len(english))
        for en_slide, tr_slide in zip(english, turkish, strict=True):
            with self.subTest(slide=en_slide["id"]):
                self.assertEqual(en_slide["id"], tr_slide["id"])
                self.assertEqual(en_slide["type"], tr_slide["type"])
                self.assertEqual(sorted(en_slide["keys"]), sorted(tr_slide["keys"]))

    def test_every_slide_has_a_type_a_headline_and_speaker_notes(self) -> None:
        """Each slide names its page type, states a headline, and has notes."""
        for lang, text in self.texts.items():
            slides = TOOL.extract_slides(text)
            types = [slide["type"] for slide in slides]
            self.assertEqual(types[0], "kapak")
            self.assertEqual(types.count("kapanis"), 1)
            for slide in slides:
                with self.subTest(lang=lang, slide=slide["id"]):
                    self.assertTrue(slide["type"])
                    self.assertTrue(slide["headline"])
                    self.assertGreaterEqual(len(slide["notes"]), 60)

    def test_every_chart_exists_in_both_themes_and_passed_its_audit(self) -> None:
        """Each slide chart has a light and a dark PNG and a clean audit."""
        audit = json.loads(
            (DECK / "figures" / "audit.json").read_text(encoding="utf-8")
        )
        for name in TOOL.CHARTS:
            for lang in DECKS:
                for theme in ("light", "dark"):
                    stem = f"{name}-{lang}-{theme}"
                    with self.subTest(figure=stem):
                        self.assertTrue((DECK / "figures" / f"{stem}.png").is_file())
                        self.assertEqual(audit[stem]["errors"], 0)
                        self.assertIn(f"figures/{stem}.png", self.texts[lang])


class FormatTests(unittest.TestCase):
    """Number formats match the deck's two languages."""

    def test_english_and_turkish_formats(self) -> None:
        """Separators, percentage sign position, ratios, and floors."""
        cases = [
            (10000, "int", "10,000", "10.000"),
            (2100, "plus", "2,100+", "2.100+"),
            (0.0377, "pct2", "3.77%", "%3,77"),
            (232.56, "x0", "×233", "×233"),
            (0.12415, "f3", "0.124", "0,124"),
            (34.628, "musd1", "$34.6M", "34,6 M$"),
            (-1.5, "f1", "−1.5", "−1,5"),
        ]
        for value, fmt, english, turkish in cases:
            with self.subTest(fmt=fmt):
                self.assertEqual(TOOL.format_value(value, fmt, "en"), english)
                self.assertEqual(TOOL.format_value(value, fmt, "tr"), turkish)
        with self.assertRaises(ValueError):
            TOOL.format_value(1.0, "pct", "en")


if __name__ == "__main__":
    unittest.main()
