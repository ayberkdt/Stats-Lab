"""Tests for the message catalogue and the localized study reports (ADR-0085).

The English output of :meth:`UQStudy.report` and
:func:`probstat.visualization.study.study_html` must be byte-identical to
the output recorded before the catalogue existed. The recorded files in
``tests/fixtures/`` were written by the code as it stood before the change,
from the deterministic study built by :func:`fixture_study`, which reaches
every branch of both templates (failed runs, cache hits, budget, aborted
stage, error, settings, warnings, non-finite values, a copula, more than
seven stages, and a surrogate child study).
"""

from __future__ import annotations

import contextlib
import math
import re
import string
import unittest
from collections.abc import Iterator
from pathlib import Path
from unittest import mock

from probstat.distributions import LogNormal, Normal
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.uq import QuantityOfInterest
from probstat.workflow import UQStudy
from probstat.workflow.study import StageRecord

FIXTURES = Path(__file__).parent / "fixtures"
CREATED = "2026-11-11T09:30:00+00:00"


def _margin(parameters):
    """Return the safety margin ``R - S``."""
    return {"g": parameters["R"] - parameters["S"]}


def _model(correlated: bool) -> ProbabilisticModel:
    """Return the two-input capacity model, optionally with a Gaussian copula."""
    inputs = (
        RandomInput("R", LogNormal(math.log(5.0), 0.1)),
        RandomInput("S", Normal(3.0, 0.8)),
    )
    joint = (
        JointInput(inputs, correlation=((1.0, 0.3), (0.3, 1.0)))
        if correlated
        else JointInput.independent(*inputs)
    )
    return ProbabilisticModel(
        joint, _margin, (QuantityOfInterest("g", "margin", "kN"),), name="capacity"
    )


def _record(stage: str, label: str, **fields) -> StageRecord:
    """Return a synthetic stage record with a fixed seed and law hash."""
    defaults = {
        "summary": {"mean": 1.2345678, "count": "12/20, [0.4, 0.8]"},
        "evaluations": 1000,
        "seed": 4242,
        "result": None,
        "input_law_hash": "a1b2c3d4",
    }
    defaults.update(fields)
    return StageRecord(stage, label, **defaults)


def fixture_study() -> UQStudy:
    """Return a study whose records reach every branch of both report templates.

    The records are synthetic, so the reports do not depend on floating-point
    results of any stage, and the creation time is fixed.
    """
    study = UQStudy(
        _model(True), seed=20261111, name="report <fixture>", evaluation_budget=50_000
    )
    study._created = CREATED
    records = [
        _record(
            "propagation",
            "Monte Carlo propagation of g",
            summary={"g.mean": 1.97531, "g.sd": 0.876, "P(g < 0)": 0.0123},
            evaluations=4096,
            settings={"size": 4096, "design": "sobol", "level": 0.95},
        ),
        _record(
            "sensitivity",
            "Sobol indices of g",
            failed_evaluations=17,
            cache_hits=256,
            settings={"names": ("R", "S"), "bootstrap": 40},
            warnings=("bootstrap interval of S_R crosses zero",),
        ),
        _record(
            "reliability",
            "FORM & MC for g < 0",
            status="aborted",
            error="RuntimeError: FORM did not converge",
            evaluations=33,
            summary={},
        ),
        _record(
            "risk",
            "Expected loss",
            summary={"loss": math.inf, "VaR[95%]": 12_345.678, "label": "a<b"},
            input_law="posterior #1 (gaussian_copula)",
            input_law_hash="ffee0011",
        ),
        _record("screening", "Morris screening", evaluations=60),
        _record("validation", "Coverage check", cache_hits=1),
        _record("extremes", "Peaks over threshold"),
        _record("design", "Expected information gain"),
        _record("emulation", "PCE surrogate of g", evaluations=120),
    ]
    study._records.extend(records)
    child = UQStudy(
        _model(False), seed=7, name="capacity surrogate", evaluation_kind="surrogate"
    )
    child._created = CREATED
    child._records.extend(
        [
            _record("propagation", "Surrogate propagation", evaluations=1_000_000),
            _record("reliability", "Surrogate MC", failed_evaluations=2),
        ]
    )
    study._children.append((9, child))
    return study


@contextlib.contextmanager
def fixed_provenance() -> Iterator[None]:
    """Fix the library, Python, and platform strings of the provenance."""
    with (
        mock.patch("probstat.__version__", "0.0.test"),
        mock.patch("platform.python_version", return_value="3.12.0"),
        mock.patch("platform.python_implementation", return_value="CPython"),
        mock.patch("platform.platform", return_value="TestOS-1.0"),
        mock.patch(
            "probstat.visualization.theme.matplotlib_available", return_value=False
        ),
    ):
        yield


def _read(name: str) -> bytes:
    """Return the bytes of a recorded fixture."""
    return (FIXTURES / name).read_bytes()


class EnglishOutputIsUnchanged(unittest.TestCase):
    """The default English reports equal the output recorded before ADR-0085."""

    def test_markdown_report_is_byte_identical(self) -> None:
        """``UQStudy.report()`` reproduces the recorded report byte for byte."""
        with fixed_provenance():
            report = fixture_study().report()
            explicit = fixture_study().report(locale="en")
        self.assertEqual(report.encode("utf-8"), _read("study_report_en.txt"))
        self.assertEqual(explicit, report)

    def test_html_report_is_byte_identical(self) -> None:
        """``study_html`` reproduces the recorded page byte for byte."""
        from probstat.visualization.study import study_html

        with fixed_provenance():
            page = study_html(fixture_study())
        self.assertEqual(page.encode("utf-8"), _read("study_report_en.html"))

    def test_locale_is_never_read_from_the_environment(self) -> None:
        """Turkish environment variables leave the default report English."""
        from probstat.visualization.study import study_html

        environment = {
            "LANG": "tr_TR.UTF-8",
            "LC_ALL": "tr_TR.UTF-8",
            "LANGUAGE": "tr",
            "PROBSTAT_LOCALE": "tr",
        }
        with mock.patch.dict("os.environ", environment), fixed_provenance():
            report = fixture_study().report()
            page = study_html(fixture_study())
        self.assertEqual(report.encode("utf-8"), _read("study_report_en.txt"))
        self.assertEqual(page.encode("utf-8"), _read("study_report_en.html"))


class CatalogueTests(unittest.TestCase):
    """Completeness and consistency of the English and Turkish catalogues."""

    def setUp(self) -> None:
        """Load both catalogues."""
        from probstat.foundations.messages import catalogue

        self.english = catalogue("en")
        self.turkish = catalogue("tr")

    def test_turkish_covers_every_key(self) -> None:
        """Both catalogues have exactly the same keys."""
        self.assertEqual(sorted(self.english), sorted(self.turkish))

    def test_plural_forms_and_placeholders_match(self) -> None:
        """Each entry has the same plural forms and placeholders in both locales."""
        from probstat.foundations.messages import placeholders

        for key, english in self.english.items():
            turkish = self.turkish[key]
            with self.subTest(key=key):
                if isinstance(english, str):
                    self.assertIsInstance(turkish, str)
                    self.assertEqual(placeholders(english), placeholders(turkish))
                    continue
                self.assertNotIsInstance(turkish, str)
                self.assertEqual(sorted(english), ["one", "other"])
                self.assertEqual(sorted(turkish), ["one", "other"])
                for form in ("one", "other"):
                    self.assertEqual(
                        placeholders(english[form]), placeholders(turkish[form])
                    )

    def test_entries_are_well_formed_templates(self) -> None:
        """Every entry parses as a format string and none is empty."""
        for locale, entries in (("en", self.english), ("tr", self.turkish)):
            for key, entry in entries.items():
                forms = [entry] if isinstance(entry, str) else list(entry.values())
                for form in forms:
                    with self.subTest(locale=locale, key=key):
                        self.assertTrue(form.strip())
                        list(string.Formatter().parse(form))

    def test_equation_keys_in_entries_are_registered(self) -> None:
        """``EQ-*`` keys cited in catalogue prose exist in the registry."""
        registry = (
            Path(__file__).parents[1] / "docs" / "equation-registry.md"
        ).read_text(encoding="utf-8")
        cited: set[str] = set()
        for entries in (self.english, self.turkish):
            for key, entry in entries.items():
                forms = [entry] if isinstance(entry, str) else list(entry.values())
                text = " ".join([key, *forms])
                cited.update(re.findall(r"EQ-[A-Z0-9-]*[0-9]", text))
        self.assertTrue(cited)
        missing = sorted(key for key in cited if f"### {key} " not in registry)
        self.assertEqual(missing, [])


class NumberAndPluralTests(unittest.TestCase):
    """Locale number formatting, plural selection, and escaping."""

    def test_turkish_uses_decimal_comma_and_group_dot(self) -> None:
        """CLDR Turkish symbols: decimal ``,`` and group ``.``."""
        from probstat.foundations.messages import format_number

        self.assertEqual(format_number(0.0243, ".3g", "tr"), "0,0243")
        self.assertEqual(format_number(1234567.891, ",.2f", "tr"), "1.234.567,89")
        self.assertEqual(format_number(1234567.891, ",.2f", "en"), "1,234,567.89")
        self.assertEqual(format_number(4096, ",", "tr"), "4.096")
        self.assertEqual(format_number(20261111, "", "tr"), "20261111")
        self.assertEqual(format_number(math.inf, ".6g", "tr"), "inf")

    def test_message_localizes_numbers_but_not_text(self) -> None:
        """Numeric fields get locale separators; string fields stay as given."""
        from probstat.foundations.messages import message

        self.assertEqual(
            message("report.budget", "tr", budget=50_000, remaining=1_234),
            "Değerlendirme bütçesi: 50.000 (kalan 1.234)",
        )
        self.assertEqual(
            message("report.input_law", "tr", label="prior 1.5", hash="ab.cd"),
            "Girdi dağılımı: prior 1.5 (özet ab.cd)",
        )

    def test_plural_categories(self) -> None:
        """CLDR plural rules: English ``one`` only for the integer 1."""
        from probstat.foundations.messages import message, plural_category

        self.assertEqual(plural_category(1, "en"), "one")
        self.assertEqual(plural_category(1.0, "en"), "other")
        self.assertEqual(plural_category(2, "en"), "other")
        self.assertEqual(plural_category(0, "en"), "other")
        self.assertEqual(plural_category(1, "tr"), "one")
        self.assertEqual(plural_category(3, "tr"), "other")
        fields = {"successes": 0, "percent": 95, "lower": 0.0, "upper": 0.79}
        one = message("bayes.live.wilson", "en", count=1, share=0.0, **fields)
        many = message("bayes.live.wilson", "en", count=12, share=0.3, **fields)
        self.assertTrue(one.startswith("After 1 trial ("))
        self.assertTrue(many.startswith("After 12 trials ("))
        with self.assertRaises(ValueError):
            message("bayes.live.wilson", "en", successes=1)

    def test_unknown_locale_is_rejected(self) -> None:
        """Only the shipped catalogues are accepted."""
        from probstat.foundations.messages import check_locale, message

        with self.assertRaises(ValueError):
            message("report.root_seed", "de", seed=1)
        with self.assertRaises(TypeError):
            check_locale(None)
        with self.assertRaises(ValueError):
            fixture_study().report(locale="fr")

    def test_html_message_escapes_fields_and_text(self) -> None:
        """Field values cannot inject markup into an HTML report."""
        from probstat.foundations.messages import html_message

        self.assertEqual(
            html_message("study_html.warning", "en", message="<b>x</b> & y"),
            "Warning: &lt;b&gt;x&lt;/b&gt; &amp; y",
        )


class TurkishReportTests(unittest.TestCase):
    """The Turkish reports are complete and use Turkish number symbols."""

    def test_markdown_report(self) -> None:
        """The Turkish Markdown report translates labels and localizes numbers."""
        with fixed_provenance():
            report = fixture_study().report(locale="tr")
        for expected in (
            "# Belirsizlik çalışması: report <fixture>",
            "- Bağımlılık: Gaussian kopulası",
            "- İleri model değerlendirmeleri: 9.309",
            "(denemeler içindeki payı %0,182)",
            "- Değerlendirme bütçesi: 50.000 (kalan 40.674)",
            "durum: **yarıda kesildi**",
            "> **Uyarı:** bootstrap interval of S_R crosses zero",
            "| Nicelik | Değer |",
            "| g.mean | 1,97531 |",
            "| VaR[95%] | 12345,7 |",
            "### Vekil model çalışması: capacity surrogate",
            "Vekil model değerlendirmeleri: 1.000.000",
        ):
            self.assertIn(expected, report)
        for english in ("Root seed", "Evaluations:", "Warning", "Quantity"):
            self.assertNotIn(english, report)

    def test_html_report(self) -> None:
        """The Turkish HTML report declares ``lang="tr"`` and keeps its structure."""
        from probstat.visualization.study import study_html

        with fixed_provenance():
            english = study_html(fixture_study())
            page = study_html(fixture_study(), locale="tr")
        self.assertTrue(page.startswith('<!doctype html><html lang="tr">'))
        for expected in (
            "Aşama sonuçlarına geç",
            '<nav aria-label="Aşamalar">',
            "Aşağıda 2 aşama daha var.",
            "<strong>17 başarısız çalıştırma</strong>",
            "1 önbellek isabeti",
            '<td class="value">1,97531</td>',
            "Aşama 9 üzerinden vekil model çalışması: capacity surrogate",
        ):
            self.assertIn(expected, page)
        self.assertEqual(english.count("<section"), page.count("<section"))
        self.assertEqual(english.count("<tr>"), page.count("<tr>"))


if __name__ == "__main__":
    unittest.main()
