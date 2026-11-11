"""Accessibility of figures and reports: contrast, text alternatives, HTML.

The contrast thresholds are WCAG 2.1 level AA [REF-W3C-WCAG21-2018]: 4.5:1
for body text (success criterion 1.4.3) and 3:1 for graphical objects such
as plotted lines and markers (criterion 1.4.11), computed with the relative
luminance and contrast ratio of [EQ-VIZ-WCAG-001] and [EQ-VIZ-WCAG-002].
"""

from __future__ import annotations

import importlib.util
import math
import re
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path

from probstat.visualization.accessibility import (
    contrast_ratio,
    palette_contrast,
    relative_luminance,
)
from probstat.visualization.theme import GRAPHITE, PORCELAIN

HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None
TEXT_MINIMUM = 4.5
GRAPHIC_MINIMUM = 3.0


class ContrastFormulaTests(unittest.TestCase):
    """Relative luminance and contrast ratio match the WCAG definitions."""

    def test_luminance_end_points_and_primaries(self) -> None:
        """Black is 0, white 1, and the primaries carry the sRGB weights."""
        self.assertEqual(relative_luminance("#000000"), 0.0)
        self.assertAlmostEqual(relative_luminance("#FFFFFF"), 1.0, places=12)
        self.assertAlmostEqual(relative_luminance("#FF0000"), 0.2126, places=12)
        self.assertAlmostEqual(relative_luminance("#00FF00"), 0.7152, places=12)
        self.assertAlmostEqual(relative_luminance("#0000FF"), 0.0722, places=12)

    def test_luminance_linearization_branches(self) -> None:
        """Both branches of the channel transfer function are evaluated."""
        low = 10 / 255
        self.assertAlmostEqual(relative_luminance("#0A0A0A"), low / 12.92, places=15)
        mid = 128 / 255
        expected = ((mid + 0.055) / 1.055) ** 2.4
        self.assertAlmostEqual(relative_luminance("#808080"), expected, places=15)

    def test_contrast_ratio_range_and_symmetry(self) -> None:
        """Black on white is 21:1, a colour on itself 1:1, order is irrelevant."""
        self.assertAlmostEqual(contrast_ratio("#000000", "#FFFFFF"), 21.0)
        self.assertEqual(contrast_ratio("#44599C", "#44599C"), 1.0)
        self.assertEqual(
            contrast_ratio("#178F6E", "#F6F4EF"), contrast_ratio("#F6F4EF", "#178F6E")
        )
        # Mid grey on white: (1 + 0.05) / (0.2159 + 0.05).
        self.assertAlmostEqual(contrast_ratio("#808080", "#FFFFFF"), 3.949, places=3)

    def test_invalid_tokens_are_rejected(self) -> None:
        """Only ``#RRGGBB`` tokens are accepted."""
        for token in ("123456", "#12345", "#GGGGGG", "red", "#1234567"):
            with self.subTest(token=token), self.assertRaises(ValueError):
                relative_luminance(token)


class PaletteContrastTests(unittest.TestCase):
    """Both chart palettes and the HTML report colours meet level AA."""

    def test_chart_text_tokens_meet_text_contrast(self) -> None:
        """Ink, muted subtitles, and the accent reach 4.5:1 on the canvas."""
        for palette in (PORCELAIN, GRAPHITE):
            ratios = palette_contrast(palette)
            for token in ("ink", "muted", "accent"):
                with self.subTest(palette=palette.name, token=token):
                    self.assertGreaterEqual(ratios[token], TEXT_MINIMUM)

    def test_series_hues_meet_graphical_contrast(self) -> None:
        """Every data hue reaches 3:1 on the canvas (non-text contrast)."""
        for palette in (PORCELAIN, GRAPHITE):
            ratios = palette_contrast(palette)
            for index in range(len(palette.series)):
                token = f"series[{index}]"
                with self.subTest(palette=palette.name, token=token):
                    self.assertGreaterEqual(ratios[token], GRAPHIC_MINIMUM)

    def test_report_stylesheet_colours_meet_text_contrast(self) -> None:
        """Text colours of the HTML report reach 4.5:1 on canvas and surface."""
        from probstat.visualization.study import _HTML_STYLE

        blocks = re.findall(r":root\s*\{([^}]*)\}", _HTML_STYLE)
        self.assertEqual(len(blocks), 2, "light and dark custom properties")
        for scheme, block in zip(("light", "dark"), blocks, strict=True):
            tokens = dict(re.findall(r"--([a-z]+):\s*(#[0-9A-Fa-f]{6})", block))
            for text in ("ink", "muted", "accent"):
                for ground in ("canvas", "surface"):
                    with self.subTest(scheme=scheme, text=text, ground=ground):
                        ratio = contrast_ratio(tokens[text], tokens[ground])
                        self.assertGreaterEqual(ratio, TEXT_MINIMUM)

    def test_report_light_tokens_match_the_chart_palette(self) -> None:
        """The report and its embedded figures share one set of colours."""
        from probstat.visualization.study import _HTML_STYLE

        blocks = re.findall(r":root\s*\{([^}]*)\}", _HTML_STYLE)
        for palette, block in zip((PORCELAIN, GRAPHITE), blocks, strict=True):
            tokens = dict(re.findall(r"--([a-z]+):\s*(#[0-9A-Fa-f]{6})", block))
            for name in ("canvas", "ink", "muted", "rule", "accent"):
                with self.subTest(palette=palette.name, token=name):
                    self.assertEqual(tokens[name].upper(), getattr(palette, name))


@unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib is optional")
class FigureDescriptionTests(unittest.TestCase):
    """Saved figures carry a text alternative built from their own labels."""

    @classmethod
    def setUpClass(cls) -> None:
        """Select a non-interactive backend once."""
        import matplotlib

        matplotlib.use("Agg")

    def tearDown(self) -> None:
        """Close all figures created by a test."""
        import matplotlib.pyplot as plt

        plt.close("all")

    def themed_axes(self):
        """Return one themed axes with a title block, labels, and a log axis."""
        from probstat.visualization.theme import (
            axis_labels,
            chart_theme,
            line,
            new_axes,
            title_block,
        )

        with chart_theme():
            ax = new_axes()
            line(ax, [1.0, 10.0, 100.0], [0.5, 0.25, 0.125], color="#178F6E")
            ax.set_xscale("log")
            title_block(ax, "error halves per decade", "Line: absolute error")
            axis_labels(ax, "Samples n", "Error")
        return ax

    def test_description_repeats_titles_axes_and_scales(self) -> None:
        """The description names the title, subtitle, axes, ranges, and scale."""
        from probstat.visualization.accessibility import describe_figure, figure_title

        ax = self.themed_axes()
        text = describe_figure(ax.figure)
        self.assertTrue(text.startswith("Error halves per decade."))
        self.assertIn("Line: absolute error.", text)
        self.assertIn("Horizontal axis Samples n", text)
        self.assertIn("(log scale)", text)
        self.assertIn("vertical axis Error", text)
        self.assertEqual(figure_title(ax), "Error halves per decade")

    def test_panels_and_colour_bars_are_enumerated(self) -> None:
        """Multi-panel figures number their panels; colour bars are named."""
        import matplotlib.pyplot as plt

        from probstat.visualization.accessibility import describe_figure

        figure, (left, right) = plt.subplots(1, 2)
        left.set_title("Left finding")
        image = right.imshow([[0.0, 1.0], [2.0, 3.0]])
        figure.colorbar(image, ax=right, label="Density")
        figure.suptitle("Two views")
        text = describe_figure(figure)
        self.assertTrue(text.startswith("Two views."))
        self.assertIn("Panel 1 of 2. Left finding.", text)
        self.assertIn("Panel 2 of 2.", text)
        self.assertIn("Colour scale Density, 0 to 3.", text)

    def test_unlabelled_figure_still_gets_a_description(self) -> None:
        """An empty figure is described as such rather than left blank."""
        import matplotlib.pyplot as plt

        from probstat.visualization.accessibility import describe_figure

        self.assertEqual(
            describe_figure(plt.figure()), "Figure without titles or axes."
        )

    def test_saved_png_carries_title_and_description(self) -> None:
        """``save_figure`` writes the PNG ``Title`` and ``Description`` chunks."""
        from PIL import Image

        from probstat.visualization.accessibility import describe_figure
        from probstat.visualization.theme import save_figure

        ax = self.themed_axes()
        expected = describe_figure(ax.figure)
        with tempfile.TemporaryDirectory() as directory:
            path = save_figure(ax.figure, Path(directory) / "figure.png")
            with Image.open(path) as image:
                chunks = dict(image.text)  # type: ignore[attr-defined]
            custom = save_figure(
                ax.figure, Path(directory) / "custom.png", description="Custom β text"
            )
            with Image.open(custom) as image:
                custom_chunks = dict(image.text)  # type: ignore[attr-defined]
        self.assertEqual(chunks["Title"], "Error halves per decade")
        self.assertEqual(chunks["Description"], expected)
        self.assertEqual(custom_chunks["Description"], "Custom β text")

    def test_metadata_follows_the_file_format(self) -> None:
        """PDF uses ``Subject``; formats without a text field get none."""
        from probstat.visualization.accessibility import image_metadata
        from probstat.visualization.theme import save_figure

        ax = self.themed_axes()
        self.assertIn("Description", image_metadata(ax, "a.svg") or {})
        self.assertIn("Subject", image_metadata(ax, "a.pdf") or {})
        self.assertIsNone(image_metadata(ax, "a.jpg"))
        with tempfile.TemporaryDirectory() as directory:
            for suffix in (".svg", ".pdf", ".jpg"):
                with self.subTest(suffix=suffix):
                    path = save_figure(ax.figure, Path(directory) / f"f{suffix}")
                    self.assertGreater(path.stat().st_size, 0)


class _ReportStructure(HTMLParser):
    """Collect landmarks, the skip link, image ``alt`` texts, and ``lang``."""

    def __init__(self) -> None:
        """Start with nothing seen."""
        super().__init__()
        self.tags: list[str] = []
        self.lang: str | None = None
        self.links: list[str] = []
        self.ids: set[str] = set()
        self.alts: list[str | None] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Record one opening tag."""
        values = dict(attrs)
        self.tags.append(tag)
        if tag == "html":
            self.lang = values.get("lang")
        if tag == "a" and values.get("href"):
            self.links.append(str(values["href"]))
        if values.get("id"):
            self.ids.add(str(values["id"]))
        if tag == "img":
            self.alts.append(values.get("alt"))


@unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib is optional")
class StudyReportAccessibilityTests(unittest.TestCase):
    """The HTML study report is navigable without sight."""

    @classmethod
    def setUpClass(cls) -> None:
        """Render one small study report."""
        import matplotlib

        matplotlib.use("Agg")
        from probstat.distributions import LogNormal, Normal
        from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
        from probstat.uq import QuantityOfInterest
        from probstat.visualization.study import study_html
        from probstat.workflow import UQStudy

        inputs = JointInput.independent(
            RandomInput("R", LogNormal(math.log(5.0), 0.1)),
            RandomInput("S", Normal(3.0, 0.8)),
        )

        def margin(p):
            """Return the safety margin."""
            return {"g": p["R"] - p["S"]}

        model = ProbabilisticModel(
            inputs, margin, (QuantityOfInterest("g", "margin"),), name="capacity"
        )
        study = UQStudy(model, seed=7)
        study.propagate(256, design="sobol")
        cls.page = study_html(study)
        cls.structure = _ReportStructure()
        cls.structure.feed(cls.page)

    def test_language_is_declared(self) -> None:
        """The document declares its language (success criterion 3.1.1)."""
        self.assertEqual(self.structure.lang, "en")

    def test_landmarks_are_present_once(self) -> None:
        """Header, navigation, main content, and footer landmarks exist."""
        for tag in ("header", "nav", "main", "footer"):
            with self.subTest(tag=tag):
                self.assertEqual(self.structure.tags.count(tag), 1)

    def test_skip_link_comes_first_and_resolves(self) -> None:
        """The first link skips to the main landmark (criterion 2.4.1)."""
        self.assertEqual(self.structure.links[0], "#content")
        self.assertIn("content", self.structure.ids)
        self.assertLess(self.page.index('href="#content"'), self.page.index("<header"))
        self.assertIn('<main id="content"', self.page)

    def test_every_image_has_a_descriptive_alt_text(self) -> None:
        """Each figure has a non-empty generated text alternative."""
        self.assertTrue(self.structure.alts)
        for alt in self.structure.alts:
            with self.subTest(alt=alt):
                self.assertTrue(alt)
                assert alt is not None
                self.assertIn("axis", alt)


if __name__ == "__main__":
    unittest.main()
