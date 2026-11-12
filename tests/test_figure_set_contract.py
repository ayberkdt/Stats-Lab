"""The figure-set contract files agree with the theme (ADR-0098).

``docs/figures/stats-lab.seti.json`` and ``stats-lab-dark.seti.json`` are the
entity, quantity, and category tables of ``probstat.visualization.theme`` in
the datawiz set format. These checks are static: they read the files and the
theme tables, and render no figure except one line per palette.
"""

from __future__ import annotations

import importlib.util
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
FIGURES = ROOT / "docs" / "figures"
VISUALIZATION = ROOT / "src" / "probstat" / "visualization"
HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None
HEX = re.compile(r"^#[0-9a-f]{6}$")
# Modules that write HTML or define the tokens may hold colour literals.
COLOUR_SOURCES = {"theme.py", "_explainer_page.py", "study.py", "interactive.py"}


def load(name: str) -> dict[str, object]:
    """Read one contract file.

    Args:
        name: File name under ``docs/figures``.

    Returns:
        The parsed JSON document.

    References:
        [REF-MATPLOTLIB]. No equation is evaluated.

    """
    loaded: dict[str, object] = json.loads(
        (FIGURES / name).read_text(encoding="utf-8")
    )
    return loaded


class FigureSetContractTests(unittest.TestCase):
    """The contract is valid, current, and matches the theme tables."""

    def test_files_are_valid_datawiz_sets(self) -> None:
        """Both files carry the fields and value types of a datawiz set."""
        from probstat.visualization.figure_set import CONTRACT_FILES

        for name in CONTRACT_FILES.values():
            with self.subTest(file=name):
                contract = load(name)
                for key in ("dergi", "hedef", "varliklar", "nicelikler", "siralar"):
                    self.assertIn(key, contract)
                self.assertEqual(contract["etiket_bicimi"], "koseli")
                self.assertEqual(contract["ondalik_ayirici"], ".")
                entities = contract["varliklar"]
                assert isinstance(entities, dict)
                self.assertGreater(len(entities), 10)
                for entity, record in entities.items():
                    self.assertRegex(record["renk"], HEX, entity)
                    self.assertIsInstance(record["isaret"], str, entity)
                    dashes = record["cizgi"]
                    self.assertTrue(
                        dashes == "-"
                        or (isinstance(dashes, list) and len(dashes) == 2),
                        entity,
                    )
                quantities = contract["nicelikler"]
                assert isinstance(quantities, dict)
                for key, record in quantities.items():
                    self.assertIn(record["eksen"], {"x", "y", "serbest"}, key)
                    self.assertFalse(record["etiket"].endswith(")"), key)

    def test_files_equal_the_theme_tables(self) -> None:
        """Each file is exactly the contract generated from the theme."""
        from probstat.visualization.figure_set import CONTRACT_FILES, contract_text
        from probstat.visualization.theme import GRAPHITE, PORCELAIN

        for palette in (PORCELAIN, GRAPHITE):
            with self.subTest(palette=palette.name):
                text = (FIGURES / CONTRACT_FILES[palette.name]).read_text(
                    encoding="utf-8"
                )
                self.assertEqual(
                    text,
                    contract_text(palette),
                    "stale contract: run PYTHONPATH=src python "
                    "tools/generate_figure_set_contract.py",
                )

    def test_entity_colours_follow_their_roles(self) -> None:
        """Every entity colour is the palette token of its semantic role."""
        from probstat.visualization.figure_set import CONTRACT_FILES
        from probstat.visualization.theme import (
            ENTITY_STYLES,
            GRAPHITE,
            PORCELAIN,
            ROLES,
        )

        tokens = {
            PORCELAIN.name: PORCELAIN,
            GRAPHITE.name: GRAPHITE,
        }
        for palette_name, file_name in CONTRACT_FILES.items():
            palette = tokens[palette_name]
            entities = load(file_name)["varliklar"]
            assert isinstance(entities, dict)
            self.assertEqual(set(entities), set(ENTITY_STYLES))
            for name, style in ENTITY_STYLES.items():
                with self.subTest(palette=palette_name, entity=name):
                    self.assertIn(style.role, ROLES)
                    expected = {
                        "observed": palette.observed,
                        "model": palette.model,
                        "reference": palette.accent,
                        "context": palette.muted,
                    }.get(style.role)
                    if expected is None:
                        expected = palette.series[int(style.role[-1]) - 1]
                    self.assertEqual(entities[name]["renk"], expected.lower())
                    self.assertEqual(entities[name]["rol"], style.role)
                    self.assertEqual(entities[name]["baglam"], style.role == "context")

    def test_quantities_have_one_title_and_unit_spelling(self) -> None:
        """Quantity titles are unique and units are written in brackets."""
        from probstat.visualization.theme import QUANTITIES, Quantity, axis_title

        titles = [quantity.name for quantity in QUANTITIES.values()]
        self.assertEqual(len(titles), len(set(titles)))
        for key, quantity in QUANTITIES.items():
            with self.subTest(quantity=key):
                self.assertNotIn("(", quantity.name)
                self.assertNotIn("log scale", quantity.name)
                self.assertEqual(axis_title(key), quantity.label)
        self.assertEqual(Quantity("Loss", "kEUR").label, "Loss [kEUR]")
        with self.assertRaises(ValueError):
            axis_title("not a quantity")

    def test_brief_entities_and_quantities_are_in_the_contract(self) -> None:
        """Recurring entities and quantities named in ADR-0098 are listed."""
        contract = load("stats-lab.seti.json")
        entities = contract["varliklar"]
        quantities = contract["nicelikler"]
        assert isinstance(entities, dict) and isinstance(quantities, dict)
        for name in (
            "truth",
            "FORM",
            "first-order index",
            "total-effect index",
            "Monte Carlo reference",
            "chain 1",
        ):
            self.assertIn(name, entities)
        titles = {record["etiket"] for record in quantities.values()}
        for title in (
            "Failure probability",
            "Number of model evaluations",
            "Kolmogorov distance",
        ):
            self.assertIn(title, titles)

    def test_plotting_modules_take_colours_from_the_theme(self) -> None:
        """No plotting module writes a hexadecimal colour literal."""
        literal = re.compile(r"[\"']#[0-9A-Fa-f]{6}[\"']")
        offenders = [
            path.name
            for path in VISUALIZATION.glob("*.py")
            if path.name not in COLOUR_SOURCES
            and literal.search(path.read_text(encoding="utf-8"))
        ]
        self.assertEqual(offenders, [])

    def test_axis_titles_never_carry_a_scale_note(self) -> None:
        """Log scales are stated by the ticks and the text alternative."""
        pattern = re.compile(r"axis_labels\([^)]*log scale")
        offenders = [
            str(path.relative_to(ROOT))
            for folder in (VISUALIZATION, ROOT / "examples")
            for path in folder.glob("*.py")
            if pattern.search(path.read_text(encoding="utf-8"))
        ]
        self.assertEqual(offenders, [])

    def test_sentence_case_leaves_symbols_alone(self) -> None:
        """Variable names are capitalized; symbols and Greek names are not."""
        from probstat.visualization.theme import sentence_case

        self.assertEqual(sentence_case("deflection"), "Deflection")
        self.assertEqual(sentence_case("log-stiffness fluctuation")[:4], "Log-")
        for text in ("x", "u_load", "eps = 0.5", "tau", "x1", "Time [s]"):
            self.assertEqual(sentence_case(text), text)

    def test_unknown_entities_are_refused(self) -> None:
        """An entity outside the contract is an error, not a new colour."""
        from probstat.visualization.theme import entity_style

        self.assertEqual(entity_style("calibration (n = 300)").role, "context")
        with self.assertRaises(ValueError):
            entity_style("a new method")

    @unittest.skipUnless(HAS_MATPLOTLIB, "matplotlib is optional")
    def test_lines_drawn_by_entity_match_the_contract(self) -> None:
        """A line drawn with ``entity=`` has the contract colour and label."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from probstat.visualization.figure_set import CONTRACT_FILES
        from probstat.visualization.theme import (
            GRAPHITE,
            PORCELAIN,
            chart_theme,
            line,
            new_axes,
        )

        for palette in (PORCELAIN, GRAPHITE):
            entities = load(CONTRACT_FILES[palette.name])["varliklar"]
            assert isinstance(entities, dict)
            with chart_theme(palette):
                axes = new_axes()
                artist = line(axes, [0.0, 1.0, 2.0], [0.0, 1.0, 0.5], entity="truth")
            with self.subTest(palette=palette.name):
                self.assertEqual(artist.get_color().lower(), entities["truth"]["renk"])
                self.assertEqual(artist.get_label(), "truth")
                self.assertAlmostEqual(
                    artist.get_linewidth(), entities["truth"]["kalinlik"]
                )
            plt.close(axes.figure)


if __name__ == "__main__":
    unittest.main()
