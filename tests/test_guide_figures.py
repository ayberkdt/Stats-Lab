"""Contract of the foundation-guide figures (ADR-0105).

``tools/build_guide_figures.py`` writes every guide figure in two palettes to
``docs/images/guides/`` and records, per group, the guide, text alternative,
and facts of each figure in ``docs/images/guides/<group>.json``. These tests
need neither Matplotlib nor datawiz: they check that every guide shows a
figure, that each figure is embedded exactly once in its own guide with its
recorded text alternative and both palettes, that the caption under it quotes
every recorded fact, and that no file is orphaned.

References:
    [REF-PROBSTAT-REPORTING], [REF-PNG-SPEC-2003], [REF-W3C-WCAG21-2018].

"""

from __future__ import annotations

import json
import re
import struct
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "docs" / "images" / "guides"
GUIDES = ROOT / "docs" / "foundations"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MAX_BYTES = 400_000
MAX_WIDTH = 2400
PICTURE = re.compile(
    r'<picture><source media="\(prefers-color-scheme: dark\)" '
    r'srcset="\.\./images/guides/(?P<dark>[a-z0-9-]+)\.png">'
    r'<img alt="(?P<alt>[^"]+)" src="\.\./images/guides/(?P<name>[a-z0-9-]+)\.png">'
    r"</picture>"
)


def _manifest() -> dict[str, dict[str, Any]]:
    """Return every recorded figure, keyed by name, with its group added."""
    entries: dict[str, dict[str, Any]] = {}
    for path in sorted(IMAGES.glob("*.json")):
        for name, entry in json.loads(path.read_text(encoding="utf-8")).items():
            if name in entries:
                raise AssertionError(f"{name} recorded in two manifests")
            entries[name] = {**entry, "group": path.stem}
    return entries


def _png_size(path: Path) -> tuple[int, int]:
    """Return the width and height from a PNG header [REF-PNG-SPEC-2003]."""
    header = path.read_bytes()[:24]
    if header[:8] != PNG_SIGNATURE or header[12:16] != b"IHDR":
        raise AssertionError(f"{path.name} is not a PNG file")
    width, height = struct.unpack(">II", header[16:24])
    return int(width), int(height)


def _caption(text: str, end: int) -> str:
    """Return the paragraph that follows a picture element."""
    rest = text[end:].lstrip("\n")
    return rest.split("\n\n", 1)[0]


class GuideFigureTests(unittest.TestCase):
    """Every guide shows checked figures in both palettes."""

    @classmethod
    def setUpClass(cls) -> None:
        """Read every manifest and every guide once."""
        cls.manifest = _manifest()
        cls.guides = {
            path.stem: path.read_text(encoding="utf-8") for path in GUIDES.glob("*.md")
        }

    def test_every_guide_has_a_figure(self) -> None:
        """Every guide in docs/foundations shows at least one figure."""
        shown = {entry["guide"] for entry in self.manifest.values()}
        missing = sorted(set(self.guides) - shown)
        self.assertEqual(missing, [], "guides without a figure")

    def test_each_figure_is_embedded_once_in_its_guide(self) -> None:
        """Each figure appears once in its guide with its text alternative."""
        for name, entry in self.manifest.items():
            with self.subTest(figure=name):
                self.assertIn(entry["guide"], self.guides)
                matches = [
                    match
                    for match in PICTURE.finditer(self.guides[entry["guide"]])
                    if match["name"] == name
                ]
                self.assertEqual(len(matches), 1, "embed exactly once")
                self.assertEqual(matches[0]["dark"], f"{name}-dark")
                self.assertEqual(matches[0]["alt"], entry["alt"])

    def test_captions_quote_every_fact(self) -> None:
        """The italic caption under a figure quotes every recorded fact."""
        for name, entry in self.manifest.items():
            text = self.guides.get(entry["guide"], "")
            for match in PICTURE.finditer(text):
                if match["name"] != name:
                    continue
                caption = _caption(text, match.end())
                with self.subTest(figure=name):
                    self.assertTrue(
                        caption.startswith("*") and caption.rstrip().endswith("*"),
                        "the caption is one italic paragraph under the picture",
                    )
                    self.assertTrue(
                        entry["facts"], "a figure records at least one fact"
                    )
                    for key, fact in entry["facts"].items():
                        self.assertIn(fact, caption, f"fact {key}")

    def test_no_picture_without_a_record(self) -> None:
        """Every guide image is a recorded figure in a picture element."""
        for stem, text in self.guides.items():
            for match in PICTURE.finditer(text):
                with self.subTest(guide=stem, figure=match["name"]):
                    self.assertIn(match["name"], self.manifest)
                    self.assertEqual(self.manifest[match["name"]]["guide"], stem)
            loose = text.count("images/guides/") - 2 * len(PICTURE.findall(text))
            with self.subTest(guide=stem):
                self.assertEqual(loose, 0, "guide images use the picture element")

    def test_files_exist_in_both_palettes_and_stay_small(self) -> None:
        """Both palettes exist as small PNG files and none is orphaned."""
        expected: set[str] = set()
        for name in self.manifest:
            for stem in (name, f"{name}-dark"):
                expected.add(f"{stem}.png")
                path = IMAGES / f"{stem}.png"
                with self.subTest(file=path.name):
                    self.assertTrue(path.is_file())
                    width, _ = _png_size(path)
                    self.assertLessEqual(width, MAX_WIDTH)
                    self.assertLessEqual(path.stat().st_size, MAX_BYTES)
        present = {path.name for path in IMAGES.glob("*.png")}
        self.assertEqual(sorted(present - expected), [], "unrecorded image files")

    def test_text_alternatives_describe_without_concluding(self) -> None:
        """Text alternatives are short descriptive sentences."""
        for name, entry in self.manifest.items():
            with self.subTest(figure=name):
                alt = entry["alt"]
                self.assertGreaterEqual(len(alt.split()), 5)
                self.assertLessEqual(len(alt), 200)
                self.assertNotIn('"', alt)


if __name__ == "__main__":
    unittest.main()
