"""Check the snippet contract of the prerequisite guides.

Each guide under ``docs/prerequisites`` follows one contract: every
````python`` block is immediately followed by a ````text`` block holding the
standard output of that snippet. Blocks of one guide share a namespace and
run in order, so a guide reads like one script. The snippets are executed,
and their output compared, by ``tests/test_documentation_examples.py``
(ADR-0095), which runs every guide of the documentation once; this module
checks only that the listed guides exist and keep the contract.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from _documentation import code_blocks

REPOSITORY_ROOT = Path(__file__).parents[1]
GUIDE_DIRECTORY = REPOSITORY_ROOT / "docs" / "prerequisites"
GUIDES = (
    "linear-algebra.md",
    "optimization.md",
    "multivariable-calculus-and-conditioning.md",
    "probability-and-statistics-refresher.md",
    "r-bridge.md",
    "numerical-integration.md",
)


def snippet_pairs(path: Path) -> list[tuple[str, str]]:
    """Return ``(code, expected_output)`` pairs from one guide.

    Args:
        path: Markdown guide.

    Returns:
        Ordered pairs of Python source and its documented standard output.

    Raises:
        AssertionError: If a Python block is not followed by a text block.

    References:
        [REF-PYTHON-AST]. No mathematical equation is evaluated.

    """
    pairs: list[tuple[str, str]] = []
    for block in code_blocks(path.read_text(encoding="utf-8")):
        if block.language != "python":
            continue
        if block.output is None:
            raise AssertionError(f"{path.name}: python block {block.line} lacks output")
        pairs.append((block.body, block.output))
    return pairs


class PrerequisiteGuideTests(unittest.TestCase):
    """Guides contain snippets that document their output."""

    def test_every_guide_exists_and_has_snippets(self) -> None:
        """Every listed guide exists and holds at least two documented snippets."""
        for name in GUIDES:
            with self.subTest(guide=name):
                path = GUIDE_DIRECTORY / name
                self.assertTrue(path.exists(), name)
                self.assertGreaterEqual(len(snippet_pairs(path)), 2)

    def test_every_guide_in_the_directory_is_listed(self) -> None:
        """No prerequisite guide escapes the contract by being unlisted."""
        found = sorted(path.name for path in GUIDE_DIRECTORY.glob("*.md"))
        self.assertEqual(found, sorted(GUIDES))


if __name__ == "__main__":
    unittest.main()
