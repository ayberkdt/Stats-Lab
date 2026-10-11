"""Execute every Python snippet in the prerequisite guides and check its output.

Each guide under ``docs/prerequisites`` follows one contract: every
````python`` block is immediately followed by a ````text`` block holding the
exact standard output of that snippet. Blocks of one guide share a namespace
and run in order, so a guide reads like one session.
"""

from __future__ import annotations

import contextlib
import io
import re
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[1]
GUIDE_DIRECTORY = REPOSITORY_ROOT / "docs" / "prerequisites"
GUIDES = (
    "linear-algebra.md",
    "optimization.md",
    "multivariable-calculus-and-conditioning.md",
    "probability-and-statistics-refresher.md",
    "r-bridge.md",
)
BLOCK_PATTERN = re.compile(r"^```(\w+)\n(.*?)^```$", re.MULTILINE | re.DOTALL)


def snippet_pairs(path: Path) -> list[tuple[str, str]]:
    """Return ``(code, expected_output)`` pairs from one guide.

    Args:
        path: Markdown guide.

    Returns:
        Ordered pairs of Python source and its documented standard output.

    Raises:
        AssertionError: If a Python block is not followed by a text block.

    References:
        Project documentation contract. No mathematical equation is evaluated.

    """
    blocks = BLOCK_PATTERN.findall(path.read_text(encoding="utf-8"))
    pairs: list[tuple[str, str]] = []
    for index, (language, body) in enumerate(blocks):
        if language != "python":
            continue
        if index + 1 >= len(blocks) or blocks[index + 1][0] != "text":
            raise AssertionError(f"{path.name}: python block {index} lacks output")
        pairs.append((body, blocks[index + 1][1]))
    return pairs


class PrerequisiteGuideTests(unittest.TestCase):
    """Guides contain only snippets that run and print what they claim."""

    def test_every_guide_exists_and_has_snippets(self) -> None:
        """The five guides exist and each holds at least two snippets."""
        for name in GUIDES:
            with self.subTest(guide=name):
                path = GUIDE_DIRECTORY / name
                self.assertTrue(path.exists(), name)
                self.assertGreaterEqual(len(snippet_pairs(path)), 2)

    def test_snippets_print_documented_output(self) -> None:
        """Running each snippet reproduces its documented output exactly."""
        for name in GUIDES:
            namespace: dict[str, object] = {"__name__": f"guide_{name}"}
            for number, (code, expected) in enumerate(
                snippet_pairs(GUIDE_DIRECTORY / name)
            ):
                with self.subTest(guide=name, snippet=number):
                    buffer = io.StringIO()
                    with contextlib.redirect_stdout(buffer):
                        exec(compile(code, f"{name}#{number}", "exec"), namespace)
                    self.assertEqual(buffer.getvalue().rstrip(), expected.rstrip())


if __name__ == "__main__":
    unittest.main()
