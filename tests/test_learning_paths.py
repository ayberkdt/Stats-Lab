"""Check the ordered learning tracks in ``docs/learning-path.md``.

Every guide in ``docs/foundations`` and ``docs/prerequisites`` must be
reachable from a step, every relative link must resolve (including heading
anchors), every step must name its requirements, laboratory, and decision
record, requirements may point only to earlier steps, and every named
laboratory must exist under ``examples/``.

References:
    [REF-PYTHON-AST]. No mathematical equation is evaluated.

"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[1]
DOCS = REPOSITORY_ROOT / "docs"
LEARNING_PATH = DOCS / "learning-path.md"
GUIDE_DIRECTORIES = ("foundations", "prerequisites")
STEP_HEADING = re.compile(r"^### ([A-Z]\d+) — (.+)$", re.MULTILINE)
LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
FIELD = re.compile(r"^- (Requires|Lab|Decision record): (.+)$", re.MULTILINE)
STEP_ID = re.compile(r"\b[A-Z]\d+\b")


def heading_anchors(path: Path) -> set[str]:
    """Return the GitHub-style heading anchors of one Markdown file.

    Args:
        path: Markdown file.

    Returns:
        Anchors of every ATX heading outside fenced code blocks, with the
        ``-1``, ``-2`` suffixes GitHub adds to repeated headings.

    References:
        [REF-PYTHON-AST]. No mathematical equation is evaluated; the anchor
        rule mirrors ``tools/generate_api_reference.py``.

    """
    anchors: set[str] = set()
    seen: dict[str, int] = {}
    fenced = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("```"):
            fenced = not fenced
            continue
        match = re.match(r"^#{1,6} (.+?)\s*#*\s*$", line)
        if fenced or match is None:
            continue
        text = match.group(1).lower()
        anchor = "".join(c for c in text if c.isalnum() or c in " -_")
        anchor = anchor.replace(" ", "-")
        count = seen.get(anchor, 0)
        seen[anchor] = count + 1
        anchors.add(anchor if count == 0 else f"{anchor}-{count}")
    return anchors


def parse_steps() -> list[dict[str, object]]:
    """Return the steps of the learning path in document order.

    Returns:
        One mapping per step with its identifier, title, body text, and the
        raw values of its ``Requires``, ``Lab``, and ``Decision record``
        fields.

    References:
        [REF-PYTHON-AST]. No mathematical equation is evaluated.

    """
    text = LEARNING_PATH.read_text(encoding="utf-8")
    matches = list(STEP_HEADING.finditer(text))
    steps: list[dict[str, object]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end() : end].split("\n## ", 1)[0]
        fields = {name: value for name, value in FIELD.findall(body)}
        steps.append(
            {"id": match.group(1), "title": match.group(2), "body": body, **fields}
        )
    return steps


def resolve_link(target: str) -> tuple[Path, str]:
    """Split a relative link of the learning path into a path and an anchor.

    Args:
        target: Link target as written in ``docs/learning-path.md``.

    Returns:
        The resolved file path (the learning path itself for a bare anchor)
        and the anchor, which is empty when absent.

    References:
        [REF-PYTHON-AST]. No mathematical equation is evaluated.

    """
    location, _, anchor = target.partition("#")
    path = LEARNING_PATH if not location else (DOCS / location).resolve()
    return path, anchor


class LearningPathTests(unittest.TestCase):
    """The tracks are complete, ordered, and point to real files.

    References:
        [REF-PYTHON-AST]. No mathematical equation is evaluated.

    """

    def setUp(self) -> None:
        """Parse the learning path once per test.

        References:
            [REF-PYTHON-AST]. No mathematical equation is evaluated.

        """
        self.steps = parse_steps()

    def test_steps_exist_and_identifiers_are_unique(self) -> None:
        """At least six tracks of steps exist and no identifier repeats.

        References:
            [REF-PYTHON-AST]. No mathematical equation is evaluated.

        """
        identifiers = [str(step["id"]) for step in self.steps]
        self.assertEqual(len(identifiers), len(set(identifiers)), identifiers)
        self.assertGreaterEqual(len({name[0] for name in identifiers}), 6)

    def test_every_step_names_requirements_lab_and_decision_record(self) -> None:
        """Each step carries all three fields.

        References:
            [REF-PYTHON-AST]. No mathematical equation is evaluated.

        """
        for step in self.steps:
            with self.subTest(step=step["id"]):
                for field in ("Requires", "Lab", "Decision record"):
                    self.assertIn(field, step)
                self.assertRegex(str(step["Decision record"]), r"\[ADR-\d{4}\]")

    def test_every_guide_is_reachable_from_a_step(self) -> None:
        """Each foundation and prerequisite guide is linked from a step.

        References:
            [REF-PYTHON-AST]. No mathematical equation is evaluated.

        """
        linked: set[Path] = set()
        for step in self.steps:
            for target in LINK.findall(str(step["body"])):
                linked.add(resolve_link(target)[0])
        for directory in GUIDE_DIRECTORIES:
            for guide in sorted((DOCS / directory).glob("*.md")):
                with self.subTest(guide=guide.name):
                    self.assertIn(guide.resolve(), linked)

    def test_every_link_resolves(self) -> None:
        """Every relative link names an existing file and, if given, anchor.

        References:
            [REF-PYTHON-AST]. No mathematical equation is evaluated.

        """
        text = LEARNING_PATH.read_text(encoding="utf-8")
        for target in LINK.findall(text):
            if re.match(r"^[a-z]+://", target):
                continue
            with self.subTest(link=target):
                path, anchor = resolve_link(target)
                self.assertTrue(path.exists(), path)
                if anchor:
                    self.assertIn(anchor, heading_anchors(path))

    def test_requirements_point_only_to_earlier_steps(self) -> None:
        """The requirement graph follows document order, so it is acyclic.

        References:
            [REF-PYTHON-AST]. No mathematical equation is evaluated.

        """
        seen: set[str] = set()
        for step in self.steps:
            with self.subTest(step=step["id"]):
                value = str(step["Requires"]).strip()
                required = [] if value == "none" else STEP_ID.findall(value)
                self.assertTrue(value == "none" or required, value)
                for name in required:
                    self.assertIn(name, seen, f"{step['id']} requires {name}")
            seen.add(str(step["id"]))

    def test_every_lab_exists(self) -> None:
        """Named laboratories are scripts under ``examples/``; others are tests.

        References:
            [REF-PYTHON-AST]. No mathematical equation is evaluated.

        """
        for step in self.steps:
            with self.subTest(step=step["id"]):
                value = str(step["Lab"])
                targets = LINK.findall(value)
                self.assertTrue(targets, value)
                prefix = "../tests/" if value.startswith("none;") else "../examples/"
                for target in targets:
                    self.assertTrue(target.startswith(prefix), target)
                    self.assertTrue(target.endswith(".py"), target)
                    self.assertTrue(resolve_link(target)[0].is_file(), target)


if __name__ == "__main__":
    unittest.main()
