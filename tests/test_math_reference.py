"""Keep the generated mathematical reference in sync with the equation registry.

``tools/generate_math_reference.py`` regroups ``docs/equation-registry.md`` by
package. These tests regenerate the pages in memory and require the committed
files to match, every record to appear exactly once, every ``- Code:`` target
to resolve, and the target-reading rule of the generator to behave as its
module docstring states.

References:
    [REF-PYTHON-AST], [REF-PYTHON-DATAMODEL]. No mathematical equation is
    evaluated.

"""

from __future__ import annotations

import importlib.util
import re
import sys
import unittest
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "math-reference"
ENTRY_HEADING = re.compile(r"^### (EQ-[A-Z0-9-]+) — ", re.MULTILINE)
REGISTRY_LINK = re.compile(r"\(\.\./equation-registry\.md#([^)]+)\)")


def load_generator() -> ModuleType:
    """Import ``tools/generate_math_reference.py`` as a module.

    Returns:
        The generator module.

    References:
        [REF-PYTHON-AST]. No mathematical equation is evaluated.

    """
    name = "generate_math_reference"
    if name in sys.modules:
        return sys.modules[name]
    path = ROOT / "tools" / "generate_math_reference.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class MathReferenceTests(unittest.TestCase):
    """The committed pages match a fresh rendering and cover the registry.

    References:
        [REF-PYTHON-AST]. No mathematical equation is evaluated.

    """

    @classmethod
    def setUpClass(cls) -> None:
        """Render the reference and parse the registry once.

        References:
            [REF-PYTHON-AST]. No mathematical equation is evaluated.

        """
        cls.generator = load_generator()
        cls.pages = cls.generator.render()
        cls.entries = cls.generator.build_entries()
        registry = cls.generator.REGISTRY.read_text(encoding="utf-8")
        cls.records = cls.generator.parse_registry(registry)

    def test_reference_is_current(self) -> None:
        """Regenerate with ``python tools/generate_math_reference.py`` on failure.

        References:
            [REF-PYTHON-AST]. No mathematical equation is evaluated.

        """
        committed = {path.name for path in OUTPUT.glob("*.md")}
        self.assertEqual(committed, set(self.pages))
        for name, fresh in self.pages.items():
            with self.subTest(page=name):
                text = (OUTPUT / name).read_text(encoding="utf-8")
                self.assertEqual(
                    text.replace("\r\n", "\n"),
                    fresh,
                    f"docs/math-reference/{name} is stale; run "
                    "PYTHONPATH=src python tools/generate_math_reference.py",
                )

    def test_every_record_appears_exactly_once(self) -> None:
        """Each ``EQ-*`` record has exactly one entry across all pages.

        References:
            [REF-PYTHON-AST]. No mathematical equation is evaluated.

        """
        found: dict[str, int] = {}
        for text in self.pages.values():
            for identifier in ENTRY_HEADING.findall(text):
                found[identifier] = found.get(identifier, 0) + 1
        expected = {record.identifier for record in self.records}
        self.assertEqual(set(found), expected)
        repeated = sorted(key for key, count in found.items() if count != 1)
        self.assertEqual(repeated, [])

    def test_every_code_target_resolves(self) -> None:
        """Every record has a target and every target imports or exists.

        References:
            [REF-PYTHON-DATAMODEL]. No mathematical equation is evaluated.

        """
        self.assertEqual(self.generator.problems(self.entries), [])
        self.assertIn("## Unresolved code targets\n\nNone:", self.pages["index.md"])

    def test_private_targets_are_listed(self) -> None:
        """Targets with an underscore segment are reported on the index page.

        References:
            [REF-PYTHON-AST]. No mathematical equation is evaluated.

        """
        index = self.pages["index.md"]
        for entry in self.entries:
            for target in entry.targets:
                if target.private:
                    self.assertIn(
                        f"- {entry.record.identifier}: `{target.path}`", index
                    )
                elif target.kind == "object":
                    self.assertFalse(
                        any(part.startswith("_") for part in target.path.split("."))
                    )

    def test_other_packages_list_shared_records(self) -> None:
        """A package named by a record placed elsewhere links back to it.

        References:
            [REF-PYTHON-AST]. No mathematical equation is evaluated.

        """
        for entry in self.entries:
            for other in entry.others:
                with self.subTest(record=entry.record.identifier, package=other):
                    page = self.pages[f"{other}.md"]
                    self.assertIn(
                        f"- [{entry.record.identifier}]({entry.page}.md#", page
                    )

    def test_registry_links_and_citations_resolve(self) -> None:
        """Registry anchors exist and every cited key has its entry text.

        References:
            [REF-PYTHON-AST]. No mathematical equation is evaluated.

        """
        anchors = {
            self.generator.anchor(f"{record.identifier} — {record.title}")
            for record in self.records
        }
        bibliography = self.generator.parse_references(
            self.generator.REFERENCES.read_text(encoding="utf-8")
        )
        for name, text in self.pages.items():
            for target in REGISTRY_LINK.findall(text):
                with self.subTest(page=name, anchor=target):
                    self.assertIn(target, anchors)
        for record in self.records:
            for key in self.generator.REFERENCE_KEY.findall(record.field("Source")):
                with self.subTest(record=record.identifier, key=key):
                    self.assertIn(key, bibliography)
                    self.assertTrue(bibliography[key])

    def test_target_rule(self) -> None:
        """Qualified paths, shorthands, call suffixes, and files follow the rule.

        References:
            [REF-PYTHON-DATAMODEL]. No mathematical equation is evaluated.

        """
        generator = self.generator
        code = (
            "`distributions.LogNormal.cdf(x)` and `LogNormal.sf`, `.quantile`, "
            "`examples.sensitivity_lab.ishigami`, `tests/test_timeseries.py`, "
            '`n`, `method="pwm"`, `Unknown.member`, `distributions.Nope`, '
            "`distributions._continuous.expit`"
        )
        record = generator.Record(
            "EQ-TEST-001",
            "Test record",
            "",
            (("Domain", ""), ("Meaning", ""), ("Code", code), ("Source", "")),
        )
        targets = generator.code_targets(record, generator._packages())
        paths = [target.path for target in targets]
        self.assertEqual(
            paths,
            [
                "distributions.LogNormal.cdf",
                "distributions.LogNormal.sf",
                "distributions.LogNormal.quantile",
                "examples/sensitivity_lab.py::ishigami",
                "tests/test_timeseries.py",
                "Unknown.member",
                "distributions.Nope",
                "distributions._continuous.expit",
            ],
        )
        problems = [target.problem for target in targets]
        self.assertEqual(problems[:5], ["", "", "", "", ""])
        self.assertEqual(problems[5], "unqualified")
        self.assertTrue(problems[6])
        self.assertEqual(problems[7], "")
        self.assertTrue(targets[7].private)
        self.assertEqual(targets[0].module, "probstat.distributions.lognormal")


if __name__ == "__main__":
    unittest.main()
