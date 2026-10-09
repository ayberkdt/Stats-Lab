"""Static tests for docstrings, equations, and bibliography integrity."""

from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src" / "probstat"
EQUATION_REGISTRY = REPOSITORY_ROOT / "docs" / "equation-registry.md"
BIBLIOGRAPHY = REPOSITORY_ROOT / "docs" / "references.md"
EQUATION_PATTERN = re.compile(r"\bEQ-[A-Z0-9-]+\b")
REFERENCE_PATTERN = re.compile(r"\bREF-[A-Z0-9-]+\b")


def source_trees() -> tuple[tuple[Path, ast.Module], ...]:
    """Parse every package source file exactly once for a contract test.

    Returns:
        Tuples containing each path and its parsed syntax tree.

    References:
        [REF-PYTHON-AST]. No mathematical equation is evaluated.

    """
    return tuple(
        (path, ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
        for path in SOURCE_ROOT.rglob("*.py")
    )


class DocumentationContractTests(unittest.TestCase):
    """Require traceable documentation for all maintained package source."""

    def test_modules_classes_and_callables_have_docstrings(self) -> None:
        """Every maintained definition has a non-empty docstring."""
        missing: list[str] = []
        for path, tree in source_trees():
            relative = path.relative_to(REPOSITORY_ROOT)
            if not ast.get_docstring(tree):
                missing.append(f"{relative}:module")
            for node in ast.walk(tree):
                documented_definitions = (
                    ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
                )
                if isinstance(node, documented_definitions):
                    if not ast.get_docstring(node):
                        missing.append(f"{relative}:{node.lineno}:{node.name}")
        self.assertEqual(missing, [], "Missing docstrings: " + ", ".join(missing))

    def test_source_equation_ids_exist_in_registry(self) -> None:
        """Every equation ID cited by source exists in the central registry."""
        registry_text = EQUATION_REGISTRY.read_text(encoding="utf-8")
        registry_ids = set(EQUATION_PATTERN.findall(registry_text))
        source_ids: set[str] = set()
        for path in SOURCE_ROOT.rglob("*.py"):
            source_ids.update(EQUATION_PATTERN.findall(path.read_text(encoding="utf-8")))
        unknown = sorted(source_ids - registry_ids)
        self.assertEqual(unknown, [], "Unknown equation IDs: " + ", ".join(unknown))

    def test_source_reference_keys_exist_in_bibliography(self) -> None:
        """Every source citation key resolves in the bibliography."""
        bibliography_text = BIBLIOGRAPHY.read_text(encoding="utf-8")
        bibliography_keys = set(REFERENCE_PATTERN.findall(bibliography_text))
        source_keys: set[str] = set()
        for path in SOURCE_ROOT.rglob("*.py"):
            source_keys.update(REFERENCE_PATTERN.findall(path.read_text(encoding="utf-8")))
        unknown = sorted(source_keys - bibliography_keys)
        self.assertEqual(unknown, [], "Unknown reference keys: " + ", ".join(unknown))

    def test_registered_equations_have_a_source_and_code_location(self) -> None:
        """Every equation record names both provenance and implementation."""
        text = EQUATION_REGISTRY.read_text(encoding="utf-8")
        records = re.split(r"(?=^### EQ-)", text, flags=re.MULTILINE)[1:]
        incomplete: list[str] = []
        for record in records:
            equation_id = EQUATION_PATTERN.search(record)
            if equation_id and ("- Source:" not in record or "- Code:" not in record):
                incomplete.append(equation_id.group())
        self.assertEqual(
            incomplete,
            [],
            "Incomplete equation records: " + ", ".join(incomplete),
        )


if __name__ == "__main__":
    unittest.main()

