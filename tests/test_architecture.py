"""Static tests for package dependency direction and acyclicity."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

SOURCE_ROOT = Path(__file__).parents[1] / "src" / "probstat"
LAYER = {
    "probstat": 0,
    "probstat.foundations": 0,
    "probstat.numerics": 0,
    "probstat.linalg": 0,
    "probstat.optimization": 1,
    "probstat.sampling": 1,
    "probstat.probability": 1,
    "probstat.statistics": 1,
    "probstat.distributions": 1,
    "probstat.inference": 2,
    "probstat.multivariate": 2,
    "probstat.uq": 2,
    "probstat.processes": 2,
    "probstat.monte_carlo": 3,
    "probstat.modeling": 3,
    "probstat.propagation": 4,
    "probstat.inverse": 4,
    "probstat.reliability": 5,
    "probstat.sensitivity": 5,
    "probstat.validation": 5,
    "probstat.risk": 6,
    "probstat.surrogates": 5,
    "probstat.filtering": 6,
    "probstat.data_assimilation": 6,
    "probstat.conformal": 6,
    "probstat.workflow": 7,
    "probstat.timeseries": 7,
    "probstat.visualization": 9,
}


def module_name(path: Path) -> str:
    """Convert a package source path to its importable module name.

    Args:
        path: Python file below ``src/probstat``.

    Returns:
        Fully qualified module or package name.

    References:
        [REF-PYTHON-DATAMODEL]. No mathematical equation is evaluated.

    """
    relative = path.relative_to(SOURCE_ROOT.parent).with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def imported_probstat_modules(path: Path) -> set[str]:
    """Extract absolute ``probstat`` import targets from one syntax tree.

    Args:
        path: Python source file to parse.

    Returns:
        A set of fully qualified import targets.

    References:
        [REF-PYTHON-AST]. No mathematical equation is evaluated.

    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(
                alias.name
                for alias in node.names
                if alias.name.startswith("probstat")
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("probstat"):
                imports.add(node.module)
    return imports


def package_layer(module: str) -> int:
    """Return the declared layer for a module or its nearest package.

    Args:
        module: Fully qualified ``probstat`` module name.

    Returns:
        Integer layer, where a larger value may depend on a smaller value.

    Raises:
        KeyError: If no architecture layer owns the module.

    References:
        Project architecture policy. No mathematical equation is evaluated.

    """
    candidates = sorted(LAYER, key=len, reverse=True)
    for package in candidates:
        if module == package or module.startswith(f"{package}."):
            return LAYER[package]
    raise KeyError(f"No architecture layer declared for {module}")


class ArchitectureTests(unittest.TestCase):
    """Reject upward dependencies and import cycles within the package."""

    def test_dependencies_point_downward(self) -> None:
        """Every internal import targets the same or a lower layer."""
        violations: list[str] = []
        for path in SOURCE_ROOT.rglob("*.py"):
            source = module_name(path)
            for target in imported_probstat_modules(path):
                if package_layer(target) > package_layer(source):
                    violations.append(f"{source} -> {target}")
        self.assertEqual(violations, [], "Upward imports: " + ", ".join(violations))

    def test_internal_import_graph_is_acyclic(self) -> None:
        """Depth-first traversal finds no cycle among repository modules."""
        paths = tuple(SOURCE_ROOT.rglob("*.py"))
        modules = {module_name(path) for path in paths}
        graph = {
            module_name(path): imported_probstat_modules(path) & modules
            for path in paths
        }
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node: str, trail: tuple[str, ...]) -> None:
            """Visit one node and fail with the complete detected cycle."""
            if node in visiting:
                cycle_start = trail.index(node)
                cycle = trail[cycle_start:] + (node,)
                self.fail("Import cycle: " + " -> ".join(cycle))
            if node in visited:
                return
            visiting.add(node)
            for target in graph[node]:
                visit(target, trail + (node,))
            visiting.remove(node)
            visited.add(node)

        for module in graph:
            visit(module, ())


if __name__ == "__main__":
    unittest.main()

