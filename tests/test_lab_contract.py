"""Static contract for the reproducible laboratories in ``examples/``.

Every lab is a teaching document as much as a script (ADR-0078). Its module
docstring carries six sections in a fixed order:

``Definitions:``
    The objects and symbols the lab works with.
``Assumptions:``
    What must hold for the printed numbers to mean what they claim.
``Equations:``
    The ``EQ-*`` identifiers evaluated by the functions the lab calls. Each
    one is registered in ``docs/equation-registry.md`` and appears in a
    package module the lab imports (directly or transitively) or in the lab
    itself, so a lab cannot cite an equation it never evaluates.
``Diagnostics:``
    The checks that tell the reader whether a result can be trusted.
``Cost and error:``
    Evaluation counts, sample sizes, and the error each number carries.
``References:``
    ``REF-*`` keys registered in ``docs/references.md``.

The script also exposes a module-level integer ``SEED`` that it uses, writes
its figures through the themed visualization layer (``save_figure`` or the
study writers, never a bare ``savefig``, so every file carries its text
description), defines ``main`` behind a ``__main__`` guard, imports without
running, and names the regression test module that exercises the library
functions it relies on. CI runs every lab in full ("Examples run").
"""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
import unittest
from collections.abc import Iterable
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[1]
EXAMPLES = REPOSITORY_ROOT / "examples"
SOURCE_ROOT = REPOSITORY_ROOT / "src"
TESTS = REPOSITORY_ROOT / "tests"
EQUATION_REGISTRY = REPOSITORY_ROOT / "docs" / "equation-registry.md"
BIBLIOGRAPHY = REPOSITORY_ROOT / "docs" / "references.md"
CI_WORKFLOW = REPOSITORY_ROOT / ".github" / "workflows" / "ci.yml"
EQUATION_PATTERN = re.compile(r"\bEQ-[A-Z0-9-]+\b")
REFERENCE_PATTERN = re.compile(r"\bREF-[A-Z0-9-]+\b")
SECTIONS = (
    "Definitions",
    "Assumptions",
    "Equations",
    "Diagnostics",
    "Cost and error",
    "References",
)
SECTION_PATTERN = re.compile(
    r"^(" + "|".join(re.escape(name) for name in SECTIONS) + r"):[ \t]*$",
    flags=re.MULTILINE,
)
FIGURE_WRITERS = frozenset({"save_figure", "save_study_figures", "write_study_html"})

# Each lab -> (regression test module, library name the lab calls and that
# test module exercises). The test checks both ends of the link.
LAB_REGRESSION_TESTS: dict[str, tuple[str, str]] = {
    "adaptive_reliability_lab": ("test_adaptive_reliability.py", "ak_mcs"),
    "advanced_monte_carlo_lab": (
        "test_uq3_variance_reduction.py",
        "compare_efficiency",
    ),
    "basic_monte_carlo_lab": ("test_uq2_propagation.py", "convergence_trace"),
    "bayesian_calibration_lab": (
        "test_uq6_inverse_calibration.py",
        "calibrate_linear_gaussian",
    ),
    "causal_lab": ("test_causal_inference.py", "estimate_aipw"),
    "central_model_lab": ("test_core_modeling.py", "ProbabilisticModel"),
    "discrepancy_calibration_lab": (
        "test_discrepancy_calibration.py",
        "CalibrationProblem",
    ),
    "experimental_design_lab": ("test_experimental_design.py", "greedy_design"),
    "extremes_lab": ("test_extreme_value_analysis.py", "block_maxima"),
    "filtering_lab": ("test_uq7_filtering.py", "bootstrap_particle_filter"),
    "forecast_calibration_lab": ("test_forecast_calibration.py", "brier_decomposition"),
    "generate_foundation_figures": (
        "test_generating_functions_and_protocols.py",
        "Categorical",
    ),
    "hierarchical_lab": (
        "test_hierarchical_and_ppc.py",
        "hierarchical_normal_posterior",
    ),
    "information_lab": ("test_information_theory.py", "ksg_mutual_information"),
    "limit_theorems_lab": ("test_limit_theorems.py", "simulate_clt_distance"),
    "missing_data_lab": ("test_missing_data.py", "fit_incomplete_multivariate_normal"),
    "mission_systems_lab": ("test_mission_assessment.py", "LifeCycleCostModel"),
    "model_selection_lab": (
        "test_generic_fitting_model_selection.py",
        "select_distribution",
    ),
    "nonlinear_calibration_lab": ("test_workflow_study.py", "UQStudy"),
    "optimal_transport_lab": ("test_transport_and_geometry.py", "sinkhorn"),
    "ppl_lab": ("test_probabilistic_programming.py", "Beta"),
    "random_field_lab": ("test_stochastic_processes.py", "karhunen_loeve_inputs"),
    "reliability_lab": (
        "test_uq4_reliability.py",
        "estimate_directional_failure_probability",
    ),
    "sensitivity_lab": ("test_uq5_sensitivity.py", "estimate_sobol_indices"),
    "statistical_fitting_lab": (
        "test_uq1_modeling_and_inference.py",
        "fit_lognormal_mle",
    ),
    "statistical_modeling_lab": ("test_timeseries.py", "fit_arima"),
    "surrogate_lab": ("test_surrogates.py", "fit_polynomial_chaos"),
    "uncertainty_visualization_lab": ("test_visualization_theme.py", "plot_fan_chart"),
    "uq_study_complete_lab": (
        "test_uq_study_integration_three.py",
        "DiscrepancyProcess",
    ),
    "uq_study_extensions_lab": ("test_uq_study_extensions.py", "UQStudy"),
    "uq_study_lab": ("test_workflow_study.py", "UQStudy"),
    "uq_workflow_lab": ("test_uq0_prediction_workflow.py", "uq_workflow_lab"),
}


def lab_paths() -> tuple[Path, ...]:
    """Return every laboratory script in a stable order.

    Returns:
        Sorted paths of ``examples/*.py``.

    References:
        [REF-PYTHON-AST]. No equation is evaluated.

    """
    return tuple(sorted(EXAMPLES.glob("*.py")))


def registered_keys(path: Path, heading: str) -> frozenset[str]:
    """Return the identifiers declared as headings of a registry file.

    Args:
        path: Markdown registry.
        heading: Heading marker, ``"##"`` or ``"###"``.

    Returns:
        Declared ``EQ-*`` or ``REF-*`` identifiers.

    References:
        [REF-PYTHON-AST]. No equation is evaluated.

    """
    pattern = re.compile(
        r"^" + re.escape(heading) + r" ((?:EQ|REF)-[A-Z0-9-]+)(?=\s|$)", re.MULTILINE
    )
    return frozenset(pattern.findall(path.read_text(encoding="utf-8")))


def docstring_sections(docstring: str) -> dict[str, str]:
    """Split a lab docstring into its contract sections.

    Args:
        docstring: Module docstring.

    Returns:
        Mapping from section name to its body text, in order of appearance.

    References:
        [REF-PYTHON-AST]. No equation is evaluated.

    """
    matches = list(SECTION_PATTERN.finditer(docstring))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(docstring)
        sections[match.group(1)] = docstring[match.end() : end].strip()
    return sections


class ModuleGraph:
    """Import graph of ``probstat`` that follows names to their definitions.

    ``from probstat.inference import f`` depends on the module that defines
    ``f``, not on the whole ``inference`` package: re-exports in package
    ``__init__`` files are followed to the defining module. A module depends
    on every name it imports, at module level or lazily inside a function.
    The closure of a lab therefore contains the code its calls can reach,
    which is what an ``EQ-*`` citation must point into.
    """

    def __init__(self) -> None:
        """Index and parse every ``probstat`` module once."""
        self.paths: dict[str, Path] = {}
        for path in (SOURCE_ROOT / "probstat").rglob("*.py"):
            parts = list(path.relative_to(SOURCE_ROOT).with_suffix("").parts)
            if parts[-1] == "__init__":
                parts.pop()
            self.paths[".".join(parts)] = path
        self.texts = {
            name: path.read_text(encoding="utf-8") for name, path in self.paths.items()
        }
        self.trees = {name: ast.parse(text) for name, text in self.texts.items()}
        self._edges: dict[str, frozenset[str]] = {}

    def _package(self, module: str) -> str:
        """Return the package that relative imports in ``module`` resolve to."""
        if self.paths[module].name == "__init__.py":
            return module
        return module.rpartition(".")[0]

    def _absolute(self, module: str, node: ast.ImportFrom) -> str:
        """Return the absolute target of a (possibly relative) import."""
        if not node.level:
            return node.module or ""
        parts = self._package(module).split(".") if module else []
        if node.level > 1:
            parts = parts[: -(node.level - 1)]
        base = ".".join(parts)
        return f"{base}.{node.module}" if node.module else base

    def defining_module(self, module: str, name: str, depth: int = 0) -> str | None:
        """Follow ``from module import name`` to the module defining ``name``.

        Args:
            module: Absolute module the name is imported from.
            name: Imported name.
            depth: Recursion guard.

        Returns:
            Defining module, or ``None`` for a name outside ``probstat``.

        """
        if f"{module}.{name}" in self.paths:
            return f"{module}.{name}"
        if module not in self.paths:
            return None
        if depth > 20:
            return module
        for node in self.trees[module].body:
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if (alias.asname or alias.name) == name:
                        target = self._absolute(module, node)
                        found = self.defining_module(target, alias.name, depth + 1)
                        return found or module
        return module

    def imports(self, tree: ast.AST, module: str) -> set[str]:
        """Return the defining modules of every name a syntax tree imports.

        Args:
            tree: Parsed module or lab script.
            module: Its dotted name (``""`` for a lab script).

        Returns:
            ``probstat`` modules the tree depends on.

        """
        found: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(
                    alias.name for alias in node.names if alias.name in self.paths
                )
            elif isinstance(node, ast.ImportFrom):
                target = self._absolute(module, node)
                for alias in node.names:
                    defining = self.defining_module(target, alias.name)
                    if defining is not None:
                        found.add(defining)
        return found

    def closure_text(self, tree: ast.AST) -> str:
        """Return the source of every module reachable from a lab script.

        Args:
            tree: Parsed lab script.

        Returns:
            Concatenated source of the transitive dependency closure.

        """
        pending = list(self.imports(tree, ""))
        seen: set[str] = set()
        while pending:
            name = pending.pop()
            if name in seen:
                continue
            seen.add(name)
            if name not in self._edges:
                self._edges[name] = frozenset(self.imports(self.trees[name], name))
            pending.extend(self._edges[name] - seen)
        return "\n".join(self.texts[name] for name in sorted(seen))


def called_names(tree: ast.AST) -> set[str]:
    """Return every function or attribute name that a syntax tree calls.

    References:
        [REF-PYTHON-AST]. No equation is evaluated.

    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                names.add(node.func.attr)
    return names


def referenced_names(tree: ast.AST) -> Iterable[str]:
    """Yield every identifier a syntax tree loads or imports.

    References:
        [REF-PYTHON-AST]. No equation is evaluated.

    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            yield node.id
        elif isinstance(node, ast.Attribute):
            yield node.attr
        elif isinstance(node, ast.ImportFrom):
            yield from (alias.name for alias in node.names)


class LabContractTests(unittest.TestCase):
    """Every laboratory satisfies the documentation and reproducibility contract."""

    @classmethod
    def setUpClass(cls) -> None:
        """Parse every lab and the two registries once."""
        cls.labs = {path.stem: path for path in lab_paths()}
        cls.trees = {
            stem: ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for stem, path in cls.labs.items()
        }
        cls.equations = registered_keys(EQUATION_REGISTRY, "###")
        cls.references = registered_keys(BIBLIOGRAPHY, "##")
        cls.graph = ModuleGraph()

    def test_labs_exist(self) -> None:
        """The contract is checked against a non-empty set of labs."""
        self.assertGreaterEqual(len(self.labs), 30)

    def test_docstrings_have_every_section_in_order(self) -> None:
        """Each docstring carries the six sections, non-empty, in fixed order."""
        for stem, tree in self.trees.items():
            with self.subTest(lab=stem):
                docstring = ast.get_docstring(tree) or ""
                sections = docstring_sections(docstring)
                self.assertEqual(tuple(sections), SECTIONS)
                for name, body in sections.items():
                    self.assertTrue(body, f"{stem}: empty {name} section")

    def test_equations_are_registered_and_evaluated(self) -> None:
        """Every cited equation is registered and reachable from the lab's code."""
        for stem, tree in self.trees.items():
            with self.subTest(lab=stem):
                docstring = ast.get_docstring(tree) or ""
                listed = EQUATION_PATTERN.findall(
                    docstring_sections(docstring).get("Equations", "")
                )
                self.assertTrue(listed, f"{stem}: no EQ-* id in Equations")
                cited = set(EQUATION_PATTERN.findall(docstring))
                self.assertEqual(sorted(cited - self.equations), [])
                own = self.labs[stem].read_text(encoding="utf-8").replace(docstring, "")
                reachable = self.graph.closure_text(tree) + own
                missing = sorted(key for key in cited if key not in reachable)
                self.assertEqual(missing, [], f"{stem}: equations not evaluated")

    def test_references_are_registered(self) -> None:
        """The References section names at least one registered ``REF-*`` key."""
        for stem, tree in self.trees.items():
            with self.subTest(lab=stem):
                docstring = ast.get_docstring(tree) or ""
                listed = REFERENCE_PATTERN.findall(
                    docstring_sections(docstring).get("References", "")
                )
                self.assertTrue(listed, f"{stem}: no REF-* key in References")
                cited = set(REFERENCE_PATTERN.findall(docstring))
                self.assertEqual(sorted(cited - self.references), [])

    def test_labs_expose_and_use_a_fixed_seed(self) -> None:
        """A module-level integer ``SEED`` exists and the lab reads it."""
        for stem, tree in self.trees.items():
            with self.subTest(lab=stem):
                assert isinstance(tree, ast.Module)
                values = [
                    node.value
                    for node in tree.body
                    if isinstance(node, ast.Assign)
                    and any(
                        isinstance(target, ast.Name) and target.id == "SEED"
                        for target in node.targets
                    )
                ]
                self.assertEqual(len(values), 1, f"{stem}: no SEED constant")
                value = values[0]
                self.assertIsInstance(value, ast.Constant)
                assert isinstance(value, ast.Constant)
                self.assertIs(type(value.value), int)
                loads = [
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Name)
                    and node.id == "SEED"
                    and isinstance(node.ctx, ast.Load)
                ]
                self.assertTrue(loads, f"{stem}: SEED is never used")

    def test_figures_go_through_the_themed_layer(self) -> None:
        """Labs import the visualization layer and save through its writers."""
        for stem, tree in self.trees.items():
            with self.subTest(lab=stem):
                modules = {
                    node.module
                    for node in ast.walk(tree)
                    if isinstance(node, ast.ImportFrom) and node.module
                }
                self.assertTrue(
                    any(name.startswith("probstat.visualization") for name in modules),
                    f"{stem}: does not import probstat.visualization",
                )
                calls = called_names(tree)
                self.assertTrue(calls & FIGURE_WRITERS, f"{stem}: no figure writer")
                self.assertNotIn("savefig", calls, f"{stem}: bypasses save_figure")

    def test_labs_define_a_guarded_main(self) -> None:
        """Each lab defines ``main`` and calls it only as a script."""
        for stem, tree in self.trees.items():
            with self.subTest(lab=stem):
                assert isinstance(tree, ast.Module)
                functions = {
                    node.name for node in tree.body if isinstance(node, ast.FunctionDef)
                }
                self.assertIn("main", functions)
                guards = [
                    node
                    for node in tree.body
                    if isinstance(node, ast.If)
                    and "__main__" in ast.unparse(node.test)
                    and "main()" in ast.unparse(node)
                ]
                self.assertEqual(len(guards), 1, f"{stem}: no __main__ guard")

    def test_labs_import_without_running(self) -> None:
        """Importing a lab defines its entry point and seed without side output."""
        for stem, path in self.labs.items():
            with self.subTest(lab=stem):
                name = f"_probstat_lab_{stem}"
                spec = importlib.util.spec_from_file_location(name, path)
                assert spec is not None and spec.loader is not None
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module
                try:
                    spec.loader.exec_module(module)
                finally:
                    sys.modules.pop(name, None)
                self.assertTrue(callable(module.main))
                self.assertIsInstance(module.SEED, int)

    def test_every_lab_has_a_regression_test(self) -> None:
        """Each lab maps to a test module that exercises a name it relies on."""
        self.assertEqual(sorted(LAB_REGRESSION_TESTS), sorted(self.labs))
        for stem, (test_name, anchor) in LAB_REGRESSION_TESTS.items():
            with self.subTest(lab=stem):
                test_path = TESTS / test_name
                self.assertTrue(test_path.is_file(), test_name)
                test_text = test_path.read_text(encoding="utf-8")
                self.assertRegex(test_text, rf"\b{re.escape(anchor)}\b")
                if anchor != stem:
                    self.assertIn(anchor, set(referenced_names(self.trees[stem])))

    def test_ci_runs_every_lab(self) -> None:
        """The CI workflow executes every script in ``examples/``."""
        workflow = CI_WORKFLOW.read_text(encoding="utf-8")
        self.assertIn('for lab in examples/*.py; do python "$lab"; done', workflow)


if __name__ == "__main__":
    unittest.main()
