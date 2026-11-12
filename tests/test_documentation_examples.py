"""Every documented example runs and every quoted number is true (ADR-0095).

The Python blocks of ``README.md``, ``docs/foundations``,
``docs/prerequisites``, and ``docs/learning-path.md`` are executed in order,
one namespace per file, and the values they quote are compared with the
values they compute (see ``tests/_documentation.py`` for the rules). The
remaining tests keep the prose honest where it can be checked mechanically:
every public name a guide mentions exists, every relative link resolves, the
README's counts and tables match the code, and every package states its
purpose, entry points, assumptions, and pitfalls next to a guide that has a
"Common pitfalls" section.

References:
    [REF-PYTHON-AST], [REF-PYTHON-UNITTEST]. No mathematical equation is
    evaluated.

"""

from __future__ import annotations

import ast
import importlib
import json
import pkgutil
import re
import sys
import tomllib
import unittest
from collections.abc import Callable
from pathlib import Path

from _documentation import (
    REPOSITORY_ROOT,
    code_blocks,
    documented_files,
    run_document,
)
from test_learning_paths import heading_anchors

import probstat

DOCS = REPOSITORY_ROOT / "docs"
README = REPOSITORY_ROOT / "README.md"
SOURCE = REPOSITORY_ROOT / "src" / "probstat"
MAX_EXEMPTIONS = 3
LINK = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)\)|!\[[^\]]*\]\(([^)\s]+)\)")
HTML_SOURCE = re.compile(r"""(?:src|srcset)="([^"]+)\"""")
INLINE_CODE = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")
DOTTED_NAME = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+$")
PITFALLS_HEADING = re.compile(r"^## Common pitfalls\s*$", re.MULTILINE)
RAW_PREFIX = "https://raw.githubusercontent.com/ayberkdt/Stats-Lab/main/"
# (blob | tree | "" for raw), repository path, optional heading anchor.
REPOSITORY_URL = re.compile(
    r"https://(?:github\.com/ayberkdt/Stats-Lab/(blob|tree)"
    r"|raw\.githubusercontent\.com/ayberkdt/Stats-Lab)/main/"
    r"([^\s\"')#}]+)(?:#([\w-]+))?"
)
DOCSTRING_SECTIONS = ("Main entry points:", "Assumptions:", "Common pitfalls:")

# Package -> its foundation (or prerequisite) guide; the package docstring
# names the same file on its ``Guide:`` line.
PACKAGE_GUIDES: dict[str, str] = {
    "causal": "foundations/causal-inference.md",
    "conformal": "foundations/conformal-prediction.md",
    "cost": "foundations/cost-analysis.md",
    "data_assimilation": "foundations/filtering-and-data-assimilation.md",
    "design": "foundations/experimental-design.md",
    "distributions": "foundations/continuous-distribution-catalog.md",
    "filtering": "foundations/filtering-and-data-assimilation.md",
    "foundations": "foundations/probabilistic-modeling.md",
    "generative": "foundations/generative-models.md",
    "inference": "foundations/hypothesis-testing-and-bootstrap.md",
    "information": "foundations/information-theory.md",
    "inverse": "foundations/inverse-problems-and-calibration.md",
    "linalg": "prerequisites/linear-algebra.md",
    "modeling": "foundations/probabilistic-modeling.md",
    "monte_carlo": "foundations/advanced-monte-carlo.md",
    "multivariate": "foundations/multivariate-laws.md",
    "numerics": "prerequisites/numerical-integration.md",
    "optimization": "prerequisites/optimization.md",
    "probability": "foundations/probability.md",
    "processes": "foundations/stochastic-processes-and-random-fields.md",
    "propagation": "foundations/uncertainty-propagation.md",
    "reliability": "foundations/advanced-reliability.md",
    "risk": "foundations/risk-analysis.md",
    "sampling": "foundations/uncertainty-propagation.md",
    "sensitivity": "foundations/sensitivity-analysis.md",
    "statistics": "foundations/descriptive-statistics.md",
    "surrogates": "foundations/surrogate-models.md",
    "systems": "foundations/system-reliability.md",
    "timeseries": "foundations/time-series.md",
    "uq": "foundations/vv-uq-foundations.md",
    "validation": "foundations/model-validation.md",
    "visualization": "foundations/visual-language.md",
    "workflow": "foundations/probabilistic-modeling.md",
}


def package_names() -> tuple[str, ...]:
    """Return the top-level ``probstat`` packages.

    Returns:
        Sorted names of the public subpackages.

    References:
        [REF-PYTHON-AST]. No mathematical equation is evaluated.

    """
    return tuple(
        sorted(
            info.name
            for info in pkgutil.iter_modules(probstat.__path__)
            if info.ispkg and not info.name.startswith("_")
        )
    )


def prose(path: Path) -> str:
    """Return a Markdown file without its fenced code blocks.

    Args:
        path: Markdown file.

    Returns:
        The text outside fences.

    References:
        [REF-PYTHON-AST]. No mathematical equation is evaluated.

    """
    return re.sub(
        r"^```.*?^```[ \t]*$",
        "",
        path.read_text(encoding="utf-8"),
        flags=re.MULTILINE | re.DOTALL,
    )


def exported_names() -> dict[str, object]:
    """Map every name in a package ``__all__`` to its object.

    Returns:
        Name to object, for every public name of every package (the first
        package wins when two export the same name).

    References:
        [REF-PYTHON-AST]. No mathematical equation is evaluated.

    """
    found: dict[str, object] = {}
    for name in package_names():
        module = importlib.import_module(f"probstat.{name}")
        for item in getattr(module, "__all__", ()):
            found.setdefault(item, getattr(module, item))
    return found


def resolve(dotted: str, exported: dict[str, object]) -> bool | None:
    """Resolve a dotted name written in a guide.

    Args:
        dotted: Name such as ``probstat.reliability.form_hlrf``,
            ``reliability.form_hlrf``, or ``UQStudy.reliability``.
        exported: Public names of every package.

    Returns:
        ``True`` when it resolves, ``False`` when it should resolve and does
        not, and ``None`` when the first segment is not a library name (a
        local variable in the prose, for example).

    References:
        [REF-PYTHON-AST]. No mathematical equation is evaluated.

    """
    parts = dotted.split(".")
    if parts[0] == "probstat":
        parts = parts[1:]
        if not parts:
            return True
    packages = set(package_names())
    target: object
    if parts[0] in packages:
        module_path = f"probstat.{parts[0]}"
        target = importlib.import_module(module_path)
        for part in parts[1:]:
            if hasattr(target, part):
                target = getattr(target, part)
                continue
            module_path = f"{module_path}.{part}"
            try:
                target = importlib.import_module(module_path)
            except ModuleNotFoundError:
                return _standard_library_name(parts)
        return True
    if parts[0] in exported and parts[0][:1].isupper():
        target = exported[parts[0]]
        for part in parts[1:]:
            if not hasattr(target, part):
                fields = getattr(target, "__dataclass_fields__", {})
                slots = getattr(target, "__slots__", ())
                annotations = getattr(target, "__annotations__", {})
                return part in fields or part in slots or part in annotations
            target = getattr(target, part)
        return True
    return None


def _standard_library_name(parts: list[str]) -> bool:
    """Return whether a dotted name exists in the standard library instead.

    ``statistics.NormalDist`` names the standard-library module that shares
    its name with ``probstat.statistics``.

    Args:
        parts: Segments of the dotted name.

    Returns:
        Whether the standard-library module has the attribute path.

    References:
        [REF-PYTHON-AST]. No mathematical equation is evaluated.

    """
    if parts[0] not in sys.stdlib_module_names:
        return False
    target: object = importlib.import_module(parts[0])
    for part in parts[1:]:
        if not hasattr(target, part):
            return False
        target = getattr(target, part)
    return True


def _document_test(path: Path) -> Callable[[unittest.TestCase], None]:
    """Build the test that runs the examples of one Markdown file.

    Args:
        path: Markdown file.

    Returns:
        A test method.

    References:
        [REF-PYTHON-UNITTEST]. No mathematical equation is evaluated.

    """

    def test(self: unittest.TestCase) -> None:
        run = run_document(path)
        if run.missing_module is not None:
            self.skipTest(f"optional dependency {run.missing_module} not installed")
        self.assertEqual(run.failures, [], "\n\n".join(run.failures))

    test.__doc__ = f"The Python examples of {path.name} run and print what they claim."
    return test


class DocumentedExampleTests(unittest.TestCase):
    """One test per Markdown file: its examples run and its quotes are true.

    References:
        [REF-PYTHON-UNITTEST]. No mathematical equation is evaluated.

    """


for _path in documented_files():
    _slug = re.sub(r"\W+", "_", str(_path.relative_to(REPOSITORY_ROOT))[:-3]).lower()
    setattr(DocumentedExampleTests, f"test_{_slug}", _document_test(_path))


class DocumentationContractTests(unittest.TestCase):
    """Names, links, counts, exemptions, and pitfalls sections.

    References:
        [REF-PYTHON-AST]. No mathematical equation is evaluated.

    """

    def test_exemptions_are_rare_and_justified(self) -> None:
        """At most a few blocks are skipped, each with a stated reason.

        References:
            [REF-PYTHON-AST]. No mathematical equation is evaluated.

        """
        exemptions: list[str] = []
        for path in documented_files():
            for block in code_blocks(path.read_text(encoding="utf-8")):
                if block.language == "python" and block.exemption is not None:
                    self.assertGreaterEqual(len(block.exemption), 20, block.exemption)
                    exemptions.append(f"{path.name}:{block.line}")
        self.assertLessEqual(len(exemptions), MAX_EXEMPTIONS, exemptions)

    def test_every_documented_file_has_runnable_examples_or_none(self) -> None:
        """Only Python, text, shell, and PowerShell fences appear.

        A block in another language (for example ``py`` or ``pycon``) would
        escape execution silently.

        References:
            [REF-PYTHON-AST]. No mathematical equation is evaluated.

        """
        allowed = {"python", "text", "bash", "powershell", ""}
        for path in documented_files():
            for block in code_blocks(path.read_text(encoding="utf-8")):
                with self.subTest(file=path.name, line=block.line):
                    self.assertIn(block.language, allowed)

    def test_prerequisite_blocks_document_their_output(self) -> None:
        """Every prerequisite snippet is followed by its exact output block.

        References:
            [REF-PYTHON-AST]. No mathematical equation is evaluated.

        """
        for path in sorted((DOCS / "prerequisites").glob("*.md")):
            blocks = [
                block
                for block in code_blocks(path.read_text(encoding="utf-8"))
                if block.language == "python"
            ]
            with self.subTest(guide=path.name):
                self.assertGreaterEqual(len(blocks), 2)
                for block in blocks:
                    self.assertIsNotNone(block.output, f"{path.name}:{block.line}")

    def test_every_public_name_in_the_guides_exists(self) -> None:
        """Dotted library names in prose and imports in code resolve.

        Inline code such as ``reliability.form_hlrf``,
        ``probstat.modeling.JointInput``, or ``UQStudy.reliability`` must
        name an existing module, function, class, or attribute; every
        ``from probstat... import`` in any fenced block must import.

        References:
            [REF-PYTHON-AST]. No mathematical equation is evaluated.

        """
        exported = exported_names()
        unresolved: list[str] = []
        for path in documented_files():
            for span in INLINE_CODE.findall(prose(path)):
                name = span.strip().split("(", 1)[0].rstrip(".")
                if DOTTED_NAME.match(name) and resolve(name, exported) is False:
                    unresolved.append(f"{path.name}: `{span}`")
            for block in code_blocks(path.read_text(encoding="utf-8")):
                if block.language != "python":
                    continue
                for node in ast.walk(ast.parse(block.body)):
                    source = node.module if isinstance(node, ast.ImportFrom) else None
                    if not (source or "").startswith("probstat"):
                        continue
                    assert isinstance(node, ast.ImportFrom) and source is not None
                    module = importlib.import_module(source)
                    for alias in node.names:
                        if not hasattr(module, alias.name):
                            where = f"{path.name}:{block.line}"
                            unresolved.append(f"{where}: {source}.{alias.name}")
        self.assertEqual(unresolved, [], "\n".join(unresolved))

    def test_every_relative_link_resolves(self) -> None:
        """Relative links, images, and heading anchors point to real targets.

        References:
            [REF-PYTHON-AST]. No mathematical equation is evaluated.

        """
        broken: list[str] = []
        for path in documented_files():
            text = prose(path)
            targets = [a or b for a, b in LINK.findall(text)]
            targets += HTML_SOURCE.findall(text)
            for target in targets:
                if re.match(r"^[a-z]+:", target):
                    continue
                location, _, anchor = target.partition("#")
                resolved = (path.parent / location).resolve() if location else path
                if not resolved.exists():
                    broken.append(f"{path.name}: {target}")
                elif anchor and resolved.suffix == ".md":
                    if anchor not in heading_anchors(resolved):
                        broken.append(f"{path.name}: {target} (anchor)")
        self.assertEqual(broken, [], "\n".join(broken))

    def test_package_docstrings_state_purpose_entry_points_and_pitfalls(self) -> None:
        """Each package docstring has the four parts and names its guide.

        References:
            [REF-PYTHON-AST]. No mathematical equation is evaluated.

        """
        self.assertEqual(sorted(PACKAGE_GUIDES), list(package_names()))
        for name, guide in PACKAGE_GUIDES.items():
            source = (SOURCE / name / "__init__.py").read_text(encoding="utf-8")
            docstring = ast.get_docstring(ast.parse(source)) or ""
            with self.subTest(package=name):
                for section in DOCSTRING_SECTIONS:
                    self.assertIn(section, docstring)
                self.assertIn(f"Guide: docs/{guide}", docstring)
                _, _, references = docstring.partition("References:")
                self.assertRegex(references, r"\b(?:REF|EQ)-[A-Z0-9-]+\b")

    def test_every_guide_has_a_common_pitfalls_section(self) -> None:
        """Foundation and prerequisite guides end their teaching with pitfalls.

        References:
            [REF-PYTHON-AST]. No mathematical equation is evaluated.

        """
        for directory in ("foundations", "prerequisites"):
            for path in sorted((DOCS / directory).glob("*.md")):
                with self.subTest(guide=path.name):
                    self.assertRegex(
                        path.read_text(encoding="utf-8"), PITFALLS_HEADING
                    )


class ReadmeTests(unittest.TestCase):
    """The README's counts, tables, and install notes match the code.

    References:
        [REF-PYTHON-AST]. No mathematical equation is evaluated.

    """

    text = README.read_text(encoding="utf-8")

    def number_before(self, phrase: str) -> int:
        """Return the integer written just before ``phrase`` in the README.

        Args:
            phrase: Words that follow the number.

        Returns:
            The number, with thousands separators removed.

        References:
            [REF-PYTHON-AST]. No mathematical equation is evaluated.

        """
        words = r"\s+".join(re.escape(word) for word in phrase.split())
        match = re.search(r"([\d,]+)\s+" + words, self.text)
        self.assertIsNotNone(match, phrase)
        assert match is not None
        return int(match.group(1).replace(",", ""))

    def assert_floor(self, phrase: str, actual: int) -> None:
        """Check a count the README writes as a floor, such as ``2,200+ tests``.

        A floor stays true while the repository grows; it must not exceed the
        actual count and must not fall below 80% of it, so a stale floor is
        caught before it becomes misleading.

        Args:
            phrase: Words that follow ``<number>+``.
            actual: Count read from the repository.

        References:
            [REF-PYTHON-AST]. No mathematical equation is evaluated.

        """
        words = r"\s+".join(re.escape(word) for word in phrase.split())
        match = re.search(r"([\d,]+)\+\s+" + words, self.text)
        self.assertIsNotNone(match, phrase)
        assert match is not None
        floor = int(match.group(1).replace(",", ""))
        with self.subTest(phrase=phrase, floor=floor, actual=actual):
            self.assertLessEqual(floor, actual)
            self.assertGreaterEqual(floor, 0.8 * actual)

    def test_counts_match_the_generated_references(self) -> None:
        """Packages, public names, equations, labs, topics, and families.

        References:
            [REF-PYTHON-AST]. No mathematical equation is evaluated.

        """
        header = (DOCS / "api-reference.md").read_text(encoding="utf-8")
        match = re.search(r"(\d+) public names in (\d+) packages", header)
        assert match is not None
        self.assertEqual(self.number_before("packages and"), len(package_names()))
        self.assertEqual(self.number_before("packages and"), int(match.group(2)))
        self.assertEqual(self.number_before("public names"), int(match.group(1)))
        registry = (DOCS / "equation-registry.md").read_text(encoding="utf-8")
        self.assertEqual(
            self.number_before("entries in the"),
            len(re.findall(r"^### EQ-", registry, flags=re.MULTILINE)),
        )
        labs = len(list((REPOSITORY_ROOT / "examples").glob("*.py")))
        self.assertIn(f"({labs} of them)", self.text)
        matrix = (DOCS / "curriculum" / "athens-2026-coverage.md").read_text(
            encoding="utf-8"
        )
        topics = re.findall(r"^\| ATHENS-[\d.]+ \|.*$", matrix, flags=re.MULTILINE)
        implemented = [row for row in topics if "| Implemented |" in row]
        self.assertEqual(self.number_before("topics of the ATHENS"), len(topics))
        self.assertEqual(self.number_before("are\nimplemented"), len(implemented))
        # Composite laws (one module each since ADR-0090) and helpers are not
        # families; the README names the composite laws separately.
        not_families = {
            "affine",
            "mixture",
            "truncated",
            "support",
            "constraints",
            "empirical",
        }
        families = [
            info.name
            for info in pkgutil.iter_modules([str(SOURCE / "distributions")])
            if not info.name.startswith("_") and info.name not in not_families
        ]
        self.assertEqual(self.number_before("distribution families"), len(families))

    def test_version_badge_matches_the_package(self) -> None:
        """The version badge shows the version in ``pyproject.toml``.

        References:
            [REF-PYTHON-AST]. No mathematical equation is evaluated.

        """
        project = tomllib.loads(
            (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )["project"]
        self.assertIn(f"badge/version-{project['version']}-", self.text)

    def test_every_optional_extra_is_listed(self) -> None:
        """Each extra of ``pyproject.toml`` is named in the README.

        References:
            [REF-PYTHON-AST]. No mathematical equation is evaluated.

        """
        project = tomllib.loads(
            (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )["project"]
        for extra in project["optional-dependencies"]:
            with self.subTest(extra=extra):
                self.assertIn(f"`{extra}`", self.text)

    def test_study_stage_table_names_real_methods(self) -> None:
        """Every stage in "One study, every question" is a ``UQStudy`` method.

        References:
            [REF-PYTHON-AST]. No mathematical equation is evaluated.

        """
        from probstat.workflow import UQStudy

        section = self.text.split("## One study, every question", 1)[1]
        rows = section.split("\n## ", 1)[0].splitlines()
        table = "\n".join(row for row in rows if row.startswith("|"))
        stages = re.findall(r"`(\w+)`", table)
        self.assertGreaterEqual(len(stages), 15)
        for stage in stages:
            with self.subTest(stage=stage):
                self.assertTrue(callable(getattr(UQStudy, stage, None)), stage)

    def test_package_table_lists_every_package_once(self) -> None:
        """The "What's inside" table covers each package (``cost`` once).

        References:
            [REF-PYTHON-AST]. No mathematical equation is evaluated.

        """
        section = self.text.split("## What's inside", 1)[1].split("\n## ", 1)[0]
        listed: set[str] = set()
        for row in section.splitlines():
            cells = row.split("|")
            if len(cells) > 3 and cells[2].strip().startswith("`"):
                listed.update(re.findall(r"`(\w+)`", cells[2]))
        self.assertEqual(listed, set(package_names()))

    def test_verification_floors_match_the_repository(self) -> None:
        """Tests, sources, decisions, documents, and property tests (ADR-0099).

        References:
            [REF-PYTHON-AST]. No mathematical equation is evaluated.

        """
        tests = REPOSITORY_ROOT / "tests"
        bibliography = (DOCS / "references.md").read_text(encoding="utf-8")
        properties = sorted(tests.glob("test_*propert*.py"))
        properties += sorted(tests.glob("test_metamorphic*.py"))
        self.assertEqual(len(properties), 4)
        self.assert_floor(
            "test cases", count_test_methods(sorted(tests.glob("test_*.py")))
        )
        self.assert_floor(
            "primary sources",
            len(re.findall(r"^## REF-", bibliography, flags=re.MULTILINE)),
        )
        self.assert_floor(
            "decision records", len(list((DOCS / "adr").glob("ADR-*.md")))
        )
        self.assert_floor("Markdown files", len(documented_files()))
        self.assert_floor(
            "property-based and metamorphic tests", count_test_methods(properties)
        )

    def test_figure_ci_and_packaging_claims_match_their_sources(self) -> None:
        """Text floor, contrast ratios, CI versions, wheel check, dependencies.

        References:
            [REF-PYTHON-AST]. No mathematical equation is evaluated.

        """
        theme = (SOURCE / "visualization" / "theme.py").read_text(encoding="utf-8")
        floor = re.search(r"^ANNOTATION_SIZE = ([0-9.]+)", theme, flags=re.MULTILINE)
        assert floor is not None
        self.assertIn(f"at least {float(floor.group(1)):g} pt", self.text)
        accessibility = _module_constants(
            REPOSITORY_ROOT / "tests" / "test_visualization_accessibility.py"
        )
        self.assertIn(f"{accessibility['TEXT_MINIMUM']:g}:1 for text", self.text)
        self.assertIn(f"{accessibility['GRAPHIC_MINIMUM']:g}:1 for marks", self.text)
        workflow = (REPOSITORY_ROOT / ".github" / "workflows" / "ci.yml").read_text(
            encoding="utf-8"
        )
        matrix = re.search(r"python-version: \[(.*?)\]", workflow)
        assert matrix is not None
        versions = re.findall(r'"([^"]+)"', matrix.group(1))
        self.assertIn(f"Python {versions[0]} to {versions[-1]}", self.text)
        for step in ("twine check --strict", "tools/smoke_test_wheel.py"):
            with self.subTest(step=step):
                self.assertIn(step, workflow)
                self.assertIn(step.removeprefix("tools/"), self.text)
        project = tomllib.loads(
            (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        )["project"]
        self.assertEqual(project["dependencies"], [])
        self.assertIn("badge/runtime%20dependencies-0-", self.text)
        self.assertIn(f"version = {{{project['version']}}}", self.text)

    def test_layer_count_matches_the_architecture_test(self) -> None:
        """The number of dependency layers is the architecture test's.

        References:
            [REF-PYTHON-AST]. No mathematical equation is evaluated.

        """
        from test_architecture import LAYER

        self.assertEqual(
            self.number_before("dependency layers"), len(set(LAYER.values()))
        )

    def test_readme_diagrams_match_the_code(self) -> None:
        """The diagram record names real stages, classes, and layers.

        ``tools/build_readme_figures.py`` writes ``readme-diagrams.json``
        next to the images; the layers must equal the architecture test's
        map, every stage must be a ``UQStudy`` method, every class must be
        exported by its package, and the README must show every image.

        References:
            [REF-PYTHON-AST]. No mathematical equation is evaluated.

        """
        from test_architecture import LAYER

        from probstat.workflow import UQStudy

        record = json.loads(
            (DOCS / "images" / "readme-diagrams.json").read_text(encoding="utf-8")
        )
        expected: dict[str, list[str]] = {}
        for module, layer in LAYER.items():
            if module != "probstat":
                expected.setdefault(str(layer), []).append(module.split(".", 1)[1])
        self.assertEqual(record["layers"]["packages"], expected)
        for stage in record["workflow"]["stages"]:
            with self.subTest(stage=stage):
                self.assertTrue(callable(getattr(UQStudy, stage, None)))
        for name, package in record["workflow"]["classes"].items():
            with self.subTest(name=name):
                module = importlib.import_module(f"probstat.{package}")
                self.assertIn(name, getattr(module, "__all__", ()))
        for diagram in ("workflow", "layers"):
            for image in record[diagram]["images"]:
                with self.subTest(image=image):
                    self.assertTrue((DOCS / "images" / image).is_file())
                    url = f"{RAW_PREFIX}docs/images/{image}"
                    self.assertTrue(url in self.text, f"README does not show {url}")

    def test_readme_uses_only_absolute_links(self) -> None:
        """Every link and image is absolute, so it also resolves on PyPI.

        The README is the package's long description; PyPI does not resolve
        relative paths (ADR-0099).

        References:
            [REF-PYTHON-AST]. No mathematical equation is evaluated.

        """
        text = prose(README)
        targets = [a or b for a, b in LINK.findall(text)] + HTML_SOURCE.findall(text)
        self.assertGreater(len(targets), 40)
        relative = [target for target in targets if not re.match(r"^https://", target)]
        self.assertEqual(relative, [])

    def test_absolute_repository_urls_resolve(self) -> None:
        """Every repository URL in the README maps to a file in the repository.

        ``github.com/ayberkdt/Stats-Lab/blob/main/<path>`` and
        ``raw.githubusercontent.com/ayberkdt/Stats-Lab/main/<path>`` must
        name a file, ``.../tree/main/<path>`` a directory, and a heading
        anchor on a Markdown file must exist.

        References:
            [REF-PYTHON-AST]. No mathematical equation is evaluated.

        """
        found = REPOSITORY_URL.findall(self.text)
        self.assertGreater(len(found), 40)
        broken: list[str] = []
        for kind, path, anchor in found:
            target = REPOSITORY_ROOT / path
            exists = target.is_dir() if kind == "tree" else target.is_file()
            if not exists:
                broken.append(f"{kind}: {path}")
            elif anchor and target.suffix == ".md":
                if anchor not in heading_anchors(target):
                    broken.append(f"{kind}: {path}#{anchor} (anchor)")
        self.assertEqual(broken, [], "\n".join(broken))
        other = re.findall(
            r"https://[^\s\"')}]*ayberkdt/Stats-Lab[^\s\"')}]*", self.text
        )
        unmapped = [
            url
            for url in other
            if not REPOSITORY_URL.fullmatch(url)
            and url.rstrip("/") != "https://github.com/ayberkdt/Stats-Lab"
            and url != "https://github.com/ayberkdt/Stats-Lab.git"
        ]
        self.assertEqual(unmapped, [])


def count_test_methods(paths: list[Path]) -> int:
    """Count the ``test*`` methods defined directly in classes of test files.

    This equals what ``unittest`` discovery collects for this repository,
    without importing the modules (the rule of
    ``tools/build_presentation_data.py``).

    Args:
        paths: Test modules.

    Returns:
        Number of test methods.

    References:
        [REF-PYTHON-AST], [REF-PYTHON-UNITTEST].

    """
    total = 0
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                total += sum(
                    1
                    for item in node.body
                    if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef)
                    and item.name.startswith("test")
                )
    return total


def _module_constants(path: Path) -> dict[str, object]:
    """Return the literal module-level constants of a Python file.

    Args:
        path: Python source file.

    Returns:
        Name to value for every ``NAME = <literal>`` assignment.

    References:
        [REF-PYTHON-AST]. No mathematical equation is evaluated.

    """
    constants: dict[str, object] = {}
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    constants[target.id] = ast.literal_eval(node.value)
                except ValueError:
                    continue
    return constants


if __name__ == "__main__":
    unittest.main()
