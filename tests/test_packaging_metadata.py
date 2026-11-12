"""Consistency of the distribution metadata with the package and CI (ADR-0096)."""

from __future__ import annotations

import re
import tomllib
import unittest
from pathlib import Path

import probstat

REPOSITORY_ROOT = Path(__file__).parents[1]
PYPROJECT = REPOSITORY_ROOT / "pyproject.toml"
CITATION = REPOSITORY_ROOT / "CITATION.cff"
CI_WORKFLOW = REPOSITORY_ROOT / ".github" / "workflows" / "ci.yml"
REPOSITORY_URL = "https://github.com/ayberkdt/Stats-Lab"


def project_table() -> dict[str, object]:
    """Return the ``[project]`` table of ``pyproject.toml``.

    Returns:
        The parsed table.

    References:
        [REF-PYPA-CORE-METADATA]. No equation is evaluated.

    """
    with PYPROJECT.open("rb") as handle:
        table = tomllib.load(handle)["project"]
    assert isinstance(table, dict)
    return table


class PackagingMetadataTests(unittest.TestCase):
    """The metadata a release publishes agrees with the code and the CI."""

    def test_versions_agree(self) -> None:
        """``pyproject.toml``, ``probstat.__version__``, and CITATION.cff agree."""
        citation = re.search(
            r"^version:\s*(\S+)\s*$", CITATION.read_text(encoding="utf-8"), re.M
        )
        self.assertIsNotNone(citation)
        assert citation is not None
        self.assertEqual(project_table()["version"], probstat.__version__)
        self.assertEqual(citation.group(1), probstat.__version__)

    def test_python_classifiers_match_the_ci_matrix(self) -> None:
        """Every tested Python version is declared, and only those."""
        matrix = re.search(
            r"python-version:\s*\[([^\]]*)\]", CI_WORKFLOW.read_text(encoding="utf-8")
        )
        assert matrix is not None
        tested = {item.strip().strip("\"'") for item in matrix.group(1).split(",")}
        classifiers = project_table()["classifiers"]
        assert isinstance(classifiers, list)
        declared = {
            item.rsplit(" :: ", 1)[1]
            for item in classifiers
            if re.fullmatch(r"Programming Language :: Python :: 3\.\d+", item)
        }
        self.assertEqual(declared, tested)
        self.assertEqual(project_table()["requires-python"], f">={min(tested)}")
        self.assertIn("Typing :: Typed", classifiers)
        self.assertTrue((REPOSITORY_ROOT / "src" / "probstat" / "py.typed").is_file())

    def test_project_urls_point_to_the_repository(self) -> None:
        """Every project URL lives under the public repository."""
        urls = project_table()["urls"]
        assert isinstance(urls, dict)
        self.assertEqual(
            set(urls),
            {"Homepage", "Repository", "Issues", "Changelog", "Documentation"},
        )
        for url in urls.values():
            self.assertTrue(str(url).startswith(REPOSITORY_URL), url)

    def test_license_files_exist(self) -> None:
        """The licence files named for the distribution are in the tree."""
        patterns = project_table()["license-files"]
        assert isinstance(patterns, list)
        for pattern in patterns:
            self.assertTrue(list(REPOSITORY_ROOT.glob(pattern)), pattern)

    def test_ci_smoke_tests_the_installed_wheel(self) -> None:
        """CI builds, checks, installs, and smoke-tests the distributions."""
        workflow = CI_WORKFLOW.read_text(encoding="utf-8")
        for step in ("-m build", "twine check --strict", "smoke_test_wheel.py"):
            self.assertIn(step, workflow)
        self.assertTrue((REPOSITORY_ROOT / "tools" / "smoke_test_wheel.py").is_file())


if __name__ == "__main__":
    unittest.main()
