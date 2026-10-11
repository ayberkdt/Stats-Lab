"""Keep the generated API reference in sync with the public API."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_generator() -> object:
    """Import ``tools/generate_api_reference.py`` as a module."""
    path = ROOT / "tools" / "generate_api_reference.py"
    spec = importlib.util.spec_from_file_location("generate_api_reference", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ApiReferenceTests(unittest.TestCase):
    """The committed reference matches a fresh rendering."""

    def test_reference_is_current(self) -> None:
        """Regenerate with ``python tools/generate_api_reference.py`` on failure."""
        generator = _load_generator()
        fresh = generator.render()  # type: ignore[attr-defined]
        committed = (ROOT / "docs" / "api-reference.md").read_text(encoding="utf-8")
        self.assertEqual(
            committed.replace("\r\n", "\n"),
            fresh,
            "docs/api-reference.md is stale; run "
            "PYTHONPATH=src python tools/generate_api_reference.py",
        )

    def test_every_package_is_grouped_and_every_name_listed(self) -> None:
        """Each public name of each package appears in the reference."""
        generator = _load_generator()
        text = generator.render()  # type: ignore[attr-defined]
        import probstat.design

        for name in probstat.design.__all__:
            self.assertIn(f"| `{name}` |", text)


if __name__ == "__main__":
    unittest.main()
