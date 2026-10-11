"""Tests for API stability labels and the deprecation decorator."""

from __future__ import annotations

import pkgutil
import unittest
import warnings

import probstat
from probstat.foundations import PACKAGE_STABILITY, deprecated, package_stability


class StabilityLabelTests(unittest.TestCase):
    """Every top-level package carries exactly one valid label."""

    def test_every_package_is_labelled(self) -> None:
        """Labels cover the installed packages and nothing else."""
        packages = {
            info.name
            for info in pkgutil.iter_modules(probstat.__path__)
            if info.ispkg and not info.name.startswith("_")
        }
        self.assertEqual(set(PACKAGE_STABILITY), packages)
        self.assertLessEqual(
            set(PACKAGE_STABILITY.values()), {"stable", "provisional", "experimental"}
        )

    def test_lookup_accepts_dotted_names(self) -> None:
        """Submodules inherit their package's label."""
        self.assertEqual(package_stability("probstat.linalg.qr"), "stable")
        self.assertEqual(package_stability("causal"), "experimental")
        with self.assertRaises(KeyError):
            package_stability("probstat.nonexistent")


class DeprecatedDecoratorTests(unittest.TestCase):
    """The decorator warns at the caller and keeps the wrapped behaviour."""

    def test_warning_message_and_behaviour(self) -> None:
        """Calls still work, warn once per call, and point at the caller."""

        @deprecated(since="0.15", removal="0.17", alternative="new_square")
        def old_square(x: float) -> float:
            """Return x squared."""
            return x * x

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(old_square(3.0), 9.0)
        self.assertEqual(len(caught), 1)
        self.assertIs(caught[0].category, DeprecationWarning)
        text = str(caught[0].message)
        self.assertIn("since 0.15", text)
        self.assertIn("removed in 0.17", text)
        self.assertIn("use new_square instead", text)
        self.assertEqual(caught[0].filename, __file__)
        self.assertEqual(old_square.__name__, "old_square")
        self.assertIn("Deprecated:", old_square.__doc__ or "")

    def test_versions_are_required(self) -> None:
        """Empty version strings are rejected when the decorator is built."""
        with self.assertRaises(ValueError):
            deprecated(since="", removal="0.17")


if __name__ == "__main__":
    unittest.main()
