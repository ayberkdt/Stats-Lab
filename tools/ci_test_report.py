"""Run the test suite and report failures as GitHub Actions annotations.

The suite runs with :mod:`unittest` discovery exactly as ``python -m unittest
discover -s tests`` does. Every failing or erroring test is also printed as a
workflow ``::error`` command, so it appears in the run's annotations, which
are readable without downloading the full log. The exit status is non-zero
when any test fails or errors.

References:
    [REF-PYTHON-UNITTEST], GitHub Actions workflow commands
    (``::error title=...::message``).

"""

from __future__ import annotations

import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MAX_MESSAGE_LINES = 12


def escape(text: str) -> str:
    """Escape a value for a workflow command.

    Args:
        text: Raw annotation text.

    Returns:
        Text with ``%``, carriage return, and line feed percent-encoded.

    References:
        [REF-PYTHON-UNITTEST]. No equation is evaluated.

    """
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def annotate(kind: str, test: unittest.TestCase, trace: str) -> None:
    """Print one ``::error`` command for a failed or erroring test.

    Args:
        kind: ``"FAIL"`` or ``"ERROR"``.
        test: The test case.
        trace: Formatted traceback from the result.

    References:
        [REF-PYTHON-UNITTEST]. No equation is evaluated.

    """
    lines = trace.strip().splitlines()
    tail = "\n".join(lines[-MAX_MESSAGE_LINES:])
    title = escape(f"{kind}: {test.id()}")
    print(f"::error title={title}::{escape(tail)}", flush=True)


def main() -> int:
    """Discover and run ``tests/``, annotating failures and errors.

    Returns:
        Process exit status: 0 when every test passes.

    References:
        [REF-PYTHON-UNITTEST]. No equation is evaluated.

    """
    # Same discovery as ``python -m unittest discover -s tests``: the start
    # directory is also the top-level directory, so test helpers in tests/
    # import by their bare module names.
    suite = unittest.defaultTestLoader.discover(str(REPOSITORY_ROOT / "tests"))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    for test, trace in result.failures:
        annotate("FAIL", test, trace)
    for test, trace in result.errors:
        annotate("ERROR", test, trace)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
