"""Contract tests for the complete ATHENS 2026 curriculum ledger."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[1]
MATRIX = REPOSITORY_ROOT / "docs" / "curriculum" / "athens-2026-coverage.md"
ROW_PATTERN = re.compile(
    r"^\| (ATHENS-\d+\.\d+) \| (.+?) \| (\w+) \| (.+?) \| (.+?) \| (.+?) \|$"
)
GROUP_SIZES = {
    1: 10,
    2: 12,
    3: 11,
    4: 12,
    5: 11,
    6: 11,
    7: 15,
    8: 15,
    9: 6,
    10: 7,
}
VALID_STATUSES = {"Implemented", "Partial", "Planned"}


def curriculum_rows() -> list[tuple[str, str, str, str, str, str]]:
    """Parse all curriculum rows into six contract fields.

    Returns:
        Ordered row fields from the canonical Markdown matrix.

    References:
        ATHENS 2026 curriculum scope. No mathematical equation is evaluated.

    """
    rows = []
    for line in MATRIX.read_text(encoding="utf-8").splitlines():
        if line.startswith("| ATHENS-"):
            match = ROW_PATTERN.fullmatch(line)
            if match is None:
                raise AssertionError(f"Malformed curriculum row: {line}")
            rows.append(match.groups())
    return rows


class AthensCoverageTests(unittest.TestCase):
    """Prevent omissions and unsubstantiated completion claims."""

    def test_all_110_curriculum_ids_are_present_once(self) -> None:
        """Every supplied topic has one stable identity and no duplicate."""
        expected = {
            f"ATHENS-{group}.{item}"
            for group, size in GROUP_SIZES.items()
            for item in range(1, size + 1)
        }
        identifiers = [row[0] for row in curriculum_rows()]
        self.assertEqual(len(identifiers), 110)
        self.assertEqual(len(set(identifiers)), 110)
        self.assertEqual(set(identifiers), expected)

    def test_status_target_and_owner_are_explicit(self) -> None:
        """Every topic has a controlled status and non-empty delivery owner."""
        for identifier, topic, status, target, owner, _ in curriculum_rows():
            with self.subTest(identifier=identifier):
                self.assertTrue(topic.strip())
                self.assertIn(status, VALID_STATUSES)
                self.assertNotEqual(target.strip(), "—")
                self.assertNotEqual(owner.strip(), "—")

    def test_claimed_evidence_paths_exist(self) -> None:
        """Implemented and partial claims resolve to repository artifacts."""
        for identifier, _, status, _, _, evidence in curriculum_rows():
            with self.subTest(identifier=identifier):
                if status == "Planned":
                    self.assertEqual(evidence, "—")
                    continue
                self.assertNotEqual(evidence, "—")
                for raw_path in evidence.split(";"):
                    relative = raw_path.strip().strip("`")
                    self.assertTrue(
                        (REPOSITORY_ROOT / relative).exists(),
                        f"Missing evidence for {identifier}: {relative}",
                    )


if __name__ == "__main__":
    unittest.main()
