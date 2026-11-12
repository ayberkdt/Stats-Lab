"""Estimated marginal means and Tukey-Kramer contrasts.

Oracles:

- Montgomery (2017), Example 5.1 (battery life): the material means
  83.17, 108.33, 125.08 with standard error ``sqrt(MSE / 12)``,
  ``MSE = 18230.75 / 27``; at 70 degrees F the cell means 57.25, 119.75,
  145.75 and Tukey's ``T_0.05 = q_0.05(3, 27) sqrt(MSE / 4)``, which
  separates material 1 from 2 and 3 but not 2 from 3 (Montgomery's
  conclusion; his table value ``q = 3.50`` gives 45.47).
- Unbalanced data with the full interaction model: marginal means are the
  unweighted averages of the cell sample means, not the raw level means.
- Additive model: marginal means equal the equal-weight average of the
  fitted values at the grid cells, computed here from the coefficients.
- Adjusted p-values: ``P(Q >= sqrt(2) |t|)`` by the studentized range.

References:
    [REF-MONTGOMERY-2017, Chapter 5], [REF-SEARLE-SPEED-MILLIKEN-1980],
    [REF-KRAMER-1956].

"""

from __future__ import annotations

import math
import statistics
import unittest

from probstat.inference import (
    factorial_design,
    marginal_means,
    studentized_range_quantile,
    studentized_range_sf,
)

MATERIAL = [m for m in (1, 2, 3) for _ in range(12)]
TEMPERATURE = [t for _ in range(3) for t in (15, 70, 125) for _ in range(4)]
LIFE = [
    130, 155, 74, 180, 34, 40, 80, 75, 20, 70, 82, 58,
    150, 188, 159, 126, 136, 122, 106, 115, 25, 70, 58, 45,
    138, 110, 168, 160, 174, 120, 150, 139, 96, 104, 82, 60,
]  # fmt: skip
MSE = 18230.75 / 27


class BalancedTests(unittest.TestCase):
    """The battery-life factorial."""

    def setUp(self) -> None:
        """Collect the factors."""
        self.factors = {"material": MATERIAL, "temperature": TEMPERATURE}

    def test_material_means_and_errors(self) -> None:
        """Raw marginal means with standard error ``sqrt(MSE / 12)``."""
        result = marginal_means(self.factors, LIFE, "material")
        for level, expected in (("1", 83.1667), ("2", 108.3333), ("3", 125.0833)):
            mean = result.mean(level)
            self.assertAlmostEqual(mean.estimate, expected, places=4)
            self.assertAlmostEqual(mean.standard_error, math.sqrt(MSE / 12), places=9)
        self.assertEqual(result.df, 27)

    def test_tukey_at_seventy_degrees(self) -> None:
        """Montgomery's simple-effect comparison of the materials at 70 F."""
        result = marginal_means(self.factors, LIFE, "material", at={"temperature": 70})
        for level, expected in (("1", 57.25), ("2", 119.75), ("3", 145.75)):
            self.assertAlmostEqual(result.mean(level).estimate, expected, places=9)
        threshold = studentized_range_quantile(0.95, 3, 27) * math.sqrt(MSE / 4)
        self.assertAlmostEqual(threshold, 45.5, delta=0.1)
        for first, second, significant in (
            ("1", "2", True),
            ("1", "3", True),
            ("2", "3", False),
        ):
            contrast = result.contrast(first, second)
            self.assertAlmostEqual(
                contrast.upper - contrast.difference, threshold, places=8
            )
            self.assertEqual(contrast.p_value < 0.05, significant)
            self.assertAlmostEqual(
                contrast.p_value,
                studentized_range_sf(
                    abs(contrast.difference) / math.sqrt(MSE / 4), 3, 27
                ),
                places=10,
            )
        reverse = result.contrast("2", "1")
        self.assertAlmostEqual(reverse.difference, 62.5, places=9)
        self.assertEqual(result.at, (("temperature", "70"),))


class UnbalancedTests(unittest.TestCase):
    """Three batteries removed."""

    def setUp(self) -> None:
        """Drop cases 0, 5, and 30."""
        keep = [i for i in range(36) if i not in (0, 5, 30)]
        self.factors = {
            "material": [MATERIAL[i] for i in keep],
            "temperature": [TEMPERATURE[i] for i in keep],
        }
        self.life = [LIFE[i] for i in keep]

    def test_full_model_means_average_the_cell_means(self) -> None:
        """Equal weights per cell, not per observation."""
        result = marginal_means(self.factors, self.life, "material")
        cells: dict[tuple[int, int], list[float]] = {}
        for m, t, y in zip(
            self.factors["material"],
            self.factors["temperature"],
            self.life,
            strict=True,
        ):
            cells.setdefault((m, t), []).append(float(y))
        for m in (1, 2, 3):
            expected = statistics.fmean(
                statistics.fmean(cells[m, t]) for t in (15, 70, 125)
            )
            raw = statistics.fmean(
                y for (mm, _), ys in cells.items() if mm == m for y in ys
            )
            self.assertAlmostEqual(result.mean(m).estimate, expected, places=9)
            if m != 2:
                self.assertNotAlmostEqual(result.mean(m).estimate, raw, places=2)

    def test_additive_model_means_from_the_coefficients(self) -> None:
        """Average the additive predictions over the three temperatures."""
        result = marginal_means(
            self.factors, self.life, "temperature", interactions=False
        )
        design, _ = factorial_design(self.factors, interactions=False)
        beta = result.fit.coefficients
        # Sum coding: material effects a1, a2, a3 = -(a1 + a2) average to 0.
        self.assertEqual(
            design.column_names[3:], ("temperature[125]", "temperature[15]")
        )
        effects = {"125": beta[3], "15": beta[4], "70": -(beta[3] + beta[4])}
        for level, effect in effects.items():
            self.assertAlmostEqual(
                result.mean(level).estimate, beta[0] + effect, places=9
            )

    def test_coding_does_not_change_the_means(self) -> None:
        """Treatment and sum coding span the same model."""
        effect = marginal_means(self.factors, self.life, "material")
        treatment = marginal_means(
            self.factors, self.life, "material", coding="treatment"
        )
        for a, b in zip(effect.means, treatment.means, strict=True):
            self.assertAlmostEqual(a.estimate, b.estimate, places=8)
            self.assertAlmostEqual(a.standard_error, b.standard_error, places=8)


class ValidationTests(unittest.TestCase):
    """Argument checks."""

    def test_invalid_arguments(self) -> None:
        """Unknown factor or level, a fixed compared factor, bad confidence."""
        factors = {"material": MATERIAL, "temperature": TEMPERATURE}
        with self.assertRaises(KeyError):
            marginal_means(factors, LIFE, "pressure")
        with self.assertRaises(KeyError):
            marginal_means(factors, LIFE, "material", at={"temperature": 99})
        with self.assertRaises(ValueError):
            marginal_means(factors, LIFE, "material", at={"material": 1})
        with self.assertRaises(ValueError):
            marginal_means(factors, LIFE, "material", confidence=1.0)


if __name__ == "__main__":
    unittest.main()
