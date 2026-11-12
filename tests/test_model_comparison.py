"""Nested F tests, ANOVA tables, Breusch-Pagan, and Durbin-Watson.

Oracles:

- Montgomery (2017), Example 5.1: battery life of three materials at three
  temperatures, four replicates. The published table gives sums of squares
  10683.72 (material), 39118.72 (temperature), 9613.78 (interaction),
  18230.75 (error), F = 7.91, 28.97, 3.56 and P = 0.0020, < 0.0001, 0.0186.
  The test also recomputes the sums of squares from cell means.
- Balanced data with sum coding: types I, II, and III coincide.
- Unbalanced data: type I sums add up to the model sum of squares; type II
  of a term equals type I with that term entered after the terms that do
  not contain it; type III equals the drop-one-term difference; type III
  with sum coding is invariant to relabelling the levels.
- The nested F test of group indicators against the intercept equals the
  one-way ANOVA F; dropping one column gives ``F = t^2``.
- Breusch-Pagan: Koenker's statistic is ``n R^2`` of the auxiliary
  regression computed here independently; on homoscedastic data the test
  rejects at about its level, on variance growing with ``x`` it rejects.
- Durbin-Watson: the defining ratio, and ``d ~ 2 (1 - r_1)`` for AR(1)
  errors.

References:
    [REF-MONTGOMERY-2017, Chapter 5], [REF-BREUSCH-PAGAN-1979],
    [REF-KOENKER-1981], [REF-DURBIN-WATSON-1950].

"""

from __future__ import annotations

import math
import random
import unittest

from probstat.inference import (
    breusch_pagan_test,
    durbin_watson,
    factorial_design,
    fit_linear_model,
    linear_model_anova,
    nested_f_test,
    one_way_anova,
)

BATTERY = {
    1: {15: (130, 155, 74, 180), 70: (34, 40, 80, 75), 125: (20, 70, 82, 58)},
    2: {15: (150, 188, 159, 126), 70: (136, 122, 106, 115), 125: (25, 70, 58, 45)},
    3: {15: (138, 110, 168, 160), 70: (174, 120, 150, 139), 125: (96, 104, 82, 60)},
}


def battery() -> tuple[list[int], list[int], list[float]]:
    """Return material, temperature, and life in long form."""
    material, temperature, life = [], [], []
    for m, cells in BATTERY.items():
        for t, values in cells.items():
            for value in values:
                material.append(m)
                temperature.append(t)
                life.append(float(value))
    return material, temperature, life


class MontgomeryBatteryTests(unittest.TestCase):
    """The two-factor factorial of Montgomery's Example 5.1."""

    def setUp(self) -> None:
        """Build the full-factorial design."""
        self.material, self.temperature, self.life = battery()
        self.design, self.terms = factorial_design(
            {"material": self.material, "temperature": self.temperature}
        )

    def test_published_table(self) -> None:
        """Sums of squares, F statistics, and P-values of Table 5.5."""
        table = linear_model_anova(self.design, self.life, self.terms, kind="III")
        expected = {
            "material": (2, 10683.72, 7.91, 0.0020),
            "temperature": (2, 39118.72, 28.97, None),
            "material:temperature": (4, 9613.78, 3.56, 0.0186),
        }
        for term, (df, ss, f, p) in expected.items():
            row = table.row(term)
            self.assertEqual(row.df, df)
            self.assertAlmostEqual(row.sum_squares, ss, places=2)
            self.assertAlmostEqual(row.f_statistic, f, places=2)
            if p is None:
                self.assertLess(row.p_value, 1e-4)
            else:
                self.assertAlmostEqual(row.p_value, p, places=4)
        residual = table.row("Residuals")
        self.assertEqual(residual.df, 27)
        self.assertAlmostEqual(residual.sum_squares, 18230.75, places=2)

    def test_cell_mean_formulas(self) -> None:
        """``SS_A = b n sum (ybar_i - ybar)^2`` and the interaction formula."""
        grand = math.fsum(self.life) / 36
        rows = {m: math.fsum(v for c in BATTERY[m].values() for v in c) / 12
                for m in BATTERY}  # fmt: skip
        cols = {
            t: math.fsum(v for m in BATTERY for v in BATTERY[m][t]) / 12
            for t in (15, 70, 125)
        }
        cell = {(m, t): math.fsum(BATTERY[m][t]) / 4 for m in BATTERY for t in cols}
        ss_a = 12 * math.fsum((v - grand) ** 2 for v in rows.values())
        ss_ab = 4 * math.fsum(
            (cell[m, t] - rows[m] - cols[t] + grand) ** 2 for m, t in cell
        )
        table = linear_model_anova(self.design, self.life, self.terms, kind="I")
        self.assertAlmostEqual(table.row("material").sum_squares, ss_a, places=8)
        self.assertAlmostEqual(
            table.row("material:temperature").sum_squares, ss_ab, places=8
        )

    def test_balanced_types_coincide(self) -> None:
        """With sum coding and equal cell sizes, I = II = III."""
        tables = [
            linear_model_anova(self.design, self.life, self.terms, kind=kind)
            for kind in ("I", "II", "III")
        ]
        for first, second in zip(tables, tables[1:], strict=False):
            for a, b in zip(first.rows, second.rows, strict=True):
                self.assertAlmostEqual(a.sum_squares, b.sum_squares, places=6)


class UnbalancedAnovaTests(unittest.TestCase):
    """Three observations removed from the battery data."""

    def setUp(self) -> None:
        """Drop cases 0, 5, and 30."""
        material, temperature, life = battery()
        keep = [i for i in range(36) if i not in (0, 5, 30)]
        self.material = [material[i] for i in keep]
        self.temperature = [temperature[i] for i in keep]
        self.life = [life[i] for i in keep]
        self.design, self.terms = factorial_design(
            {"material": self.material, "temperature": self.temperature}
        )

    def test_type_one_sums_to_the_model_sum_of_squares(self) -> None:
        """Sequential sums of squares partition ``TSS - RSS``."""
        table = linear_model_anova(self.design, self.life, self.terms, kind="I")
        model = table.fit.total_sum_squares - table.fit.residual_sum_squares
        terms = math.fsum(row.sum_squares for row in table.rows[:-1])
        self.assertAlmostEqual(terms, model, places=6)

    def test_type_two_is_type_one_with_the_term_entered_last_among_peers(self) -> None:
        """``SS_II(material) = SS_I(material | temperature)``."""
        swapped = {
            "temperature": self.terms["temperature"],
            "material": self.terms["material"],
            "material:temperature": self.terms["material:temperature"],
        }
        sequential = linear_model_anova(self.design, self.life, swapped, kind="I")
        marginal = linear_model_anova(self.design, self.life, self.terms, kind="II")
        self.assertAlmostEqual(
            marginal.row("material").sum_squares,
            sequential.row("material").sum_squares,
            places=6,
        )
        self.assertNotAlmostEqual(
            marginal.row("material").sum_squares,
            linear_model_anova(self.design, self.life, self.terms, kind="I")
            .row("material")
            .sum_squares,
            places=1,
        )

    def test_type_three_is_the_drop_one_term_difference(self) -> None:
        """Refit without the material columns and compare residual sums."""
        full = fit_linear_model(self.design.rows, self.life)
        dropped = [
            [v for j, v in enumerate(row) if j not in self.terms["material"]]
            for row in self.design.rows
        ]
        reduced = fit_linear_model(dropped, self.life)
        table = linear_model_anova(self.design, self.life, self.terms, kind="III")
        self.assertAlmostEqual(
            table.row("material").sum_squares,
            reduced.residual_sum_squares - full.residual_sum_squares,
            places=6,
        )

    def test_sum_coded_type_three_ignores_level_labels(self) -> None:
        """Renaming levels changes the last (``-1``) level but not SS_III."""
        renamed = [{1: "c", 2: "a", 3: "b"}[m] for m in self.material]
        design, terms = factorial_design(
            {"material": renamed, "temperature": self.temperature}
        )
        first = linear_model_anova(self.design, self.life, self.terms, kind="III")
        second = linear_model_anova(design, self.life, terms, kind="III")
        for a, b in zip(first.rows, second.rows, strict=True):
            self.assertAlmostEqual(a.sum_squares, b.sum_squares, places=6)

    def test_treatment_coded_type_three_main_effect_differs(self) -> None:
        """Treatment coding tests a simple effect at the reference level."""
        design, terms = factorial_design(
            {"material": self.material, "temperature": self.temperature},
            coding="treatment",
        )
        effect = linear_model_anova(self.design, self.life, self.terms, kind="III")
        treatment = linear_model_anova(design, self.life, terms, kind="III")
        self.assertNotAlmostEqual(
            effect.row("material").sum_squares,
            treatment.row("material").sum_squares,
            places=1,
        )
        self.assertAlmostEqual(
            effect.row("material:temperature").sum_squares,
            treatment.row("material:temperature").sum_squares,
            places=6,
        )


class NestedFTests(unittest.TestCase):
    """The extra-sum-of-squares F test."""

    def test_group_indicators_reproduce_one_way_anova(self) -> None:
        """Intercept-only against material indicators."""
        material, _, life = battery()
        design, _ = factorial_design({"material": material})
        reduced = fit_linear_model([(1.0,)] * len(life), life)
        full = fit_linear_model(design.rows, life)
        test = nested_f_test(reduced, full)
        groups = [[y for m, y in zip(material, life, strict=True) if m == k]
                  for k in (1, 2, 3)]  # fmt: skip
        anova = one_way_anova(*groups)
        self.assertAlmostEqual(test.statistic, anova.statistic, places=10)
        self.assertAlmostEqual(test.p_value, anova.p_value, places=12)
        self.assertEqual((test.numerator_df, test.denominator_df), (2, 33))

    def test_one_dropped_column_gives_t_squared(self) -> None:
        """``F = t^2`` and equal p-values."""
        rng = random.Random(4)
        rows = [(1.0, rng.gauss(0, 1), rng.gauss(0, 1)) for _ in range(40)]
        y = [1 + 0.5 * a + 0.2 * b + rng.gauss(0, 1) for _, a, b in rows]
        full = fit_linear_model(rows, y)
        reduced = fit_linear_model([row[:2] for row in rows], y)
        test = nested_f_test(reduced, full)
        self.assertAlmostEqual(test.statistic, full.t_values[2] ** 2, places=9)
        self.assertAlmostEqual(test.p_value, full.p_values[2], places=10)

    def test_rejects_non_nested_or_different_data(self) -> None:
        """Different responses or a column outside the full space."""
        x = [float(i) for i in range(12)]
        y = [math.sin(v) + v for v in x]
        line = fit_linear_model([(1.0, v) for v in x], y)
        square = fit_linear_model([(1.0, v * v, v**3) for v in x], y)
        with self.assertRaises(ValueError):
            nested_f_test(line, square)
        other = fit_linear_model([(1.0, v, v * v) for v in x], [v + 1 for v in y])
        with self.assertRaises(ValueError):
            nested_f_test(line, other)
        with self.assertRaises(ValueError):
            nested_f_test(line, line)


class ResidualCheckTests(unittest.TestCase):
    """Breusch-Pagan and Durbin-Watson."""

    def test_koenker_statistic_is_n_r_squared(self) -> None:
        """Auxiliary regression of squared residuals on the design."""
        rng = random.Random(7)
        x = [rng.uniform(0, 10) for _ in range(80)]
        y = [2 + 0.5 * v + rng.gauss(0, 0.2 + 0.2 * v) for v in x]
        fit = fit_linear_model([(1.0, v) for v in x], y)
        auxiliary = fit_linear_model(
            [(1.0, v) for v in x], [e * e for e in fit.residuals]
        )
        test = breusch_pagan_test(fit)
        self.assertAlmostEqual(test.statistic, 80 * auxiliary.r_squared, places=10)
        self.assertEqual(test.df, 1)
        self.assertLess(test.p_value, 0.01)
        explained = auxiliary.total_sum_squares - auxiliary.residual_sum_squares
        sigma2 = fit.residual_sum_squares / 80
        original = breusch_pagan_test(fit, studentized=False)
        self.assertAlmostEqual(
            original.statistic, explained / (2 * sigma2**2), places=8
        )

    def test_size_under_constant_variance(self) -> None:
        """About 5% rejections at the 5% level (200 replications).

        ``P(X >= 20)`` for ``X ~ Binomial(200, 0.05)`` is below 0.002.
        """
        rng = random.Random(11)
        rejections = 0
        for _ in range(200):
            x = [rng.uniform(0, 10) for _ in range(60)]
            y = [1 + 0.3 * v + rng.gauss(0, 1) for v in x]
            fit = fit_linear_model([(1.0, v) for v in x], y)
            rejections += breusch_pagan_test(fit).p_value < 0.05
        self.assertLess(rejections, 20)

    def test_durbin_watson_definition_and_ar1(self) -> None:
        """``d = sum (e_t - e_(t-1))^2 / sum e_t^2 ~ 2 (1 - rho)``."""
        self.assertAlmostEqual(durbin_watson([1.0, -1.0, 1.0, -1.0]), 3.0)
        rng = random.Random(3)
        errors = [0.0]
        for _ in range(2000):
            errors.append(0.6 * errors[-1] + rng.gauss(0, 1))
        x = [float(t) for t in range(len(errors))]
        y = [0.01 * t + e for t, e in zip(x, errors, strict=True)]
        fit = fit_linear_model([(1.0, t) for t in x], y)
        self.assertAlmostEqual(durbin_watson(fit), 2 * (1 - 0.6), delta=0.1)
        with self.assertRaises(ValueError):
            durbin_watson([0.0, 0.0])


class ValidationTests(unittest.TestCase):
    """Argument checks."""

    def test_invalid_arguments(self) -> None:
        """Unknown kind, overlapping or empty terms, bad coding, bad factors."""
        material, temperature, life = battery()
        design, terms = factorial_design({"m": material, "t": temperature})
        with self.assertRaises(ValueError):
            linear_model_anova(design, life, terms, kind="IV")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            linear_model_anova(design, life, {"a": (1, 2), "b": (2, 3)})
        with self.assertRaises(ValueError):
            linear_model_anova(design, life, {"a": ()})
        with self.assertRaises(ValueError):
            factorial_design({"m": material}, coding="helmert")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            factorial_design({"m": [1] * 5})
        with self.assertRaises(ValueError):
            factorial_design({"m": material}, interactions=("m:z",))


if __name__ == "__main__":
    unittest.main()
