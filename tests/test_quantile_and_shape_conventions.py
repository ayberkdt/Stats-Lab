"""Hyndman-Fan sample quantiles and the Joanes-Gill shape conventions.

Oracles: the nine definitions of Hyndman and Fan evaluated in exact rational
arithmetic at decimal levels, the identities of the paper (plotting
positions, averaging at the jumps of type 1, the median, symmetry,
monotonicity), the k-statistic forms ``G1 = k3 / k2^(3/2)`` and
``G2 = k4 / k2^2`` in exact arithmetic, the exact normal-theory moments
``E[G2] = 0`` and ``E[g2] = -6/(n + 1)`` judged by their Monte Carlo
standard errors, and constants from R 4.6.2 (``stats::quantile``) and
e1071 1.7.17 pasted below.

References:
    [EQ-STAT-QUANT-001], [EQ-STAT-QUANT-002], [EQ-STAT-SHAPE-001],
    [EQ-STAT-SHAPE-002], [REF-HYNDMAN-FAN-1996], [REF-JOANES-GILL-1998],
    [REF-R-E1071].

"""

from __future__ import annotations

import math
import random
import statistics as standard_library
import unittest
from fractions import Fraction

from probstat.statistics import (
    excess_kurtosis,
    median,
    quantile,
    quantiles,
    skewness,
)

_PLOTTING = {
    4: (Fraction(0), Fraction(1)),
    5: (Fraction(1, 2), Fraction(1, 2)),
    6: (Fraction(0), Fraction(0)),
    7: (Fraction(1), Fraction(1)),
    8: (Fraction(1, 3), Fraction(1, 3)),
    9: (Fraction(3, 8), Fraction(3, 8)),
}


def exact_quantile(values: list[int], level: Fraction, kind: int) -> Fraction:
    """Evaluate a Hyndman-Fan definition in exact rational arithmetic.

    Args:
        values: Integer sample.
        level: Exact rational level in ``[0, 1]``.
        kind: Type from 1 to 9.

    Returns:
        Exact quantile.

    References:
        [EQ-STAT-QUANT-002], [REF-HYNDMAN-FAN-1996].

    """
    x = [Fraction(v) for v in sorted(values)]
    n = len(x)

    def order(index: int) -> Fraction:
        return x[min(max(index, 1), n) - 1]

    if kind <= 3:
        m = Fraction(-1, 2) if kind == 3 else Fraction(0)
        j = math.floor(n * level + m)
        g = n * level + m - j
        if kind == 1:
            gamma = Fraction(0) if g == 0 else Fraction(1)
        elif kind == 2:
            gamma = Fraction(1, 2) if g == 0 else Fraction(1)
        else:
            gamma = Fraction(0) if g == 0 and j % 2 == 0 else Fraction(1)
    else:
        alpha, beta = _PLOTTING[kind]
        h = alpha + level * (n + 1 - alpha - beta)
        j = math.floor(h)
        gamma = h - j
    return (1 - gamma) * order(j) + gamma * order(j + 1)


class HyndmanFanDefinitionTests(unittest.TestCase):
    """All nine types against exact arithmetic and the paper's identities."""

    SAMPLES = (
        [7],
        [4, 1],
        [3, 1, 4, 1, 5],
        [5, 3, 9, 1, 1, 8, 2, 6],
        [12, 7, 3, 9, 1, 15, 4, 8, 10, 2, 6, 11, 5],
        list(range(1, 101)),
    )

    def test_matches_exact_rational_definitions_at_decimal_levels(self) -> None:
        """Every ``p = k/100`` agrees with the exact definition.

        Oracle: ``exact_quantile`` with ``p = Fraction(k, 100)``. Types 1-3
        return an order statistic or the average of two, so they must agree
        exactly; types 4-9 to ``1e-13`` relative.
        """
        for values in self.SAMPLES:
            data = [float(v) for v in values]
            for kind in range(1, 10):
                got = quantiles(data, [k / 100 for k in range(101)], method=kind)
                for k, value in enumerate(got):
                    expected = float(exact_quantile(values, Fraction(k, 100), kind))
                    with self.subTest(n=len(values), kind=kind, k=k):
                        if kind <= 3:
                            self.assertEqual(value, expected)
                        else:
                            self.assertLessEqual(
                                abs(value - expected), 1e-13 * max(1.0, abs(expected))
                            )

    def test_continuous_types_pass_through_their_plotting_positions(self) -> None:
        """``Q(p_k) = x_(k)`` exactly at ``p_k = (k - alpha)/(n + 1 - alpha - beta)``.

        Oracle: the defining property of types 4-9 in Hyndman and Fan.
        """
        values = [5.0, 3.0, 9.0, 1.5, 1.0, 8.0, 2.0, 6.0, 4.25, 7.0, 0.5]
        ordered = sorted(values)
        n = len(values)
        for kind, (alpha, beta) in _PLOTTING.items():
            for k in range(1, n + 1):
                level = (k - alpha) / (n + 1 - alpha - beta)
                if 0 <= level <= 1:
                    with self.subTest(kind=kind, k=k):
                        self.assertEqual(
                            quantile(values, float(level), method=kind), ordered[k - 1]
                        )

    def test_type_two_averages_the_type_one_jumps(self) -> None:
        """Type 2 is type 1 except at ``p = j/n``, where it averages two values.

        Oracle: Hyndman and Fan's description of type 2.
        """
        values = [4.0, 8.0, 15.0, 16.0, 23.0, 42.0]
        ordered = sorted(values)
        for k in range(0, 61):
            p = k / 60
            if k % 10 == 0 and 0 < k < 60:
                j = k // 10
                expected = 0.5 * (ordered[j - 1] + ordered[j])
            else:
                expected = quantile(values, p, method=1)
            with self.subTest(p=p):
                self.assertEqual(quantile(values, p, method=2), expected)

    def test_type_three_is_the_nearest_even_order_statistic(self) -> None:
        """Type 3 picks ``x_(k)``, ``k`` the integer nearest ``n p``, ties to even.

        Oracle: ``round`` (half to even) on the exact ``n Fraction(p)``,
        clamped to ``1 .. n``.
        """
        values = [float(v * v) for v in range(1, 12)]
        n = len(values)
        for k in range(101):
            index = min(max(round(n * Fraction(k, 100)), 1), n)
            with self.subTest(k=k):
                self.assertEqual(quantile(values, k / 100, method=3), values[index - 1])

    def test_median_property_and_symmetry(self) -> None:
        """Types 2 and 5-9 give the median and satisfy ``Q_x(p) = -Q_-x(1-p)``.

        Oracle: Hyndman and Fan's properties; dyadic levels ``k/64`` make
        ``1 - p`` exact.
        """
        for values in ([2.0, 7.0, 1.0, 8.0, 2.5], [2.0, 7.0, 1.0, 8.0, 2.5, 9.5]):
            negated = [-v for v in values]
            for kind in (2, 5, 6, 7, 8, 9):
                with self.subTest(n=len(values), kind=kind):
                    self.assertEqual(quantile(values, 0.5, method=kind), median(values))
                    for k in range(65):
                        p = k / 64
                        self.assertAlmostEqual(
                            quantile(values, p, method=kind),
                            -quantile(negated, 1.0 - p, method=kind),
                            places=13,
                        )

    def test_every_type_is_nondecreasing_in_p(self) -> None:
        """Each definition is a non-decreasing function of ``p``.

        Oracle: the monotonicity property P4 of Hyndman and Fan.
        """
        values = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0, 5.0]
        levels = [k / 400 for k in range(401)]
        for kind in range(1, 10):
            got = quantiles(values, levels, method=kind)
            with self.subTest(kind=kind):
                self.assertTrue(all(a <= b for a, b in zip(got, got[1:], strict=False)))
                self.assertEqual(got[0], 1.0)
                self.assertEqual(got[-1], 9.0)

    def test_agrees_with_r_quantile(self) -> None:
        """Agreement with R's ``quantile(x, p, type = k)`` for ``k = 1 .. 9``.

        Oracle: R 4.6.2 (2026-10-29 ucrt),
        ``quantile(c(3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5),
        c(0, 0.05, 0.1, 0.25, 0.3, 0.5, 0.65, 0.9, 1), type = k,
        names = FALSE)`` printed with ``%.17g``.
        """
        values = [3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0, 5.0]
        levels = (0.0, 0.05, 0.1, 0.25, 0.3, 0.5, 0.65, 0.9, 1.0)
        r_values = {
            1: (1, 1, 1, 2, 3, 4, 5, 6, 9),
            2: (1, 1, 1, 2, 3, 4, 5, 6, 9),
            3: (1, 1, 1, 2, 2, 4, 5, 6, 9),
            4: (1, 1, 1, 1.75, 2.2999999999999998, 3.5, 5, 5.9000000000000004, 9),
            5: (1, 1, 1, 2.25, 2.7999999999999998, 4, 5, 7.2000000000000011, 9),
            6: (1, 1, 1, 2, 2.5999999999999996, 4, 5, 8.4000000000000021, 9),
            7: (1, 1, 1, 2.5, 3, 4, 5, 6, 9),
            8: (1, 1, 1, 2.1666666666666665, 2.7333333333333329, 4, 5,
                7.5999999999999996, 9),
            9: (1, 1, 1, 2.1875, 2.75, 4, 5, 7.5, 9),
        }  # fmt: skip
        for kind, expected in r_values.items():
            got = quantiles(values, levels, method=kind)
            for value, reference in zip(got, expected, strict=True):
                with self.subTest(kind=kind):
                    self.assertLessEqual(abs(value - reference), 1e-14 * abs(reference))

    def test_integer_positions_follow_the_decimal_level(self) -> None:
        """``n = 100`` and ``p = 0.07`` give ``x_(7)`` for types 1 and 4, 7.5 for 2.

        Oracle: exact arithmetic, ``100 * Fraction(7, 100) = 7``. R 4.6.2's
        ``quantile(1:100, 0.07, type = 1)`` returns 8 because the product
        ``100 * 0.07`` rounds to ``7.000000000000001``; probstat follows the
        level as written.
        """
        values = [float(i) for i in range(1, 101)]
        self.assertEqual(quantile(values, 0.07, method=1), 7.0)
        self.assertEqual(quantile(values, 0.07, method=2), 7.5)
        self.assertEqual(quantile(values, 0.07, method=4), 7.0)
        self.assertAlmostEqual(quantile(values, 0.29, method=6), 29.29, places=12)

    def test_default_and_invalid_methods(self) -> None:
        """Type 7 stays the default; types outside 1-9 raise ``ValueError``.

        Oracle: ``Q_7(p) = x_(1+floor(h)) + (h - floor(h)) (...)`` with
        ``h = (n - 1) p``.
        """
        values = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        self.assertEqual(quantile(values, 0.9), quantile(values, 0.9, method=7))
        self.assertAlmostEqual(quantile(values, 0.9), 7.6, places=14)
        for method in (0, 10, 2.5, True, -1):
            with self.subTest(method=method), self.assertRaises(ValueError):
                quantile(values, 0.5, method=method)  # type: ignore[arg-type]

    def test_single_observation(self) -> None:
        """Every type returns the only observation of a sample of size one.

        Oracle: ``x_(0) = x_(1) = x_(2)`` by the end conventions.
        """
        for kind in range(1, 10):
            for p in (0.0, 0.3, 0.5, 1.0):
                self.assertEqual(quantile([2.5], p, method=kind), 2.5)


def k_statistics(values: list[int]) -> tuple[Fraction, Fraction, Fraction]:
    """Return Fisher's unbiased cumulant estimators ``k2, k3, k4`` exactly.

    Args:
        values: Integer sample of size at least four.

    Returns:
        ``(k2, k3, k4)`` from the power sums ``S_r = sum x^r``.

    References:
        [EQ-STAT-SHAPE-002], [REF-JOANES-GILL-1998].

    """
    n = len(values)
    s1, s2, s3, s4 = (sum(Fraction(v) ** r for v in values) for r in (1, 2, 3, 4))
    k2 = (n * s2 - s1**2) / (n * (n - 1))
    k3 = (n * n * s3 - 3 * n * s2 * s1 + 2 * s1**3) / (n * (n - 1) * (n - 2))
    k4 = (
        (n**3 + n * n) * s4
        - 4 * (n * n + n) * s3 * s1
        - 3 * (n * n - n) * s2**2
        + 12 * n * s2 * s1**2
        - 6 * s1**4
    ) / (n * (n - 1) * (n - 2) * (n - 3))
    return k2, k3, k4


class ShapeConventionTests(unittest.TestCase):
    """``g``, ``G``, and ``b`` skewness and kurtosis."""

    def test_type_two_is_the_k_statistic_ratio(self) -> None:
        """``G1 = k3 / k2^(3/2)`` and ``G2 = k4 / k2^2``.

        Oracle: Fisher's k-statistics from power sums in exact arithmetic
        (``G1`` compared through its square and sign).
        """
        for values in (
            [1, 2, 3, 10],
            [4, 7, 7, 1, 9, 12, 3],
            [2, 2, 3, 5, 8, 13, 21, 34],
        ):
            k2, k3, k4 = k_statistics(values)
            data = [float(v) for v in values]
            g = skewness(data, type=2)
            self.assertAlmostEqual(g * g / float(k3 * k3 / k2**3), 1.0, places=13)
            self.assertEqual(math.copysign(1.0, g), math.copysign(1.0, float(k3)))
            self.assertAlmostEqual(
                excess_kurtosis(data, type=2), float(k4 / k2**2), places=12
            )

    def test_exact_values_and_identities(self) -> None:
        """Exact rationals for ``(1, 2, 3, 10)`` and ``b = m / s^k`` identities.

        Oracle: central moments in ``Fraction``: ``m2 = 12.5``,
        ``m4 = 400.8125``; ``g2 = -0.7696``, ``G2 = 3.228``, ``b2 = -1.7454``,
        and ``b1 = m3 / s^3`` with ``s^2 = 50/3``.
        """
        values = [1, 2, 3, 10]
        n = len(values)
        mean = Fraction(sum(values), n)
        m2, m3, m4 = (sum((v - mean) ** r for v in values) / n for r in (2, 3, 4))
        g2 = m4 / m2**2 - 3
        data = [float(v) for v in values]
        self.assertEqual(g2, Fraction(-481, 625))
        self.assertAlmostEqual(excess_kurtosis(data), float(g2), places=15)
        self.assertAlmostEqual(excess_kurtosis(data, type=2), 3.228, places=14)
        self.assertAlmostEqual(excess_kurtosis(data, type=3), -1.7454, places=14)
        s2 = m2 * n / (n - 1)
        self.assertAlmostEqual(
            skewness(data, type=3), float(m3) / float(s2) ** 1.5, places=14
        )
        self.assertAlmostEqual(
            excess_kurtosis(data, type=3), float(m4 / s2**2 - 3), places=14
        )
        self.assertAlmostEqual(
            skewness(data, type=1), float(m3) / float(m2) ** 1.5, places=14
        )

    def test_agrees_with_e1071(self) -> None:
        """Agreement with ``e1071::skewness`` and ``e1071::kurtosis``.

        Oracle: R 4.6.2 with e1071 1.7.17 (``packageVersion("e1071")``),
        ``e1071::skewness(w, type = t)`` and ``e1071::kurtosis(w, type = t)``
        for ``t = 1, 2, 3``, printed with ``%.17g``.
        """
        w = [2.2, 4.1, 3.5, 4.5, 5.0, 3.7, 3.0, 2.6, 3.4, 1.6, 3.1, 3.3, 3.8, 3.1,
             4.7, 3.7, 2.5, 4.3, 4.9, 3.6, 2.9, 3.3, 3.9, 3.1, 4.8, 3.1, 3.7, 4.4,
             3.2, 4.1, 1.9, 3.4, 4.7, 3.8, 3.0, 2.6, 3.9, 3.0, 4.2, 3.5]  # fmt: skip
        r_values = {
            1: (-0.17764245555172031, -0.28410126718863848),
            2: (-0.18463986347639122, -0.15666993331054976),
            3: (-0.17102267354500736, -0.41819876712119974),
        }
        for kind, (skew, kurt) in r_values.items():
            with self.subTest(type=kind):
                self.assertLessEqual(
                    abs(skewness(w, type=kind) - skew), 1e-12 * abs(skew)
                )
                self.assertLessEqual(
                    abs(excess_kurtosis(w, type=kind) - kurt), 1e-12 * abs(kurt)
                )

    def test_normal_theory_bias(self) -> None:
        """Under normality ``E[G2] = 0`` while ``E[g2] = -6/(n + 1)``.

        Oracle: the exact moments of ``g2`` for normal samples and the
        unbiasedness of ``k4 / k2^2`` (independent of ``k2`` by Basu's
        theorem); 4000 seeded samples of size 10, judged at four Monte
        Carlo standard errors.
        """
        generator = random.Random(20261117)
        n, replicates = 10, 4000
        g2_values, big_g2_values = [], []
        for _ in range(replicates):
            sample = [generator.gauss(0.0, 1.0) for _ in range(n)]
            g2_values.append(excess_kurtosis(sample))
            big_g2_values.append(excess_kurtosis(sample, type=2))
        for values, target in ((g2_values, -6.0 / (n + 1)), (big_g2_values, 0.0)):
            error = standard_library.stdev(values) / math.sqrt(replicates)
            self.assertLess(abs(standard_library.fmean(values) - target), 4.0 * error)
        g2_error = standard_library.stdev(g2_values) / math.sqrt(replicates)
        self.assertGreater(abs(standard_library.fmean(g2_values)), 4.0 * g2_error)

    def test_default_is_unchanged_and_errors(self) -> None:
        """``type=1`` is the default; invalid types and small samples raise.

        Oracle: the definitions' domains (``n >= 3`` for ``G1``, ``n >= 4``
        for ``G2``).
        """
        data = [1.0, 2.0, 4.0, 8.0, 9.5]
        self.assertEqual(skewness(data), skewness(data, type=1))
        self.assertEqual(excess_kurtosis(data), excess_kurtosis(data, type=1))
        for kind in (0, 4, True, 1.5):
            with self.assertRaises(ValueError):
                skewness(data, type=kind)  # type: ignore[arg-type]
            with self.assertRaises(ValueError):
                excess_kurtosis(data, type=kind)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            skewness([1.0, 2.0], type=2)
        with self.assertRaises(ValueError):
            excess_kurtosis([1.0, 2.0, 4.0], type=2)
        self.assertAlmostEqual(skewness([1.0, 2.0], type=3), 0.0)


if __name__ == "__main__":
    unittest.main()
