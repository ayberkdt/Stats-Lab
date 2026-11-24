"""Cramer-von Mises null distribution and goodness-of-fit test (fully specified CDF).

Oracles:

- An independent 60-digit evaluation of the Anderson-Darling (1952) series
  with mpmath 1.3.0: ``V(x) = (pi^(3/2) sqrt(x))^-1 sum_(j>=0)
  Gamma(j + 1/2)/j! sqrt(4j+1) exp(-q_j) K_(1/4)(q_j)``, ``q_j =
  (4j+1)^2/(16 x)``, ``mp.mp.dps = 60`` and ``mp.besselk``; the pins below
  are ``V(x)`` and ``1 - V(x)``. The 95 and 99 percent points were found
  with ``mp.findroot(lambda x: V(x) - p, x0)`` and agree with the published
  rounded values 0.46136 and 0.74346.
- The moments of the limit law ``sum_k Z_k^2/(k pi)^2``: mean
  ``sum 1/(k pi)^2 = 1/6`` and second moment ``1/36 + 1/45 = 1/20``,
  integrated numerically from the tail.
- The exact ``n = 1`` law ``2 sqrt(x - 1/12)`` on ``[1/12, 1/3]``.
- SciPy 1.14.1 ``scipy.stats.cramervonmises(x, cdf)`` (the same
  Csorgo-Faraway approximation, accurate there to about ``1e-7``), used only
  to produce the pinned values below; the expressions are written next to
  each pin and nothing here imports SciPy.
- Seeded simulations of ``W_n^2`` for uniform samples, judged by binomial
  standard errors plus the stated accuracy of the approximation, and of the
  p-value of :func:`cramer_von_mises_test` under a true null.

References:
    [REF-ANDERSON-DARLING-1952], [REF-CSORGO-FARAWAY-1996],
    [REF-STEPHENS-1974], [EQ-INF-GOF-CVM-001], [EQ-INF-GOF-CVM-002],
    [EQ-INF-GOF-CVM-003], [EQ-INF-GOF-CVM-004], [EQ-INF-GOF-CVM-005].

"""

from __future__ import annotations

import math
import random
import unittest

from probstat.distributions import ChiSquare, Normal, Uniform
from probstat.inference import (
    CramerVonMisesResult,
    cramer_von_mises_asymptotic_cdf,
    cramer_von_mises_asymptotic_sf,
    cramer_von_mises_cdf,
    cramer_von_mises_sf,
    cramer_von_mises_statistic,
    cramer_von_mises_test,
)
from probstat.inference.cramer_von_mises import _lower_series, _upper_tail

GOLDEN = 0.6180339887498949


def _golden(size: int) -> list[float]:
    """Return ``(i * 0.6180339887498949) mod 1`` for ``i = 1..size``.

    References:
        [EQ-INF-GOF-CVM-001]. No equation is evaluated.

    """
    return [(index * GOLDEN) % 1.0 for index in range(1, size + 1)]


def _uniform_statistics(size: int, count: int, seed: int) -> list[float]:
    """Simulate ``W_n^2`` for ``count`` uniform samples of ``size`` values.

    References:
        [EQ-INF-GOF-CVM-001].

    """
    rng = random.Random(seed)
    midpoints = [(2 * rank - 1) / (2.0 * size) for rank in range(1, size + 1)]
    values: list[float] = []
    for _ in range(count):
        draws = sorted(rng.random() for _ in range(size))
        values.append(
            1.0 / (12.0 * size)
            + math.fsum((u - m) ** 2 for u, m in zip(draws, midpoints, strict=True))
        )
    return values


class AsymptoticLawTests(unittest.TestCase):
    """The limit law of the statistic.

    References:
        [REF-ANDERSON-DARLING-1952]. Checks [EQ-INF-GOF-CVM-002] and
        [EQ-INF-GOF-CVM-003].

    """

    def test_matches_the_sixty_digit_series(self) -> None:
        """Both the CDF and the tail agree with the mpmath series.

        References:
            [REF-ANDERSON-DARLING-1952]. Checks [EQ-INF-GOF-CVM-002] and
            [EQ-INF-GOF-CVM-003].

        """
        # (x, V(x), 1 - V(x)) from the mpmath expression in the module docstring.
        pins = (
            (0.05, 0.123719068958651, 0.876280931041349),
            (0.1, 0.4151265615932051, 0.5848734384067949),
            (0.3, 0.8648287311948942, 0.1351712688051058),
            (0.46136, 0.9499996168673487, 0.05000038313265128),
            (0.74346, 0.9900000380845446, 0.009999961915455367),
            (1.0, 0.997539547819866, 0.002460452180133964),
            (2.0, 0.9999872192638272, 1.278073617278167e-5),
            (3.0, 0.9999999243225654, 7.56774345931602e-8),
            (6.0, 0.9999999999999799, 2.008774178694868e-14),
            (12.0, 1.0, 1.975121164172937e-27),
        )
        for x, cdf, sf in pins:
            with self.subTest(x=x):
                self.assertAlmostEqual(
                    cramer_von_mises_asymptotic_cdf(x), cdf, delta=2e-15
                )
                self.assertAlmostEqual(
                    cramer_von_mises_asymptotic_sf(x), sf, delta=max(5e-15 * sf, 1e-17)
                )

    def test_percentage_points(self) -> None:
        """The 95 and 99 percent points are 0.46136 and 0.74346.

        References:
            [REF-ANDERSON-DARLING-1952]. Checks [EQ-INF-GOF-CVM-002].

        """
        # mp.findroot(lambda x: V(x) - p, x0) at 60 digits.
        self.assertAlmostEqual(
            cramer_von_mises_asymptotic_cdf(0.461361293606), 0.95, delta=1e-11
        )
        self.assertAlmostEqual(
            cramer_von_mises_asymptotic_cdf(0.743459313756), 0.99, delta=1e-11
        )
        self.assertEqual(round(0.461361293606, 5), 0.46136)
        self.assertEqual(round(0.743459313756, 5), 0.74346)

    def test_the_two_series_agree_where_they_overlap(self) -> None:
        """The Bessel series and the line integrals give the same law.

        References:
            [REF-ANDERSON-DARLING-1952]. Checks [EQ-INF-GOF-CVM-002] and
            [EQ-INF-GOF-CVM-003].

        """
        for x in (0.25, 0.4, 0.6, 0.9, 1.5):
            with self.subTest(x=x):
                self.assertAlmostEqual(
                    _lower_series(x), 1.0 - _upper_tail(x), delta=2e-15
                )

    def test_moments_are_those_of_the_series_of_squares(self) -> None:
        """The tail integrates to the mean ``1/6`` and the second moment ``1/20``.

        References:
            [REF-ANDERSON-DARLING-1952]. Checks [EQ-INF-GOF-CVM-002] and
            [EQ-INF-GOF-CVM-003].

        """
        mean = 0.0
        second = 0.0
        for low, high, panels in ((0.0, 1.0, 2000), (1.0, 12.0, 1600)):
            step = (high - low) / panels
            for index in range(panels + 1):
                x = low + index * step
                weight = 1.0 if index in (0, panels) else (4.0 if index % 2 else 2.0)
                tail = cramer_von_mises_asymptotic_sf(x)
                mean += weight * tail * step / 3.0
                second += weight * 2.0 * x * tail * step / 3.0
        self.assertAlmostEqual(mean, 1.0 / 6.0, delta=1e-10)
        self.assertAlmostEqual(second, 1.0 / 20.0, delta=1e-10)

    def test_limits_and_monotonicity(self) -> None:
        """Boundary values, infinity, and a non-decreasing CDF.

        References:
            [REF-ANDERSON-DARLING-1952]. Checks [EQ-INF-GOF-CVM-002].

        """
        self.assertEqual(cramer_von_mises_asymptotic_cdf(0.0), 0.0)
        self.assertEqual(cramer_von_mises_asymptotic_cdf(-1.0), 0.0)
        self.assertEqual(cramer_von_mises_asymptotic_cdf(math.inf), 1.0)
        self.assertEqual(cramer_von_mises_asymptotic_sf(0.0), 1.0)
        self.assertEqual(cramer_von_mises_asymptotic_sf(math.inf), 0.0)
        self.assertEqual(cramer_von_mises_asymptotic_cdf(1e-300), 0.0)
        self.assertEqual(cramer_von_mises_asymptotic_sf(1e4), 0.0)
        grid = [0.01 * index for index in range(1, 300)]
        values = [cramer_von_mises_asymptotic_cdf(x) for x in grid]
        self.assertTrue(all(a <= b for a, b in zip(values, values[1:], strict=False)))
        for x in (0.2, 0.7, 2.5):
            total = cramer_von_mises_asymptotic_cdf(x) + cramer_von_mises_asymptotic_sf(
                x
            )
            self.assertAlmostEqual(total, 1.0, delta=1e-15)

    def test_validation(self) -> None:
        """Invalid arguments raise ``TypeError`` or ``ValueError``.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION]. Checks
            [EQ-INF-GOF-CVM-002].

        """
        for function in (
            cramer_von_mises_asymptotic_cdf,
            cramer_von_mises_asymptotic_sf,
        ):
            with self.assertRaises(ValueError):
                function(math.nan)
            with self.assertRaises(TypeError):
                function("0.5")  # type: ignore[arg-type]
            with self.assertRaises(TypeError):
                function(True)  # type: ignore[arg-type]


class FiniteSampleLawTests(unittest.TestCase):
    """The finite-``n`` null law.

    References:
        [REF-CSORGO-FARAWAY-1996]. Checks [EQ-INF-GOF-CVM-004] and
        [EQ-INF-GOF-CVM-005].

    """

    def test_one_observation_is_exact(self) -> None:
        """``P(W_1^2 <= x) = 2 sqrt(x - 1/12)`` on ``[1/12, 1/3]``.

        References:
            [EQ-INF-GOF-CVM-005].

        """
        self.assertEqual(cramer_von_mises_cdf(1.0 / 12.0, 1), 0.0)
        self.assertEqual(cramer_von_mises_cdf(1.0 / 3.0, 1), 1.0)
        self.assertAlmostEqual(
            cramer_von_mises_cdf(0.2, 1), 2.0 * math.sqrt(0.2 - 1.0 / 12.0), places=15
        )
        self.assertAlmostEqual(
            cramer_von_mises_sf(0.1, 1),
            1.0 - 2.0 * math.sqrt(0.1 - 1.0 / 12.0),
            places=15,
        )
        # W_1^2 = 1/12 + (u - 1/2)^2 for the single value u; the median of
        # (u - 1/2)^2 is 1/16, so the median of W_1^2 is 1/12 + 1/16.
        self.assertAlmostEqual(
            cramer_von_mises_cdf(1.0 / 12.0 + 1.0 / 16.0, 1), 0.5, places=15
        )

    def test_matches_pinned_scipy_values(self) -> None:
        """The p-values agree with the Csorgo-Faraway implementation in SciPy.

        References:
            [REF-CSORGO-FARAWAY-1996]. Checks [EQ-INF-GOF-CVM-004].

        """
        # u = [(i * 0.6180339887498949) % 1.0 for i in range(1, n + 1)]
        # r = scipy.stats.cramervonmises(u ** 1.3, 'uniform')   (r.statistic, r.pvalue)
        pins = (
            (2, 0.09730309943004944, 0.6450497050282462),
            (3, 0.02953349819487181, 0.994976891623511),
            (5, 0.09107452640386693, 0.6501809104746699),
            (10, 0.07182483751157998, 0.7515773585097456),
            (50, 0.25226857366165784, 0.18556558338769114),
            (200, 0.9419157523430962, 0.003322384655834454),
        )
        for size, statistic, p_value in pins:
            sample = [u**1.3 for u in _golden(size)]
            with self.subTest(size=size):
                self.assertAlmostEqual(
                    cramer_von_mises_statistic(sample, Uniform(0.0, 1.0)),
                    statistic,
                    places=14,
                )
                self.assertAlmostEqual(
                    cramer_von_mises_sf(statistic, size), p_value, delta=5e-7
                )
                self.assertAlmostEqual(
                    cramer_von_mises_cdf(statistic, size), 1.0 - p_value, delta=5e-7
                )

    def test_converges_to_the_limit_law(self) -> None:
        """The finite-sample tail approaches the asymptotic tail as ``n`` grows.

        References:
            [REF-CSORGO-FARAWAY-1996]. Checks [EQ-INF-GOF-CVM-004].

        """
        for x in (0.2, 0.46, 0.74, 1.2):
            limit = cramer_von_mises_asymptotic_sf(x)
            gaps = [abs(cramer_von_mises_sf(x, n) - limit) for n in (10, 100, 1000)]
            with self.subTest(x=x):
                # The correction is psi_1(x)/n: tenfold n, tenfold smaller gap.
                self.assertAlmostEqual(gaps[1] / gaps[2], 10.0, delta=0.3)
                self.assertAlmostEqual(gaps[0] / gaps[1], 10.0, delta=1.5)

    def test_support_and_range(self) -> None:
        """The tail is one at the lower end of the support and zero at the upper.

        References:
            [REF-CSORGO-FARAWAY-1996]. Checks [EQ-INF-GOF-CVM-004].

        """
        for size in (2, 5, 40):
            low = 1.0 / (12.0 * size)
            high = size / 3.0
            self.assertEqual(cramer_von_mises_sf(low, size), 1.0)
            self.assertEqual(cramer_von_mises_sf(0.0, size), 1.0)
            self.assertEqual(cramer_von_mises_sf(high, size), 0.0)
            self.assertEqual(cramer_von_mises_sf(math.inf, size), 0.0)
            for x in (low * 1.01, 0.1, 0.5, 2.0, high * 0.99):
                value = cramer_von_mises_sf(x, size)
                self.assertTrue(0.0 <= value <= 1.0)
        self.assertEqual(cramer_von_mises_sf(100.0, 10_000), 0.0)

    def test_accuracy_against_simulation(self) -> None:
        """Upper-tail probabilities match seeded simulations within the stated accuracy.

        The stated accuracy is 1e-2 at n = 2, 4e-3 at n = 3 and 1e-3 from
        n = 5 on; the simulation error is added as five binomial standard
        errors.

        References:
            [REF-CSORGO-FARAWAY-1996]. Checks [EQ-INF-GOF-CVM-004].

        """
        replications = 40_000
        for size, stated in ((2, 1e-2), (3, 4e-3), (5, 1e-3), (12, 1e-3), (40, 1e-3)):
            statistics = sorted(_uniform_statistics(size, replications, 1000 + size))
            for level in (0.5, 0.2, 0.1, 0.05, 0.01):
                cut = statistics[int((1.0 - level) * replications)]
                empirical = sum(1 for s in statistics if s > cut) / replications
                error = 5.0 * math.sqrt(level * (1.0 - level) / replications)
                with self.subTest(size=size, level=level):
                    self.assertAlmostEqual(
                        cramer_von_mises_sf(cut, size), empirical, delta=stated + error
                    )

    def test_validation(self) -> None:
        """Invalid arguments raise ``TypeError`` or ``ValueError``.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION]. Checks
            [EQ-INF-GOF-CVM-004].

        """
        for function in (cramer_von_mises_cdf, cramer_von_mises_sf):
            with self.assertRaises(ValueError):
                function(math.nan, 5)
            with self.assertRaises(ValueError):
                function(0.3, 0)
            with self.assertRaises(ValueError):
                function(0.3, -2)
            with self.assertRaises(TypeError):
                function(0.3, 2.5)  # type: ignore[arg-type]
            with self.assertRaises(TypeError):
                function("0.3", 5)  # type: ignore[arg-type]


class CramerVonMisesTestTests(unittest.TestCase):
    """The test function and its record.

    References:
        [REF-STEPHENS-1974], [REF-CSORGO-FARAWAY-1996]. Checks
        [EQ-INF-GOF-CVM-001] and [EQ-INF-GOF-CVM-004].

    """

    def test_matches_pinned_scipy_values_for_other_laws(self) -> None:
        """A normal and a chi-square null reproduce the pinned statistic and p-value.

        References:
            [EQ-INF-GOF-CVM-001], [EQ-INF-GOF-CVM-004].

        """
        # u = np.array([(i * 0.6180339887498949) % 1.0 for i in range(1, 13)])
        # x = 2.0 * (u - 0.5)
        # scipy.stats.cramervonmises(x, 'norm')
        sample = [2.0 * (u - 0.5) for u in _golden(12)]
        result = cramer_von_mises_test(sample, Normal(0.0, 1.0))
        self.assertAlmostEqual(result.statistic, 0.11184312384200563, places=14)
        self.assertAlmostEqual(result.p_value, 0.5366278577620112, delta=5e-7)
        # u = np.array([(i * 0.6180339887498949) % 1.0 for i in range(1, 41)])
        # x = 6.0 * u ** 2
        # scipy.stats.cramervonmises(x, scipy.stats.chi2(3).cdf)
        sample = [6.0 * u**2 for u in _golden(40)]
        result = cramer_von_mises_test(sample, ChiSquare(3.0))
        self.assertAlmostEqual(result.statistic, 0.8701239724259993, places=13)
        self.assertAlmostEqual(result.p_value, 0.004694073728716108, delta=5e-7)
        self.assertEqual(result.sample_size, 40)
        self.assertIn("Csorgo-Faraway", result.method)

    def test_statistic_equals_the_existing_statistic_function(self) -> None:
        """The record carries the statistic of ``cramer_von_mises_statistic``.

        References:
            [EQ-INF-GOF-CVM-001].

        """
        sample = [0.7, -0.2, 1.9, 0.3, -1.1, 0.05]
        law = Normal(0.0, 1.0)
        self.assertEqual(
            cramer_von_mises_test(sample, law).statistic,
            cramer_von_mises_statistic(sample, law),
        )
        self.assertEqual(
            cramer_von_mises_test(iter(sample), law).p_value,
            cramer_von_mises_test(sorted(sample), law).p_value,
        )

    def test_single_observation_uses_the_exact_law(self) -> None:
        """One observation gives the exact tail and the exact-law label.

        References:
            [EQ-INF-GOF-CVM-005].

        """
        result = cramer_von_mises_test([0.9], Uniform(0.0, 1.0))
        statistic = 1.0 / 12.0 + 0.4**2
        self.assertAlmostEqual(result.statistic, statistic, places=15)
        self.assertAlmostEqual(result.p_value, 1.0 - 2.0 * 0.4, places=15)
        self.assertIn("exact", result.method)

    def test_values_outside_the_support_give_the_maximum_statistic(self) -> None:
        """Observations outside the null support give ``W^2 = n/3`` and ``p = 0``.

        References:
            [EQ-INF-GOF-CVM-001], [EQ-INF-GOF-CVM-004].

        """
        for sample in ([-1.0, -2.0, -3.0, -4.0], [5.0, 6.0, 7.0, 8.0]):
            result = cramer_von_mises_test(sample, Uniform(0.0, 1.0))
            self.assertAlmostEqual(result.statistic, 4.0 / 3.0, places=14)
            self.assertEqual(result.p_value, 0.0)

    def test_p_values_are_uniform_under_the_null(self) -> None:
        """Under a true null the p-value is uniform, judged on binomial errors.

        References:
            [REF-CSORGO-FARAWAY-1996]. Checks [EQ-INF-GOF-CVM-004].

        """
        replications = 3_000
        rng = random.Random(20261125)
        law = Normal(0.0, 1.0)
        p_values = [
            cramer_von_mises_test([rng.gauss(0.0, 1.0) for _ in range(10)], law).p_value
            for _ in range(replications)
        ]
        for level in (0.5, 0.25, 0.1, 0.05):
            fraction = sum(1 for p in p_values if p <= level) / replications
            error = 5.0 * math.sqrt(level * (1.0 - level) / replications) + 1e-3
            with self.subTest(level=level):
                self.assertAlmostEqual(fraction, level, delta=error)
        mean = math.fsum(p_values) / replications
        self.assertAlmostEqual(
            mean, 0.5, delta=5.0 * math.sqrt(1.0 / 12.0 / replications) + 1e-3
        )

    def test_wrong_null_is_rejected(self) -> None:
        """A shifted and a scaled sample are rejected with small p-values.

        References:
            [EQ-INF-GOF-CVM-004].

        """
        rng = random.Random(7)
        shifted = [rng.gauss(1.0, 1.0) for _ in range(100)]
        scaled = [rng.gauss(0.0, 2.0) for _ in range(100)]
        right = [rng.gauss(0.0, 1.0) for _ in range(100)]
        law = Normal(0.0, 1.0)
        self.assertLess(cramer_von_mises_test(shifted, law).p_value, 1e-6)
        self.assertLess(cramer_von_mises_test(scaled, law).p_value, 1e-3)
        self.assertGreater(cramer_von_mises_test(right, law).p_value, 0.01)

    def test_validation(self) -> None:
        """Invalid inputs raise ``TypeError`` or ``ValueError``.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION]. Checks
            [EQ-INF-GOF-CVM-001].

        """
        law = Normal(0.0, 1.0)
        with self.assertRaises(ValueError):
            cramer_von_mises_test([], law)
        with self.assertRaises(ValueError):
            cramer_von_mises_test([1.0, math.nan], law)
        with self.assertRaises(TypeError):
            cramer_von_mises_test([1.0, 2.0], object())  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            cramer_von_mises_test([1.0, "a"], law)

    def test_result_record_validation(self) -> None:
        """The record rejects out-of-domain fields and is immutable.

        References:
            [EQ-INF-GOF-CVM-001].

        """
        ok = CramerVonMisesResult(0.2, 3, 0.4)
        self.assertEqual(ok.sample_size, 3)
        with self.assertRaises(AttributeError):
            ok.statistic = 1.0  # type: ignore[misc]
        with self.assertRaises(ValueError):
            CramerVonMisesResult(-0.1, 3, 0.4)
        with self.assertRaises(ValueError):
            CramerVonMisesResult(2.0, 3, 0.4)
        with self.assertRaises(ValueError):
            CramerVonMisesResult(0.2, 0, 0.4)
        with self.assertRaises(ValueError):
            CramerVonMisesResult(0.2, 3, 1.4)
        with self.assertRaises(ValueError):
            CramerVonMisesResult(0.2, 3, 0.4, " ")
        with self.assertRaises(TypeError):
            CramerVonMisesResult("a", 3, 0.4)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            CramerVonMisesResult(0.2, 3, 0.4, 5)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
