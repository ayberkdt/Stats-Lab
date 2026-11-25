"""Normal tolerance factors, margins, contingency, and performance tracking.

Oracles: base R 4.6.2 (``qt`` with a noncentrality for the one-sided factor,
``qchisq`` for Howe's factor, ``binom.test`` for the Clopper-Pearson
interval, ``integrate`` and ``uniroot`` for an independent order of
integration of the two-sided equation); a brute-force Simpson quadrature
written in this file; closed-form arithmetic; and seeded Monte Carlo
coverage judged on its binomial standard error. Each R pin carries the
expression that produced it. The tests never call R.

References:
    [REF-HOWE-1969], [REF-KRISHNAMOORTHY-MATHEW-2009], [REF-YOUNG-2016],
    [REF-NIST-SEMATECH-HANDBOOK], [REF-HELTON-2011], [REF-NASA-SEH-2016],
    [REF-CLOPPER-PEARSON-1934], [REF-WILKS-1941].
    Checks [EQ-PROP-TOL-001], [EQ-PROP-TOL-002], [EQ-PROP-TOL-003],
    [EQ-PROP-TOL-004], [EQ-PROP-MARGIN-001], [EQ-PROP-MARGIN-002],
    [EQ-PROP-MARGIN-003].

"""

from __future__ import annotations

import math
import random
import unittest
from statistics import NormalDist

from probstat.propagation import (
    Milestone,
    ProbabilityBox,
    Requirement,
    TechnicalPerformanceMeasure,
    allocation_margin_fraction,
    apply_contingency,
    compliance_probability,
    confidence_of_contingency,
    contingency_for_confidence,
    convert_margin_fraction,
    normal_tolerance_bound,
    normal_tolerance_factor,
    normal_tolerance_interval,
    normal_tolerance_sample_size,
    normality_price,
    quantify_margin,
    quantify_margin_mixed,
    wilks_bound,
)

STANDARD = NormalDist()


def simpson(function, low: float, high: float, intervals: int = 4000) -> float:  # type: ignore[no-untyped-def]
    """Integrate ``function`` over ``[low, high]`` by composite Simpson.

    References:
        [REF-DAVIS-RABINOWITZ-1984, Chapter 2]. No new equation.

    """
    step = (high - low) / intervals
    total = function(low) + function(high)
    for index in range(1, intervals):
        total += (4 if index % 2 else 2) * function(low + index * step)
    return total * step / 3.0


def brute_force_one_sided_confidence(n: int, p: float, k: float) -> float:
    """Return ``P(xbar + k s >= mu + z_p sigma)`` by integrating over ``s``.

    Conditional on ``s = S / sigma`` the event is
    ``Z0 >= sqrt(n) (z_p - k s)`` for the standardized sample mean ``Z0``.
    This integrates over the chi density of ``s`` with the survival
    function of ``Z0``, not the noncentral ``t`` form of the library.

    References:
        [EQ-PROP-TOL-001], [EQ-PROP-TOL-002].

    """
    nu = n - 1
    z = STANDARD.inv_cdf(p)
    log_norm = math.log(2.0) + 0.5 * nu * math.log(0.5 * nu) - math.lgamma(0.5 * nu)

    def density(s: float) -> float:
        """Return the chi density of ``S / sigma``.

        References:
            [EQ-PROP-TOL-002]. No new equation.

        """
        if s <= 0.0:
            return 0.0
        return math.exp(log_norm + (nu - 1) * math.log(s) - 0.5 * nu * s * s)

    def integrand(s: float) -> float:
        """Return ``P(Z0 >= sqrt(n) (z - k s)) f(s)``.

        References:
            [EQ-PROP-TOL-001]. No new equation.

        """
        return (1.0 - STANDARD.cdf(math.sqrt(n) * (z - k * s))) * density(s)

    spread = 12.0 / math.sqrt(2.0 * nu)
    return simpson(integrand, max(1e-9, 1.0 - 4 * spread), 1.0 + 6 * spread)


def brute_force_two_sided_confidence(n: int, p: float, k: float) -> float:
    """Return the coverage probability of ``xbar +- k s`` by integrating over ``s``.

    Conditional on ``s`` the interval holds content ``p`` when the sample-mean
    deviation ``a = Z0 / sqrt(n)`` satisfies ``Phi(a + k s) - Phi(a - k s) >=
    p``, i.e. ``|a| <= a*(k s)``; the radius ``a*`` is found by bisection and
    ``P(|Z0| <= sqrt(n) a*) = 2 Phi(sqrt(n) a*) - 1``. This reverses the
    order of integration of the library (which fixes ``Z0`` first).

    References:
        [EQ-PROP-TOL-003].

    """
    nu = n - 1
    log_norm = math.log(2.0) + 0.5 * nu * math.log(0.5 * nu) - math.lgamma(0.5 * nu)

    def inner(w: float) -> float:
        """Return ``P(content >= p)`` given ``k s = w``.

        References:
            [EQ-PROP-TOL-003]. No new equation.

        """
        if 2.0 * STANDARD.cdf(w) - 1.0 <= p:
            return 0.0
        low, high = 0.0, w + 12.0
        for _ in range(80):
            middle = 0.5 * (low + high)
            if STANDARD.cdf(middle + w) - STANDARD.cdf(middle - w) >= p:
                low = middle
            else:
                high = middle
        return 2.0 * STANDARD.cdf(math.sqrt(n) * low) - 1.0

    def integrand(s: float) -> float:
        """Return ``inner(k s) f(s)``.

        References:
            [EQ-PROP-TOL-003]. No new equation.

        """
        density = math.exp(log_norm + (nu - 1) * math.log(s) - 0.5 * nu * s * s)
        return inner(k * s) * density

    spread = 12.0 / math.sqrt(2.0 * nu)
    kink = STANDARD.inv_cdf(0.5 * (1.0 + p)) / k
    high = 1.0 + 6 * spread
    if kink >= high:
        return 0.0
    # The content is below p for s < kink; beyond it a* grows like a square
    # root, so the substitution s = kink + t^2 makes the integrand smooth.
    return simpson(
        lambda t: integrand(kink + t * t) * 2.0 * t,
        0.0,
        math.sqrt(high - kink),
        4000,
    )


class OneSidedFactorTests(unittest.TestCase):
    """The exact one-sided factor against R and brute-force quadrature.

    References:
        [REF-KRISHNAMOORTHY-MATHEW-2009], [REF-NIST-SEMATECH-HANDBOOK].
        Checks [EQ-PROP-TOL-001] and [EQ-PROP-TOL-002].

    """

    def test_matches_r_noncentral_t_quantile(self) -> None:
        """Nine (n, p, confidence) triples agree with ``qt(..., ncp)``.

        References:
            [EQ-PROP-TOL-001].

        """
        # R: qt(g, n - 1, ncp = qnorm(p) * sqrt(n)) / sqrt(n) for each triple
        pins = (
            (2, 0.90, 0.95, 20.581467624301),
            (3, 0.95, 0.95, 7.655900133167),
            (5, 0.99, 0.95, 5.741084517245),
            (10, 0.90, 0.90, 2.065668320003),
            (10, 0.95, 0.99, 3.738314964193),
            (10, 0.95, 0.95, 2.910963413081),
            (30, 0.999, 0.95, 4.022198371288),
            (100, 0.99, 0.95, 2.683957855693),
            (100, 0.50, 0.90, 0.129016144203),
        )
        for n, p, gamma, expected in pins:
            with self.subTest(n=n, p=p, gamma=gamma):
                self.assertAlmostEqual(
                    normal_tolerance_factor(n, p, gamma), expected, delta=1e-9
                )

    def test_published_tables_and_examples(self) -> None:
        """One-sided factors against two open publications' printed values.

        Young (2016), Section "Normal tolerance interval procedures", prints
        the one-sided factors ``K.table(n = c(10, 20), alpha = c(0.01, 0.05),
        P = c(0.95, 0.99), side = 1)`` to six decimals, and the NIST/SEMATECH
        handbook (Section 7.2.6.3) works the noncentral-t factor for
        ``N = 43, p = 0.90`` at confidence 0.99 (1.8740, from the quantile
        12.28834) and ``N = 6`` (4.4111).

        References:
            [REF-YOUNG-2016, Section "Normal tolerance interval procedures"],
            [REF-NIST-SEMATECH-HANDBOOK]. Checks [EQ-PROP-TOL-001].

        """
        young = (
            (0.95, 0.99, 10, 3.738315),
            (0.95, 0.99, 20, 2.807866),
            (0.95, 0.95, 10, 2.910963),
            (0.95, 0.95, 20, 2.396002),
            (0.99, 0.99, 10, 5.073725),
            (0.99, 0.99, 20, 3.831558),
            (0.99, 0.95, 10, 3.981118),
            (0.99, 0.95, 20, 3.295157),
        )
        for p, gamma, n, printed in young:
            with self.subTest(n=n, p=p, gamma=gamma):
                self.assertAlmostEqual(
                    normal_tolerance_factor(n, p, gamma), printed, delta=6e-7
                )
        self.assertAlmostEqual(
            normal_tolerance_factor(43, 0.90, 0.99), 1.8740, delta=5e-5
        )
        self.assertAlmostEqual(
            normal_tolerance_factor(6, 0.90, 0.99), 4.4111, delta=5e-5
        )
        # t' / sqrt(n) with the printed noncentral-t quantile of the handbook.
        self.assertAlmostEqual(
            normal_tolerance_factor(43, 0.90, 0.99),
            12.28834 / math.sqrt(43),
            delta=2e-5,
        )

    def test_brute_force_integral_confirms_large_n(self) -> None:
        """At ``n = 1000`` (where R's ``qt`` warns) the factor solves its integral.

        References:
            [EQ-PROP-TOL-002].

        """
        for n, p, gamma in ((1000, 0.95, 0.95), (50, 0.9, 0.9), (4, 0.99, 0.95)):
            with self.subTest(n=n):
                k = normal_tolerance_factor(n, p, gamma)
                confidence = brute_force_one_sided_confidence(n, p, k)
                self.assertAlmostEqual(confidence, gamma, delta=2e-7)

    def test_factor_decreases_to_the_normal_quantile(self) -> None:
        """The factor decreases with ``n`` and stays above ``z_p``.

        References:
            [EQ-PROP-TOL-001].

        """
        factors = [normal_tolerance_factor(n, 0.95, 0.95) for n in (5, 10, 20, 40, 400)]
        self.assertEqual(factors, sorted(factors, reverse=True))
        self.assertGreater(factors[-1], STANDARD.inv_cdf(0.95))
        self.assertLess(factors[-1], STANDARD.inv_cdf(0.95) + 0.15)

    def test_seeded_coverage_matches_confidence(self) -> None:
        """The upper bound exceeds the true quantile with frequency ``gamma``.

        The frequency over 4000 seeded normal samples of size 8 is compared
        with the confidence on the scale of its binomial standard error.

        References:
            [EQ-PROP-TOL-001].

        """
        n, p, gamma, reps = 8, 0.9, 0.9, 4000
        k = normal_tolerance_factor(n, p, gamma)
        rng = random.Random(20261124)
        z = STANDARD.inv_cdf(p)
        hits = 0
        for _ in range(reps):
            sample = [rng.gauss(0.0, 1.0) for _ in range(n)]
            centre = sum(sample) / n
            spread = math.sqrt(sum((x - centre) ** 2 for x in sample) / (n - 1))
            hits += centre + k * spread >= z
        standard_error = math.sqrt(gamma * (1 - gamma) / reps)
        self.assertLess(abs(hits / reps - gamma), 4.0 * standard_error)


class TwoSidedFactorTests(unittest.TestCase):
    """The exact two-sided factor and Howe's approximation.

    References:
        [REF-HOWE-1969], [REF-KRISHNAMOORTHY-MATHEW-2009].
        Checks [EQ-PROP-TOL-003] and [EQ-PROP-TOL-004].

    """

    def test_matches_independent_r_integration(self) -> None:
        """Five triples agree with R's reversed-order integral and ``uniroot``.

        References:
            [EQ-PROP-TOL-003].

        """
        # R: uniroot(function(k) conf(k, n, p) - g, ...) where conf integrates
        # the chi density of s against 2 * pnorm(sqrt(n) a*(k s)) - 1 and a*
        # solves pnorm(a + w) - pnorm(a - w) = p.
        pins = (
            (3, 0.95, 0.95, 9.7887524030),
            (5, 0.90, 0.95, 4.2906040707),
            (10, 0.95, 0.95, 3.3934294787),
            (20, 0.99, 0.90, 3.3715193935),
            (50, 0.95, 0.99, 2.5804014338),
        )
        for n, p, gamma, expected in pins:
            with self.subTest(n=n, p=p, gamma=gamma):
                value = normal_tolerance_factor(n, p, gamma, sided="two-sided")
                self.assertAlmostEqual(value, expected, delta=1e-8)

    def test_published_exact_and_howe_examples(self) -> None:
        """The exact interval of a printed example and the handbook's Howe value.

        Young (2016) applies ``normtol.int(milk, alpha = 0.05, P = 0.90, side
        = 2, method = "EXACT")`` to the twenty milk-container fills of
        Krishnamoorthy and Mathew (2009) and prints the interval
        ``(0.9523519, 1.054848)`` about the mean 1.0036; its Eq. (7) is the
        integral equation solved here. The NIST/SEMATECH handbook evaluates
        Howe's formula for ``N = 43``, ``p = 0.90``, confidence 0.99 as 2.217.

        References:
            [REF-YOUNG-2016], [REF-NIST-SEMATECH-HANDBOOK], [REF-HOWE-1969].
            Checks [EQ-PROP-TOL-003] and [EQ-PROP-TOL-004].

        """
        milk = (
            0.968, 0.982, 1.030, 1.003, 1.046, 1.020, 0.997, 1.010, 1.027, 1.010,
            0.973, 1.000, 1.044, 0.995, 1.020, 0.993, 0.984, 0.981, 0.997, 0.992,
        )  # fmt: skip
        interval = normal_tolerance_interval(milk, 0.90, 0.95)
        self.assertAlmostEqual(interval.mean, 1.0036, delta=1e-12)
        self.assertAlmostEqual(interval.lower, 0.9523519, delta=1e-6)
        self.assertAlmostEqual(interval.upper, 1.054848, delta=1e-6)
        howe = normal_tolerance_factor(43, 0.90, 0.99, sided="two-sided", method="howe")
        self.assertAlmostEqual(howe, 2.217, delta=5e-4)

    def test_python_brute_force_quadrature(self) -> None:
        """The factor solves the equation integrated in the opposite order.

        References:
            [EQ-PROP-TOL-003].

        """
        for n, p, gamma in ((7, 0.90, 0.90), (15, 0.95, 0.95)):
            with self.subTest(n=n):
                k = normal_tolerance_factor(n, p, gamma, sided="two-sided")
                confidence = brute_force_two_sided_confidence(n, p, k)
                self.assertAlmostEqual(confidence, gamma, delta=2e-6)

    def test_seeded_content_coverage(self) -> None:
        """The interval holds content ``p`` with frequency ``gamma``.

        The true content of ``xbar +- k s`` for the standard normal law is
        ``Phi(b) - Phi(a)``; its exceedance frequency over 3000 seeded
        samples of size 6 is judged on the binomial standard error.

        References:
            [EQ-PROP-TOL-003].

        """
        n, p, gamma, reps = 6, 0.90, 0.90, 3000
        k = normal_tolerance_factor(n, p, gamma, sided="two-sided")
        rng = random.Random(7_654_321)
        hits = 0
        for _ in range(reps):
            sample = [rng.gauss(0.0, 1.0) for _ in range(n)]
            centre = sum(sample) / n
            spread = math.sqrt(sum((x - centre) ** 2 for x in sample) / (n - 1))
            content = STANDARD.cdf(centre + k * spread) - STANDARD.cdf(
                centre - k * spread
            )
            hits += content >= p
        standard_error = math.sqrt(gamma * (1 - gamma) / reps)
        self.assertLess(abs(hits / reps - gamma), 4.0 * standard_error)

    def test_howe_matches_its_formula_in_r(self) -> None:
        """Howe's factor equals ``z sqrt(nu (1 + 1/n) / qchisq(1 - g, nu))``.

        References:
            [EQ-PROP-TOL-004], [REF-HOWE-1969].

        """
        # R: qnorm((1 + p) / 2) * sqrt((n - 1) * (1 + 1 / n) / qchisq(1 - g, n - 1))
        pins = (
            (10, 0.95, 0.95, 3.381913490509),
            (30, 0.90, 0.99, 2.384738196951),
            (3, 0.95, 0.95, 9.992799249704),
        )
        for n, p, gamma, expected in pins:
            with self.subTest(n=n):
                value = normal_tolerance_factor(
                    n, p, gamma, sided="two-sided", method="howe"
                )
                self.assertAlmostEqual(value, expected, delta=1e-9)

    def test_howe_error_against_exact_is_the_documented_one(self) -> None:
        """Howe's relative error at ``p = gamma = 0.95`` matches the module note.

        References:
            [EQ-PROP-TOL-004], [EQ-PROP-TOL-003].

        """
        # (n, bound on |relative error|, sign): over at n <= 5, under later.
        expected = (
            (3, 0.0208, 1),
            (5, 0.0033, 1),
            (10, 0.0034, -1),
            (30, 0.0021, -1),
            (100, 0.0005, -1),
        )
        for n, size, sign in expected:
            with self.subTest(n=n):
                exact = normal_tolerance_factor(n, 0.95, 0.95, sided="two-sided")
                howe = normal_tolerance_factor(
                    n, 0.95, 0.95, sided="two-sided", method="howe"
                )
                error = howe / exact - 1.0
                self.assertEqual(math.copysign(1.0, error), float(sign))
                self.assertAlmostEqual(abs(error), size, delta=0.00006)

    def test_two_sided_exceeds_one_sided_at_the_same_content(self) -> None:
        """A two-sided factor is larger than the one-sided factor for ``p``.

        References:
            [EQ-PROP-TOL-003], [EQ-PROP-TOL-001].

        """
        one = normal_tolerance_factor(12, 0.95, 0.95)
        two = normal_tolerance_factor(12, 0.95, 0.95, sided="two-sided")
        self.assertGreater(two, one)


class BoundsAndSizesTests(unittest.TestCase):
    """Bounds, intervals, sample sizes, and the comparison with Wilks.

    References:
        [REF-KRISHNAMOORTHY-MATHEW-2009], [REF-WILKS-1941].
        Checks [EQ-PROP-TOL-001] and [EQ-PROP-WILKS-001].

    """

    VALUES = (9.1, 10.4, 9.8, 10.9, 10.2, 9.5, 10.0, 10.7, 9.9, 10.3)

    def test_bound_uses_sample_mean_and_standard_deviation(self) -> None:
        """``bound = xbar +- k s`` with the pinned factor ``k(10, .95, .95)``.

        References:
            [EQ-PROP-TOL-001].

        """
        n = len(self.VALUES)
        centre = sum(self.VALUES) / n
        spread = math.sqrt(sum((v - centre) ** 2 for v in self.VALUES) / (n - 1))
        # R: qt(0.95, 9, ncp = qnorm(0.95) * sqrt(10)) / sqrt(10)
        k = 2.910963413081
        upper = normal_tolerance_bound(self.VALUES, 0.95, 0.95)
        lower = normal_tolerance_bound(self.VALUES, 0.95, 0.95, sided="lower")
        self.assertAlmostEqual(upper.bound, centre + k * spread, delta=1e-8)
        self.assertAlmostEqual(lower.bound, centre - k * spread, delta=1e-8)
        self.assertAlmostEqual(upper.factor, k, delta=1e-9)
        self.assertEqual((upper.size, upper.sided), (10, "upper"))

    def test_interval_is_symmetric_about_the_mean(self) -> None:
        """The two-sided interval is ``xbar +- k s`` for both methods.

        References:
            [EQ-PROP-TOL-003], [EQ-PROP-TOL-004].

        """
        for method in ("exact", "howe"):
            with self.subTest(method=method):
                result = normal_tolerance_interval(
                    self.VALUES, 0.9, 0.95, method=method
                )
                self.assertAlmostEqual(
                    0.5 * (result.lower + result.upper), result.mean, delta=1e-12
                )
                self.assertAlmostEqual(
                    result.upper - result.mean,
                    result.factor * result.standard_deviation,
                    delta=1e-12,
                )
                self.assertEqual(result.method, method)

    def test_sample_size_is_the_smallest_meeting_the_target(self) -> None:
        """``n`` meets the target and ``n - 1`` does not.

        References:
            [EQ-PROP-TOL-001].

        """
        target = 2.0
        n = normal_tolerance_sample_size(0.95, 0.95, target)
        self.assertLessEqual(normal_tolerance_factor(n, 0.95, 0.95), target)
        self.assertGreater(normal_tolerance_factor(n - 1, 0.95, 0.95), target)
        two = normal_tolerance_sample_size(
            0.95, 0.95, 2.5, sided="two-sided", method="howe"
        )
        self.assertLessEqual(
            normal_tolerance_factor(two, 0.95, 0.95, sided="two-sided", method="howe"),
            2.5,
        )
        self.assertGreater(
            normal_tolerance_factor(
                two - 1, 0.95, 0.95, sided="two-sided", method="howe"
            ),
            2.5,
        )

    def test_price_of_normality_for_a_small_and_a_large_sample(self) -> None:
        """A small sample has no Wilks bound; a large one does and it is wider.

        References:
            [REF-WILKS-1941], [EQ-PROP-WILKS-001].

        """
        small = normality_price(self.VALUES, 0.95, 0.95)
        self.assertIsNone(small.wilks)
        self.assertIsNone(small.difference)
        self.assertEqual(small.wilks_required_size, 59)
        rng = random.Random(11)
        big = [rng.gauss(10.0, 1.0) for _ in range(300)]
        both = normality_price(big, 0.95, 0.95)
        assert both.wilks is not None
        self.assertEqual(both.wilks.bound, wilks_bound(big, 0.95, 0.95).bound)
        assert both.difference is not None
        self.assertAlmostEqual(
            both.difference, both.wilks.bound - both.normal.bound, delta=1e-12
        )
        # normal data: both bounds estimate the same quantile 10 + 1.645
        self.assertAlmostEqual(both.normal.bound, 11.645, delta=0.3)

    def test_price_of_normality_lower_side_bounds_the_same_quantile(self) -> None:
        """Both sides: normal and Wilks bound estimate one quantile; size is 59.

        For ``sided="lower"`` the bound ``xbar - k s`` lies below the
        ``1 - coverage`` quantile, so the Wilks bound must be taken at level
        ``1 - coverage``; at the unchanged level it would bound the
        ``coverage`` quantile from below (about 11.3 for N(10, 1) at 0.95).
        The oracle is the closed-form quantile ``10 -+ z_0.95``, the Wilks
        size 59 for 95/95 on each side, and the documented sign of
        ``difference``.

        References:
            [REF-WILKS-1941], [EQ-PROP-TOL-001], [EQ-PROP-WILKS-001].

        """
        rng = random.Random(11)
        big = [rng.gauss(10.0, 1.0) for _ in range(300)]
        z = STANDARD.inv_cdf(0.95)
        for side, quantile in (("upper", 10.0 + z), ("lower", 10.0 - z)):
            price = normality_price(big, 0.95, 0.95, sided=side)
            self.assertEqual(price.wilks_required_size, 59)
            assert price.wilks is not None
            assert price.difference is not None
            # both bounds sit within sampling error of the same quantile
            self.assertAlmostEqual(price.normal.bound, quantile, delta=0.4)
            self.assertAlmostEqual(price.wilks.bound, quantile, delta=0.6)
            self.assertAlmostEqual(
                price.wilks.bound, price.normal.bound, delta=0.6
            )
            if side == "upper":
                expected = price.wilks.bound - price.normal.bound
                self.assertGreater(price.wilks.bound, quantile - 0.6)
            else:
                expected = price.normal.bound - price.wilks.bound
                self.assertLess(price.wilks.bound, quantile + 0.6)
            self.assertAlmostEqual(price.difference, expected, delta=1e-12)
        lower = normality_price(big, 0.95, 0.95, sided="lower")
        assert lower.wilks is not None
        self.assertEqual(lower.wilks.bound, wilks_bound(big, 0.05, 0.95, sided="lower").bound)
        self.assertEqual(lower.wilks.sided, "lower")
        # a sample of 58 is too small on both sides
        for side in ("upper", "lower"):
            self.assertIsNone(normality_price(big[:58], 0.95, 0.95, sided=side).wilks)
            self.assertIsNotNone(normality_price(big[:59], 0.95, 0.95, sided=side).wilks)

    def test_argument_validation(self) -> None:
        """Bad types, ranges, and option names raise ``TypeError`` or ``ValueError``.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].

        """
        with self.assertRaises(TypeError):
            normal_tolerance_factor(5.0, 0.9, 0.9)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            normal_tolerance_factor(True, 0.9, 0.9)
        with self.assertRaises(ValueError):
            normal_tolerance_factor(1, 0.9, 0.9)
        with self.assertRaises(ValueError):
            normal_tolerance_factor(5, 1.0, 0.9)
        with self.assertRaises(ValueError):
            normal_tolerance_factor(5, 0.9, 0.0)
        with self.assertRaises(ValueError):
            normal_tolerance_factor(5, 0.9, 0.9, sided="both")
        with self.assertRaises(ValueError):
            normal_tolerance_factor(5, 0.9, 0.9, method="howe")
        with self.assertRaises(ValueError):
            normal_tolerance_factor(5, 1.0 - 1e-12, 0.9, sided="two-sided")
        with self.assertRaises(ValueError):
            normal_tolerance_bound([1.0], 0.9, 0.9)
        with self.assertRaises(ValueError):
            normal_tolerance_bound([1.0, math.nan], 0.9, 0.9)
        with self.assertRaises(ValueError):
            normal_tolerance_bound([1.0, 2.0], 0.9, 0.9, sided="middle")
        with self.assertRaises(ValueError):
            normal_tolerance_sample_size(0.95, 0.95, 1.5)
        with self.assertRaises(ValueError):
            normal_tolerance_sample_size(0.95, 0.95, 1.65, max_size=64)


class RequirementAndMarginTests(unittest.TestCase):
    """Requirements, the QMU record, and the mixed-uncertainty split.

    References:
        [REF-HELTON-2011], [REF-PILCH-TRUCANO-HELTON-2006].
        Checks [EQ-PROP-MARGIN-001].

    """

    def test_margin_of_each_requirement_kind(self) -> None:
        """Distances to an upper limit, a lower limit, and the near edge of a band.

        References:
            [EQ-PROP-MARGIN-001].

        """
        self.assertEqual(Requirement.upper_limit(100.0).margin(80.0), 20.0)
        self.assertEqual(Requirement.lower_limit(10.0).margin(14.0), 4.0)
        band = Requirement.band(10.0, 20.0)
        self.assertEqual(band.margin(13.0), 3.0)
        self.assertEqual(band.margin(18.0), 2.0)
        self.assertLess(Requirement.upper_limit(5.0).margin(6.0), 0.0)
        self.assertTrue(Requirement.upper_limit(5.0).is_met(5.0))
        self.assertFalse(band.is_met(20.5))

    def test_moments_method_has_a_closed_form(self) -> None:
        """``U = z_p u`` and ``M/U`` follow from the normal quantile.

        References:
            [EQ-PROP-MARGIN-001].

        """
        result = quantify_margin(
            Requirement.upper_limit(100.0),
            80.0,
            standard_uncertainty=5.0,
            method="moments",
        )
        z = STANDARD.inv_cdf(0.95)
        self.assertAlmostEqual(result.uncertainty, 5.0 * z, delta=1e-12)
        self.assertAlmostEqual(result.confidence_ratio, 20.0 / (5.0 * z), delta=1e-12)
        self.assertIsNone(result.confidence)
        shifted = quantify_margin(
            Requirement.lower_limit(60.0),
            80.0,
            standard_uncertainty=5.0,
            mean=78.0,
            coverage_factor=2.0,
            method="moments",
        )
        # lower side: bound = 78 - 2 * 5 = 68, U = 80 - 68 = 12, M = 20
        self.assertAlmostEqual(shifted.uncertainty, 12.0, delta=1e-12)
        self.assertAlmostEqual(shifted.confidence_ratio, 20.0 / 12.0, delta=1e-12)

    def test_wilks_and_normal_methods_use_their_bounds(self) -> None:
        """``U`` is the library bound minus the estimate, per side.

        References:
            [EQ-PROP-MARGIN-001], [REF-WILKS-1941].

        """
        data = [float(i) for i in range(1, 101)]
        # R: min(which(pbinom(0:99, 100, 0.95) >= 0.95)) -> 99 (rank of the bound)
        upper = quantify_margin(Requirement.upper_limit(150.0), 60.0, samples=data)
        self.assertEqual(upper.method, "wilks")
        self.assertEqual(upper.binding.bound, 99.0)
        self.assertEqual(upper.uncertainty, 39.0)
        self.assertEqual(upper.margin, 90.0)
        # lower side bounds the 5 % quantile from below at rank 2 by symmetry
        lower = quantify_margin(Requirement.lower_limit(0.0), 60.0, samples=data)
        self.assertEqual(lower.binding.bound, 2.0)
        self.assertEqual(lower.uncertainty, 58.0)
        normal = quantify_margin(
            Requirement.upper_limit(150.0),
            60.0,
            samples=data[:10],
            method="normal-tolerance",
        )
        centre = sum(data[:10]) / 10
        spread = math.sqrt(sum((v - centre) ** 2 for v in data[:10]) / 9)
        # R: qt(0.95, 9, ncp = qnorm(0.95) * sqrt(10)) / sqrt(10) -> 2.910963413081
        self.assertAlmostEqual(
            normal.uncertainty, centre + 2.910963413081 * spread - 60.0, delta=1e-8
        )
        self.assertEqual(normal.size, 10)

    def test_band_reports_the_binding_side(self) -> None:
        """The side with the smaller ratio binds; ratios use their own side.

        References:
            [EQ-PROP-MARGIN-001].

        """
        result = quantify_margin(
            Requirement.band(0.0, 100.0),
            70.0,
            standard_uncertainty=10.0,
            coverage_factor=2.0,
            method="moments",
        )
        sides = {side.side: side for side in result.sides}
        self.assertAlmostEqual(sides["upper"].confidence_ratio, 30.0 / 20.0)
        self.assertAlmostEqual(sides["lower"].confidence_ratio, 70.0 / 20.0)
        self.assertEqual(result.binding.side, "upper")
        self.assertAlmostEqual(result.confidence_ratio, 1.5)

    def test_ratio_when_the_bound_is_on_the_safe_side(self) -> None:
        """``U <= 0`` gives an infinite ratio for a positive margin.

        References:
            [EQ-PROP-MARGIN-001].

        """
        result = quantify_margin(
            Requirement.upper_limit(100.0),
            80.0,
            standard_uncertainty=5.0,
            mean=60.0,
            coverage_factor=1.0,
            method="moments",
        )
        self.assertLess(result.uncertainty, 0.0)
        self.assertEqual(result.confidence_ratio, math.inf)
        negative = quantify_margin(
            Requirement.upper_limit(70.0),
            80.0,
            standard_uncertainty=0.0,
            method="moments",
        )
        self.assertEqual(negative.confidence_ratio, -math.inf)

    def test_mixed_split_is_exact_on_two_outer_points(self) -> None:
        """Aleatory part from the best outer point, epistemic part is the rest.

        Two outer points have inner samples ``1..100`` and ``11..110``; the
        type-7 median of ``1..100`` is 50.5 and of ``11..110`` is 60.5.

        References:
            [EQ-PROP-MARGIN-001], [REF-HELTON-2011].

        """
        first = tuple(float(i) for i in range(1, 101))
        second = tuple(float(i) for i in range(11, 111))
        box = ProbabilityBox((0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (first, second))
        upper = quantify_margin_mixed(
            Requirement.upper_limit(200.0), 50.0, box, probability=0.5
        )
        side = upper.binding
        self.assertAlmostEqual(side.bound, 60.5, delta=1e-12)
        self.assertAlmostEqual(side.uncertainty, 10.5, delta=1e-12)
        assert side.aleatory_uncertainty is not None
        assert side.epistemic_uncertainty is not None
        self.assertAlmostEqual(side.aleatory_uncertainty, 0.5, delta=1e-12)
        self.assertAlmostEqual(side.epistemic_uncertainty, 10.0, delta=1e-12)
        lower = quantify_margin_mixed(
            Requirement.lower_limit(0.0), 50.0, box, probability=0.5
        )
        # lower side: quantile level 0.5 again; worst is the smaller median 50.5
        self.assertAlmostEqual(lower.binding.bound, 50.5, delta=1e-12)
        with self.assertRaises(TypeError):
            quantify_margin_mixed(Requirement.upper_limit(1.0), 0.0, "box")  # type: ignore[arg-type]

    def test_quantify_margin_validation(self) -> None:
        """Input combinations that do not match the method are rejected.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].

        """
        req = Requirement.upper_limit(10.0)
        with self.assertRaises(ValueError):
            quantify_margin(req, 1.0, method="moments")
        with self.assertRaises(ValueError):
            quantify_margin(req, 1.0, samples=[1.0, 2.0], method="moments")
        with self.assertRaises(ValueError):
            quantify_margin(req, 1.0)
        with self.assertRaises(ValueError):
            quantify_margin(req, 1.0, samples=[1.0] * 10)  # too few for Wilks
        with self.assertRaises(ValueError):
            quantify_margin(req, 1.0, standard_uncertainty=-1.0, method="moments")
        with self.assertRaises(ValueError):
            quantify_margin(req, 1.0, samples=[1.0, 2.0], method="bayes")
        with self.assertRaises(TypeError):
            quantify_margin(req, "x", standard_uncertainty=1.0, method="moments")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            Requirement("upper")
        with self.assertRaises(ValueError):
            Requirement.band(2.0, 1.0)
        with self.assertRaises(ValueError):
            Requirement("sideways", 0.0, 1.0)


class ComplianceTests(unittest.TestCase):
    """Compliance probability with the exact interval.

    References:
        [REF-CLOPPER-PEARSON-1934]. Checks [EQ-INF-CI-CP-001].

    """

    def test_interval_matches_r_binom_test(self) -> None:
        """90 of 100 samples meet the limit: interval and one-sided bound from R.

        References:
            [REF-CLOPPER-PEARSON-1934].

        """
        values = [float(i) for i in range(1, 101)]
        result = compliance_probability(values, 90.0)
        self.assertEqual((result.successes, result.trials), (90, 100))
        # R: binom.test(90, 100)$conf.int
        self.assertAlmostEqual(result.interval.lower, 0.823777402260, delta=1e-9)
        self.assertAlmostEqual(result.interval.upper, 0.950995310779, delta=1e-9)
        # R: binom.test(90, 100, alternative = "greater")$conf.int[1]
        assert result.lower_bound is not None
        self.assertAlmostEqual(result.lower_bound, 0.836282376724, delta=1e-9)
        lower = compliance_probability(values, 11.0, sense="lower")
        self.assertEqual(lower.successes, 90)

    def test_success_run_closed_form(self) -> None:
        """Zero failures in 59 samples give a one-sided bound ``0.05^(1/59)``.

        References:
            [REF-CLOPPER-PEARSON-1934], [REF-WILKS-1941].

        """
        result = compliance_probability([0.0] * 59, 1.0)
        assert result.lower_bound is not None
        self.assertAlmostEqual(result.lower_bound, 0.05 ** (1 / 59), delta=1e-9)
        self.assertEqual(result.probability, 1.0)
        none = compliance_probability([0.0] * 59, 1.0, confidence=0.4)
        self.assertIsNone(none.lower_bound)

    def test_requirement_object_and_validation(self) -> None:
        """A band requirement is accepted; empty or bad input is rejected.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].

        """
        band = Requirement.band(2.0, 4.0)
        self.assertEqual(
            compliance_probability([1.0, 2.0, 3.0, 5.0], band).successes, 2
        )
        with self.assertRaises(ValueError):
            compliance_probability([], 1.0)
        with self.assertRaises(ValueError):
            compliance_probability([1.0], 1.0, sense="band")
        with self.assertRaises(TypeError):
            compliance_probability([1.0], "a")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            compliance_probability([1.0], 1.0, confidence=1.0)


class ContingencyTests(unittest.TestCase):
    """Deterministic contingency arithmetic and its probabilistic reading.

    References:
        [REF-NASA-SEH-2016]. Checks [EQ-PROP-MARGIN-002] and
        [EQ-PROP-MARGIN-003].

    """

    def test_growth_then_margin_arithmetic(self) -> None:
        """100 with 15 % growth and a 25 % margin relative to the estimate.

        References:
            [EQ-PROP-MARGIN-002].

        """
        budget = apply_contingency(100.0, 0.15, 0.25)
        self.assertAlmostEqual(budget.growth_allowance, 15.0)
        self.assertAlmostEqual(budget.predicted, 115.0)
        self.assertAlmostEqual(budget.allocation, 143.75)
        self.assertAlmostEqual(budget.system_margin, 28.75)

    def test_the_two_margin_bases_differ_for_one_allocation(self) -> None:
        """The same allocation is 25 % relative to the estimate, 20 % to the allocation.

        References:
            [EQ-PROP-MARGIN-002].

        """
        other = apply_contingency(100.0, 0.15, 0.20, margin_basis="allocation")
        self.assertAlmostEqual(other.allocation, 143.75)
        self.assertAlmostEqual(allocation_margin_fraction(115.0, 143.75), 0.25)
        self.assertAlmostEqual(
            allocation_margin_fraction(115.0, 143.75, basis="allocation"), 0.20
        )
        self.assertAlmostEqual(
            convert_margin_fraction(0.25, source="estimate", target="allocation"), 0.20
        )
        self.assertAlmostEqual(
            convert_margin_fraction(0.20, source="allocation", target="estimate"), 0.25
        )
        self.assertEqual(
            convert_margin_fraction(0.3, source="estimate", target="estimate"), 0.3
        )
        self.assertLess(allocation_margin_fraction(150.0, 120.0), 0.0)

    def test_contingency_validation(self) -> None:
        """Out-of-range inputs raise ``ValueError``; wrong types ``TypeError``.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].

        """
        with self.assertRaises(ValueError):
            apply_contingency(0.0, 0.1, 0.1)
        with self.assertRaises(ValueError):
            apply_contingency(1.0, -0.1, 0.1)
        with self.assertRaises(ValueError):
            apply_contingency(1.0, 0.1, 1.0, margin_basis="allocation")
        with self.assertRaises(ValueError):
            apply_contingency(1.0, 0.1, 0.1, margin_basis="share")
        with self.assertRaises(ValueError):
            convert_margin_fraction(1.0, source="allocation", target="estimate")
        with self.assertRaises(TypeError):
            apply_contingency("1", 0.1, 0.1)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            allocation_margin_fraction(0.0, 1.0)

    def test_bridge_between_percentage_and_confidence(self) -> None:
        """Samples 100..200: the 70 % level is 170, i.e. a 70 % contingency.

        The type-7 quantile of 101 equally spaced values at level 0.7 sits at
        index 70 exactly. Reading 0.7 back gives 71 of 101 samples at or
        below 170, with the exact interval from R.

        References:
            [EQ-PROP-MARGIN-003], [REF-CLOPPER-PEARSON-1934].

        """
        data = [100.0 + i for i in range(101)]
        level = contingency_for_confidence(data, 100.0, 0.7)
        self.assertAlmostEqual(level.value, 170.0, delta=1e-12)
        self.assertAlmostEqual(level.contingency_fraction, 0.7, delta=1e-12)
        self.assertEqual(level.method, "empirical-quantile")
        back = confidence_of_contingency(data, 100.0, 0.7)
        self.assertEqual((back.successes, back.trials), (71, 101))
        # R: binom.test(71, 101)$conf.int
        self.assertAlmostEqual(back.interval.lower, 0.603852871352, delta=1e-9)
        self.assertAlmostEqual(back.interval.upper, 0.789807793505, delta=1e-9)
        guarded = contingency_for_confidence(
            data, 100.0, 0.7, statistical_confidence=0.9
        )
        self.assertEqual(guarded.method, "wilks")
        self.assertEqual(guarded.value, wilks_bound(data, 0.7, 0.9).bound)
        self.assertGreaterEqual(guarded.value, level.value)
        with self.assertRaises(ValueError):
            contingency_for_confidence(data, 0.0, 0.7)
        with self.assertRaises(ValueError):
            confidence_of_contingency(data, -1.0, 0.1)


class PerformanceMeasureTests(unittest.TestCase):
    """Tracking one measure across milestones.

    References:
        [REF-NASA-SEH-2016]. Checks [EQ-PROP-MARGIN-001] and
        [EQ-PROP-MARGIN-003].

    """

    @staticmethod
    def measure() -> TechnicalPerformanceMeasure:
        """Return a mass measure with a 100 kg limit over four milestones.

        References:
            [REF-NASA-SEH-2016]. No mathematical equation is evaluated.

        """
        rng = random.Random(5)
        sampled = tuple(rng.gauss(88.0, 2.0) for _ in range(40))
        return TechnicalPerformanceMeasure(
            "dry mass",
            "kg",
            Requirement.upper_limit(100.0),
            (
                Milestone("SRR", 70.0, standard_uncertainty=8.0, planned_margin=20.0),
                Milestone("PDR", 78.0, standard_uncertainty=4.0, planned_margin=15.0),
                Milestone("CDR", 88.0, samples=sampled, planned_margin=10.0),
                Milestone("FRR", 101.0, standard_uncertainty=0.0, planned_value=95.0),
            ),
        )

    def test_margin_history_and_plan_comparison(self) -> None:
        """Margins are ``100 - estimate``; the plan gap is margin minus plan.

        References:
            [EQ-PROP-MARGIN-001].

        """
        tpm = self.measure()
        self.assertEqual(tpm.margin_history(), (30.0, 22.0, 12.0, -1.0))
        self.assertEqual(tpm.plan_comparison(), (10.0, 7.0, 2.0, None))

    def test_compliance_probabilities(self) -> None:
        """Normal milestones match ``Phi((L - y)/u)``; zero spread is 0 or 1.

        References:
            [EQ-PROP-MARGIN-003].

        """
        probabilities = self.measure().compliance_probabilities()
        self.assertAlmostEqual(probabilities[0], STANDARD.cdf(30.0 / 8.0), delta=1e-12)
        self.assertAlmostEqual(probabilities[1], STANDARD.cdf(22.0 / 4.0), delta=1e-12)
        self.assertEqual(probabilities[3], 0.0)
        assert probabilities[2] is not None
        self.assertGreater(probabilities[2], 0.95)

    def test_assessment_flags_and_table(self) -> None:
        """The negative margin and the low ratio are flagged; the table lists all.

        References:
            [EQ-PROP-MARGIN-001].

        """
        tight = TechnicalPerformanceMeasure(
            "link margin",
            "dB",
            Requirement.lower_limit(3.0),
            (
                Milestone("PDR", 9.0, standard_uncertainty=2.0, planned_margin=5.0),
                Milestone("CDR", 6.0, standard_uncertainty=2.0, planned_margin=5.0),
                Milestone("FRR", 2.0, standard_uncertainty=1.0),
            ),
        )
        assessment = tight.assess(probability=0.95, ratio_threshold=1.5)
        z = STANDARD.inv_cdf(0.95)
        # PDR: M = 6, U = 2 z -> ratio 6 / (2 z) = 1.82 (above 1.5, on plan)
        self.assertAlmostEqual(
            assessment.milestones[0].confidence_ratio, 6.0 / (2.0 * z), delta=1e-12
        )
        reasons = [(flag.label, flag.reason) for flag in assessment.flags]
        self.assertEqual(
            reasons,
            [
                ("CDR", "below-plan"),
                ("CDR", "low-ratio"),
                ("FRR", "requirement-violated"),
                ("FRR", "low-ratio"),
            ],
        )
        text = assessment.render()
        for label in ("PDR", "CDR", "FRR", "link margin", "flags:"):
            self.assertIn(label, text)
        self.assertEqual(self.measure().assess().flags[0].label, "FRR")

    def test_sampled_milestone_uses_its_samples(self) -> None:
        """A sampled milestone reports the method and an exact interval.

        References:
            [REF-CLOPPER-PEARSON-1934].

        """
        item = self.measure().assess().milestones[2]
        self.assertEqual(item.method, "normal-tolerance")
        assert item.compliance_interval is not None
        self.assertLessEqual(
            item.compliance_interval[0], item.compliance_probability or 0
        )
        tolerance = self.measure().assess(method="normal-tolerance")
        self.assertIsNotNone(tolerance.milestones[0].uncertainty)
        with self.assertRaises(ValueError):
            self.measure().assess(method="moments")

    def test_validation(self) -> None:
        """Duplicate labels, empty lists, and doubled uncertainty are rejected.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].

        """
        req = Requirement.upper_limit(1.0)
        with self.assertRaises(ValueError):
            TechnicalPerformanceMeasure("m", "u", req, ())
        with self.assertRaises(ValueError):
            TechnicalPerformanceMeasure(
                "m", "u", req, (Milestone("a", 0.0), Milestone("a", 0.1))
            )
        with self.assertRaises(ValueError):
            Milestone("a", 0.0, standard_uncertainty=1.0, samples=(0.0, 1.0))
        with self.assertRaises(ValueError):
            Milestone("", 0.0)
        with self.assertRaises(ValueError):
            Milestone("a", 0.0, standard_uncertainty=-1.0)
        with self.assertRaises(TypeError):
            Milestone(1, 0.0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            TechnicalPerformanceMeasure("", "u", req, (Milestone("a", 0.0),))


if __name__ == "__main__":
    unittest.main()
