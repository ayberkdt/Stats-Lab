"""Chance constraints, scenario guarantees, and reliability-based design optimisation.

Exact values are pinned from base R 4.6.2 (``pbinom``, ``qbeta``, ``qnorm``,
``uniroot``); each pin carries the R expression that produced it. The tests
never call R. Sampling checks compare an observed frequency with an exact
probability on the scale of its binomial standard error, never pinning the
digits of a seeded result. The linear-normal designs have closed forms.

References:
    [REF-CAMPI-GARATTI-2008], [REF-CAMPI-GARATTI-2018],
    [REF-CLOPPER-PEARSON-1934], [REF-DU-CHEN-2004], [REF-TU-CHOI-PARK-1999],
    [REF-NOCEDAL-WRIGHT-2006]. Checks [EQ-REL-SCENARIO-001],
    [EQ-REL-SCENARIO-002], [EQ-REL-SCENARIO-003], [EQ-REL-SAA-001],
    [EQ-REL-CHANCE-NORMAL-001], [EQ-REL-CHANCE-FREE-001],
    [EQ-REL-AUGLAG-001], [EQ-REL-SORA-001], [EQ-REL-RBDO-002].

"""

from __future__ import annotations

import itertools
import math
import random
import unittest
from fractions import Fraction
from statistics import NormalDist

from probstat.distributions import Normal
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.reliability import (
    ModelLimitState,
    certify_design,
    certify_violation_frequency,
    distribution_free_chance_margin,
    minimize_constrained,
    normal_linear_chance_margin,
    normal_linear_violation_probability,
    rbdo_double_loop,
    rbdo_sora,
    scenario_confidence,
    scenario_posterior_violation_bound,
    scenario_sample_size,
    scenario_violation_bound,
    solve_chance_constrained,
    verify_design_reliability,
)
from probstat.uq import QuantityOfInterest

Y = (QuantityOfInterest("y", "response"),)
NORMAL = NormalDist()


def exact_tail(n: int, eps: float, d: int) -> float:
    """Return ``P(Binomial(n, eps) <= d - 1)`` by exact rational arithmetic."""
    p = Fraction(eps)
    total = sum(
        math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(min(d, n + 1))
    )
    return float(total)


class ScenarioGuaranteeTests(unittest.TestCase):
    """Scenario sample sizes, bounds, and wait-and-judge levels against R.

    References:
        [REF-CAMPI-GARATTI-2008], [REF-CAMPI-GARATTI-2018]. Checks
        [EQ-REL-SCENARIO-001], [EQ-REL-SCENARIO-002], [EQ-REL-SCENARIO-003].

    """

    def test_sample_sizes_of_the_published_table(self) -> None:
        """``d = 10``, ``beta = 1e-5`` reproduces the sizes of Campi-Garatti Table 2.2.

        References:
            [REF-CAMPI-GARATTI-2008]. Checks [EQ-REL-SCENARIO-001].

        """
        # R: min(which(pbinom(9, 10:40000, eps) <= 1e-5)) + 9
        #    -> 285 (0.1), 581 (0.05), 1171 (0.025), 2942 (0.01)
        for eps, expected in ((0.1, 285), (0.05, 581), (0.025, 1171), (0.01, 2942)):
            result = scenario_sample_size(eps, 1.0 - 1e-5, 10)
            self.assertEqual(result.size, expected)
            self.assertLessEqual(exact_tail(expected, eps, 10), 1e-5)
            self.assertGreater(exact_tail(expected - 1, eps, 10), 1e-5)

    def test_sample_size_pin_and_explicit_bound(self) -> None:
        """``d = 3``, ``eps = 0.05``, confidence 0.99 gives 165; explicit size 265.

        References:
            [REF-CAMPI-GARATTI-2008]. Checks [EQ-REL-SCENARIO-002].

        """
        result = scenario_sample_size(0.05, 0.99, 3)
        # R: pbinom(2, 165, 0.05) -> 0.0099540610451930526
        #    pbinom(2, 164, 0.05) -> 0.010365339090319229 (> 0.01)
        self.assertEqual(result.size, 165)
        self.assertAlmostEqual(
            result.achieved_confidence, 1 - 0.0099540610451930526, 12
        )
        # ceiling(2 / 0.05 * (2 + log(100))) in R -> 265
        self.assertEqual(result.explicit_size, 265)

    def test_explicit_size_is_sufficient_on_a_grid(self) -> None:
        """The explicit size always satisfies the exact binomial condition.

        References:
            [EQ-REL-SCENARIO-002].

        """
        for eps, conf, d in itertools.product((0.3, 0.1), (0.9, 0.999), (1, 2, 7)):
            result = scenario_sample_size(eps, conf, d)
            self.assertGreaterEqual(result.explicit_size, result.size)
            self.assertLessEqual(
                1.0 - scenario_confidence(result.explicit_size, d, eps), 1 - conf
            )
            self.assertLessEqual(exact_tail(result.size, eps, d), 1 - conf + 1e-15)
            if result.size > d:
                self.assertGreater(exact_tail(result.size - 1, eps, d), 1 - conf)

    def test_confidence_matches_the_exact_tail_and_the_library_binomial(self) -> None:
        """``scenario_confidence`` equals one minus the exact rational binomial tail.

        References:
            [EQ-REL-SCENARIO-001].

        """
        from probstat.distributions import Binomial

        for n, d, eps in ((50, 3, 0.1), (400, 8, 0.03), (12, 12, 0.5)):
            value = scenario_confidence(n, d, eps)
            self.assertAlmostEqual(value, 1.0 - exact_tail(n, eps, d), 12)
            self.assertAlmostEqual(value, 1.0 - Binomial(n, eps).cdf(d - 1), 9)

    def test_violation_bound_pin_and_round_trip(self) -> None:
        """The a-priori bound for ``N = 200``, ``d = 4`` matches R and inverts the tail.

        References:
            [EQ-REL-SCENARIO-001].

        """
        eps = scenario_violation_bound(200, 4, 0.95)
        # R: uniroot(function(e) pbinom(3, 200, e) - 0.05, c(1e-6, .999),
        #            tol = 1e-15)$root
        self.assertAlmostEqual(eps, 0.038309708498569357, 11)
        self.assertAlmostEqual(scenario_confidence(200, 4, eps), 0.95, 10)
        self.assertEqual(scenario_violation_bound(3, 4, 0.95), 1.0)
        self.assertLess(scenario_violation_bound(1000, 4, 0.95), eps)

    def test_posterior_bound_pins(self) -> None:
        """The wait-and-judge level matches R's root of the published polynomial.

        References:
            [REF-CAMPI-GARATTI-2018]. Checks [EQ-REL-SCENARIO-003].

        """
        # R: wj <- function(N,k,beta){ g <- function(t){ m <- k:N;
        #   beta/(N+1)*sum(choose(m,k)*t^(m-k)) - choose(N,k)*t^(N-k) };
        #   1-uniroot(g, c(1e-9,1-1e-12), tol=1e-15)$root }
        pins = (
            (0, 0.086787700715337035),
            (1, 0.11189369889098022),
            (2, 0.13290800279709392),
            (3, 0.15186487463093123),
            (4, 0.16950598600808442),
        )
        for k, expected in pins:
            value = scenario_posterior_violation_bound(100, k, 5, 1.0 - 1e-3)
            self.assertAlmostEqual(value, expected, 11)
        # wj(200, 2, 0.05)
        self.assertAlmostEqual(
            scenario_posterior_violation_bound(200, 2, 5, 0.95),
            0.043452452136738984,
            11,
        )

    def test_posterior_bound_solves_the_polynomial_directly(self) -> None:
        """At ``t = 1 - eps(k)`` the published polynomial vanishes (exact arithmetic).

        References:
            [REF-CAMPI-GARATTI-2018]. Checks [EQ-REL-SCENARIO-003].

        """
        n, k, beta = 60, 3, 0.02
        eps = scenario_posterior_violation_bound(n, k, 6, 1.0 - beta)
        t = Fraction(1.0 - eps)
        lhs = (
            Fraction(beta)
            / (n + 1)
            * sum(math.comb(m, k) * t ** (m - k) for m in range(k, n + 1))
        )
        rhs = math.comb(n, k) * t ** (n - k)
        self.assertAlmostEqual(float(lhs / rhs), 1.0, 9)

    def test_posterior_bound_grows_with_support_constraints(self) -> None:
        """Fewer support constraints earn a smaller violation bound.

        References:
            [REF-CAMPI-GARATTI-2018]. Checks [EQ-REL-SCENARIO-003].

        """
        values = [
            scenario_posterior_violation_bound(500, k, 20, 0.99) for k in range(21)
        ]
        self.assertEqual(values, sorted(values))
        self.assertLess(values[0], values[-1])

    def test_scenario_validation(self) -> None:
        """Out-of-range or mistyped arguments raise.

        References:
            [REF-PROBSTAT-VALIDATION]. No mathematical equation is evaluated.

        """
        with self.assertRaises(ValueError):
            scenario_sample_size(0.0, 0.9, 2)
        with self.assertRaises(ValueError):
            scenario_sample_size(0.1, 1.0, 2)
        with self.assertRaises(TypeError):
            scenario_sample_size(0.1, 0.9, 2.0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            scenario_violation_bound(10, 0, 0.9)
        with self.assertRaises(ValueError):
            scenario_confidence(10, 2, 1.0)
        with self.assertRaises(ValueError):
            scenario_posterior_violation_bound(10, 3, 2, 0.9)
        with self.assertRaises(ValueError):
            scenario_posterior_violation_bound(10, 10, 12, 0.9)


def corner_program(rng: random.Random, n: int, mu: float, sigma: float) -> tuple:
    """Solve ``min x1 + x2`` s.t. ``a_i x >= 1`` by enumerating vertices.

    Returns the optimal vertex and the list of the two defining constraints.
    A vertex of the feasible polygon solves two active constraints exactly;
    the optimum of this bounded linear program is one of them.
    """
    rows = [(rng.gauss(mu, sigma), rng.gauss(mu, sigma)) for _ in range(n)]
    best: tuple[float, tuple[float, float], tuple[int, int]] | None = None
    for i, j in itertools.combinations(range(n), 2):
        (a, b), (c, d) = rows[i], rows[j]
        det = a * d - b * c
        if abs(det) < 1e-12:
            continue
        x1 = (d - b) / det
        x2 = (a - c) / det
        if all(r[0] * x1 + r[1] * x2 >= 1.0 - 1e-12 for r in rows):
            cost = x1 + x2
            if best is None or cost < best[0]:
                best = (cost, (x1, x2), (i, j))
    assert best is not None
    return best, rows


class ScenarioCoverageTests(unittest.TestCase):
    """The scenario guarantees against repeated linear programs with exact violation.

    References:
        [REF-CAMPI-GARATTI-2008], [REF-CAMPI-GARATTI-2018]. Checks
        [EQ-REL-SCENARIO-001], [EQ-REL-SCENARIO-003].

    """

    def test_violation_probability_follows_the_fully_supported_law(self) -> None:
        """``V`` of the vertex solution has mean ``d/(N+1)`` and obeys both bounds.

        The program ``min x1 + x2`` s.t. ``a_i x >= 1`` with Gaussian rows is
        fully supported (two support constraints), so ``V ~ Beta(2, N - 1)`` and
        the a-priori bound is attained: ``P(V > eps) = beta``. The exact
        ``V = Phi((1 - mu (x1 + x2)) / (sigma ||x||))`` is evaluated for each
        replication; the checks are on the binomial standard-error scale.

        References:
            [REF-CAMPI-GARATTI-2008, Theorem 2.4]. Checks [EQ-REL-SCENARIO-001].

        """
        n, d, mu, sigma, reps = 30, 2, 1.0, 0.3, 400
        beta = 0.2
        eps_prior = scenario_violation_bound(n, d, 1.0 - beta)
        eps_post = scenario_posterior_violation_bound(n, d, d, 1.0 - beta)
        rng = random.Random(20261124)
        violation: list[float] = []
        for _ in range(reps):
            (_, (x1, x2), _), _ = corner_program(rng, n, mu, sigma)
            spread = sigma * math.hypot(x1, x2)
            violation.append(NORMAL.cdf((1.0 - mu * (x1 + x2)) / spread))
        mean = sum(violation) / reps
        variance = d * (n - d + 1) / ((n + 1) ** 2 * (n + 2))  # Beta(2, n-1)
        self.assertLess(abs(mean - d / (n + 1)), 4 * math.sqrt(variance / reps))
        exceed = sum(v > eps_prior for v in violation) / reps
        se = math.sqrt(beta * (1 - beta) / reps)
        self.assertLess(abs(exceed - beta), 4 * se)  # tight for fully supported
        exceed_post = sum(v > eps_post for v in violation) / reps
        self.assertLessEqual(exceed_post, beta + 4 * se)

    def test_solver_reproduces_the_vertex_and_counts_support_constraints(self) -> None:
        """``solve_chance_constrained`` with no violations finds the LP vertex.

        The independent oracle is the enumerated optimal vertex of the same
        sampled linear program; the support constraints are its two defining
        rows, and the a-posteriori bound equals the published formula.

        References:
            [REF-CAMPI-GARATTI-2018]. Checks [EQ-REL-SAA-001],
            [EQ-REL-SCENARIO-003].

        """
        n, mu, sigma, seed = 40, 1.0, 0.3, 7
        result = solve_chance_constrained(
            lambda x: x[0] + x[1],
            lambda x, a: 1.0 - (a[0] * x[0] + a[1] * x[1]),
            lambda rng: (rng.gauss(mu, sigma), rng.gauss(mu, sigma)),
            (1.0, 1.0),
            violation_level=0.0,
            samples=n,
            validation_samples=2000,
            seed=seed,
            count_support_constraints=True,
        )
        from probstat.foundations.random_streams import resolve_stream

        rng = resolve_stream(seed).child("optimise").generator()
        (cost, vertex, _), _ = corner_program(rng, n, mu, sigma)
        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.design[0], vertex[0], 4)
        self.assertAlmostEqual(result.design[1], vertex[1], 4)
        self.assertAlmostEqual(result.objective, cost, 4)
        self.assertEqual(result.empirical_violation, 0.0)
        self.assertEqual(result.support_constraints, 2)
        assert result.posterior_bound is not None and result.scenario_bound is not None
        self.assertAlmostEqual(
            result.posterior_bound,
            scenario_posterior_violation_bound(n, 2, 2, 0.95),
            12,
        )
        self.assertAlmostEqual(
            result.scenario_bound, scenario_violation_bound(n, 2, 0.95), 12
        )
        # the exact violation of the returned design lies under the certificate
        spread = sigma * math.hypot(*result.design)
        exact = NORMAL.cdf((1.0 - mu * sum(result.design)) / spread)
        se = math.sqrt(exact * (1 - exact) / result.certificate.samples)
        self.assertLess(abs(result.certificate.frequency - exact), 4 * se + 1e-12)


class CertificationTests(unittest.TestCase):
    """Clopper-Pearson certificates of a violation probability.

    References:
        [REF-CLOPPER-PEARSON-1934]. Checks [EQ-INF-CI-CP-001].

    """

    def test_one_sided_bounds_match_qbeta(self) -> None:
        """Upper and lower limits equal the beta quantiles of R.

        References:
            [REF-CLOPPER-PEARSON-1934]. Checks [EQ-INF-CI-CP-001].

        """
        three = certify_violation_frequency(3, 100, confidence=0.95)
        # R: qbeta(0.95, 4, 97) -> 0.07571079374983003 ; qbeta(0.05, 3, 98)
        self.assertAlmostEqual(three.upper_bound, 0.07571079374983003, 12)
        self.assertAlmostEqual(three.lower_bound, 0.0082258291076526928, 12)
        self.assertEqual(three.frequency, 0.03)
        zero = certify_violation_frequency(0, 100, confidence=0.95)
        # R: qbeta(0.95, 1, 100) -> 0.029513049607039932 ("rule of three")
        self.assertAlmostEqual(zero.upper_bound, 0.029513049607039932, 12)
        self.assertEqual(zero.lower_bound, 0.0)

    def test_coverage_is_at_least_nominal_for_every_probability(self) -> None:
        """The upper bound covers the true probability with probability >= 0.95.

        The coverage is computed exactly from the binomial pmf for ``n = 40``.

        References:
            [REF-CLOPPER-PEARSON-1934]. Checks [EQ-INF-CI-CP-001].

        """
        n = 40
        bounds = [
            certify_violation_frequency(k, n, confidence=0.95).upper_bound
            for k in range(n + 1)
        ]
        for p in (0.01, 0.05, 0.1, 0.25, 0.5, 0.8):
            coverage = sum(
                math.comb(n, k) * p**k * (1 - p) ** (n - k)
                for k in range(n + 1)
                if bounds[k] >= p
            )
            self.assertGreaterEqual(coverage, 0.95 - 1e-12)

    def test_certify_design_against_a_known_probability(self) -> None:
        """A seeded validation sample brackets the exact violation probability.

        The constraint ``xi - x`` with standard normal ``xi`` is violated with
        probability ``1 - Phi(x)``; the frequency is judged on its standard
        error and the bounds at the 0.9999 level.

        References:
            [REF-CLOPPER-PEARSON-1934]. Checks [EQ-INF-CI-CP-001].

        """
        x = 1.0
        exact = 1.0 - NORMAL.cdf(x)
        n = 20000
        cert = certify_design(
            (x,),
            lambda d, xi: xi - d[0],
            lambda rng: rng.gauss(0.0, 1.0),
            n,
            confidence=0.9999,
            seed=3,
        )
        se = math.sqrt(exact * (1 - exact) / n)
        self.assertLess(abs(cert.frequency - exact), 4 * se)
        self.assertLessEqual(cert.lower_bound, exact)
        self.assertGreaterEqual(cert.upper_bound, exact)
        again = certify_design(
            (x,),
            lambda d, xi: xi - d[0],
            lambda r: r.gauss(0.0, 1.0),
            n,
            confidence=0.9999,
            seed=3,
        )
        self.assertEqual(again.violations, cert.violations)

    def test_joint_constraint_uses_the_worst_row(self) -> None:
        """A vector constraint is violated when any component is positive.

        References:
            [REF-CLOPPER-PEARSON-1934]. Checks [EQ-INF-CI-CP-001].

        """
        x = 1.0
        exact = 1.0 - NORMAL.cdf(x) ** 2
        n = 20000
        cert = certify_design(
            (x,),
            lambda d, xi: (xi[0] - d[0], xi[1] - d[0]),
            lambda rng: (rng.gauss(0, 1), rng.gauss(0, 1)),
            n,
            seed=11,
        )
        self.assertLess(
            abs(cert.frequency - exact), 4 * math.sqrt(exact * (1 - exact) / n)
        )

    def test_certification_validation(self) -> None:
        """Invalid counts, confidences, and callables raise.

        References:
            [REF-PROBSTAT-VALIDATION]. No mathematical equation is evaluated.

        """
        with self.assertRaises(ValueError):
            certify_violation_frequency(5, 4)
        with self.assertRaises(ValueError):
            certify_violation_frequency(1, 4, confidence=0.5)
        with self.assertRaises(ValueError):
            certify_violation_frequency(1, 0)
        with self.assertRaises(TypeError):
            certify_violation_frequency(1.0, 4)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            certify_design((1.0,), 3, lambda r: 0.0, 10)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            certify_design((1.0,), lambda d, s: [], lambda r: 0.0, 10)


class DeterministicEquivalentTests(unittest.TestCase):
    """Closed-form chance constraints against exact probabilities and sampling.

    References:
        [REF-NEMIROVSKI-SHAPIRO-2006]. Checks [EQ-REL-CHANCE-NORMAL-001] and
        [EQ-REL-CHANCE-FREE-001].

    """

    MEAN = (1.0, 2.0)
    COV = ((0.25, 0.1), (0.1, 0.5))
    X = (1.5, 0.7)

    def moments(self) -> tuple[float, float]:
        """Return the mean and standard deviation of ``a'x``."""
        mean = sum(m * v for m, v in zip(self.MEAN, self.X, strict=True))
        var = sum(
            self.X[i] * self.COV[i][j] * self.X[j] for i in range(2) for j in range(2)
        )
        return mean, math.sqrt(var)

    def test_zero_margin_is_the_exact_violation_level(self) -> None:
        """At zero margin the exact violation probability equals ``eps``.

        References:
            [EQ-REL-CHANCE-NORMAL-001].

        """
        mean, std = self.moments()
        for eps in (0.01, 0.05, 0.3, 0.5, 0.7):
            bound = mean + NORMAL.inv_cdf(1 - eps) * std
            margin = normal_linear_chance_margin(
                self.X, self.MEAN, self.COV, bound, eps
            )
            self.assertAlmostEqual(margin, 0.0, 9)
            p = normal_linear_violation_probability(self.X, self.MEAN, self.COV, bound)
            self.assertAlmostEqual(p, eps, 9)
            looser = normal_linear_chance_margin(
                self.X, self.MEAN, self.COV, bound + 0.1, eps
            )
            self.assertGreater(looser, 0.0)

    def test_second_order_cone_margin_is_concave_for_small_eps(self) -> None:
        """The margin is concave in ``x`` when ``eps < 1/2`` (convex feasible set).

        References:
            [EQ-REL-CHANCE-NORMAL-001].

        """
        rng = random.Random(5)
        for _ in range(50):
            a = (rng.uniform(-2, 2), rng.uniform(-2, 2))
            b = (rng.uniform(-2, 2), rng.uniform(-2, 2))
            mid = tuple(0.5 * (u + v) for u, v in zip(a, b, strict=True))

            def f(p: tuple[float, ...]) -> float:
                """Return the margin at ``p`` for a fixed bound and level."""
                return normal_linear_chance_margin(p, self.MEAN, self.COV, 1.0, 0.1)

            self.assertGreaterEqual(f(mid), 0.5 * (f(a) + f(b)) - 1e-12)

    def test_monte_carlo_matches_the_normal_equivalent(self) -> None:
        """Sampled coefficients violate at the exact normal rate.

        Coefficients are drawn through the Cholesky factor of the covariance;
        the frequency is judged on its binomial standard error.

        References:
            [EQ-REL-CHANCE-NORMAL-001].

        """
        l11 = math.sqrt(self.COV[0][0])
        l21 = self.COV[1][0] / l11
        l22 = math.sqrt(self.COV[1][1] - l21**2)
        mean, std = self.moments()
        eps = 0.1
        bound = mean + NORMAL.inv_cdf(1 - eps) * std
        rng = random.Random(8)
        n, hits = 40000, 0
        for _ in range(n):
            z1, z2 = rng.gauss(0, 1), rng.gauss(0, 1)
            a1 = self.MEAN[0] + l11 * z1
            a2 = self.MEAN[1] + l21 * z1 + l22 * z2
            hits += a1 * self.X[0] + a2 * self.X[1] > bound
        self.assertLess(abs(hits / n - eps), 4 * math.sqrt(eps * (1 - eps) / n))

    def test_cantelli_is_attained_by_a_two_point_law(self) -> None:
        """The Cantelli margin is tight; Chebyshev is stricter; normal is looser.

        A two-point law with the given mean and variance puts mass exactly
        ``eps`` on ``m + kappa s`` (``kappa = sqrt((1 - eps)/eps)``).

        References:
            [EQ-REL-CHANCE-FREE-001].

        """
        mean, std = self.moments()
        for eps in (0.02, 0.1, 0.4):
            kappa = math.sqrt((1 - eps) / eps)
            high, low = mean + kappa * std, mean - std / kappa
            self.assertAlmostEqual(eps * high + (1 - eps) * low, mean, 12)
            variance = eps * (high - mean) ** 2 + (1 - eps) * (low - mean) ** 2
            self.assertAlmostEqual(variance, std**2, 12)
            bound = high
            margin = distribution_free_chance_margin(
                self.X, self.MEAN, self.COV, bound, eps
            )
            self.assertAlmostEqual(margin, 0.0, 9)
            # P(Y >= bound) = eps exactly for the extremal law, P(Y > bound) = 0
            # but every smaller bound is violated with probability eps: tight.
            cheb = distribution_free_chance_margin(
                self.X, self.MEAN, self.COV, bound, eps, inequality="chebyshev"
            )
            normal = normal_linear_chance_margin(
                self.X, self.MEAN, self.COV, bound, eps
            )
            self.assertLess(cheb, margin)
            self.assertGreater(normal, margin)

    def test_distribution_free_guarantee_holds_for_other_laws(self) -> None:
        """A zero Cantelli margin bounds the tail of a uniform law with equal moments.

        References:
            [EQ-REL-CHANCE-FREE-001].

        """
        mean, std = self.moments()
        half = std * math.sqrt(3.0)
        for eps in (0.05, 0.2):
            bound = mean + math.sqrt((1 - eps) / eps) * std
            tail = max(0.0, (mean + half - bound) / (2 * half))
            self.assertLessEqual(tail, eps)

    def test_equivalent_validation(self) -> None:
        """Bad shapes, covariances, levels, and inequality names raise.

        References:
            [REF-PROBSTAT-VALIDATION]. No mathematical equation is evaluated.

        """
        with self.assertRaises(ValueError):
            normal_linear_chance_margin((1.0,), (1.0, 2.0), self.COV, 1.0, 0.1)
        with self.assertRaises(ValueError):
            normal_linear_chance_margin(
                self.X, self.MEAN, ((1.0, 0.5), (0.0, 1.0)), 1.0, 0.1
            )
        with self.assertRaises(ValueError):
            normal_linear_chance_margin(
                self.X, self.MEAN, ((1.0, 2.0), (2.0, 1.0)), 1.0, 0.1
            )
        with self.assertRaises(ValueError):
            normal_linear_chance_margin(self.X, self.MEAN, self.COV, 1.0, 1.0)
        with self.assertRaises(ValueError):
            distribution_free_chance_margin(
                self.X, self.MEAN, self.COV, 1.0, 0.1, inequality="markov"
            )
        zero = ((0.0, 0.0), (0.0, 0.0))
        self.assertEqual(
            normal_linear_violation_probability(self.X, self.MEAN, zero, 1.0), 1.0
        )
        self.assertEqual(
            normal_linear_violation_probability(self.X, self.MEAN, zero, 9.0), 0.0
        )


class SampleAverageTests(unittest.TestCase):
    """Sample-average chance-constrained programs against closed forms.

    References:
        [REF-NEMIROVSKI-SHAPIRO-2006], [REF-LUEDTKE-AHMED-2008],
        [REF-CLOPPER-PEARSON-1934]. Checks [EQ-REL-SAA-001] and
        [EQ-REL-CHANCE-NORMAL-001].

    """

    MU, SIGMA, EPS = 1.0, 0.3, 0.1

    def test_normal_coefficients_against_the_cone_solution(self) -> None:
        """``min x1 + x2`` s.t. ``P(a'x >= 1) >= 0.9`` is close to the cone optimum.

        With ``a ~ N(mu 1, sigma^2 I)`` the symmetric point of the second-order
        cone program is optimal: ``t = 1 / (2 mu - z sigma sqrt(2))`` and the
        objective is ``2 t``. The sampled program (2000 draws, 10 % violations
        allowed) is a biased estimate of it; the check is that the objective is
        within 0.08 of the cone value, that the sample constraint is active,
        and that the certificate agrees with the exact violation probability
        of the returned design.

        References:
            [REF-NEMIROVSKI-SHAPIRO-2006]. Checks [EQ-REL-SAA-001],
            [EQ-REL-CHANCE-NORMAL-001].

        """
        z = NORMAL.inv_cdf(1 - self.EPS)
        t = 1.0 / (2 * self.MU - z * self.SIGMA * math.sqrt(2.0))
        result = solve_chance_constrained(
            lambda x: x[0] + x[1],
            lambda x, a: 1.0 - (a[0] * x[0] + a[1] * x[1]),
            lambda rng: (
                rng.gauss(self.MU, self.SIGMA),
                rng.gauss(self.MU, self.SIGMA),
            ),
            (1.0, 1.0),
            violation_level=self.EPS,
            samples=2000,
            validation_samples=10000,
            confidence=0.999,
            seed=1,
        )
        self.assertTrue(result.converged)
        self.assertLess(abs(result.objective - 2 * t), 0.08)
        self.assertEqual(result.allowed_violations, 200)
        self.assertAlmostEqual(result.empirical_violation, self.EPS, 12)
        self.assertIsNone(result.scenario_bound)
        mean_vec = (-self.MU, -self.MU)
        cov = ((self.SIGMA**2, 0.0), (0.0, self.SIGMA**2))
        exact = normal_linear_violation_probability(result.design, mean_vec, cov, -1.0)
        n = result.certificate.samples
        self.assertLess(
            abs(result.certificate.frequency - exact),
            4 * math.sqrt(exact * (1 - exact) / n),
        )
        self.assertGreaterEqual(result.certificate.upper_bound, exact)
        # the cone optimum itself violates at exactly eps
        cone = normal_linear_violation_probability((t, t), mean_vec, cov, -1.0)
        self.assertAlmostEqual(cone, self.EPS, 9)

    def test_joint_chance_constraint_against_the_product_form(self) -> None:
        """``min x`` s.t. ``P(xi1 <= x, xi2 <= x) >= 1 - eps``: ``Phi(x)^2 = 1 - eps``.

        The sampled solution is the order statistic of ``max(xi1, xi2)``; its
        error scale is ``sqrt(eps (1 - eps) / N) / f`` with ``f`` the density
        of the maximum at the optimum.

        References:
            [REF-LUEDTKE-AHMED-2008]. Checks [EQ-REL-SAA-001].

        """
        eps, n = 0.1, 4000
        exact = NORMAL.inv_cdf(math.sqrt(1 - eps))
        q = math.sqrt(1 - eps)
        density = 2 * q * NORMAL.pdf(exact)  # d/dx Phi(x)^2
        se = math.sqrt(eps * (1 - eps) / n) / density
        result = solve_chance_constrained(
            lambda x: x[0],
            lambda x, xi: (xi[0] - x[0], xi[1] - x[0]),
            lambda rng: (rng.gauss(0, 1), rng.gauss(0, 1)),
            (0.0,),
            violation_level=eps,
            samples=n,
            validation_samples=5000,
            seed=4,
        )
        self.assertTrue(result.converged)
        self.assertLess(abs(result.design[0] - exact), 4 * se)
        truth = 1 - NORMAL.cdf(result.design[0]) ** 2
        self.assertLess(
            abs(result.certificate.frequency - truth),
            4 * math.sqrt(truth * (1 - truth) / 5000),
        )

    def test_same_seed_same_answer(self) -> None:
        """Two runs with one seed return the same design and certificate.

        References:
            [REF-STEELE-LEA-FLOOD-2014]. No mathematical equation is evaluated.

        """
        kwargs = dict(violation_level=0.2, samples=200, validation_samples=500, seed=9)
        runs = [
            solve_chance_constrained(
                lambda x: x[0],
                lambda x, xi: xi - x[0],
                lambda rng: rng.gauss(0, 1),
                (0.0,),
                **kwargs,  # type: ignore[arg-type]
            )
            for _ in range(2)
        ]
        self.assertEqual(runs[0].design, runs[1].design)
        self.assertEqual(runs[0].certificate, runs[1].certificate)

    def test_solver_validation(self) -> None:
        """Invalid arguments raise.

        References:
            [REF-PROBSTAT-VALIDATION]. No mathematical equation is evaluated.

        """

        def sampler(rng: random.Random) -> float:
            """Draw one standard normal."""
            return rng.gauss(0, 1)

        def constraint(x: tuple[float, ...], xi: object) -> float:
            """Return a trivial margin."""
            return 0.0

        with self.assertRaises(ValueError):
            solve_chance_constrained(
                lambda x: x[0], constraint, sampler, (0.0,), violation_level=1.0
            )
        with self.assertRaises(ValueError):
            solve_chance_constrained(
                lambda x: x[0], constraint, sampler, (0.0,), samples=1
            )
        with self.assertRaises(TypeError):
            solve_chance_constrained(lambda x: x[0], 1.0, sampler, (0.0,))  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            solve_chance_constrained(
                lambda x: x[0], constraint, sampler, (0.0,), tolerance=0.0
            )


class ConstrainedMinimisationTests(unittest.TestCase):
    """The augmented-Lagrangian solver on problems with known solutions.

    References:
        [REF-NOCEDAL-WRIGHT-2006]. Checks [EQ-REL-AUGLAG-001].

    """

    def test_active_linear_constraint_and_multiplier(self) -> None:
        """``min x^2 + y^2`` s.t. ``x + y >= 1`` gives ``(1/2, 1/2)`` with multiplier 1.

        The KKT condition ``2 x = lambda`` fixes the multiplier.

        References:
            [REF-NOCEDAL-WRIGHT-2006, Section 17.4]. Checks [EQ-REL-AUGLAG-001].

        """
        result = minimize_constrained(
            lambda p: p[0] ** 2 + p[1] ** 2, lambda p: (1.0 - p[0] - p[1],), (3.0, -2.0)
        )
        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.x[0], 0.5, 5)
        self.assertAlmostEqual(result.x[1], 0.5, 5)
        self.assertAlmostEqual(result.multipliers[0], 1.0, 4)
        self.assertLessEqual(result.max_violation, 1e-7)

    def test_inactive_constraint_has_zero_multiplier(self) -> None:
        """An inactive constraint leaves the unconstrained minimiser in place.

        References:
            [REF-NOCEDAL-WRIGHT-2006, Section 17.4]. Checks [EQ-REL-AUGLAG-001].

        """
        result = minimize_constrained(
            lambda p: (p[0] - 1) ** 2 + (p[1] - 2) ** 2,
            lambda p: (p[0] + p[1] - 10.0,),
            (0.0, 0.0),
        )
        self.assertAlmostEqual(result.x[0], 1.0, 5)
        self.assertAlmostEqual(result.x[1], 2.0, 5)
        self.assertAlmostEqual(result.multipliers[0], 0.0, 8)

    def test_disc_and_box_corner(self) -> None:
        """Nearest point to ``(2, 2)`` in a disc with ``y <= 0.8`` is the corner.

        Without the box the answer is ``(1, 1)``; with ``y <= 0.8`` it is
        ``(sqrt(2 - 0.64), 0.8)``.

        References:
            [REF-NOCEDAL-WRIGHT-2006, Section 17.4]. Checks [EQ-REL-AUGLAG-001].

        """

        def objective(p: tuple[float, ...]) -> float:
            """Return the squared distance to ``(2, 2)``."""
            return (p[0] - 2) ** 2 + (p[1] - 2) ** 2

        def disc(p: tuple[float, ...]) -> tuple[float, ...]:
            """Return the disc constraint ``x^2 + y^2 - 2 <= 0``."""
            return (p[0] ** 2 + p[1] ** 2 - 2.0,)

        free = minimize_constrained(objective, disc, (0.0, 0.0))
        self.assertAlmostEqual(free.x[0], 1.0, 5)
        self.assertAlmostEqual(free.x[1], 1.0, 5)
        boxed = minimize_constrained(
            objective, disc, (0.0, 0.0), bounds=[(None, None), (None, 0.8)]
        )
        self.assertAlmostEqual(boxed.x[0], math.sqrt(2.0 - 0.64), 5)
        self.assertAlmostEqual(boxed.x[1], 0.8, 6)

    def test_contradictory_constraints_are_flagged(self) -> None:
        """Infeasible constraints end with ``converged=False`` and a reason.

        References:
            [REF-NOCEDAL-WRIGHT-2006, Section 17.4]. Checks [EQ-REL-AUGLAG-001].

        """
        result = minimize_constrained(
            lambda p: p[0] ** 2,
            lambda p: (p[0] + 1.0, 1.0 - p[0]),
            (0.0,),
            max_outer_iterations=5,
        )
        self.assertFalse(result.converged)
        self.assertIn("limit", result.message)
        self.assertGreater(result.max_violation, 0.5)

    def test_validation(self) -> None:
        """Invalid arguments and a changing or non-finite constraint raise.

        References:
            [REF-PROBSTAT-VALIDATION]. No mathematical equation is evaluated.

        """

        def f(p: tuple[float, ...]) -> float:
            """Return ``x^2``."""
            return p[0] ** 2

        with self.assertRaises(TypeError):
            minimize_constrained(f, None, (0.0,))  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            minimize_constrained(f, lambda p: (p[0],), (0.0,), penalty_growth=1.0)
        with self.assertRaises(ValueError):
            minimize_constrained(f, lambda p: (), (0.0,))
        with self.assertRaises(ValueError):
            minimize_constrained(f, lambda p: (math.nan,), (0.0,))
        with self.assertRaises(ValueError):
            minimize_constrained(f, lambda p: (p[0],), ())
        with self.assertRaises(TypeError):
            minimize_constrained(f, lambda p: (p[0],), (0.0,), max_outer_iterations=1.5)  # type: ignore[arg-type]


def linear_factory(coefficients: tuple[float, float], threshold: float, sd: float):
    """Return ``design -> limit state`` of ``y = c1 x1 + c2 x2``, ``x_i ~ N(d_i)``."""

    def factory(design: tuple[float, ...]) -> ModelLimitState:
        """Build the limit state at a design (means of the two inputs)."""
        inputs = JointInput.independent(
            RandomInput("x1", Normal(design[0], sd)),
            RandomInput("x2", Normal(design[1], sd)),
        )
        model = ProbabilisticModel(
            inputs,
            lambda p: {"y": coefficients[0] * p["x1"] + coefficients[1] * p["x2"]},
            Y,
        )
        return ModelLimitState(model, "y", threshold, "below")

    return factory


def first_cost(design: tuple[float, ...]) -> float:
    """Return the first design variable as the cost."""
    return design[0]


def sum_cost(design: tuple[float, ...]) -> float:
    """Return the sum of the first two design variables as the cost."""
    return design[0] + design[1]


def capacity_factory(design: tuple[float, ...]) -> ModelLimitState:
    """Return ``y = c x1 - x2`` with the deterministic design variable ``c``."""
    inputs = JointInput.independent(
        RandomInput("x1", Normal(10.0, 1.0)), RandomInput("x2", Normal(30.0, 5.0))
    )
    model = ProbabilisticModel(
        inputs, lambda p: {"y": p["c"] * p["x1"] - p["x2"]}, Y, fixed={"c": design[0]}
    )
    return ModelLimitState(model, "y", 0.0, "below")


DU_CHEN = (
    lambda a, b: a * a * b / 20.0 - 1.0,
    lambda a, b: (a + b - 5.0) ** 2 / 30.0 + (a - b - 12.0) ** 2 / 120.0 - 1.0,
    lambda a, b: 80.0 / (a * a + 8.0 * b + 5.0) - 1.0,
)


def nonlinear_factory(index: int):
    """Return the factory of limit state ``index`` of the three-constraint benchmark."""
    g = DU_CHEN[index]

    def factory(design: tuple[float, ...]) -> ModelLimitState:
        """Build the limit state with normal inputs of mean ``design`` and sd 0.3."""
        inputs = JointInput.independent(
            RandomInput("x1", Normal(design[0], 0.3)),
            RandomInput("x2", Normal(design[1], 0.3)),
        )
        model = ProbabilisticModel(inputs, lambda p: {"y": g(p["x1"], p["x2"])}, Y)
        return ModelLimitState(model, "y", 0.0, "below")

    return factory


class LinearRBDOTests(unittest.TestCase):
    """SORA and the double loop on linear normal limit states with closed forms.

    References:
        [REF-DU-CHEN-2004], [REF-TU-CHOI-PARK-1999]. Checks [EQ-REL-SORA-001]
        and [EQ-REL-RBDO-002].

    """

    SD, BETA = 0.3, 3.0

    def test_two_linear_constraints_have_a_symmetric_optimum(self) -> None:
        """``d1 + 2 d2 >= a`` and ``2 d1 + d2 >= a`` give ``d1 = d2 = a / 3``.

        Here ``a = 6 + beta sd sqrt(5)`` because each index is
        ``(c.d - 6) / (sd ||c||)`` with ``||c|| = sqrt(5)``; the optimal cost
        of ``min d1 + d2`` is ``2 a / 3``. Both methods must reach it and report
        FORM indices equal to the target.

        References:
            [REF-DU-CHEN-2004]. Checks [EQ-REL-SORA-001], [EQ-REL-RBDO-002].

        """
        a = 6.0 + self.BETA * self.SD * math.sqrt(5.0)
        factories = [
            linear_factory((1.0, 2.0), 6.0, self.SD),
            linear_factory((2.0, 1.0), 6.0, self.SD),
        ]
        bounds = [(0.0, 10.0), (0.0, 10.0)]
        for solver in (rbdo_sora, rbdo_double_loop):
            result = solver(
                lambda d: d[0] + d[1], factories, self.BETA, (3.0, 3.0), bounds=bounds
            )
            self.assertTrue(result.converged, result.message)
            self.assertAlmostEqual(result.cost, 2 * a / 3, 5)
            self.assertAlmostEqual(result.design[0], a / 3, 5)
            self.assertAlmostEqual(result.design[1], a / 3, 5)
            for beta in result.betas:
                self.assertAlmostEqual(beta, self.BETA, 5)
            self.assertGreater(result.evaluations, 0)

    def test_active_design_bound(self) -> None:
        """One constraint with cost ``d1 + d2``: the cheaper ``d2`` carries it all.

        ``d1 + 2 d2 >= a`` is cheapest with ``d1 = 0`` (bound active) and
        ``d2 = a / 2``; the cost is ``a / 2``.

        References:
            [REF-DU-CHEN-2004]. Checks [EQ-REL-SORA-001].

        """
        a = 6.0 + self.BETA * self.SD * math.sqrt(5.0)
        result = rbdo_sora(
            lambda d: d[0] + d[1],
            [linear_factory((1.0, 2.0), 6.0, self.SD)],
            self.BETA,
            (3.0, 3.0),
            bounds=[(0.0, 10.0), (0.0, 10.0)],
        )
        self.assertTrue(result.converged, result.message)
        self.assertAlmostEqual(result.design[0], 0.0, 5)
        self.assertAlmostEqual(result.cost, a / 2, 5)

    def test_deterministic_design_variable_needs_several_cycles(self) -> None:
        """A fixed design parameter ``c`` in ``y = c x1 - x2`` has a closed form.

        ``beta(c) = (10 c - 30) / sqrt(c^2 + 25)`` and ``beta = 3`` gives
        ``91 c^2 - 600 c + 675 = 0``; the larger root is the cheapest feasible
        ``c``. The shift depends on ``c``, so SORA iterates.

        References:
            [REF-DU-CHEN-2004]. Checks [EQ-REL-SORA-001].

        """
        expected = (600.0 + math.sqrt(600.0**2 - 4 * 91 * 675)) / 182.0
        sora = rbdo_sora(
            lambda d: d[0], [capacity_factory], 3.0, (8.0,), bounds=[(1.0, 20.0)]
        )
        self.assertTrue(sora.converged, sora.message)
        self.assertAlmostEqual(sora.design[0], expected, 6)
        self.assertGreater(len(sora.history), 1)
        loop = rbdo_double_loop(
            lambda d: d[0], [capacity_factory], 3.0, (8.0,), bounds=[(1.0, 20.0)]
        )
        self.assertAlmostEqual(loop.design[0], expected, 6)
        self.assertLess(sora.evaluations, loop.evaluations)

    def test_validation(self) -> None:
        """Invalid arguments raise.

        References:
            [REF-PROBSTAT-VALIDATION]. No mathematical equation is evaluated.

        """
        factory = linear_factory((1.0, 2.0), 6.0, self.SD)
        cost = first_cost
        with self.assertRaises(ValueError):
            rbdo_sora(cost, [], 3.0, (3.0, 3.0))
        with self.assertRaises(ValueError):
            rbdo_sora(cost, [factory], 0.0, (3.0, 3.0))
        with self.assertRaises(ValueError):
            rbdo_sora(cost, [factory], (3.0, 3.0), (3.0, 3.0))
        with self.assertRaises(TypeError):
            rbdo_sora(cost, [3], 3.0, (3.0, 3.0))  # type: ignore[list-item]
        with self.assertRaises(TypeError):
            rbdo_double_loop(None, [factory], 3.0, (3.0, 3.0))  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            rbdo_sora(cost, [factory], 3.0, (3.0, 3.0), max_cycles=0)

    def test_cycle_limit_is_reported(self) -> None:
        """One cycle on a design-dependent shift cannot converge and says why.

        References:
            [REF-DU-CHEN-2004]. Checks [EQ-REL-SORA-001].

        """
        result = rbdo_sora(
            lambda d: d[0],
            [capacity_factory],
            3.0,
            (8.0,),
            bounds=[(1.0, 20.0)],
            max_cycles=1,
        )
        self.assertFalse(result.converged)
        self.assertEqual(result.message, "cycle limit reached")
        self.assertEqual(len(result.history), 1)


class NonlinearRBDOTests(unittest.TestCase):
    """SORA against the double loop on a three-constraint nonlinear benchmark.

    References:
        [REF-DU-CHEN-2004], [REF-AOUES-CHATEAUNEUF-2010]. Checks
        [EQ-REL-SORA-001], [EQ-REL-RBDO-002].

    """

    sora = None
    loop = None
    factories: list = []

    @classmethod
    def setUpClass(cls) -> None:
        """Run both methods once from the same start.

        References:
            [REF-DU-CHEN-2004].

        """
        cls.factories = [nonlinear_factory(i) for i in range(3)]
        cost = sum_cost
        bounds = [(0.0, 10.0), (0.0, 10.0)]
        cls.sora = rbdo_sora(cost, cls.factories, 3.0, (5.0, 5.0), bounds=bounds)
        cls.loop = rbdo_double_loop(cost, cls.factories, 3.0, (5.0, 5.0), bounds=bounds)

    def test_methods_agree_and_meet_the_targets(self) -> None:
        """Both designs coincide, every FORM index reaches 3, and SORA is cheaper.

        References:
            [REF-DU-CHEN-2004]. Checks [EQ-REL-SORA-001], [EQ-REL-RBDO-002].

        """
        sora, loop = self.sora, self.loop
        assert sora is not None and loop is not None
        self.assertTrue(sora.converged, sora.message)
        self.assertTrue(loop.converged, loop.message)
        self.assertAlmostEqual(sora.cost, loop.cost, 4)
        for a, b in zip(sora.design, loop.design, strict=True):
            self.assertAlmostEqual(a, b, 4)
        for beta in sora.betas[:2]:
            self.assertAlmostEqual(beta, 3.0, 4)  # two active constraints
        self.assertGreater(sora.betas[2], 3.0)  # the third is inactive
        self.assertLess(sora.evaluations * 3, loop.evaluations)
        self.assertGreater(len(sora.history), 1)
        self.assertEqual(sora.history[-1].design, sora.design)
        self.assertGreaterEqual(min(sora.history[-1].mptp_margins), -1e-5)

    def test_verification_with_sorm_and_importance_sampling(self) -> None:
        """Independent checks stay near FORM; the index differences are reported.

        SORM and design-point importance sampling estimate the same failure
        probability, so they agree with each other on the standard-error scale
        of the sampling estimate, and both differ from FORM by the curvature
        effect (here at most 0.1 in the index).

        References:
            [REF-DERKIUREGHIAN-2005]. Checks [EQ-REL-BETA-001].

        """
        sora = self.sora
        assert sora is not None
        second = verify_design_reliability(
            sora.design, self.factories, 3.0, method="sorm"
        )
        sampled = verify_design_reliability(
            sora.design, self.factories, 3.0, method="importance", samples=3000, seed=2
        )
        for row, check in zip(second, sampled, strict=True):
            self.assertLess(abs(row.beta_difference), 0.1)
            self.assertAlmostEqual(row.form_probability, NORMAL.cdf(-row.form_beta), 12)
            self.assertEqual(row.standard_error, 0.0)
            tolerance = 4 * check.standard_error + 1e-12
            self.assertLessEqual(
                abs(check.check_probability - row.check_probability),
                tolerance + 0.05 * row.check_probability,
            )
        self.assertEqual(second[0].check_method, "sorm")

    def test_verification_of_a_linear_design_is_exact(self) -> None:
        """For a linear limit state SORM equals FORM and sampling covers ``Phi(-beta)``.

        References:
            [REF-DERKIUREGHIAN-2005]. Checks [EQ-REL-BETA-001].

        """
        factory = linear_factory((1.0, 2.0), 6.0, 0.3)
        design = (2.5, 3.0)
        beta = (design[0] + 2 * design[1] - 6.0) / (0.3 * math.sqrt(5.0))
        (row,) = verify_design_reliability(design, [factory], 2.0, method="sorm")
        self.assertAlmostEqual(row.form_beta, beta, 6)
        self.assertAlmostEqual(row.beta_difference, 0.0, 6)
        self.assertTrue(row.meets_target)
        (sampled,) = verify_design_reliability(
            design, [factory], 2.0, method="importance", samples=4000, seed=5
        )
        exact = NORMAL.cdf(-beta)
        self.assertLess(
            abs(sampled.check_probability - exact), 4 * sampled.standard_error
        )

    def test_verification_validation(self) -> None:
        """Unknown methods and bad targets raise.

        References:
            [REF-PROBSTAT-VALIDATION]. No mathematical equation is evaluated.

        """
        factory = linear_factory((1.0, 2.0), 6.0, 0.3)
        with self.assertRaises(ValueError):
            verify_design_reliability((2.0, 2.5), [factory], 3.0, method="bogus")
        with self.assertRaises(ValueError):
            verify_design_reliability((2.0, 2.5), [factory], -1.0)


if __name__ == "__main__":
    unittest.main()
