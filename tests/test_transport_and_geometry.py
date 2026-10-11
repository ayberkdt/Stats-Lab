"""Tests for optimal transport and information geometry in ``probstat.information``.

Exact transport is compared with brute force over permutations and checked
through its complementary-slackness certificate; Sinkhorn is checked for
marginal feasibility and for convergence to the exact cost as ``eps -> 0``;
closed-form Gaussian and 1-D quantities are compared with independent Simpson
integrals; closed-form Fisher information matrices are compared with the
numerical score and Hessian expectations; Fisher-Rao distances are checked
for symmetry, the triangle inequality, and their small-distance metric limit.
"""

from __future__ import annotations

import itertools
import math
import random
import unittest
from collections.abc import Callable, Sequence

from probstat.distributions import (
    Bernoulli,
    Beta,
    Binomial,
    Exponential,
    Gamma,
    Normal,
    Poisson,
)
from probstat.information import (
    alpha_divergence,
    alpha_divergence_gaussian,
    bhattacharyya_coefficient,
    bhattacharyya_distance,
    bhattacharyya_distance_gaussian,
    bures_distance,
    categorical_fisher_information,
    discrete_wasserstein,
    exact_transport,
    fisher_information,
    fisher_rao_bernoulli,
    fisher_rao_binomial,
    fisher_rao_categorical,
    fisher_rao_exponential,
    fisher_rao_normal,
    fisher_rao_normal_mean,
    fisher_rao_poisson,
    gaussian_barycenter,
    gaussian_wasserstein,
    hellinger_distance,
    hellinger_distance_gaussian,
    kl_divergence,
    kl_multivariate_normal,
    multivariate_normal_mean_fisher_information,
    normal_fisher_information,
    normal_wasserstein,
    numerical_fisher_information,
    power_cost_matrix,
    quantile_barycenter,
    quantile_wasserstein,
    renyi_divergence,
    renyi_divergence_gaussian,
    sinkhorn,
    sinkhorn_divergence,
    squared_euclidean_cost_matrix,
    total_variation_distance,
    total_variation_gaussian_shared_covariance,
    total_variation_normal,
    transport_certificate,
)
from probstat.linalg import matmul
from probstat.numerics import composite_simpson

SEED = 20261012


def _normal_pdf(x: float, mu: float, sigma: float) -> float:
    """Return the univariate normal density."""
    z = (x - mu) / sigma
    return math.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi))


def _integral(function: Callable[[float], float], lower: float, upper: float) -> float:
    """Return a fine Simpson integral used as an independent reference."""
    return composite_simpson(function, lower, upper, intervals=8000).value


def _random_pmf(rng: random.Random, size: int) -> tuple[float, ...]:
    """Return a random strictly positive PMF."""
    raw = [rng.random() + 0.05 for _ in range(size)]
    total = math.fsum(raw)
    return tuple(value / total for value in raw)


class ExactTransportTests(unittest.TestCase):
    """The transportation simplex returns certified optimal plans."""

    def test_assignment_matches_brute_force_over_permutations(self) -> None:
        """Uniform weights give the optimal permutation cost (Birkhoff)."""
        rng = random.Random(SEED)
        worst = 0.0
        for trial in range(60):
            n = 2 + trial % 5
            integer_costs = trial % 3 == 0
            matrix = [
                [
                    float(rng.randint(0, 3)) if integer_costs else rng.random()
                    for _ in range(n)
                ]
                for _ in range(n)
            ]
            weights = [1.0 / n] * n
            plan = exact_transport(weights, weights, matrix)
            brute = (
                min(
                    math.fsum(matrix[i][perm[i]] for i in range(n))
                    for perm in itertools.permutations(range(n))
                )
                / n
            )
            worst = max(worst, abs(plan.cost - brute))
            self.assertTrue(plan.certificate().optimal)
        self.assertLess(worst, 1e-12)

    def test_complementary_slackness_and_zero_duality_gap(self) -> None:
        """Potentials are dual feasible and tight wherever the plan is positive."""
        rng = random.Random(SEED + 1)
        for _ in range(40):
            m, n = rng.randint(1, 7), rng.randint(1, 7)
            a = list(_random_pmf(rng, m))
            b = [rng.random() if rng.random() > 0.25 else 0.0 for _ in range(n)]
            b[0] += 0.1
            b = [value / math.fsum(b) for value in b]
            cost = [[rng.uniform(0.0, 5.0) for _ in range(n)] for _ in range(m)]
            result = exact_transport(a, b, cost)
            u, v = result.source_potential, result.target_potential
            for i in range(m):
                for j in range(n):
                    reduced = cost[i][j] - u[i] - v[j]
                    self.assertGreater(reduced, -1e-10)
                    if result.plan[i][j] > 1e-14:
                        self.assertLess(abs(reduced), 1e-10)
            certificate = result.certificate()
            self.assertTrue(certificate.optimal)
            self.assertLess(certificate.duality_gap, 1e-12)
            self.assertLess(certificate.complementary_slackness, 1e-12)
            self.assertLess(certificate.marginal_error, 1e-12)
            positive = sum(1 for row in result.plan for value in row if value > 0)
            self.assertLessEqual(positive, m + n - 1)

    def test_certificate_rejects_a_suboptimal_plan(self) -> None:
        """The identity coupling of a crossing problem is not certified."""
        cost = [[1.0, 0.0], [0.0, 1.0]]
        weights = [0.5, 0.5]
        certificate = transport_certificate(
            [[0.5, 0.0], [0.0, 0.5]], [0.0, 0.0], [0.0, 0.0], weights, weights, cost
        )
        self.assertFalse(certificate.optimal)
        self.assertAlmostEqual(certificate.primal_cost, 1.0)
        self.assertAlmostEqual(exact_transport(weights, weights, cost).cost, 0.0)

    def test_degenerate_constant_cost_terminates(self) -> None:
        """A fully degenerate problem stops with an optimal certificate."""
        weights = [0.25] * 4
        result = exact_transport(weights, weights, [[1.0] * 4 for _ in range(4)])
        self.assertAlmostEqual(result.cost, 1.0, places=14)
        self.assertTrue(result.certificate().optimal)

    def test_one_dimensional_wasserstein_matches_sorted_matching(self) -> None:
        """Equal-weight 1-D clouds are optimally matched in sorted order."""
        rng = random.Random(SEED + 2)
        xs = [rng.gauss(0.0, 1.0) for _ in range(7)]
        ys = [rng.gauss(0.5, 2.0) for _ in range(7)]
        weights = [1.0 / 7.0] * 7
        for p in (1.0, 2.0, 3.0):
            exact = discrete_wasserstein(xs, weights, ys, weights, p=p)
            sorted_value = (
                math.fsum(
                    abs(x - y) ** p for x, y in zip(sorted(xs), sorted(ys), strict=True)
                )
                / 7.0
            ) ** (1.0 / p)
            self.assertAlmostEqual(exact, sorted_value, places=12)

    def test_cost_matrices_and_validation(self) -> None:
        """Costs use the Euclidean norm, and unbalanced problems are rejected."""
        cost = power_cost_matrix([(0.0, 0.0)], [(3.0, 4.0)], p=1.0)
        self.assertEqual(cost, ((5.0,),))
        self.assertEqual(squared_euclidean_cost_matrix([1.0], [3.0]), ((4.0,),))
        with self.assertRaises(ValueError):
            exact_transport([0.5, 0.5], [0.4, 0.4], [[0.0, 1.0], [1.0, 0.0]])
        with self.assertRaises(ValueError):
            exact_transport([1.0], [1.0], [[0.0, 1.0]])
        with self.assertRaises(ValueError):
            power_cost_matrix([(0.0, 0.0)], [1.0])


class SinkhornTests(unittest.TestCase):
    """Log-domain Sinkhorn satisfies marginals and approaches exact transport."""

    def setUp(self) -> None:
        """Build a 5 x 5 random problem with costs in ``[0, 1]``."""
        rng = random.Random(SEED + 3)
        self.a = _random_pmf(rng, 5)
        self.b = _random_pmf(rng, 5)
        self.cost = [[rng.random() for _ in range(5)] for _ in range(5)]
        self.exact = exact_transport(self.a, self.b, self.cost).cost

    def test_marginals_hold_to_tolerance(self) -> None:
        """Columns are exact and the row error is below the tolerance."""
        result = sinkhorn(self.a, self.b, self.cost, epsilon=0.05, tolerance=1e-10)
        self.assertTrue(result.converged)
        rows = [math.fsum(row) for row in result.plan]
        columns = [math.fsum(row[j] for row in result.plan) for j in range(5)]
        self.assertLess(
            math.fsum(abs(r - a) for r, a in zip(rows, self.a, strict=True)), 1e-10
        )
        self.assertLess(
            max(abs(c - b) for c, b in zip(columns, self.b, strict=True)), 1e-14
        )
        self.assertEqual(len(result.trace), result.iterations)
        self.assertAlmostEqual(result.trace[-1], result.marginal_error)

    def test_cost_converges_to_exact_as_epsilon_shrinks(self) -> None:
        """``<P, C>`` and the regularized cost decrease to the LP optimum."""
        gaps = []
        for epsilon in (0.5, 0.1, 0.02, 0.005):
            result = sinkhorn(
                self.a, self.b, self.cost, epsilon=epsilon, max_iterations=50_000
            )
            self.assertTrue(result.converged)
            self.assertGreaterEqual(result.transport_cost, self.exact - 1e-9)
            self.assertGreaterEqual(
                result.regularized_cost, result.transport_cost - 1e-9
            )
            gaps.append(result.transport_cost - self.exact)
        self.assertEqual(gaps, sorted(gaps, reverse=True))
        self.assertLess(gaps[-1], 1e-3)

    def test_epsilon_scaling_and_zero_mass_atoms(self) -> None:
        """Annealing reaches the same solution; zero-mass rows carry no mass."""
        plain = sinkhorn(self.a, self.b, self.cost, epsilon=0.05, tolerance=1e-11)
        scaled = sinkhorn(
            self.a,
            self.b,
            self.cost,
            epsilon=0.05,
            tolerance=1e-11,
            epsilon_scaling=0.5,
        )
        self.assertAlmostEqual(plain.transport_cost, scaled.transport_cost, places=9)
        a = (0.0, *self.a[1:])
        a = tuple(x / math.fsum(a) for x in a)
        result = sinkhorn(a, self.b, self.cost, epsilon=0.05)
        self.assertEqual(result.plan[0], (0.0,) * 5)

    def test_sinkhorn_divergence(self) -> None:
        """``S(alpha, alpha) = 0``, ``S > 0`` otherwise, and ``S -> W_2^2``."""
        rng = random.Random(SEED + 4)
        xs = [(rng.gauss(0, 1), rng.gauss(0, 1)) for _ in range(6)]
        ys = [(rng.gauss(2, 1), rng.gauss(0, 1)) for _ in range(6)]
        weights = [1.0 / 6.0] * 6
        self.assertLess(
            abs(sinkhorn_divergence(xs, weights, xs, weights, epsilon=0.5)), 1e-8
        )
        exact = exact_transport(
            weights, weights, squared_euclidean_cost_matrix(xs, ys)
        ).cost
        errors = []
        for epsilon in (4.0, 1.0, 0.25):
            value = sinkhorn_divergence(
                xs, weights, ys, weights, epsilon=epsilon, max_iterations=50_000
            )
            self.assertGreater(value, 0.0)
            errors.append(abs(value - exact))
        self.assertEqual(errors, sorted(errors, reverse=True))
        self.assertLess(errors[-1] / exact, 0.02)


class WassersteinClosedFormTests(unittest.TestCase):
    """Gaussian and quantile Wasserstein formulas agree with each other."""

    def test_gaussian_w2_matches_one_dimensional_formula_and_quadrature(self) -> None:
        """Bures, the 1-D formula, and the quantile integral coincide."""
        for mu0, s0, mu1, s1 in ((1.0, 2.0, -0.5, 0.7), (0.0, 1.0, 0.0, 3.0)):
            closed = normal_wasserstein(mu0, s0, mu1, s1)
            bures = gaussian_wasserstein([mu0], [[s0 * s0]], [mu1], [[s1 * s1]])
            quadrature = quantile_wasserstein(Normal(mu0, s0), Normal(mu1, s1))
            self.assertAlmostEqual(bures, closed, places=12)
            self.assertAlmostEqual(quadrature, closed, places=10)

    def test_bures_for_commuting_covariances(self) -> None:
        """Diagonal covariances give ``sum (sqrt a_i - sqrt b_i)^2``."""
        value = bures_distance([[4.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 9.0]])
        self.assertAlmostEqual(value, math.sqrt(1.0 + 4.0), places=12)
        full = [[2.0, 0.6], [0.6, 1.0]]
        self.assertLess(bures_distance(full, full), 1e-7)
        other = [[1.0, -0.3], [-0.3, 0.5]]
        self.assertAlmostEqual(
            bures_distance(full, other), bures_distance(other, full), places=10
        )

    def test_quantile_w1_matches_cdf_integral(self) -> None:
        """``W_1 = int |F0 - F1|`` for a gamma and an exponential law."""
        gamma, exponential = Gamma(2.0, 1.0), Exponential(0.8)
        reference = composite_simpson(
            lambda x: abs(gamma.cdf(x) - exponential.cdf(x)), 0.0, 80.0, intervals=8000
        ).value
        value = quantile_wasserstein(gamma, exponential, p=1.0)
        self.assertAlmostEqual(value, reference, places=7)

    def test_barycenter_of_two_normals(self) -> None:
        """Averaged quantiles give the normal with averaged mean and sd."""
        barycenter = quantile_barycenter([Normal(0.0, 1.0), Normal(4.0, 3.0)])
        target = Normal(2.0, 2.0)
        for u in (0.001, 0.1, 0.5, 0.77, 0.999):
            self.assertAlmostEqual(
                barycenter.quantile(u), target.quantile(u), places=10
            )
        self.assertAlmostEqual(barycenter.mean, 2.0)
        self.assertLess(quantile_wasserstein(barycenter, target), 1e-10)
        weighted = quantile_barycenter(
            [Normal(0.0, 1.0), Normal(4.0, 3.0)], [0.25, 0.75]
        )
        self.assertAlmostEqual(weighted.quantile(0.9), Normal(3.0, 2.5).quantile(0.9))

    def test_gaussian_barycenter_fixed_point(self) -> None:
        """The fixed point matches commuting closed forms and its own equation."""
        one_d = gaussian_barycenter([[0.0], [4.0]], [[[1.0]], [[9.0]]])
        self.assertAlmostEqual(one_d.covariance[0][0], 4.0, places=12)
        self.assertEqual(one_d.mean, (2.0,))
        diagonal = gaussian_barycenter(
            [[0.0, 0.0], [1.0, 1.0]],
            [[[4.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 9.0]]],
            [0.5, 0.5],
        )
        self.assertAlmostEqual(diagonal.covariance[0][0], 2.25, places=10)
        self.assertAlmostEqual(diagonal.covariance[1][1], 4.0, places=10)
        general = gaussian_barycenter(
            [[0, 0], [1, 1], [2, -1]],
            [[[2, 0.5], [0.5, 1]], [[1, -0.3], [-0.3, 2]], [[0.5, 0], [0, 0.5]]],
        )
        self.assertTrue(general.converged)
        self.assertLess(general.residual, 1e-12)
        self.assertEqual(general.mean, (1.0, 0.0))


class FisherInformationTests(unittest.TestCase):
    """Closed-form Fisher information matrices match numerical expectations."""

    FAMILIES: Sequence[
        tuple[Callable[[Sequence[float]], object], tuple[float, ...]]
    ] = (
        (lambda t: Normal(t[0], t[1]), (0.3, 1.7)),
        (lambda t: Exponential(t[0]), (2.5,)),
        (lambda t: Gamma(t[0], t[1]), (2.3, 1.4)),
        (lambda t: Beta(t[0], t[1]), (2.0, 3.5)),
        (lambda t: Bernoulli(t[0]), (0.3,)),
        (lambda t: Binomial(12, t[0]), (0.3,)),
        (lambda t: Poisson(t[0]), (3.2,)),
    )

    def test_closed_forms_match_score_and_hessian_quadrature(self) -> None:
        """Both numerical forms agree with the closed form to ``1e-6``."""
        for family, theta in self.FAMILIES:
            exact = fisher_information(family(theta))
            for method in ("score", "hessian"):
                numerical = numerical_fisher_information(family, theta, method=method)
                for i, row in enumerate(exact):
                    for j, value in enumerate(row):
                        self.assertLess(
                            abs(numerical[i][j] - value) / max(1.0, abs(value)),
                            1e-6,
                            msg=f"{type(family(theta)).__name__} {method}",
                        )

    def test_monte_carlo_estimate_is_within_four_standard_errors(self) -> None:
        """The score outer product averaged over draws matches ``1/sigma^2``."""
        sigma = 1.7
        estimate = numerical_fisher_information(
            lambda t: Normal(t[0], t[1]),
            (0.3, sigma),
            integration="monte_carlo",
            samples=4000,
            seed=SEED,
        )
        standard_error = math.sqrt(2.0 / 4000) / sigma**2
        self.assertLess(abs(estimate[0][0] - 1.0 / sigma**2), 4.0 * standard_error)

    def test_categorical_and_gaussian_mean_information(self) -> None:
        """The categorical information inverts ``diag(p) - p p'``; MVN gives S^-1."""
        p = (0.2, 0.3, 0.1, 0.4)
        information = categorical_fisher_information(p)
        covariance = [
            [(p[i] if i == j else 0.0) - p[i] * p[j] for j in range(3)]
            for i in range(3)
        ]
        product = matmul(information, tuple(tuple(row) for row in covariance))
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(product[i][j], 1.0 if i == j else 0.0)
        sigma = ((2.0, 0.4), (0.4, 1.0))
        product = matmul(multivariate_normal_mean_fisher_information(sigma), sigma)
        self.assertAlmostEqual(product[0][0], 1.0)
        self.assertAlmostEqual(product[0][1], 0.0)


class FisherRaoTests(unittest.TestCase):
    """Fisher-Rao distances are metrics with the Fisher local expansion."""

    def _check_metric(
        self, distance: Callable[[object, object], float], points: list[object]
    ) -> None:
        """Check symmetry, identity, and the triangle inequality on triples."""
        for x, y, z in itertools.combinations(points, 3):
            self.assertAlmostEqual(distance(x, y), distance(y, x), places=12)
            self.assertLessEqual(
                distance(x, z), distance(x, y) + distance(y, z) + 1e-12
            )
            self.assertLess(distance(x, x), 1e-12)

    def test_symmetry_and_triangle_inequality(self) -> None:
        """Every closed form passes on random triples."""
        rng = random.Random(SEED + 5)
        normals = [(rng.uniform(-2, 2), rng.uniform(0.2, 3)) for _ in range(9)]
        self._check_metric(
            lambda x, y: fisher_rao_normal(*x, *y),
            normals,
        )
        pmfs = [_random_pmf(rng, 4) for _ in range(9)]
        self._check_metric(fisher_rao_categorical, pmfs)
        probabilities = [rng.random() for _ in range(9)]
        self._check_metric(
            lambda x, y: fisher_rao_binomial(5, x, y),
            probabilities,
        )
        rates = [rng.uniform(0.1, 10) for _ in range(9)]
        self._check_metric(fisher_rao_poisson, rates)
        self._check_metric(fisher_rao_exponential, rates)
        means = [(rng.gauss(0, 1), rng.gauss(0, 1)) for _ in range(9)]
        sigma = ((2.0, 0.4), (0.4, 1.0))
        self._check_metric(
            lambda x, y: fisher_rao_normal_mean(x, y, sigma),
            means,
        )

    def test_small_distance_matches_fisher_quadratic_form(self) -> None:
        """``d(theta, theta + h v) / (h sqrt(v' I v)) -> 1``."""
        h = 1e-5
        mu, sigma, v = 0.4, 1.3, (0.6, -0.8)
        info = normal_fisher_information(sigma)
        local = math.sqrt(
            sum(v[i] * info[i][j] * v[j] for i in range(2) for j in range(2))
        )
        distance = fisher_rao_normal(mu, sigma, mu + h * v[0], sigma + h * v[1])
        self.assertAlmostEqual(distance / (h * local), 1.0, places=5)
        cases = (
            (fisher_rao_poisson, 3.0, fisher_information(Poisson(3.0))),
            (fisher_rao_exponential, 2.0, fisher_information(Exponential(2.0))),
            (fisher_rao_bernoulli, 0.3, fisher_information(Bernoulli(0.3))),
        )
        for function, theta, matrix in cases:
            ratio = function(theta, theta + h) / (h * math.sqrt(matrix[0][0]))
            self.assertAlmostEqual(ratio, 1.0, places=5)
        p = (0.2, 0.3, 0.5)
        direction = (0.5, -0.2)
        q = (p[0] + h * direction[0], p[1] + h * direction[1])
        q = (*q, 1.0 - q[0] - q[1])
        info = categorical_fisher_information(p)
        local = math.sqrt(
            sum(
                direction[i] * info[i][j] * direction[j]
                for i in range(2)
                for j in range(2)
            )
        )
        self.assertAlmostEqual(
            fisher_rao_categorical(p, q) / (h * local), 1.0, places=5
        )

    def test_closed_forms_agree_with_alternative_expressions(self) -> None:
        """The normal distance matches the artanh form; Bernoulli is categorical."""
        mu0, s0, mu1, s1 = 0.3, 1.2, -1.1, 0.4
        num = (mu0 - mu1) ** 2 + 2 * (s0 - s1) ** 2
        den = (mu0 - mu1) ** 2 + 2 * (s0 + s1) ** 2
        artanh = 2.0 * math.sqrt(2.0) * math.atanh(math.sqrt(num / den))
        self.assertAlmostEqual(fisher_rao_normal(mu0, s0, mu1, s1), artanh, places=12)
        self.assertAlmostEqual(
            fisher_rao_bernoulli(0.2, 0.7),
            fisher_rao_categorical((0.2, 0.8), (0.7, 0.3)),
            places=12,
        )
        arccos = 2.0 * math.acos(
            math.fsum(
                math.sqrt(a * b) for a, b in zip((0.2, 0.8), (0.7, 0.3), strict=True)
            )
        )
        self.assertAlmostEqual(fisher_rao_bernoulli(0.2, 0.7), arccos, places=12)


class DivergenceTests(unittest.TestCase):
    """Hellinger, Bhattacharyya, total variation, Renyi and alpha divergences."""

    P = (0.2, 0.5, 0.3)
    Q = (0.4, 0.4, 0.2)

    def test_finite_identities(self) -> None:
        """``H^2 = 1 - BC``, ``D_B = D_half/2``, ``D_A(1/2) = 4 H^2``, bounds."""
        h = hellinger_distance(self.P, self.Q)
        bc = bhattacharyya_coefficient(self.P, self.Q)
        tv = total_variation_distance(self.P, self.Q)
        self.assertAlmostEqual(h * h, 1.0 - bc, places=14)
        self.assertAlmostEqual(
            bhattacharyya_distance(self.P, self.Q),
            0.5 * renyi_divergence(self.P, self.Q, 0.5),
            places=14,
        )
        self.assertAlmostEqual(alpha_divergence(self.P, self.Q, 0.5), 4 * h * h)
        self.assertLessEqual(h * h, tv)
        self.assertLessEqual(tv, math.sqrt(1.0 - bc * bc))
        for alpha in (0.3, 2.0, -0.5):
            d_a = alpha_divergence(self.P, self.Q, alpha)
            if alpha > 0:
                renyi = renyi_divergence(self.P, self.Q, alpha)
                expected = math.log1p(alpha * (alpha - 1) * d_a) / (alpha - 1)
                self.assertAlmostEqual(renyi, expected, places=12)
        self.assertAlmostEqual(
            alpha_divergence(self.P, self.Q, 0.0), kl_divergence(self.Q, self.P)
        )

    def test_renyi_tends_to_kl_and_is_monotone(self) -> None:
        """``|D_{1 +- delta} - KL| = O(delta)``; ``D_alpha`` grows with alpha."""
        kl = kl_divergence(self.P, self.Q)
        for delta in (1e-2, 1e-4, 1e-6):
            for alpha in (1.0 - delta, 1.0 + delta):
                self.assertLess(
                    abs(renyi_divergence(self.P, self.Q, alpha) - kl), delta
                )
        orders = (0.0, 0.25, 0.5, 1.0, 2.0, 5.0, math.inf)
        values = [renyi_divergence(self.P, self.Q, a) for a in orders]
        self.assertEqual(values, sorted(values))
        self.assertAlmostEqual(values[-1], math.log(1.5))
        m0, s0, m1, s1 = [0.3], [[1.44]], [-0.4], [[0.49]]
        gauss_kl = kl_multivariate_normal(m0, s0, m1, s1)
        for delta in (1e-3, 1e-5):
            value = renyi_divergence_gaussian(m0, s0, m1, s1, 1.0 - delta)
            self.assertLess(abs(value - gauss_kl), 10 * delta)

    def test_gaussian_closed_forms_match_numerical_integration(self) -> None:
        """Bhattacharyya, Hellinger, Renyi and TV against Simpson integrals."""
        alpha = 0.3
        for mu0, s0, mu1, s1 in ((0.3, 1.2, -0.4, 0.7), (1.0, 0.5, 2.5, 2.0)):
            args = ([mu0], [[s0 * s0]], [mu1], [[s1 * s1]])

            def p0(x: float, mu0: float = mu0, s0: float = s0) -> float:
                """Return the first density."""
                return _normal_pdf(x, mu0, s0)

            def p1(x: float, mu1: float = mu1, s1: float = s1) -> float:
                """Return the second density."""
                return _normal_pdf(x, mu1, s1)

            bc = _integral(lambda x: math.sqrt(p0(x) * p1(x)), -20.0, 20.0)
            self.assertAlmostEqual(
                bhattacharyya_distance_gaussian(*args), -math.log(bc), places=9
            )
            self.assertAlmostEqual(
                hellinger_distance_gaussian(*args), math.sqrt(1.0 - bc), places=9
            )
            tv = composite_simpson(
                lambda x: 0.5 * abs(p0(x) - p1(x)), -20.0, 20.0, intervals=200_000
            ).value
            self.assertAlmostEqual(
                total_variation_normal(mu0, s0, mu1, s1), tv, places=8
            )
            affinity = _integral(
                lambda x: p0(x) ** alpha * p1(x) ** (1 - alpha), -20.0, 20.0
            )
            self.assertAlmostEqual(
                renyi_divergence_gaussian(*args, alpha),
                math.log(affinity) / (alpha - 1),
                places=9,
            )
            self.assertAlmostEqual(
                alpha_divergence_gaussian(*args, alpha),
                (affinity - 1) / (alpha * (alpha - 1)),
                places=9,
            )
        self.assertEqual(
            renyi_divergence_gaussian([0.0], [[1.44]], [0.0], [[0.49]], 2.0), math.inf
        )

    def test_gaussian_factorization_and_shared_covariance(self) -> None:
        """Independent coordinates add Bhattacharyya distances; TV reduces to 1-D."""
        joint = bhattacharyya_distance_gaussian(
            [0.0, 1.0], [[1.0, 0.0], [0.0, 4.0]], [0.5, -1.0], [[2.0, 0.0], [0.0, 1.0]]
        )
        parts = bhattacharyya_distance_gaussian(
            [0.0], [[1.0]], [0.5], [[2.0]]
        ) + bhattacharyya_distance_gaussian([1.0], [[4.0]], [-1.0], [[1.0]])
        self.assertAlmostEqual(joint, parts, places=12)
        sigma = [[2.0, 0.4], [0.4, 1.0]]
        delta = fisher_rao_normal_mean([0.0, 0.0], [1.0, -0.5], sigma)
        self.assertAlmostEqual(
            total_variation_gaussian_shared_covariance([0.0, 0.0], [1.0, -0.5], sigma),
            total_variation_normal(0.0, 1.0, delta, 1.0),
            places=12,
        )

    def test_disjoint_supports(self) -> None:
        """Disjoint supports give the boundary values."""
        p, q = (0.5, 0.5, 0.0), (0.0, 0.0, 1.0)
        self.assertEqual(total_variation_distance(p, q), 1.0)
        self.assertEqual(hellinger_distance(p, q), 1.0)
        self.assertEqual(bhattacharyya_distance(p, q), math.inf)
        self.assertEqual(renyi_divergence(p, q, 2.0), math.inf)
        self.assertEqual(renyi_divergence(p, q, 0.5), math.inf)


if __name__ == "__main__":
    unittest.main()
