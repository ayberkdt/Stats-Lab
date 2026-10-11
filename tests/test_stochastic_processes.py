"""Tests for Markov chains, continuous-time processes, SDE schemes, and KL fields."""

from __future__ import annotations

import math
import statistics
import unittest

from probstat.distributions import Exponential, Normal
from probstat.modeling import (
    ProbabilisticModel,
    RandomFieldInput,
    karhunen_loeve_inputs,
)
from probstat.processes import (
    BrownianMotion,
    CompoundPoissonProcess,
    CovarianceKernel,
    GeometricBrownianMotion,
    MarkovChain,
    OrnsteinUhlenbeck,
    PoissonProcess,
    ScalarSDE,
    brownian_increments,
    covariance_matrix,
    empirical_order,
    euler_maruyama,
    exponential_kernel_kl_modes,
    gauss_legendre,
    interval_rule,
    karhunen_loeve,
    karhunen_loeve_interval,
    milstein,
    rectangle_rule,
    strong_convergence_study,
    weak_convergence_study,
)
from probstat.propagation import propagate
from probstat.uq import QuantityOfInterest


def _gamblers_ruin(p: float, target: int) -> MarkovChain:
    """Return the gambler's-ruin chain on ``0..target`` with win probability p."""
    rows = []
    for i in range(target + 1):
        row = [0.0] * (target + 1)
        if i in (0, target):
            row[i] = 1.0
        else:
            row[i + 1] = p
            row[i - 1] = 1.0 - p
        rows.append(row)
    return MarkovChain(rows)


class MarkovChainTests(unittest.TestCase):
    """Discrete-time Markov chains against hand and closed-form solutions."""

    def test_validation_rejects_non_stochastic_matrices(self) -> None:
        """Negative entries, bad row sums, and non-square shapes raise."""
        with self.assertRaises(ValueError):
            MarkovChain([[1.1, -0.1], [0.5, 0.5]])
        with self.assertRaises(ValueError):
            MarkovChain([[0.5, 0.4], [0.5, 0.5]])
        with self.assertRaises(ValueError):
            MarkovChain([[0.5, 0.5]])
        with self.assertRaises(ValueError):
            MarkovChain([[1.0, 0.0], [0.0, 1.0]], labels=("a", "a"))

    def test_two_state_chain_matches_closed_forms(self) -> None:
        """Stationary law, n-step matrix, and first passage of a 2-state chain."""
        a, b = 0.3, 0.1
        chain = MarkovChain([[1 - a, a], [b, 1 - b]], labels=("dry", "wet"))
        pi = chain.stationary_distribution()
        self.assertAlmostEqual(pi[0], b / (a + b), places=13)
        self.assertAlmostEqual(pi[1], a / (a + b), places=13)
        for n in (0, 1, 5, 17):
            expected = b / (a + b) + a / (a + b) * (1 - a - b) ** n
            self.assertAlmostEqual(chain.n_step_matrix(n)[0][0], expected, places=13)
            self.assertAlmostEqual(
                chain.distribution_after((1.0, 0.0), n)[0], expected, places=13
            )
        passage = chain.mean_first_passage_times()
        self.assertAlmostEqual(passage[0][1], 1 / a, places=12)
        self.assertAlmostEqual(passage[1][0], 1 / b, places=12)
        self.assertAlmostEqual(passage[0][0], (a + b) / b, places=12)
        self.assertEqual(chain.index("wet"), 1)
        self.assertTrue(chain.is_aperiodic())

    def test_birth_death_chain_stationary_distribution(self) -> None:
        """A symmetric 3-state birth-death chain has pi = (1/4, 1/2, 1/4)."""
        chain = MarkovChain([[0.5, 0.5, 0.0], [0.25, 0.5, 0.25], [0.0, 0.5, 0.5]])
        for value, expected in zip(
            chain.stationary_distribution(), (0.25, 0.5, 0.25), strict=True
        ):
            self.assertAlmostEqual(value, expected, places=13)
        # Kac: mean return time equals 1 / pi_j.
        self.assertAlmostEqual(chain.mean_first_passage_times()[1][1], 2.0, places=12)
        # Hand solution of m_02 = 1 + 0.5 m_02 + 0.5 m_12, m_12 = 1 + 0.25 m_02
        # + 0.5 m_12 gives m_02 = 8 and m_12 = 6.
        passage = chain.mean_first_passage_times()
        self.assertAlmostEqual(passage[0][2], 8.0, places=12)
        self.assertAlmostEqual(passage[1][2], 6.0, places=12)

    def test_periodic_and_reducible_classification(self) -> None:
        """A 3-cycle has period 3; two closed classes make pi non-unique."""
        cycle = MarkovChain([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        self.assertTrue(cycle.is_irreducible())
        self.assertEqual(cycle.period(0), 3)
        self.assertFalse(cycle.is_aperiodic())
        for value in cycle.stationary_distribution():
            self.assertAlmostEqual(value, 1 / 3, places=13)
        bipartite = MarkovChain(
            [[0, 0.5, 0, 0.5], [0.5, 0, 0.5, 0], [0, 0.5, 0, 0.5], [0.5, 0, 0.5, 0]]
        )
        self.assertEqual(bipartite.period(2), 2)
        reducible = MarkovChain(
            [[0.5, 0.5, 0, 0], [0.5, 0.5, 0, 0], [0.25, 0, 0.5, 0.25], [0, 0, 0, 1]]
        )
        self.assertEqual(reducible.communicating_classes(), ((0, 1), (2,), (3,)))
        self.assertEqual(reducible.closed_classes(), ((0, 1), (3,)))
        self.assertEqual(reducible.transient_states(), (2,))
        self.assertFalse(reducible.is_irreducible())
        with self.assertRaises(ValueError):
            reducible.stationary_distribution()
        with self.assertRaises(ValueError):
            reducible.mean_first_passage_times()
        with self.assertRaises(ValueError):
            reducible.absorption()

    def test_single_closed_class_gives_zero_mass_to_transient_states(self) -> None:
        """Transient states carry no stationary probability."""
        chain = MarkovChain([[0.2, 0.8, 0.0], [0.0, 0.6, 0.4], [0.0, 0.3, 0.7]])
        pi = chain.stationary_distribution()
        self.assertAlmostEqual(pi[0], 0.0, places=14)
        self.assertAlmostEqual(pi[1], 3 / 7, places=13)
        self.assertAlmostEqual(pi[2], 4 / 7, places=13)

    def test_gamblers_ruin_absorption_matches_closed_form(self) -> None:
        """Absorption probabilities and durations equal the classical formulas."""
        p, target = 0.4, 5
        q, r = 1 - p, (1 - p) / p
        analysis = _gamblers_ruin(p, target).absorption()
        self.assertEqual(analysis.absorbing_states, (0, target))
        self.assertEqual(analysis.transient_states, (1, 2, 3, 4))
        for i in range(1, target):
            win = (1 - r**i) / (1 - r**target)
            duration = i / (q - p) - target / (q - p) * win
            self.assertAlmostEqual(analysis.probability(i, target), win, places=12)
            self.assertAlmostEqual(analysis.probability(i, 0), 1 - win, places=12)
            self.assertAlmostEqual(analysis.steps(i), duration, places=11)
        self.assertEqual(analysis.probability(0, 0), 1.0)
        self.assertEqual(analysis.steps(target), 0.0)
        fair = _gamblers_ruin(0.5, 6).absorption()
        for i in range(1, 6):
            self.assertAlmostEqual(fair.probability(i, 6), i / 6, places=12)
            self.assertAlmostEqual(fair.steps(i), i * (6 - i), places=11)
        self.assertAlmostEqual(fair.fundamental[0][0], 5 / 3, places=12)

    def test_simulation_is_seeded_and_ergodic(self) -> None:
        """Paths reproduce under a seed and visit states at stationary rates."""
        chain = MarkovChain([[0.5, 0.5, 0.0], [0.25, 0.5, 0.25], [0.0, 0.5, 0.5]])
        first = chain.simulate(20000, seed=3)
        self.assertEqual(first, chain.simulate(20000, seed=3))
        self.assertNotEqual(first, chain.simulate(20000, seed=4))
        for state, expected in enumerate((0.25, 0.5, 0.25)):
            self.assertAlmostEqual(
                first.count(state) / len(first), expected, delta=0.02
            )
        self.assertFalse(
            any(a == 0 and b == 2 for a, b in zip(first, first[1:], strict=False))
        )
        drawn = chain.simulate(3, initial=(0.0, 0.0, 1.0), seed=1)
        self.assertEqual(drawn[0], 2)


class ContinuousProcessTests(unittest.TestCase):
    """Exactly simulated continuous-time processes against their moments."""

    def test_poisson_counts_have_equal_mean_and_variance(self) -> None:
        """N(2) for rate 3 has mean and variance 6."""
        process = PoissonProcess(3.0)
        counts = process.sample_counts(2.0, 4000, seed=21)
        self.assertAlmostEqual(statistics.fmean(counts), 6.0, delta=0.15)
        self.assertAlmostEqual(statistics.variance(counts), 6.0, delta=0.5)
        self.assertEqual(process.mean_count(2.0), process.variance_count(2.0))
        total = math.fsum(process.count_pmf(k, 2.0) for k in range(60))
        self.assertAlmostEqual(total, 1.0, places=12)
        self.assertAlmostEqual(process.count_pmf(0, 2.0), math.exp(-6.0), places=15)
        times = process.event_times(5.0, seed=2)
        self.assertTrue(all(0 < t <= 5.0 for t in times))
        self.assertEqual(list(times), sorted(times))
        with self.assertRaises(ValueError):
            PoissonProcess(0.0)

    def test_compound_poisson_moments(self) -> None:
        """Mean lambda t E[J] and variance lambda t E[J^2] with exponential jumps."""
        process = CompoundPoissonProcess(2.0, Exponential(0.5))
        self.assertAlmostEqual(process.mean(3.0), 12.0, places=12)
        self.assertAlmostEqual(process.variance(3.0), 48.0, places=12)
        values = process.sample_values(3.0, 3000, seed=5)
        self.assertAlmostEqual(statistics.fmean(values), 12.0, delta=0.5)
        self.assertAlmostEqual(statistics.variance(values) / 48.0, 1.0, delta=0.1)
        path = process.path(3.0, seed=1)
        self.assertEqual(path.times[0], 0.0)
        self.assertEqual(path.times[-1], 3.0)

    def test_brownian_motion_with_drift(self) -> None:
        """Brownian increments have variance h; drifted paths start at x0."""
        increments = brownian_increments(2.0, 4000, seed=8)
        self.assertAlmostEqual(
            statistics.fmean(d * d for d in increments) / (2.0 / 4000), 1.0, delta=0.07
        )
        motion = BrownianMotion(0.5, 2.0)
        self.assertEqual(motion.mean(2.0, 1.0), 2.0)
        self.assertEqual(motion.variance(2.0), 8.0)
        path = motion.path(1.0, 10, initial=3.0, seed=1)
        self.assertEqual(path.values[0], 3.0)
        self.assertEqual(len(path.values), 11)

    def test_geometric_brownian_motion_exact_moments(self) -> None:
        """Exact GBM draws match E = S0 e^(mu t) and the log-normal variance."""
        gbm = GeometricBrownianMotion(0.05, 0.2)
        draws = gbm.sample_terminal(100.0, 1.0, 20000, seed=13)
        mean, variance = gbm.mean(100.0, 1.0), gbm.variance(100.0, 1.0)
        self.assertAlmostEqual(mean, 100 * math.exp(0.05), places=10)
        self.assertAlmostEqual(variance, 1e4 * math.exp(0.1) * math.expm1(0.04))
        self.assertAlmostEqual(
            statistics.fmean(draws), mean, delta=3 * math.sqrt(variance / 20000)
        )
        self.assertAlmostEqual(statistics.variance(draws) / variance, 1.0, delta=0.05)
        path = gbm.path(1.0, 50, initial=100.0, seed=2)
        self.assertTrue(all(v > 0 for v in path.values))

    def test_ornstein_uhlenbeck_transition_and_stationary_variance(self) -> None:
        """Exact OU draws match the transition law; long paths the stationary one."""
        ou = OrnsteinUhlenbeck(2.0, 1.0, 0.5)
        self.assertAlmostEqual(ou.stationary_variance, 0.0625, places=15)
        draws = ou.sample_terminal(3.0, 0.3, 20000, seed=4)
        self.assertAlmostEqual(
            statistics.fmean(draws),
            ou.mean(3.0, 0.3),
            delta=4 * math.sqrt(0.0625 / 2e4),
        )
        self.assertAlmostEqual(
            statistics.variance(draws) / ou.variance(0.3), 1.0, delta=0.05
        )
        path = ou.path(1000.0, 20000, seed=6)
        self.assertAlmostEqual(
            statistics.variance(path.values) / ou.stationary_variance, 1.0, delta=0.15
        )
        self.assertAlmostEqual(statistics.fmean(path.values), 1.0, delta=0.05)
        self.assertAlmostEqual(ou.stationary_covariance(0.0), 0.0625, places=15)


class StochasticDifferentialEquationTests(unittest.TestCase):
    """Euler-Maruyama and Milstein schemes and their convergence orders."""

    def test_strong_orders_on_geometric_brownian_motion(self) -> None:
        """EM converges strongly at order ~1/2, Milstein at order ~1 (Higham)."""
        sde = ScalarSDE.geometric_brownian_motion(2.0, 1.0)
        gbm = GeometricBrownianMotion(2.0, 1.0)

        def exact(w):
            """Return the exact GBM terminal value for W_T = w."""
            return gbm.exact(1.0, 1.0, w)

        euler = strong_convergence_study(sde, 1.0, 1.0, exact, paths=500, seed=1)
        mil = strong_convergence_study(
            sde, 1.0, 1.0, exact, paths=500, scheme="milstein", seed=1
        )
        self.assertGreater(euler.order, 0.4)
        self.assertLess(euler.order, 0.7)
        self.assertGreater(mil.order, 0.85)
        self.assertLess(mil.order, 1.15)
        self.assertLess(mil.errors[-1], euler.errors[-1])
        self.assertEqual(euler.step_sizes, (1 / 16, 1 / 32, 1 / 64, 1 / 128))

    def test_weak_order_on_geometric_brownian_motion(self) -> None:
        """EM weak error tracks the exact bias e^(lambda) - (1 + lambda h)^N."""
        sde = ScalarSDE.geometric_brownian_motion(2.0, 0.1)
        gbm = GeometricBrownianMotion(2.0, 0.1)
        counts = (16, 32, 64, 128)
        study = weak_convergence_study(
            sde, 1.0, 1.0, gbm.mean(1.0, 1.0), step_counts=counts, paths=1000, seed=2
        )
        for n, error, se in zip(
            counts, study.errors, study.standard_errors, strict=True
        ):
            bias = math.exp(2.0) - (1 + 2.0 / n) ** n
            self.assertLess(abs(error - bias), 4 * se)
        self.assertGreater(study.order, 0.8)
        self.assertLess(study.order, 1.15)

    def test_schemes_share_increments_and_validate(self) -> None:
        """Milstein equals EM for additive noise; invalid requests raise."""
        ou = ScalarSDE.ornstein_uhlenbeck(1.0, 0.0, 0.3)
        increments = brownian_increments(1.0, 20, seed=9)
        a = euler_maruyama(ou, 1.0, 1.0, 20, increments=increments)
        b = milstein(ou, 1.0, 1.0, 20, increments=increments)
        for x, y in zip(a.values, b.values, strict=True):
            self.assertAlmostEqual(x, y, places=14)
        no_derivative = ScalarSDE(lambda t, x: 0.0, lambda t, x: x)
        with self.assertRaises(ValueError):
            milstein(no_derivative, 1.0, 1.0, 4, seed=1)
        with self.assertRaises(ValueError):
            euler_maruyama(ou, 1.0, 1.0, 5, increments=increments)
        with self.assertRaises(ValueError):
            weak_convergence_study(ou, 1.0, 1.0, 0.0, step_counts=(3, 4), paths=10)
        self.assertAlmostEqual(empirical_order((0.1, 0.01), (0.2, 0.002)), 2.0)


class RandomFieldTests(unittest.TestCase):
    """Covariance kernels and Karhunen-Loeve expansions."""

    def test_kernel_values(self) -> None:
        """Kernel families take their textbook values at r = ell."""
        expected = {
            "squared_exponential": math.exp(-0.5),
            "exponential": math.exp(-1.0),
            "matern32": (1 + math.sqrt(3)) * math.exp(-math.sqrt(3)),
            "matern52": (1 + math.sqrt(5) + 5 / 3) * math.exp(-math.sqrt(5)),
        }
        for family, value in expected.items():
            kernel = CovarianceKernel(family, 2.0, 0.7)
            self.assertAlmostEqual(kernel((0.0, 0.0), (0.7, 0.0)), 2.0 * value)
            self.assertEqual(kernel((1.0,), (1.0,)), 2.0)
        ou = OrnsteinUhlenbeck(2.0, 0.0, 1.0)
        kernel = CovarianceKernel("exponential", ou.stationary_variance, 0.5)
        self.assertAlmostEqual(kernel((0.0,), (0.8,)), ou.stationary_covariance(0.8))
        with self.assertRaises(ValueError):
            CovarianceKernel("rational", 1.0, 1.0)
        matrix = covariance_matrix(kernel, [0.0, 0.5, 1.0])
        self.assertEqual(matrix[0][2], matrix[2][0])

    def test_quadrature_rules(self) -> None:
        """Gauss-Legendre is exact to degree 2n-1; all rules sum to the length."""
        nodes, weights = gauss_legendre(5, 0.0, 2.0)
        self.assertAlmostEqual(
            math.fsum(w * x**9 for x, w in zip(nodes, weights, strict=True)),
            2.0**10 / 10,
            places=10,
        )
        for rule in ("gauss_legendre", "midpoint", "trapezoid"):
            _, w = interval_rule(-1.0, 3.0, 7, rule=rule)
            self.assertAlmostEqual(math.fsum(w), 4.0, places=12)

    def test_exponential_kernel_analytic_modes(self) -> None:
        """Analytic roots solve their equations and modes are orthonormal."""
        modes = exponential_kernel_kl_modes(1.0, 1.0, 1.0, 6)
        self.assertAlmostEqual(modes[0].frequency, 0.8603335890, places=9)
        self.assertEqual([m.parity for m in modes[:3]], ["even", "odd", "even"])
        for m in modes:
            w = m.frequency
            if m.parity == "even":
                self.assertAlmostEqual(1.0 - w * math.tan(w), 0.0, places=9)
            else:
                self.assertAlmostEqual(w + math.tan(w), 0.0, places=9)
        self.assertEqual(sorted(modes, key=lambda m: -m.eigenvalue), list(modes))
        points, weights = interval_rule(-1.0, 1.0, 4000, rule="midpoint")
        for i in range(4):
            for j in range(4):
                inner = math.fsum(
                    w * modes[i](x) * modes[j](x)
                    for x, w in zip(points, weights, strict=True)
                )
                self.assertAlmostEqual(inner, 1.0 if i == j else 0.0, places=5)

    def test_nystrom_eigenvalues_match_exponential_kernel_solution(self) -> None:
        """Discrete KL on [-1, 1] reproduces the Ghanem-Spanos eigenpairs."""
        kernel = CovarianceKernel("exponential", 1.0, 1.0)
        exact = exponential_kernel_kl_modes(1.0, 1.0, 1.0, 4)
        kl = karhunen_loeve_interval(
            kernel, -1.0, 1.0, 40, rule="midpoint", variance_fraction=1.0
        )
        for k in range(4):
            self.assertLess(abs(kl.eigenvalues[k] / exact[k].eigenvalue - 1), 1e-2)
            phi = kl.eigenfunction(k)
            analytic = [exact[k](x[0]) for x in kl.points]
            sign = math.copysign(
                1.0, math.fsum(a * b for a, b in zip(phi, analytic, strict=True))
            )
            self.assertLess(
                max(abs(sign * a - b) for a, b in zip(phi, analytic, strict=True)),
                2e-3,
            )
        self.assertLess(abs(kl.eigenvalues[0] / exact[0].eigenvalue - 1), 1e-3)
        self.assertAlmostEqual(math.fsum(kl.spectrum), kl.total_variance, places=10)
        self.assertAlmostEqual(kl.total_variance, 2.0, places=12)
        coarse = karhunen_loeve_interval(kernel, -1.0, 1.0, 20, variance_fraction=1.0)
        self.assertLess(abs(coarse.eigenvalues[0] / exact[0].eigenvalue - 1), 5e-3)

    def test_truncation_by_explained_variance(self) -> None:
        """The smallest mode count reaching the target fraction is retained."""
        kernel = CovarianceKernel("squared_exponential", 1.5, 0.4)
        kl = karhunen_loeve_interval(kernel, 0.0, 1.0, 24, variance_fraction=0.99)
        self.assertGreaterEqual(kl.explained_variance_fraction, 0.99)
        shorter = math.fsum(kl.spectrum[: kl.terms - 1]) / math.fsum(kl.spectrum)
        self.assertLess(shorter, 0.99)
        self.assertTrue(all(v <= 1.5 + 1e-12 for v in kl.pointwise_variance()))
        capped = karhunen_loeve_interval(
            kernel, 0.0, 1.0, 24, variance_fraction=0.99, max_terms=2
        )
        self.assertEqual(capped.terms, 2)
        full = karhunen_loeve_interval(kernel, 0.0, 1.0, 12, variance_fraction=1.0)
        matrix = covariance_matrix(kernel, [p[0] for p in full.points])
        for i in (0, 5, 11):
            self.assertAlmostEqual(full.covariance(i, 11 - i), matrix[i][11 - i], 8)
        samples = kl.sample(4000, seed=17)
        variance = statistics.variance(s[6] for s in samples)
        self.assertAlmostEqual(variance / kl.pointwise_variance()[6], 1.0, delta=0.08)
        node = kl.points[3][0]
        self.assertAlmostEqual(kl.eigenfunction_at(1, node), kl.eigenfunction(1)[3])
        coefficients = (0.3, -1.0) + (0.0,) * (kl.terms - 2)
        self.assertAlmostEqual(
            kl.realize_at(coefficients, node), kl.realize(coefficients)[3]
        )
        with self.assertRaises(ValueError):
            kl.realize((1.0,))
        with self.assertRaises(ValueError):
            karhunen_loeve_interval(kernel, 0.0, 1.0, 8, variance_fraction=0.0)

    def test_two_dimensional_grid_expansion(self) -> None:
        """A 2-D Matern field on a weighted grid conserves integrated variance."""
        points, weights = rectangle_rule((0.0, 2.0), (0.0, 1.0), (6, 4))
        kernel = CovarianceKernel("matern32", 0.5, 0.6)
        kl = karhunen_loeve(kernel, points, weights, variance_fraction=0.9)
        self.assertAlmostEqual(kl.total_variance, 1.0, places=12)
        self.assertAlmostEqual(math.fsum(kl.spectrum), 1.0, places=10)
        self.assertGreaterEqual(kl.explained_variance_fraction, 0.9)
        self.assertEqual(len(kl.realize([0.0] * kl.terms)), 24)
        plain = karhunen_loeve(kernel, points, variance_fraction=1.0)
        self.assertAlmostEqual(math.fsum(plain.spectrum), 0.5 * 24, places=10)


class RandomFieldInputTests(unittest.TestCase):
    """A KL random field as named inputs of a probabilistic model."""

    def test_adapter_names_inputs_and_field(self) -> None:
        """Coefficients become N(0, 1) inputs and map back to field values."""
        kl = karhunen_loeve_interval(
            CovarianceKernel("matern52", 1.0, 0.3), 0.0, 1.0, 16, max_terms=3
        )
        inputs, to_field = karhunen_loeve_inputs(kl, "g")
        self.assertEqual([item.name for item in inputs], ["g_1", "g_2", "g_3"])
        self.assertTrue(all(isinstance(i.distribution, Normal) for i in inputs))
        parameters = {"g_1": 0.5, "g_2": -0.2, "g_3": 1.0}
        self.assertEqual(to_field(parameters), kl.realize((0.5, -0.2, 1.0)))
        field = RandomFieldInput(kl, "g", transform=math.exp)
        node = kl.points[4][0]
        self.assertAlmostEqual(
            field.field_at(parameters, node), field.field(parameters)[4]
        )
        with self.assertRaises(KeyError):
            field.field({"g_1": 0.0})
        with self.assertRaises(ValueError):
            RandomFieldInput(kl, " ")

    def test_bar_with_lognormal_stiffness_field_propagates(self) -> None:
        """Tip displacement of a bar with random EA(x) matches its references.

        ``u(L) = P int_0^L dx / EA(x)`` with ``EA = exp(G)`` and ``G`` a
        Gaussian field with exponential covariance. Quadrature at the KL nodes
        makes the truncated model's mean and variance available in closed
        form; a separate 10,000-draw field sample is the high-sample check.
        """
        load = 2.0
        kl = karhunen_loeve_interval(
            CovarianceKernel("exponential", 0.3**2, 0.5),
            0.0,
            1.0,
            30,
            variance_fraction=0.9,
        )
        stiffness = RandomFieldInput(kl, transform=math.exp, description="EA(x)")

        def bar(parameters):
            """Return the tip displacement of the axially loaded bar."""
            ea = stiffness.field(parameters)
            return {
                "tip": load
                * math.fsum(w / k for w, k in zip(kl.weights, ea, strict=True))
            }

        model = ProbabilisticModel(
            stiffness.joint_input(),
            bar,
            (QuantityOfInterest("tip", "Tip displacement", "m"),),
            name="random-stiffness bar",
        )
        self.assertEqual(model.dimension, kl.terms)
        study = propagate(model, 2000, design="lhs", seed=7)
        n, w, v = len(kl.points), kl.weights, kl.pointwise_variance()
        exact_mean = load * math.fsum(w[i] * math.exp(v[i] / 2) for i in range(n))
        exact_variance = load**2 * math.fsum(
            w[i] * w[j] * math.exp((v[i] + v[j]) / 2) * math.expm1(kl.covariance(i, j))
            for i in range(n)
            for j in range(n)
        )
        reference = [
            load * math.fsum(wi * math.exp(-g) for wi, g in zip(w, s, strict=True))
            for s in kl.sample(10000, seed=11)
        ]
        output = study["tip"]
        self.assertAlmostEqual(output.mean / exact_mean, 1.0, delta=0.01)
        self.assertAlmostEqual(
            output.standard_deviation**2 / exact_variance, 1.0, delta=0.1
        )
        self.assertAlmostEqual(
            statistics.variance(reference) / exact_variance, 1.0, delta=0.06
        )
        self.assertAlmostEqual(
            output.standard_deviation**2 / statistics.variance(reference),
            1.0,
            delta=0.12,
        )


if __name__ == "__main__":
    unittest.main()
