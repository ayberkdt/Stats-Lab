"""Tests for classical optimal design and Bayesian experimental design."""

from __future__ import annotations

import math
import unittest

from probstat.design import (
    ApproximateDesign,
    BayesianDesignProblem,
    ExactDesign,
    LocalLinearization,
    d_efficiency,
    d_optimal_design,
    design_criterion,
    efficient_rounding,
    equivalence_check,
    exchange_exact_design,
    expected_information_gain,
    greedy_design,
    information_matrix,
    laplace_information_gain,
    linear_gaussian_information_gain,
    linearized_information_gain,
    nested_monte_carlo_information_gain,
    polynomial_basis,
    standardized_variances,
)
from probstat.distributions import LogNormal, Normal
from probstat.inverse import CalibrationProblem, Experiment, NoiseModel
from probstat.linalg import determinant, matmul, transpose
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.uq import QuantityOfInterest

QUADRATIC_GRID = tuple(round(-1.0 + 0.1 * i, 10) for i in range(21))
DECAY_RATE = 0.5
DECAY_AMPLITUDE = 3.0


def decay_response(x: tuple[float, ...], theta: tuple[float, ...]) -> float:
    """Return the exponential-decay mean ``A exp(-k t)``."""
    return theta[0] * math.exp(-theta[1] * x[0])


def line_model(parameters):
    """Return a straight line ``a + b x``."""
    return {"y": parameters["a"] + parameters["b"] * parameters["x"]}


def slope_model(parameters):
    """Return a line through the origin ``b x``."""
    return {"y": parameters["b"] * parameters["x"]}


def decay_model(parameters):
    """Return an exponential-decay signal ``A exp(-k t)``."""
    return {"y": parameters["A"] * math.exp(-parameters["k"] * parameters["t"])}


def line_problem(sd: float) -> BayesianDesignProblem:
    """Build a linear-Gaussian design problem with prior ``N(0,1) x N(0,4)``."""
    prior = JointInput.independent(
        RandomInput("a", Normal(0.0, 1.0)), RandomInput("b", Normal(0.0, 2.0))
    )
    model = ProbabilisticModel(prior, line_model, (QuantityOfInterest("y", "y"),))
    return BayesianDesignProblem(model, (NoiseModel("y", standard_deviation=sd),))


def decay_problem() -> BayesianDesignProblem:
    """Build a nonlinear decay design problem centred on ``A=3, k=0.5``."""
    prior = JointInput.independent(
        RandomInput("A", Normal(DECAY_AMPLITUDE, 0.3)),
        RandomInput("k", LogNormal(math.log(DECAY_RATE), 0.2)),
    )
    model = ProbabilisticModel(prior, decay_model, (QuantityOfInterest("y", "y"),))
    return BayesianDesignProblem(model, (NoiseModel("y", standard_deviation=0.1),))


LINE_DESIGN = ({"x": -1.0}, {"x": 0.5}, {"x": 1.0})
LINE_JACOBIAN = ((1.0, -1.0), (1.0, 0.5), (1.0, 1.0))
LINE_PRIOR_COVARIANCE = ((1.0, 0.0), (0.0, 4.0))


def closed_form_eig(sd: float) -> float:
    """Return ``1/2 log det(I + Sigma J'J / sd^2)`` by an LU determinant."""
    gram = matmul(transpose(LINE_JACOBIAN), LINE_JACOBIAN)
    product = matmul(LINE_PRIOR_COVARIANCE, gram)
    shifted = tuple(
        tuple(product[i][j] / sd**2 + (1.0 if i == j else 0.0) for j in range(2))
        for i in range(2)
    )
    return 0.5 * math.log(determinant(shifted))


class ClassicalDesignTests(unittest.TestCase):
    """Known optimal designs, criteria, and the equivalence theorem."""

    @classmethod
    def setUpClass(cls) -> None:
        """Compute the quadratic D-optimal design once."""
        cls.quadratic = d_optimal_design(
            QUADRATIC_GRID, polynomial_basis(2), tolerance=1e-6, max_iterations=5000
        )

    def test_quadratic_regression_d_optimum_is_three_point(self) -> None:
        """Quadratic on [-1, 1]: mass 1/3 at -1, 0, 1 and det M = 4/27."""
        result = self.quadratic
        self.assertTrue(result.converged)
        condensed = result.design.condensed(min_weight=1e-4, merge_distance=0.15)
        self.assertEqual(condensed.points, ((-1.0,), (0.0,), (1.0,)))
        for weight in condensed.weights:
            self.assertAlmostEqual(weight, 1.0 / 3.0, delta=1e-4)
        self.assertAlmostEqual(result.log_det, math.log(4.0 / 27.0), delta=1e-5)

    def test_fedorov_wynn_reaches_the_same_optimum(self) -> None:
        """The vertex-direction algorithm meets its certified efficiency."""
        result = d_optimal_design(
            QUADRATIC_GRID,
            polynomial_basis(2),
            method="fedorov-wynn",
            tolerance=1e-3,
            max_iterations=5000,
        )
        self.assertTrue(result.converged)
        self.assertGreaterEqual(result.equivalence.efficiency_lower_bound, 0.999)
        optimum = ApproximateDesign.uniform((-1.0, 0.0, 1.0))
        efficiency = d_efficiency(result.design, optimum, polynomial_basis(2))
        self.assertGreaterEqual(efficiency, result.equivalence.efficiency_lower_bound)
        self.assertLessEqual(efficiency, 1.0 + 1e-12)

    def test_linear_regression_puts_half_mass_at_endpoints(self) -> None:
        """Straight line on [2, 5]: weights 1/2 at both ends."""
        grid = [2.0 + 0.15 * i for i in range(21)]
        result = d_optimal_design(grid, polynomial_basis(1))
        condensed = result.design.condensed(min_weight=1e-4)
        self.assertEqual(len(condensed.points), 2)
        self.assertAlmostEqual(condensed.points[0][0], 2.0, places=12)
        self.assertAlmostEqual(condensed.points[1][0], grid[-1], places=12)
        for weight in condensed.weights:
            self.assertAlmostEqual(weight, 0.5, delta=1e-5)

    def test_equivalence_theorem_at_and_away_from_the_optimum(self) -> None:
        """The maximum variance is p at the optimum and exceeds p elsewhere."""
        basis = polynomial_basis(2)
        optimum = ApproximateDesign.uniform((-1.0, 0.0, 1.0))
        check = equivalence_check(optimum, basis, QUADRATIC_GRID, tolerance=1e-9)
        self.assertTrue(check.satisfied)
        self.assertAlmostEqual(check.max_variance, 3.0, places=12)
        for value in check.support_variances:
            self.assertAlmostEqual(value, 3.0, places=12)
        fine = [-1.0 + 0.001 * i for i in range(2001)]
        variances = standardized_variances(optimum, basis, fine)
        for x, value in zip(fine, variances, strict=True):
            self.assertAlmostEqual(value, 3.0 - 4.5 * x**2 + 4.5 * x**4, places=10)
        self.assertLessEqual(max(variances), 3.0 + 1e-12)
        uniform = ApproximateDesign.uniform(QUADRATIC_GRID)
        weighted = math.fsum(
            w * d
            for w, d in zip(
                uniform.weights,
                standardized_variances(uniform, basis, uniform.points),
                strict=True,
            )
        )
        self.assertAlmostEqual(weighted, 3.0, places=12)
        away = equivalence_check(uniform, basis, QUADRATIC_GRID)
        self.assertFalse(away.satisfied)
        self.assertGreater(away.max_variance, 3.0)
        efficiency = d_efficiency(uniform, optimum, basis)
        self.assertGreaterEqual(efficiency, away.efficiency_lower_bound)
        self.assertLess(efficiency, 1.0)

    def test_quadratic_optimum_criteria_have_closed_forms(self) -> None:
        """A = 9, E = (5 - sqrt 17)/6, I = mean of 3 - 4.5x^2 + 4.5x^4."""
        basis = polynomial_basis(2)
        optimum = ApproximateDesign.uniform((-1.0, 0.0, 1.0))
        self.assertAlmostEqual(design_criterion(optimum, basis, "A"), 9.0, places=10)
        self.assertAlmostEqual(
            design_criterion(optimum, basis, "E"), (5.0 - math.sqrt(17.0)) / 6.0, 12
        )
        expected = math.fsum(3.0 - 4.5 * x**2 + 4.5 * x**4 for x in QUADRATIC_GRID)
        self.assertAlmostEqual(
            design_criterion(optimum, basis, "I", region=QUADRATIC_GRID),
            expected / len(QUADRATIC_GRID),
            places=10,
        )
        matrix = information_matrix(optimum, basis)
        self.assertAlmostEqual(matrix[0][2], 2.0 / 3.0, places=14)
        singular = ApproximateDesign.uniform((0.0, 1.0))
        self.assertEqual(design_criterion(singular, basis, "D"), -math.inf)
        self.assertEqual(design_criterion(singular, basis, "A"), math.inf)
        with self.assertRaises(ValueError):
            design_criterion(optimum, basis, "Z")
        with self.assertRaises(ValueError):
            design_criterion(optimum, basis, "I")

    def test_exponential_decay_local_design_is_zero_and_inverse_rate(self) -> None:
        """Local D-optimum for A exp(-k t) on [0, 6] is {0, 1/k}, weights 1/2.

        Derivation: for two points with weights 1/2 the gradient vectors are
        ``(e^-kt, -A t e^-kt)``, so ``det M = A^2 e^{-2k(t1+t2)} (t2-t1)^2 / 4``.
        It decreases in ``t1`` (take ``t1 = 0``) and ``t^2 e^{-2kt}`` peaks at
        ``t = 1/k``. The brute-force search below confirms this numerically.
        """
        fine = [0.01 * i for i in range(601)]
        search = [0.02 * i for i in range(301)]
        best = max(
            ((t1, t2) for i, t1 in enumerate(search) for t2 in search[i + 1 :]),
            key=lambda pair: math.exp(-2.0 * DECAY_RATE * sum(pair))
            * (pair[1] - pair[0]) ** 2,
        )
        self.assertEqual(best[0], 0.0)
        self.assertAlmostEqual(best[1], 1.0 / DECAY_RATE, places=12)

        basis = LocalLinearization(decay_response, (DECAY_AMPLITUDE, DECAY_RATE))
        times = [0.25 * i for i in range(25)]
        result = d_optimal_design(times, basis, tolerance=1e-6)
        self.assertTrue(result.converged)
        condensed = result.design.condensed(min_weight=1e-3)
        self.assertEqual(len(condensed.points), 2)
        self.assertAlmostEqual(condensed.points[0][0], 0.0, places=12)
        self.assertAlmostEqual(condensed.points[1][0], 1.0 / DECAY_RATE, places=12)
        for weight in condensed.weights:
            self.assertAlmostEqual(weight, 0.5, delta=1e-3)
        self.assertEqual(result.basis_evaluations, len(times))
        self.assertEqual(basis.evaluations, 2 * 2 * len(times))
        local = ApproximateDesign.uniform((0.0, 1.0 / DECAY_RATE))
        check = equivalence_check(local, basis, fine, tolerance=1e-6)
        self.assertTrue(check.satisfied)
        self.assertAlmostEqual(check.max_variance, 2.0, delta=1e-6)
        other = LocalLinearization(decay_response, (10.0 * DECAY_AMPLITUDE, DECAY_RATE))
        self.assertTrue(equivalence_check(local, other, fine, tolerance=1e-6).satisfied)

    def test_efficient_rounding_and_exchange(self) -> None:
        """Rounding keeps n runs; exchange climbs to the {-1, 0, 1} design."""
        third = ApproximateDesign.uniform((-1.0, 0.0, 1.0))
        exact = efficient_rounding(third, 10)
        self.assertEqual(exact.counts, (4, 3, 3))
        self.assertEqual(exact.size, 10)
        halves = ApproximateDesign(((2.0,), (5.0,)), (0.5, 0.5))
        self.assertEqual(efficient_rounding(halves, 7).counts, (4, 3))
        skewed = ApproximateDesign(((0.0,), (1.0,), (2.0,)), (0.7, 0.2, 0.1))
        for n in (1, 2, 5, 13):
            self.assertEqual(efficient_rounding(skewed, n).size, n)
        start = ExactDesign(((-0.5,), (0.2,), (0.9,)), (4, 3, 3))
        result = exchange_exact_design(start, QUADRATIC_GRID, polynomial_basis(2))
        self.assertTrue(result.converged)
        self.assertGreater(result.log_det, result.initial_log_det)
        self.assertEqual(result.design.size, 10)
        self.assertEqual(result.design.points, ((-1.0,), (0.0,), (1.0,)))
        self.assertEqual(sorted(result.design.counts), [3, 3, 4])
        rounded = design_criterion(
            result.design.as_approximate(), polynomial_basis(2), "D"
        )
        self.assertAlmostEqual(rounded, result.log_det, places=12)
        self.assertLess(rounded, math.log(4.0 / 27.0))

    def test_invalid_designs_are_rejected(self) -> None:
        """Weights, counts, and singular starts are validated."""
        with self.assertRaises(ValueError):
            ApproximateDesign(((0.0,), (1.0,)), (0.6, 0.6))
        with self.assertRaises(ValueError):
            ApproximateDesign(((0.0,), (1.0, 2.0)), (0.5, 0.5))
        with self.assertRaises(ValueError):
            ExactDesign(((0.0,),), (0,))
        with self.assertRaises(ValueError):
            d_optimal_design((0.0, 1.0), polynomial_basis(2))
        with self.assertRaises(ValueError):
            d_optimal_design(QUADRATIC_GRID, polynomial_basis(2), method="simplex")
        with self.assertRaises(ValueError):
            exchange_exact_design(
                ExactDesign(((0.0,),), (3,)), QUADRATIC_GRID, polynomial_basis(2)
            )
        with self.assertRaises(ValueError):
            efficient_rounding(ApproximateDesign.uniform((0.0,)), 0)


class BayesianDesignTests(unittest.TestCase):
    """Expected information gain estimators and greedy selection."""

    def test_closed_form_matches_direct_determinant(self) -> None:
        """Cholesky-whitened closed form equals the LU determinant form."""
        for sd in (0.3, 1.5):
            value = linear_gaussian_information_gain(
                LINE_JACOBIAN, LINE_PRIOR_COVARIANCE, noise_variances=[sd**2] * 3
            )
            self.assertAlmostEqual(value, closed_form_eig(sd), places=12)
            full = linear_gaussian_information_gain(
                LINE_JACOBIAN,
                LINE_PRIOR_COVARIANCE,
                noise_covariance=[
                    [sd**2 if i == j else 0.0 for j in range(3)] for i in range(3)
                ],
            )
            self.assertAlmostEqual(full, value, places=12)
        with self.assertRaises(ValueError):
            linear_gaussian_information_gain(LINE_JACOBIAN, LINE_PRIOR_COVARIANCE)

    def test_linearized_and_laplace_are_exact_for_linear_gaussian(self) -> None:
        """Both deterministic forms reproduce the closed form and cost counts."""
        problem = line_problem(1.5)
        exact = closed_form_eig(1.5)
        linearized = linearized_information_gain(problem, LINE_DESIGN)
        self.assertAlmostEqual(linearized.value, exact, delta=1e-8)
        self.assertEqual(linearized.model_evaluations, 2 * 2 * 3)
        laplace = laplace_information_gain(problem, LINE_DESIGN, samples=4, seed=1)
        self.assertAlmostEqual(laplace.value, exact, delta=1e-8)
        self.assertLess(laplace.standard_error, 1e-8)
        self.assertEqual(laplace.model_evaluations, 4 * 2 * 2 * 3)

    def test_nested_monte_carlo_matches_closed_form(self) -> None:
        """Fresh-inner and reused-sample estimators agree within 4 SE."""
        problem = line_problem(1.5)
        exact = closed_form_eig(1.5)
        fresh = nested_monte_carlo_information_gain(
            problem, LINE_DESIGN, outer=300, inner=60, seed=3
        )
        self.assertLess(abs(fresh.value - exact), 4.0 * fresh.standard_error)
        self.assertEqual(fresh.model_evaluations, 300 * (1 + 60) * 3)
        self.assertEqual(fresh.bias, "upward")
        reused = nested_monte_carlo_information_gain(
            problem, LINE_DESIGN, outer=800, reuse_samples=True, seed=3
        )
        self.assertLess(abs(reused.value - exact), 4.0 * reused.standard_error)
        self.assertEqual(reused.model_evaluations, 800 * 3)
        self.assertEqual(reused.inner_samples, 800)
        self.assertEqual(reused.bias, "downward")

    def test_nested_monte_carlo_bias_directions(self) -> None:
        """With EIG ~ 6.2 nats, fresh M=20 overshoots; reuse stays <= log N."""
        problem = line_problem(0.1)
        exact = closed_form_eig(0.1)
        fresh = nested_monte_carlo_information_gain(
            problem, LINE_DESIGN, outer=300, inner=20, seed=3
        )
        self.assertGreater(fresh.value - 3.0 * fresh.standard_error, exact)
        reused = nested_monte_carlo_information_gain(
            problem, LINE_DESIGN, outer=200, reuse_samples=True, seed=3
        )
        self.assertLess(reused.value + 3.0 * reused.standard_error, exact)
        self.assertLessEqual(reused.value, math.log(200))

    def test_information_gain_increases_as_noise_decreases(self) -> None:
        """Closed form, linearized, and CRN nested MC are monotone in noise."""
        sds = (2.0, 1.0, 0.5, 0.25)
        closed = [closed_form_eig(sd) for sd in sds]
        linearized = [
            linearized_information_gain(line_problem(sd), LINE_DESIGN).value
            for sd in sds
        ]
        nested = [
            nested_monte_carlo_information_gain(
                line_problem(sd), LINE_DESIGN, outer=300, reuse_samples=True, seed=11
            ).value
            for sd in sds
        ]
        for series in (closed, linearized, nested):
            for low, high in zip(series, series[1:], strict=False):
                self.assertGreater(high, low)

    def test_greedy_selection_picks_extreme_points_for_a_slope(self) -> None:
        """Slope-only: always x = 1; intercept and slope: -1 then 1."""
        prior = JointInput.independent(RandomInput("b", Normal(0.0, 1.0)))
        model = ProbabilisticModel(prior, slope_model, (QuantityOfInterest("y", "y"),))
        slope = BayesianDesignProblem(model, (NoiseModel("y", standard_deviation=1.0),))
        grid = [{"x": 0.25 * i} for i in range(5)]
        result = greedy_design(slope, grid, 3, method="linearized")
        self.assertEqual(result.indices, (4, 4, 4))
        self.assertAlmostEqual(result.information_gains[-1], 0.5 * math.log(4.0), 7)

        problem = line_problem(1.0)
        candidates = [{"x": -1.0 + 0.25 * i} for i in range(9)]
        both = greedy_design(problem, candidates, 2, method="linearized")
        self.assertEqual({c["x"] for c in both.selected}, {-1.0, 1.0})
        self.assertAlmostEqual(
            math.fsum(both.increments), both.information_gains[-1], places=12
        )
        expected = linear_gaussian_information_gain(
            ((1.0, -1.0), (1.0, 1.0)), LINE_PRIOR_COVARIANCE, noise_variances=(1, 1)
        )
        self.assertAlmostEqual(both.information_gains[-1], expected, delta=1e-7)

    def test_greedy_decay_design_finds_zero_and_inverse_rate(self) -> None:
        """Laplace-greedy on the decay model picks t = 0 and t = 1/k."""
        problem = decay_problem()
        candidates = [{"t": 0.5 * i} for i in range(13)]
        result = greedy_design(problem, candidates, 2, method="laplace", samples=8)
        self.assertEqual([c["t"] for c in result.selected], [0.0, 2.0])
        self.assertEqual(result.model_evaluations, 13 * 8 * 4 * (1 + 2))
        planned = greedy_design(
            problem, candidates, 1, baseline=[{"t": 0.0}], method="linearized"
        )
        self.assertEqual(planned.selected[0]["t"], 2.0)
        self.assertGreater(planned.baseline_information_gain, 0.0)
        nested = expected_information_gain(
            problem,
            [{"t": 0.0}, {"t": 2.0}],
            method="nested",
            outer=400,
            seed=2,
        )
        laplace = laplace_information_gain(
            problem, [{"t": 0.0}, {"t": 2.0}], samples=32, seed=2
        )
        tolerance = 4.0 * math.hypot(nested.standard_error, laplace.standard_error)
        self.assertLess(abs(nested.value - laplace.value), tolerance + 0.05)

    def test_problem_from_calibration_and_validation(self) -> None:
        """Calibration problems convert; invalid declarations are rejected."""
        prior = JointInput.independent(
            RandomInput("a", Normal(0.0, 1.0)), RandomInput("b", Normal(0.0, 2.0))
        )
        model = ProbabilisticModel(prior, line_model, (QuantityOfInterest("y", "y"),))
        calibration = CalibrationProblem(
            model,
            (Experiment({"y": 0.2}, {"x": 0.0}),),
            (NoiseModel("y", standard_deviation=1.5),),
        )
        problem = BayesianDesignProblem.from_calibration(calibration)
        self.assertEqual(problem.parameter_names, ("a", "b"))
        self.assertAlmostEqual(
            linearized_information_gain(problem, LINE_DESIGN).value,
            closed_form_eig(1.5),
            delta=1e-8,
        )
        unknown = CalibrationProblem(
            model,
            (Experiment({"y": 0.2}, {"x": 0.0}),),
            (NoiseModel("y", standard_deviation_prior=LogNormal(0.0, 0.5)),),
        )
        with self.assertRaises(ValueError):
            BayesianDesignProblem.from_calibration(unknown)
        with self.assertRaises(ValueError):
            BayesianDesignProblem(
                model, (NoiseModel("y", standard_deviation=1.0, degrees_of_freedom=4),)
            )
        with self.assertRaises(ValueError):
            problem.predictions((0.0, 1.0), [{"a": 1.0}])
        with self.assertRaises(ValueError):
            problem.predictions((0.0, 1.0), [])
        with self.assertRaises(ValueError):
            greedy_design(problem, LINE_DESIGN, 1, method="unknown")
        with self.assertRaises(ValueError):
            greedy_design(problem, LINE_DESIGN, 4, allow_repeats=False)


if __name__ == "__main__":
    unittest.main()
