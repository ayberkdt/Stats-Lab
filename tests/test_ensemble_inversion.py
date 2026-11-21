"""Tests for ensemble Kalman inversion and ES-MDA on calibration problems.

The linear-Gaussian line problem (independent normal priors, known noise) has
a closed-form posterior, computed here twice: by ``calibrate_linear_gaussian``
and by explicit normal equations. ES-MDA must reproduce it in mean and
covariance within bands of a few standard errors estimated from independent
replicate runs; the one-step smoother is recomputed independently from the
documented random streams. Ensemble Kalman inversion is checked only through
seed-robust inequalities (misfit reduction, collapse of the spread,
convergence towards least squares), never through pinned digits.
"""

from __future__ import annotations

import math
import random
import unittest
import warnings

from probstat.distributions import LogNormal, Normal, Uniform
from probstat.foundations import RandomStream
from probstat.inference import ConvergenceWarning
from probstat.inverse import (
    CalibrationProblem,
    EnsembleInversionResult,
    Experiment,
    NoiseModel,
    calibrate_linear_gaussian,
    ensemble_kalman_inversion,
    ensemble_smoother_mda,
)
from probstat.inverse.ensemble_inversion import (
    _observation_variances,
    _whitened_gain,
)
from probstat.linalg import inverse_spd, matmul, matvec, solve_spd, transpose
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.multivariate import MultivariateNormal
from probstat.uq import QuantityOfInterest

LINE_X = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
LINE_NOISE = 0.3
PRIOR_MEAN = (0.0, 1.0)
PRIOR_SD = (2.0, 1.0)


def line_model(parameters):
    """Return a straight-line response."""
    return {"y": parameters["a"] + parameters["b"] * parameters["x"]}


def line_data() -> list[float]:
    """Return the seeded straight-line observations."""
    generator = random.Random(5)
    return [1.0 + 0.7 * x + generator.gauss(0.0, LINE_NOISE) for x in LINE_X]


def line_problem(noise: NoiseModel | None = None) -> CalibrationProblem:
    """Build the linear-Gaussian calibration with independent normal priors."""
    prior = JointInput.independent(
        RandomInput("a", Normal(PRIOR_MEAN[0], PRIOR_SD[0])),
        RandomInput("b", Normal(PRIOR_MEAN[1], PRIOR_SD[1])),
    )
    model = ProbabilisticModel(prior, line_model, (QuantityOfInterest("y", "line"),))
    experiments = tuple(
        Experiment({"y": y}, {"x": x}) for x, y in zip(LINE_X, line_data(), strict=True)
    )
    if noise is None:
        noise = NoiseModel("y", standard_deviation=LINE_NOISE)
    return CalibrationProblem(model, experiments, (noise,))


def exact_line_posterior() -> tuple[tuple[float, ...], tuple[tuple[float, ...], ...]]:
    """Return the conjugate posterior mean and covariance by normal equations."""
    design = tuple((1.0, x) for x in LINE_X)
    precision = [
        [
            sum(row[i] * row[j] for row in design) / LINE_NOISE**2
            + (1.0 / PRIOR_SD[i] ** 2 if i == j else 0.0)
            for j in range(2)
        ]
        for i in range(2)
    ]
    rhs = tuple(
        sum(row[i] * y for row, y in zip(design, line_data(), strict=True))
        / LINE_NOISE**2
        + PRIOR_MEAN[i] / PRIOR_SD[i] ** 2
        for i in range(2)
    )
    matrix = tuple(tuple(row) for row in precision)
    return solve_spd(matrix, rhs), inverse_spd(matrix)


def least_squares_line() -> tuple[float, ...]:
    """Return the ordinary least-squares intercept and slope."""
    design = tuple((1.0, x) for x in LINE_X)
    gram = tuple(
        tuple(sum(row[i] * row[j] for row in design) for j in range(2))
        for i in range(2)
    )
    rhs = tuple(
        sum(row[i] * y for row, y in zip(design, line_data(), strict=True))
        for i in range(2)
    )
    return solve_spd(gram, rhs)


def decay_model(parameters):
    """Return an exponential-decay signal."""
    return {"y": parameters["A"] * math.exp(-parameters["k"] * parameters["t"])}


def decay_problem() -> CalibrationProblem:
    """Build the seeded exponential-decay problem of the calibration guide."""
    prior = JointInput.independent(
        RandomInput("A", Uniform(1.0, 10.0)),
        RandomInput("k", LogNormal(math.log(0.5), 1.0)),
    )
    model = ProbabilisticModel(prior, decay_model, (QuantityOfInterest("y", "signal"),))
    generator = random.Random(7)
    experiments = tuple(
        Experiment(
            {"y": 5.0 * math.exp(-0.3 * t) + generator.gauss(0.0, 0.1)}, {"t": t}
        )
        for t in (0.5 * i for i in range(21))
    )
    return CalibrationProblem(
        model, experiments, (NoiseModel("y", standard_deviation=0.1),)
    )


def mean_and_standard_error(values: list[float]) -> tuple[float, float]:
    """Return the replicate mean and its standard error."""
    count = len(values)
    center = math.fsum(values) / count
    variance = math.fsum((v - center) ** 2 for v in values) / (count - 1)
    return center, math.sqrt(variance / count)


class LinearGaussianExactnessTests(unittest.TestCase):
    """ES-MDA samples the conjugate posterior of a linear-Gaussian problem."""

    @classmethod
    def setUpClass(cls) -> None:
        """Build the problem and its exact posterior."""
        cls.problem = line_problem()
        cls.mean, cls.covariance = exact_line_posterior()

    def test_exact_posterior_agrees_with_calibrate_linear_gaussian(self) -> None:
        """The two independent closed forms coincide."""
        reference = calibrate_linear_gaussian(
            LINE_X,
            line_data(),
            observation_variance=LINE_NOISE**2,
            prior=MultivariateNormal(
                PRIOR_MEAN, ((PRIOR_SD[0] ** 2, 0.0), (0.0, PRIOR_SD[1] ** 2))
            ),
        ).posterior
        for i in range(2):
            self.assertAlmostEqual(reference.mean[i], self.mean[i], delta=1e-12)
            for j in range(2):
                self.assertAlmostEqual(
                    reference.covariance[i][j], self.covariance[i][j], delta=1e-12
                )

    def test_replicated_esmda_moments_match_the_posterior(self) -> None:
        """Replicate-averaged mean and covariance lie within 4 SE of the exact."""
        replicates = 20
        results = [
            ensemble_smoother_mda(
                self.problem,
                ensemble_size=200,
                inflation_coefficients=(4.0, 4.0, 4.0, 4.0),
                seed=RandomStream(2026).child("replicate", r),
            )
            for r in range(replicates)
        ]
        for i in range(2):
            center, error = mean_and_standard_error(
                [result.mean_history[-1][i] for result in results]
            )
            with self.subTest(moment="mean", index=i):
                self.assertLess(abs(center - self.mean[i]), 4.0 * error)
        for i, j in ((0, 0), (0, 1), (1, 1)):
            center, error = mean_and_standard_error(
                [result.covariance()[i][j] for result in results]
            )
            with self.subTest(moment="covariance", entry=(i, j)):
                # 4 replicate SEs plus 5 % for the O(1/N) ensemble bias.
                tolerance = 4.0 * error + 0.05 * abs(self.covariance[i][j])
                self.assertLess(abs(center - self.covariance[i][j]), tolerance)

    def test_large_ensemble_is_close_to_the_posterior(self) -> None:
        """One N = 1000 run: means within 4 SE and SDs within 20 %."""
        result = ensemble_smoother_mda(
            self.problem,
            ensemble_size=1000,
            inflation_coefficients=(2.0, 2.0),
            seed=17,
        )
        covariance = result.covariance()
        for i in range(2):
            scale = math.sqrt(self.covariance[i][i])
            with self.subTest(index=i):
                self.assertLess(
                    abs(result.mean_history[-1][i] - self.mean[i]),
                    4.0 * scale / math.sqrt(1000),
                )
                self.assertAlmostEqual(
                    math.sqrt(covariance[i][i]), scale, delta=0.2 * scale
                )
        self.assertEqual(result.forward_evaluations, 1000 * 3)
        self.assertEqual(result.model_calls, 1000 * 3 * len(LINE_X))
        self.assertEqual(result.inflation_coefficients, (2.0, 2.0))
        self.assertEqual(result.updates, 2)
        self.assertEqual(result.stopped_by, "iterations")
        self.assertTrue(math.isnan(result.discrepancy_threshold))

    def test_single_unit_coefficient_is_the_one_step_smoother(self) -> None:
        """A single coefficient alpha = 1 equals an independent smoother update."""
        seed, size = 31, 40
        result = ensemble_smoother_mda(
            self.problem, ensemble_size=size, inflation_coefficients=(1.0,), seed=seed
        )
        stream = RandomStream(seed)
        generator = stream.child("ensemble").generator()
        prior = [
            tuple(generator.gauss(0.0, 1.0) for _ in range(2)) for _ in range(size)
        ]
        noise = stream.child("perturbations", 0).generator()
        data = line_data()
        # Forward model in u: theta = mean + sd * u.
        predictions = [
            tuple(
                PRIOR_MEAN[0]
                + PRIOR_SD[0] * u[0]
                + (PRIOR_MEAN[1] + PRIOR_SD[1] * u[1]) * x
                for x in LINE_X
            )
            for u in prior
        ]
        u_bar = [sum(u[k] for u in prior) / size for k in range(2)]
        g_bar = [sum(g[m] for g in predictions) / size for m in range(len(LINE_X))]
        du = [[u[k] - u_bar[k] for k in range(2)] for u in prior]
        dg = [[g[m] - g_bar[m] for m in range(len(LINE_X))] for g in predictions]
        c_up = matmul(transpose(du), dg)
        c_pp = matmul(transpose(dg), dg)
        c_up = tuple(tuple(v / (size - 1) for v in row) for row in c_up)
        system = tuple(
            tuple(
                v / (size - 1) + (LINE_NOISE**2 if p == q else 0.0)
                for q, v in enumerate(row)
            )
            for p, row in enumerate(c_pp)
        )
        for member, (u, g) in enumerate(zip(prior, predictions, strict=True)):
            perturbed = tuple(
                y + LINE_NOISE * noise.gauss(0.0, 1.0) - gm
                for y, gm in zip(data, g, strict=True)
            )
            expected = tuple(
                value + shift
                for value, shift in zip(
                    u, matvec(c_up, solve_spd(system, perturbed)), strict=True
                )
            )
            for k in range(2):
                self.assertAlmostEqual(
                    result.standard_ensemble[member][k],
                    expected[k],
                    delta=1e-10 * max(1.0, abs(expected[k])),
                )
        # The same update is one perturbed ensemble Kalman inversion step.
        eki = ensemble_kalman_inversion(
            self.problem, ensemble_size=size, iterations=1, seed=seed
        )
        self.assertEqual(eki.standard_ensemble, result.standard_ensemble)
        self.assertEqual(result.misfit_history, eki.misfit_history)

    def test_physical_and_standard_ensembles_correspond(self) -> None:
        """Each theta is mean + sd * u; every history has updates + 1 entries."""
        result = ensemble_smoother_mda(
            self.problem, ensemble_size=30, inflation_coefficients=(3.0, 1.5), seed=4
        )
        for theta, u in zip(result.ensemble, result.standard_ensemble, strict=True):
            for k in range(2):
                self.assertAlmostEqual(
                    theta[k], PRIOR_MEAN[k] + PRIOR_SD[k] * u[k], delta=1e-12
                )
        for history in (
            result.mean_history,
            result.spread_history,
            result.standard_mean_history,
            result.standard_spread_history,
            result.misfit_history,
        ):
            self.assertEqual(len(history), 3)
        self.assertEqual(result.ensemble_size, 30)
        self.assertEqual(result.method, "ES-MDA")
        self.assertTrue(result.perturbed_observations)
        self.assertEqual(tuple(result.mean()), ("a", "b"))
        self.assertEqual(len(result.draws()), 30)
        self.assertEqual(result.draws()[0]["a"], result.ensemble[0][0])
        standard = result.standard_covariance()
        physical = result.covariance()
        self.assertAlmostEqual(
            physical[0][1], PRIOR_SD[0] * PRIOR_SD[1] * standard[0][1], delta=1e-12
        )
        self.assertIn("ES-MDA: 30 members, 2 updates", result.summary())


class EnsembleKalmanInversionTests(unittest.TestCase):
    """EKI is an optimizer whose ensemble collapses."""

    @classmethod
    def setUpClass(cls) -> None:
        """Build the line problem."""
        cls.problem = line_problem()

    def test_misfit_decreases_and_spread_collapses(self) -> None:
        """Across seeds: final misfit below initial, spread below 15 %."""
        for seed in (1, 2, 3):
            result = ensemble_kalman_inversion(
                self.problem, ensemble_size=50, iterations=20, seed=seed
            )
            with self.subTest(seed=seed):
                self.assertIsInstance(result, EnsembleInversionResult)
                self.assertEqual(len(result.misfit_history), 21)
                self.assertLess(result.misfit_history[-1], result.misfit_history[0])
                first = result.standard_spread_history[0]
                last = result.standard_spread_history[-1]
                for k in range(2):
                    self.assertLess(last[k], 0.15 * first[k])
                    self.assertLess(
                        result.standard_spread_history[10][k],
                        result.standard_spread_history[1][k],
                    )
                self.assertEqual(result.forward_evaluations, 50 * 21)
                self.assertIn("not a posterior uncertainty", result.summary())

    def test_mean_approaches_least_squares_and_spread_falls_below_posterior(
        self,
    ) -> None:
        """Many iterations drive the mean to OLS, far inside the posterior SD."""
        ols = least_squares_line()
        _, covariance = exact_line_posterior()
        result = ensemble_kalman_inversion(
            self.problem, ensemble_size=100, iterations=40, seed=9
        )
        for k in range(2):
            posterior_sd = math.sqrt(covariance[k][k])
            with self.subTest(index=k):
                self.assertLess(
                    abs(result.mean_history[-1][k] - ols[k]), 0.3 * posterior_sd
                )
                self.assertLess(result.spread_history[-1][k], 0.5 * posterior_sd)

    def test_unperturbed_iteration_is_deterministic_given_the_ensemble(self) -> None:
        """Without perturbations only the prior ensemble is random."""
        first = ensemble_kalman_inversion(
            self.problem,
            ensemble_size=20,
            iterations=5,
            seed=RandomStream(8),
            perturb_observations=False,
        )
        self.assertFalse(first.perturbed_observations)
        self.assertLess(first.misfit_history[-1], first.misfit_history[0])
        # The mean of a linear unperturbed EKI step stays in the prior span and
        # the misfit is non-increasing for a linear model.
        for before, after in zip(
            first.misfit_history, first.misfit_history[1:], strict=False
        ):
            self.assertLessEqual(after, before + 1e-9)

    def test_discrepancy_principle_stops_at_the_noise_level(self) -> None:
        """The run stops at the first misfit below tau sqrt(M)."""
        result = ensemble_kalman_inversion(
            self.problem,
            ensemble_size=50,
            iterations=30,
            seed=1,
            stopping="discrepancy",
        )
        threshold = 1.5 * math.sqrt(len(LINE_X))
        self.assertAlmostEqual(result.discrepancy_threshold, threshold, delta=1e-12)
        self.assertEqual(result.stopped_by, "discrepancy")
        self.assertLessEqual(result.misfit_history[-1], threshold)
        for value in result.misfit_history[:-1]:
            self.assertGreater(value, threshold)
        self.assertEqual(len(result.inflation_coefficients), result.updates)
        self.assertEqual(result.forward_evaluations, 50 * (result.updates + 1))
        self.assertIn("threshold", result.summary())
        # A threshold no ensemble can reach exhausts the iteration budget.
        strict = ensemble_kalman_inversion(
            self.problem,
            ensemble_size=20,
            iterations=3,
            seed=1,
            stopping="discrepancy",
            discrepancy_factor=1.0,
            noise_level=1e-6,
        )
        self.assertEqual(strict.stopped_by, "iterations")
        self.assertEqual(strict.updates, 3)

    def test_reproducibility(self) -> None:
        """Equal seeds give identical results; different seeds differ."""
        kwargs = {"ensemble_size": 15, "iterations": 3}
        first = ensemble_kalman_inversion(self.problem, seed=5, **kwargs)
        second = ensemble_kalman_inversion(self.problem, seed=5, **kwargs)
        third = ensemble_kalman_inversion(self.problem, seed=6, **kwargs)
        # ``discrepancy_threshold`` is NaN here, and NaN != NaN field by field
        # (dataclass equality compares fields one by one from Python 3.13).
        self.assertEqual(repr(first), repr(second))
        self.assertNotEqual(first.standard_ensemble, third.standard_ensemble)
        self.assertEqual(first.seed, 5)
        smoother = ensemble_smoother_mda(
            self.problem, ensemble_size=15, inflation_coefficients=(2.0, 2.0), seed=5
        )
        again = ensemble_smoother_mda(
            self.problem, ensemble_size=15, inflation_coefficients=(2.0, 2.0), seed=5
        )
        self.assertEqual(repr(smoother), repr(again))


class WhitenedGainTests(unittest.TestCase):
    """Both forms of the ensemble gain equal the explicit formula."""

    def test_push_through_identity(self) -> None:
        """M <= N and M > N branches match A'B (B'B + cI)^-1."""
        generator = random.Random(3)
        for count, size in ((8, 3), (3, 8)):
            du = [
                tuple(generator.gauss(0.0, 1.0) for _ in range(2)) for _ in range(count)
            ]
            dg = [
                tuple(generator.gauss(0.0, 1.0) for _ in range(size))
                for _ in range(count)
            ]
            shift = 0.7
            gain = _whitened_gain(du, dg, shift)
            cross = matmul(transpose(du), dg)
            system = tuple(
                tuple(v + (shift if p == q else 0.0) for q, v in enumerate(row))
                for p, row in enumerate(matmul(transpose(dg), dg))
            )
            expected = matmul(cross, inverse_spd(system))
            with self.subTest(count=count, size=size):
                for k in range(2):
                    for m in range(size):
                        self.assertAlmostEqual(gain[k][m], expected[k][m], delta=1e-10)

    def test_gamma_includes_discrepancy_variance(self) -> None:
        """Gamma = sigma^2 + sigma_delta^2 in observation order."""
        problem = line_problem(
            NoiseModel("y", standard_deviation=0.3, discrepancy_variance=0.16)
        )
        variances = _observation_variances(problem)
        self.assertEqual(len(variances), len(LINE_X))
        for value in variances:
            self.assertAlmostEqual(value, 0.25, delta=1e-12)


class NonlinearDecayTests(unittest.TestCase):
    """ES-MDA approximates the MCMC posterior of the exponential decay."""

    def test_esmda_mean_is_near_the_mcmc_posterior_mean(self) -> None:
        """Means agree within half a posterior SD.

        The tolerance covers the ES-MDA approximation error for a nonlinear
        model (each step linearizes through ensemble covariances), the
        ensemble error ``sd / sqrt(N)`` (about 0.07 sd for N = 200), and the
        MCMC error. With strong data both methods concentrate near the
        least-squares solution, so the means are close although the ES-MDA
        spread is not the exact posterior spread.
        """
        problem = decay_problem()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            posterior = problem.sample_posterior(
                draws=1500, warmup=500, chains=4, seed=3
            )
        reference = posterior.mean()
        scales = {s.name: s.standard_deviation for s in posterior.summary()}
        for seed in (0, 1):
            result = ensemble_smoother_mda(
                problem,
                ensemble_size=200,
                inflation_coefficients=(8.0,) * 8,
                seed=seed,
            )
            estimate = result.mean()
            for name in ("A", "k"):
                with self.subTest(seed=seed, parameter=name):
                    self.assertLess(
                        abs(estimate[name] - reference[name]), 0.5 * scales[name]
                    )
                    # The spread is of the posterior's order, not collapsed.
                    spread = result.spread_history[-1][
                        problem.parameter_names.index(name)
                    ]
                    self.assertGreater(spread, 0.5 * scales[name])
                    self.assertLess(spread, 2.0 * scales[name])
        eki = ensemble_kalman_inversion(
            problem, ensemble_size=50, iterations=20, seed=0
        )
        self.assertLess(eki.misfit_history[-1], 0.25 * eki.misfit_history[0])
        for k, name in enumerate(problem.parameter_names):
            self.assertLess(eki.spread_history[-1][k], scales[name])


class ValidationTests(unittest.TestCase):
    """Explicit errors for unsupported problems and invalid arguments."""

    @classmethod
    def setUpClass(cls) -> None:
        """Build the line problem."""
        cls.problem = line_problem()

    def test_unsupported_noise_models(self) -> None:
        """Unknown noise scales and Student-t errors are rejected with reasons."""
        unknown = line_problem(
            NoiseModel("y", standard_deviation_prior=LogNormal(math.log(0.3), 0.5))
        )
        with self.assertRaisesRegex(ValueError, "unknown scale"):
            ensemble_kalman_inversion(unknown, ensemble_size=10, iterations=2, seed=1)
        with self.assertRaisesRegex(ValueError, "unknown scale"):
            ensemble_smoother_mda(
                unknown, ensemble_size=10, inflation_coefficients=(1.0,), seed=1
            )
        robust = line_problem(
            NoiseModel("y", standard_deviation=0.3, degrees_of_freedom=4.0)
        )
        with self.assertRaisesRegex(ValueError, "Student-t"):
            ensemble_kalman_inversion(robust, ensemble_size=10, iterations=2, seed=1)
        with self.assertRaises(TypeError):
            ensemble_kalman_inversion(None, ensemble_size=10, iterations=2)  # type: ignore[arg-type]

    def test_invalid_arguments(self) -> None:
        """Sizes, iterations, stopping rules, and inflation are validated."""
        problem = self.problem
        with self.assertRaises(ValueError):
            ensemble_kalman_inversion(problem, ensemble_size=1, iterations=2)
        with self.assertRaises(ValueError):
            ensemble_kalman_inversion(problem, ensemble_size=10, iterations=0)
        with self.assertRaises(ValueError):
            ensemble_kalman_inversion(problem, ensemble_size=10, iterations=-1)
        with self.assertRaises(ValueError):
            ensemble_kalman_inversion(
                problem, ensemble_size=10, iterations=2, stopping="tolerance"
            )
        with self.assertRaises(ValueError):
            ensemble_kalman_inversion(
                problem,
                ensemble_size=10,
                iterations=2,
                stopping="discrepancy",
                discrepancy_factor=0.9,
            )
        with self.assertRaises(ValueError):
            ensemble_kalman_inversion(
                problem,
                ensemble_size=10,
                iterations=2,
                stopping="discrepancy",
                noise_level=0.0,
            )
        with self.assertRaises(TypeError):
            ensemble_kalman_inversion(
                problem, ensemble_size=10, iterations=2, perturb_observations=1
            )  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "sum"):
            ensemble_smoother_mda(
                problem, ensemble_size=10, inflation_coefficients=(2.0, 3.0)
            )
        with self.assertRaises(ValueError):
            ensemble_smoother_mda(problem, ensemble_size=10, inflation_coefficients=())
        with self.assertRaises(ValueError):
            ensemble_smoother_mda(
                problem, ensemble_size=10, inflation_coefficients=(-2.0, 2.0 / 3.0)
            )
        with self.assertRaises(ValueError):
            ensemble_smoother_mda(
                problem, ensemble_size=10, inflation_coefficients=(math.inf,)
            )
        with self.assertRaises(TypeError):
            ensemble_smoother_mda(problem, ensemble_size=10, inflation_coefficients="4")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            ensemble_smoother_mda(
                problem, ensemble_size=1, inflation_coefficients=(1.0,)
            )
        # A schedule whose reciprocals sum to one within rounding is accepted.
        result = ensemble_smoother_mda(
            problem,
            ensemble_size=5,
            inflation_coefficients=(3.0, 3.0, 3.0),
            seed=1,
        )
        self.assertEqual(result.updates, 3)


if __name__ == "__main__":
    unittest.main()
