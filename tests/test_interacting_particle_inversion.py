"""Tests for interacting Langevin, unscented Kalman and consensus inversion.

The straight-line problem of ``tests/test_ensemble_inversion.py`` (normal
priors, known noise) has a closed-form posterior, recomputed here by normal
equations. It is the oracle for all three methods:

* the corrected interacting Langevin sampler must reproduce its mean and
  covariance within replicate standard errors with a small ensemble, the
  uncorrected sampler must show the derived covariance deficit, and neither
  may collapse as ensemble Kalman inversion does;
* unscented Kalman inversion must converge to it to rounding error, with the
  derived precision recursion and halving of the error;
* consensus-based sampling must hold it as a fixed point within Monte Carlo
  error, and the optimization mode must concentrate at the MAP point.

The exponential-decay problem is compared with the MCMC posterior using
tolerances that leave room for the approximation error of each method.
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
    ConsensusSamplingResult,
    Experiment,
    InteractingLangevinResult,
    NoiseModel,
    UnscentedInversionResult,
    consensus_based_sampling,
    ensemble_kalman_inversion,
    interacting_langevin_sampler,
    unscented_kalman_inversion,
)
from probstat.inverse.consensus_sampling import _tempered_weights, _weights
from probstat.inverse.unscented_inversion import _sigma_points
from probstat.linalg import inverse_spd, solve_spd
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.uq import QuantityOfInterest

LINE_X = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
LINE_NOISE = 0.3
PRIOR_MEAN = (0.0, 1.0)
PRIOR_SD = (2.0, 1.0)
ENTRIES = ((0, 0), (0, 1), (1, 1))


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


def standard_design() -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...]]:
    """Return the whitened map ``u -> A u + b`` and the whitened data.

    ``theta = mean + sd * u``, so row ``x`` of ``A`` is
    ``(sd_a, sd_b x) / noise`` and ``b = (mean_a + mean_b x) / noise``; the
    returned data already has ``b`` subtracted.
    """
    matrix = tuple(
        (PRIOR_SD[0] / LINE_NOISE, PRIOR_SD[1] * x / LINE_NOISE) for x in LINE_X
    )
    data = tuple(
        (y - PRIOR_MEAN[0] - PRIOR_MEAN[1] * x) / LINE_NOISE
        for x, y in zip(LINE_X, line_data(), strict=True)
    )
    return matrix, data


def standard_posterior() -> tuple[
    tuple[float, ...], tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...]
]:
    """Return posterior mean, covariance and precision in the coordinates u."""
    matrix, data = standard_design()
    precision = tuple(
        tuple(
            sum(row[i] * row[j] for row in matrix) + (1.0 if i == j else 0.0)
            for j in range(2)
        )
        for i in range(2)
    )
    rhs = tuple(
        sum(row[i] * y for row, y in zip(matrix, data, strict=True)) for i in range(2)
    )
    return solve_spd(precision, rhs), inverse_spd(precision), precision


def physical_posterior() -> tuple[tuple[float, ...], tuple[tuple[float, ...], ...]]:
    """Return the conjugate posterior mean and covariance of ``(a, b)``."""
    mean, covariance, _ = standard_posterior()
    return (
        tuple(PRIOR_MEAN[i] + PRIOR_SD[i] * mean[i] for i in range(2)),
        tuple(
            tuple(PRIOR_SD[i] * PRIOR_SD[j] * covariance[i][j] for j in range(2))
            for i in range(2)
        ),
    )


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


class ExactPosteriorTests(unittest.TestCase):
    """The oracle itself is consistent."""

    def test_physical_posterior_matches_the_documented_values(self) -> None:
        """The conjugate posterior equals the values quoted in the guide."""
        mean, covariance = physical_posterior()
        self.assertAlmostEqual(mean[0], 0.6417, delta=5e-5)
        self.assertAlmostEqual(mean[1], 0.7875, delta=5e-5)
        self.assertAlmostEqual(covariance[0][0], 0.04099, delta=5e-6)
        self.assertAlmostEqual(covariance[1][1], 0.01260, delta=5e-6)
        self.assertAlmostEqual(covariance[0][1], -0.01885, delta=5e-6)


class InteractingLangevinLinearTests(unittest.TestCase):
    """The corrected sampler is exact for a linear map; the uncorrected is not."""

    replicates = 8
    size = 6
    steps = 1000

    @classmethod
    def setUpClass(cls) -> None:
        """Run seeded replicates with and without the correction."""
        cls.problem = line_problem()
        cls.mean, cls.covariance = physical_posterior()
        cls.runs = {
            flag: [
                interacting_langevin_sampler(
                    cls.problem,
                    ensemble_size=cls.size,
                    steps=cls.steps,
                    seed=RandomStream(11).child("replicate", r),
                    correction=flag,
                )
                for r in range(cls.replicates)
            ]
            for flag in (True, False)
        }

    def test_corrected_sampler_matches_the_conjugate_posterior(self) -> None:
        """Time-averaged mean and covariance lie within 4 SE of the exact."""
        results = self.runs[True]
        for i in range(2):
            center, error = mean_and_standard_error(
                [result.posterior_mean[i] for result in results]
            )
            with self.subTest(moment="mean", index=i):
                self.assertLess(abs(center - self.mean[i]), 4.0 * error)
                # The replicate error is small enough to make this a test.
                self.assertLess(error, 0.1 * math.sqrt(self.covariance[i][i]))
        for i, j in ENTRIES:
            center, error = mean_and_standard_error(
                [result.posterior_covariance[i][j] for result in results]
            )
            with self.subTest(moment="covariance", entry=(i, j)):
                # 4 replicate SEs plus 5 % for the O(h) discretization bias.
                tolerance = 4.0 * error + 0.05 * abs(self.covariance[i][j])
                self.assertLess(abs(center - self.covariance[i][j]), tolerance)
                self.assertLess(error, 0.06 * abs(self.covariance[i][j]))

    def test_uncorrected_sampler_underestimates_the_covariance(self) -> None:
        """Without the correction the pooled covariance is (J-d-1)/J of exact.

        The invariant density of the uncorrected dynamics is the product
        posterior times ``det(C)^-(d+1)/2``: the ensemble covariance is
        Wishart with ``J - d - 2`` degrees of freedom and the ensemble mean
        keeps covariance ``Sigma / J``, so the pooled covariance is
        ``(J - d - 1)/J = 0.5`` of the posterior for ``J = 6``, ``d = 2``.
        The explicit Euler step inflates the measured ratio by a few
        percent, covered by the 0.1 allowance.
        """
        expected = (self.size - 3) / self.size
        for i in (0, 1):
            ratios = [
                result.posterior_covariance[i][i] / self.covariance[i][i]
                for result in self.runs[False]
            ]
            center, error = mean_and_standard_error(ratios)
            with self.subTest(index=i):
                self.assertLess(center + 6.0 * error, 1.0)
                self.assertLess(center, 0.75)
                self.assertLess(abs(center - expected), 4.0 * error + 0.1)

    def test_ensemble_covariance_degrees_of_freedom(self) -> None:
        """Time-averaged ensemble variances: unbiased with, deficient without.

        With divisor ``J - 1`` the stationary expectation is the posterior
        variance with the correction and ``(J - d - 2)/(J - 1) = 0.4`` of it
        without (``J = 6``, ``d = 2``).
        """
        for flag, expected, allowance in ((True, 1.0, 0.05), (False, 0.4, 0.1)):
            for i in (0, 1):
                ratios = []
                for result in self.runs[flag]:
                    steps = result.step_history[result.burn_in :]
                    spreads = result.spread_history[result.burn_in :]
                    ratios.append(
                        math.fsum(
                            h * s[i] ** 2 for h, s in zip(steps, spreads, strict=True)
                        )
                        / math.fsum(steps)
                        / self.covariance[i][i]
                    )
                center, error = mean_and_standard_error(ratios)
                with self.subTest(corrected=flag, index=i):
                    self.assertLess(abs(center - expected), 4.0 * error + allowance)

    def test_spread_does_not_collapse_unlike_eki(self) -> None:
        """After burn-in the spread stays of posterior order; EKI's does not."""
        scales = [math.sqrt(self.covariance[i][i]) for i in range(2)]
        result = self.runs[True][0]
        late = result.spread_history[-200:]
        for i in range(2):
            average = math.fsum(s[i] for s in late) / len(late)
            with self.subTest(index=i):
                self.assertGreater(average, 0.6 * scales[i])
                self.assertLess(average, 1.4 * scales[i])
        eki = ensemble_kalman_inversion(
            self.problem, ensemble_size=50, iterations=40, seed=1
        )
        for i in range(2):
            self.assertLess(eki.spread_history[-1][i], 0.3 * scales[i])

    def test_result_bookkeeping(self) -> None:
        """Histories, cost, flags and summaries follow the documented layout."""
        result = self.runs[True][0]
        self.assertIsInstance(result, InteractingLangevinResult)
        self.assertEqual(result.ensemble_size, self.size)
        self.assertEqual(result.steps, self.steps)
        self.assertEqual(result.burn_in, self.steps // 2)
        self.assertTrue(result.corrected)
        self.assertTrue(result.adaptive)
        for history in (
            result.mean_history,
            result.spread_history,
            result.standard_mean_history,
            result.standard_spread_history,
            result.misfit_history,
            result.step_history,
        ):
            self.assertEqual(len(history), self.steps + 1)
        self.assertEqual(result.forward_evaluations, self.size * (self.steps + 1))
        self.assertEqual(result.model_calls, result.forward_evaluations * len(LINE_X))
        self.assertAlmostEqual(
            result.sampling_time,
            math.fsum(result.step_history[result.burn_in :]),
            delta=1e-12,
        )
        self.assertTrue(all(step > 0.0 for step in result.step_history))
        # The first step resolves the stiff prior ensemble.
        self.assertLess(result.step_history[0], 0.01)
        self.assertEqual(tuple(result.mean()), ("a", "b"))
        self.assertEqual(result.covariance(), result.posterior_covariance)
        self.assertEqual(len(result.draws()), self.size)
        for theta, u in zip(result.ensemble, result.standard_ensemble, strict=True):
            for k in range(2):
                self.assertAlmostEqual(
                    theta[k], PRIOR_MEAN[k] + PRIOR_SD[k] * u[k], delta=1e-12
                )
        for k in range(2):
            self.assertAlmostEqual(
                result.posterior_mean[k],
                PRIOR_MEAN[k] + PRIOR_SD[k] * result.standard_posterior_mean[k],
                delta=1e-12,
            )
            self.assertGreater(result.standard_mean_standard_error[k], 0.0)
        self.assertIn("interacting Langevin (corrected)", result.summary())
        self.assertIn("biased low", self.runs[False][0].summary())
        self.assertFalse(self.runs[False][0].corrected)

    def test_prior_ensemble_is_shared_with_eki(self) -> None:
        """The same seed gives the prior ensemble of ensemble Kalman inversion."""
        langevin = interacting_langevin_sampler(
            self.problem, ensemble_size=12, steps=1, seed=9
        )
        eki = ensemble_kalman_inversion(
            self.problem, ensemble_size=12, iterations=1, seed=9
        )
        self.assertEqual(
            langevin.standard_mean_history[0], eki.standard_mean_history[0]
        )
        self.assertEqual(langevin.misfit_history[0], eki.misfit_history[0])


class InteractingLangevinStepTests(unittest.TestCase):
    """One Euler-Maruyama step equals an independent recomputation."""

    def test_step_uses_the_exact_preconditioned_gradient(self) -> None:
        """For a linear map the ensemble difference is ``C grad Phi``.

        The step is recomputed from the documented random streams with the
        analytic gradient ``A'(A u - y) + u`` of the whitened potential
        preconditioned by the ensemble covariance (divisor ``J``), the
        correction ``(d + 1)/J (u_j - m)``, and the noise
        ``sqrt(2 h / J) sum_k (u_k - m) xi_jk``.
        """
        seed, size, step = 23, 7, 1e-3
        problem = line_problem()
        matrix, data = standard_design()
        stream = RandomStream(seed)
        generator = stream.child("ensemble").generator()
        prior = [
            tuple(generator.gauss(0.0, 1.0) for _ in range(2)) for _ in range(size)
        ]
        center = [sum(u[k] for u in prior) / size for k in range(2)]
        deviations = [[u[k] - center[k] for k in range(2)] for u in prior]
        covariance = [
            [sum(d[i] * d[j] for d in deviations) / size for j in range(2)]
            for i in range(2)
        ]
        for correction in (True, False):
            result = interacting_langevin_sampler(
                problem,
                ensemble_size=size,
                steps=1,
                seed=seed,
                step_size=step,
                correction=correction,
            )
            self.assertFalse(result.adaptive)
            self.assertEqual(result.step_history, (step, step))
            noise = stream.child("noise", 0).generator()
            kappa = 3.0 / size if correction else 0.0
            for j, u in enumerate(prior):
                residual = [
                    sum(row[k] * u[k] for k in range(2)) - y
                    for row, y in zip(matrix, data, strict=True)
                ]
                gradient = [
                    sum(row[k] * r for row, r in zip(matrix, residual, strict=True))
                    + u[k]
                    for k in range(2)
                ]
                draws = [noise.gauss(0.0, 1.0) for _ in range(size)]
                for i in range(2):
                    expected = (
                        u[i]
                        - step * sum(covariance[i][k] * gradient[k] for k in range(2))
                        + step * kappa * deviations[j][i]
                        + math.sqrt(2.0 * step / size)
                        * sum(
                            d[i] * xi for d, xi in zip(deviations, draws, strict=True)
                        )
                    )
                    with self.subTest(correction=correction, member=j, index=i):
                        self.assertAlmostEqual(
                            result.standard_ensemble[j][i],
                            expected,
                            delta=1e-9 * max(1.0, abs(expected)),
                        )

    def test_correction_is_the_divergence_of_the_covariance(self) -> None:
        """``div_{u_j} C(U) = (d + 1)/J (u_j - m)`` by central differences."""
        generator = random.Random(3)
        size, dimension = 5, 3
        members = [
            [generator.gauss(0.0, 1.0) for _ in range(dimension)] for _ in range(size)
        ]

        def covariance_row(points: list[list[float]], row: int) -> list[float]:
            """Return row ``row`` of the ensemble covariance (divisor J)."""
            center = [sum(p[k] for p in points) / size for k in range(dimension)]
            return [
                sum((p[row] - center[row]) * (p[k] - center[k]) for p in points) / size
                for k in range(dimension)
            ]

        center = [sum(p[k] for p in members) / size for k in range(dimension)]
        width = 1e-5
        for j in range(size):
            for row in range(dimension):
                divergence = 0.0
                for column in range(dimension):
                    values = []
                    for sign in (1.0, -1.0):
                        shifted = [list(p) for p in members]
                        shifted[j][column] += sign * width
                        values.append(covariance_row(shifted, row)[column])
                    divergence += (values[0] - values[1]) / (2.0 * width)
                expected = (dimension + 1) / size * (members[j][row] - center[row])
                self.assertAlmostEqual(divergence, expected, delta=1e-8)

    def test_reproducibility_and_seed_dependence(self) -> None:
        """Equal seeds give equal ensembles; different seeds do not."""
        problem = line_problem()
        first = interacting_langevin_sampler(problem, ensemble_size=5, steps=20, seed=4)
        second = interacting_langevin_sampler(
            problem, ensemble_size=5, steps=20, seed=4
        )
        other = interacting_langevin_sampler(problem, ensemble_size=5, steps=20, seed=5)
        self.assertEqual(first.standard_ensemble, second.standard_ensemble)
        self.assertEqual(first.step_history, second.step_history)
        self.assertNotEqual(first.standard_ensemble, other.standard_ensemble)
        self.assertEqual(first.seed, 4)
        # Fewer than twenty averaged ensembles: no batch-means error.
        self.assertTrue(all(math.isnan(v) for v in first.standard_mean_standard_error))


class UnscentedInversionLinearTests(unittest.TestCase):
    """The iteration converges to the exact posterior of a linear problem."""

    @classmethod
    def setUpClass(cls) -> None:
        """Run the inversion to a tight tolerance."""
        cls.problem = line_problem()
        cls.result = unscented_kalman_inversion(
            cls.problem, iterations=80, tolerance=1e-13
        )

    def test_fixed_point_is_the_conjugate_posterior(self) -> None:
        """Mean and covariance agree with the closed form to 1e-9 relative."""
        result = self.result
        self.assertIsInstance(result, UnscentedInversionResult)
        self.assertTrue(result.converged)
        mean, covariance = physical_posterior()
        standard_mean, standard_covariance, _ = standard_posterior()
        for i in range(2):
            self.assertAlmostEqual(result.posterior_mean[i], mean[i], delta=1e-9)
            self.assertAlmostEqual(
                result.standard_mean[i], standard_mean[i], delta=1e-9
            )
            for j in range(2):
                self.assertAlmostEqual(
                    result.posterior_covariance[i][j],
                    covariance[i][j],
                    delta=1e-9 * abs(covariance[i][j]),
                )
                self.assertAlmostEqual(
                    result.standard_covariance[i][j],
                    standard_covariance[i][j],
                    delta=1e-9 * abs(standard_covariance[i][j]),
                )
        self.assertEqual(result.covariance(), result.posterior_covariance)
        self.assertAlmostEqual(result.mean()["a"], mean[0], delta=1e-9)

    def test_iterates_follow_the_derived_precision_recursion(self) -> None:
        """``C_n^-1 = (C_{n-1}^-1 + P)/2`` and the matching mean recursion."""
        mean, _, precision = standard_posterior()
        target = tuple(
            sum(precision[i][k] * mean[k] for k in range(2)) for i in range(2)
        )
        current = ((1.0, 0.0), (0.0, 1.0))
        weighted = (0.0, 0.0)
        for n in range(1, 9):
            current = tuple(
                tuple(0.5 * (current[i][j] + precision[i][j]) for j in range(2))
                for i in range(2)
            )
            weighted = tuple(0.5 * (weighted[i] + target[i]) for i in range(2))
            covariance = inverse_spd(current)
            expected_mean = solve_spd(current, weighted)
            for i in range(2):
                with self.subTest(iteration=n, index=i):
                    self.assertAlmostEqual(
                        self.result.standard_spread_history[n][i],
                        math.sqrt(covariance[i][i]),
                        delta=1e-9,
                    )
                    self.assertAlmostEqual(
                        self.result.standard_mean_history[n][i],
                        expected_mean[i],
                        delta=1e-8,
                    )

    def test_convergence_is_monotone_with_rate_one_half(self) -> None:
        """The change decreases monotonically and halves asymptotically."""
        changes = self.result.change_history[:30]
        for earlier, later in zip(changes, changes[1:], strict=False):
            self.assertLess(later, earlier)
        for earlier, later in zip(changes[8:25], changes[9:26], strict=True):
            self.assertAlmostEqual(later / earlier, 0.5, delta=0.01)
        mean, _, _ = standard_posterior()
        errors = [
            max(abs(m[i] - mean[i]) for i in range(2))
            for m in self.result.standard_mean_history[1:30]
        ]
        for earlier, later in zip(errors, errors[1:], strict=False):
            self.assertLess(later, earlier)

    def test_cost_and_histories(self) -> None:
        """Each iteration costs 2 d + 1 = 5 forward evaluations."""
        result = self.result
        self.assertEqual(result.forward_evaluations, 5 * result.iterations)
        self.assertEqual(result.model_calls, 5 * result.iterations * len(LINE_X))
        self.assertEqual(len(result.misfit_history), result.iterations)
        self.assertEqual(len(result.change_history), result.iterations)
        for history in (
            result.standard_mean_history,
            result.standard_spread_history,
            result.mean_history,
            result.spread_history,
        ):
            self.assertEqual(len(history), result.iterations + 1)
        self.assertEqual(result.standard_mean_history[0], (0.0, 0.0))
        self.assertEqual(result.standard_spread_history[0], (1.0, 1.0))
        for k in range(2):
            self.assertAlmostEqual(
                result.mean_history[0][k], PRIOR_MEAN[k], delta=1e-12
            )
            self.assertAlmostEqual(result.spread_history[0][k], PRIOR_SD[k], delta=1e-9)
        self.assertLess(result.misfit_history[-1], result.misfit_history[0])
        self.assertIn("converged", result.summary())

    def test_sigma_point_parameter_does_not_change_the_linear_result(self) -> None:
        """Any kappa >= 0 reproduces mean and covariance for an affine map."""
        mean, covariance = physical_posterior()
        for kappa in (0.0, 0.5, 3.0):
            result = unscented_kalman_inversion(self.problem, kappa=kappa)
            with self.subTest(kappa=kappa):
                self.assertTrue(result.converged)
                self.assertEqual(result.kappa, kappa)
                for i in range(2):
                    self.assertAlmostEqual(
                        result.posterior_mean[i], mean[i], delta=1e-6
                    )
                    self.assertAlmostEqual(
                        result.posterior_covariance[i][i],
                        covariance[i][i],
                        delta=1e-6 * covariance[i][i],
                    )

    def test_sigma_points_reproduce_mean_and_covariance(self) -> None:
        """The weighted moments of the 2 d + 1 points are exact."""
        mean = (0.3, -1.2, 2.0)
        covariance = ((2.0, 0.3, -0.1), (0.3, 1.0, 0.4), (-0.1, 0.4, 0.5))
        for kappa in (0.0, 1.0):
            points, weights = _sigma_points(mean, covariance, kappa)
            self.assertEqual(len(points), 7)
            self.assertAlmostEqual(sum(weights), 1.0, delta=1e-14)
            self.assertTrue(all(w >= 0.0 for w in weights))
            for i in range(3):
                self.assertAlmostEqual(
                    sum(w * p[i] for w, p in zip(weights, points, strict=True)),
                    mean[i],
                    delta=1e-13,
                )
                for j in range(3):
                    value = sum(
                        w * (p[i] - mean[i]) * (p[j] - mean[j])
                        for w, p in zip(weights, points, strict=True)
                    )
                    self.assertAlmostEqual(value, covariance[i][j], delta=1e-13)

    def test_unconverged_run_is_flagged(self) -> None:
        """Too few iterations return ``converged=False`` and say so."""
        result = unscented_kalman_inversion(self.problem, iterations=3)
        self.assertFalse(result.converged)
        self.assertEqual(result.iterations, 3)
        self.assertEqual(result.forward_evaluations, 15)
        self.assertIn("NOT converged", result.summary())


class ConsensusSamplingLinearTests(unittest.TestCase):
    """The Gaussian posterior is a fixed point of the sampling mode."""

    @classmethod
    def setUpClass(cls) -> None:
        """Build the problem and its exact posterior."""
        cls.problem = line_problem()
        cls.mean, cls.covariance = physical_posterior()

    def test_reweighted_gaussian_has_covariance_over_one_plus_beta(self) -> None:
        """Weights exp(-beta x^2/2) on N(0, 1) draws give variance 1/(1+beta)."""
        generator = random.Random(8)
        draws = [generator.gauss(0.0, 1.0) for _ in range(40000)]
        potentials = [0.5 * x * x for x in draws]
        for beta in (0.5, 1.0, 3.0):
            weights, effective = _weights(potentials, beta)
            center = sum(w * x for w, x in zip(weights, draws, strict=True))
            variance = sum(
                w * (x - center) ** 2 for w, x in zip(weights, draws, strict=True)
            )
            with self.subTest(beta=beta):
                self.assertAlmostEqual(sum(weights), 1.0, delta=1e-12)
                self.assertAlmostEqual(variance, 1.0 / (1.0 + beta), delta=0.02)
                # J_eff / J = sqrt(1 + 2 beta) / (1 + beta) for d = 1.
                self.assertAlmostEqual(
                    effective / len(draws),
                    math.sqrt(1.0 + 2.0 * beta) / (1.0 + beta),
                    delta=0.02,
                )

    def test_tempering_keeps_the_effective_sample_size(self) -> None:
        """The chosen beta is the largest with J_eff at the target."""
        generator = random.Random(2)
        potentials = [50.0 * generator.random() for _ in range(200)]
        beta, weights, effective = _tempered_weights(potentials, 1.0, 100.0)
        self.assertLess(beta, 1.0)
        self.assertGreater(beta, 0.0)
        self.assertAlmostEqual(effective, 100.0, delta=1e-6)
        self.assertAlmostEqual(sum(weights), 1.0, delta=1e-12)
        self.assertLess(_weights(potentials, 1.01 * beta)[1], 100.0)
        flat = [1.0 + 1e-3 * generator.random() for _ in range(200)]
        self.assertEqual(_tempered_weights(flat, 1.0, 100.0)[0], 1.0)

    def test_sampling_mode_reaches_the_gaussian_posterior(self) -> None:
        """Replicate-averaged mean and covariance lie within 4 SE of exact."""
        replicates, size = 6, 400
        results = [
            consensus_based_sampling(
                self.problem,
                ensemble_size=size,
                iterations=25,
                seed=RandomStream(77).child("replicate", r),
            )
            for r in range(replicates)
        ]
        for i in range(2):
            center, error = mean_and_standard_error(
                [result.mean_history[-1][i] for result in results]
            )
            with self.subTest(moment="mean", index=i):
                self.assertLess(abs(center - self.mean[i]), 4.0 * error)
                self.assertLess(error, 0.1 * math.sqrt(self.covariance[i][i]))
        for i, j in ENTRIES:
            center, error = mean_and_standard_error(
                [result.covariance()[i][j] for result in results]
            )
            with self.subTest(moment="covariance", entry=(i, j)):
                # 4 replicate SEs plus 5 % for the finite-ensemble deficit
                # (1 + beta)/(beta J_eff), about 0.7 % here.
                tolerance = 4.0 * error + 0.05 * abs(self.covariance[i][j])
                self.assertLess(abs(center - self.covariance[i][j]), tolerance)
        result = results[0]
        self.assertIsInstance(result, ConsensusSamplingResult)
        self.assertEqual(result.mode, "sampling")
        self.assertEqual(result.forward_evaluations, size * 26)
        self.assertEqual(result.model_calls, size * 26 * len(LINE_X))
        self.assertEqual(len(result.inverse_temperature_history), 25)
        self.assertEqual(len(result.effective_size_history), 25)
        self.assertEqual(len(result.misfit_history), 26)
        # The posterior is far narrower than the prior: beta starts small,
        # never exceeds its bound, and the effective size is kept.
        self.assertLess(result.inverse_temperature_history[0], 0.1)
        self.assertEqual(result.inverse_temperature_history[-1], 1.0)
        self.assertTrue(all(b <= 1.0 for b in result.inverse_temperature_history))
        self.assertTrue(
            all(n >= 0.5 * size - 1e-6 for n in result.effective_size_history)
        )
        # Mean-field J_eff / J at the fixed point: (1 + 2 beta)/(1 + beta)^2.
        self.assertAlmostEqual(
            result.effective_size_history[-1] / size, 0.75, delta=0.08
        )
        standard = result.standard_covariance()
        self.assertAlmostEqual(
            result.covariance()[0][1],
            PRIOR_SD[0] * PRIOR_SD[1] * standard[0][1],
            delta=1e-12,
        )
        self.assertEqual(len(result.draws()), size)
        self.assertEqual(tuple(result.mean()), ("a", "b"))
        self.assertIn("Consensus-based sampling: 400 members", result.summary())

    def test_optimization_mode_concentrates_at_the_map_point(self) -> None:
        """The ensemble contracts around the minimizer of the potential."""
        scales = [math.sqrt(self.covariance[i][i]) for i in range(2)]
        for seed in (1, 2):
            result = consensus_based_sampling(
                self.problem,
                ensemble_size=300,
                iterations=40,
                mode="optimization",
                seed=seed,
            )
            sampling = consensus_based_sampling(
                self.problem, ensemble_size=300, iterations=40, seed=seed
            )
            for i in range(2):
                with self.subTest(seed=seed, index=i):
                    # MAP = posterior mean for the Gaussian posterior.
                    self.assertLess(
                        abs(result.mean_history[-1][i] - self.mean[i]),
                        0.25 * scales[i],
                    )
                    self.assertLess(
                        abs(result.best_point[i] - self.mean[i]), 0.1 * scales[i]
                    )
                    self.assertLess(result.spread_history[-1][i], 0.3 * scales[i])
                    self.assertGreater(sampling.spread_history[-1][i], 0.7 * scales[i])
            self.assertEqual(result.best_potential, min(result.potential_history))
            self.assertIn("not a posterior uncertainty", result.summary())
            self.assertEqual(result.ensemble_size, 300)

    def test_reproducibility(self) -> None:
        """Equal seeds give equal ensembles; different seeds do not."""
        runs = [
            consensus_based_sampling(
                self.problem, ensemble_size=40, iterations=5, relaxation=0.0, seed=s
            )
            for s in (6, 6, 7)
        ]
        self.assertEqual(runs[0].standard_ensemble, runs[1].standard_ensemble)
        self.assertNotEqual(runs[0].standard_ensemble, runs[2].standard_ensemble)
        self.assertEqual(runs[0].relaxation, 0.0)
        self.assertEqual(runs[0].seed, 6)


class NonlinearDecayTests(unittest.TestCase):
    """The three methods approximate the MCMC posterior of the decay model."""

    @classmethod
    def setUpClass(cls) -> None:
        """Sample the reference posterior once."""
        cls.problem = decay_problem()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            posterior = cls.problem.sample_posterior(
                draws=1500, warmup=500, chains=4, seed=3
            )
        cls.reference = posterior.mean()
        cls.scales = {s.name: s.standard_deviation for s in posterior.summary()}

    def check(self, mean, spreads, mean_tolerance, low, high) -> None:
        """Compare means (in posterior SDs) and spreads (as ratios)."""
        for k, name in enumerate(self.problem.parameter_names):
            with self.subTest(parameter=name):
                self.assertLess(
                    abs(mean[k] - self.reference[name]),
                    mean_tolerance * self.scales[name],
                )
                self.assertGreater(spreads[k], low * self.scales[name])
                self.assertLess(spreads[k], high * self.scales[name])

    def test_unscented_inversion_is_close_to_mcmc(self) -> None:
        """Mean within 0.25 SD and SDs within 20 % in 2 d + 1 runs a step.

        The posterior of this problem is close to Gaussian in ``u``; the
        tolerance covers the Gaussian approximation and the MCMC error
        (about 0.03 SD for the mean and 5 % for the SD).
        """
        result = unscented_kalman_inversion(self.problem)
        self.assertTrue(result.converged)
        self.assertLess(result.forward_evaluations, 300)
        self.check(result.posterior_mean, result.spread_history[-1], 0.25, 0.8, 1.2)

    def test_interacting_langevin_is_close_to_mcmc(self) -> None:
        """Eight members: mean within 0.5 SD and SDs within a factor 1.5.

        The gradient-free drift is not exact for this model and a run of
        integrated time of about 15 leaves a time-averaging error of the
        order of 0.1 SD for the mean and 20 % for the SD.
        """
        for seed in (0, 1):
            result = interacting_langevin_sampler(
                self.problem, ensemble_size=8, steps=700, seed=seed
            )
            deviations = tuple(
                math.sqrt(result.posterior_covariance[k][k]) for k in range(2)
            )
            with self.subTest(seed=seed):
                self.check(result.posterior_mean, deviations, 0.5, 1 / 1.5, 1.5)

    def test_consensus_sampling_is_close_to_mcmc(self) -> None:
        """Mean within 0.5 SD and spread within a factor 1.5 (J = 200)."""
        result = consensus_based_sampling(
            self.problem, ensemble_size=200, iterations=30, seed=1
        )
        self.check(
            result.mean_history[-1], result.spread_history[-1], 0.5, 1 / 1.5, 1.5
        )
        optimizer = consensus_based_sampling(
            self.problem, ensemble_size=200, iterations=30, mode="optimization", seed=1
        )
        for k, name in enumerate(self.problem.parameter_names):
            self.assertLess(optimizer.spread_history[-1][k], 0.4 * self.scales[name])
            self.assertLess(
                abs(optimizer.best_point[k] - self.reference[name]),
                0.25 * self.scales[name],
            )


class ValidationTests(unittest.TestCase):
    """Invalid inputs are rejected with explicit messages."""

    @classmethod
    def setUpClass(cls) -> None:
        """Build the line problem."""
        cls.problem = line_problem()

    def test_noise_models_without_a_fixed_gamma_are_rejected(self) -> None:
        """Unknown-scale and Student-t noise raise for all three methods."""
        unknown = line_problem(
            NoiseModel("y", standard_deviation_prior=LogNormal(math.log(0.3), 0.5))
        )
        heavy = line_problem(
            NoiseModel("y", standard_deviation=0.3, degrees_of_freedom=4.0)
        )
        for problem, text in ((unknown, "unknown scale"), (heavy, "Student-t")):
            with self.assertRaisesRegex(ValueError, text):
                interacting_langevin_sampler(problem, ensemble_size=6, steps=2)
            with self.assertRaisesRegex(ValueError, text):
                unscented_kalman_inversion(problem)
            with self.assertRaisesRegex(ValueError, text):
                consensus_based_sampling(problem, ensemble_size=6, iterations=2)
        for function in (
            lambda: interacting_langevin_sampler("p", ensemble_size=6, steps=2),
            lambda: unscented_kalman_inversion("p"),
            lambda: consensus_based_sampling("p", ensemble_size=6, iterations=2),
        ):
            with self.assertRaises(TypeError):
                function()

    def test_interacting_langevin_arguments(self) -> None:
        """Ensemble size, steps, burn-in, step and flag are validated."""
        cases = (
            ({"ensemble_size": 3, "steps": 5}, "d \\+ 2"),
            ({"ensemble_size": 6, "steps": 0}, "steps"),
            ({"ensemble_size": 6, "steps": 5, "burn_in": 6}, "burn_in"),
            ({"ensemble_size": 6, "steps": 5, "step_size": 0.0}, "step_size"),
            ({"ensemble_size": 6, "steps": 5, "step_factor": 2.0}, "step_factor"),
            ({"ensemble_size": 6, "steps": 5, "step_factor": 0.0}, "step_factor"),
        )
        for arguments, text in cases:
            with (
                self.subTest(arguments=arguments),
                self.assertRaisesRegex(ValueError, text),
            ):
                interacting_langevin_sampler(self.problem, **arguments)
        with self.assertRaises(TypeError):
            interacting_langevin_sampler(
                self.problem, ensemble_size=6, steps=5, correction=1
            )
        # A fixed step far beyond the stability limit is reported, not returned.
        with self.assertRaisesRegex(ValueError, "diverged"):
            interacting_langevin_sampler(
                self.problem, ensemble_size=6, steps=400, step_size=1.0, seed=0
            )
        full = interacting_langevin_sampler(
            self.problem, ensemble_size=4, steps=3, burn_in=3, seed=0
        )
        self.assertEqual(full.burn_in, 3)
        for k in range(2):
            self.assertAlmostEqual(
                full.posterior_mean[k], full.mean_history[-1][k], delta=1e-12
            )

    def test_unscented_inversion_arguments(self) -> None:
        """Iterations, tolerance and kappa are validated."""
        for arguments, text in (
            ({"iterations": 0}, "iterations"),
            ({"tolerance": -1.0}, "tolerance"),
            ({"kappa": -0.5}, "kappa"),
            ({"kappa": math.inf}, "kappa"),
        ):
            with (
                self.subTest(arguments=arguments),
                self.assertRaisesRegex(ValueError, text),
            ):
                unscented_kalman_inversion(self.problem, **arguments)

    def test_consensus_sampling_arguments(self) -> None:
        """Mode, sizes, temperature, relaxation and fraction are validated."""
        base = {"ensemble_size": 20, "iterations": 3}
        for arguments, text in (
            ({"ensemble_size": 3}, "d \\+ 2"),
            ({"iterations": 0}, "iterations"),
            ({"mode": "annealing"}, "mode"),
            ({"inverse_temperature": 0.0}, "inverse_temperature"),
            ({"relaxation": 1.0}, "relaxation"),
            ({"relaxation": -0.1}, "relaxation"),
            ({"minimum_effective_fraction": 1.0}, "minimum_effective_fraction"),
            ({"minimum_effective_fraction": 0.0}, "minimum_effective_fraction"),
        ):
            with (
                self.subTest(arguments=arguments),
                self.assertRaisesRegex(ValueError, text),
            ):
                consensus_based_sampling(self.problem, **{**base, **arguments})


if __name__ == "__main__":
    unittest.main()


class UnscentedSymmetricMapTests(unittest.TestCase):
    """Audit: ``converged`` does not certify closeness to the posterior."""

    def test_even_forward_map_returns_the_prior_as_converged(self) -> None:
        """For G = theta^2 and a N(0, 1) prior the prior is a fixed point.

        The posterior for y = 4 (noise 0.2) has modes near -2 and +2 and a
        standard deviation of about 2.0; the symmetric sigma points carry
        no slope, so the iteration stops at the prior after one step. The
        documented behaviour is pinned here so that a change is noticed.
        """
        prior = JointInput.independent(RandomInput("t", Normal(0.0, 1.0)))
        model = ProbabilisticModel(
            prior,
            lambda parameters: {"y": parameters["t"] ** 2},
            (QuantityOfInterest("y", "square"),),
        )
        problem = CalibrationProblem(
            model,
            (Experiment({"y": 4.0}, {}),),
            (NoiseModel("y", standard_deviation=0.2),),
        )
        result = unscented_kalman_inversion(problem)
        self.assertTrue(result.converged)
        self.assertEqual(result.iterations, 1)
        self.assertAlmostEqual(result.posterior_mean[0], 0.0, places=12)
        self.assertAlmostEqual(result.posterior_covariance[0][0], 1.0, places=12)
