"""Tests for the Kennedy-O'Hagan emulator calibration.

Everything is synthetic. The "simulator" is ``eta(x, t) = t g(x)`` with
``g(x) = 1 - exp(-3 x)`` (a saturating response, linear in the calibration
parameter), the synthetic "reality" adds a missing term,
``zeta(x) = theta* g(x) - 0.3 x^2``, in the manner of the simple physical
law with a missing term of Brynjarsdottir and O'Hagan (2014). Order of the
tests: the joint-covariance algebra against a dense oracle, the exact
linear-Gaussian posterior when the simulator is linear in ``t``, the
bias demonstration with and without discrepancy, the predictive variance
decomposition, then failure cases. Expensive coverage studies run only when
``PROBSTAT_SLOW=1``.
"""

from __future__ import annotations

import math
import os
import random
import unittest
import warnings

from probstat.linalg import cholesky, inverse_spd, matvec
from probstat.surrogates import GaussianProcessRegularization
from probstat.surrogates.gaussian_process import kernel_value
from probstat.validation.koh_emulator import (
    SAMPLERS,
    IdentifiabilityReport,
    KOHEmulatorCalibration,
    KOHPrediction,
    KOHResult,
    LogNormalPrior,
)

THETA_TRUE = 1.0
NOISE_SD = 0.05
AMPLITUDE = LogNormalPrior(math.log(0.2), 1.0)
NEW_CONTROLS = tuple((x,) for x in (0.15, 0.35, 0.55, 0.75, 0.95))


def response(x: float) -> float:
    """Return the saturating response ``g(x)`` of the synthetic simulator."""
    return 1.0 - math.exp(-3.0 * x)


def reality(x: float) -> float:
    """Return the synthetic reality, the simulator plus a missing term."""
    return THETA_TRUE * response(x) - 0.3 * x * x


def radical_inverse(index: int, base: int) -> float:
    """Return the van der Corput radical inverse of ``index`` in ``base``."""
    value = 0.0
    scale = 1.0 / base
    while index > 0:
        value += (index % base) * scale
        index //= base
        scale /= base
    return value


def simulator_design(
    runs: int = 20,
) -> tuple[
    tuple[tuple[float, ...], ...],
    tuple[tuple[float, ...], ...],
    tuple[float, ...],
]:
    """Return a Halton design plus the four corners of ``[0, 1] x [0, 2]``.

    ``runs`` Halton points (bases 2 and 3) give many distinct values of both
    inputs, which a stationary-kernel emulator needs to reproduce a response
    that is linear in ``t`` and saturating in ``x``.
    """
    controls = [(radical_inverse(k, 2),) for k in range(1, runs + 1)]
    parameters = [(2.0 * radical_inverse(k, 3),) for k in range(1, runs + 1)]
    for x in (0.0, 1.0):
        for t in (0.0, 2.0):
            controls.append((x,))
            parameters.append((t,))
    outputs = tuple(
        response(c[0]) * t[0] for c, t in zip(controls, parameters, strict=True)
    )
    return tuple(controls), tuple(parameters), outputs


def field_data(
    seed: int, count: int = 8, theta: float = THETA_TRUE
) -> tuple[tuple[tuple[float, ...], ...], tuple[float, ...]]:
    """Return noisy field data of the synthetic reality (law with missing term)."""
    generator = random.Random(seed)
    grid = [0.1 + 0.9 * i / (count - 1) for i in range(count)]
    values = tuple(
        theta * response(x) - 0.3 * x * x + generator.gauss(0.0, NOISE_SD) for x in grid
    )
    return tuple((x,) for x in grid), values


def build(
    seed: int = 101,
    *,
    runs: int = 20,
    count: int = 8,
    theta_data: float = THETA_TRUE,
    discrepancy: bool = True,
) -> KOHEmulatorCalibration:
    """Build the calibration problem of the bias demonstration."""
    controls, parameters, outputs = simulator_design(runs)
    field_x, field_y = field_data(seed, count, theta_data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return KOHEmulatorCalibration(
            controls,
            parameters,
            outputs,
            field_x,
            field_y,
            theta_bounds=[(0.0, 2.0)],
            noise_sd=NOISE_SD,
            amplitude_prior=AMPLITUDE,
            discrepancy=discrepancy,
            emulator_regularization=GaussianProcessRegularization(),
            seed=1,
        )


class DenseJointOracleTest(unittest.TestCase):
    """The block covariance and its conditional form against a dense oracle."""

    @classmethod
    def setUpClass(cls) -> None:
        """Build one small problem with a discrepancy."""
        cls.cal = build(7, runs=12, count=4)
        cls.phi = (0.8, math.log(0.1), math.log(0.4))

    def dense_covariance(self) -> tuple[tuple[float, ...], ...]:
        """Assemble Sigma from the stated blocks, independently of the class."""
        cal = self.cal
        hyper = cal.emulator.hyperparameters
        theta = (self.phi[0],)
        sigma_delta = math.exp(self.phi[1])
        ell_delta = math.exp(self.phi[2]) * cal._ranges[0]
        runs = cal.emulator.points
        field = tuple(x + theta for x in cal._field_x)
        everything = runs + field
        size = len(everything)
        rows = []
        for a in range(size):
            row = []
            for b in range(size):
                value = kernel_value(
                    cal.emulator.kernel,
                    everything[a],
                    everything[b],
                    hyper.length_scales,
                    hyper.signal_variance,
                )
                if a == b and a < len(runs):
                    value += hyper.noise_variance
                if a >= len(runs) and b >= len(runs):
                    xa = cal._field_x[a - len(runs)]
                    xb = cal._field_x[b - len(runs)]
                    value += kernel_value(
                        "matern52", xa, xb, (ell_delta,), sigma_delta**2
                    )
                    if a == b:
                        value += NOISE_SD**2
                row.append(value)
            rows.append(tuple(row))
        return tuple(rows)

    def test_joint_covariance_matches_the_stated_blocks(self) -> None:
        """``joint_covariance`` equals the block formula entry by entry."""
        mine = self.cal.joint_covariance(self.phi)
        oracle = self.dense_covariance()
        for row_a, row_b in zip(mine, oracle, strict=True):
            for a, b in zip(row_a, row_b, strict=True):
                self.assertAlmostEqual(a, b, delta=1e-12 * (1.0 + abs(b)))

    def test_conditional_form_equals_full_joint_density(self) -> None:
        """``log p(z) + log p(y | z)`` equals the dense joint Gaussian density."""
        cal = self.cal
        sigma = self.dense_covariance()
        lower = cholesky(sigma)
        stacked = tuple(cal._outputs) + tuple(cal._field_y)
        centred = tuple(v - cal._m for v in stacked)
        # Independent route: explicit inverse and log determinant from the factor.
        inverse = inverse_spd(sigma)
        quad = math.fsum(
            a * b for a, b in zip(centred, matvec(inverse, centred), strict=True)
        )
        log_det = 2.0 * math.fsum(math.log(lower[k][k]) for k in range(len(sigma)))
        expected = -0.5 * (len(sigma) * math.log(2.0 * math.pi) + log_det + quad)
        self.assertAlmostEqual(cal.log_joint_likelihood(self.phi), expected, delta=1e-8)

    def test_emulator_hyperparameters_match_library_prediction(self) -> None:
        """The conditional mean/variance of eta equals the library GP prediction."""
        cal = self.cal
        runs = len(cal.emulator.points)
        sigma = cal.joint_covariance(self.phi)
        a_block = tuple(tuple(sigma[i][:runs]) for i in range(runs))
        inverse_a = inverse_spd(a_block)
        centred = tuple(v - cal._m for v in cal._outputs)
        weights = matvec(inverse_a, centred)
        for i in (0, 2):
            cross = tuple(sigma[j][runs + i] for j in range(runs))
            mean = cal._m + math.fsum(
                a * b for a, b in zip(cross, weights, strict=True)
            )
            reduction = math.fsum(
                a * b for a, b in zip(cross, matvec(inverse_a, cross), strict=True)
            )
            eta_point = cal._field_x[i] + (self.phi[0],)
            library_mean, library_var = cal.emulator.predict_with_variance(eta_point)
            self.assertAlmostEqual(mean, library_mean, delta=1e-7)
            self.assertAlmostEqual(
                kernel_value(
                    cal.emulator.kernel,
                    eta_point,
                    eta_point,
                    cal.emulator.hyperparameters.length_scales,
                    cal.emulator.hyperparameters.signal_variance,
                )
                - reduction,
                library_var,
                delta=1e-7,
            )

    def test_cache_does_not_change_values(self) -> None:
        """Re-evaluating at the same theta, then another, gives identical numbers."""
        first = self.cal.log_posterior(self.phi)
        other = (1.3, self.phi[1], self.phi[2])
        self.cal.log_posterior(other)
        self.assertEqual(self.cal.log_posterior(self.phi), first)


class LinearGaussianOracleTest(unittest.TestCase):
    """Without discrepancy and with a simulator linear in t, theta is Gaussian."""

    def test_grid_posterior_matches_exact_linear_gaussian(self) -> None:
        """Emulated posterior of theta equals the exact truncated-normal one.

        With ``eta = t g(x)`` and no discrepancy, ``y_i = theta g(x_i) + eps_i``
        gives a Gaussian likelihood in ``theta`` with precision ``sum g^2/s^2``;
        the uniform prior on [0, 2] truncates it. The emulator interpolates the
        runs to about its predictive sd, which is small against the noise, so
        the first two moments must agree to a fraction of the posterior sd.
        """
        cal = build(11, runs=28, count=10, theta_data=0.9, discrepancy=False)
        field_x = [x[0] for x in cal._field_x]
        g = [response(x) for x in field_x]
        y = cal._field_y
        precision = math.fsum(v * v for v in g) / NOISE_SD**2
        centre = math.fsum(a * b for a, b in zip(g, y, strict=True)) / math.fsum(
            v * v for v in g
        )
        points = [2.0 * k / 800 for k in range(801)]
        log_exact = [-0.5 * precision * (t - centre) ** 2 for t in points]
        log_emulated = [cal.log_posterior((t,)) for t in points]
        for log_values in (log_exact, log_emulated):
            top = max(log_values)
            log_values[:] = [v - top for v in log_values]
        moments = []
        for log_values in (log_exact, log_emulated):
            weights = [math.exp(v) for v in log_values]
            total = math.fsum(weights)
            mean = (
                math.fsum(w * t for w, t in zip(weights, points, strict=True)) / total
            )
            var = (
                math.fsum(
                    w * (t - mean) ** 2 for w, t in zip(weights, points, strict=True)
                )
                / total
            )
            moments.append((mean, math.sqrt(var)))
        (exact_mean, exact_sd), (mean, sd) = moments
        self.assertAlmostEqual(exact_sd, 1.0 / math.sqrt(precision), delta=1e-3)
        self.assertAlmostEqual(mean, exact_mean, delta=0.1 * exact_sd)
        self.assertAlmostEqual(sd / exact_sd, 1.0, delta=0.1)

    def test_slice_sampler_matches_grid_posterior(self) -> None:
        """Slice draws reproduce the grid posterior mean within MCSE bands."""
        cal = build(11, runs=12, count=6, theta_data=0.9, discrepancy=False)
        points = [2.0 * k / 400 for k in range(401)]
        logs = [cal.log_posterior((t,)) for t in points]
        top = max(logs)
        weights = [math.exp(v - top) for v in logs]
        total = math.fsum(weights)
        mean = math.fsum(w * t for w, t in zip(weights, points, strict=True)) / total
        sd = math.sqrt(
            math.fsum(w * (t - mean) ** 2 for w, t in zip(weights, points, strict=True))
            / total
        )
        result = cal.sample(draws=200, warmup=50, seed=3)
        self.assertTrue(result.converged, result.stop_reason)
        self.assertEqual(result.failed_evaluations, 0)
        # Bands: 4 Monte Carlo standard errors with the measured ESS.
        band = 4.0 * sd / math.sqrt(result.ess_min)
        self.assertAlmostEqual(result.mean()[0], mean, delta=band)
        self.assertAlmostEqual(result.sd()[0], sd, delta=0.25 * sd)

    def test_tmcmc_evidence_matches_grid_quadrature(self) -> None:
        """TMCMC log evidence agrees with quadrature of the likelihood."""
        cal = build(11, runs=12, count=6, theta_data=0.9, discrepancy=False)
        points = [2.0 * k / 800 for k in range(801)]
        values = [math.exp(cal.log_conditional_likelihood((t,))) for t in points]
        step = points[1] - points[0]
        integral = step * (math.fsum(values) - 0.5 * (values[0] + values[-1]))
        log_evidence = math.log(integral / 2.0)
        result = cal.sample(sampler="tmcmc", particles=200, seed=4)
        self.assertTrue(result.converged, result.stop_reason)
        self.assertAlmostEqual(result.log_evidence, log_evidence, delta=0.5)
        self.assertIsNone(result.mcmc)


class BiasDemonstrationTest(unittest.TestCase):
    """A missing term biases the no-discrepancy calibration (fast single seed)."""

    SEED = 101
    RUNS = 12

    @classmethod
    def setUpClass(cls) -> None:
        """Sample both calibrations once for the class."""
        cls.with_delta = build(cls.SEED, runs=cls.RUNS)
        cls.without = cls.with_delta.without_discrepancy()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.result_with = cls.with_delta.sample(draws=100, warmup=60, seed=5)
            cls.result_without = cls.without.sample(draws=100, warmup=60, seed=5)

    def test_both_runs_converged(self) -> None:
        """Diagnostics pass and no evaluation failed."""
        for result in (self.result_with, self.result_without):
            self.assertTrue(result.converged, result.stop_reason)
            self.assertEqual(result.failed_evaluations, 0)
            self.assertEqual(result.emulator_mode, "modular")
        self.assertEqual(self.result_with.sampler, "slice")

    def test_no_discrepancy_is_overconfident_and_biased(self) -> None:
        """Without discrepancy the interval is narrow and excludes the truth."""
        low, high = self.result_without.theta_interval(0)
        self.assertFalse(low <= THETA_TRUE <= high)
        self.assertLess(self.result_without.sd()[0], 0.05)
        # Least squares of eta = theta g on the reality gives about 0.84:
        self.assertLess(self.result_without.mean()[0], 0.9)

    def test_discrepancy_widens_and_covers(self) -> None:
        """With discrepancy the posterior is wider and contains the truth."""
        low, high = self.result_with.theta_interval(0)
        self.assertTrue(low <= THETA_TRUE <= high)
        self.assertGreater(self.result_with.sd()[0], 3.0 * self.result_without.sd()[0])

    def test_identifiability_report(self) -> None:
        """Contraction, correlation and the comparison flag are consistent."""
        report = self.with_delta.identifiability(
            self.result_with, reference=self.result_without, truth=(THETA_TRUE,)
        )
        self.assertIsInstance(report, IdentifiabilityReport)
        prior_sd = 2.0 / math.sqrt(12.0)
        self.assertAlmostEqual(report.theta_prior_sd[0], prior_sd, delta=1e-12)
        expected = 1.0 - (self.result_with.sd()[0] / prior_sd) ** 2
        self.assertAlmostEqual(report.contraction[0], expected, delta=1e-12)
        self.assertTrue(-1.0 <= report.amplitude_correlation[0] <= 1.0)
        self.assertEqual(len(report.length_scale_correlation[0]), 1)
        self.assertGreater(report.sd_ratio[0], 3.0)
        # The no-discrepancy truth z-score is large, the with-discrepancy one small.
        self.assertGreater(abs(report.truth_z_without[0]), 4.0)
        self.assertLess(abs(report.truth_z_with[0]), 2.5)
        self.assertEqual(len(report.reference_overconfident), 1)
        plain = self.without.identifiability(self.result_without)
        self.assertTrue(math.isnan(plain.amplitude_correlation[0]))
        self.assertEqual(plain.sd_ratio, ())

    def test_predictive_covers_reality_with_discrepancy(self) -> None:
        """The 95 percent predictive interval of zeta contains the true reality."""
        prediction = self.with_delta.predict(
            self.result_with, NEW_CONTROLS, include_noise=False, max_draws=60
        )
        for x, low, high in zip(
            NEW_CONTROLS, prediction.lower, prediction.upper, strict=True
        ):
            self.assertTrue(low <= reality(x[0]) <= high, (x, low, high))

    def test_no_discrepancy_predictive_is_narrower(self) -> None:
        """Dropping the discrepancy shrinks the predictive sd of reality."""
        wide = self.with_delta.predict(
            self.result_with, NEW_CONTROLS, include_noise=False, max_draws=40
        )
        narrow = self.without.predict(
            self.result_without, NEW_CONTROLS, include_noise=False, max_draws=40
        )
        for a, b in zip(wide.total_variance, narrow.total_variance, strict=True):
            self.assertGreater(a, b)
        self.assertEqual(narrow.discrepancy, (0.0,) * len(NEW_CONTROLS))


class PredictiveDecompositionTest(unittest.TestCase):
    """The five variance parts sum to the mixture variance."""

    @classmethod
    def setUpClass(cls) -> None:
        """Sample a small posterior once."""
        cls.cal = build(101, runs=12, count=5)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.result = cls.cal.sample(draws=30, warmup=40, seed=9)

    def dense_oracle(self, x_new: tuple[float, ...], draws: int) -> tuple[float, float]:
        """Mixture mean and variance from explicit inverse-matrix conditionals."""
        cal = self.cal
        total = len(self.result.samples)
        indices = sorted({(k * total) // draws for k in range(draws)})
        stacked = tuple(cal._outputs) + tuple(cal._field_y)
        centred = tuple(v - cal._m for v in stacked)
        means = []
        variances = []
        hyper = cal.emulator.hyperparameters
        for index in indices:
            phi = self.result.samples[index]
            theta = (phi[0],)
            sigma_delta = math.exp(phi[1])
            ell = math.exp(phi[2]) * cal._ranges[0]
            inverse = inverse_spd(cal.joint_covariance(phi))
            point = x_new + theta
            cross = [
                kernel_value(
                    cal.emulator.kernel,
                    point,
                    p,
                    hyper.length_scales,
                    hyper.signal_variance,
                )
                for p in cal.emulator.points
            ] + [
                kernel_value(
                    cal.emulator.kernel,
                    point,
                    xf + theta,
                    hyper.length_scales,
                    hyper.signal_variance,
                )
                + kernel_value("matern52", x_new, xf, (ell,), sigma_delta**2)
                for xf in cal._field_x
            ]
            weights = matvec(inverse, cross)
            means.append(
                cal._m + math.fsum(a * b for a, b in zip(weights, centred, strict=True))
            )
            prior = hyper.signal_variance + sigma_delta**2
            variances.append(
                prior
                - math.fsum(a * b for a, b in zip(weights, cross, strict=True))
                + NOISE_SD**2
            )
        centre = math.fsum(means) / len(means)
        # Law of total variance in the second-moment form E[m^2 + v] - E[m]^2.
        second = math.fsum(m * m + v for m, v in zip(means, variances, strict=True))
        return centre, second / len(means) - centre**2

    def test_parts_sum_to_total_and_match_dense_oracle(self) -> None:
        """Exact additivity, and agreement with an independent dense computation."""
        prediction = self.cal.predict(self.result, NEW_CONTROLS[:3], max_draws=12)
        self.assertIsInstance(prediction, KOHPrediction)
        self.assertEqual(prediction.draws, 12)
        for n, x in enumerate(NEW_CONTROLS[:3]):
            parts = prediction.component_sum()[n]
            self.assertAlmostEqual(
                parts, prediction.total_variance[n], delta=1e-12 * parts
            )
            mean, variance = self.dense_oracle(x, 12)
            self.assertAlmostEqual(prediction.mean[n], mean, delta=1e-7)
            self.assertAlmostEqual(
                prediction.total_variance[n], variance, delta=1e-6 * variance
            )
            self.assertAlmostEqual(prediction.noise[n], NOISE_SD**2, delta=1e-15)
            self.assertGreaterEqual(prediction.emulator[n], 0.0)
            self.assertGreaterEqual(prediction.discrepancy[n], 0.0)
            self.assertGreaterEqual(prediction.parameter[n], 0.0)
            self.assertLess(prediction.lower[n], prediction.mean[n])
            self.assertGreater(prediction.upper[n], prediction.mean[n])

    def test_noise_free_prediction_has_no_noise_part(self) -> None:
        """``include_noise=False`` removes exactly the noise variance."""
        with_noise = self.cal.predict(self.result, NEW_CONTROLS[:2], max_draws=8)
        without = self.cal.predict(
            self.result, NEW_CONTROLS[:2], include_noise=False, max_draws=8
        )
        self.assertEqual(without.noise, (0.0, 0.0))
        for a, b in zip(with_noise.total_variance, without.total_variance, strict=True):
            self.assertAlmostEqual(a - b, NOISE_SD**2, delta=1e-12)

    def test_single_draw_has_no_parameter_variance(self) -> None:
        """With one draw the mixture has no between-draw part."""
        one = self.cal.predict(self.result, NEW_CONTROLS[:1], max_draws=1)
        self.assertEqual(one.draws, 1)
        self.assertEqual(one.parameter, (0.0,))

    def test_interval_widens_with_level(self) -> None:
        """A 99 percent mixture interval contains the 80 percent one."""
        narrow = self.cal.predict(self.result, NEW_CONTROLS[:1], level=0.8, max_draws=6)
        wide = self.cal.predict(self.result, NEW_CONTROLS[:1], level=0.99, max_draws=6)
        self.assertLess(wide.lower[0], narrow.lower[0])
        self.assertGreater(wide.upper[0], narrow.upper[0])


class EstimatedNoiseTest(unittest.TestCase):
    """The noise sd as a parameter with a log-normal prior."""

    def test_estimated_noise_samples_and_predicts(self) -> None:
        """Noise coordinate appears, is plausible, and enters the prediction."""
        controls, parameters, outputs = simulator_design(12)
        field_x, field_y = field_data(3, 6)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cal = KOHEmulatorCalibration(
                controls,
                parameters,
                outputs,
                field_x,
                field_y,
                theta_bounds=[(0.0, 2.0)],
                noise_prior=LogNormalPrior(math.log(0.05), 0.5),
                amplitude_prior=AMPLITUDE,
                emulator_regularization=GaussianProcessRegularization(),
                seed=2,
            )
            self.assertEqual(cal.names[-1], "log_noise_sd")
            result = cal.sample(draws=40, warmup=40, seed=3)
        noise = math.exp(result.mean()[-1])
        self.assertTrue(0.01 < noise < 0.3)
        prediction = cal.predict(result, NEW_CONTROLS[:1], max_draws=10)
        self.assertGreater(prediction.noise[0], 0.0)
        self.assertAlmostEqual(
            prediction.component_sum()[0], prediction.total_variance[0], delta=1e-12
        )


class ReproducibilityAndFailureTest(unittest.TestCase):
    """Determinism, validation, and counted numerical failures."""

    @classmethod
    def setUpClass(cls) -> None:
        """Build a base problem."""
        cls.cal = build(5, runs=12, count=4)

    def arguments(self) -> dict[str, object]:
        """Return valid constructor arguments for mutation."""
        controls, parameters, outputs = simulator_design(12)
        field_x, field_y = field_data(5, 4)
        return {
            "simulator_controls": controls,
            "simulator_parameters": parameters,
            "simulator_outputs": outputs,
            "field_controls": field_x,
            "field_outputs": field_y,
            "theta_bounds": [(0.0, 2.0)],
            "noise_sd": NOISE_SD,
            "emulator_regularization": GaussianProcessRegularization(),
        }

    def construct(self, **changes: object) -> KOHEmulatorCalibration:
        """Construct with modified arguments, silencing library warnings."""
        arguments = {**self.arguments(), **changes}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return KOHEmulatorCalibration(**arguments)  # type: ignore[arg-type]

    def test_same_seed_same_draws(self) -> None:
        """Two runs with one seed return identical draws."""
        first = self.cal.sample(draws=6, warmup=4, seed=12)
        second = self.cal.sample(draws=6, warmup=4, seed=12)
        self.assertEqual(first.samples, second.samples)

    def test_prior_support_and_dimension(self) -> None:
        """Outside the box the posterior is -inf; wrong lengths are rejected."""
        self.assertEqual(
            self.cal.log_posterior((2.5, math.log(0.1), math.log(0.4))), -math.inf
        )
        with self.assertRaises(ValueError):
            self.cal.log_posterior((1.0,))
        with self.assertRaises(ValueError):
            self.cal.log_prior((1.0, math.nan, 0.0))
        self.assertEqual(len(self.cal.names), self.cal.dimension)

    def test_constructor_rejects_bad_input(self) -> None:
        """Invalid shapes, bounds, noise specifications and kernels raise."""
        controls, _, _ = simulator_design(12)
        with self.assertRaises(ValueError):
            self.construct(theta_bounds=[(2.0, 0.0)])
        with self.assertRaises(ValueError):
            self.construct(theta_bounds=[(0.0, 1.0), (0.0, 1.0)])
        with self.assertRaises(ValueError):
            self.construct(noise_sd=None)
        with self.assertRaises(ValueError):
            self.construct(noise_prior=LogNormalPrior(0.0, 1.0))
        with self.assertRaises(ValueError):
            self.construct(noise_sd=-1.0)
        with self.assertRaises(ValueError):
            self.construct(noise_sd=(0.05, 0.05))
        with self.assertRaises(ValueError):
            self.construct(simulator_outputs=(1.0, 2.0))
        with self.assertRaises(ValueError):
            self.construct(field_outputs=(1.0,))
        with self.assertRaises(ValueError):
            self.construct(field_outputs=(math.nan,) * 4)
        with self.assertRaises(ValueError):
            self.construct(discrepancy_kernel="rbf")
        with self.assertRaises(ValueError):
            self.construct(field_controls=((0.1, 0.2),) * 4)
        with self.assertRaises(ValueError):
            self.construct(simulator_controls=((0.5,),) * len(controls))
        with self.assertRaises(TypeError):
            self.construct(emulator="not an emulator")

    def test_emulator_must_match_data(self) -> None:
        """A supplied emulator fitted to other runs is rejected."""
        other = self.cal.emulator
        controls, parameters, outputs = simulator_design(12)
        with self.assertRaises(ValueError):
            self.construct(
                emulator=other,
                simulator_outputs=tuple(v + 1.0 for v in outputs),
            )
        reused = self.construct(emulator=other)
        self.assertIs(reused.emulator, other)

    def test_sample_and_predict_validation(self) -> None:
        """Bad samplers, chain counts and foreign results are rejected."""
        with self.assertRaises(ValueError):
            self.cal.sample(sampler="nuts")
        with self.assertRaises(ValueError):
            self.cal.sample(chains=1)
        self.assertEqual(SAMPLERS, ("slice", "tmcmc"))
        result = self.cal.sample(draws=6, warmup=4, seed=1)
        other = self.cal.without_discrepancy()
        with self.assertRaises(ValueError):
            other.predict(result, NEW_CONTROLS)
        with self.assertRaises(ValueError):
            other.identifiability(result)
        with self.assertRaises(ValueError):
            self.cal.predict(result, ((0.1, 0.2),))
        with self.assertRaises(ValueError):
            self.cal.predict(result, NEW_CONTROLS, level=1.0)
        with self.assertRaises(ValueError):
            self.cal.predict(result, NEW_CONTROLS, max_draws=0)
        with self.assertRaises(ValueError):
            self.cal.identifiability(result, truth=(1.0, 2.0))
        with self.assertRaises(ValueError):
            self.cal.identifiability(result, contraction_threshold=2.0)
        with self.assertRaises(ValueError):
            result.theta_interval(3)
        with self.assertRaises(ValueError):
            result.theta_interval(0, level=0.0)
        with self.assertRaises(ValueError):
            result.column("missing")
        self.assertIsInstance(result, KOHResult)

    def test_unequal_known_noise_needs_prediction_noise(self) -> None:
        """Per-observation noise demands an explicit noise sd for new data."""
        cal = self.construct(noise_sd=(0.04, 0.05, 0.06, 0.05))
        result = cal.sample(draws=6, warmup=4, seed=1)
        with self.assertRaises(ValueError):
            cal.predict(result, NEW_CONTROLS[:1], max_draws=3)
        prediction = cal.predict(result, NEW_CONTROLS[:1], noise_sd=0.05, max_draws=3)
        self.assertAlmostEqual(prediction.noise[0], 0.05**2, delta=1e-15)

    def test_numerical_failures_are_counted_not_hidden(self) -> None:
        """Duplicate field points with vanishing noise give a counted failure.

        The covariance loses positive definiteness; the sampler target reports
        ``-inf`` and counts it, while the public likelihood raises.
        """
        duplicated = self.construct(
            field_controls=((0.5,),) * 4,
            noise_sd=1e-200,
            discrepancy=False,
        )
        with self.assertRaises(ValueError):
            duplicated.log_conditional_likelihood((1.0,))
        before = duplicated._failures
        self.assertEqual(duplicated._posterior_target((1.0,)), -math.inf)
        self.assertEqual(duplicated._failures, before + 1)

    def test_log_normal_prior_validation(self) -> None:
        """The prior rejects a non-positive sd and has the normal density."""
        with self.assertRaises(ValueError):
            LogNormalPrior(0.0, 0.0)
        prior = LogNormalPrior(0.0, 1.0)
        self.assertAlmostEqual(
            prior.log_density(0.0), -0.5 * math.log(2.0 * math.pi), delta=1e-14
        )


@unittest.skipUnless(os.environ.get("PROBSTAT_SLOW") == "1", "slow tier")
class CoverageStudyTest(unittest.TestCase):
    """Measured coverage over predeclared seeds (slow tier)."""

    SEEDS = tuple(range(201, 211))

    def test_theta_and_reality_coverage(self) -> None:
        """Count 95 percent coverage of theta and of reality over ten datasets."""
        hits = {True: 0, False: 0}
        reality_hits = {True: 0, False: 0}
        reality_total = 0
        for seed in self.SEEDS:
            cal = build(seed, runs=12)
            for discrepancy in (True, False):
                problem = cal if discrepancy else cal.without_discrepancy()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = problem.sample(draws=150, warmup=100, seed=seed)
                low, high = result.theta_interval(0)
                hits[discrepancy] += int(low <= THETA_TRUE <= high)
                prediction = problem.predict(
                    result, NEW_CONTROLS, include_noise=False, max_draws=40
                )
                for x, lo, hi in zip(
                    NEW_CONTROLS, prediction.lower, prediction.upper, strict=True
                ):
                    reality_hits[discrepancy] += int(lo <= reality(x[0]) <= hi)
                if discrepancy:
                    reality_total += len(NEW_CONTROLS)
        print(
            f"\nKOH coverage over {len(self.SEEDS)} seeds: theta with discrepancy "
            f"{hits[True]}, without {hits[False]}; reality "
            f"{reality_hits[True]}/{reality_total} with, "
            f"{reality_hits[False]}/{reality_total} without"
        )
        self.assertEqual(hits[False], 0)
        self.assertGreaterEqual(hits[True], 8)
        self.assertGreater(reality_hits[True], reality_hits[False])


if __name__ == "__main__":
    unittest.main()
