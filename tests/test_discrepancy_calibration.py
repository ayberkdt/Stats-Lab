"""Tests for Kennedy-O'Hagan discrepancy calibration and heteroscedastic noise."""

from __future__ import annotations

import math
import random
import unittest
from collections.abc import Mapping

from probstat.distributions import LogNormal, Normal, Uniform
from probstat.inverse import (
    CalibrationProblem,
    DiscrepancyCalibration,
    DiscrepancyProcess,
    Experiment,
    KnownNoise,
    LogLinearNoise,
    NoiseModel,
    discrepancy_kernel,
)
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.uq import QuantityOfInterest

DECAY_RATE = 0.4
DECAY_NOISE = 0.03
DECAY_POINTS = tuple(4.0 * i / 14 for i in range(15))
HELD_OUT = (0.65, 1.9, 3.14)


def decay_simulator(parameters: Mapping[str, float]) -> Mapping[str, float]:
    """Return the simulator ``4 exp(-k x)``."""
    return {"y": 4.0 * math.exp(-parameters["k"] * parameters["x"])}


def decay_truth(x: float) -> float:
    """Return the physical process: decay plus a smooth oscillating bias."""
    return 4.0 * math.exp(-DECAY_RATE * x) + 0.25 * math.sin(2.5 * x)


def decay_model() -> ProbabilisticModel:
    """Build the one-parameter decay simulator with a log-normal prior."""
    prior = JointInput.independent(RandomInput("k", LogNormal(math.log(0.5), 0.5)))
    return ProbabilisticModel(prior, decay_simulator, (QuantityOfInterest("y", "y"),))


def decay_problem() -> DiscrepancyCalibration:
    """Build the seeded synthetic problem with a known smooth discrepancy."""
    generator = random.Random(11)
    experiments = tuple(
        Experiment({"y": decay_truth(x) + generator.gauss(0.0, DECAY_NOISE)}, {"x": x})
        for x in DECAY_POINTS
    )
    discrepancy = DiscrepancyProcess(
        ("x",),
        standard_deviation_prior=LogNormal(math.log(0.2), 0.75),
        length_scale_priors=(LogNormal(math.log(0.5), 0.5),),
    )
    return DiscrepancyCalibration(
        decay_model(), experiments, "y", KnownNoise(DECAY_NOISE), discrepancy
    )


def amplitude_decay(parameters: Mapping[str, float]) -> Mapping[str, float]:
    """Return ``A exp(-k t)``."""
    return {"y": parameters["A"] * math.exp(-parameters["k"] * parameters["t"])}


def amplitude_problems(
    discrepancy: DiscrepancyProcess | None,
) -> tuple[CalibrationProblem, DiscrepancyCalibration]:
    """Build matching CalibrationProblem and DiscrepancyCalibration objects."""
    prior = JointInput.independent(
        RandomInput("A", Uniform(1.0, 10.0)),
        RandomInput("k", LogNormal(math.log(0.5), 1.0)),
    )
    model = ProbabilisticModel(prior, amplitude_decay, (QuantityOfInterest("y", "y"),))
    generator = random.Random(7)
    experiments = tuple(
        Experiment(
            {"y": 5.0 * math.exp(-0.3 * t) + generator.gauss(0.0, 0.1)}, {"t": t}
        )
        for t in (0.5 * i for i in range(21))
    )
    plain = CalibrationProblem(
        model, experiments, (NoiseModel("y", standard_deviation=0.1),)
    )
    koh = DiscrepancyCalibration(model, experiments, "y", KnownNoise(0.1), discrepancy)
    return plain, koh


def straight_line(parameters: Mapping[str, float]) -> Mapping[str, float]:
    """Return ``c0 + c1 x``."""
    return {"y": parameters["c0"] + parameters["c1"] * parameters["x"]}


def line_covariate(conditions: Mapping[str, float]) -> tuple[float, ...]:
    """Return the log-noise covariate ``g(c) = x``."""
    return (conditions["x"],)


def constant_simulator(parameters: Mapping[str, float]) -> Mapping[str, float]:
    """Return a constant simulator ``eta = c`` (confounded with ``delta``)."""
    return {"y": parameters["c"]}


class KernelAndLikelihoodTests(unittest.TestCase):
    """Check the kernel and the marginal likelihood against hand computations."""

    def test_kernels_match_closed_forms(self) -> None:
        """Squared-exponential and Matern-5/2 values match explicit formulas."""
        r = math.sqrt((0.3 / 0.5) ** 2 + (1.0 / 2.0) ** 2)
        se = discrepancy_kernel(
            "squared_exponential", (0.0, 1.0), (0.3, 2.0), (0.5, 2.0), 1.7
        )
        self.assertAlmostEqual(se, 1.7 * math.exp(-0.5 * r * r), places=14)
        matern = discrepancy_kernel("matern52", (0.0, 1.0), (0.3, 2.0), (0.5, 2.0), 1.7)
        expected = (
            1.7
            * (1.0 + math.sqrt(5.0) * r + 5.0 * r * r / 3.0)
            * math.exp(-math.sqrt(5.0) * r)
        )
        self.assertAlmostEqual(matern, expected, places=14)
        self.assertEqual(
            discrepancy_kernel("matern52", (1.0,), (1.0,), (0.2,), 2.5), 2.5
        )

    def test_marginal_likelihood_matches_dense_hand_computation(self) -> None:
        """A 3-point case agrees with a cofactor inverse and Sarrus determinant."""

        def line(parameters: Mapping[str, float]) -> Mapping[str, float]:
            """Return ``theta x``."""
            return {"y": parameters["theta"] * parameters["x"]}

        prior = JointInput.independent(RandomInput("theta", Normal(0.0, 1.0)))
        model = ProbabilisticModel(prior, line, (QuantityOfInterest("y", "y"),))
        xs, ys, noise = (0.0, 0.7, 1.5), (0.1, 0.9, 1.4), (0.2, 0.3, 0.25)
        experiments = tuple(
            Experiment({"y": y}, {"x": x}) for x, y in zip(xs, ys, strict=True)
        )
        discrepancy = DiscrepancyProcess(
            ("x",),
            standard_deviation_prior=LogNormal(0.0, 1.0),
            length_scale_priors=(LogNormal(0.0, 1.0),),
        )
        problem = DiscrepancyCalibration(
            model, experiments, "y", KnownNoise(noise), discrepancy
        )
        theta, sd, ell = 0.8, 0.6, 0.9
        c = [
            [
                sd * sd * math.exp(-0.5 * ((xs[i] - xs[j]) / ell) ** 2)
                + (noise[i] ** 2 if i == j else 0.0)
                for j in range(3)
            ]
            for i in range(3)
        ]
        det = (
            c[0][0] * (c[1][1] * c[2][2] - c[1][2] * c[2][1])
            - c[0][1] * (c[1][0] * c[2][2] - c[1][2] * c[2][0])
            + c[0][2] * (c[1][0] * c[2][1] - c[1][1] * c[2][0])
        )
        cofactor = [
            [
                (
                    c[(i + 1) % 3][(j + 1) % 3] * c[(i + 2) % 3][(j + 2) % 3]
                    - c[(i + 1) % 3][(j + 2) % 3] * c[(i + 2) % 3][(j + 1) % 3]
                )
                for j in range(3)
            ]
            for i in range(3)
        ]
        inverse = [[cofactor[j][i] / det for j in range(3)] for i in range(3)]
        r = [ys[i] - theta * xs[i] for i in range(3)]
        quad = sum(r[i] * inverse[i][j] * r[j] for i in range(3) for j in range(3))
        expected = -0.5 * quad - 0.5 * math.log(det) - 1.5 * math.log(2.0 * math.pi)
        vector = (theta, math.log(sd), math.log(ell))
        self.assertAlmostEqual(problem.log_likelihood(vector), expected, places=12)
        prior_part = (
            -0.5 * theta * theta
            - 0.5 * math.log(2.0 * math.pi)
            + LogNormal(0.0, 1.0).log_pdf(sd)
            + math.log(sd)
            + LogNormal(0.0, 1.0).log_pdf(ell)
            + math.log(ell)
        )
        self.assertAlmostEqual(
            problem.log_posterior(vector), expected + prior_part, places=10
        )


class ReductionTests(unittest.TestCase):
    """With no discrepancy the method reproduces CalibrationProblem."""

    def test_zero_discrepancy_reproduces_calibration_problem(self) -> None:
        """Log posteriors, MAP, covariance, and seeded chains agree."""
        for discrepancy in (
            None,
            DiscrepancyProcess(("t",), standard_deviation=0.0, length_scales=(1.0,)),
        ):
            plain, koh = amplitude_problems(discrepancy)
            for theta in ((5.0, 0.3), (3.0, 0.8), (9.5, 0.05)):
                self.assertAlmostEqual(
                    plain.log_posterior(theta), koh.log_posterior(theta), places=10
                )
            laplace, fit = plain.laplace_posterior(), koh.fit()
            assert fit.covariance is not None
            for i in range(2):
                self.assertAlmostEqual(laplace.mode[i], fit.estimate[i], places=7)
                for j in range(2):
                    self.assertAlmostEqual(
                        laplace.covariance[i][j], fit.covariance[i][j], places=8
                    )
            first = plain.sample_posterior(draws=300, warmup=200, chains=2, seed=5)
            second = koh.sample_posterior(draws=300, warmup=200, chains=2, seed=5)
            largest = max(
                abs(a - b)
                for ca, cb in zip(first.mcmc.chains, second.mcmc.chains, strict=True)
                for sa, sb in zip(ca.samples, cb.samples, strict=True)
                for a, b in zip(sa, sb, strict=True)
            )
            self.assertLess(largest, 1e-7)

    def test_dense_path_converges_to_diagonal_path(self) -> None:
        """A tiny discrepancy variance on the Cholesky path matches the plain one."""
        plain, koh = amplitude_problems(
            DiscrepancyProcess(("t",), standard_deviation=1e-7, length_scales=(1.0,))
        )
        self.assertTrue(koh.is_dense)
        for theta in ((5.0, 0.3), (4.0, 0.5)):
            self.assertAlmostEqual(
                plain.log_likelihood(theta), koh.log_likelihood(theta), places=6
            )


class SyntheticDiscrepancyTests(unittest.TestCase):
    """Synthetic truth with a known smooth discrepancy."""

    @classmethod
    def setUpClass(cls) -> None:
        """Sample the posterior once for all assertions in this class."""
        cls.problem = decay_problem()
        cls.posterior = cls.problem.sample_posterior(
            draws=500, warmup=400, chains=2, seed=3
        )
        cls.predictions = cls.problem.predict(
            cls.posterior, tuple({"x": x} for x in HELD_OUT), max_draws=400
        )

    def test_corrected_interval_covers_truth_where_simulator_misses(self) -> None:
        """95% corrected intervals cover all held-out truths; simulator misses x=0.65.

        Recorded run (this seed and chain length): truth 3.3338 at x=0.65;
        corrected interval (3.300, 3.388) covers it, simulator-only interval
        (2.876, 3.262) does not. At x=1.9 (truth 1.6208) and x=3.14 (truth
        1.3892) the corrected intervals (1.601, 1.688) and (1.373, 1.459)
        cover the truth; the wide simulator-only intervals happen to as well.
        """
        for x, prediction in zip(HELD_OUT, self.predictions, strict=True):
            low, high = prediction.interval(0.95, kind="corrected")
            self.assertLessEqual(low, decay_truth(x))
            self.assertGreaterEqual(high, decay_truth(x))
        low, high = self.predictions[0].interval(0.95, kind="simulator")
        self.assertFalse(low <= decay_truth(0.65) <= high)
        self.assertLess(high, decay_truth(0.65))
        self.assertGreater(self.predictions[0].discrepancy_mean, 0.15)

    def test_prediction_moments_are_consistent(self) -> None:
        """Corrected variance obeys the law of total variance and narrows the gap."""
        for x, prediction in zip(HELD_OUT, self.predictions, strict=True):
            self.assertAlmostEqual(
                prediction.corrected_mean,
                prediction.simulator_mean + prediction.discrepancy_mean,
                places=12,
            )
            self.assertLess(
                abs(prediction.corrected_mean - decay_truth(x)),
                abs(prediction.simulator_mean - decay_truth(x)),
            )
            self.assertGreater(prediction.corrected_variance, 0.0)
            self.assertIsNone(prediction.noise_variances)
        with self.assertRaises(ValueError):
            self.predictions[0].interval(kind="observed")

    def test_posterior_is_converged_and_identifiable(self) -> None:
        """R-hat is near one and the identifiability report raises no warning."""
        for summary in self.posterior.summary():
            self.assertLess(summary.rank_normalized_rhat, 1.1)
        self.assertEqual(self.posterior.mcmc.names, ("k", "sigma_delta", "length[x]"))
        report = self.problem.identifiability(self.posterior)
        self.assertEqual(report.source, "mcmc")
        self.assertIsNone(report.warning)
        self.assertLess(report.standard_deviation_ratios["k"], 0.5)

    def test_map_fit_and_plug_in_prediction(self) -> None:
        """MAP converges near k = 0.4 and plug-in predictions are Gaussian."""
        fit = self.problem.fit()
        self.assertTrue(fit.converged)
        self.assertLess(
            abs(fit.as_mapping()["k"] - DECAY_RATE), 3 * fit.standard_errors[0]
        )
        (prediction,) = self.problem.predict(fit, ({"x": 0.65},))
        low, high = prediction.interval(0.95)
        mean = prediction.corrected_mean
        sd = math.sqrt(prediction.corrected_variance)
        self.assertAlmostEqual(low, mean - 1.959963984540054 * sd, places=6)
        self.assertAlmostEqual(high, mean + 1.959963984540054 * sd, places=6)
        self.assertEqual(
            prediction.interval(kind="simulator")[0], prediction.simulator[0]
        )

    def test_sampling_is_deterministic_by_seed(self) -> None:
        """Equal seeds reproduce chains exactly; another seed differs."""
        first = self.problem.sample_posterior(draws=40, warmup=40, chains=1, seed=9)
        second = self.problem.sample_posterior(draws=40, warmup=40, chains=1, seed=9)
        third = self.problem.sample_posterior(draws=40, warmup=40, chains=1, seed=10)
        self.assertEqual(first.vector.pooled(), second.vector.pooled())
        self.assertNotEqual(first.vector.pooled(), third.vector.pooled())


class HeteroscedasticNoiseTests(unittest.TestCase):
    """Known per-observation variances and the log-linear variance model."""

    def test_log_linear_mle_recovers_coefficients(self) -> None:
        """MLE of (a, b) lies within 3 SE of the truth (-2, 0.5).

        Recorded run: a = -1.891 (SE 0.101), b = 0.466 (SE 0.044), i.e.
        z = +1.07 and -0.76.
        """
        prior = JointInput.independent(
            RandomInput("c0", Normal(0.0, 10.0)), RandomInput("c1", Normal(0.0, 10.0))
        )
        model = ProbabilisticModel(
            prior, straight_line, (QuantityOfInterest("y", "y"),)
        )
        generator = random.Random(2026)
        experiments = tuple(
            Experiment(
                {"y": 1.0 + 0.7 * x + generator.gauss(0.0, math.exp(-2.0 + 0.5 * x))},
                {"x": x},
            )
            for x in (4.0 * i / 199 for i in range(200))
        )
        noise = LogLinearNoise(
            line_covariate, Normal(0.0, 3.0), (Normal(0.0, 3.0),), ("x",)
        )
        problem = DiscrepancyCalibration(model, experiments, "y", noise)
        self.assertEqual(problem.vector_names, ("c0", "c1", "noise_a", "noise_b[x]"))
        fit = problem.fit(objective="likelihood")
        self.assertTrue(fit.converged)
        errors = fit.standard_errors
        self.assertLess(abs(fit.estimate[2] + 2.0), 3.0 * errors[2])
        self.assertLess(abs(fit.estimate[3] - 0.5), 3.0 * errors[3])
        sds = problem.noise_standard_deviations(fit.estimate)
        self.assertAlmostEqual(sds[0], math.exp(fit.estimate[2]), places=12)
        (prediction,) = problem.predict(fit, ({"x": 2.0},))
        assert prediction.noise_variances is not None
        self.assertAlmostEqual(
            prediction.noise_variances[0],
            math.exp(2.0 * (fit.estimate[2] + 2.0 * fit.estimate[3])),
            places=12,
        )

    def test_known_per_observation_noise_matches_independent_terms(self) -> None:
        """Known heteroscedastic variances give the sum of Gaussian log-densities."""
        prior = JointInput.independent(
            RandomInput("c0", Normal(0.0, 1.0)), RandomInput("c1", Normal(0.0, 1.0))
        )
        model = ProbabilisticModel(
            prior, straight_line, (QuantityOfInterest("y", "y"),)
        )
        xs, ys, sds = (0.0, 1.0, 2.0), (0.2, 1.1, 1.7), (0.1, 0.2, 0.4)
        experiments = tuple(
            Experiment({"y": y}, {"x": x}) for x, y in zip(xs, ys, strict=True)
        )
        problem = DiscrepancyCalibration(model, experiments, "y", KnownNoise(sds))
        expected = sum(
            -0.5 * math.log(2.0 * math.pi * s * s)
            - 0.5 * ((y - 0.1 - 0.8 * x) / s) ** 2
            for x, y, s in zip(xs, ys, sds, strict=True)
        )
        self.assertAlmostEqual(problem.log_likelihood((0.1, 0.8)), expected, places=12)
        (prediction,) = problem.predict(
            ((0.1, 0.8),), ({"x": 3.0},), observation_standard_deviations=(0.5,)
        )
        self.assertEqual(prediction.noise_variances, (0.25,))
        low, high = prediction.interval(0.95, kind="observed")
        self.assertAlmostEqual(high - low, 2 * 1.959963984540054 * 0.5, places=6)


class IdentifiabilityTests(unittest.TestCase):
    """The confounding report warns when theta and delta cannot be separated."""

    def test_confounded_constant_simulator_triggers_warning(self) -> None:
        """A constant simulator with a long-range discrepancy is flagged."""
        prior = JointInput.independent(RandomInput("c", Normal(0.0, 1.0)))
        model = ProbabilisticModel(
            prior, constant_simulator, (QuantityOfInterest("y", "y"),)
        )
        generator = random.Random(4)
        experiments = tuple(
            Experiment(
                {"y": 0.8 + 0.3 * math.sin(x) + generator.gauss(0.0, 0.05)}, {"x": x}
            )
            for x in (0.5 * i for i in range(8))
        )
        discrepancy = DiscrepancyProcess(
            ("x",),
            standard_deviation_prior=LogNormal(math.log(3.0), 0.3),
            length_scales=(3.0,),
        )
        problem = DiscrepancyCalibration(
            model, experiments, "y", KnownNoise(0.05), discrepancy
        )
        report = problem.identifiability(problem.fit())
        self.assertEqual(report.source, "laplace")
        self.assertGreater(report.standard_deviation_ratios["c"], 0.8)
        assert report.warning is not None
        self.assertIn("Brynjarsdottir", report.warning)
        self.assertIn("log_sigma_delta", report.cross_correlations["c"])


class ValidationTests(unittest.TestCase):
    """Invalid declarations and arguments raise clear errors."""

    def test_declarations_are_validated(self) -> None:
        """Bad discrepancy, noise, and problem declarations raise."""
        with self.assertRaises(ValueError):
            DiscrepancyProcess((), standard_deviation=1.0, length_scales=())
        with self.assertRaises(ValueError):
            DiscrepancyProcess(
                ("x",), kernel="cubic", standard_deviation=1.0, length_scales=(1.0,)
            )
        with self.assertRaises(ValueError):
            DiscrepancyProcess(("x",), length_scales=(1.0,))
        with self.assertRaises(ValueError):
            DiscrepancyProcess(("x",), standard_deviation=-1.0, length_scales=(1.0,))
        with self.assertRaises(ValueError):
            DiscrepancyProcess(("x",), standard_deviation=1.0, length_scales=(0.0,))
        with self.assertRaises(ValueError):
            DiscrepancyProcess(
                ("x", "x"), standard_deviation=1.0, length_scales=(1.0, 1.0)
            )
        with self.assertRaises(ValueError):
            KnownNoise(0.0)
        with self.assertRaises(ValueError):
            KnownNoise((0.1, -0.2))
        with self.assertRaises(TypeError):
            LogLinearNoise(3, Normal(0.0, 1.0))  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            LogLinearNoise(line_covariate, Normal(0.0, 1.0), (), ("x",))
        model = decay_model()
        experiments = (
            Experiment({"y": 1.0}, {"x": 0.0}),
            Experiment({"y": 0.5}, {"x": 1.0}),
        )
        with self.assertRaises(ValueError):
            DiscrepancyCalibration(model, experiments, "z", KnownNoise(0.1))
        with self.assertRaises(ValueError):
            DiscrepancyCalibration(model, experiments, "y", KnownNoise((0.1, 0.1, 0.1)))
        with self.assertRaises(ValueError):
            DiscrepancyCalibration(
                model,
                experiments,
                "y",
                KnownNoise(0.1),
                DiscrepancyProcess(
                    ("w",), standard_deviation=1.0, length_scales=(1.0,)
                ),
            )
        with self.assertRaises(ValueError):
            DiscrepancyCalibration(
                model,
                (Experiment({"y": 1.0}, {"x": 0.0, "k": 1.0}),),
                "y",
                KnownNoise(0.1),
            )
        with self.assertRaises(TypeError):
            DiscrepancyCalibration(model, experiments, "y", 0.1)  # type: ignore[arg-type]

    def test_run_arguments_are_validated(self) -> None:
        """Bad fit, sampling, prediction, and diagnostic arguments raise."""
        problem = decay_problem()
        with self.assertRaises(ValueError):
            problem.fit(objective="evidence")
        with self.assertRaises(ValueError):
            problem.fit(initial=(0.4,))
        with self.assertRaises(ValueError):
            problem.sample_posterior(method="gibbs")
        with self.assertRaises(ValueError):
            problem.sample_posterior(chains=0)
        with self.assertRaises(ValueError):
            problem.log_likelihood((0.4, 0.0))
        vector = problem.default_start()
        with self.assertRaises(ValueError):
            problem.predict((vector,), ())
        with self.assertRaises(ValueError):
            problem.predict((vector,), ({"z": 1.0},))
        with self.assertRaises(ValueError):
            problem.predict((vector,), ({"x": 1.0, "k": 0.3},))
        with self.assertRaises(ValueError):
            problem.predict((), ({"x": 1.0},))
        fit = problem.fit()
        with self.assertRaises(ValueError):
            problem.identifiability(fit, correlation_warning=1.5)
        with self.assertRaises(ValueError):
            problem.predict(fit, ({"x": 1.0},))[0].interval(kind="latent")


if __name__ == "__main__":
    unittest.main()
