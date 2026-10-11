"""Tests for model-validation metrics, assessment, and credibility records."""

from __future__ import annotations

import math
import random
import unittest

from probstat.distributions import Normal
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.uq import NumericalErrorBudget, QuantityOfInterest
from probstat.validation import (
    CredibilityRecord,
    PredictionStatement,
    ValidationAlternative,
    ValidationDomain,
    ValidationExperiment,
    VerificationEvidence,
    area_metric,
    asme_validation_comparison,
    assess_predictions,
    assess_validation,
    budget_standard_uncertainty,
    interpret_log_bayes_factor,
    mahalanobis_metric,
    predict_at_conditions,
    probability_integral_transform,
    reliability_metric,
    silverman_bandwidth,
    u_pooling,
    uniform_area,
    validation_bayes_factor,
)

LOADS = (1.0, 2.0, 3.0, 4.0)
STIFFNESS_MEAN = 2.0
STIFFNESS_SD = 0.1
NOISE = 0.05


def _spring_model() -> ProbabilisticModel:
    """Return ``y = k * load`` with ``k ~ N(2, 0.1^2)`` and load as a condition."""
    inputs = JointInput.independent(
        RandomInput("k", Normal(STIFFNESS_MEAN, STIFFNESS_SD))
    )

    def forward(parameters):
        """Evaluate the linear spring deflection."""
        return {"y": parameters["k"] * parameters["load"]}

    return ProbabilisticModel(
        inputs,
        forward,
        (QuantityOfInterest("y", "deflection", "mm"),),
        name="spring",
    )


def _experiments(
    *, bias: float = 0.0, replicates: int = 6, seed: int = 11
) -> list[ValidationExperiment]:
    """Generate experiments from the spring law itself, optionally biased."""
    rng = random.Random(seed)
    experiments = []
    for index, load in enumerate(LOADS):
        values = [
            rng.gauss(STIFFNESS_MEAN, STIFFNESS_SD) * load
            + rng.gauss(0.0, NOISE)
            + bias
            for _ in range(replicates)
        ]
        experiments.append(
            ValidationExperiment(
                f"test-{index}",
                values,
                {"load": load},
                measurement_uncertainty=NOISE,
                provenance="synthetic campaign generated in the test suite",
            )
        )
    return experiments


def _normal_grid(mean: float, sd: float, size: int) -> list[float]:
    """Return deterministic normal quantiles at ``(i + 1/2) / size``."""
    law = Normal(mean, sd)
    return [law.quantile((i + 0.5) / size) for i in range(size)]


class AreaMetricTests(unittest.TestCase):
    """Exact step-function area metric against hand and analytic values."""

    def test_identical_distributions_give_zero(self) -> None:
        """Identical samples have zero area."""
        values = [0.3, -1.2, 4.0, 2.2, 2.2]
        self.assertEqual(area_metric(values, list(reversed(values))), 0.0)

    def test_point_masses_shifted_by_d(self) -> None:
        """Two point masses ``d`` apart have area ``d``."""
        self.assertAlmostEqual(area_metric([1.0] * 3, [3.5]), 2.5, places=14)

    def test_point_mass_prediction_equals_mean_absolute_distance(self) -> None:
        """A point-mass prediction at 0 vs data {1, 3} gives (1 + 3) / 2."""
        self.assertAlmostEqual(area_metric([0.0, 0.0], [1.0, 3.0]), 2.0, places=14)

    def test_two_step_functions_by_hand(self) -> None:
        """F on {0,1,2,3}, S on {1.5,2.5}: .25+.25+0+.125+.125 = .75."""
        self.assertAlmostEqual(
            area_metric([0.0, 1.0, 2.0, 3.0], [1.5, 2.5]), 0.75, places=14
        )
        self.assertAlmostEqual(
            area_metric([1.5, 2.5], [0.0, 1.0, 2.0, 3.0]), 0.75, places=14
        )

    def test_normal_against_point_mass_matches_analytic_area(self) -> None:
        """Area to a point y is E|X-y| = s[z(2Phi(z)-1) + 2 phi(z)]."""
        mean, sd, y = 1.0, 2.0, 2.0
        z = (y - mean) / sd
        phi = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        exact = sd * (z * (2.0 * Normal().cdf(z) - 1.0) + 2.0 * phi)
        grid = _normal_grid(mean, sd, 20000)
        self.assertAlmostEqual(area_metric(grid, [y]), exact, delta=2e-3)

    def test_invalid_samples_raise(self) -> None:
        """Empty and non-finite samples are rejected."""
        with self.assertRaises(ValueError):
            area_metric([], [1.0])
        with self.assertRaises(ValueError):
            area_metric([1.0], [math.nan])
        with self.assertRaises(TypeError):
            area_metric(["a"], [1.0])  # type: ignore[list-item]


class UPoolingTests(unittest.TestCase):
    """Probability-integral-transform pooling across experiments."""

    def test_uniform_area_hand_values(self) -> None:
        """A single u at 1/2 gives 1/4; a single u at 0 gives the maximum 1/2."""
        self.assertAlmostEqual(uniform_area([0.5]), 0.25, places=14)
        self.assertAlmostEqual(uniform_area([0.0]), 0.5, places=14)
        grid = [(i + 0.5) / 100 for i in range(100)]
        self.assertLess(uniform_area(grid), 0.01)

    def test_uniform_area_rejects_values_outside_unit_interval(self) -> None:
        """A u value above one is invalid."""
        with self.assertRaises(ValueError):
            uniform_area([1.5])

    def test_pit_is_predictive_ecdf(self) -> None:
        """F_S(2.5) over {1,2,3,4} is one half."""
        self.assertEqual(probability_integral_transform([4, 1, 3, 2], 2.5), 0.5)

    def _pairs(self, bias: float) -> list[tuple[list[float], list[float]]]:
        """Build seeded predictive samples and observations for 12 experiments."""
        rng = random.Random(2024)
        pairs = []
        for index in range(12):
            mean = 3.0 * index
            predictions = [rng.gauss(mean, 1.0) for _ in range(800)]
            observations = [rng.gauss(mean + bias, 1.0) for _ in range(4)]
            pairs.append((predictions, observations))
        return pairs

    def test_data_from_model_gives_small_metric(self) -> None:
        """Observations drawn from the predictive laws pool to near-uniform."""
        result = u_pooling(self._pairs(0.0))
        self.assertEqual(result.count, 48)
        self.assertLess(result.area, 0.08)
        self.assertGreater(result.ks.p_value, 0.05)

    def test_biased_data_gives_large_metric(self) -> None:
        """A one-sigma bias in every experiment is detected."""
        result = u_pooling(self._pairs(1.0))
        self.assertGreater(result.area, 0.15)
        self.assertLess(result.ks.p_value, 1e-3)

    def test_empty_pooling_raises(self) -> None:
        """At least one experiment is required."""
        with self.assertRaises(ValueError):
            u_pooling([])


class GaussianAndReliabilityTests(unittest.TestCase):
    """Mahalanobis residuals and the reliability metric."""

    def test_identity_covariance(self) -> None:
        """Residuals (1, 2) with identity covariance give D^2 = 5, p = e^-2.5."""
        result = mahalanobis_metric([1.0, 2.0], [0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]])
        self.assertAlmostEqual(result.squared_distance, 5.0, places=12)
        self.assertAlmostEqual(result.p_value, math.exp(-2.5), places=10)
        self.assertEqual(result.standardized_residuals, (1.0, 2.0))

    def test_correlated_covariance_matches_explicit_inverse(self) -> None:
        """D^2 equals r' C^-1 r with the explicit 2x2 inverse."""
        c = [[2.0, 0.6], [0.6, 1.0]]
        r = (1.0, -0.5)
        det = c[0][0] * c[1][1] - c[0][1] ** 2
        inverse = [[c[1][1] / det, -c[0][1] / det], [-c[0][1] / det, c[0][0] / det]]
        expected = sum(r[i] * inverse[i][j] * r[j] for i in range(2) for j in range(2))
        result = mahalanobis_metric(r, [0.0, 0.0], c)
        self.assertAlmostEqual(result.squared_distance, expected, places=12)

    def test_invalid_covariance_raises(self) -> None:
        """Size, symmetry, and definiteness are checked."""
        with self.assertRaises(ValueError):
            mahalanobis_metric([1.0], [0.0, 0.0], [[1.0]])
        with self.assertRaises(ValueError):
            mahalanobis_metric([1.0, 1.0], [0.0, 0.0], [[1.0, 0.5], [0.0, 1.0]])
        with self.assertRaises(ValueError):
            mahalanobis_metric([1.0, 1.0], [0.0, 0.0], [[1.0, 2.0], [2.0, 1.0]])

    def test_reliability_metric_by_hand(self) -> None:
        """|{0,1,2,3} - 1| < 1.5 holds for three of four samples."""
        result = reliability_metric([0.0, 1.0, 2.0, 3.0], [1.0], 1.5)
        self.assertAlmostEqual(result.estimate, 0.75, places=14)
        self.assertAlmostEqual(result.standard_error, 0.25, places=14)
        with self.assertRaises(ValueError):
            reliability_metric([0.0, 1.0], [1.0], 0.0)
        with self.assertRaises(ValueError):
            reliability_metric([0.0], [1.0], 1.0)


class AsmeComparisonTests(unittest.TestCase):
    """ASME V&V 20 comparison-error arithmetic."""

    def test_comparison_error_and_validation_uncertainty(self) -> None:
        """E = 0.5 and u_val = sqrt(.09 + .16 + 1.44) = 1.3."""
        result = asme_validation_comparison(
            10.5,
            10.0,
            numerical_uncertainty=0.3,
            input_uncertainty=0.4,
            data_uncertainty=1.2,
        )
        self.assertAlmostEqual(result.comparison_error, 0.5, places=14)
        self.assertAlmostEqual(result.validation_uncertainty, 1.3, places=14)
        low, high = result.model_error_interval
        self.assertAlmostEqual(low, -2.1, places=12)
        self.assertAlmostEqual(high, 3.1, places=12)
        self.assertFalse(result.error_resolved)
        self.assertIn("not resolved", result.statement())

    def test_resolved_model_error(self) -> None:
        """|E| = 3 > 2 * 0.5 resolves the model-error sign."""
        result = asme_validation_comparison(
            7.0, 10.0, input_uncertainty=0.3, data_uncertainty=0.4
        )
        self.assertAlmostEqual(result.validation_uncertainty, 0.5, places=14)
        self.assertTrue(result.error_resolved)
        self.assertIn("resolved and approximately E = -3", result.statement())

    def test_numerical_error_budget_conversion(self) -> None:
        """A summed bound of 0.3 sqrt(3) becomes u_num = 0.3."""
        budget = NumericalErrorBudget(
            discretization=0.2 * math.sqrt(3.0), iterative=0.1 * math.sqrt(3.0)
        )
        self.assertAlmostEqual(budget_standard_uncertainty(budget), 0.3, places=14)
        result = asme_validation_comparison(1.0, 1.0, numerical_uncertainty=budget)
        self.assertAlmostEqual(result.numerical_uncertainty, 0.3, places=14)
        self.assertIn("sqrt(3)", result.numerical_uncertainty_source)

    def test_invalid_arguments_raise(self) -> None:
        """Negative uncertainties and non-positive coverage are rejected."""
        with self.assertRaises(ValueError):
            asme_validation_comparison(1.0, 1.0, data_uncertainty=-1.0)
        with self.assertRaises(ValueError):
            asme_validation_comparison(1.0, 1.0, coverage_factor=0.0)
        with self.assertRaises(TypeError):
            budget_standard_uncertainty(0.1)  # type: ignore[arg-type]


class BayesFactorTests(unittest.TestCase):
    """Bayes-factor sign sanity and explicit alternatives."""

    def setUp(self) -> None:
        """Draw seeded predictive samples from N(10, 1)."""
        rng = random.Random(5)
        self.predictions = [rng.gauss(10.0, 1.0) for _ in range(600)]
        self.rng = rng

    def test_data_from_model_favors_model(self) -> None:
        """Observations from the predictive law give log B > 0."""
        observations = [
            self.rng.gauss(10.0, math.sqrt(1.0 + 0.25)) for _ in range(8)
        ]
        result = validation_bayes_factor(
            self.predictions,
            observations,
            alternative=ValidationAlternative.inflated(5.0),
            noise_standard_deviation=0.5,
        )
        self.assertGreater(result.log_bayes_factor, 0.0)
        self.assertEqual(result.kernel_source, "measurement")
        self.assertIn("model", result.interpretation)

    def test_biased_data_favors_alternative(self) -> None:
        """Observations six predictive sds away give log B < 0."""
        observations = [16.5, 17.0, 16.0]
        result = validation_bayes_factor(
            self.predictions,
            observations,
            alternative=ValidationAlternative.inflated(5.0),
            noise_standard_deviation=0.5,
        )
        self.assertLess(result.log_bayes_factor, 0.0)
        self.assertIn("alternative", result.interpretation)
        self.assertLess(result.bayes_factor, 1.0)

    def test_uniform_and_normal_alternatives(self) -> None:
        """Data outside a uniform support make the alternative impossible."""
        outside = validation_bayes_factor(
            self.predictions,
            [10.0],
            alternative=ValidationAlternative.uniform(20.0, 30.0),
            noise_standard_deviation=0.5,
        )
        self.assertEqual(outside.log_bayes_factor, math.inf)
        self.assertEqual(outside.bayes_factor, math.inf)
        normal = validation_bayes_factor(
            self.predictions,
            [10.0],
            alternative=ValidationAlternative.normal(10.0, 20.0),
        )
        self.assertEqual(normal.kernel_source, "silverman")
        self.assertAlmostEqual(
            normal.kernel_standard_deviation,
            silverman_bandwidth(self.predictions),
            places=14,
        )
        self.assertGreater(normal.log_bayes_factor, 0.0)

    def test_invalid_alternatives_and_samples(self) -> None:
        """Invalid alternatives and degenerate inputs are rejected."""
        with self.assertRaises(ValueError):
            ValidationAlternative.inflated(1.0)
        with self.assertRaises(ValueError):
            ValidationAlternative.uniform(2.0, 1.0)
        with self.assertRaises(ValueError):
            ValidationAlternative("other", 1.0, 2.0, "x")
        with self.assertRaises(TypeError):
            validation_bayes_factor(
                self.predictions, [1.0], alternative="broad"  # type: ignore[arg-type]
            )
        with self.assertRaises(ValueError):
            validation_bayes_factor(
                [1.0, 1.0, 1.0],
                [1.0],
                alternative=ValidationAlternative.normal(0.0, 1.0),
            )

    def test_kass_raftery_categories(self) -> None:
        """2 log B thresholds 2, 6, 10 separate the categories."""
        self.assertIn("bare mention", interpret_log_bayes_factor(0.5))
        self.assertIn("positive", interpret_log_bayes_factor(2.0))
        self.assertIn(
            "strong evidence for the alternative", interpret_log_bayes_factor(-4.0)
        )
        self.assertIn("very strong", interpret_log_bayes_factor(6.0))


class ValidationDomainTests(unittest.TestCase):
    """Applicability-domain membership and extrapolation flags."""

    def test_one_dimensional_interval(self) -> None:
        """In 1-D the convex hull is the validation interval."""
        domain = ValidationDomain.from_experiments(_experiments())
        inside = domain.check({"load": 2.5})
        self.assertTrue(inside.inside_box)
        self.assertTrue(inside.inside_hull)
        self.assertFalse(inside.extrapolation)
        outside = domain.check({"load": 6.0})
        self.assertFalse(outside.inside_box)
        self.assertTrue(outside.extrapolation)
        self.assertEqual(outside.nearest_experiment, "test-3")
        self.assertIn("load in [1, 4]", domain.describe())

    def test_triangle_hull_versus_box(self) -> None:
        """A box corner outside the triangle is flagged; the centroid is not."""
        domain = ValidationDomain(
            ("a", "b"), ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), ("p", "q", "r")
        )
        centroid = domain.check({"a": 1 / 3, "b": 1 / 3}, distance_threshold=1.0)
        self.assertTrue(centroid.inside_hull)
        self.assertFalse(centroid.extrapolation)
        corner = domain.check({"a": 0.8, "b": 0.8})
        self.assertTrue(corner.inside_box)
        self.assertFalse(corner.inside_hull)
        self.assertTrue(corner.extrapolation)
        edge = domain.check({"a": 0.5, "b": 0.5}, distance_threshold=1.0)
        self.assertTrue(edge.inside_hull)

    def test_distance_flag_inside_hull(self) -> None:
        """An in-hull point far from every vertex is flagged as sparse."""
        domain = ValidationDomain(
            ("a", "b"), ((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), ("p", "q", "r")
        )
        check = domain.check({"a": 0.3, "b": 0.3}, distance_threshold=0.25)
        self.assertTrue(check.inside_hull)
        self.assertTrue(check.extrapolation)
        self.assertIn("sparse", check.note)

    def test_tetrahedron_in_three_dimensions(self) -> None:
        """The simplex test separates in-hull and in-box-only points in 3-D."""
        domain = ValidationDomain(
            ("x", "y", "z"),
            ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)),
            ("o", "x", "y", "z"),
        )
        self.assertTrue(domain.check({"x": 0.3, "y": 0.3, "z": 0.3}).inside_hull)
        outside = domain.check({"x": 0.4, "y": 0.4, "z": 0.4})
        self.assertTrue(outside.inside_box)
        self.assertFalse(outside.inside_hull)

    def test_degenerate_and_empty_domains(self) -> None:
        """Zero-range dimensions need equality; no conditions is trivial."""
        domain = ValidationDomain(
            ("a", "b"), ((0.0, 5.0), (1.0, 5.0)), ("p", "q")
        )
        self.assertTrue(domain.check({"a": 0.5, "b": 5.0}).inside_hull)
        off = domain.check({"a": 0.5, "b": 5.1})
        self.assertFalse(off.inside_hull)
        self.assertEqual(off.nearest_distance, math.inf)
        empty = ValidationDomain.from_experiments(
            [ValidationExperiment("only", [1.0], provenance="bench")]
        )
        self.assertTrue(empty.check({}).inside_hull)

    def test_invalid_domains_raise(self) -> None:
        """Mismatched names, missing keys, and empty input are errors."""
        first = ValidationExperiment("a", [1.0], {"load": 1.0}, provenance="x")
        second = ValidationExperiment("b", [1.0], {"temp": 1.0}, provenance="x")
        with self.assertRaises(ValueError):
            ValidationDomain.from_experiments([first, second])
        with self.assertRaises(ValueError):
            ValidationDomain.from_experiments([])
        domain = ValidationDomain.from_experiments([first])
        with self.assertRaises(KeyError):
            domain.check({"temp": 1.0})


class ValidationExperimentTests(unittest.TestCase):
    """Validation records reject incomplete or invalid data."""

    def test_derived_values(self) -> None:
        """D is the mean and u_D defaults to sigma_m / sqrt(n)."""
        record = ValidationExperiment(
            "e", [1.0, 2.0, 3.0, 4.0], {"t": 1}, 0.2, provenance="lab"
        )
        self.assertEqual(record.mean, 2.5)
        self.assertAlmostEqual(record.data_standard_uncertainty, 0.1, places=14)
        override = ValidationExperiment(
            "e", [1.0], measurement_uncertainty=0.2, provenance="lab",
            data_uncertainty=0.5,
        )
        self.assertEqual(override.data_standard_uncertainty, 0.5)

    def test_invalid_records_raise(self) -> None:
        """Empty data, empty text, negative uncertainty, and NaN are errors."""
        with self.assertRaises(ValueError):
            ValidationExperiment("e", [], provenance="lab")
        with self.assertRaises(ValueError):
            ValidationExperiment(" ", [1.0], provenance="lab")
        with self.assertRaises(ValueError):
            ValidationExperiment("e", [1.0], measurement_uncertainty=-1.0)
        with self.assertRaises(ValueError):
            ValidationExperiment("e", [1.0], {"load": math.inf})
        with self.assertRaises(ValueError):
            ValidationExperiment("e", [1.0], provenance="")


class AssessmentTests(unittest.TestCase):
    """End-to-end assessment with one seed and a narrow verdict."""

    def test_consistent_data_give_scoped_consistent_verdict(self) -> None:
        """Data generated by the model are consistent at a loose tolerance."""
        model = _spring_model()
        before = model.evaluations
        result = assess_validation(
            model,
            "y",
            _experiments(),
            tolerance=0.4,
            sample_size=400,
            seed=3,
            numerical_uncertainty=NumericalErrorBudget(discretization=0.01),
            prediction_conditions=[{"load": 2.5}, {"load": 8.0}],
        )
        self.assertEqual(model.evaluations - before, 4 * 400)
        self.assertTrue(result.consistent, result.summary())
        self.assertGreater(result.log_bayes_factor, 0.0)
        summary = result.summary()
        self.assertIn("consistent with the 4 validation experiment(s)", summary)
        self.assertIn("tolerance 0.4 mm", summary)
        self.assertIn("load in [1, 4]", summary)
        self.assertNotIn("validated", summary.lower())
        self.assertFalse(result.prediction_checks[0].extrapolation)
        self.assertTrue(result.prediction_checks[1].extrapolation)
        self.assertEqual(len(result.experiments), 4)
        for item in result.experiments:
            self.assertEqual(len(item.u_values), 6)
            self.assertFalse(item.asme.error_resolved)

    def test_same_seed_reproduces_and_bias_is_detected(self) -> None:
        """One seed reproduces everything; a bias makes the data inconsistent."""
        model = _spring_model()
        first = assess_validation(
            model, "y", _experiments(), tolerance=0.4, sample_size=300, seed=9
        )
        second = assess_validation(
            model, "y", _experiments(), tolerance=0.4, sample_size=300, seed=9
        )
        self.assertEqual(
            [e.area for e in first.experiments], [e.area for e in second.experiments]
        )
        self.assertEqual(first.u_pooling.u_values, second.u_pooling.u_values)
        biased = assess_validation(
            model,
            "y",
            _experiments(bias=1.0),
            tolerance=0.4,
            sample_size=300,
            seed=9,
        )
        self.assertFalse(biased.consistent)
        self.assertIn("inconsistent", biased.verdict)
        self.assertGreater(biased.u_pooling.area, 0.3)
        self.assertLess(biased.log_bayes_factor, 0.0)
        self.assertTrue(all(e.asme.error_resolved for e in biased.experiments))

    def test_predict_at_conditions_and_user_samples(self) -> None:
        """User-supplied samples go through the same assessment."""
        model = _spring_model()
        experiments = _experiments()
        samples = [
            predict_at_conditions(model, "y", e.conditions, 300, seed=index)
            for index, e in enumerate(experiments)
        ]
        self.assertAlmostEqual(
            sum(samples[1]) / 300, STIFFNESS_MEAN * 2.0, delta=0.05
        )
        result = assess_predictions(
            "y", experiments, samples, tolerance=0.4, seed=1,
            include_measurement_noise=False,
        )
        self.assertFalse(result.include_measurement_noise)
        self.assertEqual(
            result.experiments[0].observable_predictions,
            result.experiments[0].latent_predictions,
        )

    def test_invalid_assessment_inputs_raise(self) -> None:
        """Wrong types, unknown QoIs, shadowing, and bad tolerances are errors."""
        model = _spring_model()
        experiments = _experiments()
        with self.assertRaises(TypeError):
            assess_validation(
                object(),  # type: ignore[arg-type]
                "y",
                experiments,
                tolerance=1.0,
            )
        with self.assertRaises(KeyError):
            assess_validation(model, "z", experiments, tolerance=1.0)
        shadow = [ValidationExperiment("s", [1.0], {"k": 1.0}, provenance="x")]
        with self.assertRaises(ValueError):
            assess_validation(model, "y", shadow, tolerance=1.0, sample_size=10)
        with self.assertRaises(ValueError):
            assess_validation(model, "y", experiments, tolerance=0.0)
        with self.assertRaises(ValueError):
            assess_predictions("y", experiments, [[1.0, 2.0]], tolerance=1.0)
        with self.assertRaises(ValueError):
            assess_predictions(
                "y", experiments, [[1.0, 2.0]] * 4, tolerance=1.0, significance=1.0
            )


class CredibilityTests(unittest.TestCase):
    """Verification, validation, and prediction stay separate."""

    def test_record_reports_three_sections_and_gaps(self) -> None:
        """The summary keeps sections apart and lists extrapolation as a gap."""
        model = _spring_model()
        assessment = assess_validation(
            model, "y", _experiments(), tolerance=0.4, sample_size=200, seed=4
        )
        verification = VerificationEvidence(
            "y",
            NumericalErrorBudget(discretization=0.003, iterative=0.001),
            "three-grid refinement study",
            observed_order=1.98,
        )
        self.assertAlmostEqual(
            verification.standard_uncertainty, 0.004 / math.sqrt(3.0), places=14
        )
        samples = predict_at_conditions(model, "y", {"load": 8.0}, 400, seed=2)
        prediction = PredictionStatement.from_samples(
            "y",
            {"load": 8.0},
            samples,
            uncertainty_sources="input uncertainty only; excludes model-form error",
            domain=assessment.domain,
        )
        self.assertLess(prediction.lower, 16.0)
        self.assertGreater(prediction.upper, 16.0)
        record = CredibilityRecord((verification,), (assessment,), (prediction,))
        summary = record.summary()
        self.assertIn("Verification (", summary)
        self.assertIn("Validation (", summary)
        self.assertIn("Prediction (", summary)
        self.assertIn("no single credibility score", summary)
        self.assertTrue(any("extrapolation" in gap for gap in record.gaps))
        empty = CredibilityRecord()
        self.assertEqual(len(empty.gaps), 3)

    def test_invalid_records_raise(self) -> None:
        """Wrong evidence types and reversed intervals are rejected."""
        with self.assertRaises(TypeError):
            CredibilityRecord(verification=("bound",))  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            PredictionStatement("y", {}, 2.0, 1.0, 0.9, "inputs")
        with self.assertRaises(TypeError):
            VerificationEvidence("y", 0.1, "x")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
