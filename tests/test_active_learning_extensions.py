"""Tests for AK-MCS learning functions, stopping rules, surrogates, and designs."""

from __future__ import annotations

import math
import random
import unittest
import warnings
from collections.abc import Callable, Mapping, Sequence

from probstat.distributions import Normal
from probstat.modeling import (
    OSCILLATOR_INPUT_NAMES,
    OSCILLATOR_REFERENCE_PROBABILITY,
    RASTRIGIN_REFERENCE_PROBABILITY,
    JointInput,
    ProbabilisticModel,
    RandomInput,
    four_branch_model,
    oscillator,
    oscillator_model,
    rastrigin,
    rastrigin_model,
)
from probstat.reliability import (
    AKMCS_DESIGNS,
    LEARNING_FUNCTIONS,
    LEARNING_THRESHOLDS,
    GaussianProcessPopulationPosterior,
    ModelLimitState,
    PopulationPosterior,
    SurrogatePopulationPosterior,
    ak_mcs,
    bounds_satisfied,
    find_design_points,
    form_hlrf,
    learning_value,
    probability_bounds,
    select_learning_point,
    stopping_satisfied,
)
from probstat.surrogates import (
    GaussianProcessFactory,
    GaussianProcessRegularization,
    ProbabilisticSurrogate,
    SurrogateFactory,
    fit_gaussian_process,
)
from probstat.uq import QuantityOfInterest

Y = (QuantityOfInterest("y", "response"),)
SQRT2 = math.sqrt(2.0)


def make_state(
    function: Callable[[tuple[float, ...]], float], dimension: int
) -> ModelLimitState:
    """Wrap ``G(u)`` on standard normal inputs as a ``g <= 0`` limit state."""
    names = tuple(f"u{i + 1}" for i in range(dimension))

    def response(p: Mapping[str, float]) -> dict[str, float]:
        """Evaluate the limit state from the named inputs."""
        return {"y": function(tuple(p[name] for name in names))}

    inputs = JointInput.independent(
        *(RandomInput(name, Normal(0.0, 1.0)) for name in names)
    )
    return ModelLimitState(ProbabilisticModel(inputs, response, Y), "y", 0.0, "below")


def four_branch_seven(u: tuple[float, ...]) -> float:
    """Four-branch series limit state with ``k = 7``."""
    inner = 3.0 + (u[0] - u[1]) ** 2 / 10.0 - abs(u[0] + u[1]) / SQRT2
    return min(inner, 7.0 / SQRT2 - abs(u[0] - u[1]))


def four_branch_six() -> ModelLimitState:
    """Four-branch limit state with ``k = 6`` from the benchmark module."""
    return ModelLimitState(four_branch_model(6.0), "g", 0.0, "below")


def numeric_entropy(mean: float, std: float, nodes: int = 20_000) -> float:
    """Midpoint rule for ``|-int_{-2s}^{2s} f ln f|`` of ``N(mean, std^2)``."""
    width = 4.0 * std / nodes
    total = 0.0
    for i in range(nodes):
        g = -2.0 * std + (i + 0.5) * width
        density = math.exp(-0.5 * ((g - mean) / std) ** 2) / (
            math.sqrt(2.0 * math.pi) * std
        )
        total -= density * math.log(density) * width
    return abs(total)


def numeric_feasibility(mean: float, std: float, nodes: int = 20_000) -> float:
    """Midpoint rule for ``int_{-e}^{e} (e - |g|) f(g) dg`` with ``e = 2 std``."""
    epsilon = 2.0 * std
    width = 2.0 * epsilon / nodes
    total = 0.0
    for i in range(nodes):
        g = -epsilon + (i + 0.5) * width
        density = math.exp(-0.5 * ((g - mean) / std) ** 2) / (
            math.sqrt(2.0 * math.pi) * std
        )
        total += (epsilon - abs(g)) * density * width
    return total


class LearningFunctionTests(unittest.TestCase):
    """Closed forms of U, EFF, and H and the selection and stopping rules."""

    def test_closed_forms_match_their_integral_definitions(self) -> None:
        """EFF and H equal numerical integrals of their definitions."""
        for mean, std in ((0.0, 1.0), (0.3, 0.2), (-1.0, 0.5), (2.0, 3.0)):
            self.assertAlmostEqual(
                learning_value("EFF", mean, std), numeric_feasibility(mean, std), 8
            )
            self.assertAlmostEqual(
                learning_value("H", mean, std), numeric_entropy(mean, std), 8
            )
            self.assertEqual(learning_value("U", mean, std), abs(mean) / std)

    def test_zero_standard_deviation(self) -> None:
        """A certain prediction has U = inf (or 0 at mu = 0) and EFF = H = 0."""
        self.assertEqual(learning_value("U", 1.5, 0.0), math.inf)
        self.assertEqual(learning_value("U", 0.0, 0.0), 0.0)
        self.assertEqual(learning_value("EFF", 1.5, 0.0), 0.0)
        self.assertEqual(learning_value("H", -1.5, 0.0), 0.0)
        with self.assertRaises(ValueError):
            learning_value("U", 1.0, -1.0)
        with self.assertRaises(ValueError):
            learning_value("V", 1.0, 1.0)

    def test_selection_direction_ties_and_exclusion(self) -> None:
        """U picks the first minimum, EFF and H the first maximum."""
        means = [3.0, 0.1, 0.1, 2.0]
        variances = [1.0, 1.0, 1.0, 0.0]
        self.assertEqual(select_learning_point("U", means, variances), (1, 0.1))
        self.assertEqual(select_learning_point("U", means, variances, {1})[0], 2)
        index, value = select_learning_point("EFF", means, variances)
        self.assertEqual(index, 1)
        self.assertEqual(value, learning_value("EFF", 0.1, 1.0))
        self.assertEqual(select_learning_point("H", means, variances, {1})[0], 2)
        self.assertEqual(
            select_learning_point("U", means, variances, {0, 1, 2, 3}),
            (-1, math.inf),
        )
        self.assertEqual(
            select_learning_point("EFF", means, variances, {0, 1, 2, 3}), (-1, 0.0)
        )

    def test_stopping_defaults(self) -> None:
        """Default thresholds: min U >= 2, max EFF <= 1e-3, max H <= 0.5."""
        self.assertEqual(set(LEARNING_THRESHOLDS), set(LEARNING_FUNCTIONS))
        self.assertTrue(stopping_satisfied("U", 2.0))
        self.assertFalse(stopping_satisfied("U", 1.99))
        self.assertTrue(stopping_satisfied("EFF", 1e-3))
        self.assertFalse(stopping_satisfied("EFF", 2e-3))
        self.assertTrue(stopping_satisfied("H", 0.5))
        self.assertFalse(stopping_satisfied("H", 0.6))
        self.assertTrue(stopping_satisfied("H", 0.6, 1.0))
        with self.assertRaises(ValueError):
            stopping_satisfied("U", 3.0, 0.0)

    def test_probability_bounds(self) -> None:
        """Uncertain signs count as safe for P- and as failed for P+."""
        means = [-1.0, -0.1, 0.1, 1.0]
        variances = [0.01, 0.01, 0.01, 0.01]
        lower, estimate, upper = probability_bounds(means, variances)
        self.assertEqual((lower, estimate, upper), (0.25, 0.5, 0.75))
        self.assertFalse(bounds_satisfied(lower, estimate, upper, 0.1))
        self.assertTrue(bounds_satisfied(0.25, 0.25, 0.25, 0.1))
        self.assertTrue(bounds_satisfied(0.0, 0.0, 0.0, 0.1))
        self.assertFalse(bounds_satisfied(0.0, 0.0, 0.01, 0.1))
        with self.assertRaises(ValueError):
            bounds_satisfied(0.3, 0.2, 0.4, 0.1)


class DefaultBehaviourTests(unittest.TestCase):
    """The default AK-MCS path is unchanged by the generalization."""

    def test_default_run_reproduces_the_pre_generalization_numbers(self) -> None:
        """Four-branch k = 7, N = 5000, seed 1: the pinned learning record.

        The values were recorded with the AK-MCS implementation before the
        learning functions, the bound stopping rule, and the surrogate
        protocol were added.
        """
        state = make_state(four_branch_seven, 2)
        result = ak_mcs(state, 5000, target_cov=1.0, seed=1)
        self.assertEqual(result.probability.hex(), "0x1.bda5119ce075fp-9")
        self.assertEqual(result.evaluations, 29)
        self.assertEqual(result.added, 17)
        self.assertEqual(result.refits, 4)
        self.assertEqual(len(result.history), 19)
        # min U goes through the GP likelihood (exp, log, sqrt); its last
        # bits differ between the Windows and Linux C libraries.
        self.assertAlmostEqual(
            float(result.min_learning) / float.fromhex("0x1.5f88906cb068cp+1"),
            1.0,
            places=9,
        )
        self.assertEqual(result.message, "min U >= threshold and c.o.v. target met")
        explicit = ak_mcs(
            make_state(four_branch_seven, 2),
            5000,
            target_cov=1.0,
            seed=1,
            learning="U",
            learning_threshold=2.0,
            stopping="learning",
            design="lhs",
        )
        self.assertEqual(explicit.history, result.history)
        self.assertEqual(explicit.probability, result.probability)
        self.assertEqual(result.learning, "U")
        self.assertEqual(result.learning_threshold, 2.0)
        self.assertEqual(len(result.training_points), result.evaluations)
        self.assertEqual(
            result.training_values[0], state.standard(result.training_points[0])
        )

    def test_invalid_options(self) -> None:
        """Unknown learning, stopping, and design names are rejected."""
        state = make_state(four_branch_seven, 2)
        for keyword, value in (
            ("learning", "V"),
            ("stopping", "never"),
            ("design", "grid"),
            ("bounds_tolerance", 0.0),
            ("learning_threshold", -1.0),
        ):
            with self.subTest(keyword=keyword), self.assertRaises(ValueError):
                ak_mcs(state, 100, seed=1, **{keyword: value})  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            ak_mcs(state, 100, seed=1, extra_points=[(1.0, 2.0, 3.0)])
        with self.assertRaises(TypeError):
            ak_mcs(state, 100, seed=1, surrogate=3)  # type: ignore[arg-type]
        self.assertIn("lhs-maximin", AKMCS_DESIGNS)


class LearningAndStoppingRunTests(unittest.TestCase):
    """EFF, H, and bound stopping on the four-branch function (k = 6)."""

    def test_eff_matches_crude_monte_carlo(self) -> None:
        """EFF stops at max EFF <= 0.001 with no misclassified point."""
        result = ak_mcs(
            four_branch_six(),
            2000,
            learning="EFF",
            target_cov=1.0,
            compare_with_monte_carlo=True,
            seed=1,
        )
        self.assertTrue(result.converged)
        self.assertLessEqual(result.min_learning, 1e-3)
        self.assertEqual(result.learning_threshold, 1e-3)
        self.assertEqual(result.misclassified, 0)
        self.assertEqual(result.probability, result.reference_probability)
        self.assertTrue(result.message.startswith("max EFF <= threshold"))

    def test_entropy_learning_misclassifies_almost_nothing(self) -> None:
        """H stops at max H <= 0.5 with at most 0.1 % of the signs wrong."""
        result = ak_mcs(
            four_branch_six(),
            2000,
            learning="H",
            target_cov=1.0,
            compare_with_monte_carlo=True,
            seed=2,
        )
        self.assertTrue(result.converged)
        self.assertLessEqual(result.min_learning, 0.5)
        self.assertLessEqual(result.misclassified, 2)
        self.assertGreater(result.probability, 0.0)

    def test_bound_stopping(self) -> None:
        """Bound stopping ends with (P+ - P-) / P0 <= 10 %."""
        result = ak_mcs(
            four_branch_six(),
            2000,
            stopping="bounds",
            target_cov=1.0,
            compare_with_monte_carlo=True,
            seed=1,
        )
        self.assertTrue(result.converged)
        self.assertEqual(result.stopping, "bounds")
        width = result.upper_probability - result.lower_probability
        self.assertLessEqual(width, 0.1 * result.probability)
        self.assertLessEqual(result.lower_probability, result.probability)
        self.assertEqual(result.misclassified, 0)
        last = result.history[-1]
        self.assertLessEqual(last.lower_probability, last.probability)
        self.assertLessEqual(last.probability, last.upper_probability)


class SurrogateProtocolTests(unittest.TestCase):
    """Generic surrogate path and the population posteriors."""

    def test_generic_gaussian_process_reproduces_the_fast_path(self) -> None:
        """A GP factory gives the fast path's estimate on the same population."""
        fast = ak_mcs(four_branch_six(), 2000, target_cov=1.0, seed=1)
        factory = GaussianProcessFactory(
            regularization=GaussianProcessRegularization(), seed=1
        )
        self.assertIsInstance(factory, SurrogateFactory)
        generic = ak_mcs(
            four_branch_six(),
            2000,
            target_cov=1.0,
            surrogate=factory,
            compare_with_monte_carlo=True,
            seed=1,
        )
        self.assertTrue(generic.converged)
        self.assertEqual(generic.probability, fast.probability)
        self.assertEqual(generic.misclassified, 0)
        self.assertEqual(generic.refits, generic.added + 1)
        self.assertIsInstance(generic.surrogate, ProbabilisticSurrogate)

    def test_custom_surrogate_through_the_protocol(self) -> None:
        """Any object with predict and predict_with_variance is accepted."""

        def truth(u: Sequence[float]) -> float:
            """Plane with beta = 2."""
            return 2.0 - (u[0] + u[1]) / SQRT2

        class Exact:
            """Exact mean; variance = squared distance to the training set."""

            def __init__(self, points: Sequence[Sequence[float]]) -> None:
                self.points = [tuple(p) for p in points]

            def predict(self, x: Sequence[float]) -> float:
                return truth(x)

            def predict_with_variance(self, x: Sequence[float]) -> tuple[float, float]:
                gap = min(
                    sum((a - b) ** 2 for a, b in zip(x, p, strict=True))
                    for p in self.points
                )
                return truth(x), gap

        calls: list[int] = []

        def factory(
            points: Sequence[Sequence[float]],
            values: Sequence[float],
            previous: ProbabilisticSurrogate | None,
        ) -> ProbabilisticSurrogate:
            """Record the training-set size and return the exact surrogate."""
            calls.append(len(points))
            return Exact(points)

        result = ak_mcs(
            make_state(truth, 2),
            500,
            target_cov=1.0,
            max_added=5,
            surrogate=factory,
            compare_with_monte_carlo=True,
            seed=4,
        )
        self.assertEqual(result.misclassified, 0)
        self.assertEqual(calls[0], 12)
        self.assertEqual(calls, list(range(12, 12 + result.added + 1)))

    def test_population_posteriors_follow_the_protocol(self) -> None:
        """Both posteriors keep training data when the population is replaced."""
        generator = random.Random(7)
        train = [(generator.uniform(-3, 3), generator.uniform(-3, 3)) for _ in range(8)]
        values = [four_branch_seven(p) for p in train]
        first = [(generator.gauss(0, 1), generator.gauss(0, 1)) for _ in range(30)]
        second = [(generator.gauss(0, 1), generator.gauss(0, 1)) for _ in range(20)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp = fit_gaussian_process(
                train,
                values,
                seed=1,
                regularization=GaussianProcessRegularization(),
            )
        posterior = GaussianProcessPopulationPosterior(gp, first)
        self.assertIsInstance(posterior, PopulationPosterior)
        extra = (0.5, -0.5)
        posterior.add_training(extra, four_branch_seven(extra))
        posterior.replace_population(second)
        fresh = GaussianProcessPopulationPosterior(gp, [])
        fresh.add_training(extra, four_branch_seven(extra))
        fresh.add_population(second)
        self.assertEqual(posterior.means, fresh.means)
        self.assertEqual(posterior.variances, fresh.variances)
        self.assertEqual(len(posterior.training_points), 9)
        batch = GaussianProcessPopulationPosterior(gp, second)
        for point, mean, variance in zip(
            second, batch.means, batch.variances, strict=True
        ):
            expected_mean, expected_variance = gp.predict_with_variance(point)
            self.assertAlmostEqual(mean, expected_mean, delta=1e-9)
            self.assertAlmostEqual(max(variance, 0.0), expected_variance, delta=1e-9)

        factory = GaussianProcessFactory(seed=3)
        generic = SurrogatePopulationPosterior(factory, train, values, first)
        self.assertIsInstance(generic, PopulationPosterior)
        generic.replace_population(second)
        self.assertEqual(generic.population, second)
        self.assertEqual(
            generic.means,
            [generic.surrogate.predict_with_variance(p)[0] for p in second],
        )
        self.assertEqual(generic.fits, 1)


class InitialDesignTests(unittest.TestCase):
    """Maximin Latin hypercube and extra seed points."""

    def test_maximin_design_runs_and_is_latin(self) -> None:
        """The first 12 training points occupy distinct strata per axis."""
        result = ak_mcs(
            four_branch_six(),
            2000,
            design="lhs-maximin",
            target_cov=1.0,
            compare_with_monte_carlo=True,
            seed=1,
        )
        self.assertTrue(result.converged)
        self.assertEqual(result.misclassified, 0)
        design = result.training_points[: result.initial_size]
        for axis in range(2):
            strata = {math.floor((p[axis] + 5.0) / 10.0 * 12) for p in design}
            self.assertEqual(len(strata), 12)

    def test_design_points_as_extra_points(self) -> None:
        """FORM design points are evaluated after the design and counted."""
        state = four_branch_six()
        # G has a kink at the origin, where the finite-difference gradient
        # vanishes; the first FORM search starts next to it.
        first = form_hlrf(state, start=(0.3, 0.1))
        found = find_design_points(state, max_points=4, first=first)
        seeds = [form.design_point_u for form in found.forms]
        before = state.model.evaluations
        result = ak_mcs(
            state, 1000, target_cov=1.0, max_added=0, extra_points=seeds, seed=2
        )
        self.assertEqual(result.initial_size, 12 + len(seeds))
        self.assertEqual(result.evaluations, result.initial_size)
        self.assertEqual(state.model.evaluations - before, result.evaluations)
        self.assertEqual(result.training_points[12:], tuple(tuple(s) for s in seeds))


class BenchmarkModelTests(unittest.TestCase):
    """Oscillator and modified Rastrigin limit states and their references."""

    def test_oscillator_formula_and_model(self) -> None:
        """The oscillator ``g = 3 r - |z_max|`` at the mean inputs."""
        omega = math.sqrt(1.1)
        expected = 1.5 - abs(2.0 / 1.1 * math.sin(omega / 2.0))
        self.assertAlmostEqual(oscillator(1.0, 1.0, 0.1, 0.5, 1.0, 1.0), expected, 15)
        model = oscillator_model()
        self.assertEqual(model.inputs.names, OSCILLATOR_INPUT_NAMES)
        nominal = dict(
            zip(OSCILLATOR_INPUT_NAMES, (1.0, 1.0, 0.1, 0.5, 1.0, 1.0), strict=True)
        )
        self.assertAlmostEqual(model.evaluate(nominal)["g"], expected, 15)
        with self.assertRaises(ValueError):
            oscillator_model(force_std=0.0)
        with self.assertRaises(ValueError):
            oscillator(0.0, 1.0, 0.1, 0.5, 1.0, 1.0)

    def test_rastrigin_formula_and_model(self) -> None:
        """The Rastrigin ``g(0, 0) = 20`` and ``g(0.5, 0.5) = 10 - 2 (0.25 + 5)``."""
        self.assertEqual(rastrigin(0.0, 0.0), 20.0)
        self.assertAlmostEqual(rastrigin(0.5, 0.5), 10.0 - 2.0 * 5.25, 14)
        self.assertEqual(rastrigin_model().evaluate({"x1": 0.0, "x2": 0.0})["g"], 20.0)

    def test_references_agree_with_literature_and_a_small_sample(self) -> None:
        """References agree with the literature and a 40 000-draw check."""
        self.assertLess(abs(OSCILLATOR_REFERENCE_PROBABILITY - 2.86e-2), 3 * 2.4e-5)
        self.assertLess(abs(RASTRIGIN_REFERENCE_PROBABILITY - 7.30e-2), 3 * 2.6e-5)
        generator = random.Random(11)
        draws = 40_000
        failures = 0
        for _ in range(draws):
            m, c1, c2, r, f1, t1 = (
                generator.gauss(1.0, 0.05),
                generator.gauss(1.0, 0.1),
                generator.gauss(0.1, 0.01),
                generator.gauss(0.5, 0.05),
                generator.gauss(1.0, 0.2),
                generator.gauss(1.0, 0.2),
            )
            failures += oscillator(m, c1, c2, r, f1, t1) <= 0.0
        p = OSCILLATOR_REFERENCE_PROBABILITY
        self.assertLess(abs(failures / draws - p), 4.0 * math.sqrt(p * (1 - p) / draws))
        failures = sum(
            rastrigin(generator.gauss(0.0, 1.0), generator.gauss(0.0, 1.0)) <= 0.0
            for _ in range(draws)
        )
        p = RASTRIGIN_REFERENCE_PROBABILITY
        self.assertLess(abs(failures / draws - p), 4.0 * math.sqrt(p * (1 - p) / draws))


if __name__ == "__main__":
    unittest.main()
