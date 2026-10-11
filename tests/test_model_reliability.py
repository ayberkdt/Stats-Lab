"""Tests for model-based FORM, SORM, rare-event, system, and inverse reliability."""

from __future__ import annotations

import math
import unittest

from probstat.distributions import LogNormal, Normal
from probstat.modeling import (
    JointInput,
    ProbabilisticModel,
    RandomInput,
    gauss_hermite,
    standard_normal_cdf,
)
from probstat.reliability import (
    ModelLimitState,
    bivariate_normal_cdf,
    compare_reliability_methods,
    design_for_target_beta,
    design_point_importance_sampling,
    form_hlrf,
    gauss_legendre,
    generalized_reliability_index,
    inverse_form,
    model_monte_carlo,
    principal_curvatures,
    second_order_probabilities,
    series_system_bounds,
    sorm,
    subset_simulation,
    system_failure_probability,
)
from probstat.uq import QuantityOfInterest

Y = (QuantityOfInterest("y", "response"),)


def standard_pair() -> JointInput:
    """Return two independent standard normal inputs named u1 and u2."""
    return JointInput.independent(
        RandomInput("u1", Normal(0.0, 1.0)), RandomInput("u2", Normal(0.0, 1.0))
    )


def lognormal_margin(
    rho: float = 0.4, mu_r: float = 1.2, mu_s: float = 0.3
) -> tuple[ModelLimitState, float]:
    """Return the correlated lognormal R - S limit state and its exact index.

    ``ln R`` and ``ln S`` are jointly normal with latent correlation ``rho``,
    so ``P(R <= S) = Phi(-(mu_R - mu_S) / sqrt(s_R^2 + s_S^2 - 2 rho s_R s_S))``.
    """
    s_r, s_s = 0.15, 0.25
    inputs = JointInput(
        (
            RandomInput("r", LogNormal(mu_r, s_r)),
            RandomInput("s", LogNormal(mu_s, s_s)),
        ),
        ((1.0, rho), (rho, 1.0)),
    )
    model = ProbabilisticModel(inputs, lambda p: {"y": p["r"] - p["s"]}, Y)
    beta = (mu_r - mu_s) / math.sqrt(s_r**2 + s_s**2 - 2.0 * rho * s_r * s_s)
    return ModelLimitState(model, "y", 0.0, "below"), beta


def parabolic(kappa: float, beta: float = 3.0) -> ModelLimitState:
    """Return ``G(u) = beta - u2 + kappa u1^2 / 2`` with curvature ``kappa``."""
    model = ProbabilisticModel(
        standard_pair(),
        lambda p: {"y": beta - p["u2"] + 0.5 * kappa * p["u1"] ** 2},
        Y,
    )
    return ModelLimitState(model, "y", 0.0, "below")


def parabolic_exact(kappa: float, beta: float = 3.0) -> float:
    """Return the exact parabolic failure probability by Gauss-Hermite quadrature.

    Failure is ``u2 >= beta + kappa u1^2 / 2``, so
    ``P_f = E[Phi(-(beta + kappa U1^2 / 2))]``.
    """
    nodes, weights = gauss_hermite(30)
    return math.fsum(
        w * standard_normal_cdf(-(beta + 0.5 * kappa * z * z))
        for z, w in zip(nodes, weights, strict=True)
    )


def linear_standard(beta: float) -> ModelLimitState:
    """Return ``G(u) = beta - (u1 + u2)/sqrt(2)`` with exact ``P_f = Phi(-beta)``."""
    model = ProbabilisticModel(
        standard_pair(),
        lambda p: {"y": beta - (p["u1"] + p["u2"]) / math.sqrt(2.0)},
        Y,
    )
    return ModelLimitState(model, "y", 0.0, "below")


class ModelLimitStateTests(unittest.TestCase):
    """Check the limit-state construction and its failure convention."""

    def test_margin_sides_and_spaces_agree(self) -> None:
        """Above/below margins follow g <= 0 and u-space equals mapped x-space."""
        inputs = JointInput.independent(RandomInput("x", Normal(10.0, 2.0)))
        model = ProbabilisticModel(inputs, lambda p: {"y": 2.0 * p["x"]}, Y)
        above = ModelLimitState(model, "y", 24.0, "above")
        below = ModelLimitState(model, "y", 16.0, "below")
        self.assertAlmostEqual(above.physical((10.0,)), 4.0)
        self.assertAlmostEqual(below.physical((10.0,)), 4.0)
        self.assertTrue(above.failed((12.0,)))
        self.assertTrue(below.failed((8.0,)))
        self.assertAlmostEqual(above.standard((1.0,)), above.physical((12.0,)))
        self.assertAlmostEqual(above.output_from_margin(above.margin(7.0)), 7.0)
        self.assertEqual(above.label, "y >= 24")
        self.assertEqual(above.with_threshold(30.0).threshold, 30.0)
        self.assertEqual(above.dimension, 1)
        self.assertEqual(above.input_names, ("x",))

    def test_invalid_construction_is_rejected(self) -> None:
        """Wrong model type, unknown quantity, side, or threshold raise errors."""
        model = linear_standard(3.0).model
        with self.assertRaises(TypeError):
            ModelLimitState(object(), "y", 0.0)  # type: ignore[arg-type]
        with self.assertRaises(KeyError):
            ModelLimitState(model, "missing", 0.0)
        with self.assertRaises(ValueError):
            ModelLimitState(model, "y", 0.0, "sideways")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            ModelLimitState(model, "y", math.nan)


class FORMTests(unittest.TestCase):
    """Check HL-RF FORM on exact linear, non-normal, and correlated cases."""

    def setUp(self) -> None:
        """Build a correlated Gaussian sum with an exact reliability index."""
        inputs = JointInput(
            (RandomInput("x1", Normal(10.0, 2.0)), RandomInput("x2", Normal(5.0, 1.0))),
            ((1.0, 0.5), (0.5, 1.0)),
        )
        model = ProbabilisticModel(inputs, lambda p: {"y": p["x1"] + p["x2"]}, Y)
        self.state = ModelLimitState(model, "y", 20.0, "above")
        self.beta = 5.0 / math.sqrt(4.0 + 1.0 + 2.0 * 0.5 * 2.0 * 1.0)

    def test_linear_correlated_gaussian_is_exact(self) -> None:
        """FORM recovers the exact index, design point, and importance factors."""
        result = form_hlrf(self.state)
        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.beta, self.beta, places=8)
        self.assertAlmostEqual(
            result.probability, standard_normal_cdf(-result.beta), places=15
        )
        self.assertAlmostEqual(
            result.probability, standard_normal_cdf(-self.beta), places=9
        )
        self.assertAlmostEqual(sum(result.design_point_x), 20.0, places=6)
        self.assertAlmostEqual(sum(result.importance_factors.values()), 1.0)
        self.assertAlmostEqual(math.hypot(*result.gamma.values()), 1.0)
        self.assertNotAlmostEqual(result.gamma["x2"], result.alpha[1], places=3)
        self.assertEqual(result.evaluations, self.state.model.evaluations)
        self.assertGreater(result.evaluations, 0)

    def test_plain_hlrf_and_analytic_gradient_agree(self) -> None:
        """Plain HL-RF and an analytic gradient reach the same design point."""
        plain = form_hlrf(self.state, line_search=False)
        constant = (-2.5, -math.sqrt(0.75))
        analytic = form_hlrf(self.state, gradient=lambda u: constant)
        self.assertAlmostEqual(plain.beta, self.beta, places=8)
        self.assertAlmostEqual(analytic.beta, self.beta, places=10)
        self.assertGreater(analytic.gradient_evaluations, 0)
        self.assertLess(analytic.evaluations, plain.evaluations)

    def test_lognormal_correlated_margin_is_exact(self) -> None:
        """Non-normal correlated inputs through Nataf give the closed-form index."""
        state, beta = lognormal_margin()
        result = form_hlrf(state)
        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.beta, beta, places=8)
        r, s = result.design_point_x
        self.assertAlmostEqual(r, s, places=7)

    def test_failed_origin_gives_negative_index(self) -> None:
        """A failing median gives a negative index and probability above one half."""
        model = ProbabilisticModel(standard_pair(), lambda p: {"y": p["u1"]}, Y)
        state = ModelLimitState(model, "y", -1.0, "above")
        result = form_hlrf(state)
        self.assertAlmostEqual(result.beta, -1.0, places=8)
        self.assertAlmostEqual(result.probability, standard_normal_cdf(1.0), places=10)

    def test_non_convergence_is_reported_and_bad_inputs_raise(self) -> None:
        """Iteration limits return a flag; zero gradients and bad options raise."""
        limited = form_hlrf(parabolic(0.2), start=(1.0, 0.0), max_iterations=1)
        self.assertFalse(limited.converged)
        self.assertIn("maximum iterations", limited.message)
        flat = ProbabilisticModel(standard_pair(), lambda p: {"y": 1.0}, Y)
        with self.assertRaises(ValueError):
            form_hlrf(ModelLimitState(flat, "y", 0.0, "below"))
        with self.assertRaises(ValueError):
            form_hlrf(self.state, start=(0.0,))
        with self.assertRaises(ValueError):
            form_hlrf(self.state, tolerance=0.0)
        with self.assertRaises(TypeError):
            form_hlrf("state")  # type: ignore[arg-type]


class SORMTests(unittest.TestCase):
    """Check curvature extraction and second-order probability formulas."""

    def test_sorm_improves_form_on_curved_limit_states(self) -> None:
        """For both curvature signs SORM is closer to the exact probability."""
        for kappa in (0.2, -0.2):
            with self.subTest(kappa=kappa):
                result = sorm(parabolic(kappa))
                exact = parabolic_exact(kappa)
                self.assertAlmostEqual(result.form.beta, 3.0, places=6)
                self.assertEqual(len(result.curvatures), 1)
                self.assertAlmostEqual(result.curvatures[0], kappa, places=5)
                form_error = abs(result.form.probability - exact)
                for estimate in (
                    result.breitung_probability,
                    result.hohenbichler_rackwitz_probability,
                ):
                    self.assertLess(abs(estimate - exact), 0.3 * form_error)
                self.assertAlmostEqual(
                    result.breitung_probability,
                    standard_normal_cdf(-3.0) / math.sqrt(1.0 + 3.0 * kappa),
                    places=9,
                )
                self.assertEqual(
                    result.evaluations,
                    result.form.evaluations + result.hessian_evaluations,
                )
                self.assertAlmostEqual(
                    standard_normal_cdf(-result.breitung_beta),
                    result.breitung_probability,
                )

    def test_principal_curvatures_in_three_dimensions(self) -> None:
        """A tilted quadratic surface returns the analytic curvature spectrum."""
        gradient = (0.0, 0.0, -2.0)
        hessian = ((0.6, 0.0, 0.0), (0.0, -0.2, 0.0), (0.0, 0.0, 5.0))
        values = principal_curvatures(gradient, hessian)
        self.assertAlmostEqual(values[0], -0.1)
        self.assertAlmostEqual(values[1], 0.3)
        self.assertEqual(principal_curvatures((1.0,), ((3.0,),)), ())
        with self.assertRaises(ValueError):
            principal_curvatures((0.0, 0.0), ((1.0, 0.0), (0.0, 1.0)))

    def test_invalid_curvatures_and_indices_are_rejected(self) -> None:
        """1 + beta*kappa <= 0, beta <= 0, and unconverged FORM raise errors."""
        with self.assertRaises(ValueError):
            second_order_probabilities(3.0, (-0.4,))
        with self.assertRaises(ValueError):
            second_order_probabilities(-1.0, (0.1,))
        state = parabolic(0.2)
        unconverged = form_hlrf(state, start=(1.0, 0.0), max_iterations=1)
        with self.assertRaises(ValueError):
            sorm(state, form_result=unconverged)


class SamplingTests(unittest.TestCase):
    """Check crude Monte Carlo, design-point IS, and subset simulation."""

    def test_crude_monte_carlo_matches_exact_probability(self) -> None:
        """Crude Monte Carlo agrees with Phi(-beta) and charges one call per draw."""
        state = linear_standard(1.5)
        result = model_monte_carlo(state, 4000, seed=21)
        exact = standard_normal_cdf(-1.5)
        self.assertLess(abs(result.probability - exact), 4.0 * result.standard_error)
        self.assertEqual(result.evaluations, 4000)
        repeat = model_monte_carlo(state, 4000, seed=21)
        self.assertEqual(repeat.probability, result.probability)

    def test_design_point_importance_sampling_is_accurate(self) -> None:
        """Shifted-normal IS reaches a few-percent c.o.v. on a 6e-5 probability."""
        state, beta = lognormal_margin()
        exact = standard_normal_cdf(-beta)
        result = design_point_importance_sampling(state, 2000, seed=5)
        self.assertLess(abs(result.probability - exact), 4.0 * result.standard_error)
        self.assertLess(result.coefficient_of_variation, 0.1)
        self.assertEqual(result.evaluations, 2000)
        self.assertGreater(result.form_evaluations, 0)
        self.assertGreater(result.failures, 500)
        self.assertGreater(result.effective_sample_size, 1.0)
        with self.assertRaises(ValueError):
            design_point_importance_sampling(state, 10, center=(1.0,))

    def test_subset_simulation_estimates_small_probability(self) -> None:
        """Subset simulation reaches Phi(-4) ~ 3.2e-5 with about 4000 calls."""
        state = linear_standard(4.0)
        exact = standard_normal_cdf(-4.0)
        result = subset_simulation(state, seed=1)
        self.assertTrue(result.converged)
        self.assertLess(abs(result.probability / exact - 1.0), 0.5)
        self.assertLess(
            abs(math.log(result.probability / exact)),
            3.0 * result.coefficient_of_variation_upper,
        )
        self.assertEqual(result.thresholds[-1], 0.0)
        self.assertTrue(
            all(
                a > b
                for a, b in zip(result.thresholds, result.thresholds[1:], strict=False)
            )
        )
        self.assertTrue(all(p == 0.1 for p in result.conditional_probabilities[:-1]))
        self.assertLess(
            result.coefficient_of_variation, result.coefficient_of_variation_upper
        )
        self.assertEqual(len(result.acceptance_rates), result.levels - 1)
        self.assertLess(result.evaluations, 1000 * result.levels)
        self.assertAlmostEqual(standard_normal_cdf(-result.beta), result.probability)
        again = subset_simulation(state, seed=1)
        self.assertEqual(again.probability, result.probability)

    def test_subset_simulation_rejects_inconsistent_options(self) -> None:
        """Level probability and sample counts must define whole chains."""
        state = linear_standard(2.0)
        with self.assertRaises(ValueError):
            subset_simulation(state, level_probability=0.7)
        with self.assertRaises(ValueError):
            subset_simulation(state, samples_per_level=105, level_probability=0.1)
        with self.assertRaises(ValueError):
            subset_simulation(state, samples_per_level=100, level_probability=0.3)
        with self.assertRaises(ValueError):
            subset_simulation(state, proposal_spread=0.0)

    def test_generalized_index_edges(self) -> None:
        """Probabilities 0, 1/2, and 1 map to inf, 0, and -inf."""
        self.assertEqual(generalized_reliability_index(0.0), math.inf)
        self.assertEqual(generalized_reliability_index(1.0), -math.inf)
        self.assertAlmostEqual(generalized_reliability_index(0.5), 0.0)
        with self.assertRaises(ValueError):
            generalized_reliability_index(1.5)


def bivariate_reference(h: float, k: float, rho: float) -> float:
    """Return Phi_2 by composite Gauss-Legendre integration of a 1-D integral.

    ``Phi_2(h, k; rho) = int_{-inf}^{h} phi(x) Phi((k - rho x)/sqrt(1-rho^2)) dx``.
    """
    lower, panels = -12.0, 200
    nodes, weights = gauss_legendre(24)
    width = (h - lower) / panels
    scale = math.sqrt((1.0 - rho) * (1.0 + rho))
    total = 0.0
    for panel in range(panels):
        start = lower + panel * width
        for node, weight in zip(nodes, weights, strict=True):
            x = start + 0.5 * width * (1.0 + node)
            density = math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
            total += (
                0.5
                * width
                * weight
                * density
                * standard_normal_cdf((k - rho * x) / scale)
            )
    return total


class BivariateNormalTests(unittest.TestCase):
    """Check the bivariate normal CDF and Gauss-Legendre rule."""

    def test_orthant_identity(self) -> None:
        """Phi_2(0, 0; rho) = 1/4 + asin(rho)/(2 pi) for every regime of rho."""
        for rho in (-1.0, -0.99, -0.93, -0.5, 0.0, 0.3, 0.8, 0.95, 0.999, 1.0):
            with self.subTest(rho=rho):
                self.assertAlmostEqual(
                    bivariate_normal_cdf(0.0, 0.0, rho),
                    0.25 + math.asin(rho) / (2.0 * math.pi),
                    places=14,
                )

    def test_matches_numerical_integration(self) -> None:
        """General limits agree with an independent quadrature to 1e-12."""
        for rho in (-0.97, -0.6, 0.2, 0.7, 0.9, 0.95, 0.995):
            for h, k in ((-2.0, 1.0), (0.5, 0.3), (1.8, -0.7), (3.0, 2.5)):
                with self.subTest(rho=rho, h=h, k=k):
                    self.assertAlmostEqual(
                        bivariate_normal_cdf(h, k, rho),
                        bivariate_reference(h, k, rho),
                        places=12,
                    )

    def test_limits_symmetry_and_errors(self) -> None:
        """Degenerate correlations, infinite limits, and symmetry are exact."""
        phi = standard_normal_cdf
        self.assertAlmostEqual(bivariate_normal_cdf(1.0, 0.5, 1.0), phi(0.5))
        self.assertAlmostEqual(
            bivariate_normal_cdf(1.0, 0.5, -1.0), phi(1.0) + phi(0.5) - 1.0
        )
        self.assertEqual(bivariate_normal_cdf(-0.5, -0.5, -1.0), 0.0)
        self.assertEqual(bivariate_normal_cdf(-math.inf, 1.0, 0.3), 0.0)
        self.assertAlmostEqual(bivariate_normal_cdf(math.inf, 1.0, 0.3), phi(1.0))
        self.assertAlmostEqual(
            bivariate_normal_cdf(0.4, -1.1, 0.6), bivariate_normal_cdf(-1.1, 0.4, 0.6)
        )
        self.assertAlmostEqual(
            bivariate_normal_cdf(0.7, 1.2, 0.0), phi(0.7) * phi(1.2), places=15
        )
        with self.assertRaises(ValueError):
            bivariate_normal_cdf(0.0, 0.0, 1.5)

    def test_gauss_legendre_is_exact_for_polynomials(self) -> None:
        """An n-point rule integrates x^(2n-2) and x^(2n-1) exactly."""
        nodes, weights = gauss_legendre(7)
        self.assertAlmostEqual(math.fsum(weights), 2.0, places=14)
        self.assertAlmostEqual(
            math.fsum(w * x**12 for x, w in zip(nodes, weights, strict=True)),
            2.0 / 13.0,
            places=14,
        )
        self.assertAlmostEqual(
            math.fsum(w * x**13 for x, w in zip(nodes, weights, strict=True)), 0.0
        )
        with self.assertRaises(ValueError):
            gauss_legendre(0)


def three_components(betas: tuple[float, float, float]) -> list[ModelLimitState]:
    """Return three positively correlated linear components of one model."""
    directions = ((1.0, 0.0), (0.6, 0.8), (0.8, -0.6))
    names = ("g1", "g2", "g3")

    def model(p: dict[str, float]) -> dict[str, float]:
        """Evaluate three linear margins in standard-normal coordinates."""
        return {
            name: beta - a * p["u1"] - b * p["u2"]
            for name, beta, (a, b) in zip(names, betas, directions, strict=True)
        }

    shared = ProbabilisticModel(
        standard_pair(), model, tuple(QuantityOfInterest(n, n) for n in names)
    )
    return [ModelLimitState(shared, name, 0.0, "below") for name in names]


class SystemReliabilityTests(unittest.TestCase):
    """Check series and parallel systems and first-order series bounds."""

    def test_series_bounds_bracket_sampling_estimates(self) -> None:
        """Uni-modal and Ditlevsen bounds bracket MC and multi-point IS."""
        components = three_components((2.5, 2.7, 2.6))
        forms = [form_hlrf(component) for component in components]
        bounds = series_system_bounds(forms)
        self.assertTrue(bounds.positively_correlated)
        self.assertAlmostEqual(bounds.correlations[0][1], 0.6, places=6)
        self.assertLessEqual(bounds.unimodal_lower, bounds.ditlevsen_lower)
        self.assertLessEqual(bounds.ditlevsen_lower, bounds.ditlevsen_upper)
        self.assertLessEqual(bounds.ditlevsen_upper, bounds.unimodal_upper)
        weighted = system_failure_probability(
            components, method="is", samples=2000, seed=3, forms=forms
        )
        crude = system_failure_probability(components, samples=6000, seed=4)
        for estimate in (weighted, crude):
            with self.subTest(method=estimate.method):
                margin = 3.0 * estimate.standard_error
                self.assertGreaterEqual(
                    estimate.probability + margin, bounds.ditlevsen_lower
                )
                self.assertLessEqual(
                    estimate.probability - margin, bounds.ditlevsen_upper
                )
        self.assertEqual(crude.evaluations, 6000)
        self.assertEqual(weighted.evaluations, 2000)
        self.assertLess(weighted.coefficient_of_variation, 0.1)

    def test_two_component_ditlevsen_bounds_are_exact(self) -> None:
        """For two components both bounds equal P1 + P2 - P12."""
        components = three_components((2.5, 2.7, 2.6))[:2]
        forms = [form_hlrf(component) for component in components]
        bounds = series_system_bounds(forms)
        p1, p2 = standard_normal_cdf(-2.5), standard_normal_cdf(-2.7)
        exact = p1 + p2 - bivariate_normal_cdf(-2.5, -2.7, 0.6)
        self.assertAlmostEqual(bounds.ditlevsen_lower, exact, places=9)
        self.assertAlmostEqual(bounds.ditlevsen_upper, exact, places=9)

    def test_parallel_system_matches_bivariate_probability(self) -> None:
        """Crude MC of a two-component parallel system matches Phi_2."""
        components = three_components((1.5, 1.5, 9.0))[:2]
        result = system_failure_probability(
            components, kind="parallel", samples=6000, seed=8
        )
        exact = bivariate_normal_cdf(-1.5, -1.5, 0.6)
        self.assertLess(abs(result.probability - exact), 4.0 * result.standard_error)

    def test_invalid_systems_are_rejected(self) -> None:
        """Different joint inputs, kinds, methods, or empty systems raise."""
        components = three_components((2.0, 2.0, 2.0))
        other = linear_standard(2.0)
        other = ModelLimitState(
            ProbabilisticModel(
                JointInput.independent(
                    RandomInput("u1", Normal(0.0, 2.0)),
                    RandomInput("u2", Normal(0.0, 1.0)),
                ),
                lambda p: {"y": p["u1"]},
                Y,
            ),
            "y",
            1.0,
        )
        with self.assertRaises(ValueError):
            system_failure_probability([components[0], other], samples=10)
        with self.assertRaises(ValueError):
            system_failure_probability(components, kind="mixed", samples=10)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            system_failure_probability(components, method="qmc", samples=10)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            system_failure_probability([], samples=10)
        with self.assertRaises(ValueError):
            series_system_bounds([])


class InverseReliabilityTests(unittest.TestCase):
    """Check inverse FORM thresholds and target-reliability design."""

    def test_inverse_form_round_trip(self) -> None:
        """The returned threshold reproduces the target index under FORM."""
        state, _ = lognormal_margin()
        result = inverse_form(state, 3.0)
        self.assertTrue(result.converged)
        check = form_hlrf(state.with_threshold(result.threshold))
        self.assertAlmostEqual(check.beta, 3.0, places=6)
        self.assertAlmostEqual(math.hypot(*result.design_point_u), 3.0, places=10)

    def test_inverse_form_on_upper_tail_quantity(self) -> None:
        """For a Gaussian sum the threshold is mean + beta * standard deviation."""
        inputs = JointInput(
            (RandomInput("x1", Normal(10.0, 2.0)), RandomInput("x2", Normal(5.0, 1.0))),
            ((1.0, 0.5), (0.5, 1.0)),
        )
        model = ProbabilisticModel(inputs, lambda p: {"y": p["x1"] + p["x2"]}, Y)
        state = ModelLimitState(model, "y", 0.0, "above")
        result = inverse_form(state, 2.5)
        self.assertAlmostEqual(result.threshold, 15.0 + 2.5 * math.sqrt(7.0), places=6)
        with self.assertRaises(ValueError):
            inverse_form(state, 0.0)

    def test_design_parameter_reaches_target_index(self) -> None:
        """Brent on beta(c) finds the capacity factor with the exact index."""
        inputs = lognormal_margin()[0].model.inputs
        model = ProbabilisticModel(
            inputs, lambda p: {"y": p["c"] * p["r"] - p["s"]}, Y, fixed={"c": 1.0}
        )
        state = ModelLimitState(model, "y", 0.0, "below")
        result = design_for_target_beta(state, "c", 3.0, 0.7, 1.3)
        sigma = math.sqrt(0.15**2 + 0.25**2 - 2.0 * 0.4 * 0.15 * 0.25)
        exact = math.exp(3.0 * sigma - 0.9)
        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.value, exact, places=5)
        self.assertAlmostEqual(result.achieved_beta, 3.0, places=5)
        self.assertGreater(result.evaluations, result.form.evaluations)
        with self.assertRaises(KeyError):
            design_for_target_beta(state, "d", 3.0, 0.7, 1.3)
        with self.assertRaises(ValueError):
            design_for_target_beta(state, "c", 3.0, 1.1, 1.3)


class ComparisonTests(unittest.TestCase):
    """Check the reliability-method comparison table."""

    def test_methods_agree_on_lognormal_margin(self) -> None:
        """All estimators land near the exact 7e-3 probability."""
        state, beta = lognormal_margin(rho=0.0, mu_r=0.9)
        exact = standard_normal_cdf(-beta)
        comparison = compare_reliability_methods(
            state,
            monte_carlo_samples=3000,
            importance_samples=1000,
            subset_samples=500,
            seed=12,
        )
        methods = [row.method for row in comparison.rows]
        self.assertEqual(
            methods, ["FORM", "SORM-Breitung", "SORM-HR", "IS", "Subset", "MC"]
        )
        self.assertAlmostEqual(comparison.row("FORM").probability, exact, places=10)
        for row in comparison.rows:
            with self.subTest(method=row.method):
                self.assertGreater(row.evaluations, 0)
                if row.coefficient_of_variation is None:
                    self.assertAlmostEqual(row.probability / exact, 1.0, places=3)
                else:
                    tolerance = 4.0 * row.coefficient_of_variation
                    if row.method == "Subset":
                        tolerance *= 2.0
                    self.assertLess(abs(row.probability / exact - 1.0), tolerance)
        self.assertEqual(len(comparison.as_dicts()), 6)
        with self.assertRaises(KeyError):
            comparison.row("Kriging")
        with self.assertRaises(ValueError):
            compare_reliability_methods(state, methods=("form", "magic"))


if __name__ == "__main__":
    unittest.main()


class ComparisonRobustnessTests(unittest.TestCase):
    """Method comparison degrades gracefully when the median input fails."""

    def test_negative_beta_omits_sorm_rows(self) -> None:
        """With beta < 0 the comparison keeps FORM and MC and skips SORM."""
        from probstat.distributions import Normal as _Normal
        from probstat.modeling import JointInput as _Joint
        from probstat.modeling import ProbabilisticModel as _Model
        from probstat.modeling import RandomInput as _Input
        from probstat.reliability import ModelLimitState as _State
        from probstat.reliability import compare_reliability_methods as _compare
        from probstat.uq import QuantityOfInterest as _QoI

        model = _Model(
            _Joint.independent(_Input("x", _Normal(0.0, 1.0))),
            lambda p: {"y": p["x"]},
            (_QoI("y", "identity"),),
        )
        state = _State(model, "y", -1.0, failure="above")
        comparison = _compare(
            state, methods=("form", "sorm", "mc"), monte_carlo_samples=2000, seed=1
        )
        methods = [row.method for row in comparison.rows]
        self.assertIn("FORM", methods)
        self.assertNotIn("SORM-Breitung", methods)
        self.assertAlmostEqual(comparison.row("FORM").probability, 0.8413447, places=5)
