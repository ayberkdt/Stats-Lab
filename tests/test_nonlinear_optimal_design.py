"""Tests of nonlinear Bayesian optimal experimental design.

Experimental (added in 0.28).

The oracle is the closed-form linear-Gaussian EIG
``1/2 log det(I + L' J' R^-1 J L)``, evaluated with the dense routines of
``probstat.linalg``. Estimator bias and spread are measured over replicate
runs with fixed seeds.

References:
    [EQ-DES-NOED-001], [EQ-DES-NOED-002], [EQ-DES-NOED-003],
    [EQ-DES-NOED-004], [EQ-DES-NOED-005], [EQ-DES-NOED-006],
    [EQ-OED-EIG-LIN-001].

"""

from __future__ import annotations

import math
import os
import statistics
import unittest

from probstat.design.nonlinear_optimal import (
    NonlinearExperimentModel,
    contrastive_lower_bound_eig,
    estimate_eig,
    exhaustive_nonlinear_design,
    exponential_decay_model,
    greedy_nonlinear_design,
    laplace_eig,
    nested_monte_carlo_eig,
    source_location_model,
)
from probstat.linalg import cholesky, log_det_spd

PRIOR = ((1.0, 0.3), (0.3, 0.5))
SIGMA = 0.5


def linear_model():
    """Return g = theta_0 + theta_1 d with a correlated prior."""
    return NonlinearExperimentModel(
        lambda theta, d: theta[0] + theta[1] * d, (0.0, 0.0), PRIOR, SIGMA
    )


def linear_exact_eig(design):
    """Return the closed-form EIG of the linear model for a design."""
    factor = cholesky(PRIOR)
    jac = [(1.0, d) for d in design]
    b = [
        [sum(jac[j][k] * factor[k][i] for k in range(2)) / SIGMA for i in range(2)]
        for j in range(len(design))
    ]
    gram = tuple(
        tuple(
            sum(b[j][a] * b[j][c] for j in range(len(design)))
            + (1.0 if a == c else 0.0)
            for c in range(2)
        )
        for a in range(2)
    )
    return 0.5 * log_det_spd(gram)


class LinearGaussianOracleTests(unittest.TestCase):
    """Estimators against the closed form."""

    design = (0.0, 1.0, 2.0)

    def test_closed_form_value(self):
        """The oracle itself: one design point with a scalar prior."""
        model = NonlinearExperimentModel(
            lambda theta, d: theta[0] * d, (0.0,), ((4.0,),), 0.5
        )
        # EIG = 1/2 log(1 + 4 d^2 / 0.25)
        expected = 0.5 * math.log(1.0 + 4.0 * 2.0**2 / 0.25)
        self.assertAlmostEqual(
            laplace_eig(model, (2.0,), samples=5, seed=1).value, expected, delta=1e-8
        )

    def test_laplace_is_exact_for_a_linear_model(self):
        """Central differences of a linear map are exact to rounding."""
        for design in ((1.0,), (0.0, 1.0), self.design):
            estimate = laplace_eig(linear_model(), design, samples=20, seed=3)
            self.assertAlmostEqual(estimate.value, linear_exact_eig(design), delta=1e-7)
            self.assertEqual(estimate.bias, "approximation")

    def test_bias_directions_over_replicates(self):
        """Nested is biased up, the contrastive bound down; spreads are finite."""
        exact = linear_exact_eig(self.design)
        nested = [
            nested_monte_carlo_eig(
                linear_model(), self.design, outer=150, inner=15, seed=s
            ).value
            for s in range(12)
        ]
        bound = [
            contrastive_lower_bound_eig(
                linear_model(), self.design, samples=150, seed=100 + s
            ).value
            for s in range(12)
        ]
        nested_bias = statistics.mean(nested) - exact
        nested_se = statistics.stdev(nested) / math.sqrt(len(nested))
        bound_bias = statistics.mean(bound) - exact
        bound_se = statistics.stdev(bound) / math.sqrt(len(bound))
        # upward bias of order (1/M) times a variance ratio: far above its error
        self.assertGreater(nested_bias, 5.0 * nested_se)
        # a lower bound in expectation: the mean must not exceed exact beyond noise
        self.assertLess(bound_bias, 3.0 * bound_se)
        self.assertGreater(statistics.stdev(nested), 0.0)
        self.assertLess(statistics.stdev(bound), 0.5)

    def test_nested_bias_shrinks_with_inner_size(self):
        """Increasing M reduces the upward bias (common outer seeds)."""
        exact = linear_exact_eig(self.design)
        small = nested_monte_carlo_eig(
            linear_model(), self.design, outer=300, inner=5, seed=7
        ).value
        large = nested_monte_carlo_eig(
            linear_model(), self.design, outer=300, inner=80, seed=7
        ).value
        self.assertGreater(small - exact, large - exact)
        self.assertLess(abs(large - exact), abs(small - exact))

    def test_lower_bound_never_exceeds_log_n(self):
        """The contrastive bound is capped by log N for any design."""
        model = NonlinearExperimentModel(
            lambda theta, d: theta[0] * d, (0.0,), ((100.0,),), 0.01
        )
        samples = 64
        estimate = contrastive_lower_bound_eig(model, (3.0,), samples=samples, seed=2)
        self.assertLessEqual(estimate.value, math.log(samples) + 1e-12)
        self.assertGreater(estimate.value, math.log(samples) - 0.5)
        self.assertEqual(estimate.bias, "downward")


class NonlinearModelTests(unittest.TestCase):
    """Exponential decay and source location."""

    def test_zero_information_design(self):
        """At t = 0 the decay prediction is independent of the parameter."""
        model = exponential_decay_model(
            amplitude=1.0, log_rate_mean=0.0, log_rate_std=0.5, noise_std=0.05
        )
        for method in ("laplace", "nested", "lower-bound"):
            estimate = estimate_eig(
                model, (0.0,), method=method, samples=50, inner=10, seed=1
            )
            self.assertAlmostEqual(estimate.value, 0.0, delta=1e-12, msg=method)

    def test_decay_information_profile_and_estimator_bracket(self):
        """Information peaks near t = 1/lambda; lower bound <= nested."""
        model = exponential_decay_model(
            amplitude=1.0, log_rate_mean=0.0, log_rate_std=0.5, noise_std=0.05
        )
        laplace = {
            t: laplace_eig(model, (t,), samples=300, seed=4).value
            for t in (0.1, 1.0, 6.0)
        }
        self.assertGreater(laplace[1.0], laplace[0.1])
        self.assertGreater(laplace[1.0], laplace[6.0])
        nested = nested_monte_carlo_eig(model, (1.0,), outer=300, inner=80, seed=4)
        bound = contrastive_lower_bound_eig(model, (1.0,), samples=300, seed=4)
        self.assertLess(bound.value, nested.value + 3.0 * nested.standard_error)
        # near-linear regime: Laplace is within a few tenths of the sampling estimates
        self.assertLess(abs(laplace[1.0] - nested.value), 0.3)

    def test_laplace_error_in_strongly_nonlinear_source_model(self):
        """Laplace overshoots the sampling estimates for a strong nonlinearity."""
        model = source_location_model(
            strength=1.0, height=1.0, prior_mean=0.0, prior_std=1.0, noise_std=0.02
        )
        laplace = laplace_eig(model, (0.0,), samples=300, seed=2).value
        nested = nested_monte_carlo_eig(
            model, (0.0,), outer=300, inner=60, seed=2
        ).value
        bound = contrastive_lower_bound_eig(model, (0.0,), samples=300, seed=2).value
        self.assertLess(bound, nested + 0.2)
        self.assertGreater(laplace - nested, 0.2)

    def test_reproducibility_and_cost_accounting(self):
        """Equal seeds agree exactly; evaluation counts follow the formulas."""
        calls = [0]

        def forward(theta, d):
            calls[0] += 1
            return theta[0] * d

        model = NonlinearExperimentModel(forward, (0.0,), ((1.0,),), 0.3)
        design = (1.0, 2.0)
        a = nested_monte_carlo_eig(model, design, outer=20, inner=7, seed=5)
        self.assertEqual(calls[0], a.model_evaluations)
        self.assertEqual(a.model_evaluations, 20 * (1 + 7) * 2)
        b = nested_monte_carlo_eig(model, design, outer=20, inner=7, seed=5)
        c = nested_monte_carlo_eig(model, design, outer=20, inner=7, seed=6)
        self.assertEqual(a.value, b.value)
        self.assertNotEqual(a.value, c.value)
        calls[0] = 0
        lb = contrastive_lower_bound_eig(model, design, samples=20, seed=5)
        self.assertEqual(calls[0], lb.model_evaluations)
        self.assertEqual(lb.model_evaluations, 20 * 2)


class DesignSearchTests(unittest.TestCase):
    """Exhaustive and greedy choice."""

    @staticmethod
    def decay():
        """Return the shared decay model."""
        return exponential_decay_model(
            amplitude=1.0, log_rate_mean=0.0, log_rate_std=0.5, noise_std=0.05
        )

    def test_exhaustive_ranking_and_greedy_consistency(self):
        """Greedy is feasible in the exhaustive set, so it cannot beat it."""
        candidates = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
        full = exhaustive_nonlinear_design(
            self.decay(), candidates, 2, method="laplace", samples=150, seed=1
        )
        greedy = greedy_nonlinear_design(
            self.decay(), candidates, 2, method="laplace", samples=150, seed=1
        )
        self.assertEqual(len(full.ranking), math.comb(len(candidates), 2))
        values = [estimate.value for _, estimate in full.ranking]
        self.assertEqual(values, sorted(values, reverse=True))
        self.assertGreaterEqual(full.value, greedy.value - 1e-12)
        self.assertNotIn(0.0, full.design)
        self.assertNotIn(0.0, greedy.design)
        # the greedy path is nested: its first point is the best single sensor
        best_single = exhaustive_nonlinear_design(
            self.decay(), candidates, 1, method="laplace", samples=150, seed=1
        )
        self.assertEqual(greedy.ranking[0][0], best_single.design)
        self.assertEqual(len(greedy.ranking), 2)

    def test_repeats_and_sampling_methods(self):
        """Repeats are allowed on request; nested and bound methods run."""
        candidates = [0.5, 1.0, 2.0]
        with self.assertRaises(ValueError):
            greedy_nonlinear_design(self.decay(), candidates, 4, samples=20)
        result = greedy_nonlinear_design(
            self.decay(), candidates, 4, samples=20, allow_repeats=True
        )
        self.assertEqual(len(result.design), 4)
        for method in ("nested", "lower-bound"):
            run = exhaustive_nonlinear_design(
                self.decay(), candidates, 2, method=method, samples=40, inner=10
            )
            self.assertEqual(run.method, method)
            self.assertEqual(len(run.ranking), 3)
            self.assertGreater(run.model_evaluations, 0)

    def test_invalid_inputs(self):
        """Bad arguments raise instead of returning something."""
        model = linear_model()
        with self.assertRaises(TypeError):
            laplace_eig("model", (1.0,))
        with self.assertRaises(TypeError):
            NonlinearExperimentModel(None, (0.0,), ((1.0,),), 1.0)
        with self.assertRaises(ValueError):
            NonlinearExperimentModel(lambda t, d: 0.0, (0.0,), ((-1.0,),), 1.0)
        with self.assertRaises(ValueError):
            NonlinearExperimentModel(lambda t, d: 0.0, (0.0,), ((1.0,),), 0.0)
        with self.assertRaises(ValueError):
            NonlinearExperimentModel(
                lambda t, d: 0.0, (0.0, 0.0), ((1.0, 0.1), (0.2, 1.0)), 1.0
            )
        with self.assertRaises(ValueError):
            laplace_eig(model, ())
        with self.assertRaises(ValueError):
            laplace_eig(model, (float("nan"),))
        with self.assertRaises(ValueError):
            nested_monte_carlo_eig(model, (1.0,), outer=1)
        with self.assertRaises(ValueError):
            nested_monte_carlo_eig(model, (1.0,), inner=0)
        with self.assertRaises(ValueError):
            contrastive_lower_bound_eig(model, (1.0,), samples=1)
        with self.assertRaises(ValueError):
            laplace_eig(model, (1.0,), step=0.0)
        with self.assertRaises(ValueError):
            estimate_eig(model, (1.0,), method="bogus")
        with self.assertRaises(ValueError):
            exhaustive_nonlinear_design(model, [1.0, 1.0], 1)
        with self.assertRaises(ValueError):
            exhaustive_nonlinear_design(model, [1.0, 2.0], 3)
        with self.assertRaises(ValueError):
            exhaustive_nonlinear_design(model, list(range(40)), 10)
        with self.assertRaises(ValueError):
            exponential_decay_model(
                amplitude=0.0, log_rate_mean=0.0, log_rate_std=1.0, noise_std=1.0
            )
        with self.assertRaises(ValueError):
            source_location_model(
                strength=1.0, height=0.0, prior_mean=0.0, prior_std=1.0, noise_std=1.0
            )
        bad = NonlinearExperimentModel(
            lambda t, d: float("inf"), (0.0,), ((1.0,),), 1.0
        )
        with self.assertRaises(ValueError):
            nested_monte_carlo_eig(bad, (1.0,))


@unittest.skipUnless(os.environ.get("PROBSTAT_SLOW") == "1", "slow tier")
class SlowBiasStudy(unittest.TestCase):
    """Estimator bias and standard deviation over many replicates."""

    def test_bias_table(self):
        """Print bias and spread of the three estimators on the linear oracle."""
        design = (0.0, 1.0, 2.0)
        exact = linear_exact_eig(design)
        rows = []
        for name, run in (
            (
                "nested M=10",
                lambda s: nested_monte_carlo_eig(
                    linear_model(), design, outer=200, inner=10, seed=s
                ),
            ),
            (
                "nested M=50",
                lambda s: nested_monte_carlo_eig(
                    linear_model(), design, outer=200, inner=50, seed=s
                ),
            ),
            (
                "lower bound N=200",
                lambda s: contrastive_lower_bound_eig(
                    linear_model(), design, samples=200, seed=s
                ),
            ),
            (
                "laplace N=200",
                lambda s: laplace_eig(linear_model(), design, samples=200, seed=s),
            ),
        ):
            values = [run(s).value for s in range(100)]
            rows.append(
                (name, statistics.mean(values) - exact, statistics.stdev(values))
            )
        print(f"exact EIG = {exact:.6f}")
        for name, bias, sd in rows:
            print(f"{name:20s} bias {bias:+.4f} sd {sd:.4f}")
        self.assertGreater(rows[0][1], rows[1][1])


if __name__ == "__main__":
    unittest.main()
