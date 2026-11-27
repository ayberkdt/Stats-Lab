"""Tests for the nonlinear autoregressive multi-fidelity GP (NARGP).

Oracles independent of the code under test: the kernel is compared with its
hand-written closed form and with central finite differences; the Monte Carlo
propagation of the two-level model is compared with a Gauss-Hermite quadrature
of the level-2 predictive moments over the level-1 predictive law; the
single-level model is the library GP; benchmark errors are measured against
the exact test functions, against the library's linear AR(1) co-kriging, and
against a single-fidelity GP on the expensive runs. Stochastic checks use
fixed seeds and Monte Carlo standard-error bands. The expensive replicated
study runs only with ``PROBSTAT_SLOW=1``.
"""

from __future__ import annotations

import functools
import math
import os
import random
import unittest
import warnings

from probstat.linalg import cholesky
from probstat.modeling import gauss_hermite
from probstat.surrogates import (
    fit_gaussian_process,
    fit_recursive_cokriging,
)
from probstat.surrogates.gp_advanced import StationaryKernel, fit_kernel_gp
from probstat.surrogates.nonlinear_multifidelity import (
    NargpPrediction,
    NonlinearMultiFidelityGP,
    fit_nargp,
    nargp_kernel,
    validate_nargp,
)

SE = "squared_exponential"


def _rmse(errors: list[float]) -> float:
    """Return the root mean square of ``errors``."""
    return math.sqrt(math.fsum(e * e for e in errors) / len(errors))


def _low(x: list[float]) -> float:
    """Return the low-fidelity benchmark ``sin(8 pi x)``."""
    return math.sin(8.0 * math.pi * x[0])


def _high(x: list[float]) -> float:
    """Return ``(x - sqrt(2)) f_l(x)^2``, a nonlinear cross-correlation."""
    return (x[0] - math.sqrt(2.0)) * _low(x) ** 2


def _coverage(truth: list[float], means: list[float], variances: list[float]) -> float:
    """Return the fraction of truths inside ``mean +/- 1.96 sd``."""
    inside = sum(
        abs(y - m) <= 1.96 * math.sqrt(v)
        for y, m, v in zip(truth, means, variances, strict=True)
    )
    return inside / len(truth)


class _Benchmark:
    """Fit NARGP, AR(1) co-kriging and a single-fidelity GP on one problem."""

    def __init__(
        self,
        designs: list[list[list[float]]],
        functions: list,
        tests: list[list[float]],
        *,
        samples: int = 100,
        ar1_restarts: int = 4,
    ) -> None:
        responses = [[f(x) for x in d] for f, d in zip(functions, designs, strict=True)]
        self.truth = [functions[-1](x) for x in tests]
        self.model = fit_nargp(designs, responses, seed=1, restarts=1)
        self.predictions = self.model.predict_mc_many(tests, samples=samples, seed=2)
        ar1 = fit_recursive_cokriging(
            designs, responses, seed=1, kernel=SE, restarts=ar1_restarts
        )
        single = fit_gaussian_process(designs[-1], responses[-1], seed=1)
        ar1_out = [ar1.predict_with_variance(x) for x in tests]
        single_out = [single.predict_with_variance(x) for x in tests]
        p = self.predictions
        self.rmse = _rmse([q.mean - y for q, y in zip(p, self.truth, strict=True)])
        self.plugin_rmse = _rmse(
            [q.plugin_mean - y for q, y in zip(p, self.truth, strict=True)]
        )
        self.ar1_rmse = _rmse(
            [m - y for (m, _), y in zip(ar1_out, self.truth, strict=True)]
        )
        self.single_rmse = _rmse(
            [m - y for (m, _), y in zip(single_out, self.truth, strict=True)]
        )
        self.coverage = _coverage(
            self.truth, [q.mean for q in p], [q.variance for q in p]
        )
        self.ar1_coverage = _coverage(
            self.truth, [m for m, _ in ar1_out], [v for _, v in ar1_out]
        )
        self.single_coverage = _coverage(
            self.truth, [m for m, _ in single_out], [v for _, v in single_out]
        )


def _grid(count: int) -> list[list[float]]:
    """Return ``count`` interior midpoints of ``[0, 1]``."""
    return [[(i + 0.5) / count] for i in range(count)]


_LOW_DESIGN = [[i / 49.0] for i in range(50)]
_HIGH_DESIGN = _LOW_DESIGN[1::4]  # 13 nested points


@functools.cache
def _nonlinear_benchmark() -> _Benchmark:
    """Fit the 50 / 13 nonlinear benchmark once per process."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _Benchmark(
            [_LOW_DESIGN, _HIGH_DESIGN], [_low, _high], _grid(30), samples=80
        )


class KernelTests(unittest.TestCase):
    """The structured kernel ``k_rho k_f + k_delta``."""

    def test_closed_form_value_and_parameter_layout(self) -> None:
        """Squared-exponential components match the hand formula."""
        kernel = nargp_kernel(
            1,
            kernel=SE,
            rho_length_scales=[0.4],
            f_length_scale=0.7,
            f_variance=2.0,
            delta_length_scales=[0.3],
            delta_variance=0.5,
        )
        self.assertEqual(len(kernel.log_parameters), 2 * 1 + 3)
        a, b = (0.2, 0.9), (0.5, 0.1)
        dx, df = a[0] - b[0], a[1] - b[1]
        expected = 2.0 * math.exp(-0.5 * (dx / 0.4) ** 2) * math.exp(
            -0.5 * (df / 0.7) ** 2
        ) + 0.5 * math.exp(-0.5 * (dx / 0.3) ** 2)
        self.assertAlmostEqual(kernel.evaluate(a, b), expected, places=14)

    def test_gradient_matches_central_differences(self) -> None:
        """Exact log-parameter gradients agree with finite differences."""
        kernel = nargp_kernel(
            2, kernel="matern52", rho_length_scales=[0.5, 0.8], f_variance=1.3
        )
        a, b = (0.1, 0.4, -0.3), (0.6, 0.2, 0.5)
        theta = kernel.log_parameters
        grad = kernel.gradient(a, b)
        for j in range(len(theta)):
            step = 1e-6
            up = list(theta)
            down = list(theta)
            up[j] += step
            down[j] -= step
            fd = (
                kernel.with_log_parameters(up).evaluate(a, b)
                - kernel.with_log_parameters(down).evaluate(a, b)
            ) / (2.0 * step)
            self.assertAlmostEqual(grad[j], fd, delta=1e-7 * max(1.0, abs(fd)))

    def test_gram_matrix_is_positive_definite(self) -> None:
        """A Gram matrix on distinct augmented points has a Cholesky factor."""
        kernel = nargp_kernel(1, kernel=SE)
        pts = [(i / 7.0, math.sin(i)) for i in range(8)]
        matrix = tuple(
            tuple(kernel.evaluate(p, q) + (1e-8 if p is q else 0.0) for q in pts)
            for p in pts
        )
        self.assertEqual(len(cholesky(matrix)), 8)

    def test_invalid_arguments(self) -> None:
        """Bad dimensions, names, and scales are rejected."""
        with self.assertRaises(ValueError):
            nargp_kernel(0)
        with self.assertRaises(ValueError):
            nargp_kernel(1, kernel="rbf")
        with self.assertRaises(ValueError):
            nargp_kernel(2, rho_length_scales=[1.0])
        with self.assertRaises(ValueError):
            nargp_kernel(1, f_variance=-1.0)


class StructureTests(unittest.TestCase):
    """Single-level reduction, nesting, validation, reproducibility."""

    @classmethod
    def setUpClass(cls) -> None:
        """Fit a small two-level smooth model once."""
        cls.low = [[i / 6.0] for i in range(7)]
        cls.high = [cls.low[i] for i in (0, 2, 3, 5, 6)]

        def f1(x: list[float]) -> float:
            return math.sin(2.0 * math.pi * x[0]) + 0.3 * x[0]

        def f2(x: list[float]) -> float:
            return (x[0] - math.sqrt(2.0)) * f1(x) ** 2

        cls.f1, cls.f2 = staticmethod(f1), staticmethod(f2)
        cls.responses = [[f1(x) for x in cls.low], [f2(x) for x in cls.high]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.model = fit_nargp(
                [cls.low, cls.high], cls.responses, seed=3, restarts=1
            )

    def test_single_level_equals_library_gp(self) -> None:
        """With one level NARGP is the ordinary kernel GP (exact)."""
        values = self.responses[0]
        model = fit_nargp([self.low], [values], seed=3, restarts=1)
        self.assertEqual(model.level_count, 1)
        gp = model.levels[0]
        direct = fit_kernel_gp(
            self.low, values, gp.kernel, nugget=gp.nugget, optimize=False
        )
        x = [0.41]
        self.assertAlmostEqual(
            model.predict_plugin(x)[0], direct.predict_with_variance(x)[0], places=12
        )
        out = model.predict_mc(x, samples=5, seed=0)
        self.assertEqual(out.level, 1)
        self.assertEqual((out.mean_mcse, out.variance_mcse), (0.0, 0.0))
        self.assertAlmostEqual(out.variance, direct.predict_with_variance(x)[1])

    def test_structure_and_nesting_flag(self) -> None:
        """Nested designs are detected; non-nested designs are accepted."""
        self.assertTrue(self.model.nested)
        self.assertEqual(self.model.design_sizes, (7, 5))
        self.assertEqual(len(self.model.levels[1].points[0]), 2)
        shifted = [[0.05], [0.31], [0.52], [0.7], [0.93]]
        values = [self.f2(x) for x in shifted]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loose = fit_nargp(
                [self.low, shifted], [self.responses[0], values], seed=3, restarts=1
            )
        self.assertFalse(loose.nested)
        self.assertTrue(math.isfinite(loose.predict_plugin([0.4])[0]))

    def test_training_input_is_lower_posterior_mean(self) -> None:
        """The second coordinate of level 2 is the level-1 posterior mean."""
        top = self.model.levels[1]
        for point in top.points:
            lower = self.model.levels[0].predict_with_variance(point[:1])[0]
            self.assertAlmostEqual(point[1], lower, places=12)
        # nested: the posterior mean reproduces the observation up to the nugget
        for x, y in zip(self.high, [self.f1(p) for p in self.high], strict=True):
            lower = self.model.levels[0].predict_with_variance(x)[0]
            self.assertAlmostEqual(lower, y, delta=1e-3)

    def test_gauss_hermite_oracle_for_propagation(self) -> None:
        """MC mean and variance match quadrature of the level-2 moments."""
        nodes, weights = gauss_hermite(60)
        for x in ([0.1], [0.8]):
            m1, v1 = self.model.levels[0].predict_with_variance(x)
            moments = [
                self.model.levels[1].predict_with_variance(
                    (x[0], m1 + math.sqrt(v1) * z)
                )
                for z in nodes
            ]
            mean = math.fsum(w * m for w, (m, _) in zip(weights, moments, strict=True))
            second = math.fsum(
                w * (v + m * m) for w, (m, v) in zip(weights, moments, strict=True)
            )
            variance = second - mean * mean
            out = self.model.predict_mc(x, samples=8000, seed=11)
            self.assertLess(abs(out.mean - mean), 4.5 * out.mean_mcse + 1e-12)
            self.assertLess(
                abs(out.variance - variance), 4.5 * out.variance_mcse + 1e-12
            )
            self.assertGreater(out.mean_mcse, 0.0)

    def test_mc_error_shrinks_like_inverse_root_n(self) -> None:
        """Sixteen times the draws quarter the reported standard errors."""
        x = [0.3]
        small = self.model.predict_mc(x, samples=500, seed=5)
        large = self.model.predict_mc(x, samples=8000, seed=6)
        ratio = large.variance_mcse / small.variance_mcse
        self.assertTrue(0.15 < ratio < 0.4, ratio)  # exact value 0.25
        ratio_mean = large.mean_mcse / small.mean_mcse
        self.assertTrue(0.15 < ratio_mean < 0.4, ratio_mean)

    def test_seed_reproducibility_and_independence_of_points(self) -> None:
        """Same seed reproduces; a point's result ignores the other points."""
        a = self.model.predict_mc([0.3], samples=50, seed=9)
        b = self.model.predict_mc([0.3], samples=50, seed=9)
        self.assertEqual(a, b)
        many = self.model.predict_mc_many([[0.2], [0.6]], samples=50, seed=4)
        alone = self.model.predict_mc_many([[0.6]], samples=50, seed=4)
        self.assertNotEqual(many[1], alone[0])  # child ("point", i) depends on i
        again = self.model.predict_mc_many([[0.2], [0.6]], samples=50, seed=4)
        self.assertEqual(many, again)

    def test_invalid_inputs(self) -> None:
        """Malformed designs and arguments are rejected, not repaired."""
        good = [self.low, self.high]
        with self.assertRaises(ValueError):
            fit_nargp([], [])
        with self.assertRaises(ValueError):
            fit_nargp(good, [self.responses[0]])
        with self.assertRaises(ValueError):
            fit_nargp([[[0.0], [1.0]], [[0.0], [1.0]]], [[0.0, 1.0], [0.0, 1.0]])
        with self.assertRaises(ValueError):
            fit_nargp([self.low, [[0.1, 0.2]] * 3], [self.responses[0], [1, 2, 3]])
        with self.assertRaises(ValueError):
            fit_nargp([[[0.0], [0.0], [1.0]]], [[0.0, 0.0, 1.0]])
        with self.assertRaises(ValueError):
            fit_nargp(good, self.responses, kernel="rbf")
        with self.assertRaises(ValueError):
            fit_nargp(good, self.responses, nugget=0.0)
        with self.assertRaises(ValueError):
            fit_nargp(
                [self.low, self.high], [self.responses[0], [1.0] * 5], seed=1
            )  # constant top level
        with self.assertRaises(ValueError):
            self.model.predict_mc([0.1, 0.2])
        with self.assertRaises(ValueError):
            self.model.predict_plugin([0.1], level=3)
        with self.assertRaises(ValueError):
            self.model.predict_mc([0.1], samples=1)
        with self.assertRaises(TypeError):
            self.model.predict_mc([0.1], samples=2.5)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            validate_nargp("model", [[0.1], [0.2]], [0.0, 1.0])  # type: ignore[arg-type]

    def test_level_one_prediction_of_a_two_level_model(self) -> None:
        """Asking for level 1 returns the exact Gaussian of the first GP."""
        x = [0.37]
        out = self.model.predict_mc(x, level=1, samples=3, seed=0)
        m, v = self.model.levels[0].predict_with_variance(x)
        self.assertEqual((out.mean, out.variance), (m, v))
        self.assertIsInstance(out, NargpPrediction)


class BenchmarkTests(unittest.TestCase):
    """Literature-style benchmarks against AR(1) co-kriging and one fidelity."""

    @classmethod
    def setUpClass(cls) -> None:
        """Fit every benchmark once (shared by the tests)."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.tests = _grid(30)
            cls.nonlinear = _nonlinear_benchmark()

            def linear(x: list[float]) -> float:
                return 2.0 * _low(x) + x[0]

            cls.linear = _Benchmark(
                [_LOW_DESIGN, _HIGH_DESIGN],
                [_low, linear],
                cls.tests,
                ar1_restarts=10,
            )

            def g1(x: list[float]) -> float:
                return math.sin(5.0 * x[0]) + x[1] ** 2

            def g2(x: list[float]) -> float:
                return g1(x) ** 2 + 0.5 * x[0] * x[1]

            rng = random.Random(5)
            low2 = [[rng.random(), rng.random()] for _ in range(40)]
            tests2 = [[rng.random(), rng.random()] for _ in range(30)]
            cls.planar = _Benchmark([low2, low2[:12]], [g1, g2], tests2)

            low3 = _LOW_DESIGN
            mid3 = _LOW_DESIGN[1::2]
            top3 = _LOW_DESIGN[1::6]

            def h3(x: list[float]) -> float:
                return 0.8 * _high(x) + 0.4 * _high(x) ** 2

            cls.three = _Benchmark([low3, mid3, top3], [_low, _high, h3], cls.tests)
            cls.three_designs = (len(low3), len(mid3), len(top3))
            print(
                "\n[nargp benchmarks] rmse nargp / plug-in / ar1 / single, "
                "coverage nargp / ar1 / single"
            )
            for name, b in (
                ("nonlinear 1-D", cls.nonlinear),
                ("linear 1-D", cls.linear),
                ("2-D", cls.planar),
                ("3-level", cls.three),
            ):
                print(
                    f"  {name:14s} {b.rmse:.5f} {b.plugin_rmse:.5f} "
                    f"{b.ar1_rmse:.5f} {b.single_rmse:.5f}   "
                    f"{b.coverage:.3f} {b.ar1_coverage:.3f} {b.single_coverage:.3f}"
                )

    def test_nonlinear_cross_correlation_beats_linear_and_single(self) -> None:
        """NARGP is at least 4x more accurate than AR(1) and the single GP."""
        b = self.nonlinear
        self.assertLess(b.rmse, 0.1)
        self.assertLess(b.rmse, 0.25 * b.ar1_rmse)
        self.assertLess(b.rmse, 0.25 * b.single_rmse)

    def test_nonlinear_predictive_coverage(self) -> None:
        """The plug-in-hyperparameter 95% interval covers at least 70%.

        Measured 0.833 on 30 test points (the nominal 0.95 is not reached
        because the kernel hyperparameters are treated as known).
        """
        self.assertGreaterEqual(self.nonlinear.coverage, 0.7)

    def test_linear_relation_not_worse_than_ar1_by_a_margin(self) -> None:
        """For f_h = 2 f_l + x NARGP stays within 0.05 of AR(1) (RMS about 1.5)."""
        b = self.linear
        self.assertLess(b.rmse, b.ar1_rmse + 0.05)
        self.assertLess(b.rmse, 0.05)
        self.assertGreaterEqual(b.coverage, 0.9)

    def test_two_dimensional_case(self) -> None:
        """2-D nonlinear relation: NARGP at least 10x better than both."""
        b = self.planar
        self.assertEqual(b.model.input_dimension, 2)
        self.assertLess(b.rmse, 0.02)
        self.assertLess(b.rmse, 0.1 * b.ar1_rmse)
        self.assertLess(b.rmse, 0.1 * b.single_rmse)

    def test_three_level_case(self) -> None:
        """Three fidelities with nested designs 50 / 25 / 9."""
        b = self.three
        self.assertEqual(b.model.design_sizes, self.three_designs)
        self.assertTrue(b.model.nested)
        self.assertEqual(len(b.model.levels[2].points[0]), 2)
        self.assertLess(b.rmse, 0.05)
        self.assertLess(b.rmse, 0.2 * b.ar1_rmse)
        self.assertLess(b.rmse, 0.2 * b.single_rmse)
        # measured 0.60 (known-hyperparameter intervals are too narrow here)
        self.assertGreaterEqual(b.coverage, 0.4)
        self.assertGreater(b.coverage, b.ar1_coverage)

    def test_plugin_equals_propagated_mean_when_lower_level_is_certain(self) -> None:
        """With a well-resolved low fidelity the two predictors agree."""
        b = self.nonlinear
        for q in b.predictions:
            self.assertAlmostEqual(
                q.mean, q.plugin_mean, delta=5.0 * q.mean_mcse + 1e-4
            )

    def test_plugin_underestimates_variance_with_sparse_low_fidelity(self) -> None:
        """With 5 low runs the plug-in variance falls below the propagated one.

        Measured on 40 test points: mean ratio 0.87, smallest ratio 0.10.
        """

        def f1(x: list[float]) -> float:
            return math.sin(2.0 * math.pi * x[0]) + 0.3 * x[0]

        def f2(x: list[float]) -> float:
            return (x[0] - math.sqrt(2.0)) * f1(x) ** 2

        low = [[i / 4.0] for i in range(5)]
        high = [low[i] for i in (0, 1, 3, 4)]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = fit_nargp(
                [low, high], [[f1(x) for x in low], [f2(x) for x in high]], seed=1
            )
            preds = model.predict_mc_many(_grid(40), samples=100, seed=2)
        ratios = [p.plugin_variance / p.variance for p in preds]
        self.assertLess(math.fsum(ratios) / len(ratios), 0.95)
        self.assertLess(min(ratios), 0.5)
        # in expectation the propagated variance is never smaller; allow MC noise
        for p in preds:
            self.assertLess(
                p.plugin_variance, p.variance + 4.0 * p.variance_mcse + 1e-12
            )


class DiagnosticsTests(unittest.TestCase):
    """Top-level emulator validation and Monte Carlo error reporting."""

    @classmethod
    def setUpClass(cls) -> None:
        """Fit the 1-D nonlinear benchmark and validate on a fresh grid."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cls.model = _nonlinear_benchmark().model
            cls.points = _grid(25)
            cls.values = [_high(x) for x in cls.points]
            cls.report = validate_nargp(
                cls.model, cls.points, cls.values, samples=150, seed=7
            )

    def test_report_matches_direct_computation(self) -> None:
        """RMSE, coverage, and the library diagnostics are consistent."""
        r = self.report
        direct = _rmse(
            [p.mean - y for p, y in zip(r.predictions, self.values, strict=True)]
        )
        self.assertAlmostEqual(r.rmse, direct, places=14)
        self.assertEqual(len(r.emulator.standardized_errors), 25)
        for d, p, y in zip(
            r.emulator.standardized_errors, r.predictions, self.values, strict=True
        ):
            self.assertAlmostEqual(d, (y - p.mean) / p.standard_deviation, places=9)
        self.assertEqual(r.emulator.coverage_level, 0.95)
        self.assertAlmostEqual(r.coverage, r.emulator.coverage, delta=0.05)
        self.assertLess(r.rmse, 0.1)

    def test_monte_carlo_error_is_small_against_the_variance(self) -> None:
        """With 150 draws the variance error is below 25% of the variance."""
        self.assertLess(self.report.max_relative_variance_mcse, 0.25)
        self.assertLess(self.report.max_mean_mcse, 0.05)

    def test_validation_input_errors(self) -> None:
        """Mismatched sizes are rejected."""
        with self.assertRaises(ValueError):
            validate_nargp(self.model, [[0.1], [0.2]], [0.0])
        with self.assertRaises(ValueError):
            validate_nargp(self.model, [[0.1]], [0.0])


@unittest.skipUnless(os.environ.get("PROBSTAT_SLOW") == "1", "slow tier")
class ReplicatedStudy(unittest.TestCase):
    """Random nested expensive designs: errors and coverage over replicates."""

    def test_random_high_designs(self) -> None:
        """Median NARGP RMSE beats AR(1) over 10 random 14-point designs."""
        rng = random.Random(2026)
        tests = _grid(40)
        rows = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for _ in range(10):
                idx = sorted(rng.sample(range(50), 14))
                high = [_LOW_DESIGN[i] for i in idx]
                b = _Benchmark(
                    [_LOW_DESIGN, high], [_low, _high], tests, ar1_restarts=10
                )
                rows.append((b.rmse, b.ar1_rmse, b.single_rmse, b.coverage))
        med = [sorted(c)[len(c) // 2] for c in zip(*rows, strict=True)]
        print(
            "\n[nargp slow] median rmse nargp/ar1/single, coverage: "
            f"{med[0]:.4f} {med[1]:.4f} {med[2]:.4f} {med[3]:.3f}; "
            f"nargp rmse range {min(r[0] for r in rows):.4f}-"
            f"{max(r[0] for r in rows):.4f}"
        )
        self.assertLess(med[0], med[1])
        self.assertLess(med[0], med[2])


class ModelTypeTests(unittest.TestCase):
    """Dataclass-level validation."""

    def test_inconsistent_levels_rejected(self) -> None:
        """A level of the wrong input width is rejected."""
        low = [[0.0], [0.5], [1.0]]
        gp = fit_kernel_gp(
            low, [0.0, 1.0, 0.0], StationaryKernel(SE, 1.0, (0.5,)), optimize=False
        )
        with self.assertRaises(ValueError):
            NonlinearMultiFidelityGP((gp, gp), 1, True, SE)
        with self.assertRaises(ValueError):
            NonlinearMultiFidelityGP((), 1, True, SE)


if __name__ == "__main__":
    unittest.main()
