"""Tests for the advanced GP models of ``probstat.surrogates.gp_advanced``.

Closed forms and finite differences come first (kernel algebra, derivative
kernel blocks, operator covariances), then independent oracles (the exact GP of
``probstat.surrogates.gaussian_process``, the sparse-to-exact limit), then
noisy recovery studies with fixed seeds, then adversarial failure cases. The
file avoids NumPy and runs in well under 30 s.
"""

from __future__ import annotations

import math
import random
import unittest
import warnings
from collections.abc import Sequence

from probstat.linalg import Vector
from probstat.surrogates import (
    GaussianProcessConditioningWarning,
    fit_gaussian_process,
    gaussian_process_log_marginal_likelihood,
)
from probstat.surrogates.gp_advanced import (
    Kernel,
    LinearDifferentialOperator,
    LinearKernel,
    PeriodicKernel,
    ProductKernel,
    RationalQuadraticKernel,
    StationaryKernel,
    SumKernel,
    WhiteKernel,
    _kernel_marginal,
    _lmc_marginal,
    _sparse_terms,
    derivative_kernel_blocks,
    fit_derivative_gp,
    fit_heteroscedastic_gp,
    fit_kernel_gp,
    fit_multi_output_gp,
    fit_operator_gp,
    fit_sparse_gp,
    gaussian_process_gradient,
    operator_covariances,
    select_inducing_points,
    sparse_gp_cost,
)

SE = "squared_exponential"


def _fd_kernel_gradient(
    kernel: Kernel, x: Sequence[float], y: Sequence[float]
) -> list[float]:
    """Central differences of ``k(x, y)`` in every log parameter."""
    theta = list(kernel.log_parameters)
    out = []
    for j in range(len(theta)):
        h = 1e-6
        up, down = list(theta), list(theta)
        up[j] += h
        down[j] -= h
        out.append(
            (
                kernel.with_log_parameters(up).evaluate(x, y)
                - kernel.with_log_parameters(down).evaluate(x, y)
            )
            / (2.0 * h)
        )
    return out


class CountingKernel(Kernel):
    """Wrap a kernel and count its ``evaluate`` calls (test instrument)."""

    __slots__ = ("calls", "inner")

    def __init__(self, inner: Kernel, calls: list[int]) -> None:
        """Store the wrapped kernel and the shared counter."""
        self.inner = inner
        self.calls = calls

    @property
    def log_parameters(self) -> Vector:
        """Return the wrapped log parameters."""
        return self.inner.log_parameters

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Return the wrapped parameter names."""
        return self.inner.parameter_names

    def with_log_parameters(self, log_parameters: Sequence[float]) -> Kernel:
        """Return a counting wrapper with new parameters."""
        return CountingKernel(
            self.inner.with_log_parameters(log_parameters), self.calls
        )

    def evaluate(self, x: Sequence[float], y: Sequence[float]) -> float:
        """Count and delegate."""
        self.calls[0] += 1
        return self.inner.evaluate(x, y)

    def gradient(self, x: Sequence[float], y: Sequence[float]) -> Vector:
        """Delegate."""
        return self.inner.gradient(x, y)


class KernelAlgebraTests(unittest.TestCase):
    """Closed forms, exact gradients, and fitting of composite kernels."""

    def test_closed_forms(self) -> None:
        """Each kernel equals its textbook formula at hand-picked inputs."""
        periodic = PeriodicKernel(2.0, 0.7, 1.5)
        self.assertAlmostEqual(periodic.evaluate((0.3,), (0.3 + 1.5,)), 2.0, places=12)
        self.assertAlmostEqual(
            periodic.evaluate((0.0,), (0.375,)),
            2.0 * math.exp(-2.0 * math.sin(math.pi * 0.25) ** 2 / 0.49),
            places=12,
        )
        rq = RationalQuadraticKernel(1.3, 0.8, 2.0)
        self.assertAlmostEqual(
            rq.evaluate((0.0, 0.0), (0.6, 0.8)),
            1.3 * (1.0 + 1.0 / (2.0 * 2.0 * 0.64)) ** -2.0,
            places=12,
        )
        huge = RationalQuadraticKernel(1.0, 0.8, 1e8)
        self.assertAlmostEqual(
            huge.evaluate((0.0,), (1.0,)), math.exp(-0.5 / 0.64), places=7
        )
        linear = LinearKernel(0.5, 2.0)
        self.assertAlmostEqual(
            linear.evaluate((1.0, 2.0), (3.0, -1.0)), 0.5 + 2.0, places=14
        )

    def test_gradients_match_finite_differences(self) -> None:
        """``gradient`` is the exact derivative in the log parameters."""
        x, y = (0.3, -0.2), (1.1, 0.4)
        kernels: list[Kernel] = [
            StationaryKernel("matern52", 1.5, (0.7, 1.3)),
            StationaryKernel("matern32", 0.8, (0.5, 0.9)),
            StationaryKernel(SE, 1.5, (0.7, 1.3)),
            PeriodicKernel(1.2, 0.8, 1.7),
            RationalQuadraticKernel(0.9, 1.1, 2.0),
            LinearKernel(0.5, 2.0),
            StationaryKernel(SE, 1.0, (1.0, 1.0)) * PeriodicKernel(1.0, 1.0, 2.0)
            + LinearKernel(1.0, 1.0)
            + WhiteKernel(0.3),
        ]
        for kernel in kernels:
            exact = kernel.gradient(x, y)
            numeric = _fd_kernel_gradient(kernel, x, y)
            for a, b in zip(exact, numeric, strict=True):
                self.assertAlmostEqual(a, b, delta=1e-7 * (1.0 + abs(a)))

    def test_algebra_and_white_semantics(self) -> None:
        """Sums add, products multiply, white noise lives on the diagonal only."""
        base = StationaryKernel(SE, 1.0, (1.0,))
        white = WhiteKernel(0.25)
        total = base + white
        self.assertIsInstance(total, SumKernel)
        self.assertEqual(total.parameter_count, 3)
        self.assertEqual(total.evaluate((0.0,), (0.0,)), 1.0)
        matrix = total.matrix([(0.0,), (1.0,)])
        self.assertAlmostEqual(matrix[0][0], 1.25, places=14)
        self.assertAlmostEqual(matrix[0][1], math.exp(-0.5), places=14)
        self.assertEqual(total.cross([(0.0,)], [(0.0,)])[0][0], 1.0)
        product = base * PeriodicKernel(1.0, 1.0, 1.0)
        self.assertIsInstance(product, ProductKernel)
        self.assertAlmostEqual(
            product.evaluate((0.0,), (0.5,)),
            base.evaluate((0.0,), (0.5,))
            * PeriodicKernel(1.0, 1.0, 1.0).evaluate((0.0,), (0.5,)),
            places=14,
        )
        with self.assertRaises(ValueError):
            _ = base * white
        with self.assertRaises(TypeError):
            _ = base + 1.0  # type: ignore[operator]
        with self.assertRaises(ValueError):
            total.with_log_parameters((0.0, 0.0))
        with self.assertRaises(ValueError):
            PeriodicKernel(1.0, 1.0, -1.0)

    def test_composite_kernel_is_positive_semidefinite(self) -> None:
        """Sum and product kernels give Cholesky-factorizable matrices."""
        kernel = (
            StationaryKernel("matern52", 1.0, (0.6,)) * PeriodicKernel(1.0, 1.0, 1.3)
            + RationalQuadraticKernel(0.4, 0.9, 1.5)
            + LinearKernel(0.1, 0.1)
        )
        points = [(0.37 * i,) for i in range(12)]
        matrix = kernel.matrix(points)
        for i in range(12):
            matrix[i][i] += 1e-9
        _kernel_marginal(
            kernel,
            points,
            [math.sin(p[0]) for p in points],
            [1e-9] * 12,
            gradient=False,
        )

    def test_marginal_likelihood_matches_exact_gp_and_finite_differences(self) -> None:
        """Value and gradient match ``gaussian_process`` and central differences."""
        gen = random.Random(11)
        points = [(gen.uniform(0.0, 3.0),) for _ in range(9)]
        targets = [math.sin(2.0 * p[0]) + 0.1 * gen.gauss(0.0, 1.0) for p in points]
        theta = (0.3, math.log(0.9))
        kernel = StationaryKernel(SE, math.exp(theta[0]), (math.exp(theta[1]),))
        value, grad, _, _, _ = _kernel_marginal(
            kernel, points, targets, [1e-3] * 9, gradient=True
        )
        ref_value, ref_grad = gaussian_process_log_marginal_likelihood(
            points, targets, theta, kernel=SE, nugget=1e-3
        )
        self.assertAlmostEqual(value, ref_value, places=9)
        for a, b in zip(grad, ref_grad, strict=True):
            self.assertAlmostEqual(a, b, places=8)
        composite = (
            PeriodicKernel(1.1, 0.9, 2.2) + LinearKernel(0.3, 0.2) + WhiteKernel(0.05)
        )
        _, grad, _, _, _ = _kernel_marginal(
            composite, points, targets, [1e-6] * 9, gradient=True
        )
        base = list(composite.log_parameters)
        for j in range(len(base)):
            h = 1e-6
            up, down = list(base), list(base)
            up[j] += h
            down[j] -= h
            plus = _kernel_marginal(
                composite.with_log_parameters(up),
                points,
                targets,
                [1e-6] * 9,
                gradient=False,
            )[0]
            minus = _kernel_marginal(
                composite.with_log_parameters(down),
                points,
                targets,
                [1e-6] * 9,
                gradient=False,
            )[0]
            self.assertAlmostEqual(
                grad[j], (plus - minus) / (2.0 * h), delta=1e-5 * (1.0 + abs(grad[j]))
            )

    def test_fit_recovers_period_trend_and_noise(self) -> None:
        """Type-II ML on periodic + linear + white recovers the generating law."""
        gen = random.Random(1)
        points = [(i / 30.0 * 3.0,) for i in range(30)]
        truth = lambda x: math.sin(2.0 * math.pi * x) + 0.3 * x  # noqa: E731
        values = [truth(p[0]) + 0.05 * gen.gauss(0.0, 1.0) for p in points]
        start = (
            PeriodicKernel(1.0, 1.0, 1.2) + LinearKernel(0.5, 0.5) + WhiteKernel(0.05)
        )
        bounds = [
            (-5, 5),
            (-2, 3),
            (math.log(0.6), math.log(1.8)),
            (-8, 3),
            (-8, 3),
            (-12, 0),
        ]
        fit = fit_kernel_gp(
            points, values, start, restarts=1, seed=1, log_bounds=bounds
        )
        self.assertTrue(fit.converged, fit.stop_reason)
        periodic = fit.kernel.left.left  # type: ignore[attr-defined]
        white = fit.kernel.right  # type: ignore[attr-defined]
        self.assertAlmostEqual(periodic.period, 1.0, delta=0.03)
        self.assertAlmostEqual(white.variance, 0.05**2, delta=0.0025)
        errors = [
            abs(fit.predict((0.05 + 0.3 * j,)) - truth(0.05 + 0.3 * j))
            for j in range(9)
        ]
        self.assertLess(max(errors), 0.08)
        mean, variance = fit.predict_with_variance((1.3,))
        self.assertGreater(variance, 0.0)
        self.assertGreater(
            fit.predict_with_variance((1.3,), include_noise=True)[1], variance
        )
        self.assertFalse(fit.ill_conditioned or fit.condition_number < 1.0)
        self.assertAlmostEqual(mean, truth(1.3), delta=0.08)

    def test_failures_are_reported_not_hidden(self) -> None:
        """Singular matrices raise, conditioning is reported, inputs are validated."""
        points = [(0.0,), (0.0,), (1.0,)]
        kernel = StationaryKernel(SE, 1.0, (1.0,))
        with self.assertRaisesRegex(ValueError, "no jitter"):
            fit_kernel_gp(points, [0.0, 0.1, 1.0], kernel, nugget=1e-30, optimize=False)
        with self.assertRaises(ValueError):
            fit_kernel_gp([(0.0,), (1.0,)], [1.0, 1.0], kernel)
        with self.assertRaises(ValueError):
            fit_kernel_gp([(0.0,), (1.0,)], [1.0, 2.0], kernel, nugget=0.0)
        with self.assertRaises(TypeError):
            fit_kernel_gp([(0.0,), (1.0,)], [1.0, 2.0], "rbf")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            fit_kernel_gp(
                [(0.0,), (1.0,)], [1.0, 2.0], kernel, log_bounds=[(-1.0, 1.0)]
            )
        close = [(0.01 * i,) for i in range(12)]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = fit_kernel_gp(
                close,
                [math.sin(p[0]) for p in close],
                StationaryKernel(SE, 1.0, (5.0,)),
                nugget=1e-12,
                optimize=False,
            )
        self.assertTrue(model.ill_conditioned)
        self.assertTrue(
            any(
                issubclass(w.category, GaussianProcessConditioningWarning)
                for w in caught
            )
        )


class DerivativeGPTests(unittest.TestCase):
    """Derivative kernel blocks and joint ``(f, grad f)`` regression."""

    def test_blocks_match_finite_differences(self) -> None:
        """Gradient and Hessian blocks equal central differences of ``k``."""
        scales, variance = (0.8, 1.4), 1.7
        x, y = (0.3, 0.5), (0.9, -0.2)
        h = 1e-5
        for name in (SE, "matern52", "matern32"):

            def k(a: Sequence[float], b: Sequence[float], n: str = name) -> float:
                return derivative_kernel_blocks(n, a, b, scales, variance).value

            blocks = derivative_kernel_blocks(name, x, y, scales, variance)
            for i in range(2):
                xp, xm, yp, ym = list(x), list(x), list(y), list(y)
                xp[i] += h
                xm[i] -= h
                yp[i] += h
                ym[i] -= h
                self.assertAlmostEqual(
                    blocks.gradient_x[i], (k(xp, y) - k(xm, y)) / (2 * h), delta=1e-6
                )
                self.assertAlmostEqual(
                    blocks.gradient_y[i], (k(x, yp) - k(x, ym)) / (2 * h), delta=1e-6
                )
                for j in range(2):
                    total = 0.0
                    for sx in (1, -1):
                        for sy in (1, -1):
                            xx, yy = list(x), list(y)
                            xx[i] += sx * h
                            yy[j] += sy * h
                            total += sx * sy * k(xx, yy)
                    self.assertAlmostEqual(
                        blocks.hessian[i][j], total / (4 * h * h), delta=2e-5
                    )

    def test_squared_exponential_closed_form(self) -> None:
        """1-D SE: ``d2k/dxdy = k (1/l^2 - D^2/l^4)`` and the cross-block signs."""
        ell, s2, delta = 0.9, 1.3, 0.6
        blocks = derivative_kernel_blocks(SE, (delta,), (0.0,), (ell,), s2)
        k = s2 * math.exp(-0.5 * delta**2 / ell**2)
        self.assertAlmostEqual(
            blocks.hessian[0][0], k * (1 / ell**2 - delta**2 / ell**4), places=13
        )
        self.assertAlmostEqual(blocks.gradient_x[0], -k * delta / ell**2, places=13)
        self.assertAlmostEqual(blocks.gradient_y[0], k * delta / ell**2, places=13)

    def test_matern32_hessian_is_finite_at_coincident_points(self) -> None:
        """The limit ``3 sigma^2 / ell^2`` is used at ``Delta = 0``."""
        blocks = derivative_kernel_blocks("matern32", (0.2,), (0.2,), (0.5,), 2.0)
        self.assertAlmostEqual(blocks.hessian[0][0], 3.0 * 2.0 / 0.25, places=12)

    def test_gradient_data_reduce_error(self) -> None:
        """Exact gradients cut the error of ``f`` and ``grad f`` (same priors)."""
        f = lambda x: math.sin(3.0 * x) + 0.5 * x  # noqa: E731
        df = lambda x: 3.0 * math.cos(3.0 * x) + 0.5  # noqa: E731
        nodes = [(0.4 * i,) for i in range(6)]
        values = [f(p[0]) for p in nodes]
        common = {
            "kernel": "matern52",
            "value_noise": 1e-8,
            "gradient_noise": 1e-8,
            "optimize": False,
            "initial_signal_variance": 2.0,
            "initial_length_scales": (0.6,),
        }
        plain = fit_derivative_gp(nodes, values, **common)  # type: ignore[arg-type]
        enriched = fit_derivative_gp(
            nodes,
            values,
            nodes,
            [(df(p[0]),) for p in nodes],
            **common,  # type: ignore[arg-type]
        )
        grid = [0.05 + 0.1 * j for j in range(20)]

        def rms(model: object, quantity: str) -> float:
            total = 0.0
            for g in grid:
                if quantity == "f":
                    estimate = model.predict_with_variance((g,))[0]  # type: ignore[attr-defined]
                    total += (estimate - f(g)) ** 2
                else:
                    estimate = model.predict_gradient((g,)).mean[0]  # type: ignore[attr-defined]
                    total += (estimate - df(g)) ** 2
            return math.sqrt(total / len(grid))

        self.assertLess(rms(enriched, "f"), 0.25 * rms(plain, "f"))
        self.assertLess(rms(enriched, "g"), 0.25 * rms(plain, "g"))
        self.assertLess(
            enriched.predict_with_variance((0.7,))[1],
            plain.predict_with_variance((0.7,))[1],
        )
        self.assertLess(
            enriched.predict_gradient((0.7,)).variances[0],
            plain.predict_gradient((0.7,)).variances[0],
        )
        # fitted hyperparameters stay sensible with gradient data
        fitted = fit_derivative_gp(
            nodes,
            values,
            nodes,
            [(df(p[0]),) for p in nodes],
            kernel="matern52",
            value_noise=1e-6,
            gradient_noise=1e-6,
        )
        self.assertTrue(fitted.converged)
        self.assertLess(rms(fitted, "f"), 5e-3)
        self.assertLess(rms(fitted, "g"), 5e-2)

    def test_no_gradient_data_equals_plain_gp(self) -> None:
        """Without gradient data the joint model is the ordinary GP."""
        gen = random.Random(2)
        nodes = [(gen.uniform(0.0, 2.0), gen.uniform(0.0, 2.0)) for _ in range(8)]
        values = [math.sin(p[0]) * math.cos(p[1]) for p in nodes]
        derivative = fit_derivative_gp(
            nodes,
            values,
            kernel=SE,
            value_noise=1e-4,
            optimize=False,
            initial_signal_variance=0.7,
            initial_length_scales=(0.9, 1.1),
        )
        plain = fit_kernel_gp(
            nodes,
            values,
            StationaryKernel(SE, 0.7, (0.9, 1.1)),
            nugget=1e-4,
            optimize=False,
        )
        for query in ((0.5, 0.5), (1.5, 0.3)):
            a = derivative.predict_with_variance(query)
            b = plain.predict_with_variance(query)
            self.assertAlmostEqual(a[0], b[0], places=9)
            self.assertAlmostEqual(a[1], b[1], places=9)

    def test_plain_gp_gradient_prediction(self) -> None:
        """Gradient mean equals central differences and the joint-model route."""
        f = lambda x: math.sin(3.0 * x) + 0.5 * x  # noqa: E731
        nodes = [(0.25 * i,) for i in range(9)]
        model = fit_gaussian_process(
            nodes, [f(p[0]) for p in nodes], kernel=SE, nugget=1e-8
        )
        query = [(0.55,), (1.0,)]
        results = gaussian_process_gradient(model, query)
        h = 1e-5
        for point, result in zip(query, results, strict=True):
            numeric = (
                model.predict((point[0] + h,)) - model.predict((point[0] - h,))
            ) / (2 * h)
            self.assertAlmostEqual(result.mean[0], numeric, delta=1e-5)
            self.assertGreaterEqual(result.variances[0], 0.0)
        # independent route: joint model, same hyperparameters, no gradient data
        hyper = model.hyperparameters
        joint = fit_derivative_gp(
            nodes,
            [f(p[0]) for p in nodes],
            kernel=SE,
            value_noise=hyper.noise_variance,
            optimize=False,
            initial_signal_variance=hyper.signal_variance,
            initial_length_scales=hyper.length_scales,
        )
        other = joint.predict_gradient((0.55,))
        self.assertAlmostEqual(other.mean[0], results[0].mean[0], places=6)
        self.assertAlmostEqual(other.variances[0], results[0].variances[0], places=8)
        with self.assertRaises(TypeError):
            gaussian_process_gradient("model", query)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            gaussian_process_gradient(model, [(0.1, 0.2)])

    def test_failures(self) -> None:
        """Invalid data and singular joint matrices raise."""
        nodes = [(0.0,), (0.5,), (1.0,)]
        with self.assertRaises(ValueError):
            fit_derivative_gp(nodes, [0.0, 1.0, 0.0], [(0.0,)], [], kernel=SE)
        with self.assertRaises(ValueError):
            fit_derivative_gp(nodes, [0.0, 1.0, 0.0], kernel="rbf")
        with self.assertRaisesRegex(ValueError, "not positive definite"):
            fit_derivative_gp(
                nodes,
                [0.0, 1.0, 0.0],
                [(0.5,), (0.5,)],
                [(1.0,), (1.0,)],
                kernel=SE,
                value_noise=1e-30,
                gradient_noise=1e-30,
                optimize=False,
                initial_length_scales=(0.5,),
            )


class SparseGPTests(unittest.TestCase):
    """DTC, FITC, and VFE against the exact GP."""

    def setUp(self) -> None:
        """Create a noisy 1-D data set and the exact GP at fixed hyperparameters."""
        gen = random.Random(3)
        self.points = [(gen.uniform(0.0, 6.0),) for _ in range(40)]
        self.truth = lambda x: math.sin(x) + 0.3 * math.cos(3.0 * x)  # noqa: E731
        self.values = [
            self.truth(p[0]) + 0.1 * gen.gauss(0.0, 1.0) for p in self.points
        ]
        self.kernel = StationaryKernel(SE, 0.6, (0.8,))
        self.noise = 0.011
        self.exact = fit_kernel_gp(
            self.points,
            self.values,
            self.kernel + WhiteKernel(self.noise),
            nugget=1e-12,
            optimize=False,
        )

    def test_vfe_equals_exact_when_inducing_equal_training(self) -> None:
        """With ``Z = X`` all objectives equal the exact evidence and predictions."""
        for method in ("dtc", "fitc", "vfe"):
            sparse = fit_sparse_gp(
                self.points,
                self.values,
                self.kernel,
                method=method,
                inducing_points=self.points,
                noise_variance=self.noise,
                estimate_noise=False,
                optimize=False,
                nugget=1e-10,
            )
            self.assertAlmostEqual(
                sparse.objective,
                self.exact.log_marginal_likelihood,
                delta=1e-5,
                msg=method,
            )
            for g in (0.7, 2.9, 5.2):
                a = sparse.predict_with_variance((g,))
                b = self.exact.predict_with_variance((g,))
                self.assertAlmostEqual(a[0], b[0], delta=1e-5)
                self.assertAlmostEqual(a[1], b[1], delta=1e-5)
            self.assertGreater(sparse.inducing_condition, 1.0)

    def test_vfe_is_a_lower_bound(self) -> None:
        """For ``m < n`` the VFE bound is below the exact evidence."""
        inducing = select_inducing_points(self.points, 8, method="kmeans", seed=1)
        vfe = fit_sparse_gp(
            self.points,
            self.values,
            self.kernel,
            method="vfe",
            inducing_points=inducing,
            noise_variance=self.noise,
            estimate_noise=False,
            optimize=False,
        )
        self.assertLess(vfe.objective, self.exact.log_marginal_likelihood)
        self.assertGreater(vfe.trace_gap, 0.0)
        for g in (0.7, 2.9, 5.2):
            self.assertGreater(vfe.predict_with_variance((g,))[1], 0.0)

    def test_fitted_methods_track_the_exact_gp(self) -> None:
        """Optimized sparse GPs (m = 10) predict nearly as well as the exact GP."""
        grid = [0.3 * j for j in range(21)]
        exact = fit_kernel_gp(
            self.points,
            self.values,
            self.kernel + WhiteKernel(0.02),
            restarts=1,
            seed=1,
        )
        exact_rms = math.sqrt(
            sum((exact.predict((g,)) - self.truth(g)) ** 2 for g in grid) / 21
        )
        for method in ("dtc", "fitc", "vfe"):
            sparse = fit_sparse_gp(
                self.points,
                self.values,
                self.kernel,
                method=method,
                num_inducing=10,
                seed=1,
            )
            self.assertTrue(sparse.converged, f"{method}: {sparse.stop_reason}")
            rms = math.sqrt(
                sum((sparse.predict((g,)) - self.truth(g)) ** 2 for g in grid) / 21
            )
            self.assertLess(rms, exact_rms + 0.06, method)
            self.assertAlmostEqual(sparse.noise_variance, 0.01, delta=0.008, msg=method)

    def test_inducing_optimization_does_not_lower_the_bound(self) -> None:
        """Optimizing ``Z`` from a k-means start never decreases the VFE bound."""
        small = list(zip(self.points[:24], self.values[:24], strict=True))
        pts = [p for p, _ in small]
        vals = [v for _, v in small]
        fixed = fit_sparse_gp(
            pts,
            vals,
            self.kernel,
            num_inducing=4,
            optimize=False,
            noise_variance=self.noise,
            estimate_noise=False,
            seed=2,
        )
        moved = fit_sparse_gp(
            pts,
            vals,
            self.kernel,
            num_inducing=4,
            optimize=True,
            optimize_inducing=True,
            noise_variance=self.noise,
            estimate_noise=False,
            max_iterations=15,
            seed=2,
        )
        self.assertGreaterEqual(moved.objective, fixed.objective - 1e-9)
        self.assertNotEqual(moved.inducing_points, fixed.inducing_points)
        low, high = min(p[0] for p in pts), max(p[0] for p in pts)
        for z in moved.inducing_points:
            self.assertTrue(low - 1e-9 <= z[0] <= high + 1e-9)

    def test_cost_scaling_and_kernel_evaluation_counts(self) -> None:
        """Kernel evaluations match the cost model, which scales as n m^2."""
        calls = [0]
        counting = CountingKernel(self.kernel, calls)
        n, m = len(self.points), 6
        z = select_inducing_points(self.points, m, method="subset", seed=4)
        centered = [v - sum(self.values) / n for v in self.values]
        _sparse_terms("vfe", counting, self.points, centered, z, self.noise, 1e-8)
        cost = sparse_gp_cost(n, m, 1)
        self.assertEqual(calls[0], cost.sparse_kernel_evaluations)
        calls[0] = 0
        counting.matrix(self.points)
        self.assertEqual(calls[0], cost.exact_kernel_evaluations)
        big = sparse_gp_cost(2000, 40, 3)
        double = sparse_gp_cost(4000, 40, 3)
        self.assertAlmostEqual(double.sparse_flops / big.sparse_flops, 2.0, delta=0.1)
        self.assertAlmostEqual(double.exact_flops / big.exact_flops, 8.0, delta=0.5)
        self.assertGreater(big.speedup, 100.0)
        with self.assertRaises(ValueError):
            sparse_gp_cost(10, 11)

    def test_inducing_selection(self) -> None:
        """Selection is seeded, in range, and validated."""
        a = select_inducing_points(self.points, 5, method="kmeans", seed=7)
        b = select_inducing_points(self.points, 5, method="kmeans", seed=7)
        self.assertEqual(a, b)
        self.assertEqual(len(set(a)), 5)
        subset = select_inducing_points(self.points, 5, method="subset", seed=7)
        self.assertTrue(all(p in self.points for p in subset))
        self.assertEqual(
            select_inducing_points(self.points, len(self.points)), tuple(self.points)
        )
        with self.assertRaises(ValueError):
            select_inducing_points(self.points, 41)
        with self.assertRaises(ValueError):
            select_inducing_points(self.points, 3, method="grid")
        with self.assertRaises(ValueError):
            select_inducing_points([(1.0,)] * 4 + [(2.0,)], 3)

    def test_failures(self) -> None:
        """Singular inducing covariance is reported; invalid options raise."""
        duplicated = [(1.0,), (1.0,), (3.0,)]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                tiny = fit_sparse_gp(
                    self.points,
                    self.values,
                    self.kernel,
                    inducing_points=duplicated,
                    optimize=False,
                    nugget=1e-30,
                )
            except ValueError as error:
                self.assertIn("no jitter", str(error))
            else:
                # rounding left a positive pivot: the singularity must be reported
                self.assertGreater(tiny.inducing_condition, 1e12)
                self.assertTrue(
                    any(
                        issubclass(w.category, GaussianProcessConditioningWarning)
                        for w in caught
                    )
                )
        with self.assertRaises(ValueError):
            fit_sparse_gp(
                self.points, self.values, self.kernel + WhiteKernel(0.1), num_inducing=4
            )
        with self.assertRaises(ValueError):
            fit_sparse_gp(
                self.points, self.values, self.kernel, method="svgp", num_inducing=4
            )
        with self.assertRaises(ValueError):
            fit_sparse_gp(self.points, self.values, self.kernel)
        with self.assertRaises(ValueError):
            fit_sparse_gp(
                self.points, self.values, self.kernel, inducing_points=[(0.0, 1.0)]
            )


class HeteroscedasticGPTests(unittest.TestCase):
    """Most-likely heteroscedastic GP recovers an input-dependent noise level."""

    def test_noise_level_recovery(self) -> None:
        """Estimated noise rises with the true standard deviation ``0.03 + 0.4 x``."""
        gen = random.Random(5)
        n = 50
        points = [((i + 0.5) / n,) for i in range(n)]
        sd = lambda x: 0.03 + 0.40 * x  # noqa: E731
        truth = lambda x: math.sin(2.0 * math.pi * x)  # noqa: E731
        values = [truth(p[0]) + sd(p[0]) * gen.gauss(0.0, 1.0) for p in points]
        kernel = StationaryKernel(SE, 1.0, (0.3,))
        model = fit_heteroscedastic_gp(points, values, kernel, seed=1)
        self.assertIn(
            model.stop_reason, {"converged", "likelihood_decreased", "max_iterations"}
        )
        self.assertEqual(model.iterations, len(model.log_likelihood_history))
        self.assertEqual(model.converged, model.stop_reason != "max_iterations")
        # the retained iterate is the best-likelihood one
        self.assertAlmostEqual(
            model.mean_model.log_marginal_likelihood,
            max(model.log_likelihood_history),
            places=9,
        )
        low = [model.noise_variance((x,)) for x in (0.05, 0.15, 0.25)]
        high = [model.noise_variance((x,)) for x in (0.75, 0.85, 0.95)]
        self.assertGreater(sum(high) / sum(low), 3.0)
        self.assertTrue(all(v > 0.0 for v in low + high))
        # rank correlation of the estimate with the true variance over a grid
        grid = [0.05 * j + 0.025 for j in range(20)]
        est = [model.noise_variance((g,)) for g in grid]
        tru = [sd(g) ** 2 for g in grid]
        corr = _pearson([math.log(v) for v in est], [math.log(v) for v in tru])
        self.assertGreater(corr, 0.8)
        # predictive variance includes the noise level
        mean, latent = model.predict_with_variance((0.9,))
        self.assertAlmostEqual(
            model.predict_with_variance((0.9,), include_noise=True)[1],
            latent + model.noise_variance((0.9,)),
            places=12,
        )
        self.assertAlmostEqual(mean, truth(0.9), delta=0.5)

    def test_failures(self) -> None:
        """White kernels, tiny data sets and bad tolerances are rejected."""
        points = [(0.1 * i,) for i in range(8)]
        values = [math.sin(p[0]) for p in points]
        kernel = StationaryKernel(SE, 1.0, (0.5,))
        with self.assertRaises(ValueError):
            fit_heteroscedastic_gp(points, values, kernel + WhiteKernel(0.1))
        with self.assertRaises(ValueError):
            fit_heteroscedastic_gp(points[:3], values[:3], kernel)
        with self.assertRaises(ValueError):
            fit_heteroscedastic_gp(points, values, kernel, tolerance=0.0)
        with self.assertRaises(TypeError):
            fit_heteroscedastic_gp(points, values, "se")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            fit_heteroscedastic_gp(points, [1.0] * 8, kernel)


def _pearson(a: Sequence[float], b: Sequence[float]) -> float:
    """Return the Pearson correlation of two samples."""
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    cov = sum((x - ma) * (y - mb) for x, y in zip(a, b, strict=True))
    return cov / math.sqrt(
        sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b)
    )


class MultiOutputGPTests(unittest.TestCase):
    """Linear model of coregionalization on heterotopic data."""

    def test_gradient_is_exact(self) -> None:
        """The analytic gradient (kernel, W, kappa, noise) matches differences."""
        gen = random.Random(2)
        outputs, rank = 2, 2
        inputs = [
            [(0.1 * i,) for i in range(6)],
            [(0.05 + 0.13 * i,) for i in range(7)],
        ]
        labels = [0] * 6 + [1] * 7
        points = inputs[0] + inputs[1]
        data = [gen.gauss(0.0, 1.0) for _ in points]
        kernels = [
            StationaryKernel(SE, 1.0, (0.4,)),
            StationaryKernel("matern52", 0.5, (0.9,)),
        ]
        theta: list[float] = []
        for kernel in kernels:
            theta += list(kernel.log_parameters)
            theta += [gen.gauss(0.0, 0.7) for _ in range(outputs * rank)]
            theta += [math.log(0.2), math.log(0.3)]
        theta += [math.log(0.05), math.log(0.08)]
        args = (kernels, rank, outputs, labels, points, data, 1e-8, True)
        _, grad, _, _, _ = _lmc_marginal(theta, *args, gradient=True)
        for j in range(len(theta)):
            h = 1e-6
            up, down = list(theta), list(theta)
            up[j] += h
            down[j] -= h
            numeric = (
                _lmc_marginal(up, *args, gradient=False)[0]
                - _lmc_marginal(down, *args, gradient=False)[0]
            ) / (2 * h)
            self.assertAlmostEqual(
                grad[j], numeric, delta=1e-6 * (1.0 + abs(numeric)), msg=str(j)
            )

    def _data(
        self, sign: float
    ) -> tuple[list[list[Vector]], list[list[float]], object]:
        """Return a gap-prone output 1 and a dense correlated output 2."""
        gen = random.Random(11)
        f = lambda x: math.sin(5.0 * x) + 0.5 * x  # noqa: E731
        x1 = [(x,) for x in (0.0, 0.08, 0.16, 0.24, 0.32, 0.9, 1.0)]
        x2 = [(i / 24.0,) for i in range(25)]
        y1 = [f(p[0]) + 0.02 * gen.gauss(0.0, 1.0) for p in x1]
        y2 = [sign * 1.5 * f(p[0]) + 0.05 + 0.02 * gen.gauss(0.0, 1.0) for p in x2]
        return [x1, x2], [y1, y2], f

    def test_correlated_output_improves_prediction(self) -> None:
        """A dense correlated output 2 reduces the error of output 1 in its data gap."""
        inputs, outputs, f = self._data(1.0)
        kernel = StationaryKernel(SE, 1.0, (0.3,))
        joint = fit_multi_output_gp(inputs, outputs, kernel, seed=1)
        # W and the kernel variance trade off exactly, so the optimizer may stall on a
        # flat ridge; the report must then say so rather than claim convergence.
        self.assertTrue(
            joint.converged or "line search" in joint.stop_reason, joint.stop_reason
        )
        alone = fit_kernel_gp(
            inputs[0], outputs[0], kernel + WhiteKernel(0.01), restarts=2, seed=1
        )
        grid = [0.35 + 0.05 * j for j in range(11)]
        joint_rms = math.sqrt(
            sum((joint.predict_with_variance(0, (g,))[0] - f(g)) ** 2 for g in grid)
            / len(grid)  # type: ignore[operator]
        )
        alone_rms = math.sqrt(
            sum((alone.predict((g,)) - f(g)) ** 2 for g in grid) / len(grid)
        )  # type: ignore[operator]
        self.assertLess(joint_rms, 0.5 * alone_rms)
        self.assertGreater(joint.output_correlation(0, 1), 0.95)
        self.assertEqual(joint.output_correlation(0, 0), 1.0)
        (b,) = joint.coregionalization_matrices()
        self.assertGreater(b[0][0] * b[1][1] - b[0][1] ** 2, -1e-9)
        self.assertGreater(
            joint.predict_with_variance(1, (0.5,), include_noise=True)[1], 0.0
        )

    def test_negative_correlation_is_learned(self) -> None:
        """A reflected second output gives a strongly negative learned correlation."""
        inputs, outputs, f = self._data(-1.0)
        joint = fit_multi_output_gp(
            inputs, outputs, StationaryKernel(SE, 1.0, (0.3,)), seed=1, restarts=2
        )
        self.assertLess(joint.output_correlation(0, 1), -0.9)
        gap = 0.5
        self.assertAlmostEqual(
            joint.predict_with_variance(0, (gap,))[0], f(gap), delta=0.1
        )  # type: ignore[operator]

    def test_lmc_with_two_kernels_runs(self) -> None:
        """Two latent kernels (LMC, rank 1) fit and report a finite evidence."""
        inputs, outputs, _ = self._data(1.0)
        model = fit_multi_output_gp(
            [inputs[0], inputs[1][::2]],
            [outputs[0], outputs[1][::2]],
            [
                StationaryKernel(SE, 1.0, (0.3,)),
                StationaryKernel("matern52", 0.3, (1.0,)),
            ],
            seed=1,
            restarts=1,
            max_iterations=40,
        )
        self.assertEqual(len(model.coregionalization_matrices()), 2)
        self.assertTrue(math.isfinite(model.log_marginal_likelihood))
        self.assertGreater(model.condition_number, 1.0)

    def test_failures(self) -> None:
        """Constant outputs, white kernels and index errors raise."""
        inputs, outputs, _ = self._data(1.0)
        kernel = StationaryKernel(SE, 1.0, (0.3,))
        with self.assertRaises(ValueError):
            fit_multi_output_gp(inputs, [outputs[0], [1.0] * len(outputs[1])], kernel)
        with self.assertRaises(ValueError):
            fit_multi_output_gp(inputs, outputs, kernel + WhiteKernel(0.1))
        with self.assertRaises(ValueError):
            fit_multi_output_gp(inputs, outputs[:1], kernel)
        with self.assertRaises(TypeError):
            fit_multi_output_gp(inputs, outputs, [])
        with self.assertRaises(ValueError):
            fit_multi_output_gp(inputs, outputs, kernel, rank=0)
        model = fit_multi_output_gp(
            inputs, outputs, kernel, seed=1, restarts=1, max_iterations=5
        )
        with self.assertRaises(ValueError):
            model.predict_with_variance(2, (0.5,))
        with self.assertRaises(ValueError):
            model.output_correlation(0, 2)


class OperatorGPTests(unittest.TestCase):
    """Linear-operator-constrained GP: covariances and a known PDE solution."""

    def test_covariances_closed_form(self) -> None:
        """``d/dx`` and ``d2/dx2 - c`` covariances equal the Hermite closed forms."""
        ell, s2, delta, c = 0.9, 1.4, 0.7, 1.3
        k = s2 * math.exp(-0.5 * delta**2 / ell**2)
        d1 = operator_covariances(
            LinearDifferentialOperator.derivative(), delta, ell, s2
        )
        self.assertAlmostEqual(d1[0], k * (1 / ell**2 - delta**2 / ell**4), places=13)
        self.assertAlmostEqual(d1[1], -k * delta / ell**2, places=13)
        self.assertAlmostEqual(d1[2], k * delta / ell**2, places=13)
        d2 = k * (delta**2 / ell**4 - 1 / ell**2)
        d4 = k * (3 / ell**4 - 6 * delta**2 / ell**6 + delta**4 / ell**8)
        h = operator_covariances(
            LinearDifferentialOperator.helmholtz(c), delta, ell, s2
        )
        self.assertAlmostEqual(h[0], d4 - 2 * c * d2 + c * c * k, places=12)
        self.assertAlmostEqual(h[1], d2 - c * k, places=13)
        self.assertAlmostEqual(h[2], d2 - c * k, places=13)

    def test_covariances_match_finite_differences(self) -> None:
        """Cross-covariances of ``L u`` equal differences of ``k`` directly."""
        ell, s2, c = 0.9, 1.4, 1.3
        k = lambda d: s2 * math.exp(-0.5 * d * d / (ell * ell))  # noqa: E731
        d2 = lambda d, h=1e-3: (k(d + h) - 2 * k(d) + k(d - h)) / (h * h)  # noqa: E731

        def d4(d: float, h: float = 2e-2) -> float:
            stencil = k(d + 2 * h) - 4 * k(d + h) + 6 * k(d) - 4 * k(d - h)
            return (stencil + k(d - 2 * h)) / h**4

        for delta in (0.0, 0.4, 1.1):
            ll, lu, ul = operator_covariances(
                LinearDifferentialOperator.helmholtz(c), delta, ell, s2
            )
            self.assertAlmostEqual(
                ll,
                d4(delta) - 2 * c * d2(delta) + c * c * k(delta),
                delta=1e-3 * abs(ll),
            )
            self.assertAlmostEqual(lu, d2(delta) - c * k(delta), delta=1e-5)
            self.assertAlmostEqual(ul, lu, places=12)

    def test_recovers_known_solution_from_source_term(self) -> None:
        """``u'' - u = -2 sin x`` with ``u(0) = u(pi) = 0`` has solution ``sin x``."""
        gen = random.Random(4)
        operator = LinearDifferentialOperator.helmholtz(1.0)
        xs = [math.pi * (i + 0.5) / 12.0 for i in range(12)]
        ys = [-2.0 * math.sin(x) + 0.02 * gen.gauss(0.0, 1.0) for x in xs]
        model = fit_operator_gp(
            operator,
            xs,
            ys,
            value_points=[0.0, math.pi],
            values=[0.0, 0.0],
            operator_noise=4e-4,
        )
        grid = [math.pi * j / 20.0 for j in range(21)]
        error = max(abs(model.predict_with_variance(x)[0] - math.sin(x)) for x in grid)
        self.assertLess(error, 0.02)
        source = max(
            abs(model.predict_operator_with_variance(x)[0] + 2.0 * math.sin(x))
            for x in grid
        )
        self.assertLess(source, 0.05)
        for x in (0.5, 1.5, 2.5):
            self.assertLess(model.predict_with_variance(x)[1], 1e-3)
        self.assertGreater(model.condition_number, 1.0)
        fitted = fit_operator_gp(
            operator,
            xs,
            ys,
            value_points=[0.0, math.pi],
            values=[0.0, 0.0],
            operator_noise=4e-4,
            optimize=True,
        )
        self.assertGreaterEqual(
            fitted.log_marginal_likelihood, model.log_marginal_likelihood - 1e-6
        )
        self.assertLess(
            max(abs(fitted.predict_with_variance(x)[0] - math.sin(x)) for x in grid),
            0.02,
        )

    def test_derivative_operator_identifiability(self) -> None:
        """Observing only ``u' = 2x`` leaves the constant free; an anchor fixes it."""
        gen = random.Random(4)
        xs = [i / 10.0 for i in range(11)]
        ys = [2.0 * x + 0.01 * gen.gauss(0.0, 1.0) for x in xs]
        operator = LinearDifferentialOperator.derivative()
        free = fit_operator_gp(operator, xs, ys, operator_noise=1e-4, length_scale=0.5)
        anchored = fit_operator_gp(
            operator,
            xs,
            ys,
            value_points=[0.0],
            values=[0.0],
            operator_noise=1e-4,
            length_scale=0.5,
        )
        self.assertGreater(free.predict_with_variance(0.5)[1], 0.1)
        self.assertLess(anchored.predict_with_variance(0.5)[1], 1e-3)
        self.assertAlmostEqual(anchored.predict_with_variance(0.5)[0], 0.25, delta=0.02)
        self.assertAlmostEqual(
            free.predict_operator_with_variance(0.5)[0], 1.0, delta=0.05
        )
        self.assertAlmostEqual(
            anchored.predict_with_variance(0.5)[0]
            - anchored.predict_with_variance(0.0)[0],
            free.predict_with_variance(0.5)[0] - free.predict_with_variance(0.0)[0],
            delta=0.02,
        )

    def test_apply_and_failures(self) -> None:
        """Operator helpers validate; duplicated data with tiny noise is reported."""
        operator = LinearDifferentialOperator.helmholtz(2.0)
        self.assertAlmostEqual(operator.apply((1.0, 5.0, 3.0)), 3.0 - 2.0)
        with self.assertRaises(ValueError):
            operator.apply((1.0,))
        with self.assertRaises(ValueError):
            LinearDifferentialOperator((0.0, 0.0))
        with self.assertRaises(ValueError):
            LinearDifferentialOperator(())
        with self.assertRaises(TypeError):
            fit_operator_gp("d/dx", [0.0], [1.0])  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            fit_operator_gp(operator, [], [])
        with self.assertRaises(ValueError):
            fit_operator_gp(operator, [0.0, 1.0], [1.0])
        with self.assertRaises(ValueError):
            fit_operator_gp(operator, [0.0], [1.0], operator_noise=0.0)
        with self.assertRaisesRegex(ValueError, "not positive definite"):
            fit_operator_gp(
                LinearDifferentialOperator.derivative(),
                [0.5, 0.5],
                [1.0, 1.0],
                operator_noise=1e-300,
                length_scale=1.0,
            )


if __name__ == "__main__":
    unittest.main()
