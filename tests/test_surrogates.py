"""Tests for polynomial chaos, Gaussian-process surrogates, and the adapter."""

from __future__ import annotations

import math
import unittest

from probstat.distributions import LogNormal, Normal, Uniform
from probstat.linalg import symmetric_eigen
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput, gauss_hermite
from probstat.optimization import central_gradient
from probstat.propagation import propagate
from probstat.sampling import monte_carlo_design
from probstat.sensitivity import estimate_sobol_indices
from probstat.surrogates import (
    SurrogateModel,
    fit_gaussian_process,
    fit_gaussian_process_to_model,
    fit_polynomial_chaos,
    gaussian_process_log_marginal_likelihood,
    hermite_orthonormal,
    legendre_orthonormal,
    multi_indices,
    normalized_information_matrix,
    polynomial_chaos_from_samples,
    surrogate_probabilistic_model,
)
from probstat.uq import QuantityOfInterest

ISHIGAMI_A = 7.0
ISHIGAMI_B = 0.1


def _ishigami() -> ProbabilisticModel:
    """Return the Ishigami benchmark on independent uniform inputs."""
    inputs = JointInput.independent(
        *(RandomInput(f"x{i}", Uniform(-math.pi, math.pi)) for i in (1, 2, 3))
    )

    def model(p):
        """Ishigami function."""
        return {
            "y": math.sin(p["x1"])
            + ISHIGAMI_A * math.sin(p["x2"]) ** 2
            + ISHIGAMI_B * p["x3"] ** 4 * math.sin(p["x1"])
        }

    return ProbabilisticModel(
        inputs, model, (QuantityOfInterest("y", "Ishigami output"),), name="ishigami"
    )


def _ishigami_analytic() -> dict[str, float]:
    """Return analytic Ishigami mean, variance, and Sobol indices."""
    a, b, pi = ISHIGAMI_A, ISHIGAMI_B, math.pi
    v1 = 0.5 * (1.0 + b * pi**4 / 5.0) ** 2
    v2 = a * a / 8.0
    v13 = 8.0 * b * b * pi**8 / 225.0
    total = v1 + v2 + v13
    return {
        "mean": a / 2.0,
        "variance": total,
        "S1": v1 / total,
        "S2": v2 / total,
        "S3": 0.0,
        "ST1": (v1 + v13) / total,
        "ST2": v2 / total,
        "ST3": v13 / total,
        "S13": v13 / total,
    }


class PolynomialBasisTests(unittest.TestCase):
    """Multi-index sets and orthonormal univariate families."""

    def test_total_degree_and_hyperbolic_cardinality(self) -> None:
        """The total-degree set has ``C(d+p, p)`` terms; ``q < 1`` prunes it."""
        full = multi_indices(3, 5)
        self.assertEqual(len(full), math.comb(8, 5))
        self.assertEqual(full[0], (0, 0, 0))
        self.assertEqual(len(set(full)), len(full))
        pruned = multi_indices(3, 5, q_norm=0.5)
        self.assertLess(len(pruned), len(full))
        self.assertTrue(set(pruned) <= set(full))
        self.assertIn((5, 0, 0), pruned)
        self.assertNotIn((2, 2, 1), pruned)
        with self.assertRaises(ValueError):
            multi_indices(2, 3, q_norm=1.5)

    def test_hermite_family_is_orthonormal_under_standard_normal(self) -> None:
        """Gauss-Hermite quadrature confirms ``E[psi_m psi_n] = delta_mn``."""
        nodes, weights = gauss_hermite(20)
        tables = [hermite_orthonormal(z, 8) for z in nodes]
        for m in range(9):
            for n in range(9):
                value = math.fsum(
                    w * t[m] * t[n] for w, t in zip(weights, tables, strict=True)
                )
                self.assertAlmostEqual(value, 1.0 if m == n else 0.0, 10)

    def test_legendre_family_is_orthonormal_under_uniform(self) -> None:
        """A Golub-Welsch Gauss-Legendre rule confirms uniform orthonormality."""
        order = 12
        jacobi = tuple(
            tuple(
                max(i, j) / math.sqrt(4.0 * max(i, j) ** 2 - 1.0)
                if abs(i - j) == 1
                else 0.0
                for j in range(order)
            )
            for i in range(order)
        )
        nodes, vectors = symmetric_eigen(jacobi)
        weights = [vectors[0][k] ** 2 for k in range(order)]
        tables = [legendre_orthonormal(x, 8) for x in nodes]
        for m in range(9):
            for n in range(9):
                value = math.fsum(
                    w * t[m] * t[n] for w, t in zip(weights, tables, strict=True)
                )
                self.assertAlmostEqual(value, 1.0 if m == n else 0.0, 10)


class PolynomialChaosTests(unittest.TestCase):
    """Regression PCE accuracy, diagnostics, and Sobol post-processing."""

    def test_exact_polynomial_is_reproduced_to_machine_precision(self) -> None:
        """A cubic in Gaussian inputs is represented exactly at degree three."""
        mu1, s1, mu2, s2 = 1.0, 0.5, -2.0, 1.5
        inputs = JointInput.independent(
            RandomInput("a", Normal(mu1, s1)), RandomInput("b", Normal(mu2, s2))
        )

        def model(p):
            """Cubic polynomial model."""
            a, b = p["a"], p["b"]
            return {"y": 1.0 + 2.0 * a - a * b * b + 0.5 * b**3}

        probabilistic = ProbabilisticModel(
            inputs, model, (QuantityOfInterest("y", "cubic"),)
        )
        pce = fit_polynomial_chaos(probabilistic, degree=3, seed=2)
        self.assertLess(pce.empirical_error, 1e-24)
        self.assertLess(pce.loo_error, 1e-22)
        for point in inputs.sample(20, seed=7):
            expected = model({"a": point[0], "b": point[1]})["y"]
            self.assertAlmostEqual(pce.predict(point), expected, 9)
        mean = (
            1.0 + 2.0 * mu1 - mu1 * (s2**2 + mu2**2) + 0.5 * (mu2**3 + 3 * mu2 * s2**2)
        )
        self.assertAlmostEqual(pce.mean, mean, 9)
        self.assertEqual(probabilistic.evaluations, 2 * len(multi_indices(2, 3)))

    def test_legendre_basis_reproduces_polynomial_of_uniform_inputs(self) -> None:
        """Legendre PCE recovers mean and variance of a uniform-input quadratic."""
        inputs = JointInput.independent(
            RandomInput("u", Uniform(0.0, 2.0)), RandomInput("v", Uniform(-1.0, 3.0))
        )
        points = inputs.sample(40, seed=1)
        values = [u * u + u * v for u, v in points]
        pce = polynomial_chaos_from_samples(
            inputs, points, values, degree=2, basis="legendre"
        )
        # E[U^2] = 4/3, E[U] = 1, E[V] = 1 -> mean = 4/3 + 1.
        self.assertAlmostEqual(pce.mean, 4.0 / 3.0 + 1.0, 10)
        # Var(U^2 + UV) by exact moments of the independent uniforms.
        eu = {1: 1.0, 2: 4.0 / 3.0, 3: 2.0, 4: 16.0 / 5.0}
        ev = {1: 1.0, 2: 7.0 / 3.0}
        second = eu[4] + 2.0 * eu[3] * ev[1] + eu[2] * ev[2]
        self.assertAlmostEqual(pce.variance, second - pce.mean**2, 9)
        information = normalized_information_matrix(pce)
        self.assertAlmostEqual(information[0][0], 1.0, 12)

    def test_ishigami_sobol_indices_match_analytic_values(self) -> None:
        """Coefficient-based Sobol indices reproduce the Ishigami benchmark."""
        model = _ishigami()
        pce = fit_polynomial_chaos(
            model, degree=10, basis="legendre", q_norm=0.75, seed=1
        )
        exact = _ishigami_analytic()
        indices = pce.sobol_indices()
        tolerance = 0.01
        for i in (1, 2, 3):
            name = f"x{i}"
            self.assertLess(abs(indices.first_order[name] - exact[f"S{i}"]), tolerance)
            self.assertLess(abs(indices.total_order[name] - exact[f"ST{i}"]), tolerance)
        self.assertLess(
            abs(indices.second_order[("x1", "x3")] - exact["S13"]), tolerance
        )
        self.assertLess(abs(indices.second_order[("x1", "x2")]), tolerance)
        self.assertLess(abs(pce.mean - exact["mean"]) / exact["mean"], 0.01)
        self.assertLess(abs(pce.variance - exact["variance"]) / exact["variance"], 0.01)
        self.assertLess(pce.loo_error, 0.01)

    def test_leave_one_out_flags_an_underfitted_expansion(self) -> None:
        """A low-degree expansion of Ishigami has a large LOO error."""
        model = _ishigami()
        low = fit_polynomial_chaos(model, degree=2, basis="legendre", seed=3)
        self.assertGreater(low.loo_error, 0.2)
        self.assertGreaterEqual(low.loo_error, low.empirical_error)

    def test_hermite_basis_handles_correlated_non_gaussian_inputs(self) -> None:
        """Nataf-space Hermite PCE recovers moments of correlated lognormals."""
        first = RandomInput("p", LogNormal(0.0, 0.25))
        second = RandomInput("q", LogNormal(0.5, 0.2))
        inputs = JointInput.from_pearson((first, second), ((1.0, 0.5), (0.5, 1.0)))

        def model(p):
            """Sum of the two inputs."""
            return {"s": p["p"] + p["q"]}

        probabilistic = ProbabilisticModel(
            inputs, model, (QuantityOfInterest("s", "sum"),)
        )
        pce = fit_polynomial_chaos(probabilistic, degree=4, seed=5, size=120)
        means = inputs.means()
        sds = inputs.standard_deviations()
        variance = sds[0] ** 2 + sds[1] ** 2 + 2.0 * 0.5 * sds[0] * sds[1]
        self.assertAlmostEqual(pce.mean, means[0] + means[1], 4)
        self.assertLess(abs(pce.variance / variance - 1.0), 2e-3)
        with self.assertRaises(ValueError):
            pce.sobol_indices()
        with self.assertRaises(ValueError):
            fit_polynomial_chaos(probabilistic, degree=2, basis="legendre")

    def test_too_small_design_is_rejected(self) -> None:
        """Least squares needs more samples than basis terms."""
        model = _ishigami()
        with self.assertRaises(ValueError):
            fit_polynomial_chaos(model, degree=3, basis="legendre", size=20)


class GaussianProcessTests(unittest.TestCase):
    """Gaussian-process regression and marginal-likelihood fitting."""

    POINTS = tuple((6.0 * i / 11.0,) for i in range(12))

    @staticmethod
    def _target(x: float) -> float:
        """Return the smooth one-dimensional test function."""
        return math.sin(x) + 0.3 * x

    def test_gp_interpolates_and_recovers_a_smooth_function(self) -> None:
        """A small nugget gives interpolation and accurate off-design predictions."""
        values = [self._target(x[0]) for x in self.POINTS]
        for kernel in ("squared_exponential", "matern52"):
            gp = fit_gaussian_process(
                self.POINTS, values, kernel=kernel, nugget=1e-10, seed=4
            )
            for x, y in zip(self.POINTS, values, strict=True):
                mean, variance = gp.predict_with_variance(x)
                self.assertAlmostEqual(mean, y, 5)
                self.assertLess(variance, 1e-8)
            for k in range(11):
                x = 0.25 + 0.5 * k
                self.assertLess(abs(gp.predict((x,)) - self._target(x)), 0.01)
            _, far = gp.predict_with_variance((12.0,))
            _, near = gp.predict_with_variance((3.1,))
            self.assertGreater(far, near)
            self.assertLess(gp.leave_one_out().relative_error, 0.01)

    def test_marginal_likelihood_gradient_and_optimization(self) -> None:
        """The analytic gradient matches finite differences and BFGS improves."""
        values = [self._target(x[0]) for x in self.POINTS]
        mean = sum(values) / len(values)
        sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
        targets = [(v - mean) / sd for v in values]
        for kernel in ("squared_exponential", "matern52"):
            theta = (0.3, math.log(1.3), math.log(0.01))

            def objective(t, kernel=kernel):
                """Return the log marginal likelihood at ``t``."""
                return gaussian_process_log_marginal_likelihood(
                    self.POINTS,
                    targets,
                    t,
                    kernel=kernel,
                    nugget=1e-6,
                    estimate_noise=True,
                )[0]

            _, analytic = gaussian_process_log_marginal_likelihood(
                self.POINTS,
                targets,
                theta,
                kernel=kernel,
                nugget=1e-6,
                estimate_noise=True,
            )
            numeric, _ = central_gradient(objective, theta)
            for a, b in zip(analytic, numeric, strict=True):
                self.assertAlmostEqual(a, b, 5)
            # With the default nugget the squared-exponential kernel matrix has
            # condition number ~5e12, so a stationary point cannot be verified
            # (the fit warns); 1e-6 keeps the convergence check well posed.
            start = fit_gaussian_process(
                self.POINTS, values, kernel=kernel, nugget=1e-6, optimize=False
            )
            fitted = fit_gaussian_process(
                self.POINTS, values, kernel=kernel, nugget=1e-6, seed=1
            )
            self.assertGreater(
                fitted.log_marginal_likelihood, start.log_marginal_likelihood
            )
            self.assertTrue(fitted.converged)

    def test_gp_surrogate_of_a_probabilistic_model(self) -> None:
        """A GP trained on an LHS design emulates a smooth two-input model."""
        inputs = JointInput.independent(
            RandomInput("a", Normal(0.0, 1.0)), RandomInput("b", Uniform(0.0, 2.0))
        )

        def model(p):
            """Smooth two-input response."""
            return {"y": math.exp(0.3 * p["a"]) + p["b"] ** 2}

        probabilistic = ProbabilisticModel(
            inputs, model, (QuantityOfInterest("y", "response"),)
        )
        gp = fit_gaussian_process_to_model(
            probabilistic, size=20, seed=3, kernel="squared_exponential", restarts=2
        )
        self.assertEqual(probabilistic.evaluations, 20)
        errors = [
            abs(gp.predict(x) - model({"a": x[0], "b": x[1]})["y"])
            for x in inputs.sample(30, seed=9)
            if abs(x[0]) < 2.0
        ]
        self.assertLess(max(errors), 0.1)
        with self.assertRaises(ValueError):
            fit_gaussian_process(self.POINTS, [1.0] * len(self.POINTS))
        with self.assertRaises(ValueError):
            fit_gaussian_process(self.POINTS, [0.0] * 11 + [1.0], nugget=0.0)


class SurrogateAdapterTests(unittest.TestCase):
    """Surrogates drop into the central probabilistic-model workflows."""

    def test_pce_surrogate_runs_through_propagation_and_sensitivity(self) -> None:
        """Propagation and pick-freeze Sobol accept the surrogate model unchanged."""
        model = _ishigami()
        pce = fit_polynomial_chaos(model, degree=6, basis="legendre", seed=1)
        before = model.evaluations
        emulator = surrogate_probabilistic_model(model, {"y": pce})
        self.assertIsInstance(emulator.model, SurrogateModel)
        self.assertEqual(emulator.inputs, model.inputs)
        study = propagate(emulator, 1024, design="sobol", seed=4)
        self.assertEqual(model.evaluations, before)
        self.assertEqual(study.model_evaluations, 1024)
        self.assertLess(abs(study["y"].mean - pce.mean), 0.1)
        self.assertLess(
            abs(study["y"].standard_deviation ** 2 / pce.variance - 1.0), 0.05
        )
        sample_a = monte_carlo_design(600, 3, seed=1).points
        sample_b = monte_carlo_design(600, 3, seed=2).points
        result = estimate_sobol_indices(emulator.unit_function("y"), sample_a, sample_b)
        analytic = pce.sobol_indices()
        self.assertLess(abs(result.total_effect[1] - analytic.total_order["x2"]), 0.08)
        self.assertLess(abs(result.first_order[0] - analytic.first_order["x1"]), 0.12)

    def test_adapter_validates_its_inputs(self) -> None:
        """Unknown quantities and non-surrogates are rejected."""
        model = _ishigami()
        with self.assertRaises(KeyError):
            surrogate_probabilistic_model(model, {"z": _Constant()})
        with self.assertRaises(TypeError):
            SurrogateModel(("x1",), {"y": object()})  # type: ignore[dict-item]
        adapter = SurrogateModel(("x1", "x2", "x3"), {"y": _Constant()})
        self.assertEqual(
            adapter({"x1": 0.0, "x2": 1.0, "x3": 2.0, "extra": 5.0}), {"y": 3.0}
        )


class _Constant:
    """Trivial surrogate returning the input sum."""

    def predict(self, x):
        """Return the sum of the inputs."""
        return float(sum(x))


if __name__ == "__main__":
    unittest.main()
