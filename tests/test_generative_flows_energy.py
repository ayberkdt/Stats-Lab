"""Tests for normalizing flows, energy models, and Langevin sampling (ADR-0074)."""

from __future__ import annotations

import math
import random
import unittest
from collections.abc import Callable, Sequence

from probstat.foundations import RandomStream
from probstat.generative import (
    ActNorm,
    AffineCoupling,
    EnergyModel,
    NormalizingFlow,
    Permutation,
    PlanarFlow,
    RadialFlow,
    denoising_score_matching_objective,
    fit_denoising_score_matching,
    fit_flow,
    fit_score_matching,
    importance_normalizing_constant,
    kl_divergence_estimate,
    mean_log_likelihood,
    quadrature_normalizing_constant,
    real_nvp_flow,
    sample_langevin,
    score_matching_objective,
)
from probstat.inference import monte_carlo_standard_error
from probstat.numerics import reverse_value_and_gradient

Point = tuple[float, ...]
BANANA_CURVATURE = 0.5
BANANA_NOISE = 0.35


def gaussian_sample(size: int, seed: int) -> list[Point]:
    """Draw from N((1, -0.5), [[2.25, 1.2], [1.2, 1.0]])."""
    generator = random.Random(seed)
    points = []
    for _ in range(size):
        a, b = generator.gauss(0.0, 1.0), generator.gauss(0.0, 1.0)
        points.append((1.0 + 1.5 * a, -0.5 + 0.8 * a + 0.6 * b))
    return points


def banana_sample(size: int, seed: int) -> list[Point]:
    """Draw ``x1 ~ N(0, 1)``, ``x2 | x1 ~ N(x1^2 / 2 - 1, 0.35^2)``."""
    generator = random.Random(seed)
    points = []
    for _ in range(size):
        a = generator.gauss(0.0, 1.0)
        points.append(
            (a, BANANA_CURVATURE * a * a - 1.0 + BANANA_NOISE * generator.gauss(0, 1))
        )
    return points


def banana_log_density(point: Point) -> float:
    """Return the exact banana log-density."""
    a, b = point
    residual = (b - BANANA_CURVATURE * a * a + 1.0) / BANANA_NOISE
    return (
        -0.5 * a * a
        - 0.5 * residual * residual
        - math.log(2.0 * math.pi)
        - math.log(BANANA_NOISE)
    )


def moments(points: Sequence[Point]) -> tuple[list[float], list[list[float]]]:
    """Return the sample mean and the 1/n sample covariance."""
    n = len(points)
    d = len(points[0])
    mean = [math.fsum(p[i] for p in points) / n for i in range(d)]
    covariance = [
        [
            math.fsum((p[i] - mean[i]) * (p[j] - mean[j]) for p in points) / n
            for j in range(d)
        ]
        for i in range(d)
    ]
    return mean, covariance


def inverse_2x2(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
    """Return the inverse of a 2 x 2 matrix."""
    det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    return [
        [matrix[1][1] / det, -matrix[0][1] / det],
        [-matrix[1][0] / det, matrix[0][0] / det],
    ]


def gaussian_log_density(
    mean: Sequence[float], covariance: Sequence[Sequence[float]]
) -> Callable[[Point], float]:
    """Return the bivariate normal log-density with the given moments."""
    precision = inverse_2x2(covariance)
    det = covariance[0][0] * covariance[1][1] - covariance[0][1] ** 2

    def log_density(point: Point) -> float:
        """Evaluate the log-density."""
        d = [point[0] - mean[0], point[1] - mean[1]]
        quad = sum(d[i] * precision[i][j] * d[j] for i in range(2) for j in range(2))
        return -0.5 * quad - math.log(2.0 * math.pi) - 0.5 * math.log(det)

    return log_density


def finite_difference_log_det(
    function: Callable[[list[float]], list[float]], z: Sequence[float]
) -> float:
    """Return ``log |det J|`` of a 1-D or 2-D map by central differences."""
    h = 1e-6
    d = len(z)
    jacobian = [[0.0] * d for _ in range(d)]
    for j in range(d):
        plus, minus = list(z), list(z)
        plus[j] += h
        minus[j] -= h
        upper, lower = function(plus), function(minus)
        for i in range(d):
            jacobian[i][j] = (float(upper[i]) - float(lower[i])) / (2.0 * h)
    if d == 1:
        return math.log(abs(jacobian[0][0]))
    det = jacobian[0][0] * jacobian[1][1] - jacobian[0][1] * jacobian[1][0]
    return math.log(abs(det))


def perturbed(parameters: Sequence[float], seed: int, scale: float) -> list[float]:
    """Add Gaussian noise to parameters so that layers are far from identity."""
    generator = random.Random(seed)
    return [value + generator.gauss(0.0, scale) for value in parameters]


class FlowLayerTests(unittest.TestCase):
    """Log-determinants, inverses, and parameter gradients of each layer."""

    def layers(self) -> list[object]:
        """Return one instance of every layer type in 1-D and 2-D."""
        return [
            PlanarFlow(2),
            RadialFlow(2),
            AffineCoupling(2, hidden=(8,)),
            ActNorm(2),
            Permutation.reverse(2),
            PlanarFlow(1),
            RadialFlow(1),
            ActNorm(1),
        ]

    def test_log_determinants_match_finite_difference_jacobians(self) -> None:
        """Exact log |det J| agrees with a central-difference determinant."""
        for index, layer in enumerate(self.layers()):
            parameters = perturbed(
                layer.initial_parameters(RandomStream(3)),  # type: ignore[attr-defined]
                index,
                0.7,
            )
            for z in ([0.3, -0.8], [-1.4, 0.9], [2.0, 1.5]):
                point = z[: layer.dimension]  # type: ignore[attr-defined]
                _, log_det = layer.forward(parameters, point)  # type: ignore[attr-defined]
                expected = finite_difference_log_det(
                    lambda q, layer=layer, p=parameters: layer.forward(p, q)[0],  # type: ignore[attr-defined,misc]
                    point,
                )
                with self.subTest(layer=type(layer).__name__, z=point):
                    self.assertAlmostEqual(float(log_det), expected, places=7)

    def test_inverse_round_trip_and_log_det_sign(self) -> None:
        """inverse(forward(z)) = z and the two log-determinants cancel."""
        for index, layer in enumerate(self.layers()):
            parameters = perturbed(
                layer.initial_parameters(RandomStream(5)),  # type: ignore[attr-defined]
                index + 10,
                1.0,
            )
            for z in ([0.3, -0.8], [-2.5, 3.1], [4.0, -0.2]):
                point = z[: layer.dimension]  # type: ignore[attr-defined]
                x, log_det = layer.forward(parameters, point)  # type: ignore[attr-defined]
                back, inverse_log_det = layer.inverse(parameters, x)  # type: ignore[attr-defined]
                with self.subTest(layer=type(layer).__name__, z=point):
                    for a, b in zip(back, point, strict=True):
                        self.assertAlmostEqual(float(a), b, places=11)
                    self.assertAlmostEqual(
                        float(log_det) + float(inverse_log_det), 0.0, places=11
                    )

    def test_planar_constraint_keeps_maps_invertible(self) -> None:
        """Large raw u with w'u << -1 still gives w'u_hat > -1 and finite log det."""
        layer = PlanarFlow(2)
        parameters = [-5.0, -4.0, 1.0, 1.0, 0.3]
        for z in ([0.0, 0.0], [3.0, -3.0], [-0.15, -0.15]):
            x, log_det = layer.forward(parameters, z)
            self.assertTrue(math.isfinite(float(log_det)))
            back, _ = layer.inverse(parameters, x)
            for a, b in zip(back, z, strict=True):
                self.assertAlmostEqual(float(a), b, places=10)

    def test_density_gradient_through_numerical_inverse(self) -> None:
        """Reverse-mode gradient of log p (planar Newton step) matches differences."""
        flow = NormalizingFlow(
            (PlanarFlow(2), RadialFlow(2), PlanarFlow(2), ActNorm(2))
        )
        parameters = perturbed(flow.initial_parameters(5), 2, 0.5)
        x = (0.4, 1.1)
        _, gradient = reverse_value_and_gradient(
            lambda theta: flow.log_density(theta, x), parameters
        )
        for i in range(len(parameters)):
            plus, minus = list(parameters), list(parameters)
            plus[i] += 1e-6
            minus[i] -= 1e-6
            expected = (flow.log_density(plus, x) - flow.log_density(minus, x)) / 2e-6
            self.assertAlmostEqual(gradient[i], expected, places=6)

    def test_flow_densities_integrate_to_one(self) -> None:
        """Quadrature of exp(log p) is 1 in one and two dimensions."""
        one = NormalizingFlow((PlanarFlow(1), RadialFlow(1), PlanarFlow(1), ActNorm(1)))
        p1 = perturbed(one.initial_parameters(4), 7, 0.8)
        z1 = quadrature_normalizing_constant(
            lambda q: one.log_density(p1, q), -math.inf, math.inf, rel_tol=1e-9
        )
        self.assertLess(abs(z1.log_value), 1e-7)
        two = NormalizingFlow(
            (
                AffineCoupling(2, hidden=(4,)),
                Permutation.reverse(2),
                RadialFlow(2),
                ActNorm(2),
            )
        )
        p2 = perturbed(two.initial_parameters(6), 8, 0.25)
        z2 = quadrature_normalizing_constant(
            lambda q: two.log_density(p2, q), (-11.0, -11.0), (11.0, 11.0), rel_tol=1e-5
        )
        self.assertLess(abs(z2.log_value), 1e-5)
        self.assertLess(z2.error, 1e-4)

    def test_sample_log_density_matches_inverse_density(self) -> None:
        """Forward-map log-densities of draws equal the change-of-variables density."""
        flow = NormalizingFlow(
            (RadialFlow(2), AffineCoupling(2, hidden=(4,)), ActNorm(2))
        )
        parameters = perturbed(flow.initial_parameters(1), 3, 0.5)
        points, log_densities = flow.sample_with_log_density(parameters, 20, seed=4)
        for point, value in zip(points, log_densities, strict=True):
            self.assertAlmostEqual(
                flow.log_density(parameters, point), value, places=10
            )

    def test_invalid_layers_and_inputs_raise(self) -> None:
        """Invalid settings and shapes raise ValueError."""
        with self.assertRaises(ValueError):
            AffineCoupling(1)
        with self.assertRaises(ValueError):
            AffineCoupling(3, split=3)
        with self.assertRaises(ValueError):
            AffineCoupling(2, scale_bound=0.0)
        with self.assertRaises(ValueError):
            Permutation((0, 0))
        with self.assertRaises(ValueError):
            PlanarFlow(0)
        with self.assertRaises(ValueError):
            NormalizingFlow(())
        with self.assertRaises(ValueError):
            NormalizingFlow((PlanarFlow(1), ActNorm(2)))
        flow = NormalizingFlow((ActNorm(2),))
        with self.assertRaises(ValueError):
            flow.log_density([0.0], (0.0, 0.0))
        with self.assertRaises(ValueError):
            flow.log_density([0.0] * 4, (0.0,))
        with self.assertRaises(ValueError):
            flow.sample([0.0] * 4, 0)


class FlowFitTests(unittest.TestCase):
    """Maximum-likelihood fits of flows."""

    def test_gaussian_fit_recovers_mean_and_covariance(self) -> None:
        """A Real NVP flow fitted to Gaussian data reproduces its first two moments."""
        data = gaussian_sample(300, 11)
        fit = fit_flow(
            data,
            real_nvp_flow(2, couplings=2, hidden=(8,)),
            iterations=200,
            learning_rate=0.01,
            seed=1,
        )
        mean, covariance = moments(data)
        sample_mean, sample_covariance = moments(fit.sample(4000, seed=2))
        for i in range(2):
            self.assertLess(abs(sample_mean[i] - mean[i]), 0.15)
            for j in range(2):
                self.assertLess(abs(sample_covariance[i][j] - covariance[i][j]), 0.2)
        gaussian = mean_log_likelihood(gaussian_log_density(mean, covariance), data)
        self.assertLess(abs(fit.train_mean_log_likelihood - gaussian.estimate), 0.05)
        self.assertEqual(len(fit.losses), 200)
        self.assertLess(fit.losses[-1], fit.losses[0])

    def test_banana_flow_beats_gaussian_out_of_sample(self) -> None:
        """Held-out log-likelihood and KL to the truth improve on one Gaussian."""
        train = banana_sample(300, 5)
        test = banana_sample(300, 6)
        fit = fit_flow(
            train,
            real_nvp_flow(2, couplings=2, hidden=(8,)),
            iterations=300,
            learning_rate=0.02,
            heldout=test,
            seed=3,
        )
        mean, covariance = moments(train)
        gaussian = gaussian_log_density(mean, covariance)
        gaussian_heldout = mean_log_likelihood(gaussian, test)
        assert fit.heldout_mean_log_likelihood is not None
        assert fit.heldout_standard_error is not None
        self.assertGreater(
            fit.heldout_mean_log_likelihood,
            gaussian_heldout.estimate + 0.5,
        )
        flow_kl = kl_divergence_estimate(banana_log_density, fit.log_density, test)
        gaussian_kl = kl_divergence_estimate(banana_log_density, gaussian, test)
        self.assertGreater(flow_kl.estimate, -3.0 * flow_kl.standard_error)
        self.assertLess(flow_kl.estimate + 0.5, gaussian_kl.estimate)

    def test_one_dimensional_default_flow_and_reproducibility(self) -> None:
        """The 1-D default flow fits a skewed sample; equal seeds give equal fits."""
        generator = random.Random(9)
        data = [math.exp(0.5 * generator.gauss(0.0, 1.0)) for _ in range(120)]
        first = fit_flow(data, iterations=60, batch_size=32, seed=4)
        second = fit_flow(data, iterations=60, batch_size=32, seed=4)
        third = fit_flow(data, iterations=60, batch_size=32, seed=5)
        self.assertEqual(first.parameters, second.parameters)
        self.assertEqual(first.losses, second.losses)
        self.assertNotEqual(first.parameters, third.parameters)
        total = quadrature_normalizing_constant(
            lambda q: first.log_density(q), -math.inf, math.inf
        )
        self.assertLess(abs(total.log_value), 1e-5)

    def test_invalid_fit_inputs_raise(self) -> None:
        """Bad data and settings raise before any training."""
        flow = real_nvp_flow(2, couplings=1, hidden=(4,))
        with self.assertRaises(ValueError):
            fit_flow([(0.0, 1.0)], flow)
        with self.assertRaises(ValueError):
            fit_flow([(0.0, 1.0), (1.0, math.nan)], flow)
        with self.assertRaises(ValueError):
            fit_flow([(0.0, 1.0), (1.0,)], flow)
        with self.assertRaises(ValueError):
            fit_flow([(0.0,), (1.0,)], flow)
        with self.assertRaises(ValueError):
            fit_flow([(0.0, 1.0), (1.0, 2.0)], flow, iterations=0)
        with self.assertRaises(ValueError):
            fit_flow([(0.0, 1.0), (1.0, 2.0)], flow, batch_size=0)
        with self.assertRaises(ValueError):
            fit_flow([(0.0, 1.0), (0.0, 2.0)], flow)
        with self.assertRaises(TypeError):
            fit_flow("data", flow)


class EnergyModelTests(unittest.TestCase):
    """Score matching, denoising score matching, and the Laplacian jet."""

    def test_jet_matches_forward_over_reverse_derivatives(self) -> None:
        """Jet value, gradient, and Laplacian equal E, grad E, and trace(Hessian)."""
        for activation in ("tanh", "softplus", "sigmoid"):
            model = EnergyModel(2, hidden=(6,), activation=activation)
            parameters = perturbed(model.initial_parameters(3), 1, 0.5)
            for x in ((0.3, -0.7), (1.5, 2.0)):
                jet = model.jet(parameters, x)
                with self.subTest(activation=activation, x=x):
                    self.assertAlmostEqual(jet.value, model.energy(parameters, x), 12)
                    for a, b in zip(
                        jet.gradient, model.energy_gradient(parameters, x), strict=True
                    ):
                        self.assertAlmostEqual(a, b, places=12)
                    self.assertAlmostEqual(
                        jet.laplacian, model.energy_laplacian(parameters, x), places=10
                    )

    def test_score_matching_recovers_gaussian_precision(self) -> None:
        """Quadratic energy: the minimizer is the inverse 1/n sample covariance."""
        data = gaussian_sample(120, 11)
        mean, covariance = moments(data)
        precision = inverse_2x2(covariance)
        fit = fit_score_matching(
            data,
            EnergyModel(2, hidden=None),
            iterations=400,
            batch_size=120,
            learning_rate=0.05,
            seed=0,
        )
        fitted = fit.model.precision(fit.parameters)
        center = fit.model.center(fit.parameters)
        for i in range(2):
            self.assertAlmostEqual(center[i], mean[i], places=3)
            for j in range(2):
                self.assertLess(abs(fitted[i][j] - precision[i][j]), 1e-3)
        # J = mean |A v|^2 / 2 - tr A = tr(A S A) / 2 - tr A = -tr(A) / 2 at A = S^-1.
        exact = -0.5 * (precision[0][0] + precision[1][1])
        self.assertAlmostEqual(fit.objective, exact, places=5)
        self.assertAlmostEqual(
            score_matching_objective(fit.model, fit.parameters, data),
            fit.objective,
            places=12,
        )

    def test_denoising_score_matching_recovers_smoothed_precision(self) -> None:
        """Quadratic energy: DSM targets (S + s^2 I)^-1, not S^-1."""
        data = gaussian_sample(120, 11)
        mean, covariance = moments(data)
        noise = 0.3
        smoothed = inverse_2x2(
            [
                [covariance[0][0] + noise**2, covariance[0][1]],
                [covariance[1][0], covariance[1][1] + noise**2],
            ]
        )
        sharp = inverse_2x2(covariance)
        fit = fit_denoising_score_matching(
            data,
            EnergyModel(2, hidden=None),
            noise_scale=noise,
            iterations=400,
            batch_size=120,
            learning_rate=0.02,
            tail_fraction=0.5,
            seed=0,
        )
        fitted = fit.model.precision(fit.parameters)
        scale = max(abs(v) for row in smoothed for v in row)
        to_smoothed = max(
            abs(fitted[i][j] - smoothed[i][j]) for i in range(2) for j in range(2)
        )
        to_sharp = max(
            abs(fitted[i][j] - sharp[i][j]) for i in range(2) for j in range(2)
        )
        self.assertLess(to_smoothed, 0.05 * scale)
        self.assertGreater(to_sharp, 4.0 * to_smoothed)
        self.assertEqual(fit.noise_scale, noise)
        self.assertTrue(math.isfinite(fit.objective))
        self.assertEqual(
            denoising_score_matching_objective(
                fit.model, fit.parameters, data, noise, seed=1
            ),
            denoising_score_matching_objective(
                fit.model, fit.parameters, data, noise, seed=1
            ),
        )

    def test_neural_energy_on_banana_beats_gaussian(self) -> None:
        """A score-matched MLP energy, normalized by quadrature, beats one Gaussian."""
        train = banana_sample(300, 5)
        test = banana_sample(300, 6)
        fit = fit_score_matching(
            train,
            EnergyModel(2, hidden=(8,)),
            iterations=150,
            batch_size=32,
            learning_rate=0.03,
            seed=0,
        )
        normalizer = quadrature_normalizing_constant(
            lambda q: -fit.energy(q), (-7.0, -7.0), (7.0, 7.0), rel_tol=1e-5
        )
        self.assertLess(normalizer.error, 1e-3)
        energy_ll = mean_log_likelihood(
            lambda q: -fit.energy(q) - normalizer.log_value, test
        )
        mean, covariance = moments(train)
        gaussian_ll = mean_log_likelihood(gaussian_log_density(mean, covariance), test)
        self.assertGreater(energy_ll.estimate, gaussian_ll.estimate + 0.3)
        self.assertLess(fit.losses[-1], fit.losses[0])

    def test_normalizing_constants_of_a_gaussian_energy(self) -> None:
        """Quadrature and importance sampling recover (2 pi)^{d/2} / sqrt(det A)."""
        model = EnergyModel(2, hidden=None)
        parameters = model.initial_parameters(0, data=gaussian_sample(200, 2))
        precision = model.precision(parameters)
        det = precision[0][0] * precision[1][1] - precision[0][1] ** 2
        exact = math.log(2.0 * math.pi) - 0.5 * math.log(det)

        def log_f(q: Point) -> float:
            """Unnormalized log-density -E."""
            return -float(model.energy(parameters, q))

        center = model.center(parameters)
        quadrature = quadrature_normalizing_constant(
            log_f,
            (center[0] - 12.0, center[1] - 12.0),
            (center[0] + 12.0, center[1] + 12.0),
        )
        self.assertAlmostEqual(quadrature.log_value, exact, places=6)
        proposal = NormalizingFlow((ActNorm(2),))
        widths = [0.0, 0.0, 0.7, 0.4]
        proposal_parameters = [center[0], center[1], *widths[2:]]
        samples, log_q = proposal.sample_with_log_density(
            proposal_parameters, 4000, seed=3
        )
        importance = importance_normalizing_constant(log_f, samples, log_q)
        self.assertLess(abs(importance.log_value - exact), 4.0 * importance.error)
        assert importance.effective_sample_size is not None
        self.assertGreater(importance.effective_sample_size, 400.0)

    def test_invalid_energy_inputs_raise(self) -> None:
        """Invalid models, data, and settings raise."""
        with self.assertRaises(ValueError):
            EnergyModel(2, activation="identity")
        with self.assertRaises(ValueError):
            EnergyModel(2, hidden=())
        with self.assertRaises(ValueError):
            EnergyModel(0)
        data = gaussian_sample(10, 1)
        with self.assertRaises(ValueError):
            fit_score_matching(data, EnergyModel(3, hidden=None), iterations=1)
        with self.assertRaises(ValueError):
            fit_denoising_score_matching(data, noise_scale=0.0, iterations=1)
        with self.assertRaises(ValueError):
            fit_score_matching(data, iterations=1, initial_parameters=[0.0])
        with self.assertRaises(ValueError):
            quadrature_normalizing_constant(lambda q: 0.0, (1.0,), (0.0,))
        with self.assertRaises(ValueError):
            importance_normalizing_constant(lambda q: 0.0, [(0.0,), (1.0,)], [0.0])


class LangevinTests(unittest.TestCase):
    """Unadjusted and Metropolis-adjusted Langevin chains."""

    @staticmethod
    def correlated_log_density(x: Sequence[object]) -> object:
        """Log-density of N(0, [[1, 0.8], [0.8, 1]]) up to a constant."""
        a, b = x[0], x[1]
        return -(a * a - 1.6 * a * b + b * b) / (2.0 * 0.36)  # type: ignore[operator]

    def test_mala_matches_target_moments(self) -> None:
        """MALA means, variances, and covariance lie within 4 Monte Carlo SEs."""
        result = sample_langevin(
            self.correlated_log_density,
            (0.0, 0.0),
            n_samples=6000,
            step_size=0.3,
            burn_in=200,
            seed=7,
        )
        assert result.acceptance_rate is not None
        self.assertGreater(result.acceptance_rate, 0.4)
        self.assertLess(result.acceptance_rate, 0.99)
        targets = {
            "x": ([s[0] for s in result.samples], 0.0),
            "y": ([s[1] for s in result.samples], 0.0),
            "xx": ([s[0] ** 2 for s in result.samples], 1.0),
            "yy": ([s[1] ** 2 for s in result.samples], 1.0),
            "xy": ([s[0] * s[1] for s in result.samples], 0.8),
        }
        for name, (series, target) in targets.items():
            error = monte_carlo_standard_error([series])
            estimate = math.fsum(series) / len(series)
            with self.subTest(moment=name):
                self.assertLess(abs(estimate - target), 4.0 * error)
        self.assertEqual(result.gradient_evaluations, 1 + 200 + 6000)

    def test_ula_matches_its_biased_stationary_variance(self) -> None:
        """ULA on N(0, 1) has variance 1 / (1 - h/4); MALA removes the bias."""
        h = 0.5

        def log_density(x: Sequence[object]) -> object:
            """Return the standard normal log-density up to a constant."""
            return -0.5 * (x[0] * x[0])  # type: ignore[operator]

        ula = sample_langevin(
            log_density, 0.0, n_samples=8000, step_size=h, adjusted=False, seed=1
        )
        mala = sample_langevin(log_density, 0.0, n_samples=8000, step_size=h, seed=1)
        self.assertIsNone(ula.acceptance_rate)
        for result, target in ((ula, 1.0 / (1.0 - h / 4.0)), (mala, 1.0)):
            squares = [s[0] ** 2 for s in result.samples]
            estimate = math.fsum(squares) / len(squares)
            error = monte_carlo_standard_error([squares])
            with self.subTest(method=result.method):
                self.assertLess(abs(estimate - target), 4.0 * error)
        ula_squares = [s[0] ** 2 for s in ula.samples]
        self.assertGreater(math.fsum(ula_squares) / len(ula_squares), 1.05)

    def test_energy_fit_sampling_is_reproducible(self) -> None:
        """Equal seeds give identical chains; thinning keeps every k-th state."""
        data = gaussian_sample(60, 3)
        fit = fit_score_matching(
            data, EnergyModel(2, hidden=None), iterations=30, batch_size=60, seed=0
        )
        first = fit.sample_langevin(
            (1.0, -0.5), n_samples=50, step_size=0.2, thin=3, seed=2
        )
        second = fit.sample_langevin(
            (1.0, -0.5), n_samples=50, step_size=0.2, thin=3, seed=2
        )
        self.assertEqual(first.samples, second.samples)
        self.assertEqual(len(first.samples), 50)
        self.assertEqual(first.thin, 3)

    def test_invalid_langevin_inputs_raise(self) -> None:
        """Bad settings and a start outside the support raise."""

        def log_density(x: Sequence[object]) -> object:
            """Log-density with support x > 0."""
            from probstat.numerics import admath

            return admath.log(x[0])

        with self.assertRaises(ValueError):
            sample_langevin(log_density, (1.0,), n_samples=0, step_size=0.1)
        with self.assertRaises(ValueError):
            sample_langevin(log_density, (1.0,), n_samples=5, step_size=-0.1)
        with self.assertRaises(ValueError):
            sample_langevin(log_density, (1.0,), n_samples=5, step_size=0.1, thin=0)
        with self.assertRaises(ValueError):
            sample_langevin(log_density, (-1.0,), n_samples=5, step_size=0.1)
        result = sample_langevin(log_density, (1.0,), n_samples=200, step_size=0.5)
        self.assertTrue(all(s[0] > 0.0 for s in result.samples))


if __name__ == "__main__":
    unittest.main()
