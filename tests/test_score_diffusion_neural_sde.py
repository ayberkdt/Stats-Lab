"""Tests for score-based diffusion samplers, score matching, and neural SDEs."""

from __future__ import annotations

import math
import random
import unittest
from collections.abc import Callable, Sequence

from probstat.processes import (
    GaussianScore,
    NeuralSDE,
    NeuralSDEFit,
    OrnsteinUhlenbeck,
    SamplePath,
    ScoreNetwork,
    VarianceExplodingSDE,
    VariancePreservingSDE,
    VectorSamplePath,
    fit_neural_sde,
    milstein,
    perturb,
    perturbation_moments,
    predictor_corrector_sample,
    probability_flow_map,
    probability_flow_sample,
    reverse_sde_sample,
    train_score_network,
)

DATA_MEAN = 1.0
DATA_VARIANCE = 0.25
COUNT = 400
ScoreFunction = Callable[[tuple[float, ...], float], Sequence[float]]


def assert_moments(
    case: unittest.TestCase,
    means: tuple[float, ...],
    variances: tuple[float, ...],
    target_means: tuple[float, ...],
    target_variances: tuple[float, ...],
    count: int,
    z: float = 3.5,
) -> None:
    """Assert sample moments lie within ``z`` standard errors of the targets.

    The standard error of a Gaussian sample mean is ``sqrt(s^2 / n)`` and that
    of the unbiased variance is ``s^2 sqrt(2 / (n - 1))``.

    References:
        [EQ-MC-MEAN-001], [REF-OWEN-2013, Chapter 2].

    """
    for mean, variance, mu, s2 in zip(
        means, variances, target_means, target_variances, strict=True
    ):
        case.assertLess(abs(mean - mu), z * math.sqrt(s2 / count))
        case.assertLess(abs(variance - s2), z * s2 * math.sqrt(2.0 / (count - 1)))


class PerturbationKernelTests(unittest.TestCase):
    """VP and VE kernels match their closed forms and moment equations."""

    def test_vp_kernel_closed_form(self) -> None:
        """VP mean coefficient and variance follow ``exp(-B/2)`` and ``1 - exp(-B)``."""
        sde = VariancePreservingSDE(0.1, 20.0)
        for t in (0.0, 0.1, 0.5, 1.0):
            integral = 0.1 * t + 0.5 * 19.9 * t * t
            mean, variance = perturbation_moments(sde, (2.0, -1.0), t)
            self.assertAlmostEqual(mean[0], 2.0 * math.exp(-0.5 * integral), places=14)
            self.assertAlmostEqual(mean[1], -math.exp(-0.5 * integral), places=14)
            self.assertAlmostEqual(variance, 1.0 - math.exp(-integral), places=14)

    def test_ve_kernel_closed_form(self) -> None:
        """VE variance is ``sigma(t)^2 - sigma_min^2`` with the geometric schedule."""
        sde = VarianceExplodingSDE(0.01, 50.0)
        for t in (0.0, 0.3, 1.0):
            sigma = 0.01 * (50.0 / 0.01) ** t
            mean, variance = perturbation_moments(sde, (3.0,), t)
            self.assertEqual(mean, (3.0,))
            self.assertAlmostEqual(variance, sigma**2 - 1e-4, delta=1e-12 * sigma**2)
        self.assertEqual(sde.prior_variance, 2500.0)

    def test_kernels_solve_the_moment_equations(self) -> None:
        """``m' = -beta m / 2`` (VP) and ``v' = 2 (f/x) v + g^2`` hold numerically."""
        h = 1e-6
        for sde in (VariancePreservingSDE(), VarianceExplodingSDE()):
            for t in (0.2, 0.6):
                rate = sde.drift((1.0,), t)[0]
                m_up, m_down = sde.mean_coefficient(t + h), sde.mean_coefficient(t - h)
                v_up, v_down = sde.kernel_variance(t + h), sde.kernel_variance(t - h)
                dm = (m_up - m_down) / (2 * h)
                dv = (v_up - v_down) / (2 * h)
                self.assertAlmostEqual(dm, rate * sde.mean_coefficient(t), delta=1e-6)
                expected = 2.0 * rate * sde.kernel_variance(t) + sde.diffusion(t) ** 2
                self.assertAlmostEqual(dv / expected, 1.0, delta=1e-6)

    def test_perturb_draws_match_kernel_moments(self) -> None:
        """Monte Carlo draws of ``perturb`` match the kernel mean and variance."""
        sde = VariancePreservingSDE()
        t, n = 0.3, 2000
        draws = [perturb(sde, (1.5,), t, seed=seed)[0] for seed in range(n)]
        (mean,), variance = perturbation_moments(sde, (1.5,), t)
        sample_mean = math.fsum(draws) / n
        sample_variance = math.fsum((x - sample_mean) ** 2 for x in draws) / (n - 1)
        assert_moments(
            self, (sample_mean,), (sample_variance,), (mean,), (variance,), n
        )

    def test_gaussian_score_is_gradient_of_log_density(self) -> None:
        """The closed-form score equals a finite difference of the log-density."""
        sde = VariancePreservingSDE()
        score = GaussianScore(sde, (1.0, -0.5), (0.25, 1.0))
        t, x, h = 0.4, (0.3, 0.2), 1e-5

        def log_density(point: tuple[float, float]) -> float:
            """Return the log-density of the diffused Gaussian marginal.

            References:
                [EQ-PROC-DIFF-003].

            """
            means = score.marginal_mean(t)
            variances = score.marginal_variance(t)
            return -0.5 * math.fsum(
                (p - m) ** 2 / v + math.log(2 * math.pi * v)
                for p, m, v in zip(point, means, variances, strict=True)
            )

        exact = score(x, t)
        for j in range(2):
            up = list(x)
            down = list(x)
            up[j] += h
            down[j] -= h
            numeric = (log_density((up[0], up[1])) - log_density((down[0], down[1])))
            self.assertAlmostEqual(exact[j], numeric / (2 * h), places=6)


class ExactScoreSamplerTests(unittest.TestCase):
    """With the exact Gaussian score every sampler recovers the data law."""

    def check(self, sde: VariancePreservingSDE | VarianceExplodingSDE) -> None:
        """Run the three samplers on one SDE and check the moments.

        References:
            [EQ-PROC-DIFF-005], [EQ-PROC-DIFF-006], [EQ-PROC-DIFF-007].

        """
        score = GaussianScore(sde, DATA_MEAN, DATA_VARIANCE)
        samples = (
            reverse_sde_sample(sde, score, COUNT, steps=300, seed=11),
            predictor_corrector_sample(sde, score, COUNT, steps=200, seed=12),
            probability_flow_sample(sde, score, COUNT, steps=50, seed=13),
        )
        for sample in samples:
            with self.subTest(sde=type(sde).__name__, method=sample.method):
                assert_moments(
                    self,
                    sample.means,
                    sample.variances,
                    (DATA_MEAN,),
                    (DATA_VARIANCE,),
                    COUNT,
                )
                self.assertEqual(len(sample.points), COUNT)
        self.assertEqual(samples[0].score_evaluations, COUNT * 300)
        self.assertEqual(samples[1].score_evaluations, COUNT * 200 * 2)
        self.assertEqual(samples[2].score_evaluations, COUNT * 50 * 4)

    def test_variance_preserving_samplers(self) -> None:
        """Reverse SDE, predictor-corrector, and ODE samplers on the VP SDE."""
        self.check(VariancePreservingSDE())

    def test_variance_exploding_samplers(self) -> None:
        """Reverse SDE, predictor-corrector, and ODE samplers on the VE SDE."""
        self.check(VarianceExplodingSDE())

    def test_two_dimensional_flow(self) -> None:
        """The probability-flow sampler recovers a two-dimensional diagonal law."""
        sde = VariancePreservingSDE()
        score = GaussianScore(sde, (1.0, -0.5), (0.25, 1.0))
        sample = probability_flow_sample(sde, score, 300, dimension=2, steps=40, seed=3)
        assert_moments(
            self, sample.means, sample.variances, (1.0, -0.5), (0.25, 1.0), 300
        )

    def test_ode_map_is_deterministic_affine_and_invertible(self) -> None:
        """The flow map is reproducible, matches the exact affine map, and inverts."""
        sde = VariancePreservingSDE()
        score = GaussianScore(sde, DATA_MEAN, DATA_VARIANCE)
        prior = [-1.5, -0.2, 0.4, 2.0]
        end = sde.default_end_time
        first = probability_flow_map(sde, score, prior, steps=60)
        second = probability_flow_map(sde, score, prior, steps=60)
        self.assertEqual(first.points, second.points)
        m1, v1 = score.marginal_mean(1.0)[0], score.marginal_variance(1.0)[0]
        me, ve = score.marginal_mean(end)[0], score.marginal_variance(end)[0]
        heun = probability_flow_map(sde, score, prior, steps=200, method="heun")
        for x, (y,), (z,) in zip(prior, first.points, heun.points, strict=True):
            exact = me + (x - m1) * math.sqrt(ve / v1)
            self.assertAlmostEqual(y, exact, delta=1e-4)
            self.assertAlmostEqual(z, exact, delta=1e-3)
        back = probability_flow_map(
            sde, score, first.points, start_time=end, end_time=1.0, steps=60
        )
        for x, (y,) in zip(prior, back.points, strict=True):
            self.assertAlmostEqual(x, y, delta=1e-4)


class ScoreMatchingTests(unittest.TestCase):
    """Denoising score matching approaches the true Gaussian score."""

    def test_trained_score_approaches_true_score(self) -> None:
        """Relative grid error of the trained score is small and far below the start."""
        sde = VariancePreservingSDE()
        generator = random.Random(7)
        data = [DATA_MEAN + 0.5 * generator.gauss(0.0, 1.0) for _ in range(300)]
        exact = GaussianScore(sde, DATA_MEAN, DATA_VARIANCE)
        trained = train_score_network(
            sde, data, iterations=800, batch_size=32, learning_rate=0.02, seed=2
        )
        network = ScoreNetwork(sde, 1)
        start = network.initial_parameters(0)

        def relative_error(score: ScoreFunction, t: float) -> float:
            """Return the relative L2 score error on a grid of +-2 marginal sd.

            References:
                [EQ-PROC-DIFF-003], [EQ-PROC-DIFF-004].

            """
            mean = exact.marginal_mean(t)[0]
            sd = math.sqrt(exact.marginal_variance(t)[0])
            grid = [mean + 0.5 * k * sd for k in range(-4, 5)]
            num = math.fsum(
                (score((x,), t)[0] - exact((x,), t)[0]) ** 2 for x in grid
            )
            den = math.fsum(exact((x,), t)[0] ** 2 for x in grid)
            return math.sqrt(num / den)

        for t in (0.1, 0.3, 0.7):
            with self.subTest(t=t):
                trained_error = relative_error(trained, t)
                initial_error = relative_error(
                    lambda x, s: network.score(start, x, s), t
                )
                self.assertLess(trained_error, 0.25)
                self.assertLess(trained_error, 0.2 * initial_error)
        losses = trained.optimization.losses
        self.assertLess(trained.optimization.smoothed_loss(100), sum(losses[:20]) / 20)
        self.assertEqual(trained.training_points, 300)

    def test_training_is_reproducible(self) -> None:
        """Equal seeds give identical parameters; different seeds differ."""
        sde = VarianceExplodingSDE(0.01, 5.0)
        data = [0.1 * k for k in range(-5, 6)]
        first = train_score_network(sde, data, iterations=5, batch_size=4, seed=9)
        second = train_score_network(sde, data, iterations=5, batch_size=4, seed=9)
        third = train_score_network(sde, data, iterations=5, batch_size=4, seed=10)
        self.assertEqual(first.parameters, second.parameters)
        self.assertNotEqual(first.parameters, third.parameters)


class NeuralSDETests(unittest.TestCase):
    """Pseudo-likelihood fits of neural SDEs to Ornstein-Uhlenbeck paths."""

    process: OrnsteinUhlenbeck
    paths: list[SamplePath]
    fit: NeuralSDEFit

    @classmethod
    def setUpClass(cls) -> None:
        """Simulate exact OU paths and fit a constant-diffusion neural SDE.

        References:
            [EQ-PROC-OU-001], [EQ-PROC-NSDE-001].

        """
        cls.process = OrnsteinUhlenbeck(1.0, 0.0, 0.5)
        cls.paths = [
            cls.process.path(5.0, 250, seed=100 + i) for i in range(40)
        ]
        cls.fit = fit_neural_sde(
            NeuralSDE(hidden=(8,)),
            cls.paths,
            iterations=400,
            batch_size=100,
            learning_rate=0.02,
            seed=1,
        )

    def test_recovers_linear_drift_and_diffusion(self) -> None:
        """Grid slope, intercept, and diffusion match OU and the linear Euler fit."""
        xs: list[float] = []
        rates: list[float] = []
        for path in self.paths:
            for a, b in zip(path.values[:-1], path.values[1:], strict=True):
                xs.append(a)
                rates.append((b - a) / 0.02)
        x_bar = math.fsum(xs) / len(xs)
        r_bar = math.fsum(rates) / len(rates)
        ols = math.fsum(
            (x - x_bar) * (r - r_bar) for x, r in zip(xs, rates, strict=True)
        ) / math.fsum((x - x_bar) ** 2 for x in xs)
        grid = [-0.6 + 0.1 * k for k in range(13)]
        rows = self.fit.drift_on_grid(grid)
        values = [drift[0] for _, drift in rows]
        g_bar = math.fsum(grid) / len(grid)
        v_bar = math.fsum(values) / len(values)
        slope = math.fsum(
            (g - g_bar) * (v - v_bar) for g, v in zip(grid, values, strict=True)
        ) / math.fsum((g - g_bar) ** 2 for g in grid)
        self.assertAlmostEqual(slope, -1.0, delta=0.25)
        self.assertAlmostEqual(slope, ols, delta=0.1)
        self.assertAlmostEqual(v_bar - slope * g_bar, 0.0, delta=0.1)
        for (state,), drift in rows:
            self.assertAlmostEqual(drift[0], -state, delta=0.15)
        self.assertAlmostEqual(self.fit.diffusion(0.0)[0], 0.5, delta=0.025)
        self.assertEqual(self.fit.transitions, 40 * 250)
        self.assertTrue(math.isfinite(self.fit.mean_negative_log_likelihood))

    def test_simulation_uses_existing_integrators(self) -> None:
        """The scalar conversion reproduces ``simulate`` and runs under Milstein."""
        path = self.fit.simulate(0.3, 1.0, 50, seed=4)
        direct = milstein(self.fit.to_scalar_sde(), 0.3, 1.0, 50, seed=4)
        self.assertEqual(len(path.states), 51)
        self.assertEqual(tuple(s[0] for s in path.states), direct.values)
        again = self.fit.simulate(0.3, 1.0, 50, seed=4)
        self.assertEqual(path.states, again.states)

    def test_mlp_diffusion_and_two_dimensions(self) -> None:
        """An MLP diffusion fits briefly; a 2-D model simulates componentwise."""
        model = NeuralSDE(diffusion="mlp", hidden=(4,), diffusion_hidden=(4,))
        fit = fit_neural_sde(model, self.paths[:4], iterations=20, batch_size=16)
        self.assertGreater(fit.diffusion(0.1)[0], 0.0)
        sde = fit.to_scalar_sde()
        assert sde.diffusion_derivative is not None
        h = 1e-6
        numeric = (sde.diffusion(0.0, 0.2 + h) - sde.diffusion(0.0, 0.2 - h)) / (2 * h)
        self.assertAlmostEqual(sde.diffusion_derivative(0.0, 0.2), numeric, places=6)
        states = [(0.1 * k, -0.05 * k) for k in range(30)]
        fit2 = fit_neural_sde(
            NeuralSDE(dimension=2, hidden=(4,)),
            [VectorSamplePath(tuple(0.1 * k for k in range(30)), tuple(states))],
            iterations=10,
            batch_size=8,
        )
        path = fit2.simulate((0.0, 0.0), 1.0, 10, seed=1)
        self.assertEqual(len(path.terminal), 2)
        self.assertEqual(len(path.coordinate(1).values), 11)

    def test_fit_is_reproducible(self) -> None:
        """Equal seeds give identical neural-SDE parameters."""
        first = fit_neural_sde(NeuralSDE(hidden=(4,)), self.paths[:2], iterations=5)
        second = fit_neural_sde(NeuralSDE(hidden=(4,)), self.paths[:2], iterations=5)
        self.assertEqual(first.parameters, second.parameters)


class ReproducibilityAndValidationTests(unittest.TestCase):
    """Seeds reproduce samplers; invalid inputs raise."""

    def test_samplers_reproduce_with_seeds(self) -> None:
        """Equal seeds give identical samples for every stochastic sampler."""
        sde = VarianceExplodingSDE()
        score = GaussianScore(sde, 0.0, 1.0)
        for sampler in (reverse_sde_sample, predictor_corrector_sample):
            first = sampler(sde, score, 5, steps=20, seed=8)
            second = sampler(sde, score, 5, steps=20, seed=8)
            third = sampler(sde, score, 5, steps=20, seed=9)
            self.assertEqual(first.points, second.points)
            self.assertNotEqual(first.points, third.points)
        flow = probability_flow_sample(sde, score, 5, steps=10, seed=8)
        self.assertEqual(
            flow.points, probability_flow_sample(sde, score, 5, steps=10, seed=8).points
        )

    def test_invalid_inputs_raise(self) -> None:
        """Schedules, times, counts, methods, scores, and data are validated."""
        vp = VariancePreservingSDE()
        score = GaussianScore(vp, 0.0, 1.0)
        with self.assertRaises(ValueError):
            VariancePreservingSDE(1.0, 0.5)
        with self.assertRaises(ValueError):
            VariancePreservingSDE(0.0, 1.0)
        with self.assertRaises(ValueError):
            VarianceExplodingSDE(1.0, 1.0)
        with self.assertRaises(ValueError):
            GaussianScore(vp, (0.0, 1.0), (1.0,))
        with self.assertRaises(ValueError):
            GaussianScore(vp, 0.0, -1.0)
        with self.assertRaises(ValueError):
            perturbation_moments(vp, (0.0,), 1.5)
        with self.assertRaises(ValueError):
            reverse_sde_sample(vp, score, 0)
        with self.assertRaises(ValueError):
            reverse_sde_sample(vp, score, 3, end_time=1.0)
        with self.assertRaises(ValueError):
            reverse_sde_sample(vp, score, 3, dimension=2, steps=2, seed=1)
        with self.assertRaises(ValueError):
            predictor_corrector_sample(vp, score, 3, snr=0.0)
        with self.assertRaises(ValueError):
            predictor_corrector_sample(vp, score, 3, steps=10, seed=1)
        with self.assertRaises(ValueError):
            probability_flow_map(vp, score, [0.0], method="euler")
        with self.assertRaises(ValueError):
            probability_flow_map(vp, score, [0.0], start_time=0.5, end_time=0.5)
        with self.assertRaises(ValueError):
            probability_flow_map(vp, score, [(0.0,), (1.0, 2.0)])
        with self.assertRaises(ValueError):
            train_score_network(vp, [], iterations=1)
        with self.assertRaises(ValueError):
            train_score_network(vp, [0.0], iterations=1, time_floor=0.0)
        with self.assertRaises(ValueError):
            train_score_network(vp, [0.0], iterations=1, weighting=lambda t: -1.0)
        with self.assertRaises(ValueError):
            reverse_sde_sample(vp, lambda x, t: (math.nan,), 2, steps=2, seed=1)
        with self.assertRaises(ValueError):
            NeuralSDE(diffusion="matrix")
        with self.assertRaises(ValueError):
            NeuralSDE(dimension=0)
        with self.assertRaises(ValueError):
            fit_neural_sde(NeuralSDE(), [[0.0, 0.1, 0.2]], iterations=1)
        with self.assertRaises(ValueError):
            fit_neural_sde(NeuralSDE(), [[0.0]], step=0.1, iterations=1)
        with self.assertRaises(ValueError):
            fit_neural_sde(NeuralSDE(dimension=2), [[0.0, 0.1]], step=0.1)


if __name__ == "__main__":
    unittest.main()
