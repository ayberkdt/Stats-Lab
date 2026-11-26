"""Tests of the prior-whitened posterior targets and sampler adapters.

Experimental (added in 0.28).

References:
    [EQ-INV-TGT-001], [EQ-INV-TGT-002], [EQ-INV-TGT-003], [EQ-INV-TGT-004].

"""

from __future__ import annotations

import math
import unittest
from collections.abc import Sequence

from probstat.inverse import (
    CallbackForward,
    correlated_gaussian_benchmark,
    linear_gaussian_posterior,
)
from probstat.inverse.posterior_targets import (
    SBC_CAVEAT,
    WhitenedGaussianPriorTarget,
    physical_posterior,
    sample_hmc,
    sample_pcn,
    simulation_based_calibration_linear_gaussian,
)
from probstat.linalg import (
    DenseCovariance,
    DenseOperator,
    DiagonalCovariance,
    Vector,
)
from probstat.optimization import minimize_bfgs

_A = (
    (1.0, 0.5, -0.3),
    (0.2, -1.0, 0.4),
    (0.7, 0.1, 0.9),
    (-0.4, 0.6, 0.3),
)
_P_SQUARE = ((1.0, 0.0, 0.0), (0.4, 0.8, 0.0), (-0.2, 0.3, 0.6))
_P_RECT = ((1.0, 0.0), (0.4, 0.8), (-0.2, 0.3))


def _matvec(matrix: Sequence[Sequence[float]], v: Sequence[float]) -> Vector:
    """Return ``matrix v``."""
    return tuple(
        math.fsum(a * b for a, b in zip(row, v, strict=True)) for row in matrix
    )


def _rmatvec(matrix: Sequence[Sequence[float]], w: Sequence[float]) -> Vector:
    """Return ``matrix' w``."""
    return tuple(
        math.fsum(matrix[i][j] * w[i] for i in range(len(matrix)))
        for j in range(len(matrix[0]))
    )


def _exponential_model() -> CallbackForward:
    """Return ``G(x) = exp(A x)`` (componentwise) with analytic derivatives."""

    def value(x: Vector) -> Vector:
        """Return ``exp(A x)``."""
        return tuple(math.exp(v) for v in _matvec(_A, x))

    def jvp(x: Vector, v: Vector) -> Vector:
        """Return ``diag(exp(A x)) A v``."""
        g = value(x)
        return tuple(a * b for a, b in zip(g, _matvec(_A, v), strict=True))

    def vjp(x: Vector, w: Vector) -> Vector:
        """Return ``A' diag(exp(A x)) w``."""
        g = value(x)
        return _rmatvec(_A, tuple(a * b for a, b in zip(g, w, strict=True)))

    return CallbackForward(3, 4, value, jvp, vjp)


def _linear_model(matrix: Sequence[Sequence[float]]) -> CallbackForward:
    """Return ``G(x) = M x`` with the exact Jacobian ``M``."""
    rows = [tuple(r) for r in matrix]
    return CallbackForward(
        len(rows[0]), len(rows), lambda x: _matvec(rows, x), jacobian=lambda x: rows
    )


class TargetContractTests(unittest.TestCase):
    """The density, the gradient and the coordinate map."""

    def test_gradient_matches_central_differences(self) -> None:
        """The gradient of log_density equals finite differences."""
        noise = DiagonalCovariance([0.04, 0.09, 0.05, 0.02])
        data = (1.1, 0.7, 1.9, 0.8)
        for factor, points in (
            (_P_SQUARE, ((0.3, -0.2, 0.5), (-0.6, 0.4, 0.1))),
            (_P_RECT, ((0.3, -0.2), (-0.6, 0.4))),
        ):
            target = WhitenedGaussianPriorTarget(
                _exponential_model(),
                data,
                noise=noise,
                prior_mean=(0.1, -0.1, 0.2),
                prior_factor=DenseOperator(factor),
            )
            for z in points:
                analytic = target.gradient(z)
                for i in range(target.dimension):
                    h = 1e-6
                    up = tuple(v + (h if j == i else 0.0) for j, v in enumerate(z))
                    down = tuple(v - (h if j == i else 0.0) for j, v in enumerate(z))
                    fd = (target.log_density(up) - target.log_density(down)) / (2 * h)
                    # Central difference error ~ h^2 f''' + roundoff/h ~ 1e-8.
                    self.assertAlmostEqual(analytic[i], fd, delta=1e-6 * (1 + abs(fd)))

    def test_density_pieces_and_constants(self) -> None:
        """log_density = log_likelihood + log_prior; constants are as documented."""
        noise = DiagonalCovariance([0.25, 0.25])
        target = WhitenedGaussianPriorTarget(
            _linear_model(((1.0, 0.0), (0.0, 1.0))),
            (1.0, 2.0),
            noise=noise,
            prior_factor=DenseOperator(((1.0, 0.0), (0.0, 1.0))),
        )
        z = (0.5, -1.0)
        # W = 1 / 0.5 = 2; residual = (-0.5, -3.0); whitened = (-1, -6).
        self.assertAlmostEqual(target.log_likelihood(z), -0.5 * (1.0 + 36.0))
        self.assertAlmostEqual(
            target.log_prior(z), -0.5 * 1.25 - math.log(2.0 * math.pi)
        )
        self.assertAlmostEqual(
            target.log_density(z), target.log_likelihood(z) + target.log_prior(z)
        )
        expected = -0.5 * (2 * math.log(2.0 * math.pi) + 2 * math.log(0.25))
        self.assertAlmostEqual(target.log_likelihood_constant, expected)
        self.assertEqual(target.to_physical(z), z)

    def test_counts_and_cache(self) -> None:
        """A gradient at the point of a density call reuses the forward value."""
        target = WhitenedGaussianPriorTarget(
            _exponential_model(),
            (1.0, 1.0, 1.0, 1.0),
            noise=DiagonalCovariance([0.1] * 4),
            prior_factor=DenseOperator(_P_SQUARE),
        )
        z = (0.1, 0.2, 0.3)
        target.log_density(z)
        target.gradient(z)
        counts = target.counts
        self.assertEqual(
            (counts.log_likelihood, counts.gradient, counts.forward, counts.vjp),
            (1, 1, 1, 1),
        )

    def test_failed_forward_is_zero_likelihood(self) -> None:
        """Non-finite forward values give -inf density and a ValueError gradient."""
        model = CallbackForward(
            1, 1, lambda x: (math.nan,), jacobian=lambda x: ((1.0,),)
        )
        target = WhitenedGaussianPriorTarget(
            model,
            (0.0,),
            noise=DiagonalCovariance([1.0]),
            prior_factor=DenseOperator(((1.0,),)),
        )
        self.assertEqual(target.log_density((0.0,)), -math.inf)
        with self.assertRaises(ValueError):
            target.gradient((0.0,))

    def test_invalid_inputs_raise(self) -> None:
        """Wrong types, dimensions and non-finite values are rejected."""
        model = _linear_model(((1.0, 0.0), (0.0, 1.0)))
        noise = DiagonalCovariance([1.0, 1.0])
        factor = DenseOperator(((1.0, 0.0), (0.0, 1.0)))
        with self.assertRaises(TypeError):
            WhitenedGaussianPriorTarget(
                object(),  # type: ignore[arg-type]
                (0.0, 0.0),
                noise=noise,
                prior_factor=factor,
            )
        with self.assertRaises(ValueError):
            WhitenedGaussianPriorTarget(model, (0.0,), noise=noise, prior_factor=factor)
        with self.assertRaises(ValueError):
            WhitenedGaussianPriorTarget(
                model, (0.0, math.nan), noise=noise, prior_factor=factor
            )
        with self.assertRaises(TypeError):
            WhitenedGaussianPriorTarget(
                model, (0.0, 0.0), noise=[1.0, 1.0], prior_factor=factor
            )
        with self.assertRaises(ValueError):
            WhitenedGaussianPriorTarget(
                model,
                (0.0, 0.0),
                noise=DiagonalCovariance([1.0]),
                prior_factor=factor,
            )
        with self.assertRaises(ValueError):
            WhitenedGaussianPriorTarget(
                model,
                (0.0, 0.0),
                noise=noise,
                prior_mean=(0.0,),
                prior_factor=factor,
            )
        with self.assertRaises(ValueError):
            WhitenedGaussianPriorTarget(
                model,
                (0.0, 0.0),
                noise=noise,
                prior_factor=DenseOperator(((1.0,), (0.0,), (0.0,))),
            )
        target = WhitenedGaussianPriorTarget(
            model, (0.0, 0.0), noise=noise, prior_factor=factor
        )
        with self.assertRaises(ValueError):
            target.log_density((0.0,))
        with self.assertRaises(ValueError):
            target.gradient((0.0, math.inf))
        with self.assertRaises(TypeError):
            sample_pcn(object(), draws=10, warmup=0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            sample_hmc(target, draws=10, warmup=10, method="rk4")  # type: ignore[arg-type]


def _gaussian_problem() -> tuple[WhitenedGaussianPriorTarget, object, Vector]:
    """Return a 2-D linear Gaussian target with a nonzero mean, its exact posterior."""
    matrix = ((1.0, 0.5), (0.3, -0.8), (0.6, 0.6))
    noise = DenseCovariance([[0.09, 0.02, 0.0], [0.02, 0.16, 0.0], [0.0, 0.0, 0.25]])
    prior = DenseCovariance([[1.0, 0.3], [0.3, 0.5]])
    mean = (0.4, -0.3)
    data = (0.9, 0.8, -0.2)
    target = WhitenedGaussianPriorTarget(
        _linear_model(matrix),
        data,
        noise=noise,
        prior_mean=mean,
        prior_factor=prior,
    )
    exact = linear_gaussian_posterior(
        DenseOperator(matrix),
        data,
        noise=noise,
        prior_mean=mean,
        prior_covariance=prior,
        method="parameter_space",
    )
    return target, exact, mean


class LinearGaussianAgreementTests(unittest.TestCase):
    """The target reproduces the exact linear-Gaussian posterior."""

    def test_maximizer_is_exact_posterior_mean(self) -> None:
        """BFGS on -log_density finds the exact whitened posterior mean."""
        target, exact, _ = _gaussian_problem()
        result = minimize_bfgs(
            lambda z: -target.log_density(z),
            (0.0, 0.0),
            gradient=lambda z: tuple(-g for g in target.gradient(z)),
            gradient_tolerance=1e-10,
        )
        self.assertTrue(result.converged)
        # Quadratic with condition number < 50, gradient tolerance 1e-10.
        for a, b in zip(result.x, exact.whitened_mean, strict=True):  # type: ignore[attr-defined]
            self.assertAlmostEqual(a, b, delta=1e-7)
        for a, b in zip(target.to_physical(result.x), exact.mean, strict=True):  # type: ignore[attr-defined]
            self.assertAlmostEqual(a, b, delta=1e-7)

    def test_pcn_adapter_mean_within_mcse(self) -> None:
        """The pCN adapter on the whitened target recovers the exact posterior mean."""
        target, exact, _ = _gaussian_problem()
        result = sample_pcn(
            target,
            draws=3000,
            warmup=500,
            chains=4,
            seed=20240607,
            check_convergence=False,
        )
        physical = physical_posterior(target, result)
        exact_mean = exact.mean  # type: ignore[attr-defined]
        for summary, truth in zip(physical.summary(), exact_mean, strict=True):
            # 5 MCSE: fixed seed; MCSE is estimated, so the band is generous.
            self.assertLess(
                abs(summary.mean - truth), 5.0 * summary.mcse_mean, summary.name
            )
            self.assertGreater(summary.effective_sample_size, 100.0)
        self.assertGreater(result.acceptance_rate, 0.05)
        # Only the likelihood is evaluated: one call per proposal plus starts.
        self.assertEqual(target.counts.log_likelihood, result.evaluations)

    def test_nuts_adapter_mean_within_mcse(self) -> None:
        """NUTS with the analytic adjoint gradient recovers the exact mean."""
        target, exact, _ = _gaussian_problem()
        result = sample_hmc(
            target, draws=600, warmup=300, chains=2, seed=7, check_convergence=False
        )
        physical = physical_posterior(target, result)
        exact_mean = exact.mean  # type: ignore[attr-defined]
        for summary, truth in zip(physical.summary(), exact_mean, strict=True):
            self.assertLess(
                abs(summary.mean - truth), 5.0 * summary.mcse_mean, summary.name
            )
        self.assertGreater(target.counts.vjp, 0)
        self.assertEqual(physical.names, ("x0", "x1"))

    def test_static_hmc_runs(self) -> None:
        """Static HMC with jitter-free default runs and keeps finite draws."""
        target, exact, _ = _gaussian_problem()
        result = sample_hmc(
            target,
            draws=200,
            warmup=150,
            chains=2,
            method="hmc",
            leapfrog_steps=8,
            seed=3,
            check_convergence=False,
        )
        pooled = result.mcmc.pooled()
        self.assertEqual(len(pooled), 400)
        self.assertTrue(all(math.isfinite(v) for s in pooled for v in s))
        self.assertIsNotNone(exact)


class SBCTests(unittest.TestCase):
    """Simulation-based calibration in the linear-Gaussian case."""

    @staticmethod
    def _problem() -> tuple[DenseOperator, object, Vector, DenseCovariance]:
        """Return an L06-like problem with n = 4 and m = 2."""
        bench = correlated_gaussian_benchmark(
            size=4, observations=2, noise_level=0.5, seed=5
        )
        assert bench.prior_covariance is not None
        return (
            bench.operator,  # type: ignore[return-value]
            bench.noise,
            (0.0,) * 4,
            bench.prior_covariance,
        )

    def test_exact_sampler_is_calibrated(self) -> None:
        """Exact conditional simulation does not reject rank uniformity.

        Screening heuristic: all p-values above 1e-3 for the fixed seed (a
        correct sampler fails this with probability about 4e-3 per seed, so
        the seed was fixed in advance, not tuned).
        """
        operator, noise, mean, prior = self._problem()
        result = simulation_based_calibration_linear_gaussian(
            operator,
            noise,
            mean,
            prior,
            replicates=60,
            posterior_draws=50,
            seed=2024,
        )
        self.assertEqual(result.sampler, "exact")
        self.assertEqual(len(result.ranks), 4)
        self.assertEqual(len(result.ranks[0]), 60)
        self.assertTrue(all(0 <= r <= 50 for col in result.ranks for r in col))
        self.assertTrue(result.uniform_at(1e-3), result.p_values)
        self.assertEqual(result.caveat, SBC_CAVEAT)
        again = simulation_based_calibration_linear_gaussian(
            operator, noise, mean, prior, replicates=60, posterior_draws=50, seed=2024
        )
        self.assertEqual(again.ranks, result.ranks)

    def test_overconfident_sampler_is_detected(self) -> None:
        """A posterior with covariance scaled by 0.25 is rejected."""
        operator, noise, mean, prior = self._problem()

        def narrow(
            data: Vector, count: int, stream: object
        ) -> Sequence[Sequence[float]]:
            """Draw from mean + 0.5 (x - mean): covariance scaled by 0.25."""
            exact = linear_gaussian_posterior(
                operator,
                data,
                noise=noise,
                prior_mean=mean,
                prior_covariance=prior,
            )
            centre = exact.mean
            return tuple(
                tuple(c + 0.5 * (v - c) for v, c in zip(draw, centre, strict=True))
                for draw in exact.sample(count, seed=stream)  # type: ignore[arg-type]
            )

        result = simulation_based_calibration_linear_gaussian(
            operator,
            noise,
            mean,
            prior,
            replicates=60,
            posterior_draws=50,
            seed=2024,
            sampler=narrow,
        )
        self.assertEqual(result.sampler, "custom")
        self.assertLess(result.min_p_value(), 1e-3, result.p_values)
        self.assertFalse(result.uniform_at(1e-3))

    def test_invalid_inputs_raise(self) -> None:
        """Bad counts, dimensions, sampler and level are rejected."""
        operator, noise, mean, prior = self._problem()
        base = {"replicates": 10, "posterior_draws": 5, "seed": 1}
        with self.assertRaises(ValueError):
            simulation_based_calibration_linear_gaussian(
                operator, noise, mean, prior, **{**base, "posterior_draws": 0}
            )
        with self.assertRaises(ValueError):
            simulation_based_calibration_linear_gaussian(
                operator, noise, mean, prior, **{**base, "replicates": 1}
            )
        with self.assertRaises(ValueError):
            simulation_based_calibration_linear_gaussian(
                operator, noise, (0.0,) * 3, prior, **base
            )
        with self.assertRaises(TypeError):
            simulation_based_calibration_linear_gaussian(
                operator, [1.0, 1.0], mean, prior, **base
            )
        with self.assertRaises(ValueError):
            simulation_based_calibration_linear_gaussian(
                operator,
                noise,
                mean,
                prior,
                sampler="bogus",
                **base,  # type: ignore[arg-type]
            )

        def short(data: Vector, count: int, stream: object) -> tuple[Vector, ...]:
            """Return draws of the wrong dimension."""
            return ((0.0,) * 3,) * count

        with self.assertRaises(ValueError):
            simulation_based_calibration_linear_gaussian(
                operator,
                noise,
                mean,
                prior,
                sampler=short,
                **base,  # type: ignore[arg-type]
            )
        ok = simulation_based_calibration_linear_gaussian(
            operator, noise, mean, prior, **base
        )
        with self.assertRaises(ValueError):
            ok.uniform_at(1.5)


if __name__ == "__main__":
    unittest.main()
