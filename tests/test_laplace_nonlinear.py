"""Tests for Laplace, RTO, RML and Laplace-preconditioned MCMC.

Oracles, in the order used: closed-form dense linear-Gaussian posterior and
evidence (``probstat.linalg`` Cholesky/inverse), a two-dimensional quadrature
grid for a mildly nonlinear two-parameter posterior (moments and evidence), and
long reference chains (plain pCN) for the 1D finite-volume elliptic problem. The
slow tier (``PROBSTAT_SLOW=1``) runs the long chains and the dimension sweep.
"""

from __future__ import annotations

import math
import os
import random
import time
import unittest

from probstat.inverse.laplace_nonlinear import (
    EvaluationCounts,
    NonlinearBayesProblem,
    compute_map,
    dili_chain,
    gpcn_chain,
    laplace_approximation,
    pcn_chain,
    randomize_then_optimize,
    randomized_maximum_likelihood,
    sample_moments,
)
from probstat.inverse.pde_models import EllipticDiffusion1D
from probstat.linalg import (
    DenseOperator,
    DiagonalCovariance,
    inverse_spd,
    log_det_spd,
    solve_spd,
)

SLOW = os.environ.get("PROBSTAT_SLOW") == "1"
Vec = tuple[float, ...]


def mat_vec(a: tuple[Vec, ...], v: Vec) -> Vec:
    """Return ``a v``."""
    return tuple(math.fsum(x * y for x, y in zip(row, v, strict=True)) for row in a)


def mat_mul(a: tuple[Vec, ...], b: tuple[Vec, ...]) -> tuple[Vec, ...]:
    """Return ``a b``."""
    cols = list(zip(*b, strict=True))
    return tuple(
        tuple(math.fsum(x * y for x, y in zip(row, col, strict=True)) for col in cols)
        for row in a
    )


def transpose(a: tuple[Vec, ...]) -> tuple[Vec, ...]:
    """Return ``a'``."""
    return tuple(zip(*a, strict=True))


class LinearModel:
    """Dense linear forward map ``G(x) = A x`` implementing the protocol."""

    def __init__(self, a: tuple[Vec, ...]) -> None:
        """Store ``A``."""
        self.a = a

    @property
    def input_dimension(self) -> int:
        """Return ``n``."""
        return len(self.a[0])

    @property
    def output_dimension(self) -> int:
        """Return ``m``."""
        return len(self.a)

    def value(self, x: object) -> Vec:
        """Return ``A x``."""
        return mat_vec(self.a, tuple(x))  # type: ignore[call-overload]

    def jvp(self, x: object, v: object) -> Vec:
        """Return ``A v``."""
        return mat_vec(self.a, tuple(v))  # type: ignore[call-overload]

    def vjp(self, x: object, w: object) -> Vec:
        """Return ``A' w``."""
        return mat_vec(transpose(self.a), tuple(w))  # type: ignore[call-overload]


class TwoParameterModel:
    """Mildly nonlinear map ``R^2 -> R^3`` with analytic derivatives."""

    input_dimension = 2
    output_dimension = 3

    def value(self, x: object) -> Vec:
        """Return ``G(x)``."""
        a, b = tuple(x)  # type: ignore[call-overload]
        return (a + 0.5 * b * b, b + 0.3 * a * a, a - b + 0.2 * a * b)

    def _jac(self, x: object) -> tuple[Vec, ...]:
        """Return the Jacobian."""
        a, b = tuple(x)  # type: ignore[call-overload]
        return ((1.0, b), (0.6 * a, 1.0), (1.0 + 0.2 * b, -1.0 + 0.2 * a))

    def jvp(self, x: object, v: object) -> Vec:
        """Return ``J v``."""
        return mat_vec(self._jac(x), tuple(v))  # type: ignore[call-overload]

    def vjp(self, x: object, w: object) -> Vec:
        """Return ``J' w``."""
        return mat_vec(transpose(self._jac(x)), tuple(w))  # type: ignore[call-overload]


def two_parameter_problem() -> NonlinearBayesProblem:
    """Return the 2-parameter problem (noise 0.5, identity prior)."""
    model = TwoParameterModel()
    truth = (0.8, -0.6)
    data = tuple(
        g + d for g, d in zip(model.value(truth), (0.3, -0.2, 0.25), strict=True)
    )
    return NonlinearBayesProblem(
        model,
        data,
        noise=DiagonalCovariance([0.25] * 3),
        prior_factor=DenseOperator(((1.0, 0.0), (0.0, 1.0))),
    )


def grid_oracle(problem: NonlinearBayesProblem) -> tuple[Vec, Vec, float]:
    """Return posterior mean, variance and log evidence by 2D quadrature."""
    half, count = 4.5, 201
    h = 2.0 * half / (count - 1)
    points: list[tuple[float, float, float]] = []
    for i in range(count):
        for j in range(count):
            z = (-half + i * h, -half + j * h)
            points.append((z[0], z[1], problem.log_posterior(z)))
    top = max(p[2] for p in points)
    weights = [math.exp(p[2] - top) for p in points]
    total = math.fsum(weights)
    mean = tuple(
        math.fsum(w * p[k] for w, p in zip(weights, points, strict=True)) / total
        for k in (0, 1)
    )
    var = tuple(
        math.fsum(
            w * (p[k] - mean[k]) ** 2 for w, p in zip(weights, points, strict=True)
        )
        / total
        for k in (0, 1)
    )
    log_integral = top + math.log(total * h * h)
    evidence = problem.log_likelihood_constant - math.log(2.0 * math.pi) + log_integral
    return mean, var, evidence


def linear_problem(
    seed: int = 4, n: int = 4, m: int = 6
) -> tuple[NonlinearBayesProblem, tuple[Vec, ...], tuple[Vec, ...], Vec, float]:
    """Return a linear problem and its dense data (A, P, y, noise sd)."""
    rng = random.Random(seed)
    a = tuple(tuple(rng.uniform(-1.0, 1.0) for _ in range(n)) for _ in range(m))
    p = tuple(
        tuple((0.8 if i == j else 0.0) + 0.15 * rng.uniform(-1, 1) for j in range(n))
        for i in range(n)
    )
    y = tuple(rng.uniform(-1.0, 1.0) for _ in range(m))
    sigma = 0.3
    problem = NonlinearBayesProblem(
        LinearModel(a),
        y,
        noise=DiagonalCovariance([sigma**2] * m),
        prior_factor=DenseOperator(p),
        prior_mean=(0.1,) * n,
    )
    return problem, a, p, y, sigma


def dense_linear_posterior(
    a: tuple[Vec, ...], p: tuple[Vec, ...], y: Vec, sigma: float, mu: float = 0.1
) -> tuple[Vec, tuple[Vec, ...], float]:
    """Return posterior mean, covariance and log evidence of a linear case."""
    n, m = len(p), len(a)
    k = mat_mul(a, p)
    info = mat_mul(transpose(k), k)
    precision = tuple(
        tuple(info[i][j] / sigma**2 + (1.0 if i == j else 0.0) for j in range(n))
        for i in range(n)
    )
    prior_mean = (mu,) * n
    shift = tuple(yi - gi for yi, gi in zip(y, mat_vec(a, prior_mean), strict=True))
    rhs = tuple(v / sigma**2 for v in mat_vec(transpose(k), shift))
    z_mean = solve_spd(precision, rhs)
    cov_z = inverse_spd(precision)
    cov_x = mat_mul(mat_mul(p, cov_z), transpose(p))
    mean_x = tuple(mu + t for t in mat_vec(p, z_mean))
    # evidence: y ~ N(A mu, A C A' + sigma^2 I)
    c = mat_mul(p, transpose(p))
    s = mat_mul(mat_mul(a, c), transpose(a))
    s = tuple(
        tuple(s[i][j] + (sigma**2 if i == j else 0.0) for j in range(m))
        for i in range(m)
    )
    quad = math.fsum(r * t for r, t in zip(shift, solve_spd(s, shift), strict=True))
    evidence = -0.5 * (m * math.log(2.0 * math.pi) + log_det_spd(s) + quad)
    return mean_x, cov_x, evidence


def elliptic_problem(
    n: int, *, sigma: float = 0.005, seed: int = 3, decay: float = 1.5
) -> NonlinearBayesProblem:
    """Return the 1D elliptic problem with a cosine (KL) log-conductivity prior.

    ``-(e^x u')' = 1`` on ``(0, 1)``, ``u(0) = u(1) = 0``, six interior sensors;
    ``x = sum_j (1 + j)^-decay sqrt(2) cos(j pi s) xi_j`` with all ``n`` modes of
    the discretization, so the continuum prior is fixed as ``n`` grows.
    """
    sensors = (0.1, 0.25, 0.4, 0.55, 0.7, 0.85)
    model = EllipticDiffusion1D(n, sensors=sensors, source=1.0)
    p = tuple(
        tuple(
            (1 + j) ** (-decay)
            * (1.0 if j == 0 else math.sqrt(2.0))
            * math.cos(j * math.pi * (i + 0.5) / n)
            for j in range(n)
        )
        for i in range(n)
    )
    rng = random.Random(seed)
    truth = mat_vec(p, tuple(rng.gauss(0.0, 1.0) for _ in range(n)))
    data = tuple(g + sigma * rng.gauss(0.0, 1.0) for g in model.value(truth))
    return NonlinearBayesProblem(
        model,
        data,
        noise=DiagonalCovariance([sigma**2] * len(sensors)),
        prior_factor=DenseOperator(p),
    )


class LinearGaussianExactnessTests(unittest.TestCase):
    """Every method is exact for a linear Gaussian problem."""

    def setUp(self) -> None:
        """Build the linear problem and its dense oracle."""
        self.problem, self.a, self.p, self.y, self.sigma = linear_problem()
        self.mean, self.cov, self.evidence = dense_linear_posterior(
            self.a, self.p, self.y, self.sigma
        )

    def test_laplace_mean_covariance_and_evidence(self) -> None:
        """Laplace equals the dense posterior; evidence equals the closed form."""
        lap = laplace_approximation(self.problem, threshold=1e-12, lis_method="lanczos")
        self.assertTrue(lap.converged, lap.lis_stop_reason)
        self.assertEqual(lap.rank, 4)
        # conditioning: cond(precision) ~ 1e1..1e2 so 1e-8 is generous
        for got, want in zip(lap.mean, self.mean, strict=True):
            self.assertAlmostEqual(got, want, delta=1e-8)
        n = len(self.mean)
        for j in range(n):
            unit = tuple(1.0 if i == j else 0.0 for i in range(n))
            column = lap.covariance_action(unit)
            for i in range(n):
                self.assertAlmostEqual(column[i], self.cov[i][j], delta=1e-8)
        self.assertAlmostEqual(lap.log_evidence, self.evidence, delta=1e-7)
        exact = lap.marginal_variances()
        for i in range(n):
            self.assertAlmostEqual(exact.values[i], self.cov[i][i], delta=1e-8)

    def test_trust_region_map_agrees(self) -> None:
        """The trust-region MAP is the same point as Gauss-Newton."""
        gn = compute_map(self.problem, method="gauss_newton")
        tr = compute_map(self.problem, method="trust_region")
        self.assertTrue(gn.converged and tr.converged, (gn.stop_reason, tr.stop_reason))
        for a, b in zip(gn.physical, tr.physical, strict=True):
            self.assertAlmostEqual(a, b, delta=1e-6)
        for a, b in zip(gn.physical, self.mean, strict=True):
            self.assertAlmostEqual(a, b, delta=1e-8)

    def test_laplace_samples_and_probing_variances(self) -> None:
        """Samples and the Bekas estimator match the dense covariance within MCSE."""
        lap = laplace_approximation(self.problem, threshold=1e-12)
        draws = 6000
        samples = lap.sample(draws, seed=11)
        mean, var = sample_moments(samples)
        for i in range(4):
            sd = math.sqrt(self.cov[i][i])
            self.assertLess(abs(mean[i] - self.mean[i]), 5.0 * sd / math.sqrt(draws))
            # the variance of the sample variance is 2 sd^4/(N-1)
            self.assertLess(
                abs(var[i] - self.cov[i][i]),
                5.0 * self.cov[i][i] * math.sqrt(2.0 / (draws - 1)),
            )
        probing = lap.marginal_variances(probes=400, seed=5)
        self.assertEqual(probing.method, "probing")
        for i in range(4):
            self.assertLess(
                abs(probing.values[i] - self.cov[i][i]),
                5.0 * probing.standard_errors[i] + 1e-12,
            )

    def test_rto_is_exact_and_weights_constant(self) -> None:
        """RTO weights are constant, every proposal is accepted, moments match."""
        draws = 1500
        result = randomize_then_optimize(self.problem, draws=draws, seed=2)
        self.assertEqual(result.solver_failures, 0)
        self.assertEqual(result.acceptance_rate, 1.0)
        spread = max(result.log_weights) - min(result.log_weights)
        self.assertLess(spread, 1e-8)  # roundoff of ||H||^2 - ||Q'H||^2
        self.assertAlmostEqual(result.weight_ess, draws, delta=1e-3)
        mean, var = sample_moments(result.samples)
        for i in range(4):
            sd = math.sqrt(self.cov[i][i])
            self.assertLess(abs(mean[i] - self.mean[i]), 5.0 * sd / math.sqrt(draws))
            self.assertLess(
                abs(var[i] - self.cov[i][i]),
                5.0 * self.cov[i][i] * math.sqrt(2.0 / (draws - 1)),
            )

    def test_rml_is_exact_for_linear(self) -> None:
        """RML draws have the posterior mean and covariance."""
        draws = 1000
        result = randomized_maximum_likelihood(self.problem, draws=draws, seed=6)
        self.assertTrue(result.approximate)
        self.assertEqual(result.failures, 0)
        mean, var = sample_moments(result.samples)
        for i in range(4):
            sd = math.sqrt(self.cov[i][i])
            self.assertLess(abs(mean[i] - self.mean[i]), 5.0 * sd / math.sqrt(draws))
            self.assertLess(
                abs(var[i] - self.cov[i][i]),
                5.0 * self.cov[i][i] * math.sqrt(2.0 / (draws - 1)),
            )

    def test_gpcn_with_full_laplace_accepts_everything(self) -> None:
        """With beta = 1 and the exact covariance the gpCN proposal is the posterior."""
        lap = laplace_approximation(self.problem, threshold=1e-12)
        chain = gpcn_chain(
            self.problem, lap, draws=3000, beta=1.0, max_beta=1.0, adapt=False, seed=3
        )
        self.assertEqual(chain.acceptance_rate, 1.0)
        for i in range(4):
            sd = math.sqrt(self.cov[i][i])
            self.assertLess(
                abs(chain.mean[i] - self.mean[i]), 5.0 * sd / math.sqrt(3000)
            )

    def test_dili_on_linear_gaussian_is_valid_and_accepts_small_steps(self) -> None:
        """Preconditioned Langevin on a Gaussian: valid moments, high acceptance.

        Exact acceptance 1 is not expected (MALA is not an exact Gaussian
        sampler); the acceptance of a small step is close to one.
        """
        lap = laplace_approximation(self.problem, threshold=1e-12)
        chain = dili_chain(self.problem, lap, draws=4000, step=0.3, adapt=False, seed=4)
        self.assertGreater(chain.acceptance_rate, 0.9)
        band = 5.0 / math.sqrt(chain.min_ess)
        for i in range(4):
            sd = math.sqrt(self.cov[i][i])
            self.assertLess(abs(chain.mean[i] - self.mean[i]), band * sd)
            self.assertLess(abs(chain.variance[i] / self.cov[i][i] - 1.0), 0.3)


class TwoParameterNonlinearTests(unittest.TestCase):
    """Mildly nonlinear two-parameter posterior against a quadrature oracle."""

    @classmethod
    def setUpClass(cls) -> None:
        """Compute the grid oracle once."""
        cls.problem = two_parameter_problem()
        cls.mean, cls.var, cls.evidence = grid_oracle(cls.problem)

    def test_derivatives_match_finite_differences(self) -> None:
        """The model's jvp and vjp agree with central differences."""
        model = TwoParameterModel()
        x = (0.3, -0.7)
        for k in range(2):
            e = tuple(1.0 if i == k else 0.0 for i in range(2))
            hi = model.value(tuple(a + 1e-6 * b for a, b in zip(x, e, strict=True)))
            lo = model.value(tuple(a - 1e-6 * b for a, b in zip(x, e, strict=True)))
            fd = tuple((p - q) / 2e-6 for p, q in zip(hi, lo, strict=True))
            for a, b in zip(model.jvp(x, e), fd, strict=True):
                self.assertAlmostEqual(a, b, delta=1e-7)

    def test_laplace_discrepancy_is_reported_and_bounded(self) -> None:
        """Laplace moments and evidence versus quadrature; print the discrepancy."""
        lap = laplace_approximation(self.problem, threshold=1e-9)
        self.assertTrue(lap.converged)
        var = lap.marginal_variances().values
        mean_err = [abs(a - b) for a, b in zip(lap.mean, self.mean, strict=True)]
        var_ratio = [a / b for a, b in zip(var, self.var, strict=True)]
        evidence_err = lap.log_evidence - self.evidence
        print(
            "\n[two-parameter] Laplace-quadrature: mean err",
            [round(v, 4) for v in mean_err],
            "sd",
            [round(math.sqrt(v), 4) for v in self.var],
            "var ratio",
            [round(v, 3) for v in var_ratio],
            "log-evidence err",
            round(evidence_err, 4),
        )
        # mild nonlinearity: the Gaussian is not exact but is close
        for i in range(2):
            self.assertLess(mean_err[i], 0.5 * math.sqrt(self.var[i]))
            self.assertGreater(var_ratio[i], 0.5)
            self.assertLess(var_ratio[i], 2.0)
        self.assertLess(abs(evidence_err), 0.5)

    def test_rto_metropolis_and_importance_match_quadrature(self) -> None:
        """RTO with a correction reproduces the quadrature moments."""
        draws = 2000
        mh = randomize_then_optimize(self.problem, draws=draws, seed=7)
        imp = randomize_then_optimize(
            self.problem, draws=draws, seed=8, correction="importance"
        )
        self.assertLess(mh.solver_failures, 0.01 * draws)
        self.assertGreater(mh.acceptance_rate, 0.5)
        mean_mh, var_mh = sample_moments(mh.samples)
        mean_is, var_is = sample_moments(imp.samples, imp.weights)
        print(
            "\n[two-parameter] RTO MH acc",
            round(mh.acceptance_rate, 3),
            "weight ESS",
            round(mh.weight_ess, 1),
            "of",
            draws,
            "| mean",
            [round(v, 4) for v in mean_mh],
            "quad",
            [round(v, 4) for v in self.mean],
            "| var ratio MH",
            [round(a / b, 3) for a, b in zip(var_mh, self.var, strict=True)],
            "IS",
            [round(a / b, 3) for a, b in zip(var_is, self.var, strict=True)],
            "| solver failures",
            mh.solver_failures,
            imp.solver_failures,
        )
        for i in range(2):
            sd = math.sqrt(self.var[i])
            # an independence chain's ESS is below N; use a wide MCSE band
            self.assertLess(
                abs(mean_mh[i] - self.mean[i]), 6.0 * sd / math.sqrt(0.2 * draws)
            )
            self.assertLess(
                abs(mean_is[i] - self.mean[i]), 6.0 * sd / math.sqrt(imp.weight_ess)
            )
            self.assertLess(abs(var_mh[i] / self.var[i] - 1.0), 0.25)
            self.assertLess(abs(var_is[i] / self.var[i] - 1.0), 0.25)

    def test_rml_comparison_is_reported(self) -> None:
        """RML versus quadrature; the discrepancy is printed, the bound is loose."""
        draws = 250
        rml = randomized_maximum_likelihood(self.problem, draws=draws, seed=9)
        self.assertTrue(rml.approximate)
        self.assertLess(rml.failures, 0.01 * draws)
        mean, var = sample_moments(rml.samples)
        print(
            "\n[two-parameter] RML mean err",
            [round(abs(a - b), 4) for a, b in zip(mean, self.mean, strict=True)],
            "var ratio",
            [round(a / b, 3) for a, b in zip(var, self.var, strict=True)],
            "failures",
            rml.failures,
        )
        for i in range(2):
            self.assertLess(abs(mean[i] - self.mean[i]), 1.0 * math.sqrt(self.var[i]))

    def test_gpcn_and_dili_agree_with_quadrature(self) -> None:
        """The two Laplace-preconditioned chains are valid samplers (nonlinear G)."""
        lap = laplace_approximation(self.problem, threshold=1e-9)
        for chain in (
            gpcn_chain(self.problem, lap, draws=6000, burn_in=500, seed=1),
            dili_chain(self.problem, lap, draws=6000, burn_in=500, seed=2),
        ):
            for i in range(2):
                sd = math.sqrt(self.var[i])
                mcse = sd / math.sqrt(max(chain.ess[1 + i], 20.0))
                self.assertLess(abs(chain.mean[i] - self.mean[i]), 6.0 * mcse + 1e-3)
                # short single chains of a non-Gaussian posterior: the variance is
                # only checked to a factor (seed sweep, 6000 draws: 0.67-1.8)
                ratio = chain.variance[i] / self.var[i]
                self.assertGreater(ratio, 0.5)
                self.assertLess(ratio, 2.0)


class EllipticProblemTests(unittest.TestCase):
    """Fast checks on the PDE-backed problem."""

    def test_counts_and_laplace_use_matrix_free_actions(self) -> None:
        """The Laplace workflow reports its model calls; the evidence is finite."""
        problem = elliptic_problem(8)
        lap = laplace_approximation(problem, threshold=0.05, seed=1)
        self.assertTrue(
            lap.converged, (lap.map_estimate.stop_reason, lap.lis_stop_reason)
        )
        self.assertGreaterEqual(lap.rank, 1)
        self.assertLess(lap.rank, 8)
        self.assertTrue(math.isfinite(lap.log_evidence))
        self.assertGreater(lap.counts_hessian.jvp, 0)
        self.assertGreater(lap.counts_hessian.vjp, 0)
        self.assertTrue(math.isfinite(lap.log_evidence_truncation_bound))
        # eigenvectors orthonormal
        for i, vi in enumerate(lap.eigenvectors):
            for j, vj in enumerate(lap.eigenvectors):
                dot = math.fsum(a * b for a, b in zip(vi, vj, strict=True))
                self.assertAlmostEqual(dot, 1.0 if i == j else 0.0, delta=1e-8)

    def test_chains_agree_with_each_other_and_report_efficiency(self) -> None:
        """pCN, gpCN and DILI give consistent means; efficiency is printed."""
        problem = elliptic_problem(8)
        lap = laplace_approximation(problem, threshold=0.05, seed=1)
        start = lap.map_estimate.whitened
        runs = {
            "pcn": pcn_chain(problem, draws=2500, burn_in=600, initial=start, seed=1),
            "gpcn": gpcn_chain(problem, lap, draws=2500, burn_in=600, seed=2),
            "dili": dili_chain(problem, lap, draws=2500, burn_in=600, seed=3),
        }
        sd = [math.sqrt(v) for v in lap.marginal_variances().values]
        for name, run in runs.items():
            print(
                f"\n[elliptic n=8] {name}: acc {run.acceptance_rate:.3f} "
                f"min ESS {run.min_ess:.1f} ESS/work {run.ess_per_work:.5f} "
                f"counts {run.sampling_counts}"
            )
            self.assertGreater(run.acceptance_rate, 0.05)
            for i in (0, 4, 7):
                self.assertLess(abs(run.mean[i] - lap.mean[i]), 5.0 * sd[i] + 1e-9)
        counts = runs["pcn"].sampling_counts
        self.assertEqual(counts.vjp, 0)
        self.assertEqual(counts.value, 2500)
        self.assertEqual(runs["dili"].sampling_counts.vjp, 2500)


class ValidationTests(unittest.TestCase):
    """Rejection of invalid input and honest failure reporting."""

    def setUp(self) -> None:
        """Build a small problem."""
        self.problem, *_ = linear_problem()

    def test_constructor_validation(self) -> None:
        """Wrong types and dimensions raise."""
        a = ((1.0, 0.0), (0.0, 1.0))
        noise = DiagonalCovariance([1.0, 1.0])
        eye = DenseOperator(a)
        with self.assertRaises(TypeError):
            NonlinearBayesProblem(object(), (0.0, 0.0), noise=noise, prior_factor=eye)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            NonlinearBayesProblem(
                LinearModel(a), (0.0, 0.0), noise=1.0, prior_factor=eye
            )
        with self.assertRaises(ValueError):
            NonlinearBayesProblem(LinearModel(a), (0.0,), noise=noise, prior_factor=eye)
        with self.assertRaises(ValueError):
            NonlinearBayesProblem(
                LinearModel(a),
                (0.0, 0.0),
                noise=noise,
                prior_factor=eye,
                prior_mean=(0.0,),
            )
        with self.assertRaises(ValueError):
            NonlinearBayesProblem(
                LinearModel(a),
                (0.0, 0.0),
                noise=noise,
                prior_factor=DenseOperator(((1.0, 0.0),)),
            )
        with self.assertRaises(ValueError):
            NonlinearBayesProblem(
                LinearModel(a), (float("nan"), 0.0), noise=noise, prior_factor=eye
            )

    def test_argument_validation(self) -> None:
        """Invalid options raise instead of being corrected."""
        with self.assertRaises(TypeError):
            laplace_approximation(object())  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            laplace_approximation(self.problem, threshold=0.0)
        with self.assertRaises(ValueError):
            compute_map(self.problem, method="newton")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            randomize_then_optimize(self.problem, draws=0)
        with self.assertRaises(ValueError):
            randomize_then_optimize(self.problem, draws=3, correction="x")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            randomized_maximum_likelihood(self.problem, draws=0)
        with self.assertRaises(ValueError):
            pcn_chain(self.problem, draws=10, beta=0.0)
        with self.assertRaises(ValueError):
            pcn_chain(self.problem, draws=10, thin=0)
        with self.assertRaises(ValueError):
            pcn_chain(self.problem, draws=10, target_acceptance=1.5)
        lap = laplace_approximation(self.problem, threshold=1e-9)
        with self.assertRaises(ValueError):
            lap.sample(0)
        with self.assertRaises(ValueError):
            lap.marginal_variances(probes=1)
        with self.assertRaises(ValueError):
            lap.covariance_action((1.0,))
        with self.assertRaises(TypeError):
            gpcn_chain(self.problem, None, draws=10)  # type: ignore[arg-type]
        other, *_ = linear_problem(seed=9)
        with self.assertRaises(ValueError):
            gpcn_chain(other, lap, draws=10)
        with self.assertRaises(ValueError):
            sample_moments([(1.0,)])

    def test_chain_rejects_infeasible_start(self) -> None:
        """A start with a failing forward model is refused."""

        class Fragile(TwoParameterModel):
            """Model returning NaN at large input."""

            def value(self, x: object) -> Vec:
                """Return NaN far from the origin."""
                if abs(tuple(x)[0]) > 1.0:  # type: ignore[call-overload]
                    return (math.nan, math.nan, math.nan)
                return super().value(x)

        problem = NonlinearBayesProblem(
            Fragile(),
            (0.1, 0.2, 0.3),
            noise=DiagonalCovariance([1.0] * 3),
            prior_factor=DenseOperator(((1.0, 0.0), (0.0, 1.0))),
        )
        with self.assertRaises(ValueError):
            pcn_chain(problem, draws=5, initial=(5.0, 0.0))
        # a chain that wanders into the failure region records the rejection
        run = pcn_chain(problem, draws=400, beta=0.9, adapt=False, seed=1)
        self.assertGreater(run.nonfinite_rejections, 0)
        self.assertTrue(all(math.isfinite(v) for v in run.log_likelihoods))

    def test_evaluation_counts_helpers(self) -> None:
        """Count arithmetic and the cost unit."""
        a = EvaluationCounts(5, 3, 2)
        b = EvaluationCounts(1, 1, 1)
        self.assertEqual(a.minus(b), EvaluationCounts(4, 2, 1))
        self.assertEqual(a.work, 10)

    def test_non_converged_map_is_reported(self) -> None:
        """An iteration cap of 1 is flagged, not hidden."""
        problem = two_parameter_problem()
        result = compute_map(problem, max_iterations=1, gradient_tolerance=1e-14)
        self.assertFalse(result.converged)
        lap = laplace_approximation(problem, map_estimate=result, threshold=1e-9)
        self.assertFalse(lap.converged)


@unittest.skipUnless(SLOW, "slow tier: set PROBSTAT_SLOW=1")
class SlowStudies(unittest.TestCase):
    """Long reference comparison and the dimension-robustness sweep."""

    def test_laplace_versus_long_reference_chain(self) -> None:
        """Laplace moments of the 1D elliptic case against a long pCN reference."""
        problem = elliptic_problem(16, sigma=0.02)
        lap = laplace_approximation(problem, threshold=0.05, seed=1)
        t0 = time.perf_counter()
        ref = pcn_chain(
            problem,
            draws=150_000,
            burn_in=5_000,
            initial=lap.map_estimate.whitened,
            seed=21,
            store=False,
            thin=5,
        )
        var = lap.marginal_variances().values
        print(
            f"\n[slow elliptic n=16] reference pCN: {time.perf_counter() - t0:.1f} s,"
            f" acc {ref.acceptance_rate:.3f}, min ESS {ref.min_ess:.0f},"
            f" rank {lap.rank}"
        )
        worst = 0.0
        for i in range(16):
            sd_ref = math.sqrt(ref.variance[i])
            bias = abs(lap.mean[i] - ref.mean[i]) / sd_ref
            ratio = var[i] / ref.variance[i]
            worst = max(worst, bias)
            if i % 3 == 0:
                print(
                    f"  cell {i}: mean MCMC {ref.mean[i]:+.4f}"
                    f" Laplace {lap.mean[i]:+.4f}"
                    f" (|diff|/sd {bias:.3f}); var ratio {ratio:.3f}"
                )
        print(f"  max |mean diff|/sd = {worst:.3f}")
        self.assertLess(worst, 1.0)

    def test_dimension_robustness(self) -> None:
        """Plain pCN versus gpCN versus DILI at two discretizations."""
        rows = []
        for n in (16, 64):
            problem = elliptic_problem(n, sigma=0.01)
            lap = laplace_approximation(problem, threshold=0.5, seed=1)
            start = lap.map_estimate.whitened
            draws, burn = 20_000, 3_000
            for name, run in (
                (
                    "pcn",
                    pcn_chain(
                        problem,
                        draws=draws,
                        burn_in=burn,
                        initial=start,
                        seed=1,
                        store=False,
                    ),
                ),
                (
                    "gpcn",
                    gpcn_chain(
                        problem, lap, draws=draws, burn_in=burn, seed=2, store=False
                    ),
                ),
                (
                    "dili",
                    dili_chain(
                        problem, lap, draws=draws, burn_in=burn, seed=3, store=False
                    ),
                ),
            ):
                rows.append((n, name, lap.rank, run))
                print(
                    f"\n[slow dimension n={n}] {name}: rank {lap.rank}"
                    f" acc {run.acceptance_rate:.3f} beta {run.beta:.3f}"
                    f" step {run.step:.3f} min ESS {run.min_ess:.1f}"
                    f" ESS/work {run.ess_per_work:.5f}"
                    f" ESS/vjp {run.ess_per_gradient:.5f} counts {run.sampling_counts}"
                )
        for n, name, _rank, run in rows:
            self.assertGreater(run.acceptance_rate, 0.05, (n, name))


if __name__ == "__main__":
    unittest.main()
