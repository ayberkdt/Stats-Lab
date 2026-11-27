"""Tests for the end-to-end ``InverseStudy`` workflow.

Oracles, in the order used: the closed-form dense linear-Gaussian posterior,
evidence and resolution matrix (``probstat.linalg`` Cholesky/inverse), a brute
force two-dimensional grid for a mildly nonlinear two-parameter posterior, and
a long reference chain on the 1D elliptic problem (slow tier,
``PROBSTAT_SLOW=1``). Stochastic checks use fixed seeds and bands in units of
the Monte Carlo standard error or of the reference standard deviation.
"""

from __future__ import annotations

import math
import os
import random
import unittest

from probstat.inverse import (
    EvaluationCounts,
    LinearInverseProblem,
    NonlinearBayesProblem,
)
from probstat.inverse.pde_models import EllipticDiffusion1D
from probstat.linalg import (
    DenseOperator,
    DiagonalCovariance,
    inverse_spd,
    log_det_spd,
    solve_spd,
)
from probstat.workflow import inverse_study as module
from probstat.workflow.inverse_study import InverseStudy

SLOW = os.environ.get("PROBSTAT_SLOW") == "1"
Vec = tuple[float, ...]
Mat = tuple[Vec, ...]


def mat_vec(a: Mat, v: Vec) -> Vec:
    """Return ``a v``."""
    return tuple(math.fsum(x * y for x, y in zip(row, v, strict=True)) for row in a)


def mat_mul(a: Mat, b: Mat) -> Mat:
    """Return ``a b``."""
    cols = list(zip(*b, strict=True))
    return tuple(
        tuple(math.fsum(x * y for x, y in zip(row, col, strict=True)) for col in cols)
        for row in a
    )


def transpose(a: Mat) -> Mat:
    """Return ``a'``."""
    return tuple(zip(*a, strict=True))


class TwoParameterModel:
    """Mildly nonlinear map ``R^2 -> R^3`` with analytic derivatives."""

    input_dimension = 2
    output_dimension = 3

    def value(self, x: object) -> Vec:
        """Return ``G(x)``."""
        a, b = tuple(x)  # type: ignore[call-overload]
        return (a + 0.5 * b * b, b + 0.3 * a * a, a - b + 0.2 * a * b)

    def _jac(self, x: object) -> Mat:
        """Return the Jacobian."""
        a, b = tuple(x)  # type: ignore[call-overload]
        return ((1.0, b), (0.6 * a, 1.0), (1.0 + 0.2 * b, -1.0 + 0.2 * a))

    def jvp(self, x: object, v: object) -> Vec:
        """Return ``J v``."""
        return mat_vec(self._jac(x), tuple(v))  # type: ignore[call-overload]

    def vjp(self, x: object, w: object) -> Vec:
        """Return ``J' w``."""
        return mat_vec(transpose(self._jac(x)), tuple(w))  # type: ignore[call-overload]


def two_parameter_study(seed: int = 5) -> InverseStudy:
    """Return a study of the two-parameter problem (noise sd 0.5, unit prior)."""
    model = TwoParameterModel()
    truth = (0.8, -0.6)
    data = tuple(
        g + d for g, d in zip(model.value(truth), (0.3, -0.2, 0.25), strict=True)
    )
    return InverseStudy(
        model,
        seed=seed,
        data=data,
        noise=DiagonalCovariance([0.25] * 3),
        prior_factor=DenseOperator(((1.0, 0.0), (0.0, 1.0))),
    )


def grid_oracle(problem: NonlinearBayesProblem) -> tuple[Vec, Vec, float]:
    """Return posterior mean, variance and log evidence by a brute-force grid."""
    half, count = 4.5, 121
    h = 2.0 * half / (count - 1)
    points = []
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
    evidence = (
        problem.log_likelihood_constant
        - math.log(2.0 * math.pi)
        + top
        + math.log(total * h * h)
    )
    return mean, var, evidence


# ---------------------------------------------------------------------------
# linear-Gaussian oracle
# ---------------------------------------------------------------------------
SIGMA = 0.3
MU = 0.1


def linear_fixture(
    seed: int = 4, n: int = 4, m: int = 6
) -> tuple[Mat, Mat, Vec, LinearInverseProblem]:
    """Return ``(A, P, y, problem)`` of a small linear-Gaussian problem."""
    rng = random.Random(seed)
    a = tuple(tuple(rng.uniform(-1.0, 1.0) for _ in range(n)) for _ in range(m))
    p = tuple(
        tuple((0.8 if i == j else 0.0) + 0.15 * rng.uniform(-1, 1) for j in range(n))
        for i in range(n)
    )
    y = tuple(rng.uniform(-1.0, 1.0) for _ in range(m))
    problem = LinearInverseProblem.create(
        a, y, noise=DiagonalCovariance([SIGMA**2] * m), reference=(MU,) * n
    )
    return a, p, y, problem


def linear_study(seed: int = 7, **kwargs: object) -> InverseStudy:
    """Return a study of the linear fixture with the dense prior ``P``."""
    _, p, _, problem = linear_fixture()
    return InverseStudy(
        problem,
        seed=seed,
        prior_factor=DenseOperator(p),
        prior_mean=(MU,) * 4,
        **kwargs,  # type: ignore[arg-type]
    )


class LinearOracle:
    """Dense posterior mean, covariance, evidence and resolution of the fixture."""

    def __init__(self) -> None:
        """Compute the closed forms."""
        a, p, y, _ = linear_fixture()
        n, m = len(p), len(a)
        k = mat_mul(a, p)
        info = mat_mul(transpose(k), k)
        precision = tuple(
            tuple(info[i][j] / SIGMA**2 + (1.0 if i == j else 0.0) for j in range(n))
            for i in range(n)
        )
        shift = tuple(yi - gi for yi, gi in zip(y, mat_vec(a, (MU,) * n), strict=True))
        rhs = tuple(v / SIGMA**2 for v in mat_vec(transpose(k), shift))
        cov_z = inverse_spd(precision)
        self.mean = tuple(MU + t for t in mat_vec(p, solve_spd(precision, rhs)))
        self.cov = mat_mul(mat_mul(p, cov_z), transpose(p))
        s = mat_mul(mat_mul(a, mat_mul(p, transpose(p))), transpose(a))
        s = tuple(
            tuple(s[i][j] + (SIGMA**2 if i == j else 0.0) for j in range(m))
            for i in range(m)
        )
        quad = math.fsum(r * t for r, t in zip(shift, solve_spd(s, shift), strict=True))
        self.evidence = -0.5 * (m * math.log(2.0 * math.pi) + log_det_spd(s) + quad)
        # R = C_post A' Gamma^-1 A, the derivative of the posterior mean in x_true
        self.resolution = mat_mul(
            self.cov,
            tuple(tuple(v / SIGMA**2 for v in row) for row in mat_mul(transpose(a), a)),
        )
        self.info_trace = math.fsum(info[i][i] for i in range(n)) / SIGMA**2
        self.info_logdet = log_det_spd(precision)
        self.n = n


class LinearGaussianEndToEndTests(unittest.TestCase):
    """Every stage of a linear-Gaussian study against the dense oracle."""

    @classmethod
    def setUpClass(cls) -> None:
        """Run all stages once."""
        cls.oracle = LinearOracle()
        cls.study = linear_study()
        s = cls.study
        s.spectrum()
        s.regularize()
        s.map()
        s.laplace(threshold=1e-10)
        s.sample("dili", draws=3000, burn_in=500)
        s.resolution(trace_probes=60)
        s.compare()

    def test_spectrum_matches_trace_and_determinant(self) -> None:
        """Sum sigma^2 = ||K||_F^2 and prod (1 + sigma^2) = det precision."""
        result = self.study.latest("spectrum").result
        # both identities are exact; the SVD error is ~1e-15 relative, cond < 1e2
        self.assertAlmostEqual(
            math.fsum(result.eigenvalues), self.oracle.info_trace, delta=1e-9
        )
        logdet = math.fsum(math.log1p(v) for v in result.eigenvalues)
        self.assertAlmostEqual(logdet, self.oracle.info_logdet, delta=1e-9)
        self.assertEqual(result.rank, 4)
        self.assertEqual(result.point, "linear")
        self.assertEqual(result.noise_whitened_rank, 4)
        self.assertEqual(
            result.detectable, sum(1 for v in result.eigenvalues if v > 1.0)
        )
        self.assertGreater(result.out_of_range_norm, 0.0)

    def test_map_and_laplace_equal_the_exact_posterior(self) -> None:
        """MAP, Laplace marginal variances and evidence equal the closed form."""
        mapped = self.study.latest("map").result
        for got, want in zip(mapped.physical, self.oracle.mean, strict=True):
            self.assertAlmostEqual(got, want, delta=1e-8)
        lap = self.study.latest("laplace").result
        self.assertTrue(lap.converged)
        self.assertEqual(lap.rank, 4)
        variances = lap.marginal_variances().values
        for i in range(4):
            self.assertAlmostEqual(variances[i], self.oracle.cov[i][i], delta=1e-8)
        # conditioning of the precision is O(1e1..1e2); 1e-7 is generous
        self.assertAlmostEqual(lap.log_evidence, self.oracle.evidence, delta=1e-7)

    def test_laplace_stage_used_the_stored_map(self) -> None:
        """The laplace record did not repeat the MAP solve."""
        record = self.study.latest("laplace")
        self.assertEqual(
            record.result.counts_map, self.study.latest("map").result.counts
        )
        self.assertNotIn("the MAP was computed inside this stage", record.warnings)

    def test_regularization_choices_are_grid_minimisers_of_the_solution(self) -> None:
        """The selected Tikhonov solutions solve the normal equations."""
        a, _, y, _ = linear_fixture()
        record = self.study.latest("regularize")
        path = record.result
        self.assertIn("gcv", record.settings["rules"])
        self.assertEqual(len(record.tables[0].rows), 4)
        statuses = {row[0]: row[2] for row in record.tables[0].rows}
        self.assertEqual(statuses["gcv"], "interior")
        self.assertEqual(statuses["upre"], "interior")
        # x_lambda = x_ref + (A'A/s^2 + lam^2 I)^-1 A'(y - A x_ref)/s^2
        lam = float(record.summary["lambda (gcv)"])
        n = 4
        ref = (MU,) * n
        info = mat_mul(transpose(a), a)
        system = tuple(
            tuple(info[i][j] / SIGMA**2 + (lam**2 if i == j else 0.0) for j in range(n))
            for i in range(n)
        )
        resid = tuple(yi - gi for yi, gi in zip(y, mat_vec(a, ref), strict=True))
        rhs = tuple(v / SIGMA**2 for v in mat_vec(transpose(a), resid))
        want = tuple(r + t for r, t in zip(ref, solve_spd(system, rhs), strict=True))
        index = min(
            range(len(path.regularizations)),
            key=lambda i: abs(path.regularizations[i] - lam),
        )
        for got, expected in zip(path.estimates[index], want, strict=True):
            self.assertAlmostEqual(got, expected, delta=1e-8)

    def test_sampler_agrees_with_the_exact_posterior(self) -> None:
        """DILI moments match the closed form within the MCSE-aware band."""
        samples = self.study.latest("sample").result
        self.assertTrue(samples.asymptotically_exact)
        self.assertEqual(samples.mcse_kind, "ess")
        self.assertGreater(min(samples.ess), 200.0)
        for i in range(4):
            sd = math.sqrt(self.oracle.cov[i][i])
            # a 5 MCSE band: probability of a false alarm ~ 6e-7 per coordinate
            self.assertLess(
                abs(samples.mean[i] - self.oracle.mean[i]), 5.0 * samples.mcse[i]
            )
            self.assertLess(abs(samples.mean[i] - self.oracle.mean[i]), 0.15 * sd)
            self.assertLess(
                abs(samples.variance[i] / self.oracle.cov[i][i] - 1.0), 0.25
            )

    def test_compare_against_exact_posterior(self) -> None:
        """The comparison stage passes and reports the exact evidence."""
        comparison = self.study.latest("compare").result
        self.assertEqual(comparison.reference_kind, "exact")
        self.assertTrue(comparison.passed, comparison.failures)
        for i in range(4):
            self.assertAlmostEqual(
                comparison.reference_mean[i], self.oracle.mean[i], delta=1e-8
            )
            self.assertAlmostEqual(
                comparison.reference_variance[i], self.oracle.cov[i][i], delta=1e-8
            )
        self.assertAlmostEqual(
            comparison.reference_log_evidence, self.oracle.evidence, delta=1e-8
        )
        # Laplace is exact for a linear model: errors at rounding level
        assert comparison.laplace_mean_error_sd is not None
        assert comparison.laplace_variance_ratio is not None
        self.assertLess(max(comparison.laplace_mean_error_sd), 1e-7)
        for ratio in comparison.laplace_variance_ratio:
            self.assertAlmostEqual(ratio, 1.0, delta=1e-7)

    def test_resolution_matches_the_resolution_matrix(self) -> None:
        """R_jj, the peak and tr R equal the dense resolution matrix."""
        result = self.study.latest("resolution").result
        for k, j in enumerate(result.indices):
            # CG tolerance 1e-10 on a well-conditioned 4x4 system
            self.assertAlmostEqual(
                result.diagonal[k], self.oracle.resolution[j][j], delta=1e-7
            )
            column = [self.oracle.resolution[i][j] for i in range(4)]
            self.assertEqual(
                result.peak_index[k], max(range(4), key=lambda i: abs(column[i]))
            )
        exact_trace = math.fsum(self.oracle.resolution[i][i] for i in range(4))
        # Hutchinson with 60 Rademacher probes: allow 5 reported standard errors
        self.assertLess(
            abs(result.trace - exact_trace), 5.0 * result.trace_standard_error + 1e-9
        )
        self.assertTrue(result.all_converged)
        self.assertEqual(
            result.lis_rank,
            4
            - sum(
                1 for v in self.study.latest("spectrum").result.eigenvalues if v <= 1.0
            ),
        )

    def test_evaluation_ledger_adds_up(self) -> None:
        """Total counts are the sum of the stage counts; every stage did work."""
        records = self.study.records
        total = self.study.evaluations
        self.assertEqual(total.value, sum(r.evaluations.value for r in records))
        self.assertEqual(total.vjp, sum(r.evaluations.vjp for r in records))
        for record in records:
            self.assertGreater(record.work, 0, record.stage)
            self.assertGreaterEqual(record.seconds, 0.0)
        self.assertIsInstance(total, EvaluationCounts)

    def test_every_stage_appears_in_the_report(self) -> None:
        """The report records seed, sizes, noise, prior, stages and caveats."""
        text = self.study.report()
        self.assertIn("Root seed: 7", text)
        self.assertIn("data m = 6, parameters n = 4, whitened coordinates r = 4", text)
        self.assertIn("DiagonalCovariance", text)
        self.assertIn("Prior: Gaussian N(mu, P P')", text)
        for stage in (
            "spectrum #0",
            "regularize #0",
            "map #0",
            "laplace #0",
            "sample #0",
            "resolution #0",
            "compare #0",
        ):
            self.assertIn(f". {stage}:", text)
        self.assertIn("## Evaluation ledger", text)
        self.assertIn("## Caveats", text)
        self.assertIn("Laplace log-evidence omits", text)
        self.assertIn("dili: an exact MCMC", text)
        self.assertNotIn("wall time", text.split("## Evaluation ledger")[0])
        timed = self.study.report(include_timings=True)
        self.assertIn("wall time", timed)

    def test_other_samplers_agree_with_exact_posterior(self) -> None:
        """pCN-family, RTO, SMC and TMCMC reproduce the exact moments."""
        oracle = self.oracle
        study = linear_study(seed=11)
        runs = {
            "gpcn": {"draws": 2500, "burn_in": 500},
            "pcn": {"draws": 6000, "burn_in": 500},
            "rto": {"draws": 800, "burn_in": 0},
            "smc": {"particles": 400},
            "tmcmc": {"particles": 400},
        }
        for method, kwargs in runs.items():
            study.sample(method, **kwargs)  # type: ignore[arg-type]
            comparison = study.compare(method=method).result
            # linear model: RTO and gpCN proposals equal the posterior, pCN is the
            # slowest mixer; the screening tolerances 0.25 sd / 0.3 are loose for
            # ESS >= 100 and the MCSE allowance is added on top
            self.assertTrue(comparison.passed, (method, comparison.failures))
        # the same stage-specific evidence checks
        smc = study.latest("sample")
        self.assertTrue(smc.result.converged)
        # the pCN, RTO and SMC runs do not report an evidence except SMC/TMCMC
        for record in study.records:
            if record.stage == "sample" and record.result.method in ("smc", "tmcmc"):
                self.assertLess(
                    abs(record.result.log_evidence - oracle.evidence), 1.0, record.label
                )
                self.assertEqual(record.result.mcse_kind, "iid-naive")
                self.assertFalse(record.result.asymptotically_exact)

    def test_user_supplied_moments_reference(self) -> None:
        """A user reference with the exact moments passes; a shifted one fails."""
        study = linear_study(seed=12)
        study.sample("gpcn", draws=2000, burn_in=500)
        mean = self.oracle.mean
        variance = tuple(self.oracle.cov[i][i] for i in range(4))
        passed = study.compare((mean, variance)).result
        self.assertEqual(passed.reference_kind, "moments")
        self.assertTrue(passed.passed, passed.failures)
        shifted = tuple(
            m + 2.0 * math.sqrt(v) for m, v in zip(mean, variance, strict=True)
        )
        failed = study.compare({"mean": shifted, "variance": variance}).result
        self.assertFalse(failed.passed)
        self.assertEqual(len(failed.failures), 4)
        # a 2 sd shift seen through an MCSE of about 0.1 sd
        self.assertGreater(min(failed.mean_error_sd), 1.5)


class DeterminismTests(unittest.TestCase):
    """The stage streams make a study a function of the root seed."""

    def test_same_seed_same_report_and_samples(self) -> None:
        """Two studies with one seed give identical samples and report text."""
        first, second = linear_study(seed=3), linear_study(seed=3)
        for study in (first, second):
            study.laplace()
            study.sample("gpcn", draws=300, burn_in=100)
            study.compare()
        self.assertEqual(
            first.latest("sample").result.samples,
            second.latest("sample").result.samples,
        )
        self.assertEqual(first.report(), second.report())

    def test_different_seed_changes_samples(self) -> None:
        """Another root seed changes the chain."""
        a, b = linear_study(seed=3), linear_study(seed=4)
        for study in (a, b):
            study.sample("gpcn", draws=100, burn_in=50)
        self.assertNotEqual(
            a.latest("sample").result.samples, b.latest("sample").result.samples
        )

    def test_stage_order_does_not_change_a_stage(self) -> None:
        """A sample stage depends on (stage, run) only, not on earlier stages."""
        plain = linear_study(seed=9)
        busy = linear_study(seed=9)
        plain.sample("dili", draws=200, burn_in=100)
        busy.spectrum()
        busy.regularize()
        busy.map()
        busy.laplace()
        busy.sample("dili", draws=200, burn_in=100)
        self.assertEqual(
            plain.latest("sample").result.samples, busy.latest("sample").result.samples
        )
        self.assertEqual(plain.latest("sample").seed, busy.latest("sample").seed)

    def test_repeated_stage_uses_a_new_child_stream(self) -> None:
        """The k-th run of a stage uses stream (stage, k): runs differ."""
        study = linear_study(seed=2)
        study.sample("gpcn", draws=100, burn_in=50)
        study.sample("gpcn", draws=100, burn_in=50)
        first, second = study.records[-2], study.records[-1]
        self.assertEqual((first.run, second.run), (0, 1))
        self.assertNotEqual(first.seed, second.seed)
        self.assertNotEqual(first.result.samples, second.result.samples)
        root = module.resolve_stream(2)
        self.assertEqual(first.seed, root.child("sample", 0).seed)
        self.assertEqual(second.seed, root.child("sample", 1).seed)

    def test_automatic_prerequisite_is_its_own_stage(self) -> None:
        """A gpcn run without laplace() runs laplace as a recorded stage first."""
        study = linear_study(seed=5)
        study.sample("gpcn", draws=100, burn_in=50)
        self.assertEqual([r.stage for r in study.records], ["laplace", "sample"])
        self.assertTrue(
            any("laplace() was run first" in w for w in study.records[1].warnings)
        )
        self.assertGreater(study.records[0].work, 0)


class NonlinearStudyTests(unittest.TestCase):
    """The two-parameter nonlinear problem against a brute-force grid."""

    @classmethod
    def setUpClass(cls) -> None:
        """Run map, laplace, DILI and the quadrature comparison once."""
        cls.study = two_parameter_study()
        cls.mean, cls.var, cls.evidence = grid_oracle(cls.study.problem)
        s = cls.study
        s.spectrum()
        s.map()
        s.laplace()
        s.sample("dili", draws=4000, burn_in=1000)
        s.resolution()
        s.compare()

    def test_quadrature_reference_equals_the_grid(self) -> None:
        """The study's quadrature reference reproduces the brute-force grid."""
        comparison = self.study.latest("compare").result
        self.assertEqual(comparison.reference_kind, "quadrature")
        # the prior factor is the identity, so x = z; both are converged integrals
        # of a smooth density (grid h = 0.05, order-8 panels): 1e-4 is generous
        for i in range(2):
            self.assertAlmostEqual(
                comparison.reference_mean[i], self.mean[i], delta=1e-4
            )
            self.assertAlmostEqual(
                comparison.reference_variance[i], self.var[i], delta=1e-4
            )
        self.assertAlmostEqual(
            comparison.reference_log_evidence, self.evidence, delta=1e-3
        )

    def test_dili_agrees_with_quadrature(self) -> None:
        """DILI moments match the quadrature reference."""
        comparison = self.study.latest("compare").result
        self.assertTrue(comparison.passed, comparison.failures)
        samples = self.study.latest("sample").result
        for i in range(2):
            self.assertLess(comparison.mean_error_sd[i], 0.15)
            self.assertLess(
                abs(samples.mean[i] - comparison.reference_mean[i]),
                5.0 * samples.mcse[i] + 0.02,
            )

    def test_laplace_is_close_but_not_exact(self) -> None:
        """Laplace error is reported against quadrature and is not zero."""
        comparison = self.study.latest("compare").result
        assert comparison.laplace_mean_error_sd is not None
        assert comparison.laplace_variance_ratio is not None
        # a Gauss-Newton Gaussian for a mildly nonlinear model: sd-scale errors
        # below one posterior sd, but visibly nonzero
        self.assertGreater(max(comparison.laplace_mean_error_sd), 1e-4)
        self.assertLess(max(comparison.laplace_mean_error_sd), 1.0)
        for ratio in comparison.laplace_variance_ratio:
            self.assertGreater(ratio, 0.3)
            self.assertLess(ratio, 3.0)
        lap = self.study.latest("laplace").result
        self.assertLess(abs(lap.log_evidence - comparison.reference_log_evidence), 1.0)
        caveats = self.study.report().split("## Caveats")[1]
        self.assertIn("Gauss-Newton Hessian", caveats)

    def test_spectrum_follows_the_linearization_point(self) -> None:
        """Before map() the spectrum is at the prior mean, after it at the MAP."""
        study = two_parameter_study()
        first = study.spectrum().result
        study.map()
        second = study.spectrum().result
        self.assertEqual((first.point, second.point), ("prior mean", "MAP"))
        self.assertNotEqual(first.singular_values, second.singular_values)
        self.assertTrue(any("local statement" in w for w in study.records[0].warnings))

    def test_resolution_is_flagged_local(self) -> None:
        """The nonlinear resolution warns that it is a linearization."""
        record = self.study.latest("resolution")
        self.assertEqual(record.result.point, "MAP")
        self.assertTrue(any("linearization" in w for w in record.warnings))
        for value in record.result.diagonal:
            self.assertGreater(value, 0.0)
            self.assertLess(value, 1.0 + 1e-9)

    def test_other_samplers_against_quadrature(self) -> None:
        """RTO, gpCN, pCN, SMC and TMCMC match the quadrature moments."""
        # the quadrature moments of the class study (checked against the grid
        # above) serve as user moments, so the quadrature is not repeated
        reference = self.study.latest("compare").result
        study = two_parameter_study(seed=8)
        study.map()
        runs = {
            "gpcn": {"draws": 4000, "burn_in": 1000},
            "pcn": {"draws": 8000, "burn_in": 1000},
            "rto": {"draws": 1500, "burn_in": 100},
            "smc": {"particles": 600},
            "tmcmc": {"particles": 600},
        }
        for method, kwargs in runs.items():
            study.sample(method, **kwargs)  # type: ignore[arg-type]
            # the variance ratio of an ESS ~ 300-700 sample has a relative error
            # of about sqrt(2/ESS) = 6-8% (more for a skewed target), so the
            # default 0.3 band would leave a 3 sigma margin only; 0.4 gives 5
            comparison = study.compare(
                (reference.reference_mean, reference.reference_variance),
                method=method,
                variance_tolerance=0.4,
            ).result
            self.assertTrue(comparison.passed, (method, comparison.failures))
        evidences = [
            r.result.log_evidence
            for r in study.records
            if r.stage == "sample" and r.result.method in ("smc", "tmcmc")
        ]
        for value in evidences:
            self.assertLess(abs(value - self.evidence), 0.5)


class FailureCaseTests(unittest.TestCase):
    """Invalid use is rejected with an explicit error and no stage record."""

    def test_compare_without_samples(self) -> None:
        """compare() before sample() raises and appends nothing."""
        study = linear_study()
        with self.assertRaisesRegex(ValueError, "needs samples"):
            study.compare()
        self.assertEqual(study.records, ())

    def test_compare_with_method_that_was_not_run(self) -> None:
        """compare(method=...) of a method never sampled raises."""
        study = linear_study()
        study.sample("gpcn", draws=100, burn_in=50)
        with self.assertRaisesRegex(ValueError, "'smc'"):
            study.compare(method="smc")

    def test_invalid_sample_method(self) -> None:
        """An unknown method raises before any stage (not even laplace) runs."""
        study = linear_study()
        with self.assertRaisesRegex(ValueError, "method must be one of"):
            study.sample("hmc")
        self.assertEqual(study.records, ())
        with self.assertRaises(ValueError):
            study.sample("dili", draws=0)
        with self.assertRaises(TypeError):
            study.sample("dili", draws=2.5)  # type: ignore[arg-type]
        self.assertEqual(study.records, ())

    def test_regularize_needs_a_linear_problem(self) -> None:
        """Regularization rules are defined for linear problems only."""
        with self.assertRaisesRegex(ValueError, "linear problem"):
            two_parameter_study().regularize()
        with self.assertRaisesRegex(ValueError, "subset"):
            linear_study().regularize(rules=("magic",))

    def test_reference_errors(self) -> None:
        """Unavailable or malformed references raise ValueError."""
        nonlinear = two_parameter_study()
        nonlinear.sample("pcn", draws=50, burn_in=10)
        with self.assertRaisesRegex(ValueError, "linear problem"):
            nonlinear.compare("exact")
        with self.assertRaises(ValueError):
            nonlinear.compare("bogus")
        with self.assertRaises(ValueError):
            nonlinear.compare(([0.0, 0.0], [1.0]))
        with self.assertRaisesRegex(ValueError, "positive"):
            nonlinear.compare(([0.0, 0.0], [1.0, -1.0]))
        with self.assertRaises(ValueError):
            nonlinear.compare({"mean": [0.0, 0.0]})
        with self.assertRaises(TypeError):
            nonlinear.compare(3.5)
        with self.assertRaises(ValueError):
            nonlinear.compare(mean_tolerance_sd=0.0)
        self.assertEqual([r.stage for r in nonlinear.records], ["sample"])

    def test_quadrature_needs_at_most_two_parameters(self) -> None:
        """Four whitened parameters have no automatic quadrature reference."""
        study = linear_study()
        study.sample("gpcn", draws=100, burn_in=50)
        with self.assertRaisesRegex(ValueError, "at most two"):
            study.compare("quadrature")

    def test_constructor_validation(self) -> None:
        """Wrong problem types, extra inputs and a missing prior are rejected."""
        _, _, _, problem = linear_fixture()
        with self.assertRaises(TypeError):
            InverseStudy(object())
        with self.assertRaises(TypeError):
            InverseStudy(problem, seed=1.5)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            InverseStudy(problem, data=(0.0,) * 6)
        with self.assertRaisesRegex(ValueError, "needs data, noise and prior_factor"):
            InverseStudy(TwoParameterModel())
        penalised = LinearInverseProblem.create(
            ((1.0, 0.0), (0.0, 1.0)),
            (1.0, 2.0),
            penalty=((1.0, -1.0),),
        )
        with self.assertRaisesRegex(ValueError, "pass prior_factor"):
            InverseStudy(penalised)
        ready = two_parameter_study().problem
        with self.assertRaises(ValueError):
            InverseStudy(ready, data=(0.0, 0.0, 0.0))
        self.assertIs(InverseStudy(ready).problem, ready)

    def test_spectrum_option_validation(self) -> None:
        """Bad thresholds and oversized dense spectra are rejected."""
        study = linear_study()
        with self.assertRaises(ValueError):
            study.spectrum(threshold=0.0)
        with self.assertRaisesRegex(ValueError, "max_dimension"):
            study.spectrum(max_dimension=2)
        self.assertEqual(study.records, ())

    def test_dili_needs_a_rank_one_laplace(self) -> None:
        """A Laplace threshold above every eigenvalue leaves DILI no subspace."""
        study = linear_study()
        study.laplace(threshold=1e9)
        self.assertEqual(study.latest("laplace").result.rank, 0)
        with self.assertRaisesRegex(ValueError, "rank >= 1"):
            study.sample("dili", draws=50, burn_in=10)

    def test_failed_particle_run_is_reported_not_compared(self) -> None:
        """SMC stopped at one stage is flagged and cannot feed compare()."""
        study = linear_study()
        study.sample("smc", particles=50, options={"max_stages": 1})
        record = study.latest("sample")
        self.assertFalse(record.result.converged)
        self.assertTrue(any("did not finish" in w for w in record.warnings))
        with self.assertRaisesRegex(ValueError, "converged"):
            study.compare()

    def test_empty_report_and_latest(self) -> None:
        """A study without stages reports so and latest() raises."""
        study = linear_study()
        text = study.report()
        self.assertIn("Stages run: none", text)
        with self.assertRaises(ValueError):
            study.latest("map")


class EllipticStudySmokeTests(unittest.TestCase):
    """A PDE-based study runs every stage on a small elliptic problem."""

    def test_pipeline_on_the_elliptic_problem(self) -> None:
        """Spectrum, MAP, Laplace, DILI and resolution give finite results."""
        study = elliptic_study(8)
        study.spectrum()
        study.map()
        study.laplace()
        study.sample("dili", draws=300, burn_in=200)
        study.resolution(trace_probes=10)
        self.assertTrue(study.latest("map").result.converged)
        result = study.latest("sample").result
        self.assertTrue(all(math.isfinite(v) for v in result.variance))
        self.assertGreater(result.acceptance_rate, 0.05)
        self.assertGreaterEqual(study.latest("spectrum").result.rank, 1)
        self.assertIn("## 5. resolution #0", study.report())
        with self.assertRaisesRegex(ValueError, "no automatic reference"):
            study.compare()


def elliptic_study(n: int, *, sigma: float = 0.005, seed: int = 3) -> InverseStudy:
    """Return a study of the 1D elliptic problem with a cosine prior."""
    sensors = (0.1, 0.25, 0.4, 0.55, 0.7, 0.85)
    model = EllipticDiffusion1D(n, sensors=sensors, source=1.0)
    p = tuple(
        tuple(
            (1 + j) ** (-1.5)
            * (1.0 if j == 0 else math.sqrt(2.0))
            * math.cos(j * math.pi * (i + 0.5) / n)
            for j in range(n)
        )
        for i in range(n)
    )
    rng = random.Random(seed)
    truth = mat_vec(p, tuple(rng.gauss(0.0, 1.0) for _ in range(n)))
    data = tuple(g + sigma * rng.gauss(0.0, 1.0) for g in model.value(truth))
    return InverseStudy(
        model,
        seed=seed,
        data=data,
        noise=DiagonalCovariance([sigma**2] * len(sensors)),
        prior_factor=DenseOperator(p),
    )


@unittest.skipUnless(SLOW, "slow tier: set PROBSTAT_SLOW=1")
class SlowStudyTests(unittest.TestCase):
    """Long runs: tight agreement with quadrature and a long-chain reference."""

    def test_long_dili_against_quadrature(self) -> None:
        """100000 DILI draws on the two-parameter problem against quadrature."""
        study = two_parameter_study(seed=21)
        study.map()
        study.laplace()
        study.sample("dili", draws=100000, burn_in=5000)
        # measured: ESS 1.4e4 and 2.2e4, mean errors 0.007 sd, variance ratios
        # 0.964 and 0.987. The variance converges more slowly than the mean (the
        # tails are visited rarely, so ESS of x^2 is below the ESS of x): the
        # variance band is 0.10, not the sqrt(2/ESS) = 1% of a Gaussian
        comparison = study.compare(
            mean_tolerance_sd=0.05, variance_tolerance=0.10
        ).result
        self.assertTrue(comparison.passed, comparison.failures)
        self.assertLess(max(comparison.mean_error_sd), 0.05)

    def test_dili_against_long_pcn_on_the_elliptic_problem(self) -> None:
        """DILI means equal those of a long plain-pCN chain on 1D elliptic."""
        study = elliptic_study(8)
        study.laplace()
        study.sample("pcn", draws=60000, burn_in=3000, thin=3)
        reference = study.latest("sample").result
        study.sample("dili", draws=20000, burn_in=2000)
        dili = study.latest("sample").result
        # measured: pCN ESS 18-142 of 20000 stored draws (it mixes slowly on the
        # sharply informed modes), DILI ESS 4300-5300 of 20000; every mean
        # difference was at most 0.5 of the 5 MCSE band. The pCN variance
        # estimates are too noisy (ratios 0.86-1.45) to serve as a reference.
        self.assertGreater(min(dili.ess), 10.0 * min(reference.ess))
        for i in range(8):
            band = 5.0 * math.hypot(reference.mcse[i], dili.mcse[i])
            self.assertLess(abs(dili.mean[i] - reference.mean[i]), band)
        self.assertGreater(dili.ess_per_work, 5.0 * reference.ess_per_work)


if __name__ == "__main__":
    unittest.main()
