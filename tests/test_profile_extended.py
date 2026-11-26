"""Tests for 2D profiles, bootstrap LR calibration and ODE identifiability."""

from __future__ import annotations

import math
import os
import random
import unittest

from probstat.distributions import ChiSquare
from probstat.inverse.profile_extended import (
    BootstrapCalibrationResult,
    Profile2DResult,
    bootstrap_lr_calibration,
    ode_sensitivity,
    profile_likelihood_2d,
    profile_region_2d,
)
from probstat.inverse.profile_likelihood import efficient_information
from probstat.linalg import matmul, solve_spd, transpose
from probstat.numerics import integrate_fixed_step

X = (
    (1.0, 0.0, 0.5),
    (1.0, 1.0, -0.5),
    (1.0, 2.0, 1.0),
    (1.0, 3.0, 0.0),
    (1.0, 4.0, -1.0),
    (1.0, 5.0, 0.7),
)
Y = (0.9, 2.1, 3.2, 3.8, 5.3, 5.9)
SIGMA = 0.5


def _nll_for(data: tuple[float, ...]):  # type: ignore[no-untyped-def]
    """Return the Gaussian negative log-likelihood (known sigma) of ``data``."""

    def nll(theta: tuple[float, ...]) -> float:
        total = 0.0
        for row, y in zip(X, data, strict=True):
            r = y - sum(a * b for a, b in zip(row, theta, strict=True))
            total += r * r
        return total / (2.0 * SIGMA**2)

    return nll


def _mle() -> tuple[float, ...]:
    """Return the least-squares estimate."""
    xt = transpose(X)
    rhs = tuple(sum(a * b for a, b in zip(row, Y, strict=True)) for row in xt)
    return solve_spd(matmul(xt, X), rhs)


def _fisher() -> tuple[tuple[float, ...], ...]:
    """Return ``X'X / sigma^2``."""
    gram = matmul(transpose(X), X)
    return tuple(tuple(v / SIGMA**2 for v in row) for row in gram)


class Profile2DLinearGaussianTest(unittest.TestCase):
    """The 2D profile of a linear Gaussian model is an exact quadratic form."""

    def setUp(self) -> None:
        """Build the grid around the MLE in the efficient-information metric."""
        self.theta = _mle()
        self.feff = efficient_information(_fisher(), (1, 2))
        se1 = 1.0 / math.sqrt(self.feff[0][0])
        se2 = 1.0 / math.sqrt(self.feff[1][1])
        self.ga = tuple(self.theta[1] + se1 * k / 2.0 for k in range(-6, 7))
        self.gb = tuple(self.theta[2] + se2 * k / 2.0 for k in range(-6, 7))

    def test_profile_equals_schur_quadratic(self) -> None:
        """LR(a, b) = d' F_eff d with F_eff the Schur complement."""
        res = profile_likelihood_2d(_nll_for(Y), self.theta, (1, 2), self.ga, self.gb)
        self.assertTrue(res.all_ok)
        self.assertFalse(res.improved_estimate_found)
        self.assertEqual(res.n_failed, 0)
        f = self.feff
        for i, a in enumerate(self.ga):
            for j, b in enumerate(self.gb):
                d = (a - self.theta[1], b - self.theta[2])
                exact = (
                    f[0][0] * d[0] ** 2
                    + 2.0 * f[0][1] * d[0] * d[1]
                    + f[1][1] * d[1] ** 2
                )
                self.assertAlmostEqual(res.likelihood_ratio[i][j], exact, delta=5e-6)

    def test_serpentine_path_and_accounting(self) -> None:
        """Consecutive path cells are grid neighbours; every cell is visited once."""
        res = profile_likelihood_2d(_nll_for(Y), self.theta, (1, 2), self.ga, self.gb)
        self.assertEqual(len(res.path), len(self.ga) * len(self.gb))
        self.assertEqual(len(set(res.path)), len(res.path))
        for (r0, c0), (r1, c1) in zip(res.path, res.path[1:], strict=False):
            self.assertEqual(abs(r0 - r1) + abs(c0 - c1), 1)
        self.assertEqual([p.order for p in res.points], list(range(len(res.path))))
        self.assertEqual(
            sum(p.evaluations for p in res.points) + 1, res.total_evaluations
        )
        self.assertTrue(all(p.start in ("warm", "estimate") for p in res.points))
        self.assertEqual(res.cell(3, 4).row, 3)

    def test_region_matches_chi2_2(self) -> None:
        """Region = ellipse d' F d <= chi2_{2,0.95}; area = pi q / sqrt(det F)."""
        k = 24
        se1 = 1.0 / math.sqrt(self.feff[0][0])
        se2 = 1.0 / math.sqrt(self.feff[1][1])
        ga = tuple(self.theta[1] + 4.0 * se1 * (i / k - 0.5) * 2 for i in range(k + 1))
        gb = tuple(self.theta[2] + 4.0 * se2 * (i / k - 0.5) * 2 for i in range(k + 1))
        res = profile_likelihood_2d(_nll_for(Y), self.theta, (1, 2), ga, gb)
        reg = profile_region_2d(res, 0.95)
        q = ChiSquare(2.0).quantile(0.95)
        self.assertAlmostEqual(reg.threshold, q, places=12)
        self.assertEqual(reg.components, 1)
        self.assertFalse(reg.touches_grid_edge)
        self.assertEqual(reg.unreliable_count, 0)
        f = self.feff
        det = f[0][0] * f[1][1] - f[0][1] ** 2
        assert reg.area_estimate is not None
        exact_area = math.pi * q / math.sqrt(det)
        # cell counting has O(perimeter * cell) error; 8 % is generous here
        self.assertAlmostEqual(reg.area_estimate / exact_area, 1.0, delta=0.08)
        for i in range(len(ga)):
            for j in range(len(gb)):
                lr = res.likelihood_ratio[i][j]
                if abs(lr - q) > 1e-4:
                    self.assertEqual(reg.cells[i][j] == "inside", lr <= q)

    def test_region_touching_edge_is_flagged(self) -> None:
        """A grid narrower than the region reports an open region."""
        narrow_a = (self.theta[1] - 0.01, self.theta[1] + 0.01)
        narrow_b = (self.theta[2] - 0.01, self.theta[2] + 0.01)
        res = profile_likelihood_2d(_nll_for(Y), self.theta, (1, 2), narrow_a, narrow_b)
        reg = profile_region_2d(res)
        self.assertTrue(reg.touches_grid_edge)
        self.assertEqual(reg.inside_count, 4)


class Profile2DFailureTest(unittest.TestCase):
    """Failed and non-converged cells are reported, never filled."""

    def test_infeasible_cells_stay_failed(self) -> None:
        """Cells with a < b have no finite objective: inf, status failed."""

        def nll(t: tuple[float, ...]) -> float:
            if t[0] < t[1]:
                raise ValueError("infeasible")
            return (t[0] - 1.0) ** 2 + (t[1] - 1.0) ** 2 + (t[2] - t[0] * t[1]) ** 2

        grid = tuple(0.5 + 0.25 * k for k in range(5))
        res = profile_likelihood_2d(nll, (1.0, 1.0, 1.0), (0, 1), grid, grid)
        expected = sum(1 for a in grid for b in grid if a < b)
        self.assertEqual(res.n_failed, expected)
        self.assertGreater(expected, 0)
        for p in res.points:
            if p.value_a < p.value_b:
                self.assertEqual(p.status, "failed")
                self.assertEqual(p.start, "none")
                self.assertTrue(math.isinf(res.likelihood_ratio[p.row][p.col]))
            else:
                self.assertEqual(p.status, "ok")
                self.assertAlmostEqual(
                    res.likelihood_ratio[p.row][p.col],
                    2.0 * ((p.value_a - 1.0) ** 2 + (p.value_b - 1.0) ** 2),
                    delta=1e-5,
                )
        reg = profile_region_2d(res)
        self.assertEqual(reg.unreliable_count, expected)
        self.assertFalse(res.all_ok)
        for p in res.points:
            if p.status == "failed":
                self.assertEqual(reg.cells[p.row][p.col], "unreliable")

    def test_non_converged_cells_are_unreliable(self) -> None:
        """A one-iteration optimizer leaves cells flagged, not inside/outside."""
        theta = _mle()
        grid_a = tuple(theta[1] + 0.3 * k for k in range(-2, 3))
        grid_b = tuple(theta[2] + 0.3 * k for k in range(-2, 3))
        res = profile_likelihood_2d(
            _nll_for(Y), theta, (1, 2), grid_a, grid_b, max_iterations=1
        )
        self.assertGreater(res.n_not_converged, 0)
        reg = profile_region_2d(res)
        self.assertGreaterEqual(reg.unreliable_count, res.n_not_converged)

    def test_improved_estimate_is_reported(self) -> None:
        """A bad supplied estimate is replaced as baseline, never clipped."""
        theta = _mle()
        bad = (theta[0], theta[1] + 0.3, theta[2])
        grid = tuple(theta[1] + 0.1 * k for k in range(-5, 6))
        gb = tuple(theta[2] + 0.1 * k for k in range(-5, 6))
        res = profile_likelihood_2d(_nll_for(Y), bad, (1, 2), grid, gb)
        self.assertTrue(res.improved_estimate_found)
        assert res.improved_estimate is not None
        self.assertLess(res.nll_min, res.nll_at_estimate)
        self.assertGreaterEqual(min(min(r) for r in res.likelihood_ratio), 0.0)

    def test_validation(self) -> None:
        """Invalid arguments raise."""
        theta = _mle()
        g = (theta[1] - 1.0, theta[1] + 1.0)
        h = (theta[2] - 1.0, theta[2] + 1.0)
        nll = _nll_for(Y)
        with self.assertRaises(TypeError):
            profile_likelihood_2d(1.0, theta, (1, 2), g, h)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            profile_likelihood_2d(nll, theta, (1, 1), g, h)
        with self.assertRaises(ValueError):
            profile_likelihood_2d(nll, theta, (1, 5), g, h)
        with self.assertRaises(ValueError):
            profile_likelihood_2d(nll, theta, (1, 2), (g[1], g[0]), h)
        with self.assertRaises(ValueError):
            profile_likelihood_2d(nll, theta, (1, 2), (g[0] + 5, g[1] + 5), h)
        with self.assertRaises(ValueError):
            profile_likelihood_2d(nll, theta, (1, 2), g, h, tolerance=0.0)
        with self.assertRaises(TypeError):
            profile_region_2d("x")  # type: ignore[arg-type]
        res = profile_likelihood_2d(nll, theta, (1, 2), g, h)
        self.assertIsInstance(res, Profile2DResult)
        with self.assertRaises(ValueError):
            profile_region_2d(res, 1.0)
        with self.assertRaises(ValueError):
            res.cell(9, 0)


def _regression_problem(replicates: int, seed: int) -> BootstrapCalibrationResult:
    """Regular case: known-sigma regression, target slope; LR* is chi2(1)."""
    theta = _mle()

    def simulate(t: tuple[float, ...], rng: random.Random) -> tuple[float, ...]:
        return tuple(
            sum(a * b for a, b in zip(row, t, strict=True)) + rng.gauss(0.0, SIGMA)
            for row in X
        )

    return bootstrap_lr_calibration(
        simulate, _nll_for, theta, (1,), replicates=replicates, seed=seed
    )


def _boundary_problem(
    replicates: int, seed: int, dim: int = 1
) -> BootstrapCalibrationResult:
    """Boundary case: x ~ N(mu, I), mu >= 0 componentwise, fitted mu = 0.

    ``LR* = n |max(xbar, 0)|^2``: a mixture of chi2(k), k = 0..dim, with
    binomial weights, instead of the regular chi2(dim).
    """
    n = 10

    def simulate(
        t: tuple[float, ...], rng: random.Random
    ) -> tuple[tuple[float, ...], ...]:
        return tuple(tuple(rng.gauss(m, 1.0) for m in t) for _ in range(n))

    def factory(data: tuple[tuple[float, ...], ...]):  # type: ignore[no-untyped-def]
        def nll(t: tuple[float, ...]) -> float:
            return 0.5 * sum(
                (x - m) ** 2 for row in data for x, m in zip(row, t, strict=True)
            )

        return nll

    return bootstrap_lr_calibration(
        simulate,
        factory,
        (0.0,) * dim,
        tuple(range(dim)),
        replicates=replicates,
        seed=seed,
        lower=(0.0,) * dim,
    )


class BootstrapCalibrationTest(unittest.TestCase):
    """Bootstrap thresholds: agree with chi2 when regular, differ at a boundary."""

    def test_regular_case_agrees_with_chi2(self) -> None:
        """Known-sigma regression: LR* ~ chi2(1) exactly, so thresholds agree."""
        res = _regression_problem(300, 11)
        print(
            f"\n[regular] valid={res.valid_replicates} q={res.quantile} "
            f"se={res.quantile_se} chi2={res.chi2_threshold} "
            f"cover={res.coverage_at_chi2} +- {res.coverage_se} "
            f"disc={res.discrepancy_in_se} ci={res.quantile_ci}"
        )
        self.assertEqual(res.failed_replicates, 0)
        self.assertEqual(res.dof, 1)
        assert res.discrepancy_in_se is not None
        self.assertLess(abs(res.discrepancy_in_se), 3.0)
        self.assertIsNot(res.chi2_within_ci, False)
        self.assertLess(abs(res.coverage_at_chi2 - 0.95), 3.0 * res.coverage_se)

    def test_boundary_case_differs_from_chi2(self) -> None:
        """Two boundary coordinates: LR* is a chi2 mixture; chi2(2) is too large.

        Bootstrap null: P(LR* <= 5.99) = 0.25 + 0.5 F1(5.99) + 0.25 F2(5.99)
        = 0.980 (theory) against the nominal 0.95, and the 95 % quantile is
        about 4.23 against 5.99.
        """
        res = _boundary_problem(400, 5, dim=2)
        print(
            f"[boundary-2d] valid={res.valid_replicates} "
            f"failed={res.failed_replicates} q={res.quantile} "
            f"se={res.quantile_se} chi2={res.chi2_threshold} "
            f"cover={res.coverage_at_chi2} +- {res.coverage_se} "
            f"disc={res.discrepancy_in_se} ci={res.quantile_ci}"
        )
        self.assertEqual(res.dof, 2)
        self.assertGreater(res.valid_replicates, 380)
        assert res.quantile is not None and res.discrepancy_in_se is not None
        self.assertLess(res.quantile, res.chi2_threshold)
        self.assertLess(res.discrepancy_in_se, -2.5)
        self.assertIs(res.chi2_within_ci, False)
        self.assertGreater(res.coverage_at_chi2, 0.95 + 2.5 * res.coverage_se)
        self.assertAlmostEqual(res.coverage_at_chi2, 0.980, delta=3.0 * res.coverage_se)

    def test_reproducible_and_seed_sensitive(self) -> None:
        """Same seed, same statistics; different seed, different statistics."""
        a = _boundary_problem(40, 3)
        b = _boundary_problem(40, 3)
        c = _boundary_problem(40, 4)
        self.assertEqual(a.statistics, b.statistics)
        self.assertNotEqual(a.statistics, c.statistics)

    def test_boundary_statistic_has_point_mass(self) -> None:
        """About half of the boundary statistics are (numerically) zero."""
        res = _boundary_problem(60, 8)
        zeros = sum(1 for s in res.statistics if s < 1e-6)
        self.assertGreater(zeros, 15)
        self.assertLess(zeros, 45)

    def test_validation_and_failed_replicates(self) -> None:
        """Bad arguments raise; non-converged fits are dropped, not imputed."""

        def sim(t: tuple[float, ...], rng: random.Random) -> float:
            return rng.random()

        def factory(data: float):  # type: ignore[no-untyped-def]
            def nll(t: tuple[float, ...]) -> float:
                return 100.0 * (t[1] - t[0] ** 2) ** 2 + (1.0 - t[0]) ** 2 + data

            return nll

        start = (-1.2, 1.0)
        with self.assertRaises(ValueError):
            bootstrap_lr_calibration(sim, factory, start, (0,), replicates=5)
        with self.assertRaises(ValueError):
            bootstrap_lr_calibration(sim, factory, start, (2,), replicates=30)
        with self.assertRaises(ValueError):
            bootstrap_lr_calibration(sim, factory, start, (0,), level=1.5)
        with self.assertRaises(TypeError):
            bootstrap_lr_calibration(1, factory, start, (0,))  # type: ignore[arg-type]
        res = bootstrap_lr_calibration(
            sim, factory, start, (0,), replicates=20, max_iterations=2
        )
        self.assertEqual(res.failed_replicates, 20)
        self.assertEqual(res.valid_replicates, 0)
        self.assertIsNone(res.quantile)
        self.assertIsNone(res.quantile_se)
        self.assertTrue(math.isnan(res.coverage_at_chi2))


@unittest.skipUnless(os.environ.get("PROBSTAT_SLOW") == "1", "slow tier")
class BootstrapSlowTest(unittest.TestCase):
    """1000-replicate versions of the two calibration studies."""

    def test_regular_1000(self) -> None:
        """Regular case with 1000 replicates."""
        res = _regression_problem(1000, 101)
        print(
            f"\n[regular-1000] q={res.quantile} se={res.quantile_se} "
            f"chi2={res.chi2_threshold} cover={res.coverage_at_chi2} "
            f"+- {res.coverage_se} disc={res.discrepancy_in_se}"
        )
        assert res.discrepancy_in_se is not None
        self.assertLess(abs(res.discrepancy_in_se), 3.0)

    def test_boundary_1000(self) -> None:
        """Boundary case with 1000 replicates."""
        res = _boundary_problem(1000, 102, dim=1)
        print(
            f"\n[boundary-1000] valid={res.valid_replicates} q={res.quantile} "
            f"se={res.quantile_se} chi2={res.chi2_threshold} "
            f"cover={res.coverage_at_chi2} +- {res.coverage_se} "
            f"disc={res.discrepancy_in_se}"
        )
        assert res.discrepancy_in_se is not None
        self.assertLess(res.discrepancy_in_se, -2.0)


# ---------------------------------------------------------------------------
# ODE sensitivities
# ---------------------------------------------------------------------------


def _decay_sum(t: float, y: tuple[float, ...] | list[float], th: tuple[float, ...]):  # type: ignore[no-untyped-def]
    """``x' = -(k1 + k2) x``."""
    return (-(th[0] + th[1]) * y[0],)


class OdeSensitivityTest(unittest.TestCase):
    """Forward sensitivities, rank rule and unidentifiable directions."""

    times = tuple(0.5 * k for k in range(1, 7))

    def test_sum_model_is_rank_one(self) -> None:
        """Only k1 + k2 is identifiable: columns equal analytic -t e^{-(k1+k2)t}."""
        theta = (0.7, 0.3)
        for analytic in (True, False):
            kwargs = {}
            if analytic:
                kwargs = {
                    "jacobian_y": lambda t, y, th: ((-(th[0] + th[1]),),),
                    "jacobian_theta": lambda t, y, th: ((-y[0], -y[0]),),
                }
            res = ode_sensitivity(_decay_sum, theta, (1.0,), self.times, **kwargs)
            for k, t in enumerate(self.times):
                exact = -t * math.exp(-(theta[0] + theta[1]) * t)
                self.assertAlmostEqual(res.sensitivity[k][0], exact, delta=1e-7)
                self.assertAlmostEqual(res.sensitivity[k][1], exact, delta=1e-7)
            self.assertEqual(res.rank, 1)
            self.assertFalse(res.identifiable)
            self.assertEqual(len(res.null_directions), 1)
            v = res.null_directions[0]
            self.assertAlmostEqual(abs(v[0]), 1.0 / math.sqrt(2.0), places=6)
            self.assertAlmostEqual(v[0] + v[1], 0.0, places=6)
            self.assertTrue(math.isinf(res.std_error_scale[1]))
            self.assertEqual(
                res.rank_tolerance, res.relative_tolerance * res.singular_values[0]
            )

    def test_matches_finite_differences_of_the_solution(self) -> None:
        """Damped oscillator (c, k): S equals central differences of the solution."""

        def osc(t, y, th):  # type: ignore[no-untyped-def]
            return (y[1], -th[1] * y[0] - th[0] * y[1])

        theta = (0.4, 2.0)
        times = (0.5, 1.0, 1.5, 2.0, 3.0)
        obs = ((1.0, 0.0),)
        res = ode_sensitivity(
            osc, theta, (1.0, 0.0), times, observation=obs, steps_per_interval=40
        )

        def solve(th: tuple[float, ...]) -> list[float]:
            out = []
            t_prev, y = 0.0, (1.0, 0.0)
            for t in times:
                tr = integrate_fixed_step(
                    lambda tt, yy: osc(tt, yy, th),
                    t_prev,
                    y,
                    step=(t - t_prev) / 40,
                    steps=40,
                )
                y = tr.final_state
                t_prev = t
                out.append(y[0])
            return out

        h = 1e-5
        for k in range(2):
            up = list(theta)
            dn = list(theta)
            up[k] += h
            dn[k] -= h
            fp, fm = solve(tuple(up)), solve(tuple(dn))
            for i in range(len(times)):
                self.assertAlmostEqual(
                    res.sensitivity[i][k], (fp[i] - fm[i]) / (2 * h), delta=1e-6
                )
        self.assertTrue(res.identifiable)
        self.assertEqual(res.rank, 2)
        self.assertEqual(res.null_directions, ())

    def test_identifiable_decay_with_initial_condition(self) -> None:
        """Decay x = x0 e^{-kt}: theta = (k, x0) is identifiable (rank 2)."""

        def decay(t, y, th):  # type: ignore[no-untyped-def]
            return (-th[0] * y[0],)

        theta = (0.8, 2.0)
        res = ode_sensitivity(
            decay, theta, (theta[1],), self.times, initial_sensitivity=((0.0, 1.0),)
        )
        for k, t in enumerate(self.times):
            self.assertAlmostEqual(
                res.sensitivity[k][0],
                -t * theta[1] * math.exp(-theta[0] * t),
                delta=1e-7,
            )
            self.assertAlmostEqual(
                res.sensitivity[k][1], math.exp(-theta[0] * t), delta=1e-7
            )
        self.assertEqual(res.rank, 2)
        self.assertTrue(res.identifiable)
        self.assertTrue(math.isfinite(res.condition_number))
        self.assertTrue(all(math.isfinite(s) for s in res.std_error_scale))

    def test_product_model_null_direction(self) -> None:
        """Product model z' = a b: S = (t b, t a); null direction (a, -b)/norm."""

        def prod(t, y, th):  # type: ignore[no-untyped-def]
            return (th[0] * th[1],)

        a, b = 2.0, 3.0
        res = ode_sensitivity(prod, (a, b), (0.0,), (1.0, 2.0, 4.0))
        self.assertEqual(res.rank, 1)
        v = res.null_directions[0]
        norm = math.hypot(a, b)
        sign = 1.0 if v[0] > 0 else -1.0
        self.assertAlmostEqual(sign * v[0], a / norm, places=6)
        self.assertAlmostEqual(sign * v[1], -b / norm, places=6)
        scaled = ode_sensitivity(
            prod, (a, b), (0.0,), (1.0, 2.0, 4.0), parameter_scale=(a, b)
        )
        self.assertEqual(scaled.rank, 1)

    def test_fewer_rows_than_parameters(self) -> None:
        """One observation of two parameters: rank <= 1, a null direction exists."""

        def decay(t, y, th):  # type: ignore[no-untyped-def]
            return (-th[0] * y[0],)

        res = ode_sensitivity(
            decay, (0.8, 2.0), (2.0,), (1.0,), initial_sensitivity=((0.0, 1.0),)
        )
        self.assertEqual(res.rank, 1)
        self.assertEqual(len(res.singular_values), 2)
        self.assertEqual(len(res.null_directions), 1)
        # the null direction annihilates the single sensitivity row
        row = res.scaled_sensitivity[0]
        v = res.null_directions[0]
        self.assertAlmostEqual(row[0] * v[0] + row[1] * v[1], 0.0, places=9)

    def test_tolerance_rule_is_explicit(self) -> None:
        """Two nearly equal rates: full rank at 1e-8, rank 1 at 5e-2."""

        def rhs(t, y, th):  # type: ignore[no-untyped-def]
            return (-th[0] * y[0], -th[1] * y[1])

        args = (rhs, (1.0, 1.05), (1.0, 1.0), self.times)
        fine = ode_sensitivity(*args, observation=((1.0, 1.0),))
        loose = ode_sensitivity(
            *args, observation=((1.0, 1.0),), relative_tolerance=5e-2
        )
        # sigma_2 / sigma_1 = 0.0172 (measured), between the two tolerances
        self.assertEqual(fine.rank, 2)
        self.assertEqual(loose.rank, 1)
        self.assertGreater(fine.condition_number, 50.0)
        self.assertEqual(fine.singular_values, loose.singular_values)
        self.assertAlmostEqual(loose.rank_gap, fine.condition_number, places=9)
        self.assertEqual(len(loose.null_directions), 1)
        self.assertEqual(loose.relative_tolerance, 5e-2)

    def test_noise_weighting_scales_singular_values(self) -> None:
        """Halving the noise doubles every singular value and halves the errors."""

        def osc(t, y, th):  # type: ignore[no-untyped-def]
            return (y[1], -th[1] * y[0] - th[0] * y[1])

        args = (osc, (0.4, 2.0), (1.0, 0.0), (0.5, 1.0, 2.0))
        base = ode_sensitivity(*args, observation=((1.0, 0.0),), noise_std=0.2)
        half = ode_sensitivity(*args, observation=((1.0, 0.0),), noise_std=0.1)
        for s0, s1 in zip(base.singular_values, half.singular_values, strict=True):
            self.assertAlmostEqual(s1 / s0, 2.0, places=9)
        for e0, e1 in zip(base.std_error_scale, half.std_error_scale, strict=True):
            self.assertAlmostEqual(e0 / e1, 2.0, places=9)

    def test_validation_and_failures(self) -> None:
        """Invalid inputs raise; a blow-up is reported, not hidden."""
        ok = (_decay_sum, (0.7, 0.3), (1.0,), self.times)
        with self.assertRaises(TypeError):
            ode_sensitivity(1, (0.7, 0.3), (1.0,), self.times)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            ode_sensitivity(_decay_sum, (0.7, 0.3), (1.0,), (1.0, 0.5))
        with self.assertRaises(ValueError):
            ode_sensitivity(_decay_sum, (0.7, 0.3), (1.0,), (0.0,))
        with self.assertRaises(ValueError):
            ode_sensitivity(*ok, relative_tolerance=0.0)
        with self.assertRaises(ValueError):
            ode_sensitivity(*ok, noise_std=-1.0)
        with self.assertRaises(ValueError):
            ode_sensitivity(*ok, parameter_scale=(1.0,))
        with self.assertRaises(ValueError):
            ode_sensitivity(*ok, steps_per_interval=0)
        with self.assertRaises(ValueError):
            ode_sensitivity(*ok, observation=((1.0, 0.0),))
        with self.assertRaises(ValueError):
            ode_sensitivity(lambda t, y, th: (0.0, 0.0), (0.7,), (1.0,), (1.0,))

        def blowup(t, y, th):  # type: ignore[no-untyped-def]
            return (th[0] * y[0] * y[0],)

        with self.assertRaises(ValueError):
            ode_sensitivity(blowup, (1.0,), (1.0,), (0.9, 2.0, 50.0))


if __name__ == "__main__":
    unittest.main()
