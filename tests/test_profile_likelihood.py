"""Tests for profile likelihood, Wilks intervals and efficient information."""

from __future__ import annotations

import math
import unittest

from probstat.distributions import ChiSquare
from probstat.inverse.profile_likelihood import (
    ProfileLikelihoodResult,
    delta_method_variance,
    efficient_information,
    functional_estimability,
    profile_interval,
    profile_likelihood,
)
from probstat.linalg import inverse_spd, matmul, solve_spd, transpose

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


def _gaussian_nll(theta: tuple[float, ...]) -> float:
    """Return the Gaussian negative log-likelihood up to a constant."""
    total = 0.0
    for row, y in zip(X, Y, strict=True):
        r = y - sum(a * b for a, b in zip(row, theta, strict=True))
        total += r * r
    return total / (2.0 * SIGMA**2)


def _mle_and_inverse() -> tuple[tuple[float, ...], tuple[tuple[float, ...], ...]]:
    """Return the least-squares estimate and ``(X'X)^-1``."""
    xt = transpose(X)
    gram = matmul(xt, X)
    rhs = tuple(sum(a * b for a, b in zip(row, Y, strict=True)) for row in xt)
    return solve_spd(gram, rhs), inverse_spd(gram)


class LinearGaussianTest(unittest.TestCase):
    """Closed-form profile of a linear Gaussian model with known sigma."""

    def test_profile_is_exactly_quadratic(self) -> None:
        """LR(t) = (t - t_hat)^2 / (sigma^2 [(X'X)^-1]_tt)."""
        theta_hat, inv = _mle_and_inverse()
        for index in (0, 1, 2):
            se = SIGMA * math.sqrt(inv[index][index])
            centre = theta_hat[index]
            grid = tuple(centre + se * (k / 4.0) for k in range(-12, 13))
            res = profile_likelihood(_gaussian_nll, theta_hat, index, grid)
            self.assertTrue(res.all_converged)
            self.assertFalse(res.improved_estimate_found)
            for t, lr in zip(res.grid, res.likelihood_ratio, strict=True):
                self.assertAlmostEqual(lr, ((t - centre) / se) ** 2, delta=2e-6)

    def test_interval_matches_wald(self) -> None:
        """The 95 % interval is t_hat +- z sigma sqrt([(X'X)^-1]_tt)."""
        theta_hat, inv = _mle_and_inverse()
        se = SIGMA * math.sqrt(inv[1][1])
        z = math.sqrt(ChiSquare(1.0).quantile(0.95))
        grid = tuple(theta_hat[1] + se * (k / 10.0) for k in range(-30, 31))
        out = profile_interval(profile_likelihood(_gaussian_nll, theta_hat, 1, grid))
        assert out.lower is not None and out.upper is not None
        self.assertEqual((out.lower_status, out.upper_status), ("bounded",) * 2)
        self.assertTrue(out.connected)
        # chord of a convex LR: endpoint error <= ~6e-4 se on this grid
        self.assertAlmostEqual(out.lower, theta_hat[1] - z * se, delta=2e-3 * se)
        self.assertAlmostEqual(out.upper, theta_hat[1] + z * se, delta=2e-3 * se)

    def test_start_labels_and_accounting(self) -> None:
        """Evaluations are counted and the winning start is recorded."""
        theta_hat, inv = _mle_and_inverse()
        se = SIGMA * math.sqrt(inv[0][0])
        grid = tuple(theta_hat[0] + se * k for k in (-1, 0, 1))
        res = profile_likelihood(_gaussian_nll, theta_hat, 0, grid)
        self.assertEqual(
            sum(p.evaluations for p in res.points) + 1, res.total_evaluations
        )
        self.assertTrue(all(p.start in ("warm", "estimate") for p in res.points))


class NonIdentifiabilityTest(unittest.TestCase):
    """Structural flatness and multimodality must be reported, not hidden."""

    def test_product_model_is_flat_and_unbounded(self) -> None:
        """For y = a b the profile of a is flat and both sides are unbounded."""

        def nll(theta: tuple[float, ...]) -> float:
            return 0.5 * (2.0 - theta[0] * theta[1]) ** 2

        grid = tuple(0.5 + 0.25 * k for k in range(15))
        res = profile_likelihood(nll, (1.0, 2.0), 0, grid)
        self.assertLess(max(res.likelihood_ratio), 1e-6)
        out = profile_interval(res)
        self.assertIsNone(out.lower)
        self.assertIsNone(out.upper)
        self.assertEqual(out.lower_status, "unbounded_within_grid")
        self.assertEqual(out.upper_status, "unbounded_within_grid")

    def test_sign_symmetry_keeps_second_mode(self) -> None:
        """For y = theta^2 + noise, LR near -theta_hat is ~0; set is disconnected."""
        data = (3.9, 4.2, 4.1, 3.8, 4.0)
        theta_hat = math.sqrt(sum(data) / len(data))

        def nll(theta: tuple[float, ...]) -> float:
            return sum((y - theta[0] ** 2) ** 2 for y in data) / (2.0 * 0.3**2)

        grid = sorted({-theta_hat, theta_hat, *(0.25 * k for k in range(-12, 13))})
        res = profile_likelihood(nll, (theta_hat,), 0, grid)
        near_neg = [
            lr
            for t, lr in zip(res.grid, res.likelihood_ratio, strict=True)
            if t == -theta_hat
        ]
        self.assertLess(near_neg[0], 1e-10)
        out = profile_interval(res)
        self.assertFalse(out.connected)
        # the two modes tie; the component is around one, the other is reported
        self.assertIn(abs(out.other_regions[0]), (theta_hat,))
        self.assertEqual(len(out.other_regions), 1)
        # the barrier between the modes is resolved, not collapsed
        mid = res.likelihood_ratio[res.grid.index(0.0)]
        self.assertGreater(mid, out.threshold)

    def test_unbounded_one_side_only(self) -> None:
        """A profile that rises on one side and is flat on the other."""

        def nll(theta: tuple[float, ...]) -> float:
            return 0.5 * max(1.0 - theta[0], 0.0) ** 2 * 50.0

        grid = tuple(0.05 * k for k in range(0, 81))
        res = profile_likelihood(nll, (2.0,), 0, grid)
        out = profile_interval(res)
        self.assertEqual(out.upper_status, "unbounded_within_grid")
        self.assertIsNone(out.upper)
        self.assertEqual(out.lower_status, "bounded")
        assert out.lower is not None
        self.assertAlmostEqual(
            out.lower, 1.0 - math.sqrt(ChiSquare(1.0).quantile(0.95) / 50.0), delta=1e-2
        )


class BoundaryAndBaselineTest(unittest.TestCase):
    """Active nuisance bound, improved-estimate detection, validation."""

    @staticmethod
    def _nll(theta: tuple[float, ...]) -> float:
        a, b = theta
        return 0.5 * (a - 1.0) ** 2 + 0.5 * (b - a) ** 2

    def test_active_nuisance_bound(self) -> None:
        """With b >= 2 the profile of a is the piecewise closed form."""
        grid = tuple(0.5 + 0.25 * k for k in range(0, 13))
        res = profile_likelihood(self._nll, (1.5, 2.0), 0, grid, lower=(None, 2.0))
        self.assertFalse(res.improved_estimate_found)
        for t, p in zip(res.grid, res.profile_values, strict=True):
            exact = 0.5 * (t - 1.0) ** 2 + 0.5 * max(2.0 - t, 0.0) ** 2
            self.assertAlmostEqual(p, exact, delta=1e-6)
        for pt in res.points:
            self.assertGreaterEqual(pt.nuisance[0], 2.0 - 1e-9)

    def test_improved_estimate_detected_not_clipped(self) -> None:
        """A poor supplied estimate is reported and the baseline is refit."""
        grid = tuple(-1.0 + 0.25 * k for k in range(0, 17))
        res = profile_likelihood(self._nll, (0.0, 0.0), 0, grid)
        self.assertTrue(res.improved_estimate_found)
        self.assertLess(res.nll_min, res.nll_at_estimate)
        self.assertGreaterEqual(min(res.likelihood_ratio), 0.0)
        self.assertAlmostEqual(min(res.likelihood_ratio), 0.0, places=9)
        assert res.improved_estimate is not None
        self.assertAlmostEqual(res.improved_estimate[0], 1.0, places=6)
        self.assertAlmostEqual(res.improved_estimate[1], 1.0, places=5)
        self.assertTrue(profile_interval(res).improved_estimate_found)

    def test_extra_start_used_when_needed(self) -> None:
        """An infeasible warm region falls back to an extra start."""

        def nll(theta: tuple[float, ...]) -> float:
            if theta[1] < 0.0:
                raise ValueError("domain")
            return 0.5 * (theta[0] - 1.0) ** 2 + 0.5 * (theta[1] - 1.0) ** 2

        res = profile_likelihood(nll, (1.0, 1.0), 0, (0.5, 1.0, 1.5), starts=((2.0,),))
        self.assertAlmostEqual(res.profile_values[1], 0.0, places=8)

    def test_validation(self) -> None:
        """Invalid inputs are rejected."""
        nll = self._nll
        with self.assertRaises(ValueError):
            profile_likelihood(nll, (1.0, 1.0), 0, (1.5, 2.0))  # does not bracket
        with self.assertRaises(ValueError):
            profile_likelihood(nll, (1.0, 1.0), 0, (2.0, 0.0))  # not increasing
        with self.assertRaises(ValueError):
            profile_likelihood(nll, (1.0, 1.0), 2, (0.0, 2.0))
        with self.assertRaises(ValueError):
            profile_likelihood(nll, (1.0, 1.0), 0, (0.0, 2.0), optimizer="x")
        with self.assertRaises(ValueError):
            profile_likelihood(nll, (1.0, 1.0), 0, (0.0, 2.0), lower=(None, 3.0))
        with self.assertRaises(TypeError):
            profile_likelihood(1.0, (1.0, 1.0), 0, (0.0, 2.0))  # type: ignore[arg-type]
        res = profile_likelihood(nll, (1.0, 1.0), 0, (0.0, 1.0, 2.0))
        self.assertIsInstance(res, ProfileLikelihoodResult)
        for bad in (0.0, 1.0, -0.1):
            with self.assertRaises(ValueError):
                profile_interval(res, bad)
        with self.assertRaises(TypeError):
            profile_interval("x")  # type: ignore[arg-type]

    def test_nonfinite_neighbour_gives_no_endpoint(self) -> None:
        """A non-finite profile next to the crossing yields no fabricated limit."""

        def nll(theta: tuple[float, ...]) -> float:
            if theta[0] > 1.6:
                raise ValueError("outside domain")
            return (
                0.5 * (theta[0] - 1.0) ** 2 * 100.0 * 0.0 + 0.5 * (theta[0] - 1.0) ** 2
            )

        res = profile_likelihood(nll, (1.0,), 0, (0.0, 1.0, 2.0))
        out = profile_interval(res)
        self.assertIsNone(out.upper)
        self.assertEqual(out.upper_status, "undetermined_nonfinite")


class InformationTest(unittest.TestCase):
    """Efficient information, delta method and estimability."""

    F = ((4.0, 1.0, 0.5), (1.0, 3.0, -0.4), (0.5, -0.4, 2.0))

    def test_efficient_information_equals_inverse_of_inverse_block(self) -> None:
        """S_tt = 1 / (F^-1)_tt for one target; block version for two."""
        inv = inverse_spd(self.F)
        for t in (0, 1, 2):
            s = efficient_information(self.F, [t])
            self.assertAlmostEqual(s[0][0], 1.0 / inv[t][t], places=12)
        s2 = efficient_information(self.F, [0, 2])
        block = ((inv[0][0], inv[0][2]), (inv[2][0], inv[2][2]))
        back = inverse_spd(s2)
        for i in range(2):
            for j in range(2):
                self.assertAlmostEqual(back[i][j], block[i][j], places=12)
        self.assertEqual(efficient_information(self.F, [0, 1, 2]), self.F)

    def test_matches_profile_curvature(self) -> None:
        """Profile of the linear model has curvature equal to S_tt."""
        _, inv = _mle_and_inverse()
        gram = matmul(transpose(X), X)
        fisher = tuple(tuple(v / SIGMA**2 for v in row) for row in gram)
        eff = efficient_information(fisher, [2])[0][0]
        self.assertAlmostEqual(eff, 1.0 / (SIGMA**2 * inv[2][2]), places=10)

    def test_singular_nuisance_block_raises(self) -> None:
        """A rank-deficient F_nn means the nuisance block is not estimable."""
        singular = ((2.0, 0.5, 0.5), (0.5, 1.0, 1.0), (0.5, 1.0, 1.0))
        with self.assertRaisesRegex(ValueError, "not estimable"):
            efficient_information(singular, [0])
        # the full matrix being singular is fine when targets absorb the null
        self.assertEqual(len(efficient_information(singular, [1, 2])), 2)
        with self.assertRaises(ValueError):
            efficient_information(((1.0, 2.0), (0.0, 1.0)), [0])  # asymmetric
        with self.assertRaises(ValueError):
            efficient_information(self.F, [0, 0])
        with self.assertRaises(ValueError):
            efficient_information(self.F, [])

    def test_delta_method_variance(self) -> None:
        """The form c' C c against a hand computation; non-PSD rejected."""
        c = (1.0, -2.0)
        cov = ((2.0, 0.5), (0.5, 1.0))
        self.assertAlmostEqual(
            delta_method_variance(c, cov), 2.0 - 2.0 + 4.0, places=14
        )
        with self.assertRaises(ValueError):
            delta_method_variance((1.0, -1.0), ((1.0, 2.0), (2.0, 1.0)))
        with self.assertRaises(ValueError):
            delta_method_variance((1.0,), cov)

    def test_functional_estimability(self) -> None:
        """A c in the row space is estimable; c in the null space is not."""
        jac = ((1.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        fisher = matmul(transpose(jac), jac)
        for mat in (jac, fisher):
            ok = functional_estimability((2.0, 2.0, -1.0), mat)
            self.assertTrue(ok.estimable)
            self.assertEqual((ok.rank, ok.null_dimension), (2, 1))
            bad = functional_estimability((1.0, -1.0, 0.0), mat)
            self.assertFalse(bad.estimable)
            self.assertAlmostEqual(bad.relative_null_component, 1.0, places=12)
            mixed = functional_estimability((1.0, 0.0, 0.0), mat)
            self.assertFalse(mixed.estimable)
            self.assertAlmostEqual(
                mixed.relative_null_component, math.sqrt(0.5), places=12
            )
        with self.assertRaises(ValueError):
            functional_estimability((0.0, 0.0, 0.0), jac)
        with self.assertRaises(ValueError):
            functional_estimability((1.0, 0.0), jac)


if __name__ == "__main__":
    unittest.main()
