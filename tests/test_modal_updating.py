"""Tests for Bayesian structural updating from modal data.

Order: closed-form 2-DOF eigenproblem and one-mass likelihood/evidence, then
analytic eigenvalue sensitivities against central differences, then seeded
synthetic damage cases (coverage, localization, class selection), then the
mirror-symmetric non-identifiable case and failure cases. All data are
synthetic. Stochastic claims use predeclared seeds and are judged by measured
frequencies with bands, not pinned digits.
"""

from __future__ import annotations

import math
import os
import random
import unittest

from probstat.inverse.modal_updating import (
    ModalData,
    ModalStructure,
    ModalUpdatingProblem,
    damage_class_selection,
    generalized_eigen,
    mac,
    match_modes,
    modal_laplace,
    modal_tmcmc,
    simulate_modal_data,
)

SEEDS = tuple(range(10))
CHAIN = ModalStructure.spring_mass_chain([2.0] * 4, [1.0e6] * 4)
DAMAGED = (1.0, 0.7, 1.0, 1.0)
DOFS = (1, 2, 3)


def chain_data(theta: tuple[float, ...], seed: int) -> ModalData:
    """Return noisy 3-mode data with shapes at three dofs of the chain."""
    return simulate_modal_data(
        CHAIN,
        theta,
        (0, 1, 2),
        frequency_std=0.01,
        dofs=DOFS,
        shape_std=0.02,
        seed=seed,
    )


class EigenSolverTests(unittest.TestCase):
    """Generalized eigenproblem against closed forms."""

    def test_two_dof_closed_form(self) -> None:
        """Unit masses and springs, fixed base: lambda = (3 -+ sqrt 5) / 2."""
        s = ModalStructure.spring_mass_chain([1.0, 1.0], [1.0, 1.0])
        values = s.modes((1.0, 1.0)).eigenvalues
        self.assertAlmostEqual(values[0], (3.0 - math.sqrt(5.0)) / 2.0, places=12)
        self.assertAlmostEqual(values[1], (3.0 + math.sqrt(5.0)) / 2.0, places=12)

    def test_unequal_masses_closed_form(self) -> None:
        """Two masses (m, 2m), springs (k, k): det(K - lambda M) = 0 by hand."""
        k, m = 3.0, 0.5
        s = ModalStructure.spring_mass_chain([m, 2.0 * m], [k, k])
        # K = k [[2,-1],[-1,1]], M = m diag(1,2): 2 m^2 l^2 - 5 k m l + k^2 = 0
        disc = math.sqrt(25.0 - 8.0)
        expected = ((5.0 - disc) * k / (4.0 * m), (5.0 + disc) * k / (4.0 * m))
        values = s.modes((1.0, 1.0)).eigenvalues
        for got, want in zip(values, expected, strict=True):
            self.assertAlmostEqual(got, want, places=12)

    def test_mass_orthonormality_and_residual(self) -> None:
        """Shapes are mass-orthonormal and K phi = lambda M phi for a non-diagonal M."""
        mass = ((2.0, 0.3, 0.0), (0.3, 1.5, 0.2), (0.0, 0.2, 1.0))
        stiff = ((4.0, -1.0, 0.0), (-1.0, 3.0, -1.0), (0.0, -1.0, 2.0))
        values, shapes = generalized_eigen(stiff, mass)
        cols = tuple(zip(*shapes, strict=True))
        for i, a in enumerate(cols):
            ka = [sum(stiff[r][c] * a[c] for c in range(3)) for r in range(3)]
            ma = [sum(mass[r][c] * a[c] for c in range(3)) for r in range(3)]
            for r in range(3):
                self.assertAlmostEqual(ka[r], values[i] * ma[r], places=10)
            for j, b in enumerate(cols):
                inner = sum(b[r] * ma[r] for r in range(3))
                self.assertAlmostEqual(inner, 1.0 if i == j else 0.0, places=10)

    def test_invalid_inputs(self) -> None:
        """Non-symmetric, non-SPD mass, size mismatch and rigid modes fail."""
        eye = ((1.0, 0.0), (0.0, 1.0))
        with self.assertRaises(ValueError):
            generalized_eigen(((1.0, 0.5), (0.0, 1.0)), eye)
        with self.assertRaises(ValueError):
            generalized_eigen(eye, ((1.0, 0.0), (0.0, -1.0)))
        with self.assertRaises(ValueError):
            generalized_eigen(eye, ((1.0,),))
        free = ((1.0, -1.0), (-1.0, 1.0))
        with self.assertRaises(ValueError):
            ModalStructure(eye, [free])  # K(1) singular (rigid-body mode)
        with self.assertRaises(ValueError):
            ModalStructure.spring_mass_chain([1.0, 1.0], [1.0])


class SensitivityTests(unittest.TestCase):
    """d lambda / d theta = phi' K_i phi against central differences."""

    def test_against_finite_differences(self) -> None:
        """Relative agreement ~1e-6 (FD error ~h^2 = 1e-8, roundoff ~1e-12)."""
        theta = (1.0, 0.7, 0.9, 1.1)
        analytic = CHAIN.eigenvalue_sensitivities(theta)
        h = 1e-4
        for i in range(4):
            up = list(theta)
            dn = list(theta)
            up[i] += h
            dn[i] -= h
            lu = CHAIN.modes(up).eigenvalues
            ld = CHAIN.modes(dn).eigenvalues
            for j in range(4):
                fd = (lu[j] - ld[j]) / (2.0 * h)
                self.assertAlmostEqual(analytic[j][i] / fd, 1.0, places=6)

    def test_single_mass_exact(self) -> None:
        """The eigenvalue k theta / m has derivative k / m."""
        s = ModalStructure.spring_mass_chain([4.0], [8.0])
        self.assertAlmostEqual(s.eigenvalue_sensitivities((1.3,))[0][0], 2.0)

    def test_repeated_eigenvalue_rejected(self) -> None:
        """A double eigenvalue has no derivative; it is rejected."""
        eye = ((1.0, 0.0), (0.0, 1.0))
        s = ModalStructure(eye, [((1.0, 0.0), (0.0, 0.0)), ((0.0, 0.0), (0.0, 1.0))])
        with self.assertRaises(ValueError):
            s.eigenvalue_sensitivities((1.0, 1.0))


class ModeMatchingTests(unittest.TestCase):
    """MAC and greedy mode matching."""

    def test_mac_properties(self) -> None:
        """Scale and sign invariance, orthogonality, validation."""
        self.assertAlmostEqual(mac((1.0, 2.0), (-3.0, -6.0)), 1.0, places=14)
        self.assertEqual(mac((1.0, 0.0), (0.0, 2.0)), 0.0)
        with self.assertRaises(ValueError):
            mac((0.0, 0.0), (1.0, 1.0))
        with self.assertRaises(ValueError):
            mac((1.0,), (1.0, 2.0))

    def test_swapped_order_is_recovered(self) -> None:
        """Data modes listed in reverse order are assigned correctly."""
        sol = CHAIN.modes((1.0,) * 4)
        data = [tuple(sol.shape(j)[i] * 2.0 for i in DOFS) for j in (2, 0)]
        found = match_modes(sol.shapes, data, DOFS)
        self.assertEqual(found.model_indices, (2, 0))
        self.assertTrue(all(v > 0.999999 for v in found.mac_values))

    def test_too_many_data_modes(self) -> None:
        """More data modes than model modes is an error."""
        with self.assertRaises(ValueError):
            match_modes(((1.0,),), [(1.0,), (2.0,)], (0,))


class LikelihoodAndEvidenceTests(unittest.TestCase):
    """One-mass case where likelihood, MAP and evidence are closed forms."""

    k, m, delta, s = 8.0e5, 2.0, 0.02, 0.4

    def setUp(self) -> None:
        """Build the one-mass problem with a fixed observation."""
        self.struct = ModalStructure.spring_mass_chain([self.m], [self.k])
        self.c = 0.5 * math.log(self.k / self.m) - math.log(2.0 * math.pi)
        self.y = self.c - 0.03  # observed log frequency
        self.data = ModalData((math.exp(self.y),), (0,), self.delta)
        self.problem = ModalUpdatingProblem(
            self.struct, self.data, prior_log_std=self.s
        )
        var = self.delta**2 + 0.25 * self.s**2
        self.log_z = (
            -0.5 * math.log(2 * math.pi * var) - 0.5 * (self.y - self.c) ** 2 / var
        )

    def test_log_likelihood_closed_form(self) -> None:
        """Check log L = -log(sqrt(2 pi) delta) - 1/2 ((y - c - u/2)/delta)^2."""
        for u in (-0.2, 0.0, 0.35):
            want = (
                -0.5 * math.log(2 * math.pi * self.delta**2)
                - 0.5 * ((self.y - self.c - 0.5 * u) / self.delta) ** 2
            )
            self.assertAlmostEqual(self.problem.log_likelihood((u,)), want, places=11)

    def test_laplace_is_exact_for_linear_gaussian(self) -> None:
        """MAP, variance and evidence agree with the conjugate formulas."""
        a = 0.25 / self.delta**2 + 1.0 / self.s**2
        u_star = 0.5 * (self.y - self.c) / self.delta**2 / a
        fit = modal_laplace(self.problem)
        self.assertTrue(fit.converged)
        self.assertAlmostEqual(fit.u_map[0], u_star, places=6)
        assert fit.covariance_u is not None
        self.assertAlmostEqual(fit.covariance_u[0][0] * a, 1.0, places=5)
        self.assertAlmostEqual(fit.log_evidence, self.log_z, places=5)

    def test_tmcmc_evidence_matches_closed_form(self) -> None:
        """Mean of three TMCMC log evidences within 0.15 nats of the exact one."""
        estimates = [
            modal_tmcmc(self.problem, particles=600, seed=seed).log_evidence
            for seed in (1, 2, 3)
        ]
        mean = sum(estimates) / 3.0
        self.assertLess(abs(mean - self.log_z), 0.15, msg=f"{estimates} {self.log_z}")

    def test_shape_scale_and_sign_are_nuisance(self) -> None:
        """Scaling or flipping the whole measured shape leaves log L alone."""
        sol = CHAIN.modes((1.0, 0.8, 1.0, 1.0))
        base = tuple(sol.shape(0)[i] + 0.01 * k for k, i in enumerate(DOFS))
        values = []
        for factor in (1.0, 3.0, -0.5):
            data = ModalData(
                (30.0,), (0,), 0.01, DOFS, (tuple(factor * v for v in base),), 0.02
            )
            problem = ModalUpdatingProblem(CHAIN, data)
            values.append(problem.log_likelihood((0.0, math.log(0.8), 0.0, 0.0)))
        self.assertAlmostEqual(values[0], values[1], places=10)
        self.assertAlmostEqual(values[0], values[2], places=10)


class DamageUpdatingTests(unittest.TestCase):
    """Synthetic damage in segment 1 of a four-segment chain."""

    def test_laplace_covers_truth_and_localizes(self) -> None:
        """Predeclared seeds 0-9: coverage count and argmax damage segment."""
        covered = 0
        total = 0
        located = 0
        for seed in SEEDS:
            fit = modal_laplace(ModalUpdatingProblem(CHAIN, chain_data(DAMAGED, seed)))
            self.assertTrue(fit.converged, msg=f"seed {seed}: {fit.message}")
            assert fit.covariance_u is not None
            for i in range(4):
                sd = math.sqrt(fit.covariance_u[i][i])
                total += 1
                covered += abs(fit.u_map[i] - math.log(DAMAGED[i])) <= 1.96 * sd
            probs = [fit.damage_probability(i, 0.1) for i in range(4)]
            located += probs.index(max(probs)) == 1
        print(f"laplace coverage {covered}/{total}, localized {located}/10")
        self.assertGreaterEqual(covered, 34, msg=f"covered {covered}/{total}")
        self.assertEqual(located, len(SEEDS))

    def test_damage_probability_values(self) -> None:
        """Seed 0: damaged segment above 0.99, healthy ones below 0.2."""
        fit = modal_laplace(ModalUpdatingProblem(CHAIN, chain_data(DAMAGED, 0)))
        self.assertGreater(fit.damage_probability(1, 0.1), 0.99)
        for i in (0, 2, 3):
            self.assertLess(fit.damage_probability(i, 0.1), 0.2)

    def test_tmcmc_agrees_with_laplace(self) -> None:
        """TMCMC mean of log theta within half a posterior sd of the MAP."""
        problem = ModalUpdatingProblem(CHAIN, chain_data(DAMAGED, 1))
        fit = modal_laplace(problem)
        run = modal_tmcmc(problem, particles=300, seed=7)
        self.assertTrue(run.converged)
        samples = run.theta_samples
        assert fit.covariance_u is not None
        for i in range(4):
            mean_u = sum(math.log(s[i]) for s in samples) / len(samples)
            sd = math.sqrt(fit.covariance_u[i][i])
            self.assertLess(abs(mean_u - fit.u_map[i]), 0.5 * sd + 0.01)
        self.assertGreater(run.damage_probability(1, 0.1), 0.95)
        self.assertLess(run.damage_probability(3, 0.1), 0.3)

    def test_class_selection_picks_damaged_segment(self) -> None:
        """Measured class probabilities for predeclared seed 0: damage_1 wins."""
        for seed in (0,):
            sel = damage_class_selection(
                CHAIN, chain_data(DAMAGED, seed), particles=250, seed=seed
            )
            self.assertEqual(sel.best, "damage_1", msg=str(sel.selection))
            self.assertGreater(sel.probability("damage_1"), 0.9)

    def test_class_selection_undamaged_data(self) -> None:
        """Undamaged truth: the undamaged class has the largest probability."""
        sel = damage_class_selection(
            CHAIN, chain_data((1.0,) * 4, 4), particles=250, seed=4
        )
        self.assertEqual(sel.best, "undamaged", msg=str(sel.selection))

    def test_shapes_narrow_the_posterior(self) -> None:
        """Dropping the shapes does not narrow the posterior of theta_1."""
        sd = []
        for with_shapes in (True, False):
            data = simulate_modal_data(
                CHAIN,
                DAMAGED,
                (0, 1, 2),
                frequency_std=0.01,
                dofs=DOFS if with_shapes else (),
                shape_std=0.02,
                seed=0,
            )
            fit = modal_laplace(
                ModalUpdatingProblem(CHAIN, data, active=(1,)), initial=(-0.3,)
            )
            assert fit.covariance_u is not None
            sd.append(math.sqrt(fit.covariance_u[0][0]))
        self.assertLess(sd[0], sd[1])


class NonIdentifiableTests(unittest.TestCase):
    """Mirror-symmetric two-mass structure, springs 0 and 2 updated."""

    struct = ModalStructure.spring_mass_chain(
        [1.0, 1.0], [1.0, 1.0, 1.0], grounded="both", updated=[0, 2]
    )
    truth = (0.6, 1.0)

    def test_swap_symmetry_is_exact(self) -> None:
        """Frequencies are identical for (a, b) and (b, a)."""
        f1 = self.struct.modes((0.6, 1.0)).eigenvalues
        f2 = self.struct.modes((1.0, 0.6)).eigenvalues
        for a, b in zip(f1, f2, strict=True):
            self.assertAlmostEqual(a, b, places=12)

    def test_frequency_only_has_two_modes(self) -> None:
        """Laplace sits in one mode per start; TMCMC puts mass on both."""
        data = simulate_modal_data(
            self.struct, self.truth, (0, 1), frequency_std=0.005, seed=3
        )
        problem = ModalUpdatingProblem(self.struct, data)
        left = modal_laplace(problem, (math.log(0.7), 0.0))
        right = modal_laplace(problem, (0.0, math.log(0.7)))
        self.assertLess(left.theta_map[0], left.theta_map[1] - 0.3)
        self.assertGreater(right.theta_map[0], right.theta_map[1] + 0.3)
        # Each Laplace posterior is certain about which spring is damaged.
        self.assertGreater(left.damage_probability(0, 0.1), 0.99)
        self.assertLess(right.damage_probability(0, 0.1), 0.01)
        run = modal_tmcmc(problem, particles=400, seed=5)
        self.assertTrue(run.converged)
        samples = run.theta_samples
        share = sum(1 for s in samples if s[0] < s[1]) / len(samples)
        print(f"frequency-only share with theta_0 < theta_1: {share}")
        self.assertGreater(share, 0.3, msg=f"share {share}")
        self.assertLess(share, 0.7, msg=f"share {share}")

    def test_shapes_break_the_symmetry(self) -> None:
        """With shapes at both dofs, the true mode has more than 95 percent."""
        data = simulate_modal_data(
            self.struct,
            self.truth,
            (0, 1),
            frequency_std=0.005,
            dofs=(0, 1),
            shape_std=0.01,
            seed=3,
        )
        problem = ModalUpdatingProblem(self.struct, data)
        run = modal_tmcmc(problem, particles=400, seed=5)
        self.assertTrue(run.converged)
        samples = run.theta_samples
        share = sum(1 for s in samples if s[0] < s[1]) / len(samples)
        print(f"with shapes share with theta_0 < theta_1: {share}")
        self.assertGreater(share, 0.95, msg=f"share {share}")


class FailureTests(unittest.TestCase):
    """Invalid input and numerical failure are reported, not hidden."""

    def test_modal_data_validation(self) -> None:
        """Bad frequencies, modes, dofs and shapes are rejected."""
        with self.assertRaises(ValueError):
            ModalData((-1.0,), (0,), 0.01)
        with self.assertRaises(ValueError):
            ModalData((1.0, 2.0), (0, 0), 0.01)
        with self.assertRaises(ValueError):
            ModalData((1.0,), (0,), 0.0)
        with self.assertRaises(ValueError):
            ModalData((1.0,), (0,), 0.01, (0,), ((1.0,),), 0.1)  # one dof
        with self.assertRaises(ValueError):
            ModalData((1.0,), (0,), 0.01, (0, 1), ((0.0, 0.0),), 0.1)
        with self.assertRaises(ValueError):
            ModalData((1.0,), (0,), 0.01, (0, 1), ((1.0, 1.0),), 0.0)
        with self.assertRaises(TypeError):
            ModalData(("a",), (0,), 0.01)  # type: ignore[arg-type]

    def test_problem_validation(self) -> None:
        """Out-of-range modes, active indices and priors fail."""
        with self.assertRaises(ValueError):
            ModalUpdatingProblem(CHAIN, ModalData((10.0,), (7,), 0.01))
        good = ModalData((10.0,), (0,), 0.01)
        with self.assertRaises(ValueError):
            ModalUpdatingProblem(CHAIN, good, active=(0, 0))
        with self.assertRaises(ValueError):
            ModalUpdatingProblem(CHAIN, good, prior_log_std=-1.0)
        with self.assertRaises(ValueError):
            ModalUpdatingProblem(CHAIN, good, prior_median=(1.0, 1.0))
        with self.assertRaises(TypeError):
            ModalUpdatingProblem("x", good)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            modal_laplace(ModalUpdatingProblem(CHAIN, good, active=()))

    def test_indefinite_stiffness_has_zero_likelihood(self) -> None:
        """A parameter region with K(theta) indefinite gives -inf, not NaN."""
        eye = ((1.0, 0.0), (0.0, 1.0))
        s = ModalStructure(eye, [((2.0, 0.0), (0.0, 2.0)), ((-0.1, 0.0), (0.0, -0.1))])
        problem = ModalUpdatingProblem(s, ModalData((0.2,), (0,), 0.05))
        self.assertEqual(problem.log_likelihood((0.0, math.log(30.0))), -math.inf)
        with self.assertRaises(ValueError):
            problem.residuals((0.0, math.log(30.0)))
        self.assertTrue(math.isfinite(problem.log_likelihood((0.0, 0.0))))

    def test_nonconverged_tmcmc_has_no_summaries(self) -> None:
        """A run stopped by max_stages raises on the summaries."""
        problem = ModalUpdatingProblem(CHAIN, chain_data(DAMAGED, 0))
        run = modal_tmcmc(problem, particles=50, seed=0, max_stages=1)
        self.assertFalse(run.converged)
        with self.assertRaises(ValueError):
            run.damage_probability(1, 0.1)
        self.assertTrue(math.isnan(run.log_evidence))

    def test_damage_probability_arguments(self) -> None:
        """Fraction and index validation."""
        fit = modal_laplace(ModalUpdatingProblem(CHAIN, chain_data(DAMAGED, 0)))
        with self.assertRaises(ValueError):
            fit.damage_probability(1, 1.5)
        with self.assertRaises(ValueError):
            fit.damage_probability(9, 0.1)

    def test_truncated_prior_rejects_impossible_region(self) -> None:
        """A prior far above the upper bound cannot be sampled."""
        problem = ModalUpdatingProblem(
            CHAIN,
            chain_data(DAMAGED, 0),
            prior_median=1.0e6,
            prior_log_std=0.1,
            upper=1.0,
        )
        with self.assertRaises(ValueError):
            problem.prior_sample(random.Random(0))


@unittest.skipUnless(os.environ.get("PROBSTAT_SLOW") == "1", "slow tier")
class SlowCoverageTests(unittest.TestCase):
    """TMCMC coverage and class selection over ten seeds."""

    def test_tmcmc_coverage_and_selection(self) -> None:
        """Central 95 percent sample intervals of theta over seeds 0-9."""
        covered = 0
        total = 0
        wins = 0
        for seed in SEEDS:
            data = chain_data(DAMAGED, seed)
            run = modal_tmcmc(
                ModalUpdatingProblem(CHAIN, data), particles=500, seed=seed
            )
            self.assertTrue(run.converged)
            samples = run.theta_samples
            for i in range(4):
                values = sorted(s[i] for s in samples)
                lo = values[int(0.025 * len(values))]
                hi = values[int(0.975 * len(values))]
                total += 1
                covered += lo <= DAMAGED[i] <= hi
            sel = damage_class_selection(CHAIN, data, particles=400, seed=seed)
            wins += sel.best == "damage_1"
        print(f"slow: TMCMC coverage {covered}/{total}, class wins {wins}/10")
        self.assertGreaterEqual(covered, 34)
        self.assertEqual(wins, 10)


if __name__ == "__main__":
    unittest.main()
