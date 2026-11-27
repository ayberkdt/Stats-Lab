"""Tests for rigid-body inertia identification from manoeuvre data.

Order: the Euler integrator against the exact axisymmetric solution and the
torque-free invariants, then the parameterization and realizability, then
noise-free recovery by the regression and by Levenberg-Marquardt, then seeded
noisy-data studies (regression bias versus output-error, Laplace coverage),
then experiment design (identifiability under one-axis torques) and failure
cases. All data are synthetic. Stochastic claims use predeclared seeds and
bands from measured quantities, not pinned digits.
"""

from __future__ import annotations

import math
import os
import statistics
import unittest

from probstat.inverse.inertia_identification import (
    EXCITATION_KINDS,
    InertiaData,
    InertiaLaplace,
    InertiaProblem,
    angular_momentum,
    assess_excitation,
    check_realizable,
    excitation_torques,
    fisher_information,
    inertia_entries,
    inertia_from_theta,
    inertia_laplace,
    principal_axes,
    regression_estimate,
    rotational_energy,
    simulate_inertia_data,
    simulate_rates,
    theta_from_inertia,
    weak_prior,
)
from probstat.optimization import central_jacobian

TRUTH = ((10.0, 0.4, -0.3), (0.4, 12.0, 0.5), (-0.3, 0.5, 15.0))
TRUE_ENTRIES = inertia_entries(TRUTH)
NOMINAL = ((11.0, 0.0, 0.0), (0.0, 11.0, 0.0), (0.0, 0.0, 11.0))
DT = 1.0
STEPS = 240
SIGMA = 2e-3
SEEDS = tuple(range(8))


def manoeuvre(kind: str = "sequential", steps: int = STEPS) -> tuple:
    """Return the standard test torque sequence (0.2 N m peak)."""
    return excitation_torques(kind, steps=steps, amplitude=0.2, pulse_steps=10)


def max_abs_error(entries: tuple[float, ...]) -> float:
    """Return the largest absolute entry error against the truth."""
    return max(abs(a - b) for a, b in zip(entries, TRUE_ENTRIES, strict=True))


def fit(seed: int, *, disturbance: bool = False) -> InertiaLaplace:
    """Fit one seeded noisy data set from the nominal start."""
    data = simulate_inertia_data(TRUTH, manoeuvre(), DT, noise_std=SIGMA, seed=seed)
    mean, std = weak_prior(NOMINAL)
    problem = InertiaProblem(data, prior_mean=mean, prior_std=std)
    return inertia_laplace(problem, mean)


class EulerIntegratorTests(unittest.TestCase):
    """RK4 Euler dynamics against exact solutions and invariants."""

    def test_axisymmetric_closed_form(self) -> None:
        """Axisymmetric closed form.

        For J = diag(I, I, Iz) torque-free, wz is constant and (wx, wy)
        rotate at Omega = (Iz - I) wz / I: an exact solution.
        """
        inertia, iz = 4.0, 6.0
        j = ((inertia, 0.0, 0.0), (0.0, inertia, 0.0), (0.0, 0.0, iz))
        w0 = (0.3, -0.1, 0.5)
        omega = (iz - inertia) * w0[2] / inertia
        dt, n = 0.1, 100
        path = simulate_rates(j, [(0.0, 0.0, 0.0)] * n, dt, initial_rate=w0)
        t = n * dt
        exact = (
            w0[0] * math.cos(omega * t) - w0[1] * math.sin(omega * t),
            w0[0] * math.sin(omega * t) + w0[1] * math.cos(omega * t),
            w0[2],
        )
        for got, want in zip(path[-1], exact, strict=True):
            self.assertAlmostEqual(got, want, delta=1e-7)

    def test_torque_free_invariants_and_order(self) -> None:
        """Energy and momentum-magnitude drift are RK4-small and fall ~h^4."""
        w0 = (0.3, 0.1, -0.2)
        drifts = []
        for h in (0.2, 0.1):
            n = round(20.0 / h)
            path = simulate_rates(TRUTH, [(0.0, 0.0, 0.0)] * n, h, initial_rate=w0)
            e0 = rotational_energy(TRUTH, w0)
            h0 = math.hypot(*angular_momentum(TRUTH, w0))
            de = max(abs(rotational_energy(TRUTH, w) - e0) for w in path) / e0
            dh = (
                max(abs(math.hypot(*angular_momentum(TRUTH, w)) - h0) for w in path)
                / h0
            )
            self.assertLess(de, 1e-5)
            self.assertLess(dh, 1e-5)
            drifts.append(max(de, dh))
        # Halving h should cut the drift by roughly 2^4 (between 2^3 and 2^5).
        self.assertGreater(drifts[0] / drifts[1], 8.0)
        self.assertLess(drifts[0] / drifts[1], 40.0)

    def test_constant_torque_small_rate_is_linear(self) -> None:
        """At small rates w' = J^-1 tau; the first step matches it."""
        tau = (1e-6, 0.0, 0.0)
        path = simulate_rates(TRUTH, [tau], 1.0)
        # J^-1 e_x from the Cholesky solve, compared through J w = tau dt.
        h = angular_momentum(TRUTH, path[1])
        for got, want in zip(h, (1e-6, 0.0, 0.0), strict=True):
            self.assertAlmostEqual(got, want, delta=1e-12)

    def test_substeps_refine(self) -> None:
        """More substeps change the path by far less than the data noise."""
        coarse = simulate_rates(TRUTH, manoeuvre(), DT)
        fine = simulate_rates(TRUTH, manoeuvre(), DT, substeps=4)
        gap = max(math.dist(a, b) for a, b in zip(coarse, fine, strict=True))
        self.assertLess(gap, 1e-7)


class ParameterizationTests(unittest.TestCase):
    """Cholesky parameterization, principal axes and realizability."""

    def test_round_trip(self) -> None:
        """Theta -> J -> theta recovers the parameters; J is symmetric PD."""
        theta = theta_from_inertia(TRUTH)
        again = inertia_from_theta(theta)
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(again[i][j], TRUTH[i][j], places=12)
        wild = inertia_from_theta((-3.0, 2.0, 0.5, 40.0, -7.0, 11.0))
        self.assertTrue(check_realizable(wild).positive_definite)

    def test_principal_axes_diagonal_and_rotated(self) -> None:
        """A diagonal tensor has the coordinate axes; a rotated one gets them."""
        diag = principal_axes(((3.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 2.0)))
        self.assertEqual(diag.moments, (1.0, 2.0, 3.0))
        self.assertEqual(diag.axis(0), (0.0, 1.0, 0.0))
        c = s = math.sqrt(0.5)  # 45-degree rotation about z of diag(1, 3, 5)
        rot = (
            (c * c + 3 * s * s, c * s - 3 * s * c, 0.0),
            (c * s - 3 * s * c, s * s + 3 * c * c, 0.0),
            (0.0, 0.0, 5.0),
        )
        axes = principal_axes(rot)
        for got, want in zip(axes.moments, (1.0, 3.0, 5.0), strict=True):
            self.assertAlmostEqual(got, want, places=10)
        self.assertAlmostEqual(abs(axes.axis(0)[0]), c, places=10)
        self.assertAlmostEqual(abs(axes.axis(0)[1]), c, places=10)

    def test_realizability_triangle_inequality(self) -> None:
        """diag(1, 1, 3) is positive definite but not a rigid body."""
        bad = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 3.0))
        report = check_realizable(bad)
        self.assertTrue(report.positive_definite)
        self.assertFalse(report.realizable)
        self.assertLess(min(report.triangle_margins), 0.0)
        # A thin rod-like boundary case I3 = I1 + I2 is realizable (margin 0).
        flat = check_realizable(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 2.0)))
        self.assertTrue(flat.realizable)
        self.assertTrue(check_realizable(TRUTH).realizable)
        indefinite = check_realizable(
            ((1.0, 2.0, 0.0), (2.0, 1.0, 0.0), (0.0, 0.0, 1.0))
        )
        self.assertFalse(indefinite.realizable)

    def test_entry_jacobian_matches_finite_differences(self) -> None:
        """The exact d(LL')/dtheta enters entry_covariance: compare with FD."""
        theta = theta_from_inertia(TRUTH)
        data = simulate_inertia_data(
            TRUTH, manoeuvre(steps=12), DT, noise_std=SIGMA, seed=0
        )
        problem = InertiaProblem(data)
        eye = tuple(tuple(float(i == j) for j in range(6)) for i in range(6))
        fake = InertiaLaplace(
            problem, theta, TRUTH, (0.0, 0.0, 0.0), eye, 0.0, 0, True, "synthetic"
        )
        jac, _ = central_jacobian(
            lambda t: inertia_entries(inertia_from_theta(t)), theta
        )
        cov = fake.entry_covariance()
        for i in range(6):
            for j in range(6):
                want = math.fsum(jac[i][k] * jac[j][k] for k in range(6))
                self.assertAlmostEqual(cov[i][j], want, delta=1e-6 * (1 + abs(want)))


class NoiseFreeRecoveryTests(unittest.TestCase):
    """Estimators on noise-free data."""

    def test_regression_error_is_second_order(self) -> None:
        """Regression error is second order.

        The regression has no noise bias at zero noise; its truncation error
        falls ~dt^2 (halving dt cuts the error by ~4).
        """
        for method in ("central", "integral"):
            errors = []
            for dt, steps, pulse in ((1.0, 240, 10), (0.5, 480, 20)):
                tq = excitation_torques(
                    "sequential", steps=steps, amplitude=0.2, pulse_steps=pulse
                )
                data = simulate_inertia_data(
                    TRUTH, tq, dt, noise_std=0.0, declared_noise_std=1e-3
                )
                est = regression_estimate(data, method=method)
                errors.append(max_abs_error(est.entries))
            self.assertLess(errors[0], 5e-3)
            self.assertGreater(errors[0] / errors[1], 2.5, msg=method)

    def test_regression_recovers_disturbance(self) -> None:
        """With a disturbance the 9-unknown regression is exact to O(dt^2)."""
        d = (0.01, -0.02, 0.015)
        tq = excitation_torques("sequential", steps=480, amplitude=0.2, pulse_steps=20)
        data = simulate_inertia_data(
            TRUTH, tq, 0.5, noise_std=0.0, declared_noise_std=1e-3, disturbance=d
        )
        est = regression_estimate(data, estimate_disturbance=True)
        for got, want in zip(est.disturbance, d, strict=True):
            self.assertAlmostEqual(got, want, delta=2e-4)
        self.assertLess(max_abs_error(est.entries), 5e-3)

    def test_levenberg_marquardt_recovers_truth(self) -> None:
        """Levenberg marquardt recovers truth.

        Noise-free output-error fit from a diagonal start (declared
        sigma 1e-6; the prior pull is ~(sigma/s)^2, negligible).
        """
        data = simulate_inertia_data(
            TRUTH, manoeuvre(), DT, noise_std=0.0, declared_noise_std=1e-6
        )
        mean, std = weak_prior(NOMINAL)
        problem = InertiaProblem(data, prior_mean=mean, prior_std=std)
        lap = inertia_laplace(problem, mean)
        self.assertTrue(lap.converged, lap.message)
        self.assertLess(max_abs_error(inertia_entries(lap.inertia_map)), 1e-5)

    def test_disturbance_recovered_by_output_error(self) -> None:
        """The nine-parameter fit recovers inertia and a constant torque."""
        d = (0.01, -0.02, 0.015)
        data = simulate_inertia_data(
            TRUTH,
            manoeuvre(),
            DT,
            noise_std=0.0,
            declared_noise_std=1e-6,
            disturbance=d,
        )
        mean, std = weak_prior(NOMINAL, disturbance_std=0.1)
        problem = InertiaProblem(
            data, estimate_disturbance=True, prior_mean=mean, prior_std=std
        )
        lap = inertia_laplace(problem, mean)
        self.assertTrue(lap.converged, lap.message)
        self.assertLess(max_abs_error(inertia_entries(lap.inertia_map)), 1e-4)
        for got, want in zip(lap.disturbance_map, d, strict=True):
            self.assertAlmostEqual(got, want, delta=1e-6)


class NoisyStudyTests(unittest.TestCase):
    """Seeded noisy studies: differencing bias and Laplace coverage."""

    fits: tuple[InertiaLaplace, ...]

    @classmethod
    def setUpClass(cls) -> None:
        """Fit the predeclared seeds once."""
        cls.fits = tuple(fit(seed) for seed in SEEDS)

    def test_fits_converge_with_covariance(self) -> None:
        """Every seeded fit converges and has a Laplace covariance."""
        for lap in self.fits:
            self.assertTrue(lap.converged, lap.message)
            self.assertIsNotNone(lap.covariance_theta)

    def test_regression_bias_versus_output_error(self) -> None:
        """Regression bias versus output error.

        Finite-difference regression is biased low (errors in variables);
        output-error LM is not, on the same seeds.
        """
        regression = {"central": [], "integral": []}
        for seed in SEEDS:
            data = simulate_inertia_data(
                TRUTH, manoeuvre(), DT, noise_std=SIGMA, seed=seed
            )
            for method in regression:
                regression[method].append(
                    regression_estimate(data, method=method).entries
                )
        # Diagonal entries, summed: trace is the most strongly biased quantity.
        trace_truth = math.fsum(TRUE_ENTRIES[:3])
        central = [math.fsum(e[:3]) - trace_truth for e in regression["central"]]
        integral = [math.fsum(e[:3]) - trace_truth for e in regression["integral"]]
        lm = [
            math.fsum(inertia_entries(lap.inertia_map)[:3]) - trace_truth
            for lap in self.fits
        ]
        n = len(SEEDS)
        se_central = statistics.stdev(central) / math.sqrt(n)
        # Measured (see report): central bias ~ -1.8, integral ~ -0.4, LM ~ 0.
        self.assertLess(statistics.fmean(central), -10.0 * se_central)
        self.assertLess(statistics.fmean(central), -1.0)
        self.assertLess(statistics.fmean(central), statistics.fmean(integral))
        self.assertLess(statistics.fmean(integral), 0.0)
        self.assertLess(abs(statistics.fmean(lm)), 0.15)
        self.assertLess(abs(statistics.fmean(lm)), 0.1 * abs(statistics.fmean(central)))

    def test_regression_bias_does_not_vanish_with_more_data(self) -> None:
        """Regression bias does not vanish with more data.

        Doubling the record leaves the central-difference bias unchanged
        (an inconsistent estimator, not a small-sample effect).
        """
        trace_truth = math.fsum(TRUE_ENTRIES[:3])
        means = []
        for steps in (240, 480):
            biases = []
            for seed in range(20):
                data = simulate_inertia_data(
                    TRUTH, manoeuvre(steps=steps), DT, noise_std=SIGMA, seed=seed
                )
                est = regression_estimate(data)
                biases.append(math.fsum(est.entries[:3]) - trace_truth)
            means.append(statistics.fmean(biases))
        self.assertLess(means[1], -1.0)
        self.assertLess(abs(means[1] - means[0]), 0.35 * abs(means[0]))

    def test_laplace_covers_truth_fast(self) -> None:
        """Laplace covers truth fast.

        Truth lies within 4 posterior sd of the MAP for every entry and
        seed; the standardized errors are not systematically large.
        """
        z_values = []
        for lap in self.fits:
            sd = lap.entry_std()
            for name, got, want in zip(
                sd, inertia_entries(lap.inertia_map), TRUE_ENTRIES, strict=True
            ):
                z = (got - want) / sd[name]
                z_values.append(z)
                self.assertLess(abs(z), 4.0, msg=name)
        # 8 seeds x 6 entries: rms z near 1 (correlated entries, weak prior).
        rms = math.sqrt(statistics.fmean(z * z for z in z_values))
        self.assertLess(rms, 1.8)

    def test_posterior_of_moments_and_axes(self) -> None:
        """Posterior of moments and axes.

        Sampled principal moments cover the truth; axis angles are small and
        no realizable draw is lost for this well-conditioned tensor.
        """
        truth = principal_axes(TRUTH)
        lap = self.fits[0]
        post = lap.sample_physical(400, seed=11)
        self.assertEqual(post.accepted + post.rejected, 400)
        self.assertEqual(post.rejected, 0)
        for i in range(3):
            lo, hi = post.moment_interval(i, 0.999)
            self.assertLessEqual(lo, truth.moments[i])
            self.assertGreaterEqual(hi, truth.moments[i])
        # Angle between MAP and true axes is below the sampled 99.9% bound
        # (0.5 degree order): the posterior axis scatter is small.
        mapped = principal_axes(lap.inertia_map)
        for i in range(3):
            cos = abs(
                math.fsum(
                    a * b for a, b in zip(mapped.axis(i), truth.axis(i), strict=True)
                )
            )
            angle = math.acos(min(1.0, cos))
            self.assertLess(angle, post.angle_interval(i, 0.999)[1] * 4.0)
            self.assertLess(post.angle_interval(i, 0.999)[1], math.radians(2.0))

    def test_realizability_rejection(self) -> None:
        """Realizability rejection.

        Near the triangle boundary a share of the Laplace draws is rejected,
        and every accepted draw satisfies the inequalities.
        """
        edge = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.98))
        data = simulate_inertia_data(
            edge, manoeuvre(steps=12), DT, noise_std=SIGMA, seed=0
        )
        theta = theta_from_inertia(edge)
        cov = tuple(
            tuple(0.05**2 if i == j else 0.0 for j in range(6)) for i in range(6)
        )
        lap = InertiaLaplace(
            InertiaProblem(data), theta, edge, (0.0, 0.0, 0.0), cov, 0.0, 0, True, "x"
        )
        post = lap.sample_physical(600, seed=3)
        self.assertGreater(post.rejection_rate, 0.1)
        self.assertLess(post.rejection_rate, 0.9)
        for m in post.moments:
            self.assertLessEqual(m[2], m[0] + m[1] + 1e-12)


class ExcitationDesignTests(unittest.TestCase):
    """One-axis versus multi-axis torques, by the posterior covariance."""

    def assess(self, kind: str, amplitude: float, sigma: float):
        """Return the assessment of one family."""
        mean, std = weak_prior(NOMINAL)
        tq = excitation_torques(kind, steps=STEPS, amplitude=amplitude, pulse_steps=10)
        return assess_excitation(
            kind, TRUTH, tq, DT, noise_std=sigma, prior_mean=mean, prior_std=std
        )

    def test_one_axis_small_rates_leaves_products_unidentified(self) -> None:
        """One axis small rates leaves products unidentified.

        At small rates (w' ~ J^-1 tau) torques about x identify only the x
        column of J^-1: the y-z block, with the product Jyz, is not learned.
        Measured: Jyz shrinkage 0.81 (one axis) versus 0.002 (three axes).
        """
        one = self.assess("single_axis", 0.002, 1e-5)
        seq = self.assess("sequential", 0.002, 1e-5)
        multi = self.assess("multi_axis", 0.002, 1e-5)
        names = ("Jxx", "Jyy", "Jzz", "Jxy", "Jxz", "Jyz")
        self.assertGreater(one.shrinkage[names.index("Jyz")], 0.5)
        self.assertGreater(one.shrinkage[names.index("Jyz")], 100 * seq.shrinkage[5])
        for k in (3, 4):  # Jxy, Jxz are partly seen through J^-1 e_x
            self.assertLess(one.shrinkage[k], 0.2)
        for other in (seq, multi):
            self.assertLess(max(other.shrinkage), 0.05)
            self.assertGreater(other.d_criterion, one.d_criterion + 20.0)
        # Three directions are essentially unlearned: whitened eigenvalues < 100.
        weak = [v for v in one.whitened_eigenvalues if v < 100.0]
        self.assertEqual(len(weak), 3)
        self.assertGreater(min(seq.whitened_eigenvalues), 1e3)

    def test_large_rates_gyroscopic_coupling_helps_one_axis(self) -> None:
        """Large rates gyroscopic coupling helps one axis.

        At 0.2 N m the gyroscopic term couples the axes and even one-axis
        torques identify every entry (shrinkage < 0.01).
        """
        one = self.assess("single_axis", 0.2, SIGMA)
        self.assertLess(max(one.shrinkage), 0.01)
        multi = self.assess("multi_axis", 0.2, SIGMA)
        # Simultaneous torques give the best D-criterion at equal peak torque.
        seq = self.assess("sequential", 0.2, SIGMA)
        self.assertGreater(multi.d_criterion, one.d_criterion)
        self.assertGreater(multi.d_criterion, seq.d_criterion)

    def test_zero_excitation_is_singular_without_prior(self) -> None:
        """Zero excitation is singular without prior.

        With zero rates nothing is learned: no prior gives no covariance
        (reported, not regularized).
        """
        data = simulate_inertia_data(
            TRUTH,
            [(0.0, 0.0, 0.0)] * 20,
            DT,
            noise_std=0.0,
            declared_noise_std=1e-3,
        )
        problem = InertiaProblem(data)
        lap = inertia_laplace(problem, theta_from_inertia(NOMINAL))
        self.assertIsNone(lap.covariance_theta)
        self.assertFalse(lap.converged)
        with self.assertRaises(ValueError):
            lap.entry_covariance()
        with self.assertRaises(ValueError):
            lap.sample_physical(10, seed=0)
        with self.assertRaises(ValueError):
            regression_estimate(data)

    def test_fisher_information_symmetric_psd(self) -> None:
        """The information matrix is symmetric with positive diagonal."""
        mean, std = weak_prior(NOMINAL)
        data = simulate_inertia_data(
            TRUTH, manoeuvre(steps=30), DT, noise_std=0.0, declared_noise_std=SIGMA
        )
        info = fisher_information(
            InertiaProblem(data, prior_mean=mean, prior_std=std),
            theta_from_inertia(TRUTH),
        )
        for i in range(6):
            self.assertGreater(info[i][i], 0.0)
            for j in range(6):
                self.assertAlmostEqual(
                    info[i][j], info[j][i], delta=1e-6 * (1 + abs(info[i][j]))
                )

    def test_excitation_shapes(self) -> None:
        """Pulse trains have the stated peak norm, axes and reversal."""
        for kind in EXCITATION_KINDS:
            tq = excitation_torques(kind, steps=80, amplitude=0.3, pulse_steps=10)
            self.assertEqual(len(tq), 80)
            self.assertAlmostEqual(max(math.hypot(*t) for t in tq), 0.3, places=12)
            self.assertEqual(tq[0], tq[9])
            self.assertEqual(tq[10], (0.0, 0.0, 0.0))
            self.assertEqual(tuple(-v for v in tq[0]), tq[20])
        single = excitation_torques("single_axis", steps=80, amplitude=1.0)
        self.assertTrue(all(t[1] == 0.0 and t[2] == 0.0 for t in single))


class FailureCaseTests(unittest.TestCase):
    """Invalid input is rejected loudly."""

    def test_invalid_inertia(self) -> None:
        """Non-symmetric, indefinite or wrongly shaped inertia."""
        zero3 = (0.0, 0.0, 0.0)
        with self.assertRaises(ValueError):
            simulate_rates(
                ((1.0, 0.5, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), [zero3], 1.0
            )
        with self.assertRaises(ValueError):
            simulate_rates(
                ((1.0, 2.0, 0.0), (2.0, 1.0, 0.0), (0.0, 0.0, 1.0)), [zero3], 1.0
            )
        with self.assertRaises(ValueError):
            simulate_rates(((1.0, 0.0), (0.0, 1.0)), [zero3], 1.0)
        with self.assertRaises(ValueError):
            principal_axes(((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)))
        with self.assertRaises(TypeError):
            rotational_energy(TRUTH, ("a", 0.0, 0.0))  # type: ignore[arg-type]

    def test_invalid_options(self) -> None:
        """Bad steps, substeps, kinds, windows and thetas."""
        tq = manoeuvre(steps=20)
        with self.assertRaises(ValueError):
            simulate_rates(TRUTH, tq, 0.0)
        with self.assertRaises(ValueError):
            simulate_rates(TRUTH, tq, 1.0, substeps=0)
        with self.assertRaises(ValueError):
            simulate_rates(TRUTH, [(0.0, math.nan, 0.0)], 1.0)
        with self.assertRaises(ValueError):
            excitation_torques("spiral", steps=10, amplitude=1.0)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            excitation_torques("sequential", steps=0, amplitude=1.0)
        with self.assertRaises(ValueError):
            inertia_from_theta((0.0, 0.0, 0.0))
        with self.assertRaises(ValueError):
            inertia_from_theta((1e4, 0.0, 0.0, 0.0, 0.0, 0.0))
        data = simulate_inertia_data(TRUTH, tq, DT, noise_std=SIGMA, seed=1)
        with self.assertRaises(ValueError):
            regression_estimate(data, method="integral", window=1)
        with self.assertRaises(ValueError):
            regression_estimate(data, method="spline")  # type: ignore[arg-type]
        problem = InertiaProblem(data)
        with self.assertRaises(ValueError):
            inertia_laplace(problem, (0.0,) * 9)
        with self.assertRaises(ValueError):
            inertia_laplace(problem, (0.0,) * 6, max_iterations=0)
        with self.assertRaises(ValueError):
            problem.residuals((0.0,) * 7)

    def test_invalid_data_and_prior(self) -> None:
        """Inconsistent data and priors."""
        tq = manoeuvre(steps=10)
        with self.assertRaises(ValueError):
            InertiaData(DT, tq, tq[:-1], (0.0, 0.0, 0.0), SIGMA)
        with self.assertRaises(ValueError):
            InertiaData(DT, tq, tq, (0.0, 0.0, 0.0), -1.0)
        with self.assertRaises(ValueError):
            InertiaData(DT, tq, tq, (0.0, 0.0), SIGMA)
        with self.assertRaises(ValueError):
            simulate_inertia_data(TRUTH, tq, DT, noise_std=0.0)
        data = simulate_inertia_data(TRUTH, tq, DT, noise_std=SIGMA, seed=1)
        with self.assertRaises(ValueError):
            InertiaProblem(data, prior_mean=(0.0,) * 6)
        with self.assertRaises(ValueError):
            InertiaProblem(data, prior_mean=(0.0,) * 5, prior_std=(1.0,) * 5)
        with self.assertRaises(ValueError):
            InertiaProblem(data, prior_mean=(0.0,) * 6, prior_std=(0.0,) * 6)
        with self.assertRaises(ValueError):
            weak_prior(NOMINAL, log_std=0.0)

    def test_noise_reproducible_and_seed_dependent(self) -> None:
        """Same seed, same data; different seed, different data."""
        tq = manoeuvre(steps=10)
        a = simulate_inertia_data(TRUTH, tq, DT, noise_std=SIGMA, seed=5)
        b = simulate_inertia_data(TRUTH, tq, DT, noise_std=SIGMA, seed=5)
        c = simulate_inertia_data(TRUTH, tq, DT, noise_std=SIGMA, seed=6)
        self.assertEqual(a.rates, b.rates)
        self.assertNotEqual(a.rates, c.rates)


@unittest.skipUnless(os.environ.get("PROBSTAT_SLOW") == "1", "slow tier")
class SlowCoverageStudy(unittest.TestCase):
    """Forty-seed coverage of the Laplace posterior and regression bias."""

    def test_entry_coverage_and_bias(self) -> None:
        """Entry coverage and bias.

        2-sd coverage of every entry within a binomial band; LM bias is
        within 3 standard errors of zero.
        """
        seeds = range(100, 140)
        fits = [fit(seed) for seed in seeds]
        n = len(fits)
        for k, name in enumerate(("Jxx", "Jyy", "Jzz", "Jxy", "Jxz", "Jyz")):
            z = []
            for lap in fits:
                sd = lap.entry_std()[name]
                z.append((inertia_entries(lap.inertia_map)[k] - TRUE_ENTRIES[k]) / sd)
            covered = sum(abs(v) < 2.0 for v in z) / n
            # Binomial(40, 0.954): mean 0.954, sd 0.033; 4 sd band is [0.82, 1].
            self.assertGreater(covered, 0.82, msg=name)
            self.assertLess(abs(statistics.fmean(z)), 3.0 / math.sqrt(n) * 1.0 + 0.3)


if __name__ == "__main__":
    unittest.main()
