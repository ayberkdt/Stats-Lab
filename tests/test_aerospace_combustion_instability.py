"""Tests for linear thermoacoustic stability with uncertainty.

Oracles: the flameless duct modes in closed form (``m pi c / L``, the
friction shift ``-i alpha/2 + sqrt(.)``, and the Marble-Candel decay
``-(c/L) atanh(zeta)``), the first-order perturbation of a closed-closed
duct derived by hand, the closed-form dispersion relation of the
closed-closed duct with a flame, and central finite differences of re-solved
roots.
"""

from __future__ import annotations

import cmath
import math
import os
import unittest
from dataclasses import replace

from probstat.aerospace.combustion_instability import (
    PARAMETER_NAMES,
    UNCERTAIN_NAMES,
    ChamberConfig,
    ThermoacousticConvergenceError,
    acoustic_mode_guess,
    dispersion_function,
    eigenvalue_sensitivities,
    instability_probability,
    mode_shape,
    rayleigh_index,
    solve_mode,
    thermoacoustic_model,
    thermoacoustic_modes,
)
from probstat.modeling import ProbabilisticModel

LENGTH = 1.0
SPEED = 400.0
THETA = 2.0


def chamber(**changes: object) -> ChamberConfig:
    """Return the reference closed-closed chamber with overrides."""
    base = ChamberConfig(
        LENGTH,
        SPEED,
        0.25,
        0.05,
        0.003,
        damping_rate=0.0,
        heat_release_ratio=THETA,
    )
    return replace(base, **changes)  # type: ignore[arg-type]


def first_order_shift(config: ChamberConfig, m: int) -> complex:
    """Hand-derived first-order shift of closed-closed mode ``m``.

    From ``sin kL + K cos(k(L-x_f)) sin(k x_f) = 0`` with
    ``K = Theta n exp(i omega tau)``: ``delta omega = -(c K / 2L) sin(2 k_m x_f)``.
    """
    omega_m = m * math.pi * config.sound_speed / config.length
    k_m = omega_m / config.sound_speed
    gain = config.heat_release_ratio * config.n
    coupling = gain * cmath.exp(1j * omega_m * config.tau)
    return (
        -config.sound_speed
        * coupling
        / (2.0 * config.length)
        * math.sin(2.0 * k_m * config.flame_position)
    )


class FlamelessDuctClosedForm(unittest.TestCase):
    """With n = 0 the flame is invisible and the modes are known exactly."""

    def test_closed_closed_modes(self) -> None:
        """Check that closed closed modes."""
        modes = thermoacoustic_modes(chamber(n=0.0), 4)
        for m, mode in enumerate(modes, start=1):
            self.assertTrue(mode.converged, mode.stop_reason)
            self.assertAlmostEqual(
                mode.frequency_hz, m * SPEED / (2.0 * LENGTH), delta=1e-8
            )
            self.assertAlmostEqual(mode.growth_rate, 0.0, delta=1e-8)

    def test_open_open_and_closed_open(self) -> None:
        """Check that open open and closed open."""
        open_open = thermoacoustic_modes(chamber(n=0.0, inlet="open", outlet="open"), 3)
        mixed = thermoacoustic_modes(chamber(n=0.0, outlet="open"), 3)
        for m in (1, 2, 3):
            self.assertAlmostEqual(
                open_open[m - 1].frequency_hz, m * SPEED / 2.0, delta=1e-8
            )
            self.assertAlmostEqual(
                mixed[m - 1].frequency_hz, (m - 0.5) * SPEED / 2.0, delta=1e-8
            )

    def test_friction_shift_is_exact(self) -> None:
        """Check that friction shift is exact."""
        alpha = 300.0
        for m in (1, 2):
            omega_m = m * math.pi * SPEED / LENGTH
            exact = -0.5j * alpha + math.sqrt(omega_m**2 - alpha**2 / 4.0)
            mode = solve_mode(chamber(n=0.0, damping_rate=alpha), mode=m)
            self.assertTrue(mode.converged)
            self.assertAlmostEqual(mode.omega.real, exact.real, delta=1e-7)
            self.assertAlmostEqual(mode.growth_rate, -alpha / 2.0, delta=1e-8)

    def test_choked_nozzle_decay_rate(self) -> None:
        """Check that choked nozzle decay rate."""
        config = chamber(n=0.0, outlet="choked", nozzle_mach=0.2, gamma=1.4)
        zeta = 0.5 * 0.4 * 0.2
        expected = -(SPEED / LENGTH) * math.atanh(zeta)
        self.assertAlmostEqual(config.nozzle_admittance, zeta)
        for mode in thermoacoustic_modes(config, 3):
            self.assertTrue(mode.converged)
            self.assertAlmostEqual(mode.growth_rate, expected, delta=1e-7)

    def test_zero_mach_choked_nozzle_is_a_rigid_wall(self) -> None:
        """Check that zero mach choked nozzle is a rigid wall."""
        mode = solve_mode(chamber(n=0.0, outlet="choked", nozzle_mach=0.0))
        self.assertAlmostEqual(mode.growth_rate, 0.0, delta=1e-8)


class DispersionRelation(unittest.TestCase):
    """The transfer-matrix function agrees with the hand derivation."""

    def test_closed_closed_flame_closed_form(self) -> None:
        """Check that closed closed flame closed form."""
        config = chamber()
        for omega in (900.0 + 30.0j, 1400.0 - 50.0j, 2600.0 + 5.0j):
            k = omega / SPEED
            coupling = THETA * config.n * cmath.exp(1j * omega * config.tau)
            xf = config.flame_position
            expected = (
                1j
                * (omega / SPEED)
                / k
                * (
                    cmath.sin(k * LENGTH)
                    + coupling * cmath.cos(k * (LENGTH - xf)) * cmath.sin(k * xf)
                )
            )
            value = dispersion_function(omega, config)
            self.assertLess(abs(value - expected), 1e-12 * max(1.0, abs(expected)))

    def test_series_and_closed_form_branches_agree(self) -> None:
        """Check that series and closed form branches agree."""
        # |q| = |omega x / c|^2 crosses 1 between these two frequencies.
        config = chamber()
        below = dispersion_function(1.5 * SPEED / 0.74, config)
        above = dispersion_function(1.5 * SPEED / 0.74 * (1.0 + 1e-9), config)
        self.assertLess(abs(below - above), 1e-6 * max(1.0, abs(below)))

    def test_mode_shape_satisfies_outlet_and_continuity(self) -> None:
        """Check that mode shape satisfies outlet and continuity."""
        config = chamber(outlet="choked", nozzle_mach=0.1, damping_rate=20.0)
        mode = solve_mode(config)
        (p_end, w_end), (p_in, w_in) = mode_shape(config, mode.omega, [LENGTH, 0.0])
        a1 = -config.nozzle_admittance
        self.assertLess(abs(a1 * p_end + w_end), 1e-9 * max(abs(p_end), abs(w_end)))
        self.assertLess(abs(w_in), 1e-12)  # closed inlet: u = 0
        xf = config.flame_position
        before, after = mode_shape(config, mode.omega, [xf - 1e-9, xf + 1e-9])
        self.assertLess(abs(before[0] - after[0]), 1e-6 * abs(before[0]))


class SmallGainPerturbation(unittest.TestCase):
    """For small n the root follows the hand-derived first-order shift."""

    def test_first_order_growth_and_frequency(self) -> None:
        """Check that first order growth and frequency."""
        for m, xf, tau in ((1, 0.25, 0.003), (2, 0.15, 0.0011), (3, 0.3, 0.0007)):
            config = chamber(n=1e-3, flame_position=xf, tau=tau)
            mode = solve_mode(config, mode=m)
            shift = first_order_shift(config, m)
            omega_m = m * math.pi * SPEED / LENGTH
            self.assertTrue(mode.converged)
            self.assertGreater(abs(shift.imag), 1e-3)
            # Second-order error is O(n) relative to the first-order shift.
            self.assertAlmostEqual(
                mode.growth_rate / shift.imag, 1.0, delta=2e-2, msg=f"mode {m}"
            )
            self.assertAlmostEqual(
                (mode.omega.real - omega_m) / shift.real, 1.0, delta=2e-2
            )

    def test_growth_is_linear_in_n(self) -> None:
        """Check that growth is linear in n."""
        small = solve_mode(chamber(n=1e-4)).growth_rate
        double = solve_mode(chamber(n=2e-4)).growth_rate
        self.assertAlmostEqual(double / small, 2.0, delta=1e-3)


class RayleighCriterion(unittest.TestCase):
    """Growth changes sign when the delay crosses half an oscillation period."""

    def test_sign_follows_sin_of_omega_tau(self) -> None:
        """Check that sign follows sin of omega tau."""
        omega_1 = math.pi * SPEED / LENGTH
        period = 2.0 * math.pi / omega_1
        # sin(2 k x_f) = 1 for x_f = L/4, so growth ~ -sin(omega tau).
        for fraction, sign in ((0.25, -1), (0.4, -1), (0.6, 1), (0.75, 1)):
            mode = solve_mode(chamber(n=1e-3, tau=fraction * period))
            self.assertGreater(sign * mode.growth_rate, 0.0, msg=str(fraction))

    def test_zero_crossing_at_half_period(self) -> None:
        """Check that zero crossing at half period."""
        omega_1 = math.pi * SPEED / LENGTH
        half = math.pi / omega_1

        def growth(tau: float) -> float:
            return solve_mode(chamber(n=1e-3, tau=tau)).growth_rate

        low, high = 0.8 * half, 1.2 * half
        self.assertLess(growth(low), 0.0)
        self.assertGreater(growth(high), 0.0)
        for _ in range(40):
            mid = 0.5 * (low + high)
            if growth(mid) < 0.0:
                low = mid
            else:
                high = mid
        # Second-order terms move the crossing by O(n) relative.
        self.assertAlmostEqual(0.5 * (low + high) / half, 1.0, delta=5e-3)

    def test_rayleigh_index_sign_and_energy_balance(self) -> None:
        """Check that rayleigh index sign and energy balance."""
        omega_1 = math.pi * SPEED / LENGTH
        period = 2.0 * math.pi / omega_1
        for fraction in (0.25, 0.75):
            config = chamber(n=0.05, tau=fraction * period)
            mode = solve_mode(config)
            index = rayleigh_index(config, mode.omega)
            self.assertEqual(index.flame_growth_rate > 0.0, mode.growth_rate > 0.0)
            self.assertEqual(index.phase_cosine > 0.0, mode.growth_rate > 0.0)
            # No friction and rigid ends: the flame is the only source.
            self.assertAlmostEqual(index.boundary_growth_rate, 0.0, delta=1e-8)
            self.assertAlmostEqual(
                index.balance_growth_rate, mode.growth_rate, delta=1e-6
            )

    def test_energy_balance_with_friction_and_nozzle(self) -> None:
        """Check that energy balance with friction and nozzle."""
        config = chamber(outlet="choked", nozzle_mach=0.1, damping_rate=20.0)
        mode = solve_mode(config)
        index = rayleigh_index(config, mode.omega)
        self.assertLess(index.boundary_growth_rate, 0.0)
        self.assertLess(index.dissipation_growth_rate, 0.0)
        self.assertAlmostEqual(index.balance_error, 0.0, delta=1e-6)

    def test_off_root_frequency_leaves_an_outlet_flux(self) -> None:
        """Check that a non-root frequency violates the outlet condition."""
        config = chamber()
        ((_, w_end),) = mode_shape(config, 1000.0 + 0.0j, [LENGTH])
        self.assertGreater(abs(w_end), 1e-3)  # closed outlet needs w = 0


class EigenvalueSensitivityTests(unittest.TestCase):
    """Implicit differentiation against closed forms and finite differences."""

    def test_closed_form_at_zero_gain(self) -> None:
        """Check that closed form at zero gain."""
        config = chamber(n=0.0, tau=0.003)
        mode = solve_mode(config)
        sens = eigenvalue_sensitivities(config, mode.omega)
        omega = mode.omega
        expected_n = (
            -SPEED
            * THETA
            / (2.0 * LENGTH)
            * math.sin(2.0 * omega.real / SPEED * 0.25)
            * cmath.exp(1j * omega * 0.003)
        )
        self.assertLess(abs(sens.d_omega["n"] - expected_n), 1e-9 * abs(expected_n))
        self.assertLess(abs(sens.d_omega["damping_rate"] - (-0.5j)), 1e-12)
        self.assertLess(abs(sens.d_omega["sound_speed"] - omega / SPEED), 1e-9)
        self.assertLess(abs(sens.d_omega["length"] - (-omega / LENGTH)), 1e-9)
        self.assertLess(abs(sens.d_omega["tau"]), 1e-9)  # n = 0: tau is inert

    def test_matches_central_finite_differences(self) -> None:
        """Check that matches central finite differences."""
        for config in (
            chamber(),
            chamber(outlet="choked", nozzle_mach=0.1, damping_rate=15.0, n=0.08),
            chamber(inlet="open", outlet="open", flame_position=0.4, n=0.03),
        ):
            mode = solve_mode(config)
            self.assertTrue(mode.converged)
            sens = eigenvalue_sensitivities(config, mode.omega)
            self.assertLess(sens.residual, 1e-9)
            steps = {
                "n": 1e-6,
                "tau": 1e-8,
                "damping_rate": 1e-3,
                "sound_speed": 1e-4,
                "flame_position": 1e-6,
                "heat_release_ratio": 1e-6,
                "length": 1e-6,
            }
            self.assertEqual(set(steps), set(PARAMETER_NAMES))
            for name, step in steps.items():
                value = getattr(config, name)
                lower = value - step
                if name in ("tau", "damping_rate"):
                    lower = max(lower, 0.0)
                plus = solve_mode(replace(config, **{name: value + step}), mode.omega)
                minus = solve_mode(replace(config, **{name: lower}), mode.omega)
                self.assertTrue(plus.converged and minus.converged)
                numeric = (plus.omega - minus.omega) / (value + step - lower)
                exact = sens.d_omega[name]
                self.assertLess(
                    abs(numeric - exact),
                    1e-5 * max(abs(exact), 1e-3),
                    msg=f"{name}: {numeric} vs {exact}",
                )
            self.assertAlmostEqual(
                sens.d_growth_rate["n"], sens.d_omega["n"].imag, places=12
            )
            self.assertAlmostEqual(
                sens.d_frequency_hz["n"],
                sens.d_omega["n"].real / (2.0 * math.pi),
                places=12,
            )

    def test_sensitivity_sign_predicts_stability_change(self) -> None:
        """Check that sensitivity sign predicts stability change."""
        config = chamber(tau=0.75 * 2.0 * LENGTH / SPEED)  # destabilizing delay
        mode = solve_mode(config)
        sens = eigenvalue_sensitivities(config, mode.omega)
        self.assertGreater(mode.growth_rate, 0.0)
        self.assertGreater(sens.d_growth_rate["n"], 0.0)
        self.assertLess(sens.d_growth_rate["damping_rate"], 0.0)

    def test_degenerate_root_is_rejected(self) -> None:
        """Check that degenerate root is rejected."""
        # At omega = 0 the closed-closed dispersion function has F = F' ... a
        # trivial root of multiplicity one in F/omega; use a non-finite input.
        with self.assertRaises(ValueError):
            eigenvalue_sensitivities(chamber(), complex("nan"))


class NewtonFailureIsReported(unittest.TestCase):
    """Failures show up in the result, not in a silent fallback."""

    def test_iteration_cap(self) -> None:
        """Check that iteration cap."""
        mode = solve_mode(chamber(), 1100.0 + 0.0j, max_iterations=1)
        self.assertFalse(mode.converged)
        self.assertEqual(mode.stop_reason, "maximum iterations reached")
        self.assertEqual(mode.iterations, 1)
        self.assertFalse(mode.unstable)

    def test_overflow_is_reported(self) -> None:
        """Check that overflow is reported."""
        mode = solve_mode(chamber(), 1e7j)
        self.assertFalse(mode.converged)
        self.assertEqual(mode.stop_reason, "overflow")

    def test_converged_start_far_from_acoustic_mode(self) -> None:
        """Check that converged start far from acoustic mode."""
        # A different starting point may land on another root; it must still
        # be a root of F (small residual) when it reports convergence.
        mode = solve_mode(chamber(), 2500.0 + 0.0j)
        self.assertTrue(mode.converged)
        self.assertLess(mode.residual, 1e-8)

    def test_invalid_arguments(self) -> None:
        """Check that invalid arguments."""
        with self.assertRaises(TypeError):
            solve_mode("chamber")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            solve_mode(chamber(), tolerance=0.0)
        with self.assertRaises(ValueError):
            solve_mode(chamber(), max_iterations=0)
        with self.assertRaises(ValueError):
            solve_mode(chamber(), complex("inf"))
        with self.assertRaises(TypeError):
            solve_mode(chamber(), "1000")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            acoustic_mode_guess(chamber(), 0)
        with self.assertRaises(TypeError):
            thermoacoustic_modes(chamber(), 1.5)  # type: ignore[arg-type]

    def test_config_validation(self) -> None:
        """Check that config validation."""
        for bad in (
            {"length": 0.0},
            {"sound_speed": -1.0},
            {"flame_position": 1.0},
            {"flame_position": 0.0},
            {"tau": -1e-3},
            {"damping_rate": -1.0},
            {"gamma": 1.0},
            {"nozzle_mach": 1.0},
            {"inlet": "choked"},
            {"outlet": "wall"},
            {"n": float("nan")},
        ):
            with self.assertRaises(ValueError, msg=str(bad)):
                chamber(**bad)
        with self.assertRaises(TypeError):
            chamber(n="0.1")


class ProbabilisticModelTests(unittest.TestCase):
    """The factory output plugs into the library's UQ tools."""

    def test_model_matches_direct_solve(self) -> None:
        """Check that model matches direct solve."""
        config = chamber(damping_rate=20.0)
        model = thermoacoustic_model(config)
        self.assertEqual(
            model.inputs.names,
            ("n", "tau", "damping_rate", "sound_speed", "flame_position"),
        )
        out = model.evaluate(model.inputs.means())
        mode = solve_mode(config)
        self.assertAlmostEqual(out["growth_rate"], mode.growth_rate, places=9)
        self.assertAlmostEqual(out["frequency_hz"], mode.frequency_hz, places=9)
        self.assertEqual(model.evaluations, 1)

    def test_fixed_parameters_and_validation(self) -> None:
        """Check that fixed parameters and validation."""
        model = thermoacoustic_model(
            chamber(damping_rate=20.0), {"tau": 0.0, "sound_speed": 0.0}
        )
        self.assertEqual(model.inputs.names, ("n", "damping_rate", "flame_position"))
        with self.assertRaises(ValueError):
            thermoacoustic_model(chamber(), {"gamma": 0.1})
        with self.assertRaises(ValueError):
            thermoacoustic_model(chamber(), {"flame_position": 0.2})  # < 6 sd margin
        with self.assertRaises(ValueError):
            thermoacoustic_model(chamber(damping_rate=0.0), {"damping_rate": 0.1})
        with self.assertRaises(ValueError):
            thermoacoustic_model(
                chamber(),
                {k: 0.0 for k in UNCERTAIN_NAMES},
            )
        with self.assertRaises(ValueError):
            thermoacoustic_model(chamber(), {"n": -0.1})

    def test_non_convergence_is_raised_not_dropped(self) -> None:
        """Check that non convergence is raised not dropped."""
        model = thermoacoustic_model(chamber(), max_iterations=1, tolerance=1e-14)
        with self.assertRaises(ThermoacousticConvergenceError):
            model.evaluate(model.inputs.means())

    def test_mode_jump_is_raised(self) -> None:
        """Check that mode jump is raised."""
        model = thermoacoustic_model(chamber(), max_mode_shift=1e-9)
        with self.assertRaises(ThermoacousticConvergenceError):
            model.evaluate(model.inputs.means())


def reference_model() -> ProbabilisticModel:
    """Chamber whose growth rate is mildly stable on average."""
    config = chamber(damping_rate=20.0, outlet="choked", nozzle_mach=0.1)
    return thermoacoustic_model(config)


class InstabilityProbabilityTests(unittest.TestCase):
    """Monte Carlo, PCE surrogate and FORM on ``growth_rate > 0``."""

    def test_three_routes_agree(self) -> None:
        """Check that three routes agree."""
        model = reference_model()
        result = instability_probability(
            model,
            mc_samples=400,
            pce_degree=3,
            surrogate_samples=4000,
            seed=11,
        )
        p = result.monte_carlo_probability
        se = math.sqrt(p * (1.0 - p) / 400)
        self.assertGreater(p, 0.02)
        self.assertLess(p, 0.6)
        self.assertLess(result.surrogate_error, 1e-2)
        # Surrogate error is small, so the gap is Monte Carlo noise (two
        # independent samples): 4 combined standard errors plus 0.01.
        gap = abs(result.surrogate_probability - p)
        self.assertLess(gap, 4.0 * se * math.sqrt(2) + 0.01)
        self.assertTrue(result.form_converged)
        self.assertIsNotNone(result.form_probability)
        # FORM linearizes the boundary; it is a coarser bound, not an exact value.
        assert result.form_probability is not None
        self.assertLess(abs(result.form_probability - p), 0.1)
        self.assertEqual(
            result.form_beta is not None and result.form_beta > 0.0, p < 0.5
        )
        self.assertLess(result.monte_carlo_interval[0], p)
        self.assertGreater(result.monte_carlo_interval[1], p)
        self.assertGreater(result.std_growth_rate, 0.0)

    def test_unstable_mean_gives_probability_above_half(self) -> None:
        """Check that unstable mean gives probability above half."""
        config = chamber(damping_rate=0.0, n=0.05)
        spread = {"damping_rate": 0.0, "n": 0.05, "tau": 0.02}
        model = thermoacoustic_model(config, spread)
        result = instability_probability(
            model, mc_samples=150, surrogate_samples=1500, pce_degree=2, seed=3
        )
        self.assertGreater(result.mean_growth_rate, 0.0)
        self.assertGreater(result.monte_carlo_probability, 0.5)
        assert result.form_beta is not None
        self.assertLess(result.form_beta, 0.0)

    def test_argument_validation(self) -> None:
        """Check that argument validation."""
        with self.assertRaises(TypeError):
            instability_probability("model")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            instability_probability(reference_model(), mc_samples=0)


@unittest.skipUnless(os.environ.get("PROBSTAT_SLOW") == "1", "slow tier")
class InstabilityProbabilitySlow(unittest.TestCase):
    """A larger comparison; prints the numbers it measures."""

    def test_large_sample_comparison(self) -> None:
        """Check that large sample comparison."""
        result = instability_probability(
            reference_model(),
            mc_samples=10000,
            pce_degree=4,
            surrogate_samples=100000,
            seed=5,
        )
        print(result)
        self.assertTrue(result.surrogate_within_interval)


if __name__ == "__main__":
    unittest.main()
