"""Tests for ``probstat.aerospace.entry_reconstruction``.

Verification order: the synthetic truth against the independent integrator of
``probstat.aerospace.entry`` and against the exact ``rho / beta`` degeneracy of
the equations of motion, then the noise-free recovery of the density
correction (batch method to discretization error, filter to a looser bound),
then agreement between the two methods and between EKF and UKF, then the
identifiability study (accelerometer only, with radar, with a tight ballistic
prior, with a density anchor), then failure cases (a data gap, altitudes above
the flight, missing data, invalid input) and finally the coverage of the
reconstructed bands over predeclared seeds. All numbers are synthetic.

Tolerances are justified next to each assertion. The 40-seed coverage study
runs only with ``PROBSTAT_SLOW=1``; a 3-seed version always runs.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import os
import time
import unittest
from typing import Literal

from probstat.aerospace.entry import EntryVehicle, ExponentialAtmosphere, simulate_entry
from probstat.aerospace.entry_reconstruction import (
    BatchReconstruction,
    EntryMeasurements,
    EntryReconstructionSetup,
    EntryTruth,
    FilterReconstruction,
    IdentifiabilityReport,
    band_coverage,
    reconstruct_entry_batch,
    reconstruct_entry_filter,
    simulate_entry_measurements,
    synthesize_entry_truth,
)

SETUP = EntryReconstructionSetup()
HIGH_SNR = (30_000.0, 85_000.0)  # altitudes where the sensed acceleration >> noise
COVERAGE_WINDOW = (25_000.0, 90_000.0)
PREDECLARED_SEEDS = (1, 2, 3)  # fast tier; truth seed s, noise seed 100 + s
SLOW_SEEDS = tuple(range(1, 41))


@functools.cache
def case(
    setup: EntryReconstructionSetup,
    truth_seed: int,
    noise_seed: int,
    dropout: tuple[float, float] | None = None,
) -> tuple[EntryTruth, EntryMeasurements]:
    """Return a cached ``(truth, data)`` pair (offset 0, noisy).

    Args:
        setup: The scenario.
        truth_seed: Truth seed.
        noise_seed: Noise seed.
        dropout: Optional altitude band without data.

    Returns:
        The truth and its measurements.

    References:
        [EQ-AERO-RECON-001]. No equation is evaluated.

    """
    truth = synthesize_entry_truth(setup, seed=truth_seed)
    data = simulate_entry_measurements(
        setup, truth, seed=noise_seed, dropout_band=dropout
    )
    return truth, data


@functools.cache
def batch_fit(
    setup: EntryReconstructionSetup,
    truth_seed: int,
    noise_seed: int,
    dropout: tuple[float, float] | None = None,
) -> BatchReconstruction:
    """Return a cached batch fit of :func:`case`.

    Args:
        setup: The scenario.
        truth_seed: Truth seed.
        noise_seed: Noise seed.
        dropout: Optional altitude band without data.

    Returns:
        The batch reconstruction.

    References:
        [EQ-AERO-RECON-004]. No equation is evaluated.

    """
    return reconstruct_entry_batch(
        setup, case(setup, truth_seed, noise_seed, dropout)[1]
    )


@functools.cache
def filter_fit(
    setup: EntryReconstructionSetup,
    truth_seed: int,
    noise_seed: int,
    dropout: tuple[float, float] | None = None,
) -> FilterReconstruction:
    """Return a cached EKF fit of :func:`case`.

    Args:
        setup: The scenario.
        truth_seed: Truth seed.
        noise_seed: Noise seed.
        dropout: Optional altitude band without data.

    Returns:
        The filter reconstruction.

    References:
        [EQ-AERO-RECON-003]. No equation is evaluated.

    """
    return reconstruct_entry_filter(
        setup, case(setup, truth_seed, noise_seed, dropout)[1]
    )


def peak_altitude(truth: EntryTruth) -> float:
    """Return the altitude of the peak true deceleration.

    Args:
        truth: An ``EntryTruth``.

    Returns:
        Altitude in m.

    References:
        [EQ-AERO-RECON-002]. No equation is evaluated.

    """
    decel = truth.deceleration
    return truth.altitude[decel.index(max(decel))]


def window_indices(
    altitudes: tuple[float, ...], window: tuple[float, float]
) -> list[int]:
    """Return indices ``k >= 1`` whose altitude lies inside a window.

    Args:
        altitudes: Altitudes in m, index ``k`` is the sample ``k``.
        window: ``(lowest, highest)`` in m.

    Returns:
        The indices.

    References:
        [EQ-AERO-RECON-006]. No equation is evaluated.

    """
    return [
        k for k in range(1, len(altitudes)) if window[0] <= altitudes[k] <= window[1]
    ]


class TruthTests(unittest.TestCase):
    """The synthetic truth against independent oracles."""

    def test_matches_independent_integrator_without_perturbation(self) -> None:
        """With ``c ~ 0`` the truth equals ``simulate_entry`` (RK4, 1 s step).

        Both integrate the same equations with RK4; the remaining difference is
        the step size (1 s against 0.125 s), whose error for these smooth
        solutions is far below the 1 m and 0.05 m/s bounds.
        """
        truth = synthesize_entry_truth(SETUP, seed=1, amplitude=1e-12)
        result = simulate_entry(
            EntryVehicle(SETUP.ballistic_coefficient, 1.0),
            altitude=SETUP.entry_altitude,
            speed=SETUP.entry_speed,
            flight_path_angle=SETUP.flight_path_angle,
            atmosphere=ExponentialAtmosphere(),
            terminal_altitude=25_000.0,
            max_step=1.0,
            record=True,
        )
        self.assertEqual(result.status, "landed")
        grid = result.trajectory
        self.assertGreater(len(grid), 100)
        for time_s, h, v, gamma, _theta in grid:
            k = round(time_s)
            self.assertAlmostEqual(truth.altitude[k], h, delta=1.0)
            self.assertAlmostEqual(truth.speed[k], v, delta=0.05)
            self.assertAlmostEqual(truth.flight_path_angle[k], gamma, delta=1e-6)

    def test_sensed_deceleration_is_drag_formula(self) -> None:
        """``a = rho V^2 / (2 beta)`` with the corrected density, exactly."""
        truth = synthesize_entry_truth(SETUP, seed=5, ballistic_offset=0.2)
        beta = math.exp(truth.log_ballistic_coefficient)
        atmosphere = ExponentialAtmosphere()
        for k in (0, 40, 90, 150):
            h, v = truth.altitude[k], truth.speed[k]
            rho = atmosphere.density(h) * math.exp(truth.log_density_correction(h))
            self.assertAlmostEqual(
                truth.deceleration[k], rho * v * v / (2.0 * beta), delta=1e-9 * 200
            )

    def test_kinematics_depend_only_on_density_over_ballistic_ratio(self) -> None:
        """Scaling the base density and ``beta`` together leaves the flight unchanged.

        ``h' , V' , gamma'`` contain ``rho / beta`` only ([EQ-AERO-RECON-002]),
        so a base atmosphere scaled by ``e^d`` with ``ln beta`` shifted by ``d``
        must reproduce the trajectory to rounding error.
        """
        delta = 0.7
        scaled = dataclasses.replace(
            SETUP,
            base_atmosphere=ExponentialAtmosphere(1.225 * math.exp(delta), 7200.0),
        )
        a = synthesize_entry_truth(SETUP, seed=7, ballistic_offset=0.0)
        b = synthesize_entry_truth(scaled, seed=7, ballistic_offset=delta)
        for k in (50, 100, 150):
            self.assertAlmostEqual(a.altitude[k], b.altitude[k], delta=1e-6)
            self.assertAlmostEqual(a.speed[k], b.speed[k], delta=1e-8)
        # the sensed deceleration is rho V^2 / (2 beta): a function of rho / beta
        self.assertAlmostEqual(a.deceleration[100], b.deceleration[100], delta=1e-8)

    def test_seeded_reproducibility_and_noise_independence(self) -> None:
        """Same seeds give identical data; different noise seeds differ."""
        truth = synthesize_entry_truth(SETUP, seed=3)
        again = synthesize_entry_truth(SETUP, seed=3)
        self.assertEqual(truth, again)
        first = simulate_entry_measurements(SETUP, truth, seed=9)
        self.assertEqual(first, simulate_entry_measurements(SETUP, truth, seed=9))
        self.assertNotEqual(first, simulate_entry_measurements(SETUP, truth, seed=10))

    def test_invalid_input_and_ground_impact(self) -> None:
        """Bad arguments raise; a flight that hits the ground is rejected."""
        with self.assertRaises(TypeError):
            synthesize_entry_truth("setup", seed=1)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            dataclasses.replace(SETUP, flight_path_angle=0.1)
        with self.assertRaises(ValueError):
            dataclasses.replace(SETUP, nodes=2)
        with self.assertRaises(TypeError):
            dataclasses.replace(SETUP, nodes=True)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            dataclasses.replace(SETUP, anchor_altitudes=(200_000.0,))
        with self.assertRaises(TypeError):
            dataclasses.replace(SETUP, base_atmosphere="mars")  # type: ignore[arg-type]
        heavy = dataclasses.replace(SETUP, ballistic_coefficient=1e6, duration=200.0)
        with self.assertRaises(ValueError):
            synthesize_entry_truth(heavy, seed=1)

    def test_measurement_data_must_match_setup(self) -> None:
        """Mismatched record lengths and missing data are rejected."""
        truth = synthesize_entry_truth(SETUP, seed=1)
        data = simulate_entry_measurements(SETUP, truth, seed=2)
        short = dataclasses.replace(data, acceleration=data.acceleration[:-1])
        with self.assertRaises(ValueError):
            reconstruct_entry_batch(SETUP, short)
        with self.assertRaises(ValueError):
            reconstruct_entry_filter(SETUP, short)
        with self.assertRaises(ValueError):
            reconstruct_entry_filter(SETUP, data, method="pf")  # type: ignore[arg-type]
        empty = dataclasses.replace(data, acceleration=(None,) * SETUP.samples)
        with self.assertRaises(ValueError):
            reconstruct_entry_batch(SETUP, empty)
        with self.assertRaises(ValueError):
            simulate_entry_measurements(SETUP, truth, seed=1, dropout_band=(5.0, 1.0))


class NoiseFreeRecoveryTests(unittest.TestCase):
    """Noise-free data recover the truth to discretization error."""

    def setUp(self) -> None:
        """Build a near-noise-free scenario with a tight ballistic prior."""
        self.setup = dataclasses.replace(SETUP, accel_sigma=0.01, ballistic_log_sd=0.05)
        self.truth = synthesize_entry_truth(self.setup, seed=3)
        self.data = simulate_entry_measurements(
            self.setup, self.truth, seed=0, noise=False
        )

    def test_batch_recovers_truth_to_spline_error(self) -> None:
        """Batch error below 0.01 where the signal exceeds the noise.

        Measured: 0.0034 over 20 to 90 km. The bound 0.01 is three times that
        and well below the prior standard deviation 0.3: the error is the
        spline-interpolation error (spacing 6.6 km, length scale 10 km, sd
        0.3; the bound ``0.3 (6.6/10)^4 5/384`` is about 0.0007 for a pure
        sinusoid of that wavelength, and the truth contains shorter ones) plus
        the O(h^4) RK4 difference of the two step sizes.
        """
        fit = reconstruct_entry_batch(self.setup, self.data)
        self.assertTrue(fit.converged, fit.stop_reason)
        altitudes = [h for h in self.truth.altitude[1:] if 20_000.0 <= h <= 90_000.0]
        means, _ = fit.log_density_correction(altitudes)
        errors = [
            abs(m - self.truth.log_density_correction(h))
            for m, h in zip(means, altitudes, strict=True)
        ]
        self.assertLess(max(errors), 0.01)
        self.assertLess(abs(fit.log_ballistic_offset), 0.01)  # truth offset is 0

    def test_filter_recovers_truth_in_high_snr_band(self) -> None:
        """Filter error below 0.05 for 30 to 85 km (EKF and UKF).

        The filter prior is rougher than the smooth truth and the EKF is a
        first-order method, so its error is larger than the batch one; the
        looser bound 0.05 is about one sixth of the prior standard deviation.
        """
        methods: tuple[Literal["ekf", "ukf"], ...] = ("ekf", "ukf")
        for method in methods:
            fit = reconstruct_entry_filter(self.setup, self.data, method=method)
            for k in window_indices(fit.altitude, HIGH_SNR):
                error = abs(
                    fit.log_density_mean[k]
                    - self.truth.log_density_correction(fit.altitude[k])
                )
                self.assertLess(error, 0.05, f"{method} k={k}")

    def test_density_band_is_ordered(self) -> None:
        """The density band is positive and ``lower < median < upper``."""
        fit = reconstruct_entry_filter(self.setup, self.data)
        lower, mid, upper = fit.density_band()
        self.assertTrue(
            all(0.0 < a < b < c for a, b, c in zip(lower, mid, upper, strict=True))
        )
        with self.assertRaises(ValueError):
            fit.density_band(0.0)


class MethodAgreementTests(unittest.TestCase):
    """Filter and batch methods agree where the data are informative."""

    @classmethod
    def setUpClass(cls) -> None:
        """Run both methods once on one noisy data set."""
        cls.truth = synthesize_entry_truth(SETUP, seed=3, ballistic_offset=None)
        cls.data = simulate_entry_measurements(SETUP, cls.truth, seed=103)
        cls.batch = reconstruct_entry_batch(SETUP, cls.data)
        cls.ekf = reconstruct_entry_filter(SETUP, cls.data, method="ekf")
        cls.ukf = reconstruct_entry_filter(SETUP, cls.data, method="ukf")

    def test_batch_converged_and_fits_noise(self) -> None:
        """The batch fit converges with a whitened residual rms of about one.

        For a correct model the whitened residual rms over ``m = 150`` data has
        mean about ``sqrt((m - p_eff) / m)`` below one; the band (0.8, 1.2) is
        about five standard errors (``1 / sqrt(2 m)`` = 0.06) wide.
        """
        self.assertTrue(self.batch.converged, self.batch.stop_reason)
        self.assertGreater(self.batch.whitened_residual_rms, 0.8)
        self.assertLess(self.batch.whitened_residual_rms, 1.2)

    def test_methods_agree_on_the_ratio(self) -> None:
        """``c - b`` (what the kinematics determine) agrees within 0.15.

        Measured over this data set the maximum difference in the high-SNR
        band is below 0.1; both estimates carry posterior sds above 0.01 there
        and the filter prior is an exponential-kernel *process* while the
        batch prior is its spline discretization, so exact agreement is not
        expected.
        """
        fit = self.ekf
        indices = window_indices(fit.altitude, HIGH_SNR)
        altitudes = [fit.altitude[k] for k in indices]
        means, _ = self.batch.log_density_correction(altitudes)
        worst = max(
            abs(
                (fit.log_density_mean[k] - fit.log_ballistic_offset_mean)
                - (m - self.batch.log_ballistic_offset)
            )
            for k, m in zip(indices, means, strict=True)
        )
        self.assertLess(worst, 0.15)

    def test_ekf_and_ukf_agree_in_high_snr_band(self) -> None:
        """EKF and UKF means differ by less than 0.05 in the high-SNR band."""
        for k in window_indices(self.ekf.altitude, HIGH_SNR):
            self.assertAlmostEqual(
                self.ekf.log_density_mean[k], self.ukf.log_density_mean[k], delta=0.05
            )

    def test_filter_innovations_are_consistent_or_conservative(self) -> None:
        """Mean NIS per dimension lies in (0.05, 1.5): not over-confident.

        The filter prior is rougher than the truth, so a mean NIS below one
        (conservative) is expected; a value above 1.5 would signal an
        over-confident or biased filter.
        """
        for fit in (self.ekf, self.ukf):
            self.assertGreater(fit.mean_nis, 0.05)
            self.assertLess(fit.mean_nis, 1.5)

    def test_filter_ratio_beats_prior_in_high_snr_band(self) -> None:
        """The smoothed ratio sd is below one fifth of the prior sd (35-70 km)."""
        for k in window_indices(self.ekf.altitude, (35_000.0, 70_000.0)):
            self.assertLess(self.ekf.log_ratio_sd[k], 0.2 * SETUP.density_log_sd)


class IdentifiabilityTests(unittest.TestCase):
    """The split of density and ballistic coefficient is not identified."""

    @staticmethod
    def report(
        setup: EntryReconstructionSetup,
    ) -> tuple[IdentifiabilityReport, BatchReconstruction]:
        """Fit one data set and return the report at the peak-deceleration altitude.

        Args:
            setup: The scenario.

        Returns:
            ``(report, fit)``.

        References:
            [EQ-AERO-RECON-007]. No equation is evaluated.

        """
        truth, _ = case(setup, 3, 4)
        fit = batch_fit(setup, 3, 4)
        return fit.identifiability(peak_altitude(truth)), fit

    def test_split_is_not_identified_by_accelerometer_or_radar(self) -> None:
        """Ratio sd is tiny while the individual sds stay near their priors.

        With ``L/D = 0`` the data depend on ``c - b`` only, so the posterior
        correlation of ``ln beta`` and ``c`` must be essentially one and the
        ratio sd a small fraction of either. Radar (altitude and speed fixes)
        sharpens the ratio but cannot reduce the individual sds by more than
        a few percent (measured: 0.131 -> 0.125).
        """
        acc, fit = self.report(SETUP)
        radar, _ = self.report(dataclasses.replace(SETUP, radar_every=5))
        self.assertTrue(fit.converged)
        self.assertGreater(acc.correlation, 0.99)
        self.assertLess(acc.ratio_sd, 0.05 * acc.log_density_sd)
        # the individual sds are far above the ratio sd: about 0.13 of a prior 0.3
        self.assertGreater(acc.log_ballistic_sd, 0.1)
        self.assertGreater(radar.correlation, 0.99)
        self.assertLess(radar.ratio_sd, 0.5 * acc.ratio_sd)
        self.assertGreater(radar.log_ballistic_sd, 0.8 * acc.log_ballistic_sd)

    def test_prior_on_beta_and_density_anchor_break_the_split(self) -> None:
        """A tight ``beta`` prior or an in-situ density anchor shrinks both sds.

        Measured (accelerometer only): sd of ``ln beta`` 0.131; with prior sd
        0.02 it is 0.020; with one anchor (sd 0.05) at 40 km it is 0.047.
        """
        acc, _ = self.report(SETUP)
        tight, _ = self.report(dataclasses.replace(SETUP, ballistic_log_sd=0.02))
        anchored, _ = self.report(
            dataclasses.replace(SETUP, anchor_altitudes=(40_000.0,))
        )
        self.assertLessEqual(tight.log_ballistic_sd, 0.02)
        self.assertLess(tight.log_density_sd, 0.2 * acc.log_density_sd)
        self.assertLess(anchored.log_ballistic_sd, 0.5 * acc.log_ballistic_sd)
        self.assertLess(anchored.log_density_sd, 0.5 * acc.log_density_sd)
        # the ratio is unchanged by either: it was never the problem
        self.assertAlmostEqual(tight.ratio_sd, acc.ratio_sd, delta=0.2 * acc.ratio_sd)


class FailureCaseTests(unittest.TestCase):
    """Missing data make the bands revert toward the prior."""

    def test_gap_in_data_inflates_the_ratio_band(self) -> None:
        """No samples between 50 and 65 km: the ratio sd at 57.5 km grows.

        Both methods interpolate across the gap with the prior; measured
        growth of the ratio sd at the gap centre is about 3.5x (batch) and
        4x or more (filter). The bounds are 2x and 2.5x.
        """
        band = (50_000.0, 65_000.0)
        truth, gap = case(SETUP, 3, 4, band)
        first = next(
            k for k in range(150) if band[0] <= truth.altitude[k + 1] <= band[1]
        )
        self.assertIsNone(gap.acceleration[first])
        centre = 57_500.0
        batch_full = batch_fit(SETUP, 3, 4).identifiability(centre).ratio_sd
        batch_gap = batch_fit(SETUP, 3, 4, band).identifiability(centre).ratio_sd
        self.assertGreater(batch_gap, 2.0 * batch_full)
        filt_full = filter_fit(SETUP, 3, 4)
        filt_gap = filter_fit(SETUP, 3, 4, band)
        k = min(
            range(1, len(filt_gap.altitude)),
            key=lambda i: abs(filt_gap.altitude[i] - centre),
        )
        self.assertGreater(filt_gap.log_ratio_sd[k], 2.5 * filt_full.log_ratio_sd[k])
        self.assertLess(filt_gap.log_ratio_sd[k], 1.3 * SETUP.density_log_sd)

    def test_altitudes_above_the_flight_revert_to_the_prior(self) -> None:
        """Above the first sample the batch sd of ``c`` approaches the prior.

        At 105 km (5 km above the entry altitude, no data there) the posterior
        sd of ``c`` is measured at 0.29 against the prior 0.30; the bound is
        0.8 of the prior and not above it by more than 5 percent.
        """
        fit = batch_fit(SETUP, 3, 4)
        _, sds = fit.log_density_correction([105_000.0, 40_000.0])
        self.assertGreater(sds[0], 0.8 * SETUP.density_log_sd)
        self.assertLess(sds[0], 1.05 * SETUP.density_log_sd)
        self.assertLess(sds[1], 0.6 * SETUP.density_log_sd)

    def test_filter_band_widens_where_signal_is_below_noise(self) -> None:
        """Above 90 km the filter's ratio sd exceeds 0.8 of the prior sd."""
        fit = filter_fit(SETUP, 3, 4)
        for k in window_indices(fit.altitude, (92_000.0, 100_000.0)):
            self.assertGreater(fit.log_ratio_sd[k], 0.8 * SETUP.density_log_sd)

    def test_us76_base_atmosphere_runs(self) -> None:
        """The standard-atmosphere base works with the filter (one short run)."""
        setup = dataclasses.replace(SETUP, base_atmosphere="us76", duration=60.0)
        truth = synthesize_entry_truth(setup, seed=2)
        data = simulate_entry_measurements(setup, truth, seed=3)
        fit = reconstruct_entry_filter(setup, data)
        self.assertEqual(len(fit.times), 61)
        self.assertTrue(all(math.isfinite(x) for x in fit.log_density_mean))


class CoverageTests(unittest.TestCase):
    """Coverage of the reconstructed 95 percent band at measurement altitudes."""

    @staticmethod
    def pooled_coverage(seeds: tuple[int, ...]) -> dict[str, tuple[int, int]]:
        """Run both methods on drawn truths and pool the band coverage.

        The truth (field, and ``ln beta`` offset drawn from the prior) comes
        from ``seed`` and the noise from ``100 + seed``.

        Args:
            seeds: Predeclared truth seeds.

        Returns:
            ``{"batch": (inside, total), "filter": (inside, total)}`` for the
            1.96 sd band at the true altitudes of the accelerometer samples
            inside the coverage window.

        References:
            [EQ-AERO-RECON-007]. No equation is evaluated.

        """
        counts = {"batch": [0, 0], "filter": [0, 0]}
        for seed in seeds:
            truth = synthesize_entry_truth(SETUP, seed=seed, ballistic_offset=None)
            data = simulate_entry_measurements(SETUP, truth, seed=100 + seed)
            altitudes = truth.altitude[1:]
            batch = reconstruct_entry_batch(SETUP, data)
            means, sds = batch.log_density_correction(altitudes)
            inside, total = band_coverage(
                truth, altitudes, means, sds, altitude_window=COVERAGE_WINDOW
            )
            counts["batch"][0] += inside
            counts["batch"][1] += total
            fit = reconstruct_entry_filter(SETUP, data)
            inside, total = band_coverage(
                truth,
                fit.altitude[1:],
                fit.log_density_mean[1:],
                fit.log_density_sd[1:],
                altitude_window=COVERAGE_WINDOW,
            )
            counts["filter"][0] += inside
            counts["filter"][1] += total
        return {name: (c[0], c[1]) for name, c in counts.items()}

    def test_fast_coverage(self) -> None:
        """Pooled coverage of the 95 percent bands is at least 0.85 (3 seeds).

        The points of one flight are strongly correlated, so with three seeds
        the pooled fraction is far from binomial; 0.85 is a floor meant to
        catch over-confident bands, not a calibration test. Measured over
        eight seeds: batch 849/849, filter 847/848.
        """
        for name, (inside, total) in self.pooled_coverage(PREDECLARED_SEEDS).items():
            self.assertGreater(total, 200)
            self.assertGreaterEqual(inside / total, 0.85, name)

    @unittest.skipUnless(os.environ.get("PROBSTAT_SLOW") == "1", "slow tier")
    def test_slow_coverage_forty_seeds(self) -> None:
        """Pooled coverage over forty predeclared seeds is at least 0.90."""
        start = time.perf_counter()
        counts = self.pooled_coverage(SLOW_SEEDS)
        elapsed = time.perf_counter() - start
        for name, (inside, total) in counts.items():
            print(f"coverage {name}: {inside}/{total} = {inside / total:.4f}")
            self.assertGreaterEqual(inside / total, 0.90, name)
        print(f"slow coverage runtime: {elapsed:.1f} s")


if __name__ == "__main__":
    unittest.main()
