"""Tests for GP anomaly detection on synthetic telemetry (all data synthetic)."""

from __future__ import annotations

import math
import os
import random
import statistics
import unittest
from dataclasses import replace
from functools import cache

from probstat.distributions import StudentT
from probstat.surrogates.gp_anomaly import (
    FAULT_KINDS,
    ChannelSpec,
    NominalModel,
    ScoreSeries,
    SlidingWindowGP,
    SyntheticChannel,
    batch_window_predict,
    calibrate_cusum_threshold,
    conformal_threshold,
    contaminate_values,
    cusum,
    detection_study,
    false_alarm_rate,
    fit_nominal_model,
    gaussian_threshold,
    inject_fault,
    orbital_kernel,
    rolling_zscore,
    simulate_telemetry,
    sliding_window_scores,
    student_t_threshold,
)

SPEC = ChannelSpec(dt=8.0)  # 12 samples per 96-minute orbit, noise sd 0.3
HEAVY = ChannelSpec(dt=8.0, noise="student_t", noise_df=3.0)
TRAIN = 60  # five orbits
WINDOW = 16
TRUE_NOISE_VARIANCE = SPEC.noise_sd**2


@cache
def _path() -> SyntheticChannel:
    """Return one synthetic path: 60 training samples plus 240 continuation."""
    return simulate_telemetry(TRAIN + 240, SPEC, seed=11)


def _train() -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return the nominal training times and values."""
    path = _path()
    return path.times[:TRAIN], path.values[:TRAIN]


@cache
def _clean_model() -> NominalModel:
    """Fit the Gaussian nominal model once (about 3 s)."""
    times, values = _train()
    return fit_nominal_model(
        times, values, period=100.0, seed=1, restarts=1, max_iterations=60
    )


@cache
def _contaminated_models() -> tuple[NominalModel, NominalModel, tuple[float, ...]]:
    """Fit Gaussian and Student-t models to training data with 10% outliers."""
    times, values = _train()
    dirty = contaminate_values(values, 0.1, 4.0, seed=3)
    gaussian = fit_nominal_model(
        times, dirty, period=100.0, seed=1, restarts=1, max_iterations=40
    )
    robust = fit_nominal_model(
        times,
        dirty,
        period=100.0,
        seed=1,
        restarts=1,
        max_iterations=40,
        robust=True,
        start_from=gaussian,
        em_iterations=5,
    )
    return gaussian, robust, dirty


def _scores(spec: ChannelSpec, count: int, seed: int) -> ScoreSeries:
    """Score a fresh nominal series with a 16-sample warm-up."""
    series = simulate_telemetry(WINDOW + count, spec, seed=seed)
    return sliding_window_scores(
        _clean_model(),
        series.times[WINDOW:],
        series.values[WINDOW:],
        window=WINDOW,
        history=(series.times[:WINDOW], series.values[:WINDOW]),
    )


def _rms(values: list[float] | tuple[float, ...]) -> float:
    """Return the root mean square."""
    return math.sqrt(math.fsum(v * v for v in values) / len(values))


class GeneratorTests(unittest.TestCase):
    """The synthetic channel generator against closed forms."""

    def test_matern32_slow_component_matches_theory(self) -> None:
        """Variance and lag-one correlation equal sigma^2 and (1+lam d)e^{-lam d}."""
        spec = ChannelSpec(
            dt=8.0,
            amplitudes=(),
            trend_sd=1.0,
            trend_length_scale=40.0,
            noise_sd=1e-9,
        )
        series = simulate_telemetry(20000, spec, seed=4)
        x = series.signal
        mean = math.fsum(x) / len(x)
        var = math.fsum((v - mean) ** 2 for v in x) / len(x)
        lag = math.fsum(
            (a - mean) * (b - mean) for a, b in zip(x, x[1:], strict=False)
        ) / ((len(x) - 1) * var)
        lam_dt = math.sqrt(3.0) / 40.0 * 8.0
        # about 20000 / 10 independent draws: se(var) ~ 0.06, se(corr) ~ 0.01.
        self.assertAlmostEqual(var, 1.0, delta=0.25)
        self.assertAlmostEqual(lag, (1.0 + lam_dt) * math.exp(-lam_dt), delta=0.05)

    def test_student_t_noise_has_requested_standard_deviation(self) -> None:
        """Heavy-tailed noise is scaled to the stated standard deviation."""
        spec = ChannelSpec(
            amplitudes=(), trend_sd=0.0, noise_sd=2.0, noise="student_t", noise_df=6.0
        )
        series = simulate_telemetry(40000, spec, seed=5)
        # sd of the sample sd for nu = 6 is about 0.2 * sd / sqrt(n) * kurtosis term.
        self.assertAlmostEqual(statistics.pstdev(series.values), 2.0, delta=0.15)

    def test_reproducible_and_validated(self) -> None:
        """Same seed, same path; bad specs are rejected."""
        a = simulate_telemetry(20, SPEC, seed=2).values
        self.assertEqual(a, simulate_telemetry(20, SPEC, seed=2).values)
        self.assertNotEqual(a, simulate_telemetry(20, SPEC, seed=3).values)
        with self.assertRaises(ValueError):
            simulate_telemetry(0, SPEC)
        with self.assertRaises(ValueError):
            ChannelSpec(noise="laplace")
        with self.assertRaises(ValueError):
            ChannelSpec(noise="student_t", noise_df=2.0)
        with self.assertRaises(ValueError):
            ChannelSpec(period=-1.0)


class FitTests(unittest.TestCase):
    """The nominal model recovers the orbital period."""

    def test_periodic_kernel_recovers_orbital_period(self) -> None:
        """Fitted period within 1% of 96 min from a start 4% off."""
        model = _clean_model()
        self.assertLess(abs(model.period - SPEC.period) / SPEC.period, 0.01)
        self.assertTrue(math.isinf(model.nu))

    def test_noise_variance_is_in_the_right_range(self) -> None:
        """Noise variance within a factor 2 of the truth with 60 samples."""
        ratio = _clean_model().noise_variance / TRUE_NOISE_VARIANCE
        self.assertTrue(0.5 < ratio < 2.0, ratio)

    def test_period_bounds_exclude_the_alias(self) -> None:
        """A start whose bounds exclude the period is rejected, not clipped."""
        times, values = _train()
        with self.assertRaises(ValueError):
            fit_nominal_model(times, values, period=100.0, period_bounds=(101.0, 130.0))

    def test_kernel_layout(self) -> None:
        """The helper builds trend + periodic + white."""
        kernel = orbital_kernel(96.0, noise_variance=0.5)
        self.assertEqual(kernel.diagonal_variance, 0.5)
        # The periodic part repeats after one period; the trend part decays.
        periodic_only = orbital_kernel(96.0, trend_variance=1e-12)
        self.assertAlmostEqual(
            periodic_only.evaluate((3.0,), (99.0,)),
            periodic_only.evaluate((3.0,), (3.0,)),
            places=9,
        )


class StreamingTests(unittest.TestCase):
    """Sliding-window GP equals the batch GP on the window."""

    def test_single_point_closed_form(self) -> None:
        """One observation: mu = m + k/(k0 + s2) (y - m) and the matching variance."""
        kernel = orbital_kernel(96.0, trend_variance=2.0, periodic_variance=1.0)
        gp = SlidingWindowGP(kernel, 1.5, window=4)
        gp.update(0.0, 2.5, 0.2)
        k0 = kernel.evaluate((0.0,), (0.0,))
        k1 = kernel.evaluate((0.0,), (10.0,))
        mean, var = gp.predict(10.0)
        self.assertAlmostEqual(mean, 1.5 + k1 / (k0 + 0.2) * 1.0, places=14)
        self.assertAlmostEqual(
            var, kernel.evaluate((10.0,), (10.0,)) - k1 * k1 / (k0 + 0.2), places=14
        )

    def test_streaming_equals_batch_on_the_window(self) -> None:
        """Irregular times, per-sample noise, many window slides: 1e-9 agreement."""
        rng = random.Random(8)
        kernel = orbital_kernel(96.0, trend_variance=1.5, trend_length_scale=150.0)
        gp = SlidingWindowGP(kernel, 0.7, window=12)
        times: list[float] = []
        values: list[float] = []
        noises: list[float] = []
        clock = 0.0
        worst = 0.0
        for step in range(80):
            clock += rng.uniform(2.0, 14.0)
            value = math.sin(2 * math.pi * clock / 96.0) + rng.gauss(0.0, 0.3)
            noise = rng.uniform(0.05, 0.4)
            if times:
                tail = slice(-12, None)
                batch = batch_window_predict(
                    kernel,
                    0.7,
                    times[tail],
                    values[tail],
                    noises[tail],
                    clock,
                )
                stream = gp.predict(clock)
                worst = max(
                    worst,
                    abs(stream[0] - batch[0]),
                    abs(stream[1] - batch[1]),
                )
            gp.update(clock, value, noise)
            times.append(clock)
            values.append(value)
            noises.append(noise)
            self.assertEqual(gp.size, min(step + 1, 12))
        self.assertLess(worst, 1e-9)
        self.assertEqual(gp.times, tuple(times[-12:]))

    def test_first_score_equals_batch_prediction(self) -> None:
        """The z of the first scored sample equals the dense batch computation."""
        model = _clean_model()
        path = _path()
        scored = sliding_window_scores(
            model,
            path.times[TRAIN : TRAIN + 3],
            path.values[TRAIN : TRAIN + 3],
            window=WINDOW,
        )
        mean, latent = batch_window_predict(
            model.kernel,
            model.output_mean,
            model.times[-WINDOW:],
            model.values[-WINDOW:],
            (model.noise_variance,) * WINDOW,
            path.times[TRAIN],
        )
        expected = (path.values[TRAIN] - mean) / math.sqrt(
            latent + model.noise_variance
        )
        self.assertAlmostEqual(scored.z[0], expected, places=8)
        self.assertAlmostEqual(scored.mean[0], mean, places=8)

    def test_bounded_window_cost_is_independent_of_history(self) -> None:
        """The window never grows beyond its bound."""
        gp = SlidingWindowGP(orbital_kernel(96.0), 0.0, window=5)
        for i in range(40):
            gp.update(float(i) * 7.0, math.sin(i), 0.1)
        self.assertEqual(gp.size, 5)

    def test_gate_keeps_a_step_out_of_the_window(self) -> None:
        """Without the gate the window absorbs a step; with it z stays large."""
        model = _clean_model()
        path = _path()
        times = path.times[TRAIN : TRAIN + 40]
        values = [
            v + (6.0 if i >= 10 else 0.0)
            for i, v in enumerate(path.values[TRAIN : TRAIN + 40])
        ]
        free = sliding_window_scores(model, times, values, window=WINDOW)
        gated = sliding_window_scores(model, times, values, window=WINDOW, gate=4.0)
        late_free = _rms(free.z[30:])
        late_gated = _rms(gated.z[30:])
        self.assertGreater(late_gated, 3.0 * late_free)
        self.assertGreater(late_gated, 5.0)

    def test_robust_updates_limit_the_damage_of_a_spike(self) -> None:
        """A spike enters the window with a small weight when nu is finite."""
        model = replace(_clean_model(), nu=4.0)
        path = _path()
        times = path.times[TRAIN : TRAIN + 30]
        values = list(path.values[TRAIN : TRAIN + 30])
        values[10] += 5.0
        plain = sliding_window_scores(model, times, values, window=WINDOW)
        soft = sliding_window_scores(
            model, times, values, window=WINDOW, robust_updates=True
        )
        self.assertLess(_rms(soft.z[11:16]), _rms(plain.z[11:16]))

    def test_invalid_use_is_rejected(self) -> None:
        """Unordered times, tiny windows, bad noise."""
        kernel = orbital_kernel(96.0)
        with self.assertRaises(ValueError):
            SlidingWindowGP(kernel, 0.0, window=1)
        gp = SlidingWindowGP(kernel, 0.0, window=3)
        gp.update(1.0, 0.0, 0.1)
        with self.assertRaises(ValueError):
            gp.update(1.0, 0.0, 0.1)
        with self.assertRaises(ValueError):
            gp.update(2.0, 0.0, 0.0)
        with self.assertRaises(ValueError):
            gp.update(2.0, math.nan, 0.1)
        with self.assertRaises(ValueError):
            gp.predict(math.inf)
        self.assertEqual(gp.size, 1)


class ScoreTests(unittest.TestCase):
    """Residual scores, NLPD and CUSUM against closed forms."""

    def test_nlpd_gaussian_closed_form(self) -> None:
        """NLPD = (1/2) log(2 pi s^2) + z^2/2."""
        scored = _scores(SPEC, 30, seed=21)
        for z, var, nlpd in zip(scored.z, scored.variance, scored.nlpd, strict=True):
            self.assertAlmostEqual(
                nlpd, 0.5 * math.log(2 * math.pi * var) + 0.5 * z * z, places=12
            )

    def test_nlpd_student_t_matches_distribution_and_gaussian_limit(self) -> None:
        """The t NLPD equals -log pdf of StudentT; nu -> inf gives the Gaussian."""
        model = _clean_model()
        series = simulate_telemetry(WINDOW + 20, SPEC, seed=22)
        args = (series.times[WINDOW:], series.values[WINDOW:])
        hist = (series.times[:WINDOW], series.values[:WINDOW])
        gauss = sliding_window_scores(model, *args, window=WINDOW, history=hist)
        student = sliding_window_scores(
            replace(model, nu=5.0), *args, window=WINDOW, history=hist
        )
        near = sliding_window_scores(
            replace(model, nu=1e9), *args, window=WINDOW, history=hist
        )
        for i in range(20):
            law = StudentT(5.0, student.mean[i], math.sqrt(student.variance[i]))
            self.assertAlmostEqual(
                student.nlpd[i], -law.log_pdf(student.values[i]), places=9
            )
            self.assertAlmostEqual(near.nlpd[i], gauss.nlpd[i], places=5)

    def test_cusum_closed_form(self) -> None:
        """Hand-computed sums, reset and symmetry."""
        result = cusum([1.0, 1.0, 1.0, 0.0, 0.0], k=0.5)
        self.assertEqual(result.statistic, (0.5, 1.0, 1.5, 1.0, 0.5))
        down = cusum([-1.0, -1.0, -1.0, 0.0, 0.0], k=0.5)
        self.assertEqual(down.statistic, result.statistic)
        alarmed = cusum([1.0] * 6, k=0.5, h=1.2)
        self.assertEqual(alarmed.alarms, (2, 5))
        self.assertEqual(cusum([0.2, -0.3, 0.4], k=0.5).statistic, (0.0, 0.0, 0.0))

    def test_cusum_threshold_matches_target_rate_on_nominal_data(self) -> None:
        """The calibrated h gives at most the target rate; a lower h alarms more."""
        rng = random.Random(3)
        z = [rng.gauss(0.0, 1.0) for _ in range(4000)]
        h = calibrate_cusum_threshold(z, 0.01, k=0.5)
        self.assertLessEqual(len(cusum(z, k=0.5, h=h).alarms) / 4000, 0.01)
        self.assertGreater(len(cusum(z, k=0.5, h=0.97 * h).alarms) / 4000, 0.0)
        # A shift of 2 sd is caught within a few samples at that threshold.
        shifted = [v + 2.0 for v in z[:100]]
        self.assertTrue(cusum(shifted, k=0.5, h=h).alarms)

    def test_thresholds_closed_form(self) -> None:
        """Gaussian 1.959964, Student-t(10) 2.228139, conformal rank rule."""
        self.assertAlmostEqual(gaussian_threshold(0.05), 1.959963984540054, places=9)
        self.assertAlmostEqual(
            student_t_threshold(0.05, 10.0), 2.2281388519649385, places=7
        )
        self.assertGreater(student_t_threshold(0.01, 3.0), gaussian_threshold(0.01))
        scores = [float(i) for i in range(99, 0, -1)]  # 1..99, reversed
        # n = 99: rank = ceil(0.9 * 100) = 90. Float 0.9 * 100 = 90.00000000000001
        # would round the rank up to 91; the rational arithmetic does not.
        self.assertEqual(conformal_threshold(scores, 0.1), 90.0)
        self.assertEqual(
            conformal_threshold([float(i) for i in range(1, 20)], 0.05), 19.0
        )
        with self.assertRaises(ValueError):
            conformal_threshold([float(i) for i in range(1, 19)], 0.05)

    def test_conformal_threshold_is_exact_for_exchangeable_scores(self) -> None:
        """Leave-one-out over 5 exchangeable scores: P(new > q) = alpha exactly."""
        # n = 4 calibration scores + 1 new, alpha = 0.4: rank = ceil(0.6 * 5) = 3.
        pool = [0.3, 1.1, 1.9, 2.4, 5.0]
        hits = 0
        for new in range(5):
            cal = [v for i, v in enumerate(pool) if i != new]
            hits += pool[new] > conformal_threshold(cal, 0.4)
        self.assertEqual(hits / 5, 0.4)

    def test_rolling_baseline(self) -> None:
        """Previous-window mean and sample sd; zero spread is rejected."""
        z = rolling_zscore([10.0], 4, history=[1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(z[0], (10.0 - 2.5) / math.sqrt(5.0 / 3.0), places=12)
        with self.assertRaises(ValueError):
            rolling_zscore([1.0], 4, history=[2.0, 2.0, 2.0, 2.0])
        with self.assertRaises(ValueError):
            rolling_zscore([1.0], 4, history=[1.0, 2.0])


class FaultTests(unittest.TestCase):
    """Fault injection arithmetic."""

    def test_each_fault_kind(self) -> None:
        """Exact offsets for spike, step, drift; variance acts only after onset."""
        base = [0.0] * 10
        kw = {"onset": 4, "size": 3.0, "scale": 0.5}
        spike = inject_fault(base, "spike", duration=2, **kw)
        self.assertEqual(spike.values, (0.0,) * 4 + (1.5, 1.5) + (0.0,) * 4)
        self.assertEqual(spike.affected[4:7], (True, True, False))
        step = inject_fault(base, "step", **kw)
        self.assertEqual(step.values, (0.0,) * 4 + (1.5,) * 6)
        drift = inject_fault(base, "drift", drift_length=3, **kw)
        self.assertEqual(drift.values[3:8], (0.0, 0.5, 1.0, 1.5, 2.0))
        noisy = inject_fault(base, "variance", seed=1, **kw)
        self.assertEqual(noisy.values[:4], (0.0,) * 4)
        self.assertTrue(all(v != 0.0 for v in noisy.values[4:]))
        self.assertEqual(noisy, inject_fault(base, "variance", seed=1, **kw))

    def test_invalid_faults_are_rejected(self) -> None:
        """Unknown kind, bad onset, non-positive size."""
        with self.assertRaises(ValueError):
            inject_fault([0.0] * 5, "ramp", onset=1, size=1.0, scale=1.0)
        with self.assertRaises(ValueError):
            inject_fault([0.0] * 5, "step", onset=5, size=1.0, scale=1.0)
        with self.assertRaises(ValueError):
            inject_fault([0.0] * 5, "step", onset=1, size=0.0, scale=1.0)
        self.assertEqual(FAULT_KINDS, ("spike", "step", "drift", "variance"))


class CalibrationTests(unittest.TestCase):
    """Realized false-alarm rates of the Gaussian and conformal thresholds."""

    PAIRS = 10
    N_CAL = 400
    N_FRESH = 400

    @classmethod
    def _pairs(cls, spec: ChannelSpec) -> list[tuple[ScoreSeries, ScoreSeries]]:
        """Return independent (calibration, fresh) nominal score series."""
        return [
            (
                _scores(spec, cls.N_CAL, 1000 + 2 * r),
                _scores(spec, cls.N_FRESH, 1001 + 2 * r),
            )
            for r in range(cls.PAIRS)
        ]

    def _band(self, alpha: float) -> float:
        """Return a 4-sigma band: binomial with design effect 2 (autocorrelation)."""
        per_pair = alpha * (1 - alpha) * (1 / self.N_CAL + 1 / self.N_FRESH)
        return 4.0 * math.sqrt(2.0 * per_pair / self.PAIRS)

    def test_conformal_rate_within_binomial_band_and_gaussian_is_off(self) -> None:
        """Conformal alpha=0.05 holds; the Gaussian quantile is off if heavy-tailed."""
        pairs = self._pairs(SPEC)
        conformal = statistics.fmean(
            false_alarm_rate(f.abs_z, conformal_threshold(c.abs_z, 0.05))
            for c, f in pairs
        )
        self.assertLess(abs(conformal - 0.05), self._band(0.05))
        heavy = self._pairs(HEAVY)
        conformal_heavy = statistics.fmean(
            false_alarm_rate(f.abs_z, conformal_threshold(c.abs_z, 0.05))
            for c, f in heavy
        )
        self.assertLess(abs(conformal_heavy - 0.05), self._band(0.05))
        # alpha = 0.01 on heavy-tailed data: Gaussian quantile vs conformal.
        gauss_rate = statistics.fmean(
            false_alarm_rate(f.abs_z, gaussian_threshold(0.01)) for _, f in heavy
        )
        conf_rate = statistics.fmean(
            false_alarm_rate(f.abs_z, conformal_threshold(c.abs_z, 0.01))
            for c, f in heavy
        )
        self.assertGreater(gauss_rate, 1.8 * 0.01)
        self.assertLess(conf_rate, gauss_rate)
        self.assertLess(conf_rate, 0.01 + self._band(0.01))

    def test_conformal_threshold_needs_enough_calibration_data(self) -> None:
        """No silent infinite threshold for tiny calibration sets."""
        with self.assertRaises(ValueError):
            conformal_threshold([1.0] * 50, 0.01)


class StudyTests(unittest.TestCase):
    """Operating characteristics against the rolling-z baseline."""

    @classmethod
    def setUpClass(cls) -> None:
        """Run the fast operating-characteristic study once."""
        cls.study = detection_study(
            _clean_model(),
            SPEC,
            alpha=0.01,
            faults=("spike", "step", "variance"),
            sizes=(2.0, 4.0, 8.0),
            replicates=8,
            calibration_series=4,
            series_length=80,
            onset=30,
            window=WINDOW,
            horizon=30,
            seed=5,
        )

    def test_delay_decreases_with_fault_size(self) -> None:
        """Strictly smaller mean delay for larger step and variance faults."""
        for fault, method in (
            ("step", "gp_cusum"),
            ("step", "gp_abs_z"),
            ("variance", "gp_abs_z"),
        ):
            delays = [
                self.study.get(fault, s, method).mean_delay for s in (2.0, 4.0, 8.0)
            ]
            self.assertGreater(delays[0], delays[1], (fault, method, delays))
            self.assertGreaterEqual(delays[1], delays[2], (fault, method, delays))

    def test_detection_rate_grows_with_spike_size(self) -> None:
        """Spike detection rate is non-decreasing in size and reaches one."""
        rates = [
            self.study.get("spike", s, "gp_abs_z").detection_rate
            for s in (2.0, 4.0, 8.0)
        ]
        self.assertEqual(rates, sorted(rates))
        self.assertEqual(rates[-1], 1.0)

    def test_gp_beats_rolling_baseline_at_equal_false_alarm_rate(self) -> None:
        """Both thresholds target the same rate; the GP detects far more spikes."""
        gp = self.study.get("spike", 4.0, "gp_abs_z").detection_rate
        base = self.study.get("spike", 4.0, "baseline_abs_z").detection_rate
        self.assertGreater(gp, base + 0.5)
        step_gp = self.study.get("step", 4.0, "gp_abs_z")
        step_base = self.study.get("step", 4.0, "baseline_abs_z")
        self.assertGreater(step_gp.detection_rate, step_base.detection_rate)
        self.assertLess(step_gp.mean_delay, step_base.mean_delay)

    def test_pre_onset_alarm_rates_are_near_the_target(self) -> None:
        """No method alarms before the onset at a rate far above alpha."""
        rates = [p.pre_onset_alarm_rate for p in self.study.points]
        self.assertLess(statistics.fmean(rates), 0.03)

    def test_lookup_and_validation(self) -> None:
        """Missing rows raise; invalid arguments are rejected."""
        with self.assertRaises(KeyError):
            self.study.get("drift", 2.0, "gp_abs_z")
        with self.assertRaises(ValueError):
            detection_study(_clean_model(), SPEC, onset=500, series_length=80)
        with self.assertRaises(ValueError):
            detection_study(_clean_model(), SPEC, faults=("ramp",))
        with self.assertRaises(ValueError):
            detection_study(_clean_model(), SPEC, alpha=0.001, calibration_series=1)


class RobustFitTests(unittest.TestCase):
    """Student-t fit versus Gaussian fit on contaminated training data."""

    def test_student_t_fit_is_less_affected_by_outliers(self) -> None:
        """Noise estimate and prediction error, measured on a clean continuation."""
        gaussian, robust, dirty = _contaminated_models()
        _, clean = _train()
        flagged = [
            i for i, (a, b) in enumerate(zip(clean, dirty, strict=True)) if a != b
        ]
        lowest = sorted(range(TRAIN), key=lambda i: robust.weights[i])[: len(flagged)]
        self.assertEqual(sorted(lowest), flagged)
        self.assertTrue(all(robust.weights[i] < 0.5 for i in flagged))

        def log_error(model: NominalModel) -> float:
            return abs(math.log(model.noise_variance / TRUE_NOISE_VARIANCE))

        self.assertLess(log_error(robust), 0.7 * log_error(gaussian))
        self.assertGreater(gaussian.noise_variance, 5.0 * TRUE_NOISE_VARIANCE)

        path = _path()
        truth = path.signal[TRAIN:]

        def rmse(model: NominalModel) -> float:
            scored = sliding_window_scores(
                model, path.times[TRAIN:], path.values[TRAIN:], window=WINDOW
            )
            return _rms([m - s for m, s in zip(scored.mean, truth, strict=True)])

        self.assertLess(rmse(robust), rmse(gaussian))
        self.assertTrue(math.isfinite(robust.nu))
        self.assertAlmostEqual(robust.period, SPEC.period, delta=2.0)

    def test_gaussian_fit_has_unit_weights(self) -> None:
        """No sample is down-weighted by the Gaussian model."""
        self.assertEqual(set(_clean_model().weights), {1.0})


class FailureTests(unittest.TestCase):
    """Invalid input and adversarial cases."""

    def test_fit_rejections(self) -> None:
        """Constant data, short span, unordered times, NaN, bad nu."""
        times = tuple(float(i) * 8.0 for i in range(30))
        with self.assertRaises(ValueError):
            fit_nominal_model(times, (1.0,) * 30, period=96.0)
        with self.assertRaises(ValueError):
            fit_nominal_model(
                times[:10], tuple(float(i % 3) for i in range(10)), period=96.0
            )
        with self.assertRaises(ValueError):
            fit_nominal_model(
                times[::-1], tuple(float(i % 3) for i in range(30)), period=96.0
            )
        with self.assertRaises(ValueError):
            fit_nominal_model(times, (math.nan,) + (1.0,) * 29, period=96.0)
        with self.assertRaises(ValueError):
            fit_nominal_model(
                times,
                tuple(float(i % 3) for i in range(30)),
                period=96.0,
                robust=True,
                nu=2.0,
            )
        with self.assertRaises(ValueError):
            fit_nominal_model(
                times,
                tuple(float(i % 3) for i in range(30)),
                period=96.0,
                start_from=_clean_model(),
            )

    def test_scoring_rejections(self) -> None:
        """Times before the history, bad gate, robust updates without a t model."""
        model = _clean_model()
        path = _path()
        with self.assertRaises(ValueError):
            sliding_window_scores(model, path.times[:5], path.values[:5], window=WINDOW)
        later = (path.times[TRAIN], path.times[TRAIN + 1])
        vals = (path.values[TRAIN], path.values[TRAIN + 1])
        with self.assertRaises(ValueError):
            sliding_window_scores(model, later, vals, window=WINDOW, gate=-1.0)
        with self.assertRaises(ValueError):
            sliding_window_scores(
                model, later, vals, window=WINDOW, robust_updates=True
            )
        with self.assertRaises(ValueError):
            sliding_window_scores(model, later[::-1], vals, window=WINDOW)
        with self.assertRaises(ValueError):
            sliding_window_scores(model, later, vals[:1], window=WINDOW)

    def test_threshold_and_cusum_rejections(self) -> None:
        """Alpha outside (0, 1), negative k, too few samples."""
        for bad in (0.0, 1.0, -0.1, 2.0):
            with self.assertRaises(ValueError):
                gaussian_threshold(bad)
        with self.assertRaises(TypeError):
            gaussian_threshold("0.05")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            cusum([1.0], k=-1.0)
        with self.assertRaises(ValueError):
            cusum([1.0], h=0.0)
        with self.assertRaises(ValueError):
            calibrate_cusum_threshold([0.0] * 10, 0.01)
        with self.assertRaises(ValueError):
            false_alarm_rate([], 1.0)
        with self.assertRaises(ValueError):
            student_t_threshold(0.05, 0.0)
        with self.assertRaises(ValueError):
            contaminate_values([1.0, 2.0], 1.0, 1.0)

    def test_model_mismatch_is_visible_in_the_scores(self) -> None:
        """A wrong-period model has a much larger residual scale: no silent success."""
        wrong = replace(
            _clean_model(),
            kernel=orbital_kernel(70.0, trend_variance=0.1, periodic_variance=0.1),
        )
        series = simulate_telemetry(WINDOW + 100, SPEC, seed=31)
        good = _scores(SPEC, 100, seed=31)
        bad = sliding_window_scores(
            wrong,
            series.times[WINDOW:],
            series.values[WINDOW:],
            window=WINDOW,
            history=(series.times[:WINDOW], series.values[:WINDOW]),
        )
        self.assertGreater(_rms(bad.z), 2.0 * _rms(good.z))


@unittest.skipUnless(os.environ.get("PROBSTAT_SLOW") == "1", "slow tier")
class SlowStudyTests(unittest.TestCase):
    """Larger operating-characteristic and calibration studies."""

    def test_full_study_all_faults(self) -> None:
        """Thirty replicates over all four fault kinds; detection grows with size."""
        study = detection_study(
            _clean_model(),
            SPEC,
            alpha=0.01,
            sizes=(3.0, 6.0, 12.0),
            replicates=30,
            calibration_series=12,
            series_length=120,
            onset=50,
            window=WINDOW,
            horizon=30,
            seed=9,
        )
        for fault in ("spike", "step", "variance"):
            rates = [
                study.get(fault, s, "gp_abs_z").detection_rate for s in (3.0, 6.0, 12.0)
            ]
            self.assertEqual(rates, sorted(rates), fault)
        for point in study.points:
            print(
                f"{point.fault:9s}{point.size:6.1f} {point.method:15s}"
                f" rate={point.detection_rate:.2f} delay={point.mean_delay:6.2f}"
            )


if __name__ == "__main__":
    unittest.main()
