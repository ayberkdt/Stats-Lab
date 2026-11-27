"""Tests for the calibration of a simulator with field-valued output.

Everything is synthetic. The decay-profile "simulator" is
``eta(s; x, t) = 0.2 + x exp(-t s)`` on a grid ``s in [0, 1]`` (a source
strength ``x`` and a decay rate ``t``, in the manner of a one-dimensional
conduction profile); the synthetic "reality" is the simulator at the true rate
``theta = 1.4`` (optionally plus a Gaussian-bump bias near ``s = 0.75``) with
iid noise. Order of the tests: basis and emulator with an exactly
two-dimensional output subspace (closed form), the reduced likelihood against
a dense oracle assembled in the test from the kernel definition, synthetic
calibration with a known truth (recovery, bias with and without discrepancy,
single-location comparison, predictive bands), then failure cases. The
many-seed coverage study runs only when ``PROBSTAT_SLOW=1``.
"""

from __future__ import annotations

import math
import os
import random
import unittest
import warnings
from decimal import Decimal, getcontext

from probstat.linalg import inverse_spd, log_det_spd, matvec, symmetric_eigen
from probstat.surrogates.gaussian_process import GaussianProcessConditioningWarning
from probstat.validation import KOHResult, LogNormalPrior
from probstat.validation.functional_calibration import (
    EmulatorValidation,
    FieldBasis,
    FieldEmulator,
    FieldEmulatorPrediction,
    FieldPrediction,
    FunctionalCalibration,
    ScalarComparison,
    fit_field_basis,
    gaussian_bump_basis,
)

THETA_TRUE = 1.4
NOISE_SD = 0.01
GRID = 11
LOCATIONS = tuple(i / (GRID - 1) for i in range(GRID))
CONTROLS = ((0.6,), (1.0,), (1.4,))
BUMPS = (0.2, 0.5, 0.8)
BUMP_WIDTH = 0.18
SAMPLE = {"draws": 100, "warmup": 50}


def simulator(
    x: float, t: float, locations: tuple[float, ...] = LOCATIONS
) -> tuple[float, ...]:
    """Decay profile ``0.2 + x exp(-t s)``."""
    return tuple(0.2 + x * math.exp(-t * s) for s in locations)


def bias(s: float) -> float:
    """Structural error missing from the simulator (a bump near ``s = 0.75``)."""
    return 0.08 * math.exp(-0.5 * ((s - 0.75) / 0.12) ** 2)


def stratified_design(
    runs: int, seed: int
) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[float, ...], ...]]:
    """Return a stratified random design on ``x in [0.5, 1.5]``, ``t in [0.5, 2.5]``."""
    rng = random.Random(seed)
    xs = [(i + rng.random()) / runs for i in range(runs)]
    ts = [(i + rng.random()) / runs for i in range(runs)]
    rng.shuffle(ts)
    return (
        tuple((0.5 + x,) for x in xs),
        tuple((0.5 + 2.0 * t,) for t in ts),
    )


def field_observations(
    seed: int, *, biased: bool, locations: tuple[float, ...] = LOCATIONS
) -> tuple[tuple[float, ...], ...]:
    """Return noisy field data at the experiment controls for a seed."""
    rng = random.Random(seed)
    rows = []
    for (x,) in CONTROLS:
        truth = simulator(x, THETA_TRUE, locations)
        rows.append(
            tuple(
                v + (bias(s) if biased else 0.0) + NOISE_SD * rng.gauss(0.0, 1.0)
                for v, s in zip(truth, locations, strict=True)
            )
        )
    return tuple(rows)


_CACHE: dict[object, object] = {}


def decay_emulator() -> FieldEmulator:
    """Return the (cached) emulator of the decay-profile simulator, 16 runs."""
    if "emulator" not in _CACHE:
        controls, parameters = stratified_design(16, 1)
        fields = tuple(
            simulator(x[0], t[0]) for x, t in zip(controls, parameters, strict=True)
        )
        _CACHE["emulator"] = FieldEmulator(
            controls, parameters, fields, locations=LOCATIONS, restarts=1, seed=1
        )
    emulator = _CACHE["emulator"]
    assert isinstance(emulator, FieldEmulator)
    return emulator


def make_calibration(
    seed: int, *, biased: bool, discrepancy: bool
) -> FunctionalCalibration:
    """Build a calibration of the synthetic field data."""
    return FunctionalCalibration(
        decay_emulator(),
        CONTROLS,
        field_observations(seed, biased=biased),
        theta_bounds=[(0.5, 2.5)],
        noise_sd=NOISE_SD,
        discrepancy_basis=(
            gaussian_bump_basis(LOCATIONS, BUMPS, BUMP_WIDTH) if discrepancy else None
        ),
    )


def posterior(seed: int, *, biased: bool, discrepancy: bool) -> KOHResult:
    """Return the cached posterior of one case."""
    key = ("posterior", seed, biased, discrepancy)
    if key not in _CACHE:
        calibration = make_calibration(seed, biased=biased, discrepancy=discrepancy)
        _CACHE[key] = calibration.sample(seed=3, **SAMPLE)
    result = _CACHE[key]
    assert isinstance(result, KOHResult)
    return result


class TwoComponentFixture(unittest.TestCase):
    """Simulator whose centred output lies exactly in a two-dimensional space."""

    GRID = tuple(i / 8.0 for i in range(9))

    @staticmethod
    def field(x: float, t: float) -> tuple[float, ...]:
        """``phi0 + (x + t) phi1 + x t phi2`` with fixed smooth profiles."""
        grid = TwoComponentFixture.GRID
        return tuple(
            1.0
            + math.sin(math.pi * s)
            + (x + t) * math.cos(2.0 * math.pi * s)
            + x * t * (s - 0.3) ** 2
            for s in grid
        )

    @classmethod
    def setUpClass(cls) -> None:
        """Build 14 runs and the emulator (near-singular GP warnings muted)."""
        controls, parameters = stratified_design(14, 7)
        cls.controls = controls
        cls.parameters = parameters
        cls.fields = tuple(
            cls.field(x[0], t[0]) for x, t in zip(controls, parameters, strict=True)
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", GaussianProcessConditioningWarning)
            cls.emulator = FieldEmulator(
                controls,
                parameters,
                cls.fields,
                locations=cls.GRID,
                variance_threshold=1.0 - 1e-9,
                restarts=1,
                seed=2,
            )


class BasisExactRecoveryTests(TwoComponentFixture):
    """Closed-form case: truncation error is zero and the emulator interpolates."""

    def test_two_components_and_zero_truncation(self) -> None:
        """The basis keeps two modes; the omitted energy is rounding only."""
        basis = self.emulator.basis
        self.assertIsInstance(basis, FieldBasis)
        self.assertEqual(basis.components, 2)
        self.assertLess(basis.truncation_fraction, 1e-20)
        self.assertLess(max(basis.truncation_variance), 1e-20)
        self.assertLess(basis.singular_values[2], 1e-10 * basis.singular_values[0])

    def test_singular_values_match_gram_eigenvalues(self) -> None:
        """Oracle: eigenvalues of the Gram matrix of the scaled data are ``S^2``."""
        basis = self.emulator.basis
        scaled = [
            [(v - m) / basis.scale for v, m in zip(row, basis.mean, strict=True)]
            for row in self.fields
        ]
        count = len(basis.mean)
        gram = tuple(
            tuple(math.fsum(row[a] * row[b] for row in scaled) for b in range(count))
            for a in range(count)
        )
        values, _ = symmetric_eigen(gram)
        top = sorted(values, reverse=True)[:2]
        for eigenvalue, sigma in zip(top, basis.singular_values[:2], strict=True):
            self.assertAlmostEqual(eigenvalue, sigma * sigma, delta=1e-9 * top[0])

    def test_reconstruction_of_new_runs_is_exact(self) -> None:
        """Fields from the same simulator are reproduced by the two modes."""
        rng = random.Random(3)
        for _ in range(5):
            x, t = 0.5 + rng.random(), 0.5 + 2.0 * rng.random()
            truth = self.field(x, t)
            rebuilt = self.emulator.basis.reconstruct(
                self.emulator.basis.project(truth)
            )
            self.assertLess(
                max(abs(a - b) for a, b in zip(rebuilt, truth, strict=True)), 1e-9
            )

    def test_emulator_interpolates_the_runs(self) -> None:
        """At the training inputs the field prediction is the run, variance ~ 0.

        The GP nugget is ``1e-8`` of the standardized signal variance, so the
        mean differs from the training value by at most about that fraction of
        the weight spread; ``1e-5`` of the field scale is a safe bound.
        """
        prediction = self.emulator.predict(self.controls, self.parameters)
        self.assertIsInstance(prediction, FieldEmulatorPrediction)
        scale = self.emulator.basis.scale
        worst = max(
            abs(a - b)
            for got, run in zip(prediction.field_mean, self.fields, strict=True)
            for a, b in zip(got, run, strict=True)
        )
        self.assertLess(worst, 1e-5 * scale)
        self.assertLess(
            max(max(row) for row in prediction.emulator_variance), 1e-7 * scale**2
        )
        self.assertEqual(
            prediction.truncation_variance, self.emulator.basis.truncation_variance
        )

    def test_held_out_validation_is_near_perfect(self) -> None:
        """Held-out runs of an exactly two-dimensional simulator."""
        controls, parameters = stratified_design(6, 99)
        fields = tuple(
            self.field(x[0], t[0]) for x, t in zip(controls, parameters, strict=True)
        )
        report = self.emulator.validate(controls, parameters, fields)
        self.assertIsInstance(report, EmulatorValidation)
        self.assertLess(report.truncation_rmse, 1e-9)
        self.assertLess(report.field_relative_rmse, 0.05)
        self.assertEqual(report.components, 2)

    def test_reduced_data_of_nodal_subgrid_is_exact(self) -> None:
        """Field on a subset of grid nodes: projected data equal the true weights.

        With noise-free data inside the span of the basis the generalized
        least-squares coefficients are exact even though the subgrid basis is
        not orthonormal.
        """
        stride = (0, 2, 4, 5, 8)
        sub = tuple(self.GRID[i] for i in stride)
        x, t = 0.9, 1.7
        truth = self.field(x, t)
        calibration = FunctionalCalibration(
            self.emulator,
            [(x,)],
            [tuple(truth[i] for i in stride)],
            theta_bounds=[(0.5, 2.5)],
            noise_sd=1e-3,
            field_locations=sub,
        )
        expected = self.emulator.basis.project(truth)
        got = calibration.reduced_data[0]
        for a, b in zip(got, expected, strict=True):
            self.assertAlmostEqual(a, b, delta=1e-9)


class ReducedLikelihoodOracleTests(unittest.TestCase):
    """The reduced likelihood against a dense assembly written in the test."""

    @staticmethod
    def conditional(
        emulator: FieldEmulator, j: int, points: tuple[tuple[float, ...], ...]
    ) -> tuple[list[float], list[list[float]]]:
        """Conditional GP mean and covariance in 60-digit decimal arithmetic.

        The fitted weight processes have signal variances near ``1e4`` and
        kernel condition numbers near ``1e10``, so a float matrix inverse loses
        about six digits (measured: ``1.6e-4`` absolute error in a
        covariance entry of ``1.8e-4``); the oracle therefore uses decimals.
        """
        getcontext().prec = 60
        gp = emulator.gps[j]
        hyper = gp.hyperparameters
        sqrt5 = Decimal(5).sqrt()

        def kern(a: tuple[float, ...], b: tuple[float, ...]) -> Decimal:
            """Matern 5/2 kernel."""
            r2 = sum(
                ((Decimal(x) - Decimal(y)) / Decimal(ell)) ** 2
                for x, y, ell in zip(a, b, hyper.length_scales, strict=True)
            )
            r = r2.sqrt()
            return (
                Decimal(hyper.signal_variance)
                * (1 + sqrt5 * r + 5 * r2 / 3)
                * (-sqrt5 * r).exp()
            )

        def solve(matrix: list[list[Decimal]], rhs: list[Decimal]) -> list[Decimal]:
            """Gaussian elimination with partial pivoting."""
            size = len(matrix)
            work = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
            for c in range(size):
                pivot = max(range(c, size), key=lambda r: abs(work[r][c]))
                work[c], work[pivot] = work[pivot], work[c]
                for r in range(c + 1, size):
                    factor = work[r][c] / work[c][c]
                    for cc in range(c, size + 1):
                        work[r][cc] -= factor * work[c][cc]
            out = [Decimal(0)] * size
            for r in range(size - 1, -1, -1):
                tail = sum(work[r][c] * out[c] for c in range(r + 1, size))
                out[r] = (work[r][size] - tail) / work[r][r]
            return out

        runs = gp.points
        size = len(runs)
        matrix = [
            [
                kern(runs[a], runs[b])
                + (Decimal(hyper.noise_variance) if a == b else 0)
                for b in range(size)
            ]
            for a in range(size)
        ]
        centred = [Decimal(v) - Decimal(gp.output_mean) for v in gp.values]
        alpha = solve(matrix, centred)
        cross = [[kern(p, r) for r in runs] for p in points]
        solved = [solve(matrix, c) for c in cross]
        mean = [
            float(
                Decimal(gp.output_mean)
                + sum(x * y for x, y in zip(c, alpha, strict=True))
            )
            for c in cross
        ]
        cov = [
            [
                float(
                    kern(points[a], points[b])
                    - sum(x * y for x, y in zip(cross[a], solved[b], strict=True))
                )
                for b in range(len(points))
            ]
            for a in range(len(points))
        ]
        return mean, cov

    def test_emulator_moments_match_the_library_gp_and_dense_oracle(self) -> None:
        """Weight means/variances equal the library GP; covariances the oracle."""
        emulator = decay_emulator()
        points = ((0.7, 1.1), (1.2, 1.1), (0.9, 2.0))
        means, covs = emulator.weight_moments(points)
        for j in range(emulator.basis.components):
            for index, point in enumerate(points):
                mean, variance = emulator.gps[j].predict_with_variance(point)
                self.assertAlmostEqual(
                    means[j][index], mean, delta=1e-9 * (1 + abs(mean))
                )
                self.assertAlmostEqual(
                    covs[j][index][index], variance, delta=1e-9 * (1 + variance)
                )
            ref_mean, ref_cov = self.conditional(emulator, j, points)
            for a in range(3):
                self.assertAlmostEqual(means[j][a], ref_mean[a], delta=1e-8)
                for b in range(3):
                    self.assertAlmostEqual(covs[j][a][b], ref_cov[a][b], delta=1e-8)

    def _oracle(self, shared: bool) -> tuple[float, float]:
        """Return (library, oracle) log likelihoods for two experiments."""
        emulator = decay_emulator()
        controls = ((0.7,), (1.3,))
        data = field_observations(21, biased=True)[:2]
        discrepancy = gaussian_bump_basis(LOCATIONS, BUMPS, BUMP_WIDTH)
        calibration = FunctionalCalibration(
            emulator,
            controls,
            data,
            theta_bounds=[(0.5, 2.5)],
            noise_sd=0.02,
            discrepancy_basis=discrepancy,
            shared_discrepancy=shared,
        )
        theta, sigma_delta = 1.3, 0.05
        phi = (theta, math.log(sigma_delta))
        basis = emulator.basis
        p = basis.components
        count = len(controls)
        points = tuple(x + (theta,) for x in controls)
        mean_vector = []
        cov_blocks = [self.conditional(emulator, j, points) for j in range(p)]
        for e in range(count):
            mean_vector.extend(cov_blocks[j][0][e] for j in range(p))
        # projected data and projection of the discrepancy basis (same grid)
        reduced = [basis.project(row) for row in data]
        proj = [
            [
                sum(basis.modes[j][loc] * discrepancy[loc][k] for loc in range(GRID))
                / basis.scale
                for k in range(len(BUMPS))
            ]
            for j in range(p)
        ]
        size = count * p
        sigma = [[0.0] * size for _ in range(size)]
        for e in range(count):
            for f in range(count):
                for j in range(p):
                    sigma[e * p + j][f * p + j] += cov_blocks[j][1][e][f]
                if e == f or shared:
                    for i in range(p):
                        for j in range(p):
                            sigma[e * p + i][f * p + j] += sigma_delta**2 * sum(
                                proj[i][k] * proj[j][k] for k in range(len(BUMPS))
                            )
            for j in range(p):
                sigma[e * p + j][e * p + j] += (0.02 / basis.scale) ** 2
        residual = [reduced[e][j] for e in range(count) for j in range(p)]
        residual = [a - b for a, b in zip(residual, mean_vector, strict=True)]
        matrix = tuple(tuple(row) for row in sigma)
        quad = sum(
            a * b
            for a, b in zip(
                residual, matvec(inverse_spd(matrix), tuple(residual)), strict=True
            )
        )
        oracle = -0.5 * (size * math.log(2.0 * math.pi) + log_det_spd(matrix) + quad)
        return calibration.log_likelihood(phi), oracle

    def test_log_likelihood_shared_discrepancy(self) -> None:
        """Library value equals the dense formula (shared discrepancy)."""
        library, oracle = self._oracle(True)
        self.assertAlmostEqual(library, oracle, delta=1e-7 * (1 + abs(oracle)))

    def test_log_likelihood_independent_discrepancy(self) -> None:
        """Library value equals the dense formula (independent discrepancies)."""
        library, oracle = self._oracle(False)
        self.assertAlmostEqual(library, oracle, delta=1e-7 * (1 + abs(oracle)))

    def test_prior_support_and_posterior(self) -> None:
        """The posterior is minus infinity outside the box and finite inside."""
        calibration = make_calibration(5, biased=False, discrepancy=True)
        phi_in = (1.4, math.log(0.02))
        self.assertTrue(math.isfinite(calibration.log_posterior(phi_in)))
        self.assertEqual(calibration.log_posterior((9.0, math.log(0.02))), -math.inf)
        self.assertAlmostEqual(
            calibration.log_posterior(phi_in),
            calibration.log_prior(phi_in) + calibration.log_likelihood(phi_in),
            places=12,
        )


class EmulatorValidationTests(unittest.TestCase):
    """Held-out validation of the decay-profile emulator."""

    def test_validation_in_reduced_and_full_space(self) -> None:
        """Measured on 8 held-out runs (see report); bounds are loose on purpose."""
        emulator = decay_emulator()
        controls, parameters = stratified_design(8, 99)
        fields = tuple(
            simulator(x[0], t[0]) for x, t in zip(controls, parameters, strict=True)
        )
        report = emulator.validate(controls, parameters, fields)
        self.assertEqual(report.holdout, 8)
        self.assertLess(report.field_relative_rmse, 0.03)
        self.assertLess(report.truncation_rmse, 0.01 * emulator.basis.scale)
        for value in report.weight_standardized_rmse:
            self.assertLess(value, 2.5)
        self.assertGreaterEqual(min(report.weight_coverage), 0.75)
        self.assertGreaterEqual(report.field_coverage, 0.9)

    def test_forced_truncation_is_reported_as_separate_variance(self) -> None:
        """With one component the truncation variance tracks the held-out error."""
        controls, parameters = stratified_design(16, 1)
        fields = tuple(
            simulator(x[0], t[0]) for x, t in zip(controls, parameters, strict=True)
        )
        emulator = FieldEmulator(
            controls,
            parameters,
            fields,
            locations=LOCATIONS,
            components=1,
            restarts=1,
            seed=1,
        )
        basis = emulator.basis
        self.assertGreater(basis.truncation_fraction, 1e-4)
        held_controls, held_parameters = stratified_design(8, 99)
        held = tuple(
            simulator(x[0], t[0])
            for x, t in zip(held_controls, held_parameters, strict=True)
        )
        report = emulator.validate(held_controls, held_parameters, held)
        mean_truncation = sum(basis.truncation_variance) / GRID
        ratio = report.truncation_rmse**2 / mean_truncation
        self.assertTrue(
            0.3 < ratio < 3.0, f"held-out/training truncation ratio {ratio}"
        )
        prediction = emulator.predict(held_controls, held_parameters)
        for total, emu, trunc in zip(
            prediction.total_variance[0],
            prediction.emulator_variance[0],
            prediction.truncation_variance,
            strict=True,
        ):
            self.assertAlmostEqual(total, emu + trunc, delta=1e-15)

    def test_threshold_rule_selects_components(self) -> None:
        """Components grow with the energy threshold; ``components`` overrides."""
        rng = random.Random(1)
        fields = [
            [math.sin(math.pi * s * (1 + 0.3 * rng.random())) for s in LOCATIONS]
            for _ in range(12)
        ]
        low = fit_field_basis(fields, LOCATIONS, variance_threshold=0.5)
        high = fit_field_basis(fields, LOCATIONS, variance_threshold=0.99999)
        self.assertLessEqual(low.components, high.components)
        self.assertGreaterEqual(low.energy_fractions[low.components - 1], 0.5)
        fixed = fit_field_basis(fields, LOCATIONS, components=3)
        self.assertEqual(fixed.components, 3)
        self.assertAlmostEqual(
            fixed.truncation_fraction, 1.0 - fixed.energy_fractions[2], delta=1e-12
        )


class SyntheticCalibrationTests(unittest.TestCase):
    """Known truth ``theta = 1.4``; seeds 5 and 6 are declared before running."""

    def test_recovery_within_posterior_band(self) -> None:
        """Unbiased data, no discrepancy: truth inside the 95% band, small sd."""
        for seed in (5, 6):
            result = posterior(seed, biased=False, discrepancy=False)
            self.assertEqual(result.stop_reason, "converged")
            low, high = result.theta_interval(0, 0.95)
            self.assertTrue(low <= THETA_TRUE <= high, f"seed {seed}: {low}, {high}")
            self.assertLess(result.sd()[0], 0.02)

    def test_recovery_with_unneeded_discrepancy(self) -> None:
        """Unbiased data with a discrepancy basis: truth covered, sd inflated."""
        plain = posterior(5, biased=False, discrepancy=False)
        flexible = posterior(5, biased=False, discrepancy=True)
        low, high = flexible.theta_interval(0, 0.95)
        self.assertTrue(low <= THETA_TRUE <= high)
        self.assertGreater(flexible.sd()[0], plain.sd()[0])

    def test_bias_demonstration(self) -> None:
        """Structural error: without discrepancy the posterior is wrong and tight."""
        without = posterior(5, biased=True, discrepancy=False)
        with_delta = posterior(5, biased=True, discrepancy=True)
        z_without = abs(without.mean()[0] - THETA_TRUE) / without.sd()[0]
        z_with = abs(with_delta.mean()[0] - THETA_TRUE) / with_delta.sd()[0]
        self.assertGreater(z_without, 5.0)
        self.assertLess(z_with, 2.5)
        low, high = with_delta.theta_interval(0, 0.95)
        self.assertTrue(low <= THETA_TRUE <= high)
        self.assertGreater(with_delta.sd()[0], 1.5 * without.sd()[0])

    def test_field_posterior_is_tighter_than_single_location(self) -> None:
        """Measured gain: the scalar calibration at mid-grid is less informative."""
        calibration = make_calibration(5, biased=False, discrepancy=False)
        result = posterior(5, biased=False, discrepancy=False)
        comparison = calibration.compare_with_scalar(
            result, GRID // 2, seed=3, **SAMPLE
        )
        self.assertIsInstance(comparison, ScalarComparison)
        self.assertGreater(comparison.sd_ratio[0], 2.0)
        self.assertGreater(
            comparison.field_contraction[0], comparison.scalar_contraction[0]
        )
        self.assertAlmostEqual(comparison.field_mean[0], result.mean()[0], places=12)
        self.assertEqual(calibration.contraction(result), comparison.field_contraction)

    def test_predictive_decomposition_and_bands(self) -> None:
        """Variance parts add up; the 95% band covers most of the true field."""
        calibration = make_calibration(5, biased=True, discrepancy=True)
        result = posterior(5, biased=True, discrepancy=True)
        new_controls = ((0.8,), (1.2,))
        prediction = calibration.predict(result, new_controls, max_draws=30)
        self.assertIsInstance(prediction, FieldPrediction)
        sums = prediction.component_sum()
        for n in range(2):
            for total, parts in zip(prediction.total_variance[n], sums[n], strict=True):
                self.assertAlmostEqual(total, parts, delta=1e-9 * total)
        hits = 0
        for n, (x,) in enumerate(new_controls):
            truth = [
                v + bias(s)
                for v, s in zip(simulator(x, THETA_TRUE), LOCATIONS, strict=True)
            ]
            hits += sum(
                low <= v <= high
                for low, v, high in zip(
                    prediction.lower[n], truth, prediction.upper[n], strict=True
                )
            )
        coverage = hits / (2 * GRID)
        self.assertGreaterEqual(coverage, 0.6, f"coverage {coverage}")
        self.assertTrue(
            all(
                lo < m < hi
                for lo, m, hi in zip(
                    prediction.lower[0],
                    prediction.mean[0],
                    prediction.upper[0],
                    strict=True,
                )
            )
        )
        self.assertGreater(max(prediction.discrepancy[0]), 0.0)

    def test_noise_widens_the_observation_band(self) -> None:
        """Predicting a new observation adds exactly the noise variance."""
        calibration = make_calibration(5, biased=False, discrepancy=False)
        result = posterior(5, biased=False, discrepancy=False)
        reality = calibration.predict(
            result, ((0.8,),), include_noise=False, max_draws=10
        )
        observed = calibration.predict(
            result, ((0.8,),), include_noise=True, max_draws=10
        )
        for a, b in zip(
            reality.total_variance[0], observed.total_variance[0], strict=True
        ):
            self.assertAlmostEqual(b - a, NOISE_SD**2, delta=1e-12)
        self.assertEqual(max(reality.noise[0]), 0.0)

    def test_interpolated_field_grid(self) -> None:
        """Field on cell midpoints: basis interpolated, truth still recovered."""
        midpoints = tuple(
            0.5 * (a + b) for a, b in zip(LOCATIONS, LOCATIONS[1:], strict=False)
        )
        data = field_observations(8, biased=False, locations=midpoints)
        calibration = FunctionalCalibration(
            decay_emulator(),
            CONTROLS,
            data,
            theta_bounds=[(0.5, 2.5)],
            noise_sd=NOISE_SD,
            field_locations=midpoints,
        )
        result = calibration.sample(seed=3, **SAMPLE)
        z = abs(result.mean()[0] - THETA_TRUE) / result.sd()[0]
        self.assertLess(z, 3.5, f"z = {z}")
        self.assertEqual(calibration.field_locations, midpoints)

    def test_unknown_noise_sd_is_estimated(self) -> None:
        """Noise sd estimated from the projected data; log-normal prior."""
        calibration = FunctionalCalibration(
            decay_emulator(),
            CONTROLS,
            field_observations(5, biased=False),
            theta_bounds=[(0.5, 2.5)],
            noise_prior=LogNormalPrior(math.log(0.02), 1.0),
        )
        self.assertEqual(calibration.names, ("theta0", "log_noise_sd"))
        result = calibration.sample(seed=4, **SAMPLE)
        sd_noise = math.exp(result.mean()[1])
        self.assertTrue(0.002 < sd_noise < 0.08, f"noise sd {sd_noise}")
        self.assertLess(abs(result.mean()[0] - THETA_TRUE), 0.06)

    def test_tmcmc_agrees_with_slice_sampling(self) -> None:
        """Transitional MCMC: converged, same posterior up to Monte Carlo error."""
        calibration = make_calibration(5, biased=False, discrepancy=False)
        reference = posterior(5, biased=False, discrepancy=False)
        result = calibration.sample(sampler="tmcmc", particles=100, seed=2)
        self.assertTrue(result.converged, result.stop_reason)
        self.assertTrue(math.isfinite(result.log_evidence))
        self.assertLess(
            abs(result.mean()[0] - reference.mean()[0]), 1.5 * reference.sd()[0]
        )
        self.assertTrue(0.5 < result.sd()[0] / reference.sd()[0] < 2.0)

    def test_simulator_without_controls(self) -> None:
        """A parameter-only simulator and a single field experiment."""
        rng = random.Random(4)
        parameters = tuple((0.5 + 2.0 * (i + rng.random()) / 12,) for i in range(12))
        fields = tuple(simulator(1.0, t[0]) for t in parameters)
        emulator = FieldEmulator(
            [()] * 12, parameters, fields, locations=LOCATIONS, restarts=1, seed=1
        )
        obs = tuple(
            v + NOISE_SD * rng.gauss(0.0, 1.0) for v in simulator(1.0, THETA_TRUE)
        )
        calibration = FunctionalCalibration(
            emulator, None, [obs], theta_bounds=[(0.5, 2.5)], noise_sd=NOISE_SD
        )
        result = calibration.sample(seed=2, **SAMPLE)
        self.assertLess(abs(result.mean()[0] - THETA_TRUE), 4.0 * result.sd()[0])
        prediction = calibration.predict(result, None, max_draws=10)
        self.assertEqual(len(prediction.mean), 1)
        with self.assertRaises(ValueError):
            calibration.compare_with_scalar(result, 3)


@unittest.skipUnless(os.environ.get("PROBSTAT_SLOW") == "1", "slow tier")
class CoverageStudyTests(unittest.TestCase):
    """Many-seed study of recovery, bias and predictive coverage."""

    SEEDS = tuple(range(100, 120))

    def test_study(self) -> None:
        """Print and bound the coverage numbers over the declared seeds."""
        cover_plain = cover_flex = cover_biased_none = cover_biased_flex = 0
        sd_ratio: list[float] = []
        band_hits = band_total = 0
        for seed in self.SEEDS:
            plain = posterior(seed, biased=False, discrepancy=False)
            flex_biased = posterior(seed, biased=True, discrepancy=True)
            none_biased = posterior(seed, biased=True, discrepancy=False)
            flex = posterior(seed, biased=False, discrepancy=True)
            cover_plain += self._covers(plain)
            cover_flex += self._covers(flex)
            cover_biased_none += self._covers(none_biased)
            cover_biased_flex += self._covers(flex_biased)
            calibration = make_calibration(seed, biased=False, discrepancy=False)
            sd_ratio.append(
                calibration.compare_with_scalar(
                    plain, GRID // 2, seed=3, **SAMPLE
                ).sd_ratio[0]
            )
            biased_cal = make_calibration(seed, biased=True, discrepancy=True)
            prediction = biased_cal.predict(flex_biased, ((0.8,), (1.2,)), max_draws=30)
            for n, (x,) in enumerate(((0.8,), (1.2,))):
                truth = [
                    v + bias(s)
                    for v, s in zip(simulator(x, THETA_TRUE), LOCATIONS, strict=True)
                ]
                band_hits += sum(
                    lo <= v <= hi
                    for lo, v, hi in zip(
                        prediction.lower[n], truth, prediction.upper[n], strict=True
                    )
                )
                band_total += GRID
        count = len(self.SEEDS)
        print(
            f"\nseeds={count} theta-cover(95%): plain={cover_plain / count:.2f} "
            f"unneeded-delta={cover_flex / count:.2f} "
            f"biased/no-delta={cover_biased_none / count:.2f} "
            f"biased/delta={cover_biased_flex / count:.2f}; "
            f"scalar/field sd ratio min={min(sd_ratio):.2f} "
            f"median={sorted(sd_ratio)[count // 2]:.2f}; "
            f"95% field-band coverage of reality={band_hits / band_total:.3f}"
        )
        self.assertGreaterEqual(cover_plain / count, 0.8)
        self.assertGreaterEqual(cover_biased_flex / count, 0.8)
        self.assertLessEqual(cover_biased_none / count, 0.2)
        self.assertGreater(min(sd_ratio), 1.5)
        self.assertGreaterEqual(band_hits / band_total, 0.7)

    @staticmethod
    def _covers(result: KOHResult) -> int:
        """Return 1 if the 95% interval contains the true parameter."""
        low, high = result.theta_interval(0, 0.95)
        return int(low <= THETA_TRUE <= high)


class FailureCaseTests(unittest.TestCase):
    """Invalid input is rejected; numerical failure is reported, not hidden."""

    def test_basis_input_validation(self) -> None:
        """Bad shapes, thresholds, grids and identical runs."""
        good = [[0.0, 1.0, 2.0], [0.5, 1.2, 2.1], [0.1, 0.9, 2.5]]
        with self.assertRaises(ValueError):
            fit_field_basis([[1.0, 2.0]])
        with self.assertRaises(ValueError):
            fit_field_basis([[1.0, 2.0], [1.0]])
        with self.assertRaises(ValueError):
            fit_field_basis([[1.0, 2.0, 3.0]] * 3)
        with self.assertRaises(ValueError):
            fit_field_basis(good, [0.0, 0.0, 1.0])
        with self.assertRaises(ValueError):
            fit_field_basis(good, [0.0, 1.0])
        with self.assertRaises(ValueError):
            fit_field_basis(good, variance_threshold=0.0)
        with self.assertRaises(ValueError):
            fit_field_basis(good, components=0)
        with self.assertRaises(ValueError):
            fit_field_basis(good, components=3)
        with self.assertRaises(ValueError):
            fit_field_basis([[1.0, math.nan], [1.0, 2.0]])
        with self.assertRaises(ValueError):
            gaussian_bump_basis(LOCATIONS, BUMPS, 0.0)

    def test_emulator_validation_of_input(self) -> None:
        """Shape and kernel errors."""
        controls, parameters = stratified_design(8, 3)
        fields = tuple(
            simulator(x[0], t[0]) for x, t in zip(controls, parameters, strict=True)
        )
        with self.assertRaises(ValueError):
            FieldEmulator(controls[:7], parameters, fields, locations=LOCATIONS)
        with self.assertRaises(ValueError):
            FieldEmulator(
                controls, parameters, fields, locations=LOCATIONS, kernel="rbf"
            )
        emulator = decay_emulator()
        with self.assertRaises(ValueError):
            emulator.predict([(0.5, 0.5)], [(1.0,)])
        with self.assertRaises(ValueError):
            emulator.validate(*stratified_design(4, 1), [[0.0] * (GRID - 1)] * 4)
        with self.assertRaises(ValueError):
            emulator.validate(
                *stratified_design(2, 1), [simulator(1.0, 1.0)] * 2, level=1.0
            )

    def test_calibration_input_validation(self) -> None:
        """Noise, bounds, grids, types and bases."""
        emulator = decay_emulator()
        data = field_observations(5, biased=False)
        common = {"theta_bounds": [(0.5, 2.5)]}
        with self.assertRaises(TypeError):
            FunctionalCalibration(object(), CONTROLS, data, noise_sd=0.01, **common)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            FunctionalCalibration(emulator, CONTROLS, data, **common)
        with self.assertRaises(ValueError):
            FunctionalCalibration(
                emulator,
                CONTROLS,
                data,
                noise_sd=0.01,
                noise_prior=LogNormalPrior(0.0, 1.0),
                **common,
            )
        with self.assertRaises(ValueError):
            FunctionalCalibration(emulator, CONTROLS, data, noise_sd=-1.0, **common)
        with self.assertRaises(ValueError):
            FunctionalCalibration(emulator, CONTROLS[:2], data, noise_sd=0.01, **common)
        with self.assertRaises(ValueError):
            FunctionalCalibration(emulator, None, data, noise_sd=0.01, **common)
        with self.assertRaises(ValueError):
            FunctionalCalibration(
                emulator, CONTROLS, data, noise_sd=0.01, theta_bounds=[(2.5, 0.5)]
            )
        with self.assertRaises(ValueError):
            FunctionalCalibration(
                emulator,
                CONTROLS,
                data,
                noise_sd=0.01,
                theta_bounds=[(0.5, 2.5), (0.0, 1.0)],
            )
        with self.assertRaises(ValueError):
            FunctionalCalibration(
                emulator, CONTROLS, [row[:-1] for row in data], noise_sd=0.01, **common
            )
        with self.assertRaises(ValueError):
            FunctionalCalibration(
                emulator,
                CONTROLS,
                [row[:3] for row in data],
                noise_sd=0.01,
                field_locations=(0.0, 0.5, 1.2),
                **common,
            )
        with self.assertRaises(ValueError):
            FunctionalCalibration(
                emulator,
                CONTROLS,
                data,
                noise_sd=0.01,
                discrepancy_basis=[[1.0]] * (GRID - 1),
                **common,
            )

    def test_rank_deficient_field_grid(self) -> None:
        """Fewer field locations than components cannot be projected."""
        emulator = decay_emulator()
        data = [row[:2] for row in field_observations(5, biased=False)]
        with self.assertRaises(ValueError):
            FunctionalCalibration(
                emulator,
                CONTROLS,
                data,
                theta_bounds=[(0.5, 2.5)],
                noise_sd=0.01,
                field_locations=LOCATIONS[:2],
            )

    def test_sampler_and_prediction_options(self) -> None:
        """Bad sampler, one chain, foreign result, bad level or draw count."""
        calibration = make_calibration(5, biased=False, discrepancy=False)
        result = posterior(5, biased=False, discrepancy=False)
        with self.assertRaises(ValueError):
            calibration.sample(sampler="gibbs")
        with self.assertRaises(ValueError):
            calibration.sample(chains=1)
        foreign = posterior(5, biased=True, discrepancy=True)
        with self.assertRaises(ValueError):
            calibration.predict(foreign, ((0.8,),))
        with self.assertRaises(ValueError):
            calibration.contraction(foreign)
        with self.assertRaises(ValueError):
            calibration.predict(result, ((0.8,),), level=1.5)
        with self.assertRaises(ValueError):
            calibration.predict(result, ((0.8,),), max_draws=0)
        with self.assertRaises(ValueError):
            calibration.predict(result, None)
        with self.assertRaises(ValueError):
            calibration.predict(result, ((0.8, 0.9),))
        with self.assertRaises(ValueError):
            calibration.compare_with_scalar(result, GRID)
        with self.assertRaises(ValueError):
            calibration.log_likelihood((1.0, 2.0))

    def test_non_positive_definite_covariance_is_not_hidden(self) -> None:
        """Replicate experiments with underflowing noise: reported, not jittered.

        The single likelihood call raises; the sampler counts the rejected
        evaluations (measured: 2621 of 2713) and does not claim convergence.
        """
        emulator = decay_emulator()
        row = simulator(1.0, THETA_TRUE)
        calibration = FunctionalCalibration(
            emulator,
            ((1.0,), (1.0,)),
            [row, row],
            theta_bounds=[(0.5, 2.5)],
            noise_sd=1e-200,
        )
        with self.assertRaises(ValueError):
            calibration.log_likelihood((1.4,))
        result = calibration.sample(seed=1, draws=20, warmup=10)
        self.assertGreater(result.failed_evaluations, 0.5 * result.evaluations)
        self.assertFalse(result.converged)


if __name__ == "__main__":
    unittest.main()
