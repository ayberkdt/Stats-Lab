"""Tests for the synthetic expansion-history benchmark A02 (roadmap 20B).

Everything here is SYNTHETIC: no survey data and no claim about the real
universe. Order: closed forms (constant-w factor, LCDM, low-z limit), the
independent Simpson oracle for the distance integral and the closed-form CPL
factor as oracle for the finer truth generator, then derivative checks, the
nuisance degeneracy and the spectrum, one solver smoke, and invalid inputs.
"""

from __future__ import annotations

import itertools
import math
import unittest
from collections.abc import Callable

from probstat.inverse import (
    check_derivatives,
    gauss_newton_krylov,
    spectral_diagnostics,
)
from probstat.inverse.benchmark_cosmology import (
    BaoDistanceModel,
    CalibrationNoise,
    ExpansionHistoryModel,
    GreedyDesign,
    JointExpansionModel,
    bao_degeneracy_report,
    correlated_calibration_benchmark,
    default_prior_sd,
    greedy_redshift_design,
    interval_calibration,
    jacobian_columns,
    linear_gaussian_eig,
    nominal_expansion_point,
    synthetic_expansion_history_benchmark,
    whitened_spectrum,
)
from probstat.linalg import (
    DiagonalCovariance,
    ScaledIdentityCovariance,
    inverse_spd,
    log_det_spd,
    singular_value_decomposition,
)

C = 299792.458


def _uniform_edges(top: float, k: int) -> list[float]:
    """Return the edges of ``k`` uniform bins on ``[0, top]``."""
    return [top * j / k for j in range(k + 1)]


def _simpson(f: Callable[[float], float], a: float, b: float, n: int = 4000) -> float:
    """Return the composite Simpson integral of ``f`` (independent oracle)."""
    h = (b - a) / n
    total = f(a) + f(b)
    for i in range(1, n):
        total += (4 if i % 2 else 2) * f(a + i * h)
    return total * h / 3.0


def _jacobian(model: ExpansionHistoryModel, x: tuple[float, ...]) -> list[list[float]]:
    """Return the dense Jacobian assembled from unit-vector JVPs."""
    n, m = model.input_dimension, model.output_dimension
    columns = []
    for j in range(n):
        e = [0.0] * n
        e[j] = 1.0
        columns.append(model.jvp(x, tuple(e)))
    return [[columns[j][i] for j in range(n)] for i in range(m)]


class ClosedFormTests(unittest.TestCase):
    """Exact identities of the background model."""

    def test_constant_w_factor(self) -> None:
        """Constant ``w`` gives ``X(z) = (1 + z)^(3 (1 + w))`` exactly."""
        model = ExpansionHistoryModel(
            [0.5, 1.0], "piecewise_constant", [0, 0.4, 1.0, 2.0]
        )
        zs = (0.1, 0.4, 0.7, 1.0, 1.6, 2.0)
        for w in (-0.7, -1.0, -1.3):
            x = (1.0 + w,) * 3
            for z, got in zip(zs, model.dark_energy_factor(x, zs), strict=True):
                ref = (1.0 + z) ** (3.0 * (1.0 + w))
                self.assertAlmostEqual(got / ref, 1.0, places=13)
            self.assertEqual(model.equation_of_state(x, (0.2, 1.5)), (w, w))

    def test_lcdm_expansion_rate(self) -> None:
        """``w = -1`` gives ``E^2 = Om (1+z)^3 + 1 - Om``."""
        model = ExpansionHistoryModel(
            [0.5], "piecewise_constant", [0, 1.0], omega_m=0.27
        )
        zs = (0.0, 0.3, 1.0, 3.0)
        for z, e in zip(zs, model.expansion_rate((0.0,), zs), strict=True):
            self.assertAlmostEqual(e * e, 0.27 * (1 + z) ** 3 + 0.73, places=13)

    def test_hubble_observable(self) -> None:
        """The Hubble observable is ``H0 E(z)``."""
        model = ExpansionHistoryModel(
            [0.2, 0.9], "piecewise_constant", [0, 1.0], h0=67.0, observable="hubble"
        )
        for z, value in zip((0.2, 0.9), model.value((0.0,)), strict=True):
            ref = 67.0 * math.sqrt(0.3 * (1 + z) ** 3 + 0.7)
            self.assertAlmostEqual(value, ref, places=10)

    def test_low_redshift_limit(self) -> None:
        """``D_L ~ c z / H0`` at ``z = 0.01`` (relative error below 1e-2)."""
        model = ExpansionHistoryModel([0.01], "piecewise_constant", [0, 1.0])
        mu = model.value((0.0,))[0]
        d_l = 10.0 ** ((mu - 25.0) / 5.0)
        rel = abs(d_l - C * 0.01 / 70.0) / (C * 0.01 / 70.0)
        self.assertLess(rel, 1e-2)
        # first-order correction (1 - q0) z / 2 with q0 = Om / 2 - (1 - Om)
        self.assertAlmostEqual(rel, 0.5 * (1 + 0.55) * 0.01, delta=2e-4)

    def test_distance_matches_independent_simpson(self) -> None:
        """The discrete ``D_C`` matches Simpson on a fine grid (kinks at edges)."""
        edges = [0, 0.5, 1.0, 1.5]
        x = (0.3, -0.2, 0.5)
        zs = [0.2, 0.7, 1.2, 1.5]
        model = ExpansionHistoryModel(zs, "piecewise_constant", edges)

        def inv_e(t: float) -> float:
            """Return ``1/E`` from the exact piecewise exponent integral."""
            s = 0.0
            for j in range(3):
                a, b = edges[j], edges[j + 1]
                if t > a:
                    s += x[j] * math.log((1 + min(t, b)) / (1 + a))
            return 1.0 / math.sqrt(0.3 * (1 + t) ** 3 + 0.7 * math.exp(3 * s))

        for z, d in zip(zs, model.comoving_distance(x), strict=True):
            cuts = [0.0] + [e for e in edges[1:] if e < z] + [z]
            ref = sum(
                _simpson(inv_e, a, b, 2000)
                for a, b in zip(cuts, cuts[1:], strict=False)
            )
            self.assertAlmostEqual(d / (ref * C / 70.0), 1.0, places=11)

    def test_quadrature_converges_with_panels(self) -> None:
        """The error against Simpson drops as panels are added, to roundoff."""

        def inv_e(t: float) -> float:
            """Return ``1/E`` for LCDM with ``Om = 0.3``."""
            return 1.0 / math.sqrt(0.3 * (1 + t) ** 3 + 0.7)

        ref = _simpson(inv_e, 0.0, 1.5, 20000) * C / 70.0
        errors = []
        for p in (1, 2, 4):
            model = ExpansionHistoryModel(
                [1.5], "piecewise_constant", [0, 1.5], panels=p
            )
            errors.append(abs(model.comoving_distance((0.0,))[0] - ref) / ref)
        self.assertGreater(errors[0], errors[1])
        self.assertGreater(errors[0], 1e-11)
        self.assertLess(errors[2], 1e-12)


class DerivativeTests(unittest.TestCase):
    """Analytic JVP/VJP against central differences and the dot test."""

    def _model(
        self, observable: str
    ) -> tuple[ExpansionHistoryModel, tuple[float, ...]]:
        """Return a model with free nuisances and a generic base point."""
        if observable == "distance_modulus":
            nuisance = ("omega_m", "log10_h0", "offset")
            x = (0.3, -0.2, 0.4, 0.1, 0.28, 1.85, 0.2)
        else:
            nuisance = ("omega_m", "log10_h0")
            x = (0.3, -0.2, 0.4, 0.1, 0.28, 1.85)
        zs = [0.05 + 1.45 * i / 11 for i in range(12)]
        model = ExpansionHistoryModel(
            zs,
            "piecewise_constant",
            _uniform_edges(1.2, 4),
            observable=observable,
            nuisance=nuisance,
        )
        return model, x

    def test_jvp_matches_central_differences(self) -> None:
        """Every Jacobian column agrees with a central difference."""
        for observable in ("distance_modulus", "hubble"):
            model, x = self._model(observable)
            jac = _jacobian(model, x)
            for j in range(model.input_dimension):
                h = 1e-6
                up, dn = list(x), list(x)
                up[j] += h
                dn[j] -= h
                fp, fm = model.value(up), model.value(dn)
                for i in range(model.output_dimension):
                    fd = (fp[i] - fm[i]) / (2 * h)
                    self.assertAlmostEqual(jac[i][j], fd, delta=2e-6 * (1 + abs(fd)))

    def test_taylor_and_dot_test(self) -> None:
        """Taylor sweep passes and the adjoint dot test holds to 1e-12."""
        for observable in ("distance_modulus", "hubble"):
            model, x = self._model(observable)
            n, m = model.input_dimension, model.output_dimension
            v = tuple(0.1 * math.sin(1 + j) for j in range(n))
            w = tuple(math.cos(0.5 * i) for i in range(m))
            result = check_derivatives(model, x, v, w, dot_tolerance=1e-12)
            self.assertTrue(result.taylor_passed)
            self.assertLess(result.dot_relative_error, 1e-12)

    def test_dense_transpose_consistency(self) -> None:
        """``vjp`` equals the transpose of the JVP-assembled Jacobian."""
        model, x = self._model("distance_modulus")
        jac = _jacobian(model, x)
        w = tuple(1.0 + 0.1 * i for i in range(model.output_dimension))
        got = model.vjp(x, w)
        for j in range(model.input_dimension):
            ref = sum(jac[i][j] * w[i] for i in range(model.output_dimension))
            self.assertAlmostEqual(got[j], ref, delta=1e-12 * (1 + abs(ref)))


class DegeneracyTests(unittest.TestCase):
    """Offset versus ``H0`` degeneracy, spectrum and the benchmark."""

    def test_offset_and_h0_columns_collinear(self) -> None:
        """The offset column is all ones and the ``log10 H0`` column all ``-5``."""
        zs = [0.1 + 0.05 * i for i in range(15)]
        model = ExpansionHistoryModel(
            zs,
            "piecewise_constant",
            _uniform_edges(0.9, 3),
            nuisance=("log10_h0", "offset"),
        )
        x = (0.1, 0.0, -0.2, math.log10(70.0), 0.0)
        jac = _jacobian(model, x)
        for row in jac:
            self.assertEqual(row[4], 1.0)
            self.assertEqual(row[3], -5.0)
        sv = singular_value_decomposition(jac).singular_values
        self.assertLess(sv[-1] / sv[0], 1e-12)

    def test_whitened_spectrum_decays(self) -> None:
        """Eight bins, 30 SN-like points, ``sigma_mu = 0.1``: few modes resolved.

        Measured on the whitened Jacobian at LCDM: the singular values are
        about 12.5, 2.33, 0.73, 0.28, 0.12, 0.054, 0.023, 0.0068, so exactly 2
        exceed 1 in whitened units.
        """
        zs = [0.05 + 1.45 * i / 29 for i in range(30)]
        model = ExpansionHistoryModel(zs, "piecewise_constant", _uniform_edges(1.5, 8))
        jac = _jacobian(model, (0.0,) * 8)
        diag = spectral_diagnostics(jac, noise=ScaledIdentityCovariance(30, 0.01))
        count = sum(1 for s in diag.singular_values if s > 1.0)
        self.assertEqual(count, 2)
        self.assertLess(count, 8)
        self.assertGreater(diag.singular_values[0] / diag.singular_values[-1], 1e3)

    def test_cpl_truth_generator_against_closed_form(self) -> None:
        """The finer-grid truth data match the closed-form CPL factor."""
        case = synthetic_expansion_history_benchmark(truth="cpl", sigma_mu=0.1)
        self.assertFalse(case.same_discretization)
        w0, wa = -0.9, -0.4

        def inv_e(t: float) -> float:
            """Return ``1/E`` with ``X = (1+t)^(3(1+w0+wa)) exp(-3 wa t/(1+t))``."""
            x = (1 + t) ** (3 * (1 + w0 + wa)) * math.exp(-3 * wa * t / (1 + t))
            return 1.0 / math.sqrt(0.3 * (1 + t) ** 3 + 0.7 * x)

        for z, b in zip(case.model.redshifts, case.exact_data, strict=True):
            d_c = C / 70.0 * _simpson(inv_e, 0.0, z, 2000)
            self.assertAlmostEqual(b, 5 * math.log10((1 + z) * d_c) + 25, places=9)
        self.assertIn("weakly_identified_high_z", case.pathologies)

    def test_lcdm_flag_and_seeds(self) -> None:
        """LCDM data come from the model itself; the seed fixes the noise."""
        a = synthetic_expansion_history_benchmark(seed=5)
        b = synthetic_expansion_history_benchmark(seed=5)
        c = synthetic_expansion_history_benchmark(seed=6)
        self.assertTrue(a.same_discretization)
        self.assertEqual(a.data, b.data)
        self.assertNotEqual(a.data, c.data)
        self.assertEqual(a.exact_data, c.exact_data)

    def test_tikhonov_recovers_lcdm(self) -> None:
        """Gauss-Newton with a Tikhonov prior recovers ``x = 0`` within Fisher bands."""
        sigma, prior_sd, n = 0.02, 0.5, 8
        case = synthetic_expansion_history_benchmark(sigma_mu=sigma, seed=3)
        result = gauss_newton_krylov(
            case.model,
            case.data,
            noise=case.noise,
            regularization=1.0 / prior_sd,
            initial=(0.3,) * n,
            max_iterations=30,
        )
        self.assertTrue(result.converged, result.stop_reason)
        jac = _jacobian(case.model, case.truth)
        info = [
            [
                sum(r[i] * r[j] for r in jac) / sigma**2
                + (1 / prior_sd**2 if i == j else 0.0)
                for j in range(n)
            ]
            for i in range(n)
        ]
        cov = inverse_spd(info)
        for j in range(n):
            band = 5.0 * math.sqrt(cov[j][j])
            self.assertLess(abs(result.estimate[j]), band)
        # best-determined bin is tight; high-z bins are prior-limited
        self.assertLess(math.sqrt(cov[0][0]), 0.3 * prior_sd)


class InvalidInputTests(unittest.TestCase):
    """Rejected arguments."""

    def test_invalid_design(self) -> None:
        """Non-positive redshifts, bad ``Om``, bad edges and options raise."""
        edges = [0, 0.5, 1.0]
        pc = "piecewise_constant"
        with self.assertRaises(ValueError):
            ExpansionHistoryModel([0.0, 0.5], pc, edges)
        with self.assertRaises(ValueError):
            ExpansionHistoryModel([-0.1], pc, edges)
        for om in (0.0, 1.0, 1.2, -0.1):
            with self.assertRaises(ValueError):
                ExpansionHistoryModel([0.5], pc, edges, omega_m=om)
        with self.assertRaises(ValueError):
            ExpansionHistoryModel([0.5], pc, [0, 1.0, 0.5])
        with self.assertRaises(ValueError):
            ExpansionHistoryModel([0.5], pc, [0.1, 1.0])
        with self.assertRaises(ValueError):
            ExpansionHistoryModel([0.5], "spline", edges)
        with self.assertRaises(ValueError):
            ExpansionHistoryModel([0.5], pc, edges, observable="x")
        with self.assertRaises(ValueError):
            ExpansionHistoryModel([0.5], pc, edges, nuisance=("offset", "offset"))
        with self.assertRaises(ValueError):
            ExpansionHistoryModel(
                [0.5], pc, edges, observable="hubble", nuisance=("offset",)
            )
        with self.assertRaises(ValueError):
            ExpansionHistoryModel([0.5], pc, edges, h0=0.0)
        with self.assertRaises(TypeError):
            ExpansionHistoryModel("abc", pc, edges)  # type: ignore[arg-type]

    def test_invalid_evaluation_and_benchmark(self) -> None:
        """Wrong lengths raise, a failed trial point is nan, bad cases raise."""
        model = ExpansionHistoryModel(
            [0.5], "piecewise_constant", [0, 1.0], nuisance=("omega_m",)
        )
        with self.assertRaises(ValueError):
            model.value((0.0,))
        self.assertTrue(math.isnan(model.value((0.0, 1.5))[0]))
        with self.assertRaises(ValueError):
            model.jvp((0.0, 1.5), (1.0, 0.0))
        with self.assertRaises(ValueError):
            synthetic_expansion_history_benchmark(truth="phantom")
        with self.assertRaises(ValueError):
            synthetic_expansion_history_benchmark(sigma_mu=0.1, relative_noise=0.01)
        with self.assertRaises(ValueError):
            synthetic_expansion_history_benchmark(bins=[0.0, 0.5, 1.0])


# ---------------------------------------------------------------------------
# Roadmap 20B additions (SYNTHETIC): correlated noise, BAO, information gain
# ---------------------------------------------------------------------------

_EDGES4 = [1.5 * j / 4 for j in range(5)]


class _LinearStub:
    """A linear model ``y = J x`` given by its columns (test double)."""

    def __init__(self, columns: list[tuple[float, ...]]) -> None:
        """Store the columns."""
        self.columns = columns
        self.parameter_names = tuple(f"x{j + 1}" for j in range(len(columns)))
        self.input_dimension = len(columns)
        self.output_dimension = len(columns[0])

    def jvp(self, x: object, v: tuple[float, ...]) -> tuple[float, ...]:
        """Return ``J v``."""
        return tuple(
            sum(self.columns[j][i] * v[j] for j in range(self.input_dimension))
            for i in range(self.output_dimension)
        )


class CalibrationNoiseTests(unittest.TestCase):
    """Correlated calibration noise handled through covariance factors."""

    def test_matrix_formula_and_spd(self) -> None:
        """``Gamma = w^2 I + o^2 11' + s^2 K`` entrywise, and it is SPD."""
        cal = CalibrationNoise(0.1, offset_sd=0.2, smooth_sd=0.3, length_scale=0.5)
        zs = (0.1, 0.4, 1.0)
        g = cal.matrix(zs)
        for i, a in enumerate(zs):
            for j, b in enumerate(zs):
                ref = 0.04 + 0.09 * math.exp(-((a - b) ** 2) / 0.5)
                ref += 0.01 if i == j else 0.0
                self.assertAlmostEqual(g[i][j], ref, places=14)
        factor = cal.covariance(zs)
        # unwhiten then whiten is the identity; the factor squares to Gamma
        z = (0.3, -1.2, 0.7)
        back = factor.whiten(factor.unwhiten(z))
        for a, b in zip(back, z, strict=True):
            self.assertAlmostEqual(a, b, places=12)
        applied = factor.apply((1.0, 0.0, 0.0))
        for i in range(3):
            self.assertAlmostEqual(applied[i], g[i][0], places=12)

    def test_common_offset_whitened_norm_closed_form(self) -> None:
        """``1' Gamma^-1 1 = m / (w^2 + m o^2)``: the offset column's whitened norm."""
        m, w, o = 12, 0.1, 0.4
        zs = [0.1 + 0.1 * i for i in range(m)]
        cal = CalibrationNoise(w, offset_sd=o)
        model = ExpansionHistoryModel(
            zs,
            "piecewise_constant",
            _EDGES4,
            nuisance=("offset",),
        )
        x0 = nominal_expansion_point(model.parameter_names)
        spec = whitened_spectrum(model, x0, cal.covariance(zs), parameters=("offset",))
        self.assertAlmostEqual(
            spec.singular_values[0], math.sqrt(m / (w * w + m * o * o)), places=10
        )
        white = whitened_spectrum(
            model, x0, CalibrationNoise(w).covariance(zs), parameters=("offset",)
        )
        self.assertAlmostEqual(white.singular_values[0], math.sqrt(m) / w, places=10)
        self.assertLess(spec.singular_values[0], 0.3 * white.singular_values[0])

    def test_benchmark_noise_statistics(self) -> None:
        """Over 300 seeds the error covariance matches ``Gamma`` within MCSE."""
        cal = CalibrationNoise(0.05, offset_sd=0.08, smooth_sd=0.06, length_scale=0.4)
        n_rep = 300
        errors = []
        for seed in range(n_rep):
            case = correlated_calibration_benchmark(cal, seed=seed, count=5)
            errors.append(
                [d - b for d, b in zip(case.data, case.exact_data, strict=True)]
            )
        gamma = cal.matrix(case.model.redshifts)
        for i, j in ((0, 0), (2, 2), (0, 1), (1, 4), (3, 3)):
            est = sum(e[i] * e[j] for e in errors) / n_rep
            # var of a product of Gaussians: (G_ii G_jj + G_ij^2) / n
            mcse = math.sqrt((gamma[i][i] * gamma[j][j] + gamma[i][j] ** 2) / n_rep)
            self.assertLess(abs(est - gamma[i][j]), 4.0 * mcse, msg=str((i, j)))

    def test_benchmark_structure(self) -> None:
        """Data are exact + a draw from the factor; noise objects are exposed."""
        cal = CalibrationNoise(0.05, offset_sd=0.05, smooth_sd=0.05)
        a = correlated_calibration_benchmark(cal, truth="cpl", seed=3, count=10)
        b = correlated_calibration_benchmark(cal, truth="cpl", seed=3, count=10)
        c = correlated_calibration_benchmark(cal, truth="cpl", seed=4, count=10)
        self.assertEqual(a.data, b.data)
        self.assertNotEqual(a.data, c.data)
        self.assertEqual(a.exact_data, c.exact_data)
        self.assertEqual(a.identifier, "A02c")
        self.assertIn("correlated_noise", a.pathologies)
        self.assertFalse(a.same_discretization)
        self.assertEqual(a.noise.dimension, 10)
        diag = a.naive_noise.apply((1.0,) * 10)
        full = cal.matrix(a.model.redshifts)
        for i in range(10):
            self.assertAlmostEqual(diag[i], full[i][i], places=14)
        err = math.hypot(*(d - e for d, e in zip(a.data, a.exact_data, strict=True)))
        self.assertAlmostEqual(err, a.realized_error_norm, places=12)

    def test_interval_calibration_identity_and_closed_form(self) -> None:
        """Equal noise gives ratio 1; a scalar example matches the formula."""
        stub = _LinearStub([(1.0, 2.0, -1.0)])
        true = CalibrationNoise(0.5, offset_sd=0.7).covariance([0.1, 0.2, 0.3])
        same = interval_calibration(stub, (0.0,), true, true, (0.8,))
        self.assertAlmostEqual(same.ratio[0], 1.0, places=12)
        assumed = DiagonalCovariance([0.25, 0.25, 0.25])
        out = interval_calibration(stub, (0.0,), true, assumed, (0.8,))
        j = (1.0, 2.0, -1.0)
        p = 0.8**2
        info = sum(v * v for v in j) / 0.25 + 1.0 / p
        cov = 1.0 / info
        k = [cov * v / 0.25 for v in j]
        gt = true.apply(tuple(k))
        kj = sum(a * b for a, b in zip(k, j, strict=True))
        actual = (1.0 - kj) ** 2 * p + sum(a * b for a, b in zip(k, gt, strict=True))
        self.assertAlmostEqual(out.nominal_sd[0], math.sqrt(cov), places=12)
        self.assertAlmostEqual(out.actual_sd[0], math.sqrt(actual), places=12)
        self.assertGreater(out.ratio[0], 1.0)

    def test_ignoring_correlation_misstates_intervals(self) -> None:
        """Measured: the diagonal surrogate is 1.82 times too narrow for bin 1."""
        cal = CalibrationNoise(0.05, offset_sd=0.05, smooth_sd=0.05, length_scale=0.4)
        case = correlated_calibration_benchmark(
            cal, nuisance=("omega_m",), count=20, bins=6
        )
        sds = default_prior_sd(case.model.parameter_names)
        out = interval_calibration(
            case.model, case.truth, case.noise, case.naive_noise, sds
        )
        self.assertGreater(out.ratio[0], 1.5)
        self.assertGreater(max(out.ratio), 1.5)
        # prior-dominated high-z bins are not misstated much
        self.assertLess(abs(out.ratio[4] - 1.0), 0.05)


class BaoModelTests(unittest.TestCase):
    """The simplified BAO observable ``D_V / r_s``."""

    NUIS = ("omega_m", "log10_h0", "log10_rs")
    X = (0.1, -0.2, 0.28, 1.85, 2.17)
    ZS = [0.2, 0.5, 1.0]

    def _model(self) -> BaoDistanceModel:
        """Return a three-point BAO model with three nuisances."""
        return BaoDistanceModel(
            self.ZS, "piecewise_constant", [0, 0.5, 1.0], nuisance=self.NUIS
        )

    def test_value_matches_closed_form_composition(self) -> None:
        """``D_V = (z D_C^2 c / H)^(1/3)`` from the independent distance model."""
        model = self._model()
        inner = ExpansionHistoryModel(
            self.ZS,
            "piecewise_constant",
            [0, 0.5, 1.0],
            nuisance=("omega_m", "log10_h0"),
        )
        d_c = inner.comoving_distance(self.X[:4])
        e = inner.expansion_rate(self.X[:4], self.ZS)
        h0 = 10.0 ** self.X[3]
        rs = 10.0 ** self.X[4]
        for z, dc, ez, got in zip(self.ZS, d_c, e, model.value(self.X), strict=True):
            d_v = (z * dc * dc * C / (h0 * ez)) ** (1.0 / 3.0)
            self.assertAlmostEqual(got * rs / d_v, 1.0, places=12)

    def test_lcdm_independent_simpson(self) -> None:
        """For LCDM the value agrees with a Simpson oracle of the distance."""
        model = BaoDistanceModel(
            [0.5], "piecewise_constant", [0, 1.0], rs=150.0, nuisance=()
        )
        z, om = 0.5, 0.3
        d_c = (
            C
            / 70.0
            * _simpson(lambda t: 1 / math.sqrt(om * (1 + t) ** 3 + 1 - om), 0.0, z)
        )
        e = math.sqrt(om * (1 + z) ** 3 + 1 - om)
        d_v = (z * d_c**2 * C / (70.0 * e)) ** (1.0 / 3.0)
        self.assertAlmostEqual(model.value((0.0,))[0], d_v / 150.0, places=10)
        self.assertAlmostEqual(model.volume_distance((0.0,))[0], d_v, delta=1e-8 * d_v)

    def test_derivatives(self) -> None:
        """Central differences, Taylor sweep and the adjoint dot test."""
        model = self._model()
        cols = jacobian_columns(model, self.X)
        for j in range(5):
            h = 1e-6
            up, dn = list(self.X), list(self.X)
            up[j] += h
            dn[j] -= h
            fp, fm = model.value(up), model.value(dn)
            for i in range(3):
                fd = (fp[i] - fm[i]) / (2 * h)
                self.assertAlmostEqual(cols[j][i], fd, delta=2e-6 * (1 + abs(fd)))
        check = check_derivatives(
            model, self.X, (0.3, -0.1, 0.02, 0.01, -0.03), (1.0, -2.0, 0.5)
        )
        self.assertTrue(check.passed)
        self.assertLess(check.dot_relative_error, 1e-12)

    def test_h0_rs_exact_degeneracy(self) -> None:
        """``y`` depends on ``H0 r_s`` only: the two columns coincide exactly."""
        model = self._model()
        cols = jacobian_columns(model, self.X)
        for a, b in zip(cols[3], cols[4], strict=True):
            self.assertAlmostEqual(a / b, 1.0, places=13)
        moved = (*self.X[:3], self.X[3] + 0.07, self.X[4] - 0.07)
        for a, b in zip(model.value(moved), model.value(self.X), strict=True):
            self.assertAlmostEqual(a / b, 1.0, places=12)

    def test_invalid(self) -> None:
        """Unknown nuisances, wrong lengths and failed points."""
        with self.assertRaises(ValueError):
            BaoDistanceModel(
                [0.5], "piecewise_constant", [0, 1.0], nuisance=("offset",)
            )
        with self.assertRaises(ValueError):
            BaoDistanceModel(
                [0.5], "piecewise_constant", [0, 1.0], nuisance=("omega_m", "omega_m")
            )
        with self.assertRaises(ValueError):
            BaoDistanceModel([0.5], "piecewise_constant", [0, 1.0], rs=0.0)
        model = self._model()
        with self.assertRaises(ValueError):
            model.value((0.0, 0.0))
        with self.assertRaises(ValueError):
            model.vjp(self.X, (1.0, 2.0))
        self.assertTrue(math.isnan(model.value((0.1, -0.2, 1.5, 1.85, 2.17))[0]))


class JointAndDegeneracyTests(unittest.TestCase):
    """Which nuisance degeneracies BAO breaks and which it does not (measured)."""

    def test_joint_model_blocks_and_adjoint(self) -> None:
        """The joint value, ``jvp`` and ``vjp`` are the member blocks."""
        edges = [0, 0.5, 1.0]
        sn = ExpansionHistoryModel(
            [0.3, 0.8], "piecewise_constant", edges, nuisance=("log10_h0", "offset")
        )
        bao = BaoDistanceModel(
            [0.4, 0.9], "piecewise_constant", edges, nuisance=("log10_h0", "log10_rs")
        )
        joint = JointExpansionModel([sn, bao])
        self.assertEqual(
            joint.parameter_names, ("x1", "x2", "log10_h0", "offset", "log10_rs")
        )
        x = (0.1, -0.2, 1.85, 0.05, 2.17)
        v = (0.2, 0.1, -0.03, 0.02, 0.04)
        val = joint.value(x)
        self.assertEqual(val[:2], sn.value((0.1, -0.2, 1.85, 0.05)))
        self.assertEqual(val[2:], bao.value((0.1, -0.2, 1.85, 2.17)))
        jv = joint.jvp(x, v)
        self.assertEqual(
            jv[:2], sn.jvp((0.1, -0.2, 1.85, 0.05), (0.2, 0.1, -0.03, 0.02))
        )
        # the bin-only SN member has a zero column for log10_rs and vice versa
        cols = jacobian_columns(joint, x)
        self.assertEqual(cols[4][:2], (0.0, 0.0))
        self.assertEqual(cols[3][2:], (0.0, 0.0))
        check = check_derivatives(joint, x, v, (1.0, -1.0, 0.5, 2.0))
        self.assertLess(check.dot_relative_error, 1e-12)

    def test_degeneracies_measured(self) -> None:
        """SN, BAO and SN+BAO keep one free combination; the Hubble rate breaks it.

        Measured at the LCDM baseline (omega_m held fixed): smallest whitened
        singular values 7.6e-15 (SN), 0 (BAO), 6.8e-15 (SN+BAO) and 24.8
        (SN+BAO+Hubble) against largest values near 230-880.
        """
        report = bao_degeneracy_report()
        self.assertEqual(report.labels, ("sn", "bao", "sn+bao", "sn+bao+hubble"))
        fixed = dict(zip(report.labels, report.omega_m_fixed_spectra, strict=True))
        for label in ("sn", "bao", "sn+bao"):
            sv = fixed[label].singular_values
            self.assertLess(sv[-1], 1e-9 * sv[0], msg=label)
        broken = fixed["sn+bao+hubble"].singular_values
        self.assertGreater(broken[-1], 1.0)
        self.assertGreater(broken[-1], 1e6 * fixed["sn+bao"].singular_values[-1])
        self.assertEqual(fixed["sn+bao+hubble"].numerical_rank, 3)

    def test_null_directions_are_the_predicted_combinations(self) -> None:
        """SN: ``d offset = 5 d log10 H0``; BAO: ``d log10 H0 = -d log10 r_s``."""
        report = bao_degeneracy_report()
        fixed = dict(zip(report.labels, report.omega_m_fixed_spectra, strict=True))
        h, off = fixed["sn"].weakest_direction
        self.assertAlmostEqual(off / h, 5.0, places=6)
        h, rs = fixed["bao"].weakest_direction
        self.assertAlmostEqual(h / rs, -1.0, places=6)
        h, off, rs = fixed["sn+bao"].weakest_direction
        norm = math.sqrt(27.0)
        sign = 1.0 if h > 0 else -1.0
        self.assertAlmostEqual(sign * h * norm, 1.0, places=6)
        self.assertAlmostEqual(sign * off * norm, 5.0, places=6)
        self.assertAlmostEqual(sign * rs * norm, -1.0, places=6)

    def test_spectrum_validation(self) -> None:
        """Unknown parameters, wrong noise size and bad tolerance raise."""
        model = ExpansionHistoryModel(
            [0.3, 0.8], "piecewise_constant", [0, 1.0], nuisance=("offset",)
        )
        x0 = nominal_expansion_point(model.parameter_names)
        noise = ScaledIdentityCovariance(2, 0.01)
        with self.assertRaises(ValueError):
            whitened_spectrum(model, x0, noise, parameters=("nope",))
        with self.assertRaises(ValueError):
            whitened_spectrum(model, x0, ScaledIdentityCovariance(3, 0.01))
        with self.assertRaises(ValueError):
            whitened_spectrum(model, x0, noise, rank_tolerance=0.0)
        with self.assertRaises(ValueError):
            nominal_expansion_point(("bogus",))
        with self.assertRaises(ValueError):
            JointExpansionModel([])
        with self.assertRaises(ValueError):
            JointExpansionModel(
                [
                    ExpansionHistoryModel([0.5], "piecewise_constant", [0, 1.0]),
                    ExpansionHistoryModel([0.5], "piecewise_constant", [0, 0.5, 1.0]),
                ]
            )


class InformationGainTests(unittest.TestCase):
    """Linear-Gaussian EIG and the greedy redshift design."""

    def test_scalar_closed_form(self) -> None:
        """One parameter, white noise: ``EIG = 1/2 log(1 + p^2 |j|^2 / s^2)``."""
        j = (0.3, -0.7, 1.1)
        s, p = 0.2, 0.9
        eig = linear_gaussian_eig([j], ScaledIdentityCovariance(3, s * s), (p,))
        self.assertAlmostEqual(
            eig,
            0.5 * math.log(1.0 + p * p * sum(v * v for v in j) / (s * s)),
            places=13,
        )
        self.assertEqual(
            linear_gaussian_eig([(0.0, 0.0)], ScaledIdentityCovariance(2, 1.0), (1.0,)),
            0.0,
        )

    def test_dense_matches_explicit_inverse_with_correlated_noise(self) -> None:
        """The factor-based value equals ``1/2 logdet(I + P J' Gamma^-1 J)``."""
        cols = [(1.0, 0.5, -0.2, 0.3), (0.1, -0.4, 0.9, 0.2), (0.7, 0.7, 0.1, -0.5)]
        sds = (0.5, 1.0, 0.3)
        cal = CalibrationNoise(0.3, offset_sd=0.4, smooth_sd=0.2)
        zs = [0.1, 0.2, 0.5, 0.9]
        gamma = cal.matrix(zs)
        ginv = inverse_spd(tuple(tuple(r) for r in gamma))
        n = 3
        m = [
            [
                sds[a]
                * sds[b]
                * sum(
                    cols[a][i] * ginv[i][k] * cols[b][k]
                    for i in range(4)
                    for k in range(4)
                )
                + (1.0 if a == b else 0.0)
                for b in range(n)
            ]
            for a in range(n)
        ]
        ref = 0.5 * log_det_spd(tuple(tuple(r) for r in m))
        got = linear_gaussian_eig(cols, cal.covariance(zs), sds)
        self.assertAlmostEqual(got, ref, places=12)
        # more noise (Loewner order) can only reduce the information gain
        white = linear_gaussian_eig(cols, CalibrationNoise(0.3).covariance(zs), sds)
        self.assertLess(got, white)

    def _design(self, cal: CalibrationNoise, count: int = 3) -> GreedyDesign:
        """Return a greedy design over twelve candidates."""
        cands = [0.1 * k for k in range(1, 13)]
        return greedy_redshift_design(
            [0.2, 0.6, 1.0],
            cands,
            count,
            bin_edges=_EDGES4,
            calibration=cal,
            nuisance=("omega_m", "offset"),
        )

    def test_greedy_dense_value_verified_exactly(self) -> None:
        """The final dense EIG is the base plus the gains and the rank-one value.

        Measured: telescoping error 0 and rank-one error ~4e-16.
        """
        design = self._design(CalibrationNoise(0.1))
        self.assertLess(design.telescoping_error, 1e-12)
        self.assertLess(design.rank_one_error, 1e-11)
        self.assertAlmostEqual(design.cumulative_eig[-1], design.dense_eig, places=12)
        self.assertEqual(len(set(design.selected)), 3)
        # the first greedy choice maximizes the first-step gain
        best = max(range(12), key=lambda j: design.first_step_gains[j])
        self.assertAlmostEqual(design.selected[0], 0.1 * (best + 1), places=12)
        self.assertAlmostEqual(
            design.marginal_gains[0], max(design.first_step_gains), places=12
        )

    def test_greedy_submodular_guarantee_against_brute_force(self) -> None:
        """Gains are non-increasing and greedy beats ``(1 - 1/e)`` of the optimum."""
        cal = CalibrationNoise(0.1)
        design = self._design(cal, count=3)
        gains = design.marginal_gains
        self.assertTrue(
            all(a >= b - 1e-12 for a, b in zip(gains, gains[1:], strict=False))
        )
        cands = [0.2, 0.5, 0.8, 1.1, 1.3, 1.4, 1.5]
        common = {
            "bin_edges": _EDGES4,
            "calibration": cal,
            "nuisance": ("omega_m", "offset"),
        }
        greedy = greedy_redshift_design([0.2, 0.6, 1.0], cands, 2, **common)  # type: ignore[arg-type]
        best = 0.0
        for combo in itertools.combinations(cands, 2):
            one = greedy_redshift_design([0.2, 0.6, 1.0], list(combo), 2, **common)  # type: ignore[arg-type]
            best = max(best, one.dense_eig - one.base_eig)
        achieved = greedy.dense_eig - greedy.base_eig
        self.assertLessEqual(achieved, best + 1e-12)
        self.assertGreaterEqual(achieved, (1.0 - 1.0 / math.e) * best)

    def test_correlated_calibration_lowers_information(self) -> None:
        """A common offset and smooth drift reduce the achievable gain."""
        white = self._design(CalibrationNoise(0.1))
        corr = self._design(
            CalibrationNoise(0.1, offset_sd=0.3, smooth_sd=0.2, length_scale=0.6)
        )
        self.assertLess(corr.dense_eig, white.dense_eig)
        self.assertTrue(math.isnan(corr.rank_one_eig))
        self.assertLess(corr.telescoping_error, 1e-12)

    def test_invalid(self) -> None:
        """Bad counts, duplicates, shapes and noise specifications raise."""
        cal = CalibrationNoise(0.1)
        common = {"bin_edges": _EDGES4, "calibration": cal}
        with self.assertRaises(ValueError):
            greedy_redshift_design([0.2], [0.3, 0.4], 3, **common)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            greedy_redshift_design([0.2], [0.3, 0.3], 1, **common)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            greedy_redshift_design([0.2], [0.3, 0.4], 0, **common)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            greedy_redshift_design([0.2], [0.3], 1, bin_edges=_EDGES4, calibration=0.1)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            CalibrationNoise(0.0)
        with self.assertRaises(ValueError):
            CalibrationNoise(0.1, offset_sd=-1.0)
        with self.assertRaises(ValueError):
            CalibrationNoise(0.1, length_scale=0.0)
        with self.assertRaises(ValueError):
            linear_gaussian_eig([(1.0, 2.0)], ScaledIdentityCovariance(3, 1.0), (1.0,))
        with self.assertRaises(ValueError):
            linear_gaussian_eig([(1.0, 2.0)], ScaledIdentityCovariance(2, 1.0), (0.0,))
        with self.assertRaises(ValueError):
            correlated_calibration_benchmark(cal, sigma_mu=0.1)
        with self.assertRaises(TypeError):
            correlated_calibration_benchmark(0.1)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            default_prior_sd(("bogus",))


if __name__ == "__main__":
    unittest.main()
