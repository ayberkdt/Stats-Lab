"""Tests for the synthetic expansion-history benchmark A02 (roadmap 20B).

Everything here is SYNTHETIC: no survey data and no claim about the real
universe. Order: closed forms (constant-w factor, LCDM, low-z limit), the
independent Simpson oracle for the distance integral and the closed-form CPL
factor as oracle for the finer truth generator, then derivative checks, the
nuisance degeneracy and the spectrum, one solver smoke, and invalid inputs.
"""

from __future__ import annotations

import math
import unittest
from collections.abc import Callable

from probstat.inverse import (
    check_derivatives,
    gauss_newton_krylov,
    spectral_diagnostics,
)
from probstat.inverse.benchmark_cosmology import (
    ExpansionHistoryModel,
    synthetic_expansion_history_benchmark,
)
from probstat.linalg import (
    ScaledIdentityCovariance,
    inverse_spd,
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


if __name__ == "__main__":
    unittest.main()
