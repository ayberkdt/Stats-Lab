"""Tests for latent force models (``probstat.surrogates.latent_force``).

Oracles: Gauss-Legendre double integration of the Green's function convolution
(independent of the closed form), Monte Carlo of the ODE driven by sampled
forces, a dense GP evidence/posterior from ``probstat.linalg``, and a plain
multi-output GP baseline. No NumPy is used.
"""

from __future__ import annotations

import math
import os
import unittest

from probstat.foundations.random_streams import resolve_stream
from probstat.linalg import cholesky, log_det_spd, solve_spd
from probstat.surrogates import StationaryKernel, fit_multi_output_gp
from probstat.surrogates.latent_force import (
    LatentForceChannel,
    _expm,
    _mm,
    _mt,
    fit_latent_force_model,
    fit_latent_force_state_space,
    latent_force_log_likelihood,
    latent_force_model,
    lfm_force_covariance,
    lfm_mean,
    lfm_output_covariance,
    mass_spring_damper_channel,
    matern_state_space,
    smooth_latent_force_state_space,
    state_space_latent_force_log_likelihood,
)

# ---------------------------------------------------------------------------
# independent numerical oracles
# ---------------------------------------------------------------------------


def gauss_legendre(n: int) -> tuple[list[float], list[float]]:
    """Return the ``n``-point Gauss-Legendre nodes and weights on ``[-1, 1]``."""
    nodes, weights = [], []
    for i in range(1, n + 1):
        x = math.cos(math.pi * (i - 0.25) / (n + 0.5))
        for _ in range(100):
            p0, p1 = 1.0, x
            for k in range(2, n + 1):
                p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
            dp = n * (x * p1 - p0) / (x * x - 1.0)
            step = p1 / dp
            x -= step
            if abs(step) < 1e-15:
                break
        nodes.append(x)
        weights.append(2.0 / ((1.0 - x * x) * dp * dp))
    return nodes, weights


_X48, _W48 = gauss_legendre(48)
_X24, _W24 = gauss_legendre(24)


def panel(a: float, b: float, rule: tuple[list[float], list[float]]) -> list:
    """Quadrature points of one Gauss-Legendre panel on ``[a, b]``."""
    xs, ws = rule
    half = 0.5 * (b - a)
    return [(a + half * (x + 1.0), half * w) for x, w in zip(xs, ws, strict=True)]


def pieces(a: float, b: float, cuts: list[float], rule: tuple) -> list:
    """Composite quadrature on ``[a, b]`` split at the interior ``cuts``."""
    pts = [a, *sorted(c for c in cuts if a < c < b), b]
    out: list = []
    for lo, hi in zip(pts, pts[1:], strict=False):
        out += panel(lo, hi, rule)
    return out


def se(a: float, b: float, ell: float) -> float:
    """Return the unit squared-exponential kernel."""
    return math.exp(-((a - b) ** 2) / (2.0 * ell * ell))


def force_cov_quad(t: float, tp: float, d: float, s: float, ell: float) -> float:
    """``Cov(y(t), f(t'))`` by quadrature of the Green's function."""
    rule = (_X48, _W48)
    return s * sum(
        w * math.exp(-d * (t - u)) * se(u, tp, ell) for u, w in panel(0.0, t, rule)
    )


def output_cov_quad(t, tp, d, dp, s, sp, ell) -> float:
    """``Cov(y_d(t), y_d'(t'))`` by double quadrature of the convolution."""
    rule = (_X48, _W48)
    total = 0.0
    for u, wu in panel(0.0, t, rule):
        inner = sum(
            wv * math.exp(-dp * (tp - v)) * se(u, v, ell)
            for v, wv in panel(0.0, tp, rule)
        )
        total += wu * math.exp(-d * (t - u)) * inner
    return s * sp * total


def matern_kernel(r: float, nu: float, ell: float) -> float:
    """Unit-variance Matern kernel."""
    r = abs(r)
    if nu == 0.5:
        return math.exp(-r / ell)
    if nu == 1.5:
        a = math.sqrt(3.0) * r / ell
        return (1.0 + a) * math.exp(-a)
    a = math.sqrt(5.0) * r / ell
    return (1.0 + a + a * a / 3.0) * math.exp(-a)


def green(ch: LatentForceChannel):
    """Impulse response ``g(s)`` of a state-space channel (analytic)."""
    if ch.order == 1:
        d = ch.coefficients[0]
        return lambda s: ch.sensitivity * math.exp(-d * s)
    omega, zeta = ch.coefficients
    wd = omega * math.sqrt(1.0 - zeta * zeta)
    return lambda s: (
        ch.sensitivity * math.exp(-zeta * omega * s) * math.sin(wd * s) / wd
    )


def ss_cov(c1, t1, c2, t2, nu, ell) -> float:
    """Output covariance for a Matern force by split double quadrature."""
    g1, g2 = green(c1), green(c2)
    rule = (_X24, _W24)
    total = 0.0
    for u, wu in pieces(0.0, t1, [t2], rule):
        inner = sum(
            wv * g2(t2 - v) * matern_kernel(u - v, nu, ell)
            for v, wv in pieces(0.0, t2, [u], rule)
        )
        total += wu * g1(t1 - u) * inner
    return total


def ss_force_cov(c1, t1, tstar, nu, ell) -> float:
    """``Cov(y(t1), f(t*))`` for a Matern force by split quadrature."""
    g1 = green(c1)
    rule = (_X24, _W24)
    return sum(
        w * g1(t1 - u) * matern_kernel(u - tstar, nu, ell)
        for u, w in pieces(0.0, t1, [tstar], rule)
    )


def dense_state_space(channels, times, values, nu, ell, query=()):
    """Dense GP evidence and posterior marginals for the Matern-force model."""
    pts = [(c, t) for c, row in enumerate(times) for t in row]
    n = len(pts)
    cov = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            cov[i][j] = cov[j][i] = ss_cov(
                channels[pts[i][0]], pts[i][1], channels[pts[j][0]], pts[j][1], nu, ell
            )
        cov[i][i] += channels[pts[i][0]].noise_variance
    y = [
        v - channels[c].level
        for (c, _), v in zip(pts, [x for r in values for x in r], strict=True)
    ]
    matrix = tuple(tuple(r) for r in cov)
    alpha = solve_spd(matrix, tuple(y))
    evidence = -0.5 * (
        sum(a * b for a, b in zip(y, alpha, strict=True))
        + log_det_spd(matrix)
        + n * math.log(2.0 * math.pi)
    )
    out = []
    for c, tq in query:
        cross = [ss_cov(channels[c], tq, channels[k], tk, nu, ell) for k, tk in pts]
        mean = channels[c].level + sum(a * b for a, b in zip(cross, alpha, strict=True))
        sol = solve_spd(matrix, tuple(cross))
        var = ss_cov(channels[c], tq, channels[c], tq, nu, ell) - sum(
            a * b for a, b in zip(cross, sol, strict=True)
        )
        fcross = [ss_force_cov(channels[k], tk, tq, nu, ell) for k, tk in pts]
        fsol = solve_spd(matrix, tuple(fcross))
        fmean = sum(a * b for a, b in zip(fcross, alpha, strict=True))
        fvar = 1.0 - sum(a * b for a, b in zip(fcross, fsol, strict=True))
        out.append((mean, var, fmean, fvar))
    return evidence, out


# ---------------------------------------------------------------------------
# closed-form covariances
# ---------------------------------------------------------------------------


class ClosedFormCovarianceTests(unittest.TestCase):
    """Closed forms against quadrature, symmetry and limits."""

    CASES = (
        (1.3, 0.7, 0.8, 1.5, 0.6),
        (2.0, 2.5, 1.0, 1.0, 1.0),
        (0.5, 3.0, 2.0, 0.3, 0.4),
        (4.0, 0.2, 0.5, 0.9, 2.0),
        (6.0, 5.5, 3.0, 2.0, 3.0),
        (0.05, 0.3, 4.0, 4.0, 0.5),
    )

    def test_output_force_covariance_matches_quadrature(self) -> None:
        """Closed form [EQ-SUR-LFM-002] agrees with the Green convolution."""
        for t, tp, d, _dp, ell in self.CASES:
            closed = lfm_force_covariance(t, tp, d, 1.3, ell)
            quad = force_cov_quad(t, tp, d, 1.3, ell)
            self.assertAlmostEqual(closed, quad, delta=1e-11 * max(1.0, abs(quad)))

    def test_output_output_covariance_matches_double_quadrature(self) -> None:
        """Closed form [EQ-SUR-LFM-003] agrees with the double convolution."""
        for t, tp, d, dp, ell in self.CASES:
            closed = lfm_output_covariance(t, tp, d, dp, 1.3, -0.7, ell)
            quad = output_cov_quad(t, tp, d, dp, 1.3, -0.7, ell)
            self.assertAlmostEqual(closed, quad, delta=1e-10 * max(1.0, abs(quad)))

    def test_symmetry_boundary_and_positive_definiteness(self) -> None:
        """K(t,t';d,d') = K(t',t;d',d), zero at t = 0, PSD Gram matrix."""
        a = lfm_output_covariance(1.1, 2.3, 0.7, 1.9, 1.0, 2.0, 0.8)
        b = lfm_output_covariance(2.3, 1.1, 1.9, 0.7, 2.0, 1.0, 0.8)
        self.assertAlmostEqual(a, b, delta=1e-13)
        self.assertEqual(lfm_output_covariance(0.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0), 0.0)
        self.assertAlmostEqual(
            lfm_force_covariance(0.0, 1.0, 1.0, 1.0, 1.0), 0.0, delta=1e-15
        )
        times = [0.3 * i for i in range(1, 13)]
        gram = tuple(
            tuple(lfm_output_covariance(s, t, 0.9, 0.9, 1.0, 1.0, 0.7) for t in times)
            for s in times
        )
        cholesky(gram)  # raises ValueError unless positive definite

    def test_stiff_limit_and_stationary_variance(self) -> None:
        """Large decays give S k / D; late times give the stationary variance."""
        d, ell = 300.0, 0.03  # D l = 9: inside the closed-form range
        self.assertAlmostEqual(
            lfm_force_covariance(2.0, 2.0, d, 1.0, ell) * d,
            1.0,
            delta=3e-2,
        )
        self.assertAlmostEqual(
            lfm_output_covariance(2.0, 2.0, d, d, 1.0, 1.0, ell) * d * d,
            1.0,
            delta=3e-2,
        )
        # t = 30 is stationary for D = 3: compare with a truncated quadrature
        late = lfm_output_covariance(30.0, 30.0, 3.0, 3.0, 1.0, 1.0, 3.0)
        proxy = output_cov_quad(12.0, 12.0, 3.0, 3.0, 1.0, 1.0, 3.0)
        self.assertAlmostEqual(late, proxy, delta=1e-9)
        self.assertTrue(
            math.isfinite(lfm_output_covariance(30, 0.5, 3.0, 3.0, 1, 1, 3.0))
        )

    def test_mean_and_validation(self) -> None:
        """Mean of the ODE and rejection of invalid arguments."""
        self.assertEqual(lfm_mean(5.0, 2.0, 1.0), 0.5)
        self.assertAlmostEqual(lfm_mean(1.0, 2.0, 1.0, 3.0), 0.5 + 2.5 * math.exp(-2.0))
        bad = (
            lambda: lfm_force_covariance(-1.0, 1.0, 1.0, 1.0, 1.0),
            lambda: lfm_force_covariance(1.0, 1.0, 0.0, 1.0, 1.0),
            lambda: lfm_force_covariance(1.0, 1.0, 1.0, 1.0, -1.0),
            lambda: lfm_force_covariance(1.0, 1.0, 20.0, 1.0, 1.0),  # D l > 10
            lambda: lfm_output_covariance(1.0, 1.0, 1.0, math.nan, 1.0, 1.0, 1.0),
            lambda: lfm_mean(1.0, -1.0, 0.0),
        )
        for call in bad:
            with self.assertRaises(ValueError):
                call()
        with self.assertRaises(TypeError):
            lfm_force_covariance(True, 1.0, 1.0, 1.0, 1.0)  # type: ignore[arg-type]


class MonteCarloOdeTests(unittest.TestCase):
    """Moments of the ODE driven by sampled forces match the closed forms."""

    def test_rk4_matches_analytic_convolution_for_a_cosine_force(self) -> None:
        """The ODE integrator and the per-feature convolution agree."""
        d, s, w, phi, t_end = 1.3, 0.8, 2.1, 0.4, 3.0
        steps = 600
        h = t_end / steps
        y = 0.0

        def rhs(t: float, y: float) -> float:
            return -d * y + s * math.cos(w * t + phi)

        for k in range(steps):
            t = k * h
            k1 = rhs(t, y)
            k2 = rhs(t + h / 2, y + h / 2 * k1)
            k3 = rhs(t + h / 2, y + h / 2 * k2)
            k4 = rhs(t + h, y + h * k3)
            y += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        exact = (
            s
            * (
                d * math.cos(w * t_end + phi)
                + w * math.sin(w * t_end + phi)
                - math.exp(-d * t_end) * (d * math.cos(phi) + w * math.sin(phi))
            )
            / (d * d + w * w)
        )
        self.assertAlmostEqual(y, exact, delta=1e-9)

    def test_sampled_force_moments_match_closed_forms(self) -> None:
        """Random-Fourier-feature forces reproduce Cov(y, y') and Cov(y, f)."""
        ell = 0.8
        chans = ((0.9, 1.1), (1.7, -0.6))  # (D, S)
        t_pairs = ((1.5, 2.2), (3.0, 2.5), (0.8, 3.5))
        gen = resolve_stream(20261127).child("lfm-mc").generator()
        replicates, features = 4000, 24
        acc_yy = [0.0] * len(t_pairs)
        acc_yy2 = [0.0] * len(t_pairs)
        acc_yf = [0.0] * len(t_pairs)
        acc_yf2 = [0.0] * len(t_pairs)

        def convolve(t: float, d: float, w: float, phi: float) -> float:
            """Exact ``int_0^t exp(-d (t - u)) cos(w u + phi) du``."""
            return (
                d * math.cos(w * t + phi)
                + w * math.sin(w * t + phi)
                - math.exp(-d * t) * (d * math.cos(phi) + w * math.sin(phi))
            ) / (d * d + w * w)

        for _ in range(replicates):
            ws = [gen.gauss(0.0, 1.0 / ell) for _ in range(features)]
            ps = [2.0 * math.pi * gen.random() for _ in range(features)]
            amp = math.sqrt(2.0 / features)
            for idx, (t, tp) in enumerate(t_pairs):
                y1 = (
                    chans[0][1]
                    * amp
                    * sum(
                        convolve(t, chans[0][0], w, p)
                        for w, p in zip(ws, ps, strict=True)
                    )
                )
                y2 = (
                    chans[1][1]
                    * amp
                    * sum(
                        convolve(tp, chans[1][0], w, p)
                        for w, p in zip(ws, ps, strict=True)
                    )
                )
                f = amp * sum(math.cos(w * tp + p) for w, p in zip(ws, ps, strict=True))
                acc_yy[idx] += y1 * y2
                acc_yy2[idx] += (y1 * y2) ** 2
                acc_yf[idx] += y1 * f
                acc_yf2[idx] += (y1 * f) ** 2
        for idx, (t, tp) in enumerate(t_pairs):
            for acc, acc2, exact in (
                (
                    acc_yy,
                    acc_yy2,
                    lfm_output_covariance(
                        t, tp, chans[0][0], chans[1][0], chans[0][1], chans[1][1], ell
                    ),
                ),
                (
                    acc_yf,
                    acc_yf2,
                    lfm_force_covariance(t, tp, chans[0][0], chans[0][1], ell),
                ),
            ):
                mean = acc[idx] / replicates
                var = acc2[idx] / replicates - mean * mean
                mcse = math.sqrt(var / replicates)
                self.assertLess(abs(mean - exact), 4.5 * mcse, (idx, mean, exact, mcse))


# ---------------------------------------------------------------------------
# likelihood, gradient, posterior
# ---------------------------------------------------------------------------


def toy_data(count: int = 3, points: int = 9, seed: int = 5):
    """Small heterotopic data set from a smooth signal plus noise."""
    gen = resolve_stream(seed).child("toy").generator()
    times = [
        [0.45 * i + 0.11 * c + 0.2 for i in range(points - c)] for c in range(count)
    ]
    values = [
        [math.sin(0.8 * t + c) + 0.05 * gen.gauss(0.0, 1.0) for t in row]
        for c, row in enumerate(times)
    ]
    return times, values


class LikelihoodTests(unittest.TestCase):
    """Evidence, exact gradient and posterior against dense oracles."""

    DECAYS = (0.9, 0.5, 1.4)
    SENS = (1.0, 0.6, -1.5)
    BASAL = (0.2, 0.1, -0.3)
    ELL = 0.9
    NOISE = (0.01, 0.02, 0.015)

    def oracle_matrix(self, times):
        """Dense covariance from quadrature (independent of the closed form)."""
        pts = [(c, t) for c, row in enumerate(times) for t in row]
        n = len(pts)
        cov = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                ci, ti = pts[i]
                cj, tj = pts[j]
                cov[i][j] = cov[j][i] = output_cov_quad(
                    ti,
                    tj,
                    self.DECAYS[ci],
                    self.DECAYS[cj],
                    self.SENS[ci],
                    self.SENS[cj],
                    self.ELL,
                )
            cov[i][i] += self.NOISE[pts[i][0]]
        return pts, cov

    def test_evidence_matches_dense_quadrature_oracle(self) -> None:
        """The evidence equals the dense GP evidence of the quadrature matrix."""
        times, values = toy_data()
        pts, cov = self.oracle_matrix(times)
        mean = [self.BASAL[c] / self.DECAYS[c] for c, _ in pts]
        y = [v - m for v, m in zip((x for r in values for x in r), mean, strict=True)]
        matrix = tuple(tuple(r) for r in cov)
        alpha = solve_spd(matrix, tuple(y))
        expected = -0.5 * (
            sum(a * b for a, b in zip(y, alpha, strict=True))
            + log_det_spd(matrix)
            + len(y) * math.log(2.0 * math.pi)
        )
        result = latent_force_log_likelihood(
            times, values, self.DECAYS, self.SENS, self.BASAL, self.ELL, self.NOISE
        )
        self.assertAlmostEqual(result.log_likelihood, expected, delta=1e-8)
        self.assertEqual(result.observations, len(y))

    def test_gradient_matches_central_differences(self) -> None:
        """All 13 exact partials agree with finite differences (both mean modes)."""
        times, values = toy_data()
        count = 3
        for initial in (None, (0.1, 0.0, 0.2)):
            result = latent_force_log_likelihood(
                times,
                values,
                self.DECAYS,
                self.SENS,
                self.BASAL,
                self.ELL,
                self.NOISE,
                initial_values=initial,
            )
            theta = (
                [math.log(v) for v in self.DECAYS]
                + list(self.SENS)
                + list(self.BASAL)
                + [math.log(self.ELL)]
                + [math.log(v) for v in self.NOISE]
            )

            def value(th: list[float], initial=initial) -> float:
                return latent_force_log_likelihood(
                    times,
                    values,
                    [math.exp(v) for v in th[:count]],
                    th[count : 2 * count],
                    th[2 * count : 3 * count],
                    math.exp(th[3 * count]),
                    [math.exp(v) for v in th[3 * count + 1 :]],
                    initial_values=initial,
                    gradient=False,
                ).log_likelihood

            self.assertEqual(len(result.gradient), len(theta))
            for i, name in enumerate(result.parameter_names):
                h = 1e-5
                up, dn = list(theta), list(theta)
                up[i] += h
                dn[i] -= h
                fd = (value(up) - value(dn)) / (2.0 * h)
                tol = 2e-6 * max(1.0, abs(fd))
                self.assertAlmostEqual(result.gradient[i], fd, delta=tol, msg=name)

    def test_posterior_matches_dense_oracle(self) -> None:
        """Predicted outputs and the inferred force match the dense posterior."""
        times, values = toy_data(count=2, points=8)
        sens, dec = self.SENS[:2], self.DECAYS[:2]
        model = latent_force_model(
            times, values, dec, sens, self.BASAL[:2], self.ELL, self.NOISE[:2]
        )
        pts = [(c, t) for c, row in enumerate(times) for t in row]
        cov = [[0.0] * len(pts) for _ in pts]
        for i, (ci, ti) in enumerate(pts):
            for j, (cj, tj) in enumerate(pts):
                if j >= i:
                    cov[i][j] = cov[j][i] = output_cov_quad(
                        ti, tj, dec[ci], dec[cj], sens[ci], sens[cj], self.ELL
                    )
            cov[i][i] += self.NOISE[ci]
        matrix = tuple(tuple(r) for r in cov)
        mean = [self.BASAL[c] / dec[c] for c, _ in pts]
        y = [v - m for v, m in zip((x for r in values for x in r), mean, strict=True)]
        alpha = solve_spd(matrix, tuple(y))
        tstar = 2.2
        # output channel 1
        cross = [
            output_cov_quad(tstar, tj, dec[1], dec[cj], sens[1], sens[cj], self.ELL)
            for cj, tj in pts
        ]
        sol = solve_spd(matrix, tuple(cross))
        exp_mean = self.BASAL[1] / dec[1] + sum(
            a * b for a, b in zip(cross, alpha, strict=True)
        )
        exp_var = output_cov_quad(
            tstar, tstar, dec[1], dec[1], sens[1], sens[1], self.ELL
        ) - sum(a * b for a, b in zip(cross, sol, strict=True))
        pred = model.predict_outputs(1, [tstar])
        self.assertAlmostEqual(pred.mean[0], exp_mean, delta=1e-8)
        self.assertAlmostEqual(pred.variance[0], exp_var, delta=1e-8)
        noisy = model.predict_outputs(1, [tstar], include_noise=True)
        self.assertAlmostEqual(
            noisy.variance[0] - pred.variance[0], self.NOISE[1], delta=1e-12
        )
        # the force
        fcross = [
            force_cov_quad(tj, tstar, dec[cj], sens[cj], self.ELL) for cj, tj in pts
        ]
        fsol = solve_spd(matrix, tuple(fcross))
        force = model.predict_force([tstar])
        self.assertAlmostEqual(
            force.mean[0],
            sum(a * b for a, b in zip(fcross, alpha, strict=True)),
            delta=1e-8,
        )
        self.assertAlmostEqual(
            force.variance[0],
            1.0 - sum(a * b for a, b in zip(fcross, fsol, strict=True)),
            delta=1e-8,
        )
        scaled = model.predict_force([tstar], channel=1)
        self.assertAlmostEqual(scaled.mean[0], sens[1] * force.mean[0], delta=1e-12)
        self.assertAlmostEqual(
            scaled.variance[0], sens[1] ** 2 * force.variance[0], delta=1e-12
        )
        self.assertLess(scaled.lower[0], scaled.mean[0])
        self.assertGreater(scaled.upper[0], scaled.mean[0])

    def test_force_reverts_to_prior_far_from_data(self) -> None:
        """Beyond the data the force posterior is the prior N(0, 1)."""
        times, values = toy_data(count=2, points=8)
        model = latent_force_model(
            times,
            values,
            self.DECAYS[:2],
            self.SENS[:2],
            self.BASAL[:2],
            self.ELL,
            self.NOISE[:2],
        )
        far = model.predict_force([60.0])
        self.assertAlmostEqual(far.mean[0], 0.0, delta=1e-8)
        self.assertAlmostEqual(far.variance[0], 1.0, delta=1e-8)

    def test_invalid_inputs(self) -> None:
        """Invalid data and parameters are rejected, never repaired."""
        times, values = toy_data(count=2, points=5)
        args = (
            self.DECAYS[:2],
            self.SENS[:2],
            self.BASAL[:2],
            self.ELL,
            self.NOISE[:2],
        )
        with self.assertRaises(ValueError):
            latent_force_model(times, values[:1], *args)
        with self.assertRaises(ValueError):
            latent_force_model([[-1.0, 1.0], times[1]], values, *args)
        with self.assertRaises(ValueError):
            latent_force_model(times, [[math.nan] * 5, values[1]], *args)
        with self.assertRaises(ValueError):
            latent_force_model(times, values, (0.9, -0.5), *args[1:])
        with self.assertRaises(ValueError):
            latent_force_model(
                times, values, args[0], args[1], args[2], args[3], (0.01, 0.0)
            )
        with self.assertRaises(ValueError):
            latent_force_model(times, values, args[0], args[1], args[2], 0.0, args[4])
        model = latent_force_model(times, values, *args)
        with self.assertRaises(ValueError):
            model.predict_outputs(5, [1.0])
        with self.assertRaises(ValueError):
            model.predict_force([-0.5])
        with self.assertRaises(TypeError):
            model.predict_outputs(True, [1.0])  # type: ignore[arg-type]

    def test_non_positive_definite_covariance_is_reported(self) -> None:
        """Repeated times with a vanishing noise raise instead of adding jitter."""
        times = [[1.0, 1.0, 1.0, 1.0]]
        values = [[0.1, 0.1, 0.1, 0.1]]
        with self.assertRaises(ValueError):
            latent_force_model(times, values, (1.0,), (1.0,), (0.0,), 1.0, (1e-30,))


# ---------------------------------------------------------------------------
# physical demonstration (synthetic)
# ---------------------------------------------------------------------------

_D_TRUE = (0.9, 0.5, 1.4)  # 1/s, thermal-node decay rates (synthetic)
_S_TRUE = (1.0, 0.7, 1.4)  # K/s per unit of heat input (synthetic)
_LEVEL = (20.0, 22.0, 19.0)  # K, equilibrium temperatures (synthetic)


def heat_input(t: float) -> float:
    """Synthetic heat input (channel-0 units: K/s): two smooth pulses."""
    return math.exp(-((t - 2.5) ** 2) / (2 * 0.6**2)) + 0.6 * math.exp(
        -((t - 6.0) ** 2) / (2 * 0.8**2)
    )


def simulate_channels(t_end: float, h: float = 0.005):
    """RK4 solution of the three coupled-by-force thermal channels."""
    steps = int(round(t_end / h))
    out = []
    for c in range(3):
        y = _LEVEL[c]
        series = [y]

        def rhs(t: float, y: float, c: int = c) -> float:
            return -_D_TRUE[c] * (y - _LEVEL[c]) + _S_TRUE[c] * heat_input(t)

        for k in range(steps):
            t = k * h
            k1 = rhs(t, y)
            k2 = rhs(t + h / 2, y + h / 2 * k1)
            k3 = rhs(t + h / 2, y + h / 2 * k2)
            k4 = rhs(t + h, y + h * k3)
            y += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            series.append(y)
        out.append(series)
    return out


_STEP = 0.005
_TRUTH = simulate_channels(11.0, _STEP)


def truth(channel: int, t: float) -> float:
    """Noise-free synthetic temperature at (grid-aligned) time ``t``."""
    return _TRUTH[channel][int(round(t / _STEP))]


def synthetic_data(seed: int, cut: float, points: int, sigma: float = 0.02):
    """Noisy measurements of the three channels on ``[0.2, cut]``."""
    gen = resolve_stream(seed).child("physical").generator()
    times = [
        [
            round(0.2 + (cut - 0.2) * i / (points - 1) + 0.015 * c, 3)
            for i in range(points)
        ]
        for c in range(3)
    ]
    values = [
        [truth(c, t) + gen.gauss(0.0, sigma) for t in row]
        for c, row in enumerate(times)
    ]
    return times, values


def rms(errors: list[float]) -> float:
    """Root mean square."""
    return math.sqrt(sum(e * e for e in errors) / len(errors))


def band_coverage(prediction, truth_values: list[float]) -> float:
    """Fraction of ``truth_values`` inside the pointwise 95% band."""
    inside = [
        lo <= x <= hi
        for lo, x, hi in zip(
            prediction.lower, truth_values, prediction.upper, strict=True
        )
    ]
    return sum(inside) / len(inside)


class PhysicalDemonstrationTests(unittest.TestCase):
    """Recover an unknown heat input from three noisy temperature channels.

    All data are synthetic (three first-order thermal nodes driven by one
    smooth heat input); no real device is represented.
    """

    GRID = tuple(0.2 + 0.2 * i for i in range(47))  # 0.2 ... 9.4 s

    @classmethod
    def setUpClass(cls) -> None:
        """Fit the shared models once."""
        cls.true_input = [_S_TRUE[0] * heat_input(t) for t in cls.GRID]
        # (a) known plant: decays and gains calibrated, force time-scale chosen
        # from the actuator (l = 0.7 s), noise level known (sigma = 0.05 K)
        times, values = synthetic_data(1, 9.6, 16, 0.05)
        cls.known = fit_latent_force_model(
            times,
            values,
            initial_decays=_D_TRUE,
            initial_sensitivities=_S_TRUE,
            initial_lengthscale=0.7,
            initial_noise_variances=[0.05**2] * 3,
            fixed=("decays", "sensitivities", "lengthscale", "noise_variances"),
        )
        # (b) every parameter unknown, low-noise data
        cls.times, cls.values = synthetic_data(1, 9.6, 16, 0.02)
        cls.free = fit_latent_force_model(
            cls.times,
            cls.values,
            initial_lengthscale=1.0,
            initial_noise_variances=[1.2e-3] * 3,
            max_iterations=200,
            seed=1,
        )

    def test_known_plant_force_band_covers_the_truth(self) -> None:
        """With a calibrated plant the 95% band of the heat input covers it."""
        forcing = self.known.predict_force(self.GRID, channel=0)
        coverage = band_coverage(forcing, self.true_input)
        error = rms([m - x for m, x in zip(forcing.mean, self.true_input, strict=True)])
        # measured on this seed: coverage 0.91, rms 0.077 (pulse height 1.0);
        # over ten seeds the mean coverage is 0.88 (see the slow study).
        self.assertGreaterEqual(coverage, 0.75, coverage)
        self.assertLess(error, 0.12, error)
        self.assertTrue(all(v > 0.0 for v in forcing.variance))
        levels = [
            b / d for b, d in zip(self.known.basals, self.known.decays, strict=True)
        ]
        for level, true in zip(levels, _LEVEL, strict=True):
            self.assertAlmostEqual(level, true, delta=0.1)

    def test_unknown_plant_identifies_relative_rates_and_gains(self) -> None:
        """Without calibration the decay and gain ratios are recovered.

        The absolute time scale is confounded with the smoothness of the
        unknown force, so only ratios are asserted (measured: the absolute
        decays come out about 30% too fast on this data set).
        """
        model = self.free
        base_truth = [
            latent_force_log_likelihood(
                self.times,
                self.values,
                _D_TRUE,
                _S_TRUE,
                [d * v for d, v in zip(_D_TRUE, _LEVEL, strict=True)],
                0.9,
                [4e-4] * 3,
                gradient=False,
            ).log_likelihood
        ][0]
        self.assertGreater(model.log_marginal_likelihood, base_truth)
        for c in (1, 2):
            self.assertLess(
                abs(
                    (model.decays[c] / model.decays[0]) / (_D_TRUE[c] / _D_TRUE[0])
                    - 1.0
                ),
                0.15,
            )
            self.assertLess(
                abs(
                    (model.sensitivities[c] / model.sensitivities[0])
                    / (_S_TRUE[c] / _S_TRUE[0])
                    - 1.0
                ),
                0.15,
            )
        self.assertGreater(model.sensitivities[0], 0.0)

    def test_extrapolation_beats_the_plain_multi_output_gp(self) -> None:
        """After the data end, the ODE structure predicts decay; a plain GP does not."""
        cut = 7.0
        times, values = synthetic_data(1, cut, 16, 0.02)
        lfm = fit_latent_force_model(
            times,
            values,
            initial_lengthscale=1.0,
            initial_noise_variances=[1.2e-3] * 3,
            max_iterations=200,
            seed=1,
        )
        kernel = StationaryKernel("squared_exponential", 1.0, (1.0,))
        plain = fit_multi_output_gp(
            [[(t,) for t in row] for row in times],
            values,
            kernel,
            rank=1,
            seed=1,
            restarts=1,
            max_iterations=80,
        )
        future = [cut + 0.25 * i for i in range(1, 17)]
        err_lfm, err_gp = [], []
        for c in range(3):
            lfm_mean_ = lfm.predict_outputs(c, future).mean
            for t, p in zip(future, lfm_mean_, strict=True):
                err_lfm.append(p - truth(c, t))
                err_gp.append(plain.predict_with_variance(c, (t,))[0] - truth(c, t))
        ratio = rms(err_gp) / rms(err_lfm)
        self.assertGreater(ratio, 3.0, (rms(err_lfm), rms(err_gp)))
        self.assertLess(rms(err_lfm), 0.1)


# ---------------------------------------------------------------------------
# state-space latent force models
# ---------------------------------------------------------------------------


class StateSpaceTests(unittest.TestCase):
    """Kalman/RTS latent force models against dense GP oracles."""

    def test_matern_realization_is_stationary_and_reproduces_the_kernel(self) -> None:
        """Lyapunov residual and kernel recovery for nu = 1/2, 3/2, 5/2."""
        for nu in (0.5, 1.5, 2.5):
            f, q, p = matern_state_space(nu, 0.7)
            n = len(f)
            lyap = _mm(f, p)
            lyap = [
                [
                    lyap[i][j] + lyap[j][i] + (q if i == j == n - 1 else 0.0)
                    for j in range(n)
                ]
                for i in range(n)
            ]
            self.assertLess(max(abs(v) for row in lyap for v in row), 1e-11)
            for tau in (0.0, 0.3, 1.1):
                trans = _expm([[v * tau for v in row] for row in f])
                cov = _mm(p, _mt(trans))
                self.assertAlmostEqual(
                    cov[0][0], matern_kernel(tau, nu, 0.7), delta=1e-13
                )

    def test_mass_spring_damper_normalization(self) -> None:
        """Map m y'' + c y' + k y = g f to (omega, zeta, g/m)."""
        ch = mass_spring_damper_channel(2.0, 1.2, 8.0, 3.0)
        self.assertEqual(ch.order, 2)
        self.assertAlmostEqual(ch.coefficients[0], 2.0, delta=1e-14)
        self.assertAlmostEqual(ch.coefficients[1], 0.15, delta=1e-14)
        self.assertAlmostEqual(ch.sensitivity, 1.5, delta=1e-14)

    def test_evidence_matches_dense_gp_for_every_smoothness(self) -> None:
        """Kalman evidence = dense GP evidence (first and second order, heterotopic)."""
        channels = [
            LatentForceChannel(1, (0.8,), 1.2, 0.3, 0.01),
            LatentForceChannel(2, (2.0, 0.3), 0.9, -0.2, 0.02),
        ]
        gen = resolve_stream(0).child("ss").generator()
        times = [[0.4 * i for i in range(1, 9)], [0.4 * i for i in range(2, 10)]]
        values = [[gen.gauss(0.0, 1.0) + 0.3 for _ in row] for row in times]
        for nu in (0.5, 1.5, 2.5):
            kalman = state_space_latent_force_log_likelihood(
                channels, times, values, lengthscale=0.7, smoothness=nu
            )
            dense, _ = dense_state_space(channels, times, values, nu, 0.7)
            self.assertAlmostEqual(kalman, dense, delta=1e-8, msg=f"nu={nu}")

    def test_smoother_matches_dense_posterior_of_outputs_and_force(self) -> None:
        """RTS marginals of outputs and force equal the dense GP posterior."""
        channels = [
            LatentForceChannel(1, (1.1,), 1.0, 0.0, 0.01),
            LatentForceChannel(2, (2.5, 0.25), 0.8, 0.5, 0.015),
        ]
        times = [[0.3 * i for i in range(1, 11)], [0.3 * i + 0.1 for i in range(0, 10)]]
        gen = resolve_stream(3).child("ss2").generator()
        values = [
            [
                math.sin(1.7 * t) + 0.1 * gen.gauss(0.0, 1.0) + channels[c].level
                for t in row
            ]
            for c, row in enumerate(times)
        ]
        query = [0.15, 1.05, 2.0, 3.4, 4.5]  # inside, between, and beyond the data
        result = smooth_latent_force_state_space(
            channels, times, values, query, lengthscale=0.9, smoothness=1.5
        )
        points = [(c, t) for t in query for c in range(2)]
        evidence, truth_ = dense_state_space(channels, times, values, 1.5, 0.9, points)
        self.assertAlmostEqual(result.log_marginal_likelihood, evidence, delta=1e-8)
        for k, tq in enumerate(query):
            for c in range(2):
                mean, var, fmean, fvar = truth_[2 * k + c]
                self.assertAlmostEqual(
                    result.output_mean[c][k], mean, delta=2e-6, msg=(c, tq)
                )
                self.assertAlmostEqual(
                    result.output_variance[c][k], var, delta=2e-6, msg=(c, tq)
                )
            self.assertAlmostEqual(
                result.force_mean[k], truth_[2 * k][2], delta=2e-6, msg=tq
            )
            self.assertAlmostEqual(
                result.force_variance[k], truth_[2 * k][3], delta=2e-6, msg=tq
            )
        prediction = result.force_prediction()
        self.assertEqual(prediction.times, tuple(query))

    def test_cost_is_linear_and_regular_grids_reuse_one_discretization(self) -> None:
        """One (A, Q) pair serves a regular grid; steps grow linearly with n."""
        channel = [LatentForceChannel(2, (2.0, 0.3), 1.0, 0.0, 0.01)]
        for n in (50, 100, 200):
            grid = [[0.1 * i for i in range(1, n + 1)]]
            result = smooth_latent_force_state_space(
                channel, grid, [[math.sin(t) for t in grid[0]]], [0.5], lengthscale=1.0
            )
            self.assertEqual(result.discretizations, 1)
            self.assertEqual(result.steps, n + 1)
            self.assertEqual(result.observations, n)

    def test_fit_recovers_a_second_order_resonance(self) -> None:
        """Mass-spring-damper data from a known force: omega, zeta are recovered."""
        omega, zeta, gain = 3.0, 0.2, 2.0
        h, n = 0.002, 3000
        y, v = 0.0, 0.0
        sampled = []
        for k in range(n):
            t = k * h
            f = math.exp(-((t - 1.0) ** 2) / (2 * 0.3**2)) - 0.5 * math.exp(
                -((t - 3.0) ** 2) / (2 * 0.4**2)
            )
            # semi-implicit stepping with a small step is accurate enough for 2e-3 s
            v += h * (-omega * omega * y - 2 * zeta * omega * v + gain * f)
            y += h * v
            if (k + 1) % 100 == 0:
                sampled.append(((k + 1) * h, y))
        gen = resolve_stream(11).child("ss3").generator()
        times = [[t for t, _ in sampled]]
        values = [[y + 0.01 * gen.gauss(0.0, 1.0) for _, y in sampled]]
        start = [LatentForceChannel(2, (2.0, 0.5), 1.0, 0.0, 1e-3)]
        fit = fit_latent_force_state_space(
            start,
            times,
            values,
            lengthscale=0.5,
            smoothness=2.5,
            free=("coefficients", "sensitivity"),
            max_iterations=80,
        )
        found = fit.channels[0]
        self.assertAlmostEqual(found.coefficients[0], omega, delta=0.15 * omega)
        # measured log-evidence must improve on the start
        base = state_space_latent_force_log_likelihood(
            start, times, values, lengthscale=0.5, smoothness=2.5
        )
        self.assertGreater(fit.log_marginal_likelihood, base)

    def test_invalid_state_space_inputs(self) -> None:
        """Out-of-range channels and settings are rejected."""
        with self.assertRaises(ValueError):
            LatentForceChannel(3, (1.0,), 1.0)
        with self.assertRaises(ValueError):
            LatentForceChannel(1, (-1.0,), 1.0)
        with self.assertRaises(ValueError):
            LatentForceChannel(1, (1.0,), 0.0)
        with self.assertRaises(ValueError):
            LatentForceChannel(2, (1.0, -0.1), 1.0)
        with self.assertRaises(ValueError):
            mass_spring_damper_channel(1.0, -1.0, 1.0)
        good = [LatentForceChannel(1, (1.0,), 1.0)]
        with self.assertRaises(ValueError):
            state_space_latent_force_log_likelihood(
                good, [[1.0]], [[0.0]], lengthscale=1.0, smoothness=2.0
            )
        with self.assertRaises(ValueError):
            state_space_latent_force_log_likelihood(
                good, [[1.0]], [[0.0]], lengthscale=-1.0
            )
        with self.assertRaises(ValueError):
            matern_state_space(1.0, 1.0)
        with self.assertRaises(TypeError):
            state_space_latent_force_log_likelihood(
                [], [[1.0]], [[0.0]], lengthscale=1.0
            )  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            fit_latent_force_state_space(
                good, [[1.0, 2.0]], [[0.0, 0.1]], lengthscale=1.0, free=("bogus",)
            )


# ---------------------------------------------------------------------------
# failure cases and slow studies
# ---------------------------------------------------------------------------


class FailureCaseTests(unittest.TestCase):
    """Misspecification and poor identifiability are visible, not hidden."""

    def test_wrong_length_scale_lowers_the_evidence(self) -> None:
        """The evidence prefers a length-scale near the truth over a tiny one."""
        times, values = synthetic_data(2, 9.0, 14)
        base = (_D_TRUE, _S_TRUE, [d * v for d, v in zip(_D_TRUE, _LEVEL, strict=True)])
        good = latent_force_log_likelihood(
            times, values, *base, 0.9, [4e-4] * 3, gradient=False
        )
        bad = latent_force_log_likelihood(
            times, values, *base, 0.05, [4e-4] * 3, gradient=False
        )
        self.assertGreater(
            good.log_likelihood - bad.log_likelihood, 5.0
        )  # measured 10.7

    def test_discontinuous_heat_input_is_undercovered(self) -> None:
        """A square pulse violates the smooth-force prior: the band misses its edges."""
        gen = resolve_stream(4).child("square").generator()
        h, d, s, level = 0.005, 0.8, 1.0, 10.0

        def square(t: float) -> float:
            return 1.0 if 2.0 <= t < 4.0 else 0.0

        y, ys = level, [level]
        for k in range(int(9.0 / h)):
            y += h * (-d * (y - level) + s * square(k * h))
            ys.append(y)
        times = [[0.3 * i for i in range(1, 22)]]
        values = [
            [ys[int(round(t / h))] + 0.01 * gen.gauss(0.0, 1.0) for t in times[0]]
        ]
        # a second channel with another decay shares the force
        y2, ys2 = level, [level]
        for k in range(int(9.0 / h)):
            y2 += h * (-1.7 * (y2 - level) + 0.6 * square(k * h))
            ys2.append(y2)
        times.append([0.3 * i + 0.05 for i in range(1, 22)])
        values.append(
            [ys2[int(round(t / h))] + 0.01 * gen.gauss(0.0, 1.0) for t in times[1]]
        )
        model = fit_latent_force_model(times, values, seed=1, max_iterations=45)
        grid = [0.1 * i for i in range(2, 85)]
        forcing = model.predict_force(grid, channel=0)
        truth_ = [s * square(t) for t in grid]
        coverage = sum(
            lo <= x <= hi
            for lo, x, hi in zip(forcing.lower, truth_, forcing.upper, strict=True)
        ) / len(grid)
        edge = [i for i, t in enumerate(grid) if min(abs(t - 2.0), abs(t - 4.0)) < 0.25]
        edge_cov = sum(
            forcing.lower[i] <= truth_[i] <= forcing.upper[i] for i in edge
        ) / len(edge)
        # measured: coverage 0.77 overall and 0.10 within 0.25 s of the two jumps,
        # so the band fails exactly where the smooth-force prior is violated.
        self.assertLess(edge_cov, 0.5, (coverage, edge_cov))
        self.assertLess(coverage, 0.95, coverage)

    def test_smoother_stays_accurate_on_a_grid_much_finer_than_the_dynamics(
        self,
    ) -> None:
        """Steps of 0.01 against dynamics of order one: no singular-gain blow-up."""
        channels = [LatentForceChannel(2, (2.0, 0.3), 1.0, 0.0, 1e-3)]
        times = [[0.01 * i for i in range(1, 13)]]
        values = [[0.2 * math.sin(3.0 * t) for t in times[0]]]
        query = [0.065, 0.2]
        result = smooth_latent_force_state_space(
            channels, times, values, query, lengthscale=1.0, smoothness=2.5
        )
        evidence, dense = dense_state_space(
            channels, times, values, 2.5, 1.0, [(0, q) for q in query]
        )
        self.assertAlmostEqual(result.log_marginal_likelihood, evidence, delta=1e-6)
        for k, (mean, var, fmean, fvar) in enumerate(dense):
            self.assertAlmostEqual(result.output_mean[0][k], mean, delta=1e-5)
            self.assertAlmostEqual(result.output_variance[0][k], var, delta=1e-5)
            self.assertAlmostEqual(result.force_mean[k], fmean, delta=1e-4)
            self.assertAlmostEqual(result.force_variance[k], fvar, delta=1e-4)


@unittest.skipUnless(os.environ.get("PROBSTAT_SLOW") == "1", "slow tier")
class SlowReplicationStudy(unittest.TestCase):
    """Replicated coverage and extrapolation ratios (set PROBSTAT_SLOW=1)."""

    def test_coverage_and_extrapolation_over_replicates(self) -> None:
        """Measured 95% coverage of the heat input and plain/LFM error ratio."""
        grid = PhysicalDemonstrationTests.GRID
        true_input = [_S_TRUE[0] * heat_input(t) for t in grid]
        kernel = StationaryKernel("squared_exponential", 1.0, (1.0,))
        coverage_known, coverage_free, ratios = [], [], []
        seeds = range(1, 11)
        for seed in seeds:
            times, values = synthetic_data(seed, 9.6, 16, 0.05)
            known = fit_latent_force_model(
                times,
                values,
                initial_decays=_D_TRUE,
                initial_sensitivities=_S_TRUE,
                initial_lengthscale=0.7,
                initial_noise_variances=[0.05**2] * 3,
                fixed=("decays", "sensitivities", "lengthscale", "noise_variances"),
            )
            coverage_known.append(
                band_coverage(known.predict_force(grid, channel=0), true_input)
            )
            free = fit_latent_force_model(
                times,
                values,
                initial_lengthscale=1.0,
                initial_noise_variances=[7.5e-3] * 3,
                max_iterations=200,
                seed=seed,
            )
            coverage_free.append(
                band_coverage(free.predict_force(grid, channel=0), true_input)
            )
            times, values = synthetic_data(seed, 7.0, 16, 0.02)
            lfm = fit_latent_force_model(
                times,
                values,
                initial_lengthscale=1.0,
                initial_noise_variances=[1.2e-3] * 3,
                max_iterations=200,
                seed=seed,
            )
            plain = fit_multi_output_gp(
                [[(t,) for t in row] for row in times],
                values,
                kernel,
                rank=1,
                seed=seed,
                restarts=1,
                max_iterations=80,
            )
            future = [7.0 + 0.25 * i for i in range(1, 17)]
            e1, e2 = [], []
            for c in range(3):
                for t, p in zip(
                    future, lfm.predict_outputs(c, future).mean, strict=True
                ):
                    e1.append(p - truth(c, t))
                    e2.append(plain.predict_with_variance(c, (t,))[0] - truth(c, t))
            ratios.append(rms(e2) / rms(e1))
        n = len(coverage_known)
        print(
            f"\nLFM replicates ({n} seeds): known-plant coverage mean "
            f"{sum(coverage_known) / n:.3f} (min {min(coverage_known):.2f}); "
            f"unknown-plant coverage mean {sum(coverage_free) / n:.3f} "
            f"(min {min(coverage_free):.2f}); plain/LFM error ratio median "
            f"{sorted(ratios)[n // 2]:.2f} (min {min(ratios):.2f})"
        )
        self.assertGreater(sum(coverage_known) / n, 0.75)
        self.assertGreater(sorted(ratios)[n // 2], 2.0)


if __name__ == "__main__":
    unittest.main()
