"""Tests for the Aristoff-Bangerth Bayesian PDE-coefficient benchmark.

Published reference values used here:

* Aristoff and Bangerth, SIAM Review 65(4), 1074-1105 (2023), arXiv:2102.07263:
  Table 2.1 (``z_hat``) and Table 3.1 (posterior means with 2-sigma
  uncertainties), both transcribed in the module under test.
* The reference implementation's test vectors (deal.II code gallery
  MCMC-Laplace, ``testing/input.N.txt`` and ``testing/output.N.*.txt``,
  N = 0, 1, 2, 3), transcribed below. They were read from the published files;
  every transcribed number is compared to an independent computation, so a
  transcription error would show up as a failure.

No NumPy is used. Independent oracles: the closed Fourier series of
``-Delta u = 1`` on the unit square, a test-side dense assembly with 2 x 2 Gauss
quadrature solved by ``probstat.linalg.solve_spd`` and ``conjugate_gradient``.
"""

from __future__ import annotations

import math
import operator
import os
import random
import unittest

from probstat.inverse import benchmark_aristoff_bangerth as ab
from probstat.inverse.gauss_newton import check_derivatives
from probstat.linalg import CallbackOperator, conjugate_gradient, solve_spd

# fmt: off
# Reference test vector N=2: theta_k = k + 1 (k = 0..63); published z, 13 values
# per line, in the reference code's point ordering k = 13 (i - 1) + (j - 1).
_Z_THETA_1_TO_64 = (
    2.3014200637645553e-02, 1.7227013822907315e-02, 1.4997159065044461e-02,
    1.2732295324286869e-02, 1.1071818603171206e-02, 9.6561117538451827e-03,
    8.3293065102625404e-03, 7.2454131014024360e-03, 6.1314600311825774e-03,
    5.1061058867511583e-03, 4.0326569663379740e-03, 2.8953572878028491e-03,
    1.6370398837715770e-03,
    2.6708208929497726e-02, 2.7944310482837312e-02, 2.5429076591042966e-02,
    2.2287949636012116e-02, 1.9631062207236587e-02, 1.7277651843540597e-02,
    1.4971059542138733e-02, 1.3036771373342694e-02, 1.1022768384241193e-02,
    9.1340542987670172e-03, 7.1564541161115635e-03, 5.0507024320983352e-03,
    2.7702326678094003e-03,
    2.7463514635568793e-02, 3.3650785208337752e-02, 3.1883930619333090e-02,
    2.8809812693317820e-02, 2.5753839970316349e-02, 2.2894151768648129e-02,
    1.9952848804332998e-02, 1.7415587740170685e-02, 1.4729104934799640e-02,
    1.2171535401625464e-02, 9.4846222172053924e-03, 6.6235041212969643e-03,
    3.5753426638291271e-03,
    2.6888255504848874e-02, 3.6364432025673257e-02, 3.5613786980606923e-02,
    3.2972014453121631e-02, 2.9871773995108340e-02, 2.6787215403499223e-02,
    2.3480146953028833e-02, 2.0542216968954165e-02, 1.7387757680182615e-02,
    1.4345902966942247e-02, 1.1139837406813950e-02, 7.7298596594602784e-03,
    4.1341625423311421e-03,
    2.6284355454242896e-02, 3.7218751770994832e-02, 3.7333300114380734e-02,
    3.5223524104983271e-02, 3.2277875329490238e-02, 2.9166626407048892e-02,
    2.5699424239376219e-02, 2.2541799671195323e-02, 1.9104240739944038e-02,
    1.5755021277640925e-02, 1.2211522736296509e-02, 8.4433095505536890e-03,
    4.4929476669573563e-03,
    2.5064667652300868e-02, 3.6715641310792774e-02, 3.7570790263843468e-02,
    3.5978345760102573e-02, 3.3276772990434217e-02, 3.0259950734366731e-02,
    2.6787667514608239e-02, 2.3552762137968815e-02, 1.9993542426860987e-02,
    1.6493291177240157e-02, 1.2776830683002767e-02, 8.8202712544533692e-03,
    4.6814748026491621e-03,
    2.3546766181329545e-02, 3.5242800544445965e-02, 3.6623585685675530e-02,
    3.5479225214949679e-02, 3.3058261125005921e-02, 3.0215608001459310e-02,
    2.6854252752428948e-02, 2.3665487364281115e-02, 2.0125856247858134e-02,
    1.6618469264604030e-02, 1.2880310552003536e-02, 8.8914969048305235e-03,
    4.7166175761715230e-03,
    2.1841976199943859e-02, 3.3021357601411394e-02, 3.4684479122943354e-02,
    3.3854996261712914e-02, 3.1707048025104878e-02, 2.9084306946338394e-02,
    2.5927675071565692e-02, 2.2894438370897892e-02, 1.9507781749856169e-02,
    1.6132973987950234e-02, 1.2522742511270694e-02, 8.6573243438508897e-03,
    4.5983313052632303e-03,
    1.9709708126617209e-02, 3.0017888102359273e-02, 3.1791182770036161e-02,
    3.1191242991098829e-02, 2.9310306001339430e-02, 2.6951047789611160e-02,
    2.4082904095481125e-02, 2.1302900978082292e-02, 1.8190463808415561e-02,
    1.5075588458033537e-02, 1.1731721896174052e-02, 8.1348735663462398e-03,
    4.3346603967061390e-03,
    1.7402048823937560e-02, 2.6404284863922198e-02, 2.8038224581786043e-02,
    2.7540163275171864e-02, 2.5908891821729271e-02, 2.3844665346365236e-02,
    2.1337008588293564e-02, 1.8902776634981491e-02, 1.6176702670763047e-02,
    1.3444506987175826e-02, 1.0501014629550966e-02, 7.3170659624135917e-03,
    3.9215397428044325e-03,
    1.4559943542337656e-02, 2.1892649905029177e-02, 2.3229340431949561e-02,
    2.2763480901932191e-02, 2.1393101910591353e-02, 1.9678041170318027e-02,
    1.7620903192966279e-02, 1.5629556584529580e-02, 1.3408922506495278e-02,
    1.1184503150189687e-02, 8.7814862127284857e-03, 6.1654504994311070e-03,
    3.3360701946171770e-03,
    1.1151740025635435e-02, 1.6320706888334854e-02, 1.7159159605348567e-02,
    1.6698215885536886e-02, 1.5640823317740388e-02, 1.4360167247279459e-02,
    1.2858147034570715e-02, 1.1418911615769106e-02, 9.8239359415033863e-03,
    8.2331515577853866e-03, 6.5108338063131627e-03, 4.6245701643421100e-03,
    2.5443771299655492e-03,
    6.7802262094667116e-03, 9.3444306619875905e-03, 9.6483294861282320e-03,
    9.2502862549698472e-03, 8.6158956402762436e-03, 7.8750438598441906e-03,
    7.0440463974507077e-03, 6.2595162996001933e-03, 5.3992125154338003e-03,
    4.5514858284377326e-03, 3.6311515955564605e-03, 2.6246503174458284e-03,
    1.4879514379259684e-03,
)

# Reference test vector N=3 (random-looking, non-symmetric theta).
_THETA_3 = (
    7.424464300151675, 5.146949392577792, 0.4552998219577289, 0.7562929882343709,
    3.435713677963932, 3.346880874881098, 0.167400511173001, 1.672479474792017,
    0.1106945670253823, 0.6454462101875653, 9.497768918283253, 0.2726947271251607,
    1.292187865731028, 1.116286443377628, 2.946758821388506, 0.3016840921758895,
    2.992302682273326, 0.186815657305601, 5.446801714025209, 0.607052220581344,
    0.259998934918923, 3.3082122929957, 0.1302224148858491, 9.363310448960325,
    1.425788538242307, 3.560914244377736, 1.497533565470953, 0.3294871972590255,
    0.9981552096666537, 0.1337600156398106, 1.420495164848728, 4.397783113634144,
    0.1007186113793778, 1.712302239891422, 0.3992239619722109, 0.3034171701462479,
    0.9078517204015203, 0.1482097249767184, 0.3272030484280278, 0.1245993231538066,
    1.06393855505493, 8.955698553537829, 0.2453988949471052, 2.11156977866123,
    1.644616917593078, 1.742311152601188, 5.235343015375949, 5.386761161752736,
    4.102318040972229, 2.613728057513296, 0.1430899484797824, 0.4479741827114062,
    0.2520372829306234, 4.397166459926218, 7.833567043979766, 0.1292401950332613,
    2.697939357849716, 3.338720443400087, 2.824351939194429, 0.2018562308080854,
    0.1905391753352444, 0.3683767518794504, 0.7725756720768329, 0.2858988978666839,
)
_THETA_3_FIRST_Z = (
    1.8422836885793596e-02, 9.2657014369368101e-02, 1.7765147474190340e-01,
)

# fmt: on

_FAST_MODEL = ab.AristoffBangerthForward(1)  # 8 x 8 mesh, 49 unknowns


def _model() -> ab.AristoffBangerthForward:
    """Return the benchmark (32 x 32) model shared by the fast tests."""
    return ab.AristoffBangerthForward()


def _random_theta(seed: int) -> tuple[float, ...]:
    """Return a log-uniform positive parameter vector."""
    rng = random.Random(seed)
    return tuple(math.exp(rng.uniform(-1.5, 1.5)) for _ in range(64))


def _transpose_cells(theta: tuple[float, ...]) -> tuple[float, ...]:
    """Return theta with the cell grid reflected about the diagonal."""
    return tuple(theta[8 * c + r] for r in range(8) for c in range(8))


def _transpose_points(z: tuple[float, ...]) -> tuple[float, ...]:
    """Return z with the 13 x 13 point grid reflected about the diagonal."""
    return tuple(z[13 * j + i] for i in range(13) for j in range(13))


def _series_poisson_grid(terms: int = 99) -> list[float]:
    """Return ``10 u`` of ``-Delta u = 1`` (Fourier series) at the 169 points.

    Ordering ``k = 13 (i - 1) + (j - 1)``, ``x = i / 14``, ``y = j / 14``; the
    series is ``sum 16 sin(m pi x) sin(n pi y) / (pi^4 m n (m^2 + n^2))`` over
    odd ``m, n``. Truncation error at ``terms = 99`` is about 1e-6.
    """
    odd = range(1, terms + 1, 2)
    sines = [[math.sin(m * math.pi * i / 14.0) for m in odd] for i in range(1, 14)]
    out = []
    for i in range(13):
        for j in range(13):
            total = 0.0
            for a, m in enumerate(odd):
                for b, n in enumerate(odd):
                    coeff = 16.0 / (math.pi**4 * m * n * (m * m + n * n))
                    total += coeff * sines[i][a] * sines[j][b]
            out.append(10.0 * total)
    return out


def _dense_reference_solution(
    theta: tuple[float, ...], refine: int, *, use_cg: bool
) -> tuple[float, ...]:
    """Independent dense Q1 solve (test-side assembly, Gauss-quadrature stiffness).

    The local stiffness matrix is computed by 2 x 2 Gauss quadrature of
    ``grad phi_i . grad phi_j`` on a square cell, not from the closed form used
    in the module; the system is solved by ``solve_spd`` or CG from
    ``probstat.linalg`` and observed with bilinear interpolation.
    """
    mesh = 8 * refine
    h = 1.0 / mesh
    side = mesh - 1
    gauss = (0.5 - 0.5 / math.sqrt(3.0), 0.5 + 0.5 / math.sqrt(3.0))
    corners = ((0, 0), (1, 0), (1, 1), (0, 1))
    local = [[0.0] * 4 for _ in range(4)]
    for a, (ax, ay) in enumerate(corners):
        for b, (bx, by) in enumerate(corners):
            acc = 0.0
            for s in gauss:
                for t in gauss:
                    # d/dx phi = sign_x (1 - or s)/h ... on the unit cell
                    gax = (1 if ax else -1) * (t if ay else 1 - t) / h
                    gay = (1 if ay else -1) * (s if ax else 1 - s) / h
                    gbx = (1 if bx else -1) * (t if by else 1 - t) / h
                    gby = (1 if by else -1) * (s if bx else 1 - s) / h
                    acc += 0.25 * (gax * gbx + gay * gby) * h * h
            local[a][b] = acc

    def idx(i: int, j: int) -> int:
        if 1 <= i <= side and 1 <= j <= side:
            return (j - 1) * side + (i - 1)
        return -1

    n = side * side
    matrix = [[0.0] * n for _ in range(n)]
    for cj in range(mesh):
        for ci in range(mesh):
            coeff = theta[8 * (cj // refine) + ci // refine]
            nodes = [idx(ci + dx, cj + dy) for dx, dy in corners]
            for a in range(4):
                for b in range(4):
                    if nodes[a] >= 0 and nodes[b] >= 0:
                        matrix[nodes[a]][nodes[b]] += coeff * local[a][b]
    rhs = tuple(10.0 * h * h for _ in range(n))
    if use_cg:
        op = CallbackOperator(
            (n, n),
            lambda v: tuple(sum(map(operator.mul, row, v)) for row in matrix),
            lambda v: tuple(sum(map(operator.mul, row, v)) for row in matrix),
        )
        u = conjugate_gradient(op, rhs, tolerance=1e-13, max_iterations=4 * n).solution
    else:
        u = solve_spd(tuple(tuple(r) for r in matrix), rhs)
    out = []
    for ix in range(1, 14):
        for jy in range(1, 14):
            gx, gy = ix * mesh / 14.0, jy * mesh / 14.0
            ci, cj = int(gx), int(gy)
            sx, sy = gx - ci, gy - cj
            val = 0.0
            for (dx, dy), w in zip(
                corners,
                (
                    (1 - sx) * (1 - sy),
                    sx * (1 - sy),
                    sx * sy,
                    (1 - sx) * sy,
                ),
                strict=True,
            ):
                k = idx(ci + dx, cj + dy)
                if k >= 0:
                    val += w * u[k]
            out.append(val)
    return tuple(out)


class ReferenceDataTests(unittest.TestCase):
    """Constants and published values of the paper and its reference code."""

    def test_shapes_and_constants(self) -> None:
        """Check shapes and constants."""
        self.assertEqual(len(ab.TRUE_DATA), 169)
        self.assertEqual(len(ab.REFERENCE_POSTERIOR_MEAN), 64)
        self.assertEqual(len(ab.REFERENCE_POSTERIOR_MEAN_TWO_SIGMA), 64)
        self.assertEqual(ab.NOISE_STD, 0.05)
        self.assertEqual(ab.PRIOR_LOG_STD, 2.0)
        self.assertEqual(ab.FORCING, 10.0)
        self.assertEqual(_model().mesh_cells_per_side, 32)

    def test_table_2_1_is_diagonal_symmetric(self) -> None:
        """Check table 2 1 is diagonal symmetric."""
        # The paper (footnote 2) notes z_hat is symmetric about the diagonal.
        z = ab.TRUE_DATA
        for a, b in zip(z, _transpose_points(z), strict=True):
            # Table 2.1 digits carry the reference solver's noise (about 1e-13).
            self.assertAlmostEqual(a, b, delta=1e-10)
        self.assertAlmostEqual(z[0], 0.06076511762259369, places=15)
        self.assertAlmostEqual(max(z), 0.9439154625527653, places=15)

    def test_table_3_1_means_are_nearly_diagonal_symmetric(self) -> None:
        """Check table 3 1 means are nearly diagonal symmetric."""
        # Paper footnote 2: <theta_7> should be close to <theta_63>... and in
        # general the posterior mean is symmetric. Measured: every pair agrees
        # within 3 combined 2-sigma units EXCEPT the pair (1, 8) (1.2104 vs
        # 1.119, ratio about 6.3), which is a property of the published table.
        mean = ab.REFERENCE_POSTERIOR_MEAN
        err = ab.REFERENCE_POSTERIOR_MEAN_TWO_SIGMA
        for r in range(8):
            for c in range(r + 1, 8):
                k1, k2 = 8 * r + c, 8 * c + r
                ratio = abs(mean[k1] - mean[k2]) / math.hypot(err[k1], err[k2])
                if (k1, k2) == (1, 8):
                    self.assertGreater(ratio, 3.0)
                else:
                    self.assertLess(ratio, 3.0, msg=f"pair {k1},{k2}")

    def test_published_forward_values_theta_one(self) -> None:
        """Check published forward values theta one."""
        # Reference output.0.z.txt (theta = 1): corner, edge-mid and centre.
        z = _model().value((1.0,) * 64)
        self.assertAlmostEqual(z[0], 7.6937775561651600e-02, delta=2e-10)
        self.assertAlmostEqual(z[84], 7.3728116929320731e-01, delta=2e-10)
        self.assertAlmostEqual(z[6], 2.1596339851534335e-01, delta=2e-10)
        # output.1: theta = 10 gives z / 10 exactly (published 7.6937775560650644e-03).
        z10 = _model().value((10.0,) * 64)
        self.assertAlmostEqual(z10[0], 7.6937775560650644e-03, delta=2e-11)
        self.assertAlmostEqual(z10[84], 7.3728116929293125e-02, delta=2e-11)

    def test_published_forward_vector_theta_1_to_64(self) -> None:
        """Check published forward vector theta 1 to 64."""
        # Reference output.2.z.txt: all 169 values; also fixes the point
        # ordering k = 13 (i - 1) + (j - 1) and the cell numbering.
        theta = tuple(float(k + 1) for k in range(64))
        z = _model().value(theta)
        worst = max(abs(a - b) for a, b in zip(z, _Z_THETA_1_TO_64, strict=True))
        # Published values carry solver noise of about 1e-11.
        self.assertLess(worst, 5e-10, msg=f"max abs difference {worst:.3e}")
        # The transposed point ordering must NOT match (non-symmetric theta).
        swapped = max(
            abs(a - b)
            for a, b in zip(_transpose_points(z), _Z_THETA_1_TO_64, strict=True)
        )
        self.assertGreater(swapped, 1e-3)

    def test_published_log_likelihood_and_prior(self) -> None:
        """Check published log likelihood and prior."""
        # Reference output.N.loglikelihood / output.N.logprior (12 digits).
        theta2 = tuple(float(k + 1) for k in range(64))
        cases = (
            ((1.0,) * 64, -228.510844004, 0.0),
            ((10.0,) * 64, -5708.64422369, -42.4151848838),
            (theta2, -6412.82113914, -88.6599171951),
            (_THETA_3, -852.897808712, -14.9011148893),
        )
        for theta, loglik, logprior in cases:
            self.assertAlmostEqual(ab.log_likelihood(theta), loglik, delta=2e-8)
            self.assertAlmostEqual(ab.log_prior(theta), logprior, delta=2e-9)
            self.assertAlmostEqual(
                ab.log_posterior(theta), loglik + logprior, delta=3e-8
            )

    def test_published_forward_values_random_theta(self) -> None:
        """Check published forward values random theta."""
        z = _model().value(_THETA_3)
        for got, want in zip(z, _THETA_3_FIRST_Z, strict=False):
            self.assertAlmostEqual(got, want, delta=5e-10)

    def test_true_theta_reproduces_data_to_discretization_error(self) -> None:
        """Check true theta reproduces data to discretization error."""
        # z_hat came from a 256 x 256 Q3 model (no inverse crime). Measured
        # max |z(theta_hat) - z_hat| = 1.1e-2 (rms 2.9e-3), i.e. well below the
        # noise level 0.05 but clearly nonzero.
        z = _model().value(ab.TRUE_THETA)
        worst = max(abs(a - b) for a, b in zip(z, ab.TRUE_DATA, strict=True))
        self.assertLess(worst, 0.02)
        self.assertGreater(worst, 1e-3)


class ForwardModelTests(unittest.TestCase):
    """Closed-form, convergence, symmetry and solver-oracle checks."""

    def test_scale_covariance(self) -> None:
        """Check scale covariance."""
        # a -> c a gives u -> u / c exactly.
        theta = _random_theta(5)
        base = _model().value(theta)
        scaled = _model().value(tuple(3.7 * t for t in theta))
        for a, b in zip(base, scaled, strict=True):
            self.assertAlmostEqual(a / 3.7, b, delta=1e-13)

    def test_mesh_refinement_converges_at_second_order_to_fourier_series(self) -> None:
        """Check mesh refinement converges at second order to fourier series."""
        # Constant a = 1: u = 10 * series. Measured max-norm errors over the 169
        # points: 1.69e-2 (8x8), 3.27e-3 (16x16), 9.04e-4 (32x32).
        exact = _series_poisson_grid()
        self.assertAlmostEqual(exact[84], 0.73671353, delta=3e-6)
        errors = []
        for refine in (1, 2, 4):
            z = ab.AristoffBangerthForward(refine).value((1.0,) * 64)
            errors.append(max(abs(a - b) for a, b in zip(z, exact, strict=True)))
        self.assertGreater(errors[0] / errors[1], 3.0)
        self.assertGreater(errors[1] / errors[2], 3.0)
        self.assertLess(errors[2], 1.2e-3)

    def test_symmetry_about_the_diagonal(self) -> None:
        """Check symmetry about the diagonal."""
        model = _model()
        theta = _random_theta(7)
        z = model.value(theta)
        zt = model.value(_transpose_cells(theta))
        for a, b in zip(_transpose_points(z), zt, strict=True):
            self.assertAlmostEqual(a, b, delta=1e-13)
        symmetric = model.value(ab.TRUE_THETA)
        for a, b in zip(symmetric, _transpose_points(symmetric), strict=True):
            self.assertAlmostEqual(a, b, delta=1e-13)

    def test_banded_solver_matches_dense_solve_spd(self) -> None:
        """Check banded solver matches dense solve spd."""
        theta = _random_theta(11)
        got = _FAST_MODEL.value(theta)
        want = _dense_reference_solution(theta, 1, use_cg=False)
        for a, b in zip(got, want, strict=True):
            self.assertAlmostEqual(a, b, delta=1e-12)

    def test_banded_solver_matches_conjugate_gradient(self) -> None:
        """Check banded solver matches conjugate gradient."""
        # 16 x 16 mesh (225 unknowns), strong contrast 1e-1 ... 1e1.
        rng = random.Random(3)
        theta = tuple(10.0 ** rng.uniform(-1, 1) for _ in range(64))
        got = ab.AristoffBangerthForward(2).value(theta)
        want = _dense_reference_solution(theta, 2, use_cg=True)
        for a, b in zip(got, want, strict=True):
            self.assertAlmostEqual(a, b, delta=1e-9 * max(1.0, abs(b)))

    def test_cache_reuses_factorization(self) -> None:
        """Check cache reuses factorization."""
        model = ab.AristoffBangerthForward(1)
        theta = _random_theta(1)
        model.value(theta)
        model.jvp(theta, (1.0,) * 64)
        model.vjp(theta, (1.0,) * 169)
        self.assertEqual(model.factorizations, 1)
        model.value(tuple(2.0 * t for t in theta))
        self.assertEqual(model.factorizations, 2)

    def test_validation(self) -> None:
        """Check validation."""
        model = _FAST_MODEL
        with self.assertRaises(ValueError):
            model.value((1.0,) * 63)
        with self.assertRaises(ValueError):
            model.value((1.0,) * 63 + (0.0,))
        with self.assertRaises(ValueError):
            model.value((1.0,) * 63 + (-1.0,))
        with self.assertRaises(ValueError):
            model.value((1.0,) * 63 + (math.nan,))
        with self.assertRaises(TypeError):
            model.value((1.0,) * 63 + (True,))
        with self.assertRaises(TypeError):
            model.value("1" * 64)
        with self.assertRaises(ValueError):
            model.jvp((1.0,) * 64, (1.0,) * 10)
        with self.assertRaises(ValueError):
            model.vjp((1.0,) * 64, (math.inf,) + (0.0,) * 168)
        with self.assertRaises(ValueError):
            ab.AristoffBangerthForward(0)
        with self.assertRaises(TypeError):
            ab.AristoffBangerthForward(2.0)  # type: ignore[arg-type]

    def test_extreme_contrast_reports_failure_instead_of_hiding_it(self) -> None:
        """Check extreme contrast reports failure instead of hiding it."""
        theta = (1.0,) * 63 + (1e300,)
        z = _FAST_MODEL.value(theta)
        self.assertTrue(
            all(math.isfinite(v) for v in z) or all(math.isnan(v) for v in z)
        )
        if all(math.isnan(v) for v in z):
            self.assertEqual(ab.log_likelihood(theta, forward=_FAST_MODEL), -math.inf)
            with self.assertRaises(ValueError):
                _FAST_MODEL.jvp(theta, (1.0,) * 64)


class DerivativeTests(unittest.TestCase):
    """The exact discrete adjoint and the posterior gradient."""

    def test_adjoint_dot_test(self) -> None:
        """Check adjoint dot test."""
        model = _model()
        theta = _random_theta(21)
        rng = random.Random(22)
        v = tuple(rng.gauss(0, 1) for _ in range(64))
        w = tuple(rng.gauss(0, 1) for _ in range(169))
        lhs = sum(a * b for a, b in zip(w, model.jvp(theta, v), strict=True))
        rhs = sum(a * b for a, b in zip(model.vjp(theta, w), v, strict=True))
        self.assertAlmostEqual(lhs, rhs, delta=1e-12 * (abs(lhs) + 1.0))

    def test_check_derivatives_passes(self) -> None:
        """Check check derivatives passes."""
        theta = _random_theta(23)
        rng = random.Random(24)
        v = tuple(rng.gauss(0, 1) * t for t in theta)
        w = tuple(rng.gauss(0, 1) for _ in range(169))
        report = check_derivatives(_model(), theta, v, w, dot_tolerance=1e-10)
        self.assertTrue(report.adjoint_passed)
        self.assertTrue(report.taylor_passed)

    def test_jvp_matches_central_difference(self) -> None:
        """Check jvp matches central difference."""
        model = _model()
        theta = _random_theta(25)
        rng = random.Random(26)
        v = tuple(rng.gauss(0, 1) * t for t in theta)
        eps = 1e-5
        plus = model.value(tuple(t + eps * d for t, d in zip(theta, v, strict=True)))
        minus = model.value(tuple(t - eps * d for t, d in zip(theta, v, strict=True)))
        analytic = model.jvp(theta, v)
        scale = max(abs(a) for a in analytic)
        for p, m, a in zip(plus, minus, analytic, strict=True):
            # Central difference error ~ eps^2 * third derivative ~ 1e-10.
            self.assertAlmostEqual((p - m) / (2 * eps), a, delta=1e-7 * scale)

    def test_jvp_of_scaling_direction(self) -> None:
        """Check jvp of scaling direction."""
        # d z(c theta)/dc at c = 1 equals -z(theta) (homogeneity of degree -1).
        model = _model()
        theta = _random_theta(27)
        jv = model.jvp(theta, theta)
        for a, b in zip(jv, model.value(theta), strict=True):
            self.assertAlmostEqual(a, -b, delta=1e-12)

    def test_posterior_gradient_matches_finite_differences(self) -> None:
        """Check posterior gradient matches finite differences."""
        theta = _random_theta(31)
        value, grad = ab.log_posterior_gradient(theta)
        self.assertAlmostEqual(value, ab.log_posterior(theta), delta=1e-9)
        for k in (0, 9, 27, 63):
            eps = 1e-6 * theta[k]
            up = list(theta)
            up[k] += eps
            down = list(theta)
            down[k] -= eps
            fd = (ab.log_posterior(up) - ab.log_posterior(down)) / (2 * eps)
            self.assertAlmostEqual(fd, grad[k], delta=1e-5 * (abs(grad[k]) + 1.0))

    def test_gradient_reduces_to_prior_term_at_zero_misfit(self) -> None:
        """Check gradient reduces to prior term at zero misfit."""
        # With data equal to the model output the likelihood gradient is zero and
        # only the prior gradient -ln(theta)/(sigma_pr^2 theta) remains.
        theta = _random_theta(33)
        data = _FAST_MODEL.value(theta)
        _, grad = ab.log_posterior_gradient(theta, data=data, forward=_FAST_MODEL)
        for g, t in zip(grad, theta, strict=True):
            self.assertAlmostEqual(g, -math.log(t) / (4.0 * t), delta=1e-9)


class PriorAndLikelihoodTests(unittest.TestCase):
    """Closed forms of Eq. (2.5) and (2.6) and the pCN coordinates."""

    def test_log_prior_closed_form(self) -> None:
        """Check log prior closed form."""
        theta = (math.e,) * 64
        self.assertAlmostEqual(ab.log_prior(theta), -64.0 / 8.0, places=12)
        self.assertEqual(ab.log_prior((1.0,) * 64), 0.0)

    def test_prior_in_log_coordinates_has_mean_sigma_squared(self) -> None:
        """Check prior in log coordinates has mean sigma squared."""
        # (2.6) as a density in theta: in t = ln theta the density is
        # N(4, 4), so E[theta] = exp(4 + 2) = 403.43 as stated in the paper.
        self.assertAlmostEqual(math.exp(ab.PRIOR_LOG_MEAN + 2.0), 403.4287934927351)
        # Numerical check: integral of theta pi(theta) dtheta / integral pi.
        num = den = 0.0
        step = 5e-3
        for i in range(-6000, 6001):
            t = i * step
            w = math.exp(-t * t / 8.0 + t)  # pi(e^t) * Jacobian e^t
            num += w * math.exp(t)
            den += w
        self.assertAlmostEqual(num / den, 403.4287934927351, delta=1e-6)

    def test_theta_from_standard_normal(self) -> None:
        """Check theta from standard normal."""
        theta = ab.theta_from_standard_normal((0.0,) * 64)
        self.assertEqual(theta, (math.exp(4.0),) * 64)
        with self.assertRaises(ValueError):
            ab.theta_from_standard_normal((0.0,) * 63)
        with self.assertRaises(ValueError):
            ab.theta_from_standard_normal((1000.0,) + (0.0,) * 63)
        self.assertEqual(
            ab.standardized_log_likelihood((1000.0,) + (0.0,) * 63), -math.inf
        )
        u = (0.3,) * 64
        self.assertAlmostEqual(
            ab.standardized_log_likelihood(u),
            ab.log_likelihood(ab.theta_from_standard_normal(u)),
            places=12,
        )

    def test_likelihood_zero_misfit(self) -> None:
        """Check likelihood zero misfit."""
        theta = _random_theta(41)
        data = _FAST_MODEL.value(theta)
        self.assertEqual(ab.log_likelihood(theta, data=data, forward=_FAST_MODEL), 0.0)
        shifted = tuple(d + 0.05 for d in data)
        self.assertAlmostEqual(
            ab.log_likelihood(theta, data=shifted, forward=_FAST_MODEL),
            -169 * 0.5,
            places=10,
        )


class SamplerTests(unittest.TestCase):
    """The paper's Metropolis-Hastings sampler (fast checks on the 8 x 8 mesh)."""

    def test_reproducible_and_bookkeeping(self) -> None:
        """Check reproducible and bookkeeping."""
        a = ab.metropolis_hastings(steps=12, thin=3, seed=5, forward=_FAST_MODEL)
        b = ab.metropolis_hastings(steps=12, thin=3, seed=5, forward=_FAST_MODEL)
        self.assertEqual(a, b)
        self.assertEqual(len(a.samples), 4)
        self.assertEqual(a.evaluations, 13)
        self.assertTrue(0.0 <= a.acceptance_rate <= 1.0)
        self.assertEqual(a.proposal_std, ab.PROPOSAL_STD)
        for lp, s in zip(a.log_posterior, a.samples, strict=True):
            self.assertAlmostEqual(
                lp, ab.log_posterior(s, forward=_FAST_MODEL), delta=1e-9
            )

    def test_jacobian_factor_gives_prior_stationary_law(self) -> None:
        """Check jacobian factor gives prior stationary law."""

        # Flat likelihood (the fake model returns the data itself): the target
        # is Eq. (2.6) alone, i.e. ln theta ~ N(4, 4) as a density in theta.
        # Without the factor prod theta~/theta the chain would centre at
        # ln theta = 0. Measured mean of ln theta over kept states is printed
        # by the failure message if the band is violated; the band (4 +- 0.3)
        # is about 10 standard errors wide and excludes the wrong value 0.
        class _Flat:
            def value(self, theta: object) -> tuple[float, ...]:
                return ab.TRUE_DATA

        chain = ab.metropolis_hastings(
            steps=12000,
            thin=5,
            proposal_std=0.5,
            seed=2024,
            forward=_Flat(),  # type: ignore[arg-type]
        )
        kept = chain.samples[400:]
        mean_log = sum(math.log(v) for s in kept for v in s) / (len(kept) * 64)
        self.assertAlmostEqual(
            mean_log, 4.0, delta=0.3, msg=f"mean ln theta {mean_log}"
        )
        self.assertGreater(chain.acceptance_rate, 0.05)

    def test_validation(self) -> None:
        """Check validation."""
        with self.assertRaises(ValueError):
            ab.metropolis_hastings(steps=0)
        with self.assertRaises(ValueError):
            ab.metropolis_hastings(steps=1, thin=0)
        with self.assertRaises(TypeError):
            ab.metropolis_hastings(steps=1.5)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            ab.metropolis_hastings(steps=1, proposal_std=0.0)
        with self.assertRaises(ValueError):
            ab.metropolis_hastings(steps=1, initial=(1.0,) * 63)


@unittest.skipUnless(os.environ.get("PROBSTAT_SLOW") == "1", "slow tier")
class SlowBenchmarkTests(unittest.TestCase):
    """Slow tier: 64 x 64 mesh convergence and a short MCMC vs Table 3.1."""

    def test_finest_mesh_error_ratio(self) -> None:
        """Check finest mesh error ratio."""
        exact = _series_poisson_grid()
        errors = []
        for refine in (2, 4, 8):
            z = ab.AristoffBangerthForward(refine).value((1.0,) * 64)
            errors.append(max(abs(a - b) for a, b in zip(z, exact, strict=True)))
        self.assertGreater(errors[1] / errors[2], 3.0)

    def test_short_mcmc_against_published_means(self) -> None:
        """Compare a short run of the paper's sampler with Table 3.1.

        Two chains of 4000 proposals (seeds 11 and 12, thinned by 10, first
        150 kept states discarded). The published chains need about 1e4
        evaluations per independent sample, so this run is far too short for
        the heavy-tailed parameters (published mean above 2), whose means it
        underestimates by orders of the batch-means standard error; those are
        printed but not asserted. For the 36 parameters whose published mean is
        below 2 the standardized difference (run mean - published mean) / MCSE
        (MCSE from 16 batch means) was MEASURED as median |z| = 1.69, 24 of 36
        below 2 and 29 of 36 below 3 (maximum 7.2 at parameter 12). The
        assertions are loose guards derived from that measurement, not a fit.
        """
        chains = [
            ab.metropolis_hastings(steps=4000, thin=10, seed=seed) for seed in (11, 12)
        ]
        print("\n  acceptance:", [round(c.acceptance_rate, 3) for c in chains])
        burn, batches = 150, 8
        scores = []
        for k in range(64):
            means = []
            for chain in chains:
                series = [s[k] for s in chain.samples[burn:]]
                size = len(series) // batches
                means += [
                    sum(series[i * size : (i + 1) * size]) / size
                    for i in range(batches)
                ]
            mean = sum(means) / len(means)
            var = sum((m - mean) ** 2 for m in means) / (len(means) - 1)
            mcse = math.sqrt(var / len(means))
            ref = ab.REFERENCE_POSTERIOR_MEAN[k]
            z = (mean - ref) / mcse
            print(
                f"  theta_{k:2d}: run {mean:9.4f} +- {mcse:7.4f}  published "
                f"{ref:9.4f} +- {ab.REFERENCE_POSTERIOR_MEAN_TWO_SIGMA[k]:.4f}"
                f"  z = {z:7.2f}"
            )
            if ref < 2.0:
                scores.append(abs(z))
        scores.sort()
        print(
            f"  well-determined set: n = {len(scores)}, median |z| = "
            f"{scores[len(scores) // 2]:.2f}, below 2: "
            f"{sum(s < 2 for s in scores)}, below 3: {sum(s < 3 for s in scores)}"
        )
        self.assertEqual(len(scores), 36)
        self.assertLess(scores[len(scores) // 2], 3.0)
        self.assertGreaterEqual(sum(s < 3 for s in scores), len(scores) // 2)


if __name__ == "__main__":
    unittest.main()
