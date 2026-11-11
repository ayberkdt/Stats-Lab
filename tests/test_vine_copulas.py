"""Vine copulas: pair copulas, C- and D-vines, and sequential fitting.

Every oracle is independent of :mod:`probstat.modeling.vines`:

- h-functions are checked against central differences of the closed-form
  copula CDF and against the textbook Clayton h-function
  ``h(u | v) = v^(-theta-1) (u^-theta + v^-theta - 1)^(-1-1/theta)``;
- a Gaussian vine whose pair-copula parameters are the partial correlations
  of a correlation matrix ``R`` (each computed as
  ``-P_12 / sqrt(P_11 P_22)`` from the inverse ``P`` of the relevant
  sub-matrix of ``R``) must equal :class:`GaussianCopula` with ``R``;
- normalization and marginalization are checked by adaptive Gauss-Kronrod
  quadrature and a tensor Gauss-Legendre rule written in this file;
- the uniformity of the Rosenblatt transform of seeded samples is checked
  with tolerances of four or five standard errors.

No test imports numpy, scipy, or mpmath.
"""

from __future__ import annotations

import math
import unittest
from statistics import NormalDist

from probstat.distributions import LogNormal, Normal, Uniform
from probstat.linalg import inverse_spd
from probstat.modeling import (
    PAIR_COPULA_FAMILIES,
    ClaytonCopula,
    CVineCopula,
    DVineCopula,
    FrankCopula,
    GaussianCopula,
    GumbelCopula,
    IndependenceCopula,
    JointInput,
    PairCopula,
    ProbabilisticModel,
    RandomInput,
    StudentTCopula,
    fit_vine_copula,
    kendall_independence_p_value,
    kendall_tau,
)
from probstat.numerics import gauss_kronrod
from probstat.uq import QuantityOfInterest

_PHI = NormalDist()
_ONE_MINUS = 1.0 - 2.0**-53
P = PairCopula.from_parameters

ALL_PAIRS = (
    P("independence"),
    P("gaussian", (0.45,)),
    P("gaussian", (-0.6,)),
    P("student-t", (0.5, 4.0)),
    P("student-t", (-0.3, 7.5)),
    P("frank", (4.5,)),
    P("frank", (-3.0,)),
    *(P("clayton", (2.3,), rotation) for rotation in (0, 90, 180, 270)),
    *(P("gumbel", (1.7,), rotation) for rotation in (0, 90, 180, 270)),
)
POINTS = ((0.2, 0.7), (0.63, 0.31), (0.91, 0.86), (0.05, 0.12))


def _clamp(value: float) -> float:
    """Clamp a quadrature node into the open unit interval.

    References:
        [REF-NELSEN-2006, Chapter 2]. No equation is evaluated.

    """
    return min(max(value, 1e-300), _ONE_MINUS)


def _gauss_legendre(order: int) -> tuple[list[float], list[float]]:
    """Return Gauss-Legendre nodes and weights on ``[0, 1]`` by Newton's method.

    References:
        [REF-NIST-DLMF, Section 3.5]. No new equation is introduced.

    """
    nodes: list[float] = []
    weights: list[float] = []
    for i in range(1, order + 1):
        x = math.cos(math.pi * (i - 0.25) / (order + 0.5))
        derivative = 1.0
        for _ in range(100):
            previous, current = 1.0, x
            for k in range(2, order + 1):
                previous, current = (
                    current,
                    ((2 * k - 1) * x * current - (k - 1) * previous) / k,
                )
            derivative = order * (x * current - previous) / (x * x - 1.0)
            step = current / derivative
            x -= step
            if abs(step) < 1e-16:
                break
        nodes.append(0.5 * (x + 1.0))
        weights.append(1.0 / ((1.0 - x * x) * derivative * derivative))
    return nodes, weights


def _partial_correlation(
    matrix: tuple[tuple[float, ...], ...], i: int, j: int, given: tuple[int, ...]
) -> float:
    """Return the partial correlation of ``i`` and ``j`` given ``given``.

    ``rho_{ij;K} = -P_12 / sqrt(P_11 P_22)`` with ``P`` the inverse of the
    correlation sub-matrix on ``(i, j, K)``.

    References:
        [REF-KUROWICKA-COOKE-2006], [REF-BEDFORD-COOKE-2002].

    """
    index = (i, j, *given)
    sub = tuple(tuple(matrix[a][b] for b in index) for a in index)
    precision = inverse_spd(sub)
    return -precision[0][1] / math.sqrt(precision[0][0] * precision[1][1])


CORRELATION = (
    (1.0, 0.5, 0.3, 0.2),
    (0.5, 1.0, 0.4, 0.1),
    (0.3, 0.4, 1.0, 0.6),
    (0.2, 0.1, 0.6, 1.0),
)


def _gaussian_d_vine(order: tuple[int, ...]) -> DVineCopula:
    """Return the Gaussian D-vine with partial correlations of ``CORRELATION``.

    References:
        [REF-KUROWICKA-COOKE-2006], [REF-AAS-CZADO-FRIGESSI-BAKKEN-2009].

    """
    d = len(order)
    trees = tuple(
        tuple(
            P(
                "gaussian",
                (
                    _partial_correlation(
                        CORRELATION,
                        order[e],
                        order[e + t + 1],
                        tuple(order[e + 1 : e + t + 1]),
                    ),
                ),
            )
            for e in range(d - 1 - t)
        )
        for t in range(d - 1)
    )
    return DVineCopula(trees, order)


def _gaussian_c_vine(order: tuple[int, ...]) -> CVineCopula:
    """Return the Gaussian C-vine with partial correlations of ``CORRELATION``.

    References:
        [REF-KUROWICKA-COOKE-2006], [REF-AAS-CZADO-FRIGESSI-BAKKEN-2009].

    """
    d = len(order)
    trees = tuple(
        tuple(
            P(
                "gaussian",
                (
                    _partial_correlation(
                        CORRELATION, order[t], order[t + 1 + e], tuple(order[:t])
                    ),
                ),
            )
            for e in range(d - 1 - t)
        )
        for t in range(d - 1)
    )
    return CVineCopula(trees, order)


MIXED_TREES = (
    (P("clayton", (2.0,)), P("gumbel", (1.8,), 90), P("frank", (-3.0,))),
    (P("student-t", (-0.3, 5.0)), P("gumbel", (1.8,))),
    (P("clayton", (2.0,), 180),),
)


class PairCopulaTests(unittest.TestCase):
    """h-functions, inverses, rotations and margins of the building blocks."""

    def test_h_functions_are_cdf_derivatives(self) -> None:
        """``h_given_first = dC/du_1`` and ``h_given_second = dC/du_2``."""
        step = 1e-6
        for pair in ALL_PAIRS:
            if not pair.has_cdf:
                continue
            for u1, u2 in POINTS[:3]:
                with self.subTest(pair=pair.name, u=(u1, u2)):
                    first = (pair.cdf((u1 + step, u2)) - pair.cdf((u1 - step, u2))) / (
                        2.0 * step
                    )
                    second = (pair.cdf((u1, u2 + step)) - pair.cdf((u1, u2 - step))) / (
                        2.0 * step
                    )
                    self.assertAlmostEqual(
                        pair.h_given_first(u1, u2), first, delta=1e-7
                    )
                    self.assertAlmostEqual(
                        pair.h_given_second(u1, u2), second, delta=1e-7
                    )

    def test_clayton_h_function_closed_form(self) -> None:
        """The unrotated Clayton h-function equals the textbook expression."""
        theta = 2.3
        pair = P("clayton", (theta,))
        for u, v in POINTS:
            expected = v ** (-theta - 1.0) * (u**-theta + v**-theta - 1.0) ** (
                -1.0 - 1.0 / theta
            )
            self.assertAlmostEqual(pair.h_given_second(u, v), expected, delta=1e-13)
            self.assertAlmostEqual(pair.h_given_first(v, u), expected, delta=1e-13)

    def test_h_inverse_round_trip(self) -> None:
        """An h-function followed by its inverse returns the argument."""
        for pair in ALL_PAIRS:
            for u1, u2 in POINTS:
                with self.subTest(pair=pair.name, u=(u1, u2)):
                    w2 = pair.h_given_first(u1, u2)
                    self.assertAlmostEqual(
                        pair.h_inverse_given_first(u1, w2), u2, delta=1e-9
                    )
                    w1 = pair.h_given_second(u1, u2)
                    self.assertAlmostEqual(
                        pair.h_inverse_given_second(w1, u2), u1, delta=1e-9
                    )

    def test_rotated_density_is_reflected_base_density(self) -> None:
        """Rotated densities are the base density at the reflected point."""
        base = ClaytonCopula(2.3)
        for u1, u2 in POINTS:
            self.assertAlmostEqual(
                P("clayton", (2.3,), 90).log_density((u1, u2)),
                base.log_density((1.0 - u1, u2)),
                delta=1e-13,
            )
            self.assertAlmostEqual(
                P("clayton", (2.3,), 180).log_density((u1, u2)),
                base.log_density((1.0 - u1, 1.0 - u2)),
                delta=1e-13,
            )
            self.assertAlmostEqual(
                P("clayton", (2.3,), 270).log_density((u1, u2)),
                base.log_density((u1, 1.0 - u2)),
                delta=1e-13,
            )

    def test_tau_and_tail_dependence_of_rotations(self) -> None:
        """Rotations by 90/270 negate tau; 180 swaps the tail coefficients."""
        gumbel = GumbelCopula(1.7)
        tau = gumbel.kendall_tau()
        upper = gumbel.tail_dependence()[1]
        self.assertAlmostEqual(P("gumbel", (1.7,), 90).kendall_tau(), -tau, delta=1e-15)
        self.assertAlmostEqual(
            P("gumbel", (1.7,), 270).kendall_tau(), -tau, delta=1e-15
        )
        self.assertAlmostEqual(P("gumbel", (1.7,), 180).kendall_tau(), tau, delta=1e-15)
        self.assertEqual(P("gumbel", (1.7,), 180).tail_dependence(), (upper, 0.0))
        self.assertEqual(P("gumbel", (1.7,), 90).tail_dependence(), (0.0, 0.0))
        self.assertEqual(P("gumbel", (1.7,), 90).name, "gumbel-90")
        self.assertEqual(P("student-t", (0.5, 4.0)).parameters, (0.5, 4.0))

    def test_tail_dependence_matches_cdf_limit(self) -> None:
        """The survival Clayton upper-tail coefficient matches the CDF limit."""
        pair = P("clayton", (2.3,), 180)
        q = 1.0 - 1e-7
        limit = (1.0 - 2.0 * q + pair.cdf((q, q))) / (1.0 - q)
        self.assertAlmostEqual(pair.tail_dependence()[1], limit, delta=1e-5)
        self.assertAlmostEqual(
            pair.tail_dependence()[1], 2.0 ** (-1.0 / 2.3), delta=1e-15
        )

    def test_swapped_margin(self) -> None:
        """The ``(1, 0)`` margin of a 90 degree rotation is the 270 degree rotation."""
        pair = P("gumbel", (1.7,), 90)
        swapped = pair.marginal((1, 0))
        assert isinstance(swapped, PairCopula)
        self.assertEqual(swapped.rotation, 270)
        for u1, u2 in POINTS:
            self.assertAlmostEqual(
                swapped.log_density((u2, u1)), pair.log_density((u1, u2)), delta=1e-13
            )
        self.assertIsInstance(pair.marginal((0,)), IndependenceCopula)

    def test_frailty_sampler_matches_rotated_cdf(self) -> None:
        """Reflected frailty draws of Gumbel-90 match the rotated CDF within 5 SE."""
        pair = P("gumbel", (1.7,), 90)
        size = 8000
        sample = pair.sample(size, seed=11)
        for point in ((0.3, 0.6), (0.7, 0.2), (0.5, 0.5)):
            empirical = (
                sum(1 for a, b in sample if a <= point[0] and b <= point[1]) / size
            )
            exact = pair.cdf(point)
            error = math.sqrt(exact * (1.0 - exact) / size)
            self.assertLess(abs(empirical - exact), 5.0 * error)

    def test_conditional_density_integrates_to_one(self) -> None:
        """``int_0^1 c(u_1, v) dv = 1`` by adaptive Gauss-Kronrod for every pair."""
        for pair in ALL_PAIRS:
            for u1 in (0.13, 0.5, 0.91):
                with self.subTest(pair=pair.name, u1=u1):
                    result = gauss_kronrod(
                        lambda v, a=u1, c=pair: c.density((a, _clamp(v))),
                        0.0,
                        1.0,
                        abs_tol=1e-13,
                        rel_tol=1e-12,
                    )
                    self.assertTrue(result.converged)
                    self.assertAlmostEqual(result.value, 1.0, delta=1e-10)

    def test_invalid_pair_copulas(self) -> None:
        """Unsupported bases, rotations and parameter lists raise."""
        with self.assertRaises(ValueError):
            P("gaussian", (0.3,), 90)
        with self.assertRaises(ValueError):
            P("clayton", (2.0,), 45)
        with self.assertRaises(ValueError):
            P("clayton", (2.0, 1.0))
        with self.assertRaises(ValueError):
            P("joe", (2.0,))
        with self.assertRaises(ValueError):
            PairCopula(ClaytonCopula(2.0, 3))
        with self.assertRaises(TypeError):
            PairCopula(P("clayton", (2.0,)))
        with self.assertRaises(ValueError):
            P("frank", (0.0,))


class BivariateVineTests(unittest.TestCase):
    """A two-dimensional vine is exactly its pair copula."""

    def test_two_dimensional_vines_equal_the_pair(self) -> None:
        """Density, Rosenblatt maps, tau and CDF coincide for C- and D-vines."""
        for pair in ALL_PAIRS:
            for vine in (CVineCopula(((pair,),)), DVineCopula(((pair,),))):
                with self.subTest(pair=pair.name, vine=vine.family):
                    for u in POINTS:
                        self.assertEqual(vine.log_density(u), pair.log_density(u))
                        self.assertEqual(vine.rosenblatt(u), pair.rosenblatt(u))
                        self.assertEqual(
                            vine.inverse_rosenblatt(u), pair.inverse_rosenblatt(u)
                        )
                        if pair.has_cdf:
                            self.assertEqual(vine.cdf(u), pair.cdf(u))
                    self.assertEqual(vine.kendall_tau(0, 1), pair.kendall_tau())
                    self.assertEqual(vine.tail_dependence(1, 0), pair.tail_dependence())

    def test_reversed_order_is_the_swapped_pair(self) -> None:
        """``order=(1, 0)`` gives the copula of ``(U_2, U_1)``."""
        pair = P("clayton", (2.3,), 90)
        vine = DVineCopula(((pair,),), order=(1, 0))
        for u1, u2 in POINTS:
            self.assertAlmostEqual(
                vine.log_density((u1, u2)), pair.log_density((u2, u1)), delta=1e-15
            )
            self.assertAlmostEqual(vine.cdf((u1, u2)), pair.cdf((u2, u1)), delta=1e-15)

    def test_density_integrates_to_one(self) -> None:
        """A bounded 2-D density integrates to one by a 32-point product rule."""
        nodes, weights = _gauss_legendre(32)
        vine = CVineCopula(((P("frank", (-3.0,)),),))
        total = math.fsum(
            weights[i] * weights[j] * vine.density((nodes[i], nodes[j]))
            for i in range(32)
            for j in range(32)
        )
        self.assertAlmostEqual(total, 1.0, delta=1e-12)


class GaussianVineTests(unittest.TestCase):
    """Gaussian vines with partial correlations reproduce the Gaussian copula."""

    def test_d_vine_equals_gaussian_copula(self) -> None:
        """Density (1e-12) and both Rosenblatt maps (1e-12) agree."""
        gaussian = GaussianCopula(CORRELATION)
        vine = _gaussian_d_vine((0, 1, 2, 3))
        for u in ((0.1, 0.5, 0.7, 0.3), (0.9, 0.2, 0.4, 0.95), (0.02, 0.98, 0.5, 0.6)):
            self.assertAlmostEqual(
                vine.log_density(u), gaussian.log_density(u), delta=1e-12
            )
            for a, b in zip(vine.rosenblatt(u), gaussian.rosenblatt(u), strict=True):
                self.assertAlmostEqual(a, b, delta=1e-12)
            for a, b in zip(
                vine.inverse_rosenblatt(u), gaussian.inverse_rosenblatt(u), strict=True
            ):
                self.assertAlmostEqual(a, b, delta=1e-12)

    def test_c_vine_equals_gaussian_copula(self) -> None:
        """The C-vine with partial correlations agrees as well."""
        gaussian = GaussianCopula(CORRELATION)
        vine = _gaussian_c_vine((0, 1, 2, 3))
        for u in ((0.1, 0.5, 0.7, 0.3), (0.9, 0.2, 0.4, 0.95)):
            self.assertAlmostEqual(
                vine.log_density(u), gaussian.log_density(u), delta=1e-12
            )
            for a, b in zip(vine.rosenblatt(u), gaussian.rosenblatt(u), strict=True):
                self.assertAlmostEqual(a, b, delta=1e-12)

    def test_permuted_order_keeps_the_density(self) -> None:
        """Any vine order with matching partial correlations gives the same law."""
        gaussian = GaussianCopula(CORRELATION)
        for order in ((2, 0, 3, 1), (3, 1, 0, 2)):
            for vine in (_gaussian_d_vine(order), _gaussian_c_vine(order)):
                for u in ((0.1, 0.5, 0.7, 0.3), (0.9, 0.2, 0.4, 0.95)):
                    self.assertAlmostEqual(
                        vine.log_density(u), gaussian.log_density(u), delta=1e-12
                    )

    def test_margins_are_gaussian_sub_matrices(self) -> None:
        """Sub-vine margins equal the Gaussian margins; other subsets raise."""
        d_vine = _gaussian_d_vine((0, 1, 2, 3))
        c_vine = _gaussian_c_vine((0, 1, 2, 3))
        u = (0.3, 0.8, 0.45)
        for vine, indices in ((d_vine, (3, 1, 2)), (c_vine, (0, 3, 1))):
            margin = vine.marginal(indices)
            reference = GaussianCopula(CORRELATION).marginal(indices)
            self.assertAlmostEqual(
                margin.log_density(u), reference.log_density(u), delta=1e-12
            )
        with self.assertRaises(NotImplementedError):
            d_vine.marginal((0, 2))
        with self.assertRaises(NotImplementedError):
            c_vine.marginal((1, 2))
        self.assertIsInstance(d_vine.marginal((2,)), IndependenceCopula)


class MixedVineTests(unittest.TestCase):
    """Non-Gaussian vines: structure, normalization, Rosenblatt maps, sampling."""

    def test_edges_report_conditioning_sets(self) -> None:
        """Edges follow the C- and D-vine conditioning patterns in coordinates."""
        d_vine = DVineCopula(MIXED_TREES, order=(2, 0, 3, 1))
        c_vine = CVineCopula(MIXED_TREES, order=(2, 0, 3, 1))
        self.assertEqual(
            [(e.tree, e.conditioned, e.conditioning) for e in d_vine.edges],
            [
                (1, (2, 0), ()),
                (1, (0, 3), ()),
                (1, (3, 1), ()),
                (2, (2, 3), (0,)),
                (2, (0, 1), (3,)),
                (3, (2, 1), (0, 3)),
            ],
        )
        self.assertEqual(
            [(e.tree, e.conditioned, e.conditioning) for e in c_vine.edges],
            [
                (1, (2, 0), ()),
                (1, (2, 3), ()),
                (1, (2, 1), ()),
                (2, (0, 3), (2,)),
                (2, (0, 1), (2,)),
                (3, (3, 1), (2, 0)),
            ],
        )
        self.assertEqual(d_vine.parameter_count, 7)
        self.assertEqual(d_vine.family, "d-vine")
        self.assertAlmostEqual(d_vine.kendall_tau(0, 2), 0.5, delta=1e-15)
        with self.assertRaises(NotImplementedError):
            d_vine.kendall_tau(2, 3)

    def test_three_dimensional_density_integrates_to_one(self) -> None:
        """A bounded 3-D D-vine density integrates to one (20^3 Gauss-Legendre)."""
        vine = DVineCopula(
            ((P("frank", (4.0,)), P("frank", (-2.5,))), (P("frank", (2.0,)),)),
            order=(1, 0, 2),
        )
        nodes, weights = _gauss_legendre(20)
        total = math.fsum(
            weights[i]
            * weights[j]
            * weights[k]
            * vine.density((nodes[i], nodes[j], nodes[k]))
            for i in range(20)
            for j in range(20)
            for k in range(20)
        )
        self.assertAlmostEqual(total, 1.0, delta=1e-12)

    def test_integrating_out_an_end_variable_gives_the_pair(self) -> None:
        """``int c(u_0, u_1, v) dv = c_01(u_0, u_1)`` for a tail-dependent D-vine."""
        first = P("gaussian", (0.5,))
        second = P("clayton", (1.0,), 90)
        vine = DVineCopula(((first, second), (P("gumbel", (1.5,), 180),)))
        for a, b in ((0.2, 0.7), (0.6, 0.1)):
            last = gauss_kronrod(
                lambda v, x=a, y=b: vine.density((x, y, _clamp(v))),
                0.0,
                1.0,
                abs_tol=1e-13,
                rel_tol=1e-12,
            )
            self.assertAlmostEqual(last.value, first.density((a, b)), delta=1e-11)
            head = gauss_kronrod(
                lambda v, x=a, y=b: vine.density((_clamp(v), x, y)),
                0.0,
                1.0,
                abs_tol=1e-13,
                rel_tol=1e-12,
            )
            self.assertAlmostEqual(head.value, second.density((a, b)), delta=1e-11)

    def test_rosenblatt_round_trip(self) -> None:
        """``inverse_rosenblatt(rosenblatt(u)) = u`` for permuted mixed vines."""
        for vine in (
            DVineCopula(MIXED_TREES, order=(2, 0, 3, 1)),
            CVineCopula(MIXED_TREES, order=(2, 0, 3, 1)),
        ):
            for u in ((0.1, 0.5, 0.7, 0.3), (0.9, 0.2, 0.4, 0.95)):
                back = vine.inverse_rosenblatt(vine.rosenblatt(u))
                for a, b in zip(back, u, strict=True):
                    self.assertAlmostEqual(a, b, delta=1e-9)

    def test_rosenblatt_of_samples_is_uniform_and_independent(self) -> None:
        """Means, variances and correlations of ``w`` match ``U(0,1)^4`` within 4 SE."""
        size = 1000
        for vine in (
            DVineCopula(MIXED_TREES, order=(2, 0, 3, 1)),
            CVineCopula(MIXED_TREES, order=(1, 3, 0, 2)),
        ):
            sample = vine.sample(size, seed=29)
            transformed = [vine.rosenblatt(u) for u in sample]
            columns = list(zip(*transformed, strict=True))
            mean_error = math.sqrt(1.0 / (12.0 * size))
            variance_error = math.sqrt((1.0 / 80.0 - 1.0 / 144.0) / size)
            for column in columns:
                mean = math.fsum(column) / size
                variance = math.fsum((x - 0.5) ** 2 for x in column) / size
                self.assertLess(abs(mean - 0.5), 4.0 * mean_error)
                self.assertLess(abs(variance - 1.0 / 12.0), 4.0 * variance_error)
            for i in range(4):
                for j in range(i + 1, 4):
                    covariance = math.fsum(
                        (a - 0.5) * (b - 0.5)
                        for a, b in zip(columns[i], columns[j], strict=True)
                    )
                    correlation = 12.0 * covariance / size
                    self.assertLess(abs(correlation), 4.0 / math.sqrt(size))

    def test_sampled_first_tree_kendall_tau(self) -> None:
        """Sample tau of first-tree pairs matches the pair copula within 5 SE."""
        size = 1000
        vine = DVineCopula(MIXED_TREES, order=(2, 0, 3, 1))
        sample = vine.sample(size, seed=5)
        columns = list(zip(*sample, strict=True))
        bound = 5.0 * (2.0 / 3.0) / math.sqrt(size)
        for edge in vine.edges[:3]:
            a, b = edge.conditioned
            self.assertLess(
                abs(kendall_tau(columns[a], columns[b]) - vine.kendall_tau(a, b)), bound
            )

    def test_truncated_vine_is_padded_with_independence(self) -> None:
        """A one-tree D-vine has independence pair copulas in the later trees."""
        vine = DVineCopula(((P("clayton", (2.0,)), P("frank", (3.0,))),))
        self.assertEqual(vine.pair_copulas[1][0].base_family, "independence")
        u = (0.3, 0.6, 0.2)
        expected = P("clayton", (2.0,)).log_density(u[:2]) + P(
            "frank", (3.0,)
        ).log_density(u[1:])
        self.assertAlmostEqual(vine.log_density(u), expected, delta=1e-14)

    def test_invalid_vines(self) -> None:
        """Malformed trees, orders and points raise."""
        clayton = P("clayton", (2.0,))
        with self.assertRaises(ValueError):
            DVineCopula(((clayton, clayton), (clayton, clayton)))
        with self.assertRaises(ValueError):
            CVineCopula(((clayton, clayton), (clayton,), (clayton,)))
        with self.assertRaises(ValueError):
            CVineCopula(((clayton,),), order=(0, 0))
        with self.assertRaises(ValueError):
            CVineCopula(())
        with self.assertRaises(ValueError):
            CVineCopula(((GaussianCopula(CORRELATION),),))
        with self.assertRaises(TypeError):
            CVineCopula((("clayton",),))  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            CVineCopula(((clayton,),)).log_density((0.0, 0.5))
        with self.assertRaises(NotImplementedError):
            CVineCopula(((clayton, clayton), (clayton,))).cdf((0.5, 0.5, 0.5))

    def test_plain_bivariate_copulas_are_wrapped(self) -> None:
        """Bivariate copula objects are accepted as unrotated pair copulas."""
        vine = DVineCopula(
            ((FrankCopula(3.0), StudentTCopula(((1.0, 0.2), (0.2, 1.0)), 6.0)),)
        )
        self.assertEqual(vine.pair_copulas[0][1].name, "student-t")


class JointInputVineTests(unittest.TestCase):
    """A vine is the dependence law of a :class:`JointInput`."""

    def setUp(self) -> None:
        """Build a three-input model with a C-vine copula.

        References:
            [REF-AAS-CZADO-FRIGESSI-BAKKEN-2009].

        """
        self.vine = CVineCopula(
            (
                (P("gumbel", (2.0,)), P("clayton", (1.5,), 270)),
                (P("frank", (2.0,)),),
            ),
            order=(1, 0, 2),
        )
        self.joint = JointInput(
            (
                RandomInput("load", LogNormal(0.0, 0.3)),
                RandomInput("strength", Normal(5.0, 0.5)),
                RandomInput("angle", Uniform(0.0, 1.0)),
            ),
            copula=self.vine,
        )

    def test_model_sampling_and_evaluation(self) -> None:
        """A model on the vine input samples and evaluates every draw."""

        def margin(p: dict[str, float]) -> dict[str, float]:
            """Return the safety margin.

            References:
                [REF-AAS-CZADO-FRIGESSI-BAKKEN-2009]. No equation is evaluated.

            """
            return {"margin": p["strength"] - p["load"] * (1.0 + p["angle"])}

        model = ProbabilisticModel(
            self.joint, margin, (QuantityOfInterest("margin", "capacity minus demand"),)
        )
        self.assertEqual(model.inputs.copula_family, "c-vine")
        self.assertTrue(model.inputs.uses_rosenblatt)
        draws = model.inputs.sample(300, seed=17)
        outputs = [model.evaluate(x)["margin"] for x in draws]
        self.assertEqual(len(outputs), 300)
        self.assertEqual(model.evaluations, 300)
        laws = [entry.distribution for entry in self.joint.inputs]
        units = [[float(law.cdf(x[k])) for k, law in enumerate(laws)] for x in draws]
        tau = kendall_tau([u[1] for u in units], [u[0] for u in units])
        self.assertLess(abs(tau - 0.5), 5.0 * (2.0 / 3.0) / math.sqrt(300))

    def test_unit_map_and_log_pdf(self) -> None:
        """The unit map inverts the standard map; ``log_pdf`` is Sklar's density."""
        for point in ((0.1, 0.5, 0.9), (0.73, 0.02, 0.41)):
            x = self.joint.from_unit(point)
            for a, p in zip(self.joint.to_standard_normal(x), point, strict=True):
                self.assertAlmostEqual(a, _PHI.inv_cdf(p), delta=1e-8)
            laws = [entry.distribution for entry in self.joint.inputs]
            u = tuple(float(law.cdf(v)) for law, v in zip(laws, x, strict=True))
            expected = math.fsum(
                float(law.log_pdf(v)) for law, v in zip(laws, x, strict=True)
            ) + self.vine.log_density(u)
            self.assertAlmostEqual(self.joint.log_pdf(x), expected, delta=1e-10)

    def test_subset_uses_sub_vines(self) -> None:
        """``subset`` keeps a sub-vine margin and refuses a non-sub-vine."""
        sub = self.joint.subset(("strength", "angle"))
        self.assertEqual(sub.copula_family, "c-vine")
        with self.assertRaises(NotImplementedError):
            self.joint.subset(("load", "angle"))


class FittingTests(unittest.TestCase):
    """Sequential maximum-likelihood selection and estimation."""

    FAMILIES = ("independence", "gaussian", "clayton", "gumbel", "frank")

    def test_d_vine_recovers_structure_families_and_parameters(self) -> None:
        """Families, rotations and parameters of a 3-D D-vine are recovered.

        Twenty seeded replicate fits at ``n = 500`` (seeds 101-120) selected
        Clayton 20 times, Gumbel-90 19 times and Frank 15 times; among the
        correct selections the estimates had standard deviations 0.19, 0.074
        and 0.30. The tolerances below are about three of those standard
        deviations.
        """
        truth = DVineCopula(
            ((P("clayton", (3.0,)), P("gumbel", (2.0,), 90)), (P("frank", (2.0,)),))
        )
        sample = truth.sample(500, seed=2)
        fit = fit_vine_copula(sample, "d", self.FAMILIES)
        self.assertEqual(fit.order, (0, 1, 2))
        names = [edge.pair_copula.name for edge in fit.edges]
        self.assertEqual(names, ["clayton", "gumbel-90", "frank"])
        for edge, value, tolerance in zip(
            fit.edges, (3.0, 2.0, 2.0), (0.6, 0.25, 0.9), strict=True
        ):
            self.assertLess(abs(edge.parameters[0] - value), tolerance)
        self.assertEqual(fit.parameter_count, 3)
        total = math.fsum(fit.copula.log_density(row) for row in _pseudo(sample))
        self.assertAlmostEqual(fit.log_likelihood, total, delta=1e-8)
        self.assertAlmostEqual(fit.aic, 6.0 - 2.0 * total, delta=1e-8)
        self.assertAlmostEqual(fit.bic, 3.0 * math.log(500) - 2.0 * total, delta=1e-8)
        for edge in fit.edges:
            self.assertEqual(min(aic for _, aic in edge.candidates), edge.aic)
            self.assertAlmostEqual(
                edge.aic,
                2.0 * edge.pair_copula.parameter_count - 2.0 * edge.log_likelihood,
            )

    def test_c_vine_root_selection_and_independence_test(self) -> None:
        """The root with the largest ``sum |tau|`` is chosen; tree 3 is independent."""
        truth = CVineCopula(
            (
                (P("gumbel", (2.5,)), P("clayton", (2.0,)), P("frank", (-6.0,))),
                (P("gaussian", (0.3,)), P("gaussian", (0.5,))),
                (P("independence"),),
            ),
            order=(2, 0, 1, 3),
        )
        fit = fit_vine_copula(truth.sample(500, seed=3), "c", self.FAMILIES)
        self.assertEqual(fit.order, (2, 0, 1, 3))
        names = [edge.pair_copula.name for edge in fit.edges]
        self.assertEqual(names[:3], ["gumbel", "clayton", "frank"])
        self.assertEqual(names[5], "independence")
        self.assertGreater(fit.edges[5].independence_p_value, 0.05)
        self.assertEqual(fit.edges[5].candidates, ())
        self.assertEqual(fit.edges[3].conditioned, (0, 1))
        self.assertEqual(fit.edges[3].conditioning, (2,))

    def test_student_t_pair_is_selected_with_heavy_tails(self) -> None:
        """A t pair (rho 0.7, nu 4) is preferred to the Gaussian by AIC."""
        truth = CVineCopula(((P("student-t", (0.7, 4.0)),),))
        fit = fit_vine_copula(truth.sample(500, seed=2), "c", ("gaussian", "student-t"))
        edge = fit.edges[0]
        self.assertEqual(edge.family, "student-t")
        rho, nu = edge.parameters
        self.assertLess(abs(rho - 0.7), 0.06)
        self.assertTrue(2.0 < nu < 10.0)
        self.assertAlmostEqual(
            edge.log_likelihood,
            math.fsum(
                edge.pair_copula.log_density(row)
                for row in _pseudo(truth.sample(500, seed=2))
            ),
            delta=1e-8,
        )

    def test_fixed_order_and_uniform_input(self) -> None:
        """A given order is kept; ``rank_transform=False`` uses the data as is."""
        truth = DVineCopula(((P("frank", (5.0,)), P("gaussian", (-0.5,))),))
        sample = truth.sample(150, seed=4)
        fit = fit_vine_copula(
            sample,
            "d",
            ("gaussian", "frank"),
            order=(2, 1, 0),
            rank_transform=False,
            independence_test=False,
        )
        self.assertEqual(fit.order, (2, 1, 0))
        self.assertEqual(fit.edges[0].conditioned, (2, 1))
        total = math.fsum(fit.copula.log_density(row) for row in sample)
        self.assertAlmostEqual(fit.log_likelihood, total, delta=1e-8)

    def test_independence_p_value(self) -> None:
        """The p-value uses the null variance ``2(2n+5)/(9n(n-1))``."""
        size = 50
        tau = 0.2
        z = tau / math.sqrt(2.0 * (2 * size + 5) / (9.0 * size * (size - 1)))
        self.assertAlmostEqual(
            kendall_independence_p_value(tau, size), 2.0 * _PHI.cdf(-z), delta=1e-15
        )
        self.assertEqual(kendall_independence_p_value(0.0, size), 1.0)
        with self.assertRaises(ValueError):
            kendall_independence_p_value(1.5, size)
        with self.assertRaises(ValueError):
            kendall_independence_p_value(0.1, 1)

    def test_invalid_fitting_inputs(self) -> None:
        """Bad structures, families, levels, orders and data raise."""
        rows = [(0.1, 0.2), (0.4, 0.3), (0.8, 0.9), (0.6, 0.5)]
        with self.assertRaises(ValueError):
            fit_vine_copula(rows, "r")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            fit_vine_copula(rows, "c", ("joe",))
        with self.assertRaises(TypeError):
            fit_vine_copula(rows, "c", "gaussian")
        with self.assertRaises(ValueError):
            fit_vine_copula(rows, "c", level=1.0)
        with self.assertRaises(ValueError):
            fit_vine_copula(rows, "c", order=(0, 0))
        with self.assertRaises(ValueError):
            fit_vine_copula(rows[:2], "c")
        with self.assertRaises(ValueError):
            fit_vine_copula([(0.1,), (0.2,), (0.3,)], "c")
        with self.assertRaises(ValueError):
            fit_vine_copula(
                [(0.1, 0.2), (0.4, 1.3), (0.8, 0.9)], "c", rank_transform=False
            )
        with self.assertRaises(ValueError):
            fit_vine_copula([(0.1, 0.2), (0.4, 0.3, 0.1), (0.8, 0.9)], "c")
        with self.assertRaises(ValueError):
            negative = [(i / 10.0, 1.0 - i / 10.0) for i in range(1, 10)]
            fit_vine_copula(negative, "c", ("clayton",), rotations=False)
        self.assertIn("frank", PAIR_COPULA_FAMILIES)


def _pseudo(rows: tuple[tuple[float, ...], ...]) -> list[tuple[float, ...]]:
    """Return rank pseudo-observations ``rank / (n + 1)`` of untied rows.

    References:
        [REF-GENEST-GHOUDI-RIVEST-1995].

    """
    size = len(rows)
    columns = list(zip(*rows, strict=True))
    ranked = []
    for column in columns:
        order = sorted(range(size), key=lambda k, c=column: c[k])
        values = [0.0] * size
        for rank, index in enumerate(order, start=1):
            values[index] = rank / (size + 1.0)
        ranked.append(values)
    return list(zip(*ranked, strict=True))


if __name__ == "__main__":
    unittest.main()
