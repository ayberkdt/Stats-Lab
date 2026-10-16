"""Copula families, the Rosenblatt map, and their use by ``JointInput`` (ADR-0064).

Reference values were computed outside the repository with mpmath 1.4 (50
digits: mixed partial derivatives of the closed-form Archimedean CDFs, the
Debye integral, bivariate normal CDF by one-dimensional conditioning) and
scipy 1.18 (multivariate t density and conditional t laws), then pasted as
literals. No test imports numpy, scipy, or mpmath.
"""

from __future__ import annotations

import dataclasses
import math
import unittest
import warnings
from statistics import NormalDist

from probstat.distributions import Gumbel, LogNormal, Normal, StudentT, Uniform
from probstat.modeling import (
    ClaytonCopula,
    FrankCopula,
    GaussianCopula,
    GumbelCopula,
    IndependenceCopula,
    JointInput,
    ProbabilisticModel,
    RandomInput,
    StudentTCopula,
    frank_tau,
    nearest_correlation_matrix,
)
from probstat.numerics import gauss_kronrod
from probstat.reliability import ModelLimitState, form_hlrf
from probstat.uq import QuantityOfInterest

_PHI = NormalDist()

# (family, theta, d) -> (density, cdf, w2, w3) at u = (0.3, 0.72, 0.55)[:d];
# mpmath 50-digit differentiation of the closed-form CDF.
ARCHIMEDEAN_REFERENCE = {
    ("clayton", 2.3, 2): (0.5180457064668904, 0.29120955999008086, 0.9065221753529723),
    ("clayton", 2.3, 3): (
        0.706158297878156,
        0.27168307022944477,
        0.9065221753529723,
        0.677951944304153,
    ),
    ("gumbel", 1.7, 2): (0.7359927655082482, 0.27799753023129464, 0.8877085545661467),
    ("gumbel", 1.7, 3): (
        0.9881036863367625,
        0.22851180177027947,
        0.8877085545661467,
        0.641738895883195,
    ),
    ("frank", 4.5, 2): (0.5882252072274393, 0.28267844911476464, 0.8987707813769538),
    ("frank", 4.5, 3): (
        0.7955393247738661,
        0.2440739393656893,
        0.8987707813769538,
        0.6543019789531468,
    ),
    ("frank", -3.0, 2): (1.3316420698324078, 0.15387810832422588, 0.623067628008743),
}

FAMILIES = {"clayton": ClaytonCopula, "gumbel": GumbelCopula, "frank": FrankCopula}


def _empirical_cdf(
    sample: tuple[tuple[float, ...], ...], point: tuple[float, ...]
) -> float:
    """Return the fraction of sample points below ``point`` in every coordinate."""
    hits = sum(
        1 for row in sample if all(a <= b for a, b in zip(row, point, strict=True))
    )
    return hits / len(sample)


def _sample_kendall(pairs: tuple[tuple[float, ...], ...]) -> float:
    """Return tau-a of the first two coordinates by direct pair counting."""
    total = 0
    count = len(pairs)
    for i in range(count):
        for j in range(i + 1, count):
            sign = (pairs[i][0] - pairs[j][0]) * (pairs[i][1] - pairs[j][1])
            total += 1 if sign > 0 else -1 if sign < 0 else 0
    return 2.0 * total / (count * (count - 1))


class ArchimedeanReferenceTests(unittest.TestCase):
    """Density, CDF and Rosenblatt map against mpmath derivatives of the CDF."""

    def test_density_cdf_and_conditionals(self) -> None:
        """Every entry agrees with the 50-digit reference to 1e-12 relative."""
        for (family, theta, d), expected in ARCHIMEDEAN_REFERENCE.items():
            copula = FAMILIES[family](theta, d)
            u = (0.3, 0.72, 0.55)[:d]
            w = copula.rosenblatt(u)
            with self.subTest(family=family, theta=theta, d=d):
                self.assertAlmostEqual(copula.density(u), expected[0], delta=1e-12)
                self.assertAlmostEqual(copula.cdf(u), expected[1], delta=1e-13)
                self.assertAlmostEqual(w[0], u[0], delta=0.0)
                self.assertAlmostEqual(w[1], expected[2], delta=1e-13)
                if d == 3:
                    self.assertAlmostEqual(w[2], expected[3], delta=1e-13)

    def test_inverse_rosenblatt_round_trip(self) -> None:
        """The conditional inverse recovers ``u`` to rounding in d = 2..5.

        A conditional probability ``w`` near one carries its complement
        ``1 - w`` with absolute accuracy ``eps``, so the tolerance scales with
        ``eps / min(w, 1 - w)`` (the documented conditioning of the map).
        """
        cases = [
            ClaytonCopula(0.7, 4),
            ClaytonCopula(8.0, 2),
            GumbelCopula(1.05, 3),
            GumbelCopula(4.0, 5),
            FrankCopula(12.0, 4),
            FrankCopula(-9.0, 2),
            StudentTCopula(((1.0, 0.4, 0.1), (0.4, 1.0, -0.2), (0.1, -0.2, 1.0)), 2.5),
            GaussianCopula(((1.0, -0.7), (-0.7, 1.0))),
        ]
        points = ((0.05, 0.9, 0.4, 0.66, 0.2), (0.97, 0.03, 0.51, 0.12, 0.88))
        for copula in cases:
            for point in points:
                u = point[: copula.dimension]
                w = copula.rosenblatt(u)
                back = copula.inverse_rosenblatt(w)
                tolerance = 1e-12 + 4e-16 / min(min(w), 1.0 - max(w))
                with self.subTest(copula=copula.family, u=u):
                    for a, b in zip(back, u, strict=True):
                        self.assertAlmostEqual(a, b, delta=tolerance)

    def test_frank_kendall_tau_against_debye_integral(self) -> None:
        """Frank tau through the Debye function matches mpmath quadrature."""
        reference = {
            -8.0: -0.6026196515511075,
            -1.5: -0.16305416210507212,
            -1e-3: -0.00011111111000000002,
            0.5: 0.05541725432484424,
            2.0: 0.21389456921962013,
            7.3: 0.5751041707227644,
            30.0: 0.8739774847415348,
        }
        for theta, expected in reference.items():
            with self.subTest(theta=theta):
                self.assertAlmostEqual(frank_tau(theta), expected, delta=1e-15)
        self.assertEqual(frank_tau(0.0), 0.0)
        self.assertAlmostEqual(
            FrankCopula(7.3).kendall_tau(), 0.5751041707227644, delta=1e-15
        )

    def test_kendall_tau_by_genest_mackay_integral(self) -> None:
        """Closed-form tau equals ``1 + 4 int_0^1 phi/phi'`` (independent formula)."""
        cases = (
            (ClaytonCopula(2.3), lambda t, a: (t ** (a + 1.0) - t) / a),
            (GumbelCopula(1.7), lambda t, a: t * math.log(t) / a),
            (
                FrankCopula(4.5),
                lambda t, a: (
                    math.log(math.expm1(-a * t) / math.expm1(-a))
                    * math.expm1(a * t)
                    / a
                ),
            ),
        )
        for copula, ratio in cases:
            integral = gauss_kronrod(
                lambda t, c=copula, r=ratio: r(t, c.theta), 0.0, 1.0, rel_tol=1e-13
            ).value
            with self.subTest(family=copula.family):
                self.assertAlmostEqual(
                    copula.kendall_tau(), 1.0 + 4.0 * integral, delta=1e-11
                )

    def test_tail_dependence_matches_cdf_limits(self) -> None:
        """Closed-form tail coefficients equal ``C(q,q)/q`` and its upper analogue."""
        q = 1e-8
        clayton = ClaytonCopula(2.0)
        self.assertAlmostEqual(clayton.tail_dependence()[0], 2.0**-0.5, delta=1e-15)
        self.assertAlmostEqual(
            clayton.cdf((q, q)) / q, clayton.tail_dependence()[0], delta=1e-6
        )
        gumbel = GumbelCopula(2.0)
        upper = (1.0 - 2.0 * (1.0 - 1e-6) + gumbel.cdf((1.0 - 1e-6, 1.0 - 1e-6))) / 1e-6
        self.assertAlmostEqual(
            gumbel.tail_dependence()[1], 2.0 - math.sqrt(2.0), delta=1e-15
        )
        self.assertAlmostEqual(upper, gumbel.tail_dependence()[1], delta=1e-5)
        self.assertEqual(gumbel.tail_dependence()[0], 0.0)
        self.assertEqual(FrankCopula(3.0).tail_dependence(), (0.0, 0.0))
        self.assertEqual(clayton.tail_dependence()[1], 0.0)

    def test_closed_form_tau_values(self) -> None:
        """Clayton ``theta/(theta+2)`` and Gumbel ``1 - 1/theta``."""
        self.assertAlmostEqual(
            ClaytonCopula(3.0, 4).kendall_tau(1, 3), 0.6, delta=1e-15
        )
        self.assertAlmostEqual(GumbelCopula(4.0).kendall_tau(), 0.75, delta=1e-15)
        self.assertEqual(IndependenceCopula(3).kendall_tau(0, 2), 0.0)


class EllipticalReferenceTests(unittest.TestCase):
    """Gaussian and Student-t copulas against mpmath and scipy values."""

    def test_gaussian_cdf(self) -> None:
        """Plackett-identity CDF matches the mpmath conditioning integral."""
        cases = (
            (0.6, (0.3, 0.8), 0.2895206996492478),
            (0.6, (0.01, 0.02), 0.002891007645787436),
            (-0.9, (0.3, 0.8), 0.12002246481432251),
        )
        for rho, u, expected in cases:
            copula = GaussianCopula(((1.0, rho), (rho, 1.0)))
            with self.subTest(rho=rho, u=u):
                self.assertAlmostEqual(copula.cdf(u), expected, delta=1e-13)
        with self.assertRaises(NotImplementedError):
            GaussianCopula(((1.0, 0.1, 0.0), (0.1, 1.0, 0.0), (0.0, 0.0, 1.0))).cdf(
                (0.5, 0.5, 0.5)
            )

    def test_t_density_rosenblatt_and_tail(self) -> None:
        """t-copula log-density, conditionals, and tail coefficient match scipy."""
        matrix = ((1.0, 0.5, 0.2), (0.5, 1.0, -0.3), (0.2, -0.3, 1.0))
        copula = StudentTCopula(matrix, 3.7)
        u = (0.2, 0.9, 0.6)
        self.assertAlmostEqual(copula.log_density(u), -1.882081651773274, delta=1e-11)
        w = copula.rosenblatt(u)
        self.assertAlmostEqual(w[0], 0.2, delta=1e-13)
        self.assertAlmostEqual(w[1], 0.9664602808261041, delta=1e-12)
        self.assertAlmostEqual(w[2], 0.8849405461251757, delta=1e-12)
        lower, upper = copula.tail_dependence(0, 1)
        self.assertAlmostEqual(lower, 0.26937266319239866, delta=1e-12)
        self.assertEqual(lower, upper)
        self.assertAlmostEqual(
            copula.kendall_tau(0, 1), 2.0 / math.pi * math.asin(0.5), delta=1e-15
        )
        with self.assertRaises(NotImplementedError):
            copula.cdf(u)

    def test_gaussian_rosenblatt_equals_nataf(self) -> None:
        """The Gaussian-copula Rosenblatt map is the Nataf map of ``JointInput``."""
        matrix = ((1.0, 0.4, -0.2), (0.4, 1.0, 0.3), (-0.2, 0.3, 1.0))
        laws = (Normal(1.0, 2.0), LogNormal(0.0, 0.3), Uniform(-1.0, 2.0))
        joint = JointInput(
            tuple(RandomInput(f"x{i}", law) for i, law in enumerate(laws)), matrix
        )
        x = (2.5, 1.4, 0.3)
        u = tuple(law.cdf(value) for law, value in zip(laws, x, strict=True))
        w = GaussianCopula(matrix).rosenblatt(u)
        nataf = joint.to_standard_normal(x)
        for a, b in zip(nataf, w, strict=True):
            self.assertAlmostEqual(a, _PHI.inv_cdf(b), delta=1e-12)


class FrailtySamplingTests(unittest.TestCase):
    """Marshall-Olkin and mixture samplers reproduce the closed-form copula."""

    def _check(self, copula: object, size: int, seed: int) -> None:
        """Compare empirical and exact CDFs at three points within 5 SE."""
        sample = copula.sample(size, seed=seed)  # type: ignore[attr-defined]
        d = copula.dimension  # type: ignore[attr-defined]
        for point in ((0.3, 0.6, 0.5)[:d], (0.8, 0.85, 0.9)[:d], (0.1, 0.2, 0.15)[:d]):
            exact = copula.cdf(point)  # type: ignore[attr-defined]
            error = math.sqrt(exact * (1.0 - exact) / size)
            with self.subTest(family=copula.family, point=point):  # type: ignore[attr-defined]
                self.assertLess(abs(_empirical_cdf(sample, point) - exact), 5.0 * error)
        tau = _sample_kendall(sample[:600])
        self.assertLess(abs(tau - copula.kendall_tau()), 0.08)  # type: ignore[attr-defined]

    def test_clayton_gamma_frailty(self) -> None:
        """Gamma frailty gives the trivariate Clayton law."""
        self._check(ClaytonCopula(2.0, 3), 20000, 11)

    def test_gumbel_positive_stable_frailty(self) -> None:
        """Kanter's positive-stable frailty gives the trivariate Gumbel law."""
        self._check(GumbelCopula(1.8, 3), 20000, 12)

    def test_frank_logarithmic_frailty(self) -> None:
        """Kemp's logarithmic frailty gives the trivariate Frank law."""
        self._check(FrankCopula(5.0, 3), 20000, 13)

    def test_negative_frank_by_conditional_inversion(self) -> None:
        """Negative-theta Frank is sampled by conditional inversion."""
        self._check(FrankCopula(-4.0, 2), 20000, 14)

    def test_gaussian_sampler(self) -> None:
        """The Gaussian sampler matches its Plackett CDF."""
        self._check(GaussianCopula(((1.0, 0.55), (0.55, 1.0))), 20000, 15)

    def test_t_sampler_tail_dependence(self) -> None:
        """Chi-square mixing reproduces the t-copula joint lower tail."""
        copula = StudentTCopula(((1.0, 0.5), (0.5, 1.0)), 3.0)
        sample = copula.sample(20000, seed=16)
        joint_lower = sum(1 for a, b in sample if a <= 0.05 and b <= 0.05) / 20000
        # Conditional-inverse sampling of the same law gives an independent estimate.
        other = tuple(
            copula.inverse_rosenblatt(p)
            for p in IndependenceCopula(2).sample(20000, seed=17)
        )
        reference = sum(1 for a, b in other if a <= 0.05 and b <= 0.05) / 20000
        error = math.sqrt(2.0 * reference * (1.0 - reference) / 20000)
        self.assertLess(abs(joint_lower - reference), 5.0 * error)
        gaussian = GaussianCopula(((1.0, 0.5), (0.5, 1.0))).cdf((0.05, 0.05))
        self.assertGreater(joint_lower, gaussian + 5.0 * error)

    def test_from_unit_consistent_with_rosenblatt(self) -> None:
        """``from_unit(p) = from_standard_normal(Phi^-1(p))`` for a Gumbel copula."""
        joint = JointInput(
            (
                RandomInput("a", Normal(0.0, 1.0)),
                RandomInput("b", LogNormal(0.0, 0.5)),
                RandomInput("c", Gumbel(1.0, 2.0)),
            ),
            copula=GumbelCopula(2.2, 3),
        )
        for point in ((0.1, 0.5, 0.9), (0.73, 0.02, 0.41)):
            via_normal = joint.from_standard_normal(
                tuple(_PHI.inv_cdf(p) for p in point)
            )
            for a, b in zip(joint.from_unit(point), via_normal, strict=True):
                self.assertAlmostEqual(a, b, delta=1e-9)
            x = joint.from_unit(point)
            for a, b in zip(
                joint.to_standard_normal(x),
                (_PHI.inv_cdf(p) for p in point),
                strict=True,
            ):
                self.assertAlmostEqual(a, b, delta=1e-8)


class JointInputCopulaTests(unittest.TestCase):
    """``JointInput`` with non-Gaussian copulas and backward compatibility."""

    def test_backward_compatible_construction(self) -> None:
        """``JointInput(inputs, R)`` equals the explicit Gaussian copula form."""
        inputs = (
            RandomInput("a", Normal(0.0, 1.0)),
            RandomInput("b", LogNormal(0.0, 0.4)),
        )
        matrix = ((1.0, 0.3), (0.3, 1.0))
        old_style = JointInput(inputs, matrix)
        new_style = JointInput(inputs, copula=GaussianCopula(matrix))
        self.assertEqual(old_style, new_style)
        self.assertEqual(old_style.copula_family, "gaussian")
        self.assertFalse(old_style.uses_rosenblatt)
        for point in ((0.2, 0.7), (0.95, 0.01)):
            self.assertEqual(old_style.from_unit(point), new_style.from_unit(point))
        independent = JointInput(inputs)
        self.assertTrue(independent.is_independent)
        self.assertEqual(independent.dependence, IndependenceCopula(2))
        self.assertTrue(JointInput(inputs, copula=IndependenceCopula(2)).is_independent)
        self.assertIsNone(new_style.copula)
        self.assertEqual(new_style.dependence, GaussianCopula(matrix))
        # dataclasses.replace on the correlation keeps working.
        changed = dataclasses.replace(old_style, correlation=((1.0, -0.5), (-0.5, 1.0)))
        self.assertEqual(
            changed.dependence.kendall_tau(), 2.0 / math.pi * math.asin(-0.5)
        )
        clayton = JointInput(inputs, copula=ClaytonCopula(2.0))
        swapped = dataclasses.replace(clayton, copula=FrankCopula(3.0))
        self.assertEqual(swapped.copula_family, "frank")

    def test_non_gaussian_copula_flags(self) -> None:
        """A Clayton joint is dependent, has no correlation, and uses Rosenblatt."""
        joint = JointInput(
            (RandomInput("a", Normal(0.0, 1.0)), RandomInput("b", Normal(0.0, 1.0))),
            copula=ClaytonCopula(1.5),
        )
        self.assertFalse(joint.is_independent)
        self.assertIsNone(joint.correlation)
        self.assertTrue(joint.uses_rosenblatt)
        self.assertEqual(joint.copula_family, "clayton")

    def test_log_pdf_is_sklar_density(self) -> None:
        """``log f(x) = sum log f_i + log c(F(x))`` for a Frank joint."""
        laws = (Normal(1.0, 2.0), Gumbel(0.0, 1.0))
        joint = JointInput(
            tuple(RandomInput(f"x{i}", law) for i, law in enumerate(laws)),
            copula=FrankCopula(4.5),
        )
        x = (laws[0].quantile(0.3), laws[1].quantile(0.72))
        expected = (
            math.log(laws[0].pdf(x[0]))
            + math.log(laws[1].pdf(x[1]))
            + math.log(0.5882252072274393)
        )
        self.assertAlmostEqual(joint.log_pdf(x), expected, delta=1e-10)

    def test_subset_keeps_family(self) -> None:
        """Margins keep the family; the Clayton 2-margin CDF matches ``C(u,v,1)``."""
        copula = ClaytonCopula(2.3, 3)
        inputs = tuple(RandomInput(name, Uniform(0.0, 1.0)) for name in ("a", "b", "c"))
        joint = JointInput(inputs, copula=copula)
        sub = joint.subset(("c", "a"))
        self.assertEqual(sub.names, ("c", "a"))
        self.assertEqual(sub.dependence, ClaytonCopula(2.3, 2))
        self.assertAlmostEqual(
            copula.cdf((0.3, 0.72, 1.0)), sub.dependence.cdf((0.3, 0.72)), delta=1e-15
        )
        gaussian = JointInput(
            inputs, ((1.0, 0.2, 0.4), (0.2, 1.0, 0.1), (0.4, 0.1, 1.0))
        )
        self.assertEqual(
            gaussian.subset(("c", "a")).correlation, ((1.0, 0.4), (0.4, 1.0))
        )
        with self.assertRaises(KeyError):
            joint.subset(("z",))

    def test_validation(self) -> None:
        """Invalid parameters and combinations are rejected."""
        inputs = (
            RandomInput("a", Normal(0.0, 1.0)),
            RandomInput("b", Normal(0.0, 1.0)),
        )
        with self.assertRaises(ValueError):
            JointInput(inputs, ((1.0, 0.0), (0.0, 1.0)), ClaytonCopula(1.0))
        with self.assertRaises(ValueError):
            JointInput(inputs, copula=ClaytonCopula(1.0, 3))
        with self.assertRaises(TypeError):
            JointInput(inputs, copula="clayton")  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            ClaytonCopula(0.0)
        with self.assertRaises(ValueError):
            GumbelCopula(0.9)
        with self.assertRaises(ValueError):
            FrankCopula(-1.0, 3)
        with self.assertRaises(ValueError):
            FrankCopula(0.0)
        with self.assertRaises(TypeError):
            ClaytonCopula(True)
        with self.assertRaises(ValueError):
            StudentTCopula(((1.0, 0.2), (0.2, 1.0)), 0.0)
        with self.assertRaises(ValueError):
            GaussianCopula(((1.0, 1.2), (1.2, 1.0)))
        with self.assertRaises(ValueError):
            ClaytonCopula(1.0).density((0.0, 0.5))
        with self.assertRaises(ValueError):
            ClaytonCopula(1.0).kendall_tau(0, 0)
        self.assertEqual(ClaytonCopula(2.0).cdf((0.0, 0.4)), 0.0)
        self.assertEqual(ClaytonCopula(2.0).cdf((1.0, 0.4)), 0.4)
        self.assertEqual(ClaytonCopula(2.0).cdf((1.0, 1.0)), 1.0)


class RosenblattOrderTests(unittest.TestCase):
    """FORM under the Rosenblatt map depends on the variable order."""

    def _beta(self, order: tuple[str, str]) -> float:
        """Return the FORM index of ``P(load >= 3)`` for an input order."""
        laws = {"load": Gumbel(0.0, 1.0), "other": LogNormal(0.0, 0.5)}
        joint = JointInput(
            tuple(RandomInput(name, laws[name]) for name in order),
            copula=ClaytonCopula(2.5),
        )
        model = ProbabilisticModel(
            joint, lambda p: {"y": p["load"]}, (QuantityOfInterest("y", "load"),)
        )
        result = form_hlrf(ModelLimitState(model, "y", 3.0))
        self.assertTrue(result.converged)
        return result.beta

    def test_form_exact_when_variable_is_first(self) -> None:
        """With ``load`` first ``u_1`` depends on ``load`` only, so FORM is exact."""
        exact = -_PHI.inv_cdf(Gumbel(0.0, 1.0).sf(3.0))
        self.assertAlmostEqual(self._beta(("load", "other")), exact, delta=1e-5)

    def test_form_changes_with_order(self) -> None:
        """With ``load`` last the limit state is curved in ``u`` and FORM errs."""
        exact = -_PHI.inv_cdf(Gumbel(0.0, 1.0).sf(3.0))
        self.assertGreater(abs(self._beta(("other", "load")) - exact), 0.05)


class NearestCorrelationTests(unittest.TestCase):
    """Higham's alternating projections."""

    def test_higham_example(self) -> None:
        """Higham (2002) example: off-diagonals 0.7607, 0.1573, 0.7607.

        The digits beyond the paper's four come from an independent BFGS
        minimization over angle-parametrized correlation matrices (scipy).
        """
        result = nearest_correlation_matrix(
            ((1.0, 1.0, 0.0), (1.0, 1.0, 1.0), (0.0, 1.0, 1.0)), eigenvalue_floor=0.0
        )
        self.assertTrue(result.converged)
        self.assertTrue(result.repaired)
        self.assertAlmostEqual(result.matrix[0][1], 0.76068986, delta=2e-8)
        self.assertAlmostEqual(result.matrix[0][2], 0.15729811, delta=2e-8)
        self.assertAlmostEqual(result.matrix[1][2], 0.76068986, delta=2e-8)
        self.assertAlmostEqual(result.distance, 0.5277904635818288, delta=1e-10)

    def test_floor_and_identity(self) -> None:
        """A floor gives a factorizable matrix; a valid matrix is unchanged."""
        repaired = nearest_correlation_matrix(
            ((1.0, 0.9, -0.9), (0.9, 1.0, 0.9), (-0.9, 0.9, 1.0)), eigenvalue_floor=1e-4
        )
        self.assertGreater(repaired.minimum_eigenvalue, 0.9e-4)
        valid = ((1.0, 0.3), (0.3, 1.0))
        same = nearest_correlation_matrix(valid)
        self.assertFalse(same.repaired)
        self.assertEqual(same.matrix, valid)

    def test_rank_correlation_constructors(self) -> None:
        """``from_spearman``/``from_kendall`` use the closed forms and repair loudly."""
        inputs = tuple(RandomInput(n, Normal(0.0, 1.0)) for n in ("a", "b", "c"))
        spearman = JointInput.from_spearman(inputs[:2], ((1.0, 0.5), (0.5, 1.0)))
        assert spearman.correlation is not None
        self.assertAlmostEqual(
            spearman.correlation[0][1], 2.0 * math.sin(math.pi / 12.0), delta=1e-15
        )
        kendall = JointInput.from_kendall(inputs[:2], ((1.0, 0.5), (0.5, 1.0)))
        assert kendall.correlation is not None
        self.assertAlmostEqual(
            kendall.correlation[0][1], math.sin(math.pi / 4.0), delta=1e-15
        )
        bad = ((1.0, 0.9, -0.9), (0.9, 1.0, 0.9), (-0.9, 0.9, 1.0))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            repaired = JointInput.from_kendall(inputs, bad)
        self.assertTrue(any("nearest correlation" in str(w.message) for w in caught))
        self.assertTrue(repaired.correlation is not None)
        with self.assertRaises(ValueError):
            JointInput.from_kendall(inputs, bad, repair=False)
        with self.assertRaises(ValueError):
            JointInput.from_spearman(inputs[:2], ((1.0, 1.5), (1.5, 1.0)))

    def test_from_pearson_repair(self) -> None:
        """A non-positive-definite pairwise Nataf matrix is repaired with a warning."""
        inputs = tuple(RandomInput(n, Normal(0.0, 1.0)) for n in ("a", "b", "c"))
        bad = ((1.0, 0.9, -0.9), (0.9, 1.0, 0.9), (-0.9, 0.9, 1.0))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            joint = JointInput.from_pearson(inputs, bad)
        self.assertTrue(any("Pearson" in str(w.message) for w in caught))
        self.assertIsNotNone(joint.correlation)
        with self.assertRaises(ValueError):
            JointInput.from_pearson(inputs, bad, repair=False)


class GaussianOnlyConsumersTests(unittest.TestCase):
    """Methods whose conditional sampler is Gaussian refuse other copulas."""

    def test_shapley_and_sobol_refuse_clayton(self) -> None:
        """Shapley effects and Sobol indices raise for a Clayton joint law."""
        from probstat.sensitivity import shapley_effects, sobol_analysis

        joint = JointInput(
            (RandomInput("a", Normal(0.0, 1.0)), RandomInput("b", Normal(0.0, 1.0))),
            copula=ClaytonCopula(2.0),
        )
        model = ProbabilisticModel(
            joint, lambda p: {"y": p["a"] + p["b"]}, (QuantityOfInterest("y", "sum"),)
        )
        with self.assertRaises(ValueError):
            shapley_effects(model, "y", seed=1)
        with self.assertRaises(ValueError):
            sobol_analysis(model, "y", 64, seed=1)


class StudentTInputTests(unittest.TestCase):
    """The t copula inside ``JointInput`` with heavy-tailed margins."""

    def test_round_trip_with_cauchy_margin(self) -> None:
        """A Cauchy margin (``StudentT(1)``) works under a t copula."""
        joint = JointInput(
            (RandomInput("a", StudentT(1.0)), RandomInput("b", Normal(0.0, 1.0))),
            copula=StudentTCopula(((1.0, 0.6), (0.6, 1.0)), 4.0),
        )
        x = joint.from_standard_normal((1.3, -0.4))
        back = joint.to_standard_normal(x)
        self.assertAlmostEqual(back[0], 1.3, delta=1e-9)
        self.assertAlmostEqual(back[1], -0.4, delta=1e-9)


if __name__ == "__main__":
    unittest.main()
