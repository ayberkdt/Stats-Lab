"""Discrete and categorical inputs, epistemic nature, and intervals (ADR-0064).

Joint probability masses were computed outside the repository: the Gaussian
copula case with mpmath 1.4 (bivariate normal CDF by one-dimensional
conditioning, inclusion-exclusion over the four rectangle corners) and the
Clayton case with mpmath closed-form CDF and exact binomial CDFs. No test
imports numpy, scipy, or mpmath.
"""

from __future__ import annotations

import math
import unittest

from probstat.distributions import (
    Bernoulli,
    Binomial,
    Categorical,
    Geometric,
    NegativeBinomial,
    Normal,
    Poisson,
    StudentT,
)
from probstat.modeling import (
    ClaytonCopula,
    IntegerCodedCategorical,
    IntervalInput,
    JointInput,
    ProbabilisticModel,
    RandomInput,
    is_discrete_law,
)
from probstat.reliability import ModelLimitState, form_hlrf, inverse_form, sorm
from probstat.uq import QuantityOfInterest, UncertaintyNature


class InterfaceTests(unittest.TestCase):
    """Structural checks and the Cauchy regression."""

    def test_cauchy_input_constructs(self) -> None:
        """``StudentT(1)`` is accepted although its ``mean`` property raises."""
        cauchy = RandomInput("x", StudentT(1.0))
        joint = JointInput.independent(cauchy, RandomInput("y", Normal(0.0, 1.0)))
        draws = joint.sample(50, seed=3)
        self.assertEqual(len(draws), 50)
        self.assertAlmostEqual(joint.from_unit((0.75, 0.5))[0], 1.0, delta=1e-12)
        with self.assertRaises(ValueError):
            _ = cauchy.mean

    def test_missing_attribute_rejected(self) -> None:
        """A law without a CDF is rejected by name."""

        class Broken:
            """A law with only a density."""

            def pdf(self, x: float) -> float:
                """Return one."""
                return 1.0

        with self.assertRaises(TypeError):
            RandomInput("x", Broken())  # type: ignore[arg-type]

    def test_discrete_laws_accepted(self) -> None:
        """Poisson, Binomial, Geometric, NegativeBinomial, Bernoulli, Categorical."""
        laws = (
            Poisson(2.0),
            Binomial(5, 0.3),
            Geometric(0.4),
            NegativeBinomial(3, 0.5),
            Bernoulli(0.3),
        )
        for law in laws:
            with self.subTest(law=law):
                self.assertTrue(is_discrete_law(law))
                self.assertTrue(RandomInput("k", law).is_discrete)
        coded = RandomInput("c", Categorical({"a": 0.2, "b": 0.5, "c": 0.3}))
        self.assertIsInstance(coded.distribution, IntegerCodedCategorical)
        self.assertFalse(is_discrete_law(Normal(0.0, 1.0)))

    def test_nature_field(self) -> None:
        """Default aleatory; text and enum values are accepted; junk is not."""
        self.assertIs(
            RandomInput("x", Normal(0.0, 1.0)).nature, UncertaintyNature.ALEATORY
        )
        epistemic = RandomInput("x", Normal(0.0, 1.0), nature="epistemic")  # type: ignore[arg-type]
        self.assertIs(epistemic.nature, UncertaintyNature.EPISTEMIC)
        self.assertTrue(epistemic.is_epistemic)
        with self.assertRaises(ValueError):
            RandomInput("x", Normal(0.0, 1.0), nature="vague")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            RandomInput("x", Normal(0.0, 1.0), nature=1)  # type: ignore[arg-type]

    def test_interval_input(self) -> None:
        """Bounds are validated and the unit map is linear."""
        box = IntervalInput("e", -1.0, 3.0)
        self.assertIs(box.nature, UncertaintyNature.EPISTEMIC)
        self.assertEqual(box.width, 4.0)
        self.assertEqual(box.from_unit(0.25), 0.0)
        with self.assertRaises(ValueError):
            IntervalInput("e", 2.0, 1.0)
        with self.assertRaises(TypeError):
            IntervalInput("e", True, 1.0)
        with self.assertRaises(ValueError):
            box.from_unit(1.5)


class CategoricalCodingTests(unittest.TestCase):
    """Integer coding of a categorical law."""

    def test_coding_and_generalized_inverse(self) -> None:
        """Codes follow support order; the inverse partitions ``(0, 1)`` exactly."""
        coded = IntegerCodedCategorical(
            Categorical({"low": 0.2, "mid": 0.5, "high": 0.3})
        )
        self.assertEqual(coded.labels, ("low", "mid", "high"))
        self.assertEqual(coded.label(1.0), "mid")
        self.assertEqual(coded.code("high"), 2)
        self.assertAlmostEqual(coded.cdf(1.0), 0.7, delta=1e-15)
        self.assertAlmostEqual(coded.sf(0.0), 0.8, delta=1e-15)
        self.assertAlmostEqual(coded.mean, 1.1, delta=1e-15)
        self.assertAlmostEqual(coded.variance, 0.49, delta=1e-15)
        cells = 1000
        counts = [0, 0, 0]
        for k in range(cells):
            counts[int(coded.quantile((k + 0.5) / cells))] += 1
        self.assertEqual(counts, [200, 500, 300])
        with self.assertRaises(ValueError):
            coded.label(3)

    def test_categorical_input_in_model(self) -> None:
        """A forward model decodes the code it receives."""
        joint = JointInput.independent(
            RandomInput("c", Categorical({"steel": 0.25, "wood": 0.75})),
            RandomInput("x", Normal(0.0, 1.0)),
        )
        coding = joint.inputs[0].distribution
        assert isinstance(coding, IntegerCodedCategorical)
        self.assertEqual(coding.label(joint.from_unit((0.1, 0.5))[0]), "steel")
        self.assertEqual(coding.label(joint.from_unit((0.3, 0.5))[0]), "wood")


class GeneralizedInverseTests(unittest.TestCase):
    """Exact sampling of discrete margins through every copula path."""

    def test_from_unit_is_generalized_inverse(self) -> None:
        """``F(x - 1) < p <= F(x)`` for the first coordinate under every copula."""
        law = Poisson(3.2)
        joints = (
            JointInput.independent(
                RandomInput("k", law), RandomInput("z", Normal(0, 1))
            ),
            JointInput(
                (RandomInput("k", law), RandomInput("z", Normal(0, 1))),
                ((1.0, 0.5), (0.5, 1.0)),
            ),
            JointInput(
                (RandomInput("k", law), RandomInput("z", Normal(0, 1))),
                copula=ClaytonCopula(2.0),
            ),
        )
        for joint in joints:
            for k in range(1, 200):
                p = k / 200.0
                # The first coordinate of every Rosenblatt map is w_1 = u_1.
                x = joint.from_unit((p, 0.5))[0]
                self.assertTrue(x.is_integer())
                self.assertLess(law.cdf(x - 1.0), p)
                self.assertGreaterEqual(law.cdf(x), p)

    def test_distributional_transform_maps_back(self) -> None:
        """``from_standard_normal(to_standard_normal(x, seed))`` returns ``x``."""
        for copula_kwargs in (
            {"correlation": ((1.0, 0.6), (0.6, 1.0))},
            {"copula": ClaytonCopula(1.5)},
        ):
            joint = JointInput(
                (
                    RandomInput("k", Binomial(6, 0.4)),
                    RandomInput("z", Normal(0.0, 1.0)),
                ),
                **copula_kwargs,  # type: ignore[arg-type]
            )
            for x in ((0.0, -1.0), (3.0, 0.2), (6.0, 1.5)):
                for seed in (None, 1, 2, 3):
                    u = joint.to_standard_normal(x, seed=seed)
                    back = joint.from_standard_normal(u)
                    with self.subTest(x=x, seed=seed, copula=joint.copula_family):
                        self.assertEqual(back[0], x[0])
                        self.assertAlmostEqual(back[1], x[1], delta=1e-8)

    def test_latent_value_lies_in_count_interval(self) -> None:
        """The seeded latent value lies in ``[Phi^-1(F(x-)), Phi^-1(F(x))]``."""
        law = Poisson(2.5)
        joint = JointInput.independent(RandomInput("k", law))
        for x in (0.0, 2.0, 7.0):
            lower_p = law.cdf(x - 1.0)
            upper_p = law.cdf(x)
            for seed in range(5):
                z = joint.to_latent((x,), seed=seed)[0]
                p = 0.5 * math.erfc(-z / math.sqrt(2.0))
                self.assertGreaterEqual(p, lower_p - 1e-12)
                self.assertLessEqual(p, upper_p + 1e-12)
        with self.assertRaises(ValueError):
            joint.to_latent((1.5,))


class DiscreteLogPmfTests(unittest.TestCase):
    """Joint masses by inclusion-exclusion over the copula CDF."""

    def test_gaussian_copula_poisson_mass(self) -> None:
        """Gaussian copula, Poisson(2) and Poisson(3.5) margins, rho = 0.6."""
        joint = JointInput(
            (RandomInput("a", Poisson(2.0)), RandomInput("b", Poisson(3.5))),
            ((1.0, 0.6), (0.6, 1.0)),
        )
        reference = {
            (1.0, 3.0): 0.07067061115160898,
            (2.0, 2.0): 0.049070155976829316,
            (0.0, 5.0): 0.0031433923305646816,
        }
        for point, expected in reference.items():
            with self.subTest(point=point):
                self.assertAlmostEqual(
                    math.exp(joint.log_pdf(point)), expected, delta=1e-13
                )

    def test_clayton_binomial_mass_and_normalization(self) -> None:
        """Clayton(2), Binomial(4, 0.3) and Binomial(3, 0.6) margins."""
        first, second = Binomial(4, 0.3), Binomial(3, 0.6)
        joint = JointInput(
            (RandomInput("a", first), RandomInput("b", second)),
            copula=ClaytonCopula(2.0),
        )
        reference = {
            (1.0, 2.0): 0.2199131587291174,
            (0.0, 0.0): 0.061959334101606596,
            (4.0, 3.0): 0.004178367840824456,
        }
        for point, expected in reference.items():
            self.assertAlmostEqual(
                math.exp(joint.log_pdf(point)), expected, delta=1e-14
            )
        masses = {
            (a, b): math.exp(joint.log_pdf((float(a), float(b))))
            for a in range(5)
            for b in range(4)
        }
        self.assertAlmostEqual(math.fsum(masses.values()), 1.0, delta=1e-13)
        for a in range(5):
            marginal = math.fsum(masses[(a, b)] for b in range(4))
            self.assertAlmostEqual(marginal, first.pmf(a), delta=1e-14)
        self.assertEqual(joint.log_pdf((1.5, 1.0)), -math.inf)
        self.assertEqual(joint.log_pdf((5.0, 1.0)), -math.inf)

    def test_independent_mixed_and_dependent_mixed(self) -> None:
        """Independence gives the mixed density; a dependent mix is refused."""
        independent = JointInput.independent(
            RandomInput("k", Poisson(2.0)), RandomInput("z", Normal(0.0, 1.0))
        )
        expected = math.log(Poisson(2.0).pmf(3)) + Normal(0.0, 1.0).log_pdf(0.4)
        self.assertAlmostEqual(independent.log_pdf((3.0, 0.4)), expected, delta=1e-14)
        dependent = JointInput(
            (RandomInput("k", Poisson(2.0)), RandomInput("z", Normal(0.0, 1.0))),
            ((1.0, 0.3), (0.3, 1.0)),
        )
        with self.assertRaises(NotImplementedError):
            dependent.log_pdf((3.0, 0.4))


class GradientMethodRefusalTests(unittest.TestCase):
    """FORM-type methods refuse discrete inputs instead of answering wrongly."""

    def setUp(self) -> None:
        """Build a model with one Poisson input."""
        joint = JointInput.independent(
            RandomInput("k", Poisson(4.0)), RandomInput("z", Normal(0.0, 1.0))
        )
        model = ProbabilisticModel(
            joint, lambda p: {"y": p["k"] + p["z"]}, (QuantityOfInterest("y", "sum"),)
        )
        self.state = ModelLimitState(model, "y", 9.0)

    def test_form_sorm_inverse_form_raise(self) -> None:
        """Each entry point raises ``ValueError`` naming the discrete input."""
        with self.assertRaisesRegex(ValueError, "discrete inputs"):
            form_hlrf(self.state)
        with self.assertRaisesRegex(ValueError, "discrete inputs"):
            sorm(self.state)
        with self.assertRaisesRegex(ValueError, "discrete inputs"):
            inverse_form(self.state, 2.0)

    def test_from_pearson_refuses_discrete(self) -> None:
        """Pearson matching by Gauss-Hermite needs continuous margins."""
        with self.assertRaises(ValueError):
            JointInput.from_pearson(
                (RandomInput("k", Poisson(2.0)), RandomInput("z", Normal(0.0, 1.0))),
                ((1.0, 0.4), (0.4, 1.0)),
            )


if __name__ == "__main__":
    unittest.main()
