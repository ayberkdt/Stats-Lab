"""Tests for causal graphs, discrete structural models, effects, and E-values.

Reference values are computed independently inside each test: d-separation
by brute-force enumeration of every path in the skeleton, interventional
probabilities by hand-written sums of products, saturated-covariate effect
estimates by explicit stratification, and E-values by the closed form.
Stochastic checks use fixed seeds and tolerances in units of the reported
standard error.
"""

from __future__ import annotations

import math
import unittest
import warnings
from itertools import combinations

from probstat.causal import (
    CausalGraph,
    CyclicGraphError,
    DiscreteCausalModel,
    JointTable,
    PositivityWarning,
    backdoor_adjustment,
    bootstrap_effect,
    compare_naive_and_adjusted,
    confounding_bias_factor,
    e_value,
    estimate_aipw,
    estimate_ipw,
    estimate_regression_adjustment,
    fit_propensity_scores,
    frontdoor_adjustment,
    minimal_adjustment_sets,
    risk_ratio_from_odds_ratio,
    risk_ratio_from_standardized_difference,
    satisfies_backdoor,
    satisfies_frontdoor,
    standardized_mean_differences,
)
from probstat.foundations import RandomStream

TRUE_ATE = 2.0


def brute_force_d_separated(
    edges: list[tuple[str, str]], x: str, y: str, given: set[str]
) -> bool:
    """Decide d-separation by checking every simple path in the skeleton.

    Args:
        edges: Directed edges of a DAG.
        x: First node.
        y: Second node.
        given: Conditioning set.

    Returns:
        ``True`` if every path between ``x`` and ``y`` is blocked.

    References:
        [REF-PEARL-2009, Section 1.2.3]. No equation is evaluated.

    """
    children: dict[str, set[str]] = {}
    neighbors: dict[str, set[str]] = {}
    for parent, child in edges:
        children.setdefault(parent, set()).add(child)
        neighbors.setdefault(parent, set()).add(child)
        neighbors.setdefault(child, set()).add(parent)
    edge_set = set(edges)

    def descendants(node: str) -> set[str]:
        """Return proper descendants by depth-first search."""
        found: set[str] = set()
        stack = list(children.get(node, ()))
        while stack:
            current = stack.pop()
            if current not in found:
                found.add(current)
                stack.extend(children.get(current, ()))
        return found

    def blocked(path: list[str]) -> bool:
        """Return whether one path is blocked by ``given``."""
        for left, middle, right in zip(path, path[1:], path[2:], strict=False):
            collider = (left, middle) in edge_set and (right, middle) in edge_set
            if collider:
                if middle not in given and not descendants(middle) & given:
                    return True
            elif middle in given:
                return True
        return False

    def paths(current: str, trail: list[str]) -> list[list[str]]:
        """Enumerate simple paths from ``current`` to ``y``."""
        if current == y:
            return [trail]
        found: list[list[str]] = []
        for neighbor in neighbors.get(current, ()):
            if neighbor not in trail:
                found.extend(paths(neighbor, trail + [neighbor]))
        return found

    return all(blocked(path) for path in paths(x, [x]))


class CausalGraphTests(unittest.TestCase):
    """Graph validation, d-separation, surgery, and identification criteria."""

    def test_chain_fork_collider_and_descendant(self) -> None:
        """Chains and forks are blocked by the middle node; colliders opened."""
        chain = CausalGraph([("A", "B"), ("B", "C")])
        self.assertFalse(chain.is_d_separated("A", "C"))
        self.assertTrue(chain.is_d_separated("A", "C", "B"))
        fork = CausalGraph([("B", "A"), ("B", "C")])
        self.assertFalse(fork.is_d_separated("A", "C"))
        self.assertTrue(fork.is_d_separated("A", "C", {"B"}))
        collider = CausalGraph([("A", "B"), ("C", "B"), ("B", "D")])
        self.assertTrue(collider.is_d_separated("A", "C"))
        self.assertFalse(collider.is_d_separated("A", "C", "B"))
        self.assertFalse(collider.is_d_separated("A", "C", "D"))

    def test_m_bias(self) -> None:
        """Adjusting for the M-structure collider opens a spurious path."""
        graph = CausalGraph(
            [("U1", "X"), ("U1", "M"), ("U2", "M"), ("U2", "Y"), ("X", "Y")]
        )
        self.assertFalse(graph.is_d_separated("X", "Y", {"M"}))
        self.assertTrue(satisfies_backdoor(graph, "X", "Y", ()))
        self.assertFalse(satisfies_backdoor(graph, "X", "Y", {"M"}))
        self.assertTrue(satisfies_backdoor(graph, "X", "Y", {"M", "U1"}))
        self.assertTrue(satisfies_backdoor(graph, "X", "Y", {"M", "U2"}))
        self.assertEqual(minimal_adjustment_sets(graph, "X", "Y"), ((),))
        self.assertEqual(
            minimal_adjustment_sets(graph, "X", "Y", latent={"U1", "U2"}), ((),)
        )

    def test_d_separation_matches_path_enumeration(self) -> None:
        """Moral-graph d-separation agrees with path enumeration on random DAGs."""
        generator = RandomStream(20261012).child("dags").generator()
        names = [f"V{i}" for i in range(7)]
        checks = 0
        for _ in range(6):
            edges = [
                (names[i], names[j])
                for i in range(7)
                for j in range(i + 1, 7)
                if generator.random() < 0.35
            ]
            graph = CausalGraph(edges, names)
            for x, y in combinations(names, 2):
                rest = [n for n in names if n not in (x, y)]
                for size in range(len(rest) + 1):
                    for given in combinations(rest, size):
                        expected = brute_force_d_separated(edges, x, y, set(given))
                        self.assertEqual(
                            graph.is_d_separated(x, y, given), expected, (edges, x, y)
                        )
                        checks += 1
        self.assertEqual(checks, 6 * 21 * 32)

    def test_backdoor_sets_on_confounded_graph(self) -> None:
        """Z3 alone opens Z1 -> Z3 <- Z2; {Z1, Z3} and {Z2, Z3} are minimal."""
        graph = CausalGraph(
            [
                ("Z1", "Z3"),
                ("Z2", "Z3"),
                ("Z1", "X"),
                ("Z3", "X"),
                ("Z3", "Y"),
                ("Z2", "Y"),
                ("X", "W"),
                ("W", "Y"),
            ]
        )
        self.assertFalse(satisfies_backdoor(graph, "X", "Y", {"Z3"}))
        self.assertTrue(satisfies_backdoor(graph, "X", "Y", {"Z1", "Z3"}))
        self.assertFalse(satisfies_backdoor(graph, "X", "Y", {"Z1", "Z2", "Z3", "W"}))
        self.assertEqual(
            minimal_adjustment_sets(graph, "X", "Y"), (("Z1", "Z3"), ("Z2", "Z3"))
        )
        self.assertEqual(
            minimal_adjustment_sets(graph, "X", "Y", latent={"Z1"}), (("Z2", "Z3"),)
        )
        self.assertEqual(minimal_adjustment_sets(graph, "X", "Y", latent={"Z3"}), ())
        with self.assertRaises(ValueError):
            minimal_adjustment_sets(graph, "X", "Y", max_candidates=2)

    def test_frontdoor_criterion(self) -> None:
        """M satisfies the frontdoor criterion only without bypassing paths."""
        edges = [("U", "X"), ("U", "Y"), ("X", "M"), ("M", "Y")]
        graph = CausalGraph(edges)
        self.assertTrue(satisfies_frontdoor(graph, "X", "Y", {"M"}))
        self.assertEqual(minimal_adjustment_sets(graph, "X", "Y", latent={"U"}), ())
        self.assertFalse(
            satisfies_frontdoor(CausalGraph([*edges, ("X", "Y")]), "X", "Y", {"M"})
        )
        self.assertFalse(
            satisfies_frontdoor(CausalGraph([*edges, ("U", "M")]), "X", "Y", {"M"})
        )
        with self.assertRaises(ValueError):
            satisfies_frontdoor(graph, "X", "Y", set())

    def test_surgery_queries_and_validation(self) -> None:
        """do() removes incoming edges; invalid graphs and queries raise."""
        graph = CausalGraph([("Z", "X"), ("X", "Y"), ("Z", "Y")], nodes=["Q"])
        cut = graph.do("X")
        self.assertEqual(cut.parents("X"), frozenset())
        self.assertEqual(cut.parents("Y"), frozenset({"X", "Z"}))
        self.assertEqual(graph.remove_outgoing("X").children("X"), frozenset())
        self.assertEqual(graph.ancestors("Y"), frozenset({"X", "Z"}))
        self.assertEqual(graph.descendants("Z"), frozenset({"X", "Y"}))
        self.assertIn("Q", graph.nodes)
        self.assertEqual(
            graph, CausalGraph([("Z", "Y"), ("X", "Y"), ("Z", "X")], nodes=["Q"])
        )
        self.assertLess(graph.nodes.index("Z"), graph.nodes.index("X"))
        with self.assertRaises(CyclicGraphError):
            CausalGraph([("A", "B"), ("B", "C"), ("C", "A")])
        with self.assertRaises(ValueError):
            CausalGraph([("A", "A")])
        with self.assertRaises(KeyError):
            graph.parents("nope")
        with self.assertRaises(ValueError):
            graph.is_d_separated("X", "Y", {"X"})
        with self.assertRaises(ValueError):
            satisfies_backdoor(graph, "X", "Y", {"Y"})


def four_variable_model() -> DiscreteCausalModel:
    """Return Z -> X -> W -> Y with Z -> Y (binary variables).

    Returns:
        Discrete causal model used by several tests.

    References:
        [REF-PEARL-2009, Section 3.2]. No equation is evaluated.

    """
    binary = (0, 1)
    return DiscreteCausalModel(
        domains={"Z": binary, "X": binary, "W": binary, "Y": binary},
        parents={"Z": (), "X": ("Z",), "W": ("X",), "Y": ("W", "Z")},
        tables={
            "Z": {(): (0.7, 0.3)},
            "X": {(0,): (0.6, 0.4), (1,): (0.3, 0.7)},
            "W": {(0,): (0.8, 0.2), (1,): (0.4, 0.6)},
            "Y": {
                (0, 0): (0.9, 0.1),
                (0, 1): (0.5, 0.5),
                (1, 0): (0.6, 0.4),
                (1, 1): (0.1, 0.9),
            },
        },
    )


def simpson_model() -> DiscreteCausalModel:
    """Return severity Z confounding treatment X and recovery Y.

    Returns:
        Model in which treatment helps in each stratum but looks harmful.

    References:
        [REF-PEARL-2009, Section 6.1]. No equation is evaluated.

    """
    binary = (0, 1)
    return DiscreteCausalModel(
        domains={"Z": binary, "X": binary, "Y": binary},
        parents={"Z": (), "X": ("Z",), "Y": ("X", "Z")},
        tables={
            "Z": {(): (0.5, 0.5)},
            "X": {(0,): (0.8, 0.2), (1,): (0.2, 0.8)},
            "Y": {
                (0, 0): (0.2, 0.8),
                (1, 0): (0.1, 0.9),
                (0, 1): (0.7, 0.3),
                (1, 1): (0.6, 0.4),
            },
        },
    )


class DiscreteCausalModelTests(unittest.TestCase):
    """Truncated factorization, adjustment formulas, and Simpson reversal."""

    def test_truncated_factorization_equals_hand_computation(self) -> None:
        """P(Y=1 | do(X=x)) equals sum_z P(z) sum_w P(w | x) P(Y=1 | w, z)."""
        model = four_variable_model()
        treated = 0.7 * (0.4 * 0.1 + 0.6 * 0.4) + 0.3 * (0.4 * 0.5 + 0.6 * 0.9)
        control = 0.7 * (0.8 * 0.1 + 0.2 * 0.4) + 0.3 * (0.8 * 0.5 + 0.2 * 0.9)
        self.assertAlmostEqual(treated, 0.418, places=12)
        self.assertAlmostEqual(control, 0.286, places=12)
        law1 = model.interventional_distribution("Y", {"X": 1})
        law0 = model.interventional_distribution("Y", {"X": 0})
        self.assertAlmostEqual(law1[1], treated, places=12)
        self.assertAlmostEqual(law0[1], control, places=12)
        joint = model.interventional_joint({"X": 1})
        self.assertAlmostEqual(
            joint.probability({"Z": 1, "X": 1, "W": 1, "Y": 1}), 0.3 * 0.6 * 0.9
        )
        self.assertEqual(joint.probability({"X": 0}), 0.0)
        mutilated = model.intervene({"X": 1}).joint()
        for key, mass in joint.probabilities.items():
            event = dict(zip(joint.variables, key, strict=True))
            self.assertAlmostEqual(mutilated.probability(event), mass, places=14)
        self.assertEqual(model.intervene({"X": 1}).graph, model.graph.do("X"))

    def test_observational_joint_and_adjustment(self) -> None:
        """Backdoor and frontdoor formulas reproduce the interventional law."""
        model = four_variable_model()
        joint = model.joint()
        self.assertAlmostEqual(math.fsum(joint.probabilities.values()), 1.0)
        self.assertAlmostEqual(joint.probability({"Z": 1}), 0.3)
        self.assertAlmostEqual(
            joint.probability({"Z": 0, "X": 1, "W": 0, "Y": 1}), 0.7 * 0.4 * 0.4 * 0.1
        )
        self.assertEqual(minimal_adjustment_sets(model.graph, "X", "Y"), (("Z",),))
        self.assertTrue(satisfies_frontdoor(model.graph, "X", "Y", {"W"}))
        backdoor = backdoor_adjustment(joint, "X", "Y", ["Z"], 1)
        frontdoor = frontdoor_adjustment(
            joint.marginal(["X", "W", "Y"]), "X", "Y", ["W"], 1
        )
        self.assertAlmostEqual(backdoor[1], 0.418, places=12)
        self.assertAlmostEqual(frontdoor[1], 0.418, places=12)
        naive = joint.conditional("Y", {"X": 1})[1]
        self.assertGreater(abs(naive - 0.418), 0.02)

    def test_frontdoor_with_latent_confounder(self) -> None:
        """With U latent, the frontdoor formula on (X, M, Y) recovers do(X)."""
        binary = (0, 1)
        model = DiscreteCausalModel(
            domains={"U": binary, "X": binary, "M": binary, "Y": binary},
            parents={"U": (), "X": ("U",), "M": ("X",), "Y": ("M", "U")},
            tables={
                "U": {(): (0.6, 0.4)},
                "X": {(0,): (0.9, 0.1), (1,): (0.2, 0.8)},
                "M": {(0,): (0.75, 0.25), (1,): (0.15, 0.85)},
                "Y": {
                    (0, 0): (0.8, 0.2),
                    (0, 1): (0.4, 0.6),
                    (1, 0): (0.5, 0.5),
                    (1, 1): (0.05, 0.95),
                },
            },
        )
        observed = model.joint().marginal(["X", "M", "Y"])
        for x in binary:
            truth = model.interventional_distribution("Y", {"X": x})
            estimate = frontdoor_adjustment(observed, "X", "Y", ["M"], x)
            for y in binary:
                self.assertAlmostEqual(estimate[y], truth[y], places=12)
        effect = (
            model.interventional_distribution("Y", {"X": 1})[1]
            - model.interventional_distribution("Y", {"X": 0})[1]
        )
        naive = (
            observed.conditional("Y", {"X": 1})[1]
            - observed.conditional("Y", {"X": 0})[1]
        )
        self.assertGreater(naive - effect, 0.05)

    def test_simpson_reversal(self) -> None:
        """Treatment helps in every stratum, looks harmful marginally."""
        joint = simpson_model().joint()
        comparison = compare_naive_and_adjusted(
            joint, "X", "Y", ["Z"], outcome_value=1, treated=1, control=0
        )
        self.assertAlmostEqual(comparison.naive_treated, 0.50, places=12)
        self.assertAlmostEqual(comparison.naive_control, 0.70, places=12)
        self.assertAlmostEqual(comparison.adjusted_treated, 0.65, places=12)
        self.assertAlmostEqual(comparison.adjusted_control, 0.55, places=12)
        self.assertAlmostEqual(comparison.naive_difference, -0.20, places=12)
        self.assertAlmostEqual(comparison.adjusted_difference, 0.10, places=12)
        self.assertTrue(comparison.reversal)
        for z in (0, 1):
            stratum = (
                joint.conditional("Y", {"X": 1, "Z": z})[1]
                - joint.conditional("Y", {"X": 0, "Z": z})[1]
            )
            self.assertAlmostEqual(stratum, 0.10, places=12)
        counts = {key: round(1000 * mass) for key, mass in joint.probabilities.items()}
        from_counts = JointTable.from_counts(joint.variables, counts)
        again = compare_naive_and_adjusted(
            from_counts, "X", "Y", ["Z"], outcome_value=1, treated=1, control=0
        )
        self.assertAlmostEqual(again.adjusted_difference, 0.10, places=12)
        self.assertTrue(again.reversal)

    def test_validation_errors(self) -> None:
        """Bad tables, zero-probability events, and positivity failures raise."""
        binary = (0, 1)
        with self.assertRaises(ValueError):
            DiscreteCausalModel({"A": binary}, {"A": ()}, {"A": {(): (0.5, 0.6)}})
        with self.assertRaises(ValueError):
            DiscreteCausalModel(
                {"A": binary, "B": binary},
                {"A": (), "B": ("A",)},
                {"A": {(): (0.5, 0.5)}, "B": {(0,): (0.5, 0.5)}},
            )
        with self.assertRaises(CyclicGraphError):
            DiscreteCausalModel(
                {"A": binary, "B": binary},
                {"A": ("B",), "B": ("A",)},
                {
                    "A": {(0,): (1.0, 0.0), (1,): (1.0, 0.0)},
                    "B": {(0,): (1.0, 0.0), (1,): (1.0, 0.0)},
                },
            )
        degenerate = DiscreteCausalModel(
            {"Z": binary, "X": binary, "Y": binary},
            {"Z": (), "X": ("Z",), "Y": ("X", "Z")},
            {
                "Z": {(): (0.5, 0.5)},
                "X": {(0,): (1.0, 0.0), (1,): (0.5, 0.5)},
                "Y": {key: (0.5, 0.5) for key in ((0, 0), (0, 1), (1, 0), (1, 1))},
            },
        ).joint()
        with self.assertRaises(ValueError):
            backdoor_adjustment(degenerate, "X", "Y", ["Z"], 1)
        with self.assertRaises(ValueError):
            degenerate.conditional("Y", {"X": 1, "Z": 0})
        with self.assertRaises(TypeError):
            backdoor_adjustment(degenerate, "X", "Y", "Z", 1)
        with self.assertRaises(ValueError):
            simpson_model().interventional_distribution("X", {"X": 1})


def simulate(
    n: int, seed: int
) -> tuple[list[tuple[float, float]], list[int], list[float]]:
    """Draw confounded data with a known average treatment effect of two.

    ``X1, X2 ~ N(0, 1)``, ``logit e = -0.3 + 0.8 X1 - 0.5 X2``, and
    ``Y = 1 + 2 T + 1.5 X1 + X2 + 0.5 T X1 + N(0, 1)``, so
    ``E[Y(1) - Y(0)] = 2``.

    Args:
        n: Sample size.
        seed: Root seed.

    Returns:
        Covariate rows, treatment indicators, and outcomes.

    References:
        [REF-HERNAN-ROBINS-2020, Chapter 13]. No equation is evaluated.

    """
    generator = RandomStream(seed).child("causal-simulation").generator()
    rows: list[tuple[float, float]] = []
    treatment: list[int] = []
    outcome: list[float] = []
    for _ in range(n):
        x1, x2 = generator.gauss(0.0, 1.0), generator.gauss(0.0, 1.0)
        score = 1.0 / (1.0 + math.exp(-(-0.3 + 0.8 * x1 - 0.5 * x2)))
        t = 1 if generator.random() < score else 0
        y = 1.0 + 2.0 * t + 1.5 * x1 + x2 + 0.5 * t * x1 + generator.gauss(0.0, 1.0)
        rows.append((x1, x2))
        treatment.append(t)
        outcome.append(y)
    return rows, treatment, outcome


class EffectEstimationTests(unittest.TestCase):
    """IPW, AIPW, regression adjustment, balance, and the bootstrap."""

    rows: list[tuple[float, float]]
    treatment: list[int]
    outcome: list[float]

    @classmethod
    def setUpClass(cls) -> None:
        """Simulate one confounded data set shared by the estimator tests."""
        cls.rows, cls.treatment, cls.outcome = simulate(2000, 20261012)

    def test_naive_difference_is_biased_and_estimators_recover_ate(self) -> None:
        """Naive contrast is far from 2; IPW, AIPW, and regression are not."""
        t, y = self.treatment, self.outcome
        n1 = sum(t)
        naive = math.fsum(yi for yi, ti in zip(y, t, strict=True) if ti) / n1 - (
            math.fsum(yi for yi, ti in zip(y, t, strict=True) if not ti) / (len(t) - n1)
        )
        self.assertGreater(naive - TRUE_ATE, 0.6)
        for stabilized in (True, False):
            ipw = estimate_ipw(self.rows, t, y, stabilized=stabilized)
            self.assertLess(abs(ipw.estimate - TRUE_ATE), 3.0 * ipw.standard_error)
            self.assertLess(ipw.standard_error, 0.2)
            self.assertGreater((naive - TRUE_ATE) / ipw.standard_error, 4.0)
        aipw = estimate_aipw(self.rows, t, y)
        self.assertLess(abs(aipw.estimate - TRUE_ATE), 3.0 * aipw.standard_error)
        self.assertLess(aipw.standard_error, 0.08)
        lower, upper = aipw.confidence_interval(0.95)
        self.assertLess(lower, TRUE_ATE)
        self.assertGreater(upper, TRUE_ATE)
        regression = estimate_regression_adjustment(self.rows, t, y)
        self.assertLess(abs(regression.estimate - TRUE_ATE), 0.1)
        self.assertTrue(math.isnan(regression.standard_error))

    def test_aipw_is_doubly_robust(self) -> None:
        """AIPW stays consistent when exactly one working model omits X1."""
        t, y = self.treatment, self.outcome
        only_x2 = [(row[1],) for row in self.rows]
        wrong_outcome = estimate_aipw(self.rows, t, y, outcome_covariates=only_x2)
        wrong_regression = estimate_regression_adjustment(only_x2, t, y)
        self.assertLess(
            abs(wrong_outcome.estimate - TRUE_ATE), 3.0 * wrong_outcome.standard_error
        )
        self.assertGreater(abs(wrong_regression.estimate - TRUE_ATE), 1.0)
        wrong_propensity = estimate_aipw(self.rows, t, y, propensity_covariates=only_x2)
        wrong_ipw = estimate_ipw(only_x2, t, y)
        self.assertLess(
            abs(wrong_propensity.estimate - TRUE_ATE),
            3.0 * wrong_propensity.standard_error,
        )
        self.assertGreater(
            abs(wrong_ipw.estimate - TRUE_ATE), 8.0 * wrong_ipw.standard_error
        )
        both_wrong = estimate_aipw(only_x2, t, y)
        self.assertGreater(
            abs(both_wrong.estimate - TRUE_ATE), 8.0 * both_wrong.standard_error
        )

    def test_balance_and_trimming(self) -> None:
        """IPW weights balance covariate means; trimming drops extreme scores."""
        t, y = self.treatment, self.outcome
        ipw = estimate_ipw(self.rows, t, y)
        assert ipw.weights is not None
        raw = standardized_mean_differences(self.rows, t, names=["x1", "x2"])
        weighted = standardized_mean_differences(
            self.rows, t, weights=ipw.weights, names=["x1", "x2"]
        )
        self.assertGreater(raw.max_absolute_difference, 0.4)
        self.assertLess(weighted.max_absolute_difference, 0.1)
        x1 = [row[0] for row in self.rows]
        treated = [v for v, ti in zip(x1, t, strict=True) if ti]
        control = [v for v, ti in zip(x1, t, strict=True) if not ti]

        def variance(values: list[float]) -> float:
            """Return the sample variance."""
            mean = math.fsum(values) / len(values)
            return math.fsum((v - mean) ** 2 for v in values) / (len(values) - 1)

        expected = math.fsum(treated) / len(treated) - math.fsum(control) / len(control)
        expected /= math.sqrt(0.5 * (variance(treated) + variance(control)))
        self.assertAlmostEqual(
            raw.covariates[0].standardized_difference, expected, places=12
        )
        scores = fit_propensity_scores(self.rows, t).scores
        outside = sum(1 for e in scores if e < 0.1 or e > 0.9)
        trimmed = estimate_ipw(self.rows, t, y, trim=0.1)
        self.assertEqual(trimmed.n_trimmed, outside)
        self.assertEqual(trimmed.n_used, len(t) - outside)
        self.assertEqual(trimmed.estimand, "ATE (trimmed population)")
        self.assertGreater(outside, 0)

    def test_saturated_binary_covariate_matches_stratification(self) -> None:
        """With one binary covariate all estimators equal the standardized mean."""
        generator = RandomStream(5).child("saturated").generator()
        z = [1 if generator.random() < 0.4 else 0 for _ in range(400)]
        t = [1 if generator.random() < (0.7 if zi else 0.3) else 0 for zi in z]
        y = [
            1.0 + 2.0 * ti + 3.0 * zi + generator.gauss(0.0, 1.0)
            for ti, zi in zip(t, z, strict=True)
        ]
        expected = 0.0
        for level in (0, 1):
            arm = {
                a: [
                    yi
                    for yi, ti, zi in zip(y, t, z, strict=True)
                    if zi == level and ti == a
                ]
                for a in (0, 1)
            }
            share = z.count(level) / len(z)
            expected += share * (
                math.fsum(arm[1]) / len(arm[1]) - math.fsum(arm[0]) / len(arm[0])
            )
        rows = [(float(zi),) for zi in z]
        estimates = [
            estimate_regression_adjustment(rows, t, y).estimate,
            estimate_ipw(rows, t, y).estimate,
            estimate_ipw(rows, t, y, stabilized=False).estimate,
            estimate_aipw(rows, t, y).estimate,
        ]
        for value in estimates:
            self.assertAlmostEqual(value, expected, places=7)

    def test_positivity_and_input_errors(self) -> None:
        """Extreme scores warn; malformed treatment and trims raise."""
        rows = [(float(i),) for i in range(10)]
        t = [0, 1] * 5
        y = [float(i) for i in range(10)]
        with self.assertWarns(PositivityWarning):
            estimate_ipw(rows, t, y, propensity_scores=[0.005] + [0.5] * 9)
        with self.assertRaises(ValueError):
            estimate_ipw(rows, [0, 2] * 5, y)
        with self.assertRaises(ValueError):
            estimate_ipw(rows, [1] * 10, y)
        with self.assertRaises(ValueError):
            estimate_ipw(rows, t, y, trim=0.6)
        with self.assertRaises(ValueError):
            estimate_ipw(rows, t, y, propensity_scores=[1.0] * 10)

    def test_bootstrap_is_seeded_and_covers(self) -> None:
        """Equal seeds reproduce replicates; the AIPW interval covers 2."""
        rows, t, y = simulate(250, 77)
        first = bootstrap_effect(rows, t, y, method="regression", replicates=30, seed=3)
        second = bootstrap_effect(
            rows, t, y, method="regression", replicates=30, seed=3
        )
        self.assertEqual(first.bootstrap.replicates, second.bootstrap.replicates)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PositivityWarning)
            interval = bootstrap_effect(
                rows, t, y, method="aipw", replicates=40, seed=11
            )
        self.assertEqual(interval.method, "percentile")
        self.assertLess(interval.lower, TRUE_ATE)
        self.assertGreater(interval.upper, TRUE_ATE)
        self.assertLess(interval.upper - interval.lower, 1.5)
        with self.assertRaises(ValueError):
            bootstrap_effect(rows, t, y, method="tmle")  # type: ignore[arg-type]


class EValueTests(unittest.TestCase):
    """E-value closed form, interval rule, and bias-factor identity."""

    def test_known_risk_ratios(self) -> None:
        """RR 3.9 with lower limit 1.8 gives E = 7.26 and 3.0."""
        result = e_value(3.9, lower=1.8, upper=8.7)
        self.assertAlmostEqual(result.point, 3.9 + math.sqrt(3.9 * 2.9), places=12)
        self.assertAlmostEqual(result.point, 7.2630, places=4)
        assert result.confidence_limit is not None
        self.assertAlmostEqual(result.confidence_limit, 3.0, places=12)
        protective = e_value(0.5, lower=0.3, upper=0.8)
        self.assertAlmostEqual(protective.point, 2.0 + math.sqrt(2.0), places=12)
        assert protective.confidence_limit is not None
        self.assertAlmostEqual(
            protective.confidence_limit, 1.25 + math.sqrt(1.25 * 0.25), places=12
        )
        self.assertEqual(e_value(1.5, lower=0.9).confidence_limit, 1.0)
        self.assertIsNone(e_value(2.0).confidence_limit)
        self.assertEqual(e_value(1.0).point, 1.0)

    def test_bias_factor_and_conversions(self) -> None:
        """B(E, E) = RR; odds-ratio and SMD conversions follow their formulas."""
        for rr in (1.1, 2.0, 3.9, 10.0):
            strength = e_value(rr).point
            self.assertAlmostEqual(
                confounding_bias_factor(strength, strength), rr, places=12
            )
        self.assertAlmostEqual(confounding_bias_factor(2.0, 3.0), 1.5)
        self.assertEqual(risk_ratio_from_odds_ratio(4.0, common_outcome=True), 2.0)
        self.assertEqual(risk_ratio_from_odds_ratio(4.0, common_outcome=False), 4.0)
        self.assertAlmostEqual(
            risk_ratio_from_standardized_difference(0.5), math.exp(0.455)
        )
        with self.assertRaises(ValueError):
            e_value(-1.0)
        with self.assertRaises(ValueError):
            e_value(2.0, lower=2.5)
        with self.assertRaises(ValueError):
            e_value(2.0, upper=3.0)
        with self.assertRaises(ValueError):
            confounding_bias_factor(0.5, 2.0)


if __name__ == "__main__":
    unittest.main()
