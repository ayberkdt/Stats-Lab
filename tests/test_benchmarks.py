"""Tests for the benchmark suite: reference values, records, failure capture."""

from __future__ import annotations

import json
import math
import random
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import harness, methods, problems, references, report  # noqa: E402
from benchmarks import run as runner  # noqa: E402
from probstat.inverse import calibrate_linear_gaussian  # noqa: E402
from probstat.modeling import gauss_hermite, standard_normal_cdf  # noqa: E402
from probstat.multivariate import MultivariateNormal  # noqa: E402
from probstat.reliability import gauss_legendre  # noqa: E402
from probstat.sensitivity import linear_gaussian_shapley  # noqa: E402

REGISTRY = {problem.key: problem for problem in problems.registry()}
REQUIRED_FIELDS = {
    "problem",
    "category",
    "method",
    "kind",
    "seed",
    "quantity",
    "estimate",
    "reference",
    "abs_error",
    "rel_error",
    "standard_error",
    "z",
    "covered",
    "evaluations",
    "wall_time_s",
    "status",
    "error",
    "warnings",
    "notes",
}


def reference(key: str, quantity: str, seed: int = 1) -> float:
    """Return one registered reference value."""
    return REGISTRY[key].references(seed)[quantity].value


def tensor_rule(order: int, lower: float, upper: float) -> list[tuple[float, float]]:
    """Return Gauss-Legendre ``(node, weight)`` pairs for a uniform density."""
    nodes, weights = gauss_legendre(order)
    half = 0.5 * (upper - lower)
    return [
        (lower + half * (x + 1.0), 0.5 * w) for x, w in zip(nodes, weights, strict=True)
    ]


class ReliabilityReferenceTests(unittest.TestCase):
    """Reliability references agree with independent computations."""

    def test_linear_references_equal_phi_minus_beta(self) -> None:
        """Hyperplane references are Phi(-beta) to rounding."""
        for key, beta in (("linear-2d", 3.0), ("linear-10d", 3.5)):
            value = reference(key, "Pf")
            self.assertEqual(value, 0.5 * math.erfc(beta / math.sqrt(2.0)))
            self.assertAlmostEqual(value / standard_normal_cdf(-beta), 1.0, places=12)

    def test_parabolic_quadrature_matches_gauss_hermite(self) -> None:
        """Simpson on [-12, 12] matches 80-node Gauss-Hermite to 1e-10."""
        nodes, weights = gauss_hermite(80)
        hermite = math.fsum(
            w * standard_normal_cdf(-(3.0 + 0.25 * z * z))
            for z, w in zip(nodes, weights, strict=True)
        )
        self.assertAlmostEqual(reference("parabolic", "Pf") / hermite, 1.0, places=10)

    def test_four_branch_quadrature_is_converged_and_matches_monte_carlo(
        self,
    ) -> None:
        """Doubling the panels changes nothing; MC runs agree within 4 SE."""
        value = reference("four-branch", "Pf")
        self.assertAlmostEqual(
            references.four_branch_failure_probability(panels=8000) / value,
            1.0,
            places=12,
        )
        recorded = problems.FOUR_BRANCH_MC
        self.assertLess(
            abs(recorded["probability"] - value), 3.0 * recorded["standard_error"]
        )
        estimate, error = references.four_branch_monte_carlo(200_000, 7)
        self.assertLess(abs(estimate - value), 4.0 * error)

    def test_lognormal_closed_form_matches_quadrature(self) -> None:
        """P(R <= S) by conditioning on ln S equals the closed form."""
        (mu_r, s_r), (mu_s, s_s) = problems.LN_CAPACITY, problems.LN_DEMAND
        integral = references.simpson(
            lambda z: references.normal_pdf(z)
            * (1.0 - references.normal_tail((mu_s + s_s * z - mu_r) / s_r)),
            -12.0,
            12.0,
            4000,
        )
        self.assertAlmostEqual(
            reference("lognormal-rs", "Pf") / integral, 1.0, places=9
        )


class SensitivityReferenceTests(unittest.TestCase):
    """Sensitivity and moment references agree with numerical integration."""

    def test_ishigami_indices_and_moments_match_quadrature(self) -> None:
        """Gauss-Legendre conditional moments reproduce every closed form."""
        rule = tensor_rule(20, -math.pi, math.pi)
        f = references.ishigami
        grid = {
            (i, j, k): f((rule[i][0], rule[j][0], rule[k][0]))
            for i in range(20)
            for j in range(20)
            for k in range(20)
        }
        w = [weight for _, weight in rule]

        def expect(values: dict[int, float]) -> float:
            """Return the weighted sum over one index."""
            return math.fsum(w[i] * v for i, v in values.items())

        mean = math.fsum(w[i] * w[j] * w[k] * v for (i, j, k), v in grid.items())
        second = math.fsum(w[i] * w[j] * w[k] * v * v for (i, j, k), v in grid.items())
        variance = second - mean * mean
        exact_mean, exact_variance = references.ishigami_moments()
        self.assertAlmostEqual(mean, exact_mean, places=9)
        self.assertAlmostEqual(variance, exact_variance, places=8)

        def conditional_mean_variance(axis: int) -> float:
            """Return Var(E[Y | X_axis])."""
            means = {}
            for a in range(20):
                means[a] = math.fsum(
                    w[b] * w[c] * grid[_place(axis, a, b, c)]
                    for b in range(20)
                    for c in range(20)
                )
            center = expect(means)
            return expect({a: (m - center) ** 2 for a, m in means.items()})

        def expected_conditional_variance(axis: int) -> float:
            """Return E[Var(Y | X_~axis)] (numerator of the total index)."""
            total = 0.0
            for b in range(20):
                for c in range(20):
                    values = {a: grid[_place(axis, a, b, c)] for a in range(20)}
                    center = expect(values)
                    spread = expect({a: (v - center) ** 2 for a, v in values.items()})
                    total += w[b] * w[c] * spread
            return total

        first, total = references.ishigami_indices()
        for axis in range(3):
            name = f"x{axis + 1}"
            self.assertAlmostEqual(
                conditional_mean_variance(axis) / variance, first[axis], places=7
            )
            self.assertAlmostEqual(
                expected_conditional_variance(axis) / variance, total[axis], places=7
            )
            self.assertEqual(reference("ishigami-indices", f"S1[{name}]"), first[axis])
            self.assertEqual(reference("ishigami-indices", f"ST[{name}]"), total[axis])
        shapley = [reference("ishigami-indices", f"Sh[x{i}]") for i in (1, 2, 3)]
        self.assertAlmostEqual(math.fsum(shapley), 1.0, places=12)

    def test_g_function_partial_variances(self) -> None:
        """Simpson (exact on each linear half) gives V_i; 2-D variance product."""
        for a in problems.G_COEFFICIENTS:

            def term(x: float, coefficient: float = a) -> float:
                """Return the centred squared G-function factor."""
                return (
                    (abs(4.0 * x - 2.0) + coefficient) / (1.0 + coefficient) - 1.0
                ) ** 2

            numeric = references.simpson(term, 0.0, 0.5, 2) + references.simpson(
                term, 0.5, 1.0, 2
            )
            self.assertAlmostEqual(numeric, 1.0 / (3.0 * (1.0 + a) ** 2), places=14)
        first, total = references.g_function_indices((0.0, 1.0))
        # Each half carries probability 1/2 of U(0, 1); the kink sits at 0.5.
        rule = [
            (x, 0.5 * w)
            for x, w in tensor_rule(10, 0.0, 0.5) + tensor_rule(10, 0.5, 1.0)
        ]
        values = [
            (wx * wy, references.g_function((x, y), (0.0, 1.0)))
            for x, wx in rule
            for y, wy in rule
        ]
        mean = math.fsum(w * v for w, v in values)
        variance = math.fsum(w * v * v for w, v in values) - mean * mean
        v1, v2 = 1.0 / 3.0, 1.0 / 12.0
        self.assertAlmostEqual(mean, 1.0, places=12)
        self.assertAlmostEqual(variance, (1 + v1) * (1 + v2) - 1.0, places=12)
        self.assertAlmostEqual(first[0], v1 / variance, places=12)
        self.assertAlmostEqual(total[0], v1 * (1 + v2) / variance, places=12)

    def test_linear_shapley_reference_matches_library_and_closed_form(self) -> None:
        """Independent Shapley code equals probstat's and the 2-D formula."""
        independent = [reference("correlated-linear", f"Sh[x{i}]") for i in (1, 2, 3)]
        library = linear_gaussian_shapley(
            problems.SHAPLEY_COEFFICIENTS, problems.shapley_covariance()
        )
        for mine, theirs in zip(independent, library, strict=True):
            self.assertAlmostEqual(mine, theirs, places=12)
        self.assertAlmostEqual(math.fsum(independent), 1.0, places=12)
        b1, b2, s1, s2, rho = 1.0, 2.0, 1.0, 1.5, 0.6
        cov = ((s1 * s1, rho * s1 * s2), (rho * s1 * s2, s2 * s2))
        variance = b1 * b1 * s1 * s1 + 2 * rho * b1 * b2 * s1 * s2 + b2 * b2 * s2 * s2
        closed = (
            b1 * b1 * s1 * s1 * (1 - rho * rho / 2)
            + rho * b1 * b2 * s1 * s2
            + b2 * b2 * s2 * s2 * rho * rho / 2
        ) / variance
        sh = references.linear_gaussian_shapley_reference((b1, b2), cov)
        self.assertAlmostEqual(sh[0], closed, places=12)

    def test_exponential_integral(self) -> None:
        """Each 1-D factor of the 5-D mean and second moment matches Simpson."""
        mean, variance = references.exponential_integral(problems.EXPONENTIAL_RATES)
        product, second = 1.0, 1.0
        for c in problems.EXPONENTIAL_RATES:
            product *= references.simpson(lambda x, c=c: math.exp(c * x), 0, 1, 200)
            second *= references.simpson(lambda x, c=c: math.exp(2 * c * x), 0, 1, 200)
        self.assertAlmostEqual(mean / product, 1.0, places=10)
        self.assertAlmostEqual(variance / (second - product * product), 1.0, places=8)


def _place(axis: int, a: int, b: int, c: int) -> tuple[int, int, int]:
    """Return the grid index with ``a`` on ``axis`` and ``b, c`` elsewhere."""
    if axis == 0:
        return (a, b, c)
    if axis == 1:
        return (b, a, c)
    return (b, c, a)


class CalibrationReferenceTests(unittest.TestCase):
    """The exact posterior equals probstat's conjugate calibration."""

    def test_posterior_matches_conjugate_library_result(self) -> None:
        """Means and sds agree with calibrate_linear_gaussian to 1e-12."""
        for seed in (1, 2, 3):
            _, observations = problems.calibration_data(seed)
            result = calibrate_linear_gaussian(
                problems.CAL_CONDITIONS,
                observations,
                observation_variance=problems.CAL_NOISE_SD**2,
                prior=MultivariateNormal((0.0, 0.0), ((1.0, 0.0), (0.0, 1.0))),
            )
            refs = REGISTRY["linear-calibration"].references(seed)
            for i, name in enumerate(("a", "b")):
                self.assertAlmostEqual(
                    refs[f"mean[{name}]"].value, result.posterior.mean[i], places=12
                )
                self.assertAlmostEqual(
                    refs[f"sd[{name}]"].value,
                    math.sqrt(result.posterior.covariance[i][i]),
                    places=12,
                )

    def test_normal_quantile(self) -> None:
        """The bisection quantile inverts erfc."""
        self.assertAlmostEqual(references.normal_quantile(0.95), 1.6448536269514722)


def _broken(*args: object) -> methods.MethodOutput:
    """Raise like a buggy method would."""
    raise RuntimeError("deliberate failure for the test")


class HarnessTests(unittest.TestCase):
    """Records carry every field and failures never stop a run."""

    sizes = dict(methods.SIZES["quick"], mc_samples=500)

    def test_records_and_failure_capture(self) -> None:
        """A raising method becomes a failed record; later methods still run."""
        chosen = {m.name: m for m in methods.registry()}
        suite = (
            chosen["FORM"],
            methods.Method("Broken", "reliability", _broken),
            chosen["Crude MC"],
        )
        lines: list[str] = []
        records = harness.run_benchmarks(
            (REGISTRY["linear-2d"],),
            suite,
            self.sizes,
            seeds={"reliability": 2},
            progress=lines.append,
        )
        self.assertEqual(len(lines), 1)
        broken = [r for r in records if r["method"] == "Broken"]
        self.assertEqual(len(broken), 2)
        for record in broken:
            self.assertEqual(record["status"], "failed")
            self.assertIn("RuntimeError: deliberate failure", record["error"])
            self.assertIsNone(record["expected_failure"])
        good = [r for r in records if r["status"] == "ok"]
        self.assertEqual({r["method"] for r in good}, {"FORM", "Crude MC"})
        self.assertEqual(len([r for r in good if r["method"] == "FORM"]), 1)
        for record in good:
            self.assertTrue(REQUIRED_FIELDS <= set(record), record.keys())
            self.assertGreater(record["evaluations"], 0)
            self.assertGreaterEqual(record["wall_time_s"], 0.0)
        form = next(r for r in good if r["method"] == "FORM")
        self.assertLess(form["rel_error"], 1e-6)
        self.assertIsNone(form["z"])
        crude = [r for r in good if r["method"] == "Crude MC"]
        self.assertTrue(all(r["covered"] is not None for r in crude))
        summary = harness.aggregate(records, (REGISTRY["linear-2d"],))
        row = next(r for r in summary if r["method"] == "Broken")
        self.assertEqual((row["failed"], row["ok"]), (2, 0))
        text = report.render(
            {
                "command": "test",
                "mode": "quick",
                "generated_utc": "now",
                "total_time_s": 0.0,
                "platform": "test",
                "python": "3",
                "seeds": {"reliability": 2},
            },
            (REGISTRY["linear-2d"],),
            summary,
        )
        self.assertIn("FAILED in 2/2 run(s): RuntimeError", text)
        self.assertIn("**Failures.**", text)
        for line in text.splitlines():
            if line.startswith("|"):
                self.assertLessEqual(line.count("|") - 1, 7, line)

    def test_z_and_coverage_scoring(self) -> None:
        """The z score uses the reported SE (floored); intervals decide coverage."""
        scored = harness._score("Pf", methods.Estimate(1.1, 0.05), 1.0, None)
        self.assertAlmostEqual(scored["z"], 2.0)
        self.assertFalse(scored["covered"])
        exact = harness._score("Pf", methods.Estimate(1.0, 0.0), 1.0, None)
        self.assertEqual(exact["z"], 0.0)
        boxed = harness._score(
            "S1[x1]", methods.Estimate(0.2, None, (0.1, 0.3), 0.95), 0.25, None
        )
        self.assertIsNone(boxed["z"])
        self.assertTrue(boxed["covered"])

    def test_skips_and_expected_failures_are_recorded(self) -> None:
        """Budget skips and known limitations are visible, never silent."""
        chosen = {m.name: m for m in methods.registry()}
        records = harness.run_benchmarks(
            (REGISTRY["linear-10d"], REGISTRY["four-branch"]),
            (chosen["AK-MCS"], chosen["FORM"]),
            self.sizes,
            seeds={"reliability": 1},
        )
        skipped = [r for r in records if r["status"] == "skipped"]
        self.assertEqual([r["problem"] for r in skipped], ["linear-10d"])
        self.assertIn("budget", skipped[0]["notes"][0])
        form = next(
            r
            for r in records
            if r["problem"] == "four-branch" and r["method"] == "FORM"
        )
        self.assertEqual(form["status"], "failed")
        self.assertIsNotNone(form["expected_failure"])

    def test_calibration_records(self) -> None:
        """Laplace on the linear-Gaussian problem is exact and covers the truth."""
        chosen = {m.name: m for m in methods.registry()}
        records = harness.run_benchmarks(
            (REGISTRY["linear-calibration"],),
            (chosen["Laplace"],),
            self.sizes,
            seeds={"calibration": 1},
        )
        by_quantity = {r["quantity"]: r for r in records}
        self.assertLess(by_quantity["mean[a]"]["scaled_error"], 1e-5)
        self.assertLess(abs(by_quantity["sd[b]"]["rel_error"]), 1e-5)
        self.assertIsNotNone(by_quantity["theta[a]"]["covered"])

    def test_command_line_end_to_end(self) -> None:
        """run.py writes JSON and Markdown and exits 0 on a small selection."""
        with tempfile.TemporaryDirectory() as folder:
            json_path = Path(folder) / "out.json"
            md_path = Path(folder) / "out.md"
            buffer = StringIO()
            with redirect_stdout(buffer):
                status = runner.main(
                    [
                        "--quick",
                        "--seeds",
                        "1",
                        "--only",
                        "lognormal-rs",
                        "--json",
                        str(json_path),
                        "--report",
                        str(md_path),
                    ]
                )
            self.assertEqual(status, 0, buffer.getvalue())
            payload = json.loads(json_path.read_text("utf-8"))
            self.assertEqual(payload["meta"]["mode"], "quick")
            self.assertTrue(payload["records"])
            self.assertIn("### Lognormal capacity-demand", md_path.read_text("utf-8"))
            self.assertIn("Done in", buffer.getvalue().splitlines()[-1])


class SeededReproducibilityTests(unittest.TestCase):
    """Data generation is deterministic per seed."""

    def test_calibration_data_is_seeded(self) -> None:
        """The same seed gives the same truth and data; others differ."""
        self.assertEqual(problems.calibration_data(4), problems.calibration_data(4))
        self.assertNotEqual(problems.calibration_data(4), problems.calibration_data(5))
        random.seed(0)  # global state must not matter
        self.assertEqual(problems.calibration_data(4), problems.calibration_data(4))


if __name__ == "__main__":
    unittest.main()
