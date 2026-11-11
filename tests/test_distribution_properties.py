"""Property-based checks over the whole public distribution catalogue (ADR-0087).

Every univariate family exported from ``probstat.distributions`` is drawn
with random valid parameters (``tests/_properties.py``) and checked against
the identities that define a probability law: a bounded, non-decreasing
distribution function with the right limits, ``S = 1 - F``, inverse
consistency of the quantile, a non-negative density whose logarithm agrees
with ``log_pdf``, unit total mass by adaptive quadrature, sample moments
within standard-error bounds, the entropy as ``-E[log f]``, and documented
behaviour at support endpoints, invalid parameters and extreme parameters.
The multivariate laws of ``probstat.multivariate`` are checked against
their univariate reductions and sampling moments.

Tolerances follow forward-error analysis [REF-HIGHAM-2002, Chapter 1]: an
evaluation with relative error ``delta`` in a probability moves the
quantile by ``delta F / f``, so round trips are judged against
``|x| + F(x)/f(x)`` rather than ``|x|`` alone.
"""

from __future__ import annotations

import itertools
import math
import statistics
import unittest
from typing import Any

import probstat.distributions as dist
import probstat.multivariate as mv
from probstat.numerics import integrate
from tests._properties import (
    INVALID_VALUES,
    NON_LAW_EXPORTS,
    Case,
    Draw,
    FamilySpec,
    case_count,
    case_seed,
    categorical_specs,
    continuous_specs,
    discrete_specs,
    for_all,
)

CONTINUOUS = continuous_specs()
DISCRETE = discrete_specs()
CATEGORICAL = categorical_specs()
SAMPLE_SIZE = 2000
# Six standard errors: with about 500 deterministic moment checks the
# chance that a correct sampler fails one under a normal approximation is
# below 1e-6.
Z_BOUND = 6.0
EPS = 2.0**-52
# F and S come from separate special-function evaluations (incomplete
# beta and gamma continued fractions, normal tail ratios). Each multiplies
# a series by exp(t) for a log-prefactor t, and exp has relative condition
# number |t| [REF-HIGHAM-2002, Section 1.6]; with |t| up to about 100 in
# the operating ranges, 256 u bounds |F + S - 1| (observed maximum: 75 u,
# FisherF(1.99, 178) and a normal truncated 11.6 sigma into its tail).
COMPLEMENT_TOLERANCE = 256 * EPS


def _bounds(law: Any) -> tuple[float, float]:
    """Return the support interval ``(lower, upper)`` with infinite ends.

    Laws without a ``support`` attribute (the wrappers) are bounded by
    their extreme quantiles.

    """
    support = getattr(law, "support", None)
    if support is not None and hasattr(support, "lower"):
        lower = -math.inf if support.lower is None else float(support.lower)
        upper = math.inf if support.upper is None else float(support.upper)
        return lower, upper
    return float(law.quantile(0.0)), float(law.quantile(1.0))


def _probability_points(draw: Draw, count: int = 24) -> list[float]:
    """Return sorted probabilities, including both tails."""
    return draw.unit_points(count)


def _interior_points(law: Any, draw: Draw, count: int = 24) -> list[float]:
    """Return sorted support points drawn through the quantile function."""
    lower, upper = _bounds(law)
    points = []
    for p in _probability_points(draw, count):
        x = float(law.quantile(p)) if p <= 0.5 else float(law.isf(1.0 - p))
        if lower < x < upper:
            points.append(x)
    return sorted(points)


def _pieces(law: Any, spec: FamilySpec) -> list[float]:
    """Return quadrature breakpoints: support ends, kinks and quartile points."""
    lower, upper = _bounds(law)
    inner = {float(law.quantile(p)) for p in (0.1, 0.5, 0.9)}
    inner.update(float(point) for point in spec.kinks(law))
    points = sorted(point for point in inner if lower < point < upper)
    return [lower, *points, upper]


def _integral(
    function: Any,
    lower: float,
    upper: float,
    slack: float = 0.0,
    abs_tol: float = 1e-14,
) -> float:
    """Integrate with the library's adaptive rule.

    A run that stops short of its relative target is still accepted when its
    own error estimate is below ``1e-9 + slack``: next to a singular endpoint
    away from zero the nodes round onto the endpoint, so the target cannot be
    met although the estimate is accurate to the unresolved end mass
    (:func:`_unresolved_mass`), which the caller passes as ``slack``. Tail
    integrals pass ``abs_tol=0`` so that the relative target governs.

    """
    result = integrate(function, lower, upper, rel_tol=1e-11, abs_tol=abs_tol)
    if not (result.converged or result.error_estimate <= 1e-9 + slack):
        raise AssertionError(
            f"quadrature on [{lower!r}, {upper!r}] did not converge: {result!r}"
        )
    return result.value


def _unresolved_mass(law: Any) -> float:
    """Return the mass within 64 ulps of the finite support ends.

    Floating-point nodes cannot resolve a density inside this layer; for an
    integrable singularity ``|x - e|^(a - 1)`` with small ``a`` the layer can
    hold a noticeable mass, which bounds the attainable quadrature accuracy
    [REF-HIGHAM-2002, Section 2.1].

    """
    lower, upper = _bounds(law)
    mass = 0.0
    if math.isfinite(lower):
        mass += float(law.cdf(lower + 64.0 * math.ulp(lower)))
    if math.isfinite(upper):
        mass += float(law.sf(upper - 64.0 * math.ulp(upper)))
    return mass


def _density(law: Any, spec: FamilySpec) -> Any:
    """Return ``pdf`` with an infinite value at a support end or kink mapped to zero.

    A quadrature node that rounds onto a singular endpoint (or onto the
    singular end of a mixture component) is a measure-zero artefact of
    floating point; anywhere else an infinite density is a failure.

    """
    lower, upper = _bounds(law)
    singular = {lower, upper, *(float(point) for point in spec.kinks(law))}

    def density(x: float) -> float:
        value = float(law.pdf(x))
        if math.isinf(value):
            if x in singular:
                return 0.0
            raise AssertionError(f"infinite density at interior point {x!r}")
        return value

    return density


def _mass_condition(law: Any) -> float:
    """Return the relative condition number of a summed discrete mass.

    A mass ``exp(t)`` assembled from log-gamma terms of magnitude about
    ``K log K`` (``K`` the largest count that carries mass) has relative
    error about ``K log K`` rounding errors, because the condition number of
    ``exp`` at ``t`` is ``|t|`` [REF-HIGHAM-2002, Section 1.6].

    """
    _, upper = _bounds(law)
    top = float(law.isf(1e-12)) if math.isinf(upper) else upper
    size = abs(top) + 1.0
    return 1.0 + size * math.log1p(size)


def _resolvable(law: Any, x: float) -> bool:
    """Return whether ``[x, upper]`` is wide enough to integrate in floats."""
    _, upper = _bounds(law)
    return math.isinf(upper) or upper - x > 1e-8 * max(1.0, abs(upper))


def _interval_mass(law: Any, a: float, b: float) -> float:
    """Return ``F(b) - F(a)`` from the more accurate side of the median."""
    median = float(law.quantile(0.5))
    if a >= median:
        return float(law.sf(a)) - float(law.sf(b))
    return float(law.cdf(b)) - float(law.cdf(a))


class CatalogueDiscoveryTest(unittest.TestCase):
    """Every exported law has a generator, so none escapes the properties."""

    def test_every_exported_law_has_a_generator(self) -> None:
        """New exports fail here until a ``FamilySpec`` is added for them."""
        covered = {spec.family for spec in CONTINUOUS + DISCRETE + CATEGORICAL}
        missing = [
            name
            for name in dist.__all__
            if name not in NON_LAW_EXPORTS
            and isinstance(getattr(dist, name), type)
            and getattr(dist, name) not in covered
        ]
        self.assertEqual(missing, [])

    def test_generators_produce_valid_parameters(self) -> None:
        """Every generated parameter vector constructs a law (all families)."""

        def check(case: Case) -> None:
            self.assertIsNotNone(case.law)

        for_all(self, "constructs", CONTINUOUS + DISCRETE + CATEGORICAL, check)

    def test_cases_are_deterministic(self) -> None:
        """The same property, family and index always give the same case."""
        self.assertEqual(case_seed("p", "Normal", 3), case_seed("p", "Normal", 3))
        self.assertNotEqual(case_seed("p", "Normal", 3), case_seed("q", "Normal", 3))


class ContinuousPropertiesTest(unittest.TestCase):
    """Defining identities of the continuous univariate laws."""

    def test_cdf_is_monotone_bounded_with_limits(self) -> None:
        """``F`` is non-decreasing in ``[0, 1]`` with ``F(-inf)=0``, ``F(inf)=1``."""

        def check(case: Case) -> None:
            law = case.law
            lower, upper = _bounds(law)
            points = _interior_points(law, case.draw)
            values = [float(law.cdf(x)) for x in points]
            for x, value in zip(points, values, strict=True):
                self.assertTrue(0.0 <= value <= 1.0, (x, value))
            for (x0, f0), (x1, f1) in itertools.pairwise(
                zip(points, values, strict=True)
            ):
                self.assertLessEqual(f0, f1, (x0, x1))
            self.assertEqual(float(law.cdf(-math.inf)), 0.0)
            self.assertEqual(float(law.cdf(math.inf)), 1.0)
            self.assertEqual(float(law.sf(math.inf)), 0.0)
            if math.isfinite(lower):
                self.assertEqual(float(law.cdf(lower)), 0.0)
                self.assertEqual(float(law.pdf(lower - 1.0 - abs(lower))), 0.0)
            if math.isfinite(upper):
                self.assertEqual(float(law.sf(upper)), 0.0)
                self.assertEqual(float(law.pdf(upper + 1.0 + abs(upper))), 0.0)

        for_all(self, "cdf_monotone", CONTINUOUS, check)

    def test_survival_is_complement_of_cdf(self) -> None:
        """``S(x) + F(x) = 1`` to a few rounding errors of the larger term."""

        def check(case: Case) -> None:
            law = case.law
            for x in _interior_points(law, case.draw):
                total = float(law.sf(x)) + float(law.cdf(x))
                self.assertAlmostEqual(
                    total, 1.0, delta=COMPLEMENT_TOLERANCE, msg=repr(x)
                )

        for_all(self, "sf_complement", CONTINUOUS, check)

    def test_quantile_inverts_cdf_and_isf_inverts_sf(self) -> None:
        """``Q(F(x)) = x`` and ``Q*(S(x)) = x`` on the support interior."""

        def check(case: Case) -> None:
            law = case.law
            for x in _interior_points(law, case.draw):
                density = float(law.pdf(x))
                if not density > 0.0:
                    continue
                f_x = float(law.cdf(x))
                s_x = float(law.sf(x))
                # Forward error of Q(F(x)): a relative error delta in F moves
                # the quantile by delta F / f [REF-HIGHAM-2002, Section 1.6];
                # delta = 1e-9 covers the root-finding stopping rules.
                if 0.0 < f_x <= 0.5:
                    back = float(law.quantile(f_x))
                    tolerance = 1e-9 * (abs(x) + f_x / density)
                    self.assertLessEqual(abs(back - x), tolerance, ("Q", x, back))
                if 0.0 < s_x <= 0.5:
                    back = float(law.isf(s_x))
                    tolerance = 1e-9 * (abs(x) + s_x / density)
                    self.assertLessEqual(abs(back - x), tolerance, ("isf", x, back))

        for_all(self, "quantile_round_trip", CONTINUOUS, check)

    def test_density_is_non_negative_and_log_density_agrees(self) -> None:
        """``f >= 0`` and ``log_pdf = log(pdf)`` wherever ``f`` is a normal float."""

        def check(case: Case) -> None:
            law = case.law
            for x in _interior_points(law, case.draw):
                density = float(law.pdf(x))
                self.assertGreaterEqual(density, 0.0, repr(x))
                if density > 1e-300 and math.isfinite(density):
                    log_density = float(law.log_pdf(x))
                    self.assertAlmostEqual(
                        log_density,
                        math.log(density),
                        delta=1e-11 * max(1.0, abs(log_density)),
                        msg=repr(x),
                    )

        for_all(self, "log_pdf", CONTINUOUS, check)

    def test_density_integrates_to_interval_masses(self) -> None:
        """``int_a^b f = F(b) - F(a)`` on each piece and the total mass is one."""

        def check(case: Case) -> None:
            law = case.law
            spec = case.spec
            edges = _pieces(law, spec)
            slack = _unresolved_mass(law)
            delta = 1e-8 + slack
            density = _density(law, spec)
            total = 0.0
            for a, b in itertools.pairwise(edges):
                mass = _integral(density, a, b, slack)
                total += mass
                self.assertAlmostEqual(
                    mass, _interval_mass(law, a, b), delta=delta, msg=repr((a, b))
                )
            self.assertAlmostEqual(total, 1.0, delta=delta)

        for_all(self, "pdf_integrates", CONTINUOUS, check, heavy=True)

    def test_survival_tail_matches_quadrature(self) -> None:
        """``S(x)`` at ``x = isf(1e-10)`` equals ``int_x f`` to relative 1e-6."""

        def check(case: Case) -> None:
            law = case.law
            lower, upper = _bounds(law)
            for q in (1e-6, 1e-10):
                x = float(law.isf(q))
                if not (lower < x < upper and _resolvable(law, x)):
                    continue
                if not float(law.pdf(x)) > 0.0:
                    continue
                density = _density(law, case.spec)
                kinks = sorted(
                    k for k in map(float, case.spec.kinks(law)) if x < k < upper
                )
                edges = [x, *kinks, upper]
                tail = math.fsum(
                    _integral(density, a, b, abs_tol=0.0)
                    for a, b in itertools.pairwise(edges)
                )
                # A survival function formed as a difference of base-law
                # tails (truncation) is conditioned by the cancellation
                # factor of the spec [REF-HIGHAM-2002, Section 1.7].
                delta = 1e-6 + 8 * EPS * case.spec.tail_condition(law, x)
                self.assertAlmostEqual(
                    float(law.sf(x)) / tail, 1.0, delta=delta, msg=repr((q, x))
                )

        for_all(
            self,
            "sf_tail",
            CONTINUOUS,
            check,
            heavy=True,
            applies=lambda spec: "sf" in spec.family.__dict__,
        )

    def test_entropy_matches_quadrature(self) -> None:
        """``H = -int f log f`` by quadrature in ``x`` (an independent rule)."""

        def check(case: Case) -> None:
            law = case.law
            spec = case.spec
            edges = _pieces(law, spec)
            pdf = _density(law, spec)
            # The unresolved end layer of mass m contributes at most about
            # m (1 + |log m|) nats once its density exceeds one.
            unresolved = _unresolved_mass(law)
            slack = (
                100.0 * unresolved * (1.0 + abs(math.log(unresolved)))
                if unresolved > 0.0
                else 0.0
            )

            def integrand(x: float) -> float:
                density = pdf(x)
                return -density * math.log(density) if density > 0.0 else 0.0

            reference = math.fsum(
                _integral(integrand, a, b, slack) for a, b in itertools.pairwise(edges)
            )
            value = float(law.entropy)
            self.assertTrue(math.isfinite(value), value)
            delta = 1e-7 * max(1.0, abs(reference)) + slack
            self.assertAlmostEqual(value, reference, delta=delta)

        for_all(self, "entropy", CONTINUOUS, check, heavy=True)

    def test_sample_moments_within_standard_error(self) -> None:
        """Sample means and variances lie within ``6`` standard errors."""

        def check(case: Case) -> None:
            law = case.law
            draws = law.sample(case.spec.sample_size, seed=case.seed)
            lower, upper = _bounds(law)
            self.assertTrue(all(lower <= x <= upper for x in draws))
            _check_moments(self, law, draws)

        for_all(self, "sample_moments", CONTINUOUS, check, heavy=True)

    def test_quantile_endpoints_and_invalid_arguments(self) -> None:
        """``Q(0)``/``Q(1)`` are the support ends; out-of-range input raises."""

        def check(case: Case) -> None:
            law = case.law
            lower, upper = _bounds(law)
            self.assertEqual(float(law.quantile(0.0)), lower)
            self.assertEqual(float(law.quantile(1.0)), upper)
            self.assertEqual(float(law.isf(1.0)), lower)
            self.assertEqual(float(law.isf(0.0)), upper)
            for bad in (-0.1, 1.1, math.nan):
                with self.assertRaises(ValueError):
                    law.quantile(bad)
            for method in ("pdf", "cdf", "sf"):
                with self.assertRaises(ValueError):
                    getattr(law, method)(math.nan)

        for_all(self, "endpoints", CONTINUOUS, check)

    def test_invalid_parameters_raise(self) -> None:
        """Each constrained parameter rejects zero, negative, ``nan`` and ``inf``."""

        def check(case: Case) -> None:
            _check_invalid(self, case.spec, case.parameters)

        specs = [spec for spec in CONTINUOUS + DISCRETE if spec.constraints]
        for_all(self, "invalid", specs, check, heavy=True)

    def test_extreme_parameters_stay_consistent(self) -> None:
        """At the edges of the operating range ``F`` stays a distribution function."""

        def check(case: Case) -> None:
            law = case.law
            points = _interior_points(law, case.draw, 12)
            values = [float(law.cdf(x)) for x in points]
            for x, value in zip(points, values, strict=True):
                self.assertTrue(0.0 <= value <= 1.0, (x, value))
                self.assertAlmostEqual(
                    value + float(law.sf(x)),
                    1.0,
                    delta=COMPLEMENT_TOLERANCE,
                    msg=repr(x),
                )
                self.assertGreaterEqual(float(law.pdf(x)), 0.0)
            for v0, v1 in itertools.pairwise(values):
                self.assertLessEqual(v0, v1)
            for p in (1e-6, 0.5, 1.0 - 1e-6):
                self.assertFalse(math.isnan(float(law.quantile(p))), p)

        for_all(
            self,
            "extreme",
            CONTINUOUS,
            check,
            extreme=True,
            heavy=True,
            applies=lambda spec: spec.extreme is not None,
        )


class DiscretePropertiesTest(unittest.TestCase):
    """Defining identities of the integer-valued laws."""

    def test_cdf_is_monotone_step_function(self) -> None:
        """``F`` is non-decreasing, constant between integers, with the limits."""

        def check(case: Case) -> None:
            law = case.law
            lower, upper = _bounds(law)
            ks = _integer_points(law, case.draw)
            values = [float(law.cdf(k)) for k in ks]
            for k, value in zip(ks, values, strict=True):
                self.assertTrue(0.0 <= value <= 1.0, (k, value))
                self.assertEqual(float(law.cdf(k + 0.5)), value, k)
            for v0, v1 in itertools.pairwise(values):
                self.assertLessEqual(v0, v1)
            self.assertEqual(float(law.cdf(lower - 1.0)), 0.0)
            self.assertEqual(float(law.cdf(math.inf)), 1.0)
            self.assertEqual(float(law.cdf(-math.inf)), 0.0)
            if math.isfinite(upper):
                self.assertEqual(float(law.cdf(upper)), 1.0)
                self.assertEqual(float(law.sf(upper)), 0.0)

        for_all(self, "cdf_monotone", DISCRETE, check)

    def test_survival_is_complement_and_tail_sum(self) -> None:
        """``S = 1 - F`` and ``S(k) = sum_{j > k} p_j`` to relative 1e-10."""

        def check(case: Case) -> None:
            law = case.law
            delta = 64 * EPS * _mass_condition(law)
            for k in _integer_points(law, case.draw):
                self.assertAlmostEqual(
                    float(law.sf(k)) + float(law.cdf(k)), 1.0, delta=delta
                )
            _, upper = _bounds(law)
            for q in (1e-4, 1e-9):
                k = float(law.isf(q))
                if not k < upper:
                    continue
                stop = float(law.isf(1e-30)) if math.isinf(upper) else upper
                tail = math.fsum(
                    float(law.pmf(float(j))) for j in range(int(k) + 1, int(stop) + 1)
                )
                if tail > 1e-280:
                    # The summed tail omits at most 1e-30 of the mass.
                    self.assertAlmostEqual(
                        float(law.sf(k)) / tail, 1.0, delta=1e-10, msg=repr((q, k))
                    )

        for_all(self, "sf_tail", DISCRETE, check)

    def test_quantile_is_the_generalized_inverse(self) -> None:
        """``Q(p) = min{k : F(k) >= p}`` and ``isf(q) = min{k : S(k) <= q}``."""

        def check(case: Case) -> None:
            law = case.law
            lower, _ = _bounds(law)
            for p in _probability_points(case.draw, 16):
                k = float(law.quantile(p))
                self.assertGreaterEqual(float(law.cdf(k)), p, (p, k))
                if k > lower:
                    self.assertLess(float(law.cdf(k - 1.0)), p, (p, k))
                q = 1.0 - p
                j = float(law.isf(q))
                self.assertLessEqual(float(law.sf(j)), q, (q, j))
                if j > lower:
                    self.assertGreater(float(law.sf(j - 1.0)), q, (q, j))

        for_all(self, "quantile_inverse", DISCRETE, check)

    def test_mass_is_non_negative_and_sums_to_one(self) -> None:
        """``p_k >= 0``, ``log_pmf = log pmf``, ``F(k) = sum_{j<=k} p_j``, total one."""

        def check(case: Case) -> None:
            law = case.law
            lower, upper = _bounds(law)
            first = int(law.quantile(1e-30)) if math.isinf(lower) else int(lower)
            last = int(law.isf(1e-30)) if math.isinf(upper) else int(upper)
            if last - first > 200_000:
                first = int(law.quantile(1e-12))
                last = int(law.isf(1e-12))
            masses = [float(law.pmf(float(k))) for k in range(first, last + 1)]
            self.assertTrue(all(mass >= 0.0 for mass in masses))
            covered = float(law.cdf(last)) - (float(law.cdf(first - 1)))
            self.assertAlmostEqual(math.fsum(masses), covered, delta=1e-12)
            self.assertAlmostEqual(covered, 1.0, delta=2e-12)
            step = max(1, (last - first) // 40)
            running = float(law.cdf(first - 1))
            for k in range(first, last + 1):
                running += masses[k - first]
                if (k - first) % step == 0:
                    self.assertAlmostEqual(float(law.cdf(k)), running, delta=1e-12)
                    mass = masses[k - first]
                    if mass > 1e-300:
                        log_mass = float(law.log_pmf(float(k)))
                        self.assertAlmostEqual(
                            log_mass,
                            math.log(mass),
                            delta=1e-11 * max(1.0, abs(log_mass)),
                        )
            self.assertEqual(float(law.pmf(float(lower - 1))), 0.0)

        for_all(self, "pmf_sums", DISCRETE, check, heavy=True)

    def test_entropy_matches_mass_sum(self) -> None:
        """``H = -sum p_k log p_k`` over the support."""

        def check(case: Case) -> None:
            law = case.law
            lower, upper = _bounds(law)
            first = int(lower) if math.isfinite(lower) else int(law.quantile(1e-30))
            last = int(upper) if math.isfinite(upper) else int(law.isf(1e-30))
            if last - first > 200_000:
                return
            reference = -math.fsum(
                p * math.log(p)
                for p in (float(law.pmf(float(k))) for k in range(first, last + 1))
                if p > 0.0
            )
            self.assertAlmostEqual(
                float(law.entropy), reference, delta=1e-10 * max(1.0, reference)
            )

        for_all(self, "entropy", DISCRETE, check, heavy=True)

    def test_sample_moments_within_standard_error(self) -> None:
        """Sample means and variances lie within ``6`` standard errors."""

        def check(case: Case) -> None:
            law = case.law
            draws = law.sample(SAMPLE_SIZE, seed=case.seed)
            lower, upper = _bounds(law)
            self.assertTrue(
                all(lower <= x <= upper and float(x).is_integer() for x in draws)
            )
            _check_moments(self, law, [float(x) for x in draws])

        for_all(self, "sample_moments", DISCRETE, check, heavy=True)

    def test_endpoints_and_invalid_arguments(self) -> None:
        """``Q(0)`` is the lower end, ``Q(1)`` the upper; bad input raises."""

        def check(case: Case) -> None:
            law = case.law
            lower, upper = _bounds(law)
            self.assertEqual(float(law.quantile(0.0)), lower)
            # Q(1) is the largest point with positive mass: the upper end of
            # the support, or the atom of a degenerate law.
            top = lower if float(law.sf(lower)) == 0.0 else upper
            self.assertEqual(float(law.quantile(1.0)), top)
            self.assertEqual(float(law.isf(0.0)), top)
            self.assertEqual(float(law.isf(1.0)), lower)
            for bad in (-0.1, 1.1, math.nan):
                with self.assertRaises(ValueError):
                    law.quantile(bad)
            with self.assertRaises(TypeError):
                law.pmf(lower + 0.5)
            with self.assertRaises(ValueError):
                law.cdf(math.nan)

        for_all(self, "endpoints", DISCRETE, check)

    def test_extreme_parameters_stay_consistent(self) -> None:
        """At the edges of the operating range ``F`` stays a distribution function."""

        def check(case: Case) -> None:
            law = case.law
            ks = _integer_points(law, case.draw, 12)
            values = [float(law.cdf(k)) for k in ks]
            for k, value in zip(ks, values, strict=True):
                self.assertTrue(0.0 <= value <= 1.0, (k, value))
                self.assertAlmostEqual(
                    value + float(law.sf(k)), 1.0, delta=64 * EPS * _mass_condition(law)
                )
                self.assertGreaterEqual(float(law.pmf(k)), 0.0)
            for v0, v1 in itertools.pairwise(values):
                self.assertLessEqual(v0, v1)
            for p in (1e-6, 0.5, 1.0 - 1e-6):
                k = float(law.quantile(p))
                self.assertGreaterEqual(float(law.cdf(k)), p)

        for_all(
            self,
            "extreme",
            DISCRETE,
            check,
            extreme=True,
            heavy=True,
            applies=lambda spec: spec.extreme is not None,
        )


class CategoricalPropertiesTest(unittest.TestCase):
    """The label-valued law: masses, entropy, and sampling frequencies."""

    def test_masses_entropy_and_frequencies(self) -> None:
        """Masses sum to one, ``H = -sum p log p``, frequencies within 6 SE."""

        def check(case: Case) -> None:
            law = case.law
            probabilities = case.parameters["probabilities"]
            masses = [law.pmf(label) for label in probabilities]
            self.assertAlmostEqual(math.fsum(masses), 1.0, delta=1e-14)
            for label, mass in zip(probabilities, masses, strict=True):
                self.assertAlmostEqual(mass, probabilities[label], delta=1e-15)
                self.assertAlmostEqual(law.log_pmf(label), math.log(mass), delta=1e-12)
            reference = -math.fsum(p * math.log(p) for p in masses if p > 0.0)
            self.assertAlmostEqual(law.entropy, reference, delta=1e-12)
            draws = law.sample(SAMPLE_SIZE, seed=case.seed)
            for label, mass in zip(probabilities, masses, strict=True):
                frequency = sum(1 for x in draws if x == label) / SAMPLE_SIZE
                error = math.sqrt(mass * (1.0 - mass) / SAMPLE_SIZE)
                self.assertLessEqual(abs(frequency - mass), Z_BOUND * error + 1e-12)

        for_all(self, "categorical", CATEGORICAL, check, heavy=True)


class MultivariatePropertiesTest(unittest.TestCase):
    """Multivariate laws against their univariate reductions and moments."""

    def test_multivariate_normal_marginals_and_density(self) -> None:
        """MVN marginals are the univariate normals; ``log_pdf = log pdf``; moments."""
        for index in range(case_count(heavy=True)):
            draw = Draw(case_seed("mvn", "MultivariateNormal", index))
            with self.subTest(case=index, seed=draw.seed):
                dimension = draw.integer(1, 4)
                mean = tuple(draw.uniform(-5, 5) for _ in range(dimension))
                factor = [
                    [draw.uniform(-1, 1) if j <= i else 0.0 for j in range(dimension)]
                    for i in range(dimension)
                ]
                covariance = tuple(
                    tuple(
                        math.fsum(factor[i][k] * factor[j][k] for k in range(dimension))
                        + (0.5 if i == j else 0.0)
                        for j in range(dimension)
                    )
                    for i in range(dimension)
                )
                law = mv.MultivariateNormal(mean, covariance)
                point = tuple(m + draw.uniform(-1, 1) for m in mean)
                self.assertAlmostEqual(
                    law.log_pdf(point), math.log(law.pdf(point)), delta=1e-12
                )
                for i in range(dimension):
                    marginal = law.marginal(i)
                    self.assertAlmostEqual(marginal.mu, mean[i], delta=1e-14)
                    self.assertAlmostEqual(
                        marginal.sigma, math.sqrt(covariance[i][i]), delta=1e-14
                    )
                draws = law.sample(SAMPLE_SIZE, seed=draw.seed)
                for i in range(dimension):
                    _check_mean(self, [x[i] for x in draws], mean[i], covariance[i][i])

    def test_dirichlet_reduces_to_beta(self) -> None:
        """A two-component Dirichlet density is the Beta density; marginals are Beta."""
        for index in range(case_count()):
            draw = Draw(case_seed("dirichlet", "Dirichlet", index))
            with self.subTest(case=index, seed=draw.seed):
                a = draw.log_uniform(0.2, 30.0)
                b = draw.log_uniform(0.2, 30.0)
                law = mv.Dirichlet((a, b))
                x = draw.uniform(0.01, 0.99)
                beta = dist.Beta(a, b)
                self.assertAlmostEqual(
                    law.log_pdf((x, 1.0 - x)),
                    beta.log_pdf(x),
                    delta=1e-10 * max(1.0, abs(beta.log_pdf(x))),
                )
                self.assertAlmostEqual(law.entropy, beta.entropy, delta=1e-8)
                k = draw.integer(3, 5)
                alpha = tuple(draw.log_uniform(0.2, 30.0) for _ in range(k))
                wide = mv.Dirichlet(alpha)
                total = math.fsum(alpha)
                for i in range(k):
                    marginal = wide.marginal(i)
                    self.assertAlmostEqual(marginal.alpha, alpha[i], delta=1e-12)
                    self.assertAlmostEqual(marginal.beta, total - alpha[i], delta=1e-10)

    def test_multinomial_masses_sum_to_one(self) -> None:
        """Multinomial and Dirichlet-multinomial masses sum to one; K = 2: Binomial."""
        for index in range(case_count(heavy=True)):
            draw = Draw(case_seed("multinomial", "Multinomial", index))
            with self.subTest(case=index, seed=draw.seed):
                n = draw.integer(0, 8)
                k = draw.integer(2, 4)
                raw = [draw.uniform(0.05, 1.0) for _ in range(k)]
                probabilities = tuple(value / math.fsum(raw) for value in raw)
                alpha = tuple(draw.log_uniform(0.2, 20.0) for _ in range(k))
                law = mv.Multinomial(n, probabilities)
                compound = mv.DirichletMultinomial(n, alpha)
                outcomes = [
                    counts
                    for counts in itertools.product(range(n + 1), repeat=k)
                    if sum(counts) == n
                ]
                self.assertAlmostEqual(
                    math.fsum(law.pmf(c) for c in outcomes), 1.0, delta=1e-13
                )
                self.assertAlmostEqual(
                    math.fsum(compound.pmf(c) for c in outcomes), 1.0, delta=1e-12
                )
                pair = mv.Multinomial(n, (probabilities[0], 1.0 - probabilities[0]))
                binomial = dist.Binomial(n, probabilities[0])
                for j in range(n + 1):
                    self.assertAlmostEqual(
                        pair.pmf((j, n - j)), binomial.pmf(j), delta=1e-14
                    )

    def test_student_t_and_wishart_sample_means(self) -> None:
        """Multivariate-t and Wishart sample means lie within 6 standard errors."""
        for index in range(case_count(heavy=True)):
            draw = Draw(case_seed("mvt-wishart", "MultivariateStudentT", index))
            with self.subTest(case=index, seed=draw.seed):
                s11 = draw.log_uniform(0.5, 4.0)
                s22 = draw.log_uniform(0.5, 4.0)
                rho = draw.uniform(-0.8, 0.8)
                s12 = rho * math.sqrt(s11 * s22)
                scale = ((s11, s12), (s12, s22))
                nu = draw.uniform(5.0, 30.0)
                location = (draw.uniform(-3, 3), draw.uniform(-3, 3))
                t_law = mv.MultivariateStudentT(location, scale, nu)
                draws = t_law.sample(SAMPLE_SIZE, seed=draw.seed)
                for i in range(2):
                    _check_mean(
                        self, [x[i] for x in draws], location[i], t_law.covariance[i][i]
                    )
                dof = draw.uniform(3.0, 12.0)
                wishart = mv.Wishart(scale, dof)
                samples = wishart.sample(500, seed=draw.seed)
                for i, j in ((0, 0), (0, 1), (1, 1)):
                    # Var W_ij = dof (S_ij^2 + S_ii S_jj) for the Wishart law.
                    variance = dof * (scale[i][j] ** 2 + scale[i][i] * scale[j][j])
                    _check_mean(
                        self, [w[i][j] for w in samples], wishart.mean[i][j], variance
                    )


# ---------------------------------------------------------------------------
# Shared helpers.
# ---------------------------------------------------------------------------


def _integer_points(law: Any, draw: Draw, count: int = 24) -> list[float]:
    """Return sorted integer support points drawn through the quantile."""
    points = {float(law.quantile(p)) for p in _probability_points(draw, count)}
    lower, upper = _bounds(law)
    return sorted(k for k in points if lower <= k <= upper)


def _check_mean(
    test: unittest.TestCase, values: list[float], mean: float, variance: float
) -> None:
    """Assert a sample mean lies within ``Z_BOUND`` standard errors."""
    error = math.sqrt(variance / len(values))
    test.assertLessEqual(
        abs(statistics.fmean(values) - mean), Z_BOUND * error + 1e-12 * abs(mean)
    )


def _check_moments(
    test: unittest.TestCase, law: Any, draws: list[float] | tuple[float, ...]
) -> None:
    """Check the sample mean and variance against their standard errors.

    The mean is checked when the fourth moment is finite: the Berry-Esseen
    bound then holds with ``E|X - mu|^3 <= sigma^3 sqrt(kappa + 3)``. The
    variance is checked only for excess kurtosis ``kappa <= 10``: the normal
    approximation to ``s^2`` is governed by the sixth moment, which is not
    available generically and is large or infinite for heavier tails (a
    Pareto law of shape 4.7, ``kappa = 105``, exceeded six standard errors).
    Heavier-tailed laws are still covered by the quadrature properties.

    """
    n = len(draws)
    mean = float(law.mean)
    variance = float(law.variance)
    if not (math.isfinite(mean) and math.isfinite(variance)):
        return
    values = list(draws)
    if variance == 0.0:
        test.assertTrue(all(x == values[0] for x in values))
        return
    kurtosis = float(law.kurtosis)
    if not math.isfinite(kurtosis):
        return
    _check_mean(test, values, mean, variance)
    if kurtosis > 10.0:
        return
    # Exact Var(s^2) = sigma^4 ((kappa + 2)/n + 2/(n (n - 1))); the second
    # term matters for two-point laws, where kappa = -2.
    error = variance * math.sqrt((kurtosis + 2.0) / n + 2.0 / (n * (n - 1)))
    sample_variance = statistics.variance(values)
    test.assertLessEqual(
        abs(sample_variance - variance), Z_BOUND * error + 1e-12 * variance
    )


def _check_invalid(
    test: unittest.TestCase, spec: FamilySpec, parameters: dict[str, Any]
) -> None:
    """Assert each constrained parameter rejects its invalid values."""
    for name, kind in spec.constraints.items():
        for bad in INVALID_VALUES[kind]:
            altered = dict(parameters)
            altered[name] = bad
            with test.assertRaises((TypeError, ValueError), msg=f"{name}={bad!r}"):
                spec.make(altered)


if __name__ == "__main__":
    unittest.main()
