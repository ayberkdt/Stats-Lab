"""Metamorphic relations for fitting, statistics, inference, UQ, systems (ADR-0087).

A metamorphic relation states how an output must change when an input is
transformed, without knowing the exact output [REF-CHEN-ETAL-2018]. Every
relation below is a theorem for the estimator or algorithm concerned, not
a heuristic: affine equivariance of maximum-likelihood fits and sample
quantiles, permutation invariance of statistics that are functions of the
empirical distribution, sufficiency of conjugate posteriors, monotonicity
of confidence-interval width, invariance of Sobol indices under bijective
input reparameterization and positive output scaling, invariance of the
reliability index under positive scaling of the limit-state function,
order-independence of BDD system reliability, and unbiasedness of Monte
Carlo estimators over seeds.

Equalities that hold exactly in real arithmetic are checked to a few
rounding errors of the computation (summation order changes a floating
sum by at most ``(n - 1) u sum |x_i|`` [REF-HIGHAM-2002, Section 4.2]);
iterative fits are checked to their solver tolerance.
"""

from __future__ import annotations

import functools
import itertools
import math
import statistics as pystats
import unittest
from collections.abc import Callable, Mapping
from typing import Any

import probstat.distributions as dist
import probstat.inference as inf
import probstat.monte_carlo as mc
import probstat.statistics as st
from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
from probstat.reliability import (
    LimitState,
    ModelLimitState,
    demand_failure_probability,
    estimate_failure_probability,
    failure_rate_estimate,
    form_hlrf,
)
from probstat.sensitivity import estimate_sobol_indices, sobol_analysis
from probstat.systems import (
    SystemStructure,
    k_out_of_n,
    parallel,
    series,
)
from probstat.uq import QuantityOfInterest
from tests._properties import Draw, case_count, case_seed

EPS = 2.0**-52
Y = (QuantityOfInterest("y", "response"),)


def _draws(name: str, heavy: bool = False) -> list[Draw]:
    """Return the deterministic case generators of one relation."""
    return [Draw(case_seed(name, "relation", i)) for i in range(case_count(heavy))]


def _sample(draw: Draw, size: int | None = None) -> list[float]:
    """Return a skewed sample of 8 to 60 observations without ties."""
    count = draw.integer(8, 60) if size is None else size
    return [
        draw.random.gammavariate(2.0, 1.5) + 0.1 * index * EPS for index in range(count)
    ]


def _sum_tolerance(values: list[float]) -> float:
    """Return the order-dependence bound ``4 n u sum |x|`` of a floating sum."""
    return 4.0 * len(values) * EPS * math.fsum(abs(v) for v in values)


class AffineEquivarianceTest(unittest.TestCase):
    """Affine transformations of data map estimates by the same transformation."""

    def test_sample_quantiles_and_location_scale_statistics(self) -> None:
        """``Q_p(a + b x) = a + b Q_p(x)`` for ``b > 0`` (type 1 and 7)."""
        for draw in _draws("affine-quantiles"):
            with self.subTest(seed=draw.seed):
                x = _sample(draw)
                a, b = draw.uniform(-50, 50), draw.log_uniform(1e-2, 1e2)
                y = [a + b * v for v in x]
                scale = abs(a) + b * max(abs(v) for v in x)
                for method in (1, 7):
                    for p in (0.0, 0.1, 0.25, 0.5, 0.9, 1.0):
                        self.assertAlmostEqual(
                            st.quantile(y, p, method=method),
                            a + b * st.quantile(x, p, method=method),
                            delta=8 * EPS * scale,
                        )
                self.assertAlmostEqual(
                    st.mean(y), a + b * st.mean(x), delta=8 * EPS * scale
                )
                self.assertAlmostEqual(
                    st.median(y), a + b * st.median(x), delta=8 * EPS * scale
                )
                relative = 1e-10
                self.assertAlmostEqual(
                    st.standard_deviation(y),
                    b * st.standard_deviation(x),
                    delta=relative * b * st.standard_deviation(x),
                )
                self.assertAlmostEqual(
                    st.interquartile_range(y),
                    b * st.interquartile_range(x),
                    delta=16 * EPS * scale,
                )
                self.assertAlmostEqual(
                    st.median_absolute_deviation(y),
                    b * st.median_absolute_deviation(x),
                    delta=16 * EPS * scale,
                )
                self.assertAlmostEqual(st.skewness(y), st.skewness(x), delta=1e-9)
                self.assertAlmostEqual(
                    st.excess_kurtosis(y), st.excess_kurtosis(x), delta=1e-9
                )

    def test_reflection_reverses_type_seven_quantiles(self) -> None:
        """The type-7 quantile is symmetric: ``Q_p(-x) = -Q_(1-p)(x)``."""
        for draw in _draws("reflection-quantiles"):
            with self.subTest(seed=draw.seed):
                x = _sample(draw)
                y = [-v for v in x]
                for p in (0.05, 0.25, 0.5, 0.8):
                    self.assertAlmostEqual(
                        st.quantile(y, p),
                        -st.quantile(x, 1.0 - p),
                        delta=8 * EPS * max(abs(v) for v in x),
                    )
                self.assertAlmostEqual(st.skewness(y), -st.skewness(x), delta=1e-9)

    def test_location_scale_fits_are_equivariant(self) -> None:
        """MLE of ``a + b X`` is the transformed MLE for location-scale families."""
        families = (
            ("NORMAL", inf.NORMAL_FAMILY, ("mu", "sigma")),
            ("LOGISTIC", inf.LOGISTIC_FAMILY, ("mu", "s")),
            ("GUMBEL", inf.GUMBEL_FAMILY, ("mu", "beta")),
            ("UNIFORM", inf.UNIFORM_FAMILY, ("lower", "upper")),
        )
        for draw in _draws("affine-fits", heavy=True):
            x = [
                draw.random.gauss(0.0, 1.0) + 0.3 * draw.random.random()
                for _ in range(40)
            ]
            a, b = draw.uniform(-20, 20), draw.log_uniform(0.1, 10.0)
            y = [a + b * v for v in x]
            for label, family, names in families:
                with self.subTest(family=label, seed=draw.seed):
                    base = inf.fit_distribution(x, family)
                    moved = inf.fit_distribution(y, family)
                    self.assertEqual(base.parameter_names, names)
                    location, spread = base.estimates
                    new_location, new_spread = moved.estimates
                    # Iterative fits stop at a relative gradient tolerance;
                    # 1e-6 of the scale bounds their argmin error.
                    tolerance = 1e-6 * b * max(1.0, abs(spread), abs(location))
                    if label == "UNIFORM":
                        self.assertAlmostEqual(
                            new_location, a + b * location, delta=tolerance
                        )
                        self.assertAlmostEqual(
                            new_spread, a + b * spread, delta=tolerance
                        )
                    else:
                        self.assertAlmostEqual(
                            new_location, a + b * location, delta=tolerance
                        )
                        self.assertAlmostEqual(new_spread, b * spread, delta=tolerance)
                    # The log-likelihood shifts by the Jacobian -n log b.
                    self.assertAlmostEqual(
                        moved.log_likelihood,
                        base.log_likelihood - len(x) * math.log(b),
                        delta=1e-6 * max(1.0, abs(base.log_likelihood)),
                    )

    def test_scale_family_fits_are_equivariant(self) -> None:
        """MLE of ``b X`` rescales the scale and keeps the shape (positive families)."""
        for draw in _draws("scale-fits", heavy=True):
            x = [draw.random.gammavariate(2.5, 1.0) for _ in range(40)]
            b = draw.log_uniform(0.05, 20.0)
            y = [b * v for v in x]
            with self.subTest(family="EXPONENTIAL", seed=draw.seed):
                rate = inf.fit_distribution(x, inf.EXPONENTIAL_FAMILY).estimates[0]
                moved = inf.fit_distribution(y, inf.EXPONENTIAL_FAMILY).estimates[0]
                self.assertAlmostEqual(moved, rate / b, delta=1e-10 * rate / b)
            with self.subTest(family="GAMMA", seed=draw.seed):
                shape, rate = inf.fit_gamma_mle(x).shape, inf.fit_gamma_mle(x).rate
                fitted = inf.fit_gamma_mle(y)
                self.assertAlmostEqual(fitted.shape, shape, delta=1e-7 * shape)
                self.assertAlmostEqual(fitted.rate, rate / b, delta=1e-7 * rate / b)
            with self.subTest(family="WEIBULL", seed=draw.seed):
                base = inf.fit_weibull_mle(x)
                fitted = inf.fit_weibull_mle(y)
                self.assertAlmostEqual(
                    fitted.shape, base.shape, delta=1e-7 * base.shape
                )
                self.assertAlmostEqual(
                    fitted.scale, b * base.scale, delta=1e-7 * b * base.scale
                )
            with self.subTest(family="LOGNORMAL", seed=draw.seed):
                base = inf.fit_lognormal_mle(x)
                fitted = inf.fit_lognormal_mle(y)
                self.assertAlmostEqual(
                    fitted.mu, base.mu + math.log(b), delta=1e-12 * (1 + abs(fitted.mu))
                )
                self.assertAlmostEqual(
                    fitted.sigma, base.sigma, delta=1e-10 * base.sigma
                )

    def test_family_quantiles_are_equivariant(self) -> None:
        """``Q(p; a + b mu, b sigma) = a + b Q(p; mu, sigma)`` (location-scale laws)."""
        builders: dict[str, Callable[[float, float], Any]] = {
            "Normal": dist.Normal,
            "Logistic": dist.Logistic,
            "Laplace": dist.Laplace,
            "Cauchy": dist.Cauchy,
            "Gumbel": dist.Gumbel,
            "StudentT(4)": lambda m, s: dist.StudentT(4.0, m, s),
        }
        for draw in _draws("affine-law-quantiles"):
            mu, sigma = draw.uniform(-5, 5), draw.log_uniform(0.1, 10)
            a, b = draw.uniform(-50, 50), draw.log_uniform(0.01, 100)
            for name, build in builders.items():
                with self.subTest(family=name, seed=draw.seed):
                    law, moved = build(mu, sigma), build(a + b * mu, b * sigma)
                    for p in (1e-8, 0.01, 0.3, 0.5, 0.77, 0.999):
                        x = float(law.quantile(p))
                        # Forward error of a + b x with |x| up to the quantile:
                        # a few ulps of the largest term [REF-HIGHAM-2002, 2.2].
                        scale = abs(a) + b * (abs(mu) + abs(x - mu)) + b * sigma
                        self.assertAlmostEqual(
                            float(moved.quantile(p)), a + b * x, delta=1e-12 * scale
                        )
                    affine = dist.Affine(law, a, b)
                    for p in (0.05, 0.5, 0.95):
                        self.assertAlmostEqual(
                            float(affine.quantile(p)),
                            float(moved.quantile(p)),
                            delta=1e-11 * (abs(a) + b * (abs(mu) + 10 * sigma)),
                        )


class PermutationInvarianceTest(unittest.TestCase):
    """Functions of the empirical distribution do not depend on data order."""

    def test_descriptive_statistics(self) -> None:
        """Order statistics are exact; sums agree within the summation bound."""
        for draw in _draws("permutation-descriptive"):
            with self.subTest(seed=draw.seed):
                x = _sample(draw)
                y = list(x)
                draw.random.shuffle(y)
                for function in (
                    st.median,
                    st.interquartile_range,
                    st.median_absolute_deviation,
                ):
                    self.assertEqual(function(y), function(x))
                self.assertEqual(
                    st.quantiles(y, (0.1, 0.5, 0.9)), st.quantiles(x, (0.1, 0.5, 0.9))
                )
                self.assertEqual(
                    st.quantile(y, 0.3, method=1), st.quantile(x, 0.3, method=1)
                )
                self.assertAlmostEqual(
                    st.mean(y), st.mean(x), delta=_sum_tolerance(x) / len(x)
                )
                for function in (
                    st.variance,
                    st.standard_deviation,
                    st.skewness,
                    st.excess_kurtosis,
                ):
                    self.assertAlmostEqual(
                        function(y),
                        function(x),
                        delta=1e-11 * max(1.0, abs(function(x))),
                    )
                self.assertAlmostEqual(
                    st.trimmed_mean(y, 0.2),
                    st.trimmed_mean(x, 0.2),
                    delta=_sum_tolerance(x) / len(x),
                )
                for t in (min(x), st.median(x), max(x) + 1.0):
                    self.assertEqual(st.empirical_cdf(y, t), st.empirical_cdf(x, t))

    def test_paired_statistics_under_joint_permutation(self) -> None:
        """Correlations are invariant when pairs are permuted together."""
        for draw in _draws("permutation-paired"):
            with self.subTest(seed=draw.seed):
                x = _sample(draw)
                z = [v + draw.random.gauss(0.0, 1.0) for v in x]
                order = list(range(len(x)))
                draw.random.shuffle(order)
                xp, zp = [x[i] for i in order], [z[i] for i in order]
                self.assertAlmostEqual(
                    st.pearson_correlation(xp, zp),
                    st.pearson_correlation(x, z),
                    delta=1e-12,
                )
                self.assertAlmostEqual(
                    st.spearman_correlation(xp, zp),
                    st.spearman_correlation(x, z),
                    delta=1e-12,
                )
                self.assertAlmostEqual(
                    st.covariance(xp, zp),
                    st.covariance(x, z),
                    delta=1e-11 * max(1.0, abs(st.covariance(x, z))),
                )

    def test_edf_and_rank_tests(self) -> None:
        """KS, Anderson-Darling, two-sample KS and Mann-Whitney ignore data order."""
        for draw in _draws("permutation-edf", heavy=True):
            with self.subTest(seed=draw.seed):
                x = [draw.random.gauss(0.3, 1.2) for _ in range(draw.integer(5, 40))]
                w = [draw.random.gauss(0.0, 1.0) for _ in range(draw.integer(5, 40))]
                xs, ws = list(x), list(w)
                draw.random.shuffle(xs)
                draw.random.shuffle(ws)
                law = dist.Normal(0.0, 1.0)
                one, one_p = (
                    inf.kolmogorov_smirnov_test(x, law),
                    inf.kolmogorov_smirnov_test(xs, law),
                )
                self.assertEqual(one_p.statistic, one.statistic)
                self.assertEqual(one_p.p_value, one.p_value)
                self.assertAlmostEqual(
                    inf.anderson_darling_test(xs, law).statistic,
                    inf.anderson_darling_test(x, law).statistic,
                    delta=1e-12,
                )
                two, two_p = (
                    inf.kolmogorov_smirnov_two_sample_test(x, w),
                    inf.kolmogorov_smirnov_two_sample_test(xs, ws),
                )
                self.assertEqual(two_p.statistic, two.statistic)
                self.assertAlmostEqual(two_p.p_value, two.p_value, delta=1e-14)
                # Swapping the samples leaves the two-sided KS test unchanged.
                swapped = inf.kolmogorov_smirnov_two_sample_test(w, x)
                self.assertEqual(swapped.statistic, two.statistic)
                self.assertAlmostEqual(swapped.p_value, two.p_value, delta=1e-12)
                mw, mw_p = (
                    inf.mann_whitney_u_test(x, w),
                    inf.mann_whitney_u_test(xs, ws),
                )
                self.assertEqual(mw_p.statistic, mw.statistic)
                self.assertAlmostEqual(mw_p.p_value, mw.p_value, delta=1e-14)

    def test_ks_statistic_is_invariant_under_joint_monotone_transform(self) -> None:
        """``D_n`` is unchanged when data and law are mapped by an increasing ``h``."""
        for draw in _draws("ks-transform"):
            with self.subTest(seed=draw.seed):
                x = [draw.random.gauss(0.0, 1.0) for _ in range(draw.integer(5, 50))]
                a, b = draw.uniform(-5, 5), draw.log_uniform(0.1, 10)
                law = dist.Normal(0.0, 1.0)
                moved = inf.kolmogorov_smirnov_test(
                    [a + b * v for v in x], dist.Normal(a, b)
                )
                lognormal = inf.kolmogorov_smirnov_test(
                    [math.exp(v) for v in x], dist.LogNormal(0.0, 1.0)
                )
                base = inf.kolmogorov_smirnov_test(x, law)
                self.assertAlmostEqual(moved.statistic, base.statistic, delta=1e-12)
                self.assertAlmostEqual(lognormal.statistic, base.statistic, delta=1e-12)


class SufficiencyTest(unittest.TestCase):
    """Conjugate posteriors depend on the data only through sufficient statistics."""

    def test_poisson_gamma_posterior(self) -> None:
        """Raw-data Poisson posterior is the conjugate ``Gamma(a + sum x, b + n)``."""
        for draw in _draws("sufficiency-poisson"):
            with self.subTest(seed=draw.seed):
                n = draw.integer(1, 12)
                counts = [draw.integer(0, 9) for _ in range(n)]
                prior = dist.Gamma(
                    draw.log_uniform(0.5, 5.0), draw.log_uniform(0.5, 5.0)
                )
                posterior = failure_rate_estimate(
                    sum(counts), float(n), prior=prior
                ).posterior
                # A second data set with the same n and sum (mass moved
                # between observations) must give the same posterior.
                other = list(counts)
                if n > 1 and other[0] > 0:
                    other[0] -= 1
                    other[1] += 1
                draw.random.shuffle(other)
                grid = [posterior.quantile(p) for p in (0.05, 0.3, 0.5, 0.8, 0.97)]
                for data in (counts, other):
                    log_ratios = [
                        prior.log_pdf(lam)
                        + math.fsum(dist.Poisson(lam).log_pmf(k) for k in data)
                        - posterior.log_pdf(lam)
                        for lam in grid
                    ]
                    self.assertLess(max(log_ratios) - min(log_ratios), 1e-9)
                check = inf.check_sufficiency(
                    lambda lam, s: math.fsum(dist.Poisson(lam).log_pmf(k) for k in s),
                    lambda s: float(sum(s)),
                    tuple(counts),
                    tuple(other),
                    grid,
                )
                self.assertTrue(check.same_statistic and check.consistent)

    def test_binomial_beta_posterior(self) -> None:
        """Bernoulli sequences with equal success counts share one Beta posterior."""
        for draw in _draws("sufficiency-binomial"):
            with self.subTest(seed=draw.seed):
                n = draw.integer(1, 30)
                bits = [1 if draw.random.random() < 0.3 else 0 for _ in range(n)]
                prior = dist.Beta(
                    draw.log_uniform(0.5, 5.0), draw.log_uniform(0.5, 5.0)
                )
                posterior = demand_failure_probability(
                    sum(bits), n, prior=prior
                ).posterior
                shuffled = list(bits)
                draw.random.shuffle(shuffled)
                grid = [posterior.quantile(p) for p in (0.05, 0.4, 0.6, 0.95)]
                for data in (bits, shuffled):
                    log_ratios = [
                        prior.log_pdf(p)
                        + math.fsum(dist.Bernoulli(p).log_pmf(b) for b in data)
                        - posterior.log_pdf(p)
                        for p in grid
                    ]
                    self.assertLess(max(log_ratios) - min(log_ratios), 1e-9)
                # Grouping the trials into binomial batches changes nothing.
                batched = demand_failure_probability(
                    sum(bits[: n // 2]) + sum(bits[n // 2 :]), n, prior=prior
                ).posterior
                self.assertEqual(
                    (batched.alpha, batched.beta), (posterior.alpha, posterior.beta)
                )

    def test_normal_inverse_gamma_posterior(self) -> None:
        """Data with equal ``n``, ``sum y`` and ``sum y^2`` share one NIG posterior."""
        for draw in _draws("sufficiency-normal"):
            with self.subTest(seed=draw.seed):
                n = draw.integer(3, 25)
                y = [draw.random.gauss(1.0, 2.0) for _ in range(n)]
                # A Householder reflection that fixes the ones vector keeps the
                # sum and the sum of squares, hence (n, ybar, s^2).
                v = [draw.random.gauss(0.0, 1.0) for _ in range(n)]
                centre = math.fsum(v) / n
                v = [t - centre for t in v]
                vv = math.fsum(t * t for t in v)
                vy = math.fsum(t * s for t, s in zip(v, y, strict=True))
                z = [s - 2.0 * vy / vv * t for t, s in zip(v, y, strict=True)]
                self.assertAlmostEqual(math.fsum(z), math.fsum(y), delta=1e-10 * n)
                prior = inf.NormalInverseGamma(
                    (draw.uniform(-1, 1),),
                    ((draw.log_uniform(0.1, 10.0),),),
                    draw.log_uniform(0.5, 5.0),
                    draw.log_uniform(0.5, 5.0),
                )
                design = [(1.0,)] * n
                first = inf.condition_linear_normal_inverse_gamma(prior, design, y)
                second = inf.condition_linear_normal_inverse_gamma(prior, design, z)
                scale = 1.0 + math.fsum(t * t for t in y)
                self.assertAlmostEqual(
                    first.mean[0], second.mean[0], delta=1e-10 * scale
                )
                self.assertAlmostEqual(
                    first.scale_matrix[0][0], second.scale_matrix[0][0], delta=1e-14
                )
                self.assertEqual(first.shape, second.shape)
                self.assertAlmostEqual(first.rate, second.rate, delta=1e-10 * scale)


class IntervalMonotonicityTest(unittest.TestCase):
    """Interval width grows with the confidence level and shrinks with the sample."""

    LEVELS = (0.5, 0.8, 0.9, 0.95, 0.99, 0.999)

    def test_width_increases_with_confidence(self) -> None:
        """Normal-z, Student-t, Wilson, Clopper-Pearson and Poisson intervals nest."""
        for draw in _draws("ci-confidence"):
            with self.subTest(seed=draw.seed):
                self._confidence_case(draw)

    def _confidence_case(self, draw: Draw) -> None:
        """Check nesting over ``LEVELS`` for one generated data set."""
        x = [draw.random.gauss(3.0, 2.0) for _ in range(draw.integer(3, 40))]
        trials = draw.integer(1, 200)
        successes = draw.integer(0, trials)
        counts = [draw.integer(0, 8) for _ in range(draw.integer(1, 20))]
        builders: tuple[Callable[[float], Any], ...] = (
            lambda c: inf.normal_mean_confidence_interval(
                x, confidence=c, known_sigma=2.0
            ),
            lambda c: inf.student_t_mean_confidence_interval(x, confidence=c),
            lambda c: inf.wilson_score_interval(successes, trials, confidence=c),
            lambda c: inf.clopper_pearson_interval(successes, trials, confidence=c),
            lambda c: inf.poisson_mean_interval(counts, c),
        )
        for build in builders:
            intervals = [build(c) for c in self.LEVELS]
            for small, large in itertools.pairwise(intervals):
                self.assertLessEqual(large.lower, small.lower)
                self.assertGreaterEqual(large.upper, small.upper)
            widths = [i.upper - i.lower for i in intervals]
            # Clopper-Pearson keeps its lower end at 0 for x = 0, so widths
            # are compared non-strictly.
            for w0, w1 in itertools.pairwise(widths):
                self.assertLessEqual(w0, w1)

    def test_width_decreases_with_sample_size(self) -> None:
        """Replicating the data ``k`` times (same estimate) narrows every interval."""
        for draw in _draws("ci-sample-size"):
            with self.subTest(seed=draw.seed):
                x = [draw.random.gauss(3.0, 2.0) for _ in range(draw.integer(3, 15))]
                trials = draw.integer(1, 40)
                successes = draw.integer(0, trials)
                counts = [draw.integer(0, 8) for _ in range(draw.integer(1, 10))]
                if sum(counts) == 0:
                    counts[0] = 1
                confidence = draw.choice((0.8, 0.9, 0.95, 0.99))
                previous: dict[str, float] = {}
                for k in (1, 2, 3, 5, 8):
                    widths = {
                        "z": inf.normal_mean_confidence_interval(
                            x * k, confidence=confidence, known_sigma=2.0
                        ),
                        "t": inf.student_t_mean_confidence_interval(
                            x * k, confidence=confidence
                        ),
                        "wilson": inf.wilson_score_interval(
                            k * successes, k * trials, confidence=confidence
                        ),
                        "cp": inf.clopper_pearson_interval(
                            k * successes, k * trials, confidence=confidence
                        ),
                        "poisson": inf.poisson_mean_interval(counts * k, confidence),
                    }
                    for name, interval in widths.items():
                        width = interval.upper - interval.lower
                        if name in previous:
                            self.assertLess(width, previous[name], (name, k))
                        previous[name] = width


class SobolInvarianceTest(unittest.TestCase):
    """Sobol indices depend on the input sigma-algebras and the output up to scale."""

    @staticmethod
    def _model(
        laws: Mapping[str, Any], function: Callable[[Mapping[str, float]], float]
    ) -> ProbabilisticModel:
        """Wrap ``function`` of independent inputs as a one-QoI model."""
        inputs = JointInput.independent(
            *(RandomInput(name, law) for name, law in laws.items())
        )
        return ProbabilisticModel(inputs, lambda p: {"y": function(p)}, Y)

    def test_bijective_input_reparameterization(self) -> None:
        """Uniform inputs through ``Q_i`` give the indices of the reparameterized model.

        ``S_i`` depends only on the joint law of ``(X_i, Y)`` through
        ``sigma(X_i)``; a strictly increasing ``X_i = Q_i(U_i)`` generates
        the same sigma-algebra, so ``g(Q(U))`` with uniform ``U`` and ``g(X)``
        with ``X ~ Q(U)`` have identical indices, and the same design rows
        map to the same model values.
        """
        laws = {
            "a": dist.Exponential(1.5),
            "b": dist.Normal(0.0, 2.0),
            "c": dist.Weibull(1.7, 1.0),
        }

        def g(p: Mapping[str, float]) -> float:
            """Return a non-additive response."""
            return p["a"] * math.exp(0.3 * p["b"]) + p["c"] ** 2 + p["a"] * p["c"]

        uniform = {name: dist.Uniform(0.0, 1.0) for name in laws}

        def g_of_u(p: Mapping[str, float]) -> float:
            """Return ``g(Q(u))``."""
            return g({name: float(law.quantile(p[name])) for name, law in laws.items()})

        for draw in _draws("sobol-reparameterization", heavy=True)[:3]:
            with self.subTest(seed=draw.seed):
                seed = draw.integer(1, 10**6)
                direct = sobol_analysis(
                    self._model(laws, g), "y", 256, design="sobol", seed=seed
                )
                through = sobol_analysis(
                    self._model(uniform, g_of_u), "y", 256, design="sobol", seed=seed
                )
                for name in laws:
                    first, total = direct.as_mapping()[name]
                    first_u, total_u = through.as_mapping()[name]
                    # Equal up to the rounding of Q(F(Q(u))) in the inputs.
                    self.assertAlmostEqual(first, first_u, delta=1e-8)
                    self.assertAlmostEqual(total, total_u, delta=1e-8)

    def test_pick_freeze_estimator_identities(self) -> None:
        """Positive output scaling, monotone input maps and input permutation."""
        for draw in _draws("sobol-identities", heavy=True):
            with self.subTest(seed=draw.seed):
                self._pick_freeze_case(draw)

    def _pick_freeze_case(self, draw: Draw) -> None:
        """Check the three estimator identities on one pair of base samples."""
        n, d = 128, 3
        a = [tuple(draw.random.random() for _ in range(d)) for _ in range(n)]
        b = [tuple(draw.random.random() for _ in range(d)) for _ in range(n)]
        coefficients = [draw.uniform(0.5, 3.0) for _ in range(d)]

        def model(u: tuple[float, ...]) -> float:
            """Return an Ishigami-like response on the unit cube."""
            x = [math.pi * (2.0 * t - 1.0) for t in u]
            return (
                math.sin(x[0])
                + coefficients[1] * math.sin(x[1]) ** 2
                + coefficients[2] * x[2] ** 4 * math.sin(x[0]) / 10.0
            )

        base = estimate_sobol_indices(model, a, b)
        c = draw.log_uniform(1e-3, 1e3)
        scaled = estimate_sobol_indices(lambda u: c * model(u), a, b)
        # Monotone maps of the inputs with the inverse map inside the model.
        warp = [lambda t: t**3, math.expm1, lambda t: math.log1p(9.0 * t)]
        unwarp = [
            lambda s: s ** (1.0 / 3.0),
            math.log1p,
            lambda s: math.expm1(s) / 9.0,
        ]
        aw = [tuple(warp[i](row[i]) for i in range(d)) for row in a]
        bw = [tuple(warp[i](row[i]) for i in range(d)) for row in b]
        warped = estimate_sobol_indices(
            lambda s: model(tuple(unwarp[i](s[i]) for i in range(d))), aw, bw
        )
        perm = (2, 0, 1)
        ap = [tuple(row[j] for j in perm) for row in a]
        bp = [tuple(row[j] for j in perm) for row in b]
        inverse = {j: k for k, j in enumerate(perm)}
        permuted = estimate_sobol_indices(
            lambda s: model(tuple(s[inverse[i]] for i in range(d))), ap, bp
        )
        for i in range(d):
            # Ratios of sums of c^2-scaled terms: a few ulps per term.
            self.assertAlmostEqual(
                scaled.first_order[i], base.first_order[i], delta=1e-12
            )
            self.assertAlmostEqual(
                scaled.total_effect[i], base.total_effect[i], delta=1e-12
            )
            # The warp and its inverse round once each.
            self.assertAlmostEqual(
                warped.first_order[i], base.first_order[i], delta=1e-10
            )
            self.assertAlmostEqual(
                warped.total_effect[i], base.total_effect[i], delta=1e-10
            )
            j = perm.index(i)
            self.assertAlmostEqual(
                permuted.first_order[j], base.first_order[i], delta=1e-12
            )
            self.assertAlmostEqual(
                permuted.total_effect[j], base.total_effect[i], delta=1e-12
            )


class ReliabilityInvarianceTest(unittest.TestCase):
    """Reliability measures depend on the failure set, not on how it is described."""

    def test_form_index_is_invariant_to_limit_state_scaling(self) -> None:
        """``beta`` and the design point of ``c g`` equal those of ``g`` (``c > 0``)."""
        for draw in _draws("form-scaling", heavy=True):
            with self.subTest(seed=draw.seed):
                inputs = JointInput.independent(
                    RandomInput("r", dist.LogNormal(draw.uniform(1.0, 1.5), 0.1)),
                    RandomInput("s", dist.Gumbel(draw.uniform(1.0, 2.0), 0.3)),
                    RandomInput("t", dist.Normal(0.0, 0.5)),
                )
                c = draw.log_uniform(1e-3, 1e3)
                base = form_hlrf(_scaled_state(inputs, 1.0))
                scaled = form_hlrf(_scaled_state(inputs, c))
                self.assertTrue(base.converged and scaled.converged)
                # Both runs stop at the HL-RF tolerance 1e-6 on the iterate.
                self.assertAlmostEqual(
                    scaled.beta, base.beta, delta=1e-5 * max(1.0, base.beta)
                )
                for u, v in zip(
                    scaled.design_point_u, base.design_point_u, strict=True
                ):
                    self.assertAlmostEqual(u, v, delta=1e-4)

    def test_direct_estimate_is_invariant_to_limit_state_scaling(self) -> None:
        """Crude Monte Carlo counts exactly the same failures for ``g`` and ``c g``."""

        def g(x: tuple[float, ...]) -> float:
            """Return a nonlinear limit state."""
            return x[0] - x[1] ** 2 - 1.0

        for draw in _draws("mc-scaling"):
            with self.subTest(seed=draw.seed):
                samples = [
                    (draw.random.gauss(3.0, 1.0), draw.random.gauss(0.0, 1.0))
                    for _ in range(300)
                ]
                c = draw.log_uniform(1e-3, 1e3)
                base = estimate_failure_probability(LimitState(g, "g"), samples)
                scaled = estimate_failure_probability(
                    LimitState(functools.partial(_times, c, g), "cg"), samples
                )
                self.assertEqual(scaled.failure.probability, base.failure.probability)
                self.assertEqual(scaled.failure.successes, base.failure.successes)

    def test_bdd_reliability_is_order_independent(self) -> None:
        """System reliability is the same for every variable order and after sifting."""
        for draw in _draws("bdd-order", heavy=True):
            with self.subTest(seed=draw.seed):
                names = [f"c{i}" for i in range(draw.integer(4, 8))]
                expression = _random_structure(draw, names)
                p = {name: draw.uniform(0.5, 0.999) for name in names}
                reference = None
                for _ in range(4):
                    order = list(names)
                    draw.random.shuffle(order)
                    structure = SystemStructure(expression, order=order)
                    value = structure.reliability(p)
                    if reference is None:
                        reference = value
                        exact = _enumerated_reliability(structure, names, p)
                        self.assertAlmostEqual(value, exact, delta=1e-14)
                    self.assertAlmostEqual(value, reference, delta=1e-14)
                    sifted = structure.bdd.sift(structure.root)
                    q = {name: 1.0 - value_p for name, value_p in p.items()}
                    self.assertAlmostEqual(
                        sifted.manager.probability(sifted.roots[0], p, q),
                        reference,
                        delta=1e-14,
                    )


class MonteCarloUnbiasednessTest(unittest.TestCase):
    """The average of estimates over independent seeds converges to the truth."""

    REPLICATES = 40

    def _check(self, estimates: list[Any], truth: float) -> None:
        """Assert the seed-average is within 6 SE and reported SEs are calibrated."""
        values = [e.estimate for e in estimates]
        spread = pystats.stdev(values)
        self.assertLessEqual(
            abs(pystats.fmean(values) - truth),
            6.0 * spread / math.sqrt(len(values)) + 1e-15,
        )
        reported = pystats.fmean(e.standard_error for e in estimates)
        # The seed-to-seed spread estimates the true standard error with
        # relative error about 1/sqrt(2 R) = 0.11 for R = 40.
        self.assertTrue(0.6 <= reported / spread <= 1.6, (reported, spread))

    def test_uniform_antithetic_stratified_and_importance(self) -> None:
        """Plain, antithetic, stratified and importance-sampling means are unbiased."""
        for draw in _draws("mc-unbiased", heavy=True)[:4]:
            k = draw.uniform(0.5, 3.0)
            f = functools.partial(_exponential, k)
            truth = math.expm1(k) / k
            seeds = [draw.integer(1, 10**9) for _ in range(self.REPLICATES)]
            with self.subTest(estimator="uniform", seed=draw.seed):
                self._check(
                    [
                        mc.estimate_uniform_integral(
                            f, 0.0, 1.0, sample_size=200, seed=s
                        )
                        for s in seeds
                    ],
                    truth,
                )
            with self.subTest(estimator="antithetic", seed=draw.seed):
                self._check(
                    [
                        mc.estimate_antithetic_uniform(f, pairs=100, seed=s)
                        for s in seeds
                    ],
                    truth,
                )
            with self.subTest(estimator="stratified", seed=draw.seed):
                self._check(
                    [
                        mc.estimate_stratified_uniform(f, [20] * 10, seed=s)
                        for s in seeds
                    ],
                    truth,
                )
            with self.subTest(estimator="importance", seed=draw.seed):
                # E[X^2] = 1 for X ~ N(0, 1) with a N(0, 1.5^2) proposal.
                target, proposal = dist.Normal(0.0, 1.0), dist.Normal(0.0, 1.5)
                results = [
                    mc.estimate_importance_sampling(
                        lambda x: x * x,
                        target.pdf,
                        proposal.pdf,
                        lambda rng: rng.gauss(0.0, 1.5),
                        sample_size=200,
                        seed=s,
                    )
                    for s in seeds
                ]
                self._check([r.estimate for r in results], 1.0)


def _exponential(k: float, u: float) -> float:
    """Return ``exp(k u)``, whose integral over ``[0, 1]`` is ``expm1(k)/k``."""
    return math.exp(k * u)


def _times(c: float, function: Callable[[tuple[float, ...]], float], x: Any) -> float:
    """Return ``c function(x)``."""
    return c * function(x)


def _scaled_state(inputs: JointInput, factor: float) -> ModelLimitState:
    """Return ``factor (r - s - 0.1 t^2)`` on ``inputs``, failing below zero."""

    def response(p: Mapping[str, float]) -> dict[str, float]:
        """Evaluate the scaled limit-state function."""
        return {"y": factor * (p["r"] - p["s"] - 0.1 * p["t"] ** 2)}

    model = ProbabilisticModel(inputs, response, Y)
    return ModelLimitState(model, "y", 0.0, "below")


def _random_structure(draw: Draw, names: list[str]) -> Any:
    """Return a random coherent series/parallel/k-out-of-n structure over ``names``."""
    items: list[Any] = list(names)
    draw.random.shuffle(items)
    while len(items) > 1:
        size = min(len(items), draw.integer(2, 3))
        group, items = items[:size], items[size:]
        kind = draw.choice(("series", "parallel", "vote"))
        if kind == "series":
            node = series(*group)
        elif kind == "parallel":
            node = parallel(*group)
        else:
            node = k_out_of_n(draw.integer(1, len(group)), *group)
        items.insert(draw.integer(0, len(items)), node)
    return items[0]


def _enumerated_reliability(
    structure: SystemStructure, names: list[str], p: Mapping[str, float]
) -> float:
    """Return ``P(phi = 1)`` by summing over all ``2^n`` component states."""
    total = []
    for bits in itertools.product((0, 1), repeat=len(names)):
        state = dict(zip(names, bits, strict=True))
        if structure.bdd.evaluate(structure.root, state):
            weight = 1.0
            for name, bit in state.items():
                weight *= p[name] if bit else 1.0 - p[name]
            total.append(weight)
    return math.fsum(total)


if __name__ == "__main__":
    unittest.main()
