"""Tests for linear and generalized linear mixed models (ADR-0100).

Oracles, none of which uses the code under test:

* balanced one-way random-effects ANOVA: ML and REML variance components,
  log-likelihoods, BLUPs, and conditional variances in closed form
  (Searle, Casella and McCulloch 1992, Chapters 3-4), including the
  boundary solutions;
* balanced two-level nested design: REML equals the ANOVA estimators;
* balanced random-coefficient design (every group has the same design
  ``X_i = Z_i = X``): the per-group least-squares estimates ``b_i`` are
  sufficient with ``b_i ~ N(beta, Omega)``, ``Omega = Sigma_b +
  sigma^2 (X'X)^-1``, independent of the residual sum of squares, so ML and
  REML give ``Sigma_b = S - MSE (X'X)^-1`` with the divisor ``a`` or ``a - 1``
  for ``S``, the BLUP ``Sigma_b Omega^-1 (b_i - beta)``, and Satterthwaite df
  ``a - 1`` (REML) for every fixed effect;
* a dense marginal Gaussian likelihood ``N(X beta, sigma^2 (I + Z Lambda
  Lambda' Z'))`` with GLS for unbalanced, crossed, and slope designs;
* the exact conditional posterior of ``hierarchical_normal_posterior``;
* the exact ``F`` law of ``MSA / MSE`` for the bootstrap null distribution
  of the variance-component likelihood ratio;
* adaptive Gauss-Kronrod integration of every group integral of a GLMM;
* seeded simulation (interval coverage within Monte Carlo error).

References:
    [REF-BATES-ETAL-2015], [REF-KUZNETSOVA-ETAL-2017],
    [REF-SEARLE-CASELLA-MCCULLOCH-1992], [REF-LIU-PIERCE-1994].

"""

from __future__ import annotations

import math
import random
import unittest
import warnings

from probstat.inference import (
    MixedModelLikelihoodRatioTest,
    fit_generalized_linear_mixed_model,
    fit_glm,
    fit_linear_mixed_model,
    glmm_deviance,
    hierarchical_normal_posterior,
    mixed_model_likelihood_ratio_test,
    random_effect,
)
from probstat.inference.mixed_models import (
    _gauss_hermite,
    _laplace_state,
    _prepare_glmm,
)
from probstat.inference.sampling_distributions import f_sf
from probstat.linalg import inverse_spd, log_det_spd, solve_spd
from probstat.numerics import integrate


def one_way(a: int, m: int, sd_group: float, sd_noise: float, seed: int):
    """Simulate a balanced one-way layout and return ``(groups, y)``."""
    rng = random.Random(seed)
    groups, y = [], []
    for i in range(a):
        effect = rng.gauss(0.0, sd_group)
        for _ in range(m):
            groups.append(i)
            y.append(10.0 + effect + rng.gauss(0.0, sd_noise))
    return groups, y


def anova_one_way(groups, y, a: int, m: int):
    """Return ``(group means, grand mean, SSA, SSE)``."""
    means = [math.fsum(y[i * m : (i + 1) * m]) / m for i in range(a)]
    grand = math.fsum(y) / len(y)
    ssa = m * math.fsum((v - grand) ** 2 for v in means)
    sse = math.fsum((v - means[g]) ** 2 for v, g in zip(y, groups, strict=True))
    return means, grand, ssa, sse


def dense_log_likelihood(x, y, z_columns, sigma_b, sigma, reml):
    """Return the marginal (or restricted) log-likelihood by dense algebra.

    ``z_columns`` is a list of ``(Z_k, G_k)``: an ``n x q_k`` matrix and the
    ``q_k x q_k`` covariance of its random effects. GLS gives ``beta``.
    """
    del sigma_b
    n, p = len(y), len(x[0])
    v = [[(sigma * sigma if i == j else 0.0) for j in range(n)] for i in range(n)]
    for zk, gk in z_columns:
        q = len(gk)
        zg = [
            [math.fsum(zk[i][r] * gk[r][c] for r in range(q)) for c in range(q)]
            for i in range(n)
        ]
        for i in range(n):
            for j in range(n):
                v[i][j] += math.fsum(zg[i][c] * zk[j][c] for c in range(q))
    vt = tuple(tuple(row) for row in v)
    vinv_x = [solve_spd(vt, tuple(row[j] for row in x)) for j in range(p)]
    xtvx = tuple(
        tuple(math.fsum(x[i][k] * vinv_x[j][i] for i in range(n)) for k in range(p))
        for j in range(p)
    )
    vinv_y = solve_spd(vt, tuple(y))
    xtvy = tuple(math.fsum(x[i][j] * vinv_y[i] for i in range(n)) for j in range(p))
    beta = solve_spd(xtvx, xtvy)
    r = tuple(
        y[i] - math.fsum(b * xv for b, xv in zip(beta, x[i], strict=True))
        for i in range(n)
    )
    quad = math.fsum(a * b for a, b in zip(r, solve_spd(vt, r), strict=True))
    value = log_det_spd(vt) + quad
    if reml:
        value += log_det_spd(xtvx) + (n - p) * math.log(2.0 * math.pi)
    else:
        value += n * math.log(2.0 * math.pi)
    return -0.5 * value, beta


class OneWayClosedFormTests(unittest.TestCase):
    """Balanced one-way random effects against ANOVA closed forms."""

    @classmethod
    def setUpClass(cls) -> None:
        """Simulate one interior data set and fit it by ML and REML."""
        cls.a, cls.m = 9, 6
        cls.groups, cls.y = one_way(cls.a, cls.m, 2.0, 1.0, seed=3)
        cls.x = [[1.0] for _ in cls.y]
        cls.term = random_effect(cls.groups, name="g")
        cls.reml = fit_linear_mixed_model(cls.x, cls.y, cls.term)
        cls.ml = fit_linear_mixed_model(cls.x, cls.y, cls.term, method="ML")
        cls.means, cls.grand, cls.ssa, cls.sse = anova_one_way(
            cls.groups, cls.y, cls.a, cls.m
        )

    def test_reml_variance_components_equal_anova_estimators(self) -> None:
        """REML: sigma^2 = MSE, sigma_b^2 = (MSA - MSE) / m."""
        a, m = self.a, self.m
        msa, mse = self.ssa / (a - 1), self.sse / (a * (m - 1))
        self.assertGreater(msa, mse)
        fit = self.reml
        self.assertAlmostEqual(fit.residual_variance, mse, delta=1e-7 * mse)
        sb2 = fit.variance_components[0].covariance[0][0]
        self.assertAlmostEqual(sb2, (msa - mse) / m, delta=1e-6 * sb2)
        self.assertAlmostEqual(fit.coefficients[0], self.grand, places=10)
        self.assertAlmostEqual(
            fit.standard_errors[0], math.sqrt(msa / (a * m)), delta=1e-6
        )
        self.assertFalse(fit.singular)
        self.assertTrue(fit.converged)

    def test_ml_variance_components_equal_closed_form(self) -> None:
        """ML: sigma^2 = MSE, sigma_b^2 = (SSA / a - MSE) / m."""
        a, m = self.a, self.m
        mse = self.sse / (a * (m - 1))
        fit = self.ml
        self.assertAlmostEqual(fit.residual_variance, mse, delta=1e-7 * mse)
        sb2 = fit.variance_components[0].covariance[0][0]
        self.assertAlmostEqual(sb2, (self.ssa / a - mse) / m, delta=1e-6 * sb2)
        lam = mse + m * sb2
        self.assertAlmostEqual(
            fit.standard_errors[0], math.sqrt(lam / (a * m)), delta=1e-6
        )

    def test_log_likelihoods_equal_closed_forms(self) -> None:
        """ML and REML log-likelihoods of the one-way model in closed form.

        ``-2 l = N log 2 pi + (N - a) log s2 + a log lam + SSE/s2 + SSA/lam``
        and ``-2 l_R = (N - 1) log 2 pi + (N - a) log s2 + (a - 1) log lam
        + log N + SSE/s2 + SSA/lam`` with ``lam = s2 + m sb2``.
        """
        a, m = self.a, self.m
        n = a * m
        for fit, reml in ((self.ml, False), (self.reml, True)):
            s2 = fit.residual_variance
            lam = s2 + m * fit.variance_components[0].covariance[0][0]
            value = (n - a) * math.log(s2) + self.sse / s2 + self.ssa / lam
            if reml:
                value += (n - 1) * math.log(2 * math.pi) + (a - 1) * math.log(lam)
                value += math.log(n)
            else:
                value += n * math.log(2 * math.pi) + a * math.log(lam)
            self.assertAlmostEqual(fit.log_likelihood, -0.5 * value, places=8)
            self.assertAlmostEqual(fit.deviance, value, places=7)
            k = 3
            self.assertAlmostEqual(fit.aic, value + 2 * k, places=7)
            self.assertAlmostEqual(fit.bic, value + k * math.log(n), places=7)

    def test_blups_and_conditional_variances(self) -> None:
        """BLUP = m sb2 / (m sb2 + s2) (ybar_i - beta); var = (1/sb2 + m/s2)^-1."""
        fit, m = self.reml, self.m
        s2 = fit.residual_variance
        sb2 = fit.variance_components[0].covariance[0][0]
        shrink = m * sb2 / (m * sb2 + s2)
        estimates = fit.random_effects[0]
        for level, mean in enumerate(self.means):
            self.assertAlmostEqual(
                estimates.for_level(level)[0],
                shrink * (mean - fit.coefficients[0]),
                delta=1e-6,
            )
            self.assertAlmostEqual(
                estimates.conditional_covariances[level][0][0],
                1.0 / (1.0 / sb2 + m / s2),
                delta=1e-8,
            )
        for i, g in enumerate(self.groups):
            self.assertAlmostEqual(
                fit.fitted_values[i],
                fit.coefficients[0] + estimates.values[g][0],
                places=10,
            )
            self.assertAlmostEqual(
                fit.residuals[i] + fit.fitted_values[i], self.y[i], places=10
            )
            self.assertAlmostEqual(fit.marginal_fitted_values[i], fit.coefficients[0])

    def test_satterthwaite_df_equals_group_count_minus_one(self) -> None:
        """Var(beta) = MSA / N is a single mean square on a - 1 df."""
        self.assertAlmostEqual(self.reml.degrees_of_freedom[0], self.a - 1, delta=1e-4)
        contrast = self.reml.test_contrast([1.0])
        self.assertAlmostEqual(contrast.degrees_of_freedom, self.a - 1, delta=1e-4)
        self.assertAlmostEqual(contrast.estimate, self.reml.coefficients[0])
        low, high = self.reml.coefficient_intervals(0.95)[0]
        from probstat.inference import student_t_quantile

        half = student_t_quantile(0.975, self.a - 1) * self.reml.standard_errors[0]
        self.assertAlmostEqual(high - low, 2 * half, delta=1e-5)

    def test_reml_agrees_with_the_hierarchical_normal_posterior(self) -> None:
        """With s_j^2 = sigma^2 / m the eight-schools model is the LMM.

        Integrating ``mu`` under a flat prior leaves the restricted
        likelihood, so the posterior mode of ``tau`` (uniform prior) is the
        REML ``sigma_b``; at that ``tau`` the conditional posterior of ``mu``
        is ``N(beta, Var beta)`` and ``E[theta_j]`` is ``beta + b_j``.
        """
        fit = self.reml
        s = math.sqrt(fit.residual_variance / self.m)
        posterior = hierarchical_normal_posterior(
            self.means, [s] * self.a, grid_size=4001
        )
        tau = fit.variance_components[0].standard_deviations[0]
        mean, variance = posterior.conditional_mu(tau)
        self.assertAlmostEqual(mean, fit.coefficients[0], places=10)
        self.assertAlmostEqual(variance, fit.covariance[0][0], delta=1e-7)
        theta_means, theta_variances = posterior.conditional_theta(tau)
        shrink = s * s / (s * s + tau * tau)
        for j, (value, var) in enumerate(
            zip(theta_means, theta_variances, strict=True)
        ):
            b = fit.random_effects[0].values[j][0]
            self.assertAlmostEqual(value, fit.coefficients[0] + b, delta=1e-6)
            self.assertAlmostEqual(
                var - shrink * shrink * variance,
                fit.random_effects[0].conditional_covariances[j][0][0],
                delta=1e-7,
            )
        index = max(
            range(len(posterior.tau_grid)), key=posterior.log_density.__getitem__
        )
        spacing = posterior.tau_grid[index + 1] - posterior.tau_grid[index - 1]
        self.assertLess(abs(posterior.tau_grid[index] - tau), spacing)


class BoundaryTests(unittest.TestCase):
    """Variance components on the boundary of the parameter space."""

    def test_reml_boundary_pools_the_sums_of_squares(self) -> None:
        """MSA <= MSE gives sigma_b = 0 and sigma^2 = (SSA + SSE) / (N - 1)."""
        a, m = 8, 5
        for seed in range(40):
            groups, y = one_way(a, m, 0.0, 1.0, seed=seed)
            _, grand, ssa, sse = anova_one_way(groups, y, a, m)
            if ssa / (a - 1) < 0.8 * sse / (a * (m - 1)):
                break
        fit = fit_linear_mixed_model([[1.0]] * len(y), y, random_effect(groups))
        self.assertTrue(fit.singular)
        self.assertLess(fit.variance_components[0].standard_deviations[0], 1e-4)
        self.assertAlmostEqual(
            fit.residual_variance, (ssa + sse) / (a * m - 1), delta=1e-8
        )
        self.assertAlmostEqual(fit.degrees_of_freedom[0], a * m - 1, delta=1e-3)
        self.assertAlmostEqual(fit.coefficients[0], grand, places=10)
        ml = fit_linear_mixed_model(
            [[1.0]] * len(y), y, random_effect(groups), method="ML"
        )
        self.assertAlmostEqual(ml.residual_variance, (ssa + sse) / (a * m), delta=1e-8)
        self.assertGreaterEqual(fit.theta[0], 0.0)

    def test_boundary_occurs_exactly_when_the_anova_solution_is_negative(self) -> None:
        """sigma_b = 0 iff MSA <= MSE (REML) or SSA / a <= MSE (ML).

        Regression test: the deviance is even in ``theta``, so ``theta = 0``
        is a stationary point; an optimizer that stopped there reported a
        boundary fit for data whose ANOVA solution is positive.
        """
        a, m = 6, 4
        for seed in range(60):
            groups, y = one_way(a, m, 0.3, 1.0, seed=100 + seed)
            _, _, ssa, sse = anova_one_way(groups, y, a, m)
            mse = sse / (a * (m - 1))
            term = random_effect(groups)
            x = [[1.0]] * len(y)
            reml = fit_linear_mixed_model(x, y, term, denominator_df="asymptotic")
            ml = fit_linear_mixed_model(
                x, y, term, method="ML", denominator_df="asymptotic"
            )
            with self.subTest(seed=seed):
                self.assertEqual(reml.singular, ssa / (a - 1) <= mse)
                self.assertEqual(ml.singular, ssa / a <= mse)

    def test_nelder_mead_reaches_the_same_optimum(self) -> None:
        """The derivative-free optimizer agrees with BFGS."""
        groups, y = one_way(7, 4, 1.5, 1.0, seed=9)
        x = [[1.0]] * len(y)
        bfgs = fit_linear_mixed_model(x, y, random_effect(groups))
        simplex = fit_linear_mixed_model(
            x, y, random_effect(groups), optimizer="nelder-mead"
        )
        self.assertAlmostEqual(bfgs.deviance, simplex.deviance, places=7)
        self.assertAlmostEqual(bfgs.theta[0], simplex.theta[0], delta=1e-4)

    def test_empty_random_effects_is_least_squares(self) -> None:
        """No random terms: REML sigma^2 is the OLS mean squared error."""
        groups, y = one_way(5, 4, 1.0, 1.0, seed=8)
        x = [[1.0, float(i % 4)] for i in range(len(y))]
        fit = fit_linear_mixed_model(x, y, [])
        from probstat.inference import fit_linear_model

        ols = fit_linear_model(x, y)
        for a, b in zip(fit.coefficients, ols.coefficients, strict=True):
            self.assertAlmostEqual(a, b, places=10)
        for a, b in zip(fit.standard_errors, ols.standard_errors, strict=True):
            self.assertAlmostEqual(a, b, places=10)
        for df in fit.degrees_of_freedom:
            self.assertAlmostEqual(df, len(y) - 2, delta=1e-4)


class NestedAndRandomSlopeTests(unittest.TestCase):
    """Balanced nested and random-coefficient designs in closed form."""

    def test_nested_reml_equals_anova(self) -> None:
        """Balanced two-level nesting: REML is the ANOVA solution."""
        rng = random.Random(21)
        a, b, m = 7, 3, 4
        outer, inner, y = [], [], []
        for i in range(a):
            ea = rng.gauss(0.0, 2.0)
            for j in range(b):
                eb = rng.gauss(0.0, 1.0)
                for _ in range(m):
                    outer.append(i)
                    inner.append((i, j))
                    y.append(5.0 + ea + eb + rng.gauss(0.0, 0.7))
        n = a * b * m
        grand = math.fsum(y) / n
        cell = {}
        for key, value in zip(inner, y, strict=True):
            cell.setdefault(key, []).append(value)
        cell_mean = {k: math.fsum(v) / m for k, v in cell.items()}
        outer_mean = [
            math.fsum(cell_mean[(i, j)] for j in range(b)) / b for i in range(a)
        ]
        msa = b * m * math.fsum((v - grand) ** 2 for v in outer_mean) / (a - 1)
        msb = (
            m
            * math.fsum(
                (cell_mean[(i, j)] - outer_mean[i]) ** 2
                for i in range(a)
                for j in range(b)
            )
            / (a * (b - 1))
        )
        mse = math.fsum(
            (v - cell_mean[k]) ** 2 for k, v in zip(inner, y, strict=True)
        ) / (a * b * (m - 1))
        self.assertTrue(msa > msb > mse)
        fit = fit_linear_mixed_model(
            [[1.0]] * n,
            y,
            [random_effect(outer, name="a"), random_effect(inner, name="a:b")],
        )
        components = {c.group: c.covariance[0][0] for c in fit.variance_components}
        self.assertAlmostEqual(fit.residual_variance, mse, delta=1e-7)
        self.assertAlmostEqual(components["a:b"], (msb - mse) / m, delta=1e-6)
        self.assertAlmostEqual(components["a"], (msa - msb) / (b * m), delta=1e-6)
        self.assertAlmostEqual(fit.standard_errors[0], math.sqrt(msa / n), delta=1e-6)
        self.assertAlmostEqual(fit.degrees_of_freedom[0], a - 1, delta=1e-3)

    def test_balanced_random_coefficients(self) -> None:
        """Correlated intercepts and slopes from per-group least squares."""
        rng = random.Random(5)
        a, m = 12, 6
        xs = [float(j) for j in range(m)]
        groups, x, y = [], [], []
        for i in range(a):
            b0, b1 = rng.gauss(0.0, 3.0), rng.gauss(0.0, 1.0)
            for value in xs:
                groups.append(i)
                x.append(value)
                y.append(20.0 + b0 + (2.0 + b1) * value + rng.gauss(0.0, 1.5))
        xbar = math.fsum(xs) / m
        sxx = math.fsum((v - xbar) ** 2 for v in xs)
        per_group, rss = [], 0.0
        for i in range(a):
            yy = y[i * m : (i + 1) * m]
            ybar = math.fsum(yy) / m
            slope = (
                math.fsum((v - xbar) * (w - ybar) for v, w in zip(xs, yy, strict=True))
                / sxx
            )
            per_group.append((ybar - slope * xbar, slope))
            rss += math.fsum(
                (w - per_group[-1][0] - slope * v) ** 2
                for v, w in zip(xs, yy, strict=True)
            )
        mse = rss / (a * (m - 2))
        centre = [math.fsum(e[k] for e in per_group) / a for k in range(2)]
        xtx_inv = inverse_spd(
            ((m, math.fsum(xs)), (math.fsum(xs), math.fsum(v * v for v in xs)))
        )
        term = random_effect(groups, slopes={"x": x}, name="g")
        for method, divisor in (("REML", a - 1), ("ML", a)):
            with self.subTest(method=method):
                fit = fit_linear_mixed_model(
                    [[1.0, v] for v in x], y, term, method=method
                )
                s = [
                    [
                        math.fsum(
                            (e[j] - centre[j]) * (e[k] - centre[k]) for e in per_group
                        )
                        / divisor
                        for k in range(2)
                    ]
                    for j in range(2)
                ]
                sigma_b = [
                    [s[j][k] - mse * xtx_inv[j][k] for k in range(2)] for j in range(2)
                ]
                self.assertAlmostEqual(fit.residual_variance, mse, delta=1e-7)
                for j in range(2):
                    self.assertAlmostEqual(fit.coefficients[j], centre[j], places=8)
                    self.assertAlmostEqual(
                        fit.standard_errors[j], math.sqrt(s[j][j] / a), delta=1e-6
                    )
                    for k in range(2):
                        self.assertAlmostEqual(
                            fit.variance_components[0].covariance[j][k],
                            sigma_b[j][k],
                            delta=1e-6,
                        )
                omega = tuple(
                    tuple(sigma_b[j][k] + mse * xtx_inv[j][k] for k in range(2))
                    for j in range(2)
                )
                omega_inv = inverse_spd(omega)
                for i in (0, 5, 11):
                    d = [per_group[i][k] - centre[k] for k in range(2)]
                    blup = [
                        math.fsum(
                            sigma_b[j][h]
                            * math.fsum(omega_inv[h][k] * d[k] for k in range(2))
                            for h in range(2)
                        )
                        for j in range(2)
                    ]
                    for value, expected in zip(
                        fit.random_effects[0].values[i], blup, strict=True
                    ):
                        self.assertAlmostEqual(value, expected, delta=1e-6)
                if method == "REML":
                    for df in fit.degrees_of_freedom:
                        self.assertAlmostEqual(df, a - 1, delta=1e-3)
                    contrast = fit.test_contrast([1.0, 1.0])
                    self.assertAlmostEqual(
                        contrast.degrees_of_freedom, a - 1, delta=1e-3
                    )
                    expected = math.sqrt((s[0][0] + 2 * s[0][1] + s[1][1]) / a)
                    self.assertAlmostEqual(
                        contrast.standard_error, expected, delta=1e-6
                    )


class DenseLikelihoodTests(unittest.TestCase):
    """The profiled deviance agrees with a dense marginal likelihood."""

    def test_unbalanced_crossed_and_slope_designs(self) -> None:
        """Maximized log-likelihoods equal the dense formula at the estimates."""
        rng = random.Random(17)
        n = 60
        subject = [rng.randrange(7) for _ in range(n)]
        item = [rng.randrange(5) for _ in range(n)]
        x = [rng.uniform(-1.0, 1.0) for _ in range(n)]
        effects_s = [(rng.gauss(0, 1.0), rng.gauss(0, 0.5)) for _ in range(7)]
        effects_i = [rng.gauss(0, 0.8) for _ in range(5)]
        y = [
            1.0
            + 0.5 * x[k]
            + effects_s[subject[k]][0]
            + effects_s[subject[k]][1] * x[k]
            + effects_i[item[k]]
            + rng.gauss(0.0, 0.6)
            for k in range(n)
        ]
        design = [[1.0, v] for v in x]
        terms = [
            random_effect(subject, slopes={"x": x}, name="subject"),
            random_effect(item, name="item"),
        ]
        for method in ("REML", "ML"):
            with self.subTest(method=method):
                fit = fit_linear_mixed_model(design, y, terms, method=method)
                z_parts = []
                for term, component in zip(terms, fit.variance_components, strict=True):
                    r, levels = term.size, term.n_levels
                    zk = [[0.0] * (r * levels) for _ in range(n)]
                    gk = [[0.0] * (r * levels) for _ in range(r * levels)]
                    for k in range(n):
                        level = term.level_index[k]
                        for c in range(r):
                            zk[k][level * r + c] = term.columns[k][c]
                    for level in range(levels):
                        for c in range(r):
                            for d in range(r):
                                gk[level * r + c][level * r + d] = component.covariance[
                                    c
                                ][d]
                    z_parts.append((zk, gk))
                value, beta = dense_log_likelihood(
                    design, y, z_parts, None, fit.residual_sd, method == "REML"
                )
                self.assertAlmostEqual(fit.log_likelihood, value, places=7)
                for b_fit, b_dense in zip(fit.coefficients, beta, strict=True):
                    self.assertAlmostEqual(b_fit, b_dense, places=7)

    def test_optimum_is_stationary_for_the_dense_likelihood(self) -> None:
        """Perturbing any variance parameter lowers the dense ML likelihood."""
        groups, y = one_way(6, 3, 1.0, 1.0, seed=44)
        groups = groups[:-2]
        y = y[:-2]
        n = len(y)
        fit = fit_linear_mixed_model([[1.0]] * n, y, random_effect(groups), method="ML")
        sb, s = fit.variance_components[0].standard_deviations[0], fit.residual_sd
        z = [[1.0 if groups[k] == j else 0.0 for j in range(6)] for k in range(n)]

        def loglik(sb_value: float, s_value: float) -> float:
            g = [[sb_value**2 if i == j else 0.0 for j in range(6)] for i in range(6)]
            return dense_log_likelihood([[1.0]] * n, y, [(z, g)], None, s_value, False)[
                0
            ]

        best = loglik(sb, s)
        self.assertAlmostEqual(best, fit.log_likelihood, places=8)
        for dsb, ds in ((1e-3, 0), (-1e-3, 0), (0, 1e-3), (0, -1e-3)):
            self.assertLess(loglik(sb + dsb, s + ds), best)


class LikelihoodRatioTests(unittest.TestCase):
    """Likelihood-ratio tests between nested models."""

    def test_statistic_df_and_comparability_rules(self) -> None:
        """T = 2 (l_full - l_reduced) with chi-square and mixture p-values."""
        groups, y = one_way(10, 5, 1.0, 1.0, seed=2)
        x = [[1.0]] * len(y)
        null = fit_linear_mixed_model(x, y, [], method="ML")
        full = fit_linear_mixed_model(x, y, random_effect(groups), method="ML")
        test = mixed_model_likelihood_ratio_test(null, full)
        self.assertIsInstance(test, MixedModelLikelihoodRatioTest)
        statistic = 2 * (full.log_likelihood - null.log_likelihood)
        self.assertAlmostEqual(test.statistic, statistic)
        self.assertEqual(test.degrees_of_freedom, 1)
        from probstat.inference import chi_square_sf

        self.assertAlmostEqual(test.p_value, chi_square_sf(statistic, 1))
        self.assertAlmostEqual(test.boundary_mixture_p_value, 0.5 * test.p_value)
        self.assertTrue(math.isnan(test.bootstrap_p_value))
        other = fit_linear_mixed_model(
            [[1.0, float(i % 5)] for i in range(len(y))], y, random_effect(groups)
        )
        reml_null = fit_linear_mixed_model(x, y, random_effect(groups))
        with self.assertRaises(ValueError):
            mixed_model_likelihood_ratio_test(reml_null, other)
        with self.assertRaises(ValueError):
            mixed_model_likelihood_ratio_test(full, null)
        with self.assertRaises(ValueError):
            mixed_model_likelihood_ratio_test(null, other)

    def test_bootstrap_null_mass_at_zero_matches_the_f_law(self) -> None:
        """Under the null, P(T* = 0) = P(F(a-1, a(m-1)) <= a / (a-1)) for ML.

        The ML estimate of ``sigma_b`` is zero exactly when
        ``SSA / a <= MSE``; simulated from the null fit, ``MSA / MSE`` is
        ``F``-distributed whatever the fitted mean and variance.
        """
        a, m = 6, 4
        groups, y = one_way(a, m, 0.6, 1.0, seed=12)
        x = [[1.0]] * len(y)
        null = fit_linear_mixed_model(x, y, [], method="ML")
        full = fit_linear_mixed_model(x, y, random_effect(groups), method="ML")
        samples = 300
        test = mixed_model_likelihood_ratio_test(
            null, full, bootstrap_samples=samples, seed=2026
        )
        self.assertEqual(len(test.bootstrap_statistics), samples)
        zero = sum(1 for t in test.bootstrap_statistics if t < 1e-6) / samples
        expected = 1.0 - f_sf(a / (a - 1.0), a - 1, a * (m - 1))
        self.assertLess(
            abs(zero - expected), 4 * math.sqrt(expected * (1 - expected) / samples)
        )
        exceed = sum(
            1
            for t in test.bootstrap_statistics
            if t >= test.statistic - 1e-8 * (1 + test.statistic)
        )
        self.assertAlmostEqual(test.bootstrap_p_value, (1 + exceed) / (samples + 1))
        again = mixed_model_likelihood_ratio_test(
            null, full, bootstrap_samples=5, seed=7
        )
        repeat = mixed_model_likelihood_ratio_test(
            null, full, bootstrap_samples=5, seed=7
        )
        self.assertEqual(again.bootstrap_statistics, repeat.bootstrap_statistics)


class CoverageTests(unittest.TestCase):
    """Seeded simulation of fixed-effect interval coverage."""

    def test_satterthwaite_intervals_cover_at_nominal_rate(self) -> None:
        """95% t intervals on Satterthwaite df cover within Monte Carlo error."""
        rng = random.Random(99)
        sizes = (3, 5, 4, 6, 3, 5, 4, 7)
        beta = (2.0, -1.0)
        replicates = 300
        hits = [0, 0]
        for _ in range(replicates):
            groups, x, y = [], [], []
            for g, size in enumerate(sizes):
                effect = rng.gauss(0.0, 1.0)
                for _ in range(size):
                    value = rng.uniform(-1.0, 1.0)
                    groups.append(g)
                    x.append([1.0, value])
                    y.append(beta[0] + beta[1] * value + effect + rng.gauss(0.0, 0.8))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = fit_linear_mixed_model(x, y, random_effect(groups))
            for j, (low, high) in enumerate(fit.coefficient_intervals(0.95)):
                hits[j] += low <= beta[j] <= high
        tolerance = 3.5 * math.sqrt(0.95 * 0.05 / replicates)
        for count in hits:
            self.assertLess(abs(count / replicates - 0.95), tolerance)


class GaussHermiteTests(unittest.TestCase):
    """Gauss-Hermite rules integrate polynomials against exp(-x^2) exactly."""

    def test_moments_are_exact(self) -> None:
        """Sum w x^(2k) = Gamma(k + 1/2) for 2k <= 2n - 1."""
        for order in (1, 2, 5, 12, 30):
            nodes, weights = _gauss_hermite(order)
            self.assertEqual(len(nodes), order)
            for k in range(order):
                value = math.fsum(
                    w * x ** (2 * k) for x, w in zip(nodes, weights, strict=True)
                )
                self.assertAlmostEqual(value / math.gamma(k + 0.5), 1.0, delta=1e-12)
                odd = math.fsum(
                    w * x ** (2 * k + 1) for x, w in zip(nodes, weights, strict=True)
                )
                self.assertAlmostEqual(odd, 0.0, delta=1e-12 * math.gamma(k + 1.0))


def binomial_groups(a: int, m: int, seed: int, sd: float = 1.2):
    """Simulate a logistic random-intercept data set."""
    rng = random.Random(seed)
    groups, x, y = [], [], []
    for i in range(a):
        effect = rng.gauss(0.0, sd)
        for _ in range(m):
            value = rng.uniform(-1.0, 1.0)
            p = 1.0 / (1.0 + math.exp(-(-0.5 + value + effect)))
            groups.append(i)
            x.append([1.0, value])
            y.append(1.0 if rng.random() < p else 0.0)
    return groups, x, y


def poisson_draw(rng: random.Random, mean: float) -> int:
    """Draw a Poisson count by multiplying uniforms."""
    limit, count, product = math.exp(-mean), 0, rng.random()
    while product >= limit:
        count += 1
        product *= rng.random()
    return count


class GeneralizedMixedModelTests(unittest.TestCase):
    """Laplace and adaptive Gauss-Hermite GLMM likelihoods and fits."""

    def test_quadrature_converges_to_direct_integration(self) -> None:
        """AGHQ deviance converges to Gauss-Kronrod integration of each group."""
        groups, x, y = binomial_groups(10, 6, seed=11)
        term = random_effect(groups)
        theta, beta = 1.3, (-0.4, 1.1)
        total = 0.0
        for j in range(10):
            members = [i for i, g in enumerate(groups) if g == j]

            def integrand(u: float, members=members) -> float:
                log_p = -0.5 * u * u - 0.5 * math.log(2 * math.pi)
                for i in members:
                    eta = beta[0] + beta[1] * x[i][1] + theta * u
                    log_p += y[i] * eta - math.log1p(math.exp(eta))
                return math.exp(log_p)

            total += math.log(integrate(integrand, -12.0, 12.0, rel_tol=1e-13).value)
        exact = -2.0 * total
        errors = {
            k: abs(
                glmm_deviance(x, y, term, [theta], beta, quadrature_points=k) - exact
            )
            for k in (1, 3, 9, 25)
        }
        self.assertGreater(errors[1], 1e-2)
        self.assertLess(errors[3], errors[1])
        self.assertLess(errors[9], errors[3])
        self.assertLess(errors[25], 1e-8)

    def test_one_node_equals_the_general_laplace_path(self) -> None:
        """The scalar AGHQ rule with one node is the PIRLS Laplace deviance."""
        groups, x, y = binomial_groups(8, 5, seed=4)
        term = random_effect(groups)
        data, _, _ = _prepare_glmm(x, y, term, "binomial", None, None, None)
        laplace = _laplace_state(data, [0.9], [-0.2, 0.7]).deviance
        quadrature = glmm_deviance(x, y, term, [0.9], [-0.2, 0.7], quadrature_points=1)
        self.assertAlmostEqual(laplace, quadrature, places=9)

    def test_zero_variance_reduces_to_the_glm(self) -> None:
        """Theta = 0 gives the binomial GLM deviance with trials as weights."""
        rng = random.Random(6)
        groups = [i // 3 for i in range(24)]
        trials = [float(rng.randint(2, 9)) for _ in range(24)]
        x = [[1.0, rng.uniform(-1, 1)] for _ in range(24)]
        y = [rng.randint(0, int(t)) / t for t in trials]
        glm = fit_glm(x, y, family="binomial", weights=trials)
        for k in (1, 7):
            value = glmm_deviance(
                x,
                y,
                random_effect(groups),
                [0.0],
                glm.coefficients,
                weights=trials,
                quadrature_points=k,
            )
            self.assertAlmostEqual(value, -2 * glm.log_likelihood, places=8)

    def test_fits_agree_and_quadrature_stabilizes(self) -> None:
        """Laplace and AGHQ fits are close; AGHQ estimates settle with nodes."""
        groups, x, y = binomial_groups(15, 8, seed=11)
        term = random_effect(groups)
        laplace = fit_generalized_linear_mixed_model(x, y, term)
        fits = {
            k: fit_generalized_linear_mixed_model(
                x, y, term, method="adaptive_quadrature", quadrature_points=k
            )
            for k in (5, 15)
        }
        for j in range(2):
            self.assertLess(
                abs(laplace.coefficients[j] - fits[15].coefficients[j]),
                0.2 * fits[15].standard_errors[j],
            )
            self.assertAlmostEqual(
                fits[5].coefficients[j], fits[15].coefficients[j], delta=1e-3
            )
        self.assertAlmostEqual(fits[5].theta[0], fits[15].theta[0], delta=1e-3)
        self.assertTrue(all(f.converged for f in (laplace, *fits.values())))
        best = fits[15]
        self.assertAlmostEqual(
            glmm_deviance(
                x, y, term, best.theta, best.coefficients, quadrature_points=15
            ),
            best.deviance,
            places=8,
        )
        self.assertEqual(best.parameter_count, 3)
        self.assertAlmostEqual(best.aic, best.deviance + 6.0)
        with self.assertRaises(ValueError):
            fit_generalized_linear_mixed_model(
                x,
                y,
                random_effect(groups, slopes={"x": [r[1] for r in x]}),
                method="adaptive_quadrature",
            )

    def test_poisson_estimates_recover_the_truth(self) -> None:
        """Poisson random intercepts and slopes: estimates within 4 SE."""
        rng = random.Random(31)
        groups, x, y = [], [], []
        for i in range(40):
            b0, b1 = rng.gauss(0.0, 0.5), rng.gauss(0.0, 0.3)
            for j in range(8):
                value = j / 7.0
                groups.append(i)
                x.append(value)
                y.append(
                    float(poisson_draw(rng, math.exp(1.0 + b0 + (0.5 + b1) * value)))
                )
        design = [[1.0, v] for v in x]
        fit = fit_generalized_linear_mixed_model(
            design, y, random_effect(groups, slopes={"x": x}), family="poisson"
        )
        for estimate, error, truth in zip(
            fit.coefficients, fit.standard_errors, (1.0, 0.5), strict=True
        ):
            self.assertLess(abs(estimate - truth), 4 * error)
        self.assertEqual(len(fit.theta), 3)
        self.assertGreater(fit.variance_components[0].standard_deviations[0], 0.2)
        self.assertEqual(len(fit.random_effects[0].values), 40)
        self.assertTrue(fit.converged)


class ValidationTests(unittest.TestCase):
    """Input validation and error messages."""

    def test_invalid_inputs(self) -> None:
        """Bad options, lengths, and unidentifiable terms are rejected."""
        groups, y = one_way(4, 3, 1.0, 1.0, seed=1)
        x = [[1.0]] * len(y)
        term = random_effect(groups)
        with self.assertRaises(ValueError):
            fit_linear_mixed_model(x, y, term, method="MLE")
        with self.assertRaises(ValueError):
            fit_linear_mixed_model(x, y[:-1], term)
        with self.assertRaises(ValueError):
            fit_linear_mixed_model(x, y, random_effect(list(range(len(y)))))
        with self.assertRaises(ValueError):
            fit_linear_mixed_model(x, y, random_effect([0] * len(y)))
        with self.assertRaises(ValueError):
            random_effect(groups, intercept=False)
        with self.assertRaises(ValueError):
            random_effect(groups, slopes={"x": [1.0]})
        with self.assertRaises(TypeError):
            fit_linear_mixed_model(x, y, ["not a term"])  # type: ignore[list-item]
        with self.assertRaises(ValueError):
            fit_generalized_linear_mixed_model(x, [0.0] * len(y), term, family="gamma")
        with self.assertRaises(ValueError):
            glmm_deviance(x, [0.0] * len(y), term, [1.0], [0.0], quadrature_points=0)
        with self.assertRaises(KeyError):
            fit_linear_mixed_model(x, y, term).random_effects[0].for_level("missing")


if __name__ == "__main__":
    unittest.main()
