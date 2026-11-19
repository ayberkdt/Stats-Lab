"""Correlated measurement errors and reverse-mode gradients in calibration."""

from __future__ import annotations

import math
import random
import unittest

from probstat.distributions import LogNormal, Normal, Poisson, Uniform
from probstat.inverse import (
    CALIBRATION_GRADIENTS,
    ERROR_CORRELATION_KERNELS,
    CalibrationProblem,
    ErrorCorrelation,
    Experiment,
    NoiseModel,
    error_correlation_kernel,
    predict,
)
from probstat.linalg import (
    cholesky,
    inverse_spd,
    log_det_spd,
    matmul,
    matvec,
    symmetric_eigen,
    transpose,
)
from probstat.modeling import ClaytonCopula, JointInput, ProbabilisticModel, RandomInput
from probstat.numerics import AutodiffUnsupportedError, admath
from probstat.uq import QuantityOfInterest

DECAY_TIMES = tuple(0.5 * i for i in range(21))
LINE_POINTS = tuple(0.25 * i for i in range(13))
LINE_SIGMA = 0.3
LINE_LENGTH = 0.6


def decay_model(parameters):
    """Return an exponential-decay signal written with ``admath``."""
    return {"y": parameters["A"] * admath.exp(-parameters["k"] * parameters["t"])}


def float_decay_model(parameters):
    """Return the same signal with ``math.exp`` (not traceable)."""
    return {"y": parameters["A"] * math.exp(-parameters["k"] * parameters["t"])}


def line_model(parameters):
    """Return a straight-line response."""
    return {"y": parameters["a"] + parameters["b"] * parameters["x"]}


def decay_prior(correlated: bool = False) -> JointInput:
    """Return the decay prior, optionally with a Gaussian copula."""
    inputs = (
        RandomInput("A", Uniform(1.0, 10.0)),
        RandomInput("k", LogNormal(math.log(0.5), 1.0)),
    )
    if correlated:
        return JointInput(inputs, ((1.0, 0.4), (0.4, 1.0)))
    return JointInput.independent(*inputs)


def decay_problem(
    noise: NoiseModel, *, model=decay_model, prior: JointInput | None = None
) -> CalibrationProblem:
    """Build the seeded exponential-decay calibration problem."""
    probabilistic = ProbabilisticModel(
        decay_prior() if prior is None else prior,
        model,
        (QuantityOfInterest("y", "signal"),),
    )
    generator = random.Random(7)
    experiments = tuple(
        Experiment(
            {"y": 5.0 * math.exp(-0.3 * t) + generator.gauss(0.0, 0.1)}, {"t": t}
        )
        for t in DECAY_TIMES
    )
    return CalibrationProblem(probabilistic, experiments, (noise,))


def kernel_covariance(points, sigma, length, kernel="exponential", nugget=0.0):
    """Return ``sigma^2 M + nugget I`` computed directly from the kernel."""
    return tuple(
        tuple(
            sigma * sigma * error_correlation_kernel(kernel, abs(a - b) / length)
            + (nugget if i == j else 0.0)
            for j, b in enumerate(points)
        )
        for i, a in enumerate(points)
    )


def gaussian_log_density(residuals, covariance):
    """Return the multivariate normal log-density by an explicit inverse."""
    precision = inverse_spd(covariance)
    size = len(residuals)
    quadratic = math.fsum(
        residuals[i] * precision[i][j] * residuals[j]
        for i in range(size)
        for j in range(size)
    )
    return -0.5 * (size * math.log(2.0 * math.pi) + log_det_spd(covariance) + quadratic)


def correlated_line_problem(
    noise: NoiseModel | None = None,
) -> tuple[CalibrationProblem, tuple[float, ...]]:
    """Build a linear problem with exponentially correlated Gaussian errors."""
    prior = JointInput.independent(
        RandomInput("a", Normal(0.0, 2.0)), RandomInput("b", Normal(1.0, 1.0))
    )
    model = ProbabilisticModel(prior, line_model, (QuantityOfInterest("y", "line"),))
    covariance = kernel_covariance(LINE_POINTS, LINE_SIGMA, LINE_LENGTH)
    generator = random.Random(5)
    errors = matvec(
        cholesky(covariance), tuple(generator.gauss(0.0, 1.0) for _ in LINE_POINTS)
    )
    ys = tuple(1.0 + 0.7 * x + e for x, e in zip(LINE_POINTS, errors, strict=True))
    experiments = tuple(
        Experiment({"y": y}, {"x": x}) for x, y in zip(LINE_POINTS, ys, strict=True)
    )
    if noise is None:
        noise = NoiseModel(
            "y",
            standard_deviation=LINE_SIGMA,
            correlation=ErrorCorrelation(condition="x", length=LINE_LENGTH),
        )
    return CalibrationProblem(model, experiments, (noise,)), ys


def exact_line_posterior(ys):
    """Return the conjugate posterior and log evidence by Gaussian conditioning."""
    design = tuple((1.0, x) for x in LINE_POINTS)
    covariance = kernel_covariance(LINE_POINTS, LINE_SIGMA, LINE_LENGTH)
    noise_precision = inverse_spd(covariance)
    prior_mean = (0.0, 1.0)
    prior_precision = ((0.25, 0.0), (0.0, 1.0))
    weighted = matmul(transpose(design), noise_precision)
    information = matmul(weighted, design)
    posterior_precision = tuple(
        tuple(prior_precision[i][j] + information[i][j] for j in range(2))
        for i in range(2)
    )
    posterior_covariance = inverse_spd(posterior_precision)
    right = tuple(
        math.fsum(prior_precision[i][j] * prior_mean[j] for j in range(2))
        + math.fsum(weighted[i][k] * ys[k] for k in range(len(ys)))
        for i in range(2)
    )
    mean = matvec(posterior_covariance, right)
    marginal_mean = matvec(design, prior_mean)
    prior_covariance = ((4.0, 0.0), (0.0, 1.0))
    marginal_covariance = tuple(
        tuple(
            covariance[i][j]
            + math.fsum(
                design[i][a] * prior_covariance[a][b] * design[j][b]
                for a in range(2)
                for b in range(2)
            )
            for j in range(len(ys))
        )
        for i in range(len(ys))
    )
    evidence = gaussian_log_density(
        tuple(y - m for y, m in zip(ys, marginal_mean, strict=True)),
        marginal_covariance,
    )
    return mean, posterior_covariance, evidence, information


def central_gradient(function, point, step=1e-5):
    """Return central differences of a scalar function."""
    gradient = []
    for index in range(len(point)):
        up = tuple(v + (step if k == index else 0.0) for k, v in enumerate(point))
        down = tuple(v - (step if k == index else 0.0) for k, v in enumerate(point))
        gradient.append((function(up) - function(down)) / (2.0 * step))
    return gradient


class ErrorCorrelationDeclarationTests(unittest.TestCase):
    """Validate the declaration and its matrices."""

    def test_kernels_match_closed_forms(self) -> None:
        """Exponential, Matern 3/2 and Matern 5/2 values at h = 0.7."""
        h = 0.7
        self.assertEqual(
            ERROR_CORRELATION_KERNELS, ("exponential", "matern32", "matern52")
        )
        self.assertAlmostEqual(
            error_correlation_kernel("exponential", h), math.exp(-h), places=15
        )
        r3 = math.sqrt(3.0) * h
        self.assertAlmostEqual(
            error_correlation_kernel("matern32", h),
            (1.0 + r3) * math.exp(-r3),
            places=15,
        )
        r5 = math.sqrt(5.0) * h
        self.assertAlmostEqual(
            error_correlation_kernel("matern52", h),
            (1.0 + r5 + r5 * r5 / 3.0) * math.exp(-r5),
            places=15,
        )
        self.assertEqual(error_correlation_kernel("matern52", 0.0), 1.0)
        with self.assertRaises(ValueError):
            error_correlation_kernel("gaussian", h)

    def test_correlation_matrix(self) -> None:
        """A kernel structure builds its matrix; a declared one checks its size."""
        structure = ErrorCorrelation(condition="t", kernel="matern32", length=2.0)
        matrix = structure.correlation_matrix((0.0, 1.0, 3.0))
        self.assertEqual(matrix[0][0], 1.0)
        self.assertAlmostEqual(
            matrix[0][2], error_correlation_kernel("matern32", 1.5), places=15
        )
        self.assertEqual(matrix[1][2], matrix[2][1])
        self.assertTrue(structure.is_known)
        declared = ErrorCorrelation(matrix=((2.0, 0.5), (0.5, 1.0)))
        self.assertEqual(declared.correlation_matrix((0.0, 0.0))[0][0], 2.0)
        with self.assertRaises(ValueError):
            declared.correlation_matrix((0.0, 0.0, 0.0))
        uncertain = ErrorCorrelation(condition="t", length_prior=LogNormal(0.0, 1.0))
        self.assertFalse(uncertain.is_known)
        with self.assertRaises(ValueError):
            uncertain.correlation_matrix((0.0, 1.0))
        self.assertEqual(
            uncertain.correlation_matrix((0.0, 1.0), 1.0)[0][1], math.exp(-1)
        )

    def test_invalid_declarations(self) -> None:
        """Every invalid combination raises a clear error."""
        cases = (
            {},
            {"matrix": ((1.0,),), "condition": "t"},
            {"matrix": ((1.0, 0.2), (0.3, 1.0))},
            {"matrix": ((1.0, 2.0), (2.0, 1.0))},
            {"matrix": ((1.0, 0.0),)},
            {"matrix": ((1.0,),), "length": 1.0},
            {"condition": "t"},
            {"condition": "t", "length": 1.0, "length_prior": LogNormal(0.0, 1.0)},
            {"condition": "t", "length": -1.0},
            {"condition": "t", "length": 1.0, "kernel": "rbf"},
            {"condition": " ", "length": 1.0},
        )
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                ErrorCorrelation(**case)
        structure = ErrorCorrelation(condition="t", length=1.0)
        with self.assertRaises(ValueError):
            NoiseModel(
                "y",
                standard_deviation=1.0,
                degrees_of_freedom=4.0,
                correlation=structure,
            )
        with self.assertRaises(TypeError):
            NoiseModel("y", standard_deviation=1.0, correlation="ar1")

    def test_problem_level_validation(self) -> None:
        """Size, missing or repeated conditions, and non-PD matrices are refused."""
        identity3 = tuple(tuple(float(i == j) for j in range(3)) for i in range(3))
        noises = (
            NoiseModel(
                "y",
                standard_deviation=0.1,
                correlation=ErrorCorrelation(matrix=identity3),
            ),
            NoiseModel(
                "y",
                standard_deviation=0.1,
                correlation=ErrorCorrelation(condition="missing", length=1.0),
            ),
            NoiseModel(
                "y",
                standard_deviation=0.1,
                correlation=ErrorCorrelation(condition="A", length=1.0),
            ),
            NoiseModel(
                "y",
                standard_deviation=0.1,
                correlation=ErrorCorrelation(
                    condition="t", kernel="matern52", length=1.0e5
                ),
            ),
        )
        for noise in noises:
            with self.subTest(noise=noise), self.assertRaises(ValueError):
                decay_problem(noise)
        prior = decay_prior()
        model = ProbabilisticModel(prior, decay_model, (QuantityOfInterest("y", "s"),))
        repeated = tuple(Experiment({"y": 1.0}, {"t": 1.0}) for _ in range(3))
        structure = ErrorCorrelation(condition="t", length=1.0)
        with self.assertRaises(ValueError):
            CalibrationProblem(
                model,
                repeated,
                (NoiseModel("y", standard_deviation=0.1, correlation=structure),),
            )
        nugget = CalibrationProblem(
            model,
            repeated,
            (
                NoiseModel(
                    "y",
                    standard_deviation=0.1,
                    discrepancy_variance=0.01,
                    correlation=structure,
                ),
            ),
        )
        self.assertTrue(math.isfinite(nugget.log_likelihood((5.0, 0.3))))


class CorrelatedLikelihoodTests(unittest.TestCase):
    """Check the likelihood against independent computations."""

    def test_independent_problem_is_unchanged(self) -> None:
        """Without correlation the likelihood is the original independent sum."""
        noise = NoiseModel("y", standard_deviation=0.1)
        self.assertIsNone(noise.correlation)
        self.assertEqual(noise, NoiseModel("y", standard_deviation=0.1))
        problem = decay_problem(noise)
        self.assertFalse(problem.has_correlated_errors)
        for theta in ((5.0, 0.3), (4.2, 0.45), (7.5, 0.1)):
            residuals = problem.residuals(theta)
            expected = math.fsum(noise.log_density(r, 0.1) for r in residuals)
            self.assertEqual(problem.log_likelihood(theta), expected)
        unknown = decay_problem(
            NoiseModel("y", standard_deviation_prior=LogNormal(math.log(0.2), 1.0))
        )
        self.assertEqual(unknown.vector_names, ("A", "k", "log_sigma[y]"))
        self.assertEqual(unknown.noise_names, ("sigma[y]",))
        self.assertEqual(unknown.dimension, 3)
        coordinates = (0.3, -0.2, -2.1)
        theta = unknown.model.inputs.from_standard_normal(coordinates[:2])
        sigma = math.exp(coordinates[2])
        item = unknown.noise[0]
        expected = (
            math.fsum(item.log_density(r, sigma) for r in unknown.residuals(theta))
            - 0.5 * math.fsum(u * u for u in coordinates[:2])
            + (LogNormal(math.log(0.2), 1.0).log_pdf(sigma) + coordinates[2])
        )
        self.assertEqual(unknown.standard_log_posterior(coordinates), expected)

    def test_identity_correlation_reproduces_independent_likelihood(self) -> None:
        """M = I (declared, or a kernel whose correlations underflow) is independent."""
        size = len(DECAY_TIMES)
        identity = tuple(tuple(float(i == j) for j in range(size)) for i in range(size))
        structures = (
            ErrorCorrelation(matrix=identity),
            ErrorCorrelation(condition="t", length=1.0e-4),
        )
        for structure in structures:
            for known in (True, False):
                with self.subTest(structure=structure.condition, known=known):
                    if known:
                        plain = NoiseModel("y", standard_deviation=0.1)
                        tied = NoiseModel(
                            "y", standard_deviation=0.1, correlation=structure
                        )
                        points = ((5.0, 0.3), (4.0, 0.2))
                    else:
                        law = LogNormal(math.log(0.2), 1.0)
                        plain = NoiseModel("y", standard_deviation_prior=law)
                        tied = NoiseModel(
                            "y", standard_deviation_prior=law, correlation=structure
                        )
                        points = ((5.0, 0.3, -2.0), (4.0, 0.2, -1.0))
                    independent = decay_problem(plain)
                    correlated = decay_problem(tied)
                    self.assertTrue(correlated.has_correlated_errors)
                    for point in points:
                        self.assertAlmostEqual(
                            correlated.log_likelihood(point),
                            independent.log_likelihood(point),
                            delta=1e-12 * abs(independent.log_likelihood(point)),
                        )
                        self.assertAlmostEqual(
                            correlated.log_posterior(point),
                            independent.log_posterior(point),
                            delta=1e-12 * abs(independent.log_posterior(point)),
                        )
                    fit_a = independent.fit_least_squares()
                    fit_b = correlated.fit_least_squares()
                    for a, b in zip(fit_a.estimate, fit_b.estimate, strict=True):
                        self.assertAlmostEqual(a, b, delta=1e-9 * abs(a))

    def test_kernel_likelihood_matches_dense_formula(self) -> None:
        """Every kernel, with and without nugget, matches the explicit inverse."""
        theta = (5.1, 0.29)
        for kernel in ERROR_CORRELATION_KERNELS:
            for nugget in (0.0, 0.002):
                with self.subTest(kernel=kernel, nugget=nugget):
                    structure = ErrorCorrelation(
                        condition="t", kernel=kernel, length=1.3
                    )
                    problem = decay_problem(
                        NoiseModel(
                            "y",
                            standard_deviation=0.1,
                            discrepancy_variance=nugget,
                            correlation=structure,
                        )
                    )
                    expected = gaussian_log_density(
                        problem.residuals(theta),
                        kernel_covariance(DECAY_TIMES, 0.1, 1.3, kernel, nugget),
                    )
                    value = problem.log_likelihood(theta)
                    self.assertAlmostEqual(value, expected, delta=1e-9 * abs(expected))
                    whitened = problem.whitened_residuals(theta)
                    precision = inverse_spd(
                        kernel_covariance(DECAY_TIMES, 0.1, 1.3, kernel, nugget)
                    )
                    residuals = problem.residuals(theta)
                    quadratic = math.fsum(
                        residuals[i] * precision[i][j] * residuals[j]
                        for i in range(len(residuals))
                        for j in range(len(residuals))
                    )
                    self.assertAlmostEqual(
                        math.fsum(w * w for w in whitened),
                        quadratic,
                        delta=1e-8 * quadratic,
                    )

    def test_uncertain_length_layout_and_likelihood(self) -> None:
        """A length prior adds ``log_length[q]`` and is used by the likelihood."""
        problem = decay_problem(
            NoiseModel(
                "y",
                standard_deviation_prior=LogNormal(math.log(0.2), 1.0),
                correlation=ErrorCorrelation(
                    condition="t", kernel="matern32", length_prior=LogNormal(0.0, 0.5)
                ),
            )
        )
        self.assertEqual(
            problem.vector_names, ("A", "k", "log_sigma[y]", "log_length[y]")
        )
        self.assertEqual(problem.noise_names, ("sigma[y]", "length[y]"))
        point = (5.0, 0.3, math.log(0.15), math.log(0.8))
        self.assertAlmostEqual(problem.correlation_lengths(point)["y"], 0.8, places=14)
        expected = gaussian_log_density(
            problem.residuals((5.0, 0.3)),
            kernel_covariance(DECAY_TIMES, 0.15, 0.8, "matern32"),
        )
        self.assertAlmostEqual(
            problem.log_likelihood(point), expected, delta=1e-9 * abs(expected)
        )
        prior = (
            decay_prior().log_pdf((5.0, 0.3))
            + LogNormal(math.log(0.2), 1.0).log_pdf(0.15)
            + math.log(0.15)
            + LogNormal(0.0, 0.5).log_pdf(0.8)
            + math.log(0.8)
        )
        self.assertAlmostEqual(problem.log_prior(point), prior, places=12)
        with self.assertRaises(ValueError):
            problem.fit_least_squares()


class LinearCorrelatedPosteriorTests(unittest.TestCase):
    """Exact conjugate answers for a linear model with correlated noise."""

    @classmethod
    def setUpClass(cls) -> None:
        """Build the problem and its exact posterior once."""
        cls.problem, cls.ys = correlated_line_problem()
        cls.mean, cls.covariance, cls.evidence, cls.information = exact_line_posterior(
            cls.ys
        )

    def test_laplace_is_exact(self) -> None:
        """Laplace mean, covariance, and normalizer equal the conjugate result."""
        for gradient in CALIBRATION_GRADIENTS:
            with self.subTest(gradient=gradient):
                laplace = self.problem.laplace_posterior(gradient=gradient)
                for i in range(2):
                    self.assertAlmostEqual(laplace.mode[i], self.mean[i], places=6)
                    for j in range(2):
                        self.assertAlmostEqual(
                            laplace.covariance[i][j], self.covariance[i][j], places=6
                        )
                self.assertAlmostEqual(laplace.log_normalizer, self.evidence, places=5)
        reverse = self.problem.laplace_posterior(gradient="reverse")
        self.assertEqual(reverse.gradient_method, "reverse")
        self.assertEqual(reverse.hessian_method, "reverse")

    def test_generalized_least_squares(self) -> None:
        """LM on whitened residuals is the GLS estimate with its covariance."""
        fit = self.problem.fit_least_squares()
        design = tuple((1.0, x) for x in LINE_POINTS)
        precision = inverse_spd(kernel_covariance(LINE_POINTS, LINE_SIGMA, LINE_LENGTH))
        gls_covariance = inverse_spd(self.information)
        weighted = matmul(transpose(design), precision)
        estimate = matvec(
            gls_covariance,
            tuple(
                math.fsum(weighted[i][k] * self.ys[k] for k in range(len(self.ys)))
                for i in range(2)
            ),
        )
        for i in range(2):
            self.assertAlmostEqual(fit.estimate[i], estimate[i], places=8)
            for j in range(2):
                self.assertAlmostEqual(
                    fit.covariance[i][j], gls_covariance[i][j], places=8
                )
        identifiability = self.problem.identifiability(fit.estimate, prior_scaled=False)
        values, _ = symmetric_eigen(self.information)
        for a, b in zip(identifiability.eigenvalues, values, strict=True):
            self.assertAlmostEqual(a, b, delta=1e-6 * b)

    def test_unknown_scale_of_declared_matrix(self) -> None:
        """An unknown scale is re-estimated from the generalized RSS."""
        shape = kernel_covariance(LINE_POINTS, 1.0, LINE_LENGTH)
        problem, ys = correlated_line_problem(
            NoiseModel(
                "y",
                standard_deviation_prior=LogNormal(math.log(0.3), 1.0),
                correlation=ErrorCorrelation(matrix=shape),
            )
        )
        fit = problem.fit_least_squares()
        lower = cholesky(shape)
        residuals = problem.residuals(fit.estimate)
        from probstat.linalg import solve_lower

        whitened = solve_lower(lower, residuals)
        n = len(ys)
        expected = math.sqrt(math.fsum(w * w for w in whitened) / (n - 2))
        self.assertAlmostEqual(
            fit.noise_standard_deviations["y"], expected, delta=1e-7 * expected
        )
        nugget, _ = correlated_line_problem(
            NoiseModel(
                "y",
                standard_deviation_prior=LogNormal(math.log(0.3), 1.0),
                discrepancy_variance=0.01,
                correlation=ErrorCorrelation(matrix=shape),
            )
        )
        with self.assertRaises(ValueError):
            nugget.fit_least_squares()

    def test_mcmc_agrees_with_conjugate_posterior(self) -> None:
        """Adaptive Metropolis and NUTS with reverse gradients, 4 MCSE."""
        runs = (
            self.problem.sample_posterior(draws=1000, warmup=500, chains=4, seed=11),
            self.problem.sample_posterior(
                draws=400,
                warmup=300,
                chains=2,
                seed=11,
                method="nuts",
                gradient="reverse",
            ),
        )
        for run in runs:
            for index, summary in enumerate(run.summary()):
                with self.subTest(run=runs.index(run), parameter=summary.name):
                    self.assertLess(
                        abs(summary.mean - self.mean[index]), 4.0 * summary.mcse_mean
                    )
                    sd = math.sqrt(self.covariance[index][index])
                    self.assertAlmostEqual(
                        summary.standard_deviation, sd, delta=0.2 * sd
                    )
                    self.assertLess(summary.rank_normalized_rhat, 1.05)

    def test_smc_evidence_matches_exact_evidence(self) -> None:
        """Tempered SMC on the correlated likelihood recovers ``log p(y)``."""
        result = self.problem.log_evidence(particles=400, mutation_steps=5, seed=3)
        error = result.log_evidence_standard_error
        self.assertLess(abs(result.log_evidence - self.evidence), 4.0 * error + 0.1)


class CorrelationLengthPosteriorTests(unittest.TestCase):
    """An uncertain correlation length is recovered from simulated data."""

    def test_length_recovered_within_sampling_error(self) -> None:
        """MCMC matches the exact length posterior, which covers the truth.

        With a Gaussian prior on ``(a, b)`` the marginal likelihood of the
        length is ``N(y; H m0, H P0 H' + Sigma(ell))``, so the posterior of
        ``ell`` follows by one-dimensional quadrature on a log grid.
        """
        truth = 1.0
        points = tuple(0.3 * i for i in range(30))
        prior = JointInput.independent(
            RandomInput("a", Normal(0.0, 2.0)), RandomInput("b", Normal(1.0, 1.0))
        )
        model = ProbabilisticModel(prior, line_model, (QuantityOfInterest("y", "l"),))
        covariance = kernel_covariance(points, 0.2, truth)
        generator = random.Random(2026)
        errors = matvec(
            cholesky(covariance), tuple(generator.gauss(0.0, 1.0) for _ in points)
        )
        experiments = tuple(
            Experiment({"y": 1.0 + 0.7 * x + e}, {"x": x})
            for x, e in zip(points, errors, strict=True)
        )
        problem = CalibrationProblem(
            model,
            experiments,
            (
                NoiseModel(
                    "y",
                    standard_deviation=0.2,
                    correlation=ErrorCorrelation(
                        condition="x", length_prior=LogNormal(0.0, 0.75)
                    ),
                ),
            ),
        )
        ys = tuple(experiment.observations["y"] for experiment in experiments)
        prior_length = LogNormal(0.0, 0.75)
        grid = tuple(math.exp(-3.0 + 6.0 * k / 240) for k in range(241))
        logs = []
        for length in grid:
            covariance_y = tuple(
                tuple(
                    0.04 * error_correlation_kernel("exponential", abs(a - b) / length)
                    + 4.0
                    + a * b
                    for b in points
                )
                for a in points
            )
            residual = tuple(y - x for x, y in zip(points, ys, strict=True))
            logs.append(
                gaussian_log_density(residual, covariance_y)
                + prior_length.log_pdf(length)
                + math.log(length)
            )
        top = max(logs)
        weights = tuple(math.exp(value - top) for value in logs)
        total = math.fsum(weights)
        exact_mean = math.fsum(w * ell for w, ell in zip(weights, grid, strict=True))
        exact_mean /= total
        exact_sd = math.sqrt(
            math.fsum(
                w * (ell - exact_mean) ** 2
                for w, ell in zip(weights, grid, strict=True)
            )
            / total
        )
        posterior = problem.sample_posterior(draws=500, warmup=400, chains=4, seed=17)
        summary = {s.name: s for s in posterior.summary()}["length[y]"]
        self.assertLess(abs(summary.mean - exact_mean), 4.0 * summary.mcse_mean)
        self.assertAlmostEqual(
            summary.standard_deviation, exact_sd, delta=0.2 * exact_sd
        )
        self.assertLess(abs(exact_mean - truth), 3.0 * exact_sd)
        self.assertLess(summary.rank_normalized_rhat, 1.1)
        self.assertEqual(posterior.noise_names, ("length[y]",))

    def test_fully_standardized_methods_carry_the_length(self) -> None:
        """The pCN and SMC samplers carry the length through its quantile map."""
        problem = decay_problem(
            NoiseModel(
                "y",
                standard_deviation_prior=LogNormal(math.log(0.2), 1.0),
                correlation=ErrorCorrelation(
                    condition="t", length_prior=LogNormal(math.log(0.5), 0.5)
                ),
            )
        )
        pcn = problem.sample_posterior(
            draws=40, warmup=40, chains=2, seed=1, method="pcn"
        )
        self.assertEqual(
            pcn.standard_space.names, ("u[A]", "u[k]", "log_sigma[y]", "log_length[y]")
        )
        self.assertEqual(pcn.mcmc.names, ("A", "k", "sigma[y]", "length[y]"))
        smc = problem.sample_posterior(
            seed=1, method="smc", particles=60, mutation_steps=2
        )
        assert smc.smc is not None
        self.assertEqual(smc.smc.names[-1], "w[length[y]]")
        self.assertTrue(math.isfinite(smc.smc.log_evidence))
        self.assertTrue(all(value > 0.0 for value in smc.mcmc.pooled()[0][2:]))


class CorrelatedPredictionTests(unittest.TestCase):
    """Noise draws for correlated quantities in ``predict``."""

    def test_kernel_noise_is_drawn_jointly(self) -> None:
        """Errors at two conditions have the kernel correlation."""
        problem = decay_problem(
            NoiseModel(
                "y",
                standard_deviation=0.5,
                correlation=ErrorCorrelation(condition="t", length=1.0),
            )
        )
        draws = [{"A": 5.0, "k": 0.3}] * 4000
        prediction = predict(
            problem,
            draws,
            conditions=({"t": 0.0}, {"t": 0.5}),
            include_noise=True,
            seed=12,
        )
        first, second = prediction.get("y", 0), prediction.get("y", 1)
        assert first.observed is not None and second.observed is not None
        e1 = [o - c for o, c in zip(first.observed, first.latent, strict=True)]
        e2 = [o - c for o, c in zip(second.observed, second.latent, strict=True)]
        count = len(e1)
        m1, m2 = sum(e1) / count, sum(e2) / count
        c12 = sum((a - m1) * (b - m2) for a, b in zip(e1, e2, strict=True)) / count
        v1 = sum((a - m1) ** 2 for a in e1) / count
        v2 = sum((b - m2) ** 2 for b in e2) / count
        rho = math.exp(-0.5)
        self.assertLess(abs(c12 / math.sqrt(v1 * v2) - rho), 4.0 * (1 - rho**2) / 63.0)
        self.assertAlmostEqual(
            math.sqrt(v1), 0.5, delta=4.0 * 0.5 / math.sqrt(2 * count)
        )

    def test_matrix_correlation_noise_is_undefined(self) -> None:
        """A declared matrix raises unless an independent scale is given."""
        size = len(DECAY_TIMES)
        identity = tuple(tuple(float(i == j) for j in range(size)) for i in range(size))
        problem = decay_problem(
            NoiseModel(
                "y",
                standard_deviation=0.1,
                correlation=ErrorCorrelation(matrix=identity),
            )
        )
        draws = [{"A": 5.0, "k": 0.3}] * 10
        with self.assertRaises(ValueError):
            predict(problem, draws, conditions=({"t": 1.0},), include_noise=True)
        latent = predict(problem, draws, conditions=({"t": 1.0},))
        self.assertIsNone(latent.get("y").observed)
        override = predict(
            problem,
            draws,
            conditions=({"t": 1.0},),
            include_noise=True,
            noise_standard_deviations={"y": 0.1},
            seed=1,
        )
        self.assertIsNotNone(override.get("y").observed)

    def test_kernel_noise_needs_the_condition(self) -> None:
        """A prediction condition without the kernel variable raises."""
        problem = decay_problem(
            NoiseModel(
                "y",
                standard_deviation=0.1,
                correlation=ErrorCorrelation(
                    condition="t", length_prior=LogNormal(0.0, 1.0)
                ),
            )
        )
        with self.assertRaises(ValueError):
            predict(
                problem,
                [{"A": 5.0, "k": 0.3, "length[y]": 1.0}],
                conditions=({},),
                include_noise=True,
            )
        with self.assertRaises(ValueError):
            predict(
                problem,
                [{"A": 5.0, "k": 0.3}],
                conditions=({"t": 1.0},),
                include_noise=True,
            )
        with self.assertRaises(ValueError):
            predict(
                problem,
                [{"A": 5.0, "k": 0.3, "length[y]": 1.0}],
                conditions=({"t": 1.0}, {"t": 1.0}),
                include_noise=True,
            )


class ReverseGradientTests(unittest.TestCase):
    """Reverse-mode gradients of the calibration log-densities."""

    def assert_close(self, actual, expected, tolerance=1e-6) -> None:
        """Assert relative agreement component by component."""
        for a, b in zip(actual, expected, strict=True):
            self.assertAlmostEqual(a, b, delta=tolerance * max(1.0, abs(b)))

    def test_reverse_equals_central_differences(self) -> None:
        """Standard and physical gradients match central differences to 1e-6."""
        cases = (
            (NoiseModel("y", standard_deviation=0.1), False, (0.2, -0.1)),
            (
                NoiseModel("y", standard_deviation_prior=LogNormal(math.log(0.2), 1.0)),
                False,
                (0.1, 0.3, -2.0),
            ),
            (
                NoiseModel(
                    "y",
                    standard_deviation_prior=LogNormal(math.log(0.2), 1.0),
                    degrees_of_freedom=4.0,
                ),
                False,
                (0.1, 0.3, -2.0),
            ),
            (
                NoiseModel(
                    "y",
                    standard_deviation_prior=LogNormal(math.log(0.2), 1.0),
                    discrepancy_variance=1e-3,
                    correlation=ErrorCorrelation(
                        condition="t",
                        kernel="matern52",
                        length_prior=LogNormal(0.0, 0.5),
                    ),
                ),
                False,
                (0.2, -0.1, -2.0, 0.1),
            ),
            (NoiseModel("y", standard_deviation=0.1), True, (0.2, -0.1)),
        )
        for noise, copula, point in cases:
            with self.subTest(noise=noise, copula=copula):
                problem = decay_problem(noise, prior=decay_prior(copula))
                value, gradient = problem.standard_log_posterior_and_gradient(point)
                self.assertAlmostEqual(
                    value,
                    problem.standard_log_posterior(point),
                    delta=1e-12 * abs(value),
                )
                self.assert_close(
                    gradient, central_gradient(problem.standard_log_posterior, point)
                )
                physical = problem.to_physical(point)
                parameters = physical[:2] + tuple(math.log(v) for v in physical[2:])
                value, gradient = problem.log_posterior_and_gradient(parameters)
                self.assertAlmostEqual(
                    value, problem.log_posterior(parameters), delta=1e-9 * abs(value)
                )
                self.assert_close(
                    gradient, central_gradient(problem.log_posterior, parameters)
                )

    def test_hmc_reverse_agrees_with_central_and_costs_less(self) -> None:
        """Seeded NUTS runs agree within 4 MCSE; reverse uses fewer model runs."""
        problem = decay_problem(NoiseModel("y", standard_deviation=0.1))
        runs = {}
        counts = {}
        for gradient in CALIBRATION_GRADIENTS:
            start = problem.model.evaluations
            runs[gradient] = problem.sample_posterior(
                draws=250,
                warmup=200,
                chains=2,
                seed=21,
                method="nuts",
                gradient=gradient,
            )
            counts[gradient] = problem.model.evaluations - start
        central = {s.name: s for s in runs["central"].summary()}
        reverse = {s.name: s for s in runs["reverse"].summary()}
        for name in ("A", "k"):
            with self.subTest(parameter=name):
                bound = 4.0 * math.hypot(
                    central[name].mcse_mean, reverse[name].mcse_mean
                )
                self.assertLess(abs(central[name].mean - reverse[name].mean), bound)
        assert runs["reverse"].sampler is not None
        self.assertEqual(runs["reverse"].sampler.gradient_method, "reverse")
        self.assertLess(3 * counts["reverse"], counts["central"])

    def test_untraceable_problems_raise(self) -> None:
        """math.exp models, Rosenblatt or discrete priors raise; no fallback."""
        float_problem = decay_problem(
            NoiseModel("y", standard_deviation=0.1), model=float_decay_model
        )
        with self.assertRaises(AutodiffUnsupportedError):
            float_problem.laplace_posterior(gradient="reverse")
        with self.assertRaises(AutodiffUnsupportedError):
            float_problem.sample_posterior(
                draws=5, warmup=5, chains=1, seed=1, method="nuts", gradient="reverse"
            )
        with self.assertRaises(AutodiffUnsupportedError):
            float_problem.standard_log_posterior_and_gradient((0.0, 0.0))
        clayton = JointInput(
            (
                RandomInput("A", Uniform(1.0, 10.0)),
                RandomInput("k", LogNormal(math.log(0.5), 1.0)),
            ),
            copula=ClaytonCopula(2.0, 2),
        )
        rosenblatt = decay_problem(
            NoiseModel("y", standard_deviation=0.1), prior=clayton
        )
        with self.assertRaises(AutodiffUnsupportedError):
            rosenblatt.standard_log_posterior_and_gradient((0.0, 0.0))
        discrete_prior = JointInput.independent(
            RandomInput("A", Poisson(5.0)), RandomInput("k", LogNormal(-1.0, 0.5))
        )
        discrete = decay_problem(
            NoiseModel("y", standard_deviation=0.1), prior=discrete_prior
        )
        with self.assertRaises(AutodiffUnsupportedError):
            discrete.log_posterior_and_gradient((5.0, 0.3))

    def test_gradient_option_validation(self) -> None:
        """Unknown options and reverse with other samplers raise ValueError."""
        problem = decay_problem(NoiseModel("y", standard_deviation=0.1))
        self.assertEqual(CALIBRATION_GRADIENTS, ("central", "reverse"))
        with self.assertRaises(ValueError):
            problem.laplace_posterior(gradient="forward")
        for method in ("adaptive-metropolis", "pcn", "smc"):
            with self.subTest(method=method), self.assertRaises(ValueError):
                problem.sample_posterior(method=method, gradient="reverse")
        with self.assertRaises(ValueError):
            problem.sample_posterior(method="hmc", gradient="automatic")
        static = problem.sample_posterior(
            draws=20, warmup=20, chains=1, seed=2, method="hmc", gradient="reverse"
        )
        assert static.sampler is not None
        self.assertEqual(static.sampler.gradient_method, "reverse")
        self.assertEqual(static.mcmc.names, ("A", "k"))

    def test_copula_laplace_hessian_falls_back_to_differences(self) -> None:
        """A Gaussian-copula prior gets reverse gradients, FD Hessian, same mode."""
        problem = decay_problem(
            NoiseModel("y", standard_deviation=0.1), prior=decay_prior(True)
        )
        central = problem.laplace_posterior()
        reverse = problem.laplace_posterior(gradient="reverse")
        self.assertEqual(reverse.gradient_method, "reverse")
        self.assertEqual(reverse.hessian_method, "finite-difference")
        self.assertIsNotNone(reverse.fallback_reason)
        for a, b in zip(central.mode, reverse.mode, strict=True):
            self.assertAlmostEqual(a, b, delta=1e-5 * abs(b))


if __name__ == "__main__":
    unittest.main()
