"""Bayesian experimental design by expected information gain.

The expected information gain (EIG) of a design ``D`` (a list of experimental
conditions) is the mutual information between parameters and data,

``EIG(D) = E_{theta, y}[log p(y | theta, D) - log p(y | D)]``

(EQ-OED-EIG-001), the expected Kullback-Leibler divergence from prior to
posterior. It is invariant under bijective reparameterization of ``theta``.

A :class:`BayesianDesignProblem` reuses the declarations of
:class:`probstat.inverse.CalibrationProblem`: a
:class:`~probstat.modeling.ProbabilisticModel` whose inputs are the prior, and
one known-scale Gaussian :class:`~probstat.inverse.NoiseModel` per observed
quantity. A candidate condition is a mapping of deterministic model inputs,
exactly like :attr:`probstat.inverse.Experiment.conditions`.

Estimators:

* :func:`linear_gaussian_information_gain` -- closed form
  ``1/2 log det(I + Sigma J' R^-1 J)`` (EQ-OED-EIG-LIN-001).
* :func:`linearized_information_gain` -- the closed form for the model
  linearized at one point of the prior's standard-normal space; exact for
  linear models with Gaussian priors.
* :func:`laplace_information_gain` -- prior average of the Gauss-Newton
  Laplace approximation (EQ-OED-LAP-001).
* :func:`nested_monte_carlo_information_gain` -- nested Monte Carlo
  (EQ-OED-NMC-001), with fresh inner samples (Ryan, biased upward) or with the
  outer samples reused as the inner sample (Huan-Marzouk, biased downward).

:func:`greedy_design` adds conditions one at a time, each maximizing the EIG
of the enlarged design (EQ-OED-GREEDY-001).
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)
from probstat.inverse import CalibrationProblem, NoiseModel
from probstat.linalg import (
    Matrix,
    Vector,
    as_matrix,
    as_vector,
    cholesky,
    log_det_spd,
    matmul,
    solve_lower,
    transpose,
)
from probstat.modeling import ProbabilisticModel
from probstat.optimization import central_jacobian
from probstat.uq import evaluate_model

Condition = Mapping[str, float]
"""Deterministic model inputs of one experiment."""

_METHODS = ("linearized", "laplace", "nested")


def _log_mean_exp(values: Sequence[float]) -> float:
    """Return ``log((1/M) sum exp(v_m))`` without overflow.

    References:
        [REF-ROBERT-CASELLA-2004, Chapter 3]. No new equation is introduced.

    """
    top = max(values)
    if top == -math.inf:
        return -math.inf
    return top + math.log(math.fsum(math.exp(v - top) for v in values) / len(values))


def _half_log_det_identity_plus(gram: Matrix) -> float:
    """Return ``1/2 log det(I + G)`` for a symmetric PSD matrix ``G``.

    References:
        [EQ-OED-EIG-LIN-001], [EQ-LINALG-006].

    """
    size = len(gram)
    shifted = tuple(
        tuple(gram[i][j] + (1.0 if i == j else 0.0) for j in range(size))
        for i in range(size)
    )
    return 0.5 * log_det_spd(shifted)


def linear_gaussian_information_gain(
    jacobian: Sequence[Sequence[float]],
    prior_covariance: Sequence[Sequence[float]],
    *,
    noise_variances: Sequence[float] | None = None,
    noise_covariance: Sequence[Sequence[float]] | None = None,
) -> float:
    """Return the exact EIG of a linear-Gaussian experiment.

    For ``y = J theta + e`` with ``theta ~ N(m, Sigma)`` and ``e ~ N(0, R)``,
    ``EIG = 1/2 log det(I + Sigma J' R^-1 J)``. It is computed as
    ``1/2 log det(I + B' B)`` with ``B = L_R^-1 J L_Sigma`` and Cholesky
    factors ``L``, which is symmetric positive definite. It does not depend
    on ``m`` or on the observed data.

    Args:
        jacobian: ``m`` by ``d`` observation matrix ``J``.
        prior_covariance: ``d`` by ``d`` SPD prior covariance.
        noise_variances: Positive diagonal of ``R`` (independent noise).
        noise_covariance: Full SPD ``R``; give exactly one of the two.

    Returns:
        EIG in nats.

    Raises:
        ValueError: If shapes disagree, a covariance is not positive
            definite, or not exactly one noise argument is given.

    Equation:
        [EQ-OED-EIG-LIN-001].

    References:
        [REF-CHALONER-VERDINELLI-1995], [REF-COVER-THOMAS-2006, Chapter 8].

    """
    rows = as_matrix(jacobian, name="jacobian")
    sigma = as_matrix(prior_covariance, name="prior_covariance")
    d = len(sigma)
    if any(len(row) != d for row in sigma) or any(len(row) != d for row in rows):
        raise ValueError("jacobian columns and prior_covariance must match")
    if (noise_variances is None) == (noise_covariance is None):
        raise ValueError("give exactly one of noise_variances or noise_covariance")
    lower_prior = cholesky(sigma)
    mapped = matmul(rows, lower_prior)
    m = len(rows)
    if noise_variances is not None:
        variances = as_vector(noise_variances, name="noise_variances")
        if len(variances) != m or any(v <= 0.0 for v in variances):
            raise ValueError("noise_variances must be positive, one per row")
        whitened = tuple(
            tuple(value / math.sqrt(v) for value in row)
            for row, v in zip(mapped, variances, strict=True)
        )
    else:
        assert noise_covariance is not None
        noise = as_matrix(noise_covariance, name="noise_covariance")
        if len(noise) != m or any(len(row) != m for row in noise):
            raise ValueError("noise_covariance must be m by m")
        lower_noise = cholesky(noise)
        columns = tuple(
            solve_lower(lower_noise, tuple(row[j] for row in mapped)) for j in range(d)
        )
        whitened = transpose(columns)
    return _half_log_det_identity_plus(matmul(transpose(whitened), whitened))


# ---------------------------------------------------------------------------
# Problem declaration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BayesianDesignProblem:
    """Prior, forward model, and known Gaussian noise for design studies.

    The observation model for a design ``D = (c_1, ..., c_n)`` is
    ``y_jq = g_q(theta; c_j) + e_jq`` with independent
    ``e_jq ~ N(0, s_q^2)``, ``s_q^2 = sigma_q^2 + sigma_delta_q^2``
    (the noise and discrepancy variances of the :class:`NoiseModel`).
    Observations are ordered condition-major, then in ``noise`` order.

    Args:
        model: Probabilistic model; its inputs are the prior on ``theta``.
        noise: One known-scale Gaussian noise model per observed quantity.

    Raises:
        TypeError: If components have the wrong types.
        ValueError: If a noise model is unknown-scale or Student-t, names an
            undeclared quantity, or is repeated.

    References:
        [REF-CHALONER-VERDINELLI-1995], [REF-HUAN-MARZOUK-2013],
        [REF-KENNEDY-OHAGAN-2001].

    """

    model: ProbabilisticModel
    noise: tuple[NoiseModel, ...]

    def __post_init__(self) -> None:
        """Validate the declarations.

        References:
            [EQ-OED-EIG-001]. No equation is evaluated.

        """
        if not isinstance(self.model, ProbabilisticModel):
            raise TypeError("model must be a ProbabilisticModel")
        noise = tuple(self.noise)
        if not noise:
            raise ValueError("at least one noise model is required")
        seen: set[str] = set()
        declared = set(self.model.quantity_names)
        for index, item in enumerate(noise):
            if not isinstance(item, NoiseModel):
                raise TypeError(f"noise[{index}] must be a NoiseModel")
            if not item.is_known:
                raise ValueError("design needs known noise scales (standard_deviation)")
            if item.degrees_of_freedom is not None:
                raise ValueError("design supports Gaussian noise only")
            if item.quantity not in declared:
                raise ValueError(f"{item.quantity!r} is not a model quantity")
            if item.quantity in seen:
                raise ValueError(f"duplicate noise model for {item.quantity!r}")
            seen.add(item.quantity)
        object.__setattr__(self, "noise", noise)

    @classmethod
    def from_calibration(cls, problem: CalibrationProblem) -> BayesianDesignProblem:
        """Reuse the prior, model, and noise of a calibration problem.

        The recorded experiments are not used; pass their conditions as a
        ``baseline`` to :func:`greedy_design` to plan *additional* runs.

        Args:
            problem: Calibration problem with known Gaussian noise.

        Returns:
            Design problem sharing the model (and its evaluation counter).

        Raises:
            TypeError: If ``problem`` is not a calibration problem.
            ValueError: If its noise is unknown-scale or Student-t.

        References:
            [REF-HUAN-MARZOUK-2013]. No equation is evaluated.

        """
        if not isinstance(problem, CalibrationProblem):
            raise TypeError("problem must be a CalibrationProblem")
        return cls(problem.model, problem.noise)

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """Return the names of the uncertain parameters.

        References:
            Project API policy. No equation is evaluated.

        """
        return self.model.inputs.names

    @property
    def dimension(self) -> int:
        """Return the number of uncertain parameters.

        References:
            Project API policy. No equation is evaluated.

        """
        return self.model.dimension

    def _checked_design(self, design: Sequence[Condition]) -> tuple[Condition, ...]:
        """Validate and freeze a list of experimental conditions.

        References:
            [EQ-CAL-NL-001]. No equation is evaluated.

        """
        conditions = tuple(design)
        if not conditions:
            raise ValueError("a design needs at least one condition")
        parameters = set(self.parameter_names)
        frozen: list[Condition] = []
        for index, condition in enumerate(conditions):
            if not isinstance(condition, Mapping):
                raise TypeError(f"design[{index}] must be a mapping")
            values = {
                str(name): require_finite_real(value, name=f"conditions[{name!r}]")
                for name, value in condition.items()
            }
            shadow = parameters & set(values)
            if shadow:
                raise ValueError(
                    f"condition {index} shadows parameters {sorted(shadow)}"
                )
            frozen.append(MappingProxyType(values))
        return tuple(frozen)

    def noise_scales(self, design: Sequence[Condition]) -> Vector:
        """Return the total noise scale of every observation of a design.

        Equation:
            [EQ-CAL-LIK-001].

        References:
            [REF-KENNEDY-OHAGAN-2001].

        """
        conditions = self._checked_design(design)
        scales = tuple(
            math.sqrt((item.standard_deviation or 0.0) ** 2 + item.discrepancy_variance)
            for item in self.noise
        )
        return tuple(s for _ in conditions for s in scales)

    def predictions(
        self, theta: Sequence[float], design: Sequence[Condition]
    ) -> Vector:
        """Return noise-free predictions of every observation of a design.

        Inputs are merged as ``fixed``, then the condition, then ``theta``;
        one forward-model evaluation per condition is counted.

        Args:
            theta: Parameter vector in prior order.
            design: Experimental conditions.

        Returns:
            Predictions ordered condition-major, then by noise model.

        Raises:
            ValueError: If ``theta`` has the wrong length.

        Equation:
            [EQ-UQ-001], [EQ-CAL-NL-001].

        References:
            [REF-SMITH-UQ-2024, Chapter 8].

        """
        return self._predictions(
            as_vector(theta, name="theta"), self._checked_design(design)
        )

    def _predictions(self, theta: Vector, conditions: Sequence[Condition]) -> Vector:
        """Evaluate predictions for validated inputs.

        References:
            [EQ-CAL-NL-001], [REF-SMITH-UQ-2024, Chapter 8].

        """
        if len(theta) != self.dimension:
            raise ValueError("theta length must equal the number of parameters")
        named = self.model.inputs.as_mapping(theta)
        quantities = tuple(self.model.quantity(item.quantity) for item in self.noise)
        output: list[float] = []
        for condition in conditions:
            values = dict(self.model.fixed)
            values.update(condition)
            values.update(named)
            self.model.counter.add()
            result = evaluate_model(self.model.model, values, quantities)
            output.extend(result[item.quantity] for item in self.noise)
        return tuple(output)

    def standard_jacobian(
        self, u: Sequence[float], design: Sequence[Condition]
    ) -> tuple[Matrix, int]:
        """Return the Jacobian of noise-whitened predictions in prior space.

        With ``theta = T^-1(u)`` the Nataf map of the prior (``u ~ N(0, I)``),
        this is ``J_u = d[g(T^-1(u)) / s] / du`` by central differences.

        Args:
            u: Standard-normal coordinates of the prior.
            design: Experimental conditions.

        Returns:
            Jacobian and the number of forward-model evaluations.

        Equation:
            [EQ-OED-LAP-001], [EQ-MOD-NATAF-001], [EQ-OPT-FD-001].

        References:
            [REF-LIU-DERKIUREGHIAN-1986], [REF-NOCEDAL-WRIGHT-2006, Section 8.1].

        """
        conditions = self._checked_design(design)
        point = as_vector(u, name="u")
        if len(point) != self.dimension:
            raise ValueError("u length must equal the number of parameters")
        scales = self.noise_scales(conditions)

        def whitened(z: Vector) -> Vector:
            """Return predictions divided by noise scales.

            References:
                [EQ-OED-LAP-001]. No new equation is introduced.

            """
            theta = self.model.inputs.from_standard_normal(z)
            return tuple(
                g / s
                for g, s in zip(
                    self._predictions(theta, conditions), scales, strict=True
                )
            )

        jacobian, calls = central_jacobian(whitened, point)
        return jacobian, calls * len(conditions)


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class InformationGainEstimate:
    """Expected information gain of one design, with its cost.

    Args:
        value: EIG estimate in nats.
        standard_error: Monte Carlo standard error over the outer samples
            (0 for deterministic estimators). For nested Monte Carlo it omits
            the inner-sample bias and, with reused samples, the dependence
            between outer terms.
        method: ``"linearized"``, ``"laplace"``, ``"nested"`` or
            ``"nested-reuse"``.
        outer_samples: Outer (prior) samples.
        inner_samples: Inner samples per outer sample (0 if none).
        model_evaluations: Forward-model evaluations used.
        bias: ``"exact-if-linear-gaussian"``, ``"upward"`` (expected value
            ``>= EIG``), or ``"downward"`` (expected value ``<= EIG``).

    References:
        [REF-RYAN-2003], [REF-HUAN-MARZOUK-2013], [REF-RAINFORTH-ETAL-2018].

    """

    value: float
    standard_error: float
    method: str
    outer_samples: int
    inner_samples: int
    model_evaluations: int
    bias: str


def linearized_information_gain(
    problem: BayesianDesignProblem,
    design: Sequence[Condition],
    *,
    point: Sequence[float] | None = None,
) -> InformationGainEstimate:
    """Return the linear-Gaussian EIG of the model linearized at one point.

    In the prior's standard-normal space the prior is ``N(0, I)``, so the
    closed form (EQ-OED-EIG-LIN-001) reads ``1/2 log det(I + J_u' J_u)`` with
    whitened Jacobian ``J_u`` at ``point`` (default ``u = 0``, the prior
    median). It is exact for a linear model with a Gaussian prior, and is a
    local Bayesian D-criterion otherwise.

    Args:
        problem: Design problem.
        design: Experimental conditions.
        point: Linearization point in standard-normal coordinates.

    Returns:
        Deterministic estimate (standard error 0).

    Equation:
        [EQ-OED-EIG-LIN-001], [EQ-OED-LAP-001].

    References:
        [REF-CHALONER-VERDINELLI-1995], [REF-ATKINSON-DONEV-TOBIAS-2007].

    """
    u = (0.0,) * problem.dimension if point is None else as_vector(point, name="point")
    jacobian, evaluations = problem.standard_jacobian(u, design)
    value = _half_log_det_identity_plus(matmul(transpose(jacobian), jacobian))
    return InformationGainEstimate(
        value, 0.0, "linearized", 1, 0, evaluations, "exact-if-linear-gaussian"
    )


def _standard_normal_points(
    count: int, dimension: int, stream: RandomStream
) -> tuple[Vector, ...]:
    """Draw independent standard-normal vectors from one stream.

    References:
        [REF-MATSUMOTO-NISHIMURA-1998]. No equation is evaluated.

    """
    generator = stream.generator()
    return tuple(
        tuple(generator.gauss(0.0, 1.0) for _ in range(dimension)) for _ in range(count)
    )


def _mean_and_error(values: Sequence[float]) -> tuple[float, float]:
    """Return the sample mean and its standard error.

    References:
        [EQ-MC-MEAN-001], [REF-ROBERT-CASELLA-2004, Chapter 3].

    """
    mean = math.fsum(values) / len(values)
    if len(values) < 2:
        return mean, math.inf
    return mean, statistics.stdev(values) / math.sqrt(len(values))


def laplace_information_gain(
    problem: BayesianDesignProblem,
    design: Sequence[Condition],
    *,
    samples: int = 32,
    seed: int | RandomStream | None = None,
) -> InformationGainEstimate:
    """Return the prior-averaged Laplace approximation of the EIG.

    In standard-normal coordinates ``u`` (prior ``N(0, I)``; the EIG is
    reparameterization invariant), the Laplace posterior for data generated
    at ``u`` is approximated by ``N(u, (I + J_u'J_u)^-1)`` with the
    Gauss-Newton Hessian. Prior entropy minus expected posterior entropy is
    ``EIG ~= E_u[1/2 log det(I + J_u' J_u)]``, estimated by plain Monte Carlo
    over ``samples`` prior draws (EQ-OED-LAP-001). It is exact for linear
    models with Gaussian priors and ignores posterior non-Gaussianity
    otherwise. Cost: ``samples * 2 d * len(design)`` model evaluations.

    Args:
        problem: Design problem.
        design: Experimental conditions.
        samples: Positive number of prior draws.
        seed: Integer seed or stream (reuse it across designs for common
            random numbers).

    Returns:
        Estimate with Monte Carlo standard error.

    Raises:
        ValueError: If ``samples`` is not positive.

    Equation:
        [EQ-OED-LAP-001], [EQ-OED-EIG-LIN-001].

    References:
        [REF-CHALONER-VERDINELLI-1995], [REF-TIERNEY-KADANE-1986].

    """
    count = require_non_negative_integer(samples, name="samples")
    if count == 0:
        raise ValueError("samples must be positive")
    stream = resolve_stream(seed)
    terms: list[float] = []
    evaluations = 0
    for u in _standard_normal_points(count, problem.dimension, stream.child("laplace")):
        jacobian, calls = problem.standard_jacobian(u, design)
        evaluations += calls
        terms.append(_half_log_det_identity_plus(matmul(transpose(jacobian), jacobian)))
    value, error = _mean_and_error(terms)
    if count == 1:
        error = 0.0
    return InformationGainEstimate(
        value, error, "laplace", count, 0, evaluations, "exact-if-linear-gaussian"
    )


def nested_monte_carlo_information_gain(
    problem: BayesianDesignProblem,
    design: Sequence[Condition],
    *,
    outer: int = 1000,
    inner: int = 200,
    reuse_samples: bool = False,
    seed: int | RandomStream | None = None,
) -> InformationGainEstimate:
    """Estimate the EIG by nested Monte Carlo.

    Draw ``theta_n`` from the prior and ``y_n = g(theta_n) + e_n``. Then

    ``EIG_hat = (1/N) sum_n [log p(y_n | theta_n)
    - log((1/M) sum_m p(y_n | theta_nm))]``  (EQ-OED-NMC-001).

    * ``reuse_samples=False`` (Ryan 2003): fresh inner draws ``theta_nm``
      for every ``n``. By Jensen's inequality the inner log-mean is biased
      low, so the estimator is biased **upward**; the bias is ``O(1/M)`` and
      the variance ``O(1/N)`` (Rainforth et al. 2018), so ``M`` of order
      ``sqrt(N)`` balances them. Cost ``N (1 + M)`` model evaluations.
    * ``reuse_samples=True`` (Huan-Marzouk 2013): the outer samples serve as
      the inner sample (``M = N``, including ``theta_n`` itself), so only
      ``N`` model evaluations are needed. Each term is then a prior
      contrastive bound (Foster et al. 2020): its expectation is **below**
      the EIG, and the estimate never exceeds ``log N``. Including
      ``theta_n`` also prevents a numerically zero evidence estimate. The
      shared inner sample correlates the terms, so the reported standard
      error is optimistic.

    Gaussian normalizing constants cancel between the two log terms and are
    omitted.

    Args:
        problem: Design problem.
        design: Experimental conditions.
        outer: Outer sample size ``N`` (at least 2).
        inner: Inner sample size ``M`` (ignored when reusing samples).
        reuse_samples: Reuse the outer samples as the inner sample.
        seed: Integer seed or stream (reuse it across designs for common
            random numbers).

    Returns:
        Estimate with outer-sample standard error and cost.

    Raises:
        ValueError: If the sample sizes are too small.

    Equation:
        [EQ-OED-NMC-001], [EQ-OED-EIG-001].

    References:
        [REF-RYAN-2003], [REF-HUAN-MARZOUK-2013], [REF-RAINFORTH-ETAL-2018],
        [REF-FOSTER-ETAL-2020].

    """
    n_outer = require_non_negative_integer(outer, name="outer")
    n_inner = require_non_negative_integer(inner, name="inner")
    if n_outer < 2:
        raise ValueError("outer must be at least 2")
    if not reuse_samples and n_inner < 1:
        raise ValueError("inner must be positive without sample reuse")
    conditions = problem._checked_design(design)
    scales = problem.noise_scales(conditions)
    stream = resolve_stream(seed)
    start = problem.model.counter.count
    thetas = problem.model.inputs.sample(n_outer, seed=stream.child("outer"))
    centers = [problem._predictions(theta, conditions) for theta in thetas]
    generator = stream.child("noise").generator()
    data = [
        tuple(g + s * generator.gauss(0.0, 1.0) for g, s in zip(c, scales, strict=True))
        for c in centers
    ]

    inverse_scales = tuple(1.0 / s for s in scales)

    def log_kernel(y: Vector, g: Vector) -> float:
        """Return the Gaussian log-likelihood without its constant.

        References:
            [EQ-CAL-LIK-001]. No new equation is introduced.

        """
        total = 0.0
        for a, b, w in zip(y, g, inverse_scales, strict=True):
            r = (a - b) * w
            total += r * r
        return -0.5 * total

    terms: list[float] = []
    for index, (y, center) in enumerate(zip(data, centers, strict=True)):
        if reuse_samples:
            contrast = centers
        else:
            inner_thetas = problem.model.inputs.sample(
                n_inner, seed=stream.child("inner", index)
            )
            contrast = [problem._predictions(t, conditions) for t in inner_thetas]
        evidence = _log_mean_exp([log_kernel(y, g) for g in contrast])
        terms.append(log_kernel(y, center) - evidence)
    value, error = _mean_and_error(terms)
    return InformationGainEstimate(
        value,
        error,
        "nested-reuse" if reuse_samples else "nested",
        n_outer,
        n_outer if reuse_samples else n_inner,
        problem.model.counter.count - start,
        "downward" if reuse_samples else "upward",
    )


# ---------------------------------------------------------------------------
# Greedy sequential design
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GreedyDesignResult:
    """Conditions chosen one at a time by maximal expected information gain.

    Args:
        selected: Chosen conditions, in selection order.
        indices: Their indices in the candidate list.
        information_gains: EIG of ``baseline + selected[:k+1]`` for each step.
        increments: EIG gained at each step (conditional mutual
            information of the new condition given the earlier ones).
        baseline_information_gain: EIG of the baseline alone (0 if empty).
        candidate_scores: EIG of every candidate at every step.
        method: Estimator used.
        model_evaluations: Forward-model evaluations used.

    References:
        [REF-CHALONER-VERDINELLI-1995], [REF-HUAN-MARZOUK-2013],
        [REF-COVER-THOMAS-2006, Chapter 2].

    """

    selected: tuple[Condition, ...]
    indices: tuple[int, ...]
    information_gains: Vector
    increments: Vector
    baseline_information_gain: float
    candidate_scores: tuple[Vector, ...]
    method: str
    model_evaluations: int


def expected_information_gain(
    problem: BayesianDesignProblem,
    design: Sequence[Condition],
    *,
    method: str = "laplace",
    samples: int = 32,
    outer: int = 1000,
    inner: int = 200,
    reuse_samples: bool = True,
    seed: int | RandomStream | None = None,
) -> InformationGainEstimate:
    """Dispatch to one EIG estimator by name.

    Args:
        problem: Design problem.
        design: Experimental conditions.
        method: ``"linearized"``, ``"laplace"``, or ``"nested"``.
        samples: Prior draws for ``"laplace"``.
        outer: Outer samples for ``"nested"``.
        inner: Inner samples for ``"nested"`` without reuse.
        reuse_samples: Sample reuse for ``"nested"``.
        seed: Integer seed or stream.

    Returns:
        The estimate.

    Raises:
        ValueError: For an unknown method.

    Equation:
        [EQ-OED-EIG-001].

    References:
        [REF-CHALONER-VERDINELLI-1995], [REF-RYAN-2003].

    """
    if method == "linearized":
        return linearized_information_gain(problem, design)
    if method == "laplace":
        return laplace_information_gain(problem, design, samples=samples, seed=seed)
    if method == "nested":
        return nested_monte_carlo_information_gain(
            problem,
            design,
            outer=outer,
            inner=inner,
            reuse_samples=reuse_samples,
            seed=seed,
        )
    raise ValueError(f"method must be one of {_METHODS}")


def greedy_design(
    problem: BayesianDesignProblem,
    candidates: Sequence[Condition],
    size: int,
    *,
    baseline: Sequence[Condition] = (),
    allow_repeats: bool = True,
    method: str = "laplace",
    samples: int = 32,
    outer: int = 1000,
    inner: int = 200,
    reuse_samples: bool = True,
    seed: int | RandomStream | None = 0,
) -> GreedyDesignResult:
    """Select conditions greedily by expected information gain.

    Step ``k`` adds the candidate ``c`` maximizing
    ``EIG(baseline + selected + [c])``. By the chain rule of mutual
    information this equals maximizing the conditional gain
    ``I(theta; y_c | y_previous)`` (EQ-OED-GREEDY-001). Within one step every
    candidate is scored with the same random stream (common random numbers),
    so their differences are much less noisy than their values. Ties go to
    the earliest candidate. Greedy selection is myopic and is not guaranteed
    to find the best batch design.

    Args:
        problem: Design problem.
        candidates: Candidate conditions.
        size: Positive number of conditions to select.
        baseline: Conditions already planned or performed.
        allow_repeats: Whether a candidate may be chosen more than once.
        method: ``"linearized"``, ``"laplace"``, or ``"nested"``.
        samples: Prior draws for ``"laplace"``.
        outer: Outer samples for ``"nested"``.
        inner: Inner samples for ``"nested"`` without reuse.
        reuse_samples: Sample reuse for ``"nested"``.
        seed: Integer seed or stream.

    Returns:
        Selection, EIG path, per-step scores, and cost.

    Raises:
        ValueError: For an empty candidate list, an invalid size, or too few
            candidates without repeats.

    Equation:
        [EQ-OED-GREEDY-001], [EQ-OED-EIG-001].

    References:
        [REF-CHALONER-VERDINELLI-1995], [REF-HUAN-MARZOUK-2013],
        [REF-COVER-THOMAS-2006, Chapter 2].

    """
    pool = problem._checked_design(candidates)
    steps = require_non_negative_integer(size, name="size")
    if steps == 0:
        raise ValueError("size must be positive")
    if not allow_repeats and steps > len(pool):
        raise ValueError("not enough candidates for a selection without repeats")
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}")
    prior = tuple(baseline)
    if prior:
        prior = problem._checked_design(prior)
    stream = resolve_stream(seed)
    start = problem.model.counter.count

    def score(design: Sequence[Condition], step: int | str) -> float:
        """Return the EIG of a design with the step's common stream.

        References:
            [EQ-OED-EIG-001]. No new equation is introduced.

        """
        return expected_information_gain(
            problem,
            design,
            method=method,
            samples=samples,
            outer=outer,
            inner=inner,
            reuse_samples=reuse_samples,
            seed=stream.child("step", step),
        ).value

    base_value = score(prior, "baseline") if prior else 0.0
    chosen: list[int] = []
    gains: list[float] = []
    increments: list[float] = []
    table: list[Vector] = []
    previous = base_value
    for step in range(steps):
        current = list(prior) + [pool[i] for i in chosen]
        values = tuple(
            score(current + [candidate], step)
            if allow_repeats or index not in chosen
            else -math.inf
            for index, candidate in enumerate(pool)
        )
        best = max(range(len(pool)), key=lambda i: (values[i], -i))
        chosen.append(best)
        gains.append(values[best])
        increments.append(values[best] - previous)
        previous = values[best]
        table.append(values)
    return GreedyDesignResult(
        tuple(pool[i] for i in chosen),
        tuple(chosen),
        tuple(gains),
        tuple(increments),
        base_value,
        tuple(table),
        method,
        problem.model.counter.count - start,
    )
