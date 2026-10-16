"""Forward propagation of a :class:`~probstat.modeling.ProbabilisticModel`.

``propagate`` draws inputs from the model's joint law with a chosen design
(Monte Carlo, Latin hypercube, Halton, or Sobol), evaluates every quantity of
interest once per input, and returns raw outputs together with distributional
summaries: moments with sampling uncertainty, quantiles, exceedance curves,
and input-output rank correlations.

Accuracy statements:

- ``replicates=r`` repeats the design with ``r`` independent randomizations
  (child streams ``replicate/k``) and gives Student-t intervals over the
  replicate means and exceedance fractions. This is the honest error bar of
  randomized quasi-Monte Carlo; the single-run Monte Carlo standard error is
  conservative (often by orders of magnitude) for LHS and QMC designs.
- :meth:`OutputDistribution.quantile_interval` is a distribution-free
  order-statistic interval, exact for independent draws of a continuous law.
- :meth:`OutputDistribution.variance_interval` gives the normal-theory
  chi-square interval (valid only for near-normal outputs) or a percentile
  bootstrap.

Failed simulator runs follow the model's failure policy; under ``"skip"`` or
``"record"`` the study keeps the successful runs, lists the failures, and
reports the failure fraction. Dropping failed runs biases every estimate
unless failing is unrelated to the output (see :mod:`probstat.uq.evaluation`).
"""

from __future__ import annotations

import math
import statistics as _statistics
import warnings
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import Executor
from dataclasses import dataclass, field

from probstat.distributions import ChiSquare
from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.inference import ConfidenceInterval, student_t_quantile
from probstat.linalg import Vector
from probstat.modeling import ProbabilisticModel
from probstat.monte_carlo import (
    MonteCarloEstimate,
    ProbabilityEstimate,
    estimate_mean,
    estimate_probability,
)
from probstat.sampling import (
    SamplingDesign,
    halton,
    latin_hypercube,
    monte_carlo_design,
    sobol,
)
from probstat.statistics import (
    excess_kurtosis,
    quantile,
    skewness,
    spearman_correlation,
)
from probstat.uq import (
    EvaluationCache,
    FailedEvaluation,
    FailurePolicy,
    FrozenMapping,
)

DESIGNS = ("mc", "lhs", "halton", "sobol")


@dataclass(frozen=True, slots=True)
class QuantileInterval:
    """Distribution-free confidence interval for one output quantile.

    Args:
        probability: Quantile level ``p``.
        estimate: Type-7 sample quantile.
        lower: Order statistic ``Y_(l)``.
        upper: Order statistic ``Y_(u)``.
        lower_rank: One-based rank ``l``.
        upper_rank: One-based rank ``u``.
        confidence: Requested confidence.
        coverage: Exact coverage ``B(u-1) - B(l-1)`` of the rank pair for
            independent draws of a continuous law (at least ``confidence``).

    References:
        [REF-DAVID-NAGARAJA-2003, Section 7.1].

    """

    probability: float
    estimate: float
    lower: float
    upper: float
    lower_rank: int
    upper_rank: int
    confidence: float
    coverage: float


def _binomial_cdf_table(size: int, probability: float) -> list[float]:
    """Return ``B(k) = P(Bin(size, probability) <= k)`` for ``k = 0..size``.

    Probabilities are summed in log space from the mode outwards so that the
    table is accurate for large ``size``.

    References:
        [REF-DAVID-NAGARAJA-2003, Section 7.1]. Evaluates [EQ-PROP-QCI-001].

    """
    log_p, log_q = math.log(probability), math.log1p(-probability)
    log_n = math.lgamma(size + 1.0)
    logs = [
        log_n
        - math.lgamma(k + 1.0)
        - math.lgamma(size - k + 1.0)
        + k * log_p
        + (size - k) * log_q
        for k in range(size + 1)
    ]
    peak = max(logs)
    weights = [math.exp(value - peak) for value in logs]
    total = math.fsum(weights)
    table: list[float] = []
    running: list[float] = []
    for weight in weights:
        running.append(weight)
        table.append(min(1.0, math.fsum(running) / total))
    return table


def order_statistic_interval(
    values: Sequence[float], probability: float, *, confidence: float = 0.95
) -> QuantileInterval:
    """Return the equal-tailed distribution-free interval for a quantile.

    For ``n`` independent draws of a continuous law the event
    ``Y_(l) <= xi_p <= Y_(u)`` has probability ``B(u-1) - B(l-1)`` with
    ``B`` the ``Binomial(n, p)`` CDF, whatever the law. The ranks are
    ``l = max{k : B(k-1) <= alpha/2}`` and ``u = min{k : B(k-1) >= 1-alpha/2}``,
    so each tail is at most ``alpha/2`` and the coverage is at least
    ``1 - alpha``. For LHS and quasi-random samples the draws are not
    independent and the interval is a (usually conservative) reference.

    Args:
        values: Finite sample of at least two values.
        probability: Quantile level strictly inside ``(0, 1)``.
        confidence: Requested confidence strictly inside ``(0, 1)``.

    Returns:
        Order-statistic interval with its exact coverage.

    Raises:
        ValueError: If a level is outside ``(0, 1)`` or the sample is too
            small for the requested confidence at this level.

    Equation:
        [EQ-PROP-QCI-001], [EQ-STAT-QUANT-001].

    References:
        [REF-DAVID-NAGARAJA-2003, Section 7.1], [REF-HYNDMAN-FAN-1996].

    """
    p = require_finite_real(probability, name="probability")
    level = require_finite_real(confidence, name="confidence")
    if not 0.0 < p < 1.0:
        raise ValueError("probability must lie strictly between zero and one")
    if not 0.0 < level < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    ordered = sorted(values)
    size = len(ordered)
    if size < 2:
        raise ValueError("a quantile interval needs at least two values")
    tail = 0.5 * (1.0 - level)
    table = _binomial_cdf_table(size, p)
    # B(k - 1) for rank k = 1..n is table[k - 1].
    lower_rank = 0
    for rank in range(1, size + 1):
        if table[rank - 1] <= tail:
            lower_rank = rank
        else:
            break
    upper_rank = 0
    for rank in range(size, 0, -1):
        if table[rank - 1] >= 1.0 - tail:
            upper_rank = rank
        else:
            break
    if lower_rank == 0 or upper_rank == 0 or upper_rank <= lower_rank:
        raise ValueError(
            f"{size} values are too few for a {level:g} distribution-free "
            f"interval of the {p:g} quantile"
        )
    coverage = table[upper_rank - 1] - table[lower_rank - 1]
    return QuantileInterval(
        p,
        quantile(ordered, p),
        ordered[lower_rank - 1],
        ordered[upper_rank - 1],
        lower_rank,
        upper_rank,
        level,
        coverage,
    )


def _check_confidence(confidence: float) -> float:
    """Return a validated confidence level strictly inside ``(0, 1)``.

    References:
        Project validation policy. No equation is evaluated.

    """
    level = require_finite_real(confidence, name="confidence")
    if not 0.0 < level < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    return level


def replicate_interval(
    estimates: Sequence[float], *, confidence: float = 0.95, label: str = "mean"
) -> ConfidenceInterval:
    """Return the Student-t interval over independent replicate estimates.

    ``theta_hat = mean(theta_k)`` and ``theta_hat +- t_{1-alpha/2, r-1}
    s_r / sqrt(r)`` where ``s_r`` is the standard deviation of the ``r``
    replicate estimates. With independently randomized (R)QMC replicates the
    replicate estimates are i.i.d. and unbiased, so the interval is the
    standard RQMC error statement; the t quantile assumes their distribution
    is close to normal, which holds well for means of many points.

    Args:
        estimates: At least two replicate estimates.
        confidence: Confidence strictly inside ``(0, 1)``.
        label: Estimated quantity named in the method label.

    Returns:
        Confidence interval.

    Raises:
        ValueError: If fewer than two estimates are given.

    Equation:
        [EQ-PROP-RQMC-CI-001], [EQ-SAMP-RQMC-001].

    References:
        [REF-L-ECUYER-2018], [REF-OWEN-2013, Chapter 17].

    """
    level = _check_confidence(confidence)
    values = tuple(require_finite_real(v, name="estimate") for v in estimates)
    count = len(values)
    if count < 2:
        raise ValueError("a replicate interval needs at least two replicates")
    center = math.fsum(values) / count
    spread = math.sqrt(math.fsum((v - center) ** 2 for v in values) / (count - 1))
    half = student_t_quantile(0.5 + 0.5 * level, count - 1) * spread / math.sqrt(count)
    return ConfidenceInterval(
        center - half,
        center + half,
        level,
        f"Student-t interval over {count} randomized replicates of the {label}",
    )


@dataclass(frozen=True, slots=True)
class OutputDistribution:
    """Empirical distribution of one propagated quantity of interest.

    Args:
        name: QoI name.
        values: Propagated outputs in input-design order.
        mean_estimate: Mean with variance, standard error, and interval. The
            interval assumes independent draws; for LHS/QMC designs it is a
            conservative Monte Carlo reference, not a randomized-QMC interval
            (use ``replicates`` in :func:`propagate`).
        replicate_sizes: Number of kept outputs in each randomized replicate,
            in order (``values`` is their concatenation); empty without
            replicates.

    References:
        [REF-SMITH-UQ-2024, Chapters 3-4], [REF-HYNDMAN-FAN-1996].

    """

    name: str
    values: tuple[float, ...]
    mean_estimate: MonteCarloEstimate
    replicate_sizes: tuple[int, ...] = field(default=(), repr=False)

    @property
    def mean(self) -> float:
        """Return the sample mean.

        Equation:
            [EQ-MC-MEAN-001].

        References:
            [REF-SMITH-UQ-2024, Chapter 3].

        """
        return self.mean_estimate.estimate

    @property
    def standard_deviation(self) -> float:
        """Return the unbiased sample standard deviation.

        Equation:
            [EQ-MC-VAR-001].

        References:
            [REF-SMITH-UQ-2024, Chapter 3].

        """
        return math.sqrt(self.mean_estimate.sample_variance)

    @property
    def skewness(self) -> float:
        """Return the moment skewness of the outputs.

        Equation:
            [EQ-STAT-SHAPE-001].

        References:
            [REF-JOANES-GILL-1998].

        """
        return skewness(self.values)

    @property
    def excess_kurtosis(self) -> float:
        """Return the moment excess kurtosis of the outputs.

        Equation:
            [EQ-STAT-SHAPE-001].

        References:
            [REF-JOANES-GILL-1998].

        """
        return excess_kurtosis(self.values)

    def quantile(self, probability: float) -> float:
        """Return the type-7 sample quantile of the outputs.

        Args:
            probability: Level in ``[0, 1]``.

        Returns:
            Output quantile.

        Equation:
            [EQ-STAT-QUANT-001].

        References:
            [REF-HYNDMAN-FAN-1996].

        """
        return quantile(self.values, probability)

    def exceedance(
        self, threshold: float, *, confidence: float = 0.95
    ) -> ProbabilityEstimate:
        """Estimate ``P(Y > threshold)`` with a Wilson interval.

        Args:
            threshold: Finite threshold.
            confidence: Nominal interval confidence.

        Returns:
            Probability estimate.

        Equation:
            [EQ-PROP-THRESH-001], [EQ-MC-PROB-001].

        References:
            [REF-WILSON-1927], [REF-SMITH-UQ-2024, Chapter 4].

        """
        level = require_finite_real(threshold, name="threshold")
        return estimate_probability(
            (value > level for value in self.values), confidence=confidence
        )

    @property
    def replicates(self) -> int:
        """Return the number of randomized replicates (0 without them).

        References:
            [REF-L-ECUYER-2018]. No equation is evaluated.

        """
        return len(self.replicate_sizes)

    def _replicate_blocks(self) -> list[tuple[float, ...]]:
        """Return the non-empty replicate blocks of ``values``.

        References:
            [REF-L-ECUYER-2018]. No equation is evaluated.

        """
        blocks: list[tuple[float, ...]] = []
        start = 0
        for size in self.replicate_sizes:
            if size:
                blocks.append(self.values[start : start + size])
            start += size
        return blocks

    def replicate_means(self) -> tuple[float, ...]:
        """Return the mean of each non-empty randomized replicate.

        Equation:
            [EQ-MC-MEAN-001].

        References:
            [REF-L-ECUYER-2018].

        """
        blocks = self._replicate_blocks()
        return tuple(math.fsum(block) / len(block) for block in blocks)

    def mean_interval(self, confidence: float = 0.95) -> ConfidenceInterval:
        """Return the honest confidence interval of the mean.

        With replicates this is the Student-t interval over replicate means
        (valid for MC, LHS, and randomized QMC); without them it is the
        independent-draw normal interval, valid for plain Monte Carlo only
        and conservative for LHS and QMC.

        Args:
            confidence: Confidence strictly inside ``(0, 1)``.

        Returns:
            Confidence interval.

        Raises:
            ValueError: If fewer than two replicates have outputs.

        Equation:
            [EQ-PROP-RQMC-CI-001], [EQ-INF-CI-NMEAN-001].

        References:
            [REF-L-ECUYER-2018], [REF-OWEN-2013, Chapter 17].

        """
        if self.replicate_sizes:
            return replicate_interval(
                self.replicate_means(), confidence=confidence, label="mean"
            )
        return estimate_mean(self.values, confidence=confidence).confidence_interval

    def exceedance_interval(
        self, threshold: float, *, confidence: float = 0.95
    ) -> ConfidenceInterval:
        """Return a confidence interval for ``P(Y > threshold)``.

        With replicates: the Student-t interval over replicate exceedance
        fractions, clipped to ``[0, 1]``; when every replicate gives the same
        fraction (for example no exceedance at all) the t interval collapses
        to a point, so the Wilson interval of the pooled sample is returned
        instead and labelled as such. Without replicates: the Wilson
        interval (exact only for independent draws).

        Args:
            threshold: Finite threshold.
            confidence: Confidence strictly inside ``(0, 1)``.

        Returns:
            Confidence interval.

        Equation:
            [EQ-PROP-RQMC-CI-001], [EQ-PROP-THRESH-001],
            [EQ-INF-CI-WILSON-001].

        References:
            [REF-L-ECUYER-2018], [REF-WILSON-1927].

        """
        level = require_finite_real(threshold, name="threshold")
        if self.replicate_sizes:
            fractions = tuple(
                sum(value > level for value in block) / len(block)
                for block in self._replicate_blocks()
            )
            if len(set(fractions)) > 1:
                interval = replicate_interval(
                    fractions, confidence=confidence, label="exceedance fraction"
                )
                return ConfidenceInterval(
                    max(0.0, interval.lower),
                    min(1.0, interval.upper),
                    interval.confidence,
                    interval.method,
                )
            pooled = self.exceedance(level, confidence=confidence)
            return ConfidenceInterval(
                pooled.confidence_interval.lower,
                pooled.confidence_interval.upper,
                pooled.confidence_interval.confidence,
                "Wilson interval of the pooled sample (replicates show no spread)",
            )
        return self.exceedance(level, confidence=confidence).confidence_interval

    def quantile_interval(
        self, probability: float, *, confidence: float = 0.95
    ) -> QuantileInterval:
        """Return the distribution-free order-statistic interval of a quantile.

        Exact for independent draws of a continuous law (plain Monte Carlo);
        for LHS and quasi-random samples it is a conservative reference.

        Args:
            probability: Quantile level strictly inside ``(0, 1)``.
            confidence: Confidence strictly inside ``(0, 1)``.

        Returns:
            Interval with ranks and exact coverage.

        Equation:
            [EQ-PROP-QCI-001].

        References:
            [REF-DAVID-NAGARAJA-2003, Section 7.1].

        """
        return order_statistic_interval(
            self.values, probability, confidence=confidence
        )

    def variance_interval(
        self,
        confidence: float = 0.95,
        *,
        method: str = "normal",
        resamples: int = 2000,
        seed: int | RandomStream | None = None,
    ) -> ConfidenceInterval:
        """Return a confidence interval for the output variance.

        ``method="normal"`` is the chi-square interval
        ``[(n-1)s^2 / chi2_{1-alpha/2}, (n-1)s^2 / chi2_{alpha/2}]`` with
        ``n - 1`` degrees of freedom. It is exact only for normally
        distributed independent outputs and is badly wrong for heavy-tailed
        or skewed outputs (its coverage depends on the kurtosis even as
        ``n`` grows). ``method="bootstrap"`` is the percentile bootstrap of
        the sample variance, valid asymptotically for any law with a finite
        fourth moment, at the cost of ``resamples`` resamples.

        Args:
            confidence: Confidence strictly inside ``(0, 1)``.
            method: ``"normal"`` or ``"bootstrap"``.
            resamples: Bootstrap resamples (at least 100).
            seed: Seed or stream of the bootstrap.

        Returns:
            Confidence interval.

        Raises:
            ValueError: If the method or the number of resamples is invalid.

        Equation:
            [EQ-PROP-VARCI-001], [EQ-INF-BOOT-002].

        References:
            [REF-CASELLA-BERGER-2002, Chapter 9],
            [REF-EFRON-TIBSHIRANI-1993, Chapter 13].

        """
        level = _check_confidence(confidence)
        size = len(self.values)
        variance = self.mean_estimate.sample_variance
        tail = 0.5 * (1.0 - level)
        if method == "normal":
            law = ChiSquare(size - 1)
            return ConfidenceInterval(
                (size - 1) * variance / law.quantile(1.0 - tail),
                (size - 1) * variance / law.quantile(tail),
                level,
                "chi-square interval (assumes normal outputs)",
            )
        if method != "bootstrap":
            raise ValueError("method must be 'normal' or 'bootstrap'")
        if isinstance(resamples, bool) or not isinstance(resamples, int):
            raise ValueError("resamples must be an integer")
        if resamples < 100:
            raise ValueError("resamples must be at least 100")
        generator = resolve_stream(seed).child("variance-bootstrap").generator()
        draws = []
        for _ in range(resamples):
            sample = [self.values[generator.randrange(size)] for _ in range(size)]
            draws.append(_statistics.variance(sample))
        return ConfidenceInterval(
            quantile(draws, tail),
            quantile(draws, 1.0 - tail),
            level,
            f"percentile bootstrap of the variance ({resamples} resamples)",
        )

    def exceedance_curve(self, points: int = 50) -> tuple[tuple[float, float], ...]:
        """Return the empirical exceedance curve ``(y, P(Y > y))``.

        Args:
            points: Number of evenly spaced levels between the extremes.

        Returns:
            Pairs of output level and empirical exceedance fraction.

        Raises:
            ValueError: If ``points < 2``.

        Equation:
            [EQ-STAT-005], [EQ-PROP-THRESH-001].

        References:
            [REF-SMITH-UQ-2024, Chapter 4].

        """
        if points < 2:
            raise ValueError("points must be at least two")
        ordered = sorted(self.values)
        low, high = ordered[0], ordered[-1]
        n = len(ordered)
        curve: list[tuple[float, float]] = []
        index = 0
        for k in range(points):
            level = low + (high - low) * k / (points - 1)
            while index < n and ordered[index] <= level:
                index += 1
            curve.append((level, (n - index) / n))
        return tuple(curve)


@dataclass(frozen=True, slots=True)
class PropagationStudy:
    """Inputs, outputs, and summaries of one forward propagation.

    ``inputs`` and the output values hold the *kept* runs, aligned with each
    other; under the ``"skip"``/``"record"`` failure policies the failed
    runs are listed in ``failures`` and :attr:`failure_fraction` reports
    their share. Estimates from the kept runs are conditional on the
    simulator succeeding and are biased unless failures are ignorable.

    Args:
        model_name: Label of the propagated model.
        design: Unit-hypercube design that generated the inputs (all
            replicates concatenated).
        inputs: Physical input vectors of the kept runs in design order.
        input_names: Input names in model order.
        outputs: Read-only QoI-name mapping to output distributions.
        model_evaluations: Completed forward-model calls charged to this
            study (cache hits excluded).
        failures: Failed runs (indices refer to ``design.points``).
        failed_evaluations: Failed simulator attempts.
        cache_hits: Points answered from an evaluation cache.
        replicates: Number of randomized replicates (0 without them).

    References:
        [REF-SMITH-UQ-2024, Chapters 3-4], [REF-OWEN-2013, Chapter 10].

    """

    model_name: str
    design: SamplingDesign
    inputs: tuple[Vector, ...]
    input_names: tuple[str, ...]
    outputs: Mapping[str, OutputDistribution]
    model_evaluations: int
    failures: tuple[FailedEvaluation, ...] = field(default=(), repr=False)
    failed_evaluations: int = 0
    cache_hits: int = 0
    replicates: int = 0

    @property
    def requested(self) -> int:
        """Return the number of input points requested.

        References:
            [REF-SMITH-UQ-2024, Chapter 3]. No equation is evaluated.

        """
        return len(self.inputs) + len(self.failures)

    @property
    def failure_fraction(self) -> float:
        """Return the fraction of requested points without an output.

        References:
            [REF-SMITH-UQ-2024, Chapter 3]. No equation is evaluated.

        """
        total = self.requested
        return len(self.failures) / total if total else 0.0

    def __getitem__(self, name: str) -> OutputDistribution:
        """Return the output distribution of one QoI.

        Args:
            name: QoI name.

        Returns:
            Output distribution.

        References:
            [REF-SMITH-UQ-2024, Chapter 3]. No equation is evaluated.

        """
        return self.outputs[name]

    def input_column(self, name: str) -> tuple[float, ...]:
        """Return the sampled values of one input.

        Args:
            name: Input name.

        Returns:
            Values in design order.

        Raises:
            ValueError: If the input name is unknown.

        References:
            [REF-SMITH-UQ-2024, Chapter 3]. No equation is evaluated.

        """
        index = self.input_names.index(name)
        return tuple(point[index] for point in self.inputs)

    def rank_correlations(self, quantity: str) -> Mapping[str, float]:
        """Return Spearman correlations between each input and one QoI.

        This is a screening measure of monotone association, not a
        variance-based sensitivity index.

        Args:
            quantity: QoI name.

        Returns:
            Read-only input-name mapping to rank correlation.

        Equation:
            [EQ-STAT-RANK-001].

        References:
            [REF-HOLLANDER-WOLFE-CHICKEN-2014, Chapter 8],
            [REF-SALTELLI-2008, Chapter 1].

        """
        values = self.outputs[quantity].values
        result: dict[str, float] = {}
        for name in self.input_names:
            try:
                result[name] = spearman_correlation(self.input_column(name), values)
            except ValueError:
                result[name] = math.nan
        return FrozenMapping(result)


def make_design(
    method: str,
    size: int,
    dimension: int,
    *,
    seed: int | RandomStream | None = None,
) -> SamplingDesign:
    """Build a unit-hypercube design by name.

    Args:
        method: One of ``"mc"``, ``"lhs"``, ``"halton"``, or ``"sobol"``.
        size: Positive number of points.
        dimension: Positive dimension.
        seed: Seed for randomized designs; quasi-random designs are
            randomized (scrambled/shifted) whenever a seed is supplied.

    Returns:
        Sampling design.

    Raises:
        ValueError: If ``method`` is unknown.

    Equation:
        [EQ-SAMP-LHS-001], [EQ-SAMP-HALTON-001], [EQ-SAMP-SOBOL-001].

    References:
        [REF-OWEN-2013, Chapters 10 and 15].

    """
    if method == "mc":
        return monte_carlo_design(size, dimension, seed=seed)
    if method == "lhs":
        return latin_hypercube(size, dimension, seed=seed)
    if method == "halton":
        randomized = seed is not None
        return halton(size, dimension, seed=seed, scramble=randomized, shift=randomized)
    if method == "sobol":
        return sobol(size, dimension, seed=seed, scramble=seed is not None)
    raise ValueError(f"design must be one of {DESIGNS}")


def propagate(
    model: ProbabilisticModel,
    size: int,
    *,
    design: str = "mc",
    seed: int | RandomStream | None = None,
    confidence: float = 0.95,
    replicates: int | None = None,
    executor: Executor | None = None,
    batch: int | None = None,
    cache: EvaluationCache | None = None,
    on_failure: FailurePolicy | None = None,
) -> PropagationStudy:
    """Propagate the joint input law of a model through every QoI.

    Without ``replicates`` the design is drawn on the child stream
    ``("propagation", design)`` of ``seed`` (unchanged since version 0.15).
    With ``replicates=r`` the design of ``size`` points is drawn ``r`` times
    on the streams ``("propagation", design, "replicate", k)``; outputs are
    pooled in replicate order, the cost is ``r * size``, and
    :meth:`OutputDistribution.mean_interval` and
    :meth:`OutputDistribution.exceedance_interval` give Student-t intervals
    over the replicates (the honest randomized-QMC error statement).

    Failed simulator runs follow ``on_failure`` (default: the model's
    policy). Under ``"skip"``/``"record"`` the failed runs are dropped,
    listed in :attr:`PropagationStudy.failures`, and a warning states the
    failure fraction, because estimates from the surviving runs are biased
    unless failing is unrelated to the output.

    Args:
        model: Probabilistic model to propagate.
        size: Number of input realizations (per replicate), at least two.
        design: Sampling design name (see :data:`DESIGNS`).
        seed: Integer seed or stream; the design uses the child stream
            ``"propagation"`` so other workflows can share one root seed.
        confidence: Nominal confidence for mean intervals.
        replicates: Optional number of independent randomizations (>= 2).
        executor: Optional :class:`concurrent.futures.Executor` for parallel
            model runs (results keep design order).
        batch: Chunk size for the model's ``evaluate_batch`` hook or for
            executor jobs (see :meth:`ProbabilisticModel.evaluate_many`).
        cache: Optional exact-match evaluation cache.
        on_failure: Failure policy for this call.

    Returns:
        Propagation study with raw outputs and summaries.

    Raises:
        ValueError: If ``size < 2``, ``replicates < 2``, the design is
            unknown, or fewer than two runs succeed.

    Equation:
        [EQ-UQ-001], [EQ-MC-MEAN-001], [EQ-SAMP-INV-001],
        [EQ-SAMP-RQMC-001].

    References:
        [REF-SMITH-UQ-2024, Chapters 3-4], [REF-OWEN-2013, Chapter 10],
        [REF-L-ECUYER-2018].

    """
    if isinstance(size, bool) or not isinstance(size, int) or size < 2:
        raise ValueError("size must be an integer of at least two")
    if replicates is not None and (
        isinstance(replicates, bool)
        or not isinstance(replicates, int)
        or replicates < 2
    ):
        raise ValueError("replicates must be an integer of at least two")
    stream = resolve_stream(seed).child("propagation", design)
    if replicates is None:
        unit = make_design(design, size, model.dimension, seed=stream)
    else:
        parts = [
            make_design(
                design, size, model.dimension, seed=stream.child("replicate", k)
            )
            for k in range(replicates)
        ]
        unit = SamplingDesign(
            tuple(point for part in parts for point in part.points),
            parts[0].method,
            parts[0].randomized,
            stream.seed,
        )
    points = model.inputs.sample_design(unit)
    raw = model.evaluate_many(
        points, executor=executor, batch=batch, cache=cache, on_failure=on_failure
    )
    kept = raw.indices
    if len(kept) < 2:
        raise ValueError(
            f"only {len(kept)} of {len(points)} model runs succeeded; "
            "propagation needs at least two"
        )
    if raw.failures:
        warnings.warn(
            f"{len(raw.failures)} of {len(points)} model runs failed "
            f"({100.0 * raw.failure_fraction:.3g}%) and were dropped; "
            "estimates are conditional on success and biased unless failures "
            "are unrelated to the output",
            UserWarning,
            stacklevel=2,
        )
    replicate_sizes: tuple[int, ...] = ()
    if replicates is not None:
        counts = [0] * replicates
        for index in kept:
            counts[index // size] += 1
        replicate_sizes = tuple(counts)
    outputs = {
        name: OutputDistribution(
            name,
            values,
            estimate_mean(values, confidence=confidence),
            replicate_sizes,
        )
        for name, values in raw.items()
    }
    return PropagationStudy(
        model.name,
        unit,
        tuple(points[index] for index in kept),
        model.inputs.names,
        FrozenMapping(outputs),
        raw.model_evaluations,
        raw.failures,
        raw.failed_evaluations,
        raw.cache_hits,
        0 if replicates is None else replicates,
    )


def compare_designs(
    model: ProbabilisticModel,
    quantity: str,
    size: int,
    *,
    designs: Sequence[str] = DESIGNS,
    replicates: int = 10,
    seed: int | RandomStream | None = None,
) -> Mapping[str, tuple[float, float]]:
    """Compare designs by the spread of the mean estimate over replicates.

    Each design is repeated with independent randomization; the empirical
    standard deviation of the replicate means measures its estimation error.

    Args:
        model: Probabilistic model.
        quantity: QoI whose mean is estimated.
        size: Points per replicate.
        designs: Design names to compare.
        replicates: Independent replicates per design, at least two.
        seed: Root seed or stream.

    Returns:
        Read-only mapping ``design -> (mean of means, std of means)``.

    Raises:
        ValueError: If ``replicates < 2``.

    Equation:
        [EQ-MC-VAR-001], [EQ-SAMP-RQMC-001].

    References:
        [REF-OWEN-2013, Section 17.2], [REF-L-ECUYER-2018].

    """
    if replicates < 2:
        raise ValueError("replicates must be at least two")
    root = resolve_stream(seed)
    summary: dict[str, tuple[float, float]] = {}
    for method in designs:
        means: list[float] = []
        for replicate in range(replicates):
            study = propagate(
                model,
                size,
                design=method,
                seed=root.child("compare", method, replicate),
            )
            means.append(study[quantity].mean)
        center = math.fsum(means) / replicates
        spread = math.sqrt(
            math.fsum((value - center) ** 2 for value in means) / (replicates - 1)
        )
        summary[method] = (center, spread)
    return FrozenMapping(summary)


def summarize(
    values: Iterable[float], probabilities: Sequence[float] = (0.05, 0.5, 0.95)
) -> Mapping[str, float]:
    """Return a compact numeric summary of an output sample.

    Args:
        values: Finite output sample with at least two values.
        probabilities: Quantile levels to report.

    Returns:
        Mapping with ``mean``, ``std`` and ``q<level>`` entries.

    Equation:
        [EQ-MC-MEAN-001], [EQ-MC-VAR-001], [EQ-STAT-QUANT-001].

    References:
        [REF-HYNDMAN-FAN-1996], [REF-SMITH-UQ-2024, Chapter 3].

    """
    sample = tuple(values)
    estimate = estimate_mean(sample)
    result = {"mean": estimate.estimate, "std": math.sqrt(estimate.sample_variance)}
    for p in probabilities:
        result[f"q{p:g}"] = quantile(sample, p)
    return FrozenMapping(result)
