"""Pre-specified simulation studies for comparing statistical methods.

The structure follows the ADEMP scheme of Morris, White and Crowther (2019):
state the **A**ims, the **D**ata-generating mechanisms, the **E**stimand,
the **M**ethods, and the **P**erformance measures *before* running, then
report every performance measure with its Monte Carlo standard error.

A :class:`StudyProtocol` records those five items together with the number
of repetitions, the seed, and the primary comparison, and has a
:meth:`~StudyProtocol.fingerprint` (a SHA-256 hash of its canonical text).
Publishing the fingerprint before the run makes later changes of the design
detectable; the result carries the fingerprint of the protocol it executed.
This does not prevent a biased design, which no software can: it removes the
freedom to choose the design after seeing the results, which Pawel, Kook and
Reeve (2024) show is enough to make a useless method look superior.

Every method sees the same data set in each repetition (common random
numbers), so comparisons between methods are paired. Repetitions in which a
method fails (raises or returns non-finite values) are counted and reported
per method and never silently dropped from the denominator of the failure
rate; performance measures are computed over the repetitions in which the
method succeeded, which the report states.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Generic, TypeVar

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)

D = TypeVar("D")


@dataclass(frozen=True, slots=True)
class MethodOutput:
    """What one method returns for one data set.

    Args:
        estimate: Point estimate of the estimand.
        standard_error: Model-based standard error, or ``None`` when the
            method reports none.
        lower: Lower confidence limit, or ``None``.
        upper: Upper confidence limit, or ``None``.

    References:
        [REF-MORRIS-WHITE-CROWTHER-2019]. No equation is
        evaluated.

    """

    estimate: float
    standard_error: float | None = None
    lower: float | None = None
    upper: float | None = None


@dataclass(frozen=True, slots=True)
class DataGeneratingMechanism(Generic[D]):
    """A named way of simulating one data set with a known estimand value.

    Args:
        name: Label of the mechanism.
        generate: Maps a seeded :class:`random.Random` to one data set.
        truth: Value of the estimand under this mechanism.

    References:
        [REF-MORRIS-WHITE-CROWTHER-2019]. No equation is
        evaluated.

    """

    name: str
    generate: Callable[[random.Random], D]
    truth: float


@dataclass(frozen=True, slots=True)
class Method(Generic[D]):
    """A named analysis method applied to one data set.

    Args:
        name: Label of the method.
        fit: Maps a data set to a :class:`MethodOutput`. Exceptions are
            caught by the runner and counted as failures.

    References:
        [REF-MORRIS-WHITE-CROWTHER-2019]. No equation is
        evaluated.

    """

    name: str
    fit: Callable[[D], MethodOutput]


@dataclass(frozen=True, slots=True)
class StudyProtocol:
    """The pre-specified design of a simulation study (ADEMP).

    Args:
        aims: What the study is meant to find out.
        mechanisms: Names of the data-generating mechanisms.
        estimand: The estimand in words.
        methods: Names of the compared methods.
        measures: Names of the performance measures that will be reported.
        repetitions: Number of simulated data sets per mechanism.
        seed: Root seed.
        primary_comparison: The comparison the conclusion will rest on, in
            words (for example ``"relative precision of B against A under
            mechanism M1"``).

    References:
        [REF-MORRIS-WHITE-CROWTHER-2019] (ADEMP),
        [REF-PAWEL-KOOK-REEVE-2024] (pre-specification).

    """

    aims: str
    mechanisms: tuple[str, ...]
    estimand: str
    methods: tuple[str, ...]
    measures: tuple[str, ...]
    repetitions: int
    seed: int
    primary_comparison: str

    def __post_init__(self) -> None:
        """Validate that every ADEMP item is stated.

        Raises:
            ValueError: If a text item is empty, a list is empty or has
                duplicates, or the repetition count is below two.

        References:
            [REF-MORRIS-WHITE-CROWTHER-2019]. No equation is
            evaluated.

        """
        for label in ("aims", "estimand", "primary_comparison"):
            text = getattr(self, label)
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{label} must be stated")
        for label in ("mechanisms", "methods", "measures"):
            names = tuple(getattr(self, label))
            if not names or len(set(names)) != len(names):
                raise ValueError(f"{label} must be non-empty and without duplicates")
            object.__setattr__(self, label, names)
        if require_non_negative_integer(self.repetitions, name="repetitions") < 2:
            raise ValueError("repetitions must be at least two")
        require_non_negative_integer(self.seed, name="seed")

    def canonical_text(self) -> str:
        """Return the protocol as canonical JSON (sorted keys, no spaces).

        References:
            [REF-NIST-FIPS-180-4] (the text that is hashed). No equation is
            evaluated.

        """
        payload = {
            "aims": self.aims,
            "estimand": self.estimand,
            "measures": list(self.measures),
            "mechanisms": list(self.mechanisms),
            "methods": list(self.methods),
            "primary_comparison": self.primary_comparison,
            "repetitions": self.repetitions,
            "seed": self.seed,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def fingerprint(self) -> str:
        """Return the SHA-256 hash of the canonical protocol text.

        Returns:
            64 hexadecimal characters. Any change of the design changes it.

        References:
            [REF-NIST-FIPS-180-4] (SHA-256). No statistical equation is
            evaluated.

        """
        return hashlib.sha256(self.canonical_text().encode("utf-8")).hexdigest()


MEASURES = (
    "bias",
    "empirical_se",
    "mse",
    "model_se",
    "relative_error_model_se",
    "coverage",
    "bias_eliminated_coverage",
)


@dataclass(frozen=True, slots=True)
class Performance:
    """Performance measures of one method under one mechanism.

    Each measure is a pair ``(estimate, Monte Carlo standard error)``;
    measures a method cannot supply (no standard error or no interval
    reported) are ``(nan, nan)``.

    Args:
        mechanism: Mechanism label.
        method: Method label.
        truth: Estimand value under the mechanism.
        repetitions: Number of simulated data sets.
        failures: Repetitions in which the method raised or returned
            non-finite output.
        measures: Mapping from measure name (see :data:`MEASURES`) to
            ``(estimate, standard error)``.
        estimates: The successful point estimates, in repetition order with
            ``nan`` at failed repetitions (kept for paired comparisons).

    References:
        [REF-MORRIS-WHITE-CROWTHER-2019, Table 6], [EQ-MLAB-ADEMP-001] to
        [EQ-MLAB-ADEMP-006].

    """

    mechanism: str
    method: str
    truth: float
    repetitions: int
    failures: int
    measures: Mapping[str, tuple[float, float]]
    estimates: tuple[float, ...]

    def value(self, measure: str) -> float:
        """Return the estimate of one performance measure.

        Args:
            measure: Name from :data:`MEASURES`.

        Raises:
            KeyError: If the measure is unknown.

        References:
            [REF-MORRIS-WHITE-CROWTHER-2019, Table 6]. No equation is
            evaluated.

        """
        return self.measures[measure][0]

    def standard_error(self, measure: str) -> float:
        """Return the Monte Carlo standard error of one measure.

        Args:
            measure: Name from :data:`MEASURES`.

        Raises:
            KeyError: If the measure is unknown.

        References:
            [REF-MORRIS-WHITE-CROWTHER-2019, Table 6]. No equation is
            evaluated.

        """
        return self.measures[measure][1]


def _variance(values: Sequence[float]) -> float:
    """Return the unbiased sample variance.

    References:
        [EQ-STAT-002], [REF-CASELLA-BERGER-2002, Chapter 5].

    """
    count = len(values)
    mean = math.fsum(values) / count
    return math.fsum((v - mean) ** 2 for v in values) / (count - 1)


def performance_measures(
    estimates: Sequence[float],
    truth: float,
    *,
    standard_errors: Sequence[float] | None = None,
    intervals: Sequence[tuple[float, float]] | None = None,
) -> dict[str, tuple[float, float]]:
    """Compute performance measures with their Monte Carlo standard errors.

    With ``n`` estimates ``theta_i``, mean ``theta_bar``, and truth
    ``theta``:

    - bias ``theta_bar - theta`` with standard error ``S / sqrt(n)``
      ([EQ-MLAB-ADEMP-001]);
    - empirical standard error ``S`` with ``S / sqrt(2 (n - 1))``
      ([EQ-MLAB-ADEMP-002]; the standard error assumes normal estimates);
    - mean squared error ``mean (theta_i - theta)^2`` with the standard
      error of that mean ([EQ-MLAB-ADEMP-003]);
    - model standard error ``sqrt(mean se_i^2)`` with
      ``sqrt(Var(se^2) / (4 n ModSE^2))`` ([EQ-MLAB-ADEMP-004]);
    - relative error of the model standard error
      ``ModSE / EmpSE - 1`` with
      ``(ModSE / EmpSE) sqrt(Var(se^2) / (4 n ModSE^4) + 1 / (2 (n - 1)))``
      ([EQ-MLAB-ADEMP-005]);
    - coverage ``p`` with ``sqrt(p (1 - p) / n)`` and bias-eliminated
      coverage, which asks whether the interval contains ``theta_bar``
      instead of ``theta`` ([EQ-MLAB-ADEMP-006]).

    Args:
        estimates: At least two finite point estimates.
        truth: Estimand value.
        standard_errors: Model-based standard errors, one per estimate.
        intervals: Confidence intervals ``(lower, upper)``, one per
            estimate.

    Returns:
        Mapping from each name in :data:`MEASURES` to ``(estimate, standard
        error)``; unavailable measures are ``(nan, nan)``.

    Raises:
        ValueError: If fewer than two estimates are given or lengths differ.

    References:
        [REF-MORRIS-WHITE-CROWTHER-2019, Table 6], [EQ-MLAB-ADEMP-001] to
        [EQ-MLAB-ADEMP-006].

    """
    values = [require_finite_real(v, name="estimate") for v in estimates]
    target = require_finite_real(truth, name="truth")
    count = len(values)
    if count < 2:
        raise ValueError("at least two estimates are required")
    missing = (math.nan, math.nan)
    mean = math.fsum(values) / count
    empirical = math.sqrt(_variance(values))
    squared = [(v - target) ** 2 for v in values]
    mse = math.fsum(squared) / count
    result: dict[str, tuple[float, float]] = {
        "bias": (mean - target, empirical / math.sqrt(count)),
        "empirical_se": (empirical, empirical / math.sqrt(2.0 * (count - 1))),
        "mse": (mse, math.sqrt(_variance(squared) / count)),
        "model_se": missing,
        "relative_error_model_se": missing,
        "coverage": missing,
        "bias_eliminated_coverage": missing,
    }
    if standard_errors is not None:
        if len(standard_errors) != count:
            raise ValueError("standard_errors needs one entry per estimate")
        squares = [
            require_finite_real(s, name="standard error") ** 2 for s in standard_errors
        ]
        model = math.sqrt(math.fsum(squares) / count)
        spread = _variance(squares)
        if model > 0.0:
            result["model_se"] = (model, math.sqrt(spread / (4.0 * count * model**2)))
            if empirical > 0.0:
                ratio = model / empirical
                result["relative_error_model_se"] = (
                    ratio - 1.0,
                    ratio
                    * math.sqrt(
                        spread / (4.0 * count * model**4) + 1.0 / (2.0 * (count - 1))
                    ),
                )
    if intervals is not None:
        if len(intervals) != count:
            raise ValueError("intervals needs one entry per estimate")
        hits = sum(1 for low, high in intervals if low <= target <= high)
        centred = sum(1 for low, high in intervals if low <= mean <= high)
        for label, number in (
            ("coverage", hits),
            ("bias_eliminated_coverage", centred),
        ):
            share = number / count
            result[label] = (share, math.sqrt(share * (1.0 - share) / count))
    return result


@dataclass(frozen=True, slots=True)
class PrecisionComparison:
    """Relative precision of one method against a reference, paired.

    Args:
        mechanism: Mechanism label.
        method: Method whose precision is assessed.
        reference: Reference method.
        relative_gain: ``(EmpSE_ref / EmpSE_method)^2 - 1``; positive when
            the method is more precise.
        standard_error: Monte Carlo standard error of ``relative_gain``.
        correlation: Correlation of the paired estimates.
        pairs: Number of repetitions in which both methods succeeded.

    References:
        [REF-MORRIS-WHITE-CROWTHER-2019, Table 6] (relative % increase in
        precision), [EQ-MLAB-ADEMP-007].

    """

    mechanism: str
    method: str
    reference: str
    relative_gain: float
    standard_error: float
    correlation: float
    pairs: int


@dataclass(frozen=True, slots=True)
class SimulationStudyResult:
    """Performance of every method under every mechanism.

    Args:
        protocol: The executed protocol.
        fingerprint: Fingerprint of ``protocol`` at execution time.
        performances: One :class:`Performance` per (mechanism, method).

    References:
        [REF-MORRIS-WHITE-CROWTHER-2019] (reporting).

    """

    protocol: StudyProtocol
    fingerprint: str
    performances: tuple[Performance, ...]

    def performance(self, mechanism: str, method: str) -> Performance:
        """Return the performance of one method under one mechanism.

        Raises:
            KeyError: If the pair was not part of the study.

        References:
            [REF-MORRIS-WHITE-CROWTHER-2019, Table 6]. No equation is
            evaluated.

        """
        for record in self.performances:
            if record.mechanism == mechanism and record.method == method:
                return record
        raise KeyError((mechanism, method))

    def compare_precision(
        self, mechanism: str, method: str, reference: str
    ) -> PrecisionComparison:
        """Return the paired relative precision of a method.

        The relative gain ``(EmpSE_ref / EmpSE)^2 - 1`` has Monte Carlo
        standard error
        ``2 (EmpSE_ref / EmpSE)^2 sqrt((1 - rho^2) / (n - 1))`` with ``rho``
        the correlation of the paired estimates ([EQ-MLAB-ADEMP-007]);
        common data sets make ``rho`` large and the comparison sharp. The
        formula assumes jointly normal estimates, as do the standard errors
        of the empirical standard error and of the relative error of the
        model standard error: in 1500 simulated studies of 200 repetitions
        the reported standard errors matched the spread of the measures
        within 3% for normal data, while for the mean and median of 15
        exponential observations they were 12% (relative precision) and 8%
        (empirical standard error) too small.

        Args:
            mechanism: Mechanism label.
            method: Method to assess.
            reference: Reference method.

        Returns:
            The comparison over repetitions in which both succeeded.

        Raises:
            KeyError: If a label is unknown.
            ValueError: If fewer than three pairs are available or a
                method has zero empirical variance.

        References:
            [REF-MORRIS-WHITE-CROWTHER-2019, Table 6],
            [EQ-MLAB-ADEMP-007].

        """
        first = self.performance(mechanism, method).estimates
        second = self.performance(mechanism, reference).estimates
        pairs = [
            (a, b)
            for a, b in zip(first, second, strict=True)
            if not (math.isnan(a) or math.isnan(b))
        ]
        count = len(pairs)
        if count < 3:
            raise ValueError("at least three paired repetitions are required")
        a_values = [a for a, _ in pairs]
        b_values = [b for _, b in pairs]
        var_a, var_b = _variance(a_values), _variance(b_values)
        if var_a <= 0.0 or var_b <= 0.0:
            raise ValueError("both methods need a positive empirical variance")
        mean_a = math.fsum(a_values) / count
        mean_b = math.fsum(b_values) / count
        covariance = math.fsum((a - mean_a) * (b - mean_b) for a, b in pairs) / (
            count - 1
        )
        rho = covariance / math.sqrt(var_a * var_b)
        ratio = var_b / var_a
        error = 2.0 * ratio * math.sqrt(max(1.0 - rho * rho, 0.0) / (count - 1))
        return PrecisionComparison(
            mechanism, method, reference, ratio - 1.0, error, rho, count
        )

    def render(self) -> str:
        """Return the results as a Markdown report.

        Returns:
            The protocol with its fingerprint and one table per mechanism;
            every cell shows ``estimate (Monte Carlo standard error)``.

        References:
            [REF-MORRIS-WHITE-CROWTHER-2019], project reporting
            policy [REF-PROBSTAT-REPORTING].

        """
        protocol = self.protocol
        lines = [
            "## Simulation study",
            "",
            f"- Aims: {protocol.aims}",
            f"- Estimand: {protocol.estimand}",
            f"- Primary comparison: {protocol.primary_comparison}",
            f"- Repetitions: {protocol.repetitions}; seed: {protocol.seed}",
            f"- Protocol fingerprint (SHA-256): `{self.fingerprint}`",
        ]
        header = "| Method | Failures | " + " | ".join(protocol.measures) + " |"
        rule = "|---|---|" + "---|" * len(protocol.measures)
        for mechanism in protocol.mechanisms:
            truth = self.performance(mechanism, protocol.methods[0]).truth
            lines.extend(["", f"### {mechanism} (truth {truth:.6g})", "", header, rule])
            for method in protocol.methods:
                record = self.performance(mechanism, method)
                cells = [
                    f"{record.value(m):.4g} ({record.standard_error(m):.2g})"
                    for m in protocol.measures
                ]
                lines.append(
                    f"| {method} | {record.failures} | " + " | ".join(cells) + " |"
                )
        lines.extend(
            [
                "",
                "Cells show the estimate with its Monte Carlo standard error. "
                "Measures are computed over the repetitions in which the method "
                "succeeded; the evidence covers the listed mechanisms only.",
            ]
        )
        return "\n".join(lines)


def _safe_fit(method: Method[D], data: D) -> MethodOutput | None:
    """Return a method's output, or ``None`` if it failed.

    References:
        [REF-MORRIS-WHITE-CROWTHER-2019] (handling of
        non-convergence). No equation is evaluated.

    """
    try:
        output = method.fit(data)
    except Exception:  # noqa: BLE001 - any failure of a method is recorded
        return None
    if not isinstance(output, MethodOutput):
        return None
    numbers = [output.estimate, output.standard_error, output.lower, output.upper]
    for number in numbers:
        if number is not None and not math.isfinite(number):
            return None
    return output


def run_simulation_study(
    protocol: StudyProtocol,
    mechanisms: Sequence[DataGeneratingMechanism[D]],
    methods: Sequence[Method[D]],
    *,
    expected_fingerprint: str | None = None,
) -> SimulationStudyResult:
    """Execute a pre-specified simulation study.

    Repetition ``i`` of mechanism ``m`` generates one data set from the
    child stream ``(m, i)`` of the protocol seed, and every method analyses
    that same data set.

    Args:
        protocol: The pre-specified design. The names of ``mechanisms`` and
            ``methods`` must equal the protocol's, in order.
        mechanisms: The data-generating mechanisms.
        methods: The compared methods.
        expected_fingerprint: Fingerprint published before the run. When
            given and different from the protocol's fingerprint the study
            does not run.

    Returns:
        Performance measures with Monte Carlo standard errors.

    Raises:
        ValueError: If names do not match the protocol, a protocol measure
            is unknown, the fingerprint differs, or a method succeeds in
            fewer than two repetitions.

    References:
        [REF-MORRIS-WHITE-CROWTHER-2019] (ADEMP and Table 6),
        [REF-PAWEL-KOOK-REEVE-2024], [EQ-MLAB-ADEMP-001] to
        [EQ-MLAB-ADEMP-006].

    """
    if tuple(m.name for m in mechanisms) != protocol.mechanisms:
        raise ValueError("mechanism names must equal the protocol's, in order")
    if tuple(m.name for m in methods) != protocol.methods:
        raise ValueError("method names must equal the protocol's, in order")
    unknown = [m for m in protocol.measures if m not in MEASURES]
    if unknown:
        raise ValueError(f"unknown performance measures: {unknown}")
    fingerprint = protocol.fingerprint()
    if expected_fingerprint is not None and expected_fingerprint != fingerprint:
        raise ValueError(
            "protocol fingerprint differs from the pre-registered one: "
            f"{fingerprint} != {expected_fingerprint}"
        )
    root: RandomStream = resolve_stream(protocol.seed)
    performances: list[Performance] = []
    for mechanism in mechanisms:
        truth = require_finite_real(mechanism.truth, name="truth")
        outputs: list[list[MethodOutput | None]] = [[] for _ in methods]
        for index in range(protocol.repetitions):
            generator = root.child(mechanism.name, index).generator()
            data = mechanism.generate(generator)
            for position, method in enumerate(methods):
                outputs[position].append(_safe_fit(method, data))
        for method, results in zip(methods, outputs, strict=True):
            successes = [r for r in results if r is not None]
            if len(successes) < 2:
                raise ValueError(
                    f"method {method.name!r} succeeded in fewer than two "
                    f"repetitions under {mechanism.name!r}"
                )
            errors = [r.standard_error for r in successes]
            limits = [(r.lower, r.upper) for r in successes]
            has_errors = all(e is not None for e in errors)
            has_limits = all(a is not None and b is not None for a, b in limits)
            measures = performance_measures(
                [r.estimate for r in successes],
                truth,
                standard_errors=[float(e) for e in errors if e is not None]
                if has_errors
                else None,
                intervals=[
                    (float(a), float(b))
                    for a, b in limits
                    if a is not None and b is not None
                ]
                if has_limits
                else None,
            )
            performances.append(
                Performance(
                    mechanism.name,
                    method.name,
                    truth,
                    protocol.repetitions,
                    len(results) - len(successes),
                    MappingProxyType(measures),
                    tuple(math.nan if r is None else r.estimate for r in results),
                )
            )
    return SimulationStudyResult(protocol, fingerprint, tuple(performances))


def repetitions_for_standard_error(
    measure: str, target_standard_error: float, *, pilot: float
) -> int:
    """Return the repetitions needed for a target Monte Carlo standard error.

    - ``"bias"``: ``n = pilot / target^2`` with ``pilot`` the variance of
      the estimates ([EQ-MLAB-ADEMP-001] solved for ``n``);
    - ``"coverage"``: ``n = p (1 - p) / target^2`` with ``pilot`` the
      anticipated coverage ``p`` ([EQ-MLAB-ADEMP-006] solved for ``n``);
      ``p = 0.5`` is the worst case.

    Args:
        measure: ``"bias"`` or ``"coverage"``.
        target_standard_error: Positive target.
        pilot: Variance of the estimates (bias) or anticipated coverage.

    Returns:
        The smallest integer number of repetitions meeting the target.

    Raises:
        ValueError: If the measure is unsupported or an argument is out of
            range.

    References:
        [REF-MORRIS-WHITE-CROWTHER-2019] (choosing the number
        of repetitions), [EQ-MLAB-ADEMP-001], [EQ-MLAB-ADEMP-006].

    """
    target = require_finite_real(target_standard_error, name="target_standard_error")
    value = require_finite_real(pilot, name="pilot")
    if target <= 0.0:
        raise ValueError("target_standard_error must be positive")
    if measure == "bias":
        if value <= 0.0:
            raise ValueError("pilot variance must be positive")
        return max(2, math.ceil(value / target**2 - 1e-9))
    if measure == "coverage":
        if not 0.0 < value < 1.0:
            raise ValueError("anticipated coverage must lie in (0, 1)")
        return max(2, math.ceil(value * (1.0 - value) / target**2 - 1e-9))
    raise ValueError("measure must be 'bias' or 'coverage'")
