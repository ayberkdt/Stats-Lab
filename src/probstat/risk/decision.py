"""Decision analysis under uncertainty and a risk-matrix classifier.

:class:`DecisionProblem` holds a finite set of decisions, uncertain states with
prior probabilities, and a payoff table (utilities to maximize or losses to
minimize). It computes the expected-value table, the Bayes decision, the
expected value of perfect information (EVPI), and, through preposterior
analysis of a discrete experiment, the expected value of sample information
(EVSI).

:class:`RiskMatrix` is a communication tool only. Cox (2008) shows that risk
matrices can assign identical ratings to quantitatively very different risks,
can rank risks in reverse order, and depend on subjective bin edges; never use
a matrix category in place of the quantitative measures in this package.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from probstat.foundations import require_finite_real, require_probability

_PROBABILITY_SUM_TOLERANCE = 1e-9


def _labels(values: Iterable[object], *, name: str) -> tuple[str, ...]:
    """Validate non-empty distinct text labels.

    Args:
        values: Candidate labels.
        name: Field name for error messages.

    Returns:
        Labels as a tuple.

    Raises:
        TypeError: If a label is not text.
        ValueError: If labels are empty, blank, or repeated.

    References:
        Project validation policy. No equation is evaluated.

    """
    labels = tuple(values)
    if not labels:
        raise ValueError(f"{name} must not be empty")
    for label in labels:
        if not isinstance(label, str):
            raise TypeError(f"{name} must be strings")
        if not label.strip():
            raise ValueError(f"{name} must not be blank")
    if len(set(labels)) != len(labels):
        raise ValueError(f"{name} must be distinct")
    return labels  # type: ignore[return-value]


def _distribution(
    values: Iterable[object], size: int, *, name: str
) -> tuple[float, ...]:
    """Validate a probability vector of a given length.

    Args:
        values: Candidate probabilities.
        size: Required length.
        name: Field name for error messages.

    Returns:
        Probabilities as floats.

    Raises:
        ValueError: If the length is wrong or the sum differs from one by more
            than ``1e-9``.

    References:
        [EQ-PROB-001], [REF-KOLMOGOROV-1956].

    """
    probabilities = tuple(require_probability(p, name=name) for p in values)
    if len(probabilities) != size:
        raise ValueError(f"{name} must have {size} entries")
    if abs(math.fsum(probabilities) - 1.0) > _PROBABILITY_SUM_TOLERANCE:
        raise ValueError(f"{name} must sum to one")
    return probabilities


@dataclass(frozen=True, slots=True)
class PreposteriorAnalysis:
    """Preposterior analysis of a discrete experiment.

    Args:
        outcomes: Experiment outcome labels.
        outcome_probabilities: Predictive ``P(o) = sum_s P(o | s) p_s``.
        posteriors: Read-only outcome-to-posterior state probabilities.
        decision_rule: Read-only outcome-to-Bayes-decision mapping.
        prior_value: Expected value of the best decision without data.
        value_with_sample_information: ``sum_o P(o) best_d E[u | o]``.
        evsi: Expected value of sample information (non-negative).
        evpi: Expected value of perfect information (an upper bound on EVSI).

    References:
        [REF-RAIFFA-SCHLAIFER-1961, Chapters 1 and 4].

    """

    outcomes: tuple[str, ...]
    outcome_probabilities: Mapping[str, float]
    posteriors: Mapping[str, tuple[float, ...]]
    decision_rule: Mapping[str, str]
    prior_value: float
    value_with_sample_information: float
    evsi: float
    evpi: float


@dataclass(frozen=True)
class DecisionProblem:
    """Finite decision problem with uncertain states.

    Args:
        decisions: Distinct decision labels (rows of ``payoffs``).
        states: Distinct state labels (columns of ``payoffs``).
        probabilities: Prior state probabilities summing to one.
        payoffs: Finite ``payoffs[d][s]``; utilities when ``maximize`` is
            true, losses otherwise.
        maximize: Whether larger payoffs are better.

    References:
        [REF-RAIFFA-SCHLAIFER-1961, Chapter 1].

    """

    decisions: tuple[str, ...]
    states: tuple[str, ...]
    probabilities: tuple[float, ...]
    payoffs: tuple[tuple[float, ...], ...]
    maximize: bool = True

    def __post_init__(self) -> None:
        """Validate labels, probabilities, and the payoff table.

        Raises:
            TypeError: If labels are not text.
            ValueError: If shapes disagree, probabilities are invalid, or a
                payoff is non-finite.

        References:
            [EQ-PROB-001], [REF-RAIFFA-SCHLAIFER-1961].

        """
        decisions = _labels(self.decisions, name="decisions")
        states = _labels(self.states, name="states")
        probabilities = _distribution(
            self.probabilities, len(states), name="probabilities"
        )
        rows = tuple(tuple(row) for row in self.payoffs)
        if len(rows) != len(decisions):
            raise ValueError("payoffs must have one row per decision")
        table = tuple(
            tuple(require_finite_real(v, name="payoff") for v in row) for row in rows
        )
        if any(len(row) != len(states) for row in table):
            raise ValueError("payoffs must have one column per state")
        object.__setattr__(self, "decisions", decisions)
        object.__setattr__(self, "states", states)
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(self, "payoffs", table)

    def _best(self, values: Sequence[float]) -> int:
        """Return the index of the best value (first on ties).

        Args:
            values: Candidate values aligned with decisions.

        Returns:
            Index of the maximum (or minimum for losses).

        References:
            [REF-RAIFFA-SCHLAIFER-1961, Chapter 1]. No equation is evaluated.

        """
        target = max(values) if self.maximize else min(values)
        return list(values).index(target)

    def _expected(self, probabilities: Sequence[float]) -> tuple[float, ...]:
        """Return expected payoffs of every decision under given state weights.

        Args:
            probabilities: State probabilities.

        Returns:
            Expected payoffs aligned with decisions.

        References:
            [EQ-RISK-DEC-EU-001], [REF-RAIFFA-SCHLAIFER-1961].

        """
        return tuple(
            math.fsum(p * u for p, u in zip(probabilities, row, strict=True))
            for row in self.payoffs
        )

    def expected_values(self) -> Mapping[str, float]:
        """Return the expected payoff of every decision under the prior.

        Returns:
            Read-only decision-to-expected-payoff mapping.

        Equation:
            [EQ-RISK-DEC-EU-001].

        References:
            [REF-RAIFFA-SCHLAIFER-1961, Chapter 1].

        """
        return MappingProxyType(
            dict(zip(self.decisions, self._expected(self.probabilities), strict=True))
        )

    def optimal_decision(self) -> str:
        """Return the Bayes decision (first in declaration order on ties).

        Equation:
            [EQ-RISK-DEC-EU-001].

        References:
            [REF-RAIFFA-SCHLAIFER-1961, Chapter 1].

        """
        return self.decisions[self._best(self._expected(self.probabilities))]

    def optimal_value(self) -> float:
        """Return the expected payoff of the Bayes decision.

        Equation:
            [EQ-RISK-DEC-EU-001].

        References:
            [REF-RAIFFA-SCHLAIFER-1961, Chapter 1].

        """
        values = self._expected(self.probabilities)
        return values[self._best(values)]

    def value_with_perfect_information(self) -> float:
        """Return ``sum_s p_s best_d u(d, s)``.

        Equation:
            [EQ-RISK-DEC-EVPI-001].

        References:
            [REF-RAIFFA-SCHLAIFER-1961, Chapter 4].

        """
        columns = list(zip(*self.payoffs, strict=True))
        pick = max if self.maximize else min
        return math.fsum(
            p * pick(column)
            for p, column in zip(self.probabilities, columns, strict=True)
        )

    def expected_value_of_perfect_information(self) -> float:
        """Return EVPI, the most a rational decision maker pays for certainty.

        Returns:
            Non-negative EVPI in payoff units.

        Equation:
            [EQ-RISK-DEC-EVPI-001].

        References:
            [REF-RAIFFA-SCHLAIFER-1961, Chapter 4].

        """
        gap = self.value_with_perfect_information() - self.optimal_value()
        return max(0.0, gap if self.maximize else -gap)

    def preposterior(
        self,
        outcomes: Sequence[str],
        likelihoods: Sequence[Sequence[float]],
    ) -> PreposteriorAnalysis:
        """Return EVSI of a discrete experiment by preposterior analysis.

        Args:
            outcomes: Distinct experiment outcome labels.
            likelihoods: ``likelihoods[s][o] = P(o | s)``; each state row is a
                probability vector over outcomes.

        Returns:
            Preposterior analysis. Outcomes with zero predictive probability
            keep the prior and the prior decision; they do not contribute.

        Raises:
            ValueError: If the likelihood table is invalid.

        Equation:
            [EQ-RISK-DEC-EVSI-001], [EQ-PROB-008].

        References:
            [REF-RAIFFA-SCHLAIFER-1961, Chapters 1 and 4].

        """
        labels = _labels(outcomes, name="outcomes")
        rows = tuple(likelihoods)
        if len(rows) != len(self.states):
            raise ValueError("likelihoods must have one row per state")
        table = tuple(
            _distribution(row, len(labels), name="likelihood row") for row in rows
        )
        prior_value = self.optimal_value()
        predictive: dict[str, float] = {}
        posteriors: dict[str, tuple[float, ...]] = {}
        rule: dict[str, str] = {}
        contributions: list[float] = []
        for index, label in enumerate(labels):
            joint = tuple(p * table[s][index] for s, p in enumerate(self.probabilities))
            marginal = math.fsum(joint)
            predictive[label] = marginal
            if marginal <= 0.0:
                posteriors[label] = self.probabilities
                rule[label] = self.optimal_decision()
                continue
            posterior = tuple(value / marginal for value in joint)
            values = self._expected(posterior)
            best = self._best(values)
            posteriors[label] = posterior
            rule[label] = self.decisions[best]
            contributions.append(marginal * values[best])
        with_information = math.fsum(contributions)
        gap = with_information - prior_value
        return PreposteriorAnalysis(
            labels,
            MappingProxyType(predictive),
            MappingProxyType(posteriors),
            MappingProxyType(rule),
            prior_value,
            with_information,
            max(0.0, gap if self.maximize else -gap),
            self.expected_value_of_perfect_information(),
        )


@dataclass(frozen=True)
class RiskMatrix:
    """Likelihood-by-severity lookup table for qualitative risk communication.

    A value equal to a threshold falls into the higher bin. The matrix is a
    reporting aid with known flaws (Cox 2008): coarse bins hide order-of-
    magnitude differences, ratings can reverse quantitative rankings, and the
    result depends on arbitrary edges. Do not use it for decisions that
    quantitative risk measures can support.

    Args:
        likelihood_thresholds: Strictly increasing interior bin edges for
            likelihood (``len = rows - 1``).
        severity_thresholds: Strictly increasing interior bin edges for
            severity (``len = columns - 1``).
        categories: ``categories[likelihood_bin][severity_bin]`` labels.

    References:
        [REF-COX-2008].

    """

    likelihood_thresholds: tuple[float, ...]
    severity_thresholds: tuple[float, ...]
    categories: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        """Validate edges and the category table shape.

        Raises:
            ValueError: If edges do not strictly increase or the table shape
                does not match the number of bins.

        References:
            [REF-COX-2008]. No equation is evaluated.

        """
        likelihood = tuple(
            require_finite_real(v, name="likelihood threshold")
            for v in self.likelihood_thresholds
        )
        severity = tuple(
            require_finite_real(v, name="severity threshold")
            for v in self.severity_thresholds
        )
        for edges, name in ((likelihood, "likelihood"), (severity, "severity")):
            if any(b <= a for a, b in zip(edges, edges[1:], strict=False)):
                raise ValueError(f"{name} thresholds must strictly increase")
        table = tuple(tuple(row) for row in self.categories)
        if len(table) != len(likelihood) + 1 or any(
            len(row) != len(severity) + 1 for row in table
        ):
            raise ValueError(
                "categories shape must be (likelihood bins, severity bins)"
            )
        for row in table:
            _labels_allow_repeats(row)
        object.__setattr__(self, "likelihood_thresholds", likelihood)
        object.__setattr__(self, "severity_thresholds", severity)
        object.__setattr__(self, "categories", table)

    def cell(self, likelihood: float, severity: float) -> tuple[int, int]:
        """Return the zero-based ``(likelihood_bin, severity_bin)`` indices.

        Args:
            likelihood: Finite likelihood score or frequency.
            severity: Finite severity score or consequence.

        Returns:
            Bin indices.

        Equation:
            [EQ-RISK-MATRIX-001].

        References:
            [REF-COX-2008].

        """
        row = bisect.bisect_right(
            self.likelihood_thresholds,
            require_finite_real(likelihood, name="likelihood"),
        )
        column = bisect.bisect_right(
            self.severity_thresholds, require_finite_real(severity, name="severity")
        )
        return row, column

    def classify(self, likelihood: float, severity: float) -> str:
        """Return the category label of one risk.

        Args:
            likelihood: Finite likelihood score or frequency.
            severity: Finite severity score or consequence.

        Returns:
            Category label.

        Equation:
            [EQ-RISK-MATRIX-001].

        References:
            [REF-COX-2008].

        """
        row, column = self.cell(likelihood, severity)
        return self.categories[row][column]


def _labels_allow_repeats(values: Sequence[object]) -> None:
    """Validate that every category label is non-blank text.

    Args:
        values: Candidate labels; repeats are allowed.

    Raises:
        TypeError: If a label is not text.
        ValueError: If a label is blank.

    References:
        Project validation policy. No equation is evaluated.

    """
    for value in values:
        if not isinstance(value, str):
            raise TypeError("categories must be strings")
        if not value.strip():
            raise ValueError("categories must not be blank")
