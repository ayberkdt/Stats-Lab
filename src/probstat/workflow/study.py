"""A single study object that runs every UQ analysis on one model.

``UQStudy`` is the integration layer: it owns one
:class:`~probstat.modeling.ProbabilisticModel`, one root random stream, and a
log of every analysis performed. Each stage draws its randomness from a named
child stream, so the whole study is reproducible from one seed, and every
stage records the forward-model evaluations it charged. The study does not
re-implement any method; it calls the domain packages and keeps their results.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.modeling import ProbabilisticModel
from probstat.propagation import PropagationStudy, propagate
from probstat.reliability import (
    FORMResult,
    MethodComparison,
    ModelLimitState,
    compare_reliability_methods,
    form_hlrf,
)
from probstat.risk import LossFunction, RiskAssessment, assess_risk
from probstat.sensitivity import (
    ModelSobolResult,
    MorrisResult,
    morris_screening,
    sobol_analysis,
)


@dataclass(frozen=True, slots=True)
class StageRecord:
    """One completed analysis stage of a study.

    Args:
        stage: Stage kind such as ``"propagation"`` or ``"reliability"``.
        label: Human-readable description of the stage.
        summary: Headline numbers of the stage.
        evaluations: Forward-model evaluations charged to the stage.
        seed: Seed of the stream the stage used.
        result: The full domain result object.

    References:
        [REF-SMITH-UQ-2024, Chapter 1].

    """

    stage: str
    label: str
    summary: Mapping[str, float | str]
    evaluations: int
    seed: int
    result: Any = field(repr=False, compare=False)


class UQStudy:
    """Run propagation, sensitivity, reliability, and risk on one model.

    Args:
        model: The probabilistic model under study.
        seed: Root seed or stream for every stochastic stage.
        name: Study label (defaults to the model name).

    References:
        [REF-SMITH-UQ-2024, Chapters 1-5], [REF-SUDRET-2007, Chapter 1].

    """

    def __init__(
        self,
        model: ProbabilisticModel,
        *,
        seed: int | RandomStream = 0,
        name: str | None = None,
    ) -> None:
        """Bind the model and root stream.

        Raises:
            TypeError: If ``model`` is not a ProbabilisticModel.

        References:
            [REF-SMITH-UQ-2024, Chapter 1]. No equation is evaluated.

        """
        if not isinstance(model, ProbabilisticModel):
            raise TypeError("model must be a ProbabilisticModel")
        self._model = model
        self._root = resolve_stream(seed)
        self._name = name or model.name
        self._records: list[StageRecord] = []

    @property
    def model(self) -> ProbabilisticModel:
        """Return the model under study.

        References:
            [REF-SMITH-UQ-2024, Chapter 1].

        """
        return self._model

    @property
    def name(self) -> str:
        """Return the study label.

        References:
            [REF-SMITH-UQ-2024, Chapter 1].

        """
        return self._name

    @property
    def records(self) -> tuple[StageRecord, ...]:
        """Return completed stages in execution order.

        References:
            [REF-SMITH-UQ-2024, Chapter 1].

        """
        return tuple(self._records)

    @property
    def total_evaluations(self) -> int:
        """Return the forward-model evaluations charged by all stages.

        References:
            [REF-SMITH-UQ-2024, Chapter 3].

        """
        return sum(record.evaluations for record in self._records)

    def latest(self, stage: str) -> StageRecord:
        """Return the most recent record of a stage kind.

        Args:
            stage: Stage kind.

        Returns:
            Latest matching record.

        Raises:
            KeyError: If the stage has not been run.

        References:
            [REF-SMITH-UQ-2024, Chapter 1]. No equation is evaluated.

        """
        for record in reversed(self._records):
            if record.stage == stage:
                return record
        raise KeyError(f"stage {stage!r} has not been run")

    def stream(self, stage: str) -> RandomStream:
        """Return the child stream for the next run of a stage.

        Args:
            stage: Stage kind.

        Returns:
            Child stream ``root.child(stage, k)`` for the k-th run of the stage.

        Equation:
            [EQ-RNG-SPLITMIX-001].

        References:
            [REF-STEELE-LEA-FLOOD-2014].

        """
        count = sum(record.stage == stage for record in self._records)
        return self._root.child(stage, count)

    def _record(
        self,
        stage: str,
        label: str,
        summary: Mapping[str, float | str],
        evaluations: int,
        stream: RandomStream,
        result: Any,
    ) -> None:
        """Append one stage record.

        References:
            [REF-SMITH-UQ-2024, Chapter 1]. No equation is evaluated.

        """
        self._records.append(
            StageRecord(
                stage,
                label,
                MappingProxyType(dict(summary)),
                evaluations,
                stream.seed,
                result,
            )
        )

    def propagate(self, size: int = 4096, *, design: str = "sobol") -> PropagationStudy:
        """Propagate the joint input law through every QoI.

        Args:
            size: Number of input realizations.
            design: Sampling design name.

        Returns:
            Propagation study.

        Equation:
            [EQ-UQ-001], [EQ-MC-MEAN-001].

        References:
            [REF-SMITH-UQ-2024, Chapters 3-4].

        """
        stream = self.stream("propagation")
        study = propagate(self._model, size, design=design, seed=stream)
        summary: dict[str, float | str] = {"design": design, "size": size}
        for name, output in study.outputs.items():
            summary[f"{name}.mean"] = output.mean
            summary[f"{name}.std"] = output.standard_deviation
            summary[f"{name}.q05"] = output.quantile(0.05)
            summary[f"{name}.q95"] = output.quantile(0.95)
        self._record(
            "propagation",
            f"{design} propagation, {size} samples",
            summary,
            study.model_evaluations,
            stream,
            study,
        )
        return study

    def sensitivity(
        self,
        quantity: str,
        size: int = 1024,
        *,
        design: str = "sobol",
        bootstrap: int = 200,
    ) -> ModelSobolResult:
        """Estimate Sobol indices of one QoI with bootstrap intervals.

        Args:
            quantity: QoI name.
            size: Design rows (cost ``size (d + 2)``).
            design: Unit-hypercube design name.
            bootstrap: Bootstrap replicates.

        Returns:
            Sobol result.

        Equation:
            [EQ-SENS-FIRST-001], [EQ-SENS-TOTAL-001], [EQ-SENS-BOOT-001].

        References:
            [REF-SALTELLI-2010].

        """
        stream = self.stream("sensitivity")
        result = sobol_analysis(
            self._model, quantity, size, design=design, seed=stream, bootstrap=bootstrap
        )
        summary: dict[str, float | str] = {"quantity": quantity}
        for name, first, total in zip(
            result.input_names, result.first_order, result.total_effect, strict=True
        ):
            summary[f"S[{name}]"] = first.estimate
            summary[f"ST[{name}]"] = total.estimate
        self._record(
            "sensitivity",
            f"Sobol indices of {quantity}",
            summary,
            result.evaluations,
            stream,
            result,
        )
        return result

    def screening(self, quantity: str, trajectories: int = 20) -> MorrisResult:
        """Screen inputs of one QoI with Morris elementary effects.

        Args:
            quantity: QoI name.
            trajectories: Number of trajectories (cost ``r (d + 1)``).

        Returns:
            Morris result.

        Equation:
            [EQ-SENS-MORRIS-001].

        References:
            [REF-MORRIS-1991].

        """
        stream = self.stream("screening")
        result = morris_screening(self._model, quantity, trajectories, seed=stream)
        summary: dict[str, float | str] = {"quantity": quantity}
        for name, value in zip(result.input_names, result.mu_star, strict=True):
            summary[f"mu*[{name}]"] = value
        self._record(
            "screening",
            f"Morris screening of {quantity}",
            summary,
            result.evaluations,
            stream,
            result,
        )
        return result

    def limit_state(
        self,
        quantity: str,
        threshold: float,
        *,
        failure: Literal["above", "below"] = "below",
    ) -> ModelLimitState:
        """Return a limit state on one QoI of the study model.

        Args:
            quantity: QoI name.
            threshold: Failure threshold.
            failure: ``"below"`` (low values fail) or ``"above"``.

        Returns:
            Model limit state with ``g <= 0`` meaning failure.

        Equation:
            [EQ-REL-LIMIT-001].

        References:
            [REF-MELCHERS-BECK-2018, Chapter 3].

        """
        return ModelLimitState(self._model, quantity, threshold, failure=failure)

    def reliability(
        self,
        quantity: str,
        threshold: float,
        *,
        failure: Literal["above", "below"] = "below",
        methods: Sequence[str] = ("form", "sorm", "is", "subset", "mc"),
        monte_carlo_samples: int = 10_000,
    ) -> MethodComparison:
        """Compare reliability methods on one QoI threshold.

        Args:
            quantity: QoI name.
            threshold: Failure threshold.
            failure: ``"below"`` or ``"above"``.
            methods: Method subset.
            monte_carlo_samples: Crude Monte Carlo sample size.

        Returns:
            Method comparison (also stored with the FORM design point).

        Equation:
            [EQ-REL-PF-001].

        References:
            [REF-MELCHERS-BECK-2018, Chapters 3-5].

        """
        stream = self.stream("reliability")
        state = self.limit_state(quantity, threshold, failure=failure)
        before = self._model.evaluations
        comparison = compare_reliability_methods(
            state,
            methods=methods,
            monte_carlo_samples=monte_carlo_samples,
            seed=stream,
        )
        summary: dict[str, float | str] = {
            "quantity": quantity,
            "threshold": threshold,
            "failure": failure,
        }
        for row in comparison.rows:
            summary[f"pf[{row.method}]"] = row.probability
        self._record(
            "reliability",
            f"P({quantity} {'<=' if failure == 'below' else '>='} {threshold:g})",
            summary,
            self._model.evaluations - before,
            stream,
            comparison,
        )
        return comparison

    def form(
        self,
        quantity: str,
        threshold: float,
        *,
        failure: Literal["above", "below"] = "below",
    ) -> FORMResult:
        """Run FORM alone and record design point and importance factors.

        Args:
            quantity: QoI name.
            threshold: Failure threshold.
            failure: ``"below"`` or ``"above"``.

        Returns:
            FORM result.

        Equation:
            [EQ-REL-FORM-001].

        References:
            [REF-MELCHERS-BECK-2018, Chapter 5].

        """
        stream = self.stream("form")
        result = form_hlrf(self.limit_state(quantity, threshold, failure=failure))
        summary: dict[str, float | str] = {
            "beta": result.beta,
            "pf": result.probability,
        }
        for name, value in result.importance_factors.items():
            summary[f"alpha2[{name}]"] = value
        self._record(
            "form",
            f"FORM for {quantity} threshold {threshold:g}",
            summary,
            result.evaluations,
            stream,
            result,
        )
        return result

    def risk(
        self,
        quantity: str,
        loss: LossFunction | Callable[[float], float],
        size: int = 20_000,
        *,
        design: str = "lhs",
        levels: Sequence[float] = (0.95, 0.99),
    ) -> RiskAssessment:
        """Assess the loss distribution induced by one QoI.

        Args:
            quantity: QoI name.
            loss: Loss function of the QoI value.
            size: Propagation sample size.
            design: Sampling design name.
            levels: VaR/CVaR confidence levels.

        Returns:
            Risk assessment.

        Equation:
            [EQ-UQ-001].

        References:
            [REF-ROCKAFELLAR-URYASEV-2002], [REF-KAPLAN-GARRICK-1981].

        """
        stream = self.stream("risk")
        result = assess_risk(
            self._model, quantity, loss, size, design=design, seed=stream, levels=levels
        )
        summary: dict[str, float | str] = {
            "expected_loss": result.expected_loss.estimate,
            "P(loss>0)": result.probability_of_loss,
        }
        for level in levels:
            summary[f"VaR{level:g}"] = result.value_at_risk[level]
            summary[f"CVaR{level:g}"] = result.conditional_value_at_risk[level]
        self._record(
            "risk",
            f"loss risk of {quantity}",
            summary,
            result.model_evaluations,
            stream,
            result,
        )
        return result

    def report(self) -> str:
        """Return a Markdown report of every stage with costs and seeds.

        Returns:
            Markdown text: one section per stage with a two-column table.

        References:
            [REF-SMITH-UQ-2024, Chapter 1]. No equation is evaluated.

        """
        dependence = (
            "independent" if self._model.inputs.is_independent else "Gaussian copula"
        )
        lines = [
            f"# UQ study: {self._name}",
            "",
            f"- Inputs: {', '.join(self._model.inputs.names)}",
            f"- Quantities of interest: {', '.join(self._model.quantity_names)}",
            f"- Dependence: {dependence}",
            f"- Root seed: {self._root.seed}",
            f"- Forward-model evaluations: {self.total_evaluations:,}",
            "",
        ]
        for index, record in enumerate(self._records, start=1):
            lines.append(f"## {index}. {record.stage}: {record.label}")
            lines.append("")
            lines.append(
                f"Evaluations: {record.evaluations:,} · stream seed: {record.seed}"
            )
            lines.append("")
            lines.append("| Quantity | Value |")
            lines.append("|---|---|")
            for key, value in record.summary.items():
                text = (
                    f"{value:.6g}"
                    if isinstance(value, float) and math.isfinite(value)
                    else str(value)
                )
                lines.append(f"| {key} | {text} |")
            lines.append("")
        return "\n".join(lines)
