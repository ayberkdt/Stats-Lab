"""Auditable record of the steps of an uncertainty-quantification workflow.

A UQ study is a sequence of distinct questions: what is predicted (QoI
definition), what is uncertain (input characterization), are the equations
solved correctly (verification), do the predictions agree with data where
tested (validation), what output uncertainty do the inputs induce (forward
propagation), what do data say about the inputs (inverse calibration), and
what is finally claimed at the intended use (prediction).

:class:`UQWorkflowRecord` holds one :class:`WorkflowEntry` per step with a
status and evidence references. A step can only be marked complete with
evidence, and :attr:`UQWorkflowRecord.gaps` lists steps that were completed
before their prerequisites. :data:`WORKFLOW_GUIDE` maps every step to the
library calls that produce its evidence; ``probstat.workflow.UQStudy`` stage
records can be attached with :meth:`UQWorkflowRecord.record_stages` without
this layer importing the workflow package.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Protocol

from .taxonomy import QuantityOfInterest


class WorkflowStep(str, Enum):
    """The seven steps of a verification-validation-UQ workflow, in order.

    References:
        [REF-ROY-OBERKAMPF-2011], [REF-SMITH-UQ-2024, Chapter 1].

    """

    QOI_DEFINITION = "qoi_definition"
    INPUT_CHARACTERIZATION = "input_characterization"
    VERIFICATION = "verification"
    VALIDATION = "validation"
    PROPAGATION = "propagation"
    CALIBRATION = "calibration"
    PREDICTION = "prediction"


class StepStatus(str, Enum):
    """Progress of one workflow step.

    References:
        [REF-ROY-OBERKAMPF-2011].

    """

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETE = "complete"
    NOT_APPLICABLE = "not_applicable"


class ProblemDirection(str, Enum):
    """Whether a step maps inputs to outputs, data to inputs, or neither.

    A forward problem pushes an input law through the model; an inverse
    problem uses observed outputs to learn inputs or parameters.

    References:
        [REF-SMITH-UQ-2024, Chapters 1 and 8], [REF-TARANTOLA-2005, Chapter 1].

    """

    FORWARD = "forward"
    INVERSE = "inverse"
    EVIDENCE = "evidence"


@dataclass(frozen=True, slots=True)
class StepGuide:
    """What a workflow step asks and which library calls answer it.

    Args:
        step: Workflow step.
        question: The question the step answers.
        direction: Forward, inverse, or supporting evidence.
        library_calls: Dotted names of the public callables that produce the
            step's evidence.
        study_stages: ``UQStudy`` stage kinds that produce the evidence.

    References:
        [REF-ROY-OBERKAMPF-2011], [REF-SMITH-UQ-2024, Chapter 1].

    """

    step: WorkflowStep
    question: str
    direction: ProblemDirection
    library_calls: tuple[str, ...]
    study_stages: tuple[str, ...]


WORKFLOW_GUIDE: Mapping[WorkflowStep, StepGuide] = MappingProxyType(
    {
        WorkflowStep.QOI_DEFINITION: StepGuide(
            WorkflowStep.QOI_DEFINITION,
            "Which model output, in which unit, supports the decision?",
            ProblemDirection.EVIDENCE,
            (
                "probstat.uq.QuantityOfInterest",
                "probstat.uq.evaluate_model",
                "probstat.modeling.ProbabilisticModel",
            ),
            (),
        ),
        WorkflowStep.INPUT_CHARACTERIZATION: StepGuide(
            WorkflowStep.INPUT_CHARACTERIZATION,
            "Which inputs are uncertain, of what nature, with what joint law?",
            ProblemDirection.EVIDENCE,
            (
                "probstat.uq.UncertaintySource",
                "probstat.modeling.RandomInput",
                "probstat.modeling.JointInput",
                "probstat.modeling.fit_joint_input",
                "probstat.workflow.UQStudy.from_data",
            ),
            ("fitting",),
        ),
        WorkflowStep.VERIFICATION: StepGuide(
            WorkflowStep.VERIFICATION,
            "Are the equations solved correctly, and how large is u_num?",
            ProblemDirection.EVIDENCE,
            (
                "probstat.uq.refinement_study",
                "probstat.uq.verify_solution",
                "probstat.uq.grid_convergence_index",
                "probstat.uq.least_squares_convergence",
                "probstat.validation.VerificationEvidence",
            ),
            (),
        ),
        WorkflowStep.VALIDATION: StepGuide(
            WorkflowStep.VALIDATION,
            "Do predictions agree with independent data inside the domain?",
            ProblemDirection.EVIDENCE,
            (
                "probstat.validation.assess_validation",
                "probstat.validation.assess_predictions",
                "probstat.validation.estimate_discrepancy",
                "probstat.workflow.UQStudy.validate",
            ),
            ("validation",),
        ),
        WorkflowStep.PROPAGATION: StepGuide(
            WorkflowStep.PROPAGATION,
            "What output uncertainty does the input law induce?",
            ProblemDirection.FORWARD,
            (
                "probstat.propagation.propagate",
                "probstat.workflow.UQStudy.propagate",
                "probstat.workflow.UQStudy.sensitivity",
                "probstat.workflow.UQStudy.reliability",
            ),
            (
                "propagation",
                "sensitivity",
                "shapley",
                "screening",
                "reliability",
                "form",
                "risk",
                "emulation",
            ),
        ),
        WorkflowStep.CALIBRATION: StepGuide(
            WorkflowStep.CALIBRATION,
            "What do observed outputs say about the uncertain inputs?",
            ProblemDirection.INVERSE,
            (
                "probstat.inverse.CalibrationProblem",
                "probstat.inverse.calibrate_linear_gaussian",
                "probstat.workflow.UQStudy.calibrate",
            ),
            ("calibration",),
        ),
        WorkflowStep.PREDICTION: StepGuide(
            WorkflowStep.PREDICTION,
            "What is claimed at the intended use, with which components?",
            ProblemDirection.FORWARD,
            (
                "probstat.inverse.predict",
                "probstat.workflow.UQStudy.predict",
                "probstat.validation.predict_with_uncertainty_components",
                "probstat.validation.PredictionStatement",
            ),
            ("prediction",),
        ),
    }
)
"""Map from every workflow step to its question, direction, and library calls."""

_PREREQUISITES: Mapping[WorkflowStep, tuple[WorkflowStep, ...]] = MappingProxyType(
    {
        WorkflowStep.QOI_DEFINITION: (),
        WorkflowStep.INPUT_CHARACTERIZATION: (WorkflowStep.QOI_DEFINITION,),
        WorkflowStep.VERIFICATION: (WorkflowStep.QOI_DEFINITION,),
        WorkflowStep.VALIDATION: (
            WorkflowStep.QOI_DEFINITION,
            WorkflowStep.VERIFICATION,
        ),
        WorkflowStep.PROPAGATION: (
            WorkflowStep.QOI_DEFINITION,
            WorkflowStep.INPUT_CHARACTERIZATION,
        ),
        WorkflowStep.CALIBRATION: (
            WorkflowStep.QOI_DEFINITION,
            WorkflowStep.INPUT_CHARACTERIZATION,
        ),
        WorkflowStep.PREDICTION: (
            WorkflowStep.VERIFICATION,
            WorkflowStep.VALIDATION,
        ),
    }
)


class StageLike(Protocol):
    """Structural type of a ``probstat.workflow.StageRecord``.

    References:
        [REF-SMITH-UQ-2024, Chapter 1].

    """

    @property
    def stage(self) -> str:
        """Return the stage kind.

        References:
            [REF-SMITH-UQ-2024, Chapter 1].

        """
        ...

    @property
    def label(self) -> str:
        """Return the stage description.

        References:
            [REF-SMITH-UQ-2024, Chapter 1].

        """
        ...

    @property
    def seed(self) -> int:
        """Return the stream seed of the stage.

        References:
            [REF-STEELE-LEA-FLOOD-2014].

        """
        ...

    @property
    def evaluations(self) -> int:
        """Return the forward-model evaluations charged to the stage.

        References:
            [REF-SMITH-UQ-2024, Chapter 3].

        """
        ...


@dataclass(frozen=True, slots=True)
class WorkflowEntry:
    """Status and evidence of one workflow step.

    Args:
        step: Workflow step.
        status: Progress of the step.
        evidence: References to the artefacts that support the status
            (result summaries, file paths, study-stage seeds).
        note: Free text; required when the step is not applicable.

    Raises:
        TypeError: If the step or status has the wrong type.
        ValueError: If a complete step has no evidence or a not-applicable
            step has no justification.

    References:
        [REF-ROY-OBERKAMPF-2011].

    """

    step: WorkflowStep
    status: StepStatus = StepStatus.PENDING
    evidence: tuple[str, ...] = ()
    note: str = ""

    def __post_init__(self) -> None:
        """Validate the entry.

        References:
            [REF-ROY-OBERKAMPF-2011]. No equation is evaluated.

        """
        if not isinstance(self.step, WorkflowStep):
            raise TypeError("step must be a WorkflowStep")
        if not isinstance(self.status, StepStatus):
            raise TypeError("status must be a StepStatus")
        if isinstance(self.evidence, str):
            raise TypeError("evidence must be a sequence of strings")
        evidence = tuple(self.evidence)
        for index, item in enumerate(evidence):
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"evidence[{index}] must be a non-empty string")
        if not isinstance(self.note, str):
            raise TypeError("note must be a string")
        if self.status is StepStatus.COMPLETE and not evidence:
            raise ValueError(
                f"step {self.step.value!r} cannot be complete without evidence"
            )
        if self.status is StepStatus.NOT_APPLICABLE and not self.note.strip():
            raise ValueError(
                f"step {self.step.value!r} marked not applicable needs a note"
            )
        object.__setattr__(self, "evidence", evidence)

    @property
    def settled(self) -> bool:
        """Return whether the step is complete or justified as not applicable.

        References:
            [REF-ROY-OBERKAMPF-2011].

        """
        return self.status in (StepStatus.COMPLETE, StepStatus.NOT_APPLICABLE)


@dataclass(frozen=True, slots=True)
class UQWorkflowRecord:
    """One entry per workflow step for a named study and its QoIs.

    Args:
        name: Study label.
        quantities: Quantities of interest the workflow addresses.
        entries: Exactly one entry per :class:`WorkflowStep` (any order;
            stored in canonical order).

    Raises:
        TypeError: If an item has the wrong type.
        ValueError: If the name is empty, no QoI is given, or a step is
            missing or repeated.

    References:
        [REF-ROY-OBERKAMPF-2011], [REF-OBERKAMPF-ROY-2025, Chapters 1-2],
        [REF-SMITH-UQ-2024, Chapter 1].

    """

    name: str
    quantities: tuple[QuantityOfInterest, ...]
    entries: tuple[WorkflowEntry, ...]

    def __post_init__(self) -> None:
        """Validate and order the entries.

        References:
            [REF-ROY-OBERKAMPF-2011]. No equation is evaluated.

        """
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")
        quantities = tuple(self.quantities)
        if not quantities:
            raise ValueError("at least one quantity of interest is required")
        for index, quantity in enumerate(quantities):
            if not isinstance(quantity, QuantityOfInterest):
                raise TypeError(f"quantities[{index}] must be a QuantityOfInterest")
        by_step: dict[WorkflowStep, WorkflowEntry] = {}
        for index, entry in enumerate(self.entries):
            if not isinstance(entry, WorkflowEntry):
                raise TypeError(f"entries[{index}] must be a WorkflowEntry")
            if entry.step in by_step:
                raise ValueError(f"step {entry.step.value!r} is repeated")
            by_step[entry.step] = entry
        missing = [step.value for step in WorkflowStep if step not in by_step]
        if missing:
            raise ValueError(f"missing workflow steps: {', '.join(missing)}")
        object.__setattr__(self, "quantities", quantities)
        object.__setattr__(
            self, "entries", tuple(by_step[step] for step in WorkflowStep)
        )

    @classmethod
    def start(
        cls, name: str, quantities: Iterable[QuantityOfInterest]
    ) -> UQWorkflowRecord:
        """Return a record with every step pending.

        Args:
            name: Study label.
            quantities: Quantities of interest.

        Returns:
            New record.

        References:
            [REF-ROY-OBERKAMPF-2011]. No equation is evaluated.

        """
        return cls(
            name, tuple(quantities), tuple(WorkflowEntry(step) for step in WorkflowStep)
        )

    def entry(self, step: WorkflowStep) -> WorkflowEntry:
        """Return the entry of one step.

        Args:
            step: Workflow step.

        Returns:
            Its entry.

        Raises:
            TypeError: If ``step`` is not a :class:`WorkflowStep`.

        References:
            [REF-ROY-OBERKAMPF-2011]. No equation is evaluated.

        """
        if not isinstance(step, WorkflowStep):
            raise TypeError("step must be a WorkflowStep")
        return self.entries[list(WorkflowStep).index(step)]

    def update(
        self,
        step: WorkflowStep,
        status: StepStatus,
        *,
        evidence: Iterable[str] = (),
        note: str | None = None,
    ) -> UQWorkflowRecord:
        """Return a copy with one step's status changed and evidence appended.

        Args:
            step: Workflow step.
            status: New status.
            evidence: Evidence references appended to the existing ones.
            note: Replacement note (``None`` keeps the current note).

        Returns:
            New record; this record is unchanged.

        Raises:
            ValueError: If the resulting entry is invalid (see
                :class:`WorkflowEntry`).

        References:
            [REF-ROY-OBERKAMPF-2011]. No equation is evaluated.

        """
        current = self.entry(step)
        if isinstance(evidence, str):
            raise TypeError("evidence must be an iterable of strings")
        replacement = WorkflowEntry(
            step,
            status,
            current.evidence + tuple(evidence),
            current.note if note is None else note,
        )
        entries = tuple(
            replacement if entry.step is step else entry for entry in self.entries
        )
        return UQWorkflowRecord(self.name, self.quantities, entries)

    def record_stages(self, stages: Iterable[StageLike]) -> UQWorkflowRecord:
        """Attach ``UQStudy`` stage records as evidence of the matching steps.

        Each stage kind is mapped through :data:`WORKFLOW_GUIDE`
        (``propagation``, ``sensitivity``, ``reliability``, ... to
        propagation; ``calibration``; ``prediction``; ``validation``) and the
        step is marked complete with an evidence line naming the stage, its
        label, stream seed, and evaluation count.

        Args:
            stages: Objects with ``stage``, ``label``, ``seed``, and
                ``evaluations`` attributes (``UQStudy.records``).

        Returns:
            New record.

        Raises:
            TypeError: If a stage lacks a required attribute.
            ValueError: If a stage kind maps to no workflow step.

        References:
            [REF-SMITH-UQ-2024, Chapter 1]. No equation is evaluated.

        """
        stage_to_step = {
            stage: guide.step
            for guide in WORKFLOW_GUIDE.values()
            for stage in guide.study_stages
        }
        record = self
        for index, item in enumerate(stages):
            for attribute in ("stage", "label", "seed", "evaluations"):
                if not hasattr(item, attribute):
                    raise TypeError(f"stages[{index}] has no attribute {attribute!r}")
            kind = str(item.stage)
            if kind not in stage_to_step:
                raise ValueError(f"stage kind {kind!r} maps to no workflow step")
            evidence = (
                f"UQStudy stage '{kind}': {item.label} (stream seed {item.seed}, "
                f"{item.evaluations} model evaluations)"
            )
            record = record.update(
                stage_to_step[kind], StepStatus.COMPLETE, evidence=(evidence,)
            )
        return record

    @property
    def directions(self) -> tuple[ProblemDirection, ...]:
        """Return the problem directions exercised by completed steps.

        Returns:
            ``FORWARD`` when propagation or prediction is complete and
            ``INVERSE`` when calibration is complete, in that order.

        References:
            [REF-SMITH-UQ-2024, Chapters 1 and 8].

        """
        found: list[ProblemDirection] = []
        for direction in (ProblemDirection.FORWARD, ProblemDirection.INVERSE):
            if any(
                entry.status is StepStatus.COMPLETE
                and WORKFLOW_GUIDE[entry.step].direction is direction
                for entry in self.entries
            ):
                found.append(direction)
        return tuple(found)

    @property
    def gaps(self) -> tuple[str, ...]:
        """Return unsettled steps and steps completed before prerequisites.

        A prediction additionally needs a complete propagation or
        calibration step (some forward or inverse uncertainty analysis).

        References:
            [REF-ROY-OBERKAMPF-2011], [REF-OBERKAMPF-ROY-2025, Part IV].

        """
        gaps: list[str] = []
        for entry in self.entries:
            if not entry.settled:
                gaps.append(f"{entry.step.value} is {entry.status.value}")
        for entry in self.entries:
            if entry.status is not StepStatus.COMPLETE:
                continue
            for prerequisite in _PREREQUISITES[entry.step]:
                if not self.entry(prerequisite).settled:
                    gaps.append(
                        f"{entry.step.value} is complete but its prerequisite "
                        f"{prerequisite.value} is not settled"
                    )
        prediction = self.entry(WorkflowStep.PREDICTION)
        if prediction.status is StepStatus.COMPLETE and not any(
            self.entry(step).status is StepStatus.COMPLETE
            for step in (WorkflowStep.PROPAGATION, WorkflowStep.CALIBRATION)
        ):
            gaps.append(
                "prediction is complete without a complete propagation or "
                "calibration step"
            )
        return tuple(gaps)

    @property
    def complete(self) -> bool:
        """Return whether every step is settled and no gap remains.

        References:
            [REF-ROY-OBERKAMPF-2011].

        """
        return not self.gaps

    def summary(self) -> str:
        """Return a Markdown table of steps, statuses, evidence, and gaps.

        Returns:
            Markdown text.

        References:
            [REF-ROY-OBERKAMPF-2011]. No equation is evaluated.

        """
        names = ", ".join(quantity.name for quantity in self.quantities)
        lines = [
            f"# UQ workflow: {self.name}",
            "",
            f"Quantities of interest: {names}",
            "",
            "| Step | Direction | Status | Evidence | Note |",
            "|---|---|---|---|---|",
        ]
        for entry in self.entries:
            evidence = "<br>".join(entry.evidence) if entry.evidence else "-"
            lines.append(
                f"| {entry.step.value} | {WORKFLOW_GUIDE[entry.step].direction.value} "
                f"| {entry.status.value} | {evidence} | {entry.note or '-'} |"
            )
        lines.extend(["", "Gaps:"])
        gaps = self.gaps
        if gaps:
            lines.extend(f"- {gap}" for gap in gaps)
        else:
            lines.append("- none")
        return "\n".join(lines)
