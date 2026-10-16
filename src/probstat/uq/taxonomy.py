"""Core vocabulary and contracts for verification and uncertainty studies."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Literal, Protocol, overload, runtime_checkable

from probstat.foundations.validation import require_finite_real

from .evaluation import (
    FailedEvaluation,
    FailureLog,
    FailurePolicy,
    FrozenMapping,
    describe_error,
    require_failure_policy,
)


class UncertaintyNature(StrEnum):
    """Whether uncertainty describes variability or incomplete knowledge.

    Aleatory uncertainty models irreducible variability under the chosen model;
    epistemic uncertainty represents incomplete knowledge and may be reducible.

    References:
        [REF-SMITH-UQ-2024, Chapters 1 and 4].

    """

    ALEATORY = "aleatory"
    EPISTEMIC = "epistemic"


class UncertaintySourceKind(StrEnum):
    """Engineering source category for one uncertainty contribution.

    Nature and source are separate axes. For example, a parameter can represent
    population variability (aleatory) or imperfect knowledge (epistemic).

    References:
        [REF-OBERKAMPF-ROY-2025, Chapters 1-2].

    """

    PARAMETRIC = "parametric"
    MODEL_FORM = "model_form"
    NUMERICAL = "numerical"
    OBSERVATION = "observation"


@dataclass(frozen=True, slots=True)
class UncertaintySource:
    """A named and classified source of uncertainty in a simulation study.

    Args:
        name: Non-empty stable identifier.
        description: Non-empty explanation of the uncertain quantity or error.
        nature: Aleatory or epistemic interpretation.
        kind: Parametric, model-form, numerical, or observation source.
        unit: Optional non-empty physical unit text.

    References:
        [REF-SMITH-UQ-2024, Chapters 1 and 4].

    """

    name: str
    description: str
    nature: UncertaintyNature
    kind: UncertaintySourceKind
    unit: str | None = None

    def __post_init__(self) -> None:
        """Validate descriptive fields and explicit enum classifications.

        Raises:
            TypeError: If a field has an invalid type.
            ValueError: If required text is empty.

        References:
            [REF-SMITH-UQ-2024, Chapters 1 and 4]. No equation is evaluated.

        """
        _require_non_empty_text(self.name, name="name")
        _require_non_empty_text(self.description, name="description")
        if not isinstance(self.nature, UncertaintyNature):
            raise TypeError("nature must be an UncertaintyNature")
        if not isinstance(self.kind, UncertaintySourceKind):
            raise TypeError("kind must be an UncertaintySourceKind")
        if self.unit is not None:
            _require_non_empty_text(self.unit, name="unit")


@dataclass(frozen=True, slots=True)
class QuantityOfInterest:
    """A named simulation output whose uncertainty will be analyzed.

    Args:
        name: Non-empty key expected in model output.
        description: Non-empty physical or decision-focused interpretation.
        unit: Optional non-empty physical unit text.

    References:
        [REF-SMITH-UQ-2024, Chapters 1-3].

    """

    name: str
    description: str
    unit: str | None = None

    def __post_init__(self) -> None:
        """Validate the quantity name, description, and optional unit.

        Raises:
            TypeError: If a field has an invalid type.
            ValueError: If required text is empty.

        References:
            [REF-SMITH-UQ-2024, Chapters 1-3]. No equation is evaluated.

        """
        _require_non_empty_text(self.name, name="name")
        _require_non_empty_text(self.description, name="description")
        if self.unit is not None:
            _require_non_empty_text(self.unit, name="unit")


@runtime_checkable
class ForwardModel(Protocol):
    """A deterministic map from named input parameters to named outputs.

    Stochastic simulators should make their random stream an explicit input or
    wrap each deterministic realization behind this interface.

    References:
        [REF-SMITH-UQ-2024, Chapters 1-3].

    """

    def __call__(self, parameters: Mapping[str, float]) -> Mapping[str, float]:
        """Evaluate named outputs for one finite parameter realization.

        Args:
            parameters: Read-only-by-contract mapping of finite input values.

        Returns:
            Mapping containing each declared quantity-of-interest key.

        References:
            [EQ-UQ-001], [REF-SMITH-UQ-2024, Chapters 1-3].

        """
        ...


@overload
def evaluate_model(  # noqa: D418 - the documentation contract needs one
    model: ForwardModel,
    parameters: Mapping[str, object],
    quantities: Iterable[QuantityOfInterest],
    *,
    passthrough: Collection[str] = (),
    on_failure: Literal["raise"] = "raise",
    failures: FailureLog | None = None,
) -> Mapping[str, float]:
    """Evaluate under the raising policy: the result is never ``None``.

    References:
        [EQ-UQ-001], [REF-SMITH-UQ-2024, Chapters 1-3].

    """
    ...


@overload
def evaluate_model(  # noqa: D418 - the documentation contract needs one
    model: ForwardModel,
    parameters: Mapping[str, object],
    quantities: Iterable[QuantityOfInterest],
    *,
    passthrough: Collection[str] = (),
    on_failure: FailurePolicy,
    failures: FailureLog | None = None,
) -> Mapping[str, float] | None:
    """Evaluate under a dropping policy: ``None`` marks a failed run.

    References:
        [EQ-UQ-001], [REF-SMITH-UQ-2024, Chapters 1-3].

    """
    ...


def evaluate_model(
    model: ForwardModel,
    parameters: Mapping[str, object],
    quantities: Iterable[QuantityOfInterest],
    *,
    passthrough: Collection[str] = (),
    on_failure: FailurePolicy = "raise",
    failures: FailureLog | None = None,
) -> Mapping[str, float] | None:
    """Evaluate a forward model and extract validated quantities of interest.

    A *run failure* is an exception raised by the model, an output that is
    not a mapping, a missing requested QoI, or a non-finite or non-real QoI
    value. Invalid *arguments* (parameters, quantities) always raise, whatever
    the policy, because they are caller errors rather than simulator failures.

    Under ``on_failure="raise"`` (default) the original exception propagates
    unchanged. Under ``"skip"`` or ``"record"`` the failure is added to
    ``failures`` (with the inputs only for ``"record"``) and ``None`` is
    returned, so the caller can drop the run. Dropping runs is unbiased only
    when failures are ignorable; see :mod:`probstat.uq.evaluation`.

    Args:
        model: Deterministic named-input/named-output forward map.
        parameters: Non-empty mapping of named finite scalar inputs.
        quantities: Non-empty collection of distinctly named requested outputs.
        passthrough: Parameter names whose values are passed to the model
            unchanged and unvalidated (for example strings, integers,
            Booleans, or tuples of fixed solver options).
        on_failure: ``"raise"``, ``"skip"``, or ``"record"``.
        failures: Optional log that counts (and records) failed runs under
            every policy.

    Returns:
        Read-only mapping of requested QoI names to finite values, or ``None``
        for a failed run under ``"skip"``/``"record"``.

    Raises:
        TypeError: If the model, mappings, names, values, or QoIs are invalid.
        ValueError: If input is empty, QoI names repeat, text is empty, values
            are non-finite, a requested output is absent, or the policy is
            unknown.

    Equation:
        [EQ-UQ-001].

    References:
        [REF-SMITH-UQ-2024, Chapters 1-3].

    """
    policy = require_failure_policy(on_failure)
    if not callable(model):
        raise TypeError("model must be callable")
    if not isinstance(parameters, Mapping):
        raise TypeError("parameters must be a mapping")
    if not parameters:
        raise ValueError("parameters must not be empty")
    unchanged = frozenset(passthrough)
    checked_parameters: dict[str, object] = {}
    for name, value in parameters.items():
        _require_non_empty_text(name, name="parameter name")
        checked_parameters[name] = (
            value
            if name in unchanged
            else require_finite_real(value, name=f"parameters[{name!r}]")
        )

    requested = tuple(quantities)
    if not requested:
        raise ValueError("quantities must not be empty")
    names: set[str] = set()
    for index, quantity in enumerate(requested):
        if not isinstance(quantity, QuantityOfInterest):
            raise TypeError(f"quantities[{index}] must be a QuantityOfInterest")
        if quantity.name in names:
            raise ValueError(f"quantity name {quantity.name!r} is repeated")
        names.add(quantity.name)

    try:
        outputs = model(MappingProxyType(checked_parameters))  # type: ignore[arg-type]
        return select_outputs(outputs, requested)
    except Exception as error:
        if failures is not None or policy != "raise":
            error_type, message = describe_error(error)
            failure = FailedEvaluation(
                None,
                FrozenMapping({} if policy == "skip" else checked_parameters),
                error_type,
                message,
            )
            if failures is not None:
                failures.add(failure)
        if policy == "raise":
            raise
        return None


def select_outputs(
    outputs: object, quantities: Sequence[QuantityOfInterest]
) -> Mapping[str, float]:
    """Validate raw model output and keep the requested quantities.

    Args:
        outputs: Raw value returned by a forward model.
        quantities: Requested quantities (already validated).

    Returns:
        Read-only, picklable mapping of QoI names to finite floats.

    Raises:
        TypeError: If the output is not a mapping or a value is not real.
        ValueError: If a QoI is missing or not finite.

    Equation:
        [EQ-UQ-001].

    References:
        [REF-SMITH-UQ-2024, Chapters 1-3].

    """
    if type(outputs) is not dict and not isinstance(outputs, Mapping):
        raise TypeError("model output must be a mapping")
    selected: dict[str, float] = {}
    for quantity in quantities:
        if quantity.name not in outputs:
            raise ValueError(f"model output is missing QoI {quantity.name!r}")
        selected[quantity.name] = require_finite_real(
            outputs[quantity.name], name=f"output[{quantity.name!r}]"
        )
    return FrozenMapping(selected)


def _require_non_empty_text(value: object, *, name: str) -> str:
    """Return stripped non-empty text or raise a field-specific error.

    Args:
        value: Candidate text.
        name: Field name used in error messages.

    Returns:
        The original string after validation.

    Raises:
        TypeError: If ``value`` is not a string.
        ValueError: If it contains only whitespace.

    References:
        Project validation policy. No mathematical equation is evaluated.

    """
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must not be empty")
    return value
