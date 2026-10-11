"""Numerical simulation recorded as a model-based computational experiment.

A physical experiment fixes controlled conditions, measures responses, and
documents the apparatus. A simulation is treated the same way here: a
:class:`ComputationalExperiment` records the model identity, the physical
inputs, the *numerical settings* (grid spacing, time step, tolerances, seeds of
stochastic solvers) that act as factors of the computational apparatus, and
the resulting quantities of interest. A grid-refinement study is then a
designed computational experiment in which only the numerical settings change,
and :func:`refinement_study` turns it into a solution-verification statement.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from probstat.foundations.validation import require_finite_real

from .taxonomy import ForwardModel, QuantityOfInterest, evaluate_model
from .verification import (
    DEFAULT_SAFETY_FACTOR,
    SolutionVerification,
    verify_solution,
)


def _freeze(values: Mapping[str, float], *, name: str) -> Mapping[str, float]:
    """Validate a name-to-number mapping and return a read-only copy.

    Args:
        values: Mapping of non-empty names to finite reals.
        name: Field name used in error messages.

    Returns:
        Read-only mapping with float values.

    Raises:
        TypeError: If the argument is not a mapping or a value is not real.
        ValueError: If a name is empty or a value is non-finite.

    References:
        Project validation policy. No equation is evaluated.

    """
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping")
    frozen: dict[str, float] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{name} keys must be non-empty strings")
        frozen[key] = require_finite_real(value, name=f"{name}[{key!r}]")
    return MappingProxyType(frozen)


@dataclass(frozen=True, slots=True)
class ComputationalExperiment:
    """One recorded run of a simulation, documented like a physical experiment.

    Args:
        name: Non-empty run identifier.
        model_name: Identity (and version) of the simulation code or model.
        parameters: Physical inputs and conditions of the run.
        settings: Numerical controls of the run (spacing, step, tolerance,
            solver seed); kept separate from the physical inputs.
        outputs: Recorded quantities of interest.
        purpose: Why the run was made (verification, validation, prediction,
            design exploration, ...).

    Raises:
        TypeError: If a mapping has the wrong type.
        ValueError: If text is empty, a value is non-finite, or a setting
            shadows a parameter.

    References:
        [REF-OBERKAMPF-ROY-2025, Chapters 1-2], [REF-SMITH-UQ-2024, Chapter 1].

    """

    name: str
    model_name: str
    parameters: Mapping[str, float]
    settings: Mapping[str, float]
    outputs: Mapping[str, float]
    purpose: str = "unspecified"

    def __post_init__(self) -> None:
        """Validate and freeze the record.

        References:
            [REF-OBERKAMPF-ROY-2025, Chapters 1-2]. No equation is evaluated.

        """
        for field_name in ("name", "model_name", "purpose"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        parameters = _freeze(self.parameters, name="parameters")
        settings = _freeze(self.settings, name="settings")
        overlap = sorted(set(parameters) & set(settings))
        if overlap:
            raise ValueError(f"settings shadow parameters: {', '.join(overlap)}")
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "settings", settings)
        object.__setattr__(self, "outputs", _freeze(self.outputs, name="outputs"))

    def describe(self) -> str:
        """Return a one-line description of the run.

        Returns:
            Text naming model, parameters, settings, and outputs.

        References:
            [REF-OBERKAMPF-ROY-2025, Chapters 1-2]. No equation is evaluated.

        """

        def join(values: Mapping[str, float]) -> str:
            """Format a mapping as ``k=v`` pairs.

            Args:
                values: Mapping to format.

            Returns:
                Comma-separated text, or ``"none"``.

            References:
                Project reporting policy. No equation is evaluated.

            """
            if not values:
                return "none"
            return ", ".join(f"{k}={v:.6g}" for k, v in values.items())

        return (
            f"{self.name} [{self.purpose}] model={self.model_name}; "
            f"parameters: {join(self.parameters)}; settings: "
            f"{join(self.settings)}; outputs: {join(self.outputs)}"
        )


def run_computational_experiment(
    model: ForwardModel,
    quantities: Iterable[QuantityOfInterest],
    parameters: Mapping[str, float],
    *,
    settings: Mapping[str, float] | None = None,
    name: str = "run",
    model_name: str = "model",
    purpose: str = "unspecified",
) -> ComputationalExperiment:
    """Run a forward model once and record it as a computational experiment.

    Physical parameters and numerical settings are merged into one call of
    ``model`` (settings must not reuse a parameter name) and the requested
    quantities are extracted with :func:`evaluate_model`.

    Args:
        model: Deterministic forward model.
        quantities: Requested quantities of interest.
        parameters: Physical inputs and conditions.
        settings: Numerical controls passed to the model as extra inputs.
        name: Run identifier.
        model_name: Identity of the simulation code.
        purpose: Why the run is made.

    Returns:
        Recorded computational experiment.

    Raises:
        TypeError: If the model or a mapping is invalid.
        ValueError: If a setting shadows a parameter, a value is non-finite,
            or a requested output is missing.

    Equation:
        [EQ-UQ-001].

    References:
        [REF-OBERKAMPF-ROY-2025, Chapters 1-2], [REF-SMITH-UQ-2024, Chapters 1-3].

    """
    physical = _freeze(parameters, name="parameters")
    numerical = _freeze(settings or {}, name="settings")
    overlap = sorted(set(physical) & set(numerical))
    if overlap:
        raise ValueError(f"settings shadow parameters: {', '.join(overlap)}")
    merged = dict(physical)
    merged.update(numerical)
    outputs = evaluate_model(model, merged, quantities)
    return ComputationalExperiment(
        name, model_name, physical, numerical, outputs, purpose
    )


@dataclass(frozen=True, slots=True)
class RefinementStudy:
    """A grid-refinement computational experiment and its verification result.

    Args:
        quantity: QoI analysed.
        spacing_name: Name of the numerical setting that was refined.
        experiments: One recorded run per spacing, in the order requested.
        verification: Numerical-uncertainty statement from the runs.

    References:
        [REF-ROACHE-1998, Chapters 4-5], [REF-CELIK-ETAL-2008].

    """

    quantity: str
    spacing_name: str
    experiments: tuple[ComputationalExperiment, ...]
    verification: SolutionVerification

    @property
    def spacings(self) -> tuple[float, ...]:
        """Return the refined setting of every run.

        References:
            [REF-CELIK-ETAL-2008].

        """
        return tuple(run.settings[self.spacing_name] for run in self.experiments)

    @property
    def values(self) -> tuple[float, ...]:
        """Return the QoI of every run.

        References:
            [REF-CELIK-ETAL-2008].

        """
        return tuple(run.outputs[self.quantity] for run in self.experiments)


def refinement_study(
    model: ForwardModel,
    quantity: QuantityOfInterest,
    parameters: Mapping[str, float],
    spacings: Sequence[float],
    *,
    spacing_name: str = "h",
    settings: Mapping[str, float] | None = None,
    model_name: str = "model",
    theoretical_order: float = 2.0,
    safety_factor: float = DEFAULT_SAFETY_FACTOR,
    asymptotic_tolerance: float = 0.1,
) -> RefinementStudy:
    """Run a systematic refinement study and verify the solution.

    Every run uses the same physical ``parameters`` and fixed ``settings``;
    only the setting ``spacing_name`` changes. The QoI sequence is passed to
    :func:`~probstat.uq.verification.verify_solution` (three-grid GCI or
    least-squares order for four or more grids).

    Args:
        model: Deterministic forward model that reads ``spacing_name``.
        quantity: QoI to verify.
        parameters: Physical inputs held fixed during refinement.
        spacings: Three or more distinct positive spacings.
        spacing_name: Name of the refined numerical setting.
        settings: Other numerical settings held fixed.
        model_name: Identity of the simulation code.
        theoretical_order: Formal order of the discretization.
        safety_factor: ``F_s`` of the three-grid GCI.
        asymptotic_tolerance: Allowed deviation of the asymptotic ratio.

    Returns:
        Refinement study with runs and the verification statement.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``spacing_name`` collides with another setting or a
            parameter, or the refinement data are degenerate or divergent.

    Equation:
        [EQ-UQ-001], [EQ-VV-GCI-004], [EQ-VV-LSQ-002].

    References:
        [REF-ROACHE-1998, Chapters 4-5], [REF-CELIK-ETAL-2008],
        [REF-OBERKAMPF-ROY-2025, Chapters 6-8].

    """
    if not isinstance(quantity, QuantityOfInterest):
        raise TypeError("quantity must be a QuantityOfInterest")
    if not isinstance(spacing_name, str) or not spacing_name.strip():
        raise ValueError("spacing_name must be a non-empty string")
    fixed_settings = dict(_freeze(settings or {}, name="settings"))
    if spacing_name in fixed_settings:
        raise ValueError("spacing_name must not also appear in settings")
    runs: list[ComputationalExperiment] = []
    for index, spacing in enumerate(spacings):
        run_settings = dict(fixed_settings)
        run_settings[spacing_name] = require_finite_real(
            spacing, name=f"spacings[{index}]"
        )
        runs.append(
            run_computational_experiment(
                model,
                (quantity,),
                parameters,
                settings=run_settings,
                name=f"{quantity.name}-{spacing_name}={spacing:.6g}",
                model_name=model_name,
                purpose="solution verification (grid refinement)",
            )
        )
    verification = verify_solution(
        quantity.name,
        [run.settings[spacing_name] for run in runs],
        [run.outputs[quantity.name] for run in runs],
        theoretical_order=theoretical_order,
        safety_factor=safety_factor,
        asymptotic_tolerance=asymptotic_tolerance,
    )
    return RefinementStudy(quantity.name, spacing_name, tuple(runs), verification)
