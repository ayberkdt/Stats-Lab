"""Solution-verification and validation stages.

References:
    [REF-ROACHE-1998, Chapters 4-5], [REF-OBERKAMPF-ROY-2025, Part IV],
    [REF-JCGM-100-2008, Sections 4.3.7 and 6.2].

"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

from probstat.foundations.validation import require_finite_real
from probstat.uq import (
    DEFAULT_SAFETY_FACTOR,
    GridConvergenceResult,
    LeastSquaresConvergence,
    NumericalErrorBudget,
    NumericalErrorDistribution,
    SolutionVerification,
    verify_solution,
)
from probstat.validation import (
    ValidationAssessment,
    ValidationExperiment,
    assess_validation,
    crps_ensemble,
    ensemble_pit,
    pit_coverage,
)

from ._base import _stage, _StudyBase
from ._context import _sized


class _ValidationStages(_StudyBase):
    """Verification and validation stages of the study.

    References:
        [REF-ROACHE-1998, Chapters 4-5], [REF-OBERKAMPF-ROY-2025, Part IV].

    """

    @_stage("verification")
    def verify(
        self,
        quantity: str,
        grid_values: Sequence[float],
        refinement_ratio: float | None = None,
        *,
        grid_sizes: Sequence[float] | None = None,
        safety_factor: float | None = None,
        theoretical_order: float = 2.0,
        asymptotic_tolerance: float = 0.1,
        distribution: NumericalErrorDistribution = "uniform",
    ) -> SolutionVerification:
        """Estimate the numerical (discretization) uncertainty of one QoI.

        The stage runs :func:`~probstat.uq.verify_solution` on QoI values that
        the user computed on a sequence of grids; it adds no numerics. Three
        grids give the Celik et al. GCI (or the oscillation bound), four or
        more the least-squares observed order. The expanded uncertainty ``U``
        is converted to ``u_num`` with ``distribution`` and stored on the
        study, so ``validate(..., numerical_uncertainty="verified")`` folds it
        into the ASME V&V 20 validation uncertainty
        ``u_val = sqrt(u_num^2 + u_input^2 + u_D^2)``. :meth:`predict` does not
        add it; pass the returned statement explicitly as ``numerical=`` to
        ``probstat.validation.predict_with_uncertainty_components`` instead.
        No forward-model evaluations are charged.

        Args:
            quantity: Model QoI the grid solutions belong to.
            grid_values: QoI value on each grid. With ``refinement_ratio`` they
                are ordered fine to coarse; with ``grid_sizes`` they pair with
                the sizes in any order.
            refinement_ratio: Constant ratio ``r > 1`` between consecutive
                grids (spacings ``1, r, r^2, ...``). Give this or
                ``grid_sizes``.
            grid_sizes: Representative grid spacings, one per value.
            safety_factor: ``F_s`` of the three-grid GCI (``None`` uses the
                library default 1.25). Ignored by the four-grid rule, which
                picks 1.25 or 3 from the order credibility.
            theoretical_order: Formal order of the discretization scheme.
            asymptotic_tolerance: Allowed deviation of the asymptotic-range
                ratio (at the theoretical order) from one.
            distribution: ``U`` to ``u_num`` conversion, ``"uniform"``
                (``U / sqrt(3)``) or ``"normal95"`` (``U / 1.96``).

        Returns:
            Solution-verification statement from the library call.

        Raises:
            KeyError: If ``quantity`` is not a model QoI.
            ValueError: If both or neither of ``refinement_ratio`` and
                ``grid_sizes`` are given, the ratio is not above one, or the
                grid data are rejected by :func:`~probstat.uq.verify_solution`.

        Equation:
            [EQ-VV-GCI-004], [EQ-VV-GCI-005], [EQ-VV-LSQ-002],
            [EQ-VV-OSC-001], [EQ-VV-UNUM-001].

        References:
            [REF-CELIK-ETAL-2008], [REF-ROACHE-1998, Chapters 4-5],
            [REF-ECA-HOEKSTRA-2014], [REF-OBERKAMPF-ROY-2025, Chapters 6-8].

        """
        self._model.quantity(quantity)
        values = tuple(grid_values)
        if (refinement_ratio is None) == (grid_sizes is None):
            raise ValueError("give exactly one of refinement_ratio and grid_sizes")
        if grid_sizes is None:
            ratio = require_finite_real(refinement_ratio, name="refinement_ratio")
            if ratio <= 1.0:
                raise ValueError("refinement_ratio must be greater than one")
            spacings = tuple(ratio**index for index in range(len(values)))
        else:
            spacings = tuple(grid_sizes)
        stream = self.stream("verification")
        result = verify_solution(
            quantity,
            spacings,
            values,
            theoretical_order=theoretical_order,
            safety_factor=(
                DEFAULT_SAFETY_FACTOR if safety_factor is None else safety_factor
            ),
            asymptotic_tolerance=asymptotic_tolerance,
        )
        u_num = result.standard_uncertainty(distribution)
        self._verifications[quantity] = (result, u_num)
        summary: dict[str, float | str] = {
            "quantity": quantity,
            "method": result.method,
            "grids": result.grid_count,
            "convergence": result.convergence.value,
            "fine_value": result.fine_value,
            "observed_order": (
                "n/a" if result.observed_order is None else result.observed_order
            ),
            "extrapolated": (
                "n/a" if result.extrapolated is None else result.extrapolated
            ),
            "safety_factor": result.safety_factor,
            "U": result.expanded_uncertainty,
            "u_num": u_num,
            "u_num.distribution": distribution,
        }
        study = result.study
        if isinstance(study, GridConvergenceResult):
            ratio_at_theory = study.asymptotic_ratio_at(theoretical_order)
            summary["gci_fine"] = study.gci_fine
            summary["asymptotic_ratio"] = ratio_at_theory
            summary["asymptotic_range"] = str(
                abs(ratio_at_theory - 1.0) <= asymptotic_tolerance
            )
        elif isinstance(study, LeastSquaresConvergence):
            summary["asymptotic_range"] = str(study.credible_order)
        else:
            summary["asymptotic_range"] = "n/a (oscillatory)"
        summary["reliable"] = str(result.reliable)
        summary["warnings"] = "; ".join(result.warnings) or "none"
        self._record(
            "verification",
            f"solution verification of {quantity} on {result.grid_count} grids",
            summary,
            0,
            stream,
            result,
        )
        return result

    def numerical_uncertainty(self, quantity: str) -> float:
        """Return ``u_num`` of the latest :meth:`verify` run for a QoI.

        Args:
            quantity: QoI name.

        Returns:
            Standard numerical uncertainty in QoI units.

        Raises:
            KeyError: If the QoI has not been verified in this study.

        Equation:
            [EQ-VV-UNUM-001].

        References:
            [REF-JCGM-100-2008, Sections 4.3.7 and 6.2].

        """
        if quantity not in self._verifications:
            raise KeyError(f"run verify({quantity!r}, ...) first")
        return self._verifications[quantity][1]

    @_stage(
        "validation",
        lambda study, args: args["sample_size"] * len(_sized(args["experiments"])),
    )
    def validate(
        self,
        quantity: str,
        experiments: Sequence[ValidationExperiment],
        *,
        tolerance: float,
        sample_size: int = 2000,
        significance: float = 0.05,
        coverage_levels: Sequence[float] = (0.5, 0.9),
        numerical_uncertainty: float
        | NumericalErrorBudget
        | Literal["verified"] = 0.0,
    ) -> ValidationAssessment:
        """Compare model predictions with validation experiments.

        The verdict is scoped: consistent or inconsistent with the data at the
        stated tolerance and significance inside the validation domain, never
        a general claim that the model is validated.

        Proper-scoring and calibration evidence is recorded next to the
        verdict without changing its criteria: the mean ensemble CRPS of each
        experiment (and over all observations) in QoI units, and randomized
        ensemble PIT values of every observation (stream ``validation/k/pit``)
        with their mean and the empirical coverage of central predictive
        intervals at ``coverage_levels`` (Clopper-Pearson 95% intervals).
        PIT values of replicates within one experiment share its predictive
        sample, so they are not independent cases.

        ``numerical_uncertainty`` enters only the ASME V&V 20 comparison of
        each experiment (``u_val = sqrt(u_num^2 + u_input^2 + u_D^2)``);
        ``"verified"`` uses the ``u_num`` stored by :meth:`verify` for the
        quantity. The area-metric verdict, u-pooling, CRPS, and PIT do not
        depend on it. A non-zero declaration adds ``u_num`` to the summary.

        Args:
            quantity: QoI compared with the observations.
            experiments: Validation experiments with conditions and data.
            tolerance: Area-metric tolerance in QoI units.
            sample_size: Predictive samples per experiment.
            significance: Significance level of the u-pooling test.
            coverage_levels: Nominal central coverages reported from the PIT
                values (each in ``(0, 1)``).
            numerical_uncertainty: ``u_num``, a verification error budget, or
                ``"verified"`` for the value stored by :meth:`verify`.

        Returns:
            Validation assessment with per-experiment and pooled metrics.

        Raises:
            KeyError: If ``"verified"`` is requested before :meth:`verify`
                was run for the quantity.
            ValueError: If ``numerical_uncertainty`` is another string.

        Equation:
            [EQ-VAL-AREA-001], [EQ-VAL-SCORE-CRPS-002], [EQ-VAL-CAL-PIT-002],
            [EQ-VAL-CAL-COV-001], [EQ-VAL-ASME-001].

        References:
            [REF-FERSON-OBERKAMPF-GINZBURG-2008], [REF-OBERKAMPF-ROY-2025, Part IV],
            [REF-GNEITING-RAFTERY-2007], [REF-GNEITING-BALABDAOUI-RAFTERY-2007],
            [REF-CLOPPER-PEARSON-1934], [REF-ASME-VV20-2009].

        """
        declared = numerical_uncertainty
        u_num: float | NumericalErrorBudget
        if isinstance(declared, str):
            if declared != "verified":
                raise ValueError("numerical_uncertainty string must be 'verified'")
            u_num = self.numerical_uncertainty(quantity)
        else:
            u_num = declared
        stream = self.stream("validation")
        before = self._model.evaluations
        result = assess_validation(
            self._model,
            quantity,
            experiments,
            tolerance=tolerance,
            sample_size=sample_size,
            significance=significance,
            seed=stream,
            numerical_uncertainty=u_num,
        )
        summary: dict[str, float | str] = {
            "quantity": quantity,
            "consistent": str(result.consistent),
            "u_pooling_area": result.u_pooling.area,
            "log_bayes_factor": result.log_bayes_factor,
        }
        for index, item in enumerate(result.experiments, start=1):
            # Index-qualified keys keep every experiment even if names repeat.
            summary[f"area[{index}:{item.experiment.name}]"] = item.area
        u_declared = result.experiments[0].asme.numerical_uncertainty
        if u_declared > 0.0:
            summary["u_num"] = u_declared
            summary["u_num.source"] = (
                "verify stage" if isinstance(declared, str) else "declared"
            )
        ensembles: list[tuple[float, ...]] = []
        observations: list[float] = []
        scores: list[float] = []
        for index, item in enumerate(result.experiments, start=1):
            summary[f"crps[{index}:{item.experiment.name}]"] = item.crps
            for value in item.experiment.observations:
                ensembles.append(item.observable_predictions)
                observations.append(value)
                scores.append(crps_ensemble(item.observable_predictions, value))
        summary["crps.mean"] = math.fsum(scores) / len(scores)
        pit = ensemble_pit(ensembles, observations, seed=stream.child("pit"))
        summary["pit.mean"] = math.fsum(pit) / len(pit)
        for entry in pit_coverage(pit, coverage_levels):
            key = f"coverage[{100.0 * entry.nominal:.6g}%]"
            summary[key] = entry.rate
            summary[f"{key}.interval"] = (
                f"{entry.covered}/{entry.size}, "
                f"[{entry.interval.lower:.3f}, {entry.interval.upper:.3f}]"
            )
        self._record(
            "validation",
            f"validation of {quantity} against {len(result.experiments)} experiments",
            summary,
            self._model.evaluations - before,
            stream,
            result,
        )
        return result
