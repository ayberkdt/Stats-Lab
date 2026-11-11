"""Bayesian experimental-design stage.

References:
    [REF-CHALONER-VERDINELLI-1995], [REF-HUAN-MARZOUK-2013].

"""

from __future__ import annotations

from collections.abc import Sequence
from types import MappingProxyType

from probstat.design import (
    BayesianDesignProblem,
    Condition,
    expected_information_gain,
    greedy_design,
)
from probstat.inverse import Experiment, NoiseModel

from ._base import _stage, _StudyBase
from ._context import _format_condition
from ._records import DesignRecommendation


class _DesignStages(_StudyBase):
    """Experimental-design stage of the study.

    References:
        [REF-CHALONER-VERDINELLI-1995], [REF-HUAN-MARZOUK-2013].

    """

    @_stage("design")
    def design(
        self,
        candidates: Sequence[Condition],
        size: int = 1,
        *,
        noise: Sequence[NoiseModel] | None = None,
        include_experiments: bool = True,
        allow_repeats: bool = True,
        method: str = "laplace",
        samples: int = 32,
        outer: int = 1000,
        inner: int = 200,
        reuse_samples: bool = True,
    ) -> DesignRecommendation:
        """Choose new experimental conditions by expected information gain.

        The design problem reuses the calibration problem of the latest
        :meth:`calibrate` (prior = model input law, its Gaussian noise
        models) or, before any calibration, the ``noise`` models given here.
        With ``include_experiments`` the calibration's experimental conditions
        form the baseline, so each greedy step maximizes the expected
        conditional gain given data at those conditions. The EIG is a
        preposterior quantity under the prior: it does not condition on the
        observed values. Conditions are chosen greedily
        (stream ``design/greedy``); the full design is then re-estimated on an
        independent stream (``design/final``) to report its EIG with a
        standard error. After :meth:`calibrate_with_discrepancy` the
        Kennedy-O'Hagan experiments form the baseline, but ``noise`` is
        required because the design problem has no discrepancy term.

        Args:
            candidates: Candidate conditions (deterministic model inputs).
            size: Number of new conditions to choose.
            noise: Known-scale Gaussian noise models; overrides the
                calibration's when given.
            include_experiments: Use the calibration experiments as baseline.
            allow_repeats: Allow a candidate to be chosen more than once.
            method: ``"linearized"``, ``"laplace"``, or ``"nested"``.
            samples: Prior draws for ``"laplace"``.
            outer: Outer samples for ``"nested"``.
            inner: Inner samples for ``"nested"`` without reuse.
            reuse_samples: Sample reuse for ``"nested"``.

        Returns:
            Greedy selection and the final EIG estimate.

        Raises:
            ValueError: If there is neither a plain calibration problem nor
                ``noise``, or the design arguments are invalid.

        Equation:
            [EQ-OED-EIG-001], [EQ-OED-GREEDY-001], [EQ-OED-LAP-001].

        References:
            [REF-CHALONER-VERDINELLI-1995], [REF-HUAN-MARZOUK-2013].

        """
        plain = self._problem if self._active == "plain" else None
        if noise is not None:
            problem = BayesianDesignProblem(self._model, tuple(noise))
        elif plain is not None:
            problem = BayesianDesignProblem.from_calibration(plain)
        elif self._active == "discrepancy":
            raise ValueError(
                "design() after calibrate_with_discrepancy() needs known-scale "
                "noise models via noise= (the design problem has no discrepancy)"
            )
        else:
            raise ValueError(
                "design() needs a calibration problem: run calibrate() first "
                "or pass known-scale noise models via noise="
            )
        performed: tuple[Experiment, ...] = ()
        if plain is not None:
            performed = plain.experiments
        elif self._active == "discrepancy" and self._discrepancy is not None:
            performed = self._discrepancy.calibration.experiments
        baseline: tuple[Condition, ...] = ()
        if include_experiments:
            baseline = tuple(
                MappingProxyType(dict(item.conditions)) for item in performed
            )
        pool: tuple[Condition, ...] = tuple(
            MappingProxyType(dict(item)) for item in candidates
        )
        stream = self.stream("design")
        before = self._model.evaluations
        selection = greedy_design(
            problem,
            pool,
            size,
            baseline=baseline,
            allow_repeats=allow_repeats,
            method=method,
            samples=samples,
            outer=outer,
            inner=inner,
            reuse_samples=reuse_samples,
            seed=stream.child("greedy"),
        )
        estimate = expected_information_gain(
            problem,
            baseline + selection.selected,
            method=method,
            samples=samples,
            outer=outer,
            inner=inner,
            reuse_samples=reuse_samples,
            seed=stream.child("final"),
        )
        result = DesignRecommendation(selection, estimate, baseline, pool)
        summary: dict[str, float | str] = {
            "method": method,
            "candidates": len(pool),
            "baseline_conditions": len(baseline),
            "baseline_eig": selection.baseline_information_gain,
        }
        for step, (condition, gain, increment) in enumerate(
            zip(
                selection.selected,
                selection.information_gains,
                selection.increments,
                strict=True,
            ),
            start=1,
        ):
            summary[f"condition[{step}]"] = _format_condition(condition)
            summary[f"eig[{step}]"] = gain
            summary[f"gain[{step}]"] = increment
        summary["eig"] = estimate.value
        summary["eig_se"] = estimate.standard_error
        summary["eig_bias"] = estimate.bias
        self._record(
            "design",
            f"greedy {method} design of {len(selection.selected)} new conditions",
            summary,
            self._model.evaluations - before,
            stream,
            result,
        )
        return result
