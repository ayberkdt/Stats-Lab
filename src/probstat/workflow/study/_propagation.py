"""Forward-propagation stages: fixed-size and accuracy-targeted.

References:
    [REF-SMITH-UQ-2024, Chapters 3-4], [REF-L-ECUYER-2018],
    [REF-ROBERT-CASELLA-2004, Chapters 3-4].

"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)
from probstat.inference import student_t_quantile
from probstat.linalg import Vector
from probstat.propagation import PropagationStudy, propagate

from ._base import _stage, _StudyBase
from ._context import _Accumulators
from ._records import TargetedPropagation


class _PropagationStages(_StudyBase):
    """Propagation stages of :class:`~probstat.workflow.study.UQStudy`.

    References:
        [REF-SMITH-UQ-2024, Chapters 3-4].

    """

    @_stage(
        "propagation",
        lambda study, args: args["size"] * (args["replicates"] or 1),
    )
    def propagate(
        self,
        size: int = 4096,
        *,
        design: str = "sobol",
        replicates: int | None = None,
        batch: int | None = None,
    ) -> PropagationStudy:
        """Propagate the joint input law through every QoI.

        With ``replicates=r`` the design is randomized ``r`` times (cost
        ``r * size``) and the summary adds Student-t replicate intervals of
        each mean, the honest error bar for LHS and quasi-random designs.

        Args:
            size: Number of input realizations (per replicate).
            design: Sampling design name.
            replicates: Optional number of independent randomizations.
            batch: Chunk size for a vectorised model or executor jobs.

        Returns:
            Propagation study.

        Equation:
            [EQ-UQ-001], [EQ-MC-MEAN-001], [EQ-PROP-RQMC-CI-001].

        References:
            [REF-SMITH-UQ-2024, Chapters 3-4], [REF-L-ECUYER-2018].

        """
        stream = self.stream("propagation")
        study = propagate(
            self._model,
            size,
            design=design,
            seed=stream,
            replicates=replicates,
            batch=batch,
        )
        summary: dict[str, float | str] = {"design": design, "size": size}
        if replicates is not None:
            summary["replicates"] = float(replicates)
        for name, output in study.outputs.items():
            summary[f"{name}.mean"] = output.mean
            summary[f"{name}.std"] = output.standard_deviation
            summary[f"{name}.q05"] = output.quantile(0.05)
            summary[f"{name}.q95"] = output.quantile(0.95)
            if replicates is not None and len(output.replicate_means()) >= 2:
                interval = output.mean_interval(0.95)
                summary[f"{name}.mean 95% low (replicates)"] = interval.lower
                summary[f"{name}.mean 95% high (replicates)"] = interval.upper
        label = f"{design} propagation, {size} samples"
        if replicates is not None:
            label = f"{design} propagation, {replicates} x {size} samples"
        self._record(
            "propagation",
            label,
            summary,
            study.model_evaluations,
            stream,
            study,
        )
        return study

    def _sequential_sample(
        self,
        quantity: str,
        levels: tuple[float, ...],
        event: Callable[[float], bool] | None,
        draw: Callable[[int, int], tuple[Vector, ...]],
        budget: int,
        batch: int,
        stop: Callable[[_Accumulators], bool] | None,
    ) -> tuple[_Accumulators, int, bool]:
        """Push batches of model outputs into accumulators.

        Args:
            quantity: QoI name.
            levels: P² quantile levels.
            event: Event indicator, or ``None`` in mean mode.
            draw: ``draw(k, m)`` returns the ``k``-th batch of ``m`` inputs.
            budget: Maximum number of input points.
            batch: Points per batch.
            stop: Stopping rule checked after each batch (``None``: never).

        Returns:
            Accumulators, batches drawn, and whether ``stop`` fired.

        References:
            [REF-WELFORD-1962], [REF-JAIN-CHLAMTAC-1985]. Evaluates
            [EQ-STAT-STREAM-007].

        """
        accumulators = _Accumulators(levels, event)
        drawn = 0
        batches = 0
        while drawn < budget:
            count = min(batch, budget - drawn)
            points = draw(batches, count)
            batches += 1
            drawn += count
            outputs = self._model.evaluate_many(points)
            for value in outputs[quantity]:
                accumulators.push(value)
            if stop is not None and stop(accumulators):
                return accumulators, batches, True
        return accumulators, batches, False

    @_stage("sequential_propagation")
    def propagate_until(
        self,
        quantity: str,
        *,
        target_relative_error: float = 0.01,
        threshold: float | None = None,
        failure: Literal["above", "below"] = "above",
        batch: int = 256,
        max_samples: int = 200_000,
        quantiles: Sequence[float] = (0.05, 0.5, 0.95),
        min_samples: int = 100,
        confidence: float = 0.95,
        final: Literal["fresh", "reuse"] = "fresh",
    ) -> TargetedPropagation:
        """Size a Monte Carlo propagation by a relative-error target.

        **Pilot.** Batches of ``batch`` joint input draws (batch ``k`` on the
        child stream ``stream.child("batch", k)``) are evaluated and pushed
        into a :class:`~probstat.statistics.RunningMoments`, one
        :class:`~probstat.statistics.P2Quantile` per level, and, with a
        threshold, an :class:`~probstat.statistics.ExceedanceCounter`. After
        every batch the relative standard error of the controlled estimate is
        checked: ``s / (sqrt(n) |mean|)`` for the mean, or
        ``sqrt((1 - p) / (n p))`` for the event probability. The pilot stops
        at the first batch meeting the target (after ``min_samples``) or at
        ``max_samples`` (``converged=False``).

        **Final sample.** Stopping when the running estimate *looks* precise
        is data-dependent, so the pilot's own estimates are biased and its
        intervals under-cover (optional stopping). With ``final="fresh"``
        (default) the pilot only chooses ``n``; a fresh, independent sample
        of ``n`` draws (stream ``stream.child("final", k)``) gives every
        reported estimate and interval. Given ``n`` it is an ordinary
        fixed-size sample, so the estimates are unbiased and the intervals
        have their nominal coverage; the price is the extra ``n`` runs (at
        most ``2 * max_samples`` in total). ``final="reuse"`` reports the
        pilot itself (the behaviour before version 0.16) and is biased.

        The event matches :meth:`~probstat.propagation.OutputDistribution.exceedance`:
        ``failure="above"`` counts ``Y > t``; ``"below"`` counts the
        complement ``Y <= t``. Mean intervals are Student-t, probability
        intervals Wilson, both at ``confidence``. No output is stored. Under
        a ``"skip"``/``"record"`` failure policy failed runs are dropped and
        reported; ``n`` then counts successful runs of the pilot.

        Args:
            quantity: QoI name.
            target_relative_error: Positive relative standard-error target.
            threshold: Event threshold; switches control to the probability.
            failure: ``"above"`` (``Y > t``) or ``"below"`` (``Y <= t``).
            batch: Draws between checks, at least two.
            max_samples: Pilot draw budget, at least two.
            quantiles: Quantile levels strictly between zero and one.
            min_samples: Draws before the target may be declared met (capped
                at ``max_samples``).
            confidence: Interval confidence strictly inside ``(0, 1)``.
            final: ``"fresh"`` (unbiased, default) or ``"reuse"`` (legacy,
                biased by optional stopping).

        Returns:
            Targeted propagation result; the stage charges pilot plus final
            evaluations.

        Raises:
            TypeError: If numeric arguments have invalid types.
            ValueError: If ``quantity`` is unknown or an argument is out of
                its domain.

        Equation:
            [EQ-UQ-001], [EQ-STAT-STREAM-007], [EQ-MC-SE-001],
            [EQ-MC-PROB-001], [EQ-INF-CI-WILSON-001], [EQ-STAT-STREAM-005],
            [EQ-WF-SEQ-FRESH-001].

        References:
            [REF-ROBERT-CASELLA-2004, Chapters 3-4], [REF-WELFORD-1962],
            [REF-JAIN-CHLAMTAC-1985], [REF-WILSON-1927], [REF-STEIN-1945].

        """
        if quantity not in self._model.quantity_names:
            raise ValueError(f"{quantity!r} is not a quantity of the study model")
        target = require_finite_real(
            target_relative_error, name="target_relative_error"
        )
        if target <= 0.0:
            raise ValueError("target_relative_error must be positive")
        size = require_non_negative_integer(batch, name="batch")
        budget = require_non_negative_integer(max_samples, name="max_samples")
        if size < 2:
            raise ValueError("batch must be at least two")
        if budget < 2:
            raise ValueError("max_samples must be at least two")
        if failure not in ("above", "below"):
            raise ValueError("failure must be 'above' or 'below'")
        if final not in ("fresh", "reuse"):
            raise ValueError("final must be 'fresh' or 'reuse'")
        if isinstance(min_samples, bool) or not isinstance(min_samples, int):
            raise TypeError("min_samples must be an integer")
        if min_samples < 2:
            raise ValueError("min_samples must be at least two")
        level = require_finite_real(confidence, name="confidence")
        if not 0.0 < level < 1.0:
            raise ValueError("confidence must lie strictly between 0 and 1")
        minimum = min(min_samples, budget)
        levels = tuple(require_finite_real(p, name="quantile level") for p in quantiles)
        if any(not 0.0 < p < 1.0 for p in levels):
            raise ValueError("quantile levels must lie strictly between 0 and 1")
        limit = (
            None
            if threshold is None
            else require_finite_real(threshold, name="threshold")
        )
        event: Callable[[float], bool] | None = None
        if limit is not None:
            cut = limit
            if failure == "above":

                def event(value: float) -> bool:
                    """Return whether ``value > threshold``.

                    References:
                        [EQ-PROP-THRESH-001], [REF-SMITH-UQ-2024, Chapter 4].

                    """
                    return value > cut

            else:

                def event(value: float) -> bool:
                    """Return whether ``value <= threshold``.

                    References:
                        [EQ-PROP-THRESH-001], [REF-SMITH-UQ-2024, Chapter 4].

                    """
                    return value <= cut

        stream = self.stream("sequential_propagation")
        inputs = self._model.inputs
        before = self._model.evaluations

        def stop(accumulators: _Accumulators) -> bool:
            """Return whether the pilot has met the target.

            References:
                [EQ-STAT-STREAM-007], [REF-ROBERT-CASELLA-2004, Chapter 4].

            """
            return (
                accumulators.relative_error() <= target
                and accumulators.count >= minimum
            )

        def pilot_draw(index: int, count: int) -> tuple[Vector, ...]:
            """Return pilot batch ``index``.

            References:
                [REF-DEVROYE-1986, Section II.2]. No equation is evaluated.

            """
            return inputs.sample(count, seed=stream.child("batch", index))

        def final_draw(index: int, count: int) -> tuple[Vector, ...]:
            """Return final-sample batch ``index``.

            References:
                [REF-DEVROYE-1986, Section II.2]. No equation is evaluated.

            """
            return inputs.sample(count, seed=stream.child("final", index))

        pilot, batches, converged = self._sequential_sample(
            quantity, levels, event, pilot_draw, budget, size, stop
        )
        reported = pilot
        if final == "fresh":
            if pilot.count < 2:
                raise ValueError(
                    "the pilot produced fewer than two successful runs; "
                    "no final sample can be sized"
                )
            reported, _, _ = self._sequential_sample(
                quantity, levels, event, final_draw, pilot.count, size, None
            )
        if reported.count < 2:
            raise ValueError("fewer than two successful runs; no estimate is possible")
        relative = reported.relative_error()
        mean = reported.moments.mean
        standard_error = reported.moments.standard_error()
        half = student_t_quantile(0.5 + 0.5 * level, reported.count - 1) * (
            standard_error
        )
        counter = reported.counter
        probability: float | None = None
        probability_se: float | None = None
        interval: tuple[float, float] | None = None
        events = 0
        if counter is not None:
            probability = counter.proportion
            probability_se = counter.standard_error()
            interval = counter.wilson_interval(confidence=level)
            events = counter.events
        spent = self._model.evaluations - before
        result = TargetedPropagation(
            quantity=quantity,
            controlled="mean" if counter is None else "probability",
            target_relative_error=target,
            relative_error=relative,
            converged=converged,
            sample_size=reported.count,
            batches=batches,
            mean=mean,
            standard_error=standard_error,
            quantiles=reported.quantiles(),
            threshold=limit,
            failure=failure,
            probability=probability,
            probability_standard_error=probability_se,
            probability_interval=interval,
            events=events,
            confidence=level,
            final=final,
            pilot_size=pilot.count,
            pilot_relative_error=pilot.relative_error(),
            mean_interval=(mean - half, mean + half),
            failed_evaluations=self._model.failed_evaluations
            - self._contexts[-1].failed_before
            if self._contexts
            else 0,
        )
        percent = f"{100.0 * level:g}%"
        summary: dict[str, float | str] = {
            "quantity": quantity,
            "controlled": result.controlled,
            "target relative error": target,
            "relative error": relative,
            "pilot relative error": result.pilot_relative_error,
            "converged": "yes" if converged else "no (max_samples reached)",
            "samples": float(result.sample_size),
            "pilot samples": float(result.pilot_size),
            "batches": float(batches),
            "final sample": (
                "fresh (independent of the stopping rule)"
                if final == "fresh"
                else "reused pilot (optional-stopping bias)"
            ),
            f"{quantity}.mean": result.mean,
            f"{quantity}.se": result.standard_error,
            f"{quantity}.mean {percent} low": mean - half,
            f"{quantity}.mean {percent} high": mean + half,
        }
        for q_level, estimate in result.quantiles:
            summary[f"{quantity}.q[{q_level:g}] (P2)"] = estimate
        if limit is not None and probability is not None and interval is not None:
            relation = ">" if failure == "above" else "<="
            name = f"P[{quantity} {relation} {limit:g}]"
            summary[name] = probability
            summary[f"{name} se"] = probability_se if probability_se else 0.0
            summary[f"{name} Wilson {percent} lower"] = interval[0]
            summary[f"{name} Wilson {percent} upper"] = interval[1]
        if not converged:
            self._warn(
                f"propagate_until({quantity!r}) reached max_samples={budget} "
                f"before the target relative error {target:g}"
            )
        if final == "reuse":
            self._warn(
                "propagate_until(final='reuse') reports the pilot sample, whose "
                "estimates carry optional-stopping bias; use final='fresh'"
            )
        self._record(
            "sequential_propagation",
            f"Monte Carlo of {quantity} until relative error {target:g} "
            f"on the {result.controlled}",
            summary,
            spent,
            stream,
            result,
        )
        return result
