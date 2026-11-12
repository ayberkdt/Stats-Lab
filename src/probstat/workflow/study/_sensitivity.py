"""Sensitivity stages: Sobol indices, Shapley effects, screening, information.

References:
    [REF-SALTELLI-2010], [REF-OWEN-2014], [REF-MORRIS-1991],
    [REF-KRASKOV-STOGBAUER-GRASSBERGER-2004].

"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, cast

from probstat.foundations.random_streams import RandomStream
from probstat.information import InformationUnit, ksg_mutual_information
from probstat.modeling import ProbabilisticModel
from probstat.sensitivity import (
    ModelSobolResult,
    MorrisResult,
    MultiQuantitySobolResult,
    ShapleyResult,
    morris_screening,
    shapley_effects,
    sobol_analysis,
    sobol_analysis_all,
)

from ._base import _stage, _StudyBase
from ._records import InformationSensitivity


class _SensitivityStages(_StudyBase):
    """Sensitivity stages of :class:`~probstat.workflow.study.UQStudy`.

    References:
        [REF-SALTELLI-2010], [REF-MORRIS-1991].

    """

    def _shared_sobol(
        self, size: int, design: str, bootstrap: int
    ) -> tuple[MultiQuantitySobolResult, RandomStream] | None:
        """Return the shared Sobol design of these settings, if one was run.

        The key is the current input-law hash with ``(size, design,
        bootstrap)``, so a change of input law never reuses an old design.

        References:
            [REF-SALTELLI-2010]. No equation is evaluated.

        """
        designs = getattr(self, "_sobol_designs", None)
        if designs is None:
            return None
        found: tuple[MultiQuantitySobolResult, RandomStream] | None = designs.get(
            (self._law_hash, size, design, bootstrap)
        )
        return found

    def _sensitivity_cost(self, arguments: Mapping[str, Any]) -> int:
        """Return the known cost of a sensitivity stage (zero when shared).

        References:
            [REF-SALTELLI-2010]. No equation is evaluated.

        """
        size = arguments["size"]
        shared = self._shared_sobol(size, arguments["design"], arguments["bootstrap"])
        if shared is not None and arguments["quantity"] in shared[0].results:
            return 0
        return int(size) * (self._model.dimension + 2)

    @_stage(
        "sensitivity",
        lambda study, args: cast(_SensitivityStages, study)._sensitivity_cost(args),
    )
    def sensitivity(
        self,
        quantity: str,
        size: int = 1024,
        *,
        design: str = "sobol",
        bootstrap: int = 200,
    ) -> ModelSobolResult:
        """Estimate Sobol indices of one QoI, with intervals, from a shared design.

        One pick-freeze design gives the indices of every QoI
        (:func:`~probstat.sensitivity.sobol_analysis_all`): each model call
        returns all QoIs, so the first call costs ``size (d + 2)`` runs and a
        later call for another QoI with the same ``size``, ``design``,
        ``bootstrap``, and input law reuses that design at no cost (its
        record carries the stream seed of the stage that ran the design). The
        indices of the requested QoI equal those of
        :func:`~probstat.sensitivity.sobol_analysis` on this stage's stream.
        A QoI with zero variance is skipped for the others and raises only
        when requested; the repeat that skips it is replayed from the runs of
        the design, so it costs no extra simulator run.

        Intervals follow ``interval_method`` of the result: ``"bootstrap"``
        (``bootstrap`` row resamples) only for ``design="mc"``; for
        quasi-random and Latin-hypercube designs any ``bootstrap > 0``
        requests ``R = 8`` independent randomizations of ``size // 8`` rows
        and Student-t replicate intervals, because resampling the rows of a
        structured design gives wrong intervals (ADR-0065). The summary
        states the method, ``R``, and the interval limits.

        Args:
            quantity: QoI name.
            size: Design rows (cost ``size (d + 2)`` unless shared).
            design: Unit-hypercube design name.
            bootstrap: Interval request; bootstrap replicates for ``"mc"``.

        Returns:
            Sobol result of ``quantity``.

        Raises:
            KeyError: For an unknown QoI.
            ValueError: For invalid settings, dependent inputs, or a requested
                QoI with zero variance.

        Equation:
            [EQ-SENS-FIRST-001], [EQ-SENS-TOTAL-001], [EQ-SENS-BOOT-001],
            [EQ-SENS-RQMC-001].

        References:
            [REF-SALTELLI-2010], [REF-L-ECUYER-2018].

        """
        model = self._model
        model.quantity(quantity)
        stream = self.stream("sensitivity")
        shared = self._shared_sobol(size, design, bootstrap)
        if shared is not None and quantity in shared[0].results:
            multi, design_stream = shared
            source = "design shared with an earlier sensitivity stage, no new runs"
            evaluations = 0
        else:
            design_stream = stream
            source = "new design"
            before = model.evaluations
            multi = _shared_design(model, quantity, size, design, bootstrap, stream)
            evaluations = model.evaluations - before
            designs = getattr(self, "_sobol_designs", None)
            if designs is None:
                designs = {}
                self._sobol_designs = designs
            designs[(self._law_hash, size, design, bootstrap)] = (multi, stream)
        result = multi[quantity]
        summary: dict[str, float | str] = {"quantity": quantity}
        for name, first, total in zip(
            result.input_names, result.first_order, result.total_effect, strict=True
        ):
            summary[f"S[{name}]"] = first.estimate
            summary[f"ST[{name}]"] = total.estimate
        summary["design"] = f"{result.design} ({source})"
        summary["intervals"] = _interval_text(result)
        if result.interval_method != "none":
            for name, first, total in zip(
                result.input_names, result.first_order, result.total_effect,
                strict=True,
            ):  # fmt: skip
                summary[f"S[{name}].low"] = first.lower
                summary[f"S[{name}].high"] = first.upper
                summary[f"ST[{name}].low"] = total.lower
                summary[f"ST[{name}].high"] = total.upper
        others = [name for name in multi.quantities if name != quantity]
        if others:
            summary["also estimated by this design"] = ", ".join(others)
        self._record(
            "sensitivity",
            f"Sobol indices of {quantity}",
            summary,
            evaluations,
            design_stream,
            result,
        )
        return result

    @_stage("shapley")
    def shapley(
        self,
        quantity: str,
        *,
        permutations: int = 200,
        inner_samples: int = 3,
        variance_samples: int = 1000,
    ) -> ShapleyResult:
        """Estimate Shapley effects, valid also for dependent inputs.

        Unlike Sobol indices, Shapley effects share interaction and dependence
        contributions fairly among inputs and always sum to one.

        Args:
            quantity: QoI name.
            permutations: Random permutations when exact enumeration is not
                used (inputs above six).
            inner_samples: Conditional draws per outer sample.
            variance_samples: Draws for the output variance.

        Returns:
            Shapley result with approximate standard errors.

        Equation:
            [EQ-SENS-SHAPLEY-001].

        References:
            [REF-OWEN-2014], [REF-SONG-NELSON-STAUM-2016].

        """
        stream = self.stream("shapley")
        result = shapley_effects(
            self._model,
            quantity,
            permutations=permutations,
            inner_samples=inner_samples,
            variance_samples=variance_samples,
            seed=stream,
        )
        summary: dict[str, float | str] = {"quantity": quantity}
        for name, value in result.as_mapping().items():
            summary[f"Sh[{name}]"] = value
        self._record(
            "shapley",
            f"Shapley effects of {quantity}",
            summary,
            result.evaluations,
            stream,
            result,
        )
        return result

    @_stage(
        "screening",
        lambda study, args: args["trajectories"] * (study.model.dimension + 1),
    )
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

    @_stage(
        "information",
        lambda study, args: study._fresh_cost(args, args["max_points"]),
    )
    def information_sensitivity(
        self,
        quantity: str,
        *,
        max_points: int = 512,
        k: int = 3,
        unit: InformationUnit = "nats",
        size: int | None = None,
        design: str = "mc",
        reuse: bool = False,
    ) -> InformationSensitivity:
        """Estimate the mutual information between each input and one QoI.

        ``I(X_i; Y)`` is zero exactly when ``Y`` is independent of ``X_i``
        and is invariant under monotone transformations of either, so it
        sees non-monotone and higher-moment dependence that variance-based
        indices can miss, and it needs no independence between inputs. It is
        a moment-independent complement to Sobol indices, not a variance
        share: values are in nats (or bits) and do not sum to anything.

        The sample is ``size`` (default ``max_points``) fresh plain Monte
        Carlo runs drawn from this stage's stream, so the result does not
        depend on the order of stages; ``reuse=True`` uses the first
        ``max_points`` runs of the latest propagation instead when it is a
        plain Monte Carlo sample (quasi-random points are not independent and
        bias the KSG estimator). The KSG estimator costs ``O(n^2)`` per
        input, hence the cap.

        Args:
            quantity: QoI name.
            max_points: Maximum number of runs entering the estimator.
            k: KSG neighbour order.
            unit: ``"nats"`` or ``"bits"``.
            size: Fresh sample size (default ``max_points``).
            design: Sampling design of a fresh sample; only ``"mc"``.
            reuse: Reuse the latest plain Monte Carlo propagation.

        Returns:
            Mutual information per input.

        Raises:
            ValueError: If the quantity or ``max_points`` is invalid, or the
                sample has duplicated (input, output) pairs.

        Equation:
            [EQ-INFO-MI-001], [EQ-INFO-KSG-001].

        References:
            [REF-KRASKOV-STOGBAUER-GRASSBERGER-2004],
            [REF-COVER-THOMAS-2006, Chapter 8].

        """
        if isinstance(max_points, bool) or not isinstance(max_points, int):
            raise ValueError("max_points must be an integer")
        if max_points < 2:
            raise ValueError("max_points must be at least two")
        stream = self.stream("information")
        sample, evaluations, source = self._output_sample(
            stream, quantity, size, design, max_points, reuse
        )
        outputs = sample[quantity].values[:max_points]
        count = len(outputs)
        estimates = tuple(
            ksg_mutual_information(
                sample.input_column(name)[:count], outputs, k=k, unit=unit
            )
            for name in sample.input_names
        )
        result = InformationSensitivity(
            quantity, sample.input_names, estimates, k, count, unit, source
        )
        summary: dict[str, float | str] = {
            "quantity": quantity,
            "unit": unit,
            "k": k,
            "sample_size": count,
            "source": source,
        }
        for name, value in zip(sample.input_names, estimates, strict=True):
            summary[f"MI[{name}]"] = value
        summary["ranking"] = " > ".join(result.ranking())
        self._record(
            "information",
            f"mutual information of {quantity} with each input",
            summary,
            evaluations,
            stream,
            result,
        )
        return result


def _interval_text(result: ModelSobolResult) -> str:
    """Describe how the intervals of a Sobol result were obtained.

    References:
        [EQ-SENS-BOOT-001], [EQ-SENS-RQMC-001], [REF-L-ECUYER-2018].

    """
    if result.interval_method == "replicates":
        rows = result.result.sample_size // result.replicates
        return (
            f"{result.confidence:.0%} Student-t over {result.replicates} "
            f"independent randomizations of {rows} rows"
        )
    if result.interval_method == "bootstrap":
        return (
            f"{result.confidence:.0%} percentile over {result.bootstrap} row "
            "bootstrap resamples"
        )
    return "none"


def _shared_design(
    model: ProbabilisticModel,
    quantity: str,
    size: int,
    design: str,
    bootstrap: int,
    stream: RandomStream,
) -> MultiQuantitySobolResult:
    """Return the Sobol indices of every QoI from one pick-freeze design.

    All QoIs are estimated by one
    :func:`~probstat.sensitivity.sobol_analysis_all` call on a replay view of
    ``model``: each new input point is run once on ``model`` (its counter,
    cache, budget, and failure policy apply) and its outputs are kept. If the
    joint call fails (typically a constant QoI, whose indices are undefined),
    each QoI is estimated alone on the same stream; the design is identical,
    so every point is replayed and no model run is repeated. A QoI that fails
    alone is dropped unless it is ``quantity``, whose error is raised. The
    shared evaluation count is that of ``model``.

    References:
        [REF-SALTELLI-2010], [EQ-SENS-FIRST-001], [EQ-SENS-TOTAL-001].

    """
    names = model.inputs.names
    runs: dict[tuple[float, ...], Mapping[str, float]] = {}

    def replay(parameters: Mapping[str, Any]) -> Mapping[str, float]:
        """Return the outputs at one input point, running the model once.

        References:
            [REF-SALTELLI-2010]. No equation is evaluated.

        """
        key = tuple(float(parameters[name]) for name in names)
        known = runs.get(key)
        if known is None:
            known = model.evaluate({name: parameters[name] for name in names})
            runs[key] = known
        return known

    view = ProbabilisticModel(model.inputs, replay, model.quantities, name=model.name)
    before = model.evaluations
    try:
        joint = sobol_analysis_all(
            view,
            size,
            quantities=model.quantity_names,
            design=design,
            seed=stream,
            bootstrap=bootstrap,
        )
        return MultiQuantitySobolResult(joint.results, model.evaluations - before)
    except ValueError:
        pass
    results: dict[str, ModelSobolResult] = {}
    for name in model.quantity_names:
        try:
            results[name] = sobol_analysis(
                view, name, size, design=design, seed=stream, bootstrap=bootstrap
            )
        except ValueError:
            if name == quantity:
                raise
    return MultiQuantitySobolResult(
        MappingProxyType(results), model.evaluations - before
    )
