"""Sensitivity stages: Sobol indices, Shapley effects, screening, information.

References:
    [REF-SALTELLI-2010], [REF-OWEN-2014], [REF-MORRIS-1991],
    [REF-KRASKOV-STOGBAUER-GRASSBERGER-2004].

"""

from __future__ import annotations

from probstat.information import InformationUnit, ksg_mutual_information
from probstat.sensitivity import (
    ModelSobolResult,
    MorrisResult,
    ShapleyResult,
    morris_screening,
    shapley_effects,
    sobol_analysis,
)

from ._base import _stage, _StudyBase
from ._records import InformationSensitivity


class _SensitivityStages(_StudyBase):
    """Sensitivity stages of :class:`~probstat.workflow.study.UQStudy`.

    References:
        [REF-SALTELLI-2010], [REF-MORRIS-1991].

    """

    @_stage(
        "sensitivity",
        lambda study, args: args["size"] * (study.model.dimension + 2),
    )
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
