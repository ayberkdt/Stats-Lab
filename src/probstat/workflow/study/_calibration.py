"""Calibration, prediction, posterior adoption, and emulation stages.

References:
    [REF-KENNEDY-OHAGAN-2001], [REF-GELMAN-ETAL-2013, Chapter 11],
    [REF-SUDRET-2007, Chapter 4].

"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Self

from probstat.conformal import conformalize_surrogate
from probstat.distributions import Uniform
from probstat.inverse import (
    CalibrationPosterior,
    CalibrationProblem,
    DiscrepancyCalibration,
    DiscrepancyPrediction,
    DiscrepancyProcess,
    Experiment,
    KnownNoise,
    LogLinearNoise,
    NoiseModel,
    PosteriorPrediction,
    predict,
)
from probstat.modeling import JointInput
from probstat.surrogates import (
    fit_gaussian_process_to_model,
    fit_polynomial_chaos,
    surrogate_probabilistic_model,
)

from ..chaining import (
    DEFAULT_POSTERIOR_FAMILIES,
    PosteriorMethod,
    posterior_joint_input,
)
from ._base import _stage, _StudyBase
from ._context import _setting_value
from ._records import DiscrepancyAnalysis, DiscrepancyStudyPrediction


class _CalibrationStages(_StudyBase):
    """Calibration and emulation stages of the study.

    References:
        [REF-KENNEDY-OHAGAN-2001], [REF-SUDRET-2007, Chapter 4].

    """

    @_stage("calibration")
    def calibrate(
        self,
        experiments: Sequence[Experiment],
        noise: Sequence[NoiseModel],
        *,
        draws: int = 2000,
        warmup: int = 1000,
        chains: int = 4,
    ) -> CalibrationPosterior:
        """Calibrate the model's random inputs against observed experiments.

        The study model's joint input law is the prior over the calibration
        parameters; each experiment supplies observed QoIs and the
        deterministic conditions under which they were measured. Sampling is
        adaptive Metropolis in the prior's standard-normal space.

        Args:
            experiments: Observed experiments.
            noise: One noise model per observed QoI.
            draws: Kept draws per chain.
            warmup: Warm-up iterations per chain.
            chains: Number of chains.

        Returns:
            Posterior with diagnostics; also kept for :meth:`predict`.

        Equation:
            [EQ-CAL-NL-001].

        References:
            [REF-KENNEDY-OHAGAN-2001], [REF-GELMAN-ETAL-2013, Chapter 11].

        """
        stream = self.stream("calibration")
        problem = CalibrationProblem(self._model, tuple(experiments), tuple(noise))
        before = self._model.evaluations
        posterior = problem.sample_posterior(
            draws=draws, warmup=warmup, chains=chains, seed=stream
        )
        self._problem, self._posterior = problem, posterior
        self._active = "plain"
        summary: dict[str, float | str] = {"experiments": len(problem.experiments)}
        for entry in posterior.summary():
            summary[f"mean[{entry.name}]"] = entry.mean
            summary[f"sd[{entry.name}]"] = entry.standard_deviation
            summary[f"rhat[{entry.name}]"] = entry.rank_normalized_rhat
        self._record(
            "calibration",
            f"posterior from {len(problem.experiments)} experiments",
            summary,
            max(self._model.evaluations - before, posterior.evaluations),
            stream,
            posterior,
        )
        return posterior

    @_stage("discrepancy_calibration")
    def calibrate_with_discrepancy(
        self,
        experiments: Sequence[Experiment],
        quantity: str,
        noise: KnownNoise | LogLinearNoise,
        discrepancy: DiscrepancyProcess | None,
        *,
        draws: int = 2000,
        warmup: int = 1000,
        chains: int = 4,
        method: Literal["adaptive-metropolis", "hmc", "nuts"] = "adaptive-metropolis",
        level: float = 0.95,
        grid: int = 41,
        grid_draws: int = 200,
        correlation_warning: float = 0.5,
        ratio_warning: float = 0.8,
    ) -> DiscrepancyAnalysis:
        """Calibrate one QoI with a Kennedy-O'Hagan model-discrepancy term.

        The observation model is ``y = eta(x, theta) + delta(x) + eps`` with a
        Gaussian-process discrepancy ``delta`` and known or log-linear noise
        ``eps``. As in :meth:`calibrate`, the study model's joint input law is
        the prior of ``theta``. The chains run on this stage's stream exactly
        as :meth:`~probstat.inverse.DiscrepancyCalibration.sample_posterior`
        with ``seed=stream``. The identifiability report (theta against the
        discrepancy hyperparameters, posterior over prior standard deviation)
        is recorded in the summary. With ``grid > 1`` and a one-input
        discrepancy, simulator-only and bias-corrected predictions on an even
        grid across the experiments' range are stored for figures (their
        model runs are charged to this stage).

        After this stage :meth:`predict` reports simulator-only and
        bias-corrected intervals, and :meth:`design` needs explicit ``noise``
        models (the design problem has no discrepancy term).

        Args:
            experiments: Observed experiments; each must observe ``quantity``.
            quantity: Calibrated QoI.
            noise: :class:`~probstat.inverse.KnownNoise` or
                :class:`~probstat.inverse.LogLinearNoise`.
            discrepancy: Gaussian-process discrepancy, or ``None`` for a
                calibration with the given noise model only.
            draws: Kept draws per chain.
            warmup: Warm-up iterations per chain.
            chains: Number of chains.
            method: ``"adaptive-metropolis"``, ``"hmc"``, or ``"nuts"``.
            level: Central probability of recorded intervals.
            grid: Number of grid predictions (``0`` or ``1`` skips the grid).
            grid_draws: Posterior draws used for the grid predictions.
            correlation_warning: ``|rho|`` threshold of the identifiability
                warning.
            ratio_warning: Standard-deviation ratio threshold of the warning.

        Returns:
            Calibration problem, posterior, identifiability report, and grid
            predictions; also kept for :meth:`predict`.

        Raises:
            ValueError: If ``level`` is outside ``(0, 1)`` or ``grid`` is
                negative (and as raised by ``DiscrepancyCalibration``).

        Equation:
            [EQ-KOH-MODEL-001], [EQ-KOH-POST-001], [EQ-KOH-IDENT-001],
            [EQ-KOH-PRED-001].

        References:
            [REF-KENNEDY-OHAGAN-2001], [REF-HIGDON-ETAL-2004],
            [REF-BRYNJARSDOTTIR-OHAGAN-2014].

        """
        if not 0.0 < level < 1.0:
            raise ValueError("level must lie strictly inside (0, 1)")
        if isinstance(grid, bool) or not isinstance(grid, int) or grid < 0:
            raise ValueError("grid must be a non-negative integer")
        stream = self.stream("discrepancy_calibration")
        calibration = DiscrepancyCalibration(
            self._model, tuple(experiments), quantity, noise, discrepancy
        )
        before = self._model.evaluations
        posterior = calibration.sample_posterior(
            draws=draws, warmup=warmup, chains=chains, seed=stream, method=method
        )
        report = calibration.identifiability(
            posterior,
            correlation_warning=correlation_warning,
            ratio_warning=ratio_warning,
        )
        grid_input: str | None = None
        predictions: tuple[DiscrepancyPrediction, ...] = ()
        if grid > 1 and discrepancy is not None and len(discrepancy.inputs) == 1:
            grid_input = discrepancy.inputs[0]
            observed_x = [
                float({**self._model.fixed, **item.conditions}[grid_input])
                for item in calibration.experiments
            ]
            low, high = min(observed_x), max(observed_x)
            if high > low:
                predictions = calibration.predict(
                    posterior,
                    tuple(
                        {grid_input: low + (high - low) * i / (grid - 1)}
                        for i in range(grid)
                    ),
                    max_draws=grid_draws,
                )
            else:
                grid_input = None
        analysis = DiscrepancyAnalysis(
            calibration, posterior, report, level, grid_input, predictions
        )
        self._discrepancy = analysis
        self._active = "discrepancy"
        summary: dict[str, float | str] = {
            "quantity": quantity,
            "experiments": calibration.observation_count,
            "noise": "known" if isinstance(noise, KnownNoise) else "log-linear",
            "discrepancy": "none" if discrepancy is None else discrepancy.kernel,
            "method": method,
        }
        for entry in posterior.summary():
            summary[f"mean[{entry.name}]"] = entry.mean
            summary[f"sd[{entry.name}]"] = entry.standard_deviation
            summary[f"rhat[{entry.name}]"] = entry.rank_normalized_rhat
        summary["max |rho(theta, psi)|"] = report.max_abs_cross_correlation
        for name, ratio in report.standard_deviation_ratios.items():
            summary[f"sd ratio[{name}]"] = ratio
        summary["identifiability"] = report.warning or "no confounding flagged"
        if predictions:
            shifts = [item.discrepancy_mean for item in predictions]
            worst = max(range(len(shifts)), key=lambda i: abs(shifts[i]))
            summary["largest delta mean"] = shifts[worst]
            summary[f"at {grid_input}"] = float(
                predictions[worst].condition[str(grid_input)]
            )
        self._record(
            "discrepancy_calibration",
            f"Kennedy-O'Hagan posterior of {quantity} from "
            f"{calibration.observation_count} experiments",
            summary,
            self._model.evaluations - before,
            stream,
            analysis,
        )
        return analysis

    def _predict_discrepancy(
        self,
        quantities: Sequence[str] | None,
        conditions: Sequence[Mapping[str, float]],
        include_noise: bool,
        max_draws: int | None,
    ) -> DiscrepancyStudyPrediction:
        """Predict after a Kennedy-O'Hagan calibration and record the stage.

        Raises:
            ValueError: If another QoI is requested, or noise is requested
                with per-experiment known noise (unknown at new conditions).

        References:
            [REF-KENNEDY-OHAGAN-2001]. Equation [EQ-KOH-PRED-001].

        """
        assert self._discrepancy is not None
        analysis = self._discrepancy
        calibration = analysis.calibration
        quantity = calibration.quantity
        if quantities is not None and tuple(quantities) != (quantity,):
            raise ValueError(f"the discrepancy calibration predicts only {quantity!r}")
        targets = tuple(conditions)
        known: tuple[float, ...] | None = None
        if include_noise and isinstance(calibration.noise, KnownNoise):
            shared = calibration.noise.standard_deviations
            if not isinstance(shared, float):
                raise ValueError(
                    "include_noise needs one shared KnownNoise standard deviation"
                )
            known = tuple(shared for _ in targets)
        stream = self.stream("prediction")
        before = self._model.evaluations
        predictions = calibration.predict(
            analysis.posterior,
            targets,
            max_draws=max_draws,
            observation_standard_deviations=known,
        )
        draws = len(predictions[0].simulator)
        result = DiscrepancyStudyPrediction(
            quantity,
            predictions,
            analysis.level,
            draws,
            self._model.evaluations - before,
        )
        summary: dict[str, float | str] = {
            "draws": draws,
            "level": analysis.level,
            "calibration": "Kennedy-O'Hagan",
        }
        for index, item in enumerate(predictions):
            key = f"{quantity}#{index}"
            low, high = item.interval(analysis.level, kind="simulator")
            summary[f"simulator mean[{key}]"] = item.simulator_mean
            summary[f"simulator q_low[{key}]"] = low
            summary[f"simulator q_high[{key}]"] = high
            low, high = item.interval(analysis.level, kind="corrected")
            summary[f"corrected mean[{key}]"] = item.corrected_mean
            summary[f"corrected q_low[{key}]"] = low
            summary[f"corrected q_high[{key}]"] = high
            summary[f"delta mean[{key}]"] = item.discrepancy_mean
            if include_noise and item.noise_variances is not None:
                low, high = item.interval(analysis.level, kind="observed")
                summary[f"observed q_low[{key}]"] = low
                summary[f"observed q_high[{key}]"] = high
        self._record(
            "prediction",
            f"simulator-only and bias-corrected predictions at {len(targets)} "
            "conditions",
            summary,
            result.evaluations,
            stream,
            result,
        )
        return result

    @_stage("prediction")
    def predict(
        self,
        quantities: Sequence[str] | None = None,
        conditions: Sequence[Mapping[str, float]] = ({},),
        *,
        include_noise: bool = False,
        max_draws: int | None = 1000,
    ) -> PosteriorPrediction | DiscrepancyStudyPrediction:
        """Push the latest posterior through the forward model.

        After :meth:`calibrate` the result is a
        :class:`~probstat.inverse.PosteriorPrediction`. After
        :meth:`calibrate_with_discrepancy` it is a
        :class:`DiscrepancyStudyPrediction` whose summary separates the
        simulator alone ``eta(x, theta)`` from the bias-corrected process
        ``eta + delta`` (and, with ``include_noise``, a new observation).
        The latest calibration of either kind is used.

        Args:
            quantities: QoIs to predict (all observed QoIs by default).
            conditions: Prediction conditions.
            include_noise: Add observation noise to the predictive draws.
            max_draws: Cap on posterior draws propagated.

        Returns:
            Posterior prediction separating latent and observed uncertainty.

        Raises:
            RuntimeError: If no calibration stage has been run.

        Equation:
            [EQ-BAYES-PRED-001], [EQ-KOH-PRED-001].

        References:
            [REF-GELMAN-ETAL-2013, Section 1.3], [REF-KENNEDY-OHAGAN-2001].

        """
        if self._active == "discrepancy":
            return self._predict_discrepancy(
                quantities, conditions, include_noise, max_draws
            )
        if self._problem is None or self._posterior is None:
            raise RuntimeError("run calibrate() before predict()")
        stream = self.stream("prediction")
        result = predict(
            self._problem,
            self._posterior,
            quantities,
            conditions,
            include_noise=include_noise,
            max_draws=max_draws,
            seed=stream,
        )
        summary: dict[str, float | str] = {"draws": result.draws}
        for index, item in enumerate(result.predictions):
            summary[f"mean[{item.quantity}#{index}]"] = item.mean
            summary[f"q_low[{item.quantity}#{index}]"] = item.latent_quantiles[0]
            summary[f"q_high[{item.quantity}#{index}]"] = item.latent_quantiles[-1]
        self._record(
            "prediction",
            f"posterior predictive for {len(result.predictions)} targets",
            summary,
            result.evaluations,
            stream,
            result,
        )
        return result

    @_stage("emulation")
    def emulate(
        self,
        quantity: str,
        *,
        method: Literal["pce", "gp"] = "pce",
        degree: int = 3,
        size: int | None = None,
        basis: str | None = None,
        conformal_alpha: float | None = None,
        conformal_size: int = 200,
    ) -> Self:
        """Fit a surrogate of one QoI and return a study on the surrogate.

        The returned study shares this study's joint input law and draws its
        streams from a child of this study's root, so expensive analyses
        (large propagation, reliability, Sobol) can run on the emulator while
        this study records the training cost and accuracy. The child study is
        kept in :attr:`children`: its stages appear in this study's report
        and JSON record, and its model calls are counted as *surrogate*
        evaluations, never as forward-model evaluations of this study. The
        surrogate's approximation error (LOO error, conformal half-width) is
        in the emulation stage summary and is not propagated automatically.

        Args:
            quantity: QoI to emulate.
            method: Polynomial chaos (``pce``) or Gaussian process (``gp``).
            degree: PCE total degree (ignored for GP).
            size: Training-design size (defaults: PCE chooses from the number
                of terms; GP uses ten points per input).
            basis: PCE basis; by default Legendre when every input is an
                independent uniform law and Hermite (Nataf space) otherwise.
                Hermite polynomials on bounded inputs extrapolate badly in
                the far tails of large quasi-random designs.
            conformal_alpha: If given, calibrate split-conformal error bars of
                the emulator on ``conformal_size`` fresh true-model runs; the
                recorded half-width covers a new output with probability at
                least ``1 - conformal_alpha`` (marginally).
            conformal_size: Number of independent calibration runs.

        Returns:
            New study whose model evaluates the surrogate.

        Raises:
            ValueError: If ``method`` is unknown.

        Equation:
            [EQ-UQ-001].

        References:
            [REF-SUDRET-2007, Chapter 4], [REF-RASMUSSEN-WILLIAMS-2006].

        """
        stream = self.stream("emulation")
        before = self._model.evaluations
        accuracy: dict[str, float | str]
        if method == "pce":
            chosen = basis or (
                "legendre"
                if self._model.inputs.is_independent
                and all(
                    isinstance(item.distribution, Uniform)
                    for item in self._model.inputs.inputs
                )
                else "hermite"
            )
            fitted: Any = fit_polynomial_chaos(
                self._model,
                quantity,
                degree=degree,
                basis=chosen,
                size=size,
                seed=stream,
            )
            accuracy = {
                "loo_error": fitted.loo_error,
                "terms": float(fitted.size),
                "basis": chosen,
            }
        elif method == "gp":
            fitted = fit_gaussian_process_to_model(
                self._model,
                quantity,
                size=size or 10 * self._model.dimension,
                seed=stream,
            )
            accuracy = {"log_marginal_likelihood": fitted.log_marginal_likelihood}
        else:
            raise ValueError("method must be pce or gp")
        surrogate = surrogate_probabilistic_model(
            self._model, {quantity: fitted}, name=f"{self._model.name}-{method}"
        )
        summary: dict[str, float | str] = {"method": method, "quantity": quantity}
        summary.update(accuracy)
        if conformal_alpha is not None:
            calibration = conformalize_surrogate(
                self._model,
                fitted,
                quantity=quantity,
                alpha=conformal_alpha,
                size=conformal_size,
                seed=stream.child("conformal"),
            )
            summary["conformal_half_width"] = calibration.regressor.quantile
            summary["conformal_coverage"] = calibration.regressor.guaranteed_coverage
        self._record(
            "emulation",
            f"{method.upper()} surrogate of {quantity}",
            summary,
            self._model.evaluations - before,
            stream,
            fitted,
        )
        child = self._surrogate_study(
            surrogate,
            seed=stream.child("surrogate-study"),
            name=f"{self._name} / {method.upper()} surrogate of {quantity}",
        )
        child._law_label, child._law_hash = self._law_label, self._law_hash
        self._children.append((len(self._records), child))
        return child

    def adopt_posterior(
        self,
        inputs: Sequence[str] | None = None,
        *,
        method: PosteriorMethod = "gaussian_copula",
        candidates: Sequence[str] = DEFAULT_POSTERIOR_FAMILIES,
        max_draws: int = 4000,
    ) -> JointInput:
        """Use the latest calibration posterior as the input law from now on.

        The posterior is known only through MCMC draws; it is approximated
        by marginals plus a Gaussian copula
        (:func:`~probstat.workflow.chaining.posterior_joint_input`):
        ``"gaussian_copula"`` fits a parametric family per input by AIC among
        ``candidates`` and a Spearman-rank copula; ``"empirical"`` uses the
        interpolated empirical law of the draws (support cut at their range)
        and a normal-score copula. A Gaussian copula cannot represent curved
        or tail dependence, so the approximation is checked in the record:
        it lists posterior and adopted means and standard deviations side by
        side. Inputs not in ``inputs`` keep their prior laws (independent of
        the adopted ones). Later stages, including a new :meth:`calibrate`,
        then use the posterior; calibrating again on the same data would use
        the data twice.

        Args:
            inputs: Calibrated inputs to adopt (default: all model inputs).
            method: ``"gaussian_copula"`` or ``"empirical"``.
            candidates: Candidate family names for ``"gaussian_copula"``.
            max_draws: Posterior draws used after even thinning.

        Returns:
            The adopted joint input law.

        Raises:
            RuntimeError: If no plain :meth:`calibrate` has been run.

        Equation:
            [EQ-INF-IC-001], [EQ-MOD-COPULA-FIT-001], [EQ-WF-EMPIRICAL-001].

        References:
            [REF-GENEST-GHOUDI-RIVEST-1995], [REF-LIU-DERKIUREGHIAN-1986],
            [REF-GELMAN-ETAL-2013, Chapter 1].

        """
        if self._posterior is None:
            raise RuntimeError("run calibrate() before adopt_posterior()")
        draws = self._posterior.draws()
        law = posterior_joint_input(
            self._model.inputs,
            draws,
            inputs,
            method=method,
            candidates=candidates,
            max_draws=max_draws,
        )
        calibration_stage = sum(r.stage == "calibration" for r in self._records)
        self._law_changes += 1
        label = f"posterior #{self._law_changes} ({method})"
        summary: dict[str, float | str] = {
            "source": f"calibration run {calibration_stage}",
            "method": method,
            "draws": float(min(len(draws), max_draws)),
        }
        adopted = list(self._model.inputs.names if inputs is None else inputs)
        for name in adopted:
            column = [float(draw[name]) for draw in draws]
            mean = math.fsum(column) / len(column)
            spread = math.sqrt(
                math.fsum((v - mean) ** 2 for v in column) / (len(column) - 1)
            )
            summary[f"posterior mean[{name}]"] = mean
            summary[f"posterior sd[{name}]"] = spread
        self._set_inputs(
            law,
            label,
            summary,
            law,
            settings={
                "inputs": _setting_value(None if inputs is None else tuple(inputs)),
                "method": method,
                "candidates": _setting_value(tuple(candidates)),
                "max_draws": max_draws,
            },
        )
        return law
