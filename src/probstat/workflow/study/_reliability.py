"""Reliability and risk stages: limit states, FORM, AK-MCS, risk, extremes.

References:
    [REF-MELCHERS-BECK-2018, Chapters 3-5], [REF-ECHARD-GAYTON-LEMAIRE-2011],
    [REF-ROCKAFELLAR-URYASEV-2002], [REF-COLES-2001, Section 4.3].

"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Literal

from probstat.monte_carlo import ProbabilityEstimate, estimate_probability
from probstat.reliability import (
    AKMCSResult,
    FORMResult,
    MethodComparison,
    ModelLimitState,
    ak_mcs,
    compare_reliability_methods,
    find_design_points,
    form_hlrf,
    mixture_importance_sampling,
    multi_point_form,
)
from probstat.risk import LossFunction, RiskAssessment, assess_risk, fit_pot
from probstat.statistics import quantile
from probstat.uq import EvaluationCache

from ._base import _stage, _StudyBase
from ._context import _converged_text, _orient_return_level, _sized
from ._records import DesignPointAnalysis, TailAnalysis


class _ReliabilityStages(_StudyBase):
    """Reliability, risk, and tail stages of the study.

    References:
        [REF-MELCHERS-BECK-2018, Chapters 3-5], [REF-COLES-2001, Section 4.3].

    """

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

    @_stage(
        "reliability",
        lambda study, args: (
            args["monte_carlo_samples"] if "mc" in _sized(args["methods"]) else None
        ),
    )
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
        methods = tuple(methods)
        stream = self.stream("reliability")
        state = self.limit_state(quantity, threshold, failure=failure)
        before = self._model.evaluations
        # FORM-based rows share one HL-RF search inside the comparison; the
        # search is repeated afterwards only to read its convergence flag.
        # HL-RF is deterministic, so with a cache over the stage every run of
        # the repeat is a cache hit and costs no simulator evaluation.
        local = self._cache if self._cache is not None else EvaluationCache()
        uses_form = bool({"form", "sorm", "is", "line"} & {m.lower() for m in methods})
        with self._model.evaluation_context(cache=local):
            comparison = compare_reliability_methods(
                state,
                methods=methods,
                monte_carlo_samples=monte_carlo_samples,
                seed=stream,
            )
            check = form_hlrf(state) if uses_form else None
        summary: dict[str, float | str] = {
            "quantity": quantity,
            "threshold": threshold,
            "failure": failure,
        }
        for row in comparison.rows:
            summary[f"pf[{row.method}]"] = row.probability
        if check is not None:
            summary["FORM converged"] = _converged_text(check)
            if not check.converged:
                self._warn(
                    f"FORM for {quantity} did not converge ({check.message}); "
                    "FORM, SORM, and importance-sampling rows build on an "
                    "unconverged design point"
                )
        self._record(
            "reliability",
            f"P({quantity} {'<=' if failure == 'below' else '>='} {threshold:g})",
            summary,
            self._model.evaluations - before,
            stream,
            comparison,
        )
        return comparison

    @_stage("design_points")
    def design_points(
        self,
        quantity: str,
        threshold: float,
        *,
        failure: Literal["above", "below"] = "below",
        max_points: int = 3,
        importance_samples: int = 2000,
        start: Sequence[float] | None = None,
    ) -> DesignPointAnalysis:
        """Find several design points and estimate the system probability.

        A single FORM point underestimates ``P_f`` when the failure domain has
        several important regions. This stage searches up to ``max_points``
        design points, combines them as a series system by multi-point FORM
        (with Ditlevsen bounds), and, unless ``importance_samples`` is zero,
        checks the result by mixture importance sampling centred on every
        point.

        Args:
            quantity: QoI name.
            threshold: Failure threshold.
            failure: ``"below"`` or ``"above"``.
            max_points: Maximum number of design points.
            importance_samples: Mixture importance-sampling size (``0`` skips
                it).
            start: Standard-normal starting point of the first HL-RF search;
                needed when the limit-state gradient vanishes at the origin
                (for example at a kink of a series system).

        Returns:
            Design points, multi-point FORM, and the optional mixture estimate.

        Raises:
            ValueError: If ``importance_samples`` is negative.

        Equation:
            [EQ-REL-MDP-001], [EQ-REL-MDP-002], [EQ-REL-MDP-003].

        References:
            [REF-DERKIUREGHIAN-DAKESSIAN-1998], [REF-MELCHERS-BECK-2018,
            Chapter 5].

        """
        if importance_samples < 0:
            raise ValueError("importance_samples must be non-negative")
        stream = self.stream("design_points")
        state = self.limit_state(quantity, threshold, failure=failure)
        before = self._model.evaluations
        first = None if start is None else form_hlrf(state, start=start)
        points = find_design_points(state, max_points, first=first)
        system = multi_point_form(points, seed=stream.child("system"))
        mixture = (
            mixture_importance_sampling(
                state, points, importance_samples, seed=stream.child("mixture")
            )
            if importance_samples
            else None
        )
        summary: dict[str, float | str] = {
            "quantity": quantity,
            "threshold": threshold,
            "failure": failure,
            "design points": float(len(points.forms)),
        }
        for index, beta in enumerate(points.betas, start=1):
            summary[f"beta[{index}]"] = beta
        summary["pf[single-point FORM]"] = points.forms[0].probability
        summary["pf[multi-point FORM]"] = system.probability
        summary["pf bounds (Ditlevsen)"] = (
            f"[{system.bounds.ditlevsen_lower:.4g}, "
            f"{system.bounds.ditlevsen_upper:.4g}]"
        )
        if mixture is not None:
            summary["pf[mixture IS]"] = mixture.probability
            summary["cov[mixture IS]"] = mixture.coefficient_of_variation
        for index, point in enumerate(points.forms, start=1):
            summary[f"FORM converged[{index}]"] = _converged_text(point)
        unconverged = [
            index
            for index, point in enumerate(points.forms, start=1)
            if not point.converged
        ]
        if unconverged:
            self._warn(
                f"design point(s) {unconverged} of {quantity} come from an "
                "unconverged FORM search; multi-point FORM and mixture importance "
                "sampling build on them"
            )
        sign = "<=" if failure == "below" else ">="
        self._record(
            "design_points",
            f"{len(points.forms)} design points of {quantity} {sign} {threshold:g}",
            summary,
            self._model.evaluations - before,
            stream,
            DesignPointAnalysis(points, system, mixture),
        )
        return DesignPointAnalysis(points, system, mixture)

    @_stage(
        "active_reliability",
        lambda study, args: (
            args["population"] if args["compare_with_monte_carlo"] else None
        ),
    )
    def active_reliability(
        self,
        quantity: str,
        threshold: float,
        *,
        failure: Literal["above", "below"] = "below",
        population: int = 10_000,
        target_cov: float = 0.1,
        max_added: int = 200,
        compare_with_monte_carlo: bool = False,
    ) -> AKMCSResult:
        """Estimate a failure probability by AK-MCS active learning.

        A Gaussian-process surrogate of the limit state is refined where its
        sign is most uncertain until the learning function ``U`` exceeds 2 on
        the whole Monte Carlo population; the true model is called only for
        the training points. With ``compare_with_monte_carlo`` the whole
        population is also evaluated on the true model (counted separately in
        the result, and in this stage's evaluations).

        Args:
            quantity: QoI name.
            threshold: Failure threshold.
            failure: ``"below"`` or ``"above"``.
            population: Initial Monte Carlo population size.
            target_cov: Target coefficient of variation of the estimate; the
                population grows until it is met (``0.1`` by default, since
                halving it quadruples the population and the surrogate's
                prediction cost).
            max_added: Maximum number of training points added.
            compare_with_monte_carlo: Also run crude Monte Carlo on the same
                population.

        Returns:
            AK-MCS result with probability, true-model evaluations, and the
            final surrogate.

        Equation:
            [EQ-REL-AKMCS-001], [EQ-REL-AKMCS-002].

        References:
            [REF-ECHARD-GAYTON-LEMAIRE-2011].

        """
        stream = self.stream("active_reliability")
        state = self.limit_state(quantity, threshold, failure=failure)
        before = self._model.evaluations
        result = ak_mcs(
            state,
            population,
            target_cov=target_cov,
            max_added=max_added,
            compare_with_monte_carlo=compare_with_monte_carlo,
            seed=stream,
        )
        summary: dict[str, float | str] = {
            "quantity": quantity,
            "threshold": threshold,
            "failure": failure,
            "pf[AK-MCS]": result.probability,
            "cov": result.coefficient_of_variation,
            "population": float(result.population),
            "training runs": float(result.evaluations),
            "converged": "yes" if result.converged else f"no ({result.message})",
        }
        if compare_with_monte_carlo:
            summary["pf[MC, same population]"] = result.reference_probability
            summary["misclassified"] = float(result.misclassified)
        sign = "<=" if failure == "below" else ">="
        self._record(
            "active_reliability",
            f"AK-MCS P({quantity} {sign} {threshold:g})",
            summary,
            self._model.evaluations - before,
            stream,
            result,
        )
        return result

    @_stage("form")
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
        summary["converged"] = _converged_text(result)
        summary["iterations"] = float(result.iterations)
        if not result.converged:
            self._warn(
                f"FORM for {quantity} did not converge ({result.message}); the "
                "design point and probability are not reliable"
            )
        self._record(
            "form",
            f"FORM for {quantity} threshold {threshold:g}",
            summary,
            result.evaluations,
            stream,
            result,
        )
        return result

    @_stage("risk", lambda study, args: args["size"])
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

    @_stage("extremes", lambda study, args: study._fresh_cost(args, 4096))
    def extremes(
        self,
        quantity: str,
        *,
        tail: Literal["upper", "lower"] = "upper",
        threshold_quantile: float = 0.95,
        threshold: float | None = None,
        level: float | None = None,
        return_periods: Sequence[float] = (100.0, 1000.0),
        method: Literal["mle", "pwm"] = "mle",
        confidence: float = 0.95,
        size: int | None = None,
        design: str = "mc",
        reuse: bool = False,
    ) -> TailAnalysis:
        """Fit a peaks-over-threshold model to the tail of one QoI.

        The sample is ``size`` (default 4096) fresh plain Monte Carlo runs
        drawn from this stage's stream, so the result does not depend on the
        order of stages. ``reuse=True`` reuses the latest propagation at no
        cost when it is a plain Monte Carlo sample (a quasi-random sample is
        never reused: its points are not independent). Each model run is one
        observation (``observations_per_year = 1``), so a return period ``T``
        is a number of runs and the ``T``-run level has tail probability
        ``1/T``. The lower tail is analyzed as the upper tail of ``-Y``.

        Args:
            quantity: QoI name.
            tail: ``"upper"`` (large values) or ``"lower"`` (small values).
            threshold_quantile: Threshold as the sample quantile of the
                oriented values (used when ``threshold`` is ``None``).
            threshold: Explicit threshold in QoI units.
            level: Optional level whose tail probability is estimated both by
                the POT model and empirically (Wilson interval).
            return_periods: Return periods in model runs (each must exceed
                ``1/zeta_u``).
            method: GPD estimator, ``"mle"`` (with delta intervals) or
                ``"pwm"`` (point estimates only).
            confidence: Two-sided interval confidence.
            size: Fresh sample size (default 4096).
            design: Sampling design of a fresh sample; only ``"mc"``.
            reuse: Reuse the latest plain Monte Carlo propagation.

        Returns:
            Tail analysis in QoI units.

        Raises:
            ValueError: If the quantity, threshold quantile, tail, or level is
                invalid (a level must lie beyond the threshold).

        Equation:
            [EQ-EVA-GPD-001], [EQ-EVA-RATE-001], [EQ-EVA-POT-001],
            [EQ-EVA-POT-002], [EQ-MC-PROB-001].

        References:
            [REF-COLES-2001, Section 4.3], [REF-WILSON-1927].

        """
        if tail not in ("upper", "lower"):
            raise ValueError("tail must be 'upper' or 'lower'")
        if threshold is None and not 0.0 < threshold_quantile < 1.0:
            raise ValueError("threshold_quantile must lie strictly inside (0, 1)")
        stream = self.stream("extremes")
        sample, evaluations, source = self._output_sample(
            stream, quantity, size, design, 4096, reuse
        )
        sign = 1.0 if tail == "upper" else -1.0
        oriented = tuple(sign * value for value in sample[quantity].values)
        cut = (
            quantile(oriented, threshold_quantile)
            if threshold is None
            else sign * threshold
        )
        fit = fit_pot(oriented, cut, observations_per_year=1.0, method=method)
        levels = tuple(
            _orient_return_level(fit.return_level(period, confidence=confidence), sign)
            for period in return_periods
        )
        pot_probability: float | None = None
        empirical: ProbabilityEstimate | None = None
        if level is not None:
            point = sign * level
            if point < cut:
                raise ValueError(
                    "level lies inside the threshold; raise threshold_quantile "
                    "or pass a lower threshold"
                )
            pot_probability = fit.annual_exceedance_rate(point)
            empirical = estimate_probability(
                (value > point for value in oriented), confidence=confidence
            )
        result = TailAnalysis(
            quantity,
            tail,
            fit,
            levels,
            level,
            pot_probability,
            empirical,
            len(oriented),
            source,
            tuple(sign * value for value in sorted(v for v in oriented if v > cut)),
        )
        sigma_se, xi_se = fit.gpd.standard_errors
        summary: dict[str, float | str] = {
            "quantity": quantity,
            "tail": tail,
            "source": source,
            "design": sample.design.method,
            "sample_size": len(oriented),
            "threshold": result.threshold,
            "threshold_rule": (
                f"q{threshold_quantile:g}" if threshold is None else "given"
            ),
            "exceedances": fit.rate.exceedances,
            "zeta_u": fit.rate.probability,
            "sigma": fit.gpd.sigma,
            "sigma_se": sigma_se,
            "xi": fit.gpd.xi,
            "xi_se": xi_se,
        }
        n = len(oriented)
        for item in levels:
            key = f"x[T={item.return_period:g}]"
            summary[key] = item.level
            summary[f"{key}.low"] = item.lower
            summary[f"{key}.high"] = item.upper
            summary[f"empirical {key}"] = (
                sign * quantile(oriented, 1.0 - 1.0 / item.return_period)
                if item.return_period <= n
                else "n/a (return period exceeds the sample size)"
            )
        if level is not None and empirical is not None and pot_probability is not None:
            summary["level"] = level
            summary["P_pot"] = pot_probability
            summary["P_empirical"] = empirical.probability
            summary["P_empirical.low"] = empirical.confidence_interval.lower
            summary["P_empirical.high"] = empirical.confidence_interval.upper
            summary["P_pot/P_empirical"] = (
                pot_probability / empirical.probability
                if empirical.probability > 0.0
                else math.inf
            )
        self._record(
            "extremes",
            f"POT {tail} tail of {quantity}, threshold {result.threshold:g}",
            summary,
            evaluations,
            stream,
            result,
        )
        return result
