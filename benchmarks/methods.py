"""One adapter per library method, and the per-mode run sizes.

An adapter takes a fresh :class:`~benchmarks.problems.Instance`, a seed, and
the size table of the current mode, calls the library, and returns
:class:`MethodOutput` (estimates with the method's *own* reported standard
error or interval). Adapters never catch exceptions; the harness does, so a
failure is always recorded instead of hidden.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from benchmarks.problems import Instance, Problem
from benchmarks.references import normal_quantile
from probstat.inference import advi
from probstat.propagation import propagate
from probstat.reliability import (
    ak_mcs,
    compare_reliability_methods,
    find_design_points,
    form_hlrf,
    mixture_importance_sampling,
    model_monte_carlo,
    multi_point_form,
    series_system_bounds,
    system_failure_probability,
)
from probstat.sensitivity import (
    linear_gaussian_shapley,
    shapley_effects,
    sobol_analysis,
)
from probstat.surrogates import fit_polynomial_chaos

#: Run sizes per mode. ``seeds`` is the number of seeds per category;
#: deterministic methods run once regardless.
SIZES: dict[str, dict[str, Any]] = {
    "quick": {
        "seeds": {
            "reliability": 2,
            "sensitivity": 2,
            "propagation": 5,
            "calibration": 2,
        },
        "mc_samples": 5_000,
        "is_samples": 1_000,
        "subset_samples": 500,
        "ce_samples": 500,
        "line_samples": 50,
        "mixture_samples": 1_000,
        "system_samples": 5_000,
        "akmcs_population": 2_000,
        "akmcs_max_population": 2_000,
        "akmcs_target_cov": 1.0,
        "akmcs_max_dimension": 2,
        "sobol_size": 256,
        "bootstrap": 50,
        "pce_degree": {
            "ishigami-indices": 8,
            "g-function": 4,
            "ishigami-moments": 8,
            "exponential-5d": 3,
        },
        "shapley_outer": 100,
        "shapley_permutations": 50,
        "shapley_variance_samples": 1_000,
        "propagation_size": 256,
        "am_draws": 500,
        "nuts_draws": 200,
        "hmc_draws": 100,
        "chains": 2,
        "advi_iterations": 2_000,
    },
    "full": {
        "seeds": {
            "reliability": 20,
            "sensitivity": 10,
            "propagation": 30,
            "calibration": 20,
        },
        "mc_samples": 100_000,
        "is_samples": 2_000,
        "subset_samples": 1_000,
        "ce_samples": 1_000,
        "line_samples": 100,
        "mixture_samples": 2_000,
        "system_samples": 50_000,
        "akmcs_population": 10_000,
        "akmcs_max_population": 50_000,
        "akmcs_target_cov": 0.1,
        "akmcs_max_dimension": 10,
        "sobol_size": 2_048,
        "bootstrap": 200,
        "pce_degree": {
            "ishigami-indices": 10,
            "g-function": 5,
            "ishigami-moments": 10,
            "exponential-5d": 4,
        },
        "shapley_outer": 500,
        "shapley_permutations": 200,
        "shapley_variance_samples": 5_000,
        "propagation_size": 1_024,
        "am_draws": 2_000,
        "nuts_draws": 1_000,
        "hmc_draws": 500,
        "chains": 4,
        "advi_iterations": 10_000,
    },
}


#: Known limitations: (problem, method) -> (substring of the error, reason).
#: A matching failure is still recorded and shown, but marked as a known
#: limitation and does not make ``run.py`` exit non-zero; any other does.
_ORIGIN_KINK = (
    "gradient vanished",
    "the system margin has a symmetric kink at the origin, so the HL-RF search "
    "that starts there has no direction; every method built on a single FORM "
    "search inherits this",
)
EXPECTED_FAILURES: dict[tuple[str, str], tuple[str, str]] = {
    ("four-branch", name): _ORIGIN_KINK
    for name in (
        "FORM",
        "SORM (Breitung)",
        "SORM (Hohenbichler-Rackwitz)",
        "Multi-point FORM",
        "IS at design point",
        "Mixture IS",
        "Line sampling",
    )
}


@dataclass(frozen=True)
class Estimate:
    """One estimated quantity with the method's own uncertainty report.

    Args:
        value: Point estimate.
        standard_error: Reported standard error of the estimate (``None``
            when the method reports none, e.g. FORM or a bootstrap interval).
        interval: Reported interval (``None`` when none is reported).
        level: Nominal level of ``interval`` (``None`` for bounds).

    """

    value: float
    standard_error: float | None = None
    interval: tuple[float, float] | None = None
    level: float | None = None


@dataclass(frozen=True)
class MethodOutput:
    """Result of one adapter call.

    Args:
        estimates: ``quantity -> Estimate``.
        notes: Short diagnostic remarks (convergence, design points, ...).

    """

    estimates: Mapping[str, Estimate]
    notes: tuple[str, ...] = ()


Runner = Callable[[Instance, int, Mapping[str, Any], Problem], MethodOutput]


def _always(problem: Problem) -> bool:
    """Apply to every problem of the method's category."""
    del problem
    return True


def _never_skip(problem: Problem, sizes: Mapping[str, Any]) -> str | None:
    """Never skip."""
    del problem, sizes
    return None


@dataclass(frozen=True)
class Method:
    """A benchmarked library method.

    Args:
        name: Row label in the report.
        category: Problem category it applies to.
        run: Adapter.
        kind: ``"estimator"`` (unbiased sampling estimate with a reported
            error), ``"approximation"`` (deterministic or surrogate answer
            whose reported numbers do not bound its bias), or ``"bounds"``.
        applies: Applicability rule on the problem.
        skip: ``(problem, sizes) -> reason`` to skip for budget reasons
            (recorded in the results, never silent), or ``None``.
        deterministic: Run only for the first seed.

    """

    name: str
    category: str
    run: Runner
    kind: str = "estimator"
    applies: Callable[[Problem], bool] = _always
    skip: Callable[[Problem, Mapping[str, Any]], str | None] = _never_skip
    deterministic: bool = False


# ---------------------------------------------------------------------------
# Reliability
# ---------------------------------------------------------------------------


def _compare(key: str, row: str) -> Runner:
    """Return an adapter running one method through compare_reliability_methods."""

    def run(
        instance: Instance, seed: int, sizes: Mapping[str, Any], problem: Problem
    ) -> MethodOutput:
        """Run ``compare_reliability_methods`` restricted to one method."""
        del problem
        comparison = compare_reliability_methods(
            instance["limit_state"],
            methods=(key,),
            monte_carlo_samples=sizes["mc_samples"],
            importance_samples=sizes["is_samples"],
            subset_samples=sizes["subset_samples"],
            cross_entropy_samples=sizes["ce_samples"],
            line_samples=sizes["line_samples"],
            seed=seed,
        )
        result = comparison.row(row)
        cov = result.coefficient_of_variation
        notes: tuple[str, ...] = ()
        if cov is None:
            se = None
        elif result.probability == 0.0:
            se = 0.0
            notes = ("no failure samples (reported SE = 0)",)
        else:
            se = cov * result.probability
        return MethodOutput({"Pf": Estimate(result.probability, se)}, notes)

    return run


def _crude_mc(
    instance: Instance, seed: int, sizes: Mapping[str, Any], problem: Problem
) -> MethodOutput:
    """Run crude Monte Carlo; report its plug-in SE and its Wilson interval."""
    del problem
    result = model_monte_carlo(instance["limit_state"], sizes["mc_samples"], seed=seed)
    interval = result.failure.confidence_interval
    notes: tuple[str, ...] = ()
    if result.failure.successes == 0:
        notes = ("no failure samples: plug-in SE = 0, Wilson interval still valid",)
    return MethodOutput(
        {
            "Pf": Estimate(
                result.probability,
                result.standard_error,
                (interval.lower, interval.upper),
                interval.confidence,
            )
        },
        notes,
    )


def _multi_form(
    instance: Instance, seed: int, sizes: Mapping[str, Any], problem: Problem
) -> MethodOutput:
    """Run multi-point FORM over the design points found by bulging."""
    del sizes, problem
    points = find_design_points(instance["limit_state"], 4)
    combined = multi_point_form(points, seed=seed)
    return MethodOutput(
        {"Pf": Estimate(combined.probability)},
        (f"{len(points.forms)} design point(s), {combined.method}",),
    )


def _mixture(
    instance: Instance, seed: int, sizes: Mapping[str, Any], problem: Problem
) -> MethodOutput:
    """Run mixture importance sampling centred on all design points found."""
    del problem
    state = instance["limit_state"]
    points = find_design_points(state, 4)
    result = mixture_importance_sampling(
        state, points, sizes["mixture_samples"], seed=seed
    )
    return MethodOutput(
        {"Pf": Estimate(result.probability, result.standard_error)},
        (f"{len(points.forms)} design point(s)",),
    )


def _akmcs_skip(problem: Problem, sizes: Mapping[str, Any]) -> str | None:
    """Skip AK-MCS above the mode's dimension budget (GP cost grows quickly)."""
    if problem.dimension > sizes["akmcs_max_dimension"]:
        return (
            f"skipped: d = {problem.dimension} exceeds this mode's AK-MCS budget "
            f"(d <= {sizes['akmcs_max_dimension']})"
        )
    return None


def _akmcs(
    instance: Instance, seed: int, sizes: Mapping[str, Any], problem: Problem
) -> MethodOutput:
    """Run AK-MCS; its SE is the population Monte Carlo error only."""
    del problem
    result = ak_mcs(
        instance["limit_state"],
        sizes["akmcs_population"],
        max_population=sizes["akmcs_max_population"],
        target_cov=sizes["akmcs_target_cov"],
        seed=seed,
    )
    notes = [f"{result.added} points added, population {result.population}"]
    if not result.converged:
        notes.append("not converged: " + result.message)
    return MethodOutput(
        {"Pf": Estimate(result.probability, result.standard_error)}, tuple(notes)
    )


def _system(method: str) -> Runner:
    """Return an adapter for series-system sampling over the four branches."""

    def run(
        instance: Instance, seed: int, sizes: Mapping[str, Any], problem: Problem
    ) -> MethodOutput:
        """Run ``system_failure_probability`` on the branch limit states."""
        del problem
        result = system_failure_probability(
            instance["components"],
            kind="series",
            method=method,  # type: ignore[arg-type]
            samples=sizes["system_samples"],
            seed=seed,
        )
        return MethodOutput({"Pf": Estimate(result.probability, result.standard_error)})

    return run


def _ditlevsen(
    instance: Instance, seed: int, sizes: Mapping[str, Any], problem: Problem
) -> MethodOutput:
    """Return Ditlevsen bi-modal bounds from component FORM results."""
    del seed, sizes, problem
    forms = [form_hlrf(component) for component in instance["components"]]
    bounds = series_system_bounds(forms)
    lower, upper = bounds.ditlevsen_lower, bounds.ditlevsen_upper
    return MethodOutput(
        {"Pf": Estimate(0.5 * (lower + upper), None, (lower, upper), None)},
        (f"bounds [{lower:.4e}, {upper:.4e}]; estimate is their midpoint",),
    )


def _is_system(problem: Problem) -> bool:
    """Apply to series-system problems only."""
    return "system" in problem.tags


# ---------------------------------------------------------------------------
# Sensitivity
# ---------------------------------------------------------------------------


def _has_group(group: str) -> Callable[[Problem], bool]:
    """Return a rule matching problems whose references contain ``group``."""

    def rule(problem: Problem) -> bool:
        """Check the problem's error-kind groups."""
        return group in problem.error_kinds

    return rule


def _sobol(design: str) -> Runner:
    """Return an adapter for pick-freeze Sobol indices on one design."""

    def run(
        instance: Instance, seed: int, sizes: Mapping[str, Any], problem: Problem
    ) -> MethodOutput:
        """Run ``sobol_analysis`` with bootstrap percentile intervals."""
        del problem
        result = sobol_analysis(
            instance.model,
            instance["quantity"],
            sizes["sobol_size"],
            design=design,
            seed=seed,
            bootstrap=sizes["bootstrap"],
            confidence=0.95,
        )
        estimates: dict[str, Estimate] = {}
        for name, first, total in zip(
            result.input_names, result.first_order, result.total_effect, strict=True
        ):
            estimates[f"S1[{name}]"] = Estimate(
                first.estimate, None, (first.lower, first.upper), 0.95
            )
            estimates[f"ST[{name}]"] = Estimate(
                total.estimate, None, (total.lower, total.upper), 0.95
            )
        notes = (
            () if result.design == design else (f"design fell back to {result.design}",)
        )
        return MethodOutput(estimates, notes)

    return run


def _pce_basis(problem: Problem) -> str:
    """Return the orthonormal basis matching the problem's input laws."""
    return "legendre" if "uniform" in problem.tags else "hermite"


def _pce_sobol(
    instance: Instance, seed: int, sizes: Mapping[str, Any], problem: Problem
) -> MethodOutput:
    """Fit a polynomial chaos expansion and read Sobol indices off it."""
    degree = sizes["pce_degree"][problem.key]
    expansion = fit_polynomial_chaos(
        instance.model,
        instance["quantity"],
        degree=degree,
        basis=_pce_basis(problem),
        seed=seed,
    )
    indices = expansion.sobol_indices()
    estimates: dict[str, Estimate] = {}
    for name in indices.names:
        estimates[f"S1[{name}]"] = Estimate(indices.first_order[name])
        estimates[f"ST[{name}]"] = Estimate(indices.total_order[name])
    return MethodOutput(
        estimates, (f"degree {degree}, LOO error {expansion.loo_error:.2e}",)
    )


def _shapley(method: str) -> Runner:
    """Return an adapter for Monte Carlo Shapley effects."""

    def run(
        instance: Instance, seed: int, sizes: Mapping[str, Any], problem: Problem
    ) -> MethodOutput:
        """Run ``shapley_effects`` with exact subsets or random permutations."""
        del problem
        result = shapley_effects(
            instance.model,
            instance["quantity"],
            method=method,  # type: ignore[arg-type]
            permutations=sizes["shapley_permutations"],
            outer_samples=sizes["shapley_outer"] if method == "exact" else None,
            variance_samples=sizes["shapley_variance_samples"],
            seed=seed,
        )
        return MethodOutput(
            {
                f"Sh[{name}]": Estimate(effect, error)
                for name, effect, error in zip(
                    result.input_names,
                    result.effects,
                    result.standard_errors,
                    strict=True,
                )
            }
        )

    return run


def _closed_form_shapley(
    instance: Instance, seed: int, sizes: Mapping[str, Any], problem: Problem
) -> MethodOutput:
    """Evaluate probstat's closed-form linear-Gaussian Shapley effects."""
    del seed, sizes, problem
    from benchmarks.problems import SHAPLEY_COEFFICIENTS, shapley_covariance

    effects = linear_gaussian_shapley(SHAPLEY_COEFFICIENTS, shapley_covariance())
    return MethodOutput(
        {
            f"Sh[{name}]": Estimate(value)
            for name, value in zip(instance["names"], effects, strict=True)
        },
        ("closed form, no model evaluations",),
    )


def _is_correlated(problem: Problem) -> bool:
    """Apply to the correlated linear problem only."""
    return "correlated" in problem.tags


def _is_independent(problem: Problem) -> bool:
    """Apply to problems with independent inputs (Sobol indices)."""
    return "independent" in problem.tags


# ---------------------------------------------------------------------------
# Propagation
# ---------------------------------------------------------------------------


def _design(design: str) -> Runner:
    """Return an adapter propagating one sampling design."""

    def run(
        instance: Instance, seed: int, sizes: Mapping[str, Any], problem: Problem
    ) -> MethodOutput:
        """Run ``propagate`` and read the mean (with its interval) and variance."""
        del problem
        study = propagate(
            instance.model, sizes["propagation_size"], design=design, seed=seed
        )
        output = study[instance["quantity"]]
        estimate = output.mean_estimate
        interval = estimate.confidence_interval
        return MethodOutput(
            {
                "mean": Estimate(
                    estimate.estimate,
                    estimate.standard_error,
                    (interval.lower, interval.upper),
                    interval.confidence,
                ),
                "variance": Estimate(output.standard_deviation**2),
            }
        )

    return run


def _pce_moments(
    instance: Instance, seed: int, sizes: Mapping[str, Any], problem: Problem
) -> MethodOutput:
    """Fit a polynomial chaos expansion and read its mean and variance."""
    degree = sizes["pce_degree"][problem.key]
    expansion = fit_polynomial_chaos(
        instance.model,
        instance["quantity"],
        degree=degree,
        basis=_pce_basis(problem),
        seed=seed,
    )
    return MethodOutput(
        {"mean": Estimate(expansion.mean), "variance": Estimate(expansion.variance)},
        (f"degree {degree}, {len(expansion.design)} runs",),
    )


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

Z90 = normal_quantile(0.95)


def _gaussian_output(
    names: tuple[str, ...], mean: tuple[float, ...], sds: tuple[float, ...]
) -> dict[str, Estimate]:
    """Return mean, sd, and 90 % interval estimates of a Gaussian approximation."""
    estimates: dict[str, Estimate] = {}
    for name, center, sd in zip(names, mean, sds, strict=True):
        estimates[f"mean[{name}]"] = Estimate(center)
        estimates[f"sd[{name}]"] = Estimate(sd)
        estimates[f"theta[{name}]"] = Estimate(
            center, None, (center - Z90 * sd, center + Z90 * sd), 0.90
        )
    return estimates


def _laplace(
    instance: Instance, seed: int, sizes: Mapping[str, Any], problem: Problem
) -> MethodOutput:
    """Run the Laplace approximation (BFGS mode, finite-difference Hessian)."""
    del seed, sizes, problem
    laplace = instance["problem"].laplace_posterior()
    notes = () if laplace.converged else ("BFGS did not report convergence",)
    return MethodOutput(
        _gaussian_output(
            instance["names"], tuple(laplace.mode), tuple(laplace.standard_deviations)
        ),
        notes,
    )


def _mcmc(method: str, draws_key: str) -> Runner:
    """Return an adapter for one of the calibration samplers."""

    def run(
        instance: Instance, seed: int, sizes: Mapping[str, Any], problem: Problem
    ) -> MethodOutput:
        """Sample the posterior and summarize mean, MCSE, sd, and 90 % interval."""
        del problem
        draws = sizes[draws_key]
        posterior = instance["problem"].sample_posterior(
            draws=draws, warmup=draws, chains=sizes["chains"], seed=seed, method=method
        )
        estimates: dict[str, Estimate] = {}
        worst_rhat, worst_ess = 0.0, math.inf
        for item in posterior.summary((0.05, 0.95)):
            estimates[f"mean[{item.name}]"] = Estimate(item.mean, item.mcse_mean)
            estimates[f"sd[{item.name}]"] = Estimate(item.standard_deviation)
            estimates[f"theta[{item.name}]"] = Estimate(
                item.mean, None, (item.quantile(0.05), item.quantile(0.95)), 0.90
            )
            worst_rhat = max(worst_rhat, item.rank_normalized_rhat)
            worst_ess = min(worst_ess, item.effective_sample_size)
        notes = [f"max R-hat {worst_rhat:.3f}, min ESS {worst_ess:.0f}"]
        if posterior.sampler is not None:
            divergent = sum(
                len(chain.divergent_iterations)
                for chain in posterior.sampler.statistics
            )
            if divergent:
                notes.append(f"{divergent} divergent transitions")
        return MethodOutput(estimates, tuple(notes))

    return run


def _advi(family: str) -> Runner:
    """Return an adapter for ADVI with one variational family."""

    def run(
        instance: Instance, seed: int, sizes: Mapping[str, Any], problem: Problem
    ) -> MethodOutput:
        """Fit ADVI to the physical-scale log posterior."""
        del problem
        calibration = instance["problem"]
        result = advi(
            calibration.log_posterior,
            calibration.model.nominal(),
            family=family,
            max_iterations=sizes["advi_iterations"],
            seed=seed,
        )
        covariance = result.covariance
        sds = tuple(math.sqrt(covariance[i][i]) for i in range(len(result.mean)))
        notes = (
            () if result.converged else (f"not converged in {result.iterations} it.",)
        )
        return MethodOutput(
            _gaussian_output(instance["names"], tuple(result.mean), sds), notes
        )

    return run


def _exact_posterior(
    instance: Instance, seed: int, sizes: Mapping[str, Any], problem: Problem
) -> MethodOutput:
    """Return the exact conjugate posterior (baseline row, no probstat code)."""
    del instance, sizes, problem
    from benchmarks.problems import (
        CAL_CONDITIONS,
        CAL_NOISE_SD,
        CAL_PRIOR_SD,
        calibration_data,
    )
    from benchmarks.references import linear_gaussian_posterior

    _, observations = calibration_data(seed)
    mean, cov = linear_gaussian_posterior(
        CAL_CONDITIONS, observations, CAL_NOISE_SD, CAL_PRIOR_SD
    )
    sds = (math.sqrt(cov[0][0]), math.sqrt(cov[1][1]))
    return MethodOutput(
        _gaussian_output(("a", "b"), mean, sds),
        ("baseline: the 90 % coverage of the true posterior over these seeds",),
    )


def registry() -> tuple[Method, ...]:
    """Return every benchmarked method in report order."""
    return (
        Method(
            "FORM",
            "reliability",
            _compare("form", "FORM"),
            "approximation",
            deterministic=True,
        ),
        Method(
            "SORM (Breitung)",
            "reliability",
            _compare("sorm", "SORM-Breitung"),
            "approximation",
            deterministic=True,
        ),
        Method(
            "SORM (Hohenbichler-Rackwitz)",
            "reliability",
            _compare("sorm", "SORM-HR"),
            "approximation",
            deterministic=True,
        ),
        Method(
            "Multi-point FORM",
            "reliability",
            _multi_form,
            "approximation",
            deterministic=True,
        ),
        Method(
            "Ditlevsen bounds",
            "reliability",
            _ditlevsen,
            "bounds",
            _is_system,
            deterministic=True,
        ),
        Method("Crude MC", "reliability", _crude_mc),
        Method("IS at design point", "reliability", _compare("is", "IS")),
        Method("Mixture IS", "reliability", _mixture),
        Method("Subset simulation", "reliability", _compare("subset", "Subset")),
        Method("Line sampling", "reliability", _compare("line", "Line")),
        Method("CE-IS", "reliability", _compare("ce", "CE-IS")),
        Method("AK-MCS", "reliability", _akmcs, skip=_akmcs_skip),
        Method("System MC", "reliability", _system("mc"), applies=_is_system),
        Method("System IS", "reliability", _system("is"), applies=_is_system),
        Method(
            "Sobol pick-freeze, MC",
            "sensitivity",
            _sobol("mc"),
            applies=_is_independent,
        ),
        Method(
            "Sobol pick-freeze, LHS",
            "sensitivity",
            _sobol("lhs"),
            applies=_is_independent,
        ),
        Method(
            "Sobol pick-freeze, Halton",
            "sensitivity",
            _sobol("halton"),
            applies=_is_independent,
        ),
        Method(
            "Sobol pick-freeze, Sobol'",
            "sensitivity",
            _sobol("sobol"),
            applies=_is_independent,
        ),
        Method(
            "PCE Sobol indices",
            "sensitivity",
            _pce_sobol,
            "approximation",
            _is_independent,
        ),
        Method(
            "Shapley, exact subsets",
            "sensitivity",
            _shapley("exact"),
            applies=_has_group("Sh"),
        ),
        Method(
            "Shapley, random permutations",
            "sensitivity",
            _shapley("random"),
            applies=_has_group("Sh"),
        ),
        Method(
            "Shapley closed form (library)",
            "sensitivity",
            _closed_form_shapley,
            "approximation",
            _is_correlated,
            deterministic=True,
        ),
        Method("Monte Carlo", "propagation", _design("mc")),
        Method("Latin hypercube", "propagation", _design("lhs")),
        Method("Scrambled Halton", "propagation", _design("halton")),
        Method("Scrambled Sobol'", "propagation", _design("sobol")),
        Method("PCE moments", "propagation", _pce_moments, "approximation"),
        Method(
            "Exact posterior (baseline)", "calibration", _exact_posterior, "reference"
        ),
        # Calibration data change with the seed, so even Laplace runs per seed.
        Method("Laplace", "calibration", _laplace, "approximation"),
        Method(
            "Adaptive Metropolis",
            "calibration",
            _mcmc("adaptive-metropolis", "am_draws"),
        ),
        Method("NUTS", "calibration", _mcmc("nuts", "nuts_draws")),
        Method("Static HMC", "calibration", _mcmc("hmc", "hmc_draws")),
        Method("ADVI mean-field", "calibration", _advi("mean-field"), "approximation"),
        Method("ADVI full-rank", "calibration", _advi("full-rank"), "approximation"),
    )
