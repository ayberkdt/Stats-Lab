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

from benchmarks.problems import L96_BURN, L96_CYCLES, Instance, Problem
from benchmarks.references import normal_quantile
from probstat.data_assimilation import (
    Localization,
    Relaxation,
    ensemble_kalman_filter,
)
from probstat.filtering import kalman_filter
from probstat.inference import advi
from probstat.inverse import (
    LinearInverseProblem,
    discrepancy_principle,
    gauss_newton_krylov,
    generalized_cross_validation,
    hybrid_lsqr,
    iteratively_regularized_gauss_newton,
    krylov_regularization,
    l_curve,
    linear_gaussian_posterior,
    newton_trust_region_inversion,
    regularization_path,
    select_regularization,
    total_variation_regularized_least_squares,
    truncated_svd,
)
from probstat.linalg import CountingOperator, singular_value_decomposition
from probstat.propagation import propagate
from probstat.reliability import (
    ak_mcs,
    ak_ss,
    compare_reliability_methods,
    directional_simulation,
    find_design_points,
    form_hlrf,
    mixture_importance_sampling,
    model_monte_carlo,
    moving_particles,
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
            "inverse": 2,
            "assimilation": 2,
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
        "mp_particles": 50,
        "mp_steps": 5,
        "ds_directions": 100,
        "akss_samples": 500,
        "akss_max_dimension": 0,
        "dimension_budget": {"Crude MC": 10, "Multi-point FORM": 10, "Mixture IS": 10},
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
        "enkf_members": 50,
        "l96_members": 20,
        # lambda grid of the deblurring GCV row: 10^low ... 10^high, count.
        "gcv_exponents": (0.0, 1.0, 5),
        "tv_iterations": 300,
    },
    "full": {
        "seeds": {
            "reliability": 20,
            "sensitivity": 10,
            "propagation": 30,
            "calibration": 20,
            "inverse": 10,
            "assimilation": 10,
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
        "mp_particles": 100,
        "mp_steps": 20,
        "ds_directions": 500,
        "akss_samples": 1_000,
        "akss_max_dimension": 6,
        "dimension_budget": {"Crude MC": 10, "Multi-point FORM": 10, "Mixture IS": 10},
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
        "enkf_members": 200,
        "l96_members": 20,
        "gcv_exponents": (-0.5, 1.5, 9),
        "tv_iterations": 2_000,
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
    (problem, name): _ORIGIN_KINK
    for problem in ("four-branch", "four-branch-k7")
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
    notes = [
        f"{result.added} points added, population {result.population}, "
        f"min U {result.min_learning:.3g}"
    ]
    if not result.converged:
        notes.append("not converged: " + result.message)
    return MethodOutput(
        {"Pf": Estimate(result.probability, result.standard_error)}, tuple(notes)
    )


def _dimension_budget(name: str) -> Callable[[Problem, Mapping[str, Any]], str | None]:
    """Return a skip rule for a method whose cost grows quickly with ``d``.

    The limit is ``sizes["dimension_budget"][name]``; a method without an
    entry is never skipped.
    """

    def skip(problem: Problem, sizes: Mapping[str, Any]) -> str | None:
        """Skip above the mode's dimension budget for this method."""
        limit = sizes["dimension_budget"].get(name)
        if limit is not None and problem.dimension > limit:
            return (
                f"skipped: d = {problem.dimension} exceeds this mode's {name} "
                f"budget (d <= {limit})"
            )
        return None

    return skip


def _moving_particles(
    instance: Instance, seed: int, sizes: Mapping[str, Any], problem: Problem
) -> MethodOutput:
    """Run moving particles; report its ideal c.o.v. and its Poisson interval."""
    del problem
    result = moving_particles(
        instance["limit_state"],
        sizes["mp_particles"],
        steps=sizes["mp_steps"],
        seed=seed,
    )
    notes = [
        f"{result.moves} moves, acceptance {result.acceptance_rate:.2f}; the "
        "reported error assumes exact conditional sampling"
    ]
    if not result.converged:
        notes.append("not converged: max_moves reached (estimate biased upward)")
    lower, upper = result.confidence_interval
    return MethodOutput(
        {
            "Pf": Estimate(
                result.probability,
                result.coefficient_of_variation * result.probability,
                (lower, upper),
                result.confidence,
            )
        },
        tuple(notes),
    )


def _directional(
    instance: Instance, seed: int, sizes: Mapping[str, Any], problem: Problem
) -> MethodOutput:
    """Run directional simulation with its central-limit interval."""
    del problem
    result = directional_simulation(
        instance["limit_state"], sizes["ds_directions"], seed=seed
    )
    lower, upper = result.confidence_interval
    return MethodOutput(
        {
            "Pf": Estimate(
                result.probability,
                result.standard_error,
                (lower, upper),
                result.confidence,
            )
        },
        (f"{result.rays} rays, radius_max {result.radius_max:.2f}",),
    )


def _akss_skip(problem: Problem, sizes: Mapping[str, Any]) -> str | None:
    """Skip AK-SS outside its mode budget (seconds to minutes per run)."""
    limit = sizes["akss_max_dimension"]
    if limit < 1:
        return (
            "skipped: AK-SS runs only in full mode at this budget (seconds to "
            "minutes per run in pure Python)"
        )
    if problem.dimension > limit:
        return (
            f"skipped: d = {problem.dimension} exceeds this mode's AK-SS budget "
            f"(d <= {limit})"
        )
    return None


def _akss(
    instance: Instance, seed: int, sizes: Mapping[str, Any], problem: Problem
) -> MethodOutput:
    """Run AK-SS; its SE is the subset c.o.v. of the run on the surrogate."""
    del problem
    result = ak_ss(
        instance["limit_state"], samples_per_level=sizes["akss_samples"], seed=seed
    )
    notes = [
        f"{result.added} points added in {result.rounds} rounds, "
        f"{result.levels} levels; SE excludes surrogate misclassification"
    ]
    if not result.converged:
        notes.append("not converged: " + result.message)
    return MethodOutput(
        {
            "Pf": Estimate(
                result.probability,
                result.coefficient_of_variation * result.probability,
            )
        },
        tuple(notes),
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


# ---------------------------------------------------------------------------
# Inverse problems
# ---------------------------------------------------------------------------


def _relative_error(solution: Any, exact: Any) -> float:
    """Return ``||x - x_exact|| / ||x_exact||``."""
    difference = math.fsum((a - b) ** 2 for a, b in zip(solution, exact, strict=True))
    return math.sqrt(difference / math.fsum(b * b for b in exact))


def _tikhonov_rule(rule: str) -> Runner:
    """Return an adapter choosing the Tikhonov parameter by one rule."""

    def run(
        instance: Instance, seed: int, sizes: Mapping[str, Any], problem: Problem
    ) -> MethodOutput:
        """Choose lambda, solve, and score the solution against the truth."""
        del seed, sizes, problem
        operator = instance["problem"].operator
        data = instance["data"]
        svd = singular_value_decomposition(operator)
        if rule == "discrepancy":
            choice = discrepancy_principle(
                operator, data, instance["noise_norm"], decomposition=svd
            )
        elif rule == "gcv":
            choice = generalized_cross_validation(operator, data, decomposition=svd)
        else:
            choice = l_curve(operator, data, decomposition=svd)
        instance.model.evaluations += len(choice.grid) + 1
        error = _relative_error(choice.solution.solution, instance["exact"])
        notes = [f"lambda = {choice.regularization:.3g}"]
        if choice.at_boundary:
            notes.append("optimum at the grid boundary")
        return MethodOutput({"relerr": Estimate(error)}, tuple(notes))

    return run


def _tsvd_discrepancy(
    instance: Instance, seed: int, sizes: Mapping[str, Any], problem: Problem
) -> MethodOutput:
    """Truncate the SVD at the smallest rank whose residual meets the noise norm."""
    del seed, sizes, problem
    operator = instance["problem"].operator
    data = instance["data"]
    svd = singular_value_decomposition(operator)
    target = instance["noise_norm"]
    solution = None
    for rank in range(1, len(svd.singular_values) + 1):
        solution = truncated_svd(operator, data, rank, decomposition=svd)
        instance.model.evaluations += 1
        if solution.residual_norm <= target:
            break
    assert solution is not None
    error = _relative_error(solution.solution, instance["exact"])
    return MethodOutput({"relerr": Estimate(error)}, (f"rank k = {solution.rank}",))


def _is_dense_inverse(problem: Problem) -> bool:
    """Apply to the dense Tikhonov problems (heat and deriv2)."""
    return bool(problem.tags & {"severely-ill-posed", "mildly-ill-posed"})


def _is_deblur(problem: Problem) -> bool:
    """Apply to the matrix-free deblurring problem."""
    return "deblur" in problem.tags


def _is_tomography(problem: Problem) -> bool:
    """Apply to the tomography prototype."""
    return "tomography" in problem.tags


def _is_decay(problem: Problem) -> bool:
    """Apply to the nonlinear exponential-decay problem."""
    return "nonlinear-least-squares" in problem.tags


def _tomography_lsqr(
    instance: Instance, seed: int, sizes: Mapping[str, Any], problem: Problem
) -> MethodOutput:
    """Early-stopped LSQR with the whitened discrepancy principle (tau = 1.01)."""
    del seed, sizes, problem
    case = instance["case"]
    counted = CountingOperator(case.operator)
    inverse = LinearInverseProblem.create(counted, instance["data"], noise=case.noise)
    result = krylov_regularization(
        inverse, method="lsqr", noise_norm=instance["noise_norm"], max_iterations=200
    )
    notes = [f"stopped at k = {result.iterations} ({result.stop_reason})"]
    notes.append(_spent(instance, counted))
    if not result.converged:
        notes.append("no stopping rule fired")
    error = _relative_error(result.estimate, instance["exact"])
    return MethodOutput({"relerr": Estimate(error)}, tuple(notes))


def _tomography_hybrid(
    instance: Instance, seed: int, sizes: Mapping[str, Any], problem: Problem
) -> MethodOutput:
    """Hybrid LSQR with projected GCV, at most 30 bidiagonalization steps."""
    del seed, sizes, problem
    case = instance["case"]
    counted = CountingOperator(case.operator)
    inverse = LinearInverseProblem.create(counted, instance["data"], noise=case.noise)
    result = hybrid_lsqr(inverse, rule="gcv", max_iterations=30)
    notes = [f"{result.iterations} steps ({result.stop_reason})"]
    notes.append(_spent(instance, counted))
    error = _relative_error(result.estimate, instance["exact"])
    return MethodOutput({"relerr": Estimate(error)}, tuple(notes))


def _decay_spent(instance: Instance, counts: Any) -> str:
    """Add model evaluations of a nonlinear solve to the cost; return a note."""
    total = counts.value + counts.jvp + counts.vjp
    instance.model.evaluations += total
    return f"{counts.value} values + {counts.jvp} jvp + {counts.vjp} vjp"


def _decay_gauss_newton(
    instance: Instance, seed: int, sizes: Mapping[str, Any], problem: Problem
) -> MethodOutput:
    """Gauss-Newton with inner LSQR from ``(1, 0.5)``, unregularized."""
    del seed, sizes, problem
    case = instance["case"]
    result = gauss_newton_krylov(
        case.model, instance["data"], noise=case.noise, initial=(1.0, 0.5)
    )
    notes = [f"{result.iterations} iterations ({result.stop_reason})"]
    notes.append(_decay_spent(instance, result.counts))
    error = _relative_error(result.estimate, instance["exact"])
    return MethodOutput({"relerr": Estimate(error)}, tuple(notes))


def _decay_trust_region(
    instance: Instance, seed: int, sizes: Mapping[str, Any], problem: Problem
) -> MethodOutput:
    """Trust-region Newton-CG (Gauss-Newton Hessian) from ``(1, 0.5)``."""
    del seed, sizes, problem
    case = instance["case"]
    result = newton_trust_region_inversion(
        case.model, instance["data"], noise=case.noise, initial=(1.0, 0.5)
    )
    notes = [f"{result.iterations} iterations ({result.stop_reason})"]
    counts = result.counts
    total = counts.value + counts.jvp + counts.vjp
    instance.model.evaluations += total
    notes.append(f"{total} model, jvp and vjp evaluations")
    error = _relative_error(result.estimate, instance["exact"])
    return MethodOutput({"relerr": Estimate(error)}, tuple(notes))


def _is_squared(problem: Problem) -> bool:
    """Apply to the squared-map problem."""
    return "squared-map" in problem.tags


def _is_elliptic(problem: Problem) -> bool:
    """Apply to the elliptic log-conductivity problem."""
    return "elliptic-pde" in problem.tags


def _is_heat_initial(problem: Problem) -> bool:
    """Apply to the heat initial-condition problem."""
    return "heat-initial" in problem.tags


def _squared_gauss_newton(
    instance: Instance, seed: int, sizes: Mapping[str, Any], problem: Problem
) -> MethodOutput:
    """Gauss-Newton from ``x = (0.8, ..., 0.8)``; scored on ``|x|``."""
    del seed, sizes, problem
    case = instance["case"]
    start = (0.8,) * len(case.truth)
    result = gauss_newton_krylov(
        case.model, instance["data"], noise=case.noise, initial=start
    )
    notes = [f"{result.iterations} iterations ({result.stop_reason})"]
    notes.append(_decay_spent(instance, result.counts))
    magnitudes = [abs(v) for v in result.estimate]
    error = _relative_error(magnitudes, instance["exact"])
    return MethodOutput({"relerr": Estimate(error)}, tuple(notes))


def _elliptic_irgnm(
    instance: Instance, seed: int, sizes: Mapping[str, Any], problem: Problem
) -> MethodOutput:
    """IRGNM from ``x = 0``, ``lambda_0 = 1`` halved, discrepancy 1.1 sqrt(m)."""
    del seed, sizes, problem
    case = instance["case"]
    m = len(instance["data"])
    result = iteratively_regularized_gauss_newton(
        case.model,
        instance["data"],
        noise=case.noise,
        initial_regularization=1.0,
        initial=(0.0,) * len(case.truth),
        discrepancy=1.1 * math.sqrt(m),
    )
    notes = [f"{result.iterations} iterations ({result.stop_reason})"]
    notes.append(_decay_spent(instance, result.counts))
    error = _relative_error(result.estimate, instance["exact"])
    return MethodOutput({"relerr": Estimate(error)}, tuple(notes))


def _heat_lsqr(
    instance: Instance, seed: int, sizes: Mapping[str, Any], problem: Problem
) -> MethodOutput:
    """Early-stopped LSQR with the whitened discrepancy principle (tau = 1.01)."""
    del seed, sizes, problem
    case = instance["case"]
    counted = CountingOperator(case.operator)
    inverse = LinearInverseProblem.create(counted, instance["data"], noise=case.noise)
    result = krylov_regularization(
        inverse, method="lsqr", noise_norm=instance["noise_norm"], max_iterations=200
    )
    notes = [f"stopped at k = {result.iterations} ({result.stop_reason})"]
    notes.append(_spent(instance, counted))
    error = _relative_error(result.estimate, instance["exact"])
    return MethodOutput({"relerr": Estimate(error)}, tuple(notes))


def _heat_hybrid(
    instance: Instance, seed: int, sizes: Mapping[str, Any], problem: Problem
) -> MethodOutput:
    """Hybrid LSQR with the projected discrepancy rule, at most 24 steps."""
    del seed, sizes, problem
    case = instance["case"]
    counted = CountingOperator(case.operator)
    inverse = LinearInverseProblem.create(counted, instance["data"], noise=case.noise)
    result = hybrid_lsqr(
        inverse,
        rule="discrepancy",
        noise_norm=instance["noise_norm"],
        max_iterations=24,
    )
    notes = [f"{result.iterations} steps ({result.stop_reason})"]
    notes.append(_spent(instance, counted))
    error = _relative_error(result.estimate, instance["exact"])
    return MethodOutput({"relerr": Estimate(error)}, tuple(notes))


def _is_gaussian_posterior(problem: Problem) -> bool:
    """Apply to the correlated Gaussian posterior problem."""
    return "gaussian-posterior" in problem.tags


def _spent(instance: Instance, counted: CountingOperator) -> str:
    """Add the operator applications of a wrapper to the cost; return a note."""
    counts = counted.counts
    instance.model.evaluations += counts.forward + counts.transpose
    return f"{counts.forward} forward + {counts.transpose} transpose applications"


def _deblur_lsqr(
    instance: Instance, seed: int, sizes: Mapping[str, Any], problem: Problem
) -> MethodOutput:
    """Early-stopped LSQR stopped by the discrepancy principle (tau = 1.01)."""
    del seed, sizes, problem
    case = instance["case"]
    counted = CountingOperator(case.operator)
    inverse = LinearInverseProblem.create(counted, instance["data"], noise=case.noise)
    result = krylov_regularization(
        inverse, method="lsqr", noise_norm=instance["noise_norm"], max_iterations=200
    )
    notes = [f"stopped at k = {result.iterations} ({result.stop_reason})"]
    notes.append(_spent(instance, counted))
    if not result.converged:
        notes.append("no stopping rule fired")
    error = _relative_error(result.estimate, instance["exact"])
    return MethodOutput({"relerr": Estimate(error)}, tuple(notes))


def _deblur_gcv(
    instance: Instance, seed: int, sizes: Mapping[str, Any], problem: Problem
) -> MethodOutput:
    """Tikhonov ``L = I`` with ``lambda`` chosen by GCV on a regularization path."""
    del seed, problem
    low, high, points = sizes["gcv_exponents"]
    grid = [10.0 ** (low + (high - low) * i / (points - 1)) for i in range(points)]
    case = instance["case"]
    counted = CountingOperator(case.operator)
    inverse = LinearInverseProblem.create(counted, instance["data"], noise=case.noise)
    choice = select_regularization(regularization_path(inverse, grid), "gcv")
    if choice.estimate is None:
        raise ValueError(f"GCV selected no lambda ({choice.status})")
    notes = [f"lambda = {choice.regularization:.3g} ({choice.status})"]
    notes.append(_spent(instance, counted))
    error = _relative_error(choice.estimate, instance["exact"])
    return MethodOutput({"relerr": Estimate(error)}, tuple(notes))


#: Penalty of the total-variation benchmark row. It was chosen once from a scan
#: of ``{0.03, 0.1, 0.3}`` on the data of seeds 101-103 of the deblurring problem
#: (no benchmark run uses those seeds): with 300 PDHG iterations, 0.1 gave
#: relative errors 0.037, 0.041 and 0.037, against 0.059-0.070 at 0.03 and
#: 0.070-0.072 at 0.3.
TV_ALPHA = 0.1


def _deblur_tv(
    instance: Instance, seed: int, sizes: Mapping[str, Any], problem: Problem
) -> MethodOutput:
    """Total variation by PDHG with a fixed penalty and a capped iteration count."""
    del seed, problem
    counted = CountingOperator(instance["case"].operator)
    result = total_variation_regularized_least_squares(
        counted,
        instance["data"],
        TV_ALPHA,
        max_iterations=sizes["tv_iterations"],
    )
    notes = [f"alpha = {TV_ALPHA}, {result.optimization.iterations} PDHG iterations"]
    notes.append(_spent(instance, counted))
    if not result.converged:
        notes.append(f"not converged: {result.stop_reason} (cap of this mode)")
    error = _relative_error(result.solution, instance["exact"])
    return MethodOutput({"relerr": Estimate(error)}, tuple(notes))


def _gaussian_posterior(method: str) -> Runner:
    """Return an adapter for the library's exact Gaussian posterior mean."""

    def run(
        instance: Instance, seed: int, sizes: Mapping[str, Any], problem: Problem
    ) -> MethodOutput:
        """Solve the problem exactly and compare with the independent mean."""
        del seed, sizes, problem
        case = instance["case"]
        counted = CountingOperator(case.operator)
        posterior = linear_gaussian_posterior(
            counted,
            instance["data"],
            noise=case.noise,
            prior_covariance=case.prior_covariance,
            method=method,  # type: ignore[arg-type]
        )
        notes = (
            _spent(instance, counted),
            f"relative error to the withheld truth "
            f"{_relative_error(posterior.mean, instance['truth']):.3f}",
        )
        error = _relative_error(posterior.mean, instance["exact"])
        return MethodOutput({"relerr": Estimate(error)}, notes)

    return run


ORACLE_GRID = tuple(10.0 ** (-6.0 + 6.0 * i / 120) for i in range(121))


def _oracle_tikhonov(
    instance: Instance, seed: int, sizes: Mapping[str, Any], problem: Problem
) -> MethodOutput:
    """Return the smallest Tikhonov error on a lambda grid (needs the truth).

    The solutions come from the SVD filter form
    ``x = V diag(sigma / (sigma^2 + lambda^2)) U'b``, which equals the QR
    solution of ``tikhonov`` for ``L = I`` up to rounding and costs one SVD.
    """
    del seed, sizes, problem
    operator = instance["problem"].operator
    data = instance["data"]
    svd = singular_value_decomposition(operator)
    sigma = svd.singular_values
    beta = [
        math.fsum(svd.u[r][j] * data[r] for r in range(len(data)))
        for j in range(len(sigma))
    ]
    errors = []
    for lam in ORACLE_GRID:
        coefficients = [
            s * b / (s * s + lam * lam) for s, b in zip(sigma, beta, strict=True)
        ]
        solution = [
            math.fsum(row[i] * coefficients[i] for i in range(len(sigma)))
            for row in svd.v
        ]
        errors.append(_relative_error(solution, instance["exact"]))
    instance.model.evaluations += len(ORACLE_GRID)
    best = min(range(len(errors)), key=lambda i: errors[i])
    return MethodOutput(
        {"relerr": Estimate(errors[best])},
        (f"lambda = {ORACLE_GRID[best]:.3g} (uses the exact solution)",),
    )


# ---------------------------------------------------------------------------
# Data assimilation
# ---------------------------------------------------------------------------


def _is_tracking(problem: Problem) -> bool:
    """Apply to the linear-Gaussian tracking problem."""
    return "linear-tracking" in problem.tags


def _is_lorenz96(problem: Problem) -> bool:
    """Apply to the Lorenz-96 twin experiment."""
    return "lorenz96" in problem.tags


def _final_moments(mean: Any, covariance: Any) -> dict[str, Estimate]:
    """Return the final mean and sd estimates of position and velocity."""
    estimates: dict[str, Estimate] = {}
    for i, name in enumerate(("pos", "vel")):
        estimates[f"mean[{name}]"] = Estimate(mean[i])
        estimates[f"sd[{name}]"] = Estimate(math.sqrt(covariance[i][i]))
    return estimates


def _kalman_tracking(
    instance: Instance, seed: int, sizes: Mapping[str, Any], problem: Problem
) -> MethodOutput:
    """Run the library Kalman filter (exact up to rounding)."""
    del seed, sizes, problem
    observations = [(y,) for y in instance["observations"]]
    result = kalman_filter(instance["state_space"], observations)
    final = result.filtered[-1]
    return MethodOutput(_final_moments(final.mean, final.covariance))


def _ensemble_tracking(analysis: str) -> Runner:
    """Return an adapter running one ensemble analysis on the tracking problem."""

    def run(
        instance: Instance, seed: int, sizes: Mapping[str, Any], problem: Problem
    ) -> MethodOutput:
        """Filter with ``enkf_members`` members and report the final moments."""
        del problem
        observations = [[y] for y in instance["observations"]]
        members = sizes["enkf_members"]
        options: dict[str, Any] = {"analysis": analysis}
        notes: tuple[str, ...] = (f"N = {members}",)
        if analysis == "letkf":
            # Both state components sit at the observed point: the local
            # analysis sees the observation with weight one.
            options["localization"] = Localization((0.0, 0.0), (0.0,), 1.0)
            notes = (f"N = {members}, Gaspari-Cohn half-width 1 (weight 1)",)
        result = ensemble_kalman_filter(
            instance["state_space"],
            observations,
            ensemble_size=members,
            seed=seed,
            **options,
        )
        instance.model.evaluations += members * len(observations)
        return MethodOutput(
            _final_moments(result.analysis_means[-1], result.analysis_covariances[-1]),
            notes,
        )

    return run


def _lorenz96_filter(kind: str) -> Runner:
    """Return an adapter running one ensemble filter on the Lorenz-96 twin."""

    def run(
        instance: Instance, seed: int, sizes: Mapping[str, Any], problem: Problem
    ) -> MethodOutput:
        """Filter the twin experiment; report time-averaged RMSE and spread."""
        del problem
        truth = instance["truth"]
        dimension = len(truth.states[0])
        members = sizes["l96_members"]
        options: dict[str, Any] = {}
        if kind != "plain":
            options["localization"] = Localization(
                range(dimension), range(dimension), 2.0, period=dimension
            )
            options["relaxation"] = Relaxation("rtps", 0.5)
        if kind == "letkf":
            options["analysis"] = "letkf"
        result = ensemble_kalman_filter(
            instance["state_space"],
            truth.observations,
            ensemble_size=members,
            seed=seed,
            **options,
        )
        instance.model.evaluations += members * len(truth.observations)
        errors, spreads = [], []
        for k in range(L96_BURN + 1, L96_CYCLES + 1):
            mean = result.analysis_means[k]
            covariance = result.analysis_covariances[k]
            errors.append(
                math.sqrt(
                    math.fsum(
                        (a - b) ** 2 for a, b in zip(mean, truth.states[k], strict=True)
                    )
                    / dimension
                )
            )
            spreads.append(
                math.sqrt(
                    math.fsum(covariance[i][i] for i in range(dimension)) / dimension
                )
            )
        rmse = math.fsum(errors) / len(errors)
        spread = math.fsum(spreads) / len(spreads)
        return MethodOutput(
            {"rmse": Estimate(rmse), "spread_ratio": Estimate(spread / rmse)},
            (f"N = {members}, mean spread {spread:.3f}",),
        )

    return run


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
            skip=_dimension_budget("Multi-point FORM"),
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
        Method(
            "Crude MC",
            "reliability",
            _crude_mc,
            skip=_dimension_budget("Crude MC"),
        ),
        Method("IS at design point", "reliability", _compare("is", "IS")),
        Method(
            "Mixture IS",
            "reliability",
            _mixture,
            skip=_dimension_budget("Mixture IS"),
        ),
        Method("Subset simulation", "reliability", _compare("subset", "Subset")),
        Method("Subset (aCS)", "reliability", _compare("acs", "Subset-aCS")),
        Method("Subset (HMC)", "reliability", _compare("hmc-ss", "Subset-HMC")),
        Method("Moving particles", "reliability", _moving_particles),
        Method("Directional simulation", "reliability", _directional),
        Method("Line sampling", "reliability", _compare("line", "Line")),
        Method("CE-IS", "reliability", _compare("ce", "CE-IS")),
        Method("AK-MCS", "reliability", _akmcs, skip=_akmcs_skip),
        Method("AK-SS", "reliability", _akss, skip=_akss_skip),
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
        # Inverse-problem noise changes with the seed, so every rule runs per seed.
        Method(
            "Tikhonov, error-optimal lambda (oracle)",
            "inverse",
            _oracle_tikhonov,
            "reference",
            _is_dense_inverse,
        ),
        Method(
            "Tikhonov + discrepancy principle",
            "inverse",
            _tikhonov_rule("discrepancy"),
            "approximation",
            _is_dense_inverse,
        ),
        Method(
            "Tikhonov + GCV",
            "inverse",
            _tikhonov_rule("gcv"),
            "approximation",
            _is_dense_inverse,
        ),
        Method(
            "Tikhonov + L-curve",
            "inverse",
            _tikhonov_rule("l-curve"),
            "approximation",
            _is_dense_inverse,
        ),
        Method(
            "TSVD + discrepancy principle",
            "inverse",
            _tsvd_discrepancy,
            "approximation",
            _is_dense_inverse,
        ),
        Method(
            "LSQR + discrepancy principle (matrix-free)",
            "inverse",
            _deblur_lsqr,
            "approximation",
            _is_deblur,
        ),
        Method(
            "Tikhonov + GCV (regularization path)",
            "inverse",
            _deblur_gcv,
            "approximation",
            _is_deblur,
        ),
        Method(
            "Total variation (PDHG, fixed alpha)",
            "inverse",
            _deblur_tv,
            "approximation",
            _is_deblur,
        ),
        Method(
            "LSQR + discrepancy principle (ray transform)",
            "inverse",
            _tomography_lsqr,
            "approximation",
            _is_tomography,
        ),
        Method(
            "Hybrid LSQR + projected GCV (ray transform)",
            "inverse",
            _tomography_hybrid,
            "approximation",
            _is_tomography,
        ),
        Method(
            "Gauss-Newton (inner LSQR)",
            "inverse",
            _decay_gauss_newton,
            "approximation",
            _is_decay,
        ),
        Method(
            "Trust-region Newton-CG",
            "inverse",
            _decay_trust_region,
            "approximation",
            _is_decay,
        ),
        Method(
            "Gauss-Newton from a positive start (|x| scored)",
            "inverse",
            _squared_gauss_newton,
            "approximation",
            _is_squared,
        ),
        Method(
            "IRGNM + discrepancy (adjoint gradients)",
            "inverse",
            _elliptic_irgnm,
            "approximation",
            _is_elliptic,
        ),
        Method(
            "LSQR + discrepancy principle (heat operator)",
            "inverse",
            _heat_lsqr,
            "approximation",
            _is_heat_initial,
        ),
        Method(
            "Hybrid LSQR + projected discrepancy (heat operator)",
            "inverse",
            _heat_hybrid,
            "approximation",
            _is_heat_initial,
        ),
        Method(
            "Exact posterior, data space",
            "inverse",
            _gaussian_posterior("data_space"),
            "approximation",
            _is_gaussian_posterior,
        ),
        Method(
            "Exact posterior, parameter space",
            "inverse",
            _gaussian_posterior("parameter_space"),
            "approximation",
            _is_gaussian_posterior,
        ),
        Method(
            "Kalman filter (library)",
            "assimilation",
            _kalman_tracking,
            "approximation",
            _is_tracking,
        ),
        Method(
            "EnKF (stochastic)",
            "assimilation",
            _ensemble_tracking("stochastic"),
            "approximation",
            _is_tracking,
        ),
        Method(
            "ETKF",
            "assimilation",
            _ensemble_tracking("etkf"),
            "approximation",
            _is_tracking,
        ),  # fmt: skip
        Method(
            "LETKF",
            "assimilation",
            _ensemble_tracking("letkf"),
            "approximation",
            _is_tracking,
        ),  # fmt: skip
        Method(
            "EnKF, no localization",
            "assimilation",
            _lorenz96_filter("plain"),
            "approximation",
            _is_lorenz96,
        ),
        Method(
            "EnKF, Gaspari-Cohn c = 2 + RTPS 0.5",
            "assimilation",
            _lorenz96_filter("localized"),
            "approximation",
            _is_lorenz96,
        ),
        Method(
            "LETKF, c = 2 + RTPS 0.5",
            "assimilation",
            _lorenz96_filter("letkf"),
            "approximation",
            _is_lorenz96,
        ),
    )
