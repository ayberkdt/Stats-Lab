"""Inference entry points for traced probabilistic programs.

Every function traces the model once (:func:`~._program.trace`), hands the
unconstrained log-density to an existing engine of
:mod:`probstat.inference`, and maps the draws back to named constrained
values by re-running the model for each draw.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_non_negative_integer
from probstat.linalg import Vector

from ..hamiltonian import HamiltonianResult, hamiltonian_monte_carlo, no_u_turn_sampler
from ..mcmc import (
    LaplaceApproximation,
    MarkovChain,
    MCMCResult,
    ParameterSummary,
    adaptive_metropolis,
    laplace_approximation,
)
from ..predictive_checks import (
    Discrepancy,
    PosteriorPredictivePValues,
    Statistic,
    posterior_predictive_pvalues,
)
from ..variational import VariationalResult, advi
from ._discrete import DiscreteDraws, recover_discrete
from ._errors import ModelSpecificationError
from ._program import (
    Model,
    ModelHandle,
    ModelIR,
    SiteKind,
    SiteValue,
    flatten,
    missing_data_message,
    ppl_gradient_option,
    trace,
)

Engine = HamiltonianResult | MCMCResult | VariationalResult | LaplaceApproximation
GradientOption = Callable[[Vector], Sequence[float]] | str | None


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


def _site_draws(
    result: MCMCResult, program: ModelIR, name: str
) -> tuple[SiteValue, ...]:
    """Return the pooled draws of one latent or deterministic site.

    References:
        Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

    """
    site = next((s for s in program.value_sites if s.name == name), None)
    if site is None:
        raise KeyError(f"{name!r} is not a latent or deterministic site")
    indices = [result.index(label) for label in site.flat_names()]
    pooled = result.pooled()
    if site.structure is not None:
        structure = site.structure
        return tuple(
            structure.unflatten([draw[i] for i in indices]) for draw in pooled
        )
    if site.size is None:
        return tuple(draw[indices[0]] for draw in pooled)
    return tuple(tuple(draw[i] for i in indices) for draw in pooled)


def _mean(draws: Sequence[SiteValue]) -> SiteValue:
    """Return the elementwise mean of scalar, vector, or matrix draws.

    References:
        [EQ-MC-MEAN-001], [REF-ROBERT-CASELLA-2004, Chapter 3].

    """
    first = draws[0]
    if isinstance(first, tuple):
        width = len(first)
        if width and isinstance(first[0], tuple):
            matrices = cast(Sequence[tuple[tuple[float, ...], ...]], draws)
            return tuple(
                tuple(
                    math.fsum(m[i][j] for m in matrices) / len(draws)
                    for j in range(len(first[0]))
                )
                for i in range(width)
            )
        vectors = cast(Sequence[tuple[float, ...]], draws)
        return tuple(
            math.fsum(d[i] for d in vectors) / len(draws) for i in range(width)
        )
    return math.fsum(float(d) for d in draws if not isinstance(d, tuple)) / len(draws)


@dataclass(frozen=True, slots=True)
class PosteriorSamples:
    """Posterior draws of a probabilistic program, keyed by site name.

    Args:
        program: Traced model.
        method: Inference method label (``"nuts"``, ``"hmc"``,
            ``"metropolis"``, ``"advi"``, or ``"laplace"``).
        constrained: Draws of latent and deterministic elements on their
            natural scale, as an :class:`~probstat.inference.MCMCResult` named
            by :attr:`ModelIR.constrained_names` (``theta[0]``, ...). Chain
            structure is kept, so ``summary()``, ``R-hat`` and ESS apply.
        unconstrained: The engine's draws on ``R^d``.
        engine: Raw engine output (:class:`HamiltonianResult` for NUTS/HMC,
            :class:`MCMCResult` for Metropolis, :class:`VariationalResult`,
            or :class:`LaplaceApproximation`).
        point: Constrained values at the Laplace mode or the variational mean
            (both located on the unconstrained scale), else ``None``.
        discrete: Recovered draws and exact conditional probabilities of the
            marginalized discrete sites (``None`` if the model has none).

    References:
        [REF-CARPENTER-ETAL-2017], [REF-GELMAN-ETAL-2013, Chapter 11].

    """

    program: ModelIR
    method: str
    constrained: MCMCResult
    unconstrained: MCMCResult
    engine: Engine
    point: dict[str, SiteValue] | None = None
    discrete: DiscreteDraws | None = None

    def __getitem__(self, name: str) -> tuple[SiteValue, ...]:
        """Return pooled draws of a site.

        Floats for scalar sites, tuples for vector sites, tuples of rows for
        matrix sites, and support values for discrete sites. Under dynamic
        site structure, a draw whose recovered configuration does not
        declare the site holds ``None`` (``posterior.constrained`` keeps its
        pseudo-prior value there).

        Raises:
            KeyError: If ``name`` is not a latent, deterministic, or
                discrete site.

        References:
            Project API policy [REF-PROBSTAT-API]. Equation
            [EQ-PPL-DYNAMIC-001].

        """
        if self.discrete is not None and name in self.discrete.draws:
            return self.discrete.draws[name]
        draws = _site_draws(self.constrained, self.program, name)
        if self._always_present(name):
            return draws
        mask = self.present(name)
        masked: list[Any] = [
            draw if keep else None for draw, keep in zip(draws, mask, strict=True)
        ]
        return tuple(masked)

    def _always_present(self, name: str) -> bool:
        """Return whether every traced configuration declares a site.

        References:
            [EQ-PPL-DYNAMIC-001]. No new equation is introduced.

        """
        program = self.program
        return not program.dynamic or len(program.presence(name)) == len(
            program.structures
        )

    def present(self, name: str) -> tuple[bool, ...]:
        """Return, per draw, whether the recovered configuration declares a site.

        Always true for a model with a static site structure.

        Raises:
            KeyError: If no traced configuration declares ``name``.

        References:
            [EQ-PPL-DYNAMIC-001], [REF-CARLIN-CHIB-1995].

        """
        indices = self.program.presence(name)
        if self._always_present(name) or self.discrete is None:
            return (True,) * self.draw_count
        return tuple(chosen in indices for chosen in self.discrete.structures)

    def presence_probability(self, name: str) -> float:
        """Return the posterior probability that a site is declared.

        ``(1/S) sum_s sum_(c declares the site) p(c | theta^(s), y)``, the
        Rao-Blackwellized estimate of the posterior probability of the
        structures that contain the site (``1`` under a static structure).

        Raises:
            KeyError: If no traced configuration declares ``name``.

        Equation:
            [EQ-PPL-ENUM-002], [EQ-PPL-DYNAMIC-001].

        References:
            [REF-CARLIN-CHIB-1995], [REF-STAN-USERS-GUIDE-2-36].

        """
        indices = self.program.presence(name)
        if self._always_present(name) or self.discrete is None:
            return 1.0
        rows = self.discrete.structure_probabilities
        return math.fsum(math.fsum(row[c] for c in indices) for row in rows) / len(
            rows
        )

    def discrete_probabilities(self, name: str) -> dict[object, float]:
        """Return the posterior probability of each value of a discrete site.

        The average over draws of the exact conditional probabilities
        ``p(z = v | theta^(s), y)`` (a Rao-Blackwellized estimate).

        Raises:
            KeyError: If ``name`` is not a marginalized discrete site.

        Equation:
            [EQ-PPL-ENUM-002].

        References:
            [REF-STAN-USERS-GUIDE-2-36].

        """
        if self.discrete is None or name not in self.discrete.supports:
            raise KeyError(f"{name!r} is not a marginalized discrete site")
        return dict(
            zip(
                self.discrete.supports[name],
                self.discrete.marginal(name),
                strict=True,
            )
        )

    @property
    def gradient_method(self) -> str | None:
        """Return the gradient method of the engine (``None`` for Metropolis).

        ``"reverse"`` when the traceable program was differentiated in
        reverse mode (the default), ``"autodiff"`` for forward mode,
        ``"finite-difference"`` otherwise (see
        :attr:`gradient_fallback_reason`).

        References:
            [REF-GRIEWANK-WALTHER-2008, Chapter 3]. No equation is evaluated.

        """
        return getattr(self.engine, "gradient_method", None)

    @property
    def gradient_fallback_reason(self) -> str | None:
        """Return why automatic differentiation was not used, if requested.

        References:
            [REF-GRIEWANK-WALTHER-2008, Chapter 3]. No equation is evaluated.

        """
        for attribute in ("gradient_fallback_reason", "fallback_reason"):
            value = getattr(self.engine, attribute, None)
            if value is not None:
                return str(value)
        return None

    @property
    def warnings(self) -> tuple[str, ...]:
        """Return the convergence warnings recorded by the engine.

        References:
            [REF-VEHTARI-ETAL-2021]. No equation is evaluated.

        """
        return tuple(getattr(self.engine, "warnings", ()))

    @property
    def names(self) -> tuple[str, ...]:
        """Return latent and deterministic site names, then discrete ones.

        References:
            Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

        """
        names = tuple(site.name for site in self.program.value_sites)
        if self.discrete is not None:
            names += self.discrete.names
        return names

    @property
    def draw_count(self) -> int:
        """Return the number of pooled draws.

        References:
            Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

        """
        return len(self.constrained.chains) * self.constrained.draws_per_chain

    def mean(self, name: str) -> SiteValue:
        """Return the posterior mean of a site.

        For a discrete site with numeric values this is the Rao-Blackwellized
        mean ``sum_v v p(z = v | y)`` from :meth:`discrete_probabilities`
        (elementwise for a vector site). Under dynamic site structure, it is
        the posterior mean given that the site is declared: for a continuous
        site, the mean over the draws whose recovered configuration declares
        it; for a discrete site, the Rao-Blackwellized mean divided by the
        probability that it is declared.

        Raises:
            KeyError: If ``name`` is not a site.
            ValueError: If no draw declares the site.

        Equation:
            [EQ-MC-MEAN-001], [EQ-PPL-ENUM-002].

        References:
            [REF-ROBERT-CASELLA-2004, Chapter 3], [REF-STAN-USERS-GUIDE-2-36].

        """
        if self.discrete is not None and name in self.discrete.supports:
            probabilities = self.discrete_probabilities(name)
            mass = 1.0
            if not self._always_present(name):
                mass = math.fsum(probabilities.values())
            if mass == 0.0:
                raise ValueError(f"no draw declares the site {name!r}")
            first = next(iter(probabilities))
            if isinstance(first, tuple):
                width = len(first)
                return tuple(
                    math.fsum(
                        float(cast(tuple[float, ...], value)[k]) * p
                        for value, p in probabilities.items()
                    )
                    / mass
                    for k in range(width)
                )
            total = math.fsum(
                float(cast(float, value)) * p for value, p in probabilities.items()
            )
            return total / mass if mass != 1.0 else total
        draws = [draw for draw in self[name] if draw is not None]
        if not draws:
            raise ValueError(f"no draw declares the site {name!r}")
        return _mean(draws)

    def draws(self) -> tuple[dict[str, SiteValue], ...]:
        """Return every pooled draw as ``{site name: value}`` (chain after chain).

        Recovered discrete values are included when the model has
        marginalized discrete sites. Under dynamic site structure, each draw
        holds only the sites its recovered configuration declares.

        References:
            Project API policy [REF-PROBSTAT-API]. Equation [EQ-PPL-DYNAMIC-001].

        """
        values = tuple(self.program.unflatten(d) for d in self.constrained.pooled())
        if self.discrete is None:
            return values
        discrete = self.discrete
        if self.program.dynamic:
            structures = self.program.structures
            declared = [frozenset(s.names) for s in structures]
            return tuple(
                {
                    **{k: v for k, v in draw.items() if k in declared[chosen]},
                    **{
                        name: discrete.draws[name][s]
                        for name in discrete.draws
                        if discrete.draws[name][s] is not None
                    },
                }
                for s, (draw, chosen) in enumerate(
                    zip(values, discrete.structures, strict=True)
                )
            )
        return tuple(
            {**draw, **{name: discrete.draws[name][s] for name in discrete.draws}}
            for s, draw in enumerate(values)
        )

    def summary(
        self, probabilities: Sequence[float] = (0.05, 0.5, 0.95)
    ) -> tuple[ParameterSummary, ...]:
        """Summarize every constrained element (mean, quantiles, ESS, R-hat).

        Equation:
            [EQ-MCMC-RHAT-002], [EQ-MCMC-ESS-001], [EQ-MCMC-MCSE-001].

        References:
            [REF-VEHTARI-ETAL-2021].

        """
        return self.constrained.summary(probabilities)

    @property
    def hamiltonian(self) -> HamiltonianResult | None:
        """Return NUTS/HMC diagnostics attached to the constrained draws.

        The per-chain divergence, step-size, and tree-depth statistics of the
        engine are combined with :attr:`constrained`; ``None`` for other
        methods.

        References:
            [REF-HOFFMAN-GELMAN-2014]. No equation is evaluated.

        """
        if isinstance(self.engine, HamiltonianResult):
            return HamiltonianResult(self.constrained, self.engine.statistics)
        return None


@dataclass(frozen=True, slots=True)
class PredictiveSamples:
    """Simulated values of every site of a model, keyed by site name.

    Under dynamic site structure a site that a simulated configuration does
    not declare holds ``None`` in that draw.

    Args:
        sites: ``{name: draws}``; a draw is a float or a tuple of floats.
        kinds: ``{name: "latent" | "observed" | "deterministic"}``.
        draws: Number of simulated draws.

    References:
        [REF-GELMAN-ETAL-2013, Chapter 6], [REF-BINGHAM-ETAL-2019].

    """

    sites: dict[str, tuple[SiteValue, ...]]
    kinds: dict[str, SiteKind]
    draws: int

    def __getitem__(self, name: str) -> tuple[SiteValue, ...]:
        """Return the draws of one site.

        Raises:
            KeyError: If no site has that name.

        References:
            Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

        """
        if name not in self.sites:
            raise KeyError(f"unknown site {name!r}")
        return self.sites[name]

    @property
    def names(self) -> tuple[str, ...]:
        """Return site names in execution order.

        References:
            Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

        """
        return tuple(self.sites)

    def mean(self, name: str) -> SiteValue:
        """Return the Monte Carlo mean of a site over the draws that declare it.

        Raises:
            ValueError: If no draw declares the site.

        Equation:
            [EQ-MC-MEAN-001].

        References:
            [REF-ROBERT-CASELLA-2004, Chapter 3].

        """
        draws = [draw for draw in self[name] if draw is not None]
        if not draws:
            raise ValueError(f"no draw declares the site {name!r}")
        return _mean(draws)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _inference_program(model: Model) -> ModelIR:
    """Trace a model and check that it is ready for inference.

    References:
        Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

    """
    program = trace(model)
    if program.dimension == 0:
        raise ModelSpecificationError("the model has no latent sites to infer")
    for site in program.observed_sites:
        if site.value is None:
            raise ModelSpecificationError(missing_data_message(site.name))
    return program


def _with_constrained(
    program: ModelIR,
    method: str,
    unconstrained: MCMCResult,
    engine: Engine,
    point: dict[str, SiteValue] | None = None,
    stream: RandomStream | None = None,
) -> PosteriorSamples:
    """Map unconstrained draws to the constrained scale and wrap them.

    When the model has marginalized discrete sites, they are recovered draw
    by draw from their exact conditional law with the child stream
    ``"ppl-discrete"`` of ``stream``.

    References:
        [REF-CARPENTER-ETAL-2017], [REF-STAN-USERS-GUIDE-2-36]. Equations
        [EQ-PPL-TRANSFORM-001], [EQ-PPL-ENUM-002].

    """
    if program.per_configuration:
        return _with_configurations(
            program, method, unconstrained, engine, point, stream
        )
    constrained = unconstrained.transformed(
        program.flat_constrained, program.constrained_names
    )
    discrete = None
    if program.discrete_sites:
        source = stream if stream is not None else resolve_stream(0)
        discrete = recover_discrete(
            (program.discrete_conditional(u) for u in unconstrained.pooled()),
            source.child("ppl-discrete"),
        )
    return PosteriorSamples(
        program, method, constrained, unconstrained, engine, point, discrete
    )


def _with_configurations(
    program: ModelIR,
    method: str,
    unconstrained: MCMCResult,
    engine: Engine,
    point: dict[str, SiteValue] | None,
    stream: RandomStream | None,
) -> PosteriorSamples:
    """Recover the configuration of every draw, then report values under it.

    For a model evaluated configuration by configuration (dynamic site
    structure, or continuous sites after a global discrete site), the
    global configuration ``c^(s) ~ p(c | theta^(s), y)`` is drawn first and
    the constrained values of draw ``s`` are those of configuration
    ``c^(s)``; sites it does not declare keep their pseudo-prior values.

    References:
        [EQ-PPL-DYNAMIC-001], [EQ-PPL-ENUM-002], [REF-CARLIN-CHIB-1995].

    """
    records = [program.recovery_record(u) for u in unconstrained.pooled()]
    source = stream if stream is not None else resolve_stream(0)
    discrete = recover_discrete(
        (conditional for conditional, _ in records),
        source.child("ppl-discrete"),
        keep_structures=True,
    )
    flat = iter(
        [
            flatten(values[chosen], program.value_sites)
            for (_, values), chosen in zip(records, discrete.structures, strict=True)
        ]
    )

    def next_draw(_: Vector) -> Vector:
        """Return the precomputed constrained values of the next draw.

        ``MCMCResult.transformed`` visits the draws chain after chain, the
        order of ``pooled``.

        References:
            [EQ-PPL-DYNAMIC-001]. No new equation is introduced.

        """
        return next(flat)

    constrained = unconstrained.transformed(next_draw, program.constrained_names)
    return PosteriorSamples(
        program, method, constrained, unconstrained, engine, point, discrete
    )


def _independent_result(
    program: ModelIR,
    samples: Sequence[Vector],
    covariance: Sequence[Sequence[float]],
    label: str,
    seed: int,
) -> MCMCResult:
    """Wrap independent unconstrained draws as a one-chain result.

    References:
        [REF-ROBERT-CASELLA-2004, Chapter 3]. No equation is evaluated.

    """
    log_density = tuple(program.log_density(s) for s in samples)
    chain = MarkovChain(
        tuple(samples),
        log_density,
        1.0,
        math.nan,
        tuple(tuple(row) for row in covariance),
        len(samples),
        seed,
    )
    return MCMCResult((chain,), program.unconstrained_names, label, 0, 1)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def sample(
    model: Model,
    *,
    method: Literal["nuts", "hmc", "metropolis"] = "nuts",
    draws: int = 1000,
    warmup: int = 1000,
    chains: int = 4,
    thin: int = 1,
    seed: int | RandomStream | None = None,
    init_radius: float = 2.0,
    target_acceptance: float = 0.8,
    max_tree_depth: int = 10,
    leapfrog_steps: int = 16,
    step_size_jitter: float = 0.2,
    gradient: GradientOption = "auto",
    variant: Literal["multinomial", "slice"] = "multinomial",
) -> PosteriorSamples:
    """Draw posterior samples of a probabilistic program by MCMC.

    The model is traced, its joint log-density on the unconstrained space
    ([EQ-PPL-JOINT-001]) is passed to :func:`no_u_turn_sampler`,
    :func:`hamiltonian_monte_carlo`, or :func:`adaptive_metropolis`, and every
    kept draw is mapped back to the constrained scale by re-running the
    model. NUTS and HMC differentiate the model by reverse-mode autodiff
    when the program is traceable (one model execution recorded on a tape
    and one backward sweep per leapfrog step, [EQ-AD-REVERSE-001]);
    otherwise they use central finite differences (``2 d + 1`` executions
    per step). ``gradient_method`` of the result says which was used
    (``"reverse"`` or ``"finite-difference"``).

    Args:
        model: Model function ``model(m)``.
        method: ``"nuts"`` (default), ``"hmc"``, or ``"metropolis"``
            (adaptive Metropolis with adaptation confined to warm-up).
        draws: Kept draws per chain.
        warmup: Warm-up iterations per chain (positive for Metropolis).
        chains: Number of chains.
        thin: Keep every ``thin``-th post-warm-up draw.
        seed: Integer seed or stream; starting points and sampler use
            separate child streams.
        init_radius: Starting points are uniform on ``[-r, r]^d`` in
            unconstrained space.
        target_acceptance: Dual-averaging target for NUTS and HMC.
        max_tree_depth: NUTS tree-depth limit.
        leapfrog_steps: Leapfrog steps per HMC iteration.
        step_size_jitter: HMC step-size jitter (breaks periodic trajectories).
        gradient: ``"auto"`` (reverse mode with finite-difference
            fallback), ``"reverse"``, ``"autodiff"``/``"forward"``,
            ``"finite-difference"``/``"finite"``, or an analytic gradient
            on the unconstrained space (NUTS and HMC only).
        variant: NUTS state selection, ``"multinomial"`` or ``"slice"``.

    Returns:
        Draws by site name with the engine result attached.

    Raises:
        ModelSpecificationError: If the model is invalid for inference.
        ValueError: For an unknown method or invalid run settings.

    Equation:
        [EQ-PPL-JOINT-001], [EQ-PPL-TRANSFORM-001], [EQ-PPL-TRANSFORM-002].

    References:
        [REF-CARPENTER-ETAL-2017], [REF-HOFFMAN-GELMAN-2014], [REF-NEAL-2011],
        [REF-HAARIO-SAKSMAN-TAMMINEN-2001].

    """
    if method not in ("nuts", "hmc", "metropolis"):
        raise ValueError("method must be 'nuts', 'hmc', or 'metropolis'")
    count = require_non_negative_integer(chains, name="chains")
    if count == 0:
        raise ValueError("chains must be positive")
    program = _inference_program(model)
    stream = resolve_stream(seed)
    starts = [
        program.initial_point(child, radius=init_radius)
        for child in stream.child("ppl-init").spawn(count, label="chain")
    ]
    names = program.unconstrained_names
    engine_gradient = cast(GradientOption, ppl_gradient_option(gradient))
    engine: Engine
    sampler_stream = stream.child("ppl-sampler")
    if method == "metropolis":
        engine = adaptive_metropolis(
            program.log_density,
            starts,
            draws=draws,
            warmup=warmup,
            chains=count,
            thin=thin,
            seed=sampler_stream,
            names=names,
        )
        unconstrained = engine
    elif method == "nuts":
        engine = no_u_turn_sampler(
            program.log_density,
            starts,
            draws=draws,
            warmup=warmup,
            chains=count,
            thin=thin,
            max_tree_depth=max_tree_depth,
            target_acceptance=target_acceptance,
            seed=sampler_stream,
            names=names,
            gradient=engine_gradient,
            variant=variant,
        )
        unconstrained = engine.mcmc
    else:
        engine = hamiltonian_monte_carlo(
            program.log_density,
            starts,
            draws=draws,
            warmup=warmup,
            chains=count,
            thin=thin,
            leapfrog_steps=leapfrog_steps,
            step_size_jitter=step_size_jitter,
            target_acceptance=target_acceptance,
            seed=sampler_stream,
            names=names,
            gradient=engine_gradient,
        )
        unconstrained = engine.mcmc
    return _with_constrained(program, method, unconstrained, engine, stream=stream)


def fit_advi(
    model: Model,
    *,
    family: Literal["mean-field", "full-rank"] = "mean-field",
    draws: int = 1000,
    max_iterations: int = 10000,
    learning_rate: float = 0.02,
    gradient_samples: int = 1,
    tolerance: float = 1e-3,
    seed: int | RandomStream | None = None,
    init_radius: float = 0.0,
    gradient: GradientOption = "auto",
) -> PosteriorSamples:
    """Fit a Gaussian ADVI approximation on the unconstrained space.

    The Gaussian ``q`` lives on ``R^d``; its draws are mapped through the
    site bijections, so the implied approximation on the natural scale is
    log-normal for positive sites and logit-normal for interval sites.

    Args:
        model: Model function.
        family: ``"mean-field"`` or ``"full-rank"``.
        draws: Independent draws from the fitted ``q``.
        max_iterations: Adam iteration cap.
        learning_rate: Adam step size.
        gradient_samples: Monte Carlo draws per gradient estimate.
        tolerance: Relative ELBO change regarded as converged.
        seed: Integer seed or stream.
        init_radius: Radius of the starting mean (``0`` is the origin).
        gradient: ``"auto"`` (reverse mode with finite-difference
            fallback), ``"reverse"``, ``"autodiff"``/``"forward"``,
            ``"finite-difference"``/``"finite"``, or an analytic gradient
            on the unconstrained space.

    Returns:
        Draws by site name; ``engine`` is the :class:`VariationalResult` and
        ``point`` the constrained image of the variational mean.

    Raises:
        ModelSpecificationError: If the model is invalid for inference.

    Equation:
        [EQ-PPL-JOINT-001], [EQ-VI-ELBO-001].

    References:
        [REF-KUCUKELBIR-ETAL-2017], [REF-CARPENTER-ETAL-2017].

    """
    program = _inference_program(model)
    stream = resolve_stream(seed)
    start = program.initial_point(stream.child("ppl-init"), radius=init_radius)
    engine = advi(
        program.log_density,
        start,
        family=family,
        draws=draws,
        max_iterations=max_iterations,
        learning_rate=learning_rate,
        gradient_samples=gradient_samples,
        tolerance=tolerance,
        seed=stream.child("ppl-advi"),
        names=program.unconstrained_names,
        gradient=cast(GradientOption, ppl_gradient_option(gradient)),
    )
    return _with_constrained(
        program,
        "advi",
        engine.as_mcmc_result(),
        engine,
        program.constrain(engine.mean),
        stream,
    )


def laplace(
    model: Model,
    *,
    draws: int = 1000,
    seed: int | RandomStream | None = None,
    init_radius: float = 0.0,
    gradient: GradientOption = "auto",
) -> PosteriorSamples:
    """Approximate the posterior by a Gaussian at the unconstrained mode.

    The mode of the unconstrained density (which includes the log-Jacobian)
    is found by BFGS and the covariance is the inverse Hessian
    ([EQ-LAPLACE-001]), both by reverse-mode autodiff (gradient) and forward
    over reverse (Hessian, [EQ-AD-FOR-001]) when the program is traceable
    and by finite differences otherwise. Because of the
    Jacobian, the reported ``point`` is generally *not* the posterior mode on
    the natural scale (for ``tau = exp(u)`` it is the mode of ``tau`` times
    ``p(tau)``).

    Args:
        model: Model function.
        draws: Draws from ``N(mode, H^-1)`` mapped to the natural scale.
        seed: Integer seed or stream.
        init_radius: Radius of the BFGS starting point (``0`` is the origin).
        gradient: ``"auto"`` (reverse mode with finite-difference
            fallback), ``"reverse"``, ``"autodiff"``/``"forward"``,
            ``"finite-difference"``/``"finite"``, or an analytic gradient
            on the unconstrained space.

    Returns:
        Draws by site name; ``engine`` is the :class:`LaplaceApproximation`.

    Raises:
        ModelSpecificationError: If the model is invalid for inference.
        ValueError: If the Hessian at the located point is not definite.

    Equation:
        [EQ-PPL-JOINT-001], [EQ-LAPLACE-001].

    References:
        [REF-TIERNEY-KADANE-1986], [REF-GELMAN-ETAL-2013, Section 4.1],
        [REF-CARPENTER-ETAL-2017].

    """
    program = _inference_program(model)
    stream = resolve_stream(seed)
    start = program.initial_point(stream.child("ppl-init"), radius=init_radius)
    engine = laplace_approximation(
        program.log_density,
        start,
        gradient=cast(GradientOption, ppl_gradient_option(gradient)),
    )
    sampler = stream.child("ppl-laplace")
    samples = engine.sample(draws, seed=sampler)
    if not samples:
        raise ValueError("draws must be positive")
    unconstrained = _independent_result(
        program, samples, engine.covariance, "Laplace", sampler.seed
    )
    return _with_constrained(
        program,
        "laplace",
        unconstrained,
        engine,
        program.constrain(engine.mode),
        stream,
    )


# ---------------------------------------------------------------------------
# Predictive simulation
# ---------------------------------------------------------------------------


def _as_float(value: object) -> SiteValue:
    """Convert a simulated site value to floats (nested for matrices).

    References:
        Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

    """
    if isinstance(value, tuple):
        if value and isinstance(value[0], tuple | list):
            return tuple(tuple(float(v) for v in row) for row in value)
        return tuple(float(v) for v in value)
    return float(value)  # type: ignore[arg-type]


def _simulate(
    model: Model,
    fixed_draws: Iterable[Mapping[str, SiteValue] | None],
    stream: RandomStream,
) -> PredictiveSamples:
    """Run a model in generate mode for each set of fixed latent values.

    References:
        [REF-GELMAN-ETAL-2013, Chapter 6]. Equation [EQ-PPL-PREDICTIVE-001].

    """
    if not callable(model):
        raise TypeError("model must be callable")
    generator = stream.generator()
    sites: dict[str, list[SiteValue | None]] = {}
    kinds: dict[str, SiteKind] = {}
    count = 0
    for fixed in fixed_draws:
        handle = ModelHandle(
            "generate",
            fixed=fixed,
            require_fixed=fixed is not None,
            generator=generator,
        )
        model(handle)
        for record in handle.records:
            if record.name not in sites:
                # Dynamic site structure: a site first met in a later draw is
                # absent (None) in the earlier ones.
                sites[record.name] = [None] * count
                kinds[record.name] = record.kind
            elif kinds[record.name] != record.kind:
                raise ModelSpecificationError(
                    f"site {record.name!r} changed kind between predictive draws"
                )
            sites[record.name].append(
                record.value if record.kind == "discrete" else _as_float(record.value)
            )
        count += 1
        for values in sites.values():
            if len(values) < count:
                values.append(None)
    return PredictiveSamples(
        cast(
            dict[str, tuple[SiteValue, ...]],
            {name: tuple(values) for name, values in sites.items()},
        ),
        kinds,
        count,
    )


def prior_predictive(
    model: Model, draws: int, *, seed: int | RandomStream | None = None
) -> PredictiveSamples:
    """Simulate every site from the prior predictive distribution.

    Each draw runs the model with latent sites drawn from their priors (in
    execution order, so hierarchies are respected) and observed sites drawn
    from their distributions; observed data in the model are ignored.

    Args:
        model: Model function.
        draws: Positive number of simulations.
        seed: Integer seed or stream (same seed, same draws).

    Returns:
        Simulated values of every site.

    Raises:
        ValueError: If ``draws`` is not positive.

    Equation:
        [EQ-PPL-PREDICTIVE-001].

    References:
        [REF-GELMAN-ETAL-2013, Chapter 6], [REF-BINGHAM-ETAL-2019].

    """
    count = require_non_negative_integer(draws, name="draws")
    if count == 0:
        raise ValueError("draws must be positive")
    return _simulate(model, (None for _ in range(count)), resolve_stream(seed))


def _posterior_draws(
    posterior: PosteriorSamples | Sequence[Mapping[str, SiteValue]],
    max_draws: int | None,
) -> tuple[Mapping[str, SiteValue], ...]:
    """Return posterior draws as name-value mappings, thinned to a cap.

    References:
        Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

    """
    values = (
        posterior.draws()
        if isinstance(posterior, PosteriorSamples)
        else tuple(dict(d) for d in posterior)
    )
    if not values:
        raise ValueError("posterior must contain at least one draw")
    if max_draws is not None:
        cap = require_non_negative_integer(max_draws, name="max_draws")
        if cap == 0:
            raise ValueError("max_draws must be positive")
        stride = math.ceil(len(values) / cap)
        values = values[::stride]
    return values


def posterior_predictive(
    model: Model,
    posterior: PosteriorSamples | Sequence[Mapping[str, SiteValue]],
    *,
    seed: int | RandomStream | None = None,
    max_draws: int | None = None,
) -> PredictiveSamples:
    """Simulate replicated data from the posterior predictive distribution.

    For every posterior draw the model is re-run with each latent site fixed
    to its drawn value (by name, on the natural scale) and every observed
    site drawn from its distribution, so ``y_rep^(s) ~ p(y | theta^(s))``.
    Deterministic sites are recomputed.

    Args:
        model: Model function (usually the one that was fitted).
        posterior: :class:`PosteriorSamples` or a sequence of
            ``{site name: value}`` draws.
        seed: Integer seed or stream.
        max_draws: Optional cap; draws are thinned by a common stride.

    Returns:
        Simulated values of every site (latent sites repeat the posterior).

    Raises:
        ModelSpecificationError: If a latent site has no posterior value.

    Equation:
        [EQ-PPL-PREDICTIVE-001].

    References:
        [REF-GELMAN-ETAL-2013, Chapter 6], [REF-RUBIN-1984].

    """
    values = _posterior_draws(posterior, max_draws)
    return _simulate(model, values, resolve_stream(seed))


def predictive_pvalues(
    model: Model,
    posterior: PosteriorSamples,
    site: str,
    *,
    statistics: Mapping[str, Statistic] | None = None,
    discrepancies: Mapping[str, Discrepancy] | None = None,
    max_draws: int | None = None,
    keep_replicates: bool = False,
    seed: int | RandomStream | None = None,
) -> PosteriorPredictivePValues:
    """Run :func:`posterior_predictive_pvalues` for one observed site.

    The draws are ``posterior.constrained`` (chain structure kept, so the
    Monte Carlo error uses the indicator ESS); the simulator re-runs the
    model with latent values fixed and returns the replicated ``site``. A
    discrepancy ``T(y, theta)`` receives ``theta`` as the flat constrained
    vector ordered as ``posterior.program.constrained_names``.

    Args:
        model: Model function.
        posterior: Result of :func:`sample`, :func:`fit_advi`, or
            :func:`laplace`.
        site: Name of an observed site.
        statistics: ``{name: T(y)}`` data-only test statistics.
        discrepancies: ``{name: T(y, theta)}`` realized discrepancies.
        max_draws: Optional cap on the posterior draws used.
        keep_replicates: Whether to return the replicated datasets.
        seed: Integer seed or stream.

    Returns:
        Posterior predictive p-values and their Monte Carlo errors.

    Raises:
        KeyError: If ``site`` is not an observed site.

    Equation:
        [EQ-PPC-001], [EQ-PPL-PREDICTIVE-001].

    References:
        [REF-GELMAN-MENG-STERN-1996], [REF-GELMAN-ETAL-2013, Chapter 6].

    """
    program = posterior.program
    observed = program.observed_data(site)
    data = observed if isinstance(observed, tuple) else (observed,)
    value_sites = program.value_sites

    def simulate(theta: Vector, generator: random.Random) -> tuple[float, ...]:
        """Re-run the model with latents fixed and return the replicate.

        References:
            [REF-GELMAN-ETAL-2013, Chapter 6]. Equation [EQ-PPL-PREDICTIVE-001].

        """
        fixed = program.unflatten(theta)
        handle = ModelHandle(
            "generate",
            fixed={s.name: fixed[s.name] for s in value_sites if s.kind == "latent"},
            require_fixed=True,
            generator=generator,
        )
        model(handle)
        for record in handle.records:
            if record.name == site:
                return flatten({site: _as_float(record.value)}, (record,))
        raise ModelSpecificationError(f"site {site!r} was not simulated")

    return posterior_predictive_pvalues(
        posterior.constrained,
        simulate,
        data,
        statistics=statistics,
        discrepancies=discrepancies,
        max_draws=max_draws,
        keep_replicates=keep_replicates,
        seed=seed,
    )
