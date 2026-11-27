"""Time and profile the core numerical hot paths of the pure-Python kernels.

Usage::

    PYTHONPATH=src python tools/benchmark_hot_paths.py [--profile]
    PYTHONPATH=src python tools/benchmark_hot_paths.py --baseline-src DIR [--only NAMES]

Each workload runs three times; the table reports the median wall time in
seconds (``time.perf_counter``). With ``--baseline-src DIR`` the workloads are
timed alternately in subprocesses whose ``PYTHONPATH`` is ``DIR`` (for example
a ``git archive`` export of an earlier revision's ``src`` folder) and in this
checkout, three rounds each, so that load from other jobs hits both sides; the
table shows before, after and the speed-up of the medians. ``--profile``
prints the top ``cProfile``
entries (by cumulative time) of every workload instead of the table.

The workloads are: Gaussian-process fit on 60 points in 3 inputs, one GP
log marginal likelihood with gradient, ``cholesky`` / ``solve_spd`` /
``inverse_spd`` at ``n`` = 50, 100 and 150, Householder QR and SVD of a
60 x 40 matrix, LSQR on a 2000-unknown convolution and a subset-simulation run,
followed by the sampling workloads: subset simulation on 10 normal/lognormal
inputs, crude Monte Carlo with 20 000 draws, Sobol indices of the Ishigami function (two
estimators, N = 2048), random-walk Metropolis (5-D, 20 000 draws), a Sobol
sequence (d = 10, 2^14 points) and a Latin hypercube (20 000 x 10), and the
scalar validators (probability, integer, finite sample) and ``select_outputs``,
and the inverse-problem workloads: grid-gradient and periodic first-difference
applications, ``materialize`` of a convolution, 200 forward and 200 adjoint
applications of ``HeatSource1D`` and of ``SlabFluxOperator``, and 40
evaluations of ``EllipticDiffusion1D`` (value, ``jvp``, ``vjp``), and the
Krylov workloads: CGLS, LSMR and LSQR with a damping term on the
convolution, conjugate gradients, Lanczos and LSQR on dense matrices, and a
``CountingOperator`` loop. ``--only a,b`` keeps the workloads whose name
contains one of the fragments.
The whole script finishes in well under two minutes on a quiet machine.

References:
    [REF-PROBSTAT-API]. Timing harness only; no equation is evaluated.

"""

from __future__ import annotations

import argparse
import cProfile
import json
import math
import os
import pstats
import random
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable

REPEATS = 3


def _spd(size: int, seed: int) -> tuple[tuple[float, ...], ...]:
    """Return a reproducible well-conditioned SPD matrix ``B B' + n I``.

    Args:
        size: Matrix dimension.
        seed: Seed of the private generator.

    Returns:
        Matrix as a tuple of row tuples.

    """
    rng = random.Random(seed)
    b = [[rng.gauss(0.0, 1.0) for _ in range(size)] for _ in range(size)]
    return tuple(
        tuple(
            math.fsum(b[i][k] * b[j][k] for k in range(size))
            + (float(size) if i == j else 0.0)
            for j in range(size)
        )
        for i in range(size)
    )


def _dense(rows: int, cols: int, seed: int) -> tuple[tuple[float, ...], ...]:
    """Return a reproducible dense Gaussian matrix.

    Args:
        rows: Row count.
        cols: Column count.
        seed: Seed of the private generator.

    Returns:
        Matrix as a tuple of row tuples.

    """
    rng = random.Random(seed)
    return tuple(tuple(rng.gauss(0.0, 1.0) for _ in range(cols)) for _ in range(rows))


def _gp_predict_workload(
    points: list[list[float]], values: list[float]
) -> Callable[[], object]:
    """Return a workload predicting mean and variance at 500 points.

    The GP is fitted outside the timing (fixed hyperparameters, no search).

    Args:
        points: Training inputs (60 points in 3 inputs).
        values: Training outputs.

    Returns:
        Zero-argument callable evaluating ``predict_with_variance`` 500 times.

    """
    from probstat.surrogates import fit_gaussian_process

    fitted = fit_gaussian_process(
        points,
        values,
        optimize=False,
        initial_log_parameters=(0.0, math.log(0.5), math.log(0.5), math.log(0.5)),
    )
    rng = random.Random(19)
    queries = [[rng.random() for _ in range(3)] for _ in range(500)]
    return lambda: [fitted.predict_with_variance(q) for q in queries]


def _gp_fit_4d_workload() -> Callable[[], object]:
    """Return a workload fitting a GP to 120 points in 4 inputs (one start).

    Returns:
        Zero-argument callable running ``fit_gaussian_process``.

    """
    from probstat.surrogates import fit_gaussian_process

    rng = random.Random(23)
    pts = [[rng.random() for _ in range(4)] for _ in range(120)]
    vals = [
        math.sin(3.0 * p[0])
        + p[1] ** 2
        - 0.5 * p[2] * p[3]
        + 0.01 * rng.gauss(0.0, 1.0)
        for p in pts
    ]
    return lambda: fit_gaussian_process(
        pts, vals, seed=5, restarts=1, estimate_noise=True
    )


def build_workloads(only: str | None = None) -> dict[str, Callable[[], object]]:
    """Return the named workloads with all inputs prepared outside the timing.

    Args:
        only: Comma-separated name fragments; when given, only the workloads
            whose name contains one of them are returned.

    Returns:
        Mapping from workload name to a zero-argument callable.

    """
    import probstat.linalg as la
    from probstat.distributions import Normal
    from probstat.linalg import (
        Convolution1DOperator,
        cholesky,
        householder_qr,
        inverse_spd,
        lsqr,
        singular_value_decomposition,
        solve_spd,
    )
    from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
    from probstat.reliability import ModelLimitState, subset_simulation
    from probstat.surrogates import (
        fit_gaussian_process,
        gaussian_process_log_marginal_likelihood,
    )
    from probstat.uq import QuantityOfInterest

    del la
    rng = random.Random(7)
    points = [[rng.random() for _ in range(3)] for _ in range(60)]
    values = [
        math.sin(3.0 * p[0]) + p[1] ** 2 - 0.5 * p[2] + 0.01 * rng.gauss(0.0, 1.0)
        for p in points
    ]
    work: dict[str, Callable[[], object]] = {}
    work["gp_fit_60x3"] = lambda: fit_gaussian_process(
        points, values, seed=3, restarts=1, estimate_noise=True
    )
    log_theta = (0.0, math.log(0.5), math.log(0.5), math.log(0.5))
    work["gp_loglik_grad_60x3"] = lambda: gaussian_process_log_marginal_likelihood(
        points, values, log_theta
    )
    work["gp_predict_var_500"] = _gp_predict_workload(points, values)
    work["gp_fit_120x4"] = _gp_fit_4d_workload()
    for n in (50, 100, 150):
        matrix = _spd(n, n)
        rhs = tuple(math.sin(i) for i in range(n))
        work[f"cholesky_n{n}"] = lambda m=matrix: cholesky(m)
        work[f"solve_spd_n{n}"] = lambda m=matrix, r=rhs: solve_spd(m, r)
        work[f"inverse_spd_n{n}"] = lambda m=matrix: inverse_spd(m)
    tall = _dense(60, 40, 11)
    work["householder_qr_60x40"] = lambda: householder_qr(tall)
    work["svd_60x40"] = lambda: singular_value_decomposition(tall)
    size = 2000
    kernel = (0.05, 0.2, 0.5, 0.2, 0.05)
    operator = Convolution1DOperator(kernel, size)
    truth = tuple(math.sin(2.0 * math.pi * i / 400.0) for i in range(size))
    data = tuple(operator.matvec(truth))
    work["lsqr_conv_2000"] = lambda: lsqr(operator, data, max_iterations=60)
    inputs = JointInput.independent(
        RandomInput("u1", Normal(0.0, 1.0)), RandomInput("u2", Normal(0.0, 1.0))
    )
    model = ProbabilisticModel(
        inputs,
        lambda p: {"y": 3.5 - (p["u1"] + p["u2"]) / math.sqrt(2.0)},
        (QuantityOfInterest("y", "response"),),
    )
    state = ModelLimitState(model, "y", 0.0, "below")
    work["subset_simulation"] = lambda: subset_simulation(
        state, samples_per_level=500, seed=1
    )
    _add_sampling_workloads(work)
    _add_validation_workloads(work)
    _add_inverse_workloads(work)
    _add_krylov_workloads(work)
    if only:
        keys = only.split(",")
        return {k: v for k, v in work.items() if any(f in k for f in keys)}
    return work


def _add_sampling_workloads(work: dict[str, Callable[[], object]]) -> None:
    """Add the Monte Carlo, sampling and Markov chain workloads.

    Args:
        work: Mapping the workloads are added to, in place.

    """
    from probstat.distributions import LogNormal, Normal
    from probstat.inference.mcmc import random_walk_metropolis
    from probstat.modeling import JointInput, ProbabilisticModel, RandomInput
    from probstat.modeling.benchmarks import ishigami
    from probstat.reliability import (
        ModelLimitState,
        model_monte_carlo,
        subset_simulation,
    )
    from probstat.sampling import latin_hypercube, sobol
    from probstat.sensitivity import estimate_sobol_indices
    from probstat.uq import QuantityOfInterest

    def limit_state(dimension: int, lognormal: bool) -> ModelLimitState:
        """Return a linear limit state on normal or lognormal inputs."""
        names = [f"u{i}" for i in range(dimension)]
        inputs = JointInput.independent(
            *[
                RandomInput(
                    name,
                    LogNormal(0.0, 0.3) if lognormal and i % 2 else Normal(0.0, 1.0),
                )
                for i, name in enumerate(names)
            ]
        )
        model = ProbabilisticModel(
            inputs,
            lambda p: {"y": 4.0 - sum(p[n] for n in names) / math.sqrt(dimension)},
            (QuantityOfInterest("y", "response"),),
        )
        return ModelLimitState(model, "y", 0.0, "below")

    state_10 = limit_state(10, True)
    work["subset_10d_lognormal"] = lambda: subset_simulation(
        state_10, samples_per_level=500, seed=1
    )
    state_5 = limit_state(5, True)
    work["crude_mc_20000"] = lambda: model_monte_carlo(state_5, 20000, seed=3)
    scale = lambda row: tuple(-math.pi + 2.0 * math.pi * v for v in row)  # noqa: E731
    sample_a = [scale(r) for r in sobol(2048, 3, seed=1).points]
    sample_b = [scale(r) for r in sobol(2048, 3, seed=2).points]
    work["sobol_indices_ishigami_2048"] = lambda: [
        estimate_sobol_indices(
            lambda x: ishigami(*x), sample_a, sample_b, estimator=name
        )
        for name in ("saltelli2010", "jansen")
    ]
    covariance = tuple(
        tuple(0.5 if i == j else 0.0 for j in range(5)) for i in range(5)
    )
    work["rwm_5d_20000"] = lambda: random_walk_metropolis(
        lambda x: -0.5 * sum(v * v for v in x),
        [0.0] * 5,
        proposal_covariance=covariance,
        draws=20000,
        seed=2,
        check_convergence=False,
    )
    work["sobol_seq_d10_2e14"] = lambda: sobol(2**14, 10, seed=4)
    work["lhs_20000x10"] = lambda: latin_hypercube(20000, 10, seed=4)


def _add_validation_workloads(work: dict[str, Callable[[], object]]) -> None:
    """Add the scalar-validator and output-selection workloads.

    Args:
        work: Mapping the workloads are added to, in place.

    """
    from probstat.foundations.validation import (
        coerce_finite_sample,
        require_non_negative_integer,
        require_probability,
    )
    from probstat.uq import QuantityOfInterest, select_outputs

    rng = random.Random(5)
    probabilities = [rng.random() for _ in range(50000)]
    sample = [rng.gauss(0.0, 1.0) for _ in range(50000)]
    quantities = (QuantityOfInterest("y", "response"), QuantityOfInterest("z", "load"))
    outputs = [{"y": rng.random(), "z": rng.random()} for _ in range(20000)]

    def validate_scalars() -> int:
        """Run the probability and integer validators 50 000 times."""
        count = 0
        for p in probabilities:
            require_probability(p, name="p")
            require_non_negative_integer(count, name="n")
            count += 1
        return count

    work["validators_50000"] = validate_scalars
    work["coerce_sample_50000"] = lambda: coerce_finite_sample(sample)
    work["select_outputs_20000"] = lambda: [
        select_outputs(payload, quantities) for payload in outputs
    ]


def _add_inverse_workloads(work: dict[str, Callable[[], object]]) -> None:
    """Add the finite-difference operator, heat-stepping and PDE workloads.

    Args:
        work: Mapping the workloads are added to, in place.

    """
    from probstat.inverse.heat_conduction_inverse import (
        SlabFluxOperator,
        synthetic_ablator_slab,
    )
    from probstat.inverse.heat_models import HeatSource1D
    from probstat.inverse.pde_models import EllipticDiffusion1D
    from probstat.linalg import (
        Convolution1DOperator,
        FirstDifferenceOperator,
        GridGradientOperator,
        materialize,
    )

    rng = random.Random(9)
    grid = GridGradientOperator(80, 80, spacing=(0.5, 0.25))
    image = [rng.random() for _ in range(80 * 80)]
    grad = [rng.random() for _ in range(2 * 80 * 80)]
    work["grid_gradient_80x80_x40"] = lambda: [
        (grid.matvec(image), grid.rmatvec(grad)) for _ in range(40)
    ]
    diff = FirstDifferenceOperator(20000, spacing=0.5, boundary="periodic")
    line = [rng.random() for _ in range(20000)]
    work["first_difference_20000_x40"] = lambda: [
        (diff.matvec(line), diff.rmatvec(line)) for _ in range(40)
    ]
    conv = Convolution1DOperator((0.25, 0.5, 0.25), 150, boundary="reflect")
    work["materialize_conv_150"] = lambda: materialize(conv)
    heat = HeatSource1D(
        99,
        diffusivity=0.5,
        steps=40,
        scheme="crank_nicolson",
        observation="sensors",
        sensors=(0.2, 0.5, 0.8),
        times=(0.25, 0.5, 0.75, 1.0),
    )
    source = [rng.random() for _ in range(99)]
    observed = [rng.random() for _ in range(heat.shape[0])]
    work["heat_source_200_fwd_200_adj"] = lambda: [
        (heat.matvec(source), heat.rmatvec(observed)) for _ in range(200)
    ]
    slab = SlabFluxOperator(
        synthetic_ablator_slab(),
        (0.002, 0.004),
        flux_intervals=30,
        interval_length=0.5,
        cells=30,
        substeps=3,
    )
    flux = [rng.random() for _ in range(30)]
    readings = [rng.random() for _ in range(slab.shape[0])]
    work["slab_flux_200_fwd_200_adj"] = lambda: [
        (slab.matvec(flux), slab.rmatvec(readings)) for _ in range(200)
    ]
    elliptic = EllipticDiffusion1D(
        200, sensors=(0.2, 0.4, 0.6, 0.8), source=1.0, boundary=(0.0, 1.0)
    )
    logs = [[0.3 * math.sin(0.05 * i + s) for i in range(200)] for s in range(40)]
    weights = [0.5, -1.0, 0.25, 2.0]

    def elliptic_run() -> int:
        """Evaluate, differentiate and adjoint-differentiate 40 fields."""
        count = 0
        for x in logs:
            elliptic.value(x)
            elliptic.jvp(x, x)
            elliptic.vjp(x, weights)
            count += 1
        return count

    work["elliptic_200_x40"] = elliptic_run


def _add_krylov_workloads(work: dict[str, Callable[[], object]]) -> None:
    """Add the Krylov solver workloads on convolution and dense operators.

    Args:
        work: Mapping the workloads are added to, in place.

    """
    from probstat.linalg import (
        Convolution1DOperator,
        CountingOperator,
        DenseOperator,
        cgls,
        conjugate_gradient,
        lanczos_eigenpairs,
        lsmr,
        lsqr,
    )

    size = 2000
    operator = Convolution1DOperator((0.05, 0.2, 0.5, 0.2, 0.05), size)
    truth = tuple(math.sin(2.0 * math.pi * i / 400.0) for i in range(size))
    data = tuple(operator.matvec(truth))
    work["cgls_conv_2000"] = lambda: cgls(operator, data, max_iterations=60)
    work["lsmr_conv_2000"] = lambda: lsmr(operator, data, max_iterations=60)
    work["lsqr_conv_2000_damped"] = lambda: lsqr(
        operator, data, damping=0.01, max_iterations=60
    )
    spd = DenseOperator(_spd(200, 21))
    rhs = tuple(math.cos(0.1 * i) for i in range(200))
    work["cg_spd_200"] = lambda: conjugate_gradient(spd, rhs, max_iterations=60)
    work["lanczos_spd_200_x30"] = lambda: lanczos_eigenpairs(spd, 30, seed=5)
    tall = DenseOperator(_dense(150, 100, 13))
    target = tuple(math.sin(0.3 * i) for i in range(150))
    work["lsqr_dense_150x100"] = lambda: lsqr(tall, target, max_iterations=60)
    counted = CountingOperator(operator)
    work["counting_conv_2000_x200"] = lambda: [
        (counted.matvec(truth), counted.rmatvec(data)) for _ in range(200)
    ]


def time_workloads(
    repeats: int = REPEATS, only: str | None = None
) -> dict[str, list[float]]:
    """Return the wall times of every workload over ``repeats`` runs.

    Args:
        repeats: Timed runs per workload (an untimed warm-up precedes them
            when there is more than one).
        only: Name fragments as for :func:`build_workloads`.

    Returns:
        Mapping from workload name to the list of seconds.

    """
    result: dict[str, list[float]] = {}
    for name, task in build_workloads(only).items():
        if repeats > 1:
            task()  # warm-up (imports, caches); skipped in the one-shot workers
        runs = []
        for _ in range(repeats):
            start = time.perf_counter()
            task()
            runs.append(time.perf_counter() - start)
        result[name] = runs
    return result


def _worker(
    source: str | None, target: str, only: str | None = None
) -> dict[str, float]:
    """Time every workload once in a subprocess and return the seconds.

    Args:
        source: ``PYTHONPATH`` for the subprocess; ``None`` keeps the current one.
        target: Path of the JSON file the subprocess writes.
        only: Name fragments passed on to the subprocess.

    Returns:
        Mapping from workload name to seconds of that single run.

    """
    env = dict(os.environ)
    if source is not None:
        env["PYTHONPATH"] = source
    subprocess.run(
        [
            sys.executable,
            __file__,
            "--json",
            target,
            "--repeats",
            "1",
            *(["--only", only] if only else []),
        ],
        env=env,
        check=True,
    )
    with open(target, encoding="utf-8") as handle:
        data: dict[str, list[float]] = json.load(handle)
    return {name: runs[0] for name, runs in data.items()}


def profile_workloads(top: int, only: str | None = None) -> None:
    """Print the top ``cProfile`` entries of every workload.

    Args:
        top: Number of rows per workload.
        only: Name fragments as for :func:`build_workloads`.

    """
    for name, task in build_workloads(only).items():
        profiler = cProfile.Profile()
        profiler.enable()
        task()
        profiler.disable()
        print(f"\n=== {name} ===")
        stats = pstats.Stats(profiler)
        stats.sort_stats("tottime").print_stats(top)


def main() -> None:
    """Run the benchmark table, the profile, or the JSON worker mode."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--top", type=int, default=6)
    parser.add_argument("--baseline-src", default=None)
    parser.add_argument("--only", default=None, help="comma-separated name parts")
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--json", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.profile:
        profile_workloads(args.top, args.only)
        return
    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(time_workloads(args.repeats, args.only), handle)
        return
    before: dict[str, float] = {}
    if args.baseline_src:
        # Alternate baseline and current runs so that load from other jobs
        # affects both sides alike; report the median over the rounds.
        here = os.path.dirname(os.path.abspath(__file__))
        current = os.path.join(os.path.dirname(here), "src")
        base_runs: dict[str, list[float]] = {}
        new_runs: dict[str, list[float]] = {}
        with tempfile.TemporaryDirectory() as folder:
            target = os.path.join(folder, "run.json")
            for _ in range(args.repeats):
                pairs = ((args.baseline_src, base_runs), (current, new_runs))
                for source, store in pairs:
                    for name, seconds in _worker(source, target, args.only).items():
                        store.setdefault(name, []).append(seconds)
        before = {name: statistics.median(v) for name, v in base_runs.items()}
        after = {name: statistics.median(v) for name, v in new_runs.items()}
    else:
        after = {
            name: statistics.median(v)
            for name, v in time_workloads(args.repeats, args.only).items()
        }
    print(f"{'workload':26s} {'before_s':>9s} {'after_s':>9s} {'speedup':>8s}")
    for name, seconds in after.items():
        if name in before:
            ratio = before[name] / seconds
            print(f"{name:26s} {before[name]:9.4f} {seconds:9.4f} {ratio:7.2f}x")
        else:
            print(f"{name:26s} {'-':>9s} {seconds:9.4f} {'-':>8s}")


if __name__ == "__main__":
    main()
