"""Optimal-transport and information-geometry laboratory (printed numbers only).

Five short sections, each a table that compares an algorithm with an
independent answer:

1. exact discrete transport against brute force, with its optimality
   certificate;
2. log-domain Sinkhorn as ``eps`` shrinks, and the Sinkhorn divergence;
3. Wasserstein closed forms against quantile quadrature, and barycenters;
4. closed-form Fisher information against numerical expectations;
5. Fisher-Rao distances and the Renyi-to-KL limit.

Every random input is seeded, so the output is reproducible.
"""

from __future__ import annotations

import itertools
import math
import random
import time

from probstat.distributions import Bernoulli, Beta, Exponential, Gamma, Normal, Poisson
from probstat.information import (
    exact_transport,
    fisher_information,
    fisher_rao_categorical,
    fisher_rao_normal,
    fisher_rao_poisson,
    gaussian_barycenter,
    gaussian_wasserstein,
    hellinger_distance_gaussian,
    kl_divergence,
    normal_wasserstein,
    numerical_fisher_information,
    quantile_barycenter,
    quantile_wasserstein,
    renyi_divergence,
    sinkhorn,
    sinkhorn_divergence,
    squared_euclidean_cost_matrix,
)
from probstat.numerics import composite_simpson

SEED = 2026


def _section(number: int, title: str) -> None:
    """Print a numbered section header.

    References:
        Project example conventions. No equation is evaluated.

    """
    print()
    print(f"[{number}/5] {title}")


def exact_transport_section(rng: random.Random) -> tuple[list, list, list, float]:
    """Solve a 6 x 6 assignment exactly and compare with all 720 permutations.

    References:
        [EQ-OT-LP-001], [EQ-OT-SIMPLEX-001], [REF-PEYRE-CUTURI-2019].

    """
    _section(1, "Exact transport (6 points each, uniform weights, squared cost)")
    xs = [(rng.gauss(0, 1), rng.gauss(0, 1)) for _ in range(6)]
    ys = [(rng.gauss(1.5, 1), rng.gauss(0.5, 1)) for _ in range(6)]
    weights = [1.0 / 6.0] * 6
    cost = squared_euclidean_cost_matrix(xs, ys)
    result = exact_transport(weights, weights, cost)
    brute = (
        min(
            math.fsum(cost[i][perm[i]] for i in range(6))
            for perm in itertools.permutations(range(6))
        )
        / 6.0
    )
    certificate = result.certificate()
    print(f"  {'quantity':<34}{'value':>14}")
    print(f"  {'simplex optimal cost':<34}{result.cost:>14.10f}")
    print(f"  {'brute force over 720 permutations':<34}{brute:>14.10f}")
    print(f"  {'simplex pivots':<34}{result.iterations:>14d}")
    print(f"  {'duality gap':<34}{certificate.duality_gap:>14.2e}")
    print(
        f"  {'max P_ij |C_ij - u_i - v_j|':<34}"
        f"{certificate.complementary_slackness:>14.2e}"
    )
    print(f"  {'max dual violation':<34}{certificate.dual_infeasibility:>14.2e}")
    print(
        f"  certificate verdict: {'optimal' if certificate.optimal else 'NOT optimal'}"
    )
    return xs, ys, weights, result.cost


def sinkhorn_section(xs: list, ys: list, weights: list, exact: float) -> None:
    """Show Sinkhorn approaching the exact cost and the debiased divergence.

    References:
        [EQ-OT-SINK-001], [EQ-OT-SINK-002], [REF-CUTURI-2013],
        [REF-FEYDY-ETAL-2019].

    """
    _section(2, "Entropic transport: Sinkhorn as eps shrinks (same problem)")
    cost = squared_euclidean_cost_matrix(xs, ys)
    print("  running Sinkhorn for 4 values of eps ...")
    header = f"  {'eps':>6}{'iters':>8}{'<P,C>-exact':>13}{'OT_eps-exact':>14}"
    print(header + f"{'row error':>11}{'S_eps':>10}")
    for epsilon in (2.0, 1.0, 0.5, 0.25):
        result = sinkhorn(
            weights, weights, cost, epsilon=epsilon, max_iterations=50_000
        )
        divergence = sinkhorn_divergence(
            xs, weights, ys, weights, epsilon=epsilon, max_iterations=50_000
        )
        print(
            f"  {epsilon:>6.2f}{result.iterations:>8d}"
            f"{result.transport_cost - exact:>13.2e}"
            f"{result.regularized_cost - exact:>14.2e}"
            f"{result.marginal_error:>11.1e}{divergence:>10.5f}"
        )
    self_divergence = sinkhorn_divergence(
        xs, weights, xs, weights, epsilon=1.0, max_iterations=50_000
    )
    print(
        f"  exact cost {exact:.5f}; "
        f"S_eps(alpha, alpha) at eps 1 = {self_divergence:.1e}"
    )


def wasserstein_section() -> None:
    """Compare Gaussian closed forms with quantile quadrature; barycenters.

    References:
        [EQ-OT-GAUSS-001], [EQ-OT-W1D-001], [EQ-OT-BARY-001], [EQ-OT-BARY-002],
        [REF-ALVAREZ-ESTEBAN-ETAL-2016].

    """
    _section(3, "Wasserstein closed forms against quantile quadrature")
    pairs = (
        ("W2  N(1,2^2) vs N(-0.5,0.7^2)", Normal(1, 2), Normal(-0.5, 0.7), 2.0),
        ("W1  Gamma(2,1) vs Exp(0.8)", Gamma(2.0, 1.0), Exponential(0.8), 1.0),
    )
    gamma, exponential = pairs[1][1], pairs[1][2]
    cdf_integral = composite_simpson(
        lambda x: abs(gamma.cdf(x) - exponential.cdf(x)), 0.0, 80.0, intervals=32_000
    ).value
    references = (normal_wasserstein(1, 2, -0.5, 0.7), cdf_integral)
    bures = gaussian_wasserstein([1], [[4]], [-0.5], [[0.49]])
    print(f"  {'distance':<32}{'reference':>14}{'quadrature':>14}")
    for (label, law0, law1, p), reference in zip(pairs, references, strict=True):
        value = quantile_wasserstein(law0, law1, p=p)
        print(f"  {label:<32}{reference:>14.10f}{value:>14.10f}")
    print("  references: 1-D normal formula (W2); Simpson integral of |F0 - F1| (W1)")
    print(f"  Bures form of the first W2: {bures:.10f}")
    barycenter = quantile_barycenter([Normal(0, 1), Normal(4, 3)])
    gap = quantile_wasserstein(barycenter, Normal(2, 2))
    print(f"  1-D barycenter of N(0,1), N(4,3): W2 to N(2, 2^2) = {gap:.1e}")
    general = gaussian_barycenter(
        [[0, 0], [1, 1], [2, -1]],
        [[[2, 0.5], [0.5, 1]], [[1, -0.3], [-0.3, 2]], [[0.5, 0], [0, 0.5]]],
    )
    s = general.covariance
    print(
        f"  2-D Gaussian barycenter: S = [[{s[0][0]:.6f}, {s[0][1]:.6f}], "
        f"[{s[1][0]:.6f}, {s[1][1]:.6f}]]"
    )
    print(
        f"    after {general.iterations} fixed-point steps, "
        f"residual {general.residual:.1e}"
    )


def fisher_section() -> None:
    """Compare closed-form Fisher information with numerical expectations.

    References:
        [EQ-IG-FISHER-001], [EQ-IG-FISHER-002], [REF-MIYAMOTO-ETAL-2024].

    """
    _section(4, "Fisher information: closed form vs numerical expectation")
    families = (
        ("Normal(0.3, 1.7)", lambda t: Normal(t[0], t[1]), (0.3, 1.7)),
        ("Gamma(2.3, 1.4)", lambda t: Gamma(t[0], t[1]), (2.3, 1.4)),
        ("Beta(2, 3.5)", lambda t: Beta(t[0], t[1]), (2.0, 3.5)),
        ("Bernoulli(0.3)", lambda t: Bernoulli(t[0]), (0.3,)),
        ("Poisson(3.2)", lambda t: Poisson(t[0]), (3.2,)),
    )
    print(
        f"  {'law':<18}{'I_11 closed':>13}{'score rel err':>15}{'Hessian rel err':>17}"
    )
    for label, family, theta in families:
        exact = fisher_information(family(theta))
        errors = []
        for method in ("score", "hessian"):
            numerical = numerical_fisher_information(family, theta, method=method)
            errors.append(
                max(
                    abs(numerical[i][j] - exact[i][j]) / max(1.0, abs(exact[i][j]))
                    for i in range(len(exact))
                    for j in range(len(exact))
                )
            )
        print(f"  {label:<18}{exact[0][0]:>13.6f}{errors[0]:>15.1e}{errors[1]:>17.1e}")


def geometry_section() -> None:
    """Print Fisher-Rao distances and the Renyi-to-KL limit.

    References:
        [EQ-IG-FR-001], [EQ-IG-FR-002], [EQ-IG-FR-003], [EQ-DIV-RENYI-001],
        [EQ-DIV-GAUSS-001], [REF-VAN-ERVEN-HARREMOES-2014].

    """
    _section(5, "Fisher-Rao distances and Renyi divergence near alpha = 1")
    print(f"  {'pair':<36}{'value':>12}")
    print(
        f"  {'Fisher-Rao N(0,1) to N(1,2^2)':<36}{fisher_rao_normal(0, 1, 1, 2):>12.6f}"
    )
    pmfs = ((0.2, 0.5, 0.3), (0.4, 0.4, 0.2))
    print(
        f"  {'Fisher-Rao categorical p to q':<36}{fisher_rao_categorical(*pmfs):>12.6f}"
    )
    print(
        f"  {'Fisher-Rao Poisson(3) to Poisson(5)':<36}"
        f"{fisher_rao_poisson(3, 5):>12.6f}"
    )
    hellinger = hellinger_distance_gaussian([0], [[1]], [1], [[4]])
    print(f"  {'Hellinger N(0,1) to N(1,2^2)':<36}{hellinger:>12.6f}")
    kl = kl_divergence(*pmfs)
    print(f"  {'alpha':>8}{'D_alpha(p||q)':>16}{'minus KL':>12}")
    for alpha in (0.5, 0.99, 0.9999, 1.0001, 1.01, 2.0):
        value = renyi_divergence(*pmfs, alpha)
        print(f"  {alpha:>8}{value:>16.8f}{value - kl:>12.2e}")


def main() -> None:
    """Run the five sections and report the elapsed time.

    References:
        [REF-PEYRE-CUTURI-2019], [REF-AMARI-2016].

    """
    started = time.perf_counter()
    print("Optimal transport and information geometry lab (seed 2026)")
    rng = random.Random(SEED)
    xs, ys, weights, exact = exact_transport_section(rng)
    sinkhorn_section(xs, ys, weights, exact)
    wasserstein_section()
    fisher_section()
    geometry_section()
    elapsed = time.perf_counter() - started
    print()
    print(
        f"Done in {elapsed:.1f} s. Next: docs/foundations/optimal-transport.md "
        "explains each table."
    )


if __name__ == "__main__":
    main()
