# ADR-0068: Vine copulas (C- and D-vine pair-copula constructions)

- Status: Accepted
- Date: 2026-11-11

## Context

ADR-0064 gave `JointInput` a general `Copula` interface with the
independence, Gaussian, Student-t and exchangeable Archimedean families. Its
limitation (d) recorded that no nested or vine constructions exist, and the
roadmap (Phase 2) lists vine copulas as remaining. With one parametric copula,
every pair of inputs shares one dependence type: the Gaussian copula has no
tail dependence, and an exchangeable Archimedean copula gives every pair the
same strength and the same tail. Engineering inputs often need different
behaviour per pair, for example upper-tail dependence between two loads and
negative dependence between a load and a resistance.

## Decision

1. **Pair copulas** (`probstat.modeling.vines.PairCopula`). A pair copula is
   a bivariate base copula from `probstat.modeling.copulas` plus a rotation
   of 0, 90, 180 or 270 degrees. Only Clayton and Gumbel may be rotated; the
   Gaussian, t and Frank copulas are radially symmetric and reach negative
   dependence through their parameter. A rotation is a pair of reflections
   `u -> 1 - u` (Nelsen 2006, Section 2.4). The density, CDF, both
   h-functions (`h_given_first = dC/du_1`, `h_given_second = dC/du_2`), their
   inverses, tau and tail coefficients all follow from the base copula's
   density, CDF and second Rosenblatt coordinate (EQ-MOD-VINE-001). The base
   implementations, already verified against mpmath in ADR-0064, are reused
   rather than re-derived. `PairCopula` is itself a `Copula` of dimension
   two, and `PairCopula.from_parameters(family, parameters, rotation)` builds
   one from a name. Its sampler reflects draws of the base frailty or
   mixture sampler, which serves as an independent check of the
   conditional-inverse sampler.
2. **Vines** (`CVineCopula`, `DVineCopula`, common base `VineCopula`). The
   field `pair_copulas[t][e]` holds edge `e` of tree `t + 1`, and `order`
   maps vine positions to coordinates. Fewer than `d - 1` trees give a
   truncated vine whose remaining trees are independence.
   - The log-density, the Rosenblatt transform and its inverse are the
     likelihood and sampling algorithms of Aas et al. (2009), Algorithms 1-4
     (EQ-MOD-VINE-002 to EQ-MOD-VINE-005). The D-vine work array is held as
     two arrays, `left` and `right`, for the two arguments of each edge. The
     C-vine inverse uses `F(u_t | u_<t) = w_t` directly as the root argument,
     so it needs no forward pass.
   - The Rosenblatt transform conditions in the vine order, not the
     coordinate order. Any order gives an exact isoprobabilistic map;
     FORM/SORM results depend on it, as stated in ADR-0064.
   - `kendall_tau` and `tail_dependence` are closed-form for first-tree
     pairs, whose pair copula is the bivariate margin. Other pairs raise
     `NotImplementedError`.
   - `marginal` returns a sub-vine: for a C-vine, the first `k - 1` roots plus
     one further variable; for a D-vine, any contiguous run of the path; for
     either, the full set in any order. Other subsets raise
     `NotImplementedError`, so `JointInput.subset` refuses them.
   - The CDF exists only for bivariate vines.
   - `family` is `"c-vine"` or `"d-vine"`, which `JointInput.copula_family`
     reports.
3. **Fitting** (`fit_vine_copula(data_u, structure, families, *, rotations,
   independence_test, level, order, rank_transform)`). This is the
   sequential method of Dissmann et al. (2013), restricted to C- and D-vines.
   - Columns become rank pseudo-observations.
   - Structure: the C-vine root of each tree maximizes the sum of `|tau|` to
     the remaining variables on that tree's conditional pseudo-observations.
     The D-vine order is a greedy maximum-`|tau|` path (EQ-MOD-VINE-008). A
     user `order` replaces both heuristics.
   - Edge selection: an edge becomes independence when the asymptotic
     Kendall's tau test gives `p >= level` (EQ-MOD-VINE-006). Otherwise every
     candidate is fitted by bounded BFGS (`probstat.optimization.
     minimize_bfgs`) from its tau inversion, and the smallest AIC wins
     (EQ-MOD-VINE-007). Clayton and Gumbel enter with rotations 0/180 when
     tau > 0 and 90/270 when tau < 0. The t pair copula is fitted by profile
     likelihood: an outer search over `nu` in `[1, 50]`, and an inner search
     over `rho` with the t scores computed once per `nu`. The outer search
     dominates the cost, because the t quantile is evaluated `2n` times per
     outer step.
   - Search ranges: `|rho| <= 0.999`; Clayton `[1e-4, 28]`; Gumbel
     `[1 + 1e-6, 20]`; Frank `1e-4 <= |theta| <= 50`. The Gumbel bound keeps
     `phi(u) = (-log u)^theta` from underflowing at the clamped value
     `1 - 2^-53`.
   - Result: `VineCopulaFit` holds the vine, the structure, the order, one
     `VineEdgeFit` per edge (family, rotation, parameters, sample tau,
     independence p-value, log-likelihood, AIC, candidate AICs, boundary
     flag) and the totals: log-likelihood, AIC, BIC and parameter count.
4. New public names in `probstat.modeling`: `PairCopula`, `VineCopula`,
   `CVineCopula`, `DVineCopula`, `VineEdge`, `VineEdgeFit`, `VineCopulaFit`,
   `fit_vine_copula`, `kendall_independence_p_value`, and
   `PAIR_COPULA_FAMILIES`.

## Consequences

- Positive: a `JointInput` can carry pair-specific dependence, including
  different tails per pair and negative tail-dependent pairs. Propagation,
  reliability and every other model consumer receive it unchanged through the
  existing Rosenblatt path. Gaussian vines reproduce the Gaussian copula, so
  a vine fitted to Gaussian data loses nothing.
- Limitations:
  - (a) Only C- and D-vines are implemented; there is no general regular
    vine and no maximum-spanning-tree selection.
  - (b) Sequential estimates are not the joint maximum-likelihood estimates,
    and there are no standard errors, as for `fit_copula` (ADR-0064).
  - (c) Family selection is uncertain at moderate `n`. In twenty seeded
    replicate fits (seeds 101-120, `n = 500`) of the D-vine with Clayton(3),
    Gumbel-90(2) and Frank(2) given the middle variable, Clayton was
    selected 20 times, Gumbel-90 19 times and Frank 15 times.
  - (d) Fitting is pure Python. A 4-D C-vine with five families at
    `n = 500` takes about 6-10 s. A t pair copula takes about 3-6 s per edge,
    because of the t quantile.
  - (e) Rotation reflections `1 - u` lose relative precision for `u` below
    about `1e-16`, the same limit as ADR-0064 limitation (a).
- ADR-0064 limitation (d) ("no nested or vine constructions") is partly
  lifted; nested Archimedean copulas remain absent.

## Verification

All oracles are independent of `vines.py` and live in
`tests/test_vine_copulas.py` (35 tests).

- h-functions: they agree with central differences of the closed-form CDF to
  `1e-7` for every family and rotation that has a CDF, and with the textbook
  Clayton h-function to `1e-13`. h followed by h-inverse returns the argument
  to `1e-9` for all 15 pair copulas, at four points.
- Rotations: rotated densities equal the base density at reflected points to
  `1e-13`. Tau signs are correct and the 180 degree tail coefficients are
  swapped. The survival-Clayton upper tail matches the CDF limit. Reflected
  frailty draws of Gumbel-90 (`n = 8000`) match the rotated CDF within 5 SE
  at three points.
- A two-dimensional vine equals its pair copula exactly (density, both
  Rosenblatt maps, CDF, tau, tail dependence), with and without a reversed
  order.
- Gaussian vines: with pair parameters equal to the partial correlations of a
  4 x 4 matrix `R`, each computed from the inverse of a sub-matrix in the
  test, the D-vine and C-vine equal `GaussianCopula(R)` in log-density and
  both Rosenblatt maps to `1e-12`. The same holds for permuted orders and for
  sub-vine margins.
- Normalization: `int c(u_1, v) dv = 1` to `1e-10` (Gauss-Kronrod) for every
  pair copula. A bounded 2-D density integrates to one to `1e-12` (32^2
  Gauss-Legendre), and a 3-D Frank D-vine to `1e-12` (20^3). Integrating a
  tail-dependent 3-D D-vine over either end variable returns the remaining
  first-tree pair density to `1e-11`.
- Rosenblatt of seeded samples (`n = 1000`, permuted mixed C- and D-vines
  with t, rotated Gumbel and survival Clayton pairs): coordinate means and
  variances are within 4 SE of `U(0, 1)`, and pairwise correlations within
  `4 / sqrt(n)`. The round trip holds to `1e-9`. First-tree sample tau lies
  within 5 SE of the closed form.
- `JointInput`: a `ProbabilisticModel` on a C-vine input samples and
  evaluates 300 draws, and its first-tree tau matches. `from_unit` inverts
  `to_standard_normal` to `1e-8`. `log_pdf` equals the Sklar density to
  `1e-10`. `subset` keeps sub-vines and refuses other subsets.
- Fitting (seeded):
  - The 3-D D-vine recovers the order, the families with rotations, and the
    parameters within three replicate standard deviations. Replicate SDs:
    0.19, 0.074 and 0.30.
  - The 4-D C-vine selects the true root sequence (2, 0, 1, 3), the three
    first-tree families, and independence in tree 3.
  - A t pair (`rho = 0.7`, `nu = 4`) is preferred to the Gaussian, with
    `|rho_hat - 0.7| < 0.06` and `2 < nu_hat < 10`.
  - The total log-likelihood equals the fitted vine's log-likelihood of the
    pseudo-observations to `1e-8`, and AIC and BIC follow their formulas.
- Invalid structures, families, rotations, levels, orders, tree shapes and
  data raise `ValueError` or `TypeError`.
