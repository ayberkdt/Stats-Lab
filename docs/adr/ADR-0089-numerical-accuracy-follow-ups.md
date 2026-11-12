# ADR-0089: Numerical-accuracy follow-ups for normal tails, incomplete gamma and beta, the Dirichlet normalizer, and the Euler constant

- Status: Accepted
- Date: 2026-11-12

## Context

Three decision records measured accuracy gaps against the multiple-precision
reference backend and recorded them without a fix, and one recorded a
duplicated constant:

1. ADR-0077: `Normal.cdf` and `Normal.sf` lose `(1 + x^2)` ulp in the tails
   (1228 ulp at `x = -37`). The relative condition number of `Phi` is about
   `x^2`, and the argument `(x - mu)/(sigma sqrt 2)` was rounded before
   `math.erfc` saw it.
2. ADR-0077: `regularized_gamma_p/q` reach 460 to 1330 ulp for shapes up to
   200 and `regularized_beta` 770 ulp for shapes up to 400. The prefactors
   `x^a e^-x / Gamma(a)` and `x^a (1-x)^b / B(a, b)` are the exponential of
   a float exponent whose absolute error is `2^-53` times its largest
   cancelling term (`a log x`, `x`, `lgamma(a)`), and `exp` turns that
   absolute error into a relative one.
3. ADR-0067: the Dirichlet log-density and entropy subtract `lgamma` values
   of size `alpha log alpha`; at concentrations near `1e9` the absolute error
   is about `1e-6`.
4. ADR-0073, follow-up 3: the Euler-Mascheroni constant is defined in
   `distributions/gumbel.py`, `distributions/rayleigh.py`,
   `inference/fitting.py`, and `information/differential.py`.

Measuring the incomplete-gamma error after the prefactor fix showed two
further sources that grow with the shape: the power series and the continued
fraction near `x = a` need `O(sqrt(a))` terms whose roundings accumulate
(26 ulp at `a = 200`, 570 ulp at `a = 1e5`), and stopping the series at
`8 ulp(1)` leaves a tail of up to 30 ulp when the term ratio is about 0.7.
Measuring `Normal.log_cdf` showed a defect not recorded before:
`log(1 - Phi(z))` was taken of a rounded `1 - Phi(z)` for every `z` in
`[-5, 0)`, so `log_cdf(5)` had an error of 1.26 million ulp.

Constraints: standard library only, strict mypy, documented and cited
definitions, no change to constructors, `log_pdf`, or `pdf` methods.

## Decision

1. **Double-double kernel in `numerics._double_double` (layer 0, private).**
   Knuth's two-sum, Dekker's two-product with Veltkamp's splitting, and
   double-double addition, multiplication, division, and `exp(hi + lo) =
   exp(hi)(1 + lo)` [EQ-NUM-DD-001; REF-DEKKER-1971, REF-HIGHAM-2002]. A
   table-driven logarithm (`log(j/64)` computed once in 40-digit decimal
   arithmetic, Cody-Waite `ln 2`, a four-term `atanh` remainder; absolute
   error below `1e-22`), `log1pmx` (series for `|t| <= 1/16`, logarithm
   above; relative error below `1e-20`), and the Stirling remainder `sigma`
   (series for `a >= 15`, lifted by the recurrence of `Gamma` below;
   absolute error below `2e-18`) [EQ-NUM-DD-002]. `erfc(hi + lo) = erfc(hi)
   - (2/sqrt pi) exp(-hi^2) lo` removes the amplification of a rounded
   `erfc` argument [EQ-NUM-DD-003; REF-NIST-DLMF, Section 7.10].
2. **Normal tails.** `cdf`, `sf`, `log_cdf`, and `log_sf` standardize
   `z = (x - mu)/sigma` as a double-double (`two_sum` of `x - mu`, the
   division remainder by `two_product`) and evaluate `erfc(z/sqrt 2)/2` with
   the product by `1/sqrt 2` in double-double arithmetic [EQ-DIST-NORM-007].
   For `|z| < 0.7`, where the relative condition number is below 0.5, the
   trailing part is dropped and `math.erfc` is called directly.
   `log_standard_normal_sf` uses `log1p(-Phi(z))` for every `z < 0` (it
   used it only below `-5`) and forms `z^2` exactly in the Mills-ratio
   branch; both log functions accept an optional trailing part. The
   quantile and inverse survival functions were already within 5 ulp and
   are unchanged.
3. **Incomplete gamma prefactor.** `gamma_prefactor(a, x)` evaluates
   `exp(a log1pmx((x - a)/a) + log(a)/2 - sigma(a) - log sqrt(2 pi))` with a
   double-double exponent for every positive shape [EQ-NUM-GAMMA-003;
   REF-DIDONATO-MORRIS-1986]. The shape term `log(a)/2 - sigma(a)` is cached
   (it depends on `a` only). The float exponent is kept when it is below
   `-760` (the prefactor underflows) or when an argument lies outside
   `[1e-280, 1e280]`, where double-double products are no longer exact.
4. **Temme's expansion for `a >= 20`.** Near the mode (`|x - a| <= 0.3 a`)
   `P` and `Q` use Temme's uniform expansion with 13 coefficients
   `c_0 .. c_12`, each summed from 26 Taylor coefficients in `eta`
   [EQ-NUM-GAMMA-004; REF-TEMME-1979, REF-DIDONATO-MORRIS-1986]. The
   coefficients follow from `c_0 = 1/mu - 1/eta` and
   `c_k = c_(k-1)'(eta)/eta + (-1)^k g_k/mu` (DLMF 8.12.9-8.12.10) in exact
   rational arithmetic; the test suite repeats the derivation. The rows
   `c_0` and `c_1` agree with the 18-term tables of the former
   implementation (independently derived from the closed forms) except for
   one ulp in one entry of each. `w^2 = -a log1pmx(t)` is a double-double,
   so neither `exp(-w^2)` nor `erfc(w)` amplifies a rounding. This
   supersedes the two-coefficient expansion for `a >= 1e6` of
   EQ-NUM-GAMMA-002.
5. **Stopping rules.** The gamma series stops when a term is below
   `ulp(1)/16` of the sum (was `8 ulp(1)`). The gamma and beta continued
   fractions, once a factor is within `8 ulp(1)` of one, iterate until a
   factor is within half an ulp of one, for at most 32 further steps, which
   keeps termination guaranteed.
6. **Incomplete beta prefactor.** `beta_prefactor(a, b, x, 1 - x)` uses the
   regrouped exponent `a log1pmx(t) + b log1pmx(s) + [log(a)/2 - sigma(a)] +
   [log(b)/2 - sigma(b)] - [log(a+b)/2 - sigma(a+b)] - log sqrt(2 pi)` with
   `1 + t = x(a+b)/a`, `1 + s = (1-x)(a+b)/b`, every term a double-double;
   the smaller of `x` and `1 - x` is exact and the other is formed by
   two-sum [EQ-NUM-BETA-003]. The continued fraction itself is unchanged.
7. **Dirichlet.** `log_multivariate_beta` writes each `lgamma` in Stirling
   form; the linear terms cancel exactly, leaving
   `sum (alpha_i - 1/2) log m_i - ((K-1)/2) log alpha_0 + (K-1) log sqrt(2 pi)
   + sum sigma(alpha_i) - sigma(alpha_0)` with `m_i = alpha_i/alpha_0`.
   `Dirichlet.entropy` combines it with `psi(a) = log a - L(a)`
   (`numerics.log_minus_digamma`), so the `alpha psi(alpha)` terms cancel in
   closed form and every remaining term is of order `log alpha`
   [EQ-DIST-DIR-007]. `Dirichlet.log_pdf` is not edited; it improves through
   the normalizer.
8. **One Euler constant.** `numerics.special.EULER_GAMMA` (DLMF 5.2.3,
   exported from `probstat.numerics`) replaces the four literals;
   `distributions.gev`, which imported the Gumbel copy, imports it too. All
   four former literals round to the same double, so the change is
   numerically neutral.

## Consequences

- Normal `cdf`, `sf`, `log_cdf`, and `log_sf` are within 4 ulp over the
  whole range on the test grids; the budget enforced in the tests is 8 ulp,
  leaving room for the platform's `math.erfc` (measured at most 2.04 ulp at
  exact float arguments here).
- `P(a, x)` and `Q(a, x)` are within 10 ulp on the ADR-0077 grid and within
  about 2 ulp near the mode for every shape from 20 to `1e5`. For shapes
  below 1 near `x = a + 1` the continued fraction keeps up to 15 ulp
  (rounding accumulation over its iterations).
- `I_x(a, b)` is within 9 ulp on the ADR-0077 grid (shapes up to 400). The
  modified-Lentz continued fraction still loses about one ulp per iteration
  near the mean, and its iteration count grows with the shapes: 27 ulp at
  `(150, 90)` three standard deviations below the mean, 75 ulp at
  `(1000, 1500)`, 120 ulp at `(8000, 9000)`, 417 ulp at `(40, 5000)` just
  above the symmetry split. The errors are the same before and after this
  change (the prefactor was not the cause there). A uniform asymptotic
  expansion for large `a` and `b` (DiDonato and Morris's `BASYM`) would
  remove this; it is not implemented here.
- The double-double exponent costs time. Measured per call (Windows 11,
  Python 3.12.1, noisy machine): `Normal.cdf` 2.0 to 4.4 us for `|z| >=
  0.7` (unchanged below); `P(2.5, 1)` 16 to 44 us, `Q(200, 401)` 25 to
  54 us, `I_0.3(2.5, 40)` 37 to 106 us, `I_0.5(150, 90)` 112 to 170 us;
  `P(30, 24)` 92 to 84 us (Temme replaces about 50 series terms). The
  distribution property tests (`tests/test_distribution_properties.py`) take
  142 s instead of 98 s.
- Results of every CDF, survival function, quantile search, and likelihood
  built on these kernels change in the last bits (by up to the former
  errors). No seeded sampler uses them, so seeded draws are unchanged.
- `Dirichlet.log_pdf` at huge concentrations is limited by the
  representation of `(alpha_i - 1) log x_i` itself (one ulp of `1e9` is
  `1.2e-7`); the normalizer no longer adds to that.
- `EQ-NUM-GAMMA-002` is marked superseded by `EQ-NUM-GAMMA-004`.
- `docs/api-reference.md` and `docs/math-reference/` must be regenerated for
  `EULER_GAMMA` and the new equation records.

## Verification

`tests/test_numerical_accuracy_follow_ups.py` (12 tests) uses oracles
independent of the float code: exact rational arithmetic for two-sum and
two-product (500 random pairs), 45-digit references for the split constants,
50-digit decimal logarithms and `log Gamma` for `dd_log`, `dd_log1pmx`, and
`sigma`, the exact-rational re-derivation of the Temme table (13 x 26 entries
equal after rounding; `c_2(0) = 25/6048`), 40- and 30-digit references
(`probstat.numerics.reference`) for the normal, gamma, and beta functions,
reference `log Gamma` and the digamma asymptotic series (DLMF 5.11.2) in
50-digit arithmetic for the Dirichlet, and the four former literals for the
Euler constant. `tests/test_profiles_and_reference.py` keeps the ADR-0077
grids with tightened budgets. All numbers below: 40-digit (normal) or
30-digit (gamma, beta) references, Windows 11, Python 3.12.1.

Normal (93-point grids `z` in `[-37.5, 8]` for `cdf` and `[-8, 37.5]` for
`sf`; 84 points `z` in `[-40, 37]` plus `-1000, -200, -50` for the logs):

| Function | Parameters | Before (max ulp) | After (max ulp) | Test budget |
|---|---|---|---|---|
| `cdf` | `N(0, 1)` | 1228.5 at `x = -37.0` | 1.76 | 8 (was `2(1 + x^2)`) |
| `sf` | `N(0, 1)` | 1228.5 at `x = 37.0` | 2.97 | 8 (was `2(1 + x^2)`) |
| `cdf` / `sf` | `N(3, 0.7)` | 1392.0 / 1392.0 | 2.06 / 3.52 | 8 |
| `cdf` / `sf` | `N(-1000, 13.3)` | 943.2 / 976.6 | 1.92 / 1.92 | 8 |
| `log_cdf` | `N(0, 1)` | 1,259,569 at `z = 5` | 1.62 | 4 |
| `log_cdf` / `log_sf` | `N(3, 0.7)` | 1,259,560 | 1.72 / 1.72 | 4 |
| `log_cdf` / `log_sf` | `N(-1000, 13.3)` | 1,259,577 | 1.60 / 1.65 | 4 |

Incomplete gamma and beta (ADR-0077 grids):

| Kernel | Range | Before (max ulp) | After (max ulp) | Test budget |
|---|---|---|---|---|
| `regularized_gamma_p` | `a <= 10` | 30.5 | 1.91 | 32 (was 256) |
| `regularized_gamma_p` | `10 < a <= 200` | 460.8 at `(200, 20)` | 1.62 | 16 (was 4096) |
| `regularized_gamma_q` | `a <= 10` | 60.9 | 9.6 at `(0.5, 2)` | 32 (was 256) |
| `regularized_gamma_q` | `10 < a <= 200` | 1330.3 at `(200, 605)` | 4.27 | 16 (was 4096) |
| `regularized_beta` | shapes `<= 10` | 64.7 | 7.06 | 16 (was 256) |
| `regularized_beta` | shapes up to 400 | 769.5 at `(0.05, 300, 400)` | 8.63 | 64 (was 2048) |

Incomplete gamma near the mode, `x = f a` for
`f = 0.9, 0.97, 0.995, 1, 1.005, 1.03, 1.1`, maximum over `P` and `Q`:

| Shape `a` | Before (max ulp) | After (max ulp) |
|---|---|---|
| 20 | 63.6 | 1.3 |
| 200 | 27.3 | 1.3 |
| 1000 | 58.4 | 0.9 |
| 10000 | 185.7 | 2.0 |
| 100000 | not measured (reference about 2 s per point) | 1.6 |

Away from the mode (`f = 0.3 .. 3`) the maximum after the change is 3.7 ulp
for `a = 300, 1000, 10000`.

Dirichlet with `alpha = s (0.2, 0.3, 0.5)` (absolute errors; reference
`log Gamma` and digamma in 50-digit arithmetic; density at
`(0.21, 0.29, 0.5)`):

| `s` | `log B` before / after (ulp of `log B`) | entropy before / after | `log_pdf` before / after |
|---|---|---|---|
| `1e6` | `1.8e-10` / `5.7e-11` (`1.2e-10`) | `4.4e-10` / `7.9e-17` | `1.8e-10` / `5.7e-11` |
| `1e9` | `1.4e-6` / `5.8e-8` (`1.2e-7`) | `3.5e-6` / `7.8e-16` | `1.4e-6` / `1.3e-8` |
| `1e12` | `1.1e-4` / `1.1e-5` (`1.2e-4`) | `4.4e-3` / `2.0e-15` | `1.2e-4` / `1.5e-6` |

Euler constant: `float(0.57721566490153286060651209008240243)`,
`0.57721566490153286061`, and `0.5772156649015329` are the same double,
equal to the correctly rounded DLMF 5.2.3 value; the test asserts that
`gev`, `gumbel`, `rayleigh`, `fitting`, and `differential` hold the same
object as `probstat.numerics.EULER_GAMMA`.

The existing suites of every touched package pass unchanged
(distributions, multivariate, information, inference fitting, extreme
values); see the commit message for the command lines.
