# Finite Random Variables and Joint Laws

## 1. A random variable is a mapping

A random variable is not an outcome drawn from a distribution. It is a
deterministic measurable mapping `X: Omega -> S` applied to an uncertain source
outcome. The uncertainty belongs to the probability measure on `Omega`; the
mapping transports that measure to values in `S`.

For the finite module, `Omega` has the full power-set sigma-algebra. Therefore
every mapping into a finite hashable image is measurable. The induced PMF is the
probability of each preimage [EQ-RV-001].

```python
from probstat.probability import (
    FiniteProbabilityMeasure,
    FiniteRandomVariable,
    FiniteSampleSpace,
)

die = FiniteSampleSpace((1, 2, 3, 4, 5, 6))
fair = FiniteProbabilityMeasure.uniform(die)
parity = FiniteRandomVariable.from_function(
    fair,
    lambda outcome: "even" if outcome % 2 == 0 else "odd",
    name="Parity",
)

assert parity.pmf("even") == 0.5
assert parity.preimage("even") == die.event((2, 4, 6))
```

The structural support records every mapped value, including values reached
only through zero-mass source outcomes. `positive_support` removes those
zero-mass values. This distinction prevents the mapping image from being
silently confused with the probabilistic support.

## 2. Transformations

If `Y = g(X)`, then `Y(omega) = g(X(omega))` [EQ-RV-002]. Transformations reuse
the exact same measure object, so outcome ownership and dependence are retained.
They are not implemented by sampling and therefore introduce no Monte Carlo
error.

Multiple source values may collapse to one transformed value. The new PMF sums
all corresponding source mass automatically.

## 3. Numeric variables, CDFs, and moments

CDF evaluation [EQ-RV-003], expectation [EQ-RV-004], moments [EQ-RV-005] and
[EQ-RV-006], variance, and standard deviation [EQ-RV-007] require
finite real mapped values. Categorical variables remain valid random variables,
but calling numeric operations on them raises a domain error instead of assigning
an arbitrary encoding.

Raw and central moments answer different questions:

- `E[X^k]` measures raw powers relative to zero.
- `E[(X-E[X])^k]` measures powers around the probability-weighted center.
- variance is the second central moment, not the second raw moment.

High-order floating-point moments can overflow or lose precision. This finite
baseline validates finite inputs but does not promise arbitrary-precision
accumulation; the numerical-backend roadmap retains that future requirement.

## 4. Joint laws and marginals

Two variables form a joint law only when they share the exact same probability
measure object. This is stronger than having equal-looking weights: dependence
comes from how both mappings act on the same source outcome.

The joint PMF [EQ-RV-010] records pair mass. Marginalization [EQ-RV-011] sums out
one coordinate. A conditional PMF [EQ-RV-012] renormalizes a positive-mass joint
slice. Independence [EQ-RV-013] requires factorization for every value pair.

```python
coins = FiniteSampleSpace(("HH", "HT", "TH", "TT"))
fair_pair = FiniteProbabilityMeasure.uniform(coins)
first = FiniteRandomVariable.from_function(
    fair_pair, lambda outcome: outcome[0], name="First"
)
second = FiniteRandomVariable.from_function(
    fair_pair, lambda outcome: outcome[1], name="Second"
)

joint = first.joint_with(second)
assert joint.joint_pmf("H", "H") == 0.25
assert joint.are_independent()
```

Marginal equality alone never proves independence; independence is a property
of the entire joint factorization.

## 5. Covariance is not complete dependence

Covariance [EQ-RV-008] and correlation [EQ-RV-009] summarize linear co-movement
for numeric variables. Zero covariance does not generally imply independence.
Independence does imply zero covariance when the relevant expectations exist,
but nonlinear dependent relationships can have zero covariance.

Correlation is undefined when either variable has zero variance. The library
raises an error instead of returning NaN or inventing a convention.

## 6. Generating and characteristic functions

For a non-negative integer-valued variable, the probability-generating function
[EQ-GF-001] packages the PMF into a polynomial. In a finite model it is defined
at every finite complex argument. A negative, fractional, or categorical mapped
value violates the PGF domain and is rejected.

The moment-generating function [EQ-GF-002] is an exponential transform. Finite
support guarantees mathematical existence for every finite real `t`, but the
machine value can overflow when `tX` is large. The implementation raises
`OverflowError` rather than returning infinity and pretending the numerical
result is valid.

The characteristic function [EQ-GF-003] uses a complex unit-magnitude kernel.
It is bounded and conceptually more general than the MGF. Finite machine
multiplication can still overflow while forming an extreme phase, so that
numerical boundary is explicit too.

```python
bernoulli_space = FiniteSampleSpace((0, 1))
bernoulli_measure = FiniteProbabilityMeasure(
    bernoulli_space, {0: 0.7, 1: 0.3}
)
bernoulli = FiniteRandomVariable.from_function(
    bernoulli_measure, int, name="Bernoulli"
)

assert bernoulli.probability_generating_function(1) == 1 + 0j
assert bernoulli.moment_generating_function(0) == 1.0
assert bernoulli.characteristic_function(0) == 1 + 0j
```

References: [REF-ROSS-2019, Chapters 4-6].
