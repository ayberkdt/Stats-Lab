# Local and Global Sensitivity Analysis

Sensitivity analysis asks how model output changes when inputs change. Local and
global answers are different and must not be conflated.

## Local derivatives

`central_difference_gradient` approximates each partial derivative at one
declared point with symmetric perturbations [EQ-SENS-LOCAL-001]. It charges two
model evaluations per coordinate. The result depends on point and step size;
large derivatives do not by themselves imply large variance contributions over
an uncertain input distribution.

## Functional ANOVA and Sobol indices

For mutually independent inputs and a square-integrable scalar response,
functional ANOVA decomposes output variance into orthogonal main and interaction
components [EQ-SENS-ANOVA-001]. First-order indices measure main effects,
second-order indices isolate pair interactions, and total-effect indices include
every term involving an input [EQ-SENS-FIRST-001], [EQ-SENS-SECOND-001],
[EQ-SENS-TOTAL-001].

`estimate_sobol_indices` takes two explicitly independent base matrices and
constructs pick-freeze hybrids. It reports all model calls. Monte Carlo estimates
are intentionally not clipped to `[0,1]`: small negative or above-one estimates
are finite-sample diagnostics, while large violations signal inadequate sample
size, dependence, numerical noise, or a broken sampling design.

The current estimator assumes independent inputs. Correlated-input sensitivity,
Shapley effects, bootstrap intervals, replicated stochastic simulators, and
streaming evaluation remain future extensions.

## Ranking and decisions

Inputs are ranked by total effect because it includes interactions. The declared
threshold rule [EQ-SENS-DECISION-001] separates uncertainty-reduction priorities
from candidates for fixing during dimension reduction. This is a screening
recommendation, not a causal statement or cost-benefit optimum; physical,
epistemic, safety, and measurement-cost information must still be considered.

## Reproducible laboratory

Run `python examples/sensitivity_lab.py`. The Ishigami benchmark contains a
strong interaction: its third input has essentially no first-order effect but a
substantial total effect. The generated `ishigami-sobol-indices.png` makes this
failure mode of main-effect-only ranking visible.

## Sources

- [REF-SOBOL-1993].
- [REF-SALTELLI-2008, Chapters 1, 4, and 6].
- [REF-SALTELLI-2010].
- [REF-FORNBERG-1988].
