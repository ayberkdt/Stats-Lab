# ADR-0109: Fixed-step ODE integration to an event

- Status: Accepted
- Date: 2026-11-19

## Context

Verification laboratories need a simulator whose numerical error is a known
function of one parameter. The parachuted-package example of the ATHENS
curriculum integrates the equations of motion with explicit Euler until the
package reaches the ground, and asks how the step `dt` should be chosen. The
package had Runge-Kutta steps only as a private helper of the
probability-flow ODE in `processes.score_diffusion`, with a fixed horizon and
no stopping condition. Three needs were not covered:

- integration until an event function `g(t, y)` changes sign, with the state
  at the event as the quantity of interest;
- several methods of known order behind one call, so that an
  order-of-accuracy study can compare them;
- control over how the crossing is located inside the last step, because the
  location rule, not the method, can set the observed order (stopping at the
  first grid point past the event caps every method at order one).

## Decision

1. **One function, `numerics.integrate_to_event`.** It advances
   `y' = f(t, y)` with a fixed step by explicit Euler, Heun, or classical
   fourth-order Runge-Kutta ([EQ-NUM-ODE-001]) while `g > 0`, and locates the
   crossing in the last step by the first grid point, linear interpolation,
   or the root of `g` on the cubic Hermite interpolant of the step, found by
   the Illinois method ([EQ-NUM-ODE-002]). It returns an `EventSolution`
   with the event time and state, the step and evaluation counts, and
   optionally the path. `ODE_METHODS`, `METHOD_ORDERS`, and
   `EVENT_LOCATIONS` name the choices.
2. **Fixed step, no error control.** The step is the parameter a convergence
   study varies, so it is not adapted. Explicit methods diverge above their
   stability limit; the docstring states it and the parachute laboratory
   checks the limit of its drag transient.
3. **Hermite location by default.** Its `O(h^4)` interpolation error keeps
   the order of every offered method; the first-step and linear rules are
   kept because they are what simple simulators do and what a verification
   study must be able to expose.
4. **Grid times as `t_0 + n h`.** Repeated addition of `h` would add a
   rounding drift of order `n` units in the last place to the event time.
5. **Placement in `numerics`.** The integrator is a numerical building block
   with no probabilistic content; `numerics` imports nothing from the
   modelling or UQ layers. The private steps in `processes.score_diffusion`
   stay as they are: they integrate with a signed step over a fixed horizon
   and are bound to the score-model data structures.

## Consequences

- `tests/test_ode.py` checks the observed orders against closed-form event
  times (exponential decay, logistic growth, harmonic oscillator), the
  exactness on polynomial solutions, the work counts, and argument
  validation, with tolerances on error ratios rather than pinned values.
- `examples/parachute_vvuq_lab.py` uses it for code verification against the
  closed-form linear-drag landing, solution verification of the
  quadratic-drag model, and every simulator run of its propagation and
  validation steps.
- Adaptive step control (embedded pairs such as Dormand-Prince) and implicit
  methods for stiff problems are out of scope; adding them would extend
  `ODE_METHODS` without changing the event interface.

## References

[REF-HAIRER-NORSETT-WANNER-1993, Sections II.1 and II.6],
[REF-HAIRER-WANNER-1996, Section IV.2], [REF-DOWELL-JARRATT-1971],
[REF-PERRIN-ATHENS-VVUQ].
