# ADR-0116 — Four-dimensional variational assimilation

- Status: Accepted
- Date: 2026-11-19

## Context

`data_assimilation` had 3D-Var (one analysis time, ADR-0018) and the
ensemble filters and smoother of ADR-0112, and the guide listed 4D-Var as
not implemented. Two pieces it needs were already present:
`numerics.integrate_fixed_step` and the Lorenz factories are plain
arithmetic, so `reverse_gradient` passes through them, and the reverse-mode
tape of `numerics.reverse_autodiff` computes exact gradients at a cost that
does not grow with the number of inputs. 4D-Var also gives an exact check
against the filtering package: for linear-Gaussian models its analysis must
be the Rauch-Tung-Striebel smoother.

## Decision

1. **One module, `data_assimilation.fourdvar`, three solvers and one cost
   object.** `four_dimensional_variational` (strong constraint, control
   `x_0`, [EQ-DA-4DVAR-001]), `incremental_four_dimensional_variational`
   ([EQ-DA-4DVAR-003]), and `weak_constraint_four_dimensional_variational`
   (control `x_0..x_K`, [EQ-DA-4DVAR-004]) share
   `FourDimensionalVariationalCost`, which exposes the value, the three cost
   terms, the trajectory, and the gradient of any control vector. All three
   return one `FourDimensionalVariationalResult` with a `formulation` field.
   The cost is public because the Taylor test and the guide need `J` and
   `grad J` at arbitrary points.
2. **Model functions, not model objects.** The solvers take the one-step
   `transition` and `observation_function` as callables, as 3D-Var takes
   `h`. A `NonlinearStateSpace` is used through its raw
   `transition_function` and `observation_function`; its `transition`
   method validates with `float` conversions that a reverse-mode tape
   cannot pass. This keeps `data_assimilation` free of any `filtering`
   import (both are layer 6).
3. **The adjoint is reverse-mode automatic differentiation.** The default
   `gradient="reverse"` records the whole window on one tape and sweeps it
   once ([EQ-DA-4DVAR-002], [EQ-AD-REVERSE-001]); no adjoint is written by
   hand. With an analytic `observation_jacobian` only the model is
   recorded, and the observation forcing `-H_k' R^-1 d_k` enters as the
   cotangent of the trajectory (`vector_jacobian_product`). A model that
   reverse mode cannot pass raises `AutodiffUnsupportedError` naming
   `gradient="central"`. The library does not switch to central differences
   on its own, because they are slower for large controls and less accurate.
4. **Time convention of ADR-0018.** `x_b` describes `x_0`; `y_k` observes
   the state after `k` transitions; `None` drops a term. A window has at
   least one entry (which may be `None`); one `R` serves every time and one
   `Q` every step.
5. **Cholesky factors only, no jitter.** `B`, `R`, and `Q` are factored once.
   Weighted norms use the inverse lower factor, `||L^-1 v||^2`, computed by
   triangular solves and evaluated with `admath.fsum`, so the same code runs
   on floats and on tape variables. A covariance that is not positive
   definite raises. For the weak constraint the error says that `Q^-1` is
   required and points to the strong form for a perfect model.
6. **Gauss-Newton covariance from QR.** The analysis covariance is
   `(A'A)^-1` for the whitened residual Jacobian `A`, from
   `householder_qr(A).gram_inverse()`, without forming `A'A`. The strong and
   incremental forms report `T_k P_0 T_k'` for every time (`T_k` the
   tangent-linear propagator); the weak form reports the diagonal blocks of
   the full trajectory covariance. Jacobians use the same derivative mode as
   the gradient.
7. **Incremental 4D-Var solves the inner problem exactly.** The inner
   problem is linear least squares, so each outer loop does one Householder
   QR solve instead of an inner iterative minimization. The tangent-linear
   map is exact (no simplified inner-loop model), so the iteration is
   Gauss-Newton. It stops on `||A' r|| <= gradient_tolerance` (available from
   every linearization), on a re-linearized small increment, or after
   `outer_loops` increments. There is no line search, matching the
   operational scheme, and the guide shows a case where the cost rises in
   one outer loop.
8. **BFGS for the strong and weak forms.** `optimization.minimize_bfgs`,
   starting at `x_b` or at the background trajectory, with the absolute
   gradient tolerance `1e-10` that 3D-Var uses. The weak form's default
   iteration limit is 1000, because its control is `K + 1` times larger.

## Consequences

The linear-Gaussian checks are exact. On the constant-velocity model (10
steps, `seed=7`, one missing observation in the tests), strong 4D-Var
equals the `Q = 0` RTS smoother at every time to `1e-14`. Weak 4D-Var equals
the `q = 0.05` smoother to `2e-11` in the means (BFGS tolerance) and `1e-15`
in the covariances. Incremental 4D-Var needs one outer loop. On Lorenz-63
with a 0.5-time-unit window the analysis of `x_0` removes 58-96% of the
background error over seeds 0-9. With 2 and 4 time units, BFGS stops in
local minima for one and three of five seeds; the guide documents this
chaos caveat. A reverse-mode gradient costs about nine cost evaluations in
pure Python, whatever the control size. Central differences are cheaper
for three unknowns but nine times dearer for the 33-dimensional
weak-constraint control. Sizes stay small: the control and the Gauss-Newton
Jacobian are dense.

## Rejected alternatives

- Hand-written tangent-linear and adjoint models: they must be re-derived
  for every model and kept consistent with it. The tape gives the exact
  discrete adjoint of whatever the transition computes.
- An automatic fallback to finite differences, as `GradientEvaluator`
  offers: a silently slower, less accurate gradient hides a model that is
  not written for differentiation.
- Accepting `NonlinearStateSpace` objects directly: it would duplicate the
  structural protocol for a single attribute pair, and the validating
  methods cannot be recorded.
- An inner BFGS or conjugate-gradient loop in incremental 4D-Var: the
  quadratic problem has a direct QR solution at these sizes, which makes
  the one-outer-loop property of linear problems exact rather than
  tolerance-limited.
- Observations at `x_0` as well: this would break the shared time
  convention of the filters and the smoother.
