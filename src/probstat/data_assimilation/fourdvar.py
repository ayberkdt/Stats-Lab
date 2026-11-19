"""Four-dimensional variational (4D-Var) assimilation over a time window.

4D-Var fits a whole window of observations ``y_1..y_K`` with one model
trajectory. The time convention is the one of the filters (ADR-0018): the
background ``x_b`` with covariance ``B`` describes the state ``x_0`` before
the first observation, ``x_k = M(x_(k-1))`` is one application of the
per-step transition, and ``y_k`` observes ``x_k``; a ``None`` entry is a
missing observation and contributes nothing.

* **Strong constraint** (the model is perfect). The control variable is
  ``x_0`` and [EQ-DA-4DVAR-001]
  ``J(x_0) = 1/2 ||x_0 - x_b||^2_(B^-1)
  + 1/2 sum_k ||y_k - h(M_(0->k)(x_0))||^2_(R^-1)``.
  Its gradient [EQ-DA-4DVAR-002] needs the adjoint of the tangent-linear
  model. It is obtained here by reverse-mode automatic differentiation of
  the recorded trajectory (:mod:`probstat.numerics.reverse_autodiff`), which
  *is* the discrete adjoint: no adjoint code is written by hand.
* **Incremental** (Courtier, Thepaut & Hollingsworth 1994). Each outer loop
  re-runs the nonlinear model from the current ``x_0``, linearizes
  ``h o M_(0->k)`` around that trajectory, and solves the quadratic inner
  problem for the increment [EQ-DA-4DVAR-003] exactly, as a linear least
  squares problem by Householder QR. With the exact tangent-linear model
  this is the Gauss-Newton method (Lawless, Gratton & Nichols 2005).
* **Weak constraint** (Tremolet 2006). The control variable is the whole
  trajectory ``x_0..x_K`` and the cost adds the model-error term
  ``1/2 sum_k ||x_k - M(x_(k-1))||^2_(Q^-1)`` [EQ-DA-4DVAR-004], so ``Q``
  must be positive definite.

Weighted norms are evaluated through inverse Cholesky factors
(``||v||^2_(C^-1) = ||L^-1 v||^2`` with ``C = L L'``); no covariance is
inverted explicitly and no jitter is added. For a linear model with
Gaussian errors the strong-constraint analysis of ``x_0`` and its
Gauss-Newton covariance equal the Rauch-Tung-Striebel smoother at time 0
of a filter without process noise, and the weak-constraint analysis equals
the smoother over the whole window (Carrassi et al. 2018, Section 3.2.1).
Design decisions: docs/adr/ADR-0116-four-dimensional-variational.md.

Guide: docs/foundations/filtering-and-data-assimilation.md

References:
    [REF-LE-DIMET-TALAGRAND-1986], [REF-COURTIER-THEPAUT-HOLLINGSWORTH-1994],
    [REF-TREMOLET-2006], [REF-LAWLESS-GRATTON-NICHOLS-2005],
    [REF-CARRASSI-ETAL-2018], [REF-GRIEWANK-WALTHER-2008, Chapter 3].

"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from probstat.linalg import (
    Matrix,
    Vector,
    as_matrix,
    as_vector,
    cholesky,
    householder_qr,
    matmul,
    solve_lower,
    symmetrize,
    transpose,
)
from probstat.numerics import (
    AutodiffUnsupportedError,
    admath,
    reverse_jacobian,
    reverse_value_and_gradient,
    vector_jacobian_product,
)
from probstat.optimization import central_gradient, central_jacobian, minimize_bfgs

VARIATIONAL_GRADIENTS = ("reverse", "central")
"""Derivative modes: reverse-mode autodiff (default) or central differences."""

VARIATIONAL_FORMULATIONS = ("strong", "incremental", "weak")
"""Values of :attr:`FourDimensionalVariationalResult.formulation`."""

StateMap = Callable[[Vector], Sequence[float]]
JacobianMap = Callable[[Vector], Iterable[Iterable[float]]]
Observation = Sequence[float] | None


@dataclass(frozen=True, slots=True)
class FourDimensionalVariationalResult:
    """Outcome of a strong-constraint, incremental, or weak-constraint 4D-Var.

    Args:
        analysis: Analysed initial state ``x_0``.
        trajectory: Analysed states ``x_0..x_K`` (the model trajectory from
            ``x_0`` for the strong and incremental forms, the control
            variable itself for the weak form).
        cost: ``J`` at the analysis.
        background_cost: Background term ``1/2 ||x_0 - x_b||^2_(B^-1)``.
        model_cost: Model-error term (``0.0`` for a perfect model).
        observation_cost: Observation term summed over the window.
        analysis_covariance: Gauss-Newton covariance of ``x_0``, the inverse
            of the Gauss-Newton Hessian at the analysis.
        trajectory_covariances: Gauss-Newton marginal covariances of
            ``x_0..x_K``; for a perfect model ``T_k P_0 T_k'`` with the
            tangent-linear propagator ``T_k``.
        converged: Whether the stopping test was met.
        iterations: BFGS iterations, or outer loops of the incremental form.
        evaluations: Cost and gradient evaluations of the optimizer, or
            nonlinear trajectory runs of the incremental form.
        gradient_norm: Euclidean norm of ``grad J`` at the analysis.
        formulation: ``"strong"``, ``"incremental"``, or ``"weak"``.
        outer_costs: Incremental form only: the nonlinear cost at the start
            and after every outer loop; empty otherwise.
        message: Termination reason.

    References:
        [EQ-DA-4DVAR-001], [EQ-DA-4DVAR-004],
        [REF-LE-DIMET-TALAGRAND-1986], [REF-TREMOLET-2006].

    """

    analysis: Vector
    trajectory: tuple[Vector, ...]
    cost: float
    background_cost: float
    model_cost: float
    observation_cost: float
    analysis_covariance: Matrix
    trajectory_covariances: tuple[Matrix, ...]
    converged: bool
    iterations: int
    evaluations: int
    gradient_norm: float
    formulation: str
    outer_costs: tuple[float, ...]
    message: str


def _inverse_cholesky(matrix: Matrix, name: str) -> Matrix:
    """Return ``L^-1`` (lower triangular) for ``C = L L'``.

    Raises:
        ValueError: If ``C`` is not positive definite.

    References:
        [EQ-LINALG-001], [EQ-LINALG-002], [REF-GOLUB-VANLOAN-2013,
        Sections 3.1 and 4.2].

    """
    try:
        lower = cholesky(matrix)
    except ValueError as error:
        raise ValueError(f"{name} must be positive definite") from error
    size = len(lower)
    columns = [
        solve_lower(lower, tuple(1.0 if i == j else 0.0 for i in range(size)))
        for j in range(size)
    ]
    return tuple(
        tuple(columns[j][i] if j <= i else 0.0 for j in range(size))
        for i in range(size)
    )


def _whiten(inverse_factor: Matrix, vector: Sequence[Any]) -> tuple[Any, ...]:
    """Return ``L^-1 v`` for a lower-triangular ``L^-1``; ``v`` may hold variables.

    References:
        [EQ-LINALG-002], [REF-GOLUB-VANLOAN-2013, Section 3.1].

    """
    return tuple(
        admath.fsum(tuple(row[j] * vector[j] for j in range(i + 1)))
        for i, row in enumerate(inverse_factor)
    )


def _half_square(vector: Sequence[Any]) -> Any:
    """Return ``1/2 ||v||^2``.

    References:
        [EQ-DA-4DVAR-001].

    """
    return 0.5 * admath.fsum(tuple(value * value for value in vector))


def _precision_apply(inverse_factor: Matrix, vector: Vector) -> Vector:
    """Return ``C^-1 v = L^-T (L^-1 v)`` for floats.

    References:
        [EQ-LINALG-002], [EQ-LINALG-003].

    """
    whitened = _whiten(inverse_factor, vector)
    size = len(inverse_factor)
    return tuple(
        math.fsum(inverse_factor[i][j] * whitened[i] for i in range(j, size))
        for j in range(size)
    )


def _difference(left: Sequence[Any], right: Sequence[Any]) -> tuple[Any, ...]:
    """Return ``left - right`` entrywise.

    References:
        [EQ-DA-4DVAR-001].

    """
    return tuple(a - b for a, b in zip(left, right, strict=True))


def _method(value: str, name: str) -> str:
    """Validate a derivative mode.

    Raises:
        ValueError: If the mode is not in :data:`VARIATIONAL_GRADIENTS`.

    References:
        [EQ-AD-REVERSE-001], [EQ-OPT-FD-001].

    """
    if value not in VARIATIONAL_GRADIENTS:
        raise ValueError(
            f"{name} must be one of {VARIATIONAL_GRADIENTS}, got {value!r}"
        )
    return value


def _not_differentiable(error: AutodiffUnsupportedError) -> AutodiffUnsupportedError:
    """Return the error raised when reverse mode cannot pass a model function.

    References:
        [REF-GRIEWANK-WALTHER-2008, Chapter 3]. No equation is evaluated.

    """
    message = (
        "reverse-mode differentiation failed inside the transition or the "
        "observation function; write them with arithmetic operators and "
        "probstat.numerics.admath, or pass gradient='central' explicitly "
        f"({error})"
    )
    return AutodiffUnsupportedError(message)


class FourDimensionalVariationalCost:
    """4D-Var cost function of a control vector, with value and gradient.

    Strong constraint (no ``process_covariance``): the control is ``x_0``
    (length ``n``). Weak constraint: the control is the stacked trajectory
    ``(x_0, ..., x_K)`` (length ``n (K + 1)``). Build it with
    :func:`four_dimensional_variational_cost`; the solvers use the same
    object internally.

    Attributes:
        formulation: ``"strong"`` or ``"weak"``.
        state_dimension: ``n``.
        observation_dimension: ``m``.
        window: Number of transitions ``K`` (the length of ``observations``).
        control_dimension: Length of the control vector.

    References:
        [EQ-DA-4DVAR-001], [EQ-DA-4DVAR-002], [EQ-DA-4DVAR-004],
        [REF-LE-DIMET-TALAGRAND-1986], [REF-TREMOLET-2006].

    """

    __slots__ = (
        "_background",
        "_background_factor",
        "_model_factor",
        "_observation_factor",
        "_observation_jacobian",
        "_observation_function",
        "_observations",
        "_transition",
        "control_dimension",
        "formulation",
        "observation_dimension",
        "state_dimension",
        "window",
    )

    def __init__(
        self,
        background: Vector,
        background_factor: Matrix,
        observations: tuple[Vector | None, ...],
        transition: StateMap,
        observation_function: StateMap,
        observation_factor: Matrix,
        model_factor: Matrix | None,
        observation_jacobian: JacobianMap | None,
    ) -> None:
        """Store validated inputs (use :func:`four_dimensional_variational_cost`).

        Args:
            background: ``x_b``.
            background_factor: Inverse Cholesky factor of ``B``.
            observations: ``y_1..y_K`` with ``None`` for missing entries.
            transition: One-step model ``M``.
            observation_function: Observation operator ``h``.
            observation_factor: Inverse Cholesky factor of ``R``.
            model_factor: Inverse Cholesky factor of ``Q`` or ``None``.
            observation_jacobian: Optional analytic ``dh/dx``.

        References:
            [EQ-DA-4DVAR-001], [EQ-DA-4DVAR-004].

        """
        self._background = background
        self._background_factor = background_factor
        self._observations = observations
        self._transition = transition
        self._observation_function = observation_function
        self._observation_factor = observation_factor
        self._model_factor = model_factor
        self._observation_jacobian = observation_jacobian
        self.state_dimension = len(background)
        self.observation_dimension = len(observation_factor)
        self.window = len(observations)
        self.formulation = "strong" if model_factor is None else "weak"
        self.control_dimension = (
            self.state_dimension
            if model_factor is None
            else self.state_dimension * (self.window + 1)
        )

    # ------------------------------------------------------------------ maps
    def _step(self, state: Sequence[Any]) -> tuple[Any, ...]:
        """Apply ``M`` once with a length check.

        References:
            [EQ-DA-4DVAR-001].

        """
        image = tuple(self._transition(tuple(state)))
        if len(image) != self.state_dimension:
            raise ValueError("transition must return a state of the same length")
        return image

    def _observe(self, state: Sequence[Any]) -> tuple[Any, ...]:
        """Apply ``h`` with a length check.

        References:
            [EQ-DA-4DVAR-001].

        """
        image = tuple(self._observation_function(tuple(state)))
        if len(image) != self.observation_dimension:
            raise ValueError("h(x) length must equal the observation length")
        return image

    def _h_jacobian(self, state: Vector) -> Matrix:
        """Return the user's ``dh/dx`` with a shape check.

        References:
            [EQ-DA-4DVAR-002].

        """
        assert self._observation_jacobian is not None
        matrix = as_matrix(
            self._observation_jacobian(state), name="observation_jacobian"
        )
        if len(matrix) != self.observation_dimension or len(matrix[0]) != len(state):
            raise ValueError("observation Jacobian has the wrong shape")
        return matrix

    def _propagate(self, initial: Sequence[Any]) -> list[tuple[Any, ...]]:
        """Return ``x_0..x_K`` with ``x_k = M(x_(k-1))``.

        References:
            [EQ-DA-4DVAR-001].

        """
        states = [tuple(initial)]
        for _ in range(self.window):
            states.append(self._step(states[-1]))
        return states

    def _split(self, control: Sequence[Any]) -> list[tuple[Any, ...]]:
        """Return the states of a control vector (one state for the strong form).

        References:
            [EQ-DA-4DVAR-004].

        """
        if self._model_factor is None:
            return self._propagate(control)
        n = self.state_dimension
        return [tuple(control[k * n : (k + 1) * n]) for k in range(self.window + 1)]

    def _check(self, control: Sequence[float]) -> Vector:
        """Validate a float control vector.

        Raises:
            ValueError: If its length is wrong or an entry is not finite.

        References:
            [EQ-DA-4DVAR-001]. No equation is evaluated.

        """
        vector = as_vector(control, name="control")
        if len(vector) != self.control_dimension:
            raise ValueError(
                f"control must have length {self.control_dimension}, got {len(vector)}"
            )
        return vector

    def _terms(
        self, control: Sequence[Any], *, observations: bool = True
    ) -> tuple[Any, Any, Any]:
        """Return the background, model, and observation terms of ``J``.

        ``observations=False`` leaves the observation term out (it is then
        ``0.0``); the analytic-Jacobian gradient adds it separately.

        References:
            [EQ-DA-4DVAR-001], [EQ-DA-4DVAR-004].

        """
        states = self._split(control)
        background = _half_square(
            _whiten(self._background_factor, _difference(states[0], self._background))
        )
        model: Any = 0.0
        if self._model_factor is not None:
            pieces = [
                _half_square(
                    _whiten(
                        self._model_factor,
                        _difference(states[k], self._step(states[k - 1])),
                    )
                )
                for k in range(1, self.window + 1)
            ]
            model = admath.fsum(tuple(pieces))
        observation: Any = 0.0
        if observations:
            pieces = [
                _half_square(
                    _whiten(
                        self._observation_factor,
                        _difference(target, self._observe(states[k + 1])),
                    )
                )
                for k, target in enumerate(self._observations)
                if target is not None
            ]
            if pieces:
                observation = admath.fsum(tuple(pieces))
        return background, model, observation

    # ------------------------------------------------------------ public API
    def __call__(self, control: Sequence[float]) -> float:
        """Return ``J(control)``.

        Args:
            control: ``x_0`` (strong) or the stacked ``x_0..x_K`` (weak).

        Returns:
            The cost.

        Raises:
            ValueError: If the control has the wrong length or a model
                function returns a vector of the wrong length.

        Equation:
            [EQ-DA-4DVAR-001], [EQ-DA-4DVAR-004].

        References:
            [REF-LE-DIMET-TALAGRAND-1986], [REF-TREMOLET-2006].

        """
        return math.fsum(self.terms(control))

    def terms(self, control: Sequence[float]) -> tuple[float, float, float]:
        """Return the background, model-error, and observation terms of ``J``.

        Args:
            control: ``x_0`` (strong) or the stacked ``x_0..x_K`` (weak).

        Returns:
            ``(J_b, J_q, J_o)``; ``J_q = 0`` for the strong constraint.

        Raises:
            ValueError: If the control has the wrong length.

        Equation:
            [EQ-DA-4DVAR-001], [EQ-DA-4DVAR-004].

        References:
            [REF-LE-DIMET-TALAGRAND-1986], [REF-TREMOLET-2006].

        """
        background, model, observation = self._terms(self._check(control))
        return float(background), float(model), float(observation)

    def trajectory(self, control: Sequence[float]) -> tuple[Vector, ...]:
        """Return the states ``x_0..x_K`` described by a control vector.

        Args:
            control: ``x_0`` (strong: the model is run) or the stacked
                trajectory (weak: it is split).

        Returns:
            ``K + 1`` states.

        Raises:
            ValueError: If the control has the wrong length.

        References:
            [EQ-DA-4DVAR-001], [EQ-DA-4DVAR-004].

        """
        return tuple(
            as_vector(state, name="state")
            for state in self._split(self._check(control))
        )

    def value_and_gradient(
        self, control: Sequence[float], *, gradient: str = "reverse"
    ) -> tuple[float, Vector]:
        """Return ``J`` and ``grad J`` at a control vector.

        ``gradient="reverse"`` records the whole window on a reverse-mode
        tape and sweeps it once, which is the discrete adjoint model
        [EQ-DA-4DVAR-002]. With an analytic ``observation_jacobian`` only the
        model is recorded; the observation forcing
        ``-H_k' R^-1 (y_k - h(x_k))`` enters as the cotangent of the
        trajectory. ``gradient="central"`` uses central differences of ``J``.

        Args:
            control: ``x_0`` (strong) or the stacked ``x_0..x_K`` (weak).
            gradient: ``"reverse"`` or ``"central"``.

        Returns:
            ``(J, grad J)``.

        Raises:
            ValueError: If the control or the mode is invalid.
            AutodiffUnsupportedError: If reverse mode cannot pass the model
                or observation function.

        Equation:
            [EQ-DA-4DVAR-002], [EQ-AD-REVERSE-001], [EQ-OPT-FD-001].

        References:
            [REF-LE-DIMET-TALAGRAND-1986], [REF-GRIEWANK-WALTHER-2008,
            Chapter 3], [REF-NOCEDAL-WRIGHT-2006, Section 8.1].

        """
        point = self._check(control)
        mode = _method(gradient, "gradient")
        if mode == "central":
            values, _ = central_gradient(self.__call__, point)
            return self(point), values
        try:
            if self._observation_jacobian is None:
                return reverse_value_and_gradient(
                    lambda z: admath.fsum(self._terms(z)), point
                )
            return self._jacobian_gradient(point)
        except AutodiffUnsupportedError as error:
            raise _not_differentiable(error) from error

    def gradient(
        self, control: Sequence[float], *, gradient: str = "reverse"
    ) -> Vector:
        """Return ``grad J`` at a control vector.

        Args:
            control: ``x_0`` (strong) or the stacked ``x_0..x_K`` (weak).
            gradient: ``"reverse"`` or ``"central"``.

        Returns:
            The gradient.

        Raises:
            ValueError: If the control or the mode is invalid.
            AutodiffUnsupportedError: If reverse mode cannot pass the model.

        Equation:
            [EQ-DA-4DVAR-002].

        References:
            [REF-LE-DIMET-TALAGRAND-1986].

        """
        return self.value_and_gradient(control, gradient=gradient)[1]

    def _jacobian_gradient(self, point: Vector) -> tuple[float, Vector]:
        """Return ``J`` and ``grad J`` with an analytic observation Jacobian.

        References:
            [EQ-DA-4DVAR-002], [EQ-AD-REVERSE-001],
            [REF-GRIEWANK-WALTHER-2008, Chapter 3].

        """
        n = self.state_dimension
        states = [as_vector(s, name="state") for s in self._split(point)]
        forcing: list[Vector] = [(0.0,) * n for _ in range(self.window + 1)]
        observation_terms: list[float] = []
        for k, target in enumerate(self._observations):
            if target is None:
                continue
            state = states[k + 1]
            residual = _difference(target, self._observe(state))
            observation_terms.append(
                float(_half_square(_whiten(self._observation_factor, residual)))
            )
            weighted = _precision_apply(self._observation_factor, residual)
            jacobian = self._h_jacobian(state)
            forcing[k + 1] = tuple(
                -math.fsum(jacobian[i][j] * weighted[i] for i in range(len(weighted)))
                for j in range(n)
            )
        observation = math.fsum(observation_terms)
        if self._model_factor is None:
            departure = _difference(point, self._background)
            gradient = list(_precision_apply(self._background_factor, departure))
            background = float(
                _half_square(_whiten(self._background_factor, departure))
            )
            if self.window:
                cotangent = tuple(
                    v for k in range(1, self.window + 1) for v in forcing[k]
                )
                adjoint = vector_jacobian_product(
                    lambda x0: [v for s in self._propagate(x0)[1:] for v in s],
                    point,
                    cotangent,
                )
                gradient = [a + b for a, b in zip(gradient, adjoint, strict=True)]
            return math.fsum((background, observation)), tuple(gradient)
        value, partial = reverse_value_and_gradient(
            lambda z: admath.fsum(self._terms(z, observations=False)), point
        )
        flat = tuple(v for f in forcing for v in f)
        return math.fsum((value, observation)), tuple(
            a + b for a, b in zip(partial, flat, strict=True)
        )

    def _residual_jacobian(self, point: Vector, mode: str) -> tuple[Matrix, Vector]:
        """Return the whitened residual ``r`` and its Jacobian, ``J = r'r / 2``.

        Rows: background ``L_B^-1 (x_0 - x_b)``, then (weak form) the model
        errors ``L_Q^-1 (x_k - M(x_(k-1)))``, then the observed misfits
        ``L_R^-1 (h(x_k) - y_k)``.

        References:
            [EQ-DA-4DVAR-003], [EQ-AD-REVERSE-001], [EQ-OPT-FD-001].

        """
        analytic = self._observation_jacobian is not None

        def residuals(control: Sequence[Any]) -> list[Any]:
            """Return the differentiable residual rows.

            References:
                [EQ-DA-4DVAR-003].

            """
            states = self._split(control)
            rows = list(
                _whiten(
                    self._background_factor, _difference(states[0], self._background)
                )
            )
            if self._model_factor is not None:
                for k in range(1, self.window + 1):
                    rows.extend(
                        _whiten(
                            self._model_factor,
                            _difference(states[k], self._step(states[k - 1])),
                        )
                    )
            if analytic:
                if self._model_factor is None:
                    for state in states[1:]:
                        rows.extend(state)
                return rows
            for k, target in enumerate(self._observations):
                if target is not None:
                    rows.extend(
                        _whiten(
                            self._observation_factor,
                            _difference(self._observe(states[k + 1]), target),
                        )
                    )
            return rows

        values = tuple(float(v) for v in residuals(point))
        if mode == "central":
            matrix, _ = central_jacobian(lambda z: tuple(residuals(z)), point)
        else:
            try:
                matrix = reverse_jacobian(residuals, point)
            except AutodiffUnsupportedError as error:
                raise _not_differentiable(error) from error
        if not analytic:
            return matrix, values
        return self._attach_observation_rows(point, matrix, values)

    def _attach_observation_rows(
        self, point: Vector, matrix: Matrix, values: Vector
    ) -> tuple[Matrix, Vector]:
        """Append observation rows built from the analytic ``dh/dx``.

        For the strong form ``matrix`` holds the background rows followed by
        the trajectory Jacobians ``T_1..T_K``; they are replaced by
        ``L_R^-1 H_k T_k``.

        References:
            [EQ-DA-4DVAR-003].

        """
        n = self.state_dimension
        states = [as_vector(s, name="state") for s in self._split(point)]
        strong = self._model_factor is None
        base = n if strong else n * (self.window + 1)
        rows = [tuple(r) for r in matrix[:base]]
        residual_values = list(values[:base])
        width = len(point)
        for k, target in enumerate(self._observations):
            if target is None:
                continue
            state = states[k + 1]
            jacobian = self._h_jacobian(state)
            if strong:
                start = n + k * n
                tangent = tuple(tuple(r) for r in matrix[start : start + n])
                block = matmul(jacobian, tangent)
            else:
                block = tuple(
                    tuple(
                        jacobian[i][j - (k + 1) * n]
                        if (k + 1) * n <= j < (k + 2) * n
                        else 0.0
                        for j in range(width)
                    )
                    for i in range(len(jacobian))
                )
            whitened = matmul(self._observation_factor, block)
            rows.extend(whitened)
            residual_values.extend(
                _whiten(
                    self._observation_factor, _difference(self._observe(state), target)
                )
            )
        return tuple(rows), tuple(float(v) for v in residual_values)

    def _tangent_propagators(self, point: Vector, mode: str) -> tuple[Matrix, ...]:
        """Return ``T_k = d x_k / d x_0`` for ``k = 0..K`` (strong form).

        References:
            [EQ-DA-4DVAR-002], [EQ-AD-REVERSE-001].

        """
        n = self.state_dimension
        identity = tuple(
            tuple(1.0 if i == j else 0.0 for j in range(n)) for i in range(n)
        )
        if not self.window:
            return (identity,)

        def stacked(x0: Sequence[Any]) -> list[Any]:
            """Return ``x_1..x_K`` stacked.

            References:
                [EQ-DA-4DVAR-001].

            """
            return [v for state in self._propagate(x0)[1:] for v in state]

        if mode == "central":
            matrix, _ = central_jacobian(lambda z: tuple(stacked(z)), point)
        else:
            try:
                matrix = reverse_jacobian(stacked, point)
            except AutodiffUnsupportedError as error:
                raise _not_differentiable(error) from error
        return (identity,) + tuple(
            tuple(tuple(r) for r in matrix[k * n : (k + 1) * n])
            for k in range(self.window)
        )


def four_dimensional_variational_cost(
    background: Sequence[float],
    background_covariance: Iterable[Iterable[object]],
    observations: Sequence[Observation],
    transition: StateMap,
    observation_function: StateMap,
    observation_covariance: Iterable[Iterable[object]],
    *,
    process_covariance: Iterable[Iterable[object]] | None = None,
    observation_jacobian: JacobianMap | None = None,
) -> FourDimensionalVariationalCost:
    """Build the strong- or weak-constraint 4D-Var cost function.

    Without ``process_covariance`` the model is a strong constraint and the
    control is ``x_0`` [EQ-DA-4DVAR-001]; with it the control is the stacked
    trajectory ``x_0..x_K`` and the model-error term is added
    [EQ-DA-4DVAR-004].

    Args:
        background: Background state ``x_b`` of ``x_0``.
        background_covariance: Positive-definite ``B``.
        observations: ``y_1..y_K``; ``y_k`` observes ``x_k = M(x_(k-1))`` and
            ``None`` marks a missing observation.
        transition: One-step model ``M``; write it with arithmetic operators
            and :class:`~probstat.numerics.admath` for reverse-mode gradients.
        observation_function: Observation operator ``h``.
        observation_covariance: Positive-definite ``R`` (the same at every
            time).
        process_covariance: Positive-definite model-error covariance ``Q``
            for the weak constraint, or ``None`` for a perfect model.
        observation_jacobian: Optional analytic ``dh/dx``; then ``h`` need
            not be differentiable by reverse mode.

    Returns:
        The cost object.

    Raises:
        ValueError: If shapes disagree, the window is empty, or a covariance
            is not positive definite.
        TypeError: If ``transition`` or ``observation_function`` is not
            callable, or an observation is not a sequence.

    Equation:
        [EQ-DA-4DVAR-001], [EQ-DA-4DVAR-004].

    References:
        [REF-LE-DIMET-TALAGRAND-1986], [REF-TREMOLET-2006],
        [REF-CARRASSI-ETAL-2018, Section 3.2.1].

    """
    if not callable(transition):
        raise TypeError("transition must be callable")
    if not callable(observation_function):
        raise TypeError("observation_function must be callable")
    if observation_jacobian is not None and not callable(observation_jacobian):
        raise TypeError("observation_jacobian must be callable or None")
    x_b = as_vector(background, name="background")
    if not x_b:
        raise ValueError("background must not be empty")
    b_matrix = symmetrize(background_covariance, name="background_covariance")
    if len(b_matrix) != len(x_b):
        raise ValueError("background_covariance must match the state dimension")
    r_matrix = symmetrize(observation_covariance, name="observation_covariance")
    if isinstance(observations, str | bytes) or not isinstance(observations, Sequence):
        raise TypeError("observations must be a sequence of vectors or None")
    if not observations:
        raise ValueError("observations must contain at least one time")
    checked: list[Vector | None] = []
    for k, item in enumerate(observations, start=1):
        if item is None:
            checked.append(None)
            continue
        if isinstance(item, str | bytes) or not isinstance(item, Iterable):
            raise TypeError(f"observation {k} must be a sequence or None")
        vector = as_vector(item, name=f"observation {k}")
        if len(vector) != len(r_matrix):
            raise ValueError(
                f"observation {k} has length {len(vector)}, "
                f"observation_covariance has size {len(r_matrix)}"
            )
        checked.append(vector)
    q_factor: Matrix | None = None
    if process_covariance is not None:
        q_matrix = symmetrize(process_covariance, name="process_covariance")
        if len(q_matrix) != len(x_b):
            raise ValueError("process_covariance must match the state dimension")
        try:
            q_factor = _inverse_cholesky(q_matrix, "process_covariance")
        except ValueError as error:
            raise ValueError(
                "process_covariance must be positive definite for weak-constraint "
                "4D-Var (its inverse weights the model errors); use the "
                "strong-constraint four_dimensional_variational for a perfect model"
            ) from error
    return FourDimensionalVariationalCost(
        x_b,
        _inverse_cholesky(b_matrix, "background_covariance"),
        tuple(checked),
        transition,
        observation_function,
        _inverse_cholesky(r_matrix, "observation_covariance"),
        q_factor,
        observation_jacobian,
    )


def _positive_int(value: int, name: str) -> int:
    """Validate a positive iteration count.

    Raises:
        TypeError: If ``value`` is not an integer.
        ValueError: If it is not positive.

    References:
        [REF-NOCEDAL-WRIGHT-2006, Chapter 2]. No equation is evaluated.

    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _tolerance(value: float, name: str) -> float:
    """Validate a non-negative finite tolerance.

    Raises:
        ValueError: If the tolerance is negative or not finite.

    References:
        [REF-NOCEDAL-WRIGHT-2006, Chapter 2]. No equation is evaluated.

    """
    number = float(value)
    if not (math.isfinite(number) and number >= 0.0):
        raise ValueError(f"{name} must be finite and non-negative")
    return number


def _gauss_newton_covariances(
    cost: FourDimensionalVariationalCost, point: Vector, mode: str
) -> tuple[Matrix, tuple[Matrix, ...], Matrix, Vector]:
    """Return the Gauss-Newton covariances and the residual Jacobian at a point.

    The inverse Gauss-Newton Hessian ``(A'A)^-1`` of the whitened residual
    Jacobian ``A`` is formed from a Householder QR factorization of ``A``
    without forming ``A'A``. Strong form: the ``x_0`` covariance and
    ``T_k P_0 T_k'``; weak form: the diagonal blocks of the trajectory
    covariance.

    Returns:
        ``(P_0, (P_0..P_K), A, r)``.

    References:
        [EQ-DA-4DVAR-003], [EQ-LINALG-009],
        [REF-GOLUB-VANLOAN-2013, Section 5.3].

    """
    matrix, residual = cost._residual_jacobian(point, mode)
    inverse = householder_qr(matrix).gram_inverse()
    n = cost.state_dimension
    if cost.formulation == "weak":
        blocks = tuple(
            tuple(
                tuple(inverse[k * n + i][k * n + j] for j in range(n)) for i in range(n)
            )
            for k in range(cost.window + 1)
        )
        return blocks[0], blocks, matrix, residual
    propagators = cost._tangent_propagators(point, mode)
    blocks = tuple(
        _symmetric(matmul(matmul(t, inverse), transpose(t))) for t in propagators
    )
    return inverse, blocks, matrix, residual


def _symmetric(matrix: Matrix) -> Matrix:
    """Return ``(A + A') / 2``.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 1.1]. No equation is evaluated.

    """
    size = len(matrix)
    return tuple(
        tuple(0.5 * (matrix[i][j] + matrix[j][i]) for j in range(size))
        for i in range(size)
    )


def _norm(vector: Sequence[float]) -> float:
    """Return the Euclidean norm.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 2.2]. No equation is evaluated.

    """
    return math.sqrt(math.fsum(v * v for v in vector))


def _bfgs_result(
    cost: FourDimensionalVariationalCost,
    start: Vector,
    mode: str,
    gradient_tolerance: float,
    max_iterations: int,
) -> FourDimensionalVariationalResult:
    """Minimize a cost by BFGS and assemble the result.

    References:
        [EQ-DA-4DVAR-001], [EQ-DA-4DVAR-004], [EQ-OPT-BFGS-001],
        [REF-NOCEDAL-WRIGHT-2006, Section 6.1].

    """

    def adjoint_gradient(control: Vector) -> Vector:
        """Return the reverse-mode (adjoint) gradient.

        References:
            [EQ-DA-4DVAR-002], [EQ-AD-REVERSE-001].

        """
        return cost.value_and_gradient(control, gradient="reverse")[1]

    result = minimize_bfgs(
        cost,
        start,
        gradient=adjoint_gradient if mode == "reverse" else None,
        gradient_tolerance=gradient_tolerance,
        max_iterations=max_iterations,
    )
    background, model, observation = cost.terms(result.x)
    covariance, blocks, _, _ = _gauss_newton_covariances(cost, result.x, mode)
    trajectory = cost.trajectory(result.x)
    return FourDimensionalVariationalResult(
        trajectory[0],
        trajectory,
        math.fsum((background, model, observation)),
        background,
        model,
        observation,
        covariance,
        blocks,
        result.converged,
        result.iterations,
        result.evaluations,
        _norm(cost.gradient(result.x, gradient=mode)),
        cost.formulation,
        (),
        result.message,
    )


def four_dimensional_variational(
    background: Sequence[float],
    background_covariance: Iterable[Iterable[object]],
    observations: Sequence[Observation],
    transition: StateMap,
    observation_function: StateMap,
    observation_covariance: Iterable[Iterable[object]],
    *,
    observation_jacobian: JacobianMap | None = None,
    gradient: str = "reverse",
    gradient_tolerance: float = 1e-10,
    max_iterations: int = 500,
) -> FourDimensionalVariationalResult:
    """Minimize the strong-constraint 4D-Var cost over ``x_0`` by BFGS.

    ``J(x_0) = 1/2 ||x_0 - x_b||^2_(B^-1)
    + 1/2 sum_k ||y_k - h(M_(0->k)(x_0))||^2_(R^-1)`` [EQ-DA-4DVAR-001]
    starts from ``x_b``. The gradient [EQ-DA-4DVAR-002] is the reverse-mode
    derivative of the recorded window, i.e. the exact discrete adjoint;
    central differences are used only with ``gradient="central"``. The
    analysis covariance is the Gauss-Newton inverse Hessian
    ``(B^-1 + sum_k G_k' R^-1 G_k)^-1`` with ``G_k`` the tangent-linear map
    of ``h o M_(0->k)`` at the analysis, computed with the same derivative
    mode; it is the exact posterior covariance for linear ``M`` and ``h``.

    Args:
        background: Background state ``x_b`` of ``x_0`` (also the start).
        background_covariance: Positive-definite ``B``.
        observations: ``y_1..y_K``; ``y_k`` observes ``x_k`` after ``k``
            transitions; ``None`` marks a missing observation.
        transition: One-step model ``M`` (the same at every step).
        observation_function: Observation operator ``h``.
        observation_covariance: Positive-definite ``R``.
        observation_jacobian: Optional analytic ``dh/dx``.
        gradient: ``"reverse"`` (default) or ``"central"``.
        gradient_tolerance: Absolute BFGS tolerance on ``||grad J||``.
        max_iterations: BFGS iteration limit.

    Returns:
        Analysis, analysed trajectory, cost decomposition, Gauss-Newton
        covariances, and optimizer diagnostics.

    Raises:
        ValueError: If shapes disagree, a covariance is not positive
            definite, or an option is invalid.
        AutodiffUnsupportedError: If ``gradient="reverse"`` and the model
            or observation function is not reverse-differentiable.

    Equation:
        [EQ-DA-4DVAR-001], [EQ-DA-4DVAR-002], [EQ-OPT-BFGS-001].

    References:
        [REF-LE-DIMET-TALAGRAND-1986], [REF-CARRASSI-ETAL-2018,
        Section 3.2.1], [REF-GRIEWANK-WALTHER-2008, Chapter 3],
        [REF-NOCEDAL-WRIGHT-2006, Section 6.1].

    """
    mode = _method(gradient, "gradient")
    tolerance = _tolerance(gradient_tolerance, "gradient_tolerance")
    limit = _positive_int(max_iterations, "max_iterations")
    cost = four_dimensional_variational_cost(
        background,
        background_covariance,
        observations,
        transition,
        observation_function,
        observation_covariance,
        observation_jacobian=observation_jacobian,
    )
    return _bfgs_result(cost, cost._background, mode, tolerance, limit)


def incremental_four_dimensional_variational(
    background: Sequence[float],
    background_covariance: Iterable[Iterable[object]],
    observations: Sequence[Observation],
    transition: StateMap,
    observation_function: StateMap,
    observation_covariance: Iterable[Iterable[object]],
    *,
    observation_jacobian: JacobianMap | None = None,
    tangent_linear: str = "reverse",
    outer_loops: int = 10,
    gradient_tolerance: float = 1e-8,
    increment_tolerance: float = 1e-12,
) -> FourDimensionalVariationalResult:
    """Minimize the strong-constraint 4D-Var cost by incremental outer loops.

    Outer loop ``j`` runs the nonlinear model from ``x^j``, forms the
    innovations ``d_k = y_k - h(x^j_k)`` and the tangent-linear maps ``G_k``
    of ``h o M_(0->k)`` at ``x^j``, and solves the quadratic inner problem
    [EQ-DA-4DVAR-003]
    ``min 1/2 ||dx + x^j - x_b||^2_(B^-1) + 1/2 sum_k ||d_k - G_k dx||^2_(R^-1)``
    exactly by Householder QR of the whitened stacked system; then
    ``x^(j+1) = x^j + dx``. This is the Gauss-Newton method on the
    nonlinear cost, so a linear problem is solved in one outer loop. The
    loop stops when ``||grad J(x^j)|| <= gradient_tolerance`` (the gradient
    ``A' r`` is a by-product of each linearization), after an increment with
    ``max|dx_i| <= increment_tolerance * max(1, max|x_i|)`` has been
    re-linearized, or after ``outer_loops`` increments.

    Args:
        background: Background state ``x_b`` (also the first outer iterate).
        background_covariance: Positive-definite ``B``.
        observations: ``y_1..y_K`` (``None`` if missing).
        transition: One-step model ``M``.
        observation_function: Observation operator ``h``.
        observation_covariance: Positive-definite ``R``.
        observation_jacobian: Optional analytic ``dh/dx``.
        tangent_linear: ``"reverse"`` (reverse-mode Jacobians, default) or
            ``"central"`` (central differences).
        outer_loops: Maximum number of increments.
        gradient_tolerance: Absolute tolerance on ``||grad J||``.
        increment_tolerance: Relative tolerance on the increment.

    Returns:
        Analysis, trajectory, cost decomposition, Gauss-Newton covariances
        at the final outer iterate, and ``outer_costs`` (the nonlinear cost
        at ``x^0 = x_b`` and after every increment).

    Raises:
        ValueError: If shapes disagree, a covariance is not positive
            definite, or an option is invalid.
        AutodiffUnsupportedError: If ``tangent_linear="reverse"`` and a
            model function is not reverse-differentiable.

    Equation:
        [EQ-DA-4DVAR-003], [EQ-LINALG-009].

    References:
        [REF-COURTIER-THEPAUT-HOLLINGSWORTH-1994],
        [REF-LAWLESS-GRATTON-NICHOLS-2005], [REF-CARRASSI-ETAL-2018,
        Appendix D], [REF-GOLUB-VANLOAN-2013, Section 5.3].

    """
    mode = _method(tangent_linear, "tangent_linear")
    limit = _positive_int(outer_loops, "outer_loops")
    g_tolerance = _tolerance(gradient_tolerance, "gradient_tolerance")
    x_tolerance = _tolerance(increment_tolerance, "increment_tolerance")
    cost = four_dimensional_variational_cost(
        background,
        background_covariance,
        observations,
        transition,
        observation_function,
        observation_covariance,
        observation_jacobian=observation_jacobian,
    )
    x = cost._background
    costs: list[float] = []
    outer = 0
    small_step = False
    converged = False
    while True:
        matrix, residual = cost._residual_jacobian(x, mode)
        costs.append(0.5 * math.fsum(v * v for v in residual))
        width = len(x)
        gradient = tuple(
            math.fsum(matrix[i][j] * residual[i] for i in range(len(residual)))
            for j in range(width)
        )
        gradient_norm = _norm(gradient)
        if gradient_norm <= g_tolerance:
            converged = True
            message = "gradient norm below tolerance"
            break
        if small_step:
            converged = True
            message = "increment below tolerance"
            break
        if outer == limit:
            message = "outer-loop limit reached"
            break
        increment = householder_qr(matrix).solve(tuple(-v for v in residual))
        x = tuple(a + b for a, b in zip(x, increment, strict=True))
        outer += 1
        scale = max(1.0, max(abs(v) for v in x))
        small_step = max(abs(v) for v in increment) <= x_tolerance * scale
    background_term, _, observation_term = cost.terms(x)
    covariance, blocks, _, _ = _gauss_newton_covariances(cost, x, mode)
    return FourDimensionalVariationalResult(
        x,
        cost.trajectory(x),
        math.fsum((background_term, observation_term)),
        background_term,
        0.0,
        observation_term,
        covariance,
        blocks,
        converged,
        outer,
        len(costs),
        gradient_norm,
        "incremental",
        tuple(costs),
        message,
    )


def weak_constraint_four_dimensional_variational(
    background: Sequence[float],
    background_covariance: Iterable[Iterable[object]],
    observations: Sequence[Observation],
    transition: StateMap,
    observation_function: StateMap,
    observation_covariance: Iterable[Iterable[object]],
    process_covariance: Iterable[Iterable[object]],
    *,
    observation_jacobian: JacobianMap | None = None,
    gradient: str = "reverse",
    gradient_tolerance: float = 1e-10,
    max_iterations: int = 1000,
) -> FourDimensionalVariationalResult:
    """Minimize the weak-constraint 4D-Var cost over the whole trajectory.

    The control is ``(x_0, ..., x_K)`` and [EQ-DA-4DVAR-004]
    ``J = 1/2 ||x_0 - x_b||^2_(B^-1)
    + 1/2 sum_k ||x_k - M(x_(k-1))||^2_(Q^-1)
    + 1/2 sum_k ||y_k - h(x_k)||^2_(R^-1)``,
    minimized by BFGS from the background trajectory ``M_(0->k)(x_b)``
    with the reverse-mode gradient (or central differences on request).
    The Gauss-Newton covariance of the whole trajectory is the inverse of
    ``A'A`` for the whitened residual Jacobian ``A``; its diagonal blocks are
    returned. For linear ``M`` and ``h`` the analysis and these blocks are
    the Rauch-Tung-Striebel smoother means and covariances.

    Args:
        background: Background state ``x_b`` of ``x_0``.
        background_covariance: Positive-definite ``B``.
        observations: ``y_1..y_K`` (``None`` if missing).
        transition: One-step model ``M``.
        observation_function: Observation operator ``h``.
        observation_covariance: Positive-definite ``R``.
        process_covariance: Positive-definite model-error covariance ``Q``.
        observation_jacobian: Optional analytic ``dh/dx``.
        gradient: ``"reverse"`` (default) or ``"central"``.
        gradient_tolerance: Absolute BFGS tolerance on ``||grad J||``.
        max_iterations: BFGS iteration limit.

    Returns:
        Analysed trajectory, cost decomposition including the model-error
        term, Gauss-Newton marginal covariances, and diagnostics.

    Raises:
        ValueError: If ``Q`` (or ``B``, ``R``) is not positive definite,
            shapes disagree, or an option is invalid.
        AutodiffUnsupportedError: If ``gradient="reverse"`` and a model
            function is not reverse-differentiable.

    Equation:
        [EQ-DA-4DVAR-004], [EQ-OPT-BFGS-001].

    References:
        [REF-TREMOLET-2006], [REF-CARRASSI-ETAL-2018, Section 3.2.1],
        [REF-NOCEDAL-WRIGHT-2006, Section 6.1].

    """
    mode = _method(gradient, "gradient")
    tolerance = _tolerance(gradient_tolerance, "gradient_tolerance")
    limit = _positive_int(max_iterations, "max_iterations")
    cost = four_dimensional_variational_cost(
        background,
        background_covariance,
        observations,
        transition,
        observation_function,
        observation_covariance,
        process_covariance=process_covariance,
        observation_jacobian=observation_jacobian,
    )
    first_guess = [
        as_vector(s, name="state") for s in cost._propagate(cost._background)
    ]
    start = tuple(v for state in first_guess for v in state)
    return _bfgs_result(cost, start, mode, tolerance, limit)
