"""Bayesian updating of a structural model from identified modal data.

Experimental (added in 0.28).

Setting: a linear structure with mass matrix ``M`` (known) and stiffness
``K(theta) = K_0 + sum_i theta_i K_i`` where ``theta_i > 0`` scales the
stiffness of substructure ``i`` (``theta_i = 1`` is the nominal, undamaged
state and damage is a reduction) [EQ-INV-MODAL-001]. Modal identification of
the real structure supplies natural frequencies and, optionally, mode-shape
components at a few measured degrees of freedom; the module infers ``theta``.
All quantities are SI (``kg``, ``N/m``, ``Hz``); the shapes are mass-normalized
(``phi' M phi = 1``). The bundled builder :meth:`ModalStructure.spring_mass_chain`
is an idealization (lumped masses joined by springs, like an interstage or
truss shear model); it is a synthetic model, not a model of any real vehicle.

What is implemented (a re-derivation in the style of Beck and Katafygiotis
[REF-BECK-KATAFYGIOTIS-1998], not a copy of their formulas):

* Forward model: the generalized symmetric eigenproblem ``K phi = lambda M
  phi`` solved by the Cholesky reduction ``M = L L'`` and the dense Jacobi
  eigensolver [EQ-INV-MODAL-001]; analytic eigenvalue sensitivities
  ``d lambda_j / d theta_i = phi_j' K_i phi_j`` [EQ-INV-MODAL-002]; the modal
  assurance criterion [EQ-INV-MODAL-003] and a greedy mode matching.
* Likelihood: the identified frequencies obey ``log f_hat_k = log f_k(theta) +
  e_k`` with ``e_k ~ N(0, delta^2)`` (a relative error, exactly Gaussian in the
  logarithm) [EQ-INV-MODAL-004]. The identified shape of mode ``k`` at the
  measured degrees of freedom, scaled to unit norm, is ``a_k psi_k(theta) +
  eta_k`` with ``eta_k ~ N(0, sigma^2 I)`` and an unknown scale (and sign)
  ``a_k`` per mode that is *maximized out* (profile likelihood, not
  marginalized) [EQ-INV-MODAL-005]. Frequency and shape errors are independent
  across modes.
* Prior: ``log theta_i ~ N(log m_i, s_i^2)`` (lognormal), optionally truncated
  to ``theta_i <= upper`` for damage-only classes [EQ-INV-MODAL-006]. The
  posterior is explored in ``u = log theta``.
* Posterior: MAP by Levenberg-Marquardt on the whitened residuals, Laplace
  approximation and evidence from the Gauss-Newton Hessian
  [EQ-INV-MODAL-007], and transitional MCMC from
  :func:`probstat.inference.transitional_mcmc` [REF-CHING-CHEN-2007].
* Damage detection: ``P(theta_i < 1 - X | data)`` from the Gaussian Laplace
  marginal or from MCMC samples [EQ-INV-MODAL-008]; model-class selection
  between ``undamaged`` and ``damage_j`` classes by the TMCMC evidence
  [EQ-INV-MODAL-009] [REF-BECK-YUEN-2004].

Common pitfalls:
    - Few measured modes or degrees of freedom give a posterior that can be
      multimodal (a structure with a mirror symmetry and only frequencies
      measured gives two exact modes ``(a, b)`` and ``(b, a)``) or a ridge. The
      Laplace approximation reports one mode only; use TMCMC and look at the
      samples before quoting damage probabilities.
    - The shape scale is maximized, so the shape likelihood is a profile
      likelihood: it ignores the uncertainty of ``a_k``, and its constant counts
      all ``p`` components. Evidences of classes fitted to the same data share
      the construction, so their ratio is consistent, but the absolute
      evidence is not that of a fully marginalized model.
    - Modes are matched by their order (data mode ``k`` is model mode
      ``mode_indices[k]``). Mode crossing under damage breaks this;
      :func:`match_modes` checks it at a reference ``theta``.
    - The eigenvalue sensitivities require simple eigenvalues; repeated or
      nearly repeated eigenvalues are rejected, not regularized.
    - Prior truncation is ignored by the Laplace approximation.
    - Model-class probabilities are only as good as the class set and the
      priors; a damage class that is wrong by location may still win when the
      data cannot tell segments apart.

References:
    [REF-BECK-KATAFYGIOTIS-1998], [REF-CHING-CHEN-2007], [REF-BECK-YUEN-2004],
    [REF-GOLUB-VANLOAN-2013].

"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)
from probstat.inference.transitional_mcmc import (
    ModelSelection,
    TMCMCResult,
    model_class_selection,
    transitional_mcmc,
)
from probstat.linalg import (
    Matrix,
    Vector,
    as_matrix,
    as_vector,
    cholesky,
    dot,
    inverse_spd,
    log_det_spd,
    solve_lower,
    solve_upper,
    symmetric_eigen,
    transpose,
)
from probstat.optimization import least_squares

_LOG_TWO_PI = math.log(2.0 * math.pi)
_SYMMETRY_TOLERANCE = 1e-10
_SIMPLE_GAP = 1e-8


def _positive(value: object, name: str) -> float:
    """Return a finite positive float.

    Raises:
        TypeError: If ``value`` is not real.
        ValueError: If it is not finite and positive.

    References:
        [EQ-INV-MODAL-006]. No new equation is introduced.

    """
    number = require_finite_real(value, name=name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _check_symmetric(matrix: Matrix, name: str) -> None:
    """Raise unless ``matrix`` is symmetric to a relative tolerance.

    Raises:
        ValueError: If ``matrix`` is not symmetric.

    References:
        [EQ-INV-MODAL-001]. No new equation is introduced.

    """
    size = len(matrix)
    scale = max(abs(v) for row in matrix for v in row) or 1.0
    for i in range(size):
        for j in range(i + 1, size):
            if abs(matrix[i][j] - matrix[j][i]) > _SYMMETRY_TOLERANCE * scale:
                raise ValueError(f"{name} must be symmetric")


def _lower_solve_columns(lower: Matrix, matrix: Matrix) -> Matrix:
    """Return ``L^-1 B`` for a lower-triangular ``L`` and a square ``B``.

    References:
        [EQ-INV-MODAL-001]. No new equation is introduced.

    """
    columns = [solve_lower(lower, column) for column in transpose(matrix)]
    return transpose(tuple(columns))


def generalized_eigen(stiffness: Matrix, mass: Matrix) -> tuple[Vector, Matrix]:
    """Solve ``K phi = lambda M phi`` for symmetric ``K`` and SPD ``M``.

    With ``M = L L'`` the problem becomes the symmetric ordinary problem
    ``A y = lambda y`` for ``A = L^-1 K L^-T`` and ``phi = L^-T y``. The
    eigenvectors are mass-orthonormal (``phi_i' M phi_j = delta_ij``). The sign
    of each is fixed so that its component of largest magnitude is positive.

    Args:
        stiffness: Symmetric ``n x n`` stiffness (N/m).
        mass: Symmetric positive-definite ``n x n`` mass (kg).

    Returns:
        Eigenvalues ``lambda = omega^2`` (rad^2/s^2, ascending) and a matrix
        whose COLUMNS are the mass-normalized shapes.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If the matrices are not square, equal-sized, finite and
            symmetric, if ``mass`` is not positive definite, or if the Jacobi
            iteration does not converge.

    Equation:
        [EQ-INV-MODAL-001].

    References:
        [REF-GOLUB-VANLOAN-2013].

    """
    k = as_matrix(stiffness, name="stiffness", square=True)
    m = as_matrix(mass, name="mass", square=True)
    if len(k) != len(m):
        raise ValueError("stiffness and mass must have the same size")
    _check_symmetric(k, "stiffness")
    _check_symmetric(m, "mass")
    lower = cholesky(m)
    half = _lower_solve_columns(lower, k)
    reduced = _lower_solve_columns(lower, transpose(half))
    sym = tuple(
        tuple(0.5 * (reduced[i][j] + reduced[j][i]) for j in range(len(k)))
        for i in range(len(k))
    )
    values, vectors = symmetric_eigen(sym)
    upper = transpose(lower)
    shapes: list[Vector] = []
    for column in transpose(vectors):
        phi = solve_upper(upper, column)
        pivot = max(phi, key=abs)
        shapes.append(tuple(-v for v in phi) if pivot < 0.0 else phi)
    return values, transpose(tuple(shapes))


def mac(first: Sequence[float], second: Sequence[float]) -> float:
    """Return the modal assurance criterion of two shape vectors.

    ``MAC = (a' b)^2 / ((a' a)(b' b))`` lies in ``[0, 1]`` and is one exactly
    when the vectors are parallel (any scale and sign).

    Args:
        first: Shape vector.
        second: Shape vector of the same length.

    Returns:
        The MAC value.

    Raises:
        TypeError: If an entry is not real.
        ValueError: If the lengths differ, a vector is empty or zero.

    Equation:
        [EQ-INV-MODAL-003].

    References:
        [REF-BECK-KATAFYGIOTIS-1998].

    """
    a = as_vector(first, name="first")
    b = as_vector(second, name="second")
    if len(a) != len(b) or not a:
        raise ValueError("shape vectors must be non-empty and of equal length")
    aa = dot(a, a)
    bb = dot(b, b)
    if aa <= 0.0 or bb <= 0.0:
        raise ValueError("shape vectors must be nonzero")
    return min(1.0, dot(a, b) ** 2 / (aa * bb))


@dataclass(frozen=True, slots=True)
class ModeMatch:
    """Assignment of data modes to model modes by the MAC.

    Args:
        model_indices: Model mode index (0-based, ascending frequency) assigned
            to each data mode.
        mac_values: MAC between each data shape and its assigned model shape.

    References:
        [EQ-INV-MODAL-003].

    """

    model_indices: tuple[int, ...]
    mac_values: tuple[float, ...]


def match_modes(
    model_shapes: Matrix, data_shapes: Sequence[Sequence[float]], dofs: Sequence[int]
) -> ModeMatch:
    """Assign each measured shape to the model mode with the largest MAC.

    The MAC is evaluated on the measured degrees of freedom only. Pairs are
    taken greedily in descending MAC with every model mode used at most once.

    Args:
        model_shapes: Model shapes as COLUMNS (all degrees of freedom).
        data_shapes: One measured vector per data mode.
        dofs: Indices of the measured degrees of freedom.

    Returns:
        The assignment and its MAC values.

    Raises:
        ValueError: If there are more data modes than model modes, a shape has
            the wrong length, or an index is invalid.

    Equation:
        [EQ-INV-MODAL-003].

    References:
        [REF-BECK-KATAFYGIOTIS-1998].

    """
    shapes = as_matrix(model_shapes, name="model_shapes")
    size = len(shapes)
    indices = tuple(dofs)
    if any(not isinstance(i, int) or not 0 <= i < size for i in indices):
        raise ValueError("dofs must be valid degree-of-freedom indices")
    measured = tuple(as_vector(row, name="data shape") for row in data_shapes)
    modes = len(shapes[0])
    if len(measured) > modes:
        raise ValueError("more data modes than model modes")
    if any(len(row) != len(indices) for row in measured):
        raise ValueError("each data shape needs one entry per measured dof")
    columns = transpose(shapes)
    pairs = sorted(
        (
            (mac(row, tuple(col[i] for i in indices)), d, j)
            for d, row in enumerate(measured)
            for j, col in enumerate(columns)
        ),
        reverse=True,
    )
    chosen: dict[int, tuple[int, float]] = {}
    used: set[int] = set()
    for value, d, j in pairs:
        if d in chosen or j in used:
            continue
        chosen[d] = (j, value)
        used.add(j)
    order = range(len(measured))
    return ModeMatch(
        tuple(chosen[d][0] for d in order), tuple(chosen[d][1] for d in order)
    )


@dataclass(frozen=True, slots=True)
class ModalSolution:
    """Modes of the structure at one parameter value.

    Args:
        eigenvalues: ``lambda_j = omega_j^2`` in ascending order (rad^2/s^2).
        shapes: Mass-normalized shapes as COLUMNS (``n x n``).

    References:
        [EQ-INV-MODAL-001].

    """

    eigenvalues: Vector
    shapes: Matrix

    @property
    def frequencies_hz(self) -> Vector:
        """Return the natural frequencies ``sqrt(lambda) / (2 pi)`` in Hz.

        Raises:
            ValueError: If an eigenvalue is not positive.

        References:
            [EQ-INV-MODAL-001]. No new equation is introduced.

        """
        if any(v <= 0.0 for v in self.eigenvalues):
            raise ValueError("a non-positive eigenvalue has no frequency")
        return tuple(math.sqrt(v) / (2.0 * math.pi) for v in self.eigenvalues)

    def shape(self, index: int) -> Vector:
        """Return the mass-normalized shape of mode ``index``.

        Args:
            index: 0-based mode index.

        Returns:
            The shape vector.

        Raises:
            ValueError: If the index is out of range.

        References:
            [EQ-INV-MODAL-001]. No new equation is introduced.

        """
        if not isinstance(index, int) or not 0 <= index < len(self.eigenvalues):
            raise ValueError("mode index out of range")
        return tuple(row[index] for row in self.shapes)


class ModalStructure:
    """Mass matrix and parameterized stiffness ``K_0 + sum theta_i K_i``.

    Args:
        mass: Symmetric positive-definite mass matrix (kg).
        stiffness_matrices: The symmetric matrices ``K_i`` (N/m), one per
            parameter, each of the size of ``mass``.
        fixed_stiffness: Optional constant ``K_0`` (default zero).

    Raises:
        TypeError: If an entry is not real.
        ValueError: For inconsistent sizes, non-symmetric matrices, a mass that
            is not positive definite, no parameters, or a stiffness
            ``K(1)`` that is not positive definite (a rigid-body mode).

    Equation:
        [EQ-INV-MODAL-001].

    References:
        [REF-BECK-KATAFYGIOTIS-1998], [REF-GOLUB-VANLOAN-2013].

    """

    __slots__ = ("_fixed", "_mass", "_parts")

    def __init__(
        self,
        mass: Matrix,
        stiffness_matrices: Sequence[Matrix],
        *,
        fixed_stiffness: Matrix | None = None,
    ) -> None:
        """Validate and store the structure.

        References:
            [EQ-INV-MODAL-001].

        """
        m = as_matrix(mass, name="mass", square=True)
        size = len(m)
        _check_symmetric(m, "mass")
        cholesky(m)
        parts = tuple(
            as_matrix(k, name=f"stiffness_matrices[{i}]", square=True)
            for i, k in enumerate(stiffness_matrices)
        )
        if not parts:
            raise ValueError("at least one parameter matrix is required")
        if fixed_stiffness is None:
            fixed = tuple((0.0,) * size for _ in range(size))
        else:
            fixed = as_matrix(fixed_stiffness, name="fixed_stiffness", square=True)
        for matrix, label in [(fixed, "fixed_stiffness")] + [
            (p, f"stiffness_matrices[{i}]") for i, p in enumerate(parts)
        ]:
            if len(matrix) != size:
                raise ValueError(f"{label} must have the size of the mass matrix")
            _check_symmetric(matrix, label)
        self._mass = m
        self._parts = parts
        self._fixed = fixed
        try:
            cholesky(self.stiffness((1.0,) * len(parts)))
        except ValueError as error:
            raise ValueError("K(theta=1) must be positive definite") from error

    @classmethod
    def spring_mass_chain(
        cls,
        masses: Sequence[float],
        stiffnesses: Sequence[float],
        *,
        grounded: Literal["base", "both"] = "base",
        updated: Sequence[int] | None = None,
    ) -> ModalStructure:
        """Build a lumped mass-spring chain (shear-building idealization).

        DOF ``j`` carries mass ``masses[j]``. Spring ``i`` joins DOF ``i - 1``
        and DOF ``i`` (spring 0 joins DOF 0 to the ground). With
        ``grounded="both"`` one more spring joins the last DOF to the ground
        (``n + 1`` stiffnesses). Parameter ``theta_i`` scales spring ``i``;
        the springs not in ``updated`` are held at their nominal stiffness.

        Args:
            masses: Positive masses (kg).
            stiffnesses: Positive nominal spring stiffnesses (N/m): ``n`` for
                ``"base"`` and ``n + 1`` for ``"both"``.
            grounded: ``"base"`` (fixed at the base only) or ``"both"``.
            updated: Indices of the springs that are parameters (default all);
                the parameter order is the order given.

        Returns:
            The structure; parameter ``q`` scales spring ``updated[q]``.

        Raises:
            TypeError: If an entry is not real.
            ValueError: For non-positive values, wrong counts, or invalid or
                repeated indices.

        Equation:
            [EQ-INV-MODAL-001].

        References:
            [REF-BECK-KATAFYGIOTIS-1998].

        """
        mass_values = tuple(_positive(v, "mass") for v in masses)
        springs = tuple(_positive(v, "stiffness") for v in stiffnesses)
        n = len(mass_values)
        if n < 1:
            raise ValueError("at least one mass is required")
        if grounded not in ("base", "both"):
            raise ValueError("grounded must be 'base' or 'both'")
        expected = n if grounded == "base" else n + 1
        if len(springs) != expected:
            raise ValueError(f"expected {expected} stiffnesses, got {len(springs)}")
        chosen = tuple(range(len(springs))) if updated is None else tuple(updated)
        if len(set(chosen)) != len(chosen) or any(
            not isinstance(i, int) or not 0 <= i < len(springs) for i in chosen
        ):
            raise ValueError("updated must be distinct valid spring indices")

        def spring_matrix(i: int) -> Matrix:
            """Return the matrix of spring ``i`` at its nominal stiffness.

            References:
                [EQ-INV-MODAL-001]. No new equation is introduced.

            """
            rows = [[0.0] * n for _ in range(n)]
            k = springs[i]
            if i < n:
                rows[i][i] += k
            if i >= 1:
                rows[i - 1][i - 1] += k
                if i < n:
                    rows[i - 1][i] -= k
                    rows[i][i - 1] -= k
            return tuple(tuple(r) for r in rows)

        fixed = [[0.0] * n for _ in range(n)]
        for i in range(len(springs)):
            if i not in chosen:
                part = spring_matrix(i)
                for a in range(n):
                    for b in range(n):
                        fixed[a][b] += part[a][b]
        mass = tuple(
            tuple(mass_values[a] if a == b else 0.0 for b in range(n)) for a in range(n)
        )
        return cls(
            mass,
            [spring_matrix(i) for i in chosen],
            fixed_stiffness=tuple(tuple(r) for r in fixed),
        )

    @property
    def size(self) -> int:
        """Return the number ``n`` of degrees of freedom.

        References:
            [EQ-INV-MODAL-001]. No new equation is introduced.

        """
        return len(self._mass)

    @property
    def parameter_count(self) -> int:
        """Return the number of stiffness parameters.

        References:
            [EQ-INV-MODAL-001]. No new equation is introduced.

        """
        return len(self._parts)

    @property
    def mass(self) -> Matrix:
        """Return the mass matrix (kg).

        References:
            [EQ-INV-MODAL-001]. No new equation is introduced.

        """
        return self._mass

    def _theta(self, theta: Sequence[float]) -> Vector:
        """Validate a positive parameter vector.

        Raises:
            ValueError: If the length is wrong or an entry is not positive.

        References:
            [EQ-INV-MODAL-001]. No new equation is introduced.

        """
        values = as_vector(theta, name="theta")
        if len(values) != self.parameter_count:
            raise ValueError(f"theta needs {self.parameter_count} entries")
        if any(v <= 0.0 for v in values):
            raise ValueError("theta must be positive")
        return values

    def stiffness(self, theta: Sequence[float]) -> Matrix:
        """Return ``K(theta) = K_0 + sum_i theta_i K_i``.

        Args:
            theta: Positive stiffness scalings.

        Returns:
            The stiffness matrix (N/m).

        Raises:
            ValueError: If ``theta`` has the wrong length or is not positive.

        Equation:
            [EQ-INV-MODAL-001].

        References:
            [REF-BECK-KATAFYGIOTIS-1998].

        """
        values = self._theta(theta)
        n = self.size
        return tuple(
            tuple(
                self._fixed[a][b]
                + math.fsum(
                    t * part[a][b] for t, part in zip(values, self._parts, strict=True)
                )
                for b in range(n)
            )
            for a in range(n)
        )

    def modes(self, theta: Sequence[float]) -> ModalSolution:
        """Solve the generalized eigenproblem at ``theta``.

        Args:
            theta: Positive stiffness scalings.

        Returns:
            Eigenvalues and mass-normalized shapes.

        Raises:
            ValueError: As :meth:`stiffness` and :func:`generalized_eigen`.

        Equation:
            [EQ-INV-MODAL-001].

        References:
            [REF-BECK-KATAFYGIOTIS-1998].

        """
        values, shapes = generalized_eigen(self.stiffness(theta), self._mass)
        return ModalSolution(values, shapes)

    def eigenvalue_sensitivities(
        self, theta: Sequence[float], mode_indices: Sequence[int] | None = None
    ) -> Matrix:
        """Return ``d lambda_j / d theta_i = phi_j' K_i phi_j``.

        Row ``q`` is mode ``mode_indices[q]``, column ``i`` is parameter ``i``.

        Args:
            theta: Positive stiffness scalings.
            mode_indices: Modes to differentiate (default all).

        Returns:
            The sensitivity matrix.

        Raises:
            ValueError: If an index is invalid or a requested eigenvalue is
                not simple (relative gap to a neighbour below ``1e-8``).

        Equation:
            [EQ-INV-MODAL-002].

        References:
            [REF-BECK-KATAFYGIOTIS-1998], [REF-GOLUB-VANLOAN-2013].

        """
        solution = self.modes(theta)
        n = self.size
        chosen = tuple(range(n)) if mode_indices is None else tuple(mode_indices)
        if any(not isinstance(j, int) or not 0 <= j < n for j in chosen):
            raise ValueError("mode_indices must be valid mode indices")
        lam = solution.eigenvalues
        rows: list[Vector] = []
        for j in chosen:
            for other in (j - 1, j + 1):
                if 0 <= other < n and abs(lam[j] - lam[other]) <= _SIMPLE_GAP * max(
                    abs(lam[j]), abs(lam[other])
                ):
                    raise ValueError(f"eigenvalue {j} is not simple")
            phi = solution.shape(j)
            rows.append(
                tuple(
                    math.fsum(
                        phi[a] * part[a][b] * phi[b] for a in range(n) for b in range(n)
                    )
                    for part in self._parts
                )
            )
        return tuple(rows)


@dataclass(frozen=True, slots=True)
class ModalData:
    """Identified modal data with their error model.

    Args:
        frequencies_hz: Identified natural frequencies (Hz, positive).
        mode_indices: Distinct 0-based model mode index of each frequency.
        frequency_std: Standard deviation ``delta`` of the error of
            ``log f`` (about the relative error).
        dofs: Measured degrees of freedom (distinct, at least two) when shapes
            are given, otherwise empty.
        shapes: One vector per data mode with the identified shape at ``dofs``
            (rescaled to unit Euclidean norm; the scale and sign of the
            measured shape are irrelevant), or empty for frequency-only data.
        shape_std: Standard deviation ``sigma`` of a unit-norm shape component
            (required positive when shapes are given).

    Raises:
        TypeError: If an entry has the wrong type.
        ValueError: For invalid values or inconsistent lengths.

    Equation:
        [EQ-INV-MODAL-004], [EQ-INV-MODAL-005].

    References:
        [REF-BECK-KATAFYGIOTIS-1998].

    """

    frequencies_hz: Vector
    mode_indices: tuple[int, ...]
    frequency_std: float
    dofs: tuple[int, ...] = ()
    shapes: tuple[Vector, ...] = ()
    shape_std: float = 0.0

    def __post_init__(self) -> None:
        """Validate and normalize the fields.

        References:
            [EQ-INV-MODAL-004], [EQ-INV-MODAL-005].

        """
        freqs = as_vector(self.frequencies_hz, name="frequencies_hz")
        if not freqs or any(f <= 0.0 for f in freqs):
            raise ValueError("frequencies_hz must be non-empty and positive")
        modes = tuple(self.mode_indices)
        if (
            len(modes) != len(freqs)
            or len(set(modes)) != len(modes)
            or any(not isinstance(j, int) or j < 0 for j in modes)
        ):
            raise ValueError("mode_indices must be distinct, one per frequency")
        delta = _positive(self.frequency_std, "frequency_std")
        dofs = tuple(self.dofs)
        rows = tuple(as_vector(s, name="shape") for s in self.shapes)
        if rows or dofs:
            if len(rows) != len(freqs):
                raise ValueError("one shape per data mode is required")
            if (
                len(dofs) < 2
                or len(set(dofs)) != len(dofs)
                or any(not isinstance(i, int) or i < 0 for i in dofs)
            ):
                raise ValueError("dofs must be at least two distinct indices")
            if any(len(r) != len(dofs) for r in rows):
                raise ValueError("each shape needs one entry per measured dof")
            sigma = _positive(self.shape_std, "shape_std")
            unit: list[Vector] = []
            for r in rows:
                length = math.sqrt(dot(r, r))
                if length <= 0.0:
                    raise ValueError("measured shapes must be nonzero")
                unit.append(tuple(v / length for v in r))
            rows = tuple(unit)
        else:
            sigma = 0.0
        object.__setattr__(self, "frequencies_hz", freqs)
        object.__setattr__(self, "mode_indices", modes)
        object.__setattr__(self, "frequency_std", delta)
        object.__setattr__(self, "dofs", dofs)
        object.__setattr__(self, "shapes", rows)
        object.__setattr__(self, "shape_std", sigma)


def simulate_modal_data(
    structure: ModalStructure,
    theta: Sequence[float],
    mode_indices: Sequence[int],
    *,
    frequency_std: float,
    dofs: Sequence[int] = (),
    shape_std: float = 0.0,
    seed: int | RandomStream | None = 0,
) -> ModalData:
    """Generate synthetic identified data from the model with noise.

    ``log f_hat = log f(theta) + delta z``; each shape is the unit-norm model
    shape at ``dofs`` times a random scale in ``[0.5, 2]`` and a random sign,
    plus ``sigma`` times Gaussian components of the unit-norm shape. Synthetic
    only; it never represents measurements.

    Args:
        structure: The structure.
        theta: True stiffness scalings.
        mode_indices: Modes to "identify".
        frequency_std: ``delta`` (positive).
        dofs: Measured degrees of freedom (empty for frequency-only data).
        shape_std: ``sigma`` (positive when ``dofs`` is given).
        seed: Integer seed or stream.

    Returns:
        The data.

    Raises:
        ValueError: For invalid arguments (see :class:`ModalData`) or a mode
            beyond the structure.

    Equation:
        [EQ-INV-MODAL-004], [EQ-INV-MODAL-005].

    References:
        [REF-BECK-KATAFYGIOTIS-1998].

    """
    modes = tuple(mode_indices)
    if any(not isinstance(j, int) or not 0 <= j < structure.size for j in modes):
        raise ValueError("mode_indices must be valid mode indices")
    delta = _positive(frequency_std, "frequency_std")
    gen = resolve_stream(seed).generator()
    solution = structure.modes(theta)
    freqs = solution.frequencies_hz
    noisy = tuple(freqs[j] * math.exp(delta * gen.gauss(0.0, 1.0)) for j in modes)
    chosen = tuple(dofs)
    if not chosen:
        return ModalData(noisy, modes, delta)
    sigma = _positive(shape_std, "shape_std")
    rows: list[Vector] = []
    for j in modes:
        phi = solution.shape(j)
        part = tuple(phi[i] for i in chosen)
        length = math.sqrt(dot(part, part))
        scale = gen.uniform(0.5, 2.0) * (1.0 if gen.random() < 0.5 else -1.0)
        rows.append(
            tuple(scale * (v / length + sigma * gen.gauss(0.0, 1.0)) for v in part)
        )
    return ModalData(noisy, modes, delta, chosen, tuple(rows), sigma)


class ModalUpdatingProblem:
    """Posterior of the stiffness scalings given modal data.

    The unknowns are ``u_q = log theta_{active[q]}``; the inactive parameters
    are fixed at ``fixed_theta`` (default one: undamaged). The log posterior
    is ``log L(u) + log p(u)`` with the likelihood of [EQ-INV-MODAL-004],
    [EQ-INV-MODAL-005] and the lognormal prior of [EQ-INV-MODAL-006].

    Args:
        structure: The structure.
        data: Identified modal data.
        prior_median: Median ``m_i`` of the lognormal prior of each parameter
            (scalar or one per parameter; default 1).
        prior_log_std: ``s_i`` of ``log theta_i`` (scalar or one per
            parameter; default 0.5).
        active: Indices of the parameters being inferred (default all; may be
            empty, which gives the evidence of a fixed model).
        fixed_theta: Values of the inactive parameters (scalar or one per
            parameter; default 1).
        upper: Optional upper bound ``theta_i <= upper`` of the prior (a
            damage-only class uses 1); the prior is the truncated lognormal.

    Raises:
        TypeError: If an entry is not real.
        ValueError: For invalid values, indices, lengths, a mode index beyond
            the structure, or a measured dof beyond the structure.

    Equation:
        [EQ-INV-MODAL-004], [EQ-INV-MODAL-005], [EQ-INV-MODAL-006].

    References:
        [REF-BECK-KATAFYGIOTIS-1998].

    """

    __slots__ = (
        "_active",
        "_data",
        "_fixed",
        "_log_median",
        "_log_std",
        "_structure",
        "_upper",
    )

    def __init__(
        self,
        structure: ModalStructure,
        data: ModalData,
        *,
        prior_median: float | Sequence[float] = 1.0,
        prior_log_std: float | Sequence[float] = 0.5,
        active: Sequence[int] | None = None,
        fixed_theta: float | Sequence[float] = 1.0,
        upper: float | None = None,
    ) -> None:
        """Validate and store the problem.

        References:
            [EQ-INV-MODAL-006].

        """
        if not isinstance(structure, ModalStructure):
            raise TypeError("structure must be a ModalStructure")
        if not isinstance(data, ModalData):
            raise TypeError("data must be a ModalData")
        count = structure.parameter_count
        if max(data.mode_indices) >= structure.size:
            raise ValueError("a mode index exceeds the number of modes")
        if data.dofs and max(data.dofs) >= structure.size:
            raise ValueError("a measured dof exceeds the structure size")

        def per_parameter(value: float | Sequence[float], name: str) -> Vector:
            """Broadcast a scalar or validate a per-parameter sequence.

            References:
                [EQ-INV-MODAL-006]. No new equation is introduced.

            """
            if isinstance(value, Sequence):
                values = tuple(_positive(v, name) for v in value)
                if len(values) != count:
                    raise ValueError(f"{name} needs {count} entries")
                return values
            return (_positive(value, name),) * count

        median = per_parameter(prior_median, "prior_median")
        std = per_parameter(prior_log_std, "prior_log_std")
        fixed = per_parameter(fixed_theta, "fixed_theta")
        chosen = tuple(range(count)) if active is None else tuple(active)
        if len(set(chosen)) != len(chosen) or any(
            not isinstance(i, int) or not 0 <= i < count for i in chosen
        ):
            raise ValueError("active must be distinct valid parameter indices")
        self._upper = None if upper is None else _positive(upper, "upper")
        self._structure = structure
        self._data = data
        self._active = chosen
        self._fixed = fixed
        self._log_median = tuple(math.log(median[i]) for i in chosen)
        self._log_std = tuple(std[i] for i in chosen)

    @property
    def structure(self) -> ModalStructure:
        """Return the structure.

        References:
            [EQ-INV-MODAL-001]. No new equation is introduced.

        """
        return self._structure

    @property
    def data(self) -> ModalData:
        """Return the modal data.

        References:
            [EQ-INV-MODAL-004]. No new equation is introduced.

        """
        return self._data

    @property
    def active(self) -> tuple[int, ...]:
        """Return the indices of the inferred parameters.

        References:
            [EQ-INV-MODAL-006]. No new equation is introduced.

        """
        return self._active

    @property
    def prior_log_median(self) -> Vector:
        """Return ``log m_q`` of the active parameters (the prior mode in ``u``).

        References:
            [EQ-INV-MODAL-006]. No new equation is introduced.

        """
        return self._log_median

    @property
    def dimension(self) -> int:
        """Return the number of inferred parameters.

        References:
            [EQ-INV-MODAL-006]. No new equation is introduced.

        """
        return len(self._active)

    def theta(self, u: Sequence[float]) -> Vector:
        """Return the full ``theta`` for the inferred coordinates ``u``.

        Args:
            u: ``log theta`` of the active parameters.

        Returns:
            All ``theta_i`` (inactive ones at their fixed value).

        Raises:
            ValueError: If ``u`` has the wrong length or overflows.

        Equation:
            [EQ-INV-MODAL-006].

        References:
            [REF-BECK-KATAFYGIOTIS-1998].

        """
        point = as_vector(u, name="u")
        if len(point) != self.dimension:
            raise ValueError(f"u needs {self.dimension} entries")
        full = list(self._fixed)
        for q, i in enumerate(self._active):
            if point[q] > 700.0:
                raise ValueError("u is too large")
            full[i] = math.exp(point[q])
        return tuple(full)

    def _data_residuals(self, u: Sequence[float]) -> Vector | None:
        """Return the whitened data residuals or ``None`` if infeasible.

        ``None`` means a non-positive eigenvalue (``K(theta)`` not positive
        definite), which has zero likelihood.

        Raises:
            ValueError: If a model shape vanishes at the measured dofs.

        References:
            [EQ-INV-MODAL-004], [EQ-INV-MODAL-005].

        """
        data = self._data
        try:
            solution = self._structure.modes(self.theta(u))
        except ValueError:
            return None
        if solution.eigenvalues[0] <= 0.0:
            return None
        out: list[float] = []
        for k, j in enumerate(data.mode_indices):
            log_f = 0.5 * math.log(solution.eigenvalues[j]) - math.log(2.0 * math.pi)
            out.append((math.log(data.frequencies_hz[k]) - log_f) / data.frequency_std)
        for k, j in enumerate(data.mode_indices if data.shapes else ()):
            phi = solution.shape(j)
            psi = tuple(phi[i] for i in data.dofs)
            norm2 = dot(psi, psi)
            if norm2 <= 0.0:
                raise ValueError("model shape vanishes at the measured dofs")
            scale = dot(data.shapes[k], psi) / norm2
            out.extend(
                (m - scale * p) / data.shape_std
                for m, p in zip(data.shapes[k], psi, strict=True)
            )
        return tuple(out)

    @property
    def _log_likelihood_constant(self) -> float:
        """Return the normalizing constant of the Gaussian data density.

        References:
            [EQ-INV-MODAL-004], [EQ-INV-MODAL-005].

        """
        data = self._data
        const = (
            -0.5
            * len(data.frequencies_hz)
            * (_LOG_TWO_PI + 2.0 * math.log(data.frequency_std))
        )
        if data.shapes:
            count = len(data.shapes) * len(data.dofs)
            const -= 0.5 * count * (_LOG_TWO_PI + 2.0 * math.log(data.shape_std))
        return const

    def log_likelihood(self, u: Sequence[float]) -> float:
        """Return ``log p(data | theta(u))`` including the constants.

        Args:
            u: ``log theta`` of the active parameters.

        Returns:
            The log likelihood, ``-inf`` where ``K(theta)`` is not positive
            definite.

        Raises:
            ValueError: If ``u`` is invalid or a model shape vanishes at the
                measured degrees of freedom.

        Equation:
            [EQ-INV-MODAL-004], [EQ-INV-MODAL-005].

        References:
            [REF-BECK-KATAFYGIOTIS-1998].

        """
        residual = self._data_residuals(u)
        if residual is None:
            return -math.inf
        return self._log_likelihood_constant - 0.5 * dot(residual, residual)

    def log_prior(self, u: Sequence[float]) -> float:
        """Return the log density of the (truncated) lognormal prior in ``u``.

        Args:
            u: ``log theta`` of the active parameters.

        Returns:
            ``sum_q log N(u_q; log m_q, s_q^2)`` (the truncation constant is
            not included; it cancels in MAP, Laplace weights and TMCMC), or
            ``-inf`` outside the support.

        Raises:
            ValueError: If ``u`` has the wrong length.

        Equation:
            [EQ-INV-MODAL-006].

        References:
            [REF-BECK-KATAFYGIOTIS-1998].

        """
        point = as_vector(u, name="u")
        if len(point) != self.dimension:
            raise ValueError(f"u needs {self.dimension} entries")
        if self._upper is not None and any(v > math.log(self._upper) for v in point):
            return -math.inf
        return math.fsum(
            -0.5 * ((v - m) / s) ** 2 - math.log(s) - 0.5 * _LOG_TWO_PI
            for v, m, s in zip(point, self._log_median, self._log_std, strict=True)
        )

    def log_posterior(self, u: Sequence[float]) -> float:
        """Return ``log L(u) + log p(u)`` (unnormalized).

        Args:
            u: ``log theta`` of the active parameters.

        Returns:
            The log posterior density in ``u``.

        Raises:
            ValueError: As :meth:`log_likelihood` and :meth:`log_prior`.

        Equation:
            [EQ-INV-MODAL-006].

        References:
            [REF-BECK-KATAFYGIOTIS-1998].

        """
        prior = self.log_prior(u)
        if prior == -math.inf:
            return -math.inf
        return prior + self.log_likelihood(u)

    def residuals(self, u: Sequence[float]) -> Vector:
        """Return the stacked whitened residuals ``[data; prior]``.

        ``-log posterior = 1/2 ||r(u)||^2 + const`` with the data residuals of
        [EQ-INV-MODAL-004], [EQ-INV-MODAL-005] followed by
        ``(u_q - log m_q) / s_q``.

        Args:
            u: ``log theta`` of the active parameters.

        Returns:
            The residual vector.

        Raises:
            ValueError: If ``K(theta)`` is not positive definite or ``u`` is
                invalid.

        Equation:
            [EQ-INV-MODAL-007].

        References:
            [REF-BECK-KATAFYGIOTIS-1998].

        """
        data = self._data_residuals(u)
        if data is None:
            raise ValueError("K(theta) is not positive definite at u")
        point = as_vector(u, name="u")
        prior = tuple(
            (v - m) / s
            for v, m, s in zip(point, self._log_median, self._log_std, strict=True)
        )
        return data + prior

    def prior_sample(self, generator: random.Random) -> Vector:
        """Draw ``u`` from the (truncated) prior by rejection.

        Args:
            generator: Source of randomness.

        Returns:
            A draw with positive prior density.

        Raises:
            ValueError: If the truncation rejects 10000 consecutive draws.

        Equation:
            [EQ-INV-MODAL-006].

        References:
            [REF-CHING-CHEN-2007].

        """
        for _ in range(10000):
            draw = tuple(
                generator.gauss(m, s)
                for m, s in zip(self._log_median, self._log_std, strict=True)
            )
            if self._upper is None or all(v <= math.log(self._upper) for v in draw):
                return draw
        raise ValueError("prior truncation rejects almost every draw")


@dataclass(frozen=True, slots=True)
class ModalLaplace:
    """MAP point and Laplace approximation of a modal-updating posterior.

    Args:
        problem: The problem.
        u_map: MAP of ``u = log theta`` (active parameters).
        theta_map: Full ``theta`` at the MAP.
        log_posterior: Unnormalized log posterior at the MAP.
        covariance_u: ``(J' J)^-1`` in ``u`` (``None`` if the Gauss-Newton
            Hessian is not positive definite).
        log_evidence: Laplace evidence (NaN if ``covariance_u`` is ``None``).
        converged: Whether the optimizer met its tolerance and the Hessian is
            positive definite.
        message: Optimizer termination reason.

    Equation:
        [EQ-INV-MODAL-007].

    References:
        [REF-BECK-KATAFYGIOTIS-1998].

    """

    problem: ModalUpdatingProblem
    u_map: Vector
    theta_map: Vector
    log_posterior: float
    covariance_u: Matrix | None
    log_evidence: float
    converged: bool
    message: str

    def probability_below(self, parameter: int, level: float) -> float:
        """Return the Laplace value of ``P(theta_i < level | data)``.

        Args:
            parameter: Parameter index.
            level: Positive threshold.

        Returns:
            ``Phi((log level - u_map_i) / sd_i)`` for an active parameter; the
            indicator ``fixed < level`` for an inactive one.

        Raises:
            ValueError: If the index or level is invalid or the covariance is
                unavailable.

        Equation:
            [EQ-INV-MODAL-008].

        References:
            [REF-BECK-KATAFYGIOTIS-1998].

        """
        bound = _positive(level, "level")
        count = self.problem.structure.parameter_count
        if not isinstance(parameter, int) or not 0 <= parameter < count:
            raise ValueError("parameter index out of range")
        if parameter not in self.problem.active:
            return 1.0 if self.theta_map[parameter] < bound else 0.0
        if self.covariance_u is None:
            raise ValueError("the Laplace covariance is unavailable")
        q = self.problem.active.index(parameter)
        sd = math.sqrt(self.covariance_u[q][q])
        z = (math.log(bound) - self.u_map[q]) / sd
        return 0.5 * math.erfc(-z / math.sqrt(2.0))

    def damage_probability(self, parameter: int, fraction: float) -> float:
        """Return ``P(theta_i < 1 - fraction | data)`` (Laplace).

        Args:
            parameter: Parameter index.
            fraction: Stiffness loss ``X`` in ``(0, 1)``.

        Returns:
            The probability that the stiffness dropped by more than ``X``.

        Raises:
            ValueError: If ``fraction`` is not in ``(0, 1)`` or as
                :meth:`probability_below`.

        Equation:
            [EQ-INV-MODAL-008].

        References:
            [REF-BECK-KATAFYGIOTIS-1998].

        """
        x = require_finite_real(fraction, name="fraction")
        if not 0.0 < x < 1.0:
            raise ValueError("fraction must lie in (0, 1)")
        return self.probability_below(parameter, 1.0 - x)


def modal_laplace(
    problem: ModalUpdatingProblem,
    initial: Sequence[float] | None = None,
    *,
    max_iterations: int = 200,
    tolerance: float = 1e-10,
) -> ModalLaplace:
    """Find the MAP by Levenberg-Marquardt and return the Laplace posterior.

    The Gauss-Newton Hessian is ``J'J`` with ``J`` the Jacobian of
    :meth:`ModalUpdatingProblem.residuals` (central differences) at the MAP;
    ``covariance_u = (J'J)^-1`` and ``log Z ~ log L + log p + d/2 log 2 pi -
    1/2 log det(J'J)``. A local optimum is returned: start from several
    points when the posterior may be multimodal.

    Args:
        problem: The problem (at least one active parameter).
        initial: Start in ``u`` (default the prior median).
        max_iterations: Levenberg-Marquardt iteration limit.
        tolerance: Levenberg-Marquardt tolerance.

    Returns:
        MAP, covariance, evidence and status.

    Raises:
        ValueError: For invalid options, a start with ``K(theta)`` not
            positive definite, or no active parameter.

    Equation:
        [EQ-INV-MODAL-007].

    References:
        [REF-BECK-KATAFYGIOTIS-1998], [REF-NOCEDAL-WRIGHT-2006].

    """
    if problem.dimension < 1:
        raise ValueError("at least one active parameter is required")
    steps = require_non_negative_integer(max_iterations, name="max_iterations")
    if steps < 1:
        raise ValueError("max_iterations must be positive")
    start = (
        problem.prior_log_median
        if initial is None
        else as_vector(initial, name="initial")
    )
    fit = least_squares(
        problem.residuals, start, tolerance=tolerance, max_iterations=steps
    )
    u_map = fit.x
    jac = fit.jacobian
    d = problem.dimension
    hessian = tuple(
        tuple(math.fsum(row[a] * row[b] for row in jac) for b in range(d))
        for a in range(d)
    )
    covariance: Matrix | None
    try:
        covariance = inverse_spd(hessian)
        log_evidence = (
            problem.log_likelihood(u_map)
            + problem.log_prior(u_map)
            + 0.5 * d * _LOG_TWO_PI
            - 0.5 * log_det_spd(hessian)
        )
        spd = True
    except ValueError:
        covariance = None
        log_evidence = math.nan
        spd = False
    return ModalLaplace(
        problem,
        u_map,
        problem.theta(u_map),
        problem.log_posterior(u_map),
        covariance,
        log_evidence,
        bool(fit.converged and spd),
        fit.message,
    )


@dataclass(frozen=True, slots=True)
class ModalTMCMC:
    """Transitional-MCMC posterior of a modal-updating problem.

    Args:
        problem: The problem.
        result: The :class:`~probstat.inference.TMCMCResult` in ``u``.

    Equation:
        [EQ-INV-MODAL-006], [EQ-INV-MODAL-008].

    References:
        [REF-CHING-CHEN-2007].

    """

    problem: ModalUpdatingProblem
    result: TMCMCResult

    @property
    def converged(self) -> bool:
        """Return whether the TMCMC run reached the posterior.

        References:
            [REF-CHING-CHEN-2007]. No equation is evaluated.

        """
        return self.result.converged

    @property
    def log_evidence(self) -> float:
        """Return the TMCMC log evidence (NaN unless converged).

        References:
            [REF-CHING-CHEN-2007]. No equation is evaluated.

        """
        return self.result.log_evidence

    @property
    def theta_samples(self) -> tuple[Vector, ...]:
        """Return posterior samples of the full ``theta``.

        Raises:
            ValueError: If the run did not converge.

        Equation:
            [EQ-INV-MODAL-006].

        References:
            [REF-CHING-CHEN-2007].

        """
        if not self.result.converged:
            raise ValueError(
                f"TMCMC did not converge (stop_reason={self.result.stop_reason!r})"
            )
        return tuple(self.problem.theta(p) for p in self.result.particles)

    def probability_below(self, parameter: int, level: float) -> float:
        """Return the sample fraction with ``theta_i < level``.

        Args:
            parameter: Parameter index.
            level: Positive threshold.

        Returns:
            The fraction of posterior samples (equal weights).

        Raises:
            ValueError: If the index or level is invalid or the run did not
                converge.

        Equation:
            [EQ-INV-MODAL-008].

        References:
            [REF-CHING-CHEN-2007].

        """
        bound = _positive(level, "level")
        count = self.problem.structure.parameter_count
        if not isinstance(parameter, int) or not 0 <= parameter < count:
            raise ValueError("parameter index out of range")
        samples = self.theta_samples
        return sum(1 for s in samples if s[parameter] < bound) / len(samples)

    def damage_probability(self, parameter: int, fraction: float) -> float:
        """Return the sample estimate of ``P(theta_i < 1 - fraction | data)``.

        Args:
            parameter: Parameter index.
            fraction: Stiffness loss ``X`` in ``(0, 1)``.

        Returns:
            The posterior probability of a stiffness loss above ``X``.

        Raises:
            ValueError: If ``fraction`` is not in ``(0, 1)`` or as
                :meth:`probability_below`.

        Equation:
            [EQ-INV-MODAL-008].

        References:
            [REF-CHING-CHEN-2007].

        """
        x = require_finite_real(fraction, name="fraction")
        if not 0.0 < x < 1.0:
            raise ValueError("fraction must lie in (0, 1)")
        return self.probability_below(parameter, 1.0 - x)


def modal_tmcmc(
    problem: ModalUpdatingProblem,
    *,
    particles: int = 500,
    seed: int | RandomStream | None = 0,
    burn_in: int = 5,
    cov_target: float = 1.0,
    max_stages: int = 100,
) -> ModalTMCMC:
    """Sample the modal-updating posterior by transitional MCMC.

    Args:
        problem: The problem (at least one active parameter).
        particles: Number of samples (at least 2).
        seed: Integer seed or stream.
        burn_in: Discarded Metropolis steps per chain and stage.
        cov_target: Target coefficient of variation of the weights.
        max_stages: Maximum number of stages.

    Returns:
        The samples, schedule, evidence and status (check ``converged``).

    Raises:
        ValueError: For invalid options or no active parameter.

    Equation:
        [EQ-INV-MODAL-006].

    References:
        [REF-CHING-CHEN-2007].

    """
    if problem.dimension < 1:
        raise ValueError("at least one active parameter is required")
    names = tuple(f"log_theta_{i}" for i in problem.active)
    result = transitional_mcmc(
        problem.prior_sample,
        problem.log_prior,
        problem.log_likelihood,
        particles=particles,
        burn_in=burn_in,
        cov_target=cov_target,
        max_stages=max_stages,
        seed=seed,
        names=names,
    )
    return ModalTMCMC(problem, result)


@dataclass(frozen=True, slots=True)
class DamageClassSelection:
    """Posterior probabilities of ``undamaged`` and ``damage_j`` classes.

    Args:
        names: ``"undamaged"`` followed by ``"damage_<j>"`` per segment.
        segments: Parameter index of each damage class.
        selection: Evidences and posterior probabilities.
        runs: The TMCMC run of each damage class.

    Equation:
        [EQ-INV-MODAL-009].

    References:
        [REF-BECK-YUEN-2004], [REF-CHING-CHEN-2007].

    """

    names: tuple[str, ...]
    segments: tuple[int, ...]
    selection: ModelSelection
    runs: tuple[ModalTMCMC, ...]

    @property
    def best(self) -> str:
        """Return the class with the largest posterior probability.

        References:
            [EQ-INV-MODAL-009]. No new equation is introduced.

        """
        return self.selection.best

    def probability(self, name: str) -> float:
        """Return the posterior probability of a class.

        Args:
            name: Class name.

        Returns:
            ``P(class | data)``.

        Raises:
            ValueError: If the name is unknown.

        Equation:
            [EQ-INV-MODAL-009].

        References:
            [REF-BECK-YUEN-2004].

        """
        if name not in self.names:
            raise ValueError(f"unknown class {name!r}")
        return self.selection.posterior_probabilities[self.names.index(name)]


def damage_class_selection(
    structure: ModalStructure,
    data: ModalData,
    *,
    segments: Sequence[int] | None = None,
    prior_median: float = 1.0,
    prior_log_std: float = 0.5,
    prior_probabilities: Sequence[float] | None = None,
    particles: int = 400,
    seed: int | RandomStream | None = 0,
) -> DamageClassSelection:
    """Compare ``undamaged`` with ``damage_j`` classes by TMCMC evidence.

    Class ``undamaged`` fixes every ``theta_i = 1`` (its evidence is the
    likelihood). Class ``damage_j`` infers ``theta_j`` alone (all others
    fixed at one) with the lognormal prior truncated to ``theta_j <= 1``
    (damage can only reduce stiffness); its evidence is the TMCMC estimate.

    Args:
        structure: The structure.
        data: Identified modal data.
        segments: Parameter indices to test (default all).
        prior_median: Median of the lognormal prior before truncation.
        prior_log_std: Log-standard deviation of that prior.
        prior_probabilities: Prior class probabilities in the order
            ``undamaged, damage_<segments...>`` (default uniform).
        particles: TMCMC samples per class.
        seed: Integer seed or stream (each class gets its own child stream).

    Returns:
        The class probabilities and the runs.

    Raises:
        ValueError: For invalid arguments or a TMCMC run that did not
            converge.

    Equation:
        [EQ-INV-MODAL-009].

    References:
        [REF-BECK-YUEN-2004], [REF-CHING-CHEN-2007].

    """
    chosen = (
        tuple(range(structure.parameter_count)) if segments is None else tuple(segments)
    )
    if not chosen:
        raise ValueError("at least one segment is required")
    stream = resolve_stream(seed)
    null = ModalUpdatingProblem(structure, data, active=())
    evidences: dict[str, float | TMCMCResult] = {"undamaged": null.log_likelihood(())}
    runs: list[ModalTMCMC] = []
    for j in chosen:
        problem = ModalUpdatingProblem(
            structure,
            data,
            prior_median=prior_median,
            prior_log_std=prior_log_std,
            active=(j,),
            upper=1.0,
        )
        run = modal_tmcmc(problem, particles=particles, seed=stream.child("class", j))
        runs.append(run)
        evidences[f"damage_{j}"] = run.result
    selection = model_class_selection(evidences, prior_probabilities)
    return DamageClassSelection(tuple(evidences), chosen, selection, tuple(runs))
