"""Change of variables for densities and first-order (delta-method) propagation.

These are the multivariable-calculus bridges of the prerequisite track: a
smooth invertible map moves a density by the absolute Jacobian determinant of
its inverse, and a smooth map moves the first two moments approximately by its
Jacobian. Both accept analytic Jacobians and fall back to central differences.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from probstat.foundations.validation import require_finite_real
from probstat.linalg import (
    Matrix,
    Vector,
    as_matrix,
    as_vector,
    cholesky_semidefinite,
    log_abs_determinant,
    sandwich,
    symmetrize,
)
from probstat.optimization import central_jacobian

LogDensityFunction = Callable[[Vector], float]
VectorMap = Callable[[Vector], Sequence[float]]
JacobianMap = Callable[[Vector], Sequence[Sequence[float]]]


def _checked_step(relative_step: object) -> float:
    """Validate a positive finite relative finite-difference step.

    References:
        [REF-NOCEDAL-WRIGHT-2006, Section 8.1]. No equation is evaluated.

    """
    step = require_finite_real(relative_step, name="relative_step")
    if step <= 0.0:
        raise ValueError("relative_step must be positive")
    return step


def _jacobian(
    function: VectorMap,
    jacobian: JacobianMap | None,
    point: Vector,
    relative_step: float,
    *,
    name: str,
) -> tuple[Matrix, int]:
    """Return an analytic or central-difference Jacobian and its call count.

    Args:
        function: Vector map being differentiated.
        jacobian: Optional analytic Jacobian of ``function``.
        point: Evaluation point.
        relative_step: Central-difference step relative to ``max(1, |x_i|)``.
        name: Field name used in error messages.

    Returns:
        Rectangular Jacobian and the number of ``function`` evaluations used.

    Raises:
        ValueError: If the Jacobian is malformed or non-finite.

    Equation:
        [EQ-OPT-FD-001].

    References:
        [REF-NOCEDAL-WRIGHT-2006, Section 8.1].

    """
    if jacobian is not None:
        return as_matrix(jacobian(point), name=name), 0
    matrix, used = central_jacobian(function, point, relative_step=relative_step)
    return matrix, used


@dataclass(frozen=True, slots=True)
class TransformedDensity:
    """Density of ``Y = g(X)`` for a smooth bijection ``g`` with known inverse.

    Calling the object returns ``log p_Y(y)``. The map must be a
    diffeomorphism between open sets of equal dimension; scalars are handled
    as one-component vectors.

    Args:
        base_log_pdf: ``log p_X`` on vectors; may return ``-inf`` off support.
        inverse_map: ``g^-1`` returning a vector of the same dimension as ``y``.
        jacobian_of_inverse: Optional analytic ``d g^-1 / d y``; central
            differences of ``inverse_map`` are used when omitted.
        relative_step: Central-difference step relative to ``max(1, |y_i|)``.

    References:
        [REF-CASELLA-BERGER-2002, Sections 2.1, 4.3, and 4.6],
        [REF-MAGNUS-NEUDECKER-2019, Chapter 9].

    """

    base_log_pdf: LogDensityFunction
    inverse_map: VectorMap
    jacobian_of_inverse: JacobianMap | None = None
    relative_step: float = 1e-6

    def __post_init__(self) -> None:
        """Validate the callables and the finite-difference step.

        Raises:
            TypeError: If a supplied map is not callable.
            ValueError: If ``relative_step`` is not positive and finite.

        References:
            Project validation policy. No equation is evaluated.

        """
        if not callable(self.base_log_pdf) or not callable(self.inverse_map):
            raise TypeError("base_log_pdf and inverse_map must be callable")
        if self.jacobian_of_inverse is not None and not callable(
            self.jacobian_of_inverse
        ):
            raise TypeError("jacobian_of_inverse must be callable or None")
        object.__setattr__(self, "relative_step", _checked_step(self.relative_step))

    def _preimage(self, value: Sequence[float]) -> tuple[Vector, Vector]:
        """Return the validated point ``y`` and its preimage ``x = g^-1(y)``.

        References:
            [EQ-MVN-JAC-001], [REF-CASELLA-BERGER-2002, Section 4.6].

        """
        point = as_vector(value, name="y")
        if not point:
            raise ValueError("y must be non-empty")
        preimage = as_vector(self.inverse_map(point), name="inverse_map(y)")
        if len(preimage) != len(point):
            raise ValueError("inverse_map must preserve dimension")
        return point, preimage

    def log_abs_det_jacobian(self, value: Sequence[float]) -> float:
        """Return ``log |det J_(g^-1)(y)|``.

        Args:
            value: Point ``y`` in the image space.

        Returns:
            Log absolute Jacobian determinant of the inverse map.

        Raises:
            ValueError: If the Jacobian is not square of the right size or is
                singular, or if a map returns non-finite values.

        Equation:
            [EQ-MVN-JAC-001], [EQ-LINALG-008].

        References:
            [REF-MAGNUS-NEUDECKER-2019, Chapter 9],
            [REF-GOLUB-VANLOAN-2013, Section 3.4].

        """
        point, _ = self._preimage(value)
        matrix, _ = _jacobian(
            self.inverse_map,
            self.jacobian_of_inverse,
            point,
            self.relative_step,
            name="jacobian_of_inverse(y)",
        )
        if len(matrix) != len(point) or len(matrix[0]) != len(point):
            raise ValueError("Jacobian of the inverse map must be square")
        try:
            return log_abs_determinant(matrix)
        except ValueError as error:
            raise ValueError("Jacobian of the inverse map is singular") from error

    def log_pdf(self, value: Sequence[float]) -> float:
        """Return ``log p_Y(y) = log p_X(g^-1(y)) + log |det J_(g^-1)(y)|``.

        Args:
            value: Point ``y`` in the image space.

        Returns:
            Log-density of ``Y``; ``-inf`` where the base density vanishes.

        Raises:
            ValueError: If maps return malformed or non-finite values, the base
                log-density is NaN or ``+inf``, or the Jacobian is singular.

        Equation:
            [EQ-MVN-JAC-001].

        References:
            [REF-CASELLA-BERGER-2002, Sections 2.1, 4.3, and 4.6].

        """
        _, preimage = self._preimage(value)
        base = self.base_log_pdf(preimage)
        if isinstance(base, bool) or not isinstance(base, int | float):
            raise TypeError("base_log_pdf must return a real number")
        base_value = float(base)
        if math.isnan(base_value) or base_value == math.inf:
            raise ValueError("base_log_pdf must return a finite value or -inf")
        if base_value == -math.inf:
            return -math.inf
        return base_value + self.log_abs_det_jacobian(value)

    def pdf(self, value: Sequence[float]) -> float:
        """Return ``p_Y(y)``.

        Args:
            value: Point ``y`` in the image space.

        Returns:
            Non-negative density value.

        Raises:
            ValueError: Under the conditions documented by :meth:`log_pdf`.

        Equation:
            [EQ-MVN-JAC-001].

        References:
            [REF-CASELLA-BERGER-2002, Section 4.6].

        """
        return math.exp(self.log_pdf(value))

    def __call__(self, value: Sequence[float]) -> float:
        """Return ``log p_Y(y)``; an alias of :meth:`log_pdf`.

        References:
            [EQ-MVN-JAC-001], [REF-CASELLA-BERGER-2002, Section 4.6].

        """
        return self.log_pdf(value)


def transformed_density(
    log_pdf: LogDensityFunction,
    inverse_map: VectorMap,
    jacobian_of_inverse: JacobianMap | None = None,
    *,
    relative_step: float = 1e-6,
) -> TransformedDensity:
    """Build the log-density of ``Y = g(X)`` by the change-of-variables rule.

    Args:
        log_pdf: Log-density of ``X`` on vectors.
        inverse_map: ``g^-1``, mapping ``y`` back to ``x``.
        jacobian_of_inverse: Optional analytic Jacobian ``d g^-1 / d y``;
            ``None`` selects central differences of ``inverse_map``.
        relative_step: Central-difference step relative to ``max(1, |y_i|)``.

    Returns:
        A callable :class:`TransformedDensity` whose call returns
        ``log p_X(g^-1(y)) + log |det J_(g^-1)(y)|``.

    Raises:
        TypeError: If a map is not callable.
        ValueError: If ``relative_step`` is not positive and finite.

    Equation:
        [EQ-MVN-JAC-001].

    References:
        [REF-CASELLA-BERGER-2002, Sections 2.1, 4.3, and 4.6],
        [REF-MAGNUS-NEUDECKER-2019, Chapter 9].

    """
    return TransformedDensity(log_pdf, inverse_map, jacobian_of_inverse, relative_step)


@dataclass(frozen=True, slots=True)
class DeltaMethodResult:
    """First-order moment propagation through a smooth vector map.

    Args:
        mean: Approximate output mean ``g(mu)``.
        covariance: Approximate output covariance ``J Sigma J'``.
        jacobian: Jacobian ``J`` of ``g`` at ``mu``.
        evaluations: Calls of ``g`` (one plus any finite differences).

    References:
        [REF-CASELLA-BERGER-2002, Section 5.5.4].

    """

    mean: Vector
    covariance: Matrix
    jacobian: Matrix
    evaluations: int

    @property
    def standard_errors(self) -> Vector:
        """Return the approximate output standard deviations.

        References:
            [EQ-MVN-DELTA-001], [REF-CASELLA-BERGER-2002, Section 5.5.4].

        """
        return tuple(
            math.sqrt(max(0.0, self.covariance[i][i]))
            for i in range(len(self.covariance))
        )


def delta_method(
    function: VectorMap,
    mean: Sequence[float],
    covariance: Sequence[Sequence[float]],
    *,
    jacobian: JacobianMap | None = None,
    relative_step: float = 1e-6,
) -> DeltaMethodResult:
    """Propagate a mean and covariance through ``g`` by linearization.

    ``E[g(X)] ~= g(mu)`` and ``Cov[g(X)] ~= J Sigma J'`` with ``J`` the
    Jacobian of ``g`` at ``mu``. The approximation is exact for affine maps
    and otherwise ignores curvature (second-order bias ``tr(H Sigma)/2``).
    Normality of ``X`` is not required for the moment statement; the
    asymptotic delta-method theorem adds it for estimators.

    Args:
        function: Smooth map from ``n``-vectors to ``m``-vectors.
        mean: Finite input mean ``mu``.
        covariance: Symmetric positive-semidefinite ``n`` by ``n`` matrix.
        jacobian: Optional analytic Jacobian of ``g``; central differences
            otherwise.
        relative_step: Central-difference step relative to ``max(1, |mu_i|)``.

    Returns:
        Output mean, covariance, Jacobian, and evaluation count.

    Raises:
        TypeError: If ``function`` or ``jacobian`` is not callable.
        ValueError: If shapes disagree, the covariance is not symmetric
            positive semidefinite, or a map returns non-finite values.

    Equation:
        [EQ-MVN-DELTA-001], [EQ-LINALG-007].

    References:
        [REF-CASELLA-BERGER-2002, Section 5.5.4],
        [REF-MAGNUS-NEUDECKER-2019, Chapter 5].

    """
    if not callable(function):
        raise TypeError("function must be callable")
    if jacobian is not None and not callable(jacobian):
        raise TypeError("jacobian must be callable or None")
    step = _checked_step(relative_step)
    center = as_vector(mean, name="mean")
    if not center:
        raise ValueError("mean must be non-empty")
    sigma = symmetrize(covariance, name="covariance")
    if len(sigma) != len(center):
        raise ValueError("covariance dimension must equal mean dimension")
    cholesky_semidefinite(sigma)
    output = as_vector(function(center), name="function(mean)")
    if not output:
        raise ValueError("function must return a non-empty vector")
    matrix, used = _jacobian(function, jacobian, center, step, name="jacobian")
    if len(matrix) != len(output) or len(matrix[0]) != len(center):
        raise ValueError("Jacobian shape must be (output dimension, input dimension)")
    return DeltaMethodResult(output, sandwich(matrix, sigma), matrix, 1 + used)
