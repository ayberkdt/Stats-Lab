"""Correlated measurement errors of one observed quantity across experiments.

A :class:`~probstat.inverse.NoiseModel` with ``correlation=ErrorCorrelation(...)``
declares that the errors of its quantity are jointly Gaussian across the
experiments that observe it:

``eps_q ~ N(0, Sigma_q)``, ``Sigma_q = sigma_q^2 M_q + sigma_delta_q^2 I``,

where ``sigma_q`` is the quantity's noise scale (known or uncertain, exactly
as for independent noise), ``sigma_delta_q^2`` its declared independent
discrepancy variance, and ``M_q`` a symmetric positive-definite shape matrix:

* ``ErrorCorrelation(matrix=M)``: a declared matrix, row ``i`` for the
  ``i``-th experiment (in experiment order) that observes the quantity. A
  correlation matrix (unit diagonal) keeps ``sigma_q`` as the marginal
  standard deviation; a fully known covariance ``C`` is declared as
  ``matrix=C`` with ``standard_deviation=1.0``.
* ``ErrorCorrelation(condition="t", kernel=..., length=...)``: a stationary
  correlation ``M_ij = rho(|c_i - c_j| / ell)`` over a numeric experimental
  condition, with the exponential, Matern-3/2, or Matern-5/2 kernel
  ([EQ-CAL-CORR-001]). The length ``ell`` is known or has a prior
  (``length_prior``), in which case it is sampled on the log scale like an
  unknown noise scale.

The errors are measurement errors: they belong to the instrument or the test
series, not to the model. A correlated *model* discrepancy over the inputs is
the subject of :class:`~probstat.inverse.DiscrepancyCalibration`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from probstat.distributions._autodiff import admath
from probstat.foundations.validation import require_finite_real
from probstat.linalg import Matrix, as_matrix, cholesky
from probstat.modeling import ContinuousMarginal, RandomInput

ERROR_CORRELATION_KERNELS = ("exponential", "matern32", "matern52")
"""Stationary correlation kernels accepted by :class:`ErrorCorrelation`."""

_SQRT3 = math.sqrt(3.0)
_SQRT5 = math.sqrt(5.0)


def error_correlation_kernel(kernel: str, scaled_distance: float) -> float:
    """Return the stationary correlation ``rho(h)`` at ``h = |c - c'| / ell``.

    Exponential ``exp(-h)`` (Matern 1/2), Matern 3/2
    ``(1 + sqrt(3) h) exp(-sqrt(3) h)`` and Matern 5/2
    ``(1 + sqrt(5) h + 5 h^2 / 3) exp(-sqrt(5) h)``. The function is written
    with :class:`~probstat.numerics.admath`, so ``h`` may be a reverse-mode
    variable (an uncertain correlation length).

    Args:
        kernel: One of :data:`ERROR_CORRELATION_KERNELS`.
        scaled_distance: Non-negative distance divided by the length.

    Returns:
        Correlation in ``(0, 1]``.

    Raises:
        ValueError: For an unknown kernel name.

    Equation:
        [EQ-CAL-CORR-001].

    References:
        [REF-RASMUSSEN-WILLIAMS-2006, Section 4.2].

    """
    h = scaled_distance
    if kernel == "exponential":
        return admath.exp(-h)
    if kernel == "matern32":
        return (1.0 + _SQRT3 * h) * admath.exp(-_SQRT3 * h)
    if kernel == "matern52":
        return (1.0 + _SQRT5 * h + 5.0 * h * h / 3.0) * admath.exp(-_SQRT5 * h)
    raise ValueError(f"kernel must be one of {ERROR_CORRELATION_KERNELS}")


@dataclass(frozen=True, slots=True)
class ErrorCorrelation:
    """Correlation structure of one quantity's measurement errors.

    Give either ``matrix`` (a declared shape matrix, see the module
    docstring) or ``condition`` (a stationary kernel over that numeric
    experimental condition). With ``condition`` give exactly one of
    ``length`` (known) and ``length_prior`` (uncertain).

    Args:
        matrix: Symmetric positive-definite ``n_q`` by ``n_q`` matrix, row
            ``i`` for the ``i``-th experiment observing the quantity.
        condition: Name of the experimental condition the kernel acts on.
        kernel: ``"exponential"`` (default), ``"matern32"``, or
            ``"matern52"``.
        length: Known positive correlation length (condition units).
        length_prior: Continuous prior of an uncertain correlation length.

    Raises:
        ValueError: For an invalid combination, a non-symmetric or not
            positive-definite matrix, an unknown kernel, or a non-positive
            length.

    References:
        [REF-SEBER-WILD-2003, Chapter 6], [REF-RASMUSSEN-WILLIAMS-2006,
        Section 4.2], [EQ-CAL-CORR-001].

    """

    matrix: Matrix | None = None
    condition: str | None = None
    kernel: str = "exponential"
    length: float | None = None
    length_prior: ContinuousMarginal | None = None

    def __post_init__(self) -> None:
        """Validate the declaration and freeze the matrix.

        References:
            [EQ-CAL-CORR-001], [REF-GOLUB-VANLOAN-2013, Section 4.2].

        """
        if (self.matrix is None) == (self.condition is None):
            raise ValueError("give exactly one of matrix or condition")
        if self.matrix is not None:
            if (
                self.kernel != "exponential"
                or self.length is not None
                or self.length_prior is not None
            ):
                raise ValueError(
                    "kernel, length and length_prior apply only with condition"
                )
            matrix = as_matrix(self.matrix, name="matrix")
            size = len(matrix)
            if size == 0 or any(len(row) != size for row in matrix):
                raise ValueError("matrix must be square and non-empty")
            scale = max(abs(entry) for row in matrix for entry in row)
            for i in range(size):
                for j in range(i):
                    if abs(matrix[i][j] - matrix[j][i]) > 1e-12 * scale:
                        raise ValueError("matrix must be symmetric")
            try:
                cholesky(matrix)
            except ValueError as error:
                raise ValueError(
                    "the error correlation matrix must be positive definite"
                ) from error
            object.__setattr__(self, "matrix", matrix)
            return
        if not isinstance(self.condition, str) or not self.condition.strip():
            raise ValueError("condition must be a non-empty string")
        if self.kernel not in ERROR_CORRELATION_KERNELS:
            raise ValueError(f"kernel must be one of {ERROR_CORRELATION_KERNELS}")
        if (self.length is None) == (self.length_prior is None):
            raise ValueError("give exactly one of length or length_prior")
        if self.length is not None:
            value = require_finite_real(self.length, name="length")
            if value <= 0.0:
                raise ValueError("length must be positive")
            object.__setattr__(self, "length", value)
        if self.length_prior is not None:
            RandomInput(f"length[{self.condition}]", self.length_prior)

    @property
    def is_known(self) -> bool:
        """Return whether the structure has no uncertain length.

        References:
            [EQ-CAL-CORR-001]. No equation is evaluated.

        """
        return self.length_prior is None

    def correlation_matrix(
        self, values: Sequence[float], length: float | None = None
    ) -> Matrix:
        """Return the shape matrix ``M`` for the given condition values.

        Args:
            values: Condition values of the experiments, in order (only
                their number is used for a declared matrix).
            length: Correlation length; defaults to the known ``length``.

        Returns:
            ``M_ij = rho(|c_i - c_j| / ell)`` (unit diagonal) or the
            declared matrix.

        Raises:
            ValueError: If the size does not match a declared matrix, or no
                length is available for a kernel.

        Equation:
            [EQ-CAL-CORR-001].

        References:
            [REF-RASMUSSEN-WILLIAMS-2006, Section 4.2].

        """
        points = tuple(require_finite_real(v, name="values") for v in values)
        if self.matrix is not None:
            if len(points) != len(self.matrix):
                raise ValueError(
                    f"the declared matrix has {len(self.matrix)} rows but "
                    f"{len(points)} experiments observe the quantity"
                )
            return self.matrix
        scale = self.length if length is None else length
        if scale is None:
            raise ValueError("an uncertain correlation length must be supplied")
        ell = require_finite_real(scale, name="length")
        if ell <= 0.0:
            raise ValueError("length must be positive")
        return tuple(
            tuple(
                1.0
                if i == j
                else error_correlation_kernel(self.kernel, abs(a - b) / ell)
                for j, b in enumerate(points)
            )
            for i, a in enumerate(points)
        )
