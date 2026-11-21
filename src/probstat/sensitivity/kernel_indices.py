"""Kernel (MMD-based) first-order sensitivity indices by pick-freeze.

A positive-definite kernel ``k`` on the output embeds a distribution ``P`` as
its mean element ``mu_P = E_P[k(Y, .)]``; the squared maximum mean discrepancy
``MMD^2(P, Q) = |mu_P - mu_Q|^2`` compares two laws through all features of
the kernel. Replacing the variance of the conditional mean in a Sobol index
by the mean squared distance between the conditional and the unconditional
output law gives the kernel index of da Veiga (2021):

``S_i^k = E_{X_i}[MMD^2(P_Y, P_{Y | X_i})] / (E[k(Y, Y)] - E[k(Y, Y')])``.

With ``Y^i`` an output that shares ``X_i`` with ``Y`` and is otherwise
independent (the pick-freeze pair) and ``Y'`` an independent copy,
``E_{X_i} |mu_{Y | X_i}|^2 = E[k(Y, Y^i)]`` and ``|mu_Y|^2 = E[k(Y, Y')]``, so

``S_i^k = (E[k(Y, Y^i)] - E[k(Y, Y')]) / (E[k(Y, Y)] - E[k(Y, Y')])``

([EQ-SENS-KERNEL-001]). The linear kernel ``k(y, y') = y y'`` gives back the
first-order Sobol index. A characteristic kernel (the Gaussian kernel) gives
an index that is zero if and only if ``Y`` is independent of ``X_i`` and one
if ``Y`` is a function of ``X_i`` alone; it reacts to changes of the whole
output distribution (spread, shape, modes), not only of its mean.

Estimator ([EQ-SENS-KERNEL-002]). From the pick-freeze outputs ``y_B`` and
``y_i = f(A_B^(i))`` (``A`` with column ``i`` from ``B``), the three
expectations are estimated by the pair mean of ``k(y_B, y_i)``, the mean of
``k`` over all ordered pairs of the pooled ``2N`` outputs (a V-statistic),
and the mean of ``k(z, z)`` over the pooled outputs. For the linear kernel
this is exactly the estimator of Janon et al. (2014), ``estimator="janon"``
of :func:`probstat.sensitivity.estimate_sobol_indices`. The V-statistic has
an ``O(1/N)`` bias and costs ``O(N^2)`` kernel evaluations per input.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.modeling import ProbabilisticModel
from probstat.propagation import DESIGNS, make_design

from .analysis import _evaluate, _matrix
from .hsic import _bandwidth

KERNELS = ("gaussian", "linear")
_MEDIAN_SUBSAMPLE = 500

Kernel = Callable[[float, float], float]


@dataclass(frozen=True, slots=True)
class KernelSobolResult:
    """Kernel first-order indices from one pick-freeze design.

    Args:
        input_names: Input names.
        indices: Estimates of ``S_i^k`` (not clipped; small negative values
            are sampling noise).
        kernel: ``"gaussian"``, ``"linear"``, or ``"custom"``.
        bandwidth: Gaussian-kernel bandwidth (``nan`` for other kernels).
        sample_size: Rows ``N`` of each base matrix.
        evaluations: Model evaluations, ``N (d + 1)``.

    References:
        [REF-DA-VEIGA-2021].

    """

    input_names: tuple[str, ...]
    indices: tuple[float, ...]
    kernel: str
    bandwidth: float
    sample_size: int
    evaluations: int

    def as_mapping(self) -> Mapping[str, float]:
        """Return ``input -> kernel index``.

        References:
            [EQ-SENS-KERNEL-001], [REF-DA-VEIGA-2021].

        """
        return dict(zip(self.input_names, self.indices, strict=True))

    def ranking(self) -> tuple[str, ...]:
        """Return input names by decreasing kernel index.

        References:
            [EQ-SENS-KERNEL-001], [REF-DA-VEIGA-2021].

        """
        order = sorted(range(len(self.indices)), key=lambda i: -self.indices[i])
        return tuple(self.input_names[i] for i in order)


def _gaussian_kernel(bandwidth: float) -> Kernel:
    """Return ``k(y, y') = exp(-(y - y')^2 / (2 h^2))``.

    References:
        [EQ-SENS-KERNEL-001], [REF-DA-VEIGA-2021].

    """
    scale = -0.5 / (bandwidth * bandwidth)

    def kernel(first: float, second: float) -> float:
        """Evaluate the Gaussian kernel.

        References:
            [EQ-SENS-KERNEL-001], [REF-DA-VEIGA-2021].

        """
        return math.exp(scale * (first - second) ** 2)

    return kernel


def _linear_kernel(first: float, second: float) -> float:
    """Return the linear kernel ``k(y, y') = y y'``.

    References:
        [EQ-SENS-KERNEL-001], [REF-DA-VEIGA-2021].

    """
    return first * second


def _pair_index(
    kernel: Kernel, base: Sequence[float], frozen: Sequence[float]
) -> float:
    """Return the kernel index of one pick-freeze output pair.

    Raises:
        ValueError: If the kernel does not separate the pooled outputs (zero
            denominator) or returns a non-finite value.

    References:
        [EQ-SENS-KERNEL-002], [REF-DA-VEIGA-2021], [REF-JANON-ETAL-2014].

    """
    n = len(base)
    pooled = tuple(base) + tuple(frozen)
    size = 2 * n
    paired = math.fsum(kernel(a, b) for a, b in zip(base, frozen, strict=True)) / n
    diagonal_sum = math.fsum(kernel(z, z) for z in pooled)
    upper = math.fsum(
        math.fsum(kernel(pooled[a], pooled[b]) for b in range(a + 1, size))
        for a in range(size - 1)
    )
    independent = (diagonal_sum + 2.0 * upper) / (size * size)
    diagonal = diagonal_sum / size
    denominator = diagonal - independent
    if not math.isfinite(denominator) or not math.isfinite(paired):
        raise ValueError("the kernel returned a non-finite value")
    if not denominator > 0.0:
        raise ValueError(
            "kernel indices need E[k(Y, Y)] > E[k(Y, Y')]: the output is "
            "constant or the kernel does not separate its values"
        )
    return (paired - independent) / denominator


def _resolve_kernel(
    kernel: str | Kernel, bandwidth: str | float, reference: Sequence[float]
) -> tuple[Kernel, str, float]:
    """Return the kernel function, its label, and the Gaussian bandwidth.

    Raises:
        ValueError: For an unknown kernel name or bandwidth rule, or a
            non-positive bandwidth.

    References:
        [REF-DA-VEIGA-2021], [REF-GRETTON-ETAL-2008] (median heuristic).

    """
    if callable(kernel):
        return kernel, "custom", math.nan
    if kernel == "linear":
        return _linear_kernel, "linear", math.nan
    if kernel != "gaussian":
        raise ValueError(f"kernel must be a callable or one of {KERNELS}")
    if isinstance(bandwidth, str):
        if bandwidth not in ("median", "std"):
            raise ValueError("bandwidth must be 'median', 'std', or a positive number")
        # The median of the pairwise distances costs O(n^2 log n); an evenly
        # spaced subset of the (i.i.d.) outputs estimates the same quantity.
        step = max(1, len(reference) // _MEDIAN_SUBSAMPLE)
        subset = reference[::step] if bandwidth == "median" else reference
        h = _bandwidth(subset, bandwidth)
    else:
        h = require_finite_real(bandwidth, name="bandwidth")
        if not h > 0.0:
            raise ValueError("bandwidth must be positive")
    return _gaussian_kernel(h), "gaussian", h


def kernel_sobol_indices(
    model: Callable[[tuple[float, ...]], float],
    sample_a: Iterable[Sequence[object]],
    sample_b: Iterable[Sequence[object]],
    *,
    kernel: str | Kernel = "gaussian",
    bandwidth: str | float = "median",
    names: Sequence[str] | None = None,
) -> KernelSobolResult:
    """Estimate kernel first-order indices on caller-supplied base matrices.

    The model is run on ``B`` and on the ``d`` hybrids ``A_B^(i)`` (``A`` with
    column ``i`` from ``B``), ``N (d + 1)`` runs; ``A`` itself is not
    evaluated. For each input, with ``z`` the pooled ``2N`` outputs
    ``(y_B, y_i)``,

    ``S_i^k = (T1 - T0) / (T2 - T0)``,
    ``T1 = (1/N) sum_n k(y_B,n, y_i,n)``,
    ``T0 = (1/(2N)^2) sum_{a,b} k(z_a, z_b)``,
    ``T2 = (1/(2N)) sum_a k(z_a, z_a)``.

    The Gaussian bandwidth defaults to the median pairwise distance of
    ``y_B`` (on at most about 500 evenly spaced values); the index depends on
    the bandwidth, so quote it with the result. Indices of different inputs
    need not sum to at most one for a nonlinear kernel.

    Args:
        model: Deterministic scalar model of mutually independent inputs.
        sample_a: Independent base sample matrix.
        sample_b: Equally shaped independent base sample matrix.
        kernel: ``"gaussian"`` (default), ``"linear"``, or a symmetric
            positive-definite function ``k(y, y')``.
        bandwidth: ``"median"`` (default), ``"std"`` (sample standard
            deviation of ``y_B``), or a positive number; used only by the
            Gaussian kernel.
        names: Input names (default ``x0, x1, ...``).

    Returns:
        Kernel indices with cost accounting.

    Raises:
        TypeError: If ``model`` is not callable.
        ValueError: For invalid matrices, fewer than four rows, an unknown
            kernel or bandwidth, or a kernel that does not separate the
            outputs.

    Equation:
        [EQ-SENS-KERNEL-001], [EQ-SENS-KERNEL-002].

    References:
        [REF-DA-VEIGA-2021], [REF-JANON-ETAL-2014].

    """
    if not callable(model):
        raise TypeError("model must be callable")
    a = _matrix(sample_a, "sample_a")
    b = _matrix(sample_b, "sample_b")
    if len(a) != len(b) or len(a[0]) != len(b[0]):
        raise ValueError("sample matrices must have equal shape")
    n, dimension = len(a), len(a[0])
    if n < 4:
        raise ValueError("at least four rows are required")
    labels = tuple(f"x{i}" for i in range(dimension)) if names is None else tuple(names)
    if len(labels) != dimension:
        raise ValueError("names must match the number of input columns")
    yb = tuple(_evaluate(model, row) for row in b)
    function, label, h = _resolve_kernel(kernel, bandwidth, yb)
    indices: list[float] = []
    for i in range(dimension):
        frozen = tuple(
            _evaluate(model, a_row[:i] + (b_row[i],) + a_row[i + 1 :])
            for a_row, b_row in zip(a, b, strict=True)
        )
        indices.append(_pair_index(function, yb, frozen))
    return KernelSobolResult(labels, tuple(indices), label, h, n, n * (dimension + 1))


def model_kernel_sobol_indices(
    model: ProbabilisticModel,
    quantity: str,
    size: int,
    *,
    seed: int | RandomStream | None,
    design: str = "mc",
    kernel: str | Kernel = "gaussian",
    bandwidth: str | float = "median",
) -> KernelSobolResult:
    """Draw a pick-freeze design for a model and estimate kernel indices.

    The base matrices are the two halves of one ``2d``-dimensional unit
    design mapped through the inverse marginal CDFs, as in
    :func:`probstat.sensitivity.sobol_analysis`.

    Args:
        model: Probabilistic model with independent inputs.
        quantity: QoI name.
        size: Rows ``N`` of each base matrix (at least four); the kernel
            sums cost ``O(N^2)`` per input.
        seed: Root seed or stream (child ``"kernel-sobol"``).
        design: ``"mc"`` (default), ``"lhs"``, ``"halton"``, or ``"sobol"``.
        kernel: See :func:`kernel_sobol_indices`.
        bandwidth: See :func:`kernel_sobol_indices`.

    Returns:
        :class:`KernelSobolResult`; ``evaluations`` is the model's own count.

    Raises:
        ValueError: For dependent inputs or invalid settings.
        KeyError: For an unknown QoI.

    Equation:
        [EQ-SENS-KERNEL-001], [EQ-SENS-KERNEL-002], [EQ-SAMP-INV-001].

    References:
        [REF-DA-VEIGA-2021], [REF-SALTELLI-2010].

    """
    model.quantity(quantity)
    if not model.inputs.is_independent:
        raise ValueError("pick-freeze kernel indices assume independent inputs")
    if isinstance(size, bool) or not isinstance(size, int) or size < 4:
        raise ValueError("size must be an integer of at least four")
    if design not in DESIGNS:
        raise ValueError(f"design must be one of {DESIGNS}")
    dimension = model.dimension
    unit = make_design(
        design, size, 2 * dimension, seed=resolve_stream(seed).child("kernel-sobol")
    )
    function = model.unit_function(quantity)
    before = model.evaluations
    result = kernel_sobol_indices(
        function,
        tuple(point[:dimension] for point in unit.points),
        tuple(point[dimension:] for point in unit.points),
        kernel=kernel,
        bandwidth=bandwidth,
        names=model.inputs.names,
    )
    return KernelSobolResult(
        result.input_names,
        result.indices,
        result.kernel,
        result.bandwidth,
        result.sample_size,
        model.evaluations - before,
    )
