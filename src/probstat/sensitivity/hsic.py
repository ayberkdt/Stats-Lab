"""HSIC dependence measures between each input and the output.

The Hilbert-Schmidt independence criterion compares the joint law of
``(X_i, Y)`` with the product of its marginals through two characteristic
kernels; with Gaussian kernels it is zero if and only if ``X_i`` and ``Y``
are independent, so it detects nonmonotone and non-additive dependence that
a correlation coefficient misses (``Y = X^2`` with a symmetric ``X`` has zero
correlation but a large HSIC). On an i.i.d. sample with Gram matrices
``K_ab = k(x_a, x_b)`` and ``L_ab = l(y_a, y_b)``:

- the biased (V-statistic) estimator is ``tr(K H L H) / n^2`` with the
  centring matrix ``H = I - 11'/n`` ([EQ-SENS-HSIC-001]);
- the unbiased (U-statistic) estimator of Song et al. (2012) removes the
  ``O(1/n)`` bias and is centred at zero under independence
  ([EQ-SENS-HSIC-002]);
- the normalized index ``R2_HSIC = HSIC(X, Y) / sqrt(HSIC(X, X) HSIC(Y, Y))``
  of Da Veiga (2015) lies in ``[0, 1]`` for the biased estimator and allows
  inputs to be compared on one scale.

Tests of ``H0: X_i independent of Y``: a permutation test that shuffles the
output ([EQ-SENS-HSIC-004]), exact in level for i.i.d. rows, and the Gamma
approximation of Gretton et al. (2008) to the null law of ``n HSIC_b``
([EQ-SENS-HSIC-003]), which costs no resampling. Goal-oriented (target)
sensitivity replaces ``Y`` by a transform ``w(Y)``, such as the indicator of
an exceedance ``1{Y > c}``, before the kernel is applied, which measures the
influence of each input on the occurrence of the critical event
([EQ-SENS-HSIC-005]; Marrel and Chabridon 2021).

Kernels and bandwidths. ``k(x, x') = exp(-(x - x')^2 / (2 h^2))``. The
default bandwidth ``h`` is the sample standard deviation of the variable (the
choice of Da Veiga 2015); ``"median"`` uses the median of the pairwise
distances ``|x_a - x_b|``, the heuristic of Gretton et al. (2008), which
adapts to heavy tails but is zero when more than half of the pairs are tied.
HSIC values depend on the kernel and bandwidth; only ``R2_HSIC`` and the
p-values are comparable across inputs measured in different units.

Cost. Each input needs one ``n x n`` Gram matrix, so time and memory are
``O(n^2)`` per input (about 0.2 s and 4 MB for ``n = 1000`` in pure
Python); a permutation test adds ``O(n^2)`` per permutation and input,
shared output permutations keep it to roughly ``0.02 s`` per permutation and
input at ``n = 1000``.
"""

from __future__ import annotations

import math
from array import array
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from operator import itemgetter

from probstat.distributions import Gamma
from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.modeling import ProbabilisticModel

from .regression_based import _validated_sample, model_sample


def _fsum_product(left: Sequence[float], right: Sequence[float]) -> float:
    """Return ``sum l_i r_i`` with ``math.fsum`` (fallback for Python 3.11).

    References:
        [REF-PYTHON-MATH]. No equation is evaluated.

    """
    return math.fsum(a * b for a, b in zip(left, right, strict=False))


# Python 3.12+ ``math.sumprod`` accumulates in extended precision in C.
_sumprod: Callable[[Sequence[float], Sequence[float]], float] = getattr(
    math, "sumprod", _fsum_product
)

ESTIMATORS = ("biased", "unbiased")
TESTS = ("permutation", "gamma")
BANDWIDTHS = ("std", "median")


@dataclass(frozen=True, slots=True)
class HSICResult:
    """HSIC indices of one output with independence tests.

    Args:
        input_names: Input names.
        hsic: ``HSIC(X_i, Y)`` estimates (kernel and bandwidth dependent).
        r2_hsic: Normalized indices ``HSIC(X_i, Y) / sqrt(HSIC(X_i, X_i)
            HSIC(Y, Y))``.
        p_values: p-values of ``H0: X_i independent of Y``.
        estimator: ``"biased"`` or ``"unbiased"``.
        test: ``"permutation"`` or ``"gamma"``.
        permutations: Number of output permutations (0 for ``"gamma"``).
        bandwidths: Gaussian-kernel bandwidths of the inputs.
        output_bandwidth: Bandwidth of the (transformed) output kernel.
        sample_size: Number of rows ``n``.
        target: Description of the output transform (``None`` for the raw
            output).
        target_fraction: Fraction of rows with ``w(Y) > 0`` under a target
            transform (``nan`` without one).
        evaluations: Model evaluations spent (zero for given data).

    References:
        [REF-DA-VEIGA-2015], [REF-GRETTON-ETAL-2008],
        [REF-SONG-ETAL-2012], [REF-MARREL-CHABRIDON-2021].

    """

    input_names: tuple[str, ...]
    hsic: tuple[float, ...]
    r2_hsic: tuple[float, ...]
    p_values: tuple[float, ...]
    estimator: str
    test: str
    permutations: int
    bandwidths: tuple[float, ...]
    output_bandwidth: float
    sample_size: int
    target: str | None = None
    target_fraction: float = math.nan
    evaluations: int = 0

    def as_mapping(self) -> Mapping[str, float]:
        """Return ``input -> R2_HSIC``.

        References:
            [EQ-SENS-HSIC-001], [REF-DA-VEIGA-2015].

        """
        return dict(zip(self.input_names, self.r2_hsic, strict=True))

    def ranking(self) -> tuple[str, ...]:
        """Return input names by decreasing ``R2_HSIC``.

        References:
            [EQ-SENS-HSIC-001], [REF-DA-VEIGA-2015].

        """
        order = sorted(range(len(self.r2_hsic)), key=lambda i: -self.r2_hsic[i])
        return tuple(self.input_names[i] for i in order)

    def independent(self, level: float = 0.05) -> tuple[str, ...]:
        """Return inputs for which independence is not rejected at ``level``.

        These inputs are screening candidates: with ``d`` separate tests
        about ``d level`` dependent inputs are expected to pass by chance
        alone when all are independent, and a weak dependence can pass at a
        small ``n``.

        Args:
            level: Test level in ``(0, 1)``.

        Returns:
            Names with ``p_i >= level``, in input order.

        Raises:
            ValueError: If ``level`` is not in ``(0, 1)``.

        Equation:
            [EQ-SENS-HSIC-003], [EQ-SENS-HSIC-004].

        References:
            [REF-GRETTON-ETAL-2008], [REF-DA-VEIGA-2015].

        """
        alpha = require_finite_real(level, name="level")
        if not 0.0 < alpha < 1.0:
            raise ValueError("level must lie in (0, 1)")
        return tuple(
            name
            for name, p in zip(self.input_names, self.p_values, strict=True)
            if p >= alpha
        )


@dataclass(frozen=True, slots=True)
class _Gram:
    """Gaussian Gram matrix stored as strict upper-triangle rows.

    Args:
        upper: Row ``a < n - 1`` holds ``K[a][a+1:]``; the diagonal is one.
        row_sums: Full row sums of ``K`` (diagonal included).
        total: Sum of all entries.
        full: Full rows (kept only for the output kernel, which is permuted).

    References:
        [EQ-SENS-HSIC-001].

    """

    upper: tuple[array[float], ...]
    row_sums: tuple[float, ...]
    total: float
    full: tuple[array[float], ...] = ()


def _bandwidth(values: Sequence[float], rule: str | float) -> float:
    """Return the Gaussian-kernel bandwidth of one variable.

    Raises:
        ValueError: If the bandwidth is not positive.

    References:
        [REF-DA-VEIGA-2015], [REF-GRETTON-ETAL-2008].

    """
    n = len(values)
    if not isinstance(rule, str):
        h = require_finite_real(rule, name="bandwidth")
    elif rule == "std":
        mean = math.fsum(values) / n
        h = math.sqrt(math.fsum((v - mean) ** 2 for v in values) / (n - 1))
    else:
        distances = sorted(
            abs(values[a] - values[b]) for a in range(n) for b in range(a + 1, n)
        )
        count = len(distances)
        middle = count // 2
        h = (
            distances[middle]
            if count % 2
            else 0.5 * (distances[middle - 1] + distances[middle])
        )
    if not h > 0.0:
        raise ValueError(
            "the kernel bandwidth is zero (a constant variable, or more than "
            "half of the pairs tied under the median heuristic)"
        )
    return h


def _gram(values: Sequence[float], bandwidth: float, *, keep_full: bool) -> _Gram:
    """Build the Gaussian Gram matrix ``exp(-(x_a - x_b)^2 / (2 h^2))``.

    References:
        [EQ-SENS-HSIC-001], [REF-GRETTON-ETAL-2008].

    """
    scale = -0.5 / (bandwidth * bandwidth)
    exp = math.exp
    upper: list[array[float]] = []
    full: list[array[float]] = []
    sums: list[float] = []
    for a, xa in enumerate(values):
        row = [exp(scale * (xa - xb) * (xa - xb)) for xb in values]
        sums.append(math.fsum(row))
        upper.append(array("d", row[a + 1 :]))
        if keep_full:
            full.append(array("d", row))
    # The last strict upper row is empty; ``n - 1`` rows remain.
    return _Gram(tuple(upper[:-1]), tuple(sums), math.fsum(sums), tuple(full))


def _upper_inner(
    left: Sequence[array[float]], right: Sequence[Sequence[float]]
) -> float:
    """Return ``<K, L>`` from strict upper triangles (unit diagonals).

    References:
        [EQ-SENS-HSIC-001]. No new equation is introduced.

    """
    pairs = math.fsum(_sumprod(a, b) for a, b in zip(left, right, strict=True))
    return 2.0 * pairs + len(left) + 1


def _statistic(
    inner: float,
    k_sums: Sequence[float],
    k_total: float,
    l_sums: Sequence[float],
    l_total: float,
    estimator: str,
) -> float:
    """Return the biased or unbiased HSIC from Gram summaries.

    With unit diagonals, ``tr(K H L H) = <K, L> - (2/n) sum_a r_a s_a +
    R S / n^2`` (biased, divided by ``n^2``); the unbiased estimator uses the
    Gram matrices with their diagonals set to zero.

    References:
        [EQ-SENS-HSIC-001], [EQ-SENS-HSIC-002], [REF-SONG-ETAL-2012].

    """
    n = len(k_sums)
    cross = _sumprod(k_sums, l_sums)
    if estimator == "biased":
        return (inner - 2.0 * cross / n + k_total * l_total / (n * n)) / (n * n)
    trace = inner - n
    k_off = k_total - n
    l_off = l_total - n
    off_cross = cross - math.fsum(k_sums) - math.fsum(l_sums) + n
    return (trace + k_off * l_off / ((n - 1) * (n - 2)) - 2.0 * off_cross / (n - 2)) / (
        n * (n - 3)
    )


def _gamma_p_value(x: _Gram, y: _Gram, biased: float) -> float:
    """Return the Gamma-approximation p-value of ``n HSIC_b``.

    References:
        [EQ-SENS-HSIC-003], [REF-GRETTON-ETAL-2008].

    """
    n = len(x.row_sums)
    kx = tuple(r / n for r in x.row_sums)
    ky = tuple(r / n for r in y.row_sums)
    gx = x.total / (n * n)
    gy = y.total / (n * n)
    squares = 0.0
    for a in range(n - 1):
        ka, la = kx[a], ky[a]
        squares += math.fsum(
            ((value - ka - kb + gx) * (other - la - lb + gy)) ** 2
            for value, other, kb, lb in zip(
                x.upper[a], y.upper[a], kx[a + 1 :], ky[a + 1 :], strict=True
            )
        )
    variance = (
        2.0
        * (n - 4)
        * (n - 5)
        / (n * (n - 1) * (n - 2) * (n - 3))
        * (2.0 * squares)
        / (n * (n - 1))
    )
    mean_x = (x.total - n) / (n * (n - 1))
    mean_y = (y.total - n) / (n * (n - 1))
    mean = (1.0 + mean_x * mean_y - mean_x - mean_y) / n
    if not (variance > 0.0 and mean > 0.0):
        return math.nan
    shape = mean * mean / variance
    scale = variance * n / mean
    return Gamma.from_scale(shape, scale).sf(n * biased)


def _target_values(
    values: Sequence[float], target: float | Callable[[float], float] | None
) -> tuple[tuple[float, ...], str | None, float]:
    """Apply the target transform ``w(Y)`` to the output.

    Returns:
        Transformed values, a description, and the fraction with ``w > 0``.

    Raises:
        ValueError: If a transformed value is not finite.

    References:
        [EQ-SENS-HSIC-005], [REF-MARREL-CHABRIDON-2021].

    """
    if target is None:
        return tuple(values), None, math.nan
    if callable(target):
        transformed = tuple(
            require_finite_real(target(v), name="target(output)") for v in values
        )
        description = "w(Y) = " + getattr(target, "__name__", "callable")
    else:
        threshold = require_finite_real(target, name="target")
        transformed = tuple(1.0 if v > threshold else 0.0 for v in values)
        description = f"1{{Y > {threshold:g}}}"
    fraction = sum(1 for v in transformed if v > 0.0) / len(transformed)
    return transformed, description, fraction


def hsic_indices(
    inputs: Sequence[Sequence[float]],
    output: Sequence[float],
    *,
    names: Sequence[str] | None = None,
    kernel: str = "gaussian",
    bandwidth: str | Sequence[float] = "std",
    estimator: str = "biased",
    test: str = "permutation",
    permutations: int = 200,
    seed: int | RandomStream | None = None,
    target: float | Callable[[float], float] | None = None,
) -> HSICResult:
    """Estimate HSIC and ``R2_HSIC`` of each input with independence tests.

    For every input the Gaussian Gram matrix ``K`` is compared with the
    output Gram matrix ``L``: the biased estimate is ``tr(KHLH) / n^2``, the
    unbiased one the U-statistic of Song et al. (2012). ``R2_HSIC`` divides
    by ``sqrt(HSIC(X, X) HSIC(Y, Y))`` computed with the same estimator.
    The permutation test recomputes the chosen estimator for ``B`` random
    permutations of the output rows (shared by all inputs) and reports
    ``(1 + #{HSIC_b >= HSIC}) / (B + 1)``. The Gamma test always uses the
    biased statistic ``n HSIC_b``: the approximation of Gretton et al.
    (2008) is fitted to its null mean and variance, so under
    ``estimator="unbiased"`` it still describes ``HSIC_b`` (the unbiased
    estimate is centred at zero under independence and has no Gamma law).

    Args:
        inputs: ``n`` rows of ``d`` input values (i.i.d. rows).
        output: ``n`` output values.
        names: Input names (default ``x0, x1, ...``).
        kernel: ``"gaussian"`` (the only characteristic kernel offered).
        bandwidth: ``"std"`` (sample standard deviation, default),
            ``"median"`` (median pairwise distance), or ``d + 1`` positive
            values: one per input, the last for the (transformed) output.
        estimator: ``"biased"`` (V-statistic) or ``"unbiased"``
            (U-statistic).
        test: ``"permutation"`` or ``"gamma"``.
        permutations: Number ``B >= 1`` of output permutations.
        seed: Seed or stream of the permutations.
        target: ``None`` (raw output), a threshold ``c`` (indicator
            ``1{Y > c}``), or a callable weight ``w(Y)``.

    Returns:
        HSIC values, ``R2_HSIC``, and p-values per input.

    Raises:
        ValueError: For ragged or non-finite data, ``n < 6``, an unknown
            kernel, estimator, test, or bandwidth rule, a non-positive
            bandwidth, a constant (transformed) output, or ``permutations <
            1``.

    Equation:
        [EQ-SENS-HSIC-001], [EQ-SENS-HSIC-002], [EQ-SENS-HSIC-003],
        [EQ-SENS-HSIC-004], [EQ-SENS-HSIC-005].

    References:
        [REF-GRETTON-ETAL-2005], [REF-GRETTON-ETAL-2008],
        [REF-SONG-ETAL-2012], [REF-DA-VEIGA-2015],
        [REF-MARREL-CHABRIDON-2021], [REF-PHIPSON-SMYTH-2010].

    """
    if kernel != "gaussian":
        raise ValueError("kernel must be 'gaussian'")
    if estimator not in ESTIMATORS:
        raise ValueError(f"estimator must be one of {ESTIMATORS}")
    if test not in TESTS:
        raise ValueError(f"test must be one of {TESTS}")
    if test == "permutation" and (
        isinstance(permutations, bool)
        or not isinstance(permutations, int)
        or permutations < 1
    ):
        raise ValueError("permutations must be a positive integer")
    columns, raw_y, labels = _validated_sample(inputs, output, names)
    n, dimension = len(raw_y), len(columns)
    if n < 6:
        raise ValueError("HSIC needs at least six rows")
    y, description, fraction = _target_values(raw_y, target)
    if isinstance(bandwidth, str):
        if bandwidth not in BANDWIDTHS:
            raise ValueError(f"bandwidth must be one of {BANDWIDTHS} or values")
        rules: list[str | float] = [bandwidth] * (dimension + 1)
    else:
        rules = list(bandwidth)
        if len(rules) != dimension + 1:
            raise ValueError("bandwidth values must number d + 1 (inputs, output)")
    if min(y) == max(y):
        raise ValueError("the (transformed) output is constant; HSIC is zero")
    widths = tuple(
        _bandwidth(column, rule) for column, rule in zip(columns, rules, strict=False)
    )
    output_width = _bandwidth(y, rules[-1])
    gram_y = _gram(y, output_width, keep_full=test == "permutation")
    self_y = _statistic(
        _upper_inner(gram_y.upper, gram_y.upper),
        gram_y.row_sums,
        gram_y.total,
        gram_y.row_sums,
        gram_y.total,
        estimator,
    )
    grams: list[_Gram] = []
    values: list[float] = []
    normalized: list[float] = []
    p_values: list[float] = []
    for column, width in zip(columns, widths, strict=True):
        gram_x = _gram(column, width, keep_full=False)
        inner = _upper_inner(gram_x.upper, gram_y.upper)
        value = _statistic(
            inner,
            gram_x.row_sums,
            gram_x.total,
            gram_y.row_sums,
            gram_y.total,
            estimator,
        )
        self_x = _statistic(
            _upper_inner(gram_x.upper, gram_x.upper),
            gram_x.row_sums,
            gram_x.total,
            gram_x.row_sums,
            gram_x.total,
            estimator,
        )
        denominator = self_x * self_y
        normalized.append(
            value / math.sqrt(denominator) if denominator > 0.0 else math.nan
        )
        values.append(value)
        if test == "gamma":
            biased = (
                value
                if estimator == "biased"
                else _statistic(
                    inner,
                    gram_x.row_sums,
                    gram_x.total,
                    gram_y.row_sums,
                    gram_y.total,
                    "biased",
                )
            )
            p_values.append(_gamma_p_value(gram_x, gram_y, biased))
        else:
            grams.append(gram_x)
    if test == "permutation":
        p_values = _permutation_p_values(
            grams, gram_y, values, estimator, permutations, seed
        )
    return HSICResult(
        input_names=labels,
        hsic=tuple(values),
        r2_hsic=tuple(normalized),
        p_values=tuple(p_values),
        estimator=estimator,
        test=test,
        permutations=permutations if test == "permutation" else 0,
        bandwidths=widths,
        output_bandwidth=output_width,
        sample_size=n,
        target=description,
        target_fraction=fraction,
    )


def _permutation_p_values(
    grams: Sequence[_Gram],
    gram_y: _Gram,
    observed: Sequence[float],
    estimator: str,
    permutations: int,
    seed: int | RandomStream | None,
) -> list[float]:
    """Return Monte Carlo permutation p-values that share output shuffles.

    References:
        [EQ-SENS-HSIC-004], [REF-GRETTON-ETAL-2008], [REF-PHIPSON-SMYTH-2010].

    """
    n = len(gram_y.row_sums)
    generator = resolve_stream(seed).child("hsic-permutation").generator()
    exceed = [0] * len(grams)
    order = list(range(n))
    for _ in range(permutations):
        generator.shuffle(order)
        permuted_rows: list[Sequence[float]] = []
        for a in range(n - 1):
            source = gram_y.full[order[a]]
            tail = order[a + 1 :]
            if len(tail) > 1:
                permuted_rows.append(itemgetter(*tail)(source))
            else:
                permuted_rows.append((source[tail[0]],))
        sums = tuple(gram_y.row_sums[index] for index in order)
        for index, gram_x in enumerate(grams):
            inner = _upper_inner(gram_x.upper, permuted_rows)
            value = _statistic(
                inner, gram_x.row_sums, gram_x.total, sums, gram_y.total, estimator
            )
            if value >= observed[index]:
                exceed[index] += 1
    return [(1 + count) / (permutations + 1) for count in exceed]


def model_hsic_indices(
    model: ProbabilisticModel,
    quantity: str,
    size: int,
    *,
    seed: int | RandomStream | None,
    design: str = "mc",
    bandwidth: str | Sequence[float] = "std",
    estimator: str = "biased",
    test: str = "permutation",
    permutations: int = 200,
    target: float | Callable[[float], float] | None = None,
) -> HSICResult:
    """Sample a model, run it, and compute HSIC indices of one QoI.

    Args:
        model: Probabilistic model (dependent inputs are allowed; HSIC then
            measures the total dependence, including what an input carries
            about correlated inputs).
        quantity: QoI name.
        size: Number of model runs.
        seed: Root seed or stream (design and permutations use separate
            child streams).
        design: ``"mc"`` (i.i.d. rows, default), ``"lhs"``, ``"halton"``, or
            ``"sobol"``; tests assume i.i.d. rows.
        bandwidth: See :func:`hsic_indices`.
        estimator: ``"biased"`` or ``"unbiased"``.
        test: ``"permutation"`` or ``"gamma"``.
        permutations: Number of output permutations.
        target: Threshold or weight function of the output.

    Returns:
        :class:`HSICResult` with ``evaluations`` set.

    Raises:
        ValueError: For invalid settings or a degenerate sample.
        KeyError: For an unknown QoI.

    Equation:
        [EQ-SENS-HSIC-001], [EQ-SENS-HSIC-002], [EQ-SENS-HSIC-003],
        [EQ-SENS-HSIC-004], [EQ-SENS-HSIC-005].

    References:
        [REF-DA-VEIGA-2015], [REF-GRETTON-ETAL-2008],
        [REF-MARREL-CHABRIDON-2021].

    """
    stream = resolve_stream(seed)
    rows, values, evaluations = model_sample(
        model, quantity, size, seed=stream, design=design, label="hsic"
    )
    result = hsic_indices(
        rows,
        values,
        names=model.inputs.names,
        bandwidth=bandwidth,
        estimator=estimator,
        test=test,
        permutations=permutations,
        seed=stream.child("hsic-test"),
        target=target,
    )
    return replace(result, evaluations=evaluations)
