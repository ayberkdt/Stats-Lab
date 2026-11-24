"""Fourier amplitude sensitivity testing: extended FAST and random balance design.

Both estimators evaluate the model along a *search curve* in the unit
hypercube, ``u_k(s) = 1/2 + arcsin(sin(omega_k s + phi_k)) / pi``, which is
uniform on ``(0, 1)`` for every ``s`` drawn uniformly on ``(-pi, pi)``; each
coordinate is then mapped to its marginal law through the quantile function.
The output along the curve is analysed with a discrete Fourier transform: the
power at the harmonics of the frequency given to input ``i`` is the part of
the variance explained by ``i`` alone. Inputs must be independent.

Extended FAST (:func:`fast_analysis`, :func:`fast_indices`) gives first-order
and total-effect indices. For each input ``i`` one curve gives ``i`` a high
frequency ``omega_max`` and the other inputs a set of low frequencies
``<= omega_max / (2 M)``. The power at the harmonics ``p omega_max``
(``p <= M``) estimates the first-order variance ``V_i``; the power at all
frequencies up to ``omega_max / 2`` estimates ``V_(-i)``, the variance of
everything that does not involve ``i``, so ``S_Ti = 1 - V_(-i) / V``. ``M``
is the *interference factor* (the number of harmonics kept). A curve needs
``N >= 2 M omega_max + 1`` points (the Nyquist rule) and the ``d - 1`` low
frequencies must be distinct and, for three or more inputs, two apart, so
``N >= 4 M^2 max(2 d - 3, 1) + 1``
(:func:`fast_sample_size`). The truncation error of the spectrum is set by
``M``, not by ``N``: a longer curve at fixed ``M`` leaves the estimate nearly
unchanged, so raise ``M`` for a response with sharp features. Random phase
shifts ``phi_k`` are redrawn for each of ``R`` resamples; the spread of the
``R`` estimates is the reported standard error of the mean.

Random balance design FAST (:func:`rbd_fast_analysis`,
:func:`rbd_fast_indices`) gives first-order indices of all inputs from one
sample of ``N`` points: every input follows the same curve with frequency one
but in its own random order, and the output is re-sorted into the order of the
input of interest before the transform. The raw estimator is biased upwards
by ``(2 M / N)(1 - S_i)`` because the other inputs act as noise on the
spectrum; the correction ``(S - lambda) / (1 - lambda)`` with
``lambda = 2 M / N`` removes it (:func:`rbd_fast_analysis` reports both).

Neither method resolves interactions beyond the total index, and both
truncate the spectrum at ``M`` harmonics: a response with many sharp features
along an input needs a larger ``M`` and a larger ``N``. Compare with
:func:`~probstat.sensitivity.sobol_analysis` when the answer matters.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.foundations.validation import require_finite_real
from probstat.modeling import ProbabilisticModel

_UNIT_MARGIN = 2.0**-52


def _positive_int(value: object, name: str, minimum: int) -> int:
    """Return ``value`` as an integer of at least ``minimum``.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def fast_sample_size(dimension: int, harmonics: int = 4) -> int:
    """Return the smallest curve length of extended FAST.

    The Nyquist rule requires ``N >= 2 M omega_max + 1`` so that the highest
    harmonic analysed, ``M omega_max``, stays below ``N / 2``. The ``d - 1``
    complementary frequencies must be distinct integers no larger than
    ``omega_max / (2 M)`` (inputs that share a frequency would move along one
    line instead of filling the plane), so ``omega_max >= 2 M (d - 1)`` and
    ``N >= 4 M^2 (d - 1) + 1``. For ``d = 1`` the bound is ``4 M^2 + 1``.

    Args:
        dimension: Number of inputs ``d``.
        harmonics: Interference factor ``M`` (harmonics kept).

    Returns:
        The smallest admissible ``N`` (odd).

    Raises:
        ValueError: If an argument is below one.

    Equation:
        [EQ-SENS-FAST-003].

    References:
        [REF-SALTELLI-TARANTOLA-CHAN-1999], [REF-CUKIER-1973].

    """
    d = _positive_int(dimension, "dimension", 1)
    m = _positive_int(harmonics, "harmonics", 1)
    return 4 * m * m * max(2 * d - 3, 1) + 1


def fast_frequencies(
    dimension: int, samples: int, harmonics: int = 4
) -> tuple[int, tuple[int, ...]]:
    """Return ``omega_max`` and the complementary frequencies of a curve.

    ``omega_max = (N - 1) // (2 M)``. The ``d - 1`` other inputs receive
    integers in ``1 .. m`` with ``m = omega_max // (2 M)``, equally spaced
    (rounded down) from ``1`` to ``m``; for ``d >= 3`` neighbours are at
    least two apart. Every harmonic of a
    complementary frequency up to order ``M`` then lies at or below
    ``omega_max / 2``, below the first harmonic of ``omega_max``.

    Args:
        dimension: Number of inputs ``d``.
        samples: Curve length ``N`` (odd).
        harmonics: Interference factor ``M``.

    Returns:
        ``(omega_max, complementary frequencies)``.

    Raises:
        ValueError: If ``N`` is even or below :func:`fast_sample_size`.

    Equation:
        [EQ-SENS-FAST-003].

    References:
        [REF-SALTELLI-TARANTOLA-CHAN-1999], [REF-CUKIER-1973].

    """
    d = _positive_int(dimension, "dimension", 1)
    n = _positive_int(samples, "samples", 1)
    m = _positive_int(harmonics, "harmonics", 1)
    if n % 2 == 0:
        raise ValueError("samples must be odd")
    needed = fast_sample_size(d, m)
    if n < needed:
        raise ValueError(
            f"samples must be at least 4 M^2 max(2 d - 3, 1) + 1 = {needed} for "
            f"d = {d} inputs and harmonics M = {m}"
        )
    omega = (n - 1) // (2 * m)
    top = omega // (2 * m)
    if d == 1:
        return omega, ()
    if d == 2:
        return omega, (1,)
    return omega, tuple(1 + (k * (top - 1)) // (d - 2) for k in range(d - 1))


def _triangle(phase: float) -> float:
    """Map a phase to ``1/2 + arcsin(sin(phase)) / pi`` inside ``(0, 1)``.

    References:
        [EQ-SENS-FAST-001], [REF-SALTELLI-TARANTOLA-CHAN-1999].

    """
    u = 0.5 + math.asin(math.sin(phase)) / math.pi
    return min(max(u, _UNIT_MARGIN), 1.0 - _UNIT_MARGIN)


def _tables(count: int) -> tuple[list[float], list[float]]:
    """Return ``cos`` and ``sin`` of ``2 pi m / count`` for ``m = 0..count-1``.

    References:
        [EQ-SENS-FAST-002]. No new equation is introduced.

    """
    return (
        [math.cos(2.0 * math.pi * m / count) for m in range(count)],
        [math.sin(2.0 * math.pi * m / count) for m in range(count)],
    )


def _power(
    values: Sequence[float],
    harmonic: int,
    tables: tuple[list[float], list[float]],
) -> float:
    """Return ``2 |c_k|^2`` of the centred grid signal at harmonic ``k``.

    ``c_k = (1/N) sum_j y_j exp(-i k s_j)`` with ``s_j = 2 pi (j - h) / N``,
    ``h = (N - 1) / 2``; the grid offset only rotates the phase of ``c_k``.

    References:
        [EQ-SENS-FAST-002], [REF-CUKIER-1973].

    """
    n = len(values)
    half = (n - 1) // 2
    cos_t, sin_t = tables
    a = math.fsum(v * cos_t[(harmonic * (j - half)) % n] for j, v in enumerate(values))
    b = math.fsum(v * sin_t[(harmonic * (j - half)) % n] for j, v in enumerate(values))
    return 2.0 * (a * a + b * b) / (n * n)


def _spread(values: Sequence[float]) -> float:
    """Return the standard error ``s / sqrt(R)`` of the mean of ``R`` values.

    References:
        [REF-SALTELLI-TARANTOLA-CHAN-1999]. Elementary result.

    """
    count = len(values)
    mean = math.fsum(values) / count
    variance = math.fsum((v - mean) ** 2 for v in values) / (count - 1)
    return math.sqrt(variance / count)


@dataclass(frozen=True, slots=True)
class FastResult:
    """Extended-FAST first-order and total-effect indices.

    Args:
        input_names: Input names in order.
        first_order: Mean over the resamples of ``S_i = V_i / V``.
        total_effect: Mean over the resamples of ``S_Ti = 1 - V_(-i) / V``.
        first_order_error: Standard error of the mean of ``first_order``
            (phase-resampling spread).
        total_effect_error: Standard error of the mean of ``total_effect``.
        first_order_resamples: ``resamples x d`` first-order estimates.
        total_effect_resamples: ``resamples x d`` total-effect estimates.
        samples: Curve length ``N``.
        harmonics: Interference factor ``M``.
        max_frequency: High frequency ``omega_max`` of the input of interest.
        complementary_frequencies: Frequencies of the other ``d - 1`` inputs.
        resamples: Number of phase resamples ``R``.
        evaluations: Function evaluations, ``d R N``.
        variance: Mean output variance along the curves.
        quantity: QoI name, or ``None`` for a plain function.

    References:
        [REF-SALTELLI-TARANTOLA-CHAN-1999], [REF-CUKIER-1973].

    """

    input_names: tuple[str, ...]
    first_order: tuple[float, ...]
    total_effect: tuple[float, ...]
    first_order_error: tuple[float, ...]
    total_effect_error: tuple[float, ...]
    first_order_resamples: tuple[tuple[float, ...], ...]
    total_effect_resamples: tuple[tuple[float, ...], ...]
    samples: int
    harmonics: int
    max_frequency: int
    complementary_frequencies: tuple[int, ...]
    resamples: int
    evaluations: int
    variance: float
    quantity: str | None = None

    def as_mapping(self) -> Mapping[str, tuple[float, float]]:
        """Return ``input -> (first-order, total-effect)`` estimates.

        References:
            [EQ-SENS-FAST-002], [REF-SALTELLI-TARANTOLA-CHAN-1999].

        """
        return MappingProxyType(
            {
                name: (first, total)
                for name, first, total in zip(
                    self.input_names, self.first_order, self.total_effect, strict=True
                )
            }
        )

    def ranking(self) -> tuple[str, ...]:
        """Return input names in descending total-effect order.

        References:
            [EQ-SENS-FAST-002], [REF-SALTELLI-TARANTOLA-CHAN-1999].

        """
        order = sorted(
            range(len(self.total_effect)), key=lambda i: -self.total_effect[i]
        )
        return tuple(self.input_names[i] for i in order)


@dataclass(frozen=True, slots=True)
class RBDFastResult:
    """Random-balance-design first-order indices.

    Args:
        input_names: Input names in order.
        first_order: Reported estimate: the bias-corrected one when
            ``bias_correction`` is true, the raw one otherwise (mean over the
            replicates).
        first_order_error: Standard error of the mean over the replicates.
        raw: Mean of the uncorrected estimates ``V_i / V``.
        replicates: ``replicates x d`` reported estimates.
        samples: Sample size ``N`` per replicate.
        harmonics: Number of harmonics ``M``.
        bias_correction: Whether the Tissot-Prieur correction was applied.
        bias: ``lambda = 2 M / N``, the factor of the positive bias
            ``lambda (1 - S_i)`` of the raw estimator.
        evaluations: Function evaluations, ``R N``.
        variance: Mean output variance of the samples.
        quantity: QoI name, or ``None`` for a plain function.

    References:
        [REF-TARANTOLA-GATELLI-MARA-2006], [REF-TISSOT-PRIEUR-2012].

    """

    input_names: tuple[str, ...]
    first_order: tuple[float, ...]
    first_order_error: tuple[float, ...]
    raw: tuple[float, ...]
    replicates: tuple[tuple[float, ...], ...]
    samples: int
    harmonics: int
    bias_correction: bool
    bias: float
    evaluations: int
    variance: float
    quantity: str | None = None

    def as_mapping(self) -> Mapping[str, float]:
        """Return ``input -> first-order index``.

        References:
            [EQ-SENS-RBD-001], [REF-TARANTOLA-GATELLI-MARA-2006].

        """
        return MappingProxyType(
            dict(zip(self.input_names, self.first_order, strict=True))
        )

    def ranking(self) -> tuple[str, ...]:
        """Return input names in descending first-order order.

        References:
            [EQ-SENS-RBD-001], [REF-TARANTOLA-GATELLI-MARA-2006].

        """
        order = sorted(range(len(self.first_order)), key=lambda i: -self.first_order[i])
        return tuple(self.input_names[i] for i in order)


UnitFunction = Callable[[tuple[float, ...]], float]


def _checked(function: UnitFunction) -> UnitFunction:
    """Wrap a unit-cube function so that a non-finite output is rejected.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """

    def wrapped(point: tuple[float, ...]) -> float:
        """Evaluate and validate one output.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION]. No equation
            is evaluated.

        """
        return require_finite_real(function(point), name="the model output")

    return wrapped


def _extended_fast(
    function: UnitFunction,
    dimension: int,
    samples: int,
    harmonics: int,
    resamples: int,
    root: RandomStream,
) -> tuple[list[list[float]], list[list[float]], float]:
    """Run the curves of extended FAST on the unit cube.

    Returns:
        First-order and total-effect estimates (``resamples x d``) and the
        mean output variance.

    Raises:
        ValueError: If the output is constant along a curve.

    Equation:
        [EQ-SENS-FAST-001], [EQ-SENS-FAST-002].

    References:
        [REF-SALTELLI-TARANTOLA-CHAN-1999], [REF-CUKIER-1973].

    """
    omega, others = fast_frequencies(dimension, samples, harmonics)
    tables = _tables(samples)
    half = (samples - 1) // 2
    grid = [2.0 * math.pi * (j - half) / samples for j in range(samples)]
    first: list[list[float]] = []
    total: list[list[float]] = []
    variances: list[float] = []
    for r in range(resamples):
        first_row: list[float] = []
        total_row: list[float] = []
        for i in range(dimension):
            generator = root.child("curve", r, i).generator()
            phases = [2.0 * math.pi * generator.random() for _ in range(dimension)]
            freqs = list(others[:i]) + [omega] + list(others[i:])
            values = [
                function(
                    tuple(
                        _triangle(w * s + p) for w, p in zip(freqs, phases, strict=True)
                    )
                )
                for s in grid
            ]
            centre = math.fsum(values) / samples
            centred = [v - centre for v in values]
            variance = math.fsum(v * v for v in centred) / samples
            if variance <= 0.0:
                raise ValueError("the output is constant along the search curve")
            v_first = math.fsum(
                _power(centred, p * omega, tables) for p in range(1, harmonics + 1)
            )
            v_rest = math.fsum(
                _power(centred, k, tables) for k in range(1, omega // 2 + 1)
            )
            first_row.append(v_first / variance)
            total_row.append(1.0 - v_rest / variance)
            variances.append(variance)
        first.append(first_row)
        total.append(total_row)
    return first, total, math.fsum(variances) / len(variances)


def _fast_result(
    names: tuple[str, ...],
    function: UnitFunction,
    samples: int,
    harmonics: int,
    resamples: int,
    seed: int | RandomStream | None,
    quantity: str | None,
) -> FastResult:
    """Validate settings, run extended FAST, and package the result.

    References:
        [REF-SALTELLI-TARANTOLA-CHAN-1999]. Equations [EQ-SENS-FAST-001] to
        [EQ-SENS-FAST-003].

    """
    r = _positive_int(resamples, "resamples", 2)
    m = _positive_int(harmonics, "harmonics", 1)
    d = len(names)
    omega, others = fast_frequencies(d, samples, m)
    first, total, variance = _extended_fast(
        _checked(function), d, samples, m, r, resolve_stream(seed).child("fast")
    )
    columns = range(d)
    return FastResult(
        names,
        tuple(math.fsum(row[i] for row in first) / r for i in columns),
        tuple(math.fsum(row[i] for row in total) / r for i in columns),
        tuple(_spread([row[i] for row in first]) for i in columns),
        tuple(_spread([row[i] for row in total]) for i in columns),
        tuple(tuple(row) for row in first),
        tuple(tuple(row) for row in total),
        samples,
        m,
        omega,
        others,
        r,
        d * r * samples,
        variance,
        quantity,
    )


def _marginals(
    marginals: Sequence[Any] | Mapping[str, Any], names: Sequence[str] | None
) -> tuple[tuple[str, ...], tuple[Any, ...]]:
    """Return input names and marginal laws from a sequence or a mapping.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    if isinstance(marginals, Mapping):
        labels = tuple(marginals)
        laws = tuple(marginals.values())
        if names is not None and tuple(names) != labels:
            raise ValueError("names must match the keys of the marginals mapping")
    else:
        laws = tuple(marginals)
        labels = (
            tuple(f"x{i + 1}" for i in range(len(laws)))
            if names is None
            else tuple(names)
        )
    if not laws:
        raise ValueError("at least one marginal is required")
    if len(labels) != len(laws) or len(set(labels)) != len(labels):
        raise ValueError("names must be distinct and match the number of marginals")
    for law in laws:
        if not callable(getattr(law, "quantile", None)):
            raise TypeError("every marginal needs a quantile method")
    return labels, laws


def _through_quantiles(
    function: Callable[[tuple[float, ...]], float], laws: Sequence[Any]
) -> UnitFunction:
    """Return ``u -> f(F_1^-1(u_1), ..., F_d^-1(u_d))``.

    References:
        [EQ-SAMP-INV-001], [REF-DEVROYE-1986, Section II.2].

    """

    def composed(point: tuple[float, ...]) -> float:
        """Evaluate the function at the quantiles of a unit-cube point.

        References:
            [EQ-SAMP-INV-001], [REF-DEVROYE-1986, Section II.2].

        """
        return function(
            tuple(float(law.quantile(u)) for law, u in zip(laws, point, strict=True))
        )

    return composed


def fast_indices(
    function: Callable[[tuple[float, ...]], float],
    marginals: Sequence[Any] | Mapping[str, Any],
    samples: int,
    *,
    harmonics: int = 4,
    resamples: int = 5,
    seed: int | RandomStream | None = None,
    names: Sequence[str] | None = None,
) -> FastResult:
    """Estimate first-order and total indices of a plain function by extended FAST.

    The inputs are independent with the given marginal laws; the search curve
    is mapped to each marginal through its quantile function. Cost:
    ``d R N`` calls.

    Args:
        function: Scalar function of a tuple of ``d`` physical inputs.
        marginals: Laws with a ``quantile`` method, in input order, or a
            mapping ``name -> law``.
        samples: Odd curve length ``N`` of at least ``fast_sample_size(d, M)``;
            the frequency of the input of interest is ``(N - 1) // (2 M)``.
        harmonics: Interference factor ``M``.
        resamples: Number ``R >= 2`` of random phase resamples.
        seed: Root seed or stream.
        names: Input names (default ``x1, x2, ...``).

    Returns:
        Indices with phase-resampling standard errors.

    Raises:
        TypeError: If ``function`` is not callable, a marginal has no
            ``quantile``, or an integer setting has the wrong type.
        ValueError: If settings violate the sample-size rule, names are
            invalid, or the output is constant or not finite.

    Equation:
        [EQ-SENS-FAST-001], [EQ-SENS-FAST-002], [EQ-SENS-FAST-003].

    References:
        [REF-SALTELLI-TARANTOLA-CHAN-1999], [REF-CUKIER-1973],
        [REF-SALTELLI-2008, Chapter 5].

    """
    if not callable(function):
        raise TypeError("function must be callable")
    labels, laws = _marginals(marginals, names)
    return _fast_result(
        labels,
        _through_quantiles(function, laws),
        samples,
        harmonics,
        resamples,
        seed,
        None,
    )


def fast_analysis(
    model: ProbabilisticModel,
    quantity: str,
    samples: int,
    *,
    harmonics: int = 4,
    resamples: int = 5,
    seed: int | RandomStream | None = None,
) -> FastResult:
    """Estimate first-order and total indices of a model QoI by extended FAST.

    Cost: ``d R N`` model runs; the curve is mapped to the inputs through
    their marginal quantile functions. Prefer more samples ``N`` (a larger
    ``omega_max``) over more resamples when the standard errors are large
    compared with the truncation of the spectrum at ``M`` harmonics.

    Args:
        model: Probabilistic model with mutually independent inputs.
        quantity: QoI name.
        samples: Odd curve length ``N`` of at least ``fast_sample_size(d, M)``.
        harmonics: Interference factor ``M``.
        resamples: Number ``R >= 2`` of random phase resamples.
        seed: Root seed or stream.

    Returns:
        Indices with phase-resampling standard errors.

    Raises:
        KeyError: If the quantity is unknown.
        ValueError: If the inputs are dependent (a non-identity copula),
            settings violate the sample-size rule, or the output is constant.

    Equation:
        [EQ-SENS-FAST-001], [EQ-SENS-FAST-002], [EQ-SENS-FAST-003].

    References:
        [REF-SALTELLI-TARANTOLA-CHAN-1999], [REF-CUKIER-1973],
        [REF-SALTELLI-2008, Chapter 5].

    """
    model.quantity(quantity)
    if not model.inputs.is_independent:
        raise ValueError(
            "FAST requires independent inputs; the model's joint law has a "
            "non-identity copula. Use shapley_effects for dependent inputs"
        )
    before = model.evaluations
    result = _fast_result(
        model.inputs.names,
        model.unit_function(quantity),
        samples,
        harmonics,
        resamples,
        seed,
        quantity,
    )
    spent = model.evaluations - before
    return FastResult(
        result.input_names,
        result.first_order,
        result.total_effect,
        result.first_order_error,
        result.total_effect_error,
        result.first_order_resamples,
        result.total_effect_resamples,
        result.samples,
        result.harmonics,
        result.max_frequency,
        result.complementary_frequencies,
        result.resamples,
        spent,
        result.variance,
        quantity,
    )


def _rbd(
    function: UnitFunction,
    dimension: int,
    samples: int,
    harmonics: int,
    correction: bool,
    root: RandomStream,
) -> tuple[list[float], list[float], float]:
    """Run one random balance replicate on the unit cube.

    Returns:
        Reported and raw first-order estimates per input and the output
        variance.

    Equation:
        [EQ-SENS-RBD-001], [EQ-SENS-RBD-002].

    References:
        [REF-TARANTOLA-GATELLI-MARA-2006], [REF-TISSOT-PRIEUR-2012].

    """
    half = (samples - 1) // 2
    base = [_triangle(2.0 * math.pi * (j - half) / samples) for j in range(samples)]
    generator = root.generator()
    orders = [generator.sample(range(samples), samples) for _ in range(dimension)]
    values = [
        function(tuple(base[orders[i][k]] for i in range(dimension)))
        for k in range(samples)
    ]
    centre = math.fsum(values) / samples
    centred = [v - centre for v in values]
    variance = math.fsum(v * v for v in centred) / samples
    if variance <= 0.0:
        raise ValueError("the output is constant over the sample")
    tables = _tables(samples)
    lam = 2.0 * harmonics / samples
    raw: list[float] = []
    reported: list[float] = []
    for i in range(dimension):
        regrouped = [0.0] * samples
        for k in range(samples):
            regrouped[orders[i][k]] = centred[k]
        estimate = (
            math.fsum(_power(regrouped, m, tables) for m in range(1, harmonics + 1))
            / variance
        )
        raw.append(estimate)
        reported.append((estimate - lam) / (1.0 - lam) if correction else estimate)
    return reported, raw, variance


def _rbd_result(
    names: tuple[str, ...],
    function: UnitFunction,
    samples: int,
    harmonics: int,
    replicates: int,
    bias_correction: bool,
    seed: int | RandomStream | None,
    quantity: str | None,
) -> RBDFastResult:
    """Validate settings, run the replicates, and package the result.

    References:
        [REF-TARANTOLA-GATELLI-MARA-2006], [REF-TISSOT-PRIEUR-2012]. Equations
        [EQ-SENS-RBD-001], [EQ-SENS-RBD-002].

    """
    n = _positive_int(samples, "samples", 3)
    if n % 2 == 0:
        raise ValueError("samples must be odd")
    m = _positive_int(harmonics, "harmonics", 1)
    if 2 * m >= n:
        raise ValueError("harmonics must satisfy 2 M < samples")
    r = _positive_int(replicates, "replicates", 2)
    if not isinstance(bias_correction, bool):
        raise TypeError("bias_correction must be a bool")
    d = len(names)
    root = resolve_stream(seed).child("rbd")
    checked = _checked(function)
    rows: list[list[float]] = []
    raws: list[list[float]] = []
    variances: list[float] = []
    for index in range(r):
        reported, raw, variance = _rbd(
            checked, d, n, m, bias_correction, root.child("replicate", index)
        )
        rows.append(reported)
        raws.append(raw)
        variances.append(variance)
    columns = range(d)
    return RBDFastResult(
        names,
        tuple(math.fsum(row[i] for row in rows) / r for i in columns),
        tuple(_spread([row[i] for row in rows]) for i in columns),
        tuple(math.fsum(row[i] for row in raws) / r for i in columns),
        tuple(tuple(row) for row in rows),
        n,
        m,
        bias_correction,
        2.0 * m / n,
        r * n,
        math.fsum(variances) / r,
        quantity,
    )


def rbd_fast_indices(
    function: Callable[[tuple[float, ...]], float],
    marginals: Sequence[Any] | Mapping[str, Any],
    samples: int,
    *,
    harmonics: int = 6,
    replicates: int = 5,
    bias_correction: bool = True,
    seed: int | RandomStream | None = None,
    names: Sequence[str] | None = None,
) -> RBDFastResult:
    """Estimate first-order indices of a plain function by random balance design.

    One sample of ``N`` points serves all inputs; ``replicates`` independent
    samples give the spread. Cost: ``R N`` calls, independent of ``d``.

    Args:
        function: Scalar function of a tuple of ``d`` physical inputs.
        marginals: Laws with a ``quantile`` method, or a mapping ``name ->
            law``; the inputs are independent.
        samples: Odd sample size ``N`` with ``2 M < N``. The raw estimator
            is biased by ``(2 M / N)(1 - S_i)``; use ``N`` of several hundred.
        harmonics: Number ``M`` of harmonics kept.
        replicates: Number ``R >= 2`` of independent samples.
        bias_correction: Apply the Tissot-Prieur correction.
        seed: Root seed or stream.
        names: Input names (default ``x1, x2, ...``).

    Returns:
        Corrected (or raw) first-order indices with standard errors.

    Raises:
        TypeError: If ``function`` is not callable or a setting has the wrong
            type.
        ValueError: If settings are invalid or the output is constant.

    Equation:
        [EQ-SENS-RBD-001], [EQ-SENS-RBD-002].

    References:
        [REF-TARANTOLA-GATELLI-MARA-2006], [REF-TISSOT-PRIEUR-2012].

    """
    if not callable(function):
        raise TypeError("function must be callable")
    labels, laws = _marginals(marginals, names)
    return _rbd_result(
        labels,
        _through_quantiles(function, laws),
        samples,
        harmonics,
        replicates,
        bias_correction,
        seed,
        None,
    )


def rbd_fast_analysis(
    model: ProbabilisticModel,
    quantity: str,
    samples: int,
    *,
    harmonics: int = 6,
    replicates: int = 5,
    bias_correction: bool = True,
    seed: int | RandomStream | None = None,
) -> RBDFastResult:
    """Estimate first-order indices of a model QoI by random balance design.

    Cost: ``R N`` model runs, independent of the number of inputs.

    Args:
        model: Probabilistic model with mutually independent inputs.
        quantity: QoI name.
        samples: Odd sample size ``N`` with ``2 M < N``.
        harmonics: Number ``M`` of harmonics kept.
        replicates: Number ``R >= 2`` of independent samples.
        bias_correction: Apply the Tissot-Prieur correction.
        seed: Root seed or stream.

    Returns:
        Corrected (or raw) first-order indices with standard errors.

    Raises:
        KeyError: If the quantity is unknown.
        ValueError: If the inputs are dependent, settings are invalid, or
            the output is constant.

    Equation:
        [EQ-SENS-RBD-001], [EQ-SENS-RBD-002].

    References:
        [REF-TARANTOLA-GATELLI-MARA-2006], [REF-TISSOT-PRIEUR-2012].

    """
    model.quantity(quantity)
    if not model.inputs.is_independent:
        raise ValueError(
            "random balance design requires independent inputs; the model's "
            "joint law has a non-identity copula. Use shapley_effects instead"
        )
    before = model.evaluations
    result = _rbd_result(
        model.inputs.names,
        model.unit_function(quantity),
        samples,
        harmonics,
        replicates,
        bias_correction,
        seed,
        quantity,
    )
    return RBDFastResult(
        result.input_names,
        result.first_order,
        result.first_order_error,
        result.raw,
        result.replicates,
        result.samples,
        result.harmonics,
        result.bias_correction,
        result.bias,
        model.evaluations - before,
        result.variance,
        quantity,
    )
