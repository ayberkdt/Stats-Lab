"""Rank-based first-order Sobol indices and Chatterjee's rank correlation.

Both statistics need only one i.i.d. sample ``(x_k, y_k)``: no pick-freeze
design, no partition size, no kernel bandwidth. For input ``i`` the rows are
sorted by ``x_i``; the output read in that order, ``y_(1), ..., y_(n)``, pairs
every run with the run whose ``x_i`` is the next larger one. Because
neighbouring ``x_i`` values are close, ``y_(j)`` and ``y_(j+1)`` behave like
two outputs that share ``X_i`` and differ in all other inputs, which is the
pick-freeze pair of ``E[E[Y | X_i]^2]``. The estimator of Gamboa, Gremaud,
Klein and Lagnoux (2022) is therefore

``S_i = [(1/n) sum_j y_(j) y_(j+1) - ybar^2] / [(1/n) sum_j y_j^2 - ybar^2]``

with the cyclic convention ``y_(n+1) = y_(1)`` ([EQ-SENS-RANKSOBOL-001]). All
``d`` first-order indices come from the same ``n`` model runs.

Chatterjee's coefficient applies the same construction to the indicators
``1{Y >= t}`` and integrates over ``t``, which gives a function of the ranks
of the output only ([EQ-SENS-XI-001], [EQ-SENS-XI-002]). It estimates
``xi = int Var(E[1{Y >= t} | X]) dmu(t) / int Var(1{Y >= t}) dmu(t)`` (``mu``
the law of ``Y``), which is 0 if and only if ``X`` and ``Y`` are independent
and 1 if and only if ``Y`` is a measurable function of ``X``. Under
independence and a continuous ``Y``, ``sqrt(n) xi_n -> N(0, 2/5)``
([EQ-SENS-XI-003]), so a p-value needs no resampling.

Uncertainty. The ordinary bootstrap is not valid for these statistics: a
resample repeats rows, repeated rows are neighbours after sorting, and their
identical outputs inflate the lag-one product. This module therefore offers a
disjoint-block standard error ([EQ-SENS-RANKSOBOL-002]): the estimator is
recomputed on ``B`` disjoint blocks of the rows and the spread of the block
estimates is scaled by ``1 / sqrt(B)``. This relies on the ``sqrt(n)`` rate of
the estimator and is a diagnostic, not the asymptotic variance of the
published central limit theorem, which is not implemented.

Ties. The theory assumes continuous inputs. Tied values of ``x_i`` are put in
a uniformly random order (the convention of Chatterjee 2021), drawn from the
seed, and the affected inputs are listed in the result.
"""

from __future__ import annotations

import math
from bisect import bisect_left, bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.modeling import ProbabilisticModel

from .regression_based import _validated_sample, model_sample

XI_TESTS = ("asymptotic", "permutation")
_XI_NULL_VARIANCE = 0.4


@dataclass(frozen=True, slots=True)
class RankSobolResult:
    """Rank-based first-order Sobol indices from one input-output sample.

    Args:
        input_names: Input names.
        first_order: Estimates of ``S_i`` (not clipped to ``[0, 1]``).
        standard_errors: Disjoint-block standard errors of ``first_order``
            (``None`` when ``blocks == 0``).
        block_means: Mean of the block estimates per input (``None`` when
            ``blocks == 0``); a clear difference from ``first_order`` signals
            a bias at the block size, i.e. blocks that are too small.
        blocks: Number of disjoint blocks ``B`` (0: none).
        block_size: Rows per block.
        sample_size: Number of rows ``n``.
        output_variance: Population variance ``(1/n) sum (y - ybar)^2``.
        tied_inputs: Inputs with tied values whose order was randomized.
        evaluations: Model evaluations spent (zero for given data).

    References:
        [REF-GAMBOA-ETAL-2022].

    """

    input_names: tuple[str, ...]
    first_order: tuple[float, ...]
    standard_errors: tuple[float, ...] | None
    block_means: tuple[float, ...] | None
    blocks: int
    block_size: int
    sample_size: int
    output_variance: float
    tied_inputs: tuple[str, ...] = ()
    evaluations: int = 0

    def as_mapping(self) -> Mapping[str, float]:
        """Return ``input -> first-order index``.

        References:
            [EQ-SENS-RANKSOBOL-001], [REF-GAMBOA-ETAL-2022].

        """
        return dict(zip(self.input_names, self.first_order, strict=True))

    def ranking(self) -> tuple[str, ...]:
        """Return input names by decreasing first-order index.

        References:
            [EQ-SENS-RANKSOBOL-001], [REF-GAMBOA-ETAL-2022].

        """
        order = sorted(range(len(self.first_order)), key=lambda i: -self.first_order[i])
        return tuple(self.input_names[i] for i in order)


@dataclass(frozen=True, slots=True)
class ChatterjeeResult:
    """Chatterjee rank correlations of each input with the output.

    Args:
        input_names: Input names.
        xi: Coefficients ``xi_n(X_i, Y)``; slightly negative values occur
            under independence.
        p_values: One-sided p-values of ``H0: X_i independent of Y``
            (large ``xi`` rejects).
        test: ``"asymptotic"`` or ``"permutation"``.
        permutations: Number of permutations (0 for the asymptotic test).
        sample_size: Number of rows ``n``.
        output_ties: Whether the output has tied values (the tie-aware
            formula was used).
        tied_inputs: Inputs with tied values whose order was randomized.

    References:
        [REF-CHATTERJEE-2021].

    """

    input_names: tuple[str, ...]
    xi: tuple[float, ...]
    p_values: tuple[float, ...]
    test: str
    permutations: int
    sample_size: int
    output_ties: bool
    tied_inputs: tuple[str, ...] = ()

    def as_mapping(self) -> Mapping[str, float]:
        """Return ``input -> xi``.

        References:
            [EQ-SENS-XI-001], [REF-CHATTERJEE-2021].

        """
        return dict(zip(self.input_names, self.xi, strict=True))

    def dependent(self, level: float = 0.05) -> tuple[str, ...]:
        """Return the inputs whose independence from the output is rejected.

        Args:
            level: Significance level in ``(0, 1)`` applied to each input
                separately (no multiplicity correction).

        Returns:
            Names with ``p < level``.

        Raises:
            ValueError: If ``level`` is not in ``(0, 1)``.

        Equation:
            [EQ-SENS-XI-003].

        References:
            [REF-CHATTERJEE-2021].

        """
        if not 0.0 < level < 1.0:
            raise ValueError("level must lie in (0, 1)")
        return tuple(
            name
            for name, p in zip(self.input_names, self.p_values, strict=True)
            if p < level
        )


def _sorted_order(
    column: Sequence[float], stream: RandomStream, index: int
) -> tuple[list[int], bool]:
    """Return the row order by increasing value, ties in random order.

    Returns:
        The order and whether tied values were present.

    References:
        [REF-CHATTERJEE-2021] (random tie breaking of the conditioning
        variable). No equation is evaluated.

    """
    n = len(column)
    order = sorted(range(n), key=column.__getitem__)
    tied = any(column[order[k]] == column[order[k + 1]] for k in range(n - 1))
    if tied:
        generator = stream.child("rank-ties", index).generator()
        keys = [generator.random() for _ in range(n)]
        order = sorted(range(n), key=lambda k: (column[k], keys[k]))
    return order, tied


def _lag_one_ratio(centred: Sequence[float], order: Sequence[int]) -> float:
    """Return the cyclic lag-one autocorrelation of ``centred`` along ``order``.

    With ``c = y - ybar`` the cyclic sum satisfies
    ``(1/n) sum c_(j) c_(j+1) = (1/n) sum y_(j) y_(j+1) - ybar^2`` exactly,
    because every row occurs once as left and once as right neighbour.

    Raises:
        ValueError: If the values are constant.

    References:
        [EQ-SENS-RANKSOBOL-001], [REF-GAMBOA-ETAL-2022].

    """
    count = len(order)
    total = math.fsum(centred[k] ** 2 for k in order)
    if not total > 0.0:
        raise ValueError("the output is constant; indices are undefined")
    products = math.fsum(
        centred[order[j]] * centred[order[(j + 1) % count]] for j in range(count)
    )
    return products / total


def _rank_index(values: Sequence[float], order: Sequence[int]) -> float:
    """Return the rank-based index of the rows listed in ``order``.

    Raises:
        ValueError: If the selected outputs are constant.

    References:
        [EQ-SENS-RANKSOBOL-001], [REF-GAMBOA-ETAL-2022].

    """
    selected = [values[k] for k in order]
    mean = math.fsum(selected) / len(selected)
    return _lag_one_ratio([v - mean for v in selected], range(len(selected)))


def rank_sobol_indices(
    inputs: Sequence[Sequence[float]],
    outputs: Sequence[float],
    *,
    names: Sequence[str] | None = None,
    blocks: int = 0,
    seed: int | RandomStream | None = None,
) -> RankSobolResult:
    """Estimate all first-order Sobol indices from one sample by ranks.

    For input ``i`` the outputs are read in the order of increasing ``x_i``
    and ``S_i`` is the cyclic lag-one autocorrelation of that sequence. The
    estimator is consistent for ``Var(E[Y | X_i]) / Var(Y)`` for i.i.d. rows
    with a continuous ``X_i`` and ``E[Y^2] < inf``; no smoothness of the
    model is required. Dependent inputs are allowed and the index then has
    the meaning described in :mod:`probstat.sensitivity.given_data`.

    With ``blocks = B >= 2`` the rows are cut, in the order given, into ``B``
    disjoint blocks of ``n // B`` rows (a remainder is not used for the
    blocks); the estimator is recomputed on each block and
    ``se = sd(block estimates) / sqrt(B)`` is reported. The rows must then be
    in an order unrelated to their values (the order in which an i.i.d.
    sample was drawn is fine; shuffle a sorted table first). The standard
    error describes the mean of the block estimates and transfers to the
    full-sample estimate only through the ``sqrt(n)`` rate; it is a
    diagnostic with ``B - 1`` degrees of freedom, not a confidence
    statement. On the Ishigami function ``estimate +- 2 se`` covered the
    exact indices in 82 to 89 % of 600 samples at ``n = 100``, ``B = 5``,
    in 90 to 91 % at ``n = 500``, ``B = 10``, and in 92 to 93 % at
    ``n = 2000``, ``B = 10``.

    Args:
        inputs: ``n`` rows of ``d`` input values (i.i.d. rows).
        outputs: ``n`` output values.
        names: Input names (default ``x0, x1, ...``).
        blocks: Number of disjoint blocks for the standard error (0: none;
            otherwise between 2 and ``n // 8``).
        seed: Seed or stream used only to order tied input values.

    Returns:
        Indices, optional block standard errors, and tie diagnostics.

    Raises:
        ValueError: For ragged, mismatched or non-finite data, ``n < 8``, an
            invalid ``blocks``, or a constant output (also within a block).

    Equation:
        [EQ-SENS-RANKSOBOL-001], [EQ-SENS-RANKSOBOL-002].

    References:
        [REF-GAMBOA-ETAL-2022], [REF-CHATTERJEE-2021].

    """
    columns, y, labels = _validated_sample(inputs, outputs, names)
    n = len(y)
    if n < 8:
        raise ValueError("at least eight rows are required")
    if isinstance(blocks, bool) or not isinstance(blocks, int) or blocks < 0:
        raise ValueError("blocks must be a non-negative integer")
    if blocks == 1 or blocks > n // 8:
        raise ValueError("blocks must be 0 or lie in [2, n // 8]")
    stream = resolve_stream(seed) if seed is not None else RandomStream(0)
    mean = math.fsum(y) / n
    centred = tuple(v - mean for v in y)
    variance = math.fsum(c * c for c in centred) / n
    if not variance > 0.0:
        raise ValueError("the output is constant; indices are undefined")
    size = n // blocks if blocks else 0
    first: list[float] = []
    errors: list[float] = []
    means: list[float] = []
    tied: list[str] = []
    for i, column in enumerate(columns):
        order, has_ties = _sorted_order(column, stream, i)
        if has_ties:
            tied.append(labels[i])
        first.append(_lag_one_ratio(centred, order))
        if blocks:
            estimates = [
                _rank_index(y, [k for k in order if b * size <= k < (b + 1) * size])
                for b in range(blocks)
            ]
            centre = math.fsum(estimates) / blocks
            spread = math.fsum((e - centre) ** 2 for e in estimates) / (blocks - 1)
            means.append(centre)
            errors.append(math.sqrt(spread / blocks))
    return RankSobolResult(
        labels,
        tuple(first),
        tuple(errors) if blocks else None,
        tuple(means) if blocks else None,
        blocks,
        size,
        n,
        variance,
        tuple(tied),
    )


def model_rank_sobol_indices(
    model: ProbabilisticModel,
    quantity: str,
    size: int,
    *,
    seed: int | RandomStream | None,
    blocks: int = 0,
) -> RankSobolResult:
    """Sample a model by plain Monte Carlo and estimate rank-based indices.

    The cost is ``size`` model runs for all inputs together, against
    ``size (d + 2)`` for a pick-freeze design.

    Args:
        model: Probabilistic model (dependent inputs are allowed).
        quantity: QoI name.
        size: Number of model runs (at least eight).
        seed: Root seed or stream.
        blocks: Number of disjoint blocks for the standard error.

    Returns:
        :class:`RankSobolResult` with ``evaluations`` set.

    Raises:
        ValueError: For invalid settings or a constant output.
        KeyError: For an unknown QoI.

    Equation:
        [EQ-SENS-RANKSOBOL-001], [EQ-SENS-RANKSOBOL-002].

    References:
        [REF-GAMBOA-ETAL-2022].

    """
    stream = resolve_stream(seed)
    rows, values, evaluations = model_sample(
        model, quantity, size, seed=stream, design="mc", label="rank-sobol"
    )
    result = rank_sobol_indices(
        rows,
        values,
        names=model.inputs.names,
        blocks=blocks,
        seed=stream.child("rank-sobol-ties"),
    )
    return replace(result, evaluations=evaluations)


def _output_ranks(y: Sequence[float]) -> tuple[list[int], int, bool]:
    """Return ``r_k = #{j : y_j <= y_k}``, ``sum_k l_k (n - l_k)``, and ties.

    ``l_k = #{j : y_j >= y_k}``. Without ties ``l_k = n - r_k + 1`` and the
    sum is ``n (n^2 - 1) / 6``.

    References:
        [EQ-SENS-XI-002], [REF-CHATTERJEE-2021].

    """
    n = len(y)
    ordered = sorted(y)
    ranks = [bisect_right(ordered, v) for v in y]
    upper = [n - bisect_left(ordered, v) for v in y]
    scale = sum(count * (n - count) for count in upper)
    ties = any(ordered[k] == ordered[k + 1] for k in range(n - 1))
    return ranks, scale, ties


def _xi_from_ranks(ranks: Sequence[int], order: Sequence[int], scale: int) -> float:
    """Return ``1 - n sum_j |r_(j+1) - r_(j)| / (2 sum_k l_k (n - l_k))``.

    References:
        [EQ-SENS-XI-002], [REF-CHATTERJEE-2021].

    """
    n = len(order)
    jumps = sum(abs(ranks[order[j + 1]] - ranks[order[j]]) for j in range(n - 1))
    return 1.0 - (n * jumps) / (2 * scale)


def chatterjee_xi(
    x: Sequence[float],
    y: Sequence[float],
    *,
    seed: int | RandomStream | None = None,
) -> float:
    """Return Chatterjee's rank correlation ``xi_n(X, Y)``.

    The pairs are sorted by ``x``; with ``r_j`` the number of outputs not
    larger than the ``j``-th one in that order and ``l_j`` the number not
    smaller, ``xi_n = 1 - n sum_j |r_(j+1) - r_(j)| / (2 sum_j l_j (n - l_j))``.
    Without ties in ``y`` this is ``1 - 3 sum_j |r_(j+1) - r_(j)| / (n^2 - 1)``.
    The coefficient is not symmetric: ``xi_n(X, Y)`` measures how well ``Y``
    is a function of ``X``. For a strictly monotone relation without ties
    the value is exactly ``1 - 3 / (n + 1) = (n - 2) / (n + 1)``, the largest
    value the statistic can take.

    Args:
        x: ``n`` values of the conditioning variable.
        y: ``n`` values of the response.
        seed: Seed or stream used only to order tied ``x`` values.

    Returns:
        ``xi_n`` (between about ``-1/2`` and 1).

    Raises:
        ValueError: For mismatched or non-finite data, ``n < 3``, or a
            constant ``y``.

    Equation:
        [EQ-SENS-XI-001], [EQ-SENS-XI-002].

    References:
        [REF-CHATTERJEE-2021].

    """
    columns, values, _ = _validated_sample([(v,) for v in x], y, None)
    if len(values) < 3:
        raise ValueError("at least three pairs are required")
    ranks, scale, _ = _output_ranks(values)
    if scale == 0:
        raise ValueError("the output is constant; xi is undefined")
    stream = resolve_stream(seed) if seed is not None else RandomStream(0)
    order, _ = _sorted_order(columns[0], stream, 0)
    return _xi_from_ranks(ranks, order, scale)


def chatterjee_indices(
    inputs: Sequence[Sequence[float]],
    outputs: Sequence[float],
    *,
    names: Sequence[str] | None = None,
    test: str = "asymptotic",
    permutations: int = 999,
    seed: int | RandomStream | None = None,
) -> ChatterjeeResult:
    """Compute Chatterjee's ``xi`` of the output on each input with a test.

    ``test="asymptotic"`` uses the independence-null limit
    ``sqrt(n) xi_n -> N(0, 2/5)``, valid for a continuous output, and
    reports ``p = 1 - Phi(xi_n sqrt(5 n / 2))``; an output with ties is
    refused because the null variance then depends on the tie pattern.
    ``test="permutation"`` compares each ``xi_n`` with its values on ``B``
    random arrangements of the output ranks, which is exact in level for
    i.i.d. rows with or without ties in the output; one set of arrangements
    serves all inputs. The test detects any dependence but has low power
    against smooth, weak alternatives compared with a correlation test.

    Args:
        inputs: ``n`` rows of ``d`` input values (i.i.d. rows).
        outputs: ``n`` output values.
        names: Input names (default ``x0, x1, ...``).
        test: ``"asymptotic"`` or ``"permutation"``.
        permutations: Number ``B >= 1`` of random arrangements (permutation
            test only).
        seed: Seed or stream (tie order of the inputs and permutations).

    Returns:
        Coefficients and p-values per input.

    Raises:
        ValueError: For ragged or non-finite data, ``n < 8``, a constant
            output, an unknown test, the asymptotic test with tied outputs,
            or ``permutations < 1``.

    Equation:
        [EQ-SENS-XI-001], [EQ-SENS-XI-002], [EQ-SENS-XI-003],
        [EQ-SENS-HSIC-004].

    References:
        [REF-CHATTERJEE-2021], [REF-PHIPSON-SMYTH-2010].

    """
    columns, y, labels = _validated_sample(inputs, outputs, names)
    n = len(y)
    if n < 8:
        raise ValueError("at least eight rows are required")
    if test not in XI_TESTS:
        raise ValueError(f"test must be one of {XI_TESTS}")
    ranks, scale, ties = _output_ranks(y)
    if scale == 0:
        raise ValueError("the output is constant; xi is undefined")
    if test == "asymptotic" and ties:
        raise ValueError(
            "the output has tied values, for which the N(0, 2/5) null limit "
            "does not hold; use test='permutation'"
        )
    if test == "permutation" and (
        isinstance(permutations, bool)
        or not isinstance(permutations, int)
        or permutations < 1
    ):
        raise ValueError("permutations must be a positive integer")
    stream = resolve_stream(seed) if seed is not None else RandomStream(0)
    xi: list[float] = []
    tied: list[str] = []
    for i, column in enumerate(columns):
        order, has_ties = _sorted_order(column, stream, i)
        if has_ties:
            tied.append(labels[i])
        xi.append(_xi_from_ranks(ranks, order, scale))
    if test == "asymptotic":
        scale_z = math.sqrt(n / _XI_NULL_VARIANCE)
        p_values = [0.5 * math.erfc(v * scale_z / math.sqrt(2.0)) for v in xi]
        used = 0
    else:
        generator = stream.child("xi-permutation").generator()
        arrangement = list(range(n))
        exceed = [0] * len(xi)
        for _ in range(permutations):
            generator.shuffle(arrangement)
            value = _xi_from_ranks(ranks, arrangement, scale)
            for i, observed in enumerate(xi):
                if value >= observed:
                    exceed[i] += 1
        p_values = [(1 + count) / (permutations + 1) for count in exceed]
        used = permutations
    return ChatterjeeResult(
        labels, tuple(xi), tuple(p_values), test, used, n, ties, tuple(tied)
    )
