"""First-order sensitivity indices estimated from existing (given) data.

Any i.i.d. sample ``(x_k, y_k)`` of inputs and output (for example the runs
of a forward propagation) estimates ``S_i = Var(E[Y | X_i]) / Var(Y)``
without new model evaluations: sort the sample by ``x_i``, split it into
``M`` consecutive classes of (nearly) equal size, and compare the class means
with the overall mean (the correlation ratio of Pearson; the partition
estimator studied by Plischke, Borgonovo and Smith 2013,
[EQ-SENS-GIVEN-001]). Dependence between inputs is allowed: the index is
then the first-order "main effect" of ``X_i`` including what it carries
about correlated inputs.

Bias. The raw ratio ``eta^2 = SSB / SST`` is biased upwards: even an output
independent of ``X_i`` gives ``E[SSB] ~ (M - 1) Var(Y)``, i.e. a bias of
about ``(M - 1)(1 - S_i) / (n - 1)``. The corrected estimate
``1 - (n - 1)/(n - M) SSW / SST`` (Kelley's epsilon-squared) removes that
term and may be slightly negative for unimportant inputs. A second,
opposite bias comes from averaging ``E[Y | X_i]`` over each class: it is
``O(1/M^2)`` for a smooth conditional mean. ``M ~ sqrt(n)`` balances the
two; both biases vanish as ``n -> inf`` with ``M -> inf`` and ``M / n -> 0``.
The rows must be independent draws (a quasi-random design is fine for the
point estimate, but its rows are not independent, so do not resample them).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)


@dataclass(frozen=True, slots=True)
class GivenDataSensitivityResult:
    """First-order indices estimated by class partition of given data.

    Args:
        input_names: Input names.
        first_order: Bias-corrected estimates
            ``1 - (n - 1)/(n - M) SSW / SST``.
        first_order_raw: Uncorrected correlation ratios ``SSB / SST``.
        partitions: Number of classes ``M``.
        sample_size: Number of rows ``n``.
        output_variance: Sample variance of the output (divisor ``n - 1``).

    References:
        [REF-PLISCHKE-BORGONOVO-SMITH-2013], [REF-KELLEY-1935].

    """

    input_names: tuple[str, ...]
    first_order: tuple[float, ...]
    first_order_raw: tuple[float, ...]
    partitions: int
    sample_size: int
    output_variance: float

    def as_mapping(self) -> Mapping[str, float]:
        """Return ``input -> corrected first-order index``.

        References:
            [EQ-SENS-GIVEN-001]. No new equation is introduced.

        """
        return dict(zip(self.input_names, self.first_order, strict=True))


def given_data_first_order(
    inputs: Sequence[Sequence[float]],
    outputs: Sequence[float],
    *,
    partitions: int | None = None,
    names: Sequence[str] | None = None,
) -> GivenDataSensitivityResult:
    """Estimate first-order indices from an existing input-output sample.

    For input ``i``, the rows are sorted by ``x_i`` (ties keep their sample
    order) and cut into ``M`` consecutive classes whose sizes differ by at
    most one. With class means ``ybar_m``, sizes ``n_m``,
    ``SSB = sum_m n_m (ybar_m - ybar)^2``, ``SST = sum_k (y_k - ybar)^2`` and
    ``SSW = SST - SSB``, the raw estimate is ``SSB / SST`` and the corrected
    one ``1 - (n - 1)/(n - M) SSW / SST``.

    Args:
        inputs: ``n`` rows of ``d`` input values.
        outputs: ``n`` output values.
        partitions: Number of classes ``M`` in ``[2, n / 2]``; default
            ``max(2, min(50, floor(sqrt(n))))``.
        names: Input names (default ``x0, x1, ...``).

    Returns:
        Corrected and raw first-order estimates.

    Raises:
        ValueError: For ragged or too-small samples, non-finite values, an
            invalid ``partitions``, or a constant output.

    Equation:
        [EQ-SENS-GIVEN-001].

    References:
        [REF-PLISCHKE-BORGONOVO-SMITH-2013], [REF-KELLEY-1935],
        [REF-SALTELLI-2008, Chapter 1].

    """
    rows = tuple(
        tuple(require_finite_real(v, name=f"inputs[{k}][{j}]") for j, v in enumerate(r))
        for k, r in enumerate(inputs)
    )
    y = tuple(
        require_finite_real(v, name=f"outputs[{k}]") for k, v in enumerate(outputs)
    )
    n = len(rows)
    if n < 4 or len(y) != n:
        raise ValueError("inputs and outputs need the same length of at least four")
    dimension = len(rows[0])
    if dimension == 0 or any(len(r) != dimension for r in rows):
        raise ValueError("inputs must be a rectangular matrix with columns")
    if partitions is None:
        classes = max(2, min(50, math.isqrt(n)))
    else:
        classes = require_non_negative_integer(partitions, name="partitions")
    if not 2 <= classes <= n // 2:
        raise ValueError("partitions must lie in [2, n / 2]")
    labels = tuple(f"x{i}" for i in range(dimension)) if names is None else tuple(names)
    if len(labels) != dimension:
        raise ValueError("names must match the number of input columns")
    mean = math.fsum(y) / n
    total = math.fsum((v - mean) ** 2 for v in y)
    if not total > 0.0:
        raise ValueError("the output is constant; indices are undefined")
    raw: list[float] = []
    corrected: list[float] = []
    for i in range(dimension):
        order = sorted(range(n), key=lambda k: rows[k][i])
        between = 0.0
        terms: list[float] = []
        for m in range(classes):
            members = order[m * n // classes : (m + 1) * n // classes]
            class_mean = math.fsum(y[k] for k in members) / len(members)
            terms.append(len(members) * (class_mean - mean) ** 2)
        between = math.fsum(terms)
        within = total - between
        raw.append(between / total)
        corrected.append(1.0 - (n - 1) / (n - classes) * within / total)
    return GivenDataSensitivityResult(
        labels, tuple(corrected), tuple(raw), classes, n, total / (n - 1)
    )
