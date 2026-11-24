"""One-at-a-time swing (tornado) analysis.

Each input is moved alone from a baseline point to a low and to a high value
while every other input stays at its baseline; the output at the two ends and
the *swing* ``y_high - y_low`` are recorded and the inputs are sorted by
absolute swing. The cost is ``1 + 2 d`` model runs and the result is the data
of a tornado chart (:func:`probstat.visualization.plot_tornado`).

What a tornado chart cannot show:

* **Interactions.** An input is varied while the others sit at the baseline,
  so a term such as ``x1 * x2`` contributes only along the axes through the
  baseline point; an input that acts only through interactions with inputs
  held at a baseline of zero shows a swing of exactly zero.
* **Dependence between inputs.** The moves are along coordinate axes through
  one point. For a joint law with a non-identity copula that point and those
  moves are not consistent with the law, so :func:`tornado_analysis` refuses
  such a model (ADR-0141).
* **Non-monotone responses.** Two end points say nothing about what happens
  between them: ``y = x^2`` with ``x`` symmetric about the baseline has swing
  zero although ``x`` drives the output entirely. Every bar therefore carries
  a cheap check: the output at the baseline is a third point of the same
  response, and when it lies outside the interval spanned by the two end
  outputs the response cannot be monotone between the ends
  (``TornadoBar.nonmonotone``). The check detects an extremum inside the
  range, not every non-monotone shape (an S-shaped response passes it).

The swing is also not a variance share. The ``share`` reported here is the
share of the sum of squared half-swings, which is the quadratic-sum rule of a
response that is linear in every input; for a nonlinear model compare with
:func:`~probstat.sensitivity.sobol_analysis`.

The functions take either a :class:`~probstat.modeling.ProbabilisticModel`
(the ends are marginal quantiles, or explicit values for inputs known only as
ranges) or a plain callable with a ``low / base / high`` triple per input.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from probstat.foundations.validation import require_finite_real
from probstat.modeling import ProbabilisticModel

_BASELINE_NAMES = ("median", "mean")


@dataclass(frozen=True, slots=True)
class TornadoBar:
    """Swing of one input moved alone between a low and a high value.

    Args:
        name: Input name.
        low_input: Low value of the input.
        base_input: Baseline value of the input.
        high_input: High value of the input.
        low_output: Output with this input at ``low_input``.
        high_output: Output with this input at ``high_input``.
        swing: ``high_output - low_output``.
        share: ``(swing / 2)^2`` over the sum of that quantity across the
            inputs (``nan`` when every swing is zero).
        nonmonotone: ``True`` when the baseline output lies outside the
            interval spanned by the two end outputs, so the response has an
            extremum between the ends. ``False`` otherwise, including when
            the check does not apply (``bracketed`` is ``False``).
        bracketed: Whether ``base_input`` lies between ``low_input`` and
            ``high_input``; the non-monotonicity check needs it.

    References:
        [REF-HOWARD-1988], [REF-SALTELLI-2008, Chapter 1].

    """

    name: str
    low_input: float
    base_input: float
    high_input: float
    low_output: float
    high_output: float
    swing: float
    share: float
    nonmonotone: bool
    bracketed: bool

    @property
    def sign(self) -> int:
        """Return ``+1``, ``-1`` or ``0`` for the sign of the swing.

        References:
            [EQ-SENS-TORNADO-001], [REF-HOWARD-1988].

        """
        return (self.swing > 0.0) - (self.swing < 0.0)

    @property
    def half_swing(self) -> float:
        """Return half the absolute swing, ``|y_high - y_low| / 2``.

        References:
            [EQ-SENS-TORNADO-001], [REF-HOWARD-1988].

        """
        return 0.5 * abs(self.swing)


@dataclass(frozen=True, slots=True)
class TornadoResult:
    """Tornado data: bars sorted by absolute swing, with the baseline.

    Args:
        bars: One bar per input, largest absolute swing first (ties keep the
            input order).
        baseline_input: Baseline point in model (or caller) input order.
        baseline_output: Output at the baseline.
        input_names: Input names in the original order.
        quantity: QoI name, or ``None`` for a plain function.
        evaluations: Function evaluations spent, ``1 + 2 d``.
        low: Low probability level of a model analysis (``None`` for a plain
            function); inputs overridden by explicit values ignore it.
        high: High probability level of a model analysis (``None`` for a
            plain function).
        baseline: ``"median"``, ``"mean"``, or ``"point"``.

    References:
        [REF-HOWARD-1988], [REF-SALTELLI-2008, Chapter 1].

    """

    bars: tuple[TornadoBar, ...]
    baseline_input: tuple[float, ...]
    baseline_output: float
    input_names: tuple[str, ...]
    quantity: str | None
    evaluations: int
    low: float | None
    high: float | None
    baseline: str

    def ranking(self) -> tuple[str, ...]:
        """Return input names by decreasing absolute swing.

        References:
            [EQ-SENS-TORNADO-001], [REF-HOWARD-1988].

        """
        return tuple(bar.name for bar in self.bars)

    def as_mapping(self) -> Mapping[str, float]:
        """Return ``input -> swing`` in the original input order.

        References:
            [EQ-SENS-TORNADO-001], [REF-HOWARD-1988].

        """
        by_name = {bar.name: bar.swing for bar in self.bars}
        return MappingProxyType({name: by_name[name] for name in self.input_names})

    def nonmonotone_inputs(self) -> tuple[str, ...]:
        """Return inputs whose baseline output lies outside their end outputs.

        For these inputs the swing understates the effect (it can even be
        zero); check them with a scatter plot or a Sobol analysis.

        References:
            [EQ-SENS-TORNADO-002], [REF-HOWARD-1988].

        """
        return tuple(bar.name for bar in self.bars if bar.nonmonotone)


def _names(count: int, names: Sequence[str] | None) -> tuple[str, ...]:
    """Return validated input names (default ``x1, x2, ...``).

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    if names is None:
        return tuple(f"x{i + 1}" for i in range(count))
    labels = tuple(names)
    if len(labels) != count or len(set(labels)) != count:
        raise ValueError("names must be distinct and match the number of inputs")
    for label in labels:
        if not isinstance(label, str) or not label.strip():
            raise ValueError("names must be non-empty strings")
    return labels


def _vector(values: Sequence[float], name: str) -> tuple[float, ...]:
    """Return a non-empty tuple of finite reals.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of numbers")
    result = tuple(
        require_finite_real(value, name=f"{name}[{index}]")
        for index, value in enumerate(values)
    )
    if not result:
        raise ValueError(f"{name} must be non-empty")
    return result


def _ordered(
    values: Sequence[float] | Mapping[str, float],
    names: Sequence[str],
    label: str,
) -> tuple[float, ...]:
    """Return values in input order from a sequence or a name mapping.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION]. No equation is
        evaluated.

    """
    if isinstance(values, Mapping):
        if set(values) != set(names):
            raise ValueError(f"{label} must name exactly the inputs {tuple(names)}")
        return tuple(
            require_finite_real(values[n], name=f"{label}[{n}]") for n in names
        )
    vector = _vector(values, label)
    if len(vector) != len(names):
        raise ValueError(f"{label} must have one value per input")
    return vector


def _bars(
    function: Callable[[tuple[float, ...]], float],
    names: Sequence[str],
    low: Sequence[float],
    base: Sequence[float],
    high: Sequence[float],
) -> tuple[tuple[TornadoBar, ...], float]:
    """Evaluate the baseline and both ends of every input; return the bars.

    Equation:
        [EQ-SENS-TORNADO-001], [EQ-SENS-TORNADO-002].

    References:
        [REF-HOWARD-1988], [REF-SALTELLI-2008, Chapter 1].

    """

    def value(point: tuple[float, ...]) -> float:
        """Evaluate and validate the output at one point.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION]. No equation
            is evaluated.

        """
        return require_finite_real(function(point), name="the model output")

    centre = tuple(base)
    y_base = value(centre)
    raw: list[tuple[int, float, float]] = []
    for i in range(len(names)):
        at_low = value(centre[:i] + (low[i],) + centre[i + 1 :])
        at_high = value(centre[:i] + (high[i],) + centre[i + 1 :])
        raw.append((i, at_low, at_high))
    total = math.fsum((0.5 * (hi - lo)) ** 2 for _, lo, hi in raw)
    bars: list[TornadoBar] = []
    for i, lo, hi in raw:
        bracketed = min(low[i], high[i]) <= base[i] <= max(low[i], high[i])
        scale = 1e-12 * (abs(lo) + abs(hi) + abs(y_base))
        outside = y_base < min(lo, hi) - scale or y_base > max(lo, hi) + scale
        bars.append(
            TornadoBar(
                names[i],
                low[i],
                base[i],
                high[i],
                lo,
                hi,
                hi - lo,
                (0.5 * (hi - lo)) ** 2 / total if total > 0.0 else math.nan,
                bracketed and outside,
                bracketed,
            )
        )
    order = sorted(range(len(bars)), key=lambda k: -abs(bars[k].swing))
    return tuple(bars[k] for k in order), y_base


def tornado_function(
    function: Callable[[tuple[float, ...]], float],
    low: Sequence[float] | Mapping[str, float],
    base: Sequence[float] | Mapping[str, float],
    high: Sequence[float] | Mapping[str, float],
    *,
    names: Sequence[str] | None = None,
) -> TornadoResult:
    """Return the one-at-a-time swings of a plain function.

    Use this entry point for inputs known as ranges (``low / base / high``
    per input) rather than as laws. The caveats of the module apply: the
    result shows neither interactions nor dependence, and a swing says
    nothing about the shape of the response between the two ends; read
    :attr:`TornadoBar.nonmonotone`. Cost: ``1 + 2 d`` calls.

    Args:
        function: Scalar function of a tuple of ``d`` inputs.
        low: Low value of each input (sequence in input order, or mapping by
            name when ``names`` is given).
        base: Baseline value of each input.
        high: High value of each input.
        names: Input names (default ``x1, x2, ...``; required to use
            mappings).

    Returns:
        Bars sorted by absolute swing, the baseline output, and the cost.

    Raises:
        TypeError: If ``function`` is not callable or a value is not real.
        ValueError: If the lengths disagree, names repeat, a value is not
            finite, or the function returns a non-finite value.

    Equation:
        [EQ-SENS-TORNADO-001], [EQ-SENS-TORNADO-002].

    References:
        [REF-HOWARD-1988], [REF-SALTELLI-2008, Chapter 1].

    """
    if not callable(function):
        raise TypeError("function must be callable")
    if names is None and any(isinstance(v, Mapping) for v in (low, base, high)):
        raise ValueError("mappings need explicit names")
    reference = base if not isinstance(base, Mapping) else tuple(base)
    count = len(reference)
    labels = _names(count, names)
    lows = _ordered(low, labels, "low")
    bases = _ordered(base, labels, "base")
    highs = _ordered(high, labels, "high")
    bars, y_base = _bars(function, labels, lows, bases, highs)
    return TornadoResult(
        bars, bases, y_base, labels, None, 1 + 2 * count, None, None, "point"
    )


def tornado_analysis(
    model: ProbabilisticModel,
    quantity: str,
    *,
    low: float = 0.1,
    high: float = 0.9,
    baseline: str | Sequence[float] | Mapping[str, float] = "median",
    values: Mapping[str, tuple[float, float]] | None = None,
) -> TornadoResult:
    """Return the one-at-a-time swings of a model quantity.

    Each input is set to its ``low`` and ``high`` marginal quantile while the
    others stay at the baseline: the median of each marginal (default), its
    mean, or a point given by the caller. Inputs known only as ranges are
    given in ``values`` as ``name -> (low, high)`` and override the
    quantiles. Cost: ``1 + 2 d`` model runs.

    **A tornado chart does not show interactions** (inputs act one at a
    time), **dependence** (the model must have independent inputs; a joint
    law with a copula is refused because the moves along the axes are not
    consistent with it, ADR-0141) **or non-monotone behaviour** (only the two
    ends enter the swing; ``y = x^2`` with ``x`` symmetric about the baseline
    has swing zero). The cheap check ``TornadoBar.nonmonotone`` flags an
    extremum between the ends by comparing the output at the baseline with
    the two end outputs; it is skipped for inputs whose baseline lies outside
    the end values (for example a mean beyond the high quantile of a skewed
    law, flag ``bracketed``).

    Args:
        model: Probabilistic model with mutually independent inputs.
        quantity: QoI name.
        low: Low probability level in ``(0, 1)``.
        high: High probability level in ``(0, 1)``, above ``low``.
        baseline: ``"median"``, ``"mean"``, a point in model order, or a
            mapping by input name.
        values: Optional explicit ``(low, high)`` values per input name.

    Returns:
        Bars sorted by absolute swing, the baseline and its output.

    Raises:
        KeyError: If the quantity or an input name in a mapping is unknown.
        ValueError: If inputs are dependent, the levels are invalid, the
            baseline is invalid, or a model output is not finite.

    Equation:
        [EQ-SENS-TORNADO-001], [EQ-SENS-TORNADO-002].

    References:
        [REF-HOWARD-1988], [REF-SALTELLI-2008, Chapter 1].

    """
    model.quantity(quantity)
    if not model.inputs.is_independent:
        raise ValueError(
            "tornado analysis assumes independent inputs; the model's joint law "
            "has a non-identity copula, and one-at-a-time moves along the axes "
            "through a baseline point are not consistent with it. Use "
            "tornado_function with explicit scenario values, or shapley_effects"
        )
    p_low = require_finite_real(low, name="low")
    p_high = require_finite_real(high, name="high")
    if not 0.0 < p_low < p_high < 1.0:
        raise ValueError("levels must satisfy 0 < low < high < 1")
    names = model.inputs.names
    laws = model.inputs.inputs
    if isinstance(baseline, str):
        if baseline not in _BASELINE_NAMES:
            raise ValueError(f"baseline must be one of {_BASELINE_NAMES} or a point")
        kind = baseline
        if baseline == "median":
            centre = tuple(float(item.distribution.quantile(0.5)) for item in laws)
        else:
            centre = tuple(item.mean for item in laws)
    else:
        kind = "point"
        centre = (
            model.inputs.as_vector(baseline)
            if isinstance(baseline, Mapping)
            else _vector(baseline, "baseline")
        )
        if len(centre) != len(names):
            raise ValueError("baseline must have one value per input")
    lows = [float(item.distribution.quantile(p_low)) for item in laws]
    highs = [float(item.distribution.quantile(p_high)) for item in laws]
    if values is not None:
        for key, pair in values.items():
            if key not in names:
                raise KeyError(f"unknown input {key!r}")
            lo, hi = (require_finite_real(v, name=f"values[{key}]") for v in pair)
            lows[names.index(key)] = lo
            highs[names.index(key)] = hi
    before = model.evaluations
    bars, y_base = _bars(model.scalar_function(quantity), names, lows, centre, highs)
    return TornadoResult(
        bars,
        centre,
        y_base,
        names,
        quantity,
        model.evaluations - before,
        p_low,
        p_high,
        kind,
    )
