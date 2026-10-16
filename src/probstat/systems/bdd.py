"""Reduced ordered binary decision diagrams for exact system reliability.

A :class:`BDD` manager stores Boolean functions of named variables as
reduced ordered binary decision diagrams (Bryant 1986): every internal node
``(v, low, high)`` encodes the Shannon expansion
``f = (not x_v and f_low) or (x_v and f_high)``, the variable order is fixed
for the manager, no node has ``low == high``, and a unique table makes every
function canonical, so two functions are equal exactly when their node ids
are equal.

On top of the Boolean operations the manager provides the quantities that
system reliability needs (Rauzy 1993):

* exact probabilities ``P(f = 1)`` and ``P(f = 0)`` for independent variables
  by the Shannon recursion, as convex combinations without cancellation;
* every variable's conditional probabilities ``P(f | x_v = 0/1)`` and
  Birnbaum importance in one top-down and one bottom-up pass;
* the minimal solutions of a monotone function as a set-family diagram
  (Rauzy's ``minsol`` and ``without`` operators), from which minimal path
  and cut sets follow;
* the exact mean and variance of the multilinear reliability polynomial
  when the variable probabilities are themselves independent uncertain
  quantities, and an exact grouped variant in which several variables share
  one uncertain probability (state-of-knowledge correlation).

The engine is iterative (no Python recursion limit) and keeps every node it
creates (no garbage collection); ``max_nodes`` guards memory.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar, TypeVar

from probstat.foundations.validation import (
    require_non_negative_finite_real,
    require_probability,
)

_TERMINAL_LEVEL = 1 << 62
_PROBABILITY_SUM_TOLERANCE = 1e-9

_K = TypeVar("_K", bound=Hashable)
_V = TypeVar("_V")

Polynomial = dict[tuple[int, int], float]
MixedMoment = Callable[[int, int], float]


def _postorder(
    root: _K,
    leaf: Callable[[_K], _V | None],
    children: Callable[[_K], tuple[_K, ...]],
    combine: Callable[[_K, list[_V]], _V],
    memo: dict[_K, _V],
) -> _V:
    """Evaluate a memoized recursion on a DAG without Python recursion.

    Args:
        root: Key whose value is wanted.
        leaf: Returns the value of a base case, or ``None`` otherwise.
        children: Keys a non-base key depends on.
        combine: Value of a key from its children's values (same order).
        memo: Cache shared across calls; updated in place.

    Returns:
        The value of ``root``.

    References:
        [REF-BRYANT-1986] (memoized apply). No equation is evaluated.

    """
    if root in memo:
        return memo[root]
    stack = [root]
    expanded: dict[_K, tuple[_K, ...]] = {}
    while stack:
        key = stack[-1]
        if key in memo:
            stack.pop()
            continue
        kids = expanded.get(key)
        if kids is None:
            value = leaf(key)
            if value is not None:
                memo[key] = value
                stack.pop()
                continue
            kids = children(key)
            expanded[key] = kids
            pending = [kid for kid in kids if kid not in memo]
            if pending:
                stack.extend(pending)
                continue
        memo[key] = combine(key, [memo[kid] for kid in kids])
        del expanded[key]
        stack.pop()
    return memo[root]


class _ExactSum:
    """Exact running sum of floats (Shewchuk's non-overlapping partials).

    References:
        [REF-PYTHON-MATH] (``math.fsum`` uses the same partials).

    """

    __slots__ = ("_partials",)

    def __init__(self) -> None:
        """Start an empty sum.

        References:
            [REF-PYTHON-MATH]. No equation is evaluated.

        """
        self._partials: list[float] = []

    def add(self, x: float) -> None:
        """Add ``x`` exactly.

        Args:
            x: Finite float.

        References:
            [REF-PYTHON-MATH]. No equation is evaluated.

        """
        partials = self._partials
        index = 0
        for y in partials:
            if abs(x) < abs(y):
                x, y = y, x
            high = x + y
            low = y - (high - x)
            if low:
                partials[index] = low
                index += 1
            x = high
        partials[index:] = [x]

    def value(self) -> float:
        """Return the correctly rounded current sum.

        References:
            [REF-PYTHON-MATH]. No equation is evaluated.

        """
        return math.fsum(self._partials)


@dataclass(frozen=True, slots=True)
class VariableEffect:
    """Exact effect of fixing one variable on ``P(f)``.

    Args:
        variable: Variable name.
        true_given_one: ``P(f = 1 | x = 1)``.
        true_given_zero: ``P(f = 1 | x = 0)``.
        false_given_one: ``P(f = 0 | x = 1)``.
        false_given_zero: ``P(f = 0 | x = 0)``.
        birnbaum: ``P(f = 1 | x = 1) - P(f = 1 | x = 0) = dP(f = 1)/dp``.

    References:
        [REF-BIRNBAUM-1969], [REF-RAUZY-1993].

    """

    variable: str
    true_given_one: float
    true_given_zero: float
    false_given_one: float
    false_given_zero: float
    birnbaum: float


@dataclass(frozen=True, slots=True)
class PolynomialMoments:
    """Mean and variance of a reliability polynomial with uncertain inputs.

    Args:
        mean: ``E[P(f = 1)]`` over the variable-probability uncertainty.
        complement_mean: ``E[P(f = 0)] = 1 - mean``, computed directly.
        variance: ``Var[P(f = 1)] = Var[P(f = 0)]``.

    References:
        [REF-APOSTOLAKIS-KAPLAN-1981], [REF-RAUZY-1993].

    """

    mean: float
    complement_mean: float
    variance: float


class BDD:
    """A manager of reduced ordered binary decision diagrams.

    Node ids ``0`` and ``1`` are the constant functions ``FALSE`` and
    ``TRUE``. Variables are ordered by declaration (or by ``order``); the
    variable at a smaller level is tested first.

    Args:
        order: Variable names declared first, in this order.
        max_nodes: Guard on the number of stored nodes.

    Raises:
        ValueError: If ``order`` repeats a name or ``max_nodes`` is invalid.

    References:
        [REF-BRYANT-1986], [REF-RAUZY-1993].

    """

    FALSE: ClassVar[int] = 0
    TRUE: ClassVar[int] = 1

    def __init__(self, order: Iterable[str] = (), *, max_nodes: int = 5_000_000):
        """Create an empty manager with optional pre-declared variables.

        References:
            [REF-BRYANT-1986]. No equation is evaluated.

        """
        if isinstance(max_nodes, bool) or not isinstance(max_nodes, int):
            raise TypeError("max_nodes must be an integer")
        if max_nodes < 2:
            raise ValueError("max_nodes must be at least two")
        self._max_nodes = max_nodes
        self._names: list[str] = []
        self._levels: dict[str, int] = {}
        self._level: list[int] = [_TERMINAL_LEVEL, _TERMINAL_LEVEL]
        self._low: list[int] = [0, 1]
        self._high: list[int] = [0, 1]
        self._unique: dict[tuple[int, int, int], int] = {}
        self._cache: dict[tuple[object, ...], int] = {}
        self._monotone: dict[int, bool] = {}
        for name in order:
            if name in self._levels:
                raise ValueError(f"variable {name!r} is repeated in the order")
            self.declare(name)

    # ------------------------------------------------------------------
    # variables and nodes
    # ------------------------------------------------------------------
    def declare(self, name: str) -> int:
        """Declare a variable (if new) and return its level.

        Args:
            name: Non-empty variable name.

        Returns:
            The variable's level (0 is tested first).

        Raises:
            TypeError: If ``name`` is not a string.
            ValueError: If ``name`` is empty.

        References:
            [REF-BRYANT-1986]. No equation is evaluated.

        """
        if not isinstance(name, str):
            raise TypeError("variable names must be strings")
        if not name:
            raise ValueError("variable names must not be empty")
        level = self._levels.get(name)
        if level is None:
            level = len(self._names)
            self._names.append(name)
            self._levels[name] = level
        return level

    def variable(self, name: str) -> int:
        """Return the node of the single-variable function ``x_name``.

        Args:
            name: Variable name; declared at the end of the order if new.

        Returns:
            Node id.

        References:
            [REF-BRYANT-1986]. No equation is evaluated.

        """
        return self._mk(self.declare(name), self.FALSE, self.TRUE)

    @property
    def variables(self) -> tuple[str, ...]:
        """Return the declared variable names in order.

        References:
            [REF-BRYANT-1986].

        """
        return tuple(self._names)

    def level(self, name: str) -> int:
        """Return the level of a declared variable.

        Args:
            name: Variable name.

        Returns:
            Level index.

        Raises:
            KeyError: If the variable is not declared.

        References:
            [REF-BRYANT-1986]. No equation is evaluated.

        """
        return self._levels[name]

    def node(self, node: int) -> tuple[str, int, int]:
        """Return ``(variable, low, high)`` of an internal node.

        Args:
            node: Internal node id.

        Returns:
            The tested variable and the two child ids.

        Raises:
            ValueError: If ``node`` is a terminal or unknown.

        References:
            [REF-BRYANT-1986]. No equation is evaluated.

        """
        self._check(node)
        if node < 2:
            raise ValueError("terminal nodes have no variable")
        return self._names[self._level[node]], self._low[node], self._high[node]

    @property
    def size(self) -> int:
        """Return the number of stored nodes, terminals included.

        References:
            [REF-BRYANT-1986].

        """
        return len(self._level)

    def _check(self, node: int) -> int:
        """Validate a node id.

        Args:
            node: Candidate node id.

        Returns:
            The node id.

        Raises:
            TypeError: If ``node`` is not an integer.
            ValueError: If ``node`` is not a stored node.

        References:
            Project validation policy. No equation is evaluated.

        """
        if isinstance(node, bool) or not isinstance(node, int):
            raise TypeError("node ids must be integers")
        if not 0 <= node < len(self._level):
            raise ValueError(f"unknown node id {node}")
        return node

    def _new(self, level: int, low: int, high: int) -> int:
        """Return the unique node ``(level, low, high)`` without reduction.

        Raises:
            ValueError: If the node guard is exceeded.

        References:
            [REF-BRYANT-1986]. No equation is evaluated.

        """
        key = (level, low, high)
        found = self._unique.get(key)
        if found is not None:
            return found
        if len(self._level) >= self._max_nodes:
            raise ValueError(f"BDD node limit of {self._max_nodes} exceeded")
        node = len(self._level)
        self._level.append(level)
        self._low.append(low)
        self._high.append(high)
        self._unique[key] = node
        return node

    def _mk(self, level: int, low: int, high: int) -> int:
        """Return the reduced Boolean node ``ite(x_level, high, low)``.

        References:
            [EQ-SYS-BDD-001], [REF-BRYANT-1986].

        """
        if low == high:
            return low
        return self._new(level, low, high)

    def _family_mk(self, level: int, low: int, high: int) -> int:
        """Return a set-family node: sets of ``low`` plus ``{v} + high`` sets.

        A family node with an empty ``high`` family is the ``low`` family
        (zero-suppression); ``low == high`` is not reduced because it denotes
        a different family.

        References:
            [EQ-SYS-MINSOL-001], [REF-RAUZY-1993].

        """
        if high == self.FALSE:
            return low
        return self._new(level, low, high)

    def _cofactors(self, node: int, level: int) -> tuple[int, int]:
        """Return ``(f|x=0, f|x=1)`` for the variable at ``level``.

        Only valid when ``level`` is at or above the node's level.

        References:
            [EQ-SYS-BDD-001], [REF-BRYANT-1986].

        """
        if self._level[node] == level:
            return self._low[node], self._high[node]
        return node, node

    # ------------------------------------------------------------------
    # Boolean operations
    # ------------------------------------------------------------------
    def _terminal_case(self, op: str, a: int, b: int) -> int | None:
        """Return the result of a binary operation when it is immediate.

        References:
            [REF-BRYANT-1986]. No equation is evaluated.

        """
        if op == "and":
            if a == self.FALSE or b == self.FALSE:
                return self.FALSE
            if a == self.TRUE:
                return b
            if b == self.TRUE or a == b:
                return a
            return None
        if a == self.TRUE or b == self.TRUE:
            return self.TRUE
        if a == self.FALSE:
            return b
        if b == self.FALSE or a == b:
            return a
        return None

    def _apply(self, op: str, f: int, g: int) -> int:
        """Combine two functions with ``op`` in ``{"and", "or"}``.

        References:
            [EQ-SYS-BDD-001], [REF-BRYANT-1986].

        """
        self._check(f)
        self._check(g)
        level_of = self._level

        def leaf(key: tuple[object, ...]) -> int | None:
            """Return the immediate result or ``None``.

            References:
                [REF-BRYANT-1986]. No equation is evaluated.

            """
            _, a, b = key
            assert isinstance(a, int) and isinstance(b, int)
            return self._terminal_case(op, a, b)

        def children(key: tuple[object, ...]) -> tuple[tuple[object, ...], ...]:
            """Return the cofactor pairs of the top variable.

            References:
                [EQ-SYS-BDD-001], [REF-BRYANT-1986].

            """
            _, a, b = key
            assert isinstance(a, int) and isinstance(b, int)
            level = min(level_of[a], level_of[b])
            a0, a1 = self._cofactors(a, level)
            b0, b1 = self._cofactors(b, level)
            return (
                (op, min(a0, b0), max(a0, b0)),
                (op, min(a1, b1), max(a1, b1)),
            )

        def combine(key: tuple[object, ...], values: list[int]) -> int:
            """Build the reduced node from the two cofactor results.

            References:
                [EQ-SYS-BDD-001], [REF-BRYANT-1986].

            """
            _, a, b = key
            assert isinstance(a, int) and isinstance(b, int)
            return self._mk(min(level_of[a], level_of[b]), values[0], values[1])

        return _postorder(
            (op, min(f, g), max(f, g)), leaf, children, combine, self._cache
        )

    def apply_and(self, f: int, g: int) -> int:
        """Return the conjunction ``f and g``.

        Args:
            f: Node id.
            g: Node id.

        Returns:
            Node id of the conjunction.

        Equation:
            [EQ-SYS-BDD-001].

        References:
            [REF-BRYANT-1986].

        """
        return self._apply("and", f, g)

    def apply_or(self, f: int, g: int) -> int:
        """Return the disjunction ``f or g``.

        Args:
            f: Node id.
            g: Node id.

        Returns:
            Node id of the disjunction.

        Equation:
            [EQ-SYS-BDD-001].

        References:
            [REF-BRYANT-1986].

        """
        return self._apply("or", f, g)

    def conjunction(self, nodes: Iterable[int]) -> int:
        """Return the conjunction of several functions (``TRUE`` if none).

        Args:
            nodes: Node ids.

        Returns:
            Node id.

        References:
            [REF-BRYANT-1986]. No equation is evaluated.

        """
        result = self.TRUE
        for node in nodes:
            result = self.apply_and(result, node)
        return result

    def disjunction(self, nodes: Iterable[int]) -> int:
        """Return the disjunction of several functions (``FALSE`` if none).

        Args:
            nodes: Node ids.

        Returns:
            Node id.

        References:
            [REF-BRYANT-1986]. No equation is evaluated.

        """
        result = self.FALSE
        for node in nodes:
            result = self.apply_or(result, node)
        return result

    def _unary(self, tag: str, f: int, build: Callable[[int, int, int], int]) -> int:
        """Run a memoized structural map ``node -> build(level, lo', hi')``.

        Terminals map through ``negate``-style swapping when ``tag`` starts
        with ``"not"`` or ``"dual"`` and stay fixed otherwise.

        References:
            [REF-BRYANT-1986]. No equation is evaluated.

        """
        self._check(f)
        swap = tag in ("not", "dual")

        def leaf(key: tuple[object, ...]) -> int | None:
            """Map terminals.

            References:
                [REF-BRYANT-1986]. No equation is evaluated.

            """
            node = key[1]
            assert isinstance(node, int)
            if node < 2:
                return 1 - node if swap else node
            return None

        def children(key: tuple[object, ...]) -> tuple[tuple[object, ...], ...]:
            """Return the low and high child keys.

            References:
                [REF-BRYANT-1986]. No equation is evaluated.

            """
            node = key[1]
            assert isinstance(node, int)
            return ((tag, self._low[node]), (tag, self._high[node]))

        def combine(key: tuple[object, ...], values: list[int]) -> int:
            """Rebuild the node from its mapped children.

            References:
                [REF-BRYANT-1986]. No equation is evaluated.

            """
            node = key[1]
            assert isinstance(node, int)
            return build(self._level[node], values[0], values[1])

        return _postorder((tag, f), leaf, children, combine, self._cache)

    def negate(self, f: int) -> int:
        """Return the complement ``not f``.

        Args:
            f: Node id.

        Returns:
            Node id of the complement.

        Equation:
            [EQ-SYS-BDD-001].

        References:
            [REF-BRYANT-1986].

        """
        return self._unary("not", f, self._mk)

    def dual(self, f: int) -> int:
        """Return the dual ``f^D(y) = not f(not y)``.

        For a success function ``f(x)`` with ``x_i = 1`` meaning *component
        works*, the dual is the failure function in the variables
        ``y_i = 1 - x_i`` (*component failed*), so minimal solutions of the
        dual are minimal cut sets.

        Args:
            f: Node id.

        Returns:
            Node id of the dual.

        Equation:
            [EQ-SYS-MINSOL-001].

        References:
            [REF-BARLOW-PROSCHAN-1975], [REF-RAUZY-1993].

        """
        return self._unary("dual", f, lambda level, lo, hi: self._mk(level, hi, lo))

    def ite(self, f: int, g: int, h: int) -> int:
        """Return ``if f then g else h``.

        Args:
            f: Condition node.
            g: Node used where ``f`` holds.
            h: Node used where ``f`` fails.

        Returns:
            Node id.

        Equation:
            [EQ-SYS-BDD-001].

        References:
            [REF-BRYANT-1986].

        """
        return self.apply_or(self.apply_and(f, g), self.apply_and(self.negate(f), h))

    def restrict(self, f: int, name: str, value: bool) -> int:
        """Return the cofactor ``f|_{x_name = value}``.

        Args:
            f: Node id.
            name: Declared variable name.
            value: Truth value assigned to the variable.

        Returns:
            Node id of the cofactor.

        Raises:
            KeyError: If the variable is not declared.
            TypeError: If ``value`` is not a bool.

        Equation:
            [EQ-SYS-BDD-001].

        References:
            [REF-BRYANT-1986].

        """
        if not isinstance(value, bool):
            raise TypeError("value must be a bool")
        target = self._levels[name]
        self._check(f)
        tag = ("restrict", target, value)

        def leaf(key: tuple[object, ...]) -> int | None:
            """Return nodes below the variable unchanged.

            References:
                [REF-BRYANT-1986]. No equation is evaluated.

            """
            node = key[-1]
            assert isinstance(node, int)
            level = self._level[node]
            if level > target:
                return node
            if level == target:
                return self._high[node] if value else self._low[node]
            return None

        def children(key: tuple[object, ...]) -> tuple[tuple[object, ...], ...]:
            """Return the child keys.

            References:
                [REF-BRYANT-1986]. No equation is evaluated.

            """
            node = key[-1]
            assert isinstance(node, int)
            return ((*tag, self._low[node]), (*tag, self._high[node]))

        def combine(key: tuple[object, ...], values: list[int]) -> int:
            """Rebuild the node above the restricted variable.

            References:
                [REF-BRYANT-1986]. No equation is evaluated.

            """
            node = key[-1]
            assert isinstance(node, int)
            return self._mk(self._level[node], values[0], values[1])

        return _postorder((*tag, f), leaf, children, combine, self._cache)

    # ------------------------------------------------------------------
    # inspection
    # ------------------------------------------------------------------
    def _reachable(self, f: int) -> list[int]:
        """Return internal nodes reachable from ``f`` sorted by level.

        References:
            [REF-BRYANT-1986]. No equation is evaluated.

        """
        self._check(f)
        seen: set[int] = set()
        stack = [f]
        while stack:
            node = stack.pop()
            if node < 2 or node in seen:
                continue
            seen.add(node)
            stack.append(self._low[node])
            stack.append(self._high[node])
        return sorted(seen, key=lambda node: (self._level[node], node))

    def nodes(self, f: int) -> tuple[int, ...]:
        """Return the internal nodes reachable from ``f``, sorted by level.

        Args:
            f: Node id.

        Returns:
            Node ids; every node precedes its internal children.

        References:
            [REF-BRYANT-1986]. No equation is evaluated.

        """
        return tuple(self._reachable(f))

    def node_count(self, f: int) -> int:
        """Return the number of internal nodes reachable from ``f``.

        Args:
            f: Node id.

        Returns:
            Internal node count (terminals excluded).

        References:
            [REF-BRYANT-1986]. No equation is evaluated.

        """
        return len(self._reachable(f))

    def support(self, f: int) -> tuple[str, ...]:
        """Return the variables ``f`` depends on, in order.

        Args:
            f: Node id.

        Returns:
            Variable names.

        References:
            [REF-BRYANT-1986]. No equation is evaluated.

        """
        levels = sorted({self._level[node] for node in self._reachable(f)})
        return tuple(self._names[level] for level in levels)

    def evaluate(self, f: int, assignment: Mapping[str, bool | int]) -> bool:
        """Return ``f`` at a complete assignment of its support.

        Args:
            f: Node id.
            assignment: Variable truth values (``bool`` or ``0``/``1``).

        Returns:
            Function value.

        Raises:
            KeyError: If a support variable is missing.
            ValueError: If a value is not boolean or ``0``/``1``.

        Equation:
            [EQ-SYS-BDD-001].

        References:
            [REF-BRYANT-1986].

        """
        node = self._check(f)
        while node >= 2:
            name = self._names[self._level[node]]
            value = assignment[name]
            if value not in (0, 1) or isinstance(value, float):
                raise ValueError(f"state of {name!r} must be boolean or 0/1")
            node = self._high[node] if value else self._low[node]
        return node == self.TRUE

    def is_monotone(self, f: int) -> bool:
        """Return whether ``f`` is non-decreasing in every variable.

        A reduced ordered BDD is monotone exactly when ``low <= high`` at
        every reachable node, i.e. ``low and not high`` is ``FALSE``.

        Args:
            f: Node id.

        Returns:
            ``True`` for a monotone (coherent-type) function.

        References:
            [REF-BARLOW-PROSCHAN-1975], [REF-RAUZY-1993].

        """
        cached = self._monotone.get(f)
        if cached is not None:
            return cached
        result = all(
            self.apply_and(self._low[node], self.negate(self._high[node])) == self.FALSE
            for node in self._reachable(f)
        )
        self._monotone[f] = result
        return result

    # ------------------------------------------------------------------
    # probabilities
    # ------------------------------------------------------------------
    def _weights(
        self,
        nodes: Sequence[int],
        p: Mapping[str, float],
        q: Mapping[str, float] | None,
    ) -> dict[int, tuple[float, float]]:
        """Return validated ``(P(x=0), P(x=1))`` per level used by ``nodes``.

        Raises:
            KeyError: If a support variable has no probability.
            ValueError: If a probability is invalid or ``p + q != 1``.

        References:
            [EQ-PROB-001], [REF-KOLMOGOROV-1956].

        """
        weights: dict[int, tuple[float, float]] = {}
        for node in nodes:
            level = self._level[node]
            if level in weights:
                continue
            name = self._names[level]
            if name not in p:
                raise KeyError(f"no probability for variable {name!r}")
            one = _probability(p[name], name=f"p[{name!r}]")
            if q is None:
                zero = 1.0 - one
            else:
                if name not in q:
                    raise KeyError(f"no complement probability for {name!r}")
                zero = _probability(q[name], name=f"q[{name!r}]")
                if abs(one + zero - 1.0) > _PROBABILITY_SUM_TOLERANCE:
                    raise ValueError(f"p and q of {name!r} must sum to one")
            weights[level] = (zero, one)
        return weights

    def _values(
        self, nodes: Sequence[int], weights: Mapping[int, tuple[float, float]]
    ) -> tuple[dict[int, float], dict[int, float]]:
        """Return ``P(f_n = 1)`` and ``P(f_n = 0)`` for every node.

        Each value is a convex combination of child values, so no
        cancellation occurs in either table.

        References:
            [EQ-SYS-BDD-001], [REF-RAUZY-1993].

        """
        one: dict[int, float] = {0: 0.0, 1: 1.0}
        zero: dict[int, float] = {0: 1.0, 1: 0.0}
        for node in reversed(nodes):
            w0, w1 = weights[self._level[node]]
            low, high = self._low[node], self._high[node]
            one[node] = w0 * one[low] + w1 * one[high]
            zero[node] = w0 * zero[low] + w1 * zero[high]
        return one, zero

    def probability(
        self,
        f: int,
        p: Mapping[str, float],
        q: Mapping[str, float] | None = None,
        *,
        value: bool = True,
    ) -> float:
        """Return ``P(f = value)`` for independent variables.

        Args:
            f: Node id.
            p: ``P(x_v = 1)`` for every support variable.
            q: Optional ``P(x_v = 0)``; supply it when ``1 - p`` would lose
                accuracy (e.g. component unreliabilities near ``1e-12``).
            value: ``True`` for ``P(f = 1)``, ``False`` for ``P(f = 0)``.

        Returns:
            Exact probability (to rounding) under independence.

        Raises:
            KeyError: If a support variable has no probability.
            ValueError: If a probability is invalid.

        Equation:
            [EQ-SYS-BDD-001].

        References:
            [REF-RAUZY-1993], [REF-BRYANT-1986].

        """
        nodes = self._reachable(f)
        weights = self._weights(nodes, p, q)
        one, zero = self._values(nodes, weights)
        return min(1.0, max(0.0, one[f] if value else zero[f]))

    def variable_effects(
        self,
        f: int,
        p: Mapping[str, float],
        q: Mapping[str, float] | None = None,
        variables: Iterable[str] | None = None,
    ) -> Mapping[str, VariableEffect]:
        """Return every variable's conditional probabilities in one pass.

        A top-down pass computes the probability ``r(n)`` that the random
        path reaches node ``n``; a bottom-up pass computes ``P(f_n = 1)``
        and ``P(f_n = 0)``. Every path crosses level ``v`` exactly once,
        either at a node labelled ``v`` or along an edge that skips it, so
        ``P(f = b | x_v = c) = sum_{label(n)=v} r(n) P(f_{child_c(n)} = b)
        + sum_{edges e skipping v} w(e) P(f_{target(e)} = b)``. The skipping
        sums are accumulated exactly. Birnbaum's measure is
        ``sum_{label(n)=v} r(n) [P(f_high = 1) - P(f_low = 1)]``.

        Args:
            f: Node id.
            p: ``P(x = 1)`` for every support variable.
            q: Optional ``P(x = 0)`` (see :meth:`probability`).
            variables: Variables to report (default: all declared ones).

        Returns:
            Read-only mapping from variable name to its effect.

        Raises:
            KeyError: If a requested variable is not declared or a support
                variable has no probability.

        Equation:
            [EQ-SYS-BDD-002].

        References:
            [REF-BIRNBAUM-1969], [REF-RAUZY-1993].

        """
        nodes = self._reachable(f)
        wanted = tuple(self._names) if variables is None else tuple(variables)
        for name in wanted:
            if name not in self._levels:
                raise KeyError(f"unknown variable {name!r}")
        weights = self._weights(nodes, p, q)
        one, zero = self._values(nodes, weights)
        if f < 2:
            constant = float(f)
            return MappingProxyType(
                {
                    name: VariableEffect(
                        name, constant, constant, 1.0 - constant, 1.0 - constant, 0.0
                    )
                    for name in wanted
                }
            )
        levels = len(self._names)
        reach: dict[int, float] = {node: 0.0 for node in nodes}
        reach[f] = 1.0
        starts: dict[int, list[tuple[float, float]]] = {}
        ends: dict[int, list[tuple[float, float]]] = {}

        def span(source_level: int, target: int, weight: float) -> None:
            """Register an edge that skips levels strictly between ends.

            References:
                [EQ-SYS-BDD-002]. No further equation is evaluated.

            """
            target_level = min(self._level[target], levels)
            if target_level - source_level <= 1 or weight == 0.0:
                return
            contribution = (weight * one[target], weight * zero[target])
            starts.setdefault(source_level + 1, []).append(contribution)
            ends.setdefault(target_level, []).append(contribution)

        span(-1, f, 1.0)
        per_level: dict[int, list[int]] = {}
        for node in nodes:
            level = self._level[node]
            per_level.setdefault(level, []).append(node)
            w0, w1 = weights[level]
            mass = reach[node]
            for child, weight in ((self._low[node], w0), (self._high[node], w1)):
                if child >= 2:
                    reach[child] += mass * weight
                span(level, child, mass * weight)
        skip_one = _ExactSum()
        skip_zero = _ExactSum()
        crossing: dict[int, tuple[float, float]] = {}
        for level in range(levels):
            for a, b in starts.get(level, ()):
                skip_one.add(a)
                skip_zero.add(b)
            for a, b in ends.get(level, ()):
                skip_one.add(-a)
                skip_zero.add(-b)
            crossing[level] = (max(0.0, skip_one.value()), max(0.0, skip_zero.value()))
        result: dict[str, VariableEffect] = {}
        for name in wanted:
            level = self._levels[name]
            base_one, base_zero = crossing[level]
            labelled = per_level.get(level, [])
            t1 = math.fsum(
                [base_one, *(reach[n] * one[self._high[n]] for n in labelled)]
            )
            t0 = math.fsum(
                [base_one, *(reach[n] * one[self._low[n]] for n in labelled)]
            )
            f1 = math.fsum(
                [base_zero, *(reach[n] * zero[self._high[n]] for n in labelled)]
            )
            f0 = math.fsum(
                [base_zero, *(reach[n] * zero[self._low[n]] for n in labelled)]
            )
            differences = []
            for n in labelled:
                low, high = self._low[n], self._high[n]
                if zero[low] + zero[high] < one[low] + one[high]:
                    differences.append(reach[n] * (zero[low] - zero[high]))
                else:
                    differences.append(reach[n] * (one[high] - one[low]))
            result[name] = VariableEffect(
                name,
                _clip(t1),
                _clip(t0),
                _clip(f1),
                _clip(f0),
                math.fsum(differences),
            )
        return MappingProxyType(result)

    # ------------------------------------------------------------------
    # minimal solutions (Rauzy 1993)
    # ------------------------------------------------------------------
    def _without(self, family: int, other: int) -> int:
        """Return the sets of ``family`` that contain no set of ``other``.

        References:
            [EQ-SYS-MINSOL-001], [REF-RAUZY-1993].

        """
        level_of = self._level

        def leaf(key: tuple[object, ...]) -> int | None:
            """Return immediate results.

            References:
                [REF-RAUZY-1993]. No equation is evaluated.

            """
            _, a, b = key
            assert isinstance(a, int) and isinstance(b, int)
            if a == self.FALSE or b == self.TRUE or a == b:
                return self.FALSE
            if b == self.FALSE or a == self.TRUE:
                return a
            return None

        def children(key: tuple[object, ...]) -> tuple[tuple[object, ...], ...]:
            """Return the sub-problems of Rauzy's ``without`` recursion.

            References:
                [EQ-SYS-MINSOL-001], [REF-RAUZY-1993].

            """
            _, a, b = key
            assert isinstance(a, int) and isinstance(b, int)
            la, lb = level_of[a], level_of[b]
            if la < lb:
                return (
                    ("without", self._low[a], b),
                    ("without", self._high[a], b),
                )
            if la > lb:
                return (("without", a, self._low[b]),)
            return (
                ("without", self._low[a], self._low[b]),
                ("without", self._high[a], self._high[b]),
            )

        def combine(key: tuple[object, ...], values: list[int]) -> int:
            """Assemble the family node.

            References:
                [EQ-SYS-MINSOL-001], [REF-RAUZY-1993].

            """
            _, a, b = key
            assert isinstance(a, int) and isinstance(b, int)
            la, lb = level_of[a], level_of[b]
            if la < lb:
                return self._family_mk(la, values[0], values[1])
            if la > lb:
                return values[0]
            high = self._without(values[1], self._low[b])
            return self._family_mk(la, values[0], high)

        return _postorder(
            ("without", family, other), leaf, children, combine, self._cache
        )

    def minimal_solution_family(self, f: int) -> int:
        """Return the set-family diagram of the minimal solutions of ``f``.

        Rauzy's recursion for a monotone ``f = ite(x, f1, f0)``:
        ``minsol(f) = {S + {x} : S in minsol(f1) without minsol(f0)}
        union minsol(f0)``.

        Args:
            f: Node id of a monotone function.

        Returns:
            Family node id (``FALSE``: no solution; ``TRUE``: only the empty
            set).

        Raises:
            ValueError: If ``f`` is not monotone.

        Equation:
            [EQ-SYS-MINSOL-001].

        References:
            [REF-RAUZY-1993].

        """
        self._check(f)
        if not self.is_monotone(f):
            raise ValueError("minimal solutions require a monotone function")

        def leaf(key: tuple[object, ...]) -> int | None:
            """Return terminal families.

            References:
                [REF-RAUZY-1993]. No equation is evaluated.

            """
            node = key[1]
            assert isinstance(node, int)
            return node if node < 2 else None

        def children(key: tuple[object, ...]) -> tuple[tuple[object, ...], ...]:
            """Return the cofactor sub-problems.

            References:
                [REF-RAUZY-1993]. No equation is evaluated.

            """
            node = key[1]
            assert isinstance(node, int)
            return (("minsol", self._low[node]), ("minsol", self._high[node]))

        def combine(key: tuple[object, ...], values: list[int]) -> int:
            """Apply Rauzy's minsol step.

            References:
                [EQ-SYS-MINSOL-001], [REF-RAUZY-1993].

            """
            node = key[1]
            assert isinstance(node, int)
            low = values[0]
            return self._family_mk(
                self._level[node], low, self._without(values[1], low)
            )

        return _postorder(("minsol", f), leaf, children, combine, self._cache)

    def family_count(self, family: int) -> int:
        """Return the number of sets in a family diagram.

        Args:
            family: Family node id.

        Returns:
            Exact count (Python integer).

        References:
            [REF-RAUZY-1993]. No equation is evaluated.

        """
        self._check(family)
        counts: dict[int, int] = {0: 0, 1: 1}
        for node in reversed(self._reachable(family)):
            counts[node] = counts[self._low[node]] + counts[self._high[node]]
        return counts[family]

    def family_sets(
        self, family: int, *, limit: int = 1_000_000
    ) -> tuple[frozenset[str], ...]:
        """Enumerate the sets of a family diagram.

        Args:
            family: Family node id.
            limit: Maximum number of sets to enumerate.

        Returns:
            Sets sorted by size, then by sorted names.

        Raises:
            ValueError: If the family has more than ``limit`` sets.

        References:
            [REF-RAUZY-1993]. No equation is evaluated.

        """
        count = self.family_count(family)
        if count > limit:
            raise ValueError(f"{count} sets exceed the enumeration limit {limit}")
        found: list[frozenset[str]] = []
        stack: list[tuple[int, tuple[str, ...]]] = [(family, ())]
        while stack:
            node, chosen = stack.pop()
            if node == self.FALSE:
                continue
            if node == self.TRUE:
                found.append(frozenset(chosen))
                continue
            name = self._names[self._level[node]]
            stack.append((self._low[node], chosen))
            stack.append((self._high[node], (*chosen, name)))
        return tuple(sorted(found, key=lambda s: (len(s), sorted(s))))

    def family_onset(self, family: int, name: str) -> int:
        """Return the sub-family of sets that contain a variable.

        Args:
            family: Family node id.
            name: Declared variable name.

        Returns:
            Family node id.

        Raises:
            KeyError: If the variable is not declared.

        References:
            [REF-RAUZY-1993]. No equation is evaluated.

        """
        target = self._levels[name]
        self._check(family)

        def leaf(key: tuple[object, ...]) -> int | None:
            """Return the onset at or below the variable.

            References:
                [REF-RAUZY-1993]. No equation is evaluated.

            """
            node = key[2]
            assert isinstance(node, int)
            level = self._level[node]
            if level > target:
                return self.FALSE
            if level == target:
                return self._family_mk(level, self.FALSE, self._high[node])
            return None

        def children(key: tuple[object, ...]) -> tuple[tuple[object, ...], ...]:
            """Return child keys.

            References:
                [REF-RAUZY-1993]. No equation is evaluated.

            """
            node = key[2]
            assert isinstance(node, int)
            return (
                ("onset", target, self._low[node]),
                ("onset", target, self._high[node]),
            )

        def combine(key: tuple[object, ...], values: list[int]) -> int:
            """Rebuild the family node.

            References:
                [REF-RAUZY-1993]. No equation is evaluated.

            """
            node = key[2]
            assert isinstance(node, int)
            return self._family_mk(self._level[node], values[0], values[1])

        return _postorder(
            ("onset", target, family), leaf, children, combine, self._cache
        )

    def family_function(self, family: int) -> int:
        """Return the monotone function ``OR_{S in family} AND_{v in S} x_v``.

        Args:
            family: Family node id.

        Returns:
            Boolean node id.

        Equation:
            [EQ-SYS-MINSOL-001].

        References:
            [REF-RAUZY-1993].

        """
        self._check(family)

        def leaf(key: tuple[object, ...]) -> int | None:
            """Map terminal families to constants.

            References:
                [REF-RAUZY-1993]. No equation is evaluated.

            """
            node = key[1]
            assert isinstance(node, int)
            return node if node < 2 else None

        def children(key: tuple[object, ...]) -> tuple[tuple[object, ...], ...]:
            """Return child keys.

            References:
                [REF-RAUZY-1993]. No equation is evaluated.

            """
            node = key[1]
            assert isinstance(node, int)
            return (("famfun", self._low[node]), ("famfun", self._high[node]))

        def combine(key: tuple[object, ...], values: list[int]) -> int:
            """Return ``low or (x and high)``.

            References:
                [REF-RAUZY-1993]. No equation is evaluated.

            """
            node = key[1]
            assert isinstance(node, int)
            literal = self._mk(self._level[node], self.FALSE, self.TRUE)
            return self.apply_or(values[0], self.apply_and(literal, values[1]))

        return _postorder(("famfun", family), leaf, children, combine, self._cache)

    # ------------------------------------------------------------------
    # epistemic moments
    # ------------------------------------------------------------------
    def moments(
        self,
        f: int,
        means: Mapping[str, float],
        variances: Mapping[str, float],
    ) -> PolynomialMoments:
        """Return the exact mean and variance of ``R(p) = P(f = 1 | p)``.

        The variable probabilities ``p_v`` are independent random quantities
        with the given means ``m_v`` and variances ``s_v``. ``R`` is
        multilinear, so ``E[R] = R(m)``. For the variance a pairwise
        recursion on sub-functions ``F, G`` with cofactors ``F_1, F_0`` at
        the top variable gives the covariance
        ``C(F, G) = (m^2 + s) C(F_1, G_1) + (m (1 - m) - s) [C(F_1, G_0) +
        C(F_0, G_1)] + ((1 - m)^2 + s) C(F_0, G_0) + s dF dG`` with
        ``dF = E[F_1] - E[F_0]`` and ``C = 0`` at constants. Every
        coefficient is non-negative, and for monotone functions every term
        is non-negative, so the sum has no cancellation.

        Args:
            f: Node id.
            means: ``E[p_v]`` for every support variable.
            variances: ``Var[p_v]`` with ``0 <= s_v <= m_v (1 - m_v)``.

        Returns:
            Mean, complement mean, and variance.

        Raises:
            KeyError: If a support variable is missing.
            ValueError: If a mean or variance is invalid.

        Equation:
            [EQ-SYS-MOM-001].

        References:
            [REF-APOSTOLAKIS-KAPLAN-1981], [REF-RAUZY-1993].

        """
        nodes = self._reachable(f)
        stats = self._moment_inputs(nodes, means, variances)
        weights = {level: (1.0 - m, m) for level, (m, _) in stats.items()}
        one, zero = self._values(nodes, weights)

        def delta(high: int, low: int) -> float:
            """Return ``E[F_1] - E[F_0]`` from the smaller-magnitude table.

            References:
                [EQ-SYS-MOM-001]. No further equation is evaluated.

            """
            if zero[low] + zero[high] < one[low] + one[high]:
                return zero[low] - zero[high]
            return one[high] - one[low]

        def leaf(key: tuple[int, int]) -> float | None:
            """Return zero covariance with a constant.

            References:
                [EQ-SYS-MOM-001]. No further equation is evaluated.

            """
            return 0.0 if key[0] < 2 or key[1] < 2 else None

        def children(key: tuple[int, int]) -> tuple[tuple[int, int], ...]:
            """Return the four cofactor pairs.

            References:
                [EQ-SYS-MOM-001]. No further equation is evaluated.

            """
            a, b = key
            level = min(self._level[a], self._level[b])
            a0, a1 = self._cofactors(a, level)
            b0, b1 = self._cofactors(b, level)
            return (_pair(a1, b1), _pair(a1, b0), _pair(a0, b1), _pair(a0, b0))

        def combine(key: tuple[int, int], values: list[float]) -> float:
            """Apply the covariance recursion.

            References:
                [EQ-SYS-MOM-001]. No further equation is evaluated.

            """
            a, b = key
            level = min(self._level[a], self._level[b])
            m, s = stats[level]
            a0, a1 = self._cofactors(a, level)
            b0, b1 = self._cofactors(b, level)
            c11, c10, c01, c00 = values
            mixed = max(0.0, m * (1.0 - m) - s)
            return math.fsum(
                (
                    (m * m + s) * c11,
                    mixed * (c10 + c01),
                    ((1.0 - m) * (1.0 - m) + s) * c00,
                    s * delta(a1, a0) * delta(b1, b0),
                )
            )

        memo: dict[tuple[int, int], float] = {}
        variance = _postorder((f, f), leaf, children, combine, memo)
        return PolynomialMoments(_clip(one[f]), _clip(zero[f]), max(0.0, variance))

    def _moment_inputs(
        self,
        nodes: Sequence[int],
        means: Mapping[str, float],
        variances: Mapping[str, float],
    ) -> dict[int, tuple[float, float]]:
        """Validate ``(mean, variance)`` per level used by ``nodes``.

        Raises:
            KeyError: If a support variable is missing.
            ValueError: If a mean is not a probability or a variance lies
                outside ``[0, m (1 - m)]``.

        References:
            [EQ-SYS-MOM-001], [REF-KOLMOGOROV-1956].

        """
        stats: dict[int, tuple[float, float]] = {}
        for node in nodes:
            level = self._level[node]
            if level in stats:
                continue
            name = self._names[level]
            if name not in means or name not in variances:
                raise KeyError(f"no mean or variance for variable {name!r}")
            stats[level] = _mean_variance(means[name], variances[name], name=name)
        return stats

    def grouped_moments(
        self,
        f: int,
        blocks: Mapping[str, str],
        mixed_moments: Mapping[str, MixedMoment],
    ) -> PolynomialMoments:
        """Return the exact mean and variance with shared uncertain values.

        Every support variable belongs to a block. All variables of a block
        have the *same* random probability ``q_B`` (a block of one variable
        is an ordinary independent uncertain probability); different blocks
        are independent. ``R`` is then a polynomial in each ``q_B`` of degree
        up to the block size. Each block must occupy consecutive levels.
        Working bottom-up, a node inside block ``B`` carries its value as a
        polynomial ``sum c_jk q_B^j (1 - q_B)^k`` with non-negative
        coefficients; a child in a deeper block depends on no ``q_B`` and
        enters as its expectation; the expectation over ``q_B`` uses the
        mixed moments ``E[q_B^j (1 - q_B)^k]``. ``E[R^2]`` uses the same
        scheme on pairs of nodes. The variance is
        ``E[g^2] - E[g]^2`` for ``g`` the function or its complement,
        whichever has the smaller mean (the subtraction is the only place
        where rounding can be amplified).

        Args:
            f: Node id.
            blocks: Block label of every support variable.
            mixed_moments: Per block, ``(j, k) -> E[q^j (1 - q)^k]`` for
                ``j + k`` up to twice the block size.

        Returns:
            Mean, complement mean, and variance.

        Raises:
            KeyError: If a support variable has no block or a block has no
                moment function.
            ValueError: If a block's levels are not consecutive.

        Equation:
            [EQ-SYS-MOM-002].

        References:
            [REF-APOSTOLAKIS-KAPLAN-1981], [REF-RAUZY-1993].

        """
        nodes = self._reachable(f)
        block_of_level: dict[int, str] = {}
        for node in nodes:
            level = self._level[node]
            name = self._names[level]
            if name not in blocks:
                raise KeyError(f"variable {name!r} has no block")
            block_of_level[level] = blocks[name]
        for label in set(block_of_level.values()):
            if label not in mixed_moments:
                raise KeyError(f"block {label!r} has no moment function")
            levels = sorted(
                self._levels[n]
                for n, b in blocks.items()
                if b == label and n in self._levels
            )
            if levels and levels[-1] - levels[0] + 1 != len(levels):
                raise ValueError(f"block {label!r} does not occupy consecutive levels")

        moment_cache: dict[tuple[str, int, int], float] = {}

        def expect(label: str | None, poly: Polynomial) -> float:
            """Integrate a block polynomial over its shared value.

            References:
                [EQ-SYS-MOM-002]. No further equation is evaluated.

            """
            if label is None:
                return poly.get((0, 0), 0.0)
            terms = []
            for (j, k), coefficient in poly.items():
                key = (label, j, k)
                moment = moment_cache.get(key)
                if moment is None:
                    moment = float(mixed_moments[label](j, k))
                    moment_cache[key] = moment
                terms.append(coefficient * moment)
            return math.fsum(terms)

        def block_of(node: int) -> str | None:
            """Return the block of a node (``None`` for terminals).

            References:
                [EQ-SYS-MOM-002]. No further equation is evaluated.

            """
            return None if node < 2 else block_of_level[self._level[node]]

        def single(g: int) -> tuple[dict[int, Polynomial], dict[int, float]]:
            """Return per-node block polynomials and expectations of ``g``.

            References:
                [EQ-SYS-MOM-002]. No further equation is evaluated.

            """
            polys: dict[int, Polynomial] = {0: {}, 1: {(0, 0): 1.0}}
            values: dict[int, float] = {0: 0.0, 1: 1.0}
            for node in reversed(self._reachable(g)):
                label = block_of(node)
                poly: Polynomial = {}
                for child, shift in (
                    (self._low[node], (0, 1)),
                    (self._high[node], (1, 0)),
                ):
                    if child >= 2 and block_of(child) == label:
                        source = polys[child]
                    else:
                        source = {(0, 0): values[child]} if values[child] else {}
                    _add_shifted(poly, source, shift, 1.0)
                polys[node] = poly
                values[node] = expect(label, poly)
            return polys, values

        def second_moment(g: int) -> tuple[float, float]:
            """Return ``E[g]`` and ``E[g^2]``.

            References:
                [EQ-SYS-MOM-002]. No further equation is evaluated.

            """
            polys, values = single(g)
            pair_memo: dict[tuple[int, int], tuple[str | None, Polynomial, float]] = {}

            def pair_leaf(
                key: tuple[int, int],
            ) -> tuple[str | None, Polynomial, float] | None:
                """Return pairs that involve a constant.

                References:
                    [EQ-SYS-MOM-002]. No further equation is evaluated.

                """
                a, b = key
                if a == 0 or b == 0:
                    return (None, {}, 0.0)
                if a == 1:
                    return (block_of(b), polys[b], values[b])
                if b == 1:
                    return (block_of(a), polys[a], values[a])
                return None

            def pair_children(key: tuple[int, int]) -> tuple[tuple[int, int], ...]:
                """Return the four cofactor pairs.

                References:
                    [EQ-SYS-MOM-002]. No further equation is evaluated.

                """
                a, b = key
                level = min(self._level[a], self._level[b])
                a0, a1 = self._cofactors(a, level)
                b0, b1 = self._cofactors(b, level)
                return (_pair(a1, b1), _pair(a1, b0), _pair(a0, b1), _pair(a0, b0))

            def pair_combine(
                key: tuple[int, int],
                kids: list[tuple[str | None, Polynomial, float]],
            ) -> tuple[str | None, Polynomial, float]:
                """Multiply out ``q^2, q (1 - q), (1 - q)^2`` branches.

                References:
                    [EQ-SYS-MOM-002]. No further equation is evaluated.

                """
                a, b = key
                level = min(self._level[a], self._level[b])
                label = block_of_level[level]
                poly: Polynomial = {}
                for (kid_label, kid_poly, kid_value), shift in zip(
                    kids, ((2, 0), (1, 1), (1, 1), (0, 2)), strict=True
                ):
                    if kid_label == label:
                        source = kid_poly
                    else:
                        source = {(0, 0): kid_value} if kid_value else {}
                    _add_shifted(poly, source, shift, 1.0)
                return (label, poly, expect(label, poly))

            result = _postorder(
                (g, g), pair_leaf, pair_children, pair_combine, pair_memo
            )
            return values[g], result[2]

        mean_one, square_one = second_moment(f)
        complement = self.negate(f)
        mean_zero, square_zero = second_moment(complement)
        if mean_one <= mean_zero:
            variance = square_one - mean_one * mean_one
        else:
            variance = square_zero - mean_zero * mean_zero
        return PolynomialMoments(_clip(mean_one), _clip(mean_zero), max(0.0, variance))


def _pair(a: int, b: int) -> tuple[int, int]:
    """Return an unordered node pair as a sorted tuple.

    References:
        [EQ-SYS-MOM-001]. No further equation is evaluated.

    """
    return (a, b) if a <= b else (b, a)


def _add_shifted(
    target: Polynomial,
    source: Mapping[tuple[int, int], float],
    shift: tuple[int, int],
    scale: float,
) -> None:
    """Add ``scale q^dj (1 - q)^dk * source`` to ``target`` in place.

    References:
        [EQ-SYS-MOM-002]. No further equation is evaluated.

    """
    dj, dk = shift
    for (j, k), coefficient in source.items():
        key = (j + dj, k + dk)
        target[key] = target.get(key, 0.0) + scale * coefficient


def _clip(value: float) -> float:
    """Clip a probability to ``[0, 1]`` against rounding.

    References:
        [EQ-PROB-001], [REF-KOLMOGOROV-1956].

    """
    return min(1.0, max(0.0, value))


def _probability(value: object, *, name: str) -> float:
    """Validate a probability (rejecting booleans).

    References:
        [EQ-PROB-001], [REF-KOLMOGOROV-1956].

    """
    return require_probability(value, name=name)


def _mean_variance(mean: object, variance: object, *, name: str) -> tuple[float, float]:
    """Validate the mean and variance of an uncertain probability.

    Raises:
        ValueError: If the mean is not in ``[0, 1]`` or the variance is
            negative or above ``m (1 - m)`` (with a ``1e-12`` relative
            allowance for rounding).

    References:
        [EQ-SYS-MOM-001], [REF-KOLMOGOROV-1956].

    """
    m = _probability(mean, name=f"mean[{name!r}]")
    s = require_non_negative_finite_real(variance, name=f"variance[{name!r}]")
    bound = m * (1.0 - m)
    if s > bound * (1.0 + 1e-12) + 1e-300:
        raise ValueError(
            f"variance of {name!r} exceeds m (1 - m) = {bound}, the largest "
            "variance of a quantity in [0, 1] with that mean"
        )
    return m, min(s, bound)
