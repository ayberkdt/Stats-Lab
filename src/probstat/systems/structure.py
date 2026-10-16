"""Success-space structure expressions and exact structure analysis.

A system is described by a *structure expression* built from named
components: :func:`series` (all items must work), :func:`parallel` (at least
one item must work), :func:`k_out_of_n` (at least ``k`` items must work; the
items may be arbitrary, non-identical sub-structures) and :func:`bridge`.
A component name may appear in several branches; every occurrence denotes the
same physical component (a *repeated event*). Fault-tree logic converts with
:func:`from_fault_tree`, whose top event is system failure.

:class:`SystemStructure` compiles the expression into a reduced ordered BDD
(:class:`~probstat.systems.bdd.BDD`) and evaluates, exactly and under
independent component states, the system reliability and unreliability,
minimal path and cut sets (Rauzy's minimal-solution algorithm), the
Esary-Proschan and min-path/min-cut bounds, and the exact epistemic mean and
variance of the reliability when component reliabilities are uncertain.

Every expression is monotone (a working component never makes the system
fail), so the structure is *semi-coherent*; it is coherent when every
component is relevant (see :attr:`SystemStructure.irrelevant_components`).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias

from probstat.foundations.validation import require_probability
from probstat.risk.fault_tree import BasicEvent, FaultTree, Gate

from .bdd import BDD, MixedMoment, PolynomialMoments, _mean_variance


def _require_name(value: object) -> str:
    """Validate a component name.

    Args:
        value: Candidate name.

    Returns:
        The name.

    Raises:
        TypeError: If the name is not a string.
        ValueError: If the name is blank.

    References:
        Project validation policy. No equation is evaluated.

    """
    if not isinstance(value, str):
        raise TypeError("component names must be strings")
    if not value.strip():
        raise ValueError("component names must not be empty")
    return value


@dataclass(frozen=True, slots=True)
class Component:
    """A named component; ``x = 1`` means the component works.

    Args:
        name: Identifier; equal names denote the same component.

    References:
        [REF-BARLOW-PROSCHAN-1975].

    """

    name: str

    def __post_init__(self) -> None:
        """Validate the name.

        References:
            Project validation policy. No equation is evaluated.

        """
        _require_name(self.name)


@dataclass(frozen=True, slots=True)
class Series:
    """A series arrangement: works when every item works.

    Args:
        items: Non-empty sub-structures.

    References:
        [REF-BARLOW-PROSCHAN-1975], [REF-RAUSAND-HOYLAND-2004].

    """

    items: tuple[Expression, ...]

    def __post_init__(self) -> None:
        """Validate the items, converting names to components.

        References:
            Project validation policy. No equation is evaluated.

        """
        object.__setattr__(self, "items", _items(tuple(self.items)))


@dataclass(frozen=True, slots=True)
class Parallel:
    """A parallel (active redundancy) arrangement: works when any item works.

    Args:
        items: Non-empty sub-structures.

    References:
        [REF-BARLOW-PROSCHAN-1975], [REF-RAUSAND-HOYLAND-2004].

    """

    items: tuple[Expression, ...]

    def __post_init__(self) -> None:
        """Validate the items, converting names to components.

        References:
            Project validation policy. No equation is evaluated.

        """
        object.__setattr__(self, "items", _items(tuple(self.items)))


@dataclass(frozen=True, slots=True)
class KOutOfN:
    """A k-out-of-n:G arrangement: works when at least ``k`` items work.

    Args:
        k: Required number of working items, ``1 <= k <= n``.
        items: Sub-structures (possibly non-identical).

    References:
        [REF-BARLOW-PROSCHAN-1975], [REF-RAUSAND-HOYLAND-2004].

    """

    k: int
    items: tuple[Expression, ...]

    def __post_init__(self) -> None:
        """Validate the items and the vote threshold.

        Raises:
            TypeError: If ``k`` is not an integer or an item is invalid.
            ValueError: If ``k`` is outside ``1..n``.

        References:
            Project validation policy. No equation is evaluated.

        """
        items = _items(tuple(self.items))
        if isinstance(self.k, bool) or not isinstance(self.k, int):
            raise TypeError("k must be an integer")
        if not 1 <= self.k <= len(items):
            raise ValueError("k must satisfy 1 <= k <= number of items")
        object.__setattr__(self, "items", items)


Expression: TypeAlias = Component | Series | Parallel | KOutOfN


def _items(raw: Sequence[object]) -> tuple[Expression, ...]:
    """Validate items, converting names to components.

    Raises:
        TypeError: If an item is not an expression or a name.
        ValueError: If no item is given.

    References:
        Project validation policy. No equation is evaluated.

    """
    if not raw:
        raise ValueError("a structure needs at least one item")
    result: list[Expression] = []
    for index, item in enumerate(raw):
        if isinstance(item, str):
            result.append(Component(item))
        elif isinstance(item, Component | Series | Parallel | KOutOfN):
            result.append(item)
        else:
            raise TypeError(f"items[{index}] must be a structure expression or name")
    return tuple(result)


def series(*items: Expression | str) -> Series:
    """Return a series structure of the items.

    Args:
        *items: Sub-structures or component names.

    Returns:
        Series expression; ``R = prod R_i`` for independent disjoint items.

    Equation:
        [EQ-SYS-STRUCT-001].

    References:
        [REF-BARLOW-PROSCHAN-1975].

    """
    return Series(_items(items))


def parallel(*items: Expression | str) -> Parallel:
    """Return a parallel structure of the items.

    Args:
        *items: Sub-structures or component names.

    Returns:
        Parallel expression; ``R = 1 - prod (1 - R_i)`` for independent
        disjoint items.

    Equation:
        [EQ-SYS-STRUCT-001].

    References:
        [REF-BARLOW-PROSCHAN-1975].

    """
    return Parallel(_items(items))


def k_out_of_n(k: int, *items: Expression | str) -> KOutOfN:
    """Return a k-out-of-n:G structure (at least ``k`` items must work).

    Args:
        k: Required number of working items.
        *items: Sub-structures or component names; they need not be
            identical.

    Returns:
        Voting expression.

    Raises:
        TypeError: If ``k`` is not an integer.
        ValueError: If ``k`` is outside ``1..n``.

    Equation:
        [EQ-SYS-STRUCT-002].

    References:
        [REF-BARLOW-PROSCHAN-1975], [REF-RAUSAND-HOYLAND-2004].

    """
    return KOutOfN(k, _items(items))


def bridge(
    a: Expression | str,
    b: Expression | str,
    c: Expression | str,
    d: Expression | str,
    e: Expression | str,
) -> Parallel:
    """Return the five-item bridge network.

    ``a`` and ``b`` leave the source, ``d`` and ``e`` enter the sink, ``a``
    feeds ``d`` and ``b`` feeds ``e``, and ``c`` bridges the two middle
    nodes. The minimal path sets are ``{a, d}``, ``{b, e}``, ``{a, c, e}``
    and ``{b, c, d}``.

    Args:
        a: Source-side upper item.
        b: Source-side lower item.
        c: Bridging item.
        d: Sink-side upper item.
        e: Sink-side lower item.

    Returns:
        The bridge as a parallel arrangement of its minimal paths.

    Equation:
        [EQ-SYS-STRUCT-003].

    References:
        [REF-BARLOW-PROSCHAN-1975], [REF-RAUSAND-HOYLAND-2004].

    """
    ia, ib, ic, id_, ie = _items((a, b, c, d, e))
    return Parallel(
        (
            Series((ia, id_)),
            Series((ib, ie)),
            Series((ia, ic, ie)),
            Series((ib, ic, id_)),
        )
    )


def from_fault_tree(tree: FaultTree) -> Expression:
    """Return the success-space structure of a static fault tree.

    Basic events become components (event occurs = component failed) and
    the top event is system failure. By De Morgan duality an OR gate becomes
    a series structure, an AND gate a parallel structure, and a
    ``k``-out-of-``n`` failure gate a ``(n - k + 1)``-out-of-``n`` success
    structure. Component reliabilities are ``1 - P(basic event)``.

    Args:
        tree: A :class:`~probstat.risk.FaultTree`.

    Returns:
        Structure expression whose failure is the top event.

    Raises:
        TypeError: If ``tree`` is not a fault tree.

    Equation:
        [EQ-SYS-STRUCT-004].

    References:
        [REF-VESELY-ETAL-1981], [REF-BARLOW-PROSCHAN-1975].

    """
    if not isinstance(tree, FaultTree):
        raise TypeError("tree must be a FaultTree")
    memo: dict[str, Expression] = {}

    def convert(node: BasicEvent | Gate) -> Expression:
        """Convert one fault-tree node, memoized by gate name.

        References:
            [EQ-SYS-STRUCT-004], [REF-VESELY-ETAL-1981].

        """
        if isinstance(node, BasicEvent):
            return Component(node.name)
        if node.name in memo:
            return memo[node.name]
        children = tuple(convert(child) for child in node.inputs)
        result: Expression
        if node.kind == "or":
            result = Series(children)
        elif node.kind == "and":
            result = Parallel(children)
        else:
            assert node.k is not None
            result = KOutOfN(len(children) - node.k + 1, children)
        memo[node.name] = result
        return result

    return convert(tree.top)


def expression_components(expression: Expression) -> tuple[str, ...]:
    """Return component names by first appearance in a left-to-right DFS.

    This is the default BDD variable order (the depth-first, left-most
    heuristic): components that appear together in the expression are
    tested consecutively, which keeps series/parallel modules compact.

    Args:
        expression: Structure expression.

    Returns:
        Distinct component names.

    References:
        [REF-RAUZY-1993], [REF-BRYANT-1986]. No equation is evaluated.

    """
    seen: dict[str, None] = {}
    stack: list[Expression] = [expression]
    while stack:
        node = stack.pop()
        if isinstance(node, Component):
            seen.setdefault(node.name, None)
        else:
            stack.extend(reversed(node.items))
    return tuple(seen)


def compile_expression(
    expression: Expression,
    bdd: BDD,
    rename: Callable[[str], str] | None = None,
) -> int:
    """Compile a structure expression into a BDD node (``1`` = works).

    Series and parallel structures are conjunctions and disjunctions. A
    k-out-of-n structure uses the threshold recursion
    ``T(i, k) = (f_i and T(i + 1, k - 1)) or T(i + 1, k)``, which is exact
    for monotone items and costs ``O(n k)`` BDD operations.

    Args:
        expression: Structure expression.
        bdd: Manager to build into.
        rename: Optional map from component name to BDD variable name.

    Returns:
        Node id of the structure function.

    Equation:
        [EQ-SYS-STRUCT-001], [EQ-SYS-STRUCT-002].

    References:
        [REF-BRYANT-1986], [REF-RAUZY-1993].

    """
    memo: dict[int, int] = {}

    def build(node: Expression) -> int:
        """Compile one sub-expression, memoized by object identity.

        References:
            [EQ-SYS-STRUCT-001], [REF-BRYANT-1986].

        """
        key = id(node)
        if key in memo:
            return memo[key]
        if isinstance(node, Component):
            result = bdd.variable(rename(node.name) if rename else node.name)
        elif isinstance(node, Series):
            result = bdd.conjunction(build(item) for item in node.items)
        elif isinstance(node, Parallel):
            result = bdd.disjunction(build(item) for item in node.items)
        else:
            parts = [build(item) for item in node.items]
            result = _threshold(bdd, parts, node.k)
        memo[key] = result
        return result

    return build(expression)


def _threshold(bdd: BDD, parts: Sequence[int], k: int) -> int:
    """Return ``at least k of parts hold`` for monotone ``parts``.

    References:
        [EQ-SYS-STRUCT-002], [REF-BARLOW-PROSCHAN-1975].

    """
    n = len(parts)
    # row[j] = "at least j of parts[i:] hold", built from i = n down to 0.
    row = [bdd.TRUE] + [bdd.FALSE] * k
    for i in range(n - 1, -1, -1):
        updated = [bdd.TRUE]
        for j in range(1, k + 1):
            updated.append(bdd.apply_or(bdd.apply_and(parts[i], row[j - 1]), row[j]))
        row = updated
    return row[k]


@dataclass(frozen=True, slots=True)
class ReliabilityBounds:
    """Exact reliability with classical cut/path-set bounds.

    Args:
        reliability: Exact system reliability.
        esary_proschan_lower: ``prod_cuts [1 - prod_{i in K} (1 - p_i)]``.
        esary_proschan_upper: ``1 - prod_paths [1 - prod_{i in P} p_i]``.
        min_path_lower: ``max_paths prod_{i in P} p_i``.
        min_cut_upper: ``min_cuts [1 - prod_{i in K} (1 - p_i)]``.

    References:
        [REF-ESARY-PROSCHAN-1963], [REF-BARLOW-PROSCHAN-1975].

    """

    reliability: float
    esary_proschan_lower: float
    esary_proschan_upper: float
    min_path_lower: float
    min_cut_upper: float


@dataclass(frozen=True, slots=True)
class ReliabilityMoments:
    """Epistemic mean and variance of the system reliability.

    Args:
        mean: ``E[R]`` over the uncertain component reliabilities.
        variance: ``Var[R]``.
        standard_deviation: ``sqrt(Var[R])``.
        unreliability_mean: ``E[1 - R]``, computed directly.
        shared_groups: Groups of components that share one uncertain value.
        beta_assumed_groups: Shared groups whose higher moments were taken
            from the beta law matching the given mean and variance.

    References:
        [REF-APOSTOLAKIS-KAPLAN-1981].

    """

    mean: float
    variance: float
    standard_deviation: float
    unreliability_mean: float
    shared_groups: tuple[str, ...] = ()
    beta_assumed_groups: tuple[str, ...] = ()


class SystemStructure:
    """A compiled structure function with exact reliability analysis.

    Args:
        expression: Structure expression (or a single component name).
        order: Optional BDD variable order; listed components come first,
            the rest follow in depth-first order of appearance.

    Raises:
        TypeError: If the expression is invalid.
        ValueError: If ``order`` names unknown or repeated components.

    References:
        [REF-BARLOW-PROSCHAN-1975], [REF-RAUZY-1993], [REF-BRYANT-1986].

    """

    def __init__(
        self, expression: Expression | str, *, order: Sequence[str] | None = None
    ) -> None:
        """Compile the expression.

        References:
            [REF-BRYANT-1986]. No equation is evaluated.

        """
        if isinstance(expression, str):
            expression = Component(expression)
        if not isinstance(expression, Component | Series | Parallel | KOutOfN):
            raise TypeError("expression must be a structure expression")
        self._expression: Expression = expression
        found = expression_components(expression)
        chosen: list[str] = []
        for name in order or ():
            if name not in found:
                raise ValueError(f"order names unknown component {name!r}")
            if name in chosen:
                raise ValueError(f"order repeats component {name!r}")
            chosen.append(name)
        chosen.extend(name for name in found if name not in chosen)
        self._components = tuple(chosen)
        self._bdd = BDD(self._components)
        self._root = compile_expression(expression, self._bdd)
        self._path_family: int | None = None
        self._cut_family: int | None = None
        self._failure_root: int | None = None

    @property
    def expression(self) -> Expression:
        """Return the structure expression.

        References:
            [REF-BARLOW-PROSCHAN-1975].

        """
        return self._expression

    @property
    def components(self) -> tuple[str, ...]:
        """Return the component names in BDD variable order.

        References:
            [REF-BARLOW-PROSCHAN-1975].

        """
        return self._components

    @property
    def bdd(self) -> BDD:
        """Return the BDD manager (variables are component names).

        References:
            [REF-BRYANT-1986].

        """
        return self._bdd

    @property
    def root(self) -> int:
        """Return the BDD node of the structure function (``1`` = works).

        References:
            [REF-BRYANT-1986].

        """
        return self._root

    @property
    def failure_root(self) -> int:
        """Return the BDD node of the failure function in failed-state literals.

        Its variable ``y_i = 1`` means component ``i`` failed; it is the dual
        of the structure function.

        References:
            [REF-BARLOW-PROSCHAN-1975], [REF-RAUZY-1993].

        """
        if self._failure_root is None:
            self._failure_root = self._bdd.dual(self._root)
        return self._failure_root

    @property
    def irrelevant_components(self) -> tuple[str, ...]:
        """Return components the structure function does not depend on.

        References:
            [REF-BARLOW-PROSCHAN-1975].

        """
        support = set(self._bdd.support(self._root))
        return tuple(name for name in self._components if name not in support)

    @property
    def node_count(self) -> int:
        """Return the number of internal BDD nodes of the structure function.

        References:
            [REF-BRYANT-1986].

        """
        return self._bdd.node_count(self._root)

    def structure_function(self, states: Mapping[str, bool | int]) -> int:
        """Return ``phi(x)``: ``1`` if the system works in a state vector.

        Args:
            states: Component states (``True``/``1`` = works).

        Returns:
            ``1`` or ``0``.

        Raises:
            KeyError: If a relevant component is missing or a name is unknown.
            ValueError: If a state is not boolean or ``0``/``1``.

        Equation:
            [EQ-SYS-BDD-001].

        References:
            [REF-BARLOW-PROSCHAN-1975].

        """
        for name in states:
            if name not in self._components:
                raise KeyError(f"unknown component {name!r}")
        return int(self._bdd.evaluate(self._root, states))

    def probabilities(
        self,
        reliabilities: Mapping[str, float],
        unreliabilities: Mapping[str, float] | None = None,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Validate component reliabilities and their complements.

        Args:
            reliabilities: ``P(component works)`` for every component.
            unreliabilities: Optional accurate ``P(component failed)``.

        Returns:
            ``(p, q)`` tables keyed by component.

        Raises:
            KeyError: If a component is missing or a name is unknown.
            ValueError: If a value is invalid or ``p + q`` is not one.

        References:
            [EQ-PROB-001], [REF-KOLMOGOROV-1956].

        """
        p: dict[str, float] = {}
        q: dict[str, float] = {}
        for table in (reliabilities, unreliabilities or {}):
            for name in table:
                if name not in self._components:
                    raise KeyError(f"unknown component {name!r}")
        for name in self._components:
            if name not in reliabilities:
                raise KeyError(f"no reliability for component {name!r}")
            p[name] = require_probability(reliabilities[name], name=f"p[{name!r}]")
            if unreliabilities is None:
                q[name] = 1.0 - p[name]
            else:
                if name not in unreliabilities:
                    raise KeyError(f"no unreliability for component {name!r}")
                q[name] = require_probability(
                    unreliabilities[name], name=f"q[{name!r}]"
                )
                if abs(p[name] + q[name] - 1.0) > 1e-9:
                    raise ValueError(
                        f"reliability and unreliability of {name!r} differ"
                    )
        return p, q

    def reliability(
        self,
        reliabilities: Mapping[str, float],
        *,
        unreliabilities: Mapping[str, float] | None = None,
    ) -> float:
        """Return the exact system reliability for independent components.

        Args:
            reliabilities: ``P(component works)`` for every component.
            unreliabilities: Optional accurate complements.

        Returns:
            ``P(phi(X) = 1)``.

        Equation:
            [EQ-SYS-BDD-001].

        References:
            [REF-RAUZY-1993], [REF-BARLOW-PROSCHAN-1975].

        """
        p, q = self.probabilities(reliabilities, unreliabilities)
        return self._bdd.probability(self._root, p, q, value=True)

    def unreliability(
        self,
        reliabilities: Mapping[str, float],
        *,
        unreliabilities: Mapping[str, float] | None = None,
    ) -> float:
        """Return the exact system unreliability ``P(phi(X) = 0)``.

        It is evaluated directly (not as ``1 - R``), so small failure
        probabilities keep full relative accuracy when ``unreliabilities``
        are supplied.

        Args:
            reliabilities: ``P(component works)`` for every component.
            unreliabilities: Optional accurate complements.

        Returns:
            System failure probability.

        Equation:
            [EQ-SYS-BDD-001].

        References:
            [REF-RAUZY-1993].

        """
        p, q = self.probabilities(reliabilities, unreliabilities)
        return self._bdd.probability(self._root, p, q, value=False)

    def path_family(self) -> int:
        """Return the BDD family of minimal path sets.

        References:
            [EQ-SYS-MINSOL-001], [REF-RAUZY-1993].

        """
        if self._path_family is None:
            self._path_family = self._bdd.minimal_solution_family(self._root)
        return self._path_family

    def cut_family(self) -> int:
        """Return the BDD family of minimal cut sets.

        References:
            [EQ-SYS-MINSOL-001], [REF-RAUZY-1993].

        """
        if self._cut_family is None:
            self._cut_family = self._bdd.minimal_solution_family(self.failure_root)
        return self._cut_family

    def minimal_path_sets(
        self, *, limit: int = 1_000_000
    ) -> tuple[frozenset[str], ...]:
        """Return the minimal path sets, sorted by size then names.

        Args:
            limit: Enumeration guard.

        Returns:
            Minimal sets of components whose joint functioning makes the
            system work.

        Raises:
            ValueError: If there are more than ``limit`` path sets.

        Equation:
            [EQ-SYS-MINSOL-001].

        References:
            [REF-RAUZY-1993], [REF-BARLOW-PROSCHAN-1975].

        """
        return self._bdd.family_sets(self.path_family(), limit=limit)

    def minimal_cut_sets(self, *, limit: int = 1_000_000) -> tuple[frozenset[str], ...]:
        """Return the minimal cut sets, sorted by size then names.

        Args:
            limit: Enumeration guard.

        Returns:
            Minimal sets of components whose joint failure fails the system.

        Raises:
            ValueError: If there are more than ``limit`` cut sets.

        Equation:
            [EQ-SYS-MINSOL-001].

        References:
            [REF-RAUZY-1993], [REF-BARLOW-PROSCHAN-1975].

        """
        return self._bdd.family_sets(self.cut_family(), limit=limit)

    def count_minimal_cut_sets(self) -> int:
        """Return the number of minimal cut sets without enumerating them.

        References:
            [REF-RAUZY-1993]. No equation is evaluated.

        """
        return self._bdd.family_count(self.cut_family())

    def count_minimal_path_sets(self) -> int:
        """Return the number of minimal path sets without enumerating them.

        References:
            [REF-RAUZY-1993]. No equation is evaluated.

        """
        return self._bdd.family_count(self.path_family())

    def bounds(
        self,
        reliabilities: Mapping[str, float],
        *,
        unreliabilities: Mapping[str, float] | None = None,
        limit: int = 1_000_000,
    ) -> ReliabilityBounds:
        """Return the exact reliability with Esary-Proschan and min-max bounds.

        For independent (more generally, associated) components of a
        coherent system,
        ``prod_j [1 - prod_{i in K_j} q_i] <= R <= 1 - prod_r [1 - prod_{i in
        P_r} p_i]`` (Esary-Proschan), and
        ``max_r prod_{i in P_r} p_i <= R <= min_j [1 - prod_{i in K_j} q_i]``
        (min-path / min-cut bounds). The Esary-Proschan lower bound is the
        fault-tree *min-cut upper bound* on the unreliability.

        Args:
            reliabilities: Component reliabilities.
            unreliabilities: Optional accurate complements.
            limit: Enumeration guard on cut and path sets.

        Returns:
            Exact value and the four bounds.

        Equation:
            [EQ-SYS-BOUND-001], [EQ-SYS-BOUND-002].

        References:
            [REF-ESARY-PROSCHAN-1963], [REF-BARLOW-PROSCHAN-1975],
            [REF-ESARY-PROSCHAN-WALKUP-1967].

        """
        p, q = self.probabilities(reliabilities, unreliabilities)
        exact = self._bdd.probability(self._root, p, q)
        paths = self.minimal_path_sets(limit=limit)
        cuts = self.minimal_cut_sets(limit=limit)
        path_probabilities = [math.prod(p[i] for i in path) for path in paths]
        cut_probabilities = [math.prod(q[i] for i in cut) for cut in cuts]
        return ReliabilityBounds(
            reliability=exact,
            esary_proschan_lower=_product_of_complements(cut_probabilities),
            esary_proschan_upper=1.0 - _product_of_complements(path_probabilities),
            min_path_lower=max(path_probabilities, default=0.0),
            min_cut_upper=1.0 - max(cut_probabilities, default=0.0),
        )

    def reliability_moments(
        self,
        means: Mapping[str, float],
        variances: Mapping[str, float],
        *,
        shared: Mapping[str, str] | None = None,
        shared_moments: Mapping[str, Sequence[float]] | None = None,
    ) -> ReliabilityMoments:
        """Return the exact epistemic mean and variance of the reliability.

        Without ``shared`` the component reliabilities are independent
        uncertain quantities with the given means and variances, and the
        result is exact (pairwise BDD covariance recursion).

        Identical components whose reliability is estimated from the same
        data share one uncertain value (state-of-knowledge correlation,
        Apostolakis and Kaplan 1981); treating them as independent
        understates the mean unreliability of redundant structures. List
        such components in ``shared`` (component -> group label). All members
        of a group must have equal mean and variance. ``R`` is then a
        polynomial of degree up to the group size in the shared value, and
        its moments need the raw moments ``E[q^k]`` up to twice the group
        size. Give them in ``shared_moments`` (group -> ``(E[q], E[q^2],
        ...)``); for a group without them the beta law with the given mean
        and variance is assumed, and the group is reported in
        ``beta_assumed_groups``. With explicit raw moments the mixed moments
        ``E[q^j (1 - q)^k]`` are formed by an alternating binomial sum,
        which loses accuracy for very large groups.

        Args:
            means: ``E[p_i]`` for every component.
            variances: ``Var[p_i]`` for every component.
            shared: Optional component -> group label.
            shared_moments: Optional group -> raw moments ``E[q^k]``,
                ``k = 1, 2, ...``.

        Returns:
            Mean, variance, standard deviation, and mean unreliability.

        Raises:
            KeyError: If a component is missing or unknown.
            ValueError: If moments are invalid, group members disagree, or too
                few raw moments are given.

        Equation:
            [EQ-SYS-MOM-001], [EQ-SYS-MOM-002].

        References:
            [REF-APOSTOLAKIS-KAPLAN-1981], [REF-RAUZY-1993].

        """
        for table in (means, variances):
            for name in table:
                if name not in self._components:
                    raise KeyError(f"unknown component {name!r}")
        if not shared:
            result = self._bdd.moments(self._root, means, variances)
            return _moments_result(result, (), ())
        return self._shared_moments(means, variances, shared, shared_moments or {})

    def _shared_moments(
        self,
        means: Mapping[str, float],
        variances: Mapping[str, float],
        shared: Mapping[str, str],
        shared_moments: Mapping[str, Sequence[float]],
    ) -> ReliabilityMoments:
        """Evaluate moments with shared groups on a group-contiguous order.

        References:
            [EQ-SYS-MOM-002], [REF-APOSTOLAKIS-KAPLAN-1981].

        """
        members: dict[str, list[str]] = {}
        for name, group in shared.items():
            if name not in self._components:
                raise KeyError(f"unknown component {name!r}")
            if not isinstance(group, str) or not group:
                raise TypeError("group labels must be non-empty strings")
            members.setdefault(group, []).append(name)
        for group in shared_moments:
            if group not in members:
                raise KeyError(f"shared_moments names unknown group {group!r}")
        order: list[str] = []
        for name in self._components:
            if name in order:
                continue
            label = shared.get(name)
            if label is None:
                order.append(name)
            else:
                order.extend(m for m in self._components if shared.get(m) == label)
        blocks: dict[str, str] = {}
        laws: dict[str, MixedMoment] = {}
        assumed: list[str] = []
        for name in self._components:
            if name not in means or name not in variances:
                raise KeyError(f"no mean or variance for component {name!r}")
            group_label = shared.get(name)
            if group_label is None:
                label = f"\x00{name}"
                blocks[name] = label
                m, s = _mean_variance(means[name], variances[name], name=name)
                laws[label] = _beta_mixed_moment(m, s)
                continue
            blocks[name] = group_label
            if group_label in laws:
                continue
            stats = {
                _mean_variance(means[m], variances[m], name=m)
                for m in members[group_label]
            }
            if len(stats) != 1:
                raise ValueError(f"members of group {group_label!r} need equal moments")
            m, s = stats.pop()
            if group_label in shared_moments:
                laws[group_label] = _raw_mixed_moment(
                    m,
                    s,
                    shared_moments[group_label],
                    2 * len(members[group_label]),
                    group_label,
                )
            else:
                laws[group_label] = _beta_mixed_moment(m, s)
                assumed.append(group_label)
        bdd = BDD(order)
        root = compile_expression(self._expression, bdd)
        result = bdd.grouped_moments(root, blocks, laws)
        return _moments_result(result, tuple(sorted(members)), tuple(sorted(assumed)))


def _moments_result(
    moments: PolynomialMoments, groups: tuple[str, ...], assumed: tuple[str, ...]
) -> ReliabilityMoments:
    """Wrap BDD polynomial moments as a reliability result.

    References:
        [EQ-SYS-MOM-001]. No further equation is evaluated.

    """
    return ReliabilityMoments(
        mean=moments.mean,
        variance=moments.variance,
        standard_deviation=math.sqrt(moments.variance),
        unreliability_mean=moments.complement_mean,
        shared_groups=groups,
        beta_assumed_groups=assumed,
    )


def _product_of_complements(probabilities: Sequence[float]) -> float:
    """Return ``prod (1 - a_j)`` accurately (``log1p`` summation).

    References:
        [EQ-SYS-BOUND-001], [REF-PYTHON-MATH].

    """
    if any(value >= 1.0 for value in probabilities):
        return 0.0
    return math.exp(math.fsum(math.log1p(-value) for value in probabilities))


def _beta_mixed_moment(mean: float, variance: float) -> MixedMoment:
    """Return ``(j, k) -> E[q^j (1 - q)^k]`` for the matching beta law.

    The beta law with mean ``m`` and variance ``s`` has
    ``alpha = m nu``, ``beta = (1 - m) nu``, ``nu = m (1 - m) / s - 1``, and
    ``E[q^j (1 - q)^k] = prod_{i<j} (alpha + i) prod_{i<k} (beta + i) /
    prod_{i<j+k} (alpha + beta + i)``. The degenerate cases ``s = 0`` (a
    point mass) and ``s = m (1 - m)`` (a Bernoulli law) are handled exactly.

    References:
        [EQ-SYS-MOM-002], [REF-APOSTOLAKIS-KAPLAN-1981].

    """
    bound = mean * (1.0 - mean)
    if variance <= 0.0 or bound == 0.0:

        def point(j: int, k: int) -> float:
            """Return mixed moments of a point mass.

            References:
                [EQ-SYS-MOM-002]. No further equation is evaluated.

            """
            return mean**j * (1.0 - mean) ** k

        return point
    if variance >= bound:

        def bernoulli(j: int, k: int) -> float:
            """Return mixed moments of a Bernoulli law.

            References:
                [EQ-SYS-MOM-002]. No further equation is evaluated.

            """
            if j and k:
                return 0.0
            if j:
                return mean
            if k:
                return 1.0 - mean
            return 1.0

        return bernoulli
    nu = bound / variance - 1.0
    alpha, beta = mean * nu, (1.0 - mean) * nu

    def moment(j: int, k: int) -> float:
        """Return the beta mixed moment.

        References:
            [EQ-SYS-MOM-002]. No further equation is evaluated.

        """
        value = 1.0
        for i in range(j):
            value *= (alpha + i) / (alpha + beta + i)
        for i in range(k):
            value *= (beta + i) / (alpha + beta + j + i)
        return value

    return moment


def _raw_mixed_moment(
    mean: float, variance: float, raw: Sequence[float], order: int, group: str
) -> MixedMoment:
    """Return mixed moments from raw moments ``E[q^k]``.

    ``E[q^j (1 - q)^k] = sum_i C(k, i) (-1)^i E[q^(j + i)]``.

    Raises:
        ValueError: If fewer than ``order`` raw moments are given, a moment
            is not in ``[0, 1]``, or the first two disagree with the mean and
            variance.

    References:
        [EQ-SYS-MOM-002]. No further equation is evaluated.

    """
    values = [1.0] + [
        require_probability(value, name=f"shared_moments[{group!r}]") for value in raw
    ]
    if len(values) - 1 < order:
        raise ValueError(
            f"group {group!r} needs raw moments up to order {order}, "
            f"got {len(values) - 1}"
        )
    if abs(values[1] - mean) > 1e-9 or abs(values[2] - mean * mean - variance) > 1e-9:
        raise ValueError(f"raw moments of group {group!r} disagree with mean/variance")

    def moment(j: int, k: int) -> float:
        """Return the alternating-sum mixed moment.

        References:
            [EQ-SYS-MOM-002]. No further equation is evaluated.

        """
        return max(
            0.0,
            math.fsum(
                math.comb(k, i) * (-1) ** i * values[j + i] for i in range(k + 1)
            ),
        )

    return moment
