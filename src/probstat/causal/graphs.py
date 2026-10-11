"""Causal directed acyclic graphs, d-separation, and graphical identification.

A :class:`CausalGraph` is a validated directed acyclic graph whose edges read
"is a direct cause of". The module answers three graphical questions:

- **d-separation** ``X _||_ Y | Z`` by the moralized ancestral graph
  (EQ-CAUSAL-DSEP-001): restrict to the ancestors of ``X u Y u Z``, marry
  co-parents, drop directions, delete ``Z``, and test reachability.
- **Backdoor criterion** (EQ-CAUSAL-BD-001) and an exhaustive search for all
  minimal adjustment sets on small graphs.
- **Frontdoor criterion** (EQ-CAUSAL-FDC-001).

Graph surgery for ``do(X = x)`` deletes every edge into ``X``
(:meth:`CausalGraph.do`). The library does not implement the complete
identification algorithm for arbitrary semi-Markovian models
[REF-SHPITSER-PEARL-2006]; only the backdoor and frontdoor routes are offered.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from itertools import combinations
from types import MappingProxyType

DEFAULT_MAX_CANDIDATES = 20


class CyclicGraphError(ValueError):
    """Raised when the supplied edges contain a directed cycle.

    References:
        [REF-PEARL-2009, Section 1.2]. No equation is evaluated.

    """


def _check_name(node: object) -> str:
    """Validate one node name.

    Args:
        node: Candidate node name.

    Returns:
        The name.

    Raises:
        TypeError: If the name is not a string.
        ValueError: If the name is empty.

    References:
        Project validation policy. No equation is evaluated.

    """
    if not isinstance(node, str):
        raise TypeError(f"node names must be strings, got {node!r}")
    if not node:
        raise ValueError("node names must be non-empty")
    return node


@dataclass(frozen=True, slots=True, init=False)
class CausalGraph:
    """An immutable, validated causal directed acyclic graph.

    Nodes are stored in a deterministic topological order (Kahn's algorithm
    with ties broken alphabetically) and edges are sorted, so two graphs with
    the same node and edge sets compare equal regardless of input order.

    Attributes:
        nodes: Node names in topological order.
        edges: Directed edges ``(parent, child)``.

    References:
        [REF-PEARL-2009, Sections 1.2 and 3.2],
        [REF-HERNAN-ROBINS-2020, Chapter 6].

    """

    nodes: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    _parents: Mapping[str, frozenset[str]] = field(repr=False, compare=False)
    _children: Mapping[str, frozenset[str]] = field(repr=False, compare=False)

    def __init__(
        self, edges: Iterable[tuple[str, str]] = (), nodes: Iterable[str] = ()
    ) -> None:
        """Validate edges and isolated nodes and fix a topological order.

        Args:
            edges: Directed edges ``(parent, child)``.
            nodes: Optional extra nodes, for example isolated variables.

        Raises:
            TypeError: If a node name is not a string or an edge is not a pair.
            ValueError: If a name is empty, an edge is a self-loop, or the
                graph has no nodes.
            CyclicGraphError: If the edges contain a directed cycle.

        References:
            [REF-PEARL-2009, Section 1.2]. No equation is evaluated.

        """
        names: set[str] = {_check_name(node) for node in nodes}
        edge_set: set[tuple[str, str]] = set()
        for edge in edges:
            if not isinstance(edge, tuple) or len(edge) != 2:
                raise TypeError(f"edges must be (parent, child) pairs, got {edge!r}")
            parent, child = _check_name(edge[0]), _check_name(edge[1])
            if parent == child:
                raise ValueError(f"self-loop on {parent!r} is not allowed")
            edge_set.add((parent, child))
            names.update((parent, child))
        if not names:
            raise ValueError("a causal graph needs at least one node")
        parents: dict[str, set[str]] = {name: set() for name in names}
        children: dict[str, set[str]] = {name: set() for name in names}
        for parent, child in edge_set:
            parents[child].add(parent)
            children[parent].add(child)
        indegree = {name: len(parents[name]) for name in names}
        ready = sorted(name for name in names if indegree[name] == 0)
        order: list[str] = []
        while ready:
            current = ready.pop(0)
            order.append(current)
            for child in sorted(children[current]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
            ready.sort()
        if len(order) != len(names):
            cyclic = sorted(name for name in names if indegree[name] > 0)
            raise CyclicGraphError(f"directed cycle among nodes {cyclic}")
        position = {name: index for index, name in enumerate(order)}
        object.__setattr__(self, "nodes", tuple(order))
        object.__setattr__(
            self,
            "edges",
            tuple(sorted(edge_set, key=lambda e: (position[e[0]], position[e[1]]))),
        )
        object.__setattr__(
            self,
            "_parents",
            MappingProxyType({k: frozenset(v) for k, v in parents.items()}),
        )
        object.__setattr__(
            self,
            "_children",
            MappingProxyType({k: frozenset(v) for k, v in children.items()}),
        )

    def _node_set(self, nodes: str | Iterable[str], *, name: str) -> frozenset[str]:
        """Normalize one name or an iterable of names to a checked set.

        Args:
            nodes: A node name or an iterable of node names.
            name: Argument name used in error messages.

        Returns:
            Frozen set of known node names.

        Raises:
            KeyError: If a node is not in the graph.

        References:
            Project validation policy. No equation is evaluated.

        """
        members = frozenset((nodes,) if isinstance(nodes, str) else nodes)
        unknown = sorted(str(node) for node in members - set(self.nodes))
        if unknown:
            raise KeyError(f"{name} contains unknown node(s) {unknown}")
        return members

    def parents(self, node: str) -> frozenset[str]:
        """Return the direct causes of a node.

        Args:
            node: Node name.

        Returns:
            Set of parents.

        Raises:
            KeyError: If the node is unknown.

        References:
            [REF-PEARL-2009, Section 1.2]. No equation is evaluated.

        """
        (member,) = self._node_set(node, name="node")
        return self._parents[member]

    def children(self, node: str) -> frozenset[str]:
        """Return the direct effects of a node.

        Args:
            node: Node name.

        Returns:
            Set of children.

        Raises:
            KeyError: If the node is unknown.

        References:
            [REF-PEARL-2009, Section 1.2]. No equation is evaluated.

        """
        (member,) = self._node_set(node, name="node")
        return self._children[member]

    def _reach(
        self, start: frozenset[str], step: Mapping[str, frozenset[str]]
    ) -> frozenset[str]:
        """Return every node reachable from ``start`` in one or more steps.

        Args:
            start: Starting nodes.
            step: Parent or child adjacency.

        Returns:
            Reachable nodes; a start node is included only when it is reached
            from another start node.

        References:
            [REF-PEARL-2009, Section 1.2]. No equation is evaluated.

        """
        seen: set[str] = set()
        stack = [neighbor for node in start for neighbor in step[node]]
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            stack.extend(step[current])
        return frozenset(seen)

    def ancestors(self, nodes: str | Iterable[str]) -> frozenset[str]:
        """Return the proper ancestors of a node or a set of nodes.

        Args:
            nodes: Node name or names.

        Returns:
            Nodes with a directed path of length at least one into some member.
            A member appears only if it is an ancestor of another member.

        Raises:
            KeyError: If a node is unknown.

        References:
            [REF-PEARL-2009, Section 1.2]. No equation is evaluated.

        """
        return self._reach(self._node_set(nodes, name="nodes"), self._parents)

    def descendants(self, nodes: str | Iterable[str]) -> frozenset[str]:
        """Return the proper descendants of a node or a set of nodes.

        Args:
            nodes: Node name or names.

        Returns:
            Nodes reachable by a directed path of length at least one.

        Raises:
            KeyError: If a node is unknown.

        References:
            [REF-PEARL-2009, Section 1.2]. No equation is evaluated.

        """
        return self._reach(self._node_set(nodes, name="nodes"), self._children)

    def ancestral_set(self, nodes: str | Iterable[str]) -> frozenset[str]:
        """Return the nodes together with all of their ancestors.

        Args:
            nodes: Node name or names.

        Returns:
            The smallest ancestral set containing the nodes.

        Raises:
            KeyError: If a node is unknown.

        References:
            [REF-LAURITZEN-ETAL-1990]. No equation is evaluated.

        """
        members = self._node_set(nodes, name="nodes")
        return members | self._reach(members, self._parents)

    def is_d_separated(
        self,
        x: str | Iterable[str],
        y: str | Iterable[str],
        given: str | Iterable[str] = (),
    ) -> bool:
        """Test whether ``given`` d-separates ``x`` from ``y``.

        Uses the moralized ancestral graph: ``X`` and ``Y`` are d-separated by
        ``Z`` exactly when ``Z`` separates them in the undirected moral graph
        of the ancestral set of ``X u Y u Z``.

        Args:
            x: First node or node set (non-empty).
            y: Second node or node set (non-empty).
            given: Conditioning node or set.

        Returns:
            ``True`` when every path between ``x`` and ``y`` is blocked.

        Raises:
            KeyError: If a node is unknown.
            ValueError: If ``x`` or ``y`` is empty or the three sets overlap.

        Equation:
            [EQ-CAUSAL-DSEP-001].

        References:
            [REF-PEARL-2009, Section 1.2.3], [REF-LAURITZEN-ETAL-1990].

        """
        xs = self._node_set(x, name="x")
        ys = self._node_set(y, name="y")
        zs = self._node_set(given, name="given")
        if not xs or not ys:
            raise ValueError("x and y must be non-empty")
        if xs & ys or xs & zs or ys & zs:
            raise ValueError("x, y, and given must be pairwise disjoint")
        relevant = self.ancestral_set(xs | ys | zs)
        adjacency: dict[str, set[str]] = {node: set() for node in relevant}
        for node in relevant:
            parent_list = sorted(self._parents[node])
            for parent in parent_list:
                adjacency[node].add(parent)
                adjacency[parent].add(node)
            for first, second in combinations(parent_list, 2):
                adjacency[first].add(second)
                adjacency[second].add(first)
        seen = set(xs)
        stack = list(xs)
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor in zs or neighbor in seen:
                    continue
                if neighbor in ys:
                    return False
                seen.add(neighbor)
                stack.append(neighbor)
        return True

    def do(self, nodes: str | Iterable[str]) -> CausalGraph:
        """Return the mutilated graph for an intervention on ``nodes``.

        Every edge pointing into an intervened node is removed; the nodes
        themselves are kept.

        Args:
            nodes: Intervened node or nodes.

        Returns:
            The graph ``G`` with incoming edges of ``nodes`` deleted.

        Raises:
            KeyError: If a node is unknown.

        References:
            [REF-PEARL-2009, Section 3.2]. No equation is evaluated.

        """
        members = self._node_set(nodes, name="nodes")
        return CausalGraph(
            (edge for edge in self.edges if edge[1] not in members), self.nodes
        )

    def remove_outgoing(self, nodes: str | Iterable[str]) -> CausalGraph:
        """Return the graph with every edge leaving ``nodes`` deleted.

        Args:
            nodes: Node or nodes whose outgoing edges are cut.

        Returns:
            The graph written ``G`` with ``X`` underlined in the literature.

        Raises:
            KeyError: If a node is unknown.

        References:
            [REF-PEARL-2009, Section 3.4]. No equation is evaluated.

        """
        members = self._node_set(nodes, name="nodes")
        return CausalGraph(
            (edge for edge in self.edges if edge[0] not in members), self.nodes
        )


def _single(graph: CausalGraph, node: str, *, name: str) -> str:
    """Validate a single node argument.

    Args:
        graph: Graph that must contain the node.
        node: Candidate node.
        name: Argument name for error messages.

    Returns:
        The node name.

    Raises:
        TypeError: If ``node`` is not a string.
        KeyError: If the node is unknown.

    References:
        Project validation policy. No equation is evaluated.

    """
    if not isinstance(node, str):
        raise TypeError(f"{name} must be a single node name")
    if node not in graph.nodes:
        raise KeyError(f"{name} {node!r} is not in the graph")
    return node


def _treatment_outcome(
    graph: CausalGraph, treatment: str | Iterable[str], outcome: str | Iterable[str]
) -> tuple[frozenset[str], frozenset[str]]:
    """Validate disjoint, non-empty treatment and outcome sets.

    Args:
        graph: Causal graph.
        treatment: Treatment node or nodes.
        outcome: Outcome node or nodes.

    Returns:
        Treatment and outcome sets.

    Raises:
        KeyError: If a node is unknown.
        ValueError: If a set is empty or the sets overlap.

    References:
        Project validation policy. No equation is evaluated.

    """
    xs = graph._node_set(treatment, name="treatment")
    ys = graph._node_set(outcome, name="outcome")
    if not xs or not ys:
        raise ValueError("treatment and outcome must be non-empty")
    if xs & ys:
        raise ValueError("treatment and outcome must be disjoint")
    return xs, ys


def satisfies_backdoor(
    graph: CausalGraph,
    treatment: str | Iterable[str],
    outcome: str | Iterable[str],
    adjustment: Iterable[str] = (),
) -> bool:
    """Test the backdoor criterion for an adjustment set.

    ``Z`` satisfies the criterion relative to ``(X, Y)`` when no member of
    ``Z`` descends from ``X`` and ``Z`` blocks every path between ``X`` and
    ``Y`` that starts with an arrow into ``X``; the second condition is
    checked as d-separation in the graph with the edges leaving ``X`` cut.

    Args:
        graph: Causal graph.
        treatment: Treatment node or nodes.
        outcome: Outcome node or nodes.
        adjustment: Candidate adjustment set (disjoint from both).

    Returns:
        ``True`` if the set satisfies the backdoor criterion.

    Raises:
        KeyError: If a node is unknown.
        ValueError: If the sets are empty where required or overlap.

    Equation:
        [EQ-CAUSAL-BD-001], [EQ-CAUSAL-DSEP-001].

    References:
        [REF-PEARL-2009, Definition 3.3.1 and Theorem 3.3.2].

    """
    xs, ys = _treatment_outcome(graph, treatment, outcome)
    zs = graph._node_set(adjustment, name="adjustment")
    if zs & (xs | ys):
        raise ValueError("adjustment must not contain treatment or outcome nodes")
    if zs & graph.descendants(xs):
        return False
    return graph.remove_outgoing(xs).is_d_separated(xs, ys, zs)


def minimal_adjustment_sets(
    graph: CausalGraph,
    treatment: str | Iterable[str],
    outcome: str | Iterable[str],
    *,
    latent: Iterable[str] = (),
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> tuple[tuple[str, ...], ...]:
    """Enumerate every minimal observed set satisfying the backdoor criterion.

    Candidates are the observed non-descendants of the treatment inside the
    ancestral set of ``X u Y`` in the graph with the treatment's outgoing
    edges removed: intersecting any valid set with that ancestral set keeps it
    valid, so every minimal set lies there. Subsets are scanned by increasing
    size and supersets of sets already found are skipped.

    Args:
        graph: Causal graph.
        treatment: Treatment node or nodes.
        outcome: Outcome node or nodes.
        latent: Unobserved nodes that may not be adjusted for.
        max_candidates: Largest candidate pool allowed (``2^k`` subsets).

    Returns:
        Minimal adjustment sets as sorted tuples, ordered by size and then
        lexicographically. An empty result means no observed backdoor set
        exists; ``((),)`` means no adjustment is needed.

    Raises:
        KeyError: If a node is unknown.
        ValueError: If treatment or outcome is latent, the sets overlap, or
            the candidate pool exceeds ``max_candidates``.

    Equation:
        [EQ-CAUSAL-BD-001].

    References:
        [REF-PEARL-2009, Section 3.3.1], [REF-LAURITZEN-ETAL-1990].

    """
    xs, ys = _treatment_outcome(graph, treatment, outcome)
    hidden = graph._node_set(latent, name="latent")
    if hidden & (xs | ys):
        raise ValueError("treatment and outcome must be observed")
    if isinstance(max_candidates, bool) or not isinstance(max_candidates, int):
        raise TypeError("max_candidates must be an integer")
    pruned = graph.remove_outgoing(xs)
    candidates = sorted(
        pruned.ancestral_set(xs | ys) - xs - ys - graph.descendants(xs) - hidden
    )
    if len(candidates) > max_candidates:
        raise ValueError(
            f"{len(candidates)} candidate nodes exceed max_candidates="
            f"{max_candidates}; exhaustive search would test 2^k subsets"
        )
    found: list[frozenset[str]] = []
    for size in range(len(candidates) + 1):
        for subset in combinations(candidates, size):
            members = frozenset(subset)
            if any(previous <= members for previous in found):
                continue
            if pruned.is_d_separated(xs, ys, members):
                found.append(members)
    return tuple(tuple(sorted(members)) for members in found)


def satisfies_frontdoor(
    graph: CausalGraph, treatment: str, outcome: str, mediators: Iterable[str]
) -> bool:
    """Test the frontdoor criterion for a mediator set.

    ``M`` satisfies the criterion relative to ``(X, Y)`` when (i) ``M``
    intercepts every directed path from ``X`` to ``Y``; (ii) there is no
    unblocked backdoor path from ``X`` to ``M``; and (iii) ``X`` blocks every
    backdoor path from ``M`` to ``Y``. Conditions (ii) and (iii) are checked
    as d-separations in the graphs with the edges leaving ``X`` (respectively
    ``M``) cut.

    Args:
        graph: Causal graph.
        treatment: Single treatment node ``X``.
        outcome: Single outcome node ``Y``.
        mediators: Non-empty mediator set ``M``.

    Returns:
        ``True`` if the set satisfies the frontdoor criterion.

    Raises:
        KeyError: If a node is unknown.
        TypeError: If treatment or outcome is not a single name.
        ValueError: If ``M`` is empty or overlaps ``{X, Y}``, or ``X == Y``.

    Equation:
        [EQ-CAUSAL-FDC-001], [EQ-CAUSAL-DSEP-001].

    References:
        [REF-PEARL-2009, Definition 3.3.3 and Theorem 3.3.4].

    """
    x = _single(graph, treatment, name="treatment")
    y = _single(graph, outcome, name="outcome")
    if x == y:
        raise ValueError("treatment and outcome must differ")
    ms = graph._node_set(mediators, name="mediators")
    if not ms:
        raise ValueError("mediators must be non-empty")
    if ms & {x, y}:
        raise ValueError("mediators must not contain treatment or outcome")
    seen: set[str] = set()
    stack = [child for child in graph.children(x) if child not in ms]
    while stack:
        current = stack.pop()
        if current == y:
            return False
        if current in seen:
            continue
        seen.add(current)
        stack.extend(child for child in graph.children(current) if child not in ms)
    if not graph.remove_outgoing(x).is_d_separated(x, ms):
        return False
    return graph.remove_outgoing(ms).is_d_separated(ms, y, x)
