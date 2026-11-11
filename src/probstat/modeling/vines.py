"""Vine copulas: pair-copula constructions of a multivariate copula.

A regular vine (Bedford and Cooke 2002) factorizes a ``d``-dimensional copula
density into ``d (d - 1) / 2`` bivariate copula densities, the *pair
copulas*, arranged in ``d - 1`` nested trees. Tree ``t`` couples two
variables conditionally on ``t - 1`` others; the arguments of each pair
copula are conditional distribution functions obtained recursively from the
previous tree by *h-functions*, the partial derivatives of the bivariate
copula CDFs (Aas, Czado, Frigessi and Bakken 2009).

Two vine classes are implemented:

- :class:`CVineCopula` (canonical vine): tree ``t`` is a star around one root
  variable, so every pair in tree ``t`` is conditioned on the first ``t - 1``
  roots;
- :class:`DVineCopula` (drawable vine): tree ``t`` is a path, so every pair
  is conditioned on the variables lying between its two ends in the path
  order.

The building block is :class:`PairCopula`: a bivariate copula of
:mod:`probstat.modeling.copulas` (independence, Gaussian, Student-t, Clayton,
Gumbel, Frank) with its two h-functions and their inverses, plus the 90, 180
and 270 degree rotations of the Clayton and Gumbel copulas, which give
negative dependence (90, 270) and the opposite tail (180).

Both vine classes implement the :class:`~probstat.modeling.copulas.Copula`
interface (log-density, Rosenblatt transform and inverse, first-tree
Kendall's tau and tail dependence, sub-vine margins, sampling) and can be the
dependence law of a :class:`~probstat.modeling.JointInput`. Their Rosenblatt
transform conditions in the vine variable order (``order``), not in the
coordinate order.

:func:`fit_vine_copula` estimates a C- or D-vine from data by the sequential
(tree-by-tree) maximum-likelihood method: rank pseudo-observations, a
structure chosen from absolute Kendall's tau (Dissmann, Brechmann, Czado and
Kurowicka 2013), per-edge family selection by AIC with an optional
Kendall's-tau independence test, and starting values from Kendall's tau
inversion. The result, :class:`VineCopulaFit`, reports every edge.
"""

from __future__ import annotations

import math
import random
from abc import abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import ClassVar, Literal

from probstat.distributions import StudentT
from probstat.foundations.validation import (
    require_finite_real,
    require_non_negative_integer,
)
from probstat.linalg import Vector
from probstat.optimization import minimize_bfgs
from probstat.statistics import ranks

from .copulas import (
    ClaytonCopula,
    Copula,
    FrankCopula,
    GaussianCopula,
    GumbelCopula,
    IndependenceCopula,
    StudentTCopula,
    _open,
    _unit_point,
)
from .fitting import (
    archimedean_theta_from_tau,
    kendall_tau,
    kendall_to_latent_correlation,
)

VineStructure = Literal["c", "d"]
_NORMAL = NormalDist()
_PAIR_BASES = (
    IndependenceCopula,
    GaussianCopula,
    StudentTCopula,
    ClaytonCopula,
    GumbelCopula,
    FrankCopula,
)
_ROTATABLE = ("clayton", "gumbel")
_FLIPS: dict[int, tuple[bool, bool]] = {
    0: (False, False),
    90: (True, False),
    180: (True, True),
    270: (False, True),
}
PAIR_COPULA_FAMILIES = (
    "independence",
    "gaussian",
    "student-t",
    "clayton",
    "gumbel",
    "frank",
)
_RHO_BOUND = 0.999
_CLAYTON_BOUNDS = (1e-4, 28.0)
_GUMBEL_BOUNDS = (1.0 + 1e-6, 20.0)
_FRANK_BOUNDS = (1e-4, 50.0)
_NU_BOUNDS = (1.0, 50.0)


def _reflect(value: float, flip: bool) -> float:
    """Return ``1 - value`` if ``flip`` else ``value``, clamped into ``(0, 1)``.

    Equation:
        [EQ-MOD-VINE-001].

    References:
        [REF-NELSEN-2006, Section 2.4]. No new equation is introduced.

    """
    return _open(1.0 - value if flip else value)


def _validated_rotation(value: object, family: str) -> int:
    """Validate a pair-copula rotation for a base family.

    Raises:
        TypeError: If the rotation is not an integer.
        ValueError: If it is not 0, 90, 180 or 270, or a non-zero rotation is
            requested for a family other than Clayton or Gumbel.

    References:
        Project validation policy, [REF-NELSEN-2006, Section 2.4]. No equation
        is evaluated.

    """
    rotation = require_non_negative_integer(value, name="rotation")
    if rotation not in _FLIPS:
        raise ValueError("rotation must be 0, 90, 180, or 270 degrees")
    if rotation and family not in _ROTATABLE:
        raise ValueError("only the Clayton and Gumbel pair copulas can be rotated")
    return rotation


@dataclass(frozen=True)
class PairCopula(Copula):
    """Bivariate building block of a vine: a base copula and a rotation.

    With ``(V_1, V_2)`` distributed as the base copula, the rotated pair is
    ``U_1 = 1 - V_1`` (90 and 180 degrees) or ``V_1``, and ``U_2 = 1 - V_2``
    (180 and 270 degrees) or ``V_2``. Its density is
    ``c(u_1, u_2) = c_base(r_1(u_1), r_2(u_2))`` with the matching
    reflections ``r_i``. The two h-functions are the conditional CDFs
    ``P(U_2 <= u_2 | U_1 = u_1) = dC/du_1`` (:meth:`h_given_first`) and
    ``P(U_1 <= u_1 | U_2 = u_2) = dC/du_2`` (:meth:`h_given_second`); in the
    notation of Aas et al. (2009) ``h(u_1 | u_2) = h_given_second(u_1, u_2)``.

    Rotations of 90 and 270 degrees give negative Kendall's tau; 180 degrees
    is the survival copula, which moves the Clayton lower tail to the upper
    tail and the Gumbel upper tail to the lower tail. Only Clayton and Gumbel
    may be rotated, because the Gaussian, Student-t and Frank copulas are
    radially symmetric and already cover negative dependence through their
    parameter.

    Args:
        copula: Bivariate base copula: :class:`IndependenceCopula`,
            :class:`GaussianCopula`, :class:`StudentTCopula`,
            :class:`ClaytonCopula`, :class:`GumbelCopula` or
            :class:`FrankCopula` of dimension two.
        rotation: Counter-clockwise rotation in degrees: 0, 90, 180 or 270.

    References:
        [REF-AAS-CZADO-FRIGESSI-BAKKEN-2009], [REF-NELSEN-2006, Section 2.4].

    """

    copula: Copula
    rotation: int = 0
    dimension: int = field(init=False)
    family: ClassVar[str] = "pair"

    def __post_init__(self) -> None:
        """Validate the base copula and the rotation.

        Raises:
            TypeError: If the base is not one of the supported bivariate
                families or the rotation is not an integer.
            ValueError: If the base is not bivariate or the rotation is not
                allowed for the family.

        References:
            [REF-NELSEN-2006, Section 2.4]. No equation is evaluated.

        """
        base = self.copula
        if not isinstance(base, _PAIR_BASES):
            raise TypeError(
                "a pair copula needs an independence, Gaussian, Student-t, "
                "Clayton, Gumbel or Frank base copula"
            )
        if base.dimension != 2:
            raise ValueError("a pair copula must be bivariate")
        rotation = _validated_rotation(self.rotation, base.family)
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "dimension", 2)

    @classmethod
    def from_parameters(
        cls, family: str, parameters: Sequence[float] = (), rotation: int = 0
    ) -> PairCopula:
        """Build a pair copula from a family name and its parameters.

        Args:
            family: ``"independence"`` (no parameter), ``"gaussian"``
                (``rho``), ``"student-t"`` (``rho``, ``nu``), ``"clayton"``,
                ``"gumbel"`` or ``"frank"`` (``theta``).
            parameters: Family parameters in the order above.
            rotation: Rotation in degrees (Clayton and Gumbel only).

        Returns:
            Pair copula.

        Raises:
            TypeError: If a parameter is not real.
            ValueError: If the family is unknown or the parameter count or
                range is invalid.

        References:
            [REF-AAS-CZADO-FRIGESSI-BAKKEN-2009], [REF-NELSEN-2006,
            Chapter 4]. No equation is evaluated.

        """
        values = tuple(require_finite_real(p, name="parameter") for p in parameters)
        expected = {
            "independence": 0,
            "gaussian": 1,
            "student-t": 2,
            "clayton": 1,
            "gumbel": 1,
            "frank": 1,
        }
        if family not in expected:
            raise ValueError(f"family must be one of {PAIR_COPULA_FAMILIES}")
        if len(values) != expected[family]:
            raise ValueError(
                f"the {family} pair copula takes {expected[family]} parameter(s)"
            )
        base: Copula
        if family == "independence":
            base = IndependenceCopula(2)
        elif family == "gaussian":
            base = GaussianCopula(((1.0, values[0]), (values[0], 1.0)))
        elif family == "student-t":
            base = StudentTCopula(((1.0, values[0]), (values[0], 1.0)), values[1])
        elif family == "clayton":
            base = ClaytonCopula(values[0])
        elif family == "gumbel":
            base = GumbelCopula(values[0])
        else:
            base = FrankCopula(values[0])
        return cls(base, rotation)

    @property
    def base_family(self) -> str:
        """Return the family name of the base copula (``"clayton"``, ...).

        References:
            [REF-NELSEN-2006, Section 2.4]. No equation is evaluated.

        """
        return self.copula.family

    @property
    def name(self) -> str:
        """Return the family name with the rotation, such as ``"gumbel-90"``.

        References:
            [REF-NELSEN-2006, Section 2.4]. No equation is evaluated.

        """
        if self.rotation:
            return f"{self.base_family}-{self.rotation}"
        return self.base_family

    @property
    def parameters(self) -> tuple[float, ...]:
        """Return the parameters: ``()``, ``(rho,)``, ``(rho, nu)`` or ``(theta,)``.

        References:
            [REF-NELSEN-2006, Chapter 4]. No equation is evaluated.

        """
        base = self.copula
        if isinstance(base, GaussianCopula):
            return (base.correlation[0][1],)
        if isinstance(base, StudentTCopula):
            return (base.correlation[0][1], base.nu)
        if isinstance(base, ClaytonCopula | GumbelCopula | FrankCopula):
            return (base.theta,)
        return ()

    @property
    def parameter_count(self) -> int:
        """Return the number of free parameters (for information criteria).

        References:
            [REF-AKAIKE-1974]. No equation is evaluated.

        """
        return len(self.parameters)

    @property
    def _flips(self) -> tuple[bool, bool]:
        """Return the reflections ``(1 - u_1?, 1 - u_2?)`` of the rotation.

        References:
            [EQ-MOD-VINE-001], [REF-NELSEN-2006, Section 2.4].

        """
        return _FLIPS[self.rotation]

    def _base_h(self, value: float, given: float) -> float:
        """Return ``P(V_2 <= value | V_1 = given)`` of the exchangeable base.

        Equation:
            [EQ-MOD-COP-ROSEN-001], [EQ-MOD-VINE-001].

        References:
            [REF-AAS-CZADO-FRIGESSI-BAKKEN-2009], [REF-ROSENBLATT-1952].

        """
        return _open(self.copula.rosenblatt((_open(given), _open(value)))[1])

    def _base_h_inverse(self, probability: float, given: float) -> float:
        """Solve ``P(V_2 <= v | V_1 = given) = probability`` for ``v``.

        Equation:
            [EQ-MOD-COP-ROSEN-001], [EQ-MOD-VINE-001].

        References:
            [REF-AAS-CZADO-FRIGESSI-BAKKEN-2009], [REF-ROSENBLATT-1952].

        """
        return _open(
            self.copula.inverse_rosenblatt((_open(given), _open(probability)))[1]
        )

    def log_density(self, u: Sequence[float]) -> float:
        """Return ``log c_base(r_1(u_1), r_2(u_2))``.

        Args:
            u: Point in ``(0, 1)^2``.

        Returns:
            Log pair-copula density.

        Equation:
            [EQ-MOD-VINE-001].

        References:
            [REF-NELSEN-2006, Section 2.4].

        """
        point = _unit_point(u, 2)
        flip_first, flip_second = self._flips
        return self.copula.log_density(
            (_reflect(point[0], flip_first), _reflect(point[1], flip_second))
        )

    def pair_log_density(self, first: float, second: float) -> float:
        """Return ``log c(first, second)`` for two separate arguments.

        Unlike :meth:`log_density` the arguments are clamped into the open
        unit interval instead of being rejected on the boundary, as the
        h-function outputs of a vine are.

        Args:
            first: First argument in ``(0, 1)``.
            second: Second argument in ``(0, 1)``.

        Returns:
            Log pair-copula density.

        Equation:
            [EQ-MOD-VINE-001].

        References:
            [REF-NELSEN-2006, Section 2.4].

        """
        flip_first, flip_second = self._flips
        return self.copula.log_density(
            (_reflect(first, flip_first), _reflect(second, flip_second))
        )

    def h_given_first(self, first: float, second: float) -> float:
        """Return ``P(U_2 <= second | U_1 = first) = dC(first, second)/du_1``.

        Args:
            first: Conditioning value ``u_1`` in ``(0, 1)``.
            second: Value ``u_2`` in ``(0, 1)``.

        Returns:
            Conditional probability, clamped into ``[1e-300, 1 - 2^-53]``.

        Equation:
            [EQ-MOD-VINE-001].

        References:
            [REF-AAS-CZADO-FRIGESSI-BAKKEN-2009], [REF-NELSEN-2006, Section 2.4].

        """
        flip_first, flip_second = self._flips
        value = self._base_h(_reflect(second, flip_second), _reflect(first, flip_first))
        return _reflect(value, flip_second)

    def h_given_second(self, first: float, second: float) -> float:
        """Return ``P(U_1 <= first | U_2 = second) = dC(first, second)/du_2``.

        This is the h-function ``h(u_1 | u_2)`` of Aas et al. (2009).

        Args:
            first: Value ``u_1`` in ``(0, 1)``.
            second: Conditioning value ``u_2`` in ``(0, 1)``.

        Returns:
            Conditional probability, clamped into ``[1e-300, 1 - 2^-53]``.

        Equation:
            [EQ-MOD-VINE-001].

        References:
            [REF-AAS-CZADO-FRIGESSI-BAKKEN-2009], [REF-NELSEN-2006, Section 2.4].

        """
        flip_first, flip_second = self._flips
        value = self._base_h(_reflect(first, flip_first), _reflect(second, flip_second))
        return _reflect(value, flip_first)

    def h_inverse_given_first(self, first: float, probability: float) -> float:
        """Solve ``h_given_first(first, u_2) = probability`` for ``u_2``.

        Args:
            first: Conditioning value ``u_1`` in ``(0, 1)``.
            probability: Conditional probability in ``(0, 1)``.

        Returns:
            ``u_2``, clamped into ``[1e-300, 1 - 2^-53]``.

        Equation:
            [EQ-MOD-VINE-001].

        References:
            [REF-AAS-CZADO-FRIGESSI-BAKKEN-2009], [REF-NELSEN-2006, Section 2.4].

        """
        flip_first, flip_second = self._flips
        value = self._base_h_inverse(
            _reflect(probability, flip_second), _reflect(first, flip_first)
        )
        return _reflect(value, flip_second)

    def h_inverse_given_second(self, probability: float, second: float) -> float:
        """Solve ``h_given_second(u_1, second) = probability`` for ``u_1``.

        Args:
            probability: Conditional probability in ``(0, 1)``.
            second: Conditioning value ``u_2`` in ``(0, 1)``.

        Returns:
            ``u_1``, clamped into ``[1e-300, 1 - 2^-53]``.

        Equation:
            [EQ-MOD-VINE-001].

        References:
            [REF-AAS-CZADO-FRIGESSI-BAKKEN-2009], [REF-NELSEN-2006, Section 2.4].

        """
        flip_first, flip_second = self._flips
        value = self._base_h_inverse(
            _reflect(probability, flip_first), _reflect(second, flip_second)
        )
        return _reflect(value, flip_first)

    def _interior_cdf(self, u: Vector) -> float:
        """Return the rotated CDF from the base CDF.

        ``C_90 = u_2 - C(1 - u_1, u_2)``, ``C_180 = u_1 + u_2 - 1 +
        C(1 - u_1, 1 - u_2)``, ``C_270 = u_1 - C(u_1, 1 - u_2)``.

        Raises:
            NotImplementedError: For a Student-t base (no tractable CDF).

        Equation:
            [EQ-MOD-VINE-001].

        References:
            [REF-NELSEN-2006, Section 2.4].

        """
        first, second = u
        flip_first, flip_second = self._flips
        base = self.copula
        if flip_first and flip_second:
            return first + second - 1.0 + base.cdf((1.0 - first, 1.0 - second))
        if flip_first:
            return second - base.cdf((1.0 - first, second))
        if flip_second:
            return first - base.cdf((first, 1.0 - second))
        return base.cdf((first, second))

    @property
    def has_cdf(self) -> bool:
        """Return whether the CDF is available (every base except Student-t).

        References:
            [REF-NELSEN-2006, Chapter 2]. No equation is evaluated.

        """
        return not isinstance(self.copula, StudentTCopula)

    def kendall_tau(self, i: int = 0, j: int = 1) -> float:
        """Return the base tau, negated for the 90 and 270 degree rotations.

        Args:
            i: First coordinate index.
            j: Second coordinate index.

        Returns:
            Kendall's tau.

        Raises:
            ValueError: If the indices are not ``0`` and ``1``.

        Equation:
            [EQ-MOD-COP-TAU-001], [EQ-MOD-VINE-001].

        References:
            [REF-NELSEN-2006, Chapter 5], [REF-NELSEN-2006, Section 2.4].

        """
        tau = self.copula.kendall_tau(i, j)
        flip_first, flip_second = self._flips
        return -tau if flip_first != flip_second else tau

    def tail_dependence(self, i: int = 0, j: int = 1) -> tuple[float, float]:
        """Return the tail-dependence coefficients of the rotated pair.

        The 180 degree rotation swaps the lower and upper coefficients. The
        90 and 270 degree rotations move the dependence into the off-diagonal
        corners; the Clayton and Gumbel copulas have no off-diagonal tail
        dependence, so both coefficients are zero.

        Args:
            i: First coordinate index.
            j: Second coordinate index.

        Returns:
            ``(lambda_L, lambda_U)``.

        Raises:
            ValueError: If the indices are not ``0`` and ``1``.

        Equation:
            [EQ-MOD-COP-TAIL-001], [EQ-MOD-VINE-001].

        References:
            [REF-NELSEN-2006, Chapter 5], [REF-NELSEN-2006, Section 2.4].

        """
        lower, upper = self.copula.tail_dependence(i, j)
        flip_first, flip_second = self._flips
        if flip_first and flip_second:
            return upper, lower
        if flip_first or flip_second:
            return 0.0, 0.0
        return lower, upper

    def rosenblatt(self, u: Sequence[float]) -> Vector:
        """Return ``(u_1, h_given_first(u_1, u_2))``.

        Args:
            u: Point in ``(0, 1)^2``.

        Returns:
            Independent uniforms.

        Equation:
            [EQ-MOD-COP-ROSEN-001], [EQ-MOD-VINE-001].

        References:
            [REF-ROSENBLATT-1952], [REF-AAS-CZADO-FRIGESSI-BAKKEN-2009].

        """
        point = _unit_point(u, 2)
        return point[0], self.h_given_first(point[0], point[1])

    def inverse_rosenblatt(self, w: Sequence[float]) -> Vector:
        """Return ``(w_1, h_inverse_given_first(w_1, w_2))``.

        Args:
            w: Point in ``(0, 1)^2``.

        Returns:
            Copula point.

        Equation:
            [EQ-MOD-COP-ROSEN-001], [EQ-MOD-VINE-001].

        References:
            [REF-ROSENBLATT-1952], [REF-AAS-CZADO-FRIGESSI-BAKKEN-2009].

        """
        point = _unit_point(w, 2)
        return point[0], self.h_inverse_given_first(point[0], point[1])

    def _draw(self, generator: random.Random) -> Vector:
        """Draw from the base sampler and reflect the coordinates.

        Equation:
            [EQ-MOD-VINE-001], [EQ-MOD-COP-ARCH-003].

        References:
            [REF-NELSEN-2006, Section 2.4], [REF-MARSHALL-OLKIN-1988].

        """
        first, second = self.copula._draw(generator)
        flip_first, flip_second = self._flips
        return _reflect(first, flip_first), _reflect(second, flip_second)

    def swapped(self) -> PairCopula:
        """Return the pair copula of ``(U_2, U_1)``.

        The base families are exchangeable, so swapping the arguments maps
        the 90 degree rotation to 270 degrees and keeps 0 and 180.

        Returns:
            Pair copula with the arguments exchanged.

        Equation:
            [EQ-MOD-VINE-001].

        References:
            [REF-NELSEN-2006, Section 2.4]. No new equation is introduced.

        """
        swap = {0: 0, 90: 270, 180: 180, 270: 90}
        return PairCopula(self.copula, swap[self.rotation])

    def marginal(self, indices: Sequence[int]) -> Copula:
        """Return a one-dimensional margin, the pair itself, or its swap.

        Args:
            indices: Distinct indices among ``0`` and ``1``.

        Returns:
            Independence copula for one index; the pair for ``(0, 1)``; the
            swapped pair for ``(1, 0)``.

        Raises:
            ValueError: If the indices are invalid.

        References:
            [REF-NELSEN-2006, Section 2.10].

        """
        chosen = _chosen_indices(indices, 2)
        if len(chosen) == 1:
            return IndependenceCopula(1)
        return self if chosen == (0, 1) else self.swapped()


def _chosen_indices(indices: Sequence[int], dimension: int) -> tuple[int, ...]:
    """Validate a non-empty sequence of distinct coordinate indices.

    Raises:
        TypeError: If an index is not an integer.
        ValueError: If it is empty, repeats, or leaves the range.

    References:
        Project validation policy. No equation is evaluated.

    """
    chosen = tuple(require_non_negative_integer(i, name="index") for i in indices)
    if not chosen or len(set(chosen)) != len(chosen):
        raise ValueError("indices must be a non-empty set of distinct indices")
    if max(chosen) >= dimension:
        raise ValueError("an index exceeds the copula dimension")
    return chosen


def _as_pair(value: object) -> PairCopula:
    """Return a :class:`PairCopula`, wrapping a supported bivariate copula.

    Raises:
        TypeError: If the value is not a pair copula or a supported family.

    References:
        [REF-AAS-CZADO-FRIGESSI-BAKKEN-2009]. No equation is evaluated.

    """
    if isinstance(value, PairCopula):
        return value
    if isinstance(value, _PAIR_BASES):
        return PairCopula(value)
    raise TypeError("vine entries must be PairCopula or bivariate copula objects")


@dataclass(frozen=True, slots=True)
class VineEdge:
    """One edge of a vine: a pair copula with its conditioned and conditioning sets.

    Args:
        tree: Tree number, ``1`` for the first tree.
        conditioned: Coordinate indices ``(a, b)``; the pair copula's first
            argument is ``F(u_a | conditioning)``, its second
            ``F(u_b | conditioning)``.
        conditioning: Coordinate indices conditioned on (empty in tree 1).
        pair_copula: The pair copula of the edge.

    References:
        [REF-BEDFORD-COOKE-2002], [REF-AAS-CZADO-FRIGESSI-BAKKEN-2009].

    """

    tree: int
    conditioned: tuple[int, int]
    conditioning: tuple[int, ...]
    pair_copula: PairCopula


@dataclass(frozen=True)
class VineCopula(Copula):
    """Common machinery of the C-vine and D-vine pair-copula constructions.

    ``pair_copulas[t][e]`` is edge ``e`` of tree ``t + 1`` (0-based ``t``);
    tree ``t + 1`` has ``d - 1 - t`` edges. Fewer than ``d - 1`` trees give a
    truncated vine: the missing trees are filled with independence pair
    copulas. ``order`` lists the coordinate index at each vine position; the
    default is the identity. Entries that are supported bivariate copulas
    (not :class:`PairCopula`) are wrapped without rotation.

    Args:
        pair_copulas: Trees of pair copulas.
        order: Permutation of ``0, ..., d - 1`` (vine position to coordinate).

    References:
        [REF-AAS-CZADO-FRIGESSI-BAKKEN-2009], [REF-BEDFORD-COOKE-2002],
        [REF-JOE-2014].

    """

    pair_copulas: tuple[tuple[PairCopula, ...], ...]
    order: tuple[int, ...] | None = None
    dimension: int = field(init=False)
    _position: tuple[int, ...] = field(init=False, repr=False, compare=False)
    structure: ClassVar[str] = "vine"
    family: ClassVar[str] = "vine"

    def __post_init__(self) -> None:
        """Validate the trees and the order; pad a truncated vine.

        Raises:
            TypeError: If the trees are malformed or an entry is not a
                supported pair copula.
            ValueError: If tree sizes, the number of trees or the order are
                inconsistent with the dimension.

        References:
            [REF-AAS-CZADO-FRIGESSI-BAKKEN-2009]. No equation is evaluated.

        """
        trees = self.pair_copulas
        if isinstance(trees, str | bytes) or not isinstance(trees, Sequence):
            raise TypeError("pair_copulas must be a sequence of trees")
        if not trees:
            raise ValueError("a vine needs at least one tree")
        rows: list[tuple[PairCopula, ...]] = []
        for tree in trees:
            if isinstance(tree, str | bytes) or not isinstance(tree, Sequence):
                raise TypeError("each tree must be a sequence of pair copulas")
            rows.append(tuple(_as_pair(entry) for entry in tree))
        dimension = len(rows[0]) + 1
        if dimension < 2:
            raise ValueError("the first tree needs at least one pair copula")
        if len(rows) > dimension - 1:
            raise ValueError(
                f"a {dimension}-dimensional vine has {dimension - 1} trees"
            )
        for t, row in enumerate(rows):
            if len(row) != dimension - 1 - t:
                raise ValueError(
                    f"tree {t + 1} must have {dimension - 1 - t} pair copulas"
                )
        independence = PairCopula(IndependenceCopula(2))
        for t in range(len(rows), dimension - 1):
            rows.append(tuple(independence for _ in range(dimension - 1 - t)))
        if self.order is None:
            order = tuple(range(dimension))
        else:
            order = tuple(
                require_non_negative_integer(i, name="order") for i in self.order
            )
            if sorted(order) != list(range(dimension)):
                raise ValueError("order must be a permutation of 0, ..., d - 1")
        position = [0] * dimension
        for p, coordinate in enumerate(order):
            position[coordinate] = p
        object.__setattr__(self, "pair_copulas", tuple(rows))
        object.__setattr__(self, "order", order)
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "_position", tuple(position))

    @property
    def vine_order(self) -> tuple[int, ...]:
        """Return the coordinate index at each vine position.

        References:
            [REF-AAS-CZADO-FRIGESSI-BAKKEN-2009]. No equation is evaluated.

        """
        assert self.order is not None
        return self.order

    @property
    def parameter_count(self) -> int:
        """Return the total number of pair-copula parameters.

        References:
            [REF-AKAIKE-1974]. No equation is evaluated.

        """
        return sum(pair.parameter_count for row in self.pair_copulas for pair in row)

    @abstractmethod
    def _edge_sets(
        self, tree: int, edge: int
    ) -> tuple[tuple[int, int], tuple[int, ...]]:
        """Return the conditioned and conditioning vine positions of an edge.

        References:
            [REF-AAS-CZADO-FRIGESSI-BAKKEN-2009].

        """

    @abstractmethod
    def _log_density_positions(self, values: list[float]) -> float:
        """Return the log-density for coordinates listed in vine order.

        References:
            [REF-AAS-CZADO-FRIGESSI-BAKKEN-2009, Algorithms 1-4].

        """

    @abstractmethod
    def _rosenblatt_positions(self, values: list[float]) -> list[float]:
        """Return ``F(u_p | u_<p)`` for coordinates listed in vine order.

        References:
            [REF-AAS-CZADO-FRIGESSI-BAKKEN-2009, Algorithms 1-4].

        """

    @abstractmethod
    def _inverse_positions(self, values: list[float]) -> list[float]:
        """Invert :meth:`_rosenblatt_positions`.

        References:
            [REF-AAS-CZADO-FRIGESSI-BAKKEN-2009, Algorithms 1-4].

        """

    @abstractmethod
    def _sub_vine(self, positions: tuple[int, ...]) -> VineCopula | None:
        """Return the sub-vine on sorted vine positions, or ``None`` if none exists.

        References:
            [REF-AAS-CZADO-FRIGESSI-BAKKEN-2009], [REF-JOE-2014].

        """

    @property
    def edges(self) -> tuple[VineEdge, ...]:
        """Return every edge with coordinate-index conditioned and conditioning sets.

        References:
            [REF-BEDFORD-COOKE-2002]. No equation is evaluated.

        """
        order = self.vine_order
        result = []
        for t, row in enumerate(self.pair_copulas):
            for e, pair in enumerate(row):
                (a, b), given = self._edge_sets(t, e)
                result.append(
                    VineEdge(
                        t + 1,
                        (order[a], order[b]),
                        tuple(order[g] for g in given),
                        pair,
                    )
                )
        return tuple(result)

    def _in_order(self, point: Vector) -> list[float]:
        """Return the coordinates of ``point`` listed in vine order.

        References:
            [REF-AAS-CZADO-FRIGESSI-BAKKEN-2009]. No equation is evaluated.

        """
        return [point[coordinate] for coordinate in self.vine_order]

    def _from_order(self, values: Sequence[float]) -> Vector:
        """Return vine-ordered values placed back at their coordinates.

        References:
            [REF-AAS-CZADO-FRIGESSI-BAKKEN-2009]. No equation is evaluated.

        """
        return tuple(values[p] for p in self._position)

    def log_density(self, u: Sequence[float]) -> float:
        """Return the pair-copula-construction log-density.

        Args:
            u: Point in ``(0, 1)^d``.

        Returns:
            ``log c(u)``, the sum of the log pair-copula densities evaluated
            at the conditional distribution functions of every edge.

        Raises:
            ValueError: If the point is outside the open unit cube.

        Equation:
            [EQ-MOD-VINE-002], [EQ-MOD-VINE-003], [EQ-MOD-VINE-004].

        References:
            [REF-AAS-CZADO-FRIGESSI-BAKKEN-2009, Algorithms 1-4].

        """
        point = _unit_point(u, self.dimension)
        return self._log_density_positions(self._in_order(point))

    def rosenblatt(self, u: Sequence[float]) -> Vector:
        """Return the Rosenblatt transform in the vine order.

        With ``p_k = order[k]``, ``w_{p_1} = u_{p_1}`` and
        ``w_{p_k} = C(u_{p_k} | u_{p_1}, ..., u_{p_{k-1}})``; the conditionals
        come from the h-function recursion of the vine.

        Args:
            u: Point in ``(0, 1)^d``.

        Returns:
            Independent uniforms, at the coordinates of ``u``.

        Equation:
            [EQ-MOD-COP-ROSEN-001], [EQ-MOD-VINE-002], [EQ-MOD-VINE-005].

        References:
            [REF-ROSENBLATT-1952], [REF-AAS-CZADO-FRIGESSI-BAKKEN-2009,
            Algorithms 1-4].

        """
        point = _unit_point(u, self.dimension)
        return self._from_order(self._rosenblatt_positions(self._in_order(point)))

    def inverse_rosenblatt(self, w: Sequence[float]) -> Vector:
        """Invert the vine-order Rosenblatt transform by nested h-inverses.

        Args:
            w: Point in ``(0, 1)^d``.

        Returns:
            Copula point ``u`` with ``rosenblatt(u) = w``; coordinates are
            clamped into ``[1e-300, 1 - 2^-53]``.

        Equation:
            [EQ-MOD-COP-ROSEN-001], [EQ-MOD-VINE-002], [EQ-MOD-VINE-005].

        References:
            [REF-ROSENBLATT-1952], [REF-AAS-CZADO-FRIGESSI-BAKKEN-2009,
            Algorithms 1-4].

        """
        point = _unit_point(w, self.dimension)
        return self._from_order(self._inverse_positions(self._in_order(point)))

    def _first_tree_pair(self, i: int, j: int) -> PairCopula:
        """Return the first-tree pair copula joining coordinates ``i`` and ``j``.

        Raises:
            ValueError: If the indices are invalid.
            NotImplementedError: If ``i`` and ``j`` are not joined in tree 1.

        References:
            [REF-AAS-CZADO-FRIGESSI-BAKKEN-2009]. No equation is evaluated.

        """
        first = require_non_negative_integer(i, name="i")
        second = require_non_negative_integer(j, name="j")
        if first >= self.dimension or second >= self.dimension or first == second:
            raise ValueError("i and j must be distinct indices below the dimension")
        order = self.vine_order
        for e, pair in enumerate(self.pair_copulas[0]):
            (a, b), _ = self._edge_sets(0, e)
            if {order[a], order[b]} == {first, second}:
                return pair
        raise NotImplementedError(
            "closed-form dependence measures are available for first-tree pairs only"
        )

    def kendall_tau(self, i: int = 0, j: int = 1) -> float:
        """Return Kendall's tau of a pair joined in the first tree.

        A first-tree pair copula is the bivariate margin of the vine, so its
        tau is the population tau of the pair.

        Args:
            i: First coordinate index.
            j: Second coordinate index.

        Returns:
            Kendall's tau.

        Raises:
            ValueError: If the indices are invalid.
            NotImplementedError: If the pair is not a first-tree edge.

        Equation:
            [EQ-MOD-COP-TAU-001], [EQ-MOD-VINE-001].

        References:
            [REF-AAS-CZADO-FRIGESSI-BAKKEN-2009], [REF-NELSEN-2006, Chapter 5].

        """
        return self._first_tree_pair(i, j).kendall_tau()

    def tail_dependence(self, i: int = 0, j: int = 1) -> tuple[float, float]:
        """Return the tail-dependence coefficients of a first-tree pair.

        Args:
            i: First coordinate index.
            j: Second coordinate index.

        Returns:
            ``(lambda_L, lambda_U)``.

        Raises:
            ValueError: If the indices are invalid.
            NotImplementedError: If the pair is not a first-tree edge.

        Equation:
            [EQ-MOD-COP-TAIL-001], [EQ-MOD-VINE-001].

        References:
            [REF-NELSEN-2006, Section 2.4], [REF-NELSEN-2006, Chapter 5].

        """
        return self._first_tree_pair(i, j).tail_dependence()

    def _interior_cdf(self, u: Vector) -> float:
        """Return the CDF of a bivariate vine (its single pair copula).

        Raises:
            NotImplementedError: For dimensions above two or a Student-t pair.

        Equation:
            [EQ-MOD-VINE-001].

        References:
            [REF-NELSEN-2006, Chapter 2].

        """
        if self.dimension != 2:
            raise NotImplementedError(
                "the vine copula CDF is implemented for two dimensions only"
            )
        values = self._in_order(u)
        return self.pair_copulas[0][0].cdf((values[0], values[1]))

    @property
    def has_cdf(self) -> bool:
        """Return whether :meth:`cdf` is available (bivariate vines only).

        References:
            [REF-NELSEN-2006, Chapter 2]. No equation is evaluated.

        """
        return self.dimension == 2 and self.pair_copulas[0][0].has_cdf

    def marginal(self, indices: Sequence[int]) -> Copula:
        """Return the margin of a coordinate subset when it is a sub-vine.

        A C-vine margin exists in closed form for the first ``k - 1`` roots
        plus any one further variable; a D-vine margin for any contiguous run
        of the path. The full set in any order is always available.

        Args:
            indices: Distinct coordinate indices, in the order of the margin.

        Returns:
            Independence copula for one index, otherwise a vine of the same
            structure whose coordinates follow ``indices``.

        Raises:
            ValueError: If the indices are invalid.
            NotImplementedError: If the subset is not a sub-vine.

        References:
            [REF-AAS-CZADO-FRIGESSI-BAKKEN-2009], [REF-JOE-2014],
            [REF-NELSEN-2006, Section 2.10].

        """
        chosen = _chosen_indices(indices, self.dimension)
        if len(chosen) == 1:
            return IndependenceCopula(1)
        positions = tuple(sorted(self._position[c] for c in chosen))
        sub = self._sub_vine(positions)
        if sub is None:
            raise NotImplementedError(
                f"coordinates {chosen} do not form a sub-vine of this {self.family}"
            )
        order = self.vine_order
        new_order = tuple(chosen.index(order[p]) for p in positions)
        return type(self)(sub.pair_copulas, new_order)


@dataclass(frozen=True)
class CVineCopula(VineCopula):
    """Canonical vine: tree ``t`` is a star around the ``t``-th root.

    With vine positions ``1, ..., d`` (``order``), edge ``e`` of tree ``t``
    joins the root ``t`` and position ``t + e`` given positions
    ``1, ..., t - 1``; the pair copula's first argument is the root's
    conditional distribution function. The density is
    ``c = prod_t prod_e c_{t, t+e | 1..t-1}(F(u_t | u_<t), F(u_{t+e} | u_<t))``.

    Args:
        pair_copulas: ``d - 1`` trees (or fewer, truncated) with
            ``d - t`` pair copulas in tree ``t``.
        order: Coordinate index of each vine position (identity by default).

    References:
        [REF-AAS-CZADO-FRIGESSI-BAKKEN-2009, Algorithms 1-4],
        [REF-BEDFORD-COOKE-2002].

    """

    structure: ClassVar[str] = "c"
    family: ClassVar[str] = "c-vine"

    def _edge_sets(
        self, tree: int, edge: int
    ) -> tuple[tuple[int, int], tuple[int, ...]]:
        """Return ``((tree, tree + 1 + edge), (0, ..., tree - 1))`` in vine positions.

        References:
            [REF-AAS-CZADO-FRIGESSI-BAKKEN-2009]. No equation is evaluated.

        """
        return (tree, tree + 1 + edge), tuple(range(tree))

    def _log_density_positions(self, values: list[float]) -> float:
        """Return the C-vine log-density (the C-vine likelihood algorithm).

        ``v`` holds ``F(u_k | u_0, ..., u_{t-1})`` for the current tree ``t``.

        Equation:
            [EQ-MOD-VINE-002], [EQ-MOD-VINE-003].

        References:
            [REF-AAS-CZADO-FRIGESSI-BAKKEN-2009, Algorithms 1-4].

        """
        d = self.dimension
        v = list(values)
        total = 0.0
        for t, row in enumerate(self.pair_copulas):
            root = v[t]
            for e, pair in enumerate(row):
                k = t + 1 + e
                total += pair.pair_log_density(root, v[k])
                if t < d - 2:
                    v[k] = pair.h_given_first(root, v[k])
        return total

    def _rosenblatt_positions(self, values: list[float]) -> list[float]:
        """Return ``w_k = F(u_k | u_0, ..., u_{k-1})`` by the h-recursion.

        At the start of tree ``k`` the working value of position ``k`` is
        ``F(u_k | u_0, ..., u_{k-1})``, which is ``w_k``.

        Equation:
            [EQ-MOD-VINE-002], [EQ-MOD-VINE-005].

        References:
            [REF-AAS-CZADO-FRIGESSI-BAKKEN-2009, Algorithms 1-4].

        """
        d = self.dimension
        v = list(values)
        w = [v[0]]
        for t, row in enumerate(self.pair_copulas):
            root = v[t]
            for e, pair in enumerate(row):
                k = t + 1 + e
                v[k] = pair.h_given_first(root, v[k])
            w.append(v[t + 1])
        assert len(w) == d
        return w

    def _inverse_positions(self, values: list[float]) -> list[float]:
        """Invert the C-vine Rosenblatt transform (the C-vine sampling algorithm).

        ``u_k = h_0^-1(... h_{k-1}^-1(w_k | w_{k-1}) ... | w_0)``: the root
        argument of edge ``(t, k)`` is ``F(u_t | u_<t) = w_t``.

        Equation:
            [EQ-MOD-VINE-002], [EQ-MOD-VINE-005].

        References:
            [REF-AAS-CZADO-FRIGESSI-BAKKEN-2009, Algorithms 1-4].

        """
        result = [values[0]]
        for k in range(1, self.dimension):
            current = values[k]
            for t in range(k - 1, -1, -1):
                pair = self.pair_copulas[t][k - t - 1]
                current = pair.h_inverse_given_first(values[t], current)
            result.append(current)
        return result

    def _sub_vine(self, positions: tuple[int, ...]) -> VineCopula | None:
        """Return the C-vine on roots ``0, ..., k - 2`` and one more position.

        References:
            [REF-AAS-CZADO-FRIGESSI-BAKKEN-2009], [REF-JOE-2014].

        """
        size = len(positions)
        if list(positions[:-1]) != list(range(size - 1)):
            return None
        trees = tuple(
            tuple(
                self.pair_copulas[t][positions[t + 1 + e] - t - 1]
                for e in range(size - 1 - t)
            )
            for t in range(size - 1)
        )
        return CVineCopula(trees)


@dataclass(frozen=True)
class DVineCopula(VineCopula):
    """Drawable vine: every tree is a path along the vine order.

    With vine positions ``1, ..., d`` (``order``), edge ``e`` of tree ``t``
    joins positions ``e`` and ``e + t`` given ``e + 1, ..., e + t - 1``; the
    pair copula's first argument is ``F(u_e | between)``, its second
    ``F(u_{e+t} | between)``. The density is
    ``c = prod_t prod_e c_{e, e+t | e+1..e+t-1}(F(u_e | .), F(u_{e+t} | .))``.

    Args:
        pair_copulas: ``d - 1`` trees (or fewer, truncated) with
            ``d - t`` pair copulas in tree ``t``.
        order: Coordinate index of each vine position (identity by default).

    References:
        [REF-AAS-CZADO-FRIGESSI-BAKKEN-2009, Algorithms 1-4],
        [REF-BEDFORD-COOKE-2002].

    """

    structure: ClassVar[str] = "d"
    family: ClassVar[str] = "d-vine"

    def _edge_sets(
        self, tree: int, edge: int
    ) -> tuple[tuple[int, int], tuple[int, ...]]:
        """Return ``((edge, edge + tree + 1), (edge + 1, ..., edge + tree))``.

        References:
            [REF-AAS-CZADO-FRIGESSI-BAKKEN-2009]. No equation is evaluated.

        """
        return (edge, edge + tree + 1), tuple(range(edge + 1, edge + tree + 1))

    def _next_arguments(
        self, tree: int, left: list[float], right: list[float]
    ) -> tuple[list[float], list[float]]:
        """Return the pair-copula arguments of tree ``tree + 1`` from tree ``tree``.

        ``left[e] = F(u_e | u_{e+1..e+t})`` and
        ``right[e] = F(u_{e+t+1} | u_{e+1..e+t})``; the next tree uses
        ``h_given_second`` of edge ``e`` and ``h_given_first`` of edge
        ``e + 1``.

        Equation:
            [EQ-MOD-VINE-002].

        References:
            [REF-AAS-CZADO-FRIGESSI-BAKKEN-2009, Algorithms 1-4].

        """
        row = self.pair_copulas[tree]
        count = len(row) - 1
        new_left = [row[e].h_given_second(left[e], right[e]) for e in range(count)]
        new_right = [
            row[e + 1].h_given_first(left[e + 1], right[e + 1]) for e in range(count)
        ]
        return new_left, new_right

    def _log_density_positions(self, values: list[float]) -> float:
        """Return the D-vine log-density (the D-vine likelihood algorithm).

        Equation:
            [EQ-MOD-VINE-002], [EQ-MOD-VINE-004].

        References:
            [REF-AAS-CZADO-FRIGESSI-BAKKEN-2009, Algorithms 1-4].

        """
        d = self.dimension
        left = values[:-1]
        right = values[1:]
        total = 0.0
        for t, row in enumerate(self.pair_copulas):
            for e, pair in enumerate(row):
                total += pair.pair_log_density(left[e], right[e])
            if t < d - 2:
                left, right = self._next_arguments(t, left, right)
        return total

    def _rosenblatt_positions(self, values: list[float]) -> list[float]:
        """Return ``w_k = F(u_k | u_0, ..., u_{k-1})`` from edge 0 of tree ``k``.

        Edge 0 of tree ``k`` joins positions 0 and ``k`` given ``1..k-1``, so
        ``w_k = h_given_first(left[0], right[0])`` of that edge.

        Equation:
            [EQ-MOD-VINE-002], [EQ-MOD-VINE-005].

        References:
            [REF-AAS-CZADO-FRIGESSI-BAKKEN-2009, Algorithms 1-4].

        """
        d = self.dimension
        left = values[:-1]
        right = values[1:]
        w = [values[0]]
        for t, row in enumerate(self.pair_copulas):
            w.append(row[0].h_given_first(left[0], right[0]))
            if t < d - 2:
                left, right = self._next_arguments(t, left, right)
        return w

    def _inverse_positions(self, values: list[float]) -> list[float]:
        """Invert the D-vine Rosenblatt transform (the D-vine sampling algorithm).

        For position ``k`` the chain of edges ``(t, k - 1 - t)``,
        ``t = k - 1, ..., 0``, ends at ``k``; each h-inverse uses the stored
        ``left`` argument ``F(u_{k-1-t} | u_{k-t..k-1})``. After ``u_k`` is
        known the ``left`` arguments needed by position ``k + 1`` are updated
        with ``h_given_second``.

        Equation:
            [EQ-MOD-VINE-002], [EQ-MOD-VINE-005].

        References:
            [REF-AAS-CZADO-FRIGESSI-BAKKEN-2009, Algorithms 1-4].

        """
        d = self.dimension
        trees = self.pair_copulas
        result = [values[0]]
        left: dict[tuple[int, int], float] = {(0, 0): values[0]}
        for k in range(1, d):
            right: dict[int, float] = {}
            current = values[k]
            for t in range(k - 1, -1, -1):
                e = k - 1 - t
                current = trees[t][e].h_inverse_given_first(left[(t, e)], current)
                right[t] = current
            result.append(current)
            if k == d - 1:
                break
            left[(0, k)] = current
            for t in range(k):
                e = k - 1 - t
                left[(t + 1, e)] = trees[t][e].h_given_second(left[(t, e)], right[t])
        return result

    def _sub_vine(self, positions: tuple[int, ...]) -> VineCopula | None:
        """Return the D-vine on a contiguous run of path positions.

        References:
            [REF-AAS-CZADO-FRIGESSI-BAKKEN-2009], [REF-JOE-2014].

        """
        start = positions[0]
        size = len(positions)
        if list(positions) != list(range(start, start + size)):
            return None
        trees = tuple(
            tuple(self.pair_copulas[t][start + e] for e in range(size - 1 - t))
            for t in range(size - 1)
        )
        return DVineCopula(trees)


# ---------------------------------------------------------------------------
# Sequential estimation and selection
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VineEdgeFit:
    """The selected pair copula of one vine edge with its fit statistics.

    Args:
        tree: Tree number, ``1`` for the first tree.
        conditioned: Coordinate indices ``(a, b)`` of the pair; the pair
            copula's first argument belongs to ``a``.
        conditioning: Coordinate indices conditioned on.
        family: Base family name of the selected pair copula.
        rotation: Rotation in degrees.
        parameters: Estimated parameters (see :attr:`PairCopula.parameters`).
        pair_copula: The selected pair copula.
        kendall_tau: Sample Kendall's tau of the edge's pseudo-observations.
        independence_p_value: Two-sided p-value of the asymptotic Kendall's
            tau test of independence.
        log_likelihood: Maximized edge log-likelihood.
        aic: ``2 k - 2 log L`` of the selected family.
        candidates: ``(name, AIC)`` of every family fitted to the edge, in
            the order tried (empty when the independence test decided).
        at_boundary: Whether an estimate lies at a bound of the search range.

    References:
        [REF-DISSMANN-BRECHMANN-CZADO-KUROWICKA-2013], [REF-AKAIKE-1974].

    """

    tree: int
    conditioned: tuple[int, int]
    conditioning: tuple[int, ...]
    family: str
    rotation: int
    parameters: tuple[float, ...]
    pair_copula: PairCopula
    kendall_tau: float
    independence_p_value: float
    log_likelihood: float
    aic: float
    candidates: tuple[tuple[str, float], ...]
    at_boundary: bool


@dataclass(frozen=True, slots=True)
class VineCopulaFit:
    """A sequentially fitted C- or D-vine copula.

    Args:
        copula: The fitted vine.
        structure: ``"c"`` or ``"d"``.
        order: Coordinate index at each vine position.
        edges: Per-edge fits, tree by tree.
        log_likelihood: Total log-likelihood, the sum over edges; equal to
            ``sum_i log c(u_i)`` of the fitted vine on the pseudo-observations.
        aic: ``2 k - 2 log L`` with ``k`` the total parameter count.
        bic: ``k log n - 2 log L``.
        parameter_count: Total number of estimated parameters.
        sample_size: Number of observations ``n``.

    References:
        [REF-DISSMANN-BRECHMANN-CZADO-KUROWICKA-2013],
        [REF-AAS-CZADO-FRIGESSI-BAKKEN-2009], [REF-AKAIKE-1974],
        [REF-SCHWARZ-1978].

    """

    copula: VineCopula
    structure: str
    order: tuple[int, ...]
    edges: tuple[VineEdgeFit, ...]
    log_likelihood: float
    aic: float
    bic: float
    parameter_count: int
    sample_size: int


@dataclass(frozen=True, slots=True)
class _EdgeChoice:
    """Internal record of the family selected for one edge.

    References:
        [REF-DISSMANN-BRECHMANN-CZADO-KUROWICKA-2013]. No equation is evaluated.

    """

    pair: PairCopula
    log_likelihood: float
    aic: float
    tau: float
    p_value: float
    candidates: tuple[tuple[str, float], ...]
    at_boundary: bool


def _pair_log_likelihood(
    pair: PairCopula, first: Sequence[float], second: Sequence[float]
) -> float:
    """Return ``sum_i log c(a_i, b_i)`` of one pair copula.

    Equation:
        [EQ-MOD-VINE-007].

    References:
        [REF-AAS-CZADO-FRIGESSI-BAKKEN-2009, Algorithms 1-4].

    """
    return math.fsum(
        pair.pair_log_density(a, b) for a, b in zip(first, second, strict=True)
    )


def kendall_independence_p_value(tau: float, size: int) -> float:
    """Return the two-sided p-value of the asymptotic Kendall's tau test.

    Under independence of continuous variables the sample tau has mean zero
    and variance ``2 (2 n + 5) / (9 n (n - 1))``; the statistic
    ``z = tau / sqrt(var)`` is asymptotically standard normal and the
    p-value is ``2 Phi(-|z|)``.

    Args:
        tau: Sample Kendall's tau in ``[-1, 1]``.
        size: Sample size of at least two.

    Returns:
        Two-sided p-value in ``[0, 1]``.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``tau`` is outside ``[-1, 1]`` or ``size < 2``.

    Equation:
        [EQ-MOD-VINE-006].

    References:
        [REF-KENDALL-1938], [REF-HOLLANDER-WOLFE-CHICKEN-2014, Chapter 8],
        [REF-DISSMANN-BRECHMANN-CZADO-KUROWICKA-2013].

    """
    value = require_finite_real(tau, name="tau")
    if abs(value) > 1.0:
        raise ValueError("tau must lie in [-1, 1]")
    n = require_non_negative_integer(size, name="size")
    if n < 2:
        raise ValueError("size must be at least two")
    variance = 2.0 * (2.0 * n + 5.0) / (9.0 * n * (n - 1.0))
    return min(1.0, 2.0 * _NORMAL.cdf(-abs(value) / math.sqrt(variance)))


def _bounded_maximum(
    log_likelihood: Callable[[float], float],
    start: float,
    bounds: tuple[float, float],
    *,
    gradient_tolerance: float = 1e-6,
    step_tolerance: float = 1e-9,
) -> tuple[float, float, bool]:
    """Maximize a one-parameter log-likelihood by bounded BFGS.

    Failures of the objective (an invalid parameter or an underflow) count
    as ``-inf`` so that the line search shortens the step. The tolerances
    refer to the unconstrained coordinate of the box transform.

    Returns:
        Arg-max, maximum, and whether the arg-max lies at a bound.

    Equation:
        [EQ-MOD-VINE-007], [EQ-OPT-BFGS-001], [EQ-OPT-BOX-001].

    References:
        [REF-NOCEDAL-WRIGHT-2006, Sections 6.1 and 8.1],
        [REF-DISSMANN-BRECHMANN-CZADO-KUROWICKA-2013].

    """
    low, high = bounds
    width = high - low
    x0 = min(max(start, low + 1e-3 * width), high - 1e-3 * width)

    def objective(x: Vector) -> float:
        """Return the negative log-likelihood, ``inf`` where it fails.

        References:
            [REF-NOCEDAL-WRIGHT-2006, Section 6.1].

        """
        try:
            value = -log_likelihood(x[0])
        except (ValueError, ArithmeticError, OverflowError):
            return math.inf
        return value if math.isfinite(value) else math.inf

    start_value = -objective((x0,))
    if not math.isfinite(start_value):
        grid = [low + width * (k + 0.5) / 16.0 for k in range(16)]
        finite = [(-objective((x,)), x) for x in grid]
        finite = [entry for entry in finite if math.isfinite(entry[0])]
        if not finite:
            return x0, -math.inf, False
        start_value, x0 = max(finite)
    result = minimize_bfgs(
        objective,
        (x0,),
        bounds=((low, high),),
        gradient_tolerance=gradient_tolerance,
        step_tolerance=step_tolerance,
        max_iterations=100,
    )
    best = result.x[0]
    value = -result.value
    if not value >= start_value:
        best, value = x0, start_value
    tolerance = 1e-6 * max(1.0, abs(low), abs(high))
    at_boundary = best - low <= tolerance or high - best <= tolerance
    return best, value, at_boundary


def _t_profile(
    first: Sequence[float], second: Sequence[float], nu: float, rho_start: float
) -> tuple[float, float]:
    """Return the best ``rho`` and the t-pair log-likelihood for fixed ``nu``.

    The scores ``x = T_nu^-1(a)``, ``y = T_nu^-1(b)`` are computed once; the
    bivariate t-copula log-density of [EQ-MOD-COP-T-001] with ``d = 2`` is
    then a cheap function of ``rho``.

    Equation:
        [EQ-MOD-COP-T-001], [EQ-MOD-VINE-007].

    References:
        [REF-DEMARTA-MCNEIL-2005], [REF-NOCEDAL-WRIGHT-2006, Section 6.1].

    """
    law = StudentT(nu)
    x = [float(law.quantile(value)) for value in first]
    y = [float(law.quantile(value)) for value in second]
    count = len(x)
    squares = [a * a + b * b for a, b in zip(x, y, strict=True)]
    products = [a * b for a, b in zip(x, y, strict=True)]
    marginal = math.fsum(
        math.log1p(a * a / nu) + math.log1p(b * b / nu)
        for a, b in zip(x, y, strict=True)
    )
    constant = (
        math.lgamma(0.5 * (nu + 2.0))
        + math.lgamma(0.5 * nu)
        - 2.0 * math.lgamma(0.5 * (nu + 1.0))
    )

    def log_likelihood(rho: float) -> float:
        """Return the bivariate t-copula log-likelihood at ``rho``.

        Equation:
            [EQ-MOD-COP-T-001].

        References:
            [REF-DEMARTA-MCNEIL-2005].

        """
        one_minus = (1.0 - rho) * (1.0 + rho)
        scale = nu * one_minus
        quadratic = math.fsum(
            math.log1p((s - 2.0 * rho * p) / scale)
            for s, p in zip(squares, products, strict=True)
        )
        return (
            count * (constant - 0.5 * math.log(one_minus))
            - 0.5 * (nu + 2.0) * quadratic
            + 0.5 * (nu + 1.0) * marginal
        )

    rho, value, _ = _bounded_maximum(
        log_likelihood, rho_start, (-_RHO_BOUND, _RHO_BOUND)
    )
    return rho, value


def _fit_student_t(
    first: Sequence[float], second: Sequence[float], tau: float
) -> tuple[PairCopula, float, bool]:
    """Fit the bivariate t pair copula by profile maximum likelihood.

    The outer bounded BFGS search runs over ``nu`` in ``[1, 50]``; for each
    ``nu`` the inner search maximizes over ``rho`` from the Kendall start
    ``sin(pi tau / 2)``.

    Equation:
        [EQ-MOD-COP-T-001], [EQ-MOD-COPULA-FIT-005], [EQ-MOD-VINE-007].

    References:
        [REF-DEMARTA-MCNEIL-2005], [REF-DISSMANN-BRECHMANN-CZADO-KUROWICKA-2013].

    """
    rho_start = kendall_to_latent_correlation(tau)
    cache: dict[float, tuple[float, float]] = {}

    def profile(log_nu: float) -> float:
        """Return the profile log-likelihood at ``nu = exp(log_nu)``.

        References:
            [REF-DEMARTA-MCNEIL-2005].

        """
        if log_nu not in cache:
            cache[log_nu] = _t_profile(first, second, math.exp(log_nu), rho_start)
        return cache[log_nu][1]

    log_bounds = (math.log(_NU_BOUNDS[0]), math.log(_NU_BOUNDS[1]))
    log_nu, value, at_boundary = _bounded_maximum(
        profile,
        math.log(6.0),
        log_bounds,
        gradient_tolerance=1e-4,
        step_tolerance=1e-6,
    )
    if log_nu not in cache:
        profile(log_nu)
    rho = cache[log_nu][0]
    pair = PairCopula(StudentTCopula(((1.0, rho), (rho, 1.0)), math.exp(log_nu)))
    return pair, value, at_boundary


def _fit_one_parameter(
    family: str,
    rotation: int,
    first: Sequence[float],
    second: Sequence[float],
    tau: float,
) -> tuple[PairCopula, float, bool]:
    """Fit a one-parameter pair copula from its Kendall's tau start.

    Equation:
        [EQ-MOD-COPULA-FIT-004], [EQ-MOD-COPULA-FIT-005], [EQ-MOD-VINE-007].

    References:
        [REF-GENEST-RIVEST-1993], [REF-DISSMANN-BRECHMANN-CZADO-KUROWICKA-2013].

    """
    magnitude = abs(tau)
    bounds: tuple[float, float]
    if family == "gaussian":
        start = kendall_to_latent_correlation(tau)
        bounds = (-_RHO_BOUND, _RHO_BOUND)
    elif family == "frank":
        start = archimedean_theta_from_tau("frank", tau)
        low, high = _FRANK_BOUNDS
        bounds = (low, high) if tau > 0.0 else (-high, -low)
    elif family == "clayton":
        start = archimedean_theta_from_tau("clayton", magnitude)
        bounds = _CLAYTON_BOUNDS
    else:
        start = archimedean_theta_from_tau("gumbel", magnitude)
        bounds = _GUMBEL_BOUNDS

    def log_likelihood(parameter: float) -> float:
        """Return the edge log-likelihood at one parameter value.

        Equation:
            [EQ-MOD-VINE-007].

        References:
            [REF-AAS-CZADO-FRIGESSI-BAKKEN-2009].

        """
        pair = PairCopula.from_parameters(family, (parameter,), rotation)
        return _pair_log_likelihood(pair, first, second)

    best, value, at_boundary = _bounded_maximum(log_likelihood, start, bounds)
    return PairCopula.from_parameters(family, (best,), rotation), value, at_boundary


def _rotations_for(tau: float, rotations: bool) -> tuple[int, ...]:
    """Return the Clayton/Gumbel rotations whose tau sign matches the sample.

    References:
        [REF-NELSEN-2006, Section 2.4]. No equation is evaluated.

    """
    if tau > 0.0:
        return (0, 180) if rotations else (0,)
    if tau < 0.0 and rotations:
        return (90, 270)
    return ()


def _select_edge(
    first: Sequence[float],
    second: Sequence[float],
    families: tuple[str, ...],
    *,
    rotations: bool,
    independence_test: bool,
    level: float,
    tau: float | None = None,
) -> _EdgeChoice:
    """Select and fit the pair copula of one edge by AIC.

    Equation:
        [EQ-MOD-VINE-006], [EQ-MOD-VINE-007], [EQ-INF-IC-001].

    Raises:
        ValueError: If no candidate family can represent the sample tau.

    References:
        [REF-DISSMANN-BRECHMANN-CZADO-KUROWICKA-2013], [REF-AKAIKE-1974].

    """
    size = len(first)
    sample_tau = kendall_tau(first, second) if tau is None else tau
    p_value = kendall_independence_p_value(sample_tau, size)
    independence = PairCopula(IndependenceCopula(2))
    if (independence_test and p_value >= level) or sample_tau == 0.0:
        return _EdgeChoice(independence, 0.0, 0.0, sample_tau, p_value, (), False)
    fitted: list[tuple[PairCopula, float, bool]] = []
    for family in families:
        if family == "independence":
            fitted.append((independence, 0.0, False))
        elif family == "student-t":
            fitted.append(_fit_student_t(first, second, sample_tau))
        elif family in ("gaussian", "frank"):
            fitted.append(_fit_one_parameter(family, 0, first, second, sample_tau))
        else:
            for rotation in _rotations_for(sample_tau, rotations):
                fitted.append(
                    _fit_one_parameter(family, rotation, first, second, sample_tau)
                )
    if not fitted:
        raise ValueError(
            f"no candidate family can represent Kendall's tau {sample_tau:.3g}; "
            "add 'gaussian', 'frank' or 'independence', or allow rotations"
        )
    scored = [
        (pair, value, 2.0 * pair.parameter_count - 2.0 * value, boundary)
        for pair, value, boundary in fitted
    ]
    best = min(range(len(scored)), key=lambda k: scored[k][2])
    pair, value, aic, boundary = scored[best]
    candidates = tuple((entry[0].name, entry[2]) for entry in scored)
    return _EdgeChoice(pair, value, aic, sample_tau, p_value, candidates, boundary)


def _validated_rows(
    data_u: Sequence[Sequence[float]], rank_transform: bool
) -> tuple[tuple[float, ...], ...]:
    """Validate the data and return its columns as pseudo-observations.

    Raises:
        TypeError: If the data are malformed or not real.
        ValueError: If rows have unequal length, fewer than three rows or two
            columns are given, or (without rank transform) a value lies
            outside ``(0, 1)``.

    Equation:
        [EQ-MOD-COPULA-FIT-002], [EQ-STAT-RANK-001].

    References:
        [REF-GENEST-GHOUDI-RIVEST-1995].

    """
    if isinstance(data_u, str | bytes) or not isinstance(data_u, Sequence):
        raise TypeError("data_u must be a sequence of rows")
    rows = []
    for row in data_u:
        if isinstance(row, str | bytes) or not isinstance(row, Sequence):
            raise TypeError("each row of data_u must be a sequence of numbers")
        rows.append(tuple(require_finite_real(value, name="data_u") for value in row))
    if len(rows) < 3:
        raise ValueError("vine fitting needs at least three rows")
    dimension = len(rows[0])
    if dimension < 2:
        raise ValueError("vine fitting needs at least two columns")
    if any(len(row) != dimension for row in rows):
        raise ValueError("every row of data_u must have the same length")
    columns = tuple(zip(*rows, strict=True))
    if not rank_transform:
        for column in columns:
            if any(not 0.0 < value < 1.0 for value in column):
                raise ValueError("data_u must lie in (0, 1) without rank transform")
        return tuple(tuple(column) for column in columns)
    scale = len(rows) + 1.0
    return tuple(tuple(r / scale for r in ranks(column)) for column in columns)


def _greedy_path(taus: dict[tuple[int, int], float], dimension: int) -> tuple[int, ...]:
    """Return a D-vine order by a greedy maximum-``|tau|`` path.

    Start from the pair with the largest ``|tau|`` and repeatedly attach, at
    either end, the unused variable with the largest ``|tau|`` to that end
    (ties broken by the lower index).

    Equation:
        [EQ-MOD-VINE-008].

    References:
        [REF-DISSMANN-BRECHMANN-CZADO-KUROWICKA-2013].

    """

    def weight(i: int, j: int) -> float:
        """Return ``|tau_ij|``.

        References:
            [REF-DISSMANN-BRECHMANN-CZADO-KUROWICKA-2013].

        """
        return abs(taus[(min(i, j), max(i, j))])

    pairs = sorted(taus, key=lambda key: (-abs(taus[key]), key))
    path = list(pairs[0])
    unused = [k for k in range(dimension) if k not in path]
    while unused:
        best: tuple[float, int, int] | None = None
        for k in unused:
            for end in (0, 1):
                tip = path[0] if end == 0 else path[-1]
                candidate = (weight(k, tip), -k, -end)
                if best is None or candidate > best:
                    best = candidate
        assert best is not None
        _, negative_k, negative_end = best
        k = -negative_k
        if negative_end == 0:
            path.insert(0, k)
        else:
            path.append(k)
        unused.remove(k)
    return tuple(path)


def _validated_fixed_order(
    order: Sequence[int] | None, dimension: int
) -> tuple[int, ...] | None:
    """Validate a user-given vine order.

    Raises:
        ValueError: If the order is not a permutation of ``0, ..., d - 1``.

    References:
        Project validation policy. No equation is evaluated.

    """
    if order is None:
        return None
    chosen = tuple(require_non_negative_integer(i, name="order") for i in order)
    if sorted(chosen) != list(range(dimension)):
        raise ValueError("order must be a permutation of 0, ..., d - 1")
    return chosen


def _validated_families(families: Sequence[str]) -> tuple[str, ...]:
    """Validate the candidate family names.

    Raises:
        TypeError: If ``families`` is a bare string.
        ValueError: If it is empty, repeats, or names an unknown family.

    References:
        Project validation policy. No equation is evaluated.

    """
    if isinstance(families, str):
        raise TypeError("families must be a sequence of family names")
    chosen = tuple(families)
    if not chosen or len(set(chosen)) != len(chosen):
        raise ValueError("families must be a non-empty set of family names")
    for family in chosen:
        if family not in PAIR_COPULA_FAMILIES:
            raise ValueError(f"family must be one of {PAIR_COPULA_FAMILIES}")
    return chosen


def fit_vine_copula(
    data_u: Sequence[Sequence[float]],
    structure: VineStructure = "c",
    families: Sequence[str] = PAIR_COPULA_FAMILIES,
    *,
    rotations: bool = True,
    independence_test: bool = True,
    level: float = 0.05,
    order: Sequence[int] | None = None,
    rank_transform: bool = True,
) -> VineCopulaFit:
    """Fit a C- or D-vine copula by sequential maximum likelihood.

    1. Each column is replaced by its rank pseudo-observations
       ``rank / (n + 1)`` (unless ``rank_transform=False``).
    2. Structure: a D-vine order is a greedy maximum-``|tau|`` path over the
       pairwise sample Kendall's tau; a C-vine root in each tree is the
       remaining variable with the largest sum of ``|tau|`` to the other
       remaining variables, computed on that tree's conditional
       pseudo-observations. A given ``order`` replaces both heuristics.
    3. Tree by tree, every edge is fitted on its pseudo-observations. If
       ``independence_test`` is set and the Kendall's tau test does not
       reject independence at ``level``, the independence copula is chosen.
       Otherwise each candidate family (with the Clayton/Gumbel rotations
       matching the sign of tau) is fitted by bounded BFGS from its Kendall's
       tau inversion (the t copula by profile likelihood over ``nu``), and
       the smallest AIC wins.
    4. The h-functions of the selected pair copulas give the
       pseudo-observations of the next tree.

    Search ranges: ``|rho| <= 0.999``; Clayton ``theta`` in ``[1e-4, 28]``;
    Gumbel ``theta`` in ``[1 + 1e-6, 20]``; Frank ``1e-4 <= |theta| <= 50``;
    t-copula ``nu`` in ``[1, 50]``. A maximum on a bound is flagged per edge.
    Sequential estimates are consistent but not the joint maximum-likelihood
    estimates; no standard errors are reported.

    Args:
        data_u: Rows of observations (``n >= 3`` rows, ``d >= 2`` columns);
            raw data or pseudo-observations.
        structure: ``"c"`` (canonical vine) or ``"d"`` (drawable vine).
        families: Candidate base families among :data:`PAIR_COPULA_FAMILIES`.
        rotations: Whether the Clayton and Gumbel candidates include their
            rotations (90/270 for negative tau, 180 for positive tau).
        independence_test: Whether to choose independence when the
            Kendall's tau test does not reject it.
        level: Significance level of the independence test, in ``(0, 1)``.
        order: Optional fixed vine order (coordinate index at each position).
        rank_transform: Whether to replace the data by rank
            pseudo-observations; if ``False`` they must lie in ``(0, 1)``.

    Returns:
        Fitted vine with per-edge families, parameters, log-likelihoods and
        AICs, and the total log-likelihood, AIC and BIC.

    Raises:
        TypeError: If the data or options are malformed.
        ValueError: If the structure, a family, the level or the order is
            invalid, the data are too small or constant, or no candidate
            family can represent an edge.

    Equation:
        [EQ-MOD-VINE-006], [EQ-MOD-VINE-007], [EQ-MOD-VINE-008],
        [EQ-MOD-COPULA-FIT-002], [EQ-INF-IC-001], [EQ-INF-IC-003].

    References:
        [REF-DISSMANN-BRECHMANN-CZADO-KUROWICKA-2013],
        [REF-AAS-CZADO-FRIGESSI-BAKKEN-2009, Algorithms 1-4],
        [REF-GENEST-GHOUDI-RIVEST-1995], [REF-AKAIKE-1974], [REF-SCHWARZ-1978].

    """
    if structure not in ("c", "d"):
        raise ValueError("structure must be 'c' or 'd'")
    candidates = _validated_families(families)
    alpha = require_finite_real(level, name="level")
    if not 0.0 < alpha < 1.0:
        raise ValueError("level must lie in (0, 1)")
    if not isinstance(rotations, bool) or not isinstance(independence_test, bool):
        raise TypeError("rotations and independence_test must be booleans")
    columns = _validated_rows(data_u, rank_transform)
    dimension = len(columns)
    size = len(columns[0])
    fixed = _validated_fixed_order(order, dimension)

    def select(
        first: Sequence[float], second: Sequence[float], tau: float | None = None
    ) -> _EdgeChoice:
        """Select one edge with the shared options.

        References:
            [REF-DISSMANN-BRECHMANN-CZADO-KUROWICKA-2013].

        """
        return _select_edge(
            first,
            second,
            candidates,
            rotations=rotations,
            independence_test=independence_test,
            level=alpha,
            tau=tau,
        )

    choices: dict[tuple[int, int], _EdgeChoice]
    if structure == "c":
        vine_order, choices = _fit_c_vine(columns, fixed, select)
        trees = tuple(
            tuple(
                choices[(vine_order[t], vine_order[t + 1 + e])].pair
                for e in range(dimension - 1 - t)
            )
            for t in range(dimension - 1)
        )
        copula: VineCopula = CVineCopula(trees, vine_order)
    else:
        vine_order, rows = _fit_d_vine(columns, fixed, select)
        copula = DVineCopula(
            tuple(tuple(c.pair for c in row) for row in rows), vine_order
        )
        choices = {}
        for t, row in enumerate(rows):
            for e, choice in enumerate(row):
                choices[(vine_order[e], vine_order[e + t + 1])] = choice
    edges = []
    for edge in copula.edges:
        choice = choices[edge.conditioned]
        edges.append(
            VineEdgeFit(
                edge.tree,
                edge.conditioned,
                edge.conditioning,
                choice.pair.base_family,
                choice.pair.rotation,
                choice.pair.parameters,
                choice.pair,
                choice.tau,
                choice.p_value,
                choice.log_likelihood,
                choice.aic,
                choice.candidates,
                choice.at_boundary,
            )
        )
    total = math.fsum(edge.log_likelihood for edge in edges)
    count = sum(edge.pair_copula.parameter_count for edge in edges)
    return VineCopulaFit(
        copula,
        structure,
        copula.vine_order,
        tuple(edges),
        total,
        2.0 * count - 2.0 * total,
        count * math.log(size) - 2.0 * total,
        count,
        size,
    )


_Selector = Callable[[Sequence[float], Sequence[float], "float | None"], _EdgeChoice]


def _fit_c_vine(
    columns: tuple[tuple[float, ...], ...],
    fixed: tuple[int, ...] | None,
    select: _Selector,
) -> tuple[tuple[int, ...], dict[tuple[int, int], _EdgeChoice]]:
    """Fit a C-vine tree by tree, choosing each root by the largest ``sum |tau|``.

    Returns:
        The vine order and the edge choices keyed by ``(root, partner)``.

    Equation:
        [EQ-MOD-VINE-002], [EQ-MOD-VINE-008].

    References:
        [REF-DISSMANN-BRECHMANN-CZADO-KUROWICKA-2013],
        [REF-AAS-CZADO-FRIGESSI-BAKKEN-2009, Algorithms 1-4].

    """
    dimension = len(columns)
    current: dict[int, Sequence[float]] = dict(enumerate(columns))
    roots: list[int] = []
    choices: dict[tuple[int, int], _EdgeChoice] = {}
    for t in range(dimension - 1):
        remaining = [k for k in range(dimension) if k not in roots]
        taus: dict[tuple[int, int], float] = {}
        if fixed is None:
            for a_index, a in enumerate(remaining):
                for b in remaining[a_index + 1 :]:
                    taus[(a, b)] = kendall_tau(current[a], current[b])
            root = max(
                remaining,
                key=lambda k: (
                    math.fsum(
                        abs(taus[(min(k, j), max(k, j))]) for j in remaining if j != k
                    ),
                    -k,
                ),
            )
        else:
            root = fixed[t]
        roots.append(root)
        updated: dict[int, Sequence[float]] = {}
        for partner in remaining:
            if partner == root:
                continue
            key = (min(root, partner), max(root, partner))
            choice = select(current[root], current[partner], taus.get(key))
            choices[(root, partner)] = choice
            if t < dimension - 2:
                pair = choice.pair
                updated[partner] = [
                    pair.h_given_first(a, b)
                    for a, b in zip(current[root], current[partner], strict=True)
                ]
        current = updated
    roots.extend(k for k in range(dimension) if k not in roots)
    return tuple(roots), choices


def _fit_d_vine(
    columns: tuple[tuple[float, ...], ...],
    fixed: tuple[int, ...] | None,
    select: _Selector,
) -> tuple[tuple[int, ...], list[list[_EdgeChoice]]]:
    """Fit a D-vine tree by tree along a greedy maximum-``|tau|`` path.

    Returns:
        The vine order and the edge choices per tree.

    Equation:
        [EQ-MOD-VINE-002], [EQ-MOD-VINE-008].

    References:
        [REF-DISSMANN-BRECHMANN-CZADO-KUROWICKA-2013],
        [REF-AAS-CZADO-FRIGESSI-BAKKEN-2009, Algorithms 1-4].

    """
    dimension = len(columns)
    taus: dict[tuple[int, int], float] = {}
    if fixed is None:
        for a in range(dimension):
            for b in range(a + 1, dimension):
                taus[(a, b)] = kendall_tau(columns[a], columns[b])
        order = _greedy_path(taus, dimension) if dimension > 2 else (0, 1)
    else:
        order = fixed
    values: list[Sequence[float]] = [columns[k] for k in order]
    left: list[Sequence[float]] = values[:-1]
    right: list[Sequence[float]] = values[1:]
    rows: list[list[_EdgeChoice]] = []
    for t in range(dimension - 1):
        row = []
        for e in range(dimension - 1 - t):
            known = None
            if t == 0:
                a, b = order[e], order[e + 1]
                known = taus.get((min(a, b), max(a, b)))
            row.append(select(left[e], right[e], known))
        rows.append(row)
        if t < dimension - 2:
            count = len(row) - 1
            new_left: list[Sequence[float]] = [
                [
                    row[e].pair.h_given_second(a, b)
                    for a, b in zip(left[e], right[e], strict=True)
                ]
                for e in range(count)
            ]
            new_right: list[Sequence[float]] = [
                [
                    row[e + 1].pair.h_given_first(a, b)
                    for a, b in zip(left[e + 1], right[e + 1], strict=True)
                ]
                for e in range(count)
            ]
            left, right = new_left, new_right
    return order, rows
