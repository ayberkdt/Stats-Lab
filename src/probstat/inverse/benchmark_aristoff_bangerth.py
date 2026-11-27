"""Aristoff-Bangerth benchmark: Bayesian inversion of PDE coefficients.

Implements the community benchmark of Aristoff and Bangerth (SIAM Review 65(4),
1074-1105, 2023; arXiv:2102.07263) exactly as specified in its Section 2.

Forward model (all quantities dimensionless, as in the paper): the Poisson-type
problem ``-div(a grad u) = f`` in ``Omega = (0, 1)^2``, ``u = 0`` on the
boundary, ``f = 10``, with ``a`` piecewise constant on a uniform ``8 x 8`` grid
(cell ``k = 8 row + col``, ``row`` counted upward from the bottom edge, ``col``
from the left edge; ``a|_cell(k) = theta_k``) [EQ-INV-ABBENCH-001]. It is
discretized by bilinear (Q1) finite elements on a uniform ``32 x 32`` mesh
(1089 nodes, 961 unknowns; ``fine_cells_per_cell = 4`` per coefficient cell per
side) and observed through the nodal-interpolation operator ``M`` at the
``13 x 13`` points ``x = (i, j) / 14``, ``1 <= i, j <= 13``
[EQ-INV-ABBENCH-002]. The data model is ``z = M U(theta) + N(0, 0.05^2 I)``,
the likelihood is ``exp(-|z_hat - z(theta)|^2 / (2 * 0.05^2))`` and the prior
is ``prod_i exp(-(ln theta_i)^2 / (2 * 2^2))`` taken literally as a function of
``theta`` (paper Eq. (2.6), no Jacobian factor) [EQ-INV-ABBENCH-003]; the
posterior is their product [EQ-INV-ABBENCH-004]. The measurement vector
``z_hat`` (paper Table 2.1) and the reference posterior means with their
2-sigma uncertainties (paper Table 3.1; Metropolis-Hastings, ``2e11`` samples)
are transcribed as module constants.

:class:`AristoffBangerthForward` implements the ``DifferentiableForward``
protocol of :mod:`probstat.inverse.gauss_newton` with the value, the Jacobian
action ``J v = -M A^-1 (sum_k v_k A_k) u`` and the exact discrete adjoint
``J' w = -(p' A_k u)_k`` with ``p = A^-1 M' w`` [EQ-INV-ABBENCH-005]. The
linear solver is a pure-Python banded Cholesky factorization (half-bandwidth
``32``): :mod:`probstat.linalg` offers dense Cholesky and Krylov solvers but no
banded or sparse direct solver, and a dense ``961 x 961`` Cholesky is far too
slow. The factorization is cached for one ``theta`` so that ``value``, ``jvp``
and ``vjp`` at the same point share it. The tests use
``probstat.linalg.solve_spd`` and ``conjugate_gradient`` as independent oracles.

Literal readings and ambiguities (all documented, none tuned):

* Paper Section 2.2.1 writes the observation points as
  ``x_k = (i/14, j/14)`` with ``k = 13 (i - 1) + (j - 1)`` (``y`` runs
  fastest), while the numbers printed in Figure 2.1 look like ``x`` running
  fastest. This module follows the text formula, which is also the ordering of
  the reference code's published test vectors (checked in the tests). Because
  the problem is symmetric about the diagonal and ``z_hat`` is symmetric, the
  two readings define the same posterior up to ``theta -> theta^T``.
* The prior (2.6) is a (non-normalized) density in ``theta``. In
  ``t = ln theta`` it is ``N(sigma_pr^2, sigma_pr^2)`` (the paper's "mean
  ``sigma_pr^2``", mean of ``theta_k`` about 403.43), so a Metropolis chain in
  ``t`` must include the factor ``prod theta~/theta`` exactly as paper
  Appendix A.2 does; pCN in the standard-normal variable ``u`` with
  ``t = sigma_pr^2 + sigma_pr u`` targets the same posterior.
* The "true" parameters ``theta_hat`` (``a = 1`` except ``0.1`` on cells
  9, 10, 17, 18 and ``10`` on cells 45, 46, 53, 54) are taken from the
  reference code's ``main()`` (deal.II code gallery, MCMC-Laplace), consistent
  with the paper's Figure 2.2; the paper text itself gives only the pictures.
  ``z_hat`` was generated with a ``256 x 256`` Q3 model, so ``z(theta_hat)``
  from this module (Q1, ``32 x 32``) is deliberately not equal to ``z_hat``.

Assumptions:
    ``theta > 0`` componentwise. Extreme contrast can make the Cholesky pivot
    non-positive in floating point; ``value`` then returns NaN entries and the
    derivative actions raise ``ValueError`` (no jitter or fallback).

Experimental (added in 0.28).

References:
    [REF-ARISTOFF-BANGERTH-2023], [REF-ELMAN-SILVESTER-WATHEN-2014],
    [REF-GOLUB-VANLOAN-2013], [REF-METROPOLIS-ETAL-1953],
    [REF-HASTINGS-1970], [REF-STUART-2010].

"""

from __future__ import annotations

import math
import operator
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

from probstat.foundations.random_streams import RandomStream, resolve_stream
from probstat.linalg import Vector

GRID_CELLS: int = 8
"""Coefficient cells per side of the unit square."""

PARAMETER_DIMENSION: int = GRID_CELLS * GRID_CELLS
"""Number of parameters (64)."""

MEASUREMENT_POINTS_PER_SIDE: int = 13
"""Observation points per side (13)."""

DATA_DIMENSION: int = MEASUREMENT_POINTS_PER_SIDE**2
"""Number of measurements (169)."""

PAPER_FINE_CELLS_PER_CELL: int = 4
"""Fine cells per coefficient cell per side: the paper's 32 x 32 mesh."""

FORCING: float = 10.0
"""Constant right-hand side ``f`` (paper Section 2.2.1)."""

NOISE_STD: float = 0.05
"""Measurement standard deviation ``sigma`` (paper Eq. (2.5))."""

PRIOR_LOG_STD: float = 2.0
"""Prior parameter ``sigma_pr`` (paper Eq. (2.6))."""

PRIOR_LOG_MEAN: float = PRIOR_LOG_STD**2
"""Mean of ``ln theta_k`` under the prior as a density in ``theta``."""

PROPOSAL_STD: float = 0.0725
"""Log-space proposal standard deviation of the paper's sampler (A.2)."""

TRUE_DATA: tuple[float, ...] = (
    0.06076511762259369,
    0.09601910120848481,
    0.1238852517838584,
    0.1495184117375201,
    0.1841596127549784,
    0.2174525028261122,
    0.2250996160898698,
    0.2197954769002993,
    0.2074695698370926,
    0.1889996477663016,
    0.1632722532153726,
    0.1276782480038186,
    0.07711845915789312,
    0.09601910120848552,
    0.2000589533367983,
    0.3385592591951766,
    0.3934300024647806,
    0.4040223892461541,
    0.4122329537843092,
    0.4100480091545554,
    0.3949151637189968,
    0.3697873264791232,
    0.33401826235924,
    0.2850397806663382,
    0.2184260032478671,
    0.1271121156350957,
    0.1238852517838611,
    0.3385592591951819,
    0.7119285162766475,
    0.8175712861756428,
    0.6836254116578105,
    0.5779452419831157,
    0.5555615956136897,
    0.5285181561736719,
    0.491439702849224,
    0.4409367494853282,
    0.3730060082060772,
    0.2821694983395214,
    0.1610176733857739,
    0.1495184117375257,
    0.3934300024647929,
    0.8175712861756562,
    0.9439154625527653,
    0.8015904115095128,
    0.6859683749254024,
    0.6561235366960599,
    0.6213197201867315,
    0.5753611315000049,
    0.5140091754526823,
    0.4325325506354165,
    0.3248315148915482,
    0.1834600412730086,
    0.1841596127549917,
    0.4040223892461832,
    0.6836254116578439,
    0.8015904115095396,
    0.7870119561144977,
    0.7373108331395808,
    0.7116558878070463,
    0.6745179049094283,
    0.6235300574156917,
    0.5559332704045935,
    0.4670304994474178,
    0.3499809143811,
    0.19688263746294,
    0.2174525028261253,
    0.4122329537843404,
    0.5779452419831566,
    0.6859683749254372,
    0.7373108331396063,
    0.7458811983178246,
    0.7278968022406559,
    0.6904793535357751,
    0.6369176452710288,
    0.5677443693743215,
    0.4784738764865867,
    0.3602190632823262,
    0.2031792054737325,
    0.2250996160898818,
    0.4100480091545787,
    0.5555615956137137,
    0.6561235366960938,
    0.7116558878070715,
    0.727896802240657,
    0.7121928678670187,
    0.6712187391428729,
    0.6139157775591492,
    0.5478251665295381,
    0.4677122687599031,
    0.3587654911000848,
    0.2050734291675918,
    0.2197954769003094,
    0.3949151637190157,
    0.5285181561736911,
    0.6213197201867471,
    0.6745179049094407,
    0.690479353535786,
    0.6712187391428787,
    0.6178408289359514,
    0.5453605027237883,
    0.489575966490909,
    0.4341716881061278,
    0.3534389974779456,
    0.2083227496961347,
    0.207469569837099,
    0.3697873264791366,
    0.4914397028492412,
    0.5753611315000203,
    0.6235300574157017,
    0.6369176452710497,
    0.6139157775591579,
    0.5453605027237935,
    0.4336604929612851,
    0.4109641743019312,
    0.3881864790111245,
    0.3642640090182592,
    0.2179599909280145,
    0.1889996477663011,
    0.3340182623592461,
    0.4409367494853381,
    0.5140091754526943,
    0.5559332704045969,
    0.5677443693743304,
    0.5478251665295453,
    0.4895759664908982,
    0.4109641743019171,
    0.395727260284338,
    0.3778949322004734,
    0.3596268271857124,
    0.2191250268948948,
    0.1632722532153683,
    0.2850397806663325,
    0.373006008206081,
    0.4325325506354207,
    0.4670304994474315,
    0.4784738764866023,
    0.4677122687599041,
    0.4341716881061055,
    0.388186479011099,
    0.3778949322004602,
    0.3633362567187364,
    0.3464457261905399,
    0.2096362321365655,
    0.1276782480038148,
    0.2184260032478634,
    0.2821694983395252,
    0.3248315148915535,
    0.3499809143811097,
    0.3602190632823333,
    0.3587654911000799,
    0.3534389974779268,
    0.3642640090182283,
    0.35962682718569,
    0.3464457261905295,
    0.3260728953424643,
    0.180670595355394,
    0.07711845915789244,
    0.1271121156350963,
    0.1610176733857757,
    0.1834600412730144,
    0.1968826374629443,
    0.2031792054737354,
    0.2050734291675885,
    0.2083227496961245,
    0.2179599909279998,
    0.2191250268948822,
    0.2096362321365551,
    0.1806705953553887,
    0.1067965550010013,
)
"""Measurement vector ``z_hat`` (paper Table 2.1, section 2.2.4), 169 entries.

Point ordering ``k = 13 (i - 1) + (j - 1)``, ``x = i / 14``; the table is
symmetric under exchanging ``i`` and ``j``, so the ordering is immaterial.
"""

# Paper Table 3.1 (Section 3.2): sample means <theta_k> of the posterior, from
# N = 2000 Metropolis-Hastings chains of length 1e8, and the estimated
# 2-sigma uncertainties of those means (printed as "mean +- 2 sigma").
REFERENCE_POSTERIOR_MEAN: tuple[float, ...] = (
    76.32,
    1.2104,
    0.97738,
    0.882007,
    0.971859,
    0.947832,
    1.08529,
    11.39,
    1.119,
    0.0937215,
    0.1157992,
    0.5815,
    0.9472,
    6.258,
    9.334,
    1.08151,
    0.977449,
    0.1157962,
    0.461,
    267.01,
    30.87,
    7.189,
    12.39,
    0.949863,
    0.881977,
    0.5828,
    267.72,
    369.35,
    234.59,
    13.29,
    22.36,
    0.988806,
    0.9719,
    0.9509,
    30.76,
    233.93,
    1.169,
    0.8327,
    88.52,
    0.987809,
    0.947816,
    6.26,
    7.119,
    13.2,
    0.8327,
    176.73,
    283.38,
    0.914212,
    1.08521,
    9.386,
    12.44,
    22.5,
    88.57,
    283.41,
    218.65,
    0.933451,
    11.35,
    1.08143,
    0.949869,
    0.98877,
    0.987866,
    0.914247,
    0.933426,
    1.59984,
)
"""Posterior means ``<theta_k>`` (paper Table 3.1), ``k = 0, ..., 63``."""

REFERENCE_POSTERIOR_MEAN_TWO_SIGMA: tuple[float, ...] = (
    0.3,
    0.0094,
    5.1e-05,
    3.9e-05,
    4.8e-05,
    6.4e-05,
    0.00011,
    0.1,
    0.011,
    2.7e-06,
    3.9e-06,
    0.0022,
    0.0079,
    0.079,
    0.09,
    0.00011,
    5.2e-05,
    3.8e-06,
    0.02,
    0.55,
    0.19,
    0.089,
    0.11,
    7.3e-05,
    3.9e-05,
    0.002,
    0.62,
    0.64,
    0.53,
    0.14,
    0.16,
    7.4e-05,
    4.9e-05,
    0.0079,
    0.19,
    0.52,
    0.012,
    0.0057,
    0.33,
    7.9e-05,
    6.5e-05,
    0.076,
    0.087,
    0.13,
    0.0035,
    0.44,
    0.58,
    7.7e-05,
    0.00011,
    0.089,
    0.12,
    0.17,
    0.33,
    0.57,
    0.49,
    8.7e-05,
    0.11,
    0.00011,
    7.4e-05,
    7.4e-05,
    8.3e-05,
    7.7e-05,
    8.7e-05,
    0.0003,
)
"""Estimated 2-sigma uncertainty of each Table 3.1 mean."""

TRUE_THETA: tuple[float, ...] = tuple(
    0.1 if k in (9, 10, 17, 18) else 10.0 if k in (45, 46, 53, 54) else 1.0
    for k in range(PARAMETER_DIMENSION)
)
"""Coefficients ``theta_hat`` used to generate ``z_hat`` (see module notes)."""

_LOCAL_STIFFNESS: tuple[tuple[float, ...], ...] = (
    (4.0 / 6.0, -1.0 / 6.0, -2.0 / 6.0, -1.0 / 6.0),
    (-1.0 / 6.0, 4.0 / 6.0, -1.0 / 6.0, -2.0 / 6.0),
    (-2.0 / 6.0, -1.0 / 6.0, 4.0 / 6.0, -1.0 / 6.0),
    (-1.0 / 6.0, -2.0 / 6.0, -1.0 / 6.0, 4.0 / 6.0),
)


def _as_theta(theta: object, name: str = "theta") -> Vector:
    """Validate a positive finite parameter vector of length 64.

    Args:
        theta: Candidate sequence.
        name: Name used in messages.

    Returns:
        The entries as a tuple of floats.

    Raises:
        TypeError: If ``theta`` is not a sequence of real numbers.
        ValueError: If the length is not 64 or an entry is not positive and
            finite.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].

    """
    if isinstance(theta, (str, bytes)) or not isinstance(theta, Sequence):
        raise TypeError(f"{name} must be a sequence of real numbers")
    out = []
    for entry in theta:
        if isinstance(entry, bool) or not isinstance(entry, (int, float)):
            raise TypeError(f"{name} must contain real numbers")
        out.append(float(entry))
    if len(out) != PARAMETER_DIMENSION:
        raise ValueError(f"{name} must have length {PARAMETER_DIMENSION}")
    if not all(math.isfinite(v) and v > 0.0 for v in out):
        raise ValueError(f"{name} must be positive and finite")
    return tuple(out)


def _as_real_vector(values: object, size: int, name: str) -> Vector:
    """Validate a finite real vector of a given length.

    Args:
        values: Candidate sequence.
        size: Required length.
        name: Name used in messages.

    Returns:
        The entries as a tuple of floats.

    Raises:
        TypeError: If ``values`` is not a sequence of real numbers.
        ValueError: If the length is wrong or an entry is not finite.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].

    """
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} must be a sequence of real numbers")
    out = []
    for entry in values:
        if isinstance(entry, bool) or not isinstance(entry, (int, float)):
            raise TypeError(f"{name} must contain real numbers")
        out.append(float(entry))
    if len(out) != size:
        raise ValueError(f"{name} must have length {size}")
    if not all(math.isfinite(v) for v in out):
        raise ValueError(f"{name} must be finite")
    return tuple(out)


def _band_cholesky(rows: list[list[float]], bandwidth: int) -> bool:
    """Factorize a banded SPD matrix in place by Cholesky (``A = L L'``).

    ``rows[i]`` holds ``A[i][max(0, i - bandwidth) .. i]`` (lower triangle) and
    is overwritten by the same slice of ``L``.

    Args:
        rows: Lower band rows, modified in place.
        bandwidth: Half-bandwidth of the matrix.

    Returns:
        ``True`` on success, ``False`` if a pivot is not positive or not finite.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 4.3], [EQ-INV-ABBENCH-005].

    """
    mul = operator.mul
    for i, row in enumerate(rows):
        lo = i - bandwidth if i > bandwidth else 0
        for j in range(lo, i):
            # lo >= max(0, j - bandwidth) for j <= i, so the overlap starts at lo.
            other = rows[j]
            shift = lo - (j - bandwidth if j > bandwidth else 0)
            width = j - lo
            acc = row[width]
            if width:
                acc -= sum(map(mul, row[:width], other[shift : shift + width]))
            row[width] = acc / other[-1]
        pivot = row[-1] - sum(map(mul, row[:-1], row[:-1]))
        if not (pivot > 0.0 and math.isfinite(pivot)):
            return False
        row[-1] = math.sqrt(pivot)
    return True


def _band_solve(
    rows: list[list[float]], bandwidth: int, rhs: list[float]
) -> list[float]:
    """Solve ``L L' x = rhs`` with the factor from :func:`_band_cholesky`.

    Args:
        rows: Banded Cholesky factor rows.
        bandwidth: Half-bandwidth.
        rhs: Right-hand side.

    Returns:
        The solution vector.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 4.3], [EQ-INV-ABBENCH-005].

    """
    mul = operator.mul
    size = len(rows)
    x = [0.0] * size
    for i in range(size):
        row = rows[i]
        lo = max(0, i - bandwidth)
        x[i] = (rhs[i] - sum(map(mul, row[:-1], x[lo:i]))) / row[-1]
    for i in range(size - 1, -1, -1):
        row = rows[i]
        lo = max(0, i - bandwidth)
        xi = x[i] / row[-1]
        x[i] = xi
        for offset in range(i - lo):
            x[lo + offset] -= row[offset] * xi
    return x


class AristoffBangerthForward:
    """The benchmark forward map ``theta -> z`` with exact derivatives.

    Solves the Q1 discretization of ``-div(a grad u) = 10`` on the unit square
    (``u = 0`` on the boundary) for the piecewise-constant ``a`` given by the 64
    values ``theta`` and returns the nodal interpolant at the ``13 x 13``
    observation points. Satisfies the ``DifferentiableForward`` protocol of
    :mod:`probstat.inverse.gauss_newton` (``input_dimension``,
    ``output_dimension``, ``value``, ``jvp``, ``vjp``).

    Args:
        fine_cells_per_cell: Fine mesh cells per coefficient cell per side; the
            benchmark uses ``4`` (``32 x 32`` mesh). Other values give the
            mesh hierarchy used for convergence studies.

    Raises:
        TypeError: If ``fine_cells_per_cell`` is not an integer.
        ValueError: If ``fine_cells_per_cell < 1``.

    Equation:
        [EQ-INV-ABBENCH-001], [EQ-INV-ABBENCH-002], [EQ-INV-ABBENCH-005].

    References:
        [REF-ARISTOFF-BANGERTH-2023], [REF-ELMAN-SILVESTER-WATHEN-2014].

    """

    __slots__ = (
        "_bandwidth",
        "_cache",
        "_cells",
        "_mesh",
        "_obs",
        "_unknowns",
        "factorizations",
    )

    _cache: tuple[Vector, list[list[float]], list[float]] | None

    def __init__(self, fine_cells_per_cell: int = PAPER_FINE_CELLS_PER_CELL) -> None:
        """Build the mesh topology and the observation operator.

        Args:
            fine_cells_per_cell: Fine cells per coefficient cell per side.

        Raises:
            TypeError: If ``fine_cells_per_cell`` is not an integer.
            ValueError: If ``fine_cells_per_cell < 1``.

        References:
            [REF-ARISTOFF-BANGERTH-2023], [EQ-INV-ABBENCH-002].

        """
        if isinstance(fine_cells_per_cell, bool) or not isinstance(
            fine_cells_per_cell, int
        ):
            raise TypeError("fine_cells_per_cell must be an integer")
        if fine_cells_per_cell < 1:
            raise ValueError("fine_cells_per_cell must be at least 1")
        refine = fine_cells_per_cell
        mesh = GRID_CELLS * refine
        side = mesh - 1
        self._mesh = mesh
        self._unknowns = side * side
        self._bandwidth = side + 1

        def unknown(i: int, j: int) -> int:
            """Return the unknown index of node ``(i, j)`` or ``-1`` (boundary).

            References:
                [EQ-INV-ABBENCH-002].

            """
            if 1 <= i <= side and 1 <= j <= side:
                return (j - 1) * side + (i - 1)
            return -1

        cells = []
        for cj in range(mesh):
            for ci in range(mesh):
                param = GRID_CELLS * (cj // refine) + ci // refine
                nodes = (
                    unknown(ci, cj),
                    unknown(ci + 1, cj),
                    unknown(ci + 1, cj + 1),
                    unknown(ci, cj + 1),
                )
                cells.append((param, nodes))
        self._cells = tuple(cells)
        obs = []
        for ix in range(1, MEASUREMENT_POINTS_PER_SIDE + 1):
            for jy in range(1, MEASUREMENT_POINTS_PER_SIDE + 1):
                gx = ix * mesh / (MEASUREMENT_POINTS_PER_SIDE + 1)
                gy = jy * mesh / (MEASUREMENT_POINTS_PER_SIDE + 1)
                ci = min(int(gx), mesh - 1)
                cj = min(int(gy), mesh - 1)
                sx = gx - ci
                sy = gy - cj
                weights = (
                    (1.0 - sx) * (1.0 - sy),
                    sx * (1.0 - sy),
                    sx * sy,
                    (1.0 - sx) * sy,
                )
                nodes = self._cells[cj * mesh + ci][1]
                obs.append(
                    tuple(
                        (node, w)
                        for node, w in zip(nodes, weights, strict=True)
                        if node >= 0 and w != 0.0
                    )
                )
        self._obs = tuple(obs)
        self._cache = None
        self.factorizations: int = 0

    @property
    def input_dimension(self) -> int:
        """Return the parameter dimension ``64``.

        References:
            [EQ-INV-ABBENCH-001].

        """
        return PARAMETER_DIMENSION

    @property
    def output_dimension(self) -> int:
        """Return the observation dimension ``169``.

        References:
            [EQ-INV-ABBENCH-002].

        """
        return DATA_DIMENSION

    @property
    def mesh_cells_per_side(self) -> int:
        """Return the number of fine mesh cells per side (32 for the benchmark).

        References:
            [EQ-INV-ABBENCH-002].

        """
        return self._mesh

    def _assemble_solve(
        self, theta: Vector
    ) -> tuple[list[list[float]], list[float]] | None:
        """Assemble ``A(theta)``, factorize it and solve ``A u = b``.

        Args:
            theta: Validated parameters.

        Returns:
            ``(factor_rows, u)`` or ``None`` if the factorization failed.

        References:
            [EQ-INV-ABBENCH-001], [EQ-INV-ABBENCH-005].

        """
        band = self._bandwidth
        rows = [[0.0] * (min(i, band) + 1) for i in range(self._unknowns)]
        for param, nodes in self._cells:
            coeff = theta[param]
            for a in range(4):
                ia = nodes[a]
                if ia < 0:
                    continue
                lo = max(0, ia - band)
                row = rows[ia]
                local = _LOCAL_STIFFNESS[a]
                for b in range(4):
                    ib = nodes[b]
                    if 0 <= ib <= ia:
                        row[ib - lo] += coeff * local[b]
        if not _band_cholesky(rows, band):
            return None
        self.factorizations += 1
        h = 1.0 / self._mesh
        load = [FORCING * h * h] * self._unknowns
        return rows, _band_solve(rows, band, load)

    def _state(self, theta: Vector) -> tuple[list[list[float]], list[float]] | None:
        """Return the cached factor and solution for ``theta``.

        Args:
            theta: Validated parameters.

        Returns:
            ``(factor_rows, u)`` or ``None`` on factorization failure.

        References:
            [EQ-INV-ABBENCH-005].

        """
        cache = self._cache
        if cache is not None and cache[0] == theta:
            return cache[1], cache[2]
        state = self._assemble_solve(theta)
        if state is None:
            self._cache = None
            return None
        self._cache = (theta, state[0], state[1])
        return state

    def _observe(self, u: Sequence[float]) -> Vector:
        """Apply the observation operator ``M`` to a nodal vector.

        Args:
            u: Nodal values of the interior unknowns.

        Returns:
            The 169 interpolated values.

        References:
            [EQ-INV-ABBENCH-002].

        """
        return tuple(math.fsum(w * u[node] for node, w in obs) for obs in self._obs)

    def _observe_transpose(self, w: Sequence[float]) -> list[float]:
        """Apply ``M'`` to a data-space vector.

        Args:
            w: Data-space vector of length 169.

        Returns:
            The unknown-space vector ``M' w``.

        References:
            [EQ-INV-ABBENCH-005].

        """
        out = [0.0] * self._unknowns
        for obs, wk in zip(self._obs, w, strict=True):
            for node, weight in obs:
                out[node] += weight * wk
        return out

    def value(self, theta: Sequence[float]) -> Vector:
        """Return the predicted measurements ``z(theta)``.

        Args:
            theta: 64 positive coefficients.

        Returns:
            169 predicted measurements (NaN entries if the Cholesky pivot of
            the stiffness matrix is not positive in floating point).

        Raises:
            TypeError: If ``theta`` is not a real sequence.
            ValueError: If ``theta`` has the wrong length or a non-positive or
                non-finite entry.

        Equation:
            [EQ-INV-ABBENCH-002].

        References:
            [REF-ARISTOFF-BANGERTH-2023], [REF-ELMAN-SILVESTER-WATHEN-2014].

        """
        point = _as_theta(theta)
        state = self._state(point)
        if state is None:
            return (math.nan,) * DATA_DIMENSION
        return self._observe(state[1])

    def _require_state(
        self, theta: Sequence[float]
    ) -> tuple[list[list[float]], list[float]]:
        """Return ``(factor, u)`` for ``theta`` or raise if the solve failed.

        Args:
            theta: Candidate parameters.

        Returns:
            Factor rows and the solution ``u``.

        Raises:
            ValueError: If the factorization failed.

        References:
            [EQ-INV-ABBENCH-005].

        """
        state = self._state(_as_theta(theta))
        if state is None:
            raise ValueError("stiffness matrix is not numerically positive definite")
        return state

    def _cell_products(self, u: Sequence[float]) -> list[tuple[float, ...]]:
        """Return ``K_loc u_loc`` for every fine cell (boundary values are 0).

        Args:
            u: Interior nodal solution.

        Returns:
            One 4-tuple per fine cell.

        References:
            [EQ-INV-ABBENCH-005].

        """
        out = []
        for _, nodes in self._cells:
            loc = [u[n] if n >= 0 else 0.0 for n in nodes]
            out.append(
                tuple(
                    sum(map(operator.mul, _LOCAL_STIFFNESS[a], loc)) for a in range(4)
                )
            )
        return out

    def jvp(self, theta: Sequence[float], v: Sequence[float]) -> Vector:
        """Return the Jacobian action ``J(theta) v``.

        ``J v = -M A^-1 (sum_k v_k A_k) u``, the exact derivative of the
        discrete forward map.

        Args:
            theta: 64 positive coefficients.
            v: Direction of length 64.

        Returns:
            The 169 directional derivatives.

        Raises:
            TypeError: If an argument is not a real sequence.
            ValueError: For wrong lengths, invalid ``theta``, a non-finite
                ``v`` or a failed factorization.

        Equation:
            [EQ-INV-ABBENCH-005].

        References:
            [REF-ARISTOFF-BANGERTH-2023], [REF-GOLUB-VANLOAN-2013].

        """
        factor, u = self._require_state(theta)
        direction = _as_real_vector(v, PARAMETER_DIMENSION, "v")
        products = self._cell_products(u)
        rhs = [0.0] * self._unknowns
        for (param, nodes), prod in zip(self._cells, products, strict=True):
            scale = direction[param]
            for a in range(4):
                if nodes[a] >= 0:
                    rhs[nodes[a]] += scale * prod[a]
        du = _band_solve(factor, self._bandwidth, rhs)
        return tuple(-value for value in self._observe(du))

    def vjp(self, theta: Sequence[float], w: Sequence[float]) -> Vector:
        """Return the exact discrete adjoint action ``J(theta)' w``.

        ``(J' w)_k = -p' A_k u`` with ``A p = M' w``.

        Args:
            theta: 64 positive coefficients.
            w: Data-space vector of length 169.

        Returns:
            The 64 components of ``J' w``.

        Raises:
            TypeError: If an argument is not a real sequence.
            ValueError: For wrong lengths, invalid ``theta``, a non-finite
                ``w`` or a failed factorization.

        Equation:
            [EQ-INV-ABBENCH-005].

        References:
            [REF-ARISTOFF-BANGERTH-2023], [REF-GOLUB-VANLOAN-2013].

        """
        factor, u = self._require_state(theta)
        weights = _as_real_vector(w, DATA_DIMENSION, "w")
        p = _band_solve(factor, self._bandwidth, self._observe_transpose(weights))
        products = self._cell_products(u)
        grad = [0.0] * PARAMETER_DIMENSION
        for (param, nodes), prod in zip(self._cells, products, strict=True):
            acc = 0.0
            for a in range(4):
                if nodes[a] >= 0:
                    acc += p[nodes[a]] * prod[a]
            grad[param] -= acc
        return tuple(grad)


@lru_cache(maxsize=1)
def _default_forward() -> AristoffBangerthForward:
    """Return the shared benchmark forward model (32 x 32 mesh).

    Returns:
        A module-level :class:`AristoffBangerthForward` instance.

    References:
        [EQ-INV-ABBENCH-002].

    """
    return AristoffBangerthForward()


def log_prior(theta: Sequence[float]) -> float:
    """Return the non-normalized log prior of paper Eq. (2.6).

    ``log pi_pr(theta) = -sum_i (ln theta_i)^2 / (2 sigma_pr^2)`` with
    ``sigma_pr = 2``, as a function of ``theta`` (no Jacobian factor).

    Args:
        theta: 64 positive coefficients.

    Returns:
        The log prior (zero at ``theta = 1``).

    Raises:
        TypeError: If ``theta`` is not a real sequence.
        ValueError: If ``theta`` has the wrong length or a non-positive or
            non-finite entry.

    Equation:
        [EQ-INV-ABBENCH-003].

    References:
        [REF-ARISTOFF-BANGERTH-2023].

    """
    point = _as_theta(theta)
    return -math.fsum(math.log(t) ** 2 for t in point) / (2.0 * PRIOR_LOG_STD**2)


def log_likelihood(
    theta: Sequence[float],
    *,
    data: Sequence[float] | None = None,
    forward: AristoffBangerthForward | None = None,
) -> float:
    """Return the non-normalized log likelihood of paper Eq. (2.5).

    ``log L = -|z_hat - z(theta)|^2 / (2 sigma^2)`` with ``sigma = 0.05``.

    Args:
        theta: 64 positive coefficients.
        data: Measurement vector (169 entries); defaults to :data:`TRUE_DATA`.
        forward: Forward model; defaults to the shared 32 x 32 benchmark model.

    Returns:
        The log likelihood, or ``-inf`` if the forward solve failed.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: For invalid ``theta`` or ``data``.

    Equation:
        [EQ-INV-ABBENCH-003].

    References:
        [REF-ARISTOFF-BANGERTH-2023].

    """
    model = _default_forward() if forward is None else forward
    z_hat = TRUE_DATA if data is None else _as_real_vector(data, DATA_DIMENSION, "data")
    predicted = model.value(theta)
    if not all(math.isfinite(v) for v in predicted):
        return -math.inf
    misfit = math.fsum((a - b) ** 2 for a, b in zip(z_hat, predicted, strict=True))
    return -misfit / (2.0 * NOISE_STD**2)


def log_posterior(
    theta: Sequence[float],
    *,
    data: Sequence[float] | None = None,
    forward: AristoffBangerthForward | None = None,
) -> float:
    """Return the non-normalized log posterior ``log L + log pi_pr`` (Eq. (2.1)).

    Args:
        theta: 64 positive coefficients.
        data: Measurement vector; defaults to :data:`TRUE_DATA`.
        forward: Forward model; defaults to the shared 32 x 32 benchmark model.

    Returns:
        ``log_likelihood(theta) + log_prior(theta)`` (``-inf`` on solver
        failure).

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: For invalid ``theta`` or ``data``.

    Equation:
        [EQ-INV-ABBENCH-004].

    References:
        [REF-ARISTOFF-BANGERTH-2023].

    """
    return log_likelihood(theta, data=data, forward=forward) + log_prior(theta)


def log_posterior_gradient(
    theta: Sequence[float],
    *,
    data: Sequence[float] | None = None,
    forward: AristoffBangerthForward | None = None,
) -> tuple[float, Vector]:
    """Return the log posterior and its gradient with respect to ``theta``.

    ``grad_k = [J' (z_hat - z)]_k / sigma^2 - ln(theta_k) / (sigma_pr^2 theta_k)``
    with the exact discrete adjoint ``J'``.

    Args:
        theta: 64 positive coefficients.
        data: Measurement vector; defaults to :data:`TRUE_DATA`.
        forward: Forward model; defaults to the shared 32 x 32 benchmark model.

    Returns:
        ``(log_posterior, gradient)``.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: For invalid ``theta`` or ``data`` or a failed solve.

    Equation:
        [EQ-INV-ABBENCH-004], [EQ-INV-ABBENCH-005].

    References:
        [REF-ARISTOFF-BANGERTH-2023].

    """
    model = _default_forward() if forward is None else forward
    point = _as_theta(theta)
    z_hat = TRUE_DATA if data is None else _as_real_vector(data, DATA_DIMENSION, "data")
    predicted = model.value(point)
    if not all(math.isfinite(v) for v in predicted):
        raise ValueError("forward solve failed (non-finite prediction)")
    residual = tuple(a - b for a, b in zip(z_hat, predicted, strict=True))
    value = -math.fsum(r * r for r in residual) / (2.0 * NOISE_STD**2)
    value += log_prior(point)
    adjoint = model.vjp(point, residual)
    gradient = tuple(
        g / NOISE_STD**2 - math.log(t) / (PRIOR_LOG_STD**2 * t)
        for g, t in zip(adjoint, point, strict=True)
    )
    return value, gradient


def theta_from_standard_normal(u: Sequence[float]) -> Vector:
    """Map standard-normal coordinates to ``theta`` for pCN-type samplers.

    ``theta_k = exp(sigma_pr^2 + sigma_pr u_k)``. Under ``u ~ N(0, I)`` the
    law of ``theta`` has density proportional to paper Eq. (2.6).

    Args:
        u: 64 finite reals.

    Returns:
        The 64 coefficients.

    Raises:
        TypeError: If ``u`` is not a real sequence.
        ValueError: If ``u`` has the wrong length or is not finite, or if the
            exponent overflows.

    Equation:
        [EQ-INV-ABBENCH-003].

    References:
        [REF-ARISTOFF-BANGERTH-2023], [REF-COTTER-ROBERTS-STUART-WHITE-2013].

    """
    vector = _as_real_vector(u, PARAMETER_DIMENSION, "u")
    exponent = tuple(PRIOR_LOG_MEAN + PRIOR_LOG_STD * v for v in vector)
    if any(abs(e) > 700.0 for e in exponent):
        raise ValueError("u maps outside the representable coefficient range")
    return tuple(math.exp(e) for e in exponent)


def standardized_log_likelihood(
    u: Sequence[float],
    *,
    data: Sequence[float] | None = None,
    forward: AristoffBangerthForward | None = None,
) -> float:
    """Return ``log L(theta(u))`` in standard-normal coordinates for pCN.

    Pass this to :func:`probstat.inference.preconditioned_crank_nicolson` to
    target the benchmark posterior: the prior (2.6) is exactly ``N(0, I)`` in
    ``u`` (see :func:`theta_from_standard_normal`). Out-of-range ``u`` gives
    ``-inf`` (a rejected proposal).

    Args:
        u: 64 finite reals.
        data: Measurement vector; defaults to :data:`TRUE_DATA`.
        forward: Forward model; defaults to the shared 32 x 32 benchmark model.

    Returns:
        The log likelihood at ``theta(u)`` or ``-inf``.

    Raises:
        TypeError: If an argument has the wrong type.
        ValueError: If ``u`` has the wrong length or is not finite.

    Equation:
        [EQ-INV-ABBENCH-003].

    References:
        [REF-ARISTOFF-BANGERTH-2023], [REF-COTTER-ROBERTS-STUART-WHITE-2013].

    """
    vector = _as_real_vector(u, PARAMETER_DIMENSION, "u")
    try:
        theta = theta_from_standard_normal(vector)
    except ValueError:
        return -math.inf
    return log_likelihood(theta, data=data, forward=forward)


@dataclass(frozen=True, slots=True)
class BenchmarkChain:
    """Result of :func:`metropolis_hastings`.

    Attributes:
        samples: Kept states ``theta`` (tuples of 64 floats), after thinning.
        log_posterior: Log posterior at each kept state.
        acceptance_rate: Accepted proposals divided by proposals (all steps).
        evaluations: Posterior evaluations performed.
        proposal_std: Log-space proposal standard deviation used.
        seed: Root seed of the random stream.

    References:
        [REF-ARISTOFF-BANGERTH-2023].

    """

    samples: tuple[Vector, ...]
    log_posterior: tuple[float, ...]
    acceptance_rate: float
    evaluations: int
    proposal_std: float
    seed: int


def metropolis_hastings(
    *,
    steps: int,
    thin: int = 1,
    initial: Sequence[float] | None = None,
    proposal_std: float = PROPOSAL_STD,
    seed: int | RandomStream | None = None,
    forward: AristoffBangerthForward | None = None,
) -> BenchmarkChain:
    """Run the paper's Metropolis-Hastings sampler (Appendix A.2).

    The proposal is ``theta~_k = theta_k exp(xi_k)``, ``xi_k ~ N(0, s^2)``, and
    the acceptance probability is
    ``min(1, pi(theta~) / pi(theta) * prod_k theta~_k / theta_k)``
    (the factor accounts for the non-symmetric log-normal proposal). The
    default start is ``theta = 1`` and the default ``s = 0.0725``, as in the
    paper (acceptance rate just under 24% there).

    Args:
        steps: Number of proposals.
        thin: Keep every ``thin``-th state (including the last).
        initial: Start; defaults to ``(1, ..., 1)``.
        proposal_std: Proposal standard deviation ``s > 0`` in log space.
        seed: Integer seed or stream; ``None`` is not reproducible.
        forward: Forward model; defaults to the shared 32 x 32 benchmark model.

    Returns:
        A :class:`BenchmarkChain`.

    Raises:
        TypeError: For non-integer ``steps``/``thin`` or invalid ``initial``.
        ValueError: For ``steps < 1``, ``thin < 1``, ``proposal_std <= 0`` or a
            start with non-finite log posterior.

    Equation:
        [EQ-INV-ABBENCH-004].

    References:
        [REF-ARISTOFF-BANGERTH-2023], [REF-METROPOLIS-ETAL-1953],
        [REF-HASTINGS-1970].

    """
    for name, count in (("steps", steps), ("thin", thin)):
        if isinstance(count, bool) or not isinstance(count, int):
            raise TypeError(f"{name} must be an integer")
        if count < 1:
            raise ValueError(f"{name} must be at least 1")
    if (
        isinstance(proposal_std, bool)
        or not isinstance(proposal_std, (int, float))
        or not (math.isfinite(proposal_std) and proposal_std > 0.0)
    ):
        raise ValueError("proposal_std must be positive and finite")
    model = _default_forward() if forward is None else forward
    stream = resolve_stream(seed)
    rng = stream.generator()
    current = _as_theta(
        (1.0,) * PARAMETER_DIMENSION if initial is None else initial, "initial"
    )
    current_lp = log_posterior(current, forward=model)
    if not math.isfinite(current_lp):
        raise ValueError("initial state has non-finite log posterior")
    samples = []
    trace = []
    accepted = 0
    evaluations = 1
    for step in range(1, steps + 1):
        shifts = [rng.gauss(0.0, proposal_std) for _ in range(PARAMETER_DIMENSION)]
        proposal = tuple(t * math.exp(s) for t, s in zip(current, shifts, strict=True))
        evaluations += 1
        proposal_lp = log_posterior(proposal, forward=model)
        log_alpha = proposal_lp - current_lp + math.fsum(shifts)
        if math.log(rng.random()) < log_alpha:
            current, current_lp = proposal, proposal_lp
            accepted += 1
        if step % thin == 0 or step == steps:
            samples.append(current)
            trace.append(current_lp)
    return BenchmarkChain(
        tuple(samples),
        tuple(trace),
        accepted / steps,
        evaluations,
        float(proposal_std),
        stream.seed,
    )
