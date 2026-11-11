"""Normalizing flows: invertible layers over a standard-normal base.

A flow writes a random vector as ``x = T(z)`` with ``z ~ N(0, I)`` and an
invertible ``T = T_K o ... o T_1``; the change-of-variables formula gives the
exact density ``log p(x) = log N(T^{-1}(x)) + sum_k log |det J_{T_k^{-1}}|``
([EQ-GEN-FLOW-001]). Every layer maps the base side to the data side
(:meth:`forward`) and back (:meth:`inverse`) and returns the log-determinant
of the map it applies.

Layers
    :class:`PlanarFlow` and :class:`RadialFlow` (Rezende and Mohamed 2015)
    with the invertibility reparameterizations of their Appendix A;
    :class:`AffineCoupling` (Dinh, Sohl-Dickstein and Bengio 2017) with an
    :class:`~probstat.numerics.MLP` for the tanh-bounded log-scale and the
    shift; :class:`ActNorm`, an element-wise affine map with data-dependent
    initialization (Kingma and Dhariwal 2018); and the volume-preserving
    :class:`Permutation`.

Inverses
    Coupling, actnorm, permutation, and radial layers invert in closed form
    (the radial norm solves a quadratic). The planar inverse needs the root
    of a strictly increasing scalar equation; it is found in floating point
    and then one Newton step is evaluated on the (possibly taped) parameters,
    which carries the implicit-function derivative of the root, so planar
    layers can also be trained by maximum likelihood ([EQ-GEN-FLOW-003]).

Training
    :func:`fit_flow` maximizes the mean log-likelihood of mini-batches with
    :func:`probstat.optimization.adam_minimize`, differentiating the loss by
    :func:`probstat.numerics.reverse_value_and_gradient` ([EQ-GEN-FLOW-007]).

All arithmetic is scalar Python: the classes target low-dimensional density
estimation (a handful of coordinates, a few thousand points), not images.

References:
    [REF-PAPAMAKARIOS-ETAL-2021, Sections 2 and 3],
    [REF-REZENDE-MOHAMED-2015, Section 4.1 and Appendix A],
    [REF-DINH-SOHL-DICKSTEIN-BENGIO-2017, Section 3],
    [REF-KINGMA-DHARIWAL-2018, Section 3.1].

"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol

from ..foundations.random_streams import RandomStream
from ..numerics import MLP, admath, reverse_value_and_gradient
from ..optimization import adam_minimize
from ._common import (
    Point,
    as_point,
    as_points,
    mean_and_standard_error,
    positive_integer,
    positive_real,
    primal,
    resolve,
    standard_normal_log_density,
)

LayerOutput = tuple[list[Any], Any]


def _softplus(value: Any) -> Any:
    """Return ``log(1 + exp(value))`` without overflow, for any numeric kind.

    References:
        [EQ-GEN-FLOW-002], [REF-REZENDE-MOHAMED-2015, Appendix A].

    """
    if value > 0:
        return value + admath.log1p(admath.exp(-value))
    return admath.log1p(admath.exp(value))


def _dot(left: Sequence[Any], right: Sequence[Any]) -> Any:
    """Return the inner product of two vectors of any numeric kind.

    References:
        [EQ-GEN-FLOW-002]. Elementary linear algebra.

    """
    total: Any = 0.0
    for a, b in zip(left, right, strict=True):
        total = total + a * b
    return total


def _check_length(parameters: Sequence[Any], expected: int, owner: str) -> None:
    """Raise unless ``parameters`` has the expected length.

    Raises:
        ValueError: On a length mismatch.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].
        No equation is evaluated.

    """
    if len(parameters) != expected:
        raise ValueError(
            f"{owner} expects {expected} parameters, got {len(parameters)}"
        )


def _check_point(point: Sequence[Any], dimension: int, owner: str) -> None:
    """Raise unless ``point`` has the layer dimension.

    Raises:
        ValueError: On a length mismatch.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].
        No equation is evaluated.

    """
    if len(point) != dimension:
        raise ValueError(f"{owner} expects points of dimension {dimension}")


class FlowLayer(Protocol):
    """Structural interface of one invertible flow layer.

    ``forward`` maps the base side to the data side and returns
    ``log |det dx/dz|``; ``inverse`` maps back and returns
    ``log |det dz/dx|``. Both accept floats or reverse-mode tape variables.

    References:
        [EQ-GEN-FLOW-001], [REF-PAPAMAKARIOS-ETAL-2021, Section 2.1].

    """

    closed_form_inverse: ClassVar[bool]

    @property
    def dimension(self) -> int:
        """Return the dimension the layer acts on.

        References:
            Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

        """
        ...

    @property
    def parameter_count(self) -> int:
        """Return the number of trainable parameters.

        References:
            Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

        """
        ...

    def initial_parameters(self, stream: RandomStream) -> tuple[float, ...]:
        """Return reproducible starting parameters.

        References:
            Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

        """
        ...

    def forward(self, parameters: Sequence[Any], z: Sequence[Any]) -> LayerOutput:
        """Return ``(x, log |det dx/dz|)``.

        References:
            [EQ-GEN-FLOW-001].

        """
        ...

    def inverse(self, parameters: Sequence[Any], x: Sequence[Any]) -> LayerOutput:
        """Return ``(z, log |det dz/dx|)``.

        References:
            [EQ-GEN-FLOW-001].

        """
        ...


def _positive_dimension(value: object) -> int:
    """Validate a layer dimension.

    Raises:
        ValueError: If ``value`` is not a positive integer.

    References:
        Project validation policy [REF-PROBSTAT-VALIDATION].
        No equation is evaluated.

    """
    return positive_integer(value, name="dimension")


@dataclass(frozen=True)
class PlanarFlow:
    """Planar flow ``x = z + u_hat tanh(w'z + b)``.

    ``u_hat = u + (m(w'u) - w'u) w / |w|^2`` with ``m(a) = -1 + softplus(a)``
    enforces ``w'u_hat > -1``, the sufficient invertibility condition for
    ``tanh``; then ``log |det| = log(1 + (1 - tanh^2(w'z + b)) w'u_hat)``
    ([EQ-GEN-FLOW-002]). Parameters are ``(u, w, b)``. The inverse solves
    the scalar equation ``a + w'u_hat tanh(a + b) = w'x`` for ``a = w'z``
    ([EQ-GEN-FLOW-003]).

    Attributes:
        dimension: Number of coordinates.

    References:
        [EQ-GEN-FLOW-002], [EQ-GEN-FLOW-003],
        [REF-REZENDE-MOHAMED-2015, Equations 10-12 and Appendix A.1].

    """

    dimension: int
    closed_form_inverse: ClassVar[bool] = False

    def __post_init__(self) -> None:
        """Validate the dimension.

        Raises:
            ValueError: If the dimension is not a positive integer.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].
            No equation is evaluated.

        """
        _positive_dimension(self.dimension)

    @property
    def parameter_count(self) -> int:
        """Return ``2 d + 1``.

        References:
            [EQ-GEN-FLOW-002]. Counting only.

        """
        return 2 * self.dimension + 1

    def initial_parameters(self, stream: RandomStream) -> tuple[float, ...]:
        """Draw ``u ~ N(0, 0.1^2 I)``, ``w ~ N(0, I / d)``, and ``b = 0``.

        Args:
            stream: Random stream of this layer.

        Returns:
            Flat parameters ``(u, w, b)``.

        References:
            [EQ-GEN-FLOW-002], [REF-REZENDE-MOHAMED-2015, Section 4.1].

        """
        generator = stream.child("planar").generator()
        d = self.dimension
        u = [generator.gauss(0.0, 0.1) for _ in range(d)]
        w = [generator.gauss(0.0, 1.0 / math.sqrt(d)) for _ in range(d)]
        return (*u, *w, 0.0)

    def _constrained(
        self, parameters: Sequence[Any]
    ) -> tuple[list[Any], list[Any], Any, Any]:
        """Return ``(u_hat, w, b, w'u_hat)`` from the raw parameters.

        Raises:
            ValueError: If the parameter vector has the wrong length.

        References:
            [EQ-GEN-FLOW-002], [REF-REZENDE-MOHAMED-2015, Appendix A.1].

        """
        _check_length(parameters, self.parameter_count, "PlanarFlow")
        d = self.dimension
        u = list(parameters[:d])
        w = list(parameters[d : 2 * d])
        b = parameters[2 * d]
        norm_squared = _dot(w, w)
        if primal(norm_squared) == 0.0:
            return u, w, b, 0.0
        wu = _dot(w, u)
        projected = -1.0 + _softplus(wu)
        correction = (projected - wu) / norm_squared
        u_hat = [ui + correction * wi for ui, wi in zip(u, w, strict=True)]
        return u_hat, w, b, projected

    def forward(self, parameters: Sequence[Any], z: Sequence[Any]) -> LayerOutput:
        """Return ``x = z + u_hat tanh(w'z + b)`` and its log-determinant.

        Args:
            parameters: Flat ``(u, w, b)``.
            z: Base-side point.

        Returns:
            ``(x, log |det dx/dz|)``.

        Raises:
            ValueError: On a length mismatch.

        References:
            [EQ-GEN-FLOW-002], [REF-REZENDE-MOHAMED-2015, Equations 10-12].

        """
        _check_point(z, self.dimension, "PlanarFlow")
        u_hat, w, b, wu_hat = self._constrained(parameters)
        t = admath.tanh(_dot(w, z) + b)
        x = [zi + ui * t for zi, ui in zip(z, u_hat, strict=True)]
        return x, admath.log(1.0 + (1.0 - t * t) * wu_hat)

    def inverse(self, parameters: Sequence[Any], x: Sequence[Any]) -> LayerOutput:
        """Return ``z = T^{-1}(x)`` and ``log |det dz/dx|``.

        The root ``a*`` of ``g(a) = a + c tanh(a + b) - w'x`` with
        ``c = w'u_hat > -1`` is found in floating point by safeguarded
        Newton iteration; one Newton step from ``a*`` evaluated on the
        parameters then gives ``a`` with the derivative ``-g_theta / g_a``
        of the implicit function theorem ([EQ-GEN-FLOW-003]).

        Args:
            parameters: Flat ``(u, w, b)``.
            x: Data-side point.

        Returns:
            ``(z, log |det dz/dx|)``.

        Raises:
            ValueError: On a length mismatch.

        References:
            [EQ-GEN-FLOW-003], [REF-REZENDE-MOHAMED-2015, Equations 21-23],
            [REF-GRIEWANK-WALTHER-2008, Chapter 15].

        """
        _check_point(x, self.dimension, "PlanarFlow")
        u_hat, w, b, wu_hat = self._constrained(parameters)
        target = _dot(w, x)
        root = _planar_root(primal(wu_hat), primal(b), primal(target))
        t_root = admath.tanh(root + b)
        residual = root + wu_hat * t_root - target
        slope = 1.0 + wu_hat * (1.0 - t_root * t_root)
        a = root - residual / slope
        t = admath.tanh(a + b)
        z = [xi - ui * t for xi, ui in zip(x, u_hat, strict=True)]
        return z, -admath.log(1.0 + (1.0 - t * t) * wu_hat)


def _planar_root(c: float, b: float, target: float) -> float:
    """Solve ``a + c tanh(a + b) = target`` for ``c > -1`` in floating point.

    The left side is strictly increasing with slope in
    ``[min(1, 1 + c), max(1, 1 + c)]`` and the root lies within ``|c|`` of
    ``target``; Newton steps that leave the bracket are replaced by
    bisection.

    References:
        [EQ-GEN-FLOW-003], [REF-REZENDE-MOHAMED-2015, Appendix A.1].

    """
    low = target - abs(c) - 1e-12
    high = target + abs(c) + 1e-12
    a = target
    for _ in range(100):
        t = math.tanh(a + b)
        value = a + c * t - target
        if value > 0.0:
            high = a
        else:
            low = a
        slope = 1.0 + c * (1.0 - t * t)
        candidate = a - value / slope
        if not low < candidate < high:
            candidate = 0.5 * (low + high)
        if abs(candidate - a) <= 1e-15 * max(1.0, abs(a)):
            return candidate
        a = candidate
    return a


@dataclass(frozen=True)
class RadialFlow:
    """Radial flow ``x = z + beta (z - z0) / (alpha + |z - z0|)``.

    ``alpha = softplus(alpha_raw) > 0`` and
    ``beta = -alpha + softplus(beta_raw) > -alpha`` enforce invertibility
    (Rezende and Mohamed, Appendix A.2); with ``r = |z - z0|`` and
    ``h = 1 / (alpha + r)`` the log-determinant is
    ``(d - 1) log(1 + beta h) + log(1 + beta alpha h^2)``
    ([EQ-GEN-FLOW-004]). Parameters are ``(z0, alpha_raw, beta_raw)``. The
    inverse radius is the positive root of
    ``r^2 + (alpha + beta - rho) r - rho alpha = 0`` with ``rho = |x - z0|``.

    Attributes:
        dimension: Number of coordinates.

    References:
        [EQ-GEN-FLOW-004],
        [REF-REZENDE-MOHAMED-2015, Equation 14 and Appendix A.2].

    """

    dimension: int
    closed_form_inverse: ClassVar[bool] = True

    def __post_init__(self) -> None:
        """Validate the dimension.

        Raises:
            ValueError: If the dimension is not a positive integer.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].
            No equation is evaluated.

        """
        _positive_dimension(self.dimension)

    @property
    def parameter_count(self) -> int:
        """Return ``d + 2``.

        References:
            [EQ-GEN-FLOW-004]. Counting only.

        """
        return self.dimension + 2

    def initial_parameters(self, stream: RandomStream) -> tuple[float, ...]:
        """Draw ``z0 ~ N(0, I)``, ``alpha_raw = 0``, ``beta_raw ~ N(0, 0.1^2)``.

        ``beta_raw = 0`` would give ``beta = 0``, the identity map.

        Args:
            stream: Random stream of this layer.

        Returns:
            Flat parameters ``(z0, alpha_raw, beta_raw)``.

        References:
            [EQ-GEN-FLOW-004], [REF-REZENDE-MOHAMED-2015, Appendix A.2].

        """
        generator = stream.child("radial").generator()
        z0 = [generator.gauss(0.0, 1.0) for _ in range(self.dimension)]
        return (*z0, 0.0, generator.gauss(0.0, 0.1))

    def _constrained(self, parameters: Sequence[Any]) -> tuple[list[Any], Any, Any]:
        """Return ``(z0, alpha, beta)`` from the raw parameters.

        Raises:
            ValueError: If the parameter vector has the wrong length.

        References:
            [EQ-GEN-FLOW-004], [REF-REZENDE-MOHAMED-2015, Appendix A.2].

        """
        _check_length(parameters, self.parameter_count, "RadialFlow")
        d = self.dimension
        alpha = _softplus(parameters[d])
        beta = -alpha + _softplus(parameters[d + 1])
        return list(parameters[:d]), alpha, beta

    def _log_determinant(self, alpha: Any, beta: Any, radius: Any) -> Any:
        """Return ``log |det dx/dz|`` at base-side radius ``r``.

        References:
            [EQ-GEN-FLOW-004], [REF-REZENDE-MOHAMED-2015, Section 4.1].

        """
        h = 1.0 / (alpha + radius)
        total = admath.log(1.0 + beta * alpha * h * h)
        if self.dimension > 1:
            total = total + (self.dimension - 1) * admath.log(1.0 + beta * h)
        return total

    def forward(self, parameters: Sequence[Any], z: Sequence[Any]) -> LayerOutput:
        """Return the radial map of ``z`` and its log-determinant.

        Args:
            parameters: Flat ``(z0, alpha_raw, beta_raw)``.
            z: Base-side point.

        Returns:
            ``(x, log |det dx/dz|)``.

        Raises:
            ValueError: On a length mismatch.

        References:
            [EQ-GEN-FLOW-004], [REF-REZENDE-MOHAMED-2015, Equation 14].

        """
        _check_point(z, self.dimension, "RadialFlow")
        z0, alpha, beta = self._constrained(parameters)
        difference = [zi - ci for zi, ci in zip(z, z0, strict=True)]
        radius = admath.hypot(*difference)
        factor = beta / (alpha + radius)
        x = [zi + factor * di for zi, di in zip(z, difference, strict=True)]
        return x, self._log_determinant(alpha, beta, radius)

    def inverse(self, parameters: Sequence[Any], x: Sequence[Any]) -> LayerOutput:
        """Return the closed-form inverse and ``log |det dz/dx|``.

        Taking norms of ``x - z0 = (z - z0)(1 + beta / (alpha + r))`` gives
        ``rho = r (1 + beta / (alpha + r))``, a quadratic in ``r`` whose
        positive root is computed without cancellation.

        Args:
            parameters: Flat ``(z0, alpha_raw, beta_raw)``.
            x: Data-side point.

        Returns:
            ``(z, log |det dz/dx|)``.

        Raises:
            ValueError: On a length mismatch.

        References:
            [EQ-GEN-FLOW-004],
            [REF-REZENDE-MOHAMED-2015, Equations 24-26].

        """
        _check_point(x, self.dimension, "RadialFlow")
        z0, alpha, beta = self._constrained(parameters)
        difference = [xi - ci for xi, ci in zip(x, z0, strict=True)]
        rho = admath.hypot(*difference)
        linear = alpha + beta - rho
        root = admath.sqrt(linear * linear + 4.0 * rho * alpha)
        if linear > 0:
            radius = 2.0 * rho * alpha / (linear + root)
        else:
            radius = 0.5 * (root - linear)
        shrink = 1.0 / (1.0 + beta / (alpha + radius))
        z = [ci + shrink * di for ci, di in zip(z0, difference, strict=True)]
        return z, -self._log_determinant(alpha, beta, radius)


@dataclass(frozen=True)
class AffineCoupling:
    """Affine coupling layer with an MLP log-scale and shift.

    The first ``split`` coordinates pass through; the others are
    ``x_j = z_j exp(s_j) + t_j`` with ``(raw, t) = MLP(z_{1:split})`` and the
    tanh-bounded log-scale ``s = c tanh(raw / c)``, ``c = scale_bound``, so
    ``log |det| = sum_j s_j`` ([EQ-GEN-FLOW-005]). The network's output
    layer starts at zero, so a new layer is the identity.

    Attributes:
        dimension: Number of coordinates (at least two).
        hidden: Hidden-layer widths of the conditioner network.
        split: Number of conditioning coordinates (default ``d // 2``).
        scale_bound: Positive bound ``c`` of each log-scale.
        activation: Hidden activation of the network.

    References:
        [EQ-GEN-FLOW-005],
        [REF-DINH-SOHL-DICKSTEIN-BENGIO-2017, Equations 4-8],
        [REF-PAPAMAKARIOS-ETAL-2021, Section 3.1.2].

    """

    dimension: int
    hidden: tuple[int, ...] = (16,)
    split: int | None = None
    scale_bound: float = 2.0
    activation: str = "tanh"
    network: MLP = field(init=False, repr=False)
    closed_form_inverse: ClassVar[bool] = True

    def __post_init__(self) -> None:
        """Validate the settings and build the conditioner network.

        Raises:
            ValueError: For a dimension below two, a split outside
                ``[1, d - 1]``, a non-positive scale bound, or an invalid
                network architecture.

        References:
            [EQ-NN-MLP-001], [REF-DINH-SOHL-DICKSTEIN-BENGIO-2017, Section 3.2].

        """
        d = _positive_dimension(self.dimension)
        if d < 2:
            raise ValueError("AffineCoupling needs dimension >= 2")
        split = d // 2 if self.split is None else self.split
        if isinstance(split, bool) or not isinstance(split, int) or not 1 <= split < d:
            raise ValueError("split must be an integer in [1, dimension - 1]")
        bound = positive_real(self.scale_bound, name="scale_bound")
        hidden = tuple(self.hidden)
        network = MLP((split, *hidden, 2 * (d - split)), activation=self.activation)
        object.__setattr__(self, "split", split)
        object.__setattr__(self, "scale_bound", bound)
        object.__setattr__(self, "hidden", hidden)
        object.__setattr__(self, "network", network)

    @property
    def _split(self) -> int:
        """Return the resolved number of conditioning coordinates.

        References:
            [EQ-GEN-FLOW-005]. No equation is evaluated.

        """
        assert self.split is not None
        return self.split

    @property
    def parameter_count(self) -> int:
        """Return the conditioner network's parameter count.

        References:
            [EQ-NN-MLP-001]. Counting only.

        """
        return self.network.parameter_count

    def initial_parameters(self, stream: RandomStream) -> tuple[float, ...]:
        """Return Glorot-initialized hidden layers and a zero output layer.

        Args:
            stream: Random stream of this layer.

        Returns:
            Flat network parameters (the layer starts as the identity).

        References:
            [EQ-NN-GLOROT-001], [REF-GLOROT-BENGIO-2010, Equation 16].

        """
        return self.network.initial_parameters(
            stream.child("coupling"), output_scale=0.0
        )

    def _scale_and_shift(
        self, parameters: Sequence[Any], condition: Sequence[Any]
    ) -> tuple[list[Any], list[Any]]:
        """Return the bounded log-scales and shifts for the conditioning part.

        Raises:
            ValueError: If the parameter vector has the wrong length.

        References:
            [EQ-GEN-FLOW-005], [EQ-NN-MLP-001].

        """
        _check_length(parameters, self.parameter_count, "AffineCoupling")
        outputs = self.network.forward(parameters, condition)
        width = self.dimension - self._split
        bound = self.scale_bound
        scales = [bound * admath.tanh(raw / bound) for raw in outputs[:width]]
        return scales, list(outputs[width:])

    def forward(self, parameters: Sequence[Any], z: Sequence[Any]) -> LayerOutput:
        """Return ``x`` with ``x_j = z_j exp(s_j) + t_j`` and ``sum_j s_j``.

        Args:
            parameters: Conditioner network parameters.
            z: Base-side point.

        Returns:
            ``(x, log |det dx/dz|)``.

        Raises:
            ValueError: On a length mismatch.

        References:
            [EQ-GEN-FLOW-005],
            [REF-DINH-SOHL-DICKSTEIN-BENGIO-2017, Equations 4-6].

        """
        _check_point(z, self.dimension, "AffineCoupling")
        k = self._split
        condition = list(z[:k])
        scales, shifts = self._scale_and_shift(parameters, condition)
        transformed = [
            zj * admath.exp(s) + t
            for zj, s, t in zip(z[k:], scales, shifts, strict=True)
        ]
        total: Any = 0.0
        for s in scales:
            total = total + s
        return condition + transformed, total

    def inverse(self, parameters: Sequence[Any], x: Sequence[Any]) -> LayerOutput:
        """Return ``z`` with ``z_j = (x_j - t_j) exp(-s_j)`` and ``-sum_j s_j``.

        Args:
            parameters: Conditioner network parameters.
            x: Data-side point.

        Returns:
            ``(z, log |det dz/dx|)``.

        Raises:
            ValueError: On a length mismatch.

        References:
            [EQ-GEN-FLOW-005],
            [REF-DINH-SOHL-DICKSTEIN-BENGIO-2017, Equations 7-8].

        """
        _check_point(x, self.dimension, "AffineCoupling")
        k = self._split
        condition = list(x[:k])
        scales, shifts = self._scale_and_shift(parameters, condition)
        restored = [
            (xj - t) * admath.exp(-s)
            for xj, s, t in zip(x[k:], scales, shifts, strict=True)
        ]
        total: Any = 0.0
        for s in scales:
            total = total - s
        return condition + restored, total


@dataclass(frozen=True)
class ActNorm:
    """Element-wise affine layer ``x = b + exp(l) * z`` (activation normalization).

    Parameters are the shifts ``b`` followed by the log-scales ``l``;
    ``log |det| = sum_i l_i`` ([EQ-GEN-FLOW-006]). The default parameters
    are the identity; :meth:`NormalizingFlow.data_initialization` sets them
    so that the data reaching the layer leave it with zero mean and unit
    variance per coordinate.

    Attributes:
        dimension: Number of coordinates.

    References:
        [EQ-GEN-FLOW-006], [REF-KINGMA-DHARIWAL-2018, Section 3.1 and Table 1].

    """

    dimension: int
    closed_form_inverse: ClassVar[bool] = True

    def __post_init__(self) -> None:
        """Validate the dimension.

        Raises:
            ValueError: If the dimension is not a positive integer.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].
            No equation is evaluated.

        """
        _positive_dimension(self.dimension)

    @property
    def parameter_count(self) -> int:
        """Return ``2 d``.

        References:
            [EQ-GEN-FLOW-006]. Counting only.

        """
        return 2 * self.dimension

    def initial_parameters(self, stream: RandomStream) -> tuple[float, ...]:
        """Return the identity parameters (zero shifts and log-scales).

        Args:
            stream: Unused; accepted for the common layer interface.

        Returns:
            ``2 d`` zeros.

        References:
            [EQ-GEN-FLOW-006]. No equation is evaluated.

        """
        del stream
        return (0.0,) * self.parameter_count

    def forward(self, parameters: Sequence[Any], z: Sequence[Any]) -> LayerOutput:
        """Return ``b + exp(l) * z`` and ``sum_i l_i``.

        Raises:
            ValueError: On a length mismatch.

        References:
            [EQ-GEN-FLOW-006], [REF-KINGMA-DHARIWAL-2018, Table 1].

        """
        _check_point(z, self.dimension, "ActNorm")
        _check_length(parameters, self.parameter_count, "ActNorm")
        d = self.dimension
        x = [parameters[i] + admath.exp(parameters[d + i]) * z[i] for i in range(d)]
        total: Any = 0.0
        for i in range(d):
            total = total + parameters[d + i]
        return x, total

    def inverse(self, parameters: Sequence[Any], x: Sequence[Any]) -> LayerOutput:
        """Return ``(x - b) * exp(-l)`` and ``-sum_i l_i``.

        Raises:
            ValueError: On a length mismatch.

        References:
            [EQ-GEN-FLOW-006], [REF-KINGMA-DHARIWAL-2018, Table 1].

        """
        _check_point(x, self.dimension, "ActNorm")
        _check_length(parameters, self.parameter_count, "ActNorm")
        d = self.dimension
        z = [(x[i] - parameters[i]) * admath.exp(-parameters[d + i]) for i in range(d)]
        total: Any = 0.0
        for i in range(d):
            total = total - parameters[d + i]
        return z, total


@dataclass(frozen=True)
class Permutation:
    """Fixed coordinate permutation ``x_i = z_{order[i]}`` (volume preserving).

    Attributes:
        order: A permutation of ``0, ..., d - 1``.

    References:
        [EQ-GEN-FLOW-001], [REF-DINH-SOHL-DICKSTEIN-BENGIO-2017, Section 3.5].

    """

    order: tuple[int, ...]
    closed_form_inverse: ClassVar[bool] = True

    def __post_init__(self) -> None:
        """Validate the permutation.

        Raises:
            ValueError: If ``order`` is not a permutation of ``range(d)``.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].
            No equation is evaluated.

        """
        order = tuple(self.order)
        if not order or sorted(order) != list(range(len(order))):
            raise ValueError("order must be a permutation of 0, ..., d - 1")
        object.__setattr__(self, "order", order)

    @classmethod
    def reverse(cls, dimension: int) -> Permutation:
        """Return the coordinate reversal of ``dimension`` coordinates.

        Raises:
            ValueError: If ``dimension`` is not a positive integer.

        References:
            [REF-DINH-SOHL-DICKSTEIN-BENGIO-2017, Section 3.5].
            No equation is evaluated.

        """
        d = _positive_dimension(dimension)
        return cls(tuple(range(d - 1, -1, -1)))

    @property
    def dimension(self) -> int:
        """Return the number of coordinates.

        References:
            Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

        """
        return len(self.order)

    @property
    def parameter_count(self) -> int:
        """Return zero (no trainable parameters).

        References:
            [EQ-GEN-FLOW-001]. Counting only.

        """
        return 0

    def initial_parameters(self, stream: RandomStream) -> tuple[float, ...]:
        """Return the empty parameter vector.

        References:
            Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

        """
        del stream
        return ()

    def forward(self, parameters: Sequence[Any], z: Sequence[Any]) -> LayerOutput:
        """Return the permuted point and log-determinant ``0``.

        Raises:
            ValueError: On a length mismatch.

        References:
            [EQ-GEN-FLOW-001]. A permutation has ``|det| = 1``.

        """
        _check_point(z, self.dimension, "Permutation")
        _check_length(parameters, 0, "Permutation")
        return [z[index] for index in self.order], 0.0

    def inverse(self, parameters: Sequence[Any], x: Sequence[Any]) -> LayerOutput:
        """Return the un-permuted point and log-determinant ``0``.

        Raises:
            ValueError: On a length mismatch.

        References:
            [EQ-GEN-FLOW-001]. A permutation has ``|det| = 1``.

        """
        _check_point(x, self.dimension, "Permutation")
        _check_length(parameters, 0, "Permutation")
        z: list[Any] = [0.0] * self.dimension
        for position, index in enumerate(self.order):
            z[index] = x[position]
        return z, 0.0


@dataclass(frozen=True)
class NormalizingFlow:
    """Composition ``x = T_K(... T_1(z))`` of flow layers over ``z ~ N(0, I)``.

    Parameters live in one flat vector, layer by layer, so the flow can be
    differentiated by reverse mode and trained by Adam.

    Attributes:
        layers: Non-empty sequence of layers of one dimension, applied in
            order from the base side to the data side.

    References:
        [EQ-GEN-FLOW-001], [REF-PAPAMAKARIOS-ETAL-2021, Equations 2-6 and 25-28].

    """

    layers: tuple[FlowLayer, ...]

    def __post_init__(self) -> None:
        """Validate the layers.

        Raises:
            ValueError: If there is no layer or the dimensions differ.

        References:
            Project validation policy [REF-PROBSTAT-VALIDATION].
            No equation is evaluated.

        """
        layers = tuple(self.layers)
        if not layers:
            raise ValueError("a flow needs at least one layer")
        if len({layer.dimension for layer in layers}) != 1:
            raise ValueError("all layers must share one dimension")
        object.__setattr__(self, "layers", layers)

    @property
    def dimension(self) -> int:
        """Return the dimension of the base and data spaces.

        References:
            Project API policy [REF-PROBSTAT-API]. No equation is evaluated.

        """
        return self.layers[0].dimension

    @property
    def parameter_count(self) -> int:
        """Return the total number of parameters.

        References:
            [EQ-GEN-FLOW-001]. Counting only.

        """
        return sum(layer.parameter_count for layer in self.layers)

    @property
    def closed_form_inverse(self) -> bool:
        """Return whether every layer inverts in closed form.

        References:
            [EQ-GEN-FLOW-003]. No equation is evaluated.

        """
        return all(layer.closed_form_inverse for layer in self.layers)

    def initial_parameters(self, seed: int | RandomStream = 0) -> tuple[float, ...]:
        """Return reproducible starting parameters of every layer.

        Args:
            seed: Integer seed or :class:`RandomStream`.

        Returns:
            Flat parameter vector of length :attr:`parameter_count`.

        References:
            [REF-STEELE-LEA-FLOOD-2014]. No equation is evaluated.

        """
        stream = resolve(seed)
        parameters: list[float] = []
        for index, layer in enumerate(self.layers):
            parameters.extend(layer.initial_parameters(stream.child("layer", index)))
        return tuple(parameters)

    def _slices(self, parameters: Sequence[Any]) -> list[Sequence[Any]]:
        """Split the flat vector into per-layer parameter vectors.

        Raises:
            ValueError: If the vector has the wrong length.

        References:
            [EQ-GEN-FLOW-001]. No equation is evaluated.

        """
        _check_length(parameters, self.parameter_count, "NormalizingFlow")
        slices: list[Sequence[Any]] = []
        offset = 0
        for layer in self.layers:
            slices.append(parameters[offset : offset + layer.parameter_count])
            offset += layer.parameter_count
        return slices

    def forward(self, parameters: Sequence[Any], z: Sequence[Any]) -> LayerOutput:
        """Map a base point to the data side.

        Args:
            parameters: Flat parameter vector.
            z: Base-side point.

        Returns:
            ``(x, log |det dx/dz|)``, the log-determinants summed over layers.

        Raises:
            ValueError: On a length mismatch.

        References:
            [EQ-GEN-FLOW-001], [REF-PAPAMAKARIOS-ETAL-2021, Equations 26 and 28].

        """
        _check_point(z, self.dimension, "NormalizingFlow")
        point: list[Any] = list(z)
        total: Any = 0.0
        for layer, values in zip(self.layers, self._slices(parameters), strict=True):
            point, log_det = layer.forward(values, point)
            total = total + log_det
        return point, total

    def inverse(self, parameters: Sequence[Any], x: Sequence[Any]) -> LayerOutput:
        """Map a data point to the base side.

        Args:
            parameters: Flat parameter vector.
            x: Data-side point.

        Returns:
            ``(z, log |det dz/dx|)``.

        Raises:
            ValueError: On a length mismatch.

        References:
            [EQ-GEN-FLOW-001], [REF-PAPAMAKARIOS-ETAL-2021, Equations 5 and 27].

        """
        _check_point(x, self.dimension, "NormalizingFlow")
        slices = self._slices(parameters)
        point: list[Any] = list(x)
        total: Any = 0.0
        for layer, values in zip(reversed(self.layers), reversed(slices), strict=True):
            point, log_det = layer.inverse(values, point)
            total = total + log_det
        return point, total

    def log_density(self, parameters: Sequence[Any], x: Sequence[Any]) -> Any:
        """Return the exact log-density ``log N(T^{-1}(x)) + log |det J_{T^{-1}}|``.

        Args:
            parameters: Flat parameter vector (floats or tape variables).
            x: Data-side point.

        Returns:
            The log-density, of the numeric kind of the inputs.

        Raises:
            ValueError: On a length mismatch.

        References:
            [EQ-GEN-FLOW-001], [REF-PAPAMAKARIOS-ETAL-2021, Equation 3].

        """
        z, log_det = self.inverse(parameters, x)
        return standard_normal_log_density(z) + log_det

    def sample_with_log_density(
        self, parameters: Sequence[float], n: int, seed: int | RandomStream = 0
    ) -> tuple[tuple[Point, ...], tuple[float, ...]]:
        """Draw ``n`` points and their log-densities from the forward map.

        The log-density of ``x = T(z)`` is ``log N(z) - log |det J_T(z)|``,
        so no inverse is needed.

        Args:
            parameters: Flat float parameter vector.
            n: Positive number of draws.
            seed: Integer seed or :class:`RandomStream`.

        Returns:
            ``(points, log densities)``.

        Raises:
            ValueError: For a non-positive ``n`` or a length mismatch.

        References:
            [EQ-GEN-FLOW-001], [REF-PAPAMAKARIOS-ETAL-2021, Equation 2].

        """
        count = positive_integer(n, name="n")
        values = [float(value) for value in parameters]
        generator = resolve(seed).child("flow-sample").generator()
        points: list[Point] = []
        log_densities: list[float] = []
        for _ in range(count):
            z = [generator.gauss(0.0, 1.0) for _ in range(self.dimension)]
            x, log_det = self.forward(values, z)
            points.append(tuple(float(value) for value in x))
            log_densities.append(float(standard_normal_log_density(z)) - float(log_det))
        return tuple(points), tuple(log_densities)

    def sample(
        self, parameters: Sequence[float], n: int, seed: int | RandomStream = 0
    ) -> tuple[Point, ...]:
        """Draw ``n`` points ``x = T(z)``, ``z ~ N(0, I)``.

        Args:
            parameters: Flat float parameter vector.
            n: Positive number of draws.
            seed: Integer seed or :class:`RandomStream`.

        Returns:
            The draws.

        Raises:
            ValueError: For a non-positive ``n`` or a length mismatch.

        References:
            [EQ-GEN-FLOW-001], [REF-PAPAMAKARIOS-ETAL-2021, Equation 1].

        """
        return self.sample_with_log_density(parameters, n, seed)[0]

    def data_initialization(
        self, parameters: Sequence[float], data: object
    ) -> tuple[float, ...]:
        """Return parameters whose actnorm layers standardize the data.

        The data are pulled back through the layers from the data side; each
        :class:`ActNorm` met on the way gets the per-coordinate mean and
        standard deviation of the points reaching it as shift and scale, so
        those points leave it with zero mean and unit variance.

        Args:
            parameters: Flat float parameter vector to start from.
            data: Points of the flow's dimension (at least two).

        Returns:
            Updated flat parameter vector.

        Raises:
            ValueError: For invalid data or a coordinate with zero spread at
                an actnorm layer.

        References:
            [EQ-GEN-FLOW-006], [REF-KINGMA-DHARIWAL-2018, Section 3.1].

        """
        points = [list(p) for p in as_points(data, minimum=2)]
        if len(points[0]) != self.dimension:
            raise ValueError(f"data must have dimension {self.dimension}")
        slices = [list(map(float, values)) for values in self._slices(parameters)]
        for index in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[index]
            if isinstance(layer, ActNorm):
                d = layer.dimension
                n = len(points)
                for i in range(d):
                    column = [p[i] for p in points]
                    mean = math.fsum(column) / n
                    spread = math.sqrt(math.fsum((v - mean) ** 2 for v in column) / n)
                    if spread <= 0.0:
                        raise ValueError("a coordinate has zero spread at an ActNorm")
                    slices[index][i] = mean
                    slices[index][d + i] = math.log(spread)
            points = [
                [primal(v) for v in layer.inverse(slices[index], p)[0]] for p in points
            ]
        return tuple(value for values in slices for value in values)


def real_nvp_flow(
    dimension: int,
    *,
    couplings: int = 4,
    hidden: tuple[int, ...] = (16,),
    scale_bound: float = 2.0,
) -> NormalizingFlow:
    """Return a Real NVP style flow: couplings, reversals, and a final actnorm.

    Layers from the base side: ``couplings`` affine coupling layers with a
    coordinate reversal between consecutive ones, then an :class:`ActNorm`
    on the data side (data-initialized by :func:`fit_flow`).

    Args:
        dimension: Data dimension (at least two).
        couplings: Positive number of coupling layers.
        hidden: Hidden widths of each conditioner network.
        scale_bound: Bound of each log-scale.

    Returns:
        The flow.

    Raises:
        ValueError: For an invalid setting.

    References:
        [REF-DINH-SOHL-DICKSTEIN-BENGIO-2017, Section 3.5],
        [REF-KINGMA-DHARIWAL-2018, Section 3.1].

    """
    count = positive_integer(couplings, name="couplings")
    layers: list[FlowLayer] = []
    for index in range(count):
        if index:
            layers.append(Permutation.reverse(dimension))
        layers.append(AffineCoupling(dimension, hidden=hidden, scale_bound=scale_bound))
    layers.append(ActNorm(dimension))
    return NormalizingFlow(tuple(layers))


@dataclass(frozen=True)
class FlowFit:
    """Maximum-likelihood fit of a :class:`NormalizingFlow`.

    Args:
        flow: The fitted flow.
        parameters: Fitted flat parameters.
        losses: Mini-batch negative mean log-likelihood at every iteration.
        train_mean_log_likelihood: Mean log-density of the training data.
        train_standard_error: Its standard error ``s / sqrt(n)``.
        heldout_mean_log_likelihood: Mean log-density of held-out data, or
            ``None``.
        heldout_standard_error: Its standard error, or ``None``.
        iterations: Adam iterations.
        batch_size: Mini-batch size.

    References:
        [EQ-GEN-FLOW-007], [REF-PAPAMAKARIOS-ETAL-2021, Section 2.3.1].

    """

    flow: NormalizingFlow
    parameters: tuple[float, ...]
    losses: tuple[float, ...]
    train_mean_log_likelihood: float
    train_standard_error: float
    heldout_mean_log_likelihood: float | None
    heldout_standard_error: float | None
    iterations: int
    batch_size: int

    def log_density(self, x: object) -> float:
        """Return the fitted log-density at one point.

        Raises:
            ValueError: If ``x`` is not a finite point of the flow dimension.

        References:
            [EQ-GEN-FLOW-001].

        """
        point = as_point(x, self.flow.dimension)
        return float(self.flow.log_density(self.parameters, point))

    def sample(self, n: int, seed: int | RandomStream = 0) -> tuple[Point, ...]:
        """Draw ``n`` points from the fitted flow.

        Raises:
            ValueError: If ``n`` is not a positive integer.

        References:
            [EQ-GEN-FLOW-001].

        """
        return self.flow.sample(self.parameters, n, seed)


def _mean_log_likelihood(
    flow: NormalizingFlow, parameters: Sequence[float], points: Sequence[Point]
) -> tuple[float, float]:
    """Return the mean log-density of ``points`` and its standard error.

    References:
        [EQ-GEN-DIAG-002], [EQ-GEN-FLOW-001].

    """
    values = [float(flow.log_density(parameters, point)) for point in points]
    return mean_and_standard_error(values)


def fit_flow(
    data: object,
    flow: NormalizingFlow | None = None,
    *,
    iterations: int = 400,
    batch_size: int = 64,
    learning_rate: float = 0.01,
    heldout: object | None = None,
    initial_parameters: Sequence[float] | None = None,
    data_initialization: bool = True,
    tail_fraction: float = 0.0,
    seed: int | RandomStream = 0,
) -> FlowFit:
    """Fit a normalizing flow by maximum likelihood with Adam mini-batches.

    Each iteration draws ``batch_size`` training points without replacement
    and takes an Adam step on ``-mean log p(x)`` ([EQ-GEN-FLOW-007]),
    differentiated by reverse mode. The default flow is
    :func:`real_nvp_flow` for ``d >= 2`` and four planar layers followed by
    an actnorm for ``d = 1``.

    Args:
        data: Training points (at least two).
        flow: Flow to fit; ``None`` selects the default.
        iterations: Positive number of Adam steps.
        batch_size: Positive mini-batch size (capped at the data size).
        learning_rate: Adam step size.
        heldout: Optional held-out points for an out-of-sample likelihood.
        initial_parameters: Optional starting parameters.
        data_initialization: Initialize the actnorm layers from the data
            (:meth:`NormalizingFlow.data_initialization`).
        tail_fraction: Fraction of final iterates to average, in ``[0, 1)``.
        seed: Integer seed or :class:`RandomStream`.

    Returns:
        :class:`FlowFit` with the loss trace and the mean log-likelihoods.

    Raises:
        ValueError: For invalid data or settings, or a non-finite loss.

    References:
        [EQ-GEN-FLOW-007], [EQ-OPT-ADAM-002],
        [REF-PAPAMAKARIOS-ETAL-2021, Equations 13-15].

    """
    points = as_points(data, minimum=2)
    dimension = len(points[0])
    if flow is None:
        if dimension >= 2:
            flow = real_nvp_flow(dimension)
        else:
            planar: list[FlowLayer] = [PlanarFlow(1) for _ in range(4)]
            flow = NormalizingFlow((*planar, ActNorm(1)))
    if flow.dimension != dimension:
        raise ValueError(f"data dimension {dimension} does not match the flow")
    held = None if heldout is None else as_points(heldout, name="heldout")
    if held is not None and len(held[0]) != dimension:
        raise ValueError("heldout points must match the data dimension")
    steps = positive_integer(iterations, name="iterations")
    size = min(positive_integer(batch_size, name="batch_size"), len(points))
    stream = resolve(seed)
    if initial_parameters is None:
        start = flow.initial_parameters(stream.child("initial"))
    else:
        start = tuple(float(value) for value in initial_parameters)
        _check_length(start, flow.parameter_count, "fit_flow")
    if data_initialization:
        start = flow.data_initialization(start, points)
    fitted_flow = flow

    def objective(
        parameters: tuple[float, ...], batch_stream: RandomStream
    ) -> tuple[float, tuple[float, ...]]:
        """Return the mini-batch negative mean log-likelihood and its gradient.

        References:
            [EQ-GEN-FLOW-007], [REF-PAPAMAKARIOS-ETAL-2021, Equation 15].

        """
        generator = batch_stream.child("batch").generator()
        batch = [points[i] for i in generator.sample(range(len(points)), size)]

        def loss(theta: Sequence[Any]) -> Any:
            """Return ``-mean log p(x)`` over the batch.

            References:
                [EQ-GEN-FLOW-007].

            """
            total: Any = 0.0
            for point in batch:
                total = total + fitted_flow.log_density(theta, point)
            return -total / len(batch)

        return reverse_value_and_gradient(loss, parameters)

    result = adam_minimize(
        objective,
        start,
        iterations=steps,
        learning_rate=learning_rate,
        tail_fraction=tail_fraction,
        seed=stream.child("adam"),
    )
    train_mean, train_error = _mean_log_likelihood(flow, result.x, points)
    held_mean: float | None = None
    held_error: float | None = None
    if held is not None:
        held_mean, held_error = _mean_log_likelihood(flow, result.x, held)
    return FlowFit(
        flow=flow,
        parameters=result.x,
        losses=result.losses,
        train_mean_log_likelihood=train_mean,
        train_standard_error=train_error,
        heldout_mean_log_likelihood=held_mean,
        heldout_standard_error=held_error,
        iterations=steps,
        batch_size=size,
    )
