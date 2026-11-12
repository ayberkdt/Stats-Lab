"""Eigenvalues of a small real non-symmetric matrix (Hessenberg QR).

The matrix is reduced to upper Hessenberg form by stabilized elementary
similarity transformations (Gaussian elimination with partial pivoting) and
its eigenvalues are found by the implicitly double-shifted Francis QR
iteration with deflation on negligible subdiagonal entries [EQ-TS-VAR-003].
The arrays are 1-based internally so the recursion reads like its published
statement. Used for the companion matrix of a vector autoregression, whose
size ``K p`` is small.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def _sign(magnitude: float, sign_source: float) -> float:
    """Return ``|magnitude|`` with the sign of ``sign_source``.

    Args:
        magnitude: Value whose absolute value is used.
        sign_source: Value whose sign is used (``+`` for zero).

    Returns:
        Signed magnitude.

    References:
        [REF-GOLUB-VANLOAN-2013, Section 7.5]. No equation is evaluated.

    """
    return abs(magnitude) if sign_source >= 0.0 else -abs(magnitude)


def _hessenberg(a: list[list[float]], n: int) -> None:
    """Reduce a 1-based square array to upper Hessenberg form in place.

    Args:
        a: 1-based ``(n + 1) x (n + 1)`` array (row and column 0 unused).
        n: Matrix size.

    Equation:
        [EQ-TS-VAR-003].

    References:
        [REF-GOLUB-VANLOAN-2013, Section 7.4].

    """
    for m in range(2, n):
        x = 0.0
        i = m
        for j in range(m, n + 1):
            if abs(a[j][m - 1]) > abs(x):
                x = a[j][m - 1]
                i = j
        if i != m:
            for j in range(m - 1, n + 1):
                a[i][j], a[m][j] = a[m][j], a[i][j]
            for j in range(1, n + 1):
                a[j][i], a[j][m] = a[j][m], a[j][i]
        if x != 0.0:
            for i in range(m + 1, n + 1):
                y = a[i][m - 1]
                if y != 0.0:
                    y /= x
                    a[i][m - 1] = y
                    for j in range(m, n + 1):
                        a[i][j] -= y * a[m][j]
                    for j in range(1, n + 1):
                        a[j][m] += y * a[j][i]
    for i in range(1, n + 1):
        for j in range(1, i - 1):
            a[i][j] = 0.0


def eigenvalues(matrix: Sequence[Sequence[float]]) -> tuple[complex, ...]:
    """Return the eigenvalues of a real square matrix.

    Args:
        matrix: Finite real ``n x n`` matrix.

    Returns:
        The ``n`` eigenvalues (complex pairs adjacent), sorted by decreasing
        modulus.

    Raises:
        ValueError: If the matrix is not square or the QR iteration does not
            converge in 30 iterations for some eigenvalue.

    Equation:
        [EQ-TS-VAR-003].

    References:
        [REF-GOLUB-VANLOAN-2013, Sections 7.4 and 7.5].

    """
    n = len(matrix)
    if any(len(row) != n for row in matrix):
        raise ValueError("matrix must be square")
    if n == 0:
        return ()
    a = [[0.0] * (n + 1)] + [[0.0] + [float(v) for v in row] for row in matrix]
    _hessenberg(a, n)
    wr = [0.0] * (n + 1)
    wi = [0.0] * (n + 1)
    anorm = 0.0
    for i in range(1, n + 1):
        for j in range(max(i - 1, 1), n + 1):
            anorm += abs(a[i][j])
    nn = n
    t = 0.0
    while nn >= 1:
        its = 0
        while True:
            low = 1
            for candidate in range(nn, 1, -1):
                s = abs(a[candidate - 1][candidate - 1]) + abs(a[candidate][candidate])
                if s == 0.0:
                    s = anorm
                if abs(a[candidate][candidate - 1]) + s == s:
                    a[candidate][candidate - 1] = 0.0
                    low = candidate
                    break
            x = a[nn][nn]
            if low == nn:
                wr[nn] = x + t
                wi[nn] = 0.0
                nn -= 1
            else:
                y = a[nn - 1][nn - 1]
                w = a[nn][nn - 1] * a[nn - 1][nn]
                if low == nn - 1:
                    p = 0.5 * (y - x)
                    q = p * p + w
                    z = math.sqrt(abs(q))
                    x += t
                    if q >= 0.0:
                        z = p + _sign(z, p)
                        wr[nn - 1] = wr[nn] = x + z
                        if z != 0.0:
                            wr[nn] = x - w / z
                        wi[nn - 1] = wi[nn] = 0.0
                    else:
                        wr[nn - 1] = wr[nn] = x + p
                        wi[nn] = z
                        wi[nn - 1] = -z
                    nn -= 2
                else:
                    if its == 30:
                        raise ValueError("QR iteration did not converge")
                    if its in (10, 20):
                        t += x
                        for i in range(1, nn + 1):
                            a[i][i] -= x
                        s = abs(a[nn][nn - 1]) + abs(a[nn - 1][nn - 2])
                        y = x = 0.75 * s
                        w = -0.4375 * s * s
                    its += 1
                    m = nn - 2
                    p = q = r = 0.0
                    z = 0.0
                    while m >= low:
                        z = a[m][m]
                        r = x - z
                        s = y - z
                        p = (r * s - w) / a[m + 1][m] + a[m][m + 1]
                        q = a[m + 1][m + 1] - z - r - s
                        r = a[m + 2][m + 1]
                        s = abs(p) + abs(q) + abs(r)
                        p /= s
                        q /= s
                        r /= s
                        if m == low:
                            break
                        u = abs(a[m][m - 1]) * (abs(q) + abs(r))
                        v = abs(p) * (
                            abs(a[m - 1][m - 1]) + abs(z) + abs(a[m + 1][m + 1])
                        )
                        if u + v == v:
                            break
                        m -= 1
                    for i in range(m + 2, nn + 1):
                        a[i][i - 2] = 0.0
                        if i != m + 2:
                            a[i][i - 3] = 0.0
                    for k in range(m, nn):
                        if k != m:
                            p = a[k][k - 1]
                            q = a[k + 1][k - 1]
                            r = 0.0
                            if k != nn - 1:
                                r = a[k + 2][k - 1]
                            x = abs(p) + abs(q) + abs(r)
                            if x != 0.0:
                                p /= x
                                q /= x
                                r /= x
                        s = _sign(math.sqrt(p * p + q * q + r * r), p)
                        if s != 0.0:
                            if k == m:
                                if low != m:
                                    a[k][k - 1] = -a[k][k - 1]
                            else:
                                a[k][k - 1] = -s * x
                            p += s
                            x = p / s
                            y = q / s
                            z = r / s
                            q /= p
                            r /= p
                            for j in range(k, nn + 1):
                                p = a[k][j] + q * a[k + 1][j]
                                if k != nn - 1:
                                    p += r * a[k + 2][j]
                                    a[k + 2][j] -= p * z
                                a[k + 1][j] -= p * y
                                a[k][j] -= p * x
                            mmin = nn if nn < k + 3 else k + 3
                            for i in range(low, mmin + 1):
                                p = x * a[i][k] + y * a[i][k + 1]
                                if k != nn - 1:
                                    p += z * a[i][k + 2]
                                    a[i][k + 2] -= p * r
                                a[i][k + 1] -= p * q
                                a[i][k] -= p
            if low >= nn - 1:
                break
    values = [complex(wr[i], wi[i]) for i in range(1, n + 1)]
    values.sort(key=lambda value: (-abs(value), -value.real, -value.imag))
    return tuple(values)
