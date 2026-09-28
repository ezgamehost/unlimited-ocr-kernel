"""Reproduce the FP64 Chebyshev interpolation used by the native GELU kernel.

This is an offline preparation aid, not an inference dependency. The function
erf(sqrt(z))/sqrt(z) is analytic at zero, with limit 2/sqrt(pi).
"""
import math
import numpy as np
from numpy.polynomial import Chebyshev, Polynomial


def target(t):
    return np.array([
        math.erf(math.sqrt(8 * (float(x) + 1))) / math.sqrt(8 * (float(x) + 1))
        if x > -1 else 2 / math.sqrt(math.pi)
        for x in t
    ])


if __name__ == "__main__":
    polynomial = Chebyshev.interpolate(target, 24).convert(kind=Polynomial)
    grid = np.linspace(-1, 1, 200001)
    print("max sampled erf(u)/u error:", np.max(np.abs(polynomial(grid) - target(grid))))
    print("\n".join(f"    {float(value)!r}," for value in polynomial.coef))
