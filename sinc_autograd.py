"""This file contains a custom implementation of the sine cardinalis (sinc)
and its autograd derivatives.

The implementation of sinc itself differs from the numpy implementation.
While this version corresponds to the simple definition sin(x)/x, the
numpy implementation includes a prefactor such that np.sinc(x)=sinc(np.pi*x).
The main reason for this implementation, though, are the regularized
derivatives up to third order.
"""
from autograd.extend import defvjp, primitive
from pennylane import numpy as np


@primitive
def sinc(x):
    x = np.asanyarray(x)
    y = np.where(x == 0, 1.0e-20, x)
    return np.sin(y) / y


@primitive
def dsinc(x):
    x = np.asanyarray(x)
    y = np.where(x == 0, 1.0e-20, x)
    return (np.cos(y) * y - np.sin(y)) / (y ** 2)


@primitive
def d2sinc(x):
    x = np.asanyarray(x)
    return np.where(x == 0, -1 / 3, (np.sin(x) * (2 - x ** 2) - 2 * np.cos(x) * x) / (x ** 3))


def d3sinc(x):
    x = np.asanyarray(x)
    y = np.where(x == 0, 1.0e-20, x)
    return (np.sin(y) * (3 * y ** 2 - 6) + np.cos(y) * (6 * y - y ** 3)) / (y ** 4)


defvjp(sinc, lambda ans, x: lambda g: g * dsinc(x))
defvjp(dsinc, lambda ans, x: lambda g: g * d2sinc(x))
defvjp(d2sinc, lambda ans, x: lambda g: g * d3sinc(x))
