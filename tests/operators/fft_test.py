import pytest
import numpy as np

import random

import pyop.operators as operators
import pyop

from functools import reduce
from operator import mul

num_tests = 100

array_max_size = 5
dimensions_max = 4


###############
#  FFT Tests  #
###############

def testFftInputErrors():

    with pytest.raises(ValueError):
        operators.fft((8, 8), (8,))

    with pytest.raises(ValueError):
        operators.fft((8, 8), (8, -8))

    with pytest.raises(ValueError):
        operators.fft((8, -8), (8, 8))


def testFftRandom():
    for _ in range(num_tests):
        d = random.randint(1, dimensions_max + 1)

        arr = np.random.rand(*tuple(
            random.randint(1, array_max_size + 1) for _ in range(d)))

        order = random.choice(('C', 'F'))

        s = tuple(random.randint(1, array_max_size*2) for _ in range(d))

        F = operators.fft(arr.shape, s = s, order = order)

        pyop.adjointTest(F)

        np.testing.assert_allclose(
            np.reshape(F._forward(np.ravel(arr, order)), s, order),
            np.fft.fftn(arr, s = s))


def testIfftInputErrors():

    with pytest.raises(ValueError):
        operators.ifft((8, 8), (8,))

    with pytest.raises(ValueError):
        operators.ifft((8, 8), (8, -8))

    with pytest.raises(ValueError):
        operators.ifft((8, -8), (8, 8))


def testIfftRandom():
    for _ in range(num_tests):
        d = random.randint(1, dimensions_max + 1)

        arr = np.random.rand(*tuple(
            random.randint(1, array_max_size + 1) for _ in range(d)))

        order = random.choice(('C', 'F'))

        s = tuple(random.randint(1, array_max_size*2) for _ in range(d))

        F = operators.ifft(arr.shape, s = s, order = order)

        pyop.adjointTest(F)

        np.testing.assert_allclose(
            np.reshape(F._forward(np.ravel(arr, order)), s, order),
            np.fft.ifftn(arr, s = s))


#####################
#  FFT Shift Tests  #
#####################

def randomAxes(ndim):
    return tuple(random.randint(0, ndim)
        for _ in range(random.randint(0, ndim + 2)))

def testFftshiftInputErrors():

    with pytest.raises(ValueError):
        operators.fftshift((8, -8), (1,))

    with pytest.raises(ValueError):
        operators.fftshift((8, 8), (-1,))

    with pytest.raises(ValueError):
        operators.fftshift((8, 8), (2, ))


def testFftshiftRandom():
    for _ in range(num_tests):
        d = random.randint(1, dimensions_max + 1)

        arr = np.random.rand(*tuple(
            random.randint(1, array_max_size + 1) for _ in range(d)))

        order = random.choice(('C', 'F'))

        axes = random.choice((None, randomAxes(arr.ndim - 1)))

        F = operators.fftshift(arr.shape, axes, order = order)

        pyop.adjointTest(F)

        np.testing.assert_allclose(
            np.reshape(F._forward(np.ravel(arr, order)), arr.shape, order),
            np.fft.fftshift(arr, axes))


def testIfftshiftInputErrors():

    with pytest.raises(ValueError):
        operators.ifftshift((8, -8), (1,))

    with pytest.raises(ValueError):
        operators.ifftshift((8, 8), (-1,))

    with pytest.raises(ValueError):
        operators.ifftshift((8, 8), (2, ))


def testIfftshiftRandom():
    for _ in range(num_tests):
        d = random.randint(1, dimensions_max + 1)

        arr = np.random.rand(*tuple(
            random.randint(1, array_max_size + 1) for _ in range(d)))

        order = random.choice(('C', 'F'))

        axes = random.choice((None, randomAxes(arr.ndim - 1)))

        F = operators.ifftshift(arr.shape, axes, order = order)

        pyop.adjointTest(F)

        np.testing.assert_allclose(
            np.reshape(F._forward(np.ravel(arr, order)), arr.shape, order),
            np.fft.ifftshift(arr, axes))


####################
#  FFT Wrap Tests  #
####################

def identityOp(shape):
    return pyop.LinearOperator((shape, shape), lambda x: x, lambda x: x)


def testFftwrapRandom():
    for _ in range(num_tests):
        d = random.randint(1, dimensions_max + 1)

        arr = np.random.rand(*tuple(
            random.randint(1, array_max_size + 1) for _ in range(d)))

        s = tuple(random.randint(1, array_max_size*2) for _ in range(d))

        shift = random.choice(("all", "none", randomAxes(arr.ndim - 1)))

        order = random.choice(('C', 'F'))

        I = identityOp(reduce(mul, s))
        J = operators.fftwrap(I, arr.shape, s, shift, order)

        pyop.adjointTest(J)


def testIfftwrapRandom():
    for _ in range(num_tests):
        d = random.randint(1, dimensions_max + 1)

        arr = np.random.rand(*tuple(
            random.randint(1, array_max_size + 1) for _ in range(d)))

        s = tuple(random.randint(1, array_max_size*2) for _ in range(d))

        shift = random.choice(("all", "none", randomAxes(arr.ndim - 1)))

        order = random.choice(('C', 'F'))

        I = identityOp(reduce(mul, s))
        J = operators.ifftwrap(I, arr.shape, s, shift, order)

        pyop.adjointTest(J)
