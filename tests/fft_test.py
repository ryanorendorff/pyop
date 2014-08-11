import pytest
import numpy as np

import random

import pyop.operators as operators
import pyop

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
