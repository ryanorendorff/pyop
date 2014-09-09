#pylint: disable=W0104,W0108
import pyop.operators as operators
from pyop import adjointTest

import random

import numpy as np

from numpy import linspace, reshape, array, zeros, ravel
import scipy.signal as signal
from scipy.misc import derivative

num_tests = 25

#################
#  Convolution  #
#################

def test2DConvolution():

    A = reshape(linspace(1,9,9), (3,3), order = 'C')
    kernel = array([[-1, 1]])
    kernel_square = array([[0, 0, 0], [0, -1, 1], [0, 0, 0]])

    C = operators.convolve(kernel, (3, 3), order = 'F')
    adjointTest(C)

    np.testing.assert_allclose(
        reshape(C._forward(ravel(A, order = 'F')), (3,3), order = 'F'),
        signal.convolve(A, kernel, 'same'))

    np.testing.assert_allclose(
        reshape(C._forward(ravel(A, order = 'F')), (3,3), order = 'F'),
        signal.convolve(A, kernel_square, 'same'))

    np.testing.assert_allclose(
        reshape(C._adjoint(ravel(A, order = 'F')), (3,3), order = 'F'),
        signal.convolve(A, np.flipud(np.fliplr(kernel_square)), 'same'))


def test2DConvolutionSquare():

    A = reshape(linspace(1,9,9), (3,3), order = 'C')
    kernel = array([[0, -1, 0],
                    [-1, 0, 1],
                    [0,  1, 0]])

    C = operators.convolve(kernel, (3, 3), order = 'F')
    adjointTest(C)

    np.testing.assert_allclose(
        reshape(C._forward(ravel(A, order = 'F')), (3,3), order = 'F'),
        signal.convolve(A, kernel, 'same'))

    np.testing.assert_allclose(
        reshape(C._adjoint(ravel(A, order = 'F')), (3,3), order = 'F'),
        signal.convolve(A, np.flipud(np.fliplr(kernel)), 'same'))


def test3DConvolution():
    kernel = array([[[-1, 1], [1, 0]], [[1, 0], [0, 0]]])
    kernel_cube = array([
        zeros((3,3)),
        [[0, 0, 0], [0, -1, 1], [0, 1, 0]],
        [[0, 0, 0], [0,  1, 0], [0, 0, 0]]])

    A = reshape(linspace(1,9,9), (3,3), order = 'C')
    B = array((A, A, A))

    C = operators.convolve(kernel, (3, 3, 3), order = 'F')
    adjointTest(C)

    np.testing.assert_allclose(
        reshape(C._forward(ravel(B, order = 'F')), (3, 3, 3), order = 'F'),
        signal.convolve(B, kernel, 'same'))

    np.testing.assert_allclose(
        reshape(C._forward(ravel(B, order = 'F')), (3, 3, 3), order = 'F'),
        signal.convolve(B, kernel_cube, 'same'))

    np.testing.assert_allclose(
        reshape(C._adjoint(ravel(B, order = 'F')), (3, 3, 3), order = 'F'),
        signal.convolve(B,
            np.flipud(np.fliplr(kernel_cube[:,:,::-1])), 'same'))


def testConvolutionRandom():
    kernel_max_size = 5
    image_max_size = 5
    dimensions_max = 4

    for _ in range(num_testspass):
        d = random.randint(1, dimensions_max + 1)

        kernel = np.random.rand(*tuple(
            random.randint(1, kernel_max_size + 1) for _ in range(d)))

        image = np.random.rand(*tuple(
            random.randint(1, image_max_size + 1) for _ in range(d)))

        order = random.choice(('C', 'F', 'A'))

        C = operators.convolve(kernel, image.shape, order)
        adjointTest(C)

        np.testing.assert_allclose(
            reshape(C._forward(ravel(image, order)), image.shape, order),
            signal.convolve(image, kernel, 'same'))