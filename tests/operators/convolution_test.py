#pylint: disable=W0104,W0108
import pyop.operators as operators
from pyop import adjointTest

import pytest
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

    for _ in range(num_tests):
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

##############
#  Gradient  #
##############

def test1stOrderGradient():
    values = np.array([x**2 for x in range(-5, 5)])
    # uses scipy.misc.derivative to test accuracy of central diff
    true_difference = [derivative(lambda x: x**2, x0=i, n=1, order=3)
                       for i in range(-4, 4)]
    CD = operators.gradient(1, 3, values.shape)

    # test only non-edge values due to limitations of the central difference
    np.testing.assert_allclose(CD(values)[1:-1], true_difference)

def test1stOrderGradientNonUnitStep():
    values = np.array([x**2 for x in linspace(-5, 4, 19)])
    true_difference = [derivative(lambda x: x**2, x0=i, n=1, order=3, dx=0.5)
                       for i in linspace(-4.5, 3.5, 17)]
    CD = operators.gradient(1, 3, values.shape, step=(0.5, ))

    # test only non-edge values due to limitations of the central difference
    np.testing.assert_allclose(CD(values)[1:-1], true_difference)

def test2D1stOrderGradient():
    values = np.sqrt(range(0, 10)).reshape(1, -1).repeat(10, 0)
    values = values * values.T

    kernel = np.array([[  0 , -0.5,  0 ],
                       [-0.5,   0 , 0.5],
                       [  0 ,  0.5,  0 ]])

    true_difference = np.array([(kernel * values[i:i+3, j:j+3]).sum()
        for i in range(8)
        for j in range(8)]).reshape(8, 8)

    CD = operators.gradient(1, 3, values.shape)

    # test only non-edge values due to limitations of the central difference
    np.testing.assert_allclose(CD(values.ravel()).reshape(10, 10)[1:-1, 1:-1],
                               true_difference)

def test3D1stOrderGradientRandom():
    values = np.random.random((10,10,10))

    kernel = np.array([[[0,   0 , 0],
                        [0, -0.5, 0],
                        [0,   0 , 0]],

                       [[  0 , -0.5,  0 ],
                        [-0.5,   0 , 0.5],
                        [  0 ,  0.5,  0 ]],

                       [[0,  0 , 0],
                        [0, 0.5, 0],
                        [0,  0 , 0]]])

    true_difference = np.array([(kernel * values[i:i+3, j:j+3, k:k+3]).sum()
        for i in range(8)
        for j in range(8)
        for k in range(8)]).reshape(8, 8, 8)

    CD = operators.gradient(1, 3, values.shape)

    # test only non-edge values due to limitations of the central difference
    np.testing.assert_allclose(CD(values.ravel()).reshape(
        10, 10, 10)[1:-1, 1:-1, 1:-1], true_difference)

def test3D1stOrderGradientRandomMultipleStep():
    values = np.random.random((10,10,10))

    kernel = np.array([[[0,    0  , 0],
                        [0,  -1/3., 0],
                        [0,    0  , 0]],

                       [[  0, -0.25, 0],
                        [ -1,     0  , 1],
                        [  0,  0.25, 0]],

                       [[0,   0 , 0],
                        [0, 1/3., 0],
                        [0,   0 , 0]]])

    true_difference = np.array([(kernel * values[i:i+3, j:j+3,k:k+3]).sum()
        for i in range(8)
        for j in range(8)
        for k in range(8)]).reshape(8, 8, 8)

    CD = operators.gradient(1, 3, values.shape, step=(1.5, 2, 0.5))

    # test only non-edge values due to limitations of the central difference
    np.testing.assert_allclose(CD(values.ravel()).reshape(
        10, 10, 10)[1:-1, 1:-1, 1:-1], true_difference)

def test2ndOrderGradient():
    values = np.sqrt(range(0, 10))
    true_difference = [derivative(np.sqrt, x0=i, n=2, order=3)
                       for i in range(1, 9)]
    CD = operators.gradient(2, 3, values.shape)

    # test only non-edge values due to limitations of the central difference
    np.testing.assert_allclose(CD(values)[1:-1], true_difference)

def test2ndOrderGradientNonUnitStep():
    values = np.sqrt(linspace(0, 10, 21))
    true_difference = [derivative(np.sqrt, x0=i, n=2, order=3, dx=0.5)
                       for i in linspace(0.5, 9.5, 19)]
    CD = operators.gradient(2, 3, values.shape, step=(0.5, ))

    # test only non-edge values due to limitations of the central difference
    np.testing.assert_allclose(CD(values)[1:-1], true_difference)

def test2ndOrderGradient5Points():
    values = np.sqrt(range(0, 15))
    true_difference = [derivative(np.sqrt, x0=i, n=2, order=5)
                       for i in range(2, 13)]
    CD = operators.gradient(2, 5, values.shape)

    # test only non-edge values due to limitations of the central difference
    # we remove 2 from either side this time due to the 5 points used
    np.testing.assert_allclose(CD(values)[2:-2], true_difference)

def test3rdOrderGradient7Points():
    values = np.sqrt(range(0, 15))
    true_difference = [derivative(np.sqrt, x0=i, n=3, order=7)
                       for i in range(3, 12)]
    CD = operators.gradient(3, 7, values.shape)

    # test only non-edge values due to limitations of the central difference
    # we remove 3 from either side this time due to the 7 points used
    np.testing.assert_allclose(CD(values)[3:-3], true_difference)

def testMismatchStepShapeLength():
    with pytest.raises(ValueError) as e:
        operators.gradient(1, 7, shape=(10, 10, 10), step=(1, 2))
    assert e.value.args[0] == "Shape and step must have same ndims (length)."

def testNotEnoughPointsForDifference():
    with pytest.raises(ValueError) as e:
        operators.gradient(1, 7, (10, 6, 10))
    assert e.value.args[0] == ("Shape's dims must have at least as many "
                               "points as central difference weights.")

def testCentralDiffErrors():
    with pytest.raises(ValueError) as e:
        operators.gradient(1, 4, (10, ))
    assert e.value.args[0] == "The number of points must be odd."

    with pytest.raises(ValueError) as e:
        operators.gradient(5, 5, (10, ))
    assert e.value.args[0] == ("Number of points must be at least "
                               "the derivative order + 1.")
