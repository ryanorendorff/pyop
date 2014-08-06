#pylint: disable=W0104,W0108
import pytest
import pyop

import random

import numpy as np

num_tests = 100
matrix_max_size = 100

def __testOperatorVersusMatrix(mat, op):

    assert mat.shape == op.shape

    rand_input = np.random.rand(*(mat.shape)).T

    np.testing.assert_allclose(op.dot(rand_input),
                               mat.dot(rand_input))

    np.testing.assert_allclose(op.T.dot(rand_input.T),
                               mat.T.dot(rand_input.T))


#######################################################################
#                                Tests                                #
#######################################################################

######################
#  To/From a matrix  #
######################

def testToLinearOperator():
    for _ in range(num_tests):
        A_mat = np.random.rand(random.randint(1, matrix_max_size),
                               random.randint(1, matrix_max_size))

        A_op = pyop.toLinearOperator(A_mat)

        __testOperatorVersusMatrix(A_mat, A_op)


def testToLinearOperatorInputCheck():
    vec = np.ones(1)
    twod = np.ones((1,1))
    threed = np.ones((1,1,1))
    fourd = np.ones((1,1,1,1))

    with pytest.raises(ValueError):
        pyop.toLinearOperator(vec)

    _ = pyop.toLinearOperator(twod)

    with pytest.raises(ValueError):
        pyop.toLinearOperator(threed)

    with pytest.raises(ValueError):
        pyop.toLinearOperator(fourd)


def testToMatrix():
    for _ in range(num_tests):
        shape = random.randint(1, matrix_max_size)
        A_mat = np.eye(shape)

        A_op = pyop.LinearOperator((shape, shape), lambda x:x, lambda x:x)
        np.testing.assert_allclose(A_mat, pyop.toMatrix(A_op))


################################
#  To another functional form  #
################################

def testToScipyLinearOperator():
    for _ in range(num_tests):
        shape = random.randint(1, matrix_max_size)

        ## Does not pass adjoint test, just for testing
        A_op = pyop.LinearOperator((shape, shape), lambda x:x, lambda x:2*x)
        A_sci = pyop.toScipyLinearOperator(A_op)

        input_mat = np.random.rand(shape, shape)

        np.testing.assert_allclose(A_op(input_mat), A_sci(input_mat))
        np.testing.assert_allclose(A_op.T(input_mat),
                                   A_sci.rmatvec(input_mat))
