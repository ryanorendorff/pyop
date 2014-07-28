#pylint: disable=W0104,W0108
import pytest
import pyop
import pyop.operators as operators

import random

import numpy as np

num_tests = 1000
matrix_max_size = 100

def __testOperatorVersusMatrix(mat, op):

    assert mat.shape == op.shape

    rand_input = np.random.rand(*(mat.shape)).T

    np.testing.assert_allclose(op.dot(rand_input),
                               mat.dot(rand_input))

    np.testing.assert_allclose(op.T.dot(rand_input.T),
                               mat.T.dot(rand_input.T))


###########
#  Zeros  #
###########

def testZerosFunction():
    for _ in range(num_tests):
        shape = (random.randint(1, matrix_max_size),
                 random.randint(1, matrix_max_size))

        Z_mat = np.zeros(shape)
        Z_op = operators.zeros(shape)

        __testOperatorVersusMatrix(Z_mat, Z_op)


def testZerosAdjoint():
    for _ in range(num_tests):
        shape = (random.randint(1, matrix_max_size),
                 random.randint(1, matrix_max_size))

        Z_op = operators.zeros(shape)

        pyop.adjointTest(Z_op)


##########
#  Ones  #
##########

def testOnesFunction():
    for _ in range(num_tests):
        shape = (random.randint(1, matrix_max_size),
                 random.randint(1, matrix_max_size))

        O_mat = np.ones(shape)
        O_op = operators.ones(shape)

        __testOperatorVersusMatrix(O_mat, O_op)


def testOnesAdjoint():
    for _ in range(num_tests):
        shape = (random.randint(1, matrix_max_size),
                 random.randint(1, matrix_max_size))

        O_op = operators.ones(shape)

        pyop.adjointTest(O_op)


#########
#  Eye  #
#########

def testEyeFunction():
    for _ in range(num_tests):
        shape = (random.randint(1, matrix_max_size),
                 random.randint(1, matrix_max_size))

        I_mat = np.ones(shape)
        I_op = operators.ones(shape)

        __testOperatorVersusMatrix(I_mat, I_op)


def testEyeAdjoint():
    for _ in range(num_tests):
        shape = (random.randint(1, matrix_max_size),
                 random.randint(1, matrix_max_size))

        I_op = operators.ones(shape)

        pyop.adjointTest(I_op)


############
#  Select  #
############

def testSelectAdjoint():
    for _ in range(num_tests):
        rows = random.randint(1, 100)
        perm = [random.randint(0, rows - 1)
                for _ in range(random.randint(1, rows))]

        S_op = operators.select(rows, perm)

        pyop.adjointTest(S_op)


##########
#  Diag  #
##########
def testDiagFunction():
    for _ in range(num_tests):
        rand_vec = np.random.rand(random.randint(1, 100))

        D_mat = np.diag(rand_vec)
        D_op  = operators.diag(rand_vec)

        __testOperatorVersusMatrix(D_mat, D_op)


def testDiagVector():
    ## Tests a vector input, which fails without ensure2dColumn
    D = pyop.operators.diag(np.array([1, 2, 1, 1]))

    np.testing.assert_allclose(D(np.array([2, 2, 2, 2])),
                               np.array([2, 4, 2, 2]))


def testDiagAdjoint():
    for _ in range(num_tests):
        rand_vec = np.random.rand(random.randint(1, 100))

        D_op = operators.diag(rand_vec)

        pyop.adjointTest(D_op)
