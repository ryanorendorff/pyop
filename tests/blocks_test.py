#pylint: disable=W0104,W0108
import pytest
import pyop

import numpy as np
import random

from tools import operatorVersusMatrix

num_tests = 250
max_matrix_size = 10

############
#  hstack  #
############

def testHstackFunction():
    for _ in range(num_tests):
        rows = random.randint(1, max_matrix_size)
        randcols = lambda: random.randint(1, max_matrix_size)

        A_mat = np.random.rand(rows, randcols())
        A_op = pyop.toLinearOperator(A_mat)

        B_mat = np.random.rand(rows, randcols())
        B_op = pyop.toLinearOperator(B_mat)

        C_mat = np.random.rand(rows, randcols())
        C_op = pyop.toLinearOperator(C_mat)

        horz_blocks_mat = [A_mat, B_mat, C_mat]
        horz_blocks_op = [A_op, B_op, C_op]

        D_mat = np.hstack(horz_blocks_mat)
        D_op = pyop.hstack(horz_blocks_op)

        operatorVersusMatrix(D_mat, D_op)


############
#  vstack  #
############

def testVstackFunction():
    for _ in range(num_tests):
        randrows = lambda: random.randint(1, max_matrix_size)
        cols = random.randint(1, max_matrix_size)

        A_mat = np.random.rand(randrows(), cols)
        A_op = pyop.toLinearOperator(A_mat)

        B_mat = np.random.rand(randrows(), cols)
        B_op = pyop.toLinearOperator(B_mat)

        C_mat = np.random.rand(randrows(), cols)
        C_op = pyop.toLinearOperator(C_mat)

        vert_blocks_mat = [A_mat, B_mat, C_mat]
        vert_blocks_op = [A_op, B_op, C_op]

        D_mat = np.vstack(vert_blocks_mat)
        D_op = pyop.vstack(vert_blocks_op)

        operatorVersusMatrix(D_mat, D_op)


########
# bmat #
########

def testBmat():
    for _ in range(num_tests):
        randrows = lambda: random.randint(1, max_matrix_size)
        randcols = randrows

        row1 = randrows()
        row2 = randrows()

        col1 = randcols()
        col2 = randcols()

        A_mat = np.random.rand(row1, col1)
        A_op = pyop.toLinearOperator(A_mat)

        B_mat = np.random.rand(row1, col2)
        B_op = pyop.toLinearOperator(B_mat)

        C_mat = np.random.rand(row2, col1)
        C_op = pyop.toLinearOperator(C_mat)

        D_mat = np.random.rand(row2, col2)
        D_op = pyop.toLinearOperator(D_mat)

        blocks_mat = [[A_mat, B_mat], [C_mat, D_mat]]
        blocks_op = [[A_op, B_op], [C_op, D_op]]

        E_mat = np.vstack([np.hstack(blocks_mat[0]), np.hstack(blocks_mat[1])])
        E_op = pyop.bmat(blocks_op)

        operatorVersusMatrix(E_mat, E_op)


#######################
# Test block diagonal #
#######################

def testBlockDiag():
    for _ in range(num_tests):
        randrows = lambda: random.randint(1, max_matrix_size)
        randcols = randrows

        row1 = randrows()
        row2 = randrows()
        row3 = randrows()

        col1 = randcols()
        col2 = randcols()
        col3 = randcols()

        A_mat = np.random.rand(row1, col1)
        A_op = pyop.toLinearOperator(A_mat)

        B_mat = np.random.rand(row2, col2)
        B_op = pyop.toLinearOperator(B_mat)

        C_mat = np.random.rand(row3, col3)
        C_op = pyop.toLinearOperator(C_mat)

        zeros1 = np.zeros((row1, col2 + col3))
        zeros2 = np.zeros((row2, col1))
        zeros3 = np.zeros((row2, col3))
        zeros4 = np.zeros((row3, col1 + col2))

        blocks_mat = [[A_mat, zeros1], [zeros2, B_mat, zeros3],
                [zeros4, C_mat]]
        blocks_op = [A_op, B_op, C_op]

        E_mat = np.vstack([np.hstack(blocks_mat[0]), np.hstack(blocks_mat[1]),
            np.hstack(blocks_mat[2])])
        E_op = pyop.blockDiag(blocks_op)

        operatorVersusMatrix(E_mat, E_op)
        operatorVersusMatrix(E_mat.T, E_op.T)


#########################
# Test incorrect inputs #
#########################

def testInputs():
    blocks_op = []
    with pytest.raises(ValueError):
        A_op = pyop.bmat(blocks_op)

    with pytest.raises(ValueError):
        A_op = pyop.blockDiag(blocks_op)

    blocks_op = [[]]
    with pytest.raises(ValueError):
        A_op = pyop.bmat(blocks_op)

    randrows = lambda: random.randint(1, max_matrix_size)
    randcols = randrows

    row1 = randrows()
    row2 = randrows()

    col1 = randcols()
    col2 = randcols()

    A_mat = np.random.rand(row1, col1)
    A_op = pyop.toLinearOperator(A_mat)

    B_mat = np.random.rand(row1, col2)
    B_op = pyop.toLinearOperator(B_mat)

    C_mat = np.random.rand(row2, col1)
    C_op = pyop.toLinearOperator(C_mat)

    D_mat = np.random.rand(row2, col2)
    D_op = pyop.toLinearOperator(D_mat)

    blocks_op = [[A_op, B_op, A_op, B_op], []]
    with pytest.raises(ValueError):
        E_op = pyop.bmat(blocks_op)

    blocks_op = [[A_op, B_op], [D_op]]
    with pytest.raises(ValueError):
        E_op = pyop.bmat(blocks_op)

    blocks_op = [[A_op], [C_op, D_op]]
    with pytest.raises(ValueError):
        E_op = pyop.bmat(blocks_op)

    A_mat = np.random.rand(row1 + 1, col1)
    A_op = pyop.toLinearOperator(A_mat)

    blocks_op = [[A_op, B_op], [C_op, D_op]]
    with pytest.raises(ValueError):
        E_op = pyop.bmat(blocks_op)

    C_mat = np.random.rand(row2, col1 + 1)
    C_op = pyop.toLinearOperator(C_mat)

    blocks_op = [[A_op, B_op], [C_op, D_op]]
    with pytest.raises(ValueError):
        E_op = pyop.bmat(blocks_op)
