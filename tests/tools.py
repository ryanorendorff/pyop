import numpy as np

def testOperatorVersusMatrix(mat, op):

    assert mat.shape == op.shape

    rand_input = np.random.rand(*(mat.shape)).T

    np.testing.assert_allclose(op.dot(rand_input),
                               mat.dot(rand_input))

    np.testing.assert_allclose(op.T.dot(rand_input.T),
                               mat.T.dot(rand_input.T))
