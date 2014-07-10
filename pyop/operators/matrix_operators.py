import numpy as np
import scipy.sparse

from pyop import LinearOperator, ensure2dColumn

def zeros(shape):
    ''' PyOp version of zeros array function (only 2D).

    Parameters
    ----------
    shape : pair
        The shape of the LinearOperator (if it were a matrix).

    Returns
    -------
    LinearOperator
        A functional version of numpy.zeros()
    '''

    @ensure2dColumn
    def zeroInput(op_shape, x):
        return np.zeros((op_shape[0], x.shape[1]))

    return LinearOperator(shape, zeroInput, zeroInput)


def ones(shape):
    ''' PyOp version of ones array function (only 2D).

    Parameters
    ----------
    shape : pair
        The shape of the LinearOperator (if it were a matrix).

    Returns
    -------
    LinearOperator
        A functional version of numpy.ones()
    '''

    @ensure2dColumn
    def sumColumns(op_shape, x):
        column_sums = np.sum(x, axis = 0)
        return np.tile(column_sums, (op_shape[0], 1))

    return LinearOperator(shape, sumColumns, sumColumns)


def eye(shape):
    ''' PyOp version of eye array function (only 2D).

    Parameters
    ----------
    shape : pair
        The shape of the LinearOperator (if it were a matrix).

    Returns
    -------
    LinearOperator
        A functional version of numpy.eye()
    '''
    @ensure2dColumn
    def identity(op_shape, x):
        m, n = op_shape
        p, q = x.shape

        if m > n:
            return np.vstack([x, np.zeros((m - p, q))])
        elif m <= n:
            return x[:m]

    return LinearOperator(shape, identity, identity)
