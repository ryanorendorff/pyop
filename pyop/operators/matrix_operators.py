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


def select(rows, perm):
    ''' Select only certain rows of a matrix.

    This operator selects only certain rows from a matrix. It can be
    particularly useful for consesus or sharing optimization problems.

    The input perm can be in any order, and duplicates can be made. In this
    manner, it is possible to make a permutation matrix by including an
    input that includes each row once and only once.

    Parameters
    ----------
    rows : integer
        The number of total rows selecting from.
    perm : list
        A list of the rows to take. This will define the shape of the
        resulting LinearOperator.

    Returns
    -------
    LinearOperator
        A LinearOperator that performs the selection on np.array inputs.
    '''

    @ensure2dColumn
    def subset(_, x):
        return x[perm]

    @ensure2dColumn
    def expand(op_shape, x):
        ret_shape = (op_shape[0], x.shape[1])

        ret = np.zeros(ret_shape)
        np.add.at(ret, perm, x)

        return ret

    return LinearOperator((len(perm), rows), subset, expand)
