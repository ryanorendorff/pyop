import numpy as np
import scipy.sparse

from functools import partial

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

    def zeroInput(x, op_shape):
        return np.zeros((op_shape, x.shape[1]))

    return LinearOperator(shape,
            ensure2dColumn(partial(zeroInput, op_shape = shape[0])),
            ensure2dColumn(partial(zeroInput, op_shape = shape[1])))


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

    def sumColumns(x, op_shape):
        column_sums = np.sum(x, axis = 0)
        return np.tile(column_sums, (op_shape, 1))

    return LinearOperator(shape,
            ensure2dColumn(partial(sumColumns, op_shape = shape[0])),
            ensure2dColumn(partial(sumColumns, op_shape = shape[1])))



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
    def identity(x, op_shape):
        m, n = op_shape
        p, q = x.shape

        if m > n:
            return np.vstack([x, np.zeros((m - p, q))])
        elif m <= n:
            return x[:m]

    return LinearOperator(shape,
            ensure2dColumn(partial(identity, op_shape = shape)),
            ensure2dColumn(partial(identity, op_shape = shape[::-1])))


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
    def subset(x):
        return x[perm]

    @ensure2dColumn
    def expand(x):
        ret_shape = (rows, x.shape[1])

        ret = np.zeros(ret_shape)
        np.add.at(ret, perm, x)

        return ret

    return LinearOperator((len(perm), rows), subset, expand)


def diag(v):
    ''' Create a LinearOperator that emulates a diagonal matrix.

    Creates a LinearOperator that scales each row of its input by the
    corresponding element of the input vector v. The length of the vector v
    defines the shape of the operator (n by n).

    Parameters
    ----------
    v : 1-D array
        An array by which to scale each of rows of the input.

    Returns
    -------
    LinearOperator
        A LinearOperator that scales np.array inputs.
    '''

    def forwardAdjoint(x):
        return v[:, np.newaxis] * x

    return LinearOperator( (len(v), len(v)),
            forwardAdjoint, forwardAdjoint)

