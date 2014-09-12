"""
The operators specified in this module act like their matrix equivalent,
and attempt to match tbe basic building blocks in NumPy and other numerical
packages.
"""
import numpy as np

from functools import partial

from pyop import LinearOperator, matmat

from scipy.misc import doccer


docdict = {
    'shape' :
"""shape : pair
    The shape of the LinearOperator (if it were a matrix).""",

    ## The see also section.
    'zeros' : "zeros : Matrix free version of the zeros matrix.",
    'ones' : "ones : Matrix free version of the ones matrix.",
    'eye' : "eye : Matrix free version of the eye matrix.",
    'diag' : "diag : Convert a 1D array to matrix free diagonal matrix.",
    'select' : "select : Select certain rows out of a matrix."
    }

docfill = doccer.filldoc(docdict)

@docfill
def zeros(shape):
    ''' PyOp version of zeros array function (only 2D).

    Returns a new LinearOperator that emulates a matrix filled with zeros.

    Parameters
    ----------
    %(shape)s

    Returns
    -------
    LinearOperator
        A functional version of numpy.zeros()

    See Also
    --------
    %(ones)s
    %(eye)s
    %(diag)s
    %(select)s

    Examples
    --------
    >>> from pyop.operators import zeros
    >>> from pyop import toMatrix
    >>> toMatrix(zeros((2, 1)))
    array([[ 0.],
           [ 0.]])
    >>> s = (2,2)
    >>> toMatrix(zeros(s))
    array([[ 0.,  0.],
           [ 0.,  0.]])
    '''

    def zeroInput(x, op_shape):
        return np.zeros((op_shape, x.shape[1]))

    return LinearOperator(shape,
            matmat(partial(zeroInput, op_shape = shape[0])),
            matmat(partial(zeroInput, op_shape = shape[1])))


@docfill
def ones(shape):
    ''' PyOp version of ones array function (only 2D).

    Returns a new LinearOperator that emulates a matrix filled with ones.

    Parameters
    ----------
    %(shape)s

    Returns
    -------
    LinearOperator
        A functional version of numpy.ones()

    See Also
    --------
    %(zeros)s
    %(eye)s
    %(diag)s
    %(select)s

    Examples
    --------
    >>> from pyop.operators import ones
    >>> from pyop import toMatrix
    >>> toMatrix(ones((2, 1)))
    array([[ 1.],
           [ 1.]])
    >>> s = (2,2)
    >>> toMatrix(ones(s))
    array([[ 1.,  1.],
           [ 1.,  1.]])
    '''

    def sumColumns(x, op_shape):
        column_sums = np.sum(x, axis = 0)
        return np.tile(column_sums, (op_shape, 1))

    return LinearOperator(shape,
            matmat(partial(sumColumns, op_shape = shape[0])),
            matmat(partial(sumColumns, op_shape = shape[1])))



@docfill
def eye(shape):
    ''' PyOp version of eye array function (only 2D).

    Returns a new LinearOperator that emulates the identity matrix.

    Parameters
    ----------
    %(shape)s

    Returns
    -------
    LinearOperator
        A functional version of numpy.eye()

    See Also
    --------
    %(zeros)s
    %(ones)s
    %(diag)s
    %(select)s

    Examples
    --------
    >>> from pyop.operators import eye
    >>> from pyop import toMatrix
    >>> toMatrix(eye((2, 1)))
    array([[ 1.],
           [ 0.]])
    >>> s = (2,2)
    >>> toMatrix(eye(s))
    array([[ 1.,  0.],
           [ 0.,  1.]])
    '''
    def identity(x, op_shape):
        m, n = op_shape
        p, q = x.shape

        if m > n:
            return np.vstack([x, np.zeros((m - p, q))])
        elif m <= n:
            return x[:m]

    return LinearOperator(shape,
            matmat(partial(identity, op_shape = shape)),
            matmat(partial(identity, op_shape = shape[::-1])))


@docfill
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
        The number of total rows to be selected from.
    perm : list
        A list of the rows to take. This will define the shape of the
        resulting LinearOperator.

    Returns
    -------
    LinearOperator
        A LinearOperator that performs the selection on np.array inputs.

    See Also
    --------
    %(zeros)s
    %(ones)s
    %(eye)s
    %(diag)s

    Examples
    --------
    >>> from pyop.operators import select
    >>> import numpy as np
    >>> S = select(4, [0, 1, 3])
    >>> S(np.array([1, 2, 3, 4]))
    array([1, 2, 4])
    >>> S = select(4, [0, 1, 1])
    >>> S(np.array([1, 2, 3, 4]))
    array([1, 2, 2])
    '''

    @matmat
    def subset(x):
        return x[perm]

    @matmat
    def expand(x):
        ret_shape = (rows, x.shape[1])

        ret = np.zeros(ret_shape)
        np.add.at(ret, perm, x)

        return ret

    return LinearOperator((len(perm), rows), subset, expand)


@docfill
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

    See Also
    --------
    %(zeros)s
    %(ones)s
    %(eye)s
    %(select)s

    Examples
    --------
    >>> from pyop.operators import diag
    >>> from pyop import toMatrix
    >>> import numpy as np
    >>> toMatrix(diag(np.array([1, 2, 3, 4])))
    array([[ 1.,  0.,  0.,  0.],
           [ 0.,  2.,  0.,  0.],
           [ 0.,  0.,  3.,  0.],
           [ 0.,  0.,  0.,  4.]])
    '''

    @matmat
    def forwardAdjoint(x):
        return v[:, np.newaxis] * x

    return LinearOperator( (len(v), len(v)),
            forwardAdjoint, forwardAdjoint)

