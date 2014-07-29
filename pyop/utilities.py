import numpy as np
import scipy.sparse

from functools import update_wrapper

import six

def ensure2dColumn(f):
    ''' Convert 1D array input to 2D column array and back to 1D after
    calculation.

    This function is meant to work only with functions following the
    LinearOperator calling declaration, where the first argument is the
    operator's size (commonly op_shape) and the second is the piece of data
    to work on (commonly x).

    If the function receives a sparse input, then this wrapper does nothing
    since a sparse matrix cannot be squeeze or reshaped without significant
    alterations in its structure.

    Parameters
    ----------
    f : function with arguments (x)
        The function to ensure that the data input (x) is at least 2
        dimensions.

    Returns
    -------
    function
        A function that ensures the input data is at least 2 dimensional and
        that the result is a 1D array, in the case of a dense input.

    Examples
    --------
    >>> import numpy as np
    >>> from pyop import LinearOperator
    >>> #
    >>> def squareEye(shape):
    ...     #
    ...     @ensure2dColumn
    ...     def id(x):
    ...         return x
    ...     #
    ...     return LinearOperator((shape, shape), id, id)
    >>> #
    >>> I = squareEye(4)
    >>> I(np.arange(4))
    array([0, 1, 2, 3])
    '''

    def wrapper(x):
        ## If the input is sparse, then pass through without alteration.
        if scipy.sparse.issparse(x):
            return f(x)

        ## Convert a 1D x into a column 2D array.
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        res = f(x)

        ## Return 2D array to 1D if applicable.
        return  np.squeeze(res)

    if six.PY2:
        return wrapper
    else:
        return update_wrapper(wrapper, f)


def vector(f):
    ''' Operates on an input matrix one column at a time.

    A function that implements a matrix-matrix function from a matrix-
    vector function. This decorator takes an input matrix, feeds each column
    as a 1D vector to the function f, and then combines the results.

    Often it is more natural to define the matrix-vector function, since
    the matrix-matrix function is just the collection of the matrix-vector
    function applied to each column of the input. For these cases, use this
    decorator.

    Parameters
    ----------
    f : function with arguments (x)
        The function that implements a matrix-vector product.

    Returns
    -------
    function
        A function that implements a matrix-matrix function.

    See Also
    --------
    ensure2dColumn : ensures the input to a function is 2 dimensional.

    Examples
    --------
    >>> import numpy as np
    >>> from pyop import vector
    >>> #
    >>> @vector
    ... def multFirstColumn(column):
    ...     img = column.reshape((2, 2), order = 'C')
    ...     img[:, 0] *= 2
    ...     return img.flatten(0)
    >>> #
    >>> multFirstColumn(np.array([[1, 1, 1, 1], [2, 1, 2, 1]]).T)
    array([[2, 4],
           [1, 1],
           [2, 4],
           [1, 1]])
    '''

    @ensure2dColumn
    def wrapper(x):
        return np.column_stack(f(c) for c in x.T)

    if six.PY2:
        return wrapper
    else:
        return update_wrapper(wrapper, f)

