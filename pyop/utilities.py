import numpy as np

from functools import wraps

def ensure2dColumn(f):
    ''' Convert 1D array input to 2D column array and back to 1D after
    calculation.

    This function is meant to work only with functions following the
    LinearOperator calling declaration, where the first argument is the
    operator's size (commonly op_shape) and the second is the piece of data
    to work on (commonly x).

    Parameters
    ----------
    f : function with arguments (op_shape, x)
        The function to ensure that the data input (x) is at least 2
        dimensions.

    Returns
    -------
    function
        A function that ensures the input data is at least 2 dimensional and
        that the result is a 1D array, when possible.

    Examples
    --------
    >>> import numpy as np
    >>> from pyop import LinearOperator
    >>> #
    >>> def squareEye(shape):
    ...     #
    ...     @ensure2dColumn
    ...     def id(op_shape, x):
    ...         return x
    ...     #
    ...     return LinearOperator((shape, shape), id, id)
    >>> #
    >>> I = squareEye(4)
    >>> I(np.arange(4))
    array([0, 1, 2, 3])
    '''

    @wraps(f)
    def wrapper(op_shape, x):
        ## Convert a 1D x into a column 2D array.
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        res = f(op_shape, x)

        ## Return 2D array to 1D if applicable.
        return np.squeeze(res)

    return wrapper
