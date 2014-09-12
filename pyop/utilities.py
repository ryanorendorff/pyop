'''
Often, certain forward and adjoint functions are simpler to define as a
matrix-matrix multiplication or to operate on a reshape version of the
input. The decorators defined here allow for functions of these forms to be
modified to matrix-matrix functions easily. The examples for each decorator
give a good sample of how to use such functions.
'''

import numpy as np
import scipy.sparse

from functools import update_wrapper

import six

def __wrapIfPy3(wrapper, f):
    if six.PY2:
        return wrapper
    else:
        return update_wrapper(wrapper, f)


def matmat(f):
    ''' Convert 1D array to 2D column and back to 1D after calculation.

    This function is meant to work only with functions following the
    :class:`.LinearOperator` calling declaration, which is a function taking
    in only one argument, the data to work on.

    If the function receives a sparse input, then this wrapper does nothing
    since a sparse matrix cannot be squeeze or reshaped without significant
    alterations in its structure.

    Parameters
    ----------
    f : function
        The function that is programmed to operate on arrays of two
        dimensions.

    Returns
    -------
    function
        A function that ensures the input data is at least 2 dimensional and
        that the result is a 1D array if the input was also 1D.

    See Also
    --------
    matvec : Defines a matrix-matrix( function from matrix-matvec.
    matvectorized : Like matvec but reshapes the input automatically.

    Examples
    --------
    >>> import numpy as np
    >>> from pyop import LinearOperator
    >>> #
    >>> def squareEye(shape):
    ...     #
    ...     @matmat
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
            y = x.reshape(-1, 1)
        else:
            y = x

        res = f(y)

        if x.ndim == 1:
            return np.ravel(res)

        return res

    return __wrapIfPy3(wrapper, f)


def matvec(f):
    ''' Operates on an input matrix one column at a time.

    A function that implements a matrix-matrix function from a matrix-
    vector function. This decorator takes an input matrix, feeds each column
    as a 1D array to the function f, and then combines the results.

    Often it is more natural to define matrix-vector function, since the
    matrix-matrix function is just the collection of the matrix-vector
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
    matmat : ensures the input to a function is 2 dimensional.
    matvectorized : Like matvec but reshapes the input automatically.

    Examples
    --------
    >>> import numpy as np
    >>> from pyop import matvec
    >>> #
    >>> @matvec
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

    @matmat
    def wrapper(x):
        return np.column_stack(f(c) for c in x.T)

    return __wrapIfPy3(wrapper, f)


def matvectorized(shape, order = 'C'):
    ''' Decorator to turn a function on a matrix into matrix-matrix product.

    This decorator is an extension of matvec. It operates on each column
    of a matrix at a time, but before passing the column to the decorated
    function, matvectorized reshapes the column based on the dimensions and
    order supplied into a nD array.

    Parameters
    ----------
    shape : tuple
        The shape of the data in non-matvec form.

    order = {'C', 'F', 'A'}, optional
        The order by which the matvecized image is reshaped. This is the
        same parameter as given to functions like numpy.reshape. For a
        discussion of the memory efficiency of different orders and how that
        is determined by the underlying format, see the documentation of
        commands that take an order argument.

    Returns
    -------
    function
        A function that can be used to decorate another function.

    Examples
    --------
    >>> import numpy as np
    >>> from pyop import matvectorized
    >>> #
    >>> @matvectorized((2, 2))
    ... def multFirstColumn(img):
    ...     img[:, 0] *= 2
    ...     return img
    >>> #
    >>> multFirstColumn(np.array([[1, 1, 1, 1], [2, 1, 2, 1]]).T)
    array([[2, 4],
           [1, 1],
           [2, 4],
           [1, 1]])

    See Also
    --------
    matmat : ensures the input to a function is 2 dimensional.
    matvec : Like matvectorized but does not reshapes the input.
    '''
    def decorator(f):

        @matvec
        def wrapper(column):
            arr = np.reshape(column, shape, order)
            res = f(arr)
            return np.ravel(res, order)

        return __wrapIfPy3(wrapper, f)

    return decorator
