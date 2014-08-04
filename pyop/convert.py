from pyop.linop import LinearOperator


import numpy as np
import scipy.sparse as sp

import scipy.sparse.linalg as linalg

####################
#  From/To Matrix  #
####################

def toLinearOperator(m):
    ''' Lifts a numpy.ndarray into the LinearOperator type.

    When using this function, all of the functional space and time
    saving benefits are lost since the result is an operator that
    performs a standard matrix multiplication.

    Parameters
    ----------
    m : numpy.ndarray or scipy.sparse
        the matrix to convert into an LinearOperator.

    Returns
    -------
    LinearOperator
        the transform lifted into the LinearOperator context.
    '''
    return LinearOperator(m.shape, m.dot, m.T.dot)


def toMatrix(O, sparse = False):
    ''' Convert an LinearOperator into its matrix form.

    Converting an LinearOperator into a matrix could make a large
    amount of space and time. One of the purposes of the LinearOperator
    abstraction is to allow transformations that would otherwise require
    too much space or time to be computed.

    This function will not work if the functions passed in the creation of
    the LinearOperator do not respect matrix inputs. This enforcement cannot
    be easily done automatically, and as such the developer will need to be
    concious of this fact when writing LinearOperators.

    Parameters
    ----------
    sparse: bool, optional
        Passes in a sparse matrix to the LinearOperator instead of a dense
        one. For this parameter to work, the functions that a LinearOperator
        calls (forward, adjoint) must respect the type of the input.

    Returns
    -------
    numpy.ndarray
        the matrix representation of the transform.
    '''
    if sparse:
        I = sp.eye(O.shape[1])
    else:
        I = np.eye(O.shape[1]).T ## Turn into Fortran ordering.

    return O(I)


###########
#  Scipy  #
###########

def toScipyLinearOperator(O, dtype = np.float):
    ''' Converts a LinearOperator into a scipy.sparse.linalg.LinearOperator

    Scipy comes with come handy functions for calculating properties of a
    linear operator in a matrix free manner, using their LinearOperator
    class. However, that class is difficult to use for composition because
    it does not support the transpose operation and has no block matrix
    creation function. This function is designed to allow the developer to
    work with the composition functions provided by the PyOp package while
    still being able to utilise these SciPy functions.

    Parameters
    ----------
    O: LinearOperator
        The LinearOperator to convert into the Scipy format.

    Returns
    -------
    scipy.sparse.linalg.LinearOperator
        The operator O as a Scipy LinearOperator, suitable for the Scipy
        matrix free functions.

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.sparse.linalg as linalg
    >>> A = LinearOperator((4,4), lambda x: x, lambda  x: x)
    >>> B = LinearOperator((4,4), lambda x: x, lambda  x: x)
    >>> C = (A + 2*B)
    >>> D = toScipyLinearOperator(C.T*C)
    >>> D(np.eye(4))
    array([[ 9.,  0.,  0.,  0.],
           [ 0.,  9.,  0.,  0.],
           [ 0.,  0.,  9.,  0.],
           [ 0.,  0.,  0.,  9.]])
    >>> linalg.eigsh(D, 1)[0][0] ## First eigenvalue
    9.0
    '''
    return linalg.LinearOperator(O.shape, O._forward, O._adjoint,
            dtype=dtype)

