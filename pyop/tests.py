'''
These are tests to assist with creating :class:`.LinearOperator`.
'''
import numpy as np
from numpy.testing import assert_allclose

def adjointTest(O, significant = 7):
    ''' Test for verifying forward and adjoint functions in LinearOperator.

    adjointTest verifies correctness for the forward and adjoint functions
    for an operator via asserting :math:`<A^H y, x> = <y, A x>`

    Parameters
    ----------
    O : LinearOperator
        The LinearOperator to test.

    significant : int, optional
        Perform the test with a numerical accuracy of "significant" digits.

    Examples
    --------
    >>> from pyop import LinearOperator
    >>> A = LinearOperator((4,4), lambda _, x: x, lambda _, x: x)
    >>> adjointTest(A)
    >>> B = LinearOperator((4,4), lambda _, x: x, lambda _, x: 2*x)
    >>> adjointTest(B)
    ... # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
        ...
    AssertionError:
    '''

    x = np.random.rand(O.shape[1])
    y = np.random.rand(O.shape[0])

    assert_allclose(O.T(y).dot(x), y.dot(O(x)), rtol = 10**(-significant))
