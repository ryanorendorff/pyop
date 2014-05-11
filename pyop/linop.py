from pyop.error import \
        AllDimensionMismatch, InnerDimensionMismatch, MissingAdjoint

## For __pow__
from itertools import repeat
from functools import reduce
from operator import mul

## Check for __scaledmul__
from numbers import Number

## Uses NumPy style docstrings: http://goo.gl/xd873p
class LinearOperator(object):
    ''' LinearOperators for performing linear transformations without
    matrices.

    To use a matrix in this class, provide a function

    Parameters
    ----------
    shape : tuple
        a two element tuple describing the number of rows/columns this
        operator represents if it were in matrix form.
    forward : function
        the functional representation of a linear transform.
    adjoint : function, optional
        the adjoint of a linear transformation in functional form.

    Attributes
    ----------
    shape : (int, int)
        a pair representing the rows/columns of this transformation, if it
        were in matrix form.
    T : LinearOperator
        the transpose/adjoint of the current operator (if defined).

    Examples
    --------
    >>> from numpy import array
    >>> identity = LinearOperator((4,4), lambda _, x: x)
    >>> identity(array([1,2,3,4]))
    array([1, 2, 3, 4])
    >>> identity = LinearOperator((4,4), lambda _, x: x, lambda _, x: x)
    >>> identity.T(array([1,2,3,4]))
    array([1, 2, 3, 4])
    '''

    def __init__(self, shape, forward, adjoint=None):

        self._forward = forward

        if adjoint is None:
            self._adjoint = LinearOperator.__missingAdjoint
        else:
            self._adjoint = adjoint

        assert len(shape) == 2 and isinstance(shape, tuple)
        self._shape = shape


    @property
    def shape(self):
        return self._shape


    @staticmethod
    def __missingAdjoint(_, __): # pylint: disable=W0613
        raise MissingAdjoint()


    @property
    def T(self):
        ''' Provides the transpose of the current operator.

        Returns
        -------
        LinearOperator
            the transpose of the current operator.

        Raises
        ------
        MissingAdjoint
            if the transpose function of the original operator was not
            defined.
        '''
        if self._adjoint is LinearOperator.__missingAdjoint:
            raise MissingAdjoint()

        return LinearOperator(self._shape[::-1],
                self._adjoint, self._forward)


    ########################
    #  Dimension Checkers  #
    ########################

    @staticmethod
    def __checkSameDims(a, b):
        return a.shape == b.shape


    @staticmethod
    def __checkInnerDims(a, b):
        return a.shape[1] == b.shape[0]


    def __call__(self, x):
        if LinearOperator.__checkInnerDims(self, x):
            return self._forward(self._shape, x)
        else:
            raise InnerDimensionMismatch(self._shape, x.shape)


    ## Numpy 1.9 will allow overriding dot.
    #def __numpy_ufunc__(self, ufunc, method, i, inputs, **kwargs):

    ##############
    #  Numerics  #
    ##############

    def __add__(self, other):
        if not LinearOperator.__checkSameDims(self, other):
            raise AllDimensionMismatch(self, other)

        return LinearOperator(self._shape,
                lambda _, x: self(x) + other(x),
                lambda _, x: self.T(x) + other.T(x))


    def __sub__(self, other):
        if not LinearOperator.__checkSameDims(self, other):
            raise AllDimensionMismatch(self, other)

        return LinearOperator(self._shape,
                lambda _, x: self(x) - other(x),
                lambda _, x: self.T(x) - other.T(x))


    def dot(self, other):
        return self*other


    def __scaledmul__(self, other):
        return LinearOperator(self.shape,
                lambda _, x: other*self(x),
                lambda _, x: other*self(x))


    def __mul__(self, other):
        if isinstance(other, Number):
            return self.__scaledmul__(other)

        if not LinearOperator.__checkInnerDims(self, other):
            raise InnerDimensionMismatch(self, other)

        if isinstance(other, LinearOperator):
            return LinearOperator((self._shape[0], other._shape[1]),
                    lambda _, x: self(other(x)),
                    lambda _, x: other.T(self.T(x)))
        else:
            return self(other)


    def __rmul__(self, other):
        if isinstance(other, Number):
            return self.__scaledmul__(other)
        else:
            return NotImplemented


    def __pow__(self, power):
        if not isinstance(power, int):
            return NotImplemented
        assert power > 0, "Power must be > 0"

        return reduce(mul, repeat(self, power))


    def __neg__(self):
        return LinearOperator(self._shape,
                lambda _, x: self(-x),
                lambda _, x: self(-x))


    def __pos__(self):
        return self.__copy__()


    ####################
    #  Block Creation  #
    ####################

    @staticmethod
    def fromBlocks(blocks):
        ## See hmatrix fromBlocks, in lib/Data/Packed/Matrix.hs. Allows for
        ## constant values to be automagically be expanded to blocks of the
        ## correct size.
        pass


    @staticmethod
    def diagBlocks(blocks):
        ## Use fromBlocks and the fact that it can take constant values and
        ## turn them into blocks.
        pass



    ##################
    #  OO functions  #
    ##################

    def __copy__(self):
        return LinearOperator(self._shape, self._forward, self._adjoint)


    def __repr__(self):
        return "LinearOperator(%r, %r, %r)" % (self._shape,
                self._forward, self._adjoint)


    def __str__(self):
        return self.__repr__()


    ##############
    #  Equality  #
    ##############

    def __eq__(self, other):
        if self._shape != other._shape:
            return False

        if self._forward != other._forward:
            return False

        if self._adjoint != other._adjoint:
            return False

        return True


    def __ne__(self, other):
        return not self == other



