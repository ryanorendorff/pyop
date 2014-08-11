'''
The functions below create LinearOperator versions of fftn and the like,
operating on vectorized versions of an input array.
'''

import numpy as np

## For calculating shape of FFT LinearOperators
from functools import reduce
from operator import mul

from pyop import matvectorized, LinearOperator


##########
#  FFTs  #
##########

def __fft(f, dual, shape, s, order):
    ''' f  is which fft function to use (fft, ifft), dual is the
    dual to it. '''

    ## If s is None then just make it shape, no padding or cropping.
    if s is None:
        s = shape

    if len(s) != len(shape):
        raise ValueError("FFT does not understand different sized "
                         "shape ({}), s ({})".format(shape, s))

    for d in shape:
        if d < 0:
            raise ValueError("shape must be positive. {}".format(shape))

    for d in s:
        if d < 0:
            raise ValueError("s must be positive. {}".format(s))

    domain = reduce(mul, shape)
    codomain = reduce(mul, s)

    ## Some helper functions
    positive = lambda x: 0 if x < 0 else x
    column_deficit = lambda d: positive(shape[d] - s[d])

    @matvectorized(shape, order)
    def forward(x):
        return f(x, s = s)


    @matvectorized(s, order)
    def adjoint(x):
        res = dual(x)
        res_pad = np.pad(res,
            [(0, column_deficit(d)) for d, _ in enumerate(shape)],
            'constant', constant_values = 0)
        return res_pad[tuple(slice(None, d) for d in shape)]


    return LinearOperator((codomain, domain), forward, adjoint)


def fft(shape, s = None, order = 'C'):
    ''' Matrix free DFT operator in N-dimensions.

    Performs the FFT on an N dimensional input that has been vectorized.
    The adjoint of this operator is the IFFT. The parameter s influences
    precisely the relation between the forward and adjoint: if the input is
    zero padded due to s, then the adjoint will crop back down to the shape
    size. Similarly, if s crops the input, then the adjoint will pad the
    output with zeros to match the shape size.

    The results, even if the input is real, will be complex.

    Parameters
    ----------
    shape : tuple
        The shape of the array to perform an FFT on.
    s : sequence of ints, optional
        Length of each axis in the input. Must be the same number of
        elements as shape. If the length for a dimension is shorter the
        shape value, then the input is cropped. If the dimension is longer,
        it is padded with zeros. If s is not given (the default), then s =
        shape (no cropping or padding).
    order = {'C', 'F'}, optional
        The order by which the vectorized image is reshaped. This is the
        same parameter as given to functions like numpy.reshape. For a
        discussion of the memory efficiency of different orders and how that
        is determined by the underlying format, see the documentation of
        commands that take an order argument.

    Returns
    -------
    LinearOperator
        A LinearOperator performing the fft on all dimensions.

    Raises
    ------
    ValueError
        If either shape or s contain a negative element.
    ValueError
        If the length of s is not the length of shape.

    See Also
    --------
    ifft : LinearOperator version of ifftn, use if setting the s parameter
        for the ifft is required.
    fftshift : LinearOperator version of fftshift
    ifftshift : LinearOperator version of ifftshift

    Examples
    --------
    >>> import numpy as np
    >>> from pyop.operators import fft
    >>> a = np.array([0, 1, 2, 3, 2, 1, 0])
    >>> F = fft(a.shape)
    >>> F(a)
    array([ 9.00000000+0.j        , -4.54891734-2.19064313j,
            0.19202147+0.24078731j, -0.14310413-0.62698017j,
           -0.14310413+0.62698017j,  0.19202147-0.24078731j,
           -4.54891734+2.19064313j])
    >>> np.real((F.T*F)(a))
    array([  2.95002118e-15,   1.00000000e+00,   2.00000000e+00,
             3.00000000e+00,   2.00000000e+00,   1.00000000e+00,
            -1.90323947e-15])
    '''
    return __fft(np.fft.fftn, np.fft.ifftn, shape, s, order)


def ifft(shape, s = None, order = 'C'):
    ''' Matrix free inverse DFT operator in N-dimensions.

    Performs the IFFT on an N dimensional input that has been vectorized.
    The adjoint of this operator is the FFT. The parameter s influences
    precisely the relation between the forward and adjoint: if the input is
    zero padded due to s, then the adjoint will crop back down to the shape
    size. Similarly, if s crops the input, then the adjoint will pad the
    output with zeros to match the shape size.

    The results, even if the input is real, will be complex.

    Parameters
    ----------
    shape : tuple
        The shape of the array to perform an IFFT on.
    s : sequence of ints, optional
        Length of each axis in the input. Must be the same number of
        elements as shape. If the length for a dimension is shorter the
        shape value, then the input is cropped. If the dimension is longer,
        it is padded with zeros. If s is not given (the default), then s =
        shape (no cropping or padding).
    order = {'C', 'F'}, optional
        The order by which the vectorized image is reshaped. This is the
        same parameter as given to functions like numpy.reshape. For a
        discussion of the memory efficiency of different orders and how that
        is determined by the underlying format, see the documentation of
        commands that take an order argument.

    Returns
    -------
    LinearOperator
        A LinearOperator performing the fft on all dimensions.

    Raises
    ------
    ValueError
        If either shape or s contain a negative element.
    ValueError
        If the length of s is not the length of shape.

    See Also
    --------
    fft : LinearOperator version of fftn, use if setting the s parameter
        for the fft is required.
    fftshift : LinearOperator version of fftshift
    ifftshift : LinearOperator version of ifftshift

    Examples
    --------
    >>> import numpy as np
    >>> from pyop.operators import ifft
    >>> a = np.array([0, 1, 2, 3, 2, 1, 0])
    >>> F = ifft(a.shape)
    >>> F(a)
    array([ 1.28571429+0.j        , -0.64984533+0.31294902j,
            0.02743164-0.03439819j, -0.02044345+0.0895686j ,
           -0.02044345-0.0895686j ,  0.02743164+0.03439819j,
           -0.64984533-0.31294902j])
    >>> np.real((F.T*F)(a))
    array([  2.78943535e-15,   1.00000000e+00,   2.00000000e+00,
             3.00000000e+00,   2.00000000e+00,   1.00000000e+00,
            -2.05391260e-15])
    '''
    return __fft(np.fft.ifftn, np.fft.fftn, shape, s, order)
