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


################
#  FFT Shifts  #
################

def __fftshift(f, shape, axes, order):
    ''' f is which shift to perform (fftshift, ifftshift) '''

    for d in shape:
        if d < 0:
            raise ValueError("shape must be positive. {}".format(shape))

    if axes is not None:
        for a in axes:
            if a < 0:
                raise ValueError("axes must be positive. {}".format(axes))

            if a >= len(shape):
                raise ValueError("Out of bound axes. {}".format(axes))


    domain = reduce(mul, shape)

    @matvectorized(shape, order)
    def forward(x):
        return f(x, axes = axes)

    return LinearOperator((domain, domain), forward, forward)


def fftshift(shape, axes = None, order = 'C'):
    ''' Shift the zero frequency to the center.

    Flips the axes in an input to move the zero frequency. By default, all
    of the axes are flipped, moving the DC component to the center of the
    array. However, this can be modified by altering the axes input.

    This operator is self adjoint. This means that it will likely be used
    with the ifftshift to sandwich an operator, such as `ifftshift *
    operator * fftshift`.

    Parameters
    ----------
    shape : tuple
        The shape of the array to perform an FFT on.
    axes : int or shape tuple, optional
        Axes to shift. The None default shifts all axes. Axes can be
        specified more than once and they will be flipped multiple times.
    order = {'C', 'F'}, optional
        The order by which the vectorized image is reshaped. This is the
        same parameter as given to functions like numpy.reshape. For a
        discussion of the memory efficiency of different orders and how that
        is determined by the underlying format, see the documentation of
        commands that take an order argument.

    Returns
    -------
    LinearOperator
        A LinearOperator performing the fftshift.

    See Also
    --------
    fft : LinearOperator version of fftn, use if setting the s parameter
        for the fft is required.
    ifft : LinearOperator version of ifftn, use if setting the s parameter
        for the ifft is required.
    ifftshift : LinearOperator version of ifftshift

    Examples
    --------
    >>> import numpy as np
    >>> from pyop.operators import fft, fftshift
    >>> a = np.array([0, 1, 2, 3, 2, 1, 0])
    >>> F = fft(a.shape)
    >>> F(a)
    array([ 9.00000000+0.j        , -4.54891734-2.19064313j,
            0.19202147+0.24078731j, -0.14310413-0.62698017j,
           -0.14310413+0.62698017j,  0.19202147-0.24078731j,
           -4.54891734+2.19064313j])
    >>> S = fftshift(a.shape)
    >>> (S*F)(a)
    array([-0.14310413+0.62698017j,  0.19202147-0.24078731j,
           -4.54891734+2.19064313j,  9.00000000+0.j        ,
           -4.54891734-2.19064313j,  0.19202147+0.24078731j,
           -0.14310413-0.62698017j])
    '''
    return __fftshift(np.fft.fftshift, shape, axes, order)


def ifftshift(shape, axes = None, order = 'C'):
    ''' Shift the zero frequency to the center.

    Flips the axes in an input to move the zero frequency. By default, all
    of the axes are flipped, moving the DC component to the center of the
    array. However, this can be modified by altering the axes input.

    This operator is self adjoint. This means that it will likely be used
    with the ifftshift to sandwich an operator, such as `fftshift *
    operator * ifftshift`.

    Parameters
    ----------
    shape : tuple
        The shape of the array to perform an FFT on.
    axes : int or shape tuple, optional
        Axes to shift. The None default shifts all axes. Axes can be
        specified more than once and they will be flipped multiple times.
    order = {'C', 'F'}, optional
        The order by which the vectorized image is reshaped. This is the
        same parameter as given to functions like numpy.reshape. For a
        discussion of the memory efficiency of different orders and how that
        is determined by the underlying format, see the documentation of
        commands that take an order argument.

    Returns
    -------
    LinearOperator
        A LinearOperator performing the fftshift.

    See Also
    --------
    fft : LinearOperator version of fftn, use if setting the s parameter
        for the fft is required.
    ifft : LinearOperator version of ifftn, use if setting the s parameter
        for the ifft is required.
    fftshift : LinearOperator version of fftshift

    Examples
    --------
    >>> import numpy as np
    >>> from pyop.operators import ifft, ifftshift
    >>> a = np.array([0, 1, 2, 3, 2, 1, 0])
    >>> F = ifft(a.shape)
    >>> F(a)
    array([ 1.28571429+0.j        , -0.64984533+0.31294902j,
            0.02743164-0.03439819j, -0.02044345+0.0895686j ,
           -0.02044345-0.0895686j ,  0.02743164+0.03439819j,
           -0.64984533-0.31294902j])
    >>> S = ifftshift(a.shape)
    >>> (S*F)(a)
    array([-0.02044345+0.0895686j , -0.02044345-0.0895686j ,
            0.02743164+0.03439819j, -0.64984533-0.31294902j,
            1.28571429+0.j        , -0.64984533+0.31294902j,
            0.02743164-0.03439819j])
    '''
    return __fftshift(np.fft.ifftshift, shape, axes, order)


##################################
#  Composition Helper Functions  #
##################################

def __fftwrap(fft_func, O, shape, s, shift, order):

    if fft_func is fft:
        shift_func = fftshift
        shift_dual = ifftshift

    if fft_func is ifft:
        shift_func = ifftshift
        shift_dual = fftshift

    if shift is "all":
        shift = None

    F = fft_func(shape, s, order)


    if shift is "none":
        return F.T * O * F
    else:
        S = shift_func(s, shift, order)
        Sinv = shift_dual(s, shift, order)

        return F.T * Sinv * O * S * F


def fftwrap(O, shape, s = None, shift = 'none', order = 'C'):
    ''' Surrounds an operator with FFT operations.

    Given an operator O, this function returns the following in the case of
    no shift (shift = 'none'),

    ..math:: F^T * O * F

    and the following when shift is any other value.

    ..math:: F^T * Sinv * O * S * F

    where

    - :math:`F` is the FFT operation
    - :math:`S` is a FFT shift
    - :math:`Sinv` is an IFFT shift

    Parameters
    ----------
    O : LinearOperator
        The operator to wrap.
    shape : tuple
        The shape of the array to perform an FFT on.
    s : sequence of ints, optional
        Length of each axis in the input. Must be the same number of
        elements as shape. If the length for a dimension is shorter the
        shape value, then the input is cropped. If the dimension is longer,
        it is padded with zeros. If s is not given (the default), then s =
        shape (no cropping or padding).
    shift : "none", "all" or shape tuple
        The axes to shift. If "none", no FFT shift is performed. If "all",
        every axis is shifted to so the DC value is in the center of the
        array. Otherwise shift takes a tuple input that shifts only the
        specified axes. Duplicates are allowed, this causes the axes to be
        flipped multiple times.
    order = {'C', 'F'}, optional
        The order by which the vectorized image is reshaped. This is the
        same parameter as given to functions like numpy.reshape. For a
        discussion of the memory efficiency of different orders and how that
        is determined by the underlying format, see the documentation of
        commands that take an order argument.

    Returns
    -------
    LinearOperator
        A LinearOperator that performs some action in FFT/x-space domain.

    See Also
    --------
    fft : LinearOperator version of fftn, use if setting the s parameter
        for the fft is required.
    ifft : LinearOperator version of ifftn, use if setting the s parameter
        for the ifft is required.
    fftshift : LinearOperator version of fftshift
    ifftshift : LinearOperator version of ifftshift
    ifftwrap : The same as fftshift but starts with an IFFT.

    Examples
    --------
    >>> import numpy as np
    >>> from pyop.operators import fftwrap
    >>> from pyop import LinearOperator
    >>> a = np.array([1, 1, 1, 1])
    >>> I = LinearOperator((4, 4), lambda x: x, lambda x: x)
    >>> F = fftwrap(I, (4,))
    >>> F(a)
    array([ 1.+0.j,  1.+0.j,  1.+0.j,  1.+0.j])
    '''
    return __fftwrap(fft, O, shape, s, shift, order)


def ifftwrap(O, shape, s = None, shift = 'none', order = 'C'):
    ''' Surrounds an operator with IFFT operations.

    Given an operator O, this function returns the following in the case of
    no shift (shift = 'none'),

    ..math:: F^T * O * F

    and the following when shift is any other value.

    ..math:: F^T * Sinv * O * S * F

    where

    - :math:`F` is the IFFT operation
    - :math:`S` is an IFFT shift
    - :math:`Sinv` is a FFT shift

    Parameters
    ----------
    O : LinearOperator
        The operator to wrap.
    shape : tuple
        The shape of the array to perform an FFT on.
    s : sequence of ints, optional
        Length of each axis in the input. Must be the same number of
        elements as shape. If the length for a dimension is shorter the
        shape value, then the input is cropped. If the dimension is longer,
        it is padded with zeros. If s is not given (the default), then s =
        shape (no cropping or padding).
    shift : "none", "all" or shape tuple
        The axes to shift. If "none", no FFT shift is performed. If "all",
        every axis is shifted to so the DC value is in the center of the
        array. Otherwise shift takes a tuple input that shifts only the
        specified axes. Duplicates are allowed, this causes the axes to be
        flipped multiple times.
    order = {'C', 'F'}, optional
        The order by which the vectorized image is reshaped. This is the
        same parameter as given to functions like numpy.reshape. For a
        discussion of the memory efficiency of different orders and how that
        is determined by the underlying format, see the documentation of
        commands that take an order argument.

    Returns
    -------
    LinearOperator
        A LinearOperator that performs some action in FFT/x-space domain.

    See Also
    --------
    fft : LinearOperator version of fftn, use if setting the s parameter
        for the fft is required.
    ifft : LinearOperator version of ifftn, use if setting the s parameter
        for the ifft is required.
    fftshift : LinearOperator version of fftshift
    ifftshift : LinearOperator version of ifftshift
    fftwrap : The same as fftshift but starts with an FFT.

    Examples
    --------
    >>> import numpy as np
    >>> from pyop.operators import ifftwrap
    >>> from pyop import LinearOperator
    >>> a = np.array([1, 1, 1, 1])
    >>> I = LinearOperator((4, 4), lambda x: x, lambda x: x)
    >>> F = ifftwrap(I, (4,))
    >>> F(a)
    array([ 1.+0.j,  1.+0.j,  1.+0.j,  1.+0.j])
    '''
    return __fftwrap(ifft, O, shape, s, shift, order)
