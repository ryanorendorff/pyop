r'''
In practice, :class:`~pyop.linop.LinearOperator` instances composed of distinct
sub-blocks are common. In general, access to both the constituent operators and
larger operators is desired.  The :mod:`~pyop.block` module provides several
functions to help build block instances of :class:`~pyop.linop.LinearOperator`
from simpler components.

:func:`~pyop.block.hstack` and :func:`~pyop.block.vstack` are the workhorses
of :mod:`~pyop.block` and are analogous to their numpy counterparts. These
functions can be used alone to squash a row or column of
:class:`~pyop.linop.LinearOperator` instances, respectively, into a single
:class:`~pyop.linop.LinearOperator` instance. These functions expect a
list of :class:`~pyop.linop.LinearOperator` instances as input.

:func:`~pyop.block.bmat` is the generic block-builder, and takes in a
list of lists of :class:`~pyop.linop.LinearOperator` instances and returns
a single block operator. :func:`~pyop.block.bmat` assumes the list of
lists is in row-major order.

:func:`~pyop.block.blockDiag` allows for easy creation of common block
diagonal operators, given a list of the diagonal component operators.

.. math::
  E = \begin{bmatrix} A & B \\ C & D \end{bmatrix}

We can easily build :math:`E` or just its rows and columns using
:mod:`~pyop.block` functions. ::

  A = LinearOperator((10, 12), forward, adjoint)
  B = LinearOperator((10, 30), forward, adjoint)
  C = LinearOperator((15, 12), forward, adjoint)
  D = LinearOperator((15, 30), forward, adjoint)

  row1 = hstack([A, B])
  row2 = hstack([C, D])

  col1 = vstack([A, C])
  col2 = vstack([B, D])

  E1 = vstack([row1, row2])
  E2 = hstack([col1, col2])
  E3 = bmat([[A, B], [C, D]])

In this example, `E1`, `E2`, and `E3` are equivalent.

We can also easily make block diagonal operators with
:func:`~pyop.block.blockDiag`:

.. math::
  D = \begin{bmatrix} A & \mathbf{0} & \mathbf{0} \\ \mathbf{0} & B &
  \mathbf{0} \\ \mathbf{0} & \mathbf{0} & C \end{bmatrix}

::

  A = LinearOperator(op_shape1, forward1, adjoint1)
  B = LinearOperator(op_shape2, forward2, adjoint2)
  C = LinearOperator(op_shape3, forward3, adjoint3)

  D = blockDiag([A, B, C])
'''

from numpy import vsplit, vstack, tile, concatenate, cumsum, add
from numpy import vstack as npvstack
from pyop import LinearOperator, matmat
from scipy.misc import doccer

import six

docdict = {
    'blocks' :
'''blocks : [LinearOperator]
    A list of LinearOperator objects.''',
    'LinearOperator' :
'''LinearOperator
    The new block operator.''',

    ## The see also section.
    'blockDiag' : '''blockDiag : Construct a LinearOperator from block
    diagonal components.''',
    'bmat' : '''bmat : Construct a LinearOperator from LinearOperator
    subcomponents.''',
    'hstack' : '''hstack : Squash a row of LinearOperators to a single block
    LinearOperator.''',
    'vstack' : '''vstack : Squash a column of LinearOperators to a single
    block LinearOperator.'''
    }

docfill = doccer.filldoc(docdict)

@docfill
def bmat(blocks):
    ''' Converts a list of lists into a new operator.

    The new operator is composed of blocks described by the list.

    Parameters
    ----------
    blocks : [[LinearOperator]]
        A list of lists, with each base component a linear operator (objects
        instantiated from the LinearOperator class).

    Returns
    -------
    %(LinearOperator)s

    See Also
    --------
    %(blockDiag)s
    %(hstack)s
    %(vstack)s

    Examples
    --------
    >>> from pyop import toLinearOperator, toMatrix
    >>> from pyop.block import bmat
    >>> from numpy import array
    >>> A = toLinearOperator(array([[1., 2.], [0., 4.]]))
    >>> B = toLinearOperator(array([[3., 0.], [5., 0.]]))
    >>> C = toLinearOperator(array([[6., 0., 0., 7], [0., 8., 9., 0.]]))
    >>> blocks = [[A, B], [C]]
    >>> D = bmat(blocks)
    >>> toMatrix(D)
    array([[ 1.,  2.,  3.,  0.],
           [ 0.,  4.,  5.,  0.],
           [ 6.,  0.,  0.,  7.],
           [ 0.,  8.,  9.,  0.]])
    '''

    if len(blocks) == 0:
        raise ValueError('Empty list supplied to block operator.')

    ## First collapse the rows using horz_cat.
    vert_block_op = [hstack(row) for row in blocks]
    ## Next collapse the column into one block operator using vert_cat.
    block_op = vstack(vert_block_op)

    return block_op


@docfill
def blockDiag(blocks):
    ''' Converts a list of operators into a new operator.

    The new operator is composed of diagonal blocks described by the list.

    Parameters
    ----------
    %(blocks)s

    Returns
    -------
    %(LinearOperator)s

    See Also
    --------
    %(bmat)s

    Examples
    --------
    >>> from pyop.block import blockDiag
    >>> from pyop import toLinearOperator, toMatrix
    >>> from numpy import array
    >>> A = toLinearOperator(array([[1., 2.], [3., 4.]]))
    >>> B = toLinearOperator(array([[5., 6.], [7., 8.]]))
    >>> C = blockDiag([A, B])
    >>> toMatrix(C)
    array([[ 1.,  2.,  0.,  0.],
           [ 3.,  4.,  0.,  0.],
           [ 0.,  0.,  5.,  6.],
           [ 0.,  0.,  7.,  8.]])
    '''

    if len(blocks) == 0:
        raise ValueError('Empty list supplied to diagonal block operator.')

    rows = sum(b.shape[0] for b in blocks)
    cols = sum(b.shape[1] for b in blocks)

    ## Generate a list containing the indices to split the vector x
    ## to be sent to each component of the block operator.
    forward_splitting_idx = cumsum([b.shape[1] for b in blocks])
    adjoint_splitting_idx = cumsum([b.shape[0] for b in blocks])

    @matmat
    def forwardFunction(x):

        ## Split vector subcomponents on the block operator lengths.
        ## TODO: Rename for general  matrix inputs.
        vec_components = vsplit(x, forward_splitting_idx)

        ## Apply each operator to corresponding subvector and concatenate
        ## the results.
        sub_outvecs = (b(v) for (b, v) in six.moves.zip(blocks,
            vec_components))

        ## Concatenate the output sub-vectors together.
        return npvstack(sub_outvecs)


    @matmat
    def adjointFunction(x):

        ## Split vector subcomponents on the block operator lengths.
        ## TODO: Rename for general  matrix inputs.
        vec_components = vsplit(x, adjoint_splitting_idx)

        ## Apply each operator to corresponding subvector and concatenate
        ## the results.
        sub_outvecs = (b.T(v) for (b, v) in six.moves.zip(blocks,
            vec_components))

        ## Concatenate the output sub-vectors together.
        return npvstack(sub_outvecs)


    return LinearOperator((rows, cols),
            forwardFunction,
            adjointFunction)


def __horzcat(horz_blocks):
    ''' Converts list of horizontal operators into one linear operator.'''

    ## Generate a list containing the indices to split the vector x
    ## to be sent to each component of the block operator.
    splitting_idx = cumsum([b.shape[1] for b in horz_blocks])

    @matmat
    def opFunction(x):

        ## Split vector subcomponents based on the block operator lengths.
        ## TODO: Rename all of these to imply matrix not vector
        vec_components = vsplit(x, splitting_idx)

        ## Apply each operator to its corresponding subvector and add the
        ## results. Two cases for forward and adjoint functions.
        sub_outvecs = (b(v) for (b, v) in six.moves.zip(horz_blocks,
            vec_components))

        ## Add the vectors together.
        return sum(sub_outvecs)


    return opFunction


def __vertcat(vert_blocks):
    ''' Converts list of vertical operators into one operator.'''

    @matmat
    def opFunction(x):

        ## Apply each operator (forward or adjoint) to the input vector to
        ## get output vector sub-components.
        sub_outvecs = (b(x) for b in vert_blocks)
        ## Concatenate the output sub-vectors together.
        return npvstack(sub_outvecs)


    return opFunction


@docfill
def hstack(blocks):
    ''' Converts list of operators into one operator.

    The new operator is created assuming the list corresponds to a row of
    blocks in a larger block matrix-like operator.

    Parameters
    ----------
    %(blocks)s

    Returns
    -------
    %(LinearOperator)s

    See Also
    --------
    %(vstack)s
    %(bmat)s

    Examples
    --------
    >>> from pyop.block import hstack
    >>> from pyop import toLinearOperator, toMatrix
    >>> from numpy import array
    >>> A = toLinearOperator(array([[1., 2.], [4., 5.]]))
    >>> B = toLinearOperator(array([[3.], [6.]]))
    >>> C = hstack([A, B])
    >>> toMatrix(C)
    array([[ 1.,  2.,  3.],
           [ 4.,  5.,  6.]])
    '''

    if len(blocks) == 0:
        raise ValueError('Horizontal concatenation of empty list.')

    rows = blocks[0].shape[0]
    cols = sum(h.shape[1] for h in blocks)
    if not all(b.shape[0] == rows for b in blocks):
        raise ValueError('Block operator horizontal concatenation failed: '
                         'row mismatch.')

    return LinearOperator((rows, cols),
            __horzcat(blocks),
            __vertcat([h.T for h in blocks]))


@docfill
def vstack(blocks):
    ''' Converts list of operators into one operator.

    The new operator is created assuming the list corresponds to a column of
    blocks in a larger block matrix-like operator.

    Parameters
    ----------
    %(blocks)s

    Returns
    -------
    %(LinearOperator)s

    See Also
    --------
    %(hstack)s
    %(bmat)s

    Examples
    --------
    >>> from pyop.block import vstack
    >>> from pyop import toLinearOperator, toMatrix
    >>> from numpy import array
    >>> A = toLinearOperator(array([[1., 4.], [2., 5.]]))
    >>> B = toLinearOperator(array([[3., 6.]]))
    >>> C = vstack([A, B])
    >>> toMatrix(C)
    array([[ 1.,  4.],
           [ 2.,  5.],
           [ 3.,  6.]])
    '''

    if len(blocks) == 0:
        raise ValueError('Vertical concatenation of empty list.')

    rows = sum(v.shape[0] for v in blocks)
    cols = blocks[0].shape[1]
    if not all(b.shape[1] == cols for b in blocks):
        raise ValueError('Block operator vertical concatenation failed: '
                         'column mismatch.')

    return LinearOperator((rows, cols),
            __vertcat(blocks),
            __horzcat([v.T for v in blocks]))
