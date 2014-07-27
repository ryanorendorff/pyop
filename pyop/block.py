from numpy import vsplit, vstack, tile, concatenate, cumsum, add
from pyop import LinearOperator, ensure2dColumn

import six

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
    LinearOperator
        The new block operator.
    '''

    if len(blocks) == 0:
        raise ValueError('Empty list supplied to block operator.')

    ## First collapse the rows using horz_cat.
    vert_block_op = [horzcat(row) for row in blocks]
    ## Next collapse the column into one block operator using vert_cat.
    block_op = vertcat(vert_block_op)

    return block_op


def blockDiag(blocks):
    ''' Converts a list of operators into a new operator.

    The new operator is composed of diagonal blocks described by the list.

    Parameters
    ----------
    blocks : [LinearOperator]
        A list of operators that constitute consecutive blocks in a larger
        block diagonal operator.

    Returns
    -------
    LinearOperator
        The new block diagonal operator.
    '''

    rows = sum(h.shape[0] for b in blocks)
    cols = sum(h.shape[1] for b in blocks)

    ## Generate a list containing the indices to split the vector x
    ## to be sent to each component of the block operator.
    forward_splitting_idx = cumsum([b.shape[1] for b in blocks])
    adjoint_splitting_idx = cumsum([b.shape[0] for b in blocks])

    @ensure2dColumn
    def forwardFunction(x):

        ## Split vector subcomponents on the block operator lengths.
        ## TODO: Rename for general  matrix inputs.
        vec_components = vsplit(x, forward_splitting_idx)

        ## Apply each operator to corresponding subvector and concatenate
        ## the results.
        sub_outvecs = (b(v) for (b, v) in six.moves.zip(blocks,
            vec_components))

        ## Concatenate the output sub-vectors together.
        return vstack(sub_outvecs)


    @ensure2dColumn
    def adjointFunction(x):

        ## Split vector subcomponents on the block operator lengths.
        ## TODO: Rename for general  matrix inputs.
        vec_components = vsplit(x, adjoint_splitting_idx)

        ## Apply each operator to corresponding subvector and concatenate
        ## the results.
        sub_outvecs = (b.T(v) for (b, v) in six.moves.zip(blocks,
            vec_components))

        ## Concatenate the output sub-vectors together.
        return vstack(sub_outvecs)


    return LinearOperator((rows, cols),
            forwardFunction,
            adjointFunction)


def __hstack(horz_blocks):
    ''' Converts list of horizontal operators into one linear operator.'''

    ## Generate a list containing the indices to split the vector x
    ## to be sent to each component of the block operator.
    splitting_idx = cumsum([b.shape[1] for b in horz_blocks])

    @ensure2dColumn
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


def __vstack(vert_blocks):
    ''' Converts list of vertical operators into one operator.'''

    @ensure2dColumn
    def opFunction(x):

        ## Apply each operator (forward or adjoint) to the input vector to
        ## get output vector sub-components.
        sub_outvecs = (b(x) for b in vert_blocks)
        ## Concatenate the output sub-vectors together.
        return vstack(sub_outvecs)


    return opFunction


def horzcat(horz_blocks):
    ''' Converts list of operators into one operator.

    The new operator is created assuming the list corresponds to a row of
    blocks in a larger block matrix-like operator.

    Parameters
    ----------
    horz_blocks : [LinearOperator]
        A list of LinearOperator objects.

    Returns
    -------
    LinearOperator
        The new horizontally stacked block operator.
    '''

    if len(horz_blocks) == 0:
        raise ValueError('Horizontal concatenation of empty list.')

    rows = horz_blocks[0].shape[0]
    cols = sum(h.shape[1] for h in horz_blocks)
    if not all(b.shape[0] == rows for b in horz_blocks):
        raise ValueError('''Block operator horizontal concatenation failed:
                         row mismatch.''')

    return LinearOperator((rows, cols),
            __hstack(horz_blocks),
            __vstack([h.T for h in horz_blocks]))


def vertcat(vert_blocks):
    ''' Converts list of operators into one operator.

    The new operator is created assuming the list corresponds to a column of
    blocks in a larger block matrix-like operator.

    Parameters
    ----------
    horz_blocks: [LinearOperator]
        A list of LinearOperator objects.

    Returns
    -------
    LinearOperator
        The new vertically stacked block operator.
    '''

    if len(vert_blocks) == 0:
        raise ValueError('Vertical concatenation of empty list.')

    rows = sum(v.shape[0] for v in vert_blocks)
    cols = vert_blocks[0].shape[1]
    if not all(b.shape[1] == cols for b in vert_blocks):
        raise ValueError('''Block operator vertical concatenation failed:
                          column mismatch.''')

    return LinearOperator((rows, cols),
            __vstack(vert_blocks),
            __hstack([v.T for v in vert_blocks]))
