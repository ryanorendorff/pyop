Block Module Tutorial
=====================

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
