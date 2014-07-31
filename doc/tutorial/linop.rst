Linear Operator Tutorial
========================

:class:`pyop.linop.LinearOperators` are actually quite simple to create.
The most simple case is to just pass forward and adjoint functions to the
:class:`pyop.linop.LinearOperator` constructor, like the following example
that implements a square identity. ::

  def id(x):
    return x

  I = LinearOperator((4, 4), id, id)

Often it is helpful to create a :class:`pyop.linop.LinearOperator` in a
nested function to create many similar operators that define the same basic
morphism over different dimensions. ::

  def squareIdentity(shape):

      def id(x):
          return x

      return LinearOperator((shape, shape), id, id)

Note that in the above example, since ``squareIdentity`` is a nested
fuction, the inner ``id`` function can operate on any data created before it
in the ``squareIdentity`` lexical environment.

For some good examples of defining operators, look at the source for those
already defined in PyOp itself.
