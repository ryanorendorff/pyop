Linear Operator Tutorial
========================

PyOp is designed to generalize the concept of a linear operator in the
context of scientific computing and software development.

:class:`~pyop.linop.LinearOperator` instances are quite simple to create.
Below we discuss different ways of defining operators. For more information
look at the source for those already defined in PyOp itself.

Unbound Functions
-----------------

The basic way to create a :class:`~pyop.linop.LinearOperator` instance is to
just pass forward and adjoint functions to the
:class:`~pyop.linop.LinearOperator` constructor. The following example
implements a square identity in this manner. ::

  def id(x):
    return x

  I = LinearOperator((4, 4), id, id)

This is a great way to test new forward and adjoint functions alone during
development.

Nested Functions/Closures
-------------------------

Often it is helpful to create a :class:`~pyop.linop.LinearOperator` in a
nested function. ::

  def squareIdentity(shape):

      # Do some pre-processing here.

      def id(x):
          return x

      return LinearOperator((shape, shape), id, id)

Note that in the above example, since ``squareIdentity`` is a nested fuction,
the inner ``id`` function can operate on any data created before it in the
``squareIdentity`` lexical environment.  This can very useful when realizing
complicated operators or, for example, to create many similar operators that
define the same basic morphism over different dimensions, as shown here.
