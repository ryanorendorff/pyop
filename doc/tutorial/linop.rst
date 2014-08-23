Linear Operator Tutorial
========================

PyOp is designed to generalize the concept of a linear operator in the
context of scientific computing and software development.

:class:`~pyop.linop.LinearOperator` instances are quite simple to create.
Below we discuss different ways of defining operators. For more information
look at the source for those already defined in PyOp itself.

The last part of this tutorial motivates and describes the use of PyOp
Linear Operators in the context of image reconstruction.

Unbound Functions
-----------------

The basic way to create a :class:`~pyop.linop.LinearOperator` instance is to
just pass forward and adjoint functions to the
:class:`~pyop.linop.LinearOperator` constructor. The following example
implements a square identity in this manner. ::

  def id(x):
    return x

  I = LinearOperator((4, 4), id, id)

Nested Functions/Closures
-------------------------

Often it is helpful to create a :class:`~pyop.linop.LinearOperator` in a
nested function. ::

  def squareIdentity(shape):

      def id(x):
          return x

      return LinearOperator((shape, shape), id, id)

Note that in the above example, since ``squareIdentity`` is a nested fuction,
the inner ``id`` function can operate on any data created before it in the
``squareIdentity`` lexical environment.  This can very useful when realizing
complicated operators or, for example, to create many similar operators that
define the same basic morphism over different dimensions, as shown here.

Classes
-------

It is also possible to pass :class:`~pyop.linop.LinearOperator` a forward and
adjoint function defined as methods bound to an object of some class. This can
provide similar utility as using nested functions/closures, although we believe
nested functions/closures to generally be a better route. ::

  class SquareIdentity(object):

      def __init__(self, shape):
          if not shape[0] == shape[1]:
              raise ValueError('SquareIdentity object must square.')
          self.shape = shape

      def forward(self, x):
          return x

      def adjoint(self, x):
          return x

  I = SquareIdentity((4, 4))
  Iop = LinearOperator(I.shape, I.forward, I.adjoint)

The forward and adjoint passed to the constructor need not be functions. They
can also be objects with a call method defined. ::

  class SquareIdentity(object):

      def __init__(self, shape):
          if not shape[0] == shape[1]:
              raise ValueError('SquareIdentity object must square.')
          self.shape = shape

      @property
      def T(self):
          return SquareIdentity(self.shape)

      def __call__(self, x):
          return x

  I = SquareIdentity((4, 4))
  Iop = LinearOperator(I.shape, I, I.T)

Image Reconstruction Example
----------------------------

This section of the tutorial largely discusses linear operators in the context
of image/signal processing; this is only for convenience of explanation and
expertise of the authors. At its core, PyOp is intended to be completely
general.

Linear systems abound in engineering disciplines such as imaging. Indeed, most
if not all of the major imaging systems used in the world are linear and
shift-invariant (LSI). This is not just happenstance -- LSI is a crucial
feature for efficiency and quantitation of an imaging system and, for example,
its noise properties.

There are major ramifications for image reconstruction given LSI properties.
For example, one popular method for image reconstruction is to solve a
so-called inverse problem.  If a system is LSI, it can always be described with
the notation `y = Ax` where `y` is the output of the system and `A`
characterizes all of the linear transformations associated with the system.
These could include the fundamental physics of the modality, signal processing
steps, and the particulars of the scanning sequence. To recover the image `x`,
we must invert this system.

In practice, `A` is the composition of many linear transformations or
constructed from block components, and it is often these different
sub-components that one can formulate. For example, we often have

.. math::
  A = A_1 A_2 \dots A_N \\
  A = \begin{bmatrix} A_1 & A_2 \\ A_3 & A_4 \end{bmatrix}

or a combination these. As long as we can define each of these sub-components,
composition or blocking of linear operators is straightforward.  In our image
reconstruction problem, we can then find the solution to the inverse problem
to reconstruct an image:

.. math::
  \hat{x} = A^{\dagger} y

where :math:`A^{\dagger}` is the Moore-Penrose inverse of `A`. For
ill-conditioned problems, we may add Tikhonov regularization:

.. math::
  \hat{x} = (A^\top A + \lambda I)^{-1} A^\top y

We may also want to incorporate *a priori* information such as positivity of
the image `x` leading to a convex optimization formulation:

.. math::
  \begin{align*}
      \min_x ~~& \left \Vert Ax - y \right \Vert_2^2 \\
      \mathrm{s.t.} ~~& x \succeq 0
  \end{align*}

Often, `A` and the subcomponents are defined in terms of matrices. When this is
possible and easy, then formulating and solving the reconstruction problem is
very straightforward.  However, in some applications, including image
reconstruction, the matrix form (even when sparse and stored accordingly) can
be too big to fit in memory and/or cause cache thrashing issues. In these
cases, "matrix-free" methods become much more attractive, if not necessary.

A matrix is only one way of representing a linear transformation. In the most
general case, a linear transformation need only have some functional
representation for its forward operation. However the adjoint operation is
often just as necessary. For example, in all of the image reconstruction
examples above, the adjoint is necessary, and for the first two, an inverse is
necessary to solve for `x`. When we encapsulate these operations in the form of
a function, we can store this information on the order of kB of memory as
opposed to the GBs needed for large matrices.

When using linear operators, we need to be able to properly formulate the
forward and adjoint operations and accommodate various forms of the operator
depending on the specific context (e.g., matrix vs. functional).  PyOp is
designed to give users the flexibility to seamlessly create, compose, and use
linear operators defined as dense matrices, sparse matrices, in functional
form, or any arbitrary mix.
