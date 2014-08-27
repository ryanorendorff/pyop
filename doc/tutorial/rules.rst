Operator Rules
==============

The most basic form of a :class:`~pyop.linop.LinearOperator` takes in a shape
defining the operation and a function (called the forward function) that
operators on only one input. However, certain operators are practically
useless. ::

  A = LinearOperator((4, 4), lambda x: array([0, 2, 0, 0])
  A(array([1, 1, 1, 1])) # The result is probably not what you wanted.

While the :class:`~pyop.linop.LinearOperator` cannot automatically determine
non-linear or bogus forward functions (or at least not easily), it does
perform the following checks when the ``call`` method is used.

- both the operator and the input have a shape attribute,
- the inner dimensions of the operator and the input match, and
- the return of the forward function matches the expected dimensions
  (the outer dimensions).

Therefore, it is impossible to use the following operators due to the checks.
 ::

  LinearOperator((4, 4), lambda x: "Hinc lucem et pocula sacra.")
  LinearOperator((4, 4), lambda x: "Pax et Lux.")
  LinearOperator((4, 4), lambda x, y: x + y)

Similarly, the following will be rejected because only positive pair shapes
make sense, as they define a mapping from one finite dimension (the domain)
to another (the codomain). ::

  LinearOperator((4, 0), lambda x: x)
  LinearOperator((4, -1), lambda x: x)
  LinearOperator((4, ), lambda x: x)

Even though there is no automatic way to determine if an operator provides
a correct (or linear) function--at least without intense lexical analyzing
that could solve only the most basic of cases--adhering to the following
rules will help ensure that an operator is functioning properly.


The Rules
---------

1. :class:`~pyop.linop.LinearOperator` instances are nearly useless without an
   adjoint function. While it is not required, it becomes impossible to
   perform calculations such as an eigenvalue decomposition without an
   adjoint, as well as any first order convex optimization solver.
2. For :class:`~pyop.linop.LinearOperator` instances that have an adjoint
   function defined, they should pass the :func:`~pyop.tests.adjointTest`.
   This is a basic check that the forward and adjoint are indeed are each
   other's "opposites", although it does not guarantee that the operator is
   defined correctly.
3. If there is a reference operation (such as a matrix), check that the
   function output matches the reference. While an operator that passes the
   adjoint test is usually doing what you expect, this is not always the
   case (defining the convolution operator is an example where an adjoint
   test can easily pass without defining the correct operator).
4. Forward and adjoint functions that are defined on NumPy arrays should
   respect the dimensionality of their input. This means that 1D inputs
   should result in 1D outputs, and 2D inputs should lead to 2D outputs.
   This follows the way that many NumPy functions are styled.
5. Forward and adjoint functions should be defined to work on 2D array inputs
   when the inputs are NumPy arrays. This is to say that the functions
   should be defined as matrix-matrix multiplications instead of
   matrix-vector multiplications, for example. This allows for
   the :func:`~pyop.convert.toMatrix` function to work, which is
   helpful for turning an operator into its matrix form for further
   manipulation (for example, factorization or low rank approximations).
   :func:`~pyop.convert.toMatrix` even works on operators that are created
   out of any of the composition or combining tools, as long as all the
   functions are defined as matrix-matrix multiplies.
6. Do not make operator functions that hold state. Since the forward
   and adjoint functions can be anything that implements ``call``, they
   can be functions, closures, or classes. All of these can contain state,
   but doing so breaks the immutable flavour of the operator and makes
   understanding the code more difficult. For example ::

     res = [1, 1]
     def forward(x):
         res *= x
         return res

     A = LinearOperator(..., forward, ...)

     ## These two sequences lead to different results for the
     ## second statement.
     A(array([1, 0]))
     A(array([2, 1]))

     A(array([0, 0]))
     A(array([2, 1]))


   In other words, the forward and adjoint functions, if they are
   to match those of standard linear operator theory, must be `pure
   <https://en.wikipedia.org/wiki/Pure_function>`_ functions.


Tips
----

Here are some tips for developing operators that come from sometimes painful
experience.

- Use IPython for testing, it is quite helpful. :-)
- Define the forward and adjoint functions in a normal namespace before
  putting them inside either a nested function or a class. They are simpler
  to iterate through in this form, while the nested/class form is really
  just a nice (simple) packaging of the result.
- Use the :mod:`~pyop.utilities` decorators when possible to ease the
  creation of matrix-matrix functions. These decorators convert vector or
  vectorized (a nD array vectorized) functions to matrix-matrix functions
  and take care of concatenating the results together, flattening along the
  correct dimensions, etc.


Notes
-----

One thing to note is that these checks do `not` specify the input data
type. While the input to a :class:`~pyop.linop.LinearOperator` is often a
NumPy ndarray, it is entirely possible to use any input type that defines
a shape pair. This could be useful, for example, if the input was a graph
that was simpler to express and operate on as a ``Graph`` type instead of an
adjacency matrix. This is an (accidental) result of Python's duck typing.

:class:`~pyop.linop.LinearOperator` instances are designed to behave
in an `immutable` manner, although they are strictly not immutable as
a determined programmer can always redefine the functions held by a
:class:`~pyop.linop.LinearOperator` or modify the instance (or class)
at runtime. However, in the course of normal programming, they can be
treated as a non-hashable immutable (I realise the silliness) as all of the
composition rules create new operators containing the old ones. This means
that if an operator is given to a function, it will not change after the
function executes.
