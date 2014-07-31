Simple Matrix Free Operators
============================

PyOp is a framework for creating Linear Operators in a functional form.
Operators specified in this format have a few benefits.

- The memory to store the operator can be much smaller. For example,
  an image reconstruction done by the authors used to require storing
  a _sparse_ marix that was over 150 GB. With this matrix free version,
  the amount of space went down to kilobytes.
- The time to perform an operation can be greatly reduced. In the same
  image reconstruction problem, converting only one of the operations
  to a matrix free linear operator, while using a gradient descent
  algorithm, reduced the computation time by an order of magnitude.
- For some (although by no means all) morphisms it is simpler to
  describe the operation in a function based form than converting the
  operation into a matrix.
- The matrix free operators can be converted back into matrices easily
  should the need for certain numerical tools be required (factorizations),
  although many facilities are provided that work on matrix free inputs.

This package makes it simple to create matrix free linear operators. Perhaps
more importantly, it provides simple composition tools, allowing operators
to be chained together with the same tools one would find in a matrix
package (for example, adding operators or creating block operators).


Tutorials
---------

Creating matrix free linear operators is straight forward, but a few
conditions must be kept in mind in order to make the most of this package.
The following tutorials walk through these conditions and an example of
creating an operator.

.. toctree::
  tutorial/rules
  tutorial/linop


Tools for Creating Linear Operators
-----------------------------------

.. toctree::
    api/linop
    api/utilities
    api/convert
    api/block
    api/tests


Prefined Operators
------------------

.. toctree::
    pyop.operators
