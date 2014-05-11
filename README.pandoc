Python Linear Transformation Operators (PyOp)
=============================================

This package aims to replace standard matrix linear transformations on
vector spaces with their function based equivalent. These "functional"
operators can be significantly faster and require less storage space, while
maintaining the same composition properties of standard matrices. For
example, the DFT can be represented as a matrix but requires O(n^2) storage
and, in the simplest formulation, O(n^2) operations to compute the FFT.
However, the FFT/DFT algorithm takes only O(n log n) time and less than
quadratic space (I cannot find a good reference for the space complexity of
the FFT).

Included in this package are the base classes used to create these
functional operators as well as a standard set of operators, such as the
identity transform, the zero transform, etc.


Package Requirements
--------------------
- Python 2.7, Python 3.3, or Python 3.4
- numpy>=1.8
- six>=1.6

For testing you will need `pytest>=2.5`
