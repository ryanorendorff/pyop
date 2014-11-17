PyOp Motivation
===============

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
  \min_x ~~ \left \Vert Ax - y \right \Vert_2

In general, this problem has an analytical solution given by:

.. math::
  \hat{x} = A^{\dagger} y

where :math:`A^{\dagger}` is the Moore-Penrose inverse of `A`. For
ill-conditioned problems, we may add Tikhonov regularization:

.. math::
  \min_x ~~ \left \Vert Ax - y \right \Vert_2 + \sqrt{\lambda} \left \Vert
  x \right \Vert_2

which has the analytical solution:

.. math::
  \hat{x} = (A^\top A + \lambda I)^{-1} A^\top y

We may also want to incorporate *a priori* information such as positivity of
the image `x` leading to a convex optimization formulation:

.. math::
  \min_x ~~& \left \Vert Ax - y \right \Vert_2^2 \\
  \mathrm{s.t.} ~~& x \succeq 0

which has no analytical solution.

Often, `A` and the subcomponents are defined in terms of matrices. When this is
possible and easy, then formulating and solving the reconstruction problem is
very straightforward.  However, in some applications, including image
reconstruction, the matrix form (even when sparse and stored accordingly) can
be too big to fit in memory and/or cause cache thrashing issues. In these
cases, "matrix-free" methods become much more attractive, if not necessary.

A matrix is only one way of representing a linear transformation. In the most
general case, a linear transformation need only have some functional
representation for its forward operation. However the adjoint operation is
often just as necessary.  When we encapsulate these operations in the form of a
function, we can store this information on the order of kB of memory as opposed
to the GBs needed for large matrices.

Considering the image reconstruction problems described above, when couched as
a convex optimization problem, gradient descent or other methods can be used to
solve the reconstruction problem using only the forward and adjoint operations
(compared to the analytical solutions which require an inverse).

In general, when using linear operators, we need to be able to properly
formulate the forward and adjoint operations and accommodate various forms of
the operator depending on the specific context (e.g., matrix vs. functional).
PyOp is designed to give users the flexibility to seamlessly create, compose,
and use linear operators defined as dense matrices, sparse matrices, in
functional form, or any arbitrary mix.
