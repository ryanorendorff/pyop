Version 0.1.1 (2017/11/30 01:39 UTC-07)
=======================================

Fix adjointTest to use a relative tolerance directly. Prior, it used
a number of significant digits that the results should compare to, but
this was implemented incorrectly.


Version 0.1.0 (2014/11/17 09:12 UTC-08)
=======================================

Initial public version. This version includes

- base LinearOperator class
- Transformation decorators
    - matmat
    - matvec
    - matvectorized
- Conversion between different operator types (mostly to a SciPy LinearOperator
  to be able to use the matrix free functions in SciPy.linalg).
- Creation of block operators
- Basic matrix operators
    - eye
    - zeros
    - ones
    - diag
    - select
- Convolution operators, operating on vectorized inputs.
    - nD convolution
    - nD gradient
- FFT operators
    - FFT, IFFT
    - FFT, IFFT shift operators
    - Wrapping of other operators (fftwrap, ifftwrap)
