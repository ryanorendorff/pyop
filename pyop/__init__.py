from .linop import (
      LinearOperator
    )

from .convert import(
      toScipyLinearOperator
    , toLinearOperator
    , toMatrix
    )

from .tests import adjointTest

from .utilities import matmat, matvec, matvectorized

from .block import bmat, blockDiag, horzcat, vertcat

from . import operators
