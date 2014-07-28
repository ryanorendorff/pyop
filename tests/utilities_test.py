#pylint: disable=W0104,W0108
import pytest
import pyop

import numpy as np


#######################################################################
#                                Tests                                #
#######################################################################

def testEnsure2dColumn(capsys):

    @pyop.ensure2dColumn
    def printShape(x):
        print(x.shape)
        return x

    input_vec = np.random.rand(10)
    output = printShape(input_vec)
    print_out, _ = capsys.readouterr()

    np.testing.assert_allclose(input_vec, output)
    assert print_out == "(10, 1)\n"

    input_vec = np.random.rand(10, 10)
    output = printShape(input_vec)
    print_out, _ = capsys.readouterr()

    np.testing.assert_allclose(input_vec, output)
    assert print_out == "(10, 10)\n"
