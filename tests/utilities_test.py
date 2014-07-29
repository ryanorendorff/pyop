#pylint: disable=W0104,W0108
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


############
#  Vector  #
############
@pyop.vector
def multFirstColumn(column):
    img = column.reshape((2, 2), order = 'C')
    img[:, 0] *= 2
    return img.flatten(0)


def testVectorOnMatrix():
    np.testing.assert_allclose(
        multFirstColumn(np.array([[1, 1, 1, 1], [2, 1, 2, 1]]).T),
        np.array([[2, 4], [1, 1], [2, 4], [1, 1]]))


def testVectorOnVector():
    np.testing.assert_allclose(
        multFirstColumn(np.array([1, 1, 1, 1])),
        np.array(np.array([2, 1, 2, 1])))
