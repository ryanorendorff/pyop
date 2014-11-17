#pylint: disable=W0104,W0108
import pyop

import numpy as np


#######################################################################
#                                Tests                                #
#######################################################################

def tesMatmat(capsys):

    @pyop.matmat
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
#  vector  #
############

@pyop.matvec
def multFirstColumn(column):
    img = column.reshape((2, 2), order = 'C')
    img[:, 0] *= 2
    return np.ravel(img, order = 'C')


def testVectorOnMatrix():
    np.testing.assert_allclose(
        multFirstColumn(np.array([[1, 1, 1, 1], [2, 1, 2, 1]]).T),
        np.array([[2, 4], [1, 1], [2, 4], [1, 1]]))


def testVectorOnVector():
    np.testing.assert_allclose(
        multFirstColumn(np.array([1, 1, 1, 1])),
        np.array(np.array([2, 1, 2, 1])))


#################
#  vectorArray  #
#################

@pyop.matvectorized((2,2))
def multFirstColumnImg(img):
    img[:, 0] *= 2
    return img


def testVectorArrayOnMatrix():
    np.testing.assert_allclose(
        multFirstColumnImg(np.array([[1, 1, 1, 1], [2, 1, 2, 1]]).T),
        np.array([[2, 4], [1, 1], [2, 4], [1, 1]]))


def testVectorArrayOnVector():
    np.testing.assert_allclose(
        multFirstColumnImg(np.array([1, 1, 1, 1])),
        np.array(np.array([2, 1, 2, 1])))
