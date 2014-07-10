#pylint: disable=W0104,W0108
import pytest
import pyop

import numpy as np


#####################
#  Common Matrices  #
#####################

a_44 = np.random.rand(4,4)
b_44 = np.random.rand(4,4)
c_45 = np.random.rand(4,5)
d_54 = np.random.rand(5,4)
e_64 = np.random.rand(5,4)
i_44 = np.eye(4)
i_55 = np.eye(5)
v_4 = np.random.rand(4)
w_4 = np.random.rand(4)
x_5 = np.random.rand(5)
y_5 = np.random.rand(5)


#######################
#  Operator Matrices  #
#######################

aop_44 = pyop.toLinearOperator(a_44)
bop_44 = pyop.toLinearOperator(b_44)
cop_45 = pyop.toLinearOperator(c_45)
dop_54 = pyop.toLinearOperator(d_54)
iop_44 = pyop.toLinearOperator(i_44)
iop_55 = pyop.toLinearOperator(i_55)


#######################################################################
#                                Tests                                #
#######################################################################

def testShapes():
    assert a_44.shape == aop_44.shape
    assert a_44.T.shape == aop_44.T.shape


def testForward():
    assert np.array_equal(np.dot(a_44, v_4), aop_44(v_4))
    assert np.array_equal(np.dot(c_45, x_5), cop_45(x_5))

    with pytest.raises(pyop.error.InnerDimensionMismatch):
        aop_44(x_5)

    with pytest.raises(pyop.error.InnerDimensionMismatch):
        aop_44(d_54)


def testAdjoint():
    assert np.array_equal(np.dot(a_44.T, v_4), aop_44.T(v_4))
    assert np.array_equal(np.dot(c_45.T, v_4), cop_45.T(v_4))

    with pytest.raises(pyop.error.MissingAdjoint):
        a = pyop.LinearOperator((4,4), lambda _, x: x)
        a.T


def testAdd():
    assert np.array_equal((a_44 + b_44), pyop.toMatrix(aop_44 + bop_44))

    with pytest.raises(pyop.error.AllDimensionMismatch):
        aop_44 + cop_45

    with pytest.raises(pyop.error.AllDimensionMismatch):
        cop_45 + aop_44


def testSub():
    assert np.array_equal((a_44 - b_44), pyop.toMatrix(aop_44 - bop_44))

    with pytest.raises(pyop.error.AllDimensionMismatch):
        aop_44 - cop_45

    with pytest.raises(pyop.error.AllDimensionMismatch):
        cop_45 - aop_44


def testMul():
    assert np.array_equal(np.dot(a_44, b_44),
            pyop.toMatrix(aop_44 * bop_44))

    np.testing.assert_allclose(np.dot(np.dot(c_45, d_54), a_44),
                          pyop.toMatrix(cop_45*dop_54*aop_44))

def testPow():
    assert np.array_equal(np.dot(a_44, np.dot(a_44, np.dot(a_44, a_44))),
            pyop.toMatrix(aop_44**4))


def testNeg():
    assert np.array_equal(-a_44, pyop.toMatrix(-aop_44))


def testPos():
    assert np.array_equal(+a_44, pyop.toMatrix(+aop_44))


def testFromMatrix():
    aop = pyop.toLinearOperator(a_44)

    assert a_44.shape == aop.shape
    assert np.array_equal(np.dot(a_44, v_4), aop(v_4))
    assert np.array_equal(np.dot(a_44.T, v_4), aop.T(v_4))


def testToMatrix():
    one = np.ones((5,4))

    one_fn = lambda shape, x: np.tile(np.sum(x,0), (shape[0], 1))
    one_op = pyop.LinearOperator((5,4), one_fn, one_fn)

    assert np.array_equal(one, pyop.toMatrix(one_op))
    assert np.array_equal(one.T, pyop.toMatrix(one_op.T))


def testEquality():
    def func(s, x):
        return s + x

    a = pyop.LinearOperator((4,4), func, func)
    b = pyop.LinearOperator((4,4), func, func)

    assert a == b

    c = pyop.LinearOperator((5,4), func, func)

    assert a != c

    d = pyop.LinearOperator((4,4), lambda s, x: func(s, x), func)

    assert a != d

    e = pyop.LinearOperator((4,4), lambda s, x: func(s, x), func)

    assert d != e

    f = pyop.LinearOperator((4,4), func, lambda s, x: func(s, x))

    assert d != f
    assert e != f
