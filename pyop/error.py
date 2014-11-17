class DimensionMismatch(ValueError):
    pass


class AllDimensionMismatch(ValueError):
    def __init__(self, shape1, shape2):

        super(AllDimensionMismatch, self).__init__(
                "All dimensions must match. %s, %s" % (shape1, shape2))


class InnerDimensionMismatch(DimensionMismatch):
    def __init__(self, shape1, shape2):

        super(InnerDimensionMismatch, self).__init__(
                "Inner dimensions must match. %s, %s" % (shape1, shape2))


class ZeroDimension(DimensionMismatch):
    def __init__(self, shape):
        super(ZeroDimension, self).__init__(
                "Shape is of zero dimensions. {}".format(shape))


class HighOrderTensor(DimensionMismatch):
    def __init__(self, shape):
        super(HighOrderTensor, self).__init__(
                "Shape is of >2 dimensions. {}".format(shape))


class MissingAdjoint(AttributeError):
    def __init__(self):
        super(MissingAdjoint, self).__init__(
                "LinearOperator missing transpose function.")


