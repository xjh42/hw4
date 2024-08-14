from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_Z_original = array_api.max(Z, axis=self.axes, keepdims=True)
        max_Z = array_api.max(Z, axis=self.axes)
        out = array_api.log(array_api.sum(array_api.exp(Z - max_Z_original), axis=self.axes)) + max_Z
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        z = node.inputs[0]
        max_z = z.realize_cached_data().max(axis=self.axes, keepdims=True)
        exp_z = exp(z - max_z)
        sum_exp_z = summation(exp_z, axes=self.axes)
        grad = out_grad / sum_exp_z
        expand_shape = list(z.shape)
        axes = range(len(expand_shape)) if self.axes is None else self.axes
        for axe in axes:
            expand_shape[axe] = 1
        grad = grad.reshape(expand_shape).broadcast_to(z.shape)
        return exp_z * grad
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

