from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

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
        max_Z_original = Z.max(axis=self.axes, keepdims=True)
        if self.axes is None:
            new_shape = [max_Z_original.shape[i] if i < max_Z_original.ndim else 1 for i in range(Z.ndim)]
            max_Z_original = max_Z_original.reshape(new_shape)
        max_Z_original = max_Z_original.broadcast_to(Z.shape)
        max_Z = Z.max(axis=self.axes)
        sub_Z = Z - max_Z_original
        exp_sub_Z = sub_Z.exp()
        sum_exp_sub_Z = exp_sub_Z.sum(axis=self.axes)
        return sum_exp_sub_Z.log() + max_Z
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        z = node.inputs[0]
        max_z = z.realize_cached_data().max(axis=self.axes, keepdims=True)
        z_shape = z.shape
        z_dim = len(z_shape)
        max_z_dim = len(max_z.shape)
        if self.axes is None:
            new_shape = [max_z.shape[i] if i < max_z_dim else 1 for i in range(z_dim)]
            max_z = max_z.reshape(new_shape)
        max_z = max_z.broadcast_to(z_shape)
        exp_z = exp(z - max_z)
        sum_exp_z = summation(exp_z, axes=self.axes)
        grad = out_grad / sum_exp_z
        expand_shape = list(z.shape)
        axes = range(len(expand_shape)) if self.axes is None else self.axes
        if type(self.axes) == int:
            axes = [self.axes]
        for axe in axes:
            expand_shape[axe] = 1
        grad = grad.reshape(expand_shape).broadcast_to(z.shape)
        return exp_z * grad
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

