"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype))
        if bias:
            self.bias = Parameter(init.kaiming_uniform(out_features, 1, device=device, dtype=dtype).transpose())
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        y = X.matmul(self.weight)
        if self.bias is not None:
            broad_bias = self.bias.broadcast_to(y.shape)
            y = y + broad_bias
        return y
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        flatten_size = 1
        for s in X.shape[1:]:
            flatten_size = flatten_size * s
        return X.reshape((X.shape[0], flatten_size))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for m in self.modules:
            x = m(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        device = logits.device
        dtype = logits.dtype
        
        y_one_hot = init.one_hot(logits.shape[1], y, device=device, dtype=dtype)
        res = (ops.logsumexp(logits, (1,)).sum() - (y_one_hot * logits).sum())
        out = res / logits.shape[0]
        return out
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(self.dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(self.dim, device=device, dtype=dtype, requires_grad=True))
        self.running_mean = init.zeros(self.dim, device=device, dtype=dtype)
        self.running_var = init.ones(self.dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            mean = (x.sum((0,)) / x.shape[0]).reshape((1, x.shape[1])).broadcast_to(x.shape)
            var = (((x - mean) ** 2).sum((0,)) / x.shape[0]).reshape((1, x.shape[1])).broadcast_to(x.shape)
            norm_mean = mean.sum((0,)) / x.shape[0]
            norm_var = var.sum((0,)) / x.shape[0]
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * norm_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * norm_var
            norm_x = (x - mean) / ((var + self.eps) ** 0.5)
            return self.weight.broadcast_to(x.shape) * norm_x + self.bias.broadcast_to(x.shape)
        else:
            broad_mean = self.running_mean.reshape((1, x.shape[1])).broadcast_to(x.shape)
            broad_var = self.running_var.reshape((1, x.shape[1])).broadcast_to(x.shape)
            norm_x = (x - broad_mean) / ((broad_var + self.eps) ** 0.5)
            return self.weight.broadcast_to(x.shape) * norm_x + self.bias.broadcast_to(x.shape)
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(self.dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(self.dim, device=device, dtype=dtype, requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        mean_x = (x.sum((1,)) / x.shape[1]).reshape((x.shape[0], 1)).broadcast_to(x.shape)
        var_x = ((x - mean_x)**2).sum((1,)) / x.shape[1]
        deno_x = (var_x + self.eps) ** 0.5
        # with reshape: broadcast for each col
        deno_x = deno_x.reshape((x.shape[0], 1)).broadcast_to(x.shape)
        norm_x = (x - mean_x) / deno_x
        # layer norm
        return self.weight.broadcast_to(x.shape) * norm_x + self.bias.broadcast_to(x.shape)
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            mask = init.rand_binomial(1, 1 - self.p,  x.shape)
            
            mask = mask.reshape(x.shape)
            return mask * x / (1 - self.p)
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION
