from typing import Optional

from .backward import *
from .optimize import *
from ..tensor import Tensor

str2backward = {
    'add': add_, 'subtract': sub_, 'multiply': mul_, 'true_divide': div_, 'power': pow_,
    'leaky_relu': leaky_relu_, 'relu': relu_, 'sigmoid': sigmoid_, 'softmax': softmax_,
    'convolution2d': convolution2d_, 'max_pool2d': max_pool2d_, 'linear': linear_,
    'reshape': reshape_
}
str2optimize = {
    'convolution2d': convolution2d_param, 'linear': linear_param
}


class AutoGrad:
    func: callable
    optim: Optional[callable]
    inputs: list[Tensor]
    output: Tensor
    params: dict  # 模型参数
    func_params: dict  # 调用函数时可能会用到的参数
    grad_in: Optional[Tensor]  # 反向传播后的结果

    def __init__(self, func, inputs, output, params=None, **func_params):
        if params is None:
            params = {}
        self.grad_in = None
        self.func = str2backward[func]
        self.optim = str2optimize.get(func, None)
        self.inputs = inputs
        self.output = output
        self.params = params
        self.func_params = func_params

    def __call__(self, grad: Tensor):
        return self.grad(grad)

    def param_grad(self, grad: Tensor):
        if hasattr(self, 'optim') and self.optim is not None:
            grad_in = self.grad_in if hasattr(self, 'grad_in') else None
            grads = self.optim(grad=grad, x=self.inputs, y=self.output, grad_in=grad_in,
                               **self.params, **self.func_params)
            for key in grads.keys():
                self.params[key].grad = grads[key]

    def grad(self, grad: Tensor):
        res = self.func(grad=grad, x=self.inputs, y=self.output, **self.params, **self.func_params)
        if isinstance(res, list):
            for g in res:
                if isinstance(g, Tensor):
                    self.grad_in = g
                    break
        return res

    def __str__(self):
        return f'AutoGrad<{self.func.__name__.replace("_", "")}>'
