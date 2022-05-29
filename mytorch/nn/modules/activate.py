from .basic import Module
from ...autograd import AutoGrad
from ...tensor import Tensor
from .. import functional as F

__all__ = ['LeakyRelu', 'Relu', 'Sigmoid', 'Softmax']


class LeakyRelu(Module):
    alpha: float

    def __init__(self, alpha: float = 0.01):
        super().__init__('LeakyRelu')
        self.alpha = alpha

    def forward(self, x: Tensor) -> Tensor:
        y = x.middleware(x.copy)
        y = F.leaky_relu(y, self.alpha)
        if y.requires_grad:
            y.grad_fn = AutoGrad(func='leaky_relu', inputs=x, output=y, alpha=self.alpha)
            y.set_front([x])
        return y


class Relu(Module):
    def __init__(self):
        super().__init__('Relu')

    def forward(self, x: Tensor) -> Tensor:
        y = x.middleware(x.copy)
        y = F.relu(y)
        if y.requires_grad:
            y.grad_fn = AutoGrad(func='relu', inputs=x, output=y)
            y.set_front([x])
        return y


class Sigmoid(Module):
    def __init__(self):
        super().__init__('Sigmoid')

    def forward(self, x: Tensor) -> Tensor:
        y = x.middleware(x.copy)
        y = F.sigmoid(y)
        if y.requires_grad:
            y.grad_fn = AutoGrad(func='sigmoid', inputs=x, output=y)
            y.set_front([x])
        return y


class Softmax(Module):
    def forward(self, x: Tensor) -> Tensor:
        y = x.middleware(x.copy)
        y = F.softmax(y)
        if y.requires_grad:
            y.grad_fn = AutoGrad(func='softmax', inputs=x, output=y)
            y.set_front([x])
        return y
