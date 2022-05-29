import numpy as np

from ...tensor import Tensor

__all__ = ['add_', 'sub_', 'mul_', 'div_', 'pow_', 'reshape_']


def add_(grad: Tensor, **_):
    return [grad, grad]


def sub_(grad: Tensor, **_) -> list[Tensor]:
    return [grad, -grad]


def mul_(grad: Tensor, x: list, **_) -> list[Tensor]:
    return [grad * x[1], grad * x[0]]


def div_(grad: Tensor, x: list, **_) -> list[Tensor]:
    return [grad / x[1], -grad * x[0] / x[1] ** 2]


def pow_(grad: Tensor, x: list, **_) -> list[Tensor]:
    return [grad * x[1] * x[0] ** (x[1] - 1),
            grad * x[0] ** x[1] * np.log(x[0])]


def reshape_(grad: Tensor, x: Tensor, **_) -> list[Tensor]:
    """ reshape后反向传播时梯度也需要reshape回去

    :param grad: (batch size, *shape)
    :param x: (batch size, *shape)
    :return:
    """
    assert grad.shape[0] == x.shape[0]
    return [grad.reshape(*x.shape)]
