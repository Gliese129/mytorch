import numpy as np
from mytorch.tensor import Tensor

__all__ = ['leaky_relu', 'relu', 'sigmoid', 'softmax']


def leaky_relu(x: Tensor, alpha=0.2) -> Tensor:
    return x.middleware(np.where, x > 0, x, x * alpha)


def relu(x: Tensor) -> Tensor:
    return x.middleware(np.where, x > 0, x, 0)


def sigmoid(x: Tensor) -> Tensor:
    return 1 / (1 + x.middleware(np.exp, -x))


def softmax(x: Tensor, dim=1) -> Tensor:
    # 为防止e^x溢出，先对dim上的x做归一化
    tot = np.sum(x, axis=dim, keepdims=True)
    x /= tot
    x = x.middleware(np.exp, x)
    return x / x.middleware(np.sum, x, axis=dim, keepdims=True)
