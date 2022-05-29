import numpy as np

from ...tensor import Tensor

__all__ = ['leaky_relu_', 'relu_', 'sigmoid_', 'softmax_']


def leaky_relu_(grad: Tensor, alpha=0.2, **_) -> list[Tensor]:
    return [grad.middleware(np.where, grad > 0, grad, grad * alpha)]


def relu_(grad: Tensor, *_, **__) -> list[Tensor]:
    return [grad.middleware(np.where, grad > 0, grad, 0)]


def sigmoid_(grad: Tensor, y: Tensor, **_) -> list[Tensor]:
    return [grad * y * (1 - y)]


def softmax_(grad: Tensor, y: Tensor, **_) -> list[Tensor]:
    """

    :param grad: d_loss/d_pre_layer (batch_size, class nums)
    :param y: result after softmax (batch_size, class nums)
    :return: d_loss/d_layer (batch_size, class nums)
    """
    res = Tensor.new(*grad.shape)
    for i in range(grad.shape[1]):
        for j in range(grad.shape[1]):
            if i == j:
                res[:, i] += y[:, j] * (1 - y[:, j])
            else:
                res[:, i] -= y[:, j] * y[:, i]
    return [res * grad]
