import numpy as np

from ...tensor import Tensor
from ...utils.basic import one_hot

__all__ = ['cross_entropy']


def cross_entropy(y_hat: Tensor, y: Tensor) -> Tensor:
    """ Cross entropy loss

    :param y_hat: prediction, with shape (batch size, class num)
    :param y: ground truth, with shape (batch size, class num) or (batch size, )
    :return: loss
    """
    if y.dim() == 1:
        y = one_hot(y, y_hat.shape[-1])
    assert y_hat.shape == y.shape
    return y_hat.middleware(np.where, y_hat > 0, -y * y_hat * np.log(y_hat), 0)


