from typing import Union

from ..tensor import Tensor


def to_pair(x: Union[any, tuple[any, any]]) -> tuple:
    if type(x) == tuple:
        assert len(x) == 2
        assert isinstance(x[0], type(x[1]))
        return x
    return x, x


def one_hot(y: Tensor, class_num=None) -> Tensor:
    """ one hot

    :param class_num: class nums if none, use y.max() + 1
    :param y: shape (batch size, ), int
    :return:
    """
    assert y.dim() == 1 and y.dtype == int
    if class_num is None:
        class_num = y.max() + 1
    y_one_hot = Tensor.new(*y.shape, class_num, fill=.0)
    for b in range(y.shape[0]):
        y_one_hot[b, y[b]] = 1.0
    return y_one_hot

