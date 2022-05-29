import numpy as np

from ...tensor import Tensor
from ...utils.images import add_padding

__all__ = ['convolution2d_param', 'linear_param']


def convolution2d_param(x: Tensor, w: Tensor, b: Tensor, grad_in: Tensor, stride=1, **_) -> dict[str, Tensor]:
    from ...nn import functional as F
    """ 参考教程： https://microsoft.github.io/ai-edu/%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8B/A2-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86/%E7%AC%AC8%E6%AD%A5%20-%20%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/17.3-%E5%8D%B7%E7%A7%AF%E7%9A%84%E5%8F%8D%E5%90%91%E4%BC%A0%E6%92%AD%E5%8E%9F%E7%90%86.html#1735

    :param b: bias, shape (num,)
    :param w: kernel, with shape (num, channel, height, width), and sizes are odd
    :param stride:
    :param x: input, with shape (batch size, channel, height, width)
    :param grad_in: d_loss/d_layer, with shape (batch size, channel, y_height, y_width)
    :return: d_loss/d_w, d_loss/d_b, with shape w(num, channel, height, width), b(num, )
    """
    assert x.dim() == 4 and grad_in.dim() == 4 and w.dim() == 4
    x = add_padding(x, (w.shape[2] // 2, w.shape[3] // 2))
    # 参照教程，input_grad对input卷积后的结果就是d_loss/d_w
    w_grad = Tensor.new(*w.shape)
    for batch in grad_in:  # 顺序取每一个batch，结果扩展为(num, channel, height, width)
        batch = Tensor(batch, requires_grad=True)
        batch = batch.middleware(np.stack, (batch, batch, batch))
        grad_ = F.convolution2d(x, batch, Tensor(0), stride=stride)
        grad_ = grad_.sum(axis=0)
        w_grad += grad_
    b_grad = grad_in.sum(axis=(1, 2, 3))  # 参照教程，d_b=sum(d_in)
    return {'w': w_grad, 'b': b_grad}


def linear_param(x: Tensor, grad: Tensor, **_) -> dict[str, Tensor]:
    """ 参考教程：https://blog.csdn.net/GoodShot/article/details/79330545 (用于核对公式)

    :param x: input, with shape (batch_size, in_channel)
    :param grad: d_loss/d_layer+1, with shape (b, out_channel)
    :return: d_loss/d_w with shape (out_channel, in_channel), d_loss/d_b, with shape (out_channel)
    """
    assert x.dim() == 2 and grad.dim() == 2
    w_grad = grad.T @ x
    b_grad = grad.sum(axis=0)
    return {'w': w_grad, 'b': b_grad}
