from typing import Union, Tuple
from ...utils.basic import to_pair
from ...utils.images import add_padding
from ...utils.images import pool_relocate
from ...tensor import Tensor

__all__ = ['convolution2d_', 'max_pool2d_', 'linear_']


def max_pool2d_(x: Tensor, y: Tensor, grad: Tensor, pos: Tensor,
                stride: Union[int, Tuple[int, int]], padding: Union[int, Tuple[int, int]], **_) -> list[Tensor]:
    """

    :param stride:
    :param padding:
    :param x: input data (batch_size, channel, height, width)
    :param y: result after max_pool2d (batch_size, channel, height, width)
    :param grad: d_loss/d_pre_layer (batch_size, channel, y_height, y_width)
    :param pos: position of max_pool2d (batch_size, channel, y_height, y_width)
    :return: d_loss/d_layer (batch_size, channel, x_height, x_width)
    """
    stride = to_pair(stride)
    padding = to_pair(padding)
    res = Tensor.new(*x.shape)
    for b in range(y.shape[0]):
        for c in range(y.shape[1]):
            for h in range(y.shape[2]):
                for w in range(y.shape[3]):
                    x_h, x_w = pool_relocate(h, w, stride, pos[b, c, h, w])
                    res[b, c, x_h, x_w] = grad[b, c, h, w]
    # 去掉padding
    res = res[:, :, padding[0]:res.shape[2] - padding[0], padding[1]:res.shape[3] - padding[1]]
    return [res]


def linear_(x: Tensor, y: Tensor, grad: Tensor, w: Tensor, **_) -> list[Tensor]:
    """

    :param x: input data (batch_size, x_dim)
    :param y: result after linear (batch_size, y_dim)
    :param grad: d_loss/d_pre_layer (batch_size, y_dim)
    :param w: weight (out_channel, in_channel)
    :return: d_loss/d_layer (batch_size, in_channel)
    """
    res = Tensor.new(*x.shape, requires_grad=x.requires_grad)
    for b in range(y.shape[0]):
        res[b] = w.T @ grad[b]
    return [res]


def convolution2d_(x: Tensor, grad: Tensor, kernel: Tensor,
                   stride: Union[int, Tuple[int, int]] = 1,
                   padding:  Union[int, Tuple[int, int]] = 0, **_) -> list[Tensor]:
    """ 参考教程：https://microsoft.github.io/ai-edu/%E5%9F%BA%E7%A1%80%E6%95%99%E7%A8%8B/A2-%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E6%9C%AC%E5%8E%9F%E7%90%86/%E7%AC%AC8%E6%AD%A5%20-%20%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/17.3-%E5%8D%B7%E7%A7%AF%E7%9A%84%E5%8F%8D%E5%90%91%E4%BC%A0%E6%92%AD%E5%8E%9F%E7%90%86.html
        p.s. 卷积这玩意儿真的算是所有模型里面最难了的，4个维度的卷积运算，想找一个简洁的矩阵乘法公式几乎不可能，所以就只能用5重循环代替了

    :param padding:
    :param x: input data (batch_size, channel, height, width)
    :param grad: d_loss/d_pre_layer (batch_size, channel, y_height, y_width)
    :param kernel: weight (num, kernel_channel, kernel_height, kernel_width)
    :param stride: stride
    :return: d_loss/d_layer (batch_size, channel, x_height, x_width)
    """
    grad = add_padding(grad, (kernel.shape[2] - 1, kernel.shape[3] - 1))
    dense_res = Tensor.new(grad.shape[0], grad.shape[1],
                           grad.shape[2] - kernel.shape[2] + 1, grad.shape[3] - kernel.shape[3] + 1)
    kernel_ = Tensor.rotate180(kernel)
    stride = to_pair(stride)
    padding = to_pair(padding)
    # 反向传播，按照in的每个点分步计算
    for b in range(x.shape[0]):
        for c in range(x.shape[1]):
            for h in range(x.shape[2]):
                for w in range(x.shape[3]):
                    for n in range(kernel.shape[0]):
                        # 在添加了padding后，grad对kernel_的卷积矩阵就是当前通道的梯度
                        dense_res[b, c, h, w] += \
                            (kernel_[n, c] * grad[b, n, h:h + kernel.shape[2], w:w + kernel.shape[3]]).sum()
    # 为解决stride不是1的情况，需要把res重新池化，此时结果形状为添加padding后的输入形状
    res = Tensor.new(x.shape[0], x.shape[1], x.shape[2] + padding[0] * 2, x.shape[3] + padding[1] * 2)
    for b in range(dense_res.shape[0]):
        for c in range(dense_res.shape[1]):
            for h in range(dense_res.shape[2]):
                for w in range(dense_res.shape[3]):
                    res[b, c, h * stride[0], w * stride[1]] = dense_res[b, c, h, w]
    # 去掉padding
    res = res[:, :, padding[0]:res.shape[2] - padding[0], padding[1]:res.shape[3] - padding[1]]
    return [res]
