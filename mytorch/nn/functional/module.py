import math
from typing import Union, Tuple
from mytorch.utils.basic import to_pair
from mytorch.utils.images import pool_relocate
from mytorch.tensor import Tensor

__all__ = ['convolution2d', 'max_pool2d', 'linear']


def convolution2d(img: Tensor, kernel: Tensor, bias: Tensor, stride=1) -> Tensor:
    """ 2d kernel convolution

    :param img: batches of images, with shape (batch_size, channel, height, width)
    :param kernel: kernel with shape (num, channel, height, width), and sizes are odd
    :param bias: bias with shape (num,)
    :param stride: stride size, int or (row stride, col stride)
    :return: output, with shape (batch_size, channel, height, width)
    """
    assert img.dim() == 4
    assert type(stride) == int
    assert kernel.dim() == 4
    assert kernel.shape[1] == img.shape[1]

    stride = to_pair(stride)

    h_len = math.floor((img.shape[2] - kernel.shape[2] + 1) / stride[0])
    w_len = math.floor((img.shape[3] - kernel.shape[3] + 1) / stride[1])

    result = Tensor.new(img.shape[0], kernel.shape[0], h_len, w_len, requires_grad=img.requires_grad)
    for idx in range(kernel.shape[0]):
        for i in range(h_len):
            for j in range(w_len):
                h_range = (i * stride[0], i * stride[0] + kernel.shape[2])
                w_range = (j * stride[1], j * stride[1] + kernel.shape[3])
                result[:, idx, i, j] = \
                    (img[:, :, h_range[0]:h_range[1], w_range[0]:w_range[1]] * kernel[idx]).sum(axis=(1, 2, 3))
    bias = bias.reshape(1, -1, 1, 1)
    return result + bias


def max_pool2d(img: Tensor, stride: Union[int, Tuple[int, int]]) -> Tuple[Tensor, Tensor]:
    """ 2d max pool

    :param img: batches of images, with shape (batch_size, channel, height, width)
    :param stride: strides, which should be int or (int, int)
        remind that pool size is the same as stride size
    :return: images after max pooling and the position of the max points
    """
    assert img.dim() == 4
    stride = to_pair(stride)

    h_len = math.floor(img.shape[2] / stride[0])
    w_len = math.floor(img.shape[3] / stride[1])
    result = Tensor.new(img.shape[0], img.shape[1], h_len, w_len, requires_grad=img.requires_grad)
    position = Tensor.new(img.shape[0], img.shape[1], h_len, w_len)
    for i in range(img.shape[0]):
        for c in range(img.shape[1]):
            for h in range(h_len):
                for w in range(w_len):
                    position[i, c, h, w] = img[i, c,
                                               h * stride[0]:h * stride[0] + stride[0],
                                               w * stride[1]:w * stride[1] + stride[1]].argmax()
                    origin_pos = pool_relocate(h, w, stride, position[i, c, h, w])
                    result[i, c, h, w] = img[i, c, origin_pos[0], origin_pos[1]]
    return result, position


def linear(w: Tensor, b: Tensor, x: Tensor) -> Tensor:
    """ linear function

    :param w: weights, with shape (out_channel, in_channel)
    :param b: bias, with shape (out_channel,)
    :param x: input, with shape (batch_size, in_channel)
    :return: output, with shape (batch_size, out_channel)
    """
    assert w.dim() == 2 and b.dim() == 1 and x.dim() == 2
    assert w.shape[1] == x.shape[1]
    assert w.shape[0] == b.shape[0]
    return (w @ x.T).T + b
