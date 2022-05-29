from typing import Tuple, Union

from ..modules import Module
from ...autograd import AutoGrad
from ...tensor import Tensor
from ...utils.basic import to_pair
from .. import functional as F
from ...utils.images import add_padding

__all__ = ['Conv2d', 'MaxPool2d', 'Linear']


class Conv2d(Module):
    weight: Tensor
    bias: Tensor
    stride: Union[int, Tuple[int, int]]
    padding: Union[int, Tuple[int, int]]
    in_channels = property(lambda self: self.weight.shape[1])
    out_channels = property(lambda self: self.weight.shape[0])

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 padding: Union[int, Tuple[int, int]] = 0, bias=True):
        super().__init__()
        kernel_size = to_pair(kernel_size)
        self.stride = stride
        self.padding = padding
        #  初始化成0会导致网络梯度优化的时候寄掉
        self.weight = Tensor.new(out_channels, in_channels, kernel_size[0], kernel_size[1], requires_grad=True, fill=1.)
        if bias:
            self.bias = Tensor.new(out_channels, requires_grad=True)
        self.params.update({'w': self.weight, 'b': self.bias})

    def forward(self, x: Tensor) -> Tensor:
        self.input = x
        y = x.middleware(x.copy)
        y = add_padding(y, self.padding)
        y = F.convolution2d(y, self.weight, self.bias, self.stride)
        if self.bias is not None:
            bias = self.bias.reshape(1, -1, 1, 1)
            y += bias
        if y.requires_grad:
            #  直接将y的grad_fn输出指向x
            y.grad_fn = AutoGrad(func='convolution2d', inputs=x, output=y, kernel=self.weight,
                                 stride=self.stride, padding=self.padding,
                                 params=self.params)
            y.set_front([x])
        self.output = y
        y.debug = 'conv2d'
        return y


class MaxPool2d(Module):
    """
    注意：stride和kernel是相同大小的，所以不需要指定kernel_size

    """
    stride: Union[int, Tuple[int, int]]
    padding: Union[int, Tuple[int, int]]

    def __init__(self, stride: Union[int, Tuple[int, int]] = None,
                 padding: Union[int, Tuple[int, int]] = 0):
        super().__init__()
        self.stride = to_pair(stride)
        self.padding = to_pair(padding)

    def forward(self, x: Tensor) -> Tensor:
        self.input = x
        y = x.middleware(x.copy)
        y = add_padding(y, self.padding)
        y, pos = F.max_pool2d(y, self.stride)
        if y.requires_grad:
            y.grad_fn = AutoGrad(func='max_pool2d', inputs=x, output=y, pos=pos,
                                 stride=self.stride, padding=self.padding)
            y.set_front([x])
        self.output = y
        y.debug = 'max_pool2d'
        return y


class Linear(Module):
    weight: Tensor
    bias: Tensor

    def __init__(self, in_features: int, out_features: int, bias=True):
        super().__init__()
        #  初始化成0会导致网络梯度优化的时候寄掉
        self.weight = Tensor.new(out_features, in_features, requires_grad=True, fill=1.)
        if bias:
            self.bias = Tensor.new(out_features, requires_grad=True)
        self.params.update({'w': self.weight, 'b': self.bias})

    def forward(self, x: Tensor) -> Tensor:
        self.input = x
        y = x.middleware(x.copy)
        y = F.linear(self.weight, self.bias, y)
        if y.requires_grad:
            y.grad_fn = AutoGrad(func='linear', inputs=x, output=y, params=self.params)
            y.set_front([x])
        self.output = y
        y.debug = 'linear'
        return y
