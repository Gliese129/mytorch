from typing import Union

import numpy as np

CALCULATE_FUNCTIONS = ['add', 'subtract', 'multiply', 'true_divide', 'power']


class Tensor(np.ndarray):
    """ a tensor class using numpy

    """

    is_leaf: bool
    requires_grad: bool
    grad_fn: callable
    grad: 'Tensor'
    _front: list['Tensor']
    is_loss: bool

    def __new__(cls, data, requires_grad=False):
        if isinstance(data, cls):
            if requires_grad:
                data._requires_grad(True)
            return data
        obj = np.array(data).view(cls)
        obj.requires_grad = requires_grad
        obj.is_loss = False
        if requires_grad:
            if obj.dtype not in (np.float32, np.float64):
                raise ValueError('Tensor requires_grad only support float dtype')
            obj.grad_fn = None
            obj.is_leaf = True
        else:
            obj.is_leaf = False
        obj._front = []
        return obj

    @classmethod
    def new(cls, *shape, fill=0., requires_grad=False):
        res = np.full(shape, fill_value=fill)
        return Tensor(res, requires_grad=requires_grad)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if ufunc.__name__ in CALCULATE_FUNCTIONS:  # 如果是运算的话，将参与运算的数据全部处理成Tensor
            inputs = [Tensor(input_) for input_ in inputs]

        args = []
        requires_grad = False
        for input_ in inputs:
            if isinstance(input_, Tensor):  # 只要有一个Tensor要求求导，则结果的requires_grad也会为True
                args.append(input_.view(np.ndarray))
                if hasattr(input_, 'requires_grad') and input_.requires_grad:
                    requires_grad = True
            else:
                args.append(input_)

        kwargs.pop('out', None)
        res = getattr(ufunc, method)(*args, **kwargs)
        res = Tensor(res)
        res._front = [input_ for input_ in inputs if isinstance(input_, Tensor)]  # 记录下前置Tensor，方便backward更新
        res._requires_grad(requires_grad)

        if ufunc.__name__ in CALCULATE_FUNCTIONS:
            from .autograd import AutoGrad
            res.grad_fn = AutoGrad(ufunc.__name__, inputs, res, **kwargs)
        return res

    def _requires_grad(self, requires_grad=True):
        self.requires_grad = requires_grad
        self.is_leaf = False

    def __str__(self):
        data_str = super().__str__()
        if self.requires_grad:
            return f'Tensor({data_str}, requires_grad=True, ' \
                   f'grad_fn={self.grad_fn if hasattr(self, "grad_fn") else None})'
        return f'Tensor({data_str})'

    def middleware(self, func: callable, *args, **kwargs) -> 'Tensor':
        """ apply a function using self, and make the result a Tensor
        remind that the result won't implement grad_fn automatically

        :param func:
        :param args:
        :param kwargs:
        :return:
        """
        res = func(*args, **kwargs)
        res = Tensor(res)
        res._requires_grad(self.requires_grad)
        res._front = [self]
        return res

    def dim(self) -> int:
        return len(self.shape)

    @staticmethod
    def rotate180(x):
        y = Tensor.new(*x.shape, requires_grad=x.requires_grad)
        y.dtype = x.dtype
        y[:, :, ::-1, ::-1] = x
        return y

    def backward(self, grad: 'Tensor' = None):
        from .autograd import AutoGrad

        if hasattr(self, 'is_loss') and self.is_loss:
            self.grad = Tensor(self)
            for front in self._front:
                front.backward(self.grad)  # loss直接传播到前一层
        else:
            self.grad = grad
            if hasattr(self, 'grad_fn') and isinstance(self.grad_fn, AutoGrad):  # 如果没有，则代表到了叶子节点
                front_grad = self.grad_fn(self.grad)
                self.grad_fn.param_grad(self.grad)
                for idx, front in enumerate(self._front):
                    front.backward(front_grad[idx])  # 将链式反向传播

    def set_front(self, front: Union[list['Tensor'], 'Tensor']):
        if isinstance(front, Tensor):
            self._front = [front]
        else:
            self._front = front

    def reshape(self, *shape, order='C'):
        from .autograd import AutoGrad

        res = super().reshape(shape, order=order)
        res = Tensor(res, requires_grad=self.requires_grad)
        res.set_front(self)
        res._requires_grad(self.requires_grad)
        res.grad_fn = AutoGrad('reshape', inputs=self, output=res)
        return res
