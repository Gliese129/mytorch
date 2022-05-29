from ..nn import Module
from ..tensor import Tensor


class Optimizer:
    parameters: list[Tensor]
    grads: list[Tensor]
    lr: float

    def __init__(self, net: Module, lr=0.01):
        self.parameters = []
        self.grads = []
        self.lr = lr
        self._get_all_params(net)

    def _get_all_params(self, module: Module):
        # 递归遍历所有子模块
        if module.modules and len(module.modules) > 0:
            for m in module.modules:
                self._get_all_params(m)
        # 直接加入param即可，grad可通过param.grad获取
        for param in module.params.values():
            self.parameters.append(param)
            self.grads.append(Tensor(0))

    def zero_grad(self):
        for grad in self.grads:
            grad[...] = 0.

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):

    def step(self):
        for param, grad in zip(self.parameters, self.grads):
            grad += param.grad
            param -= self.lr * grad

