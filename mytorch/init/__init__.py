import numpy as np

from ..nn import Module
from ..tensor import Tensor

__all__ = ['init_func', 'Initializer']


class init_func:
    @staticmethod
    def normal(param: Tensor, loc: float = 0., scale: float = 1.):
        # [...]保证只修改值
        param[...] = np.random.normal(loc=loc, scale=scale, size=param.shape)


class Initializer:
    init_params: dict
    func: callable

    def __init__(self, func=init_func.normal, **kwargs):
        self.init_params = kwargs
        self.func = func

    def init(self, module: Module):
        # 递归遍历所有子模块
        if module.modules and len(module.modules) > 0:
            for m in module.modules:
                self.init(m)
        for param in module.params.values():
            self.func(param, **self.init_params)

    def __call__(self, *args, **kwargs):
        self.init(*args, **kwargs)
