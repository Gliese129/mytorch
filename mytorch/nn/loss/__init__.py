from ...nn import functional as F
from ...tensor import Tensor

__all__ = ['CrossEntropyLoss']


class Loss:
    def loss(self, y_hat, y):
        raise NotImplementedError

    def __call__(self, y_hat, y):
        res = self.loss(y_hat, y)
        res = Tensor(res, requires_grad=True)
        res.is_loss = True
        res.set_front([y_hat])
        return res


class CrossEntropyLoss(Loss):
    def loss(self, y_hat, y):
        return F.cross_entropy(y_hat, y)
