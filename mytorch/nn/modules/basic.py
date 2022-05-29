from collections import OrderedDict

from ...tensor import Tensor


class Module:
    input: Tensor
    output: Tensor
    params: dict[str, Tensor]
    __name__ = 'Basic'

    def __init__(self, name: str = None):
        self.params = OrderedDict()
        if name is not None:
            self.__name__ = name

    def forward(self, *inputs, **kwargs):
        raise NotImplementedError

    @property
    def modules(self):
        layers = []
        for item in self.__dict__.values():
            if isinstance(item, Module):
                layers.append(item)
        return layers

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)

    def __str__(self):
        return f'Module<{self.__name__}, params: {self.params.keys()}>'


class Sequential(Module):
    def __init__(self, *layers: Module):
        super().__init__()
        self.layers = layers

    def forward(self, x: Tensor) -> Tensor:
        self.input = x
        for layer in self.layers:
            x = layer(x)
        self.output = x
        return x

    def __str__(self):
        layers = [str(layer) for layer in self.layers]
        layers = '\n'.join(layers)
        return f'Sequential(\n{layers}\n)'
