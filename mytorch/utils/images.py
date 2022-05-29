from typing import Tuple, Union
from .basic import to_pair
from ..tensor import Tensor


def pool_relocate(row: int, col: int, pool_size: Tuple[int, tuple[int, int]], pos: int) -> (int, int):
    """ 定位池化处理后点在原图中的位置

    :param row: current row
    :param col: current col
    :param pool_size:
    :param pos: argmax position of the point
    :return: real position in origin images
    """
    pool_size = to_pair(pool_size)
    pos = (pos // pool_size[0], pos % pool_size[1])
    return int(row * pool_size[0] + pos[0]), int(col * pool_size[1] + pos[1])


def add_padding(img: Tensor, padding: Union[int, Tuple[int, int]]) -> Tensor:
    """ add padding to images

    :param img: batches of images, with shape (batch_size, channel, height, width)
    :param padding: padding size
    :return: images after padding
    """
    if padding in [0, (0, 0)]:
        return img
    assert img.dim() == 4
    padding = to_pair(padding)
    result = Tensor.new(img.shape[0], img.shape[1], img.shape[2] + padding[0] * 2, img.shape[3] + padding[1] * 2)
    result[:, :, padding[0]:result.shape[2] - padding[0], padding[1]:result.shape[3] - padding[1]] = img
    return result
