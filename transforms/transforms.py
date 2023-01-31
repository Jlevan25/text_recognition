import math
from typing import Union, Tuple, List

import numpy as np
import torch
from PIL import Image
from torch import nn


class VerticalResize(nn.Module):
    def __init__(self, height: int, save_ratio: bool = True, interpolation=None):
        super().__init__()
        self.height = height
        self.interpolation = interpolation
        self.save_ratio = save_ratio

    def forward(self, image: Image):
        w, h = image.size
        width = (w * self.height) // h if self.save_ratio else w

        return image.resize(size=(width, self.height), resample=self.interpolation)


class HorizontalLimit(nn.Module):
    def __init__(self, min_: int, max_: int, interpolation=None):
        super().__init__()
        self.min_ = min_
        self.max_ = max_
        self.interpolation = interpolation

    def forward(self, image: Image):
        w, h = image.size
        if w > self.max_:
            image.resize(size=(self.max_, h), resample=self.interpolation)
        elif self.min_ > w:
            image.resize(size=(self.min_, h), resample=self.interpolation)

        return image


class RandomHorizontalCrop(nn.Module):
    def __init__(self, max_indent: Union[int, float]):
        if isinstance(max_indent, float) and max_indent > 1.:
            raise ValueError('float bigger than 1')
        super().__init__()
        self.max_indent = max_indent

    def forward(self, image: Image):
        w, h = image.size
        max_indent = (int(w * self.max_indent) if self.max_indent < 1 else self.max_indent) + 1
        if max_indent > 1:
            left_indent = np.random.randint(max_indent)
            right_indent = np.random.randint(max_indent)
            if w > left_indent + right_indent:
                crop_box = (left_indent, 0, w - right_indent, h)
                return image.crop(box=crop_box)
        return image


class RandomHorizontalResize(nn.Module):
    def __init__(self, min_scale: float = 0.08, max_scale: float = 1., interpolation=None):
        if min_scale > max_scale:
            raise ValueError("min_scale bigger than max_scale")

        if min_scale == 0:
            raise ValueError("min_scale can not be zero")

        super().__init__()
        self.min_scale = min_scale
        self.max_scale = max_scale
        self._not_equal_scales = min_scale != max_scale
        self.interpolation = interpolation

    def forward(self, image: Image):
        w, h = image.size
        scale = np.random.uniform(self.min_scale, self.max_scale) if self._not_equal_scales else self.min_scale
        return image.resize(size=(math.ceil(w * scale), h), resample=self.interpolation)


class GaussianNoise(nn.Module):
    # todo std< ?
    def __init__(self, mean: float = 0., std: float = .2):
        super().__init__()
        self.std = std
        self.mean = mean

    def forward(self, tensor):
        return tensor + torch.normal(self.mean, self.std, tensor.shape)

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'
