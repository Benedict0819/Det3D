from ..registry import BACKBONES, READERS
import torch
from torch import nn
import numpy as np

@READERS.register_module
class FeatureNormalizer(nn.Module):
    def __init__(self, mean, std, dim, axis=1, name=None, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.mean = np.array(mean)
        self.std = np.array(std)
        self.axis = axis
        self.view_shape = [1] * dim
        self.view_shape[axis] = -1

    def forward(self, x):

        mean_tensor = torch.tensor(self.mean, dtype=x.dtype, device=x.device).view(self.view_shape)
        std_tensor = torch.tensor(self.std, dtype=x.dtype, device=x.device).view(self.view_shape)
        x = (x - mean_tensor) / std_tensor
        return x