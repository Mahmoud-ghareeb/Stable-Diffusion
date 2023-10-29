import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, units):
        super().__init__()

        self.q = nn.Linear(units, units)
        self.k = nn.Linear(units, units)
        self.v = nn.Linear(units, units)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        input_shape = x.shape
        b, c, h, w = input_shape

        x = x.reshape(shape=(b, c, w*h))

        Q = self.q(x).view()
        K = self.k(x)
        V = self.v(x)
