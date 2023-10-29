import torch
import torch.nn as nn
from attention import SelfAttention


class SelfAttention(nn.Module):
    def __init__(self, units: int):
        super().__init__()

        self.gn = nn.GroupNorm(32, units)
        self.attention = SelfAttention(units)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        res = x
        b, c, h, w = x.shape
        # x:(b, c, h, w) => (b, h*w, c)
        x = x.view(b, c, h*w).transpose(-1, -2)
        x = self.attention(x)
        x = x.transpose(-1, -2).view(b, c, h, w)
        x += res

        return x
