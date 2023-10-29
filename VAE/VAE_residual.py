import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_units, out_units):
        super().__init__()

        self.gn1 = nn.GroupNorm(32, in_units)
        self.conv1 = nn.Conv2d(in_units, out_units, kernel_size=3, padding=1)

        self.gn2 = nn.GroupNorm(32, out_units)
        self.conv2 = nn.Conv2d(out_units, out_units, kernel_size=3, padding=1)

        if in_units == out_units:
            self.res = nn.Identity()
        else:
            self.res = nn.Conv2d(in_units, out_units, kernel_size=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        res = self.res(x)

        x = self.gn1(x)
        x = F.silu(x)
        x = self.conv1(x)
        x = self.gn2(x)
        x = F.silu(x)
        x = self.conv2(x)

        x += res

        return x
