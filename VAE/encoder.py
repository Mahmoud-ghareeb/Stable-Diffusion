import torch
import torch.nn as nn
from torch.nn import functional as F

from attention import SelfAttention
from decoder import ResidualBlock


class Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # x:(b, 3, w, h) => (b, 128, w, h)
            self.conv_block(3, 128, 128, 128, 1, 1),
            # x:(b, 128, w, h) => (b, 256, w/2, h/2)
            self.conv_block(128, 128, 128, 256, 2, 0),
            # x:(b, 256, w/2, h/2) => (b, 512, w/4, h/4)
            self.conv_block(256, 256, 256, 512, 2, 0),
            # x:(b, 512, w/4, h/4) => (b, 512, w/8, h/8)
            self.conv_block(512, 512, 512, 512, 2, 0),

            ResidualBlock(512, 512),
            SelfAttention(512),
            ResidualBlock(512, 512),
            nn.GroupNorm(32, 512),
            nn.SiLU(),

            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=1)
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)

        mean, log_var = torch.chunk(x, 2, dim=1)
        log_var = torch.clamp(x, -30, 20)
        var = log_var.exp()
        stdev = var.sqrt()

        # noise => N(0, 1)
        # x => N(mean, var)
        # to convert from N(0,1) to N(mean, var)
        x = mean + stdev * noise
        # exist in the paper no explanation :(
        x *= 0.18215

        return x

    def conv_block(self, conv_in: int, conv_out: int, res_in: int, res_out: int, stride: int, padding: int):
        return nn.Sequential([
            nn.Conv2d(conv_in, conv_out, kernel_size=3,
                      stride=stride, padding=padding),
            ResidualBlock(res_in, res_out),
            ResidualBlock(res_out, res_out)
        ])
