import torch
import torch.nn as nn
from torch.nn import functional as F

from StableDiffusion.VAE.VAE_attention import VAE_Attention
from StableDiffusion.VAE.VAE_residual import ResidualBlock


class ConvBlock(nn.Module):
    def __init__(self, conv_in: int, conv_out: int, res_in: int, res_out: int, stride: int, padding: int):
        super().__init__()

        self.conv = nn.Conv2d(conv_in, conv_out, kernel_size=3,
                              stride=stride, padding=padding)
        self.res1 = ResidualBlock(res_in, res_out)
        self.res2 = ResidualBlock(res_out, res_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.conv(x)
        x = self.res1(x)
        x = self.res2(x)

        return x


class Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # x:(b, 3, h, w) => (b, 128, h, w)
            ConvBlock(3, 128, 128, 128, 1, 1),
            # x:(b, 128, h, w) => (b, 256, h/2, w/2)
            ConvBlock(128, 128, 128, 256, 2, 0),
            # x:(b, 256, h/2, w/2) => (b, 512, h/4, w/4)
            ConvBlock(256, 256, 256, 512, 2, 0),
            # x:(b, 512, h/4, w/4) => (b, 512, h/8, w/8)
            ConvBlock(512, 512, 512, 512, 2, 0),

            ResidualBlock(512, 512),
            VAE_Attention(512),
            ResidualBlock(512, 512),
            nn.GroupNorm(32, 512),
            nn.SiLU(),

            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=1, padding=0)
        )

    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)

        mean, log_var = torch.chunk(x, 2, dim=1)
        log_var = torch.clamp(log_var, -30, 20)
        var = log_var.exp()
        stdev = var.sqrt()

        # noise => N(0, 1)
        # x => N(mean, var)
        # to convert from N(0,1) to N(mean, var)
        x = mean + stdev * noise
        # exist in the paper no explanation :(
        x *= 0.18215

        return x
