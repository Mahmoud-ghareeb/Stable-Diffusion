import torch
import torch.nn as nn
from torch.nn import functional as F

from StableDiffusion.attention import SelfAttention

from dataclasses import dataclass


@dataclass
class Args:
    n_layers: int = 12
    d_model: int = 768
    vocab_size: int = 49408
    seq_len: int = 77
    n_heads: int = 12


class CLIPEmbedding(nn.Module):
    def __init__(self, args: Args) -> None:
        super().__init__()

        self.embd = nn.Embedding(args.vocab_size, args.d_model)
        self.pos_embd = nn.Parameter(torch.zeros(args.seq_len, args.d_model))

    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        x = self.embd(x)

        return x + self.pos_embd


class CLIPLayer(nn.Module):
    def __init__(self, args: Args) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(args.d_model)
        self.atten = SelfAttention(args.n_heads, args.d_model)
        self.norm2 = nn.LayerNorm(args.d_model)
        self.linear1 = nn.Linear(args.d_model, 4*args.d_model)
        self.linear2 = nn.Linear(4*args.d_model, args.d_model)

    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:

        x = self.norm1(x)
        x = self.atten(x)
        x = self.norm2(x)
        x = self.linear1(x)
        x = self.linear2(x)

        return x


class CLIP(nn.Module):
    def __init__(self, args: Args) -> None:
        super().__init__()

        self.embd = CLIPEmbedding(args)

        self.layers = nn.ModuleList([
            CLIPLayer(args) for _ in range(args.n_layers)
        ])

        self.norm = nn.LayerNorm(args.d_model)

    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:

        x = self.embd(x)
        for layer in self.layers:
            x = layer(x)

        return self.norm(x)
