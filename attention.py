import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelfAttention(nn.Module):
    def __init__(self, n_head: int, d_model: int):
        super().__init__()

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)

        self.d_head = d_model//n_head
        self.n_head = n_head

    def forward(self, x: torch.Tensor, mask: bool = False) -> torch.Tensor:

        # x:(b, seq, d_model)
        input_shape = x.shape
        b, s, d = input_shape

        Q = self.q(x).view(b, s, self.n_head, self.d_head).transpose(1, 2)
        K = self.k(x).view(b, s, self.n_head, self.d_head).transpose(1, 2)
        V = self.v(x).view(b, s, self.n_head, self.d_head).transpose(1, 2)

        weight = Q @ K.transpose(-1, -2)
        if mask:
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)

        weight /= math.sqrt(self.d_head)
        atten_scores = F.softmax(weight, dim=-1)
        output = atten_scores @ V

        x = output.transpose(1, 2).reshape(input_shape)

        return self.O(x)
