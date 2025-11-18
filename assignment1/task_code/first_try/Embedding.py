import torch
import torch.nn as nn
import math

class Embedding(nn.Module):
    def __init__(
        self,
        d_model,
        vocab_size
    ):
        super().__init__()
        weight_tensor = torch.empty(vocab_size, d_model)
        self.std = 1
        nn.init.trunc_normal_(
            weight_tensor,
            std=self.std,
            a=-3,
            b=3
        )
        self.weight = nn.Parameter(
            weight_tensor,
            requires_grad=True
        )

    def forward(self, x):
        return self.weight[x]
    