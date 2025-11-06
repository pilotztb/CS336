import math
import torch
import torch.nn as nn
from einops import einsum

class Linear(nn.Module): 
    def __init__(
        self,
        d_in,
        d_out
    ):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.std = math.sqrt(2 / (self.d_in + self.d_out))
        weight_tensor = torch.empty(d_out, d_in)
        nn.init.trunc_normal_(weight_tensor, std=self.std, a=-3*self.std, b=3*self.std)
        self.weight = nn.Parameter(
            weight_tensor,
            requires_grad=True
        )
        
    def forward(
        self,
        x
    ):
        return einsum(x, self.weight, "... din, dout din -> ... dout")