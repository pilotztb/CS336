import torch 
import torch.nn as nn
from .Linear import Linear
from .func import silu

class SwiGLU(nn.Module):
    def __init__(
        self, 
        d_model,
        d_ff
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(self.d_model, self.d_ff)
        self.w3 = Linear(self.d_model, self.d_ff)
        self.w2 = Linear(self.d_ff, self.d_model)


    def forward(self, x):
        x = silu(self.w1(x)) * self.w3(x)
        x = self.w2(x)
        return x
