import torch
import torch.nn as nn
import math
from einops import einsum

class Linear(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        std = math.sqrt(2 / (d_in + d_out))
        weight_tensor = torch.empty(d_out, d_in)
        nn.init.trunc_normal_(weight_tensor, std=std, a=-3*std, b=3*std)
        self.weight = nn.Parameter(
            weight_tensor,
            requires_grad=True
        )

    def forward(self, x):
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
    
    def extra_print(self):
        return f"d_out = {self.weight.shape[0]}, d_in = {self.weight.shape[1]}"
    

class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        std = 1.0
        weight_tensor = torch.empty(self.vocab_size, self.d_model)
        nn.init.trunc_normal_(weight_tensor, std=std, a=-3*std, b=3*std)
        self.weight = nn.Parameter(
            weight_tensor, 
            requires_grad=True
        )

    def forward(self, x):
        return self.weight[x]
    

class RMSNorm(nn.Module): 
    def __init__(
        self, 
        d_model,
        eps=1e-5,
        device=None
    ):
        super().__init__()
        weight_tensor = torch.ones(d_model, device=device)
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(
            weight_tensor,
            requires_grad=True
        )
    
    def forward(self, x):
        # 也可以算数据预处理吧
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # 计算
        tmp_compute_res = x.pow(2).mean(-1, keepdim=True)
        tmp_compute_res = torch.rsqrt(tmp_compute_res + self.eps)
        tmp_compute_res = x * tmp_compute_res
        x = self.weight * tmp_compute_res
        
        return x.to(in_dtype)


def silu(x):
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model,
        d_ff
    ):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x):
        gate = silu(self.w1(x))
        content = self.w3(x)
        gated_content = gate * content
        return self.w2(gated_content)