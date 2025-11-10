import torch 
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model,
        eps=1e-5
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        weight_tensor = torch.ones(d_model)
        self.weight = nn.Parameter(
            weight_tensor,
            requires_grad=True
        )

    def forward(self, x):
        x_type = x.dtype
        new_x = x.to(torch.float32)
        new_x_initial = new_x
        new_x = new_x.pow(2).mean(-1, keepdims=True)
        new_x = torch.rsqrt(new_x + self.eps)
        new_x = new_x_initial * new_x
        new_x = new_x * self.weight
        ans = new_x.to(x_type)
        return ans