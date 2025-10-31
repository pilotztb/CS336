import torch
import torch.nn as nn
import math
from einops import einsum, rearrange
import einx

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
    

class RotaryEmbedding(nn.Module): # 1.1
    def __init__(self, context_length: int, dim: int, theta: float = 10000.0): # 1.2
        super().__init__() # 1.3
        # 1.4: 注册一个非参数的缓冲区 (buffer)
        self.register_buffer(
            "_freq_cis_cache", # 1.5
            RotaryEmbedding._init_cache(context_length, dim, theta), # 1.6
            persistent=False # 1.7
        )
    
    @staticmethod # 2.1
    def _init_cache(context_length: int, dim: int, theta: float) : # 2.2
        # --- 段落 2a: 计算旋转频率 ---
        assert dim % 2 == 0 # 2.3
        
        # 2.4: 计算 d = [0, 2, 4, ..., dim-2] / dim
        d = torch.arange(0, dim, 2) / dim 
        # 2.5: 计算 freqs = theta ** -d (即 Θ^(-2k/d_k))
        freqs = theta ** -d 
        # 2.6: t = [0, 1, ..., context_length-1]
        t = torch.arange(context_length) 

        # 2.7: 计算 freqs = t @ freqs.T (计算所有 (i, k) 组合的 θ_ik)
        # 形状: (context_length, dim/2)
        freqs = einsum(t, freqs, "t, f -> t f") 

        # --- 段落 2b: 计算并缓存 sin 和 cos ---
        # 2.8: 计算所有 sin 和 cos 值
        cos, sin = torch.cos(freqs), torch.sin(freqs) 
        # 2.9: 将 cos 和 sin 堆叠在一起
        # 最终缓存形状: (2, context_length, dim/2)
        return torch.stack((cos, sin)) 

    # --- 段落 3: forward 方法 ---
    def forward(self, x, pos_ids): # 3.1
        # --- 段落 3a: 拆分向量为 (偶数, 奇数) 对 ---
        # 3.2: x 形状 (..., d) -> x1, x2 形状 (..., dim/2)
        x1, x2 = rearrange(x, '... (half_d xy) -> xy ... half_d', xy=2) 

        # --- 段落 3b: 查找对应位置的 sin/cos 值 ---
        # 3.3: 从缓存 _freq_cis_cache 中根据 pos_ids 查找
        cos, sin = einx.get_at('cos_sin [pos] half_dim, ... -> cos_sin ... half_dim', self._freq_cis_cache, pos_ids)

        # --- 段落 3c: 执行 2D 旋转 ---
        # 3.4: x_rot = x_even * cos - x_odd * sin
        x1_rot = cos * x1 - sin * x2 
        # 3.5: x_rot = x_even * sin + x_odd * cos
        x2_rot = sin * x1 + cos * x2 
        
        # --- 段落 3d: 重组向量 ---
        # 3.6: 将 (偶数, 奇数) 对重新交错合并
        result = einx.rearrange('... x_half, ... x_half -> ... (x_half (1 + 1))', x1_rot, x2_rot).contiguous()
        return result # 3.7
    
    def extra_repr(self): # 4.1
        return f"context_length={self._freq_cis_cache.shape[0]}, dim/2={self._freq_cis_cache.shape[1]}" # 4.2