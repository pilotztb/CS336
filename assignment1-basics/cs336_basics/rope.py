# 需要导入的库
import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int # 用于类型提示
from einops import rearrange, einsum # 用于张量操作
import einx # 优秀代码中还使用了 einx

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
    def _init_cache(context_length: int, dim: int, theta: float) -> Float[Tensor, " 2 context_length half_dim"]: # 2.2
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
    def forward(self, x: Float[Tensor, " ... seq d"], pos_ids: Int[Tensor, " ... seq"]) -> Float[Tensor, " ... seq d"]: # 3.1
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