import torch
import torch.nn as nn
from .RotaryPositionEmbeding import RotaryPositionEmbedding
from .MHSA import MHSA
from .SwiGLU import SwiGLU
from .RMSNorm import RMSNorm

class TransformerBlock(nn.Module): # 1.1
    """A single Transformer layer.
    一个单独的 Transformer 层 (模块)。
    """
    def __init__(
        self,
        d_model: int,     # d_model = 512
        num_heads: int,   # num_heads = 16
        d_ff: int,        # d_ff = 1344 (来自 config.json)
        positional_encoder: RotaryPositionEmbedding, # 1.2
    ):
        super().__init__()
        
        # 1.3: 初始化第一个子层 (注意力)
        # 含义: 创建一个 CausalMultiHeadSelfAttention 实例。
        #       注意，它将 positional_encoder (RoPE 模块) 
        #       *传递* 给了 MHSA 的构造函数。
        self.attn = MHSA(
            d_model=d_model,
            num_heads=num_heads,
            positional_encoder=positional_encoder, 
        )
        
        # 1.4: 初始化第二个子层 (FFN)
        # 含义: 创建一个 SwiGLU 实例，
        #       它将 d_model=512 维向量上投影到 d_ff=1344 维，
        #       然后再投影回 512 维。
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff)
        
        # 1.5: 初始化第一个 RMSNorm (用于 Attention 之前)
        # 含义: 创建一个 RMSNorm 实例，
        #       它将用于在进入 Attention 子层之前进行归一化。
        self.ln1 = RMSNorm(d_model)
        
        # 1.6: 初始化第二个 RMSNorm (用于 FFN 之前)
        # 含义: 创建 *另一个* 独立的 RMSNorm 实例，
        #       它将用于在进入 FFN 子层之前进行归一化。
        self.ln2 = RMSNorm(d_model)

    def forward(self, x: torch.Tensor): # 2.1
        
        # 形状: (B, S, D_m) -> (32, 256, 512)
        # 含义: (即输入 x)
        #   32 (B - Batch Size): 批次中总共有 32 个独立的序列。
        #   256 (S - Sequence Length): 每个序列片段中有 256 个 Token。
        #   512 (D_m - Model Dimension): 每个 Token 用一个 512 维的浮点数向量表示。
        
        # --- 子层 1: 注意力 (Attention Sub-layer) ---
        
        # 2.2: (Pre-Norm) 先归一化 -> (Norm)
        # 形状: (32, 256, 512)
        # 含义:
        #   形状不变。
        #   将输入的 512 维向量通过 self.ln1 (RMSNorm) 进行归一化，
        #   为下一步的 Attention 计算准备好稳定的输入。
        x_norm1 = self.ln1(x)
        
        # 2.3: (Attention) 再计算注意力 -> (Attention)
        # 形状: (32, 256, 512)
        # 含义:
        #   形状不变。
        #   将 *归一化后* 的 x_norm1 送入 CausalMultiHeadSelfAttention 模块。
        #   输出的 x_attn 是 V 向量的加权求和 结果，已经融合了 Token 间的信息。
        x_attn = self.attn(x_norm1)
        
        # 2.4: (Residual) 残差连接 -> (Add)
        # 形状: (32, 256, 512)
        # 含义:
        #   这是图 2 中的第一个 Add。
        #   它将注意力层的输出 x_attn (行 2.3)
        #   与 *原始* 的、*未归一化* 的输入 x (行 2.1)
        #   进行逐元素相加。
        #   这是保证梯度回传的“捷径”。
        attn_sublayer_output = x + x_attn

        # --- 子层 2: 前馈网络 (FFN Sub-layer) ---
        
        # 2.5: (Pre-Norm) 再次归一化 -> (Norm)
        # 形状: (32, 256, 512)
        # 含义:
        #   形状不变。
        #   将第一个子层（Attn + Add）的输出 attn_sublayer_output
        #   送入 *第二个* RMSNorm (self.ln2) 进行归一化。
        x_norm2 = self.ln2(attn_sublayer_output)
        
        # 2.6: (FFN) 再计算 FFN -> (FFN)
        # 形状: (32, 256, 512)
        # 含义:
        #   形状不变。
        #   将 *第二次归一化后* 的 x_norm2 送入 SwiGLU (FFN) 模块。
        #   FFN 会对每个 Token 的 512 维向量独立进行一次复杂的非线性变换。
        x_ffn = self.ffn(x_norm2)
        
        # 2.7: (Residual) 第二次残差连接 -> (Add)
        # 形状: (32, 256, 512)
        # 含义:
        #   这是图 2 中的第二个 Add。
        #   它将 FFN 层的输出 x_ffn (行 2.6)
        #   与 *第一个子层的输出* attn_sublayer_output (行 2.4)
        #   进行逐元素相加。
        ffn_sublayer_output = attn_sublayer_output + x_ffn
        
        # 最终输出形状: (B, S, D_m) -> (32, 256, 512)
        # 含义:
        #   返回这个 TransformerBlock 的最终输出。
        #   这个张量现在准备好被送入 *下一个* TransformerBlock 
        #   （或者，如果是最后一层，则送入最终的 RMSNorm）。
        return ffn_sublayer_output # 2.8