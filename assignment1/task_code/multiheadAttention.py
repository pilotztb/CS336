# (位于 hw1-basics/scripts/model.py)
from einops import rearrange, einsum
import einx
# ... 依赖我们之前实现的 Linear, RotaryEmbedding, scaled_dot_product_attention ...

class CausalMultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention
    多头自注意力机制
    """

    def __init__(
        self,
        d_model: int,     # d_model = 512
        num_heads: int,   # num_heads = 16
        positional_encoder: RotaryEmbedding, # 3.1: 传入 RoPE 模块
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model   # 512
        self.num_heads = num_heads # 16
        
        # self.d_k (维度) = 512 // 16 = 32, 这里的512是因为对于每个token，用一个向量表示，这个向量的维度是512维
        # (CN) d_k (d_head) 是每个注意力头的维度。
        self.d_k = d_model // num_heads
        
        # self.d_v (维度) = 32
        # (CN) 按照作业要求，d_v (值的维度) 设置为等于 d_k。
        self.d_v = self.d_k # 讲义要求 d_k = d_v

        # 3.2: 步骤1 - 定义 Q, K, V, O 投影层
        # (CN) Q 的投影层。输入: 512 -> 输出: 16 * 32 = 512
        self.q_proj = Linear(self.d_model, self.num_heads * self.d_k)
        
        # (CN) K 的投影层。输入: 512 -> 输出: 16 * 32 = 512
        self.k_proj = Linear(self.d_model, self.num_heads * self.d_k)
        
        # (CN) V 的投影层。输入: 512 -> 输出: 16 * 32 = 512
        self.v_proj = Linear(self.d_model, self.num_heads * self.d_v)
        
        # (CN) O (输出) 投影层。输入: 16 * 32 = 512 -> 输出: 512
        self.output_proj = Linear(self.num_heads * self.d_v, self.d_model)

        # (CN) 存储传入的 RoPE 模块实例。
        self.positional_encoder = positional_encoder  # RoPE

    def forward(self, x: Float[Tensor, " ... seq d_k"], token_positions: Int[Tensor, " ... seq"] | None = None) -> Float[Tensor, " ... seq d_v"]:
            
            # 传入的 x (来自上一个 RMSNorm)
            # 形状: (B, S, D_m) -> (32, 256, 512)
            # 含义:
            #   32 (B - Batch Size): 批次中总共有 32 个独立的序列
            #   256 (S - Sequence Length): 每个序列片段中有 256 个 Token
            #   512 (D_m - Model Dimension): 对于一个 Token，使用了一个 512 维的浮点数向量来表示它
            
            # 解包形状。b=[32], sequence_length=256, d_model=512
            *b, sequence_length, d_model = x.size()
            assert d_model == self.d_model

            # 3.3: 步骤1 - 执行投影 (Project)
            
            # Q 形状: (32, 256, 512) -> (32, 256, 512)
            # 含义: 形状不变。
            #   32 (B): 32 个序列。
            #   256 (S): 256 个 Token。
            #   512 (D_m): 一个 Token 的 512 维向量被 Linear 变换了，现在是它对应的 512 维的“总查询 (Q)”向量。
            Q = self.q_proj(x)
            
            # K 形状: (32, 256, 512) -> (32, 256, 512)
            # 含义: 同上，这是 512 维的“总键 (K)”向量。
            K = self.k_proj(x)
            
            # V 形状: (32, 256, 512) -> (32, 256, 512)
            # 含义: 同上，这是 512 维的“总值 (V)”向量。
            V = self.v_proj(x)

            # 3.4: 步骤2 - 拆分多头 (Split Heads)
            # “从 d_model 维度中拆分出 num_heads 个头”
            
            # Q, K, V 形状: (32, 256, 512) -> (32, 16, 256, 32)
            # 含义: (B, S, D_m) -> (B, H, S, D_h)
            #   32 (B - Batch Size): 批次中的 32 个序列。
            #   16 (H - Num Heads): 这是您问的“16是什么意思？”
            #       是的，这是 16 个“注意力头”。
            #       我们将 512 维的“总向量”拆分成了 16 份，给 16 个“专家”并行处理。
            #   256 (S - Sequence Length): 每个序列中的 256 个 Token。
            #   32 (D_h - Head Dimension): 这是您问的“32是什么意思？”
            #       是的，这是“一个头”的维度 (d_head)。
            #       因为 512 (D_m) / 16 (H) = 32 (D_h)。
            #       您说的“现在只用一个包含32个元素的向量来表示原来对应的一个单词？” —— 完全正确！
            #       在 Head 0 的视角里，它只看到一个 32 维向量；Head 1 也只看到一个不同的 32 维向量。
            Q, K, V = (
                rearrange(X, "... seq (heads d) -> ... heads seq d", heads=self.num_heads)
                for X in (Q, K, V)
            )

            if token_positions is None:
                # 3.5: (辅助) 创建 token_positions (绝对位置 [0, 1, ..., 255])
                # 这是您问的“为什么要这样搞”
                
                # 1. 创建基础序列
                # 形状: (S,) -> (256,)
                # 含义: 一个包含绝对位置 [0, 1, ..., 255] 的一维向量。
                _seq = torch.arange(sequence_length, device=x.device)
                
                # 2. 为“批次(B)”广播添加 '1'
                # 形状: (256,) -> (1, 256)
                # 含义: (S,) -> (1, S)。
                #       len(b) 是 1 (因为 x 的批次维度只有 32)。
                #       einx 在 seq 前面加上 len(b) 个大小为 1 的维度。
                #       这个 '1' 将用于广播匹配 B=32。
                token_positions = einx.rearrange("seq -> b... seq", _seq, b=[1] * len(b))
            
            # 3. (RoPE 需要) 为“头(H)”广播添加 '1'
            # 形状: (1, 256) -> (1, 1, 256)
            # 含义: (1, S) -> (1, 1, S)。
            #       在 'S' 之前再插入一个大小为 '1' 的维度。
            #       这个 '1' 将用于广播匹配 H=16。
            #       最终这个 (1, 1, 256) 的张量可以高效地被所有 (32, 16, 256) 的 token 复用。
            token_positions = rearrange(token_positions, "... seq -> ... 1 seq")

            # 3.6: 步骤 3a - 应用 RoPE (旋转 Q 和 K)
            
            # Q 形状: (32, 16, 256, 32) -> (32, 16, 256, 32)
            # 含义: (B, H, S, D_h) -> (B, H, S, D_h)。
            #       形状不变。RoPE 使用 (1, 1, 256) 的位置信息，
            #       并行地旋转了所有 32*16 个 32 维的子序列，将位置信息注入到了向量的“内容”中。
            Q = self.positional_encoder(Q, token_positions)
            
            # K 形状: (32, 16, 256, 32) -> (32, 16, 256, 32)
            # 含义: 同上，K 向量的内容也被旋转了。
            K = self.positional_encoder(K, token_positions)
            
            # V 向量 (Value) 不需要旋转，保持不变。

            # 3.7: 步骤 3b - 构建因果掩码 (Causal Mask)
            # 这是您问的“这里为什么要构建因果掩码”
            # 含义: 这是为了防止模型“作弊”。我们是“仅解码器”架构，
            #       在预测第 10 个词时，决不能让它看到第 11 个词的信息。
            #       这个掩码会强制实现“只能看到过去和现在”。
            
            # seq 形状: (256,)
            seq = torch.arange(sequence_length, device=x.device)
            
            # qi 形状: (256,) -> (1, 1, 256, 1)
            # 含义: (S,) -> (1_Bcast, 1_Hcast, S_query, 1)。创建了一个“查询”索引（行索引）。
            qi = einx.rearrange('query -> b... 1 query 1', seq, b=[1] * len(b))
            
            # kj 形状: (256,) -> (1, 1, 1, 256)
            # 含义: (S,) -> (1_Bcast, 1_Hcast, 1, S_key)。创建了一个“键”索引（列索引）。
            kj = einx.rearrange('key   -> b... 1 1   key', seq, b=[1] * len(b))
            
            # causal_mask 形状: (1, 1, 256, 1) >= (1, 1, 1, 256) -> (1, 1, 256, 256)
            # 含义: (1_Bcast, 1_Hcast, S_query, S_key)。
            #       通过广播 qi 和 kj，创建了一个布尔 (True/False) 掩码。
            #       值为 True 的地方 (qi >= kj) 代表“允许关注”(现在和过去)。
            #       值为 False 的地方 (qi < kj) 代表“禁止关注”(未来)。
            causal_mask = qi >= kj

            # 3.8: 步骤 3c - 调用 SDPA (执行注意力计算)
            
            # 输入 Q,K,V 形状: (32, 16, 256, 32) (B, H, S, D_h)
            # 输入 mask 形状: (1, 1, 256, 256) (B_bcast, H_bcast, S_query, S_key)
            # 输出 attn_output 形状: (32, 16, 256, 32) (B, H, S, D_h)
            # 含义: SDPA 函数在 H=16 个头上并行执行了完整的
            #       (Q @ K.T) / sqrt(d_k) -> Mask -> Softmax -> @ V 流程。
            #       输出的 attn_output 张量，其内容已经是 V 向量的加权求和 结果。
            attn_output = scaled_dot_product_attention(K=K, Q=Q, V=V, mask=causal_mask)

            # 3.9: 步骤 4 - 拼接多头 (Concatenate Heads)
            
            # 输入 attn_output 形状: (32, 16, 256, 32)
            # 输出 attn_output 形状: (32, 256, 512)
            # 含义: (B, H, S, D_h) -> (B, S, D_m)
            #   32 (B): 批次中的 32 个序列。
            #   256 (S): 每个序列中的 256 个 Token。
            #   512 (D_m): 合并后的维度。
            #       rearrange 将 16 个 32 维的“专家”输出向量重新拼接 (Concatenate) 
            #       回一个 512 维的“总输出”向量。
            attn_output = rearrange(attn_output, "batch heads seq d_v -> batch seq (heads d_v)").contiguous()

            # 3.10: 步骤 5 - 输出投影 (Final Projection)
            
            # 输入 attn_output 形状: (32, 256, 512)
            # 输出 output 形状: (32, 256, 512)
            # 含义: (B, S, D_m) -> (B, S, D_m)
            #       将拼接好的 512 维向量再通过一个 Linear(512, 512) 层，
            #       进行一次最终的“信息融合”，得到这个 CausalMultiHeadSelfAttention 模块的最终输出。
            output = self.output_proj(attn_output)
            
            # (EN) Final Output Shape: (B, S, D_m) -> (32, 256, 512)
            # (CN) 最终输出形状: (B, S, D_m) -> (32, 256, 512)
            return output