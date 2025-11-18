import torch
import torch.nn as nn
from .Linear import Linear
from .func import scaled_dot_attention
from .RotaryPositionEmbeding import RotaryPositionEmbedding
import einops

class MHSA(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        positional_encoder: RotaryPositionEmbedding | None
    ) :
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        # q, k, v三个投影层，将输入x转化为Q, K, V
        self.q_proj = Linear(self.d_model, self.d_model)
        self.k_proj = Linear(self.d_model, self.d_model)
        self.v_proj = Linear(self.d_model, self.d_model)
        # 输出投影, 但是为什么输出投影层的输入和输出维度相同？
        # 答：因为上面一步融合的时候只是简单拼接，通过这里的投影可以进一步融合信息
        self.output_proj = Linear(self.d_model, self.d_model)
        self.positional_encoder = positional_encoder

    def forward(
        self, 
        x,
        token_position_id: torch.Tensor | None = None
    ):
        sequence_length, d_model = x.size(-2), x.size(-1)
        # 进行投影
        # Q, K, V的形状是[batch_size, 不知道, seq_length, d_model]
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        # 投影后的权重矩阵在之前写的自注意力里面

        # 分头行动
        Q, K, V = (
            einops.rearrange(X, "... seq_length (num_heads d_head) -> ... num_heads seq_length d_head", num_heads=self.num_heads)
            for X in (Q, K, V)
        )

        # 在传入自注意力前先位置编码
        if token_position_id is None:
            token_position_id = torch.arange(sequence_length, device=x.device)
        # 为什么需要在0这个位置再插入一个维度
        token_position_id = token_position_id.unsqueeze(0)
        if self.positional_encoder is not None:
            Q = self.positional_encoder(Q, token_position_id)
            K = self.positional_encoder(K, token_position_id)

        # 构造mask
        initial_mask = torch.arange(sequence_length, device=x.device)
        qi = initial_mask.unsqueeze(1)
        kj = initial_mask.unsqueeze(0)
        mask = qi >= kj

        # 计算注意力, 不过为什么多出一个维度就可以实现多头注意力了？
        # 答：此时Q, K, V的形状是[batch_size, num_heads, seq_length, d_head]
        # 在scaled_dot_attention中，由于ensum只处理最后两个维度，所以pytorch会自动处理前两个维度，在前两个维度上进行批处理，pytorch认为有batch_size * head_num这么多批，每次处理一批
        x = scaled_dot_attention(Q, K, V, mask)

        # 多头合并
        x = einops.rearrange(x, "... num_heads seq_length d_head -> ... seq_length (num_heads d_head)").contiguous()

        # 再次投影， 为什么需要这个投影？
        # 答：因为上面一步融合的时候只是简单拼接，通过这里的投影可以进一步融合信息
        x = self.output_proj(x)

        return x

        


        
