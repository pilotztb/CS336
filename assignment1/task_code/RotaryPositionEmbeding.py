import torch 
import torch.nn as nn
import einops
import einx


class RotaryPositionEmbedding(nn.Module):
    def __init__(
        self,
        context_length,
        d_head,
        theta = 10000.0
    ) -> None:
        super().__init__()
        self.register_buffer(
            "_cos_sin_matrix",
            RotaryPositionEmbedding._init_cos_sin_matrix(context_length, d_head, theta),
            persistent=False
        )

    @staticmethod
    def _init_cos_sin_matrix(context_length, d_head, theta):
        assert d_head % 2 == 0

        vector_i = torch.arange(context_length) 
        vector_k = torch.arange(0, d_head, 2)
        vector_deno_pow = vector_k / d_head
        vector_deno = theta ** -vector_deno_pow
        freq_matrix = einops.einsum(vector_i, vector_deno, "vector_i, vector_deno -> vector_i vector_deno")
        cos_matrix, sin_matrix = torch.cos(freq_matrix), torch.sin(freq_matrix)
        return torch.stack((cos_matrix, sin_matrix))

    def forward(
        self,
        x,
        pos_id
    ):
        x_even, x_odd = einops.rearrange(x, "... (half_dim xy) -> xy ... half_dim", xy = 2)
        cos_matrix, sin_matrix = einx.get_at("cos_sin [pos] half_dim, ... -> cos_sin ... half_dim", self._cos_sin_matrix, pos_id)
        x_even_rot = cos_matrix * x_even - sin_matrix * x_odd
        x_odd_rot = sin_matrix * x_even + cos_matrix * x_odd
        result = einx.rearrange("... x_half, ... x_half -> ... (x_half (1 + 1))", x_even_rot, x_odd_rot).contiguous()
        return result