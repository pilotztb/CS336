import torch
import torch.nn as nn
import einops
import math

def silu(x):
    return x * torch.sigmoid(x)

def softmax(x, dim=-1):
    new_x = x - torch.max(x, dim=dim, keepdim=True)[0]
    numerator = torch.exp(new_x)
    denominator = torch.sum(numerator, dim=dim, keepdim=True)
    return numerator / denominator

def scaled_dot_attention(
    Q,
    K, 
    V,
    mask
):
    for_scale_d = Q.shape[-1]
    initial_attention = einops.einsum(Q, K, "... q d, ... k d -> ... q k")
    initial_attention = initial_attention / math.sqrt(for_scale_d)
    if mask is not None:
        initial_attention = torch.where(mask, initial_attention, -torch.inf)
    attention_weight = softmax(initial_attention, -1)
    return einops.einsum(attention_weight, V, "... q v, ... v d -> ... q d")


