def scaled_dot_product_attention(
    Q: Float[Tensor, " ... queries d_k"], # 1.1
    K: Float[Tensor, " ... keys    d_k"], # 1.1
    V: Float[Tensor, " ... keys    d_v"], # 1.1
    mask: Bool[Tensor, " ... queries keys"] | None = None, # 1.2
) -> Float[Tensor, " ... queries d_v"]:
    """Scaled dot-product attention."""
    
    # --- 步骤 1: 计算分数 (Score) & 步骤 2: 缩放 (Scale) ---
    d_k = K.shape[-1] # 2.1
    # 2.2: 计算 QK^T / sqrt(d_k)
    attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k) 

    # --- 步骤 3: 应用掩码 (Mask) ---
    if mask is not None: # 3.1
        # 3.2: 将 mask 为 False 的位置设置为 -inf
        attention_scores = torch.where(mask, attention_scores, float("-inf")) 

    # --- 步骤 4: 归一化 (Softmax) ---
    # 4.1: 调用手搓的 softmax 函数
    attention_weights = softmax(attention_scores, dim=-1)  

    # --- 步骤 5: 加权求和 (Weighted Sum) ---
    # 5.1: 计算 Attention_Weights @ V
    return einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v") 