
def softmax(x, dim=-1): # 1.1
    # 1.2: 减去最大值 (稳定技巧)
    rescaled_input = x - torch.max(x, dim=dim, keepdim=True)[0] 
    # 1.3: 计算 e^(v - c)
    exponentiated_rescaled_input = torch.exp(rescaled_input) 
    # 1.4: 计算分母 Σ[e^(v - c)]
    denominator = torch.sum(exponentiated_rescaled_input, dim=dim, keepdim=True) 
    # 1.5: 执行除法
    return exponentiated_rescaled_input / denominator 