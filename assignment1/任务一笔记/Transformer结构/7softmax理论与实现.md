根据课程讲义，我们在 Transformer Block 内部的学习顺序是：`RMSNorm`、FFN (`SwiGLU`)、`RoPE`（它在自注意力内部使用）。我们接下来要学习的就是 Transformer Block 的**第一个**核心计算单元：**自注意力 (Self-Attention)**。

讲义在 §3.5.4 节 指出，在实现完整的“缩放点积注意力”之前，我们必须先实现一个它依赖的核心辅助函数：**`softmax`**。

因此，我们先来学习 `softmax` 函数。

---

# `softmax` 函数：理论知识

## 1. 作用与目的

`softmax` 函数在 Transformer（乃至整个深度学习）中扮演着一个至关重要的角色：它是一个**“归一化”函数**，能将一组**任意的实数（logits）**转换成一组**概率分布**。

* **在自注意力 (Self-Attention) 中的角色**：
    * 自注意力机制（我们马上会学）的第一步是计算 Query 向量 (Q) 和 Key 向量 (K) 的**点积**。这个点积的结果是一个**“注意力分数 (attention score)”** 矩阵。
    * 这个矩阵里的分数可以是**任何值**（例如 `-10.5`, `8.2`, `500.0`）。这些分数代表了“一个词应该对另一个词**多大程度**的关注”，分数越高，关注度越大。
    * **问题**：我们不能直接使用这些原始分数。我们需要将它们转换成一种“权重”，使得：
        1.  所有权重都是**正数**（0 到 1 之间）。
        2.  一个 token 对其他所有 token 的关注度**总和为 1**（100%）。
    * **`softmax` 的目的**：`softmax` 函数 正是用来实现这个转换的。它接收原始的“注意力分数”向量，并输出一个“注意力权重”向量（概率分布），确保所有权重加起来等于 1。

## 2. 数学原理

给定一个包含 $n$ 个分数的输入向量 $v = [v_1, v_2, ..., v_n]$，`softmax` 函数 对**每一个**元素 $v_i$ 进行如下计算：

$$softmax(v)_i = \frac{e^{v_i}}{\sum_{j=1}^{n} e^{v_j}}$$

* **$e^{v_i}$ (分子)**：
    * `e` 是自然对数的底数（约 2.718）。
    * `e` 的指数函数（`exp()`）有一个很好的特性：无论输入 $v_i$ 是正数、负数还是零，`exp(v_i)` **永远是一个正数**。这就满足了我们“权重必须是正数”的需求。
* **$\sum_{j=1}^{n} e^{v_j}$ (分母)**：
    * 这是对向量中**所有**元素的指数 `exp(v_j)` 进行**求和**。
* **$\frac{...}{...}$ (除法)**：
    * 用某一个元素的指数值（分子）除以**所有**元素的指数值之和（分母）。
    * 这确保了所有输出值（$\frac{e^{v_1}}{\sum ...}, \frac{e^{v_2}}{\sum ...}, ...$）加起来**总和**正好等于 1。

**例子**：
* 输入分数：`v = [1.0, 3.0, 2.0]`
* $e^{1.0} \approx 2.718$
* $e^{3.0} \approx 20.086$
* $e^{2.0} \approx 7.389$
* **分母 (总和)**：$2.718 + 20.086 + 7.389 = 30.193$
* **输出 (概率)**：
    * $softmax(v)_1 = 2.718 / 30.193 \approx 0.09$
    * $softmax(v)_2 = 20.086 / 30.193 \approx 0.67$
    * $softmax(v)_3 = 7.389 / 30.193 \approx 0.24$
* **检查**：$0.09 + 0.67 + 0.24 = 1.0$。转换成功！

## 3. 讲义中的关键要求与约束

* **实现 `softmax` 函数**：需要实现一个函数，而不是一个类。
* **沿维度 (`dim`) 计算**：函数需要能沿着张量的**特定维度** `dim` 来执行 `softmax`。例如，在 `(batch_size, seq_len, seq_len)` 的分数矩阵上，我们通常希望沿着**最后一个**维度（`dim=-1`）进行 `softmax`。
* **禁止使用内置实现**：不能使用 `torch.nn.functional.softmax` 或 `torch.nn.Softmax`。
* **关键要求：数值稳定性 (Numerical Stability)**：
    * **问题**：如果输入分数 `v_i` 很大（例如 `1000`），`exp(1000)` 会返回一个**无穷大 (`inf`)** 的值，这称为**上溢 (Overflow)**。计算 `inf / inf` 会得到 `NaN` (Not a Number)，导致模型训练崩溃。
    * **解决方案 (数学技巧)**：**`softmax` 函数有一个重要特性：`softmax(v) = softmax(v - c)`**，其中 `c` 是**任意**常数。
    * **证明**：
        $$\frac{e^{v_i - c}}{\sum e^{v_j - c}} = \frac{e^{v_i} \cdot e^{-c}}{\sum (e^{v_j} \cdot e^{-c})} = \frac{e^{v_i} \cdot e^{-c}}{e^{-c} \cdot \sum e^{v_j}} = \frac{e^{v_i}}{\sum e^{v_j}}$$
    * **讲义要求的实现**：为了防止上溢，我们在计算 `softmax` 之前，可以从向量 $v$ 的**所有**元素中减去**同一个**常数 `c`，而结果**不变**。
    * **最佳选择**：讲义明确要求我们使用这个技巧：**选择 $c = \max(v)$（即向量 $v$ 中的最大值）。**
    * **为什么这能行**：
        1.  我们计算 `rescaled_v = v - max(v)`。
        2.  `rescaled_v` 中的最大值**必定是 0**（$\max(v) - \max(v) = 0$）。
        3.  `rescaled_v` 中的其他值都是**负数或 0**。
        4.  计算 `exp(rescaled_v)` 时，最大的指数值是 `exp(0) = 1`。
        5.  其他指数值是 `exp(负数)`，结果是介于 0 和 1 之间的小数。
        6.  这样，`exp` 的输入永远不会大于 0，**完美地避免了上溢 (`inf`) 的问题**。

---

# `softmax` 函数：代码实现讲解

## **代码实现 (源自优秀代码 `hw1-basics/scripts/model.py`)**

在 `hw1-basics/scripts/model.py` 文件的开头附近，定义了 `softmax` 函数：

```python
# (位于 hw1-basics/scripts/model.py 顶部)

def softmax(x, dim=-1): # 1.1
    # 1.2: 减去最大值 (稳定技巧)
    rescaled_input = x - torch.max(x, dim=dim, keepdim=True)[0] 
    # 1.3: 计算 e^(v - c)
    exponentiated_rescaled_input = torch.exp(rescaled_input) 
    # 1.4: 计算分母 Σ[e^(v - c)]
    denominator = torch.sum(exponentiated_rescaled_input, dim=dim, keepdim=True) 
    # 1.5: 执行除法
    return exponentiated_rescaled_input / denominator 
```

-----

## **逐行分析**

### **行 1.1 `def softmax(x, dim=-1):`**

  * **作用**：定义 `softmax` 函数。
  * **解释**：
      * `x`: 输入的张量（即原始分数/logits）。
      * `dim=-1`: 指定沿着哪个维度进行 `softmax` 运算。`-1` 是 Python 中“最后一个维度”的便捷写法。这完全符合讲义中“需要能沿着张量的特定维度 `dim`” 的要求。

### **行 1.2 `rescaled_input = x - torch.max(x, dim=dim, keepdim=True)[0]`**

  * **作用**：**实现“减去最大值”的数值稳定技巧**。
  * **解释**：
      * `torch.max(x, dim=dim, keepdim=True)`: 这是一个关键操作。
          * `dim=dim`：告诉 `torch.max` 沿着我们指定的维度（例如 `-1`）查找最大值。
          * `keepdim=True`：**非常重要**。假设 `x` 的形状是 `(B, S, S)`，我们沿 `dim=-1` 找最大值。
              * 如果 `keepdim=False`（默认），结果形状会是 `(B, S)`，这会导致后续 `x - max_val` 时广播 (broadcasting) 失败。
              * 设置 `keepdim=True`，`torch.max` 会保持被缩减的维度（大小变为 1），返回结果的形状是 `(B, S, 1)`。
          * `[0]`: `torch.max` 函数会返回一个元组 `(values, indices)`，即最大值张量和最大值所在的索引张量。我们**只**需要**值**，所以我们通过 `[0]` 来获取 `values` 张量（形状 `(B, S, 1)`）。
      * `x - ...`: `x` (形状 `(B, S, S)`) 减去 `torch.max(...)[0]` (形状 `(B, S, 1)`)。PyTorch 的**广播机制**会自动将 `(B, S, 1)` 张量“复制” S 次，使得 `x` 的**每一行**（`dim=-1` 维度上的 S 个元素）都减去了**那一行的最大值**。
      * `rescaled_input`: 这就是我们理论中的 $v - c$（其中 $c = \max(v)$）。

### **行 1.3 `exponentiated_rescaled_input = torch.exp(rescaled_input)`**

  * **作用**：计算 $e^{(v-c)}$，即公式的**分子**部分。
  * **解释**：`torch.exp()` 逐元素地计算 `rescaled_input` 中每个元素的 $e$ 次幂。由于 `rescaled_input` 中的最大值是 0，`torch.exp` 的最大输出是 `exp(0) = 1`，从而**避免了上溢 (`inf`)**。

### **行 1.4 `denominator = torch.sum(exponentiated_rescaled_input, dim=dim, keepdim=True)`**

  * **作用**：计算 $\sum e^{(v_j - c)}$，即公式的**分母**部分。
  * **解释**：
      * `torch.sum(...)`: 对 `exponentiated_rescaled_input`（$e^{(v-c)}$）进行求和。
      * `dim=dim`: 告诉 `torch.sum` 沿着我们指定的维度（例如 `-1`）进行求和。
      * `keepdim=True`：**再次**保持维度。这使得 `denominator` (分母) 的形状与 `torch.max` 的结果类似，也是 `(B, S, 1)`。

### **行 1.5 `return exponentiated_rescaled_input / denominator`**

  * **作用**：执行最终的除法 $\frac{e^{v_i - c}}{\sum e^{v_j - c}}$。
  * **解释**：
      * `exponentiated_rescaled_input` (分子)：形状 `(B, S, S)`。
      * `denominator` (分母)：形状 `(B, S, 1)`。
      * `... / ...`: PyTorch 再次使用**广播机制**。它会将分母 `(B, S, 1)` 自动“复制” S 次，使得分子 `(B, S, S)` 中的**每一行**都除以**对应的那一行的总和**。
      * 这完美地实现了 `softmax` 的逐行（或逐 `dim`）归一化。

-----

**总结**：

优秀代码中的 `softmax` 函数 是一个**健壮且高效**的实现：

  * 它严格实现了 `softmax` 的数学定义。
  * 它通过 `x - torch.max(x, ...)[0]` 巧妙地实现了“减去最大值”的**数值稳定技巧**。
  * 它通过在 `torch.max` 和 `torch.sum` 中都使用 `keepdim=True`，充分利用了 PyTorch 的**广播机制**，避免了额外的 `unsqueeze` 或 `reshape` 操作。

您对 `softmax` 的代码实现还有疑问吗？如果清楚了，我们就可以继续学习 `Scaled Dot-Product Attention`（缩放点积注意力）的理论知识了。