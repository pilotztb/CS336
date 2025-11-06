# 关于数据流动补充

我们的目标：追踪一个批次 (batch) 的数据，从**整数 ID** 变为**最终的注意力输出**。

## 第 0 步：我们的“演员表”（具体参数）

* 来自 `config.json` 和 `train.py`：
    * `batch_size` (B): **32**
    * `context_length` (S): **256**
    * `d_model` (D_m): **512**
    * `num_heads` (H): **16**
* 计算得出：
    * `d_head` (D_h): **32** (因为 `d_model / num_heads` = 512 / 16 = 32)

---

## 第 1 步：`get_batch` (创建批次)

* **操作**: `get_batch(...)`
* **形状**: `(32, 256)`
* **内容**: 32 x 256 的**整数 Token ID**。

> **您的困惑 1：“32个起始位置...32个句子，256是一个句子有256个单词？”**

**精确解答**：您的理解**基本完全正确**，只需修正两个小词：

1.  **32 是 32 个“序列 (Sequence)”**：您说“32个句子”非常接近。更准确地说，`train.dat` 是一个巨大的、连续的 ID 序列（可能包含了几万个故事）。`get_batch` 只是随机地从这个长序列中**切**出了 32 个**独立的片段 (chunk)**。每个片段就是一个训练样本，我们称之为“序列”。
2.  **256 是 256 个“Token”**：您说“256个单词”也非常接近。因为我们用的是 BPE 分词器，一个单词（比如 "upon"）可能会被拆分成多个 Token（比如 `" u"`, `"pon"`）。所以，`256` 代表每个序列片段都有 256 个 BPE Token。

**结论**：`x` 张量 `(32, 256)` 代表 **32 个独立的训练序列，每个序列长度为 256 个 Token ID**。

---

## 第 2 步：`Embedding` (词嵌入)

* **操作**: `x_emb = self.token_embeddings(x)`
* **输入形状**: `(32, 256)` (整数)
* **输出形状**: `(32, 256, 512)` (浮点数)

> **您的困惑 2：“接着词嵌入，就是将每个ID映射为一个向量，这个向量是512维？”**

**精确解答**：**是的！您 100% 正确！**

这正是 `3Embedding层理论与实现.md` 笔记中的“查找表”理论。
`Embedding` 模块 内部有一个大表 `self.weight`，形状是 `(10000, 512)`。

它对输入 `(32, 256)` 中的**每一个** ID（总共 32 * 256 = 8192 个 ID），都去查找这个表，用一个 **512 维**的浮点数向量替换它。

**结论**：我们现在有了一个形状为 `(B, S, D_m)` 即 **`(32, 256, 512)`** 的张量。**到目前为止，每一个 Token（原来的单词/子词）都由一个 512 维的向量表示**。

---

## 第 3 步：`RMSNorm` (预归一化)

* **操作**: `x_norm = self.ln1(x_emb)` (在 `TransformerBlock` 中)
* **输入形状**: `(32, 256, 512)`
* **输出形状**: `(32, 256, 512)` (内容被归一化)

> **您的困惑 3：“再接下来额？为什么首先就要归一化？”**

**精确解答**：这是一个**极其重要**的架构设计问题！

1.  **这是“Pre-Norm” (预归一化) 架构**：
    您可能在其他地方见过“Post-Norm”（后归一化）架构，即在`Attention` 和 `FFN` *之后*才进行归一化。
2.  **为什么用 Pre-Norm？**
    作业 PDF §3.5 和 Figure 2 明确指定我们实现 `Pre-Norm` 架构（这也是 Llama 等现代模型的选择）。
    * **理论**：`Attention` 和 `FFN` 是非常复杂的计算，它们的输入值如果过大或过小，会导致计算不稳定和**梯度爆炸/消失**。
    * **目的**：在**进入**这些复杂计算（如 `self.attn`）**之前**，先用 `RMSNorm`“洗”一遍数据，将每个 512 维向量的尺度（scale）拉回到一个可控的范围。
    * **好处**：这就像在进行精细的化学实验前先校准所有试剂的浓度。它使得训练过程**极其稳定**，梯度可以平滑地回传，尤其是在模型非常深（有很多层 `TransformerBlock`）的时候。

**结论**：我们首先归一化，是为了**保证**下一步 `Attention` 计算的**数值稳定性**。

---

## 第 4 步：`Linear(QKV)` + `rearrange` (多头拆分)

这是您最核心的困惑点，我们把它分解开：

### 4a. `Linear` 投影 (Project)

* **操作**: `Q = self.q_proj(x_norm)`
* **输入**: `(32, 256, 512)`
* **输出 (Q)**: `(32, 256, 512)`
* **含义**: 我们仍然用一个 512 维的向量来表示 Q。这一步只是一个 `Linear` 变换，内容改变了，但“一个 Token 对应一个 512 维向量”这个概念**还没变**。

### 4b. `rearrange` 拆分 (Split)

* **操作**: `rearrange(Q, "... seq (heads d) -> ... heads seq d", ...)`
* **输入**: `(32, 256, 512)` (B, S, D_m)
* **输出**: `(32, 16, 256, 32)` (B, H, S, D_h)

> **您的困惑 4：“16是什么意思？16个头？32是什么意思？现在只用一个包含32个元素的向量来表示原来对应的一个单词？”**

**精确解答**：**是的！您又一次 100% 正确地抓住了重点！**

这正是“**多头** (Multi-Head)”注意力的**核心意义**：

1.  **为什么这么做？**
    模型认为，只用**一种**方式（一个 512 维的“大脑袋”）去计算“相关性”是**不够**的。
    它想**并行**地计算 **16 种**（`H=16`）**不同类型**的“相关性”。
    * 比如，`Head 0`（第0个头）可能专门学习寻找**语法**关系。
    * `Head 1`（第1个头）可能专门学习寻找**同义词**关系。
    * ...
    * `Head 15`（第15个头）可能专门学习**上文指代**关系。

2.  **如何实现并行？**
    模型说：“我没有 16 * 512 这么多的维度可以用。我总共只有 512 维。好吧，我把这 512 维**平均分**给 16 个头。”
    * **`512 / 16 = 32`**

3.  **`rearrange` 的真正含义**：
    `rearrange` 操作 就像一个“分牌”的荷官。它拿起**每一个** Token 的 512 维向量（`[v1, v2, ..., v512]`），然后：
    * 把 `[v1, ..., v32]`（前32个元素） **分给 Head 0**。
    * 把 `[v33, ..., v64]`（下32个元素） **分给 Head 1**。
    * ...
    * 把 `[v481, ..., v512]`（最后32个元素）**分给 Head 15**。

4.  **回答您的困惑**：
    * **`16` 是什么？** 是的，它就是 `num_heads`（16个头）。我们现在进入了 16 个“平行宇宙”。
    * **`32` 是什么？** 是的，它是 `d_head`（`512 / 16`）。它是**每个头**（每个“专家”）能看到的**维度**。
    * **“现在只用一个32维向量表示单词？”** **是的！** 对于 `Head 0` 来说，它**只能**看到一个 32 维的向量。对于 `Head 1` 来说，它也**只能**看到一个**不同**的 32 维向量。这 16 个 32 维向量**共同**代表了原始的 512 维信息。

**结论**：`rearrange` 之后，形状 `(32, 16, 256, 32)` (B, H, S, D_h) 的含义是：
* 我们有 **32** 个序列。
* 在**每个**序列中，我们**并行**运行 **16** 个注意力“专家”（Head）。
* 在**每个**“专家”的视角里，它看到 **256** 个 Token。
* 在**每个**“专家”的视角里，**每个** Token 都由一个 **32** 维的向量表示。

---

## 第 5 步：`RoPE` (旋转编码)

* **操作**: `Q = self.positional_encoder(Q, ...)`
* **输入**: `(32, 16, 256, 32)`
* **输出**: `(32, 16, 256, 32)`

**含义**：
现在，`RoPE` 开始工作。它会**并行**地对**所有** `32 * 16 = 512` 个“子序列”（每个序列长 256，维度 32）应用旋转。
`RoPE` 的 `dim` 参数 在这里等于 `d_head`（即 32），它会在**每个头**的 32 维空间内独立进行旋转。

## 总结 (数据流终点)

1.  **`get_batch`**: `(32, 256)` —— 32 个序列，每个 256 个 Token ID。
2.  **`Embedding`**: `(32, 256, 512)` —— 每个 Token ID 变成了 512 维的“总义”向量。
3.  **`RMSNorm`**: `(32, 256, 512)` —— 归一化，为计算做准备。
4.  **`q_proj` (Linear)**: `(32, 256, 512)` —— 512 维的“总义”向量被变换为 512 维的“总查询”向量。
5.  **`rearrange`**: `(32, 16, 256, 32)` —— 512 维的“总查询”向量被**拆分**给 16 个“专家”（Head），每个专家只看到一个 32 维的“子查询”向量。
6.  **`RoPE`**: `(32, 16, 256, 32)` —— **每个** 32 维的“子查询”向量被旋转，注入位置信息。
7.  **`SDPA`**: `(32, 16, 256, 32)` —— 16 个专家**并行**地在它们各自的 32 维世界里完成注意力和加权求和。

（在此之后，模型会再用 `rearrange` 把 `(32, 16, 256, 32)` **拼回** `(32, 256, 512)`，然后送入下一个 `RMSNorm` 和 `SwiGLU`）。

希望这个更详细、专门针对您困惑点的解答，能让整个流程变得清晰起来！

# 因果多头自注意力 (MHSA)：理论知识

## 1\. 作用与目的

如果我们直接使用 `SDPA`，就好像只有一个“通用”的注意力机制。而 **“多头” (Multi-Head)** 的思想是，我们不应该只用一种方式来计算“相关性”。

**一个非常贴切的比喻**：

  * **单头 (SDPA)**：您派**一个**“通才”专家去分析一个句子。他必须同时关注语法、语义、上下文等所有方面，可能会顾此失彼。
  * **多头 (MHSA)**：您派一个**“专家团队”**（例如 16 个专家，即 `num_heads=16`）去分析句子。
      * **专家 1 (Head 1)**：可能专门学会了寻找“主谓宾”关系。
      * **专家 2 (Head 2)**：可能专门学会了寻找“代词指代”关系（例如 "it" 指的是 "the cat"）。
      * **专家 3 (Head 3)**：可能专门学会了“相邻词”的搭配。
      * ...等等。

**MHSA 的目的**：

1.  它不是让一个 `d_model` 维度的向量（例如 512 维）单独计算注意力...
2.  ...而是将 `d_model` **拆分**成 `num_heads` 个更小的“头”（例如 16 个头，每个头 32 维，因为 $16 \times 32 = 512$）。
3.  让**每一个头**（每一位“专家”）**独立地**、**并行地**执行自己的 `SDPA` 计算。
4.  最后，将所有 `num_heads` 个专家的“见解”（输出向量）**拼接 (Concatenate)** 起来，并通过一个最终的线性层 (Output Projection) 将它们“融合”成一个统一的 `d_model` 维输出向量。

**补充：**

  * **“Self” (自)**：指 Q、K、V **全部**来源于**同一个**输入 `x`（即 token 序列本身），这就是“自注意力”。
  * **“Causal” (因果)**：指我们必须应用**因果掩码**（即我们上一步在 `SDPA` 中学的 `mask`），以确保任何 token (Query) 只能关注到它自己**以及**它之前的 token (Keys)，**绝不能**关注“未来”的 token。

## 2\. 结构与数学原理

一个完整的 (Causal) Multi-Head Self-Attention 模块，处理一个输入张量 `x` (形状 `... seq_len, d_model`)，需要执行以下 5 个步骤：

**步骤 1：投影 (Projections)**
首先，我们不能直接把 `x` 当作 Q, K, V。我们需要为 Q, K, V 和最终的 Output 准备 4 组**可学习的**权重矩阵（即 4 个 `Linear` 层）。

  * $W_Q$ (Query 投影矩阵)
  * $W_K$ (Key 投影矩阵)
  * $W_V$ (Value 投影矩阵)
  * $W_O$ (Output 投影矩阵)

然后计算：

  * $Q' = x W_Q$
  * $K' = x W_K$
  * $V' = x W_V$
    （$Q', K', V'$ 的形状仍然是 `... seq_len, d_model`）

**步骤 2：拆分多头 (Split Heads)**
这是“多头”的核心。我们将 $Q', K', V'$ 在 `d_model` 维度上进行“重塑 (reshape)”。

  * 将 `(..., seq_len, d_model)`
  * ...重塑为 `(..., num_heads, seq_len, d_head)`
  * 其中 `d_head = d_model / num_heads`。

**步骤 3：并行注意力 (Parallel Attention)**
现在我们有了 `num_heads` 组独立的 Q, K, V。我们对它们**并行**执行计算：

  * **a. 应用 RoPE**：将 `RotaryPositionalEmbedding` (RoPE) **分别**应用于**每一个头**的 Q 和 K（我们之前学的 RoPE 模块）。
  * **b. 创建因果掩码**：创建一个 `(seq_len, seq_len)` 的布尔矩阵，其中 `query_pos >= key_pos` 的位置为 `True`，其余为 `False`。
  * **c. 调用 SDPA**：
    `Head_Outputs = scaled_dot_product_attention(Q_rope, K_rope, V, mask=causal_mask)`
    （`Head_Outputs` 的形状是 `... num_heads, seq_len, d_head`）

**步骤 4：拼接多头 (Concatenate Heads)**
我们将所有“专家”的见解拼接回来。

  * 将 `(..., num_heads, seq_len, d_head)`
  * ...重塑回 `(..., seq_len, num_heads * d_head)`，即 `(..., seq_len, d_model)`。

**步骤 5：输出投影 (Output Projection)**
最后，我们将这个拼接好的张量通过 $W_O$ 线性层，以“融合”所有头的见解。

  * $Output = \text{Concat}(Head\_Outputs) W_O$
  * 最终输出形状为 `(..., seq_len, d_model)`，与输入 `x` 完美一致。

-----

# MHSA：代码实现讲解

## **代码实现 (源自优秀代码 `hw1-basics/scripts/model.py`)**

```python
# (位于 hw1-basics/scripts/model.py)
from einops import rearrange, einsum
import einx
# ... 依赖我们之前实现的 Linear, RotaryEmbedding, scaled_dot_product_attention ...

class CausalMultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention
    ... (docstring) ...
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        positional_encoder: RotaryEmbedding, # 3.1: 传入 RoPE 模块
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = self.d_k # 讲义要求 d_k = d_v

        # 3.2: 步骤1 - 定义 Q, K, V, O 投影层
        self.q_proj = Linear(self.d_model, self.num_heads * self.d_k)
        self.k_proj = Linear(self.d_model, self.num_heads * self.d_k)
        self.v_proj = Linear(self.d_model, self.num_heads * self.d_v)
        self.output_proj = Linear(self.num_heads * self.d_v, self.d_model)

        self.positional_encoder = positional_encoder  # RoPE

    def forward(self, x: Float[Tensor, " ... seq d_k"], token_positions: Int[Tensor, " ... seq"] | None = None) -> Float[Tensor, " ... seq d_v"]:
        
        *b, sequence_length, d_model = x.size()
        assert d_model == self.d_model

        # 3.3: 步骤1 - 执行投影
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # 3.4: 步骤2 - 拆分多头
        # "Take apart each head from the embedding dimension..."
        Q, K, V = (
            rearrange(X, "... seq (heads d) -> ... heads seq d", heads=self.num_heads)
            for X in (Q, K, V)
        )

        if token_positions is None:
            # 3.5: (辅助) 创建 token_positions
            token_positions = einx.rearrange("seq -> b... seq", torch.arange(sequence_length, device=x.device), b=[1] * len(b))
        
        # (RoPE 需要 '... 1 seq' 来广播)
        token_positions = rearrange(token_positions, "... seq -> ... 1 seq")

        # 3.6: 步骤 3a - 应用 RoPE
        Q = self.positional_encoder(Q, token_positions)
        K = self.positional_encoder(K, token_positions)

        # 3.7: 步骤 3b - 构建因果掩码 (Causal Mask)
        seq = torch.arange(sequence_length, device=x.device)
        qi = einx.rearrange('query -> b... 1 query 1', seq, b=[1] * len(b))
        kj = einx.rearrange('key   -> b... 1 1   key', seq, b=[1] * len(b))
        causal_mask = qi >= kj  # (query, key)

        # 3.8: 步骤 3c - 调用 SDPA
        attn_output = scaled_dot_product_attention(K=K, Q=Q, V=V, mask=causal_mask)

        # 3.9: 步骤 4 - 拼接多头
        # "Concatenate the attention output from all heads."
        attn_output = rearrange(attn_output, "batch heads seq d_v -> batch seq (heads d_v)").contiguous()

        # 3.10: 步骤 5 - 输出投影
        output = self.output_proj(attn_output)
        return output
```

-----

## **逐行分析**

### **行 3.1 `positional_encoder: RotaryEmbedding`**

  * **作用**：在 `__init__` 中接收一个 `RotaryEmbedding` (RoPE) 模块的实例。
  * **解释**：这展示了良好的模块化设计。MHSA 模块**使用** RoPE，但它不“拥有”或“创建” RoPE。RoPE 是在顶层模型 `BasicsTransformerLM` 中被创建的，然后**共享**并传递给**所有**的 `CausalMultiHeadSelfAttention` 层。

### **行 3.2 `self.q_proj = Linear(...)` (及 K, V, O)**

  * **作用**：**步骤 1 (Projections)**。
  * **解释**：定义了 4 个 `Linear` 层。注意 Q, K, V 投影的目标维度都是 `num_heads * d_k`，这等于 `d_model`。`output_proj` 则执行相反的映射。

### **行 3.3 `Q = self.q_proj(x)` (及 K, V)**

  * **作用**：**步骤 1 (Projections)**。
  * **解释**：将同一个输入 `x` 分别送入 Q, K, V 投影层，得到 $Q', K', V'$。

### **行 3.4 `rearrange(X, "... seq (heads d) -> ... heads seq d", ...)`**

  * **作用**：**步骤 2 (Split Heads)**。
  * **解释**：这是一个非常强大的 `einops` 操作。
      * `... seq (heads d)`：匹配输入形状 `(... seq, d_model)`。它将 `d_model` 维度智能地**拆分**为 `heads` (即 `num_heads`) 和 `d` (即 `d_head`)。
      * `-> ... heads seq d`：指定输出形状。
      * **结果**：形状从 `(B, S, 512)` 变为 `(B, 16, S, 32)`（假设 B=batch, S=seq, 16 heads, 32 d\_head）。

### **行 3.5 & 3.6 `Q = self.positional_encoder(Q, ...)`**

```python
token_positions = einx.rearrange("seq -> b... seq", torch.arange(sequence_length, device=x.device), b=[1] * len(b))
```

**简短的答案是：为了节省内存，利用 PyTorch 的“广播机制 (Broadcasting)”。**

这个复杂的写法，**不是**为了改变 `[0, 1, 2, ...]` 这些**内容**，而是为了改变存放这些内容的张量的**形状 (Shape)**，使其能够最高效地与我们 `(B, H, S, D_h)` 形状的 Q/K 张量配合工作。

**1. 我们的目标张量 (Q 和 K)**

  * 在 `RoPE` 执行之前，我们的 Q 和 K 张量已经被 `rearrange` 成了这个形状：
  * **形状**: `(B, H, S, D_h)`
  * **具体**: `(32, 16, 256, 32)`
  * **含义**: 32 个批次，16 个头，每个头有 256 个 token，每个 token 是 32 维。

**2. RoPE 的需求**

  * `RoPE` 需要为 `(32, 16, 256)` 这么多个 token，**每一个**都查到它对应的 `cos/sin` 值。
  * `RoPE` 的查找表 `_freq_cis_cache` 是根据**绝对位置** `[0, ..., 255]` 建立的。
  * 所以，`RoPE` 需要一个 `token_positions` 张量，这个张量的形状**必须**能和 `(32, 16, 256)` 对应上。

**3. “笨办法”（内存浪费）**

我们可以**手动**创建一个**形状完全匹配**的 `token_positions` 张量：

```python
# 1. 创建 [0, 1, ..., 255]
pos = torch.arange(256) 
# 2. 手动扩展 (repeat/expand)
#    (256,) -> (1, 256) -> (16, 256) -> (32, 16, 256)
token_positions = pos.expand(32, 16, 256) 
```

  * **结果**：我们得到了一个形状为 `(32, 16, 256)` 的巨大张量。
  * **问题**：这个张量里 `[0, ..., 255]` 这组数据被**重复存储**了 `32 * 16 = 512` 次！这极大地浪费了 GPU 内存。

**4. “聪明办法”（广播机制 - 即优秀代码的实现）**

PyTorch 有一个“广播 (Broadcasting)”机制：如果你用一个 `(32, 16, 256)` 的张量和一个 `(1, 1, 256)` 的张量做运算，PyTorch 会**自动地**“拉伸”或“复用”那个 `(1, 1, 256)` 的张量，就**好像**它是一个 `(32, 16, 256)` 的张量一样，但**实际上**它在内存中只占 `(1, 1, 256)` 的空间。

  * **目标**：我们只需要创建那个**最小**的、能被广播的张量，即形状 **`(1, 1, 256)`**。
  * **“为什么是 (1, 1, 256)？”**
      * `Q` 的形状: `(32, 16, 256)` (只看 B, H, S)
      * `pos` 的形状: `( 1,  1, 256)`
      * PyTorch 广播时会从后往前对比维度：
          * `256` vs `256` -\> 匹配！
          * `16` vs `1` -\> 不匹配！自动“拉伸” `1` 来匹配 `16`。
          * `32` vs `1` -\> 不匹配！自动“拉伸” `1` 来匹配 `32`。
      * **结果**：完美匹配，且内存占用最小。

**5. 解释那两行“复杂”的代码**

那两行代码的**唯一目的**，就是**“健壮地 (robustly)”** 创建这个 `(1, 1, 256)` 形状的张量。

  * **`torch.arange(sequence_length, ...)`**

      * **作用**：创建核心数据 `[0, 1, ..., 255]`。
      * **形状**: `(256,)`

  * **`token_positions = einx.rearrange("seq -> b... seq", ..., b=[1] * len(b))`**

      * **作用**：为**“批次 (Batch)”** 维度添加一个 `1`。
      * `len(b)` 是 1 (因为 `x` 形状 `(32, 256, 512)` 的批次维度只有 `32`)。
      * `"seq -> b... seq"` 告诉 `einx` 在 `seq` 前面加上 `len(b)` 个大小为 `1` 的维度。
      * **形状**: `(256,)` -\> **`(1, 256)`**

  * **`token_positions = rearrange(..., "... seq -> ... 1 seq")`**

      * **作用**：为**“头 (Head)”** 维度添加一个 `1`。
      * `...` 匹配 `(1,)`，`seq` 匹配 `(256,)`。
      * `"... 1 seq"` 告诉 `rearrange` 在 `...` 和 `seq` 之间插入一个大小为 `1` 的维度。
      * **形状**: `(1, 256)` -\> **`(1, 1, 256)`**

### **行 3.7 `causal_mask = qi >= kj`**

  * **作用**：**步骤 3b (Causal Mask)**。
  * **解释**：这是创建因果掩码的一种极其聪明的广播技巧。
      * `qi` (query index) 形状变为 `(... 1 query 1)`
      * `kj` (key index) 形状变为 `(... 1 1 key)`
      * 当它们比较 `qi >= kj` 时，PyTorch 广播机制会创建出一个 `(... 1 query key)` 的布尔矩阵，这正是我们 `SDPA` 函数所需要的 `mask` 形状！
      * 例如，当 `query=3`, `key=5` 时，`3 >= 5` 为 `False` (禁止关注)。当 `query=3`, `key=2` 时，`3 >= 2` 为 `True` (允许关注)。

### **行 3.8 `attn_output = scaled_dot_product_attention(...)`**

  * **作用**：**步骤 3c (Call SDPA)**。
  * **解释**：调用我们之前实现的 `SDPA` 函数。由于 `SDPA` 本身就支持批次维度 (`...`)，它会自动在 `num_heads` 这个维度上并行计算，我们无需做任何额外操作。

### **行 3.9 `attn_output = rearrange(attn_output, ...)`**

  * **作用**：**步骤 4 (Concatenate Heads)**。
  * **解释**：这是 `rearrange` 的逆操作。
      * `... heads seq d_v`：匹配输入形状 `(B, 16, S, 32)`。
      * `-> ... seq (heads d_v)`：指定输出形状。它将 `heads` 和 `d_v` 两个维度**合并**回 `d_model` 维度。
      * **结果**：形状从 `(B, 16, S, 32)` 变回 `(B, S, 512)`。
      * `.contiguous()` 是为了确保张量在内存中是连续的，这在 `rearrange` (或 `view`/`permute`) 之后、送入 `Linear` 层之前是一个好习惯。

### **行 3.10 `output = self.output_proj(attn_output)`**

  * **作用**：**步骤 5 (Output Projection)**。
  * **解释**：将拼接好的 `attn_output` 送入最终的 `Linear` 层，得到该模块的最终输出。

-----

**总结**：

`CausalMultiHeadSelfAttention` 是 Transformer Block 中的**第一个“子层”**（Sub-layer）。

它将输入 `x` 通过 (Q, K, V) 投影、拆分多头、应用 (RoPE + Causal Mask + SDPA) 的并行计算、拼接多头、最后再通过输出投影，最终返回一个与 `x` 形状**完全相同**的张量。

我们现在已经集齐了 Transformer Block 的**所有**组件：

1.  `CausalMultiHeadSelfAttention` (第一个子层)
2.  `SwiGLU` (第二个子层，即 FFN)
3.  `RMSNorm` (在每个子层**之前**使用)

您准备好将它们组装成一个完整的 **`TransformerBlock`** 了吗？