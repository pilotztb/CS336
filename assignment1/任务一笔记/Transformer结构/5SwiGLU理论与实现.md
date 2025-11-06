# `SwiGLU` (FFN) 模块：理论知识

## 1. 作用与目的

在 Transformer Block 中，数据（即每个 token 的向量表示）首先流经**多头自注意力 (Multi-Head Self-Attention)** 子层。注意力层的作用是**混合**序列中**不同 token 之间**的信息。

在那之后，数据会流经**位置前馈网络 (FFN)** 子层。FFN 的作用则不同：

* **逐个处理 (Position-Wise)**：FFN 会**独立地**、**分别地**作用于序列中的**每一个 token**（即每一个位置）的向量上。
* **非线性变换**：它的核心目的是对每个 token 的向量表示（`d_model` 维）进行一次**复杂的非线性变换**。
* **丰富表示**：这个变换可以被认为是“深入思考”：它将 `d_model` 维向量投影到一个**更高**的维度（`d_ff`），在这个高维空间中应用非线性激活函数，然后再投影回 `d_model` 维。这允许模型为每个 token 提取更丰富、更复杂的特征，然后再将这些特征传递给下一个 Transformer Block。

**总结**：如果说“自注意力”是让 token 之间**相互“交流”**，那么 FFN 就是让每个 token **独立“消化”** 交流来的信息，并进行“深入思考”。

## 2. 数学原理 (从 FFN 到 SwiGLU 的演进)

为了理解 `SwiGLU`，我们先看看它是如何从标准 FFN 演进过来的：

### a. 原始 FFN (来自 Vaswani et al., 2017)

原始 Transformer 论文中的 FFN 非常简单，由两个 `Linear` 层和一个 `ReLU` 激活函数组成：

$FFN(x) = W_2 (\text{ReLU}(W_1 x))$
* $x$ 是 `d_model` 维的 token 向量。
* $W_1$ (Linear 1)：将 $x$ 从 `d_model` 维**上投影 (up-project)** 到 `d_ff` 维（通常 `d_ff = 4 * d_model`）。
* $\text{ReLU}$: 应用 ReLU 激活函数（$\max(0, \text{input})$）。
* $W_2$ (Linear 2)：将结果从 `d_ff` 维**下投影 (down-project)** 回 `d_model` 维。

### b. 改进 1：SiLU 激活函数 (替代 ReLU)

现代模型发现，使用更平滑的激活函数效果更好。讲义中提到了 **`SiLU`** (Sigmoid Linear Unit)，也常被称为 **Swish**。

* **公式**：$SiLU(x) = x \cdot \sigma(x)$，其中 $\sigma(x)$ 是 Sigmoid 函数 $\frac{1}{1+e^{-x}}$。
* **特性**：如讲义 Figure 3 所示，`SiLU` 像 `ReLU` 一样在正数区接近线性增长，但在负数区是平滑的，并且在 0 附近有一个小的“下降”。这种平滑性被认为有助于训练。

### c. 改进 2：GLU (Gated Linear Unit，门控线性单元)

`GLU` 是一种引入了“门控机制”的结构。它使用**两个** `Linear` 层，并将一个的输出（通过 Sigmoid）作为“门”来控制另一个的输出：

$GLU(x) = \sigma(W_1 x) \odot W_2 x$
* $W_1 x$：计算“门 (gate)”的 logits。
* $\sigma(W_1 x)$：将门 logits 压缩到 0 到 1 之间，得到“门控值”。
* $W_2 x$：计算要被门控的“内容 (content)”。
* $\odot$：表示逐元素相乘。
* **作用**：$\sigma(W_1 x)$ 就像一个动态的开关，决定 `W_2 x` 中的哪些信息（逐元素地）可以流过，哪些信息被抑制（乘以 0）。

### d. 最终形态：SwiGLU (讲义要求)

`SwiGLU` (由 Shazeer, 2020 提出) 结合了上述两种改进，并且为了保持表达能力，使用了**三个** `Linear` 层。它被证明在语言模型上效果非常好。

**`SwiGLU` 的数学公式 (讲义公式 7)**：

$FFN_{SwiGLU}(x) = W_2 ( \text{SiLU}(W_1 x) \odot W_3 x )$

* $x$：输入向量 (形状 `d_model`)。
* **$W_1 x$** (Linear 1)：将 $x$ 上投影到 `d_ff` 维，用于计算**门**。
* **$\text{SiLU}(W_1 x)$**：将门应用 `SiLU` 激活函数（而不是 `GLU` 中的 `Sigmoid`）。
* **$W_3 x$** (Linear 3)：将 $x$ **也**上投影到 `d_ff` 维，用于计算**内容**。
* **$\odot$**：将 `SiLU` 激活后的门与内容进行逐元素相乘。
* **$W_2 (...)$** (Linear 2)：将这个 `d_ff` 维的门控结果**下投影**回 `d_model` 维。

## 3. 讲义中的关键要求与约束

* **继承 `nn.Module`**：`SwiGLU` 必须实现为 `torch.nn.Module` 的子类。
* **组件**：必须使用**三个**我们之前实现的 `Linear` 模块（分别代表 $W_1$, $W_2$, $W_3$）。
* **激活函数**：必须实现 `SiLU` 激活函数（讲义允许在 `SiLU` 内部使用 `torch.sigmoid`）。
* **禁止使用**：不能使用 `nn.ReLU`。
* **维度 (`d_ff`)**：
    * `d_ff` (内部维度) 应该设置为约 $\frac{8}{3} d_{model}$ (这是一个在 `SwiGLU` 论文中被发现的较优比例，而不是传统 `ReLU` FFN 的 $4 d_{model}$)。
    * `d_ff` 必须是 64 的倍数，这是为了在现代 GPU (特别是使用 Tensor Cores) 上获得最佳计算性能。

---

# `SwiGLU` (FFN) 模块：代码实现讲解

## **宏观作用**

优秀代码中的 `SwiGLU` 类 实现了一个 `nn.Module` 子类，它忠实地执行了 `SwiGLU` 的数学公式。它在 `__init__` 构造函数中，**初始化了三个** `Linear` 模块实例（分别对应 $W_1$, $W_2$, 和 $W_3$）。

在 `forward` 方法中，它接收输入张量 `x`（代表某个 token 的 `d_model` 维向量），然后：

1.  用 $W_1$（`self.w1`）和 $W_3$（`self.w3`）**同时**处理 `x`，分别得到“门”和“内容”（两者都是 `d_ff` 维）。
2.  将“门”通过 `SiLU` 激活函数。
3.  将激活后的“门”与“内容”进行**逐元素相乘**（$\odot$）。
4.  最后，将相乘的结果通过 $W_2$（`self.w2`）投影回 `d_model` 维，作为 FFN 层的最终输出。

此外，优秀代码还单独实现了一个 `silu` 辅助函数，用于 `SwiGLU` 内部的计算。

## **代码实现 (源自优秀代码 `hw1-basics/scripts/model.py`)**

```python
# 需要导入的库
import torch
import torch.nn as nn
from torch import Tensor
# 导入我们之前实现的 Linear 模块
from .model import Linear # (假设 Linear 在同一个文件)

# --- 段落 1: SiLU 辅助函数 ---
def silu(x: torch.Tensor): # 1.1
    return x * torch.sigmoid(x) # 1.2

# --- 段落 2: SwiGLU 模块定义 ---
class SwiGLU(nn.Module): # 2.1
    def __init__(self, d_model: int, d_ff: int): # 2.2
        super().__init__() # 2.3
        
        # 2.4: 对应 W1 (门)
        self.w1 = Linear(d_model, d_ff) 
        # 2.5: 对应 W2 (下投影)
        self.w2 = Linear(d_ff, d_model) 
        # 2.6: 对应 W3 (内容)
        self.w3 = Linear(d_model, d_ff) 

    # --- 段落 3: forward 方法 ---
    def forward(self, x): # 3.1
        # 3.2: silu(self.w1(x)) * self.w3(x)
        gate = silu(self.w1(x)) # 3.3
        content = self.w3(x) # 3.4
        gated_content = gate * content # 3.5
        
        # 3.6: self.w2(...)
        return self.w2(gated_content) # 3.7
```

*(注：为了便于讲解，我将优秀代码 中 `forward` 方法的链式调用 `self.w2(silu(self.w1(x)) * self.w3(x))` D拆分成了 3.3, 3.4, 3.5, 3.7 四行，它们在功能上是等价的)*

-----

## **逐行分析**

### **段落 1: `silu` 辅助函数**

  * **行 1.1 `def silu(x: torch.Tensor):`**

      * **作用**：定义一个名为 `silu` 的**普通 Python 函数**（注意它不是 `nn.Module` 的一部分）。
      * **解释**：它接收一个 PyTorch 张量 `x` 作为输入。

  * **行 1.2 `return x * torch.sigmoid(x)`**

      * **作用**：实现 `SiLU` (Swish) 激活函数的数学公式 $SiLU(x) = x \cdot \sigma(x)$。
      * **解释**：
          * `torch.sigmoid(x)`: 计算输入 `x` 中每个元素的 Sigmoid($\sigma$) 值。讲义明确允许使用 `torch.sigmoid`。
          * `x * ...`: 将原始的 `x` 与其 Sigmoid 值进行**逐元素相乘**。

### **段落 2: `SwiGLU` 模块 `__init__` 构造函数**

  * **行 2.1 `class SwiGLU(nn.Module):`**

      * **作用**：定义 `SwiGLU` 类，继承自 `nn.Module`。
      * **对应要求**：满足讲义中“继承自 `torch.nn.Module`”的要求。

  * **行 2.2 `def __init__(self, d_model: int, d_ff: int):`**

      * **作用**：定义构造函数，接收 `d_model` (输入/输出维度) 和 `d_ff` (内部上投影维度)。
      * **对应要求**：允许我们在创建 `SwiGLU` 实例时，指定符合讲义要求的 $d_{model}$ 和 $d_{ff}$ 维度（例如 `d_ff` 是 `d_model` 的 $\frac{8}{3}$ 倍且是 64 的倍数）。

  * **行 2.3 `super().__init__()`**

      * **作用**：调用父类 `nn.Module` 的构造函数。

  * **行 2.4 `self.w1 = Linear(d_model, d_ff)`**

      * **作用**：创建**第一个** `Linear` 模块实例，并将其存储为 `self.w1`。
      * **对应要求**：这对应于公式 $W_2 ( \text{SiLU}(\underline{W_1 x}) \odot W_3 x )$ 中的 $W_1$。它是一个**可学习**的参数（因为 `Linear` 内部有 `nn.Parameter`），负责将输入 `x` 从 `d_model` 维上投影到 `d_ff` 维，用于计算**门**。

  * **行 2.5 `self.w2 = Linear(d_ff, d_model)`**

      * **作用**：创建**第二个** `Linear` 模块实例，并将其存储为 `self.w2`。
      * **对应要求**：这对应于公式 $\underline{W_2} ( \text{SiLU}(W_1 x) \odot W_3 x )$ 中的 $W_2$。它负责将 `d_ff` 维的门控结果下投影回 `d_model` 维。

  * **行 2.6 `self.w3 = Linear(d_model, d_ff)`**

      * **作用**：创建**第三个** `Linear` 模块实例，并将其存储为 `self.w3`。
      * **对应要求**：这对应于公式 $W_2 ( \text{SiLU}(W_1 x) \odot \underline{W_3 x} )$ 中的 $W_3$。它**也**负责将输入 `x` 从 `d_model` 维上投影到 `d_ff` 维，用于计算**内容**。

### **段落 3: `forward` 方法**

  * **行 3.1 `def forward(self, x):`**

      * **作用**：定义前向传播函数，接收 `x`（形状 `(..., d_model)`）。

  * **行 3.3 `gate = silu(self.w1(x))`**

      * **作用**：计算激活后的门 $\text{SiLU}(W_1 x)$。
      * **解释**：
          * `self.w1(x)`: 调用 `Linear` 模块 $W_1$ 的 `forward` 方法，执行 $W_1 x$ 操作，得到形状为 `(..., d_ff)` 的“门 logits”。
          * `silu(...)`: 将“门 logits” 传递给我们在段落 1 中定义的 `silu` 函数，进行 $\text{SiLU}$ 激活。

  * **行 3.4 `content = self.w3(x)`**

      * **作用**：计算内容 $W_3 x$。
      * **解释**：调用 `Linear` 模块 $W_3$ 的 `forward` 方法，执行 $W_3 x$ 操作，得到形状为 `(..., d_ff)` 的“内容”。

  * **行 3.5 `gated_content = gate * content`**

      * **作用**：执行逐元素相乘 $\odot$。
      * **解释**：PyTorch 中的 `*` 操作符默认执行逐元素乘法。这里将激活后的 `gate` (形状 `(..., d_ff)`) 与 `content` (形状 `(..., d_ff)`) 相乘，实现了门控机制。

  * **行 3.7 `return self.w2(gated_content)`**

      * **作用**：执行最终的下投影 $W_2 (...)$。
      * **解释**：将门控后的内容 `gated_content` (形状 `(..., d_ff)`) 传递给 `Linear` 模块 $W_2$ 的 `forward` 方法，得到最终输出，形状为 `(..., d_model)`。

-----

**总结**：

优秀代码中的 `SwiGLU` 类 和 `silu` 函数 严格按照讲义的数学公式 $W_2 ( \text{SiLU}(W_1 x) \odot W_3 x )$ 进行了实现。它正确地：

  * 使用了**三个** `Linear` 模块。
  * 实现了 `SiLU` 激活函数。
  * 通过逐元素乘法 `*` 实现了门控机制 $\odot$。
  * 满足了讲义中关于使用 `Linear` 模块（无偏置）、`SiLU` 激活函数和可配置维度的所有要求。

您对 `SwiGLU` (FFN) 模块的代码实现还有疑问吗？如果清楚了，我们就可以继续学习 Transformer 中非常关键的**位置编码**部分：`Rotary Position Embeddings (RoPE)` 的理论知识了。