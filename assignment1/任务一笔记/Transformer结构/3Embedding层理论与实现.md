# `Embedding` 模块：理论知识

## 1. 作用与目的

`Embedding` 模块（嵌入层）是 Transformer 语言模型的**入口**，扮演着至关重要的**桥梁**角色。它的核心任务是将我们通过分词器（如 BPE）得到的**离散的**、**类别化的**输入（即 token ID，本质上是整数）转换为神经网络能够有效处理的**连续的**、**低维度的**、**密集的**向量表示（称为嵌入向量）。

* **为什么需要嵌入层？**
    * **Token ID 的局限性**：分词器将文本（如单词 "cat"）映射到一个整数 ID（例如 `5078`）。这些 ID 本身是任意分配的，它们之间的数值关系（例如 `5078` 和 `5079`）通常**不包含任何语义信息**（"cat" 和 ID 为 `5079` 的词在意义上可能毫无关联）。直接将这些离散且无序的 ID 输入神经网络效果会很差。
    * **高维稀疏 vs 低维密集**：如果使用 one-hot 编码（创建一个长度等于词汇表大小，只有一个位置是 1，其余都是 0 的向量）来表示每个 token ID，虽然包含了类别信息，但会导致输入维度极其巨大（等于词汇表大小，可能有数万甚至数十万维）且非常稀疏（只有一个 1）。这在计算上效率低下，且难以让模型学习到词与词之间的关系。
    * **学习语义表示**：`Embedding` 层通过为**每个** token ID 关联一个**可学习的**向量（例如 512 维），解决了上述问题。这个向量（嵌入向量）的**数值**是在模型训练过程中通过反向传播**学习**得到的。训练的目标会促使意义相近或用法相似的词（例如 "cat" 和 "dog"，或者 "run" 和 "ran"）拥有**相似**的嵌入向量（在向量空间中彼此靠近）。这样，嵌入向量就**捕获并编码**了 token 的**语义信息**和**上下文关系**。
    
* **在 Transformer 中的角色**：
    * 作为模型处理 token ID 的**第一层**（见讲义 Figure 1）。
    
    * 接收的输入是一个形状为 `(batch_size, sequence_length)` 的张量，其中包含整数 token ID。
    
      * 关于上面这句话的展开讲解
    
        好的，我们来详细解释这句话：
    
        > "接收的输入是一个形状为 `(batch_size, sequence_length)` 的张量，其中包含整数 token ID。"
    
        这句话描述的是 `Embedding` 层在训练（或推理）时**接收的数据是什么样子**。让我们把它拆解开来：
    
        1.  **张量 (Tensor)**：
    
              * 在 PyTorch (以及 TensorFlow, NumPy 等库) 中，张量是用来表示多维数组的主要数据结构。
              * 你可以把它想象成一个（可能是多维的）矩形网格，里面装着数字。
              * 0 维张量是一个标量（单个数字）。
              * 1 维张量是一个向量（一行数字）。
              * 2 维张量是一个矩阵（二维表格）。
              * 3 维及以上张量就是更高维度的数组。
    
        2.  **形状为 `(batch_size, sequence_length)`**：
    
              * 这描述了这个张量是一个**二维**的数组（一个矩阵）。
              * **第一个维度 (`batch_size`)**：代表**批次大小**。为了提高训练效率，我们通常不会一次只处理一个文本序列，而是将多个序列**打包**在一起，形成一个“批次 (batch)”。`batch_size` 就是这个批次中包含的**序列的数量**。例如，`batch_size=32` 意味着我们一次处理 32 个不同的文本序列。
              * **第二个维度 (`sequence_length`)**：代表**序列长度**。在 Transformer 模型中，我们通常将输入文本处理成固定长度的片段。`sequence_length` 就是**每个**文本序列片段包含的\*\* token 的数量\*\*。例如，`sequence_length=256` 意味着每个序列片段都有 256 个 token。
    
        3.  **其中包含整数 token ID**：
    
              * 这说明了这个二维张量（矩阵）里面**存储**的是什么类型的数字。
              * **Token ID**：是 BPE 分词器（您之前实现的）将原始文本转换后的结果。每个单词或子词都被映射成了一个**唯一的整数 ID**。例如，"Hello" 可能对应 ID `15496`，"world" 可能对应 ID `995`。
              * **整数**: 这些 ID 都是整数（`int` 类型，在 PyTorch 中通常是 `torch.LongTensor`）。
              * **张量内容**: 因此，这个形状为 `(batch_size, sequence_length)` 的张量，它的**每一个**位置上都存着一个**整数**，这个整数就是**对应批次**中**对应序列位置**的那个 token 的 ID。
    
        **综合示例：**
    
        假设 `batch_size = 2`，`sequence_length = 5`。`Embedding` 层接收的输入张量 `I` 可能是这样的：
    
        ```
        I = [
              [ 15496,   995,    62,   198, 50256 ],  # 第一个序列 (包含 5 个 token ID)
              [   502,  4967,   649,   25, 50256 ]   # 第二个序列 (包含 5 个 token ID)
            ]
        # 这个张量的形状是 (2, 5)
        ```
    
          * 这个张量有 2 行 (`batch_size`)，5 列 (`sequence_length`)。
          * 每一行代表一个独立的文本序列片段。
          * 每个单元格里的数字（例如 `15496`, `995`）就是一个整数 token ID。
    
        `Embedding` 层接收到这个 `(2, 5)` 的整数张量后，就会去查找每个 ID 对应的嵌入向量，最终输出一个形状为 `(2, 5, d_model)` 的浮点数张量。
    
        希望这个解释能帮助您理解 `Embedding` 层的输入是什么样子了！
    
    * 输出一个形状为 `(batch_size, sequence_length, d_model)` 的张量，其中 `d_model` 是嵌入向量的维度。这个输出张量包含了每个输入 token ID 对应的、学习到的密集向量表示，然后才被送入后续的 Transformer Block。
    
      * 关于上面的详细讲解
    
        好的，我们来详细讲解 `Embedding` 层输出的这句话：
    
        > "输出一个形状为 `(batch_size, sequence_length, d_model)` 的张量，其中 `d_model` 是嵌入向量的维度。这个输出张量包含了每个输入 token ID 对应的、学习到的密集向量表示，然后才被送入后续的 Transformer Block。"
    
        这句话描述了 `Embedding` 层执行完它的“查找”任务后，**产生的结果是什么样子**以及**结果的含义**。
    
        -----
    
        **逐个部分详细解释**
    
        1.  **"输出一个 ... 张量"**：
    
              * `Embedding` 层的计算结果仍然是一个 PyTorch 张量（多维数组）。
    
        2.  **"形状为 `(batch_size, sequence_length, d_model)`"**：
    
              * 这描述了这个输出张量的**维度**和**大小**。它是一个**三维**张量。
              * **`batch_size` (第一个维度)**：这个维度的大小与输入张量的第一个维度**完全相同**。它代表批次中有多少个独立的序列。
              * **`sequence_length` (第二个维度)**：这个维度的大小也与输入张量的第二个维度**完全相同**。它代表每个序列中有多少个 token 位置。
              * **`d_model` (第三个维度)**：这是**新增加**的一个维度，也是 `Embedding` 层最核心的变化。`d_model` 是您在创建 `Embedding` 层时指定的**嵌入向量的维度**（例如 512）。
              * **对比输入**：输入张量的形状是 `(batch_size, sequence_length)`，它是一个二维张量，每个位置只有一个整数 ID。输出张量是三维的，输入中的**每一个整数 ID** 都被**替换**成了一个长度为 `d_model` 的**向量**。
    
        3.  **"其中 `d_model` 是嵌入向量的维度"**：
    
              * 再次强调了第三个维度的大小 `d_model` 代表了**每一个** token ID 被映射成的那个**向量的长度**。这个长度是模型的一个超参数，决定了模型用来表示每个词的“丰富程度”。
    
        4.  **"这个输出张量包含了每个输入 token ID 对应的、学习到的密集向量表示"**：
    
              * **"每个输入 token ID 对应的"**：输出张量 `X_emb` 和输入张量 `I` 在前两个维度上是一一对应的。`X_emb[b, t, :]` 这个向量，就是由输入 `I[b, t]` 这个 ID 通过查找嵌入矩阵得到的。
              * **"学习到的"**: 这些 `d_model` 维的向量不是固定的，它们是 `Embedding` 层的**可训练参数**（存储在嵌入矩阵 $W_e$ 中）。在模型训练过程中，这些向量的值会通过反向传播和梯度下降不断**调整和优化**，以便更好地捕捉词的语义和用法。
              * **"密集向量表示 (dense vector representation)"**: 这与之前提到的 one-hot 编码（稀疏表示）相对。
                  * **稀疏 (Sparse)**：像 one-hot 向量那样，绝大部分元素都是 0，只有很少（一个）非零元素。
                  * **密集 (Dense)**：嵌入向量通常包含**许多**非零的浮点数值，几乎没有 0。这些数值共同编码了词的丰富语义信息。`d_model` 维向量中的**每一个**元素都可能对表示词的意义有所贡献。
    
        5.  **"然后才被送入后续的 Transformer Block"**：
    
              * 这句话点明了 `Embedding` 层的**位置和作用**。它产生的这个 `(batch_size, sequence_length, d_model)` 形状的、包含密集语义向量的张量，就是**后续 Transformer 核心层（Transformer Block）所需要的输入格式**。Transformer Block 内部的自注意力、前馈网络等操作都是在这个 `d_model` 维的向量空间上进行的。
    
        **综合示例（续）：**
    
        假设输入 `I` 是：
    
        ```
        I = [
             [ 15496,    995,     62 ],  # batch 0
             [   502,   4967,    649 ]   # batch 1
            ]
        # 形状 (2, 3)
        ```
    
        并且 `d_model = 4`（为了简化）。`Embedding` 层内部有一个 `(vocab_size, 4)` 的嵌入矩阵 `W_e`。
    
        那么输出张量 `X_emb` 的形状将是 `(2, 3, 4)`，它看起来可能是这样的（数值是示意）：
    
        ```
        X_emb = [
                 # batch 0
                 [ # sequence pos 0 (ID 15496 -> vec0)
                   [ 0.1, -0.2,  0.5,  0.9 ],
                   # sequence pos 1 (ID 995 -> vec1)
                   [ 0.8,  0.1, -0.3,  0.4 ],
                   # sequence pos 2 (ID 62 -> vec2)
                   [ -0.5, 0.6,  0.7, -0.1 ]
                 ],
                 # batch 1
                 [ # sequence pos 0 (ID 502 -> vec3)
                   [ 0.3,  0.4,  0.1, -0.6 ],
                   # sequence pos 1 (ID 4967 -> vec4)
                   [ -0.7, 0.9, -0.2,  0.0 ],
                   # sequence pos 2 (ID 649 -> vec5)
                   [ 0.2, -0.8,  0.3,  0.5 ]
                 ]
               ]
        # 形状 (2, 3, 4)
        ```
    
          * `X_emb[0, 0, :]` (即 `[ 0.1, -0.2, 0.5, 0.9 ]`) 就是 ID `15496` 对应的 4 维嵌入向量 (`vec0`)。
          * `X_emb[1, 2, :]` (即 `[ 0.2, -0.8, 0.3, 0.5 ]`) 就是 ID `649` 对应的 4 维嵌入向量 (`vec5`)。
    
        这个 `(2, 3, 4)` 的张量 `X_emb` 随后会被传递给第一个 Transformer Block 进行处理。

## 2. 数学原理：查找表 (Lookup Table)

`Embedding` 层的核心原理非常简单：它本质上是一个大型的**查找表 (Lookup Table)**。

1.  **嵌入矩阵 (Embedding Matrix)**：
    * 模块内部的核心是一个巨大的**权重矩阵**，通常称为 $W_e$。
    * $W_e$ 的形状是 `(vocab_size, d_model)`，其中 `vocab_size` 是词汇表的大小（例如 10000），`d_model` 是嵌入向量的维度（例如 512）。
    * 这个矩阵可以被看作是有 `vocab_size` 行，每一行是一个长度为 `d_model` 的向量。
    * **关键**：$W_e$ 矩阵的**第 `i` 行**，就代表词汇表中 token ID 为 `i` 的那个词所对应的**嵌入向量**。这个矩阵的所有元素都是**可学习的参数**。

2.  **查找操作**：
    * 当 `Embedding` 层接收到一个**单个** token ID（例如整数 `5078`）时，它的操作就是**直接取出** $W_e$ 矩阵的**第 5078 行**作为输出向量。
    * 当输入是一个**批处理**的 token ID 张量 `I`（形状 `(batch_size, sequence_length)`）时，`Embedding` 层会并行地对 `I` 中的**每一个**整数 ID 执行这个查找操作。
    * 输出结果 `X` 是一个新的张量，形状为 `(batch_size, sequence_length, d_model)`，其中 `X[b, t, :]` 就是输入张量中 `I[b, t]` 这个 ID 对应的嵌入向量（即 $W_e$ 的第 `I[b, t]` 行）。
    * 这个查找操作在 PyTorch 中可以通过高效的**张量索引 (tensor indexing)** 实现：`X = self.weight[I]`，其中 `self.weight` 就是嵌入矩阵 $W_e$。

## 3. 讲义中的关键要求与约束

* **继承 `nn.Module`**：必须作为 `torch.nn.Module` 的子类来实现。
* **权重参数 (Embedding Matrix `weight`)**：
    * 必须存储为一个 `nn.Parameter` 对象，形状为 `(num_embeddings, embedding_dim)` (即 `vocab_size`, `d_model`)。讲义建议使用 `num_embeddings` 和 `embedding_dim` 作为 `__init__` 的参数名。
    * 存储时，`d_model` (嵌入维度) 应该是**最后一个**维度。
* **权重初始化**：
    * 必须使用**截断正态分布**进行初始化。
    * 讲义 §3.4.1 明确规定 `Embedding` 的初始化参数为：$\mathcal{N}(\mu=0, \sigma^2=1)$（即标准差 **`std=1.0`**），截断范围为 $[-3\sigma, 3\sigma]$ (即 `a=-3.0`, `b=3.0`)。
* **`forward` 方法**：
    * 接收的输入 `token_ids` 必须是 `torch.LongTensor`（长整型张量）。
    * 输入张量可以有任意数量的前导维度 (`...`)。
    * 返回的输出张量是浮点类型，形状为 `(..., embedding_dim)`。
    * 实现方式是通过**张量索引**完成查找。
* **禁止使用内置实现**：**绝对不能**使用 `torch.nn.Embedding` 或 `torch.nn.functional.embedding`。

---

好的，理论部分我们已经清楚了。`Embedding` 层就是一个大型的、可学习的**查找表**，它将整数 ID 映射为向量。

现在，我们来看一下“优秀代码”是如何根据这些理论和要求来实现 `Embedding` 模块的。

-----

# `Embedding` 模块：代码实现讲解

## **代码实现 (源自优秀代码 `hw1-basics/scripts/model.py`)**

```python
# 需要导入的库 (除了之前的)
import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Float, Int # 需要 Int 类型提示

class Embedding(nn.Module): # 1.1: 继承 nn.Module
    def __init__(self, d_model: int, vocab_size: int): # 1.2: 构造函数签名
        """
        An embedding layer initialized with truncated normal distribution (std=1.0).
        """ # 1.3: 文档字符串
        
        super().__init__() # 1.4: 调用父类构造函数
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # --- 段落 2: 参数初始化 ---
        std = 1.0 # 2.1: 按照讲义要求，标准差为 1.0
        # 2.2: 创建空的权重张量 (嵌入矩阵)
        weight_tensor = torch.empty(self.vocab_size, self.d_model) 
        # 2.3: 使用截断正态分布填充张量 (in-place 操作)
        nn.init.trunc_normal_(weight_tensor, std=std, a=-3*std, b=3*std) 
        # 2.4: 将张量包装成 nn.Parameter 并存储
        self.weight: Float[Tensor, " vocab_size d_model"] = nn.Parameter(
            weight_tensor, requires_grad=True
        )

    # --- 段落 3: 前向传播 ---
    def forward(self, x: Int[Tensor, " ..."]) -> Float[Tensor, " ... d_model"]: # 3.1: forward 函数签名
        # 3.2: 执行索引查找
        return self.weight[x] # 3.3

    # --- 段落 4: (可选) 辅助方法 ---
    def extra_repr(self): # 4.1: 定义对象的字符串表示
        return f"vocab_size={self.vocab_size}, d_model={self.d_model}" # 4.2
```

-----

## **逐行分析**

### **段落 1: 类定义与构造函数基础**

  * **行 1.1 `class Embedding(nn.Module):`**

      * **作用**：定义 `Embedding` 类并指定它继承自 `nn.Module`。
      * **对应要求**：满足讲义中“继承自 `torch.nn.Module`”的要求。

  * **行 1.2 `def __init__(self, d_model: int, vocab_size: int):`**

      * **作用**：定义构造函数。
      * **对应要求**：满足讲义推荐的接口，`d_model` 对应 `embedding_dim` (嵌入维度)，`vocab_size` 对应 `num_embeddings` (词汇表大小)。 (注：优秀代码将讲义中的 `num_embeddings` 命名为 `vocab_size`，`embedding_dim` 命名为 `d_model`，这很常见)。

  * **行 1.4 `super().__init__()`**

      * **作用**：调用父类 `nn.Module` 的构造函数。
      * **对应要求**：继承 `nn.Module` 时的标准做法。

### **段落 2: 参数初始化**

  * **行 2.1 `std = 1.0`**

      * **作用**：设置初始化所需的标准差 $\sigma$。
      * **对应要求**：严格按照讲义 §3.4.1 的要求，为 `Embedding` 层设置 $\sigma=1.0$。

  * **行 2.2 `weight_tensor = torch.empty(self.vocab_size, self.d_model)`**

      * **作用**：创建嵌入矩阵 $W_e$ 的张量。
      * **对应要求**：权重矩阵的形状要求是 `(vocab_size, d_model)` (即 `num_embeddings`, `embedding_dim`)。

  * **行 2.3 `nn.init.trunc_normal_(weight_tensor, std=std, a=-3*std, b=3*std)`**

      * **作用**：使用截断正态分布原地填充 `weight_tensor`。
      * **对应要求**：严格按照讲义要求进行初始化：使用 `trunc_normal_`，`std=1.0`，截断范围 `a` (下界) 和 `b` (上界) 设为 `-3.0` 和 `3.0`。

  * **行 2.4 (`self.weight: ... = nn.Parameter(...)`)**

      * **作用**：将初始化好的嵌入矩阵注册为模型的可训练参数 `self.weight`。
      * **对应要求**：满足“存储为 `nn.Parameter`”的要求。

### **段落 3: 前向传播**

  * **行 3.1 `def forward(self, x: Int[Tensor, " ..."]) -> Float[Tensor, " ... d_model"]:`**

      * **作用**：定义前向传播函数。
      * **对应要求**：
          * 输入 `x` 的类型提示为 `Int[Tensor, " ..."]`，表明它是一个包含任意前导维度的**整数张量**（即 token ID），这符合 `torch.LongTensor` 的要求。
          * 输出类型提示为 `Float[Tensor, " ... d_model"]`，表明返回的是浮点型向量，并且在 `x` 的维度基础上追加了 `d_model` 维度。

  * **行 3.3 `return self.weight[x]`**

      * **作用**：执行核心的**索引查找**操作。
      * **对应要求**：
          * 这**不是**函数调用，而是利用了 PyTorch 的张量索引功能。
          * `self.weight` 是 `(vocab_size, d_model)` 的矩阵，`x` 是 `(...)` 的整数张量。
          * `self.weight[x]` 会自动取出 `x` 中每个 ID 对应的 `self.weight` 矩阵的**行**，并构建一个形状为 `(..., d_model)` 的新张量。
          * 这完美地实现了 `Embedding` 的查找表功能，并且**没有**使用被禁止的 `nn.Embedding` 或 `nn.functional.embedding`。

### **段落 4: (可选) 辅助方法**

  * **行 4.1 & 4.2 `def extra_repr(self): ...`**
      * **作用**：自定义对象的字符串表示，方便打印和调试。

-----

**总结**：

`Embedding` 模块的代码实现甚至比 `Linear` 还要简洁。它完美地体现了“查找表”这一理论：

  * `__init__` 负责构建并初始化这个查找表 (`self.weight`)。
  * `forward` 负责执行查找操作 (`self.weight[x]`)。

您对 `Embedding` 模块的代码实现还有疑问吗？如果清楚了，我们就可以继续学习下一个组件：`RMSNorm` 模块的理论知识了。