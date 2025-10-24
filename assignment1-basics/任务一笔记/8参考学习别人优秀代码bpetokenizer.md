# 任务需求：实现 `BPETokenizer` 类

## **Key English Content (来自 `cs336_spring2025_assignment1_basics.pdf`)**

* **Deliverable**: "Implement a `Tokenizer` class that... encodes text into integer IDs and decodes integer IDs into text."
* **Required Interface**: "We recommend the following interface:"
    * `def __init__(self, vocab, merges, special_tokens=None)`: "Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens."
    * `def encode(self, text: str) -> list[int]`: "Encode an input text into a sequence of token IDs."
    * `def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]`: "Given an iterable of strings... return a generator that lazily yields token IDs."
    * `def decode(self, ids: list[int]) -> str`: "Decode a sequence of token IDs into text."
* **Encoding Logic**: "First, our pre-tokenizer would split this string... Then, we'll look at each pre-token and apply the BPE merges... in the same order of creation."
* **Decoding Logic**: "simply look up each ID's corresponding entries in the vocabulary (a byte sequence), concatenate them together, and then decode the bytes to a Unicode string... using `errors='replace'`"

-----

## **通俗易懂的中文翻译与讲解**

### **核心任务 (Deliverable)**

你需要实现一个 `BPETokenizer` **类**。这个类的实例（对象）在创建时会接收我们上一步训练好的 `vocab` 和 `merges`，然后它就能提供 `.encode()` 和 `.decode()` 两个核心方法。

### **类需要实现的方法 (Interface)**

1.  `__init__(self, vocab, merges, special_tokens=None)`:

    * 这是类的**构造函数**。
    * 它在 `new BPETokenizer(...)` 时被调用。
    * 它的任务是接收 `vocab` 和 `merges`，并把它们**存储**起来。
    * **关键优化**：我们还会在这一步**预先计算**一些辅助数据（比如“合并优先级”字典），让 `encode` 方法能跑得飞快。

2.  `encode(self, text: str) -> list[int]`:

    * **编码**方法。
    * 接收一个普通字符串（` "Hello <|endoftext|>"  `），返回一个整数ID列表（`[15496, 220, 50256]`）。

3.  `encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]`:
    * **流式编码**方法。
    * 它接收一个**迭代器**（比如一个打开的文件句柄），并**逐个**（lazily）地产出 token ID。
    * 这是为了能处理G/T级别的超大文件，而不会撑爆内存。
    
4.  `decode(self, ids: list[int]) -> str`:
    * **解码**方法。
    * 接收一个整数ID列表（`[15496, 220, 50256]`），返回一个普通字符串（` "Hello <|endoftext|>"  `）。

-----

# 第零步：准备工作 (代码框架)

和上次一样，我们先搭好框架。我们将重用上次的 `import` 和 `train_bpe` 函数，并在下面添加 `BPETokenizer` 类的框架。

```python
from collections import defaultdict, Counter
from dataclasses import dataclass
import regex as re
from typing import Dict, List, Tuple, Optional, Iterable, Any # <-- 确保导入了 Optional, Iterable, Any
import os

# --- BPE 训练函数 (来自您的上一份笔记) ---
# ... (这里省略 _get_initial_word_freqs, _get_pair_freqs, _merge_word_freqs, train_bpe 的代码) ...
# ... 假设 train_bpe 函数已经在这里定义好了 ...

# --- 这是我们本次要实现的主体 ---

# 训练时使用的 PAT，在编码时也必须使用完全相同的规则
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class BPETokenizer:
    """
    一个完整的BPE分词器实现，包含编码(encode)和解码(decode)。
    """
    
    def __init__(
        self, 
        vocab: Dict[int, bytes], 
        merges: List[Tuple[bytes, bytes]], 
        special_tokens: Optional[List[str]] = None
    ):
        """
        初始化分词器。
        """
        # (代码将在第一步填充)
        pass

    def encode(self, text: str) -> List[int]:
        """
        将一个原始字符串编码为一个 token ID 列表。
        """
        # (代码将在第三步填充)
        return []

    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        """
        将一个字符串迭代器（如文件）编码为 token ID 迭代器。
        """
        # (代码将在第四步填充)
        pass

    def decode(self, ids: List[int]) -> str:
        """
        将一个 token ID 列表解码回一个字符串。
        """
        # (代码将在第二步填充)
        return ""
```

-----

好的，非常抱歉上次的解释与代码不一致，给您带来了困扰。现在我们严格按照您提供的“优秀代码” (`adapters.py` 中的 `Tokenizer.__init__`) 来重新梳理 `__init__` 方法的学习笔记。

-----

# 第一步：实现 `__init__` (构造与预计算)

## **目标**

实现 `BPETokenizer` 类的 `__init__` 构造函数。

## **逻辑**

`__init__` 方法在创建一个 `BPETokenizer` 对象时被调用 (例如 `tokenizer = BPETokenizer(vocab, merges, special_tokens)` )。它的核心任务是接收训练好的 `vocab` (词汇表) 和 `merges` (合并规则)，并将它们以及 `special_tokens` 存储起来，同时进行一些**预计算**，以便后续的 `encode` (编码) 和 `decode` (解码) 方法能够高效运行。

具体步骤如下：

1.  **存储基础信息**：将传入的 `vocab`, `merges`, 和 `special_tokens` 保存到 `self` 对象中 (即实例变量)。
2.  **创建反向词汇表 (`byte_to_token_id`)**：`vocab` 是从 ID (整数) 到 Bytes (字节) 的映射。为了在 `encode` 时能快速地根据 Bytes 找到对应的 ID，我们预先计算并存储一个反向的字典，即从 Bytes 到 ID 的映射。这对应您笔记中的**（优化2）**。
3.  **创建合并优先级 (`bpe_ranks`)**：`merges` 是一个**有序**列表，其顺序代表了合并的优先级 (越靠前优先级越高)。为了在 `encode` 时能快速查询任意一对字节 `(b1, b2)` 的合并优先级 (即它在 `merges` 列表中的位置)，我们预先计算并存储一个字典，键是字节对 `(b1, b2)`，值是它的优先级 (整数索引)。这对应您笔记中的**（优化1）**。
4.  **处理特殊 Token**：
      * 确保 `special_tokens` 列表存在 (如果传入 `None` 则变为空列表)。
      * 将字符串形式的特殊 token 转换为 `bytes` 形式存储起来 (`special_token_bytes`)。
      * 检查每个特殊 token (bytes形式) 是否已经存在于 `byte_to_token_id` (即是否在初始 `vocab` 中)。
      * 如果某个特殊 token **不在** 初始 `vocab` 中，则需要将其**添加**到 `self.vocab` 和 `self.byte_to_token_id` 中，确保它有一个唯一的 ID。

**注意**：与我之前错误的解释不同，这段优秀代码**并不会**在 `__init__` 中预编译正则表达式 (`PAT` 或用于分割特殊 token 的模式)。这些正则表达式模式会在 `encode` 或其辅助方法中直接使用。

## **代码实现**

```python
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """
        根据 "优秀代码" (adapters.py) 实现的构造函数
        """
        # 1. 存储基础信息
        self.vocab = vocab
        self.merges = merges # merges 列表本身也需要存储，虽然主要查询靠 bpe_ranks

        # 2. 创建反向词汇表 (bytes -> id)，用于 encode
        self.byte_to_token_id = {v: k for k, v in vocab.items()}

        # 3. 创建合并优先级查找表 (pair -> rank)，用于 encode
        #    zip(merges, range(len(merges))) 会产生 ((b1, b2), 0), ((b3, b4), 1), ...
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
            
        # 4. 处理特殊 token
        self.special_tokens = special_tokens or [] # 确保列表存在
        # 将特殊 token 字符串转换为 bytes
        self.special_token_bytes = [token.encode("utf-8") for token in self.special_tokens]
        
        # 确保特殊 token 都有对应的 ID
        for token_bytes in self.special_token_bytes:
            if token_bytes not in self.byte_to_token_id:
                # 如果这个特殊 token 不在初始 vocab 中
                # 分配一个新的 ID (当前 vocab 的大小)
                new_id = len(self.vocab)
                # 添加到 vocab (id -> bytes)
                self.vocab[new_id] = token_bytes
                # 添加到反向词汇表 (bytes -> id)
                self.byte_to_token_id[token_bytes] = new_id

    # ... (其他方法的 pass 不变) ...
```



