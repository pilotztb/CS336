# 测试train_bpe

1. **连接测试适配器**: 打开 `tests/adapters.py` 文件。找到 `run_train_bpe` 函数，并修改它，让它调用您在 `cs336_basics/tokenizer.py` 中实现的 `train_bpe` 函数。
    ```python
    # 在 tests/adapters.py 中
    from cs336_basics.tokenizer import train_bpe # 导入您的函数
    
    def run_train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
        # 调用您自己的实现
        bpe_params = train_bpe(input_path, vocab_size, special_tokens)
        # 返回符合测试要求的结果
        return bpe_params.vocab, bpe_params.merges
    ```
2. **运行 `train_bpe` 测试**: 现在，您可以运行针对训练函数的测试了。在您的终端中执行以下命令：
    ```bash
    uv run pytest tests/test_train_bpe.py
    ```
    根据测试的反馈进行调试，直到所有与`train_bpe`相关的测试都通过。

## 关于环境

`uv run` 这个命令会自动处理虚拟环境。具体来说，当您在项目根目录下（即包含 `pyproject.toml` 文件的目录）运行 `uv run pytest tests/test_train_bpe.py` 时，`uv` 会执行以下操作：

1.  **检查环境**：它会检查是否存在一个满足 `pyproject.toml` 文件中定义的依赖项的虚拟环境。
2.  **创建/同步环境**：如果这个环境不存在，`uv` 会**自动创建一个**。如果环境已存在但依赖不完整，`uv` 会自动安装缺失的包（例如 `pytest`, `numpy`, `regex` 等）。
3.  **在环境中执行命令**：`uv` 会在该虚拟环境中**执行**您指定的命令，也就是 `pytest tests/test_train_bpe.py`。

**简而言之：** 您不需要手动使用 `uv venv` 创建或 `source .venv/bin/activate` 激活任何环境。`uv run` 命令一步到位，为您处理好了所有环境相关的准备工作并执行测试。

#### **第二步：实现 `Tokenizer` 类**

[cite\_start]根据PDF中 **Problem (tokenizer)** 的要求 [cite: 980]，您需要在 `cs336_basics/tokenizer.py` 文件中继续添加 `Tokenizer` 类的实现。

这个类负责使用训练好的规则进行编码和解码。下面是一个您可以直接使用的框架，您需要填充其中的逻辑：

```python
# 在 cs336_basics/tokenizer.py 文件中继续添加

class Tokenizer:
    def __init__(self, params: BPETokenizerParams):
        """
        使用训练好的参数来初始化分词器。
        """
        self.vocab = params.vocab
        self.merges = params.merges
        
        # 为了加速查找，将merges列表转换为一个字典。
        # 键是(bytes, bytes)对，值是它们的合并顺序（或优先级）。
        self.merge_ranks = {pair: i for i, pair in enumerate(self.merges)}
        
        # 同样，也需要一个反向的vocab用于解码
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # 预分词用的正则表达式 (从PDF中复制)
        self.pat = regex.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def encode(self, text: str) -> list[int]:
        """
        将输入的字符串编码为token ID列表。
        """
        # [cite_start]1. 使用PAT正则表达式进行预分词 [cite: 957]。
        # 2. 遍历每个预分词的词块(chunk)。
        # 3. 将词块转换为字节序列，再转为初始的ID序列。
        # 4. 在ID序列上，按照self.merge_ranks的优先级，循环应用merges规则，
        #    [cite_start]直到没有可合并的项 [cite: 959]。
        # 5. 将所有处理后的ID序列拼接起来，返回最终结果。
        
        # ... 在这里填充您的编码逻辑 ...
        pass

    def decode(self, ids: list[int]) -> str:
        """
        将token ID列表解码回字符串。
        """
        # [cite_start]1. 遍历ID列表，使用self.vocab查找每个ID对应的字节(bytes) [cite: 977]。
        # 2. 将所有字节拼接成一个大的bytes对象。
        # 3. 使用 .decode("utf-8", errors="replace") 将其解码为字符串，
        #    [cite_start]以处理可能出现的无效UTF-8序列 [cite: 979]。
        
        # ... 在这里填充您的解码逻辑 ...
        pass
```

#### **第三步：整合并测试 `Tokenizer` 类**

1.  **连接测试适配器**: 再次打开 `tests/adapters.py`，找到 `get_tokenizer` 函数，让它导入并实例化您刚刚创建的 `Tokenizer` 类。
2.  **运行 `Tokenizer` 测试**: 在终端中执行：
    ```bash
    uv run pytest tests/test_tokenizer.py
    ```
    同样，根据测试反馈进行调试，直到所有测试通过。

完成以上所有步骤后，您就真正地完成了“学习日 3/10”的全部任务，并拥有了一个功能完整、通过官方验证的BPE分词器。