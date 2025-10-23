from collections import defaultdict, Counter
from dataclasses import dataclass
import regex as re  # 明确使用 regex 库并别名为 re
from typing import Dict, List, Tuple, Optional, Iterable, Any
import os

# --- 数据类定义 ---
@dataclass(frozen=True)
class BPETokenizerParams:
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]

# --- 辅助函数：完全仿照参考代码的实现 ---
# (这部分与您的代码相同，是正确的)
def _get_initial_word_freqs(input_path: str, special_tokens: list[str]) -> Dict[Tuple[bytes, ...], int]:
    """
    步骤1：预分词。读取文件，切分单词，并统计每个单词（表示为字节元组）的出现频率。
    """
    # 这个 PAT 字符串由于使用了 \p 语法，必须由 regex 库处理
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    word_freqs = defaultdict(int)
    
    # 仿照参考代码，使用非捕获组分割特殊字符
    special_pattern = "|".join(map(re.escape, special_tokens))
    if special_pattern:
        chunks = re.split(special_pattern, text)
    else:
        chunks = [text]
    
    for chunk in chunks:
        for match in re.finditer(PAT, chunk):
            word_bytes = match.group(0).encode('utf-8')
            # 将单词转换为字节元组，例如 "cat" -> (b'c', b'a', b't')
            word_tuple = tuple(bytes([b]) for b in word_bytes)
            word_freqs[word_tuple] += 1
            
    return word_freqs

def _get_pair_freqs(word_freqs: Dict[Tuple[bytes, ...], int]) -> Dict[Tuple[bytes, bytes], int]:
    """
    步骤2：统计所有相邻字节对的频率。
    """
    pair_freqs = defaultdict(int)
    for word_tuple, freq in word_freqs.items():
        for i in range(len(word_tuple) - 1):
            pair = (word_tuple[i], word_tuple[i+1])
            pair_freqs[pair] += freq
    return pair_freqs

# --- BPE 训练主函数 ---
# (这部分与您的代码相同，是正确的)
def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """根据输入语料训练一个BPE分词器（仿照参考代码实现）。"""
    
    # 1. 初始化词汇表
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for token_str in special_tokens:
        token_bytes = token_str.encode('utf-8')
        if token_bytes not in vocab.values():
            vocab[len(vocab)] = token_bytes

    # 2. 获取初始的单词频率
    word_freqs = _get_initial_word_freqs(input_path, special_tokens)

    # 3. 主合并循环
    num_merges = vocab_size - len(vocab)
    merges: List[Tuple[bytes, bytes]] = []

    for i in range(num_merges):
        # a. 统计当前所有相邻字节对的频率
        pair_freqs = _get_pair_freqs(word_freqs)
        
        if not pair_freqs:
            break

        # b. 找到最高频的字节对，并处理平局
        max_freq = max(pair_freqs.values())
        candidates = [pair for pair, freq in pair_freqs.items() if freq == max_freq]
        best_pair = max(candidates) # 字典序平局规则

        # c. 创建新的合并token，并更新 vocab 和 merges 列表
        p1, p2 = best_pair
        new_token_bytes = p1 + p2
        new_id = len(vocab)
        
        vocab[new_id] = new_token_bytes
        merges.append(best_pair)

        # d. 更新单词频率字典 (采用参考代码中的两阶段更新法，以保证效率和正确性)
        new_word_freqs = defaultdict(int)
        for word_tuple, freq in word_freqs.items():
            # 查找并替换当前单词中的 best_pair
            new_word_tuple = []
            j = 0
            while j < len(word_tuple):
                if j + 1 < len(word_tuple) and (word_tuple[j], word_tuple[j+1]) == best_pair:
                    new_word_tuple.append(new_token_bytes)
                    j += 2
                else:
                    new_word_tuple.append(word_tuple[j])
                    j += 1
            new_word_freqs[tuple(new_word_tuple)] += freq
        word_freqs = new_word_freqs

    return vocab, merges


# --- Tokenizer 类实现 (已根据“优秀代码” adapters.py 完善) ---

# 这是一个来自 adapters.py 的辅助函数
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
def to_bytes_tuple(word: str) -> Tuple[bytes]:
    l = list(word.encode("utf-8"))
    l = [bytes([x]) for x in l]
    return tuple(l)


class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """
        根据 "优秀代码" (adapters.py) 重新实现的构造函数
        """
        self.vocab = vocab
        # (修复问题1：AttributeError 需要的)
        # 预计算 byte -> id 映射
        self.byte_to_token_id = {v: k for k, v in vocab.items()}
        self.merges = merges

        # 预计算合并的优先级 (rank)
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
            
        # 处理特殊 token
        self.special_tokens = special_tokens or []
        self.special_token_bytes = [token.encode("utf-8") for token in self.special_tokens]
        
        # 确保特殊 token 在词汇表中
        for token_bytes in self.special_token_bytes:
            if token_bytes not in self.byte_to_token_id:
                # 如果不在，就添加进去
                new_id = len(self.vocab)
                self.vocab[new_id] = token_bytes
                self.byte_to_token_id[token_bytes] = new_id

    def encode(self, text: str) -> list[int]:
        """
        根据 "优秀代码" (adapters.py) 重新实现的 encode
        """
        tokens = []

        # (修复问题2：AssertionError 的关键)
        # 按长度倒序排序特殊 token，确保优先匹配最长的
        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
        
        pattern = "|".join(map(re.escape, sorted_special_tokens))
        if pattern:
            # 使用带捕获组的 split，将特殊 token 也作为块返回
            parts = re.split(f"({pattern})", text)
        else:
            parts = [text]

        for part in parts:
            if not part:
                continue
                
            if part in self.special_tokens:
                # 如果是特殊 token，直接添加 ID
                tokens.append(self.byte_to_token_id[part.encode("utf-8")])
            else:
                # 否则，走 BPE 流程
                tokens.extend(self._tokenize_normal(part))

        return tokens

    # (修复问题1：AttributeError 新增的方法)
    def encode_iterable(self, iterable: Iterable[str]) -> iter:
        """
        讲义要求的流式编码方法。
        """
        for chunk in iterable:
            yield from self.encode(chunk)


    def decode(self, ids: list[int]) -> str:
        """
        解码 (您的实现是正确的，这里保持一致)
        """
        # 查找所有字节
        all_bytes = [self.vocab.get(token_id) for token_id in ids]
        
        # 过滤掉 None (无效ID) 并连接
        full_bytes = b"".join(b for b in all_bytes if b is not None)
        
        # 解码为字符串，替换无效的 UTF-8 序列
        return full_bytes.decode("utf-8", errors="replace")

    def _tokenize_normal(self, text: str) -> list[int]:
        """
        辅助函数：对普通文本块（非特殊token）进行BPE
        (来自 adapters.py)
        """
        # 1. 预分词
        pre_tokens = []
        for m in re.finditer(PAT, text):
            word = m.group(0)
            pre_tokens.append(word)

        token_ids = []
        for token in pre_tokens:
            # 2. 转换为字节元组
            byte_tuple = to_bytes_tuple(token)
            
            # 3. 应用BPE合并
            merged = self._apply_merges(byte_tuple)
            
            # 4. 转换回 token ID
            token_ids.extend(self.byte_to_token_id[b] for b in merged)
        
        return token_ids

    def _apply_merges(self, byte_tuple: tuple[bytes, ...]) -> list[bytes]:
        """
        辅助函数：对一个字节元组应用所有 BPE 合并规则
        (来自 adapters.py)
        """
        word: list[bytes] = list(byte_tuple)

        # 辅助函数：获取所有相邻对
        def get_pairs(word: list[bytes]):
            pairs = set()
            prev_char = word[0]
            for char in word[1:]:
                pairs.add((prev_char, char))
                prev_char = char
            return pairs
        
        pairs = get_pairs(word)

        if not pairs:
            return word

        while True:
            # 找到优先级最高 (rank 最小) 的合并
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            
            if bigram not in self.bpe_ranks:
                break # 没有更多可用的合并了
            
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    # 找到 first 的下一个位置
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                # 检查是否是我们要合并的 bigram
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word) # 重新计算下一轮的 pairs

        return word