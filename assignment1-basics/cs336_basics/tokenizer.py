from collections import defaultdict
from dataclasses import dataclass
import regex  # 参考代码使用的是 regex 库 (aliased as re)
from typing import Dict, List, Tuple

# --- 数据类定义 ---
@dataclass(frozen=True)
class BPETokenizerParams:
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]

# --- 辅助函数：完全仿照参考代码的实现 ---

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
    special_pattern = "|".join(map(regex.escape, special_tokens))
    if special_pattern:
        chunks = regex.split(special_pattern, text)
    else:
        chunks = [text]
    
    for chunk in chunks:
        for match in regex.finditer(PAT, chunk):
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

# --- Tokenizer 类实现 (为下一步做准备) ---
# ... (Tokenizer类的代码可以暂时省略，或者使用我们之前完善的版本) ...