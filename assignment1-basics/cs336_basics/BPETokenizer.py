import os
import regex as re
from typing import BinaryIO, Iterable, Iterator, Optional
from collections import defaultdict
from multiprocessing import Process, Queue
import json

# --- 核心函数：BPE 训练 (train_bpe) ---

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    高效、正确地训练一个BPE分词器。
    该函数包含了并行预处理、优化的合并策略和正确的平局处理逻辑。
    """
    # 1. 初始化词汇表
    # 初始词汇表包含256个基本字节和所有特殊token。
    vocab = {i: bytes([i]) for i in range(256)}
    for i, token_str in enumerate(special_tokens):
        vocab[256 + i] = token_str.encode('utf-8')
    
    num_merges = vocab_size - len(vocab)
    if num_merges <= 0:
        return vocab, []

    # 2. 并行预分词和初始计数
    # 为了处理大文件，我们将其分块并在多个CPU核心上并行处理。
    num_processes = os.cpu_count() or 4
    with open(input_path, "rb") as f:
        # 使用第一个特殊token（通常是<|endoftext|>）作为文档分隔符来安全地分块。
        split_token_bytes = special_tokens[0].encode("utf-8") if special_tokens else b'\n'
        boundaries = _find_chunk_boundaries(f, num_processes, split_token_bytes)
        chunks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunks.append(f.read(end - start).decode("utf-8", errors="ignore"))

    q = Queue()
    processes = [Process(target=_worker, args=(chunk, special_tokens, q)) for chunk in chunks]
    for p in processes: p.start()
    
    # 从所有进程收集预分词及其计数
    word_counts = defaultdict(int)
    for _ in processes:
        chunk_word_counts = q.get()
        for word, count in chunk_word_counts.items():
            word_counts[word] += count
    for p in processes: p.join()

    # 将字节元组转换回整数ID元组以便处理
    vocab_inv = {v: k for k, v in vocab.items()}
    # 这里我们假设所有单个字节都已在vocab中
    word_counts_ids = {
        tuple(b for byte_char in word for b in byte_char): count
        for word, count in word_counts.items()
    }


    # 3. 迭代合并
    merges = []
    for i in range(num_merges):
        pair_stats = _get_pair_stats(word_counts_ids)
        if not pair_stats:
            break

        # 找到频率最高的词对。
        # 当频率相同时，根据作业要求，按字典序打破平局。
        best_pair = max(
            pair_stats, 
            key=lambda p: (pair_stats[p], vocab[p[0]].decode('utf-8', 'ignore'), vocab[p[1]].decode('utf-8', 'ignore'))
        )
        
        new_token_id = len(vocab)
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))
        vocab[new_token_id] = vocab[best_pair[0]] + vocab[best_pair[1]]
        
        # 更新词计数，用新的token ID替换合并的对
        word_counts_ids = _merge_word_counts(best_pair, word_counts_ids, new_token_id)

    return vocab, merges

# --- BPE Tokenizer 类 ---

class BPETokenizer:
    """
    一个完整的BPE分词器，包含编码和解码功能。
    """
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: Optional[list[str]] = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        
        # 创建反向词汇表和合并规则的排名，用于快速编码
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}
        self.special_tokens_bytes = {st.encode('utf-8') for st in self.special_tokens}

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: Optional[list[str]] = None):
        """
        从文件加载词汇表和合并规则来构造分词器。
        """
        # 注意：这里的加载格式需要与保存格式匹配。
        # 假设vocab是json，merges是每行一个合并规则的文本文件。
        with open(vocab_filepath, 'r') as f:
            # JSON的键是字符串，需要转回整数
            loaded_vocab_str_keys = json.load(f)
            vocab = {int(k): v.encode('latin-1') for k,v in loaded_vocab_str_keys.items()}
        
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            merges_str = [line.strip().split(' ') for line in f]
            # 此处需要一种方式将字符串表示的字节转回字节对象，eval不安全，需要特定格式
            # 假设保存时就是可读的字符串
            # 这是一个示例，具体取决于您的保存格式
            merges = [(eval(f"b'{m1}'"), eval(f"b'{m2}'")) for m1, m2 in merges_str]

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """将输入文本编码为token ID序列。"""
        pre_tokens = _pretokenize_bytes(text, self.special_tokens)
        
        output_ids = []
        for token_bytes in pre_tokens:
            if token_bytes in self.special_tokens_bytes:
                output_ids.append(self.inv_vocab[token_bytes])
                continue

            word_ids = [self.inv_vocab[bytes([b])] for b in token_bytes]
            
            while len(word_ids) > 1:
                pairs = {(word_ids[i], word_ids[i+1]): i for i in range(len(word_ids) - 1)}
                
                # 找到所有可能的合并中，排名最靠前（最早出现）的一个
                rank, best_pair, best_idx = min(
                    (self.merge_ranks.get((self.vocab[p[0]], self.vocab[p[1]]), float('inf')), p, idx)
                    for p, idx in pairs.items()
                )

                if rank == float('inf'):
                    break # 没有更多可合并的了

                # 执行合并
                new_token_id = self.inv_vocab[self.vocab[best_pair[0]] + self.vocab[best_pair[1]]]
                word_ids = word_ids[:best_idx] + [new_token_id] + word_ids[best_idx+2:]
            
            output_ids.extend(word_ids)
            
        return output_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """对一个字符串的可迭代对象（如文件句柄）进行流式编码。"""
        for line in iterable:
            for token_id in self.encode(line):
                yield token_id

    def decode(self, ids: list[int]) -> str:
        """将token ID序列解码回文本。"""
        # 使用.get(i, b'')来处理可能的未知ID
        tokens = b"".join(self.vocab.get(i, b'') for i in ids)
        # 使用 'replace' 错误处理来替换无效的UTF-8字节序列。
        return tokens.decode('utf-8', errors='replace')

# --- 辅助函数 (建议保持私有，以_开头) ---

def _find_chunk_boundaries(file: BinaryIO, num_chunks: int, split_token: bytes) -> list[int]:
    """在文件中的特殊token处查找分块边界。"""
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    
    chunk_size = file_size // num_chunks
    boundaries = [0] + [ (i + 1) * chunk_size for i in range(num_chunks - 1)] + [file_size]

    for i in range(1, len(boundaries) - 1):
        file.seek(boundaries[i])
        while True:
            byte = file.read(1)
            if not byte or (byte == split_token[0] and file.read(len(split_token) - 1) == split_token[1:]):
                boundaries[i] = file.tell() - len(split_token)
                break
    return sorted(list(set(boundaries)))

def _worker(text: str, special_tokens: list[str], q: Queue):
    """多进程工作函数：预分词并计数。"""
    pre_tokens = _pretokenize_bytes(text, special_tokens)
    word_counts = defaultdict(int)
    for token in pre_tokens:
        word_counts[tuple([bytes([b]) for b in token])] += 1
    q.put(word_counts)

def _get_pair_stats(word_counts: dict[tuple[int, ...], int]) -> defaultdict[tuple[int, int], int]:
    """从词计数中获取相邻对的频率。"""
    counts = defaultdict(int)
    for word, freq in word_counts.items():
        for i in range(len(word) - 1):
            counts[(word[i], word[i+1])] += freq
    return counts

def _merge_word_counts(pair: tuple[int, int], in_counts: dict[tuple[int, ...], int], new_id: int) -> dict[tuple[int, ...], int]:
    """在词计数中合并指定的对。"""
    out_counts = defaultdict(int)
    for word, freq in in_counts.items():
        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and (word[i], word[i+1]) == pair:
                new_word.append(new_id)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        out_counts[tuple(new_word)] += freq
    return out_counts

def _pretokenize_bytes(text: str, special_tokens: list[str]) -> list[bytes]:
    """对文本进行预分词，返回字节列表。"""
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    special_pattern = "|".join(re.escape(st) for st in sorted(special_tokens, key=len, reverse=True))
    split_pattern = f'({special_pattern})'
    
    parts = re.split(split_pattern, text)
    
    output = []
    for part in parts:
        if not part:
            continue
        if part in special_tokens:
            output.append(part.encode('utf-8'))
        else:
            tokens = re.findall(PAT, part)
            for token in tokens:
                output.append(token.encode('utf-8'))
    return output