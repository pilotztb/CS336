from collections import defaultdict
from dataclasses import dataclass

# 这是 lecture_01.py 中的 BPETokenizerParams，为了代码能跑通，我们把它也复制过来
@dataclass(frozen=True)
class BPETokenizerParams:
    vocab: dict[int, bytes]
    merges: dict[tuple[int, int], int]

# 这是 lecture_01.py 中的 merge 函数
def merge(indices: list[int], pair: tuple[int, int], new_index: int) -> list[int]:
    new_indices = []
    i = 0
    while i < len(indices):
        if i + 1 < len(indices) and indices[i] == pair[0] and indices[i + 1] == pair[1]:
            new_indices.append(new_index)
            i += 2
        else:
            new_indices.append(indices[i])
            i += 1
    return new_indices

# 这是 lecture_01.py 中的 train_bpe 函数
def train_bpe(string: str, num_merges: int) -> BPETokenizerParams:
    indices = list(map(int, string.encode("utf-8")))
    merges: dict[tuple[int, int], int] = {}
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}

    for i in range(num_merges): 
        counts = defaultdict(int)
        for index1, index2 in zip(indices, indices[1:]):
            counts[(index1, index2)] += 1

        pair = max(counts, key=counts.get)

        new_index = 256 + i
        merges[pair] = new_index
        vocab[new_index] = vocab[pair[0]] + vocab[pair[1]]
        indices = merge(indices, pair, new_index)

    return BPETokenizerParams(vocab=vocab, merges=merges)

# 我们在这里调用函数并打印结果
if __name__ == "__main__":
    test_string = "ababa"
    print(f"原始字符串: {test_string}")
    print("-" * 30)

    # 我们只进行2次合并，方便观察
    params = train_bpe(test_string, num_merges=2)

    print("-" * 30)
    print("最终结果:")
    print("词汇表 (Vocab):")
    # 为了方便查看，我们打印解码后的词汇
    decoded_vocab = {k: v.decode('utf-8', errors='ignore') for k, v in params.vocab.items() if k >= 256}
    print(decoded_vocab)

    print("\n合并规则 (Merges):")
    print(params.merges)