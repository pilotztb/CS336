from collections import defaultdict
from dataclasses import dataclass
import regex

@dataclass(frozen=True)
class BPETokenizerParams:
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]

# 预处理大文件
def preProcessLargeFile(input_path, spectial_tokens):
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    if(spectial_tokens):
        spectialRegStr = "|".join(map(regex.escape, spectial_tokens))
    else:
        spectialRegStr = ""

    ansDict = defaultdict(int)
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                if(spectialRegStr):
                    firstSplitTextList = regex.split(f"({spectialRegStr})", line)
                else:
                    firstSplitTextList = [line]
                
                for seg in firstSplitTextList:
                    if seg in spectial_tokens:
                        continue
                    
                    tmpList = regex.findall(PAT, seg)
                    
                    for ele in tmpList:
                        ansDict[ele] += 1

    except FileNotFoundError:
        print(f"not find file in {input_path}")
    except Exception as e:
        print(f"error:{e}")

    return ansDict

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> BPETokenizerParams:
    # 初始化字典，包括当前0-255的utf-8和spectial
    vocab = {x : bytes([x]) for x in range(256)} 
    for idx, ele in enumerate(special_tokens):
        vocab[256 + idx] = ele.encode('utf-8')

    targetFileTodict = preProcessLargeFile(input_path, special_tokens)
    num_merges = vocab_size - len(vocab)
    if num_merges < 0:
        num_merges = 0

    for i in range(num_merges):
        pass
