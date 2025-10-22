from collections import defaultdict
from dataclasses import dataclass
import regex

@dataclass
class BPETokenizer:
    vocab : dict[int, bytes]
    merges : list[tuple[bytes, bytes]]

#统计每个单词出现的频率
def _getInitialWordFreq(
    input_path:str,
    spectial_token:list[str]
)->dict[tuple[bytes, ...], int]:
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    with open(input_path, "r", "utf-8") as f:
        text = f.read()
    # 为了使用spectial_token划分原来text，首先需要将列表转化为正则表达式
    spectialPattern = "|".join(map(regex.escape, spectial_token))

    # 利用spectial_token定义的正则表达式划分原来的text
    if(spectialPattern):
        initialListText = regex.split(spectialPattern, text)
    else:
        initialListText = [text]


    # 遍历整个list，对于每个划分出的段落，使用PAT定义的进一步划分，然后将划分出的单词str字符串转为tuple（bytes），
    wordFreq = defaultdict(int)
    for paragraph in initialListText:
        for match in regex.finditer(PAT, paragraph):
            word = match.group(0).encode("utf-8")
            tupleKey = tuple(bytes([x]) for x in word)
            wordFreq[tupleKey] += 1
    return wordFreq
    