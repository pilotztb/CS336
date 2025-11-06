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
    
    with open(input_path, "r", encoding = "utf-8") as f:
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


def _getAdjPairFreq(
    wordFreq:dict[tuple[bytes, ...], int],
)->dict[tuple[bytes, bytes], int]:
    adjPairFreq = defaultdict(int)
    for wordTuple, freq in wordFreq.items():
        for i in range(len(wordTuple) - 1):
            pair = (wordTuple[i], wordTuple[i + 1])
            adjPairFreq[pair] += freq
    return adjPairFreq
    

def _mergeMaxAdjPairFreqWord(
    wordFreq : dict[tuple[bytes, ...], int],
    mergePair : tuple[bytes, bytes]
):
    newWordFreq = defaultdict(int)
    mergeToken = mergePair[0] + mergePair[1]
    for wordTuple, freq in wordFreq.items():
        newWordList = []
        i = 0
        n = len(wordTuple)
        while(i < n):
            if(i + 1 < n and wordTuple[i] == mergePair[0] and wordTuple[i + 1] == mergePair[1]):
                newWordList.append(mergeToken)
                i += 2
            else:
                newWordList.append(wordTuple[i])
                i += 1
        newWordFreq[tuple(newWordList)] += freq
    return newWordFreq

def train_bpe(
    inputPath : str,
    vocabSize : int,
    spectialToken : list[str]
) :
    vocab = {x : bytes([x]) for x in range(256)}
    
    for tokenStr in spectialToken:
        tokenBytes = tokenStr.encode("utf-8")
        if tokenBytes not in vocab.values():
            vocab[len(vocab)] = tokenBytes
    

    wordBytesFreq = _getInitialWordFreq(inputPath, spectialToken)
    
    numMerges = vocabSize - len(vocab)
    merges = []
    for i in range(numMerges):
        allAdjPairFreq = _getAdjPairFreq(wordBytesFreq)

        if not allAdjPairFreq:
            break

        max_freq = max(allAdjPairFreq.values())
        candidates = [pair for pair, freq in allAdjPairFreq.items() if freq == max_freq]
        best_pair = max(candidates) # 字典序平局规则

        wordBytesFreq = _mergeMaxAdjPairFreqWord(wordBytesFreq, best_pair)

        merges.append(best_pair)
        newToken = best_pair[0] + best_pair[1]
        vocab[len(vocab)] = newToken

    return vocab, merges

       