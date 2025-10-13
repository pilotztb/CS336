# BPE分词 
# 将一个字符串，先转换成bytes列表，
# 然后统计列表中相邻的的两个bytes出现次数，
# 然后用一个循环，每次取出其中出现次数最多的bytes对，合成一个
from collections import defaultdict
def mergeFun(initialList, targetPair, targetVal):
    newList = []
    i = 0
    while i < len(initialList):
        if(i + 1 < len(initialList) and initialList[i] == targetPair[0] and initialList[i + 1] == targetPair[1]):
            newList.append(targetVal)
            i += 2
        else:
            newList.append(initialList[i])
            i += 1
    return newList

def bpeFunc(string, numMerges): # 第一个参数表示待转化字符串，第二个表示希望合并次数
    funList = list(map(int, string.encode('utf-8')))    # 将字符串转位bytes列表
    # merges: dict[tuple[int, int], int] = {}     # 这个字典目前不知道干啥用
    toMerges = {}
    vocab = {x : bytes([x]) for x in range(256)}

    for i in range(numMerges):
        counts = defaultdict(int)
        for index1, index2 in zip(funList, funList[1:]):
            counts[(index1, index2)] += 1

        pair = max(counts, key = counts.get)

        newIndex = 256 + i
        toMerges[pair] = newIndex
        vocab[newIndex] = vocab[pair[0]] + vocab[pair[1]]
        funList = merges(funList, pair, newIndex)



    