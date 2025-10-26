from collections import defaultdict, Counter
import regex as re

class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        spectial_tokens: list[str]
    ):
        self.vocab = vocab
        self.merges = merges
        self.spectialTokens = spectial_tokens or []

        self.bytesToIntVocab = {v : k for k, v in vocab.items()}

        self.bpeRanks = dict(zip(merges, range(len(merges))))

        self.spectialTokenBytes = [token.encode("utf-8") for token in self.spectialTokens]

        for tokenBytes in self.spectialTokenBytes:
            if tokenBytes not in self.bytesToIntVocab:
                newID = len(self.vocab)
                self.vocab[newID] = tokenBytes
                self.bytesToIntVocab[tokenBytes] = newID



    def encode(self, text: str) -> list[int]:
        """
        将一个原始字符串编码为一个 token ID 列表。
        """
        return []
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        """
        将一个字符串迭代器（如文件）编码为 token ID 迭代器。
        """
        pass


    def decode(self, ids: list[int]) -> str:
        """
        将一个 token ID 列表解码回一个字符串。
        """
        allBytes = [self.vocab.get(id) for id in ids]
        clearNoneBytes = b"".join(b for b in allBytes if b is not None)
        return clearNoneBytes.decode("utf-8", errors="replace")
    

    def _mergeAccordingRank(self, tupleBytes) -> list[bytes]:
        listBytes = list(tupleBytes)
        adjPairs = set(zip(listBytes, listBytes[1:]))

        if not adjPairs:
            return listBytes
        
        while True:
            # 首先需要根据找到优先级最高的
            targetMerge = min(adjPairs, key=lambda pair: self.bpeRanks.get(pair, float('inf')))

            if targetMerge not in self.bpeRanks:
                break

            # 然后合并，合并分两步，先将不需要合并的用extend加入，再将需要合并的用+和append加入
            first, second = targetMerge
            i = 0
            newListBytes = []
            while i < len(listBytes):
                try:
                    j = listBytes.index(first, i)
                except ValueError:
                    newListBytes.extend(listBytes[i: ])
                    break
                else:
                    newListBytes.extend(listBytes[i: j])
                    i = j

                if listBytes[i] == first and i + 1 < len(listBytes) and listBytes[i + 1] == second:
                    newListBytes.append(first + second)
                    i += 2
                else:
                    newListBytes.append(listBytes[i])
                    i += 1
            
            listBytes = newListBytes
            if len(listBytes) == 1:
                break
            else:
                adjPairs = set(zip(listBytes, listBytes[1:]))

        return listBytes

            

